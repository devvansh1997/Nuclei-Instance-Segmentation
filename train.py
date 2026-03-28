"""
train.py

Train MobileSAM + LoRA on one fold of 5-fold cross-validation.

Training strategy (v2):
  - For each image, run the image encoder ONCE → cache the embedding.
  - Iterate over ALL nucleus instances in the image (capped by
    max_nuclei_per_image if set); build a centroid point prompt for each.
  - Accumulate per-nucleus losses; call backward once per image.
  - Mask decoder + prompt encoder are fully trainable (not LoRA-only).
  - LoRA (rank 16, alpha 32) applied only to the image encoder qkv layers.
  - Validation uses the same per-nucleus proxy pass to select best checkpoint.
  - Full AJI/PQ evaluation runs in evaluate.py.

Usage:
    python train.py --config configs/debug.yaml   --fold 0
    python train.py --config configs/train_a100.yaml --fold 0
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.dataset import get_fold_datasets
from models.sam_lora import build_model
from utils.logger import setup_logger, log_system_info, log_config
from utils.losses import BCEDiceLoss, iou_prediction_loss

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collate — keep numpy arrays as lists so SAM preprocessing runs per-image
# ---------------------------------------------------------------------------

def _collate_fn(batch: list) -> dict:
    """
    Custom collate: keeps all arrays as Python lists.
    SAM's ResizeLongestSide preprocessing must run per-image because
    it records original/input sizes needed for coordinate scaling.
    """
    keys = batch[0].keys()
    return {k: [s[k] for s in batch] for k in keys}


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

def _build_scheduler(
    optimizer:       torch.optim.Optimizer,
    cfg:             dict,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    training_cfg  = cfg["training"]
    total_epochs  = training_cfg["epochs"]
    warmup_epochs = training_cfg.get("warmup_epochs", 0)
    sched_type    = training_cfg.get("lr_scheduler", "cosine")

    total_steps  = total_epochs  * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if sched_type == "cosine":
        def _lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(warmup_steps)
            progress = float(step - warmup_steps) / float(
                max(total_steps - warmup_steps, 1)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    elif sched_type == "step":
        step_size = max(total_epochs // 3, 1)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=0.1
        )

    else:  # "none"
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


# ---------------------------------------------------------------------------
# Per-nucleus prompt helpers
# ---------------------------------------------------------------------------

def _get_nucleus_ids(
    instance_mask:  np.ndarray,
    max_nuclei:     int,
    rng:            np.random.Generator,
) -> List[int]:
    """
    Return a list of nucleus instance IDs to process for one image.

    Parameters
    ----------
    instance_mask : (H, W) uint16  — per-nucleus instance IDs (0 = background).
    max_nuclei    : Cap on number of nuclei per image. 0 = no cap (use all).
    rng           : Numpy random Generator for reproducible subsampling.

    Returns
    -------
    List of int instance IDs.  Empty list if the mask has no nuclei.
    """
    unique_ids  = np.unique(instance_mask)
    nucleus_ids = [int(i) for i in unique_ids if i > 0]

    if not nucleus_ids:
        return []

    if max_nuclei > 0 and len(nucleus_ids) > max_nuclei:
        nucleus_ids = rng.choice(
            nucleus_ids, size=max_nuclei, replace=False
        ).tolist()

    return nucleus_ids


def _build_point_prompt(
    instance_mask: np.ndarray,
    nucleus_id:    int,
    orig_size:     Tuple[int, int],
    model,
    bg_rows:       np.ndarray,
    bg_cols:       np.ndarray,
    rng:           np.random.Generator,
    device:        torch.device,
    num_pos:       int = 1,
    num_neg:       int = 1,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Build a centroid point prompt for one nucleus and its 256×256 GT mask.

    Parameters
    ----------
    instance_mask : (H, W) uint16 — full instance mask for the image.
    nucleus_id    : ID of the nucleus to build a prompt for.
    orig_size     : (H, W) of the original image (for coordinate scaling).
    model         : MobileSAMLoRA (for scale_coords).
    bg_rows/cols  : Pre-computed background pixel coordinates.
    rng           : Numpy random Generator.
    device        : Target torch device.
    num_pos/neg   : Number of positive/negative points.

    Returns
    -------
    (coords_t, labels_t, gt_256) on device, or None if the nucleus is empty.
        coords_t : (1, N, 2) float32
        labels_t : (1, N)   int64
        gt_256   : (1, 1, 256, 256) float32
    """
    inst_binary = (instance_mask == nucleus_id).astype(np.float32)
    rows, cols  = np.where(inst_binary)

    if rows.size == 0:
        return None

    # Centroid as positive point (SAM convention: x=col, y=row)
    centroid_x, centroid_y = cols.mean(), rows.mean()
    pos_coords = np.array(
        [[centroid_x, centroid_y]] * num_pos, dtype=np.float32
    )

    # Random background pixel(s) as negative points
    if bg_rows.size > 0:
        neg_idx    = rng.choice(len(bg_rows), size=num_neg,
                                replace=(len(bg_rows) < num_neg))
        neg_coords = np.stack(
            [bg_cols[neg_idx].astype(np.float32),
             bg_rows[neg_idx].astype(np.float32)],
            axis=1,
        )
    else:
        # Fallback: duplicate positive point (edge case — no background)
        neg_coords = pos_coords.copy()

    point_coords = np.concatenate([pos_coords, neg_coords], axis=0)  # (N, 2)
    point_labels = np.array(
        [1] * num_pos + [0] * num_neg, dtype=np.int64
    )

    scaled   = model.scale_coords(point_coords, orig_size)
    coords_t = torch.tensor(scaled[None],       dtype=torch.float32).to(device)
    labels_t = torch.tensor(point_labels[None], dtype=torch.int64).to(device)

    # GT resized to 256×256 (SAM decoder output resolution)
    gt_256 = F.interpolate(
        torch.tensor(inst_binary[None, None], dtype=torch.float32),
        size=(256, 256),
        mode="nearest",
    ).to(device)

    return coords_t, labels_t, gt_256


# ---------------------------------------------------------------------------
# Per-epoch helpers
# ---------------------------------------------------------------------------

def _train_one_epoch(
    model,
    dataloader:  DataLoader,
    optimizer:   torch.optim.Optimizer,
    scheduler:   torch.optim.lr_scheduler.LRScheduler,
    criterion:   BCEDiceLoss,
    scaler:      torch.amp.GradScaler,
    cfg:         dict,
    device:      torch.device,
    amp_dtype,
    epoch:       int,
) -> dict:
    """
    One training epoch — all nuclei per image, embedding cached per image.

    For each batch:
      1. For each image: encode once → reuse embedding for all nucleus prompts.
      2. Accumulate per-nucleus losses (stack + mean) → one backward per image.
      3. Single optimizer step after all images in the batch.

    Gradient normalization: each image contributes equally (mean over its
    nuclei), and images are averaged over the batch.
    """
    model.train()

    max_nuclei = cfg["training"].get("max_nuclei_per_image", 0)
    num_pos    = cfg["training"].get("num_pos_points", 1)
    num_neg    = cfg["training"].get("num_neg_points", 1)
    grad_clip  = float(cfg["training"].get("grad_clip", 1.0))
    use_amp    = amp_dtype is not None
    rng        = np.random.default_rng()   # Fresh RNG per epoch

    sum_loss = sum_bce = sum_dice = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        n_images        = len(batch["image"])
        batch_loss      = batch_bce = batch_dice = 0.0
        n_images_valid  = 0

        for image_np, inst_mask in zip(batch["image"], batch["instance_mask"]):

            nucleus_ids = _get_nucleus_ids(inst_mask, max_nuclei, rng)
            if not nucleus_ids:
                continue

            # -- Preprocess image (CPU) ------------------------------------------
            img_tensor, orig_size, _ = model.preprocess_image(image_np)
            img_tensor = img_tensor.to(device)
            bg_rows, bg_cols = np.where(inst_mask == 0)

            # -- Encode image ONCE (GPU) -- gradients must flow through LoRA ------
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                image_emb, image_pe = model.encode_image(img_tensor)

            # -- Decode each nucleus, accumulate losses ---------------------------
            nucleus_losses: List[torch.Tensor] = []
            batch_bce_accum = batch_dice_accum = 0.0

            for nid in nucleus_ids:
                prompt = _build_point_prompt(
                    inst_mask, nid, orig_size, model,
                    bg_rows, bg_cols, rng, device, num_pos, num_neg,
                )
                if prompt is None:
                    continue
                coords_t, labels_t, gt_256 = prompt

                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    low_res_masks, iou_preds = model.decode_masks(
                        image_emb, image_pe, coords_t, labels_t
                    )
                    seg_loss, bce, dice = criterion(low_res_masks, gt_256)
                    iou_loss            = iou_prediction_loss(
                        iou_preds, low_res_masks, gt_256
                    )

                nucleus_losses.append(seg_loss + 0.1 * iou_loss)
                batch_bce_accum  += bce.item()
                batch_dice_accum += dice.item()

            if not nucleus_losses:
                continue

            n_valid = len(nucleus_losses)

            # Mean over nuclei → normalize by n_images for batch average
            image_loss = torch.stack(nucleus_losses).mean() / n_images
            scaler.scale(image_loss).backward()  # Frees this image's graph

            batch_loss += image_loss.item() * n_images   # Undo /n_images for logging
            batch_bce  += batch_bce_accum / n_valid
            batch_dice += batch_dice_accum / n_valid
            n_images_valid += 1

        if n_images_valid == 0:
            logger.debug(
                "Epoch %d  Batch %d: no valid samples — skipping.",
                epoch, batch_idx,
            )
            continue

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        avg_loss = batch_loss / n_images_valid
        avg_bce  = batch_bce  / n_images_valid
        avg_dice = batch_dice / n_images_valid

        sum_loss  += avg_loss
        sum_bce   += avg_bce
        sum_dice  += avg_dice
        n_batches += 1

        logger.debug(
            "Epoch %3d  Batch %4d/%d  loss=%.4f  bce=%.4f  dice=%.4f  lr=%.2e",
            epoch, batch_idx + 1, len(dataloader),
            avg_loss, avg_bce, avg_dice,
            scheduler.get_last_lr()[0],
        )

    denom = max(n_batches, 1)
    return {
        "loss": sum_loss / denom,
        "bce":  sum_bce  / denom,
        "dice": sum_dice / denom,
    }


# ---------------------------------------------------------------------------

@torch.no_grad()
def _validate_one_epoch(
    model,
    dataloader: DataLoader,
    criterion:  BCEDiceLoss,
    cfg:        dict,
    device:     torch.device,
    amp_dtype,
    epoch:      int,
) -> dict:
    """
    Proxy validation — same per-nucleus approach as training.

    Caps at 8 nuclei per image for speed (enough for a reliable proxy metric).
    Fixed RNG seed → reproducible val_dice across epochs.
    """
    model.eval()

    VAL_NUCLEI_CAP = 8       # Small cap: just needs to be a stable proxy
    use_amp        = amp_dtype is not None
    rng            = np.random.default_rng(seed=0)

    sum_loss = sum_dice = 0.0
    n_batches = 0

    for batch in dataloader:
        batch_loss = batch_dice = 0.0
        n_images_valid = 0

        for image_np, inst_mask in zip(batch["image"], batch["instance_mask"]):
            nucleus_ids = _get_nucleus_ids(inst_mask, VAL_NUCLEI_CAP, rng)
            if not nucleus_ids:
                continue

            img_tensor, orig_size, _ = model.preprocess_image(image_np)
            img_tensor = img_tensor.to(device)
            bg_rows, bg_cols = np.where(inst_mask == 0)

            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=use_amp):
                image_emb, image_pe = model.encode_image(img_tensor)

            image_loss = image_dice = 0.0
            n_valid = 0

            for nid in nucleus_ids:
                prompt = _build_point_prompt(
                    inst_mask, nid, orig_size, model,
                    bg_rows, bg_cols, rng, device,
                )
                if prompt is None:
                    continue
                coords_t, labels_t, gt_256 = prompt

                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=use_amp):
                    low_res_masks, _ = model.decode_masks(
                        image_emb, image_pe, coords_t, labels_t
                    )
                    loss, _, dice = criterion(low_res_masks, gt_256)

                image_loss += loss.item()
                image_dice += dice.item()
                n_valid    += 1

            if n_valid > 0:
                batch_loss += image_loss / n_valid
                batch_dice += image_dice / n_valid
                n_images_valid += 1

        if n_images_valid > 0:
            sum_loss  += batch_loss / n_images_valid
            sum_dice  += batch_dice / n_images_valid
            n_batches += 1

    denom = max(n_batches, 1)
    return {
        "val_loss": sum_loss / denom,
        "val_dice": 1.0 - (sum_dice / denom),   # Convert dice LOSS → dice SCORE
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_dir(cfg: dict, fold_idx: int) -> Path:
    save_dir = Path(cfg["output"]["save_dir"])
    return save_dir / f"fold_{fold_idx}" / "checkpoints"


def _save_checkpoint(model, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_lora_weights(str(path))
    meta_path = path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.debug("Checkpoint saved → %s", path)


# ---------------------------------------------------------------------------
# Main training function (one fold)
# ---------------------------------------------------------------------------

def train_fold(cfg: dict, fold_idx: int) -> dict:
    """
    Train for one fold and return best validation metrics.

    Parameters
    ----------
    cfg      : Full config dict.
    fold_idx : Zero-based fold index.

    Returns
    -------
    dict with keys: fold, best_val_dice, best_epoch
    """
    run_name = f"fold_{fold_idx}"
    logger_  = setup_logger(cfg, run_name)
    log_system_info(logger_)
    log_config(logger_, cfg)

    # -----------------------------------------------------------------------
    # Device + mixed precision
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger_.info("Device: %s", device)

    mp_str = cfg["training"].get("mixed_precision", "none").lower()
    if mp_str == "fp16":
        amp_dtype = torch.float16
    elif mp_str == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = None

    # GradScaler only needed for fp16 (bf16 and fp32 don't need it)
    use_scaler = (device.type == "cuda") and (mp_str == "fp16")
    scaler     = torch.amp.GradScaler("cuda", enabled=use_scaler)
    logger_.info("Mixed precision: %s | GradScaler: %s", mp_str, use_scaler)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_dataset, val_dataset = get_fold_datasets(cfg, fold_idx)

    num_workers = cfg["data"].get("num_workers", 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = (device.type == "cuda"),
        collate_fn  = _collate_fn,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = (device.type == "cuda"),
        collate_fn  = _collate_fn,
    )

    logger_.info(
        "DataLoaders | train_batches=%d | val_batches=%d",
        len(train_loader), len(val_loader),
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = build_model(cfg, device)

    # -----------------------------------------------------------------------
    # Optimizer — now includes mask decoder + prompt encoder params
    # -----------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = float(cfg["training"]["lr"]),
        weight_decay = float(cfg["training"]["weight_decay"]),
    )
    logger_.info(
        "Optimizer: AdamW | lr=%.2e | wd=%.2e | trainable_params=%d",
        cfg["training"]["lr"], cfg["training"]["weight_decay"],
        sum(p.numel() for p in trainable_params),
    )

    # -----------------------------------------------------------------------
    # Scheduler
    # -----------------------------------------------------------------------
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    n_epochs   = cfg["training"]["epochs"]
    save_every = cfg["debug"].get("save_every_n_epochs", 5)
    ckpt_dir   = _checkpoint_dir(cfg, fold_idx)

    max_nuclei = cfg["training"].get("max_nuclei_per_image", 0)
    logger_.info(
        "Starting training | fold=%d | epochs=%d | max_nuclei_per_image=%s",
        fold_idx, n_epochs, max_nuclei if max_nuclei > 0 else "all",
    )

    best_val_dice = 0.0
    best_epoch    = 0
    history       = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_metrics = _train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, cfg, device, amp_dtype, epoch,
        )

        val_metrics = _validate_one_epoch(
            model, val_loader, criterion, cfg, device, amp_dtype, epoch,
        )

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        logger_.info(
            "Epoch %3d/%d | loss=%.4f  bce=%.4f  dice_loss=%.4f "
            "| val_loss=%.4f  val_dice=%.4f | lr=%.2e | %.1fs",
            epoch, n_epochs,
            train_metrics["loss"], train_metrics["bce"], train_metrics["dice"],
            val_metrics["val_loss"], val_metrics["val_dice"],
            lr_now, elapsed,
        )

        row = {"epoch": epoch, **train_metrics, **val_metrics, "lr": lr_now}
        history.append(row)

        if val_metrics["val_dice"] > best_val_dice:
            best_val_dice = val_metrics["val_dice"]
            best_epoch    = epoch
            _save_checkpoint(
                model,
                path = ckpt_dir / "best_lora.pt",
                meta = {"epoch": epoch, "val_dice": best_val_dice, "fold": fold_idx},
            )
            logger_.info(
                "  ↑ New best val_dice=%.4f at epoch %d — checkpoint saved.",
                best_val_dice, epoch,
            )

        if epoch % save_every == 0:
            _save_checkpoint(
                model,
                path = ckpt_dir / f"epoch_{epoch:03d}_lora.pt",
                meta = {"epoch": epoch, "val_dice": val_metrics["val_dice"],
                        "fold": fold_idx},
            )

    # -----------------------------------------------------------------------
    # Save training history
    # -----------------------------------------------------------------------
    history_path = (
        Path(cfg["output"]["save_dir"]) / f"fold_{fold_idx}" / "train_history.json"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger_.info("Training history saved → %s", history_path)

    result = {
        "fold":          fold_idx,
        "best_val_dice": best_val_dice,
        "best_epoch":    best_epoch,
    }
    logger_.info(
        "Fold %d training complete | best_val_dice=%.4f at epoch %d",
        fold_idx, best_val_dice, best_epoch,
    )
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train MobileSAM+LoRA on one NuInsSeg fold."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--fold",   required=True, type=int, help="Fold index (0-4).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    train_fold(cfg, args.fold)


if __name__ == "__main__":
    main()
