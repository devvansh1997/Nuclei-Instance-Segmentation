"""
train.py

Train MobileSAM + LoRA on one fold of 5-fold cross-validation.

Training strategy:
  - For each batch image, randomly sample one nucleus instance.
  - Use its centroid as a positive point prompt + one background pixel as negative.
  - Supervise the decoder output with BCE + Dice loss at 256×256 resolution.
  - Auxiliary IOU prediction loss to keep SAM's confidence head calibrated.
  - Validation uses the same point-prompted forward pass (proxy Dice) to
    select the best checkpoint.  Full AJI/PQ evaluation runs in evaluate.py.

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

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from data.dataset import get_fold_datasets
from models.sam_lora import build_model, sample_point_prompts
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
    optimizer: torch.optim.Optimizer,
    cfg: dict,
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
# Per-epoch helpers
# ---------------------------------------------------------------------------

def _process_batch(
    batch:    dict,
    model,
    cfg:      dict,
    device:   torch.device,
    rng:      np.random.Generator,
) -> tuple:
    """
    Prepare tensors for one batch.

    Returns
    -------
    image_batch       : (B, 3, 1024, 1024) float32 on device
    point_coords      : (B, N, 2) float32 on device
    point_labels      : (B, N) int64 on device
    gt_256            : (B, 1, 256, 256) float32 on device  (resized GT masks)
    n_valid           : int  number of samples successfully processed
    """
    num_pos = cfg["training"].get("num_pos_points", 1)
    num_neg = cfg["training"].get("num_neg_points", 1)

    image_tensors   = []
    coords_list     = []
    labels_list     = []
    gt_list         = []

    for image_np, inst_mask in zip(batch["image"], batch["instance_mask"]):
        try:
            pt_coords, pt_labels, inst_binary, _ = sample_point_prompts(
                inst_mask, num_pos=num_pos, num_neg=num_neg, rng=rng
            )
        except ValueError as exc:
            logger.debug("Point sampling failed — skipping sample: %s", exc)
            continue

        img_tensor, orig_size, _ = model.preprocess_image(image_np)
        scaled_coords = model.scale_coords(pt_coords, orig_size)

        image_tensors.append(img_tensor)                    # (1, 3, 1024, 1024)
        coords_list.append(scaled_coords)                   # (N, 2)
        labels_list.append(pt_labels)                       # (N,)
        gt_list.append(inst_binary.astype(np.float32))      # (H, W) float32

    if not image_tensors:
        return None, None, None, None, 0

    B = len(image_tensors)
    N = coords_list[0].shape[0]
    H, W = gt_list[0].shape

    image_batch = torch.cat(image_tensors, dim=0).to(device)     # (B, 3, 1024, 1024)

    point_coords = torch.tensor(
        np.stack(coords_list, axis=0), dtype=torch.float32
    ).to(device)                                                   # (B, N, 2)

    point_labels = torch.tensor(
        np.stack(labels_list, axis=0), dtype=torch.int64
    ).to(device)                                                   # (B, N)

    gt_tensor = torch.tensor(
        np.stack(gt_list, axis=0)
    ).unsqueeze(1).to(device)                                      # (B, 1, H, W)

    # Resize GT to 256×256 (SAM low-res decoder output space)
    # Use nearest-neighbour to preserve binary mask values exactly
    gt_256 = F.interpolate(gt_tensor, size=(256, 256), mode="nearest")  # (B, 1, 256, 256)

    return image_batch, point_coords, point_labels, gt_256, B


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
    model.train()

    num_pos  = cfg["training"].get("num_pos_points", 1)
    num_neg  = cfg["training"].get("num_neg_points", 1)
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    rng = np.random.default_rng()   # New RNG each epoch for diverse sampling

    sum_loss = sum_bce = sum_dice = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        image_batch, point_coords, point_labels, gt_256, n_valid = _process_batch(
            batch, model, cfg, device, rng
        )

        if n_valid == 0:
            logger.debug("Epoch %d  Batch %d: no valid samples — skipping.", epoch, batch_idx)
            continue

        # SAM's mask decoder is not designed for true multi-image batches —
        # process each image individually and accumulate gradients before
        # a single optimizer step (equivalent to an averaged batch loss).
        optimizer.zero_grad()
        use_amp = amp_dtype is not None

        batch_loss = batch_bce = batch_dice = 0.0

        for i in range(n_valid):
            img_i    = image_batch[i:i+1]      # (1, 3, 1024, 1024)
            coords_i = point_coords[i:i+1]     # (1, N, 2)
            labels_i = point_labels[i:i+1]     # (1, N)
            gt_i     = gt_256[i:i+1]           # (1, 1, 256, 256)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                low_res_masks, iou_preds = model(img_i, coords_i, labels_i)
                seg_loss, bce, dice      = criterion(low_res_masks, gt_i)
                iou_loss                 = iou_prediction_loss(iou_preds, low_res_masks, gt_i)
                # Normalise by n_valid so the effective loss = batch average
                step_loss                = (seg_loss + 0.1 * iou_loss) / n_valid

            scaler.scale(step_loss).backward()   # Accumulate gradients

            batch_loss += step_loss.item() * n_valid   # Undo normalisation for logging
            batch_bce  += bce.item()
            batch_dice += dice.item()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        avg_batch_loss = batch_loss / n_valid
        avg_batch_bce  = batch_bce  / n_valid
        avg_batch_dice = batch_dice / n_valid

        sum_loss  += avg_batch_loss
        sum_bce   += avg_batch_bce
        sum_dice  += avg_batch_dice
        n_batches += 1

        logger.debug(
            "Epoch %3d  Batch %4d/%d  loss=%.4f  bce=%.4f  dice=%.4f  lr=%.2e",
            epoch, batch_idx + 1, len(dataloader),
            avg_batch_loss, avg_batch_bce, avg_batch_dice,
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
    Proxy validation using point-prompted forward pass (same as training).
    Reports val_loss and val_dice — fast alternative to full AMG evaluation.
    """
    model.eval()
    rng = np.random.default_rng(seed=0)   # Fixed seed → reproducible val metric

    sum_loss = sum_dice = 0.0
    n_batches = 0

    for batch in dataloader:
        image_batch, point_coords, point_labels, gt_256, n_valid = _process_batch(
            batch, model, cfg, device, rng
        )
        if n_valid == 0:
            continue

        use_amp = amp_dtype is not None
        batch_loss = batch_dice = 0.0

        for i in range(n_valid):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                low_res_masks, _ = model(
                    image_batch[i:i+1], point_coords[i:i+1], point_labels[i:i+1]
                )
                loss, _, dice = criterion(low_res_masks, gt_256[i:i+1])
            batch_loss += loss.item()
            batch_dice += dice.item()

        sum_loss  += batch_loss / n_valid
        sum_dice  += batch_dice / n_valid
        n_batches += 1

    denom = max(n_batches, 1)
    return {
        "val_loss": sum_loss / denom,
        "val_dice": 1.0 - (sum_dice / denom),  # Convert dice LOSS → dice SCORE
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
        drop_last   = True,   # Avoid batch of 1 which can cause BN issues
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
    # Optimizer (only LoRA parameters)
    # -----------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = float(cfg["training"]["lr"]),
        weight_decay = float(cfg["training"]["weight_decay"]),
    )
    logger_.info(
        "Optimizer: AdamW | lr=%.2e | wd=%.2e | param_groups=%d params",
        cfg["training"]["lr"], cfg["training"]["weight_decay"],
        sum(p.numel() for p in trainable_params),
    )

    # -----------------------------------------------------------------------
    # Scheduler (steps per batch, not per epoch)
    # -----------------------------------------------------------------------
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    n_epochs           = cfg["training"]["epochs"]
    save_every         = cfg["debug"].get("save_every_n_epochs", 5)
    ckpt_dir           = _checkpoint_dir(cfg, fold_idx)

    best_val_dice = 0.0
    best_epoch    = 0
    history       = []

    logger_.info("Starting training | fold=%d | epochs=%d", fold_idx, n_epochs)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = _train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, scaler, cfg, device, amp_dtype, epoch,
        )

        # Validate
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

        # Save best checkpoint
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

        # Periodic checkpoint
        if epoch % save_every == 0:
            _save_checkpoint(
                model,
                path = ckpt_dir / f"epoch_{epoch:03d}_lora.pt",
                meta = {"epoch": epoch, "val_dice": val_metrics["val_dice"], "fold": fold_idx},
            )

    # -----------------------------------------------------------------------
    # Save training history
    # -----------------------------------------------------------------------
    history_path = Path(cfg["output"]["save_dir"]) / f"fold_{fold_idx}" / "train_history.json"
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
