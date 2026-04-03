"""
evaluate.py

Evaluate a trained MobileSAM + LoRA model on one fold's validation set.

Two inference modes (selected via evaluation.inference_mode in config):

  "watershed" (default, recommended):
      HED colour deconvolution → hematoxylin foreground → distance-transform
      watershed → centroid per nucleus → fine-tuned SAM prompted with that
      centroid.  Matches the training-time prompting strategy, giving a fair
      assessment of what LoRA learned.  Image embedding is computed once per
      image; only the lightweight mask decoder is called per nucleus.

  "amg":
      SamAutomaticMaskGenerator — original approach, kept for ablation.
      AMG thresholds are calibrated for vanilla SAM and may not transfer
      well after LoRA fine-tuning.

Metrics per image, averaged over the fold:
    Dice — binary foreground overlap
    AJI  — Aggregated Jaccard Index (instance-level)
    PQ   — Panoptic Quality = SQ × RQ

Results are saved as JSON so cross_validate.py can aggregate across folds.

Usage:
    python evaluate.py --config configs/debug.yaml         --fold 0
    python evaluate.py --config configs/train_a100.yaml    --fold 0

Re-evaluate all folds using existing checkpoints (no re-training):
    python cross_validate.py --config configs/train_a100.yaml --skip-train
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data.dataset import get_fold_datasets
from models.sam_lora import build_model
from utils.logger import setup_logger, log_system_info, log_config
from utils.metrics import compute_all_metrics, aggregate_metrics, masks_to_instance_map
from utils.visualization import save_fold_visualizations

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AMG import guard
# ---------------------------------------------------------------------------
try:
    from mobile_sam import SamAutomaticMaskGenerator
    _AMG_AVAILABLE = True
except ImportError:
    _AMG_AVAILABLE = False
    logger.error("mobile_sam not found — cannot run evaluation.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_best_checkpoint(model, cfg: dict, fold_idx: int) -> None:
    """Load the best LoRA checkpoint saved during training for this fold."""
    ckpt_path = (
        Path(cfg["output"]["save_dir"])
        / f"fold_{fold_idx}"
        / "checkpoints"
        / "best_lora.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found for fold {fold_idx}: {ckpt_path}\n"
            "Run train.py for this fold first."
        )
    model.load_lora_weights(str(ckpt_path))
    logger.info("Loaded best trainable weights from %s", ckpt_path)


def _build_amg(model, cfg: dict) -> "SamAutomaticMaskGenerator":
    """Instantiate SamAutomaticMaskGenerator with config-driven parameters."""
    if not _AMG_AVAILABLE:
        raise ImportError("mobile_sam package required for evaluation.")

    eval_cfg = cfg["evaluation"]
    amg = SamAutomaticMaskGenerator(
        model                  = model.sam,
        points_per_side        = eval_cfg.get("points_per_side",        32),
        pred_iou_thresh        = eval_cfg.get("pred_iou_thresh",        0.75),
        stability_score_thresh = eval_cfg.get("stability_score_thresh", 0.85),
        box_nms_thresh         = eval_cfg.get("box_nms_thresh",         0.7),
        min_mask_region_area   = eval_cfg.get("min_mask_region_area",   100),
    )
    logger.info(
        "AMG config | points_per_side=%d | pred_iou_thresh=%.2f "
        "| stability_thresh=%.2f | min_area=%d",
        eval_cfg.get("points_per_side", 32),
        eval_cfg.get("pred_iou_thresh", 0.75),
        eval_cfg.get("stability_score_thresh", 0.85),
        eval_cfg.get("min_mask_region_area", 100),
    )
    return amg


# ---------------------------------------------------------------------------
# GT Box Inference
# ---------------------------------------------------------------------------

def _gt_box_inference(
    model,
    image:         np.ndarray,
    gt_inst_mask:  np.ndarray,
    cfg:           dict,
    device:        torch.device,
) -> np.ndarray:
    """
    Inference using Ground Truth bounding boxes.

    This mode bypasses automated detection and gives SAM the exact
    "answer" box for every nucleus. Matches the reference repository's
    evaluation protocol.

    Parameters
    ----------
    model        : MobileSAMLoRA.
    image        : (H, W, 3) uint8 RGB.
    gt_inst_mask : (H, W) uint16 — Ground Truth instance IDs.
    cfg          : Full config dict.
    device       : torch.device.

    Returns
    -------
    (H, W) uint16 instance map.
    """
    mask_thresh = cfg["evaluation"].get("mask_threshold", 0.5)
    H, W = image.shape[:2]

    # Preprocess once
    img_tensor, orig_size, input_size = model.preprocess_image(image)

    with torch.no_grad():
        image_embeddings, image_pe = model.encode_image(img_tensor)

    instance_map = np.zeros((H, W), dtype=np.uint16)

    # Get boxes for all Ground Truth instances
    gt_ids = np.unique(gt_inst_mask)
    gt_ids = gt_ids[gt_ids > 0]

    for inst_id, nid in enumerate(gt_ids, start=1):
        rows, cols = np.where(gt_inst_mask == nid)
        # SAM box format: [x1, y1, x2, y2]
        box = np.array(
            [cols.min(), rows.min(), cols.max(), rows.max()],
            dtype=np.float32
        )

        # Scale box to SAM's 1024x1024 input space
        scaled_box = model.resize_transform.apply_boxes(box[None, :], orig_size)
        box_t = torch.tensor(scaled_box, dtype=torch.float32).to(device)

        with torch.no_grad():
            sparse_emb, dense_emb = model.sam.prompt_encoder(
                points=None,
                boxes=box_t,
                masks=None,
            )
            low_res_masks, _ = model.sam.mask_decoder(
                image_embeddings         = image_embeddings,
                image_pe                 = image_pe,
                sparse_prompt_embeddings = sparse_emb,
                dense_prompt_embeddings  = dense_emb,
                multimask_output         = False,
            )

        full_res = model.postprocess_masks(low_res_masks, input_size, orig_size)
        mask_bin = (full_res.sigmoid() > mask_thresh).squeeze().cpu().numpy()

        # Fill the instance map
        instance_map[mask_bin] = inst_id

    return instance_map


# ---------------------------------------------------------------------------
# Main evaluation function (one fold)
# ---------------------------------------------------------------------------

def evaluate_fold(cfg: dict, fold_idx: int) -> dict:
    """
    Evaluate the trained model on the validation split of one fold.

    Parameters
    ----------
    cfg      : Full config dict.
    fold_idx : Zero-based fold index.

    Returns
    -------
    dict with fold-level metrics: {fold, dice, aji, pq, sq, rq}
    """
    run_name = f"fold_{fold_idx}_eval"
    logger_  = setup_logger(cfg, run_name)
    log_system_info(logger_)
    log_config(logger_, cfg)

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger_.info("Device: %s", device)

    # -----------------------------------------------------------------------
    # Model — load pretrained SAM + best weights
    # -----------------------------------------------------------------------
    model = build_model(cfg, device)
    _load_best_checkpoint(model, cfg, fold_idx)
    model.eval()

    # -----------------------------------------------------------------------
    # Data — validation split only
    # -----------------------------------------------------------------------
    _, val_dataset = get_fold_datasets(cfg, fold_idx)
    logger_.info("Evaluating %d validation images.", len(val_dataset))

    # -----------------------------------------------------------------------
    # Inference setup
    # -----------------------------------------------------------------------
    eval_cfg       = cfg["evaluation"]
    inference_mode = eval_cfg.get("inference_mode", "gt_box")
    iou_thresh     = eval_cfg.get("iou_threshold", 0.5)
    min_area       = eval_cfg.get("min_mask_region_area", 50)
    num_vis        = cfg["output"].get("num_vis_samples", 8)

    if inference_mode == "amg":
        amg = _build_amg(model, cfg)
    else:
        amg = None
        logger_.info(f"Inference mode: {inference_mode}")

    per_image_metrics: list = []
    vis_samples:       list = []

    # -----------------------------------------------------------------------
    # Inference loop
    # -----------------------------------------------------------------------
    for idx in range(len(val_dataset)):
        sample = val_dataset[idx]
        image         = sample["image"]           # (H, W, 3) uint8
        gt_inst_mask  = sample["instance_mask"]   # (H, W) uint16
        tissue        = sample["tissue"]
        image_path    = sample["image_path"]

        logger_.debug(
            "[%d/%d] %s | gt_instances=%d",
            idx + 1, len(val_dataset),
            Path(image_path).name,
            int(np.max(gt_inst_mask)),
        )

        t0 = time.time()

        try:
            if inference_mode == "gt_box":
                pred_inst_mask = _gt_box_inference(
                    model, image, gt_inst_mask, cfg, device
                )
            else:
                # AMG inference — expects uint8 RGB
                with torch.no_grad():
                    raw_masks = amg.generate(image)
                pred_inst_mask = masks_to_instance_map(raw_masks, min_area=min_area)
        except Exception as exc:
            logger_.error(
                "Inference failed on %s: %s", Path(image_path).name, exc
            )
            per_image_metrics.append(
                {"dice": 0.0, "aji": 0.0, "pq": 0.0, "sq": 0.0, "rq": 0.0}
            )
            continue

        elapsed_ms = (time.time() - t0) * 1000

        logger_.debug(
            "  %s: %d instances | %.1f ms",
            inference_mode, int(np.max(pred_inst_mask)), elapsed_ms,
        )

        # Compute metrics
        metrics = compute_all_metrics(
            pred_inst_mask, gt_inst_mask, iou_threshold=iou_thresh
        )
        per_image_metrics.append(metrics)

        logger_.debug(
            "  Dice=%.4f  AJI=%.4f  PQ=%.4f  SQ=%.4f  RQ=%.4f",
            metrics["dice"], metrics["aji"],
            metrics["pq"], metrics["sq"], metrics["rq"],
        )

        # Collect visualisation samples (first num_vis images)
        if len(vis_samples) < num_vis:
            vis_samples.append({
                "image":      image,
                "gt_mask":    gt_inst_mask,
                "pred_mask":  pred_inst_mask,
                "metrics":    metrics,
                "tissue":     tissue,
                "image_path": image_path,
            })

    # -----------------------------------------------------------------------
    # Aggregate fold-level metrics
    # -----------------------------------------------------------------------
    fold_metrics = aggregate_metrics(per_image_metrics)

    logger_.info("=" * 60)
    logger_.info("FOLD %d RESULTS  (%d images)", fold_idx, len(per_image_metrics))
    logger_.info("=" * 60)
    logger_.info("  Dice : %.4f", fold_metrics["dice"])
    logger_.info("  AJI  : %.4f", fold_metrics["aji"])
    logger_.info("  PQ   : %.4f", fold_metrics["pq"])
    logger_.info("  SQ   : %.4f", fold_metrics["sq"])
    logger_.info("  RQ   : %.4f", fold_metrics["rq"])
    logger_.info("=" * 60)

    # -----------------------------------------------------------------------
    # Save per-image + fold-level results
    # -----------------------------------------------------------------------
    results_dir  = Path(cfg["output"]["save_dir"]) / f"fold_{fold_idx}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-image
    per_image_path = results_dir / "per_image_metrics.json"
    with open(per_image_path, "w") as f:
        json.dump(per_image_metrics, f, indent=2)
    logger_.info("Per-image metrics saved → %s", per_image_path)

    # Fold summary
    fold_result = {"fold": fold_idx, **fold_metrics}
    fold_path   = results_dir / "fold_metrics.json"
    with open(fold_path, "w") as f:
        json.dump(fold_result, f, indent=2)
    logger_.info("Fold metrics saved → %s", fold_path)

    # -----------------------------------------------------------------------
    # Visualisations
    # -----------------------------------------------------------------------
    if vis_samples:
        save_fold_visualizations(
            samples     = vis_samples,
            fold_idx    = fold_idx,
            save_dir    = cfg["output"]["save_dir"],
            num_samples = num_vis,
        )

    return fold_result
        except Exception as exc:
            logger_.error(
                "Inference failed on %s: %s", Path(image_path).name, exc
            )
            per_image_metrics.append(
                {"dice": 0.0, "aji": 0.0, "pq": 0.0, "sq": 0.0, "rq": 0.0}
            )
            continue

        elapsed_ms = (time.time() - t0) * 1000

        logger_.debug(
            "  %s: %d instances | %.1f ms",
            inference_mode, int(np.max(pred_inst_mask)), elapsed_ms,
        )

        # Compute metrics
        metrics = compute_all_metrics(
            pred_inst_mask, gt_inst_mask, iou_threshold=iou_thresh
        )
        per_image_metrics.append(metrics)

        logger_.debug(
            "  Dice=%.4f  AJI=%.4f  PQ=%.4f  SQ=%.4f  RQ=%.4f",
            metrics["dice"], metrics["aji"],
            metrics["pq"], metrics["sq"], metrics["rq"],
        )

        # Collect visualisation samples (first num_vis images)
        if len(vis_samples) < num_vis:
            vis_samples.append({
                "image":      image,
                "gt_mask":    gt_inst_mask,
                "pred_mask":  pred_inst_mask,
                "metrics":    metrics,
                "tissue":     tissue,
                "image_path": image_path,
            })

    # -----------------------------------------------------------------------
    # Aggregate fold-level metrics
    # -----------------------------------------------------------------------
    fold_metrics = aggregate_metrics(per_image_metrics)

    logger_.info("=" * 60)
    logger_.info("FOLD %d RESULTS  (%d images)", fold_idx, len(per_image_metrics))
    logger_.info("=" * 60)
    logger_.info("  Dice : %.4f", fold_metrics["dice"])
    logger_.info("  AJI  : %.4f", fold_metrics["aji"])
    logger_.info("  PQ   : %.4f", fold_metrics["pq"])
    logger_.info("  SQ   : %.4f", fold_metrics["sq"])
    logger_.info("  RQ   : %.4f", fold_metrics["rq"])
    logger_.info("=" * 60)

    # -----------------------------------------------------------------------
    # Save per-image + fold-level results
    # -----------------------------------------------------------------------
    results_dir  = Path(cfg["output"]["save_dir"]) / f"fold_{fold_idx}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-image
    per_image_path = results_dir / "per_image_metrics.json"
    with open(per_image_path, "w") as f:
        json.dump(per_image_metrics, f, indent=2)
    logger_.info("Per-image metrics saved → %s", per_image_path)

    # Fold summary
    fold_result = {"fold": fold_idx, **fold_metrics}
    fold_path   = results_dir / "fold_metrics.json"
    with open(fold_path, "w") as f:
        json.dump(fold_result, f, indent=2)
    logger_.info("Fold metrics saved → %s", fold_path)

    # -----------------------------------------------------------------------
    # Visualisations
    # -----------------------------------------------------------------------
    if vis_samples:
        save_fold_visualizations(
            samples     = vis_samples,
            fold_idx    = fold_idx,
            save_dir    = cfg["output"]["save_dir"],
            num_samples = num_vis,
        )

    return fold_result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained MobileSAM+LoRA on one NuInsSeg fold."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--fold",   required=True, type=int, help="Fold index (0-4).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    evaluate_fold(cfg, args.fold)


if __name__ == "__main__":
    main()
