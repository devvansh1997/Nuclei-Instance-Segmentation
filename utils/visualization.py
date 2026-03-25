"""
utils/visualization.py

Visualization utilities for nuclei instance segmentation results.

Produces side-by-side comparison figures:
    [Original image | GT instances (coloured) | Predicted instances (coloured)]

Each nucleus instance is assigned a distinct random colour.
Boundaries are drawn on the overlay panel to make small nuclei visible.

Public API:
    visualize_predictions(...)   -- save a 3-panel comparison figure
    colorize_instances(...)      -- convert instance mask → RGB colour image
    save_fold_visualizations(...)-- batch-save N examples for a fold
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for HPC (no display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------

def colorize_instances(
    instance_mask: np.ndarray,
    seed:          int = 42,
    alpha:         float = 0.85,
) -> np.ndarray:
    """
    Convert a (H, W) integer instance mask to a (H, W, 3) uint8 RGB image.

    Each unique non-zero instance ID receives a randomly sampled colour.
    Background (ID = 0) is rendered as black.

    Parameters
    ----------
    instance_mask : (H, W) int/uint16 — instance IDs (0 = background).
    seed          : Random seed for colour generation (fixed = reproducible).
    alpha         : Brightness scaling for generated colours (0–1).
                    Keeps colours away from very dark shades for visibility.

    Returns
    -------
    (H, W, 3) uint8 RGB image.
    """
    H, W = instance_mask.shape
    rgb  = np.zeros((H, W, 3), dtype=np.uint8)

    unique_ids = [int(i) for i in np.unique(instance_mask) if i > 0]
    if not unique_ids:
        return rgb

    # Generate one colour per instance ID using a seeded RNG
    rng    = np.random.default_rng(seed)
    # Sample in float, scale by alpha, clip, convert to uint8
    colours = (
        rng.uniform(low=80, high=255, size=(len(unique_ids), 3)) * alpha
    ).astype(np.uint8)

    for colour, instance_id in zip(colours, unique_ids):
        rgb[instance_mask == instance_id] = colour

    return rgb


def _draw_boundaries(
    instance_mask: np.ndarray,
    rgb:           np.ndarray,
    boundary_colour: tuple = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw instance boundaries (white contours) onto an RGB image.

    Uses a simple morphological edge detection: boundary pixels are those
    where the instance ID differs from at least one neighbour.
    """
    try:
        import cv2
        rgb_out = rgb.copy()
        for instance_id in np.unique(instance_mask):
            if instance_id == 0:
                continue
            binary = (instance_mask == instance_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(rgb_out, contours, -1, boundary_colour, thickness)
        return rgb_out
    except ImportError:
        # If OpenCV is unavailable fall back to no boundaries
        logger.debug("OpenCV not found — skipping boundary drawing.")
        return rgb


# ---------------------------------------------------------------------------
# Main visualisation function
# ---------------------------------------------------------------------------

def visualize_predictions(
    image:         np.ndarray,
    gt_mask:       np.ndarray,
    pred_mask:     np.ndarray,
    save_path:     str,
    title:         Optional[str] = None,
    draw_boundary: bool = True,
    dpi:           int  = 150,
) -> None:
    """
    Save a 3-panel comparison figure:
        Panel 1 : Original H&E image
        Panel 2 : Ground-truth instance segmentation (coloured)
        Panel 3 : Predicted instance segmentation (coloured)

    Parameters
    ----------
    image         : (H, W, 3) uint8 RGB — original H&E patch.
    gt_mask       : (H, W) int/uint16  — ground-truth instance IDs.
    pred_mask     : (H, W) int/uint16  — predicted instance IDs.
    save_path     : Output file path (PNG).
    title         : Optional super-title (e.g. "Fold 0 — tissue_type").
    draw_boundary : If True, draw white contours around instances.
    dpi           : Figure resolution (default 150).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Colourize instance masks
    gt_rgb   = colorize_instances(gt_mask,   seed=42)
    pred_rgb = colorize_instances(pred_mask, seed=42)

    if draw_boundary:
        gt_rgb   = _draw_boundaries(gt_mask,   gt_rgb)
        pred_rgb = _draw_boundaries(pred_mask, pred_rgb)

    # Count instances for subtitle labels
    n_gt   = int(np.max(gt_mask))
    n_pred = int(np.max(pred_mask))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original H&E", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(gt_rgb)
    axes[1].set_title(f"Ground Truth  ({n_gt} nuclei)", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title(f"Prediction  ({n_pred} nuclei)", fontsize=11)
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Visualisation saved → %s", save_path)


def visualize_overlay(
    image:     np.ndarray,
    gt_mask:   np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    title:     Optional[str] = None,
    alpha:     float = 0.45,
    dpi:       int   = 150,
) -> None:
    """
    Save a 2-panel overlay figure (GT and Pred overlaid on the original image).

    Useful as a compact alternative to the 3-panel figure for the report.

    Parameters
    ----------
    alpha : Transparency of the coloured mask overlay (0=transparent, 1=opaque).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gt_rgb   = colorize_instances(gt_mask,   seed=42)
    pred_rgb = colorize_instances(pred_mask, seed=42)

    # Blend: overlay = alpha * mask_rgb + (1-alpha) * image
    image_f    = image.astype(np.float32)
    gt_overlay = np.clip(
        alpha * gt_rgb.astype(np.float32) + (1 - alpha) * image_f, 0, 255
    ).astype(np.uint8)
    pred_overlay = np.clip(
        alpha * pred_rgb.astype(np.float32) + (1 - alpha) * image_f, 0, 255
    ).astype(np.uint8)

    n_gt   = int(np.max(gt_mask))
    n_pred = int(np.max(pred_mask))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(gt_overlay)
    axes[0].set_title(f"GT overlay  ({n_gt} nuclei)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(pred_overlay)
    axes[1].set_title(f"Pred overlay  ({n_pred} nuclei)", fontsize=11)
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.debug("Overlay visualisation saved → %s", save_path)


# ---------------------------------------------------------------------------
# Batch saving for a full fold
# ---------------------------------------------------------------------------

def save_fold_visualizations(
    samples:       List[dict],
    fold_idx:      int,
    save_dir:      str,
    num_samples:   int = 8,
) -> None:
    """
    Save comparison figures for up to *num_samples* images from a fold.

    Parameters
    ----------
    samples    : List of dicts, each with keys:
                   "image"         (H, W, 3) uint8
                   "gt_mask"       (H, W) uint16
                   "pred_mask"     (H, W) uint16
                   "metrics"       dict {dice, aji, pq} for this image
                   "tissue"        str  tissue type name
                   "image_path"    str  source file path
    fold_idx   : Zero-based fold index (used in filenames).
    save_dir   : Directory to save figures into.
    num_samples: Maximum number of figures to save.
    """
    save_dir = Path(save_dir) / f"fold_{fold_idx}" / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)

    n = min(num_samples, len(samples))
    logger.info("Saving %d visualisation(s) for fold %d → %s", n, fold_idx, save_dir)

    for i, sample in enumerate(samples[:n]):
        image     = sample["image"]
        gt_mask   = sample["gt_mask"]
        pred_mask = sample["pred_mask"]
        metrics   = sample.get("metrics", {})
        tissue    = sample.get("tissue", "unknown")
        img_name  = Path(sample.get("image_path", f"sample_{i}")).stem

        metric_str = "  |  ".join(
            f"{k.upper()}={v:.3f}" for k, v in metrics.items()
            if k in ("dice", "aji", "pq")
        )
        title = f"Fold {fold_idx}  |  {tissue}  |  {img_name}\n{metric_str}"

        # 3-panel side-by-side
        visualize_predictions(
            image     = image,
            gt_mask   = gt_mask,
            pred_mask = pred_mask,
            save_path = str(save_dir / f"{img_name}_comparison.png"),
            title     = title,
        )

        # Overlay variant (compact, good for report)
        visualize_overlay(
            image     = image,
            gt_mask   = gt_mask,
            pred_mask = pred_mask,
            save_path = str(save_dir / f"{img_name}_overlay.png"),
            title     = title,
        )

    logger.info("Visualisations saved for fold %d.", fold_idx)
