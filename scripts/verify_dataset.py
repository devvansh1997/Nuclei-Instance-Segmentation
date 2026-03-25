"""
scripts/verify_dataset.py

Verify the NuInsSeg dataset structure before training.

Checks:
  1. Dataset root exists and contains tissue-type subdirectories.
  2. Each tissue directory has both images/ and masks/ subdirectories.
  3. Every image file has a paired mask file (same stem, any supported extension).
  4. Mask files are readable and actually uint16 (not silently clipped to uint8).
  5. Image and mask spatial dimensions match for every pair.
  6. Masks contain at least one nucleus (non-zero pixel) — flags empty masks.
  7. Reports instance count statistics (min / max / mean nuclei per image).
  8. Reports per-tissue-type image counts.
  9. Flags any orphan files (image without mask, mask without image).

Usage:
    python scripts/verify_dataset.py --config configs/debug.yaml
    python scripts/verify_dataset.py --config configs/train_a100.yaml

Exit code 0 = all checks passed.
Exit code 1 = one or more issues found (see ERROR lines in output).
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging setup (standalone — does not depend on utils/logger.py)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("verify_dataset")

# Supported extensions
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_MASK_EXTS  = {".tiff", ".tif", ".png"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mask_safe(mask_path: Path):
    """Load mask, try tifffile first then OpenCV. Returns (array, backend_used)."""
    ext = mask_path.suffix.lower()
    if ext in {".tiff", ".tif"}:
        try:
            import tifffile
            mask = tifffile.imread(str(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            return mask, "tifffile"
        except ImportError:
            pass
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None, "failed"
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask, "opencv"


def load_image_safe(image_path: Path):
    """Load image with OpenCV. Returns (array, shape) or (None, None)."""
    import cv2
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    return img, img.shape[:2]


def find_mask(masks_dir: Path, stem: str):
    """Find a mask file matching stem in any supported extension."""
    for ext in _MASK_EXTS:
        candidate = masks_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def verify(root: Path) -> bool:
    """
    Run all verification checks on the dataset root.
    Returns True if all checks pass, False if any errors found.
    """
    errors   = []
    warnings = []

    # -----------------------------------------------------------------------
    # Check 1: root exists
    # -----------------------------------------------------------------------
    logger.info("=" * 65)
    logger.info("NuInsSeg Dataset Verification")
    logger.info("Root: %s", root.resolve())
    logger.info("=" * 65)

    if not root.exists():
        logger.error("Dataset root does not exist: %s", root)
        logger.error(
            "Download from https://zenodo.org/records/10518968 and unzip "
            "so that the path above exists."
        )
        return False

    # -----------------------------------------------------------------------
    # Check 2: tissue-type subdirectories
    # -----------------------------------------------------------------------
    tissue_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not tissue_dirs:
        logger.error("No subdirectories found under root — unexpected structure.")
        return False

    logger.info("Found %d top-level subdirectories.", len(tissue_dirs))

    # -----------------------------------------------------------------------
    # Check 3–9: per-tissue and per-pair checks
    # -----------------------------------------------------------------------
    tissue_counts          = {}
    all_instance_counts    = []
    dtype_counter          = defaultdict(int)
    backend_counter        = defaultdict(int)
    total_pairs            = 0
    total_errors           = 0
    total_empty_masks      = 0
    orphan_images          = []
    orphan_masks           = []

    for tissue_dir in tissue_dirs:
        images_dir = tissue_dir / "tissue images"
        masks_dir  = tissue_dir / "label masks"

        # Check 'tissue images/' and 'label masks/' exist
        if not images_dir.exists():
            warnings.append(f"{tissue_dir.name}: missing 'tissue images/' subdirectory — skipped.")
            logger.warning("%-35s  missing 'tissue images/' — skipped", tissue_dir.name)
            continue
        if not masks_dir.exists():
            warnings.append(f"{tissue_dir.name}: missing 'label masks/' subdirectory — skipped.")
            logger.warning("%-35s  missing 'label masks/' — skipped", tissue_dir.name)
            continue

        image_files = sorted([
            f for f in images_dir.iterdir()
            if f.suffix.lower() in _IMAGE_EXTS
        ])
        mask_files = sorted([
            f for f in masks_dir.iterdir()
            if f.suffix.lower() in _MASK_EXTS
        ])

        mask_stems = {f.stem for f in mask_files}
        image_stems = {f.stem for f in image_files}

        # Detect orphan masks
        for m in mask_files:
            if m.stem not in image_stems:
                orphan_masks.append(str(m))

        tissue_pairs = 0

        for img_path in image_files:
            mask_path = find_mask(masks_dir, img_path.stem)

            # Check 3: paired mask exists
            if mask_path is None:
                orphan_images.append(str(img_path))
                errors.append(f"No mask found for: {img_path}")
                total_errors += 1
                continue

            # Check 4: load mask and verify dtype
            mask, backend = load_mask_safe(mask_path)
            backend_counter[backend] += 1

            if mask is None:
                errors.append(f"Failed to load mask: {mask_path}")
                total_errors += 1
                continue

            dtype_counter[str(mask.dtype)] += 1

            if mask.dtype == np.uint8:
                warnings.append(
                    f"Mask is uint8 (expected uint16): {mask_path.name} — "
                    "instance IDs > 255 will be clipped. Check loading pipeline."
                )
                logger.warning(
                    "  DTYPE  %-40s  uint8 — may be clipped!", mask_path.name
                )

            # Check 5: image/mask shape match
            img, img_shape = load_image_safe(img_path)
            if img is None:
                errors.append(f"Failed to load image: {img_path}")
                total_errors += 1
                continue

            if img_shape != mask.shape[:2]:
                errors.append(
                    f"Shape mismatch: {img_path.name} "
                    f"image={img_shape} mask={mask.shape[:2]}"
                )
                total_errors += 1
                continue

            # Check 6: mask has at least one nucleus
            n_instances = len(np.unique(mask)) - 1   # subtract background
            if n_instances == 0:
                total_empty_masks += 1
                warnings.append(f"Empty mask (no nuclei): {mask_path}")
                logger.warning("  EMPTY  %s", mask_path.name)
            else:
                all_instance_counts.append(n_instances)

            tissue_pairs += 1
            total_pairs  += 1

        tissue_counts[tissue_dir.name] = tissue_pairs
        logger.info(
            "  %-35s  %3d image/mask pairs",
            tissue_dir.name, tissue_pairs,
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 65)
    logger.info("SUMMARY")
    logger.info("=" * 65)
    logger.info("Total tissue types   : %d", len(tissue_counts))
    logger.info("Total valid pairs    : %d", total_pairs)
    logger.info("Orphan images        : %d", len(orphan_images))
    logger.info("Orphan masks         : %d", len(orphan_masks))
    logger.info("Empty masks (warn)   : %d", total_empty_masks)

    logger.info("")
    logger.info("Mask dtype distribution:")
    for dtype, count in sorted(dtype_counter.items()):
        flag = " ← EXPECTED" if dtype == "uint16" else " ← WARNING"
        logger.info("  %-10s  %d files%s", dtype, count, flag)

    logger.info("")
    logger.info("Mask loader backend used:")
    for backend, count in sorted(backend_counter.items()):
        logger.info("  %-10s  %d files", backend, count)

    if all_instance_counts:
        logger.info("")
        logger.info("Nuclei per image statistics (non-empty masks):")
        logger.info("  Min    : %d",   min(all_instance_counts))
        logger.info("  Max    : %d",   max(all_instance_counts))
        logger.info("  Mean   : %.1f", np.mean(all_instance_counts))
        logger.info("  Median : %.1f", np.median(all_instance_counts))

    logger.info("")
    if errors:
        logger.info("ERRORS (%d):", len(errors))
        for e in errors:
            logger.error("  %s", e)

    if warnings:
        logger.info("WARNINGS (%d):", len(warnings))
        for w in warnings:
            logger.warning("  %s", w)

    logger.info("=" * 65)
    if total_errors == 0:
        logger.info("RESULT: ALL CHECKS PASSED")
        logger.info("Dataset is ready for training.")
    else:
        logger.error("RESULT: %d ERROR(S) FOUND — fix before training.", total_errors)
    logger.info("=" * 65)

    return total_errors == 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify NuInsSeg dataset structure.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (data.root is read from it).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    root = Path(cfg["data"]["root"])
    ok   = verify(root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
