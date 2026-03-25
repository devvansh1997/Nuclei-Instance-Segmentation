"""
data/dataset.py

NuInsSeg dataset loader with 5-fold cross-validation support.

Expected dataset layout on disk (one subdirectory per tissue type):

    <data.root>/
    ├── human_adrenal_gland/
    │   ├── images/
    │   │   ├── img_001.png
    │   │   └── ...
    │   └── masks/
    │       ├── img_001.tiff   ← 16-bit uint16 instance-ID mask
    │       └── ...
    ├── human_bladder/
    │   └── ...
    └── ...

Mask encoding (NuInsSeg convention):
    0          → background
    1, 2, 3 …  → individual nucleus instance IDs

Instance masks are stored as uint16 TIFF files.
Do NOT read them with PIL — PIL silently clips uint16 to uint8.
Use tifffile (or cv2 with IMREAD_UNCHANGED).

Public API:
    discover_dataset(root)              → (image_paths, mask_paths, tissue_labels)
    get_fold_splits(cfg)                → list of (train_idxs, val_idxs) per fold
    get_fold_datasets(cfg, fold_idx)    → (train_dataset, val_dataset)
    NuInsSegDataset                     → torch.utils.data.Dataset
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset

from data.transforms import get_train_transforms, get_val_transforms

logger = logging.getLogger(__name__)

# Supported image / mask extensions
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
_MASK_EXTS  = {".tiff", ".tif", ".png"}


# ---------------------------------------------------------------------------
# Mask loading helpers
# ---------------------------------------------------------------------------

def _load_mask(mask_path: Path) -> np.ndarray:
    """
    Load a NuInsSeg instance mask as uint16.

    Tries tifffile first (most reliable for uint16 TIFF).
    Falls back to OpenCV with IMREAD_UNCHANGED.

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint16
    """
    ext = mask_path.suffix.lower()

    if ext in {".tiff", ".tif"}:
        try:
            import tifffile
            mask = tifffile.imread(str(mask_path))
            # Ensure 2-D (squeeze channel dim if present)
            if mask.ndim == 3:
                mask = mask[..., 0]
            return mask.astype(np.uint16)
        except ImportError:
            logger.warning(
                "tifffile not found — falling back to OpenCV for TIFF loading. "
                "Install tifffile: pip install tifffile"
            )

    # OpenCV fallback (also handles PNG masks)
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.uint16)


def _load_image(image_path: Path) -> np.ndarray:
    """
    Load an H&E image as uint8 RGB.

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype uint8
    """
    import cv2
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    # OpenCV loads as BGR — convert to RGB
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_dataset(root: Path) -> tuple:
    """
    Recursively scan *root* for paired (image, mask) files.

    Walks every subdirectory.  A pair is valid when:
      - A file exists under any `images/` folder
      - A mask with the same stem exists under the sibling `masks/` folder

    Parameters
    ----------
    root : Path   Dataset root directory (data.root in config).

    Returns
    -------
    image_paths   : list[Path]  — absolute paths to image files
    mask_paths    : list[Path]  — absolute paths to corresponding mask files
    tissue_labels : list[str]   — tissue-type name for each pair (for logging)
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Download NuInsSeg from https://zenodo.org/records/10518968 "
            f"and place it at {root}"
        )

    image_paths:   list = []
    mask_paths:    list = []
    tissue_labels: list = []

    # Each immediate subdirectory of root is a tissue type
    tissue_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    if not tissue_dirs:
        raise RuntimeError(
            f"No tissue-type subdirectories found under {root}. "
            "Expected structure: <root>/<tissue_type>/images/ and masks/"
        )

    logger.debug("Scanning %d tissue directories under %s", len(tissue_dirs), root)

    for tissue_dir in tissue_dirs:
        images_dir = tissue_dir / "tissue images"
        masks_dir  = tissue_dir / "label masks"

        if not images_dir.exists() or not masks_dir.exists():
            logger.debug(
                "Skipping %s — missing 'tissue images/' or 'label masks/' subdirectory.",
                tissue_dir.name,
            )
            continue

        # Collect all valid image files in this tissue directory
        candidates = sorted(
            [f for f in images_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
        )

        tissue_pairs = 0
        for img_path in candidates:
            # Try to find the matching mask by stem (same filename, any mask ext)
            mask_path = _find_mask(masks_dir, img_path.stem)
            if mask_path is None:
                logger.debug(
                    "No mask found for image %s — skipping.", img_path.name
                )
                continue

            image_paths.append(img_path)
            mask_paths.append(mask_path)
            tissue_labels.append(tissue_dir.name)
            tissue_pairs += 1

        logger.debug("  %-35s  %d pairs", tissue_dir.name, tissue_pairs)

    if not image_paths:
        raise RuntimeError(
            f"No valid image/mask pairs found under {root}. "
            "Check the dataset structure and file extensions."
        )

    logger.info(
        "Dataset discovery: %d image/mask pairs across %d tissue types.",
        len(image_paths), len(set(tissue_labels)),
    )

    # Log tissue-type distribution at DEBUG level
    if logger.isEnabledFor(logging.DEBUG):
        from collections import Counter
        counts = Counter(tissue_labels)
        for tissue, count in sorted(counts.items()):
            logger.debug("  %s: %d images", tissue, count)

    return image_paths, mask_paths, tissue_labels


def _find_mask(masks_dir: Path, stem: str) -> Optional[Path]:
    """Return the mask file for a given image stem, or None if not found."""
    for ext in _MASK_EXTS:
        candidate = masks_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# 5-fold split
# ---------------------------------------------------------------------------

def get_fold_splits(cfg: dict) -> list:
    """
    Compute 5-fold cross-validation splits over all discovered images.

    Parameters
    ----------
    cfg : Full config dict.

    Returns
    -------
    list of (train_indices, val_indices) tuples — one per fold.
    """
    data_cfg = cfg["data"]
    root     = Path(data_cfg["root"])
    n_folds  = data_cfg.get("n_folds", 5)
    seed     = data_cfg.get("seed", 42)

    image_paths, mask_paths, tissue_labels = discover_dataset(root)
    n_total = len(image_paths)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = []

    for fold_idx, (train_idxs, val_idxs) in enumerate(kf.split(range(n_total))):
        splits.append((train_idxs.tolist(), val_idxs.tolist()))
        logger.debug(
            "Fold %d: %d train | %d val",
            fold_idx, len(train_idxs), len(val_idxs),
        )

    logger.info(
        "5-fold splits created: %d total images, ~%d per fold.",
        n_total, n_total // n_folds,
    )
    return image_paths, mask_paths, tissue_labels, splits


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class NuInsSegDataset(Dataset):
    """
    PyTorch Dataset for the NuInsSeg nuclei instance segmentation dataset.

    Each item returned is a dict with keys:
        "image"         : (H, W, 3) np.uint8   — RGB H&E patch
        "instance_mask" : (H, W)    np.uint16  — per-nucleus instance IDs (0=BG)
        "binary_mask"   : (H, W)    np.bool_   — foreground / background
        "image_path"    : str                  — absolute path (for debugging)
        "mask_path"     : str                  — absolute path (for debugging)
        "tissue"        : str                  — tissue type name

    The training loop is responsible for sampling point prompts from the
    instance mask.  Keeping prompt sampling out of the Dataset keeps this
    class simple and reusable.
    """

    def __init__(
        self,
        image_paths:   list,
        mask_paths:    list,
        tissue_labels: list,
        transform=None,
        debug:         bool = False,
        max_samples:   Optional[int] = None,
    ):
        """
        Parameters
        ----------
        image_paths   : list[Path]  Absolute paths to images.
        mask_paths    : list[Path]  Absolute paths to matching instance masks.
        tissue_labels : list[str]   Tissue type for each pair (for logging).
        transform     : SegmentationTransform or None.
        debug         : If True and max_samples is set, cap dataset size.
        max_samples   : Maximum number of samples to use (debug mode).
        """
        assert len(image_paths) == len(mask_paths) == len(tissue_labels), (
            "image_paths, mask_paths, tissue_labels must have equal length."
        )

        self.image_paths   = [Path(p) for p in image_paths]
        self.mask_paths    = [Path(p) for p in mask_paths]
        self.tissue_labels = tissue_labels
        self.transform     = transform

        # Cap dataset size in debug mode
        if debug and max_samples is not None:
            cap = int(max_samples)
            if len(self.image_paths) > cap:
                logger.warning(
                    "DEBUG: capping dataset from %d to %d samples.",
                    len(self.image_paths), cap,
                )
                self.image_paths   = self.image_paths[:cap]
                self.mask_paths    = self.mask_paths[:cap]
                self.tissue_labels = self.tissue_labels[:cap]

        logger.info(
            "NuInsSegDataset ready: %d samples | transform=%s",
            len(self), "yes" if transform else "none",
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        mask_path  = self.mask_paths[idx]

        # --- Load -------------------------------------------------------
        try:
            image         = _load_image(image_path)          # (H, W, 3) uint8
            instance_mask = _load_mask(mask_path)             # (H, W)    uint16
        except Exception as exc:
            logger.error(
                "Failed to load sample %d | image=%s | mask=%s | error=%s",
                idx, image_path, mask_path, exc,
            )
            raise

        # Sanity check: image and mask must have matching spatial dimensions
        if image.shape[:2] != instance_mask.shape[:2]:
            logger.error(
                "Shape mismatch at idx=%d: image=%s mask=%s | image=%s mask=%s",
                idx, image_path.name, mask_path.name,
                image.shape[:2], instance_mask.shape[:2],
            )
            raise ValueError(
                f"Image/mask spatial shape mismatch at index {idx}: "
                f"image {image.shape[:2]} vs mask {instance_mask.shape[:2]}"
            )

        # --- Augment ----------------------------------------------------
        if self.transform is not None:
            image, instance_mask = self.transform(image, instance_mask)

        # --- Derived fields ---------------------------------------------
        binary_mask = (instance_mask > 0)   # foreground = any nucleus

        # Log shapes on the very first item to catch any silent misreads
        if idx == 0:
            logger.debug(
                "Sample[0] loaded | image=%s dtype=%s | mask=%s dtype=%s "
                "| n_instances=%d | foreground_frac=%.3f",
                image.shape, image.dtype,
                instance_mask.shape, instance_mask.dtype,
                len(np.unique(instance_mask)) - 1,  # -1 to exclude background
                binary_mask.mean(),
            )

        return {
            "image":         image,
            "instance_mask": instance_mask,
            "binary_mask":   binary_mask,
            "image_path":    str(image_path),
            "mask_path":     str(mask_path),
            "tissue":        self.tissue_labels[idx],
        }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_fold_datasets(cfg: dict, fold_idx: int) -> tuple:
    """
    Build train and validation NuInsSegDataset objects for one CV fold.

    Parameters
    ----------
    cfg      : Full config dict.
    fold_idx : Zero-based fold index (0 … n_folds-1).

    Returns
    -------
    train_dataset : NuInsSegDataset  with training-time augmentation
    val_dataset   : NuInsSegDataset  with validation-time (no augmentation)
    """
    data_cfg  = cfg["data"]
    debug_cfg = cfg.get("debug", {})

    debug       = debug_cfg.get("enabled", False)
    max_samples = debug_cfg.get("max_samples", None)

    image_paths, mask_paths, tissue_labels, splits = get_fold_splits(cfg)
    n_folds = data_cfg.get("n_folds", 5)

    if fold_idx < 0 or fold_idx >= n_folds:
        raise ValueError(
            f"fold_idx={fold_idx} is out of range [0, {n_folds - 1}]."
        )

    train_idxs, val_idxs = splits[fold_idx]

    # Gather paths for this fold
    train_images  = [image_paths[i]   for i in train_idxs]
    train_masks   = [mask_paths[i]    for i in train_idxs]
    train_tissues = [tissue_labels[i] for i in train_idxs]

    val_images    = [image_paths[i]   for i in val_idxs]
    val_masks     = [mask_paths[i]    for i in val_idxs]
    val_tissues   = [tissue_labels[i] for i in val_idxs]

    logger.info(
        "Fold %d/%d | train=%d | val=%d",
        fold_idx, n_folds - 1, len(train_images), len(val_images),
    )

    train_dataset = NuInsSegDataset(
        image_paths   = train_images,
        mask_paths    = train_masks,
        tissue_labels = train_tissues,
        transform     = get_train_transforms(),
        debug         = debug,
        max_samples   = max_samples,
    )

    val_dataset = NuInsSegDataset(
        image_paths   = val_images,
        mask_paths    = val_masks,
        tissue_labels = val_tissues,
        transform     = get_val_transforms(),
        debug         = debug,
        # Validation: cap at half of max_samples to keep debug fast
        max_samples   = (max_samples // 2) if (debug and max_samples) else None,
    )

    return train_dataset, val_dataset
