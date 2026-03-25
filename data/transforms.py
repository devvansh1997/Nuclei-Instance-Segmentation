"""
data/transforms.py

Augmentation pipelines for training and validation.

Design notes:
  - Spatial transforms (flip, rotate) are applied identically to both the
    image and the instance mask to preserve spatial correspondence.
  - Photometric transforms (brightness, contrast, colour jitter) are applied
    to the image ONLY — instance mask pixel values are instance IDs and must
    not be altered.
  - No resizing or SAM-specific normalisation here.  That is handled inside
    the model wrapper (models/sam_lora.py) using SAM's own ResizeLongestSide
    so that point-prompt coordinates can be scaled correctly.
  - Instance masks are uint16 (NuInsSeg uses up to ~300 nuclei per image).
    albumentations is asked to treat them as integer masks so no interpolation
    is applied during spatial transforms.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import albumentations as A
    _ALBUMENTATIONS_OK = True
    logger.debug("albumentations %s loaded.", A.__version__)
except ImportError:
    _ALBUMENTATIONS_OK = False
    logger.warning(
        "albumentations not found. Falling back to numpy-only transforms. "
        "Install with: pip install 'albumentations>=1.4.0'"
    )


# ---------------------------------------------------------------------------
# albumentations-based pipelines (preferred)
# ---------------------------------------------------------------------------

def _build_train_pipeline_albu() -> "A.Compose":
    """
    Training augmentation pipeline using albumentations.

    Spatial ops applied to image + mask jointly.
    Photometric ops applied to image only (albumentations skips masks for these).
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # Photometric — image only
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.0,
                p=0.5,
            ),
        ],
        # Tell albumentations to treat 'instance_mask' exactly like 'mask'
        # so that identical spatial transforms are applied.
        additional_targets={"instance_mask": "mask"},
    )


def _build_val_pipeline_albu() -> "A.Compose":
    """
    Validation pipeline — no augmentation, just a pass-through.
    Kept as a Compose so the call interface is identical to training.
    """
    return A.Compose(
        [],
        additional_targets={"instance_mask": "mask"},
    )


# ---------------------------------------------------------------------------
# Fallback numpy-only transforms (used if albumentations is unavailable)
# ---------------------------------------------------------------------------

class _NumpyTransform:
    """
    Minimal spatial augmentation without albumentations.
    Applies the same random op to image and instance_mask.
    """

    def __init__(self, is_train: bool):
        self.is_train = is_train

    def __call__(self, image: np.ndarray, instance_mask: np.ndarray):
        """
        Parameters
        ----------
        image         : (H, W, 3) uint8
        instance_mask : (H, W)    uint16

        Returns
        -------
        image         : (H, W, 3) uint8
        instance_mask : (H, W)    uint16
        """
        if not self.is_train:
            return image, instance_mask

        # Random horizontal flip
        if np.random.rand() < 0.5:
            image         = np.fliplr(image).copy()
            instance_mask = np.fliplr(instance_mask).copy()

        # Random vertical flip
        if np.random.rand() < 0.5:
            image         = np.flipud(image).copy()
            instance_mask = np.flipud(instance_mask).copy()

        # Random 90° rotation (k = 0,1,2,3 quarter turns)
        k = np.random.randint(0, 4)
        if k > 0:
            image         = np.rot90(image,         k=k).copy()
            instance_mask = np.rot90(instance_mask, k=k).copy()

        # Brightness/contrast jitter (image only)
        if np.random.rand() < 0.5:
            alpha = 1.0 + np.random.uniform(-0.2, 0.2)   # contrast
            beta  =       np.random.uniform(-20,  20)     # brightness
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        return image, instance_mask


# ---------------------------------------------------------------------------
# Wrapper that unifies both backends behind a single interface
# ---------------------------------------------------------------------------

class SegmentationTransform:
    """
    Unified transform wrapper.

    Usage
    -----
    transform = SegmentationTransform(is_train=True)
    image_out, mask_out = transform(image, instance_mask)

    Parameters
    ----------
    is_train : bool
        True  → apply data augmentation (used for training split)
        False → pass-through only       (used for validation split)
    """

    def __init__(self, is_train: bool):
        self.is_train = is_train

        if _ALBUMENTATIONS_OK:
            self._pipeline = (
                _build_train_pipeline_albu() if is_train
                else _build_val_pipeline_albu()
            )
            self._backend = "albumentations"
        else:
            self._pipeline = _NumpyTransform(is_train=is_train)
            self._backend  = "numpy"

        logger.debug(
            "SegmentationTransform initialised | is_train=%s | backend=%s",
            is_train, self._backend,
        )

    def __call__(
        self,
        image: np.ndarray,
        instance_mask: np.ndarray,
    ) -> tuple:
        """
        Apply transforms to image and instance mask.

        Parameters
        ----------
        image         : (H, W, 3) uint8   — RGB image
        instance_mask : (H, W)    uint16  — per-nucleus instance IDs

        Returns
        -------
        image         : (H, W, 3) uint8
        instance_mask : (H, W)    uint16
        """
        if self._backend == "albumentations":
            # albumentations expects masks as 2-D arrays; it handles uint16 in v1.4+
            # Cast to int32 as a safe intermediate to avoid any internal uint16 issues,
            # then restore to uint16 after (values fit — max ~65535).
            mask_int32 = instance_mask.astype(np.int32)

            result = self._pipeline(image=image, instance_mask=mask_int32)

            image_out = result["image"]
            mask_out  = result["instance_mask"].astype(np.uint16)
        else:
            image_out, mask_out = self._pipeline(image, instance_mask)

        return image_out, mask_out


# ---------------------------------------------------------------------------
# Factory functions (used by dataset.py)
# ---------------------------------------------------------------------------

def get_train_transforms() -> SegmentationTransform:
    """Return training-time augmentation pipeline."""
    return SegmentationTransform(is_train=True)


def get_val_transforms() -> SegmentationTransform:
    """Return validation-time pass-through pipeline."""
    return SegmentationTransform(is_train=False)
