"""
models/sam_lora.py

MobileSAM + LoRA wrapper for nuclei instance segmentation.

This module:
  1. Loads a pretrained MobileSAM checkpoint.
  2. Injects LoRA into specified attention Linear layers.
  3. Freezes all non-LoRA parameters.
  4. Provides a clean forward pass for training (point-prompted per instance).
  5. Provides preprocessing / postprocessing utilities that keep SAM's
     coordinate system consistent (ResizeLongestSide → pad to 1024×1024).
  6. Provides save/load for LoRA-only weights (small checkpoint files).

Public API:
    MobileSAMLoRA           -- the main model class
    build_model(cfg, device)-- factory: load, inject, freeze, log, move to device
    sample_point_prompts    -- sample pos/neg point prompts from an instance mask
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.lora import (
    inject_lora,
    freeze_non_lora,
    freeze_image_encoder_non_lora,
    count_parameters,
    log_parameter_summary,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MobileSAM import guard
# ---------------------------------------------------------------------------
try:
    from mobile_sam import sam_model_registry
    from mobile_sam.utils.transforms import ResizeLongestSide
    _MOBILE_SAM_AVAILABLE = True
except ImportError:
    _MOBILE_SAM_AVAILABLE = False
    logger.error(
        "mobile_sam package not found. "
        "Install with: pip install mobile-sam"
    )

_SAM_MODEL_TYPE = "vit_t"   # MobileSAM uses TinyViT ("vit_t")
_SAM_IMAGE_SIZE = 1024      # SAM's fixed input resolution


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class MobileSAMLoRA(nn.Module):
    """
    MobileSAM with LoRA fine-tuning for nuclei instance segmentation.

    Architecture overview:
        Image (H×W×3)
            ↓  ResizeLongestSide(1024) + pad + normalise
        (1, 3, 1024, 1024)
            ↓  TinyViT image encoder  [LoRA applied to attn qkv + proj]
        image_embeddings (1, 256, 64, 64)
            ↓  PromptEncoder (point prompts)
        sparse_embeddings, dense_embeddings
            ↓  MaskDecoder (TwoWayTransformer) [LoRA applied to cross-attn]
        low_res_masks (1, 1, 256, 256)   iou_predictions (1, 1)
            ↓  postprocess_masks (bilinear upsample)
        masks (1, 1, H, W)

    LoRA targets:
        Image encoder (TinyViT attention): qkv, proj
        Mask decoder  (cross-attention)  : q_proj, k_proj, v_proj, out_proj
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        if not _MOBILE_SAM_AVAILABLE:
            raise ImportError(
                "mobile_sam is required. Install with: pip install mobile-sam"
            )

        model_cfg  = cfg["model"]
        checkpoint = Path(model_cfg["checkpoint"])
        rank       = int(model_cfg["lora_rank"])
        alpha      = float(model_cfg["lora_alpha"])
        dropout    = float(model_cfg.get("lora_dropout", 0.0))
        targets    = list(model_cfg["lora_targets"])

        # ------------------------------------------------------------------
        # 1. Load pretrained MobileSAM
        # ------------------------------------------------------------------
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found at: {checkpoint.resolve()}\n"
                "Download from https://github.com/ChaoningZhang/MobileSAM "
                "and place the file at the path above."
            )

        logger.info("Loading MobileSAM (%s) from %s", _SAM_MODEL_TYPE, checkpoint)
        self.sam = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=str(checkpoint))
        logger.info("MobileSAM loaded.")

        # ------------------------------------------------------------------
        # 2. Inject LoRA
        # ------------------------------------------------------------------
        n_replaced = inject_lora(
            model          = self.sam,
            rank           = rank,
            alpha          = alpha,
            dropout        = dropout,
            target_modules = targets,
        )

        if n_replaced == 0:
            raise RuntimeError(
                "inject_lora replaced 0 layers. "
                "Check that lora_targets in the config match actual layer names. "
                "Run with log_level=DEBUG to see all module names."
            )

        # ------------------------------------------------------------------
        # 3. Freeze image encoder (non-LoRA only).
        #    Mask decoder + prompt encoder remain fully trainable so they
        #    can adapt freely — not just through LoRA branches.
        # ------------------------------------------------------------------
        freeze_image_encoder_non_lora(self.sam)

        # ------------------------------------------------------------------
        # 4. Log parameter breakdown (always at INFO — needed for the report)
        # ------------------------------------------------------------------
        log_parameter_summary(self.sam, logger)

        # ------------------------------------------------------------------
        # 5. Preprocessing transform (shared between train and eval)
        # ------------------------------------------------------------------
        self.resize_transform = ResizeLongestSide(_SAM_IMAGE_SIZE)

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------

    def preprocess_image(
        self,
        image_np: np.ndarray,
    ) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
        """
        Prepare a single numpy image for MobileSAM.

        Steps:
          1. ResizeLongestSide(1024) — scales image so longest side = 1024.
          2. Convert to float32 tensor (C, H', W').
          3. model.preprocess() — normalise (SAM pixel mean/std) + zero-pad
             shorter side to reach (3, 1024, 1024).

        Parameters
        ----------
        image_np : (H, W, 3) uint8 RGB numpy array.

        Returns
        -------
        image_tensor  : (1, 3, 1024, 1024) float32 tensor (on model device).
        original_size : (H, W)  original spatial dimensions (for postprocessing).
        input_size    : (H', W') after ResizeLongestSide (before padding).
        """
        original_size = tuple(image_np.shape[:2])          # (H, W)

        # Resize
        image_resized = self.resize_transform.apply_image(image_np)   # (H', W', 3)
        input_size    = tuple(image_resized.shape[:2])                 # (H', W')

        # To tensor → move to model device → preprocess (normalise + pad)
        # pixel_mean and pixel_std are registered buffers on the model's device,
        # so image_tensor must be on the same device before calling preprocess().
        model_device = next(self.sam.parameters()).device
        image_tensor = (
            torch.as_tensor(image_resized, dtype=torch.float32)
            .permute(2, 0, 1)        # (3, H', W')
            .unsqueeze(0)            # (1, 3, H', W')
            .to(model_device)
        )
        image_tensor = self.sam.preprocess(image_tensor)               # (1, 3, 1024, 1024)

        logger.debug(
            "preprocess_image | original=%s → resized=%s → tensor=%s",
            original_size, input_size, tuple(image_tensor.shape),
        )

        return image_tensor, original_size, input_size

    def encode_image(
        self,
        image_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the image encoder and return (image_embeddings, image_pe).

        Call ONCE per image and cache the result.  Pass both outputs to
        decode_masks() for every nucleus prompt — avoids re-running the
        encoder (the expensive part) for each nucleus.

        Do NOT wrap this in torch.no_grad() during training — gradients
        must flow back through the encoder to reach the LoRA weights.

        Parameters
        ----------
        image_tensor : (1, 3, 1024, 1024) float32 — preprocessed image.

        Returns
        -------
        image_embeddings : (1, 256, 64, 64)  encoder output.
        image_pe         : (1, 256, 64, 64)  dense positional encoding
                           from the prompt encoder (constant per image size).
        """
        image_embeddings = self.sam.image_encoder(image_tensor)
        image_pe         = self.sam.prompt_encoder.get_dense_pe()
        return image_embeddings, image_pe

    def decode_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe:         torch.Tensor,
        point_coords:     torch.Tensor,
        point_labels:     torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run prompt encoder + mask decoder using a cached image embedding.

        Call once per nucleus after encode_image() has been called once for
        the containing image.  This is the inner loop of the training step.

        Parameters
        ----------
        image_embeddings : (1, 256, 64, 64) — from encode_image().
        image_pe         : (1, 256, 64, 64) — from encode_image().
        point_coords     : (1, N, 2) float32 — scaled point prompts.
        point_labels     : (1, N)   int64    — 1=positive, 0=negative.

        Returns
        -------
        low_res_masks   : (1, 1, 256, 256) float32 — raw decoder logits.
        iou_predictions : (1, 1)           float32 — predicted mask IoU.
        """
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings         = image_embeddings,
            image_pe                 = image_pe,
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings  = dense_embeddings,
            multimask_output         = False,
        )
        return low_res_masks, iou_predictions

    def scale_coords(
        self,
        coords_xy:     np.ndarray,
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Scale (x, y) point coordinates from the original image space to
        SAM's resized image space.

        SAM uses (x, y) = (col, row) coordinate convention.
        ResizeLongestSide.apply_coords handles the scale factor correctly.

        Parameters
        ----------
        coords_xy     : (N, 2) float/int array in (x, y) = (col, row) format.
        original_size : (H, W) of the original image.

        Returns
        -------
        (N, 2) float64 array in SAM input space.
        """
        return self.resize_transform.apply_coords(
            coords_xy.astype(np.float32), original_size
        )

    # -----------------------------------------------------------------------
    # Forward pass (training)
    # -----------------------------------------------------------------------

    def forward(
        self,
        image_tensor:  torch.Tensor,
        point_coords:  torch.Tensor,
        point_labels:  torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-image forward pass for training.

        IMPORTANT — SAM's mask decoder is NOT designed for true multi-image
        batches.  Its internal repeat_interleave doubles the batch dimension
        when N_prompts > 1, causing shape mismatches.  Always call with
        batch size = 1 (one image at a time).  To train on a batch of B
        images, call this method B times and accumulate gradients.

        Parameters
        ----------
        image_tensor  : (1, 3, 1024, 1024) float32 — single preprocessed image.
        point_coords  : (1, N, 2) float32 — point prompts in SAM input space.
                        SAM convention: (x, y) = (col, row).
        point_labels  : (1, N) int64 — 1 = positive, 0 = negative.

        Returns
        -------
        low_res_masks    : (1, 1, 256, 256) float32 — raw logits from decoder.
        iou_predictions  : (1, 1) float32 — predicted mask IoU scores.
        """
        assert image_tensor.shape[0] == 1, (
            "MobileSAM forward() expects batch_size=1. "
            "Loop over images and accumulate gradients in the training loop."
        )

        # --- Image encoding -----------------------------------------------
        image_embeddings = self.sam.image_encoder(image_tensor)
        logger.debug("image_embeddings shape: %s", tuple(image_embeddings.shape))

        # --- Prompt encoding ----------------------------------------------
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=None,
        )

        # --- Mask decoding ------------------------------------------------
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings         = image_embeddings,
            image_pe                 = self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings  = dense_embeddings,
            multimask_output         = False,
        )

        logger.debug(
            "low_res_masks=%s  iou_predictions=%s",
            tuple(low_res_masks.shape), tuple(iou_predictions.shape),
        )

        return low_res_masks, iou_predictions

    # -----------------------------------------------------------------------
    # Postprocessing
    # -----------------------------------------------------------------------

    def postprocess_masks(
        self,
        low_res_masks: torch.Tensor,
        input_size:    Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Upsample low-resolution mask logits back to the original image size.

        Parameters
        ----------
        low_res_masks : (B, 1, 256, 256) — raw decoder output.
        input_size    : (H', W') after ResizeLongestSide (before padding).
        original_size : (H, W)   of the original input image.

        Returns
        -------
        (B, 1, H, W) float32 tensor — mask logits at original resolution.
        """
        return self.sam.postprocess_masks(low_res_masks, input_size, original_size)

    # -----------------------------------------------------------------------
    # Checkpoint utilities (LoRA-only saves)
    # -----------------------------------------------------------------------

    def save_lora_weights(self, path: str) -> None:
        """
        Save all trainable parameters (LoRA layers + Mask Decoder + Prompt Encoder).

        Storing these separately keeps checkpoint files small while ensuring that
        all learned adaptation (both in the encoder and the heads) is preserved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all parameters that were being optimized
        trainable_state = {
            k: v
            for k, v in self.sam.state_dict().items()
            if any(substring in k for substring in ["lora_A", "lora_B", "mask_decoder", "prompt_encoder"])
        }

        torch.save(trainable_state, str(path))
        logger.info(
            "Trainable weights saved → %s  (%d tensors, %.2f MB)",
            path,
            len(trainable_state),
            sum(v.numel() * v.element_size() for v in trainable_state.values()) / 1e6,
        )

    def load_lora_weights(self, path: str) -> None:
        """
        Load previously saved trainable weights into the current model.

        Uses strict=False so that frozen image encoder weights are not
        overwritten or complained about.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Trainable weights checkpoint not found: {path}")

        trainable_state = torch.load(str(path), map_location="cpu")
        missing, unexpected = self.sam.load_state_dict(trainable_state, strict=False)

        logger.info(
            "Trainable weights loaded ← %s | tensors=%d | missing=%d | unexpected=%d",
            path, len(trainable_state), len(missing), len(unexpected),
        )

        if unexpected:
            logger.warning("Unexpected keys when loading trainable weights: %s", unexpected)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(cfg: dict, device: torch.device) -> MobileSAMLoRA:
    """
    Build and return a MobileSAMLoRA model on the specified device.

    Parameters
    ----------
    cfg    : Full config dict.
    device : torch.device to move the model to.

    Returns
    -------
    MobileSAMLoRA  (on device, LoRA injected, non-LoRA frozen)
    """
    logger.info("Building MobileSAMLoRA model on device: %s", device)
    model = MobileSAMLoRA(cfg)
    model = model.to(device)
    logger.info("Model moved to %s.", device)
    return model


# ---------------------------------------------------------------------------
# Point prompt sampling utility
# ---------------------------------------------------------------------------

def sample_point_prompts(
    instance_mask: np.ndarray,
    num_pos:       int = 1,
    num_neg:       int = 1,
    rng:           Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Sample positive and negative point prompts from a NuInsSeg instance mask.

    Strategy:
      - Randomly select one nucleus instance from the mask.
      - For each positive point, use the centroid of the selected instance.
        (Centroid is a stable, unambiguous representative point.)
      - For each negative point, sample a random background pixel.

    Parameters
    ----------
    instance_mask : (H, W) uint16  — per-nucleus instance IDs (0 = background).
    num_pos       : Number of positive points to sample.
    num_neg       : Number of negative points to sample.
    rng           : Optional numpy random Generator for reproducibility.
                    If None, uses np.random.default_rng() (non-deterministic).

    Returns
    -------
    point_coords    : (num_pos + num_neg, 2) float32  — (x, y) = (col, row) format.
    point_labels    : (num_pos + num_neg,)  int64     — 1=positive, 0=negative.
    instance_binary : (H, W) bool  — binary mask of the selected nucleus.
    instance_id     : int  — the selected instance ID (for debugging).

    Raises
    ------
    ValueError  If the mask contains no nuclei or no background pixels.
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- Identify instances -----------------------------------------------
    unique_ids = np.unique(instance_mask)
    nucleus_ids = unique_ids[unique_ids > 0]     # Exclude background (0)

    if nucleus_ids.size == 0:
        raise ValueError(
            "instance_mask contains no nuclei (all pixels are background). "
            "Check that the mask file was loaded correctly (uint16, not clipped)."
        )

    # --- Select one nucleus randomly --------------------------------------
    instance_id = int(rng.choice(nucleus_ids))
    instance_binary = (instance_mask == instance_id)    # (H, W) bool

    # --- Positive points (centroid of selected nucleus) -------------------
    rows, cols = np.where(instance_binary)
    centroid_y = rows.mean()    # row  → y in (x,y) convention
    centroid_x = cols.mean()    # col  → x in (x,y) convention

    pos_coords = np.array(
        [[centroid_x, centroid_y]] * num_pos,
        dtype=np.float32,
    )                                                   # (num_pos, 2)

    # --- Negative points (random background pixels) -----------------------
    bg_rows, bg_cols = np.where(instance_mask == 0)

    if bg_rows.size == 0:
        raise ValueError(
            "instance_mask has no background pixels — cannot sample negative points."
        )

    neg_indices = rng.choice(len(bg_rows), size=num_neg, replace=(len(bg_rows) < num_neg))
    neg_coords  = np.stack(
        [bg_cols[neg_indices].astype(np.float32),    # x = col
         bg_rows[neg_indices].astype(np.float32)],   # y = row
        axis=1,
    )                                                   # (num_neg, 2)

    # --- Combine ----------------------------------------------------------
    point_coords = np.concatenate([pos_coords, neg_coords], axis=0)   # (N, 2)
    point_labels = np.array(
        [1] * num_pos + [0] * num_neg, dtype=np.int64
    )                                                                   # (N,)

    logger.debug(
        "sample_point_prompts | instance_id=%d | centroid=(%.1f, %.1f) "
        "| n_pixels=%d | pos=%d | neg=%d",
        instance_id, centroid_x, centroid_y, rows.size, num_pos, num_neg,
    )

    return point_coords, point_labels, instance_binary, instance_id
