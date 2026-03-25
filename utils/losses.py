"""
utils/losses.py

Loss functions for MobileSAM-LoRA fine-tuning.

Combined BCE + Dice loss is standard for binary segmentation tasks and
works well with SAM's output logits.

An optional IOU prediction loss (MSE between predicted and actual IOU) can
be added to supervise SAM's IOU head — this encourages the model to produce
well-calibrated confidence scores at inference time.

Public API:
    BinaryDiceLoss      -- Dice loss alone
    BCEDiceLoss         -- BCE + Dice (primary training loss)
    iou_prediction_loss -- MSE on the SAM IOU head output
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dice loss
# ---------------------------------------------------------------------------

class BinaryDiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.

        Dice = 2 * |P ∩ G| / (|P| + |G|)
        Loss = 1 - Dice

    Expects raw logits as input (sigmoid is applied internally).

    Parameters
    ----------
    smooth : Small constant to avoid division by zero (default 1e-6).
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) raw model output (before sigmoid).
        targets : (B, 1, H, W) binary float ground truth {0.0, 1.0}.

        Returns
        -------
        Scalar Dice loss.
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dims for per-sample Dice, then average over batch
        probs   = probs.view(probs.shape[0], -1)       # (B, H*W)
        targets = targets.view(targets.shape[0], -1)   # (B, H*W)

        intersection = (probs * targets).sum(dim=1)    # (B,)
        union        = probs.sum(dim=1) + targets.sum(dim=1)  # (B,)

        dice_per_sample = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice_per_sample.mean()

        return loss


# ---------------------------------------------------------------------------
# Combined BCE + Dice loss (primary loss)
# ---------------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Soft Dice loss.

        L = bce_weight * BCE(logits, targets)
          + dice_weight * Dice(logits, targets)

    BCE is well-suited for per-pixel classification and handles class
    imbalance reasonably.  Dice directly optimises the overlap metric and is
    robust to foreground/background imbalance — important for sparse nuclei
    fields where background pixels dominate.

    Parameters
    ----------
    bce_weight  : Weight for BCE component  (default 1.0).
    dice_weight : Weight for Dice component (default 1.0).
    smooth      : Dice smoothing constant   (default 1e-6).
    """

    def __init__(
        self,
        bce_weight:  float = 1.0,
        dice_weight: float = 1.0,
        smooth:      float = 1e-6,
    ) -> None:
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss   = BinaryDiceLoss(smooth=smooth)

        logger.debug(
            "BCEDiceLoss | bce_weight=%.2f | dice_weight=%.2f",
            bce_weight, dice_weight,
        )

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W) raw model output.
        targets : (B, 1, H, W) binary float ground truth {0.0, 1.0}.

        Returns
        -------
        total_loss : scalar tensor (backpropagable).
        bce        : scalar tensor (for logging, detached from graph).
        dice       : scalar tensor (for logging, detached from graph).
        """
        bce  = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = self.bce_weight * bce + self.dice_weight * dice

        return total, bce.detach(), dice.detach()


# ---------------------------------------------------------------------------
# IOU prediction loss (auxiliary — supervises SAM's IOU head)
# ---------------------------------------------------------------------------

def iou_prediction_loss(
    iou_predictions: torch.Tensor,
    logits:          torch.Tensor,
    targets:         torch.Tensor,
    threshold:       float = 0.5,
) -> torch.Tensor:
    """
    MSE loss between SAM's predicted IOU scores and the actual IOU
    between the thresholded predicted mask and the ground truth.

    This is an auxiliary loss that encourages the SAM IOU head to output
    well-calibrated confidence scores, which are used at inference time
    to filter low-quality masks.

    Parameters
    ----------
    iou_predictions : (B, 1) IOU scores from SAM's mask decoder.
    logits          : (B, 1, H, W) raw mask logits.
    targets         : (B, 1, H, W) binary float ground truth.
    threshold       : Sigmoid threshold for binarising predicted masks (default 0.5).

    Returns
    -------
    Scalar MSE loss.
    """
    with torch.no_grad():
        pred_binary = (torch.sigmoid(logits) > threshold).float()
        # Compute actual IOU per sample
        intersection = (pred_binary * targets).sum(dim=(-2, -1))       # (B, 1)
        union        = (pred_binary + targets).clamp(0, 1).sum(dim=(-2, -1))  # (B, 1)
        actual_iou   = intersection / (union + 1e-6)                   # (B, 1)

    return F.mse_loss(iou_predictions, actual_iou)
