"""
models/lora.py

LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

Reference:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
    https://arxiv.org/abs/2106.09685

Public API:
    LoRALinear              -- nn.Linear wrapper with a low-rank branch
    inject_lora(...)        -- walk a model and replace matching Linear layers
    freeze_non_lora(model)  -- freeze everything except lora_A / lora_B
    count_parameters(model) -- return {total, trainable, frozen} counts
"""

import logging
import math
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds a low-rank adaptation branch.

    Forward computation:
        h = W x  +  (B A x) * (alpha / rank)

    where:
        W    : original frozen weight  (in_features → out_features)
        A    : (rank, in_features)     initialised with Kaiming uniform
        B    : (out_features, rank)    initialised to zero

    B = 0 at initialisation guarantees that the LoRA branch contributes
    nothing at the start of training.  The model therefore begins from the
    pretrained checkpoint, and the adaptation is learned from there.

    Parameters
    ----------
    linear  : The original nn.Linear to wrap (and freeze).
    rank    : Low-rank dimension r (paper typically uses 4 or 8).
    alpha   : Scaling factor; effective scale = alpha / rank.
    dropout : Dropout applied to the input of the LoRA branch (0 = off).
    """

    def __init__(
        self,
        linear:  nn.Linear,
        rank:    int,
        alpha:   float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}.")
        if rank > min(linear.in_features, linear.out_features):
            raise ValueError(
                f"LoRA rank ({rank}) must not exceed "
                f"min(in_features={linear.in_features}, "
                f"out_features={linear.out_features})."
            )

        self.linear  = linear           # Original frozen weights live here
        self.rank    = rank
        self.scaling = alpha / rank     # Pre-computed; avoids division every forward

        # LoRA parameters — only these will require gradients
        self.lora_A = nn.Parameter(torch.empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, rank))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Initialise A with Kaiming uniform (same default as nn.Linear weight init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero — ensures LoRA is a no-op at the start of training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base path (frozen pretrained weights)
        base = self.linear(x)
        # LoRA path: x → dropout → A^T → B^T → scale
        lora = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora

    def extra_repr(self) -> str:
        return (
            f"in={self.linear.in_features}, out={self.linear.out_features}, "
            f"rank={self.rank}, scaling={self.scaling:.4f}"
        )


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_lora(
    model:          nn.Module,
    rank:           int,
    alpha:          float,
    dropout:        float,
    target_modules: List[str],
) -> int:
    """
    Walk *model* and replace every nn.Linear whose local attribute name
    (i.e. the last component of its dotted path) is listed in
    *target_modules* with a LoRALinear wrapper.

    Replacement is done in two passes to avoid mutating the module tree
    while iterating over it.

    Parameters
    ----------
    model          : The model to modify in-place.
    rank           : LoRA rank r.
    alpha          : LoRA scaling factor.
    dropout        : Dropout probability for the LoRA branch.
    target_modules : List of local attribute names to target.
                     E.g. ["qkv", "proj", "q_proj", "k_proj", "v_proj", "out_proj"]

    Returns
    -------
    int  Number of Linear layers replaced.
    """
    if not target_modules:
        logger.warning("inject_lora called with empty target_modules — nothing replaced.")
        return 0

    # --- Pass 1: collect targets (avoid mutating during iteration) ----------
    replacements = []  # (parent_module, attr_name, original_linear, full_name)

    for full_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # local name = last dotted component
        local_name = full_name.split(".")[-1]
        if local_name not in target_modules:
            continue

        # Navigate to the parent module
        if "." in full_name:
            parent_path, _ = full_name.rsplit(".", 1)
            parent = _get_submodule(model, parent_path)
        else:
            parent = model

        replacements.append((parent, local_name, module, full_name))

    if not replacements:
        logger.warning(
            "inject_lora: no Linear layers matched target_modules=%s. "
            "Check that the model is loaded before calling inject_lora, "
            "and that the layer names are correct.",
            target_modules,
        )
        return 0

    # --- Pass 2: replace ---------------------------------------------------
    for parent, attr, original_linear, full_name in replacements:
        lora_layer = LoRALinear(
            linear  = original_linear,
            rank    = rank,
            alpha   = alpha,
            dropout = dropout,
        )
        setattr(parent, attr, lora_layer)
        logger.debug(
            "LoRA injected | %-55s  in=%-4d out=%-4d rank=%d",
            full_name, original_linear.in_features,
            original_linear.out_features, rank,
        )

    logger.info(
        "LoRA injection complete: %d layers replaced | rank=%d | alpha=%s | dropout=%s",
        len(replacements), rank, alpha, dropout,
    )
    return len(replacements)


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------

def freeze_non_lora(model: nn.Module) -> None:
    """
    Freeze all model parameters **except** lora_A and lora_B matrices.

    Must be called *after* inject_lora() so that LoRALinear wrappers are
    already in place.

    After this call:
        param.requires_grad == True   ←→   "lora_A" or "lora_B" in param name
    """
    n_trainable = 0
    n_frozen    = 0

    for name, param in model.named_parameters():
        is_lora = ("lora_A" in name) or ("lora_B" in name)
        param.requires_grad_(is_lora)
        if is_lora:
            n_trainable += param.numel()
        else:
            n_frozen += param.numel()

    logger.debug(
        "freeze_non_lora: %d trainable params | %d frozen params",
        n_trainable, n_frozen,
    )


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> dict:
    """
    Return a breakdown of parameter counts for a model.

    Returns
    -------
    dict with keys:
        "total"     : int  All parameters (trainable + frozen)
        "trainable" : int  Parameters with requires_grad=True  (LoRA only after freeze)
        "frozen"    : int  Parameters with requires_grad=False
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    return {"total": total, "trainable": trainable, "frozen": frozen}


def log_parameter_summary(model: nn.Module, logger_: logging.Logger) -> None:
    """
    Log a human-readable parameter summary table.

    Logs each LoRA layer with its (in, out, rank) dimensions, then prints
    the total / trainable / frozen counts.
    """
    counts = count_parameters(model)

    logger_.info("=" * 60)
    logger_.info("PARAMETER SUMMARY")
    logger_.info("=" * 60)
    logger_.info("  %-50s  %s", "Layer (LoRA only)", "# params")
    logger_.info("  " + "-" * 58)

    lora_total = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            p = module.lora_A.numel() + module.lora_B.numel()
            lora_total += p
            logger_.info("  %-50s  %d", name, p)

    logger_.info("  " + "-" * 58)
    logger_.info("  LoRA trainable params : %12s", f"{counts['trainable']:,}")
    logger_.info("  Frozen params         : %12s", f"{counts['frozen']:,}")
    logger_.info("  Total params          : %12s", f"{counts['total']:,}")
    logger_.info(
        "  Trainable ratio       : %11.4f%%",
        100.0 * counts["trainable"] / max(counts["total"], 1),
    )
    logger_.info("=" * 60)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_submodule(model: nn.Module, dotted_path: str) -> nn.Module:
    """
    Navigate to a submodule by its dotted attribute path.
    An empty path returns the model itself.
    """
    if not dotted_path:
        return model
    module = model
    for part in dotted_path.split("."):
        module = getattr(module, part)
    return module
