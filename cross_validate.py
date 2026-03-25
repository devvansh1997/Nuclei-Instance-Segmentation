"""
cross_validate.py

Run full 5-fold cross-validation: train + evaluate each fold, then print
an aggregated results table matching Table 3 of the NuInsSeg paper.

Usage:
    # Full 5-fold run
    python cross_validate.py --config configs/debug.yaml
    python cross_validate.py --config configs/train_a100.yaml

    # Run specific folds only (useful for resuming or parallelising on HPC)
    python cross_validate.py --config configs/train_a100.yaml --folds 0,1,2

    # Skip training (evaluate already-trained checkpoints)
    python cross_validate.py --config configs/train_a100.yaml --skip-train

    # Skip evaluation (train only)
    python cross_validate.py --config configs/train_a100.yaml --skip-eval

    # Auto-resume: skip folds whose best_lora.pt already exists
    python cross_validate.py --config configs/train_a100.yaml --resume
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import yaml

from train    import train_fold
from evaluate import evaluate_fold
from utils.logger import setup_logger, log_system_info, log_config

logger = logging.getLogger(__name__)

_METRICS = ["dice", "aji", "pq", "sq", "rq"]


# ---------------------------------------------------------------------------
# Resume check
# ---------------------------------------------------------------------------

def _checkpoint_exists(cfg: dict, fold_idx: int) -> bool:
    """Return True if best_lora.pt exists for this fold."""
    ckpt = (
        Path(cfg["output"]["save_dir"])
        / f"fold_{fold_idx}"
        / "checkpoints"
        / "best_lora.pt"
    )
    return ckpt.exists()


def _fold_eval_exists(cfg: dict, fold_idx: int) -> bool:
    """Return True if fold_metrics.json already exists for this fold."""
    path = (
        Path(cfg["output"]["save_dir"])
        / f"fold_{fold_idx}"
        / "fold_metrics.json"
    )
    return path.exists()


def _load_fold_metrics(cfg: dict, fold_idx: int) -> dict:
    """Load previously saved fold_metrics.json."""
    path = (
        Path(cfg["output"]["save_dir"])
        / f"fold_{fold_idx}"
        / "fold_metrics.json"
    )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def _print_results_table(fold_results: list) -> None:
    """
    Print a results table matching the style of NuInsSeg Table 3.

    fold_results : list of dicts, each with keys fold + _METRICS
    """
    header_metrics = ["Dice", "AJI", "PQ", "SQ", "RQ"]
    col_w = 8

    sep  = "  ".join(["-" * 4] + ["-" * col_w] * len(header_metrics))
    head = "  ".join(
        [f"{'Fold':<4}"] + [f"{m:>{col_w}}" for m in header_metrics]
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("5-FOLD CROSS-VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(head)
    logger.info(sep)

    all_vals = {m: [] for m in _METRICS}

    for row in sorted(fold_results, key=lambda r: r["fold"]):
        vals = "  ".join(
            f"{row[m]:>{col_w}.4f}" for m in _METRICS
        )
        logger.info("%-4d  %s", row["fold"], vals)
        for m in _METRICS:
            all_vals[m].append(row[m])

    logger.info(sep)

    means = {m: float(np.mean(all_vals[m])) for m in _METRICS}
    stds  = {m: float(np.std(all_vals[m]))  for m in _METRICS}

    mean_str = "  ".join(f"{means[m]:>{col_w}.4f}" for m in _METRICS)
    std_str  = "  ".join(f"{stds[m]:>{col_w}.4f}"  for m in _METRICS)

    logger.info("%-4s  %s", "Mean", mean_str)
    logger.info("%-4s  %s", "Std",  std_str)
    logger.info("=" * 60)
    logger.info("")

    # Also log a compact one-liner for easy copy-paste into the report
    logger.info(
        "Summary: Dice=%.4f±%.4f  AJI=%.4f±%.4f  PQ=%.4f±%.4f",
        means["dice"], stds["dice"],
        means["aji"],  stds["aji"],
        means["pq"],   stds["pq"],
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_cross_validation(
    cfg:        dict,
    folds:      list,
    skip_train: bool,
    skip_eval:  bool,
    resume:     bool,
) -> dict:
    """
    Orchestrate train + evaluate for each requested fold.

    Parameters
    ----------
    cfg        : Full config dict.
    folds      : List of fold indices to run.
    skip_train : If True, skip training (use existing checkpoints).
    skip_eval  : If True, skip evaluation (train only).
    resume     : If True, skip folds whose checkpoint already exists.

    Returns
    -------
    dict with keys:
        "fold_results" : list of per-fold metric dicts
        "mean"         : dict of mean metrics
        "std"          : dict of std metrics
    """
    n_folds = cfg["data"].get("n_folds", 5)

    # Validate requested folds
    for f in folds:
        if f < 0 or f >= n_folds:
            raise ValueError(
                f"Fold index {f} out of range [0, {n_folds - 1}]."
            )

    fold_results = []
    train_times  = []
    eval_times   = []

    logger.info(
        "Cross-validation | folds=%s | skip_train=%s | skip_eval=%s | resume=%s",
        folds, skip_train, skip_eval, resume,
    )

    for fold_idx in folds:
        logger.info("")
        logger.info("━" * 60)
        logger.info("FOLD %d / %d", fold_idx, n_folds - 1)
        logger.info("━" * 60)

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        if skip_train:
            logger.info("Fold %d: training skipped (--skip-train).", fold_idx)

        elif resume and _checkpoint_exists(cfg, fold_idx):
            logger.info(
                "Fold %d: checkpoint already exists — skipping training (--resume).",
                fold_idx,
            )

        else:
            logger.info("Fold %d: starting training...", fold_idx)
            t0 = time.time()
            train_result = train_fold(cfg, fold_idx)
            elapsed = time.time() - t0
            train_times.append(elapsed)
            logger.info(
                "Fold %d: training done in %.1f min | best_val_dice=%.4f @ epoch %d",
                fold_idx, elapsed / 60,
                train_result["best_val_dice"], train_result["best_epoch"],
            )

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if skip_eval:
            logger.info("Fold %d: evaluation skipped (--skip-eval).", fold_idx)
            continue

        elif resume and _fold_eval_exists(cfg, fold_idx):
            logger.info(
                "Fold %d: evaluation results already exist — loading (--resume).",
                fold_idx,
            )
            fold_result = _load_fold_metrics(cfg, fold_idx)

        else:
            logger.info("Fold %d: starting evaluation...", fold_idx)
            t0 = time.time()
            fold_result = evaluate_fold(cfg, fold_idx)
            elapsed = time.time() - t0
            eval_times.append(elapsed)
            logger.info(
                "Fold %d: evaluation done in %.1f min",
                fold_idx, elapsed / 60,
            )

        fold_results.append(fold_result)

        # Log fold result immediately so partial results are visible
        logger.info(
            "Fold %d | Dice=%.4f  AJI=%.4f  PQ=%.4f",
            fold_idx,
            fold_result["dice"], fold_result["aji"], fold_result["pq"],
        )

    if not fold_results:
        logger.warning("No evaluation results collected — nothing to aggregate.")
        return {}

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    means = {m: float(np.mean([r[m] for r in fold_results])) for m in _METRICS}
    stds  = {m: float(np.std( [r[m] for r in fold_results])) for m in _METRICS}

    _print_results_table(fold_results)

    # -----------------------------------------------------------------------
    # Timing summary
    # -----------------------------------------------------------------------
    if train_times:
        logger.info(
            "Training time  | total=%.1f min | per_fold=%.1f min (avg)",
            sum(train_times) / 60, np.mean(train_times) / 60,
        )
    if eval_times:
        logger.info(
            "Eval time      | total=%.1f min | per_fold=%.1f min (avg)",
            sum(eval_times) / 60, np.mean(eval_times) / 60,
        )

    # -----------------------------------------------------------------------
    # Save final aggregated results
    # -----------------------------------------------------------------------
    final = {
        "config":       str(Path(cfg.get("_config_path", "unknown")).resolve()),
        "folds_run":    folds,
        "fold_results": fold_results,
        "mean":         means,
        "std":          stds,
    }

    save_dir  = Path(cfg["output"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path  = save_dir / "cv_results.json"

    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    logger.info("Final CV results saved → %s", out_path)

    return final


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="5-fold cross-validation for MobileSAM+LoRA on NuInsSeg."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--folds", default=None,
        help="Comma-separated fold indices to run (default: all). E.g. --folds 0,1,2",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training — evaluate existing checkpoints only.",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip evaluation — train only.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip folds whose checkpoint / eval results already exist.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = args.config   # Store for results metadata

    n_folds = cfg["data"].get("n_folds", 5)
    if args.folds:
        folds = [int(f.strip()) for f in args.folds.split(",")]
    else:
        folds = list(range(n_folds))

    # Shared logger for the orchestrator itself
    run_logger = setup_logger(cfg, "cross_validate")
    log_system_info(run_logger)
    log_config(run_logger, cfg)

    run_cross_validation(
        cfg        = cfg,
        folds      = folds,
        skip_train = args.skip_train,
        skip_eval  = args.skip_eval,
        resume     = args.resume,
    )


if __name__ == "__main__":
    main()
