"""
utils/metrics.py

Evaluation metrics for nuclei instance segmentation.

Follows the evaluation protocol in Table 3 of the NuInsSeg paper:
    Mahbod et al., "NuInsSeg: A fully annotated dataset for nuclei instance
    segmentation in H&E-stained histological images." Scientific Data, 2024.

Implemented metrics:
    Dice  -- binary overlap (foreground nuclei vs background)
    AJI   -- Aggregated Jaccard Index (instance-level overlap)
    PQ    -- Panoptic Quality = SQ * RQ (instance detection + quality)

All metric functions operate on numpy arrays of instance masks:
    0          → background
    1, 2, 3 …  → individual nucleus instance IDs

Public API:
    dice_coefficient(pred_binary, gt_binary)       → float
    aggregated_jaccard_index(pred_inst, gt_inst)   → float
    panoptic_quality(pred_inst, gt_inst, ...)      → dict {pq, sq, rq, tp, fp, fn}
    compute_all_metrics(pred_inst, gt_inst)        → dict {dice, aji, pq, sq, rq}
    masks_to_instance_map(mask_list)               → np.ndarray (H, W) uint16
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dice coefficient (binary)
# ---------------------------------------------------------------------------

def dice_coefficient(
    pred_binary: np.ndarray,
    gt_binary:   np.ndarray,
) -> float:
    """
    Compute the binary Dice coefficient.

        Dice = 2 * |P ∩ G| / (|P| + |G|)

    Parameters
    ----------
    pred_binary : (H, W) bool or uint8 — predicted foreground mask.
    gt_binary   : (H, W) bool or uint8 — ground-truth foreground mask.

    Returns
    -------
    float in [0, 1].  Returns 1.0 if both masks are empty (no nuclei).
    """
    pred = pred_binary.astype(bool)
    gt   = gt_binary.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    denominator  = pred.sum() + gt.sum()

    if denominator == 0:
        # Both masks empty — perfect score by convention
        return 1.0

    return float(2.0 * intersection / denominator)


# ---------------------------------------------------------------------------
# AJI — Aggregated Jaccard Index
# ---------------------------------------------------------------------------

def aggregated_jaccard_index(
    pred_instance_mask: np.ndarray,
    gt_instance_mask:   np.ndarray,
) -> float:
    """
    Compute the Aggregated Jaccard Index (AJI).

    AJI measures instance-level overlap by aggregating Jaccard scores across
    all GT instances, penalising unmatched predictions as false positives.

    Algorithm (from Kumar et al., 2017):
      For each GT instance G_i:
        1. Find the prediction P_j with the maximum intersection with G_i.
        2. If P_j has not been used, add |G_i ∩ P_j| to numerator and
           |G_i ∪ P_j| to denominator. Mark P_j as used.
        3. If no unused P_j overlaps, add |G_i| to denominator.
      For each unused prediction P_k:
        4. Add |P_k| to the denominator (false positive penalty).

    Parameters
    ----------
    pred_instance_mask : (H, W) int/uint16 — predicted instance IDs (0=BG).
    gt_instance_mask   : (H, W) int/uint16 — ground-truth instance IDs (0=BG).

    Returns
    -------
    float in [0, 1].
    """
    gt_ids   = _get_instance_ids(gt_instance_mask)
    pred_ids = _get_instance_ids(pred_instance_mask)

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0:
        return 0.0
    if len(pred_ids) == 0:
        return 0.0

    # Pre-compute areas
    gt_areas   = {i: np.sum(gt_instance_mask == i)   for i in gt_ids}
    pred_areas = {i: np.sum(pred_instance_mask == i) for i in pred_ids}

    # Efficiently find all intersections using a single pass
    # (gt_id, pred_id) -> intersection_area
    intersections = {}
    combined = gt_instance_mask.astype(np.int64) * (pred_instance_mask.max() + 1) + pred_instance_mask
    unique_combined, counts = np.unique(combined, return_counts=True)
    
    max_pred_id = pred_instance_mask.max()
    for val, count in zip(unique_combined, counts):
        gt_id = val // (max_pred_id + 1)
        pred_id = val % (max_pred_id + 1)
        if gt_id > 0 and pred_id > 0:
            intersections[(int(gt_id), int(pred_id))] = int(count)

    numerator   = 0
    denominator = 0
    used_preds  = set()

    # Step 1-3: Match GT to Pred
    for gt_id in gt_ids:
        best_pred_id   = None
        best_intersect = 0
        
        # Find best UNUSED prediction
        for pred_id in pred_ids:
            intersect = intersections.get((gt_id, pred_id), 0)
            if intersect > best_intersect and pred_id not in used_preds:
                best_intersect = intersect
                best_pred_id   = pred_id

        if best_pred_id is not None:
            union = gt_areas[gt_id] + pred_areas[best_pred_id] - best_intersect
            numerator   += best_intersect
            denominator += union
            used_preds.add(best_pred_id)
        else:
            # GT unmatched
            denominator += gt_areas[gt_id]

    # Step 4: Penalty for unused predictions
    for pred_id in pred_ids:
        if pred_id not in used_preds:
            denominator += pred_areas[pred_id]

    if denominator == 0:
        return 1.0

    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# PQ — Panoptic Quality
# ---------------------------------------------------------------------------

def panoptic_quality(
    pred_instance_mask: np.ndarray,
    gt_instance_mask:   np.ndarray,
    iou_threshold:      float = 0.5,
) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) and its components SQ and RQ.

        PQ = SQ × RQ
        SQ = mean IoU of matched (TP) pairs
        RQ = F1  = 2·TP / (2·TP + FP + FN)

    Matching is greedy: pairs are accepted if IoU > iou_threshold (default 0.5).
    Each GT and prediction instance can be matched at most once.

    Parameters
    ----------
    pred_instance_mask : (H, W) int/uint16 — predicted instance IDs (0=BG).
    gt_instance_mask   : (H, W) int/uint16 — ground-truth instance IDs (0=BG).
    iou_threshold      : Minimum IoU to count a pair as TP (default 0.5).

    Returns
    -------
    dict with keys:
        "pq"  float  Panoptic Quality
        "sq"  float  Segmentation Quality  (mean IoU of TPs)
        "rq"  float  Recognition Quality   (F1 score)
        "tp"  int    True positives
        "fp"  int    False positives
        "fn"  int    False negatives
    """
    gt_ids   = _get_instance_ids(gt_instance_mask)
    pred_ids = _get_instance_ids(pred_instance_mask)

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return dict(pq=1.0, sq=1.0, rq=1.0, tp=0, fp=0, fn=0)
    if len(gt_ids) == 0:
        return dict(pq=0.0, sq=0.0, rq=0.0, tp=0, fp=len(pred_ids), fn=0)
    if len(pred_ids) == 0:
        return dict(pq=0.0, sq=0.0, rq=0.0, tp=0, fp=0, fn=len(gt_ids))

    # Pre-compute areas
    gt_areas   = {i: int(np.sum(gt_instance_mask == i))   for i in gt_ids}
    pred_areas = {i: int(np.sum(pred_instance_mask == i)) for i in pred_ids}

    # Compute pairwise IoU for all (GT, pred) pairs that have any overlap
    # Use the combined mask trick to find overlapping pairs efficiently
    iou_matrix: Dict[tuple, float] = {}

    for gt_id in gt_ids:
        gt_mask = (gt_instance_mask == gt_id)
        for pred_id in pred_ids:
            pred_mask  = (pred_instance_mask == pred_id)
            intersect  = int(np.logical_and(gt_mask, pred_mask).sum())
            if intersect == 0:
                continue
            union = gt_areas[gt_id] + pred_areas[pred_id] - intersect
            iou_matrix[(gt_id, pred_id)] = intersect / union

    # Greedy matching: highest IoU first, each instance used at most once
    matched_pairs = []
    matched_gt    = set()
    matched_pred  = set()

    for (gt_id, pred_id), iou in sorted(
        iou_matrix.items(), key=lambda x: x[1], reverse=True
    ):
        if iou < iou_threshold:
            break   # All remaining pairs have lower IoU — sorted, so we can stop
        if gt_id in matched_gt or pred_id in matched_pred:
            continue
        matched_pairs.append((gt_id, pred_id, iou))
        matched_gt.add(gt_id)
        matched_pred.add(pred_id)

    tp = len(matched_pairs)
    fp = len(pred_ids) - len(matched_pred)
    fn = len(gt_ids)  - len(matched_gt)

    sq  = float(np.mean([iou for _, _, iou in matched_pairs])) if tp > 0 else 0.0
    rq  = float(2 * tp / (2 * tp + fp + fn))                   if (tp + fp + fn) > 0 else 0.0
    pq  = sq * rq

    logger.debug(
        "PQ | gt=%d pred=%d | TP=%d FP=%d FN=%d | SQ=%.4f RQ=%.4f PQ=%.4f",
        len(gt_ids), len(pred_ids), tp, fp, fn, sq, rq, pq,
    )

    return dict(pq=pq, sq=sq, rq=rq, tp=tp, fp=fp, fn=fn)


# ---------------------------------------------------------------------------
# Combined metric computation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred_instance_mask: np.ndarray,
    gt_instance_mask:   np.ndarray,
    iou_threshold:      float = 0.5,
) -> Dict[str, float]:
    """
    Compute Dice, AJI, and PQ for a single image prediction.

    Parameters
    ----------
    pred_instance_mask : (H, W) int/uint16  predicted instance IDs (0=BG).
    gt_instance_mask   : (H, W) int/uint16  ground-truth instance IDs (0=BG).
    iou_threshold      : Matching threshold for PQ (default 0.5).

    Returns
    -------
    dict with keys: "dice", "aji", "pq", "sq", "rq"
    """
    # Binary masks for Dice
    pred_binary = (pred_instance_mask > 0)
    gt_binary   = (gt_instance_mask   > 0)

    dice = dice_coefficient(pred_binary, gt_binary)
    aji  = aggregated_jaccard_index(pred_instance_mask, gt_instance_mask)
    pq_results = panoptic_quality(pred_instance_mask, gt_instance_mask, iou_threshold)

    return {
        "dice": dice,
        "aji":  aji,
        "pq":   pq_results["pq"],
        "sq":   pq_results["sq"],
        "rq":   pq_results["rq"],
    }


def aggregate_metrics(per_image_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Average per-image metric dicts into a single fold-level summary.

    Parameters
    ----------
    per_image_metrics : list of dicts from compute_all_metrics().

    Returns
    -------
    dict with mean of each metric key.
    """
    if not per_image_metrics:
        logger.warning("aggregate_metrics received an empty list.")
        return {}

    keys = per_image_metrics[0].keys()
    return {
        k: float(np.mean([m[k] for m in per_image_metrics]))
        for k in keys
    }


# ---------------------------------------------------------------------------
# Inference helper: convert SAM mask list → instance map
# ---------------------------------------------------------------------------

def masks_to_instance_map(
    mask_list: List[dict],
    min_area:  int = 0,
) -> np.ndarray:
    """
    Convert the output of SAM's SamAutomaticMaskGenerator to a
    (H, W) uint16 instance map compatible with the metric functions.

    SAM returns a list of dicts, each with:
        "segmentation" : (H, W) bool
        "area"         : int
        "predicted_iou": float
        "stability_score": float
        ...

    Masks are assigned instance IDs 1, 2, 3 … in order of decreasing area
    (largest nucleus first, consistent with NuInsSeg convention).
    Overlapping pixels are assigned to whichever instance is painted last
    (smallest area wins — avoids small masks being buried under large ones).

    Parameters
    ----------
    mask_list : Output from SamAutomaticMaskGenerator.generate(image).
    min_area  : Discard masks smaller than this many pixels (default 0).

    Returns
    -------
    (H, W) uint16 instance map.  0 = background.
    """
    if not mask_list:
        # Determine shape from first available mask or return empty
        logger.warning("masks_to_instance_map received an empty mask list.")
        return np.zeros((1, 1), dtype=np.uint16)

    # Filter by area
    filtered = [m for m in mask_list if m["area"] >= min_area]

    if not filtered:
        h = mask_list[0]["segmentation"].shape[0]
        w = mask_list[0]["segmentation"].shape[1]
        return np.zeros((h, w), dtype=np.uint16)

    # Sort descending by area — paint largest first, smallest last
    # (smaller masks on top to prevent being fully occluded)
    filtered.sort(key=lambda m: m["area"], reverse=True)

    h, w = filtered[0]["segmentation"].shape
    instance_map = np.zeros((h, w), dtype=np.uint16)

    for instance_id, mask_dict in enumerate(filtered, start=1):
        seg = mask_dict["segmentation"].astype(bool)
        instance_map[seg] = instance_id

        if instance_id >= 65535:          # uint16 max
            logger.warning(
                "Instance count (%d) exceeds uint16 range — truncating.", instance_id
            )
            break

    logger.debug(
        "masks_to_instance_map | input=%d masks | filtered=%d | unique_ids=%d",
        len(mask_list), len(filtered), instance_map.max(),
    )
    return instance_map


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_instance_ids(instance_mask: np.ndarray) -> List[int]:
    """Return sorted list of non-zero unique instance IDs."""
    ids = np.unique(instance_mask)
    return [int(i) for i in ids if i > 0]
