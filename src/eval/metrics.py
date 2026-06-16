"""Per-pixel segmentation evaluation: per-class IoU and mIoU (WP B3).

Confusion-matrix approach:
    TP[c] = conf[c, c]
    FN[c] = conf[c, :].sum() - TP[c]   # gt=c but pred≠c
    FP[c] = conf[:, c].sum() - TP[c]   # pred=c but gt≠c
    IoU[c] = TP / (TP + FN + FP)

mIoU is averaged over classes present in the ground-truth after masking.
"""
from __future__ import annotations

import numpy as np


def compute_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """Compute per-class IoU and mIoU between a predicted and ground-truth mask.

    Parameters
    ----------
    pred:
        (H, W) integer predicted labels.
    gt:
        (H, W) integer ground-truth labels.
    num_classes:
        Number of valid class ids (0 … num_classes-1).
    ignore_index:
        Pixels with this label in *either* pred or gt are excluded.
    valid_mask:
        Optional (H, W) bool array — additional pixels to exclude, e.g.
        the output of ``compute_fb_mask``.  False pixels are excluded.

    Returns
    -------
    dict with keys:
        ``per_class`` : dict mapping class_id → IoU float (nan if absent in gt)
        ``miou``      : float — mean IoU over classes present in gt
        ``present``   : list[int] — class ids that appear in the (masked) gt
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have the same shape; got {pred.shape} vs {gt.shape}"
        )
    if valid_mask is not None and valid_mask.shape != gt.shape:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} != gt shape {gt.shape}"
        )

    keep = (gt != ignore_index) & (pred != ignore_index)
    if valid_mask is not None:
        keep &= valid_mask

    gt_flat = gt[keep].astype(np.int64)
    pred_flat = pred[keep].astype(np.int64)

    in_range = (
        (gt_flat >= 0) & (gt_flat < num_classes) &
        (pred_flat >= 0) & (pred_flat < num_classes)
    )
    gt_flat = gt_flat[in_range]
    pred_flat = pred_flat[in_range]

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(conf, (gt_flat, pred_flat), 1)

    tp = np.diag(conf)
    fn = conf.sum(axis=1) - tp
    fp = conf.sum(axis=0) - tp
    denom = tp + fn + fp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)

    present = [c for c in range(num_classes) if conf[c].sum() > 0]
    miou = float(np.nanmean([iou[c] for c in present])) if present else float("nan")

    return {
        "per_class": {c: float(iou[c]) for c in range(num_classes)},
        "miou": miou,
        "present": present,
    }
