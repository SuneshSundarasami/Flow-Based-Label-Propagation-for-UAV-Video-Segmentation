"""Per-pixel segmentation evaluation: per-class IoU and mIoU (WP B3).

Confusion-matrix approach:
    TP[c] = conf[c, c]
    FN[c] = conf[c, :].sum() - TP[c]   # gt=c but pred≠c
    FP[c] = conf[:, c].sum() - TP[c]   # pred=c but gt≠c
    IoU[c] = TP / (TP + FN + FP)

``compute_iou`` always returns **two** IoU evaluations:

* ``"all"``        — over every pixel except those labelled ``ignore_index``
* ``"valid_only"`` — additionally excludes pixels where ``valid_mask`` is False
                     (e.g. the forward-backward occlusion mask from B2).
                     Equals ``"all"`` when no ``valid_mask`` is supplied.

mIoU is averaged over classes present in the ground-truth after masking.
"""
from __future__ import annotations

import numpy as np


def _iou_from_keep(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    keep: np.ndarray,
) -> dict:
    """Compute IoU stats over the pixels selected by *keep*."""
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


def compute_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
    valid_mask: np.ndarray | None = None,
) -> dict:
    """Compute per-class IoU and mIoU in two ways simultaneously.

    Parameters
    ----------
    pred:
        (H, W) integer predicted labels.
    gt:
        (H, W) integer ground-truth labels.
    num_classes:
        Number of valid class ids (0 … num_classes-1).
    ignore_index:
        Pixels with this label in *either* pred or gt are always excluded.
    valid_mask:
        Optional (H, W) bool array from ``compute_fb_mask``.
        False pixels are excluded from the ``"valid_only"`` result but
        **not** from ``"all"``.

    Returns
    -------
    dict with two sub-dicts, each containing ``per_class``, ``miou``,
    and ``present``:

    ``"all"``
        IoU evaluated on every non-ignored pixel (ignores *valid_mask*).
        Use this to compare against baselines that have no occlusion mask.
    ``"valid_only"``
        IoU restricted to pixels where *valid_mask* is True.
        Use this for the fairest measure of the warp quality (excludes
        occluded / flow-unreliable regions). Equals ``"all"`` when
        *valid_mask* is None.
    """
    if pred.shape != gt.shape:
        raise ValueError(
            f"pred and gt must have the same shape; got {pred.shape} vs {gt.shape}"
        )
    if valid_mask is not None and valid_mask.shape != gt.shape:
        raise ValueError(
            f"valid_mask shape {valid_mask.shape} != gt shape {gt.shape}"
        )

    keep_all = (gt != ignore_index) & (pred != ignore_index)
    result_all = _iou_from_keep(pred, gt, num_classes, keep_all)

    if valid_mask is not None:
        keep_valid = keep_all & valid_mask
        result_valid = _iou_from_keep(pred, gt, num_classes, keep_valid)
    else:
        result_valid = result_all

    return {
        "all": result_all,
        "valid_only": result_valid,
    }
