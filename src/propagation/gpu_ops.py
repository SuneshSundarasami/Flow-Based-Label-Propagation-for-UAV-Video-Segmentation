"""Batched, GPU-resident versions of warp / occlusion / IoU.

These mirror the numpy/torch-CPU reference implementations in :mod:`warp` and
:mod:`eval.metrics` exactly, but operate on whole ``(B, ...)`` batches kept on
the GPU so the backend runner never round-trips per-pair work through the host.

The reference modules remain the source of truth for correctness (and their
unit tests); this module is the throughput path used by ``backend_runner``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _sample_grid(flow_bwd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (grid, in_bounds) for backward-flow sampling.

    ``flow_bwd`` is ``(B, 2, H, W)``.  ``grid`` is ``(B, H, W, 2)`` normalised to
    [-1, 1] for ``grid_sample(align_corners=True)``; ``in_bounds`` is ``(B, H, W)``
    bool marking pixels whose source stays inside the frame.
    """
    b, _, h, w = flow_bwd.shape
    device = flow_bwd.device
    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    src_x = xs.unsqueeze(0) + flow_bwd[:, 0]
    src_y = ys.unsqueeze(0) + flow_bwd[:, 1]
    in_bounds = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)
    norm_x = 2.0 * src_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(h - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)  # (B, H, W, 2)
    return grid, in_bounds


def warp_masks_gpu(
    masks: torch.Tensor,
    flow_bwd: torch.Tensor,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Nearest-neighbour backward-warp a batch of label masks.

    ``masks`` is ``(B, H, W)`` integer labels; ``flow_bwd`` is ``(B, 2, H, W)``
    target->keyframe flow.  Out-of-bounds pixels become ``ignore_index``.
    Returns ``(B, H, W)`` int64 on the same device.
    """
    grid, in_bounds = _sample_grid(flow_bwd)
    warped = F.grid_sample(
        masks.float().unsqueeze(1), grid,
        mode="nearest", padding_mode="zeros", align_corners=True,
    ).squeeze(1).long()
    warped[~in_bounds] = ignore_index
    return warped


def fb_masks_gpu(
    flow_fwd: torch.Tensor,
    flow_bwd: torch.Tensor,
    threshold: float = 1.5,
) -> torch.Tensor:
    """Batched forward-backward consistency mask.

    ``flow_fwd``/``flow_bwd`` are ``(B, 2, H, W)``.  Returns ``(B, H, W)`` bool,
    True where the round-trip error is below *threshold* and the source is in
    bounds.
    """
    grid, in_bounds = _sample_grid(flow_bwd)
    fwd_sampled = F.grid_sample(
        flow_fwd, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )  # (B, 2, H, W)
    error = (flow_bwd + fwd_sampled).norm(dim=1)  # (B, H, W)
    return in_bounds & (error < threshold)


def _miou_from_conf(conf: torch.Tensor) -> float:
    """mIoU over ground-truth-present classes from a (C, C) confusion matrix."""
    tp = torch.diagonal(conf)
    fn = conf.sum(dim=1) - tp
    fp = conf.sum(dim=0) - tp
    denom = tp + fn + fp
    present = conf.sum(dim=1) > 0
    if not bool(present.any()):
        return float("nan")
    iou = tp[present] / denom[present].clamp_min(1)
    return float(iou.mean())


def iou_pair_gpu(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
    ignore_index: int,
    valid_mask: torch.Tensor,
) -> tuple[float, float]:
    """Return (miou_all, miou_valid) for one predicted/gt label map.

    Matches ``eval.metrics.compute_iou``: ``all`` keeps every non-ignored,
    in-range pixel; ``valid`` additionally keeps only ``valid_mask`` pixels.
    """
    keep_all = (
        (gt != ignore_index) & (pred != ignore_index)
        & (gt >= 0) & (gt < num_classes)
        & (pred >= 0) & (pred < num_classes)
    )
    g = gt[keep_all]
    p = pred[keep_all]
    conf_all = torch.bincount(
        g * num_classes + p, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).double()

    keep_valid = keep_all & valid_mask
    gv = gt[keep_valid]
    pv = pred[keep_valid]
    conf_valid = torch.bincount(
        gv * num_classes + pv, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes).double()

    return _miou_from_conf(conf_all), _miou_from_conf(conf_valid)
