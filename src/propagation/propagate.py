"""End-to-end single-keyframe label propagation (WP B4).

Given one annotated keyframe and a list of target frames (any resolution),
this module:
  1. Estimates backward (t→k) and forward (k→t) optical flow for each target.
  2. Warps the keyframe mask into the target via backward warp (B1).
  3. Computes a forward-backward validity mask to flag occluded pixels (B2).
  4. Optionally computes IoU against a ground-truth mask when one is provided (B3).

All computation stays on numpy/torch; no I/O or visualization here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from eval import compute_iou
from flow import FlowEstimator
from warp import compute_fb_mask, warp_mask


def forward_targets(
    annotated_indices: Sequence[int],
    keyframe_index: int,
    n_targets: int,
) -> List[int]:
    """Return up to *n_targets* annotated frame indices after *keyframe_index*.

    Used by the full-video batched run (C1): each annotated keyframe is
    propagated forward to its next few annotated neighbours, which are the
    only frames with ground truth available for IoU evaluation.
    """
    later = [i for i in annotated_indices if i > keyframe_index]
    return later[:n_targets]


@dataclass
class FrameResult:
    """Per-target output of :func:`propagate_keyframe`."""
    target_index: int
    distance: int
    warped_mask: np.ndarray   # (H, W) int — propagated labels
    valid_mask: np.ndarray    # (H, W) bool — True where flow is reliable
    valid_pct: float          # percentage of valid pixels
    iou: Optional[dict]       # compute_iou output, or None if no GT
    timings: Dict[str, float] = field(default_factory=dict)  # step -> seconds


def propagate_keyframe(
    keyframe_mask: np.ndarray,
    keyframe_frame: np.ndarray,
    target_frames: List[np.ndarray],
    target_indices: List[int],
    keyframe_index: int,
    model: FlowEstimator,
    fb_threshold: float = 1.5,
    ignore_index: int = 255,
    gt_masks: Optional[List[Optional[np.ndarray]]] = None,
    num_classes: Optional[int] = None,
) -> List[FrameResult]:
    """Propagate a keyframe mask to each target frame.

    Parameters
    ----------
    keyframe_mask:
        (H, W) integer label map for the keyframe.
    keyframe_frame:
        (H, W, 3) uint8 RGB image of the keyframe.
    target_frames:
        List of (H, W, 3) uint8 RGB images to propagate into.
    target_indices:
        Frame indices (in the original video) for each target — used to compute
        the temporal distance from the keyframe.
    keyframe_index:
        Frame index of the keyframe.
    model:
        Any object satisfying the :class:`~flow.FlowEstimator` protocol.
    fb_threshold:
        Forward-backward round-trip error threshold in pixels (B2).
    ignore_index:
        Label value assigned to out-of-bounds and ignored pixels.
    gt_masks:
        Optional list of GT masks (one per target, None if unavailable).
        Required alongside *num_classes* to compute IoU.
    num_classes:
        Number of valid class ids (0 … num_classes-1). Required for IoU.

    Returns
    -------
    List of :class:`FrameResult`, one per target frame.
    """
    if len(target_frames) != len(target_indices):
        raise ValueError("target_frames and target_indices must have the same length")
    if gt_masks is not None and len(gt_masks) != len(target_frames):
        raise ValueError("gt_masks must have the same length as target_frames")

    n = len(target_frames)

    # One batched call: first N slots are bwd (tgt→kf), last N are fwd (kf→tgt).
    # Single GPU launch for all 2*N pairs simultaneously.
    t0 = time.perf_counter()
    all_flows = model.estimate_flow_batch(
        [(tgt, keyframe_frame) for tgt in target_frames]
        + [(keyframe_frame, tgt) for tgt in target_frames]
    )
    t_flow_total = time.perf_counter() - t0

    flows_bwd = all_flows[:n]
    flows_fwd = all_flows[n:]

    results: List[FrameResult] = []
    for i, tgt_idx in enumerate(target_indices):
        t0 = time.perf_counter()
        warped = warp_mask(keyframe_mask, flows_bwd[i], ignore_index=ignore_index)
        t_warp = time.perf_counter() - t0

        t0 = time.perf_counter()
        valid = compute_fb_mask(flows_fwd[i], flows_bwd[i], threshold=fb_threshold)
        t_fb = time.perf_counter() - t0

        valid_pct = 100.0 * float(valid.mean())

        t0 = time.perf_counter()
        iou_result: Optional[dict] = None
        if (
            gt_masks is not None
            and gt_masks[i] is not None
            and num_classes is not None
        ):
            iou_result = compute_iou(
                warped,
                gt_masks[i],  # type: ignore[arg-type]
                num_classes,
                ignore_index=ignore_index,
                valid_mask=valid,
            )
        t_iou = time.perf_counter() - t0

        results.append(
            FrameResult(
                target_index=tgt_idx,
                distance=abs(tgt_idx - keyframe_index),
                warped_mask=warped,
                valid_mask=valid,
                valid_pct=valid_pct,
                iou=iou_result,
                timings={
                    "flow": t_flow_total / n,
                    "warp": t_warp,
                    "fb": t_fb,
                    "iou": t_iou,
                },
            )
        )
    return results
