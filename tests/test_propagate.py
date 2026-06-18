"""B4 unit tests for propagate_keyframe."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from propagation import propagate_keyframe, FrameResult


class _ZeroFlow:
    """Stub FlowEstimator that always returns zero flow (identity warp)."""
    def estimate_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        return np.zeros((*img1.shape[:2], 2), dtype=np.float32)


class _ShiftFlow:
    """Stub that shifts by +1 in x for fwd, -1 for bwd (inverse pair)."""
    def __init__(self, shift: float = 1.0):
        self._shift = shift

    def estimate_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        flow = np.zeros((*img1.shape[:2], 2), dtype=np.float32)
        flow[..., 0] = self._shift
        return flow


def _frame(h=4, w=4) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_zero_flow_identity_warp():
    kf_mask = np.array([[0, 1], [2, 1]], dtype=np.int32)
    results = propagate_keyframe(
        keyframe_mask=kf_mask,
        keyframe_frame=_frame(2, 2),
        target_frames=[_frame(2, 2)],
        target_indices=[1],
        keyframe_index=0,
        model=_ZeroFlow(),
    )
    assert len(results) == 1
    np.testing.assert_array_equal(results[0].warped_mask, kf_mask)


def test_zero_flow_all_valid():
    results = propagate_keyframe(
        keyframe_mask=np.zeros((4, 4), dtype=np.int32),
        keyframe_frame=_frame(),
        target_frames=[_frame()],
        target_indices=[5],
        keyframe_index=0,
        model=_ZeroFlow(),
    )
    assert results[0].valid_pct == pytest.approx(100.0)


def test_distance_is_absolute():
    results = propagate_keyframe(
        keyframe_mask=np.zeros((2, 2), dtype=np.int32),
        keyframe_frame=_frame(2, 2),
        target_frames=[_frame(2, 2)],
        target_indices=[50],
        keyframe_index=0,
        model=_ZeroFlow(),
    )
    assert results[0].distance == 50
    assert results[0].target_index == 50


def test_result_shapes():
    h, w = 4, 6
    kf_mask = np.zeros((h, w), dtype=np.int32)
    results = propagate_keyframe(
        keyframe_mask=kf_mask,
        keyframe_frame=_frame(h, w),
        target_frames=[_frame(h, w), _frame(h, w)],
        target_indices=[1, 2],
        keyframe_index=0,
        model=_ZeroFlow(),
    )
    assert len(results) == 2
    for r in results:
        assert r.warped_mask.shape == (h, w)
        assert r.valid_mask.shape == (h, w)
        assert r.valid_mask.dtype == bool


def test_iou_computed_with_gt():
    kf_mask = np.array([[0, 0], [1, 1]], dtype=np.int32)
    gt_mask = np.array([[0, 0], [1, 1]], dtype=np.int32)  # perfect match
    results = propagate_keyframe(
        keyframe_mask=kf_mask,
        keyframe_frame=_frame(2, 2),
        target_frames=[_frame(2, 2)],
        target_indices=[1],
        keyframe_index=0,
        model=_ZeroFlow(),
        gt_masks=[gt_mask],
        num_classes=2,
    )
    r = results[0]
    assert r.iou is not None
    assert r.iou["all"]["miou"] == pytest.approx(1.0)
    assert r.iou["valid_only"]["miou"] == pytest.approx(1.0)


def test_iou_none_when_no_gt():
    results = propagate_keyframe(
        keyframe_mask=np.zeros((2, 2), dtype=np.int32),
        keyframe_frame=_frame(2, 2),
        target_frames=[_frame(2, 2)],
        target_indices=[1],
        keyframe_index=0,
        model=_ZeroFlow(),
        gt_masks=[None],
        num_classes=2,
    )
    assert results[0].iou is None


def test_iou_none_when_no_gt_masks_arg():
    results = propagate_keyframe(
        keyframe_mask=np.zeros((2, 2), dtype=np.int32),
        keyframe_frame=_frame(2, 2),
        target_frames=[_frame(2, 2)],
        target_indices=[1],
        keyframe_index=0,
        model=_ZeroFlow(),
    )
    assert results[0].iou is None


def test_multiple_targets():
    kf_mask = np.ones((3, 3), dtype=np.int32)
    targets = [1, 2, 3]
    results = propagate_keyframe(
        keyframe_mask=kf_mask,
        keyframe_frame=_frame(3, 3),
        target_frames=[_frame(3, 3)] * 3,
        target_indices=targets,
        keyframe_index=0,
        model=_ZeroFlow(),
    )
    assert len(results) == 3
    for r, t_idx in zip(results, targets):
        assert r.target_index == t_idx
        np.testing.assert_array_equal(r.warped_mask, kf_mask)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        propagate_keyframe(
            keyframe_mask=np.zeros((2, 2), dtype=np.int32),
            keyframe_frame=_frame(2, 2),
            target_frames=[_frame(2, 2), _frame(2, 2)],
            target_indices=[1],  # length mismatch
            keyframe_index=0,
            model=_ZeroFlow(),
        )


def test_gt_masks_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        propagate_keyframe(
            keyframe_mask=np.zeros((2, 2), dtype=np.int32),
            keyframe_frame=_frame(2, 2),
            target_frames=[_frame(2, 2)],
            target_indices=[1],
            keyframe_index=0,
            model=_ZeroFlow(),
            gt_masks=[None, None],  # length mismatch
            num_classes=2,
        )
