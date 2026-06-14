"""B1/B2 unit tests for flow-based mask warping and occlusion detection."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp import warp_mask, compute_fb_mask


def test_zero_flow_is_identity():
    mask = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    flow = np.zeros((*mask.shape, 2), dtype=np.float32)
    out = warp_mask(mask, flow)
    np.testing.assert_array_equal(out, mask)


def test_integer_shift_right():
    # Keyframe has class 1 in column 1; flow (u=-1, v=0) makes every target
    # pixel sample one column to the left in the keyframe, so the class-1
    # stripe appears at column 2 in the warped output.
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:, 1] = 1
    flow = np.zeros((4, 4, 2), dtype=np.float32)
    flow[..., 0] = -1.0  # u = -1: target col x samples from keyframe col x-1
    out = warp_mask(mask, flow)
    assert (out[:, 2] == 1).all(), "class-1 stripe should shift to column 2"


def test_out_of_bounds_pixels_get_ignore_index():
    mask = np.zeros((4, 4), dtype=np.uint8)
    flow = np.zeros((4, 4, 2), dtype=np.float32)
    flow[..., 0] = 100.0  # all source coords fall far outside the keyframe
    out = warp_mask(mask, flow, ignore_index=255)
    assert (out == 255).all()


def test_custom_ignore_index():
    mask = np.zeros((3, 3), dtype=np.uint8)
    flow = np.full((3, 3, 2), -999.0, dtype=np.float32)
    out = warp_mask(mask, flow, ignore_index=0)
    assert (out == 0).all()


def test_shape_mismatch_raises():
    mask = np.zeros((4, 4), dtype=np.uint8)
    bad_flow = np.zeros((3, 4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        warp_mask(mask, bad_flow)


def test_output_dtype_matches_input():
    for dt in (np.uint8, np.int32):
        mask = np.arange(6, dtype=dt).reshape(2, 3)
        flow = np.zeros((2, 3, 2), dtype=np.float32)
        out = warp_mask(mask, flow)
        assert out.dtype == dt


# ---------------------------------------------------------------------------
# B2: forward-backward occlusion mask
# ---------------------------------------------------------------------------

def test_fb_zero_flows_all_valid():
    h, w = 8, 8
    fwd = np.zeros((h, w, 2), dtype=np.float32)
    bwd = np.zeros((h, w, 2), dtype=np.float32)
    valid = compute_fb_mask(fwd, bwd, threshold=1.5)
    assert valid.shape == (h, w)
    assert valid.all(), "zero flows should give zero round-trip error"


def test_fb_perfect_inverse_flows_valid_interior():
    # fwd=(+1,0), bwd=(-1,0): round-trip residual is zero for interior pixels.
    h, w = 8, 8
    fwd = np.zeros((h, w, 2), dtype=np.float32)
    fwd[..., 0] = 1.0
    bwd = np.zeros((h, w, 2), dtype=np.float32)
    bwd[..., 0] = -1.0
    valid = compute_fb_mask(fwd, bwd, threshold=1.5)
    # Interior pixels (x >= 1) should be valid; x=0 has src_x=-1 (out of bounds).
    assert valid[:, 1:].all(), "interior pixels should be valid"
    assert not valid[:, 0].any(), "left border out-of-bounds → invalid"


def test_fb_large_inconsistency_all_invalid():
    h, w = 6, 6
    fwd = np.zeros((h, w, 2), dtype=np.float32)
    fwd[..., 0] = 10.0   # forward says go right 10 px
    bwd = np.zeros((h, w, 2), dtype=np.float32)
    # backward says stay (don't move) → residual = 10 px → all invalid
    valid = compute_fb_mask(fwd, bwd, threshold=1.5)
    assert not valid.any()


def test_fb_threshold_respected():
    h, w = 4, 4
    fwd = np.zeros((h, w, 2), dtype=np.float32)
    fwd[..., 0] = 1.0   # residual will be ~1 px
    bwd = np.zeros((h, w, 2), dtype=np.float32)
    assert compute_fb_mask(fwd, bwd, threshold=2.0)[:, :].any(), "below threshold → valid"
    assert not compute_fb_mask(fwd, bwd, threshold=0.5).any(), "above threshold → invalid"


def test_fb_shape_mismatch_raises():
    fwd = np.zeros((4, 4, 2), dtype=np.float32)
    bwd = np.zeros((3, 4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_fb_mask(fwd, bwd)
