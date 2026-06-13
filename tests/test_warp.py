"""B1 unit tests for flow-based mask warping."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from warp import warp_mask


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
