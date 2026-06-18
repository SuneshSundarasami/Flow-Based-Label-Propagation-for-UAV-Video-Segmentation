"""B3 unit tests for compute_iou (both 'all' and 'valid_only' variants)."""
import sys
from pathlib import Path
import math

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval import compute_iou


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _all(r):
    return r["all"]


def _valid(r):
    return r["valid_only"]


# ---------------------------------------------------------------------------
# "all" variant — behaviour when no valid_mask is supplied
# ---------------------------------------------------------------------------

def test_perfect_prediction_all():
    gt = np.array([[0, 1], [1, 2]], dtype=np.int32)
    pred = gt.copy()
    r = compute_iou(pred, gt, num_classes=3)
    assert _all(r)["miou"] == pytest.approx(1.0)
    assert _all(r)["per_class"][0] == pytest.approx(1.0)
    assert _all(r)["per_class"][1] == pytest.approx(1.0)
    assert _all(r)["per_class"][2] == pytest.approx(1.0)


def test_all_wrong_prediction_all():
    gt = np.array([[0, 0], [0, 0]], dtype=np.int32)
    pred = np.array([[1, 1], [1, 1]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=2)
    assert _all(r)["per_class"][0] == pytest.approx(0.0)
    assert _all(r)["per_class"][1] == pytest.approx(0.0)


def test_partial_overlap_all():
    # gt=[0,0,1], pred=[0,0,0]
    # class 0: TP=2, FP=1, FN=0 → IoU = 2/3
    # class 1: TP=0, FP=0, FN=1 → IoU = 0/1 = 0
    gt = np.array([[0, 0, 1]], dtype=np.int32)
    pred = np.array([[0, 0, 0]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=2)
    assert _all(r)["per_class"][0] == pytest.approx(2 / 3)
    assert _all(r)["per_class"][1] == pytest.approx(0.0)
    assert _all(r)["miou"] == pytest.approx((2 / 3 + 0.0) / 2)


def test_ignore_index_excluded_all():
    gt = np.array([[0, 255], [1, 1]], dtype=np.int32)
    pred = np.array([[0, 0], [1, 1]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=2, ignore_index=255)
    assert _all(r)["miou"] == pytest.approx(1.0)


def test_nan_for_absent_class_all():
    gt = np.array([[0, 0]], dtype=np.int32)
    pred = np.array([[0, 0]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=3)
    assert _all(r)["per_class"][0] == pytest.approx(1.0)
    assert math.isnan(_all(r)["per_class"][1])
    assert math.isnan(_all(r)["per_class"][2])
    assert _all(r)["miou"] == pytest.approx(1.0)
    assert _all(r)["present"] == [0]


def test_empty_after_ignore_returns_nan_all():
    gt = np.array([[255]], dtype=np.int32)
    pred = np.array([[0]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=2, ignore_index=255)
    assert math.isnan(_all(r)["miou"])
    assert _all(r)["present"] == []


# ---------------------------------------------------------------------------
# "valid_only" variant — additional pixel exclusion via valid_mask
# ---------------------------------------------------------------------------

def test_valid_mask_excludes_wrong_pixel_valid_only():
    gt = np.array([[0, 0], [1, 1]], dtype=np.int32)
    pred = np.array([[1, 0], [1, 1]], dtype=np.int32)  # (0,0) is wrong
    valid = np.array([[False, True], [True, True]], dtype=bool)
    r = compute_iou(pred, gt, num_classes=2, valid_mask=valid)
    # valid_only excludes the wrong pixel → perfect IoU
    assert _valid(r)["per_class"][0] == pytest.approx(1.0)
    assert _valid(r)["per_class"][1] == pytest.approx(1.0)
    assert _valid(r)["miou"] == pytest.approx(1.0)


def test_all_includes_wrong_pixel_all():
    # Same setup: "all" should see the mistake and give IoU < 1
    gt = np.array([[0, 0], [1, 1]], dtype=np.int32)
    pred = np.array([[1, 0], [1, 1]], dtype=np.int32)
    valid = np.array([[False, True], [True, True]], dtype=bool)
    r = compute_iou(pred, gt, num_classes=2, valid_mask=valid)
    # class 0: TP=1, FP=0, FN=1 (pixel (0,0)) → IoU = 1/2
    assert _all(r)["per_class"][0] == pytest.approx(0.5)
    assert _all(r)["miou"] < 1.0


def test_valid_only_equals_all_when_no_mask():
    gt = np.array([[0, 1], [1, 0]], dtype=np.int32)
    pred = gt.copy()
    r = compute_iou(pred, gt, num_classes=2)
    assert _valid(r)["miou"] == pytest.approx(_all(r)["miou"])
    for c in range(2):
        assert _valid(r)["per_class"][c] == pytest.approx(_all(r)["per_class"][c])


def test_valid_only_all_invalid_returns_nan():
    gt = np.array([[0, 1]], dtype=np.int32)
    pred = gt.copy()
    valid = np.zeros((1, 2), dtype=bool)  # all invalid
    r = compute_iou(pred, gt, num_classes=2, valid_mask=valid)
    assert math.isnan(_valid(r)["miou"])
    # "all" is unaffected — still perfect
    assert _all(r)["miou"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_iou(np.zeros((3, 3), dtype=np.int32),
                    np.zeros((4, 3), dtype=np.int32), num_classes=3)


def test_valid_mask_shape_mismatch_raises():
    gt = np.zeros((3, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        compute_iou(gt.copy(), gt, num_classes=3,
                    valid_mask=np.ones((2, 3), dtype=bool))
