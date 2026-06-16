"""B3 unit tests for compute_iou."""
import sys
from pathlib import Path
import math

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval import compute_iou


def test_perfect_prediction():
    gt = np.array([[0, 1], [1, 2]], dtype=np.int32)
    pred = gt.copy()
    r = compute_iou(pred, gt, num_classes=3)
    assert r["miou"] == pytest.approx(1.0)
    assert r["per_class"][0] == pytest.approx(1.0)
    assert r["per_class"][1] == pytest.approx(1.0)
    assert r["per_class"][2] == pytest.approx(1.0)


def test_all_wrong_prediction():
    gt = np.array([[0, 0], [0, 0]], dtype=np.int32)
    pred = np.array([[1, 1], [1, 1]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=2)
    assert r["per_class"][0] == pytest.approx(0.0)
    # class 1 has FP only → IoU = 0/4 = 0
    assert r["per_class"][1] == pytest.approx(0.0)


def test_partial_overlap():
    # gt=[0,0,1], pred=[0,0,0]
    # class 0: TP=2, FP=1, FN=0 → IoU = 2/3
    # class 1: TP=0, FP=0, FN=1 → IoU = 0/1 = 0
    gt = np.array([[0, 0, 1]], dtype=np.int32)
    pred = np.array([[0, 0, 0]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=2)
    assert r["per_class"][0] == pytest.approx(2 / 3)
    assert r["per_class"][1] == pytest.approx(0.0)
    assert r["miou"] == pytest.approx((2 / 3 + 0.0) / 2)


def test_ignore_index_excluded():
    # pixel (0,1) has gt=255 (ignore); pred=0 there should not count
    gt = np.array([[0, 255], [1, 1]], dtype=np.int32)
    pred = np.array([[0, 0], [1, 1]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=2, ignore_index=255)
    assert r["miou"] == pytest.approx(1.0)


def test_valid_mask_excludes_wrong_pixel():
    gt = np.array([[0, 0], [1, 1]], dtype=np.int32)
    pred = np.array([[1, 0], [1, 1]], dtype=np.int32)  # (0,0) is wrong
    valid = np.array([[False, True], [True, True]], dtype=bool)
    r = compute_iou(pred, gt, num_classes=2, valid_mask=valid)
    assert r["per_class"][0] == pytest.approx(1.0)
    assert r["per_class"][1] == pytest.approx(1.0)
    assert r["miou"] == pytest.approx(1.0)


def test_nan_for_absent_class():
    gt = np.array([[0, 0]], dtype=np.int32)
    pred = np.array([[0, 0]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=3)
    assert r["per_class"][0] == pytest.approx(1.0)
    assert math.isnan(r["per_class"][1])
    assert math.isnan(r["per_class"][2])
    assert r["miou"] == pytest.approx(1.0)
    assert r["present"] == [0]


def test_shape_mismatch_raises():
    pred = np.zeros((3, 3), dtype=np.int32)
    gt = np.zeros((4, 3), dtype=np.int32)
    with pytest.raises(ValueError):
        compute_iou(pred, gt, num_classes=3)


def test_valid_mask_shape_mismatch_raises():
    pred = np.zeros((3, 3), dtype=np.int32)
    gt = np.zeros((3, 3), dtype=np.int32)
    bad_mask = np.ones((2, 3), dtype=bool)
    with pytest.raises(ValueError):
        compute_iou(pred, gt, num_classes=3, valid_mask=bad_mask)


def test_empty_after_masking_returns_nan():
    gt = np.array([[255]], dtype=np.int32)
    pred = np.array([[0]], dtype=np.int32)
    r = compute_iou(pred, gt, num_classes=2, ignore_index=255)
    assert math.isnan(r["miou"])
    assert r["present"] == []
