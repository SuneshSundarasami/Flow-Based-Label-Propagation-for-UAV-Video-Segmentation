"""C1 tests: target scheduling and full-video result CSV aggregation."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval import concat_csvs, keyframe_rows, read_csv, write_csv
from propagation import FrameResult, forward_targets


# ---------------------------------------------------------------------------
# forward_targets — keyframe scheduling
# ---------------------------------------------------------------------------

def test_forward_targets_picks_next_n():
    ann = [0, 50, 100, 150, 200]
    assert forward_targets(ann, 0, 3) == [50, 100, 150]


def test_forward_targets_excludes_self_and_past():
    ann = [0, 50, 100, 150]
    assert forward_targets(ann, 50, 5) == [100, 150]


def test_forward_targets_clamps_to_available():
    ann = [0, 50, 100]
    assert forward_targets(ann, 50, 10) == [100]


def test_forward_targets_last_keyframe_empty():
    ann = [0, 50, 100]
    assert forward_targets(ann, 100, 3) == []


# ---------------------------------------------------------------------------
# keyframe_rows — canonical schema
# ---------------------------------------------------------------------------

def _result(target_index, distance, miou_all, miou_valid, per_class_valid):
    iou = {
        "all": {"miou": miou_all, "per_class": {}, "present": []},
        "valid_only": {"miou": miou_valid, "per_class": per_class_valid, "present": []},
    }
    return FrameResult(
        target_index=target_index,
        distance=distance,
        warped_mask=np.zeros((2, 2), dtype=np.int32),
        valid_mask=np.ones((2, 2), dtype=bool),
        valid_pct=95.0,
        iou=iou,
    )


def test_keyframe_rows_schema_and_values():
    results = [
        _result(50, 50, 0.80, 0.85, {0: 0.9, 1: 0.7, 2: float("nan")}),
        _result(100, 100, 0.60, 0.65, {0: 0.8, 1: 0.5, 2: 0.4}),
    ]
    rows = keyframe_rows(results, keyframe_index=0, num_classes=3)
    assert len(rows) == 2
    expected_cols = ["keyframe", "target_frame", "distance", "valid_pct",
                     "miou_all", "miou_valid", "iou_class0", "iou_class1", "iou_class2"]
    assert list(rows[0].keys()) == expected_cols
    assert rows[0]["keyframe"] == 0
    assert rows[0]["target_frame"] == 50
    assert rows[0]["distance"] == 50
    assert rows[0]["miou_all"] == "0.8000"
    assert rows[0]["miou_valid"] == "0.8500"
    assert rows[0]["iou_class0"] == "0.9000"
    assert rows[0]["iou_class2"] == "nan"  # NaN per-class formatted as 'nan'


def test_keyframe_rows_no_iou_is_nan():
    r = FrameResult(
        target_index=10, distance=10,
        warped_mask=np.zeros((2, 2), dtype=np.int32),
        valid_mask=np.ones((2, 2), dtype=bool),
        valid_pct=100.0, iou=None,
    )
    rows = keyframe_rows([r], keyframe_index=0, num_classes=2)
    assert rows[0]["miou_all"] == "nan"
    assert rows[0]["miou_valid"] == "nan"
    assert rows[0]["iou_class0"] == "nan"


# ---------------------------------------------------------------------------
# write / read / concat
# ---------------------------------------------------------------------------

def test_write_read_round_trip(tmp_path):
    results = [_result(50, 50, 0.8, 0.85, {0: 0.9, 1: 0.7})]
    rows = keyframe_rows(results, 0, num_classes=2)
    path = tmp_path / "keyframe_0.csv"
    write_csv(path, rows)
    back = read_csv(path)
    assert back[0]["keyframe"] == "0"
    assert back[0]["miou_valid"] == "0.8500"


def test_write_csv_empty_is_noop(tmp_path):
    path = tmp_path / "empty.csv"
    write_csv(path, [])
    assert not path.exists()


def test_concat_merges_and_sorts(tmp_path):
    # keyframe 100 written first, then 0 — concat must sort by (keyframe, distance)
    rows100 = keyframe_rows([_result(150, 50, 0.7, 0.7, {0: 0.7, 1: 0.7})], 100, 2)
    rows0 = keyframe_rows(
        [_result(100, 100, 0.6, 0.6, {0: 0.6, 1: 0.6}),
         _result(50, 50, 0.8, 0.8, {0: 0.8, 1: 0.8})], 0, 2)
    p100 = tmp_path / "keyframe_100.csv"
    p0 = tmp_path / "keyframe_0.csv"
    write_csv(p100, rows100)
    write_csv(p0, rows0)

    out = tmp_path / "all_pairs.csv"
    n = concat_csvs([p100, p0], out)  # deliberately out of order
    assert n == 3

    merged = read_csv(out)
    keys = [(int(r["keyframe"]), int(r["distance"])) for r in merged]
    assert keys == [(0, 50), (0, 100), (100, 50)]  # sorted


def test_concat_empty_returns_zero(tmp_path):
    out = tmp_path / "all_pairs.csv"
    assert concat_csvs([], out) == 0
