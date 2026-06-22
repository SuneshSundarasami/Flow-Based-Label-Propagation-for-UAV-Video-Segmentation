"""C4 tests: per-class IoU breakdown."""
from __future__ import annotations

import pytest

from eval.per_class import summarize_per_class


def test_summarize_per_class_uses_names_and_sorts_by_mean():
    rows = [
        {"iou_class0": "0.80", "iou_class1": "0.20"},
        {"iou_class0": "0.60", "iou_class1": "0.40"},
    ]

    out = summarize_per_class(rows, {0: "road", 1: "car"})

    assert [row["class_name"] for row in out] == ["car", "road"]
    assert out[0]["n_pairs"] == 2
    assert out[0]["mean"] == "0.3000"
    assert float(out[0]["std"]) == pytest.approx(0.1414, abs=1e-4)
    assert out[1]["mean"] == "0.7000"


def test_summarize_per_class_ignores_nan_values():
    rows = [
        {"iou_class0": "nan", "iou_class1": "0.50"},
        {"iou_class0": "nan", "iou_class1": "nan"},
    ]

    out = summarize_per_class(rows)

    by_id = {row["class_id"]: row for row in out}
    assert by_id[0]["n_pairs"] == 0
    assert by_id[0]["mean"] == "nan"
    assert by_id[1]["n_pairs"] == 1
    assert by_id[1]["mean"] == "0.5000"
