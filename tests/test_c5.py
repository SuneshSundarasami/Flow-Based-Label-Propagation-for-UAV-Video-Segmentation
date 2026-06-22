"""C5 tests: failure-case pair selection."""
from __future__ import annotations

from eval.failures import select_worst_pairs


def test_select_worst_pairs_sorts_by_metric_then_keyframe_target():
    rows = [
        {"keyframe": "50", "target_frame": "100", "miou_valid": "0.30"},
        {"keyframe": "0", "target_frame": "50", "miou_valid": "0.20"},
        {"keyframe": "100", "target_frame": "150", "miou_valid": "0.20"},
        {"keyframe": "150", "target_frame": "200", "miou_valid": "nan"},
    ]

    out = select_worst_pairs(rows, n=2)

    assert [(row["keyframe"], row["target_frame"]) for row in out] == [
        ("0", "50"),
        ("100", "150"),
    ]


def test_select_worst_pairs_uses_requested_metric():
    rows = [
        {"keyframe": "0", "target_frame": "50", "miou_all": "0.80"},
        {"keyframe": "50", "target_frame": "100", "miou_all": "0.10"},
    ]

    out = select_worst_pairs(rows, metric="miou_all", n=1)

    assert out[0]["keyframe"] == "50"


def test_select_worst_pairs_can_filter_low_valid_pixel_rows():
    rows = [
        {"keyframe": "0", "target_frame": "50", "miou_valid": "0.01", "valid_pct": "0.0"},
        {"keyframe": "50", "target_frame": "100", "miou_valid": "0.20", "valid_pct": "10.0"},
    ]

    out = select_worst_pairs(rows, n=1, min_valid_pct=5.0)

    assert out[0]["keyframe"] == "50"
