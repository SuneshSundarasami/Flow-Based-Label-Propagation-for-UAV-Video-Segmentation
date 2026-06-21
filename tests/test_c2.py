"""C2 tests: mIoU-vs-distance aggregation."""
from __future__ import annotations

import math

import pytest

from eval.decay import aggregate_decay


def test_aggregate_decay_groups_by_distance_and_sorts():
    rows = [
        {"distance": "100", "miou_valid": "0.50", "miou_all": "0.40", "valid_pct": "80"},
        {"distance": "50", "miou_valid": "0.80", "miou_all": "0.70", "valid_pct": "90"},
        {"distance": "100", "miou_valid": "0.70", "miou_all": "0.60", "valid_pct": "60"},
    ]

    out = aggregate_decay(rows)

    assert [row["distance"] for row in out] == [50, 100]
    assert out[0]["n_pairs"] == 1
    assert out[0]["miou_valid_mean"] == "0.8000"
    assert out[0]["miou_valid_std"] == "0.0000"
    assert out[1]["n_pairs"] == 2
    assert out[1]["miou_valid_mean"] == "0.6000"
    assert float(out[1]["miou_valid_std"]) == pytest.approx(math.sqrt(0.02), abs=1e-4)
    assert out[1]["valid_pct_mean"] == "70.0000"


def test_aggregate_decay_ignores_nan_metrics():
    rows = [
        {"distance": "50", "miou_valid": "nan", "miou_all": "0.70", "valid_pct": "90"},
        {"distance": "50", "miou_valid": "0.60", "miou_all": "nan", "valid_pct": "nan"},
    ]

    out = aggregate_decay(rows)

    assert out[0]["n_pairs"] == 2
    assert out[0]["miou_valid_mean"] == "0.6000"
    assert out[0]["miou_all_mean"] == "0.7000"
    assert out[0]["valid_pct_mean"] == "90.0000"
