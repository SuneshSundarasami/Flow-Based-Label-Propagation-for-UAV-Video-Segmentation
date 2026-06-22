"""C3 tests: timeline difficulty heatmap matrix."""
from __future__ import annotations

import math

from eval.heatmap import build_heatmap_matrix, write_heatmap_csv


def test_build_heatmap_matrix_sorts_axes_and_marks_missing_cells():
    rows = [
        {"keyframe": "100", "distance": "50", "miou_valid": "0.80"},
        {"keyframe": "0", "distance": "100", "miou_valid": "0.60"},
        {"keyframe": "0", "distance": "50", "miou_valid": "0.70"},
    ]

    keyframes, distances, matrix = build_heatmap_matrix(rows)

    assert keyframes == [0, 100]
    assert distances == [50, 100]
    assert matrix[0] == [0.70, 0.60]
    assert matrix[1][0] == 0.80
    assert math.isnan(matrix[1][1])


def test_build_heatmap_matrix_averages_duplicate_cells_and_ignores_nan():
    rows = [
        {"keyframe": "0", "distance": "50", "miou_valid": "0.50"},
        {"keyframe": "0", "distance": "50", "miou_valid": "0.70"},
        {"keyframe": "0", "distance": "50", "miou_valid": "nan"},
    ]

    _, _, matrix = build_heatmap_matrix(rows)

    assert matrix == [[0.60]]


def test_write_heatmap_csv(tmp_path):
    path = tmp_path / "heatmap.csv"

    write_heatmap_csv(path, [0, 50], [50, 100], [[0.5, math.nan], [0.7, 0.8]])

    assert path.read_text().splitlines() == [
        "keyframe,distance_50,distance_100",
        "0,0.5000,nan",
        "50,0.7000,0.8000",
    ]
