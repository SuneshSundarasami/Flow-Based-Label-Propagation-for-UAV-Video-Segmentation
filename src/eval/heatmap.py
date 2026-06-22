"""Timeline heatmap aggregation for full-video propagation results (WP C3)."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence


def _to_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def build_heatmap_matrix(
    rows: Iterable[dict],
    metric: str = "miou_valid",
) -> tuple[list[int], list[int], list[list[float]]]:
    """Return keyframes, distances, and a metric matrix for heatmap plotting."""
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    keyframes: set[int] = set()
    distances: set[int] = set()

    for row in rows:
        keyframe = int(row["keyframe"])
        distance = int(row["distance"])
        value = _to_float(row.get(metric))
        keyframes.add(keyframe)
        distances.add(distance)
        if not math.isnan(value):
            grouped[(keyframe, distance)].append(value)

    keyframe_list = sorted(keyframes)
    distance_list = sorted(distances)
    matrix: list[list[float]] = []
    for keyframe in keyframe_list:
        values: list[float] = []
        for distance in distance_list:
            cell_values = grouped.get((keyframe, distance), [])
            values.append(mean(cell_values) if cell_values else math.nan)
        matrix.append(values)

    return keyframe_list, distance_list, matrix


def write_heatmap_csv(
    path: str | Path,
    keyframes: Sequence[int],
    distances: Sequence[int],
    matrix: Sequence[Sequence[float]],
) -> None:
    """Write a keyframe x distance matrix with one row per keyframe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["keyframe"] + [f"distance_{d}" for d in distances]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for keyframe, values in zip(keyframes, matrix):
            row = {"keyframe": keyframe}
            for distance, value in zip(distances, values):
                row[f"distance_{distance}"] = (
                    "nan" if math.isnan(value) else f"{value:.4f}"
                )
            writer.writerow(row)
