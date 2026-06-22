"""Failure-case selection helpers for full-video results (WP C5)."""
from __future__ import annotations

import math
from typing import Iterable


def _to_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def select_worst_pairs(
    rows: Iterable[dict],
    metric: str = "miou_valid",
    n: int = 2,
    min_valid_pct: float = 0.0,
) -> list[dict]:
    """Return the lowest-scoring rows with finite values for *metric*."""
    candidates = []
    for row in rows:
        value = _to_float(row.get(metric))
        valid_pct = _to_float(row.get("valid_pct"))
        if math.isnan(value):
            continue
        if not math.isnan(valid_pct) and valid_pct < min_valid_pct:
            continue
        candidates.append((value, int(row["keyframe"]), int(row["target_frame"]), row))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return [row for _, _, _, row in candidates[:n]]
