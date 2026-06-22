"""Per-class IoU breakdown for full-video propagation results (WP C4)."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence


def _to_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _stats(values: Sequence[float]) -> dict[str, str | int]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"n_pairs": 0, "mean": "nan", "std": "nan", "min": "nan", "max": "nan"}
    return {
        "n_pairs": len(clean),
        "mean": f"{mean(clean):.4f}",
        "std": f"{stdev(clean):.4f}" if len(clean) > 1 else "0.0000",
        "min": f"{min(clean):.4f}",
        "max": f"{max(clean):.4f}",
    }


def summarize_per_class(
    rows: Iterable[dict],
    class_names: dict[int, str] | None = None,
) -> list[dict]:
    """Summarize valid-only IoU columns named ``iou_classN``."""
    rows = list(rows)
    class_ids = sorted(
        int(key.removeprefix("iou_class"))
        for row in rows
        for key in row
        if key.startswith("iou_class")
    )
    class_ids = sorted(set(class_ids))

    summary: list[dict] = []
    for cid in class_ids:
        key = f"iou_class{cid}"
        stats = _stats([_to_float(row.get(key)) for row in rows])
        summary.append({
            "class_id": cid,
            "class_name": (class_names or {}).get(cid, str(cid)),
            **stats,
        })

    def _rank_key(row: dict) -> tuple[int, float]:
        value = _to_float(row["mean"])
        return (row["n_pairs"] == 0, value if not math.isnan(value) else -1.0)

    return sorted(summary, key=_rank_key)


def write_per_class_csv(path: str | Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
