"""Distance-decay aggregation for full-video propagation results (WP C2)."""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence


METRIC_COLUMNS = ("miou_valid", "miou_all", "valid_pct")


def _to_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _stats(values: Sequence[float]) -> dict[str, str]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return {"mean": "nan", "std": "nan", "min": "nan", "max": "nan"}

    std = stdev(clean) if len(clean) > 1 else 0.0
    return {
        "mean": f"{mean(clean):.4f}",
        "std": f"{std:.4f}",
        "min": f"{min(clean):.4f}",
        "max": f"{max(clean):.4f}",
    }


def aggregate_decay(rows: Iterable[dict]) -> list[dict]:
    """Aggregate full-video propagation rows by absolute frame distance."""
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["distance"])].append(row)

    out: list[dict] = []
    for distance in sorted(grouped):
        group = grouped[distance]
        summary = {"distance": distance, "n_pairs": len(group)}
        for metric in METRIC_COLUMNS:
            stats = _stats([_to_float(row.get(metric)) for row in group])
            for name, value in stats.items():
                summary[f"{metric}_{name}"] = value
        out.append(summary)
    return out


def read_rows(path: str | Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def write_decay_csv(path: str | Path, rows: Sequence[dict]) -> None:
    if not rows:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
