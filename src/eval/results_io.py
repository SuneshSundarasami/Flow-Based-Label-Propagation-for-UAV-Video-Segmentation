"""Canonical CSV schema for propagation results (WP B5 / C1).

A single per-pair row records the IoU of one (keyframe → target) propagation:

    keyframe, target_frame, distance, valid_pct,
    miou_all, miou_valid, iou_class0, iou_class1, ...

* ``miou_all``    — mIoU over every non-ignored pixel.
* ``miou_valid``  — mIoU over forward-backward valid pixels only.
* ``iou_classN``  — per-class IoU on the valid-only pixels (the primary measure).

Both ``run_propagation.py`` (single keyframe, B5) and ``run_full_video.py``
(all keyframes, C1) emit this exact layout, so C1 can simply concatenate the
per-keyframe CSVs into one full-video table.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, List, Sequence


def _fmt(value: float | None) -> str:
    """Format a metric: 4 decimals, or 'nan' for missing/NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.4f}"


def keyframe_rows(
    results: Sequence,
    keyframe_index: int,
    num_classes: int,
) -> List[dict]:
    """Convert ``propagate_keyframe`` output into per-pair CSV row dicts.

    Parameters
    ----------
    results:
        Sequence of ``FrameResult`` (one per target).
    keyframe_index:
        Frame index of the keyframe these results came from.
    num_classes:
        Number of classes; produces ``iou_class0 … iou_class{n-1}`` columns.
    """
    rows: List[dict] = []
    for r in results:
        iou_all = r.iou["all"] if r.iou else None
        iou_valid = r.iou["valid_only"] if r.iou else None

        row = {
            "keyframe": keyframe_index,
            "target_frame": r.target_index,
            "distance": r.distance,
            "valid_pct": f"{r.valid_pct:.2f}",
            "miou_all": _fmt(iou_all["miou"] if iou_all else None),
            "miou_valid": _fmt(iou_valid["miou"] if iou_valid else None),
        }
        for c in range(num_classes):
            row[f"iou_class{c}"] = _fmt(
                iou_valid["per_class"].get(c) if iou_valid else None
            )
        rows.append(row)
    return rows


def write_csv(path: str | Path, rows: Sequence[dict]) -> None:
    """Write *rows* to *path* as CSV, header taken from the first row."""
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> List[dict]:
    """Read a CSV written by :func:`write_csv` back into a list of dicts."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def concat_csvs(paths: Iterable[str | Path], out_path: str | Path) -> int:
    """Concatenate several keyframe CSVs into one full-video table.

    Rows are read in the order *paths* is given and written sorted by
    (keyframe, distance) for a stable, readable table. The header is taken
    from the first non-empty file. Returns the number of rows written.
    """
    all_rows: List[dict] = []
    fieldnames: List[str] | None = None
    for p in paths:
        for row in read_csv(p):
            if fieldnames is None:
                fieldnames = list(row.keys())
            all_rows.append(row)

    if not all_rows or fieldnames is None:
        return 0

    def _key(row: dict):
        return (int(row["keyframe"]), int(row["distance"]))

    all_rows.sort(key=_key)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    return len(all_rows)
