"""Run the existing DJI_0043 960px SegProp baseline through Table 1 stages."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEGPROP_ROOT = _REPO_ROOT / "third_party" / "segprop"
sys.path.insert(0, str(_SEGPROP_ROOT))

import segprop  # type: ignore  # noqa: E402
import stats  # type: ignore  # noqa: E402

EXPECTED = {
    "i01": (0.884, 0.801),
    "i02": (0.894, 0.817),
    "i03": (0.896, 0.819),
    "i04": (0.897, 0.821),
    "i05": (0.897, 0.821),
    "i06": (0.897, 0.821),
    "i07": (0.897, 0.821),
    "filtered": (0.903, 0.829),
}


def frame_copy(index: int) -> bool:
    return index % 100 == 0


def frame_filter(index: int) -> bool:
    return (index % 50 != 0) or (index % 100 == 0)


def main() -> int:
    root = _REPO_ROOT / "outputs" / "segprop_baseline" / "DJI_0043"
    video = "DJI_0043"
    flow = (
        str(root / "flow_960" / f"{video}_forward.h5"),
        str(root / "flow_960" / f"{video}_backward.h5"),
    )
    output = root / "output_960"
    labels_odd = root / "labels_960" / "train_odd"

    for it in range(2, 8):
        prev = output / f"i{it-1:02d}" / video
        cur = output / f"i{it:02d}" / video
        print(f"[table1-960] iterate i{it:02d}: {prev} -> {cur}", flush=True)
        segprop.iterate(
            flow,
            str(prev),
            str(cur),
            pv_series=[0, 5, 10],
            frame_copy=frame_copy,
            device="cuda",
            overwrite=False,
        )

    filtered = output / "filtered" / video
    print(f"[table1-960] denoise: {output / 'i07' / video} -> {filtered}", flush=True)
    segprop.denoise(
        flow,
        str(output / "i07" / video),
        str(filtered),
        pv_series=[0, 1, 3, 5, 7],
        method="self",
        frame_filter=frame_filter,
        device="cuda",
        overwrite=False,
    )

    rows = []
    for stage in ["i01", "i02", "i03", "i04", "i05", "i06", "i07", "filtered"]:
        stage_root = output / stage
        fmeasure, miou = stats.evaluate(str(stage_root), str(labels_odd))
        paper_f, paper_i = EXPECTED[stage]
        rows.append({
            "stage": stage,
            "reproduced_mf1": f"{float(fmeasure):.6f}",
            "reproduced_miou": f"{float(miou):.6f}",
            "paper_mf1": f"{paper_f:.6f}",
            "paper_miou": f"{paper_i:.6f}",
            "delta_mf1": f"{float(fmeasure) - paper_f:.6f}",
            "delta_miou": f"{float(miou) - paper_i:.6f}",
            "note": "DJI_0043 960px baseline, not full paper train split",
        })

    csv_path = root / "table1_reproduction_960_dji0043.csv"
    md_path = root / "table1_reproduction_960_dji0043.md"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(md_path, "w") as fh:
        fields = list(rows[0])
        fh.write("| " + " | ".join(fields) + " |\n")
        fh.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in rows:
            fh.write("| " + " | ".join(row[field] for field in fields) + " |\n")

    print(f"[table1-960] wrote {csv_path}", flush=True)
    print(f"[table1-960] wrote {md_path}", flush=True)
    for row in rows:
        print(row, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
