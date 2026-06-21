"""C2: aggregate and plot mIoU decay versus propagation distance.

Reads the C1 ``all_pairs.csv`` table and writes:

* ``decay_summary.csv`` — mean/std/min/max grouped by distance.
* ``miou_decay.png``    — mIoU(valid) curve with a +/-1 std band.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_REPO_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_REPO_ROOT / "outputs" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import load_config  # noqa: E402
from eval.decay import aggregate_decay, read_rows, write_decay_csv  # noqa: E402


def _pre_load_config(argv=None) -> dict:
    if argv is None:
        argv = sys.argv[1:]
    cfg_path = None
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            cfg_path = argv[i + 1]
            break
        if arg.startswith("--config="):
            cfg_path = arg.split("=", 1)[1]
            break
    return load_config(override_path=cfg_path)


def _video_name(cfg: dict, frame_glob: str) -> str:
    parts = Path(frame_glob).parts
    if len(parts) >= 2 and parts[0] == "frames":
        return parts[1]
    return str(cfg.get("paths", {}).get("video", "video"))


def _float_column(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def _plot_decay(summary: list[dict], out_path: Path) -> None:
    distances = _float_column(summary, "distance")
    miou = _float_column(summary, "miou_valid_mean")
    miou_std = _float_column(summary, "miou_valid_std")
    miou_low = [max(0.0, m - s) for m, s in zip(miou, miou_std)]
    miou_high = [min(1.0, m + s) for m, s in zip(miou, miou_std)]
    valid_pct = _float_column(summary, "valid_pct_mean")

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(distances, miou, marker="o", color="#1f77b4", label="mIoU valid")
    ax.fill_between(distances, miou_low, miou_high, color="#1f77b4", alpha=0.18,
                    label="+/- 1 std")
    ax.set_xlabel("Propagation distance (frames)")
    ax.set_ylabel("mIoU on FB-valid pixels")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.28)

    ax2 = ax.twinx()
    ax2.plot(distances, valid_pct, marker="s", linestyle="--", color="#555555",
             label="valid pixels")
    ax2.set_ylabel("FB-valid pixels (%)")
    ax2.set_ylim(0.0, 100.0)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower left")
    ax.set_title("Propagation quality decays with keyframe distance")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _fmt(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def main() -> int:
    cfg = _pre_load_config()
    video_name = _video_name(cfg, cfg["data"]["frame_glob"])
    default_root = _REPO_ROOT / cfg["paths"]["output_dir"] / "results" / video_name
    default_in = default_root / "propagation_results" / "all_pairs.csv"
    default_out = default_root / "analysis"

    ap = argparse.ArgumentParser(
        description="C2: produce mIoU-vs-distance decay summary and plot."
    )
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="Override YAML deep-merged on top of default.yaml.")
    ap.add_argument("--input", default=str(default_in),
                    help="C1 all_pairs.csv path. [default: %(default)s]")
    ap.add_argument("--out-dir", default=str(default_out),
                    help="Directory for C2 outputs. [default: %(default)s]")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not in_path.exists():
        print(f"[c2] missing input CSV: {in_path}")
        return 1

    rows = read_rows(in_path)
    summary = aggregate_decay(rows)
    if not summary:
        print(f"[c2] no rows found in {in_path}")
        return 1

    summary_csv = out_dir / "decay_summary.csv"
    plot_path = out_dir / "miou_decay.png"
    write_decay_csv(summary_csv, summary)
    _plot_decay(summary, plot_path)

    first = summary[0]
    last = summary[-1]
    print(f"[c2] input       : {in_path}")
    print(f"[c2] pairs       : {len(rows)}")
    print(f"[c2] distances   : {len(summary)} ({first['distance']}..{last['distance']} frames)")
    print(f"[c2] mIoU valid  : {_fmt(float(first['miou_valid_mean']))} -> "
          f"{_fmt(float(last['miou_valid_mean']))}")
    print(f"[c2] wrote       : {summary_csv}")
    print(f"[c2] wrote       : {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
