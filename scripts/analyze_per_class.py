"""C4: summarize per-class propagation IoU over the full video."""
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
from data import load_palette  # noqa: E402
from eval.decay import read_rows  # noqa: E402
from eval.per_class import summarize_per_class, write_per_class_csv  # noqa: E402


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


def _to_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _plot_per_class(summary: list[dict], out_path: Path) -> None:
    rows = [row for row in summary if row["n_pairs"] > 0]
    names = [row["class_name"] for row in rows]
    means = [_to_float(row["mean"]) for row in rows]
    stds = [_to_float(row["std"]) for row in rows]
    colors = ["#d62728" if m < 0.4 else "#ffbf00" if m < 0.65 else "#2ca02c" for m in means]

    fig_h = max(4.8, 0.36 * len(rows))
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    y = list(range(len(rows)))
    ax.barh(y, means, xerr=stds, color=colors, alpha=0.88, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Mean valid-only IoU")
    ax.set_title("Per-class propagation quality")
    ax.grid(axis="x", alpha=0.25)

    for yi, row in zip(y, rows):
        ax.text(
            min(0.98, _to_float(row["mean"]) + 0.02),
            yi,
            f"n={row['n_pairs']}",
            va="center",
            fontsize=8,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    cfg = _pre_load_config()
    video_name = _video_name(cfg, cfg["data"]["frame_glob"])
    default_root = _REPO_ROOT / cfg["paths"]["output_dir"] / "results" / video_name
    default_in = default_root / "propagation_results" / "all_pairs.csv"
    default_out = default_root / "analysis"

    ap = argparse.ArgumentParser(description="C4: summarize per-class IoU.")
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="Override YAML deep-merged on top of default.yaml.")
    ap.add_argument("--input", default=str(default_in),
                    help="C1 all_pairs.csv path. [default: %(default)s]")
    ap.add_argument("--out-dir", default=str(default_out),
                    help="Directory for C4 outputs. [default: %(default)s]")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not in_path.exists():
        print(f"[c4] missing input CSV: {in_path}")
        return 1

    palette = load_palette()
    rows = read_rows(in_path)
    summary = summarize_per_class(rows, palette.names)
    if not summary:
        print(f"[c4] no per-class columns found in {in_path}")
        return 1

    csv_path = out_dir / "per_class_iou.csv"
    plot_path = out_dir / "per_class_iou.png"
    write_per_class_csv(csv_path, summary)
    _plot_per_class(summary, plot_path)

    observed = [row for row in summary if row["n_pairs"] > 0]
    best = max(observed, key=lambda row: _to_float(row["mean"]))
    worst = min(observed, key=lambda row: _to_float(row["mean"]))
    print(f"[c4] input       : {in_path}")
    print(f"[c4] classes     : {len(observed)} observed / {len(summary)} total")
    print(f"[c4] best        : {best['class_name']} ({float(best['mean']):.3f})")
    print(f"[c4] worst       : {worst['class_name']} ({float(worst['mean']):.3f})")
    print(f"[c4] wrote       : {csv_path}")
    print(f"[c4] wrote       : {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
