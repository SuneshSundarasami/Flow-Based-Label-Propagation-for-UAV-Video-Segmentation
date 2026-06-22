"""C3: create a keyframe-by-distance propagation difficulty heatmap.

Reads the C1 ``all_pairs.csv`` table and writes:

* ``difficulty_heatmap_matrix.csv`` — keyframe x distance metric matrix.
* ``difficulty_heatmap.png``        — timeline heatmap figure.
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
import numpy as np

sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import load_config  # noqa: E402
from eval.decay import read_rows  # noqa: E402
from eval.heatmap import build_heatmap_matrix, write_heatmap_csv  # noqa: E402


METRIC_CHOICES = ("miou_valid", "miou_all", "valid_pct")


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


def _plot_heatmap(
    keyframes: list[int],
    distances: list[int],
    matrix: list[list[float]],
    metric: str,
    out_path: Path,
) -> None:
    values = np.array(matrix, dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")

    fig_h = min(12.0, max(4.8, 0.08 * len(keyframes)))
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    vmin, vmax = (0.0, 100.0) if metric == "valid_pct" else (0.0, 1.0)
    img = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel("Propagation distance (frames)")
    ax.set_ylabel("Keyframe index")
    ax.set_title(f"Propagation difficulty heatmap ({metric})")
    ax.set_xticks(range(len(distances)))
    ax.set_xticklabels([str(d) for d in distances], rotation=45, ha="right")

    if len(keyframes) <= 25:
        y_ticks = list(range(len(keyframes)))
    else:
        step = max(1, math.ceil(len(keyframes) / 18))
        y_ticks = list(range(0, len(keyframes), step))
        if y_ticks[-1] != len(keyframes) - 1:
            y_ticks.append(len(keyframes) - 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(keyframes[i]) for i in y_ticks])

    label = "FB-valid pixels (%)" if metric == "valid_pct" else "IoU"
    cbar = fig.colorbar(img, ax=ax, shrink=0.95)
    cbar.set_label(label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _finite_min_max(matrix: list[list[float]]) -> tuple[float, float]:
    values = [value for row in matrix for value in row if not math.isnan(value)]
    if not values:
        return math.nan, math.nan
    return min(values), max(values)


def main() -> int:
    cfg = _pre_load_config()
    video_name = _video_name(cfg, cfg["data"]["frame_glob"])
    default_root = _REPO_ROOT / cfg["paths"]["output_dir"] / "results" / video_name
    default_in = default_root / "propagation_results" / "all_pairs.csv"
    default_out = default_root / "analysis"

    ap = argparse.ArgumentParser(
        description="C3: produce a keyframe-by-distance difficulty heatmap."
    )
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="Override YAML deep-merged on top of default.yaml.")
    ap.add_argument("--input", default=str(default_in),
                    help="C1 all_pairs.csv path. [default: %(default)s]")
    ap.add_argument("--out-dir", default=str(default_out),
                    help="Directory for C3 outputs. [default: %(default)s]")
    ap.add_argument("--metric", default="miou_valid", choices=METRIC_CHOICES,
                    help="Metric to show in the heatmap. [default: %(default)s]")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not in_path.exists():
        print(f"[c3] missing input CSV: {in_path}")
        return 1

    rows = read_rows(in_path)
    keyframes, distances, matrix = build_heatmap_matrix(rows, metric=args.metric)
    if not keyframes or not distances:
        print(f"[c3] no rows found in {in_path}")
        return 1

    matrix_csv = out_dir / "difficulty_heatmap_matrix.csv"
    plot_path = out_dir / "difficulty_heatmap.png"
    write_heatmap_csv(matrix_csv, keyframes, distances, matrix)
    _plot_heatmap(keyframes, distances, matrix, args.metric, plot_path)

    min_v, max_v = _finite_min_max(matrix)
    print(f"[c3] input       : {in_path}")
    print(f"[c3] metric      : {args.metric}")
    print(f"[c3] keyframes   : {len(keyframes)} ({keyframes[0]}..{keyframes[-1]})")
    print(f"[c3] distances   : {len(distances)} ({distances[0]}..{distances[-1]} frames)")
    print(f"[c3] value range : {min_v:.3f}..{max_v:.3f}")
    print(f"[c3] wrote       : {matrix_csv}")
    print(f"[c3] wrote       : {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
