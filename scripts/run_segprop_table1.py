"""Run the SegProp Table 1 pipeline on prepared Ruralscapes data."""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[1]

TABLE1_EXPECTED = {
    "i01": (0.884, 0.801),
    "i02": (0.894, 0.817),
    "i03": (0.896, 0.819),
    "i04": (0.897, 0.821),
    "i05": (0.897, 0.821),
    "i06": (0.897, 0.821),
    "i07": (0.897, 0.821),
    "filtered": (0.903, 0.829),
}


@dataclass(frozen=True)
class SegPropPaths:
    out_root: Path
    labels_even: Path
    labels_odd: Path
    flow_forward: Path
    flow_backward: Path
    output_root: Path


def frame_copy(index: int) -> bool:
    return index % 100 == 0


def frame_filter(index: int) -> bool:
    return (index % 50 != 0) or (index % 100 == 0)


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _stage_name(iteration: int) -> str:
    return f"i{iteration:02d}"


def paths_for_video(out_root: Path, video: str) -> SegPropPaths:
    return SegPropPaths(
        out_root=out_root,
        labels_even=out_root / "labels_2k" / "train_even" / video,
        labels_odd=out_root / "labels_2k" / "train_odd" / video,
        flow_forward=out_root / "flow_2k_fn2" / f"{video}_forward.h5",
        flow_backward=out_root / "flow_2k_fn2" / f"{video}_backward.h5",
        output_root=out_root / "output_2k",
    )


def validate_inputs(paths: SegPropPaths) -> None:
    required = [
        paths.labels_even,
        paths.labels_odd,
        paths.flow_forward,
        paths.flow_backward,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing SegProp Table 1 inputs:\n{joined}")
    if not any(paths.labels_even.glob("*.npz")):
        raise FileNotFoundError(f"no train_even labels found in {paths.labels_even}")
    if not any(paths.labels_odd.glob("*.npz")):
        raise FileNotFoundError(f"no train_odd labels found in {paths.labels_odd}")


def import_segprop_modules(segprop_root: Path) -> tuple[ModuleType, ModuleType]:
    if not (segprop_root / "segprop.py").exists():
        raise FileNotFoundError(f"SegProp checkout not found at {segprop_root}")
    sys.path.insert(0, str(segprop_root))
    import segprop  # type: ignore  # noqa: E402
    import stats  # type: ignore  # noqa: E402
    return segprop, stats


def run_video_table1(
    *,
    video: str,
    paths: SegPropPaths,
    segprop_module,
    iterations: int,
    device: str,
    overwrite: bool,
    prune_intermediate: bool,
) -> Path:
    validate_inputs(paths)
    flow_h5 = (str(paths.flow_forward), str(paths.flow_backward))

    first_out = paths.output_root / "i01" / video
    segprop_module.vote(
        flow_h5,
        str(paths.labels_even),
        str(first_out),
        precalc_flow=False,
        device=device,
        overwrite=overwrite,
    )

    for iteration in range(2, iterations + 1):
        prev = paths.output_root / _stage_name(iteration - 1) / video
        cur = paths.output_root / _stage_name(iteration) / video
        segprop_module.iterate(
            flow_h5,
            str(prev),
            str(cur),
            pv_series=[0, 5, 10],
            frame_copy=frame_copy,
            device=device,
            overwrite=overwrite,
        )

    final_stage = paths.output_root / _stage_name(iterations) / video
    filtered = paths.output_root / "filtered" / video
    segprop_module.denoise(
        flow_h5,
        str(final_stage),
        str(filtered),
        pv_series=[0, 1, 3, 5, 7],
        method="self",
        frame_filter=frame_filter,
        device=device,
        overwrite=overwrite,
    )

    if prune_intermediate:
        for iteration in range(1, iterations):
            shutil.rmtree(paths.output_root / _stage_name(iteration) / video, ignore_errors=True)

    return filtered


def evaluate_table1(
    *,
    out_root: Path,
    stats_module,
    max_iteration: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    out_root = out_root.resolve()
    label_root = out_root / "labels_2k" / "train_odd"
    for stage in [_stage_name(i) for i in range(1, max_iteration + 1)] + ["filtered"]:
        stage_root = out_root / "output_2k" / stage
        if not stage_root.exists():
            continue
        fmeasure, miou = stats_module.evaluate(str(stage_root), str(label_root))
        expected = TABLE1_EXPECTED.get(stage)
        rows.append({
            "stage": stage,
            "reproduced_mf1": f"{float(fmeasure):.6f}",
            "reproduced_miou": f"{float(miou):.6f}",
            "paper_mf1": "" if expected is None else f"{expected[0]:.6f}",
            "paper_miou": "" if expected is None else f"{expected[1]:.6f}",
            "delta_mf1": "" if expected is None else f"{float(fmeasure) - expected[0]:.6f}",
            "delta_miou": "" if expected is None else f"{float(miou) - expected[1]:.6f}",
        })
    return rows


def write_table(rows: list[dict[str, str]], out_root: Path) -> tuple[Path, Path]:
    csv_path = out_root / "table1_reproduction.csv"
    md_path = out_root / "table1_reproduction.md"
    if not rows:
        raise ValueError("no Table 1 rows to write")
    fieldnames = list(rows[0])
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(md_path, "w") as fh:
        fh.write("| " + " | ".join(fieldnames) + " |\n")
        fh.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in rows:
            fh.write("| " + " | ".join(row[name] for name in fieldnames) + " |\n")
    return csv_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SegProp Table 1 vote/iterate/denoise pipeline.")
    parser.add_argument("--out-root", default="outputs/segprop_paper_repro")
    parser.add_argument("--segprop-root", default="third_party/segprop")
    parser.add_argument("--split-file", default="data/Ruralscapes/ruralscapes_training_videos.txt")
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prune-intermediate", action="store_true",
                        help="Delete per-video i01..i06 outputs after filtered output is produced.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Run propagation only; do not evaluate/write Table 1 CSV/Markdown.")
    args = parser.parse_args()

    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    out_root = Path(args.out_root)
    videos = args.videos if args.videos else _read_split(Path(args.split_file))
    segprop_module, stats_module = import_segprop_modules(Path(args.segprop_root))

    print(f"[segprop-table1] videos     : {', '.join(videos)}")
    print(f"[segprop-table1] out root   : {out_root}")
    print(f"[segprop-table1] iterations : {args.iterations}")
    print(f"[segprop-table1] device     : {args.device}")

    for video in videos:
        print(f"[segprop-table1] running {video}")
        run_video_table1(
            video=video,
            paths=paths_for_video(out_root, video),
            segprop_module=segprop_module,
            iterations=args.iterations,
            device=args.device,
            overwrite=args.overwrite,
            prune_intermediate=args.prune_intermediate,
        )

    if not args.skip_eval:
        rows = evaluate_table1(out_root=out_root, stats_module=stats_module, max_iteration=args.iterations)
        csv_path, md_path = write_table(rows, out_root)
        print(f"[segprop-table1] wrote {csv_path}")
        print(f"[segprop-table1] wrote {md_path}")

    print("[segprop-table1] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
