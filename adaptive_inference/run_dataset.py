"""Metrics-only batch runner: for every UAVid sequence, run BOTH the
baseline (ConvNeXt-Large, one frame at a time, matching how it's actually
served) and the adaptive hybrid pipeline, and write one CSV per sequence.
No mask images, no video output -- just per-frame timing/quality numbers.

Models are loaded once and reused across every sequence; only frames +
results are held per-video, so peak memory doesn't grow with dataset size.

    python -m adaptive_inference.run_dataset --seq seq16 --max-frames 30   # smoke test
    python -m adaptive_inference.run_dataset                                # full dataset

Output: <out_dir>/<split>/<seq>.csv with columns
    frame, baseline_elapsed_s, baseline_miou,
    hybrid_source, hybrid_elapsed_s, hybrid_valid_pct, hybrid_miou
(same schema as visualizations/create_adaptive_hybrid_comparison_video.py's
CSV, minus the video). miou columns are blank for frames/sequences with no
matching GT (all of uavid_test, and any non-keyframe in train/val).
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterator

from .config import REPO_ROOT, load_config
from .models import build_flow_model, build_segmentation_model, class_names, ignore_index
from .pipeline import SegmentationRunner, decode_video, run_adaptive_pipeline
from .run import compute_miou, load_gt

DEFAULT_ROOT = REPO_ROOT / "data/uavid/uavid_v1.5_official_release"
DEFAULT_SPLITS = ("uavid_train", "uavid_val", "uavid_test")


def discover_sequences(root: Path, splits: tuple[str, ...], only_seq: str | None):
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for seq_dir in sorted(split_dir.iterdir()):
            if only_seq is not None and seq_dir.name != only_seq:
                continue
            video = seq_dir / "images.mp4"
            if video.exists():
                gt_dir = seq_dir / "Labels"
                yield split, seq_dir.name, video, gt_dir if gt_dir.is_dir() else None


def process_video(video_path, gt_dir, infer, seg_model, flow_model, cfg, num_classes, ignore_id,
                   max_frames: int | None) -> Iterator[dict]:
    frames = decode_video(video_path, n_frames=max_frames)
    for frame, hybrid_result in zip(frames, run_adaptive_pipeline(frames, seg_model, flow_model, cfg)):
        i = hybrid_result.frame_index
        baseline_mask, baseline_t = infer(frame)

        gt = None
        if gt_dir is not None:
            gt_path = gt_dir / f"{i:06d}.png"
            if gt_path.exists():
                gt = load_gt(gt_path)

        baseline_miou = compute_miou(baseline_mask, gt, num_classes, ignore_id) if gt is not None else None
        hybrid_miou = compute_miou(hybrid_result.mask, gt, num_classes, ignore_id) if gt is not None else None

        yield {
            "frame": i,
            "baseline_elapsed_s": f"{baseline_t:.4f}",
            "baseline_miou": "" if baseline_miou is None else f"{baseline_miou:.4f}",
            "hybrid_source": hybrid_result.source,
            "hybrid_elapsed_s": f"{hybrid_result.elapsed_s:.4f}",
            "hybrid_valid_pct": f"{hybrid_result.valid_pct:.4f}",
            "hybrid_miou": "" if hybrid_miou is None else f"{hybrid_miou:.4f}",
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="UAVid official-release root")
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="comma-separated split dir names")
    ap.add_argument("--seq", default=None, help="restrict to one sequence name (e.g. seq16) -- for smoke testing")
    ap.add_argument("--config", default=None, help="YAML overrides merged onto config.yaml")
    ap.add_argument("--threshold", type=float, default=None, help="override adaptive.valid_threshold")
    ap.add_argument("--max-frames", type=int, default=None, help="decode only the first N frames per video")
    ap.add_argument("--out-dir", default=None, help="default: outputs/adaptive_inference_metrics")
    ap.add_argument("--overwrite", action="store_true", help="reprocess sequences whose CSV already exists")
    args = ap.parse_args()

    overrides: dict = {}
    if args.threshold is not None:
        overrides.setdefault("adaptive", {})["valid_threshold"] = args.threshold
    cfg = load_config(args.config, overrides)

    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "outputs" / "adaptive_inference_metrics"
    root = Path(args.root)
    splits = tuple(args.splits.split(","))

    sequences = list(discover_sequences(root, splits, args.seq))
    if not sequences:
        print(f"no sequences found under {root} for splits {splits} (seq filter: {args.seq!r})")
        return 1
    print(f"found {len(sequences)} sequence(s): {[s[1] for s in sequences]}")

    print(f"loading models (segmentation={cfg.segmentation.backbone}, flow={cfg.flow.backend}, "
          f"threshold={cfg.adaptive.valid_threshold})...")
    seg_model = build_segmentation_model(cfg)
    flow_model = build_flow_model(cfg)
    infer = SegmentationRunner(seg_model, cfg.adaptive.device)
    num_classes = len(class_names())
    ignore_id = ignore_index()

    for split, seq_name, video_path, gt_dir in sequences:
        csv_path = out_dir / split / f"{seq_name}.csv"
        if csv_path.exists() and not args.overwrite:
            print(f"[{split}/{seq_name}] skip (csv exists: {csv_path})")
            continue

        print(f"[{split}/{seq_name}] decoding {video_path} ...")
        t0 = time.time()
        rows = []
        for row in process_video(video_path, gt_dir, infer, seg_model, flow_model, cfg,
                                  num_classes, ignore_id, args.max_frames):
            rows.append(row)
            if row["frame"] % 100 == 0:
                print(f"[{split}/{seq_name}]   frame {row['frame']}   elapsed={time.time()-t0:.1f}s", flush=True)

        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        n_model_calls = sum(1 for r in rows if r["hybrid_source"] == "model")
        print(f"[{split}/{seq_name}] {len(rows)} frames in {time.time()-t0:.1f}s "
              f"({n_model_calls} hybrid model calls, {len(rows)-n_model_calls} flow-propagated) -> {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
