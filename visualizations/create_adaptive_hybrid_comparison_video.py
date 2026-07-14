"""Side-by-side comparison video: ConvNeXt-Large run on every frame vs.
the adaptive hybrid pipeline (model keyframe + flow propagation, re-inferring
once confidence drops below a threshold), both overlaid on the original
UAVid footage. Each panel is rendered at true 1920x1080 (native UAVid frames
are exactly 3840x2160 = 2x that, so no aspect-ratio distortion); the two
panels sit side by side, giving a 3840x1080 output canvas.

    python visualizations/create_adaptive_hybrid_comparison_video.py \
        --video data/uavid/uavid_v1.5_official_release/uavid_val/seq16/images.mp4 \
        --gt-dir data/uavid/uavid_v1.5_official_release/uavid_val/seq16/Labels \
        --max-frames 150

All the actual segmentation/flow-propagation logic lives in
adaptive_inference/ -- this script only decodes, runs both methods per
frame, composites, and writes video + a per-frame CSV. The three compositing
helpers below (_overlay/_text/_resize_rgb) are copied verbatim from
create_method_comparison_video.py rather than importing that module: it
unconditionally does `from data import RuralscapesVideo` at module load,
which permanently binds the shared `data` module name to the Ruralscapes
package for the rest of the process (see adaptive_inference/models.py's
module-name-collision note) -- incompatible with also using
adaptive_inference in the same interpreter. The helpers themselves are
~12 lines of generic cv2/numpy compositing with no dataset-specific logic.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from adaptive_inference.config import load_config  # noqa: E402
from adaptive_inference.models import (  # noqa: E402
    build_flow_model, build_segmentation_model, class_names, ignore_index,
)
from adaptive_inference.pipeline import SegmentationRunner, decode_video, run_adaptive_pipeline  # noqa: E402
from adaptive_inference.run import colorize_mask, compute_miou, load_gt  # noqa: E402


def _overlay(frame_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * frame_rgb + alpha * mask_rgb).astype(np.uint8)


def _text(panel_rgb: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(panel, label, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbor resize for a class-id mask (INTER_AREA would blend ids)."""
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="path to a UAVid sequence's images.mp4")
    ap.add_argument("--gt-dir", default=None, help="directory of <frame_index>.png GT labels")
    ap.add_argument("--config", default=None, help="YAML overrides merged onto adaptive_inference/config.yaml")
    ap.add_argument("--threshold", type=float, default=None, help="override adaptive.valid_threshold")
    ap.add_argument("--max-frames", type=int, default=None, help="decode/render only the first N frames")
    ap.add_argument("--out", default=None, help="output .mp4 path (default: visualizations/<seq>/adaptive_hybrid_comparison.mp4)")
    ap.add_argument("--fps", type=float, default=20.0, help="output video fps (source UAVid is 20fps)")
    ap.add_argument("--panel-width", type=int, default=1920, help="per-panel width; height is derived to match the source's 16:9 aspect")
    ap.add_argument("--panel-height", type=int, default=None, help="override the auto-derived (16:9) panel height")
    ap.add_argument("--alpha", type=float, default=0.45, help="segmentation overlay opacity")
    args = ap.parse_args()

    overrides: dict = {}
    if args.threshold is not None:
        overrides.setdefault("adaptive", {})["valid_threshold"] = args.threshold
    cfg = load_config(args.config, overrides)

    print(f"loading models (segmentation={cfg.segmentation.backbone}, flow={cfg.flow.backend}, "
          f"threshold={cfg.adaptive.valid_threshold})...")
    seg_model = build_segmentation_model(cfg)
    flow_model = build_flow_model(cfg)
    infer = SegmentationRunner(seg_model, cfg.adaptive.device)

    print(f"decoding {args.video}...")
    frames = decode_video(args.video, n_frames=args.max_frames)
    print(f"  {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")

    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    num_classes = len(class_names())
    ignore_id = ignore_index()

    seq_name = Path(args.video).resolve().parent.name
    out_path = Path(args.out) if args.out else _REPO_ROOT / "visualizations" / seq_name / "adaptive_hybrid_comparison.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src_h, src_w = frames[0].shape[:2]
    w = args.panel_width
    h = args.panel_height if args.panel_height is not None else round(w * src_h / src_w)
    print(f"panel size {w}x{h} (source aspect {src_w}x{src_h} preserved), "
          f"output canvas {w*2}x{h} (side by side)")
    video_writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w * 2, h),
    )
    if not video_writer.isOpened():
        raise RuntimeError(f"could not open video writer: {out_path}")

    print(f"rendering {len(frames)} frames (both methods) -> {out_path} ...")
    t0 = time.time()
    rows = []
    try:
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

            # Resize (mask: nearest-neighbor, frame: area) down to panel size *before*
            # colorizing/blending -- both are only ever displayed at panel resolution,
            # so blending at native 4K first wastes ~4x the necessary compute per panel.
            frame_panel = _resize_rgb(frame, w, h)
            baseline_panel = _overlay(frame_panel, colorize_mask(_resize_mask(baseline_mask, w, h)), args.alpha)
            hybrid_panel = _overlay(frame_panel, colorize_mask(_resize_mask(hybrid_result.mask, w, h)), args.alpha)

            baseline_label = f"ConvNeXt-Large, every frame   frame={i}"
            if baseline_miou is not None:
                baseline_label += f"   mIoU={baseline_miou:.3f}"
            hybrid_label = (f"Adaptive hybrid (thr={cfg.adaptive.valid_threshold:.2f})   frame={i}   "
                             f"src={hybrid_result.source}   coverage={hybrid_result.valid_pct*100:.0f}%")
            if hybrid_miou is not None:
                hybrid_label += f"   mIoU={hybrid_miou:.3f}"

            left = _text(baseline_panel, baseline_label)
            right = _text(hybrid_panel, hybrid_label)
            video_writer.write(np.concatenate([left, right], axis=1))

            rows.append({
                "frame": i,
                "baseline_elapsed_s": f"{baseline_t:.4f}",
                "baseline_miou": "" if baseline_miou is None else f"{baseline_miou:.4f}",
                "hybrid_source": hybrid_result.source,
                "hybrid_elapsed_s": f"{hybrid_result.elapsed_s:.4f}",
                "hybrid_valid_pct": f"{hybrid_result.valid_pct:.4f}",
                "hybrid_miou": "" if hybrid_miou is None else f"{hybrid_miou:.4f}",
            })
            if i % 50 == 0:
                print(f"  frame {i}/{len(frames)}   elapsed={time.time()-t0:.1f}s", flush=True)
    finally:
        video_writer.release()

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_model_calls = sum(1 for r in rows if r["hybrid_source"] == "model")
    print(f"\n{len(frames)} frames rendered in {time.time()-t0:.1f}s "
          f"({n_model_calls} hybrid model calls, {len(frames)-n_model_calls} flow-propagated)")
    print(f"wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB) and {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
