"""CLI entry point for the adaptive hybrid pipeline.

    python -m adaptive_inference.run --video path/to/images.mp4
    python -m adaptive_inference.run --video seq16.mp4 --config my_overrides.yaml
    python -m adaptive_inference.run --video seq16.mp4 --gt-dir path/to/Labels \
        --threshold 0.90 --max-frames 200

Writes one label PNG per frame to <output.out_dir>/<video-stem>/, plus a
per-frame CSV (frame index, source, valid_pct, elapsed_s) and, if --gt-dir is
given, mIoU at every frame with a matching GT file.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .config import REPO_ROOT, load_config
from .models import build_flow_model, build_segmentation_model, class_names, ignore_index
from .pipeline import decode_video, run_adaptive_pipeline

PALETTE = [(128, 0, 128), (112, 148, 32), (64, 64, 0), (255, 16, 255), (0, 128, 128), (255, 0, 0)]


def colorize_mask(mask: np.ndarray, palette: list[tuple[int, int, int]] = PALETTE) -> np.ndarray:
    """(H, W) class-id mask -> (H, W, 3) RGB visualization via a LUT lookup

    (~2.4x faster than looping over classes with boolean masks at 4K).
    """
    lut = np.zeros((256, 3), np.uint8)
    lut[:len(palette)] = palette
    return lut[mask]


def save_mask(mask: np.ndarray, path: Path, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path.with_suffix(".npy"), mask)
        return
    if fmt == "grayscale_png":
        Image.fromarray(mask).convert("L").save(path, optimize=True)
        return
    if fmt == "palette_png":
        img = Image.fromarray(mask).convert("P")
        flat_palette = [v for c in PALETTE for v in c]
        img.putpalette(flat_palette + [0] * (768 - len(flat_palette)))
        img.save(path, optimize=True)
        return
    if fmt == "rgb_png":
        Image.fromarray(colorize_mask(mask)).save(path, optimize=True)
        return
    raise ValueError(f"unknown output.format: {fmt!r}")


def compute_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int, ignore_id: int) -> float:
    from eval import compute_iou  # src/eval
    return compute_iou(pred.astype(np.int64), gt.astype(np.int64), num_classes,
                        ignore_index=ignore_id)["all"]["miou"]


def load_gt(path: Path) -> np.ndarray:
    from data import rgb_label_to_ids  # image_segmentation/data.py
    return rgb_label_to_ids(np.asarray(Image.open(path).convert("RGB")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="path to a video file")
    ap.add_argument("--config", default=None, help="YAML overrides merged onto config.yaml")
    ap.add_argument("--threshold", type=float, default=None, help="override adaptive.valid_threshold")
    ap.add_argument("--backbone", default=None, help="override segmentation.backbone")
    ap.add_argument("--flow-backend", default=None, choices=["sea_raft", "flownet2"])
    ap.add_argument("--max-frames", type=int, default=None, help="decode only the first N frames")
    ap.add_argument("--gt-dir", default=None,
                     help="directory of <frame_index>.png GT labels for quality checking")
    ap.add_argument("--out-dir", default=None, help="override output.out_dir")
    args = ap.parse_args()

    overrides: dict = {}
    if args.threshold is not None:
        overrides.setdefault("adaptive", {})["valid_threshold"] = args.threshold
    if args.backbone is not None:
        overrides.setdefault("segmentation", {})["backbone"] = args.backbone
    if args.flow_backend is not None:
        overrides.setdefault("flow", {})["backend"] = args.flow_backend
    if args.out_dir is not None:
        overrides.setdefault("output", {})["out_dir"] = args.out_dir

    cfg = load_config(args.config, overrides)

    print(f"loading models (segmentation={cfg.segmentation.backbone}, "
          f"flow={cfg.flow.backend}, threshold={cfg.adaptive.valid_threshold})...")
    seg_model = build_segmentation_model(cfg)
    flow_model = build_flow_model(cfg)

    print(f"decoding {args.video}...")
    t0 = time.time()
    frames = decode_video(args.video, n_frames=args.max_frames)
    print(f"  {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, "
          f"decoded in {time.time()-t0:.2f}s")

    out_dir = cfg.resolve(cfg.output.out_dir) / Path(args.video).stem
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    num_classes = len(class_names())
    ignore_id = ignore_index()

    rows = []
    model_calls = 0
    t_start = time.time()
    for result in run_adaptive_pipeline(frames, seg_model, flow_model, cfg):
        if cfg.output.save_masks:
            save_mask(result.mask, out_dir / f"{result.frame_index:06d}.png", cfg.output.format)

        miou = None
        if gt_dir is not None:
            gt_path = gt_dir / f"{result.frame_index:06d}.png"
            if gt_path.exists():
                miou = compute_miou(result.mask, load_gt(gt_path), num_classes, ignore_id)

        if result.source == "model":
            model_calls += 1
        rows.append({
            "frame": result.frame_index, "source": result.source,
            "valid_pct": f"{result.valid_pct:.4f}", "elapsed_s": f"{result.elapsed_s:.4f}",
            "miou": "" if miou is None else f"{miou:.4f}",
        })
        if miou is not None:
            print(f"  frame {result.frame_index:>5d} ({result.source:>5s}): mIoU={miou:.4f}")

    t_total = time.time() - t_start

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(frames)} frames: {model_calls} model calls, "
          f"{len(frames) - model_calls} flow-propagated, total {t_total:.1f}s")
    print(f"wrote masks + {csv_path.name} to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
