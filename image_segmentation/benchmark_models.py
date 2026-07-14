"""Benchmark all trained segmentation checkpoints under runs/: param count,
native-resolution inference time (mean/std over real frames, after warmup),
peak GPU memory. Uses the same SegmentationRunner code path as
adaptive_inference, so numbers are directly comparable to pipeline timings
elsewhere in the repo.

Val mIoU / per-class IoU are NOT computed here -- those come from the
per-epoch training logs at runs/<run>/log.txt (see docs/model_comparison.md
for how to parse them). This script only measures what the logs can't:
actual wall-clock inference cost and memory footprint on this GPU.

    image_segmentation/.venv/bin/python benchmark_models.py
    image_segmentation/.venv/bin/python benchmark_models.py --video path/to/other.mp4
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adaptive_inference.config import load_config  # noqa: E402
from adaptive_inference.models import build_segmentation_model  # noqa: E402
from adaptive_inference.pipeline import SegmentationRunner, decode_video  # noqa: E402

# backbone name -> checkpoint, relative to repo root. Add a line here whenever
# a new run/ directory is trained and should be included in the comparison.
RUNS = {
    "convnext_large":  "image_segmentation/runs/convnext_large_0713_2033/best.pth",
    "swin_large":      "image_segmentation/runs/swin_large_0713_2230/best.pth",
    "hiera_small":     "image_segmentation/runs/hiera_small_0713_2343/best.pth",
    "hiera_base_plus": "image_segmentation/runs/hiera_base_plus_0713_2156/best.pth",
}

DEFAULT_VIDEO = REPO_ROOT / "data/uavid/uavid_v1.5_official_release/uavid_val/seq16/images.mp4"


def benchmark_one(name: str, ckpt: str, frames: list[np.ndarray], n_warmup: int, n_timed: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = load_config(overrides={"segmentation": {"backbone": name, "checkpoint": ckpt}})

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = build_segmentation_model(cfg)
    n_params = next(float(l.split(":")[1].strip().rstrip("M params"))
                     for l in buf.getvalue().splitlines() if "params" in l)

    infer = SegmentationRunner(model, cfg.adaptive.device)
    for i in range(n_warmup):
        infer(frames[i])

    times = np.array([infer(frames[n_warmup + i])[1] for i in range(n_timed)])
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    del model, infer
    torch.cuda.empty_cache()
    return dict(params_m=n_params, mean_s=float(times.mean()), std_s=float(times.std()),
                min_s=float(times.min()), max_s=float(times.max()), peak_mem_gb=peak_mem_gb)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(DEFAULT_VIDEO))
    ap.add_argument("--n-warmup", type=int, default=3)
    ap.add_argument("--n-timed", type=int, default=10)
    ap.add_argument("--out", default=None, help="optional path to write results as JSON")
    args = ap.parse_args()

    frames = decode_video(args.video, n_frames=args.n_warmup + args.n_timed)
    print(f"decoded {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}\n")

    results = {}
    for name, ckpt in RUNS.items():
        r = benchmark_one(name, ckpt, frames, args.n_warmup, args.n_timed)
        results[name] = r
        print(f"{name:20s} params={r['params_m']:6.1f}M  "
              f"infer={r['mean_s']*1000:7.1f}ms +/- {r['std_s']*1000:5.1f}ms  "
              f"(min={r['min_s']*1000:.1f} max={r['max_s']*1000:.1f})  "
              f"peak_mem={r['peak_mem_gb']:.2f}GB")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
