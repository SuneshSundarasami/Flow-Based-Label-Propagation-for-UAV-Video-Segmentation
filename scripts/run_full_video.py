"""C1 full-video batched propagation across all keyframes.

For every annotated keyframe in the video, propagate its mask forward to the
next N annotated frames (the only frames with ground truth for IoU) and record
per-pair IoU. Each keyframe's results are cached to
``outputs/results/keyframe_{k}.csv``; on rerun, keyframes whose CSV already
exists are skipped (pass ``--force`` to recompute). All per-keyframe CSVs are
then concatenated into ``outputs/results/all_pairs.csv`` — the complete
full-video table consumed by C2–C5.

All settings default from ``src/config/default.yaml``; every flag is optional.

Usage
-----
    # full run with config defaults
    python scripts/run_full_video.py

    # quick partial run (first 3 keyframes), CPU
    python scripts/run_full_video.py --limit 3 --device cpu

    # recompute everything from scratch
    python scripts/run_full_video.py --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import load_config  # noqa: E402
from data import RuralscapesVideo  # noqa: E402
from eval import concat_csvs, keyframe_rows, write_csv  # noqa: E402
from flow import SeaRaftFlow  # noqa: E402
from propagation import forward_targets, propagate_keyframe  # noqa: E402

_VIZ_W, _VIZ_H = 960, 540


def _pre_load_config(argv=None) -> dict:
    """Load config early so argparse can use config values as defaults."""
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


def _resize(img: np.ndarray, w: int, h: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=interp)


def main() -> int:
    cfg = _pre_load_config()

    ap = argparse.ArgumentParser(
        description="C1: batched keyframe propagation over the whole video."
    )
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="Override YAML deep-merged on top of default.yaml.")
    ap.add_argument("--root", default=cfg["paths"]["dataset_root"])
    ap.add_argument("--frame-glob", default=cfg["data"]["frame_glob"])
    ap.add_argument("--mask-glob", default=cfg["data"]["mask_glob"])
    ap.add_argument("--mask-format", default=cfg["data"]["mask_format"],
                    choices=["color", "indexed"])
    ap.add_argument("--n-targets", type=int, default=cfg["propagation"]["n_targets"],
                    help="Annotated frames after each keyframe to evaluate. [default: %(default)s]")
    ap.add_argument("--device", default=cfg["flow"]["device"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N keyframes (quick partial run).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute keyframes even if their CSV already exists.")
    args = ap.parse_args()

    results_dir = _REPO_ROOT / cfg["paths"]["output_dir"] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    video = RuralscapesVideo(
        root=args.root,
        frame_glob=args.frame_glob,
        mask_glob=args.mask_glob,
        mask_format=args.mask_format,
    )
    ann = video.annotated_indices
    if len(ann) < 2:
        print("[c1] need at least 2 annotated frames; found", len(ann))
        return 1

    num_classes = video.num_classes
    keyframes = ann[:-1]  # last annotated frame has no forward targets
    if args.limit is not None:
        keyframes = keyframes[: args.limit]

    fb_threshold = cfg["occlusion"]["fb_threshold"]
    ignore_index = cfg["eval"]["ignore_index"]

    print(f"[c1] {len(ann)} annotated frames; processing {len(keyframes)} keyframes "
          f"(n_targets={args.n_targets}, classes={num_classes})")

    # ------------------------------------------------------------------
    # Decide which keyframes actually need computing (caching).
    # ------------------------------------------------------------------
    def _csv_path(k: int) -> Path:
        return results_dir / f"keyframe_{k}.csv"

    todo = [k for k in keyframes if args.force or not _csv_path(k).exists()]
    cached = [k for k in keyframes if k not in todo]
    print(f"[c1] {len(cached)} cached, {len(todo)} to compute")

    model = None
    if todo:
        print(f"[c1] loading SEA-RAFT on {args.device}")
        model = SeaRaftFlow(
            cfg["flow"]["model_cfg"],
            checkpoint=str(_REPO_ROOT / cfg["paths"]["sea_raft_checkpoint"]),
            iters=cfg["flow"]["iters"],
            device=args.device,
        )

    t0 = time.time()
    for n, kf_idx in enumerate(todo, 1):
        targets = forward_targets(ann, kf_idx, args.n_targets)
        if not targets:
            continue

        kf_frame = _resize(video.load_frame(kf_idx), _VIZ_W, _VIZ_H)
        kf_mask = _resize(video.load_mask(kf_idx), _VIZ_W, _VIZ_H,
                          interp=cv2.INTER_NEAREST)

        tgt_frames, tgt_gt = [], []
        for t_idx in targets:
            tgt_frames.append(_resize(video.load_frame(t_idx), _VIZ_W, _VIZ_H))
            tgt_gt.append(
                _resize(video.load_mask(t_idx), _VIZ_W, _VIZ_H,
                        interp=cv2.INTER_NEAREST)
                if video.has_mask(t_idx) else None
            )

        results = propagate_keyframe(
            keyframe_mask=kf_mask,
            keyframe_frame=kf_frame,
            target_frames=tgt_frames,
            target_indices=targets,
            keyframe_index=kf_idx,
            model=model,
            fb_threshold=fb_threshold,
            ignore_index=ignore_index,
            gt_masks=tgt_gt,
            num_classes=num_classes,
        )

        rows = keyframe_rows(results, kf_idx, num_classes)
        write_csv(_csv_path(kf_idx), rows)

        elapsed = time.time() - t0
        eta = elapsed / n * (len(todo) - n)
        miou = [float(r["miou_valid"]) for r in rows if r["miou_valid"] != "nan"]
        mean_miou = f"{np.mean(miou):.3f}" if miou else "n/a"
        print(f"[c1] [{n}/{len(todo)}] kf={kf_idx} -> {targets} "
              f"mean mIoU(valid)={mean_miou}  (ETA {eta:6.0f}s)")

    # ------------------------------------------------------------------
    # Concatenate all keyframe CSVs into the full-video table.
    # ------------------------------------------------------------------
    csv_paths = [_csv_path(k) for k in keyframes if _csv_path(k).exists()]
    all_csv = results_dir / "all_pairs.csv"
    n_rows = concat_csvs(csv_paths, all_csv)
    print(f"[c1] wrote {all_csv} ({n_rows} pairs from {len(csv_paths)} keyframes)")
    print("[c1] done — full-video table ready for C2–C5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
