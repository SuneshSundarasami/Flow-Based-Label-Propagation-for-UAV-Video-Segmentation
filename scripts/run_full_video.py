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
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import load_config  # noqa: E402
from data import RuralscapesVideo  # noqa: E402
from eval import concat_csvs, keyframe_rows, write_csv  # noqa: E402
from flow import SeaRaftFlow  # noqa: E402
from propagation import forward_targets, propagate_keyframe  # noqa: E402

_VIZ_W, _VIZ_H = 960, 540


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

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("[c1] loading dataset ...")
    video = RuralscapesVideo(
        root=args.root,
        frame_glob=args.frame_glob,
        mask_glob=args.mask_glob,
        mask_format=args.mask_format,
    )
    ann = video.annotated_indices
    if len(ann) < 2:
        print(f"[c1] need at least 2 annotated frames; found {len(ann)}")
        return 1

    num_classes = video.num_classes
    keyframes = ann[:-1]
    if args.limit is not None:
        keyframes = keyframes[: args.limit]

    fb_threshold = cfg["occlusion"]["fb_threshold"]
    ignore_index = cfg["eval"]["ignore_index"]

    # ------------------------------------------------------------------
    # Startup summary
    # ------------------------------------------------------------------
    def _csv_path(k: int) -> Path:
        return results_dir / f"keyframe_{k}.csv"

    todo = [k for k in keyframes if args.force or not _csv_path(k).exists()]
    n_cached = len(keyframes) - len(todo)

    print(f"[c1] video        : {video.summary()}")
    print(f"[c1] annotated    : {len(ann)} frames  (first={ann[0]}, last={ann[-1]})")
    print(f"[c1] keyframes    : {len(keyframes)} to process  (limit={args.limit})")
    print(f"[c1] n_targets    : {args.n_targets} per keyframe")
    print(f"[c1] classes      : {num_classes}  fb_threshold={fb_threshold}px  ignore={ignore_index}")
    print(f"[c1] output dir   : {results_dir}")
    print(f"[c1] cache        : {n_cached} already done, {len(todo)} to compute  (--force={args.force})")
    print(f"[c1] device       : {args.device}")
    print()

    # ------------------------------------------------------------------
    # Load model (only if there's work to do)
    # ------------------------------------------------------------------
    model = None
    if todo:
        print(f"[c1] loading SEA-RAFT ({cfg['flow']['model_cfg']}) on {args.device} ...")
        t_load = time.time()
        model = SeaRaftFlow(
            cfg["flow"]["model_cfg"],
            checkpoint=str(_REPO_ROOT / cfg["paths"]["sea_raft_checkpoint"]),
            iters=cfg["flow"]["iters"],
            device=args.device,
        )
        print(f"[c1] model loaded in {time.time() - t_load:.1f}s\n")

    # ------------------------------------------------------------------
    # Main loop — outer tqdm over keyframes, inner tqdm over targets
    # ------------------------------------------------------------------
    t0 = time.time()
    kf_bar = tqdm(todo, desc="keyframes", unit="kf",
                  dynamic_ncols=True, position=0)

    for kf_idx in kf_bar:
        targets = forward_targets(ann, kf_idx, args.n_targets)
        if not targets:
            kf_bar.write(f"[c1] kf={kf_idx}: no forward targets, skipping")
            continue

        kf_bar.set_postfix_str(f"kf={kf_idx} targets={targets}")

        # Load keyframe
        kf_frame = _resize(video.load_frame(kf_idx), _VIZ_W, _VIZ_H)
        kf_mask = _resize(video.load_mask(kf_idx), _VIZ_W, _VIZ_H,
                          interp=cv2.INTER_NEAREST)

        # Inner loop — one target at a time so tqdm can tick per flow call
        all_results = []
        tgt_bar = tqdm(
            targets,
            desc=f"  kf {kf_idx:>5}",
            unit="tgt",
            leave=False,
            dynamic_ncols=True,
            position=1,
        )
        for t_idx in tgt_bar:
            tgt_bar.set_postfix_str(f"tgt={t_idx}  dist={abs(t_idx - kf_idx)}")

            tgt_frame = _resize(video.load_frame(t_idx), _VIZ_W, _VIZ_H)
            gt_mask = (
                _resize(video.load_mask(t_idx), _VIZ_W, _VIZ_H,
                        interp=cv2.INTER_NEAREST)
                if video.has_mask(t_idx) else None
            )

            result = propagate_keyframe(
                keyframe_mask=kf_mask,
                keyframe_frame=kf_frame,
                target_frames=[tgt_frame],
                target_indices=[t_idx],
                keyframe_index=kf_idx,
                model=model,
                fb_threshold=fb_threshold,
                ignore_index=ignore_index,
                gt_masks=[gt_mask],
                num_classes=num_classes,
            )[0]
            all_results.append(result)

            miou_v = result.iou["valid_only"]["miou"] if result.iou else float("nan")
            tgt_bar.set_postfix_str(
                f"tgt={t_idx}  dist={result.distance}  "
                f"valid={result.valid_pct:.1f}%  mIoU={miou_v:.3f}"
            )

        tgt_bar.close()

        # Write per-keyframe CSV
        rows = keyframe_rows(all_results, kf_idx, num_classes)
        write_csv(_csv_path(kf_idx), rows)

        # Summary line for this keyframe (printed above the tqdm bar)
        mious = [r.iou["valid_only"]["miou"] for r in all_results
                 if r.iou and not np.isnan(r.iou["valid_only"]["miou"])]
        mean_miou = f"{np.mean(mious):.3f}" if mious else "n/a"
        elapsed = time.time() - t0
        kf_bar.write(
            f"[c1] kf={kf_idx:>5}  targets={targets}  "
            f"mean mIoU(valid)={mean_miou}  elapsed={elapsed:.0f}s"
        )

    kf_bar.close()

    # ------------------------------------------------------------------
    # Concatenate all keyframe CSVs into the full-video table
    # ------------------------------------------------------------------
    print()
    print("[c1] concatenating per-keyframe CSVs ...")
    csv_paths = [_csv_path(k) for k in keyframes if _csv_path(k).exists()]
    all_csv = results_dir / "all_pairs.csv"
    n_rows = concat_csvs(csv_paths, all_csv)
    total = time.time() - t0
    print(f"[c1] wrote {all_csv}")
    print(f"[c1] {n_rows} pairs from {len(csv_paths)} keyframes  total={total:.0f}s")
    print("[c1] done — full-video table ready for C2–C5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
