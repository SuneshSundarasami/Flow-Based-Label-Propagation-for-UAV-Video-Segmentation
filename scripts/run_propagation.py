"""B4 end-to-end single-keyframe propagation with mIoU evaluation.

Loads the first annotated frame from a Ruralscapes video as the keyframe,
propagates its mask to subsequent annotated frames via SEA-RAFT optical flow,
and reports per-frame and per-class IoU (evaluated on valid pixels only).

Results are printed as a table and saved to outputs/propagation/results.csv.

Usage
-----
    python scripts/run_propagation.py \\
        --root data/Ruralscapes \\
        --frame-glob "frames/DJI_0043/*.jpg" \\
        --mask-glob "labels/manual_labels/DJI_0043/*.png" \\
        --mask-format color \\
        --n-targets 5 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import imageio.v3 as iio  # noqa: E402

from config import load_config  # noqa: E402
from data import RuralscapesVideo  # noqa: E402
from flow import SeaRaftFlow  # noqa: E402
from propagation import propagate_keyframe  # noqa: E402

_VIZ_W, _VIZ_H = 960, 540


def _resize(img: np.ndarray, w: int, h: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=interp)


def _colorize(label: np.ndarray, video: RuralscapesVideo) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cid, rgb in video.palette.rgb.items():
        out[label == cid] = rgb
    return out


def _overlay(frame: np.ndarray, mask_color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return (frame * (1 - alpha) + mask_color * alpha).astype(np.uint8)


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None or math.isnan(v):
        return "  n/a "
    return f"{v * 100 if pct else v:6.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frame-glob", default="frames/DJI_0043/*.jpg")
    ap.add_argument("--mask-glob", default="labels/manual_labels/DJI_0043/*.png")
    ap.add_argument("--mask-format", default="color", choices=["color", "indexed"])
    ap.add_argument("--n-targets", type=int, default=5,
                    help="number of annotated frames after the keyframe to evaluate")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = _REPO_ROOT / "outputs" / "propagation"
    out_dir.mkdir(parents=True, exist_ok=True)

    video = RuralscapesVideo(
        root=args.root,
        frame_glob=args.frame_glob,
        mask_glob=args.mask_glob,
        mask_format=args.mask_format,
    )
    ann = video.annotated_indices
    if len(ann) < 2:
        print("[prop] need at least 2 annotated frames; found", len(ann))
        return 1

    kf_idx = ann[0]
    targets = ann[1: 1 + args.n_targets]
    num_classes = video.num_classes
    class_names = [video.palette.names.get(c, str(c)) for c in range(num_classes)]

    print(f"[prop] keyframe={kf_idx}, targets={targets}, num_classes={num_classes}")
    print(f"[prop] loading SEA-RAFT on {args.device}")

    model = SeaRaftFlow(
        cfg["flow"]["model_cfg"],
        checkpoint=str(_REPO_ROOT / cfg["paths"]["sea_raft_checkpoint"]),
        iters=cfg["flow"]["iters"],
        device=args.device,
    )

    kf_frame_full = video.load_frame(kf_idx)
    kf_mask_full = video.load_mask(kf_idx)
    kf_frame = _resize(kf_frame_full, _VIZ_W, _VIZ_H)
    kf_mask = _resize(kf_mask_full, _VIZ_W, _VIZ_H, interp=cv2.INTER_NEAREST)

    tgt_frames, tgt_gt_masks = [], []
    for t_idx in targets:
        tgt_frames.append(_resize(video.load_frame(t_idx), _VIZ_W, _VIZ_H))
        tgt_gt_masks.append(
            _resize(video.load_mask(t_idx), _VIZ_W, _VIZ_H, interp=cv2.INTER_NEAREST)
            if video.has_mask(t_idx) else None
        )

    fb_threshold = cfg["occlusion"]["fb_threshold"]
    ignore_index = cfg["eval"]["ignore_index"]

    results = propagate_keyframe(
        keyframe_mask=kf_mask,
        keyframe_frame=kf_frame,
        target_frames=tgt_frames,
        target_indices=targets,
        keyframe_index=kf_idx,
        model=model,
        fb_threshold=fb_threshold,
        ignore_index=ignore_index,
        gt_masks=tgt_gt_masks,
        num_classes=num_classes,
    )

    # ------------------------------------------------------------------
    # Print table  (two mIoU columns: all pixels vs valid-only pixels)
    # ------------------------------------------------------------------
    col_w = max(10, max(len(n) for n in class_names))
    header_classes = "  ".join(f"{n:>{col_w}}" for n in class_names)
    print(f"\n{'dist':>6}  {'valid%':>7}  {'mIoU(all)':>10}  {'mIoU(valid)':>11}  {header_classes}")
    sep_w = 6 + 2 + 7 + 2 + 10 + 2 + 11 + 2 + (col_w + 2) * num_classes
    print("-" * sep_w)

    csv_rows = []
    for r in results:
        iou_all = r.iou["all"] if r.iou else None
        iou_valid = r.iou["valid_only"] if r.iou else None

        miou_all_str = _fmt(iou_all["miou"] if iou_all else None)
        miou_valid_str = _fmt(iou_valid["miou"] if iou_valid else None)

        # per-class columns use the valid-only IoU (the primary measure)
        class_strs = []
        for c in range(num_classes):
            v = iou_valid["per_class"].get(c) if iou_valid else None
            class_strs.append(f"{_fmt(v):>{col_w}}")
        class_cols = "  ".join(class_strs)
        print(f"+{r.distance:>5}  {r.valid_pct:>6.1f}%  {miou_all_str:>10}  {miou_valid_str:>11}  {class_cols}")

        def _fv(v):
            return f"{v:.4f}" if not math.isnan(v) else "nan"

        base = {"distance": r.distance, "valid_pct": f"{r.valid_pct:.2f}"}

        row_all = dict(base)
        row_valid = dict(base)
        if r.iou:
            row_all["miou"] = _fv(iou_all["miou"])
            row_valid["miou"] = _fv(iou_valid["miou"])
            for c in range(num_classes):
                row_all[class_names[c]] = _fv(iou_all["per_class"].get(c, float("nan")))
                row_valid[class_names[c]] = _fv(iou_valid["per_class"].get(c, float("nan")))
        csv_rows.append((row_all, row_valid))

    # ------------------------------------------------------------------
    # Save two CSVs
    # ------------------------------------------------------------------
    if csv_rows:
        def _fv(v):
            return f"{v:.4f}" if not math.isnan(v) else "nan"

        rows_all = [r[0] for r in csv_rows]
        rows_valid = [r[1] for r in csv_rows]

        for path, rows in [
            (out_dir / "results_all_pixels.csv", rows_all),
            (out_dir / "results_valid_pixels.csv", rows_valid),
        ]:
            fieldnames = list(rows[0].keys())
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"[prop] saved {path}")

    # ------------------------------------------------------------------
    # Save 4-panel images
    # ------------------------------------------------------------------
    panel_kf = _overlay(kf_frame, _colorize(kf_mask, video))
    cv2.putText(panel_kf, f"keyframe {kf_idx} (GT)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel_kf, f"keyframe {kf_idx} (GT)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    for r, t_idx in zip(results, targets):
        tgt_frame = tgt_frames[targets.index(t_idx)]
        warped_color = _colorize(r.warped_mask, video)
        panel_warp = _overlay(tgt_frame, warped_color)
        panel_gt = _overlay(tgt_frame, _colorize(
            _resize(video.load_mask(t_idx), _VIZ_W, _VIZ_H, interp=cv2.INTER_NEAREST),
            video,
        )) if video.has_mask(t_idx) else tgt_frame.copy()

        validity_vis = tgt_frame.copy()
        validity_vis[~r.valid_mask] = [180, 30, 30]

        _miou_all = r.iou["all"]["miou"] if r.iou else float("nan")
        _miou_v = r.iou["valid_only"]["miou"] if r.iou else float("nan")
        miou_label = (
            f"mIoU all={_miou_all:.3f}  valid={_miou_v:.3f}"
            if r.iou and not math.isnan(_miou_all) else "mIoU=n/a"
        )
        labels = [
            f"keyframe {kf_idx} (GT)",
            f"target {t_idx} (warped, +{r.distance}f)  {miou_label}",
            f"target {t_idx} (GT)",
            f"validity ({r.valid_pct:.1f}% valid)",
        ]
        panels = [panel_kf, panel_warp, panel_gt, validity_vis]
        for panel, label in zip(panels, labels):
            cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 1, cv2.LINE_AA)

        out_img = np.concatenate([panel_kf, panel_warp, panel_gt, validity_vis], axis=1)
        path = out_dir / f"prop_kf{kf_idx}_to_{t_idx}.png"
        iio.imwrite(path, out_img)
        print(f"[prop] wrote {path}")

    print("[prop] done — open outputs/propagation/ to inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
