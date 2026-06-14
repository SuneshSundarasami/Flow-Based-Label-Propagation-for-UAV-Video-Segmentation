"""B1 visual check: warp a keyframe mask to neighbour frames and compare.

Loads annotated frames from Ruralscapes, estimates backward flow
(target→keyframe) with SEA-RAFT, warps the keyframe mask, and writes a
three-panel image for each target distance:

    [keyframe + GT mask] | [target + warped mask] | [target + GT mask]

The third panel lets you see at a glance how well the warp matches reality.

Usage
-----
    python scripts/check_warp.py \\
        --root data/Ruralscapes \\
        --frame-glob "frames/DJI_0043/*.jpg" \\
        --mask-glob "labels/manual_labels/DJI_0043/*.png" \\
        --mask-format color \\
        --n-targets 3 \\
        --device cuda
"""
from __future__ import annotations

import argparse
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
from warp import warp_mask, compute_fb_mask  # noqa: E402

_VIZ_W, _VIZ_H = 960, 540  # resize target for flow + display


def _resize(img: np.ndarray, w: int, h: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=interp)


def _colorize(label: np.ndarray, video: RuralscapesVideo) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cid, rgb in video.palette.rgb.items():
        out[label == cid] = rgb
    return out


def _overlay(frame: np.ndarray, mask_color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend colourised mask onto the frame."""
    return (frame * (1 - alpha) + mask_color * alpha).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frame-glob", default="frames/DJI_0043/*.jpg")
    ap.add_argument("--mask-glob",
                    default="labels/manual_labels/DJI_0043/*.png")
    ap.add_argument("--mask-format", default="color",
                    choices=["color", "indexed"])
    ap.add_argument("--n-targets", type=int, default=3,
                    help="how many annotated frames after the keyframe to check")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = _REPO_ROOT / "outputs" / "warp_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    video = RuralscapesVideo(
        root=args.root,
        frame_glob=args.frame_glob,
        mask_glob=args.mask_glob,
        mask_format=args.mask_format,
    )
    ann = video.annotated_indices
    if len(ann) < 2:
        print("[warp] need at least 2 annotated frames; found", len(ann))
        return 1

    kf_idx = ann[0]
    targets = ann[1: 1 + args.n_targets]
    print(f"[warp] keyframe={kf_idx}, targets={targets}")

    print(f"[warp] loading SEA-RAFT on {args.device}")
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

    for t_idx in targets:
        dist = t_idx - kf_idx
        tgt_frame_full = video.load_frame(t_idx)
        tgt_frame = _resize(tgt_frame_full, _VIZ_W, _VIZ_H)

        # backward flow t→k and forward flow k→t for FB check
        flow_bwd = model.estimate_flow(tgt_frame, kf_frame)
        flow_fwd = model.estimate_flow(kf_frame, tgt_frame)

        warped = warp_mask(kf_mask, flow_bwd)

        fb_threshold = load_config()["occlusion"]["fb_threshold"]
        valid = compute_fb_mask(flow_fwd, flow_bwd, threshold=fb_threshold)
        valid_pct = 100.0 * valid.mean()

        tgt_mask_full = video.load_mask(t_idx)
        tgt_gt = _resize(tgt_mask_full, _VIZ_W, _VIZ_H, interp=cv2.INTER_NEAREST)

        # validity panel: white=valid, red=occluded
        validity_vis = tgt_frame.copy()
        validity_vis[~valid] = [180, 30, 30]

        panel_kf = _overlay(kf_frame, _colorize(kf_mask, video))
        panel_warp = _overlay(tgt_frame, _colorize(warped, video))
        panel_gt = _overlay(tgt_frame, _colorize(tgt_gt, video))

        labels = [
            f"keyframe {kf_idx} (GT)",
            f"target {t_idx} (warped, +{dist}f)",
            f"target {t_idx} (GT)",
            f"validity mask ({valid_pct:.1f}% valid)",
        ]
        for panel, label in zip([panel_kf, panel_warp, panel_gt, validity_vis], labels):
            cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 1, cv2.LINE_AA)

        print(f"[warp] +{dist}f: {valid_pct:.1f}% pixels valid (FB threshold={fb_threshold}px)")
        out = np.concatenate([panel_kf, panel_warp, panel_gt, validity_vis], axis=1)
        path = out_dir / f"warp_kf{kf_idx}_to_{t_idx}.png"
        iio.imwrite(path, out)
        print(f"[warp] wrote {path}")

    print(f"[warp] done — open outputs/warp_check/ to inspect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
