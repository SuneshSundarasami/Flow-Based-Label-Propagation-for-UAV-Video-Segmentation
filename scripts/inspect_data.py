"""A3 check: confirm the Ruralscapes loader works and print a class legend.

Loads one frame/mask pair, prints a summary and per-class pixel counts, and
writes a side-by-side frame|colored-mask preview to outputs/data_check/.

Usage
-----
    python scripts/inspect_data.py --root data/ruralscapes/<video> \
        --frame-glob "frames/*.jpg" --mask-glob "masks/*.png" \
        --mask-format color
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import imageio.v3 as iio  # noqa: E402

from uav_flowprop.data import RuralscapesVideo  # noqa: E402


def _colorize(label: np.ndarray, video: RuralscapesVideo) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cid, rgb in video.palette.rgb.items():
        out[label == cid] = rgb
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--frame-glob", default="frames/*.jpg")
    ap.add_argument("--mask-glob", default="masks/*.png")
    ap.add_argument("--mask-format", default="color", choices=["color", "indexed"])
    ap.add_argument("--palette", default=None)
    args = ap.parse_args()

    video = RuralscapesVideo(
        root=args.root, frame_glob=args.frame_glob, mask_glob=args.mask_glob,
        mask_format=args.mask_format, palette_path=args.palette)
    print("[data]", video.summary())

    if not video.annotated_indices:
        print("[data] no annotated frames matched -- check the mask glob.")
        return 1

    idx = video.annotated_indices[0]
    frame = video.load_frame(idx)
    mask = video.load_mask(idx)
    print(f"[data] sample frame {idx}: image {frame.shape}, mask {mask.shape}")
    print("[data] class legend (id: name -> pixels):")
    for cid in video.palette.ids:
        n = int((mask == cid).sum())
        print(f"    {cid:>3}: {video.palette.names[cid]:<14} {n:>10}")
    n_ignore = int((mask == video.palette.ignore_index).sum())
    print(f"    {video.palette.ignore_index:>3}: {'<ignore>':<14} {n_ignore:>10}")

    out_dir = _REPO_ROOT / "outputs" / "data_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = np.concatenate([frame, _colorize(mask, video)], axis=1)
    iio.imwrite(out_dir / f"frame_{idx}_overlay.png", preview)
    print(f"[data] wrote {out_dir / f'frame_{idx}_overlay.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
