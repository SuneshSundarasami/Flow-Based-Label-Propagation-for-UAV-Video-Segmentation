"""Pre-resize labelled Ruralscapes frames and masks for GeoSeg training."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PALETTE = np.array([
    [0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 127, 0],
    [255, 255, 255], [255, 0, 255], [127, 127, 127], [0, 0, 255],
    [0, 255, 255], [127, 127, 63], [255, 0, 0], [127, 127, 0],
], dtype=np.uint8)
INDEX_RE = re.compile(r"_(\d+)\.png$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="data/Ruralscapes")
    ap.add_argument("--split-files", nargs="+", default=["splits/ruralscapes_geoseg_train.txt", "splits/ruralscapes_geoseg_val.txt"])
    ap.add_argument("--output-root", default="data/Ruralscapes/frames_geoseg_256")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    root, output = Path(args.dataset_root), Path(args.output_root)
    videos = [line.strip() for split in args.split_files for line in Path(split).read_text().splitlines() if line.strip()]
    for video in videos:
        labels = root / "labels" / "manual_labels" / video
        frames = root / "frames_labelled" / video
        out = output / video
        out.mkdir(parents=True, exist_ok=True)
        for label in sorted(labels.glob("segfull_*.png")):
            index = int(INDEX_RE.search(label.name).group(1))
            frame = cv2.imread(str(frames / f"frame_{index:06d}.jpg"), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(frames / f"frame_{index:06d}.jpg")
            frame = cv2.resize(frame, (args.size, args.size), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"frame_{index:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rgb = np.asarray(Image.open(label).convert("RGB"))
            indexed = np.full(rgb.shape[:2], 255, dtype=np.uint8)
            for cid, color in enumerate(PALETTE):
                indexed[np.all(rgb == color, axis=2)] = cid
            indexed = cv2.resize(indexed, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / f"mask_{index:06d}.png"), indexed)
        print(f"prepared {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
