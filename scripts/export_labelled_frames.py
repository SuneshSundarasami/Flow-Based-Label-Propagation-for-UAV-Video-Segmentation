"""Export RGB frames that have Ruralscapes manual labels.

The Ruralscapes archive ships videos as MP4 files and labels as
``segfull_<video>_<frame_index>.png``. This helper extracts only those labelled
frame indices, preserving the numeric index in the output filename so
``RuralscapesVideo`` can match frames and masks.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2

_INDEX_RE = re.compile(r"_(\d+)\.png$")


def _label_index(path: Path) -> int:
    match = _INDEX_RE.search(path.name)
    if not match:
        raise ValueError(f"could not parse frame index from {path.name}")
    return int(match.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input MP4 path")
    ap.add_argument("--labels", required=True, help="manual-label directory")
    ap.add_argument("--out", required=True, help="output frame directory")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    video_path = Path(args.video)
    label_dir = Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(label_dir.glob("segfull_*.png"), key=_label_index)
    if not labels:
        raise SystemExit(f"no labels matched {label_dir / 'segfull_*.png'}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written = 0
    for label in labels:
        index = _label_index(label)
        if index >= frame_count:
            print(f"[frames] skip {index}: beyond video frame count {frame_count}",
                  file=sys.stderr)
            continue
        out_path = out_dir / f"frame_{index:06d}.jpg"
        if out_path.exists() and not args.overwrite:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            print(f"[frames] failed to read frame {index}", file=sys.stderr)
            continue
        if not cv2.imwrite(str(out_path), frame):
            raise SystemExit(f"failed to write {out_path}")
        written += 1

    cap.release()
    print(f"[frames] labels={len(labels)} written={written} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
