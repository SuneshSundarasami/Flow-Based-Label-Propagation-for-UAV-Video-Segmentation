"""Prepare Ruralscapes data in the format expected by SegProp.

This script prepares the paper-style TrainEven/TrainOdd label split and, when
requested, exports dense 2K frames from the MP4 videos. It does not run
FlowNet2 or SegProp.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data.ruralscapes import load_palette  # noqa: E402

_LABEL_RE = re.compile(r"segfull_(.+)_(\d+)\.png$")
_DEFAULT_SIZE = (2048, 1080)


def _label_index(path: Path) -> int:
    match = _LABEL_RE.match(path.name)
    if not match:
        raise ValueError(f"could not parse label filename: {path.name}")
    return int(match.group(2))


def _segprop_label_name(path: Path) -> str:
    match = _LABEL_RE.match(path.name)
    if not match:
        raise ValueError(f"could not parse label filename: {path.name}")
    return f"{match.group(1)}_{int(match.group(2)):06d}.npz"


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((*mask.shape, num_classes), dtype=bool)
    for cid in range(num_classes):
        out[..., cid] = mask == cid
    return out


def _build_rgb_lut(palette) -> np.ndarray:
    lut = np.full(1 << 24, palette.ignore_index, dtype=np.uint16)
    for cid, color in palette.rgb.items():
        r, g, b = color
        lut[(r << 16) | (g << 8) | b] = cid
    return lut


def _color_mask_to_index_fast(mask_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    keys = (
        (mask_rgb[..., 0].astype(np.uint32) << 16)
        | (mask_rgb[..., 1].astype(np.uint32) << 8)
        | mask_rgb[..., 2].astype(np.uint32)
    )
    return lut[keys].astype(np.int32)


def _save_npz(path: Path, one_hot: np.ndarray, compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(path, map=one_hot, votes=one_hot)
    else:
        np.savez(path, map=one_hot, votes=one_hot)


def prepare_labels(
    *,
    dataset_root: Path,
    out_root: Path,
    videos: list[str],
    size: tuple[int, int],
    compress: bool,
    overwrite: bool,
) -> dict[str, int]:
    """Convert manual RGB labels to SegProp one-hot NPZ labels."""
    palette = load_palette()
    rgb_lut = _build_rgb_lut(palette)
    labels_root = out_root / "labels_2k"
    counts: dict[str, int] = {}

    for video in videos:
        label_dir = dataset_root / "labels" / "manual_labels" / video
        labels = sorted(label_dir.glob("segfull_*.png"), key=_label_index)
        if not labels:
            print(f"[segprop-prep] no labels found for {video}: {label_dir}")
            counts[video] = 0
            continue

        video_rows = []
        for ordinal, label_path in enumerate(tqdm(labels, desc=f"labels {video}", unit="label")):
            out_name = _segprop_label_name(label_path)
            all_path = labels_root / "all" / video / out_name
            split = "train_even" if ordinal % 2 == 0 else "train_odd"
            split_path = labels_root / split / video / out_name

            if overwrite or not all_path.exists():
                raw = cv2.imread(str(label_path), cv2.IMREAD_COLOR)
                if raw is None:
                    raise FileNotFoundError(f"could not read label: {label_path}")
                rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                mask = _color_mask_to_index_fast(rgb, rgb_lut)
                mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
                one_hot = _one_hot(mask, palette.num_classes)
                _save_npz(all_path, one_hot, compress=compress)

            split_path.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not split_path.exists():
                copyfile(all_path, split_path)

            video_rows.append({
                "ordinal": ordinal,
                "frame_index": _label_index(label_path),
                "split": split,
                "source": str(label_path.relative_to(dataset_root)),
                "segprop_label": str(all_path.relative_to(out_root)),
            })

        map_path = out_root / "metadata" / f"{video}_labels.csv"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["ordinal", "frame_index", "split", "source", "segprop_label"],
            )
            writer.writeheader()
            writer.writerows(video_rows)

        counts[video] = len(labels)
        print(f"[segprop-prep] labels {video}: {len(labels)} -> {labels_root}")

    return counts


def export_frames(
    *,
    dataset_root: Path,
    out_root: Path,
    videos: list[str],
    size: tuple[int, int],
    overwrite: bool,
) -> dict[str, int]:
    """Export dense MP4 frames at SegProp's paper-like 2K resolution."""
    frames_root = out_root / "frames_2k"
    counts: dict[str, int] = {}

    for video in videos:
        video_path = dataset_root / "videos" / f"{video}.MP4"
        out_dir = frames_root / video
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"could not open video: {video_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        written = 0
        for index in tqdm(range(frame_count), desc=f"frames {video}", unit="frame"):
            out_path = out_dir / f"{video}_{index:06d}.jpg"
            ok, frame = cap.read()
            if not ok:
                print(f"[segprop-prep] failed to read {video} frame {index}")
                break
            if out_path.exists() and not overwrite:
                continue
            resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(out_path), resized):
                raise OSError(f"failed to write {out_path}")
            written += 1

        cap.release()
        counts[video] = written
        print(f"[segprop-prep] frames {video}: written={written} out={out_dir}")

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Ruralscapes labels/frames for a SegProp reproduction."
    )
    parser.add_argument("--dataset-root", default="data/Ruralscapes")
    parser.add_argument("--out-root", default="outputs/segprop_paper_repro")
    parser.add_argument("--split-file", default="data/Ruralscapes/ruralscapes_training_videos.txt")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Specific videos to prepare; defaults to --split-file.")
    parser.add_argument("--steps", nargs="+", default=["labels"],
                        choices=["labels", "frames"],
                        help="Preparation steps to run. Use '--steps labels frames' for both.")
    parser.add_argument("--width", type=int, default=_DEFAULT_SIZE[0])
    parser.add_argument("--height", type=int, default=_DEFAULT_SIZE[1])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-compress", action="store_true",
                        help="Write faster but larger NPZ files.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    videos = args.videos if args.videos else _read_split(Path(args.split_file))
    size = (args.width, args.height)

    print(f"[segprop-prep] videos     : {', '.join(videos)}")
    print(f"[segprop-prep] size       : {size[0]}x{size[1]}")
    print(f"[segprop-prep] out root   : {out_root}")
    print(f"[segprop-prep] steps      : {', '.join(args.steps)}")

    if "labels" in args.steps:
        prepare_labels(
            dataset_root=dataset_root,
            out_root=out_root,
            videos=videos,
            size=size,
            compress=not args.no_compress,
            overwrite=args.overwrite,
        )
    if "frames" in args.steps:
        export_frames(
            dataset_root=dataset_root,
            out_root=out_root,
            videos=videos,
            size=size,
            overwrite=args.overwrite,
        )

    print("[segprop-prep] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
