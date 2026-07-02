"""Generate SegProp-compatible FlowNet2 H5 files from dense frame folders.

The actual FlowNet2 runtime is intentionally external to this project. This
script supplies the reproducible harness around it: frame ordering, pair
generation, H5 shape/dtype, vector-order validation, and resumability.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

_FRAME_RE = re.compile(r"(\d+)(?=\D*$)")


@dataclass(frozen=True)
class FrameInfo:
    index: int
    path: Path


def _parse_frame_index(path: Path) -> int:
    match = _FRAME_RE.search(path.stem)
    if not match:
        raise ValueError(f"could not parse frame index from {path.name}")
    return int(match.group(1))


def list_frames(frame_dir: Path) -> list[FrameInfo]:
    frames = [
        FrameInfo(_parse_frame_index(path), path)
        for path in frame_dir.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    frames.sort(key=lambda item: item.index)
    if len(frames) < 2:
        raise FileNotFoundError(f"need at least 2 frames in {frame_dir}")
    return frames


def read_frame_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read frame: {path}")
    return image.shape[:2]


def validate_flow(flow: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    expected = (*shape, 2)
    if flow.shape != expected:
        raise ValueError(f"expected flow shape {expected}, got {flow.shape}")
    if not np.isfinite(flow).all():
        raise ValueError("flow contains NaN or infinite values")
    return flow.astype(np.float32, copy=False)


def run_flow_command(template: str, img1: Path, img2: Path, out_path: Path) -> np.ndarray:
    command = template.format(
        img1=str(img1),
        img2=str(img2),
        out=str(out_path),
    )
    subprocess.run(shlex.split(command), check=True)
    if not out_path.exists():
        raise FileNotFoundError(f"flow command did not create {out_path}")
    return np.load(out_path)


def _load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text())
    return {int(index) for index in data.get("done", [])}


def _save_done(path: Path, done: set[int]) -> None:
    path.write_text(json.dumps({"done": sorted(done)}, indent=2) + "\n")


def _open_flow_h5(path: Path, n_pairs: int, shape: tuple[int, int], overwrite: bool) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and path.exists():
        path.unlink()
    mode = "a" if path.exists() else "w"
    h5 = h5py.File(path, mode)
    expected = (n_pairs, shape[0], shape[1], 2)
    if "flow" not in h5:
        h5.create_dataset(
            "flow",
            shape=expected,
            dtype="float32",
            chunks=(1, shape[0], shape[1], 2),
        )
    elif h5["flow"].shape != expected:
        h5.close()
        raise ValueError(f"{path} has flow shape {h5['flow'].shape}, expected {expected}")
    return h5


def generate_video_flows(
    *,
    video: str,
    frames_root: Path,
    out_root: Path,
    flow_command: str,
    overwrite: bool = False,
    limit_pairs: int | None = None,
) -> tuple[Path, Path]:
    frame_dir = frames_root / video
    frames = list_frames(frame_dir)
    shape = read_frame_shape(frames[0].path)
    n_pairs = len(frames) - 1 if limit_pairs is None else min(limit_pairs, len(frames) - 1)

    flow_root = out_root / "flow_2k_fn2"
    forward_path = flow_root / f"{video}_forward.h5"
    backward_path = flow_root / f"{video}_backward.h5"
    done_path = flow_root / f"{video}_progress.json"
    can_resume = forward_path.exists() and backward_path.exists() and not overwrite
    done = _load_done(done_path) if can_resume else set()

    with _open_flow_h5(forward_path, n_pairs, shape, overwrite) as fwd_h5, \
            _open_flow_h5(backward_path, n_pairs, shape, overwrite) as bwd_h5:
        fwd = fwd_h5["flow"]
        bwd = bwd_h5["flow"]

        for pair_i in tqdm(range(n_pairs), desc=f"FlowNet2 {video}", unit="pair"):
            if pair_i in done:
                continue
            img1 = frames[pair_i].path
            img2 = frames[pair_i + 1].path
            with tempfile.TemporaryDirectory(prefix="fn2_pair_") as tmp:
                tmpdir = Path(tmp)
                fwd_flow = run_flow_command(flow_command, img1, img2, tmpdir / "forward.npy")
                bwd_flow = run_flow_command(flow_command, img2, img1, tmpdir / "backward.npy")
            fwd[pair_i] = validate_flow(fwd_flow, shape)
            bwd[pair_i] = validate_flow(bwd_flow, shape)
            fwd_h5.flush()
            bwd_h5.flush()
            done.add(pair_i)
            _save_done(done_path, done)

    return forward_path, backward_path


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SegProp-compatible FlowNet2 H5 files from frames_2k."
    )
    parser.add_argument("--out-root", default="outputs/segprop_paper_repro")
    parser.add_argument("--frames-root", default=None,
                        help="Defaults to <out-root>/frames_2k.")
    parser.add_argument("--split-file", default="data/Ruralscapes/ruralscapes_training_videos.txt")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="Specific videos to process; defaults to --split-file.")
    parser.add_argument("--flow-command", required=True,
                        help=(
                            "External FlowNet2 command template. It must accept "
                            "{img1}, {img2}, and {out}, and write an .npy flow "
                            "array with shape (H, W, 2) in x,y vector order."
                        ))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-pairs", type=int, default=None,
                        help="Debug/smoke option: process only the first N adjacent pairs.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    frames_root = Path(args.frames_root) if args.frames_root else out_root / "frames_2k"
    videos = args.videos if args.videos else _read_split(Path(args.split_file))

    print(f"[fn2-h5] videos      : {', '.join(videos)}")
    print(f"[fn2-h5] frames root : {frames_root}")
    print(f"[fn2-h5] out root    : {out_root}")

    for video in videos:
        forward, backward = generate_video_flows(
            video=video,
            frames_root=frames_root,
            out_root=out_root,
            flow_command=args.flow_command,
            overwrite=args.overwrite,
            limit_pairs=args.limit_pairs,
        )
        print(f"[fn2-h5] {video}: {forward}")
        print(f"[fn2-h5] {video}: {backward}")

    print("[fn2-h5] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
