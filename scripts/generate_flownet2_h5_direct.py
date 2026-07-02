"""Generate SegProp-compatible FlowNet2 H5 files with one persistent model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from flownet2_pair import _load_model, _pad_to_multiple_of_64, _read_rgb
from generate_flownet2_h5 import list_frames, read_frame_shape, validate_flow


def _read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


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
        actual = h5["flow"].shape
        h5.close()
        raise ValueError(f"{path} has flow shape {actual}, expected {expected}")
    return h5


def _configure_runtime(cpu_threads: int | None) -> None:
    if cpu_threads is None:
        cpu_threads = os.cpu_count() or 1
    cv2.setNumThreads(cpu_threads)
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _pair_batch_tensor(img_pairs: list[tuple[np.ndarray, np.ndarray]], device: str) -> torch.Tensor:
    tensors = []
    for img1, img2 in img_pairs:
        if img1.shape != img2.shape:
            raise ValueError(f"image shapes differ: {img1.shape} vs {img2.shape}")
        pair = np.stack([img1, img2], axis=0).transpose(3, 0, 1, 2)
        tensors.append(pair.astype(np.float32))
    batch = np.stack(tensors, axis=0)
    return torch.from_numpy(batch).to(device)


def _infer_flow(
    model,
    img1_path: Path,
    img2_path: Path,
    device: str,
) -> np.ndarray:
    img1 = _read_rgb(img1_path)
    img2 = _read_rgb(img2_path)
    pair_batch = _pair_batch_tensor([(img1, img2)], device)
    pair_batch, original_shape = _pad_to_multiple_of_64(pair_batch)
    with torch.no_grad():
        flows = model(pair_batch)
    flow = flows.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)[0]
    return flow[: original_shape[0], : original_shape[1]]


def _infer_bidirectional(
    model,
    img1_path: Path,
    img2_path: Path,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    img1 = _read_rgb(img1_path)
    img2 = _read_rgb(img2_path)
    pair_batch = _pair_batch_tensor([(img1, img2), (img2, img1)], device)
    pair_batch, original_shape = _pad_to_multiple_of_64(pair_batch)
    with torch.no_grad():
        flows = model(pair_batch)
    flows = flows.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.float32)
    flows = flows[:, : original_shape[0], : original_shape[1]]
    return flows[0], flows[1]


def generate_video_flows(
    *,
    model,
    video: str,
    frames_root: Path,
    out_root: Path,
    device: str,
    batch_directions: bool,
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
            if batch_directions:
                fwd_flow, bwd_flow = _infer_bidirectional(model, img1, img2, device)
            else:
                fwd_flow = _infer_flow(model, img1, img2, device)
                bwd_flow = _infer_flow(model, img2, img1, device)
            fwd[pair_i] = validate_flow(fwd_flow, shape)
            bwd[pair_i] = validate_flow(bwd_flow, shape)
            fwd_h5.flush()
            bwd_h5.flush()
            done.add(pair_i)
            _save_done(done_path, done)

    return forward_path, backward_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SegProp-compatible FlowNet2 H5 files with a persistent model."
    )
    parser.add_argument("--out-root", default="outputs/segprop_paper_repro")
    parser.add_argument("--frames-root", default=None,
                        help="Defaults to <out-root>/frames_2k.")
    parser.add_argument("--split-file", default="data/Ruralscapes/ruralscapes_training_videos.txt")
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--checkpoint",
                        default="third_party/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=None,
                        help="CPU/OpenCV/Torch thread count. Defaults to all visible cores.")
    parser.add_argument("--batch-directions", action="store_true",
                        help="Infer forward/backward flow together in one model pass. "
                             "Useful on larger GPUs, but can be slower on tight-memory setups.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-pairs", type=int, default=None)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    frames_root = Path(args.frames_root) if args.frames_root else out_root / "frames_2k"
    videos = args.videos if args.videos else _read_split(Path(args.split_file))

    print(f"[fn2-h5-direct] videos      : {', '.join(videos)}")
    print(f"[fn2-h5-direct] frames root : {frames_root}")
    print(f"[fn2-h5-direct] out root    : {out_root}")
    print(f"[fn2-h5-direct] device      : {args.device}")
    print(f"[fn2-h5-direct] cpu threads : {args.cpu_threads or (os.cpu_count() or 1)}")
    print(f"[fn2-h5-direct] batch dirs  : {args.batch_directions}")

    _configure_runtime(args.cpu_threads)
    model = _load_model(Path(args.checkpoint), args.device)

    for video in videos:
        forward, backward = generate_video_flows(
            model=model,
            video=video,
            frames_root=frames_root,
            out_root=out_root,
            device=args.device,
            batch_directions=args.batch_directions,
            overwrite=args.overwrite,
            limit_pairs=args.limit_pairs,
        )
        print(f"[fn2-h5-direct] {video}: {forward}")
        print(f"[fn2-h5-direct] {video}: {backward}")

    print("[fn2-h5-direct] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
