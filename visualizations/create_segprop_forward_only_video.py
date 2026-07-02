"""Visualize a forward-only SegProp projection against standard SegProp i01."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_map(path: Path) -> np.ndarray:
    with np.load(path) as data:
        value = data["map"]
    return value.astype(bool)


def _colorize(one_hot: np.ndarray, palette: dict[int, tuple[int, int, int]]) -> np.ndarray:
    indexed = np.argmax(one_hot, axis=2)
    out = np.zeros((*indexed.shape, 3), dtype=np.uint8)
    for class_id, rgb in palette.items():
        out[indexed == class_id] = rgb
    return out


def _overlay(frame: np.ndarray, label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return ((1.0 - alpha) * frame + alpha * label).astype(np.uint8)


def _panel(image: np.ndarray, caption: str, width: int, height: int) -> np.ndarray:
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(image, (0, 0), (width, 42), (0, 0, 0), -1)
    cv2.putText(image, caption, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def _miou(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred = np.argmax(prediction, axis=2)
    gt = np.argmax(ground_truth, axis=2)
    scores = []
    for class_id in range(ground_truth.shape[2]):
        gt_class = gt == class_id
        if not gt_class.any():
            continue
        union = np.sum((pred == class_id) | gt_class)
        scores.append(float(np.sum((pred == class_id) & gt_class) / union))
    return float(np.mean(scores))


def _forward_project(source_map: np.ndarray, flow_xy: np.ndarray, device: str) -> np.ndarray:
    """Replicate SegProp's previous-keyframe forward endpoint projection.

    ``flow_xy`` contains the adjacent forward FlowNet2 fields from source to
    target. Standard i01 also votes from the future anchor and inverse/current
    projections; this function intentionally retains only this one vote.
    """
    source = torch.from_numpy(source_map).to(device)
    flow = torch.from_numpy(flow_xy).to(device).flip(3)  # FlowNet x/y -> SegProp y/x
    height, width, _ = source.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    location = torch.stack((ys, xs), dim=2).float()
    for step in flow:
        y = location[..., 0].round().long().clamp(0, height - 1)
        x = location[..., 1].round().long().clamp(0, width - 1)
        location = location + step[y, x]
        location[..., 0].clamp_(0, height - 1)
        location[..., 1].clamp_(0, width - 1)

    destination_y = location[..., 0].round().long()
    destination_x = location[..., 1].round().long()
    projected = torch.zeros_like(source)
    projected[destination_y, destination_x] = source
    return projected.cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="DJI_0101")
    parser.add_argument("--frames-root", default="outputs/segprop_paper_repro/frames_2k")
    parser.add_argument("--labels-root", default="outputs/segprop_paper_repro/labels_2k")
    parser.add_argument("--segprop-root", default="outputs/segprop_paper_repro/output_2k/i01")
    parser.add_argument("--flow-root", default="outputs/segprop_paper_repro/flow_2k_fn2")
    parser.add_argument("--out-dir", default="visualizations")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    palette = {
        0: (0, 255, 0), 1: (0, 127, 0), 2: (255, 255, 0), 3: (255, 127, 0),
        4: (255, 255, 255), 5: (255, 0, 255), 6: (127, 127, 127), 7: (0, 0, 255),
        8: (0, 255, 255), 9: (127, 127, 63), 10: (255, 0, 0), 11: (127, 127, 0),
    }
    root = _REPO_ROOT
    frame_dir = root / args.frames_root / args.video
    even_dir = root / args.labels_root / "train_even" / args.video
    odd_dir = root / args.labels_root / "train_odd" / args.video
    i01_dir = root / args.segprop_root / args.video
    flow_path = root / args.flow_root / f"{args.video}_forward.h5"
    targets = sorted(int(p.stem.rsplit("_", 1)[1]) for p in odd_dir.glob("*.npz") if int(p.stem.rsplit("_", 1)[1]) % 100 == 50)
    if not targets:
        raise RuntimeError("no held-out odd frames found")

    out_dir = root / args.out_dir / args.video
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "segprop_forward_only_vs_i01.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (args.width * 2, args.height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {out_path}")

    with h5py.File(flow_path, "r") as flow_h5:
        try:
            for target in targets:
                source = target - 50
                source_path = even_dir / f"{args.video}_{source:06d}.npz"
                target_path = odd_dir / f"{args.video}_{target:06d}.npz"
                i01_path = i01_dir / f"{args.video}_{target:06d}.npz"
                frame_path = frame_dir / f"{args.video}_{target:06d}.jpg"
                forward_only = _forward_project(
                    _load_map(source_path), flow_h5["flow"][source:target], args.device,
                )
                standard = _load_map(i01_path)
                ground_truth = _load_map(target_path)
                forward_only_miou = _miou(forward_only, ground_truth)
                standard_miou = _miou(standard, ground_truth)
                frame = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
                top = np.concatenate([
                    _panel(frame, f"{args.video} frame {target} original", args.width, args.height),
                    _panel(
                        _overlay(frame, _colorize(forward_only, palette)),
                        f"SegProp forward-only  mIoU={forward_only_miou:.3f}",
                        args.width, args.height,
                    ),
                ], axis=1)
                bottom = np.concatenate([
                    _panel(
                        _overlay(frame, _colorize(standard, palette)),
                        f"SegProp i01  mIoU={standard_miou:.3f}",
                        args.width, args.height,
                    ),
                    _panel(_overlay(frame, _colorize(ground_truth, palette)), f"Ground truth frame {target}", args.width, args.height),
                ], axis=1)
                writer.write(np.concatenate([top, bottom], axis=0))
                print(
                    f"[viz-forward] frame={target} forward-only mIoU={forward_only_miou:.4f} "
                    f"i01 mIoU={standard_miou:.4f}",
                    flush=True,
                )
        finally:
            writer.release()
    print(f"[viz-forward] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
