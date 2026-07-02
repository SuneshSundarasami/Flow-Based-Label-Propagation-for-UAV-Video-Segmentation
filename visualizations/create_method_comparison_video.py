"""Create a side-by-side SEA-RAFT vs SegProp i01 visualization video."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data import RuralscapesVideo  # noqa: E402
from eval import compute_iou  # noqa: E402
from flow import SeaRaftFlow  # noqa: E402
from propagation import propagate_keyframe  # noqa: E402


def _colorize(label: np.ndarray, video: RuralscapesVideo) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id, rgb in video.palette.rgb.items():
        out[label == class_id] = rgb
    return out


def _overlay(frame_rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * frame_rgb + alpha * mask_rgb).astype(np.uint8)


def _text(panel_rgb: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(panel, label, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _load_segprop_map(path: Path) -> np.ndarray:
    with np.load(path) as data:
        class_map = data["map"]
    if class_map.ndim == 3:
        return np.argmax(class_map, axis=2).astype(np.int32)
    return class_map.astype(np.int32)


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="DJI_0101")
    parser.add_argument("--frames-root", default="outputs/segprop_paper_repro/frames_2k")
    parser.add_argument("--labels-root", default="data/Ruralscapes/labels/manual_labels")
    parser.add_argument("--segprop-root", default="outputs/segprop_paper_repro/output_2k/i01")
    parser.add_argument("--checkpoint", default="third_party/SEA-RAFT/checkpoints/model.safetensors")
    parser.add_argument("--model-cfg", default="third_party/SEA-RAFT/config/eval/spring-L.json")
    parser.add_argument("--out-dir", default="visualizations")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iters", type=int, default=12)
    args = parser.parse_args()

    video = RuralscapesVideo(
        root=".",
        frame_glob=f"{args.frames_root}/{args.video}/*.jpg",
        mask_glob=f"{args.labels_root}/{args.video}/*.png",
        mask_format="color",
    )
    targets = [i for i in video.annotated_indices if i % 100 == 50]
    keyframes = [i - 50 for i in targets]
    targets = [i for i, k in zip(targets, keyframes) if k in video.frames]
    keyframes = [i - 50 for i in targets]
    if not targets:
        raise RuntimeError("no even-to-odd comparison targets found")

    out_dir = _REPO_ROOT / args.out_dir / args.video
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "searaft_vs_segprop_i01.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (args.width * 2, args.height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {out_path}")

    model = SeaRaftFlow(
        args.model_cfg, checkpoint=args.checkpoint,
        iters=args.iters, device=args.device,
    )
    rows = []
    try:
        for keyframe_index, target_index in zip(keyframes, targets):
            keyframe_rgb = video.load_frame(keyframe_index)
            target_rgb = video.load_frame(target_index)
            keyframe_mask = video.load_mask(keyframe_index)
            target_mask = video.load_mask(target_index)
            target_shape = target_rgb.shape[:2]
            keyframe_mask = cv2.resize(
                keyframe_mask, (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            target_mask = cv2.resize(
                target_mask, (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            result = propagate_keyframe(
                keyframe_mask=keyframe_mask,
                keyframe_frame=keyframe_rgb,
                target_frames=[target_rgb],
                target_indices=[target_index],
                keyframe_index=keyframe_index,
                model=model,
                gt_masks=[target_mask],
                num_classes=video.num_classes,
            )[0]
            searaft_map = result.warped_mask
            segprop_path = (
                _REPO_ROOT / args.segprop_root / args.video
                / f"{args.video}_{target_index:06d}.npz"
            )
            if not segprop_path.exists():
                raise FileNotFoundError(f"missing SegProp prediction: {segprop_path}")
            segprop_map = _load_segprop_map(segprop_path)
            searaft_iou = result.iou["valid_only"]["miou"] if result.iou else float("nan")
            segprop_iou = compute_iou(
                segprop_map, target_mask, video.num_classes,
                ignore_index=video.palette.ignore_index,
            )["all"]["miou"]

            original = _resize_rgb(target_rgb, args.width, args.height)
            ours = _resize_rgb(
                _overlay(target_rgb, _colorize(searaft_map, video), 0.5),
                args.width, args.height,
            )
            theirs = _resize_rgb(
                _overlay(target_rgb, _colorize(segprop_map, video), 0.5),
                args.width, args.height,
            )
            ground_truth = _resize_rgb(
                _overlay(target_rgb, _colorize(target_mask, video), 0.5),
                args.width, args.height,
            )
            top = np.concatenate([
                _text(original, f"{args.video} frame {target_index} original"),
                _text(ours, f"SEA-RAFT direct  valid mIoU={searaft_iou:.3f}"),
            ], axis=1)
            bottom = np.concatenate([
                _text(theirs, f"SegProp i01  mIoU={segprop_iou:.3f}"),
                _text(ground_truth, f"Ground truth frame {target_index}"),
            ], axis=1)
            writer.write(np.concatenate([top, bottom], axis=0))
            rows.append((target_index, searaft_iou, segprop_iou))
            print(
                f"[viz] frame={target_index} SEA-RAFT valid mIoU={searaft_iou:.4f} "
                f"SegProp i01 mIoU={segprop_iou:.4f}", flush=True,
            )
    finally:
        writer.release()

    (out_dir / "README.md").write_text(
        "# SEA-RAFT vs SegProp i01\n\n"
        f"Video: `{args.video}`\n\n"
        f"Comparison video: `{out_path.name}`\n\n"
        "Panels are original frame, direct SEA-RAFT propagation, and SegProp i01.\n\n"
        "| Target frame | SEA-RAFT valid mIoU | SegProp i01 mIoU |\n"
        "|---:|---:|---:|\n"
        + "\n".join(f"| {idx} | {ours:.4f} | {seg:.4f} |" for idx, ours, seg in rows)
        + "\n",
        encoding="utf-8",
    )
    print(f"[viz] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
