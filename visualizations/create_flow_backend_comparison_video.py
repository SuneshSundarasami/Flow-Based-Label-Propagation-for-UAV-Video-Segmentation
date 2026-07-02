"""Render a direct-propagation comparison: SEA-RAFT vs FlowNet2."""
from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data import RuralscapesVideo  # noqa: E402
from flow import FlowNet2Flow, SeaRaftFlow  # noqa: E402
from propagation import propagate_keyframe  # noqa: E402


@dataclass
class Prediction:
    label: np.ndarray
    miou_all: float


def _colorize(label: np.ndarray, video: RuralscapesVideo) -> np.ndarray:
    colored = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id, rgb in video.palette.rgb.items():
        colored[label == class_id] = rgb
    return colored


def _overlay(frame: np.ndarray, label: np.ndarray, video: RuralscapesVideo) -> np.ndarray:
    return (0.5 * frame + 0.5 * _colorize(label, video)).astype(np.uint8)


def _panel(image: np.ndarray, caption: str, width: int, height: int) -> np.ndarray:
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel, (0, 0), (width, 42), (0, 0, 0), -1)
    cv2.putText(panel, caption, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _resized_mask(video: RuralscapesVideo, index: int, shape: tuple[int, int]) -> np.ndarray:
    mask = video.load_mask(index)
    height, width = shape
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _predict(
    video: RuralscapesVideo,
    model,
    pairs: list[tuple[int, int]],
) -> dict[int, Prediction]:
    predictions: dict[int, Prediction] = {}
    for keyframe_index, target_index in pairs:
        keyframe = video.load_frame(keyframe_index)
        target = video.load_frame(target_index)
        target_shape = target.shape[:2]
        result = propagate_keyframe(
            keyframe_mask=_resized_mask(video, keyframe_index, target_shape),
            keyframe_frame=keyframe,
            target_frames=[target],
            target_indices=[target_index],
            keyframe_index=keyframe_index,
            model=model,
            gt_masks=[_resized_mask(video, target_index, target_shape)],
            num_classes=video.num_classes,
        )[0]
        predictions[target_index] = Prediction(
            label=result.warped_mask,
            miou_all=result.iou["all"]["miou"],
        )
        print(
            f"[viz-backends] {keyframe_index}->{target_index} "
            f"mIoU={result.iou['all']['miou']:.4f}",
            flush=True,
        )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="DJI_0101")
    parser.add_argument("--frames-root", default="outputs/segprop_paper_repro/frames_2k")
    parser.add_argument("--masks-root", default="data/Ruralscapes/labels/manual_labels")
    parser.add_argument("--checkpoint-searaft", default="third_party/SEA-RAFT/checkpoints/model.safetensors")
    parser.add_argument("--searaft-config", default="third_party/SEA-RAFT/config/eval/spring-L.json")
    parser.add_argument("--checkpoint-flownet2", default="third_party/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar")
    parser.add_argument("--out-dir", default="visualizations")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    video = RuralscapesVideo(
        root=".",
        frame_glob=f"{args.frames_root}/{args.video}/*.jpg",
        mask_glob=f"{args.masks_root}/{args.video}/*.png",
        mask_format="color",
    )
    targets = [index for index in video.annotated_indices if index % 100 == 50]
    pairs = [(target - 50, target) for target in targets if target - 50 in video.frames]
    if not pairs:
        raise RuntimeError("no even-to-odd target pairs found")

    print("[viz-backends] running SEA-RAFT", flush=True)
    searaft = SeaRaftFlow(
        args.searaft_config,
        checkpoint=args.checkpoint_searaft,
        iters=12,
        device=args.device,
    )
    searaft_predictions = _predict(video, searaft, pairs)
    del searaft
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    print("[viz-backends] running FlowNet2", flush=True)
    flownet2 = FlowNet2Flow(args.checkpoint_flownet2, device=args.device)
    flownet2_predictions = _predict(video, flownet2, pairs)
    del flownet2
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    out_dir = _REPO_ROOT / args.out_dir / args.video
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "searaft_vs_flownet2_direct.mp4"
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (args.width * 2, args.height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {out_path}")
    try:
        for keyframe_index, target_index in pairs:
            target = video.load_frame(target_index)
            target_mask = _resized_mask(video, target_index, target.shape[:2])
            sea = searaft_predictions[target_index]
            fn2 = flownet2_predictions[target_index]
            top = np.concatenate([
                _panel(target, f"{args.video} frame {target_index} original", args.width, args.height),
                _panel(_overlay(target, sea.label, video), f"SEA-RAFT direct  mIoU={sea.miou_all:.3f}", args.width, args.height),
            ], axis=1)
            bottom = np.concatenate([
                _panel(_overlay(target, fn2.label, video), f"FlowNet2 direct  mIoU={fn2.miou_all:.3f}", args.width, args.height),
                _panel(_overlay(target, target_mask, video), f"Ground truth frame {target_index}", args.width, args.height),
            ], axis=1)
            writer.write(np.concatenate([top, bottom], axis=0))
    finally:
        writer.release()
    print(f"[viz-backends] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
