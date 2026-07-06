"""Run the trained GeoSeg model on one video’s 1-FPS labelled frames."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from geoseg.datasets.ruralscapes_dataset import (
   CLASSES,
   IGNORE_INDEX,
    PALETTE,
    RuralscapesDataset,
    _mask_to_index,
)
from train_supervision import Supervision_Train
from tools.cfg import py2cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="third_party/GeoSeg/config/ruralscapes/unetformer.py")
    parser.add_argument("--output-root", default="outputs/geoseg_inference")
    parser.add_argument("--batch-size", type=int, default=6)
    return parser.parse_args()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    config = py2cfg(args.config)
    split_file = Path("/tmp/geoseg_inference_video.txt")
    split_file.write_text(args.video + "\n")
    dataset = RuralscapesDataset(
        split_file=str(split_file),
        dataset_root="data/Ruralscapes",
        frames_root="data/Ruralscapes/frames_labelled",
        image_size=None,
        pad_to_multiple=32,
        train=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Supervision_Train.load_from_checkpoint(
        args.checkpoint, config=config, map_location=device
    ).to(device).eval()
    output_dir = Path(args.output_root) / args.video
    output_dir.mkdir(parents=True, exist_ok=True)

    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    gpu_seconds = 0.0
    wall_seconds = 0.0
    frame_count = 0
    for start in range(0, len(dataset), args.batch_size):
        samples = [dataset[i] for i in range(start, min(start + args.batch_size, len(dataset)))]
        images = torch.stack([sample["img"] for sample in samples]).to(device, non_blocking=True)
        sync(device)
        gpu_start = time.perf_counter()
        with torch.inference_mode():
            output = model.net(images)
            if isinstance(output, (tuple, list)):
                output = output[0]
            predictions = output.argmax(dim=1).cpu().numpy().astype(np.uint8)
        sync(device)
        elapsed = time.perf_counter() - gpu_start
        gpu_seconds += elapsed
        wall_start = time.perf_counter()
        for offset, sample in enumerate(samples):
            frame_path, mask_path, image_id = dataset.samples[start + offset]
            original = np.asarray(Image.open(mask_path))
            original_mask = original if original.ndim == 2 else _mask_to_index(original[..., :3])
            height, width = original_mask.shape
            prediction = predictions[offset, :height, :width]
            valid = original_mask != IGNORE_INDEX
            np.add.at(confusion, (original_mask[valid], prediction[valid]), 1)
           Image.fromarray(prediction).save(output_dir / f"{image_id}_prediction.png")
            frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
            color_mask = PALETTE[prediction]
            overlay = (0.55 * frame_rgb + 0.45 * color_mask).clip(0, 255).astype(np.uint8)
            Image.fromarray(overlay).save(output_dir / f"{image_id}_overlay.jpg", quality=92)
            frame_count += 1
        wall_seconds += time.perf_counter() - wall_start

    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(1) + confusion.sum(0) - intersection
    iou = np.divide(intersection, union, out=np.full(len(CLASSES), np.nan), where=union > 0)
    result = {
        "video": args.video,
        "frames": frame_count,
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "batch_size": args.batch_size,
        "inference_seconds": gpu_seconds,
        "seconds_per_frame": gpu_seconds / frame_count,
        "frames_per_second": frame_count / gpu_seconds,
        "total_result_save_seconds": wall_seconds,
        "pixel_accuracy": float(intersection.sum() / confusion.sum()),
        "mean_iou": float(np.nanmean(iou)),
        "per_class_iou": {name: (None if np.isnan(score) else float(score))
                          for name, score in zip(CLASSES, iou)},
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
