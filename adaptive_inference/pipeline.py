"""Core algorithm: infer a keyframe, propagate it with optical flow, re-infer
once cumulative forward-backward-consistency coverage drops to the configured
threshold, repeat.

Flow for the next ``steps_per_chunk`` frame-pairs is computed in one batched,
GPU-resident launch (both directions together) and immediately consumed by
the cheap sequential mask-chaining step -- this keeps peak GPU memory flat
regardless of video length (holding every frame's flow field at once for a
long video is hundreds of GB) while still getting most of the benefit of
batching over doing one pair at a time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import torch

from .config import Config
from .models import ignore_index, normalization_stats


def decode_video(video_path: str | Path, n_frames: int | None = None) -> list[np.ndarray]:
    """Sequential decode -> list of (H, W, 3) uint8 RGB frames.

    Sequential ``cap.read()`` rather than seeking to each frame: on the
    UAVid videos' codec, per-frame seeking is far slower than a straight
    forward read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")
    frames = []
    while n_frames is None or len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


@dataclass
class StepResult:
    frame_index: int
    mask: np.ndarray    # (H, W) uint8 label ids
    source: str         # "model" | "flow"
    valid_pct: float    # cumulative FB-valid coverage (1.0 right after a model call)
    elapsed_s: float


class SegmentationRunner:
    """Thin, timed wrapper around a loaded segmentation model."""

    def __init__(self, model: torch.nn.Module, device: str):
        self.model = model
        self.device = device
        mean, std = normalization_stats()
        self._mean = torch.tensor(mean).view(3, 1, 1)
        self._std = torch.tensor(std).view(3, 1, 1)

    def __call__(self, frame_rgb: np.ndarray) -> tuple[np.ndarray, float]:
        x = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
        x = ((x - self._mean) / self._std).unsqueeze(0).to(self.device, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.bfloat16):
            logits = self.model(x)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        torch.cuda.synchronize()
        return pred, time.time() - t0


def run_adaptive_pipeline(
    frames: list[np.ndarray],
    seg_model: torch.nn.Module,
    flow_model,
    cfg: Config,
) -> Iterator[StepResult]:
    """Yield one :class:`StepResult` per frame, in order.

    ``flow_model`` must already have ``scale``/``max_images_per_launch`` set
    (``models.build_flow_model`` does this from *cfg*).
    """
    from propagation.gpu_ops import warp_masks_gpu, fb_masks_gpu  # src/propagation

    device = cfg.adaptive.device
    threshold = cfg.adaptive.valid_threshold
    chunk_size = cfg.flow.steps_per_chunk
    fb_threshold = cfg.flow.fb_threshold
    dtype = torch.bfloat16 if cfg.flow.precision == "bf16" else torch.float32
    ignore_id = ignore_index()

    infer = SegmentationRunner(seg_model, device)

    pred0, t0 = infer(frames[0])
    current_mask = torch.from_numpy(pred0.astype(np.int64)).to(device).unsqueeze(0)
    current_valid = torch.ones_like(current_mask, dtype=torch.bool)
    yield StepResult(0, pred0, "model", 1.0, t0)

    n_steps = len(frames) - 1
    step = 0
    while step < n_steps:
        chunk_end = min(step + chunk_size, n_steps)
        chunk = list(range(step, chunk_end))
        fwd_pairs = [(frames[i], frames[i + 1]) for i in chunk]
        bwd_pairs = [(frames[i + 1], frames[i]) for i in chunk]

        t_chunk0 = time.time()
        with torch.autocast(device, dtype=dtype):
            flow_fwd_chunk = flow_model.estimate_flow_batch_tensor(fwd_pairs).float()
            flow_bwd_chunk = flow_model.estimate_flow_batch_tensor(bwd_pairs).float()
        torch.cuda.synchronize()
        chunk_flow_time = time.time() - t_chunk0
        per_step_flow_time = chunk_flow_time / len(chunk)

        for local_i, s in enumerate(chunk):
            t_step0 = time.time()
            flow_fwd = flow_fwd_chunk[local_i:local_i + 1]
            flow_bwd = flow_bwd_chunk[local_i:local_i + 1]

            new_mask = warp_masks_gpu(current_mask, flow_bwd, ignore_index=ignore_id)
            step_valid = fb_masks_gpu(flow_fwd, flow_bwd, threshold=fb_threshold)
            warped_prior_valid = warp_masks_gpu(current_valid.long(), flow_bwd, ignore_index=0)
            new_valid = (warped_prior_valid == 1) & step_valid

            frame_idx = s + 1
            valid_pct = new_valid.float().mean().item()

            if valid_pct <= threshold:
                pred, t_model = infer(frames[frame_idx])
                current_mask = torch.from_numpy(pred.astype(np.int64)).to(device).unsqueeze(0)
                current_valid = torch.ones_like(current_mask, dtype=torch.bool)
                yield StepResult(frame_idx, pred, "model", 1.0, t_model)
            else:
                current_mask, current_valid = new_mask, new_valid
                mask_np = current_mask.squeeze(0).to(torch.uint8).cpu().numpy()
                yield StepResult(frame_idx, mask_np, "flow", valid_pct,
                                  per_step_flow_time + (time.time() - t_step0))

        del flow_fwd_chunk, flow_bwd_chunk
        step = chunk_end
