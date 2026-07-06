"""Cache direct-propagation results from one optical-flow backend.

Run this module once for SEA-RAFT and once for FlowNet2.  It saves predicted
masks and measurements; visualizers can then render comparisons without loading
or running either flow model again.

Throughput design
-----------------
The video's frame pairs are processed in GPU-resident batches: a thread pool
decodes frames/masks ahead of time (saturating CPU cores on JPEG/PNG work) while
the GPU runs one batched forward pass per chunk and does the warp, occlusion, and
IoU on-device (:mod:`propagation.gpu_ops`).  ``--batch-size`` trades GPU memory
for throughput; ``--loader-workers`` sizes the decode pool.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from data import RuralscapesVideo
from flow import FlowNet2Flow, SeaRaftFlow
from propagation.gpu_ops import fb_masks_gpu, iou_pair_gpu, warp_masks_gpu

_REPO_ROOT = Path(__file__).resolve().parents[2]


def comparison_pairs(video: RuralscapesVideo) -> list[tuple[int, int]]:
    """Return the even-keyframe to odd-keyframe pairs used in comparisons."""
    targets = [index for index in video.annotated_indices if index % 100 == 50]
    return [(target - 50, target) for target in targets if target - 50 in video.frames]


def _resized_mask(video: RuralscapesVideo, index: int, shape: tuple[int, int]) -> np.ndarray:
    mask = video.load_mask(index)
    height, width = shape
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _load_pair(
    video: RuralscapesVideo, keyframe_index: int, target_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode the frames and label masks a pair needs (runs in worker threads)."""
    keyframe = video.load_frame(keyframe_index)
    target = video.load_frame(target_index)
    shape = target.shape[:2]
    keyframe_mask = _resized_mask(video, keyframe_index, shape)
    gt_mask = _resized_mask(video, target_index, shape)
    return keyframe, target, keyframe_mask, gt_mask


def _chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def run_and_cache_backend(
    *,
    video: RuralscapesVideo,
    model: Any,
    backend_name: str,
    output_dir: str | Path,
    pairs: list[tuple[int, int]] | None = None,
    batch_size: int = 4,
    loader_workers: int = 8,
    fb_threshold: float = 1.5,
) -> list[dict[str, str]]:
    """Propagate *pairs* with one backend and persist predictions and metrics.

    Each target creates ``<target>.npz`` containing ``label``, ``valid``,
    ``keyframe``, and ``target``.  ``metrics.csv`` records the corresponding
    all-pixel and forward-backward-valid mIoU values.

    Pairs are decoded by a ``loader_workers``-wide thread pool and processed
    ``batch_size`` pairs at a time on the GPU (one forward pass per chunk covers
    both flow directions).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = comparison_pairs(video) if pairs is None else pairs
    if not pairs:
        raise ValueError("no comparison pairs to process")

    device = getattr(model, "device", torch.device("cpu"))
    num_classes = video.num_classes
    ignore_index = video.palette.ignore_index

    def _save(path: Path, label, valid, kf, tgt) -> None:
        np.savez_compressed(path, label=label, valid=valid,
                            keyframe=np.int64(kf), target=np.int64(tgt))

    rows: list[dict[str, str]] = []
    save_futures = []
    with ThreadPoolExecutor(max_workers=loader_workers) as pool, \
            ThreadPoolExecutor(max_workers=loader_workers) as save_pool:
        # Submit every decode up front; workers race ahead of the GPU so I/O
        # overlaps compute.  Futures are consumed in order, chunk by chunk.
        futures = [pool.submit(_load_pair, video, kf, tgt) for kf, tgt in pairs]

        for chunk in _chunks(list(zip(pairs, futures)), batch_size):
            batch = [(pair, fut.result()) for pair, fut in chunk]
            keyframes = [d[0] for _, d in batch]
            targets = [d[1] for _, d in batch]
            n = len(batch)

            # One GPU launch for both directions: first N are backward (t->k),
            # last N are forward (k->t).
            flows = model.estimate_flow_batch_tensor(
                list(zip(targets, keyframes)) + list(zip(keyframes, targets))
            )
            flow_bwd = flows[:n]
            flow_fwd = flows[n:]

            kf_masks = torch.from_numpy(
                np.stack([d[2] for _, d in batch])
            ).to(device, non_blocking=True).long()
            gt_masks = torch.from_numpy(
                np.stack([d[3] for _, d in batch])
            ).to(device, non_blocking=True).long()

            warped = warp_masks_gpu(kf_masks, flow_bwd, ignore_index=ignore_index)
            valid = fb_masks_gpu(flow_fwd, flow_bwd, threshold=fb_threshold)
            valid_pct = valid.flatten(1).float().mean(dim=1) * 100.0

            # One IoU pass per image on-device, then a single host sync for the
            # whole chunk's predictions (instead of a sync per image/save).
            ious = [
                iou_pair_gpu(warped[i], gt_masks[i], num_classes, ignore_index, valid[i])
                for i in range(n)
            ]
            warped_cpu = warped.to(torch.int32).cpu().numpy()
            valid_cpu = valid.cpu().numpy().astype(bool)
            valid_pct_cpu = valid_pct.cpu().numpy()

            for i, ((keyframe_index, target_index), _) in enumerate(batch):
                iou_all, iou_valid = ious[i]
                # Compression + disk write happen on the save pool so the GPU
                # loop is never blocked on zlib.
                save_futures.append(save_pool.submit(
                    _save, output_dir / f"{target_index:06d}.npz",
                    warped_cpu[i], valid_cpu[i], keyframe_index, target_index,
                ))
                rows.append({
                    "keyframe": str(keyframe_index),
                    "target": str(target_index),
                    "miou_all": f"{iou_all:.6f}",
                    "miou_valid": f"{iou_valid:.6f}",
                    "valid_pct": f"{float(valid_pct_cpu[i]):.2f}",
                })
                print(
                    f"[{backend_name}] {keyframe_index}->{target_index} "
                    f"mIoU(all)={iou_all:.4f} mIoU(valid)={iou_valid:.4f}",
                    flush=True,
                )

        for fut in save_futures:  # surface any write errors before exiting
            fut.result()

    rows.sort(key=lambda r: int(r["target"]))
    with open(output_dir / "metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="DJI_0101")
    parser.add_argument("--backend", required=True, choices=["sea_raft", "flownet2"])
    parser.add_argument("--frames-root", default="outputs/segprop_paper_repro/frames_2k")
    parser.add_argument("--masks-root", default="data/Ruralscapes/labels/manual_labels")
    parser.add_argument("--output-root", default="outputs/backend_predictions")
    parser.add_argument("--checkpoint-searaft", default="third_party/SEA-RAFT/checkpoints/model.safetensors")
    parser.add_argument("--searaft-config", default="third_party/SEA-RAFT/config/eval/spring-L.json")
    parser.add_argument("--checkpoint-flownet2", default="third_party/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Pairs per GPU chunk (each uses 2x images for fwd+bwd flow).")
    parser.add_argument("--loader-workers", type=int, default=16,
                        help="Threads decoding frames/masks ahead of the GPU.")
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    video = RuralscapesVideo(
        root=_REPO_ROOT,
        frame_glob=f"{args.frames_root}/{args.video}/*.jpg",
        mask_glob=f"{args.masks_root}/{args.video}/*.png",
        mask_format="color",
    )
    if args.backend == "sea_raft":
        model = SeaRaftFlow(
            args.searaft_config,
            checkpoint=args.checkpoint_searaft,
            iters=12,
            device=args.device,
        )
    else:
        model = FlowNet2Flow(args.checkpoint_flownet2, device=args.device)

    output_dir = _REPO_ROOT / args.output_root / args.video / args.backend
    run_and_cache_backend(
        video=video,
        model=model,
        backend_name=args.backend,
        output_dir=output_dir,
        batch_size=args.batch_size,
        loader_workers=args.loader_workers,
    )
    print(f"[backend] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
