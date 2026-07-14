"""Build the segmentation model and the optical-flow model from config.

Both live in sibling packages (image_segmentation/, src/) that aren't on
sys.path by default and both happen to define a module named ``data`` --
image_segmentation puts itself first on sys.path so its ``data``/``models``
win the name over src's Ruralscapes-specific ``data`` package.

This has to unconditionally move both paths to the front (not just insert if
absent): Python auto-prepends a running script's own directory to
sys.path[0], so a script living inside image_segmentation/ (e.g.
benchmark_models.py) already has that path present, just at a lower-priority
position -- a plain "insert if not present" check would then skip it and
leave src/ (inserted fresh at index 0) winning the name collision.
"""
from __future__ import annotations

import sys

import torch

from .config import REPO_ROOT, Config

_IMG_SEG = REPO_ROOT / "image_segmentation"
_SRC = REPO_ROOT / "src"
for _p in (str(_SRC), str(_IMG_SEG)):  # image_segmentation last -> highest priority
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


def build_segmentation_model(cfg: Config) -> torch.nn.Module:
    from models import build_model  # image_segmentation/models.py

    seg_cfg = cfg.segmentation
    model = build_model(
        seg_cfg.backbone,
        num_classes=len(class_names()),
        sam2_checkpoint_dir=str(cfg.resolve(seg_cfg.sam2_checkpoint_dir)),
        upsampling=seg_cfg.upsampling,
        decoder_style=seg_cfg.decoder_style,
    )
    ckpt = torch.load(cfg.resolve(seg_cfg.checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(cfg.adaptive.device).eval()
    return model


def build_flow_model(cfg: Config):
    flow_cfg = cfg.flow
    if flow_cfg.backend == "sea_raft":
        from flow import SeaRaftFlow  # src/flow

        model = SeaRaftFlow(
            model_cfg=str(cfg.resolve(flow_cfg.model_cfg)),
            checkpoint=str(cfg.resolve(flow_cfg.checkpoint)),
            iters=flow_cfg.iters,
            device=cfg.adaptive.device,
        )
        model.args.scale = flow_cfg.scale
        model.max_images_per_launch = 2 * flow_cfg.steps_per_chunk
        return model
    if flow_cfg.backend == "flownet2":
        from flow import FlowNet2Flow  # src/flow

        return FlowNet2Flow(
            checkpoint=str(cfg.resolve(flow_cfg.flownet2_checkpoint)),
            device=cfg.adaptive.device,
        )
    raise ValueError(f"unknown flow backend: {flow_cfg.backend!r}")


def class_names() -> list[str]:
    from data import CLASSES  # image_segmentation/data.py
    return CLASSES


def normalization_stats() -> tuple[tuple[float, ...], tuple[float, ...]]:
    from data import _MEAN, _STD  # image_segmentation/data.py
    return _MEAN, _STD


def ignore_index() -> int:
    from data import IGNORE_INDEX  # image_segmentation/data.py
    return IGNORE_INDEX
