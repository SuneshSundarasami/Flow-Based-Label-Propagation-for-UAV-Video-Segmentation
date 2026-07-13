"""Config loading for the adaptive hybrid pipeline.

Deep-merges the bundled ``config.yaml`` with an optional override file and/or
a dict of CLI overrides, then wraps the result in small dataclasses so the
rest of the package gets attribute access instead of dict lookups.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_PATH = Path(__file__).with_name("config.yaml")


def _deep_merge(base: dict, override: Mapping[str, Any]) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class SegmentationConfig:
    backbone: str
    checkpoint: str
    decoder_style: str = "ssp"
    upsampling: str = "bilinear"
    sam2_checkpoint_dir: str = "image_segmentation/sam2_checkpoints"


@dataclass
class FlowConfig:
    backend: str = "sea_raft"
    model_cfg: str = "third_party/SEA-RAFT/config/eval/spring-L.json"
    checkpoint: str = "third_party/SEA-RAFT/checkpoints/model.safetensors"
    flownet2_checkpoint: str = "third_party/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar"
    iters: int = 12
    scale: int = -2
    fb_threshold: float = 1.5
    precision: str = "bf16"
    steps_per_chunk: int = 16


@dataclass
class AdaptiveConfig:
    valid_threshold: float = 0.95
    device: str = "cuda"


@dataclass
class OutputConfig:
    save_masks: bool = True
    format: str = "palette_png"
    out_dir: str = "outputs/adaptive_inference"


@dataclass
class Config:
    segmentation: SegmentationConfig
    flow: FlowConfig = field(default_factory=FlowConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def resolve(self, rel_path: str) -> Path:
        """Resolve a config path relative to the repo root."""
        p = Path(rel_path)
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(override_path: str | Path | None = None,
                 overrides: Mapping[str, Any] | None = None) -> Config:
    with open(_DEFAULT_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if override_path is not None:
        with open(override_path, "r", encoding="utf-8") as fh:
            raw = _deep_merge(raw, yaml.safe_load(fh) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)

    return Config(
        segmentation=SegmentationConfig(**raw["segmentation"]),
        flow=FlowConfig(**raw["flow"]),
        adaptive=AdaptiveConfig(**raw["adaptive"]),
        output=OutputConfig(**raw["output"]),
    )
