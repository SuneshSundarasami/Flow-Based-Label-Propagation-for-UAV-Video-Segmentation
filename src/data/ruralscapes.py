"""Ruralscapes loader (WP A3).

Ruralscapes is a UAV/aerial dataset of high-resolution videos with dense semantic
segmentation masks annotated on a subset of frames. The exact on-disk layout
varies by how you extracted it, so this loader is deliberately configurable:

* **Frames** are discovered with a glob (e.g. ``frames/*.jpg``). Each frame's
  integer index is parsed from its filename (the trailing number).
* **Masks** are discovered with a glob and matched to frames by index. They are
  either:
    - ``indexed``: single-channel PNGs whose pixel value is the class id, or
    - ``color``:   RGB PNGs decoded against the palette (see ``palette.yaml``).

Only frames that have a corresponding mask are exposed as "annotated" / keyframe
candidates; all frames are available for warping targets.

This is the project's main unknown (see plan.md). Once you have the dataset,
verify: the class palette, the ignore index, and the annotation spacing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

try:  # pillow is the lightest robust image reader; fall back to imageio.
    from PIL import Image

    def _imread(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path))
except Exception:  # pragma: no cover
    import imageio.v3 as iio

    def _imread(path: Path) -> np.ndarray:
        return np.asarray(iio.imread(path))


_INDEX_RE = re.compile(r"(\d+)(?=\D*$)")  # last run of digits in the stem


def _parse_index(path: Path) -> int:
    m = _INDEX_RE.search(path.stem)
    if not m:
        raise ValueError(f"could not parse a frame index from '{path.name}'")
    return int(m.group(1))


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------
_PALETTE_PATH = Path(__file__).with_name("palette.yaml")


@dataclass
class Palette:
    ignore_index: int
    ids: List[int]
    names: Dict[int, str]
    rgb: Dict[int, Tuple[int, int, int]]

    @property
    def num_classes(self) -> int:
        return len(self.ids)


def load_palette(path: str | Path | None = None) -> Palette:
    path = Path(path) if path else _PALETTE_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    ids, names, rgb = [], {}, {}
    for c in data["classes"]:
        ids.append(c["id"])
        names[c["id"]] = c["name"]
        if c.get("rgb") is not None:
            rgb[c["id"]] = tuple(c["rgb"])
    return Palette(ignore_index=data.get("ignore_index", 255),
                   ids=ids, names=names, rgb=rgb)


def _build_rgb_lut(palette: Palette) -> Tuple[np.ndarray, np.ndarray]:
    """Return (colors Nx3 int32, ids N) arrays for vectorized color->id mapping."""
    colors = np.array([palette.rgb[i] for i in palette.ids if i in palette.rgb],
                      dtype=np.int32)
    ids = np.array([i for i in palette.ids if i in palette.rgb], dtype=np.int32)
    return colors, ids


def color_mask_to_index(mask_rgb: np.ndarray, palette: Palette,
                        tol: int = 0) -> np.ndarray:
    """Map an (H, W, 3) RGB mask to an (H, W) int label image via the palette.

    Pixels matching no class (within ``tol`` per channel) get ``ignore_index``.
    """
    colors, ids = _build_rgb_lut(palette)
    h, w = mask_rgb.shape[:2]
    flat = mask_rgb[..., :3].reshape(-1, 3).astype(np.int32)
    out = np.full(flat.shape[0], palette.ignore_index, dtype=np.int32)
    # nearest palette color by L-inf distance; assign if within tolerance.
    # Vectorized over classes (N is small: ~12).
    best = np.full(flat.shape[0], np.iinfo(np.int32).max, dtype=np.int32)
    for color, cid in zip(colors, ids):
        d = np.abs(flat - color).max(axis=1)
        hit = d < best
        out[hit] = cid
        best[hit] = d[hit]
    if tol > 0:
        out[best > tol] = palette.ignore_index
    else:
        out[best != 0] = palette.ignore_index
    return out.reshape(h, w)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
@dataclass
class FramePair:
    keyframe_index: int
    target_index: int
    keyframe: np.ndarray            # (H, W, 3) uint8 RGB
    target: np.ndarray             # (H, W, 3) uint8 RGB
    keyframe_mask: np.ndarray      # (H, W) int label image
    target_mask: Optional[np.ndarray]  # (H, W) int or None if target unlabelled

    @property
    def distance(self) -> int:
        return abs(self.target_index - self.keyframe_index)


class RuralscapesVideo:
    """Indexed access to one Ruralscapes video's frames and dense masks.

    Parameters
    ----------
    root : directory containing this video's frames and masks.
    frame_glob : glob (relative to root) selecting RGB frames.
    mask_glob : glob (relative to root) selecting masks.
    mask_format : "indexed" or "color".
    palette_path : palette YAML; defaults to the bundled template.
    """

    def __init__(
        self,
        root: str | Path,
        frame_glob: str = "frames/*.jpg",
        mask_glob: str = "masks/*.png",
        mask_format: str = "color",
        palette_path: str | Path | None = None,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"dataset root not found: {self.root}")
        if mask_format not in ("indexed", "color"):
            raise ValueError("mask_format must be 'indexed' or 'color'")
        self.mask_format = mask_format
        self.palette = load_palette(palette_path)

        self.frames: Dict[int, Path] = {
            _parse_index(p): p for p in sorted(self.root.glob(frame_glob))
        }
        self.masks: Dict[int, Path] = {
            _parse_index(p): p for p in sorted(self.root.glob(mask_glob))
        }
        if not self.frames:
            raise FileNotFoundError(
                f"no frames matched '{frame_glob}' under {self.root}")
        self.frame_indices: List[int] = sorted(self.frames)
        self.annotated_indices: List[int] = sorted(
            i for i in self.masks if i in self.frames)

    # -- counts -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.frame_indices)

    @property
    def num_classes(self) -> int:
        return self.palette.num_classes

    # -- single-item access -------------------------------------------------
    def load_frame(self, index: int) -> np.ndarray:
        img = _imread(self.frames[index])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return np.ascontiguousarray(img[..., :3]).astype(np.uint8)

    def load_mask(self, index: int) -> np.ndarray:
        if index not in self.masks:
            raise KeyError(f"frame {index} has no annotated mask")
        raw = _imread(self.masks[index])
        if self.mask_format == "indexed":
            if raw.ndim == 3:
                raw = raw[..., 0]
            return raw.astype(np.int32)
        return color_mask_to_index(raw, self.palette)

    def has_mask(self, index: int) -> bool:
        return index in self.masks

    # -- pair access (the unit the propagation pipeline consumes) ----------
    def get_pair(self, keyframe_index: int, target_index: int) -> FramePair:
        if not self.has_mask(keyframe_index):
            raise KeyError(f"keyframe {keyframe_index} is not annotated")
        return FramePair(
            keyframe_index=keyframe_index,
            target_index=target_index,
            keyframe=self.load_frame(keyframe_index),
            target=self.load_frame(target_index),
            keyframe_mask=self.load_mask(keyframe_index),
            target_mask=self.load_mask(target_index)
            if self.has_mask(target_index) else None,
        )

    def summary(self) -> str:
        return (
            f"RuralscapesVideo(root={self.root.name}, frames={len(self.frames)}, "
            f"annotated={len(self.annotated_indices)}, "
            f"classes={self.num_classes}, ignore={self.palette.ignore_index})"
        )
