"""Thin wrapper around the SEA-RAFT optical-flow model (WP A2).

SEA-RAFT lives in the ``third_party/SEA-RAFT`` git submodule. This module hides
the submodule's ``sys.path`` gymnastics and exposes a clean

    estimate_flow(img1, img2) -> (H, W, 2) float32 numpy array

where each image is an (H, W, 3) uint8 RGB array and the returned flow maps a
pixel in ``img1`` to its location in ``img2`` (forward flow img1 -> img2).
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Resolve the submodule and make its ``core`` package importable. SEA-RAFT
# internally does ``import datasets`` / ``from raft import RAFT`` assuming ``core``
# is on the path, so we add both the repo root and ``core``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEA_RAFT_ROOT = _REPO_ROOT / "third_party" / "SEA-RAFT"
_SEA_RAFT_CORE = _SEA_RAFT_ROOT / "core"


def _ensure_on_path() -> None:
    for p in (_SEA_RAFT_ROOT, _SEA_RAFT_CORE):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    if not _SEA_RAFT_ROOT.exists():
        raise FileNotFoundError(
            f"SEA-RAFT submodule not found at {_SEA_RAFT_ROOT}. Run "
            "`git submodule update --init --recursive`."
        )


def _load_model_args(model_cfg: str | Path, iters: int) -> Namespace:
    _ensure_on_path()
    from config.parser import json_to_args  # type: ignore

    args = json_to_args(str(model_cfg))
    args.iters = iters
    # Fields the demo path expects but the json may not set.
    for field, default in (("scale", 0), ("var_min", 0), ("var_max", 10)):
        if not hasattr(args, field):
            setattr(args, field, default)
    return args


class SeaRaftFlow:
    """Loads SEA-RAFT once and estimates flow for image pairs.

    Parameters
    ----------
    model_cfg : path to a SEA-RAFT eval JSON (e.g. config/eval/spring-M.json).
    checkpoint : local ``.pth`` path. Mutually exclusive with ``url``.
    url : HuggingFace id (e.g. ``MemorySlices/Tartan-C-T-TSKH-spring540x960-M``).
    iters : recurrent refinement iterations at inference.
    device : "cuda" or "cpu".
    """

    def __init__(
        self,
        model_cfg: str | Path,
        checkpoint: Optional[str | Path] = None,
        url: Optional[str] = None,
        iters: int = 4,
        device: str = "cuda",
    ):
        if (checkpoint is None) == (url is None):
            raise ValueError("provide exactly one of `checkpoint` or `url`")
        _ensure_on_path()
        from raft import RAFT  # type: ignore
        from utils.utils import load_ckpt  # type: ignore

        self.args = _load_model_args(model_cfg, iters)
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )

        if checkpoint is not None:
            model = RAFT(self.args)
            load_ckpt(model, str(checkpoint))
        else:
            model = RAFT.from_pretrained(url, args=self.args)

        self.model = model.to(self.device).eval()

    # -- internal -----------------------------------------------------------
    def _calc_flow(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        """Mirror SEA-RAFT's calc_flow: scale -> forward -> rescale flow."""
        scale = getattr(self.args, "scale", 0)
        img1 = F.interpolate(image1, scale_factor=2 ** scale, mode="bilinear",
                             align_corners=False)
        img2 = F.interpolate(image2, scale_factor=2 ** scale, mode="bilinear",
                             align_corners=False)
        output = self.model(img1, img2, iters=self.args.iters, test_mode=True)
        flow = output["flow"][-1]
        flow = F.interpolate(flow, scale_factor=0.5 ** scale, mode="bilinear",
                             align_corners=False) * (0.5 ** scale)
        return flow

    # -- public -------------------------------------------------------------
    @torch.no_grad()
    def estimate_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Estimate forward flow (img1 -> img2). Returns (H, W, 2) float32."""
        from utils.utils import InputPadder  # type: ignore

        if img1.shape != img2.shape or img1.ndim != 3 or img1.shape[2] != 3:
            raise ValueError(
                f"expected matching (H, W, 3) RGB images, got {img1.shape} and "
                f"{img2.shape}"
            )
        h, w = img1.shape[:2]
        t1 = torch.from_numpy(img1).float().permute(2, 0, 1)[None].to(self.device)
        t2 = torch.from_numpy(img2).float().permute(2, 0, 1)[None].to(self.device)

        padder = InputPadder(t1.shape)
        t1, t2 = padder.pad(t1, t2)
        flow = self._calc_flow(t1, t2)
        flow = padder.unpad(flow)[0]  # (2, H, W)
        return flow.permute(1, 2, 0).cpu().numpy().astype(np.float32)[:h, :w]


def estimate_flow(
    img1: np.ndarray,
    img2: np.ndarray,
    model_cfg: str | Path,
    checkpoint: Optional[str | Path] = None,
    url: Optional[str] = None,
    iters: int = 4,
    device: str = "cuda",
) -> np.ndarray:
    """One-shot convenience wrapper (loads the model each call).

    Prefer :class:`SeaRaftFlow` when processing many pairs, so the checkpoint is
    loaded only once.
    """
    model = SeaRaftFlow(model_cfg, checkpoint=checkpoint, url=url, iters=iters,
                        device=device)
    return model.estimate_flow(img1, img2)
