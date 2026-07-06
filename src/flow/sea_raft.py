"""Thin wrapper around the SEA-RAFT optical-flow model (WP A2).

SEA-RAFT lives in the ``third_party/SEA-RAFT`` git submodule. This module hides
the submodule's ``sys.path`` gymnastics and exposes a clean

    estimate_flow(img1, img2) -> (H, W, 2) float32 numpy array

where each image is an (H, W, 3) uint8 RGB array and the returned flow maps a
pixel in ``img1`` to its location in ``img2`` (forward flow img1 -> img2).
"""
from __future__ import annotations

import sys
import importlib.util
from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Resolve the submodule and make its ``core`` package importable. SEA-RAFT
# internally does ``import datasets`` / ``from raft import RAFT`` assuming ``core``
# is on the path, so we add both the repo root and ``core``.
# This file lives at <repo>/src/flow/sea_raft.py -> parents[2] is <repo>.
_REPO_ROOT = Path(__file__).resolve().parents[2]
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
    parser_path = _SEA_RAFT_ROOT / "config" / "parser.py"
    spec = importlib.util.spec_from_file_location("sea_raft_config_parser", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load SEA-RAFT parser from {parser_path}")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)

    args = parser.json_to_args(str(model_cfg))
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
            checkpoint = Path(checkpoint)
            if checkpoint.suffix == ".safetensors":
                from safetensors.torch import load_file

                model.load_state_dict(load_file(str(checkpoint)), strict=False)
            else:
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
        return self.estimate_flow_batch([(img1, img2)])[0]

    # cuDNN raises CUDNN_STATUS_NOT_SUPPORTED once a conv activation at 2K crosses
    # ~2**31 elements, which happens beyond ~16 images per forward pass.  Larger
    # requests are split into safe sub-batches so any caller batch size just works.
    max_images_per_launch: int = 16

    @torch.no_grad()
    def estimate_flow_batch_tensor(
        self,
        pairs: "list[tuple[np.ndarray, np.ndarray]]",
    ) -> "torch.Tensor":
        """Estimate flow for multiple pairs, returning a device tensor.

        Returns a ``(B, 2, H, W)`` float32 tensor on ``self.device`` so callers
        can keep the flow on the GPU for downstream warping/metrics without a
        host round-trip.  All pairs must share spatial dimensions.  Requests
        larger than :attr:`max_images_per_launch` are run in sub-batches and
        concatenated (see the cuDNN note above).
        """
        from utils.utils import InputPadder  # type: ignore

        if not pairs:
            return torch.empty(0, device=self.device)

        cap = self.max_images_per_launch
        if len(pairs) > cap:
            parts = [
                self.estimate_flow_batch_tensor(pairs[i:i + cap])
                for i in range(0, len(pairs), cap)
            ]
            return torch.cat(parts, dim=0)

        h, w = pairs[0][0].shape[:2]
        t1s = torch.stack(
            [torch.from_numpy(a).float().permute(2, 0, 1) for a, _ in pairs]
        ).to(self.device, non_blocking=True)  # (B, 3, H, W)
        t2s = torch.stack(
            [torch.from_numpy(b).float().permute(2, 0, 1) for _, b in pairs]
        ).to(self.device, non_blocking=True)  # (B, 3, H, W)

        padder = InputPadder(t1s.shape)
        t1s, t2s = padder.pad(t1s, t2s)
        flow = self._calc_flow(t1s, t2s)   # (B, 2, H_pad, W_pad)
        flow = padder.unpad(flow)           # (B, 2, H, W)
        return flow[..., :h, :w].contiguous()

    @torch.no_grad()
    def estimate_flow_batch(
        self,
        pairs: "list[tuple[np.ndarray, np.ndarray]]",
    ) -> "list[np.ndarray]":
        """Estimate flow for multiple (img1, img2) pairs in one forward pass.

        All pairs must have the same spatial dimensions. Returns a list of
        (H, W, 2) float32 arrays in the same order as *pairs*.
        """
        if not pairs:
            return []
        flow = self.estimate_flow_batch_tensor(pairs)
        flow = flow.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        return [flow[i] for i in range(len(pairs))]


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
