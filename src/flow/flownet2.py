"""Persistent FlowNet2 optical-flow backend for direct mask propagation."""
from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FLOWNET2_ROOT = _REPO_ROOT / "third_party" / "flownet2-pytorch"


def _load_torch_shared_libs() -> None:
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    for name in [
        "libc10.so",
        "libtorch.so",
        "libtorch_cpu.so",
        "libtorch_python.so",
        "libc10_cuda.so",
        "libtorch_cuda.so",
    ]:
        path = torch_lib / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


class FlowNet2Flow:
    """Load FlowNet2 once and expose the project's ``FlowEstimator`` interface.

    FlowNet2 expects image pairs as ``(B, 3, 2, H, W)`` and requires spatial
    dimensions divisible by 64. Pairs are inferred sequentially because that
    is more reliable on the project's 6 GB laptop GPU than a two-direction
    batch at 2K resolution.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        if not _FLOWNET2_ROOT.exists():
            raise FileNotFoundError(f"FlowNet2 checkout not found: {_FLOWNET2_ROOT}")
        checkpoint = Path(checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"FlowNet2 checkpoint not found: {checkpoint}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for FlowNet2 inference")

        _load_torch_shared_libs()
        sys.path.insert(0, str(_FLOWNET2_ROOT))
        from models import FlowNet2  # type: ignore

        self.device = torch.device(device)
        args = type("FlowNet2Args", (), {"fp16": False, "rgb_max": 255.0})()
        self.model = FlowNet2(args).to(self.device)
        state = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()

    @staticmethod
    def _pair_tensor(img1: np.ndarray, img2: np.ndarray, device: torch.device) -> torch.Tensor:
        if img1.shape != img2.shape:
            raise ValueError(f"image shapes differ: {img1.shape} vs {img2.shape}")
        pair = np.stack([img1, img2], axis=0).transpose(3, 0, 1, 2)
        return torch.from_numpy(pair.astype(np.float32)).unsqueeze(0).to(device)

    @staticmethod
    def _pad_to_multiple_of_64(pair: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, _, height, width = pair.shape
        padded_h = ((height + 63) // 64) * 64
        padded_w = ((width + 63) // 64) * 64
        if (padded_h, padded_w) == (height, width):
            return pair, (height, width)
        padded = torch.zeros(
            (pair.shape[0], pair.shape[1], pair.shape[2], padded_h, padded_w),
            dtype=pair.dtype,
            device=pair.device,
        )
        padded[..., :height, :width] = pair
        return padded, (height, width)

    @staticmethod
    def _batch_tensor(
        pairs: list[tuple[np.ndarray, np.ndarray]], device: torch.device
    ) -> torch.Tensor:
        """Stack *pairs* into FlowNet2's ``(B, 3, 2, H, W)`` input tensor."""
        imgs1 = np.stack([a for a, _ in pairs])  # (B, H, W, 3)
        imgs2 = np.stack([b for _, b in pairs])
        # (B, 2, H, W, 3) -> (B, 3, 2, H, W)
        batch = np.stack([imgs1, imgs2], axis=1).transpose(0, 4, 1, 2, 3)
        return torch.from_numpy(batch.astype(np.float32)).to(device, non_blocking=True)

    @torch.no_grad()
    def estimate_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        pair = self._pair_tensor(img1, img2, self.device)
        pair, (height, width) = self._pad_to_multiple_of_64(pair)
        flow = self.model(pair).squeeze(0)
        return flow.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)[:height, :width]

    @torch.no_grad()
    def estimate_flow_batch_tensor(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> torch.Tensor:
        """Estimate flow for many pairs in one forward pass, returned on device.

        Returns a ``(B, 2, H, W)`` float32 tensor on ``self.device``.  All pairs
        must share spatial dimensions.
        """
        if not pairs:
            return torch.empty(0, device=self.device)
        height, width = pairs[0][0].shape[:2]
        batch = self._batch_tensor(pairs, self.device)
        batch, _ = self._pad_to_multiple_of_64(batch)
        flow = self.model(batch)  # (B, 2, H_pad, W_pad)
        return flow[..., :height, :width].contiguous()

    def estimate_flow_batch(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[np.ndarray]:
        if not pairs:
            return []
        flow = self.estimate_flow_batch_tensor(pairs)
        flow = flow.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
        return [flow[i] for i in range(len(pairs))]
