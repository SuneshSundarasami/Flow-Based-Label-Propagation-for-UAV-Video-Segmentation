"""Optical flow estimation (SEA-RAFT backbone)."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .sea_raft import SeaRaftFlow, estimate_flow


@runtime_checkable
class FlowEstimator(Protocol):
    """Contract for any optical flow backend.

    Implement this to make a new model a drop-in replacement for SeaRaftFlow.
    Both images must be (H, W, 3) uint8 RGB arrays; returned flow is
    (H, W, 2) float32 in pixels (forward: img1 -> img2).
    """

    def estimate_flow(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray: ...


__all__ = ["FlowEstimator", "SeaRaftFlow", "estimate_flow"]
