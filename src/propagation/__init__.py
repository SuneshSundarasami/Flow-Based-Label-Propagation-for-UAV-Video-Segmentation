"""End-to-end single-keyframe label propagation (Phase B, WP B4)."""
from .propagate import FrameResult, forward_targets, propagate_keyframe

__all__ = ["propagate_keyframe", "FrameResult", "forward_targets"]
