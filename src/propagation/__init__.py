"""End-to-end single-keyframe label propagation (Phase B, WP B4)."""
from .propagate import FrameResult, propagate_keyframe

__all__ = ["propagate_keyframe", "FrameResult"]
