"""Dataset loading for Ruralscapes UAV video + dense segmentation masks."""
from .ruralscapes import RuralscapesVideo, FramePair, load_palette

__all__ = ["RuralscapesVideo", "FramePair", "load_palette"]
