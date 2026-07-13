"""Adaptive hybrid segmentation: infer a keyframe with a trained segmentation
model, propagate it forward with optical flow, and re-infer once cumulative
forward-backward-consistency coverage drops below a configured threshold.

    from adaptive_inference.config import load_config
    from adaptive_inference.models import build_segmentation_model, build_flow_model
    from adaptive_inference.pipeline import decode_video, run_adaptive_pipeline

    cfg = load_config()
    frames = decode_video("path/to/video.mp4")
    seg_model = build_segmentation_model(cfg)
    flow_model = build_flow_model(cfg)
    for result in run_adaptive_pipeline(frames, seg_model, flow_model, cfg):
        ...  # result.mask, result.source, result.valid_pct

See config.yaml for tunables and README.md for background on the threshold
trade-off and the memory pitfalls this pipeline works around.
"""
from .config import Config, load_config
from .pipeline import StepResult, decode_video, run_adaptive_pipeline

__all__ = ["Config", "load_config", "StepResult", "decode_video", "run_adaptive_pipeline"]
