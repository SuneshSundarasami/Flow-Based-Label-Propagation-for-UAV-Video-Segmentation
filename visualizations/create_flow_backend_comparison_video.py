"""Render cached SEA-RAFT and FlowNet2 predictions into one comparison video.

Inference is intentionally not performed here.  First run
``python -m propagation.backend_runner --backend sea_raft`` and the matching
FlowNet2 command; this script only reads their NPZ/CSV results.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data import RuralscapesVideo  # noqa: E402


def _palette_lut(video: RuralscapesVideo) -> np.ndarray:
    """RGB lookup table indexed by class id; ids without a colour stay black."""
    size = max(256, max(video.palette.rgb) + 1)
    lut = np.zeros((size, 3), dtype=np.uint8)
    for class_id, rgb in video.palette.rgb.items():
        lut[class_id] = rgb
    return lut


def _colorize(label: np.ndarray, video: RuralscapesVideo,
              lut: np.ndarray | None = None) -> np.ndarray:
    # Single LUT gather instead of one full-image boolean pass per class.
    if lut is None:
        lut = _palette_lut(video)
    return lut[np.clip(label, 0, len(lut) - 1)]


def _overlay(frame: np.ndarray, label: np.ndarray, video: RuralscapesVideo,
             lut: np.ndarray | None = None) -> np.ndarray:
    return (0.5 * frame + 0.5 * _colorize(label, video, lut)).astype(np.uint8)


def _panel(image: np.ndarray, caption: str, width: int, height: int) -> np.ndarray:
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel, (0, 0), (width, 42), (0, 0, 0), -1)
    cv2.putText(panel, caption, (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _metrics(path: Path) -> dict[int, dict[str, str]]:
    with open(path, newline="") as fh:
        return {int(row["target"]): row for row in csv.DictReader(fh)}


def _load_prediction(path: Path) -> tuple[np.ndarray, int]:
    with np.load(path) as saved:
        return saved["label"].astype(np.int32), int(saved["keyframe"])


def _resized_mask(video: RuralscapesVideo, index: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    return cv2.resize(video.load_mask(index), (width, height), interpolation=cv2.INTER_NEAREST)


def _transcode_to_h264(source: Path, destination: Path) -> None:
    """Re-encode *source* to an H.264 MP4 that Chromium/VSCode can play back.

    OpenCV can only emit MPEG-4 Part 2 (``mp4v``) here, which VSCode's HTML5
    video preview cannot decode.  ``yuv420p`` and ``+faststart`` keep the result
    broadly compatible and quick to start streaming.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(destination),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default="DJI_0101")
    parser.add_argument("--frames-root", default="outputs/segprop_paper_repro/frames_2k")
    parser.add_argument("--masks-root", default="data/Ruralscapes/labels/manual_labels")
    parser.add_argument("--predictions-root", default="outputs/backend_predictions")
    parser.add_argument("--out-dir", default="visualizations")
    # Each panel is Full HD by default, producing a 2x2 4K (3840x2160) video.
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--gif-width", type=int, default=1280,
                        help="Width of the preview GIF (height auto); the MP4 stays full 4K. 0 skips the GIF.")
    args = parser.parse_args()

    video = RuralscapesVideo(
        root=_REPO_ROOT,
        frame_glob=f"{args.frames_root}/{args.video}/*.jpg",
        mask_glob=f"{args.masks_root}/{args.video}/*.png",
        mask_format="color",
    )
    prediction_root = _REPO_ROOT / args.predictions_root / args.video
    sea_dir = prediction_root / "sea_raft"
    fn2_dir = prediction_root / "flownet2"
    sea_metrics = _metrics(sea_dir / "metrics.csv")
    fn2_metrics = _metrics(fn2_dir / "metrics.csv")
    targets = sorted(set(sea_metrics) & set(fn2_metrics))
    if not targets:
        raise RuntimeError("no target frames shared by the SEA-RAFT and FlowNet2 caches")

    out_dir = _REPO_ROOT / args.out_dir / args.video
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "searaft_vs_flownet2_direct.mp4"
    # OpenCV can only write MPEG-4 Part 2 here, so render to a temporary file and
    # re-encode it to H.264 in place below.
    raw_path = out_dir / "searaft_vs_flownet2_direct.raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                             (args.width * 2, args.height * 2))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {raw_path}")
    lut = _palette_lut(video)  # built once; reused for every overlay
    try:
        for target_index in targets:
            sea_label, sea_keyframe = _load_prediction(sea_dir / f"{target_index:06d}.npz")
            fn2_label, fn2_keyframe = _load_prediction(fn2_dir / f"{target_index:06d}.npz")
            if sea_keyframe != fn2_keyframe:
                raise ValueError(f"keyframe mismatch for target {target_index}: {sea_keyframe} vs {fn2_keyframe}")
            target = video.load_frame(target_index)
            target_mask = _resized_mask(video, target_index, target.shape[:2])
            top = np.concatenate([
                _panel(target, f"{args.video} frame {target_index} original", args.width, args.height),
                _panel(_overlay(target, sea_label, video, lut),
                       f"SEA-RAFT direct  mIoU={float(sea_metrics[target_index]['miou_all']):.3f}",
                       args.width, args.height),
            ], axis=1)
            bottom = np.concatenate([
                _panel(_overlay(target, fn2_label, video, lut),
                       f"FlowNet2 direct  mIoU={float(fn2_metrics[target_index]['miou_all']):.3f}",
                       args.width, args.height),
                _panel(_overlay(target, target_mask, video, lut),
                       f"Ground truth frame {target_index}", args.width, args.height),
            ], axis=1)
            writer.write(np.concatenate([top, bottom], axis=0))
    finally:
        writer.release()

    _transcode_to_h264(raw_path, out_path)
    raw_path.unlink(missing_ok=True)
    print(f"[viz-backends] wrote {out_path}")

    if args.gif_width > 0:
        # Downscaled preview GIF (the MP4 above keeps full 4K). scale=-2 keeps the
        # aspect ratio with an even height that the encoder accepts.
        gif_path = out_dir / "searaft_vs_flownet2_direct_preview.gif"
        filter_graph = (
            f"fps={args.fps:g},scale={args.gif_width}:-2:flags=lanczos,"
            "split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse"
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(out_path), "-vf", filter_graph, str(gif_path)],
            check=True,
        )
        print(f"[viz-backends] wrote {gif_path} ({args.gif_width}px wide)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
