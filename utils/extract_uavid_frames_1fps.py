"""Extract every UAVid sequence video at 1 FPS.

UAVid ships each sequence as ``<seq>/images.mp4`` at its native 20 FPS
alongside 10 GT-labelled frames spaced 5 seconds apart. This grabs one frame
per second from every sequence's video (across uavid_train/val/test), which
is denser than the GT but far lighter than the full 20 FPS stream.

    python utils/extract_uavid_frames_1fps.py
    python utils/extract_uavid_frames_1fps.py --root data/uavid/uavid_v1.5_official_release \\
        --out data/uavid/uavid_v1.5_official_release_1fps --overwrite

Frames are named by their actual frame index in the source video (e.g.
frame_000000.jpg, frame_000020.jpg, ... for a 20 FPS video), so they line up
directly with the GT filenames (000000.png, 000100.png, ...) for cross-checking.

Saved as JPEG (quality 90, ~3MB/frame) rather than PNG (~21MB/frame) — at this
quality the extra compression is negligible next to the source video's own
lossy encoding, and keeps ~1,900 frames across all sequences to ~6GB instead
of ~40GB.

Runs sequentially (one sequence at a time) by default: each video is decoded
by reading forward and keeping every Nth frame rather than seeking to each
target frame (seeking on this codec is far slower than a straight read), and
a single video's decode already saturates most CPU cores internally via
FFmpeg's own threading -- running many sequences at once via --workers just
causes core oversubscription and ends up slower, not faster.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import cv2

SPLITS = ("uavid_train", "uavid_val", "uavid_test")
JPEG_QUALITY = 90


def extract_video(video_path: Path, out_dir: Path, overwrite: bool) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps))  # frames between samples for ~1 FPS

    out_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % step == 0:
            out_path = out_dir / f"frame_{index:06d}.jpg"
            if out_path.exists() and not overwrite:
                skipped += 1
            else:
                if not cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
                    raise SystemExit(f"failed to write {out_path}")
                written += 1
        index += 1

    cap.release()
    return written, skipped


def _worker(job: tuple[Path, Path, bool]) -> tuple[str, int, int]:
    video_path, out_dir, overwrite = job
    written, skipped = extract_video(video_path, out_dir, overwrite)
    return f"{out_dir.parent.name}/{out_dir.name}", written, skipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/uavid/uavid_v1.5_official_release",
                     help="UAVid release root (contains uavid_train/uavid_val/uavid_test)")
    ap.add_argument("--out", default="data/uavid/uavid_v1.5_official_release_1fps",
                     help="output root; mirrors <split>/<seq>/ under --root")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                     help="parallel worker processes (default: 1 -- see module docstring "
                          "for why more usually doesn't help here)")
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)

    videos = sorted(root.glob("*/*/images.mp4"))
    videos = [v for v in videos if v.parent.parent.name in SPLITS]
    if not videos:
        raise SystemExit(f"no sequence videos found under {root}/<split>/<seq>/images.mp4")

    jobs = [
        (v, out_root / v.parent.parent.name / v.parent.name, args.overwrite)
        for v in videos
    ]
    workers = max(1, args.workers)

    total_written = total_skipped = 0
    with mp.Pool(processes=workers) as pool:
        for seq_label, written, skipped in pool.imap_unordered(_worker, jobs):
            total_written += written
            total_skipped += skipped
            print(f"[frames] {seq_label}: wrote {written}, skipped {skipped}")

    print(f"[frames] done: {len(videos)} sequences, {total_written} written, "
          f"{total_skipped} skipped, {workers} workers -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
