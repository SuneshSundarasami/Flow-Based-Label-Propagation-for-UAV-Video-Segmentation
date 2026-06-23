"""Prepare sampled-frame Ruralscapes inputs for the cloned SegProp baseline.

This first baseline uses the annotated frames already exported for C1. The
frames are re-indexed densely (0, 1, 2, ...) so SegProp can treat consecutive
annotated frames as consecutive time steps. Even sampled indices are used as
source labels; odd sampled indices are held out for evaluation.
"""
from __future__ import annotations

import argparse
import csv
from shutil import copyfile
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import load_config  # noqa: E402
from data import RuralscapesVideo  # noqa: E402

_VIZ_W, _VIZ_H = 960, 540


def _pre_load_config(argv=None) -> dict:
    if argv is None:
        argv = sys.argv[1:]
    cfg_path = None
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            cfg_path = argv[i + 1]
            break
        if arg.startswith("--config="):
            cfg_path = arg.split("=", 1)[1]
            break
    return load_config(override_path=cfg_path)


def _video_name(cfg: dict, frame_glob: str) -> str:
    parts = Path(frame_glob).parts
    if len(parts) >= 2 and parts[0] == "frames":
        return parts[1]
    return str(cfg.get("paths", {}).get("video", "video"))


def _one_hot(mask: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((*mask.shape, num_classes), dtype=bool)
    for cid in range(num_classes):
        out[..., cid] = mask == cid
    return out


def main() -> int:
    cfg = _pre_load_config()
    video_name = _video_name(cfg, cfg["data"]["frame_glob"])
    default_out = (
        _REPO_ROOT / cfg["paths"]["output_dir"] / "segprop_baseline" / video_name
    )

    ap = argparse.ArgumentParser(
        description="Prepare sampled annotated frames for the SegProp baseline."
    )
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="Override YAML deep-merged on top of default.yaml.")
    ap.add_argument("--out-root", default=str(default_out),
                    help="SegProp baseline working directory. [default: %(default)s]")
    args = ap.parse_args()

    video = RuralscapesVideo(
        root=cfg["paths"]["dataset_root"],
        frame_glob=cfg["data"]["frame_glob"],
        mask_glob=cfg["data"]["mask_glob"],
        mask_format=cfg["data"]["mask_format"],
    )
    out_root = Path(args.out_root)
    labels_root = out_root / "labels_960"
    video_dir = video_name
    for split in ("all", "train_even", "train_odd"):
        (labels_root / split / video_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for sampled_idx, frame_idx in enumerate(video.annotated_indices):
        mask = cv2.resize(
            video.load_mask(frame_idx),
            (_VIZ_W, _VIZ_H),
            interpolation=cv2.INTER_NEAREST,
        )
        one_hot = _one_hot(mask, video.num_classes)
        name = f"{video_name}_{sampled_idx:06d}.npz"
        all_path = labels_root / "all" / video_dir / name
        np.savez(all_path, map=one_hot, votes=one_hot)
        split = "train_even" if sampled_idx % 2 == 0 else "train_odd"
        copyfile(all_path, labels_root / split / video_dir / name)
        rows.append({"sampled_index": sampled_idx, "frame_index": frame_idx})

    map_path = out_root / "sampled_frame_map.csv"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sampled_index", "frame_index"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[segprop-prep] video      : {video_name}")
    print(f"[segprop-prep] labels     : {len(rows)} sampled frames")
    print(f"[segprop-prep] size       : {_VIZ_W}x{_VIZ_H}")
    print(f"[segprop-prep] wrote      : {labels_root}")
    print(f"[segprop-prep] wrote      : {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
