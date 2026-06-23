"""Generate SegProp-compatible H5 flow files from SEA-RAFT.

The cloned SegProp code expects two H5 files with a ``flow`` dataset:
forward sampled-frame flow ``i -> i+1`` and backward sampled-frame flow
``i+1 -> i``. Values are stored as (x, y) flow vectors, matching SegProp's
README; SegProp flips them internally to its (y, x) coordinate convention.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import load_config  # noqa: E402
from data import RuralscapesVideo  # noqa: E402
from flow import SeaRaftFlow  # noqa: E402

warnings.filterwarnings("ignore", message=".*meshgrid.*indexing.*", category=UserWarning)

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


def _resize(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (_VIZ_W, _VIZ_H), interpolation=cv2.INTER_LINEAR)


def _open_flow_file(path: Path, n: int, force: bool) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    if force and path.exists():
        path.unlink()
    fh = h5py.File(path, "a")
    if "flow" not in fh:
        fh.create_dataset(
            "flow",
            shape=(n, _VIZ_H, _VIZ_W, 2),
            dtype="float32",
            chunks=(1, _VIZ_H, _VIZ_W, 2),
            compression="gzip",
            compression_opts=1,
        )
        fh.create_dataset("done", shape=(n,), dtype="bool")
    return fh


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    cfg = _pre_load_config()
    video_name = _video_name(cfg, cfg["data"]["frame_glob"])
    default_out = (
        _REPO_ROOT / cfg["paths"]["output_dir"] / "segprop_baseline" / video_name
    )

    ap = argparse.ArgumentParser(
        description="Generate SEA-RAFT sampled-frame H5 flows for SegProp."
    )
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="Override YAML deep-merged on top of default.yaml.")
    ap.add_argument("--out-root", default=str(default_out),
                    help="SegProp baseline working directory. [default: %(default)s]")
    ap.add_argument("--device", default=cfg["flow"]["device"],
                    help="Torch device for SEA-RAFT. [default: %(default)s]")
    ap.add_argument("--force", action="store_true",
                    help="Recompute all flow pairs from scratch.")
    args = ap.parse_args()

    video = RuralscapesVideo(
        root=cfg["paths"]["dataset_root"],
        frame_glob=cfg["data"]["frame_glob"],
        mask_glob=cfg["data"]["mask_glob"],
        mask_format=cfg["data"]["mask_format"],
    )
    ann = video.annotated_indices
    if len(ann) < 2:
        print("[segprop-flow] need at least 2 annotated frames")
        return 1

    out_root = Path(args.out_root)
    flow_dir = out_root / "flow_960"
    fwd_path = flow_dir / f"{video_name}_forward.h5"
    bwd_path = flow_dir / f"{video_name}_backward.h5"
    n_pairs = len(ann) - 1

    model = SeaRaftFlow(
        cfg["flow"]["model_cfg"],
        checkpoint=str(_REPO_ROOT / cfg["paths"]["sea_raft_checkpoint"]),
        iters=cfg["flow"]["iters"],
        device=args.device,
    )

    t0 = time.time()
    with _open_flow_file(fwd_path, n_pairs, args.force) as fwd_h5, _open_flow_file(
        bwd_path, n_pairs, args.force
    ) as bwd_h5:
        todo = [
            i for i in range(n_pairs)
            if args.force or not (bool(fwd_h5["done"][i]) and bool(bwd_h5["done"][i]))
        ]
        print(f"[segprop-flow] video      : {video_name}")
        print(f"[segprop-flow] pairs      : {n_pairs} ({len(todo)} to compute)")
        print(f"[segprop-flow] output     : {flow_dir}")
        print(f"[segprop-flow] device     : {model.device}")

        for i in tqdm(todo, desc="segprop flows", unit="pair", dynamic_ncols=True):
            a_idx, b_idx = ann[i], ann[i + 1]
            frame_a = _resize(video.load_frame(a_idx))
            frame_b = _resize(video.load_frame(b_idx))
            flow_ab, flow_ba = model.estimate_flow_batch([
                (frame_a, frame_b),
                (frame_b, frame_a),
            ])
            fwd_h5["flow"][i] = flow_ab.astype(np.float32)
            bwd_h5["flow"][i] = flow_ba.astype(np.float32)
            fwd_h5["done"][i] = True
            bwd_h5["done"][i] = True
            fwd_h5.flush()
            bwd_h5.flush()

    print(f"[segprop-flow] wrote      : {fwd_path}")
    print(f"[segprop-flow] wrote      : {bwd_path}")
    print(f"[segprop-flow] total      : {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
