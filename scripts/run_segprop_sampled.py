"""Run the cloned SegProp vote baseline on sampled annotated frames."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "segprop"))

from config import load_config  # noqa: E402
import segprop  # noqa: E402
import stats  # noqa: E402


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


def main() -> int:
    cfg = _pre_load_config()
    video_name = _video_name(cfg, cfg["data"]["frame_glob"])
    default_root = (
        _REPO_ROOT / cfg["paths"]["output_dir"] / "segprop_baseline" / video_name
    )

    ap = argparse.ArgumentParser(
        description="Run SegProp vote baseline on sampled annotated frames."
    )
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="Override YAML deep-merged on top of default.yaml.")
    ap.add_argument("--work-root", default=str(default_root),
                    help="SegProp baseline working directory. [default: %(default)s]")
    ap.add_argument("--device", default="cpu",
                    help="Torch device for SegProp. [default: %(default)s]")
    ap.add_argument("--force", action="store_true",
                    help="Recompute existing SegProp outputs.")
    args = ap.parse_args()

    work_root = Path(args.work_root)
    flow_paths = (
        work_root / "flow_960" / f"{video_name}_forward.h5",
        work_root / "flow_960" / f"{video_name}_backward.h5",
    )
    train_even = work_root / "labels_960" / "train_even" / video_name
    train_odd = work_root / "labels_960" / "train_odd" / video_name
    output_dir = work_root / "output_960" / "i01" / video_name

    missing = [p for p in (*flow_paths, train_even, train_odd) if not p.exists()]
    if missing:
        print("[segprop-run] missing required inputs:")
        for p in missing:
            print(f"  - {p}")
        return 1

    segprop.vote(
        (str(flow_paths[0]), str(flow_paths[1])),
        str(train_even),
        str(output_dir),
        precalc_flow=False,
        device=args.device,
        overwrite=args.force,
    )
    fmeasure, miou = stats.evaluate(
        str(work_root / "output_960" / "i01"),
        str(work_root / "labels_960" / "train_odd"),
    )

    print(f"[segprop-run] output     : {output_dir}")
    print(f"[segprop-run] fmeasure   : {fmeasure:.4f}")
    print(f"[segprop-run] miou       : {miou:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
