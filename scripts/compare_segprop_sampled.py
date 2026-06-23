"""Compare sampled-frame SegProp vote baseline against our matching C1 rows."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "third_party" / "segprop"))

from config import load_config  # noqa: E402
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


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def _our_even_to_odd_rows(all_pairs: Path, frame_map: Path) -> list[dict]:
    sampled_to_frame = {
        int(row["sampled_index"]): int(row["frame_index"])
        for row in _read_csv(frame_map)
    }
    wanted = {
        (sampled_to_frame[i], sampled_to_frame[i + 1])
        for i in range(0, max(sampled_to_frame), 2)
        if i in sampled_to_frame and i + 1 in sampled_to_frame
    }
    return [
        row for row in _read_csv(all_pairs)
        if (int(row["keyframe"]), int(row["target_frame"])) in wanted
    ]


def main() -> int:
    cfg = _pre_load_config()
    video_name = _video_name(cfg, cfg["data"]["frame_glob"])
    default_results = (
        _REPO_ROOT / cfg["paths"]["output_dir"] / "results" / video_name
    )
    default_segprop = (
        _REPO_ROOT / cfg["paths"]["output_dir"] / "segprop_baseline" / video_name
    )

    ap = argparse.ArgumentParser(
        description="Compare SegProp sampled baseline with matching C1 rows."
    )
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="Override YAML deep-merged on top of default.yaml.")
    ap.add_argument("--results-root", default=str(default_results),
                    help="Our results root. [default: %(default)s]")
    ap.add_argument("--segprop-root", default=str(default_segprop),
                    help="SegProp baseline root. [default: %(default)s]")
    args = ap.parse_args()

    results_root = Path(args.results_root)
    segprop_root = Path(args.segprop_root)
    all_pairs = results_root / "propagation_results" / "all_pairs.csv"
    frame_map = segprop_root / "sampled_frame_map.csv"
    segprop_output = segprop_root / "output_960" / "i01"
    train_odd = segprop_root / "labels_960" / "train_odd"
    out_csv = results_root / "analysis" / "segprop_sampled_comparison.csv"

    required = [all_pairs, frame_map, segprop_output, train_odd]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("[segprop-compare] missing required inputs:")
        for p in missing:
            print(f"  - {p}")
        return 1

    segprop_f, segprop_miou = stats.evaluate(str(segprop_output), str(train_odd))
    ours = _our_even_to_odd_rows(all_pairs, frame_map)
    ours_all = [_to_float(row["miou_all"]) for row in ours]
    ours_valid = [_to_float(row["miou_valid"]) for row in ours]
    ours_all = [v for v in ours_all if not math.isnan(v)]
    ours_valid = [v for v in ours_valid if not math.isnan(v)]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "method": "ours_searaft_direct",
            "eval_pairs": len(ours),
            "fmeasure": "nan",
            "miou_all": f"{mean(ours_all):.4f}",
            "miou_valid": f"{mean(ours_valid):.4f}",
            "notes": "Matching train_even->train_odd sampled rows from C1",
        },
        {
            "method": "segprop_vote_sampled_searaft_flow",
            "eval_pairs": "train_odd",
            "fmeasure": f"{segprop_f:.4f}",
            "miou_all": f"{segprop_miou:.4f}",
            "miou_valid": "nan",
            "notes": "SegProp vote on sampled annotated frames, evaluated with SegProp stats",
        },
    ]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[segprop-compare] ours pairs : {len(ours)}")
    print(f"[segprop-compare] ours mIoU : all={mean(ours_all):.4f} valid={mean(ours_valid):.4f}")
    print(f"[segprop-compare] segprop   : f={segprop_f:.4f} miou={segprop_miou:.4f}")
    print(f"[segprop-compare] wrote     : {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
