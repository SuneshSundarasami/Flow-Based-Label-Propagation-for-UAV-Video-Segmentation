"""C5: visualize worst propagation pairs from the full-video results."""
from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

import cv2
import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import load_config  # noqa: E402
from data import RuralscapesVideo  # noqa: E402
from eval.decay import read_rows  # noqa: E402
from eval.failures import select_worst_pairs  # noqa: E402
from flow import SeaRaftFlow  # noqa: E402
from propagation import propagate_keyframe  # noqa: E402

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


def _resize(img: np.ndarray, w: int, h: int, interp=cv2.INTER_LINEAR) -> np.ndarray:
    return cv2.resize(img, (w, h), interpolation=interp)


def _colorize(label: np.ndarray, video: RuralscapesVideo) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cid, rgb in video.palette.rgb.items():
        out[label == cid] = rgb
    return out


def _overlay(frame: np.ndarray, mask_color: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return (frame * (1 - alpha) + mask_color * alpha).astype(np.uint8)


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _error_map(frame: np.ndarray, warped: np.ndarray, gt: np.ndarray,
               valid: np.ndarray, ignore_index: int) -> np.ndarray:
    out = frame.copy()
    eval_mask = valid & (gt != ignore_index)
    correct = eval_mask & (warped == gt)
    wrong = eval_mask & (warped != gt)
    invalid = ~valid
    out[correct] = (0.55 * out[correct] + 0.45 * np.array([40, 180, 70])).astype(np.uint8)
    out[wrong] = (0.45 * out[wrong] + 0.55 * np.array([220, 40, 40])).astype(np.uint8)
    out[invalid] = (0.50 * out[invalid] + 0.50 * np.array([80, 80, 80])).astype(np.uint8)
    return out


def _write_index(out_dir: Path, records: list[dict]) -> None:
    lines = [
        "# C5 Failure Cases",
        "",
        "Worst pairs are selected by lowest `miou_valid` in the C1 full-video table.",
        "In the error panel, green means correct valid pixels, red means valid",
        "misclassified pixels, and gray means forward-backward invalid pixels.",
        "",
        "| Rank | Keyframe | Target | Distance | mIoU(valid) | Valid % | Figure |",
        "|------|----------|--------|----------|-------------|---------|--------|",
    ]
    for rec in records:
        lines.append(
            f"| {rec['rank']} | {rec['keyframe']} | {rec['target_frame']} | "
            f"{rec['distance']} | {rec['miou_valid']:.4f} | "
            f"{rec['valid_pct']:.1f} | [{rec['figure']}]({rec['figure']}) |"
        )
    lines.extend([
        "",
        "Interpretation: low-quality pairs usually combine longer propagation",
        "distance with lower valid-pixel coverage, which points to parallax,",
        "motion discontinuities, occlusion, or small/rare classes changing shape",
        "between keyframe and target.",
        "",
    ])
    (out_dir / "failure_cases.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cfg = _pre_load_config()
    video_name = _video_name(cfg, cfg["data"]["frame_glob"])
    default_root = _REPO_ROOT / cfg["paths"]["output_dir"] / "results" / video_name
    default_in = default_root / "propagation_results" / "all_pairs.csv"
    default_out = default_root / "failure_cases"

    ap = argparse.ArgumentParser(description="C5: visualize worst propagation pairs.")
    ap.add_argument("--config", default=None, metavar="YAML",
                    help="Override YAML deep-merged on top of default.yaml.")
    ap.add_argument("--input", default=str(default_in),
                    help="C1 all_pairs.csv path. [default: %(default)s]")
    ap.add_argument("--out-dir", default=str(default_out),
                    help="Directory for C5 outputs. [default: %(default)s]")
    ap.add_argument("--n", type=int, default=2,
                    help="Number of worst pairs to visualize. [default: %(default)s]")
    ap.add_argument("--metric", default="miou_valid",
                    help="Metric column used to rank failures. [default: %(default)s]")
    ap.add_argument("--min-valid-pct", type=float, default=5.0,
                    help="Ignore rows with fewer FB-valid pixels. [default: %(default)s]")
    ap.add_argument("--device", default=cfg["flow"]["device"],
                    help="Torch device for SEA-RAFT. [default: %(default)s]")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    if not in_path.exists():
        print(f"[c5] missing input CSV: {in_path}")
        return 1

    rows = read_rows(in_path)
    worst = select_worst_pairs(
        rows,
        metric=args.metric,
        n=args.n,
        min_valid_pct=args.min_valid_pct,
    )
    if not worst:
        print(
            f"[c5] no finite values for metric '{args.metric}' "
            f"with valid_pct >= {args.min_valid_pct} in {in_path}"
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    video = RuralscapesVideo(
        root=cfg["paths"]["dataset_root"],
        frame_glob=cfg["data"]["frame_glob"],
        mask_glob=cfg["data"]["mask_glob"],
        mask_format=cfg["data"]["mask_format"],
    )
    model = SeaRaftFlow(
        cfg["flow"]["model_cfg"],
        checkpoint=str(_REPO_ROOT / cfg["paths"]["sea_raft_checkpoint"]),
        iters=cfg["flow"]["iters"],
        device=args.device,
    )

    records: list[dict] = []
    for rank, row in enumerate(worst, start=1):
        kf_idx = int(row["keyframe"])
        tgt_idx = int(row["target_frame"])
        kf_frame = _resize(video.load_frame(kf_idx), _VIZ_W, _VIZ_H)
        tgt_frame = _resize(video.load_frame(tgt_idx), _VIZ_W, _VIZ_H)
        kf_mask = _resize(video.load_mask(kf_idx), _VIZ_W, _VIZ_H,
                          interp=cv2.INTER_NEAREST)
        gt_mask = _resize(video.load_mask(tgt_idx), _VIZ_W, _VIZ_H,
                          interp=cv2.INTER_NEAREST)

        result = propagate_keyframe(
            keyframe_mask=kf_mask,
            keyframe_frame=kf_frame,
            target_frames=[tgt_frame],
            target_indices=[tgt_idx],
            keyframe_index=kf_idx,
            model=model,
            fb_threshold=cfg["occlusion"]["fb_threshold"],
            ignore_index=cfg["eval"]["ignore_index"],
            gt_masks=[gt_mask],
            num_classes=video.num_classes,
        )[0]

        miou = result.iou["valid_only"]["miou"] if result.iou else math.nan
        panel_kf = _label(_overlay(kf_frame, _colorize(kf_mask, video)),
                          f"keyframe {kf_idx} GT")
        panel_warp = _label(_overlay(tgt_frame, _colorize(result.warped_mask, video)),
                            f"warped to {tgt_idx}  mIoU={miou:.3f}")
        panel_gt = _label(_overlay(tgt_frame, _colorize(gt_mask, video)),
                          f"target {tgt_idx} GT")
        panel_err = _label(
            _error_map(tgt_frame, result.warped_mask, gt_mask, result.valid_mask,
                       cfg["eval"]["ignore_index"]),
            f"error / validity  valid={result.valid_pct:.1f}%",
        )

        fig = np.concatenate([panel_kf, panel_warp, panel_gt, panel_err], axis=1)
        figure_name = f"failure_{rank}_kf{kf_idx}_to_{tgt_idx}.png"
        iio.imwrite(out_dir / figure_name, fig)
        records.append({
            "rank": rank,
            "keyframe": kf_idx,
            "target_frame": tgt_idx,
            "distance": result.distance,
            "miou_valid": miou,
            "valid_pct": result.valid_pct,
            "figure": figure_name,
        })
        print(f"[c5] wrote       : {out_dir / figure_name}")

    _write_index(out_dir, records)
    print(f"[c5] input       : {in_path}")
    print(
        f"[c5] selected    : {len(records)} worst pairs by {args.metric} "
        f"(valid_pct >= {args.min_valid_pct})"
    )
    print(f"[c5] wrote       : {out_dir / 'failure_cases.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
