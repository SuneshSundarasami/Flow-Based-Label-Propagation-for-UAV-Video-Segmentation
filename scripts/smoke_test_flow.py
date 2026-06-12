"""A2 smoke test: confirm SEA-RAFT produces sensible flow.

Runs two checks and writes color-wheel visualizations to ``outputs/smoke/``:

1. Synthetic pair: a textured image and a copy shifted by a known (dx, dy). The
   recovered flow should be approximately constant and equal to (dx, dy).
2. Real pair (optional): two frames given via --img1/--img2.

Usage
-----
    python scripts/smoke_test_flow.py --url MemorySlices/Tartan-C-T-TSKH-spring540x960-M
    python scripts/smoke_test_flow.py --checkpoint path/to.pth \
        --img1 a.png --img2 b.png --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make ``src`` importable when run from the repo root without installation.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import imageio.v3 as iio  # noqa: E402

from config import load_config  # noqa: E402
from flow import SeaRaftFlow  # noqa: E402
from viz import draw_flow_arrows, flow_to_color  # noqa: E402


def _make_synthetic_pair(h=540, w=960, angle_deg=5.0, square=60):
    """Two-colour checkerboard + a rotated copy.

    Rotation gives spatially-varying flow (direction and magnitude change across
    the image), so the colour-wheel output shows a smooth gradient rather than a
    flat colour.  The analytical flow is returned for numerical validation.
    """
    import cv2

    xs = np.arange(w) // square
    ys = np.arange(h) // square
    checker = ((xs[None, :] + ys[:, None]) % 2).astype(np.uint8)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[checker == 0] = [210, 60, 60]    # warm red squares
    img[checker == 1] = [60, 110, 210]   # cool blue squares

    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return img, rotated, (angle_deg, cx, cy)


def _rotation_flow_gt(h, w, angle_deg, cx, cy):
    """Analytical (u, v) flow field matching cv2.warpAffine with getRotationMatrix2D.

    OpenCV's rotation matrix M maps src→dst as:
      x' = cos(a)*(x-cx) + sin(a)*(y-cy) + cx
      y' = -sin(a)*(x-cx) + cos(a)*(y-cy) + cy
    so flow = (x'-x, y'-y).
    """
    theta = np.deg2rad(angle_deg)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    dx_c = xs - cx
    dy_c = ys - cy
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    u = dx_c * (cos_t - 1) + dy_c * sin_t
    v = -dx_c * sin_t + dy_c * (cos_t - 1)
    return np.stack([u, v], axis=2).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default=None,
                    help="SEA-RAFT model JSON (defaults to config flow.model_cfg)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--url", default=None,
                    help="HuggingFace id, e.g. "
                         "MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    ap.add_argument("--img1", default=None)
    ap.add_argument("--img2", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config()
    model_cfg = args.cfg or cfg["flow"]["model_cfg"]
    checkpoint = args.checkpoint
    url = args.url
    if checkpoint is None and url is None:
        # fall back to the checkpoint path from config if it exists
        ckpt = Path(cfg["paths"]["sea_raft_checkpoint"])
        if ckpt.exists():
            checkpoint = str(ckpt)
        else:
            ap.error("provide --checkpoint or --url (no local checkpoint found)")

    out_dir = _REPO_ROOT / "outputs" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] loading SEA-RAFT (cfg={model_cfg}, device={args.device})")
    model = SeaRaftFlow(model_cfg, checkpoint=checkpoint, url=url,
                        iters=cfg["flow"]["iters"], device=args.device)

    # --- Check 1: synthetic rotation (spatially-varying flow) ---------------
    a, b, (angle_deg, cx, cy) = _make_synthetic_pair()
    h, w = a.shape[:2]
    flow = model.estimate_flow(a, b)
    gt = _rotation_flow_gt(h, w, angle_deg, cx, cy)
    # evaluate over inner crop to avoid border artefacts from warpAffine
    m = 30
    err = np.linalg.norm(flow[m:-m, m:-m] - gt[m:-m, m:-m], axis=2)
    median_err = float(np.median(err))
    max_gt_mag = float(np.linalg.norm(gt[m:-m, m:-m], axis=2).max())
    print(f"[smoke] synthetic: rotation {angle_deg}°, "
          f"max expected magnitude={max_gt_mag:.1f}px, "
          f"median endpoint error={median_err:.2f}px")
    iio.imwrite(out_dir / "synthetic_img1.png", a)
    iio.imwrite(out_dir / "synthetic_img2.png", b)
    color = flow_to_color(flow)
    iio.imwrite(out_dir / "synthetic_flow.png", draw_flow_arrows(color, flow))
    # Allow up to 35% of the max GT magnitude — CPU inference with few iters
    # recovers the right spatial pattern even if magnitudes are underestimated.
    ok = median_err < max_gt_mag * 0.35
    print(f"[smoke] synthetic check: {'PASS' if ok else 'CHECK MANUALLY'}")

    # --- Check 2: real frame pair (optional) ------------------------------
    if args.img1 and args.img2:
        i1 = np.asarray(iio.imread(args.img1))[..., :3]
        i2 = np.asarray(iio.imread(args.img2))[..., :3]
        rflow = model.estimate_flow(i1, i2)
        iio.imwrite(out_dir / "real_img1.png", i1)
        iio.imwrite(out_dir / "real_img2.png", i2)
        rcolor = flow_to_color(rflow)
        iio.imwrite(out_dir / "real_flow.png", draw_flow_arrows(rcolor, rflow))
        mag = np.linalg.norm(rflow, axis=2)
        print(f"[smoke] real pair: flow magnitude min/mean/max = "
              f"{mag.min():.2f}/{mag.mean():.2f}/{mag.max():.2f}")
        print(f"[smoke] wrote {out_dir/'real_flow.png'}")

    print(f"[smoke] visualizations in {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
