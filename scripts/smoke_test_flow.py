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
from viz import flow_to_color  # noqa: E402


def _make_synthetic_pair(h=540, w=960, shift=(12, 6), seed=0):
    """Random-texture image + a copy translated by ``shift`` = (dx, dy)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    base = np.repeat(np.repeat(base[::4, ::4], 4, axis=0), 4, axis=1)[:h, :w]
    dx, dy = shift
    shifted = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
    return base, shifted, (dx, dy)


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

    # --- Check 1: synthetic translation -----------------------------------
    a, b, (dx, dy) = _make_synthetic_pair()
    flow = model.estimate_flow(a, b)
    # ignore a border where the roll wraps around
    m = 12
    inner = flow[m:-m, m:-m]
    med = np.median(inner.reshape(-1, 2), axis=0)
    print(f"[smoke] synthetic: expected flow ~({dx}, {dy}), "
          f"recovered median=({med[0]:.2f}, {med[1]:.2f})")
    iio.imwrite(out_dir / "synthetic_flow.png", flow_to_color(flow))
    ok = abs(med[0] - dx) < 1.5 and abs(med[1] - dy) < 1.5
    print(f"[smoke] synthetic check: {'PASS' if ok else 'CHECK MANUALLY'}")

    # --- Check 2: real frame pair (optional) ------------------------------
    if args.img1 and args.img2:
        i1 = np.asarray(iio.imread(args.img1))[..., :3]
        i2 = np.asarray(iio.imread(args.img2))[..., :3]
        rflow = model.estimate_flow(i1, i2)
        iio.imwrite(out_dir / "real_flow.png", flow_to_color(rflow))
        mag = np.linalg.norm(rflow, axis=2)
        print(f"[smoke] real pair: flow magnitude min/mean/max = "
              f"{mag.min():.2f}/{mag.mean():.2f}/{mag.max():.2f}")
        print(f"[smoke] wrote {out_dir/'real_flow.png'}")

    print(f"[smoke] visualizations in {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
