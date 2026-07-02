"""Download the official FlowNet2 checkpoint referenced by the upstream README."""
from __future__ import annotations

import argparse
from pathlib import Path

import gdown

FLOWNET2_FILE_ID = "1LLYdnObBYzs7rDYYUCGscuC2nwO7y85I"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the official FlowNet2 checkpoint.")
    parser.add_argument(
        "--out",
        default="third_party/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar",
        help="Output checkpoint path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.overwrite:
        print(f"[flownet2-ckpt] exists: {out_path}")
        return 0

    url = f"https://drive.google.com/uc?id={FLOWNET2_FILE_ID}"
    print(f"[flownet2-ckpt] downloading -> {out_path}")
    gdown.download(url, str(out_path), quiet=False)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise FileNotFoundError(f"download failed: {out_path}")
    print(f"[flownet2-ckpt] done: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
