"""Launch the FlowNet2 pair runner in a fresh GPU-enabled WSL session."""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def _quote(value: str) -> str:
    return shlex.quote(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FlowNet2 pair inference via fresh WSL.")
    parser.add_argument("--img1", required=True)
    parser.add_argument("--img2", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--checkpoint",
        default="/home/workuser/DLRV/Project/third_party/flownet2-pytorch/checkpoints/FlowNet2_checkpoint.pth.tar",
    )
    parser.add_argument("--distro", default="Ubuntu-24.04")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    inner = (
        f"cd {_quote(str(repo_root))} && "
        "export LD_LIBRARY_PATH=/home/workuser/miniconda3/envs/uav-flowprop/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH && "
        "/home/workuser/miniconda3/envs/uav-flowprop/bin/python scripts/flownet2_pair.py "
        f"--img1 {_quote(args.img1)} "
        f"--img2 {_quote(args.img2)} "
        f"--out {_quote(args.out)} "
        f"--checkpoint {_quote(args.checkpoint)}"
    )

    command = [
        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
        "-NoProfile",
        "-Command",
        f'wsl -d {args.distro} bash -lc {_quote(inner)}',
    ]
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
