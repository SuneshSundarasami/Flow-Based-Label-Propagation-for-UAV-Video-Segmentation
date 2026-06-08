"""Download a SEA-RAFT checkpoint from HuggingFace into the submodule.

The pipeline can also load weights directly via ``--url`` (no local file), but
caching a ``.pth`` locally makes offline runs reproducible.

Usage
-----
    python scripts/download_checkpoint.py \
        --repo MemorySlices/Tartan-C-T-TSKH-spring540x960-M
"""
from __future__ import annotations

import argparse
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEST_DIR = _REPO_ROOT / "third_party" / "SEA-RAFT" / "checkpoints"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
                    help="HuggingFace model repo id")
    ap.add_argument("--filename", default=None,
                    help="specific file to download; defaults to the repo's .pth")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files

    _DEST_DIR.mkdir(parents=True, exist_ok=True)
    filename = args.filename
    if filename is None:
        files = list_repo_files(args.repo)
        pth = [f for f in files if f.endswith(".pth") or f.endswith(".safetensors")]
        if not pth:
            raise SystemExit(f"no weight file found in {args.repo}: {files}")
        filename = pth[0]

    print(f"[download] {args.repo}:{filename} -> {_DEST_DIR}")
    path = hf_hub_download(repo_id=args.repo, filename=filename,
                           local_dir=str(_DEST_DIR))
    print(f"[download] saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
