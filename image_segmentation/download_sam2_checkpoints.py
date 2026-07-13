#!/usr/bin/env python3
"""Download SAM2 Hiera checkpoints needed by the hiera_* backbones.

    python download_sam2_checkpoints.py                  # small + base_plus
    python download_sam2_checkpoints.py --sizes small
    python download_sam2_checkpoints.py --dir sam2_checkpoints --sizes tiny small base_plus large

Fetches straight from Meta's public SAM2 release bucket (the same URLs
https://github.com/facebookresearch/sam2#sam-2-checkpoints points at) and
skips any file that's already the right size, so re-running is a no-op.
"""
import argparse
import os
import urllib.request

BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824"
SIZES = ("tiny", "small", "base_plus", "large")


def download(size: str, out_dir: str):
    name = f"sam2_hiera_{size}.pt"
    url = f"{BASE_URL}/{name}"
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest):
        print(f"[skip] {dest} already exists")
        return
    print(f"[get] {url} -> {dest}")
    os.makedirs(out_dir, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="sam2_checkpoints", help="output directory")
    ap.add_argument("--sizes", nargs="+", default=["small", "base_plus"],
                    choices=SIZES, help="which checkpoint sizes to fetch")
    args = ap.parse_args()
    for size in args.sizes:
        download(size, args.dir)


if __name__ == "__main__":
    main()
