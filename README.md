# Flow-Based Label Propagation for UAV Video Segmentation

DLRV project. Propagates dense segmentation labels from annotated **keyframes** to
neighbouring frames of a UAV video (Ruralscapes) along **SEA-RAFT** optical flow,
and characterises how label quality decays with distance from the keyframe.

See [`plan.md`](plan.md) for the full work-package breakdown and
[`docs/literature_notes.md`](docs/literature_notes.md) for background.

## Status

**Phase A (Foundation) — in progress.** Package scaffold, SEA-RAFT wrapper +
smoke test, and the Ruralscapes loader are in place. A1 is verified: the
`uav-flowprop` conda environment builds, local packages import with `src` on the
Python path, and the Phase A lightweight tests pass. A2 is verified on CPU with
a downloaded SEA-RAFT checkpoint, one synthetic pair, and the SEA-RAFT sample
real pair. A3's loader/inspection path is fixture-verified; final real-data
verification still needs the Ruralscapes dataset.

## Setup (conda)

```bash
# 1. create and activate the environment
conda env create -f environment.yml
conda activate uav-flowprop

# 2. pull the SEA-RAFT submodule (if not cloned with --recurse-submodules)
git submodule update --init --recursive
```

The source packages live flat under `src/`; scripts and tests add `src/` to the
path automatically, so no install step is required.

Verify the A1 scaffold:

```bash
PYTHONPATH=src conda run -n uav-flowprop python -c \
    "import config, data, flow, viz, warp, eval; print('imports ok')"
conda run -n uav-flowprop pytest -q
```

## Repository layout

```
src/                   # source packages (flat)
  config/              # default.yaml + loader (deep-merge overrides)
  flow/                # SEA-RAFT wrapper: estimate_flow(img1, img2)
  warp/                # mask warping + FB occlusion        (Phase B)
  eval/                # mIoU / per-class IoU                (Phase B)
  data/                # Ruralscapes loader + class palette
  viz/                 # flow color-wheel visualization
scripts/               # CLI entry points (smoke tests, downloads)
tests/                 # pytest (pythonpath=src)
third_party/SEA-RAFT/  # pinned git submodule (optical flow backbone)
docs/                  # literature notes, write-ups
data/                  # dataset goes here, OUTSIDE src (git-ignored)
outputs/               # run artifacts (git-ignored)
Proposal/              # the original DLRV proposal (LaTeX + PDF)
```

## Dataset

Ruralscapes was downloaded from the official project page:
<https://sites.google.com/site/aerialimageunderstanding/semantics-through-time-semi-supervised-segmentation-of-aerial-videos>.

The local archive is expected at `data/Ruralscapes.zip` and is intentionally not
tracked by git. The archive contains `Ruralscapes/videos/*.MP4` and dense manual
labels under `Ruralscapes/labels/manual_labels/<video>/segfull_*.png`. For A3,
extract the archive and export frames from the selected MP4 so the loader can
match frame indices against the labelled masks.

## Phase A: verifying the foundation

**SEA-RAFT optical flow (WP A2)** — download/cache a checkpoint and run the smoke
test:

```bash
python scripts/download_checkpoint.py \
    --repo MemorySlices/Tartan-C-T-TSKH-spring540x960-M
python scripts/smoke_test_flow.py --device cpu \
    --img1 third_party/SEA-RAFT/custom/image1.jpg \
    --img2 third_party/SEA-RAFT/custom/image2.jpg
```

The verified CPU run recovered synthetic flow median `(11.85, 6.74)` for an
expected `(12, 6)` shift and wrote `synthetic_flow.png` / `real_flow.png` to
`outputs/smoke/`. Use `--device cuda` when GPU access is available.

**Ruralscapes loader (WP A3)** — once the dataset is under `data/`:

```bash
python scripts/inspect_data.py --root data/ruralscapes/<video> \
    --frame-glob "frames/*.jpg" --mask-glob "masks/*.png" --mask-format color
```

This prints the frame range, annotation count/spacing, mask encoding/unique ids,
a class legend, and writes a frame|mask preview to `outputs/data_check/`.
The loader/inspection path can be checked without the full dataset using the
committed fixture:

```bash
python scripts/inspect_data.py --root tests/fixtures/ruralscapes_demo \
    --frame-glob "frames/*.ppm" --mask-glob "masks/*.ppm" \
    --mask-format color --palette tests/fixtures/ruralscapes_demo/palette.yaml
```

**Verify the class palette** in
[`src/data/palette.yaml`](src/data/palette.yaml)
against the real dataset — the placeholder RGB values are not authoritative.
