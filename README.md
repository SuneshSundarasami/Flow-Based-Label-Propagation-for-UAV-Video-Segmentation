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
real pair. A3 is verified on the real Ruralscapes `DJI_0043` video/labels.

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

```bash
unzip data/Ruralscapes.zip -d data
python scripts/export_labelled_frames.py \
    --video data/Ruralscapes/videos/DJI_0043.MP4 \
    --labels data/Ruralscapes/labels/manual_labels/DJI_0043 \
    --out data/Ruralscapes/frames/DJI_0043
```

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

The smoke test runs two checks and writes results to `outputs/smoke/`:

- **Synthetic pair** — a red/blue checkerboard rotated by 5°, giving a
  spatially-varying flow field (different direction and magnitude at every
  pixel). Outputs: `synthetic_img1.png`, `synthetic_img2.png`,
  `synthetic_flow.png` (colour-wheel + sparse arrow overlay).
- **Real pair** — the two sample frames shipped with SEA-RAFT (`custom/`),
  which appear to be from the Spring benchmark. Outputs: `real_img1.png`,
  `real_img2.png`, `real_flow.png` (colour-wheel + sparse arrow overlay).

The colour-wheel images are annotated with `draw_flow_arrows` from
`src/viz/flow_viz.py`, which overlays a sparse grid of arrows (one per 60 px)
so direction is readable at a glance. Use `--device cuda` when GPU access is
available.

**Ruralscapes loader (WP A3)** — verified on `DJI_0043`:

```bash
python scripts/inspect_data.py --root data/Ruralscapes \
    --frame-glob "frames/DJI_0043/*.jpg" \
    --mask-glob "labels/manual_labels/DJI_0043/*.png" \
    --mask-format color
```

This prints the frame range, annotation count/spacing, mask encoding/unique ids,
a class legend, and writes frame|mask side-by-side previews to
`outputs/data_check/`. Pass `--n-samples N` to save overlays for the first N
annotated frames (default 1). The verified `DJI_0043` run loaded 142 matched
frame/mask pairs with median annotation spacing of 50 frames.
The loader/inspection path can be checked without the full dataset using the
committed fixture:

```bash
python scripts/inspect_data.py --root tests/fixtures/ruralscapes_demo \
    --frame-glob "frames/*.ppm" --mask-glob "masks/*.ppm" \
    --mask-format color --palette tests/fixtures/ruralscapes_demo/palette.yaml
```

The RGB palette in [`src/data/palette.yaml`](src/data/palette.yaml) follows
`CLASS_COLORS_RGB` from the official SegProp preprocessing script; class-name
ordering should still be treated carefully before reporting per-class results.
