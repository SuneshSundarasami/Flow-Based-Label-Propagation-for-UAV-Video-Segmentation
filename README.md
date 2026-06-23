# Flow-Based Label Propagation for UAV Video Segmentation

DLRV project. Propagates dense segmentation labels from annotated **keyframes** to
neighbouring frames of a UAV video (Ruralscapes) along **SEA-RAFT** optical flow,
and characterises how label quality decays with distance from the keyframe.

See [`plan.md`](plan.md) for the full work-package breakdown and
[`docs/literature_notes.md`](docs/literature_notes.md) for background.

## Status

**Phase A (Foundation) — complete.** Package scaffold, SEA-RAFT wrapper +
smoke test, and the Ruralscapes loader are in place and verified. `uav-flowprop`
conda env builds; SEA-RAFT runs on CPU with a downloaded checkpoint; `DJI_0043`
data loads (142 matched frame/mask pairs, median annotation spacing 50 frames).

**Phase B (Core pipeline) — complete.**

| WP | What | Status |
|----|------|--------|
| B1 | Nearest-neighbour mask warp (`warp/mask_warp.py`) | ☑ done — 6 tests |
| B2 | Forward–backward occlusion mask (`warp/occlusion.py`) | ☑ done — 5 tests |
| B3 | mIoU + per-class IoU metric (`eval/metrics.py`) | ☑ done — 12 tests |
| B4 | End-to-end single-keyframe propagation (`propagation/`) | ☑ done — 10 tests |
| B5 | Config + CLI runner | ☑ done — 9 tests |

**Phase C (Scale, analyse, report) — C1-C5 done; C6 pending.**

| WP | What | Status |
|----|------|--------|
| C1 | Full-video batched run (`scripts/run_full_video.py`) | ☑ done — 10 tests |
| C2 | mIoU-vs-distance decay curve (`scripts/analyze_decay.py`) | ☑ done — 2 tests |
| C3 | Difficulty heatmap over timeline (`scripts/analyze_heatmap.py`) | ☑ done — 3 tests |
| C4 | Per-class IoU breakdown (`scripts/analyze_per_class.py`) | ☑ done — 2 tests |
| C5 | Failure-case visualisations (`scripts/visualize_failures.py`) | ☑ done — 2 tests |
| C6 | Report write-up | ☐ pending |

All 69 unit tests pass (`conda run -n uav-flowprop pytest -q`).

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
  flow/                # SEA-RAFT wrapper + FlowEstimator Protocol
  warp/                # mask_warp (B1) + occlusion FB check (B2)
  eval/                # mIoU / per-class IoU (B3) + results CSV schema (B5/C1)
  propagation/         # single-keyframe pipeline (B4) + target scheduling (C1)
  data/                # Ruralscapes loader + class palette
  viz/                 # flow color-wheel visualization
scripts/               # CLI entry points
  smoke_test_flow.py   # A2 flow smoke test
  inspect_data.py      # A3 data loader check
  check_warp.py        # B1/B2 visual 4-panel check
  run_propagation.py   # B4/B5 single-keyframe run with mIoU table + CSVs
  run_full_video.py    # C1 full-video batched run -> all_pairs.csv
  analyze_decay.py     # C2 mIoU-vs-distance summary + plot
  analyze_heatmap.py   # C3 keyframe x distance difficulty heatmap
  analyze_per_class.py # C4 per-class IoU breakdown
  visualize_failures.py # C5 worst-pair visualizations
  prepare_segprop_sampled.py # SegProp sampled-frame label prep
  generate_segprop_flows.py  # SegProp H5 flows via SEA-RAFT
  run_segprop_sampled.py     # SegProp vote baseline
  compare_segprop_sampled.py # compare SegProp vs matching C1 rows
tests/                 # pytest (pythonpath=src)
third_party/SEA-RAFT/  # pinned git submodule (optical flow backbone)
third_party/segprop/   # SegProp baseline code
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

## Phase B: core pipeline

### B1/B2 — warp + occlusion visual check

Warps the keyframe mask to the next 3 annotated frames and writes 4-panel
images (keyframe GT / warped / target GT / validity mask) to `outputs/warp_check/`:

```bash
conda run -n uav-flowprop python scripts/check_warp.py \
    --root data/Ruralscapes \
    --frame-glob "frames/DJI_0043/*.jpg" \
    --mask-glob "labels/manual_labels/DJI_0043/*.png" \
    --mask-format color \
    --n-targets 3 \
    --device cuda
```

Validity mask: white = FB round-trip error < 1.5 px (valid); red = occluded /
unreliable. Interior red marks arise from parallax, motion discontinuities, and
true occlusions — not only at frame borders.

### B3/B4/B5 — end-to-end propagation with mIoU table

Propagates from the first annotated keyframe to N subsequent annotated targets,
prints a distance / valid% / mIoU(all) / mIoU(valid) / per-class IoU table to
the terminal, and saves 4-panel images plus two CSVs:

- `outputs/propagation/results_all_pixels.csv` — IoU over all non-ignored pixels
- `outputs/propagation/results_valid_pixels.csv` — IoU restricted to FB-valid pixels
- `outputs/results/<video>/propagation_results/keyframe_<K>.csv` — per-pair table (`keyframe`, `target_frame`, `distance`, `valid_pct`, `miou_all`, `miou_valid`, `iou_class0`, `iou_class1`, …); this is the format C1 aggregates across all keyframes

All settings come from `src/config/default.yaml`, so a bare command is enough:

```bash
conda run -n uav-flowprop python scripts/run_propagation.py
```

Override individual settings on the CLI:

```bash
conda run -n uav-flowprop python scripts/run_propagation.py \
    --n-targets 10 --device cpu
```

Or supply an override YAML (deep-merged on top of defaults):

```bash
conda run -n uav-flowprop python scripts/run_propagation.py \
    --config experiments/long_window.yaml
```

IoU is evaluated in two modes: **all pixels** (every non-ignored pixel, for
fair baseline comparison) and **valid-only** (additionally excludes FB-invalid
pixels, the primary quality measure). Flow model: SEA-RAFT spring-L
(12 refinement iterations) — roughly 3× lower EPE than SegProp's FlowNet2
on Sintel clean.

## Phase C: scale, analyse, report

### C1 — full-video batched run

Propagates from **every** annotated keyframe to its next N annotated frames and
records per-pair IoU for the whole video:

```bash
conda run --no-capture-output -n uav-flowprop python scripts/run_full_video.py
```

`--no-capture-output` is required so that progress lines appear in real time
(without it, `conda run` buffers stdout and nothing shows until the process
exits). Each keyframe's results are cached to
`outputs/results/<video>/propagation_results/keyframe_<K>.csv`; reruns skip
keyframes already computed (use `--force` to recompute). For the default config
this is `outputs/results/DJI_0043/propagation_results/keyframe_<K>.csv`. All
per-keyframe CSVs are concatenated, sorted by `(keyframe, distance)`, into:

- `outputs/results/<video>/propagation_results/all_pairs.csv` — the complete full-video table
  (`keyframe`, `target_frame`, `distance`, `valid_pct`, `miou_all`,
  `miou_valid`, `iou_class0`, …) consumed by C2–C5.

Useful flags: `--limit N` (process only the first N keyframes — handy for a
quick partial run), `--n-targets` (annotated targets per keyframe),
`--device cpu`, and `--config <YAML>`.

### C2 — mIoU-vs-distance decay curve

Aggregates the C1 full-video table by propagation distance and writes a summary
CSV plus a plot:

```bash
conda run -n uav-flowprop python scripts/analyze_decay.py
```

Default outputs for `DJI_0043`:

- `outputs/results/DJI_0043/analysis/decay_summary.csv` — count, mean, std,
  min, and max for `miou_valid`, `miou_all`, and `valid_pct` per distance.
- `outputs/results/DJI_0043/analysis/miou_decay.png` — mIoU(valid) decay curve
  with a ±1 std band and the mean FB-valid pixel percentage.

### C3 — difficulty heatmap over timeline

Builds a keyframe-by-distance heatmap from the C1 table:

```bash
conda run -n uav-flowprop python scripts/analyze_heatmap.py
```

Default outputs for `DJI_0043`:

- `outputs/results/DJI_0043/analysis/difficulty_heatmap_matrix.csv` — matrix
  with keyframes as rows and propagation distances as columns.
- `outputs/results/DJI_0043/analysis/difficulty_heatmap.png` — timeline
  heatmap of `miou_valid`, where darker/low-value cells mark difficult
  propagation intervals.

Use `--metric miou_all` or `--metric valid_pct` to plot a different C1 metric.

### C4 — per-class IoU breakdown

Ranks semantic classes by valid-only IoU over the full-video table:

```bash
conda run -n uav-flowprop python scripts/analyze_per_class.py
```

Default outputs for `DJI_0043`:

- `outputs/results/DJI_0043/analysis/per_class_iou.csv` — per-class count,
  mean, std, min, and max IoU.
- `outputs/results/DJI_0043/analysis/per_class_iou.png` — ranked bar chart of
  mean valid-only IoU with standard-deviation bars.

### C5 — failure-case visualisations

Selects the worst propagation pairs by `miou_valid`, recomputes only those
pairs, and writes side-by-side visual checks:

```bash
conda run --no-capture-output -n uav-flowprop python scripts/visualize_failures.py
```

Default outputs for `DJI_0043`:

- `outputs/results/DJI_0043/failure_cases/failure_cases.md` — index and short
  interpretation of the selected failures.
- `outputs/results/DJI_0043/failure_cases/failure_<N>_kf<K>_to_<T>.png` —
  keyframe GT / warped target / target GT / error-validity panel.

By default C5 ignores pairs with fewer than 5% FB-valid pixels so the selected
cases are visually meaningful. Use `--n 4` to inspect more cases,
`--metric miou_all` to rank by all-pixel mIoU, or `--min-valid-pct 0` to include
near-empty valid regions.

### SegProp sampled baseline

The SegProp repository is included under `third_party/segprop`. The first
baseline comparison uses the same 142 annotated `DJI_0043` frames as C1,
re-indexed densely so SegProp can run on sampled-frame steps. Even sampled
frames are treated as source labels and odd sampled frames as held-out
evaluation labels.

```bash
conda run -n uav-flowprop python scripts/prepare_segprop_sampled.py
conda run --no-capture-output -n uav-flowprop python scripts/generate_segprop_flows.py
conda run --no-capture-output -n uav-flowprop python scripts/run_segprop_sampled.py
conda run -n uav-flowprop python scripts/compare_segprop_sampled.py
```

This is a sampled-frame SegProp-vote baseline using SEA-RAFT H5 flows, not the
paper's original dense-video FlowNet2 setup. Current `DJI_0043` result:

- Ours, matching even→odd C1 rows: `miou_all=0.7396`, `miou_valid=0.7551`
- SegProp vote sampled baseline: `fmeasure=0.8023`, `miou_all=0.7058`

The comparison table is written to
`outputs/results/DJI_0043/analysis/segprop_sampled_comparison.csv`.
