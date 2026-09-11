# Flow-Based Label Propagation for UAV Video Segmentation

DLRV project. A drone that acts on what it sees needs a semantic label for
every frame, but running a segmentation model on every 4K frame costs about
370 ms — roughly seven times longer than the flight it describes, on an A100
rather than on anything an aircraft could carry.

Consecutive frames of aerial video are nearly identical, so most of that work
is redundant. This project segments a video by running the model on a keyframe,
carrying the labels forward along optical flow, and calling the model again only
when the propagated labels can no longer be trusted. The decision is made from
the flow itself, without ground truth and without looking ahead.

A demo of the pipeline is on [YouTube](https://youtu.be/JxAIijdjqFs).

## How it works

1. **Infer.** A trained segmentation model labels the first frame.
2. **Propagate.** Backward optical flow warps the mask onto the next frame.
   Sampling is nearest-neighbour, since labels are categorical.
3. **Check.** A forward-backward consistency test marks each pixel valid or not:
   step to the claimed source and back, and reject round trips longer than
   1.5 px.
4. **Accumulate.** The validity mask is itself warped forward and intersected
   with each new check, so a pixel stays invalid once it fails. Coverage is the
   fraction still valid, and it only falls between model calls.
5. **Re-infer** when coverage drops to the threshold, which resets coverage to 1.

Flow is computed for every consecutive pair before the decision is made, so
flow is the fixed cost and the model call the marginal one. The threshold
therefore buys quality rather than speed.

## Results

On the UAVid validation split — 7 sequences, 6,307 frames at 4K, ConvNeXt-Large
with SEA-RAFT, threshold 0.95:

| | model every frame | adaptive hybrid |
|---|---:|---:|
| Runtime | 2,433 s | 1,476 s (**1.65×** faster) |
| mIoU | 0.7851 | 0.7554 (**96.2%** retained) |
| Model calls | 6,307 | 1,241 (**19.7%** of frames) |

Model calls follow motion under one fixed threshold and no per-sequence tuning:
the most dynamic sequence triggers 474 calls and the calmest 87. Where a scene
needs more inference the cost is paid in speed, not accuracy — the worst
sequence falls to 1.30× while its quality stays in line with the rest.

Segmentation backbones, each trained behind the same UPerNet decoder
(validation mIoU at the best epoch):

| Backbone | Val mIoU | Inference |
|---|---:|---:|
| ConvNeXt-Large | 0.8166 | 365 ms |
| Swin-Large | 0.8081 | 433 ms |
| Hiera-Base+ | 0.7802 | 327 ms |

Flow backends, over 90 matched Ruralscapes pairs: SEA-RAFT reaches 0.7957 mIoU
at 88.4% valid coverage against FlowNet2's 0.7951 at 88.8%, while being 3.6×
faster per pair. SEA-RAFT is the default. Running its correlation volume at
quarter resolution costs under 0.01 mIoU and is the largest single speed gain
in the flow path.

The residual quality loss concentrates on small moving objects, principally
`person` and `vehicle`: equal class weighting amplifies small regions,
nearest-neighbour warping erodes their boundaries at every step, and
independent motion is where correspondence is least reliable.

## Repository layout

```
adaptive_inference/    # the hybrid pipeline: gate, propagation, CLI runners
image_segmentation/    # UPerNet training, backbones, inference, benchmarking
src/                   # flow wrappers, warping, FB check, metrics, data loaders
scripts/               # dataset preparation and analysis entry points
thesis-report-master/  # the project report (LaTeX source + PDF)
docs/                  # technical reference, model comparison, literature notes
splits/                # dataset split definitions
third_party/           # SEA-RAFT, SSP, GeoSeg submodules
tests/                 # pytest suite
data/, outputs/        # datasets and run artifacts (git-ignored)
```

## Setup

```bash
conda env create -f environment.yml
conda activate uav-flowprop
git submodule update --init --recursive
```

Source packages live flat under `src/`; scripts add it to the path, so there is
no install step. Paths in `adaptive_inference/config.yaml` resolve against the
repository root regardless of the working directory.

## Running it

Train a segmentation backbone:

```bash
python image_segmentation/train.py \
    --config image_segmentation/configs/uavid_convnext_large.yaml
```

Run the adaptive pipeline over a whole dataset split, writing per-frame metrics
to `outputs/adaptive_inference_metrics/`:

```bash
python -m adaptive_inference.run_dataset --splits uavid_val
```

Or over a single video, optionally scored against ground truth:

```bash
python -m adaptive_inference.run --video path/to/seq16/images.mp4 \
    --gt-dir path/to/seq16/Labels --threshold 0.95
```

Useful overrides: `--threshold`, `--backbone`, `--flow-backend sea_raft|flownet2`,
`--max-frames`, and `--config` for a YAML merged onto the defaults.

Build the report:

```bash
cd thesis-report-master
python scripts/make_figures.py    # figures + generated_numbers.tex
python scripts/audit_numbers.py   # re-derives every value typed into a table
latexmk -pdf report.tex
```

## Configuration

Defaults used for every reported experiment, in
`adaptive_inference/config.yaml`:

| Setting | Value |
|---|---|
| backbone / decoder | `convnext_large` / UPerNet |
| flow backend | `sea_raft`, `spring-L`, 12 refinements |
| `scale` | −2 (quarter internal resolution) |
| `fb_threshold` | 1.5 px |
| `valid_threshold` | 0.95 |
| `steps_per_chunk` | 16 |
| precision | `bf16` |

Flow cannot be precomputed for a whole video — forward and backward fields for
900 4K frames would occupy about 120 GB — so the pipeline streams in chunks and
discards each one once its frames are labelled, keeping peak memory flat
regardless of video length.

## Datasets

**UAVid** drives the pipeline experiments: 4096×2160 aerial sequences, 6 classes
after mapping, with ten densely annotated frames per validation sequence.

**Ruralscapes** is used for the flow-backend comparison and the earlier
keyframe-propagation work, and is downloaded from the
[official project page](https://sites.google.com/site/aerialimageunderstanding/semantics-through-time-semi-supervised-segmentation-of-aerial-videos)
into `data/Ruralscapes/`:

```bash
python scripts/download_ruralscapes.py   # ~12.4 GiB, resumable
```

Both live under `data/`, which is not tracked.

## Report

[`thesis-report-master/report.pdf`](thesis-report-master/report.pdf) is the full
write-up: method, experiments, results and threats to validity. Every number in
it is regenerated from a committed result file by `make_figures.py` and checked
by `audit_numbers.py`, so the prose cannot drift from the data; the few
measurements whose runs left no artifact are marked where they appear.

## Related work in this repository

Beyond the adaptive pipeline, the repository also contains a reproduction of
SegProp-style offline propagation from human keyframes on Ruralscapes, and the
single-keyframe decay analysis that preceded it (`scripts/run_full_video.py`
and the `analyze_*.py` family). These share the warping and metric code under
`src/` and are described in [`docs/`](docs/).
