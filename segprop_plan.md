# Paper-Like SegProp Table 1 and Table 3 Reproduction Plan

## Summary

Reproduce the SegProp paper's **Table 1 label-propagation evaluation** and
**Table 3 propagation-module ablation** on the Ruralscapes **training split**,
using **FlowNet2 optical flow**, **2K paper-like resolution**, and SegProp's
`vote -> iterate -> denoise` pipeline.

Target Table 1 result from `docs/2010.01910v1.pdf`:

| Method | Final mF1 | Final mIoU |
|---|---:|---:|
| SegProp from Alg. 1 + filtering | ~0.903 | ~0.829 |

Target Table 3 ablation results:

| Method | Iteration | Target overall mF1 |
|---|---:|---:|
| Zhu et al. [6] | 1 | 0.846 |
| SegProp | 1 | 0.884 |
| SegProp + Zhu et al. [6] | 1 | 0.892 |
| SegProp + Homography | 1 | 0.894 |
| SegProp + Homography + Filtering | 1 | 0.904 |

This plan still excludes the paper's Table 2 semi-supervised neural-network
training experiments.

## Current Status

- Shared preparation is complete locally for the official training split:
  13 videos and 37,666 2K frames, labels, and metadata.
- FlowNet2 H5 generation and the Table 1 runner are implemented and tested.
- A full 2K `DJI_0101` Table 1 run completed: i01 mF1/mIoU is
  `0.899522 / 0.824061`; filtered is `0.909171 / 0.841056`.
- The paper-level result still requires processing every training video and
  aggregating them. Table 3 is not implemented yet.

## Key Changes

- Clone SegProp as a local third-party dependency:
  - `third_party/segprop -> https://github.com/vlicaret/segprop.git`
  - Keep our repo integration scripts separate under `scripts/`.
- Add a documented FlowNet2 setup path:
  - Use the existing `uav-flowprop` WSL GPU environment and the local checkout.
  - Add a smoke test gate: one adjacent frame pair must produce valid flow before large generation starts.
- Add optional external baseline integrations for Table 3:
  - Zhu et al. [6] propagation, isolated as an external dependency.
  - Homography voting, implemented around FlowNet2 correspondences and RANSAC.
- Add a paper-repro work root:
  - `outputs/segprop_paper_repro/`
  - All dense frames, labels, H5 flows, SegProp outputs, ablations, and tables stay ignored by git.

## Implementation Plan

1. **Shared Data Preparation** — *complete locally*
   - Read train videos from `data/Ruralscapes/ruralscapes_training_videos.txt`.
   - Export dense frames for each train video at 2K resolution, matching SegProp preprocessing:
     - original 4K frame -> downsample by 2 to roughly `2048x1080`
     - preserve original frame indices in filenames.
   - Convert manual PNG labels to SegProp `.npz` one-hot maps using the same palette/order as `ruralscapes_preprocess.py`.
   - Split labels by sorted manual-label ordinal:
     - even ordinal -> `labels_2k/train_even/<video>/`
     - odd ordinal -> `labels_2k/train_odd/<video>/`
     - all labels -> `labels_2k/all/<video>/`

2. **Shared FlowNet2 H5 Generation** — *implemented; `DJI_0101` complete*
   - Generate two H5 files per train video:
     - `flow_2k_fn2/<video>_forward.h5`
     - `flow_2k_fn2/<video>_backward.h5`
   - Each H5 contains a dataset named `flow`.
   - Shape must be `[num_frames - 1, height, width, 2]`.
   - Store vectors as `(x, y)` flow, matching SegProp README; SegProp flips internally.
   - Generate per video, not all videos at once, to avoid disk overflow.
   - Keep flow generation resumable with a sidecar progress file.
   - Current implementation: `scripts/generate_flownet2_h5.py` calls an external
     FlowNet2 command template per adjacent pair, validates each `.npy` flow,
     and writes forward/backward H5 files.

3. **Run SegProp Table 1 Pipeline** — *implemented; `DJI_0101` complete*
   - For each train video, run:
     - `segprop.vote(...)` from `train_even` to `output_2k/i01/<video>`
     - `segprop.iterate(...)` for iterations `i02` through `i07`
     - `segprop.denoise(...)` for the `+ Filt.` result
   - Use the paper/demo parameters:
     - `iterations = 7`
     - `pv_series=[0, 5, 10]` for iteration
     - final denoise `pv_series=[0, 1, 3, 5, 7]`
     - `frame_copy(index): index % 100 == 0`
     - `frame_filter(index): (index % 50 != 0) or (index % 100 == 0)`
   - Process one video at a time and aggregate metrics immediately.
   - Current implementation: `scripts/run_segprop_table1.py` imports the real
     `third_party/segprop` checkout, runs `vote -> iterate -> denoise`, and
     writes `table1_reproduction.{csv,md}` unless `--skip-eval` is used.
   - Intermediates are retained by default for debugging and can be pruned with
     `--prune-intermediate` after each video's filtered output is produced.

4. **Run Table 3 Ablation Pipeline** — *not implemented*
   - Produce the five Table 3 rows in fixed paper order:
     - `Zhu et al. [6]`: standalone one-iteration Zhu propagation baseline.
     - `SegProp`: one iteration with optical-flow votes only.
     - `SegProp + Zhu et al. [6]`: one iteration with Zhu votes added to SegProp votes.
     - `SegProp + Homography`: one iteration with optical-flow votes plus homography votes.
     - `SegProp + Homography + Filtering`: homography-vote result followed by SegProp's final filtering.
   - Implement homography voting as an extra vote module:
     - use already computed FlowNet2 maps as point correspondences
     - estimate homographies robustly with RANSAC
     - transform connected class regions from labelled frames into the target frame
     - convert transformed regions into additional class-vote maps
   - Treat Zhu et al. [6] as an external baseline dependency:
     - if official/runnable code is available, run it and save produced votes
     - if not available, keep its row as paper-only reference and mark reproduced value as missing
   - Run a short `DJI_0043` block before scaling to all training videos.

5. **Evaluation + Comparison Tables**
   - Evaluate Table 1 iterations and filtering against `labels_2k/train_odd`.
   - Compute Table 1 metrics:
     - mean F-measure over classes
     - mean IoU over classes
   - Compute Table 3 metric:
     - mean F-measure over classes only, matching the paper's Table 3.
   - Write:
     - `outputs/segprop_paper_repro/table1_reproduction.csv`
     - `outputs/segprop_paper_repro/table1_reproduction.md`
     - `outputs/segprop_paper_repro/table3_ablation_reproduction.csv`
     - `outputs/segprop_paper_repro/table3_ablation_reproduction.md`
   - Include expected paper rows next to reproduced rows and report absolute deltas.

## Test Plan

- Unit tests:
  - TrainEven/TrainOdd split from sorted labels.
  - PNG palette -> one-hot `.npz` conversion.
  - H5 flow shape and vector order validation using synthetic flow.
  - Metric aggregation matches known toy confusion-matrix values.
  - Homography vote module preserves class ids on a synthetic translated/planar mask.
  - RANSAC homography estimation rejects outlier flow correspondences.
  - Table 3 writer emits all five rows in fixed paper order.
- Smoke tests:
  - FlowNet2 produces a nonzero flow for one adjacent frame pair.
  - SegProp `vote` runs on a tiny 3-frame synthetic video.
  - Homography voting runs on a tiny synthetic planar sequence.
  - One real short block from `DJI_0043` runs through Table 1 and Table 3 paths before scaling.
- Acceptance checks:
  - Full train-video Table 1 run produces iterations `1..7` and `+ Filt.`.
  - Full train-video Table 3 run produces all five ablation rows.
  - Final Table 1 reproduced values are close to paper:
    - mF1 near `0.903`
    - mIoU near `0.829`
  - Final Table 3 reproduced values are close to paper:
    - Zhu et al. [6] near `0.846`
    - SegProp near `0.884`
    - SegProp + Zhu et al. [6] near `0.892`
    - SegProp + Homography near `0.894`
    - SegProp + Homography + Filtering near `0.904`
  - If values differ materially, report likely causes: FlowNet2 implementation/checkpoint mismatch, resolution mismatch, frame indexing mismatch, Zhu implementation mismatch, homography estimation details, or SegProp parameter mismatch.

## Assumptions

- Target is **Table 1 and Table 3**, not Table 2 semi-supervised neural-network training.
- Runtime target is **WSL after GPU access is fixed**.
- Resolution target is **2K paper-like**, not the faster 960x540 smoke baseline.
- Flow backend is **FlowNet2**, matching the paper.
- Because FlowNet2, SegProp, and Zhu et al. [6] are legacy, their runtime dependencies should be isolated from the existing `uav-flowprop` environment.
- Zhu et al. [6] may require external code; if it cannot be obtained or run, its reproduced Table 3 row should be explicitly marked unavailable rather than approximated silently.
- Large generated data should stay in `outputs/` and remain untracked.
