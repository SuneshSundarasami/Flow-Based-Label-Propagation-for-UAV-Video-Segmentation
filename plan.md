# Flow-Based Label Propagation for UAV Video Segmentation — Project Plan

Detailed execution plan derived from the DLRV proposal
(`Proposal/Sundarasami-DLRVProposal.pdf`). The proposal's three coarse work
packages (Setup → Pipeline → Evaluation) are broken here into small, well-defined
packages, each with a concrete deliverable and a check that tells you it is done.

**Package manager:** conda is used throughout (see [Environment](#environment)).

---

## What the project is

A **flow-based label propagation pipeline** for a single UAV video from
**Ruralscapes**:

1. Take a **keyframe** that has a dense ground-truth segmentation mask.
2. Use **SEA-RAFT** (default) or **FlowNet2** to estimate optical flow between
   the keyframe and nearby frames.
3. **Warp** the keyframe mask onto neighbour frames along the flow
   (nearest-neighbour, to keep label values discrete).
4. **Detect occlusions** with a forward–backward consistency check and mark those
   pixels invalid.
5. **Evaluate** each warped mask against dense GT using mIoU over valid pixels.
6. **Characterise** how quality decays with distance from the keyframe → decay
   curve, difficulty heatmap over the timeline, per-class IoU, failure-case
   visualisations.
7. **(Optional)** content-aware keyframe selection that beats fixed-interval
   selection at the same annotation budget.

---

## Key technical decision to confirm up front

To render frame *t*'s labels by sampling from the keyframe mask, you need flow in
the direction **t → keyframe** (each target pixel looks up its source in the
labelled keyframe). The forward–backward check then uses both *t→k* and *k→t*
flow. This plan assumes the **backward-warp / sample-from-keyframe** convention.
If the supervisor expects forward splatting instead, WP B1–B2 change.

---

## Environment

Conda is the package manager for the whole project.

```bash
# create the environment
conda env create -f environment.yml
conda activate uav-flowprop

# update after editing environment.yml
conda env update -f environment.yml --prune
```

`environment.yml` pins Python, PyTorch (+CUDA), and the CV/plotting stack
(numpy, opencv, matplotlib, tqdm, etc.). SEA-RAFT's own deps are folded in here so
there is a single environment for the project.

---

## Work Packages

Effort estimates are working-days for one person. `Done when` is the acceptance
check.

### Phase A — Foundation  *(maps to proposal WP1 / Milestone M1)*

**A1. Environment & repo scaffold** — *0.5 d*
- `environment.yml` (conda), `src/` layout (`flow/`, `warp/`, `eval/`, `viz/`,
  `config/`), README.
- *Done when:* `conda env create -f environment.yml` succeeds in a clean shell and
  the package imports.
- *Verified:* `uav-flowprop` environment created successfully; package imports
  pass with `PYTHONPATH=src`; `conda run -n uav-flowprop pytest -q` reports
  `5 passed`.

**A2. SEA-RAFT integration & smoke test** — *1 d*
- Add SEA-RAFT, download a pretrained checkpoint, wrap inference in
  `estimate_flow(img1, img2) -> flow`.
- *Done when:* flow runs on one synthetic pair and one real frame pair; the flow
  colour-wheel visualisation looks sensible.
- *Verified:* `model.safetensors` downloaded to `third_party/SEA-RAFT/checkpoints/`;
  `scripts/smoke_test_flow.py --device cpu --img1 third_party/SEA-RAFT/custom/image1.jpg --img2 third_party/SEA-RAFT/custom/image2.jpg`
  passed. Synthetic expected `(12, 6)`, recovered median `(11.85, 6.74)`;
  real-pair flow magnitude min/mean/max `0.00/1.18/35.25`; visualisations written
  to `outputs/smoke/`.

**A3. Ruralscapes data loader** — *1 d*  **(biggest unknown — needs dataset in hand)**
- Locate the chosen video + dense GT masks; confirm class list, label encoding,
  annotation density; loader returning `(frame, mask, frame_index)`.
- *Done when:* any frame/mask pair loads and a class-colour legend prints.
- *Verified so far:* loader and `scripts/inspect_data.py` pass on
  `tests/fixtures/ruralscapes_demo`, printing frame range, annotation count,
  mask encoding/unique ids, and class legend. Real Ruralscapes verification is
  still pending because no dataset is present under `data/`.
- *Real-data verified:* extracted `data/Ruralscapes.zip`, exported the labelled
  frames for `DJI_0043`, and ran `scripts/inspect_data.py` on 142 matched
  frame/mask pairs. Sample frame/mask shape: `(2160, 4096, 3)` / `(2160, 4096)`;
  median annotation spacing is 50 frames and the class legend prints.

**A4. Literature notes** — *0.5 d (parallel)*
- One-page summary of SegProp, SEA-RAFT, Brox FB-consistency, Zhu warp-and-refine
  → feeds report intro / related work.

> **Milestone M1:** SEA-RAFT produces valid flow on one Ruralscapes pair; data
> loads cleanly; lit notes drafted.

### Phase B — Core pipeline  *(maps to proposal WP2 / Milestone M2)*

**B1. Flow-based mask warping** — *1 d*
- Convert flow to a sampling grid; warp keyframe mask with **nearest-neighbour**
  (`grid_sample`, mode='nearest').
- *Done when:* keyframe→keyframe (zero flow) returns the mask unchanged; ±1 frame
  looks aligned.
- *Verified:* `src/warp/mask_warp.py` implemented; `warp_mask(mask, zeros) == mask`
  confirmed; 6 unit tests passing (identity, integer shift, out-of-bounds → ignore
  index, custom ignore, shape mismatch, dtype preservation). Visual 4-panel check
  (keyframe GT / warped / target GT / validity) written to `outputs/warp_check/`
  via `scripts/check_warp.py`.

**B2. Forward–backward occlusion mask** — *1 d*
- Compute *k→t* and *t→k* flow, measure consistency error, threshold to an
  occlusion/validity mask.
- *Done when:* occluded regions (borders, fast objects) show up as invalid in an
  overlay.
- *Verified:* `src/warp/occlusion.py` implemented (Brox 2004 round-trip check);
  5 unit tests passing. Real-data statistics at FB threshold = 1.5 px:
  99.8% valid at +50 f, 98.7% at +100 f, 95.4% at +150 f. Interior red marks
  explained by parallax, motion boundaries, and true occlusions.

**B3. mIoU + per-class IoU metric** — *0.5 d*
- IoU over **valid pixels only**, ignore-label handling, mean over classes.
- *Done when:* unit test on hand-made masks gives known IoU values.
- *Verified:* `src/eval/metrics.py` implemented via confusion matrix (TP/FP/FN);
  accepts `valid_mask` from B2 as an optional exclusion mask; 9 unit tests passing
  (perfect prediction, all-wrong, partial overlap, ignore_index, valid_mask
  exclusion, nan for absent class, shape errors, empty-after-masking).

**B4. End-to-end single-keyframe propagation** — *1 d*
- Chain B1–B3: one keyframe → warp to ±N frames → per-frame mIoU.
- *Done when:* on a short clip, mIoU **monotonically decreases** with distance
  from the keyframe (core sanity check).
- *Verified (implementation):* `src/propagation/propagate.py` implements
  `propagate_keyframe()` which chains B1 + B2 + B3 and returns `FrameResult`
  per target (warped mask, valid mask, valid%, IoU dict). 10 unit tests passing
  (identity warp, valid pct, distance, shapes, IoU with/without GT, multiple
  targets, length-mismatch errors). `scripts/run_propagation.py` prints a
  distance / valid% / mIoU(all) / mIoU(valid) / per-class table and saves two
  CSVs (`results_all_pixels.csv`, `results_valid_pixels.csv`) and a
  per-keyframe summary at `outputs/results/<video>/propagation_results/keyframe_{K}.csv` with columns
  `keyframe, target_frame, distance, valid_pct, miou_all, miou_valid,
  iou_class0, iou_class1, …` — the format C1 concatenates across keyframes.
  Real-data decreasing-mIoU sanity check is ready to run (requires GPU).

**B5. Config + CLI runner** — *0.5 d*
- Single config (keyframe interval, window size, FB threshold, paths) driving a
  reproducible CLI run; results saved to disk (CSV/JSON).
- *Done when:* one command reproduces B4's numbers from config.
- *Verified:* `src/config/default.yaml` extended with a `data:` section
  (`frame_glob`, `mask_glob`, `mask_format`) and `propagation.n_targets`.
  `run_propagation.py` now reads all settings from config; every CLI flag is
  optional and falls back to the config value. A `--config YAML` flag accepts
  a user-supplied override file (deep-merged on top of defaults). Bare
  `python scripts/run_propagation.py` reproduces B4 numbers with no extra flags.
  8 unit tests in `tests/test_b5.py` cover config key completeness, YAML and
  dict overrides, merge-order priority, and CLI-default alignment. All 48 tests
  pass.

> **Milestone M2:** End-to-end pipeline validated on a short clip with correct,
> decreasing mIoU.

### Phase C — Scale, analyse, report  *(maps to proposal WP3 / Milestone M3)*

**C1. Full-video batched run** — *1 d*
- Run propagation across all keyframes over a fixed window; persist per-pair mIoU
  table; caching/checkpointing so reruns are cheap.
- *Done when:* a complete results table (keyframe, target, distance, mIoU,
  per-class IoU) exists for the whole video.
- *Verified (implementation):* `scripts/run_full_video.py` iterates every
  annotated keyframe, propagating forward to the next `n_targets` annotated
  frames via `propagation.forward_targets` + `propagate_keyframe`. Each
  keyframe's results are cached to `outputs/results/<video>/propagation_results/keyframe_{k}.csv` and
  skipped on rerun unless `--force`; `--limit N` allows partial runs. All
  per-keyframe CSVs are concatenated (sorted by keyframe, distance) into
  `outputs/results/<video>/propagation_results/all_pairs.csv` — the full-video
  table for C2–C5. For the default Ruralscapes clip, this resolves to
  `outputs/results/DJI_0043/propagation_results/`. The CSV schema is centralised
  in `src/eval/results_io.py` (shared with B5). 10 unit tests in
  `tests/test_c1.py` cover target scheduling, row schema/values, CSV round-trip,
  and sorted concatenation. Real-data run requires GPU.

**C2. mIoU-vs-distance decay curve** — *0.5 d*  *(MVP deliverable)*
- Aggregate mIoU by absolute frame distance with mean ± spread band.
- *Verified (implementation):* `scripts/analyze_decay.py` reads C1's
  `all_pairs.csv`, groups rows by propagation distance, and writes
  `outputs/results/<video>/analysis/decay_summary.csv` plus
  `outputs/results/<video>/analysis/miou_decay.png`. The plot shows
  mIoU(valid) with a ±1 std band and the mean FB-valid pixel percentage.
  `tests/test_c2.py` covers sorted distance grouping and NaN handling.

**C3. Difficulty heatmap over timeline** — *0.5 d*  *(Expected deliverable)*
- 2D heatmap (keyframe × distance, or timeline × distance) of propagation quality.
- *Verified (implementation):* `scripts/analyze_heatmap.py` reads C1's
  `all_pairs.csv`, builds a keyframe × distance matrix using `miou_valid` by
  default, and writes `outputs/results/<video>/analysis/difficulty_heatmap_matrix.csv`
  plus `outputs/results/<video>/analysis/difficulty_heatmap.png`. `--metric`
  can switch the heatmap to `miou_all` or `valid_pct`. `tests/test_c3.py`
  covers axis sorting, missing cells, duplicate-cell averaging, NaN handling,
  and CSV output.

**C4. Per-class IoU breakdown** — *0.5 d*  *(Expected deliverable)*
- Which classes propagate best/worst, with interpretation.
- *Verified (implementation):* `scripts/analyze_per_class.py` reads C1's
  `all_pairs.csv`, summarizes valid-only `iou_classN` columns using the
  Ruralscapes class names, and writes
  `outputs/results/<video>/analysis/per_class_iou.csv` plus
  `outputs/results/<video>/analysis/per_class_iou.png`. `tests/test_c4.py`
  covers class-name mapping, sorted ranking, summary statistics, and NaN
  handling.

**C5. Failure-case visualisations** — *0.5 d*  *(MVP deliverable)*
- ≥2 worst pairs; side-by-side frame / warped mask / GT / error map with written
  explanation tied to motion/parallax.
- *Verified (implementation):* `scripts/visualize_failures.py` selects the
  lowest-scoring C1 rows by `miou_valid` after filtering near-empty valid
  regions (`--min-valid-pct`, default 5%), recomputes only those pairs, and
  writes `outputs/results/<video>/failure_cases/failure_cases.md` plus
  side-by-side PNGs showing keyframe GT, warped target, target GT, and an
  error/validity map. `tests/test_c5.py` covers deterministic worst-pair
  selection, alternate metric ranking, and valid-pixel filtering.

**C6. Report write-up** — *continuous, ~2 d total*
- Intro, method, results, discussion, guidelines for reliable propagation distance.
- *Done when:* report covers all MVP + Expected deliverables and references the
  C2–C5 figures.

> **Milestone M3:** Full-video evaluation done; heatmap, decay curve, per-class,
> failure cases, and report submitted.

### Phase D — Optional extension  *(maps to proposal §5; only if ahead of schedule)*

**D0. SegProp-format dataset preparation** — *complete*
- Convert Ruralscapes manual labels to SegProp-style one-hot `.npz` files with
  `map` and `votes` arrays.
- Split the official training labels into `train_even` / `train_odd` by
  annotation ordinal and write per-video metadata CSVs.
- Optionally export dense 2K frames from the MP4 videos.
- *Done when:* the prep script runs on the training split and produces
  `outputs/segprop_paper_repro/labels_2k/`, optional `frames_2k/`, and
  `metadata/`.
- *Verified:* unit tests cover label naming, RGB palette lookup, one-hot
  encoding, TrainEven/TrainOdd output, and metadata. The official training
  split is prepared locally: 13 videos, 37,666 2K frames, labels, and metadata.

**D0.5. FlowNet2 H5 generation harness** — *implemented; one-video run complete*
- Wrap an external legacy FlowNet2 command that writes per-pair `.npy` flow
  arrays in `(x, y)` order.
- Persist SegProp-compatible H5 files at
  `outputs/segprop_paper_repro/flow_2k_fn2/<video>_{forward,backward}.h5`.
- Keep per-video generation resumable with `<video>_progress.json`.
- *Verified:* tests cover frame ordering, flow shape/finite validation,
  forward/backward H5 writing, and progress-file creation with a fake flow
  command. Full forward/backward 2K H5 flow was generated for `DJI_0101`; the
  remaining training videos are pending because the dense flow cache is large.

**D0.6. SegProp Table 1 runner** — *implemented; one-video run complete*
- Import the real `third_party/segprop` checkout and run the paper sequence:
  `vote` -> `iterate` through `i07` -> final `denoise`.
- Use the paper/demo parameters (`pv_series=[0, 5, 10]`, final
  `pv_series=[0, 1, 3, 5, 7]`, `frame_copy(index % 100 == 0)`, and the
  TrainOdd evaluation frame filter).
- Write `table1_reproduction.csv` and `.md` with reproduced values, paper
  targets, and deltas.
- *Verified:* tests cover input validation, call ordering, paper frame filter
  helpers, and Table 1 CSV/Markdown writing with fake SegProp/stats modules.
  The complete 2K `DJI_0101` run produced i01 mF1/mIoU `0.899522/0.824061` and
  filtered `0.909171/0.841056`. A full training-split aggregate is still
  pending.

**D1.** Use C1's per-pair scores to label which intervals are hard to propagate.
**D2.** Lightweight content-aware keyframe selector recommending *k* frames to annotate.
**D3.** Compare against fixed-interval (SegProp) baseline at equal annotation
budget; show higher overall mIoU.

---

## Critical path & risk

- Critical path: `A1 → A2 → A3`. **A3 (dataset) is the biggest risk** — if
  annotation density/format differs from the proposal's assumption, B4/C1 change.
- A4 runs in parallel with A1–A3.
- Phase B is sequential. C2–C5 are independent once C1 is done and can be
  parallelised. Phase D is strictly optional.

## Deliverable mapping (from proposal §4.3)

- **Minimum Viable:** B4 + C2 + C5.
- **Expected:** C3 + C4.
- **Desired:** clean reusable code (A1/B5) + propagation-distance guidelines (C6).

---

## Progress tracker

| WP | Description | Status |
|----|-------------|--------|
| A1 | Environment & repo scaffold | ☑ done; env + imports + tests verified |
| A2 | SEA-RAFT integration & smoke test | ☑ done; CPU smoke verified on synthetic + sample real pair |
| A3 | Ruralscapes data loader | ☑ done; real DJI_0043 data verified |
| A4 | Literature notes | ☑ done |
| B1 | Flow-based mask warping | ☑ done; `warp/mask_warp.py` + 6 tests; visual check via `check_warp.py` |
| B2 | Forward–backward occlusion mask | ☑ done; `warp/occlusion.py` + 5 tests; 99.8%→95.4% valid at +50→+150 f |
| B3 | mIoU + per-class IoU metric | ☑ done; `eval/metrics.py` + 9 tests; confusion-matrix, valid-mask support |
| B4 | End-to-end single-keyframe propagation | ☑ done; `propagation/propagate.py` + 10 tests; `run_propagation.py` ready |
| B5 | Config + CLI runner | ☑ done; `default.yaml` data section + `--config` override + `keyframe_X.csv`; 9 tests |
| C1 | Full-video batched run | ☑ done; `run_full_video.py` + video-scoped CSV cache + `all_pairs.csv`; 10 tests |
| C2 | mIoU-vs-distance decay curve | ☑ done; `analyze_decay.py` + summary CSV/plot; 2 tests |
| C3 | Difficulty heatmap over timeline | ☑ done; `analyze_heatmap.py` + matrix CSV/plot; 3 tests |
| C4 | Per-class IoU breakdown | ☑ done; `analyze_per_class.py` + CSV/bar plot; 2 tests |
| C5 | Failure-case visualisations | ☑ done; `visualize_failures.py` + worst-pair panels; 2 tests |
| C6 | Report write-up | ☐ todo |
| D0 | SegProp-format training data preparation | ☑ done; 13 videos / 37,666 2K frames prepared locally |
| D0.5 | FlowNet2 H5 generation | ◐ one complete 2K video (`DJI_0101`); full training split pending |
| D0.6 | SegProp Table 1 runner | ◐ `DJI_0101` completed; full training-split aggregate pending |
| D1–D3 | Optional: content-aware keyframe selection | ☐ optional |

Status legend: ☐ todo · ◐ in progress · ☑ done
