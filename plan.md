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
2. Use **SEA-RAFT** to estimate optical flow between the keyframe and nearby frames.
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
  distance / valid% / mIoU / per-class table and saves `outputs/propagation/results.csv`.
  Real-data decreasing-mIoU sanity check is ready to run (requires GPU).

**B5. Config + CLI runner** — *0.5 d*
- Single config (keyframe interval, window size, FB threshold, paths) driving a
  reproducible CLI run; results saved to disk (CSV/JSON).
- *Done when:* one command reproduces B4's numbers from config.

> **Milestone M2:** End-to-end pipeline validated on a short clip with correct,
> decreasing mIoU.

### Phase C — Scale, analyse, report  *(maps to proposal WP3 / Milestone M3)*

**C1. Full-video batched run** — *1 d*
- Run propagation across all keyframes over a fixed window; persist per-pair mIoU
  table; caching/checkpointing so reruns are cheap.
- *Done when:* a complete results table (keyframe, target, distance, mIoU,
  per-class IoU) exists for the whole video.

**C2. mIoU-vs-distance decay curve** — *0.5 d*  *(MVP deliverable)*
- Aggregate mIoU by absolute frame distance with mean ± spread band.

**C3. Difficulty heatmap over timeline** — *0.5 d*  *(Expected deliverable)*
- 2D heatmap (keyframe × distance, or timeline × distance) of propagation quality.

**C4. Per-class IoU breakdown** — *0.5 d*  *(Expected deliverable)*
- Which classes propagate best/worst, with interpretation.

**C5. Failure-case visualisations** — *0.5 d*  *(MVP deliverable)*
- ≥2 worst pairs; side-by-side frame / warped mask / GT / error map with written
  explanation tied to motion/parallax.

**C6. Report write-up** — *continuous, ~2 d total*
- Intro, method, results, discussion, guidelines for reliable propagation distance.
- *Done when:* report covers all MVP + Expected deliverables and references the
  C2–C5 figures.

> **Milestone M3:** Full-video evaluation done; heatmap, decay curve, per-class,
> failure cases, and report submitted.

### Phase D — Optional extension  *(maps to proposal §5; only if ahead of schedule)*

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
| B5 | Config + CLI runner | ☐ todo |
| C1 | Full-video batched run | ☐ todo |
| C2 | mIoU-vs-distance decay curve | ☐ todo |
| C3 | Difficulty heatmap over timeline | ☐ todo |
| C4 | Per-class IoU breakdown | ☐ todo |
| C5 | Failure-case visualisations | ☐ todo |
| C6 | Report write-up | ☐ todo |
| D1–D3 | Optional: content-aware keyframe selection | ☐ optional |

Status legend: ☐ todo · ◐ in progress · ☑ done
