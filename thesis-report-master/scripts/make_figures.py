"""Regenerate every data figure and derived number used in the report.

Reads only committed result artifacts -- nothing here needs a GPU, torch, or a
re-run of any experiment:

    outputs/adaptive_inference_metrics/uavid_val/*.csv   adaptive vs baseline, per frame
    outputs/backend_predictions/*/*/metrics.csv          SEA-RAFT vs FlowNet2 propagation
    image_segmentation/runs/*/log.txt                    per-epoch val mIoU / per-class IoU
    Presentation/*.pptx                                  the qualitative panels shown in the defence

Writes vector PDFs (and one composited PNG) into ``images/`` plus
``generated_numbers.tex``: a set of LaTeX macros holding every number quoted in
the prose, so the text can never drift from the data.

    python scripts/make_figures.py
"""
from __future__ import annotations

import ast
import csv
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
REPO = REPORT.parent
IMAGES = REPORT / "images"
IMAGES.mkdir(exist_ok=True)

VAL_CSV_DIR = REPO / "outputs/adaptive_inference_metrics/uavid_val"
BACKEND_DIR = REPO / "outputs/backend_predictions"
RUNS = REPO / "image_segmentation/runs"

# ---------------------------------------------------------------- styling ---
TEXTWIDTH_IN = 6.3          # 16 cm text block of the MAS template
C_MODEL = "#1B4F72"
C_FLOW = "#B9770E"
C_ACCENT = "#7B241C"
C_MUTED = "#5D6D7E"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

NUMBERS: dict[str, str] = {}


def emit(name: str, value, fmt: str = "{}") -> None:
    """Record a LaTeX macro \\<name>{} holding *value*.

    A TeX control sequence may only contain letters -- a digit silently
    truncates the name and produces a cascade of unrelated errors, so reject
    it here where the cause is obvious.
    """
    if not name.isalpha():
        raise ValueError(
            f"macro name {name!r} must be letters only: TeX control sequences "
            "cannot contain digits or punctuation")
    NUMBERS[name] = fmt.format(value)


# =====================================================================
# 1. Load the adaptive-inference validation results
# =====================================================================
def load_val_rows() -> list[dict]:
    """All 7 val sequences, annotated with distance-since-last-model-call."""
    rows: list[dict] = []
    for path in sorted(VAL_CSV_DIR.glob("*.csv")):
        seq = path.stem
        seq_rows = list(csv.DictReader(path.open()))
        dist = 0
        for r in seq_rows:
            if r["hybrid_source"] == "model":
                dist = 0
            else:
                dist += 1
            r["_seq"] = seq
            r["_dist"] = dist
            r["_frame"] = int(r["frame"])
            r["_base_t"] = float(r["baseline_elapsed_s"])
            r["_hyb_t"] = float(r["hybrid_elapsed_s"])
            r["_cov"] = float(r["hybrid_valid_pct"])
            r["_base_m"] = float(r["baseline_miou"]) if r["baseline_miou"] else None
            r["_hyb_m"] = float(r["hybrid_miou"]) if r["hybrid_miou"] else None
        rows += seq_rows
    if not rows:
        raise SystemExit(f"no val CSVs under {VAL_CSV_DIR}")
    return rows


def per_sequence_table(rows: list[dict]) -> list[dict]:
    """One record per sequence: calls, both timing conventions, mIoU."""
    out = []
    for seq in sorted({r["_seq"] for r in rows}):
        rs = [r for r in rows if r["_seq"] == seq]
        flow = [r for r in rs if r["hybrid_source"] == "flow"]
        model = [r for r in rs if r["hybrid_source"] == "model"]
        base_t = sum(r["_base_t"] for r in rs)
        hyb_t = sum(r["_hyb_t"] for r in rs)
        gt = [r for r in rs if r["_base_m"] is not None]
        out.append({
            "seq": seq,
            "frames": len(rs),
            "calls": len(model),
            "pct_model": 100.0 * len(model) / len(rs),
            "base_t": base_t,
            "hyb_t": hyb_t,
            "model_t": sum(r["_hyb_t"] for r in model),
            "flow_t": sum(r["_hyb_t"] for r in flow),
            "speedup": base_t / hyb_t,
            "base_m": sum(r["_base_m"] for r in gt) / len(gt),
            "hyb_m": sum(r["_hyb_m"] for r in gt) / len(gt),
        })
    return out


# =====================================================================
# 4. Per-epoch training logs
# =====================================================================
LINE_RE = re.compile(
    r"epoch\s+(\d+) \| train ([\d.]+) \| val ([\d.]+) \| mIoU ([\d.]+) \| (\{.*\})")


def parse_runs() -> dict[str, dict]:
    """Best-epoch validation mIoU per trained backbone, from its log."""
    best: dict[str, dict] = {}
    for log in sorted(RUNS.glob("*/log.txt")):
        run = log.parent.name
        key = re.sub(r"_\d{4}_\d{4}$", "", run)
        top = None
        last = None
        for line in log.open():
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            rec = {"epoch": int(m.group(1)), "train": float(m.group(2)),
                   "val": float(m.group(3)), "miou": float(m.group(4)),
                   "per_class": ast.literal_eval(m.group(5))}
            last = rec
            if top is None or rec["miou"] > top["miou"]:
                top = rec
        if top:
            top["epochs_run"] = last["epoch"]
            top["run"] = run
            best[key] = top
    return best


# =====================================================================
# 5. Figures taken from the project presentation
# =====================================================================
DECK_PPTX = REPO / "Presentation/AdaptiveFlowSegmentation-Presentation.pptx"

# Each entry: media part inside the .pptx, output stem, target pixel width,
# which GIF frame carries the flow vectors (None for stills), and how many
# pixels to trim from the top. The flow overlay carries a slide step-marker
# badge in its top-left corner, measured at 68 px tall, which is presentation
# furniture rather than content; the two still panels carry in-image captions
# that the report's own captions refer to, so they are kept whole.
DECK_FIGURES = [
    ("ppt/media/image5.png", "fig_deck_segmentation", 1900, None, 0),
    ("ppt/media/image7.png", "fig_deck_hybrid", 1900, None, 0),
    ("ppt/media/image6.gif", "fig_deck_flowvectors", 1500, 1, 70),
]


def deck_figures() -> None:
    """Extract the presentation's own rendered panels into ``images/``.

    These are the qualitative figures shown in the defence: they were produced
    by the pipeline on the A100 and are not reproducible here, so
    they are lifted from the deck rather than re-rendered. Downsampled to
    roughly 300 dpi at the template's text width.
    """
    import io
    import zipfile

    if not DECK_PPTX.exists():
        print(f"  ! {DECK_PPTX.name} not found, deck figures skipped")
        return
    with zipfile.ZipFile(DECK_PPTX) as z:
        for part, stem, width, gif_frame, croptop in DECK_FIGURES:
            im = Image.open(io.BytesIO(z.read(part)))
            if gif_frame is not None:
                im.seek(gif_frame)
            im = im.convert("RGB")
            if croptop:
                im = im.crop((0, croptop, im.width, im.height))
            h = round(width * im.height / im.width)
            im = im.resize((width, h), Image.LANCZOS)
            # JPEG, not PNG: these are photographic overlays, and lossless
            # coding of the dense flow-vector panel alone costs 2.5 MB.
            out = IMAGES / f"{stem}.jpg"
            im.save(out, quality=88, optimize=True, progressive=True)
            print(f"  images/{out.name}  ({width}x{h}, "
                  f"{out.stat().st_size // 1024} KB)")


# =====================================================================
# 6. Qualitative panel composited from the cached propagation artifacts
# =====================================================================
RURAL_PALETTE = np.array([
    [0, 255, 0], [0, 127, 0], [255, 255, 0], [255, 127, 0],
    [255, 255, 255], [255, 0, 255], [127, 127, 127], [0, 0, 255],
    [0, 255, 255], [127, 127, 63], [255, 0, 0], [127, 127, 0],
], dtype=np.uint8)


def colorize(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(labels.shape + (3,), np.uint8)
    valid = (labels >= 0) & (labels < len(RURAL_PALETTE))
    out[valid] = RURAL_PALETTE[labels[valid]]
    return out


def _font(size: int):
    from PIL import ImageFont
    for cand in ("arial.ttf", "DejaVuSans.ttf", "calibri.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label(img: Image.Image, text: str, size: int = 30) -> Image.Image:
    """Draw a caption strip under a panel."""
    from PIL import ImageDraw
    strip = int(size * 1.55)
    canvas = Image.new("RGB", (img.width, img.height + strip), "white")
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((2, img.height + int(size * 0.22)), text, fill=(15, 15, 15),
           font=_font(size))
    return canvas


def fig_qualitative(video: str = "DJI_0101", target: int = 250) -> bool:
    """Six panels: the labelled keyframe, the target frame, both backends'
    propagated masks, the dense GT, and the pixels FB consistency rejects.

    NOTE: this is the *direct* single-jump propagation regime used for the
    backend comparison (one flow estimate spanning `target - keyframe`
    frames), not the adaptive pipeline's frame-by-frame chaining.
    """
    sea = BACKEND_DIR / video / "sea_raft" / f"{target:06d}.npz"
    fn2 = BACKEND_DIR / video / "flownet2" / f"{target:06d}.npz"
    tgt_f = FRAMES_2K / video / f"{video}_{target:06d}.jpg"
    gt = RURAL_GT / video / f"segfull_{video}_{target:06d}.png"
    missing = [str(p) for p in (sea, fn2, tgt_f, gt) if not p.exists()]
    if missing:
        print("  SKIP fig_qualitative_propagation.png -- missing:")
        for m in missing:
            print(f"      {m}")
        return False

    d_sea, d_fn2 = np.load(sea), np.load(fn2)
    lab_sea, valid_sea = d_sea["label"], d_sea["valid"]
    lab_fn2 = d_fn2["label"]
    kf = int(d_sea["keyframe"])
    h, w = lab_sea.shape

    tgt = Image.open(tgt_f).convert("RGB").resize((w, h), Image.LANCZOS)
    gt_img = Image.open(gt).convert("RGB").resize((w, h), Image.NEAREST)

    kf_f = FRAMES_2K / video / f"{video}_{kf:06d}.jpg"
    kf_gt = RURAL_GT / video / f"segfull_{video}_{kf:06d}.png"
    if kf_f.exists():
        kf_img = Image.open(kf_f).convert("RGB").resize((w, h), Image.LANCZOS)
    elif kf_gt.exists():
        kf_img = Image.open(kf_gt).convert("RGB").resize((w, h), Image.NEAREST)
    else:
        kf_img = Image.new("RGB", (w, h), "white")

    # invalid-pixel overlay: dim the target frame, paint FB-rejected red
    over = (np.asarray(tgt).astype(np.float32) * 0.40).astype(np.uint8)
    over[~valid_sea] = (210, 25, 25)
    invalid_pct = 100 * float((~valid_sea).mean())

    panels = [
        (kf_img, f"(a) labelled keyframe {kf}"),
        (tgt, f"(b) target frame {target}  (+{target - kf} frames)"),
        (gt_img, "(c) dense ground truth at target"),
        (Image.fromarray(colorize(lab_sea)), "(d) propagated, SEA-RAFT"),
        (Image.fromarray(colorize(lab_fn2)), "(e) propagated, FlowNet2"),
        (Image.fromarray(over),
         f"(f) pixels rejected by the FB check ({invalid_pct:.1f}%)"),
    ]

    pw = 820
    scaled = []
    for img, cap in panels:
        ph = int(round(img.height * pw / img.width))
        scaled.append(_label(img.resize((pw, ph), Image.LANCZOS), cap))

    cols, gap = 3, 12
    ph = scaled[0].height
    rows_n = (len(scaled) + cols - 1) // cols
    sheet = Image.new("RGB",
                      (cols * pw + (cols - 1) * gap,
                       rows_n * ph + (rows_n - 1) * gap), "white")
    for i, img in enumerate(scaled):
        r, c = divmod(i, cols)
        sheet.paste(img, (c * (pw + gap), r * (ph + gap)))
    out = IMAGES / "fig_qualitative_propagation.png"
    sheet.save(out, optimize=True)

    emit("QualTarget", target)
    emit("QualKeyframe", kf)
    emit("QualDistance", target - kf)
    emit("QualInvalidPct", f"{invalid_pct:.1f}")
    print(f"  images/fig_qualitative_propagation.png  "
          f"({sheet.width}x{sheet.height}, {out.stat().st_size // 1024} KB, "
          f"kf {kf} -> {target})")
    return True


# =====================================================================
# 7. Derived numbers for the prose
# =====================================================================
def emit_all_numbers(rows: list[dict], per_seq: list[dict],
                     best: dict[str, dict]) -> None:
    gt = [r for r in rows if r["_base_m"] is not None]
    mod = [r for r in gt if r["hybrid_source"] == "model"]

    def mean(rs, k):
        return sum(r[k] for r in rs) / len(rs)

    base_t = sum(p["base_t"] for p in per_seq)
    hyb_t = sum(p["hyb_t"] for p in per_seq)
    calls = sum(p["calls"] for p in per_seq)
    frames = sum(p["frames"] for p in per_seq)

    emit("NumSeq", len(per_seq))
    emit("NumFrames", f"{frames:,}")
    emit("NumGT", len(gt))
    emit("NumGTModel", len(mod))
    emit("TotalCalls", f"{calls:,}")
    emit("MeanCalls", f"{calls / len(per_seq):.0f}")
    emit("PctModel", f"{100 * calls / frames:.1f}")

    emit("BaseTime", f"{base_t:.0f}")
    emit("HybTime", f"{hyb_t:.0f}")
    emit("Speedup", f"{base_t / hyb_t:.2f}")

    emit("BaseMIoU", f"{mean(gt, '_base_m'):.4f}")
    emit("HybMIoU", f"{mean(gt, '_hyb_m'):.4f}")
    emit("Retention", f"{100 * mean(gt, '_hyb_m') / mean(gt, '_base_m'):.1f}")

    # Single-frame inference latencies as reported in the project presentation.
    # The benchmark run wrote no persistent log, so these are recorded here as
    # constants rather than re-derived, and are footnoted as such in the report.
    emit("LatConvnext", 365)
    emit("LatSwin", 433)
    emit("LatHieraBasePlus", 327)
    emit("LatTypical", 370)
    emit("ClipMinutes", f"{901 * 0.370 / 60:.1f}")
    # How far one frame overruns the 50 ms that 20 FPS allows.
    emit("RealtimeFactor", f"{0.370 / (1 / 20):.1f}")

    for key, rec in best.items():
        tag = "".join(part.capitalize() for part in key.split("_"))
        emit(f"MIoU{tag}", f"{rec['miou']:.4f}")
        emit(f"Epoch{tag}", rec["epoch"])
        emit(f"EpochsRun{tag}", rec["epochs_run"])
        emit(f"ValLoss{tag}", f"{rec['val']:.4f}")

    # flow-backend comparison, pooled over the paired Ruralscapes runs
    for backend in ("sea_raft", "flownet2"):
        allrows = []
        for mfile in sorted(BACKEND_DIR.glob(f"*/{backend}/metrics.csv")):
            allrows += list(csv.DictReader(mfile.open()))
        if not allrows:
            continue
        tag = "SeaRaft" if backend == "sea_raft" else "FlowNetTwo"
        n = len(allrows)
        emit(f"{tag}Pairs", n)
        for col, name in (("miou_all", "MIoUAll"), ("miou_valid", "MIoUValid"),
                          ("valid_pct", "Coverage")):
            v = sum(float(r[col]) for r in allrows) / n
            emit(f"{tag}{name}", f"{v:.4f}" if "MIoU" in name else f"{v:.1f}")


def write_numbers() -> None:
    out = REPORT / "generated_numbers.tex"
    lines = [
        "% Auto-generated by scripts/make_figures.py -- DO NOT EDIT BY HAND.",
        "% Every number quoted in the report body is defined here so the prose",
        "% cannot drift from the underlying result files.",
        "",
    ]
    for k in sorted(NUMBERS):
        lines.append(rf"\newcommand{{\{k}}}{{{NUMBERS[k]}}}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  generated_numbers.tex   ({len(NUMBERS)} macros)")


def main() -> int:
    print("reading result artifacts ...")
    rows = load_val_rows()
    per_seq = per_sequence_table(rows)
    best = parse_runs()
    print(f"  {len(rows)} frames over {len(per_seq)} sequences, "
          f"{len(best)} trained runs")

    print("writing figures ...")
    deck_figures()

    emit_all_numbers(rows, per_seq, best)
    write_numbers()

    print("\nper-sequence summary:")
    print(f"{'seq':7s}{'calls':>6s}{'%model':>8s}{'base s':>9s}"
          f"{'hyb s':>8s}{'speedup':>9s}{'base m':>9s}{'hyb m':>8s}")
    for p in per_seq:
        print(f"{p['seq']:7s}{p['calls']:6d}{p['pct_model']:7.1f}%"
              f"{p['base_t']:9.1f}{p['hyb_t']:8.1f}{p['speedup']:9.2f}"
              f"{p['base_m']:9.4f}{p['hyb_m']:8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
