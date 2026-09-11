"""Re-derive every hard-coded number in the report and diff it against source.

`make_figures.py` guarantees that macro-driven values match the data, because it
writes them. This script covers the remaining risk: values typed directly into
the LaTeX tables. It re-computes each from the committed result files and fails
loudly on any mismatch.

    python scripts/audit_numbers.py        # exit 0 = every value checks out

Values that provably cannot be re-derived (development measurements whose logs
were not retained) are listed under UNVERIFIABLE and are marked with a
provenance note in the report itself.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VAL = REPO / "outputs/adaptive_inference_metrics/uavid_val"
BACKEND = REPO / "outputs/backend_predictions"

FAILURES: list[str] = []
CHECKS = 0


def chk(label: str, data: float, report: float, tol: float) -> None:
    global CHECKS
    CHECKS += 1
    if abs(data - report) > tol:
        FAILURES.append(f"{label}: report={report} data={data:.4f} (tol {tol})")
        print(f"  FAIL {label:42s} report={report:<9} data={data:.4f}")


def load() -> dict[str, list[dict]]:
    out = {}
    for p in sorted(VAL.glob("*.csv")):
        out[p.stem] = list(csv.DictReader(p.open()))
    if not out:
        sys.exit(f"no result CSVs under {VAL}")
    return out


# ---------------------------------------------------------------------------
# Table 5.4, the per-sequence validation results, exactly as typed in the
# report: calls, baseline s, hybrid s, speedup, baseline mIoU, hybrid mIoU.
# The time columns are printed to whole seconds, so the tolerance is half a
# second; everything else is checked tightly.
TIMING = {
    "seq16": (71,  337, 181, 1.86, 0.763, 0.711),
    "seq17": (208, 334, 210, 1.59, 0.780, 0.758),
    "seq18": (76,  334, 184, 1.82, 0.835, 0.813),
    "seq19": (194, 364, 222, 1.64, 0.744, 0.728),
    "seq20": (474, 363, 279, 1.30, 0.773, 0.739),
    "seq36": (131, 365, 211, 1.73, 0.842, 0.831),
    "seq37": (87,  335, 189, 1.77, 0.759, 0.708),
}
POOLED = {"base_s": 2433, "hyb_s": 1476, "speedup": 1.65,
          "base_m": 0.785, "hyb_m": 0.755}

# Section 5.2 / Table 5.3, the quality half of the flow-backend comparison
BACKENDS = {
    "sea_raft": (90, 0.7957, 0.8162, 88.4),
    "flownet2": (90, 0.7951, 0.8160, 88.8),
}

# Claims made in prose that are arithmetic on the table above
PROSE = {
    "aggregate propagation cost in mIoU (Section 5.5)": 0.030,
    "backend propagation mIoU gap (Section 5.2)": 0.0006,
    "backend valid-pixel mIoU gap (Section 5.2)": 0.0001,
    "backend coverage gap in points (Section 5.2)": 0.4,
}

UNVERIFIABLE = [
    "Table 5.1 inference latency for ConvNeXt-L (365 ms), Swin-L (433 ms) and"
    " Hiera-B+ (327 ms) -- single-frame benchmark, log not retained",
    "Table 5.3 per-pair flow timings (150 ms / 532 ms) and the throughputs"
    " derived from them -- log not retained",
    "Table 5.2 rows tau=0.75 and tau=0.90 -- development runs, per-frame logs"
    " not retained",
]


def main() -> int:
    rows = load()

    print("Table 5.4 -- per-sequence validation results")
    tb = th = 0.0
    tcalls = tframes = 0
    for seq, rr in rows.items():
        model = [r for r in rr if r["hybrid_source"] == "model"]
        base = sum(float(r["baseline_elapsed_s"]) for r in rr)
        hyb = sum(float(r["hybrid_elapsed_s"]) for r in rr)
        gt = [r for r in rr if r["baseline_miou"]]
        bm = sum(float(r["baseline_miou"]) for r in gt) / len(gt)
        hm = sum(float(r["hybrid_miou"]) for r in gt) / len(gt)
        w = TIMING[seq]
        chk(f"{seq} calls", len(model), w[0], 0)
        chk(f"{seq} baseline s", base, w[1], 0.5)
        chk(f"{seq} hybrid s", hyb, w[2], 0.5)
        chk(f"{seq} speedup", base / hyb, w[3], 0.005)
        chk(f"{seq} baseline mIoU", bm, w[4], 0.0005)
        chk(f"{seq} hybrid mIoU", hm, w[5], 0.0005)
        tb += base
        th += hyb
        tcalls += len(model)
        tframes += len(rr)

    print("Table 5.4 -- pooled row")
    gt = [r for rr in rows.values() for r in rr if r["baseline_miou"]]
    mean = lambda rs, k: sum(float(r[k]) for r in rs) / len(rs)
    chk("pooled baseline s", tb, POOLED["base_s"], 0.5)
    chk("pooled hybrid s", th, POOLED["hyb_s"], 0.5)
    chk("pooled speedup", tb / th, POOLED["speedup"], 0.005)
    chk("pooled baseline mIoU", mean(gt, "baseline_miou"), POOLED["base_m"], 0.0005)
    chk("pooled hybrid mIoU", mean(gt, "hybrid_miou"), POOLED["hyb_m"], 0.0005)
    chk("pooled mean calls", tcalls / len(rows), 177, 0.5)
    chk("n GT points", len(gt), 70, 0)
    # model-source frames must be bit-identical to the baseline, not merely close
    mod = [r for r in gt if r["hybrid_source"] == "model"]
    if mean(mod, "baseline_miou") != mean(mod, "hybrid_miou"):
        FAILURES.append("model-source frames are not identical to the baseline")

    print("Table 5.3 -- flow backends (quality and coverage only)")
    agg = {}
    for backend, want in BACKENDS.items():
        rr = []
        for m in sorted(BACKEND.glob(f"*/{backend}/metrics.csv")):
            rr += list(csv.DictReader(m.open()))
        agg[backend] = (mean(rr, "miou_all"), mean(rr, "miou_valid"),
                        mean(rr, "valid_pct"))
        chk(f"{backend} pairs", len(rr), want[0], 0)
        chk(f"{backend} mIoU all", agg[backend][0], want[1], 0.00005)
        chk(f"{backend} mIoU valid", agg[backend][1], want[2], 0.00005)
        chk(f"{backend} coverage pct", agg[backend][2], want[3], 0.05)

    print("Prose claims that are arithmetic on the tables above")
    chk("aggregate mIoU cost",
        mean(gt, "baseline_miou") - mean(gt, "hybrid_miou"),
        PROSE["aggregate propagation cost in mIoU (Section 5.5)"], 0.0005)
    chk("backend mIoU all gap",
        agg["sea_raft"][0] - agg["flownet2"][0],
        PROSE["backend propagation mIoU gap (Section 5.2)"], 0.00005)
    chk("backend mIoU valid gap",
        agg["sea_raft"][1] - agg["flownet2"][1],
        PROSE["backend valid-pixel mIoU gap (Section 5.2)"], 0.00005)
    chk("backend coverage gap",
        agg["flownet2"][2] - agg["sea_raft"][2],
        PROSE["backend coverage gap in points (Section 5.2)"], 0.05)

    print(f"\npooled: baseline {tb:.1f}s, hybrid {th:.1f}s, {tb / th:.3f}x, "
          f"{tcalls} calls over {tframes} frames "
          f"({100 * tcalls / tframes:.1f}%)")

    print(f"\nNot re-derivable from committed artifacts ({len(UNVERIFIABLE)} "
          "items, each marked in the report):")
    for u in UNVERIFIABLE:
        print(f"  - {u}")

    if FAILURES:
        print(f"\n{len(FAILURES)} of {CHECKS} checks FAILED:")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print(f"\nAll {CHECKS} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
