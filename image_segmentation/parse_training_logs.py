"""Parse runs/*/log.txt (one line per epoch, written by train.py) to find
each run's best epoch by val mIoU and its per-class IoU breakdown.

    image_segmentation/.venv/bin/python parse_training_logs.py
    image_segmentation/.venv/bin/python parse_training_logs.py --out results.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

LINE_RE = re.compile(r"epoch\s+(\d+) \| train ([\d.]+) \| val ([\d.]+) \| mIoU ([\d.]+) \| (\{.*\})")


def parse_log(log_path: Path) -> dict:
    best = None
    n_epochs = 0
    for line in log_path.read_text().splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        n_epochs += 1
        epoch, train_loss, val_loss, miou, classes = m.groups()
        miou = float(miou)
        if best is None or miou > best["miou"]:
            best = dict(epoch=int(epoch), train_loss=float(train_loss), val_loss=float(val_loss),
                        miou=miou, per_class_iou=ast.literal_eval(classes))
    best["epochs_trained"] = n_epochs
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default=str(Path(__file__).resolve().parent / "runs"))
    ap.add_argument("--out", default=None, help="optional path to write results as JSON")
    args = ap.parse_args()

    results = {}
    for run_dir in sorted(Path(args.runs_dir).glob("*/")):
        log_path = run_dir / "log.txt"
        if not log_path.exists():
            continue
        r = parse_log(log_path)
        results[run_dir.name] = r
        print(f"{run_dir.name}: trained {r['epochs_trained']} epochs, best @ epoch {r['epoch']}  "
              f"train={r['train_loss']:.4f} val={r['val_loss']:.4f} mIoU={r['miou']:.4f}")
        for k, v in r["per_class_iou"].items():
            print(f"    {k:12s} {v:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
