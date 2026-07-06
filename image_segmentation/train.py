"""
Train an image semantic-segmentation model.

    python train.py --config configs/uavid_convnext_large.yaml
    python train.py --config configs/uavid_hiera_small.yaml --set train.epochs=80 model.backbone=swin_large

Config is a YAML file (see configs/). Any field can be overridden on the CLI
with `--set dotted.key=value`. The loop trains with AdamW + cosine schedule,
evaluates val mIoU each epoch, keeps the best checkpoint, and early-stops once
mIoU plateaus.
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from data import build_loaders, build_ssp_loaders, CLASSES, IGNORE_INDEX
from models import build_model


# ----------------------------- config ------------------------------------- #
def load_config(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)  # parse ints/floats/bools
    return cfg


# ----------------------------- metrics ------------------------------------ #
class ConfusionMatrix:
    """Streaming confusion matrix -> per-class IoU and mean IoU."""

    def __init__(self, n):
        self.n = n
        self.mat = np.zeros((n, n), dtype=np.int64)

    def update(self, pred, target):
        k = (target >= 0) & (target < self.n)
        self.mat += np.bincount(self.n * target[k].astype(int) + pred[k],
                                minlength=self.n ** 2).reshape(self.n, self.n)

    def ious(self):
        m = self.mat
        inter = np.diag(m)
        union = m.sum(1) + m.sum(0) - inter
        return inter / np.maximum(union, 1)

    def miou(self):
        return float(np.mean(self.ious()))


# ----------------------------- schedule ----------------------------------- #
def cosine_warmup(optimizer, warmup_iters, total_iters, start_lr=0.0, final_lr=0.0):
    def fn(it):
        if it < warmup_iters:
            return start_lr + it / max(1, warmup_iters) * (1 - start_lr)
        p = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        return max(final_lr, final_lr + (1 - final_lr) * 0.5 * (1 + math.cos(math.pi * p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ----------------------------- train/eval --------------------------------- #
def run_epoch(model, loader, criterion, device, amp, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    cm = ConfusionMatrix(len(CLASSES)) if not train else None
    for img, mask in tqdm(loader, leave=False):
        # Keep the operation order identical to SSP's train_one_epoch.
        # In particular, gradients are cleared before the forward pass.
        if train:
            optimizer.zero_grad()
        img, mask = img.to(device), mask.to(device)
        if train:
            if amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(img)
                    loss = criterion(out, mask)
            else:
                out = model(img)
                loss = criterion(out, mask)
        else:
            with torch.no_grad():
                if amp:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(img)
                        loss = criterion(out, mask)
                else:
                    out = model(img)
                    loss = criterion(out, mask)
        if train:
            loss.backward()
            optimizer.step()
            scheduler.step()
        else:
            pred = out.argmax(1).cpu().numpy()
            cm.update(pred.ravel(), mask.cpu().numpy().ravel())
        total += loss.item()
    return total / len(loader), (cm if not train else None)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", help="override cfg: dotted.key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    d, m, t, o = cfg["data"], cfg["model"], cfg["train"], cfg["output"]
    set_seed(t.get("seed"))
    device = "cuda"
    run_name = f"{m['backbone']}_{time.strftime('%m%d_%H%M')}"
    run_dir = os.path.join(o["dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[run] {run_dir}")

    if d.get("pipeline") == "ssp":
        train_loader, val_loader = build_ssp_loaders(d["root"], d["crop_size"], t["batch_size"], t["num_workers"], d.get("augmentation", True))
    else:
        train_loader, val_loader = build_loaders(d["root"], d["crop_size"], t["batch_size"], t["num_workers"])
    model = build_model(m["backbone"], len(CLASSES), m.get("sam2_checkpoint_dir", "base_checkpoints"), m.get("upsampling", "bilinear"), m.get("decoder_style", "ssp")).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    iters_per_epoch = len(train_loader)
    scheduler = cosine_warmup(optimizer, t["warmup_epochs"] * iters_per_epoch, t["epochs"] * iters_per_epoch, t.get("start_lr", 0.0), t.get("final_lr", 0.0))
    amp = t.get("amp", True)

    best_miou, best_epoch, since_improved = -1.0, 0, 0
    log_path = os.path.join(run_dir, "log.txt")
    run_epochs = t.get("run_epochs", t["epochs"])
    for epoch in range(1, run_epochs + 1):
        tr_loss, _ = run_epoch(model, train_loader, criterion, device, amp, optimizer, scheduler)
        val_loss, cm = run_epoch(model, val_loader, criterion, device, amp)
        miou = cm.miou()
        per_class = {c: round(float(v), 4) for c, v in zip(CLASSES, cm.ious())}
        line = f"epoch {epoch:3d} | train {tr_loss:.4f} | val {val_loss:.4f} | mIoU {miou:.4f} | {per_class}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

        # checkpoint: always save last, snapshot best
        torch.save({"model": model.state_dict(), "epoch": epoch, "miou": miou, "cfg": cfg},
                   os.path.join(run_dir, "last.pth"))
        if miou > best_miou + 1e-9:
            best_miou, best_epoch, since_improved = miou, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "miou": miou, "cfg": cfg},
                       os.path.join(run_dir, "best.pth"))
        else:
            since_improved += 1

        # plateau early-stop
        es = t.get("early_stop", {})
        if es and epoch >= es.get("min_epochs", 40) and since_improved >= es.get("patience", 15):
            print(f"[early-stop] no >+{es.get('delta',0)} improvement for {since_improved} epochs")
            break

    print(f"[done] best mIoU {best_miou:.4f} @ epoch {best_epoch}  ->  {run_dir}/best.pth")


if __name__ == "__main__":
    main()
