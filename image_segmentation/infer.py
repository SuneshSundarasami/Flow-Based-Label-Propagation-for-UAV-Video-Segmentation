"""
Run a trained checkpoint on images: save colored predictions and/or raw logits.

    python infer.py --checkpoint runs/convnext_large_.../best.pth \
                    --images path/to/frames --out preds [--save-logits]

Logits are saved as compressed .npz (key "arr_0", shape [num_classes, H, W]) —
the format consumed by knowledge-distillation training.
"""
import argparse
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from data import CLASSES, _MEAN, _STD
from models import build_model

# palette for the 6 classes (background/ignore -> black)
PALETTE = [(128, 0, 128), (112, 148, 32), (64, 64, 0), (255, 16, 255), (0, 128, 128), (255, 0, 0)]


def preprocess(img):
    x = torch.from_numpy(np.array(img.convert("RGB"))).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(_MEAN).view(3, 1, 1)
    std = torch.tensor(_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def colorize(pred):
    out = np.zeros((*pred.shape, 3), np.uint8)
    for cid, color in enumerate(PALETTE):
        out[pred == cid] = color
    return Image.fromarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, help="dir of .png/.jpg frames")
    ap.add_argument("--out", default="preds")
    ap.add_argument("--save-logits", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(
        cfg["model"]["backbone"], len(CLASSES),
        cfg["model"].get("sam2_checkpoint_dir", "sam2_checkpoints"),
        cfg["model"].get("upsampling", "bilinear"),
        cfg["model"].get("decoder_style", "ssp"),
    )
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    print(f"[infer] loaded {args.checkpoint} (epoch {ckpt['epoch']}, mIoU {ckpt['miou']:.4f})")

    os.makedirs(args.out, exist_ok=True)
    if args.save_logits:
        os.makedirs(os.path.join(args.out, "logits"), exist_ok=True)
    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    for fn in files:
        img = Image.open(os.path.join(args.images, fn))
        x = preprocess(img).cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
            logits = model(x)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        colorize(pred).save(os.path.join(args.out, os.path.splitext(fn)[0] + ".png"))
        if args.save_logits:
            arr = logits[0].float().cpu().numpy()
            np.savez_compressed(os.path.join(args.out, "logits", os.path.splitext(fn)[0] + ".npz"), arr)
    print(f"[infer] wrote {len(files)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
