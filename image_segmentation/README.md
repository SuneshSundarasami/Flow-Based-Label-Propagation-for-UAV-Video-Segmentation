# image_segmentation

A small, self-contained trainer for **image semantic segmentation** on aerial
(UAVid) imagery, pulled out of the SSP research repo. It trains a single-frame
segmentation model with a configurable backbone and reports validation mIoU.

Originally built to train the **teacher / base image model** used for knowledge
distillation, but it works standalone for any of the supported backbones.

## What's here

```
configs/        one YAML per backbone — the main thing you edit
data.py         UAVid dataset (reads the raw release directly), augmentations, loaders
models.py       build_model() + backbone registry (add backbones here)
hiera.py        vendored Hiera (SAM2) backbone — don't edit
train.py        training loop: mIoU, cosine schedule, best ckpt, early stop
infer.py        predictions / pseudo-logits from a checkpoint
```

No nested packages, no framework — five short top-level modules.

## Supported backbones

Set `model.backbone` in a config to any key below (see `models.BACKBONES`):

| key | source | notes |
|-----|--------|-------|
| `hiera_small`, `hiera_base_plus` | SAM2 (local weights) | needs `sam2_hiera_<size>.pt` |
| `convnext_large`, `convnext_small` | 🤗 `openmmlab/upernet-convnext-*` | UPerNet |
| `swin_large`, `swin_small` | 🤗 `openmmlab/upernet-swin-*` | UPerNet |
| `segformer_b2`, `segformer_b3` | 🤗 `nvidia/mit-b*` | SegFormer |

**Add a backbone** = one line in `BACKBONES` in `models.py`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements.txt
```

Use whichever CUDA wheel index matches your driver (see
https://pytorch.org/get-started/locally/); the rest of `requirements.txt`
(transformers, albumentations, etc.) is backend-agnostic. Activate with
`source .venv/bin/activate`, or just call `.venv/bin/python` directly as
shown below.

For `hiera_*` backbones, download the SAM2 weights into the dir named by
`model.sam2_checkpoint_dir` (default `sam2_checkpoints/`):
`sam2_hiera_small.pt`, `sam2_hiera_base_plus.pt`
(from https://github.com/facebookresearch/sam2#sam-2-checkpoints).

## Data

The dataset is read **directly from the raw UAVid release** — nothing is copied
or pre-processed to disk. Point `data.root` at the release folder:

```
<root>/uavid_train/<seq>/Images/*.png   # RGB frames (labeled keyframes)
<root>/uavid_train/<seq>/Labels/*.png   # RGB color-coded masks
<root>/uavid_val/<seq>/{Images,Labels}
```

Split is by folder (`uavid_train/` → train, `uavid_val/` → val). RGB label
colors are mapped to training ids on the fly. Classes (background is ignored):
road, vegetation, tree, person, vehicle, building. To change the label scheme,
edit `CLASSES` / `COLOR_TO_ID` in `data.py`.

## Train

```bash
.venv/bin/python train.py --config configs/uavid_convnext_large.yaml
```

Override any config field on the CLI:

```bash
.venv/bin/python train.py --config configs/uavid_hiera_small.yaml \
    --set train.epochs=80 train.batch_size=8 model.backbone=swin_large
```

Each run writes to `runs/<backbone>_<time>/`: `config.yaml`, `log.txt`,
`last.pth`, `best.pth`. Training early-stops once val mIoU plateaus
(`train.early_stop` in the config).

## Inference / pseudo-logits

```bash
.venv/bin/python infer.py --checkpoint runs/<run>/best.pth --images <frames_dir> --out preds
.venv/bin/python infer.py --checkpoint runs/<run>/best.pth --images <frames_dir> --out preds --save-logits
```

`--save-logits` writes per-frame `.npz` soft labels (for knowledge distillation).
