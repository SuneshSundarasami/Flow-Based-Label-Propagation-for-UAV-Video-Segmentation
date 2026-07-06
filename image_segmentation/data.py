"""
UAVid semantic-segmentation dataset (6-class, no VDD) read DIRECTLY from the
raw official release — no pre-processing, no extra files on disk.

Point `data.root` at the release folder:

    <root>/uavid_train/<seq>/Images/*.png   RGB frames (labeled keyframes)
    <root>/uavid_train/<seq>/Labels/*.png   RGB color-coded masks
    <root>/uavid_val/<seq>/{Images,Labels}

Split is by folder: everything under `uavid_train/` is train, `uavid_val/` is
val. RGB label colors are mapped to training ids on the fly via a lookup table
(background -> ignore). Edit `CLASSES` / `COLOR_TO_ID` to change the scheme.
"""
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASSES = ["road", "vegetation", "tree", "person", "vehicle", "building"]
IGNORE_INDEX = 255
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# Original UAVid Label RGB -> training id (background/clutter -> IGNORE_INDEX).
COLOR_TO_ID = {
    (0, 0, 0): IGNORE_INDEX,   # background / clutter -> ignored
    (128, 64, 128): 0,         # road
    (128, 128, 0): 1,          # low vegetation
    (0, 128, 0): 2,            # tree
    (64, 64, 0): 3,            # human / person
    (64, 0, 128): 4,           # moving car
    (192, 0, 192): 4,          # static car (merged)
    (128, 0, 0): 5,            # building
}

# 2^24 lookup table (16 MB) mapping packed RGB -> training id. Built once.
_LUT = np.full(1 << 24, IGNORE_INDEX, dtype=np.uint8)
for (_r, _g, _b), _cid in COLOR_TO_ID.items():
    _LUT[(_r << 16) | (_g << 8) | _b] = _cid


def rgb_label_to_ids(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.uint32)
    packed = (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]
    return _LUT[packed]


def build_transforms(crop_size: int, train: bool):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=(-0.25, 0.4), p=0.5),
            A.PadIfNeeded(crop_size, crop_size, border_mode=0, value=0, mask_value=IGNORE_INDEX),
            A.RandomCrop(crop_size, crop_size),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ])
    return A.Compose([
        A.PadIfNeeded(crop_size, crop_size, border_mode=0, value=0, mask_value=IGNORE_INDEX),
        A.CenterCrop(crop_size, crop_size),
        A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


class UAVidDataset(Dataset):
    def __init__(self, root: str, split: str, crop_size: int):
        split_dir = os.path.join(root, "uavid_train" if split == "train" else "uavid_val")
        self.samples = []  # (image_path, label_path)
        for seq in sorted(os.listdir(split_dir)):
            img_dir = os.path.join(split_dir, seq, "Images")
            lbl_dir = os.path.join(split_dir, seq, "Labels")
            if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
                continue
            for fn in sorted(os.listdir(lbl_dir)):
                if fn.endswith(".png") and os.path.isfile(os.path.join(img_dir, fn)):
                    self.samples.append((os.path.join(img_dir, fn), os.path.join(lbl_dir, fn)))
        self.tf = build_transforms(crop_size, train=(split == "train"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_path, lbl_path = self.samples[i]
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = rgb_label_to_ids(np.array(Image.open(lbl_path).convert("RGB")))
        out = self.tf(image=img, mask=mask)
        return out["image"], out["mask"].long()


def build_loaders(root: str, crop_size: int, batch_size: int, num_workers: int):
    train_ds = UAVidDataset(root, "train", crop_size)
    val_ds = UAVidDataset(root, "val", crop_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  classes={len(CLASSES)}")
    return train_loader, val_loader


# --------------------------------------------------------------------------- #
# SSP-compatible path
# --------------------------------------------------------------------------- #
class SSPUAVidDataset(Dataset):
    """Local reproduction of SSP's prepared UAVid image dataset."""

    def __init__(self, root: str, split: str, crop_size, train: bool, augmentation: bool):
        self.root = root
        self.crop_size = tuple(crop_size)
        with open(os.path.join(root, "train.txt" if split == "train" else "val.txt")) as f:
            sequences = [line.strip() for line in f if line.strip()]
        self.samples = []
        for sequence in sequences:
            origin = os.path.join(root, "data", sequence, "origin")
            masks = os.path.join(root, "data", sequence, "mask")
            for filename in sorted(os.listdir(origin)):
                if os.path.isfile(os.path.join(masks, filename)):
                    self.samples.append((os.path.join(origin, filename), os.path.join(masks, filename)))
        self.augment = self._build_augment() if augmentation and train else None
        self.center_crop = A.CenterCrop(self.crop_size[0], self.crop_size[0]) if self.crop_size[0] != self.crop_size[1] else A.NoOp()
        self.normalize = A.Compose([
            A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=255.0),
            ToTensorV2(),
        ])

    def _build_augment(self):
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.Perspective(p=1, scale=(0.01, 0.05)),
                A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.0625,
                                   interpolation=1, border_mode=0, value=0, mask_value=0, p=1.0),
            ], p=0.2),
            A.OneOf([A.RandomBrightnessContrast(p=0.5), A.HueSaturationValue(p=0.5),
                     A.RandomGamma(p=0.5), A.CLAHE(p=0.5)], p=0.8),
            A.OneOf([A.ISONoise(p=0.5), A.GaussNoise(p=0.5),
                     A.ImageCompression(p=0.5), A.Sharpen(p=0.5)], p=0.6),
            A.RandomCrop(self.crop_size[0], self.crop_size[0]) if self.crop_size[0] != self.crop_size[1] else A.NoOp(),
        ], is_check_shapes=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB").resize((self.crop_size[1], self.crop_size[0]), Image.BILINEAR)
        mask = Image.open(mask_path).resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
        image, mask = np.array(image), np.array(mask)
        if self.augment is not None:
            transformed = self.augment(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        cropped = self.center_crop(image=image, mask=mask)
        image, mask = cropped["image"], cropped["mask"]
        image = self.normalize(image=image)["image"]
        mask = mask.copy()
        mask[mask == 0] = IGNORE_INDEX
        mask = mask - 1
        mask[mask >= len(CLASSES)] = IGNORE_INDEX
        return image, torch.LongTensor(mask)


def build_ssp_loaders(root: str, crop_size, batch_size: int, num_workers: int, augmentation: bool = True):
    """Build the exact SSP sample split and drop-last behavior."""
    train_ds = SSPUAVidDataset(root, "train", crop_size, train=True, augmentation=augmentation)
    val_ds = SSPUAVidDataset(root, "val", crop_size, train=False, augmentation=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
                              num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True,
                            num_workers=num_workers, persistent_workers=num_workers > 0)
    print(f"[data:ssp] train={len(train_ds)}  val={len(val_ds)}  classes={len(CLASSES)}")
    return train_loader, val_loader
