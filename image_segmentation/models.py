"""
Model factory for image semantic segmentation.

Two families of backbones are supported through one uniform interface:

  * Hugging Face models (ConvNeXt / Swin via UPerNet, SegFormer) — loaded
    directly from `transformers`, no custom code needed.
  * Hiera (the SAM2 image encoder) + a small UPerNet head — vendored locally
    in `hiera.py` and the head below, initialised from SAM2 weights.

Every model returned by `build_model` is an `nn.Module` whose `forward(x)`
returns class logits upsampled to the input resolution: shape (B, num_classes, H, W).

To add a backbone, add one entry to `BACKBONES`.
"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from hiera import Hiera
from upernet import UperNetHead


# --------------------------------------------------------------------------- #
# Backbone registry — add a line here to support a new backbone.
#   type "hf_upernet":  UPerNet segmentation model on the HF hub
#   type "hf_segformer": SegFormer (encoder id, head is created for num_classes)
#   type "hiera":        Hiera-SAM2 encoder + local UPerNet head
# --------------------------------------------------------------------------- #
BACKBONES = {
    "convnext_small": {"type": "hf_upernet", "hf": "openmmlab/upernet-convnext-small"},
    "convnext_large": {"type": "hf_upernet", "hf": "openmmlab/upernet-convnext-large"},
    "swin_small":     {"type": "hf_upernet", "hf": "openmmlab/upernet-swin-small"},
    "swin_large":     {"type": "hf_upernet", "hf": "openmmlab/upernet-swin-large"},
    "segformer_b2":   {"type": "hf_segformer", "hf": "nvidia/mit-b2"},
    "segformer_b3":   {"type": "hf_segformer", "hf": "nvidia/mit-b3"},
    "hiera_small":     {"type": "hiera", "size": "small"},
    "hiera_base_plus": {"type": "hiera", "size": "base_plus"},
}


def build_model(
    backbone: str,
    num_classes: int,
    sam2_checkpoint_dir: str = "base_checkpoints",
    upsampling: str = "bilinear",
    decoder_style: str = "ssp",
) -> nn.Module:
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Options: {list(BACKBONES)}")
    spec = BACKBONES[backbone]
    if spec["type"] == "hf_upernet":
        model = HFSegModel(spec["hf"], num_classes, kind="upernet")
    elif spec["type"] == "hf_segformer":
        model = HFSegModel(spec["hf"], num_classes, kind="segformer")
    elif spec["type"] == "hiera":
        model = HieraUPerNet(spec["size"], num_classes, sam2_checkpoint_dir, upsampling, decoder_style)
    else:
        raise ValueError(spec["type"])
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {backbone}: {n:.1f}M params")
    return model


# --------------------------------------------------------------------------- #
# Hugging Face wrapper (ConvNeXt/Swin UPerNet, SegFormer)
# --------------------------------------------------------------------------- #
class HFSegModel(nn.Module):
    def __init__(self, hf_id: str, num_classes: int, kind: str):
        super().__init__()
        if kind == "upernet":
            # NB: the official transformers UperNet correctly loads openmmlab
            # weights. (The version bundled in some research repos does not.)
            from transformers import UperNetForSemanticSegmentation
            self.model = UperNetForSemanticSegmentation.from_pretrained(
                hf_id, num_labels=num_classes, ignore_mismatched_sizes=True)
        else:
            from transformers import AutoModelForSemanticSegmentation
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                hf_id, num_labels=num_classes, ignore_mismatched_sizes=True)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


# --------------------------------------------------------------------------- #
# Hiera (SAM2) encoder + UPerNet head
# --------------------------------------------------------------------------- #
_HIERA_CFG = {
    #            embed_dim, num_heads, stages,        global_att_blocks, win_pe_bkg, head_ch
    "tiny":      (96,  1, [1, 2, 7, 2],  [5, 7, 9],    [7, 7],   128),
    "small":     (96,  1, [1, 2, 11, 2], [7, 10, 13],  [7, 7],   256),
    "base_plus": (112, 2, [2, 3, 16, 3], [12, 16, 20], [14, 14], 512),
}


class HieraUPerNet(nn.Module):
    def __init__(self, size: str, num_classes: int, sam2_checkpoint_dir: str, upsampling: str = "bilinear", decoder_style: str = "ssp"):
        super().__init__()
        self.upsampling = upsampling
        embed_dim, num_heads, stages, gab, winpe, head_ch = _HIERA_CFG[size]
        self.backbone = Hiera(
            embed_dim=embed_dim, num_heads=num_heads, stages=stages,
            global_att_blocks=gab, window_pos_embed_bkg_spatial_size=winpe)
        ckpt_path = f"{sam2_checkpoint_dir}/sam2_hiera_{size}.pt"
        state = torch.load(ckpt_path, weights_only=True)["model"]
        state = {k.replace("image_encoder.trunk.", ""): v for k, v in state.items()}
        missing = self.backbone.load_state_dict(state, strict=False)
        print(f"[hiera] loaded {ckpt_path} (unmatched keys: {len(missing.unexpected_keys)})")
        in_ch = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        if decoder_style == "ssp":
            self.head = UperNetHead(n_classes=num_classes, in_channels=in_ch, channels=head_ch)
        elif decoder_style == "legacy":
            self.head = LegacyUPerNetHead(num_classes, in_ch, channels=head_ch)
        else:
            raise ValueError(f"Unknown Hiera decoder style: {decoder_style}")

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.head(feats)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode=self.upsampling, align_corners=False)
        return logits


# --------------------------------------------------------------------------- #
# UPerNet head (self-contained; used only by the Hiera model)
# --------------------------------------------------------------------------- #
class _ConvModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _PPM(nn.ModuleList):
    def __init__(self, pool_scales, in_ch, channels):
        super().__init__()
        self.pool_scales = pool_scales
        for s in pool_scales:
            self.append(nn.Sequential(nn.AdaptiveAvgPool2d(s), _ConvModule(in_ch, channels, 1)))

    def forward(self, x):
        outs = []
        for block in self:
            y = block(x)
            outs.append(F.interpolate(y, size=x.shape[2:], mode="bilinear", align_corners=False))
        return outs


class LegacyUPerNetHead(nn.Module):
    """UPerNet decoder (PPM + FPN). `in_channels` is the 4-level feature dim list."""

    def __init__(self, num_classes, in_channels: List[int], channels=512, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.psp = _PPM(pool_scales, in_channels[-1], channels)
        self.bottleneck = _ConvModule(in_channels[-1] + len(pool_scales) * channels, channels, 3, padding=1)
        self.lateral_convs = nn.ModuleList(_ConvModule(c, channels, 1) for c in in_channels[:-1])
        self.fpn_convs = nn.ModuleList(_ConvModule(channels, channels, 3, padding=1) for _ in in_channels[:-1])
        self.fpn_bottleneck = _ConvModule(len(in_channels) * channels, channels, 3, padding=1)
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def _psp_forward(self, inputs):
        x = inputs[-1]
        outs = [x] + self.psp(x)
        return self.bottleneck(torch.cat(outs, dim=1))

    def forward(self, feats):
        laterals = [conv(feats[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(self._psp_forward(feats))
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False)
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(len(fpn_outs) - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False)
        out = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        return self.classifier(out)
