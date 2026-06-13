"""Flow-based nearest-neighbour mask warping (WP B1).

Convention: ``flow`` is in the t→k direction (backward warp). Each pixel
(x, y) in the target frame t samples from location (x + u, y + v) in the
keyframe k.  This lets us directly assign keyframe labels to target pixels.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def warp_mask(
    mask: np.ndarray,
    flow: np.ndarray,
    ignore_index: int = 255,
) -> np.ndarray:
    """Warp a keyframe label mask into a target frame using backward flow.

    Parameters
    ----------
    mask:
        (H, W) integer array — the keyframe's segmentation mask.
    flow:
        (H, W, 2) float32 — flow from target→keyframe.  ``flow[y, x] = (u, v)``
        means target pixel (x, y) maps back to keyframe location (x+u, y+v).
    ignore_index:
        Label assigned to pixels whose source falls outside the keyframe bounds.

    Returns
    -------
    (H, W) array with the same dtype as ``mask``.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be (H, W), got {mask.shape}")
    if flow.ndim != 3 or flow.shape[2] != 2 or flow.shape[:2] != mask.shape:
        raise ValueError(
            f"flow must be (H, W, 2) matching mask shape, "
            f"got flow={flow.shape}, mask={mask.shape}"
        )

    h, w = mask.shape

    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    flow_t = torch.from_numpy(flow)
    src_x = xs + flow_t[..., 0]
    src_y = ys + flow_t[..., 1]

    valid = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

    # Normalize pixel coords to [-1, 1] for grid_sample (align_corners=True).
    norm_x = 2.0 * src_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(h - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    mask_t = torch.from_numpy(mask.astype(np.int32)).float().unsqueeze(0).unsqueeze(0)

    warped = F.grid_sample(
        mask_t, grid, mode="nearest", padding_mode="zeros", align_corners=True
    )
    out = warped.squeeze().numpy().astype(mask.dtype)
    out[~valid.numpy()] = ignore_index
    return out
