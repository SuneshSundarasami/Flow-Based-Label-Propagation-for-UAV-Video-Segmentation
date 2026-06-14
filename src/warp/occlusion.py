"""Forward-backward consistency occlusion detection (WP B2).

A pixel in the target frame t is marked *valid* if warping it back to the
keyframe k and then forward again lands near its origin (Brox et al., 2004).

Round-trip for pixel p in t:
    1. q  = p  + flow_bwd[p]          # backward t→k: find source in k
    2. r  = q  + flow_fwd(q)          # forward  k→t: sample fwd flow at q
    3. error = ||r - p|| = ||flow_bwd[p] + flow_fwd(q)||

Pixels with error > threshold, or whose source q falls outside the frame,
are flagged as occluded / unreliable.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_fb_mask(
    flow_fwd: np.ndarray,
    flow_bwd: np.ndarray,
    threshold: float = 1.5,
) -> np.ndarray:
    """Compute a per-pixel validity mask via forward-backward consistency.

    Parameters
    ----------
    flow_fwd:
        (H, W, 2) float32 — forward flow, keyframe k → target t.
    flow_bwd:
        (H, W, 2) float32 — backward flow, target t → keyframe k.
    threshold:
        Round-trip error (pixels) above which a pixel is marked occluded.

    Returns
    -------
    (H, W) bool array — True where the pixel is valid (not occluded).
    """
    if flow_fwd.shape != flow_bwd.shape or flow_fwd.ndim != 3 or flow_fwd.shape[2] != 2:
        raise ValueError(
            f"flow_fwd and flow_bwd must both be (H, W, 2); "
            f"got {flow_fwd.shape} and {flow_bwd.shape}"
        )

    h, w = flow_fwd.shape[:2]

    ys, xs = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    bwd = torch.from_numpy(flow_bwd)
    src_x = xs + bwd[..., 0]
    src_y = ys + bwd[..., 1]

    # Pixels whose backward source lands outside the keyframe are always invalid.
    in_bounds = (src_x >= 0) & (src_x < w) & (src_y >= 0) & (src_y < h)

    # Sample forward flow at the (non-integer) source locations.
    norm_x = 2.0 * src_x / max(w - 1, 1) - 1.0
    norm_y = 2.0 * src_y / max(h - 1, 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    fwd_t = torch.from_numpy(flow_fwd).permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
    fwd_sampled = F.grid_sample(
        fwd_t, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    fwd_sampled = fwd_sampled.squeeze(0).permute(1, 2, 0).numpy()  # (H, W, 2)

    residual = flow_bwd + fwd_sampled
    error = np.linalg.norm(residual, axis=2)

    valid = in_bounds.numpy() & (error < threshold)
    return valid
