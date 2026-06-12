"""Optical-flow color-wheel visualization (Middlebury convention).

Used by the A2 smoke test to sanity-check SEA-RAFT output.
"""
from __future__ import annotations

import numpy as np


def _make_color_wheel() -> np.ndarray:
    """Build the Middlebury color wheel (Ncols x 3, uint8)."""
    ry, yg, gc, cb, bm, mr = 15, 6, 4, 11, 13, 6
    ncols = ry + yg + gc + cb + bm + mr
    wheel = np.zeros((ncols, 3), dtype=np.uint8)
    col = 0

    def fill(start, count, ch_inc, ch_dec, inc_val):
        nonlocal col
        ramp = np.floor(255 * np.arange(count) / count).astype(np.uint8)
        if ch_inc is not None:
            wheel[start:start + count, ch_inc] = ramp if inc_val else 255
        if ch_dec is not None:
            wheel[start:start + count, ch_dec] = 255 - ramp
        col += count

    # R->Y
    wheel[0:ry, 0] = 255
    wheel[0:ry, 1] = np.floor(255 * np.arange(ry) / ry)
    col = ry
    # Y->G
    wheel[col:col + yg, 0] = 255 - np.floor(255 * np.arange(yg) / yg)
    wheel[col:col + yg, 1] = 255
    col += yg
    # G->C
    wheel[col:col + gc, 1] = 255
    wheel[col:col + gc, 2] = np.floor(255 * np.arange(gc) / gc)
    col += gc
    # C->B
    wheel[col:col + cb, 1] = 255 - np.floor(255 * np.arange(cb) / cb)
    wheel[col:col + cb, 2] = 255
    col += cb
    # B->M
    wheel[col:col + bm, 2] = 255
    wheel[col:col + bm, 0] = np.floor(255 * np.arange(bm) / bm)
    col += bm
    # M->R
    wheel[col:col + mr, 2] = 255 - np.floor(255 * np.arange(mr) / mr)
    wheel[col:col + mr, 0] = 255
    return wheel


_COLOR_WHEEL = _make_color_wheel()


def flow_to_color(flow: np.ndarray, max_magnitude: float | None = None) -> np.ndarray:
    """Convert an (H, W, 2) flow field to an (H, W, 3) uint8 RGB image.

    Hue encodes direction, saturation/value encode magnitude (normalized by
    ``max_magnitude``, or the field's own max if not given).
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"expected (H, W, 2) flow, got {flow.shape}")

    u = flow[..., 0].astype(np.float64)
    v = flow[..., 1].astype(np.float64)
    invalid = ~np.isfinite(u) | ~np.isfinite(v)
    u = np.where(invalid, 0.0, u)
    v = np.where(invalid, 0.0, v)

    rad = np.sqrt(u ** 2 + v ** 2)
    if max_magnitude is None:
        max_magnitude = max(rad.max(), 1e-5)
    rad = rad / max_magnitude

    angle = np.arctan2(-v, -u) / np.pi  # [-1, 1]
    ncols = _COLOR_WHEEL.shape[0]
    fk = (angle + 1.0) / 2.0 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % ncols
    f = fk - k0

    img = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    for ch in range(3):
        c0 = _COLOR_WHEEL[k0, ch] / 255.0
        c1 = _COLOR_WHEEL[k1, ch] / 255.0
        c = (1 - f) * c0 + f * c1
        # increase saturation with radius (Middlebury convention)
        c = np.where(rad <= 1, 1 - rad * (1 - c), c * 0.75)
        img[..., ch] = np.floor(255 * c).astype(np.uint8)

    img[invalid] = 0
    return img


def draw_flow_arrows(
    color_img: np.ndarray,
    flow: np.ndarray,
    stride: int = 60,
    scale: float = 1.0,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """Overlay a sparse arrow grid on a colour-wheel flow image.

    Arrows are drawn at every ``stride`` pixels; length = flow magnitude * scale.
    A thin dark outline is added so arrows stay readable on bright backgrounds.
    """
    import cv2

    out = color_img.copy()
    h, w = flow.shape[:2]
    for y in range(stride // 2, h, stride):
        for x in range(stride // 2, w, stride):
            u, v = float(flow[y, x, 0]), float(flow[y, x, 1])
            ex = int(round(x + u * scale))
            ey = int(round(y + v * scale))
            if (ex, ey) == (x, y):
                continue
            cv2.arrowedLine(out, (x, y), (ex, ey), (0, 0, 0), thickness + 1,
                            tipLength=0.3)
            cv2.arrowedLine(out, (x, y), (ex, ey), color, thickness,
                            tipLength=0.3)
    return out
