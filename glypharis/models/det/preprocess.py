"""Page preprocessing for detector inference (NumPy and OpenCV only, no torch).

One function shared by validation, the label ceiling, Phase 10 and Phase 17, so every consumer
feeds the network the same padded image:

1. scale the page so its long side is ``long_side`` (pages are enlarged as well as shrunk), with
   ``W' = round(W * s)`` and ``H' = round(H * s)``; ``INTER_AREA`` when shrinking, ``INTER_LINEAR``
   when enlarging;
2. pad right and bottom with `PAD_RGB` to multiples of `PAD_MULTIPLE`.

``cv2.resize`` is edge-aligned, so a point maps as ``x' = sx * x`` with the exact per-axis factors
``sx = W'/W`` and ``sy = H'/H`` that `ResizeInfo` carries; `to_page_coords` undoes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

PAD_MULTIPLE: Final = 32
PAD_RGB: Final = (124, 116, 104)  # round(255 * ImageNet mean): ~0 after the model normalises
PAD_BGR: Final = (104, 116, 124)  # the same colour, for warps of a BGR decode


@dataclass(frozen=True)
class ResizeInfo:
    """Geometry of one prepared page: original size, scaled size and padded size, in pixels."""

    orig_w: int
    orig_h: int
    w: int
    h: int
    pad_w: int
    pad_h: int

    @property
    def sx(self) -> float:
        return self.w / self.orig_w

    @property
    def sy(self) -> float:
        return self.h / self.orig_h


def scaled_size(orig_w: int, orig_h: int, long_side: int) -> tuple[int, int]:
    """Size of a page scaled so that its long side is ``long_side``."""
    if orig_w <= 0 or orig_h <= 0 or long_side <= 0:
        raise ValueError(f"bad sizes: page {orig_w}x{orig_h}, long side {long_side}")
    s = long_side / max(orig_w, orig_h)
    return max(1, round(orig_w * s)), max(1, round(orig_h * s))


def _pad_to(n: int) -> int:
    return -(-n // PAD_MULTIPLE) * PAD_MULTIPLE


def prepare_page(rgb: np.ndarray, long_side: int) -> tuple[np.ndarray, ResizeInfo]:
    """Resize an ``(H, W, 3)`` uint8 RGB page and pad it to ``(pad_h, pad_w, 3)``."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected an (H, W, 3) uint8 RGB image, got {rgb.shape} {rgb.dtype}")
    orig_h, orig_w = rgb.shape[:2]
    w, h = scaled_size(orig_w, orig_h, long_side)
    if (w, h) == (orig_w, orig_h):
        resized = rgb
    else:
        shrink = long_side < max(orig_w, orig_h)
        interp = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
        resized = cv2.resize(np.ascontiguousarray(rgb), (w, h), interpolation=interp)
    info = ResizeInfo(orig_w, orig_h, w, h, _pad_to(w), _pad_to(h))
    out = np.empty((info.pad_h, info.pad_w, 3), np.uint8)
    out[...] = PAD_RGB
    out[:h, :w] = resized
    return out, info


def to_page_coords(points: np.ndarray, info: ResizeInfo) -> np.ndarray:
    """Map ``(..., 2)`` network-scale edge coordinates back to the original page (float64)."""
    pts = np.array(points, dtype=np.float64)
    if pts.shape[-1] != 2:
        raise ValueError(f"expected (..., 2) points, got shape {pts.shape}")
    pts[..., 0] /= info.sx
    pts[..., 1] /= info.sy
    return pts
