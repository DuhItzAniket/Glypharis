"""Cut a text line out of a page and straighten it into a horizontal strip."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class CropSettings:
    height: int = 64  # stored crop height; the recognizer resizes to its own input height
    margin: float = 0.15  # context kept on every side, as a fraction of the line height
    max_aspect: float = 64.0  # longer lines are not cut (see glypharis.data.pages)
    jpeg_quality: int = 95

    def to_dict(self) -> dict[str, float | int]:
        return {k: getattr(self, k) for k in ("height", "margin", "max_aspect", "jpeg_quality")}


def _lerp(a: tuple[float, float], b: tuple[float, float], t: float) -> tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def expand_quad(
    quad: list[list[float]] | list[tuple[float, float]], margin: float
) -> list[tuple[float, float]]:
    """Grow a TL, TR, BR, BL quadrilateral by ``margin`` × line height on every side."""
    tl, tr, br, bl = [(float(p[0]), float(p[1])) for p in quad]
    h = (math.dist(tl, bl) + math.dist(tr, br)) / 2
    w = (math.dist(tl, tr) + math.dist(bl, br)) / 2
    if h <= 0 or w <= 0:
        return [tl, tr, br, bl]
    tv, tw = margin, margin * h / w  # both sides grow by margin × height
    # Extend top/bottom edges outward along the left and right sides, then left/right along the
    # (already extended) top and bottom edges.
    tl2, bl2 = _lerp(bl, tl, 1 + tv), _lerp(tl, bl, 1 + tv)
    tr2, br2 = _lerp(br, tr, 1 + tv), _lerp(tr, br, 1 + tv)
    tl3, tr3 = _lerp(tr2, tl2, 1 + tw), _lerp(tl2, tr2, 1 + tw)
    bl3, br3 = _lerp(br2, bl2, 1 + tw), _lerp(bl2, br2, 1 + tw)
    return [tl3, tr3, br3, bl3]


def background_grey(gray: Image.Image) -> int:
    """Median grey of a 16×16 thumbnail: a cheap estimate of the page background."""
    return int(np.median(np.asarray(gray.resize((16, 16)))))


def rectify(
    gray: Image.Image, quad: list[tuple[float, float]], settings: CropSettings, fill: int
) -> Image.Image:
    """Map the quadrilateral to an upright strip of ``settings.height`` pixels.

    ``gray`` is the page in mode "L". Uses PIL's bilinear QUAD transform, which is adequate for
    near-planar text lines. Points outside the page get ``fill`` (see `background_grey`) rather than
    black, so margins look like paper.
    """
    tl, tr, br, bl = quad
    h = (math.dist(tl, bl) + math.dist(tr, br)) / 2
    w = (math.dist(tl, tr) + math.dist(bl, br)) / 2
    out_h = settings.height
    out_w = max(1, round(out_h * w / max(h, 1e-6)))  # aspect ratio kept; never squeezed
    data = (*tl, *bl, *br, *tr)  # PIL order: upper-left, lower-left, lower-right, upper-right
    return gray.transform(
        (out_w, out_h),
        Image.Transform.QUAD,
        data,
        resample=Image.Resampling.BICUBIC,
        fillcolor=fill,
    )


def encode_jpeg(img: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
