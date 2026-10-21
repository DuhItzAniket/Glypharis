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


def line_quad(polygon: list[list[float]]) -> list[list[float]]:
    """TL, TR, BR, BL corners of a line polygon.

    Four-point polygons are returned as they are. Longer ones (densified GDC lines) run along the
    top edge left to right and back along the bottom edge, so the corners are the ends of each half.
    """
    if len(polygon) <= 4:
        return polygon
    n = len(polygon) // 2
    return [polygon[0], polygon[n - 1], polygon[n], polygon[-1]]


def _push(a: tuple[float, float], b: tuple[float, float], dist: float) -> tuple[float, float]:
    """Move ``b`` further from ``a`` by ``dist`` along the line a -> b."""
    d = math.dist(a, b)
    return b if d <= 0 else _lerp(a, b, 1 + dist / d)


def cut_line(
    gray: Image.Image, polygon: list[list[float]], settings: CropSettings, fill: int
) -> Image.Image:
    """Cut one line into an upright strip, following the polygon if it bends.

    A four-point polygon goes through `expand_quad` and `rectify`, exactly as the training crops
    were made. A longer polygon (paired top and bottom points along a curved line, as GDC writes
    them) is straightened segment by segment at one common scale and the pieces are joined, so text
    on a bowed page is not cut off in the middle.
    """
    n = len(polygon) // 2
    if len(polygon) <= 4 or len(polygon) % 2:
        return rectify(gray, expand_quad(line_quad(polygon), settings.margin), settings, fill)
    top = [(float(x), float(y)) for x, y in polygon[:n]]
    bottom = [(float(x), float(y)) for x, y in polygon[n:]][::-1]
    h = sum(math.dist(t, b) for t, b in zip(top, bottom, strict=True)) / n
    if h <= 0:
        return rectify(gray, expand_quad(line_quad(polygon), settings.margin), settings, fill)
    pad = settings.margin * h
    top2 = [_push(b, t, pad) for t, b in zip(top, bottom, strict=True)]
    bot2 = [_push(t, b, pad) for t, b in zip(top, bottom, strict=True)]
    top2[0], bot2[0] = _push(top2[1], top2[0], pad), _push(bot2[1], bot2[0], pad)
    top2[-1], bot2[-1] = _push(top2[-2], top2[-1], pad), _push(bot2[-2], bot2[-1], pad)
    out_h = settings.height
    scale = out_h / (h + 2 * pad)
    edges = [0.0]
    for i in range(n - 1):
        w = (math.dist(top2[i], top2[i + 1]) + math.dist(bot2[i], bot2[i + 1])) / 2
        edges.append(edges[-1] + w * scale)
    cols = [round(e) for e in edges]
    out = Image.new("L", (max(1, cols[-1]), out_h), fill)
    for i in range(n - 1):
        w = cols[i + 1] - cols[i]
        if w <= 0:
            continue
        data = (*top2[i], *bot2[i], *bot2[i + 1], *top2[i + 1])
        piece = gray.transform((w, out_h), Image.Transform.QUAD, data,
                               resample=Image.Resampling.BICUBIC, fillcolor=fill)  # fmt: skip
        out.paste(piece, (cols[i], 0))
    return out


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
