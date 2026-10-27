"""DBNet training targets for one canvas: shrink map and mask, threshold map and mask (§4.1, §4.4).

The targets follow DBNet (Liao et al. 2020) exactly:

- **shrink map**: each text polygon is offset inwards by D = A(1 − r²)/L. A mitre inward offset
  of a convex polygon is the Vatti offset the reference code computes with pyclipper, so shapely
  does it here.
- **threshold map**: inside the polygon dilated by D (round joins), 1 − min(d/D, 1) with d the
  exact distance to the polygon's edges, mapped linearly to [thresh_min, thresh_max]; the maximum
  wins where lines overlap.
- **masks**: the shrink mask is 0 on don't-care lines and ignore regions and 1 elsewhere, the
  threshold mask is the dilated text polygons minus ignore regions.

Pixels are set by exact centre sampling: pixel (i, j) belongs to a shape when its centre
(j + ½, i + ½) is strictly inside it, so the expected pixel count equals the shape's area. Text
lines that are too small for the scale, or whose kernel is under a pixel thick or holds no pixel
centre, become don't care for that sample, so every line that stays text has kernel pixels.

Care wins: a text kernel stays mask 1 inside a don't-care polygon, because the evaluator matches
text lines before don't-care lines and a kernel there must remain a positive target.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Final, cast

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from glypharis.config.schema import DetLabelConfig
from glypharis.data.det_geometry import min_area_rect
from glypharis.data.pages import PageLine
from glypharis.eval.page_metrics import _poly, has_trusted_box, is_box_line

LABEL_VERSION: Final = 1

# Why a ground-truth line is or is not a text target, in the order the reasons are checked.
CARE, ILLEGIBLE, HANDWRITTEN, VERTICAL, EMPTY_TEXT, UNTRUSTED_BOX = range(6)

# A kernel whose minimum-area rectangle is thinner than this is too thin to learn.
MIN_KERNEL_THICKNESS = 1.0


def line_reason(line: PageLine) -> int:
    """CARE for lines the detector is scored on (`is_box_line`), else the first reason it is not."""
    if is_box_line(line):
        return CARE
    if not line.legible:
        return ILLEGIBLE
    if line.handwritten:
        return HANDWRITTEN
    if line.vertical:
        return VERTICAL
    if not line.text.strip():
        return EMPTY_TEXT
    if not has_trusted_box(line):
        return UNTRUSTED_BOX
    raise AssertionError(f"line {line.line_id} is neither a box line nor has a reason")


@dataclass
class LabelStats:
    """What one or more `make_labels` calls did, summed with `add`.

    Lines whose dilated box misses the canvas are left out of every count. ``dont_care`` counts the
    lines that were not text to begin with; demoted lines are counted only by their own counter.
    """

    care_drawn: int = 0  # text lines with a kernel target
    demoted_small: int = 0  # rotated short side below min_text_size
    demoted_thin_kernel: int = 0  # kernel empty, under a pixel thick, or without a pixel centre
    multipart_kernel: int = 0  # drawn kernels that split into several parts
    dont_care: int = 0
    kernel_px: int = 0  # shrink-map pixels
    masked_px: int = 0  # shrink-mask zeros

    def add(self, other: LabelStats) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


@dataclass
class DetLabels:
    shrink_map: np.ndarray  # (H, W) uint8 {0, 1}
    shrink_mask: np.ndarray  # (H, W) uint8 {0, 1}
    thresh_map: np.ndarray  # (H, W) float32 T* in [thresh_min, thresh_max]
    thresh_mask: np.ndarray  # (H, W) uint8 {0, 1}
    care_kept: np.ndarray  # (n_polys,) bool: a text target here (care, not demoted, on the canvas)
    stats: LabelStats


def shrink_distance(area: float, perimeter: float, r: float) -> float:
    """DBNet's offset D = A(1 − r²)/L (0 for a degenerate polygon)."""
    return area * (1.0 - r * r) / perimeter if perimeter > 0 else 0.0


def _points(poly: np.ndarray) -> Sequence[Sequence[float]]:
    # `_poly` only needs len() and Polygon(), which an (N, 2) array supports.
    return cast(Sequence[Sequence[float]], poly)


def _centre_grid(x0: int, y0: int, x1: int, y1: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre coordinates of the box [x0, x1) x [y0, y1) as a (1, W) row and an (H, 1)
    column, which broadcast to (H, W)."""
    return (np.arange(x0, x1) + 0.5)[None, :], (np.arange(y0, y1) + 0.5)[:, None]


def _centre_hits(geom: BaseGeometry, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """(y1 − y0, x1 − x0) bool: pixel centres of the box [x0, x1) x [y0, y1) strictly inside."""
    hits: np.ndarray = shapely.contains_xy(geom, *_centre_grid(x0, y0, x1, y1))
    return hits


def rasterize(canvas: np.ndarray, geom: BaseGeometry, value: int) -> int:
    """Set every canvas pixel whose centre is strictly inside ``geom`` to ``value``.

    Returns the number of pixels set. Only the geometry's bounding box, clipped to the canvas, is
    sampled.
    """
    if geom.is_empty:
        return 0
    h, w = canvas.shape[:2]
    gx0, gy0, gx1, gy1 = geom.bounds
    x0, y0 = max(math.floor(gx0), 0), max(math.floor(gy0), 0)
    x1, y1 = min(math.ceil(gx1), w), min(math.ceil(gy1), h)
    if x1 <= x0 or y1 <= y0:
        return 0
    shapely.prepare(geom)
    hits = _centre_hits(geom, x0, y0, x1, y1)
    canvas[y0:y1, x0:x1][hits] = value
    return int(hits.sum())


def count_centres(geom: BaseGeometry) -> int:
    """Pixel centres strictly inside ``geom``, over its own bounding box and never clipped.

    The count depends only on the geometry, so it is the same for every crop of a sample (crop
    origins are whole pixels).
    """
    if geom.is_empty:
        return 0
    gx0, gy0, gx1, gy1 = geom.bounds
    x0, y0, x1, y1 = math.floor(gx0), math.floor(gy0), math.ceil(gx1), math.ceil(gy1)
    if x1 <= x0 or y1 <= y0:
        return 0
    shapely.prepare(geom)
    return int(_centre_hits(geom, x0, y0, x1, y1).sum())


def segment_distance(xs: np.ndarray, ys: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance (float64) from each point to the nearest segment of ``ring``.

    ``ring`` is (M, 2); it is closed here if its last point differs from its first.
    """
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    r = np.asarray(ring, dtype=np.float64).reshape(-1, 2)
    best = np.full(np.broadcast(x, y).shape, np.inf)
    if len(r) == 0:
        return best
    if not np.array_equal(r[0], r[-1]) or len(r) == 1:
        r = np.vstack([r, r[:1]])
    for (ax, ay), (bx, by) in itertools.pairwise(r):
        dx, dy = bx - ax, by - ay
        px, py = x - ax, y - ay
        ll = dx * dx + dy * dy
        t = np.clip((px * dx + py * dy) / ll, 0.0, 1.0) if ll > 0 else 0.0
        ex, ey = px - t * dx, py - t * dy
        np.minimum(best, ex * ex + ey * ey, out=best)
    out: np.ndarray = np.sqrt(best)
    return out


def _draw_ramp(canvas: np.ndarray, thresh_mask: np.ndarray, poly: BaseGeometry, d: float) -> None:
    """Add one text line's threshold ramp, 1 − min(dist/D, 1) over the polygon dilated by D."""
    h, w = canvas.shape
    gx0, gy0, gx1, gy1 = poly.bounds
    x0, y0 = max(math.floor(gx0 - d), 0), max(math.floor(gy0 - d), 0)
    x1, y1 = min(math.ceil(gx1 + d), w), min(math.ceil(gy1 + d), h)
    if x1 <= x0 or y1 <= y0:
        return
    cx, cy = _centre_grid(x0, y0, x1, y1)
    dist = segment_distance(cx, cy, np.asarray(poly.exterior.coords))
    # Within D of the boundary, or inside the polygon: exactly the round dilation by D. Only the
    # centres farther than D need the (slower) containment test.
    region = dist <= d
    fy, fx = np.nonzero(~region)
    if len(fy):
        region[fy, fx] = shapely.contains_xy(poly, cx[0, fx], cy[fy, 0])
    ramp = np.where(region, 1.0 - np.minimum(dist / d, 1.0), 0.0)
    sub = canvas[y0:y1, x0:x1]
    np.maximum(sub, ramp, out=sub, casting="same_kind")
    thresh_mask[y0:y1, x0:x1] |= region.astype(np.uint8)


def make_labels(
    polys: Sequence[np.ndarray],
    care: np.ndarray,
    ignore: Sequence[np.ndarray],
    canvas_hw: tuple[int, int],
    cfg: DetLabelConfig,
) -> DetLabels:
    """DBNet targets for (K, 2) polygons in canvas edge coordinates (§4.4).

    ``care`` marks the lines the detector is scored on (`line_reason` == CARE); every other line is
    don't care, and so is every ``ignore`` region. Polygons are never clipped: a line cut by the
    canvas edge stays text up to the edge. Lines whose box, grown by D, misses the canvas cannot
    touch it and are skipped. Polygons go through the evaluator's own validity repair first.
    """
    h, w = canvas_hw
    r = cfg.shrink_ratio
    n = len(polys)
    care = np.asarray(care, dtype=bool).reshape(-1)
    if len(care) != n:
        raise ValueError(f"{n} polygons but {len(care)} care flags")
    shrink_map = np.zeros((h, w), np.uint8)
    shrink_mask = np.ones((h, w), np.uint8)
    canvas = np.zeros((h, w), np.float32)
    thresh_mask = np.zeros((h, w), np.uint8)
    care_kept = np.zeros(n, dtype=bool)
    stats = LabelStats()
    kept: list[tuple[int, BaseGeometry]] = []

    geoms = np.empty(n, dtype=object)
    geoms[:] = [_poly(_points(p)) for p in polys]
    dist = np.array(
        [
            shrink_distance(a, length, r)
            for a, length in zip(shapely.area(geoms), shapely.length(geoms), strict=True)
        ]
    )
    b = shapely.bounds(geoms)  # NaN for empty polygons, which then fail every test below
    on_canvas = (b[:, 2] + dist > 0) & (b[:, 3] + dist > 0)
    on_canvas &= (b[:, 0] - dist < w) & (b[:, 1] - dist < h)

    small = np.zeros(n, dtype=bool)
    thin = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(care & on_canvas):
        short = min_area_rect(shapely.get_coordinates(geoms[i]))[1]
        small[i] = short < cfg.min_text_size
    idx = np.flatnonzero(care & on_canvas & ~small)
    kernels = shapely.buffer(geoms[idx], -dist[idx], join_style="mitre")
    for i, k in zip(idx, kernels, strict=True):
        # Thickness alone does not guarantee a pixel centre (a thin rotated square, a 1 px
        # strip with its edges on centre rows), so the centre count of the whole kernel decides.
        if (
            k.is_empty
            or min_area_rect(shapely.get_coordinates(k))[1] < MIN_KERNEL_THICKNESS
            or count_centres(k) == 0
        ):
            thin[i] = True
        else:
            kept.append((int(i), k))

    dont = on_canvas & ~care
    for i in np.flatnonzero(dont | small | thin):
        rasterize(shrink_mask, geoms[i], 0)
    stats.demoted_small = int(small.sum())
    stats.demoted_thin_kernel = int(thin.sum())
    stats.dont_care = int(dont.sum())

    ignore_geoms = [_poly(_points(p)) for p in ignore]
    for g in ignore_geoms:
        rasterize(shrink_mask, g, 0)
    for j, k in kept:
        rasterize(shrink_mask, k, 1)  # care wins over don't care
        rasterize(shrink_map, k, 1)
        _draw_ramp(canvas, thresh_mask, geoms[j], float(dist[j]))
        care_kept[j] = True
        stats.multipart_kernel += int(k.geom_type == "MultiPolygon")
    for g in ignore_geoms:
        rasterize(thresh_mask, g, 0)

    stats.care_drawn = len(kept)
    stats.kernel_px = int(shrink_map.sum(dtype=np.int64))
    stats.masked_px = int(h * w - shrink_mask.sum(dtype=np.int64))
    thresh_map = (cfg.thresh_min + (cfg.thresh_max - cfg.thresh_min) * canvas).astype(np.float32)
    return DetLabels(shrink_map, shrink_mask, thresh_map, thresh_mask, care_kept, stats)
