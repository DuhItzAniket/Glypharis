"""Detector post-processing: probability map -> text-line boxes (NumPy, OpenCV and shapely only).

There is no torch here: the same code runs in validation, `TextLineDetector`, the label ceiling,
Phase 9's offline sweeps and Phase 17's exported runtime. Two stages:

- `find_candidates` binarises the map at ``bin_thresh`` and turns each 8-connected component into a
  `Candidate`: its exact mean probability, pixel count, two kernel rectangles and a simplified outer
  contour. It does not depend on the operating point, so validation runs it once per page.
- `boxes_from_candidates` grows each kernel back to a text line for one operating point, maps it to
  original-image coordinates and clips it to the image. ``unclip_mode`` ``dbnet`` uses DBNet's
  distance D = ratio·A/L on the minimum-area rectangle of the contour (the convention its ratio 1.5
  was calibrated with); ``inverse`` exactly inverts the training shrink for a rectangle, applied to
  the area-preserving rectangle. ``box_type`` picks the output shape: the expanded rectangle
  (``quad``) or, for curved lines, the kernel outline offset by the unclip distance (``poly``),
  which on a straight kernel is the quad with round corners.

Coordinates are edge coordinates: pixel (i, j) covers [j, j+1) x [i, i+1), so its centre is
(j + ½, i + ½). Angles are in degrees with y pointing down, folded to (−90°, 90°]. They are computed
from edge vectors, never read from `cv2.minAreaRect`, whose angle convention changed in OpenCV 4.5.

Text direction (`order_quad`): a predicted box follows its long axis only when long/short is at
least `DIRECTION_MIN_ASPECT`; otherwise the edge nearest horizontal is the text direction, so an
upright short word (taller than wide) is not turned by 90°. Callers that have a ground-truth
polygon pass its TL→TR direction instead (`text_angle`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from glypharis.config.schema import DetOperatingPoint, DetPostConfig
from glypharis.eval.page_metrics import PredLine
from glypharis.models.det.preprocess import ResizeInfo, to_page_coords

DIRECTION_MIN_ASPECT: Final = 2.0  # predicted boxes follow the long axis only at long/short >= 2
CANDIDATE_CONTOUR_EPS_PX: Final = 1.0  # approxPolyDP tolerance of the stored contour, network px
POLY_QUAD_SEGS: Final = 2  # round-join segments per quarter circle in the poly offset
# Share of contour_rect that the widened kernel outline must cover for a poly box to treat the
# kernel as straight. Straight kernels measured at least 0.984 (thin ones lose corner bits at their
# ends); arcs with a sagitta of a few pixels fall below.
POLY_STRAIGHT_COVER: Final = 0.98
# Moment anisotropy (λ1 − λ2)/(λ1 + λ2) below which a component has no usable principal axis.
# A rectangle of aspect a gives (a² − 1)/(a² + 1): 0.2 is reached at a ≈ 1.22.
ISOTROPIC_MOMENTS: Final = 0.2

Rect = tuple[float, float, float, float, float]  # cx, cy, w, h, angle of the w side (degrees)


@dataclass(frozen=True)
class Candidate:
    """One connected component of the binarised map, in network edge coordinates."""

    score: float  # exact mean probability over the component
    area_px: int
    contour_rect: Rect  # cx, cy, w, h, angle: minimum-area rectangle of the contour pixel centres
    area_rect: Rect  # cx, cy, length, width, angle; width = area_px / length
    contour: tuple[tuple[float, float], ...]  # simplified outer contour of pixel centres

    def to_dict(self) -> dict[str, Any]:
        """Plain data for the saved-candidates files (tuples become JSON lists)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Candidate:
        return cls(
            score=float(d["score"]),
            area_px=int(d["area_px"]),
            contour_rect=_rect5(d["contour_rect"]),
            area_rect=_rect5(d["area_rect"]),
            contour=tuple((float(x), float(y)) for x, y in d["contour"]),
        )


@dataclass(frozen=True)
class DetBox:
    """One detected line in original-image edge coordinates."""

    polygon: list[list[float]]  # clipped to the image, K >= 4: the rectangle or the offset contour
    quad: list[list[float]]  # TL, TR, BR, BL of the polygon's minimum-area rectangle (Phase 10)
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    score: float
    aspect: float  # long / short side of the quad's rectangle
    long_axis_angle: float  # in (-90, 90], y down; Phase 10 may try it when aspect is low

    def to_pred_line(self) -> PredLine:
        return PredLine(self.polygon, "", self.score)


@dataclass
class PostCounters:
    """Where components went. ``components = tiny + truncated + low_score + too_thin + outside +
    kept`` once the counters of both stages are added; ``poly_fallback`` is a subset of ``kept``."""

    components: int = 0
    tiny: int = 0
    truncated: int = 0
    low_score: int = 0
    too_thin: int = 0
    outside: int = 0
    kept: int = 0
    poly_fallback: int = 0  # kept boxes that fell back to the rectangle in poly mode

    def add(self, other: PostCounters) -> None:
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


# ------------------------------------------------------------------------------------ geometry


def _rect5(values: Sequence[Any]) -> Rect:
    cx, cy, w, h, angle = (float(v) for v in values)
    return cx, cy, w, h, angle


def _fold(angle: float) -> float:
    """An undirected angle in degrees folded to (-90, 90]."""
    a = (angle + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


def _edge_angle(dx: float, dy: float) -> float:
    return _fold(math.degrees(math.atan2(dy, dx)))


def _angle_gap(a: float, b: float) -> float:
    """Distance between two undirected angles, in [0, 90]."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _rect_from_points(points: np.ndarray) -> Rect:
    """Minimum-area rectangle of points as (cx, cy, w, h, angle of the w side)."""
    r = cv2.minAreaRect(np.asarray(points, dtype=np.float32).reshape(-1, 1, 2))
    box = cv2.boxPoints(r).astype(np.float64)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    c = box.mean(axis=0)
    return (
        float(c[0]),
        float(c[1]),
        float(math.hypot(e1[0], e1[1])),
        float(math.hypot(e2[0], e2[1])),
        _edge_angle(float(e1[0]), float(e1[1])),
    )


def _rect_corners(rect: Rect) -> np.ndarray:
    cx, cy, w, h, angle = rect
    t = math.radians(angle)
    u = np.array([math.cos(t), math.sin(t)])
    v = np.array([-u[1], u[0]])
    c = np.array([cx, cy])
    return np.array(
        [c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2,
         c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2]
    )  # fmt: skip


def _area_rect(xs: np.ndarray, ys: np.ndarray, fallback_angle: float) -> Rect:
    """Area-preserving rectangle of a component from its pixel centres.

    The axis comes from the second central moments, the length is the projected extent plus one
    pixel and the width is pixel count / length, so the rectangle has the component's area. Unlike
    a minimum-area rectangle around pixel centres or corners, this is unbiased at any angle.

    A square has no principal axis (both moments are equal at every angle), so for a nearly
    isotropic component the axis is ``fallback_angle``, the contour rectangle's angle.
    """
    mx, my = float(xs.mean()), float(ys.mean())
    dx, dy = xs - mx, ys - my
    mu20, mu02, mu11 = float((dx * dx).mean()), float((dy * dy).mean()), float((dx * dy).mean())
    spread = mu20 + mu02
    if spread > 0 and math.hypot(mu20 - mu02, 2.0 * mu11) / spread < ISOTROPIC_MOMENTS:
        theta = math.radians(fallback_angle)
    else:
        theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    ux, uy = math.cos(theta), math.sin(theta)
    proj = dx * ux + dy * uy
    lo, hi = float(proj.min()), float(proj.max())
    length = hi - lo + 1.0
    mid = (lo + hi) / 2.0
    return mx + ux * mid, my + uy * mid, length, xs.size / length, _fold(math.degrees(theta))


def text_angle(quad: Sequence[Sequence[float]] | np.ndarray) -> float:
    """Text direction of a TL, TR, BR, BL quad: the angle of (TR - TL) + (BR - BL) in (-90, 90].

    Raises ValueError for anything but four points, or when the two edges cancel out.
    """
    q = np.asarray(quad, dtype=np.float64)
    if q.shape != (4, 2):
        raise ValueError(f"text_angle needs a 4-point quad, got shape {q.shape}")
    v = (q[1] - q[0]) + (q[2] - q[3])
    if not np.any(v):
        raise ValueError("text_angle: the quad has no text direction")
    return _edge_angle(float(v[0]), float(v[1]))


def order_quad(rect: np.ndarray, direction_deg: float | None = None) -> np.ndarray:
    """Corners of a rectangle as TL, TR, BR, BL for its text direction u.

    With ``direction_deg`` (a ground-truth TL→TR direction) u is the rectangle edge nearest that
    direction. Without it u is the long edge when long/short >= `DIRECTION_MIN_ASPECT`, and
    otherwise the edge nearest horizontal; ties go to the long edge. u points right (or down when
    vertical), v is u turned 90° clockwise on screen, and TL = c - u·w/2 - v·h/2 and so on.
    """
    r = np.asarray(rect, dtype=np.float64).reshape(4, 2)
    e1, e2 = r[1] - r[0], r[2] - r[1]
    l1, l2 = float(math.hypot(e1[0], e1[1])), float(math.hypot(e2[0], e2[1]))
    a1, a2 = _edge_angle(float(e1[0]), float(e1[1])), _edge_angle(float(e2[0]), float(e2[1]))
    first_is_long = l1 >= l2
    if direction_deg is not None:
        g1, g2 = _angle_gap(a1, direction_deg), _angle_gap(a2, direction_deg)
        take_first = g1 < g2 or (g1 == g2 and first_is_long)
    elif min(l1, l2) > 0 and max(l1, l2) / min(l1, l2) < DIRECTION_MIN_ASPECT:
        g1, g2 = abs(a1), abs(a2)  # gap to horizontal
        take_first = g1 < g2 or (g1 == g2 and first_is_long)
    else:
        take_first = first_is_long
    angle, w, h = (a1, l1, l2) if take_first else (a2, l2, l1)
    c = r.mean(axis=0)
    return _rect_corners((float(c[0]), float(c[1]), w, h, angle))


# --------------------------------------------------------------------------------- candidates


def find_candidates(
    prob: np.ndarray, info: ResizeInfo, post: DetPostConfig
) -> tuple[list[Candidate], PostCounters]:
    """Components of ``prob > bin_thresh`` inside the valid region, independent of the operating
    point. ``prob`` is the (pad_h, pad_w) map; only ``[:h, :w]`` is read, so the padding never
    produces candidates."""
    p = np.asarray(prob, dtype=np.float32)[: info.h, : info.w]
    counters = PostCounters()
    bitmap = (p > post.bin_thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bitmap, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    counters.components = n - 1
    ids = np.flatnonzero(areas[1:] >= post.min_component_area) + 1  # label 0 is background
    counters.tiny = counters.components - ids.size
    if ids.size > post.max_candidates:
        # A runtime guard for collapsed maps: keep the largest, ties broken by label index.
        order = np.lexsort((ids, -areas[ids]))
        counters.truncated = ids.size - post.max_candidates
        ids = np.sort(ids[order[: post.max_candidates]])
    if ids.size == 0:
        return [], counters
    sums = np.bincount(labels.ravel(), weights=p.ravel().astype(np.float64), minlength=n)
    cands = [_candidate(labels, int(k), stats[k], float(sums[k] / areas[k])) for k in ids]
    return cands, counters


def _candidate(labels: np.ndarray, k: int, stat: np.ndarray, score: float) -> Candidate:
    x0, y0, w, h, area = (int(v) for v in stat[:5])
    mask = (labels[y0 : y0 + h, x0 : x0 + w] == k).astype(np.uint8)
    # A zero border keeps contour tracing independent of how OpenCV treats the image edge.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # One 8-connected component has one outer contour; the longest is taken defensively.
    pts = max(contours, key=len).reshape(-1, 2).astype(np.float64)
    pts += (x0 - 1 + 0.5, y0 - 1 + 0.5)  # undo the border, then pixel index -> pixel centre
    simplified = cv2.approxPolyDP(
        pts.astype(np.float32).reshape(-1, 1, 2), CANDIDATE_CONTOUR_EPS_PX, closed=True
    ).reshape(-1, 2)
    ys, xs = np.nonzero(mask)
    contour_rect = _rect_from_points(pts)
    return Candidate(
        score=score,
        area_px=area,
        contour_rect=contour_rect,
        area_rect=_area_rect(xs + (x0 + 0.5), ys + (y0 + 0.5), contour_rect[4]),
        contour=tuple((float(x), float(y)) for x, y in simplified),
    )


# -------------------------------------------------------------------------------------- boxes


def unclip_distance(
    mode: Literal["dbnet", "inverse"],
    area: float,
    perimeter: float,
    ratio: float,
    shrink_ratio: float,
) -> float:
    """Offset distance that grows a kernel of this area and perimeter back to a text line.

    ``dbnet``: ratio·A/L. ``inverse``: ratio times the positive root of
    4(1 + r²)D² + r²·L·D − (1 − r²)·A = 0, which exactly inverts the training shrink
    D = A₀(1 − r²)/L₀ for a rectangle (A₀ = A + L·D + 4D², L₀ = L + 8D). For other shapes the mitre
    corner term differs and the inverse is approximate. A rectangle with sides a, b passes
    area = a·b and perimeter = 2(a + b).
    """
    if area <= 0 or perimeter <= 0:
        return 0.0
    if mode == "dbnet":
        return ratio * area / perimeter
    r2 = shrink_ratio * shrink_ratio
    root = math.sqrt(r2 * r2 * perimeter * perimeter + 16.0 * (1 + r2) * (1 - r2) * area)
    return ratio * (root - r2 * perimeter) / (8.0 * (1 + r2))


def _kernel_rect(c: Candidate, mode: Literal["dbnet", "inverse"]) -> Rect:
    """The kernel rectangle a quad grows: `contour_rect` for dbnet, `area_rect` for inverse."""
    return c.contour_rect if mode == "dbnet" else c.area_rect


def _quad_output(c: Candidate, op: DetOperatingPoint, shrink_ratio: float) -> Rect:
    cx, cy, a, b, angle = _kernel_rect(c, op.unclip_mode)
    d = unclip_distance(op.unclip_mode, a * b, 2 * (a + b), op.unclip_ratio, shrink_ratio)
    return cx, cy, a + 2 * d, b + 2 * d, angle


def _largest_polygon(g: Any) -> Polygon | None:
    """``g`` if it is one valid polygon, else its largest valid polygon part, else None."""
    parts = [g] if isinstance(g, Polygon) else list(getattr(g, "geoms", []))
    parts = [p for p in parts if isinstance(p, Polygon) and not p.is_empty and p.is_valid]
    return max(parts, key=lambda p: p.area) if parts else None


def _offset_to_area(k: Polygon | None, area: float) -> Polygon | None:
    """``k`` offset with mitre joins by (area − A)/L, the first-order distance to ``area``."""
    if k is None or k.length <= 0:
        return None
    d = (area - k.area) / k.length
    if abs(d) < 1e-6:
        return k  # GEOS returns an empty polygon for offsets of about 1e-15
    return _largest_polygon(k.buffer(d, join_style="mitre"))


def _poly_output(c: Candidate, op: DetOperatingPoint, shrink_ratio: float) -> Polygon | None:
    """The kernel outline K offset by the unclip distance, or None when the quad must be used.

    The contour is first grown to the component's pixel area, the pixel-boundary outline that
    `area_rect` measures. A fixed half-pixel offset does that only for an unsimplified
    axis-aligned staircase, and on a slanted simplified contour it over-counts (by 11% on a 6 px
    kernel at 30°), so the offset is solved from the area to first order. Widened by the contour's
    own tolerance and cut by `contour_rect`, which bounds every pixel centre, it gives the band.

    - **Straight kernel:** the band covers at least `POLY_STRAIGHT_COVER` of the rectangle, so
      the kernel is straight to the contour's precision, and K is the quad's own kernel rectangle:
      the box is the quad with round corners. The contour cannot stand in for the rectangle.
      Simplified to 1 px, a 3 px kernel lies a few tenths of a pixel off it, and dbnet's centre
      contour runs up to a pixel thinner, so dbnet15 poly boxes on slightly slanted thin kernels
      fell to IoU 0.62 with their quads.
    - **Curved kernel:** the rectangle is a loose bound that only trims the band's edges. K is the
      band brought to the quad's kernel area, min(rectangle area, pixel area), which takes the
      widening off again, so a curved band gets the distance of a straight band of its own length
      and width.
    """
    if len(c.contour) < 4:
        return None
    k = Polygon(c.contour)
    if k.is_empty or not k.is_valid:
        return None
    outline = _offset_to_area(k, c.area_px)
    if outline is None:
        return None
    bound = Polygon(_rect_corners(c.contour_rect))
    widened = outline.buffer(CANDIDATE_CONTOUR_EPS_PX, join_style="mitre")
    band = _largest_polygon(bound.intersection(widened))
    rect = _kernel_rect(c, op.unclip_mode)
    if band is not None and band.area >= POLY_STRAIGHT_COVER * bound.area:
        k = Polygon(_rect_corners(rect))
    else:
        k = _offset_to_area(band, min(rect[2] * rect[3], c.area_px))
    if k is None:
        return None
    d = unclip_distance(op.unclip_mode, k.area, k.length, op.unclip_ratio, shrink_ratio)
    out = k.buffer(d, join_style="round", quad_segs=POLY_QUAD_SEGS)
    if not isinstance(out, Polygon) or out.is_empty:
        return None
    return Polygon(out.exterior)  # an offset that closes a gap must not leave a hole


def _clip_to_image(points: np.ndarray, image: Polygon) -> Polygon | None:
    """Exact intersection with the image; a shape cut into pieces keeps its largest piece."""
    p = Polygon(points)
    if p.is_empty or p.area <= 0:
        return None
    if not p.is_valid:
        p = shapely.make_valid(p)
    g = shapely.intersection(p, image)
    if not isinstance(g, Polygon):
        parts = [x for x in getattr(g, "geoms", []) if isinstance(x, Polygon) and not x.is_empty]
        if not parts:
            return None
        g = max(parts, key=lambda x: x.area)
    if g.is_empty or g.area <= 0:
        return None
    return Polygon(g.exterior)


def _make_box(poly: Polygon, score: float) -> DetBox:
    rect = _rect_from_points(np.asarray(poly.exterior.coords))
    quad = order_quad(_rect_corners(rect))
    _, _, w, h, angle = rect
    long_side, short_side = max(w, h), min(w, h)
    long_angle = angle if w >= h else _fold(angle + 90.0)
    # Ring in TL→TR→BR→BL winding (clockwise on screen, positive shoelace area with y down),
    # starting at the vertex nearest TL. For readable output only; the evaluator ignores it.
    ring = np.asarray(orient(poly, sign=1.0).exterior.coords, dtype=np.float64)[:-1]
    start = int(np.argmin(np.hypot(*(ring - quad[0]).T)))
    ring = np.roll(ring, -start, axis=0)
    while len(ring) < 4:  # a clipped triangle: split its longest edge so K >= 4
        nxt = np.roll(ring, -1, axis=0)
        i = int(np.argmax(np.hypot(*(nxt - ring).T)))
        ring = np.insert(ring, i + 1, (ring[i] + nxt[i]) / 2, axis=0)
    x0, y0, x1, y1 = poly.bounds
    return DetBox(
        polygon=ring.tolist(),
        quad=quad.tolist(),
        bbox=(float(x0), float(y0), float(x1), float(y1)),
        score=score,
        aspect=long_side / short_side if short_side > 0 else math.inf,
        long_axis_angle=long_angle,
    )


def boxes_from_candidates(
    cands: Sequence[Candidate],
    op: DetOperatingPoint,
    post: DetPostConfig,
    shrink_ratio: float,
    info: ResizeInfo,
) -> tuple[list[DetBox], PostCounters]:
    """Boxes for one operating point, sorted by (min y, min x), in original-image coordinates.

    The counters cover this stage only (``low_score``, ``too_thin``, ``outside``, ``kept``,
    ``poly_fallback``); add them to `find_candidates`' counters for the full identity.
    ``shrink_ratio`` is ``det.labels.shrink_ratio``, so labels and unclip share one r.
    """
    counters = PostCounters()
    image = shapely.box(0.0, 0.0, float(info.orig_w), float(info.orig_h))
    boxes: list[DetBox] = []
    for c in cands:
        if c.score < post.box_thresh:
            counters.low_score += 1
            continue
        offset = _poly_output(c, op, shrink_ratio) if post.box_type == "poly" else None
        fell_back = post.box_type == "poly" and offset is None
        if offset is None:
            rect = _quad_output(c, op, shrink_ratio)
            short_side = min(rect[2], rect[3])
            points = _rect_corners(rect)
        else:
            points = np.asarray(offset.exterior.coords, dtype=np.float64)[:-1]
            short_side = min(_rect_from_points(points)[2:4])
        if short_side < post.min_box_side:
            counters.too_thin += 1
            continue
        poly = _clip_to_image(to_page_coords(points, info), image)
        if poly is None:
            counters.outside += 1
            continue
        boxes.append(_make_box(poly, c.score))
        counters.kept += 1
        counters.poly_fallback += int(fell_back)
    boxes.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return boxes, counters
