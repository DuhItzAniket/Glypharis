"""Reading order of detected lines: recursive XY-cut after deskewing (Phase 10).

1. **Deskew.** The page's dominant text angle is the length-weighted median of the lines' reading
   directions (TL->TR); when it is more than `DESKEW_MIN_DEG` from horizontal, every point is
   rotated back by it, so a photographed page that is turned a little reads like a straight one.
2. **XY-cut.** Each line becomes the axis-aligned box of its rotated corners. A group of lines is
   split at its widest empty band: a vertical band (a column gutter) when it is at least
   `COLUMN_GAP` median line heights wide and no narrower than the widest horizontal band, otherwise
   the widest horizontal band (between lines). Splitting at one band at a time, largest first, keeps
   a header that spans two columns above both of them and reads two-column text column by column.
3. **Rows.** A group with no empty band left (lines that overlap) is read row by row: lines whose
   vertical extents overlap by more than half the smaller height share a row; rows top to bottom,
   each left to right.

Only the geometry is used, so the order is a heuristic. HierText has no reading order to measure it
against; the tests pin the layouts the plan names (single column, two columns, mixed blocks, ragged
lines, rotated regions), and GDC, which has one, measures it in Phase 14.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

DESKEW_MIN_DEG = 1.0
COLUMN_GAP = 1.0  # vertical band width, in median line heights, that counts as a column gutter
ROW_OVERLAP = 0.5


def _angle(q: np.ndarray) -> float:
    dx, dy = q[1] - q[0]
    return math.degrees(math.atan2(dy, dx))


def dominant_angle(quads: Sequence[np.ndarray]) -> float:
    """Length-weighted median of the TL->TR directions, in degrees, folded into (-90, 90]."""
    if not quads:
        return 0.0
    angles, weights = [], []
    for q in quads:
        a = _angle(q)
        a = (a + 90.0) % 180.0 - 90.0  # a line read upside down still gives its axis
        angles.append(a)
        weights.append(float(np.linalg.norm(q[1] - q[0])) + 1e-6)
    order = np.argsort(angles)
    cum = np.cumsum(np.asarray(weights)[order])
    k = int(np.searchsorted(cum, cum[-1] / 2))
    return float(np.asarray(angles)[order][k])


def _boxes(quads: Sequence[np.ndarray], angle: float) -> np.ndarray:
    """(n, 4) x0, y0, x1, y1 of each quad rotated by ``-angle`` degrees."""
    t = math.radians(-angle)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    out = np.zeros((len(quads), 4))
    for i, q in enumerate(quads):
        p = q @ rot.T
        out[i] = (*p.min(axis=0), *p.max(axis=0))
    return out


def _widest_gap(lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    """(width, position) of the widest empty band between the intervals [lo, hi]; (0, nan) if
    they leave none."""
    order = np.argsort(lo)
    best, pos, reach = 0.0, math.nan, hi[order[0]]
    for i in order[1:]:
        if lo[i] > reach and lo[i] - reach > best:
            best, pos = float(lo[i] - reach), float((lo[i] + reach) / 2)
        reach = max(reach, hi[i])
    return best, pos


def _rows(boxes: np.ndarray, idx: list[int]) -> list[int]:
    rows: list[list[int]] = []
    for i in sorted(idx, key=lambda k: ((boxes[k, 1] + boxes[k, 3]) / 2, boxes[k, 0])):
        y0, y1 = boxes[i, 1], boxes[i, 3]
        for row in rows:
            r0, r1 = boxes[row, 1].min(), boxes[row, 3].max()
            small = min(y1 - y0, r1 - r0)
            if small > 0 and min(y1, r1) - max(y0, r0) > ROW_OVERLAP * small:
                row.append(i)
                break
        else:
            rows.append([i])
    return [i for row in rows for i in sorted(row, key=lambda k: boxes[k, 0])]


def _xy_cut(boxes: np.ndarray, idx: list[int], line_h: float) -> list[int]:
    if len(idx) <= 1:
        return idx
    b = boxes[idx]
    gx, px = _widest_gap(b[:, 0], b[:, 2])
    gy, py = _widest_gap(b[:, 1], b[:, 3])
    if gx >= COLUMN_GAP * line_h and gx >= gy:
        left = [i for i in idx if boxes[i, 2] <= px]
        right = [i for i in idx if boxes[i, 2] > px]
        return _xy_cut(boxes, left, line_h) + _xy_cut(boxes, right, line_h)
    if gy > 0:
        top = [i for i in idx if boxes[i, 3] <= py]
        bottom = [i for i in idx if boxes[i, 3] > py]
        return _xy_cut(boxes, top, line_h) + _xy_cut(boxes, bottom, line_h)
    if gx > 0:
        left = [i for i in idx if boxes[i, 2] <= px]
        right = [i for i in idx if boxes[i, 2] > px]
        return _xy_cut(boxes, left, line_h) + _xy_cut(boxes, right, line_h)
    return _rows(boxes, idx)


def reading_order(quads: Sequence[Sequence[Sequence[float]]]) -> list[int]:
    """Indices of ``quads`` (TL, TR, BR, BL each, page coordinates) in reading order."""
    qs = [np.asarray(q, dtype=np.float64).reshape(-1, 2)[:4] for q in quads]
    if len(qs) <= 1:
        return list(range(len(qs)))
    angle = dominant_angle(qs)
    boxes = _boxes(qs, angle if abs(angle) > DESKEW_MIN_DEG else 0.0)
    heights = [float(np.linalg.norm(q[3] - q[0])) for q in qs]
    line_h = max(float(np.median(heights)), 1.0)
    return _xy_cut(boxes, list(range(len(qs))), line_h)
