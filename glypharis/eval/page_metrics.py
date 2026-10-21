"""Detection and end-to-end page metrics (EXPERIMENT_PROTOCOL §3).

Ground truth is a `glypharis.data.pages.Page`. Which ground-truth lines count:

- **text lines** (order-aware page text): legible, not handwritten, not vertical. Handwriting and
  rotated text are out of scope for an English printed-text engine (ADR-0008).
- **box lines** (detection, bag-of-lines, E2E F1): text lines whose box is trusted, i.e. no GDC
  alignment score or a score of at least `MIN_ALIGN_SCORE`.
- every other ground-truth line is **don't care**: a prediction matched to it is ignored, and it is
  never a miss. Predictions lying mostly inside a page ignore region or a don't-care line are
  ignored too.

Matching is one-to-one by polygon IoU (Hungarian assignment); a pair needs IoU >= threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from glypharis.data.pages import Page, PageLine
from glypharis.data.text import normalize_eval
from glypharis.eval.text_metrics import Track, char_distance

MIN_ALIGN_SCORE = 0.3
IGNORE_OVERLAP = 0.5  # share of a prediction's area inside don't-care regions to ignore it
E2E_MAX_LINE_CER = 0.1


@dataclass(frozen=True)
class PredLine:
    polygon: list[list[float]]
    text: str = ""
    confidence: float | None = None


def is_text_line(line: PageLine) -> bool:
    return line.legible and not line.handwritten and not line.vertical and bool(line.text.strip())


def is_box_line(line: PageLine) -> bool:
    return is_text_line(line) and (line.align_score is None or line.align_score >= MIN_ALIGN_SCORE)


def _poly(points: Sequence[Sequence[float]]) -> Polygon:
    p = Polygon(points)
    if not p.is_valid:
        p = shapely.make_valid(p)
        if not isinstance(p, Polygon):  # keep the largest piece of a repaired shape
            parts = [g for g in getattr(p, "geoms", []) if isinstance(g, Polygon)]
            p = max(parts, key=lambda g: g.area) if parts else Polygon()
    return p


def iou_matrix(preds: list[Polygon], gts: list[Polygon]) -> np.ndarray:
    """IoU for every prediction x ground-truth pair (sparse via an STR-tree, dense result)."""
    m = np.zeros((len(preds), len(gts)), dtype=np.float64)
    if not preds or not gts:
        return m
    tree = shapely.STRtree(gts)
    pi, gi = tree.query(preds, predicate="intersects")
    if len(pi):
        a = np.array(preds, dtype=object)[pi]
        b = np.array(gts, dtype=object)[gi]
        inter = shapely.area(shapely.intersection(a, b))
        union = shapely.area(a) + shapely.area(b) - inter
        m[pi, gi] = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return m


@dataclass
class PageResult:
    doc_id: str
    # detection at each IoU threshold: (tp, fp, fn)
    det: dict[float, tuple[int, int, int]] = field(default_factory=dict)
    matched_iou_sum: float = 0.0
    # order-aware page text
    aware_errors: int = 0
    aware_ref_chars: int = 0
    # order-agnostic bag of lines (matching at IoU 0.5)
    bag_errors: int = 0
    bag_ref_chars: int = 0
    # end-to-end line F1 (matched at IoU 0.5 and line CER <= E2E_MAX_LINE_CER)
    e2e_tp: int = 0
    e2e_pred: int = 0
    e2e_gt: int = 0


def evaluate_page(
    gt: Page,
    preds: Sequence[PredLine],
    thresholds: Sequence[float] = (0.5, 0.7),
    track: Track = "strict",
    score_text: bool = True,
) -> PageResult:
    res = PageResult(gt.doc_id)
    box_lines = [ln for ln in gt.lines if is_box_line(ln)]
    care_ids = {ln.line_id for ln in box_lines}
    dont = [ln for ln in gt.lines if ln.line_id not in care_ids]
    gt_lines = box_lines + dont
    gt_polys = [_poly(ln.polygon) for ln in gt_lines]
    n_care = len(box_lines)
    pred_polys = [_poly(p.polygon) for p in preds]

    ignore_area = shapely.union_all([_poly(p) for p in gt.ignore] + gt_polys[n_care:])
    pred_area = np.array([p.area for p in pred_polys])
    if pred_polys and not ignore_area.is_empty:
        inside = shapely.area(shapely.intersection(np.array(pred_polys, dtype=object), ignore_area))
        mostly_ignored = inside >= IGNORE_OVERLAP * np.maximum(pred_area, 1e-9)
    else:
        mostly_ignored = np.zeros(len(pred_polys), dtype=bool)

    iou = iou_matrix(pred_polys, gt_polys)
    match_05: dict[int, int] = {}
    for thr in thresholds:
        tp = fp = 0
        matched_gt: set[int] = set()
        matched_pred: dict[int, int] = {}
        if iou.size:
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols, strict=True):
                if iou[r, c] >= thr:
                    matched_pred[int(r)] = int(c)
        for p in range(len(preds)):
            g = matched_pred.get(p)
            if g is not None and g < n_care:
                tp += 1
                matched_gt.add(g)
                if thr == thresholds[0]:
                    res.matched_iou_sum += float(iou[p, g])
            elif g is not None or mostly_ignored[p]:
                continue  # matched a don't-care line, or lies in an ignore region
            else:
                fp += 1
        res.det[thr] = (tp, fp, n_care - len(matched_gt))
        if thr == 0.5:
            match_05 = {p: g for p, g in matched_pred.items()}

    if not score_text:
        return res

    # order-aware page text: every text line in ground-truth order vs predictions in their order
    ref = normalize_eval("\n".join(ln.text for ln in gt.lines if is_text_line(ln)), track)
    hyp = normalize_eval("\n".join(p.text for p in preds), track)
    res.aware_errors, res.aware_ref_chars = char_distance(hyp, ref), len(ref)

    # bag of lines and E2E F1 on trusted boxes
    used_gt: set[int] = set()
    for p, pred in enumerate(preds):
        g = match_05.get(p)
        ptext = normalize_eval(pred.text, track)
        if g is not None and g < n_care:
            rtext = normalize_eval(gt_lines[g].text, track)
            d = char_distance(ptext, rtext)
            res.bag_errors += d
            used_gt.add(g)
            res.e2e_pred += 1
            if rtext and d / len(rtext) <= E2E_MAX_LINE_CER:
                res.e2e_tp += 1
        elif g is not None or mostly_ignored[p]:
            continue
        else:
            res.bag_errors += len(ptext)  # a spurious line: every character is an insertion
            res.e2e_pred += 1
    for g, ln in enumerate(box_lines):
        rtext = normalize_eval(ln.text, track)
        res.bag_ref_chars += len(rtext)
        if g not in used_gt:
            res.bag_errors += len(rtext)  # a missed line: every character is a deletion
    res.e2e_gt = n_care
    return res


@dataclass
class DetSummary:
    precision: float
    recall: float
    f1: float


def det_summary(results: Sequence[PageResult], thr: float) -> DetSummary:
    tp = sum(r.det[thr][0] for r in results)
    fp = sum(r.det[thr][1] for r in results)
    fn = sum(r.det[thr][2] for r in results)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return DetSummary(p, rc, 2 * p * rc / (p + rc) if p + rc else 0.0)
