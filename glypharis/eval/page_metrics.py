"""Detection and end-to-end page metrics (EXPERIMENT_PROTOCOL §3).

Ground truth is a `glypharis.data.pages.Page`. Which ground-truth lines count:

- **text lines** (order-aware page text): legible, not handwritten, not vertical. Handwriting and
  rotated text are out of scope for an English printed-text engine (ADR-0008).
- **box lines** (detection, bag-of-lines, E2E F1): text lines whose box is trusted, i.e. no GDC
  alignment score or a score of at least `MIN_ALIGN_SCORE`.
- **recognition lines** (recognition-only benchmarks cut from pages): see `is_rec_line`.
- every other ground-truth line is **don't care**: a prediction matched to it is ignored, and it is
  never a miss. Predictions lying mostly inside a page ignore region or a don't-care line are
  ignored too. The same rule applies to every metric, the order-aware page text included.

Matching is one-to-one by polygon IoU and needs IoU >= threshold. It first maximises the number of
pairs that pass the threshold, then their summed IoU (`match_pairs`). Predictions are matched to
the lines that count before the leftovers are matched to don't-care lines, so a don't-care line can
never take a match away from a line that counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import shapely
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from glypharis.data.pages import Page, PageLine
from glypharis.data.text import normalize_eval
from glypharis.eval.text_metrics import Track, char_distance, word_distance

MIN_ALIGN_SCORE = 0.3
IGNORE_OVERLAP = 0.5  # share of a prediction's area inside don't-care regions to ignore it
E2E_MAX_LINE_CER = 0.1
TEXT_MATCH_IOU = 0.5  # matching used by bag-of-lines, E2E F1 and the order-aware filter
# Recognition benchmarks keep lines the recognizer's training data left out for these reasons.
REC_EVAL_KEEPS = ("out_of_charset", "too_long")


@dataclass(frozen=True)
class PredLine:
    polygon: list[list[float]]
    text: str = ""
    confidence: float | None = None


def is_text_line(line: PageLine) -> bool:
    return line.legible and not line.handwritten and not line.vertical and bool(line.text.strip())


def has_trusted_box(line: PageLine) -> bool:
    return line.align_score is None or line.align_score >= MIN_ALIGN_SCORE


def is_box_line(line: PageLine) -> bool:
    return is_text_line(line) and has_trusted_box(line)


def is_rec_line(line: PageLine) -> bool:
    """Lines in a recognition-only benchmark cut from pages.

    Every text line with a trusted box (the crop is cut from the box), apart from lines on the
    exclusion list. Lines with characters outside the GLYPHARIS charset and very long lines stay in:
    EXPERIMENT_PROTOCOL §2 counts out-of-charset characters as errors no model can avoid (reported
    as the charset ceiling), and a benchmark must not be shaped by one system's limits.
    """
    usable = line.rec_ok or line.rec_skip in REC_EVAL_KEEPS
    return usable and is_text_line(line) and has_trusted_box(line)


def _poly(points: Sequence[Sequence[float]]) -> Polygon:
    if len(points) < 3:
        return Polygon()
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


def match_pairs(iou: np.ndarray, thr: float) -> dict[int, int]:
    """One-to-one matching of rows to columns with IoU >= thr.

    Maximises the number of matched pairs first and their summed IoU second: each valid pair is
    worth K + IoU with K larger than any possible number of pairs, so one more pair always beats
    any IoU gain. Maximising the plain IoU sum can trade a valid pair for an invalid one.
    """
    if iou.size == 0:
        return {}
    valid = iou >= thr
    if not valid.any():
        return {}
    k = min(iou.shape) + 1
    weight = np.where(valid, k + iou, 0.0)
    rows, cols = linear_sum_assignment(-weight)
    return {int(r): int(c) for r, c in zip(rows, cols, strict=True) if valid[r, c]}


def match_two_tier(iou: np.ndarray, n_first: int, thr: float) -> dict[int, int]:
    """Match against the first ``n_first`` columns, then the unmatched rows against the rest."""
    first = match_pairs(iou[:, :n_first], thr)
    rest_rows = [r for r in range(iou.shape[0]) if r not in first]
    if not rest_rows or iou.shape[1] == n_first:
        return first
    second = match_pairs(iou[np.ix_(rest_rows, list(range(n_first, iou.shape[1])))], thr)
    return {**first, **{rest_rows[r]: n_first + c for r, c in second.items()}}


def _mostly_inside(pred_polys: list[Polygon], area: Polygon) -> np.ndarray:
    if not pred_polys or area.is_empty:
        return np.zeros(len(pred_polys), dtype=bool)
    pred_area = np.array([p.area for p in pred_polys])
    inside = shapely.area(shapely.intersection(np.array(pred_polys, dtype=object), area))
    return np.asarray(inside >= IGNORE_OVERLAP * np.maximum(pred_area, 1e-9), dtype=bool)


@dataclass
class PageResult:
    doc_id: str
    # detection at each IoU threshold: (tp, fp, fn)
    det: dict[float, tuple[int, int, int]] = field(default_factory=dict)
    matched_iou_sum: float = 0.0  # over true positives at the first threshold
    # order-aware page text
    aware_errors: int = 0
    aware_ref_chars: int = 0
    aware_word_errors: int = 0
    aware_ref_words: int = 0
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
    ignore_polys = [_poly(p) for p in gt.ignore]

    # Predictions in an ignore region or mostly over don't-care lines are not false positives.
    mostly_ignored = _mostly_inside(pred_polys, shapely.union_all(ignore_polys + gt_polys[n_care:]))

    iou = iou_matrix(pred_polys, gt_polys)
    for i, thr in enumerate(thresholds):
        matched = match_two_tier(iou, n_care, thr)
        tp = fp = 0
        for p in range(len(preds)):
            g = matched.get(p)
            if g is not None and g < n_care:
                tp += 1
                if i == 0:
                    res.matched_iou_sum += float(iou[p, g])
            elif g is None and not mostly_ignored[p]:
                fp += 1
        res.det[thr] = (tp, fp, n_care - tp)

    if not score_text:
        return res

    # Order-aware page text: every text line in ground-truth order against the predictions in the
    # system's order, leaving out predictions on don't-care text (non-text lines, ignore regions)
    # exactly as the other metrics do. Text lines with an untrusted box are in the reference, so
    # predictions on them stay in the hypothesis.
    text_ids = {ln.line_id for ln in gt_lines if is_text_line(ln)}
    order = sorted(range(len(gt_lines)), key=lambda g: gt_lines[g].line_id not in text_ids)
    n_text = len(text_ids)
    text_match = match_two_tier(iou[:, order], n_text, TEXT_MATCH_IOU)
    non_text = [gt_polys[g] for g in order[n_text:]]
    off_text = _mostly_inside(pred_polys, shapely.union_all(ignore_polys + non_text))
    kept: list[str] = []
    for i, pred_line in enumerate(preds):
        tg = text_match.get(i)
        on_text_line = tg is not None and tg < n_text
        unmatched_on_page = tg is None and not off_text[i]
        if on_text_line or unmatched_on_page:  # matched to a non-text line, or off-text: dropped
            kept.append(pred_line.text)
    ref = normalize_eval("\n".join(ln.text for ln in gt.lines if is_text_line(ln)), track)
    hyp = normalize_eval("\n".join(kept), track)
    res.aware_errors, res.aware_ref_chars = char_distance(hyp, ref), len(ref)
    res.aware_word_errors, res.aware_ref_words = word_distance(hyp, ref), len(ref.split())

    # Bag of lines and E2E F1 on trusted boxes.
    match = match_two_tier(iou, n_care, TEXT_MATCH_IOU)
    used_gt: set[int] = set()
    for p, pred in enumerate(preds):
        g = match.get(p)
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


def vertical_view(gt: Page) -> Page:
    """The page with its vertical lines as the only lines that count.

    GDC reports rotated lines (margin stamps, axis titles) separately (EXPERIMENT_PROTOCOL §3).
    Scoring this view with `evaluate_page` gives detection, bag-of-lines and E2E F1 for those lines;
    every other line becomes don't care, so predictions on horizontal text are ignored rather than
    counted as false positives. Its order-aware text has no meaning and is not reported.
    """
    lines = [
        replace(ln, vertical=False)
        if ln.vertical and ln.legible and not ln.handwritten and ln.text.strip()
        else replace(ln, legible=False)
        for ln in gt.lines
    ]
    return replace(gt, lines=lines)


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
