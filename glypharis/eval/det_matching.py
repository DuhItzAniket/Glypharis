"""Detection match details and bounded-cost detection counting (spec §8.3).

`glypharis.eval.page_metrics.evaluate_page` is the definition of a detection count. This module
reuses its helpers (`_poly`, `iou_matrix`, `match_pairs`, `match_two_tier`, `_mostly_inside`)
without changing that file, and adds:

- `detection_matches`: the status of every prediction and ground-truth line at one threshold, plus
  every overlapping pair, for overlays and Phase 9's merge/split counts. Its counts equal
  `evaluate_page` (parity test).
- `fast_det_counts`: the same detection counts at a cost that does not grow with noise. Before each
  assignment it drops the rows (predictions) and columns (ground-truth lines) with no IoU at or
  above the threshold. Such a row or column carries weight 0 and can never be part of a valid
  match, so the set of optimal matchings is unchanged: tp, fn and the matched IoU sum always agree
  with `evaluate_page`, and fp can differ only if two assignments tie exactly in weight and
  `linear_sum_assignment` breaks the tie differently on the smaller matrix.
- `det_cis`: page-bootstrap CIs for precision, recall and F1.
- the Phase 5 validation page set (`baseline_doc_ids`), reproduced here because package code never
  imports `scripts/run_baselines.py`; a test ties the two together.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import shapely
from shapely.geometry import Polygon

from glypharis.data.pages import Page
from glypharis.eval.page_metrics import (
    PageResult,
    PredLine,
    _mostly_inside,
    _poly,
    iou_matrix,
    is_box_line,
    match_pairs,
    match_two_tier,
)
from glypharis.eval.stats import Interval, ratio_ci

BASELINE_VAL_PAGES: Final = 400  # the Phase 5 validation page sample (run_baselines.E2E_VAL_PAGES)
BASELINE_SEED: Final = 5  # run_baselines.SEED

PredStatus = Literal["tp", "fp", "ignored"]
GtStatus = Literal["tp", "fn", "dont_care"]


@dataclass
class DetMatches:
    """Detection outcome of one page at one IoU threshold, indexed like the inputs."""

    doc_id: str
    thr: float
    pred_status: list[PredStatus]  # ignored: matched to a don't-care line, or mostly inside one
    pred_gt: list[int | None]  # matched line, as an index into gt.lines (file order)
    pred_iou: list[float]  # IoU with the matched line, 0.0 when unmatched
    gt_status: list[GtStatus]  # per gt.lines
    overlaps: list[tuple[int, int, float, float, float]]  # pred, gt, IoU, inter/pred, inter/gt

    def counts(self) -> tuple[int, int, int]:
        """(tp, fp, fn), as `evaluate_page` reports them in ``det[thr]``."""
        return (
            self.pred_status.count("tp"),
            self.pred_status.count("fp"),
            self.gt_status.count("fn"),
        )


@dataclass
class _PageGeometry:
    order: list[int]  # gt.lines indices, care lines first, as evaluate_page orders them
    n_care: int
    pred_polys: list[Polygon]
    gt_polys: list[Polygon]
    iou: np.ndarray  # predictions x ordered lines
    mostly_ignored: np.ndarray  # bool per prediction


def _geometry(gt: Page, preds: Sequence[PredLine]) -> _PageGeometry:
    # Mirrors the set-up of evaluate_page line for line, so the IoUs are bit-identical.
    care = [i for i, ln in enumerate(gt.lines) if is_box_line(ln)]
    care_set = set(care)
    order = care + [i for i in range(len(gt.lines)) if i not in care_set]
    gt_polys = [_poly(gt.lines[i].polygon) for i in order]
    pred_polys = [_poly(p.polygon) for p in preds]
    ignore_polys = [_poly(p) for p in gt.ignore]
    n_care = len(care)
    mostly_ignored = _mostly_inside(pred_polys, shapely.union_all(ignore_polys + gt_polys[n_care:]))
    return _PageGeometry(
        order, n_care, pred_polys, gt_polys, iou_matrix(pred_polys, gt_polys), mostly_ignored
    )


def detection_matches(gt: Page, preds: Sequence[PredLine], thr: float = 0.5) -> DetMatches:
    """Per-prediction and per-line statuses at ``thr``, with every overlapping pair."""
    g = _geometry(gt, preds)
    matched = match_two_tier(g.iou, g.n_care, thr)
    pred_status: list[PredStatus] = []
    pred_gt: list[int | None] = []
    pred_iou: list[float] = []
    gt_status: list[GtStatus] = ["dont_care"] * len(gt.lines)
    for i in g.order[: g.n_care]:
        gt_status[i] = "fn"
    for p in range(len(preds)):
        col = matched.get(p)
        if col is None:
            pred_status.append("ignored" if g.mostly_ignored[p] else "fp")
            pred_gt.append(None)
            pred_iou.append(0.0)
            continue
        line = g.order[col]
        if col < g.n_care:
            pred_status.append("tp")
            gt_status[line] = "tp"
        else:
            pred_status.append("ignored")
        pred_gt.append(line)
        pred_iou.append(float(g.iou[p, col]))

    overlaps: list[tuple[int, int, float, float, float]] = []
    rows, cols = np.nonzero(g.iou > 0)
    if rows.size:
        a = np.array(g.pred_polys, dtype=object)[rows]
        b = np.array(g.gt_polys, dtype=object)[cols]
        inter = shapely.area(shapely.intersection(a, b))
        area_a, area_b = shapely.area(a), shapely.area(b)
        for k, (p, col) in enumerate(zip(rows.tolist(), cols.tolist(), strict=True)):
            overlaps.append(
                (
                    p,
                    g.order[col],
                    float(g.iou[p, col]),
                    float(inter[k] / area_a[k]) if area_a[k] > 0 else 0.0,
                    float(inter[k] / area_b[k]) if area_b[k] > 0 else 0.0,
                )
            )
    return DetMatches(gt.doc_id, thr, pred_status, pred_gt, pred_iou, gt_status, overlaps)


def _pruned_match(
    iou: np.ndarray, rows: np.ndarray, cols: np.ndarray, thr: float
) -> dict[int, int]:
    """`match_pairs` on the rows x cols submatrix, keeping only rows and columns with a valid
    pair; returns matches in full-matrix indices."""
    if rows.size == 0 or cols.size == 0:
        return {}
    sub = iou[np.ix_(rows, cols)]
    valid = sub >= thr
    keep_r, keep_c = np.flatnonzero(valid.any(axis=1)), np.flatnonzero(valid.any(axis=0))
    if keep_r.size == 0:
        return {}
    m = match_pairs(sub[np.ix_(keep_r, keep_c)], thr)
    return {int(rows[keep_r[r]]): int(cols[keep_c[c]]) for r, c in m.items()}


def fast_det_counts(
    gt: Page, preds: Sequence[PredLine], thresholds: Sequence[float] = (0.5, 0.7)
) -> PageResult:
    """`evaluate_page(gt, preds, thresholds, score_text=False)` detection counts at bounded cost.

    Fills ``det`` and ``matched_iou_sum`` (first threshold) only; the text fields stay 0. Tier 2
    (don't-care lines) sees the predictions tier 1 left unmatched, as `match_two_tier` does.
    """
    res = PageResult(gt.doc_id)
    g = _geometry(gt, preds)
    n_pred, n_gt = g.iou.shape
    care_cols = np.arange(g.n_care)
    dont_cols = np.arange(g.n_care, n_gt)
    all_rows = np.arange(n_pred)
    for i, thr in enumerate(thresholds):
        first = _pruned_match(g.iou, all_rows, care_cols, thr)
        rest = np.array([r for r in range(n_pred) if r not in first], dtype=np.int64)
        second = _pruned_match(g.iou, rest, dont_cols, thr)
        tp = len(first)
        fp = sum(
            1
            for p in range(n_pred)
            if p not in first and p not in second and not g.mostly_ignored[p]
        )
        res.det[thr] = (tp, fp, g.n_care - tp)
        if i == 0:
            # Summed in prediction order, as evaluate_page does.
            res.matched_iou_sum = sum(float(g.iou[p, first[p]]) for p in sorted(first))
    return res


def det_cis(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> dict[str, Interval]:
    """Page-bootstrap CIs (`ratio_ci`) for micro precision, recall and F1 = 2tp / (2tp + fp + fn).

    The arrays hold one value per page; keys are ``precision``, ``recall`` and ``f1``.
    """
    t, f_p, f_n = (np.asarray(x, dtype=np.float64) for x in (tp, fp, fn))
    return {
        "precision": ratio_ci(t, t + f_p, "pages"),
        "recall": ratio_ci(t, t + f_n, "pages"),
        "f1": ratio_ci(2 * t, 2 * t + f_p + f_n, "pages"),
    }


def baseline_doc_ids(doc_ids_in_file_order: Sequence[str]) -> list[str]:
    """The Phase 5 validation page sample (`run_baselines.page_indices`), in pages-file order.

    All pages when there are at most `BASELINE_VAL_PAGES`; otherwise a seeded draw without
    replacement whose indices are sorted, so the pages keep their file order.
    """
    ids = list(doc_ids_in_file_order)
    n = len(ids)
    if n <= BASELINE_VAL_PAGES:
        return ids
    idx = np.sort(np.random.default_rng(BASELINE_SEED).choice(n, BASELINE_VAL_PAGES, replace=False))
    return [ids[int(i)] for i in idx]
