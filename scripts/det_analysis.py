"""Detector error analysis on validation (Phase 9 plan items 1 and 6).

    python scripts/det_analysis.py --config configs/det/gw-det-001.yaml --run runs/GW-DET-001 \
        --out reports/error_cases/detector/GW-DET-001
    python scripts/det_analysis.py --config configs/det/gw-det-001.yaml --run DIR --box-shape \
        --out reports/error_cases/detector/GW-DET-006 [--set det.post.bin_thresh=...]

Reads a run directory's full-validation files (``val_full_best_boxes.jsonl.gz``,
``val_full_best_candidates.jsonl.gz``, ``eval_full_best.json``; a training run or
`scripts/det_eval.py --save-full`) and the HierText validation pages. Every definition is the one
registered in `docs/PHASE_STATUS.md` (Phase 09) before any of this ran:

- slices: line recall by height and text angle, page P/R/F1 by quartile of care-line density, blur
  and contrast, with page-bootstrap CIs;
- merges (a prediction covering at least two care lines, each with intersection / line ≥ 0.5) and
  splits (a care line with at least two predictions, each with intersection / prediction ≥ 0.5);
- every missed care line and every false alarm in one category, first match wins;
- galleries of up to `GALLERY_N` crops per category, drawn with `default_rng([GALLERY_SEED, k])`
  over the sorted members, independent of results, and the worst pages;
- the kernel excess of the trained map (RISK_REGISTER R-21).

``--box-shape`` instead compares ``quad`` and ``poly`` boxes built from the same saved candidates
(GW-DET-006, RISK_REGISTER R-22), on all pages and on the curved care lines. Validation only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from PIL import Image, ImageDraw

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.config.schema import DetConfig
from glypharis.data.det_dataset import PageIndex, read_gt_pages
from glypharis.data.det_geometry import decode_page
from glypharis.data.download import sha256_file
from glypharis.data.pages import Page
from glypharis.eval.det_matching import DetMatches, det_cis, detection_matches
from glypharis.eval.det_overlay import (
    GT_FN,
    GT_TP,
    HIERTEXT_ATTRIBUTION,
    PRED_FP,
    PRED_TP,
    contact_sheet,
    draw_overlay,
)
from glypharis.eval.det_slices import LineAttrs, PageAttrs, load_or_build_slices
from glypharis.eval.page_metrics import PredLine, is_box_line
from glypharis.eval.stats import Interval, paired_ratios_diff_ci, ratio_ci
from glypharis.models.det.postprocess import Candidate, boxes_from_candidates
from glypharis.models.det.preprocess import ResizeInfo
from glypharis.training.det_eval import (
    box_from_dict,
    interval_dict,
    kernel_excess,
    read_records,
)

THR: Final = 0.5
COVER: Final = 0.5  # merge: intersection / line; split and fragment: intersection / prediction
TOUCH: Final = 0.1  # partly found / background
LOCAL_LOW: Final = 0.3  # poorly localised: best IoU in [0.3, 0.5)
GALLERY_N: Final = 24
GALLERY_SEED: Final = 20261027
WORST_PAGES: Final = 12
WORST_MIN_CARE: Final = 5
CURVED_MIN_VERTICES: Final = 6
CURVED_MAX_FILL: Final = 0.8  # polygon area / minimum-area-rectangle area
HEIGHT_BINS: Final = ((0, 8), (8, 12), (12, 20), (20, 40), (40, math.inf))
ANGLE_BINS: Final = ((0, 5), (5, 15), (15, 45), (45, math.inf))
FN_CATS: Final = (
    "merged",
    "split",
    "poorly_localised",
    "partly_found",
    "missed_low_score",
    "missed_dropped",
    "missed_no_candidate",
)
FP_CATS: Final = ("merge", "fragment", "poorly_localised", "background", "other")


# ----------------------------------------------------------------------------- inputs


@dataclasses.dataclass
class PageData:
    page: Page
    preds: list[PredLine]
    scores: list[float]
    m: DetMatches
    cands: list[Candidate]
    info: ResizeInfo


def load_run(run: Path, op: str, gt: Mapping[str, Page]) -> dict[str, PageData]:
    boxes = {
        r["doc_id"]: r["boxes"][op] for r in read_records(run / "val_full_best_boxes.jsonl.gz")
    }
    cands = {r["doc_id"]: r for r in read_records(run / "val_full_best_candidates.jsonl.gz")}
    if not boxes or set(boxes) != set(cands):
        raise SystemExit(f"{run}: boxes and candidates missing or not for the same pages")
    out = {}
    for doc_id in sorted(boxes):
        page = gt[doc_id]
        found = [box_from_dict(b) for b in boxes[doc_id]]
        preds = [b.to_pred_line() for b in found]
        c = cands[doc_id]
        out[doc_id] = PageData(
            page,
            preds,
            [b.score for b in found],
            detection_matches(page, preds, THR),
            [Candidate.from_dict(x) for x in c["candidates"]],
            ResizeInfo(**c["info"]),
        )
    return out


# ------------------------------------------------------------------------- categories


def _care(page: Page) -> list[int]:
    return [i for i, ln in enumerate(page.lines) if is_box_line(ln)]


def classify_page(
    d: PageData, post_box_thresh: float
) -> tuple[dict[int, str], dict[int, str], int, int]:
    """(fn category per care-line index, fp category per prediction index, merges, split lines)."""
    page, m = d.page, d.m
    care = set(_care(page))
    by_gt: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    by_pred: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)
    for p, g, iou, ip, ig in m.overlaps:
        by_gt[g].append((p, iou, ip, ig))
        by_pred[p].append((g, iou, ip, ig))
    merge_preds = {
        p
        for p, ov in by_pred.items()
        if sum(1 for g, _, _, ig in ov if g in care and ig >= COVER) >= 2
    }
    merged_lines = {
        g for p in merge_preds for g, _, _, ig in by_pred[p] if g in care and ig >= COVER
    }
    split_lines = {
        g
        for g, ov in by_gt.items()
        if g in care and sum(1 for _, _, ip, _ in ov if ip >= COVER) >= 2
    }

    centres = [(c.area_rect[0] / d.info.sx, c.area_rect[1] / d.info.sy, c.score) for c in d.cands]
    fn: dict[int, str] = {}
    for g in sorted(care):
        if m.gt_status[g] != "fn":
            continue
        ov = by_gt.get(g, [])
        best_iou = max((iou for _, iou, _, _ in ov), default=0.0)
        best_cover = max((ig for _, _, _, ig in ov), default=0.0)
        if g in merged_lines:
            fn[g] = "merged"
        elif g in split_lines:
            fn[g] = "split"
        elif best_iou >= LOCAL_LOW:
            fn[g] = "poorly_localised"
        elif best_cover >= TOUCH:
            fn[g] = "partly_found"
        else:
            poly = np.asarray(page.lines[g].polygon, dtype=np.float32).reshape(-1, 1, 2)
            inside = [
                s
                for x, y, s in centres
                if cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
            ]
            if not inside:
                fn[g] = "missed_no_candidate"
            elif max(inside) < post_box_thresh:
                fn[g] = "missed_low_score"
            else:
                fn[g] = "missed_dropped"
    fp: dict[int, str] = {}
    for p, status in enumerate(m.pred_status):
        if status != "fp":
            continue
        ov = by_pred.get(p, [])
        care_ov = [(iou, ip) for g, iou, ip, _ in ov if g in care]
        if p in merge_preds:
            fp[p] = "merge"
        elif any(ip >= COVER for _, ip in care_ov):
            fp[p] = "fragment"
        elif any(LOCAL_LOW <= iou < THR for iou, _ in care_ov):
            fp[p] = "poorly_localised"
        elif max((ip for _, _, ip, _ in ov), default=0.0) < TOUCH:
            fp[p] = "background"
        else:
            fp[p] = "other"
    return fn, fp, len(merge_preds), len(split_lines)


# ------------------------------------------------------------------------------ tables


def _bin(x: float, bins: Sequence[tuple[float, float]]) -> int:
    return next(i for i, (lo, hi) in enumerate(bins) if lo <= x < hi)


def _bin_label(lo: float, hi: float, unit: str) -> str:
    return f">= {lo:g} {unit}" if math.isinf(hi) else f"{lo:g}–{hi:g} {unit}"


def _ci(iv: Interval) -> str:
    return f"{iv.value:.3f} [{iv.low:.3f}, {iv.high:.3f}]"


def line_recall_table(
    rows: list[tuple[str, int, bool]], n_bins: int, labels: list[str], title: str
) -> list[str]:
    """rows: (doc_id, bin, found). Recall per bin with page-resampled CIs."""
    out = [f"| {title} | Lines | Recall@0.5 |", "|---|---|---|"]
    docs = sorted({d for d, _, _ in rows})
    pos = {d: i for i, d in enumerate(docs)}
    for k in range(n_bins):
        num = np.zeros(len(docs))
        den = np.zeros(len(docs))
        for d, b, found in rows:
            if b == k:
                den[pos[d]] += 1
                num[pos[d]] += found
        if den.sum():
            out.append(f"| {labels[k]} | {int(den.sum()):,} | {_ci(ratio_ci(num, den, 'pages'))} |")
    return out


def page_quartile_table(
    attr: str, pages: Sequence[PageAttrs], counts: Mapping[str, tuple[int, int, int]], label: str
) -> list[str]:
    vals = np.array([getattr(p, attr) for p in pages], dtype=np.float64)
    edges = np.quantile(vals, [0.25, 0.5, 0.75])
    out = [
        f"| {label} quartile | Range | Pages | Precision@0.5 | Recall@0.5 | F1@0.5 |",
        "|---|---|---|---|---|---|",
    ]
    q = np.searchsorted(edges, vals, side="right")
    bounds = [vals.min(), *edges, vals.max()]
    for k in range(4):
        members = [p.doc_id for p, qi in zip(pages, q, strict=True) if qi == k]
        a = np.array([counts[d] for d in members], dtype=np.float64).reshape(-1, 3)
        ci = det_cis(a[:, 0], a[:, 1], a[:, 2])
        out.append(
            f"| Q{k + 1} | {bounds[k]:.3g}–{bounds[k + 1]:.3g} | {len(members)} | "
            f"{_ci(ci['precision'])} | {_ci(ci['recall'])} | {_ci(ci['f1'])} |"
        )
    return out


# ---------------------------------------------------------------------------- galleries


def _crop_tile(
    rgb: np.ndarray,
    focus: Sequence[Sequence[float]],
    gt: Page,
    preds: Sequence[PredLine],
    m: DetMatches,
) -> Image.Image:
    pts = np.asarray(focus, dtype=np.float64)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    pad = max(12.0, 0.3 * min(x1 - x0, y1 - y0) + 0.15 * max(x1 - x0, y1 - y0))
    h, w = rgb.shape[:2]
    cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    cx1, cy1 = int(min(w, x1 + pad)), int(min(h, y1 + pad))
    tile = Image.fromarray(np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1, :3]))
    f = 320 / max(tile.width, tile.height)
    tile = tile.resize((max(1, round(tile.width * f)), max(1, round(tile.height * f))))
    draw = ImageDraw.Draw(tile)

    def outline(poly: Sequence[Sequence[float]], colour: tuple[int, int, int], width: int) -> None:
        q = [((x - cx0) * f, (y - cy0) * f) for x, y in poly]
        draw.line([*q, q[0]], fill=colour, width=width)

    for line, gs in zip(gt.lines, m.gt_status, strict=True):
        if gs != "dont_care":
            outline(line.polygon, GT_TP if gs == "tp" else GT_FN, 3 if gs == "fn" else 1)
    for pred, ps in zip(preds, m.pred_status, strict=True):
        if ps != "ignored":
            outline(pred.polygon, PRED_TP if ps == "tp" else PRED_FP, 2)
    return tile


def galleries(
    out: Path,
    data_root: Path,
    pages: Mapping[str, PageData],
    fn_members: Mapping[str, list[tuple[str, int]]],
    fp_members: Mapping[str, list[tuple[str, int]]],
) -> list[str]:
    gal = out / "galleries"
    gal.mkdir(parents=True, exist_ok=True)
    picks: dict[str, list[tuple[str, int, str]]] = {}
    for k, (kind, cat, members) in enumerate(
        [("fn", c, fn_members[c]) for c in FN_CATS] + [("fp", c, fp_members[c]) for c in FP_CATS]
    ):
        ms = sorted(members)
        n = min(GALLERY_N, len(ms))
        idx = np.sort(np.random.default_rng([GALLERY_SEED, k]).choice(len(ms), n, replace=False))
        picks[f"{kind}_{cat}"] = [(ms[int(i)][0], ms[int(i)][1], kind) for i in idx]
    need = sorted({d for p in picks.values() for d, _, _ in p})
    tiles: dict[tuple[str, int, str], Image.Image] = {}
    for doc_id in need:
        d = pages[doc_id]
        rgb = decode_page(data_root / d.page.image, (d.page.width, d.page.height))
        for p in picks.values():
            for doc, i, kind in p:
                if doc == doc_id:
                    focus = d.page.lines[i].polygon if kind == "fn" else d.preds[i].polygon
                    tiles[(doc, i, kind)] = _crop_tile(rgb, focus, d.page, d.preds, d.m)
    lines = [
        "## Galleries",
        "",
        f"Up to {GALLERY_N} crops per category, drawn by `default_rng([{GALLERY_SEED}, k])` over "
        "the sorted members (k = the category's position: missed-line categories first, then "
        "false-alarm ones), independent of results. Ground truth: found green, missed orange "
        "(thick); predictions: tp blue, fp red; don't-care lines not drawn. "
        f"{HIERTEXT_ATTRIBUTION}.",
        "",
    ]
    for name, p in picks.items():
        sheet = contact_sheet([tiles[x] for x in p], cols=4, tile_long=320)
        sheet.save(gal / f"{name}.jpg", quality=85)
        lines.append(f"- `{name}` ({len(p)} shown): ![{name}](galleries/{name}.jpg)")

    def page_f1(d: PageData) -> float:
        tp, fp, fn = d.m.counts()
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0

    worst = sorted(
        (d for d in pages.values() if len(_care(d.page)) >= WORST_MIN_CARE),
        key=lambda d: (page_f1(d), d.page.doc_id),
    )[:WORST_PAGES]
    wt = []
    for d in worst:
        rgb = decode_page(data_root / d.page.image, (d.page.width, d.page.height))
        wt.append(draw_overlay(rgb, d.page, d.preds, d.m, 1024))
    contact_sheet(wt, cols=4, tile_long=640).save(gal / "worst_pages.jpg", quality=85)
    lines.append(
        f"- worst {WORST_PAGES} pages (lowest F1@0.5, at least {WORST_MIN_CARE} care "
        "lines): ![worst pages](galleries/worst_pages.jpg)"
    )
    return lines


# ------------------------------------------------------------------------------ analysis


def analyse(
    args: argparse.Namespace,
    det: DetConfig,
    data_root: Path,
    pages: dict[str, PageData],
    gt_all: Mapping[str, Page],
) -> dict[str, Any]:
    pages_sha = sha256_file(data_root / det.data.val_pages[0])
    table = load_or_build_slices(
        [gt_all[d] for d in sorted(pages)], pages_sha, data_root, det.infer.long_side
    )
    line_attr: dict[tuple[str, str], LineAttrs] = {(a.doc_id, a.line_id): a for a in table.lines}
    page_attr = {p.doc_id: p for p in table.pages}

    fn_members: dict[str, list[tuple[str, int]]] = {c: [] for c in FN_CATS}
    fp_members: dict[str, list[tuple[str, int]]] = {c: [] for c in FP_CATS}
    fn_by_height: dict[str, Counter[int]] = {c: Counter() for c in FN_CATS}
    merges = split_lines = care_total = tp_total = fp_total = fn_total = 0
    height_rows, angle_rows, tall_rows = [], [], []
    counts: dict[str, tuple[int, int, int]] = {}
    excess: list[tuple[float, float]] = []
    for doc_id, d in pages.items():
        fn, fp, n_merge, n_split = classify_page(d, det.post.box_thresh)
        merges += n_merge
        split_lines += n_split
        tp, fpc, fnc = d.m.counts()
        counts[doc_id] = (tp, fpc, fnc)
        tp_total, fp_total, fn_total = tp_total + tp, fp_total + fpc, fn_total + fnc
        for g in _care(d.page):
            care_total += 1
            a = line_attr[(doc_id, d.page.lines[g].line_id)]
            found = d.m.gt_status[g] == "tp"
            hb = _bin(a.height, HEIGHT_BINS)
            height_rows.append((doc_id, hb, found))
            angle_rows.append((doc_id, _bin(abs(a.angle), ANGLE_BINS), found))
            tall_rows.append((doc_id, int(a.tall_box), found))
            if g in fn:
                fn_members[fn[g]].append((doc_id, g))
                fn_by_height[fn[g]][hb] += 1
        for p, cat in fp.items():
            fp_members[cat].append((doc_id, p))
        excess += kernel_excess(d.cands, d.page, d.info, det.labels)

    h_labels = [_bin_label(lo, hi, "px") for lo, hi in HEIGHT_BINS]
    a_labels = [_bin_label(lo, hi, "°") for lo, hi in ANGLE_BINS]
    md = [
        f"# Detector error analysis: {args.label}",
        "",
        "Generated by `scripts/det_analysis.py`; do not edit by hand. Definitions as registered in",
        "`docs/PHASE_STATUS.md` (Phase 09) before this ran. Validation pages only, IoU 0.5, at",
        f"operating point `{args.op}`; heights and angles at the {det.infer.long_side} px "
        "inference scale.",
        "",
        f"Pages {len(pages):,}; care lines {care_total:,}; tp {tp_total:,}, fp {fp_total:,}, "
        f"fn {fn_total:,}.",
        "",
        "## Missed lines (fn) by category",
        "",
        "| Category | Lines | Share of fn | " + " | ".join(h_labels) + " |",
        "|---|---|---|" + "---|" * len(h_labels),
    ]
    for c in FN_CATS:
        n = len(fn_members[c])
        md.append(
            f"| {c} | {n:,} | {n / fn_total:.1%} | "
            + " | ".join(f"{fn_by_height[c][k]:,}" for k in range(len(HEIGHT_BINS)))
            + " |"
        )
    md += [
        "",
        "`missed_low_score`: a candidate sat in the line but scored below `box_thresh`; "
        "`missed_dropped`: a candidate above it produced no box near the line (size filter); "
        "`missed_no_candidate`: no component of the binarised map sat in the line.",
        "",
        "## False alarms (fp) by category",
        "",
        "| Category | Predictions | Share of fp |",
        "|---|---|---|",
    ]
    for c in FP_CATS:
        n = len(fp_members[c])
        md.append(f"| {c} | {n:,} | {n / fp_total:.1%} |" if fp_total else f"| {c} | 0 | - |")
    md += [
        "",
        "## Merges and splits",
        "",
        f"- Merge predictions (one box over at least two care lines): {merges:,}",
        f"- Split care lines (one line, at least two boxes): {split_lines:,} "
        f"({split_lines / care_total:.2%} of care lines)",
        "",
        "## Line recall by height, angle and box shape",
        "",
    ]
    md += line_recall_table(
        height_rows, len(HEIGHT_BINS), h_labels, "Height (rectangle short side)"
    )
    md += ["", *line_recall_table(angle_rows, len(ANGLE_BINS), a_labels, "abs(text angle)")]
    md += [
        "",
        *line_recall_table(
            tall_rows, 2, ["other", "tall box (> 45° from horizontal)"], "Box shape"
        ),
    ]
    attrs = [page_attr[d] for d in sorted(pages) if page_attr[d].n_care > 0]
    md += [
        "",
        "## Pages by density, blur and contrast",
        "",
        "Quartiles over the validation pages with care lines; page-bootstrap CIs.",
        "",
    ]
    for attr, label in (
        ("density_per_mp", "Care lines per megapixel"),
        ("blur_lapvar", "Blur (variance of the Laplacian; low = blurry)"),
        ("contrast", "Contrast (grey-level std / 255)"),
    ):
        md += [f"**{label}**", "", *page_quartile_table(attr, attrs, counts, label.split(" (")[0])]
        md.append("")
    ex = np.array(excess) if excess else np.zeros((0, 2))
    kx: dict[str, Any] = {
        "candidates": len(ex),
        "excess_px_median": float(np.median(ex[:, 0])) if len(ex) else None,
        "ratio_median": float(np.median(ex[:, 1])) if len(ex) else None,
        "ratio_iqr": [float(x) for x in np.quantile(ex[:, 1], [0.25, 0.75])] if len(ex) else None,
    }
    md += (
        [
            "## Kernel excess (RISK_REGISTER R-21)",
            "",
            "Predicted kernel width against the label kernel of the care line each candidate sits "
            "in "
            f"({kx['candidates']:,} candidates): median excess {kx['excess_px_median']:.2f} px, "
            f"median ratio {kx['ratio_median']:.2f} (IQR {kx['ratio_iqr'][0]:.2f}–"
            f"{kx['ratio_iqr'][1]:.2f}). A ratio near 1 is the regime `inverse` unclip assumes; "
            "well above 1, the one `dbnet` unclip assumes.",
            "",
        ]
        if len(ex)
        else []
    )
    if args.galleries:
        md += galleries(args.out, data_root, pages, fn_members, fp_members)
    res = {
        "label": args.label,
        "run": args.run.as_posix(),
        "op": args.op,
        "pages": len(pages),
        "care_lines": care_total,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "fn_categories": {c: len(fn_members[c]) for c in FN_CATS},
        "fn_by_height": {c: [fn_by_height[c][k] for k in range(len(HEIGHT_BINS))] for c in FN_CATS},
        "fp_categories": {c: len(fp_members[c]) for c in FP_CATS},
        "merge_predictions": merges,
        "split_lines": split_lines,
        "kernel_excess": kx,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    (args.out / "analysis.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return res


# ---------------------------------------------------------------------------- box shape


def curved(line_polygon: Sequence[Sequence[float]]) -> bool:
    pts = np.asarray(line_polygon, dtype=np.float32).reshape(-1, 2)
    if len(pts) < CURVED_MIN_VERTICES:
        return False
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    rect = float(w * h)
    area = abs(float(cv2.contourArea(pts)))
    return rect > 0 and area / rect < CURVED_MAX_FILL


def box_shape(
    args: argparse.Namespace, det: DetConfig, pages: dict[str, PageData]
) -> dict[str, Any]:
    op = next(o for o in det.post.operating_points if o.name == args.op)
    per: dict[str, list[tuple[int, int, int, int, int]]] = {"quad": [], "poly": []}
    n_curved = 0
    for d in pages.values():
        curved_idx = [g for g in _care(d.page) if curved(d.page.lines[g].polygon)]
        n_curved += len(curved_idx)
        for kind in ("quad", "poly"):
            post = dataclasses.replace(det.post, box_type=kind)
            boxes, _ = boxes_from_candidates(d.cands, op, post, det.labels.shrink_ratio, d.info)
            m = detection_matches(d.page, [b.to_pred_line() for b in boxes], THR)
            tp, fp, fn = m.counts()
            ctp = sum(1 for g in curved_idx if m.gt_status[g] == "tp")
            per[kind].append((tp, fp, fn, ctp, len(curved_idx)))
    q = np.array(per["quad"], dtype=np.float64)
    p = np.array(per["poly"], dtype=np.float64)
    f1 = paired_ratios_diff_ci(
        2 * p[:, 0],
        2 * p[:, 0] + p[:, 1] + p[:, 2],
        2 * q[:, 0],
        2 * q[:, 0] + q[:, 1] + q[:, 2],
        "pages",
    )
    rec = paired_ratios_diff_ci(p[:, 3], p[:, 4], q[:, 3], q[:, 4], "pages")
    adopt = bool(rec.low > 0 and f1.high >= 0)

    def f1_of(a: np.ndarray) -> float:
        tp, fp, fn = a[:, :3].sum(axis=0)
        return float(2 * tp / (2 * tp + fp + fn))

    res = {
        "label": args.label,
        "run": args.run.as_posix(),
        "op": args.op,
        "pages": len(pages),
        "curved_lines": n_curved,
        "post": dataclasses.asdict(det.post),
        "f1_quad": f1_of(q),
        "f1_poly": f1_of(p),
        "curved_recall_quad": float(q[:, 3].sum() / q[:, 4].sum()) if n_curved else None,
        "curved_recall_poly": float(p[:, 3].sum() / p[:, 4].sum()) if n_curved else None,
        "f1_diff_poly_minus_quad": interval_dict(f1),
        "curved_recall_diff_poly_minus_quad": interval_dict(rec),
        "phase14_box_type": "poly" if adopt else "quad",
    }
    md = [
        f"# Box shape, quad or poly: {args.label}",
        "",
        "Generated by `scripts/det_analysis.py --box-shape`; do not edit by hand. Both box types",
        "from the same saved candidates, on all validation pages; curved care lines have at least "
        f"{CURVED_MIN_VERTICES} polygon vertices and fill less than {CURVED_MAX_FILL} of their "
        "minimum-area rectangle. Paired page-bootstrap 95% CIs.",
        "",
        "| | quad | poly | poly − quad [95% CI] |",
        "|---|---|---|---|",
        f"| F1@0.5, all {len(pages):,} pages | {res['f1_quad']:.4f} | {res['f1_poly']:.4f} | "
        f"{f1.value:+.4f} [{f1.low:+.4f}, {f1.high:+.4f}] |",
        f"| Recall@0.5, {n_curved:,} curved lines | {res['curved_recall_quad'] or 0:.4f} | "
        f"{res['curved_recall_poly'] or 0:.4f} | "
        f"{rec.value:+.4f} [{rec.low:+.4f}, {rec.high:+.4f}] |",
        "",
        f"Rule (GW-DET-006): `poly` for Phase 14 if the curved-line recall CI lies above 0 and the "
        f"F1 CI does not lie below 0. Result: **{res['phase14_box_type']}**.",
        "",
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "box_shape.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    (args.out / "box_shape.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--op", default=None, help="operating point (default: the run's stored)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--box-shape", action="store_true")
    parser.add_argument("--no-galleries", dest="galleries", action="store_false")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides_from_args(args))
    if cfg.det is None:
        raise SystemExit(f"{args.config}: not a detector config")
    det, data_root = cfg.det, Path(cfg.paths.data_root)
    ev = json.loads((args.run / "eval_full_best.json").read_text(encoding="utf-8"))
    args.op = args.op or ev["stored_op"]
    args.label = args.label or args.run.name
    pages_file = data_root / det.data.val_pages[0]
    ids = PageIndex.build(pages_file, "validation", data_root).doc_ids()
    gt = read_gt_pages(pages_file, ids)
    pages = load_run(args.run, args.op, gt)
    if args.box_shape:
        res = box_shape(args, det, pages)
        print(json.dumps({k: v for k, v in res.items() if k != "post"}, indent=1))
    else:
        res = analyse(args, det, data_root, pages, gt)
        print(json.dumps({k: v for k, v in res.items() if k != "fn_by_height"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
