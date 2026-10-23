"""Detection overlays for the run report (spec §10.9): one page with its match statuses, and a
contact sheet of several pages.

Statuses come from `det_matching.detection_matches`, which a parity test ties to `evaluate_page`, so
an overlay shows exactly what was counted. Colours:

- ground truth: care line matched (tp) green, missed (fn) orange and thick, don't care and ignore
  regions grey;
- predictions: tp blue, fp red, ignored (on don't-care text) grey and dashed.

A legend strip under the page gives the counts at the match threshold, and, for HierText pages, the
annotation and image licences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from glypharis.data.pages import Page
from glypharis.eval.det_matching import DetMatches
from glypharis.eval.page_metrics import PredLine

HIERTEXT_ATTRIBUTION: Final = "HierText annotations CC BY-SA 4.0, images CC BY 2.0"

RGB = tuple[int, int, int]
GT_TP: Final[RGB] = (40, 170, 60)
GT_FN: Final[RGB] = (255, 140, 0)
GREY: Final[RGB] = (150, 150, 150)
PRED_TP: Final[RGB] = (30, 90, 255)
PRED_FP: Final[RGB] = (230, 20, 20)
BACKGROUND: Final[RGB] = (255, 255, 255)
LEGEND_ROW_PX: Final = 16


def _scaled(points: Sequence[Sequence[float]], f: float) -> list[tuple[float, float]]:
    return [(float(x) * f, float(y) * f) for x, y in points]


def _outline(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], colour: RGB,
             width: int) -> None:  # fmt: skip
    if len(pts) >= 2:
        draw.line([*pts, pts[0]], fill=colour, width=width, joint="curve")


def _dashed(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], colour: RGB, width: int,
            dash: float = 6.0) -> None:  # fmt: skip
    for (x0, y0), (x1, y1) in zip(pts, [*pts[1:], pts[0]], strict=True):
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(length // dash))
        for k in range(0, steps, 2):
            t0, t1 = k / steps, min(1.0, (k + 1) / steps)
            draw.line(
                [(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                 (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)],
                fill=colour, width=width,
            )  # fmt: skip


def draw_overlay(
    rgb: np.ndarray,
    gt: Page,
    preds: Sequence[PredLine],
    m: DetMatches,
    long_side: int = 1024,
) -> Image.Image:
    """The page scaled to ``long_side`` with ground truth and predictions drawn by status.

    ``rgb`` is the page as its polygons expect it (EXIF orientation applied), HWC uint8.
    """
    if len(m.pred_status) != len(preds) or len(m.gt_status) != len(gt.lines):
        raise ValueError("draw_overlay: the matches do not belong to these predictions and lines")
    h, w = rgb.shape[:2]
    f = long_side / max(w, h)
    size = (max(1, round(w * f)), max(1, round(h * f)))
    page = Image.fromarray(np.ascontiguousarray(rgb[..., :3])).resize(
        size, Image.Resampling.BILINEAR
    )
    thin = max(1, round(long_side / 512))
    draw = ImageDraw.Draw(page)

    for region in gt.ignore:
        _outline(draw, _scaled(region, f), GREY, thin)
    # Ground truth first, so predictions stay visible on top of it.
    for line, gt_state in zip(gt.lines, m.gt_status, strict=True):
        pts = _scaled(line.polygon, f)
        if gt_state == "tp":
            _outline(draw, pts, GT_TP, thin)
        elif gt_state == "fn":
            _outline(draw, pts, GT_FN, 2 * thin + 1)
        else:
            _outline(draw, pts, GREY, thin)
    for pred, pred_state in zip(preds, m.pred_status, strict=True):
        pts = _scaled(pred.polygon, f)
        if pred_state == "ignored":
            _dashed(draw, pts, GREY, thin)
        else:
            _outline(draw, pts, PRED_TP if pred_state == "tp" else PRED_FP, thin)

    tp, fp, fn = m.counts()
    entries: list[tuple[RGB, str]] = [
        (GT_TP, f"GT tp {m.gt_status.count('tp')}"),
        (GT_FN, f"GT fn {fn}"),
        (GREY, f"GT don't care {m.gt_status.count('dont_care')}"),
        (PRED_TP, f"pred tp {tp}"),
        (PRED_FP, f"pred fp {fp}"),
        (GREY, f"pred ignored {m.pred_status.count('ignored')}  (IoU {m.thr:g})"),
    ]
    notes = [f"{gt.doc_id}"]
    if gt.source == "hiertext":
        notes.append(HIERTEXT_ATTRIBUTION)
    return _with_legend(page, entries, "   ".join(notes))


def _with_legend(page: Image.Image, entries: list[tuple[RGB, str]], note: str) -> Image.Image:
    font = ImageFont.load_default()
    probe = ImageDraw.Draw(page)
    rows: list[list[tuple[RGB, str]]] = [[]]
    x = 4
    for colour, text in entries:  # wrap the legend to the page width
        item = LEGEND_ROW_PX + int(probe.textlength(text, font=font)) + 12
        if rows[-1] and x + item > page.width:
            rows.append([])
            x = 4
        rows[-1].append((colour, text))
        x += item
    strip_h = LEGEND_ROW_PX * (len(rows) + 1) + 4
    out = Image.new("RGB", (page.width, page.height + strip_h), BACKGROUND)
    out.paste(page, (0, 0))
    draw = ImageDraw.Draw(out)
    y = page.height + 2
    for row in rows:
        x = 4
        for colour, text in row:
            draw.rectangle([x, y + 3, x + 10, y + 13], fill=colour)
            draw.text((x + LEGEND_ROW_PX, y + 2), text, fill=(0, 0, 0), font=font)
            x += LEGEND_ROW_PX + int(draw.textlength(text, font=font)) + 12
        y += LEGEND_ROW_PX
    draw.text((4, y + 2), note, fill=(80, 80, 80), font=font)
    return out


def contact_sheet(tiles: Sequence[Image.Image], cols: int = 3, tile_long: int = 640) -> Image.Image:
    """Tiles scaled to a long side of ``tile_long`` on a grid of ``cols`` columns.

    Each tile keeps its aspect ratio and is centred in a square cell. No tiles gives a single
    labelled empty cell, so an empty slice still produces a sheet.
    """
    if cols < 1 or tile_long < 1:
        raise ValueError("contact_sheet needs cols >= 1 and tile_long >= 1")
    if not tiles:
        sheet = Image.new("RGB", (tile_long, tile_long // 4), BACKGROUND)
        ImageDraw.Draw(sheet).text((8, 8), "no pages", fill=(80, 80, 80))
        return sheet
    n_cols = min(cols, len(tiles))
    n_rows = math.ceil(len(tiles) / n_cols)
    gap = 4
    sheet = Image.new(
        "RGB",
        (n_cols * tile_long + (n_cols + 1) * gap, n_rows * tile_long + (n_rows + 1) * gap),
        BACKGROUND,
    )
    for i, tile in enumerate(tiles):
        f = tile_long / max(tile.width, tile.height)
        t = tile.convert("RGB").resize(
            (max(1, round(tile.width * f)), max(1, round(tile.height * f))),
            Image.Resampling.LANCZOS,
        )
        r, c = divmod(i, n_cols)
        x = gap + c * (tile_long + gap) + (tile_long - t.width) // 2
        y = gap + r * (tile_long + gap) + (tile_long - t.height) // 2
        sheet.paste(t, (x, y))
    return sheet
