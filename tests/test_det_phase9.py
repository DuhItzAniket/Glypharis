"""Phase 9 tools: the post-processing sweep, the error categories, box shape and pairing."""

from __future__ import annotations

import dataclasses
import json
import math

import cv2
import det_analysis
import det_pair
import det_sweep
import numpy as np
from det_fixtures import make_line, rect_polygon

from glypharis.config.schema import DetConfig
from glypharis.data.pages import Page
from glypharis.eval.det_matching import detection_matches
from glypharis.eval.page_metrics import PredLine
from glypharis.models.det.postprocess import Candidate
from glypharis.models.det.preprocess import ResizeInfo
from glypharis.training.det_eval import ceiling_geometry, score_page

LONG_SIDE = 400


def _page() -> Page:
    lines = [
        make_line("a", rect_polygon(120, 60, 160, 20)),
        make_line("b", rect_polygon(120, 120, 160, 14, 3.0)),
        make_line("c", rect_polygon(300, 220, 120, 10)),
        make_line("u", rect_polygon(300, 270, 80, 12), legible=False),
    ]
    return Page("p0", "x.jpg", 400, 300, "detsynth", "validation", lines)


def _map(page: Page) -> np.ndarray:
    """An 8-bit probability map: shrunk kernels of different strengths and a noise blob."""
    info = ceiling_geometry(page, LONG_SIDE)
    levels = np.zeros((info.h, info.w), np.uint8)
    for line, value in zip(page.lines, (240, 170, 120, 200), strict=True):
        pts = np.asarray(line.polygon, np.float32)
        c = pts.mean(axis=0)
        cv2.fillPoly(levels, [np.round(c + 0.6 * (pts - c)).astype(np.int32)], value)
    cv2.circle(levels, (40, 260), 3, 150, -1)
    cv2.circle(levels, (360, 30), 1, 90, -1)
    return levels


def test_sweep_counts_equal_score_page_at_every_setting(tmp_path):
    page = _page()
    levels = _map(page)
    cv2.imwrite(str(tmp_path / "p0.png"), levels)
    det = DetConfig()
    det = dataclasses.replace(det, infer=dataclasses.replace(det.infer, long_side=LONG_SIDE))
    det_sweep._init(json.dumps(dataclasses.asdict(det)), {"p0": page.to_dict()}, LONG_SIDE,
                    str(tmp_path))  # fmt: skip
    doc_id, counts = det_sweep.score_page_grid("p0")
    assert doc_id == "p0" and counts.shape == (len(det_sweep.grid()), 3)
    info = ceiling_geometry(page, LONG_SIDE)
    prob = np.zeros((info.pad_h, info.pad_w), np.float32)
    prob[: info.h, : info.w] = levels / 255.0
    ops = {op.name: op for op in det_sweep.UNCLIP}
    for k, (b, a, op, t) in enumerate(det_sweep.grid()):
        post = dataclasses.replace(det.post, bin_thresh=b, min_component_area=a, box_thresh=t,
                                   operating_points=[ops[op]])  # fmt: skip
        ref = score_page(prob, info, page, "val", dataclasses.replace(det, post=post), False)
        assert tuple(counts[k]) == ref.record.det[op][0.5], (b, a, op, t)
    assert det_sweep.RECORD in det_sweep.grid() and len(det_sweep.grid()) == 440


def _cand(cx: float, cy: float, score: float) -> Candidate:
    return Candidate(score, 12, (cx, cy, 6.0, 2.0, 0.0), (cx, cy, 6.0, 2.0, 0.0), ())


def test_error_categories_follow_the_registered_order():
    lines = [
        make_line("l0", rect_polygon(100, 40, 120, 16)),
        make_line("l1", rect_polygon(100, 60, 120, 16)),  # l0 and l1 under one merged box
        make_line("l2", rect_polygon(300, 40, 120, 16)),  # cut into two short boxes
        make_line("l3", rect_polygon(100, 200, 120, 16)),  # a low-score candidate only
        make_line("l4", rect_polygon(300, 200, 120, 16)),  # nothing
        make_line("l5", rect_polygon(100, 260, 120, 16)),  # one box half its length
    ]
    page = Page("q", "x.jpg", 400, 300, "detsynth", "validation", lines)
    preds = [
        PredLine(rect_polygon(100, 50, 124, 40)),
        PredLine(rect_polygon(268, 40, 48, 16)),
        PredLine(rect_polygon(332, 40, 48, 16)),
        PredLine(rect_polygon(370, 280, 20, 20)),  # on nothing
        PredLine(rect_polygon(80, 260, 70, 22)),
    ]
    info = ResizeInfo(400, 300, 200, 150, 224, 160)  # network at half scale
    d = det_analysis.PageData(page, preds, [0.9] * len(preds), detection_matches(page, preds, 0.5),
                              [_cand(50, 100, 0.4)], info)  # fmt: skip
    fn, fp, merges, splits = det_analysis.classify_page(d, post_box_thresh=0.6)
    assert fn == {0: "merged", 1: "merged", 2: "split", 3: "missed_low_score",
                  4: "missed_no_candidate", 5: "poorly_localised"}  # fmt: skip
    assert fp == {0: "merge", 1: "fragment", 2: "fragment", 3: "background", 4: "fragment"}
    assert (merges, splits) == (1, 1)


def test_curved_lines_need_vertices_and_a_poor_rectangle_fit():
    assert not det_analysis.curved(rect_polygon(100, 100, 200, 20))
    arc = [[100 + 80 * math.cos(t), 100 + 80 * math.sin(t)] for t in np.linspace(3.4, 6.0, 8)]
    arc += [[100 + 60 * math.cos(t), 100 + 60 * math.sin(t)] for t in np.linspace(6.0, 3.4, 8)]
    assert det_analysis.curved(arc)
    straight = [[x, 0.0] for x in range(0, 100, 20)] + [[x, 10.0] for x in range(80, -1, -20)]
    assert not det_analysis.curved(straight)


def test_pairing_adopts_only_a_ci_above_zero():
    rng = np.random.default_rng(3)
    base = {f"d{i}": (int(rng.integers(20, 40)), 3, 10) for i in range(200)}
    better = {d: (tp + 4, fp, fn - 4) for d, (tp, fp, fn) in base.items()}
    res = det_pair.pair(better, base)
    assert res["adopt"] and res["diff"]["low"] > 0 and res["a"]["f1_50"] > res["b"]["f1_50"]
    assert not det_pair.pair(base, base)["adopt"]
