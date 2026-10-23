from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely
import torch
from det_fixtures import make_line, rect_polygon
from shapely import affinity
from shapely.geometry import Polygon

from glypharis.config.schema import DetConfig, DetLabelConfig, DetModelConfig, DetPostConfig
from glypharis.data.crops import line_quad
from glypharis.data.det_labels import make_labels
from glypharis.data.pages import Page
from glypharis.eval.page_metrics import evaluate_page
from glypharis.models.det.dbnet import build_detector
from glypharis.models.det.detector import (
    RELEASE_FORMAT,
    DetectorSpec,
    TextLineDetector,
    export_detector,
)
from glypharis.models.det.postprocess import (
    Candidate,
    PostCounters,
    boxes_from_candidates,
    find_candidates,
    order_quad,
    text_angle,
    unclip_distance,
)
from glypharis.models.det.preprocess import ResizeInfo, prepare_page, scaled_size
from glypharis.training.checkpoint import save_checkpoint

R = 0.4
POST = DetPostConfig()
OPS = {op.name: op for op in POST.operating_points}


def _info(w: int, h: int) -> ResizeInfo:
    """Identity geometry: a page already at network scale."""
    return ResizeInfo(w, h, w, h, -(-w // 32) * 32, -(-h // 32) * 32)


def _raster(geom: Polygon, h: int, w: int) -> np.ndarray:
    """1.0 where the pixel centre is strictly inside ``geom`` (the label rasterisation rule)."""
    out = np.zeros((h, w), np.float32)
    x0, y0, x1, y1 = geom.bounds
    xa, ya = max(math.floor(x0), 0), max(math.floor(y0), 0)
    xb, yb = min(math.ceil(x1), w), min(math.ceil(y1), h)
    if xb > xa and yb > ya:
        ys, xs = np.mgrid[ya:yb, xa:xb]
        out[ya:yb, xa:xb] = shapely.contains_xy(geom, xs + 0.5, ys + 0.5)
    return out


def _kernel(poly: Polygon) -> Polygon:
    """DBNet's training shrink: mitre offset by D = A(1 - r²)/L."""
    return poly.buffer(-poly.area * (1 - R * R) / poly.length, join_style="mitre")


def _prob(geoms: list[Polygon], info: ResizeInfo) -> np.ndarray:
    prob = np.zeros((info.pad_h, info.pad_w), np.float32)
    for g in geoms:
        prob[: info.h, : info.w] = np.maximum(prob[: info.h, : info.w], _raster(g, info.h, info.w))
    return prob


def _gt_box(h: float, aspect: float, angle: float, offset: tuple[float, float] = (0.3, 0.6),
            margin: float = 30.0) -> Polygon:  # fmt: skip
    gt = affinity.rotate(shapely.box(0, 0, h * aspect, h), angle, origin=(0, 0))
    x0, y0, *_ = gt.bounds
    return affinity.translate(gt, margin + offset[0] - x0, margin + offset[1] - y0)


def _detect(prob: np.ndarray, info: ResizeInfo, op: str, post: DetPostConfig = POST):
    cands, found = find_candidates(prob, info, post)
    boxes, counted = boxes_from_candidates(cands, OPS[op], post, R, info)
    found.add(counted)
    return boxes, found


def _iou(a: Polygon, b: Polygon) -> float:
    return a.intersection(b).area / a.union(b).area


def _round_trip(gt: Polygon, op: str, box_type: str = "quad") -> float:
    info = _info(int(gt.bounds[2] + 40), int(gt.bounds[3] + 40))
    boxes, _ = _detect(_prob([_kernel(gt)], info), info, op, DetPostConfig(box_type=box_type))
    assert len(boxes) == 1
    return _iou(Polygon(boxes[0].polygon), gt)


# ----------------------------------------------------------------------------- unclip distance


def test_inverse_unclip_matches_the_hand_computed_example():
    # A 100x20 line shrinks by D = 7 to an 86x6 kernel (A = 516, L = 184).
    assert unclip_distance("inverse", 516.0, 184.0, 1.0, R) == pytest.approx(7.0, rel=1e-12)
    assert unclip_distance("dbnet", 516.0, 184.0, 1.5, R) == pytest.approx(1.5 * 516 / 184)
    assert unclip_distance("inverse", 0.0, 0.0, 1.0, R) == 0.0


def test_inverse_unclip_is_exact_for_shrunk_rectangles_at_any_angle():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.uniform(4, 400), rng.uniform(4, 60)
        rect = affinity.rotate(shapely.box(0, 0, a, b), rng.uniform(-90, 90))
        d_s = rect.area * (1 - R * R) / rect.length
        k = rect.buffer(-d_s, join_style="mitre")
        d = unclip_distance("inverse", k.area, k.length, 1.0, R)
        assert abs(d - d_s) / d_s < 1e-9


def test_inverse_unclip_is_close_on_skewed_parallelograms():
    # Exact only for rectangles: the mitre corner term of a skewed shape is not 4D².
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(500):
        a, b = rng.uniform(20, 200), rng.uniform(8, 40)
        shift = b * math.tan(math.radians(rng.uniform(-30, 30)))
        para = Polygon([(0, 0), (a, 0), (a + shift, b), (shift, b)])
        d_s = para.area * (1 - R * R) / para.length
        k = para.buffer(-d_s, join_style="mitre")
        if k.is_empty:
            continue
        worst = max(worst, abs(unclip_distance("inverse", k.area, k.length, 1.0, R) - d_s) / d_s)
    assert 0 < worst < 0.07


# ------------------------------------------------------------------------------- round trips


@pytest.mark.parametrize("h", [16, 32, 64])
@pytest.mark.parametrize("aspect", [1, 4, 10, 30])
def test_round_trip_from_gt_kernels(h, aspect):
    # Axis-aligned kernels are quantised to whole rows, so 0° is judged over sub-pixel offsets.
    offsets = [(0.1, 0.1), (0.35, 0.35), (0.6, 0.6), (0.85, 0.85)]
    straight = [_round_trip(_gt_box(h, aspect, 0, o), "inverse10") for o in offsets]
    assert np.mean(straight) >= 0.85
    for angle in (7, 15, 30):
        iou = _round_trip(_gt_box(h, aspect, angle), "inverse10")
        # A square's area rectangle takes the contour rectangle's axis; both its sides are
        # amplified by the inverse, so it is held to the axis-aligned bound.
        assert iou >= (0.95 if aspect >= 4 else 0.85), (angle, iou)
    if aspect == 30:  # the documented weakness of DBNet's ratio on thin r = 0.4 kernels
        assert _round_trip(_gt_box(h, aspect, 0), "dbnet15") < 0.6
        assert _round_trip(_gt_box(h, aspect, 15), "dbnet15") < 0.6


def test_area_rectangle_width_is_unbiased():
    rng = np.random.default_rng(2)
    errors = []
    for _ in range(200):
        width, length, angle = rng.uniform(2, 8), rng.uniform(80, 200), rng.uniform(0, 45)
        band = _gt_box(width, length / width, angle, tuple(rng.uniform(0, 1, 2)), margin=10)
        info = _info(int(band.bounds[2] + 10), int(band.bounds[3] + 10))
        cands, _ = find_candidates(_prob([band], info), info, DetPostConfig(min_component_area=1))
        assert len(cands) == 1
        errors.append(cands[0].area_rect[3] - width)
        assert cands[0].area_rect[2] * cands[0].area_rect[3] == pytest.approx(cands[0].area_px)
    assert abs(np.mean(errors)) < 0.05


# ------------------------------------------------------------------------ coordinate mapping


@pytest.mark.parametrize(("orig", "net"), [((1000, 700), (1280, 896)), ((2000, 1500), (1280, 960))])
def test_known_boxes_map_back_to_the_page(orig, net):
    assert scaled_size(*orig, 1280) == net
    info = ResizeInfo(orig[0], orig[1], net[0], net[1], net[0], net[1])
    # Sizes whose shrink distance is a whole pixel (200x40: D = 14, 100x20: D = 7), at whole-pixel
    # positions, so the rasterised kernel is exact and the inverse recovers the box exactly.
    boxes_net = [(100.0, 50.0, 300.0, 90.0), (500.0, 300.0, 600.0, 320.0)]
    prob = _prob([_kernel(shapely.box(*b)) for b in boxes_net], info)
    boxes, _ = _detect(prob, info, "inverse10")
    assert len(boxes) == 2
    for box, (x0, y0, x1, y1) in zip(boxes, boxes_net, strict=True):
        expected = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) / (info.sx, info.sy)
        assert np.abs(np.array(box.quad) - expected).max() < 1.0
        assert np.abs(np.array(box.polygon) - expected).max() < 1.0


def test_probability_in_the_padding_never_produces_candidates():
    info = ResizeInfo(300, 200, 300, 200, 320, 224)
    prob = np.zeros((224, 320), np.float32)
    prob[200:, :] = 1.0
    prob[:, 300:] = 1.0
    cands, counters = find_candidates(prob, info, POST)
    assert cands == [] and counters.components == 0


def test_boxes_are_clipped_to_the_image():
    info = _info(200, 100)
    line = shapely.box(-20, 80, 120, 110)  # runs off the left and bottom edges
    boxes, _ = _detect(_prob([_kernel(line)], info), info, "inverse10")
    assert len(boxes) == 1
    poly = Polygon(boxes[0].polygon)
    image = shapely.box(0, 0, 200, 100)
    assert image.buffer(1e-9).contains(poly)
    x0, y0, x1, y1 = boxes[0].bbox
    assert x0 == 0.0 and y1 == 100.0 and x1 <= 200 and y0 >= 0
    assert poly.area == pytest.approx(image.intersection(poly).area)


# ------------------------------------------------------------------------- scores and counters


def test_score_is_the_exact_component_mean():
    rng = np.random.default_rng(3)
    info = _info(64, 64)
    prob = np.zeros((64, 64), np.float32)
    patch = rng.uniform(0.31, 1.0, size=(10, 30)).astype(np.float32)
    prob[20:30, 10:40] = patch
    cands, _ = find_candidates(prob, info, POST)
    assert len(cands) == 1
    assert cands[0].score == pytest.approx(float(patch.astype(np.float64).mean()), rel=1e-12)
    assert cands[0].area_px == 300


@pytest.mark.parametrize(
    "post",
    [
        DetPostConfig(max_candidates=5),
        DetPostConfig(min_component_area=1, min_box_side=0.0, box_thresh=0.5),
        DetPostConfig(min_component_area=1, min_box_side=6.0, box_type="poly"),
    ],
)
def test_counters_account_for_every_component(post):
    rng = np.random.default_rng(4)
    info = _info(160, 128)
    prob = (rng.uniform(0, 1, (128, 160)) ** 3).astype(np.float32)  # speckle of every size
    seen: dict[str, int] = {}
    for op in post.operating_points:
        cands, c = find_candidates(prob, info, post)
        boxes, counted = boxes_from_candidates(cands, op, post, R, info)
        c.add(counted)
        total = c.tiny + c.truncated + c.low_score + c.too_thin + c.outside + c.kept
        assert c.components == total > 0
        assert c.kept == len(boxes) and c.poly_fallback <= c.kept
        for key, value in c.to_dict().items():
            seen[key] = seen.get(key, 0) + value
    assert seen["low_score"] > 0 and seen["kept"] > 0
    if post.max_candidates == 5:
        assert seen["truncated"] > 0 and seen["tiny"] > 0
    if post.min_box_side == 0.0:
        assert seen["outside"] > 0  # single-pixel kernels give zero-area dbnet boxes
    if post.min_box_side == 6.0:
        assert seen["too_thin"] > 0


def test_post_counters_add_and_to_dict():
    a, b = PostCounters(components=3, kept=2, tiny=1), PostCounters(components=1, low_score=1)
    a.add(b)
    assert a.to_dict() == {"components": 4, "tiny": 1, "truncated": 0, "low_score": 1,
                           "too_thin": 0, "outside": 0, "kept": 2, "poly_fallback": 0}  # fmt: skip


# ---------------------------------------------------------------------------------- filters


def test_box_thresh_filters_on_the_mean_score():
    info = _info(64, 64)
    prob = np.zeros((64, 64), np.float32)
    prob[20:30, 10:50] = 0.5
    assert _detect(prob, info, "inverse10")[1].low_score == 1
    boxes, counters = _detect(prob, info, "inverse10", DetPostConfig(box_thresh=0.4))
    assert len(boxes) == 1 and counters.low_score == 0


def test_min_box_side_drops_thin_boxes():
    info = _info(64, 32)
    prob = np.zeros((32, 64), np.float32)
    prob[10, 10:50] = 1.0  # a one-row kernel: the inverse grows it to about 4.3 px
    boxes, counters = _detect(prob, info, "inverse10", DetPostConfig(min_box_side=5.0))
    assert boxes == [] and counters.too_thin == 1
    assert len(_detect(prob, info, "inverse10")[0]) == 1
    # Collinear pixel centres have a zero-width contour rectangle, so dbnet adds nothing.
    assert _detect(prob, info, "dbnet15")[1].too_thin == 1


def test_max_candidates_keeps_the_largest_components():
    info = _info(96, 32)
    prob = np.zeros((32, 96), np.float32)
    prob[4:7, 4:8] = 1.0  # 12 px
    prob[4:10, 30:35] = 1.0  # 30 px
    prob[4:8, 60:65] = 1.0  # 20 px
    cands, counters = find_candidates(prob, info, DetPostConfig(max_candidates=2))
    assert sorted(c.area_px for c in cands) == [20, 30]
    assert counters.truncated == 1 and counters.components == 3


def test_empty_and_all_ones_maps():
    info = ResizeInfo(150, 100, 150, 100, 160, 128)
    boxes, counters = _detect(np.zeros((128, 160), np.float32), info, "dbnet15")
    assert boxes == [] and counters == PostCounters()
    ones = np.ones((128, 160), np.float32)
    for op in OPS:
        for box_type in ("quad", "poly"):
            boxes, counters = _detect(ones, info, op, DetPostConfig(box_type=box_type))
            assert len(boxes) == 1 and counters.kept == 1
            assert Polygon(boxes[0].polygon).area == pytest.approx(150 * 100)
            assert boxes[0].bbox == (0.0, 0.0, 150.0, 100.0)


# ------------------------------------------------------------------------------------ quads


def test_order_quad_orders_corners_of_a_horizontal_box():
    rect = np.array([[50, 30], [10, 30], [10, 20], [50, 20]], dtype=float)  # any start, any turn
    quad = order_quad(rect)
    np.testing.assert_allclose(quad, [[10, 20], [50, 20], [50, 30], [10, 30]], atol=1e-9)
    x, y = quad[:, 0], quad[:, 1]
    # Clockwise on screen with y down: a positive shoelace area in image coordinates.
    assert 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) > 0


def test_perfect_boxes_score_one_in_evaluate_page():
    info = _info(640, 480)
    polys = [
        rect_polygon(150, 60, 220, 24),
        rect_polygon(400, 150, 300, 30, 12.0),
        rect_polygon(200, 300, 260, 40, -20.0),
        rect_polygon(480, 380, 180, 20, 5.0),
    ]
    prob = _prob([_kernel(Polygon(p)) for p in polys], info)
    page = Page("p", "x.jpg", 640, 480, "detsynth", "validation",
                [make_line(f"l{i}", p) for i, p in enumerate(polys)])  # fmt: skip
    # The inverse recovers a line from its exact kernel (dbnet15 does not on thin r = 0.4 kernels).
    boxes, _ = _detect(prob, info, "inverse10")
    res = evaluate_page(page, [b.to_pred_line() for b in boxes], (0.5, 0.7), score_text=False)
    assert res.det == {0.5: (4, 0, 0), 0.7: (4, 0, 0)}
    assert [b.bbox[1] for b in boxes] == sorted(b.bbox[1] for b in boxes)


# --------------------------------------------------------------------------- text direction


def _corners(w: float, h: float, angle: float = 0.0, cx: float = 50.0, cy: float = 50.0):
    return np.array(rect_polygon(cx, cy, w, h, angle))


def test_near_square_upright_box_reads_horizontally():
    quad = order_quad(_corners(12, 16))  # long axis vertical, aspect 1.33
    assert text_angle(quad) == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(quad[0], [44, 42], atol=1e-9)  # TL is the top-left corner


def test_tall_box_at_aspect_three_follows_its_long_axis():
    quad = order_quad(_corners(10, 30))
    assert text_angle(quad) == pytest.approx(90.0)
    assert math.dist(quad[0], quad[1]) == pytest.approx(30.0)


@pytest.mark.parametrize("angle", [30.0, 60.0, -30.0])
def test_rotated_line_follows_its_long_axis(angle):
    assert text_angle(order_quad(_corners(50, 10, angle))) == pytest.approx(angle)


def test_direction_overrides_the_box_rule():
    quad = order_quad(_corners(10, 30), direction_deg=0.0)
    assert text_angle(quad) == pytest.approx(0.0, abs=1e-9)
    assert math.dist(quad[0], quad[1]) == pytest.approx(10.0)
    rotated = order_quad(_corners(50, 10, 30.0), direction_deg=-60.0)
    assert text_angle(rotated) == pytest.approx(-60.0)


def test_text_angle_of_a_downward_hiertext_quad_is_90():
    polygon = [[30.0, 10.0], [30.0, 90.0], [18.0, 90.0], [18.0, 10.0]]  # TL, TR, BR, BL
    assert text_angle(line_quad(polygon)) == pytest.approx(90.0)
    with pytest.raises(ValueError):
        text_angle(polygon[:3])
    with pytest.raises(ValueError):
        text_angle([[0, 0], [10, 0], [0, 0], [10, 0]])


def test_det_box_aspect_and_long_axis_angle_match_the_rectangle():
    info = _info(400, 300)
    gt = Polygon(rect_polygon(200, 150, 240, 30, 20.0))
    boxes, _ = _detect(_prob([_kernel(gt)], info), info, "inverse10")
    (box,) = boxes
    assert box.aspect == pytest.approx(8.0, rel=0.05)
    assert box.long_axis_angle == pytest.approx(20.0, abs=0.5)
    assert text_angle(box.quad) == pytest.approx(20.0, abs=0.5)
    assert len(box.polygon) >= 4


# ----------------------------------------------------------------------------------- box type


@pytest.mark.parametrize("h", [12, 16, 32, 48])
@pytest.mark.parametrize("aspect", [10, 20])
def test_poly_and_quad_agree_on_straight_kernels(h, aspect):
    # Small slants matter most: there a 1 px simplified contour is furthest from its rectangle.
    for angle, offset in itertools.product((0, 1, 2, 3, 5, 7, 15, 30), ((0.3, 0.6), (0.1, 0.85))):
        gt = _gt_box(h, aspect, angle, offset)
        info = _info(int(gt.bounds[2] + 40), int(gt.bounds[3] + 40))
        labels = make_labels([np.asarray(gt.exterior.coords)[:-1]], np.ones(1, bool), [],
                             (info.pad_h, info.pad_w), DetLabelConfig())  # fmt: skip
        prob = labels.shrink_map.astype(np.float32)
        for op in OPS:
            quads = _detect(prob, info, op)[0]
            polys, counters = _detect(prob, info, op, DetPostConfig(box_type="poly"))
            if counters.poly_fallback or not quads:
                # A kernel two pixel rows thick has a two-point centre contour: the poly box
                # falls back to the quad, and dbnet15 drops it as too thin.
                assert polys == quads and h == 12
                continue
            (quad,), (poly,) = quads, polys
            # The poly box is the quad with round corners, which cost about 0.2/aspect of its
            # area, so both share one minimum-area rectangle.
            iou = _iou(Polygon(quad.polygon), Polygon(poly.polygon))
            assert iou >= 0.97, (op, angle, offset, iou)
            np.testing.assert_allclose(poly.quad, quad.quad, atol=1e-3)


def _arc_polygon() -> list[list[float]]:
    """A curved, densified line (top edge left to right, then the bottom edge back), 40 px thick."""
    cx, cy = 400.0, 560.0
    t = np.radians(np.linspace(120, 60, 25))
    top = [[cx + 420 * math.cos(a), cy - 420 * math.sin(a)] for a in t]
    bottom = [[cx + 380 * math.cos(a), cy - 380 * math.sin(a)] for a in t[::-1]]
    return top + bottom


def test_poly_follows_a_curved_line_that_the_quad_cannot():
    arc = _arc_polygon()
    info = _info(800, 600)
    labels = make_labels([np.array(arc)], np.ones(1, bool), [], (info.pad_h, info.pad_w),
                         DetLabelConfig())  # fmt: skip
    prob = labels.shrink_map.astype(np.float32)
    gt = Polygon(arc)
    (quad,) = _detect(prob, info, "inverse10")[0]
    (poly,), counters = _detect(prob, info, "inverse10", DetPostConfig(box_type="poly"))
    assert counters.poly_fallback == 0 and len(poly.polygon) > 4
    poly_iou, quad_iou = _iou(Polygon(poly.polygon), gt), _iou(Polygon(quad.polygon), gt)
    assert poly_iou >= 0.8 and poly_iou > quad_iou
    assert len(poly.quad) == 4


def test_short_contour_falls_back_to_the_quad():
    info = _info(64, 64)
    prob = np.zeros((64, 64), np.float32)
    prob[20:28, 10:50] = 1.0
    (cand,), _ = find_candidates(prob, info, POST)
    short = dataclasses.replace(cand, contour=cand.contour[:3])
    poly_post = DetPostConfig(box_type="poly")
    for op in OPS.values():
        boxes, counters = boxes_from_candidates([short], op, poly_post, R, info)
        assert counters.poly_fallback == 1 and counters.kept == 1
        assert boxes == boxes_from_candidates([short], op, POST, R, info)[0]


def test_non_convex_poly_cut_by_the_border_keeps_its_largest_part():
    info = _info(200, 100)
    # A U whose bottom bar lies below the image: clipping leaves its two legs, the right one longer.
    u = ((20.0, 60.0), (30.0, 60.0), (30.0, 130.0), (100.0, 130.0), (100.0, 40.0), (110.0, 40.0),
         (110.0, 140.0), (20.0, 140.0))  # fmt: skip
    cand = Candidate(0.9, 2500, (65.0, 90.0, 90.0, 100.0, 0.0), (65.0, 90.0, 90.0, 27.8, 0.0), u)
    post = DetPostConfig(box_type="poly", min_box_side=0.0)
    (box,), counters = boxes_from_candidates([cand], OPS["dbnet15"], post, R, info)
    assert counters.poly_fallback == 0
    # The same candidate on a page tall enough to hold the whole offset U.
    (whole,), _ = boxes_from_candidates([cand], OPS["dbnet15"], post, R, _info(200, 200))
    assert len(whole.polygon) > 8  # a U, not the fallback rectangle
    parts = list(Polygon(whole.polygon).intersection(shapely.box(0, 0, 200, 100)).geoms)
    assert len(parts) == 2
    largest = max(parts, key=lambda p: p.area)
    assert _iou(Polygon(box.polygon), largest) > 1 - 1e-9
    assert box.bbox[0] > 80  # the right leg


def test_boxes_from_saved_candidates_equal_the_direct_output():
    info = _info(800, 600)
    polys = [_arc_polygon(), rect_polygon(150, 80, 200, 30, 8.0), rect_polygon(600, 500, 120, 24)]
    labels = make_labels([np.array(p) for p in polys], np.ones(3, bool), [],
                         (info.pad_h, info.pad_w), DetLabelConfig())  # fmt: skip
    prob = labels.shrink_map.astype(np.float32)
    cands, _ = find_candidates(prob, info, POST)
    assert len(cands) == 3
    saved = json.loads(json.dumps([c.to_dict() for c in cands]))
    restored = [Candidate.from_dict(d) for d in saved]
    assert restored == cands
    for box_type in ("quad", "poly"):
        post = DetPostConfig(box_type=box_type)
        for op in OPS.values():
            direct = boxes_from_candidates(cands, op, post, R, info)
            assert boxes_from_candidates(restored, op, post, R, info) == direct


def test_postprocess_stays_torch_free(repo_root):
    code = "import sys, glypharis.models.det.postprocess; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------------------- detector API


def _spec(op: str = "inverse10", long_side: int = 128) -> DetectorSpec:
    det = DetConfig()
    det.infer.long_side = long_side
    return DetectorSpec(det.model, det.labels, det.infer, det.post, op, 1)


class _KernelBackend:
    """A stand-in network: the label kernels of known page lines at the prepared geometry."""

    def __init__(self, polygons: list[list[list[float]]], info: ResizeInfo) -> None:
        self.polygons, self.info = polygons, info

    def __call__(self, image: np.ndarray) -> np.ndarray:
        assert image.shape == (self.info.pad_h, self.info.pad_w, 3) and image.dtype == np.uint8
        scaled = [np.array(p) * (self.info.sx, self.info.sy) for p in self.polygons]
        return _prob([_kernel(Polygon(p)) for p in scaled], self.info)


def test_text_line_detector_chains_preprocess_backend_and_postprocess():
    page = np.full((300, 400, 3), 255, np.uint8)
    polygons = [rect_polygon(150, 60, 220, 30), rect_polygon(250, 200, 260, 40, 10.0)]
    spec = _spec(long_side=320)
    _, info = prepare_page(page, 320)
    det = TextLineDetector(_KernelBackend(polygons, info), spec)
    prob, got_info = det.prob_map(page)
    assert prob.shape == (info.pad_h, info.pad_w) and got_info == info
    boxes = det.detect(page)
    assert len(boxes) == 2
    for box, poly in zip(boxes, polygons, strict=True):
        assert _iou(Polygon(box.polygon), Polygon(poly)) > 0.9


def test_detector_spec_round_trip_and_checks():
    spec = _spec("dbnet15")
    assert spec.op().unclip_mode == "dbnet"
    d = spec.to_dict()
    assert json.loads(json.dumps(d)) == d  # plain data only
    assert DetectorSpec.from_dict(d) == spec
    with pytest.raises(ValueError, match="not one of"):
        dataclasses.replace(spec, operating_point="nope")


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    det = DetConfig()
    det.model = DetModelConfig("mobilenet_v3_small", "scratch", None, None, 16, False)
    det.infer.long_side = 96
    torch.manual_seed(0)
    model = build_detector(det.model, Path("unused"))
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    spec = DetectorSpec(det.model, det.labels, det.infer, det.post, "inverse10", 1)
    save_checkpoint(
        path,
        {
            "model": model.state_dict(),
            "config": {"experiment_id": "GW-DET-TEST", "det": dataclasses.asdict(det)},
            "config_hash": "0" * 64,
            "selected_op": "dbnet15",
            "spec": spec.to_dict(),  # stored at another point than selected_op on purpose
            "state": {"epoch": 3},
            "git": {"commit": "abc", "branch": "x", "dirty": False},
            "rng": {"numpy": np.random.get_state()},  # what makes weights_only refuse it
        },
    )
    return path


def test_from_checkpoint_defaults_to_the_selected_operating_point(tiny_checkpoint):
    det = TextLineDetector.from_checkpoint(tiny_checkpoint)
    assert det.spec.operating_point == "dbnet15"
    other = TextLineDetector.from_checkpoint(tiny_checkpoint, operating_point="inverse10")
    assert other.spec.operating_point == "inverse10"
    prob, info = det.prob_map(np.zeros((60, 90, 3), np.uint8))
    assert prob.shape == (info.pad_h, info.pad_w) and prob.dtype == np.float32
    assert info.w == 96


def test_release_file_matches_the_checkpoint(tiny_checkpoint, tmp_path):
    out = export_detector(tiny_checkpoint, tmp_path / "det.release.pt")
    payload = torch.load(out, weights_only=True)
    assert payload["format"] == RELEASE_FORMAT
    assert payload["spec"]["operating_point"] == "dbnet15"
    assert payload["experiment_id"] == "GW-DET-TEST" and payload["epoch"] == 3
    assert payload["git_commit"] == "abc" and payload["config_hash"] == "0" * 64
    assert len(payload["source_sha256"]) == 64
    rng = np.random.default_rng(5)
    page = rng.integers(0, 256, (70, 90, 3), dtype=np.uint8)
    released = TextLineDetector.from_release(out)
    trained = TextLineDetector.from_checkpoint(tiny_checkpoint)
    assert released.spec == trained.spec
    np.testing.assert_array_equal(released.prob_map(page)[0], trained.prob_map(page)[0])
    assert released.detect(page) == trained.detect(page)


def test_from_release_refuses_a_training_checkpoint(tiny_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="not a detector release file"):
        TextLineDetector.from_release(tiny_checkpoint)
    other = tmp_path / "other.pt"
    torch.save({"format": "something-else"}, other)
    with pytest.raises(ValueError, match=RELEASE_FORMAT):
        TextLineDetector.from_release(other)
