"""DBNet label generation: exact shrink and threshold targets, roles, precedence and demotion."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import shapely
import shapely.affinity
from det_fixtures import make_line, random_quad, rect_polygon
from shapely.geometry import LinearRing, Polygon, box

from glypharis.config.schema import DetLabelConfig
from glypharis.data.det_geometry import min_area_rect
from glypharis.data.det_labels import (
    CARE,
    EMPTY_TEXT,
    HANDWRITTEN,
    ILLEGIBLE,
    UNTRUSTED_BOX,
    VERTICAL,
    LabelStats,
    count_centres,
    line_reason,
    make_labels,
    rasterize,
    segment_distance,
    shrink_distance,
)
from glypharis.eval.page_metrics import _poly, is_box_line

CFG = DetLabelConfig()  # r = 0.4, T in [0.3, 0.7], min_text_size 6


def _arr(poly: Polygon | list[list[float]]) -> np.ndarray:
    if isinstance(poly, Polygon):
        return np.asarray(poly.exterior.coords)[:-1]
    return np.asarray(poly, dtype=np.float64)


def _centres(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:h, 0:w]
    return xs + 0.5, ys + 0.5


def _kernel(poly: np.ndarray, r: float = 0.4) -> shapely.Geometry:
    g = _poly(poly)
    return g.buffer(-shrink_distance(g.area, g.length, r), join_style="mitre")


# ------------------------------------------------------------------------------------- shrink
def test_hand_computed_shrink_of_a_100_by_20_rectangle():
    assert shrink_distance(2000.0, 240.0, 0.4) == pytest.approx(7.0, abs=1e-12)
    lab = make_labels([_arr(box(10, 10, 110, 30))], np.array([True]), [], (60, 140), CFG)
    # Kernel [17, 103] x [17, 23]: centres strictly inside are columns 17..102 and rows 17..22.
    assert int(lab.shrink_map.sum()) == 516
    assert lab.shrink_map[17:23, 17:103].all()
    assert lab.care_kept.tolist() == [True]
    assert lab.stats.care_drawn == 1 and lab.stats.kernel_px == 516
    assert lab.shrink_mask.all()  # no don't-care anywhere
    assert lab.shrink_map.dtype == np.uint8 and lab.thresh_map.dtype == np.float32


def test_threshold_map_values_mask_and_area():
    # Edges on rows and columns of pixel centres, so centres sit exactly on the edge or at D.
    poly = box(10.5, 10.5, 110.5, 30.5)
    lab = make_labels([_arr(poly)], np.array([True]), [], (48, 130), CFG)
    t = lab.thresh_map
    assert t[10, 40] == pytest.approx(0.7, abs=1e-6)  # on the top edge
    assert t[30, 40] == pytest.approx(0.7, abs=1e-6)  # on the bottom edge
    assert t[14, 40] == pytest.approx(0.3 + 0.4 * (1 - 4 / 7), abs=1e-6)  # 4 px inside
    for row in (3, 20, 37):  # D above, at least D inside, D below
        assert t[row, 40] == pytest.approx(0.3, abs=1e-6)
    assert t[0, 0] == pytest.approx(0.3, abs=1e-6)  # outside the mask: thresh_min
    assert t.min() >= 0.3 - 1e-6 and t.max() <= 0.7 + 1e-6

    # Off the centre lattice, the mask's pixel count estimates the dilated area A + L·D + πD².
    generic = box(10.2, 10.3, 110.2, 30.3)
    area = int(make_labels([_arr(generic)], np.array([True]), [], (48, 130), CFG).thresh_mask.sum())
    expected = generic.area + generic.length * 7.0 + np.pi * 49.0
    assert abs(area - expected) / expected < 0.02


def test_threshold_mask_is_the_round_dilation_by_d():
    rng = np.random.default_rng(3)
    for _ in range(5):
        poly = random_quad(rng, (60, 60), height=(10, 20), aspect=(2, 5), max_angle=40)
        poly += 70
        g = _poly(poly)
        d = shrink_distance(g.area, g.length, 0.4)
        lab = make_labels([poly], np.array([True]), [], (200, 200), CFG)
        xs, ys = _centres(200, 200)
        ref = shapely.contains_xy(g.buffer(d, quad_segs=256), xs, ys)
        diff = lab.thresh_mask.astype(bool) != ref
        # Only centres within a hair of the true circle can disagree with its polygon version.
        dist = segment_distance(xs[diff], ys[diff], _arr(g))
        assert np.all(np.abs(dist - d) < 1e-3)
        assert diff.sum() <= 2


# --------------------------------------------------------------------------------------- roles
def test_line_reason_is_care_exactly_for_box_lines():
    flags = itertools.product(
        (True, False), (True, False), (True, False), ("", "  ", "abc"), (None, 0.1, 0.5)
    )
    for legible, handwritten, vertical, text, align in flags:
        line = make_line("x", [[0, 0], [1, 0], [1, 1], [0, 1]], text, legible=legible,
                         handwritten=handwritten, vertical=vertical, align_score=align)  # fmt: skip
        assert (line_reason(line) == CARE) == is_box_line(line)
    poly = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert line_reason(make_line("a", poly, legible=False, handwritten=True)) == ILLEGIBLE
    assert line_reason(make_line("b", poly, handwritten=True)) == HANDWRITTEN
    assert line_reason(make_line("c", poly, vertical=True)) == VERTICAL
    assert line_reason(make_line("d", poly, " ")) == EMPTY_TEXT
    assert line_reason(make_line("e", poly, align_score=0.1)) == UNTRUSTED_BOX


# ---------------------------------------------------------------------------------- precedence
def test_care_wins_over_overlapping_dont_care():
    care = _arr(box(10, 10, 110, 30))
    dont = _arr(box(40, 0, 70, 40))  # crosses the care line
    lab = make_labels([care, dont], np.array([True, False]), [], (50, 130), CFG)
    kernel = lab.shrink_map.astype(bool)
    xs, ys = _centres(50, 130)
    inside_dont = shapely.contains_xy(box(40, 0, 70, 40), xs, ys)
    assert kernel[inside_dont].any()
    assert (lab.shrink_mask[kernel] == 1).all()
    assert (lab.shrink_mask[inside_dont & ~kernel] == 0).all()
    assert (lab.shrink_mask[~inside_dont] == 1).all()
    assert lab.stats.dont_care == 1 and lab.care_kept.tolist() == [True, False]


def test_ignore_regions_zero_both_masks_except_care_kernels():
    care = _arr(box(10, 10, 110, 30))
    ignore = _arr(box(0, 0, 50, 50))
    lab = make_labels([care], np.array([True]), [ignore], (50, 130), CFG)
    xs, ys = _centres(50, 130)
    inside = shapely.contains_xy(box(0, 0, 50, 50), xs, ys)
    kernel = lab.shrink_map.astype(bool)
    assert (kernel & inside).any()
    assert (lab.shrink_mask[kernel] == 1).all()
    assert (lab.shrink_mask[inside & ~kernel] == 0).all()
    assert (lab.thresh_mask[inside] == 0).all()
    assert lab.thresh_mask[~inside].any()
    assert lab.stats.masked_px == int((inside & ~kernel).sum())


# ----------------------------------------------------------------------------------- demotion
def test_small_lines_are_demoted_to_dont_care():
    small = _arr(box(10, 10, 110, 15))  # 5 px short side < min_text_size 6
    lab = make_labels([small], np.array([True]), [], (30, 130), CFG)
    assert lab.stats.demoted_small == 1 and not lab.care_kept[0]
    assert lab.shrink_map.sum() == 0 and lab.thresh_mask.sum() == 0
    xs, ys = _centres(30, 130)
    assert (lab.shrink_mask[shapely.contains_xy(box(10, 10, 110, 15), xs, ys)] == 0).all()


def test_thin_kernels_are_demoted_to_dont_care():
    cfg = DetLabelConfig(min_text_size=0.0)
    thin = _arr(box(10, 10, 210, 15))  # kernel 5·(32 + 5)/205 ≈ 0.9 px thick
    lab = make_labels([thin], np.array([True]), [], (30, 230), cfg)
    assert min_area_rect(np.asarray(_kernel(thin).exterior.coords))[1] < 1.0
    assert lab.stats.demoted_thin_kernel == 1 and not lab.care_kept[0]
    assert lab.shrink_map.sum() == 0 and lab.stats.demoted_small == 0


def test_no_care_line_keeps_mask_one_without_kernel_centres():
    rng = np.random.default_rng(11)
    checked = demoted = 0
    for _ in range(50):
        polys = [random_quad(rng, (120, 120), height=(2, 14), aspect=(1, 12), skew=0.4)
                 for _ in range(10)]  # fmt: skip
        care = rng.random(10) < 0.8
        lab = make_labels(polys, care, [], (120, 120), CFG)
        for poly, kept in zip(polys, lab.care_kept, strict=True):
            if kept:
                k = _kernel(poly)
                assert count_centres(k) > 0  # over the full kernel, not the canvas
                assert min_area_rect(shapely.get_coordinates(k))[1] >= 1.0
                checked += 1
        assert not (lab.care_kept & ~care).any()
        demoted += lab.stats.demoted_small + lab.stats.demoted_thin_kernel
    assert checked > 100 and demoted > 10  # both branches were exercised


# ------------------------------------------------------------------------------- centre count
def test_count_centres_misses_shapes_that_dodge_every_centre():
    # A 1.2 x 1.2 square turned 45° and centred on a pixel corner, i.e. in the middle of a cell of
    # the pixel-centre lattice: its half-diagonal 0.85 is shorter than the 1.0 L1 distance to the
    # nearest centres.
    h = 1.2 / np.sqrt(2)
    diamond = Polygon([(5 - h, 5), (5, 5 - h), (5 + h, 5), (5, 5 + h)])
    assert count_centres(diamond) == 0
    strip = box(3, 10.5, 50, 11.5)  # exactly 1.0 wide, edges on rows of pixel centres
    assert count_centres(strip) == 0
    assert count_centres(box(3, 10.4, 50, 11.4)) == 47


def test_a_line_whose_kernel_is_a_centreless_strip_is_demoted():
    cfg = DetLabelConfig(min_text_size=0.0)
    line = _arr(box(0, 8.5, 100, 13.5))  # D = 2.0 exactly: kernel [2, 98] x [10.5, 11.5]
    k = _kernel(line)
    assert k.bounds == pytest.approx((2.0, 10.5, 98.0, 11.5))
    assert min_area_rect(np.asarray(k.exterior.coords))[1] >= 1.0  # not thin by thickness
    lab = make_labels([line], np.array([True]), [], (24, 100), cfg)
    assert lab.stats.demoted_thin_kernel == 1 and not lab.care_kept[0]
    # The same line cut by a small canvas is demoted the same way.
    cut = make_labels([line - [60, 0]], np.array([True]), [], (24, 20), cfg)
    assert cut.stats.demoted_thin_kernel == 1


def test_count_centres_ignores_integer_shifts_and_canvas_cuts():
    rng = np.random.default_rng(5)
    for _ in range(100):
        g = _poly(random_quad(rng, (50, 50), height=(1, 6), aspect=(1, 8)))
        dx, dy = (int(v) for v in rng.integers(-40, 40, 2))
        assert count_centres(g) == count_centres(shapely.affinity.translate(g, dx, dy))
    # A line mostly off the canvas keeps its kernel target on the part that is on it.
    lab = make_labels([_arr(box(-80, 10, 20, 30))], np.array([True]), [], (40, 40), CFG)
    assert lab.care_kept[0] and lab.shrink_map[:, :13].any()


# ------------------------------------------------------------------------------- rasterisation
def test_rasterize_equals_contains_xy_on_pixel_centres():
    rng = np.random.default_rng(7)
    xs, ys = _centres(70, 90)
    for _ in range(20):
        g = _poly(random_quad(rng, (90, 70), height=(3, 30), aspect=(1, 6), skew=0.5))
        canvas = np.zeros((70, 90), np.uint8)
        n = rasterize(canvas, g, 3)
        ref = shapely.contains_xy(g, xs, ys)
        assert np.array_equal(canvas == 3, ref) and n == int(ref.sum())


def test_labels_of_a_crop_are_crops_of_the_full_labels():
    rng = np.random.default_rng(13)
    full_hw, crop_hw = (160, 200), (64, 96)
    for trial in range(10):
        polys = [random_quad(rng, (200, 160), height=(6, 24), aspect=(1, 10)) for _ in range(12)]
        care = rng.random(12) < 0.75
        ignore = [random_quad(rng, (200, 160), height=(10, 30), aspect=(1, 2))]
        oy, ox = int(rng.integers(0, 96)), int(rng.integers(0, 104))
        full = make_labels(polys, care, ignore, full_hw, CFG)
        o = np.array([ox, oy], dtype=np.float64)
        crop = make_labels([p - o for p in polys], care, [ignore[0] - o], crop_hw, CFG)
        win = np.s_[oy : oy + crop_hw[0], ox : ox + crop_hw[1]]
        for key in ("shrink_map", "shrink_mask", "thresh_mask"):
            assert np.array_equal(getattr(crop, key), getattr(full, key)[win]), (trial, key)
        np.testing.assert_allclose(crop.thresh_map, full.thresh_map[win], atol=1e-6)


def test_segment_distance_matches_shapely():
    rng = np.random.default_rng(17)
    for _ in range(10):
        ring = rng.uniform(0, 50, (int(rng.integers(3, 9)), 2))
        xs, ys = rng.uniform(-20, 70, 500), rng.uniform(-20, 70, 500)
        ref = shapely.distance(shapely.points(xs, ys), LinearRing(ring))
        np.testing.assert_allclose(segment_distance(xs, ys, ring), ref, rtol=0, atol=1e-9)


def test_label_stats_add_sums_every_field():
    a = LabelStats(1, 2, 3, 4, 5, 6, 7)
    a.add(LabelStats(10, 20, 30, 40, 50, 60, 70))
    assert a == LabelStats(11, 22, 33, 44, 55, 66, 77)


def test_empty_inputs_give_all_negative_labels():
    lab = make_labels([], np.zeros(0, bool), [], (16, 24), CFG)
    assert lab.shrink_map.sum() == 0 and lab.shrink_mask.all() and lab.thresh_mask.sum() == 0
    assert np.allclose(lab.thresh_map, 0.3) and lab.care_kept.shape == (0,)
    with pytest.raises(ValueError, match="care flags"):
        make_labels([_arr(rect_polygon(8, 8, 10, 4))], np.zeros(2, bool), [], (16, 16), CFG)
