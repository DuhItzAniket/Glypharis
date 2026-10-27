"""Crop geometry: the half-pixel convention, warp consistency, decoding and channel order."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import shapely
from det_fixtures import rect_polygon, write_colour_png, write_exif6_jpeg
from PIL import Image

from glypharis.config.schema import DetDataConfig
from glypharis.data.det_geometry import (
    Affine,
    View,
    _reduce_factor,
    decode_page,
    min_area_rect,
    render_view,
    sample_view,
)
from glypharis.data.det_labels import rasterize
from glypharis.data.pages import Page, open_page_image
from glypharis.models.det.preprocess import PAD_RGB


def _rotation(scale: float, angle_deg: float, tx: float, ty: float) -> Affine:
    t = math.radians(angle_deg)
    c, s = scale * math.cos(t), scale * math.sin(t)
    return Affine(np.array([[c, -s, tx], [s, c, ty]]))


def _page(path: Path, root: Path, wh: tuple[int, int]) -> Page:
    return Page("p", path.relative_to(root).as_posix(), wh[0], wh[1], "detsynth", "validation")


# -------------------------------------------------------------------------------------- Affine
def test_affine_apply_compose_and_scale():
    a = _rotation(1.5, 20, 3.0, -2.0)
    b = Affine.scale(2.0, 0.5)
    pts = np.array([[0.0, 0.0], [10.0, 4.0], [-3.0, 7.5]])
    np.testing.assert_allclose(a.compose(b).apply(pts), a.apply(b.apply(pts)), atol=1e-12)
    np.testing.assert_allclose(Affine.scale(1, 1).apply(pts), pts)
    # Scaling by k in edge coordinates moves pixel centres by (k − 1)/2 in index coordinates.
    np.testing.assert_allclose(Affine.scale(2, 2).cv2_matrix(), [[2, 0, 0.5], [0, 2, 0.5]])
    with pytest.raises(ValueError):
        Affine(np.eye(3))


def test_cv2_matrix_puts_a_dot_at_the_transformed_edge_coordinate():
    # A small Gaussian dot sampled at pixel centres, so its intensity centroid is exact.
    x0, y0, sigma = 30.3, 20.7, 1.2
    ys, xs = np.mgrid[0:64, 0:64]
    dot = np.exp(-((xs + 0.5 - x0) ** 2 + (ys + 0.5 - y0) ** 2) / (2 * sigma**2))
    img = (dot * 250).astype(np.float32)
    for aff in (_rotation(1.0, 0, 7.0, 3.0), _rotation(1.3, 25, 10.0, -12.0),
                _rotation(0.8, -15, 20.0, 18.0)):  # fmt: skip
        out = cv2.warpAffine(img, aff.cv2_matrix(), (96, 96), flags=cv2.INTER_LINEAR)
        oy, ox = np.mgrid[0:96, 0:96]
        cx = float((out * (ox + 0.5)).sum() / out.sum())
        cy = float((out * (oy + 0.5)).sum() / out.sum())
        tx, ty = aff.apply(np.array([[x0, y0]]))[0]
        assert abs(cx - tx) < 0.05 and abs(cy - ty) < 0.05


# --------------------------------------------------------------------------------- min rect
def test_min_area_rect_sides_and_long_axis_angle():
    for angle in (0.0, 30.0, -30.0, 60.0, -75.0, 90.0):
        long, short, a = min_area_rect(np.array(rect_polygon(50, 50, 40, 10, angle)))
        assert long == pytest.approx(40, abs=1e-3) and short == pytest.approx(10, abs=1e-3)
        assert a == pytest.approx(angle if angle != -90 else 90, abs=1e-3)
        assert -90 < a <= 90
    assert min_area_rect(np.zeros((0, 2))) == (0.0, 0.0, 0.0)


# ------------------------------------------------------------------------------- sample_view
def test_sample_view_places_a_small_page_fully_inside_the_crop():
    data = DetDataConfig(crop_size=256, scale_min=1.0, scale_max=1.0, rotate_max_deg=10)
    corners = np.array([[0, 0], [160, 0], [160, 128], [0, 128]], dtype=np.float64)
    for seed in range(20):
        view = sample_view(np.random.default_rng(seed), (160, 128), np.zeros((0, 2)), data, 160)
        p = view.affine.apply(corners)
        assert p.min() >= -0.5 and p.max() <= 256.5
        assert view.scale == pytest.approx(1.0) and not view.text_centred


def test_sample_view_text_crops_aim_at_a_care_line_with_whole_pixel_origins():
    data = DetDataConfig(crop_size=128, scale_min=0.8, scale_max=1.2, text_crop_p=1.0)
    centre = np.array([[600.0, 500.0]])  # well inside a 1200 x 1000 page
    for seed in range(20):
        view = sample_view(np.random.default_rng(seed), (1200, 1000), centre, data, 1200)
        q = view.affine.apply(centre)[0]
        assert view.text_centred
        assert np.all(q >= 0.1 * 128 - 1) and np.all(q <= 0.9 * 128 + 1)
        # The origin is whole pixels: translation = c1 − A·c0 − o with o integral.
        t = math.radians(view.angle_deg)
        f = view.scale
        ext = f * np.array([1200 * abs(math.cos(t)) + 1000 * abs(math.sin(t)),
                            1200 * abs(math.sin(t)) + 1000 * abs(math.cos(t))])  # fmt: skip
        a = view.affine.m[:, :2]
        o = ext / 2 - a @ np.array([600.0, 500.0]) - view.affine.m[:, 2]
        np.testing.assert_allclose(o, np.round(o), atol=1e-9)
    # The draws depend only on the generator.
    v1 = sample_view(np.random.default_rng(4), (1200, 1000), centre, data, 1200)
    v2 = sample_view(np.random.default_rng(4), (1200, 1000), centre, data, 1200)
    assert np.array_equal(v1.affine.m, v2.affine.m) and v1.angle_deg == v2.angle_deg


# ------------------------------------------------------------------------------------ warping
def _polygon_page(tmp_path: Path) -> tuple[Path, tuple[int, int], np.ndarray]:
    w, h = 400, 300
    poly = np.array(rect_polygon(200, 150, 220, 90, 15))
    canvas = np.zeros((h, w), np.uint8)
    rasterize(canvas, shapely.Polygon(poly), 255)
    path = tmp_path / "poly.png"
    Image.fromarray(np.repeat(canvas[..., None], 3, axis=2)).save(path)
    return path, (w, h), poly


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def test_render_view_matches_the_transformed_polygon(tmp_path):
    path, wh, poly = _polygon_page(tmp_path)
    data = DetDataConfig(crop_size=256, scale_min=0.5, scale_max=2.0, text_crop_p=1.0)
    centre = poly.mean(0, keepdims=True)
    scales, angles = [], []
    for seed in range(20):
        view = sample_view(np.random.default_rng(seed), wh, centre, data, 400)
        img = render_view(path, wh, view, 256, reduced_decode=False)
        ref = np.zeros((256, 256), np.uint8)
        rasterize(ref, shapely.Polygon(view.affine.apply(poly)), 1)
        assert _iou(img[..., 0] > 127, ref.astype(bool)) >= 0.97, seed
        scales.append(view.scale)
        angles.append(view.angle_deg)
    assert min(scales) < 1 < max(scales) and all(a != 0 for a in angles)


def test_reduced_decode_matches_full_decode_under_the_adjusted_matrix(tmp_path):
    path, wh, poly = _polygon_page(tmp_path)
    data = DetDataConfig(crop_size=192, scale_min=0.4, scale_max=0.5, text_crop_p=1.0)
    centre = poly.mean(0, keepdims=True)
    for seed in range(10):
        view = sample_view(np.random.default_rng(seed), wh, centre, data, 400)
        assert _reduce_factor(path, wh, view.scale, True) == 2
        reduced = render_view(path, wh, view, 192, reduced_decode=True)[..., 0] > 127
        full = render_view(path, wh, view, 192, reduced_decode=False)[..., 0] > 127
        ref = np.zeros((192, 192), np.uint8)
        rasterize(ref, shapely.Polygon(view.affine.apply(poly)), 1)
        assert _iou(reduced, full) >= 0.97 and _iou(reduced, ref.astype(bool)) >= 0.97


def test_reduce_factor_rules(tmp_path):
    jpg, png = tmp_path / "a.jpg", tmp_path / "a.png"
    assert _reduce_factor(jpg, (1001, 999), 0.3, True) == 2
    assert _reduce_factor(jpg, (1001, 999), 0.12, True) == 8
    assert _reduce_factor(jpg, (1001, 999), 0.12, False) == 1
    assert _reduce_factor(jpg, (1001, 999), 0.51, True) == 1
    # OpenCV resizes non-JPEG decodes with sizes rounded down: only exact divisors qualify.
    assert _reduce_factor(png, (1001, 1000), 0.3, True) == 1
    assert _reduce_factor(png, (1000, 600), 0.12, True) == 8
    assert _reduce_factor(png, (1000, 604), 0.12, True) == 4


# ----------------------------------------------------------------------------------- decoding
def test_decode_page_applies_exif_orientation_like_open_page_image(tmp_path):
    path = tmp_path / "raw" / "exif6.jpg"
    wh = write_exif6_jpeg(path, (64, 40))
    assert wh == (40, 64)
    rgb = decode_page(path, wh)
    ref = np.asarray(open_page_image(_page(path, tmp_path, wh), tmp_path))
    assert rgb.shape == ref.shape == (64, 40, 3)
    assert np.abs(rgb.astype(int) - ref.astype(int)).mean() < 2
    with pytest.raises(ValueError, match="expected"):
        decode_page(path, (64, 40))  # the stored frame, before the EXIF rotation
    with pytest.raises(ValueError, match="expected"):
        decode_page(path, (41, 64))


def test_channel_order_is_rgb_everywhere(tmp_path):
    path = tmp_path / "raw" / "colour.png"
    wh = write_colour_png(path, (48, 36))
    w, h = wh
    rgb = decode_page(path, wh)
    assert np.array_equal(rgb, np.asarray(open_page_image(_page(path, tmp_path, wh), tmp_path)))
    assert tuple(rgb[5, 5]) == (255, 0, 0) and tuple(rgb[5, 20]) == (0, 255, 0)

    identity = View(1.0, 0.0, False, Affine.scale(1.0, 1.0))
    crop = render_view(path, wh, identity, 64, reduced_decode=True)
    assert np.array_equal(crop[:h, :w], rgb)
    assert (crop[h:] == PAD_RGB).all() and (crop[:, w:] == PAD_RGB).all()

    half = View(0.5, 0.0, False, Affine.scale(0.5, 0.5))
    assert _reduce_factor(path, wh, 0.5, True) == 2
    crop2 = render_view(path, wh, half, 32, reduced_decode=True)
    rgb2 = decode_page(path, wh, reduce=2)
    assert rgb2.shape == (18, 24, 3)
    assert np.array_equal(crop2[:18, :24], rgb2)
    assert (crop2[18:] == PAD_RGB).all() and (crop2[:, 24:] == PAD_RGB).all()
