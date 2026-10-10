from __future__ import annotations

import cv2
import matplotlib
import numpy as np
import pytest
from PIL import Image, ImageFilter

from glypharis.data.gdc import (
    PackLayout,
    PdfLine,
    align_capture,
    compose_pack_page,
    extract_lines,
    project_polygons,
    render_page,
)

matplotlib.use("Agg")
LAYOUT = PackLayout(dpi=100)  # small sheets keep the tests fast


@pytest.fixture(scope="module")
def text_pdf(tmp_path_factory):
    import matplotlib.pyplot as plt

    path = tmp_path_factory.mktemp("pdf") / "doc.pdf"
    fig = plt.figure(figsize=(6, 4))
    fig.text(0.1, 0.8, "The first line of text", fontsize=14)
    fig.text(0.1, 0.6, "Office 2024: 15% off", fontsize=14)  # ligature-prone word
    fig.text(0.1, 0.3, "Third line", fontsize=10)
    fig.savefig(path)
    plt.close(fig)
    return path


def test_extract_lines_reads_text_layer(text_pdf):
    lines = extract_lines(text_pdf, 0, scale=100 / 72)
    texts = [ln.text for ln in lines]
    assert texts == ["The first line of text", "Office 2024: 15% off", "Third line"]
    (x0, y0, x1, y1) = lines[0].box
    # x: 0.1 x 6 in at 100 dpi = 60 px. y: the baseline sits at 0.2 x 4 in = 80 px from the top,
    # so the glyph tops are above it and the descenders slightly below.
    assert 50 < x0 < 70 and x1 > x0 + 100 and 55 < y0 < 80 < y1 < 90
    assert lines[0].box[1] < lines[1].box[1] < lines[2].box[1]  # top-to-bottom in image space


def test_render_page_size(text_pdf):
    img = render_page(text_pdf, 0, dpi=50)
    assert img.size == (300, 200) and img.mode == "RGB"


def _sheet(pack_page=7):
    page = Image.new("RGB", (600, 800), "white")
    lines = [PdfLine("hello world", (100, 100, 300, 130)), PdfLine("second", (100, 400, 250, 430))]
    return compose_pack_page(page, lines, pack_page, f"GDC-{pack_page:04d}", LAYOUT)


def _photo(sheet: Image.Image, H: np.ndarray, size=(1100, 1300)) -> Image.Image:
    a = cv2.warpPerspective(np.asarray(sheet), H, size, borderValue=(90, 80, 70))
    return Image.fromarray(a)


@pytest.mark.parametrize(
    "H",
    [
        np.array([[1.0, 0.05, 80], [-0.04, 1.0, 60], [1e-5, 2e-5, 1.0]]),  # tilted phone shot
        np.array([[0.0, -1.0, 1250], [1.0, 0.0, 40], [0.0, 0.0, 1.0]]),  # sheet rotated 90 degrees
    ],
)
def test_alignment_recovers_page_and_boxes(H):
    sheet, gt = _sheet()
    photo = _photo(sheet, H, size=(1300, 1300)).filter(ImageFilter.GaussianBlur(1))
    a = align_capture(photo, LAYOUT)
    assert a is not None and a.pack_page == 7 and a.markers_found == 4
    assert a.reprojection_error_px < 2
    got = project_polygons([g["polygon"] for g in gt], a.homography)
    want = project_polygons([g["polygon"] for g in gt], H)
    assert np.abs(np.array(got) - np.array(want)).max() < 3


def test_three_markers_are_enough_and_two_are_not():
    sheet, _ = _sheet(pack_page=3)
    arr = np.asarray(sheet).copy()
    for corner in (2, 3):
        x, y = LAYOUT.marker_corners(corner)[0].astype(int)
        s = LAYOUT.px(LAYOUT.marker_in)
        arr[y : y + s, x : x + s] = 255
        if corner == 2:
            a = align_capture(Image.fromarray(arr), LAYOUT)
            assert a is not None and a.pack_page == 3 and a.markers_found == 3
    assert align_capture(Image.fromarray(arr), LAYOUT) is None


def test_no_markers_returns_none():
    assert align_capture(Image.new("RGB", (400, 400), "white"), LAYOUT) is None


def test_pack_page_range():
    with pytest.raises(ValueError):
        compose_pack_page(Image.new("RGB", (10, 10)), [], 250, "x", LAYOUT)
