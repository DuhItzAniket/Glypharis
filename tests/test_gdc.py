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
    group_chars,
    is_small_caps_font,
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
    sheet, gt, _ = compose_pack_page(page, lines, pack_page, f"GDC-{pack_page:04d}", LAYOUT)
    return sheet, gt


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


# ---------------------------------------------------------------------------- line grouping


def _chars(text, x0, y, w=5.0, h=10.0):
    """Characters of one printed row: boxes (left, bottom, right, top), PDF coordinates."""
    out = []
    for i, ch in enumerate(text):
        box = None if ch == " " else (x0 + i * w, y, x0 + (i + 1) * w, y + h)
        out.append((ch, box, 0.0))
    return out


def test_group_chars_splits_after_end_of_line_hyphen():
    chars = _chars("many in", 0, 100) + _chars("￾", 35, 100) + _chars("struments", 0, 85)
    assert [t for t, _, _ in group_chars(chars)] == ["many in-", "struments"]


def test_group_chars_splits_on_row_change_without_break():
    chars = _chars("first row", 0, 100) + _chars("second row", 0, 85)
    lines = group_chars(chars)
    assert [t for t, _, _ in lines] == ["first row", "second row"]
    assert lines[0][1][1] == 100 and lines[1][1][3] == 95


def test_group_chars_keeps_superscripts_on_the_line():
    chars = _chars("x", 0, 100) + _chars("2", 5, 104, h=6) + _chars(" + y", 10, 100)
    assert [t for t, _, _ in group_chars(chars)] == ["x2 + y"]


def test_group_chars_honours_generated_breaks():
    breaks = [("\r", None, 0.0), ("\n", None, 0.0)]
    chars = [*_chars("left col", 0, 100), *breaks, *_chars("right", 300, 100)]
    assert [t for t, _, _ in group_chars(chars)] == ["left col", "right"]


def test_group_chars_keeps_rotated_text_together():
    # A margin stamp rotated by 270 degrees: glyphs stacked in one column, no generated break
    # before it. The rotation change alone separates it from the body text.
    rotated = [
        (ch, (50.0, 200.0 - 8 * i, 58.0, 208.0 - 8 * i), 270.0) for i, ch in enumerate("arXiv")
    ]
    lines = group_chars([*_chars("body", 100, 300), *rotated])
    assert [(t, v) for t, _, v in lines] == [("body", False), ("arXiv", True)]


def test_footnote_mark_does_not_make_a_line_vertical():
    # "VLP" followed by a raised footnote mark above the last letter, then more text on the row.
    chars = [*_chars("models for VLP", 0, 100), ("1", (66.0, 107.0, 69.0, 113.0), 0.0),
             *_chars(" and more", 70, 100)]  # fmt: skip
    lines = group_chars(chars)
    assert [(t, v) for t, _, v in lines] == [("models for VLP1 and more", False)]


def test_ignore_regions_are_mapped_to_the_sheet():
    page = Image.new("RGB", (600, 800), "white")
    _, _, ignore = compose_pack_page(page, [], 1, "x", LAYOUT, [(0, 0, 600, 800)])
    x0, y0, x1, y1 = LAYOUT.content_box
    (ax, ay), _, (bx, by), _ = ignore[0]
    assert x0 <= ax < bx <= x1 and y0 <= ay < by <= y1


def test_group_chars_splits_at_wide_gaps_but_not_word_spaces():
    # "Total" ... form field ... "12": a 40-unit gap with 10-unit glyphs. Word spaces stay.
    chars = [*_chars("Tax-exempt interest . . .", 0, 100), *_chars("2a", 170, 100)]
    assert [t for t, _, _ in group_chars(chars)] == ["Tax-exempt interest . . .", "2a"]


@pytest.mark.parametrize(
    ("name", "small_caps"),
    [("SFCC1200", True), ("ABCDEF+CMCSC10", True), ("Garamond-SC", True),
     ("LibreCaslon SmallCaps", True),
     ("SFRM1200", False), ("CMR10", False), ("Times-Roman", False), ("ABCDEF+Arial-BoldMT", False)],
)  # fmt: skip
def test_small_caps_font_detection(name, small_caps):
    assert is_small_caps_font(name) is small_caps


def test_raised_quote_after_comma_stays_on_the_line():
    chars = [*_chars('"changes of position', 0, 100), (",", (100.0, 98.0, 102.0, 101.0), 0.0),
             ("”", (103.0, 107.0, 107.0, 110.0), 0.0), *_chars(" which", 108, 100)]  # fmt: skip
    assert [t for t, _, _ in group_chars(chars)] == ['"changes of position,” which']
