from __future__ import annotations

import cv2
import gdc_scan_build
import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box

from glypharis.data.gdc import PdfLine, densify_box


def line(text: str, y0: float, y1: float) -> PdfLine:
    return PdfLine(text, (100.0, y0, 900.0, y1))


def test_footer_written_first_moves_to_the_end():
    lines = [line("7", 3000, 3040), line("Header", 80, 120), line("Body", 400, 440)]
    assert [ln.text for ln in gdc_scan_build.reading_order(lines)] == ["Header", "Body", "7"]
    # Nothing starts below every other line: the order is kept.
    lines = [line("A", 80, 120), line("B", 400, 440), line("C", 430, 470)]
    assert [ln.text for ln in gdc_scan_build.reading_order(lines)] == ["A", "B", "C"]


def test_outside_bands_cover_everything_but_the_content_box():
    content = [[300.0, 200.0], [1700.0, 260.0], [1650.0, 2300.0], [350.0, 2250.0]]
    pts = densify_box(content, 100.0)
    bands = gdc_scan_build.outside_bands(pts, len(pts) // 2, 2000, 2600)
    covered = shapely.union_all([Polygon(b) for b in bands])
    outside = box(0, 0, 2000, 2600).difference(Polygon(pts))
    assert outside.difference(covered).area < 1.0
    assert covered.intersection(Polygon(pts)).area < 1.0


def test_content_box_pads_lines_and_images():
    lines = [line("A", 100, 140), line("B", 300, 340)]
    assert gdc_scan_build.content_box(lines, [(50.0, 500.0, 950.0, 900.0)], 10.0) == (
        40.0, 90.0, 960.0, 910.0,
    )  # fmt: skip


def test_page_homography_recovers_a_known_warp():
    page = Image.new("L", (1275, 1650), 255)
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=28)
    rng = np.random.default_rng(0)
    for row in range(40):
        words = " ".join("".join(rng.choice(list("abcdefghij0123"), 6)) for _ in range(6))
        draw.text((80, 60 + row * 38), words, fill=0, font=font)
    sheet = np.asarray(page)  # the page at DPI * WORK_SCALE
    full = 1 / gdc_scan_build.WORK_SCALE  # sheet pixels -> page pixels at DPI
    # Page pixels at DPI -> photo pixels: a mild perspective, as in a hand-held shot.
    H_true = np.array([[1.1, 0.05, 150], [-0.03, 1.12, 200], [1e-5, 2e-5, 1]])
    photo = cv2.warpPerspective(sheet, H_true @ np.diag([full, full, 1]), (3072, 4080),
                                borderValue=40)  # fmt: skip
    H, inliers = gdc_scan_build.page_homography(photo, sheet)
    assert inliers >= gdc_scan_build.MIN_INLIERS
    corners = np.float32([[0, 0], [2550, 0], [2550, 3300], [0, 3300]])[None]
    err = cv2.perspectiveTransform(corners, H) - cv2.perspectiveTransform(corners, H_true)
    assert float(np.abs(err).max()) < 5.0
