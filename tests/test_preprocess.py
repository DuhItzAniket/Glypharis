from __future__ import annotations

import io
import math
import pickle

import numpy as np
import pytest
from PIL import Image

from glypharis.data.crops import (
    CropSettings,
    background_grey,
    cut_line,
    encode_jpeg,
    expand_quad,
    line_quad,
    rectify,
)
from glypharis.data.exclusions import Exclusions
from glypharis.data.hiertext import ImageAnnotation, Line
from glypharis.data.linepack import LinePackReader, LinePackWriter
from glypharis.data.pages import Page, clip_polygon, hiertext_to_page, read_pages, write_pages
from glypharis.data.text import Charset

CHARSET = Charset(list(" abcdefghijklmnopqrstuvwxyz"))
RECT = ((10.0, 10.0), (90.0, 10.0), (90.0, 30.0), (10.0, 30.0))


def _line(i, text="hello", vertices=RECT, legible=True, handwritten=False, vertical=False):
    return Line(f"img/p0/l{i}", vertices, text, legible, handwritten, vertical, ())


def _ann(lines):
    return ImageAnnotation("img", 100, 50, 1, tuple(lines))


# ------------------------------------------------------------------------------------ pages


def test_conversion_reasons_and_clipping():
    lines = [
        _line(0),
        _line(1, legible=False),
        _line(2, handwritten=True),
        _line(3, vertical=True),
        _line(4, text="​"),  # empty after normalisation
        _line(5, text="héllo"),  # é not in the test charset
        _line(6, vertices=((0, 0), (99, 0), (99, 1), (0, 1))),  # aspect 99 > 64
        _line(7, vertices=((-5, 10), (90, 10), (90, 60), (-5, 60))),  # partly outside: clipped
        _line(8),  # excluded for every use
        _line(9),  # excluded from recognizer only
    ]
    excl = Exclusions(
        "hiertext", 1, frozenset(),
        frozenset({("train", "img/p0/l8")}), frozenset({("train", "img/p0/l9")}),
    )  # fmt: skip
    page = hiertext_to_page(_ann(lines), "train", "raw/x.jpg", CHARSET, excl)
    assert page is not None
    by_id = {ln.line_id: ln for ln in page.lines}
    assert "img/p0/l8" not in by_id
    reasons = {k: v.rec_skip for k, v in by_id.items()}
    assert reasons == {
        "img/p0/l0": None, "img/p0/l1": "illegible", "img/p0/l2": "handwritten",
        "img/p0/l3": "vertical", "img/p0/l4": "empty", "img/p0/l5": "out_of_charset",
        "img/p0/l6": "too_long", "img/p0/l7": None, "img/p0/l9": "excluded",
    }  # fmt: skip
    assert by_id["img/p0/l7"].polygon == [[0.0, 10.0], [90.0, 10.0], [90.0, 50.0], [0.0, 50.0]]
    assert by_id["img/p0/l0"].rec_ok is True


def test_excluded_image_returns_none():
    excl = Exclusions("hiertext", 1, frozenset({("train", "img")}), frozenset(), frozenset())
    assert hiertext_to_page(_ann([_line(0)]), "train", "x", CHARSET, excl) is None


def test_pages_round_trip_is_exact_and_deterministic(tmp_path):
    page = hiertext_to_page(
        _ann([_line(0, text="ab  c"), _line(1, legible=False)]),
        "validation", "raw/v.jpg", CHARSET, Exclusions.empty("hiertext"),
    )  # fmt: skip
    assert page is not None
    write_pages(tmp_path / "a.jsonl.gz", [page, page])
    back = list(read_pages(tmp_path / "a.jsonl.gz"))
    assert back == [page, page]
    assert back[0].lines[0].text_original == "ab  c" and back[0].lines[0].text == "ab c"
    write_pages(tmp_path / "b.jsonl.gz", [page, page])
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()


def test_page_from_dict_round_trip():
    page = Page("d", "x.jpg", 10, 5, "s", "train")
    assert Page.from_dict(page.to_dict()) == page


def test_clip_polygon():
    assert clip_polygon([(-1, 2), (12, -3)], 10, 5) == [[0.0, 2.0], [10.0, 0.0]]


# ------------------------------------------------------------------------------------ crops


def test_expand_quad_grows_by_margin_times_height():
    quad = expand_quad([[10, 10], [90, 10], [90, 30], [10, 30]], 0.25)
    (x0, y0), (x1, _), (_, y2), _ = quad
    assert (x0, y0, x1, y2) == pytest.approx((5, 5, 95, 35))  # 0.25 × height 20 = 5 px


def test_expand_quad_degenerate_is_unchanged():
    q = [[1, 1], [1, 1], [1, 1], [1, 1]]
    assert expand_quad(q, 0.2) == [(1.0, 1.0)] * 4


def test_rectify_keeps_reading_order_and_aspect():
    page = Image.new("L", (200, 100), 255)
    page.paste(0, (20, 40, 40, 60))  # dark block at the left end of the line
    settings = CropSettings(height=32)
    crop = rectify(
        page, [(20, 40), (180, 40), (180, 60), (20, 60)], settings, background_grey(page)
    )
    assert crop.size == (256, 32)  # 160 × 20 box → aspect 8 at height 32
    a = np.asarray(crop, dtype=np.float32)
    assert a[:, :30].mean() < 60 and a[:, -30:].mean() > 200


def test_rectify_rotated_line_comes_out_horizontal():
    page = Image.new("L", (200, 200), 255)
    angle = math.radians(30)
    c, s = math.cos(angle), math.sin(angle)
    quad = [(50, 50), (50 + 100 * c, 50 + 100 * s), (50 + 100 * c - 20 * s, 50 + 100 * s + 20 * c),
            (50 - 20 * s, 50 + 20 * c)]  # fmt: skip
    crop = rectify(page, quad, CropSettings(height=20), 255)
    assert crop.size == (100, 20)


def _band(xs, centre, half):
    """Densified polygon: top edge left to right, then bottom edge right to left."""
    top = [[float(x), centre(x) - half] for x in xs]
    bottom = [[float(x), centre(x) + half] for x in reversed(xs)]
    return top + bottom


def test_line_quad_takes_the_corners_of_a_long_polygon():
    poly = _band([10, 50, 90], lambda x: 20.0, 5)
    assert line_quad(poly) == [[10.0, 15.0], [90.0, 15.0], [90.0, 25.0], [10.0, 25.0]]
    assert line_quad(poly[:4]) == poly[:4]


def test_cut_line_on_a_straight_band_matches_the_quad_crop():
    page = Image.new("L", (300, 100), 255)
    page.paste(0, (40, 45, 60, 55))
    settings = CropSettings(height=32)
    straight = _band(list(range(20, 281, 20)), lambda x: 50.0, 10)
    a = cut_line(page, straight, settings, 255)
    b = rectify(page, expand_quad(line_quad(straight), settings.margin), settings, 255)
    assert abs(a.width - b.width) <= 2 and a.height == b.height == 32
    w = min(a.width, b.width)
    diff = np.abs(np.asarray(a, np.float32)[:, :w] - np.asarray(b, np.float32)[:, :w])
    assert diff.mean() < 8


def test_cut_line_follows_a_curved_line():
    page = Image.new("L", (400, 200), 255)
    px = page.load()

    def centre(x):
        return 100 + 60 * ((x - 200) / 180) ** 2  # a bowed line, 60 px of sag at the ends

    for x in range(20, 381):
        for y in range(int(centre(x)) - 6, int(centre(x)) + 7):
            px[x, y] = 0
    poly = _band(list(range(20, 381, 30)), centre, 10)
    settings = CropSettings(height=40)
    strip = np.asarray(cut_line(page, poly, settings, 255), np.float32)
    mid = strip[16:24, 8:-8]  # the band stays in the middle rows all the way along
    assert mid.mean() < 60
    assert strip[:4].mean() > 200 and strip[-4:].mean() > 200
    corners = np.asarray(
        rectify(page, expand_quad(line_quad(poly), settings.margin), settings, 255), np.float32
    )
    assert corners[16:24].mean() > mid.mean() + 100  # the corner crop loses the middle of the line


def test_outside_area_uses_fill_not_black():
    page = Image.new("L", (50, 50), 200)
    crop = rectify(page, [(-50, 0), (40, 0), (40, 20), (-50, 20)], CropSettings(height=20), 200)
    assert np.asarray(crop).min() > 150


def test_encode_jpeg_decodes():
    img = Image.new("L", (40, 10), 128)
    back = Image.open(io.BytesIO(encode_jpeg(img, 90)))
    assert back.size == (40, 10) and back.mode == "L"


# --------------------------------------------------------------------------------- linepack


def _crop_bytes(value: int, w: int) -> bytes:
    return encode_jpeg(Image.new("L", (w, 16), value), 95)


def test_linepack_round_trip_and_verify(tmp_path):
    prefix = tmp_path / "pack" / "lines_train"
    with LinePackWriter(prefix, {"split": "train"}) as w:
        for i in range(5):
            w.add(
                f"k{i}",
                f"d{i}",
                "train",
                "src",
                f"label {i}",
                _crop_bytes(40 * i, 30 + i),
                (30 + i, 16),
            )
        meta = w.close()
    assert meta["count"] == 5 and meta["split"] == "train"
    r = LinePackReader(prefix)
    assert len(r) == 5 and r.verify()
    assert [rec.label for rec in r.records] == [f"label {i}" for i in range(5)]
    img = r.image(3)
    assert img.size == (33, 16) and abs(np.asarray(img).mean() - 120) < 3
    assert r.read_bytes(2) == _crop_bytes(80, 32)

    clone = pickle.loads(pickle.dumps(r))  # what a DataLoader worker receives
    assert clone._fh is None and clone.read_bytes(0) == r.read_bytes(0)
    r.close()
    clone.close()

    with open(prefix.with_suffix(".bin"), "r+b") as fh:
        fh.write(b"\0")
    assert not LinePackReader(prefix).verify()


def test_linepack_rejects_mismatched_index(tmp_path):
    prefix = tmp_path / "p"
    with LinePackWriter(prefix) as w:
        w.add("k", "d", "train", "s", "x", b"abc", (1, 1))
    idx = prefix.with_suffix(".index.jsonl")
    idx.write_text(idx.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        LinePackReader(prefix)
