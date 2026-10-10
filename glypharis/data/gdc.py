"""GDC (Glypharis Document Challenge): print-and-capture tooling.

1. `extract_lines` reads text lines and their boxes from a born-digital PDF page (PDFium).
2. `compose_pack_page` places the rendered page on an A4 sheet with four ArUco markers. Marker ids
   ``4 * pack_page + corner`` (corner 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left)
   identify both the page and its orientation in a photo.
3. `align_capture` finds the markers in a phone photo or scan, fits a homography from sheet to
   photo, and projects the ground-truth line boxes into the photo.

Ground truth therefore comes from the PDF text layer, never from an OCR engine.
"""

from __future__ import annotations

import ctypes
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from glypharis.data.text import normalize_label

MARKER_DICT_NAME = "DICT_4X4_1000"
MARKERS_PER_PAGE = 4
MAX_PACK_PAGES = 1000 // MARKERS_PER_PAGE


# --------------------------------------------------------------------------- PDF text lines


@dataclass(frozen=True)
class PdfLine:
    text: str
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in page pixels at the render scale
    vertical: bool = False  # rotated text read top-to-bottom or bottom-to-top


Box = tuple[float, float, float, float]  # left, bottom, right, top in PDF points (y up)
GAP_BREAK = 2.5  # horizontal gap, in glyph heights, that ends a line
_SMALL_CAPS = re.compile(r"^(cmcsc|eccc|sfcc|lmromancaps)|smallcaps|small caps|[-_ ]sc$")


def is_small_caps_font(name: str) -> bool:
    """Small-caps fonts store lowercase text but print capitals (e.g. TeX's cmcsc, CM-Super sfcc).

    Subset prefixes such as ``ABCDEF+`` are ignored.
    """
    base = name.split("+", 1)[-1].lower()
    return bool(_SMALL_CAPS.search(base))


SOFT_LINE_END_HYPHEN = "\ufffe"  # PDFium's code for a hyphen that ends a printed line


Char = tuple[str, Box | None, float]  # character, box (None for spaces), rotation in degrees


def _union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))  # fmt: skip


def _rotated(angle: float) -> bool:
    """Text not running left to right: anything more than 20 degrees from upright."""
    a = angle % 360
    return 20 < a < 340


def _new_visual_line(line: list[Box], cur: Box, rotated: bool) -> bool:
    """True when ``cur`` starts a new printed line.

    Upright text is compared with the line so far (its union box), not with the previous glyph:
    small punctuation such as a low comma followed by a raised quote must not look like a row
    change. The line ends when the text moves to another row (super- and subscripts shift by less
    than 0.6 line heights and stay to the right) or leaves a horizontal gap wider than
    ``GAP_BREAK`` line heights (form fields, table columns, running headers). Word spaces and dot
    leaders are far narrower. Rotated text: the line ends when the text moves to another column or
    leaves a gap of more than three glyph sizes.
    """
    prev = line[-1]
    if rotated:
        w = max(prev[2] - prev[0], cur[2] - cur[0], 1e-6)
        h = max(prev[3] - prev[1], cur[3] - cur[1], 1e-6)
        other_column = abs((cur[0] + cur[2]) / 2 - (prev[0] + prev[2]) / 2) > 0.6 * w
        gap = abs((cur[1] + cur[3]) / 2 - (prev[1] + prev[3]) / 2) > 3 * h
        return other_column or gap
    u = _union(line)
    h = max(u[3] - u[1], cur[3] - cur[1], 1e-6)
    cur_mid = (cur[1] + cur[3]) / 2
    outside = cur_mid < u[1] or cur_mid > u[3]  # the glyph's centre is not within the line band
    shift = abs(cur_mid - (u[1] + u[3]) / 2)
    if outside and shift > 0.6 * h and (cur[0] < prev[2] - 0.5 * h or shift > 1.5 * h):
        return True
    return cur[0] - prev[2] > GAP_BREAK * h


def group_chars(chars: list[Char]) -> list[tuple[str, Box, bool]]:
    """Group a PDF character stream into printed lines: ``(text, box, vertical)``.

    Breaks at PDFium's generated line breaks, after an end-of-line hyphen (U+FFFE, kept as "-",
    because it is printed; PDFium alone joins a hyphenated line with the next one), where the text
    rotation changes, and where the geometry shows the text moved to another row or column.
    Rotated runs (e.g. the arXiv margin stamp, axis titles) are returned with ``vertical=True``.
    """
    lines: list[tuple[str, Box, bool]] = []
    text: list[str] = []
    boxes: list[Box] = []
    run_rotated = False

    def flush() -> None:
        label = normalize_label("".join(text))
        if label and boxes:
            lines.append((label, _union(boxes), run_rotated))
        text.clear()
        boxes.clear()

    for ch, box, angle in chars:
        if ch in ("\r", "\n", ""):
            flush()
            continue
        rotated = _rotated(angle)
        if text and rotated != run_rotated:
            flush()
        if not text:
            run_rotated = rotated
        if box is not None and boxes and _new_visual_line(boxes, box, run_rotated):
            flush()
            run_rotated = rotated
        if ch == SOFT_LINE_END_HYPHEN:
            text.append("-")
            if box is not None:
                boxes.append(box)
            flush()
            continue
        text.append(ch)
        if box is not None:
            boxes.append(box)
    flush()
    return lines


def _font_name(pdfium_c: Any, tp: Any, index: int) -> str:
    buf = ctypes.create_string_buffer(256)
    flags = ctypes.c_int(0)
    pdfium_c.FPDFText_GetFontInfo(tp.raw, index, buf, 256, ctypes.byref(flags))
    return buf.value.decode("latin-1")


def extract_lines(pdf_path: str | Path, page_index: int, scale: float) -> list[PdfLine]:
    """Text lines of one PDF page with boxes in pixels at ``scale`` (see `group_chars`).

    Labels go through `normalize_label`, so ligatures are written out exactly as in training
    labels.
    """
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        height_pt = page.get_height()
        tp = page.get_textpage()
        chars: list[Char] = []
        for i in range(tp.count_chars()):
            ch = tp.get_text_range(i, 1)
            box: Box | None = None
            if ch and not ch.isspace():
                left, bottom, right, top = tp.get_charbox(i)
                if right > left and top > bottom:
                    box = (left, bottom, right, top)
            angle = pdfium_c.FPDFText_GetCharAngle(tp.raw, i)  # radians, -1 on error
            if ch.islower() and is_small_caps_font(_font_name(pdfium_c, tp, i)):
                ch = ch.upper()  # printed as a capital
            chars.append((ch, box, math.degrees(angle) if angle >= 0 else 0.0))
        tp.close()
        page.close()
    finally:
        pdf.close()
    out = []
    for text, (left, bottom, right, top), vertical in group_chars(chars):
        # PDF y grows upwards; image y grows downwards.
        box = (left * scale, (height_pt - top) * scale, right * scale, (height_pt - bottom) * scale)
        out.append(PdfLine(text, box, vertical))  # fmt: skip
    return out


def extract_image_regions(
    pdf_path: str | Path, page_index: int, scale: float, min_size_pt: float = 12.0
) -> list[tuple[float, float, float, float]]:
    """Boxes (pixels at ``scale``, image coordinates) of raster images placed on the page.

    Text inside a raster image is printed but has no text layer, so it has no ground truth. These
    regions are evaluated as "don't care" (EXPERIMENT_PROTOCOL §3, detection).
    """
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        height_pt = page.get_height()
        out = []
        for obj in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=8):
            left, bottom, right, top = obj.get_bounds()
            if right - left >= min_size_pt and top - bottom >= min_size_pt:
                out.append((left * scale, (height_pt - top) * scale, right * scale,
                            (height_pt - bottom) * scale))  # fmt: skip
        page.close()
        return out
    finally:
        pdf.close()


def render_page(pdf_path: str | Path, page_index: int, dpi: int = 300) -> Image.Image:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        img: Image.Image = page.render(scale=dpi / 72).to_pil().convert("RGB")
        page.close()
        return img
    finally:
        pdf.close()


# ------------------------------------------------------------------------------- pack pages


@dataclass(frozen=True)
class PackLayout:
    dpi: int = 300
    paper_in: tuple[float, float] = (8.27, 11.69)  # A4 portrait
    margin_in: float = 0.35  # printers cannot print to the edge
    marker_in: float = 0.55
    gap_in: float = 0.1

    def px(self, inches: float) -> int:
        return round(inches * self.dpi)

    @property
    def size(self) -> tuple[int, int]:
        return self.px(self.paper_in[0]), self.px(self.paper_in[1])

    def marker_corners(self, corner: int) -> np.ndarray:
        """The 4 corner points (TL, TR, BR, BL order, as OpenCV reports them) of a marker."""
        w, h = self.size
        m, s = self.px(self.margin_in), self.px(self.marker_in)
        x, y = {0: (m, m), 1: (w - m - s, m), 2: (w - m - s, h - m - s), 3: (m, h - m - s)}[corner]
        return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]], dtype=np.float32)

    @property
    def content_box(self) -> tuple[int, int, int, int]:
        """Area for the document page, clear of markers and the label line."""
        w, h = self.size
        top = self.px(self.margin_in + self.marker_in + self.gap_in)
        side = self.px(self.margin_in)
        return side, top, w - side, h - top


def _aruco() -> Any:
    import cv2

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT_NAME))


def compose_pack_page(
    page: Image.Image,
    lines: list[PdfLine],
    pack_page: int,
    label: str,
    layout: PackLayout,
    ignore: list[tuple[float, float, float, float]] | None = None,
) -> tuple[Image.Image, list[dict[str, Any]], list[list[list[float]]]]:
    """Place a rendered page on a marked A4 sheet.

    Returns the sheet, the lines in sheet pixels, and the "don't care" polygons in sheet pixels.
    """
    import cv2

    if not 0 <= pack_page < MAX_PACK_PAGES:
        raise ValueError(f"pack page {pack_page} outside 0..{MAX_PACK_PAGES - 1}")
    w, h = layout.size
    sheet = Image.new("RGB", (w, h), "white")
    x0, y0, x1, y1 = layout.content_box
    f = min((x1 - x0) / page.width, (y1 - y0) / page.height)
    pw, ph = round(page.width * f), round(page.height * f)
    ox, oy = x0 + (x1 - x0 - pw) // 2, y0 + (y1 - y0 - ph) // 2
    sheet.paste(page.resize((pw, ph), Image.Resampling.LANCZOS), (ox, oy))

    dictionary = _aruco()
    for corner in range(MARKERS_PER_PAGE):
        size = layout.px(layout.marker_in)
        marker = cv2.aruco.generateImageMarker(
            dictionary, MARKERS_PER_PAGE * pack_page + corner, size
        )
        pts = layout.marker_corners(corner)
        sheet.paste(Image.fromarray(marker).convert("RGB"), (int(pts[0][0]), int(pts[0][1])))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=layout.px(0.14))
    draw.text((w // 2, h - layout.px(layout.margin_in + layout.marker_in / 2)), label,
              fill="black", font=font, anchor="mm")  # fmt: skip

    def rect(b: tuple[float, float, float, float]) -> list[list[float]]:
        x0_, y0_, x1_, y1_ = b
        return [[ox + x0_ * f, oy + y0_ * f], [ox + x1_ * f, oy + y0_ * f],
                [ox + x1_ * f, oy + y1_ * f], [ox + x0_ * f, oy + y1_ * f]]  # fmt: skip

    out = [{"text": ln.text, "vertical": ln.vertical, "polygon": rect(ln.box)} for ln in lines]
    return sheet, out, [rect(b) for b in (ignore or [])]


# -------------------------------------------------------------------------------- alignment


@dataclass
class Alignment:
    pack_page: int
    homography: np.ndarray  # 3x3, sheet pixels -> photo pixels
    markers_found: int
    reprojection_error_px: float  # RMS over all detected marker corners


def align_capture(photo: Image.Image, layout: PackLayout) -> Alignment | None:
    """Find the pack page shown in ``photo`` and the sheet-to-photo homography."""
    import cv2

    gray = np.asarray(photo.convert("L"))
    detector = cv2.aruco.ArucoDetector(_aruco(), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    by_page: dict[int, list[tuple[int, np.ndarray]]] = {}
    for c, i in zip(corners, ids.ravel(), strict=True):
        by_page.setdefault(int(i) // MARKERS_PER_PAGE, []).append((int(i) % MARKERS_PER_PAGE, c[0]))
    page, found = max(by_page.items(), key=lambda kv: len(kv[1]))
    if len(found) < 3:
        return None  # fewer than 3 markers cannot pin down a perspective reliably
    src = np.concatenate([layout.marker_corners(corner) for corner, _ in found])
    dst = np.concatenate([pts.astype(np.float32) for _, pts in found])
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return None
    proj = cv2.perspectiveTransform(src[None], H)[0]
    rms = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
    return Alignment(page, H, len(found), rms)


def project_polygons(polys: list[list[list[float]]], H: np.ndarray) -> list[list[list[float]]]:
    import cv2

    if not polys:
        return []
    pts = np.array(polys, dtype=np.float32).reshape(1, -1, 2)
    out = cv2.perspectiveTransform(pts, H)[0].reshape(len(polys), 4, 2)
    return [[[float(x), float(y)] for x, y in p] for p in out]
