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


def extract_lines(pdf_path: str | Path, page_index: int, scale: float) -> list[PdfLine]:
    """Text lines of one PDF page, as PDFium orders them, with boxes in pixels at ``scale``.

    PDFium inserts line breaks between text runs; each run of characters between breaks becomes one
    line. The box is the union of the non-space character boxes. Labels go through
    `normalize_label`, so ligatures are written out exactly as in training labels.
    """
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_index]
        height_pt = page.get_height()
        tp = page.get_textpage()
        n = tp.count_chars()
        lines: list[PdfLine] = []
        chars: list[str] = []
        boxes: list[tuple[float, float, float, float]] = []

        def flush() -> None:
            text = normalize_label("".join(chars))
            if text and boxes:
                x0 = min(b[0] for b in boxes) * scale
                x1 = max(b[2] for b in boxes) * scale
                y0 = (height_pt - max(b[3] for b in boxes)) * scale  # PDF y grows upwards
                y1 = (height_pt - min(b[1] for b in boxes)) * scale
                lines.append(PdfLine(text, (x0, y0, x1, y1)))
            chars.clear()
            boxes.clear()

        for i in range(n):
            ch = tp.get_text_range(i, 1)
            if ch in ("\r", "\n", ""):
                flush()
                continue
            chars.append(ch)
            if not ch.isspace():
                left, bottom, right, top = tp.get_charbox(i)
                if right > left and top > bottom:
                    boxes.append((left, bottom, right, top))
        flush()
        tp.close()
        page.close()
        return lines
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
    page: Image.Image, lines: list[PdfLine], pack_page: int, label: str, layout: PackLayout
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Place a rendered page on a marked A4 sheet; return the sheet and lines in sheet pixels."""
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

    out = []
    for ln in lines:
        bx0, by0, bx1, by1 = ln.box
        out.append(
            {
                "text": ln.text,
                "polygon": [
                    [ox + bx0 * f, oy + by0 * f],
                    [ox + bx1 * f, oy + by0 * f],
                    [ox + bx1 * f, oy + by1 * f],
                    [ox + bx0 * f, oy + by1 * f],
                ],
            }
        )
    return sheet, out


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
