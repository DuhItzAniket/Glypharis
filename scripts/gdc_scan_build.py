"""Build (and optionally freeze) GDC-scan: one private printed document, photographed.

Usage:
    python scripts/gdc_scan_build.py [--data-root data] [--freeze]

Inputs, in ``data/raw/gdc_scan/source/`` (local only, never committed):
    original.pdf   the born-digital document that was printed
    scan.pdf       the phone scanner app's PDF: one photo per printed page, same order

GDC-scan has no corner markers, so each photo is matched to its page differently from GDC v1:
1. the photo's embedded JPEG is extracted byte for byte (``data/raw/gdc_scan/captures/``);
2. a homography from the clean page (rendered from ``original.pdf``) to the photo is fitted on
   SIFT feature matches with RANSAC;
3. the GDC v1 refinement (smooth optical flow, `glypharis.data.gdc.refine_alignment`) corrects
   curl and lens distortion, and each line gets the same local alignment score as in GDC v1.

Ground truth is the PDF text layer (`glypharis.data.gdc.extract_lines`), never an OCR engine. Two
rules adapt it to this document:
- a line that starts below every other line is moved to the end, so the page text follows the
  printed reading order;
- ignore regions are the PDF's raster images plus everything outside the page's content box,
  padded by ``CONTENT_PAD`` inches.

Re-running over an existing pages file must reproduce every ground-truth line exactly; otherwise
nothing is written.

Writes:
    data/processed/gdc_scan/pages_test.jsonl.gz     one page record per photo (split "test")
    data/processed/gdc_scan/aligned_qa/<page>.jpg   ground truth drawn on each photo, for checking
    data/manifests/gdc_scan/alignment.json          per-photo result
With --freeze (only when every page aligns without problems):
    data/manifests/gdc_scan/gdc_scan_v1.json        hashes of sources, captures and pages

The committed manifests hold hashes and counts only: the document is private, so no text, file
name of the source or image leaves the machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from gdc_align import (
    DENSIFY_STEP,
    MAX_IRREGULARITY,
    MIN_LINE_SCORE,
    MIN_PAGE_NCC,
    WORK_SCALE,
    _line_score,
    changed_lines,
)
from PIL import Image, ImageDraw, ImageOps

from glypharis.data.gdc import (
    PdfLine,
    densify_box,
    edge_irregularity,
    extract_image_regions,
    extract_lines,
    refine_alignment,
    render_page,
)
from glypharis.data.pages import Page, PageLine, read_pages, write_pages
from glypharis.data.text import Charset

DPI = 300  # ground-truth coordinates: page pixels at 300 dpi
CONTENT_PAD = 0.15  # inches around the page's text and images that still count as content
FEATURE_SCALE = 0.35  # photo scale for SIFT matching (the page is matched at 150 dpi)
MIN_INLIERS = 50  # fewer RANSAC inliers than this means the page was not found
GDC_SCAN_VERSION = "1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reading_order(lines: list[PdfLine]) -> list[PdfLine]:
    """Move lines that start below every other line to the end."""
    footer = [ln for ln in lines if all(ln.box[1] > o.box[3] for o in lines if o is not ln)]
    return [ln for ln in lines if ln not in footer] + footer


def content_box(
    lines: list[PdfLine], images: list[tuple[float, float, float, float]], pad: float
) -> tuple[float, float, float, float]:
    boxes = [ln.box for ln in lines] + images
    return (min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad,
            max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad)  # fmt: skip


def outside_bands(mapped: np.ndarray, n: int, width: int, height: int) -> list[np.ndarray]:
    """Four polygons covering the photo outside a mapped content box.

    ``mapped`` is the densified box (``n`` points along the top edge left to right, then the bottom
    edge right to left) in photo pixels. Each band joins one side of the box to the photo edge;
    together they cover everything outside the box, overlapping at the corners.
    """
    top, bottom = mapped[:n], mapped[n:][::-1]
    w, h = float(width), float(height)
    left = np.array([top[0], bottom[0]])
    right = np.array([top[-1], bottom[-1]])
    return [
        np.vstack([[[0, 0], [w, 0]], top[::-1]]),
        np.vstack([[[w, h], [0, h]], bottom]),
        np.vstack([[[0, h], [0, 0]], left]),
        np.vstack([[[w, 0], [w, h]], right[::-1]]),
    ]


def page_homography(photo_gray: np.ndarray, sheet_gray: np.ndarray) -> tuple[np.ndarray, int]:
    """Homography from page pixels at ``DPI`` to photo pixels, from SIFT matches and RANSAC.

    ``sheet_gray`` is the page rendered at ``DPI * WORK_SCALE``.
    """
    import cv2

    small = cv2.resize(photo_gray, None, fx=FEATURE_SCALE, fy=FEATURE_SCALE,
                       interpolation=cv2.INTER_AREA)  # fmt: skip
    sift = cv2.SIFT_create(8000)  # type: ignore[attr-defined]
    k1, d1 = sift.detectAndCompute(sheet_gray, None)
    k2, d2 = sift.detectAndCompute(small, None)
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < MIN_INLIERS:
        return np.eye(3), 0
    src = np.float32([k1[g.queryIdx].pt for g in good]) / WORK_SCALE
    dst = np.float32([k2[g.trainIdx].pt for g in good]) / FEATURE_SCALE
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)
    if H is None:
        return np.eye(3), 0
    return H, int(inliers.sum())


def extract_captures(scan_pdf: Path, out_dir: Path) -> list[Path]:
    """Write each page's embedded photo, unchanged, as ``page_NN.jpg``."""
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(scan_pdf))
    paths = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            images = list(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]))
            if len(images) != 1:
                raise SystemExit(f"scan page {i + 1} holds {len(images)} images, expected one")
            stem = out_dir / f"page_{i + 1:02d}"
            images[0].extract(str(stem), fb_format="png")
            found = [p for p in out_dir.glob(f"{stem.name}.*")]
            if len(found) != 1 or found[0].suffix != ".jpg":
                raise SystemExit(f"scan page {i + 1}: expected one embedded JPEG, got {found}")
            paths.append(found[0])
            page.close()
    finally:
        pdf.close()
    return paths


def build_page(
    index: int,
    photo: Image.Image,
    image_rel: str,
    original: Path,
    charset: Charset,
) -> tuple[Page, dict[str, Any]]:
    photo_gray = np.asarray(photo.convert("L"))
    sheet = np.asarray(render_page(original, index, dpi=int(DPI * WORK_SCALE)).convert("L"))
    H, inliers = page_homography(photo_gray, sheet)
    stats: dict[str, Any] = {"inliers": inliers}
    page = Page(
        doc_id=f"gdcscan-p{index + 1:02d}",
        image=image_rel,
        width=photo.size[0],
        height=photo.size[1],
        source="gdc_scan",
        split="test",
    )
    if inliers < MIN_INLIERS:
        stats["status"] = "not_found"
        return page, stats
    ref, i0, refined = refine_alignment(photo_gray, sheet, H, WORK_SCALE)
    scale = DPI / 72
    lines = reading_order(extract_lines(original, index, scale))
    images = extract_image_regions(original, index, scale)

    def mapped(box: tuple[float, float, float, float]) -> np.ndarray:
        x0, y0, x1, y1 = box
        return ref.map_points(densify_box([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], DENSIFY_STEP))

    def as_list(pts: np.ndarray) -> list[list[float]]:
        return [[round(float(x), 2), round(float(y), 2)] for x, y in pts]

    box = mapped(content_box(lines, images, CONTENT_PAD * DPI))
    page.ignore = [as_list(mapped(b)) for b in images] + [
        as_list(b) for b in outside_bands(box, len(box) // 2, *photo.size)
    ]
    for i, ln in enumerate(lines):
        x0, y0, x1, y1 = ln.box
        poly = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        pts = mapped(ln.box)
        score = _line_score(i0, refined, poly, ref.scale)
        if edge_irregularity(pts) > MAX_IRREGULARITY:
            score = 0.0
        reason = "vertical" if ln.vertical else "out_of_charset" if charset.oov(ln.text) else None
        page.lines.append(
            PageLine(
                line_id=f"gdcscan-p{index + 1:02d}/l{i}",
                polygon=as_list(pts),
                text_original=ln.text,
                text=ln.text,
                legible=True,
                handwritten=False,
                vertical=ln.vertical,
                rec_ok=reason is None,
                rec_skip=reason,
                align_score=round(score, 3),
            )
        )
    scores = [ln.align_score or 0.0 for ln in page.lines if not ln.vertical]
    stats.update(
        status="ok" if ref.ncc_refined >= MIN_PAGE_NCC else "poor_alignment",
        ncc_homography=round(ref.ncc_homography, 3),
        ncc_refined=round(ref.ncc_refined, 3),
        flow_consistent=round(ref.consistent_fraction, 3),
        lines=len(page.lines),
        lines_low_score=sum(1 for s in scores if s < MIN_LINE_SCORE),
        lines_out_of_charset=sum(1 for ln in page.lines if ln.rec_skip == "out_of_charset"),
        ignore_regions=len(page.ignore),
    )
    return page, stats


def draw_qa(photo: Image.Image, page: Page, path: Path) -> None:
    qa = photo.copy()
    draw = ImageDraw.Draw(qa)
    for poly in page.ignore:
        draw.polygon([tuple(p) for p in poly], outline=(0, 90, 255), width=5)
    for ln in page.lines:
        ok = (ln.align_score or 0.0) >= MIN_LINE_SCORE
        color = (255, 0, 0) if ok else (255, 150, 0)  # orange: box not trusted
        draw.polygon([tuple(p) for p in ln.polygon], outline=color, width=3)
    qa.thumbnail((1600, 1600))
    qa.save(path, quality=85)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.data_root)
    source = root / "raw" / "gdc_scan" / "source"
    original, scan = source / "original.pdf", source / "scan.pdf"
    charset = Charset.load(root / "manifests" / "charset.json")
    out_dir = root / "processed" / "gdc_scan"
    qa_dir = out_dir / "aligned_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    import pypdfium2 as pdfium

    n_original = len(pdfium.PdfDocument(str(original)))
    captures = extract_captures(scan, root / "raw" / "gdc_scan" / "captures")
    if len(captures) != n_original:
        raise SystemExit(f"{len(captures)} photos for {n_original} pages")

    pages: list[Page] = []
    results: list[dict[str, Any]] = []
    for i, path in enumerate(captures):
        rel = path.relative_to(root).as_posix()
        with Image.open(path) as raw_img:
            # Honour a phone rotation tag as every reader of the page file does (GDC v1 too).
            photo = ImageOps.exif_transpose(raw_img).convert("RGB")
        page, stats = build_page(i, photo, rel, original, charset)
        results.append({"file": rel, "sha256": _sha256(path), "size": list(photo.size), **stats})
        if stats["status"] != "not_found":
            pages.append(page)
            draw_qa(photo, page, qa_dir / f"{path.stem}.jpg")
        print(f"{path.stem}: {stats}")

    out_pages = out_dir / "pages_test.jsonl.gz"
    if out_pages.exists():
        changed = changed_lines(list(read_pages(out_pages)), pages)
        if changed:
            print(f"refusing to write: {len(changed)} ground-truth lines would change, e.g. "
                  f"{changed[:3]}")  # fmt: skip
            return 1
    write_pages(out_pages, pages)
    problems = [r for r in results if r["status"] != "ok"]
    report = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "rules": {
            "dpi": DPI,
            "work_scale": WORK_SCALE,
            "feature_scale": FEATURE_SCALE,
            "min_inliers": MIN_INLIERS,
            "min_page_ncc": MIN_PAGE_NCC,
            "min_line_score": MIN_LINE_SCORE,
            "max_irregularity": MAX_IRREGULARITY,
            "content_pad_in": CONTENT_PAD,
        },
        "photos": len(captures),
        "pages": len(pages),
        "lines": sum(len(p.lines) for p in pages),
        "problems": problems,
        "results": results,
    }
    man_dir = root / "manifests" / "gdc_scan"
    man_dir.mkdir(parents=True, exist_ok=True)
    (man_dir / "alignment.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(pages)}/{len(captures)} pages aligned, {report['lines']} lines, "
          f"{len(problems)} problems")  # fmt: skip

    if args.freeze:
        if problems:
            print("not freezing: every page must align without problems")
            return 1
        frozen = {
            "gdc_scan_version": GDC_SCAN_VERSION,
            "frozen_utc": report["generated_utc"],
            "frozen": True,
            "pages": len(pages),
            "lines": report["lines"],
            "pages_sha256": _sha256(out_pages),
            "sources_sha256": {"original": _sha256(original), "scan": _sha256(scan)},
            "captures": {r["file"]: r["sha256"] for r in results},
        }
        (man_dir / "gdc_scan_v1.json").write_text(
            json.dumps(frozen, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"GDC-scan v{GDC_SCAN_VERSION} frozen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
