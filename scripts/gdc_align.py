"""Align GDC captures to their sheets and (optionally) freeze GDC.

Usage:
    python scripts/gdc_align.py [--captures data/raw/gdc/captures] [--freeze]

For every image in the captures folder:
1. find the corner markers, identify the sheet and fit the sheet-to-photo homography;
2. refine it with smooth optical flow against the clean sheet (rendered from the print pack), which
   corrects bent or curled paper and lens distortion (`glypharis.data.gdc.refine_alignment`);
3. map the sheet's ground-truth lines (densified along their top and bottom edges) and ignore
   regions into the photo, and score each line's local alignment. The ignore regions are the source
   PDF's raster images plus the sheet outside the content box (markers and the printed sheet label,
   which has no ground truth).

Re-running over an existing pages file must reproduce every ground-truth line exactly (text,
polygon, flags and alignment score); otherwise nothing is written. Only ignore regions may change.

Writes:
    data/processed/gdc/pages_test.jsonl.gz     one page record per aligned capture (split "test")
    data/processed/gdc/aligned_qa/<sheet>.jpg  ground truth drawn on each capture, for checking
    data/manifests/gdc/alignment.json          per-capture result, missing and duplicate sheets
With --freeze (only when every sheet is aligned without problems):
    data/manifests/gdc/gdc_v1.json             hashes of captures and pages: the frozen record
                                               (version 1.1 adds the sheet-margin ignore regions)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from glypharis.data.gdc import (
    PackLayout,
    Refinement,
    align_capture,
    densify_box,
    edge_irregularity,
    ncc,
    refine_alignment,
    sheet_margin_bands,
)
from glypharis.data.pages import Page, PageLine, read_pages, write_pages
from glypharis.data.text import Charset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
WORK_SCALE = 0.5  # refinement grid: half the sheet resolution (150 dpi for a 300 dpi pack)
MIN_PAGE_NCC = 0.5  # refined sheet/photo agreement below this asks for a retake
MIN_LINE_SCORE = 0.3  # lines below this have uncertain boxes (text ground truth is still exact)
MAX_IRREGULARITY = 0.25  # zigzag mapped polygons (in line heights) get score 0
DENSIFY_STEP = 100.0  # sheet pixels between polygon points along a line
GDC_VERSION = "1.1"  # 1.0: lines frozen 2026-10-21; 1.1: sheet-margin ignore regions added


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sheets(path: Path) -> dict[int, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {rec["sheet"]: rec for rec in map(json.loads, fh)}


def render_sheet(pack_pdf: Path, sheet: int, dpi: float) -> np.ndarray:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pack_pdf))
    try:
        page = pdf[sheet]
        img = page.render(scale=dpi / 72).to_pil().convert("L")
        page.close()
        return np.asarray(img)
    finally:
        pdf.close()


def _line_score(
    i0: np.ndarray, refined: np.ndarray, poly: list[list[float]], scale: float
) -> float:
    xs = [p[0] * scale for p in poly]
    ys = [p[1] * scale for p in poly]
    h, w = i0.shape
    x0, x1 = max(0, int(min(xs)) - 2), min(w, int(max(xs)) + 3)
    y0, y1 = max(0, int(min(ys)) - 2), min(h, int(max(ys)) + 3)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return 0.0
    a, b = i0[y0:y1, x0:x1], refined[y0:y1, x0:x1]
    return ncc(a, b, np.ones_like(a))


def capture_to_page(
    image_rel: str,
    size: tuple[int, int],
    sheet: dict[str, Any],
    ref: Refinement,
    i0: np.ndarray,
    refined: np.ndarray,
    charset: Charset,
    layout: PackLayout,
) -> Page:
    def mapped_pts(poly: list[list[float]]) -> np.ndarray:
        return ref.map_points(densify_box(poly, DENSIFY_STEP))

    def as_list(pts: np.ndarray) -> list[list[float]]:
        return [[round(float(x), 2), round(float(y), 2)] for x, y in pts]

    page = Page(
        doc_id=f"gdc-{sheet['doc_key']}-p{sheet['page']}",
        image=image_rel,
        width=size[0],
        height=size[1],
        source="gdc",
        split="test",
        ignore=[
            as_list(mapped_pts(p)) for p in [*sheet.get("ignore", []), *sheet_margin_bands(layout)]
        ],
    )
    for i, ln in enumerate(sheet["lines"]):
        if ln["vertical"]:
            reason: str | None = "vertical"
        elif charset.oov(ln["text"]):
            reason = "out_of_charset"
        else:
            reason = None
        pts = mapped_pts(ln["polygon"])
        score = _line_score(i0, refined, ln["polygon"], ref.scale)
        if edge_irregularity(pts) > MAX_IRREGULARITY:
            score = 0.0
        page.lines.append(
            PageLine(
                line_id=f"gdc-{sheet['sheet']:04d}/l{i}",
                polygon=as_list(pts),
                text_original=ln["text"],
                text=ln["text"],
                legible=True,
                handwritten=False,
                vertical=bool(ln["vertical"]),
                rec_ok=reason is None,
                rec_skip=reason,
                align_score=round(score, 3),
            )
        )
    return page


def changed_lines(old: list[Page], new: list[Page]) -> list[str]:
    """Existing ground-truth lines that a new run would change or drop (ignore regions excluded).

    New lines, from sheets captured since the last run, are allowed."""
    before = {ln.line_id: ln for p in old for ln in p.lines}
    after = {ln.line_id: ln for p in new for ln in p.lines}
    return sorted(k for k in before if k not in after or before[k] != after[k])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--captures", default=None)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.data_root)
    captures = Path(args.captures) if args.captures else root / "raw" / "gdc" / "captures"
    sheets = load_sheets(root / "processed" / "gdc" / "sheets.jsonl.gz")
    pack = json.loads((root / "manifests" / "gdc" / "pack.json").read_text(encoding="utf-8"))
    dpi = pack["layout"]["dpi"]
    layout = PackLayout(dpi=dpi)
    pack_pdf = root / "processed" / "gdc" / "print_pack.pdf"
    charset = Charset.load(root / "manifests" / "charset.json")
    qa_dir = root / "processed" / "gdc" / "aligned_qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    pages: list[Page] = []
    seen: dict[int, str] = {}
    files = sorted(p for p in captures.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    for path in files:
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
        with Image.open(path) as raw_img:
            img = ImageOps.exif_transpose(raw_img).convert("RGB")  # honour phone rotation tags
        a = align_capture(img, layout)
        entry: dict[str, Any] = {"file": rel, "sha256": _sha256(path), "size": list(img.size)}
        if a is None:
            entry["status"] = "no_markers"
        elif a.pack_page not in sheets:
            entry.update(status="unknown_sheet", sheet=a.pack_page)
        elif a.pack_page in seen:
            entry.update(status="duplicate", sheet=a.pack_page, first=seen[a.pack_page])
        else:
            sheet = sheets[a.pack_page]
            seen[a.pack_page] = rel
            sheet_img = render_sheet(pack_pdf, a.pack_page, dpi * WORK_SCALE)
            photo_gray = np.asarray(img.convert("L"))
            ref, i0, refined = refine_alignment(photo_gray, sheet_img, a.homography, WORK_SCALE)
            page = capture_to_page(rel, img.size, sheet, ref, i0, refined, charset, layout)
            pages.append(page)
            scores = [ln.align_score or 0.0 for ln in page.lines if not ln.vertical]
            low = sum(1 for s in scores if s < MIN_LINE_SCORE)
            entry.update(
                status="ok" if ref.ncc_refined >= MIN_PAGE_NCC else "poor_alignment",
                sheet=a.pack_page,
                condition=sheet["condition"],
                markers=a.markers_found,
                marker_rms_px=round(a.reprojection_error_px, 2),
                ncc_homography=round(ref.ncc_homography, 3),
                ncc_refined=round(ref.ncc_refined, 3),
                flow_consistent=round(ref.consistent_fraction, 3),
                lines=len(page.lines),
                lines_low_score=low,
            )
            qa = img.copy()
            draw = ImageDraw.Draw(qa)
            for ln in page.lines:
                ok = (ln.align_score or 0.0) >= MIN_LINE_SCORE
                color = (255, 0, 0) if ok else (255, 150, 0)  # orange: box not trusted
                draw.polygon([tuple(p) for p in ln.polygon], outline=color, width=2)
            for poly in page.ignore:
                draw.polygon([tuple(p) for p in poly], outline=(0, 90, 255), width=3)
            qa.thumbnail((1600, 1600))
            qa.save(qa_dir / f"{a.pack_page:04d}.jpg", quality=85)
        results.append(entry)

    pages.sort(key=lambda p: p.lines[0].line_id if p.lines else p.doc_id)
    out_pages = root / "processed" / "gdc" / "pages_test.jsonl.gz"
    if out_pages.exists():
        changed = changed_lines(list(read_pages(out_pages)), pages)
        if changed:
            print(f"refusing to write: {len(changed)} ground-truth lines would change, e.g. "
                  f"{changed[:3]}; the frozen lines must stay as they are")  # fmt: skip
            return 1
    write_pages(out_pages, pages)
    missing = sorted(set(sheets) - set(seen))
    report = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "rules": {
            "work_scale": WORK_SCALE,
            "min_page_ncc": MIN_PAGE_NCC,
            "min_line_score": MIN_LINE_SCORE,
            "max_irregularity": MAX_IRREGULARITY,
        },
        "captures": len(files),
        "aligned": len(seen),
        "missing_sheets": missing,
        "problems": [r for r in results if r["status"] != "ok"],
        "results": results,
    }
    man = root / "manifests" / "gdc" / "alignment.json"
    man.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(seen)}/{len(sheets)} sheets aligned, {len(missing)} missing, "
          f"{len(report['problems'])} problems")  # fmt: skip

    if args.freeze:
        if missing or report["problems"]:
            print("not freezing: every sheet must be aligned without problems")
            return 1
        frozen = {
            "gdc_version": GDC_VERSION,
            "changes": {
                "1.0": "ground-truth lines frozen 2026-10-21",
                "1.1": "ignore regions for the sheet outside the content box (markers and "
                "the printed sheet label); lines unchanged",
            },
            "frozen_utc": report["generated_utc"],
            "frozen": True,
            "pages": len(pages),
            "lines": sum(len(p.lines) for p in pages),
            "pages_sha256": _sha256(out_pages),
            "captures": {r["file"]: r["sha256"] for r in results},
            "sheets_gt_sha256": pack["sheets_gt_sha256"],
        }
        (root / "manifests" / "gdc" / "gdc_v1.json").write_text(
            json.dumps(frozen, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        print(f"GDC v{GDC_VERSION} frozen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
