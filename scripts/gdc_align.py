"""Align GDC captures to their sheets and (optionally) freeze GDC.

Usage:
    python scripts/gdc_align.py [--captures data/raw/gdc/captures] [--freeze]

For every image in the captures folder: find the corner markers, identify the sheet, fit the
sheet-to-photo homography, and project the sheet's ground-truth lines and ignore regions into the
photo. Writes:
    data/processed/gdc/pages_test.jsonl.gz     one page record per aligned capture (split "test")
    data/processed/gdc/aligned_qa/<sheet>.jpg  ground truth drawn on each capture, for checking
    data/manifests/gdc/alignment.json          per-capture result, missing and duplicate sheets
With --freeze (only when every sheet is aligned and checked):
    data/manifests/gdc/gdc_v1.json             hashes of captures and pages: the frozen record
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

from PIL import Image, ImageDraw, ImageOps

from glypharis.data.gdc import PackLayout, align_capture, project_polygons
from glypharis.data.pages import Page, PageLine, write_pages
from glypharis.data.text import Charset

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic"}
MAX_RMS_PX = 6.0  # marker reprojection error above this is flagged for a retake


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_sheets(path: Path) -> dict[int, dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {rec["sheet"]: rec for rec in map(json.loads, fh)}


def capture_to_page(
    image_rel: str, size: tuple[int, int], sheet: dict[str, Any], H: Any, charset: Charset
) -> Page:
    polys = project_polygons([ln["polygon"] for ln in sheet["lines"]], H)
    page = Page(
        doc_id=f"gdc-{sheet['doc_key']}-p{sheet['page']}",
        image=image_rel,
        width=size[0],
        height=size[1],
        source="gdc",
        split="test",
        ignore=project_polygons(sheet.get("ignore", []), H),
    )
    for i, (ln, poly) in enumerate(zip(sheet["lines"], polys, strict=True)):
        reason = (
            "vertical"
            if ln["vertical"]
            else ("out_of_charset" if charset.oov(ln["text"]) else None)
        )
        page.lines.append(
            PageLine(
                line_id=f"gdc-{sheet['sheet']:04d}/l{i}",
                polygon=poly,
                text_original=ln["text"],
                text=ln["text"],
                legible=True,
                handwritten=False,
                vertical=bool(ln["vertical"]),
                rec_ok=reason is None,
                rec_skip=reason,
            )
        )
    return page


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
    layout = PackLayout(dpi=pack["layout"]["dpi"])
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
            status = "ok" if a.reprojection_error_px <= MAX_RMS_PX else "high_error"
            entry.update(
                status=status,
                sheet=a.pack_page,
                condition=sheet["condition"],
                markers=a.markers_found,
                rms_px=round(a.reprojection_error_px, 2),
            )
            page = capture_to_page(rel, img.size, sheet, a.homography, charset)
            pages.append(page)
            qa = img.copy()
            draw = ImageDraw.Draw(qa)
            for ln in page.lines:
                draw.polygon([tuple(p) for p in ln.polygon], outline=(255, 0, 0), width=2)
            for poly in page.ignore:
                draw.polygon([tuple(p) for p in poly], outline=(0, 90, 255), width=3)
            qa.thumbnail((1600, 1600))
            qa.save(qa_dir / f"{a.pack_page:04d}.jpg", quality=85)
        results.append(entry)

    pages.sort(key=lambda p: p.lines[0].line_id if p.lines else p.doc_id)
    out_pages = root / "processed" / "gdc" / "pages_test.jsonl.gz"
    write_pages(out_pages, pages)
    missing = sorted(set(sheets) - set(seen))
    report = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "captures": len(files),
        "aligned": sum(1 for r in results if r["status"] in ("ok", "high_error")),
        "missing_sheets": missing,
        "problems": [r for r in results if r["status"] != "ok"],
        "results": results,
    }
    man = root / "manifests" / "gdc" / "alignment.json"
    man.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{report['aligned']}/{len(sheets)} sheets aligned, {len(missing)} missing, "
          f"{len(report['problems'])} problems")  # fmt: skip

    if args.freeze:
        if missing or report["problems"]:
            print("not freezing: every sheet must be aligned without problems")
            return 1
        frozen = {
            "gdc_version": 1,
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
        print("GDC v1 frozen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
