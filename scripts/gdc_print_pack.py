"""Build the GDC print pack from a page selection.

Usage:
    python scripts/gdc_print_pack.py [--selection data/manifests/gdc/selection.yaml]

The selection lists source PDFs (downloaded through data/sources.yaml as dataset ``gdc_sources``),
the pages to use, and a capture condition per page. Outputs:
    data/processed/gdc/print_pack.pdf        A4 sheets with corner markers: print this once
    data/processed/gdc/sheets.jsonl.gz       ground-truth lines per sheet, in sheet pixels
    data/processed/gdc/qa/<sheet>.png        ground-truth boxes drawn on each sheet, for checking
    data/manifests/gdc/capture_checklist.md  what to do with each sheet
    data/manifests/gdc/pack.json             hashes of the pack and the ground truth
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from glypharis.data.gdc import (
    PackLayout,
    compose_pack_page,
    extract_image_regions,
    extract_lines,
    render_page,
)

CONDITIONS = {
    "scan_app": "Phone scanner app (Google Drive scan or Microsoft Lens), page flat, auto crop off "
    "or keep the whole sheet with all four corner squares visible.",
    "photo_flat": "Plain phone photo from straight above, good daylight, sheet flat on a table.",
    "photo_angled": "Phone photo at a clear angle (about 30-45 degrees), sheet flat.",
    "low_light": "Phone photo in dim light or with a shadow across part of the sheet; no flash.",
    "distant": "Phone photo from further away (sheet fills about a third of the frame).",
    "rotated": "Sheet turned sideways or upside down on the table, photo from above.",
    "curved": "Sheet held or bent slightly (like an open book page), photo from above.",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--selection", default="data/manifests/gdc/selection.yaml")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    root = Path(args.data_root)
    sel = yaml.safe_load(Path(args.selection).read_text(encoding="utf-8"))
    layout = PackLayout(dpi=args.dpi)
    out_dir = root / "processed" / "gdc"
    qa_dir = out_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    sheets: list[Image.Image] = []
    records: list[dict[str, Any]] = []
    sheet_no = 0
    for doc in sel["documents"]:
        pdf = root / "raw" / "gdc_sources" / doc["file"]
        for entry in doc["pages"]:
            page_index, condition = int(entry["page"]) - 1, entry["condition"]
            if condition not in CONDITIONS:
                raise ValueError(f"{doc['key']} p{entry['page']}: unknown condition {condition!r}")
            scale = args.dpi / 72
            lines = extract_lines(pdf, page_index, scale)
            page_img = render_page(pdf, page_index, args.dpi)
            label = f"GDC-{sheet_no:04d}  {doc['key']} p.{entry['page']}  [{condition}]"
            regions = extract_image_regions(pdf, page_index, scale)
            sheet, gt, ignore = compose_pack_page(page_img, lines, sheet_no, label, layout, regions)
            sheets.append(sheet)
            records.append(
                {
                    "sheet": sheet_no,
                    "doc_key": doc["key"],
                    "source_file": doc["file"],
                    "page": entry["page"],
                    "condition": condition,
                    "group": doc.get("group", ""),
                    "lines": gt,
                    "ignore": ignore,
                }
            )
            qa = sheet.copy()
            draw = ImageDraw.Draw(qa)
            for ln in gt:
                color = (0, 160, 0) if ln["vertical"] else (255, 0, 0)
                draw.polygon([tuple(p) for p in ln["polygon"]], outline=color)
            for poly in ignore:
                draw.polygon([tuple(p) for p in poly], outline=(0, 90, 255), width=3)
            qa.thumbnail((1240, 1754))
            qa.save(qa_dir / f"{sheet_no:04d}.png")
            sheet_no += 1

    pack_pdf = out_dir / "print_pack.pdf"
    sheets[0].save(pack_pdf, save_all=True, append_images=sheets[1:], resolution=args.dpi)
    gt_path = out_dir / "sheets.jsonl.gz"
    with (
        open(gt_path, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
    ):
        for rec in records:
            gz.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))

    man_dir = root / "manifests" / "gdc"
    man_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["condition"]] = counts.get(rec["condition"], 0) + 1
    (man_dir / "pack.json").write_text(
        json.dumps(
            {
                "sheets": len(records),
                "lines": sum(len(r["lines"]) for r in records),
                "conditions": counts,
                "layout": {
                    "dpi": layout.dpi,
                    "paper_in": layout.paper_in,
                    "marker_in": layout.marker_in,
                },
                "sheets_gt_sha256": _sha256(gt_path),
                "selection_sha256": _sha256(Path(args.selection)),
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    check = [
        "# GDC capture checklist",
        "",
        "Print `data/processed/gdc/print_pack.pdf` once on A4, black and white is fine, at 100 % "
        "scale (no 'fit to page'). Then capture every sheet as described below and save the files "
        "in `data/raw/gdc/captures/` (any file names; the corner squares identify each sheet).",
        "",
        "Rules for every capture: all four black corner squares must be in the picture; no flash; "
        "keep the original image (no filters, no 'enhance').",
        "",
        "## Conditions",
        "",
        *[f"- **{k}** ({counts.get(k, 0)} sheets): {v}" for k, v in CONDITIONS.items()],
        "",
        "## Sheets",
        "",
        "| Sheet | Document | Page | Condition | Done |",
        "|---|---|---|---|---|",
        *[
            f"| GDC-{r['sheet']:04d} | {r['doc_key']} | {r['page']} | {r['condition']} | ☐ |"
            for r in records
        ],
        "",
    ]
    (man_dir / "capture_checklist.md").write_text("\n".join(check), encoding="utf-8", newline="\n")
    print(
        f"{len(records)} sheets, {sum(len(r['lines']) for r in records)} lines -> "
        f"{pack_pdf.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
