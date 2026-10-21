from __future__ import annotations

import gzip
import json
import shutil

import cv2
import gdc_align
import numpy as np
from PIL import Image, ImageFilter

from glypharis.data.gdc import PackLayout, PdfLine, compose_pack_page
from glypharis.data.pages import read_pages
from glypharis.data.text import Charset

LAYOUT = PackLayout(dpi=100)


def _setup(root, n_sheets=2):
    sheets = []
    images = []
    for i in range(n_sheets):
        page = Image.new("RGB", (600, 800), "white")
        lines = [
            PdfLine(f"line {i} one", (100, 100, 300, 130)),
            PdfLine("rotated", (40, 300, 60, 500), True),
        ]
        sheet, gt, ignore = compose_pack_page(
            page, lines, i, f"GDC-{i:04d}", LAYOUT, [(300, 500, 500, 700)]
        )
        sheets.append({"sheet": i, "doc_key": "doc", "page": i + 1, "condition": "photo_flat",
                       "group": "paper", "lines": gt, "ignore": ignore})  # fmt: skip
        images.append(sheet)
    gdc = root / "processed" / "gdc"
    gdc.mkdir(parents=True)
    with gzip.open(gdc / "sheets.jsonl.gz", "wt", encoding="utf-8") as fh:
        for s in sheets:
            fh.write(json.dumps(s) + "\n")
    man = root / "manifests" / "gdc"
    man.mkdir(parents=True)
    (man / "pack.json").write_text(json.dumps({"layout": {"dpi": 100}, "sheets_gt_sha256": "x"}))
    images[0].save(gdc / "print_pack.pdf", save_all=True, append_images=images[1:], resolution=100)
    Charset(list(" abcdefghijklmnopqrstuvwxyz0123456789")).save(root / "manifests" / "charset.json")
    return images


def _capture(sheet: Image.Image, path, angle_deg=4.0):
    a = np.asarray(sheet)
    h, w = a.shape[:2]
    side = int(1.2 * max(w, h))  # the whole sheet stays in frame at any angle, as in a real photo
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 0.9)
    M[:, 2] += [(side - w) / 2, (side - h) / 2]
    warped = cv2.warpAffine(a, M, (side, side), borderValue=(80, 80, 80))
    Image.fromarray(warped).filter(ImageFilter.GaussianBlur(0.8)).save(path, quality=92)


def test_alignment_report_and_freeze(tmp_path):
    root = tmp_path / "data"
    images = _setup(root)
    caps = root / "raw" / "gdc" / "captures"
    caps.mkdir(parents=True)
    _capture(images[0], caps / "a.jpg")
    _capture(images[0], caps / "b.jpg", angle_deg=-3)  # duplicate of sheet 0
    Image.new("RGB", (400, 300), "white").save(caps / "c.jpg")  # no markers

    assert gdc_align.main(["--data-root", str(root)]) == 0
    report = json.loads((root / "manifests/gdc/alignment.json").read_text(encoding="utf-8"))
    status = {r["file"].split("/")[-1]: r["status"] for r in report["results"]}
    assert status == {"a.jpg": "ok", "b.jpg": "duplicate", "c.jpg": "no_markers"}
    assert report["missing_sheets"] == [1]
    assert gdc_align.main(["--data-root", str(root), "--freeze"]) == 1  # refuses: problems remain

    pages = list(read_pages(root / "processed/gdc/pages_test.jsonl.gz"))
    assert len(pages) == 1 and pages[0].source == "gdc" and pages[0].split == "test"
    first, rotated = pages[0].lines
    assert first.text == "line 0 one" and first.rec_ok
    assert rotated.vertical and rotated.rec_skip == "vertical"
    assert len(pages[0].ignore) == 1

    for f in ("b.jpg", "c.jpg"):
        (caps / f).unlink()
    _capture(images[1], caps / "d.jpg", angle_deg=90)  # sheet photographed sideways
    assert gdc_align.main(["--data-root", str(root), "--freeze"]) == 0
    frozen = json.loads((root / "manifests/gdc/gdc_v1.json").read_text(encoding="utf-8"))
    assert frozen["frozen"] and frozen["pages"] == 2 and len(frozen["captures"]) == 2
    shutil.rmtree(root / "processed/gdc/aligned_qa")
