"""Cut the line crops of a page-based test set into a line pack (Phase 5).

Recognition-only benchmarks need ground-truth line crops. HierText train and validation crops were
cut in Phase 4; the test sets are cut here, once, so every system (the baselines now, GLYPHARIS
later) reads exactly the same crops. Lines are chosen by `glypharis.eval.page_metrics.is_rec_line`
and cut with `glypharis.data.crops.cut_line`, which follows curved GDC lines.

    python scripts/build_eval_crops.py --eval-set hiertext_test
    python scripts/build_eval_crops.py --eval-set gdc_test

Cutting crops is preprocessing, not evaluation: no model sees the data and nothing is scored, so
the test-access log is not touched.

Outputs: data/processed/<dataset>/lines_test.{bin,index.jsonl,meta.json} (git-ignored) and
data/manifests/processed/eval_crops.json (committed).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from glypharis.data.crops import CropSettings, background_grey, cut_line, encode_jpeg
from glypharis.data.linepack import LinePackWriter
from glypharis.data.pages import Page, open_page_image, read_pages
from glypharis.eval.page_metrics import is_rec_line

SETS = {
    "hiertext_test": ("processed/hiertext/pages_test.jsonl.gz", "processed/hiertext/lines_test"),
    "gdc_test": ("processed/gdc/pages_test.jsonl.gz", "processed/gdc/lines_test"),
}


def crop_page(args: tuple[dict[str, Any], str]) -> list[tuple[str, str, bytes, tuple[int, int]]]:
    page_dict, data_root = args
    page = Page.from_dict(page_dict)
    settings = CropSettings()
    gray = open_page_image(page, data_root, "L")
    fill = background_grey(gray)
    out = []
    for ln in page.lines:
        if is_rec_line(ln):
            crop = cut_line(gray, ln.polygon, settings, fill)
            out.append((ln.line_id, ln.text, encode_jpeg(crop, settings.jpeg_quality), crop.size))
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-set", required=True, choices=sorted(SETS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    pages_rel, pack_rel = SETS[args.eval_set]
    pages = list(read_pages(root / pages_rel))
    settings = CropSettings()
    meta = {"eval_set": args.eval_set, "pages": pages_rel, "crop_settings": settings.to_dict(),
            "selection": "glypharis.eval.page_metrics.is_rec_line"}  # fmt: skip
    jobs = [(p.to_dict(), str(root)) for p in pages]
    with (
        LinePackWriter(root / pack_rel, meta) as writer,
        ProcessPoolExecutor(max_workers=args.workers) as pool,
    ):
        for page, crops in zip(pages, pool.map(crop_page, jobs, chunksize=4), strict=True):
            for key, label, data, size in crops:
                writer.add(key, page.doc_id, page.split, page.source, label, data, size)
        pack_meta = writer.close()

    man_path = root / "manifests" / "processed" / "eval_crops.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    manifest[args.eval_set] = {
        "pages": pages_rel,
        "pages_sha256": sha256(root / pages_rel),
        "pack": pack_rel,
        "count": pack_meta["count"],
        "bin_sha256": pack_meta["bin_sha256"],
        "crop_settings": settings.to_dict(),
    }
    man_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n",
                        encoding="utf-8", newline="\n")  # fmt: skip
    print(f"{args.eval_set}: {pack_meta['count']} crops -> {pack_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
