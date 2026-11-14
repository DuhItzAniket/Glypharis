"""Cut the line crops of a page-based evaluation set into a line pack (Phases 5 and 14).

Recognition-only benchmarks need ground-truth line crops, cut once so every system (the baselines
now, GLYPHARIS later) reads exactly the same crops. Lines are chosen by
`glypharis.eval.page_metrics.is_rec_line` and cut with `glypharis.data.crops.cut_line`, which
follows curved GDC lines. The selection keeps lines with characters outside the GLYPHARIS charset
and very long lines, which the Phase 4 training packs leave out: an evaluation set must not be
shaped by one system's limits (EXPERIMENT_PROTOCOL §2). That is why HierText validation gets its
own evaluation pack instead of reusing the Phase 4 validation pack.

    python scripts/build_eval_crops.py --eval-set hiertext_val
    python scripts/build_eval_crops.py --eval-set hiertext_test
    python scripts/build_eval_crops.py --eval-set gdc_test
    python scripts/build_eval_crops.py --eval-set funsd_test
    python scripts/build_eval_crops.py --eval-set gdc_scan_test

Cutting crops is preprocessing, not evaluation: no model sees the data and nothing is scored, so
the test-access log is not touched.

Phase 14 adds GDC-scan and FUNSD with the Phase 5 settings unchanged. Its plan fixes each pack's
size in advance (docs/PHASE_STATUS.md, Phase 14 item 3: "Both counts are asserted"), and
`registered_count` asserts it on every build of those two sets: FUNSD 2,647 (`FUNSD_LINES`);
GDC-scan 779 (`GDC_SCAN_LINES`, which gdc_scan_v1.json must also record) minus its vertical lines
minus its lines with untrusted boxes, counted from the page records, where the untrusted count
must equal the low-score lines alignment.json recorded (3). The build refuses, before any image
is decoded, when the number of selected lines differs, and checks the written pack again.
`--expect-count` adds the same check for any set.

Outputs: the line pack named in SETS (git-ignored) and data/manifests/processed/eval_crops.json
(committed): the page file's and the pack's hashes, the pack's index hash (`index_sha256`, the
labels) and, for a registered set, ``expected_count``.
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
from glypharis.data.linepack import LinePackReader, LinePackWriter
from glypharis.data.pages import Page, open_page_image, read_pages
from glypharis.eval.page_metrics import has_trusted_box, is_rec_line

# Registered pack sizes (Phase 14 plan, item 3), asserted on every build of these sets.
FUNSD_LINES = 2647
GDC_SCAN_LINES = 779  # every GDC-scan line, as gdc_scan_v1.json records
GDC_SCAN_RECORD = "manifests/gdc_scan/gdc_scan_v1.json"  # under the data root
GDC_SCAN_ALIGNMENT = "manifests/gdc_scan/alignment.json"  # per photo: lines_low_score

SETS = {
    "hiertext_val": (
        "processed/hiertext/pages_validation.jsonl.gz",
        "processed/hiertext/lines_validation_eval",
    ),
    "hiertext_test": ("processed/hiertext/pages_test.jsonl.gz", "processed/hiertext/lines_test"),
    "gdc_test": ("processed/gdc/pages_test.jsonl.gz", "processed/gdc/lines_test"),
    # Phase 14
    "gdc_scan_test": (
        "processed/gdc_scan/pages_test.jsonl.gz",
        "processed/gdc_scan/lines_test",
    ),
    "funsd_test": ("processed/funsd/pages_test.jsonl.gz", "processed/funsd/lines_test"),
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


def selected_count(pages: list[Page]) -> int:
    """How many lines `crop_page` cuts, counted from the page records alone."""
    return sum(is_rec_line(ln) for p in pages for ln in p.lines)


def registered_count(root: Path, eval_set: str, pages: list[Page]) -> int | None:
    """The pack size the Phase 14 plan registered for ``eval_set`` (None for the Phase 5 sets).

    FUNSD: `FUNSD_LINES`. GDC-scan: all lines minus vertical minus untrusted-box lines, counted
    from the page records; the line total must be `GDC_SCAN_LINES` in the records and in
    gdc_scan_v1.json, and the untrusted count must equal the sum of alignment.json's
    ``lines_low_score`` (counted there over non-vertical lines). Any difference refuses."""
    if eval_set == "funsd_test":
        return FUNSD_LINES
    if eval_set != "gdc_scan_test":
        return None
    for rel in (GDC_SCAN_RECORD, GDC_SCAN_ALIGNMENT):
        if not (root / rel).is_file():
            raise SystemExit(f"gdc_scan_test: data/{rel} is missing; the registered pack size "
                             "cannot be checked, so nothing is written")  # fmt: skip
    record = json.loads((root / GDC_SCAN_RECORD).read_text(encoding="utf-8"))
    alignment = json.loads((root / GDC_SCAN_ALIGNMENT).read_text(encoding="utf-8"))
    lines = [ln for p in pages for ln in p.lines]
    if len(lines) != GDC_SCAN_LINES or record.get("lines") != GDC_SCAN_LINES:
        raise SystemExit(f"gdc_scan_test: the page records hold {len(lines)} lines and "
                         f"data/{GDC_SCAN_RECORD} records {record.get('lines')}; the plan "
                         f"registers {GDC_SCAN_LINES}; nothing written")  # fmt: skip
    vertical = sum(ln.vertical for ln in lines)
    untrusted = sum(not ln.vertical and not has_trusted_box(ln) for ln in lines)
    low = sum(int(r.get("lines_low_score", 0)) for r in alignment["results"])
    if untrusted != low:
        raise SystemExit(f"gdc_scan_test: {untrusted} lines with untrusted boxes in the page "
                         f"records, data/{GDC_SCAN_ALIGNMENT} records {low}; nothing "
                         "written")  # fmt: skip
    return len(lines) - vertical - untrusted


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crop_settings_record(settings: CropSettings) -> dict[str, Any]:
    """Crop settings as applied here; the aspect cap is a training filter, unused here."""
    rec: dict[str, Any] = settings.to_dict()
    rec["max_aspect"] = "not applied (evaluation keeps long lines)"
    return rec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-set", required=True, choices=sorted(SETS))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify", action="store_true",
                        help="re-cut in memory, check the pack is identical, refresh only "
                             "the manifest (pages file changed, lines did not)")  # fmt: skip
    parser.add_argument("--expect-count", type=int, default=None, metavar="N",
                        help="refuse unless the pack holds exactly N crops")  # fmt: skip
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    pages_rel, pack_rel = SETS[args.eval_set]
    pages = list(read_pages(root / pages_rel))
    registered = registered_count(root, args.eval_set, pages)
    expected = [n for n in (registered, args.expect_count) if n is not None]
    n_sel = selected_count(pages)
    for n in expected:
        if n_sel != n:
            raise SystemExit(f"{args.eval_set}: {n_sel} lines selected, expected {n}; nothing "
                             "written")  # fmt: skip
    settings = CropSettings()
    meta = {"eval_set": args.eval_set, "pages": pages_rel,
            "crop_settings": crop_settings_record(settings),
            "selection": "glypharis.eval.page_metrics.is_rec_line"}  # fmt: skip
    jobs = [(p.to_dict(), str(root)) for p in pages]
    if args.verify:
        reader = LinePackReader(root / pack_rel)
        n = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for crops in pool.map(crop_page, jobs, chunksize=4):
                for key, label, data, _ in crops:
                    rec = reader.records[n] if n < len(reader) else None
                    if (
                        rec is None
                        or (rec.key, rec.label) != (key, label)
                        or (reader.read_bytes(n) != data)
                    ):
                        raise SystemExit(f"{args.eval_set}: crop {n} ({key}) differs from the "
                                         "pack; rebuild it instead of verifying")  # fmt: skip
                    n += 1
        if n != len(reader):
            raise SystemExit(f"{args.eval_set}: {n} crops cut, pack has {len(reader)}")
        pack_meta = {"count": n, "bin_sha256": reader.meta["bin_sha256"]}
    else:
        with (
            LinePackWriter(root / pack_rel, meta) as writer,
            ProcessPoolExecutor(max_workers=args.workers) as pool,
        ):
            for page, crops in zip(pages, pool.map(crop_page, jobs, chunksize=4), strict=True):
                for key, label, data, size in crops:
                    writer.add(key, page.doc_id, page.split, page.source, label, data, size)
            pack_meta = writer.close()
    for n in expected:
        if pack_meta["count"] != n:
            raise SystemExit(f"{args.eval_set}: the pack holds {pack_meta['count']} crops, "
                             f"expected {n}; the manifest is not updated")  # fmt: skip

    man_path = root / "manifests" / "processed" / "eval_crops.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    entry: dict[str, Any] = {
        "pages": pages_rel,
        "pages_sha256": sha256(root / pages_rel),
        "pack": pack_rel,
        "count": pack_meta["count"],
        "bin_sha256": pack_meta["bin_sha256"],
        "index_sha256": sha256((root / pack_rel).with_suffix(".index.jsonl")),
        "crop_settings": crop_settings_record(settings),
    }
    if registered is not None:
        entry["expected_count"] = registered
    manifest[args.eval_set] = entry
    man_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=1) + "\n",
                        encoding="utf-8", newline="\n")  # fmt: skip
    print(f"{args.eval_set}: {pack_meta['count']} crops -> {pack_rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
