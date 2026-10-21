"""Convert HierText to the internal page format and pack recognizer line crops.

Usage:
    python scripts/prepare_hiertext.py [--workers 8]

Inputs: raw HierText (Phase 2), exclusion list (Phase 3), charset (`data/manifests/charset.json`).
Outputs (data/processed is git-ignored; the manifest and report are committed):
    data/processed/hiertext/pages_<split>.jsonl.gz   all splits
    data/processed/hiertext/lines_<split>.{bin,index.jsonl,meta.json}   train and validation only;
        test crops are cut when a test evaluation is allowed (Phases 5, 14, 22)
    data/manifests/processed/hiertext.json   hashes, settings and counts
    reports/data_analysis/hiertext/preprocessing.md, line_crops.png
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from glypharis.data.crops import CropSettings, background_grey, encode_jpeg, expand_quad, rectify
from glypharis.data.exclusions import Exclusions
from glypharis.data.hiertext import SPLITS, load_split
from glypharis.data.linepack import LinePackReader, LinePackWriter
from glypharis.data.pages import (
    PAGE_FORMAT_VERSION,
    REC_SKIP_REASONS,
    Page,
    hiertext_to_page,
    write_pages,
)
from glypharis.data.text import NORMALIZATION_VERSION, Charset
from glypharis.utils.gitinfo import git_info
from glypharis.utils.logsetup import setup_logging

log = logging.getLogger("prepare_hiertext")
PACKED_SPLITS = ("train", "validation")


def crop_page(
    args: tuple[dict[str, Any], str, dict[str, Any]],
) -> list[tuple[str, str, bytes, tuple[int, int]]]:
    """Worker: cut every recognizer-usable line of one page."""
    page_dict, data_root, settings_dict = args
    settings = CropSettings(**settings_dict)
    page = Page.from_dict(page_dict)
    out = []
    with Image.open(Path(data_root) / page.image) as img:
        gray = img.convert("L")
    fill = background_grey(gray)
    for line in page.lines:
        if not line.rec_ok:
            continue
        quad = expand_quad(line.polygon, settings.margin)
        crop = rectify(gray, quad, settings, fill)
        out.append((line.line_id, line.text, encode_jpeg(crop, settings.jpeg_quality), crop.size))
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 23):
            h.update(block)
    return h.hexdigest()


def contact_sheet(reader: LinePackReader, path: Path, n: int = 40, seed: int = 0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = random.Random(seed).sample(range(len(reader)), min(n, len(reader)))
    fig, axes = plt.subplots(len(idx), 1, figsize=(10, 0.75 * len(idx)))
    for ax, i in zip(np.atleast_1d(axes), idx, strict=True):
        ax.imshow(np.asarray(reader.image(i)), cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_title(repr(reader.records[i].label), fontsize=7, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args(argv)
    root = Path(args.data_root)
    setup_logging(root / "cache" / "logs", run_id="prepare_hiertext")

    charset = Charset.load(root / "manifests" / "charset.json")
    excl = Exclusions.load(root / "manifests" / "exclusions" / "hiertext.json")
    settings = CropSettings()
    out_dir = root / "processed" / "hiertext"
    manifest: dict[str, Any] = {
        "dataset": "hiertext",
        "page_format_version": PAGE_FORMAT_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "charset_version": charset.version,
        "charset_classes": charset.num_classes,
        "exclusion_list_version": excl.version,
        "crop_settings": settings.to_dict(),
        "splits": {},
    }
    reasons: dict[str, collections.Counter[str]] = {}

    for split in SPLITS:
        _, anns = load_split(root / "raw" / "hiertext" / "gt" / f"{split}.jsonl.gz")
        pages: list[Page] = []
        for ann in anns:
            page = hiertext_to_page(
                ann,
                split,
                f"raw/hiertext/images/{split}/{ann.image_id}.jpg",
                charset,
                excl,
                settings.max_aspect,
            )
            if page is not None:
                pages.append(page)
        page_file = out_dir / f"pages_{split}.jsonl.gz"
        write_pages(page_file, pages)
        c = collections.Counter(ln.rec_skip or "rec_ok" for p in pages for ln in p.lines)
        reasons[split] = c
        entry: dict[str, Any] = {
            "pages": len(pages),
            "images_excluded": len(anns) - len(pages),
            "lines": sum(c.values()),
            "lines_rec_ok": c["rec_ok"],
            "rec_skip": {r: c[r] for r in REC_SKIP_REASONS},
            "pages_file": page_file.relative_to(root).as_posix(),
            "pages_sha256": _sha256(page_file),
        }
        log.info("%s: %d pages, %d lines, %d usable for the recognizer", split, len(pages),
                 entry["lines"], c["rec_ok"])  # fmt: skip

        if split in PACKED_SPLITS:
            prefix = out_dir / f"lines_{split}"
            start = time.monotonic()
            jobs = [(p.to_dict(), str(root), settings.to_dict()) for p in pages]
            meta = {"split": split, "source": "hiertext", "crop_settings": settings.to_dict()}
            with (
                ProcessPoolExecutor(max_workers=args.workers) as pool,
                LinePackWriter(prefix, meta) as w,
            ):
                for page, crops in zip(pages, pool.map(crop_page, jobs, chunksize=8), strict=True):
                    for key, label, data, size in crops:
                        w.add(key, page.doc_id, split, "hiertext", label, data, size)
                pack_meta = w.close()
            log.info("  packed %d crops (%.0f MiB) in %.0fs", pack_meta["count"],
                     pack_meta["bytes"] / 2**20, time.monotonic() - start)  # fmt: skip
            reader = LinePackReader(prefix)
            widths = np.array([r.width for r in reader.records], dtype=np.float64)
            entry["pack"] = {
                "prefix": prefix.relative_to(root).as_posix(),
                "count": pack_meta["count"],
                "bytes": pack_meta["bytes"],
                "bin_sha256": pack_meta["bin_sha256"],
                "width_at_height_64": {
                    "p50": float(np.percentile(widths, 50)),
                    "p95": float(np.percentile(widths, 95)),
                    "p99": float(np.percentile(widths, 99)),
                    "max": float(widths.max()),
                },
            }
            if split == "train":
                sheet = Path(args.reports) / "data_analysis" / "hiertext" / "line_crops.png"
                sheet.parent.mkdir(parents=True, exist_ok=True)
                contact_sheet(reader, sheet)
            reader.close()
        manifest["splits"][split] = entry

    man_path = root / "manifests" / "processed" / "hiertext.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")

    g = git_info()
    rows = [
        "# HierText preprocessing",
        "",
        f"Generated by `scripts/prepare_hiertext.py` on {datetime.now(UTC).date().isoformat()} "
        f"from commit `{(g.commit or 'unknown')[:7]}`. Do not edit by hand. Hashes: "
        "`data/manifests/processed/hiertext.json`.",
        "",
        f"Inputs: exclusion list v{excl.version}, charset v{charset.version} "
        f"({charset.num_classes} classes), normalisation v{NORMALIZATION_VERSION}. Crops: "
        f"height {settings.height} px, margin {settings.margin} × line height, grayscale JPEG "
        f"q{settings.jpeg_quality}; lines longer than {settings.max_aspect:g} × their height "
        "are not cut.",
        "",
        "## Lines by recognizer use",
        "",
        "| Split | Pages | Lines | Usable | " + " | ".join(REC_SKIP_REASONS) + " |",
        "|---|---|---|---|" + "---|" * len(REC_SKIP_REASONS),
    ]
    for split in SPLITS:
        e = manifest["splits"][split]
        rows.append(
            f"| {split} | {e['pages']:,} | {e['lines']:,} | {e['lines_rec_ok']:,} | "
            + " | ".join(f"{e['rec_skip'][r]:,}" for r in REC_SKIP_REASONS)
            + " |"
        )
    rows += [
        "",
        "## Packed crops",
        "",
        "| Split | Crops | Size | Width @64 px: median / p95 / p99 / max |",
        "|---|---|---|---|",
    ]
    for split in PACKED_SPLITS:
        pk = manifest["splits"][split]["pack"]
        w = pk["width_at_height_64"]
        size = f"{pk['bytes'] / 2**20:,.0f} MiB"
        dist = f"{w['p50']:.0f} / {w['p95']:.0f} / {w['p99']:.0f} / {w['max']:.0f}"
        rows.append(f"| {split} | {pk['count']:,} | {size} | {dist} |")
    rows += [
        "",
        "Test crops are not cut yet: test data is only touched in Phases 5, 14 and 22.",
        "",
        "Sample of 40 train crops: `line_crops.png` (seed 0).",
        "",
    ]
    rp = Path(args.reports) / "data_analysis" / "hiertext" / "preprocessing.md"
    rp.write_text("\n".join(rows), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
