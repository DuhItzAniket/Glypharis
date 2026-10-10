"""Measured sample counts recorded in each dataset manifest.

Counts come from the downloaded files themselves, so DATASETS.md can quote measured numbers
instead of published ones. Deep statistics are Phase 3's job (scripts/audit_data.py); these
are just totals.
"""

from __future__ import annotations

import gzip
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def _hiertext_annotation_counts(path: Path) -> dict[str, int]:
    """Images, paragraphs, lines and words in a HierText ground-truth file.

    The files are named .jsonl.gz but hold one JSON document with an ``annotations`` list; both
    layouts are accepted so a format change upstream shows up as a count change, not a crash.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    try:
        doc = json.loads(text)
        records = doc.get("annotations", [doc]) if isinstance(doc, dict) else doc
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    paragraphs = lines = words = 0
    for rec in records:
        for para in rec.get("paragraphs", []):
            paragraphs += 1
            for line in para.get("lines", []):
                lines += 1
                words += len(line.get("words", []))
    return {"images": len(records), "paragraphs": paragraphs, "lines": lines, "words": words}


def count_hiertext(raw: Path) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        entry: dict[str, Any] = {"image_files": _count_images(raw / "images" / split)}
        gt = raw / "gt" / f"{split}.jsonl.gz"
        if gt.exists():
            entry["annotations"] = _hiertext_annotation_counts(gt)
        counts[split] = entry
    return counts


def count_synthtext(raw: Path) -> dict[str, Any]:
    archive = raw / "SynthText.zip"
    if not archive.exists():
        return {}
    with zipfile.ZipFile(archive) as zf:
        names = [i.filename for i in zf.infolist() if not i.is_dir()]
    images = sum(1 for n in names if Path(n).suffix.lower() in IMAGE_SUFFIXES)
    gt = [n for n in names if n.endswith("gt.mat")]
    return {"archive_members": len(names), "images": images, "annotation_files": gt}


COUNTERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "hiertext": count_hiertext,
    "synthtext": count_synthtext,
}
