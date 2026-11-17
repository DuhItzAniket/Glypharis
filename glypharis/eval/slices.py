"""GDC page slices for Phase 14: capture condition and source document of every page.

Both maps are keyed by ``Page.image`` (the capture's path relative to the data root, e.g.
``raw/gdc/captures/000.jpg``), which is what ``alignment.json`` records as each result's
``file``. They are built from the tracked manifests alone, so no test page file is opened:

- ``alignment.json`` gives file -> sheet and file -> capture condition;
- ``gdc_v1.json`` (the freeze record) gives file -> capture SHA-256, checked against alignment;
- ``selection.yaml`` gives the sheets in print order (documents, then their pages, as
  `scripts/gdc_print_pack.py` numbered them), so sheet -> source document; its SHA-256 is
  checked against ``pack.json``.

Every count the Phase 14 plan registered is checked, so a changed manifest stops the scoring
instead of silently moving pages between slices.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

GDC_MANIFESTS = Path("data/manifests/gdc")
# Capture conditions of the 106 GDC v1.1 pages (pack.json, alignment.json).
GDC_CONDITION_COUNTS = {
    "scan_app": 37,
    "photo_flat": 16,
    "photo_angled": 16,
    "low_light": 11,
    "distant": 11,
    "curved": 8,
    "rotated": 7,
}
GDC_DOCUMENTS = 21
# H1′'s clean slice: the two conditions with no deliberate degradation (53 pages).
CLEAN_SLICE = ("scan_app", "photo_flat")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _results(manifests: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = _json(manifests / "alignment.json")["results"]
    files = [r["file"] for r in results]
    if len(set(files)) != len(files):
        raise ValueError("alignment.json lists a capture file twice")
    bad = [r["file"] for r in results if r.get("status") != "ok"]
    if bad:
        raise ValueError(f"alignment.json has captures that are not aligned: {bad[:3]}")
    return results


def gdc_conditions(manifests: Path = GDC_MANIFESTS) -> dict[str, str]:
    """``Page.image`` -> capture condition, for the 106 GDC pages.

    Checks that the capture files and their SHA-256 values equal the freeze record's
    (``gdc_v1.json`` ``captures``) and that the condition counts are the registered ones.
    """
    results = _results(manifests)
    frozen: dict[str, str] = _json(manifests / "gdc_v1.json")["captures"]
    files = {r["file"] for r in results}
    if files != set(frozen):
        raise ValueError(
            f"alignment.json and gdc_v1.json list different captures: "
            f"{sorted(files ^ set(frozen))[:3]}"
        )
    changed = [r["file"] for r in results if r["sha256"] != frozen[r["file"]]]
    if changed:
        raise ValueError(f"capture SHA-256 differs from gdc_v1.json: {changed[:3]}")
    conditions = {r["file"]: r["condition"] for r in results}
    counts = dict(Counter(conditions.values()))
    if counts != GDC_CONDITION_COUNTS:
        raise ValueError(f"capture condition counts {counts} != {GDC_CONDITION_COUNTS}")
    return conditions


def _sheets(manifests: Path) -> list[dict[str, Any]]:
    """The selection's sheets in print order: [{"doc_key", "page", "condition"}, ...]."""
    path = manifests / "selection.yaml"
    pack = _json(manifests / "pack.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != pack["selection_sha256"]:
        raise ValueError(f"{path.as_posix()} SHA-256 differs from pack.json's selection_sha256")
    selection = yaml.safe_load(path.read_text(encoding="utf-8"))
    sheets = [
        {"doc_key": doc["key"], "page": entry["page"], "condition": entry["condition"]}
        for doc in selection["documents"]
        for entry in doc["pages"]
    ]
    if len(sheets) != pack["sheets"]:
        raise ValueError(f"selection has {len(sheets)} sheets, pack.json {pack['sheets']}")
    return sheets


def gdc_sources(manifests: Path = GDC_MANIFESTS) -> dict[str, str]:
    """``Page.image`` -> source document key, the clusters of GDC's cluster bootstraps.

    Runs `gdc_conditions`'s checks, then checks that each capture's condition in alignment.json
    equals its sheet's in the selection, and that the pages come from the registered number of
    documents.
    """
    conditions = gdc_conditions(manifests)
    sheets = _sheets(manifests)
    sources: dict[str, str] = {}
    for r in _results(manifests):
        sheet = int(r["sheet"])
        if not 0 <= sheet < len(sheets):
            raise ValueError(f"{r['file']}: sheet {sheet} is not in the selection")
        if sheets[sheet]["condition"] != conditions[r["file"]]:
            raise ValueError(f"{r['file']}: condition differs between alignment and selection")
        sources[r["file"]] = sheets[sheet]["doc_key"]
    if len(set(sources.values())) != GDC_DOCUMENTS:
        raise ValueError(f"{len(set(sources.values()))} source documents, not {GDC_DOCUMENTS}")
    return sources
