"""The Phase 14 input checks: every file a run reads equals what the preparation recorded.

Shared by the GLYPHARIS runner (`scripts/ext_eval.py`), the baseline runner, the scorer and the
preprocessing check (`scripts/phase14_prepare.py`), so they cannot disagree about what "the
prepared file" is (docs/PHASE_STATUS.md, Phase 14 item 6, Order step 2).

- `recorded_inputs`: the hashes and counts the preparation wrote into data/manifests, keyed by the
  data-root-relative path of a page file or a pack prefix.
- `check_inputs`: hashes a set's page file and line pack (bytes only) and refuses
  (`InputError`, a SystemExit) on any difference from the record.
- `duplicate_line_ids`, `line_boxes`: page line ids are unique, and every crop has its line.
- `image_sizes_match`: each page image's size from its header (EXIF orientation applied, no pixel
  decoded) equals the page record's, so a run cannot stop on a size mismatch after its test access
  was logged.

Nothing here decodes an image or reads a label's text into a message: hashing and header reads are
preprocessing, not a test access.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackReader
from glypharis.data.pages import Page, read_pages
from glypharis.eval.sets import EVAL_SETS

# Data-root-relative manifests that record prepared files (pages_sha256 / bin_sha256 entries).
RECORD_MANIFESTS = ("manifests/processed/eval_crops.json", "manifests/processed/benchmarks.json")
GDC_SCAN_MANIFEST = "manifests/gdc_scan/gdc_scan_v1.json"  # the freeze record of GDC-scan's pages
GDC_MANIFEST = "manifests/gdc/gdc_v1.json"  # GDC's freeze record (page count and hash)
HIERTEXT_MANIFEST = "manifests/processed/hiertext.json"  # the Phase 4 page files
# Formats whose orientation both readers (PIL for the baselines, cv2 for GLYPHARIS) take from the
# EXIF block alone; any other format is refused by the header check rather than guessed.
HEADER_FORMATS = ("JPEG", "PNG", "WEBP")
MODES = ("rec", "e2e", "oracle")
# What every GLYPHARIS run of the phase must share (`Frozen.provenance` keys): resume identity
# and the scorer's one-environment check read this same tuple.
RUN_ENV = ("device", "device_name", "torch", "cuda", "cudnn", "numerics")
# EXIF orientations that turn the stored image by 90 or 270 degrees (width and height swap).
_SWAPPED = {5, 6, 7, 8}


class InputError(SystemExit):
    """An input differs from its record, or has none; nothing may run on it."""


# ---------------------------------------------------------------------------------- records


def recorded_inputs(data: Path) -> dict[str, dict[str, Any]]:
    """Hashes and counts the preparation recorded, keyed by the data-root-relative path of a page
    file or a pack prefix.

    Each entry holds ``sha256``, ``manifest`` (where the hash came from), ``count`` and
    ``count_manifest`` (where the count came from; None when no manifest records one), and, for a
    pack, ``index_sha256`` and ``expected_count`` when a manifest records them. Two manifests that
    disagree about one file's hash, count or index hash stop the run.
    """
    found: dict[str, dict[str, Any]] = {}

    def add(path: str, sha: str, count: Any, manifest: str, **extra: Any) -> None:
        where = f"data/{manifest}"
        entry = found.get(path)
        if entry is None:
            entry = found[path] = {"sha256": sha, "manifest": where, "count": None,
                                   "count_manifest": None}  # fmt: skip
        elif entry["sha256"] != sha:
            raise InputError(f"{path}: {entry['manifest']} and {where} record different hashes")
        if count is not None:
            if entry["count"] is None:
                entry["count"], entry["count_manifest"] = count, where
            elif int(entry["count"]) != int(count):
                raise InputError(f"{path}: {entry['count_manifest']} and {where} record "
                                 "different counts")  # fmt: skip
        for key, value in extra.items():
            if value is None:
                continue
            if key in entry and entry[key] != value:
                raise InputError(f"{path}: two manifests record different {key} values")
            entry[key] = value

    for rel in RECORD_MANIFESTS:
        path = data / rel
        if not path.is_file():
            continue
        for item in json.loads(path.read_text(encoding="utf-8")).values():
            if not isinstance(item, dict):
                continue
            if item.get("pages") and item.get("pages_sha256"):
                add(item["pages"], item["pages_sha256"], item.get("page_count"), rel)
            if item.get("pack") and item.get("bin_sha256"):
                add(item["pack"], item["bin_sha256"], item.get("count"), rel,
                    index_sha256=item.get("index_sha256"),
                    expected_count=item.get("expected_count"))  # fmt: skip
    scan = data / GDC_SCAN_MANIFEST
    if scan.is_file():
        doc = json.loads(scan.read_text(encoding="utf-8"))
        add(str(EVAL_SETS["gdc_scan_test"]["pages"]), doc["pages_sha256"], doc.get("pages"),
            GDC_SCAN_MANIFEST)  # fmt: skip
    gdc = data / GDC_MANIFEST
    if gdc.is_file():
        doc = json.loads(gdc.read_text(encoding="utf-8"))
        if doc.get("pages_sha256"):
            add(str(EVAL_SETS["gdc_test"]["pages"]), doc["pages_sha256"], doc.get("pages"),
                GDC_MANIFEST)  # fmt: skip
    hiertext = data / HIERTEXT_MANIFEST
    if hiertext.is_file():
        splits = json.loads(hiertext.read_text(encoding="utf-8")).get("splits", {})
        for split in splits.values():
            if split.get("pages_file") and split.get("pages_sha256"):
                add(split["pages_file"], split["pages_sha256"], split.get("pages"),
                    HIERTEXT_MANIFEST)  # fmt: skip
    return found


def _compare(what: str, got: str, count: int | None, recorded: dict[str, Any] | None,
             require: bool) -> None:  # fmt: skip
    if recorded is None:
        if require:
            raise InputError(f"{what} has no recorded hash in data/manifests; it was not "
                             "prepared and checked, so it is not run")  # fmt: skip
        return
    if got != recorded["sha256"]:
        raise InputError(f"{what} has SHA-256 {got}, {recorded['manifest']} records "
                         f"{recorded['sha256']}; refusing to run")  # fmt: skip
    if count is not None and recorded.get("count") is not None and int(recorded["count"]) != count:
        raise InputError(f"{what} holds {count} items, {recorded['count_manifest']} records "
                         f"{recorded['count']}; refusing to run")  # fmt: skip


def check_inputs(
    data: Path, eval_set: str, mode: str, n_pages: int | None, *, require: bool = True
) -> dict[str, Any]:
    """Hash the files a run reads and compare them with the preparation's record.

    Only bytes are hashed and nothing is decoded, so this is the preprocessing check, not an
    access. In ``rec`` mode the pack's .bin must match its own .meta.json and the record, its
    .index.jsonl (the labels) must match a recorded ``index_sha256``, and its size a recorded
    ``expected_count`` (the plan's registered pack size). The page file, read in every mode of a
    set with pages, must match its recorded hash; its page count is compared only where the pages
    were read anyway (``n_pages``). With ``require`` (every real run) an input without a recorded
    hash, or a pack without a recorded index hash, is refused.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}, not {mode}")
    spec = EVAL_SETS[eval_set]
    recorded = recorded_inputs(data)
    out: dict[str, Any] = {}
    if mode == "rec":
        if not spec["pack"]:
            raise InputError(f"{eval_set} has no line pack")
        prefix = data / spec["pack"]
        reader = LinePackReader(prefix)
        bin_sha = sha256_file(prefix.with_suffix(".bin"))
        if bin_sha != reader.meta["bin_sha256"]:
            raise InputError(f"{spec['pack']}.bin does not match its own .meta.json hash")
        rec = recorded.get(spec["pack"])
        _compare(f"{spec['pack']}.bin", bin_sha, len(reader), rec, require)
        index_sha = sha256_file(prefix.with_suffix(".index.jsonl"))
        if rec is not None:
            if rec.get("index_sha256") is None:
                if require:
                    raise InputError(f"{spec['pack']}.index.jsonl has no recorded index_sha256 in "
                                     f"{rec['manifest']}; its labels were not checked, so it is "
                                     "not run")  # fmt: skip
            elif index_sha != rec["index_sha256"]:
                raise InputError(f"{spec['pack']}.index.jsonl has SHA-256 {index_sha}, "
                                 f"{rec['manifest']} records {rec['index_sha256']}; refusing to "
                                 "run")  # fmt: skip
            expected = rec.get("expected_count")
            if expected is not None and int(expected) != len(reader):
                raise InputError(f"{spec['pack']} holds {len(reader)} crops, {rec['manifest']} "
                                 f"registers {expected}; refusing to run")  # fmt: skip
        out["pack"] = {
            "path": f"data/{spec['pack']}",
            "bin_sha256": bin_sha,
            "index_sha256": index_sha,
            "count": len(reader),
            "recorded_in": (rec or {}).get("manifest"),
        }
    elif not spec["pages"]:
        raise InputError(f"{eval_set} has no pages")
    if spec["pages"]:
        sha = sha256_file(data / spec["pages"])
        rec = recorded.get(spec["pages"])
        _compare(spec["pages"], sha, n_pages, rec, require)
        out["pages"] = {"path": f"data/{spec['pages']}", "sha256": sha, "count": n_pages,
                        "recorded_in": (rec or {}).get("manifest")}  # fmt: skip
    return out


# ------------------------------------------------------------------------------------ pages


def duplicate_line_ids(pages: Iterable[Page]) -> list[str]:
    """Line ids that appear more than once across ``pages`` (sorted; empty when all are unique).

    Rec results and oracle lines are joined to their page lines by id (Phase 15), so an id must
    name one line."""
    counts = Counter(ln.line_id for page in pages for ln in page.lines)
    return sorted(k for k, n in counts.items() if n > 1)


def line_boxes(data: Path, eval_set: str) -> dict[str, list[list[float]]]:
    """Each crop's line polygon on its page (empty for a set without pages, IIIT5K).

    Built before the model loads and before the access is logged, so a duplicated line id or a
    crop without a line is refused with nothing logged."""
    spec = EVAL_SETS[eval_set]
    boxes: dict[str, list[list[float]]] = {}
    if not spec["pages"]:
        return boxes
    for page in read_pages(data / spec["pages"]):
        for ln in page.lines:
            if ln.line_id in boxes:
                raise InputError(f"{spec['pages']}: line id {ln.line_id} appears twice")
            boxes[ln.line_id] = ln.polygon
    missing = [r.key for r in LinePackReader(data / spec["pack"]).records if r.key not in boxes]
    if missing:
        raise InputError(f"{len(missing)} crops of {spec['pack']} have no line in "
                         f"{spec['pages']} (e.g. {missing[0]})")  # fmt: skip
    return boxes


def exif_orientation(img: Image.Image) -> int:
    """The EXIF (or XMP) orientation tag of a lazily opened image; 1 when it has none.

    `Image.Image.getexif` is called directly: the PNG plugin's override loads the pixels when the
    header holds no EXIF block, and this must stay a header read."""
    value = Image.Image.getexif(img).get(ExifTags.Base.Orientation)
    try:
        return int(value) if value is not None else 1
    except (TypeError, ValueError):
        return 1


def _exif_only_orientation(img: Image.Image) -> int:
    """The orientation in the EXIF block alone (what cv2 applies); 1 when there is none."""
    raw = img.info.get("exif")
    if not raw:
        return 1
    exif = Image.Exif()
    exif.load(raw)
    value = exif.get(ExifTags.Base.Orientation)
    try:
        return int(value) if value is not None else 1
    except (TypeError, ValueError):
        return 1


def header_size(path: Path) -> tuple[tuple[int, int], int]:
    """An image's (width, height) as `open_page_image` sees it (EXIF orientation applied) and its
    orientation tag, read from the header only: no pixel is decoded.

    Refused (ValueError): a format outside `HEADER_FORMATS`, and an orientation that PIL finds
    but the EXIF block does not hold (an XMP-only tag), which PIL would apply and cv2 would not."""
    with Image.open(path) as img:
        if img.format not in HEADER_FORMATS:
            raise ValueError(f"format {img.format} is not checked")
        w, h = img.size
        orientation = exif_orientation(img)
        if orientation != _exif_only_orientation(img):
            raise ValueError("an orientation outside the EXIF block")
    return ((h, w) if orientation in _SWAPPED else (w, h)), orientation


def image_sizes_match(data: Path, pages: Iterable[Page]) -> list[str]:
    """Pages whose image size (header, EXIF orientation applied) differs from the record's
    ``width`` and ``height``, or whose image is missing or unreadable, one message each naming
    the image file. Empty when every page matches."""
    problems: list[str] = []
    for page in pages:
        try:
            (w, h), _ = header_size(data / page.image)
        except (OSError, SyntaxError, ValueError) as exc:  # missing, unreadable or not an image
            problems.append(f"{page.image}: cannot read its header ({type(exc).__name__})")
            continue
        if (w, h) != (page.width, page.height):
            problems.append(f"{page.image}: image is {w}x{h}, the page record says "
                            f"{page.width}x{page.height}")  # fmt: skip
    return problems
