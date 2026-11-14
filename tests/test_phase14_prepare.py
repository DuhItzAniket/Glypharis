"""Phase 14 preprocessing check and index-hash recording (scripts/phase14_prepare.py).

Everything runs on a synthetic data root in tmp_path holding every Phase 14 set; no prepared
benchmark is read and the branch and tag lookups are stand-ins.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import build_eval_crops
import phase14_prepare
import pytest
from PIL import Image

from glypharis.data.crops import encode_jpeg
from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackReader, LinePackWriter
from glypharis.data.pages import Page, PageLine, read_pages, write_pages
from glypharis.eval.sets import EVAL_SETS

MARKER = "PREPARE-MARKER"
W, H = 120, 80
PAGE_SETS = ("hiertext_val", "hiertext_test", "gdc_test", "funsd_test", "gdc_scan_test",
             "omnidocbench_test")  # fmt: skip
PACK_SETS = ("hiertext_val", "hiertext_test", "gdc_test", "funsd_test", "gdc_scan_test")
GDC_PHOTO = "raw/gdc/captures/000.jpg"
SCAN_PHOTO = "raw/gdc_scan/captures/page_01.jpg"
IMAGES = {"gdc_test": GDC_PHOTO, "gdc_scan_test": SCAN_PHOTO}


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _line(line_id: str, y: float, **kw: Any) -> PageLine:
    fields = {"legible": True, "handwritten": False, "vertical": False, "rec_ok": True, **kw}
    return PageLine(line_id, _box(5, y, 100, y + 12), MARKER, MARKER, **fields)


def _exif(orientation: int) -> Image.Exif:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif


def _pages(name: str) -> list[Page]:
    split = EVAL_SETS[name]["split"]
    image = IMAGES.get(name, "raw/page.png")
    n = 2 if name in ("hiertext_val", "hiertext_test", "omnidocbench_test") else 1
    pages = []
    for i in range(n):
        lines = [_line(f"{name}{i}/l{j}", 10 + 20 * j) for j in range(2)]
        if name == "gdc_scan_test":  # a vertical line and an untrusted box: neither is cut
            lines.append(_line(f"{name}{i}/v0", 50, vertical=True, rec_ok=False,
                               rec_skip="vertical", align_score=0.9))  # fmt: skip
            lines.append(_line(f"{name}{i}/u0", 60, align_score=0.1))
        pages.append(Page(f"{name}{i}", image, W, H, name, split, lines))
    return pages


def _pack(data: Path, name: str, keys: list[str]) -> None:
    crop = encode_jpeg(Image.new("L", (40, 16), 255), 95)
    with LinePackWriter(data / EVAL_SETS[name]["pack"]) as w:
        for key in keys:
            w.add(key, key.split("/")[0], EVAL_SETS[name]["split"], name, MARKER, crop, (40, 16))


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")


def _read(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def _pack_entry(data: Path, name: str) -> dict[str, Any]:
    prefix = data / EVAL_SETS[name]["pack"]
    return {"pack": EVAL_SETS[name]["pack"], "count": len(LinePackReader(prefix)),
            "bin_sha256": sha256_file(prefix.with_suffix(".bin")),
            "index_sha256": sha256_file(prefix.with_suffix(".index.jsonl"))}  # fmt: skip


def _pages_entry(data: Path, name: str) -> dict[str, Any]:
    pages = EVAL_SETS[name]["pages"]
    return {"pages": pages, "pages_sha256": sha256_file(data / pages)}


def _manifests(data: Path) -> None:
    """Every record the preparation writes, computed from the files now on disk."""
    crops = {}
    for name in PACK_SETS:
        crops[name] = {**_pages_entry(data, name), **_pack_entry(data, name)}
        if name in phase14_prepare.REGISTERED_PACKS:
            crops[name]["expected_count"] = 2
    _write(data / "manifests/processed/eval_crops.json", dict(sorted(crops.items())))
    bench = {
        "funsd_test": {**_pages_entry(data, "funsd_test"), "page_count": 1},
        "iiit5k_test": _pack_entry(data, "iiit5k_test"),
        "omnidocbench_en": {**_pages_entry(data, "omnidocbench_test"), "page_count": 2},
    }
    _write(data / "manifests/processed/benchmarks.json", bench)
    scan_sha, gdc_sha = sha256_file(data / SCAN_PHOTO), sha256_file(data / GDC_PHOTO)
    _write(data / phase14_prepare.GDC_SCAN_RECORD,
           {"pages": 1, "lines": 4, "pages_sha256": _pages_entry(data, "gdc_scan_test")
            ["pages_sha256"], "captures": {SCAN_PHOTO: scan_sha}})  # fmt: skip
    _write(data / phase14_prepare.GDC_SCAN_ALIGNMENT,
           {"lines": 4, "results": [{"file": SCAN_PHOTO, "sha256": scan_sha, "size": [W, H],
                                     "lines": 4, "lines_low_score": 1}]})  # fmt: skip
    _write(data / phase14_prepare.GDC_RECORD, {"captures": {GDC_PHOTO: gdc_sha}})
    _write(
        data / phase14_prepare.GDC_ALIGNMENT,
        {"results": [{"file": GDC_PHOTO, "sha256": gdc_sha, "size": [W, H]}]},
    )


@pytest.fixture
def data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "raw/gdc/captures").mkdir(parents=True)
    (root / "raw/gdc_scan/captures").mkdir(parents=True)
    Image.new("RGB", (W, H), "white").save(root / "raw/page.png")
    # A GDC phone photo stored on its side with a rotation tag (as GDC's captures may be).
    Image.new("RGB", (H, W), "white").save(root / GDC_PHOTO, exif=_exif(6))
    Image.new("RGB", (W, H), "white").save(root / SCAN_PHOTO)
    for name in PAGE_SETS:
        pages = _pages(name)
        write_pages(root / EVAL_SETS[name]["pages"], pages)
        if name in PACK_SETS:
            _pack(root, name, [ln.line_id for p in pages for ln in p.lines
                               if not ln.vertical and ln.align_score != 0.1])  # fmt: skip
    _pack(root, "iiit5k_test", ["iiit5k/0", "iiit5k/1"])
    _manifests(root)
    monkeypatch.setattr(build_eval_crops, "FUNSD_LINES", 2)
    monkeypatch.setattr(build_eval_crops, "GDC_SCAN_LINES", 4)
    monkeypatch.setattr(phase14_prepare, "require_branch", lambda name, start=None: None)
    monkeypatch.setattr(phase14_prepare, "tag_exists", lambda tag=None: False)
    # The Phase 5 batch preview reads Phase 5 run metadata; it has its own test below.
    monkeypatch.setattr(phase14_prepare, "check_phase5_batches", lambda data: ({}, []))
    return root


def _main(data: Path, command: str) -> int:
    return phase14_prepare.main([command, "--data-root", str(data)])


def _rewrite_pages(data: Path, name: str, change: Any) -> None:
    path = data / EVAL_SETS[name]["pages"]
    pages = list(read_pages(path))
    change(pages)
    write_pages(path, pages)
    _manifests(data)  # the hash is re-recorded, so only the planted fault remains


# ------------------------------------------------------------------------------------ check


def test_planned_modes_follow_both_registered_plans():
    modes = {
        name: phase14_prepare.planned_modes(name) for name in phase14_prepare.PHASE14_EVAL_SETS
    }
    assert modes == {"hiertext_val": ["rec", "e2e"], "hiertext_test": ["rec", "e2e"],
                     "gdc_test": ["rec", "e2e", "oracle"],
                     "gdc_scan_test": ["rec", "e2e", "oracle"], "funsd_test": ["rec", "e2e"],
                     "iiit5k_test": ["rec"], "omnidocbench_test": ["e2e"]}  # fmt: skip


def test_a_prepared_root_passes_and_prints_counts_only(data, capsys):
    before = {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}
    assert _main(data, "check") == 0
    out = capsys.readouterr().out
    assert "check: passed" in out and MARKER not in out
    assert "gdc_scan_test [rec, e2e, oracle]: pack 2, pages 1, lines 4, photos 1: ok" in out
    assert {p: p.read_bytes() for p in data.rglob("*") if p.is_file()} == before  # writes nothing


def _plant(data: Path, fault: str) -> None:
    crops = data / "manifests/processed/eval_crops.json"
    if fault == "pages_hash":
        doc = _read(crops)
        doc["gdc_test"]["pages_sha256"] = "0" * 64
        _write(crops, doc)
    elif fault == "index_missing":
        doc = _read(crops)
        del doc["hiertext_test"]["index_sha256"]
        _write(crops, doc)
    elif fault == "index_changed":
        with (data / f"{EVAL_SETS['iiit5k_test']['pack']}.index.jsonl").open("a") as fh:
            fh.write("\n")
    elif fault == "missing_pack":
        (data / f"{EVAL_SETS['funsd_test']['pack']}.bin").unlink()
    elif fault == "duplicate_line":
        _rewrite_pages(data, "omnidocbench_test", lambda ps: ps[1].lines.append(ps[0].lines[0]))
    elif fault == "duplicate_page":
        _rewrite_pages(data, "omnidocbench_test", lambda ps: ps.append(ps[0]))
    elif fault == "crop_without_line":
        _rewrite_pages(data, "hiertext_val", lambda ps: ps[0].lines.pop(0))
    elif fault == "size":
        _rewrite_pages(data, "funsd_test", lambda ps: setattr(ps[0], "width", W + 1))
    elif fault == "registered":
        doc = _read(crops)
        del doc["funsd_test"]["expected_count"]
        _write(crops, doc)
    elif fault == "untrusted":
        doc = _read(data / phase14_prepare.GDC_SCAN_ALIGNMENT)
        doc["results"][0]["lines_low_score"] = 2
        _write(data / phase14_prepare.GDC_SCAN_ALIGNMENT, doc)
    elif fault == "scan_rotation":
        Image.new("RGB", (W, H), "white").save(data / SCAN_PHOTO, exif=_exif(3))
        _manifests(data)
    elif fault == "scan_size":
        doc = _read(data / phase14_prepare.GDC_SCAN_ALIGNMENT)
        doc["results"][0]["size"] = [H, W]
        _write(data / phase14_prepare.GDC_SCAN_ALIGNMENT, doc)
    elif fault == "scan_bytes":
        Image.new("RGB", (W, H), "black").save(data / SCAN_PHOTO)
    elif fault == "gdc_hashes":
        doc = _read(data / phase14_prepare.GDC_RECORD)
        doc["captures"][GDC_PHOTO] = "0" * 64
        _write(data / phase14_prepare.GDC_RECORD, doc)
    elif fault == "gdc_list":
        doc = _read(data / phase14_prepare.GDC_RECORD)
        doc["captures"]["raw/gdc/captures/001.jpg"] = "0" * 64
        _write(data / phase14_prepare.GDC_RECORD, doc)
    else:
        raise AssertionError(fault)


@pytest.mark.parametrize(
    ("fault", "message"),
    [("pages_hash", r"gdc_test \(rec\): processed/gdc/pages_test.jsonl.gz has SHA-256"),
     ("index_missing", r"hiertext_test \(rec\): .*lines_test.index.jsonl has no recorded"),
     ("index_changed", r"iiit5k_test \(rec\): .*words_test.index.jsonl has SHA-256"),
     ("missing_pack", r"funsd_test \(rec\): an input cannot be read \(FileNotFoundError\)"),
     ("duplicate_line", "omnidocbench/pages_test.jsonl.gz: 1 line ids appear more than once"),
     ("duplicate_page", "omnidocbench/pages_test.jsonl.gz: 1 page ids appear more than once"),
     ("crop_without_line", "lines_validation_eval: 1 crops have no line"),
     ("size", "raw/page.png: image is 120x80, the page record says 121x80"),
     ("registered", "funsd_test records expected_count None, the plan registers 2"),
     ("untrusted", "1 lines with untrusted boxes in the page records"),
     ("scan_rotation", "page_01.jpg: carries EXIF orientation 3"),
     ("scan_size", r"page_01.jpg: image is 120x80, .*alignment.json records \[80, 120\]"),
     ("scan_bytes", r"page_01.jpg: SHA-256 [0-9a-f]{64}, data/manifests/gdc_scan/alignment"),
     ("gdc_hashes", "000.jpg: data/manifests/gdc/alignment.json and data/manifests/gdc/gdc_v1"),
     ("gdc_list", "list different captures")],
)  # fmt: skip
def test_every_failure_kind_is_reported_without_text(data, capsys, fault, message):
    _plant(data, fault)
    assert _main(data, "check") == 1
    out = capsys.readouterr().out
    fails = [ln for ln in out.splitlines() if ln.startswith("FAIL ")]
    assert any(re.search(message, ln) for ln in fails), fails
    assert MARKER not in out
    assert out.rstrip().endswith("failures")


def test_both_commands_refuse_off_the_phase_14_branch(data, monkeypatch):
    def off_branch(name, start=None):
        raise RuntimeError(f"this run must start on branch {name}; HEAD is on main")

    monkeypatch.setattr(phase14_prepare, "require_branch", off_branch)
    for command in ("check", "record-index-hashes"):
        with pytest.raises(SystemExit, match="HEAD is on main"):
            _main(data, command)


# ------------------------------------------------------------------------- index hashes


def _drop_index(data: Path) -> tuple[Path, Path]:
    crops = data / "manifests/processed/eval_crops.json"
    bench = data / "manifests/processed/benchmarks.json"
    doc = _read(crops)
    for name in ("gdc_test", "hiertext_test", "hiertext_val"):
        del doc[name]["index_sha256"]
    _write(crops, doc)
    doc = _read(bench)
    del doc["iiit5k_test"]["index_sha256"]
    _write(bench, doc)
    return crops, bench


def test_record_index_hashes_adds_the_missing_values_once(data, capsys):
    full = {p: p.read_bytes() for p in (data / "manifests/processed").iterdir()}
    crops, _ = _drop_index(data)
    assert _main(data, "check") == 1  # without them every rec run would refuse
    capsys.readouterr()
    assert _main(data, "record-index-hashes") == 0
    out = capsys.readouterr().out
    assert "record-index-hashes: 4 added" in out and MARKER not in out
    # The values equal the index files' hashes, written in place after bin_sha256.
    assert {p: p.read_bytes() for p in (data / "manifests/processed").iterdir()} == full
    keys = list(_read(crops)["gdc_test"])
    assert keys.index("index_sha256") == keys.index("bin_sha256") + 1
    # Idempotent: a second run adds nothing and changes no byte.
    assert _main(data, "record-index-hashes") == 0
    assert "record-index-hashes: 0 added" in capsys.readouterr().out
    assert {p: p.read_bytes() for p in (data / "manifests/processed").iterdir()} == full
    assert _main(data, "check") == 0


def test_record_index_hashes_never_overwrites_and_needs_the_bin_to_match(data, capsys):
    crops, _ = _drop_index(data)
    doc = _read(crops)
    doc["funsd_test"]["index_sha256"] = "e" * 64  # an existing value stays as it is
    _write(crops, doc)
    with (data / f"{EVAL_SETS['gdc_test']['pack']}.bin").open("ab") as fh:
        fh.write(b"x")
    before = {p: p.read_bytes() for p in (data / "manifests/processed").iterdir()}
    assert _main(data, "record-index-hashes") == 1
    out = capsys.readouterr().out
    assert "FAIL processed/gdc/lines_test.bin has SHA-256" in out and MARKER not in out
    # One pack failed, so nothing at all is written.
    assert {p: p.read_bytes() for p in (data / "manifests/processed").iterdir()} == before
    with (data / f"{EVAL_SETS['gdc_test']['pack']}.bin").open("r+b") as fh:
        fh.truncate(fh.seek(0, 2) - 1)
    assert _main(data, "record-index-hashes") == 0
    assert _read(crops)["funsd_test"]["index_sha256"] == "e" * 64


def test_record_index_hashes_refuses_once_the_tag_exists(data, monkeypatch):
    _drop_index(data)
    before = {p: p.read_bytes() for p in (data / "manifests/processed").iterdir()}
    seen = []
    monkeypatch.setattr(phase14_prepare, "tag_exists", lambda tag=None: seen.append(tag) or True)
    with pytest.raises(SystemExit, match="phase14-eval tag exists"):
        _main(data, "record-index-hashes")
    assert seen == ["phase14-eval"]
    assert {p: p.read_bytes() for p in (data / "manifests/processed").iterdir()} == before


def test_phase5_batches_are_checked_before_the_tag(tmp_path, monkeypatch):
    calls = []

    def check(data, system, task, batch, *, fixed=False):
        calls.append((system, task, batch, fixed))
        if system == "trocr":
            raise SystemExit(f"{system} ({task}): inference batch {batch}, Phase 5 ran [8]")
        return {"inference_batch": batch}

    monkeypatch.setattr(phase14_prepare.run_baselines, "check_phase5_batch", check)
    seen, fail = phase14_prepare.check_phase5_batches(tmp_path)
    assert ("trocr", "rec", 16, False) in calls  # TrOCR batches its crops at the runner default
    assert ("trocr", "e2e") not in {(c[0], c[1]) for c in calls}  # no detector: not run
    assert all(c[2] == 1 and c[3] for c in calls if c[0] != "trocr")
    assert ("ppocr_server", "e2e", 1, True) in calls  # the subset run is planned
    assert fail == ["trocr (rec): inference batch 16, Phase 5 ran [8]"]
    assert "tesseract (rec)" in seen


def test_records_must_be_committed_before_the_tag(tmp_path, monkeypatch):
    assert phase14_prepare.check_records_committed(tmp_path / "data") == []  # not the repo's
