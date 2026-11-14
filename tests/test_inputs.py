"""Phase 14 input checks (glypharis/eval/inputs.py): records, hashes, line ids, image headers.

Everything runs on a synthetic data root in tmp_path; no prepared benchmark is read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageFile

from glypharis.data.crops import encode_jpeg
from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackWriter
from glypharis.data.pages import Page, PageLine, write_pages
from glypharis.eval import inputs
from glypharis.eval.inputs import InputError
from glypharis.eval.sets import EVAL_SETS

MARKER = "INPUTS-MARKER"


def _box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _line(line_id: str) -> PageLine:
    return PageLine(line_id, _box(0, 0, 50, 10), MARKER, MARKER, True, False, False, True)


def _pages(name: str, n: int = 2) -> list[Page]:
    return [Page(f"{name}{i}", "raw/page.png", 120, 80, name, "test",
                 [_line(f"{name}{i}/l{j}") for j in range(2)]) for i in range(n)]  # fmt: skip


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """GDC pages and pack (2 pages, 4 crops), plus the eval_crops.json record of both."""
    root = tmp_path / "data"
    (root / "raw").mkdir(parents=True)
    Image.new("RGB", (120, 80), "white").save(root / "raw/page.png")
    spec = EVAL_SETS["gdc_test"]
    pages = _pages("gdc")
    write_pages(root / spec["pages"], pages)
    crop = encode_jpeg(Image.new("L", (40, 16), 255), 95)
    with LinePackWriter(root / spec["pack"]) as w:
        for p in pages:
            for ln in p.lines:
                w.add(ln.line_id, p.doc_id, "test", "gdc", MARKER, crop, (40, 16))
    _record(root, {"gdc_test": _entry(root)})
    return root


def _entry(root: Path, **kw: object) -> dict[str, object]:
    spec = EVAL_SETS["gdc_test"]
    prefix = root / spec["pack"]
    entry = {"pages": spec["pages"], "pages_sha256": sha256_file(root / spec["pages"]),
             "pack": spec["pack"], "count": 4,
             "bin_sha256": sha256_file(prefix.with_suffix(".bin")),
             "index_sha256": sha256_file(prefix.with_suffix(".index.jsonl"))}  # fmt: skip
    return {**entry, **kw}


def _record(root: Path, doc: dict[str, object], rel: str = inputs.RECORD_MANIFESTS[0]) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(json.dumps(doc), encoding="utf-8")


# ---------------------------------------------------------------------------------- records


def test_a_complete_record_passes_and_names_where_each_value_came_from(data):
    out = inputs.check_inputs(data, "gdc_test", "rec", 2)
    assert out["pack"]["count"] == 4 and out["pages"]["count"] == 2
    assert out["pack"]["recorded_in"] == "data/manifests/processed/eval_crops.json"
    assert out["pages"]["recorded_in"] == "data/manifests/processed/eval_crops.json"
    for mode in ("e2e", "oracle"):
        assert set(inputs.check_inputs(data, "gdc_test", mode, 2)) == {"pages"}
    with pytest.raises(ValueError, match="mode must be one of"):
        inputs.check_inputs(data, "gdc_test", "pages", 2)
    with pytest.raises(InputError, match="has no pages"):
        inputs.check_inputs(data, "iiit5k_test", "e2e", None)
    assert issubclass(InputError, SystemExit)
    assert inputs.RUN_ENV == ("device", "device_name", "torch", "cuda", "cudnn", "numerics")


def test_a_count_error_names_the_manifest_that_recorded_the_count(data):
    spec = EVAL_SETS["gdc_test"]
    _record(data, {"x": {"pages": spec["pages"], "pages_sha256": _entry(data)["pages_sha256"],
                         "page_count": 3}}, inputs.RECORD_MANIFESTS[1])  # fmt: skip
    rec = inputs.recorded_inputs(data)[spec["pages"]]
    assert rec["manifest"] == "data/manifests/processed/eval_crops.json"
    assert rec["count_manifest"] == "data/manifests/processed/benchmarks.json"
    with pytest.raises(InputError, match="holds 2 items, data/manifests/processed/benchmarks"):
        inputs.check_inputs(data, "gdc_test", "e2e", 2)
    inputs.check_inputs(data, "gdc_test", "rec", None)  # pages not read: no count compared


@pytest.mark.parametrize(
    ("field", "message"),
    [("pages_sha256", "record different hashes"), ("count", "record different counts"),
     ("index_sha256", "different index_sha256 values")],
)  # fmt: skip
def test_two_manifests_that_disagree_stop_the_run(data, field, message):
    entry = _entry(data, **{field: 5 if field == "count" else "f" * 64})
    _record(data, {"y": entry}, inputs.RECORD_MANIFESTS[1])
    with pytest.raises(InputError, match=message):
        inputs.recorded_inputs(data)


@pytest.mark.parametrize(
    ("change", "message"),
    [({"bin_sha256": "0" * 64}, r"lines_test\.bin has SHA-256"),
     ({"index_sha256": "0" * 64}, r"lines_test\.index\.jsonl has SHA-256"),
     ({"count": 5}, "holds 4 items"),
     ({"expected_count": 5}, "registers 5"),
     ({"pages_sha256": "0" * 64}, r"pages_test\.jsonl\.gz has SHA-256")],
)  # fmt: skip
def test_any_difference_from_the_record_is_refused(data, change, message):
    _record(data, {"gdc_test": _entry(data, **change)})
    with pytest.raises(InputError, match=message) as err:
        inputs.check_inputs(data, "gdc_test", "rec", 2, require=False)
    assert MARKER not in str(err.value)


def test_require_refuses_unrecorded_inputs_and_packs_without_an_index_hash(data):
    entry = _entry(data)
    del entry["index_sha256"]
    _record(data, {"gdc_test": entry})
    with pytest.raises(InputError, match="no recorded index_sha256"):
        inputs.check_inputs(data, "gdc_test", "rec", 2)
    inputs.check_inputs(data, "gdc_test", "rec", 2, require=False)
    inputs.check_inputs(data, "gdc_test", "e2e", 2)  # a page mode reads no pack
    (data / inputs.RECORD_MANIFESTS[0]).unlink()
    with pytest.raises(InputError, match="no recorded hash"):
        inputs.check_inputs(data, "gdc_test", "e2e", 2)
    assert (
        inputs.check_inputs(data, "gdc_test", "rec", 2, require=False)["pack"]["recorded_in"]
        is None
    )


def test_a_pack_that_differs_from_its_own_meta_is_refused(data):
    prefix = data / EVAL_SETS["gdc_test"]["pack"]
    with prefix.with_suffix(".bin").open("ab") as fh:
        fh.write(b"x")
    with pytest.raises(InputError, match=r"does not match its own .meta.json"):
        inputs.check_inputs(data, "gdc_test", "rec", 2, require=False)


def test_gdc_scan_and_hiertext_page_records_are_read(data):
    scan = EVAL_SETS["gdc_scan_test"]["pages"]
    _record(data, {"pages": 28, "pages_sha256": "a" * 64}, inputs.GDC_SCAN_MANIFEST)
    _record(data, {"splits": {"test": {"pages_file": "processed/hiertext/pages_test.jsonl.gz",
                                       "pages_sha256": "b" * 64, "pages": 1634}}},
            inputs.HIERTEXT_MANIFEST)  # fmt: skip
    found = inputs.recorded_inputs(data)
    assert found[scan]["count"] == 28
    assert found[scan]["count_manifest"] == f"data/{inputs.GDC_SCAN_MANIFEST}"
    assert found["processed/hiertext/pages_test.jsonl.gz"]["count"] == 1634


def test_gdc_page_count_comes_from_its_freeze_record(data):
    pages = EVAL_SETS["gdc_test"]["pages"]
    sha = sha256_file(data / pages)
    _record(data, {"pages": 106, "pages_sha256": sha}, inputs.GDC_MANIFEST)
    found = inputs.recorded_inputs(data)
    assert found[pages]["count"] == 106
    assert found[pages]["count_manifest"] == f"data/{inputs.GDC_MANIFEST}"
    with pytest.raises(InputError, match=r"gdc_v1.json records 106"):
        inputs.check_inputs(data, "gdc_test", "e2e", 2, require=False)
    _record(data, {"pages": 2, "pages_sha256": "f" * 64}, inputs.GDC_MANIFEST)
    with pytest.raises(InputError, match="different"):  # disagrees with eval_crops.json's hash
        inputs.recorded_inputs(data)


# ------------------------------------------------------------------------------------ pages


def test_duplicate_line_ids_and_line_boxes(data):
    pages = _pages("gdc")
    assert inputs.duplicate_line_ids(pages) == []
    pages[1].lines.append(_line("gdc0/l1"))
    pages[1].lines.append(_line("gdc0/l1"))
    assert inputs.duplicate_line_ids(pages) == ["gdc0/l1"]
    assert set(inputs.line_boxes(data, "gdc_test")) == {f"gdc{i}/l{j}" for i in (0, 1)
                                                         for j in (0, 1)}  # fmt: skip
    write_pages(data / EVAL_SETS["gdc_test"]["pages"], pages)
    with pytest.raises(InputError, match="appears twice"):
        inputs.line_boxes(data, "gdc_test")
    assert inputs.line_boxes(data, "iiit5k_test") == {}


def _exif(orientation: int) -> Image.Exif:
    exif = Image.Exif()
    exif[0x0112] = orientation
    return exif


def test_image_sizes_come_from_the_header_with_the_orientation_applied(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (60, 40)).save(raw / "plain.png")
    Image.new("RGB", (60, 40)).save(raw / "turned.jpg", exif=_exif(6))
    Image.new("RGB", (60, 40)).save(raw / "mirrored.jpg", exif=_exif(2))
    Image.new("RGB", (60, 40)).save(raw / "turned.png", exif=_exif(8))

    def no_decode(self, *a, **k):
        raise AssertionError("pixels decoded")

    monkeypatch.setattr(ImageFile.ImageFile, "load", no_decode)
    monkeypatch.setattr(Image.Image, "load", no_decode)
    assert inputs.header_size(raw / "plain.png") == ((60, 40), 1)
    assert inputs.header_size(raw / "turned.jpg") == ((40, 60), 6)
    assert inputs.header_size(raw / "mirrored.jpg") == ((60, 40), 2)
    assert inputs.header_size(raw / "turned.png") == ((40, 60), 8)

    def page(image: str, w: int, h: int) -> Page:
        return Page(image, image, w, h, "gdc", "test", [_line(f"{image}/l0")])

    good = [page("raw/plain.png", 60, 40), page("raw/turned.jpg", 40, 60),
            page("raw/turned.png", 40, 60)]  # fmt: skip
    assert inputs.image_sizes_match(tmp_path, good) == []
    bad = [page("raw/turned.jpg", 60, 40), page("raw/missing.jpg", 60, 40)]
    problems = inputs.image_sizes_match(tmp_path, bad)
    assert problems == ["raw/turned.jpg: image is 40x60, the page record says 60x40",
                        "raw/missing.jpg: cannot read its header (FileNotFoundError)"]  # fmt: skip
    assert not any(MARKER in p for p in problems)


def test_header_check_refuses_xmp_only_orientation_and_other_formats(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    xmp = (b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/'
           b'22-rdf-syntax-ns#"><rdf:Description xmlns:tiff="http://ns.adobe.com/tiff/1.0/" '
           b'tiff:Orientation="6"/></rdf:RDF></x:xmpmeta>')  # fmt: skip
    Image.new("RGB", (60, 40)).save(raw / "xmp.jpg", xmp=xmp)
    Image.new("RGB", (60, 40)).save(raw / "page.tif")
    with Image.open(raw / "xmp.jpg") as img:
        if inputs.exif_orientation(img) == 1:
            pytest.skip("this Pillow does not read XMP orientation; nothing to guard against")
    with pytest.raises(ValueError, match="outside the EXIF block"):
        inputs.header_size(raw / "xmp.jpg")
    with pytest.raises(ValueError, match="format TIFF"):
        inputs.header_size(raw / "page.tif")
    page = Page("raw/xmp.jpg", "raw/xmp.jpg", 60, 40, "gdc", "test", [_line("x/l0")])
    assert inputs.image_sizes_match(tmp_path, [page]) == [
        "raw/xmp.jpg: cannot read its header (ValueError)"
    ]
