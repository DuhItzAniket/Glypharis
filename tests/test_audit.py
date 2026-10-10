from __future__ import annotations

import gzip
import json
from pathlib import Path

import audit_data
import numpy as np
import pytest
from PIL import Image

from glypharis.data.audit import (
    Thresholds,
    check_line,
    duplicate_flags,
    find_duplicates,
    find_text_overlaps,
    image_stats,
    invalid_unicode_chars,
    line_geometry,
    summarise,
    text_overlap_flags,
)
from glypharis.data.hiertext import ImageAnnotation, Line, Word, parse_record

TH = Thresholds()


def _line(vertices, text="hello", legible=True, words=None, line_id="img/p0/l0") -> Line:
    if words is None:
        words = tuple(Word(vertices, w, True, False, False) for w in text.split())
    return Line(line_id, tuple(vertices), text, legible, False, False, words)


def _ann(lines, w=100, h=50, image_id="img") -> ImageAnnotation:
    return ImageAnnotation(image_id, w, h, 1, tuple(lines))


RECT = ((10, 10), (60, 10), (60, 20), (10, 20))

# --------------------------------------------------------------------------------- reader


def test_parse_record_assigns_stable_line_ids():
    rec = {
        "image_id": "abc",
        "image_width": 10,
        "image_height": 5,
        "paragraphs": [
            {"lines": [{"vertices": [[0, 0], [1, 0], [1, 1], [0, 1]], "text": "a", "words": []}]},
            {"lines": [{}, {"text": "b"}]},
        ],
    }
    ann = parse_record(rec)
    assert [line.line_id for line in ann.lines] == ["abc/p0/l0", "abc/p1/l0", "abc/p1/l1"]
    assert ann.paragraphs == 2
    assert ann.lines[1].legible is True  # missing flag defaults to legible


# ------------------------------------------------------------------------------- geometry


def test_geometry_of_axis_aligned_box():
    g = line_geometry(RECT, 100, 50)
    assert (g.width, g.height, g.angle_deg, g.area, g.inside_fraction) == (50, 10, 0, 500, 1.0)


def test_geometry_of_rotated_and_partly_outside_box():
    g = line_geometry(((0, 0), (10, 10), (5, 15), (-5, 5)), 100, 50)
    assert g.angle_deg == pytest.approx(45)
    assert 0 < g.inside_fraction < 1


# ---------------------------------------------------------------------------- line checks


def _codes(line, ann=None):
    ann = ann or _ann([line])
    return {f.code for f in check_line("train", ann, line, TH)}


def test_clean_line_has_no_flags():
    assert _codes(_line(RECT)) == set()


@pytest.mark.parametrize(
    ("line", "code"),
    [
        (_line(((0, 0), (1, 0), (1, 1))), "bad_polygon"),
        (_line(((5, 5), (6, 5), (6, 5), (5, 5))), "degenerate_box"),
        (_line(((90, 40), (300, 40), (300, 400), (90, 400))), "mostly_outside"),
        (_line(((-5, 10), (60, 10), (60, 20), (-5, 20))), "out_of_range"),
        (_line(RECT, text="  ", words=()), "empty_text_legible"),
        (_line(RECT, text="a\x07b", words=()), "invalid_unicode"),
        (_line(RECT, text="e\u0301", words=()), "not_nfc"),
        (_line(((10, 10), (60, 10), (60, 12), (10, 12))), "tiny_line"),
        (_line(RECT, text="ab cd", words=(Word(RECT, "abcd", True, False, False),)),
         "word_text_mismatch"),
    ],
)  # fmt: skip
def test_line_flags(line, code):
    assert code in _codes(line)


def test_illegible_lines_skip_text_checks():
    assert _codes(_line(RECT, text="", legible=False, words=())) == set()


def test_invalid_unicode_chars():
    assert invalid_unicode_chars("ok\tfine") == []
    assert invalid_unicode_chars("a�\x00") == ["U+FFFD", "U+E000", "U+0000"]


# --------------------------------------------------------------------------- image stats


def _jpeg(path: Path, seed: int, size=(300, 200)) -> Path:
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1] // 10, size[0] // 10)) * 255).astype(np.uint8)
    Image.fromarray(arr).resize(size, Image.Resampling.BILINEAR).convert("RGB").save(
        path, quality=95
    )
    return path


def test_image_stats_and_hash_robustness(tmp_path):
    a = image_stats(str(_jpeg(tmp_path / "a.jpg", 1)), long_side=128)
    assert a["ok"] and (a["width"], a["height"]) == (300, 200)
    assert 0 <= a["brightness"] <= 1 and a["blur_lapvar"] > 0
    with Image.open(tmp_path / "a.jpg") as img:
        img.resize((150, 100)).save(tmp_path / "small.jpg", quality=80)
    b = image_stats(str(tmp_path / "small.jpg"), long_side=128)
    c = image_stats(str(_jpeg(tmp_path / "c.jpg", 2)), long_side=128)
    dist = lambda x, y: bin(x ^ y).count("1")  # noqa: E731
    assert dist(a["dhash"], b["dhash"]) <= TH.dhash_max_distance
    assert dist(a["phash"], b["phash"]) <= TH.phash_max_distance
    assert dist(a["phash"], c["phash"]) > TH.phash_max_distance


def test_corrupt_image_reported_not_raised(tmp_path):
    good = _jpeg(tmp_path / "g.jpg", 3)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(good.read_bytes()[:200])
    rec = image_stats(str(bad))
    assert rec["ok"] is False and "decode" in rec["error"]
    assert image_stats(str(tmp_path / "missing.jpg"))["ok"] is False


# ---------------------------------------------------------------------------- duplicates


def test_duplicates_and_split_priority():
    items = [
        ("train", "t1", "sha-x", 0b1111, 0b1111),
        ("test", "s1", "sha-x", 0b1111, 0b1111),  # exact copy of t1
        ("validation", "v1", "sha-y", 0b1110, 0b1111),  # near copy (1 bit)
        ("train", "t2", "sha-z", 2**63 - 1, 2**62),  # unrelated
    ]
    pairs = find_duplicates(items, TH)
    assert {(p.a[1], p.b[1], p.exact) for p in pairs} == {
        ("t1", "s1", True), ("t1", "v1", False), ("s1", "v1", False),
    }  # fmt: skip
    dropped = {(f.split, f.image_id, f.code) for f in duplicate_flags(pairs)}
    assert ("train", "t1", "exact_duplicate_cross_split") in dropped
    assert ("validation", "v1", "near_duplicate_cross_split") in dropped  # test beats validation
    assert not any(s == "test" for s, _, _ in dropped)


def test_text_overlap_ignores_common_lines_and_same_split():
    shared = {f"Distinct label {i}" for i in range(6)}
    texts = {
        ("train", "a"): shared | {"only in a"},
        ("test", "b"): shared,
        ("train", "c"): shared,  # same split as a: not leakage
        ("validation", "d"): {"Distinct label 0", "something else", "more text", "another"},
    }
    pairs = find_text_overlaps(texts, TH)
    keys = {(p.a, p.b) for p in pairs}
    assert (("test", "b"), ("train", "a")) in keys
    assert (("train", "a"), ("train", "c")) not in keys
    assert all("validation" not in (p.a[0], p.b[0]) for p in pairs)  # only 1 shared line
    flags = text_overlap_flags(pairs)
    assert {(f.split, f.image_id) for f in flags} == {("train", "a"), ("train", "c")}

    common = {key: lines | {"EXIT sign"} for key, lines in texts.items()}
    tight = Thresholds(text_max_images=3, text_min_shared=7)
    assert find_text_overlaps(common, tight) == []  # "EXIT sign" is in 4 images: ignored


def test_summarise():
    s = summarise([1, 2, 3, 4])
    assert s["n"] == 4 and s["p50"] == 2.5 and s["min"] == 1 and s["max"] == 4
    assert summarise([]) == {"n": 0}


# ------------------------------------------------------------------------------ end to end


def _write_gt(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"info": {}, "annotations": records}, fh)


def _rec(image_id, w=300, h=200, lines=None):
    lines = lines or [{"vertices": [[10, 10], [100, 10], [100, 30], [10, 30]], "text": "Hello",
                       "legible": True, "words": [{"vertices": [[10, 10], [100, 10], [100, 30],
                       [10, 30]], "text": "Hello"}]}]  # fmt: skip
    return {
        "image_id": image_id,
        "image_width": w,
        "image_height": h,
        "paragraphs": [{"lines": lines}],
    }


def test_audit_script_end_to_end(tmp_path):
    root, reports = tmp_path / "data", tmp_path / "reports"
    raw = root / "raw" / "hiertext"
    for split in ("train", "validation", "test"):
        (raw / "images" / split).mkdir(parents=True)
    _jpeg(raw / "images/train/t1.jpg", 1)
    _jpeg(raw / "images/train/t2.jpg", 2)
    (raw / "images/test/s1.jpg").write_bytes((raw / "images/train/t1.jpg").read_bytes())
    _jpeg(raw / "images/validation/v1.jpg", 4, size=(320, 200))  # size mismatch vs gt
    _write_gt(raw / "gt/train.jsonl.gz", [_rec("t1"), _rec("t2"), _rec("ghost")])
    _write_gt(raw / "gt/validation.jsonl.gz", [_rec("v1")])
    _write_gt(raw / "gt/test.jsonl.gz", [_rec("s1")])

    assert audit_data.main(["--dataset", "hiertext", "--data-root", str(root),
                            "--reports", str(reports), "--workers", "1"]) == 0  # fmt: skip

    excl = json.loads((root / "manifests/exclusions/hiertext.json").read_text(encoding="utf-8"))
    reasons = {(e["split"], e["image_id"]): e["reasons"] for e in excl["images"]}
    assert reasons[("train", "t1")] == ["exact_duplicate_cross_split"]
    assert reasons[("train", "ghost")] == ["missing_image"]
    assert reasons[("validation", "v1")] == ["size_mismatch"]
    assert ("test", "s1") not in reasons
    assert excl["version"] == 1
    report = (reports / "data_quality_report.md").read_text(encoding="utf-8")
    assert "exact_duplicate_cross_split" in report
    assert (reports / "data_analysis/hiertext/distributions.png").exists()

    # Unchanged data: same exclusion list, same version.
    audit_data.main(["--dataset", "hiertext", "--data-root", str(root), "--reports", str(reports),
                     "--workers", "1"])  # fmt: skip
    again = json.loads((root / "manifests/exclusions/hiertext.json").read_text(encoding="utf-8"))
    assert again == excl
