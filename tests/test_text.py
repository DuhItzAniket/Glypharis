from __future__ import annotations

import json

import pytest

from glypharis.data.exclusions import Exclusions
from glypharis.data.text import (
    Charset,
    build_charset,
    coverage,
    in_scope,
    normalize_eval,
    normalize_label,
)

# ---------------------------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("  hello   world ", "hello world"),
        ("été", "été"),  # NFC composes
        ("ofﬁce", "office"),  # ligature written out
        ("a b c", "a b c"),  # typographic spaces
        ("zero​width­", "zerowidth"),
        ("bell\x07�", "bell"),
        ("star★️", "star★"),  # variation selector dropped
        ("line\nbreak\ttab", "line break tab"),
        ("“quoted” — dash", "“quoted” — dash"),  # kept as written
        ("CaSe", "CaSe"),
    ],
)
def test_normalize_label(raw, label):
    assert normalize_label(raw) == label


def test_normalize_label_is_idempotent():
    samples = ["ofﬁce  “x”", "é", "a  b"]
    for s in samples:
        assert normalize_label(normalize_label(s)) == normalize_label(s)


def test_eval_tracks():
    ref = "“Ofﬁce” — 2½"
    assert normalize_eval(ref, "strict") == "“Office” — 2½"
    assert normalize_eval(ref, "normalised") == '"office" - 21⁄2'
    with pytest.raises(ValueError):
        normalize_eval(ref, "loose")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------- scope


@pytest.mark.parametrize(
    ("ch", "ok"),
    [
        ("a", True), ("é", True), ("μ", True), ("7", True), ("€", True),
        (" ", True), ("—", True),
        ("о", False),  # Cyrillic o
        ("क", False),  # Devanagari ka
        ("।", False),  # Devanagari danda (punctuation of another script)
        ("・", False),  # Katakana middle dot
        ("中", False),  # CJK
    ],
)  # fmt: skip
def test_in_scope(ch, ok):
    assert in_scope(ch) is ok


# --------------------------------------------------------------------------------- charset


def test_build_charset_orders_by_codepoint_and_reports_drops():
    chars, rare, oos = build_charset({"b": 50, "a": 50, "z": 2, "о": 100}, min_count=10)
    assert chars == ["a", "b"]
    assert rare == {"z": 2}
    assert oos == {"о": 100}


def test_charset_encode_decode_and_ctc():
    cs = Charset(["a", "b", " "])
    assert cs.num_classes == 4 and cs.BLANK == 0
    ids = cs.encode("ab a")
    assert ids == [1, 2, 3, 1]
    assert cs.decode(ids) == "ab a"
    assert cs.ctc_greedy_decode([1, 1, 0, 1, 2, 2, 0, 0, 3]) == "aab "
    assert cs.oov("abc") == {"c"}
    with pytest.raises(KeyError):
        cs.encode("abc")


def test_charset_rejects_duplicates_and_multichar():
    with pytest.raises(ValueError):
        Charset(["a", "a"])
    with pytest.raises(ValueError):
        Charset(["ab"])


def test_charset_round_trip(tmp_path):
    cs = Charset(["x", "é"], version=3, metadata={"min_count": 10})
    cs.save(tmp_path / "cs.json")
    back = Charset.load(tmp_path / "cs.json")
    assert back.characters == cs.characters and back.version == 3
    assert back.metadata["min_count"] == 10
    doc = json.loads((tmp_path / "cs.json").read_text(encoding="utf-8"))
    assert doc["codepoints"] == ["U+0078", "U+00E9"]
    assert doc["vocabulary_size"] == 3


def test_coverage():
    assert coverage({"a": 3, "b": 1}, Charset(["a"])) == 0.75
    assert coverage({}, Charset(["a"])) == 1.0


def test_repo_charset_file(repo_root):
    path = repo_root / "data" / "manifests" / "charset.json"
    if not path.exists():
        pytest.skip("charset not built")
    cs = Charset.load(path)
    assert " " in cs and "a" in cs and "0" in cs
    assert all(in_scope(ch) for ch in cs.characters)
    assert cs.metadata["coverage"]["validation"]["characters_in_scope"] >= 0.999


# ------------------------------------------------------------------------------ exclusions


def test_exclusions(tmp_path):
    doc = {
        "version": 2,
        "dataset": "d",
        "images": [{"split": "train", "image_id": "i1", "reasons": ["x"]}],
        "lines": [
            {"split": "train", "line_id": "i2/p0/l0", "scope": "all", "reasons": ["y"]},
            {"split": "test", "line_id": "i3/p0/l1", "scope": "recognizer", "reasons": ["z"]},
        ],
    }
    (tmp_path / "e.json").write_text(json.dumps(doc), encoding="utf-8")
    ex = Exclusions.load(tmp_path / "e.json")
    assert ex.version == 2
    assert ex.image_excluded("train", "i1") and not ex.image_excluded("test", "i1")
    assert ex.line_excluded("train", "i2/p0/l0", for_recognizer=False)
    assert ex.line_excluded("test", "i3/p0/l1", for_recognizer=True)
    assert not ex.line_excluded("test", "i3/p0/l1", for_recognizer=False)
    assert not Exclusions.empty("d").image_excluded("train", "i1")
