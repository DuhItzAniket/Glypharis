"""Phase 15 error-analysis helpers (glypharis/eval/errors.py) on hand-made strings and rows."""

from __future__ import annotations

import math
import string
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import pytest
from rapidfuzz.distance import Levenshtein

from glypharis.eval import errors as er
from glypharis.eval import stats

CHARSET = frozenset(string.ascii_letters + string.digits + " .,-!?$")


def _unit(errs: int, chars: int, label: str, page: str, doc: str = "d") -> dict[str, Any]:
    return {"char_errors": errs, "ref_chars": chars, "label": label, "page": page, "doc": doc}


# ------------------------------------------------------------------------------ numbers only


def test_numbers_only_refused() -> None:
    assert er.NUMBERS_ONLY == ("gdc_scan_test",)
    er.refuse_numbers_only("gdc_test", "x")
    with pytest.raises(ValueError, match="numbers only"):
        er.refuse_numbers_only("gdc_scan_test", "a table")
    with pytest.raises(ValueError, match="numbers only"):
        er.Confusions("gdc_scan_test")
    with pytest.raises(ValueError, match="numbers only"):
        er.WordErrors("gdc_scan_test")


# ---------------------------------------------------------------------------- classes, ops


def test_char_class() -> None:
    got = {c: er.char_class(c, CHARSET) for c in "aZ7. $"}
    assert got == {
        "a": "lower",
        "Z": "upper",
        "7": "digit",
        ".": "punct",
        " ": "space",
        "$": "punct",
    }
    assert er.char_class("é", CHARSET) == "oov"
    assert er.char_class("é", CHARSET | {"é"}) == "lower"
    # Out of charset is checked first, before the space and punctuation rules.
    assert er.char_class(" ", frozenset("a")) == "oov"
    assert er.char_class("²", CHARSET | {"²"}) == "other"
    assert er.char_class("€", CHARSET | {"€"}) == "punct"  # a symbol, category Sc
    assert set(er.CLASSES) == {"lower", "upper", "digit", "punct", "space", "other", "oov"}


@pytest.mark.parametrize(
    ("ref", "pred"),
    [
        ("kitten", "sitting"),
        ("", "abc"),
        ("abc", ""),
        ("same", "same"),
        ("aaab", "ab"),
        ("the cat", "thecat"),
        ("rn", "m"),
    ],
)
def test_edit_ops_count_is_the_distance(ref: str, pred: str) -> None:
    ops = er.edit_ops(ref, pred)
    assert len(ops) == Levenshtein.distance(ref, pred)
    for op, r, p in ops:
        assert op in er.OPS
        assert (r == "") == (op == "ins") and (p == "") == (op == "del")


def test_edit_ops_content() -> None:
    assert er.edit_ops("kitten", "sitting") == [
        ("sub", "k", "s"),
        ("sub", "e", "i"),
        ("ins", "", "g"),
    ]
    assert er.edit_ops("ab", "b") == [("del", "a", "")]


# ----------------------------------------------------------------------------- confusions


def test_confusions_counts_and_top() -> None:
    c = er.Confusions("gdc_test")
    assert c.add("cat", "cot", char_errors=1) == 1
    c.add("the cat", "thecat")  # merged words: a deleted space
    c.add("ab", "a b")  # split word: an inserted space
    c.add("a b", "a.b")  # a substituted space
    c.add("aa", "oo")
    t = c.totals()
    assert t["lines"] == 5
    assert t["ref_chars"] == 3 + 7 + 2 + 3 + 2
    assert (t["sub"], t["ins"], t["del"]) == (4, 1, 1)
    assert t["errors"] == 6 and t["cer"] == pytest.approx(6 / 17)
    assert t["sub_rate"] == pytest.approx(4 / 17)
    assert (t["space_inserted"], t["space_deleted"], t["space_substituted"]) == (1, 1, 1)
    subs = c.top("sub")
    assert subs[0][:3] == ("a", "o", 3) and subs[0][3] == pytest.approx(3 / 6)  # "a": 6 in refs
    assert subs[1][:3] == (" ", ".", 1) and subs[1][3] == pytest.approx(1 / 2)
    assert c.top("sub", 1) == subs[:1]
    assert c.top("ins") == [("", " ", 1, None)]
    assert c.top("del") == [(" ", "", 1, 0.5)]
    with pytest.raises(ValueError, match="kind"):
        c.top("swap")


def test_confusions_checks_char_errors() -> None:
    c = er.Confusions("hiertext_test")
    with pytest.raises(ValueError, match="char_errors"):
        c.add("abc", "abd", char_errors=2)


def test_top_ties_sort_by_characters() -> None:
    c = er.Confusions("gdc_test")
    c.add("ba", "yx")
    assert [r[:3] for r in c.top("sub")] == [("a", "x", 1), ("b", "y", 1)]


def test_class_matrix_rows_sum_to_one() -> None:
    c = er.Confusions("gdc_test")
    c.add("Ab1", "Ab1")
    c.add("Ab1", "AB")  # b -> B (lower read as upper), 1 deleted
    c.add("aé", "a")  # out-of-charset reference character deleted
    c.add("a", "a!")  # an inserted punctuation mark
    m = c.class_matrix(CHARSET)
    assert m["columns"][-1] == "deleted" and m["rows"] == list(er.CLASSES)
    assert m["counts"]["lower"]["lower"] == 1 + 1 + 1 and m["counts"]["lower"]["upper"] == 1
    assert m["counts"]["digit"] == {**dict.fromkeys(er.OUTCOMES, 0), "digit": 1, "deleted": 1}
    assert m["counts"]["oov"]["deleted"] == 1
    assert m["ref_chars"]["upper"] == 2
    assert m["rates"]["lower"]["upper"] == pytest.approx(1 / 4)
    for r in er.CLASSES:
        total = sum(m["rates"][r].values())
        assert total == pytest.approx(1.0 if m["ref_chars"][r] else 0.0)
    assert sum(m["ref_chars"].values()) == c.totals()["ref_chars"]
    assert m["inserted"]["punct"] == 1


# ------------------------------------------------------------------------------ word errors


def test_word_ops_and_errors() -> None:
    assert er.word_ops("the cat sat", "the cot sat down") == [
        ("sub", "cat", "cot"),
        ("ins", "", "down"),
    ]
    assert er.word_ops("a b", "a") == [("del", "b", "")]
    assert len(er.word_ops("x y z", "q")) == Levenshtein.distance(["x", "y", "z"], ["q"])
    w = er.WordErrors("hiertext_test")
    w.add("the cat sat", "the cot sat")
    w.add("the cat", "the cot")
    w.add("hello world", "help world")  # distance 2 between the words
    w.add("one two", "one")
    w.add("one", "one extra")
    assert w.top() == [("cat", "cot", 2), ("hello", "help", 1)]
    assert w.top(1) == [("cat", "cot", 2)]
    assert w.one_char_share() == pytest.approx(2 / 3)
    t = w.totals()
    assert (t["sub"], t["ins"], t["del"], t["errors"]) == (3, 1, 1, 5)
    assert t["ref_words"] == 3 + 2 + 2 + 2 + 1
    assert er.WordErrors("funsd_test").one_char_share() == 0.0


# ------------------------------------------------------------------------------------- bins


def test_quantile_edges_and_bins() -> None:
    edges = er.quantile_edges([1, 2, 3, 4, 5, None, float("nan")])
    assert edges == (2.0, 3.0, 4.0)
    assert er.quantile_edges([1, 1, 1, 1, 9]) == (1.0,)  # duplicate edges merged
    with pytest.raises(ValueError):
        er.quantile_edges([None])
    assert [er.bin_of(v, edges) for v in (1.0, 2.0, 2.5, 3.0, 4.0, 9.0)] == [0, 1, 1, 2, 3, 3]
    assert er.bin_of(None, edges) is None and er.bin_of(float("nan"), edges) is None
    assert er.bin_labels(edges) == ["< 2", "[2, 3)", "[3, 4)", "≥ 4"]
    assert er.bin_labels(er.IOU_BINS, low=0.5) == ["[0.5, 0.7)", "[0.7, 0.85)", "≥ 0.85"]
    assert er.bin_labels(()) == ["all"]
    assert len(er.bin_labels(edges)) == len(edges) + 1


def test_rotation_and_iou_bins() -> None:
    assert er.fold_angle(-3.0) == 3.0
    assert er.fold_angle(178.0) == pytest.approx(2.0)
    assert er.fold_angle(-90.0) == 90.0 and er.fold_angle(270.0) == 90.0
    assert [er.rotation_bin(d) for d in (0.5, -2.0, 4.9, 175.0, 15.0, -60.0)] == [0, 1, 1, 2, 3, 3]
    assert er.rotation_bin(None) is None
    assert [er.bin_of(v, er.IOU_BINS) for v in (0.5, 0.7, 0.84, 0.85, 1.0)] == [0, 1, 1, 2, 2]
    assert [er.unmatched_bin(v) for v in (0.0, 0.01, 0.29, 0.3, 0.49, 0.5, 0.9)] == [
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert len(er.UNMATCHED_BINS) == 4


# --------------------------------------------------------------------------- content flags


def test_content_flags() -> None:
    assert er.has_digit("page 3") and not er.has_digit("page")
    assert er.has_punct("a, b") and er.has_punct("$5") and not er.has_punct("ab c")
    assert er.all_upper("NOTICE 1") and not er.all_upper("A") and not er.all_upper("Notice")
    assert not er.all_upper("12")
    assert er.has_long_word("an extraordinary day") and not er.has_long_word("short words")
    assert er.has_long_word("(constitution)") and not er.has_long_word("123456789012")
    assert er.has_oov("café", CHARSET) and not er.has_oov("cafe", CHARSET)


def test_label_fragment() -> None:
    assert er.is_label_fragment("a", "abcdef")  # 6 >= 3 * 1 + 3
    assert not er.is_label_fragment("a", "abcde")
    assert er.is_label_fragment("ab", "x" * 9) and not er.is_label_fragment("ab", "x" * 8)
    assert not er.is_label_fragment("abc", "x" * 50)
    assert er.is_label_fragment("", "abc")


def test_primary_category_precedence() -> None:
    edges = {"line_height_px": (10.0, 20.0, 30.0), "blur": (50.0,), "contrast": (20.0,)}
    full = {
        "oov": True,
        "rotation_deg": 8.0,
        "line_height_px": 5.0,
        "blur": 1.0,
        "contrast": 1.0,
        "digit": True,
        "punct": True,
        "long_word": True,
    }
    order = []
    attrs = dict(full)
    for cat, drop in [
        ("out_of_charset", "oov"),
        ("rotation_ge_5", "rotation_deg"),
        ("height_q1", "line_height_px"),
        ("blur_q1", "blur"),
        ("contrast_q1", "contrast"),
        ("digit", "digit"),
        ("punctuation", "punct"),
        ("long_word", "long_word"),
    ]:
        assert er.primary_category(attrs, edges) == cat
        order.append(cat)
        del attrs[drop]  # a missing field skips its rule
    assert er.primary_category(attrs, edges) == "other"
    assert tuple([*order, "other"]) == er.CATEGORIES
    # Values outside the lowest quartile, small or folded angles, and false flags do not count.
    calm = {
        "oov": False,
        "rotation_deg": 176.0,
        "line_height_px": 10.0,
        "blur": 50.0,
        "contrast": None,
        "digit": False,
    }
    assert er.primary_category(calm, edges) == "other"
    assert er.primary_category({"rotation_deg": -175.0}, {}) == "rotation_ge_5"
    assert er.primary_category({"blur": 1.0}, {}) == "other"  # no edges: rule skipped


# ------------------------------------------------------------------------------- page budget


def _pair(pred: int | None, ref_chars: int, char_errors: int, best_iou: float) -> dict[str, Any]:
    vals = {"char_errors": char_errors, "ref_chars": ref_chars}
    return {"pred": pred, "best_iou": best_iou, "strict": vals, "normalised": dict(vals)}


def test_page_budget() -> None:
    row = {
        "key": "p1",
        "pairs": [
            _pair(0, 10, 2, 0.9),
            _pair(None, 5, 5, 0.0),
            _pair(None, 4, 4, 0.4),
            _pair(1, 8, 0, 0.8),
            _pair(None, 3, 3, 0.6),
        ],
        "spurious": [2],
    }
    b = er.page_budget(row, 20, "strict")
    assert (b["missed"], b["matched"], b["spurious"]) == (12, 2, 6)
    assert b["unmatched"] == {"0": 5, "(0, 0.3)": 0, "[0.3, 0.5)": 4, "≥ 0.5": 3}
    assert sum(b["unmatched"].values()) == b["missed"]
    assert er.page_budget({**row, "spurious": []}, 14, "normalised")["spurious"] == 0


def test_page_budget_asserts() -> None:
    row = {"key": "p", "pairs": [_pair(0, 10, 2, 0.9), _pair(None, 5, 5, 0.1)], "spurious": [1]}
    with pytest.raises(ValueError, match="< 0"):
        er.page_budget(row, 6, "strict")
    with pytest.raises(ValueError, match="no spurious line"):
        er.page_budget({**row, "spurious": []}, 8, "strict")
    for bad in (float("nan"), None):
        broken = {"key": "p", "pairs": [_pair(None, 5, 5, bad)], "spurious": []}  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="unmatched bins"):
            er.page_budget(broken, 5, "strict")


# --------------------------------------------------------------------------------- line order


def test_inversion_counts() -> None:
    pairs: list[dict[str, Any]] = [{"pred": p} for p in (0, 2, None, 1, 3)]
    assert er.inversion_counts(pairs) == (1, 3)
    assert er.inversion_counts([{"pred": 3}, {"pred": 2}, {"pred": 1}]) == (2, 2)
    assert er.inversion_counts([{"pred": None}]) == (0, 0)
    assert er.inversion_counts([]) == (0, 0)


def test_top_edge_order() -> None:
    def box(y: float) -> list[list[float]]:
        return [[0, y], [10, y], [10, y + 5], [0, y + 5]]

    assert er.top_edge_order([box(0), box(10), box(10), box(5), box(20)]) == (3, 4)
    assert er.top_edge_order([box(30), box(20), box(10)]) == (0, 2)
    assert er.top_edge_order([box(1)]) == (0, 0)
    tilted = [[[0, 4], [10, 2], [10, 9], [0, 9]], [[0, 3], [10, 3], [10, 8], [0, 8]]]
    assert er.top_edge_order(tilted) == (1, 1)  # top is the minimum y: 2 then 3


# ------------------------------------------------------------------------------ slice tables


def test_slice_rows_counts_shares_and_cis() -> None:
    units = [_unit(1, 10, "a", f"p{i % 6}", f"d{i % 3}") for i in range(40)]
    units += [_unit(3, 10, "b", "p0", "d0") for _ in range(4)]
    units += [{**_unit(9, 9, "x", "p1"), "label": None}]  # left out of every slice
    rows = er.slice_rows(
        units,
        lambda u: u["label"],
        page_of=lambda u: u["page"],
        cluster=lambda u: u["doc"],
        resamples=50,
    )
    assert [r["label"] for r in rows] == ["a", "b"]
    a, b = rows
    assert (a["lines"], a["pages"], a["clusters"], a["ref_chars"], a["char_errors"]) == (
        40,
        6,
        3,
        400,
        40,
    )
    assert a["cer"] == pytest.approx(0.1)
    total = 40 + 12 + 9
    assert a["share"] == pytest.approx(40 / total) and b["share"] == pytest.approx(12 / total)
    assert a["ci_lines"].unit == "lines" and a["ci_lines"].value == pytest.approx(0.1)
    assert a["ci_pages"].unit == "pages" and a["ci_clusters"].unit == "documents"
    # Below the minimum cell size: counts only.
    assert (b["lines"], b["pages"], b["ci_lines"], b["ci_pages"], b["ci_clusters"]) == (
        4,
        1,
        None,
        None,
        None,
    )


def test_slice_rows_page_ci_keeps_empty_pages() -> None:
    """Pages with no line in the slice stay in the resample as 0/0 (Phase 15 plan, item 3)."""
    units = [_unit(i % 3, 5, "s" if i % 2 else "t", f"p{i % 10}") for i in range(80)]
    rows = er.slice_rows(units, lambda u: u["label"], page_of=lambda u: u["page"], resamples=200)
    s = next(r for r in rows if r["label"] == "s")
    pages = sorted({u["page"] for u in units})
    num = np.zeros(len(pages))
    den = np.zeros(len(pages))
    for u in units:
        if u["label"] == "s":
            num[pages.index(u["page"])] += u["char_errors"]
            den[pages.index(u["page"])] += u["ref_chars"]
    assert (den == 0).any()  # odd lines sit on odd pages only
    want = stats.ratio_ci(num, den, "pages", resamples=200)
    assert s["ci_pages"] == want
    assert s["ci_clusters"] is None and s["clusters"] is None


def test_slice_rows_without_pages_and_int_labels() -> None:
    units = [_unit(1, 4, "", "p") | {"bin": i % 2} for i in range(60)]
    rows = er.slice_rows(units, lambda u: u["bin"], resamples=20)
    assert [r["label"] for r in rows] == [0, 1]
    assert all(r["pages"] is None and r["ci_pages"] is None for r in rows)
    assert all(r["ci_lines"] is not None for r in rows)
    assert sum(r["share"] for r in rows) == pytest.approx(1.0)
    few = er.slice_rows(units, lambda u: u["bin"], min_lines=31, resamples=20)
    assert all(r["ci_lines"] is None for r in few)
    assert er.slice_rows([], lambda u: u["bin"]) == []


def test_share_ci() -> None:
    den = {"p1": 50.0, "p2": 30.0, "p3": 10.0, "p4": 10.0}
    num = {"p1": 10.0, "p2": 3.0, "p3": 2.0, "p4": 5.0}
    out = er.share_ci(num, den, resamples=100)
    assert out["value"] == pytest.approx(0.2) and out["pages"] == 4
    assert out["ci_pages"] == stats.ratio_ci(
        np.array([10.0, 3, 2, 5]), np.array([50.0, 30, 10, 10]), "pages", resamples=100
    )
    assert out["pages_80"] == 2  # 50 + 30 = 80% of 100
    assert out["clusters"] is None and out["ci_clusters"] is None
    docs = {"p1": "a", "p2": "a", "p3": "b", "p4": "c"}
    out = er.share_ci(num, den, docs, resamples=100)
    assert out["clusters"] == 3 and out["ci_clusters"].unit == "documents"
    assert out["ci_clusters"] == stats.ratio_ci(
        np.array([13.0, 2, 5]), np.array([80.0, 10, 10]), "documents", resamples=100
    )
    assert er.share_ci({"p": 0.0}, {"p": 0.0})["pages_80"] == 0
    with pytest.raises(ValueError, match="different pages"):
        er.share_ci({"p1": 1.0}, den)
    with pytest.raises(ValueError, match="without a cluster"):
        er.share_ci(num, den, {"p1": "a"})
    assert math.isclose(er.share_ci(num, {**den, "p1": 60.0})["value"], 20 / 110)


def test_slice_rows_min_pages_gate() -> None:
    """Enough lines but fewer than 5 pages: counts only."""
    units = [_unit(1, 10, "a", f"p{i % 4}") for i in range(40)]
    rows = er.slice_rows(units, lambda u: u["label"], page_of=lambda u: u["page"], resamples=20)
    assert (rows[0]["lines"], rows[0]["pages"]) == (40, 4)
    assert rows[0]["ci_lines"] is rows[0]["ci_pages"] is None
    five = [_unit(1, 10, "a", f"p{i % 5}") for i in range(40)]
    rows = er.slice_rows(five, lambda u: u["label"], page_of=lambda u: u["page"], resamples=20)
    assert rows[0]["ci_lines"] is not None and rows[0]["ci_pages"] is not None


def test_slice_rows_set_total_and_page_ci() -> None:
    units = [_unit(1, 10, "a", f"p{i % 6}", f"d{i % 3}") for i in range(40)]
    rows = er.slice_rows(units, lambda u: u["label"], total=80.0, resamples=20)
    assert rows[0]["share"] == pytest.approx(0.5)  # a share of the whole set's errors
    with pytest.raises(ValueError, match="below"):
        er.slice_rows(units, lambda u: u["label"], total=39.0)
    rows = er.slice_rows(
        units,
        lambda u: u["label"],
        page_of=lambda u: u["page"],
        cluster=lambda u: u["doc"],
        page_ci=False,
        resamples=20,
    )
    assert rows[0]["ci_pages"] is None and rows[0]["pages"] == 6
    assert rows[0]["ci_clusters"] is not None and rows[0]["ci_lines"] is not None


def test_share_ci_without_intervals() -> None:
    den = {"p1": 50.0, "p2": 30.0}
    num = {"p1": 10.0, "p2": 3.0}
    out = er.share_ci(num, den, {"p1": "a", "p2": "b"}, page_ci=False, cluster_ci=False)
    assert out["ci_pages"] is None and out["ci_clusters"] is None
    assert out["clusters"] == 2 and out["value"] == pytest.approx(13 / 80)


def test_class_matrix_counts_exact_correct_characters() -> None:
    c = er.Confusions("gdc_test")
    c.add("abc", "abd")  # c -> d: same class, not correct
    c.add("A1", "A1")
    m = c.class_matrix(CHARSET)
    assert m["counts"]["lower"]["lower"] == 3 and m["correct"]["lower"] == 2
    assert m["correct_rates"]["lower"] == pytest.approx(2 / 3)
    assert m["correct_rates"]["upper"] == 1.0 and m["correct_rates"]["digit"] == 1.0
    assert m["correct_rates"]["space"] == 0.0  # an empty row


def test_protected_folder() -> None:
    data = PurePosixPath("/repo/data")
    assert er.protected_folder(data / "processed/phase14/x", data) == "processed/phase14"
    assert er.protected_folder(data / "processed/phase14", data) == "processed/phase14"
    assert er.protected_folder(data / "processed/baselines/G/x.md", data) == "processed/baselines"
    assert (
        er.protected_folder(data / "processed/baselines_phase14/a", data)
        == "processed/baselines_phase14"
    )
    assert er.protected_folder(data / "manifests/processed/x.json", data) == "manifests/processed"
    for free in ("processed/phase15/galleries", "processed/phase140", "manifests/gdc", "../x"):
        assert er.protected_folder(data / free, data) is None
    assert er.protected_folder(PurePosixPath("/repo/reports/x"), data) is None


def test_hiertext_page() -> None:
    assert er.hiertext_page("0a1b2c/p3/l12") == "0a1b2c"
    for bad in ("0a1b2c/l12", "a/b/p1/l2", "0a1b2c/p3/l", "0a1b2c/p3/l1/x", ""):
        with pytest.raises(ValueError, match="HierText line id"):
            er.hiertext_page(bad)
