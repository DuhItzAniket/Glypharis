"""Phase 14 metrics and statistics: the alnum track, unpaired and cluster bootstraps, verdicts,
ECE differences and the GDC slices. Synthetic arrays and tmp manifests only."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pytest
import yaml

import glypharis.eval.calib_stats as calib_stats
import glypharis.eval.stats as stats
from glypharis.eval.calib_stats import ece_diff_ci
from glypharis.eval.slices import (
    CLEAN_SLICE,
    GDC_CONDITION_COUNTS,
    GDC_DOCUMENTS,
    gdc_conditions,
    gdc_sources,
)
from glypharis.eval.stats import (
    Interval,
    cluster_paired_ratio_diff_ci,
    cluster_unpaired_ratio_diff_ci,
    paired_ratio_diff_ci,
    ratio_ci,
    unpaired_ratio_diff_ci,
    verdict,
)
from glypharis.eval.text_metrics import (
    RecAccumulator,
    Track,
    alnum,
    outside_charset,
    score_pair,
    track_charset,
    word_accuracy,
)
from glypharis.pipeline.calibration import ece

REPO_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------- alnum track


def test_alnum_casefolds_then_keeps_only_ascii_alphanumerics():
    assert alnum("Hello, World! 42") == "helloworld42"
    assert alnum("  O'NEIL-2 ") == "oneil2"
    assert alnum("ÉCOLE") == "cole"  # accented letters are outside 0-9a-z
    assert alnum("STRASSE") == alnum("Straße")  # casefold writes ß as ss
    assert alnum("...") == ""


def test_project_tracks_unchanged():
    assert get_args(Track) == ("strict", "normalised")


def test_score_pair_and_accumulator_on_alnum_track():
    assert score_pair("house.", "HOUSE", "alnum").char_errors == 0
    assert score_pair("house.", "HOUSE", "strict").char_errors == 6
    acc = RecAccumulator(track="alnum")
    acc.add("cat", "CAT")
    acc.add("dog", "DOT")
    acc.add("x", "--")  # empty on the alnum track: counted, not scored
    s = acc.summary()
    assert (s.samples, s.empty_references) == (2, 1)
    assert s.cer == pytest.approx(1 / 6)
    assert s.sequence_accuracy == pytest.approx(0.5)


def test_word_accuracy_arrays_feed_ratio_ci():
    refs = ["HOUSE", "CAT", "--", "Tree"]
    preds = ["house", "dog", "x", "TREE!"]
    correct, ones = word_accuracy(refs, preds, "alnum")
    assert correct.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert ones.tolist() == [1.0, 1.0, 0.0, 1.0]  # the empty reference drops out
    assert ratio_ci(correct, ones, "words").value == pytest.approx(2 / 3)
    correct_s, ones_s = word_accuracy(refs, preds, "strict")
    assert correct_s.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert ones_s.tolist() == [1.0, 1.0, 1.0, 1.0]
    acc = RecAccumulator(track="alnum")
    acc.extend(zip(preds, refs, strict=True))
    assert acc.summary().sequence_accuracy == pytest.approx(correct.sum() / ones.sum())
    with pytest.raises(ValueError):
        word_accuracy(["a"], ["a", "b"], "alnum")


def test_charset_ceiling_on_alnum_track():
    allowed = track_charset("ABCabc", "alnum")
    assert allowed == frozenset(" abc")
    assert outside_charset(["AB-CD"], allowed, "alnum") == (1, 4)  # "abcd": only d is outside


# ------------------------------------------------------------------------- bootstrap helpers


def _errors(rng: np.random.Generator, n: int, rate: float) -> tuple[np.ndarray, np.ndarray]:
    den = rng.integers(5, 40, n).astype(float)
    err = rng.binomial(den.astype(int), rate).astype(float)
    return err, den


def test_unpaired_with_factor_zero_is_ratio_ci():
    rng = np.random.default_rng(0)
    ea, da = _errors(rng, 300, 0.1)
    eb, db = _errors(rng, 200, 0.05)
    iv = unpaired_ratio_diff_ci(ea, da, eb, db, factor=0.0, unit="lines")
    ref = ratio_ci(ea, da, "lines")
    assert (iv.value, iv.low, iv.high, iv.resamples) == (ref.value, ref.low, ref.high, 1000)


def test_unpaired_value_and_independent_sides():
    rng = np.random.default_rng(1)
    ea, da = _errors(rng, 400, 0.2)
    eb, db = _errors(rng, 500, 0.1)
    iv = unpaired_ratio_diff_ci(ea, da, eb, db, factor=1.5, unit="lines")
    assert iv.value == pytest.approx(ea.sum() / da.sum() - 1.5 * eb.sum() / db.sum())
    assert iv.low < iv.value < iv.high
    # Identical sides: had both used one stream, every resample difference would be 0.
    same = unpaired_ratio_diff_ci(ea, da, ea, da, factor=1.0, unit="lines")
    assert same.value == 0.0
    assert same.low < 0 < same.high


def test_block_draws_do_not_change_the_interval(monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(2)
    ea, da = _errors(rng, 257, 0.1)
    eb, db = _errors(rng, 131, 0.07)
    b = rng.binomial(da.astype(int), 0.05).astype(float)
    clusters = rng.integers(0, 17, 257)
    full = unpaired_ratio_diff_ci(ea, da, eb, db, 1.5, "lines", resamples=200)
    full_c = cluster_paired_ratio_diff_ci(ea, b, da, clusters, "docs")
    monkeypatch.setattr(stats, "_BLOCK_CELLS", 1000)  # a few rows per block
    assert unpaired_ratio_diff_ci(ea, da, eb, db, 1.5, "lines", resamples=200) == full
    assert cluster_paired_ratio_diff_ci(ea, b, da, clusters, "docs") == full_c


def test_cluster_paired_with_singleton_clusters_is_paired_ratio_diff_ci():
    rng = np.random.default_rng(3)
    a, den = _errors(rng, 250, 0.1)
    b = rng.binomial(den.astype(int), 0.06).astype(float)
    iv = cluster_paired_ratio_diff_ci(a, b, den, np.arange(250), "pages", factor=2.0)
    ref = paired_ratio_diff_ci(a, 2 * b, den, "pages")
    assert (iv.value, iv.low, iv.high) == (ref.value, ref.low, ref.high)
    assert iv.unit == "pages"


def test_cluster_paired_widens_on_clustered_errors_and_ignores_unit_order():
    rng = np.random.default_rng(4)
    clusters = np.repeat(np.arange(12), 20)
    den = np.full(clusters.size, 30.0)
    level = rng.uniform(0.0, 0.4, 12)[clusters]  # errors vary between clusters, not within
    a = np.round(den * level)
    b = np.round(den * 0.1)
    by_unit = cluster_paired_ratio_diff_ci(a, b, den, np.arange(a.size), "pages")
    by_cluster = cluster_paired_ratio_diff_ci(a, b, den, clusters, "documents")
    assert by_cluster.value == by_unit.value
    assert by_cluster.high - by_cluster.low > 2 * (by_unit.high - by_unit.low)
    perm = rng.permutation(a.size)
    names = np.array([f"doc-{c:02d}" for c in clusters])
    shuffled = cluster_paired_ratio_diff_ci(a[perm], b[perm], den[perm], names[perm], "documents")
    assert shuffled == by_cluster


def test_cluster_unpaired_with_singleton_clusters_is_unpaired():
    rng = np.random.default_rng(5)
    ea, da = _errors(rng, 120, 0.15)
    eb, db = _errors(rng, 300, 0.05)
    ref = unpaired_ratio_diff_ci(ea, da, eb, db, 1.5, "lines")
    iv = cluster_unpaired_ratio_diff_ci(ea, da, np.arange(120), eb, db, 1.5, "lines")
    assert iv == ref
    grouped = cluster_unpaired_ratio_diff_ci(ea, da, np.arange(120) // 10, eb, db, 1.5, "docs")
    assert grouped.value == ref.value
    assert grouped.unit == "docs"


def test_bootstrap_input_checks_and_empty_sets():
    with pytest.raises(ValueError):
        unpaired_ratio_diff_ci([1, 2], [3], [1], [2], 1.0, "lines")
    with pytest.raises(ValueError):
        cluster_paired_ratio_diff_ci([1, 2], [1, 2], [3, 3], [0], "docs")
    with pytest.raises(ValueError):
        cluster_unpaired_ratio_diff_ci([1], [2], [0, 1], [1], [2], 1.0, "docs")
    empty = unpaired_ratio_diff_ci([], [], [1.0], [2.0], 1.0, "lines")
    assert empty.resamples == 0
    assert cluster_paired_ratio_diff_ci([], [], [], [], "docs").resamples == 0


@pytest.mark.parametrize(
    ("low", "high", "side", "expected"),
    [
        (-0.3, -0.1, "below", "met"),
        (0.1, 0.3, "below", "not met"),
        (-0.1, 0.1, "below", "not shown"),
        (-0.2, 0.0, "below", "not shown"),  # touching 0 is not lying below it
        (0.1, 0.3, "above", "met"),
        (-0.3, -0.1, "above", "not met"),
        (0.0, 0.2, "above", "not shown"),
    ],
)
def test_verdict(low: float, high: float, side: str, expected: str):
    assert verdict(Interval((low + high) / 2, low, high, 1000, "lines"), side) == expected


def test_verdict_rejects_unknown_side():
    with pytest.raises(ValueError):
        verdict(Interval(0.0, -1.0, 1.0, 1000, "lines"), "under")


# ------------------------------------------------------------------------------- ECE diff CI


def _calibration_lines(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.2, 1.0, n)
    correct = rng.uniform(size=n) < p * 0.8
    over = np.clip(p * 1.15, 0, 1)  # overconfident, like T = 1
    return np.clip(p * 0.95, 0, 1), over, correct


def test_ece_diff_line_bootstrap_matches_the_gw_e2e_005_scheme():
    after, before, y = _calibration_lines(400, 6)
    iv = ece_diff_ci(after, before, y, factor=1.0, unit="lines", resamples=200, seed=20261028)
    # GW-E2E-005's loop (scripts/rec_calibrate.py bootstrap_diff), written out.
    rng = np.random.default_rng(20261028)
    diffs = []
    for _ in range(200):
        s = rng.integers(0, len(y), len(y))
        diffs.append(ece(after[s], y[s]) - ece(before[s], y[s]))
    assert iv.value == pytest.approx(ece(after, y) - ece(before, y))
    assert iv.low == float(np.quantile(diffs, 0.025))
    assert iv.high == float(np.quantile(diffs, 0.975))
    assert (iv.resamples, iv.unit) == (200, "lines")


def test_ece_diff_factor_and_singleton_clusters():
    after, before, y = _calibration_lines(300, 7)
    half = ece_diff_ci(after, before, y, factor=0.5, unit="lines", resamples=100)
    assert half.value == pytest.approx(ece(after, y) - 0.5 * ece(before, y))
    singles = ece_diff_ci(after, before, y, 0.5, "lines", clusters=np.arange(300), resamples=100)
    assert singles == half


def test_ece_diff_cluster_bootstrap_keeps_whole_clusters(monkeypatch: pytest.MonkeyPatch):
    # One cluster of 10 lines and one of 1: a resample holds 2, 11 or 20 lines.
    clusters = np.array([0] * 10 + [1])
    conf = np.linspace(0.5, 0.9, 11)
    y = np.ones(11, dtype=bool)
    monkeypatch.setattr(calib_stats, "ece", lambda c, _y, _bins: float(len(c)))
    iv = ece_diff_ci(conf, conf, y, factor=0.0, unit="pages", clusters=clusters, resamples=200)
    assert iv.value == 11.0
    assert {iv.low, iv.high} <= {2.0, 11.0, 20.0}


def test_ece_diff_input_checks():
    with pytest.raises(ValueError):
        ece_diff_ci([0.5, 0.6], [0.5], [True, False], 1.0, "lines")
    with pytest.raises(ValueError):
        ece_diff_ci([0.5, 0.6], [0.5, 0.6], [True, False], 1.0, "pages", clusters=[0])
    assert ece_diff_ci([], [], [], 1.0, "lines").resamples == 0


# ------------------------------------------------------------------------------- GDC slices


def _write_manifests(root: Path, mutate: Any = None) -> Path:
    """Tmp GDC manifests with the registered counts: 106 sheets from 21 documents."""
    conds = [c for c, k in GDC_CONDITION_COUNTS.items() for _ in range(k)]
    random.Random(0).shuffle(conds)
    sizes = [6] + [5] * (GDC_DOCUMENTS - 1)
    docs, sheet = [], 0
    for d, size in enumerate(sizes):
        pages = [{"page": p + 1, "condition": conds[sheet + p]} for p in range(size)]
        docs.append({"key": f"doc{d:02d}", "file": f"src/doc{d:02d}.pdf", "pages": pages})
        sheet += size
    selection = {"seed": 11, "documents": docs}
    order = list(range(106))
    random.Random(1).shuffle(order)  # capture i shows sheet order[i]
    results = [
        {
            "file": f"raw/gdc/captures/{i:03d}.jpg",
            "sha256": hashlib.sha256(str(i).encode()).hexdigest(),
            "status": "ok",
            "sheet": s,
            "condition": conds[s],
        }
        for i, s in enumerate(order)
    ]
    state = {"selection": selection, "results": results}
    if mutate:
        mutate(state)
    man = root / "manifests" / "gdc"
    man.mkdir(parents=True)
    sel_path = man / "selection.yaml"
    sel_path.write_text(yaml.safe_dump(state["selection"], sort_keys=False), encoding="utf-8")
    pack = {"sheets": 106, "selection_sha256": hashlib.sha256(sel_path.read_bytes()).hexdigest()}
    (man / "pack.json").write_text(json.dumps(pack), encoding="utf-8")
    (man / "alignment.json").write_text(json.dumps({"results": state["results"]}), encoding="utf-8")
    captures = {r["file"]: r["sha256"] for r in results}
    (man / "gdc_v1.json").write_text(json.dumps({"captures": captures}), encoding="utf-8")
    return man


def test_gdc_slices_from_tmp_manifests(tmp_path: Path):
    man = _write_manifests(tmp_path)
    conditions = gdc_conditions(man)
    sources = gdc_sources(man)
    assert len(conditions) == len(sources) == 106
    assert sum(c in CLEAN_SLICE for c in conditions.values()) == 53
    assert len(set(sources.values())) == GDC_DOCUMENTS
    results = json.loads((man / "alignment.json").read_text(encoding="utf-8"))["results"]
    first = next(r for r in results if r["sheet"] == 0)
    assert sources[first["file"]] == "doc00"
    last = next(r for r in results if r["sheet"] == 105)
    assert sources[last["file"]] == f"doc{GDC_DOCUMENTS - 1:02d}"


def test_gdc_conditions_refuses_a_changed_capture_hash(tmp_path: Path):
    man = _write_manifests(tmp_path)
    alignment = json.loads((man / "alignment.json").read_text(encoding="utf-8"))
    alignment["results"][5]["sha256"] = "0" * 64
    (man / "alignment.json").write_text(json.dumps(alignment), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        gdc_conditions(man)


def test_gdc_conditions_refuses_other_condition_counts(tmp_path: Path):
    def move(state: dict[str, Any]) -> None:
        r = next(r for r in state["results"] if r["condition"] == "rotated")
        r["condition"] = "scan_app"

    with pytest.raises(ValueError, match="counts"):
        gdc_conditions(_write_manifests(tmp_path, move))


def test_gdc_sources_refuses_an_edited_selection(tmp_path: Path):
    man = _write_manifests(tmp_path)
    sel = man / "selection.yaml"
    sel.write_text(sel.read_text(encoding="utf-8") + "# edited\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selection_sha256"):
        gdc_sources(man)


def test_gdc_sources_refuses_a_condition_mismatch(tmp_path: Path):
    def swap(state: dict[str, Any]) -> None:
        a = next(r for r in state["results"] if r["condition"] == "rotated")
        b = next(r for r in state["results"] if r["condition"] == "curved")
        a["condition"], b["condition"] = b["condition"], a["condition"]

    man = _write_manifests(tmp_path, swap)
    assert len(gdc_conditions(man)) == 106  # counts still hold
    with pytest.raises(ValueError, match="condition differs"):
        gdc_sources(man)


@pytest.mark.skipif(
    not (REPO_ROOT / "data/manifests/gdc/alignment.json").exists(),
    reason="GDC manifests not present",
)
def test_gdc_slices_on_the_tracked_manifests():
    # Manifest JSON/YAML only: no page file is opened.
    man = REPO_ROOT / "data/manifests/gdc"
    conditions = gdc_conditions(man)
    sources = gdc_sources(man)
    assert set(conditions) == set(sources)
    assert sum(c in CLEAN_SLICE for c in conditions.values()) == 53
    assert len(set(sources.values())) == GDC_DOCUMENTS
