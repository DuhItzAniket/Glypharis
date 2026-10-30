from __future__ import annotations

import json

import numpy as np
import pytest
import synth_study as ss

# 3 dense pages (>= DOC_MIN_LINES lines) and 2 sparse ones
PAGES = {f"d{i}": ss.DOC_MIN_LINES for i in range(3)} | {"s0": 5, "s1": 8}
LINES = [f"{p}/p0/l{j}" for p, n in PAGES.items() for j in range(n)]
LABEL = "abcdefghij"  # 10 characters


def write_index(root):
    path = root / f"{ss.HIERTEXT_VAL}.index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"key": k}) + "\n" for k in LINES), encoding="utf-8")


def fake_run(roots, run_id, err_rate, seed, glyph_c=False):
    """A run whose predictions miss each character with probability ``err_rate``."""
    rng = np.random.default_rng(seed)
    d = roots / run_id
    d.mkdir(parents=True)
    samples, errs = [], 0
    for k in LINES:
        pred = "".join("#" if rng.random() < err_rate else ch for ch in LABEL)
        errs += sum(a != b for a, b in zip(pred, LABEL, strict=True))
        samples.append(json.dumps({"key": k, "label": LABEL, "pred": pred}))
    (d / "val_best_samples.jsonl").write_text("\n".join(samples) + "\n", encoding="utf-8")
    cer = errs / (10 * len(LINES))
    (d / "summary.json").write_text(json.dumps({"best": {"val_cer": cer}}), encoding="utf-8")
    n = 500
    np.savez_compressed(
        d / "glyphsynth_val.npz",
        errors=rng.binomial(10, err_rate, n),
        refs=np.full(n, 10),
        keys=np.array([f"g{i}" for i in range(n)]),
    )
    if glyph_c:
        k = [
            (f"x{i}", c, lv)
            for i in range(60)
            for c, lv in [("clean", 0), ("blur", 1), ("blur", 2), ("rot", 1), ("rot", 2)]
        ]
        np.savez_compressed(
            d / "glyph_c.npz",
            errors=rng.binomial(10, err_rate, len(k)),
            refs=np.full(len(k), 10),
            lines=np.array([x[0] for x in k]),
            corruption=np.array([x[1] for x in k]),
            level=np.array([x[2] for x in k]),
        )


def test_doc_pages_keeps_only_dense_pages(tmp_path):
    write_index(tmp_path)
    assert ss.doc_pages(tmp_path / f"{ss.HIERTEXT_VAL}.index.jsonl") == {"d0", "d1", "d2"}


def test_hiertext_units_checks_the_logged_cer(tmp_path):
    fake_run(tmp_path, "R", 0.2, 0)
    units = ss.hiertext_units(tmp_path / "R", {"d0"})
    assert set(units) == {"d0"} and units["d0"][1] == 10 * ss.DOC_MIN_LINES
    s = json.loads((tmp_path / "R/summary.json").read_text())
    s["best"]["val_cer"] += 0.01
    (tmp_path / "R/summary.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="logged CER"):
        ss.hiertext_units(tmp_path / "R", None)


def test_verdict_needs_the_ci_and_the_seed_ranges():
    ref = np.full(50, 100.0)
    ctl = np.array([np.full(50, 20.0), np.full(50, 21.0), np.full(50, 22.0)])
    better = ss.verdict(np.array([np.full(50, 10.0)]), ctl, ref, "pages")
    assert better["verdict"] == "helps" and better["diff"]["high"] < 0
    inside = ss.verdict(np.array([np.full(50, 21.5)]), ctl, ref, "pages")
    assert inside["verdict"] == "no clear effect"
    worse = ss.verdict(np.array([np.full(50, 30.0)] * 3), ctl, ref, "pages")
    assert worse["verdict"] == "hurts"


def test_compare_end_to_end(tmp_path):
    data = tmp_path / "data"
    write_index(data)
    runs = tmp_path / "runs"
    arms = {
        ss.CONTROL: ["c1", "c2", "c3"],
        "+25K": ["a1", "a2", "a3"],
        "+100K": ["m1", "m2", "m3"],
        "+250K": ["b1", "b2", "b3"],
        "+100K, 6 fonts": ["f1"],
        "+100K, no capture": ["n1"],
    }
    rates = {"c": 0.30, "a": 0.20, "m": 0.15, "b": 0.13, "f": 0.25, "n": 0.25}
    for ids in arms.values():
        for i, rid in enumerate(ids):
            fake_run(runs, rid, rates[rid[0]], seed=hash(rid) % 1000 + i, glyph_c=rid[0] in "mn")
    res = ss.compare(arms, [tmp_path / "elsewhere", runs], tmp_path / "out", data)
    a = res["arms"]
    assert a["+25K"]["doc_vs_control"]["verdict"] == "helps"
    assert res["h2a"]["supported"] and "+250K" in res["h2a"]["sizes_that_help"]
    assert res["h2a"]["gain_per_1k_lines"]["diminishing"] is True
    assert res["h2b"]["fonts"]["many_fonts_help"] is True
    assert res["h2b"]["capture"]["capture_helps"] is True
    assert res["h2b"]["capture"]["clean_guard_holds"] is True
    assert res["h2b"]["supported"] is True
    md = (tmp_path / "out/phase12_comparison.md").read_text(encoding="utf-8")
    assert "H2'a" in md and "H2'b" in md
