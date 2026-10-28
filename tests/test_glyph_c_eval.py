from __future__ import annotations

import json

import glyph_c_eval as gce
import numpy as np
import pytest

LINES = [f"l{i}" for i in range(40)]
CELLS = [("blur", 1), ("blur", 2), ("noise", 1), ("noise", 2)]


def keys() -> list[tuple[str, str, int]]:
    out = []
    for line in LINES:
        out.append((line, "clean", 0))
        out += [(line, c, lv) for c, lv in CELLS]
    return out


def fake_run(root, run_id, err_rate, val_cer, seed):
    """A scored run: every reference is 10 characters; errors drawn around ``err_rate``."""
    rng = np.random.default_rng(seed)
    k = keys()
    refs = np.full(len(k), 10)
    errors = rng.binomial(10, err_rate, len(k))
    d = root / run_id
    d.mkdir(parents=True)
    np.savez_compressed(
        d / "glyph_c.npz",
        errors=errors,
        refs=refs,
        lines=np.array([x[0] for x in k]),
        corruption=np.array([x[1] for x in k]),
        level=np.array([x[2] for x in k]),
    )
    (d / "summary.json").write_text(json.dumps({"best": {"val_cer": val_cer}}), encoding="utf-8")


def test_parse_key_keeps_bars_in_the_line_id():
    assert gce.parse_key("doc|7|gaussian_blur|3") == ("doc|7", "gaussian_blur", 3)
    assert gce.parse_key("x|clean|0") == ("x", "clean", 0)


def test_mce_is_the_unweighted_mean_of_cell_micro_cer():
    k = [("a", "clean", 0), ("a", "blur", 1), ("b", "blur", 1), ("a", "noise", 1)]
    errors = np.array([5, 1, 3, 2])
    refs = np.array([10, 10, 30, 4])
    res = gce.summarise(errors, refs, k)
    assert res["clean_cer"] == pytest.approx(0.5)
    # blur 1: (1 + 3) / 40 = 0.1 (micro over lines), noise 1: 2 / 4 = 0.5; mCE is their plain mean
    assert res["cer_per_corruption_level"] == {"blur": {"1": 0.1}, "noise": {"1": 0.5}}
    assert res["mce"] == pytest.approx(0.3)


def test_per_line_cells_match_summarise(tmp_path):
    fake_run(tmp_path, "R", 0.2, 0.08, 0)
    lines, err, ref = gce._per_line_cells(tmp_path / "R")
    z = np.load(tmp_path / "R" / "glyph_c.npz")
    k = list(zip(z["lines"].tolist(), z["corruption"].tolist(), z["level"].tolist(), strict=True))
    assert lines == sorted(LINES)
    assert err.shape == (len(LINES), len(CELLS))
    assert gce.mce_of(err, ref) == pytest.approx(gce.summarise(z["errors"], z["refs"], k)["mce"])


def test_compare_verdicts_and_h3_guard(tmp_path):
    arms = {
        "A (none)": ["a1", "a2", "a3"],
        "B (worse)": ["b1", "b2", "b3"],
        "D (better)": ["d1", "d2", "d3"],
    }
    for i in range(3):
        fake_run(tmp_path, f"a{i + 1}", 0.30, 0.080, i)
        fake_run(tmp_path, f"b{i + 1}", 0.45, 0.080, 10 + i)
        fake_run(tmp_path, f"d{i + 1}", 0.10, 0.083, 20 + i)
    res = gce.compare(arms, [tmp_path / "missing", tmp_path], tmp_path / "out")
    rows = {r["arm"]: r for r in res["rows"]}
    assert "verdict" not in rows["A (none)"]
    assert rows["B (worse)"]["verdict"] == "hurts"
    assert rows["D (better)"]["verdict"] == "helps"
    assert rows["D (better)"]["diff"]["high"] < 0
    # 0.083 is within 5% of 0.080, so the clean guard holds
    assert res["h3"] == {"d_verdict": "helps", "clean_guard_holds": True, "supported": True}
    assert (tmp_path / "out" / "phase11_comparison.md").read_text(encoding="utf-8").count("|") > 20


def test_compare_h3_fails_on_the_clean_guard(tmp_path):
    arms = {"A": ["a1", "a2", "a3"], "D": ["d1", "d2", "d3"]}
    for i in range(3):
        fake_run(tmp_path, f"a{i + 1}", 0.30, 0.080, i)
        fake_run(tmp_path, f"d{i + 1}", 0.10, 0.090, 20 + i)
    res = gce.compare(arms, [tmp_path], tmp_path / "out")
    assert res["h3"]["d_verdict"] == "helps"
    assert res["h3"]["clean_guard_holds"] is False
    assert res["h3"]["supported"] is False


def test_compare_needs_every_run_scored(tmp_path):
    with pytest.raises(SystemExit, match="not scored"):
        gce.compare({"A": ["nope"]}, [tmp_path], tmp_path / "out")


def test_phase11_arms_are_three_seeds_each():
    arms = gce.ARMS["phase11"]
    assert next(iter(arms)).startswith("A")
    assert all(len(ids) == 3 for ids in arms.values())
    ids = [i for ids in arms.values() for i in ids]
    assert len(set(ids)) == 15
