from __future__ import annotations

import json

import ablation_ladder as al
import pytest
from test_synth_study import fake_run, write_index

RATES = {"L0": 0.30, "L1": 0.25, "L2": 0.255, "L3": 0.20, "L4": 0.15, "L5": 0.149}


def setup(tmp_path, monkeypatch, rates):
    data = tmp_path / "data"
    write_index(data)
    runs = tmp_path / "runs"
    ladder = {f"{k} row": [f"{k}a", f"{k}b", f"{k}c"] for k in rates}
    rows = list(ladder)
    for k, rate in rates.items():
        for i, rid in enumerate(ladder[f"{k} row"]):
            fake_run(runs, rid, rate, seed=100 * int(k[1]) + i, glyph_c=True)
    monkeypatch.setattr(al, "LADDER", ladder)
    monkeypatch.setattr(al, "ROWS", rows)
    monkeypatch.setattr(
        al,
        "PARENT",
        {rows[1]: rows[0], rows[2]: rows[1], rows[3]: rows[1], rows[4]: rows[3], rows[5]: rows[4]},
    )
    return runs, data, rows


def test_ladder_keeps_rows_by_the_registered_rules(tmp_path, monkeypatch):
    runs, data, rows = setup(tmp_path, monkeypatch, RATES)
    res = al.ladder([runs], tmp_path / "out", data)
    kept = {r: res["rows"][r].get("kept") for r in rows}
    assert kept[rows[1]] and not kept[rows[2]] and kept[rows[3]] and kept[rows[4]]
    assert not kept[rows[5]]  # L5 changes nothing measurable
    assert res["final_option"] == "A"
    md = (tmp_path / "out/phase13_ladder.md").read_text(encoding="utf-8")
    assert "option **A**" in md
    assert al.main(["decide", "--out", str(tmp_path / "out")]) == 0


def test_l5_kept_on_glyph_c_gives_option_b(tmp_path, monkeypatch):
    runs, data, rows = setup(tmp_path, monkeypatch, {**RATES, "L5": 0.10})
    res = al.ladder([runs], tmp_path / "out", data)
    assert res["rows"][rows[5]]["mce_vs_parent"]["verdict"] == "helps"
    assert res["final_option"] == "B"


def test_no_gain_from_the_full_model_asks_the_owner(tmp_path, monkeypatch):
    runs, data, _ = setup(tmp_path, monkeypatch, {**RATES, "L4": 0.20})
    res = al.ladder([runs], tmp_path / "out", data)
    assert res["final_option"] == "ask"


def test_final_report(tmp_path, monkeypatch):
    data = tmp_path / "data"
    write_index(data)
    runs = tmp_path / "runs"
    for i, rid in enumerate(["f1", "f2", "f3"]):
        fake_run(runs, rid, 0.10, seed=i, glyph_c=True)
    fake_run(runs, "base", 0.20, seed=9)
    monkeypatch.setattr(al, "FINAL", {"A": ["f1", "f2", "f3"]})
    monkeypatch.setattr(al, "BASELINE_FULL", "base")
    res = al.final("A", [runs], tmp_path / "out", data)
    assert res["val_vs_baseline"]["verdict"] == "helps"
    assert len(res["mce_seeds"]) == 3
    saved = json.loads((tmp_path / "out/phase13_final.json").read_text(encoding="utf-8"))
    assert saved["option"] == "A"


def test_missing_glyph_c_scores_stop_the_ladder(tmp_path, monkeypatch):
    runs, data, _ = setup(tmp_path, monkeypatch, RATES)
    (runs / "L5a" / "glyph_c.npz").unlink()
    with pytest.raises(SystemExit, match=r"glyph_c\.npz"):
        al.ladder([runs], tmp_path / "out", data)
