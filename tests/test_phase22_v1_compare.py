"""The Phase 22 metric inventory and comparers, on made-up data in tmp_path only."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import evaluate_baselines as eb
import phase14_score as ps
import phase22_v1_compare as cmp
import phase22_v1_repro as repro
import pytest
from test_phase14_score import (
    MARKER,
    _recorded,
    _rerecord,
    no_phase14_tag,  # noqa: F401
)
from test_phase14_score import layout as score_layout  # noqa: F401
from test_phase22_v1_repro import stamp_rerun_root

from glypharis.pipeline import frozen

REPO = Path(__file__).resolve().parents[1]
VALUE = {"value": 0.1, "low": 0.05, "high": 0.15, "unit": "lines"}
IDENTITY = {
    "model": {"record_sha256": "1", "torch": "2.x"},
    "settings": {"temperature": 2.35},
    "batch": 256,
    "tag_commit": "1234",
    "inputs": {"pack": "p"},
}


@pytest.fixture(autouse=True)
def pinned_tree(monkeypatch):
    """Every comparison runs from this worktree, with the tag check answered here."""
    monkeypatch.chdir(REPO)
    calls = []

    def check(tag="phase14-eval", *, root=Path(".")):
        calls.append((tag, Path(root)))
        return {"tag": tag, "tag_commit": "e" * 40, "head": "d" * 40, "trees": "equal"}

    monkeypatch.setattr(frozen, "pinned_code_check", check)
    return calls


def _entry(experiment="GW-REC-038", task="rec", eval_set="hiertext_val", system="glypharis"):
    return {
        "experiment": experiment,
        "system": system,
        "version": "GLYPHARIS v1",
        "settings": {"temperature": 2.35},
        "task": task,
        "eval_set": eval_set,
        "split": "validation",
        "samples": 10,
        "expected": 10,
        "runs": 1,
        "legacy_meta": False,
        "tracks": {
            "strict": {
                "samples": 10,
                "cer": dict(VALUE),
                "wer": dict(VALUE),
                "ned": 0.2,
                "charset_ceiling": 0.0,
            }
        },
    }


def _results():
    bin_row = {"lines": 5, "confidence": 0.8, "accuracy": 0.7}
    return {
        "experiment": "GW-E2E-009",
        "provenance": {"date": "2026-11-17", "scorer": {"commit": "abc"}},
        "predictions": [_entry(), _entry("GW-BASE-002", system="ppocr_mobile")],
        "subset": [_entry("GW-E2E-007", "e2e", "omnidocbench_test")],
        "gdc_conditions": {
            "conditions": {"curved": 3},
            "rows": [
                {
                    "experiment": "GW-REC-039",
                    "system": "glypharis",
                    "task": "rec",
                    "condition": "curved",
                    "units": 3,
                    "tracks": {"strict": {"cer": dict(VALUE)}},
                },
                {
                    "experiment": "GW-REC-039",
                    "system": "glypharis",
                    "task": "rec",
                    "condition": "flat",
                    "units": 4,
                    "tracks": {"strict": {"cer": dict(VALUE)}},
                },
                {
                    "experiment": "GW-BASE-002",
                    "system": "ppocr_mobile",
                    "task": "rec",
                    "condition": "curved",
                    "units": 3,
                    "tracks": {"strict": {"cer": dict(VALUE)}},
                },
            ],
        },
        "gdc_scan_in_charset": [
            {
                "experiment": "GW-REC-039",
                "system": "glypharis",
                "lines": 3,
                "left_out": 1,
                "tracks": {"strict": {"cer": dict(VALUE)}},
            }
        ],
        "gdc_scan_in_charset_pages": [
            {
                "experiment": "GW-E2E-007",
                "system": "glypharis",
                "task": "e2e",
                "pages": 2,
                "tracks": {"strict": {"cer": dict(VALUE)}},
            }
        ],
        "comparisons": [
            {
                "task": "rec",
                "eval_set": "gdc_test",
                "glypharis": "GW-REC-039",
                "baseline": "GW-BASE-002",
                "system": "ppocr_mobile",
                "subset": None,
                "units": 3,
                "difference": {"strict": {"cer": {**VALUE, "resamples": 1000}}},
            }
        ],
        "official": {
            "hiertext": [
                {"experiment": "GW-E2E-007", "system": "glypharis", "pages": 3, "line": {"PQ": 0.4}}
            ],
            "omnidocbench": [
                {
                    "experiment": "GW-E2E-007",
                    "system": "glypharis",
                    "scope": "all",
                    "pages": 3,
                    "match_debug": {
                        "page_count": 3,
                        "text_match_fallback_counts": {"page_timeout": 1},
                        "text_match_fallback_pages": {"page_timeout": ["odb_1"]},
                    },
                    "text_block": {"edit_whole": 0.2, "ALL_page_avg": float("nan")},
                },
                {
                    "experiment": "GW-E2E-007",
                    "system": "glypharis",
                    "scope": "subset",
                    "pages": 1,
                    "match_debug": {"page_count": 1},
                    "text_block": {"edit_whole": 0.3, "ALL_page_avg": 0.3},
                },
            ],
            "evaluators": {"hiertext": {"commit": "abc"}},
        },
        "calibration": {
            "sets": {
                "gdc_test": {
                    "experiment": "GW-REC-039",
                    "lines": 10,
                    "ece_t": 0.02,
                    "reliability_t": [bin_row],
                }
            }
        },
        "hypotheses": {
            "H1′ CER": {"verdict": "met", "tests": {"strict_pages": dict(VALUE)}},
            "H1′ latency": {"mean_ms": {"GW-EFF-005": 900.0}},
            "H1′ holds": True,
            "H5′": {
                "cer_gdc": 0.1,
                "verdicts": {"lines": "met"},
                "supported_on_gdc": True,
                "information": {"funsd_test": {"cer_ratio": 1.7}},
            },
        },
        "handoff": {"files": ["data/processed/phase14/handoff/a.jsonl"], "seed": 1},
    }


def _inventory(tmp_path, doc=None, *extra):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_results() if doc is None else doc), encoding="utf-8")
    out = tmp_path / "out/metric_inventory.json"
    assert cmp.main(["--results", str(path), "inventory", "--out", str(out), *extra]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------- the inventory


def test_inventory_lists_glypharis_metrics_by_kind_without_values(tmp_path):
    inv = _inventory(tmp_path)
    assert inv["predictions_enumerated"] is True
    point = set(inv["metrics"]["point"])
    assert "predictions[GW-REC-038|rec|hiertext_val].tracks.strict.cer.value" in point
    assert "calibration.sets.gdc_test.reliability_t.0.confidence" in point
    assert "official.hiertext[GW-E2E-007].line.PQ" in point
    assert "comparisons[GW-REC-039|GW-BASE-002|ppocr_mobile|rec|gdc_test|None]." in "".join(
        p[: p.index("]") + 2] for p in point if p.startswith("comparisons")
    )
    assert "predictions[GW-REC-038|rec|hiertext_val].samples" in inv["metrics"]["count"]
    assert "gdc_conditions.conditions.conditions.curved" in inv["metrics"]["count"]
    assert "hypotheses.H1′ CER.verdict" in inv["metrics"]["verdict"]
    assert "hypotheses.H1′ holds" in inv["metrics"]["verdict"]
    assert "hypotheses.H5′.verdicts.lines" in inv["metrics"]["verdict"]
    assert "hypotheses.H5′.supported_on_gdc" in inv["metrics"]["verdict"]
    assert "hypotheses.H1′ latency.mean_ms.GW-EFF-005" in point
    everything = json.dumps(inv["metrics"], ensure_ascii=False)
    assert "GW-BASE-002|rec" not in everything  # no baseline block
    assert ".low" not in everything and "match_debug" not in everything
    assert "settings" not in everything and "provenance" not in everything
    assert "0.05" not in everything and "900" not in everything


def test_inventory_reports_values_that_are_not_plain_fractions(tmp_path):
    doc = _results()
    doc["hypotheses"]["H5′"]["cer_gdc"] = 62.5  # a percentage would look like this
    scale = _inventory(tmp_path, doc)["scale"]
    assert scale["patterns_with_a_value_above_1"] == [
        "hypotheses.H1′ latency.mean_ms.GW-EFF-005",
        "hypotheses.H5′.cer_gdc",
    ]
    assert scale["ratios_not_fractions"] == ["hypotheses.H5′.information.funsd_test.cer_ratio"]
    assert scale["patterns_with_nan"] == ["official.omnidocbench[].text_block.ALL_page_avg"]


def test_a_large_negative_value_is_reported_too(tmp_path):
    doc = _results()
    doc["hypotheses"]["H5′"]["cer_gdc"] = -2.0
    scale = _inventory(tmp_path, doc)["scale"]
    assert "hypotheses.H5′.cer_gdc" in scale["patterns_with_a_value_above_1"]


def test_inventory_covers_every_glypharis_block_and_keeps_rows_apart(tmp_path):
    point = set(_inventory(tmp_path)["metrics"]["point"])
    for path in (
        "subset[GW-E2E-007|e2e|omnidocbench_test].tracks.strict.cer.value",
        "gdc_conditions.rows[GW-REC-039|rec|curved].tracks.strict.cer.value",
        "gdc_conditions.rows[GW-REC-039|rec|flat].tracks.strict.cer.value",
        "gdc_scan_in_charset[GW-REC-039].tracks.strict.cer.value",
        "gdc_scan_in_charset_pages[GW-E2E-007|e2e].tracks.strict.cer.value",
        "official.omnidocbench[GW-E2E-007|all].text_block.edit_whole",
        "official.omnidocbench[GW-E2E-007|subset].text_block.edit_whole",
    ):
        assert path in point, path


def test_two_rows_with_one_label_stop_the_inventory(tmp_path):
    doc = _results()
    doc["official"]["hiertext"].append(copy.deepcopy(doc["official"]["hiertext"][0]))
    path = tmp_path / "results.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="two rows produce the metric"):
        cmp.main(["--results", str(path), "inventory", "--out", str(tmp_path / "x.json")])


def test_skipping_predictions_never_touches_them_and_is_marked(tmp_path):
    doc = _results()
    doc["predictions"] = "not even a list"
    inv = _inventory(tmp_path, doc, "--skip-predictions")
    assert inv["predictions_enumerated"] is False
    assert not any(p.startswith("predictions") for ps_ in inv["metrics"].values() for p in ps_)
    assert inv["metrics"]["point"]


def test_inventory_needs_a_glypharis_entry(tmp_path):
    doc = _results()
    doc["predictions"] = []
    path = tmp_path / "results.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="no GLYPHARIS entry"):
        cmp.main(["--results", str(path), "inventory", "--out", str(tmp_path / "x.json")])


# ------------------------------------------------------------------------------ the comparison


def _pair(change):
    old = _entry()
    new = copy.deepcopy(old)
    change(new)
    return cmp.compare_metrics(cmp.metrics_of(old), cmp.metrics_of(new))


def test_equal_entries_pass():
    res = _pair(lambda e: None)
    assert res["passed"] and res["failures"] == []
    assert res["checked"]["point"] == 4 and res["checked"]["count"] == 3


def test_a_change_of_0_0004_passes_and_0_0006_fails():
    def moved(by):
        def change(e):
            e["tracks"]["strict"]["cer"]["value"] += by

        return change

    assert _pair(moved(0.0004))["passed"]
    assert _pair(moved(-0.0004))["passed"]
    res = _pair(moved(0.0006))
    assert not res["passed"]
    assert res["failures"][0]["path"] == "tracks.strict.cer.value"
    assert res["failures"][0]["difference"] == pytest.approx(0.0006)


@pytest.mark.parametrize(
    ("by", "passes"),
    [
        (0.00049, True),
        (0.0005, True),
        (0.00051, False),
        (-0.00049, True),
        (-0.0005, True),
        (-0.00051, False),
    ],
)
def test_the_tolerance_boundary_in_both_directions(by, passes):
    def change(e):
        e["tracks"]["strict"]["cer"]["value"] += by

    assert _pair(change)["passed"] is passes


def test_two_none_values_are_equal():
    old = cmp.metrics_of({"e2e_f1": None})
    assert cmp.compare_metrics(old, cmp.metrics_of({"e2e_f1": None}))["passed"]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ({"verdict": "met"}, {"verdict": "not shown"}),
        ({"H1′ holds": True}, {"H1′ holds": False}),
        ({"supported_on_gdc": True}, {"supported_on_gdc": False}),
        ({"verdicts": {"lines": "met"}}, {"verdicts": {"lines": "not met"}}),
        ({"some_flag": True}, {"some_flag": False}),
    ],
)
def test_a_changed_verdict_or_flag_fails(old, new):
    res = cmp.compare_metrics(cmp.metrics_of(old), cmp.metrics_of(new))
    assert not res["passed"] and res["failures"][0]["problem"] == "not equal"


def test_a_bound_on_one_side_only_is_not_judged():
    old = cmp.metrics_of({"cer": {"value": 0.1, "low": 0.0}})
    assert cmp.compare_metrics(old, cmp.metrics_of({"cer": {"value": 0.1}}))["passed"]


def test_the_largest_differences_are_listed_first():
    old = cmp.metrics_of({"a": 0.1, "b": 0.2, "c": 0.3})
    new = cmp.metrics_of({"a": 0.1001, "b": 0.2003, "c": 0.3})
    res = cmp.compare_metrics(old, new)
    assert [d["path"] for d in res["largest_differences"]] == ["b", "a"]
    assert res["largest_point_difference_within_tolerance"] == pytest.approx(0.0003)


def test_a_missing_metric_fails_and_so_does_an_extra_one():
    res = _pair(lambda e: e["tracks"]["strict"].pop("wer"))
    assert not res["passed"]
    assert "tracks.strict.wer.value" in {f["path"] for f in res["failures"]}

    def extra(e):
        e["tracks"]["strict"]["new_metric"] = 0.1

    assert not _pair(extra)["passed"]


def test_none_against_a_number_fails_and_nan_matches_nan():
    def blank(e):
        e["tracks"]["strict"]["cer"] = None

    assert not _pair(blank)["passed"]
    old = {"a": {"edit_whole": math.nan}}
    assert cmp.compare_metrics(cmp.metrics_of(old), cmp.metrics_of(copy.deepcopy(old)))["passed"]
    assert not cmp.compare_metrics(cmp.metrics_of(old), cmp.metrics_of({"a": {"edit_whole": 0.1}}))[
        "passed"
    ]


def test_a_count_must_be_equal():
    def change(e):
        e["samples"] = 11

    res = _pair(change)
    assert not res["passed"] and res["failures"][0]["problem"] == "not equal"


def test_provenance_and_bounds_are_not_judged():
    def change(e):
        e["settings"] = {"temperature": 9.0}
        e["version"] = "other"
        e["runs"] = 7
        e["legacy_meta"] = True
        e["provenance"] = {"commit": 7}
        e["tracks"]["strict"]["cer"]["low"] = 0.0
        e["tracks"]["strict"]["cer"]["unit"] = "pages"

    res = _pair(change)
    assert res["passed"]
    assert res["largest_bound_difference_not_judged"] == pytest.approx(0.05)


def test_record_differences_ignore_time_and_hold_no_text(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows = [
        {"key": "1", "pred": MARKER, "conf_t": 0.5, "time_ms": 1.0, "seg": 0},
        {"key": "2", "pred": "same", "conf_t": 0.5, "time_ms": 1.0, "seg": 0},
        {"key": "3", "pred": "same", "conf_t": 0.5, "time_ms": 1.0, "seg": 0},
    ]
    other = copy.deepcopy(rows)
    other[0]["pred"] = "changed"
    other[1]["conf_t"] = 0.6
    other[2].update(time_ms=9.0, seg=3)
    for path, doc in ((a, rows), (b, other)):
        path.write_text("".join(json.dumps(r) + "\n" for r in doc), encoding="utf-8")
    counts = cmp.differing_records(a, b)
    assert counts == {"records": 3, "only_in_one_file": 0, "differing": 2, "text_differing": 1}
    assert MARKER not in json.dumps(counts)


def test_records_on_one_side_only_are_counted(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"key": "1", "pred": "x"}\n{"key": "2", "pred": "x"}\n', encoding="utf-8")
    b.write_text('{"key": "1", "pred": "x"}\n{"key": "3", "pred": "x"}\n', encoding="utf-8")
    counts = cmp.differing_records(a, b)
    assert counts["records"] == 1 and counts["only_in_one_file"] == 2


def test_page_records_compare_their_line_texts(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    page = {"key": "p", "lines": [{"text": "x", "polygon": [[0, 0]]}], "time_ms": 1.0}
    moved = {"key": "p", "lines": [{"text": "x", "polygon": [[1, 1]]}], "time_ms": 2.0}
    a.write_text(json.dumps(page) + "\n", encoding="utf-8")
    b.write_text(json.dumps(moved) + "\n", encoding="utf-8")
    assert cmp.differing_records(a, b)["differing"] == 1
    assert cmp.differing_records(a, b)["text_differing"] == 0


# ------------------------------------------------------------------------------- the rehearsal


def _set_identities(data):
    for job in ps.glypharis_jobs(data):
        path = job.path.with_suffix(".meta.json")
        meta = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**meta, "identity": IDENTITY}), encoding="utf-8")
    _rerecord(data)


def _change_identity(root, folder="GW-REC-038", eval_set="hiertext_val"):
    path = root / "processed/phase14" / folder / f"{eval_set}.meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["identity"] = {**IDENTITY, "batch": 64, "model": {**IDENTITY["model"], "torch": "other"}}
    path.write_text(json.dumps(meta), encoding="utf-8")
    _rerecord(root)


@pytest.fixture
def rehearsal(score_layout, tmp_path):  # noqa: F811
    layout = score_layout
    """Phase 14 files, an identical re-run root, and a results.json scored from them."""
    data = layout["data"]
    _set_identities(data)
    rerun = tmp_path / "rerun"
    shutil.copytree(data, rerun)
    stamp_rerun_root(rerun)
    ctx = ps.FullContext(data, layout["log"], 14, "made-up reference")
    entries = []
    for experiment in cmp.REHEARSAL_RUNS:
        job = cmp._job(data, experiment)
        entries.append(eb.score_file(job.path, ctx, meta=ps.glypharis_meta(job)))
    doc = {**_results(), "predictions": entries}
    results = tmp_path / "reports/results.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(doc), encoding="utf-8")
    inventory = tmp_path / "reports/metric_inventory.json"
    assert cmp.main(["--results", str(results), "inventory", "--out", str(inventory)]) == 0
    return {
        "data": data,
        "rerun": rerun,
        "results": results,
        "inventory": inventory,
        "log": layout["log"],
        "out": tmp_path / "reports/rehearsal.json",
        "doc": doc,
    }


def _rehearse(r):
    return cmp.main(
        [
            "--results",
            str(r["results"]),
            "rehearsal",
            "--rerun-root",
            str(r["rerun"]),
            "--phase14-data",
            str(r["data"]),
            "--inventory",
            str(r["inventory"]),
            "--access-log",
            str(r["log"]),
            "--out",
            str(r["out"]),
        ]
    )


def _reinventory(r):
    args = ["--results", str(r["results"]), "inventory", "--out", str(r["inventory"])]
    assert cmp.main(args) == 0


def test_rehearsal_scores_the_two_validation_entries_and_logs_nothing(rehearsal, monkeypatch):
    before = rehearsal["log"].read_bytes()
    scored = []
    real = eb.score_file

    def spy(path, ctx, **kw):
        scored.append((path.parent.name, path.name, ctx.phase))
        return real(path, ctx, **kw)

    monkeypatch.setattr(eb, "score_file", spy)
    assert _rehearse(rehearsal) == 0
    assert scored == [
        ("GW-REC-038", "hiertext_val.jsonl", 22),
        ("GW-E2E-006", "hiertext_val.jsonl", 22),
    ]
    assert rehearsal["log"].read_bytes() == before
    doc = json.loads(rehearsal["out"].read_text(encoding="utf-8"))
    assert doc["passed"] is True and set(doc["runs"]) == {"GW-REC-042", "GW-E2E-014"}
    assert doc["runs"]["GW-REC-042"]["records"]["differing"] == 0
    assert str(rehearsal["rerun"]) not in rehearsal["out"].read_text(encoding="utf-8")


def test_rehearsal_records_the_pinned_commit(rehearsal, pinned_tree):
    assert _rehearse(rehearsal) == 0
    doc = json.loads(rehearsal["out"].read_text(encoding="utf-8"))
    assert doc["wrapper_tag_commit"] == "e" * 40 and doc["head"] == "d" * 40
    assert pinned_tree[0][0] == "phase22-repro"


def test_rehearsal_refuses_an_unpinned_tree(rehearsal, monkeypatch):
    def refuse(tag="phase14-eval", *, root=Path(".")):
        raise RuntimeError("uncommitted changes")

    monkeypatch.setattr(frozen, "pinned_code_check", refuse)
    with pytest.raises(SystemExit, match="not pinned"):
        _rehearse(rehearsal)
    assert not rehearsal["out"].exists()


def test_rehearsal_refuses_to_compare_phase_14_with_itself(rehearsal):
    rehearsal["rerun"] = rehearsal["data"]
    with pytest.raises(SystemExit, match="is the Phase 14 data root"):
        _rehearse(rehearsal)
    assert not rehearsal["out"].exists()


def test_rehearsal_refuses_files_the_rerun_did_not_write(rehearsal):
    path = rehearsal["rerun"] / "processed/phase14/GW-REC-038/hiertext_val.meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    del meta["phase22"]
    path.write_text(json.dumps(meta), encoding="utf-8")
    _rerecord(rehearsal["rerun"])
    with pytest.raises(SystemExit, match="was not written by"):
        _rehearse(rehearsal)


def test_rehearsal_fails_when_the_run_identity_differs(rehearsal):
    _change_identity(rehearsal["rerun"])
    assert _rehearse(rehearsal) == 1
    doc = json.loads(rehearsal["out"].read_text(encoding="utf-8"))
    assert doc["passed"] is False
    assert doc["runs"]["GW-REC-042"]["identity_differences"] == ["batch", "model.torch"]
    assert doc["runs"]["GW-E2E-014"]["passed"] is True


def test_rehearsal_scores_the_rerun_file_not_phase_14s(rehearsal):
    path = rehearsal["rerun"] / "processed/phase14/GW-REC-038/hiertext_val.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["pred"] = "wrong"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    _rerecord(rehearsal["rerun"])
    assert _rehearse(rehearsal) == 1
    run = json.loads(rehearsal["out"].read_text(encoding="utf-8"))["runs"]["GW-REC-042"]
    assert run["failures"] and run["records"]["text_differing"] == len(rows)


def test_a_missing_rerun_file_is_refused(rehearsal):
    (rehearsal["rerun"] / "processed/phase14/GW-E2E-006/hiertext_val.jsonl").unlink()
    with pytest.raises(SystemExit, match="is missing"):
        _rehearse(rehearsal)


def test_repeated_reference_entries_are_refused(rehearsal):
    doc = copy.deepcopy(rehearsal["doc"])
    doc["predictions"].append(copy.deepcopy(doc["predictions"][0]))
    raw = json.dumps(doc).encode("utf-8")
    rehearsal["results"].write_bytes(raw)
    inventory = json.loads(rehearsal["inventory"].read_text(encoding="utf-8"))
    inventory["source_sha256"] = hashlib.sha256(raw).hexdigest()
    rehearsal["inventory"].write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(SystemExit, match="holds 2 entries"):
        _rehearse(rehearsal)


def test_a_rehearsal_that_touches_the_log_is_void(rehearsal, monkeypatch):
    real = eb.score_file

    def spy(path, ctx, **kw):
        with rehearsal["log"].open("a", encoding="utf-8") as fh:
            fh.write("| a row |\n")
        return real(path, ctx, **kw)

    monkeypatch.setattr(eb, "score_file", spy)
    with pytest.raises(SystemExit, match="result is void"):
        _rehearse(rehearsal)
    assert not rehearsal["out"].exists()


def test_rehearsal_fails_on_a_difference_above_the_tolerance(rehearsal):
    doc = copy.deepcopy(rehearsal["doc"])
    doc["predictions"][0]["tracks"]["strict"]["cer"]["value"] += 0.0006
    rehearsal["results"].write_text(json.dumps(doc), encoding="utf-8")
    _reinventory(rehearsal)
    assert _rehearse(rehearsal) == 1
    report = json.loads(rehearsal["out"].read_text(encoding="utf-8"))
    assert report["passed"] is False
    (failure,) = report["runs"]["GW-REC-042"]["failures"]
    assert failure["path"] == "tracks.strict.cer.value"


def test_rehearsal_passes_a_difference_within_the_tolerance(rehearsal):
    doc = copy.deepcopy(rehearsal["doc"])
    doc["predictions"][0]["tracks"]["strict"]["cer"]["value"] += 0.0004
    rehearsal["results"].write_text(json.dumps(doc), encoding="utf-8")
    _reinventory(rehearsal)
    assert _rehearse(rehearsal) == 0


def test_a_phase14_file_that_differs_from_its_hash_stops_the_comparer(rehearsal, monkeypatch):
    path = rehearsal["data"] / "processed/phase14/GW-REC-038/hiertext_val.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    calls = []
    monkeypatch.setattr(eb, "score_file", lambda *a, **k: calls.append(a))
    with pytest.raises(SystemExit, match="differs from its hash"):
        _rehearse(rehearsal)
    assert calls == [] and not rehearsal["out"].exists()


def test_a_phase14_metadata_change_stops_the_comparer(rehearsal):
    path = rehearsal["data"] / "processed/phase14/GW-E2E-006/hiertext_val.meta.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from its hash"):
        _rehearse(rehearsal)


def test_an_unrecorded_rerun_file_is_refused(rehearsal):
    manifest = rehearsal["rerun"] / ps.OUTPUTS
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["outputs"] = [e for e in doc["outputs"] if "GW-E2E-006" not in e["file"]]
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="not in the outputs manifest"):
        _rehearse(rehearsal)


def test_rehearsal_counts_differing_records(rehearsal):
    path = rehearsal["rerun"] / "processed/phase14/GW-REC-038/hiertext_val.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["conf_t"] += 1e-6
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = rehearsal["rerun"] / ps.OUTPUTS
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    entry = _recorded(rehearsal["rerun"], path, "GW-REC-038")
    doc["outputs"] = [entry if e["file"] == entry["file"] else e for e in doc["outputs"]]
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    assert _rehearse(rehearsal) == 0
    report = json.loads(rehearsal["out"].read_text(encoding="utf-8"))
    assert report["runs"]["GW-REC-042"]["records"]["differing"] == 1
    assert report["runs"]["GW-REC-042"]["records"]["text_differing"] == 0


@pytest.mark.parametrize("problem", ["missing", "incomplete", "stale"])
def test_rehearsal_needs_a_complete_current_inventory(rehearsal, problem):
    if problem == "missing":
        rehearsal["inventory"].unlink()
        match = "inventory is missing"
    elif problem == "incomplete":
        args = ["--results", str(rehearsal["results"]), "inventory"]
        assert cmp.main([*args, "--out", str(rehearsal["inventory"]), "--skip-predictions"]) == 0
        match = "skipped the predictions"
    else:
        doc = json.loads(rehearsal["inventory"].read_text(encoding="utf-8"))
        doc["source_sha256"] = hashlib.sha256(b"other").hexdigest()
        rehearsal["inventory"].write_text(json.dumps(doc), encoding="utf-8")
        match = "another results file"
    with pytest.raises(SystemExit, match=match):
        _rehearse(rehearsal)


def test_no_gdc_scan_marker_reaches_the_inventory_or_the_rehearsal(rehearsal, capsys):
    assert (
        MARKER.encode()
        in (rehearsal["data"] / "processed/phase14/GW-REC-039/gdc_scan_test.jsonl").read_bytes()
    )
    assert _rehearse(rehearsal) == 0
    captured = capsys.readouterr()
    assert MARKER not in captured.out + captured.err
    assert MARKER.encode() not in rehearsal["out"].read_bytes()
    assert MARKER.encode() not in rehearsal["inventory"].read_bytes()


# --------------------------------------------------------------------------- item 7: the hash list


def _store(root):
    for rel in (
        "processed/phase14/GW-REC-039/gdc_scan_test.jsonl",
        "processed/phase14/logs/GW-REC-039_gdc_scan_test.log",
        "processed/baselines/GW-BASE-002/ppocr_mobile_rec_gdc_scan_test.meta.json",
        "processed/baselines_phase14/GW-BASE-002/a.jsonl",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel, encoding="utf-8")


def test_the_hash_list_is_complete_and_repeatable(tmp_path):
    data = tmp_path / "data"
    _store(data)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    assert cmp.main(["v1-hashes", "--data", str(data), "--out", str(first)]) == 0
    assert cmp.main(["v1-hashes", "--data", str(data), "--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    doc = json.loads(first.read_text(encoding="utf-8"))
    rel = "processed/baselines_phase14/GW-BASE-002/a.jsonl"
    assert len(doc["files"]) == 4
    assert doc["files"][rel] == {
        "sha256": hashlib.sha256(rel.encode()).hexdigest(),
        "bytes": len(rel),
    }
    assert str(tmp_path) not in first.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "rel",
    [
        "processed/phase14/handoff/scan_page_1.png",
        "processed/phase14/scan_doc/gdc_scan_test.jsonl",
    ],
)
def test_a_name_that_may_identify_scan_content_stops_the_list(tmp_path, rel):
    data = tmp_path / "data"
    _store(data)
    (data / rel).parent.mkdir(parents=True, exist_ok=True)
    (data / rel).write_text("x", encoding="utf-8")
    out = tmp_path / "list.json"
    with pytest.raises(SystemExit, match="may name GDC-scan content"):
        cmp.main(["v1-hashes", "--data", str(data), "--out", str(out)])
    assert not out.exists()


def test_a_missing_store_stops_the_list(tmp_path):
    with pytest.raises(SystemExit, match="is missing"):
        cmp.main(["v1-hashes", "--data", str(tmp_path), "--out", str(tmp_path / "x.json")])


# ------------------------------------------------------------------------- item 10: the exact check


def _pass_a(tmp_path, change=None):
    old = _results()
    new = copy.deepcopy(old)
    new["experiment"] = repro.PASS_A
    new["provenance"] = {"date": "2026-12-01", "scorer": {"commit": "other"}, "files": 9}
    new["handoff"]["files"] = ["elsewhere/handoff/a.jsonl"]
    if change is not None:
        change(new)
    reference, report = tmp_path / "phase14.json", tmp_path / "pass_a.json"
    reference.write_text(json.dumps(old), encoding="utf-8")
    report.write_text(json.dumps(new), encoding="utf-8")
    out = tmp_path / "verdict.json"
    argv = ["--results", str(reference), "pass-a", "--pass-a", str(report), "--out", str(out)]
    code = cmp.main(argv)
    return code, json.loads(out.read_text(encoding="utf-8"))


def _odb(doc, row=0):
    return doc["official"]["omnidocbench"][row]


def test_pass_a_ignores_only_provenance_and_path_strings(tmp_path, pinned_tree):
    code, doc = _pass_a(tmp_path)
    assert code == 0 and doc["identical"] is True and doc["differing"] == []
    assert doc["cause"] is None and doc["wrapper_tag_commit"] == "e" * 40
    assert pinned_tree[0][0] == "phase22-repro"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["calibration"]["sets"]["gdc_test"].update(ece_t=0.02 + 5e-13),
        lambda d: d["predictions"][0]["tracks"]["strict"]["cer"].update(low=0.051),
        lambda d: d["predictions"][0]["settings"].update(temperature=2.0),
        lambda d: d["predictions"][0].update(runs=2),
        lambda d: d["handoff"].update(seed=2),
        lambda d: d["hypotheses"]["H1′ CER"].update(verdict="not shown"),
        lambda d: d["hypotheses"].update({"H1′ holds": 1}),
        lambda d: d["hypotheses"]["H5′"].update(cer_gdc=None),
        lambda d: _odb(d)["match_debug"].update(page_count=2),
        lambda d: d["official"]["hiertext"].append({}),
        lambda d: d["handoff"].pop("seed"),
    ],
)
def test_pass_a_fails_on_any_other_difference(tmp_path, change):
    code, doc = _pass_a(tmp_path, change)
    assert code == 1 and doc["identical"] is False and doc["differing"]
    assert doc["unexplained_count"] == doc["differing_count"] and doc["cause"] is None


def test_exact_equality_of_nan_none_and_flags():
    old = {"nan": math.nan, "none": None, "flag": True, "number": 1}
    assert list(cmp.exact_differences(old, {**old, "number": 1.0})) == []
    assert cmp._same_exactly(math.nan, math.nan) and cmp._same_exactly(None, None)
    assert not cmp._same_exactly(None, 0) and not cmp._same_exactly(True, 1)
    assert not cmp._same_exactly(math.nan, 0.0)


def test_a_changed_fallback_list_alone_is_disclosed_but_not_a_difference(tmp_path):
    def change(d):
        _odb(d)["match_debug"]["text_match_fallback_pages"]["page_timeout"] = ["odb_2"]

    code, doc = _pass_a(tmp_path, change)
    assert code == 0 and doc["identical"] is True
    (row,) = doc["omnidocbench_fallback_differences"]
    assert row["pages"]["page_timeout"] == {
        "only_in_phase14": ["odb_1"],
        "only_in_pass_a": ["odb_2"],
    }


def test_an_official_number_with_changed_fallbacks_fails_with_that_cause(tmp_path):
    def change(d):
        _odb(d)["match_debug"]["text_match_fallback_pages"]["page_timeout"] = ["odb_2"]
        _odb(d)["text_block"]["edit_whole"] = 0.25

    code, doc = _pass_a(tmp_path, change)
    assert code == 1 and doc["identical"] is False
    assert "timeout-fallback pages" in doc["cause"] and doc["unexplained_count"] == 0


def test_fallbacks_of_one_row_do_not_explain_another(tmp_path):
    def change(d):
        _odb(d)["match_debug"]["text_match_fallback_pages"]["page_timeout"] = ["odb_2"]
        _odb(d, 1)["text_block"]["edit_whole"] = 0.35
        d["calibration"]["sets"]["gdc_test"]["ece_t"] = 0.9

    code, doc = _pass_a(tmp_path, change)
    assert code == 1 and doc["cause"] is None and doc["unexplained_count"] == 2


def test_an_official_number_without_changed_fallbacks_is_unexplained(tmp_path):
    def change(d):
        _odb(d)["text_block"]["edit_whole"] = 0.25

    code, doc = _pass_a(tmp_path, change)
    assert code == 1 and doc["cause"] is None and doc["unexplained_count"] == 1


def test_pass_a_needs_its_own_report(tmp_path):
    with pytest.raises(SystemExit, match="is not GW-E2E-015's report"):
        _pass_a(tmp_path, lambda d: d.update(experiment="GW-E2E-018"))


# -------------------------------------------------------------------- items 12 and 13 after Pass B


def _finish_pass_b(r):
    ps.write_pass(r["rerun"], {"experiment": repro.PASS_B, "status": "done"})


def _line_counts(r, *extra):
    r["lines"] = r["out"].with_name("line_comparison.json")
    return cmp.main(
        [
            "line-counts",
            "--rerun-root",
            str(r["rerun"]),
            "--phase14-data",
            str(r["data"]),
            "--access-log",
            str(r["log"]),
            "--out",
            str(r["lines"]),
            *extra,
        ]
    )


def _log_rows(r):
    return [ln for ln in r["log"].read_text(encoding="utf-8").splitlines() if ln.startswith("| 20")]


def test_line_counts_log_one_row_per_file_and_write_counts_only(rehearsal, capsys):
    _finish_pass_b(rehearsal)
    assert _line_counts(rehearsal) == 0
    rows = _log_rows(rehearsal)
    assert len(rows) == 12
    assert all(f"| {repro.PASS_B} | glypharis (" in r for r in rows)
    assert all(r.endswith(f"line comparison | 22 | {cmp.LINE_REASON} |") for r in rows)
    doc = json.loads(rehearsal["lines"].read_text(encoding="utf-8"))
    assert len(doc["runs"]) == 12 and doc["runs"]["GW-REC-043 gdc_test"]["differing"] == 0
    assert doc["runs"]["GW-E2E-017 gdc_scan_test"]["only_in_one_file"] == 0
    captured = capsys.readouterr()
    assert MARKER not in captured.out + captured.err
    assert MARKER.encode() not in rehearsal["lines"].read_bytes()
    assert MARKER not in rehearsal["log"].read_text(encoding="utf-8")


def test_line_counts_need_a_finished_pass_b(rehearsal):
    with pytest.raises(SystemExit, match="has not finished"):
        _line_counts(rehearsal)
    assert _log_rows(rehearsal) == []


def test_a_changed_phase_14_file_stops_the_line_counts_before_any_row(rehearsal):
    _finish_pass_b(rehearsal)
    path = rehearsal["data"] / "processed/phase14/GW-REC-039/gdc_test.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="differs from its hash"):
        _line_counts(rehearsal)
    assert _log_rows(rehearsal) == []


def test_line_counts_run_once_and_resume_only_when_stopped(rehearsal, monkeypatch):
    _finish_pass_b(rehearsal)
    real = cmp.differing_records
    calls = []

    def stop_at_the_third(old, new):
        calls.append(old)
        if len(calls) == 3:
            raise RuntimeError("stopped")
        return real(old, new)

    monkeypatch.setattr(cmp, "differing_records", stop_at_the_third)
    with pytest.raises(RuntimeError, match="stopped"):
        _line_counts(rehearsal)
    assert len(_log_rows(rehearsal)) == 3
    monkeypatch.setattr(cmp, "differing_records", real)
    with pytest.raises(SystemExit, match="--resume"):
        _line_counts(rehearsal)
    assert _line_counts(rehearsal, "--resume") == 0
    rows = _log_rows(rehearsal)
    assert len(rows) == 15 and all(r.endswith("(resumed) |") for r in rows[3:])
    with pytest.raises(SystemExit, match="written once"):
        _line_counts(rehearsal, "--resume")


def test_a_resume_needs_a_started_comparison(rehearsal):
    _finish_pass_b(rehearsal)
    with pytest.raises(SystemExit, match="logged no row"):
        _line_counts(rehearsal, "--resume")


def _verdict(r, change=None):
    doc = copy.deepcopy(r["doc"])
    doc.update(experiment=repro.PASS_B, id_map=repro.RERUN_IDS, id_map_note=repro.ID_MAP_HEADER)
    if change is not None:
        change(doc)
    report = r["out"].with_name("pass_b.json")
    report.write_text(json.dumps(doc), encoding="utf-8")
    r["verdict"] = r["out"].with_name("v1_reproduction.json")
    return cmp.main(
        [
            "--results",
            str(r["results"]),
            "verdict",
            "--rerun-root",
            str(r["rerun"]),
            "--phase14-data",
            str(r["data"]),
            "--pass-b",
            str(report),
            "--inventory",
            str(r["inventory"]),
            "--out",
            str(r["verdict"]),
        ]
    )


def _verdict_doc(r):
    return json.loads(r["verdict"].read_text(encoding="utf-8"))


def test_the_verdict_passes_and_writes_both_reports(rehearsal):
    _finish_pass_b(rehearsal)
    before = rehearsal["log"].read_bytes()
    assert _verdict(rehearsal) == 0
    doc = _verdict_doc(rehearsal)
    assert doc["passed"] is True and doc["tolerance"]["checked"]["point"] > 0
    report = rehearsal["verdict"].with_suffix(".md").read_text(encoding="utf-8")
    assert "**Verdict: passed.**" in report
    assert rehearsal["log"].read_bytes() == before


def test_the_verdict_fails_above_the_tolerance_and_shows_the_largest(rehearsal):
    _finish_pass_b(rehearsal)

    def change(doc):
        doc["hypotheses"]["H5′"]["cer_gdc"] += 0.0006

    assert _verdict(rehearsal, change) == 1
    doc = _verdict_doc(rehearsal)
    assert doc["passed"] is False and doc["failures"] == {"reliability_bins": 0, "other": 1}
    assert doc["tolerance"]["largest_differences"][0]["path"] == "hypotheses.H5′.cer_gdc"
    assert "**Verdict: FAILED.**" in rehearsal["verdict"].with_suffix(".md").read_text("utf-8")


def test_the_verdict_counts_bin_failures_apart_and_lists_changed_verdicts(rehearsal):
    _finish_pass_b(rehearsal)

    def change(doc):
        doc["calibration"]["sets"]["gdc_test"]["reliability_t"][0]["accuracy"] += 0.01
        doc["hypotheses"]["H1′ CER"]["verdict"] = "not shown"

    assert _verdict(rehearsal, change) == 1
    doc = _verdict_doc(rehearsal)
    assert doc["failures"] == {"reliability_bins": 1, "other": 1}
    assert [f["path"] for f in doc["changed_verdicts"]] == ["hypotheses.H1′ CER.verdict"]


def test_the_verdict_fails_when_a_run_identity_differs(rehearsal):
    _finish_pass_b(rehearsal)
    _change_identity(rehearsal["rerun"], "GW-E2E-008", "gdc_test")
    assert _verdict(rehearsal) == 1
    assert _verdict_doc(rehearsal)["identity_differences"] == {
        "GW-E2E-017 gdc_test": ["batch", "model.torch"]
    }


def test_the_verdict_refuses_an_unmarked_report_a_stale_inventory_or_an_open_pass(rehearsal):
    with pytest.raises(SystemExit, match="has not finished"):
        _verdict(rehearsal)
    _finish_pass_b(rehearsal)
    with pytest.raises(SystemExit, match="marked report"):
        _verdict(rehearsal, lambda doc: doc.pop("id_map"))
    inventory = json.loads(rehearsal["inventory"].read_text(encoding="utf-8"))
    inventory["metrics"]["point"].pop()
    rehearsal["inventory"].write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not list"):
        _verdict(rehearsal)
