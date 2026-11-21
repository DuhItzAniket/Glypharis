"""The phone candidate's scorer (scripts/phase22_score.py), on made-up data in tmp_path only.

The data root is `test_phase14_score`'s synthetic layout: v1's Phase 14 files, their outputs
manifest, the sets, the official evaluators' fakes. The candidate's prediction files are written
here in the record and metadata schema of Phase 14's runner, as `scripts/phase22_eval.py` writes
them. No real prediction file, test set, bundle or access log is read or written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import phase14_score as ps
import phase22_score as p22
import pytest
from test_phase14_score import (
    MARKER,
    _inputs,
    no_phase14_tag,  # noqa: F401
)
from test_phase14_score import layout as score_layout  # noqa: F401

from glypharis.data.pages import read_pages, write_pages
from glypharis.eval.page_metrics import PageResult
from glypharis.eval.sets import EVAL_SETS

TAG_COMMIT = "c" * 40
LABEL = "phone candidate (new version, not v1)"
V1_SOURCES = {
    "GW-REC-045": "GW-REC-039",
    "GW-E2E-020": "GW-E2E-007",
    "GW-REC-044": "GW-REC-038",
    "GW-E2E-019": "GW-E2E-006",
}


def says_v1(text: str) -> bool:
    """Whether ``text`` names v1 other than to say it is not v1."""
    import re

    return re.search(r"(?<![A-Za-z0-9])v1(?![0-9])", text.replace("not v1", "")) is not None


def _tree_digest(folder: Path) -> dict[str, str]:
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _records(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def _cand_path(data: Path, exp: str, eval_set: str) -> Path:
    return data / "processed/phase22" / exp / f"{eval_set}.jsonl"


def _meta_path(data: Path, exp: str, eval_set: str) -> Path:
    return _cand_path(data, exp, eval_set).with_suffix(".meta.json")


def write_candidate(data, exp, task, eval_set, bundle_sha, rows=None, **meta_fields):
    """A candidate prediction file and its metadata, as the runner leaves a finished run."""
    if rows is None:
        src = data / "processed/phase14" / V1_SOURCES[exp] / f"{eval_set}.jsonl"
        rows = [{**r, "seg": 0} for r in _records(src)]
    path = _cand_path(data, exp, eval_set)
    _write(path, rows)
    spec = EVAL_SETS[eval_set]
    meta = {
        "system": "glypharis_phone_candidate",
        "task": task,
        "mode": task,
        "eval_set": eval_set,
        "benchmark": spec["benchmark"],
        "split": spec["split"],
        "experiment": exp,
        "phase": 22,
        "version": LABEL,
        "expected": len(rows),
        "samples": len(rows),
        "temperature": 2.35,
        "settings": {"threads": 10, "batching": "one line per recognizer run"},
        "version_label": LABEL,
        "bundle_manifest_sha256": bundle_sha,
        "tag_commit": TAG_COMMIT,
        "smoke": None,
        "pinned": {
            "tag": p22.TAG,
            "tag_commit": TAG_COMMIT,
            "head": TAG_COMMIT,
            "trees": "equal",
        },
        "inputs": _inputs(data, eval_set, task),
        "runs": [{"segment": 0, "status": "done", "reason": f"Phase 22 ({exp})"}],
        **meta_fields,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return path


@pytest.fixture
def cand(score_layout, tmp_path, monkeypatch):  # noqa: F811
    lay = score_layout
    data = lay["data"]
    # GDC-scan's file names carry the marker too; its texts already do.
    scan = [
        replace(p, image=f"raw/gdc_scan/{MARKER}_{i}.jpg")
        for i, p in enumerate(lay["pages"]["gdc_scan_test"])
    ]
    write_pages(data / EVAL_SETS["gdc_scan_test"]["pages"], scan)

    reports = tmp_path / "reports/final_validation/candidate"
    record = reports / "GW-EFF-014"
    bundle = tmp_path / "runs/release/glypharis-phone-candidate"
    manifest = {
        "format": "glypharis-phone-bundle-1",
        "version": "glypharis_phone_candidate",
        "version_label": LABEL,
        "arm": "fused-single-line",
        "batching": "one line per recognizer run",
        "recognizer": {"temperature": 2.35},
    }
    for folder in (bundle, record):
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bundle_sha = hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest()
    (record / "equivalence.json").write_text(
        json.dumps({"pass": True, "bundle_manifest_sha256": bundle_sha}), encoding="utf-8"
    )
    v1_report = tmp_path / "reports/external/results.json"
    v1_report.parent.mkdir(parents=True, exist_ok=True)
    v1_report.write_text(
        json.dumps(
            {
                "experiment": "GW-E2E-009",
                "predictions": [
                    {
                        "experiment": "GW-REC-039",
                        "system": "glypharis",
                        "task": "rec",
                        "eval_set": "funsd_test",
                        "version": p22.V1_LABEL,
                    },
                    {
                        "experiment": "GW-REC-039",
                        "system": "glypharis",
                        "task": "rec",
                        "eval_set": "gdc_test",
                        "version": p22.V1_LABEL,
                    },
                    {
                        "experiment": "GW-BASE-002",
                        "system": "ppocr_mobile",
                        "task": "rec",
                        "eval_set": "gdc_test",
                        "version": "fake",
                    },
                ],
                "official": {
                    "hiertext": [
                        {
                            "experiment": "GW-E2E-007",
                            "system": "glypharis",
                            "line": {"E2E-Fscore": 0.4},
                        }
                    ],
                    "omnidocbench": [],
                },
            }
        ),
        encoding="utf-8",
    )
    for exp, (task, sets) in {**p22.TEST_RUNS}.items():
        for eval_set in sets:
            write_candidate(data, exp, task, eval_set, bundle_sha)
    for exp, (task, _) in p22.GATE_RUNS.items():
        write_candidate(data, exp, task, p22.GATE_SET, bundle_sha)

    monkeypatch.setattr(p22, "REPORTS", reports)
    monkeypatch.setattr(p22, "BUNDLE", bundle)
    monkeypatch.setattr(p22, "BUNDLE_RECORD", record)
    monkeypatch.setattr(p22, "ACCESS_LOG", lay["log"])
    monkeypatch.setattr(p22, "V1_REPORT", v1_report)
    tree = {
        "pinned": {"tag": p22.TAG, "tag_commit": TAG_COMMIT, "head": TAG_COMMIT, "trees": "equal"},
        "v1_code": {"tag": "v1-code", "v1_code": "equal"},
    }
    monkeypatch.setattr(p22, "tree_state", lambda: json.loads(json.dumps(tree)))
    commits: dict[str, bool] = {"committed": True}
    monkeypatch.setattr(p22, "committed", lambda path: commits["committed"])
    return {
        **lay,
        "reports": reports,
        "record": record,
        "bundle": bundle,
        "bundle_sha": bundle_sha,
        "commits": commits,
        "tree": tree,
    }


def _main(c, command, *extra):
    return p22.main([command, "--data-root", str(c["data"]), *extra])


def _rows(log: Path) -> list[str]:
    return [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.startswith("| 20")]


def _ready(c):
    """The gate met and the hashes written: what the scoring pass needs before it runs."""
    assert _main(c, "gate") == 0
    assert _main(c, "hashes") == 0


def _gate_doc(c):
    return json.loads((c["reports"] / "validation_gate.json").read_text(encoding="utf-8"))


def _write_gate(c, **changes):
    doc = {**_gate_doc(c), **changes}
    (c["reports"] / "validation_gate.json").write_text(json.dumps(doc), encoding="utf-8")


def _pass_record(c):
    return c["data"] / p22.PASS_RECORD


# ------------------------------------------------------------------------------- the gate


def test_gate_writes_three_met_verdicts_with_the_bundle_and_tag(cand):
    assert _main(cand, "gate") == 0
    doc = _gate_doc(cand)
    assert doc["verdicts"] == {"line_cer": "met", "page_cer": "met", "e2e_f1": "met"}
    assert doc["all_met"] is True
    assert doc["bundle_manifest_sha256"] == cand["bundle_sha"]
    assert (doc["tag"], doc["tag_commit"]) == ("phase22-candidate-eval", TAG_COMMIT)
    assert doc["margin"] == 0.0005
    assert doc["experiments"] == {"GW-REC-044": "GW-REC-038", "GW-E2E-019": "GW-E2E-006"}
    # Unit intervals only on validation, and a validation split logs no access row.
    assert all("document_interval" not in t for t in doc["tests"].values())
    assert _rows(cand["log"]) == []
    assert {e["version"] for e in doc["candidate"]} == {LABEL}
    assert {e["experiment"] for e in doc["candidate"]} == {"GW-REC-044", "GW-E2E-019"}


def test_an_unmet_gate_is_written_and_returns_failure(cand):
    data = cand["data"]
    path = _cand_path(data, "GW-REC-044", "hiertext_val")
    rows = [{**r, "pred": ""} for r in _records(path)]
    write_candidate(data, "GW-REC-044", "rec", "hiertext_val", cand["bundle_sha"], rows)
    assert _main(cand, "gate") == 1
    doc = _gate_doc(cand)
    assert doc["verdicts"]["line_cer"] == "not met"
    assert doc["all_met"] is False and doc["verdict"] == "not met"


def test_the_gate_runs_once(cand):
    assert _main(cand, "gate") == 0
    with pytest.raises(SystemExit, match="the gate is run once"):
        _main(cand, "gate")


def test_the_gate_refuses_a_v1_file_that_differs_from_the_outputs_manifest(cand):
    path = cand["data"] / "processed/phase14/GW-REC-038/hiertext_val.jsonl"
    path.write_bytes(path.read_bytes() + b"\n\n")
    with pytest.raises(SystemExit, match="differs from its hash"):
        _main(cand, "gate")
    assert not (cand["reports"] / "validation_gate.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("bundle_manifest_sha256", "0" * 64, "another bundle"),
        ("pinned", {"tag": "phase22-candidate-eval", "tag_commit": "d" * 40, "trees": "equal"},
         "not run at"),
        ("runs", [{"status": "running"}], "did not finish"),
        ("smoke", {"limit": 2}, "a smoke run"),
    ],
)  # fmt: skip
def test_a_candidate_file_of_another_bundle_tag_or_unfinished_run_is_refused(
    cand, field, value, match
):
    data = cand["data"]
    path = _meta_path(data, "GW-REC-044", "hiertext_val")
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta[field] = value
    path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit, match=match):
        _main(cand, "gate")


def test_the_bundle_must_be_the_frozen_and_checked_one(cand):
    eq = cand["record"] / "equivalence.json"
    eq.write_text(json.dumps({"pass": False, "bundle_manifest_sha256": cand["bundle_sha"]}))
    with pytest.raises(SystemExit, match="does not pass this bundle"):
        _main(cand, "gate")
    eq.unlink()
    with pytest.raises(SystemExit, match="the bundle's check comes first"):
        _main(cand, "gate")
    (cand["bundle"] / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from the committed copy"):
        _main(cand, "gate")


def test_a_bundle_label_that_does_not_say_not_v1_is_refused(cand):
    doc = {"version_label": p22.V1_LABEL, "arm": "fused", "recognizer": {"temperature": 2.35}}
    for folder in (cand["bundle"], cand["record"]):
        (folder / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    sha = hashlib.sha256((cand["bundle"] / "manifest.json").read_bytes()).hexdigest()
    (cand["record"] / "equivalence.json").write_text(
        json.dumps({"pass": True, "bundle_manifest_sha256": sha}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="does not say it is not v1"):
        _main(cand, "gate")


# ------------------------------------------------------------------------------ the hashes


def test_hashes_record_the_ten_files_and_their_metadata(cand):
    assert _main(cand, "hashes") == 0
    doc = json.loads((cand["reports"] / "prediction_hashes.json").read_text(encoding="utf-8"))
    assert len(doc["files"]) == 10
    assert doc["bundle_manifest_sha256"] == cand["bundle_sha"]
    data = cand["data"]
    for e in doc["files"]:
        path = data / e["file"].removeprefix("data/")
        assert e["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert e["lines"] == len(_records(path))
        meta = data / e["meta"].removeprefix("data/")
        assert e["meta_sha256"] == hashlib.sha256(meta.read_bytes()).hexdigest()
    # Repeating it only confirms; a changed file never rewrites the record.
    assert _main(cand, "hashes") == 0
    path = _cand_path(data, "GW-REC-045", "funsd_test")
    path.write_text(path.read_text(encoding="utf-8").replace('"seg": 0', '"seg": 1'), "utf-8")
    with pytest.raises(SystemExit, match="exists and differs"):
        _main(cand, "hashes")


def test_hashes_refuse_an_incomplete_file(cand):
    path = _cand_path(cand["data"], "GW-E2E-020", "gdc_test")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(SystemExit, match="not every prediction file is complete"):
        _main(cand, "hashes")
    assert not (cand["reports"] / "prediction_hashes.json").exists()


# ------------------------------------------------------------------------------ the pass


def test_scoring_pass_end_to_end(cand, capsys):
    _ready(cand)
    before = len(_rows(cand["log"]))
    assert before == 0
    assert _main(cand, "score") == 0
    res = json.loads((cand["reports"] / "results.json").read_text(encoding="utf-8"))
    rows = _rows(cand["log"])
    assert len(rows) == 14
    assert all(r.endswith("| 22 | Phase 22 candidate scoring (GW-E2E-021) |") for r in rows)
    ids = [r.split("|")[3].strip() for r in rows]
    assert sorted(set(ids)) == ["GW-E2E-007", "GW-E2E-020", "GW-REC-039", "GW-REC-045"]
    assert ids.count("GW-REC-045") == 5 and ids.count("GW-E2E-020") == 5
    assert ids.count("GW-REC-039") == 2 and ids.count("GW-E2E-007") == 2
    assert json.loads(_pass_record(cand).read_text(encoding="utf-8"))["status"] == "done"
    # Same readings on both sides: every test sits exactly at the margin's side of 0.
    h = res["h11"]
    assert h["verdict"] == "met" and h["holds"] is True and len(h["verdicts"]) == 6
    gdc = h["tests"]["gdc_test"]
    for t in gdc.values():
        assert set(t["verdicts"]) == {"units", "documents"}
        assert t["documents"] == 3
    assert all("verdicts" not in t for t in h["tests"]["hiertext_test"].values())
    # The official evaluators saw the candidate's files only.
    assert cand["calls"]["hiertext"] == ["GW-E2E-020_glypharis"]
    assert {n for n, _ in cand["calls"]["omnidocbench"]} == {
        "GW-E2E-020_glypharis_all",
        "GW-E2E-020_glypharis_subset100",
    }
    assert [r["experiment"] for r in res["candidate"]["official"]["hiertext"]] == ["GW-E2E-020"]
    # v1: the 4 re-scored files under its own label, the rest copied from its report.
    assert {e["version"] for e in res["v1"]["rescored"]} == {p22.V1_LABEL}
    assert len(res["v1"]["rescored"]) == 4
    assert [(e["experiment"], e["eval_set"]) for e in res["v1"]["reported"]["predictions"]] == [
        ("GW-REC-039", "funsd_test")
    ]
    assert set(res["candidate"]["calibration"]["sets"]) == set(p22.TEST_RUNS["GW-REC-045"][1])
    assert res["candidate"]["calibration"]["temperature"] == 2.35
    assert res["provenance"]["converted"] == {"hiertext": 1, "omnidocbench": 1}
    for name in ("results.json", "results.md", "h11.md"):
        assert (cand["reports"] / name).is_file()
    capsys.readouterr().out.encode("cp1252")


def test_no_candidate_entry_or_report_says_v1(cand):
    _ready(cand)
    assert _main(cand, "score") == 0
    res = json.loads((cand["reports"] / "results.json").read_text(encoding="utf-8"))
    entries = [*res["candidate"]["predictions"], *res["candidate"]["subset"]]
    assert len(res["candidate"]["predictions"]) == 10
    for e in entries:
        assert e["version"] == LABEL
        assert not says_v1(json.dumps(e, ensure_ascii=False)), e["experiment"]
    assert not says_v1(json.dumps(res["candidate"], ensure_ascii=False))
    assert not says_v1((cand["reports"] / "results.md").read_text(encoding="utf-8"))
    gate = _gate_doc(cand)
    for e in gate["candidate"]:
        assert not says_v1(json.dumps(e, ensure_ascii=False))
    # The check itself sees v1's label.
    assert says_v1(p22.V1_LABEL) and says_v1("glypharis_v1") and not says_v1(LABEL)


def test_nothing_of_phase14_or_the_baselines_changes(cand):
    data = cand["data"]
    watched = [data / "processed/phase14", data / "processed/baselines"]
    watched += [p for p in (data / "processed").glob("baselines*") if p.is_dir()]
    before = {str(f): _tree_digest(f) for f in watched}
    manifest = (data / ps.OUTPUTS).read_bytes()
    _ready(cand)
    assert _main(cand, "score") == 0
    assert {str(f): _tree_digest(f) for f in watched} == before
    assert (data / ps.OUTPUTS).read_bytes() == manifest
    assert not (data / "processed/phase14/GW-E2E-009").exists()
    assert (data / "processed/phase22/GW-E2E-021/pass.json").is_file()


def test_the_gdc_scan_marker_appears_nowhere(cand, capsys):
    data = cand["data"]
    # The marker is in the inputs, so the test would see it if it leaked.
    assert MARKER.encode() in _cand_path(data, "GW-REC-045", "gdc_scan_test").read_bytes()
    assert MARKER.encode() in _cand_path(data, "GW-E2E-020", "gdc_scan_test").read_bytes()
    scan = read_pages(data / EVAL_SETS["gdc_scan_test"]["pages"])
    assert all(MARKER in p.image for p in scan)
    _ready(cand)
    assert _main(cand, "score") == 0
    out = capsys.readouterr()
    checked = [p for p in cand["reports"].rglob("*") if p.is_file()]
    checked += [cand["log"], _pass_record(cand)]
    assert any(p.name == "results.json" for p in checked)
    for p in checked:
        body = p.read_bytes().lower()
        assert MARKER.lower().encode() not in body, p.name
    assert MARKER.lower() not in (out.out + out.err).lower()
    # No absolute path reaches a report either.
    tmp = str(cand["tmp"])
    for p in cand["reports"].glob("*"):
        if p.suffix in (".json", ".md"):
            text = p.read_text(encoding="utf-8")
            assert tmp not in text and Path(tmp).as_posix() not in text, p.name


def test_a_second_scoring_pass_is_refused(cand):
    _ready(cand)
    assert _main(cand, "score") == 0
    rows = len(_rows(cand["log"]))
    with pytest.raises(SystemExit, match="one-shot"):
        _main(cand, "score")
    # Moving the report aside does not open a second pass: the pass record says done.
    (cand["reports"] / "results.json").rename(cand["tmp"] / "aside.json")
    with pytest.raises(SystemExit, match="records a finished pass"):
        _main(cand, "score")
    with pytest.raises(SystemExit, match="records a finished pass"):
        _main(cand, "score", "--resume")
    assert len(_rows(cand["log"])) == rows


def test_a_stopped_pass_needs_resume_and_logs_its_rows_as_resumed(cand, monkeypatch):
    _ready(cand)

    def stop(*args, **kwargs):
        raise RuntimeError("stopped")

    with monkeypatch.context() as mp:
        mp.setattr(p22, "h11", stop)
        with pytest.raises(RuntimeError, match="stopped"):
            _main(cand, "score")
    assert json.loads(_pass_record(cand).read_text(encoding="utf-8"))["status"] == "started"
    first = len(_rows(cand["log"]))
    assert first == 14
    with pytest.raises(SystemExit, match="score --resume"):
        _main(cand, "score")
    assert len(_rows(cand["log"])) == first
    assert _main(cand, "score", "--resume") == 0
    rows = _rows(cand["log"])[first:]
    assert len(rows) == 14
    assert all(r.endswith("| Phase 22 candidate scoring (GW-E2E-021) (resumed) |") for r in rows)
    record = json.loads(_pass_record(cand).read_text(encoding="utf-8"))
    assert record["status"] == "done" and len(record["resumed_utc"]) == 1


def test_resume_with_no_pass_is_refused(cand):
    _ready(cand)
    with pytest.raises(SystemExit, match="records no pass to resume"):
        _main(cand, "score", "--resume")
    assert _rows(cand["log"]) == []


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"verdicts": {"line_cer": "met", "page_cer": "not shown", "e2e_f1": "met"},
          "all_met": False}, "not all three gate tests are met"),
        ({"bundle_manifest_sha256": "0" * 64}, "another bundle"),
        ({"tag_commit": "d" * 40}, "another tag commit"),
    ],
)  # fmt: skip
def test_scoring_refuses_an_unmet_or_mismatched_gate(cand, changes, match):
    _ready(cand)
    _write_gate(cand, **changes)
    with pytest.raises(SystemExit, match=match):
        _main(cand, "score")
    assert _rows(cand["log"]) == [] and not _pass_record(cand).exists()


def test_scoring_refuses_a_missing_gate(cand):
    assert _main(cand, "hashes") == 0
    with pytest.raises(SystemExit, match="the validation gate comes first"):
        _main(cand, "score")
    assert _rows(cand["log"]) == []


def test_scoring_refuses_hashes_missing_uncommitted_or_different(cand):
    assert _main(cand, "gate") == 0
    with pytest.raises(SystemExit, match="run `hashes` and commit it first"):
        _main(cand, "score")
    assert _main(cand, "hashes") == 0
    cand["commits"]["committed"] = False
    with pytest.raises(SystemExit, match="not committed"):
        _main(cand, "score")
    cand["commits"]["committed"] = True
    path = _cand_path(cand["data"], "GW-REC-045", "iiit5k_test")
    path.write_text(path.read_text(encoding="utf-8").replace('"seg": 0', '"seg": 2'), "utf-8")
    with pytest.raises(SystemExit, match="differ from"):
        _main(cand, "score")
    assert _rows(cand["log"]) == [] and not _pass_record(cand).exists()


def test_scoring_stops_if_v1_files_do_not_match_the_outputs_manifest(cand):
    _ready(cand)
    path = cand["data"] / "processed/phase14/GW-E2E-007/gdc_test.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(SystemExit, match="differs from its hash"):
        _main(cand, "score")
    meta = cand["data"] / "processed/phase14/GW-REC-039/hiertext_test.meta.json"
    path.write_bytes(path.read_bytes()[:-1])
    meta.write_bytes(meta.read_bytes() + b" ")
    with pytest.raises(SystemExit, match="run metadata"):
        _main(cand, "score")
    assert _rows(cand["log"]) == [] and not _pass_record(cand).exists()


def test_scoring_refuses_a_candidate_run_at_another_tag(cand):
    _ready(cand)
    cand["tree"]["pinned"]["tag_commit"] = "d" * 40
    with pytest.raises(SystemExit, match="not run at"):
        _main(cand, "score")
    assert _rows(cand["log"]) == []


def test_the_branch_and_pins_are_checked_first(monkeypatch):
    calls = []

    def branch(name, start):
        calls.append(("branch", name))
        raise RuntimeError(f"this run must start on branch {name}; HEAD is on main")

    monkeypatch.setattr(p22.evsets, "require_branch", branch)
    with pytest.raises(SystemExit, match="phase/22-final-validation"):
        p22.tree_state()
    monkeypatch.setattr(p22.evsets, "require_branch", lambda name, start: None)

    def unpinned(tag, *, root):
        calls.append(("pin", tag))
        raise RuntimeError(f"the tag {tag} does not exist; the code is not pinned")

    monkeypatch.setattr(p22.frozen, "pinned_code_check", unpinned)
    with pytest.raises(SystemExit, match="not pinned at phase22-candidate-eval"):
        p22.tree_state()
    assert calls == [("branch", "phase/22-final-validation"), ("pin", "phase22-candidate-eval")]


# ------------------------------------------------------------------- H11's formulas, units


def _rec(err, den) -> ps.RecUnits:
    err, den = np.asarray(err, dtype=np.float64), np.asarray(den, dtype=np.float64)
    zeros = np.zeros_like(err)
    keys = [f"k{i}" for i in range(err.size)]
    return ps.RecUnits(
        keys, {"strict": {"err": err, "den": den, "werr": zeros, "wden": zeros, "exact": zeros}}
    )


def _pages(bag_err, bag_ref, tp, pred, gt) -> ps.PageUnits:
    res = [
        PageResult(f"p{i}", bag_errors=int(b), bag_ref_chars=int(r), e2e_tp=int(t),
                   e2e_pred=int(p), e2e_gt=int(g))
        for i, (b, r, t, p, g) in enumerate(zip(bag_err, bag_ref, tp, pred, gt, strict=True))
    ]  # fmt: skip
    return ps.PageUnits([r.doc_id for r in res], {"strict": res, "normalised": res})


RNG = np.random.default_rng(7)
V_ERR = RNG.integers(0, 4, 200)
NOISE = np.where(np.arange(200) % 2 == 0, 2, -2)


@pytest.mark.parametrize(
    ("cand_err", "want"),
    [(np.maximum(V_ERR - 1, 0), "met"), (V_ERR + 4, "not met"), (V_ERR + NOISE + 2, "not shown")],
)
def test_line_cer_verdicts(cand_err, want):
    den = np.full(200, 10)
    v = _rec(V_ERR + 2, den)
    t = p22.line_cer_test(_rec(cand_err, den), v, 0.0005)
    assert t["verdict"] == want
    assert t["side"] == "below" and t["units"] == 200


def test_line_cer_leaves_out_empty_references_and_refuses_other_references():
    v = _rec([1, 0, 2], [10, 0, 10])
    t = p22.line_cer_test(_rec([1, 5, 2], [10, 0, 10]), v, 0.0005)
    assert t["units"] == 2
    assert t["interval"]["value"] == pytest.approx(-0.0005)
    with pytest.raises(SystemExit, match="disagree on the references"):
        p22.line_cer_test(_rec([1, 0, 2], [10, 1, 10]), v, 0.0005)


@pytest.mark.parametrize(
    ("cand_err", "want"),
    [(V_ERR, "met"), (V_ERR + 7, "not met"), (V_ERR + 2 + NOISE, "not shown")],
)
def test_page_cer_verdicts(cand_err, want):
    ref, zeros = np.full(200, 20), np.zeros(200)
    v = _pages(V_ERR + 2, ref, zeros, zeros, zeros)
    c = _pages(cand_err, ref, zeros, zeros, zeros)
    assert p22.page_cer_test(c, v, 0.0005)["verdict"] == want


@pytest.mark.parametrize(
    ("cand_tp", "want"),
    [(np.full(200, 8), "met"), (np.full(200, 6), "not met"), (8 + NOISE, "not shown")],
)
def test_e2e_f1_verdicts(cand_tp, want):
    pred = gt = np.full(200, 10)
    v = _pages(np.zeros(200), np.ones(200), np.full(200, 8), pred, gt)
    c = _pages(np.zeros(200), np.ones(200), cand_tp, pred, gt)
    t = p22.e2e_f1_test(c, v, 0.0005)
    assert t["verdict"] == want and t["side"] == "above"
    if want == "met":  # the same readings: exactly the margin above 0
        assert t["interval"]["low"] == pytest.approx(0.0005)


def test_gdc_document_check_is_combined_with_the_unit_verdict():
    # Better by 2 on one document's 50 crops, worse by 1 on the other's: the crops agree, the two
    # documents do not.
    den = np.full(100, 10)
    v_err = np.full(100, 3)
    c_err = np.concatenate([np.full(50, 1), np.full(50, 4)])
    groups = ["doc-a"] * 50 + ["doc-b"] * 50
    t = p22.line_cer_test(_rec(c_err, den), _rec(v_err, den), 0.0005, groups)
    assert t["verdicts"] == {"units": "met", "documents": "not shown"}
    assert t["verdict"] == "not shown" and t["documents"] == 2
    # One document per crop: the document bootstrap is the unit bootstrap.
    own = [f"d{i}" for i in range(100)]
    t = p22.line_cer_test(_rec(c_err, den), _rec(v_err, den), 0.0005, own)
    assert t["verdicts"] == {"units": "met", "documents": "met"} and t["verdict"] == "met"
    assert t["document_interval"]["unit"] == "documents"


def test_gdc_document_check_on_pages_and_e2e_f1():
    n = 60
    groups = [f"doc-{i % 3}" for i in range(n)]
    ref = np.full(n, 20)
    tp, d = np.full(n, 8), np.full(n, 10)
    v = _pages(np.full(n, 4), ref, tp, d, d)
    c = _pages(np.full(n, 4), ref, tp, d, d)
    page = p22.page_cer_test(c, v, 0.0005, groups)
    f1 = p22.e2e_f1_test(c, v, 0.0005, groups)
    assert page["verdict"] == f1["verdict"] == "met"
    assert page["documents"] == f1["documents"] == 3
    assert f1["document_interval"]["value"] == pytest.approx(0.0005)
    worse = _pages(np.full(n, 4), ref, np.full(n, 5), d, d)
    f1 = p22.e2e_f1_test(worse, v, 0.0005, groups)
    assert f1["verdicts"] == {"units": "not met", "documents": "not met"}


def test_h11_rule_and_margin():
    assert p22.overall(["met"] * 6) == "met"
    assert p22.overall(["met"] * 5 + ["not shown"]) == "not shown"
    assert p22.overall(["met", "not shown", "not met", "met", "met", "met"]) == "not met"
    assert p22.margin_for("fused-int8") == 0.005
    assert p22.margin_for("fused-single-line") == p22.margin_for("fused") == 0.0005
