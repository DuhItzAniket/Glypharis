"""The phone candidate's runner (scripts/phase22_eval.py), on made-up data in tmp_path only.

The data root, the tiny untrained model and the access-log layout are `test_ext_eval`'s; the bundle,
its committed manifest, the equivalence check, the operator check and the gate are made-up files in
a stand-in repository. No real bundle, prediction file, test set or access log is read or written.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import types
from pathlib import Path

import ext_eval
import phase22_eval as pe
import pytest
from test_ext_eval import MARKER, METRIC_KEYS, _keys, _records, _rows, tiny_frozen
from test_ext_eval import root as ext_root  # noqa: F401

from glypharis.data.pages import read_pages, write_pages
from glypharis.efficiency.candidate import BATCHING, SingleLineRecognizer, single_line
from glypharis.efficiency.phone_bundle import PHONE_BUNDLE_FORMAT, VERSION, VERSION_LABEL
from glypharis.eval.sets import EVAL_SETS

TAG_COMMIT = "d" * 40
PINNED = {"tag": pe.PIN_TAG, "tag_commit": TAG_COMMIT, "head": "e" * 40, "trees": "equal"}
V1_CODE = {"tag": "v1-code", "tag_commit": "f" * 40, "head": "e" * 40, "v1_code": "equal"}
REASON = "Phase 22 candidate (test)"
ARM = "fused-single-line"
THREADS = {"requested": 10, "intra_op": 10, "inter_op": 10, "env": {}}
# The fields Phase 14's runner writes into every metadata file (`ext_eval.run`).
PHASE14_FIELDS = {
    "system", "task", "mode", "eval_set", "benchmark", "split", "experiment", "phase", "version",
    "expected", "samples", "temperature", "confidences_at", "settings", "provenance", "pinned",
    "inputs", "identity", "runs", "note",
}  # fmt: skip
ADDED_FIELDS = {"version_label", "tag_commit", "bundle_manifest_sha256", "bundle", "v1_code",
                "clearance", "smoke"}  # fmt: skip


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")


def _digest(folder: Path) -> dict[str, str]:
    return {p.relative_to(folder).as_posix(): _sha(p) for p in sorted(folder.rglob("*"))
            if p.is_file()}  # fmt: skip


def stand_in_ocr():
    """The tiny model as the candidate: one line per network run, labelled not v1."""
    ocr = tiny_frozen().ocr
    ocr.recognizer = single_line(ocr.recognizer)
    ocr.model_version = f"{VERSION} {ARM} ({VERSION_LABEL})"
    return ocr


@pytest.fixture
def cand(ext_root, tmp_path, monkeypatch):  # noqa: F811
    """`test_ext_eval`'s data root, a stand-in repository holding a made-up frozen bundle, its
    committed manifest, a passed equivalence check and a ``go`` operator check; every git check
    passes. The gate is written by `gate` when a test needs one."""
    data, log, _ = ext_root
    repo = tmp_path / "repo"
    bundle = repo / pe.BUNDLE
    bundle.mkdir(parents=True)
    files = {}
    for name in ("detector.onnx", "recognizer.fused.onnx", "charset.json"):
        (bundle / name).write_bytes(f"made-up {name}".encode())
        files[name] = _sha(bundle / name)
    manifest = {
        "format": PHONE_BUNDLE_FORMAT, "version": VERSION, "version_label": VERSION_LABEL,
        "arm": ARM, "batching": BATCHING[ARM], "files": files,
        "recognizer_file": "recognizer.fused.onnx", "detector": {"spec": {"made": "up"}},
        "recognizer": {"height": 32, "max_width": 512, "temperature": 2.35},
        "pipeline": {"reading_order": "made-up"},
        "v1": {"version": "glypharis_v1", "note": "the detector is v1's"},
    }  # fmt: skip
    _write(bundle / "manifest.json", manifest)
    (repo / pe.BUNDLE_REPORT).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle / "manifest.json", repo / pe.BUNDLE_REPORT)
    sha = _sha(bundle / "manifest.json")
    _write(repo / pe.EQUIVALENCE, {"pass": True, "bundle_manifest_sha256": sha})
    _write(repo / pe.OPERATOR_CHECK, {"experiment": "GW-EFF-015", "verdict": "go"})
    calls = {"load": 0}

    def load(path, threads):
        assert Path(path) == repo / pe.BUNDLE and threads == pe.THREADS
        calls["load"] += 1
        return stand_in_ocr()

    monkeypatch.setattr(pe, "repo_root", lambda start=None: repo)
    monkeypatch.setattr(pe, "require_branch", lambda name, start=None: None)
    monkeypatch.setattr(pe, "pinned_code_check", lambda tag=None, root=None: dict(PINNED))
    monkeypatch.setattr(pe, "assert_v1_code", lambda root=None: dict(V1_CODE))
    monkeypatch.setattr(pe, "ACCESS_LOG", log)
    monkeypatch.setattr(pe, "REQUIRE_RECORDED", False)  # the tiny sets have no manifests
    monkeypatch.setattr(pe, "_threads", lambda: dict(THREADS))
    monkeypatch.setattr(pe, "check_threads", lambda n: dict(THREADS))
    monkeypatch.setattr(pe, "check_sessions", lambda ocr, n: {"detector": None, "recognizer": None})
    monkeypatch.setattr(pe, "load_phone_bundle", load)

    def no_manifest(*args, **kwargs):
        pytest.fail("the runner wrote Phase 14's outputs manifest")

    monkeypatch.setattr(ext_eval, "record_output", no_manifest)

    def gate(**change):
        doc = {"experiments": ["GW-REC-044", "GW-E2E-019"],
               "tests": {t: {"verdict": "met"} for t in pe.GATE_TESTS}, "met": True,
               "bundle_manifest_sha256": sha, "tag_commit": TAG_COMMIT, **change}  # fmt: skip
        _write(repo / pe.GATE, doc)

    return types.SimpleNamespace(data=data, log=log, repo=repo, calls=calls, sha=sha, gate=gate,
                                 manifest=manifest)  # fmt: skip


def _run(c, mode, eval_set, experiment, *extra, reason=REASON):
    return pe.main(["run", "--mode", mode, "--set", eval_set, "--experiment", experiment,
                    "--reason", reason, "--data-root", str(c.data), *extra])  # fmt: skip


def _out(c, experiment, eval_set):
    return c.data / "processed/phase22" / experiment / f"{eval_set}.jsonl"


def _meta(c, experiment, eval_set):
    path = _out(c, experiment, eval_set).with_suffix(".meta.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _crash_on(monkeypatch, ocr, call, what="recognizer"):
    """Make the stand-in raise on the ``call``-th recognizer read or page prediction."""
    n = {"calls": 0}
    target = ocr.recognizer if what == "recognizer" else ocr
    name = "read" if what == "recognizer" else "predict"
    original = getattr(target, name)

    def crashing(*args, **kwargs):
        n["calls"] += 1
        if n["calls"] == call:
            raise RuntimeError("simulated crash")
        return original(*args, **kwargs)

    monkeypatch.setattr(target, name, crashing)


def _load_this(monkeypatch, c, ocr):
    def load(path, threads):
        c.calls["load"] += 1
        return ocr

    monkeypatch.setattr(pe, "load_phone_bundle", load)


def _crashed_rec_pass(c, monkeypatch, call=2):
    """GW-REC-045 on gdc_test stopped by a crash: 2 of 6 crops written when ``call`` is 2."""
    c.gate()
    monkeypatch.setattr(pe, "CROPS_PER_CALL", 2)
    ocr = stand_in_ocr()
    _crash_on(monkeypatch, ocr, call)
    _load_this(monkeypatch, c, ocr)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(c, "rec", "gdc_test", "GW-REC-045")
    monkeypatch.setattr(pe, "load_phone_bundle", lambda *a: stand_in_ocr())


# ------------------------------------------------------------------------------- the records


def test_validation_crops_write_phase14_records_and_log_nothing(cand, capsys):
    assert _run(cand, "rec", "hiertext_val", "GW-REC-044") == 0
    recs = _records(_out(cand, "GW-REC-044", "hiertext_val"))
    assert len(recs) == 6 and _rows(cand.log) == []
    # Phase 14's crop record (`ext_eval._rec_records`), plus the segment.
    assert set(recs[0]) == {"key", "ref", "pred", "conf_t1", "conf_t", "time_ms",
                            "crop_height_px", "blur", "contrast", "line_height_px",
                            "rotation_deg", "seg"}  # fmt: skip
    meta = _meta(cand, "GW-REC-044", "hiertext_val")
    assert (meta["phase"], meta["experiment"], meta["split"]) == (22, "GW-REC-044", "validation")
    assert meta["clearance"] is None and meta["smoke"] is None
    assert capsys.readouterr().out.strip().endswith("GW-REC-044 rec hiertext_val: 6 new, 6 total")


def test_test_pages_log_one_phase22_row_naming_the_candidate(cand):
    cand.gate()
    assert _run(cand, "e2e", "gdc_test", "GW-E2E-020") == 0
    (row,) = _rows(cand.log)
    assert row.endswith(
        f"| gdc test | GW-E2E-020 | glypharis_phone_candidate (e2e) | 22 | {REASON} |"
    )
    recs = _records(_out(cand, "GW-E2E-020", "gdc_test"))
    assert [r["key"] for r in recs] == ["gdc0", "gdc1"]
    assert set(recs[0]) == {"key", "lines", "time_ms", "seg"}
    for line in recs[0]["lines"]:
        assert set(line) == {"polygon", "quad", "text", "confidence", "conf_t1", "det_score"}
    meta = _meta(cand, "GW-E2E-020", "gdc_test")
    assert meta["clearance"]["operator_check"]["verdict"] == "go"
    gate = meta["clearance"]["validation_gate"]
    assert gate["tests_met"] == sorted(pe.GATE_TESTS)
    assert (gate["bundle_manifest_sha256"], gate["tag_commit"]) == (cand.sha, TAG_COMMIT)
    assert gate["file"] == pe.GATE.as_posix()


def test_crops_are_read_256_per_call_one_line_per_network_run(cand, monkeypatch):
    ocr = stand_in_ocr()
    assert isinstance(ocr.recognizer, SingleLineRecognizer)
    sizes, runs = [], []
    read, forward = ocr.recognizer.read, ocr.recognizer.model.forward

    def counted_read(crops, temperatures=()):
        sizes.append(len(crops))
        return read(crops, temperatures)

    def counted_forward(x, *args, **kwargs):
        runs.append(int(x.shape[0]))
        return forward(x, *args, **kwargs)

    monkeypatch.setattr(ocr.recognizer, "read", counted_read)
    monkeypatch.setattr(ocr.recognizer.model, "forward", counted_forward)
    _load_this(monkeypatch, cand, ocr)
    assert pe.CROPS_PER_CALL == 256
    assert _run(cand, "rec", "hiertext_val", "GW-REC-044") == 0
    assert sizes == [6] and runs == [1] * 6
    assert _meta(cand, "GW-REC-044", "hiertext_val")["identity"]["batch"] == 256

    # With a smaller call size the crops go in calls of that size, in pack order.
    sizes.clear()
    monkeypatch.setattr(pe, "CROPS_PER_CALL", 4)
    _out(cand, "GW-REC-044", "hiertext_val").unlink()
    assert _run(cand, "rec", "hiertext_val", "GW-REC-044") == 0
    assert sizes == [4, 2]


def test_meta_holds_phase14_fields_and_the_bundle_identity(cand):
    cand.gate()
    assert _run(cand, "rec", "gdc_test", "GW-REC-045") == 0
    meta = _meta(cand, "GW-REC-045", "gdc_test")
    assert set(meta) == PHASE14_FIELDS | ADDED_FIELDS
    assert (
        meta["bundle_manifest_sha256"] == cand.sha == meta["identity"]["bundle"]["manifest_sha256"]
    )
    assert meta["tag_commit"] == TAG_COMMIT == meta["identity"]["tag_commit"]
    assert meta["pinned"] == PINNED and meta["v1_code"] == V1_CODE
    assert meta["bundle"]["files"] == cand.manifest["files"]
    assert meta["bundle"]["committed_manifest"]["file"] == pe.BUNDLE_REPORT.as_posix()
    assert meta["settings"] == {k: cand.manifest[k] for k in
                                ("detector", "recognizer", "pipeline", "batching")}  # fmt: skip
    assert (meta["expected"], meta["samples"]) == (6, 6)
    assert meta["confidences_at"] == [1.0, meta["temperature"]]
    # What the Phase 14 scorer finds in a run's metadata is there.
    assert {"device", "device_name", "torch", "cuda", "cudnn", "numerics"} <= set(
        meta["provenance"]
    )
    assert meta["provenance"]["device"] == "cpu"
    assert meta["identity"]["environment"]["numerics"]["threads"] == 10
    (run,) = meta["runs"]
    assert run["status"] == "done" and run["threads_checked"] == {"before": THREADS,
                                                                    "after": THREADS}  # fmt: skip
    # No path in the metadata is absolute.
    text = json.dumps(meta)
    assert str(cand.repo) not in text and str(cand.data) not in text


def test_no_metric_is_written_and_no_entry_says_v1(cand):
    cand.gate()
    assert _run(cand, "rec", "gdc_test", "GW-REC-045") == 0
    assert _run(cand, "e2e", "gdc_test", "GW-E2E-020") == 0
    for exp in ("GW-REC-045", "GW-E2E-020"):
        meta = _meta(cand, exp, "gdc_test")
        assert not set(_keys(meta)) & METRIC_KEYS
        assert meta["system"] == "glypharis_phone_candidate"
        assert "not v1" in meta["version"] and "not v1" in meta["version_label"]
        assert "glypharis_v1" not in json.dumps(meta)
    assert all("glypharis_phone_candidate" in r and "glypharis (" not in r for r in _rows(cand.log))


# ------------------------------------------------------------------------- operator and gate


def test_a_no_go_operator_check_is_refused_before_anything_is_read(cand):
    cand.gate()
    _write(cand.repo / pe.OPERATOR_CHECK, {"verdict": "no-go", "no_go_reasons": ["made up"]})
    with pytest.raises(SystemExit, match="operator check says 'no-go'"):
        _run(cand, "rec", "gdc_test", "GW-REC-045")
    assert _rows(cand.log) == [] and cand.calls["load"] == 0
    assert not _out(cand, "GW-REC-045", "gdc_test").parent.exists()
    (cand.repo / pe.OPERATOR_CHECK).unlink()
    with pytest.raises(SystemExit, match=r"operator check .* is missing"):
        _run(cand, "rec", "gdc_test", "GW-REC-045")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (None, "validation gate .* is missing"),
        (
            {
                "tests": {
                    "line_cer": {"verdict": "met"},
                    "page_cer": {"verdict": "not shown"},
                    "e2e_f1": {"verdict": "met"},
                }
            },
            "not met \\(page_cer\\)",
        ),
        (
            {
                "tests": {
                    "line_cer": {"verdict": "met"},
                    "page_cer": {"verdict": "met"},
                    "e2e_f1": {"verdict": "not met"},
                }
            },
            "not met \\(e2e_f1\\)",
        ),
        ({"met": False}, "not met \\(met: false\\)"),
        (
            {"tests": {"line_cer": {"verdict": "met"}, "page_cer": {"verdict": "met"}}},
            "holds tests",
        ),
        ({"tests": "met"}, "holds tests"),
        ({"bundle_manifest_sha256": "0" * 64}, "another bundle"),
        ({"bundle_manifest_sha256": None}, "another bundle"),
        ({"tag_commit": "0" * 40}, "another tag commit"),
    ],
)
def test_an_unmet_or_mismatched_gate_is_refused(cand, change, message):
    if change is not None:
        cand.gate(**change)
    with pytest.raises(SystemExit, match=message):
        _run(cand, "e2e", "gdc_test", "GW-E2E-020")
    assert _rows(cand.log) == [] and cand.calls["load"] == 0


def test_the_gate_may_list_its_tests(cand):
    cand.gate(tests=[{"name": t, "verdict": "met"} for t in pe.GATE_TESTS])
    assert _run(cand, "rec", "gdc_test", "GW-REC-045") == 0


def test_validation_runs_need_no_gate_and_are_not_run_again_once_it_exists(cand):
    assert not (cand.repo / pe.GATE).exists()
    assert _run(cand, "e2e", "hiertext_val", "GW-E2E-019") == 0
    cand.gate()
    assert _run(cand, "e2e", "hiertext_val", "GW-E2E-019") == 0  # complete: nothing to do
    _out(cand, "GW-E2E-019", "hiertext_val").unlink()
    with pytest.raises(SystemExit, match="gate has been scored"):
        _run(cand, "e2e", "hiertext_val", "GW-E2E-019")
    assert cand.calls["load"] == 1 and _rows(cand.log) == []


def test_an_unpinned_run_cannot_clear_the_gate(cand, monkeypatch):
    def refuse(tag=None, root=None):
        raise RuntimeError(f"the tag {tag} does not exist; the code is not pinned")

    cand.gate()
    monkeypatch.setattr(pe, "pinned_code_check", refuse)
    with pytest.raises(SystemExit, match="pinned code check failed"):
        _run(cand, "rec", "gdc_test", "GW-REC-045")
    assert _rows(cand.log) == [] and cand.calls["load"] == 0


# --------------------------------------------------------------------------- second passes


@pytest.mark.parametrize("how", ["delete", "move", "delete both", "move both"])
def test_a_second_pass_after_the_output_is_moved_or_deleted_is_refused(cand, tmp_path, how):
    cand.gate()
    assert _run(cand, "rec", "gdc_test", "GW-REC-045") == 0
    out = _out(cand, "GW-REC-045", "gdc_test")
    files = [out, out.with_suffix(".meta.json")] if "both" in how else [out]
    for f in files:
        if how.startswith("delete"):
            f.unlink()
        else:
            f.rename(tmp_path / f.name)
    with pytest.raises(SystemExit, match="does not start a second pass"):
        _run(cand, "rec", "gdc_test", "GW-REC-045")
    with pytest.raises(SystemExit, match="does not start a second pass"):
        _run(cand, "rec", "gdc_test", "GW-REC-045", "--resume")
    assert len(_rows(cand.log)) == 1 and cand.calls["load"] == 1


def test_a_second_pass_after_a_crashed_output_is_deleted_is_refused(cand, monkeypatch):
    _crashed_rec_pass(cand, monkeypatch)
    _out(cand, "GW-REC-045", "gdc_test").unlink()
    with pytest.raises(SystemExit, match="does not start a second pass"):
        _run(cand, "rec", "gdc_test", "GW-REC-045", "--resume")
    assert len(_rows(cand.log)) == 1


@pytest.mark.parametrize("keep", [0, 3])
def test_a_finished_pass_cut_short_is_not_a_resume(cand, keep):
    cand.gate()
    assert _run(cand, "rec", "gdc_test", "GW-REC-045") == 0
    out = _out(cand, "GW-REC-045", "gdc_test")
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(lines[:keep]), encoding="utf-8", newline="\n")
    with pytest.raises(SystemExit, match=r"this pass ended|cut short or emptied"):
        _run(cand, "rec", "gdc_test", "GW-REC-045", "--resume")
    assert len(_rows(cand.log)) == 1 and cand.calls["load"] == 1


def test_a_complete_test_set_is_not_run_again(cand, monkeypatch):
    cand.gate()
    assert _run(cand, "e2e", "gdc_test", "GW-E2E-020") == 0
    out = _out(cand, "GW-E2E-020", "gdc_test")
    before = _digest(out.parent)
    monkeypatch.setattr(pe, "load_phone_bundle", lambda *a: pytest.fail("model loaded"))
    assert _run(cand, "e2e", "gdc_test", "GW-E2E-020") == 0
    assert len(_rows(cand.log)) == 1 and _digest(out.parent) == before


def test_an_output_without_a_logged_pass_is_refused(cand):
    cand.gate()
    out = _out(cand, "GW-REC-045", "gdc_test")
    out.parent.mkdir(parents=True)
    out.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="no logged prediction pass"):
        _run(cand, "rec", "gdc_test", "GW-REC-045")
    out.unlink()
    out.with_suffix(".meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="no logged prediction pass"):
        _run(cand, "rec", "gdc_test", "GW-REC-045")
    assert _rows(cand.log) == [] and cand.calls["load"] == 0


def test_resume_without_a_started_pass_is_refused(cand):
    cand.gate()
    with pytest.raises(SystemExit, match="no prediction pass was started"):
        _run(cand, "rec", "gdc_test", "GW-REC-045", "--resume")
    assert _rows(cand.log) == [] and cand.calls["load"] == 0


def test_scoring_rows_do_not_count_as_a_prediction_pass(cand):
    cand.gate()
    with cand.log.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(f"| 2026-11-21 | gdc test | GW-REC-045 | {pe.SYSTEM} (rec), scoring | 22 | x |\n")
        fh.write(f"| 2026-11-21 | gdc test | GW-REC-039 | {pe.SYSTEM} (rec) | 22 | x |\n")
    assert _run(cand, "rec", "gdc_test", "GW-REC-045") == 0


# ----------------------------------------------------------------------------------- resumes


def test_a_crash_resumes_only_with_resume_and_its_row_says_so(cand, monkeypatch):
    _crashed_rec_pass(cand, monkeypatch)
    out = _out(cand, "GW-REC-045", "gdc_test")
    assert len(_records(out)) == 2
    with pytest.raises(SystemExit, match="only as a resume, with --resume"):
        _run(cand, "rec", "gdc_test", "GW-REC-045")
    assert len(_rows(cand.log)) == 1
    assert _run(cand, "rec", "gdc_test", "GW-REC-045", "--resume") == 0
    recs = _records(out)
    assert len({r["key"] for r in recs}) == len(recs) == 6
    assert [r["seg"] for r in recs] == [0, 0, 1, 1, 1, 1]
    rows = _rows(cand.log)
    assert len(rows) == 2
    assert rows[1].endswith(f"| {REASON} (resumed: 2 of 6 samples already predicted) |")
    meta = _meta(cand, "GW-REC-045", "gdc_test")
    assert [r["status"] for r in meta["runs"]] == ["running", "done"] and meta["samples"] == 6


def test_a_crash_before_the_first_record_resumes_with_a_marked_row(cand, monkeypatch):
    _crashed_rec_pass(cand, monkeypatch, call=1)
    assert _records(_out(cand, "GW-REC-045", "gdc_test")) == []
    assert _run(cand, "rec", "gdc_test", "GW-REC-045", "--resume") == 0
    rows = _rows(cand.log)
    assert len(rows) == 2 and "(resumed: 0 of 6 samples already predicted) |" in rows[1]


def test_a_crash_in_pages_resumes(cand, monkeypatch):
    cand.gate()
    ocr = stand_in_ocr()
    _crash_on(monkeypatch, ocr, 2, what="ocr")
    _load_this(monkeypatch, cand, ocr)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(cand, "e2e", "gdc_test", "GW-E2E-020")
    monkeypatch.setattr(pe, "load_phone_bundle", lambda *a: stand_in_ocr())
    assert _run(cand, "e2e", "gdc_test", "GW-E2E-020", "--resume") == 0
    assert [r["key"] for r in _records(_out(cand, "GW-E2E-020", "gdc_test"))] == ["gdc0", "gdc1"]
    assert "(resumed: 1 of 2 samples already predicted)" in _rows(cand.log)[1]


@pytest.mark.parametrize("what", ["environment", "bundle", "inputs"])
def test_a_resume_with_another_identity_is_refused_before_logging(cand, monkeypatch, what):
    _crashed_rec_pass(cand, monkeypatch)
    if what == "environment":
        real = pe.environment

        def other(ocr, sessions):
            return {**real(ocr, sessions), "torch": "0.0.0"}

        monkeypatch.setattr(pe, "environment", other)
    elif what == "bundle":
        meta_path = _out(cand, "GW-REC-045", "gdc_test").with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["identity"]["bundle"]["manifest_sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    else:
        real_inputs = pe.check_inputs

        def other_inputs(*args, **kwargs):
            got = real_inputs(*args, **kwargs)
            got["pack"]["bin_sha256"] = "0" * 64
            return got

        monkeypatch.setattr(pe, "check_inputs", other_inputs)
    with pytest.raises(SystemExit, match="a resume must be the same run"):
        _run(cand, "rec", "gdc_test", "GW-REC-045", "--resume")
    assert len(_rows(cand.log)) == 1


# --------------------------------------------------------------------- plan, bundle, guards


@pytest.mark.parametrize(
    ("mode", "eval_set", "experiment", "message"),
    [
        ("rec", "gdc_test", "GW-REC-039", "not a rec experiment of this phase"),
        ("e2e", "gdc_test", "GW-E2E-007", "not a e2e experiment of this phase"),
        ("rec", "hiertext_val", "GW-REC-045", "runs on hiertext_test"),
        ("rec", "gdc_test", "GW-REC-044", "runs on hiertext_val"),
        ("e2e", "glyphsynth_val", "GW-E2E-019", "runs on hiertext_val"),
        ("rec", "omnidocbench_test", "GW-REC-045", "not omnidocbench_test"),
        ("e2e", "iiit5k_test", "GW-E2E-020", "not iiit5k_test"),
        ("e2e", "gdc_test", "GW-REC-045", "not a e2e experiment"),
    ],
)
def test_runs_outside_the_registered_plan_are_refused(cand, mode, eval_set, experiment, message):
    with pytest.raises(SystemExit, match=message):
        _run(cand, mode, eval_set, experiment)
    assert cand.calls["load"] == 0


def test_there_is_no_oracle_mode(cand, capsys):
    with pytest.raises(SystemExit):
        _run(cand, "oracle", "gdc_test", "GW-E2E-020")
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("committed", "differs from reports/final_validation"),
        ("no manifest", "manifest.json is missing"),
        ("no equivalence", "equivalence check .* is missing"),
        ("equivalence failed", "equivalence check did not pass"),
        ("equivalence other bundle", "made on another bundle manifest"),
        ("file changed", "bundle refused"),
        ("says v1", "does not say it is a new version"),
    ],
)
def test_the_bundle_must_be_the_frozen_checked_one(cand, fault, message):
    repo, bundle = cand.repo, cand.repo / pe.BUNDLE
    if fault == "committed":
        _write(repo / pe.BUNDLE_REPORT, {**cand.manifest, "arm": "fused"})
    elif fault == "no manifest":
        (bundle / "manifest.json").unlink()
    elif fault == "no equivalence":
        (repo / pe.EQUIVALENCE).unlink()
    elif fault == "equivalence failed":
        _write(repo / pe.EQUIVALENCE, {"pass": False, "bundle_manifest_sha256": cand.sha})
    elif fault == "equivalence other bundle":
        _write(repo / pe.EQUIVALENCE, {"pass": True, "bundle_manifest_sha256": "0" * 64})
    elif fault == "file changed":
        (bundle / "charset.json").write_text("changed", encoding="utf-8")
    else:
        doc = {**cand.manifest, "version": "glypharis_v1", "version_label": "GLYPHARIS v1"}
        _write(bundle / "manifest.json", doc)
        shutil.copyfile(bundle / "manifest.json", repo / pe.BUNDLE_REPORT)
        sha = _sha(bundle / "manifest.json")
        _write(repo / pe.EQUIVALENCE, {"pass": True, "bundle_manifest_sha256": sha})
    with pytest.raises(SystemExit, match=message):
        _run(cand, "rec", "hiertext_val", "GW-REC-044")
    assert cand.calls["load"] == 0


def test_a_loaded_recognizer_must_follow_the_bundles_batching_rule(cand, monkeypatch):
    def batched(path, threads):
        ocr = tiny_frozen().ocr
        ocr.model_version = f"{VERSION} {ARM} ({VERSION_LABEL})"
        return ocr

    monkeypatch.setattr(pe, "load_phone_bundle", batched)
    with pytest.raises(SystemExit, match="batching rule"):
        _run(cand, "rec", "hiertext_val", "GW-REC-044")


def test_the_branch_the_tag_and_the_v1_code_are_required(cand, monkeypatch):
    def refuse(tag=None, root=None):
        raise RuntimeError("the tag does not exist")

    monkeypatch.setattr(pe, "pinned_code_check", refuse)
    with pytest.raises(SystemExit, match="pinned code check failed"):
        _run(cand, "rec", "hiertext_val", "GW-REC-044")  # a validation run needs it too
    monkeypatch.setattr(pe, "pinned_code_check", lambda tag=None, root=None: dict(PINNED))

    def not_v1(root=None):
        raise pe.V1CodeError("the v1 code differs from v1-code")

    monkeypatch.setattr(pe, "assert_v1_code", not_v1)
    with pytest.raises(SystemExit, match="v1 code check failed"):
        _run(cand, "rec", "hiertext_val", "GW-REC-044")
    monkeypatch.setattr(pe, "assert_v1_code", lambda root=None: dict(V1_CODE))

    def wrong_branch(name, start=None):
        raise RuntimeError(f"this run must start on branch {name}; HEAD is on main")

    monkeypatch.setattr(pe, "require_branch", wrong_branch)
    with pytest.raises(SystemExit, match="phase/22-final-validation"):
        _run(cand, "rec", "hiertext_val", "GW-REC-044")
    assert cand.calls["load"] == 0


def test_the_split_comes_from_the_records(cand):
    spec = EVAL_SETS["hiertext_val"]
    pages = list(read_pages(cand.data / spec["pages"]))
    write_pages(cand.data / spec["pages"], [dataclasses.replace(p, split="test") for p in pages])
    with pytest.raises(SystemExit, match="records name splits"):
        _run(cand, "e2e", "hiertext_val", "GW-E2E-019")


# ------------------------------------------------------------------------------- smoke runs


@pytest.mark.parametrize(
    ("extra", "eval_set", "experiment", "message"),
    [
        (["--limit", "2"], "hiertext_val", "GW-REC-044", "needs --out-root"),
        (["--allow-unpinned"], "hiertext_val", "GW-REC-044", "needs --out-root"),
        (["--limit", "0", "--out-root", "{tmp}"], "hiertext_val", "GW-REC-044", "at least 1"),
        (["--out-root", "{data}/processed/phase22"], "hiertext_val", "GW-REC-044", "outside"),
        (["--out-root", "{data}/processed/phase14"], "hiertext_val", "GW-REC-044", "outside"),
        (["--limit", "2", "--out-root", "{tmp}"], "gdc_test", "GW-REC-045", "smoke runs"),
        (["--allow-unpinned", "--out-root", "{tmp}"], "gdc_test", "GW-REC-045", "smoke runs"),
        (["--out-root", "{tmp}"], "gdc_test", "GW-REC-045", "smoke runs"),
        (["--resume"], "hiertext_val", "GW-REC-044", "--resume is for test sets"),
    ],
)
def test_smoke_options_are_for_validation_only_and_outside_processed(
    cand, tmp_path, extra, eval_set, experiment, message
):
    cand.gate()
    args = [a.format(tmp=tmp_path / "smoke", data=cand.data) for a in extra]
    with pytest.raises(SystemExit, match=message):
        _run(cand, "rec", eval_set, experiment, *args)
    assert _rows(cand.log) == [] and cand.calls["load"] == 0


def test_a_smoke_run_writes_its_limit_outside_the_registered_folders(cand, tmp_path, monkeypatch):
    def refuse(tag=None, root=None):
        raise RuntimeError("the tag does not exist")

    monkeypatch.setattr(pe, "pinned_code_check", refuse)
    counted = []
    real_inputs = pe.check_inputs

    def recorded_count(data, eval_set, mode, n_pages, **kwargs):
        counted.append(n_pages)
        return real_inputs(data, eval_set, mode, n_pages, **kwargs)

    monkeypatch.setattr(pe, "check_inputs", recorded_count)
    cand.gate()  # a smoke run is not refused once the gate exists
    smoke = tmp_path / "smoke"
    for mode, exp in (("rec", "GW-REC-044"), ("e2e", "GW-E2E-019")):
        args = ["--limit", "1", "--out-root", str(smoke), "--allow-unpinned"]
        assert _run(cand, mode, "hiertext_val", exp, *args) == 0
        out = smoke / exp / "hiertext_val.jsonl"
        assert len(_records(out)) == 1
        meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["smoke"]["limit"] == 1 and meta["smoke"]["allow_unpinned"] is True
        assert meta["expected"] == 1 and meta["pinned"]["trees"] == "not pinned"
        assert meta["tag_commit"] is None
    # The page file is checked whole against its recorded count, not the limited pages.
    assert counted == [None, 2]
    assert not (cand.data / "processed/phase22").exists() and _rows(cand.log) == []


# ---------------------------------------------------------------------------------- isolation


def test_nothing_changes_under_phase14_the_baselines_or_the_outputs_manifest(cand):
    data = cand.data
    planted = {
        "processed/phase14/GW-REC-039/gdc_test.jsonl": '{"key": "v1"}\n',
        "processed/phase14/GW-REC-039/gdc_test.meta.json": "{}\n",
        "processed/baselines/GW-BASE-001/tesseract_rec_gdc_test.jsonl": '{"key": "b"}\n',
        "processed/baselines_phase14/GW-BASE-010/tesseract_rec_gdc_test.jsonl": '{"key": "b"}\n',
        "manifests/processed/phase14_outputs.json": '{"outputs": []}\n',
    }
    for rel, text in planted.items():
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_text(text, encoding="utf-8")
    watched = ("processed/phase14", "processed/baselines", "processed/baselines_phase14")
    before = {w: _digest(data / w) for w in watched}
    manifest = _sha(data / "manifests/processed/phase14_outputs.json")
    assert _run(cand, "rec", "hiertext_val", "GW-REC-044") == 0
    cand.gate()
    for mode, exp, eval_set in (("rec", "GW-REC-045", "gdc_test"), ("rec", "GW-REC-045",
                                "iiit5k_test"), ("e2e", "GW-E2E-020", "gdc_test")):  # fmt: skip
        assert _run(cand, mode, eval_set, exp) == 0
    assert {w: _digest(data / w) for w in watched} == before
    assert _sha(data / "manifests/processed/phase14_outputs.json") == manifest
    written = set(_digest(data / "processed/phase22"))
    assert written == {f"{exp}/{s}.{ext}" for exp, s in (("GW-REC-044", "hiertext_val"),
                       ("GW-REC-045", "gdc_test"), ("GW-REC-045", "iiit5k_test"),
                       ("GW-E2E-020", "gdc_test")) for ext in ("jsonl", "meta.json")}  # fmt: skip


@pytest.fixture
def marked_scan(cand, monkeypatch):
    """GDC-scan with the marker in its references (`test_ext_eval`'s), its image file name and
    every prediction the stand-in makes."""
    spec = EVAL_SETS["gdc_scan_test"]
    shutil.copyfile(cand.data / "raw/page.png", cand.data / f"raw/{MARKER}.png")
    pages = [dataclasses.replace(p, image=f"raw/{MARKER}.png")
             for p in read_pages(cand.data / spec["pages"])]  # fmt: skip
    write_pages(cand.data / spec["pages"], pages)

    def marked():
        ocr = stand_in_ocr()
        read = ocr.recognizer.read

        def marking(crops, temperatures=()):
            return [dataclasses.replace(r, text=f"{MARKER} {r.text}")
                    for r in read(crops, temperatures)]  # fmt: skip

        ocr.recognizer.read = marking
        return ocr

    monkeypatch.setattr(pe, "load_phone_bundle", lambda *a: marked())
    cand.gate()
    return cand


@pytest.mark.parametrize(("mode", "experiment"), [("rec", "GW-REC-045"), ("e2e", "GW-E2E-020")])
def test_the_gdc_scan_marker_appears_in_no_log_row_console_output_or_metadata(
    marked_scan, capsys, mode, experiment
):
    c = marked_scan
    assert _run(c, mode, "gdc_scan_test", experiment) == 0
    out = _out(c, experiment, "gdc_scan_test")
    assert MARKER in out.read_text(encoding="utf-8")  # the git-ignored predictions hold it
    printed = capsys.readouterr()
    assert MARKER not in printed.out + printed.err
    assert MARKER not in c.log.read_text(encoding="utf-8")
    assert MARKER not in out.with_suffix(".meta.json").read_text(encoding="utf-8")
    assert len(_rows(c.log)) == 1


def test_a_refused_private_set_names_no_file_or_key(marked_scan, capsys):
    c = marked_scan
    spec = EVAL_SETS["gdc_scan_test"]
    pages = [dataclasses.replace(p, width=p.width + 1) for p in read_pages(c.data / spec["pages"])]
    write_pages(c.data / spec["pages"], pages)
    with pytest.raises(SystemExit) as refused:
        _run(c, "e2e", "gdc_scan_test", "GW-E2E-020")
    assert "differ in size" in str(refused.value) and MARKER not in str(refused.value)
    # A crop without its line on a page: `line_boxes` would name the key.
    with pytest.raises(SystemExit) as refused:
        write_pages(c.data / spec["pages"], [dataclasses.replace(p, lines=p.lines[1:])
                                             for p in pages])  # fmt: skip
        _run(c, "rec", "gdc_scan_test", "GW-REC-045")
    assert "not shown for a private set" in str(refused.value)
    assert "scan0" not in str(refused.value) and MARKER not in str(refused.value)
    printed = capsys.readouterr()
    assert MARKER not in printed.out + printed.err and _rows(c.log) == []


# ------------------------------------------------------------------- review fixes: edge paths


def test_a_crashed_output_cut_short_or_emptied_is_not_resumed(cand, monkeypatch):
    _crashed_rec_pass(cand, monkeypatch)  # 2 of 6 written, and the metadata says so
    out = _out(cand, "GW-REC-045", "gdc_test")
    assert _meta(cand, "GW-REC-045", "gdc_test")["runs"][-1]["written"] == 2
    for keep in (1, 0):
        lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
        out.write_text("".join(lines[:keep]), encoding="utf-8", newline="\n")
        with pytest.raises(SystemExit, match="cut short or emptied"):
            _run(cand, "rec", "gdc_test", "GW-REC-045", "--resume")
    assert len(_rows(cand.log)) == 1


def test_a_pass_killed_after_its_last_record_closes_as_done(cand):
    cand.gate()
    assert _run(cand, "rec", "gdc_test", "GW-REC-045") == 0
    path = _out(cand, "GW-REC-045", "gdc_test").with_suffix(".meta.json")
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["runs"][-1]["status"] = "running"  # as if killed after the last record was written
    _write(path, meta)
    assert _run(cand, "rec", "gdc_test", "GW-REC-045") == 0
    last = _meta(cand, "GW-REC-045", "gdc_test")["runs"][-1]
    assert last["status"] == "done" and last["closed_by_later_start"] is True
    assert len(_rows(cand.log)) == 1 and cand.calls["load"] == 1


def test_a_crash_right_after_the_access_row_resumes(cand, monkeypatch):
    cand.gate()

    def crash(*args, **kwargs):
        pe.open_test_access.__wrapped__(*args, **kwargs)  # the row is written
        raise KeyboardInterrupt("killed after the row")

    real = pe.open_test_access
    crash.__wrapped__ = real  # type: ignore[attr-defined]
    monkeypatch.setattr(pe, "open_test_access", crash)
    pe.open_test_access.__wrapped__ = real  # type: ignore[attr-defined]
    with pytest.raises(KeyboardInterrupt):
        _run(cand, "rec", "gdc_test", "GW-REC-045")
    monkeypatch.setattr(pe, "open_test_access", real)
    assert len(_rows(cand.log)) == 1
    assert _meta(cand, "GW-REC-045", "gdc_test")["runs"][-1]["status"] == "pending"
    assert _run(cand, "rec", "gdc_test", "GW-REC-045", "--resume") == 0
    rows = _rows(cand.log)
    assert len(rows) == 2 and "(resumed: 0 of 6 samples already predicted)" in rows[1]


def test_an_attempt_stopped_before_its_access_row_starts_fresh(cand, monkeypatch):
    cand.gate()

    def stop(*args, **kwargs):
        raise KeyboardInterrupt("stopped before the row")

    real = pe.open_test_access
    monkeypatch.setattr(pe, "open_test_access", stop)
    with pytest.raises(KeyboardInterrupt):
        _run(cand, "rec", "gdc_test", "GW-REC-045")
    assert _rows(cand.log) == []
    assert _meta(cand, "GW-REC-045", "gdc_test")["runs"][-1]["status"] == "pending"
    monkeypatch.setattr(pe, "open_test_access", real)
    assert _run(cand, "rec", "gdc_test", "GW-REC-045") == 0  # not a resume: a first pass
    rows = _rows(cand.log)
    assert len(rows) == 1 and "resumed" not in rows[0]
    assert [r["status"] for r in _meta(cand, "GW-REC-045", "gdc_test")["runs"]] == ["done"]


def test_a_refused_validation_run_keeps_the_gated_metadata(cand):
    assert _run(cand, "e2e", "hiertext_val", "GW-E2E-019") == 0
    cand.gate()
    out = _out(cand, "GW-E2E-019", "hiertext_val")
    out.rename(out.with_suffix(".moved"))
    with pytest.raises(SystemExit, match="gate has been scored"):
        _run(cand, "e2e", "hiertext_val", "GW-E2E-019")
    assert out.with_suffix(".meta.json").is_file()


def test_a_private_set_that_fails_mid_pass_names_no_file(marked_scan, monkeypatch, capsys):
    c = marked_scan
    ocr = stand_in_ocr()

    def failing(*args, **kwargs):
        raise ValueError(f"cannot decode data/raw/{MARKER}.png")

    monkeypatch.setattr(ocr, "predict", failing)
    _load_this(monkeypatch, c, ocr)
    with pytest.raises(SystemExit) as stopped:
        _run(c, "e2e", "gdc_scan_test", "GW-E2E-020")
    assert "not shown for a private set" in str(stopped.value)
    assert MARKER not in str(stopped.value)
    printed = capsys.readouterr()
    assert MARKER not in printed.out + printed.err
