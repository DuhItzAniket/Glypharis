"""The Phase 22 v1 wrapper (scripts/phase22_v1_repro.py), on made-up data in tmp_path only.

The re-run tests reuse `test_ext_eval`'s tiny data root and stand-in model, and the re-score tests
reuse `test_phase14_score`'s synthetic layout. No real prediction file, test set, checkpoint or
access log is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import types
from pathlib import Path

import evaluate_baselines as eb
import ext_eval
import phase14_score as ps
import phase22_v1_repro as repro
import pytest
from test_ext_eval import MARKER as RERUN_MARKER
from test_ext_eval import root as ext_root  # noqa: F401
from test_phase14_score import MARKER as SCORE_MARKER
from test_phase14_score import (
    _rerecord,
    no_phase14_tag,  # noqa: F401
)
from test_phase14_score import layout as score_layout  # noqa: F401

from glypharis.pipeline import frozen

REPO = Path(__file__).resolve().parents[1]
WRAPPER_COMMIT = "f" * 40
REASON = "Phase 22 v1 re-run (test)"


def _tree_digest(folder: Path) -> dict[str, str]:
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _link_folder(link: Path, target: Path) -> None:
    """A junction on Windows (no privilege needed), a symlink elsewhere."""
    target.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


@pytest.fixture
def pinned(monkeypatch):
    calls = []

    def check(tag="phase14-eval", *, root=Path(".")):
        calls.append((tag, Path(root)))
        return {"tag": tag, "tag_commit": "e" * 40, "head": "e" * 40, "trees": "equal"}

    monkeypatch.setattr(frozen, "pinned_code_check", check)
    return calls


@pytest.fixture
def rerun(ext_root, pinned, tmp_path, monkeypatch):  # noqa: F811
    """`root` as an isolated re-run root, started from a stand-in for the tag worktree."""
    data, log, calls = ext_root
    (data / "processed/phase14").mkdir(parents=True)
    monkeypatch.chdir(REPO)
    monkeypatch.setattr(ext_eval, "TEST_DEVICE", "cuda")  # the wrapper keeps the registered device
    monkeypatch.setattr(repro, "ACCESS_LOG", log)
    monkeypatch.setattr(repro, "wrapper_commit", lambda: WRAPPER_COMMIT)
    monkeypatch.setattr(repro, "TEST_ORDER", ())  # most tests start one set on its own
    for name in ("PHASE14_BRANCH", "open_test_access", "_write_json"):
        monkeypatch.setattr(ext_eval, name, getattr(ext_eval, name))
    store = tmp_path / "phase14_store"
    (store / "GW-REC-039").mkdir(parents=True)
    (store / "GW-REC-039/gdc_test.jsonl").write_text('{"key": "original"}\n', encoding="utf-8")
    return data, log, calls, store


def _rerun(data, mode, eval_set, folder, new_id, reason=REASON):
    return repro.main(
        [
            "rerun",
            mode,
            "--set",
            eval_set,
            "--folder-id",
            folder,
            "--experiment",
            new_id,
            "--reason",
            reason,
            "--data-root",
            str(data),
        ]
    )


def _rows(log):
    return [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.startswith("| 20")]


def _out(data, folder, eval_set):
    return data / "processed/phase14" / folder / f"{eval_set}.jsonl"


def _meta(data, folder, eval_set):
    path = _out(data, folder, eval_set).with_suffix(".meta.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest(data):
    return json.loads((data / ext_eval.OUTPUTS_MANIFEST).read_text(encoding="utf-8"))


def _drop_manifest_entries(data):
    doc = _manifest(data)
    doc["outputs"] = []
    (data / ext_eval.OUTPUTS_MANIFEST).write_text(json.dumps(doc), encoding="utf-8")


def _cut_output(data, folder, eval_set, keep):
    """Keep the first ``keep`` records; returns how many the output held."""
    out = _out(data, folder, eval_set)
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(lines[:keep]), encoding="utf-8", newline="\n")
    return len(lines)


def _set_last_status(data, folder, eval_set, status):
    path = _out(data, folder, eval_set).with_suffix(".meta.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["runs"][-1]["status"] = status
    path.write_text(json.dumps(doc), encoding="utf-8")


def _crash(data, folder, eval_set, keep):
    """What a real crash leaves: a cut output, a ``running`` run and no manifest entry."""
    total = _cut_output(data, folder, eval_set, keep)
    _set_last_status(data, folder, eval_set, "running")
    _drop_manifest_entries(data)
    return total


def stamp_rerun_root(data):
    """Make every GLYPHARIS file under ``data`` look as `rerun` leaves it, and record it again."""
    for job in ps.glypharis_jobs(data):
        path = job.path.with_suffix(".meta.json")
        meta = json.loads(path.read_text(encoding="utf-8"))
        new_id = repro.RERUN_IDS[job.experiment]
        meta.update(phase=22, experiment=new_id, folder_id=job.experiment)
        meta["phase22"] = {
            "experiment": new_id,
            "folder_id": job.experiment,
            "wrapper_tag": repro.WRAPPER_TAG,
        }
        path.write_text(json.dumps(meta), encoding="utf-8")
    _rerecord(data)


# ------------------------------------------------------------------------------- the re-run


@pytest.mark.parametrize(
    ("mode", "folder", "new_id", "benchmark"),
    [
        ("rec", "GW-REC-039", "GW-REC-043", "gdc test"),
        ("e2e", "GW-E2E-007", "GW-E2E-016", "gdc test"),
        ("oracle", "GW-E2E-008", "GW-E2E-017", "gdc test"),
    ],
)
def test_rows_and_metadata_say_phase_22_with_the_new_id(rerun, mode, folder, new_id, benchmark):
    data, log, _, _ = rerun
    assert _rerun(data, mode, "gdc_test", folder, new_id) == 0
    (row,) = _rows(log)
    assert row.endswith(f"| {benchmark} | {new_id} | glypharis ({mode}) | 22 | {REASON} |")
    meta = _meta(data, folder, "gdc_test")
    assert (meta["phase"], meta["experiment"], meta["folder_id"]) == (22, new_id, folder)
    assert meta["phase22"]["wrapper_commit"] == WRAPPER_COMMIT
    assert meta["phase22"]["wrapper_tag"] == "phase22-repro"
    # What the unchanged scorer reads from the metadata is still found.
    assert ps._find_pin(meta)["trees"] == "equal"
    assert ps._find_pin(meta)["tag"] == "phase14-eval"
    (entry,) = _manifest(data)["outputs"]
    assert entry["file"] == f"data/processed/phase14/{folder}/gdc_test.jsonl"
    assert entry["experiment"] == folder


def test_the_detached_worktree_and_the_phase22_log_are_set(rerun):
    data, log, _, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    assert ext_eval.PHASE14_BRANCH == "HEAD"
    assert ext_eval.ACCESS_LOG.is_absolute()
    assert log.resolve() == ext_eval.ACCESS_LOG


def test_validation_rehearsal_logs_nothing_and_carries_its_id(rerun):
    data, log, _, _ = rerun
    assert _rerun(data, "rec", "hiertext_val", "GW-REC-038", "GW-REC-042") == 0
    assert _rows(log) == []
    assert _meta(data, "GW-REC-038", "hiertext_val")["experiment"] == "GW-REC-042"


def test_nothing_is_written_outside_the_isolated_root(rerun, pinned, tmp_path):
    data, log, _, _ = rerun
    real_log = REPO / "research/test_access_log.md"

    def outside():
        return {
            k: v
            for k, v in _tree_digest(tmp_path).items()
            if not k.startswith("data/") and k != log.name
        }

    before, inside = outside(), _tree_digest(data)
    real_before = real_log.read_bytes()
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    assert outside() == before and real_log.read_bytes() == real_before
    added = set(_tree_digest(data)) - set(inside)
    assert added == {
        "processed/phase14/GW-REC-039/gdc_test.jsonl",
        "processed/phase14/GW-REC-039/gdc_test.meta.json",
        "manifests/processed/phase14_outputs.json",
    }
    assert pinned == [("phase22-repro", repro.WRAPPER_ROOT)]


def test_a_wrong_id_pair_is_refused(rerun):
    data, log, calls, _ = rerun
    with pytest.raises(SystemExit, match="not the registered re-run"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-042")
    assert _rows(log) == [] and calls["load"] == 0


def test_imports_from_outside_the_worktree_are_refused(rerun, tmp_path, monkeypatch):
    data, log, calls, _ = rerun
    monkeypatch.setattr(repro, "git_toplevel", lambda start: tmp_path.resolve())
    with pytest.raises(SystemExit, match="must come from the working tree"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0


def test_an_unpinned_wrapper_is_refused(rerun, monkeypatch):
    data, log, calls, _ = rerun

    def refuse(tag="phase14-eval", *, root=Path(".")):
        raise RuntimeError("the tag phase22-repro does not exist")

    monkeypatch.setattr(frozen, "pinned_code_check", refuse)
    with pytest.raises(SystemExit, match="not pinned"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0


def test_a_data_root_inside_a_worktree_is_refused(rerun):
    _, log, calls, _ = rerun
    with pytest.raises(SystemExit, match="inside a worktree"):
        _rerun(REPO / "data", "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0


def test_a_linked_phase14_folder_is_refused(rerun):
    data, log, calls, store = rerun
    (data / "processed/phase14").rmdir()
    _link_folder(data / "processed/phase14", store)
    with pytest.raises(SystemExit, match="is a link"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0


def _link_away(data, rel, store):
    """Replace the real folder ``rel`` by a link to a copy of it in ``store``."""
    folder = data / rel
    target = store / rel.replace("/", "_")
    if folder.exists():
        shutil.move(str(folder), str(target))
    _link_folder(folder, target)


@pytest.mark.parametrize(
    "rel", ["manifests", "manifests/processed", "processed/phase14/GW-REC-039"]
)
def test_linked_folders_the_run_writes_through_are_refused(rerun, rel):
    data, log, calls, store = rerun
    (data / "manifests/processed").mkdir(parents=True, exist_ok=True)
    _link_away(data, rel, store)
    with pytest.raises(SystemExit, match="is a link"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0


def test_a_linked_manifests_folder_is_refused_even_when_empty(rerun):
    data, _, _, store = rerun
    shutil.rmtree(data / "manifests/processed", ignore_errors=True)
    _link_away(data, "manifests", store)
    with pytest.raises(SystemExit, match="manifests under --data-root is a link"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")


def test_a_recorded_output_with_no_logged_attempt_is_refused(rerun):
    data, log, calls, _ = rerun
    out = _out(data, "GW-REC-039", "gdc_test")
    out.parent.mkdir(parents=True)
    out.write_text("", encoding="utf-8")
    manifest = data / ext_eval.OUTPUTS_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    entry = {"file": "data/processed/phase14/GW-REC-039/gdc_test.jsonl", "sha256": "0" * 64}
    manifest.write_text(json.dumps({"outputs": [entry]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="no logged attempt"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0


def test_a_crashed_run_whose_output_was_moved_cannot_start_again(rerun):
    data, log, calls, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    _crash(data, "GW-REC-039", "gdc_test", 1)
    out = _out(data, "GW-REC-039", "gdc_test")
    out.rename(out.with_suffix(".moved"))
    with pytest.raises(SystemExit, match="its output is gone"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert len(_rows(log)) == 1 and calls["load"] == 1


def test_a_hard_linked_outputs_manifest_is_refused(rerun, tmp_path):
    data, log, calls, _ = rerun
    manifest = data / ext_eval.OUTPUTS_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"outputs": []}', encoding="utf-8")
    os.link(manifest, tmp_path / "phase14_outputs_elsewhere.json")
    with pytest.raises(SystemExit, match="hard-linked"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0


def test_the_worktree_itself_is_refused_as_a_data_root(rerun):
    with pytest.raises(SystemExit, match="inside a worktree"):
        _rerun(REPO, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")


@pytest.mark.parametrize("rel", ["manifests", "processed/phase14"])
def test_a_root_without_its_folders_is_refused(rerun, rel):
    data, _, _, _ = rerun
    shutil.rmtree(data / rel)
    with pytest.raises(SystemExit, match="is missing under --data-root"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")


def test_a_glypharis_module_from_elsewhere_is_refused(rerun, tmp_path, monkeypatch):
    data, _, calls, _ = rerun
    module = types.ModuleType("glypharis.elsewhere")
    module.__file__ = str(tmp_path / "elsewhere.py")
    monkeypatch.setitem(sys.modules, "glypharis.elsewhere", module)
    with pytest.raises(SystemExit, match="must come from the working tree"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert calls["load"] == 0


def test_a_runner_helper_from_elsewhere_is_refused(rerun, tmp_path, monkeypatch):
    data, _, _, _ = rerun
    monkeypatch.setattr(sys.modules["run_baselines"], "__file__", str(tmp_path / "x.py"))
    with pytest.raises(SystemExit, match="must come from the working tree"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")


def test_a_module_that_is_not_loaded_is_refused():
    with pytest.raises(SystemExit, match="1 are not loaded"):
        repro.check_imports(REPO, ("a_module_nobody_imported",))


@pytest.mark.parametrize(
    ("eval_set", "folder", "new_id"),
    [("gdc_test", "GW-REC-039", "GW-REC-043"), ("hiertext_val", "GW-REC-038", "GW-REC-042")],
)
def test_the_runner_gets_the_registered_flags_and_nothing_else(
    rerun, monkeypatch, eval_set, folder, new_id
):
    data, log, _, _ = rerun
    seen = []
    monkeypatch.setattr(ext_eval, "ACCESS_LOG", Path("research/test_access_log.md"))
    monkeypatch.setattr(ext_eval, "main", lambda argv: seen.append(argv) or 0)
    assert _rerun(data, "rec", eval_set, folder, new_id) == 0
    assert seen == [
        [
            "rec",
            "--set",
            eval_set,
            "--experiment",
            folder,
            "--reason",
            REASON,
            "--data-root",
            str(data),
        ]
    ]
    assert log.resolve() == ext_eval.ACCESS_LOG


def test_only_the_expected_access_request_is_logged(rerun):
    _, log, _, _ = rerun
    repro.patch_ext_eval(ext_eval, "GW-REC-039", "GW-REC-043", {}, None)
    with pytest.raises(SystemExit, match="unexpected access request"):
        ext_eval.open_test_access("gdc", "test", "GW-REC-040", "x", 14, "why", log)
    with pytest.raises(SystemExit, match="unexpected access request"):
        ext_eval.open_test_access("gdc", "test", "GW-REC-039", "x", 5, "why", log)
    assert _rows(log) == []


def test_the_outputs_manifest_is_not_stamped(rerun):
    data, _, _, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    assert set(_manifest(data)) == {"note", "outputs"}


def test_a_refused_tag_worktree_stops_before_any_record_is_read(rerun, monkeypatch):
    data, log, _, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    _crash(data, "GW-REC-039", "gdc_test", 1)
    opened = []
    monkeypatch.setattr(ext_eval, "set_records", lambda *a, **k: opened.append(a))

    def refuse(tag=None, root=None):
        raise RuntimeError("the worktree is not at phase14-eval")

    monkeypatch.setattr(ext_eval, "pinned_code_check", refuse)
    with pytest.raises(SystemExit, match="pinned code check failed"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert opened == [] and len(_rows(log)) == 1


def test_a_wrong_branch_stops_before_any_record_is_read(rerun, monkeypatch):
    data, _, _, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    _crash(data, "GW-REC-039", "gdc_test", 1)
    opened = []
    monkeypatch.setattr(ext_eval, "set_records", lambda *a, **k: opened.append(a))

    def refuse(name, start=None):
        raise RuntimeError(f"not on {name}")

    monkeypatch.setattr(ext_eval, "require_branch", refuse)
    with pytest.raises(SystemExit, match="not on HEAD"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert opened == []


def test_a_set_outside_the_plan_is_refused_before_any_record(rerun):
    data, log, _, _ = rerun
    with pytest.raises(SystemExit, match="runs on"):
        _rerun(data, "oracle", "iiit5k_test", "GW-E2E-008", "GW-E2E-017")
    assert _rows(log) == []


def test_the_registered_order_is_phase_14s_and_covers_every_test_run():
    assert repro.TEST_ORDER[:7] == (
        ("rec", "hiertext_test"),
        ("rec", "gdc_test"),
        ("rec", "gdc_scan_test"),
        ("rec", "funsd_test"),
        ("rec", "iiit5k_test"),
        ("oracle", "gdc_test"),
        ("oracle", "gdc_scan_test"),
    )
    assert repro.TEST_ORDER[7:] == (
        ("e2e", "gdc_scan_test"),
        ("e2e", "funsd_test"),
        ("e2e", "gdc_test"),
        ("e2e", "omnidocbench_test"),
        ("e2e", "hiertext_test"),
    )
    planned = {
        (mode, eval_set)
        for mode, runs in ext_eval.PLAN.items()
        for sets in runs.values()
        for eval_set in sets
        if ext_eval.EVAL_SETS[eval_set]["split"] == "test"
    }
    assert set(repro.TEST_ORDER) == planned and len(repro.TEST_ORDER) == 12


def test_a_test_set_waits_for_the_one_before_it(rerun, monkeypatch):
    data, log, calls, _ = rerun
    order = (("rec", "gdc_test"), ("rec", "gdc_scan_test"))
    monkeypatch.setattr(repro, "TEST_ORDER", order)
    with pytest.raises(SystemExit, match="comes after rec gdc_test"):
        _rerun(data, "rec", "gdc_scan_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    assert _rerun(data, "rec", "gdc_scan_test", "GW-REC-039", "GW-REC-043") == 0
    assert len(_rows(log)) == 2


def test_the_order_does_not_bind_a_validation_run(rerun, monkeypatch):
    data, _, _, _ = rerun
    monkeypatch.setattr(repro, "TEST_ORDER", (("rec", "gdc_test"), ("rec", "hiertext_val")))
    assert _rerun(data, "rec", "hiertext_val", "GW-REC-038", "GW-REC-042") == 0


def test_a_manifest_entry_without_its_output_is_refused_before_the_run(rerun):
    data, log, calls, _ = rerun
    manifest = data / ext_eval.OUTPUTS_MANIFEST
    entry = {"file": "data/processed/phase14/GW-REC-039/gdc_test.jsonl", "sha256": "0" * 64}
    manifest.write_text(json.dumps({"outputs": [entry]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="records an output"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0


# -------------------------------------------------------------------- one attempt per set


def test_a_second_attempt_at_a_finished_set_is_refused(rerun):
    data, log, calls, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    with pytest.raises(SystemExit, match="once per set"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert len(_rows(log)) == 1 and calls["load"] == 1


def test_moving_the_output_aside_does_not_allow_a_second_attempt(rerun):
    data, log, calls, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    out = _out(data, "GW-REC-039", "gdc_test")
    out.rename(out.with_suffix(".moved"))
    _drop_manifest_entries(data)
    with pytest.raises(SystemExit, match="new registered ID"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert len(_rows(log)) == 1 and calls["load"] == 1


def test_a_crashed_run_resumes_and_says_so(rerun):
    data, log, _, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    total = _crash(data, "GW-REC-039", "gdc_test", 2)
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    rows = _rows(log)
    assert len(rows) == 2
    assert f"| 22 | {REASON} (resumed: 2 of {total} samples already predicted) |" in rows[1]
    out = _out(data, "GW-REC-039", "gdc_test")
    assert len(out.read_text(encoding="utf-8").splitlines()) == total
    meta = _meta(data, "GW-REC-039", "gdc_test")
    assert meta["phase"] == 22 and len(meta["runs"]) == 2


def test_a_run_that_died_before_its_first_record_resumes_as_a_resume(rerun):
    data, log, _, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    total = _crash(data, "GW-REC-039", "gdc_test", 0)
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    assert f"(resumed: 0 of {total} samples already predicted) |" in _rows(log)[1]


@pytest.mark.parametrize("keep", [0, 1])
@pytest.mark.parametrize("listed", [True, False])
def test_a_finished_set_cut_short_is_a_second_attempt_not_a_resume(rerun, keep, listed):
    data, log, calls, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    assert _meta(data, "GW-REC-039", "gdc_test")["runs"][-1]["status"] == "done"
    _cut_output(data, "GW-REC-039", "gdc_test", keep)
    if not listed:
        _drop_manifest_entries(data)
    with pytest.raises(SystemExit, match="not a crash resume"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert len(_rows(log)) == 1 and calls["load"] == 1


def test_a_running_status_cannot_reopen_a_recorded_set(rerun):
    data, log, calls, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    _cut_output(data, "GW-REC-039", "gdc_test", 1)
    _set_last_status(data, "GW-REC-039", "gdc_test", "running")
    with pytest.raises(SystemExit, match="not a crash resume"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert len(_rows(log)) == 1 and calls["load"] == 1


@pytest.mark.parametrize("keep", [1, None])
def test_an_attempt_without_its_metadata_is_refused(rerun, keep):
    data, log, calls, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    _drop_manifest_entries(data)
    if keep is not None:
        _cut_output(data, "GW-REC-039", "gdc_test", keep)
    _out(data, "GW-REC-039", "gdc_test").with_suffix(".meta.json").unlink()
    with pytest.raises(SystemExit, match="run metadata is missing"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert len(_rows(log)) == 1 and calls["load"] == 1


def test_rows_of_other_sets_and_ids_are_not_attempts(rerun):
    data, log, calls, _ = rerun
    rows = (
        "| 2026-11-19 | hiertext test | GW-REC-043 | glypharis (rec) | 22 | another set |\n"
        "| 2026-11-19 | gdc test | GW-REC-039 | glypharis (rec) | 14 | the folder ID |\n"
    )
    log.write_text(log.read_text(encoding="utf-8") + rows, encoding="utf-8")
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    assert calls["load"] == 1


def test_a_complete_unrecorded_output_is_closed_without_a_row_or_a_model(rerun):
    data, log, calls, _ = rerun
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    _drop_manifest_entries(data)
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    assert len(_rows(log)) == 1 and calls["load"] == 1
    assert len(_manifest(data)["outputs"]) == 1


def test_an_output_with_no_logged_attempt_is_refused(rerun):
    data, log, calls, _ = rerun
    out = _out(data, "GW-REC-039", "gdc_test")
    out.parent.mkdir(parents=True)
    out.write_text('{"key": "gdc0/l0"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="no logged attempt"):
        _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043")
    assert _rows(log) == [] and calls["load"] == 0


def test_scoring_rows_are_not_prediction_attempts(rerun):
    data, log, _, _ = rerun
    row = "| 2026-11-19 | gdc test | GW-REC-043 | glypharis (rec), scoring | 22 | Pass B |\n"
    log.write_text(log.read_text(encoding="utf-8") + row, encoding="utf-8")
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0


def test_a_gdc_scan_marker_reaches_no_row_metadata_manifest_or_console(rerun, capsys):
    data, log, _, _ = rerun
    assert _rerun(data, "rec", "gdc_scan_test", "GW-REC-039", "GW-REC-043") == 0
    assert _rerun(data, "oracle", "gdc_scan_test", "GW-E2E-008", "GW-E2E-017") == 0
    captured = capsys.readouterr()
    assert RERUN_MARKER not in captured.out + captured.err
    assert RERUN_MARKER not in log.read_text(encoding="utf-8")
    assert RERUN_MARKER not in (data / ext_eval.OUTPUTS_MANIFEST).read_text(encoding="utf-8")
    for folder in ("GW-REC-039", "GW-E2E-008"):
        meta = _out(data, folder, "gdc_scan_test").with_suffix(".meta.json")
        assert RERUN_MARKER not in meta.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------ the re-score


@pytest.fixture
def scored(score_layout, pinned, monkeypatch):  # noqa: F811
    layout = score_layout
    monkeypatch.chdir(REPO)
    for name in ("Scorer", "PHASE14_BRANCH", "EXPERIMENT", "REASON"):
        monkeypatch.setattr(ps, name, getattr(ps, name))
    monkeypatch.setattr(eb, "open_test_access", eb.open_test_access)
    assert ps.main(["hash-phase5", "--data-root", str(layout["data"])]) == 0
    layout["record"] = layout["args"][layout["args"].index("--record") + 1]
    return layout


def _rescore(layout, command, experiment=None, *extra, out=None):
    argv = ["rescore", command, "--data-root", str(layout["data"])]
    argv += ["--eff-dir", str(layout["eff"]), "--record", layout["record"]]
    if experiment:
        argv += ["--experiment", experiment]
    if out is not None:
        argv += ["--out", str(out)]
    return repro.main([*argv, *extra])


def _final(layout, name):
    return layout["tmp"] / "reports/final_validation" / name


def _phase22_rows(log, reason):
    return [ln for ln in _rows(log) if f"| 22 | {reason}" in ln]


def test_check_logs_nothing_and_needs_the_pin(scored, pinned):
    before = scored["log"].read_bytes()
    assert _rescore(scored, "check") == 0
    assert scored["log"].read_bytes() == before
    assert pinned == [("phase22-repro", REPO.resolve())]
    assert ps.EXPERIMENT == repro.PASS_A and ps.PHASE == 14


def test_check_refuses_a_missing_phase5_pack(scored):
    baseline = scored["data"] / "processed/baselines/GW-BASE-002"
    (baseline / "ppocr_mobile_rec_glyphsynth_val.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"processed/glyphsynth/lines_val"):
        _rescore(scored, "check")


def test_rescore_refuses_an_unpinned_tree(scored, monkeypatch):
    def refuse(tag="phase14-eval", *, root=Path(".")):
        raise RuntimeError("uncommitted changes")

    monkeypatch.setattr(frozen, "pinned_code_check", refuse)
    out = _final(scored, "GW-E2E-015")
    with pytest.raises(SystemExit, match="not pinned"):
        _rescore(scored, "score", "GW-E2E-015", out=out)
    assert _rows(scored["log"]) == [] and not out.exists()


def test_rescore_refuses_linked_scorer_folders(scored):
    target = scored["tmp"] / "elsewhere"
    _link_folder(scored["data"] / "processed/phase14/handoff", target)
    with pytest.raises(SystemExit, match="is a link"):
        _rescore(scored, "check")


@pytest.mark.parametrize("rel", ["GW-E2E-009", "GW-E2E-009/official"])
def test_rescore_refuses_linked_pass_folders(scored, rel):
    folder = scored["data"] / "processed/phase14" / rel
    folder.parent.mkdir(parents=True, exist_ok=True)
    _link_folder(folder, scored["tmp"] / "elsewhere")
    with pytest.raises(SystemExit, match="is a link"):
        _rescore(scored, "check")


def test_rescore_refuses_a_linked_manifest_folder(scored):
    _link_away(scored["data"], "manifests/processed", scored["tmp"])
    with pytest.raises(SystemExit, match="is a link"):
        _rescore(scored, "check")


def test_rescore_refuses_imports_from_elsewhere(scored, tmp_path, monkeypatch):
    monkeypatch.setattr(repro, "git_toplevel", lambda start: tmp_path.resolve())
    with pytest.raises(SystemExit, match="must come from the working tree"):
        _rescore(scored, "check")


def test_the_scorer_uses_the_phase_22_branch(scored):
    assert _rescore(scored, "check") == 0
    assert ps.PHASE14_BRANCH == "phase/22-final-validation"


def test_every_phase_5_file_is_required(scored, monkeypatch):
    spec = ps.EVAL_SETS["hiertext_val"]
    jobs = [
        ps.Job(Path("r.jsonl"), "GW-BASE-001", "x", "rec", "hiertext_val", 5),
        ps.Job(Path("e.jsonl"), "GW-BASE-001", "x", "e2e", "hiertext_val", 5),
    ]
    monkeypatch.setattr(ps, "phase5_jobs", lambda data: jobs)
    assert repro.phase5_inputs(ps, scored["data"]) == sorted(
        [
            f"{spec['pack']}.bin",
            f"{spec['pack']}.index.jsonl",
            f"{spec['pack']}.meta.json",
            spec["pages"],
        ]
    )


def test_rescore_refuses_listed_handoff_files(scored):
    path = scored["data"] / ps.OUTPUTS
    doc = json.loads(path.read_text(encoding="utf-8"))
    entry = {"file": "data/processed/phase14/handoff/x.jsonl", "sha256": "0", "lines": 1}
    doc["outputs"].append(entry)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="hand-off"):
        _rescore(scored, "check")


def test_pass_ids_are_tied_to_the_id_map(scored):
    with pytest.raises(SystemExit, match="this pass is GW-E2E-018"):
        _rescore(scored, "check", "GW-E2E-015", "--id-map", "rerun")
    with pytest.raises(SystemExit, match="this pass is GW-E2E-015"):
        _rescore(scored, "check", "GW-E2E-018")
    with pytest.raises(SystemExit, match="reports/final_validation/GW-E2E-015"):
        _rescore(scored, "score", "GW-E2E-015", out=scored["tmp"] / "reports/external")


def test_pass_a_logs_phase_22_under_the_producing_ids(scored):
    out = _final(scored, "GW-E2E-015")
    assert _rescore(scored, "score", "GW-E2E-015", out=out) == 0
    reason = "Phase 22 v1 re-score (GW-E2E-015)"
    rows = _phase22_rows(scored["log"], reason)
    assert len(rows) == 5 + 6 + 12 and len(rows) == len(_rows(scored["log"]))
    cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
    assert all(c[4] == "22" for c in cells)
    ids = {c[2] for c in cells}
    assert {"GW-REC-039", "GW-E2E-007", "GW-E2E-008"} <= ids
    assert not ids & set(repro.PASS_B_IDS.values())
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["experiment"] == "GW-E2E-015" and "id_map" not in res
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert repro.ID_MAP_HEADER not in readme


def test_pass_b_maps_ids_and_marks_every_report(scored):
    stamp_rerun_root(scored["data"])
    out = _final(scored, "GW-E2E-018")
    assert _rescore(scored, "score", "GW-E2E-018", "--id-map", "rerun", out=out) == 0
    rows = _phase22_rows(scored["log"], "Phase 22 v1 re-run scoring (GW-E2E-018)")
    assert len(rows) == 5 + 6 + 12 and len(rows) == len(_rows(scored["log"]))
    ids = [r.split("|")[3].strip() for r in rows]
    assert not set(ids) & set(repro.PASS_B_IDS)
    assert ids.count("GW-REC-043") == 5 and ids.count("GW-E2E-016") == 5
    assert ids.count("GW-E2E-017") == 2
    assert any(i.startswith("GW-BASE-") for i in ids)
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["id_map"] == {
        "GW-REC-038": "GW-REC-042",
        "GW-E2E-006": "GW-E2E-014",
        "GW-REC-039": "GW-REC-043",
        "GW-E2E-007": "GW-E2E-016",
        "GW-E2E-008": "GW-E2E-017",
    }
    assert res["id_map_note"] == repro.ID_MAP_HEADER
    assert res["experiment"] == "GW-E2E-018"
    # The unchanged scorer still labels the entries with the folder IDs; the map explains them.
    assert any(e["experiment"] == "GW-REC-039" for e in res["predictions"])
    reports = sorted(out.glob("*.md"))
    assert reports
    for md in reports:
        assert md.read_text(encoding="utf-8").startswith(repro.ID_MAP_HEADER), md.name
    # Repeating the annotation changes nothing.
    before = _tree_digest(out)
    assert _rescore(scored, "annotate", "GW-E2E-018", "--id-map", "rerun", out=out) == 0
    assert _tree_digest(out) == before


def test_pass_b_reports_hold_no_gdc_scan_marker(scored, capsys):
    stamp_rerun_root(scored["data"])
    out = _final(scored, "GW-E2E-018")
    assert _rescore(scored, "score", "GW-E2E-018", "--id-map", "rerun", out=out) == 0
    captured = capsys.readouterr()
    assert SCORE_MARKER not in captured.out + captured.err
    assert SCORE_MARKER not in scored["log"].read_text(encoding="utf-8")
    for path in out.rglob("*"):
        if path.is_file():
            assert SCORE_MARKER.encode() not in path.read_bytes(), path.name


def test_the_report_header_says_what_the_values_are():
    assert repro.ID_MAP_HEADER == (
        "> **Re-run values, not Phase 14's.** The scorer labels them with the folder IDs: "
        "GW-REC-038 = GW-REC-042, GW-E2E-006 = GW-E2E-014, GW-REC-039 = GW-REC-043, "
        "GW-E2E-007 = GW-E2E-016, GW-E2E-008 = GW-E2E-017."
    )


def test_pass_b_refuses_files_the_rerun_did_not_stamp(scored):
    out = _final(scored, "GW-E2E-018")
    with pytest.raises(SystemExit, match="was not written by"):
        _rescore(scored, "score", "GW-E2E-018", "--id-map", "rerun", out=out)
    assert _rows(scored["log"]) == [] and not out.exists()


@pytest.mark.parametrize(
    "change",
    [
        {"phase": 14},
        {"experiment": "GW-REC-039"},
        {"folder_id": "GW-REC-038"},
        {"phase22": {"experiment": "GW-REC-043", "folder_id": "GW-REC-039"}},
        {
            "phase22": {
                "experiment": "GW-REC-042",
                "folder_id": "GW-REC-039",
                "wrapper_tag": repro.WRAPPER_TAG,
            }
        },
        {"pinned": {"tag": "phase14-eval", "tag_commit": "1", "trees": "differ"}},
    ],
)
def test_pass_b_refuses_a_wrong_stamp(scored, change):
    data = scored["data"]
    stamp_rerun_root(data)
    path = data / "processed/phase14/GW-REC-039/gdc_test.meta.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**meta, **change}), encoding="utf-8")
    _rerecord(data)
    with pytest.raises(SystemExit, match="was not written by"):
        _rescore(scored, "check", "GW-E2E-018", "--id-map", "rerun")


def test_pass_a_refuses_the_rerun_files(scored):
    stamp_rerun_root(scored["data"])
    out = _final(scored, "GW-E2E-015")
    with pytest.raises(SystemExit, match="is not Phase 14's"):
        _rescore(scored, "score", "GW-E2E-015", out=out)
    assert _rows(scored["log"]) == []


def test_a_stopped_pass_cannot_resume_as_the_other_pass(scored):
    stamp_rerun_root(scored["data"])
    record = {"experiment": "GW-E2E-015", "status": "started", "started_utc": "x"}
    ps.write_pass(scored["data"], {**record, "resumed_utc": []})
    out = _final(scored, "GW-E2E-018")
    with pytest.raises(SystemExit, match="this root's pass is GW-E2E-015"):
        _rescore(scored, "score", "GW-E2E-018", "--id-map", "rerun", "--resume", out=out)
    assert _rows(scored["log"]) == []


@pytest.mark.parametrize("id_map", [False, True])
def test_a_stopped_pass_resumes_as_itself(scored, id_map):
    data = scored["data"]
    name, extra = ("GW-E2E-018", ("--id-map", "rerun")) if id_map else ("GW-E2E-015", ())
    if id_map:
        stamp_rerun_root(data)
    out = _final(scored, name)
    assert _rescore(scored, "score", name, *extra, out=out) == 0
    (out / "results.json").unlink()
    ps.write_pass(data, {**ps.read_pass(data), "status": "started"})
    listed = json.loads((data / ps.OUTPUTS).read_text(encoding="utf-8"))["outputs"]
    assert any("/handoff/" in entry["file"] for entry in listed)
    first = len(_rows(scored["log"]))
    assert _rescore(scored, "score", name, *extra, "--resume", out=out) == 0
    resumed = _rows(scored["log"])[first:]
    assert len(resumed) == first and all(r.endswith("(resumed) |") for r in resumed)


@pytest.mark.parametrize(
    "out",
    [
        "reports/final_validation/../external",
        "reports/external/final_validation",
        "reports/final_validation/other",
        "reports/final_validation/GW-E2E-015",
    ],
)
def test_annotate_writes_only_to_its_own_report(tmp_path, monkeypatch, out):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / out
    target.mkdir(parents=True)
    (target / "results.json").write_text('{"experiment": "GW-E2E-018"}\n', encoding="utf-8")
    argv = ["rescore", "annotate", "--id-map", "rerun", "--experiment", "GW-E2E-018"]
    with pytest.raises(SystemExit, match="--out must be"):
        repro.main([*argv, "--out", out])
    assert "id_map" not in (target / "results.json").read_text(encoding="utf-8")


def test_annotate_refuses_another_pass_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "reports/final_validation/GW-E2E-018"
    target.mkdir(parents=True)
    (target / "results.json").write_text('{"experiment": "GW-E2E-015"}\n', encoding="utf-8")
    argv = ["rescore", "annotate", "--id-map", "rerun", "--experiment", "GW-E2E-018"]
    with pytest.raises(SystemExit, match="is not GW-E2E-018's report"):
        repro.main([*argv, "--out", "reports/final_validation/GW-E2E-018"])


def test_score_cannot_write_under_reports_external(scored):
    out = scored["tmp"] / "reports/external/final_validation"
    with pytest.raises(SystemExit, match="--out must be"):
        _rescore(scored, "score", "GW-E2E-015", out=out)
    assert not out.exists() and _rows(scored["log"]) == []


def test_rescore_arguments_are_required(tmp_path):
    with pytest.raises(SystemExit, match="score needs --experiment and --out"):
        repro.main(["rescore", "score", "--data-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="check needs --data-root"):
        repro.main(["rescore", "check"])
    out = tmp_path / "reports/final_validation/GW-E2E-015"
    with pytest.raises(SystemExit, match="annotate is for Pass B"):
        repro.main(["rescore", "annotate", "--experiment", "GW-E2E-015", "--out", str(out)])


def test_the_early_request_check_matches_the_runner(rerun, monkeypatch):
    data, _, _, _ = rerun
    seen = []
    real = ext_eval.check_request
    monkeypatch.setattr(ext_eval, "check_request", lambda a: seen.append(a) or real(a))
    assert _rerun(data, "rec", "gdc_test", "GW-REC-039", "GW-REC-043") == 0
    first = seen[0]
    assert isinstance(first, argparse.Namespace)
    assert (first.experiment, first.device, first.allow_unpinned) == ("GW-REC-039", "cuda", False)
