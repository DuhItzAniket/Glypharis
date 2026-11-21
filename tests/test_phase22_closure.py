"""Tests for scripts/phase22_closure.py: each closure check passes on unchanged records and fails,
without stopping, on a changed one."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import phase14_score as ps
import phase22_closure as cl
import pytest

ROW = (
    "| 2026-11-21 | gdc test | GW-REC-045 | glypharis (rec) | 22 | Phase 22 candidate test run |\n"
)
HEAD = (
    "# Test-Set Access Log\n\n"
    "| Date | Split / benchmark | Experiment ID | Checkpoint | Phase | Reason |\n"
    "|---|---|---|---|---|---|\n"
    "| 2026-10-21 | gdc test | GW-BASE-002 | ppocr_mobile (rec) | 5 | Phase 5 baseline |\n"
)


def test_compare_files_names_every_difference() -> None:
    before = {"a": {"sha256": "1", "bytes": 1}, "b": {"sha256": "2", "bytes": 2}}
    assert cl.compare_files(before, dict(before))["passed"]
    now = {"a": {"sha256": "1", "bytes": 1}, "b": {"sha256": "9", "bytes": 2}, "c": {}}
    got = cl.compare_files(before, now)
    assert (got["changed"], got["added"], got["missing"], got["passed"]) == (
        ["b"],
        ["c"],
        [],
        False,
    )
    assert cl.compare_files(before, {"a": before["a"]})["missing"] == ["b"]


def test_result_folder_must_equal_every_list(tmp_path: Path) -> None:
    folder = tmp_path / "result"
    folder.mkdir()
    for name in ("x.json", "y.json"):
        (folder / name).write_text("{}", encoding="utf-8")
    same = tmp_path / "a.txt"
    same.write_text("y.json\nx.json\n", encoding="utf-8")
    assert cl.check_result_folder(folder, [same, same])["passed"]
    other = tmp_path / "b.txt"
    other.write_text("x.json\n", encoding="utf-8")
    got = cl.check_result_folder(folder, [same, other])
    assert got["each_list_equal"] == [True, False] and not got["passed"]
    assert not cl.check_result_folder(folder, [])["passed"]
    assert not cl.check_result_folder(tmp_path / "gone", [same])["passed"]


def test_access_log_prefix_and_counts() -> None:
    start = HEAD.encode()
    head = (
        HEAD + ROW + ROW.replace("run |", "run (resumed: 3 of 9 samples already predicted) |")
    ).encode()
    got = cl.check_access_log(start, head, head)
    assert got["passed"] and got["rows_at_start"] == 1 and got["rows_added"] == 2
    assert got["rows_added_resumed"] == 1
    assert sum(r["rows"] for r in got["rows_added_by_phase_and_reason"]) == 2
    assert not cl.check_access_log(start, head, start)["passed"]  # working copy differs
    edited = head.replace(b"Phase 5 baseline", b"Phase 5 baseline, edited")
    got = cl.check_access_log(start, edited, edited)
    assert not got["prefix"] and not got["passed"] and got["rows_added"] == 0


def _manifest(data: Path, entries: list[dict[str, object]]) -> None:
    path = data / ps.OUTPUTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"note": "", "outputs": entries}), encoding="utf-8")


def test_phase14_outputs_reports_a_changed_or_missing_file(tmp_path: Path) -> None:
    data = tmp_path / "data"
    run = data / "processed/phase14/GW-REC-039"
    run.mkdir(parents=True)
    out = run / "gdc_test.jsonl"
    out.write_text('{"k": 1}\n{"k": 2}\n', encoding="utf-8")
    _manifest(data, [ps.file_entry(out, data)])
    got = cl.check_phase14_outputs(data)
    assert got["passed"] and got["checked"] == 1
    out.write_text('{"k": 1}\n', encoding="utf-8")
    got = cl.check_phase14_outputs(data)
    assert not got["passed"] and got["problem"]
    out.unlink()
    got = cl.check_phase14_outputs(data)
    assert not got["passed"] and got["missing"] == ["processed/phase14/GW-REC-039/gdc_test.jsonl"]


def _git(repo: Path, *args: str) -> None:
    ident = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(["git", *ident, *args], cwd=repo, check=True, capture_output=True)


@pytest.mark.parametrize(
    ("path", "flagged"),
    [("glypharis/eval/m.py", True), ("scripts/phase14_score.py", True), ("docs/x.md", False)],
)
def test_protected_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, flagged: bool
) -> None:
    repo = tmp_path / "repo"
    target = repo / path
    target.parent.mkdir(parents=True)
    target.write_text("a\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "start")
    _git(repo, "tag", "start")
    target.write_text("b\n", encoding="utf-8")
    _git(repo, "commit", "-q", "-am", "change")
    monkeypatch.chdir(repo)
    got = cl.check_protected("start")
    assert got["passed"] is not flagged
    assert got["changed_files"] == ([path] if flagged else [])


def test_render_marks_a_failed_check() -> None:
    log = cl.check_access_log(HEAD.encode(), (HEAD + ROW).encode(), (HEAD + ROW).encode())
    checks = {
        "v1_files": {
            "now": 2,
            "before": 2,
            "missing": [],
            "changed": ["x"],
            "added": [],
            "passed": False,
        },
        "omnidocbench_result": {
            "files_now": 1,
            "each_list_equal": [True],
            "lists_before": 1,
            "passed": True,
        },
        "phase14_outputs": {
            "checked": 1,
            "entries": 1,
            "meta_hashes": 0,
            "missing": [],
            "passed": True,
        },
        "access_log": log,
        "protected_paths": {
            "start": "phase-21",
            "exit_code": 0,
            "changed_files": [],
            "paths": ["configs"],
            "passed": True,
        },
    }
    text = cl.render({"head": "abc", "date": "2026-11-21", "passed": False, "checks": checks})
    assert "**All checks: FAILED.**" in text
    assert "changed 1" in text and "| 22 | Phase 22 candidate test run | 1 |" in text


def test_phase14_outputs_fail_without_a_manifest_or_entries(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    assert not cl.check_phase14_outputs(data)["passed"]
    _manifest(data, [])
    got = cl.check_phase14_outputs(data)
    assert not got["passed"] and got["entries"] == 0
