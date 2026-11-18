from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import experiment_summary as es
import pytest

REPO = Path(__file__).resolve().parents[1]

ENTRY = """### <a id="{anchor}"></a>{id} — {title}

- **Date allocated / run:** {date} / {run}
- **Phase:** {phase}
{extra}- **Hypothesis (written before the run):** none
- **Result:** {result}
  continued on a second line
- **Interpretation:** fine

"""

REGISTRY_HEAD = """# Experiment Registry

| ID | Date allocated | Phase | Question / hypothesis | Status | Key result | Report |
|---|---|---|---|---|---|---|
| [GW-BASE-001](#gw-base-001) | 2026-01-01 | 05 | Base question | done | 12.34% CER; 1.5 s per page; 1,200 lines; F1@0.5 hash `a1b2c3` | `reports/base/` |
| [GW-REC-001](#gw-rec-001) | 2026-01-02 | 06 | Rec question | done | val CER 7.53% [7.38, 7.69]; arm +250K | `reports/training/GW-REC-001/` |
| [GW-REC-002](#gw-rec-002) | 2026-01-03 | 17 (v1) | Re-run | done (amended) | val CER 6.30% | `reports/training/GW-REC-002/`, `reports/shared.md` |
| [GW-REC-003](#gw-rec-003) | 2026-01-04 | 17 | Abandoned idea | abandoned | never started | — |
| [GW-REC-004](#gw-rec-004) | 2026-01-05 | 06 | Conditional run | not run | condition not met | — |

"""  # noqa: E501

TEMPLATE = """---

## Entry template

```markdown
### <a id="gw-xxx-nnn"></a>GW-XXX-NNN — <short title>

- **Date allocated / run:** YYYY-MM-DD / YYYY-MM-DD
- **Result:**
```
"""


def entry(experiment: str, date: str, phase: str, run: str = "2026-02-01", extra: str = "") -> str:
    return ENTRY.format(
        anchor=experiment.lower(),
        id=experiment,
        title="title",
        date=date,
        run=run,
        phase=phase,
        extra=extra,
        result="recorded",
    )


def registry() -> str:
    return (
        REGISTRY_HEAD
        + entry("GW-BASE-001", "2026-01-01", "05")
        + entry("GW-REC-001", "2026-01-02", "06")
        + entry(
            "GW-REC-002",
            "2026-01-03",
            "17 (v1 re-run)",
            extra="- **Superseded by:** GW-REC-001, for the same pages only.\n",
        )
        + entry("GW-REC-003", "2026-01-04", "17", run="—")
        + entry("GW-REC-004", "2026-01-05", "06", run="—")
        + TEMPLATE
    )


def ledger() -> dict[str, object]:
    row = {"phase": "06", "jobs": 2, "failed_jobs": 1}
    return {
        "jobs": [],
        "experiments": {
            "GW-BASE-001": {**row, "device": "cpu", "wall_h": 1.0, "train_h": None},
            "GW-REC-001": {**row, "device": "cpu", "wall_h": 0.5, "train_h": None},
            "GW-REC-001 (GPU jobs)": {**row, "device": "gpu", "wall_h": 2.0, "train_h": 1.5},
            "Phase 16 scoring": {**row, "device": "cpu", "wall_h": 0.25, "train_h": None},
        },
        "totals": {"gpu_h": 2.0, "gpu_failed_h": 0.0, "cpu_h": 1.75, "cpu_failed_h": 0.0},
    }


FILES = {
    "research/experiments.md": registry(),
    "research/ablations.md": "# Ablations\n\n| A | B |\n|---|---|\n| x | GW-REC-001 |\n\n"
    "Prose may cite GW-REC-003.\n",
    "research/failures.md": "# Failures\n\n### YYYY-MM-DD — <experiment ID> — <summary>\n\n"
    "### 2026-01-04 — GW-REC-003 / GW-BASE-001 — it stopped, unlike GW-REC-002\n",
    "research/findings.md": "| F1 | GW-REC-001 |\n",
    "research/hypotheses.md": "H1 rests on GW-REC-002.\n",
    "research/test_access_log.md": "| Date | Split | Experiment ID | Checkpoint | Phase | Why |\n"
    "|---|---|---|---|---|---|\n"
    "| 2026-01-01 | gdc test | GW-BASE-001 | x | 5 | baseline |\n"
    "| 2026-01-02 | gdc test | GW-BASE-001 | x | 14 | external |\n",
    "reports/compute/ledger.json": json.dumps(ledger()),
    "reports/compute/ledger.md": "# Ledger\n\nPhases 9, 10, 14 and 15\nare not counted either.\n",
    "reports/base/result.json": json.dumps(
        {"experiment": "GW-BASE-001", "cer": 0.12341, "page_ms": 1500.0, "lines": 1200}
    ),
    "reports/training/GW-REC-001/report.md": "# GW-REC-001\n\nCER 7.53 [7.38, 7.69]; +250K lines\n",
    "reports/training/GW-REC-002/report.md": "# GW-REC-002\n\nCER 6.301\n",
    "reports/shared.md": "Nothing about this run.\n",
}


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    for path, text in FILES.items():
        full = tmp_path / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8", newline="\n")
    return tmp_path


def edit(root: Path, path: str, old: str, new: str) -> None:
    full = root / path
    text = full.read_text(encoding="utf-8")
    assert old in text, old
    full.write_text(text.replace(old, new), encoding="utf-8", newline="\n")


def run(root: Path, *args: str) -> int:
    return es.main(["--root", str(root), *args])


def outputs_exist(root: Path) -> list[str]:
    return [path for path in es.OUTPUTS if (root / path).exists()]


REG = "research/experiments.md"
REC2_ROW = "| [GW-REC-002](#gw-rec-002) | 2026-01-03 | 17 (v1) | Re-run | done (amended) |"

BROKEN = [
    # R1: rows and entries match, once each, and the link fits the anchor.
    ("R1", REG, '<a id="gw-rec-004"></a>GW-REC-004', '<a id="gw-rec-005"></a>GW-REC-005'),
    ("R1", REG, "[GW-REC-002](#gw-rec-002)", "[GW-REC-002](#gw-rec-001)"),
    ("R1", REG, "| [GW-REC-004](#gw-rec-004) | 2026-01-05 | 06 | Conditional run |",
     "| [GW-REC-004](#gw-rec-004) | 2026-01-05 | 06 |"),
    ("R1", REG, '### <a id="gw-rec-004"></a>GW-REC-004 — title', "### GW-REC-004 — title"),
    ("R1", REG, "| [GW-REC-004](#gw-rec-004)",
     "| [GW-REC-001](#gw-rec-001) | 2026-01-02 | 06 | Copy | done | x | `reports/base/` |\n"
     "| [GW-REC-004](#gw-rec-004)"),
    # R2: known prefixes, no gaps.
    ("R2", REG, "GW-REC-004", "GW-REC-009"),
    ("R2", REG, "GW-BASE-001", "GW-NEW-001"),
    # R3: the status list.
    ("R3", REG, "| Conditional run | not run |", "| Conditional run | skipped |"),
    # R4: date and phase agree.
    ("R4", REG, "run:** 2026-01-02 /", "run:** 2026-01-09 /"),
    ("R4", REG, "- **Phase:** 17 (v1 re-run)", "- **Phase:** 18 (v1 re-run)"),
    # R5: what each status needs.
    ("R5", REG, "| val CER 6.30% |", "| — |"),
    ("R5", REG, "- **Date allocated / run:** 2026-01-02 / 2026-02-01",
     "- **Date allocated / run:** 2026-01-02 / —"),
    ("R5", "research/failures.md", "GW-REC-003 / GW-BASE-001", "GW-BASE-001"),
    ("R5", REG, "| condition not met | — |", "| condition not met | `reports/base/` |"),
    ("R5", REG, "| Rec question | done | val CER 7.53% [7.38, 7.69]; arm +250K |",
     "| Rec question | planned | — |"),
    ("R5", REG, "| Rec question | done | val CER 7.53% [7.38, 7.69]; arm +250K |",
     "| Rec question | running | 7.53 |"),
    # R6: report paths.
    ("R6", REG, "`reports/base/`", "`reports/missing/`"),
    ("R6", REG, "`reports/base/`", "`research/`"),
    ("R6", REG, "`reports/base/`", "—"),
    ("R6", REG, "`reports/base/`", "no path here"),
    ("R6", "reports/base/result.json", '"GW-BASE-001"', '"another run"'),
    # R7: the numbers are in the listed files.
    ("R7", REG, "val CER 6.30%", "val CER 6.40%"),
    ("R7", REG, "1,200 lines", "1,300 lines"),
    ("R7", REG, "arm +250K", "arm +300K"),
    # R8: citations resolve.
    ("R8", "research/findings.md", "GW-REC-001", "GW-REC-099"),
    ("R8", "research/ablations.md", "| x | GW-REC-001 |", "| x | GW-REC-003 |"),
    ("R8", REG, "- **Superseded by:** GW-REC-001", "- **Superseded by:** GW-REC-004"),
    ("R8", REG, "- **Superseded by:** GW-REC-001", "- **Superseded by:** a later run"),
    ("R8", "reports/compute/ledger.json", '"Phase 16 scoring"', '"Some other job"'),
    ("R8", "reports/compute/ledger.json", '"GW-BASE-001"', '"GW-BASE-009"'),
    # R9: the ledger adds up and still says what it leaves out.
    ("R9", "reports/compute/ledger.json", '"gpu_h": 2.0', '"gpu_h": 3.0'),
    ("R9", "reports/compute/ledger.md", "Phases 9, 10, 14 and 15", "Phases 9 and 10"),
    # R10: the test log.
    ("R10", "research/test_access_log.md", "| x | 14 |", "| x | 7 |"),
    ("R10", "research/test_access_log.md", "| gdc test | GW-BASE-001 | x | 5 |",
     "| gdc test | someone | x | 5 |"),
    # R11: nothing personal reaches the output.
    ("R11", REG, "| Rec question |", "| Rec question C:\\Users\\someone |"),
    ("R11", REG, "| Rec question |", "| Rec question /home/someone |"),
    ("R11", REG, "| Rec question |", "| Rec question someone@example.com |"),
]  # fmt: skip


@pytest.mark.parametrize(("rule", "path", "old", "new"), BROKEN)
def test_each_rule_fails_and_writes_nothing(repo, capsys, rule, path, old, new):
    edit(repo, path, old, new)
    assert run(repo) == 1
    err = capsys.readouterr().err
    assert any(line.startswith(f"{rule} ") for line in err.splitlines()), err
    assert "nothing written" in err
    assert outputs_exist(repo) == []


def test_a_failure_keeps_existing_outputs_unchanged(repo):
    assert run(repo) == 0
    before = {path: (repo / path).read_bytes() for path in es.OUTPUTS}
    edit(repo, REG, "val CER 6.30%", "val CER 6.40%")
    assert run(repo) == 1
    assert {path: (repo / path).read_bytes() for path in es.OUTPUTS} == before


def test_planned_row_with_a_report_folder_fails(repo, capsys):
    edit(repo, REG, "| Rec question | done | val CER 7.53% [7.38, 7.69]; arm +250K |",
         "| Rec question | planned | — |")  # fmt: skip
    assert run(repo) == 1
    assert "reports exist" in capsys.readouterr().err


def test_planned_row_without_a_report_passes(repo):
    edit(repo, REG, "| Conditional run | not run | condition not met | — |",
         "| Conditional run | planned | — | `reports/training/GW-REC-004/` |")  # fmt: skip
    assert run(repo) == 0


def test_entry_without_a_result_fails(repo, capsys):
    text = (repo / REG).read_text(encoding="utf-8")
    start = text.index('<a id="gw-rec-004">')
    fixed = text[:start] + text[start:].replace(
        "- **Result:** recorded\n  continued on a second line\n", "- **Result:** —\n", 1
    )
    (repo / REG).write_text(fixed, encoding="utf-8", newline="\n")
    assert run(repo) == 1
    assert "GW-REC-004 (not run): the entry has no Result" in capsys.readouterr().err


def test_valid_registry_writes_all_outputs(repo):
    assert run(repo) == 0
    assert sorted(outputs_exist(repo)) == sorted(es.OUTPUTS)


def test_parse_fields_handles_variants_and_ignores_the_template():
    problems: list[str] = []
    text = registry().replace(
        '- **Interpretation:** fine\n\n### <a id="gw-rec-001">',
        '- **Result (Phase 14, 2026-11-17, GW-E2E-009):** later\n\n### <a id="gw-rec-001">',
    )
    entries = es.parse_entries(text, problems)
    assert problems == []
    assert [e.id for e in entries] == [
        "GW-BASE-001", "GW-REC-001", "GW-REC-002", "GW-REC-003", "GW-REC-004",
    ]  # fmt: skip
    first = entries[0]
    assert first.fields["Result"] == ("recorded continued on a second line", "later")
    assert "Hypothesis" in first.fields
    assert entries[2].first("Superseded by") == "GW-REC-001, for the same pages only."


def test_failure_headings_can_hold_several_ids_and_the_summary_does_not_count():
    text = "### 2026-01-04 — GW-EFF-003 / GW-EFF-004 / GW-E2E-004 — broke, unlike GW-REC-001\n"
    assert es.parse_failures(text) == {"GW-EFF-003", "GW-EFF-004", "GW-E2E-004"}


def number(text: str) -> es.NumberToken:
    tokens = es.tokenize(text)
    assert len(tokens) == 1
    return tokens[0]


@pytest.mark.parametrize(
    ("report", "printed", "found"),
    [
        ("7.534", "7.53", True),  # rounds to the printed digit
        ("7.535", "7.53", True),  # exactly half a unit away
        ("7.536", "7.53", False),
        ("-7.53", "7.53", True),  # the sign is ignored
        ("7.53", "−7.53", True),
        ("0.0753", "7.53%", True),  # fraction to percent
        ("1,500.0", "1.5 s", True),  # ms to s
        ("7.63", "7.53", False),
    ],
)
def test_numbers_match_with_the_allowed_changes(tmp_path, report, printed, found):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "r.md").write_text(f"value {report}\n", encoding="utf-8")
    numbers = es.Numbers(tmp_path)
    assert numbers.holds("reports/r.md", number(printed)) is found


def test_digits_inside_a_hash_do_not_count_as_a_number(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "r.md").write_text("commit a7530bc4 and x7.53y\n", encoding="utf-8")
    assert not es.Numbers(tmp_path).holds("reports/r.md", number("7.53"))


def test_a_file_named_after_the_run_names_it(repo):
    edit(repo, "reports/base/result.json", '"GW-BASE-001"', '"another run"')
    source = repo / "reports/base/result.json"
    source.rename(source.with_name("GW-BASE-001.json"))
    assert run(repo) == 0


def test_a_folder_named_after_the_run_is_not_enough(repo, capsys):
    edit(repo, "reports/training/GW-REC-001/report.md", "# GW-REC-001", "# A run")
    assert run(repo) == 1
    assert "GW-REC-001: no file under its report paths names it" in capsys.readouterr().err


def test_generated_files_are_not_evidence(repo, capsys):
    edit(repo, REG, "`reports/training/GW-REC-002/`, `reports/shared.md`", "`reports/shared.md`")
    assert run(repo) == 1
    capsys.readouterr()
    (repo / es.OUT_MD).write_text("GW-REC-002 6.30\n", encoding="utf-8")
    edit(repo, REG, "`reports/shared.md`", "`reports/`")
    edit(repo, "reports/training/GW-REC-002/report.md", "GW-REC-002", "x")
    edit(repo, "reports/training/GW-REC-002/report.md", "6.301", "1")
    assert run(repo) == 1
    assert "GW-REC-002: 6.30 is not under" in capsys.readouterr().err


def hardening(inputs: int, errors: int) -> dict[str, object]:
    kind = {"inputs": inputs, "results": inputs - errors, "glypharis_errors": {"E": errors}}
    return {"inputs": {"torch": {"kinds": {"a": kind}}, "onnx": {"kinds": {"a": kind}}}}


def test_summed_counts_come_from_the_report(repo, monkeypatch):
    path = "reports/training/GW-REC-002/hardening.json"
    (repo / path).write_text(json.dumps(hardening(269, 222)), encoding="utf-8")
    monkeypatch.setitem(es.SUMMED_COUNTS, "GW-REC-002", path)
    edit(repo, REG, "| val CER 6.30% |", "| val CER 6.30%; 269 inputs, 222 errors |")
    assert run(repo) == 0


def test_a_summed_count_the_report_does_not_give_fails(repo, monkeypatch, capsys):
    path = "reports/training/GW-REC-002/hardening.json"
    (repo / path).write_text(json.dumps(hardening(268, 222)), encoding="utf-8")
    monkeypatch.setitem(es.SUMMED_COUNTS, "GW-REC-002", path)
    edit(repo, REG, "| val CER 6.30% |", "| val CER 6.30%; 269 inputs, 222 errors |")
    assert run(repo) == 1
    assert "269 is not under" in capsys.readouterr().err


def test_summed_counts_need_the_backends_to_agree(tmp_path):
    report = hardening(10, 4)
    report["inputs"]["onnx"] = hardening(11, 4)["inputs"]["onnx"]  # type: ignore[index]
    (tmp_path / "h.json").write_text(json.dumps(report), encoding="utf-8")
    assert es.summed_counts(tmp_path, "h.json") == set()


def rows_of(root: Path) -> dict[str, dict[str, str]]:
    text = (root / es.OUT_CSV).read_text(encoding="utf-8")
    return {row["id"]: row for row in csv.DictReader(io.StringIO(text))}


def test_ledger_suffixes_merge_into_their_experiment(repo):
    assert run(repo) == 0
    row = rows_of(repo)["GW-REC-001"]
    assert (row["gpu_h"], row["cpu_h"], row["train_h"]) == ("2.00", "0.50", "1.50")
    assert (row["jobs"], row["failed_jobs"]) == ("4", "2")


def test_unattributed_ledger_row_is_shown_apart(repo):
    assert run(repo) == 0
    text = (repo / es.OUT_ROOT).read_text(encoding="utf-8")
    assert "Not attributed to one experiment: Phase 16 scoring, 0.25 CPU-hours" in text


def test_experiments_without_a_ledger_row_are_listed_by_reason(repo):
    assert run(repo) == 0
    text = (repo / es.OUT_ROOT).read_text(encoding="utf-8")
    assert "3 registered experiments have no ledger row" in text
    assert "- Abandoned or not run (2): GW-REC-003, GW-REC-004" in text
    assert "names neither them nor their phase (1): GW-REC-002" in text


def test_flags_are_derived(repo):
    assert run(repo) == 0
    rows = rows_of(repo)
    assert rows["GW-BASE-001"]["test_accesses"] == "2"
    assert rows["GW-BASE-001"]["failure_note"] == "yes"  # second ID in a failure heading
    assert rows["GW-REC-002"]["failure_note"] == "no"  # only in that heading's summary
    assert rows["GW-REC-001"]["in_ablation_table"] == "yes"
    assert rows["GW-REC-003"]["in_ablation_table"] == "no"  # prose does not count
    assert rows["GW-REC-002"]["superseded_by"] == "GW-REC-001"
    text = (repo / es.OUT_ROOT).read_text(encoding="utf-8")
    assert "test log: 2; failure note" in text
    assert "superseded by [GW-REC-001](research/experiments.md#gw-rec-001)" in text
    assert "for the same pages only." in text


def test_links_use_the_registry_anchors(repo):
    assert run(repo) == 0
    root_text = (repo / es.OUT_ROOT).read_text(encoding="utf-8")
    assert "[GW-REC-001](research/experiments.md#gw-rec-001)" in root_text
    summary = (repo / es.OUT_MD).read_text(encoding="utf-8")
    assert "[GW-REC-001](../research/experiments.md#gw-rec-001)" in summary


def test_phases_sort_numerically_with_v1_after_the_plain_phase(repo):
    assert run(repo) == 0
    text = (repo / es.OUT_ROOT).read_text(encoding="utf-8")
    headings = [line for line in text.split("\n") if line.startswith("## Phase ")]
    assert headings == ["## Phase 05", "## Phase 06", "## Phase 17", "## Phase 17 (v1)"]


def test_output_is_deterministic_with_lf_only(repo):
    assert run(repo) == 0
    first = {path: (repo / path).read_bytes() for path in es.OUTPUTS}
    assert run(repo) == 0
    assert first == {path: (repo / path).read_bytes() for path in es.OUTPUTS}
    for path, data in first.items():
        assert b"\r" not in data, path
        assert data.endswith(b"\n"), path
        assert str(repo).encode() not in data, path


def test_check_passes_when_current_and_writes_nothing(repo):
    assert run(repo) == 0
    before = {path: (repo / path).stat().st_mtime_ns for path in es.OUTPUTS}
    assert run(repo, "--check") == 0
    assert before == {path: (repo / path).stat().st_mtime_ns for path in es.OUTPUTS}


def test_check_ignores_crlf(repo):
    assert run(repo) == 0
    for path in es.OUTPUTS:
        full = repo / path
        full.write_bytes(full.read_bytes().replace(b"\n", b"\r\n"))
    assert run(repo, "--check") == 0


def test_check_fails_when_stale(repo, capsys):
    assert run(repo) == 0
    edit(repo, REG, "| Base question |", "| Another question |")
    before = (repo / es.OUT_MD).read_bytes()
    assert run(repo, "--check") == 1
    assert f"{es.OUT_ROOT} is out of date" in capsys.readouterr().err
    assert (repo / es.OUT_MD).read_bytes() == before


def test_check_fails_when_outputs_are_missing_and_creates_none(repo):
    assert run(repo, "--check") == 1
    assert outputs_exist(repo) == []


def test_check_fails_on_a_broken_registry(repo):
    assert run(repo) == 0
    edit(repo, REG, "| Conditional run | not run |", "| Conditional run | skipped |")
    assert run(repo, "--check") == 1


def test_missing_input_is_reported(repo, capsys):
    (repo / "research/findings.md").unlink()
    assert run(repo) == 1
    assert "cannot read research/findings.md" in capsys.readouterr().err
    assert outputs_exist(repo) == []


def test_committed_summary_is_current():
    assert es.main(["--root", str(REPO), "--check"]) == 0
