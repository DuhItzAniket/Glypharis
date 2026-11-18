"""Check the experiment registry and write the summaries generated from it (Phase 21, item 7).

    python scripts/experiment_summary.py [--root .] [--check]

Reads `research/experiments.md`, `research/ablations.md`, `research/failures.md`,
`research/findings.md`, `research/hypotheses.md`, `research/test_access_log.md`,
`reports/compute/ledger.json` and `ledger.md`, and the files under each row's Report paths. It opens
nothing else: no data, run or checkpoint folder. It writes `EXPERIMENTS.md` (one table per phase),
`reports/experiment_summary.csv` and `reports/experiment_summary.md` (one flat row per experiment),
with LF line ends and no timestamp, hash or absolute path, so the same registry always gives the
same bytes.

Every rule below is checked first. Any failure is printed with its file and line, the exit status is
1 and nothing is written.

- R1: index rows and entries match one to one, without duplicates, and each row links to its
  entry's anchor.
- R2: IDs have a known prefix and are contiguous per prefix.
- R3: the status is in the registry's list, with an optional note in brackets.
- R4: the allocation date and the phase number agree between row and entry.
- R5: by status. done: a key result, a Result and a run date. failed or abandoned: a Result and the
  ID in a `research/failures.md` heading. not run: a Result and no report. planned or running: no
  key result and no report on disk.
- R6: every Report path exists under `reports/`. `—` is for runs that never produced one. For a done
  row, a file under its paths names the ID, in its text or its own file name.
- R7: each key-result number with a decimal point, a thousands comma or an exponent, or of 100 or
  more, is found in a file under that row's paths. The sign is ignored, rounding to the printed
  digit is allowed, and so are fraction to percent and ms to s. A number written with K or M
  (`+250K`, the name of a Phase 12 arm) must appear in a file exactly as written. This is a loose
  presence check of registry cells: it does not show that a number is the right quantity, and it is
  not evidence for the paper, which has `scripts/trace_numbers.py`.
- R8: IDs cited in the research files and the ledger are registered, IDs in ablation table rows are
  done, and a `Superseded by` names a done experiment.
- R9: the ledger's rows add up to its totals per device, and its note still leaves out the phases
  this summary says it leaves out.
- R10: every test-log row names an ID and a phase allowed to read test data.
- R11: the generated text has no drive path, home folder or email address.

`--check` builds the same text in memory and compares it with the files on disk after turning CRLF
into LF. Exit status 0 when they are equal, 1 otherwise. It writes nothing.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Final

from numtrace import (
    SOURCE_SUFFIXES,
    IndexedFile,
    NumberToken,
    ReportIndex,
    TraceError,
    half_unit,
    tokenize,
)

REGISTRY: Final = "research/experiments.md"
ABLATIONS: Final = "research/ablations.md"
FAILURES: Final = "research/failures.md"
TEST_LOG: Final = "research/test_access_log.md"
CITING_FILES: Final = (ABLATIONS, FAILURES, "research/findings.md", "research/hypotheses.md")
LEDGER_JSON: Final = "reports/compute/ledger.json"
LEDGER_MD: Final = "reports/compute/ledger.md"
OUT_ROOT: Final = "EXPERIMENTS.md"
OUT_CSV: Final = "reports/experiment_summary.csv"
OUT_MD: Final = "reports/experiment_summary.md"
OUTPUTS: Final = (OUT_ROOT, OUT_CSV, OUT_MD)
# Generated from the registry or beside it, so they can never count as a row's evidence.
NOT_EVIDENCE: Final = frozenset(
    {OUT_CSV, OUT_MD, "reports/benchmark_results/comparison_table.md",
     "reports/benchmark_results/comparison_table.csv"}
)  # fmt: skip

PREFIXES: Final = ("BASE", "REC", "DET", "E2E", "DATA", "EFF")
STATUSES: Final = ("done", "failed", "abandoned", "not run", "planned", "running")
TEST_PHASES: Final = frozenset({5, 14, 22})
NONE_MARK: Final = "—"
BLANK: Final = frozenset({"", NONE_MARK, "-"})
# Ledger rows that belong to no single experiment.
UNATTRIBUTED: Final = frozenset({"Phase 16 scoring"})
# What the ledger's "Not in these logs" note leaves out; R9 checks that the sentence is still there.
UNCOUNTED_PHASES: Final = frozenset({9, 10, 14, 15})
UNCOUNTED_NOTE: Final = "Phases 9, 10, 14 and 15 are not counted"
# Fraction to percent, and ms to s.
SCALES: Final = (Fraction(1), Fraction(100), Fraction(1, 1000))
# Units that make a number larger than it is printed; these are matched as written.
MULTIPLIERS: Final = {"K": 1000, "M": 1_000_000}
# Key-result counts no report prints, because they are sums over a report's per-kind table.
SUMMED_COUNTS: Final = {"GW-E2E-012": "reports/hardening/GW-E2E-012.json"}

ID_TEXT: Final = r"GW-[A-Z0-9]+-[0-9]{3}"
ID_RE: Final = re.compile(ID_TEXT)
ID_PARTS: Final = re.compile(r"GW-([A-Z0-9]+)-([0-9]{3})")
ROW_RE: Final = re.compile(rf"^\| \[({ID_TEXT})\]\(#([^)]*)\) \|")
HEADING_RE: Final = re.compile(rf'^### <a id="([^"]*)"></a>({ID_TEXT}) — (.*)$')
FIELD_RE: Final = re.compile(r"^- \*\*([^*]+?):\*\*[ \t]*(.*)$")
STATUS_RE: Final = re.compile(r"^(planned|running|done|failed|abandoned|not run)( \(.+\))?$")
PATH_RE: Final = re.compile(r"`([^`]+)`")
LEDGER_KEY_RE: Final = re.compile(rf"^({ID_TEXT})(?: \(.+\))?$")
PERSONAL_RES: Final = (
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"),
    re.compile(r"/(?:Users|home)/"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
)

CSV_COLUMNS: Final = (
    "id", "prefix", "phase", "date_allocated", "status", "question", "key_result", "report_paths",
    "gpu_h", "cpu_h", "train_h", "jobs", "failed_jobs", "test_accesses", "failure_note",
    "in_ablation_table", "superseded_by",
)  # fmt: skip


class InputError(Exception):
    """A file the check needs cannot be read."""


@dataclass(frozen=True)
class Row:
    """One index row of the registry."""

    id: str
    link: str
    date: str
    phase: str
    question: str
    status: str
    key_result: str
    report: str
    line: int

    @property
    def base_status(self) -> str:
        return self.status.split(" (", 1)[0]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(PATH_RE.findall(self.report))


@dataclass(frozen=True)
class Entry:
    """One `###` entry of the registry. ``fields`` maps a field name to all values it has."""

    id: str
    anchor: str
    title: str
    line: int
    fields: dict[str, tuple[str, ...]]

    def first(self, name: str) -> str | None:
        values = self.fields.get(name)
        return values[0] if values else None


@dataclass
class Compute:
    """What the ledger holds for one experiment, with its device rows added together."""

    gpu_h: float | None = None
    cpu_h: float | None = None
    train_h: float | None = None
    jobs: int = 0
    failed_jobs: int = 0


@dataclass
class Sources:
    """Everything read from the repository, parsed."""

    rows: list[Row] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    compute: dict[str, Compute] = field(default_factory=dict)
    unattributed: dict[str, dict[str, Any]] = field(default_factory=dict)
    totals: dict[str, float] = field(default_factory=dict)
    test_accesses: Counter[str] = field(default_factory=Counter)
    failure_ids: set[str] = field(default_factory=set)
    ablation_ids: set[str] = field(default_factory=set)


def read_text(root: Path, path: str) -> str:
    """A repository file as text with LF line ends."""
    try:
        return (root / path).read_bytes().decode("utf-8").replace("\r\n", "\n")
    except (OSError, UnicodeDecodeError) as error:
        raise InputError(f"cannot read {path}: {error}") from error


def split_cells(line: str) -> list[str]:
    """The cells of a Markdown table row; ``\\|`` stays inside its cell."""
    body = line.strip().removeprefix("|").removesuffix("|")
    return [cell.replace("\\|", "|").strip() for cell in re.split(r"(?<!\\)\|", body)]


def parse_rows(text: str, problems: list[str]) -> list[Row]:
    """The index rows, which all come before the first entry."""
    rows = []
    for number, line in enumerate(text.split("\n"), 1):
        if line.startswith("### "):
            break
        match = ROW_RE.match(line)
        if not match:
            continue
        cells = split_cells(line)
        if len(cells) != 7:
            problems.append(f"R1 {REGISTRY}:{number}: {len(cells)} cells, expected 7")
            continue
        _, date, phase, question, status, key_result, report = cells
        rows.append(
            Row(match[1], match[2], date, phase, question, status, key_result, report, number)
        )
    return rows


def parse_entries(text: str, problems: list[str]) -> list[Entry]:
    """The entries, from the first `###` heading up to the entry template."""
    lines = text.split("\n")
    entries: list[Entry] = []
    current: dict[str, list[str]] | None = None
    last: tuple[str, int] | None = None
    head: tuple[str, str, str, int] | None = None

    def close() -> None:
        if head is not None and current is not None:
            fields = {name: tuple(v.strip() for v in vals) for name, vals in current.items()}
            entries.append(Entry(head[1], head[0], head[2], head[3], fields))

    for number, line in enumerate(lines, 1):
        if line.startswith("## Entry template"):
            break
        if line.startswith("### "):
            close()
            match = HEADING_RE.match(line)
            if not match:
                problems.append(f"R1 {REGISTRY}:{number}: entry heading without an anchor and ID")
                head, current, last = None, None, None
                continue
            head = (match[1], match[2], match[3], number)
            current, last = {}, None
            continue
        if current is None:
            continue
        found = FIELD_RE.match(line)
        if found:
            # "Result (Phase 14, ...)" and "Result" are both results.
            name = found[1].split(" (", 1)[0].strip()
            current.setdefault(name, []).append(found[2])
            last = (name, len(current[name]) - 1)
        elif line.startswith("- ") or line.strip() == "---":
            last = None
        elif last is not None and line.strip():
            current[last[0]][last[1]] += " " + line.strip()
    close()
    return entries


def parse_ledger(root: Path, sources: Sources, problems: list[str]) -> None:
    """Add the ledger's rows to their experiments; R9 is checked here, on the raw rows."""
    try:
        ledger = json.loads(read_text(root, LEDGER_JSON))
    except json.JSONDecodeError as error:
        raise InputError(f"cannot parse {LEDGER_JSON}: {error}") from error
    sums = {"gpu": 0.0, "cpu": 0.0}
    for key, row in ledger["experiments"].items():
        device = row["device"]
        if device not in sums:
            problems.append(f"R9 {LEDGER_JSON}: {key}: unknown device {device!r}")
            continue
        sums[device] += row["wall_h"]
        match = LEDGER_KEY_RE.match(key)
        if key in UNATTRIBUTED:
            sources.unattributed[key] = row
        elif not match:
            problems.append(f"R8 {LEDGER_JSON}: row {key!r} is neither an ID nor a known label")
        else:
            add_compute(sources.compute.setdefault(match[1], Compute()), row)
    sources.totals = ledger["totals"]
    for device, total in sums.items():
        if abs(total - ledger["totals"][f"{device}_h"]) > 1e-6:
            problems.append(
                f"R9 {LEDGER_JSON}: {device} rows add up to {total:.3f} h, "
                f"the total says {ledger['totals'][f'{device}_h']:.3f} h"
            )
    if UNCOUNTED_NOTE not in " ".join(read_text(root, LEDGER_MD).split()):
        problems.append(f"R9 {LEDGER_MD}: the note {UNCOUNTED_NOTE!r} is gone")


def add_compute(total: Compute, row: dict[str, Any]) -> None:
    hours = row["wall_h"]
    if row["device"] == "gpu":
        total.gpu_h = (total.gpu_h or 0.0) + hours
    else:
        total.cpu_h = (total.cpu_h or 0.0) + hours
    if row.get("train_h") is not None:
        total.train_h = (total.train_h or 0.0) + row["train_h"]
    total.jobs += row["jobs"]
    total.failed_jobs += row["failed_jobs"]


def parse_failures(text: str) -> set[str]:
    """IDs named in a failure heading: `### date — ID / ID — summary`."""
    found: set[str] = set()
    for line in text.split("\n"):
        if line.startswith("### "):
            parts = line.split(" — ")
            if len(parts) >= 2:
                found.update(ID_RE.findall(parts[1]))
    return found


def parse_ablation_rows(text: str) -> dict[str, int]:
    """IDs in the table rows of the ablation file, each with its first line."""
    found: dict[str, int] = {}
    for number, line in enumerate(text.split("\n"), 1):
        if line.startswith("|"):
            for experiment in ID_RE.findall(line):
                found.setdefault(experiment, number)
    return found


def parse_test_log(text: str, problems: list[str]) -> Counter[str]:
    """Rows per experiment in the test-access log; R10 is checked on the way."""
    counts: Counter[str] = Counter()
    for number, line in enumerate(text.split("\n"), 1):
        if not line.startswith("|"):
            continue
        cells = split_cells(line)
        if cells[0] == "Date" or set(cells[0]) <= {"-", ":"}:
            continue
        where = f"R10 {TEST_LOG}:{number}"
        if len(cells) != 6:
            problems.append(f"{where}: {len(cells)} cells, expected 6")
            continue
        experiments = ID_RE.findall(cells[2])
        if not experiments:
            problems.append(f"{where}: no experiment ID in {cells[2]!r}")
        if not cells[4].isdigit() or int(cells[4]) not in TEST_PHASES:
            problems.append(f"{where}: phase {cells[4]!r} may not read test data")
        counts.update(experiments)
    return counts


def phase_number(phase: str) -> int | None:
    match = re.match(r"[0-9]+", phase)
    return int(match[0]) if match else None


def check_registry(rows: list[Row], entries: list[Entry], problems: list[str]) -> None:
    """R1 to R4."""
    row_ids = Counter(row.id for row in rows)
    entry_ids = Counter(entry.id for entry in entries)
    for experiment in sorted(row_ids | entry_ids):
        if row_ids[experiment] > 1:
            problems.append(f"R1 {REGISTRY}: {experiment} has {row_ids[experiment]} index rows")
        if entry_ids[experiment] > 1:
            problems.append(f"R1 {REGISTRY}: {experiment} has {entry_ids[experiment]} entries")
        if not row_ids[experiment]:
            problems.append(f"R1 {REGISTRY}: {experiment} has an entry but no index row")
        if not entry_ids[experiment]:
            problems.append(f"R1 {REGISTRY}: {experiment} has an index row but no entry")
    numbers: dict[str, list[int]] = {}
    for experiment in row_ids:
        match = ID_PARTS.fullmatch(experiment)
        assert match is not None  # ROW_RE only accepts this form
        if match[1] not in PREFIXES:
            problems.append(f"R2 {REGISTRY}: {experiment} has an unknown prefix")
        numbers.setdefault(match[1], []).append(int(match[2]))
    for prefix, found in sorted(numbers.items()):
        missing = sorted(set(range(1, max(found) + 1)) - set(found))
        if missing:
            problems.append(f"R2 {REGISTRY}: GW-{prefix} is missing numbers {missing}")
    by_id = {entry.id: entry for entry in entries}
    for row in rows:
        where = f"{REGISTRY}:{row.line}: {row.id}"
        if not STATUS_RE.match(row.status):
            problems.append(f"R3 {where}: status {row.status!r} is not in the list")
        entry = by_id.get(row.id)
        if entry is None:
            continue
        if row.link != entry.anchor or entry.anchor != row.id.lower():
            problems.append(f"R1 {where}: link #{row.link} and anchor {entry.anchor} differ")
        dates = entry.first("Date allocated / run")
        allocated = dates.split(" / ", 1)[0].strip() if dates else None
        if allocated != row.date:
            problems.append(f"R4 {where}: allocated {row.date} here, {allocated} in the entry")
        phase = entry.first("Phase")
        if phase is None or phase_number(phase) != phase_number(row.phase):
            problems.append(f"R4 {where}: phase {row.phase} here, {phase} in the entry")


def has_text(entry: Entry, name: str) -> bool:
    return any(value not in BLANK for value in entry.fields.get(name, ()))


def check_status(root: Path, sources: Sources, problems: list[str]) -> None:
    """R5."""
    by_id = {entry.id: entry for entry in sources.entries}
    for row in sources.rows:
        entry = by_id.get(row.id)
        if entry is None or not STATUS_RE.match(row.status):
            continue
        where = f"R5 {REGISTRY}:{row.line}: {row.id} ({row.base_status})"
        status = row.base_status
        if status in ("planned", "running"):
            if row.key_result not in BLANK:
                problems.append(f"{where}: has a key result")
            on_disk = [p for p in row.paths if (root / p).exists()]
            folder = f"reports/training/{row.id}"
            if (root / folder).exists():
                on_disk.append(folder)
            if on_disk:
                problems.append(f"{where}: reports exist: {', '.join(on_disk)}")
            continue
        if not has_text(entry, "Result"):
            problems.append(f"{where}: the entry has no Result")
        if status == "done":
            if row.key_result in BLANK:
                problems.append(f"{where}: no key result")
            dates = entry.first("Date allocated / run") or ""
            run = dates.split(" / ", 1)[1].strip() if " / " in dates else ""
            if run in BLANK:
                problems.append(f"{where}: no run date")
        elif status in ("failed", "abandoned") and row.id not in sources.failure_ids:
            problems.append(f"{where}: not in a heading of {FAILURES}")
        elif status == "not run" and row.report not in BLANK:
            problems.append(f"{where}: names a report")


def report_files(root: Path, path: str) -> list[str]:
    """The readable files at or under a report path, as sorted repository paths."""
    full = root / path
    found = [full] if full.is_file() else sorted(p for p in full.rglob("*") if p.is_file())
    names = (p.relative_to(root).as_posix() for p in found)
    return [n for n in names if n.endswith(SOURCE_SUFFIXES) and n not in NOT_EVIDENCE]


class Numbers:
    """The numbers of indexed report files, searchable by value."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.index = ReportIndex(root)
        self._sorted: dict[str, tuple[list[float], list[Decimal]]] = {}
        self._text: dict[str, str] = {}

    def file(self, path: str) -> IndexedFile:
        return self.index.file(path)

    def _values(self, path: str) -> tuple[list[float], list[Decimal]]:
        if path not in self._sorted:
            values = sorted({abs(v.value) for v in self.file(path).values})
            self._sorted[path] = ([float(v) for v in values], values)
        return self._sorted[path]

    def names(self, path: str, experiment: str) -> bool:
        """True when the file's text or its own name holds the experiment ID."""
        return experiment in self.file(path).ids or experiment in PurePosixPath(path).name

    def has_literal(self, path: str, token: NumberToken) -> bool:
        """True when the file has the number exactly as printed, with its unit."""
        if path not in self._text:
            self._text[path] = read_text(self.root, path)
        wanted = re.escape(token.raw + token.unit)
        return re.search(rf"(?<![\w.]){wanted}(?!\w)", self._text[path]) is not None

    def holds(self, path: str, token: NumberToken) -> bool:
        """True when the file has this number, up to sign, rounding and the allowed scales."""
        if token.unit in MULTIPLIERS:
            return self.has_literal(path, token)
        printed = Fraction(abs(token.value))
        half = half_unit(token.exponent)
        floats, values = self._values(path)
        for scale in SCALES:
            low, high = float((printed - half) / scale), float((printed + half) / scale)
            margin = max(abs(low), abs(high)) * 1e-9
            start = bisect.bisect_left(floats, low - margin)
            stop = bisect.bisect_right(floats, high + margin)
            if any(abs(Fraction(v) * scale - printed) <= half for v in values[start:stop]):
                return True
        return False


def is_checked(token: NumberToken) -> bool:
    """The numbers R7 looks for; small whole numbers would match anything."""
    raw = token.raw.lower()
    size = abs(token.value) * MULTIPLIERS.get(token.unit, 1)
    return token.exponent < 0 or "," in raw or "e" in raw or size >= 100


def summed_counts(root: Path, path: str) -> set[Decimal]:
    """Inputs and typed errors added over the kinds of a hardening report, if backends agree."""
    report = json.loads(read_text(root, path))
    per_backend = []
    for backend in report["inputs"].values():
        kinds = backend["kinds"].values()
        inputs = sum(kind["inputs"] for kind in kinds)
        errors = sum(sum(kind["glypharis_errors"].values()) for kind in kinds)
        per_backend.append((inputs, errors))
    if not per_backend or len(set(per_backend)) != 1:
        return set()
    return {Decimal(n) for n in per_backend[0]}


def check_reports(root: Path, sources: Sources, problems: list[str]) -> None:
    """R6 and R7."""
    numbers = Numbers(root)
    for row in sources.rows:
        status = row.base_status
        if status in ("planned", "running"):
            continue
        where = f"{REGISTRY}:{row.line}: {row.id}"
        if row.report in BLANK:
            if status not in ("abandoned", "not run"):
                problems.append(f"R6 {where}: {status} but no report path")
            continue
        if not row.paths:
            problems.append(f"R6 {where}: no path in the Report cell")
            continue
        files: list[str] = []
        for path in row.paths:
            if not path.startswith("reports/") or ".." in path.split("/"):
                problems.append(f"R6 {where}: {path} is not under reports/")
            elif not (root / path).exists():
                problems.append(f"R6 {where}: {path} does not exist")
            else:
                files.extend(report_files(root, path))
        readable = []
        for name in files:
            try:
                numbers.file(name)
            except (TraceError, ValueError) as error:
                problems.append(f"R7 {where}: cannot read {name}: {error}")
            else:
                readable.append(name)
        if status == "done" and not any(numbers.names(n, row.id) for n in readable):
            problems.append(f"R6 {where}: no file under its report paths names it")
        check_numbers(root, row, readable, numbers, where, problems)


def check_numbers(
    root: Path, row: Row, files: list[str], numbers: Numbers, where: str, problems: list[str]
) -> None:
    """R7 for one row."""
    summed: set[Decimal] = set()
    if row.id in SUMMED_COUNTS and (root / SUMMED_COUNTS[row.id]).is_file():
        summed = summed_counts(root, SUMMED_COUNTS[row.id])
    for token in tokenize(row.key_result):
        if not is_checked(token) or abs(token.value) in summed:
            continue
        if not any(numbers.holds(name, token) for name in files):
            problems.append(f"R7 {where}: {token.raw} is not under its report paths")


def check_citations(root: Path, sources: Sources, problems: list[str]) -> None:
    """R8."""
    status = {row.id: row.base_status for row in sources.rows}
    for path in (*CITING_FILES, TEST_LOG):
        for number, line in enumerate(read_text(root, path).split("\n"), 1):
            for experiment in ID_RE.findall(line):
                if experiment not in status:
                    problems.append(f"R8 {path}:{number}: {experiment} is not registered")
    for experiment in sorted(sources.compute):
        if experiment not in status:
            problems.append(f"R8 {LEDGER_JSON}: {experiment} is not registered")
    ablation_rows = parse_ablation_rows(read_text(root, ABLATIONS))
    for experiment, number in ablation_rows.items():
        if experiment in status and status[experiment] != "done":
            problems.append(
                f"R8 {ABLATIONS}:{number}: {experiment} is {status[experiment]}, not done"
            )
    for entry in sources.entries:
        value = entry.first("Superseded by")
        if value is None:
            continue
        target = superseded_target(value)
        if target is None or status.get(target) != "done":
            problems.append(
                f"R8 {REGISTRY}:{entry.line}: {entry.id} is superseded by {value!r}, "
                "which is not a done experiment"
            )


def superseded_target(value: str) -> str | None:
    match = re.match(ID_TEXT, value)
    return match[0] if match else None


def check_personal(outputs: dict[str, str], problems: list[str]) -> None:
    """R11."""
    for path, text in outputs.items():
        for number, line in enumerate(text.split("\n"), 1):
            for pattern in PERSONAL_RES:
                match = pattern.search(line)
                if match:
                    problems.append(f"R11 {path}:{number}: {match[0]!r} looks personal")


def id_key(experiment: str) -> tuple[int, int]:
    match = ID_PARTS.fullmatch(experiment)
    assert match is not None
    prefix = PREFIXES.index(match[1]) if match[1] in PREFIXES else len(PREFIXES)
    return prefix, int(match[2])


def phase_key(phase: str) -> tuple[int, int, str]:
    """Numeric order, with `17 (v1)` straight after `17`."""
    return phase_number(phase) or 0, 1 if "(" in phase else 0, phase


def hours(value: float | None, missing: str) -> str:
    return missing if value is None else f"{value:.2f}"


def entry_link(experiment: str, prefix: str) -> str:
    return f"[{experiment}]({prefix}{REGISTRY}#{experiment.lower()})"


def report_links(row: Row, prefix: str) -> str:
    if not row.paths:
        return NONE_MARK
    return ", ".join(f"[`{path}`]({prefix}{path})" for path in row.paths)


def superseded_map(sources: Sources) -> dict[str, tuple[str, str]]:
    """Experiment -> (the run replacing it, what the entry says about the replacement)."""
    found = {}
    for entry in sources.entries:
        value = entry.first("Superseded by")
        target = superseded_target(value) if value else None
        if value and target:
            found[entry.id] = (target, value[len(target) :].lstrip(" ,;:"))
    return found


def flags(row: Row, sources: Sources, replaced: dict[str, tuple[str, str]], prefix: str) -> str:
    found = []
    if sources.test_accesses[row.id]:
        found.append(f"test log: {sources.test_accesses[row.id]}")
    if row.id in sources.failure_ids:
        found.append("failure note")
    if row.id in sources.ablation_ids:
        found.append("ablation")
    if row.id in replaced:
        found.append(f"superseded by {entry_link(replaced[row.id][0], prefix)}")
    return "; ".join(found) or NONE_MARK


def table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(cell.replace("|", "\\|") for cell in r) + " |" for r in rows)
    return lines


def status_counts(rows: list[Row]) -> list[str]:
    counts = Counter((row.id.split("-")[1], row.base_status) for row in rows)
    body = []
    for prefix in PREFIXES:
        cells = [str(counts[(prefix, status)]) for status in STATUSES]
        body.append([f"GW-{prefix}", str(sum(map(int, cells))), *cells])
    totals = [str(sum(counts[(p, status)] for p in PREFIXES)) for status in STATUSES]
    body.append(["all", str(len(rows)), *totals])
    return table(["Prefix", "Registered", *STATUSES], body)


def compute_gap(rows: list[Row], compute: dict[str, Compute]) -> list[str]:
    """Bullets naming the registered experiments that have no ledger row, by reason."""
    absent = [row for row in rows if row.id not in compute]
    never = [r.id for r in absent if r.base_status in ("abandoned", "not run", "planned")]
    uncounted = [
        r.id for r in absent if r.id not in never and phase_number(r.phase) in UNCOUNTED_PHASES
    ]
    other = [r.id for r in absent if r.id not in never and r.id not in uncounted]
    groups = (
        ("Abandoned or not run", never),
        ("In phases the ledger says it does not count (9, 10, 14, 15)", uncounted),
        ("Ran, with no ledger row; its note names neither them nor their phase", other),
    )
    lines = [f"{len(absent)} registered experiments have no ledger row:", ""]
    for label, members in groups:
        if members:
            names = ", ".join(sorted(members, key=id_key))
            lines.append(f"- {label} ({len(members)}): {names}")
    return lines


def render_root(sources: Sources) -> str:
    rows = sources.rows
    replaced = superseded_map(sources)
    totals = sources.totals
    lines = [
        "# Experiments",
        "",
        "Generated by `scripts/experiment_summary.py` from `research/experiments.md`; do not edit "
        "by hand.",
        "",
        "Each ID links to its full entry. Key results are copied from the registry. The generator "
        "checks that their numbers occur in the listed reports, which shows they are present, not "
        "that they are the right quantity. The paper's numbers are traced separately.",
        "",
        "## Status",
        "",
        *status_counts(rows),
        "",
        "## Scope",
        "",
        "- v1 is GW-REC-034 with GW-DET-001, and it keeps PyTorch as its default backend.",
        "- The laptop CPU latency targets of ADR-0011 are recorded as missed for v1.",
        "- Phones are the target hardware; no phone has been measured.",
        "- The all-ONNX build with fused attention is a new version, not v1, to be evaluated in "
        "Phase 22.",
        "- Final validation (Phase 22) has not been run.",
        "- No weights are distributed; release is Phase 23.",
        "- Dataset licences and attribution are in [DATASETS.md](DATASETS.md).",
        "",
        "## Compute",
        "",
        f"The queue logs in [`{LEDGER_MD}`]({LEDGER_MD}) hold {totals['gpu_h']:.1f} GPU-hours and "
        f"{totals['cpu_h']:.1f} CPU-hours. This is not the project's total: jobs run outside those "
        "queues are not in it, as the ledger explains. The hours below are wall-clock time per "
        "experiment, and `—` means the ledger has no job for it on that device.",
        "",
    ]
    for label, job in sorted(sources.unattributed.items()):
        lines.append(
            f"Not attributed to one experiment: {label}, {job['wall_h']:.2f} "
            f"{job['device'].upper()}-hours over {job['jobs']} jobs."
        )
        lines.append("")
    lines.extend(compute_gap(rows, sources.compute))
    lines.extend(
        [
            "",
            "## Flags",
            "",
            "- `test log: N`: rows in `research/test_access_log.md`. Analyses that only re-read "
            "saved outputs add none, so this counts logged access, not every use of test results.",
            "- `failure note`: the ID is in a heading of `research/failures.md`.",
            "- `ablation`: the ID is in a table row of `research/ablations.md`.",
            "- `superseded by`: a v1 re-run replaces this result, within the limits listed at the "
            "end of this page. The original result stays as recorded.",
            "",
        ]
    )
    header = ("ID", "Question", "Status", "Key result", "Reports", "GPU-h", "CPU-h", "Flags")
    for phase in sorted({row.phase for row in rows}, key=phase_key):
        members = sorted((r for r in rows if r.phase == phase), key=lambda r: id_key(r.id))
        body = []
        for row in members:
            used = sources.compute.get(row.id, Compute())
            body.append(
                [
                    entry_link(row.id, ""),
                    row.question,
                    row.status,
                    row.key_result,
                    report_links(row, ""),
                    hours(used.gpu_h, NONE_MARK),
                    hours(used.cpu_h, NONE_MARK),
                    flags(row, sources, replaced, ""),
                ]
            )
        lines.extend([f"## Phase {phase}", "", *table(header, body), ""])
    lines.extend(["## Superseded results", ""])
    body = [
        [entry_link(old, ""), entry_link(new, ""), scope or NONE_MARK]
        for old, (new, scope) in sorted(replaced.items(), key=lambda item: id_key(item[0]))
    ]
    lines.extend([*table(("Result", "Replaced by", "Limit of the replacement"), body), ""])
    return "\n".join(lines)


def summary_rows(sources: Sources, missing: str) -> list[dict[str, str]]:
    """One flat record per experiment, in ID order."""
    replaced = superseded_map(sources)
    records = []
    for row in sorted(sources.rows, key=lambda r: id_key(r.id)):
        used = sources.compute.get(row.id)
        in_ledger = used is not None
        used = used or Compute()
        records.append(
            {
                "id": row.id,
                "prefix": row.id.split("-")[1],
                "phase": row.phase,
                "date_allocated": row.date,
                "status": row.status,
                "question": row.question,
                "key_result": row.key_result,
                "report_paths": ";".join(row.paths),
                "gpu_h": hours(used.gpu_h, missing),
                "cpu_h": hours(used.cpu_h, missing),
                "train_h": hours(used.train_h, missing),
                "jobs": str(used.jobs) if in_ledger else missing,
                "failed_jobs": str(used.failed_jobs) if in_ledger else missing,
                "test_accesses": str(sources.test_accesses[row.id]),
                "failure_note": "yes" if row.id in sources.failure_ids else "no",
                "in_ablation_table": "yes" if row.id in sources.ablation_ids else "no",
                "superseded_by": replaced[row.id][0] if row.id in replaced else missing,
            }
        )
    return records


def render_csv(sources: Sources) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary_rows(sources, ""))
    return out.getvalue()


def render_md(sources: Sources) -> str:
    body = []
    for record in summary_rows(sources, NONE_MARK):
        cells = [record[name] for name in CSV_COLUMNS]
        cells[0] = entry_link(record["id"], "../")
        cells[7] = record["report_paths"].replace(";", ", ") or NONE_MARK
        body.append(cells)
    lines = [
        "# Experiment summary",
        "",
        "Generated by `scripts/experiment_summary.py` from `research/experiments.md`; do not edit "
        "by hand. One row per registered experiment; the same data is in "
        "`experiment_summary.csv`. Hours are wall-clock time from `reports/compute/ledger.json`, "
        "added over an experiment's rows, and `—` means the ledger has none. `test_accesses` "
        "counts rows in `research/test_access_log.md`.",
        "",
        *table(CSV_COLUMNS, body),
        "",
    ]
    return "\n".join(lines)


def load(root: Path, problems: list[str]) -> Sources:
    """Parse every input and run R1 to R10."""
    text = read_text(root, REGISTRY)
    sources = Sources(rows=parse_rows(text, problems), entries=parse_entries(text, problems))
    parse_ledger(root, sources, problems)
    sources.test_accesses = parse_test_log(read_text(root, TEST_LOG), problems)
    sources.failure_ids = parse_failures(read_text(root, FAILURES))
    sources.ablation_ids = set(parse_ablation_rows(read_text(root, ABLATIONS)))
    check_registry(sources.rows, sources.entries, problems)
    check_status(root, sources, problems)
    check_reports(root, sources, problems)
    check_citations(root, sources, problems)
    return sources


def build(root: Path) -> tuple[dict[str, str], list[str]]:
    """The output files by path, and every problem found. Writes nothing."""
    problems: list[str] = []
    sources = load(root, problems)
    outputs = {
        OUT_ROOT: render_root(sources),
        OUT_CSV: render_csv(sources),
        OUT_MD: render_md(sources),
    }
    check_personal(outputs, problems)
    return outputs, problems


def stale_files(root: Path, outputs: dict[str, str]) -> list[str]:
    """The outputs missing on disk or different from ``outputs``, ignoring CRLF."""
    stale = []
    for path, text in outputs.items():
        full = root / path
        current = full.read_bytes().replace(b"\r\n", b"\n") if full.is_file() else None
        if current != text.encode("utf-8"):
            stale.append(path)
    return stale


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the registry and write its summaries.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="compare, write nothing")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        outputs, problems = build(root)
    except InputError as error:
        print(f"experiment_summary: {error}", file=sys.stderr)
        return 1
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"experiment_summary: {len(problems)} problems, nothing written", file=sys.stderr)
        return 1
    if args.check:
        stale = stale_files(root, outputs)
        for path in stale:
            print(f"experiment_summary: {path} is out of date", file=sys.stderr)
        return 1 if stale else 0
    for path, text in outputs.items():
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
