"""The comparison table of models, datasets, accuracy and cost (Phase 21, plan item 8).

    python scripts/experiment_comparison.py [--root .] [--check]

Builds the table the brief asks for (Model, Dataset, CER, WER, F1, Latency, Params, Model Size,
VRAM) from four generated reports, and from nothing else:

- `reports/external/results.json`: accuracy on the Phase 14 sets, and on the test sets the baselines
  share with Phase 5;
- `reports/baselines/results.json`: the baselines' accuracy on validation sets, and their profiles;
- `reports/efficiency/phase16/results.json`: parameters, file sizes and the GLYPHARIS cost rows;
- `reports/benchmark_results/efficiency_comparison.json`: latency and model size side by side.

Writes `reports/benchmark_results/comparison_table.md` and `.csv`, with LF line ends and no
timestamp. A value no source holds is `n/a`; nothing is estimated, copied from a similar row or
converted from another measurement. Each row names the experiment its accuracy comes from and the
one its latency comes from. The strict track is shown, and IIIT5K also has its alphanumeric headline
track. Oracle runs and the 100-page OmniDocBench subset are left out.

Before writing, the sources are checked against each other: results both accuracy files hold must be
equal, the latency rows must match the profiles they were copied from, and the GLYPHARIS file sizes
must add up. Any disagreement is printed, the exit status is 1 and nothing is written. `--check`
compares the files on disk with what would be written, after turning CRLF into LF, and writes
nothing.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from experiment_summary import InputError, check_personal, read_text, stale_files, table

EXTERNAL: Final = "reports/external/results.json"
BASELINES: Final = "reports/baselines/results.json"
PHASE16: Final = "reports/efficiency/phase16/results.json"
BENCHMARK: Final = "reports/benchmark_results/efficiency_comparison.json"
OUT_MD: Final = "reports/benchmark_results/comparison_table.md"
OUT_CSV: Final = "reports/benchmark_results/comparison_table.csv"

NA: Final = "n/a"
GLYPHARIS: Final = "GLYPHARIS v1"
# v1's default backend (ADR-0011, owner decision of 2026-11-18), and the row its latency comes from.
SHIPPED: Final = "PyTorch, CPU, 10 threads"
LATENCY_SET: Final = "hiertext_val"
# In display order. Anything else in a source is an error, so no set is dropped silently.
SETS: Final = {
    "hiertext_val": "HierText validation",
    "glyphsynth_val": "GlyphSynth validation",
    "hiertext_test": "HierText test",
    "gdc_test": "GDC test",
    "gdc_scan_test": "GDC-scan test",
    "funsd_test": "FUNSD test",
    "iiit5k_test": "IIIT5K test",
    "omnidocbench_test": "OmniDocBench test",
}
TRACKS: Final = {"strict": "strict", "alnum": "alphanumeric (headline)"}
TASKS: Final = {"rec": "recognition", "e2e": "end_to_end"}
LATENCY_UNIT: Final = {"rec": "ms per line", "e2e": "ms per page"}
TOLERANCE: Final = 1e-9

CSV_COLUMNS: Final = (
    "table", "task", "model", "dataset", "track", "samples", "interval_unit",
    "cer_pct", "cer_low", "cer_high",
    "cer_order_aware_pct", "cer_order_aware_low", "cer_order_aware_high",
    "wer_pct", "wer_low", "wer_high",
    "f1_pct", "f1_low", "f1_high",
    "detection_f1_pct", "detection_f1_low", "detection_f1_high",
    "latency_ms", "latency_unit", "latency_setup", "latency_sitting",
    "params", "model_mib", "vram_mib", "accuracy_id", "latency_id",
)  # fmt: skip
METRICS: Final = ("cer", "cer_order_aware", "wer", "f1", "detection_f1")
COST_COLUMNS: Final = (
    ("Params", "params"),
    ("Model MiB", "model_mib"),
    ("VRAM MiB", "vram_mib"),
    ("Accuracy ID", "accuracy_id"),
    ("Latency ID", "latency_id"),
)
RECOGNITION_COLUMNS: Final = (
    ("Model", "model"),
    ("Dataset", "dataset"),
    ("Track", "track"),
    ("Lines", "samples"),
    ("CER %", "cer"),
    ("WER %", "wer"),
    ("F1 %", "f1"),
    ("Latency (ms per line)", "latency_ms"),
    ("Latency setup", "setup"),
    *COST_COLUMNS,
)
END_TO_END_COLUMNS: Final = (
    ("Model", "model"),
    ("Dataset", "dataset"),
    ("Track", "track"),
    ("Pages", "samples"),
    ("Page CER % (bag of lines)", "cer"),
    ("Page CER % (order-aware)", "cer_order_aware"),
    ("Page WER % (order-aware)", "wer"),
    ("E2E F1 %", "f1"),
    ("Detection F1 % (IoU 0.5)", "detection_f1"),
    ("Latency (ms per page)", "latency_ms"),
    ("Latency setup", "setup"),
    *COST_COLUMNS,
)
RUNTIME_COLUMNS: Final = (
    ("Model", "model"),
    ("Task", "task"),
    ("Dataset", "dataset"),
    ("Latency", "latency_ms"),
    ("Unit", "latency_unit"),
    ("Setup", "latency_setup"),
    ("Params", "params"),
    ("Model MiB", "model_mib"),
    ("VRAM MiB (torch peak)", "vram_mib"),
    ("Accuracy ID", "accuracy_id"),
    ("Latency ID", "latency_id"),
)


@dataclass(frozen=True)
class Reports:
    """The four source files, parsed."""

    external: dict[str, Any]
    baselines: dict[str, Any]
    phase16: dict[str, Any]
    benchmark: dict[str, Any]


def load_json(root: Path, path: str) -> dict[str, Any]:
    try:
        loaded: dict[str, Any] = json.loads(read_text(root, path))
    except json.JSONDecodeError as error:
        raise InputError(f"cannot parse {path}: {error}") from error
    return loaded


def load_reports(root: Path) -> Reports:
    return Reports(
        load_json(root, EXTERNAL),
        load_json(root, BASELINES),
        load_json(root, PHASE16),
        load_json(root, BENCHMARK),
    )


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def is_glypharis(record: dict[str, Any]) -> bool:
    return bool(record["system"] == "glypharis")


def accuracy_records(reports: Reports, problems: list[str]) -> list[dict[str, Any]]:
    """Recognition and end-to-end results, each once. Phase 14 re-scored Phase 5's test results."""
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in reports.external["predictions"]:
        if record["task"] not in TASKS:
            continue  # the oracle runs
        key = (record["experiment"], record["task"], record["eval_set"])
        if key in records:
            problems.append(f"{EXTERNAL}: {key} appears twice")
        records[key] = record
    for record in reports.baselines["predictions"]:
        key = (record["experiment"], record["task"], record["eval_set"])
        other = records.get(key)
        if other is None:
            records[key] = record
        elif other["tracks"] != record["tracks"] or other.get("detection") != record.get(
            "detection"
        ):
            problems.append(f"{EXTERNAL} and {BASELINES} differ on {key}")
    for key in records:
        if key[2] not in SETS:
            problems.append(f"{key}: unknown evaluation set")
    return list(records.values())


def latency_rows(reports: Reports, problems: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """The latency rows by (system, task), once they agree with the files they came from."""
    rows = reports.benchmark["rows"]
    if reports.benchmark["eval_set"] != LATENCY_SET:
        problems.append(f"{BENCHMARK}: latency is on {reports.benchmark['eval_set']}")
    if rows != reports.phase16["benchmark"]["rows"]:
        problems.append(f"{BENCHMARK} and {PHASE16} hold different benchmark rows")
    by_key = {(row["system"], row["task"]): row for row in rows}
    if len(by_key) != len(rows):
        problems.append(f"{BENCHMARK}: a system and task appear twice")
    by_experiment = {
        (row["experiment"], row["task"]): row
        for row in rows
        if not row["system"].startswith(GLYPHARIS)
    }
    for profile in reports.baselines["efficiency"]:
        key = (profile["experiment"], profile["task"])
        row = by_experiment.get(key)
        where = f"{BASELINES}: profile {key}"
        if row is None:
            problems.append(f"{where} has no row in {BENCHMARK}")
            continue
        if profile["eval_set"] != LATENCY_SET or profile["batch"] != 1:
            problems.append(f"{where} is not {LATENCY_SET} at batch 1")
        for name in ("mean_ms", "p95_ms", "peak_mem_mib", "model_mib"):
            if not close(profile[name], row[name]):
                problems.append(f"{where}: {name} differs from {BENCHMARK}")
    return by_key


def glypharis_size(reports: Reports, cost: dict[str, Any]) -> float:
    """The MiB of the files a GLYPHARIS cost row loads."""
    files = reports.phase16["gw_eff_006"]["files"]
    suffix = "onnx" if cost["runtime"].startswith("onnxruntime") else "pt"
    size: float = files[f"recognizer.{suffix}"]["mib"]
    if cost["task"] == "e2e":
        size += files[f"detector.{suffix}"]["mib"]
    return size


def glypharis_params(reports: Reports, task: str) -> int:
    parameters = reports.phase16["cost"]["parameters"]
    return int(parameters["recognizer" if task == "rec" else "total"])


def cost_latency(cost: dict[str, Any]) -> float:
    """Milliseconds per line or page. A batched line row is timed per batch in ``mean_ms``."""
    return float(cost["mean_line_ms"] if cost["task"] == "rec" else cost["mean_ms"])


def check_cost_rows(
    reports: Reports, latency: dict[tuple[str, str], dict[str, Any]], problems: list[str]
) -> None:
    """The GLYPHARIS rows of the comparison must be the Phase 16 cost rows."""
    for key, cost in reports.phase16["cost"]["rows"].items():
        where = f"{PHASE16}: cost row {key}"
        if cost["task"] not in TASKS:
            problems.append(f"{where}: unknown task {cost['task']!r}")
            continue
        if cost["task"] == "rec" and not close(cost["mean_ms"] / cost["batch"], cost_latency(cost)):
            problems.append(f"{where}: mean_line_ms is not mean_ms over the batch")
        row = latency.get((f"{GLYPHARIS} {cost['label']}", cost["task"]))
        if row is None:
            continue
        if not close(row["mean_ms"], cost["mean_ms"]):
            problems.append(f"{where}: mean_ms differs from {BENCHMARK}")
        if not close(row["model_mib"], glypharis_size(reports, cost)):
            problems.append(f"{where}: the file sizes do not add up to {BENCHMARK}")


def threads_text(threads: Any) -> str:
    """The CPU setting of a profile, as its record states it."""
    if isinstance(threads, dict):
        counts = [value for value in threads.values() if value is not None]
        threads = counts[0] if counts else "default"
    return f"CPU, {threads} threads"


def fixed(value: float | None, digits: int) -> str:
    return NA if value is None else f"{value:,.{digits}f}"


def metric_cells(name: str, metric: dict[str, Any] | None) -> dict[str, str]:
    """A fraction in the source, as percent cells."""
    if metric is None:
        return {f"{name}_pct": NA, f"{name}_low": NA, f"{name}_high": NA}
    return {
        f"{name}_pct": f"{100 * metric['value']:.2f}",
        f"{name}_low": f"{100 * metric['low']:.2f}",
        f"{name}_high": f"{100 * metric['high']:.2f}",
    }


def empty_row() -> dict[str, str]:
    return dict.fromkeys(CSV_COLUMNS, NA)


def interval_unit(metrics: Sequence[dict[str, Any] | None], where: str, problems: list[str]) -> str:
    units = {metric["unit"] for metric in metrics if metric is not None}
    if len(units) > 1:
        problems.append(f"{where}: intervals over {sorted(units)}")
    return ", ".join(sorted(units)) or NA


def accuracy_row(
    record: dict[str, Any],
    track: str,
    reports: Reports,
    latency: dict[tuple[str, str], dict[str, Any]],
    names: dict[str, str],
    problems: list[str],
) -> dict[str, str]:
    task = record["task"]
    scores = record["tracks"][track]
    experiment = record["experiment"]
    where = f"{experiment} {task} {record['eval_set']} {track}"
    row = empty_row()
    if task == "rec":
        metrics = {"cer": scores["cer"], "wer": scores["wer"]}
        samples = scores["samples"]
    else:
        detection = record.get("detection")
        metrics = {
            "cer": scores.get("page_cer_bag_of_lines"),
            "cer_order_aware": scores.get("page_cer_order_aware"),
            "wer": scores.get("page_wer_order_aware"),
            "f1": scores.get("e2e_f1"),
            "detection_f1": detection["iou_0.5"]["f1"] if detection else None,
        }
        samples = scores["pages"]
    for name in METRICS:
        row.update(metric_cells(name, metrics.get(name)))
    row.update(
        table=TASKS[task],
        task=task,
        dataset=SETS.get(record["eval_set"], record["eval_set"]),
        track=TRACKS[track],
        samples=f"{samples:,}",
        interval_unit=interval_unit(list(metrics.values()), where, problems),
        latency_unit=LATENCY_UNIT[task],
        accuracy_id=experiment,
    )
    if is_glypharis(record):
        row["model"] = GLYPHARIS
        row["params"] = f"{glypharis_params(reports, task):,}"
        cost = latency.get((f"{GLYPHARIS} {SHIPPED}", task))
        setup = SHIPPED
    else:
        row["model"] = names.get(experiment, NA)
        if experiment not in names:
            problems.append(f"{where}: {experiment} has no name in {BENCHMARK}")
        cost = latency.get((row["model"], task))
        setup = threads_text(cost["threads"]) if cost else NA
    if cost is not None:
        row.update(
            latency_ms=fixed(cost["mean_ms"], 1),
            latency_setup=setup,
            latency_sitting=cost["sitting"],
            model_mib=fixed(cost["model_mib"], 2),
            latency_id=cost["experiment"],
        )
    return row


def runtime_rows(reports: Reports) -> list[dict[str, str]]:
    """Every GLYPHARIS cost row: the only rows with VRAM, because only they include a GPU."""
    rows = []
    experiment = glypharis_cost_id(reports)
    for cost in reports.phase16["cost"]["rows"].values():
        vram = cost.get("peak_vram") or {}
        row = empty_row()
        row.update(
            table="runtime",
            task=cost["task"],
            model=GLYPHARIS,
            dataset=SETS[LATENCY_SET],
            latency_ms=fixed(cost_latency(cost), 1),
            latency_unit=LATENCY_UNIT[cost["task"]],
            latency_setup=cost["label"],
            latency_sitting="Phase 16",
            params=f"{glypharis_params(reports, cost['task']):,}",
            model_mib=fixed(glypharis_size(reports, cost), 2),
            vram_mib=fixed(vram.get("max_memory_allocated_mib"), 1),
            latency_id=experiment,
        )
        rows.append(row)
    return rows


def glypharis_cost_id(reports: Reports) -> str:
    ids = {
        row["experiment"]
        for row in reports.benchmark["rows"]
        if row["system"].startswith(GLYPHARIS)
    }
    return ids.pop() if len(ids) == 1 else NA


def sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    order = list(SETS)
    place = order.index(record["eval_set"]) if record["eval_set"] in SETS else len(order)
    return place, 0 if is_glypharis(record) else 1, record["experiment"]


def build_rows(reports: Reports, problems: list[str]) -> list[dict[str, str]]:
    latency = latency_rows(reports, problems)
    check_cost_rows(reports, latency, problems)
    names = {
        row["experiment"]: row["system"]
        for row in reports.benchmark["rows"]
        if not row["system"].startswith(GLYPHARIS)
    }
    records = sorted(accuracy_records(reports, problems), key=sort_key)
    rows = []
    for task in TASKS:
        for track in TRACKS:
            for record in records:
                if record["task"] == task and track in record["tracks"]:
                    rows.append(accuracy_row(record, track, reports, latency, names, problems))
    rows.extend(runtime_rows(reports))
    return rows


def with_interval(row: dict[str, str], name: str) -> str:
    value = row[f"{name}_pct"]
    return value if value == NA else f"{value} [{row[f'{name}_low']}, {row[f'{name}_high']}]"


def setup_cell(row: dict[str, str]) -> str:
    if row["latency_setup"] == NA:
        return NA
    return f"{row['latency_setup']}; {row['latency_sitting']}"


def cell(row: dict[str, str], key: str) -> str:
    if key in METRICS:
        return with_interval(row, key)
    return setup_cell(row) if key == "setup" else row[key]


def section_table(columns: Sequence[tuple[str, str]], rows: list[dict[str, str]]) -> list[str]:
    """A Markdown table of (header, row key) columns."""
    header = [title for title, _ in columns]
    return table(header, [[cell(row, key) for _, key in columns] for row in rows])


def sample_sizes(rows: list[dict[str, str]], glypharis: bool) -> str:
    """How many items each side was scored on, on HierText validation."""
    parts = []
    for table_name, unit in (("recognition", "lines"), ("end_to_end", "pages")):
        counts = {
            r["samples"]
            for r in rows
            if r["table"] == table_name
            and r["dataset"] == SETS[LATENCY_SET]
            and (r["model"] == GLYPHARIS) == glypharis
        }
        if counts:
            parts.append(f"{' or '.join(sorted(counts))} {unit}")
    return " and ".join(parts) or NA


def not_run_lines(reports: Reports, problems: list[str]) -> list[str]:
    names = {}
    for record in reports.external["predictions"]:
        if not is_glypharis(record):
            names[record["system"]] = record["experiment"]
    display = {
        row["experiment"]: row["system"]
        for row in reports.benchmark["rows"]
        if not row["system"].startswith(GLYPHARIS)
    }
    lines = []
    for key, reason in sorted(reports.external["not_run"].items()):
        parts = key.split(" ")
        if len(parts) != 3 or parts[0] not in names or parts[2] not in SETS:
            problems.append(f"{EXTERNAL}: cannot read the not_run key {key!r}")
            continue
        model = display.get(names[parts[0]], parts[0])
        lines.append(f"- {model}, {parts[1]}, {SETS[parts[2]]}: {reason}.")
    return lines


def render_md(rows: list[dict[str, str]], reports: Reports, problems: list[str]) -> str:
    recognition = [r for r in rows if r["table"] == "recognition"]
    end_to_end = [r for r in rows if r["table"] == "end_to_end"]
    runtime = [r for r in rows if r["table"] == "runtime"]
    lines = [
        "# Comparison table",
        "",
        f"Generated by `scripts/experiment_comparison.py` from `{EXTERNAL}`, `{BASELINES}`, "
        f"`{PHASE16}` and `{BENCHMARK}`; do not edit by hand. The same rows are in "
        "`comparison_table.csv`.",
        "",
        "**Latency is measured on HierText validation, in different sittings (Phase 5 for the "
        "baselines, Phase 16 for GLYPHARIS), so no latency difference may be stated from this "
        "table.** A model's latency is the same on every row: it was not measured on that row's "
        "dataset.",
        "",
        "- A cell is `n/a` when these sources hold no value for it. Nothing is estimated.",
        "- Accuracy values are percentages with a 95% bootstrap interval. Intervals are over lines "
        "in the recognition table (words for IIIT5K) and over pages in the end-to-end table.",
        "- The track is strict unless the Track column says otherwise. On IIIT5K the alphanumeric "
        "row is the headline, because the strict track charges case against upper-case labels.",
        "- GLYPHARIS accuracy comes from the strict fp32 evaluation on the GPU. Its latency in "
        f"the first two tables is the {SHIPPED} profile of v1's default backend, so the two "
        "numbers are from different runs.",
        f"- On HierText validation the baselines were scored on {sample_sizes(rows, False)}, and "
        f"GLYPHARIS on {sample_sizes(rows, True)}. The baselines' counts are seeded subsamples, so "
        "those rows are not directly comparable with the GLYPHARIS rows.",
        "- GDC-scan is one document: a case study whose intervals are conditional on it.",
        "- On OmniDocBench the reference unit is a layout block, so bag-of-lines CER and both F1 "
        "values are `n/a`. Its order-aware CER is this project's metric, not the official "
        "protocol.",
        "- The sources give no parameter count for the baselines. VRAM exists only for the "
        "GLYPHARIS GPU runs in the last table; every latency in the first two tables is a CPU run.",
        "",
        "## Recognition (ground-truth line crops)",
        "",
        *section_table(RECOGNITION_COLUMNS, recognition),
        "",
        "## End to end (pages)",
        "",
        *section_table(END_TO_END_COLUMNS, end_to_end),
        "",
        "A model and dataset with no row has no scored result in these files. The runs recorded "
        "as not done:",
        "",
        *not_run_lines(reports, problems),
        "",
        "## GLYPHARIS v1 by runtime",
        "",
        "Cost only: these profiles did not score accuracy, so the accuracy ID is `n/a`. A batched "
        "row's latency is the batch time divided by the lines in the batch.",
        "",
        *section_table(RUNTIME_COLUMNS, runtime),
        "",
    ]
    return "\n".join(lines)


def render_csv(rows: list[dict[str, str]]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def build(root: Path) -> tuple[dict[str, str], list[str]]:
    """The output files by path, and every problem found. Writes nothing."""
    problems: list[str] = []
    reports = load_reports(root)
    rows = build_rows(reports, problems)
    outputs = {OUT_MD: render_md(rows, reports, problems), OUT_CSV: render_csv(rows)}
    check_personal(outputs, problems)
    return outputs, problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the model comparison table.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true", help="compare, write nothing")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        outputs, problems = build(root)
    except (InputError, KeyError, TypeError) as error:
        print(f"experiment_comparison: unusable source: {error!r}", file=sys.stderr)
        return 1
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"experiment_comparison: {len(problems)} problems, nothing written", file=sys.stderr)
        return 1
    if args.check:
        stale = stale_files(root, outputs)
        for path in stale:
            print(f"experiment_comparison: {path} is out of date", file=sys.stderr)
        return 1 if stale else 0
    for path, text in outputs.items():
        full = root / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
