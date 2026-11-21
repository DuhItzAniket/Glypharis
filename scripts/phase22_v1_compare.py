"""The metric inventory and every comparison of the v1 reproduction (Phase 22, items 6 to 13).

    python scripts/phase22_v1_compare.py inventory
    python scripts/phase22_v1_compare.py v1-hashes
    python scripts/phase22_v1_compare.py rehearsal --rerun-root $ROOTS/rerun
    python scripts/phase22_v1_compare.py pass-a
    python scripts/phase22_v1_compare.py line-counts --rerun-root $ROOTS/rerun
    python scripts/phase22_v1_compare.py verdict --rerun-root $ROOTS/rerun

``inventory`` lists, from reports/external/results.json, every metric of the GLYPHARIS blocks: its
path and how it is judged, never its value. ``point`` values must agree within `TOLERANCE`, a
fraction (0.05 points); ``count``, ``flag`` and ``verdict`` entries are compared exactly. Confidence
bounds are shown, not judged, so they are not listed. Provenance, settings and labels are left out,
and so is OmniDocBench's ``match_debug``, whose fallback counts follow the wall clock: items 10 and
13 report those. Each point value's scale is checked: patterns holding a value above 1, a ratio or
a NaN are named under ``scale``. ``--skip-predictions`` leaves the ``predictions`` block unread and
marks the inventory incomplete; the commands that judge values refuse such an inventory, or one
made from another results file.

``v1-hashes`` lists the SHA-256 and size of every file under the three v1 stores. It stops on a file
name that could identify GDC-scan content. The list holds no date, so two lists compare by bytes.

``rehearsal`` scores GW-REC-038 and GW-E2E-006 under the re-run root with the unchanged
`evaluate_baselines.score_file`, compares them with the same two entries of results.json, and
counts the crops and pages whose records differ, ignoring ``time_ms`` and ``seg``. The split is
validation, so no access row is written; the log's bytes are compared to prove it.

``pass-a`` compares Pass A's results.json with Phase 14's: every number exactly, outside the
provenance and path strings, with OmniDocBench's fallback page lists compared on their own.

``line-counts`` is item 12: one logged row per test file, then counts of differing crops and pages.
``verdict`` is item 13: every inventory path of Pass B against Phase 14 at the tolerance.

Phase 14's files are checked against data/manifests/processed/phase14_outputs.json before they are
read, and the re-run's against its root's manifest. A re-run file must carry the wrapper's stamp and
the run identity Phase 14 recorded (model, environment, settings, batch, inputs, tag). Only numbers,
paths and IDs are written, never text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import evaluate_baselines as eb
import phase14_score as ps
import phase22_v1_repro as repro

from glypharis.data.download import sha256_file
from glypharis.eval.access import open_test_access
from glypharis.pipeline import frozen

TOLERANCE = 0.0005  # as a fraction: 0.05 points (Phase 14 item 6)
_EPS = 1e-12  # floating-point noise in a difference, far below the tolerance
RESULTS = Path("reports/external/results.json")
FINAL = Path("reports/final_validation")
INVENTORY = FINAL / "metric_inventory.json"
REHEARSAL = FINAL / "rehearsal.json"
V1_FILES = FINAL / "v1_files_before.json"
PASS_A_RESULTS = FINAL / repro.PASS_A / "results.json"
PASS_A_VERDICT = FINAL / repro.PASS_A / "verdict.json"
PASS_B_RESULTS = FINAL / repro.PASS_B / "results.json"
LINE_COUNTS = FINAL / repro.PASS_B / "line_comparison.json"
REPRODUCTION = FINAL / "v1_reproduction.json"
ACCESS_LOG = Path("research/test_access_log.md")
REASON = "Phase 22 v1 re-run rehearsal scoring (GW-REC-042, GW-E2E-014)"
LINE_REASON = f"Phase 22 v1 line comparison, counts only ({repro.PASS_B})"
EVAL_SET = "hiertext_val"
REHEARSAL_RUNS = {"GW-REC-038": ("rec", "GW-REC-042"), "GW-E2E-006": ("e2e", "GW-E2E-014")}
IGNORED_RECORD_FIELDS = {"time_ms", "seg"}
V1_STORES = ("processed/phase14", "processed/baselines", "processed/baselines_phase14")
SCAN_FILE_NAME = re.compile(r"(^|_)gdc_scan_test(\.meta\.json|\.jsonl|\.log)$")
FALLBACK_PAGES = "text_match_fallback_pages"
LARGEST_SHOWN = 20

IGNORED_KEYS = {
    "settings", "version", "runs", "legacy_meta", "experiment", "system", "task", "eval_set",
    "split", "subset", "unit", "resamples", "rule", "track", "scope", "condition", "baseline",
    "glypharis", "match_debug", "save_name", "provenance",
}  # fmt: skip
COUNT_KEYS = {
    "samples", "expected", "empty_references", "pages", "pages_predicted", "gt_lines", "lines",
    "left_out", "units", "documents",
}  # fmt: skip
VERDICT_KEYS = {"verdict", "H1′ holds", "supported_on_gdc"}
RATIO_KEYS = {"cer_ratio"}
JUDGED = ("point", "count", "flag", "verdict")


# ------------------------------------------------------------------------------ the metrics


def kind_of(parts: tuple[str, ...], value: Any) -> str | None:
    """How a leaf is judged; None when it is not a metric."""
    key = parts[-1]
    if IGNORED_KEYS.intersection(parts):
        return None
    if key in VERDICT_KEYS or "verdicts" in parts:
        return "verdict"
    if key in ("low", "high"):
        return "bound"
    if isinstance(value, bool):
        return "flag"
    if isinstance(value, str):
        return None
    if key in COUNT_KEYS or (isinstance(value, int) and "conditions" in parts):
        return "count"
    return "point"


def _leaves(value: Any, parts: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], Any]]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _leaves(v, (*parts, str(k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _leaves(v, (*parts, str(i)))
    else:
        yield parts, value


def metrics_of(block: Any, label: str = "") -> dict[str, tuple[str, Any]]:
    """path -> (kind, value) for every metric leaf under ``block``."""
    out = {}
    for parts, value in _leaves(block, ()):
        if parts and (kind := kind_of(parts, value)) is not None:
            out[".".join(filter(None, (label, *parts)))] = (kind, value)
    return out


def _label(name: str, row: dict[str, Any], fields: tuple[str, ...]) -> str:
    return f"{name}[{'|'.join(str(row.get(f)) for f in fields)}]"


def _rows(
    name: str, rows: Any, fields: tuple[str, ...], glypharis_only: bool = True
) -> Iterator[tuple[str, Any]]:
    for row in rows or []:
        if not glypharis_only or row.get("system") == ps.GLYPHARIS:
            yield _label(name, row, fields), row


def glypharis_blocks(doc: dict[str, Any], *, skip_predictions: bool) -> Iterator[tuple[str, Any]]:
    """(label, block) for every GLYPHARIS block of a results.json. List rows are labelled by what
    they describe, so a row's position does not matter."""
    fields: tuple[str, ...] = ("experiment", "task", "eval_set")
    if not skip_predictions:
        yield from _rows("predictions", doc.get("predictions"), fields)
    yield from _rows("subset", doc.get("subset"), fields)
    conditions = doc.get("gdc_conditions") or {}
    yield "gdc_conditions.conditions", {"conditions": conditions.get("conditions")}
    fields = ("experiment", "task", "condition")
    yield from _rows("gdc_conditions.rows", conditions.get("rows"), fields)
    yield from _rows("gdc_scan_in_charset", doc.get("gdc_scan_in_charset"), ("experiment",))
    fields = ("experiment", "task")
    rows = doc.get("gdc_scan_in_charset_pages")
    yield from _rows("gdc_scan_in_charset_pages", rows, fields)
    fields = ("glypharis", "baseline", "system", "task", "eval_set", "subset")
    yield from _rows("comparisons", doc.get("comparisons"), fields, glypharis_only=False)
    official = doc.get("official") or {}
    yield from _rows("official.hiertext", official.get("hiertext"), ("experiment",))
    fields = ("experiment", "scope")
    yield from _rows("official.omnidocbench", official.get("omnidocbench"), fields)
    yield "calibration", doc.get("calibration")
    yield "hypotheses", doc.get("hypotheses")


def all_metrics(doc: dict[str, Any], *, skip_predictions: bool) -> dict[str, tuple[str, Any]]:
    """Every metric of the GLYPHARIS blocks. Two rows with one label would hide a metric: stop."""
    out: dict[str, tuple[str, Any]] = {}
    for label, block in glypharis_blocks(doc, skip_predictions=skip_predictions):
        new = metrics_of(block, label)
        repeated = sorted(out.keys() & new.keys())
        if repeated:
            raise SystemExit(f"two rows produce the metric {repeated[0]}")
        out.update(new)
    return out


def _pattern(path: str) -> str:
    return re.sub(r"\.\d+(?=\.|$)", ".#", re.sub(r"\[[^\]]*\]", "[]", path))


def scale_report(metrics: dict[str, tuple[str, Any]]) -> dict[str, Any]:
    """Which point metrics are not plainly fractions, by pattern and with no value."""
    above: set[str] = set()
    nan: set[str] = set()
    ratio: set[str] = set()
    patterns: set[str] = set()
    for path, (kind, value) in metrics.items():
        if kind != "point":
            continue
        pattern = _pattern(path)
        patterns.add(pattern)
        if path.rsplit(".", 1)[-1] in RATIO_KEYS:
            ratio.add(pattern)
        if isinstance(value, int | float):
            if math.isnan(value):
                nan.add(pattern)
            elif abs(value) > 1:
                above.add(pattern)
    return {
        "point_patterns": len(patterns),
        "ratios_not_fractions": sorted(ratio),
        "patterns_with_a_value_above_1": sorted(above - ratio),
        "patterns_with_nan": sorted(nan),
    }


def build_inventory(results: Path, *, skip_predictions: bool) -> dict[str, Any]:
    raw = results.read_bytes()
    doc = json.loads(raw)
    if skip_predictions:
        doc.pop("predictions", None)
    elif not any(e.get("system") == ps.GLYPHARIS for e in doc.get("predictions") or []):
        raise SystemExit("results.json holds no GLYPHARIS entry under predictions")
    metrics = all_metrics(doc, skip_predictions=skip_predictions)
    return {
        "note": "Generated by scripts/phase22_v1_compare.py; do not edit by hand. Paths only.",
        "source": results.as_posix(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "tolerance": TOLERANCE,
        "predictions_enumerated": not skip_predictions,
        "counts": {k: sum(1 for v, _ in metrics.values() if v == k) for k in JUDGED},
        "scale": scale_report(metrics),
        "metrics": {
            k: sorted(p for p, (v, _) in metrics.items() if v == k) for k in JUDGED
        },
    }  # fmt: skip


# ----------------------------------------------------------------------------- the comparison


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _difference(a: Any, b: Any) -> float | None:
    """|a - b| for two numbers; 0 for two None or two NaN; None when they cannot be compared."""
    if (a is None and b is None) or (_is_nan(a) and _is_nan(b)):
        return 0.0
    numeric = isinstance(a, int | float) and isinstance(b, int | float)
    if not numeric or _is_nan(a) or _is_nan(b):
        return None
    return abs(float(a) - float(b))


def compare_metrics(
    reference: dict[str, tuple[str, Any]],
    rerun: dict[str, tuple[str, Any]],
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:
    """Judge ``rerun`` against ``reference``. A metric on one side only fails."""
    failures: list[dict[str, Any]] = []
    checked = {k: 0 for k in JUDGED}
    largest = 0.0
    largest_bound = 0.0
    differences: list[tuple[float, str]] = []
    for path in sorted(set(reference) | set(rerun)):
        if path not in reference or path not in rerun:
            kind = (reference.get(path) or rerun[path])[0]
            if kind != "bound":
                side = "the re-run" if path in reference else "Phase 14"
                failures.append({"path": path, "kind": kind, "problem": f"missing from {side}"})
            continue
        kind, old = reference[path]
        new = rerun[path][1]
        diff = _difference(old, new)
        if kind == "bound":
            largest_bound = max(largest_bound, diff or 0.0)
            continue
        checked[kind] += 1
        if kind == "point":
            if diff:
                differences.append((diff, path))
            if diff is None or diff > tolerance + _EPS:
                failures.append(
                    {"path": path, "kind": kind, "phase14": old, "rerun": new, "difference": diff}
                )
            else:
                largest = max(largest, diff)
        elif old != new and not (_is_nan(old) and _is_nan(new)):
            failures.append(
                {"path": path, "kind": kind, "phase14": old, "rerun": new, "problem": "not equal"}
            )
    differences.sort(key=lambda item: (-item[0], item[1]))
    return {
        "tolerance": tolerance,
        "passed": not failures,
        "checked": checked,
        "largest_point_difference_within_tolerance": largest,
        "largest_bound_difference_not_judged": largest_bound,
        "largest_differences": [
            {"path": path, "difference": diff} for diff, path in differences[:LARGEST_SHOWN]
        ],
        "failures": failures,
    }


def _texts(record: dict[str, Any]) -> list[str]:
    if "lines" in record:
        return [str(ln.get("text")) for ln in record["lines"]]
    return [str(record.get("pred"))]


def _kept(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in IGNORED_RECORD_FIELDS}


def differing_records(phase14: Path, rerun: Path) -> dict[str, int]:
    """Counts only: records (crops or pages) that differ, and those whose text differs."""
    old = {str(r["key"]): r for r in eb.read_records(phase14)}
    new = {str(r["key"]): r for r in eb.read_records(rerun)}
    common = set(old) & set(new)
    return {
        "records": len(common),
        "only_in_one_file": len(set(old) ^ set(new)),
        "differing": sum(_kept(old[k]) != _kept(new[k]) for k in common),
        "text_differing": sum(_texts(old[k]) != _texts(new[k]) for k in common),
    }


def _same_exactly(a: Any, b: Any) -> bool:
    """No tolerance. A bool equals only that bool; NaN equals NaN; None equals only None."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return a == b or (_is_nan(a) and _is_nan(b))
    return bool(type(a) is type(b) and a == b)


def _is_path(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"\S*[/\\]\S*", value))


def _shown_value(value: Any) -> Any:
    """Numbers, flags and None as they are; text and blocks are never copied into a verdict."""
    return "(not shown)" if isinstance(value, str | dict | list) else value


def exact_differences(old: Any, new: Any, parts: tuple[str, ...] = ()) -> Iterator[dict[str, Any]]:
    """Every place two results differ. The fallback page lists are compared on their own, and two
    path strings are never compared."""
    path = ".".join(parts)
    if parts and parts[-1] == FALLBACK_PAGES:
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(old.keys() | new.keys()):
            if key not in old or key not in new:
                side = "Phase 14" if key not in old else "Pass A"
                yield {"path": ".".join((*parts, key)), "problem": f"missing from {side}"}
            else:
                yield from exact_differences(old[key], new[key], (*parts, key))
    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            yield {"path": path, "problem": "lists of different lengths"}
            return
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            yield from exact_differences(a, b, (*parts, str(i)))
    elif not (_is_path(old) and _is_path(new)) and not _same_exactly(old, new):
        yield {"path": path, "phase14": _shown_value(old), "pass_a": _shown_value(new)}


def _fallback_rows(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return list((doc.get("official") or {}).get("omnidocbench") or [])


def _page_lists(row: dict[str, Any]) -> dict[str, set[str]]:
    pages = (row.get("match_debug") or {}).get(FALLBACK_PAGES) or {}
    return {str(k): {str(p) for p in v} for k, v in pages.items()}


def fallback_differences(old: dict[str, Any], new: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Row index -> the pages each OmniDocBench row's timeout fallbacks changed, by cause."""
    out: dict[int, dict[str, Any]] = {}
    new_rows = _fallback_rows(new)
    for i, row in enumerate(_fallback_rows(old)):
        other = new_rows[i] if i < len(new_rows) else {}
        a, b = _page_lists(row), _page_lists(other)
        changed = {
            cause: {
                "only_in_phase14": sorted(a.get(cause, set()) - b.get(cause, set())),
                "only_in_pass_a": sorted(b.get(cause, set()) - a.get(cause, set())),
            }
            for cause in sorted(a.keys() | b.keys())
            if a.get(cause, set()) != b.get(cause, set())
        }
        if changed:
            label = _label("official.omnidocbench", row, ("experiment", "system", "scope"))
            out[i] = {"row": label, "pages": changed}
    return out


def _fallback_row_of(path: str) -> int | None:
    found = re.match(r"official\.omnidocbench\.(\d+)\.", path)
    return int(found.group(1)) if found else None


# ------------------------------------------------------------------------------- the checks


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require_tree() -> dict[str, Any]:
    """Imports from this worktree, and the worktree at the tag every comparison is pinned to."""
    worktree = repro.git_toplevel(Path.cwd())
    names = ("phase14_score", "evaluate_baselines", "run_baselines", "phase22_v1_repro")
    repro.check_imports(worktree, (*names, "glypharis"))
    try:
        pin: dict[str, Any] = frozen.pinned_code_check(repro.WRAPPER_TAG, root=worktree)
    except RuntimeError as exc:
        raise SystemExit(f"the tree is not pinned at {repro.WRAPPER_TAG}: {exc}") from exc
    return pin


def _job(data: Path, experiment: str) -> Any:
    task = REHEARSAL_RUNS[experiment][0]
    path = data / ps.GLYPHARIS_ROOT / experiment / f"{EVAL_SET}.jsonl"
    return ps.Job(path, experiment, ps.GLYPHARIS, task, EVAL_SET, ps.PHASE)


def _test_job(data: Path, mode: str, eval_set: str) -> Any:
    folder = repro.TEST_FOLDERS[mode]
    path = data / ps.GLYPHARIS_ROOT / folder / f"{eval_set}.jsonl"
    return ps.Job(path, folder, ps.GLYPHARIS, mode, eval_set, ps.PHASE)


def verify_jobs(data: Path, jobs: Iterable[Any], what: str) -> None:
    """Every file and its metadata against the root's outputs manifest, or stop."""
    jobs = list(jobs)
    missing = [j.label for j in jobs if not j.path.is_file()]
    if missing:
        raise SystemExit(f"{what}: {missing[0]} is missing")
    unrecorded = ps.check_outputs(data, [ps.file_entry(j.path, data) for j in jobs])
    if unrecorded:
        raise SystemExit(f"{what}: {unrecorded[0]['file']} is not in the outputs manifest")


def verify_files(data: Path, what: str) -> None:
    verify_jobs(data, [_job(data, e) for e in REHEARSAL_RUNS], what)


def check_two_roots(phase14: Path, rerun: Path) -> None:
    if phase14.resolve() == rerun.resolve():
        raise SystemExit("--rerun-root is the Phase 14 data root; nothing would be compared")


def identity_differences(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """What differs in the identity `ext_eval` records: names only, one level deep."""
    a, b = old.get("identity"), new.get("identity")
    if not isinstance(a, dict) or not isinstance(b, dict):
        return ["identity block missing"]
    out = []
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        if x == y:
            continue
        if isinstance(x, dict) and isinstance(y, dict):
            out += [f"{key}.{k}" for k in sorted(x.keys() | y.keys()) if x.get(k) != y.get(k)]
        else:
            out.append(key)
    return out


def run_identity(phase14_job: Any, rerun_job: Any) -> list[str]:
    """Stops unless the file is the wrapper's; returns how its flags differ from Phase 14's."""
    meta = repro.check_rerun_meta(
        ps, rerun_job.path.with_suffix(".meta.json"), rerun_job.experiment
    )
    old_path = phase14_job.path.with_suffix(".meta.json")
    return identity_differences(json.loads(old_path.read_text(encoding="utf-8")), meta)


def check_inventory(path: Path, results_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit("the metric inventory is missing; generate it before any re-run")
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if not doc.get("predictions_enumerated"):
        raise SystemExit("the metric inventory is incomplete: it skipped the predictions block")
    if doc.get("source_sha256") != results_sha256:
        raise SystemExit("the metric inventory was made from another results file")
    return doc


def check_pass_b_finished(data: Path) -> None:
    state = ps.read_pass(data) or {}
    if (state.get("experiment"), state.get("status")) != (repro.PASS_B, "done"):
        raise SystemExit(f"{repro.PASS_B} has not finished under this root")


# ------------------------------------------------------------------------------- the commands


def run_inventory(args: argparse.Namespace) -> int:
    doc = build_inventory(args.results, skip_predictions=args.skip_predictions)
    _write(args.out, doc)
    state = "complete" if doc["predictions_enumerated"] else "INCOMPLETE (predictions skipped)"
    print(f"{args.out.as_posix()}: {state}; {doc['counts']}")
    return 0


def v1_file_list(data: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for store in V1_STORES:
        root = data / store
        if not root.is_dir():
            raise SystemExit(f"{store} is missing under the data root")
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(data).as_posix()
            folder = rel.rsplit("/", 1)[0].lower()
            if "scan" in rel.lower() and ("scan" in folder or not SCAN_FILE_NAME.search(path.name)):
                raise SystemExit(
                    f"a file under {store} may name GDC-scan content; nothing was written"
                )
            files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def run_v1_hashes(args: argparse.Namespace) -> int:
    files = v1_file_list(args.data)
    note = "Generated by scripts/phase22_v1_compare.py; do not edit by hand. Hashes only."
    _write(args.out, {"note": note, "stores": list(V1_STORES), "files": files})
    print(f"{args.out.as_posix()}: {len(files)} files")
    return 0


def _reference(doc: dict[str, Any], experiment: str) -> dict[str, Any]:
    task = REHEARSAL_RUNS[experiment][0]
    want = (experiment, ps.GLYPHARIS, task, EVAL_SET)
    found = [
        e
        for e in doc.get("predictions") or []
        if (e.get("experiment"), e.get("system"), e.get("task"), e.get("eval_set")) == want
    ]
    if len(found) != 1:
        raise SystemExit(f"results.json holds {len(found)} entries for {experiment} {EVAL_SET}")
    return dict(found[0])


def run_rehearsal(args: argparse.Namespace) -> int:
    pin = require_tree()
    raw = args.results.read_bytes()
    check_inventory(args.inventory, _sha256(raw))
    check_two_roots(args.phase14_data, args.rerun_root)
    verify_files(args.phase14_data, "Phase 14")
    verify_files(args.rerun_root, "re-run")
    flags = {
        experiment: run_identity(
            _job(args.phase14_data, experiment), _job(args.rerun_root, experiment)
        )
        for experiment in REHEARSAL_RUNS
    }
    log_before = args.access_log.read_bytes() if args.access_log.is_file() else None
    results = json.loads(raw)
    ctx = ps.FullContext(args.rerun_root, args.access_log, 22, REASON)
    runs: dict[str, Any] = {}
    for experiment, (_, new_id) in REHEARSAL_RUNS.items():
        job = _job(args.rerun_root, experiment)
        entry = eb.score_file(job.path, ctx, meta=ps.glypharis_meta(job))
        verdict = compare_metrics(
            metrics_of(_reference(results, experiment)), metrics_of(entry), TOLERANCE
        )
        phase14_file = _job(args.phase14_data, experiment).path
        runs[new_id] = {
            "folder": experiment,
            "eval_set": EVAL_SET,
            **verdict,
            "passed": verdict["passed"] and not flags[experiment],
            "identity_differences": flags[experiment],
            "records": differing_records(phase14_file, job.path),
        }
    log_after = args.access_log.read_bytes() if args.access_log.is_file() else None
    if log_after != log_before:
        raise SystemExit("the rehearsal changed the test-access log; its result is void")
    passed = all(r["passed"] for r in runs.values())
    _write(
        args.out,
        {
            "note": "Generated by scripts/phase22_v1_compare.py; do not edit by hand.",
            "date": datetime.now(UTC).date().isoformat(),
            "results_sha256": _sha256(raw),
            "wrapper_tag_commit": pin.get("tag_commit"),
            "head": pin.get("head"),
            "passed": passed,
            "runs": runs,
        },
    )
    print(f"{args.out.as_posix()}: {'passed' if passed else 'FAILED'}")
    return 0 if passed else 1


def run_pass_a(args: argparse.Namespace) -> int:
    """Item 10: every number equal, and owner decision 6 for OmniDocBench's wall-clock limit."""
    pin = require_tree()
    old_raw, new_raw = args.results.read_bytes(), args.pass_a.read_bytes()
    old, new = json.loads(old_raw), json.loads(new_raw)
    if new.get("experiment") != repro.PASS_A:
        raise SystemExit(f"{args.pass_a.as_posix()} is not {repro.PASS_A}'s report")
    skip = ("provenance", "experiment")
    differing = list(
        exact_differences(
            {k: v for k, v in old.items() if k not in skip},
            {k: v for k, v in new.items() if k not in skip},
        )
    )
    fallback = fallback_differences(old, new)
    unexplained = [d for d in differing if _fallback_row_of(d["path"]) not in fallback]
    cause = None
    if differing and not unexplained:
        cause = "OmniDocBench official numbers differ and so do their timeout-fallback pages"
    identical = not differing
    _write(
        args.out,
        {
            "note": "Generated by scripts/phase22_v1_compare.py; do not edit by hand.",
            "experiment": repro.PASS_A,
            "date": datetime.now(UTC).date().isoformat(),
            "phase14_sha256": _sha256(old_raw),
            "pass_a_sha256": _sha256(new_raw),
            "wrapper_tag_commit": pin.get("tag_commit"),
            "head": pin.get("head"),
            "identical": identical,
            "cause": cause,
            "differing_count": len(differing),
            "unexplained_count": len(unexplained),
            "differing": differing,
            "omnidocbench_fallback_differences": list(fallback.values()),
        },
    )
    print(f"{args.out.as_posix()}: {'identical' if identical else 'NOT identical'}")
    return 0 if identical else 1


def run_line_counts(args: argparse.Namespace) -> int:
    """Item 12: counts of differing crops and pages. Each file pair is logged before it is read."""
    pin = require_tree()
    if args.out.exists():
        raise SystemExit(f"{args.out.as_posix()} exists: the comparison is written once")
    check_two_roots(args.phase14_data, args.rerun_root)
    check_pass_b_finished(args.rerun_root)
    pairs = [
        (
            mode,
            eval_set,
            _test_job(args.phase14_data, mode, eval_set),
            _test_job(args.rerun_root, mode, eval_set),
        )
        for mode, eval_set in repro.TEST_ORDER
    ]
    verify_jobs(args.phase14_data, [p[2] for p in pairs], "Phase 14")
    verify_jobs(args.rerun_root, [p[3] for p in pairs], "re-run")
    for _, _, old_job, new_job in pairs:
        run_identity(old_job, new_job)
    checkpoint = "glypharis ({}), line comparison"
    logged = 0
    for mode, eval_set, _, _ in pairs:
        spec = ps.EVAL_SETS[eval_set]
        logged += repro.logged_predictions(
            args.access_log,
            spec["benchmark"],
            spec["split"],
            repro.PASS_B,
            checkpoint.format(mode),
        )
    if logged and not args.resume:
        raise SystemExit("the comparison has logged rows already; finish it with --resume")
    if args.resume and not logged:
        raise SystemExit("--resume: the comparison has logged no row")
    reason = LINE_REASON + ps.RESUMED if args.resume else LINE_REASON
    runs: dict[str, Any] = {}
    for mode, eval_set, old_job, new_job in pairs:
        spec = ps.EVAL_SETS[eval_set]
        open_test_access(
            spec["benchmark"],
            spec["split"],
            repro.PASS_B,
            checkpoint.format(mode),
            repro.PHASE,
            reason,
            args.access_log,
        )
        new_id = repro.RERUN_IDS[new_job.experiment]
        runs[f"{new_id} {eval_set}"] = differing_records(old_job.path, new_job.path)
    _write(
        args.out,
        {
            "note": "Generated by scripts/phase22_v1_compare.py; counts only, no text.",
            "experiment": repro.PASS_B,
            "date": datetime.now(UTC).date().isoformat(),
            "wrapper_tag_commit": pin.get("tag_commit"),
            "head": pin.get("head"),
            "ignored_fields": sorted(IGNORED_RECORD_FIELDS),
            "runs": runs,
        },
    )
    print(f"{args.out.as_posix()}: {len(runs)} files compared")
    return 0


def _is_bin_value(path: str) -> bool:
    return re.search(r"\.reliability_t1?\.", path) is not None


def render_verdict(doc: dict[str, Any]) -> str:
    result = doc["tolerance"]
    state = "passed" if doc["passed"] else "FAILED"
    lines = [
        "# v1 reproduction: tolerance verdict (GW-E2E-018)",
        "",
        "Generated by `scripts/phase22_v1_compare.py`; do not edit by hand. Phase 14's results "
        "stay as recorded.",
        "",
        f"**Verdict: {state}.** Limit: {result['tolerance']} as a fraction (0.05 points).",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for kind in JUDGED:
        lines.append(f"| {kind} values checked | {result['checked'][kind]} |")
    lines += [
        f"| Failures, reliability-bin values | {doc['failures']['reliability_bins']} |",
        f"| Failures, all other values | {doc['failures']['other']} |",
        f"| Changed verdicts and flags | {len(doc['changed_verdicts'])} |",
        f"| Runs whose recorded identity differs | {len(doc['identity_differences'])} |",
        "",
        "## Largest point differences",
        "",
        "| Path | Difference |",
        "|---|---|",
    ]
    lines += [f"| `{d['path']}` | {d['difference']:.6g} |" for d in result["largest_differences"]]
    if doc["changed_verdicts"]:
        lines += [
            "",
            "## Changed verdicts and flags",
            "",
            "| Path | Phase 14 | Re-run |",
            "|---|---|---|",
        ]
        lines += [
            f"| `{f['path']}` | {f.get('phase14')} | {f.get('rerun')} |"
            for f in doc["changed_verdicts"]
        ]
    return "\n".join(lines) + "\n"


def run_verdict(args: argparse.Namespace) -> int:
    """Item 13: every inventory path of Pass B against Phase 14."""
    pin = require_tree()
    old_raw, new_raw = args.results.read_bytes(), args.pass_b.read_bytes()
    inventory = check_inventory(args.inventory, _sha256(old_raw))
    old, new = json.loads(old_raw), json.loads(new_raw)
    if new.get("experiment") != repro.PASS_B or new.get("id_map") != repro.RERUN_IDS:
        raise SystemExit(f"{args.pass_b.as_posix()} is not {repro.PASS_B}'s marked report")
    check_two_roots(args.phase14_data, args.rerun_root)
    check_pass_b_finished(args.rerun_root)
    reference = all_metrics(old, skip_predictions=False)
    listed = {p for kind in JUDGED for p in inventory["metrics"][kind]}
    if listed != {p for p, (kind, _) in reference.items() if kind in JUDGED}:
        raise SystemExit("the metric inventory does not list Phase 14's metrics exactly")
    result = compare_metrics(reference, all_metrics(new, skip_predictions=False))
    old_jobs = ps.glypharis_jobs(args.phase14_data)
    new_jobs = ps.glypharis_jobs(args.rerun_root)
    verify_jobs(args.phase14_data, old_jobs, "Phase 14")
    verify_jobs(args.rerun_root, new_jobs, "re-run")
    flags = {}
    for old_job, new_job in zip(old_jobs, new_jobs, strict=True):
        changed = run_identity(old_job, new_job)
        if changed:
            flags[f"{repro.RERUN_IDS[new_job.experiment]} {new_job.eval_set}"] = changed
    failures = result["failures"]
    bins = sum(_is_bin_value(f["path"]) for f in failures)
    passed = bool(result["passed"]) and not flags
    doc = {
        "note": "Generated by scripts/phase22_v1_compare.py; do not edit by hand.",
        "experiment": repro.PASS_B,
        "date": datetime.now(UTC).date().isoformat(),
        "phase14_sha256": _sha256(old_raw),
        "pass_b_sha256": _sha256(new_raw),
        "wrapper_tag_commit": pin.get("tag_commit"),
        "head": pin.get("head"),
        "passed": passed,
        "tolerance": result,
        "failures": {"reliability_bins": bins, "other": len(failures) - bins},
        "changed_verdicts": [f for f in failures if f["kind"] in ("verdict", "flag")],
        "identity_differences": flags,
    }
    _write(args.out, doc)
    report = args.out.with_suffix(".md")
    report.write_text(render_verdict(doc), encoding="utf-8", newline="\n")
    print(f"{args.out.as_posix()}: {'passed' if passed else 'FAILED'}")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="list the metrics of the GLYPHARIS blocks")
    inv.add_argument("--out", type=Path, default=INVENTORY)
    inv.add_argument(
        "--skip-predictions",
        action="store_true",
        help="leave the predictions block unread; the inventory is marked incomplete",
    )
    hashes = sub.add_parser("v1-hashes", help="hash every file of the v1 stores")
    hashes.add_argument("--data", type=Path, default=Path("data"))
    hashes.add_argument("--out", type=Path, default=V1_FILES)
    reh = sub.add_parser("rehearsal", help="score and compare the two validation runs")
    pass_a = sub.add_parser("pass-a", help="item 10: compare Pass A with Phase 14 exactly")
    pass_a.add_argument("--pass-a", type=Path, default=PASS_A_RESULTS)
    pass_a.add_argument("--out", type=Path, default=PASS_A_VERDICT)
    lines = sub.add_parser("line-counts", help="item 12: count differing crops and pages")
    verdict = sub.add_parser("verdict", help="item 13: judge Pass B at the tolerance")
    verdict.add_argument("--pass-b", type=Path, default=PASS_B_RESULTS)
    verdict.add_argument("--inventory", type=Path, default=INVENTORY)
    verdict.add_argument("--out", type=Path, default=REPRODUCTION)
    reh.add_argument("--inventory", type=Path, default=INVENTORY)
    reh.add_argument("--access-log", type=Path, default=ACCESS_LOG)
    reh.add_argument("--out", type=Path, default=REHEARSAL)
    lines.add_argument("--access-log", type=Path, default=ACCESS_LOG)
    lines.add_argument("--out", type=Path, default=LINE_COUNTS)
    lines.add_argument("--resume", action="store_true")
    for command in (reh, lines, verdict):
        command.add_argument("--rerun-root", type=Path, required=True)
        command.add_argument("--phase14-data", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    commands = {
        "inventory": run_inventory,
        "v1-hashes": run_v1_hashes,
        "rehearsal": run_rehearsal,
        "pass-a": run_pass_a,
        "line-counts": run_line_counts,
        "verdict": run_verdict,
    }
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
