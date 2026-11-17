"""Phase 16 scorer: the cost table, the Pareto points, ADR-0011's targets and the four verdicts
(GW-EFF-006 to 011).

    python scripts/phase16_score.py check                    # read and check; write nothing
    python scripts/phase16_score.py score [--allow-missing]  # write the reports

Reads, under the repository root (``--root``), what the Phase 16 tools write:

- GW-EFF-006: ``reports/efficiency/phase16/v1_bundle.json`` (`scripts/export_check.py --frozen
  --report`), ``reference_manifest.json`` and ``reference_check.json``
  (`scripts/phase16_reference.py`), and the bundle itself (``--bundle``, for file sizes and
  hashes);
- GW-EFF-007: the v1 profiles under ``reports/efficiency/phase16/profiles/`` named as in `COST`
  (`scripts/rec_profile.py --experiment` for lines, `scripts/e2e_profile.py --v1` for pages);
- GW-EFF-008, 009 and 011: ``reports/efficiency/phase16/variants/<variant>/`` (``pages.json``,
  ``lines.json``, ``profile_<pages|lines>_10t_<experiment>_run<1|2>.json``,
  `scripts/phase16_variants.py`); each latency sitting has its own files (`SITTINGS`), and a
  profile file must name its sitting's experiment;
- GW-EFF-010: the fresh GW-BASE-002 profile under data/processed/baselines_phase16/ (`scripts/
  run_baselines.py --phase 16`) and ``profiles/pages_onnx_cpu_10t_GW-EFF-010.json``;
- the freeze record and the Phase 5 baseline report (``reports/baselines/results.json``).

Every file's SHA-256 goes into results.json. Before a number is used its file is checked: the
conditions, threads, warm-up and timed counts by Phase 14's `profile_problems` (imported, given
each file's name); then Phase 16's own rules (`run_problems`): the conditions at the end as well,
the thread variables set before torch was imported, the torch pools and every ONNX Runtime
session at the run's threads (sequential execution, arena off, spinning at the default), the v1
code equal to the tag's, no smoke run, and every bundle the GW-EFF-006 bundle by its hash. A
problem in a file makes what rests on it void; the reports list why.

The four confirmatory tests, by their registered rules (docs/PHASE_STATUS.md, Phase 16 items 4 to
6; research/hypotheses.md):

- H7′ latency: v1-onnx and int8-recognizer line profiles of the GW-EFF-008 sitting, in the
  order fp32, INT8, INT8, fp32; each line's time is the mean of its two runs; met if the CI of
  mean(t_INT8 - 0.75 x t_fp32) lies below 0.
- H7′ accuracy: the 5,000 lines, strict track; `paired_ratio_diff_ci(errors_INT8 - 0.005 x
  ref_chars, errors_fp32, ref_chars, "lines")` below 0. H7′ holds only if both halves are met.
- H6′: rec-GW-REC-003 (CRNN) and rec-GW-REC-020 (SVTR), each first passing its export check on
  the 5,000 lines (its ``lines.json`` ended before the sitting began), profiled in the GW-EFF-011
  sitting in the order CRNN, SVTR, SVTR, CRNN; mean(t_SVTR - 1.5 x t_CRNN) below 0.
- H1′ ONNX: Phase 14's rule with the GW-EFF-006 bundle, named by its SHA-256 (``v1_bundle.json``)
  and checked against the references before the sitting began: the fresh GW-BASE-002 profile,
  then the bundle within `SITTING_GAP_HOURS` (`sitting_problems`), the same pages, and the CI of
  the per-page difference below 0.

The Pareto points' latency differences (descriptive) are given only when the GW-EFF-009 sitting
of that set ran as one mirrored sitting (`order_problems` over its runs in the registered order);
line profiles must have the padding their row registers (`ProfileSpec.padding`).

Every other interval is descriptive and carries no verdict word; the README counts the intervals
the reports print. p95 is a point value. ADR-0011's five targets are read against each v1 build.
Bootstrap: `glypharis.eval.stats` defaults (1,000 resamples, seed 20261010, 95% percentile).

Writes reports/efficiency/phase16/ (README.md, cost_table.md, int8.md, pareto.md, pareto.png,
verdicts.md, results.json) and reports/benchmark_results/efficiency_comparison.md and .json, which
set the v1 builds beside the Phase 5 baseline rows of reports/baselines/results.json (different
sittings, no difference stated). ``score`` refuses while a registered file is missing unless
``--allow-missing`` is given; the missing files are then listed and their results left out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from phase14_score import SITTING_GAP_HOURS, iv, profile_problems, sitting_problems
from run_baselines import (
    EXPERIMENTS,
    PROFILE_SAMPLES,
    PROFILE_THREADS,
    SYSTEM_NAMES,
    WARMUP,
    output_path,
    sha256_file,
)

from glypharis.efficiency.threads import THREAD_ENV
from glypharis.efficiency.variants import (
    SITTINGS,
    V1_LABEL,
    VARIANT_LABEL,
    VARIANTS,
    profile_file,
    sitting_order,
)
from glypharis.eval.stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    Interval,
    mean_ci,
    paired_ratio_diff_ci,
    paired_ratios_diff_ci,
    ratio_ci,
    verdict,
)
from glypharis.utils.gitinfo import git_info

NOTE = "Generated by `scripts/phase16_score.py` (GW-EFF-006 to 011); do not edit by hand."
OUT = Path("reports/efficiency/phase16")
BENCH = Path("reports/benchmark_results")
BASELINE_RESULTS = Path("reports/baselines/results.json")
FREEZE_RECORD = Path("configs/final/glypharis_v1.yaml")
BUNDLE_DIR = Path("runs/release/glypharis-v1")  # export_check --frozen --out (git-ignored)
BUNDLE_REPORT = "v1_bundle.json"
REF_MANIFEST = "reference_manifest.json"
REF_CHECK = "reference_check.json"
PROFILES = "profiles"
VARIANT_DIR = "variants"
BUNDLE_FILES = ("detector.pt", "detector.onnx", "recognizer.pt", "recognizer.onnx")

THREADS = PROFILE_THREADS  # every Phase 16 measurement runs at 10 threads
LINES = 5000  # the seeded validation lines (export_check.LINES)
PAGES = 400  # the Phase 5 validation pages
EVAL_SET = "hiertext_val"
BASELINE = "ppocr_mobile"  # GW-BASE-002, H1′'s control
INT8_FACTOR = 0.75
INT8_CER_MARGIN = 0.005
SVTR_FACTOR = 1.5
REFERENCE_ARM = "v1-onnx"  # Pareto differences are taken against v1's fp32 ONNX build
PARETO, H7_SITTING, H6_SITTING = "GW-EFF-009", "GW-EFF-008", "GW-EFF-011"
SITTING_SECTIONS = {H7_SITTING: "H7′", H6_SITTING: "H6′", PARETO: "GW-EFF-009"}
CONFIRMATORY = ("H7′ latency", "H7′ accuracy", "H6′", "H1′ ONNX")
SAME_SITTING = "different sittings, no difference stated"


@dataclass(frozen=True)
class ProfileSpec:
    """One v1 profile file under ``profiles/`` and what it must be."""

    name: str
    task: str  # "rec" (lines) or "e2e" (pages)
    backend: str
    device: str
    threads: int | None  # None: ONNX Runtime's default
    batch: int
    experiment: str
    label: str
    run: str | None = None  # the recognizer run of a line profile, when not v1's
    padding: str | None = None  # a line profile's (rec_profile --padding); None for pages

    @property
    def file(self) -> str:
        return f"{self.name}.json"


COST: tuple[ProfileSpec, ...] = (
    ProfileSpec(
        "lines_torch_cpu_10t",
        "rec",
        "torch",
        "cpu",
        10,
        1,
        "GW-EFF-007",
        "PyTorch, CPU, 10 threads",
        padding="pipeline",
    ),
    ProfileSpec(
        "lines_onnx_cpu_10t",
        "rec",
        "onnx",
        "cpu",
        10,
        1,
        "GW-EFF-007",
        "fp32 ONNX, CPU, 10 threads",
        padding="pipeline",
    ),
    ProfileSpec(
        "lines_onnx_cpu_default",
        "rec",
        "onnx",
        "cpu",
        None,
        1,
        "GW-EFF-007",
        "fp32 ONNX, CPU, default threads",
        padding="pipeline",
    ),
    ProfileSpec(
        "lines_torch_cuda_b1",
        "rec",
        "torch",
        "cuda",
        10,
        1,
        "GW-EFF-007",
        "PyTorch, GPU, batch 1",
        padding="pipeline",
    ),
    ProfileSpec(
        "lines_torch_cuda_b32",
        "rec",
        "torch",
        "cuda",
        10,
        32,
        "GW-EFF-007",
        "PyTorch, GPU, batch 32 (batched as the pipeline batches)",
        padding="pipeline",
    ),
    ProfileSpec(
        "pages_torch_cpu_10t",
        "e2e",
        "torch",
        "cpu",
        10,
        1,
        "GW-EFF-007",
        "PyTorch, CPU, 10 threads",
    ),
    ProfileSpec(
        "pages_onnx_cpu_10t",
        "e2e",
        "onnx",
        "cpu",
        10,
        1,
        "GW-EFF-007",
        "fp32 ONNX, CPU, 10 threads",
    ),
    ProfileSpec(
        "pages_onnx_cpu_default",
        "e2e",
        "onnx",
        "cpu",
        None,
        1,
        "GW-EFF-007",
        "fp32 ONNX, CPU, default threads",
    ),
    ProfileSpec("pages_torch_cuda", "e2e", "torch", "cuda", 10, 1, "GW-EFF-007", "PyTorch, GPU"),
)
H1_GLYPH = ProfileSpec(
    "pages_onnx_cpu_10t_GW-EFF-010",
    "e2e",
    "onnx",
    "cpu",
    10,
    1,
    "GW-EFF-010",
    "fp32 ONNX, CPU, 10 threads (H1′ sitting)",
)
H6_TORCH: tuple[ProfileSpec, ...] = tuple(
    ProfileSpec(
        f"lines_torch_cpu_10t_{run}",
        "rec",
        "torch",
        "cpu",
        10,
        1,
        "GW-EFF-011",
        f"{run} in PyTorch, CPU, 10 threads (information, Phase 7's read_line input)",
        run=run,
        padding="phase7",  # Phase 7's table re-confirmed on its own input tensor
    )
    for run in ("GW-REC-003", "GW-REC-020")
)
# ADR-0011's targets: (name, set, field, limit, strict "<" or "<=").
ADR_TARGETS = (
    ("line mean", "lines", "mean_ms", 25.0, False),
    ("line p95", "lines", "p95_ms", 70.0, False),
    ("page mean", "pages", "mean_ms", 1500.0, False),
    ("page p95", "pages", "p95_ms", 2700.0, False),
    ("peak RAM on pages", "pages", "peak_mem_mib", 1836.0, True),
)
# The v1 builds ADR-0011 is read against: (label, line profile, page profile).
ADR_BUILDS = (
    ("v1 PyTorch, CPU, 10 threads", "lines_torch_cpu_10t", "pages_torch_cpu_10t"),
    ("v1 fp32 ONNX, CPU, 10 threads", "lines_onnx_cpu_10t", "pages_onnx_cpu_10t"),
    ("v1 fp32 ONNX, CPU, default threads", "lines_onnx_cpu_default", "pages_onnx_cpu_default"),
)
# Mirrored orders of the two two-arm sittings: (variant, run) in the registered order.
H7_ORDER = tuple(sitting_order(H7_SITTING, "lines"))  # fp32, INT8, INT8, fp32
H6_ORDER = tuple(sitting_order(H6_SITTING, "lines"))  # CRNN, SVTR, SVTR, CRNN


# ------------------------------------------------------------------------------------ inputs


@dataclass(frozen=True)
class Layout:
    root: Path
    data: Path
    out: Path
    bench: Path
    bundle: Path
    baselines: Path
    record: Path

    @classmethod
    def at(cls, root: Path, data: Path | None = None, bundle: Path | None = None) -> Layout:
        return cls(
            root=root,
            data=data if data is not None else root / "data",
            out=root / OUT,
            bench=root / BENCH,
            bundle=bundle if bundle is not None else root / BUNDLE_DIR,
            baselines=root / BASELINE_RESULTS,
            record=root / FREEZE_RECORD,
        )

    def profile(self, name: str) -> Path:
        return self.out / PROFILES / f"{name}.json"

    def variant(self, name: str, file: str) -> Path:
        return self.out / VARIANT_DIR / name / file

    def baseline_profile(self) -> Path:
        path = output_path(self.data, EXPERIMENTS[BASELINE], BASELINE, "e2e", EVAL_SET, 16)
        return path.with_name(f"profile_{path.stem}.json")


def variant_profile_file(unit: str, run: int, experiment: str = PARETO) -> str:
    """A variant's latency run of ``experiment``'s sitting (`scripts/phase16_variants.py`)."""
    return profile_file(unit, THREADS, experiment, run)


class Inputs:
    """Every file the scorer reads, with its SHA-256; the missing ones are listed."""

    def __init__(self, lay: Layout) -> None:
        self.lay = lay
        self.files: dict[str, str | None] = {}
        self.missing: list[str] = []

    def rel(self, path: Path) -> str:
        for base, prefix in ((self.lay.root, ""), (self.lay.data, "data/")):
            try:
                return prefix + path.resolve().relative_to(base.resolve()).as_posix()
            except ValueError:
                continue
        return path.as_posix()

    def json(self, path: Path, required: bool = True) -> dict[str, Any] | None:
        rel = self.rel(path)
        if not path.is_file():
            self.files[rel] = None
            if required and rel not in self.missing:
                self.missing.append(rel)
            return None
        self.files[rel] = sha256_file(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise SystemExit(f"{rel} is not a JSON object")
        return doc


# ------------------------------------------------------------------------------------ checks


def _time(v: Any) -> datetime | None:
    """An ISO 8601 time with its UTC offset; None when missing, unreadable or without offset."""
    if not isinstance(v, str):
        return None
    try:
        t = datetime.fromisoformat(v)
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def _positive_times(v: Any, n: int) -> bool:
    return (
        isinstance(v, list)
        and len(v) == n
        and all(
            isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) and x > 0
            for x in v
        )
    )


def thread_checks(p: dict[str, Any]) -> list[dict[str, Any]]:
    """The live thread records a profile took before and after its timed items (the profilers
    write ``threads_check``, the variants script ``threads_before`` and ``threads_after``)."""
    checks = p.get("threads_check") or {}
    found = [
        checks.get("before"),
        checks.get("after"),
        p.get("threads_before"),
        p.get("threads_after"),
    ]
    return [c for c in found if isinstance(c, dict)]


def sessions(p: dict[str, Any]) -> list[dict[str, Any]]:
    """Every ONNX Runtime session's options a profile records: one dict (a line profile) or one
    per pipeline part (None for a PyTorch part)."""
    s = p.get("session_options")
    if not isinstance(s, dict):
        return []
    if "intra_op_num_threads" in s:
        return [s]
    return [v for v in s.values() if isinstance(v, dict)]


def onnx_parts(p: dict[str, Any]) -> int:
    """How many ONNX Runtime sessions the profile's build should have."""
    runtime = (p.get("settings") or {}).get("runtime")
    if isinstance(runtime, dict):  # a variant profile: {detector, recognizer}
        return sum(str(v).startswith("onnxruntime") for v in runtime.values())
    if p.get("backend") != "onnx":
        return 0
    return 2 if p.get("task") == "e2e" else 1


def run_problems(
    name: str, p: dict[str, Any], threads: int | None = THREADS, batch: int = 1
) -> list[str]:
    """Why a Phase 16 profile cannot be used (empty if it can): Phase 14's `profile_problems`,
    then the conditions at the end, threads, sessions, the v1 code, a smoke run and the times."""
    out = profile_problems(name, {**p, "batch": 1}, None, threads)  # type: ignore[arg-type]
    if p.get("batch") != batch:
        out.append(f"{name}: batch {p.get('batch')}, not {batch}")
    if p.get("smoke"):
        out.append(f"{name}: a smoke run")
    end = p.get("conditions_end")
    if not isinstance(end, dict):
        out.append(f"{name}: the conditions at the end are not recorded")
    else:
        if end.get("on_ac_power") is not True:
            out.append(f"{name}: not on mains power at the end")
        if "balanced" not in str(end.get("power_plan") or "").lower():
            out.append(f"{name}: not the Balanced power plan at the end")
    t = p.get("threads") or {}
    if threads is None:
        if t.get("requested") != "default":
            out.append(f"{name}: threads {t.get('requested')!r}, not ONNX Runtime's default")
    else:
        if t.get("requested") != threads:
            out.append(f"{name}: {t.get('requested')!r} threads requested, not {threads}")
        if t.get("env_before_torch_import") is not True:
            out.append(f"{name}: the thread variables were not set before torch was imported")
        env = t.get("env") or {}
        if any(env.get(k) != str(threads) for k in THREAD_ENV):
            out.append(f"{name}: thread variables {env}, not {threads}")
        checks = thread_checks(p)
        if len(checks) < 2:
            out.append(f"{name}: the torch pools were not checked before and after")
        for c in checks:
            if (c.get("intra_op"), c.get("inter_op")) != (threads, threads):
                out.append(
                    f"{name}: torch pools {c.get('intra_op')}/{c.get('inter_op')}, not {threads}"
                )
                break
    found = sessions(p)
    if len(found) != onnx_parts(p):
        out.append(f"{name}: {len(found)} ONNX Runtime sessions recorded, {onnx_parts(p)} expected")
    for s in found:
        want = {
            "intra_op_num_threads": threads or 0,
            "execution_mode": "ORT_SEQUENTIAL",
            "enable_cpu_mem_arena": False,
            "allow_spinning": "default",
        }
        wrong = {k: s.get(k) for k, v in want.items() if s.get(k) != v}
        if wrong:
            out.append(f"{name}: ONNX Runtime session options {wrong}, expected {want}")
            break
    if (p.get("v1_code") or {}).get("v1_code") != "equal":
        out.append(f"{name}: the v1 code was not checked equal to the tag's")
    if not _positive_times(p.get("warmup_times_ms"), WARMUP):
        out.append(f"{name}: not {WARMUP} warm-up times")
    if not _positive_times(p.get("times_ms"), PROFILE_SAMPLES):
        out.append(f"{name}: a time is missing (not {PROFILE_SAMPLES} positive times)")
    start, end_t = _time(p.get("started_utc")), _time(p.get("ended_utc"))
    if start is None or end_t is None:
        out.append(f"{name}: started_utc and ended_utc (ISO 8601 with offset) not recorded")
    elif end_t < start:
        out.append(f"{name}: ended_utc is before started_utc")
    return out


def gpu_notes(name: str, p: dict[str, Any]) -> list[str]:
    """The GPU state is recorded on every profile; a machine without nvidia-smi records None,
    which is noted, not void (Phase 16 item 3 lists what voids a sitting)."""
    g = p.get("gpu_state")
    if not isinstance(g, dict):
        return [f"{name}: GPU state not recorded"]
    return [f"{name}: GPU state at the {k} not recorded" for k in ("start", "end") if not g.get(k)]


def order_problems(runs: Sequence[tuple[str, dict[str, Any]]]) -> list[str]:
    """Why ``runs`` (in the registered order) are not one sitting in that order: each starts no
    earlier than the previous one ended and at most `SITTING_GAP_HOURS` after it."""
    out = []
    order = ", ".join(n for n, _ in runs)
    prev: tuple[str, datetime] | None = None
    for name, p in runs:
        start, end = _time(p.get("started_utc")), _time(p.get("ended_utc"))
        if start is None or end is None:
            out.append(f"{name}: started_utc and ended_utc not recorded")
            prev = None
            continue
        if prev is not None:
            if start < prev[1]:
                out.append(
                    f"{name} started before {prev[0]} ended: the registered order is {order}"
                )
            elif start - prev[1] > timedelta(hours=SITTING_GAP_HOURS):
                out.append(
                    f"{name} started more than {SITTING_GAP_HOURS} hours after {prev[0]} "
                    "ended: not one sitting"
                )
        prev = (name, end)
    return out


def overlaps(runs: Sequence[tuple[str, dict[str, Any]]]) -> list[tuple[str, str]]:
    """Pairs of profiles that ran at the same time (one process per run, one run at a time)."""
    spans = []
    for name, p in runs:
        start, end = _time(p.get("started_utc")), _time(p.get("ended_utc"))
        if start is not None and end is not None:
            spans.append((start, end, name))
    spans.sort()
    out = []
    for i, (_, end, name) in enumerate(spans):
        for start2, _, name2 in spans[i + 1 :]:
            if start2 >= end:
                break
            out.append((name, name2))
    return out


def same_items(runs: Sequence[tuple[str, dict[str, Any]]]) -> list[str]:
    """The runs timed the same items in the same order (input sizes, and keys or page ids where
    both runs record them)."""
    out = []
    first_name, first = runs[0]
    for name, p in runs[1:]:
        if p.get("input_sizes") != first.get("input_sizes"):
            out.append(f"{name} and {first_name}: the input sizes differ item by item")
        for k in ("keys", "page_ids"):
            if k in p and k in first and p[k] != first[k]:
                out.append(f"{name} and {first_name}: the {k} differ")
    return out


# ---------------------------------------------------------------------------------- numbers


def summary(times: Sequence[float], unit: str) -> dict[str, Any]:
    t = np.asarray(times, dtype=np.float64)
    return {
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_ci": iv(mean_ci(t, unit)),
    }


def slope_ci(
    lines: Sequence[float],
    times: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval:
    """Least-squares slope of page time on lines per page (ms per line), with a page-bootstrap
    percentile interval; resamples whose lines do not vary are left out."""
    x = np.asarray(lines, dtype=np.float64)
    y = np.asarray(times, dtype=np.float64)
    n = x.size

    def slopes(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        xc = xs - xs.mean(axis=-1, keepdims=True)
        yc = ys - ys.mean(axis=-1, keepdims=True)
        var = (xc * xc).sum(axis=-1)
        cov = (xc * yc).sum(axis=-1)
        return np.divide(cov, var, out=np.full(var.shape, np.nan), where=var > 0)

    value = float(slopes(x[None], y[None])[0])
    idx = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    s = slopes(x[idx], y[idx])
    s = s[np.isfinite(s)]
    q = (1 - level) / 2
    lo, hi = (float(v) for v in np.quantile(s, [q, 1 - q])) if s.size else (value, value)
    return Interval(value, lo, hi, resamples, "pages")


def arm(runs: Sequence[dict[str, Any]], unit: str) -> dict[str, Any]:
    """An arm of two runs: each item's time is the mean of its two runs; both runs are kept."""
    t = np.mean([np.asarray(p["times_ms"], dtype=np.float64) for p in runs], axis=0)
    return {
        "times_ms": [float(v) for v in t],
        **summary(t, unit),
        "run_means_ms": [float(np.mean(p["times_ms"])) for p in runs],
        "peak_mem_mib": max(float(p.get("peak_mem_mib") or 0.0) for p in runs),
        "input_sizes": runs[0].get("input_sizes"),
    }


def _arrays(d: dict[str, Any], *keys: str) -> list[np.ndarray]:
    return [np.asarray(d[k], dtype=np.float64) for k in keys]


def line_accuracy(doc: dict[str, Any]) -> dict[str, Any]:
    err, den = _arrays(doc["per_line"]["strict"], "errors", "ref_chars")
    return {"cer": iv(ratio_ci(err, den, "lines")), "lines": int(err.size)}


def page_accuracy(doc: dict[str, Any]) -> dict[str, Any]:
    c = doc["per_page"]["strict"]
    err, den, tp, pred, gt = _arrays(
        c, "bag_errors", "bag_ref_chars", "e2e_tp", "e2e_pred", "e2e_gt"
    )
    return {
        "page_cer": iv(ratio_ci(err, den, "pages")),
        "e2e_f1": iv(ratio_ci(2 * tp, pred + gt, "pages")),
        "pages": int(err.size),
    }


def model_parameters(record: dict[str, Any]) -> dict[str, int | None]:
    det = record.get("detector", {}).get("parameters")
    rec = record.get("recognizer", {}).get("parameters")
    total = det + rec if isinstance(det, int) and isinstance(rec, int) else None
    return {"detector": det, "recognizer": rec, "total": total}


# ------------------------------------------------------------------------------------ scorer


@dataclass
class Section:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Scorer:
    """Reads every file once and computes every number; `render` turns the result into text."""

    def __init__(self, lay: Layout) -> None:
        self.lay = lay
        self.inputs = Inputs(lay)
        self.sections: dict[str, Section] = {}
        self.record: dict[str, Any] = {}
        self.record_sha: str | None = None
        self.bundle_sha: str | None = None  # the GW-EFF-006 bundle's manifest SHA-256
        self.bundle_files: dict[str, str] = {}
        self.bundle_ok = False  # the bundle passed Phase 17's four checks
        self.bundle_named_utc: str | None = None  # when v1_bundle.json named it by its hash
        self.reference_checked_utc: str | None = None  # when its reference check ended
        self.profiles: dict[str, dict[str, Any] | None] = {}
        # Every variant file as read (None: missing), and its problems (empty: usable).
        self.variants: dict[str, dict[str, dict[str, Any] | None]] = {}
        self.variant_problems: dict[str, dict[str, list[str]]] = {}
        self.overlapping: set[str] = set()  # "<variant>/<profile file>" that ran beside another

    def section(self, name: str) -> Section:
        return self.sections.setdefault(name, Section())

    # -------------------------------------------------------------------------- GW-EFF-006

    def gw_eff_006(self) -> dict[str, Any]:
        sec = self.section("GW-EFF-006")
        lay = self.lay
        if lay.record.is_file():
            self.record = yaml.safe_load(lay.record.read_text(encoding="utf-8"))
            self.record_sha = sha256_file(lay.record)
            self.inputs.files[self.inputs.rel(lay.record)] = self.record_sha
        else:
            self.inputs.missing.append(self.inputs.rel(lay.record))
        bundle = self.inputs.json(lay.out / BUNDLE_REPORT)
        manifest = self.inputs.json(lay.out / REF_MANIFEST)
        check = self.inputs.json(lay.out / REF_CHECK)
        out: dict[str, Any] = {"bundle": None, "reference_check": None, "files": {}}
        if bundle is not None:
            self.bundle_sha = bundle.get("manifest_sha256")
            self.bundle_named_utc = bundle.get("created_utc")
            self.bundle_files = dict(bundle.get("files") or {})
            out["bundle"] = {
                k: bundle.get(k)
                for k in ("bundle", "version", "manifest_sha256", "files", "record_sha256")
            }
            if bundle.get("equals_record") is not True:
                sec.problems.append("v1_bundle.json: the bundle was not checked against the record")
            if bundle.get("record_sha256") != self.record_sha:
                sec.problems.append("v1_bundle.json: built from another freeze record")
            if (bundle.get("v1_code") or {}).get("v1_code") != "equal":
                sec.problems.append("v1_bundle.json: the v1 code was not checked equal")
        out["files"] = self._bundle_files(sec)
        if check is not None:
            out["reference_check"] = self._reference_check(check, manifest, sec)
            self.reference_checked_utc = check.get("ended_utc")
        self.bundle_ok = bool(out["reference_check"] and out["reference_check"]["pass"]) and not (
            sec.problems
        )
        out["bundle_passes"] = self.bundle_ok
        return out

    def _bundle_files(self, sec: Section) -> dict[str, Any]:
        """Size and hash of each bundle file on disk, checked against v1_bundle.json."""
        files: dict[str, Any] = {}
        folder = self.lay.bundle
        if not (folder / "manifest.json").is_file():
            sec.notes.append(
                f"{self.inputs.rel(folder)}: the bundle is not on this machine; file "
                "sizes are not reported"
            )
            return files
        if self.bundle_sha is not None and sha256_file(folder / "manifest.json") != self.bundle_sha:
            sec.problems.append(f"{self.inputs.rel(folder)}: manifest differs from v1_bundle.json")
        for name in BUNDLE_FILES:
            path = folder / name
            if not path.is_file():
                sec.problems.append(f"{self.inputs.rel(path)} is missing")
                continue
            sha = sha256_file(path)
            if self.bundle_files and sha != self.bundle_files.get(name):
                sec.problems.append(f"{self.inputs.rel(path)}: SHA-256 differs from the manifest")
            files[name] = {"sha256": sha, "mib": path.stat().st_size / 2**20}
        return files

    def _reference_check(
        self, check: dict[str, Any], manifest: dict[str, Any] | None, sec: Section
    ) -> dict[str, Any]:
        name = REF_CHECK
        if check.get("complete") is not True or check.get("limit") is not None:
            sec.problems.append(f"{name}: a smoke check (references incomplete)")
        if check.get("bundle_manifest_sha256") != self.bundle_sha:
            sec.problems.append(f"{name}: checked another bundle than v1_bundle.json's")
        if self.bundle_files and check.get("bundle_files") != self.bundle_files:
            sec.problems.append(f"{name}: the bundle's file hashes differ from v1_bundle.json")
        committed = self.lay.out / REF_MANIFEST
        if manifest is None:
            sec.problems.append(f"{REF_MANIFEST}: the references' manifest is not committed")
        elif check.get("reference_manifest_sha256") != sha256_file(committed):
            sec.problems.append(f"{name}: the references differ from the committed manifest")
        if manifest is not None and manifest.get("provenance", {}).get("record_sha256") != (
            self.record_sha
        ):
            sec.problems.append(f"{REF_MANIFEST}: written from another freeze record")
        if (check.get("v1_code") or {}).get("v1_code") != "equal":
            sec.problems.append(f"{name}: the v1 code was not checked equal")
        if (check.get("threads") or {}).get("requested") != THREADS:
            sec.problems.append(f"{name}: not run at {THREADS} threads")
        tol = check.get("tolerances") or {}
        return {
            "pass": check.get("pass") is True,
            "tolerances": tol,
            "started_utc": check.get("started_utc"),
            "ended_utc": check.get("ended_utc"),
            "detector_pages": (check.get("detector") or {}).get("pages"),
            "recognizer_lines": (check.get("recognizer") or {}).get("lines"),
            "end_to_end_pages": (check.get("end_to_end") or {}).get("pages"),
            "page_cer_torch": (check.get("end_to_end") or {}).get("page_cer_torch"),
            "page_cer_onnx": (check.get("end_to_end") or {}).get("page_cer_onnx"),
            "near_tie_differences": (check.get("recognizer") or {}).get("near_tie_differences"),
            "reference_manifest_sha256": check.get("reference_manifest_sha256"),
        }

    # -------------------------------------------------------------------------- GW-EFF-007

    def v1_profile(self, spec: ProfileSpec, sec: Section, required: bool = True) -> dict | None:
        """A v1 profile from the B profilers, checked against its spec; None if missing or void."""
        p = self.inputs.json(self.lay.profile(spec.name), required)
        self.profiles[spec.name] = p
        if p is None:
            return None
        name = spec.file
        probs = run_problems(name, p, spec.threads, spec.batch)
        want = {
            "backend": spec.backend,
            "device": spec.device,
            "experiment": spec.experiment,
            "eval_set": EVAL_SET,
        }
        probs += [f"{name}: {k} {p.get(k)!r}, not {v!r}" for k, v in want.items() if p.get(k) != v]
        if spec.task == "e2e":
            if (p.get("system"), p.get("task"), p.get("build")) != ("glypharis", "e2e", "v1"):
                probs.append(f"{name}: not a v1 page profile")
            if spec.backend == "onnx" and (p.get("bundle") or {}).get("manifest_sha256") != (
                self.bundle_sha
            ):
                probs.append(f"{name}: not the GW-EFF-006 bundle")
            if spec.backend == "torch" and (p.get("provenance") or {}).get("record_sha256") != (
                self.record_sha
            ):
                probs.append(f"{name}: built from another freeze record")
        else:
            mf = p.get("model_file") or {}
            rec = self.record.get("recognizer", {})
            if spec.run is not None:
                if p.get("run") != spec.run:
                    probs.append(f"{name}: run {p.get('run')!r}, not {spec.run}")
            elif spec.backend == "onnx":
                if mf.get("sha256") != self.bundle_files.get("recognizer.onnx"):
                    probs.append(f"{name}: not the GW-EFF-006 bundle's recognizer.onnx")
            elif mf.get("sha256") != rec.get("checkpoint_sha256"):
                probs.append(f"{name}: not the freeze record's recognizer checkpoint")
            if spec.run is None and (p.get("image_height"), p.get("max_width")) != (
                (rec.get("input") or {}).get("height"),
                (rec.get("input") or {}).get("max_width"),
            ):
                probs.append(f"{name}: input height or width cap differs from the record")
            if p.get("precision") != "fp32":
                probs.append(f"{name}: precision {p.get('precision')!r}, not fp32")
            if p.get("padding") != spec.padding:
                probs.append(f"{name}: padding {p.get('padding')!r}, not {spec.padding!r}")
        if spec.device == "cuda" and not isinstance(p.get("peak_vram"), dict):
            probs.append(f"{name}: peak VRAM not recorded")
        sec.notes += gpu_notes(name, p)
        if probs:
            sec.problems += probs
            return None
        return p

    def gw_eff_007(self) -> dict[str, Any]:
        sec = self.section("GW-EFF-007")
        rows: dict[str, Any] = {}
        for spec in COST:
            p = self.v1_profile(spec, sec)
            if p is None:
                rows[spec.name] = None
                continue
            unit = "pages" if spec.task == "e2e" else "lines"
            row: dict[str, Any] = {
                "label": spec.label,
                "task": spec.task,
                "file": self.inputs.rel(self.lay.profile(spec.name)),
                "runtime": p.get("runtime") or (p.get("settings") or {}).get("runtime"),
                "threads": spec.threads,
                "batch": spec.batch,
                **summary(p["times_ms"], "items" if spec.batch > 1 else unit),
                "peak_mem_mib": p.get("peak_mem_mib"),
                "peak_vram": p.get("peak_vram"),
                "started_utc": p.get("started_utc"),
            }
            if spec.task == "rec":
                row["padding"] = p.get("padding")
                row["lines_per_second"] = p.get("lines_per_second")
                row["mean_line_ms"] = float(np.mean(p["times_ms"])) / spec.batch
                row["parameters"] = p.get("parameters")
            else:
                row["mean_lines"] = float(np.mean(p["lines"]))
                row["slope_ms_per_line"] = iv(slope_ci(p["lines"], p["times_ms"]))
                if p.get("stages"):
                    row["stages"] = {
                        s: float(np.mean([d[s] for d in p["stages_ms"]])) for s in p["stages"]
                    }
                    row["stage_sum_max_rel_diff"] = p.get("stage_sum_max_rel_diff")
                    row["file_decode_ms"] = p.get("mean_file_decode_ms")
            rows[spec.name] = row
        params = model_parameters(self.record)
        rec_row = rows.get("lines_torch_cpu_10t")
        if rec_row is not None and rec_row.get("parameters") != params["recognizer"]:
            sec.problems.append(
                f"the line profile counts {rec_row.get('parameters')} recognizer "
                f"parameters, the record {params['recognizer']}"
            )
        return {"rows": rows, "parameters": params}

    # ----------------------------------------------------------------------------- variants

    def variant_doc(
        self, name: str, file: str, sec: Section, experiment: str | None = None
    ) -> dict[str, Any] | None:
        """A variant's file, checked; None if missing or void. A profile must be a run of
        ``experiment``'s sitting."""
        doc = self.inputs.json(self.lay.variant(name, file))
        self.variants.setdefault(name, {})[file] = doc
        if doc is None:
            return None
        label = f"{name}/{file}"
        probs = []
        want_version = V1_LABEL if VARIANTS[name].setting is None else VARIANT_LABEL
        if doc.get("variant") != name or doc.get("version") != want_version:
            probs.append(f"{label}: variant {doc.get('variant')!r}, version {doc.get('version')!r}")
        if doc.get("smoke") and not file.startswith("profile_"):  # run_problems says it for those
            probs.append(f"{label}: a smoke run")
        if (doc.get("eval_set"), doc.get("device")) != (EVAL_SET, "cpu"):
            probs.append(f"{label}: not {EVAL_SET} on the CPU")
        if (doc.get("v1_code") or {}).get("v1_code") != "equal":
            probs.append(f"{label}: the v1 code was not checked equal")
        build = doc.get("build") or {}
        if (build.get("provenance") or {}).get("record_sha256") != self.record_sha:
            probs.append(f"{label}: built from another freeze record")
        bundle = (build.get("variant_json") or {}).get("bundle")
        if name != "v1-torch" and (bundle or {}).get("manifest_sha256") != self.bundle_sha:
            probs.append(f"{label}: not built on the GW-EFF-006 bundle")
        if file == "lines.json" and (doc.get("lines") != LINES or doc.get("limit") is not None):
            probs.append(f"{label}: {doc.get('lines')} lines, not {LINES}")
        if file == "lines.json" and doc.get("batch") != 1:
            # Batch 1, as the line latency is timed: dynamic INT8 scales a whole input tensor.
            probs.append(f"{label}: lines read {doc.get('batch')!r} at a time, not one at a time")
        if file == "pages.json" and (doc.get("pages") != PAGES or doc.get("limit") is not None):
            probs.append(f"{label}: {doc.get('pages')} pages, not {PAGES}")
        if file.startswith("profile_"):
            probs += run_problems(label, doc, THREADS, 1)
            sec.notes += gpu_notes(label, doc)
            if doc.get("kind") != "phase16-variant-profile":
                probs.append(f"{label}: not a variant profile")
            if experiment is not None and doc.get("experiment") != experiment:
                probs.append(f"{label}: experiment {doc.get('experiment')!r}, not {experiment}")
        self.variant_problems.setdefault(name, {})[file] = probs
        if probs:
            sec.problems += probs
            return None
        return doc

    def load_sittings(self) -> None:
        """Every latency run of the three sittings (`SITTINGS`), read and checked once; runs that
        ran at the same time as another are marked (one process per run, one run at a time)."""
        spans: list[tuple[str, dict[str, Any]]] = []
        where: dict[str, str] = {}
        for experiment, units in SITTINGS.items():
            sec = self.section(SITTING_SECTIONS[experiment])
            for unit, names in units.items():
                for name in names:
                    for r in (1, 2):
                        file = variant_profile_file(unit, r, experiment)
                        self.variant_doc(name, file, sec, experiment)
                        raw = (self.variants.get(name) or {}).get(file)
                        if raw is not None:
                            spans.append((f"{name}/{file}", raw))
                            where[f"{name}/{file}"] = experiment
        for first, second in overlaps(spans):
            for label in {where[first], where[second]}:
                self.section(SITTING_SECTIONS[label]).problems.append(
                    f"{first} and {second} ran at the same time"
                )
            self.overlapping |= {first, second}

    def _run(self, name: str, file: str) -> dict[str, Any] | None:
        """A sitting's run as read by `load_sittings`; None if missing, void or overlapping."""
        doc = (self.variants.get(name) or {}).get(file)
        if doc is None or (self.variant_problems.get(name) or {}).get(file):
            return None
        return None if f"{name}/{file}" in self.overlapping else doc

    def variant_arm(
        self, name: str, unit: str, sec: Section, experiment: str = PARETO
    ) -> dict[str, Any] | None:
        runs = [
            (f"{name} run {r}", self._run(name, variant_profile_file(unit, r, experiment)))
            for r in (1, 2)
        ]
        if any(p is None for _, p in runs):
            return None
        good = [(n, p) for n, p in runs if p is not None]
        probs = same_items(good)
        shas = {(p.get("build") or {}).get("variant_json_sha256") for _, p in good}
        if len(shas) != 1:
            probs.append(f"{name}: the two runs used different variant files")
        if probs:
            sec.problems += probs
            return None
        out = arm([p for _, p in good], unit)
        out["runs"] = {
            n: {k: p.get(k) for k in ("started_utc", "ended_utc", "mean_ms")} for n, p in good
        }
        return out

    def pareto_sitting(self, unit: str, sec: Section) -> dict[str, Any]:
        """Whether the GW-EFF-009 runs of ``unit`` ran as one mirrored sitting in the registered
        order (the arms that have both runs; a point cut from the plan is left out)."""
        present = {
            name
            for name in SITTINGS[PARETO][unit]
            if all(
                (self.variants.get(name) or {}).get(variant_profile_file(unit, r)) is not None
                for r in (1, 2)
            )
        }
        order = sitting_order(PARETO, unit, present)
        runs = [
            (f"{n} run {r}", (self.variants.get(n) or {})[variant_profile_file(unit, r)])
            for n, r in order
        ]
        problems = order_problems(runs)
        if problems:
            sec.notes.append(
                f"{PARETO} {unit}: the runs are not one mirrored sitting in the registered order "
                f"({problems[0]}); the latency differences are not given"
            )
        return {"order": [n for n, _ in runs], "one_sitting": not problems, "problems": problems}

    def gw_eff_009(self) -> dict[str, Any]:
        sec = self.section("GW-EFF-009")
        points: list[dict[str, Any]] = []
        arms: dict[tuple[str, str], dict[str, Any] | None] = {}
        acc: dict[tuple[str, str], dict[str, Any] | None] = {}
        for name, spec in VARIANTS.items():
            for unit in spec.measure:
                arms[(name, unit)] = self.variant_arm(name, unit, sec)
                acc[(name, unit)] = self.variant_doc(name, f"{unit}.json", sec)
        sittings = {unit: self.pareto_sitting(unit, sec) for unit in SITTINGS[PARETO]}
        for (name, unit), a in arms.items():
            spec = VARIANTS[name]
            doc = acc[(name, unit)]
            point: dict[str, Any] = {
                "variant": name,
                "set": unit,
                "version": V1_LABEL if spec.setting is None else VARIANT_LABEL,
                "setting": spec.setting,
                "about": spec.about,
                "latency": a,
                "accuracy": None,
                "difference": None,
            }
            if doc is not None:
                point["accuracy"] = line_accuracy(doc) if unit == "lines" else page_accuracy(doc)
                point["changed"] = doc.get("changed")
                point["files"] = (doc.get("build") or {}).get("variant_json", {}).get("files")
            points.append(point)
        for point in points:
            if point["variant"] != REFERENCE_ARM:
                one = sittings[point["set"]]["one_sitting"]
                point["difference"] = self._difference(point, points, acc, sec, one)
        return {"points": points, "reference": REFERENCE_ARM, "sittings": sittings}

    def _difference(
        self,
        point: dict[str, Any],
        points: list[dict[str, Any]],
        acc: dict[tuple[str, str], dict[str, Any] | None],
        sec: Section,
        one_sitting: bool = True,
    ) -> dict[str, Any] | None:
        """The point minus v1's fp32 ONNX build on the same items (descriptive); the latency
        difference only when both arms ran in one mirrored sitting (``one_sitting``)."""
        unit = point["set"]
        ref = next((p for p in points if p["variant"] == REFERENCE_ARM and p["set"] == unit), None)
        if ref is None:
            return None
        out: dict[str, Any] = {}
        a, b = point["latency"], ref["latency"]
        if a is not None and b is not None and not one_sitting:
            out["latency_note"] = "not one mirrored sitting: no latency difference"
        elif a is not None and b is not None:
            if a["input_sizes"] != b["input_sizes"]:
                sec.problems.append(
                    f"{point['variant']} ({unit}): other items than {REFERENCE_ARM}"
                )
            else:
                d = np.asarray(a["times_ms"]) - np.asarray(b["times_ms"])
                out["latency_ms"] = iv(mean_ci(d, unit))
        da, db = acc[(point["variant"], unit)], acc[(REFERENCE_ARM, unit)]
        if da is not None and db is not None:
            if unit == "lines":
                ea, ra = _arrays(da["per_line"]["strict"], "errors", "ref_chars")
                eb, rb = _arrays(db["per_line"]["strict"], "errors", "ref_chars")
                if da["per_line"]["keys"] != db["per_line"]["keys"] or not np.array_equal(ra, rb):
                    sec.problems.append(
                        f"{point['variant']} (lines): other lines than {REFERENCE_ARM}"
                    )
                else:
                    out["cer"] = iv(paired_ratio_diff_ci(ea, eb, ra, "lines"))
            else:
                ca, cb = da["per_page"]["strict"], db["per_page"]["strict"]
                ea, ra, ta, pa, ga = _arrays(
                    ca, "bag_errors", "bag_ref_chars", "e2e_tp", "e2e_pred", "e2e_gt"
                )
                eb, rb, tb, pb, gb = _arrays(
                    cb, "bag_errors", "bag_ref_chars", "e2e_tp", "e2e_pred", "e2e_gt"
                )
                same = da["per_page"]["doc_ids"] == db["per_page"]["doc_ids"]
                if not same or not np.array_equal(ra, rb):
                    sec.problems.append(
                        f"{point['variant']} (pages): other pages than {REFERENCE_ARM}"
                    )
                else:
                    out["page_cer"] = iv(paired_ratio_diff_ci(ea, eb, ra, "pages"))
                    out["e2e_f1"] = iv(
                        paired_ratios_diff_ci(2 * ta, pa + ga, 2 * tb, pb + gb, "pages")
                    )
        return out or None

    # ----------------------------------------------------------------------------- verdicts

    def _sitting(
        self, experiment: str, sec: Section
    ) -> tuple[list[str], dict[str, list[dict[str, Any]]]] | None:
        """The four line runs of ``experiment``'s mirrored sitting (read by `load_sittings`), or
        None if one is missing. Returns why the sitting is void (empty if it is not) and the runs
        by arm."""
        order = sitting_order(experiment, "lines")
        runs: list[tuple[str, dict[str, Any]]] = []
        void: list[str] = []
        for name, r in order:
            file = variant_profile_file("lines", r, experiment)
            doc = (self.variants.get(name) or {}).get(file)
            if doc is None:
                return None
            runs.append((f"{name} run {r}", doc))
            void += (self.variant_problems.get(name) or {}).get(file, [])
            if f"{name}/{file}" in self.overlapping:
                void.append(f"{name}/{file} ran at the same time as another profile")
        new = order_problems(runs) + same_items(runs)
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for (name, _), (_, doc) in zip(order, runs, strict=True):
            by_arm.setdefault(name, []).append(doc)
        for name, docs in by_arm.items():
            if len({(d.get("build") or {}).get("variant_json_sha256") for d in docs}) != 1:
                new.append(f"{name}: the two runs used different variant files")
        sec.problems += new
        return void + new, by_arm

    def h7(self) -> dict[str, dict[str, Any]]:
        sec = self.section("H7′")
        fp32, int8 = "v1-onnx", "int8-recognizer"
        lat: dict[str, Any] = {
            "rule": f"mean_ci of per-line t_INT8 - {INT8_FACTOR} x t_fp32 (each line the mean of "
            "its two runs), order fp32, INT8, INT8, fp32 in one sitting, below 0",
            "order": [f"{n} run {r}" for n, r in H7_ORDER],
        }
        got = self._sitting(H7_SITTING, sec)
        if got is None:
            lat["verdict"] = "missing"
        else:
            void, by_arm = got
            if not void and not self.bundle_ok:
                void.append("the GW-EFF-006 bundle did not pass the reference check")
            if void:
                lat.update(verdict="void", void=void)
            else:
                a32, a8 = arm(by_arm[fp32], "lines"), arm(by_arm[int8], "lines")
                t = np.asarray(a8["times_ms"]) - INT8_FACTOR * np.asarray(a32["times_ms"])
                test = mean_ci(t, "lines")
                lat.update(
                    arms={"fp32": a32, "int8": a8}, test=iv(test), verdict=verdict(test, "below")
                )
        accu: dict[str, Any] = {
            "rule": f"paired_ratio_diff_ci(errors_INT8 - {INT8_CER_MARGIN} x ref_chars, "
            "errors_fp32, ref_chars, 'lines') on the 5,000 lines, strict track, greedy decoding, "
            "below 0",
        }
        docs = {n: (self.variants.get(n) or {}).get("lines.json") for n in (fp32, int8)}
        a, b = docs[int8], docs[fp32]
        if a is None or b is None:
            accu["verdict"] = "missing"
        else:
            void = [
                m
                for n in (fp32, int8)
                for m in (self.variant_problems.get(n) or {}).get("lines.json", [])
            ]
            ea, ra = _arrays(a["per_line"]["strict"], "errors", "ref_chars")
            eb, rb = _arrays(b["per_line"]["strict"], "errors", "ref_chars")
            if a["per_line"]["keys"] != b["per_line"]["keys"] or not np.array_equal(ra, rb):
                void.append("the two arms read other lines")
                sec.problems.append("H7′ accuracy: the two arms read other lines")
            if not self.bundle_ok:
                void.append("the GW-EFF-006 bundle did not pass the reference check")
            if void:
                accu.update(verdict="void", void=void)
            else:
                test = paired_ratio_diff_ci(ea - INT8_CER_MARGIN * ra, eb, ra, "lines")
                accu.update(
                    cer={"fp32": float(eb.sum() / ra.sum()), "int8": float(ea.sum() / ra.sum())},
                    test=iv(test),
                    verdict=verdict(test, "below"),
                )
        int8_doc = a or (self.variants.get(int8) or {}).get(
            variant_profile_file("lines", 1, H7_SITTING)
        )
        quant = ((int8_doc or {}).get("build") or {}).get("variant_json", {}).get("quantization")
        halves = (lat["verdict"], accu["verdict"])
        if halves == ("met", "met"):
            overall = "holds"
        elif "not met" in halves:
            overall = "does not hold"
        elif any(v in ("missing", "void") for v in halves):
            overall = "pending"
        else:
            overall = "not shown"
        return {
            "H7′ latency": lat,
            "H7′ accuracy": accu,
            "H7′": {
                "verdict": overall,
                "rule": "holds only if both halves are met",
                "quantization": quant,
            },
        }

    def h6(self) -> dict[str, Any]:
        sec = self.section("H6′")
        crnn, svtr = "rec-GW-REC-003", "rec-GW-REC-020"
        out: dict[str, Any] = {
            "rule": f"mean_ci of per-line t_SVTR - {SVTR_FACTOR} x t_CRNN (each line the mean of "
            "its two runs), order CRNN, SVTR, SVTR, CRNN in one sitting, fp32 ONNX, 10 threads, "
            "below 0; each model first passes its export check on the 5,000 lines",
            "order": [f"{n} run {r}" for n, r in H6_ORDER],
            "arms": {"CRNN": crnn, "SVTR": svtr},
        }
        checks: dict[str, Any] = {}
        not_tested = []
        for name in (crnn, svtr):
            doc = (self.variants.get(name) or {}).get("lines.json")
            if doc is None:
                checks[name] = None
                continue
            if (self.variant_problems.get(name) or {}).get("lines.json"):
                not_tested.append(f"{name}: its lines file failed its checks (GW-EFF-009)")
            tc = doc.get("torch_check") or {}
            checks[name] = {
                k: tc.get(k)
                for k in (
                    "lines",
                    "logit_max_abs_diff",
                    "identical_text",
                    "near_tie_differences",
                    "pass",
                )
            }
            if tc.get("pass") is not True or tc.get("lines") != LINES:
                not_tested.append(f"{name}: did not pass its export check on the {LINES:,} lines")
            model = ((doc.get("build") or {}).get("settings") or {}).get("recognizer.model") or {}
            checks[name]["height"] = model.get("height")
            if model.get("height") != 32:
                not_tested.append(f"{name}: input height {model.get('height')}, not 32")
        out["export_checks"] = checks
        got = self._sitting(H6_SITTING, sec)
        first = (self.variants.get(H6_ORDER[0][0]) or {}).get(
            variant_profile_file("lines", H6_ORDER[0][1], H6_SITTING)
        )
        began = _time((first or {}).get("started_utc"))
        late = []
        for name in (crnn, svtr):
            doc = (self.variants.get(name) or {}).get("lines.json")
            ended = _time((doc or {}).get("ended_utc"))
            if doc is not None and began is not None and (ended is None or ended > began):
                late.append(
                    f"{name}: its export check on the {LINES:,} lines ended "
                    f"{(doc or {}).get('ended_utc')}, not before the sitting began "
                    f"({(first or {}).get('started_utc')})"
                )
        if got is not None and late:
            sec.problems += late
            got = (got[0] + late, got[1])
        if not_tested:
            out.update(verdict="not tested", reasons=not_tested)
        elif got is None or any(v is None for v in checks.values()):
            out["verdict"] = "missing"
        else:
            void, by_arm = got
            if void:
                out.update(verdict="void", void=void)
            else:
                ac, asv = arm(by_arm[crnn], "lines"), arm(by_arm[svtr], "lines")
                t = np.asarray(asv["times_ms"]) - SVTR_FACTOR * np.asarray(ac["times_ms"])
                test = mean_ci(t, "lines")
                out.update(
                    arms_ms={"CRNN": ac, "SVTR": asv}, test=iv(test), verdict=verdict(test, "below")
                )
        info: dict[str, Any] = {}
        isec = self.section("H6′ information")
        for spec in H6_TORCH:
            p = self.v1_profile(spec, isec, required=False)
            if p is not None:
                info[spec.run or spec.name] = summary(p["times_ms"], "lines")
        out["information_pytorch_10t"] = info
        return out

    def h1(self) -> dict[str, Any]:
        sec = self.section("H1′ ONNX")
        out: dict[str, Any] = {
            "rule": "mean_ci of per-page time(v1 fp32 ONNX, GW-EFF-006 bundle, 10 threads) - "
            "time(GW-BASE-002, 10 threads), same pages in the same order, the baseline first "
            f"and the bundle within {SITTING_GAP_HOURS} hours (Phase 14's rule), below 0",
            "bundle_manifest_sha256": self.bundle_sha,
        }
        base = self.inputs.json(self.lay.baseline_profile())
        gsec = Section()
        glyph = self.v1_profile(H1_GLYPH, gsec)
        raw = self.profiles.get(H1_GLYPH.name)
        sec.notes += gsec.notes
        if base is None or raw is None:
            out["verdict"] = "missing"
            return out
        void = list(gsec.problems)
        void += profile_problems("GW-BASE-002", base, None)
        if (
            base.get("phase"),
            base.get("experiment"),
            base.get("system"),
            base.get("task"),
            base.get("eval_set"),
        ) != (16, EXPERIMENTS[BASELINE], BASELINE, "e2e", EVAL_SET):
            void.append("GW-BASE-002: not the Phase 16 profile of PP-OCRv5 mobile on the pages")
        if (base.get("phase14_settings_check") or {}).get("result") != "equal":
            void.append("GW-BASE-002: settings not checked equal to the Phase 14 profile")
        end = base.get("conditions_end") or {}
        if (
            end.get("on_ac_power") is not True
            or "balanced" not in str(end.get("power_plan") or "").lower()
        ):
            void.append("GW-BASE-002: not on mains power and the Balanced plan at the end")
        sec.notes += gpu_notes("GW-BASE-002", base)
        if base.get("input_sizes") != raw.get("input_sizes"):
            void.append("the input sizes differ page by page")
        void += [m.replace("GW-EFF-005", "GW-EFF-010") for m in sitting_problems(base, raw)]
        if not self.bundle_ok:
            void.append("the GW-EFF-006 bundle did not pass the reference check")
        began = _time(base.get("started_utc"))
        named, checked = _time(self.bundle_named_utc), _time(self.reference_checked_utc)
        if began is not None and (named is None or named > began):
            void.append(
                f"the GW-EFF-006 bundle was not named by its SHA-256 before the sitting "
                f"(v1_bundle.json created_utc {self.bundle_named_utc!r})"
            )
        if began is not None and (checked is None or checked > began):
            void.append(
                f"the bundle's reference check did not end before the sitting "
                f"(reference_check.json ended_utc {self.reference_checked_utc!r})"
            )
        out["bundle_named_utc"] = self.bundle_named_utc
        out["sitting"] = {
            n: {k: p.get(k) for k in ("started_utc", "ended_utc")}
            for n, p in (("GW-BASE-002", base), ("GW-EFF-010", raw))
        }
        out["power_mode"] = {
            n: (p.get("conditions") or {}).get("power_mode")
            for n, p in (("GW-BASE-002", base), ("GW-EFF-010", raw))
        }
        if void or glyph is None:
            sec.problems += void
            out.update(verdict="void", void=void)
            return out
        diff = np.asarray(glyph["times_ms"], dtype=np.float64) - np.asarray(base["times_ms"])
        test = mean_ci(diff, "pages")
        out.update(
            mean_ms={
                "GW-BASE-002": float(np.mean(base["times_ms"])),
                "GW-EFF-010": float(np.mean(glyph["times_ms"])),
            },
            test=iv(test),
            verdict=verdict(test, "below"),
        )
        return out

    # ---------------------------------------------------------------------- ADR and benchmark

    def adr(self, cost: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for label, line, page in ADR_BUILDS:
            src = {"lines": cost["rows"].get(line), "pages": cost["rows"].get(page)}
            targets = []
            for name, unit, key, limit, strict in ADR_TARGETS:
                row = src[unit]
                value = None if row is None else row.get(key)
                ok = None
                if value is not None:
                    ok = value < limit if strict else value <= limit
                targets.append(
                    {"target": name, "limit": limit, "strict": strict, "value": value, "within": ok}
                )
            rows.append({"build": label, "targets": targets})
        return rows

    def benchmark(self, cost: dict[str, Any], e006: dict[str, Any]) -> dict[str, Any]:
        doc = self.inputs.json(self.lay.baselines)
        rows = []
        for e in (doc or {}).get("efficiency", []):
            if e.get("eval_set") != EVAL_SET:
                continue
            rows.append(
                {
                    "system": SYSTEM_NAMES.get(e["system"], e["system"]),
                    "experiment": e.get("experiment"),
                    "task": e.get("task"),
                    "mean_ms": e.get("mean_ms"),
                    "p95_ms": e.get("p95_ms"),
                    "peak_mem_mib": e.get("peak_mem_mib"),
                    "model_mib": e.get("model_mib"),
                    "threads": e.get("threads"),
                    "sitting": "Phase 5",
                }
            )
        files = e006.get("files") or {}

        def mib(*names: str) -> float | None:
            if not all(n in files for n in names):
                return None
            return sum(files[n]["mib"] for n in names)

        for label, line, page in ADR_BUILDS:
            onnx = "ONNX" in label
            ext = "onnx" if onnx else "pt"
            for task, key in (("rec", line), ("e2e", page)):
                row = cost["rows"].get(key)
                if row is None:
                    continue
                names = (
                    (f"recognizer.{ext}",)
                    if task == "rec"
                    else (f"detector.{ext}", f"recognizer.{ext}")
                )
                rows.append(
                    {
                        "system": f"GLYPHARIS {label}",
                        "experiment": "GW-EFF-007",
                        "task": task,
                        "mean_ms": row["mean_ms"],
                        "p95_ms": row["p95_ms"],
                        "peak_mem_mib": row["peak_mem_mib"],
                        "model_mib": mib(*names),
                        "threads": row["threads"] if row["threads"] is not None else "default",
                        "sitting": "Phase 16",
                    }
                )
        return {
            "rows": rows,
            "label": SAME_SITTING,
            "eval_set": EVAL_SET,
            "phase5_results": self.inputs.rel(self.lay.baselines),
        }

    # --------------------------------------------------------------------------------- run

    def run(self) -> dict[str, Any]:
        e006 = self.gw_eff_006()
        cost = self.gw_eff_007()
        self.load_sittings()
        pareto = self.gw_eff_009()
        verdicts = {**self.h7(), "H6′": self.h6(), "H1′ ONNX": self.h1()}
        info = git_info()
        return {
            "generated_by": "scripts/phase16_score.py",
            "experiments": "GW-EFF-006 to 011",
            "git_commit": info.commit,
            "gw_eff_006": e006,
            "cost": cost,
            "adr_0011": self.adr(cost),
            "pareto": pareto,
            "verdicts": verdicts,
            "benchmark": self.benchmark(cost, e006),
            "inputs": dict(sorted(self.inputs.files.items())),
            "missing": sorted(self.inputs.missing),
            "problems": {k: s.problems for k, s in self.sections.items() if s.problems},
            "notes": {k: s.notes for k, s in self.sections.items() if s.notes},
        }


# ------------------------------------------------------------------------------------ render


class Printer:
    """Formats intervals and counts each distinct one printed, confirmatory or descriptive."""

    def __init__(self) -> None:
        self.printed: dict[str, str] = {}

    def ci(
        self,
        key: str,
        d: dict[str, Any] | None,
        scale: float = 1.0,
        digits: int = 1,
        signed: bool = False,
        kind: str = "descriptive",
    ) -> str:
        if d is None:
            return "n/a"
        self.printed.setdefault(key, kind)
        f = f"{{:+,.{digits}f}}" if signed else f"{{:,.{digits}f}}"
        v, lo, hi = (scale * d[k] for k in ("value", "low", "high"))
        return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]"

    def counts(self) -> dict[str, int]:
        kinds = list(self.printed.values())
        return {
            "printed": len(kinds),
            "confirmatory": kinds.count("confirmatory"),
            "descriptive": kinds.count("descriptive"),
        }


def table(head: list[str], rows: Iterable[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def num(v: Any, digits: int = 1, scale: float = 1.0) -> str:
    if v is None:
        return "n/a"
    return f"{scale * float(v):,.{digits}f}"


def _vram(v: dict[str, Any] | None) -> str:
    """torch's peak allocation / this process at the end / the whole GPU at the end."""
    if not v:
        return "n/a"
    return " / ".join(
        num(v.get(k))
        for k in (
            "max_memory_allocated_mib",
            "nvidia_smi_process_mib_end",
            "nvidia_smi_whole_gpu_used_mib_end",
        )
    )


VRAM_HEAD = "Peak VRAM MiB (torch peak / process at end / whole GPU at end, nvidia-smi)"


def render_cost(res: dict[str, Any], pr: Printer) -> str:
    cost = res["cost"]
    rows = cost["rows"]
    out = [NOTE, "", "# GW-EFF-007: what v1 costs on this laptop", ""]
    out += [
        "HierText validation, 10 warm-up and 100 timed items, one process per run; CPU runs at "
        "10 threads unless stated (torch intra and inter 10, each ONNX Runtime session "
        "intra-op 10, sequential). Means carry a descriptive 95% bootstrap interval over the "
        "timed items; p95 is a point value. A line is read with `TextRecognizer.read`, as the "
        "pipeline reads it (sorted by width, split at its line and column caps, at least 64 "
        "padding columns). VRAM: torch's `max_memory_allocated`, then nvidia-smi's memory at "
        "the end for this process (n/a where the Windows driver does not report it) and for the "
        "whole GPU (the display and other programs included).",
        "",
    ]
    p = cost["parameters"]
    out += ["## Parameters and file sizes", ""]
    files = res["gw_eff_006"].get("files") or {}
    out += table(
        ["Part", "Parameters", "Release .pt (MiB)", "fp32 ONNX (MiB)"],
        [
            [
                part,
                num(p[part], 0),
                num((files.get(f"{part}.pt") or {}).get("mib"), 2),
                num((files.get(f"{part}.onnx") or {}).get("mib"), 2),
            ]
            for part in ("detector", "recognizer")
        ]
        + [["total", num(p["total"], 0), "", ""]],
    )
    int8 = next(
        (
            pt
            for pt in res["pareto"]["points"]
            if pt["variant"] == "int8-recognizer" and pt.get("files")
        ),
        None,
    )
    if int8:
        f = int8["files"].get("recognizer.int8.onnx") or {}
        out += [
            f"Dynamic INT8 recognizer (a new version, GW-EFF-008): {num(f.get('bytes'), 2, 2**-20)}"
            " MiB.",
            "",
        ]
    out += ["## Lines (one line per item; batch 32 is 32 lines per item)", ""]
    line_rows = []
    for spec in COST:
        if spec.task != "rec":
            continue
        r = rows.get(spec.name)
        if r is None:
            line_rows.append([spec.label, "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        unit = "items" if spec.batch > 1 else "lines"
        line_rows.append(
            [
                spec.label,
                pr.ci(f"cost/{spec.name}/mean", r["mean_ci"], digits=2) + f" ({unit})",
                num(r["p95_ms"], 2),
                num(r.get("lines_per_second"), 1),
                num(r["peak_mem_mib"]),
                _vram(r.get("peak_vram")),
            ]
        )
    out += table(
        [
            "Build",
            "Mean ms per item",
            "p95 ms",
            "Lines/s",
            "Peak RAM MiB",
            VRAM_HEAD,
        ],
        line_rows,
    )
    out += ["## Pages (one page at a time, lines batched as the pipeline batches them)", ""]
    page_rows = []
    for spec in COST:
        if spec.task != "e2e":
            continue
        r = rows.get(spec.name)
        if r is None:
            page_rows.append([spec.label, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        page_rows.append(
            [
                spec.label,
                pr.ci(f"cost/{spec.name}/mean", r["mean_ci"], digits=0),
                num(r["p95_ms"], 0),
                num(r["peak_mem_mib"]),
                num(r["mean_lines"]),
                pr.ci(f"cost/{spec.name}/slope", r["slope_ms_per_line"], digits=1),
                _vram(r.get("peak_vram")),
            ]
        )
    out += table(
        [
            "Build",
            "Mean ms",
            "p95 ms",
            "Peak RAM MiB",
            "Lines per page",
            "ms per line (least squares, page bootstrap)",
            VRAM_HEAD,
        ],
        page_rows,
    )
    staged = [(s, rows[s.name]) for s in COST if rows.get(s.name) and rows[s.name].get("stages")]
    if staged:
        names = list(staged[0][1]["stages"])
        out += ["## Page time by stage (mean ms per page)", ""]
        out += [
            "Read off one clock per page; the stages sum to the page time, and each page's sum "
            "was checked within 1% of a second clock around the call by the profiler. As in "
            "GW-EFF-005 and the baselines' profiles, the page time starts from the decoded "
            "image: `to_rgb` is the RGB array made from it, and reading and decoding the image "
            "file happen before the clock starts (their mean is the last column, outside the "
            "page time).",
            "",
        ]
        body = []
        for spec, r in staged:
            total = sum(r["stages"].values())
            body.append(
                [
                    spec.label,
                    *(
                        f"{num(r['stages'][n])} ({100 * r['stages'][n] / total:.0f}%)"
                        for n in names
                    ),
                    num(total),
                    num(r["mean_ms"]),
                    num(r.get("file_decode_ms")),
                ]
            )
        out += table(
            ["Build", *names, "Sum", "Page mean", "Image file decode (outside the page time)"],
            body,
        )
    return "\n".join(out)


def _verdict_block(
    name: str, v: dict[str, Any], pr: Printer, scale: float, digits: int, unit: str
) -> list[str]:
    out = [f"## {name}: {v['verdict']}", "", f"Rule: {v['rule']}.", ""]
    if "test" in v:
        out += [
            f"Interval: {pr.ci(name, v['test'], scale, digits, True, 'confirmatory')} {unit}.",
            "",
        ]
    for k in ("void", "reasons"):
        if v.get(k):
            out += [
                f"{'Void' if k == 'void' else 'Not tested'}:",
                "",
                *(f"- {m}" for m in v[k]),
                "",
            ]
    return out


def render_verdicts(res: dict[str, Any], pr: Printer) -> str:
    v = res["verdicts"]
    out = [NOTE, "", "# Phase 16's four confirmatory tests", ""]
    out += [
        "Each is met if its interval lies below 0, not met if above, otherwise not shown. Void: "
        "a recorded condition failed, and the sitting is repeated (research/failures.md); "
        "missing: its files are not there yet.",
        "",
    ]
    out += table(["Test", "Verdict"], [[k, v[k]["verdict"]] for k in (*CONFIRMATORY, "H7′")])
    out += _verdict_block("H7′ latency", v["H7′ latency"], pr, 1.0, 2, "ms per line")
    lat = v["H7′ latency"]
    if "arms" in lat:
        out += table(
            ["Arm", "Mean ms (two runs averaged)", "Run means ms", "p95 ms"],
            [
                [
                    k,
                    num(a["mean_ms"], 2),
                    ", ".join(num(m, 2) for m in a["run_means_ms"]),
                    num(a["p95_ms"], 2),
                ]
                for k, a in lat["arms"].items()
            ],
        )
    out += _verdict_block("H7′ accuracy", v["H7′ accuracy"], pr, 100.0, 3, "points")
    if "cer" in v["H7′ accuracy"]:
        c = v["H7′ accuracy"]["cer"]
        out += [f"Strict line CER: fp32 {100 * c['fp32']:.3f}%, INT8 {100 * c['int8']:.3f}%.", ""]
    out += [f"**H7′: {v['H7′']['verdict']}** (holds only if both halves are met).", ""]
    out += _verdict_block("H6′", v["H6′"], pr, 1.0, 2, "ms per line")
    h6 = v["H6′"]
    if h6.get("export_checks"):
        out += table(
            ["Model", "Lines", "Max logit diff", "Identical text", "Near ties", "Height", "Pass"],
            [
                [
                    k,
                    num(c.get("lines"), 0),
                    f"{c['logit_max_abs_diff']:.2e}"
                    if c.get("logit_max_abs_diff") is not None
                    else "n/a",
                    num(c.get("identical_text"), 0),
                    num(c.get("near_tie_differences"), 0),
                    str(c.get("height")),
                    str(c.get("pass")),
                ]
                for k, c in h6["export_checks"].items()
                if c is not None
            ],
        )
    if "arms_ms" in h6:
        out += table(
            ["Arm", "Mean ms (two runs averaged)", "Run means ms"],
            [
                [k, num(a["mean_ms"], 2), ", ".join(num(m, 2) for m in a["run_means_ms"])]
                for k, a in h6["arms_ms"].items()
            ],
        )
    if h6.get("information_pytorch_10t"):
        out += ["PyTorch at 10 threads (information, not part of the test):", ""]
        out += table(
            ["Model", "Mean ms", "p95 ms"],
            [
                [k, num(s["mean_ms"], 2), num(s["p95_ms"], 2)]
                for k, s in h6["information_pytorch_10t"].items()
            ],
        )
    h1 = v["H1′ ONNX"]
    out += _verdict_block("H1′ ONNX", h1, pr, 1.0, 1, "ms per page")
    out += [
        f"Bundle: manifest SHA-256 `{h1.get('bundle_manifest_sha256')}`; the verdict applies "
        "to this bundle only.",
        "",
    ]
    if "mean_ms" in h1:
        out += [
            f"Mean page time: GW-BASE-002 {num(h1['mean_ms']['GW-BASE-002'])} ms, v1 fp32 "
            f"ONNX {num(h1['mean_ms']['GW-EFF-010'])} ms.",
            "",
        ]
    if h1.get("sitting"):
        out += table(
            ["Profile", "Started (UTC)", "Ended (UTC)", "Power mode"],
            [
                [n, str(s["started_utc"]), str(s["ended_utc"]), str(h1["power_mode"].get(n))]
                for n, s in h1["sitting"].items()
            ],
        )
    return "\n".join(out)


def render_int8(res: dict[str, Any], pr: Printer) -> str:
    v = res["verdicts"]
    out = [NOTE, "", "# GW-EFF-008: dynamic INT8 of the v1 fp32 ONNX recognizer (H7′)", ""]
    out += [
        "The INT8 recognizer is a new version, not v1. Recipe: ONNX Runtime's "
        "`quant_pre_process` (defaults), then `quantize_dynamic(op_types_to_quantize="
        '["MatMul"], per_channel=False, reduce_range=False, weight_type=QInt8, '
        'extra_options={"MatMulConstBOnly": True})`; the attention products have no '
        "constant input and stay fp32.",
        "",
    ]
    q = v["H7′"].get("quantization")
    if q:
        out += table(
            ["", "fp32", "INT8"],
            [
                [
                    "file (MiB)",
                    num((q.get("input") or {}).get("bytes"), 2, 2**-20),
                    num((q.get("output") or {}).get("bytes"), 2, 2**-20),
                ],
                [
                    "SHA-256",
                    f"`{(q.get('input') or {}).get('sha256')}`",
                    f"`{(q.get('output') or {}).get('sha256')}`",
                ],
                *(
                    [
                        k,
                        str((q.get("nodes_fp32") or {}).get(k)),
                        str((q.get("nodes_int8") or {}).get(k)),
                    ]
                    for k in sorted(set(q.get("nodes_fp32") or {}) | set(q.get("nodes_int8") or {}))
                ),
            ],
        )
        out += [f"onnxruntime {q.get('onnxruntime')}, onnx {q.get('onnx')}.", ""]
    for half, digits, scale, unit in (
        ("H7′ latency", 2, 1.0, "ms per line"),
        ("H7′ accuracy", 3, 100.0, "points"),
    ):
        h = v[half]
        line = f"- {half}: **{h['verdict']}**"
        if "test" in h:
            line += f", {pr.ci(half, h['test'], scale, digits, True, 'confirmatory')} {unit}"
        out.append(line)
    out += [f"- H7′: **{v['H7′']['verdict']}**", ""]
    pts = [p for p in res["pareto"]["points"] if p["variant"] == "int8-recognizer"]
    for p in pts:
        d = p.get("difference") or {}
        if p["set"] == "pages" and d:
            key = "pareto/int8-recognizer/pages"  # the keys pareto.md prints them under
            cer = pr.ci(f"{key}/d_cer", d.get("page_cer"), 100, 2, True)
            ms = pr.ci(f"{key}/d_latency", d.get("latency_ms"), 1, 1, True)
            out += [
                "On the pages (GW-EFF-009, descriptive), INT8 minus v1 fp32 ONNX: page CER "
                f"{cer} points, page time {ms} ms.",
                "",
            ]
    out += ["Details of the latency sitting and the line CERs: verdicts.md.", ""]
    return "\n".join(out)


def render_pareto(res: dict[str, Any], pr: Printer) -> str:
    pts = res["pareto"]["points"]
    out = [NOTE, "", "# GW-EFF-009: accuracy against latency", ""]
    out += [
        'Every point but v1 is a new version ("variant (new version, not v1)"), measured on '
        "validation only; none replaces v1. Latency: fp32 ONNX at 10 threads unless the "
        "variant says otherwise, each item the mean of its two runs. Intervals are "
        "descriptive; differences are the point minus v1's fp32 ONNX build on the same items. "
        "Plot: pareto.png.",
        "",
    ]
    for unit, head in (
        (
            "pages",
            "Pages: page CER and E2E F1 on the 400 Phase 5 pages (strict), "
            "page time on the 110 profile pages",
        ),
        (
            "lines",
            "Lines: line CER on the 5,000 seeded lines (strict, greedy, one line at a time), "
            "line time on 100 `rec_samples` lines",
        ),
    ):
        out += [f"## {head}", ""]
        sitting = (res["pareto"].get("sittings") or {}).get(unit) or {}
        if sitting.get("one_sitting") is False:
            out += [
                f"The {unit} runs of GW-EFF-009 are not one mirrored sitting in the registered "
                f"order ({sitting['problems'][0]}), so no latency difference (Δ ms) is given.",
                "",
            ]
        rows = []
        for p in pts:
            if p["set"] != unit:
                continue
            a, lat, d = p.get("accuracy") or {}, p.get("latency"), p.get("difference") or {}
            key = f"pareto/{p['variant']}/{unit}"
            cer = a.get("page_cer" if unit == "pages" else "cer")
            row = [p["variant"], p["version"], pr.ci(f"{key}/cer", cer, 100, 2)]
            if unit == "pages":
                row.append(pr.ci(f"{key}/f1", a.get("e2e_f1"), 100, 2))
            row += [
                pr.ci(
                    f"{key}/mean", lat["mean_ci"] if lat else None, 1, 1 if unit == "pages" else 2
                ),
                num(lat["p95_ms"], 1 if unit == "pages" else 2) if lat else "n/a",
                num(lat["peak_mem_mib"]) if lat else "n/a",
                pr.ci(
                    f"{key}/d_cer", d.get("page_cer" if unit == "pages" else "cer"), 100, 2, True
                ),
                pr.ci(
                    f"{key}/d_latency", d.get("latency_ms"), 1, 1 if unit == "pages" else 2, True
                ),
            ]
            rows.append(row)
        head_row = ["Point", "Version", "CER %"] + (["E2E F1 %"] if unit == "pages" else [])
        head_row += ["Mean ms", "p95 ms", "Peak RAM MiB", "Δ CER (points)", "Δ ms"]
        out += table(head_row, rows)
    return "\n".join(out)


def plot_pareto(res: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=120)
    for ax, unit, xlabel, ylabel in (
        (axes[0], "pages", "page time, mean ms (110 profile pages)", "page CER, % (400 pages)"),
        (axes[1], "lines", "line time, mean ms (100 lines)", "line CER, % (5,000 lines)"),
    ):
        for k, p in enumerate(res["pareto"]["points"]):
            lat, acc = p.get("latency"), p.get("accuracy")
            if p["set"] != unit or not lat or not acc:
                continue
            cer = acc["page_cer" if unit == "pages" else "cer"]
            x, y = lat["mean_ms"], 100 * cer["value"]
            v1 = p["version"] == V1_LABEL
            colour = "#c0392b" if v1 else "#1f5fa8"
            ax.errorbar(
                x,
                y,
                xerr=[[x - lat["mean_ci"]["low"]], [lat["mean_ci"]["high"] - x]],
                yerr=[[y - 100 * cer["low"]], [100 * cer["high"] - y]],
                fmt="none",
                ecolor=colour,
                lw=0.8,
                alpha=0.6,
            )
            ax.scatter(
                [x], [y], marker="*" if v1 else "o", s=180 if v1 else 40, color=colour, zorder=3
            )
            label = f"{p['variant']} (v1)" if v1 else f"{p['variant']} (new version)"
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(5, 4 if k % 2 else -9),
                fontsize=7,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(unit)
        ax.grid(True, lw=0.4, alpha=0.5)
    handles = [
        Line2D([], [], marker="*", color="#c0392b", ls="", markersize=12, label="v1"),
        Line2D([], [], marker="o", color="#1f5fa8", ls="", label=VARIANT_LABEL),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def render_benchmark(res: dict[str, Any]) -> str:
    b = res["benchmark"]
    out = [NOTE, "", "# Efficiency: GLYPHARIS v1 beside the Phase 5 baselines", ""]
    out += [
        f"HierText validation, CPU, batch 1, 10 warm-up and 100 timed items. The baseline rows "
        f"come from {b['phase5_results']} (Phase 5); the v1 rows from Phase 16 (GW-EFF-007). "
        f"They were measured in {SAME_SITTING}.",
        "",
    ]
    for task, head in (("rec", "Lines"), ("e2e", "Pages")):
        out += [f"## {head}", ""]
        out += table(
            ["System", "ID", "Mean ms", "p95 ms", "Peak RAM MiB", "Model MiB", "Sitting"],
            [
                [
                    r["system"],
                    str(r["experiment"]),
                    num(r["mean_ms"], 1),
                    num(r["p95_ms"], 1),
                    num(r["peak_mem_mib"]),
                    num(r["model_mib"], 2),
                    r["sitting"],
                ]
                for r in b["rows"]
                if r["task"] == task
            ],
        )
    return "\n".join(out)


def render_readme(res: dict[str, Any], counts: dict[str, int]) -> str:
    out = [NOTE, "", "# Phase 16: efficiency of the frozen v1 and its variants", ""]
    out += [
        "Validation data only, on this laptop. v1 is the frozen build (configs/final/"
        "glypharis_v1.yaml at the phase14-eval code); every other point is a new version.",
        "",
    ]
    out += [
        "| File | What |",
        "|---|---|",
        "| cost_table.md | GW-EFF-007: parameters, sizes, line and page latency, memory, the "
        "stage split |",
        "| int8.md | GW-EFF-008: the INT8 recognizer and H7′ |",
        "| pareto.md, pareto.png | GW-EFF-009: accuracy against latency |",
        "| verdicts.md | the four confirmatory tests (H7′ latency and accuracy, H6′, H1′ ONNX) |",
        "| results.json | every number above, the input files' SHA-256, problems and notes |",
        "| ../../benchmark_results/efficiency_comparison.md | v1 beside the Phase 5 baselines |",
        "",
    ]
    e = res["gw_eff_006"]
    rc = e.get("reference_check") or {}
    out += ["## GW-EFF-006: the v1 fp32 ONNX bundle", ""]
    outcome = ("pass" if rc.get("pass") else "fail") if rc else "missing"
    out += [
        f"Bundle manifest SHA-256 `{(e.get('bundle') or {}).get('manifest_sha256')}`; Phase 17's "
        f"four checks against the PyTorch references: **{outcome}** (usable for the ONNX "
        f"results: {e.get('bundle_passes')}).",
        "",
    ]
    if rc.get("tolerances"):
        out += table(
            ["Check", "Limit", "Measured", "Pass"],
            [
                [k, str(t.get("limit")), str(t.get("measured")), str(t.get("pass"))]
                for k, t in rc["tolerances"].items()
            ],
        )
    out += ["## Verdicts", ""]
    out += table(
        ["Test", "Verdict"], [[k, res["verdicts"][k]["verdict"]] for k in (*CONFIRMATORY, "H7′")]
    )
    out += [
        f"Intervals printed in these reports: {counts['printed']} ({counts['confirmatory']} "
        f"confirmatory, {counts['descriptive']} descriptive). Only the confirmatory ones carry "
        "a verdict.",
        "",
    ]
    out += ["## ADR-0011's targets against each v1 build", ""]
    out += [
        "Each v1 build's value against ADR-0011's targets, within or over. The targets are not "
        "tightened here; that is the owner's call.",
        "",
    ]
    head = [
        "Build",
        *(f"{n} ({'<' if s else '≤'} {num(lim, 0)})" for n, _, _, lim, s in ADR_TARGETS),
    ]
    body = []
    for r in res["adr_0011"]:
        cells = [r["build"]]
        for t in r["targets"]:
            if t["value"] is None:
                cells.append("n/a")
            else:
                cells.append(f"{num(t['value'], 1)}: {'within' if t['within'] else 'over'}")
        body.append(cells)
    out += table(head, body)
    out += ["## Options for the owner (page targets; not a decision)", ""]
    opts = []
    for p in res["pareto"]["points"]:
        lat, acc = p.get("latency"), p.get("accuracy")
        if p["set"] != "pages" or not lat:
            continue
        within = [lat["mean_ms"] <= 1500, lat["p95_ms"] <= 2700, lat["peak_mem_mib"] < 1836]
        opts.append(
            [
                p["variant"],
                p["version"],
                num(lat["mean_ms"], 0),
                num(lat["p95_ms"], 0),
                num(lat["peak_mem_mib"]),
                f"{sum(within)} of 3",
                num(acc["page_cer"]["value"], 2, 100) if acc else "n/a",
            ]
        )
    out += table(
        [
            "Point",
            "Version",
            "Page mean ms",
            "Page p95 ms",
            "Peak RAM MiB",
            "Page targets within",
            "Page CER %",
        ],
        opts,
    )
    if res["missing"]:
        out += ["## Missing files", "", *(f"- {m}" for m in res["missing"]), ""]
    if res["problems"]:
        out += ["## Problems (what rests on these is void or left out)", ""]
        for k, ps in res["problems"].items():
            out += [f"- {k}: {p}" for p in ps]
        out.append("")
    if res["notes"]:
        out += ["## Notes", ""]
        for k, ns in res["notes"].items():
            out += [f"- {k}: {n}" for n in ns]
        out.append("")
    out += [f"Scorer commit `{res['git_commit']}`.", ""]
    return "\n".join(out)


def render(res: dict[str, Any]) -> tuple[dict[str, str], Printer]:
    """The report texts by file name under `OUT`, and the printer that counted their intervals."""
    pr = Printer()
    texts = {
        "cost_table.md": render_cost(res, pr),
        "verdicts.md": render_verdicts(res, pr),
        "int8.md": render_int8(res, pr),
        "pareto.md": render_pareto(res, pr),
    }
    texts["README.md"] = render_readme(res, pr.counts())  # prints no interval itself
    return texts, pr


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8", newline="\n")


def write_reports(lay: Layout, res: dict[str, Any]) -> list[Path]:
    texts, pr = render(res)
    res = {**res, "intervals": {**pr.counts(), "printed_keys": pr.printed}}
    written = []
    for name, text in texts.items():
        _write(lay.out / name, text)
        written.append(lay.out / name)
    _write(lay.out / "results.json", json.dumps(res, indent=1, ensure_ascii=False))
    plot_pareto(res, lay.out / "pareto.png")
    written += [lay.out / "results.json", lay.out / "pareto.png"]
    _write(lay.bench / "efficiency_comparison.md", render_benchmark(res))
    written.append(lay.bench / "efficiency_comparison.md")
    _write(
        lay.bench / "efficiency_comparison.json",
        json.dumps(
            {"generated_by": res["generated_by"], **res["benchmark"]}, indent=1, ensure_ascii=False
        ),
    )
    written.append(lay.bench / "efficiency_comparison.json")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["check", "score"])
    parser.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    parser.add_argument("--data-root", type=Path, default=None, help="default <root>/data")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help=f"the GW-EFF-006 bundle (default <root>/{BUNDLE_DIR.as_posix()})",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="score: write the reports while registered files are missing",
    )
    args = parser.parse_args(argv)
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:  # the hypothesis names (H7′) on a Windows console
        reconfigure(errors="backslashreplace")
    lay = Layout.at(args.root, args.data_root, args.bundle)
    res = Scorer(lay).run()
    for m in res["missing"]:
        print(f"missing: {m}")
    for k, ps in res["problems"].items():
        for p in ps:
            print(f"problem ({k}): {p}")
    verdicts = ", ".join(f"{k} {res['verdicts'][k]['verdict']}" for k in (*CONFIRMATORY, "H7′"))
    print(f"verdicts: {verdicts}")
    if args.command == "check":
        return 1 if res["missing"] or res["problems"] else 0
    if res["missing"] and not args.allow_missing:
        raise SystemExit(
            f"{len(res['missing'])} registered files are missing; nothing written "
            "(--allow-missing writes the reports without them)"
        )
    for path in write_reports(lay, res):
        print(f"wrote {path.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
