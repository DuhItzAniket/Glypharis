"""GW-E2E-009: the one Phase 14 scoring pass, the comparisons, the five verdicts and the hand-off.

    python scripts/phase14_score.py hash-phase5   # once, before any Phase 14 run
    python scripts/phase14_score.py check         # completeness, hashes, provenance; no score
    python scripts/phase14_score.py score         # the scoring pass (PHASE_STATUS, Phase 14)
    python scripts/phase14_score.py score --resume  # finish a pass that stopped after it began

`hash-phase5` records the SHA-256 and line count of every file under
data/processed/baselines/ in data/manifests/processed/phase5_baseline_files.json. It refuses to
change a recorded hash: the Phase 5 files are hashed before any Phase 14 run and never written.

Every path the scorer writes into phase14_outputs.json, phase5_baseline_files.json and the
hand-off list of results.json is repository-relative (``data/...``), as `scripts/ext_eval.py`
writes them; a recorded path without the ``data/`` prefix is still read.

`score` runs once, in this order, and stops at the first refusal:

0. Preflight, before any score is computed and before anything is written: the branch; every
   planned prediction file present, with its run metadata and exactly the keys of its set (read
   from the keys alone, as the runners do); the Phase 5 files against their recorded hashes;
   every Phase 14 file (and each GLYPHARIS run's metadata) against phase14_outputs.json; every
   GLYPHARIS run's provenance against the freeze record and the pinned tag; the latency sitting
   (both profiles present, not void, the baseline first and within `SITTING_GAP_HOURS`); every
   set a Phase 14 file reads against the preparation's record (`check_inputs`) and against each
   GLYPHARIS run's recorded inputs; the official evaluators' ground truth against its manifest
   hash, their environments and code folders present, and a dry conversion of every prediction
   file they will read. Only then are the baseline files and the fresh baseline profile added to
   phase14_outputs.json.

The pass is one evaluation even when it stops part-way: `score` writes
data/processed/phase14/GW-E2E-009/pass.json ("started") before its first logged row and marks it
"done" at the end. A started pass that did not finish refuses a plain `score`; `score --resume`
clears the official/ and handoff/ folders the pass owns and runs it again, every scoring row
logged with the reason "Phase 14 scoring (GW-E2E-009) (resumed)". A done pass is never re-run.
1. The Phase 5 reproduction check: every Phase 5 file is re-scored with
   `evaluate_baselines.evaluate_file` (the Phase 5 path) and each scored test file logs one row
   under its producing ID (`<system> (<task>), scoring`, phase 14, reason
   "Phase 14 scoring (GW-E2E-009)"). The entries and the efficiency block must equal
   `git show phase-05:reports/baselines/results.json` exactly, and every stored ``ref`` must equal
   the current pack label; any difference stops the pass.
2. Every Phase 14 file is scored with `evaluate_baselines.score_file` (one row per test file):
   both tracks, the charset ceiling, IIIT5K's alphanumeric track, GDC-scan's vertical lines and
   OmniDocBench's block-level n/a come from there. On top: the OmniDocBench subset for every
   system (computed from the same per-page results, so a file is logged once), GDC per capture
   condition, GDC-scan without its out-of-charset lines (information), and GLYPHARIS minus each
   baseline on the same units (descriptive intervals).
3. The official HierText line-level score for every system with HierText test pages, and the
   official OmniDocBench text-block and reading-order edit distances (all pages, and the 100-page
   subset for every system). These read files already scored and logged in this pass.
4. The confirmatory tests, exactly as registered: H1′ CER (strict, normalised, source-document
   cluster), H1′ latency (GW-EFF-005 against the fresh GW-BASE-002 profile), H5′ (lines, and the
   GDC side by source document), H8′ transfer and by half (lines, and pages). Every other interval
   is descriptive.
5. The Phase 15 hand-off under data/processed/phase14/handoff/: per-line recognition results with
   the per-line attributes, detector match and IoU, capture condition, the oracle reading, the
   pipeline's matched pairs, per-page values of every system, and GDC crops of every line read
   with an error plus 200 correct lines (seed 20261115). GDC-scan contributes numbers only. The
   files' SHA-256 and line counts go to phase14_outputs.json.
6. Aggregates only under reports/external/ (results.json and Markdown), with the scorer commit.
   No GDC-scan text and no per-sample test value reaches a report; the one per-sample list
   is H1′'s latency (per-page times on validation pages), which the plan asks for.

Bootstrap: `glypharis.eval.stats` defaults (1,000 resamples, seed 20261010, 95% percentile).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import evaluate_baselines as eb
import numpy as np
import official_hiertext
import official_omnidocbench
import shapely
import yaml
from run_baselines import (
    EXPERIMENTS,
    PHASE5_ROOT,
    PHASE14_PLAN,
    PROFILE_SAMPLES,
    PROFILE_THREADS,
    SYSTEM_NAMES,
    WARMUP,
    expected_keys,
    log_path,
    output_path,
    sha256_file,
)

from glypharis.data.linepack import LinePackReader
from glypharis.data.pages import Page, PageLine, read_pages
from glypharis.eval.calib_stats import ece_diff_ci
from glypharis.eval.inputs import RUN_ENV, check_inputs
from glypharis.eval.page_metrics import (
    TEXT_MATCH_IOU,
    PageResult,
    PredLine,
    _mostly_inside,
    _poly,
    evaluate_page,
    iou_matrix,
    is_box_line,
    is_text_line,
    match_two_tier,
)
from glypharis.eval.sets import (
    EVAL_SETS,
    PHASE5_HASHES,
    PHASE14_BRANCH,
    SUBSET_NAME,
    require_branch,
    subset_pages,
)
from glypharis.eval.slices import CLEAN_SLICE, gdc_conditions, gdc_sources
from glypharis.eval.stats import (
    Interval,
    cluster_paired_ratio_diff_ci,
    cluster_unpaired_ratio_diff_ci,
    mean_ci,
    paired_ratio_diff_ci,
    paired_ratios_diff_ci,
    unpaired_ratio_diff_ci,
    verdict,
)
from glypharis.eval.text_metrics import RecTrack, normalise, score_pair
from glypharis.pipeline.calibration import ece, reliability
from glypharis.pipeline.frozen import PIN_TAG, tag_exists
from glypharis.utils.gitinfo import git_info

EXPERIMENT = "GW-E2E-009"
REASON = "Phase 14 scoring (GW-E2E-009)"
PHASE = 14
PHASE5_TAG = "phase-05"
PHASE5_RESULTS = "reports/baselines/results.json"
FREEZE_RECORD = Path("configs/final/glypharis_v1.yaml")
# Under the data root, like every path the runners write (PHASE5_HASHES too, from sets).
OUTPUTS = Path("manifests/processed/phase14_outputs.json")
GLYPHARIS_ROOT = Path("processed/phase14")
HANDOFF = GLYPHARIS_ROOT / "handoff"
OFFICIAL = GLYPHARIS_ROOT / EXPERIMENT / "official"
# The pass's own record: "started" before its first logged row, "done" at the end.
PASS_RECORD = GLYPHARIS_ROOT / EXPERIMENT / "pass.json"
RESUMED = " (resumed)"  # appended to REASON on every row a resumed pass logs
# Every input a Phase 14 file reads must be recorded by the preparation (tests' tiny sets are not).
REQUIRE_RECORDED = True
# The official evaluators' code lies under this root (None: `official_hiertext.ROOT`).
EVALUATOR_ROOT: Path | None = None
DEFAULT_OUT = Path("reports/external")
DEFAULT_EFF = Path("reports/external/GW-EFF-005")
# The test-access log of every scoring row: the repository's own (tests point it elsewhere).
ACCESS_LOG: Path | None = None
LATENCY_BASELINE = "ppocr_mobile"  # GW-BASE-002, H1′'s control
LATENCY_PROFILE = "profile_glypharis_{threads}t.json"
SITTING_GAP_HOURS = 2  # GW-EFF-005 starts at most this long after the baseline profile ended
HANDOFF_SEED = 20261115
HANDOFF_CORRECT = 200
GLYPHARIS = "glypharis"
H1_FACTOR = 2.0
H5_FACTOR = 1.5
H8_HALF = 0.5
NUMBERS_ONLY = ("gdc_scan_test",)  # sets whose text and images never leave the data folder
TRACKS = eb.TRACKS
ALNUM = eb.ALNUM

# The GLYPHARIS prediction files (docs/PHASE_STATUS.md, Phase 14 item 2): ID -> (task, sets).
# They cover every crop and page of a set, validation included (no subsample).
GLYPHARIS_RUNS: dict[str, tuple[str, tuple[str, ...]]] = {
    "GW-REC-039": (
        "rec",
        ("hiertext_test", "gdc_test", "gdc_scan_test", "funsd_test", "iiit5k_test"),
    ),
    "GW-E2E-007": (
        "e2e",
        ("hiertext_test", "gdc_test", "gdc_scan_test", "funsd_test", "omnidocbench_test"),
    ),
    "GW-E2E-008": ("oracle", ("gdc_test", "gdc_scan_test")),
    "GW-REC-038": ("rec", ("hiertext_val",)),
    "GW-E2E-006": ("e2e", ("hiertext_val",)),
}
GLYPHARIS_VERSION = "GLYPHARIS v1 (configs/final/glypharis_v1.yaml)"
# The baselines' Phase 14 coverage, as registered with the runner.
BASELINE_PLAN: dict[tuple[str, str, str], str | None] = dict(PHASE14_PLAN)


# ------------------------------------------------------------------------------- the jobs


@dataclass(frozen=True)
class Job:
    """One prediction file to score."""

    path: Path
    experiment: str
    system: str
    task: str  # rec, e2e or oracle
    eval_set: str
    phase: int  # 5: a Phase 5 file re-scored; 14: a Phase 14 file
    subset: bool = False  # the file covers the OmniDocBench subset only

    @property
    def glypharis(self) -> bool:
        return self.system == GLYPHARIS

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.experiment, self.system, self.task, self.eval_set)

    @property
    def label(self) -> str:
        sub = f" ({SUBSET_NAME})" if self.subset else ""
        return f"{self.experiment} {self.system} {self.task} {self.eval_set}{sub}"

    @property
    def page_task(self) -> str:
        return "rec" if self.task == "rec" else "e2e"


def _rel(path: Path, data: Path) -> str:
    return path.relative_to(data).as_posix()


def data_rel(path: Path, data: Path) -> str:
    """A file under the data root as the repository names it (``data/...``), never absolute: the
    form every manifest and results.json holds (`ext_eval.data_rel`)."""
    return "data/" + _rel(path, data)


def shown(path: Path) -> str:
    """A path as a report may show it: relative to the working directory (the repository root),
    or only its name when it lies elsewhere, so no absolute path is ever written."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def glypharis_jobs(data: Path) -> list[Job]:
    return [
        Job(data / GLYPHARIS_ROOT / exp / f"{s}.jsonl", exp, GLYPHARIS, task, s, PHASE)
        for exp, (task, sets) in GLYPHARIS_RUNS.items()
        for s in sets
    ]


def baseline_jobs(data: Path) -> list[Job]:
    jobs = []
    for (system, task, eval_set), planned in sorted(BASELINE_PLAN.items()):
        if planned not in (None, SUBSET_NAME):
            continue  # registered as not run, with its reason
        sub = planned == SUBSET_NAME
        exp = EXPERIMENTS[system]
        path = output_path(data, exp, system, task, eval_set, phase=PHASE, subset=sub)
        jobs.append(Job(path, exp, system, task, eval_set, PHASE, sub))
    return jobs


def phase5_jobs(data: Path) -> list[Job]:
    """The Phase 5 prediction files, selected exactly as `evaluate_baselines --phase 5` does."""
    jobs = []
    for f in sorted((data / "processed" / PHASE5_ROOT).glob("*/*.jsonl")):
        parts = eb.name_parts(f)
        if parts is not None:
            system, task, eval_set, sub = parts
            jobs.append(Job(f, f.parent.name, system, task, eval_set, 5, sub))
    return jobs


def full_keys(data: Path, eval_set: str, task: str) -> list[str]:
    """Every crop (rec) or page (e2e, oracle) of a set, in pack or page-file order."""
    spec = EVAL_SETS[eval_set]
    if task == "rec":
        return [r.key for r in LinePackReader(data / spec["pack"]).records]
    return [p.doc_id for p in read_pages(data / spec["pages"])]


def job_keys(data: Path, job: Job) -> list[str]:
    if job.glypharis:
        return full_keys(data, job.eval_set, job.page_task)
    return expected_keys(data, job.eval_set, job.task, job.subset)


class FullContext(eb.Context):
    """The scorer's context for GLYPHARIS files, which hold every crop and page of their set
    (GW-REC-038 and GW-E2E-006 run the whole validation set, not the baselines' subsample)."""

    def keys(self, eval_set: str, task: str, subset: bool = False) -> list[str]:
        if subset:
            raise ValueError("GLYPHARIS files cover whole sets")
        k = (eval_set, "rec" if task == "rec" else "e2e", False)
        if k not in self._keys:
            self._keys[k] = full_keys(self.data, eval_set, k[1])
        return self._keys[k]


def glypharis_meta(job: Job) -> dict[str, Any]:
    """What `score_file` needs to describe a GLYPHARIS file; the runner's own metadata gives the
    settings."""
    run = json.loads(job.path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return {
        "system": GLYPHARIS,
        "task": job.task,
        "eval_set": job.eval_set,
        "experiment": job.experiment,
        "version": GLYPHARIS_VERSION,
        "settings": run.get("settings"),
        "expected": run.get("expected"),
        "runs": run.get("runs", []),
    }


# ------------------------------------------------------------------------------- preflight


def check_complete(data: Path, jobs: Sequence[Job]) -> dict[str, int]:
    """Refuse unless every planned file exists with its metadata and exactly its set's keys.

    Reads the keys alone, as the runners do before a run, so nothing is scored or printed while
    any prediction pass is incomplete."""
    problems, counts = [], {}
    for job in jobs:
        rel = data_rel(job.path, data)
        if not job.path.is_file():
            problems.append(f"{rel}: missing")
            continue
        if not job.path.with_suffix(".meta.json").is_file():
            problems.append(f"{rel}: no run metadata")
        keys = [str(r["key"]) for r in eb.read_records(job.path)]
        want = job_keys(data, job)
        seen = Counter(keys)
        twice = sum(1 for c in seen.values() if c > 1)
        missing, stray = len(set(want) - set(seen)), len(set(seen) - set(want))
        if twice or missing or stray:
            problems.append(
                f"{rel}: {len(seen)} of {len(want)} keys ({missing} missing, "
                f"{stray} outside the set, {twice} repeated)"
            )
        counts[rel] = len(keys)
    if problems:
        raise SystemExit(
            "not every prediction file is complete, so nothing is scored:\n  "
            + "\n  ".join(problems)
        )
    return counts


def _count_lines(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    return sum(1 for ln in path.read_bytes().splitlines() if ln.strip())


def file_entry(path: Path, data: Path) -> dict[str, Any]:
    return {"file": data_rel(path, data), "sha256": sha256_file(path), "lines": _count_lines(path)}


def _phase5_files(data: Path) -> list[Path]:
    return sorted(p for p in (data / "processed" / PHASE5_ROOT).rglob("*") if p.is_file())


def hash_phase5(data: Path) -> int:
    """Record every Phase 5 file's hash; an existing record may only be confirmed, not changed."""
    entries = [file_entry(p, data) for p in _phase5_files(data)]
    if not entries:
        raise SystemExit(f"no files under processed/{PHASE5_ROOT}")
    path = data / PHASE5_HASHES
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))["files"]
        if _by_file(old) != _by_file(entries):
            raise SystemExit(
                f"{PHASE5_HASHES.as_posix()} exists and the Phase 5 files differ from "
                "it; the record is never rewritten"
            )
        print(f"{len(entries)} Phase 5 files match data/{PHASE5_HASHES.as_posix()}")
        return 0
    if tag_exists():
        raise SystemExit(
            f"the {PIN_TAG} tag exists: the Phase 5 files are hashed and the record committed "
            "before the tag (Order step 2), never after it"
        )
    doc = {
        "purpose": "Phase 14: the Phase 5 baseline files, hashed before any Phase 14 run; they "
        "are read, never written (docs/PHASE_STATUS.md, Phase 14 item 4)",
        "hashed": datetime.now(UTC).date().isoformat(),
        "files": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(
        f"hashed {len(entries)} Phase 5 files into data/{PHASE5_HASHES.as_posix()}; commit it "
        "before the tag"
    )
    return 0


def verify_phase5(data: Path) -> dict[str, Any]:
    path = data / PHASE5_HASHES
    if not path.is_file():
        raise SystemExit(
            f"{PHASE5_HASHES.as_posix()} is missing: run `hash-phase5` before any Phase 14 run"
        )
    recorded = {
        _norm_file(e["file"]): e["sha256"] for e in json.loads(path.read_text("utf-8"))["files"]
    }
    current = {_rel(p, data): sha256_file(p) for p in _phase5_files(data)}
    changed = sorted(f for f in set(recorded) | set(current) if recorded.get(f) != current.get(f))
    if changed:
        raise SystemExit(
            f"{len(changed)} Phase 5 files differ from their recorded hashes, e.g. "
            f"data/{changed[0]}; scoring stops"
        )
    return {
        "record": f"data/{PHASE5_HASHES.as_posix()}",
        "record_sha256": sha256_file(path),
        "files": len(current),
    }


# phase14_outputs.json as `scripts/ext_eval.py` writes it: entries under "outputs", sorted by file.
_OUTPUTS_NOTE = (
    "Phase 14 per-sample outputs (git-ignored under data/processed/phase14/): SHA-256 and line "
    "counts only, checked before Phases 15 and 22 read them"
)


def _read_outputs(data: Path) -> dict[str, Any]:
    path = data / OUTPUTS
    if not path.is_file():
        return {"note": _OUTPUTS_NOTE, "outputs": []}
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or not isinstance(doc.get("outputs"), list):
        raise SystemExit(f'{OUTPUTS.as_posix()} does not hold an "outputs" list')
    return doc


def _norm_file(name: str) -> str:
    """A recorded path, data-root-relative: ``data/...`` (the written form) and the bare form
    both name the same file."""
    return name.removeprefix("data/")


def _by_file(entries: Iterable[dict[str, Any]]) -> dict[str, tuple[Any, Any]]:
    return {_norm_file(e["file"]): (e["sha256"], e.get("lines")) for e in entries}


def check_outputs(data: Path, entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Refuse when a file differs from its recorded hash or count, or its run metadata from a
    recorded ``meta_sha256`` (the GLYPHARIS runner records one: the pin and provenance the void
    checks read live there); return the entries not recorded yet. Writes nothing."""
    known = {_norm_file(e["file"]): e for e in _read_outputs(data)["outputs"]}
    new = []
    for e in entries:
        name = _norm_file(e["file"])
        old = known.get(name)
        if old is None:
            new.append(e)
            continue
        if (old["sha256"], old.get("lines")) != (e["sha256"], e["lines"]):
            raise SystemExit(
                f"{e['file']} differs from its hash in {OUTPUTS.as_posix()}; scoring stops"
            )
        if old.get("meta_sha256") is not None:
            meta = (
                data / _norm_file(str(old["meta"]))
                if old.get("meta")
                else (data / name).with_suffix(".meta.json")
            )
            got = sha256_file(meta) if meta.is_file() else None
            if got != old["meta_sha256"]:
                raise SystemExit(
                    f"the run metadata of {e['file']} differs from its hash in "
                    f"{OUTPUTS.as_posix()}; scoring stops"
                )
    return new


def record_outputs(data: Path, entries: Sequence[dict[str, Any]]) -> int:
    """Add the entries not recorded yet to phase14_outputs.json (hashes and counts only); an
    entry already there must match."""
    new = check_outputs(data, entries)
    if not new:
        return 0
    doc = _read_outputs(data)
    doc["outputs"] = sorted([*doc["outputs"], *new], key=lambda e: str(e.get("file")))
    path = data / OUTPUTS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return len(new)


def _find(meta: dict[str, Any], *fields: str) -> dict[str, Any] | None:
    """The block of a run's metadata holding ``fields`` (at the top or one level down), so the
    check does not depend on what the runner calls the block."""
    for v in (meta, *meta.values()):
        if isinstance(v, dict) and all(f in v for f in fields):
            return v
    return None


def _find_pin(meta: dict[str, Any]) -> dict[str, Any] | None:
    """The `pinned_code_check` result a GLYPHARIS run recorded."""
    return _find(meta, "trees", "tag_commit")


def check_glypharis_runs(jobs: Sequence[Job], record: Path, charset: Path) -> dict[str, Any]:
    """Every GLYPHARIS run at one pinned tag commit, from the freeze record's checkpoints and the
    charset file the scorer reads.

    A mismatch found here, before scoring, makes the run void (Phase 14 item 6)."""
    rec = yaml.safe_load(record.read_text(encoding="utf-8"))
    want = {
        "record_sha256": sha256_file(record),
        "recognizer_checkpoint_sha256": rec["recognizer"]["checkpoint_sha256"],
        "detector_checkpoint_sha256": rec["detector"]["checkpoint_sha256"],
        "charset_sha256": sha256_file(charset),
    }
    problems, tags, envs = [], set(), set()
    for job in jobs:
        meta = json.loads(job.path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        pin = _find_pin(meta)
        if pin is None or pin.get("trees") != "equal":
            problems.append(f"{job.label}: not run at the pinned code")
        else:
            tags.add(str(pin["tag_commit"]))
        prov = _find(meta, "recognizer_checkpoint_sha256") or {}
        for k, v in want.items():
            if prov.get(k) != v:
                problems.append(f"{job.label}: {k} differs from the freeze record or charset")
        # H5′ compares validation with test "under the same code and precision": one device,
        # one software stack and one set of numerics flags across every run.
        envs.add(json.dumps({k: prov.get(k) for k in RUN_ENV}, sort_keys=True))
    if len(tags) > 1:
        problems.append(f"GLYPHARIS runs come from {len(tags)} different tag commits")
    if len(envs) > 1:
        problems.append(f"GLYPHARIS runs differ in {', '.join(RUN_ENV)} ({len(envs)} variants)")
    if problems:
        raise SystemExit(
            "void GLYPHARIS runs (provenance differs from the plan); nothing is "
            "scored:\n  " + "\n  ".join(problems)
        )
    return {
        "runs": len(jobs),
        "tag_commit": next(iter(tags), None),
        **want,
    }


# What a GLYPHARIS run's "inputs" block (`check_inputs` at run time) must still say.
_INPUT_FIELDS = {"pack": ("path", "bin_sha256", "index_sha256"), "pages": ("path", "sha256")}


def check_job_inputs(data: Path, jobs: Sequence[Job]) -> dict[str, Any]:
    """Every set a Phase 14 file reads equals what the preparation recorded, and what each
    GLYPHARIS run recorded reading.

    `check_inputs` (bytes only, nothing decoded) runs once per set and mode, with
    `REQUIRE_RECORDED`; its refusal stops the pass. A GLYPHARIS run whose ``inputs`` block names
    another hash than the files on disk now read other inputs than the ones scored, so it is void
    (Phase 14 item 6). The Phase 5 files are covered by their own hashes and the stored-reference
    check of step 1."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for job in jobs:
        k = (job.eval_set, job.task)
        if k not in found:
            n = None if job.task == "rec" else len(full_keys(data, job.eval_set, "e2e"))
            found[k] = check_inputs(data, job.eval_set, job.task, n, require=REQUIRE_RECORDED)
    problems = []
    for job in jobs:
        if not job.glypharis:
            continue
        meta = json.loads(job.path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        got = meta.get("inputs")
        got = got if isinstance(got, dict) else {}
        for block, fields in _INPUT_FIELDS.items():
            want = found[(job.eval_set, job.task)].get(block)
            if want is None:
                continue
            have = got.get(block)
            have = have if isinstance(have, dict) else {}
            for f in fields:
                if have.get(f) != want[f]:
                    problems.append(
                        f"{job.label}: recorded {block} {f} differs from {want['path']}"
                    )
    if problems:
        raise SystemExit(
            "void GLYPHARIS runs (their recorded inputs differ from the files scored); nothing "
            "is scored:\n  " + "\n  ".join(problems)
        )
    return {
        "checked": sorted(f"{s} {m}" for s, m in found),
        "required_recorded": REQUIRE_RECORDED,
    }


# name -> (environment, code folder, a file the run reads inside it)
EVALUATORS: dict[str, tuple[str, Path, Path | None]] = {
    "hiertext": (official_hiertext.HIERTEXT_VENV, official_hiertext.HIERTEXT_DIR, None),
    "omnidocbench": (
        official_omnidocbench.OMNIDOCBENCH_VENV,
        official_omnidocbench.OMNIDOCBENCH_DIR,
        official_omnidocbench.PIPELINE_CONFIG,
    ),
}
# The raw files the official evaluators score against: (dataset manifest, file key).
OFFICIAL_GT = {"hiertext": ("hiertext", "gt_test"), "omnidocbench": ("omnidocbench", "annotations")}


def check_evaluators(data: Path) -> dict[str, Any]:
    """Everything step 3 needs that no score does: the ground truth against its manifest hash
    (bytes only), each evaluator's environment and code, and its pinned entry. Refuses before any
    row is logged, so a missing environment cannot stop the pass after scoring began."""
    root = official_hiertext.ROOT if EVALUATOR_ROOT is None else EVALUATOR_ROOT
    problems, info = [], {}
    for name, (dataset, key) in OFFICIAL_GT.items():
        try:
            verified_raw(data, dataset, key)
        except SystemExit as exc:
            problems.append(f"{name}: {exc}")
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"{name}: ground truth not readable ({type(exc).__name__})")
    for name, (venv, code, inner) in EVALUATORS.items():
        try:
            official_hiertext.venv_python(venv)
        except (FileNotFoundError, OSError, subprocess.CalledProcessError) as exc:
            problems.append(f"{name}: {exc}")
        if not (root / code).is_dir():
            problems.append(f"{name}: evaluator code {code.as_posix()} is missing")
        elif inner is not None and not (root / code / inner).is_file():
            problems.append(f"{name}: {(code / inner).as_posix()} is missing")
        try:
            info[name] = official_hiertext.evaluator_info(name, root)
        except (OSError, KeyError, ValueError) as exc:
            problems.append(f"{name}: no pinned entry ({type(exc).__name__}: {exc})")
    if problems:
        raise SystemExit(
            "the official evaluators are not ready; nothing is scored:\n  " + "\n  ".join(problems)
        )
    return info


def dry_convert(data: Path, jobs: Sequence[Job]) -> dict[str, int]:
    """Convert every prediction file the official evaluators will read, as step 3 does, into a
    temporary folder under the data root, and delete it.

    Reads prediction files only (model outputs), never the ground truth, so it is preflight, not
    a scoring access. A polygon of fewer than 3 points, a repeated page or a page outside the set
    is refused here instead of after the scoring rows are logged."""
    hier = [j for j in jobs if j.task == "e2e" and j.eval_set == "hiertext_test"]
    odb = [j for j in jobs if j.task == "e2e" and j.eval_set == "omnidocbench_test"]
    if not hier and not odb:
        return {"hiertext": 0, "omnidocbench": 0}
    problems = []
    parent = data / GLYPHARIS_ROOT / EXPERIMENT
    made = not parent.exists()
    parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="preflight-", dir=parent) as tmp:
            work = Path(tmp)
            ids = full_keys(data, "hiertext_test", "e2e") if hier else []
            for i, job in enumerate(hier):
                try:
                    pred = official_hiertext.to_hiertext_predictions(
                        eb.read_records(job.path), image_ids=ids
                    )
                    official_hiertext.write_predictions(pred, work / "hiertext" / f"{i}.json")
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    problems.append(f"{job.label}: {type(exc).__name__}: {exc}")
            odb_pages = EVAL_SETS["omnidocbench_test"]["pages"]
            pages = {p.doc_id: p for p in read_pages(data / odb_pages)} if odb else {}
            scopes = {"all": set(pages), SUBSET_NAME: set(subset_pages(data)) if odb else set()}
            for i, job in enumerate(odb):
                try:
                    records = eb.read_records(job.path)
                    for scope in (SUBSET_NAME,) if job.subset else ("all", SUBSET_NAME):
                        official_omnidocbench.write_predictions(
                            [
                                {**r, "image": pages[str(r["key"])].image}
                                for r in records
                                if str(r["key"]) in scopes[scope]
                            ],
                            work / "omnidocbench" / f"{i}_{scope}",
                        )
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    problems.append(f"{job.label}: {type(exc).__name__}: {exc}")
    finally:
        if made and parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    if problems:
        raise SystemExit(
            "a prediction file cannot be converted for its official evaluator; nothing is "
            "scored:\n  " + "\n  ".join(problems)
        )
    return {"hiertext": len(hier), "omnidocbench": len(odb)}


def latency_paths(data: Path, eff_dir: Path) -> dict[str, Path]:
    base = output_path(
        data, EXPERIMENTS[LATENCY_BASELINE], LATENCY_BASELINE, "e2e", "hiertext_val", phase=PHASE
    )
    p5 = output_path(data, EXPERIMENTS[LATENCY_BASELINE], LATENCY_BASELINE, "e2e", "hiertext_val")
    return {
        "baseline": base.with_name(f"profile_{base.stem}.json"),
        "glypharis": eff_dir / LATENCY_PROFILE.format(threads=PROFILE_THREADS),
        "glypharis_16t": eff_dir / LATENCY_PROFILE.format(threads=16),
        "phase5_baseline": p5.with_name(f"profile_{p5.stem}.json"),
    }


def pinned_state() -> dict[str, Any]:
    """The scorer's own commit against the tag: recorded in the report, not required."""
    from glypharis.pipeline.frozen import pinned_code_check

    try:
        return pinned_code_check()
    except RuntimeError as exc:
        return {"trees": "differ", "error": str(exc)}


def load_phase5_results(tag: str = PHASE5_TAG) -> tuple[dict[str, Any], str]:
    """The Phase 5 report's results.json as committed at the tag, and the tag's commit."""
    show = subprocess.run(
        ["git", "show", f"{tag}:{PHASE5_RESULTS}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    rev = subprocess.run(
        ["git", "rev-list", "-n", "1", tag], capture_output=True, text=True, check=False
    )
    if show.returncode != 0 or rev.returncode != 0:
        raise SystemExit(f"cannot read {PHASE5_RESULTS} at tag {tag}")
    return json.loads(show.stdout), rev.stdout.strip()


# ------------------------------------------------------------------------- per-unit values


@dataclass
class RecUnits:
    """Per-crop values of one file in the set's key order; an empty reference has den 0."""

    keys: list[str]
    tracks: dict[str, dict[str, np.ndarray]]  # track -> err, den, werr, wden, exact

    def select(self, keys: Sequence[str]) -> RecUnits:
        pos = {k: i for i, k in enumerate(self.keys)}
        idx = np.array([pos[k] for k in keys], dtype=np.int64)
        return RecUnits(
            list(keys), {t: {n: a[idx] for n, a in d.items()} for t, d in self.tracks.items()}
        )


@dataclass
class PageUnits:
    """Per-page results of one file in the set's key order, per track."""

    keys: list[str]
    results: dict[str, list[PageResult]]

    def select(self, keys: Sequence[str]) -> PageUnits:
        pos = {k: i for i, k in enumerate(self.keys)}
        return PageUnits(
            list(keys), {t: [r[pos[k]] for k in keys] for t, r in self.results.items()}
        )

    def arr(self, track: str, f: Callable[[PageResult], float]) -> np.ndarray:
        return np.array([f(r) for r in self.results[track]], dtype=np.float64)


def rec_tracks(eval_set: str) -> tuple[RecTrack, ...]:
    if EVAL_SETS[eval_set]["benchmark"] in eb.ALNUM_BENCHMARKS:
        return (ALNUM, *TRACKS)
    return TRACKS


def rec_unit(eval_set: str) -> str:
    return "words" if EVAL_SETS[eval_set]["benchmark"] in eb.ALNUM_BENCHMARKS else "lines"


def rec_units(
    by_key: dict[str, dict[str, Any]], keys: Sequence[str], tracks: Iterable[RecTrack]
) -> RecUnits:
    out: dict[str, dict[str, np.ndarray]] = {}
    for t in tracks:
        a = {n: np.zeros(len(keys)) for n in ("err", "den", "werr", "wden", "exact")}
        for i, k in enumerate(keys):
            r = by_key[k]
            if not normalise(r["ref"], t):
                continue  # left out of every ratio, as RecAccumulator does
            s = score_pair(r["pred"], r["ref"], t)
            a["err"][i], a["den"][i] = s.char_errors, s.ref_chars
            a["werr"][i], a["wden"][i], a["exact"][i] = s.word_errors, s.ref_words, s.exact
        out[t] = a
    return RecUnits(list(keys), out)


def preds_of(record: dict[str, Any]) -> list[PredLine]:
    return [PredLine(ln["polygon"], ln["text"], ln.get("confidence")) for ln in record["lines"]]


def page_units(
    pages: dict[str, Page], by_key: dict[str, dict[str, Any]], keys: Sequence[str]
) -> PageUnits:
    return PageUnits(
        list(keys),
        {t: [evaluate_page(pages[k], preds_of(by_key[k]), track=t) for k in keys] for t in TRACKS},
    )


def _agree(what: str, entry_ci: dict[str, Any] | None, num: np.ndarray, den: np.ndarray) -> None:
    """The per-unit values must give the entry's own number, or the two code paths diverged."""
    got = float(num.sum() / den.sum()) if den.sum() else None
    want = None if entry_ci is None else float(entry_ci["value"])
    differ = got is not None and want is not None and abs(got - want) > 1e-12
    if (got is None) != (want is None) or differ:
        raise RuntimeError(f"{what}: per-unit values give {got}, the scorer {want}")


def check_units(job: Job, entry: dict[str, Any], units: RecUnits | PageUnits) -> None:
    if isinstance(units, RecUnits):
        for t, a in units.tracks.items():
            _agree(f"{job.label} {t} CER", entry["tracks"][t]["cer"], a["err"], a["den"])
        return
    for t in TRACKS:
        e = entry["tracks"][t]
        if e.get("page_cer_bag_of_lines") is not None or not EVAL_SETS[job.eval_set]["block_level"]:
            _agree(
                f"{job.label} {t} bag-of-lines CER",
                e["page_cer_bag_of_lines"],
                units.arr(t, lambda r: r.bag_errors),
                units.arr(t, lambda r: r.bag_ref_chars),
            )
        if e.get("page_cer_order_aware") is not None:
            _agree(
                f"{job.label} {t} order-aware CER",
                e["page_cer_order_aware"],
                units.arr(t, lambda r: r.aware_errors),
                units.arr(t, lambda r: r.aware_ref_chars),
            )


# ------------------------------------------------------------------------ matching per line


@dataclass
class PageMatch:
    """How a page's predicted lines meet its box lines, as bag-of-lines matches them."""

    box_lines: list[PageLine]
    matched: dict[int, tuple[int, float]]  # box line -> (prediction, IoU)
    best_iou: list[float]  # per box line, over every prediction
    status: list[str]  # per prediction: matched, spurious or ignored


def match_page(gt: Page, preds: Sequence[PredLine]) -> PageMatch:
    """`evaluate_page`'s bag-of-lines matching (IoU 0.5, lines that count first), per line."""
    box = [ln for ln in gt.lines if is_box_line(ln)]
    care = {ln.line_id for ln in box}
    polys = [_poly(ln.polygon) for ln in box + [ln for ln in gt.lines if ln.line_id not in care]]
    n = len(box)
    pred_polys = [_poly(p.polygon) for p in preds]
    dont_care = shapely.union_all([_poly(p) for p in gt.ignore] + polys[n:])
    ignored = _mostly_inside(pred_polys, dont_care)
    iou = iou_matrix(pred_polys, polys)
    match = match_two_tier(iou, n, TEXT_MATCH_IOU)
    matched: dict[int, tuple[int, float]] = {}
    status = []
    for p in range(len(preds)):
        g = match.get(p)
        if g is not None and g < n:
            matched[g] = (p, float(iou[p, g]))
            status.append("matched")
        else:
            status.append("ignored" if g is not None or ignored[p] else "spurious")
    best = iou[:, :n].max(axis=0).tolist() if len(preds) and n else [0.0] * n
    return PageMatch(box, matched, [float(b) for b in best], status)


def box_rotation(polygon: Sequence[Sequence[float]]) -> float | None:
    """Angle in degrees, in (-90, 90], of the long side of the polygon's minimum-area rectangle."""
    p = _poly(polygon)
    if p.is_empty or p.area <= 0:
        return None
    rect = shapely.minimum_rotated_rectangle(p)
    pts = list(rect.exterior.coords)[:4]
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    (x0, y0), (x1, y1) = max(edges, key=lambda e: math.dist(e[0], e[1]))
    a = math.degrees(math.atan2(y1 - y0, x1 - x0))
    while a <= -90:
        a += 180
    while a > 90:
        a -= 180
    return round(a, 2)


def in_charset(text: str, allowed: frozenset[str]) -> bool:
    """Whether every character of ``text`` (strict track) is in the GLYPHARIS charset."""
    return all(ch in allowed for ch in normalise(text, "strict"))


def in_charset_view(gt: Page, allowed: frozenset[str]) -> tuple[Page, int, int]:
    """The page with its box lines that hold a character outside the charset made don't care,
    as `vertical_view` makes GDC's other lines don't care: a prediction on such a line is
    ignored and the line is neither found nor missed. Returns the view, the box lines that still
    count and the box lines left out."""
    lines, kept, left = [], 0, 0
    for ln in gt.lines:
        if is_box_line(ln) and not in_charset(ln.text, allowed):
            lines.append(replace(ln, legible=False))
            left += 1
        else:
            kept += is_box_line(ln)
            lines.append(ln)
    return replace(gt, lines=lines), kept, left


# ----------------------------------------------------------------------------- intervals


def iv(x: Interval) -> dict[str, Any]:
    return {
        "value": x.value,
        "low": x.low,
        "high": x.high,
        "unit": x.unit,
        "resamples": x.resamples,
    }


def combined(verdicts: Sequence[str]) -> str:
    """A registered verdict that must agree across its checks; otherwise "not shown"."""
    return verdicts[0] if len(set(verdicts)) == 1 else "not shown"


def compare_rec(g: RecUnits, b: RecUnits, unit: str) -> dict[str, Any]:
    """GLYPHARIS minus a baseline on the same crops (descriptive)."""
    out: dict[str, Any] = {}
    for t, ga in g.tracks.items():
        ba = b.tracks[t]
        if not np.array_equal(ga["den"], ba["den"]):
            raise RuntimeError("two files of one set disagree on its references")
        m, mw = ga["den"] > 0, ga["wden"] > 0
        d = {
            "cer": iv(paired_ratio_diff_ci(ga["err"][m], ba["err"][m], ga["den"][m], unit)),
            "wer": iv(paired_ratio_diff_ci(ga["werr"][mw], ba["werr"][mw], ga["wden"][mw], unit)),
        }
        if t == ALNUM:
            d["word_accuracy"] = iv(
                paired_ratio_diff_ci(ga["exact"][m], ba["exact"][m], np.ones(int(m.sum())), unit)
            )
        out[t] = d
    return out


def _bag(u: PageUnits, t: str) -> tuple[np.ndarray, np.ndarray]:
    return u.arr(t, lambda r: r.bag_errors), u.arr(t, lambda r: r.bag_ref_chars)


def _aware(u: PageUnits, t: str) -> tuple[np.ndarray, np.ndarray]:
    return u.arr(t, lambda r: r.aware_errors), u.arr(t, lambda r: r.aware_ref_chars)


def _same_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not np.array_equal(a, b):
        raise RuntimeError("two files of one set disagree on its reference characters")
    return a


def compare_e2e(g: PageUnits, b: PageUnits, eval_set: str, factor: float = 1.0) -> dict[str, Any]:
    """GLYPHARIS minus ``factor`` x a baseline per page (descriptive; H1′ uses factor 2)."""
    spec = EVAL_SETS[eval_set]
    out: dict[str, Any] = {}
    for t in TRACKS:
        d: dict[str, Any] = {}
        if not spec["block_level"]:
            (eg, rg), (ep, rp) = _bag(g, t), _bag(b, t)
            d["page_cer_bag_of_lines"] = iv(
                paired_ratio_diff_ci(eg, factor * ep, _same_ref(rg, rp), "pages")
            )
        if spec["reading_order"] or spec["block_level"]:
            (eg, rg), (ep, rp) = _aware(g, t), _aware(b, t)
            d["page_cer_order_aware"] = iv(
                paired_ratio_diff_ci(eg, factor * ep, _same_ref(rg, rp), "pages")
            )
        out[t] = d
    if not spec["block_level"] and factor == 1.0:
        f1 = [
            (
                u.arr("strict", lambda r: 2 * r.det[0.5][0]),
                u.arr("strict", lambda r: 2 * r.det[0.5][0] + r.det[0.5][1] + r.det[0.5][2]),
            )
            for u in (g, b)
        ]
        out["detection_f1_0.5"] = iv(paired_ratios_diff_ci(*f1[0], *f1[1], "pages"))
    return out


# ----------------------------------------------------------------------------- calibration


def correct_flags(
    by_key: dict[str, dict[str, Any]], keys: Sequence[str], track: RecTrack
) -> np.ndarray:
    """Exact match on ``track`` for every line (GW-E2E-005: strict; IIIT5K: alnum)."""
    return np.array(
        [
            float(normalise(by_key[k]["pred"], track) == normalise(by_key[k]["ref"], track))
            for k in keys
        ]
    )


def confidences(
    by_key: dict[str, dict[str, Any]], keys: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    try:
        t = np.array([float(by_key[k]["conf_t"]) for k in keys])
        t1 = np.array([float(by_key[k]["conf_t1"]) for k in keys])
    except KeyError as exc:
        raise SystemExit(
            f"a GLYPHARIS record has no {exc}: confidences at T = 1 and the frozen "
            "T are needed for calibration"
        ) from exc
    return t, t1


def calibration_block(
    conf_t: np.ndarray,
    conf_t1: np.ndarray,
    y: np.ndarray,
    track: str,
    clusters: Sequence[object] | None = None,
    cluster_unit: str = "pages",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "lines": int(y.size),
        "track": track,
        "accuracy": float(y.mean()) if y.size else 0.0,
        "ece_t1": ece(conf_t1, y),
        "ece_t": ece(conf_t, y),
        "reliability_t1": reliability(conf_t1, y),
        "reliability_t": reliability(conf_t, y),
        "diff_lines": iv(ece_diff_ci(conf_t, conf_t1, y, 1.0, "lines")),
    }
    if clusters is not None:
        out["diff_clusters"] = iv(ece_diff_ci(conf_t, conf_t1, y, 1.0, cluster_unit, clusters))
        out["half_lines"] = iv(ece_diff_ci(conf_t, conf_t1, y, H8_HALF, "lines"))
        out["half_clusters"] = iv(ece_diff_ci(conf_t, conf_t1, y, H8_HALF, cluster_unit, clusters))
    return out


def pipeline_calibration(
    pages: dict[str, Page], by_key: dict[str, dict[str, Any]], keys: Sequence[str]
) -> dict[str, Any]:
    """Pipeline lines: a line is right when matched to a box line with exactly its text (strict);
    a spurious line is wrong; lines on don't-care text are left out."""
    conf_t, conf_t1, y, page_ids = [], [], [], []
    for k in keys:
        rec = by_key[k]
        m = match_page(pages[k], preds_of(rec))
        gt_of = {p: g for g, (p, _) in m.matched.items()}
        for p, ln in enumerate(rec["lines"]):
            if m.status[p] == "ignored":
                continue
            ok = False
            if p in gt_of:
                ok = normalise(ln["text"], "strict") == normalise(
                    m.box_lines[gt_of[p]].text, "strict"
                )
            conf_t.append(float(ln["confidence"]))
            conf_t1.append(float(ln["conf_t1"]))
            y.append(float(ok))
            page_ids.append(k)
    return calibration_block(
        np.array(conf_t), np.array(conf_t1), np.array(y), "strict", page_ids, "pages"
    )


# ------------------------------------------------------------------------------- the pass


@dataclass
class Scored:
    job: Job
    entry: dict[str, Any]
    units: RecUnits | PageUnits | None = None
    records: dict[str, dict[str, Any]] | None = None  # GLYPHARIS only, for the hand-off


class Scorer:
    """State of one scoring pass."""

    def __init__(self, data: Path, access: Path, reason: str = REASON) -> None:
        self.data = data
        self.ctx = eb.Context(data, access, PHASE, reason)
        self.gctx = FullContext(data, access, PHASE, reason)
        self.gctx._pages = self.ctx._pages  # one page cache for both
        self.scored: dict[tuple[str, str, str, str, bool], Scored] = {}
        self._packs: dict[str, dict[str, Any]] = {}

    def pack(self, eval_set: str) -> dict[str, Any]:
        """key -> crop record of a set's pack (doc id and label)."""
        if eval_set not in self._packs:
            reader = LinePackReader(self.data / EVAL_SETS[eval_set]["pack"])
            self._packs[eval_set] = {r.key: r for r in reader.records}
        return self._packs[eval_set]

    def get(
        self, experiment: str, system: str, task: str, eval_set: str, subset: bool = False
    ) -> Scored | None:
        return self.scored.get((experiment, system, task, eval_set, subset))

    def units_for(self, job: Job, ctx: eb.Context) -> tuple[RecUnits | PageUnits, dict[str, Any]]:
        by_key = {str(r["key"]): r for r in eb.read_records(job.path)}
        keys = ctx.keys(job.eval_set, job.task, job.subset)
        if job.task == "rec":
            return rec_units(by_key, keys, rec_tracks(job.eval_set)), by_key
        return page_units(ctx.pages(job.eval_set), by_key, keys), by_key

    # Step 1 --------------------------------------------------------------------------------

    def reproduce_phase5(self, jobs: Sequence[Job], phase5: dict[str, Any]) -> dict[str, Any]:
        """Re-score every Phase 5 file through the Phase 5 path and require Phase 5's numbers."""
        entries = []
        for job in jobs:
            entry = eb.evaluate_file(job.path, self.ctx)
            if entry is not None:
                entries.append(entry)
        refs_checked, ref_diffs = 0, []
        for job in jobs:
            if job.task != "rec":
                continue
            labels = self.pack(job.eval_set)
            for r in eb.read_records(job.path):
                refs_checked += 1
                crop = labels.get(str(r["key"]))
                if crop is None or crop.label != r["ref"]:
                    ref_diffs.append(job.label)
                    break
        profiles = [
            eb.read_profile(p)
            for p in sorted((self.data / "processed" / PHASE5_ROOT).glob("*/profile_*.json"))
        ]

        def keyed(items: Iterable[dict[str, Any]]) -> dict[tuple[Any, ...], Any]:
            return {
                (e["experiment"], e["system"], e["task"], e["eval_set"]): canon(e) for e in items
            }

        diffs = []
        now, then = keyed(entries), keyed(phase5["predictions"])
        for k in sorted(set(now) | set(then)):
            if now.get(k) != then.get(k):
                fields = sorted(
                    f
                    for f in set(now.get(k, {})) | set(then.get(k, {}))
                    if now.get(k, {}).get(f) != then.get(k, {}).get(f)
                )
                diffs.append(f"{' '.join(k)}: {', '.join(fields) or 'missing'}")
        if canon(profiles) != canon(phase5["efficiency"]):
            diffs.append("the efficiency block")
        diffs += [f"{label}: a stored ref differs from the pack label" for label in ref_diffs]
        if diffs:
            raise SystemExit(
                "the Phase 5 re-score does not reproduce "
                f"{PHASE5_TAG}:{PHASE5_RESULTS}; scoring stops:\n  " + "\n  ".join(diffs)
            )
        by_key = {(e["experiment"], e["system"], e["task"], e["eval_set"]): e for e in entries}
        for job in jobs:
            entry = by_key.get(job.key)
            if entry is not None and not entry.get("no_meta"):
                self.scored[(*job.key, job.subset)] = Scored(job, entry)
        return {
            "entries": len(entries),
            "profiles": len(profiles),
            "refs_checked": refs_checked,
            "result": "equal",
        }

    def phase5_units(self) -> None:
        """Per-unit values of the Phase 5 test files, for the comparisons and conditions."""
        for s in self.scored.values():
            if s.job.phase == 5 and EVAL_SETS[s.job.eval_set]["split"] == "test":
                s.units, _ = self.units_for(s.job, self.ctx)
                check_units(s.job, s.entry, s.units)

    # Step 2 --------------------------------------------------------------------------------

    def score_phase14(self, jobs: Sequence[Job]) -> None:
        for job in jobs:
            ctx = self.gctx if job.glypharis else self.ctx
            meta = glypharis_meta(job) if job.glypharis else None
            entry = eb.score_file(job.path, ctx, meta=meta)
            units, by_key = self.units_for(job, ctx)
            check_units(job, entry, units)
            self.scored[(*job.key, job.subset)] = Scored(
                job, entry, units, by_key if job.glypharis else None
            )
            print(f"scored {job.label}")

    def subset_entries(self) -> list[dict[str, Any]]:
        """Every OmniDocBench system on the 100 subset pages, from the per-page results already
        computed (a whole-set file is not scored, or logged, a second time)."""
        keep = set(subset_pages(self.data))
        out = []
        for s in self.scored.values():
            if s.job.eval_set != "omnidocbench_test" or s.job.task == "rec":
                continue
            if s.job.subset:
                out.append(s.entry)
                continue
            assert isinstance(s.units, PageUnits)
            keys = [k for k in s.units.keys if k in keep]
            out.append(self._block_entry(s.entry, s.units.select(keys), keys))
        return out

    def _block_entry(
        self, entry: dict[str, Any], units: PageUnits, keys: list[str]
    ) -> dict[str, Any]:
        spec = EVAL_SETS[entry["eval_set"]]
        if not spec["block_level"]:
            raise ValueError("the subset view is defined for the block-level set only")
        pages = self.ctx.pages(entry["eval_set"])
        refs = [ln.text for k in keys for ln in pages[k].lines if is_text_line(ln)]
        e = {
            k: entry.get(k)
            for k in (
                "experiment",
                "system",
                "version",
                "settings",
                "task",
                "eval_set",
                "split",
                "runs",
                "legacy_meta",
            )
        }
        e.update(samples=len(keys), expected=len(keys), subset=SUBSET_NAME, detection=None)
        e["tracks"] = {
            t: {
                **eb.score_blocks(units.results[t]),
                "charset_ceiling": eb.ceiling(refs, self.ctx.allowed[t], t),
            }
            for t in TRACKS
        }
        return e

    def test_files(self, task: str, eval_set: str) -> list[Scored]:
        return [
            s
            for s in self.scored.values()
            if s.job.task == task and s.job.eval_set == eval_set and s.units is not None
        ]

    def comparisons(self) -> list[dict[str, Any]]:
        """GLYPHARIS minus every baseline on the same crops or pages (descriptive intervals)."""
        out = []
        for exp, task in (("GW-REC-039", "rec"), ("GW-E2E-007", "e2e")):
            for eval_set in GLYPHARIS_RUNS[exp][1]:
                g = self.get(exp, GLYPHARIS, task, eval_set)
                if g is None or g.units is None:
                    continue
                for b in self.test_files(task, eval_set):
                    if b.job.glypharis:
                        continue
                    gu = g.units.select(b.units.keys)  # type: ignore[union-attr]
                    if task == "rec":
                        assert isinstance(gu, RecUnits) and isinstance(b.units, RecUnits)
                        diff = compare_rec(gu, b.units, rec_unit(eval_set))
                    else:
                        assert isinstance(gu, PageUnits) and isinstance(b.units, PageUnits)
                        diff = compare_e2e(gu, b.units, eval_set)
                    out.append(
                        {
                            "task": task,
                            "eval_set": eval_set,
                            "glypharis": exp,
                            "baseline": b.job.experiment,
                            "system": b.job.system,
                            "subset": SUBSET_NAME if b.job.subset else None,
                            "units": len(b.units.keys),
                            "difference": diff,
                        }
                    )
        return out

    def gdc_condition_tables(self, conditions: dict[str, str]) -> dict[str, Any]:
        """Every system on GDC per capture condition (information)."""
        pages = self.ctx.pages("gdc_test")
        cond_of_page = {k: conditions[p.image] for k, p in pages.items()}
        crops = self.pack("gdc_test")
        names = sorted(set(conditions.values()))
        rows: list[dict[str, Any]] = []
        for task in ("rec", "e2e", "oracle"):
            for s in self.test_files(task, "gdc_test"):
                for c in names:
                    row: dict[str, Any] = {
                        "experiment": s.job.experiment,
                        "system": s.job.system,
                        "task": task,
                        "condition": c,
                        "tracks": {},
                    }
                    if isinstance(s.units, RecUnits):
                        m = np.array([cond_of_page[crops[k].doc_id] == c for k in s.units.keys])
                        row["units"] = int(m.sum())
                        for t, a in s.units.tracks.items():
                            mt = m & (a["den"] > 0)  # empty references drop out, as in entries
                            row["tracks"][t] = {"cer": eb.ci(a["err"][mt], a["den"][mt], "lines")}
                    else:
                        assert isinstance(s.units, PageUnits)
                        keys = [k for k in s.units.keys if cond_of_page[k] == c]
                        u = s.units.select(keys)
                        row["units"] = len(keys)
                        for t in TRACKS:
                            row["tracks"][t] = {
                                "page_cer_bag_of_lines": eb.ci(*_bag(u, t), "pages"),
                                "page_cer_order_aware": eb.ci(*_aware(u, t), "pages"),
                            }
                    rows.append(row)
        return {"conditions": {c: Counter(cond_of_page.values())[c] for c in names}, "rows": rows}

    def gdc_scan_in_charset(self) -> list[dict[str, Any]]:
        """GDC-scan recognition without the lines holding characters outside the charset
        (information only; numbers only)."""
        out = []
        allowed = self.ctx.allowed["strict"]
        for s in self.test_files("rec", "gdc_scan_test"):
            assert isinstance(s.units, RecUnits)
            crops = self.pack("gdc_scan_test")
            keep = np.array([in_charset(crops[k].label, allowed) for k in s.units.keys])
            row: dict[str, Any] = {
                "experiment": s.job.experiment,
                "system": s.job.system,
                "lines": int(keep.sum()),
                "left_out": int((~keep).sum()),
                "tracks": {},
            }
            for t, a in s.units.tracks.items():
                mt = keep & (a["den"] > 0)
                row["tracks"][t] = {"cer": eb.ci(a["err"][mt], a["den"][mt], "lines")}
            out.append(row)
        return out

    def gdc_scan_in_charset_pages(self) -> list[dict[str, Any]]:
        """GDC-scan end to end and with the oracle boxes, the reference lines holding characters
        outside the charset made don't care (`in_charset_view`), every system with GDC-scan page
        files: bag-of-lines page CER and E2E F1 (information only; numbers only)."""
        allowed = self.ctx.allowed["strict"]
        views = {k: in_charset_view(p, allowed) for k, p in self.ctx.pages("gdc_scan_test").items()}
        out = []
        for task in ("e2e", "oracle"):
            for s in self.test_files(task, "gdc_scan_test"):
                assert isinstance(s.units, PageUnits)
                keys = s.units.keys
                by_key = s.records or {str(r["key"]): r for r in eb.read_records(s.job.path)}
                row: dict[str, Any] = {
                    "experiment": s.job.experiment,
                    "system": s.job.system,
                    "task": task,
                    "pages": len(keys),
                    "lines": sum(views[k][1] for k in keys),
                    "left_out": sum(views[k][2] for k in keys),
                    "tracks": {},
                }
                for t in TRACKS:
                    res = [evaluate_page(views[k][0], preds_of(by_key[k]), track=t) for k in keys]
                    tp = np.array([r.e2e_tp for r in res], dtype=np.float64)
                    row["tracks"][t] = {
                        "page_cer_bag_of_lines": eb.ci(
                            np.array([r.bag_errors for r in res], dtype=np.float64),
                            np.array([r.bag_ref_chars for r in res], dtype=np.float64),
                            "pages",
                        ),
                        "e2e_f1": eb.ci(
                            2 * tp,
                            np.array([r.e2e_pred + r.e2e_gt for r in res], dtype=np.float64),
                            "pages",
                        ),
                    }
                out.append(row)
        return out

    # Step 3 --------------------------------------------------------------------------------

    def official_hiertext(self, work: Path) -> list[dict[str, Any]]:
        """HierText's official line-level end-to-end score, every system with test pages."""
        gt = verified_raw(self.data, "hiertext", "gt_test")
        ids = official_hiertext.image_ids(gt)
        out = []
        for s in self.test_files("e2e", "hiertext_test"):
            records = eb.read_records(s.job.path)
            folder = work / "hiertext" / f"{s.job.experiment}_{s.job.system}"
            pred = official_hiertext.to_hiertext_predictions(records, image_ids=ids)
            official_hiertext.write_predictions(pred, folder / "predictions.json")
            scores = official_hiertext.run_hiertext(gt, folder / "predictions.json", folder)
            out.append(
                {
                    "experiment": s.job.experiment,
                    "system": s.job.system,
                    "pages": len(ids),
                    "pages_predicted": len(records),
                    "line": scores["line"],
                }
            )
        return out

    def official_omnidocbench(self, work: Path) -> list[dict[str, Any]]:
        """OmniDocBench's official text-block and reading-order edit distances: all English pages
        for every system that ran them, and the 100-page subset for every system."""
        gt_all = verified_raw(self.data, "omnidocbench", "annotations")
        pages = self.ctx.pages("omnidocbench_test")
        stem = {k: Path(p.image).stem for k, p in pages.items()}
        subset = subset_pages(self.data)
        scopes: dict[str, list[str]] = {"all": list(pages), SUBSET_NAME: subset}
        gts = {}
        for scope, keys in scopes.items():
            gts[scope] = work / "omnidocbench" / f"gt_{scope}.json"
            official_omnidocbench.filtered_ground_truth(gt_all, {stem[k] for k in keys}, gts[scope])
        out = []
        for s in self.test_files("e2e", "omnidocbench_test"):
            records = eb.read_records(s.job.path)
            for scope in (SUBSET_NAME,) if s.job.subset else ("all", SUBSET_NAME):
                keep = set(scopes[scope])
                name = f"{s.job.experiment}_{s.job.system}_{scope}"
                pred_dir = work / "omnidocbench" / name
                official_omnidocbench.write_predictions(
                    [
                        {**r, "image": pages[str(r["key"])].image}
                        for r in records
                        if str(r["key"]) in keep
                    ],
                    pred_dir,
                )
                res = official_omnidocbench.run_omnidocbench(
                    gts[scope], pred_dir, work / "omnidocbench" / f"{name}_work"
                )
                summary = {
                    el: {
                        k: res[el][k]
                        for k in ("ALL_page_avg", "edit_whole", "edit_sample_avg", "by_attribute")
                    }
                    for el in official_omnidocbench.ELEMENTS
                }
                out.append(
                    {
                        "experiment": s.job.experiment,
                        "system": s.job.system,
                        "scope": scope,
                        "pages": res["pages"],
                        "match_debug": res.get("match_debug", {}),
                        **summary,
                    }
                )
        return out

    # Step 4 --------------------------------------------------------------------------------

    def h1_cer(self, conditions: dict[str, str], sources: dict[str, str]) -> dict[str, Any]:
        g = self.get("GW-E2E-007", GLYPHARIS, "e2e", "gdc_test")
        p = self.get(EXPERIMENTS[LATENCY_BASELINE], LATENCY_BASELINE, "e2e", "gdc_test")
        if g is None or p is None or g.units is None or p.units is None:
            raise SystemExit("H1' needs GW-E2E-007 and GW-BASE-002 end to end on GDC")
        assert isinstance(g.units, PageUnits) and isinstance(p.units, PageUnits)
        pages = self.ctx.pages("gdc_test")
        clean_images = {img for img, c in conditions.items() if c in CLEAN_SLICE}
        clean = [k for k in p.units.keys if pages[k].image in clean_images]
        if len(clean) != len(clean_images) or not set(clean) <= set(g.units.keys):
            raise SystemExit("a clean-slice page is missing from a file; H1' waits for it")
        gu, pu = g.units.select(clean), p.units.select(clean)
        tests = {}
        for t in TRACKS:
            (eg, rg), (ep, rp) = _bag(gu, t), _bag(pu, t)
            tests[f"{t}_pages"] = iv(
                paired_ratio_diff_ci(eg, H1_FACTOR * ep, _same_ref(rg, rp), "pages")
            )
        (eg, rg), (ep, _) = _bag(gu, "strict"), _bag(pu, "strict")
        clusters = [sources[pages[k].image] for k in clean]
        tests["strict_documents"] = iv(
            cluster_paired_ratio_diff_ci(eg, ep, rg, clusters, "documents", factor=H1_FACTOR)
        )
        verdicts = {k: verdict(_interval(v), "below") for k, v in tests.items()}
        info: dict[str, Any] = {"order_aware": compare_e2e(gu, pu, "gdc_test", H1_FACTOR)}
        info["conditions"] = {}
        for c in sorted(set(conditions.values())):
            keys = [k for k in p.units.keys if conditions[pages[k].image] == c]
            info["conditions"][c] = compare_e2e(
                g.units.select(keys), p.units.select(keys), "gdc_test", H1_FACTOR
            )
        info["external"] = {}
        for eval_set in ("hiertext_test", "gdc_scan_test", "funsd_test", "omnidocbench_test"):
            gs = self.get("GW-E2E-007", GLYPHARIS, "e2e", eval_set)
            ps = self.get(EXPERIMENTS[LATENCY_BASELINE], LATENCY_BASELINE, "e2e", eval_set)
            if gs is None or ps is None or ps.units is None or gs.units is None:
                continue
            assert isinstance(ps.units, PageUnits)
            info["external"][eval_set] = compare_e2e(
                gs.units.select(ps.units.keys),  # type: ignore[arg-type]
                ps.units,
                eval_set,
                H1_FACTOR,
            )
        return {
            "rule": "CI of bag-of-lines page CER(GLYPHARIS) - 2 x CER(GW-BASE-002) on GDC's "
            "clean slice below 0; strict pages, normalised pages and strict source "
            "documents must agree",
            "pages": len(clean),
            "documents": len(set(clusters)),
            "tests": tests,
            "verdicts": verdicts,
            "verdict": combined(list(verdicts.values())),
            "information": info,
        }

    def h5(self, sources: dict[str, str]) -> dict[str, Any]:
        g = self.get("GW-REC-039", GLYPHARIS, "rec", "gdc_test")
        v = self.get("GW-REC-038", GLYPHARIS, "rec", "hiertext_val")
        if g is None or v is None:
            raise SystemExit("H5' needs GW-REC-039 on GDC and GW-REC-038")
        assert isinstance(g.units, RecUnits) and isinstance(v.units, RecUnits)
        ga, va = g.units.tracks["strict"], v.units.tracks["strict"]
        mg, mv = ga["den"] > 0, va["den"] > 0
        crops, pages = self.pack("gdc_test"), self.ctx.pages("gdc_test")
        clusters = [
            sources[pages[crops[k].doc_id].image]
            for k, keep in zip(g.units.keys, mg, strict=True)
            if keep
        ]
        line = unpaired_ratio_diff_ci(
            ga["err"][mg], ga["den"][mg], va["err"][mv], va["den"][mv], H5_FACTOR, "lines"
        )
        docs = cluster_unpaired_ratio_diff_ci(
            ga["err"][mg],
            ga["den"][mg],
            clusters,
            va["err"][mv],
            va["den"][mv],
            H5_FACTOR,
            "documents",
        )
        verdicts = {"lines": verdict(line, "above"), "documents": verdict(docs, "above")}
        info = {}
        for eval_set in ("hiertext_test", "funsd_test", "gdc_scan_test", "iiit5k_test"):
            s = self.get("GW-REC-039", GLYPHARIS, "rec", eval_set)
            if s is None:
                continue
            assert isinstance(s.units, RecUnits)
            a = s.units.tracks["strict"]
            m = a["den"] > 0
            info[eval_set] = {
                "cer_ratio": float(a["err"][m].sum() / a["den"][m].sum()) / _cer(va)
                if a["den"][m].sum() and _cer(va)
                else None,
                "difference": iv(
                    unpaired_ratio_diff_ci(
                        a["err"][m],
                        a["den"][m],
                        va["err"][mv],
                        va["den"][mv],
                        H5_FACTOR,
                        rec_unit(eval_set),
                    )
                ),
            }
        return {
            "rule": "CI of strict CER(GDC crops) - 1.5 x CER(HierText validation crops) "
            "above 0; line and source-document bootstraps must agree",
            "cer_gdc": _cer(ga),
            "cer_val": _cer(va),
            "tests": {"lines": iv(line), "documents": iv(docs)},
            "verdicts": verdicts,
            "verdict": combined(list(verdicts.values())),
            "information": info,
        }

    def calibration(self) -> dict[str, Any]:
        """H8′ on GDC's crops, and every other set's reliability as information."""
        out: dict[str, Any] = {"sets": {}}
        for eval_set in GLYPHARIS_RUNS["GW-REC-039"][1] + GLYPHARIS_RUNS["GW-REC-038"][1]:
            exp = "GW-REC-038" if eval_set == "hiertext_val" else "GW-REC-039"
            s = self.get(exp, GLYPHARIS, "rec", eval_set)
            if s is None or s.records is None:
                continue
            keys = list(s.units.keys) if s.units is not None else list(s.records)
            track: RecTrack = ALNUM if eval_set == "iiit5k_test" else "strict"
            y = correct_flags(s.records, keys, track)
            ct, ct1 = confidences(s.records, keys)
            clusters = None
            if eval_set == "gdc_test":
                crops = self.pack(eval_set)
                clusters = [crops[k].doc_id for k in keys]
            out["sets"][eval_set] = {
                "experiment": exp,
                **calibration_block(ct, ct1, y, track, clusters, "pages"),
            }
        g = self.get("GW-E2E-007", GLYPHARIS, "e2e", "gdc_test")
        if g is not None and g.records is not None and g.units is not None:
            out["gdc_pipeline_lines"] = pipeline_calibration(
                self.ctx.pages("gdc_test"), g.records, g.units.keys
            )
        gdc = out["sets"].get("gdc_test")
        if gdc is None:
            raise SystemExit("H8' needs GW-REC-039 on GDC")
        transfer = {
            "lines": verdict(_interval(gdc["diff_lines"]), "below"),
            "pages": verdict(_interval(gdc["diff_clusters"]), "below"),
        }
        half = {
            "lines": verdict(_interval(gdc["half_lines"]), "below"),
            "pages": verdict(_interval(gdc["half_clusters"]), "below"),
        }
        out["h8"] = {
            "rule": "GDC crops, strict exact match, 15 equal-mass bins: CI of ECE(T = 2.35) - "
            "ECE(T = 1) below 0 (transfers) and of ECE(T = 2.35) - 0.5 x ECE(T = 1) "
            "below 0 (by half); line and page bootstraps must agree",
            "transfers": {"verdicts": transfer, "verdict": combined(list(transfer.values()))},
            "by_half": {"verdicts": half, "verdict": combined(list(half.values()))},
        }
        out["h8"]["supported_on_gdc"] = (
            out["h8"]["transfers"]["verdict"] == "met" and out["h8"]["by_half"]["verdict"] == "met"
        )
        return out

    # Step 5 --------------------------------------------------------------------------------

    def handoff(
        self, folder: Path, conditions: dict[str, str], sources: dict[str, str]
    ) -> list[Path]:
        """Phase 15's inputs: everything it reads about the test sets, computed in this pass."""
        folder.mkdir(parents=True, exist_ok=True)  # run_score refused a non-empty one
        written: list[Path] = []
        e2e_match: dict[str, dict[str, tuple[bool, float, float]]] = {}
        for eval_set in GLYPHARIS_RUNS["GW-E2E-007"][1]:
            g = self.get("GW-E2E-007", GLYPHARIS, "e2e", eval_set)
            if g is None or g.records is None or g.units is None:
                continue
            e2e_match[eval_set], path = self._pairs(folder, eval_set, g, conditions)
            written.append(path)
            written.append(self._pages_file(folder, eval_set))
        for eval_set in GLYPHARIS_RUNS["GW-REC-039"][1]:
            s = self.get("GW-REC-039", GLYPHARIS, "rec", eval_set)
            if s is not None and s.records is not None:
                written.append(
                    self._rec_lines(
                        folder, eval_set, s, e2e_match.get(eval_set, {}), conditions, sources
                    )
                )
        s = self.get("GW-REC-039", GLYPHARIS, "rec", "gdc_test")
        if s is not None and s.records is not None:
            written.append(self._gdc_crops(folder, s))
        return written

    def _text_ok(self, eval_set: str) -> bool:
        return eval_set not in NUMBERS_ONLY

    def _pairs(
        self, folder: Path, eval_set: str, g: Scored, conditions: dict[str, str]
    ) -> tuple[dict[str, tuple[bool, float, float]], Path]:
        """The pipeline's matched pairs per page; returns line id -> (matched, IoU, best IoU)."""
        assert g.records is not None and g.units is not None
        pages = self.ctx.pages(eval_set)
        text = self._text_ok(eval_set)
        by_line: dict[str, tuple[bool, float, float]] = {}
        path = folder / f"e2e_pairs_{eval_set}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for k in g.units.keys:
                rec, page = g.records[k], pages[k]
                m = match_page(page, preds_of(rec))
                rows = []
                for gi, ln in enumerate(m.box_lines):
                    p, iou = m.matched.get(gi, (None, 0.0))
                    by_line[ln.line_id] = (p is not None, iou, m.best_iou[gi])
                    row: dict[str, Any] = {
                        "line_id": ln.line_id,
                        "pred": p,
                        "iou": round(iou, 4),
                        "best_iou": round(m.best_iou[gi], 4),
                        "rotation": box_rotation(ln.polygon),
                    }
                    ptext = rec["lines"][p]["text"] if p is not None else ""
                    for t in TRACKS:
                        s = score_pair(ptext, ln.text, t)
                        row[t] = {"char_errors": s.char_errors, "ref_chars": s.ref_chars}
                    if p is not None:
                        row["confidence"] = rec["lines"][p].get("confidence")
                        row["conf_t1"] = rec["lines"][p].get("conf_t1")
                        row["det_score"] = rec["lines"][p].get("det_score")
                    if text:
                        row["ref"], row["text"] = ln.text, ptext
                    rows.append(row)
                out = {
                    "key": k,
                    "pairs": rows,
                    "spurious": [i for i, st in enumerate(m.status) if st == "spurious"],
                    "ignored": [i for i, st in enumerate(m.status) if st == "ignored"],
                }
                if eval_set == "gdc_test":
                    out["condition"] = conditions[page.image]
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
        return by_line, path

    def _pages_file(self, folder: Path, eval_set: str) -> Path:
        """Per-page values of every system on a page set (numbers only), for paired analyses."""
        path = folder / f"pages_{eval_set}.jsonl"
        files = [s for task in ("e2e", "oracle") for s in self.test_files(task, eval_set)]
        rows: dict[str, dict[str, Any]] = {}
        for s in files:
            assert isinstance(s.units, PageUnits)
            name = f"{s.job.experiment} {s.job.system} {s.job.task}"
            for i, k in enumerate(s.units.keys):
                vals: dict[str, Any] = {}
                for t in TRACKS:
                    r = s.units.results[t][i]
                    vals[t] = {
                        "bag_errors": r.bag_errors,
                        "bag_ref_chars": r.bag_ref_chars,
                        "aware_errors": r.aware_errors,
                        "aware_ref_chars": r.aware_ref_chars,
                    }
                r = s.units.results["strict"][i]
                vals["det_0.5"] = list(r.det[0.5])
                rows.setdefault(k, {"key": k, "systems": {}})["systems"][name] = vals
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for k in sorted(rows):
                fh.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
        return path

    def _rec_lines(
        self,
        folder: Path,
        eval_set: str,
        s: Scored,
        det: dict[str, tuple[bool, float, float]],
        conditions: dict[str, str],
        sources: dict[str, str],
    ) -> Path:
        """GW-REC-039 per line: readings, per-line attributes, detector match, oracle reading."""
        assert s.records is not None and isinstance(s.units, RecUnits)
        crops = self.pack(eval_set)
        has_pages = EVAL_SETS[eval_set]["pages"] is not None
        pages = self.ctx.pages(eval_set) if has_pages else {}
        lines = {ln.line_id: ln for p in pages.values() for ln in p.lines}
        oracle = self._oracle_texts(eval_set)
        text = self._text_ok(eval_set)
        path = folder / f"rec_lines_{eval_set}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for k in s.units.keys:
                rec, crop = s.records[k], crops[k]
                row: dict[str, Any] = {"key": k, "doc_id": crop.doc_id}
                for t in rec_tracks(eval_set):
                    sc = score_pair(rec["pred"], rec["ref"], t)
                    row[t] = {
                        "char_errors": sc.char_errors,
                        "ref_chars": sc.ref_chars,
                        "exact": sc.exact,
                    }
                # The runner's per-line numbers (confidences, time, crop height, blur, contrast),
                # flat or in a block of numbers; never a string, so no text slips through.
                row.update(
                    {
                        f: v
                        for f, v in rec.items()
                        if f not in ("key", "ref", "pred")
                        and (
                            _number(v)
                            or (isinstance(v, dict) and v and all(_number(x) for x in v.values()))
                        )
                    }
                )
                if k in lines:
                    row["rotation"] = box_rotation(lines[k].polygon)
                if k in det:
                    row["det_matched"], row["det_iou"], row["det_best_iou"] = det[k]
                if eval_set == "gdc_test":
                    image = pages[crop.doc_id].image
                    row["condition"], row["source"] = conditions[image], sources[image]
                if k in oracle:
                    sc = score_pair(oracle[k], rec["ref"], "strict")
                    row["oracle_strict"] = {"char_errors": sc.char_errors, "exact": sc.exact}
                    if text:
                        row["oracle_pred"] = oracle[k]
                if text:
                    row["ref"], row["pred"] = rec["ref"], rec["pred"]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def _oracle_texts(self, eval_set: str) -> dict[str, str]:
        o = self.get("GW-E2E-008", GLYPHARIS, "oracle", eval_set)
        if o is None or o.records is None:
            return {}
        return {
            str(ln["line_id"]): str(ln["text"])
            for r in o.records.values()
            for ln in r["lines"]
            if "line_id" in ln
        }

    def _gdc_crops(self, folder: Path, s: Scored) -> Path:
        """GDC line crops of every line read with an error (strict) and 200 correct lines drawn
        with seed 20261115; the crops are copied byte for byte from the pack."""
        assert s.records is not None and s.units is not None
        y = correct_flags(s.records, s.units.keys, "strict")
        wrong = [k for k, ok in zip(s.units.keys, y, strict=True) if not ok]
        right = sorted(k for k, ok in zip(s.units.keys, y, strict=True) if ok)
        n = min(HANDOFF_CORRECT, len(right))
        picks = np.sort(np.random.default_rng(HANDOFF_SEED).choice(len(right), n, replace=False))
        chosen = [("error", k) for k in wrong] + [("correct", right[int(i)]) for i in picks]
        reader = LinePackReader(self.data / EVAL_SETS["gdc_test"]["pack"])
        pos = {r.key: i for i, r in enumerate(reader.records)}
        out = folder / "gdc_crops"
        for kind in ("error", "correct"):
            (out / kind).mkdir(parents=True, exist_ok=True)
        index = out / "index.jsonl"
        with index.open("w", encoding="utf-8", newline="\n") as fh:
            for kind, k in chosen:
                raw = reader.read_bytes(pos[k])
                ext = ".png" if raw.startswith(b"\x89PNG") else ".jpg"
                # Line ids hold slashes; a hash of the id makes a flat, stable file name.
                name = f"{kind}/{hashlib.sha256(k.encode()).hexdigest()[:16]}{ext}"
                (out / name).write_bytes(raw)
                rec = s.records[k]
                fh.write(
                    json.dumps(
                        {
                            "key": k,
                            "kind": kind,
                            "file": name,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "ref": rec["ref"],
                            "pred": rec["pred"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        reader.close()
        return index


def _number(v: Any) -> bool:
    return v is None or (isinstance(v, int | float) and not isinstance(v, bool))


def _cer(a: dict[str, np.ndarray]) -> float:
    m = a["den"] > 0
    return float(a["err"][m].sum() / a["den"][m].sum()) if a["den"][m].sum() else 0.0


def _interval(d: dict[str, Any]) -> Interval:
    return Interval(d["value"], d["low"], d["high"], d["resamples"], d["unit"])


def canon(x: Any) -> Any:
    """A value as JSON would store it, so a re-score compares exactly with a stored report."""
    return json.loads(json.dumps(x))


def verified_raw(data: Path, dataset: str, file_key: str) -> Path:
    """A raw benchmark file the official evaluators read, checked against its manifest hash."""
    manifest = json.loads((data / "manifests" / f"{dataset}.json").read_text(encoding="utf-8"))
    entry = manifest["files"][file_key]
    path = data / entry["path"]
    if sha256_file(path) != entry["sha256"]:
        raise SystemExit(f"{entry['path']} differs from its hash in manifests/{dataset}.json")
    return path


# --------------------------------------------------------------------------------- latency


def _count(v: Any) -> int | None:
    """A thread count: an int, never a bool (``True`` is an int in Python)."""
    return v if type(v) is int else None


def profile_problems(
    name: str, p: dict[str, Any], runs: dict[str, Any] | None, threads: int = PROFILE_THREADS
) -> list[str]:
    """Why one latency profile cannot be used (empty if it can).

    Both runners record the count as ``settings.cpu_threads``; GLYPHARIS's profiler also records
    the torch pools (``threads.intra_op``, ``threads.inter_op``), which must equal it, and whether
    the thread variables were set before torch was imported. A GLYPHARIS profile (``runs`` given:
    `check_glypharis_runs`'s result) must also come from the pinned code at the prediction runs'
    tag and from the freeze record's model files."""
    out = []
    cond = p.get("conditions") or {}
    if cond.get("on_ac_power") is not True:
        out.append(f"{name}: not on mains power")
    if "balanced" not in str(cond.get("power_plan") or "").lower():
        out.append(f"{name}: not the Balanced power plan")
    if cond.get("power_mode") is None:
        out.append(f"{name}: power mode not recorded")
    n = _count((p.get("settings") or {}).get("cpu_threads"))
    if n != threads:
        out.append(f"{name}: {n} threads, not {threads}")
    if (
        p.get("batch") != 1
        or len(p.get("times_ms") or []) != PROFILE_SAMPLES
        or p.get("warmup") != WARMUP
    ):
        out.append(f"{name}: not {WARMUP} warm-up and {PROFILE_SAMPLES} timed pages at batch 1")
    if runs is None:
        return out
    t = p.get("threads") or {}
    pools = {k: _count(t.get(k)) for k in ("intra_op", "inter_op")}
    if any(v != threads for v in pools.values()):
        out.append(f"{name}: torch pools {pools}, not {threads}")
    if t.get("env_before_torch_import") is not True:
        out.append(f"{name}: the thread variables were not set before torch was imported")
    pin = p.get("pinned") or {}
    if pin.get("trees") != "equal":
        out.append(f"{name}: not run at the pinned code")
    elif pin.get("tag_commit") != runs.get("tag_commit"):
        out.append(f"{name}: pinned to another tag commit than the prediction runs")
    prov = p.get("provenance") or {}
    for k in MODEL_HASHES:
        if prov.get(k) != runs.get(k):
            out.append(f"{name}: {k} differs from the freeze record or charset")
    return out


MODEL_HASHES = (
    "record_sha256",
    "recognizer_checkpoint_sha256",
    "detector_checkpoint_sha256",
    "charset_sha256",
)


def _utc(v: Any) -> datetime | None:
    """An ISO 8601 time with its UTC offset (as both profilers write ``started_utc`` and
    ``ended_utc``); None when it is missing, unreadable or has no offset."""
    if not isinstance(v, str):
        return None
    try:
        t = datetime.fromisoformat(v)
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def sitting_problems(base: dict[str, Any], glyph: dict[str, Any]) -> list[str]:
    """Why the two 10-thread profiles are not one sitting with the baseline first: the plan's
    "a fresh GW-BASE-002 profile in one sitting: the baseline first, then GLYPHARIS". GW-EFF-005
    must start no earlier than the baseline ended and at most `SITTING_GAP_HOURS` later; a
    missing or unreadable time is void."""
    out = []
    times = {}
    for name, p in (("GW-BASE-002", base), ("GW-EFF-005", glyph)):
        start, end = _utc(p.get("started_utc")), _utc(p.get("ended_utc"))
        if start is None or end is None:
            out.append(f"{name}: started_utc and ended_utc (ISO 8601 with offset) not recorded")
        elif end < start:
            out.append(f"{name}: ended_utc is before started_utc")
        times[name] = (start, end)
    if out:
        return out
    base_end, glyph_start = times["GW-BASE-002"][1], times["GW-EFF-005"][0]
    assert base_end is not None and glyph_start is not None
    if base_end > glyph_start:
        out.append("GW-EFF-005 started before GW-BASE-002 ended: the baseline runs first")
    elif glyph_start - base_end > timedelta(hours=SITTING_GAP_HOURS):
        out.append(
            f"GW-EFF-005 started more than {SITTING_GAP_HOURS} hours after GW-BASE-002 ended: "
            "not one sitting"
        )
    return out


def latency_void(paths: dict[str, Path], runs: dict[str, Any]) -> list[str]:
    """Every recorded condition that makes H1′'s latency sitting void (Phase 14 item 5).

    Run in preflight, before any score: a void sitting is found mechanically, moved aside by hand
    and repeated, and the first sitting that is not void is the one of record. Only the two
    10-thread files decide it (their conditions, their input sizes and their order in one
    sitting); the 16-thread file is information and its problems are reported beside it
    (`h1_latency`), never as a void."""
    base = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    glyph = json.loads(paths["glypharis"].read_text(encoding="utf-8"))
    void = profile_problems("GW-BASE-002", base, None)
    void += profile_problems("GW-EFF-005", glyph, runs)
    if base.get("input_sizes") != glyph.get("input_sizes"):
        void.append("the input sizes differ page by page")
    void += sitting_problems(base, glyph)
    return void


def h1_latency(paths: dict[str, Path], runs: dict[str, Any]) -> dict[str, Any]:
    """H1′'s latency half: the fresh GW-BASE-002 profile and GW-EFF-005, one sitting.

    Preflight has already refused a void sitting; the check is repeated here so the verdict can
    never rest on one."""
    base = json.loads(paths["baseline"].read_text(encoding="utf-8"))
    glyph = json.loads(paths["glypharis"].read_text(encoding="utf-8"))
    void = latency_void(paths, runs)
    out: dict[str, Any] = {
        "rule": "mean_ci of per-page time(GLYPHARIS, 10 threads) - time(GW-BASE-002, 10 threads), "
        "same pages in the same order, below 0",
        "files": {
            k: {"file": shown(p), "sha256": sha256_file(p)}
            for k, p in paths.items()
            if k in ("baseline", "glypharis")
        },
        "times_ms": {"GW-BASE-002": base["times_ms"], "GW-EFF-005": glyph["times_ms"]},
        "mean_ms": {
            "GW-BASE-002": float(np.mean(base["times_ms"])),
            "GW-EFF-005": float(np.mean(glyph["times_ms"])),
        },
        "power_mode": {
            "GW-BASE-002": (base.get("conditions") or {}).get("power_mode"),
            "GW-EFF-005": (glyph.get("conditions") or {}).get("power_mode"),
        },
        "sitting": {
            name: {k: p.get(k) for k in ("started_utc", "ended_utc")}
            for name, p in (("GW-BASE-002", base), ("GW-EFF-005", glyph))
        },
        "disclosed": "the baseline runs PaddleOCR's two orientation models, as for its accuracy "
        "numbers",
    }
    if void:
        out.update(void=void, verdict="void")
    else:
        diff = np.asarray(glyph["times_ms"], dtype=np.float64) - np.asarray(base["times_ms"])
        test = mean_ci(diff, "pages")
        out.update(test=iv(test), verdict=verdict(test, "below"))
    info: dict[str, Any] = {}
    if paths["phase5_baseline"].is_file():
        info["phase5_baseline_mean_ms"] = eb.read_profile(paths["phase5_baseline"])["mean_ms"]
    if paths["glypharis_16t"].is_file():
        p16 = json.loads(paths["glypharis_16t"].read_text(encoding="utf-8"))
        info["glypharis_16t_mean_ms"] = float(np.mean(p16["times_ms"]))
        info["glypharis_16t_sha256"] = sha256_file(paths["glypharis_16t"])
        info["glypharis_16t_problems"] = profile_problems(
            "GW-EFF-005 (16 threads)", p16, runs, threads=16
        )
    out["information"] = info
    return out


# ---------------------------------------------------------------------------------- report

NOTE = "Generated by `scripts/phase14_score.py` (GW-E2E-009); do not edit by hand."


def pct(d: dict[str, Any] | None) -> str:
    return eb.pct(d)


def pp(d: dict[str, Any] | None, scale: float = 100.0, digits: int = 2) -> str:
    """A difference with its CI, signed: percentage points by default."""
    if d is None:
        return "n/a"
    v, lo, hi = (scale * d[k] for k in ("value", "low", "high"))
    return f"{v:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def _name(system: str) -> str:
    return "GLYPHARIS v1" if system == GLYPHARIS else SYSTEM_NAMES.get(system, system)


def _set(e: dict[str, Any]) -> str:
    return e["eval_set"] + (f" ({e['subset']})" if e.get("subset") else "")


def _n(e: dict[str, Any]) -> str:
    return f"{e['samples']:,}" + ("" if e["samples"] == e["expected"] else f" of {e['expected']:,}")


def render(res: dict[str, Any]) -> dict[str, str]:
    entries = res["predictions"]
    order = {s: i for i, s in enumerate(EVAL_SETS)}
    entries = sorted(entries, key=lambda e: (order[e["eval_set"]], e["experiment"], e["task"]))
    rec = [e for e in entries if e["task"] == "rec"]
    pages = [e for e in entries if e["task"] != "rec"]
    out: dict[str, str] = {}

    lines = [
        "# Phase 14: recognition",
        "",
        f"{NOTE} Ground-truth crops in, text out; CER and WER "
        "in % with the 95% bootstrap CI over lines (words for IIIT5K). The charset ceiling is "
        "the share of reference characters outside the GLYPHARIS charset.",
        "",
    ]
    head = ["ID", "System", "Set", "Lines", "CER %", "WER %", "Seq acc %", "Charset ceiling %"]
    for t in TRACKS:
        rows = [
            [
                e["experiment"],
                _name(e["system"]),
                _set(e),
                _n(e),
                pct(e["tracks"][t]["cer"]),
                pct(e["tracks"][t]["wer"]),
                f"{100 * e['tracks'][t]['sequence_accuracy']:.2f}",
                f"{100 * e['tracks'][t]['charset_ceiling']:.2f}",
            ]
            for e in rec
        ]
        lines += [f"## Track: {t}", "", *table(head, rows)]
    alnum = [e for e in rec if ALNUM in e["tracks"]]
    if alnum:
        lines += [
            "## IIIT5K, alphanumeric track (headline)",
            "",
            "Both sides lower-cased, then "
            "every character outside 0-9a-z removed; lexicon-free; images as published. The "
            "strict track above charges case against the upper-case labels.",
            "",
        ]
        rows = [
            [
                e["experiment"],
                _name(e["system"]),
                _n(e),
                pct(e["tracks"][ALNUM]["word_accuracy"]),
                pct(e["tracks"][ALNUM]["cer"]),
            ]
            for e in alnum
        ]
        lines += table(["ID", "System", "Words", "Word acc %", "CER %"], rows)
    if res["gdc_scan_in_charset"]:
        lines += [
            "## GDC-scan without out-of-charset lines (information)",
            "",
            "GDC-scan is one document: a case study whose CIs are conditional on it.",
            "",
        ]
        rows = [
            [
                r["experiment"],
                _name(r["system"]),
                f"{r['lines']:,}",
                str(r["left_out"]),
                pct(r["tracks"]["strict"]["cer"]),
                pct(r["tracks"]["normalised"]["cer"]),
            ]
            for r in res["gdc_scan_in_charset"]
        ]
        lines += table(
            ["ID", "System", "Lines", "Left out", "CER % strict", "CER % normalised"], rows
        )
    out["recognition.md"] = "\n".join(lines)

    lines = [
        "# Phase 14: detection and end to end",
        "",
        f"{NOTE} Values in % with the 95% "
        "bootstrap CI over pages. Order-aware page CER is reported where the set has a "
        "reading order (GDC, GDC-scan, OmniDocBench); on OmniDocBench it is the project's "
        "metric, not the official protocol, and detection, bag-of-lines and E2E F1 are n/a "
        "because its reference unit is a layout block.",
        "",
    ]
    rows = []
    for e in pages:
        d = e.get("detection")
        if d is None:
            rows.append(
                [e["experiment"], _name(e["system"]), e["task"], _set(e), _n(e), *["n/a"] * 3]
            )
        else:
            rows.append(
                [
                    e["experiment"],
                    _name(e["system"]),
                    e["task"],
                    _set(e),
                    _n(e),
                    pct(d["iou_0.5"]["precision"]),
                    pct(d["iou_0.5"]["recall"]),
                    pct(d["iou_0.5"]["f1"]),
                ]
            )
    lines += [
        "## Detection",
        "",
        *table(["ID", "System", "Task", "Set", "Pages", "P @0.5", "R @0.5", "F1 @0.5"], rows),
    ]
    head = [
        "ID",
        "System",
        "Task",
        "Set",
        "Pages",
        "Page CER % (order-aware)",
        "Page CER % (bag of lines)",
        "E2E F1 %",
        "Charset ceiling %",
    ]
    for t in TRACKS:
        rows = [
            [
                e["experiment"],
                _name(e["system"]),
                e["task"],
                _set(e),
                _n(e),
                pct(e["tracks"][t].get("page_cer_order_aware")),
                pct(e["tracks"][t].get("page_cer_bag_of_lines")),
                pct(e["tracks"][t].get("e2e_f1")),
                f"{100 * e['tracks'][t]['charset_ceiling']:.2f}",
            ]
            for e in [*pages, *res["subset"]]
        ]
        lines += [f"## Track: {t}", "", *table(head, rows)]
    vert = [e for e in pages if "vertical" in e]
    if vert:
        rows = [
            [
                e["experiment"],
                _name(e["system"]),
                e["task"],
                e["eval_set"],
                str(e["vertical"]["strict"]["lines"]),
                pct(e["vertical"]["strict"]["recall"]),
                pct(e["vertical"]["strict"]["cer"]),
                pct(e["vertical"]["normalised"]["cer"]),
            ]
            for e in vert
        ]
        lines += [
            "## Vertical lines (GDC, GDC-scan)",
            "",
            *table(
                [
                    "ID",
                    "System",
                    "Task",
                    "Set",
                    "Lines",
                    "Recall @0.5 %",
                    "CER % strict",
                    "CER % normalised",
                ],
                rows,
            ),
        ]
    scan = res.get("gdc_scan_in_charset_pages") or []
    if scan:
        lines += [
            "## GDC-scan without out-of-charset lines (information)",
            "",
            "Reference lines holding a character outside the GLYPHARIS charset are don't care "
            "(a prediction on one is ignored). GDC-scan is one document: a case study whose CIs "
            "are conditional on it.",
            "",
        ]
        rows = [
            [
                r["experiment"],
                _name(r["system"]),
                r["task"],
                str(r["pages"]),
                f"{r['lines']:,}",
                str(r["left_out"]),
                pct(r["tracks"]["strict"]["page_cer_bag_of_lines"]),
                pct(r["tracks"]["normalised"]["page_cer_bag_of_lines"]),
                pct(r["tracks"]["strict"]["e2e_f1"]),
            ]
            for r in scan
        ]
        lines += table(
            [
                "ID",
                "System",
                "Task",
                "Pages",
                "Lines",
                "Left out",
                "Page CER % strict (bag of lines)",
                "Page CER % normalised (bag of lines)",
                "E2E F1 % strict",
            ],
            rows,
        )
    cond = res["gdc_conditions"]
    if cond:
        lines += [
            "## GDC per capture condition (information)",
            "",
            "Pages per condition: "
            + ", ".join(f"{c} {n}" for c, n in cond["conditions"].items())
            + ".",
            "",
        ]
        rows = []
        for r in cond["rows"]:
            t = r["tracks"]["strict"]
            val = t["cer"] if "cer" in t else t["page_cer_bag_of_lines"]
            aware = t.get("page_cer_order_aware")
            rows.append(
                [
                    r["experiment"],
                    _name(r["system"]),
                    r["task"],
                    r["condition"],
                    str(r["units"]),
                    pct(val),
                    pct(aware),
                ]
            )
        lines += table(
            [
                "ID",
                "System",
                "Task",
                "Condition",
                "Units",
                "CER % strict (crops or bag of lines)",
                "Page CER % strict (order-aware)",
            ],
            rows,
        )
    out["end_to_end.md"] = "\n".join(lines)

    lines = [
        "# Phase 14: official evaluators",
        "",
        f"{NOTE} Pinned evaluators (`data/manifests/evaluators.json`) in their own environments.",
        "",
        "## HierText, line level, end to end (`--eval_lines --e2e`)",
        "",
        "Each predicted line is given as a line holding one word with the line's polygon and "
        "text. The word-level score needs word output, which no system produces.",
        "",
    ]
    keys = sorted({k for r in res["official"]["hiertext"] for k in r["line"]})
    rows = [
        [
            r["experiment"],
            _name(r["system"]),
            f"{r['pages_predicted']:,} of {r['pages']:,}",
            *(f"{r['line'][k]:.4f}" for k in keys),
        ]
        for r in res["official"]["hiertext"]
    ]
    lines += table(["ID", "System", "Pages", *keys], rows)
    lines += [
        "## OmniDocBench, text block and reading order (Edit_dist, quick_match)",
        "",
        "Formula and table metrics off, so the Overall score is not computable. Lower is better.",
        "",
    ]
    rows = [
        [
            r["experiment"],
            _name(r["system"]),
            r["scope"],
            str(r["pages"]),
            f"{r['text_block']['ALL_page_avg']:.4f}",
            f"{r['text_block']['edit_whole']:.4f}",
            f"{r['reading_order']['ALL_page_avg']:.4f}",
        ]
        for r in res["official"]["omnidocbench"]
    ]
    lines += table(
        [
            "ID",
            "System",
            "Pages",
            "Count",
            "text_block page avg",
            "text_block whole",
            "reading_order page avg",
        ],
        rows,
    )
    out["official.md"] = "\n".join(lines)

    lines = [
        "# Phase 14: GLYPHARIS minus each baseline (descriptive)",
        "",
        f"{NOTE} Differences "
        "in percentage points with the 95% paired bootstrap CI over the same lines or pages. "
        "These intervals are descriptive and carry no verdict.",
        "",
    ]
    rows = []
    for c in res["comparisons"]:
        for t in (*((ALNUM,) if ALNUM in c["difference"] else ()), *TRACKS):
            d = c["difference"].get(t)
            if d is None:
                continue
            main = d.get("cer") or d.get("page_cer_bag_of_lines") or d.get("page_cer_order_aware")
            rows.append(
                [
                    c["glypharis"],
                    c["baseline"],
                    _name(c["system"]),
                    c["task"],
                    c["eval_set"] + (f" ({c['subset']})" if c["subset"] else ""),
                    t,
                    f"{c['units']:,}",
                    pp(main),
                    pp(d.get("page_cer_order_aware"))
                    if c["task"] != "rec"
                    else pp(d.get("word_accuracy")),
                ]
            )
    lines += table(
        [
            "GLYPHARIS",
            "Baseline",
            "System",
            "Task",
            "Set",
            "Track",
            "Units",
            "CER (crops or bag of lines) pp",
            "Order-aware CER / word acc pp",
        ],
        rows,
    )
    out["comparisons.md"] = "\n".join(lines)

    cal = res["calibration"]
    lines = [
        "# Phase 14: calibration",
        "",
        f"{NOTE} Line confidences of the frozen recognizer at "
        "T = 1 and T = 2.35; ECE with 15 equal-mass bins; a line is right when it matches "
        "its label exactly on the strict track (IIIT5K: alphanumeric track). Only GDC's "
        "crops carry H8′'s verdicts (README.md); every other row is information.",
        "",
    ]
    rows = [
        [
            s["experiment"],
            name,
            f"{s['lines']:,}",
            s["track"],
            f"{s['accuracy']:.4f}",
            f"{s['ece_t1']:.4f}",
            f"{s['ece_t']:.4f}",
            pp(s["diff_lines"], 1.0, 4),
        ]
        for name, s in cal["sets"].items()
    ]
    if "gdc_pipeline_lines" in cal:
        s = cal["gdc_pipeline_lines"]
        rows.append(
            [
                "GW-E2E-007",
                "gdc_test (pipeline lines)",
                f"{s['lines']:,}",
                s["track"],
                f"{s['accuracy']:.4f}",
                f"{s['ece_t1']:.4f}",
                f"{s['ece_t']:.4f}",
                pp(s["diff_lines"], 1.0, 4),
            ]
        )
    lines += table(
        [
            "ID",
            "Set",
            "Lines",
            "Track",
            "Accuracy",
            "ECE T=1",
            "ECE T=2.35",
            "ECE(2.35) - ECE(1) [CI]",
        ],
        rows,
    )
    lines += [f"![{n}](reliability_{n}.png)" for n in cal["sets"]] + [""]
    out["calibration.md"] = "\n".join(lines)
    out["README.md"] = render_summary(res)
    return out


def _verdict_line(name: str, v: str) -> str:
    return f"- **{name}: {v}**"


def render_summary(res: dict[str, Any]) -> str:
    h = res["hypotheses"]
    prov = res["provenance"]
    rep = res["reproduction"]
    lines = [
        "# Phase 14: external evaluation (GW-E2E-009)",
        "",
        f"{NOTE} One scoring pass over every complete prediction file. Scorer commit "
        f"`{prov['scorer']['commit']}` (dirty: {prov['scorer']['dirty']}); pinned trees against "
        f"`phase14-eval`: {prov['pinned'].get('trees')}. Bootstrap: 1,000 resamples, seed "
        "20261010, 95% percentile. Tables: [recognition.md](recognition.md), "
        "[end_to_end.md](end_to_end.md), [official.md](official.md), "
        "[comparisons.md](comparisons.md), [calibration.md](calibration.md); every number is in "
        "`results.json`.",
        "",
        "## Phase 5 reproduction",
        "",
        f"The Phase 5 entries ({rep['entries']}) and profiles ({rep['profiles']}) re-scored here "
        f"equal `{PHASE5_TAG}:{PHASE5_RESULTS}` exactly (tag commit `{rep['tag_commit']}`), "
        f"and {rep['refs_checked']:,} "
        "stored references equal the current pack labels.",
        "",
        "## Confirmatory tests (Phase 14 plan, item 5)",
        "",
        "Met: the CI lies on the claimed side of 0; not met: on the other side; otherwise not "
        "shown.",
        "",
    ]
    h1 = h["H1′ CER"]
    t = h1["tests"]
    lines += [
        _verdict_line("H1′, CER half", h1["verdict"]),
        f"  - {h1['pages']} clean-slice pages, {h1['documents']} source documents; CER(GLYPHARIS) "
        "- 2 x CER(PP-OCRv5 mobile), bag of lines, in percentage points:",
        f"  - strict, pages: {pp(t['strict_pages'])} ({h1['verdicts']['strict_pages']})",
        f"  - normalised, pages: {pp(t['normalised_pages'])} "
        f"({h1['verdicts']['normalised_pages']})",
        f"  - strict, documents: {pp(t['strict_documents'])} "
        f"({h1['verdicts']['strict_documents']})",
    ]
    lat = h["H1′ latency"]
    if lat["verdict"] == "void":
        lines += [
            _verdict_line("H1′, latency half", "void sitting"),
            *(f"  - {v}" for v in lat["void"]),
        ]
    else:
        lines += [
            _verdict_line("H1′, latency half", lat["verdict"]),
            f"  - mean per-page difference (GLYPHARIS - PP-OCRv5 mobile, 10 threads): "
            f"{pp(lat['test'], 1.0)} ms; means {lat['mean_ms']['GW-EFF-005']:.1f} ms and "
            f"{lat['mean_ms']['GW-BASE-002']:.1f} ms. Disclosed: {lat['disclosed']}.",
        ]
    lines.append(f"- **H1′ holds: {'yes' if h['H1′ holds'] else 'no'}** (it needs both halves met)")
    h5 = h["H5′"]
    lines += [
        _verdict_line("H5′", h5["verdict"]),
        f"  - strict CER on GDC crops {100 * h5['cer_gdc']:.2f}%, on HierText validation crops "
        f"(GW-REC-038) {100 * h5['cer_val']:.2f}%; CER_GDC - 1.5 x CER_val: lines "
        f"{pp(h5['tests']['lines'])} ({h5['verdicts']['lines']}), documents "
        f"{pp(h5['tests']['documents'])} ({h5['verdicts']['documents']})",
    ]
    h8 = h["H8′"]
    lines += [
        _verdict_line("H8′, transfers", h8["transfers"]["verdict"]),
        _verdict_line("H8′, transfers by half", h8["by_half"]["verdict"]),
        f"  - H8 supported on GDC: {'yes' if h8['supported_on_gdc'] else 'no'} (it needs both "
        "met). Intervals in [calibration.md](calibration.md).",
        "",
        "## Notes",
        "",
        "- GDC-scan is one document: a case study whose CIs are conditional on it. Every tracked "
        "file carries its numbers only.",
        "- OmniDocBench's order-aware page CER is the project's metric, not the official protocol; "
        "the official components are in [official.md](official.md).",
        "- The comparisons and every interval outside the five tests above are descriptive.",
        "",
    ]
    return "\n".join(lines)


def diagrams(out: Path, cal: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, s in cal["sets"].items():
        fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=120)
        ax.plot([0, 1], [0, 1], color="#9a9a9a", lw=1, ls="--", label="perfect calibration")
        for pts, label, colour in (
            (s["reliability_t1"], "T = 1", "#c0392b"),
            (s["reliability_t"], "T = 2.35", "#1f5fa8"),
        ):
            ax.plot(
                [p["confidence"] for p in pts],
                [p["accuracy"] for p in pts],
                marker="o",
                ms=3.5,
                lw=1.5,
                color=colour,
                label=label,
            )
        ax.set_xlabel("mean confidence in bin")
        ax.set_ylabel("share of lines read exactly right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Reliability, {name} (15 equal-mass bins)", fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(out / f"reliability_{name}.png")
        plt.close(fig)


# ------------------------------------------------------------------------------------ main


@dataclass
class Options:
    data: Path
    out: Path
    eff_dir: Path
    access: Path
    record: Path
    resume: bool = False


def check_information_profile(path: Path) -> None:
    """The 16-thread profile is information and never voids the sitting, but `h1_latency` reads
    it after the scoring rows are logged: refuse an unreadable one now, before anything is."""
    if not path.is_file():
        return
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        times = doc["times_ms"]
        ok = isinstance(times, list) and bool(times) and all(_number(t) for t in times)
    except (OSError, ValueError, KeyError, TypeError):
        ok = False
    if not ok:
        raise SystemExit(f"{shown(path)} is not a readable profile (times_ms); move it aside")


def preflight(opts: Options, write: bool) -> dict[str, Any]:
    """Every refusal that needs no score (step 0), all of them before anything is written.

    Only when every check has passed are the baseline files and the fresh baseline profile added
    to phase14_outputs.json (``write``; `check` writes nothing). A refused pass therefore leaves
    the manifest as it was, and a repeated latency sitting can be recorded afresh."""
    try:
        require_branch(PHASE14_BRANCH)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    data = opts.data
    if not opts.access.is_file():  # before the pass is marked started, so nothing is half-done
        raise SystemExit(f"the test-access log {shown(opts.access)} does not exist")
    g_jobs, b_jobs = glypharis_jobs(data), baseline_jobs(data)
    counts = check_complete(data, [*g_jobs, *b_jobs])
    phase5 = verify_phase5(data)
    lat = latency_paths(data, opts.eff_dir)
    for k in ("baseline", "glypharis"):
        if not lat[k].is_file():
            raise SystemExit(f"{shown(lat[k])} is missing: the H1' latency sitting comes first")
    # The GLYPHARIS runner records each complete output itself; one it did not record is not a
    # finished pass. The baseline runner records nothing, so its outputs are added at the end.
    unrecorded = check_outputs(data, [file_entry(j.path, data) for j in g_jobs])
    if unrecorded:
        raise SystemExit(
            f"{len(unrecorded)} GLYPHARIS outputs are not in {OUTPUTS.as_posix()}, e.g. "
            f"{unrecorded[0]['file']}; their runs did not finish"
        )
    outputs = [file_entry(j.path, data) for j in b_jobs]
    outputs.append(file_entry(lat["baseline"], data))
    new = check_outputs(data, outputs)
    runs = check_glypharis_runs(g_jobs, opts.record, data / "manifests" / "charset.json")
    void = latency_void(lat, runs)
    if void:
        raise SystemExit(
            "the latency sitting is void; nothing is scored. Move both profiles aside (record it "
            "in research/failures.md) and repeat the sitting:\n  " + "\n  ".join(void)
        )
    check_information_profile(lat["glypharis_16t"])
    inputs = check_job_inputs(data, [*g_jobs, *b_jobs])
    evaluators = check_evaluators(data)
    phase5_test = [
        j
        for j in phase5_jobs(data)
        if EVAL_SETS[j.eval_set]["split"] == "test" and j.path.with_suffix(".meta.json").is_file()
    ]
    converted = dry_convert(data, [*phase5_test, *b_jobs, *g_jobs])
    added = record_outputs(data, outputs) if write else len(new)
    return {
        "files": len(counts),
        "phase5_hashes": phase5,
        "outputs_added": added,
        "glypharis_runs": runs,
        "inputs": inputs,
        "evaluators": evaluators,
        "converted": converted,
        "latency": {k: shown(p) for k, p in lat.items()},
    }


def read_pass(data: Path) -> dict[str, Any] | None:
    path = data / PASS_RECORD
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_pass(data: Path, record: dict[str, Any]) -> None:
    path = data / PASS_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def drop_handoff_outputs(data: Path) -> int:
    """Remove the hand-off entries a stopped pass recorded in phase14_outputs.json: the resumed
    pass writes the hand-off again and records it afresh. Nothing else in the manifest changes."""
    path = data / OUTPUTS
    if not path.is_file():
        return 0
    doc = _read_outputs(data)
    prefix = HANDOFF.as_posix() + "/"
    keep = [e for e in doc["outputs"] if not _norm_file(str(e.get("file"))).startswith(prefix)]
    dropped = len(doc["outputs"]) - len(keep)
    if dropped:
        doc["outputs"] = keep
        path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    return dropped


def start_pass(opts: Options) -> tuple[dict[str, Any], bool]:
    """The refusals about the pass itself: one-shot, and a stopped pass only resumed on purpose.

    Returns the pass's existing record (None for a new pass) and whether this run resumes it."""
    data = opts.data
    if eb._under(opts.out, eb.PHASE5_REPORTS):
        raise SystemExit(f"{opts.out.as_posix()} is under {eb.PHASE5_REPORTS.as_posix()}")
    if (opts.out / "results.json").exists():
        raise SystemExit(
            f"{(opts.out / 'results.json').as_posix()} exists: this is a one-shot "
            "pass; a re-score after a protocol change row writes elsewhere"
        )
    state = read_pass(data)
    if state is not None and state.get("status") == "done":
        raise SystemExit(
            f"data/{PASS_RECORD.as_posix()} records a finished pass: this is a one-shot pass; a "
            "re-score after a protocol change row writes elsewhere"
        )
    if state is not None and not opts.resume:
        raise SystemExit(
            f"data/{PASS_RECORD.as_posix()} records a scoring pass started "
            f"{state.get('started_utc')} that did not finish: run `score --resume` to finish it "
            "as the same evaluation (its rows are logged as resumed)"
        )
    if state is None and opts.resume:
        raise SystemExit(f"--resume: data/{PASS_RECORD.as_posix()} records no pass to resume")
    if state is None:
        for folder in (data / HANDOFF, data / OFFICIAL):
            if folder.exists() and any(folder.iterdir()):
                raise SystemExit(f"{data_rel(folder, data)} is not empty: move it aside first")
    return state or {}, state is not None


def run_score(opts: Options) -> int:
    data = opts.data
    state, resuming = start_pass(opts)
    pre = preflight(opts, write=True)
    evaluators = pre.pop("evaluators")
    reason = REASON + RESUMED if resuming else REASON
    if resuming:
        # The folders this pass owns; what the stopped run wrote there is written again.
        for folder in (data / HANDOFF, data / OFFICIAL):
            if folder.exists():
                shutil.rmtree(folder)
        drop_handoff_outputs(data)
    phase5, tag_commit = load_phase5_results()
    conditions, sources = gdc_conditions(), gdc_sources()

    record = {
        "experiment": EXPERIMENT,
        "status": "started",
        "started_utc": state.get("started_utc") or _now(),
        "resumed_utc": [*state.get("resumed_utc", []), *([_now()] if resuming else [])],
        "reason": reason,
    }
    write_pass(data, record)  # before the first logged row
    sc = Scorer(data, opts.access, reason)
    # 1. The Phase 5 reproduction check, before any other number.
    repro = sc.reproduce_phase5(phase5_jobs(data), phase5)
    repro["tag_commit"] = tag_commit
    print(f"Phase 5 reproduced: {repro['entries']} entries, {repro['profiles']} profiles")
    # 2. Every Phase 14 file, then the views built on the same per-unit values.
    sc.score_phase14([*baseline_jobs(data), *glypharis_jobs(data)])
    sc.phase5_units()
    subset = sc.subset_entries()
    comparisons = sc.comparisons()
    cond_tables = sc.gdc_condition_tables(conditions)
    in_charset = sc.gdc_scan_in_charset()
    in_charset_pages = sc.gdc_scan_in_charset_pages()
    # 3. The official evaluators, on files scored and logged above.
    work = data / OFFICIAL
    official = {
        "hiertext": sc.official_hiertext(work),
        "omnidocbench": sc.official_omnidocbench(work),
        "evaluators": evaluators,
    }
    # 4. The five confirmatory tests.
    h1 = sc.h1_cer(conditions, sources)
    lat = h1_latency(latency_paths(data, opts.eff_dir), pre["glypharis_runs"])
    h5 = sc.h5(sources)
    cal = sc.calibration()
    hypotheses = {
        "H1′ CER": h1,
        "H1′ latency": lat,
        "H1′ holds": h1["verdict"] == "met" and lat["verdict"] == "met",
        "H5′": h5,
        "H8′": cal.pop("h8"),
    }
    # 5. The Phase 15 hand-off, hashed into the outputs manifest.
    written = sc.handoff(data / HANDOFF, conditions, sources)
    record_outputs(data, [file_entry(p, data) for p in written])
    # 6. Aggregates only.
    git = git_info()
    # Phase 5's validation entries stay in the Phase 5 report; they were only reproduced here.
    entries = [
        s.entry
        for s in sc.scored.values()
        if s.job.phase == PHASE or EVAL_SETS[s.job.eval_set]["split"] == "test"
    ]
    res = {
        "experiment": EXPERIMENT,
        "provenance": {
            "scorer": {"commit": git.commit, "branch": git.branch, "dirty": git.dirty},
            "pinned": pinned_state(),
            "date": datetime.now(UTC).date().isoformat(),
            "access_log_reason": reason,
            "pass": {k: record[k] for k in ("started_utc", "resumed_utc")},
            **pre,
        },
        "reproduction": repro,
        "predictions": entries,
        "subset": subset,
        "gdc_conditions": cond_tables,
        "gdc_scan_in_charset": in_charset,
        "gdc_scan_in_charset_pages": in_charset_pages,
        "comparisons": comparisons,
        "official": official,
        "calibration": cal,
        "hypotheses": hypotheses,
        "handoff": {
            "files": [data_rel(p, data) for p in written],
            "seed": HANDOFF_SEED,
            "correct_lines": HANDOFF_CORRECT,
        },
    }
    opts.out.mkdir(parents=True, exist_ok=True)
    for name, text in render(res).items():
        (opts.out / name).write_text(text, encoding="utf-8", newline="\n")
    diagrams(opts.out, cal)
    (opts.out / "results.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    write_pass(data, {**record, "status": "done", "ended_utc": _now()})
    # ASCII only: a cp1252 console or a redirected stdout cannot encode the prime.
    print(
        f"wrote {opts.out.as_posix()}: H1' CER {h1['verdict']}, latency {lat['verdict']}, "
        f"H5' {h5['verdict']}, H8' {hypotheses['H8′']['transfers']['verdict']} / "
        f"{hypotheses['H8′']['by_half']['verdict']}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("hash-phase5", "check", "score"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eff-dir", type=Path, default=DEFAULT_EFF)
    parser.add_argument("--record", type=Path, default=FREEZE_RECORD)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="score: finish a pass that started and stopped (its rows are logged as resumed)",
    )
    args = parser.parse_args(argv)
    if args.command == "hash-phase5":
        return hash_phase5(args.data_root)
    opts = Options(
        args.data_root,
        args.out,
        args.eff_dir,
        ACCESS_LOG if ACCESS_LOG is not None else log_path(),
        args.record,
        args.resume,
    )
    if args.command == "check":
        pre = preflight(opts, write=False)
        print(
            f"ready: {pre['files']} prediction files complete, {pre['outputs_added']} not yet "
            f"in {OUTPUTS.as_posix()}, Phase 5 files match their hashes"
        )
        return 0
    return run_score(opts)


if __name__ == "__main__":
    sys.exit(main())
