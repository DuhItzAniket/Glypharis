"""Run the phone candidate on an evaluation set and write its predictions (Phase 22, item 17).

    python scripts/phase22_eval.py run --mode rec --set hiertext_val --experiment GW-REC-044 \
        --reason "Phase 22 candidate validation gate (GW-REC-044)"
    python scripts/phase22_eval.py run --mode e2e --set gdc_test --experiment GW-E2E-020 \
        --reason "Phase 22 candidate test pages (GW-E2E-020)"
    python scripts/phase22_eval.py run --mode rec --set gdc_test --experiment GW-REC-045 \
        --reason "Phase 22 candidate test crops (GW-REC-045)" --resume

The model is the frozen phone bundle (`BUNDLE`, ``glypharis-phone-bundle-1``), loaded with
`load_phone_bundle` alone on ONNX Runtime, CPU only, `THREADS` threads (set before numpy and torch
are imported). It is a new version, not v1: its metadata and log rows name it as the candidate.

Which experiment runs which set is fixed by `PLAN` (docs/PHASE_STATUS.md, Phase 22 items 18 and 19):
GW-REC-044 and GW-E2E-019 on HierText validation, for the gate; GW-REC-045 on the crops and
GW-E2E-020 on the pages of the five test sets each. There is no oracle mode. Every run:

1. starts only on branch `PHASE22_BRANCH`, and runs `pinned_code_check` at `PIN_TAG` and
   `assert_v1_code`. A test set is refused when either fails; a validation run is refused too,
   unless it is a smoke run (below) with ``--allow-unpinned``;
2. checks the bundle: the SHA-256 of its ``manifest.json`` must equal the committed copy's
   (`BUNDLE_REPORT`), `EQUIVALENCE` must say ``"pass": true`` for that hash, every file must match
   the manifest (`read_phone_manifest`), and the version label must say it is not v1;
3. for a test set, before the access log or any record is read: `OPERATOR_CHECK` must say ``go``,
   and `GATE` must hold the three gate tests (`GATE_TESTS`), all ``met``, with this run's bundle
   hash and tag commit. A validation set needs neither and logs no access row; it is not test data;
4. for a test set, reads the access log: once it holds a prediction row for the set's benchmark,
   split and ID, the set runs again only as a resume of a crash, with ``--resume``: the output must
   still exist, be incomplete, its last segment still ``running`` and its identity the same. A
   moved, deleted or emptied output, or one whose pass ended, cannot start a second pass; that
   needs a new registered ID. An output with no logged row is refused too;
5. takes the split from the records, hashes the inputs against data/manifests (`check_inputs`),
   finds each crop's line on its page (`line_boxes`) and checks every page image's header size
   (`image_sizes_match`), all before the access;
6. loads the bundle, checks its sessions and thread counts, then logs the access with
   `open_test_access` (phase 22) before any image is decoded.

The records are Phase 14's (`ext_eval._rec_records` and ``_page_records`` through `Candidate`, an
adapter exposing ``.ocr`` and ``.recognizer``), so `phase14_score` and `evaluate_baselines` score
them unchanged. Crops are read `CROPS_PER_CALL` per recognizer call; the bundle's recognizer runs
its network on one line at a time (its batching rule). Outputs:
data/processed/phase22/<ID>/<set>.jsonl and <set>.meta.json, Phase 14's metadata plus the bundle's
manifest hash and files, the tag commit, the v1-code check, the thread counts and, for a test set,
the operator check and gate it was cleared by. Records are appended as they are made, so a crash
resumes where it stopped, and its log row says "(resumed: N of M ...)". The runner computes no
metric and writes no outputs manifest; prediction_hashes.json is written by phase22_score.py.

GDC-scan is private: for it, no key, text or file name is printed, only counts.

Smoke runs on validation data: ``--limit N`` writes the first N crops or pages (crops are still
recognised a whole call at a time, so up to `CROPS_PER_CALL` are read) and needs ``--out-root``, a
folder outside the data root's ``processed``; ``--allow-unpinned`` runs without the tag. None of
these is accepted for a test set, and the metadata of a smoke run say so (``smoke``).
"""

# ruff: noqa: E402  (the thread variables are set before numpy and torch are imported)

from __future__ import annotations

import sys
from typing import Any

from glypharis.efficiency.threads import set_threads

THREADS = 10  # the registered thread count (Phase 22 item 19), CPU only

_THREADS: dict[str, Any] | None = None
if __name__ == "__main__":
    _THREADS = set_threads(THREADS)

import argparse
import itertools
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import ext_eval
import numpy as np
import torch
from run_baselines import done_keys

from glypharis.data.download import sha256_file
from glypharis.efficiency.candidate import CROPS_PER_CALL, batching_of
from glypharis.efficiency.onnx_build import check_sessions
from glypharis.efficiency.phone_bundle import (
    MANIFEST,
    PhoneBundleError,
    load_phone_bundle,
    read_phone_manifest,
)
from glypharis.efficiency.threads import check_threads
from glypharis.efficiency.v1 import V1CodeError, assert_v1_code
from glypharis.eval.access import is_test_split, open_test_access
from glypharis.eval.inputs import InputError, check_inputs, image_sizes_match, line_boxes
from glypharis.eval.sets import EVAL_SETS, access_log, repo_root, require_branch
from glypharis.pipeline.frozen import pinned_code_check
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer
from glypharis.utils.gitinfo import git_info

PHASE = 22
PHASE22_BRANCH = "phase/22-final-validation"
PIN_TAG = "phase22-candidate-eval"
MODES = ("rec", "e2e")
# Registered with the Phase 22 plan (items 18 and 19): experiment -> its sets.
PLAN: dict[str, dict[str, tuple[str, ...]]] = {
    "rec": {
        "GW-REC-044": ("hiertext_val",),
        "GW-REC-045": ("hiertext_test", "gdc_test", "gdc_scan_test", "funsd_test", "iiit5k_test"),
    },
    "e2e": {
        "GW-E2E-019": ("hiertext_val",),
        "GW-E2E-020": (
            "hiertext_test",
            "gdc_test",
            "gdc_scan_test",
            "funsd_test",
            "omnidocbench_test",
        ),
    },
}
OUTPUT_ROOT = "phase22"  # under <data>/processed
SYSTEM = "glypharis_phone_candidate"  # the access log's name for this system
V1_NAME = "glypharis_v1"
# Repository-relative; the runner resolves them under the repository root.
BUNDLE = Path("runs/release/glypharis-phone-candidate")
BUNDLE_REPORT = Path("reports/final_validation/candidate/GW-EFF-014/manifest.json")
EQUIVALENCE = Path("reports/final_validation/candidate/GW-EFF-014/equivalence.json")
OPERATOR_CHECK = Path("reports/final_validation/GW-EFF-015/operator_check.json")
GATE = Path("reports/final_validation/candidate/validation_gate.json")
# The gate's three tests (H11's formulas on HierText validation), named as
# `glypharis.efficiency.candidate.accuracy_tests` names them.
GATE_TESTS = ("line_cer", "page_cer", "e2e_f1")
ACCESS_LOG = Path("research/test_access_log.md")
REQUIRE_RECORDED = True  # every input needs a recorded hash; the tests' tiny sets have none
PRIVATE_BENCHMARKS = ("gdc_scan",)  # nothing but counts is printed for these
T1 = ext_eval.T1


# ------------------------------------------------------------------------------- the model


@dataclass(frozen=True)
class Candidate:
    """What `ext_eval`'s record makers read from a model: the pipeline and its recognizer."""

    ocr: GlypharisOCR

    @property
    def recognizer(self) -> TextRecognizer:
        return self.ocr.recognizer


def _threads() -> dict[str, Any]:
    return _THREADS if _THREADS is not None else set_threads(THREADS, require_fresh=False)


def environment(ocr: GlypharisOCR, sessions: dict[str, Any]) -> dict[str, Any]:
    """The run environment, under the keys the Phase 14 scorer compares (``RUN_ENV``: device,
    device name, torch, cuda, cudnn, numerics) and a few more."""
    import onnxruntime

    env = {
        "device": "cpu",
        "device_name": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda": None,
        "cudnn": None,
        "numerics": {
            "backend": "onnxruntime (CPU)",
            "onnxruntime": onnxruntime.__version__,
            "threads": THREADS,
            "session_options": sessions,
        },
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    return dict(json.loads(json.dumps(env)))


# ------------------------------------------------------------------------------------ paths


def log_path() -> Path:
    """The test-access log: `ACCESS_LOG` when absolute (tests point it at a temporary file),
    otherwise the repository's own log, found from the repository root."""
    return ACCESS_LOG if ACCESS_LOG.is_absolute() else access_log()


def out_root(args: argparse.Namespace, data: Path) -> Path:
    return args.out_root if args.out_root is not None else data / "processed" / OUTPUT_ROOT


def output_path(root: Path, experiment: str, eval_set: str) -> Path:
    return root / experiment / f"{eval_set}.jsonl"


def shown(path: Path, data: Path) -> str:
    """A file as messages name it: ``data/...`` under the data root, else its last three parts."""
    try:
        return "data/" + path.resolve().relative_to(data.resolve()).as_posix()
    except ValueError:
        return "/".join(path.parts[-3:])


def checkpoint(mode: str) -> str:
    """The access log's checkpoint cell for a prediction row (scoring rows add ", scoring")."""
    return f"{SYSTEM} ({mode})"


def private(eval_set: str) -> bool:
    return str(EVAL_SETS[eval_set]["benchmark"]) in PRIVATE_BENCHMARKS


def _read_json(path: Path, what: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"{what} ({path.name}) is missing")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{what} ({path.name}) is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise SystemExit(f"{what} ({path.name}) is not a JSON object")
    return doc


def _file(path: Path, rel: Path) -> dict[str, Any]:
    return {"file": rel.as_posix(), "sha256": sha256_file(path)}


# ------------------------------------------------------------------------------------ checks


def check_request(args: argparse.Namespace, data: Path) -> None:
    """Refusals that need no data: the registered plan and the smoke options."""
    sets_for = PLAN[args.mode].get(args.experiment)
    if sets_for is None:
        raise SystemExit(
            f"{args.experiment} is not a {args.mode} experiment of this phase; registered: "
            f"{', '.join(sorted(PLAN[args.mode]))}"
        )
    if args.set not in sets_for:
        raise SystemExit(f"{args.experiment} runs on {', '.join(sets_for)}, not {args.set}")
    if not args.reason.strip():
        raise SystemExit("a reason is required (it goes into the test-access log)")
    smoke = args.limit is not None or args.allow_unpinned or args.out_root is not None
    if is_test_split(EVAL_SETS[args.set]["split"]):
        if smoke:
            raise SystemExit("--limit, --out-root and --allow-unpinned are for validation smoke "
                             "runs; a test set runs whole and pinned, in its folder")  # fmt: skip
        return
    if args.resume:
        raise SystemExit("--resume is for test sets; a validation run resumes by itself")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if (args.limit is not None or args.allow_unpinned) and args.out_root is None:
        raise SystemExit("a smoke run (--limit, --allow-unpinned) needs --out-root, outside "
                         f"data/processed/{OUTPUT_ROOT}")  # fmt: skip
    if args.out_root is not None:
        processed = (data / "processed").resolve()
        where = args.out_root.resolve()
        if where == processed or processed in where.parents:
            raise SystemExit("--out-root must lie outside the data root's processed folder")


def pinned_state(args: argparse.Namespace, test: bool, root: Path) -> dict[str, Any]:
    """`pinned_code_check` at `PIN_TAG`; a failure stops the run unless a validation smoke run
    passed ``--allow-unpinned``, and is then recorded instead."""
    try:
        return pinned_code_check(PIN_TAG, root=root)
    except RuntimeError as exc:
        if test or not args.allow_unpinned:
            raise SystemExit(f"pinned code check failed: {exc}") from exc
        return {"tag": PIN_TAG, "tag_commit": None, "trees": "not pinned", "error": str(exc),
                "allow_unpinned": True}  # fmt: skip


def v1_state(root: Path) -> dict[str, Any]:
    try:
        return assert_v1_code(root)
    except V1CodeError as exc:
        raise SystemExit(f"v1 code check failed: {exc}") from exc


def bundle_state(root: Path) -> dict[str, Any]:
    """The bundle's identity, after checking it is the frozen, equivalence-checked one."""
    path = root / BUNDLE / MANIFEST
    if not path.is_file():
        raise SystemExit(f"{BUNDLE.as_posix()}/{MANIFEST} is missing: item 16 freezes the bundle")
    sha = sha256_file(path)
    committed = root / BUNDLE_REPORT
    if not committed.is_file() or sha256_file(committed) != sha:
        raise SystemExit(f"the bundle's manifest differs from {BUNDLE_REPORT.as_posix()}")
    equivalence = _read_json(root / EQUIVALENCE, "the bundle's equivalence check (GW-EFF-014)")
    if equivalence.get("pass") is not True:
        raise SystemExit("the bundle's equivalence check did not pass")
    if equivalence.get("bundle_manifest_sha256") != sha:
        raise SystemExit("the equivalence check was made on another bundle manifest")
    try:
        manifest = read_phone_manifest(root / BUNDLE)
    except PhoneBundleError as exc:
        raise SystemExit(f"bundle refused: {exc}") from exc
    label = str(manifest.get("version_label", ""))
    if manifest.get("version") in (None, V1_NAME) or "not v1" not in label:
        raise SystemExit("the bundle's version label does not say it is a new version, not v1")
    return {
        "path": BUNDLE.as_posix(),
        "manifest_sha256": sha,
        "committed_manifest": _file(committed, BUNDLE_REPORT),
        "equivalence": _file(root / EQUIVALENCE, EQUIVALENCE),
        "format": manifest["format"],
        "version": manifest["version"],
        "version_label": label,
        "arm": manifest["arm"],
        "batching": manifest["batching"],
        "files": manifest["files"],
        "manifest": manifest,
    }


def gate_verdicts(gate: dict[str, Any]) -> dict[str, str]:
    """The gate's test name -> verdict, from a ``tests`` mapping or list."""
    tests = gate.get("tests")
    if isinstance(tests, dict):
        items = [(str(k), v) for k, v in tests.items()]
    elif isinstance(tests, list):
        items = [(str(t.get("name") if isinstance(t, dict) else None), t) for t in tests]
    else:
        return {}
    return {k: str(v.get("verdict")) if isinstance(v, dict) else str(v) for k, v in items}


def check_clearance(root: Path, bundle_sha: str, tag_commit: str | None) -> dict[str, Any]:
    """Before a test access: the operator check is ``go`` and the validation gate's three tests
    are ``met`` for this bundle at this tag commit."""
    op = _read_json(root / OPERATOR_CHECK, "the Android operator check (GW-EFF-015)")
    if op.get("verdict") != "go":
        raise SystemExit(f"the operator check says {op.get('verdict')!r}, not 'go'; no test set "
                         "is opened")  # fmt: skip
    gate = _read_json(root / GATE, "the validation gate (GW-REC-044, GW-E2E-019)")
    verdicts = gate_verdicts(gate)
    if set(verdicts) != set(GATE_TESTS):
        raise SystemExit(f"the validation gate holds tests {sorted(verdicts)}, not "
                         f"{list(GATE_TESTS)}; no test set is opened")  # fmt: skip
    unmet = sorted(k for k, v in verdicts.items() if v != "met")
    if unmet or gate.get("met") is False:
        raise SystemExit(f"the validation gate is not met ({', '.join(unmet) or 'met: false'}); "
                         "no test set is opened")  # fmt: skip
    if gate.get("bundle_manifest_sha256") != bundle_sha:
        raise SystemExit("the validation gate was scored on another bundle; no test set is opened")
    if tag_commit is None or gate.get("tag_commit") != tag_commit:
        raise SystemExit("the validation gate was scored at another tag commit; no test set is "
                         "opened")  # fmt: skip
    return {
        "operator_check": {**_file(root / OPERATOR_CHECK, OPERATOR_CHECK), "verdict": "go"},
        "validation_gate": {**_file(root / GATE, GATE), "tests_met": sorted(verdicts),
                            "bundle_manifest_sha256": bundle_sha, "tag_commit": tag_commit},
    }  # fmt: skip


def logged_predictions(log: Path, benchmark: str, split: str, exp: str, ckpt: str) -> int:
    """Prediction rows of this set and ID in the access log. Scoring rows name another
    checkpoint."""
    if not log.is_file():
        raise SystemExit("the test-access log does not exist")
    want = [f"{benchmark} {split}", exp, ckpt]
    rows = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows += len(cells) == 6 and cells[1:4] == want
    return rows


def _meta_runs(meta_path: Path) -> list[dict[str, Any]]:
    return list(json.loads(meta_path.read_text(encoding="utf-8")).get("runs") or [])


def _last_status(meta_path: Path) -> str | None:
    runs = _meta_runs(meta_path)
    return runs[-1].get("status") if runs else None


def recorded_progress(meta_path: Path) -> int:
    """How many records the metadata says were written: the pass records its progress as it
    goes, so a crashed output that was cut short or emptied afterwards shows it."""
    doc = json.loads(meta_path.read_text(encoding="utf-8"))
    runs = doc.get("runs") or []
    last = runs[-1] if runs else {}
    return max(int(doc.get("samples") or 0),
               int(last.get("already_done") or 0) + int(last.get("written") or 0))  # fmt: skip


def close_complete(meta_path: Path, samples: int) -> None:
    """Close the metadata of an output whose every record is written but whose last segment
    never reached its end (killed after its last record): ``done``, marked as closed by a later
    start. Phase 14's `ext_eval.close_meta` words that status differently."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    runs = meta.get("runs") or []
    if runs and runs[-1].get("status") in ("running", "pending"):
        runs[-1].update(status="done", closed_by_later_start=True,
                        written=samples - int(runs[-1].get("already_done") or 0),
                        ended=datetime.now(UTC).isoformat(timespec="seconds"))  # fmt: skip
        meta["samples"] = samples
        ext_eval._write_json(meta_path, meta)


def attempt_state(args: argparse.Namespace, out: Path, done: set[str], total: int) -> str:
    """``fresh``, ``resume`` or ``complete`` for a test set; a second pass stops here.

    The access log decides, not the output folder: once a prediction row exists, only a crash
    (an incomplete output whose last segment is still ``running``, or ``pending`` when the crash
    came right after the row) may continue, with ``--resume``, and only if the output holds at
    least the records its metadata says were written. An empty output and metadata left
    ``pending`` by an attempt that stopped before its access row count as never started."""
    spec = EVAL_SETS[args.set]
    name = f"{args.experiment} {args.mode} {args.set}"
    meta_path = out.with_suffix(".meta.json")
    rows = logged_predictions(log_path(), spec["benchmark"], spec["split"], args.experiment,
                              checkpoint(args.mode))  # fmt: skip
    if not rows:
        stale = out.exists() or meta_path.exists()
        never_logged = (
            out.is_file() and meta_path.is_file() and not done
            and _last_status(meta_path) == "pending" and len(_meta_runs(meta_path)) == 1
        )  # fmt: skip
        if stale and not never_logged:
            raise SystemExit(f"{name}: an output exists with no logged prediction pass; refusing")
        if args.resume:
            raise SystemExit(f"{name}: --resume, but no prediction pass was started")
        return "fresh"
    if not out.exists():
        raise SystemExit(f"{name}: a prediction pass is logged and its output is gone. Moving or "
                         "deleting it does not start a second pass; that needs a new registered "
                         "ID")  # fmt: skip
    if not meta_path.is_file():
        raise SystemExit(f"{name}: a prediction pass is logged but its run metadata is missing")
    if len(done) < recorded_progress(meta_path):
        raise SystemExit(f"{name}: the output holds fewer records than its metadata says were "
                         "written; it was cut short or emptied, and that does not start a second "
                         "pass")  # fmt: skip
    if len(done) >= total:
        return "complete"
    if _last_status(meta_path) not in ("running", "pending"):
        raise SystemExit(f"{name}: this pass ended, so this is not a crash resume. A further "
                         "pass needs a new registered ID")  # fmt: skip
    if not args.resume:
        raise SystemExit(f"{name}: a started pass continues only as a resume, with --resume")
    return "resume"


def check_pages(args: argparse.Namespace, data: Path, pages: list[Any]) -> None:
    bad = image_sizes_match(data, pages)
    if bad:
        detail = "" if private(args.set) else f": {'; '.join(bad[:5])}"
        raise SystemExit(f"{len(bad)} page images of {args.set} differ in size from their page "
                         f"records, refusing before the access{detail}")  # fmt: skip


def checked_inputs(
    args: argparse.Namespace, data: Path, n_pages: int | None
) -> tuple[dict[str, Any], dict[str, list[list[float]]]]:
    """`check_inputs` (``n_pages``: the whole page file, also on a smoke run) and, for crops,
    `line_boxes`; for a private set a refusal names no key."""
    try:
        inputs = check_inputs(data, args.set, args.mode, n_pages, require=REQUIRE_RECORDED)
        boxes = line_boxes(data, args.set) if args.mode == "rec" else {}
    except InputError as exc:
        if private(args.set):
            raise SystemExit(f"{args.set}: an input differs from its record (details are not "
                             "shown for a private set)") from exc  # fmt: skip
        raise
    return inputs, boxes


def identity(
    bundle: dict[str, Any],
    env: dict[str, Any],
    settings: dict[str, Any],
    batch: int | None,
    pinned: dict[str, Any],
    inputs: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    """What every segment of one evaluation must share: the bundle, the run environment, the
    settings, the recognition batch (None for pages), the tag commit, the inputs and, for a smoke
    run, its limit. HEAD may move between segments (commits of the access log)."""
    doc = {"bundle": {"manifest_sha256": bundle["manifest_sha256"], "files": bundle["files"]},
           "environment": env, "settings": settings, "batch": batch,
           "tag_commit": pinned.get("tag_commit"), "inputs": inputs, "limit": limit}  # fmt: skip
    return dict(json.loads(json.dumps(doc)))


def settings_of(manifest: dict[str, Any]) -> dict[str, Any]:
    return {k: manifest[k] for k in ("detector", "recognizer", "pipeline", "batching")}


# ------------------------------------------------------------------------------------- runs


def run(args: argparse.Namespace, data: Path) -> int:
    spec = EVAL_SETS[args.set]
    split = spec["split"]
    test = is_test_split(split)
    root = repo_root()
    try:
        require_branch(PHASE22_BRANCH)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    pinned = pinned_state(args, test, root)  # before anything is read or loaded
    v1_code = v1_state(root)
    bundle = bundle_state(root)
    clearance = None
    if test:  # before the access log or any record is read
        clearance = check_clearance(root, bundle["manifest_sha256"], pinned.get("tag_commit"))

    keys, pages, splits = ext_eval.set_records(data, args.set, args.mode)
    if splits != {split}:
        raise SystemExit(f"{args.set}: records name splits {sorted(splits)}, the set is "
                         f"registered as {split}")  # fmt: skip
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{args.set}: a key appears twice in its records")
    n_pages = None if pages is None else len(pages)
    if args.limit is not None:
        keys = keys[: args.limit]
        pages = None if pages is None else pages[: args.limit]
    out = output_path(out_root(args, data), args.experiment, args.set)
    done = done_keys(out)
    stray = done - set(keys)
    if stray:
        raise SystemExit(f"{shown(out, data)} holds {len(stray)} keys outside {args.set}; "
                         "move it aside")  # fmt: skip
    meta_path = out.with_suffix(".meta.json")
    state = attempt_state(args, out, done, len(keys)) if test else "validation"
    if not test and args.out_root is None and (root / GATE).exists() and len(done) < len(keys):
        # Before anything is touched: a refused run leaves the gated files as they are.
        raise SystemExit(f"{GATE.as_posix()} exists: the gate has been scored, and its "
                         "validation runs are not run again")  # fmt: skip
    if state == "fresh":
        for stale in (out, meta_path):  # an attempt that never reached its access row
            stale.unlink(missing_ok=True)
        done = set()
    if not test and meta_path.exists() and not out.exists():
        meta_path.unlink()  # metadata of validation predictions moved aside: start afresh
    if len(done) >= len(keys):
        # Nothing to do: no model load and no row in the test-access log.
        if meta_path.exists():
            close_complete(meta_path, len(done))
        print(f"{args.experiment} {args.mode} {args.set}: already complete ({len(done)})")
        return 0

    inputs, boxes = checked_inputs(args, data, n_pages)
    if args.mode == "e2e":
        check_pages(args, data, pages or [])
    threads = _threads()
    try:
        ocr = load_phone_bundle(root / BUNDLE, THREADS)
    except PhoneBundleError as exc:
        raise SystemExit(f"bundle refused: {exc}") from exc
    manifest = bundle.pop("manifest")
    if batching_of(ocr.recognizer) != manifest["batching"]:
        raise SystemExit("the loaded recognizer does not read under the bundle's batching rule")
    sessions = check_sessions(ocr, THREADS)
    threads_before = check_threads(THREADS)
    env = environment(ocr, sessions)
    settings = settings_of(manifest)
    batch = CROPS_PER_CALL if args.mode == "rec" else None
    ident = identity(bundle, env, settings, batch, pinned, inputs, args.limit)
    meta: dict[str, Any] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    if (done or state == "resume") and meta.get("identity") != ident:
        raise SystemExit(f"{shown(out, data)} was started with another bundle, environment, "
                         "settings, inputs or tag; a resume must be the same run")  # fmt: skip

    reason = args.reason.strip()
    if done or state == "resume":
        reason += f" (resumed: {len(done)} of {len(keys)} samples already predicted)"

    out.parent.mkdir(parents=True, exist_ok=True)
    git = git_info(root)
    provenance = {"bundle": bundle["path"], "bundle_manifest_sha256": bundle["manifest_sha256"],
                  "files": bundle["files"], "arm": bundle["arm"], "batching": bundle["batching"],
                  "version": bundle["version"], "version_label": bundle["version_label"], **env,
                  "git_commit": git.commit, "git_dirty": git.dirty}  # fmt: skip
    runs = list(meta.get("runs", []))
    seg = len(runs)
    run_info: dict[str, Any] = {
        "segment": seg, "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "pending", "already_done": len(done), "written": 0, "git_commit": git.commit,
        "git_dirty": git.dirty, "pinned": pinned, "device": "cpu", "batch": batch,
        "threads": threads, "threads_checked": {"before": threads_before},
        "provenance": provenance, "reason": reason,
    }  # fmt: skip
    runs.append(run_info)
    temperature = ocr.recognizer.temperature
    meta = {
        "system": SYSTEM, "task": args.mode, "mode": args.mode, "eval_set": args.set,
        "benchmark": spec["benchmark"], "split": split, "experiment": args.experiment,
        "phase": PHASE, "version": ocr.model_version, "version_label": bundle["version_label"],
        "expected": len(keys), "samples": len(done), "temperature": temperature,
        "confidences_at": [T1, temperature], "settings": settings, "provenance": provenance,
        "pinned": pinned, "tag_commit": pinned.get("tag_commit"),
        "bundle_manifest_sha256": bundle["manifest_sha256"], "bundle": bundle,
        "v1_code": v1_code, "clearance": clearance, "inputs": inputs, "identity": ident,
        "smoke": None if args.out_root is None else {
            "limit": args.limit, "allow_unpinned": bool(args.allow_unpinned),
            "note": "a smoke run outside the registered folders; never scored"},
        "runs": runs,
        "note": "predictions only, of the phone candidate (a new version, not v1): this runner "
                "computes no metric (GW-E2E-021 scores once); time_ms is diagnostic",
    }  # fmt: skip
    # The metadata and the output exist, the segment `pending`, before the access row is
    # written: a crash right after the row then leaves a pass that resumes, not one that looks
    # moved or deleted. An attempt that stops before the row leaves them `pending` and empty.
    ext_eval._write_json(meta_path, meta)
    out.touch()
    open_test_access(spec["benchmark"], split, args.experiment, checkpoint(args.mode), PHASE,
                     reason, log_path())  # fmt: skip
    run_info["status"] = "running"
    ext_eval._write_json(meta_path, meta)

    model = Candidate(ocr)
    records = (
        ext_eval._rec_records(model, data, args.set, done, CROPS_PER_CALL, boxes)
        if args.mode == "rec"
        else ext_eval._page_records(model, data, pages or [], done, "e2e")
    )
    start = time.monotonic()
    written = 0
    every = CROPS_PER_CALL if args.mode == "rec" else 1  # progress kept in the metadata
    try:
        with out.open("a", encoding="utf-8", newline="\n") as fh, torch.inference_mode():
            for record in itertools.islice(records, len(keys) - len(done)):
                fh.write(json.dumps({**record, "seg": seg}, ensure_ascii=False) + "\n")
                fh.flush()
                written += 1
                if written % every == 0:
                    run_info["written"] = written
                    meta["samples"] = len(done) + written
                    ext_eval._write_json(meta_path, meta)
                if written % (1000 if args.mode == "rec" else 100) == 0:
                    print(f"{len(done) + written}/{len(keys)}", flush=True)
    except Exception as exc:
        if not private(args.set):
            raise
        # A private set's error can name a file; only the count is shown.
        raise SystemExit(f"{args.set}: the pass stopped after {len(done) + written} of "
                         f"{len(keys)} records ({type(exc).__name__}; details are not shown for "
                         "a private set)") from None  # fmt: skip

    total = len(done) + written
    run_info["threads_checked"]["after"] = check_threads(THREADS)
    run_info.update(status="done", written=written, wall_seconds=round(time.monotonic() - start, 1),
                    ended=datetime.now(UTC).isoformat(timespec="seconds"))  # fmt: skip
    meta["samples"] = total
    ext_eval._write_json(meta_path, meta)
    if total != len(keys) or ext_eval.count_lines(out) != len(keys):
        raise SystemExit(f"{shown(out, data)}: {ext_eval.count_lines(out)} records for "
                         f"{len(keys)} keys")  # fmt: skip
    print(f"{args.experiment} {args.mode} {args.set}: {written} new, {total} total")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="the candidate on one set, predictions only")
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--set", required=True, choices=sorted(EVAL_SETS))
    p.add_argument("--experiment", required=True)
    p.add_argument("--reason", required=True, help="goes into the test-access log")
    p.add_argument("--resume", action="store_true", help="continue a test pass after a crash")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--limit", type=int, help="validation smoke run: the first N crops or pages")
    p.add_argument("--out-root", type=Path, help="validation smoke run: where outputs go")
    p.add_argument("--allow-unpinned", action="store_true",
                   help="validation smoke run: without the tag")  # fmt: skip
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    check_request(args, args.data_root)
    return run(args, args.data_root)


if __name__ == "__main__":
    sys.exit(main())
