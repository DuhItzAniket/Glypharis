"""Reproduce v1 through the unchanged Phase 14 code (Phase 22, item 6).

Re-run, from a detached ``phase14-eval`` worktree, with this script from the Phase 22 worktree:

    cd $TAG_WT
    PYTHONPATH='.;scripts' $PY -P $P22/scripts/phase22_v1_repro.py rerun rec --set hiertext_val \
        --folder-id GW-REC-038 --experiment GW-REC-042 \
        --reason "Phase 22 v1 re-run rehearsal (GW-REC-042)" --data-root $ROOTS/rerun

Re-score, from the Phase 22 worktree:

    PYTHONPATH='.;scripts' $PY -P scripts/phase22_v1_repro.py rescore check \
        --data-root $ROOTS/rescore
    PYTHONPATH='.;scripts' $PY -P scripts/phase22_v1_repro.py rescore score \
        --experiment GW-E2E-015 --out reports/final_validation/GW-E2E-015 --data-root $ROOTS/rescore
    PYTHONPATH='.;scripts' $PY -P scripts/phase22_v1_repro.py rescore score --id-map rerun \
        --experiment GW-E2E-018 --out reports/final_validation/GW-E2E-018 --data-root $ROOTS/rerun

Nothing protected is edited: the wrapper changes module attributes of `ext_eval`, `phase14_score`
and `evaluate_baselines` in its own process and then calls their ``main``.

``rerun`` keeps the folder IDs the scorer hard-codes and writes under ``--data-root`` only. It
refuses imports from outside the working tree, a tree or wrapper that is not pinned, a data root
inside a worktree or with a linked ``processed/phase14`` or outputs manifest, a test set started
out of Phase 14's order, and a second attempt at a test set. An attempt is a prediction row in the
access log; only a run whose metadata still says ``running`` may continue, as a resume. A run whose
records are all written but which never reached the outputs manifest is closed by `ext_eval`
without reading an image or logging a row. Log rows and run metadata say phase 22 and carry the new
ID; the metadata also records the wrapper commit. The pin checks run before any test record is read.

``rescore`` runs `phase14_score` on an isolated root, with the phase 22 ID and reason on every row.
``ps.PHASE`` stays 14 because it routes the baseline paths; only the scorer's contexts change. Each
pass takes only its own files: Pass A refuses metadata stamped by ``rerun``, Pass B needs that stamp
on every file, and a stopped pass resumes only as itself. With ``--id-map rerun`` (Pass B), rows are
logged under the re-run IDs, and the reports, which the scorer labels with the folder IDs, get an
``id_map`` block and a header line. ``--out`` must be ``reports/final_validation/<the pass's ID>``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib
import inspect
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

WRAPPER_ROOT = Path(__file__).resolve().parents[1]
ACCESS_LOG = WRAPPER_ROOT / "research" / "test_access_log.md"
WRAPPER_TAG = "phase22-repro"
PHASE22_BRANCH = "phase/22-final-validation"
PHASE = 22
SOURCE_PHASE = 14
# Folder the unchanged code writes and scores -> the ID this phase registered for the run.
RERUN_IDS = {
    "GW-REC-038": "GW-REC-042",
    "GW-E2E-006": "GW-E2E-014",
    "GW-REC-039": "GW-REC-043",
    "GW-E2E-007": "GW-E2E-016",
    "GW-E2E-008": "GW-E2E-017",
}
# The test folders, whose scoring rows Pass B logs under the re-run IDs. Validation logs no row.
PASS_B_IDS = {k: RERUN_IDS[k] for k in ("GW-REC-039", "GW-E2E-007", "GW-E2E-008")}
TEST_FOLDERS = {"rec": "GW-REC-039", "oracle": "GW-E2E-008", "e2e": "GW-E2E-007"}
# The order Phase 14 ran its test sets, from the start times in its run metadata.
TEST_ORDER = (
    ("rec", "hiertext_test"),
    ("rec", "gdc_test"),
    ("rec", "gdc_scan_test"),
    ("rec", "funsd_test"),
    ("rec", "iiit5k_test"),
    ("oracle", "gdc_test"),
    ("oracle", "gdc_scan_test"),
    ("e2e", "gdc_scan_test"),
    ("e2e", "funsd_test"),
    ("e2e", "gdc_test"),
    ("e2e", "omnidocbench_test"),
    ("e2e", "hiertext_test"),
)
PASS_A = "GW-E2E-015"
PASS_B = "GW-E2E-018"
PASS_REASONS = {
    PASS_A: f"Phase 22 v1 re-score ({PASS_A})",
    PASS_B: f"Phase 22 v1 re-run scoring ({PASS_B})",
}
ID_MAP_HEADER = (
    "> **Re-run values, not Phase 14's.** The scorer labels them with the folder IDs: "
    + ", ".join(f"{old} = {new}" for old, new in RERUN_IDS.items())
    + "."
)
# The scorer writes these under its data root, and `score --resume` deletes two of them.
OWNED_FOLDERS = ("handoff", "GW-E2E-009", "GW-E2E-009/official")
OUTPUTS_MANIFEST = "manifests/processed/phase14_outputs.json"


# ------------------------------------------------------------------------------- the tree


def _git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"git {args[0]} failed: {done.stderr.strip()[:200]}")
    return done.stdout.strip()


def git_toplevel(start: Path) -> Path:
    return Path(_git(start, "rev-parse", "--show-toplevel")).resolve()


def worktrees(start: Path) -> list[Path]:
    lines = _git(start, "worktree", "list", "--porcelain").splitlines()
    return [Path(ln.split(" ", 1)[1]).resolve() for ln in lines if ln.startswith("worktree ")]


def check_imports(root: Path, names: Sequence[str]) -> None:
    """Refuse unless each named module, and every loaded ``glypharis`` module, comes from ``root``.

    ``-P`` keeps this script's own folder off ``sys.path``; this check does not rely on it."""
    root = root.resolve()
    outside = []
    for name, module in list(sys.modules.items()):
        if name not in names and name.split(".")[0] != "glypharis":
            continue
        file = getattr(module, "__file__", None)
        if file is not None and root not in Path(file).resolve().parents:
            outside.append(name)
    missing = [n for n in names if n not in sys.modules]
    if outside or missing:
        raise SystemExit(
            f"imports must come from the working tree: {len(outside)} modules load from "
            f"elsewhere (e.g. {sorted(outside)[:3]}), {len(missing)} are not loaded"
        )


def check_data_root(data: Path, start: Path, real: Sequence[str]) -> None:
    """The root lies outside every worktree, and what the code writes there is really there.

    A junction or link at ``processed/phase14``, at the outputs manifest or at a folder the scorer
    owns would send writes, or `score --resume`'s ``rmtree``, into the Phase 14 store."""
    if not data.is_dir():
        raise SystemExit("--data-root is not a folder")
    root = data.resolve()
    for tree in worktrees(start):
        if root == tree or tree in root.parents:
            raise SystemExit("--data-root lies inside a worktree; use the isolated root")
    for rel in ("manifests", "processed/phase14"):
        if not (data / rel).is_dir():
            raise SystemExit(f"{rel} is missing under --data-root")
    written = (
        "manifests",
        "manifests/processed",
        OUTPUTS_MANIFEST,
        "processed/phase14",
        *(f"processed/phase14/{f}" for f in real),
    )
    for rel in written:
        path = data / rel
        if path.exists() and path.resolve() != root / rel:
            raise SystemExit(f"{rel} under --data-root is a link; it must be a real one")
    manifest = data / OUTPUTS_MANIFEST
    if manifest.exists() and manifest.stat().st_nlink != 1:
        raise SystemExit("the outputs manifest under --data-root is hard-linked; copy it")


def wrapper_commit() -> str:
    return _git(WRAPPER_ROOT, "rev-parse", "HEAD")


def require_wrapper_pin(frozen: Any) -> dict[str, Any]:
    try:
        pin: dict[str, Any] = frozen.pinned_code_check(WRAPPER_TAG, root=WRAPPER_ROOT)
    except RuntimeError as exc:
        raise SystemExit(f"the wrapper is not pinned at {WRAPPER_TAG}: {exc}") from exc
    return pin


# -------------------------------------------------------------------------------- re-run


def logged_predictions(log: Path, benchmark: str, split: str, exp: str, checkpoint: str) -> int:
    """Prediction rows of this set and ID. Scoring rows name another checkpoint."""
    if not log.is_file():
        raise SystemExit("the test-access log does not exist")
    want = [f"{benchmark} {split}", exp, checkpoint]
    rows = 0
    for line in log.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows += len(cells) == 6 and cells[1:4] == want
    return rows


def _recorded(ext: Any, data: Path, out: Path) -> bool:
    manifest = data / ext.OUTPUTS_MANIFEST
    if not manifest.is_file():
        return False
    entries = json.loads(manifest.read_text(encoding="utf-8")).get("outputs", [])
    return any(e.get("file") == ext.data_rel(out, data) for e in entries)


def attempt_state(ext: Any, args: argparse.Namespace, data: Path, new_id: str) -> str:
    """``fresh``, ``resume``, ``close`` or ``validation``; a repeated test attempt stops here.

    `ext_eval` starts afresh once an output has been moved aside, so the access log decides. Only
    a crash leaves the last run ``running`` (`ext_eval.run`), so a finished set that was emptied or
    cut short is a second attempt, not a resume."""
    spec = ext.EVAL_SETS[args.set]
    out = ext.output_path(data, args.folder_id, args.set)
    name = f"{new_id} {args.mode} {args.set}"
    recorded = _recorded(ext, data, out)
    if recorded and not out.exists():
        raise SystemExit(f"{name}: the root's manifest records an output that is not there")
    if not ext.is_test_split(spec["split"]):
        return "validation"
    rows = logged_predictions(
        ACCESS_LOG, spec["benchmark"], spec["split"], new_id, f"{ext.SYSTEM} ({args.mode})"
    )
    done = ext.done_keys(out)
    if not rows:
        if done or recorded:
            raise SystemExit(f"{name}: an output exists with no logged attempt; refusing")
        return "fresh"
    if not out.exists():
        raise SystemExit(
            f"{name}: already attempted and its output is gone. A further attempt needs a new "
            "registered ID"
        )
    meta_path = out.with_suffix(".meta.json")
    if not meta_path.is_file():
        raise SystemExit(f"{name}: an attempt is logged but its run metadata is missing")
    runs = json.loads(meta_path.read_text(encoding="utf-8")).get("runs") or []
    last_status = runs[-1].get("status") if runs else None
    keys, _, _ = ext.set_records(data, args.set, args.mode)
    if len(done) < len(keys):
        if recorded or last_status != "running":
            raise SystemExit(
                f"{name}: this set finished, so this is not a crash resume. A further attempt "
                "needs a new registered ID"
            )
        return "resume"
    if recorded:
        raise SystemExit(
            f"{name}: already complete. The test re-run is done once per set; a further attempt "
            "needs a new registered ID"
        )
    return "close"


def check_test_order(ext: Any, args: argparse.Namespace, data: Path) -> None:
    """A test set starts only when the set before it, in Phase 14's order, is recorded."""
    if not ext.is_test_split(ext.EVAL_SETS[args.set]["split"]):
        return
    previous = dict(zip(TEST_ORDER[1:], TEST_ORDER[:-1], strict=True)).get((args.mode, args.set))
    if previous is None:
        return
    mode, eval_set = previous
    if not _recorded(ext, data, ext.output_path(data, TEST_FOLDERS[mode], eval_set)):
        raise SystemExit(
            f"{args.mode} {args.set} comes after {mode} {eval_set} in Phase 14's order, and that "
            "run is not recorded under this root"
        )


def check_tag_worktree(ext: Any, args: argparse.Namespace) -> None:
    """`ext_eval`'s own refusals, before the wrapper opens any record. It repeats them itself."""
    request = argparse.Namespace(
        mode=args.mode,
        experiment=args.folder_id,
        set=args.set,
        reason=args.reason,
        device="cuda",  # `ext_eval.main`'s default, which the re-run uses
        allow_unpinned=False,
    )
    ext.check_request(request)
    try:
        ext.require_branch("HEAD")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    ext.pinned_state(request, True)


def patch_ext_eval(
    ext: Any, folder: str, new_id: str, stamp: dict[str, Any], resume_total: int | None
) -> None:
    """Module attributes only: the detached tag worktree, the Phase 22 log, IDs and metadata.

    ``resume_total`` is set for a resume, whose row must say so even with no record written yet
    (`ext_eval` adds the words only when it finds records)."""
    ext.PHASE14_BRANCH = "HEAD"
    ext.ACCESS_LOG = ACCESS_LOG.resolve()
    # Unwrapped first, so a second call in one process does not stack the wrappers.
    log_access = inspect.unwrap(ext.open_test_access)
    write_json = inspect.unwrap(ext._write_json)

    @functools.wraps(log_access)
    def open_access(
        benchmark: str,
        split: str,
        experiment_id: str,
        checkpoint: str,
        phase: int,
        reason: str,
        log_path: Path,
    ) -> None:
        if experiment_id != folder or phase != SOURCE_PHASE:
            raise SystemExit(f"unexpected access request for {experiment_id}, phase {phase}")
        if resume_total is not None and "(resumed" not in reason:
            reason += f" (resumed: 0 of {resume_total} samples already predicted)"
        log_access(benchmark, split, new_id, checkpoint, PHASE, reason, log_path)

    @functools.wraps(write_json)
    def stamped_write(path: Path, doc: dict[str, Any]) -> None:
        if path.name.endswith(".meta.json"):
            doc = {**doc, "phase": PHASE, "experiment": new_id, "folder_id": folder, **stamp}
        write_json(path, doc)

    ext.open_test_access = open_access
    ext._write_json = stamped_write


def run_rerun(args: argparse.Namespace) -> int:
    if RERUN_IDS.get(args.folder_id) != args.experiment:
        raise SystemExit(f"{args.experiment} is not the registered re-run of {args.folder_id}")
    tag_root = git_toplevel(Path.cwd())
    ext = importlib.import_module("ext_eval")
    frozen = importlib.import_module("glypharis.pipeline.frozen")
    check_imports(tag_root, ("ext_eval", "run_baselines", "glypharis"))
    data = Path(args.data_root)
    check_data_root(data, tag_root, (args.folder_id,))
    pin = require_wrapper_pin(frozen)
    check_tag_worktree(ext, args)
    check_test_order(ext, args, data)
    state = attempt_state(ext, args, data, args.experiment)
    total = None
    if state == "resume":
        total = len(ext.set_records(data, args.set, args.mode)[0])
    script = Path(__file__).resolve()
    stamp = {
        "phase22": {
            "experiment": args.experiment,
            "folder_id": args.folder_id,
            "wrapper": "scripts/phase22_v1_repro.py",
            "wrapper_commit": wrapper_commit(),
            "wrapper_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "wrapper_tag": WRAPPER_TAG,
            "wrapper_tag_commit": pin.get("tag_commit"),
        }
    }
    patch_ext_eval(ext, args.folder_id, args.experiment, stamp, total)
    code: int = ext.main(
        [
            args.mode,
            "--set",
            args.set,
            "--experiment",
            args.folder_id,
            "--reason",
            args.reason,
            "--data-root",
            str(data),
        ]
    )
    return code


# ------------------------------------------------------------------------------- re-score


def phase5_inputs(ps: Any, data: Path) -> list[str]:
    """Every pack and page file the Phase 5 re-score reads, data-root relative."""
    needed: set[str] = set()
    for job in ps.phase5_jobs(data):
        spec = ps.EVAL_SETS[job.eval_set]
        if job.task == "rec":
            needed.update(f"{spec['pack']}{ext}" for ext in (".bin", ".index.jsonl", ".meta.json"))
        else:
            needed.add(spec["pages"])
    return sorted(needed)


def check_rescore_root(ps: Any, data: Path) -> None:
    missing = [p for p in phase5_inputs(ps, data) if not (data / p).is_file()]
    if missing:
        raise SystemExit(
            f"{len(missing)} files the Phase 5 re-score reads are missing, e.g. {missing[0]}"
        )
    if ps.read_pass(data) is None:
        prefix = ps.HANDOFF.as_posix() + "/"
        doc = ps._read_outputs(data)
        if any(ps._norm_file(str(e.get("file"))).startswith(prefix) for e in doc["outputs"]):
            raise SystemExit("the root's manifest still lists hand-off files; drop them first")


def check_rerun_meta(ps: Any, path: Path, folder: str) -> dict[str, Any]:
    """The run metadata of a file `rerun` wrote for ``folder``, or stop."""
    new_id = RERUN_IDS[folder]
    meta: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    stamp = meta.get("phase22")
    pin = ps._find_pin(meta) or {}
    fields = (meta.get("phase"), meta.get("experiment"), meta.get("folder_id"))
    if (
        fields != (PHASE, new_id, folder)
        or not isinstance(stamp, dict)
        or (stamp.get("experiment"), stamp.get("folder_id")) != (new_id, folder)
        or stamp.get("wrapper_tag") != WRAPPER_TAG
        or (pin.get("tag"), pin.get("trees")) != ("phase14-eval", "equal")
    ):
        raise SystemExit(f"{folder}/{path.name} was not written by the {new_id} re-run")
    return meta


def check_root_matches_pass(ps: Any, data: Path, expected: str, id_map: bool) -> None:
    """Pass A scores Phase 14's files and Pass B the re-run's; a stopped pass stays itself."""
    state = ps.read_pass(data)
    if state is not None and state.get("experiment") != expected:
        raise SystemExit(f"this root's pass is {state.get('experiment')}, not {expected}")
    for job in ps.glypharis_jobs(data):
        path = job.path.with_suffix(".meta.json")
        if not path.is_file():
            continue  # the scorer's preflight refuses it
        if id_map:
            check_rerun_meta(ps, path, job.experiment)
        elif "phase22" in json.loads(path.read_text(encoding="utf-8")):
            raise SystemExit(f"{job.label}: a re-run's file is not Phase 14's ({expected})")


def patch_scorer(ps: Any, eb: Any, experiment: str, id_map: bool) -> None:
    ps.PHASE14_BRANCH = PHASE22_BRANCH
    ps.EXPERIMENT = experiment
    ps.REASON = PASS_REASONS[experiment]

    class Phase22Scorer(ps.Scorer):
        """The unchanged scorer, logging phase 22. ``ps.PHASE`` routes baseline paths."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.ctx.phase = PHASE
            self.gctx.phase = PHASE

    ps.Scorer = Phase22Scorer
    if id_map:
        log_access = eb.open_test_access

        def open_access(
            benchmark: str, split: str, experiment_id: str, *rest: Any, **kwargs: Any
        ) -> None:
            mapped = PASS_B_IDS.get(experiment_id, experiment_id)
            log_access(benchmark, split, mapped, *rest, **kwargs)

        eb.open_test_access = open_access


def annotate(out: Path) -> None:
    """Say on Pass B's files that their folder IDs label re-run values. Safe to repeat."""
    path = out / "results.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("experiment") != PASS_B:
        raise SystemExit(f"{path.as_posix()} is not {PASS_B}'s report")
    doc = {"id_map": dict(RERUN_IDS), "id_map_note": ID_MAP_HEADER, **doc}
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    for md in sorted(out.glob("*.md")):
        body = md.read_text(encoding="utf-8")
        if not body.startswith(ID_MAP_HEADER):
            md.write_text(f"{ID_MAP_HEADER}\n\n{body}", encoding="utf-8", newline="\n")


def run_rescore(args: argparse.Namespace) -> int:
    expected = PASS_B if args.id_map else PASS_A
    if args.experiment not in (None, expected):
        raise SystemExit(f"this pass is {expected}, not {args.experiment}")
    if args.command != "check":
        if args.out is None or args.experiment is None:
            raise SystemExit(f"{args.command} needs --experiment and --out")
        if args.out.resolve().parts[-3:] != ("reports", "final_validation", args.experiment):
            raise SystemExit(f"--out must be reports/final_validation/{args.experiment}")
    if args.command == "annotate":
        if not args.id_map:
            raise SystemExit("annotate is for Pass B (--id-map rerun)")
        annotate(args.out)
        return 0
    if args.data_root is None:
        raise SystemExit(f"{args.command} needs --data-root")
    worktree = git_toplevel(Path.cwd())
    ps = importlib.import_module("phase14_score")
    eb = importlib.import_module("evaluate_baselines")
    frozen = importlib.import_module("glypharis.pipeline.frozen")
    check_imports(worktree, ("phase14_score", "evaluate_baselines", "run_baselines", "glypharis"))
    check_data_root(args.data_root, worktree, OWNED_FOLDERS)
    try:
        frozen.pinned_code_check(WRAPPER_TAG, root=worktree)
    except RuntimeError as exc:
        raise SystemExit(f"the tree is not pinned at {WRAPPER_TAG}: {exc}") from exc
    check_rescore_root(ps, args.data_root)
    check_root_matches_pass(ps, args.data_root, expected, bool(args.id_map))
    patch_scorer(ps, eb, expected, bool(args.id_map))
    argv = [args.command, "--data-root", str(args.data_root)]
    for flag, value in (
        ("--out", args.out),
        ("--eff-dir", args.eff_dir),
        ("--record", args.record),
    ):
        if value is not None:
            argv += [flag, str(value)]
    if args.resume:
        argv.append("--resume")
    code: int = ps.main(argv)
    if code == 0 and args.command == "score" and args.id_map:
        annotate(args.out)
    return code


# ------------------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    rerun = sub.add_parser("rerun", help="run the frozen model from the tag worktree")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--set", required=True)
    common.add_argument("--folder-id", required=True, choices=sorted(RERUN_IDS))
    common.add_argument("--experiment", required=True, choices=sorted(RERUN_IDS.values()))
    common.add_argument("--reason", required=True)
    common.add_argument("--data-root", required=True)
    modes = rerun.add_subparsers(dest="mode", required=True)
    for mode in ("rec", "e2e", "oracle"):
        modes.add_parser(mode, parents=[common])
    rescore = sub.add_parser("rescore", help="run the unchanged scorer on an isolated root")
    rescore.add_argument("command", choices=("check", "score", "annotate"))
    rescore.add_argument("--data-root", type=Path)
    rescore.add_argument("--experiment", choices=(PASS_A, PASS_B))
    rescore.add_argument("--id-map", choices=("rerun",))
    rescore.add_argument("--out", type=Path)
    rescore.add_argument("--eff-dir", type=Path)
    rescore.add_argument("--record", type=Path)
    rescore.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    return run_rerun(args) if args.action == "rerun" else run_rescore(args)


if __name__ == "__main__":
    sys.exit(main())
