"""Check every Phase 14 input before the tag, and record the crop packs' index hashes (Phase 14).

    python scripts/phase14_prepare.py record-index-hashes
    python scripts/phase14_prepare.py check

The plan's Order step 2 (docs/PHASE_STATUS.md, Phase 14 item 6) as one command. ``check``, for
every evaluation set Phase 14 reads (`PHASE14_EVAL_SETS`):

- `glypharis.eval.inputs.check_inputs` for each mode the plan runs on the set (`ext_eval.PLAN`
  and `run_baselines.PHASE14_PLAN`), with every input required to have a recorded hash and every
  pack a recorded index hash, exactly as the runs will check them;
- page ids and line ids are unique, and every crop of a pack has its line on a page;
- every page image's header size (EXIF orientation applied) equals its page record's;
- the new packs (FUNSD, GDC-scan) record the size the plan registered
  (`build_eval_crops.registered_count`);
- GDC-scan's photos carry no rotation tag, have the size alignment.json records, and hash to the
  value both alignment.json and gdc_scan_v1.json record;
- GDC's alignment.json capture hashes equal gdc_v1.json's, and the photos hash to them.

``record-index-hashes`` adds ``index_sha256`` (the SHA-256 of ``<pack>.index.jsonl``, the labels)
to every pack entry of eval_crops.json and benchmarks.json that lacks one, after that pack's .bin
hash matches both its record and its own .meta.json. It never overwrites a value, writes nothing
if any pack fails, and refuses once the ``phase14-eval`` tag exists: the hashes are part of what
the tag pins.

Only bytes are hashed and image headers read: no model runs, no pixel is decoded and nothing is
scored, so this is preprocessing, not a test access. The output holds counts, file names and
hashes only, never a label or a line's text. Both commands refuse to start off `PHASE14_BRANCH`;
``check`` writes nothing and exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import build_eval_crops
import ext_eval
import run_baselines

from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackReader
from glypharis.data.pages import Page, read_pages
from glypharis.eval.inputs import (
    MODES,
    RECORD_MANIFESTS,
    InputError,
    check_inputs,
    duplicate_line_ids,
    header_size,
    image_sizes_match,
)
from glypharis.eval.sets import (
    EVAL_SETS,
    PHASE5_HASHES,
    PHASE14_BRANCH,
    SUBSET_NAME,
    repo_root,
    require_branch,
)
from glypharis.pipeline.frozen import PIN_TAG
from glypharis.pipeline.frozen import tag_exists as _tag_exists

RUNNER_BATCH = 16  # run_baselines' --batch default, which the Phase 5 runs used
# The tracked records the tag pins; they must be committed before it (Order step 2).
STEP2_RECORDS = ("data/manifests",)
PHASE14_EVAL_SETS = ("hiertext_val", "hiertext_test", "gdc_test", "gdc_scan_test", "funsd_test",
                     "iiit5k_test", "omnidocbench_test")  # fmt: skip
REGISTERED_PACKS = ("funsd_test", "gdc_scan_test")  # pack sizes fixed by the plan (item 3)
EVAL_CROPS = RECORD_MANIFESTS[0]
GDC_RECORD = "manifests/gdc/gdc_v1.json"  # under the data root
GDC_ALIGNMENT = "manifests/gdc/alignment.json"
GDC_SCAN_RECORD = build_eval_crops.GDC_SCAN_RECORD
GDC_SCAN_ALIGNMENT = build_eval_crops.GDC_SCAN_ALIGNMENT
_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, EOFError)


def planned_modes(eval_set: str) -> list[str]:
    """The modes the registered plan runs on ``eval_set``: GLYPHARIS (`ext_eval.PLAN`) and the
    baselines (`run_baselines.PHASE14_PLAN` rows that run, the OmniDocBench subset included)."""
    modes = {mode for mode, runs in ext_eval.PLAN.items() for sets in runs.values()
             if eval_set in sets}  # fmt: skip
    modes |= {task for (_, task, name), why in run_baselines.PHASE14_PLAN.items()
              if name == eval_set and why in (None, SUBSET_NAME)}  # fmt: skip
    return [m for m in MODES if m in modes]


def _load(data: Path, rel: str) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((data / rel).read_text(encoding="utf-8"))
    return doc


# ------------------------------------------------------------------------------------ check


def _check_pages(data: Path, eval_set: str, pages: list[Page], fail: list[str]) -> None:
    spec = EVAL_SETS[eval_set]
    ids = Counter(p.doc_id for p in pages)
    if twice := sum(n > 1 for n in ids.values()):
        fail.append(f"{spec['pages']}: {twice} page ids appear more than once")
    if dup := duplicate_line_ids(pages):
        fail.append(f"{spec['pages']}: {len(dup)} line ids appear more than once")
    fail.extend(image_sizes_match(data, pages))
    if spec["pack"] and "rec" in planned_modes(eval_set):
        try:
            keys = [r.key for r in LinePackReader(data / spec["pack"]).records]
        except _READ_ERRORS as exc:
            fail.append(f"{spec['pack']}: cannot be read ({type(exc).__name__})")
            return
        lines = {ln.line_id for p in pages for ln in p.lines}
        if missing := sum(k not in lines for k in keys):
            fail.append(f"{spec['pack']}: {missing} crops have no line in {spec['pages']}")


def _check_registered(data: Path, eval_set: str, pages: list[Page], fail: list[str]) -> None:
    try:
        registered = build_eval_crops.registered_count(data, eval_set, pages)
    except SystemExit as exc:
        fail.append(str(exc))
        return
    entry = (_load(data, EVAL_CROPS) if (data / EVAL_CROPS).is_file() else {}).get(eval_set)
    if not isinstance(entry, dict) or entry.get("expected_count") != registered:
        recorded = entry.get("expected_count") if isinstance(entry, dict) else None
        fail.append(f"data/{EVAL_CROPS}: {eval_set} records expected_count {recorded}, the plan "
                    f"registers {registered} (cut it with build_eval_crops.py)")  # fmt: skip


def _check_captures(data: Path, record_rel: str, alignment_rel: str, fail: list[str],
                    *, no_rotation: bool) -> int:  # fmt: skip
    """Capture photos against the freeze record and the alignment record; returns how many."""
    record = _load(data, record_rel).get("captures") or {}
    results = _load(data, alignment_rel).get("results") or []
    aligned = {str(r.get("file")): r for r in results}
    if set(aligned) != set(record):
        fail.append(f"data/{alignment_rel} and data/{record_rel} list different captures "
                    f"({len(set(aligned) ^ set(record))} differ)")  # fmt: skip
    for name, result in sorted(aligned.items()):
        if name in record and result.get("sha256") != record[name]:
            fail.append(f"{name}: data/{alignment_rel} and data/{record_rel} record different "
                        "hashes")  # fmt: skip
        path = data / name
        try:
            (w, h), orientation = header_size(path)
            sha = sha256_file(path)
        except _READ_ERRORS as exc:
            fail.append(f"{name}: cannot be read ({type(exc).__name__})")
            continue
        if sha != result.get("sha256"):
            fail.append(f"{name}: SHA-256 {sha}, data/{alignment_rel} records "
                        f"{result.get('sha256')}")  # fmt: skip
        if no_rotation and orientation != 1:
            fail.append(f"{name}: carries EXIF orientation {orientation}; the photos must have "
                        "no rotation tag")  # fmt: skip
        if no_rotation and [w, h] != list(result.get("size") or []):
            fail.append(f"{name}: image is {w}x{h}, data/{alignment_rel} records "
                        f"{result.get('size')}")  # fmt: skip
    return len(aligned)


def check(data: Path) -> list[str]:
    """Run every step-2 check; print one summary line per set (counts only); return failures."""
    failures: list[str] = []
    for eval_set in PHASE14_EVAL_SETS:
        spec = EVAL_SETS[eval_set]
        fail: list[str] = []
        modes = planned_modes(eval_set)
        pages: list[Page] | None = None
        if spec["pages"]:
            try:
                pages = list(read_pages(data / spec["pages"]))
            except _READ_ERRORS as exc:
                fail.append(f"{spec['pages']}: cannot be read ({type(exc).__name__})")
        counts: dict[str, Any] = {}
        for mode in modes:
            if spec["pages"] and pages is None:
                break
            try:
                out = check_inputs(data, eval_set, mode, None if pages is None else len(pages))
            except InputError as exc:
                fail.append(f"{eval_set} ({mode}): {exc}")
                continue
            except _READ_ERRORS as exc:
                fail.append(f"{eval_set} ({mode}): an input cannot be read ({type(exc).__name__})")
                continue
            counts.update({k: v["count"] for k, v in out.items()})
        if pages is not None:
            _check_pages(data, eval_set, pages, fail)
            counts["lines"] = sum(len(p.lines) for p in pages)
            if eval_set in REGISTERED_PACKS:
                _check_registered(data, eval_set, pages, fail)
        try:
            if eval_set == "gdc_scan_test":
                counts["photos"] = _check_captures(data, GDC_SCAN_RECORD, GDC_SCAN_ALIGNMENT, fail,
                                                   no_rotation=True)  # fmt: skip
            elif eval_set == "gdc_test":
                counts["photos"] = _check_captures(data, GDC_RECORD, GDC_ALIGNMENT, fail,
                                                   no_rotation=False)  # fmt: skip
        except _READ_ERRORS as exc:
            fail.append(f"{eval_set}: a capture record cannot be read ({type(exc).__name__})")
        shown = ", ".join(f"{k} {v}" for k, v in counts.items() if v is not None)
        print(f"{eval_set} [{', '.join(modes)}]: {shown or 'nothing read'}: "
              f"{'ok' if not fail else f'{len(fail)} failures'}")  # fmt: skip
        failures.extend(fail)
    return failures


# ------------------------------------------------------------------------- index hashes


def tag_exists(tag: str = PIN_TAG) -> bool:
    """Whether ``tag`` is set in the repository that holds the working directory (a git error
    raises; it never reads as "no tag")."""
    try:
        return _tag_exists(tag, root=repo_root())
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def check_phase5_batches(data: Path) -> tuple[dict[str, Any], list[str]]:
    """The batch each planned Phase 14 baseline run must use, checked against Phase 5's run
    metadata now (`run_baselines.check_phase5_batch`, metadata only), so a refusal is found
    before the tag rather than at the run. TrOCR batches its crops (``--batch``, default
    `RUNNER_BATCH`); every other system reads one image at a time."""
    seen: dict[str, Any] = {}
    fail: list[str] = []
    for system, task in sorted({(s, t) for (s, t, _e), v in run_baselines.PHASE14_PLAN.items()
                                if v in (None, SUBSET_NAME)}):  # fmt: skip
        fixed = system != "trocr"
        batch = 1 if fixed else RUNNER_BATCH
        try:
            seen[f"{system} ({task})"] = run_baselines.check_phase5_batch(
                data, system, task, batch, fixed=fixed
            )["inference_batch"]
        except SystemExit as exc:
            fail.append(str(exc))
    return seen, fail


def check_records_committed(data: Path) -> list[str]:
    """The Phase 5 hash record exists, and every record under data/manifests is committed: the
    tag pins them, so an uncommitted one makes every run after the tag refuse. Checked only when
    ``data`` is the repository's own data folder."""
    root = repo_root()
    if data.resolve() != (root / "data").resolve():
        return []
    fail = []
    phase5 = "data/" + PHASE5_HASHES.as_posix()
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", phase5], cwd=root,
                             capture_output=True, text=True, timeout=30)  # fmt: skip
    if tracked.returncode != 0:
        fail.append(f"{phase5} is not committed: run `phase14_score.py hash-phase5` and commit it")
    status = subprocess.run(["git", "status", "--porcelain", "--no-renames",
                             "--untracked-files=all", "--", *STEP2_RECORDS], cwd=root,
                            capture_output=True, text=True, timeout=30)  # fmt: skip
    dirty = [ln[3:] for ln in status.stdout.splitlines() if ln.strip()]
    if status.returncode != 0 or dirty:
        fail.append(f"uncommitted records under {', '.join(STEP2_RECORDS)}: {dirty[:10]}; commit "
                    "them before the tag")  # fmt: skip
    return fail


def record_index_hashes(data: Path) -> tuple[int, list[str]]:
    """Add the missing ``index_sha256`` values (see the module docstring); returns how many were
    added and the failures. Nothing is written when anything fails."""
    if tag_exists(PIN_TAG):
        raise SystemExit(f"the {PIN_TAG} tag exists: the index hashes are recorded before the tag "
                         "(Order step 2), never after it")  # fmt: skip
    failures: list[str] = []
    updated: dict[str, dict[str, Any]] = {}
    names: list[str] = []
    added = 0
    for rel in RECORD_MANIFESTS:
        if not (data / rel).is_file():
            continue
        doc = _load(data, rel)
        changed = False
        for name, entry in doc.items():
            if not (isinstance(entry, dict) and entry.get("pack") and entry.get("bin_sha256")):
                continue
            if entry.get("index_sha256"):
                continue  # never overwritten
            prefix = data / entry["pack"]
            try:
                bin_sha = sha256_file(prefix.with_suffix(".bin"))
                own = json.loads(prefix.with_suffix(".meta.json").read_text(encoding="utf-8"))
                index_sha = sha256_file(prefix.with_suffix(".index.jsonl"))
            except _READ_ERRORS as exc:
                failures.append(f"{entry['pack']}: cannot be read ({type(exc).__name__})")
                continue
            if bin_sha != entry["bin_sha256"] or bin_sha != own.get("bin_sha256"):
                failures.append(f"{entry['pack']}.bin has SHA-256 {bin_sha}; data/{rel} records "
                                f"{entry['bin_sha256']} and its .meta.json "
                                f"{own.get('bin_sha256')}")  # fmt: skip
                continue
            out: dict[str, Any] = {}
            for key, value in entry.items():
                out[key] = value
                if key == "bin_sha256":
                    out["index_sha256"] = index_sha
            doc[name] = out
            changed = True
            added += 1
            names.append(f"data/{rel} {name}")
        if changed:
            updated[rel] = doc
    if failures:
        print("record-index-hashes: nothing written")
        return 0, failures
    for rel, doc in updated.items():
        path = data / rel
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
        tmp.replace(path)
    for name in names:  # only once the files are written
        print(f"{name}: index_sha256 recorded")
    return added, failures


# ------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("command", choices=("check", "record-index-hashes"))
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)
    try:
        require_branch(PHASE14_BRANCH)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    data = Path(args.data_root)
    if args.command == "check":
        failures = check(data)
        batches, batch_fail = check_phase5_batches(data)
        print(f"Phase 5 batches: {len(batches)} planned baseline runs checked: "
              f"{'ok' if not batch_fail else f'{len(batch_fail)} failures'}")  # fmt: skip
        failures += batch_fail + check_records_committed(data)
    else:
        added, failures = record_index_hashes(data)
        if not failures:
            print(f"record-index-hashes: {added} added")
    for line in failures:
        print(f"FAIL {line}")
    print(f"{args.command}: {'passed' if not failures else f'{len(failures)} failures'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
