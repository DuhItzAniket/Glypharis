"""Phase 22 item 14: the phone candidate's arms on the laptop and the accuracy part of selection
(GW-EFF-013).

    python scripts/phase22_candidate.py list
    python scripts/phase22_candidate.py build --bundle runs/release/glypharis-v1 --threads 10
    python scripts/phase22_candidate.py eval --arm fused-int8 --set lines \
        --bundle runs/release/glypharis-v1 --threads 10
    python scripts/phase22_candidate.py profile --arm fused --run 1 \
        --bundle runs/release/glypharis-v1 --threads 10
    python scripts/phase22_candidate.py accuracy
    python scripts/phase22_candidate.py laptop

The arms and the rule are in `glypharis.efficiency.candidate` and `research/hypotheses.md` (H11).
Every arm is a new version, not v1, and every output says so.

- ``build`` checks GW-EFF-009's fused recognizer against its recorded hash and writes the INT8 one
  with H7′'s recipe to ``data/processed/phase22/candidate/fused-int8/`` once, with its record.
- ``eval --set lines``: greedy line CER on the 5,000 validation lines seeded 20261028 (the export
  check's sample, as GW-EFF-009), read under the arm's batching rule: `CROPS_PER_CALL` crops per
  call to the arm's recognizer, which batches them by width, or for ``fused-single-line`` one line
  per network run. Per-line errors and reference lengths on both tracks, and each line's logits
  against a reference read the same way: PyTorch v1 for ``fused``, ``fused`` for the other arms
  (information; the selection judges CER).
- ``eval --set pages``: page CER (bag of lines) and E2E F1 on the 400 Phase 5 validation pages,
  scored as GW-EFF-009 scores them (`phase16_variants.eval_pages`), per-page counts kept.
- ``profile``: 10 warm-up and 100 timed Phase 5 profile pages, one at a time, as Phase 16 times
  them, with peak RAM and the machine conditions; the arm loaded as its bundle would load it (the
  v1 bundle through `load_bundle_onnx` and the arm's recognizer file, its hash checked). One
  process per arm and run; ``list`` prints the sitting's order (run 1 forward, run 2 back). The
  laptop's times decide nothing while a phone can be used.
- ``accuracy`` reads the six eval reports and writes the three verdicts of each arm against
  ``fused``: which arms go to the phone.
- ``laptop`` reads the six profiles (checked to be full, clean runs on the same pages, made in the
  mirrored order) and writes each arm's run means and peaks, and the speed and memory conditions
  computed on the laptop's page times: information only, since a phone can be used. The pages
  and profile reports' ``batching`` label names the recognizer's rule; on a page the pipeline
  hands the recognizer all of that page's crops in one call, and `CROPS_PER_CALL` applies to the
  line evaluation only.

The v1 code must equal the ``v1-code`` tag's before anything loads (`assert_v1_code`). Validation
data only. Nothing is overwritten. ``--threads N`` is required for build, eval and profile and is
applied before numpy and torch are imported. ``--limit`` and ``--smoke`` cut the item counts for a
smoke test; they need ``--out`` and mark the output ``"smoke": true``.
"""

# ruff: noqa: E402  (the thread variables are set before numpy and torch are imported)

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from glypharis.efficiency.threads import set_threads

THREADS_FLAG = "--threads"


def requested_threads(argv: Sequence[str]) -> int | None:
    """``--threads N`` or ``--threads=N`` from ``argv`` when it is a positive integer."""
    for i, arg in enumerate(argv):
        value = None
        if arg == THREADS_FLAG and i + 1 < len(argv):
            value = argv[i + 1]
        elif arg.startswith(THREADS_FLAG + "="):
            value = arg.split("=", 1)[1]
        if value is not None:
            try:
                n = int(value)
            except ValueError:
                return None
            return n if n > 0 else None
    return None


THREADS: dict[str, Any] | None = None
if __name__ == "__main__":
    _n = requested_threads(sys.argv[1:])
    if _n is not None:
        THREADS = set_threads(_n)

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from export_check import LINES, SEED
from phase16_variants import EVAL_SET, TRACKS, eval_pages, eval_paths, line_sample, profile_pages
from run_baselines import PROFILE_SAMPLES, WARMUP, cpu_busy_percent, git_state, peak_working_set

from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackReader
from glypharis.efficiency.candidate import (
    ARMS,
    BATCHING,
    BUILD_FILE,
    CROPS_PER_CALL,
    DEFAULT_ARM,
    EXPERIMENT,
    FUSED_SHA256,
    INT8_FILE,
    MARGINS,
    PHASE16_FUSED,
    REPORTS,
    VERSION_LABEL,
    WORK_ROOT,
    accuracy_tests,
    arm_order,
    batching_of,
    check_fused,
    memory_test,
    recognizer_file,
    single_line,
    sitting_order,
    speed_tests,
)
from glypharis.efficiency.conditions import ProfileConditions
from glypharis.efficiency.files import write_json_atomic, write_text_atomic
from glypharis.efficiency.onnx_build import (
    check_sessions,
    load_bundle_onnx,
    onnx_recognizer,
    with_parts,
)
from glypharis.efficiency.quant import quantize_recognizer
from glypharis.efficiency.threads import check_threads
from glypharis.efficiency.v1 import assert_v1_code
from glypharis.efficiency.variants import (
    WORK_ROOT as PHASE16_WORK,
)
from glypharis.efficiency.variants import (
    Build,
    build_frozen,
    load_variant,
    report_record,
)
from glypharis.eval.text_metrics import score_pair
from glypharis.pipeline.frozen import FREEZE_RECORD
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer

FUSED_VARIANT = "fused-attention"  # GW-EFF-009's name for the fused arm


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _refuse_existing(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise SystemExit(f"{path.as_posix()} exists: move it aside first")


def _write_new(path: Path, doc: dict[str, Any]) -> None:
    _refuse_existing(path)
    write_json_atomic(path, doc)


def _thread_record(n: int) -> dict[str, Any]:
    return THREADS if THREADS is not None else set_threads(n, require_fresh=False)


def _report_dir(arm: str) -> Path:
    return REPORTS / EXPERIMENT / arm


# ------------------------------------------------------------------------------------ arms


def int8_record(work_root: Path = WORK_ROOT) -> dict[str, Any]:
    """The INT8 build's record, its output file checked against it."""
    path = work_root / "fused-int8" / BUILD_FILE
    if not path.is_file():
        raise SystemExit(f"{path.as_posix()}: no INT8 build; run build first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    out = work_root / "fused-int8" / INT8_FILE
    if not out.is_file() or sha256_file(out) != doc["output"]["sha256"]:
        raise SystemExit(f"{out.as_posix()} is missing or not the file {path.as_posix()} records")
    if doc["input"]["sha256"] != FUSED_SHA256:
        raise SystemExit(f"{path.as_posix()} was not made from GW-EFF-009's fused recognizer")
    return doc


def arm_recognizer(
    arm: str, fused: TextRecognizer, threads: int, work_root: Path
) -> TextRecognizer:
    """The arm's recognizer, made from the fused one (same charset and settings)."""
    if arm == "fused":
        return fused
    if arm == "fused-single-line":
        return single_line(fused)
    int8_record(work_root)
    return onnx_recognizer(work_root / arm / INT8_FILE, fused, threads)


def arm_build(arm: str, fused: Build, threads: int, work_root: Path) -> Build:
    """The arm's pipeline: GW-EFF-009's checked fused build with the arm's recognizer."""
    rec = arm_recognizer(arm, fused.ocr.recognizer, threads, work_root)
    ocr = fused.ocr if rec is fused.ocr.recognizer else with_parts(fused.ocr, recognizer=rec)
    ocr.model_version = f"{fused.ocr.model_version} arm {arm} ({VERSION_LABEL})"
    record = {
        "arm": arm,
        "version": VERSION_LABEL,
        "batching": batching_of(rec),
        "recognizer_file": recognizer_file(arm, work_root).as_posix(),
        "recognizer_sha256": sha256_file(recognizer_file(arm, work_root)),
        "fused_build": report_record(fused.record),
    }
    if arm == "fused-int8":
        record["int8"] = int8_record(work_root)
    return Build(arm, ocr, record, fused.torch_reference if arm == DEFAULT_ARM else None)


def fused_build(args: argparse.Namespace) -> Build:
    check_fused(PHASE16_FUSED)
    frozen = build_frozen(".", args.record)
    return load_variant(
        FUSED_VARIANT, frozen, args.bundle, PHASE16_WORK / FUSED_VARIANT, args.threads
    )


# -------------------------------------------------------------------------------- accuracy


def eval_lines(
    rec: TextRecognizer, ref: TextRecognizer, ref_name: str, pack: Path, limit: int | None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Greedy line CER of ``rec`` on the seeded lines, `CROPS_PER_CALL` crops per call, and each
    line's logits and text against ``ref`` read the same way (information only)."""
    reader = LinePackReader(pack)
    idx = line_sample(len(reader))
    if limit is not None:
        idx = idx[:limit]
    per: dict[str, dict[str, list[int]]] = {t: {"errors": [], "ref_chars": []} for t in TRACKS}
    keys: list[str] = []
    texts: list[dict[str, Any]] = []
    worst, same = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(idx), CROPS_PER_CALL):
            chunk = [int(i) for i in idx[start : start + CROPS_PER_CALL]]
            for i in chunk:
                if reader.records[i].split != "validation":
                    raise SystemExit(f"line {reader.records[i].key} is not a validation line")
            crops = [reader.image(i) for i in chunk]
            logits = rec.logits(crops)
            ref_logits = ref.logits(crops)
            for i, lg, lr in zip(chunk, logits, ref_logits, strict=True):
                r = reader.records[i]
                text = rec.charset.ctc_greedy_decode(lg.argmax(-1).tolist())
                keys.append(r.key)
                texts.append({"key": r.key, "text": text})
                for track in TRACKS:
                    s = score_pair(text, r.label, track)  # type: ignore[arg-type]
                    per[track]["errors"].append(s.char_errors)
                    per[track]["ref_chars"].append(s.ref_chars)
                if lr.shape != lg.shape:
                    raise SystemExit(f"line {r.key}: {ref_name} gives {tuple(lr.shape)} frames")
                worst = max(worst, float((lr - lg).abs().max()))
                same += ref.charset.ctc_greedy_decode(lr.argmax(-1).tolist()) == text
    n = len(keys)
    cer = {t: sum(per[t]["errors"]) / max(1, sum(per[t]["ref_chars"])) for t in TRACKS}
    result = {
        "lines": n,
        "crops_per_call": CROPS_PER_CALL,
        "batching": batching_of(rec),
        "cer": cer,
        "per_line": {"keys": keys, "index": [int(i) for i in idx], **per},
        "against_reference": {
            "reference": ref_name,
            "reference_batching": batching_of(ref),
            "logit_max_abs_diff": worst,
            "identical_text": same,
            "lines": n,
            "note": "information; the selection rule judges line CER",
        },
    }
    return result, texts


def _texts_sha(texts: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def _common(
    args: argparse.Namespace,
    build: Build,
    v1_code: dict[str, Any],
    threads: dict[str, Any],
    sessions: dict[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT,
        "arm": build.name,
        "version": VERSION_LABEL,
        "eval_set": EVAL_SET,
        "device": "cpu",
        "threads": threads,
        "session_options": sessions,
        "v1_code": v1_code,
        "build": build.record,
        **git_state(),
    }


def _out(args: argparse.Namespace, name: str) -> Path:
    if args.out is not None:
        return Path(args.out)
    if getattr(args, "limit", None) is not None or getattr(args, "smoke", None) is not None:
        raise SystemExit("a smoke run (--limit/--smoke) needs --out")
    return _report_dir(args.arm) / name


# -------------------------------------------------------------------------------- commands


def cmd_list(_: argparse.Namespace) -> int:
    for arm in ARMS:
        margin = MARGINS.get(arm)
        extra = "the default and the reference" if margin is None else f"margin {margin}"
        print(f"{arm:18s} {BATCHING[arm]}; {extra}")
    order = ", ".join(f"{a} run {r}" for a, r in sitting_order())
    print(f"\nLaptop sitting ({EXPERIMENT}, pages): {order}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    threads = _thread_record(args.threads)
    v1_code = assert_v1_code(".")
    work = args.work_root / "fused-int8"
    record_path = work / BUILD_FILE
    _refuse_existing(record_path, work / INT8_FILE)
    fused_sha = check_fused(PHASE16_FUSED)
    record = quantize_recognizer(PHASE16_FUSED, work / INT8_FILE)
    if record["input"]["sha256"] != fused_sha:
        raise SystemExit("the quantiser read a different file than the one checked")
    doc = {
        "experiment": EXPERIMENT,
        "arm": "fused-int8",
        "version": VERSION_LABEL,
        **record,
        "v1_code": v1_code["v1_code"],
        "threads": threads["requested"],
        "created_utc": _utc(),
    }
    _write_new(record_path, doc)
    # The arm must load and run on the checked fused build before anything is measured.
    build = arm_build("fused-int8", fused_build(args), args.threads, args.work_root)
    check_sessions(build.ocr, args.threads)
    print(json.dumps({"output": record["output"], "nodes_int8": record["nodes_int8"]}, indent=1))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    threads = _thread_record(args.threads)
    out = _out(args, f"{args.set}.json")
    folder = out.parent if args.limit is not None else args.work_root / args.arm
    items_path = folder / f"{out.stem}_outputs.jsonl"
    _refuse_existing(out, items_path)
    v1_code = assert_v1_code(".")
    pages_file, pack = eval_paths(args.record, args.data_root)
    fused = fused_build(args)
    build = arm_build(args.arm, fused, args.threads, args.work_root)
    sessions = check_sessions(build.ocr, args.threads)
    check_threads(args.threads)
    started = _utc()
    if args.set == "lines":
        if args.arm == DEFAULT_ARM:
            assert fused.torch_reference is not None
            ref, ref_name = fused.torch_reference, "PyTorch v1 (the frozen recognizer)"
        else:
            ref, ref_name = fused.ocr.recognizer, "fused"
        result, items = eval_lines(build.ocr.recognizer, ref, ref_name, pack, args.limit)
        source = {
            "pack": pack.relative_to(args.data_root).as_posix(),
            "seed": SEED,
            "sample": f"{LINES} lines, the export check's",
            "decoding": "greedy CTC",
        }
    else:
        result, items = eval_pages(build, args.data_root, pages_file, args.limit)
        result["batching"] = batching_of(build.ocr.recognizer)
        source = {
            "pages_file": pages_file.relative_to(args.data_root).as_posix(),
            "sample": "the 400 Phase 5 pages (baseline_doc_ids)",
        }
    ended = _utc()
    check_threads(args.threads)
    _refuse_existing(out, items_path)
    write_text_atomic(items_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items))
    doc = {
        "kind": f"phase22-candidate-{args.set}",
        **_common(args, build, v1_code, threads, sessions),
        "set": args.set,
        "source": source,
        "smoke": args.limit is not None,
        "limit": args.limit,
        "started_utc": started,
        "ended_utc": ended,
        **result,
        "outputs": {
            "path": items_path.as_posix(),
            "sha256": sha256_file(items_path),
            "content_sha256": _texts_sha(items),
        },
    }
    _write_new(out, doc)
    head = (
        f"line CER {100 * result['cer']['strict']:.2f}% on {result['lines']} lines"
        if args.set == "lines"
        else f"page CER {100 * result['page_cer']:.2f}%, E2E F1 {result['e2e_f1']:.4f} on "
        f"{result['pages']} pages"
    )
    print(f"{args.arm}: {head}; written to {out.as_posix()}")
    return 0


def deployed_arm(
    arm: str, bundle: Path, threads: int, work_root: Path
) -> tuple[GlypharisOCR, dict[str, Any]]:
    """The arm as its bundle would load it: the v1 bundle (manifest checked) and the arm's own
    recognizer file, its hash checked; no PyTorch model and no training checkpoint."""
    path = recognizer_file(arm, work_root)
    if arm == "fused-int8":
        want = int8_record(work_root)["output"]["sha256"]
        if sha256_file(path) != want:
            raise SystemExit(f"{path.as_posix()} is not the INT8 file its record names")
    else:
        check_fused(path)
    base = load_bundle_onnx(bundle, threads)
    rec = onnx_recognizer(path, base.recognizer, threads)
    if arm == "fused-single-line":
        rec = single_line(rec)
    ocr = with_parts(base, recognizer=rec)
    loaded = ["the v1 bundle on ONNX Runtime (load_bundle_onnx)", path.as_posix()]
    return ocr, {
        "loaded": loaded,
        "recognizer_sha256": sha256_file(path),
        "batching": batching_of(rec),
    }


def cmd_profile(args: argparse.Namespace) -> int:
    import psutil

    threads = _thread_record(args.threads)
    n = args.threads
    out = _out(args, f"profile_pages_t{n}_run{args.run}.json")
    _refuse_existing(out)
    if args.arm not in arm_order(args.run):
        raise SystemExit(f"{args.arm} is not an arm")
    warmup, timed = (WARMUP, PROFILE_SAMPLES) if args.smoke is None else (1, args.smoke)
    v1_code = assert_v1_code(".")
    torch.set_grad_enabled(False)
    proc = psutil.Process()
    conditions = ProfileConditions()
    conditions.start()
    ocr, memory = deployed_arm(args.arm, args.bundle, n, args.work_root)
    sessions = check_sessions(ocr, n)
    threads_before = check_threads(n)
    started = _utc()
    got = profile_pages(ocr, args.data_root, warmup + timed)
    ended = _utc()
    threads_after = check_threads(n)
    peak = round(peak_working_set(proc) / 2**20, 1)
    conditions.end()
    # Then the checked chain: GW-EFF-009's fused build and the arm on it, same recognizer file.
    build = arm_build(args.arm, fused_build(args), n, args.work_root)
    if build.record["recognizer_sha256"] != memory["recognizer_sha256"]:
        raise SystemExit("the timed recognizer is not the checked one")
    times = got.pop("times")
    t = np.asarray(times[warmup:])
    doc: dict[str, Any] = {
        "kind": "phase22-candidate-profile",
        "system": "glypharis",
        "task": "e2e",
        **_common(args, build, v1_code, threads, sessions),
        "set": "pages",
        "run": args.run,
        "smoke": args.smoke is not None,
        "timing": "GlypharisOCR.predict of the RGB page array, one page at a time",
        "threads_before": threads_before,
        "threads_after": threads_after,
        "warmup": warmup,
        "started_utc": started,
        "ended_utc": ended,
        "warmup_times_ms": times[:warmup],
        "times_ms": times[warmup:],
        **{k: v[warmup:] for k, v in got.items()},
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "peak_mem_mib": peak,
        "peak_mem": {**memory, "scope": "the deployed arm, read right after the timed pages"},
        **conditions.fields(),
        "cpu_busy_percent_after": cpu_busy_percent(),
        "decides": "nothing while a phone can be used (selection rule)",
    }
    _write_new(out, doc)
    print(
        f"{args.arm} run {args.run}: mean {doc['mean_ms']:.1f} ms, p95 {doc['p95_ms']:.1f} ms; "
        f"peak {peak} MiB; written to {out.as_posix()}"
    )
    return 0


def _read_eval(arm: str, name: str) -> dict[str, Any]:
    path = _report_dir(arm) / name
    if not path.is_file():
        raise SystemExit(f"{path.as_posix()}: run eval for {arm} first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("smoke") or doc.get("arm") != arm or doc.get("experiment") != EXPERIMENT:
        raise SystemExit(f"{path.as_posix()} is not a full {EXPERIMENT} run of {arm}")
    if doc.get("batching") != BATCHING[arm]:
        raise SystemExit(f"{path.as_posix()} was not read under {arm}'s batching rule")
    return doc


def _md(doc: dict[str, Any]) -> str:
    rows = [
        f"# {EXPERIMENT}: accuracy part of the phone candidate's selection",
        "",
        "Generated by `scripts/phase22_candidate.py accuracy`; do not edit by hand. Strict track, "
        "laptop, against `fused`; each verdict must be `met` (research/hypotheses.md, H11).",
        "",
        "| Arm | Margin | Line CER | Page CER | E2E F1 | Goes to the phone |",
        "|---|---|---|---|---|---|",
    ]
    for arm, res in doc["arms"].items():
        cells = []
        for name in ("line_cer", "page_cer", "e2e_f1"):
            t = res["tests"][name]
            iv = t["interval"]
            cells.append(
                f"{iv['value']:+.5f} [{iv['low']:+.5f}, {iv['high']:+.5f}], {t['verdict']}"
            )
        rows.append(
            f"| {arm} | {res['margin']} | "
            + " | ".join(cells)
            + f" | {'yes' if res['met'] else 'no'} |"
        )
    rows += [
        "",
        "Measured values (strict):",
        "",
        "| Arm | Line CER | Page CER | E2E F1 |",
        "|---|---|---|---|",
    ]
    for arm, m in doc["measured"].items():
        rows.append(
            f"| {arm} | {100 * m['line_cer']:.3f}% | {100 * m['page_cer']:.3f}% "
            f"| {m['e2e_f1']:.4f} |"
        )
    rows += ["", f"To the phone: {', '.join(doc['to_phone'])}."]
    return "\n".join(rows) + "\n"


def cmd_accuracy(args: argparse.Namespace) -> int:
    out = REPORTS / EXPERIMENT / "accuracy.json"
    _refuse_existing(out, out.with_suffix(".md"))
    lines = {a: _read_eval(a, "lines.json") for a in ARMS}
    pages = {a: _read_eval(a, "pages.json") for a in ARMS}
    shas = {a: lines[a]["build"]["recognizer_sha256"] for a in ARMS}
    for a in ARMS:
        if pages[a]["build"]["recognizer_sha256"] != shas[a]:
            raise SystemExit(f"{a}: the lines and pages runs used different recognizers")
    arms = {
        a: accuracy_tests(a, lines[a], pages[a], lines[DEFAULT_ARM], pages[DEFAULT_ARM])
        for a in ARMS
        if a != DEFAULT_ARM
    }
    doc = {
        "note": "Generated by scripts/phase22_candidate.py accuracy; do not edit by hand.",
        "experiment": EXPERIMENT,
        "version": VERSION_LABEL,
        "rule": "research/hypotheses.md, H11, selection rule: accuracy",
        "reports": {
            a: {s: (_report_dir(a) / f"{s}.json").as_posix() for s in ("lines", "pages")}
            for a in ARMS
        },
        "recognizer_sha256": shas,
        "measured": {
            a: {
                "line_cer": lines[a]["cer"]["strict"],
                "page_cer": pages[a]["page_cer"],
                "e2e_f1": pages[a]["e2e_f1"],
            }
            for a in ARMS
        },
        "arms": arms,
        "to_phone": [DEFAULT_ARM] + [a for a, r in arms.items() if r["met"]],
        "date": datetime.now(UTC).date().isoformat(),
        **git_state(),
    }
    _write_new(out, doc)
    write_text_atomic(out.with_suffix(".md"), _md(doc))
    print(f"to the phone: {', '.join(doc['to_phone'])}; written to {out.as_posix()}")
    return 0


def read_profiles(threads: int = 10) -> dict[str, list[dict[str, Any]]]:
    """The laptop sitting's six profiles, ``{arm: [run 1, run 2]}``, checked to be full runs of
    the committed code on the same pages, made in the sitting's mirrored order."""
    out: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        runs = []
        for run in (1, 2):
            path = _report_dir(arm) / f"profile_pages_t{threads}_run{run}.json"
            if not path.is_file():
                raise SystemExit(f"{path.as_posix()}: run profile for {arm} run {run} first")
            doc = json.loads(path.read_text(encoding="utf-8"))
            if (
                doc.get("smoke")
                or doc.get("arm") != arm
                or doc.get("run") != run
                or doc.get("experiment") != EXPERIMENT
                or doc.get("git_dirty")
            ):
                raise SystemExit(f"{path.as_posix()} is not a full, clean {EXPERIMENT} run")
            runs.append(doc)
        out[arm] = runs
    sizes = out[DEFAULT_ARM][0]["input_sizes"]
    for arm, runs in out.items():
        for doc in runs:
            if doc["input_sizes"] != sizes:
                raise SystemExit(f"{arm} run {doc['run']} timed other pages than fused run 1")
    starts = [out[a][r - 1]["started_utc"] for a, r in sitting_order()]
    if starts != sorted(starts):
        raise SystemExit("the profiles were not made in the sitting's mirrored order")
    return out


def _laptop_md(doc: dict[str, Any]) -> str:
    rows = [
        f"# {EXPERIMENT}: the arms on the laptop CPU (information)",
        "",
        "Generated by `scripts/phase22_candidate.py laptop`; do not edit by hand. "
        f"{doc['decides']}. Page time is `GlypharisOCR.predict` per page, {doc['threads']} "
        f"threads, {doc['pages']} timed pages per run, mirrored order in one sitting.",
        "",
        "| Arm | Mean ms, run 1 | Mean ms, run 2 | Peak RAM MiB, larger run | "
        "Speed test against fused | Memory not above fused |",
        "|---|---|---|---|---|---|",
    ]
    for arm, m in doc["arms"].items():
        speed = m.get("speed")
        cell = "reference"
        if speed is not None:
            iv = speed["tests"]["both_runs"]["interval"]
            cell = (
                f"{iv['value']:+.1f} [{iv['low']:+.1f}, {iv['high']:+.1f}] ms, "
                f"{'met' if speed['met'] else 'not met'}"
            )
        mem = m.get("memory")
        mem_cell = "reference" if mem is None else ("yes" if mem["met"] else "no")
        rows.append(
            f"| {arm} | {m['mean_ms'][0]:.1f} | {m['mean_ms'][1]:.1f} | {m['peak_mib']:.1f} "
            f"| {cell} | {mem_cell} |"
        )
    rows += [
        "",
        "The speed test is `mean_ci(t_a - 0.9 * t_f, 'pages')` on the mean of the two runs; "
        "`met` needs it and each run alone below 0 (research/hypotheses.md, H11).",
    ]
    return "\n".join(rows) + "\n"


def cmd_laptop(args: argparse.Namespace) -> int:
    out = REPORTS / EXPERIMENT / "laptop.json"
    _refuse_existing(out, out.with_suffix(".md"))
    profiles = read_profiles()
    fused = profiles[DEFAULT_ARM]
    arms: dict[str, Any] = {}
    for arm, runs in profiles.items():
        entry: dict[str, Any] = {
            "mean_ms": [r["mean_ms"] for r in runs],
            "p95_ms": [r["p95_ms"] for r in runs],
            "peak_mib": max(r["peak_mem_mib"] for r in runs),
            "peak_mib_runs": [r["peak_mem_mib"] for r in runs],
            "recognizer_sha256": runs[0]["build"]["recognizer_sha256"],
            "reports": [
                (_report_dir(arm) / f"profile_pages_t10_run{r['run']}.json").as_posix()
                for r in runs
            ],
        }
        if arm != DEFAULT_ARM:
            entry["speed"] = speed_tests(
                [r["times_ms"] for r in runs], [r["times_ms"] for r in fused]
            )
            entry["memory"] = memory_test(
                [r["peak_mem_mib"] for r in runs], [r["peak_mem_mib"] for r in fused]
            )
        arms[arm] = entry
    doc = {
        "note": "Generated by scripts/phase22_candidate.py laptop; do not edit by hand.",
        "experiment": EXPERIMENT,
        "version": VERSION_LABEL,
        "decides": "Information only: the laptop's times decide nothing while a phone can be used",
        "threads": 10,
        "pages": len(fused[0]["times_ms"]),
        "order": [f"{a} run {r}" for a, r in sitting_order()],
        "arms": arms,
        "date": datetime.now(UTC).date().isoformat(),
        **git_state(),
    }
    _write_new(out, doc)
    write_text_atomic(out.with_suffix(".md"), _laptop_md(doc))
    print(f"written to {out.as_posix()} and {out.with_suffix('.md').as_posix()}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the arms, their batching and the laptop sitting's order")
    sub.add_parser("accuracy", help="the accuracy verdicts of each arm against fused")
    sub.add_parser("laptop", help="the laptop sitting's times and peaks (information)")
    for name in ("build", "eval", "profile"):
        p = sub.add_parser(name)
        p.add_argument(
            "--bundle", type=Path, required=True, help="the GW-EFF-006 v1 fp32 ONNX bundle"
        )
        p.add_argument(THREADS_FLAG, type=int, required=True)
        p.add_argument("--work-root", type=Path, default=WORK_ROOT)
        p.add_argument("--record", type=Path, default=FREEZE_RECORD)
        p.add_argument("--data-root", type=Path, default=Path("data"))
        if name == "build":
            continue
        p.add_argument("--arm", required=True, choices=list(ARMS))
        p.add_argument("--out", type=Path, default=None)
        if name == "eval":
            p.add_argument("--set", required=True, choices=["pages", "lines"])
            p.add_argument("--limit", type=int, default=None, help="smoke test: first N items")
        else:
            p.add_argument("--run", type=int, required=True, choices=[1, 2])
            p.add_argument(
                "--smoke", type=int, default=None, help="smoke test: 1 warm-up and N timed pages"
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in ("list", "accuracy", "laptop"):
        return {"list": cmd_list, "accuracy": cmd_accuracy, "laptop": cmd_laptop}[args.command](
            args
        )
    if args.threads < 1:
        raise SystemExit("--threads must be a positive integer")
    for flag in ("limit", "smoke"):
        value = getattr(args, flag, None)
        if value is not None and value < 1:
            raise SystemExit(f"--{flag} must be a positive integer")
    return {"build": cmd_build, "eval": cmd_eval, "profile": cmd_profile}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
