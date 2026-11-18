"""Phase 16 variants of v1: build, accuracy on validation, latency and peak RAM (GW-EFF-008, 009).

    python scripts/phase16_variants.py list
    python scripts/phase16_variants.py build --variant int8-recognizer --bundle DIR --threads 10
    python scripts/phase16_variants.py eval --variant int8-recognizer --bundle DIR --set lines \
        --threads 10
    python scripts/phase16_variants.py profile --variant int8-recognizer --bundle DIR \
        --set lines --threads 10 --experiment GW-EFF-008 --run 1

A variant changes exactly one setting of the v1 fp32 ONNX build (the GW-EFF-006 bundle, --bundle)
and is a new version, not v1; `glypharis.efficiency.variants` builds it and asserts every other
setting, and every output carries ``"version": "variant (new version, not v1)"``. ``v1-torch``
and ``v1-onnx`` are v1 itself, measured by the same code (``"version": "v1"``).

- ``build`` writes the variant's model files and ``variant.json`` to
  ``data/processed/phase16/variants/<variant>/`` (``--work``) once; ``--checkpoint`` gives the
  training checkpoint for ``det-GW-DET-003`` and ``rec-GW-REC-*`` by path.
- ``eval --set pages``: page CER (bag of lines) and E2E F1 on the 400 Phase 5 validation pages, on
  the CPU, scored by `glypharis.eval.page_metrics.evaluate_page` (`e2e_eval.score`, both tracks),
  with per-page counts for paired intervals. ``eval --set lines``: line CER of greedy decoding on
  the 5,000 validation lines seeded 20261028 from the recognizer's validation pack (the export
  check's sample), each line read on its own (batch 1, as the line latency is timed: dynamic INT8
  scales each input tensor as a whole, so a line's INT8 logits would depend on its batch-mates),
  with per-line errors and reference lengths for `paired_ratio_diff_ci`, and, for an ONNX
  recognizer made from a PyTorch model, the export check's logit and text rules against that
  model on the same lines, read the same way.
- ``profile --set pages``: 10 warm-up and 100 timed Phase 5 profile pages one at a time, as
  GW-EFF-005 times them; ``--set lines``: 10 warm-up and 100 timed ``rec_samples`` lines at batch
  1 (`TextRecognizer.read` of one crop: grey, resize, scale, pad, network, greedy decoding). Per
  item ``times_ms``, input sizes, peak RAM, the thread pools and ONNX Runtime session options
  (asserted before and after the timed items), and the machine and GPU conditions at start and
  end. One process per run; the lead runs the arms of each sitting in its registered order
  (``list`` prints them: run 1 of each arm, then run 2 in reverse) and names each run ``--run 1``
  or ``--run 2``. ``--experiment`` names the sitting and is part of the file name, so H7′'s
  (GW-EFF-008), H6′'s (GW-EFF-011) and the Pareto points' (GW-EFF-009) runs are separate files.
  Every build but ``v1-torch`` is timed as a deployed build loads it (`deployed_variant`: the
  bundle and the variant's own files, no training checkpoint), so the peak RAM, read right after
  the timed items, is that build's; the full chain of checks (`load_variant`) runs after that,
  and the profile is written only if both pipelines hold the same settings (`same_build`).
  ``v1-torch`` is built from the freeze record (`load_frozen`), as `scripts/e2e_profile.py --v1
  --backend torch` builds it.

``--threads N`` is required for build, eval and profile: it is applied before numpy and torch are
imported (`glypharis.efficiency.threads.set_threads`), and every ONNX session gets N intra-op
threads. The v1 code must equal the ``v1-code`` tag's (`assert_v1_code`; ``phase14-eval`` for the
Phase 16 runs, as their ``v1_code`` records show) before anything loads. Nothing is overwritten.
``--limit`` (eval) and ``--smoke`` (profile) cut the item counts for a smoke test; they need
``--out`` and the output is marked ``"smoke": true`` so no report uses it. Validation data only: the
page and line sets are fixed by the freeze record's configs, not by the command line.
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


# Set by a command-line run before numpy and torch load; None when a test imports this module.
THREADS: dict[str, Any] | None = None
if __name__ == "__main__":
    _n = requested_threads(sys.argv[1:])
    if _n is not None:
        THREADS = set_threads(_n)

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from e2e_eval import metrics, score
from export_check import LINES, LOGIT_TOL, NEAR_TIE, NEAR_TIE_SHARE, SEED
from run_baselines import (
    PROFILE_SAMPLES,
    WARMUP,
    cpu_busy_percent,
    e2e_samples,
    git_state,
    peak_working_set,
    rec_samples,
)

from glypharis.config.loader import load_config
from glypharis.data.det_dataset import PageIndex, read_gt_pages
from glypharis.data.det_geometry import decode_page
from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackReader
from glypharis.data.pages import open_page_image
from glypharis.efficiency.conditions import ProfileConditions
from glypharis.efficiency.files import write_json_atomic, write_text_atomic
from glypharis.efficiency.onnx_build import check_sessions
from glypharis.efficiency.threads import check_threads
from glypharis.efficiency.v1 import assert_v1_code
from glypharis.efficiency.variants import (
    SITTINGS,
    VARIANTS,
    WORK_ROOT,
    Build,
    build_frozen,
    deployed_variant,
    load_variant,
    prepare_variant,
    profile_file,
    report_record,
    same_build,
    sitting_order,
    variant_spec,
)
from glypharis.eval.det_matching import baseline_doc_ids
from glypharis.eval.page_metrics import PageResult, PredLine
from glypharis.eval.text_metrics import score_pair
from glypharis.pipeline.frozen import FREEZE_RECORD, Frozen
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer

EVAL_SET = "hiertext_val"
REPORTS = Path("reports/efficiency/phase16/variants")
EXPERIMENTS = tuple(SITTINGS)  # GW-EFF-008, 011 and 009
TRACKS = ("strict", "normalised")
LINE_BATCH = 1  # each accuracy line is read on its own, as the line latency is timed


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _refuse_existing(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise SystemExit(f"{path.as_posix()} exists: move it aside first")


def _write_new(path: Path, doc: dict[str, Any]) -> None:
    _refuse_existing(path)
    write_json_atomic(path, doc)


def eval_paths(record: Path, data_root: Path) -> tuple[Path, Path]:
    """The 400-page file and the 5,000-line pack, from the configs the freeze record names: the
    detector's validation pages and the recognizer's validation pack."""
    doc = yaml.safe_load(record.read_text(encoding="utf-8"))
    det = load_config(doc["detector"]["config"])
    rec = load_config(doc["recognizer"]["config"])
    if det.det is None:
        raise SystemExit(f"{doc['detector']['config']} has no detector section")
    return data_root / det.det.data.val_pages[0], data_root / rec.data.val[0]


# --------------------------------------------------------------------------------- accuracy


def line_sample(n_lines: int) -> np.ndarray:
    """The export check's sample: `LINES` indices drawn with `SEED`, sorted."""
    return np.sort(np.random.default_rng(SEED).choice(n_lines, min(LINES, n_lines), replace=False))


def _near_tie(la: torch.Tensor, lb: torch.Tensor) -> tuple[int, float]:
    """The first frame whose top class differs, and the reference's top-2 gap there."""
    frame = int(torch.nonzero(la.argmax(-1) != lb.argmax(-1))[0])
    top2 = la[frame].topk(2).values
    return frame, float(top2[0] - top2[1])


def eval_lines(
    build: Build, pack: Path, limit: int | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Greedy line CER on the seeded validation lines (both tracks, per line), and the export
    check against ``build.torch_reference`` when there is one, every line read on its own
    (`LINE_BATCH`) by both. Returns the result and the texts."""
    reader = LinePackReader(pack)
    idx = line_sample(len(reader))
    if limit is not None:
        idx = idx[:limit]
    rec = build.ocr.recognizer
    ref_model = build.torch_reference
    per: dict[str, dict[str, list[int]]] = {t: {"errors": [], "ref_chars": []} for t in TRACKS}
    keys: list[str] = []
    texts: list[dict[str, Any]] = []
    worst, same, near, other = 0.0, 0, 0, []
    for start in range(0, len(idx), LINE_BATCH):
        chunk = [int(i) for i in idx[start : start + LINE_BATCH]]
        for i in chunk:
            if reader.records[i].split != "validation":
                raise SystemExit(f"line {reader.records[i].key} is not a validation line")
        crops = [reader.image(i) for i in chunk]
        logits = rec.logits(crops)
        ref_logits = ref_model.logits(crops) if ref_model is not None else [None] * len(chunk)
        for i, lg, lr in zip(chunk, logits, ref_logits, strict=True):
            r = reader.records[i]
            text = rec.charset.ctc_greedy_decode(lg.argmax(-1).tolist())
            keys.append(r.key)
            texts.append({"key": r.key, "text": text})
            for track in TRACKS:
                s = score_pair(text, r.label, track)  # type: ignore[arg-type]
                per[track]["errors"].append(s.char_errors)
                per[track]["ref_chars"].append(s.ref_chars)
            if lr is None:
                continue
            assert ref_model is not None
            worst = max(worst, float((lr - lg).abs().max()))
            ref_text = ref_model.charset.ctc_greedy_decode(lr.argmax(-1).tolist())
            if ref_text == text:
                same += 1
                continue
            frame, gap = _near_tie(lr, lg)
            if gap < NEAR_TIE:
                near += 1
            else:
                other.append({"key": r.key, "frame": frame, "gap": gap})
    n = len(keys)
    cer = {t: sum(per[t]["errors"]) / max(1, sum(per[t]["ref_chars"])) for t in TRACKS}
    check = None
    if ref_model is not None:
        check = {
            "reference": "the PyTorch model this ONNX recognizer was made from",
            "lines": n,
            "logit_max_abs_diff": worst,
            "identical_text": same,
            "near_tie_differences": near,
            "other_differences": other,
            "tolerances": {
                "logits": LOGIT_TOL,
                "near_tie": NEAR_TIE,
                "near_tie_share": NEAR_TIE_SHARE,
            },
            "pass": worst <= LOGIT_TOL and not other and near <= NEAR_TIE_SHARE * n,
        }
    result = {
        "lines": n,
        "batch": LINE_BATCH,
        "cer": cer,
        "per_line": {"keys": keys, "index": [int(i) for i in idx], **per},
        "torch_check": check,
    }
    return result, texts


def _counts(rs: list[PageResult]) -> dict[str, list[int]]:
    return {
        "bag_errors": [r.bag_errors for r in rs],
        "bag_ref_chars": [r.bag_ref_chars for r in rs],
        "e2e_tp": [r.e2e_tp for r in rs],
        "e2e_pred": [r.e2e_pred for r in rs],
        "e2e_gt": [r.e2e_gt for r in rs],
        "det_tp": [r.det[0.5][0] for r in rs],
        "det_fp": [r.det[0.5][1] for r in rs],
        "det_fn": [r.det[0.5][2] for r in rs],
    }


def eval_pages(
    build: Build, data_root: Path, pages_file: Path, limit: int | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Page CER and E2E F1 of the whole pipeline on the Phase 5 validation pages (both tracks,
    per-page counts). Returns the result and the predictions, one record per page."""
    ids = baseline_doc_ids(PageIndex.build(pages_file, "validation", data_root).doc_ids())
    if limit is not None:
        ids = ids[:limit]
    gt = read_gt_pages(pages_file, ids)
    preds: dict[str, list[PredLine]] = {}
    records = []
    with torch.inference_mode():
        for d in ids:
            page = gt[d]
            if page.split != "validation":
                raise SystemExit(f"page {d} is from the {page.split} split")
            rgb = decode_page(data_root / page.image, (page.width, page.height))
            t0 = time.perf_counter()
            res = build.ocr.predict(rgb)
            ms = (time.perf_counter() - t0) * 1000
            preds[d] = [PredLine(ln.polygon, ln.text, ln.confidence) for ln in res.lines]
            records.append(
                {
                    "key": d,
                    "time_ms": round(ms, 1),
                    "lines": [
                        {
                            "polygon": ln.polygon,
                            "text": ln.text,
                            "confidence": ln.confidence,
                            "det_score": ln.det_score,
                        }
                        for ln in res.lines
                    ],
                }
            )
    results = {t: score(preds, gt, ids, t) for t in TRACKS}
    strict = results["strict"]
    bag = sum(r.bag_errors for r in strict) / max(1, sum(r.bag_ref_chars for r in strict))
    tp = sum(r.e2e_tp for r in strict)
    den = sum(r.e2e_pred + r.e2e_gt for r in strict)
    result = {
        "pages": len(ids),
        "page_cer": bag,
        "e2e_f1": 2 * tp / den if den else 0.0,
        "metrics": {t: metrics(results[t]) for t in TRACKS},
        "per_page": {"doc_ids": ids, **{t: _counts(results[t]) for t in TRACKS}},
    }
    return result, records


# ---------------------------------------------------------------------------------- latency


def profile_pages(ocr: GlypharisOCR, data_root: Path, total: int) -> dict[str, Any]:
    """``total`` profile pages one at a time, timed as GW-EFF-005 times them (RGB array to
    ordered lines). Lists include the warm-up pages."""
    times: list[float] = []
    parts: list[dict[str, float]] = []
    n_lines: list[int] = []
    sizes: list[list[int]] = []
    for _, page in zip(range(total), e2e_samples(data_root, EVAL_SET), strict=False):
        if page.split != "validation":
            raise SystemExit(f"page {page.doc_id} is from the {page.split} split")
        img = open_page_image(page, data_root)
        t0 = time.perf_counter()
        res = ocr.predict(np.asarray(img.convert("RGB")))
        times.append(round((time.perf_counter() - t0) * 1000, 3))
        parts.append({k: round(v * 1000, 3) for k, v in res.seconds.items()})
        n_lines.append(len(res.lines))
        sizes.append(list(img.size))
    if len(times) < total:
        raise SystemExit(f"{EVAL_SET} gave {len(times)} pages; need {total}")
    return {"times": times, "parts_ms": parts, "lines": n_lines, "input_sizes": sizes}


def profile_lines(ocr: GlypharisOCR, data_root: Path, total: int) -> dict[str, Any]:
    """``total`` `rec_samples` lines at batch 1. Lists include the warm-up lines."""
    times: list[float] = []
    keys: list[str] = []
    sizes: list[list[int]] = []
    rec: TextRecognizer = ocr.recognizer
    with torch.inference_mode():
        for _, (key, _label, img) in zip(
            range(total), rec_samples(data_root, EVAL_SET), strict=False
        ):
            t0 = time.perf_counter()
            rec.read([img])
            times.append(round((time.perf_counter() - t0) * 1000, 3))
            keys.append(key)
            sizes.append(list(img.size))
    if len(times) < total:
        raise SystemExit(f"{EVAL_SET} gave {len(times)} lines; need {total}")
    return {"times": times, "keys": keys, "input_sizes": sizes}


# --------------------------------------------------------------------------------- commands


def _thread_record(n: int) -> dict[str, Any]:
    """The `set_threads` record of this process: from the import for a command-line run; a
    process that loaded torch first (a test) records that the variables came too late."""
    return THREADS if THREADS is not None else set_threads(n, require_fresh=False)


def _texts_sha(texts: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()


def _common(
    args: argparse.Namespace,
    build: Build,
    v1_code: dict[str, Any],
    threads: dict[str, Any],
    sessions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": args.experiment or variant_spec(args.variant).experiment,
        "variant": build.name,
        "version": build.version,
        "setting": build.record["setting"],
        "changed": build.record["changed"],
        "eval_set": EVAL_SET,
        "device": "cpu",
        "threads": threads,
        "session_options": sessions or check_sessions(build.ocr, args.threads),
        "v1_code": v1_code,
        "build": report_record(build.record),
        **git_state(),
    }


def _out(args: argparse.Namespace, default: str) -> Path:
    if args.out is not None:
        return Path(args.out)
    if getattr(args, "limit", None) is not None or getattr(args, "smoke", None) is not None:
        raise SystemExit("a smoke run (--limit/--smoke) needs --out")
    return REPORTS / args.variant / default


def _frozen(args: argparse.Namespace) -> Frozen:
    return build_frozen(".", args.record)


def _work(args: argparse.Namespace) -> Path:
    return Path(args.work) if args.work is not None else WORK_ROOT / args.variant


def _bundle(args: argparse.Namespace) -> Path | None:
    return None if args.variant == "v1-torch" else args.bundle


def cmd_list(_: argparse.Namespace) -> int:
    for v in VARIANTS.values():
        what = v.setting or "(v1)"
        extra = f"; needs --checkpoint ({v.checkpoint})" if v.checkpoint else ""
        print(f"{v.name:22s} {what:22s} {', '.join(v.measure):12s} {v.about}{extra}")
    print("\nLatency sittings (profile --experiment), runs in this order in one sitting:")
    for experiment, units in SITTINGS.items():
        for unit in units:
            order = ", ".join(f"{a} run {r}" for a, r in sitting_order(experiment, unit))
            print(f"{experiment} {unit}: {order}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    threads = _thread_record(args.threads)
    v1_code = assert_v1_code(".")
    frozen = _frozen(args)
    doc = prepare_variant(
        args.variant, frozen, _bundle(args), _work(args), args.threads, args.checkpoint
    )
    build = load_variant(args.variant, frozen, _bundle(args), _work(args), args.threads)
    check_sessions(build.ocr, args.threads)
    print(
        json.dumps(
            {
                "variant": args.variant,
                "version": build.version,
                "changed": build.record["changed"],
                "files": doc["files"],
                "threads": threads["requested"],
                "v1_code": v1_code["v1_code"],
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    threads = _thread_record(args.threads)
    out = _out(args, f"{args.set}.json")
    # Predictions and texts stay with the model files (git-ignored); a smoke run's go beside it.
    folder = out.parent if args.limit is not None else _work(args)
    items_path = folder / f"{out.stem}_outputs.jsonl"
    _refuse_existing(out, items_path)  # before hours of evaluation, not after
    v1_code = assert_v1_code(".")
    pages_file, pack = eval_paths(args.record, args.data_root)
    frozen = _frozen(args)
    build = load_variant(args.variant, frozen, _bundle(args), _work(args), args.threads)
    check_threads(args.threads)
    started = _utc()
    if args.set == "lines":
        result, items = eval_lines(build, pack, args.limit)
        source = {
            "pack": pack.relative_to(args.data_root).as_posix(),
            "seed": SEED,
            "sample": f"{LINES} lines, the export check's",
            "track": "strict",
            "decoding": "greedy CTC",
        }
    else:
        result, items = eval_pages(build, args.data_root, pages_file, args.limit)
        source = {
            "pages_file": pages_file.relative_to(args.data_root).as_posix(),
            "sample": "the 400 Phase 5 pages (baseline_doc_ids)",
            "track": "strict",
        }
    ended = _utc()
    check_threads(args.threads)
    _refuse_existing(out, items_path)
    write_text_atomic(items_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items))
    doc = {
        "kind": f"phase16-variant-{args.set}",
        **_common(args, build, v1_code, threads),
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
    print(f"{args.variant} ({build.version}): {head}; written to {out.as_posix()}")
    return 0


def check_sitting(variant: str, unit: str, experiment: str) -> None:
    """Refuse a profile that is not an arm of ``experiment``'s sitting for ``unit``."""
    arms = SITTINGS[experiment].get(unit, ())
    if variant not in arms:
        raise SystemExit(f"{experiment} times {unit} of {', '.join(arms) or 'no variant'}, "
                         f"not of {variant}")  # fmt: skip


def cmd_profile(args: argparse.Namespace) -> int:
    import psutil

    threads = _thread_record(args.threads)
    n = args.threads
    unit = "pages" if args.set == "pages" else "lines"
    check_sitting(args.variant, unit, args.experiment)
    out = _out(args, profile_file(unit, n, args.experiment, args.run))
    _refuse_existing(out)
    warmup, timed = (WARMUP, PROFILE_SAMPLES) if args.smoke is None else (1, args.smoke)
    v1_code = assert_v1_code(".")
    torch.set_grad_enabled(False)
    proc = psutil.Process()
    conditions = ProfileConditions()
    conditions.start()  # before the model loads, so the CPU check sees an idle machine
    frozen: Frozen | None = None
    build: Build | None = None
    if args.variant == "v1-torch":
        frozen = _frozen(args)
        build = load_variant(args.variant, frozen, None, _work(args), n)
        ocr = build.ocr
        memory = {"scope": "v1 built from the freeze record (load_frozen), then timed",
                  "loaded": ["load_frozen('cpu')"], "numerics": None}  # fmt: skip
    else:
        # A deployed build: the bundle and the variant's files only, so the peak RAM is its own.
        ocr, memory = deployed_variant(args.variant, args.bundle, _work(args), n)
        memory["scope"] = ("the deployed build (bundle and variant files, no training "
                           "checkpoint), read right after the timed items")  # fmt: skip
    sessions = check_sessions(ocr, n)
    threads_before = check_threads(n)
    started = _utc()
    if args.set == "pages":
        got = profile_pages(ocr, args.data_root, warmup + timed)
    else:
        got = profile_lines(ocr, args.data_root, warmup + timed)
    ended = _utc()
    threads_after = check_threads(n)
    peak = round(peak_working_set(proc) / 2**20, 1)  # before the checks load anything else
    conditions.end()
    if build is None:
        # The full chain (record, PyTorch v1, bundle, variant), then the timed build against it.
        frozen = _frozen(args)
        build = load_variant(args.variant, frozen, _bundle(args), _work(args), n)
        same_build(ocr, build)
    times = got.pop("times")
    t = np.asarray(times[warmup:])
    settings = build.record["settings"]
    doc: dict[str, Any] = {
        "kind": "phase16-variant-profile",
        "system": "glypharis",
        "task": "e2e" if args.set == "pages" else "rec",
        **_common(args, build, v1_code, threads, sessions),
        "set": args.set,
        "run": args.run,
        "smoke": args.smoke is not None,
        "settings": {
            "cpu_threads": n,  # the baselines' key, as Phase 14's profiles record it
            "intra_op_threads": threads_before["intra_op"],
            "inter_op_threads": threads_before["inter_op"],
            "device": "cpu",
            "runtime": {
                "detector": settings["detector.runtime"],
                "recognizer": settings["recognizer.runtime"],
            },
            "detector_long_side": settings["detector.infer"]["long_side"],
            "timing": (
                "GlypharisOCR.predict of the RGB page array, one page at a time"
                if args.set == "pages"
                else "TextRecognizer.read of one crop: grey, resize, scale, pad, network, "
                "greedy decoding"
            ),
        },
        "session_options": sessions,
        "threads_before": threads_before,
        "threads_after": threads_after,
        "batch": 1,
        "warmup": warmup,
        "started_utc": started,
        "ended_utc": ended,
        "warmup_times_ms": times[:warmup],
        "times_ms": times[warmup:],
        **{k: v[warmup:] for k, v in got.items()},
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "peak_mem_mib": peak,
        "peak_mem": memory,
        **conditions.fields(),
        "cpu_busy_percent_after": cpu_busy_percent(),
    }
    if args.set == "pages":
        doc["mean_lines"] = float(np.mean(doc["lines"]))
    _write_new(out, doc)
    print(
        f"{args.variant} ({build.version}) run {args.run}, {unit}: mean {doc['mean_ms']:.1f} ms, "
        f"p95 {doc['p95_ms']:.1f} ms; peak {doc['peak_mem_mib']} MiB; "
        f"written to {out.as_posix()}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="the variants and the setting each one changes")
    for name in ("build", "eval", "profile"):
        p = sub.add_parser(name)
        p.add_argument("--variant", required=True, choices=list(VARIANTS))
        p.add_argument(
            "--bundle", type=Path, default=None, help="the GW-EFF-006 v1 fp32 ONNX bundle"
        )
        p.add_argument(THREADS_FLAG, type=int, required=True)
        p.add_argument(
            "--work",
            type=Path,
            default=None,
            help=f"model files (default {WORK_ROOT.as_posix()}/<variant>)",
        )
        p.add_argument("--record", type=Path, default=FREEZE_RECORD)
        p.add_argument("--data-root", type=Path, default=Path("data"))
        if name == "build":
            p.add_argument("--checkpoint", type=Path, default=None)
            continue
        p.add_argument("--set", required=True, choices=["pages", "lines"])
        p.add_argument(
            "--experiment",
            default=None,
            required=name == "profile",
            choices=list(EXPERIMENTS),
            help="the sitting (profile: part of the file name); eval: default the variant's",
        )
        p.add_argument("--out", type=Path, default=None)
        if name == "eval":
            p.add_argument("--limit", type=int, default=None, help="smoke test: first N items")
        else:
            p.add_argument("--run", type=int, required=True, choices=[1, 2])
            p.add_argument(
                "--smoke", type=int, default=None, help="smoke test: 1 warm-up and N timed items"
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        return cmd_list(args)
    if args.threads < 1:
        raise SystemExit("--threads must be a positive integer")
    if args.variant != "v1-torch" and args.bundle is None:
        raise SystemExit(f"{args.variant} is built on the v1 fp32 ONNX bundle: give --bundle")
    for flag in ("limit", "smoke"):
        value = getattr(args, flag, None)
        if value is not None and value < 1:
            raise SystemExit(f"--{flag} must be a positive integer")
    return {"build": cmd_build, "eval": cmd_eval, "profile": cmd_profile}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
