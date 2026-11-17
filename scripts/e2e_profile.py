"""CPU page time of the GLYPHARIS pipeline, measured like the baselines (GW-EFF-002, GW-EFF-005).

    python scripts/e2e_profile.py --frozen --threads 10
    python scripts/e2e_profile.py --config configs/det/gw-det-001.yaml \
        --det-checkpoint checkpoints/GW-DET-001/best.pt --rec-run runs/GW-REC-024 \
        [--set det.post... --op ...] --out reports/efficiency/GW-EFF-002.json
    python scripts/e2e_profile.py --config configs/det/gw-det-001.yaml \
        --bundle runs/release/glypharis-0.1.0 --backend onnx --experiment GW-EFF-004 \
        --out reports/efficiency/GW-EFF-004.json

Same protocol and pages as `scripts/run_baselines.py --profile` for the baselines' page profiles:
the seeded Phase 5 HierText validation sample, one page at a time, 10 warm-up pages then 100 timed.
A timing covers a decoded page to its ordered lines: RGB conversion, detection, line crops,
recognition and reading order, on the CPU. Peak memory is this process's. Run on an otherwise idle
machine.

``--frozen`` (Phase 14, GW-EFF-005, H1′'s latency half) builds the pipeline from the freeze record
with `glypharis.pipeline.frozen.load_frozen("cpu")`, so every model setting is the record's and
nothing comes from the command line. ``--threads N`` is required there: it puts N into the OpenMP
and MKL variables (`THREAD_ENV`, the ones `run_baselines` records) before numpy and torch are
imported, and sets torch's intra-op and inter-op threads; the counts torch reports are asserted
before and after the timed pages and recorded, with ``settings["cpu_threads"]`` named as in the
baselines' settings. The code must be at the ``phase14-eval`` tag (`pinned_code_check`, before the
model loads). The result goes to reports/external/GW-EFF-005/profile_glypharis_<N>t.json, which is
never overwritten: the first sitting that is not void is the one of record. It holds aggregates, the
per-page times and input sizes, and when the sitting ran (``started_utc`` just before the first
warm-up page, ``ended_utc`` after the last timed page, as the baselines' profiles record them, so
the scorer can check that the baseline ran first in the same sitting); the pages are validation
pages and no test data is involved.

Without ``--frozen`` the earlier experiments' command line works as before: PyTorch fp32 from the
runs with default threads, or a model bundle on either backend (``--bundle``, Phase 17), where ONNX
Runtime is the build ADR-0011 is judged on.
"""

# ruff: noqa: E402  (the thread variables are set before numpy and torch are imported)

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

# The same variables as run_baselines.THREAD_ENV (a test keeps them equal). They are listed here
# because importing run_baselines would load numpy, and with it an OpenMP runtime, too early.
THREAD_ENV = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS")
THREADS_FLAG = "--threads"


def requested_threads(argv: Sequence[str]) -> int | None:
    """The value of ``--threads`` in ``argv`` (``--threads N`` or ``--threads=N``) when it is a
    positive integer; argparse reports anything malformed later."""
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


def set_thread_env(n: int) -> dict[str, str]:
    """Put ``n`` into every `THREAD_ENV` variable; returns what was set."""
    for key in THREAD_ENV:
        os.environ[key] = str(n)
    return {key: os.environ[key] for key in THREAD_ENV}


if __name__ == "__main__":
    _n = requested_threads(sys.argv[1:])
    if _n is not None:
        set_thread_env(_n)
# Whether the variables were in place before torch loaded: true for a command-line run with
# --threads, false when a process that already holds torch imports this module (the tests).
_TORCH_PRELOADED = "torch" in sys.modules
_ENV_AT_IMPORT = {key: os.environ.get(key) for key in THREAD_ENV}

import argparse
import contextlib
import dataclasses
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_baselines import (
    PROFILE_SAMPLES,
    WARMUP,
    cpu_busy_percent,
    e2e_samples,
    git_state,
    peak_working_set,
    run_conditions,
)

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.data.pages import open_page_image
from glypharis.eval.sets import EVAL_SETS, require_phase5_hashes
from glypharis.export.bundle import load_bundle
from glypharis.models.det.detector import TextLineDetector
from glypharis.pipeline.frozen import load_frozen, pinned_code_check
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer

EVAL_SET = "hiertext_val"
PAGE_MEAN_MS, PAGE_P95_MS, PAGE_RAM_MIB = 1500.0, 2700.0, 1836.0  # ADR-0011
FROZEN_EXPERIMENT = "GW-EFF-005"
FROZEN_OUT = Path("reports/external") / FROZEN_EXPERIMENT
FROZEN_FLAG = "--frozen"


def frozen_out_path(threads: int) -> Path:
    """Where a frozen profile at ``threads`` threads goes (repo-relative)."""
    return FROZEN_OUT / f"profile_glypharis_{threads}t.json"


def set_torch_threads(n: int) -> dict[str, int]:
    """Set torch's intra-op and inter-op threads to ``n`` and return what torch reports.

    The inter-op pool can be sized only once per process and only before it starts. If that is no
    longer possible and it holds another size, the run is refused rather than recorded at a thread
    count it did not have.
    """
    torch.set_num_threads(n)
    with contextlib.suppress(RuntimeError):  # already fixed in this process; checked below
        torch.set_num_interop_threads(n)
    got = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    if got != {"intra_op": n, "inter_op": n}:
        raise SystemExit(f"asked for {n} threads, torch runs {got['intra_op']} intra-op and "
                         f"{got['inter_op']} inter-op; run in a fresh process")  # fmt: skip
    return got


def check_torch_threads(n: int) -> None:
    """The thread counts are still ``n``: nothing reset them since `set_torch_threads`."""
    got = (torch.get_num_threads(), torch.get_num_interop_threads())
    if got != (n, n):
        raise SystemExit(f"thread counts changed during the run: {got}, expected ({n}, {n})")


def _time_pages(
    ocr: GlypharisOCR, root: Path, *, validation_only: bool = False
) -> tuple[list[float], list[dict[str, float]], list[int], list[list[int]]]:
    """Run `WARMUP` + `PROFILE_SAMPLES` pages of `EVAL_SET` one at a time. Returns per-page times
    (ms), stage times (ms), line counts and input sizes, warm-up pages first."""
    times: list[float] = []
    parts: list[dict[str, float]] = []
    n_lines: list[int] = []
    sizes: list[list[int]] = []
    split = EVAL_SETS[EVAL_SET]["split"]
    for _, page in zip(range(WARMUP + PROFILE_SAMPLES), e2e_samples(root, EVAL_SET), strict=False):
        if validation_only and page.split != split:
            raise SystemExit(f"page {page.doc_id} is from the {page.split} split; "
                             f"{EVAL_SET} profiles use {split} pages only")  # fmt: skip
        img = open_page_image(page, root)
        t0 = time.perf_counter()
        res = ocr.predict(np.asarray(img.convert("RGB")))
        times.append(round((time.perf_counter() - t0) * 1000, 3))
        parts.append({k: round(v * 1000, 3) for k, v in res.seconds.items()})
        n_lines.append(len(res.lines))
        sizes.append(list(img.size))
    if len(times) < WARMUP + PROFILE_SAMPLES:
        raise SystemExit(f"{EVAL_SET} gave {len(times)} pages; need {WARMUP + PROFILE_SAMPLES}")
    return times, parts, n_lines, sizes


def _aggregates(
    times: list[float], parts: list[dict[str, float]], n_lines: list[int]
) -> dict[str, float]:
    t = np.asarray(times)
    return {
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_detect_ms": float(np.mean([p["detect"] for p in parts])),
        "mean_recognise_ms": float(np.mean([p["recognise"] for p in parts])),
        "mean_lines": float(np.mean(n_lines)),
    }


def _write(path: Path, res: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")


def main_frozen(argv: list[str]) -> int:
    """GW-EFF-005: the frozen pipeline at ``--threads`` threads (see the module docstring)."""
    parser = argparse.ArgumentParser(description="CPU page profile of the frozen GLYPHARIS v1")
    parser.add_argument(FROZEN_FLAG, action="store_true", help="build from the freeze record")
    parser.add_argument(
        THREADS_FLAG, type=int, required=True, help="torch threads and the OpenMP/MKL variables"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="run although the code is not at the tag (tests only; the result "
                             "records why the check failed)")  # fmt: skip
    args = parser.parse_args(argv)
    n = args.threads
    if n < 1:
        parser.error("--threads must be a positive integer")
    out = frozen_out_path(n)
    if out.exists():
        # A void sitting is moved aside by hand and recorded in research/failures.md.
        raise SystemExit(f"{out.as_posix()} exists: move it aside first")
    require_phase5_hashes(args.data_root)  # Phase 14 plan, item 4: before any Phase 14 run
    import psutil

    env_before_import = not _TORCH_PRELOADED and all(
        _ENV_AT_IMPORT[k] == str(n) for k in THREAD_ENV
    )
    thread_env = set_thread_env(n)  # already in place for a command-line run
    threads = set_torch_threads(n)
    try:
        pinned: dict[str, Any] = pinned_code_check()
    except RuntimeError as exc:
        if not args.allow_unpinned:
            raise SystemExit(f"{exc}; {FROZEN_EXPERIMENT} runs at the tag") from exc
        pinned = {"error": str(exc)}
    torch.set_grad_enabled(False)
    proc = psutil.Process()
    conditions = run_conditions()  # before the model loads, so the CPU check sees an idle machine
    git = git_state()
    frozen = load_frozen("cpu")
    check_torch_threads(n)
    started = datetime.now(UTC).isoformat(timespec="seconds")  # just before the first warm-up page
    times, parts, n_lines, sizes = _time_pages(frozen.ocr, args.data_root, validation_only=True)
    ended = datetime.now(UTC).isoformat(timespec="seconds")  # after the last timed page
    check_torch_threads(n)
    timed = parts[WARMUP:]
    settings = {
        **frozen.settings(),
        "device": "cpu",
        "precision": "fp32",
        "runtime": f"torch {torch.__version__} (CPU)",
        "cpu_threads": n,  # the baselines' key, so one assertion reads both profiles
        "intra_op_threads": threads["intra_op"],
        "inter_op_threads": threads["inter_op"],
    }
    res: dict[str, Any] = {
        "system": "glypharis",
        "task": "e2e",
        "eval_set": EVAL_SET,
        "experiment": FROZEN_EXPERIMENT,
        "version": frozen.provenance.get("name"),
        "settings": settings,
        "threads": {
            "requested": n,
            **threads,
            "env": thread_env,
            "env_before_torch_import": env_before_import,
        },
        "batch": 1,
        "started_utc": started,
        "ended_utc": ended,
        "warmup": WARMUP,
        "warmup_times_ms": times[:WARMUP],
        "times_ms": times[WARMUP:],
        "parts_ms": timed,
        "lines": n_lines[WARMUP:],
        "input_sizes": sizes[WARMUP:],
        **_aggregates(times[WARMUP:], timed, n_lines[WARMUP:]),
        "peak_mem_mib": round(peak_working_set(proc) / 2**20, 1),
        "provenance": frozen.provenance,
        "pinned": pinned,
        "conditions": conditions,
        "cpu_busy_percent_after": cpu_busy_percent(),
        **git,
    }
    _write(out, res)
    print(
        f"{FROZEN_EXPERIMENT} at {n} threads: page mean {res['mean_ms']:.0f} ms, p95 "
        f"{res['p95_ms']:.0f} ms ({res['mean_lines']:.1f} lines); peak {res['peak_mem_mib']} MiB; "
        f"written to {out.as_posix()}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if FROZEN_FLAG in argv:
        return main_frozen(argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--det-checkpoint", default=None)
    parser.add_argument("--op", default=None)
    parser.add_argument("--rec-run", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--experiment", default="GW-EFF-002")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    import psutil

    torch.set_grad_enabled(False)
    cfg = load_config(args.config, overrides_from_args(args))
    assert cfg.det is not None
    proc = psutil.Process()
    conditions = run_conditions()
    if args.bundle is not None:
        ocr = load_bundle(args.bundle, args.backend)
        runtime = f"{args.backend} bundle {args.bundle.name}"
    else:
        if args.det_checkpoint is None or args.rec_run is None:
            parser.error("give --bundle, or --det-checkpoint and --rec-run")
        det = TextLineDetector.from_checkpoint(args.det_checkpoint, "cpu")
        det_spec = dataclasses.replace(
            det.spec,
            infer=cfg.det.infer,
            post=cfg.det.post,
            operating_point=args.op or det.spec.operating_point,
        )
        ocr = GlypharisOCR(
            TextLineDetector(det.backend, det_spec), TextRecognizer.from_run(args.rec_run)
        )
        runtime = f"torch {torch.__version__} (CPU)"
    spec = ocr.detector.spec
    root = Path(cfg.paths.data_root)
    times, parts, n_lines, sizes = _time_pages(ocr, root)
    t = np.asarray(times[WARMUP:])
    timed = parts[WARMUP:]
    res: dict[str, Any] = {
        "experiment": args.experiment,
        "long_side": spec.infer.long_side,
        "operating_point": spec.operating_point,
        "post": dataclasses.asdict(spec.post),
        "rec_run": args.rec_run.as_posix() if args.rec_run else None,
        "bundle": args.bundle.name if args.bundle else None,
        "eval_set": EVAL_SET,
        "warmup": WARMUP,
        "precision": "fp32",
        "runtime": runtime,
        "torch_threads": torch.get_num_threads(),
        "times_ms": times[WARMUP:],
        "parts_ms": timed,
        "lines": n_lines[WARMUP:],
        "input_sizes": sizes[WARMUP:],
        **_aggregates(times[WARMUP:], timed, n_lines[WARMUP:]),
        "peak_mem_mib": round(peak_working_set(proc) / 2**20, 1),
        "meets_target": bool(
            t.mean() <= PAGE_MEAN_MS
            and np.percentile(t, 95) <= PAGE_P95_MS
            and peak_working_set(proc) / 2**20 < PAGE_RAM_MIB
        ),
        "conditions": conditions,
        "cpu_busy_percent_after": cpu_busy_percent(),
        **git_state(),
    }
    _write(args.out, res)
    print(
        f"page mean {res['mean_ms']:.0f} ms, p95 {res['p95_ms']:.0f} ms (detect "
        f"{res['mean_detect_ms']:.0f}, recognise {res['mean_recognise_ms']:.0f}, "
        f"{res['mean_lines']:.1f} lines); peak {res['peak_mem_mib']} MiB; "
        f"meets ADR-0011: {res['meets_target']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
