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

``--v1`` (Phase 16, GW-EFF-007 and GW-EFF-010) profiles the v1 build on the same pages and counts.
The output names are the ones `scripts/phase16_score.py` reads (its `COST` and `H1_GLYPH`):

    python scripts/e2e_profile.py --v1 --backend torch --threads 10 --experiment GW-EFF-007 \
        --out reports/efficiency/phase16/profiles/pages_torch_cpu_10t.json
    python scripts/e2e_profile.py --v1 --backend torch --threads 10 --experiment GW-EFF-007 \
        --device cuda --out reports/efficiency/phase16/profiles/pages_torch_cuda.json
    python scripts/e2e_profile.py --v1 --backend onnx --bundle <v1 bundle> --threads 10 \
        --experiment GW-EFF-007 --out reports/efficiency/phase16/profiles/pages_onnx_cpu_10t.json
    python scripts/e2e_profile.py --v1 --backend onnx --bundle <v1 bundle> --threads default \
        --experiment GW-EFF-007 \
        --out reports/efficiency/phase16/profiles/pages_onnx_cpu_default.json
    python scripts/e2e_profile.py --v1 --backend onnx --bundle <v1 bundle> --threads 10 \
        --experiment GW-EFF-010 \
        --out reports/efficiency/phase16/profiles/pages_onnx_cpu_10t_GW-EFF-010.json

PyTorch is built from the freeze record (`load_frozen`, CPU or CUDA); ONNX from the GW-EFF-006
bundle, both sessions from `glypharis.efficiency.threads.ort_session`, after checking its manifest
hashes and that its settings equal the record's. ``--threads N`` is set before numpy and torch are
imported (`set_threads`) and asserted before and after the timed pages, with every ONNX session's
options; ``--threads default`` (ONNX only) leaves torch, the variables and ONNX Runtime at their
defaults and records what they were. The v1 code paths must equal ``v1-code`` (`assert_v1_code`;
``phase14-eval`` for the Phase 16 runs, as their ``v1_code`` records show) before the model loads.
Run conditions and the GPU's state are recorded at the start and the end. Each page runs through the
same pipeline pieces `GlypharisOCR.predict` calls, in the same order, with one clock read between
them (`staged_predict`): the page time runs from the first reading to the last, as GW-EFF-005 times
a page (from the decoded image: the image file is read and decoded by `open_page_image` before the
clock starts, so its time, recorded per page as ``file_decode_ms``, is not part of the page time;
the first stage, ``to_rgb``, is the RGB array `load_rgb` makes). The nine stages of `STAGES` are
checked against a second clock read around the whole call: they must sum to it within the larger of
1% and 5 ms (asserted). After the timed pages the first timed page is read once more with
`GlypharisOCR.predict`, untimed by the stages, and must give the same lines; both readings' times
are recorded. On CUDA every stage boundary is synchronised, and peak VRAM is recorded from torch
(``max_memory_allocated``), with nvidia-smi's memory at the end for this process where the driver
reports it (Windows' WDDM driver does not) and for the whole GPU (the display and other programs
included). The output is never overwritten.
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


V1_FLAG = "--v1"
# The thread record of a --v1 command-line run, made before numpy and torch load; None otherwise.
_V1_THREADS: dict[str, object] | None = None

if __name__ == "__main__":
    _n = requested_threads(sys.argv[1:])
    if _n is not None and V1_FLAG in sys.argv[1:]:
        from glypharis.efficiency.threads import set_threads

        _V1_THREADS = set_threads(_n)
    elif _n is not None:
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
import yaml
from PIL import Image
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
from glypharis.data.crops import background_grey, cut_line
from glypharis.data.download import sha256_file
from glypharis.data.pages import Page, open_page_image
from glypharis.efficiency.conditions import ProfileConditions, gpu_process_memory_mib, gpu_state
from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.onnx_build import check_sessions, load_bundle_onnx
from glypharis.efficiency.threads import (
    ThreadError,
    check_threads,
    set_threads,
    thread_record,
)
from glypharis.efficiency.v1 import V1CodeError, assert_v1_code
from glypharis.eval.sets import EVAL_SETS, require_phase5_hashes
from glypharis.export.bundle import load_bundle, read_manifest
from glypharis.models.det.detector import TextLineDetector
from glypharis.models.det.postprocess import boxes_from_candidates, find_candidates
from glypharis.models.det.preprocess import prepare_page
from glypharis.pipeline.frozen import FREEZE_RECORD, Frozen, load_frozen, pinned_code_check
from glypharis.pipeline.ocr import GlypharisOCR, OcrLine, OCRResult, load_rgb
from glypharis.pipeline.reading_order import reading_order
from glypharis.pipeline.recognizer import (
    LineReading,
    TextRecognizer,
    _padded,
    line_confidence,
    prepare_crop,
)

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


# --------------------------------------------------------------------- Phase 16: the v1 build

# The page time split, in pipeline order (`staged_predict`).
STAGES = (
    "to_rgb",  # the decoded page to an RGB array (`load_rgb`); the file is decoded before the clock
    "det_preprocess",  # `prepare_page`: long side to the spec's, padded to 32
    "det_network",  # the probability map (the backend's call, with its copies to and from)
    "det_postprocess",  # `find_candidates` and `boxes_from_candidates`
    "crop",  # grey page, background grey and `cut_line` for every box
    "rec_batch",  # `prepare_crop` per line and the padded batch tensors
    "rec_network",  # the recognizer on each batch, and each line's logits back on the CPU
    "rec_decode",  # greedy CTC text and confidence per line
    "reading_order",  # `reading_order` and the result lines
)
STAGE_TOLERANCE = 0.01  # the stages sum to the page time within 1% ...
STAGE_SLACK_MS = 5.0  # ... or 5 ms: on CUDA each stage boundary adds a synchronisation


class _Laps:
    """One clock for a page: each lap adds the time since the previous reading to a stage, and
    `total_ms` runs from the first reading to the last, so the page time and its stages come from
    the same readings. ``sync`` (CUDA) runs before each reading."""

    def __init__(self, sync: Any = None) -> None:
        self.sync = sync
        self.ms = dict.fromkeys(STAGES, 0.0)
        if sync is not None:
            sync()
        self.t = self.t0 = time.perf_counter()

    def __call__(self, stage: str) -> None:
        if self.sync is not None:
            self.sync()
        now = time.perf_counter()
        self.ms[stage] += (now - self.t) * 1000
        self.t = now

    @property
    def total_ms(self) -> float:
        return (self.t - self.t0) * 1000


def _recognise(rec: TextRecognizer, crops: list[Image.Image], laps: _Laps) -> list[LineReading]:
    """`TextRecognizer.read` (through `TextRecognizer.logits`) step by step: the same batches,
    padding, numerics and decoding."""
    with torch.inference_mode():
        xs = [prepare_crop(c, rec.height, rec.max_width) for c in crops]
        widths = [x.shape[1] for x in xs]
        out: list[torch.Tensor | None] = [None] * len(xs)
        for idx in rec._batches(widths):
            width = max(_padded(widths[i]) for i in idx)
            batch = torch.empty(len(idx), 1, rec.height, width)
            for j, i in enumerate(idx):
                x = xs[i]
                batch[j, 0, :, : x.shape[1]] = x
                batch[j, 0, :, x.shape[1] :] = x[:, -1].median()
            w = torch.tensor([widths[i] for i in idx])
            laps("rec_batch")
            with rec._numerics():
                logits = rec.model(batch.to(rec.device), w.to(rec.device))
            lengths = rec.model.frames(w).clamp(max=logits.shape[0])
            for j, i in enumerate(idx):
                out[i] = logits[: int(lengths[j]), j].float().cpu()
            laps("rec_network")
        laps("rec_batch")  # a page without lines: the empty list
        reads = []
        for lg in (o for o in out if o is not None):
            ids = lg.argmax(dim=-1).tolist()
            text = rec.charset.ctc_greedy_decode(ids)
            reads.append(LineReading(text, line_confidence(lg, rec.temperature), lg.shape[0]))
        laps("rec_decode")
    return reads


def staged_predict(
    ocr: GlypharisOCR, img: Image.Image, sync: Any = None, laps: _Laps | None = None
) -> tuple[OCRResult, dict[str, float]]:
    """``ocr.predict(np.asarray(img.convert("RGB")))`` with a clock between its pieces: the same
    calls in the same order (`GlypharisOCR.predict`, `TextLineDetector.detect`,
    `GlypharisOCR.read_quads`, `TextRecognizer.read`), so the lines are the same. Returns the result
    and the milliseconds per stage (`STAGES`); ``laps`` is the clock, started by the caller."""
    laps = laps or _Laps(sync)
    rgb = load_rgb(np.asarray(img.convert("RGB")), ocr.max_pixels)
    laps("to_rgb")
    det = ocr.detector
    image, info = prepare_page(rgb, det.spec.infer.long_side)
    laps("det_preprocess")
    prob = det.backend(image)
    laps("det_network")
    cands, _ = find_candidates(prob, info, det.spec.post)
    boxes, _ = boxes_from_candidates(
        cands, det.spec.op(), det.spec.post, det.spec.labels.shrink_ratio, info
    )
    laps("det_postprocess")
    gray = Image.fromarray(rgb).convert("L")
    fill = background_grey(gray)
    crops = [cut_line(gray, [list(p) for p in b.quad], ocr.crop, fill) for b in boxes]
    laps("crop")
    reads = _recognise(ocr.recognizer, crops, laps)
    quads = [b.quad for b in boxes]
    order = reading_order(quads) if ocr.order else list(range(len(boxes)))
    lines = [
        OcrLine(
            boxes[i].polygon,
            boxes[i].quad,
            reads[i].text,
            reads[i].confidence,
            boxes[i].score,
            reads[i].at,
        )
        for i in order
    ]
    result = OCRResult(rgb.shape[1], rgb.shape[0], lines, {}, ocr.model_version)
    laps("reading_order")
    return result, laps.ms


def _same_lines(a: OCRResult, b: OCRResult) -> bool:
    return (a.width, a.height) == (b.width, b.height) and [ln.to_dict() for ln in a.lines] == [
        ln.to_dict() for ln in b.lines
    ]


def time_pages_staged(
    ocr: GlypharisOCR, root: Path, sync: Any = None, on_start: Any = None
) -> dict[str, Any]:
    """`WARMUP` + `PROFILE_SAMPLES` validation pages of `EVAL_SET`, one at a time, through
    `staged_predict`. Per page: the time from the decoded page to its ordered lines (as
    `_time_pages`, read off the stages' clock), the stages, the time of a second clock around the
    whole call, the time `open_page_image` took (outside the page time), the line count, the input
    size and the page id; refused if a page is not a validation page, if there are too few, or if
    a page's stages do not sum to the second clock's time within `STAGE_TOLERANCE`. ``on_start``
    is called just before the first page is read."""
    times: list[float] = []
    walls: list[float] = []
    file_ms: list[float] = []
    stages: list[dict[str, float]] = []
    n_lines: list[int] = []
    sizes: list[list[int]] = []
    ids: list[str] = []
    worst = 0.0
    first: Page | None = None
    split = EVAL_SETS[EVAL_SET]["split"]
    if on_start is not None:
        on_start()
    for k, page in zip(range(WARMUP + PROFILE_SAMPLES), e2e_samples(root, EVAL_SET),
                       strict=False):  # fmt: skip
        if page.split != split:
            raise SystemExit(f"page {page.doc_id} is from the {page.split} split; "
                             f"{EVAL_SET} profiles use {split} pages only")  # fmt: skip
        f0 = time.perf_counter()
        img = open_page_image(page, root)  # file read and decode, outside the page time
        f1 = time.perf_counter()
        laps = _Laps(sync)  # the page's first reading: from here to its ordered lines
        res, ms = staged_predict(ocr, img, laps=laps)
        wall = (time.perf_counter() - f1) * 1000  # a clock the stages do not read
        t = laps.total_ms
        diff = abs(sum(ms.values()) - wall)
        gap = diff / wall if wall > 0 else 0.0
        if diff > max(STAGE_TOLERANCE * wall, STAGE_SLACK_MS):
            raise SystemExit(
                f"page {page.doc_id}: the stages sum to {sum(ms.values()):.3f} ms, "
                f"the call took {wall:.3f} ms (more than 1% and 5 ms apart)"
            )
        worst = max(worst, gap)
        times.append(round(t, 3))
        walls.append(round(wall, 3))
        file_ms.append(round((f1 - f0) * 1000, 3))
        stages.append({s: round(v, 3) for s, v in ms.items()})
        n_lines.append(len(res.lines))
        sizes.append(list(img.size))
        ids.append(page.doc_id)
        if k == WARMUP:
            first = page
    if len(times) < WARMUP + PROFILE_SAMPLES:
        raise SystemExit(f"{EVAL_SET} gave {len(times)} pages; need {WARMUP + PROFILE_SAMPLES}")
    return {"times": times, "stages": stages, "lines": n_lines, "sizes": sizes, "ids": ids,
            "worst_gap": worst, "first_timed": first, "walls": walls,
            "file_decode": file_ms}  # fmt: skip


def staged_check(ocr: GlypharisOCR, page: Page, root: Path) -> dict[str, Any]:
    """`staged_predict` and `GlypharisOCR.predict` give the same lines on ``page``; SystemExit
    otherwise. Run after the timed pages, so the timings are untouched."""
    img = open_page_image(page, root)
    t0 = time.perf_counter()
    staged, _ = staged_predict(ocr, img)
    t1 = time.perf_counter()
    plain = ocr.predict(np.asarray(img.convert("RGB")))
    t2 = time.perf_counter()
    if not _same_lines(staged, plain):
        raise SystemExit(f"page {page.doc_id}: the staged pipeline read other lines than "
                         "GlypharisOCR.predict")  # fmt: skip
    return {"page_id": page.doc_id, "lines": len(plain.lines), "same_as_predict": True,
            "staged_ms": round((t1 - t0) * 1000, 3),
            "predict_ms": round((t2 - t1) * 1000, 3)}  # fmt: skip


def record_differences(ocr: GlypharisOCR, record: dict[str, Any]) -> list[str]:
    """Where the settings ``ocr`` holds differ from the freeze record's (the comparison
    `load_frozen` makes, without the strict-fp32 flag, which an ONNX recognizer does not use)."""
    got = Frozen(ocr, ocr.recognizer, {}).settings()
    det, rec = record["detector"], record["recognizer"]
    pairs = [
        ("detector resolution", got["detector"]["infer"], det["infer"]),
        ("detector post-processing", got["detector"]["post"], det["post"]),
        ("detector operating point", got["detector"]["operating_point"], det["operating_point"]),
        ("detector label version", got["detector"]["label_version"], det["label_version"]),
        ("recognizer input", got["recognizer"]["input"], rec["input"]),
        ("recognizer temperature", got["recognizer"]["temperature"], rec["temperature"]),
    ]
    return [f"{what}: {g!r}, record {w!r}" for what, g, w in pairs if g != w]


def _rel_name(path: Path) -> str:
    """``path`` relative to the working directory when it lies inside it, else its name only,
    so a profile holds no absolute paths."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _onnx_build(
    bundle: Path, threads: int | None
) -> tuple[GlypharisOCR, dict[str, Any], dict[str, Any]]:
    """The bundle on ONNX Runtime, refused unless its hashes and settings match the freeze
    record; returns the pipeline, what the profile records of the bundle, and the provenance."""
    record = yaml.safe_load(FREEZE_RECORD.read_text(encoding="utf-8"))
    manifest = read_manifest(bundle)  # every file hash, before anything loads
    ocr = load_bundle_onnx(bundle, threads)
    wrong = record_differences(ocr, record)
    if wrong:
        raise SystemExit(f"{bundle.name} is not the v1 build: " + "; ".join(wrong))
    info = {
        "path": _rel_name(bundle),
        "manifest_sha256": sha256_file(bundle / "manifest.json"),
        "version": manifest.get("version"),
        "files": manifest["files"],
        "onnx_opset": manifest.get("onnx_opset"),
        "sources": manifest.get("sources"),
        "record_settings": "equal",
    }
    provenance = {
        "record": FREEZE_RECORD.as_posix(),
        "record_sha256": sha256_file(FREEZE_RECORD),
        "name": record.get("name"),
        "bundle_manifest_sha256": info["manifest_sha256"],
    }
    return ocr, info, provenance


def _parse_threads(value: str) -> int | str:
    """``--threads``: a positive integer, or ``default``."""
    if value == "default":
        return value
    n = int(value)  # argparse reports the ValueError as a usage error
    if n < 1:
        raise ValueError(value)
    return n


def _v1_threads(n: int | str) -> dict[str, Any]:
    """The thread record: made before numpy and torch loaded for a command-line run; in a process
    that already holds them (the tests) set now, and recorded as not set before the import. With
    ``default`` nothing is set and the live values are recorded."""
    if isinstance(n, int):
        if _V1_THREADS is not None:
            return dict(_V1_THREADS)
        return set_threads(n, require_fresh=False)
    preloaded = [m for m in ("numpy", "torch") if m in sys.modules]
    return {"requested": "default", **thread_record(), "env_before_torch_import": None,
            "preloaded": preloaded}  # fmt: skip


def _checked_threads(n: int | str) -> dict[str, Any]:
    try:
        return check_threads(n) if isinstance(n, int) else thread_record()
    except ThreadError as exc:
        raise SystemExit(str(exc)) from exc


def main_v1(argv: list[str]) -> int:
    """Phase 16: the v1 build's page profile (see the module docstring)."""
    parser = argparse.ArgumentParser(description="Page profile of the GLYPHARIS v1 build")
    parser.add_argument(V1_FLAG, action="store_true", help="the v1 build (Phase 16)")
    parser.add_argument("--backend", choices=["torch", "onnx"], required=True)
    parser.add_argument(THREADS_FLAG, type=_parse_threads, required=True,
                        help="N, or 'default' (ONNX only: nothing set)")  # fmt: skip
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=None, help="the v1 bundle (ONNX)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    n: int | str = args.threads
    if args.backend == "onnx" and args.bundle is None:
        parser.error("--backend onnx needs --bundle")
    if args.backend == "torch" and args.bundle is not None:
        parser.error("--backend torch builds from the freeze record; --bundle is for onnx")
    if args.backend == "onnx" and args.device != "cpu":
        parser.error("ONNX Runtime runs on the CPU here")
    if n == "default" and args.backend != "onnx":
        parser.error("--threads default is measured for ONNX only")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: no CUDA device")
    if args.out.exists():
        # A void sitting is moved aside by hand and recorded in research/failures.md.
        raise SystemExit(f"{args.out.as_posix()} exists: move it aside first")
    import psutil

    try:
        threads = _v1_threads(n)
        pinned = assert_v1_code(".")  # before the model loads
    except (ThreadError, V1CodeError) as exc:
        raise SystemExit(str(exc)) from exc
    torch.set_grad_enabled(False)
    proc = psutil.Process()
    cond = ProfileConditions(cpu=lambda: run_conditions(), gpu=lambda: gpu_state())
    cond.start()  # before the model loads, so the CPU check sees an idle machine
    git = git_state()
    cuda = args.device == "cuda"
    sync = torch.cuda.synchronize if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    ort_threads = n if isinstance(n, int) else None
    if args.backend == "torch":
        frozen = load_frozen(args.device)
        ocr, bundle_info, provenance = frozen.ocr, None, frozen.provenance
        runtime = f"torch {torch.__version__} ({provenance.get('device_name', args.device)})"
    else:
        import onnxruntime

        ocr, bundle_info, provenance = _onnx_build(args.bundle, ort_threads)
        runtime = f"onnxruntime {onnxruntime.__version__} (CPU)"
    try:
        sessions = check_sessions(ocr, ort_threads)
    except ThreadError as exc:
        raise SystemExit(str(exc)) from exc
    before = _checked_threads(n)
    clock: dict[str, str] = {}

    def started() -> None:  # just before the first warm-up page
        clock["started"] = datetime.now(UTC).isoformat(timespec="seconds")

    run = time_pages_staged(ocr, args.data_root, sync, started)
    ended = datetime.now(UTC).isoformat(timespec="seconds")  # after the last timed page
    after = _checked_threads(n)
    peak = round(peak_working_set(proc) / 2**20, 1)
    vram: dict[str, Any] | None = None
    if cuda:
        vram = {
            "max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "max_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }
    if vram is not None:  # while this process still holds its CUDA memory
        vram["nvidia_smi_process_mib_end"] = gpu_process_memory_mib(os.getpid())
    end = cond.end()
    if vram is not None:
        gpus = (end.get("gpu") or {}).get("gpus") or [{}]
        vram["nvidia_smi_whole_gpu_used_mib_end"] = gpus[0].get("memory_used_mib")
    check = staged_check(ocr, run["first_timed"], args.data_root)
    times, stages, lines = run["times"], run["stages"], run["lines"]
    timed = stages[WARMUP:]
    settings = {
        **Frozen(ocr, ocr.recognizer, {}).settings(),
        "device": args.device,
        "precision": "fp32",
        "backend": args.backend,
        "runtime": runtime,
        "cpu_threads": n if isinstance(n, int) else None,  # the baselines' key
        "intra_op_threads": after.get("intra_op"),
        "inter_op_threads": after.get("inter_op"),
    }
    t = np.asarray(times[WARMUP:])
    res: dict[str, Any] = {
        "system": "glypharis",
        "task": "e2e",
        "eval_set": EVAL_SET,
        "experiment": args.experiment,
        "build": "v1",
        "backend": args.backend,
        "device": args.device,
        "version": ocr.model_version,
        "settings": settings,
        "threads": threads,
        "threads_check": {"before": before, "after": after},
        "session_options": sessions,
        "batch": 1,
        "started_utc": clock["started"],
        "ended_utc": ended,
        "warmup": WARMUP,
        "warmup_times_ms": times[:WARMUP],
        "times_ms": times[WARMUP:],
        "stages": list(STAGES),
        "warmup_stages_ms": stages[:WARMUP],
        "stages_ms": timed,
        "stage_sum_max_rel_diff": round(run["worst_gap"], 6),
        "stage_check": "the stages' sum against a second clock around each staged call",
        "call_ms": run["walls"][WARMUP:],
        "file_decode_ms": run["file_decode"][WARMUP:],
        "mean_file_decode_ms": float(np.mean(run["file_decode"][WARMUP:])),
        "mean_stages_ms": {s: float(np.mean([p[s] for p in timed])) for s in STAGES},
        "lines": lines[WARMUP:],
        "input_sizes": run["sizes"][WARMUP:],
        "page_ids": run["ids"][WARMUP:],
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_lines": float(np.mean(lines[WARMUP:])),
        "peak_mem_mib": peak,
        "peak_vram": vram,
        "staged_check": check,
        "provenance": provenance,
        "bundle": bundle_info,
        "v1_code": pinned,
        **cond.fields(),
        "cpu_busy_percent_after": cpu_busy_percent(),
        **git,
    }
    if args.out.exists():
        raise SystemExit(f"{args.out.as_posix()} exists: move it aside first")
    write_json_atomic(args.out, res, ensure_ascii=True)
    print(
        f"{args.experiment} v1 {args.backend} on {args.device} at {n} threads: page mean "
        f"{res['mean_ms']:.0f} ms, p95 {res['p95_ms']:.0f} ms ({res['mean_lines']:.1f} lines); "
        f"peak {peak} MiB; written to {args.out.as_posix()}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if FROZEN_FLAG in argv and V1_FLAG in argv:
        raise SystemExit(f"{FROZEN_FLAG} (GW-EFF-005) and {V1_FLAG} (Phase 16) are separate runs")
    if FROZEN_FLAG in argv:
        return main_frozen(argv)
    if V1_FLAG in argv:
        return main_v1(argv)
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
