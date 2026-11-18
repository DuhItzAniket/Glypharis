"""CPU speed and memory of trained GLYPHARIS recognizers, measured like the baselines.

    python scripts/rec_profile.py --run GW-REC-003        # one run, in its own process
    python scripts/rec_profile.py --table                 # the table from every profiled run

Same protocol as `scripts/run_baselines.py --profile` (EXPERIMENT_PROTOCOL §3), on the same crops:
the seeded HierText validation sample the baselines were timed on, one line at a time (batch 1), the
first 10 discarded as warm-up and the next 100 timed. A timing covers everything inference does with
one crop: grayscale, resize to the model's height, normalise, pad, the forward pass and greedy CTC
decoding. The model is the run's `best.pt`, on the CPU in fp32 with PyTorch's default thread count:
a stand-in for the ONNX Runtime build of Phase 17 (ADR-0011). Run conditions are recorded as for the
baselines; the machine should be otherwise idle, with no training running.

Each run is profiled in its own process, so its peak memory is its own. Results go to
`reports/training/phase07/cpu_latency/<run>.json`; `--table` writes `cpu_latency.md` next to them,
with the baselines' recognition profiles for reference and ADR-0011's line target.

``--experiment`` (Phase 16: GW-EFF-007, 008, 011) profiles one recognizer under stated conditions,
on the same crops and counts, and writes one file (``--out-json``, never overwritten):

    python scripts/rec_profile.py --experiment GW-EFF-008 --run <run dir> --threads 10 \
        --backend onnx --onnx <recognizer.onnx> [--precision int8] --out-json <file>
    python scripts/rec_profile.py --experiment GW-EFF-007 --run <run dir> --checkpoint <best.pt> \
        --threads 10 --backend torch --device cuda --batch 32 --out-json <file>

``--run`` is an experiment ID under ``--runs`` or a run directory (absolute paths work); its
resolved config gives the charset, input height, maximum width and architecture. PyTorch loads
``--checkpoint`` (default the run's ``best.pt``) with v1's numerics (`strict_inference`, strict
fp32); ONNX Runtime runs the exported model file ``--onnx`` on an `ort_session` with its options
recorded and asserted (the PyTorch model is built on the meta device for its stride and class count
only, so no weights load). ``--threads N`` is set before numpy and torch are imported
(`set_threads`) and asserted before and after the timed items; ``--threads default`` (ONNX only)
sets nothing and records the live values. One item is ``--batch`` consecutive crops (1 or 32):
``--padding pipeline`` (the default) reads them with `TextRecognizer.read`, exactly as the pipeline
reads a page's lines: sorted by width, split into batches at `MAX_BATCH_LINES` and
`MAX_BATCH_COLUMNS`, each line padded by at least `PAD_MARGIN` columns, greedy text and confidence
(at batch 1 the input the variants' line profiles and the pipeline time); ``--padding phase7``
(batch 1 only) pads to the line rounded up to `PAD_MULTIPLE`, as Phase 7's `read_line` does, for
re-confirming Phase 7's table. Each item's time is recorded; on CUDA the device is synchronised
before and after each item and peak VRAM is recorded from torch (``max_memory_allocated``) with
nvidia-smi's memory at the end, for this process where the driver reports it (Windows' WDDM
driver does not) and for the whole GPU (the display and other programs included).
``--precision int8`` is checked against the ONNX file's operators after the timed items. The v1
code paths must equal ``phase14-eval`` (`assert_v1_code`); run conditions and the GPU's state are
recorded at the start and the end.
"""

# ruff: noqa: E402  (the thread variables are set before numpy and torch are imported)

from __future__ import annotations

import sys
from collections.abc import Sequence

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


# The thread record of a command-line run with --threads N, made before numpy and torch load.
_THREADS: dict[str, object] | None = None

if __name__ == "__main__":
    _n = requested_threads(sys.argv[1:])
    if _n is not None:
        from glypharis.efficiency.threads import set_threads

        _THREADS = set_threads(_n)

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from run_baselines import (
    PROFILE_SAMPLES,
    SYSTEM_NAMES,
    WARMUP,
    cpu_busy_percent,
    git_state,
    peak_working_set,
    rec_samples,
    run_conditions,
)

from glypharis.config.loader import load_config
from glypharis.data.download import sha256_file
from glypharis.data.rec_dataset import PAD_MULTIPLE, resized_width
from glypharis.data.text import Charset
from glypharis.efficiency.conditions import ProfileConditions, gpu_process_memory_mib, gpu_state
from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.onnx_build import SessionLineModel
from glypharis.efficiency.threads import (
    ThreadError,
    check_session,
    check_threads,
    ort_session,
    set_threads,
    thread_record,
)
from glypharis.efficiency.v1 import V1CodeError, assert_v1_code
from glypharis.models.rec.crnn import build_recognizer
from glypharis.pipeline.frozen import strict_inference
from glypharis.pipeline.recognizer import (
    MAX_BATCH_COLUMNS,
    MAX_BATCH_LINES,
    PAD_MARGIN,
    TextRecognizer,
    prepare_crop,
)
from glypharis.training.checkpoint import load_checkpoint
from glypharis.training.rec_trainer import greedy_decode

OUT = Path("reports/training/phase07/cpu_latency")
TARGET_MEAN_MS, TARGET_P95_MS = 25.0, 70.0  # ADR-0011, line recognition
EVAL_SET = "hiertext_val"


def load_run(run: str, runs: Path) -> tuple[Any, torch.nn.Module, Charset, dict[str, Any]]:
    cfg = load_config(runs / run / "config.resolved.yaml")
    charset = Charset.load(cfg.data.charset or Path(cfg.paths.data_root) / "manifests/charset.json")
    model = build_recognizer(charset.num_classes, cfg.model, cfg.model_name or "crnn")
    model.load_state_dict(load_checkpoint(Path(cfg.checkpoint.dir) / "best.pt")["model"])
    model.eval()
    summary = json.loads((runs / run / "summary.json").read_text(encoding="utf-8"))
    return cfg, model, charset, summary


def read_line(model: Any, charset: Charset, img: Image.Image, height: int, max_width: int) -> str:
    """One crop to text, as inference would do it."""
    g = img.convert("L")
    w = resized_width(g.width, g.height, height, max_width)
    g = g.resize((w, height), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(g, dtype=np.float32) / 127.5 - 1.0)
    batch = torch.empty(1, 1, height, math.ceil(w / PAD_MULTIPLE) * PAD_MULTIPLE)
    batch[0, 0, :, :w] = x
    batch[0, 0, :, w:] = x[:, -1].median()
    widths = torch.tensor([w])
    with torch.inference_mode():
        logits = model(batch, widths)
    lengths = model.frames(widths).clamp(max=logits.shape[0])
    return greedy_decode(logits, lengths, charset)[0]


def best_val_cer(run_dir: Path, best_epoch: int) -> float | None:
    for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("type") == "epoch" and r.get("epoch") == best_epoch:
            return float(r["val_cer"])
    return None


def profile(run: str, runs: Path, data: Path, out: Path) -> dict[str, Any]:
    import psutil

    proc = psutil.Process()
    conditions = run_conditions()  # before the model loads, so the CPU check sees an idle machine
    cfg, model, charset, summary = load_run(run, runs)
    height, max_width = cfg.data.image_height, cfg.data.max_width
    times, sizes = [], []
    samples = (img for _, _, img in rec_samples(data, EVAL_SET))
    for _, img in zip(range(WARMUP + PROFILE_SAMPLES), samples, strict=False):
        t0 = time.perf_counter()
        read_line(model, charset, img, height, max_width)
        times.append(round((time.perf_counter() - t0) * 1000, 3))
        sizes.append(list(img.size))
    if len(times) < WARMUP + PROFILE_SAMPLES:
        raise SystemExit(f"{EVAL_SET} gave {len(times)} samples; profiling needs "
                         f"{WARMUP} warm-up + {PROFILE_SAMPLES} timed")  # fmt: skip
    t = np.asarray(times[WARMUP:])
    res = {
        "run": run, "model_name": cfg.model_name or "crnn", "image_height": height,
        "frame_stride": cfg.model.frame_stride, "parameters": int(summary["parameters"]),
        "model_mib": round(summary["model_bytes"] / 2**20, 2),
        "val_cer": best_val_cer(runs / run, int(summary["best_epoch"])),
        "eval_set": EVAL_SET, "batch": 1, "warmup": WARMUP, "precision": "fp32",
        "runtime": f"torch {torch.__version__} (CPU)", "torch_threads": torch.get_num_threads(),
        "warmup_times_ms": times[:WARMUP], "times_ms": times[WARMUP:],
        "input_sizes": sizes[WARMUP:],
        "mean_ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95)),
        "peak_mem_mib": round(peak_working_set(proc) / 2**20, 1),
        "conditions": conditions, "cpu_busy_percent_after": cpu_busy_percent(), **git_state(),
    }  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run}.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"profile {run}: mean {res['mean_ms']:.1f} ms, p95 {res['p95_ms']:.1f} ms, "
          f"peak {res['peak_mem_mib']} MiB")  # fmt: skip
    return res


def table(out: Path, data: Path) -> str:
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("GW-REC-*.json"))]
    refs = []
    for p in sorted((data / "processed" / "baselines").glob(f"*/profile_*_rec_{EVAL_SET}.json")):
        b = json.loads(p.read_text(encoding="utf-8"))
        t = np.asarray(b["times_ms"])
        refs.append((b["experiment"], SYSTEM_NAMES.get(b["system"], b["system"]), float(t.mean()),
                     float(np.percentile(t, 95)), b["peak_mem_mib"]))  # fmt: skip

    def meets(mean: float, p95: float) -> str:
        return "yes" if mean <= TARGET_MEAN_MS and p95 <= TARGET_P95_MS else "no"

    md = [
        "# Phase 7: recognizer CPU latency",
        "",
        "Generated by `scripts/rec_profile.py`; do not edit by hand. The baselines' protocol",
        f"and crops: HierText validation, batch 1, {WARMUP} warm-up then {PROFILE_SAMPLES} timed",
        "lines, on the CPU. A timing covers preprocessing, the forward pass and greedy decoding.",
        "GLYPHARIS runs in PyTorch (fp32, default threads) until the ONNX export of Phase 17.",
        f"ADR-0011's line target: mean at most {TARGET_MEAN_MS:.0f} ms and p95 at most "
        f"{TARGET_P95_MS:.0f} ms.",
        "",
        "| Run | Model | Height | Frame stride | Parameters | Val CER % | ms mean | ms p95 | "
        "Peak RAM MiB | Threads | CPU busy % before/after | Meets target |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cer = "-" if r["val_cer"] is None else f"{100 * r['val_cer']:.2f}"
        busy = f"{r['conditions'].get('cpu_busy_percent_before')} / {r['cpu_busy_percent_after']}"
        md.append(f"| {r['run']} | {r['model_name']} | {r['image_height']} | {r['frame_stride']} | "
                  f"{r['parameters']:,} | {cer} | {r['mean_ms']:.1f} | {r['p95_ms']:.1f} | "
                  f"{r['peak_mem_mib']:,.0f} | {r['torch_threads']} | {busy} | "
                  f"{meets(r['mean_ms'], r['p95_ms'])} |")  # fmt: skip
    if refs:
        md += ["", "Baselines on the same lines (`reports/baselines/efficiency.md`):", "",
               "| ID | System | ms mean | ms p95 | Peak RAM MiB | Meets target |",
               "|---|---|---|---|---|---|"]  # fmt: skip
        md += [f"| {e} | {n} | {m:.1f} | {p:.1f} | {mem:,.0f} | {meets(m, p)} |"
               for e, n, m, p, mem in refs]  # fmt: skip
    plans = sorted({str(r["conditions"].get("power_plan")) for r in rows})
    md += ["", f"Power plan: {'; '.join(plans) or '-'}.", ""]
    text = "\n".join(md)
    (out.parent / "cpu_latency.md").write_text(text, encoding="utf-8", newline="\n")
    return text


# ------------------------------------------------------------------- Phase 16 (--experiment)

BATCHES = (1, 32)
PADDINGS = ("phase7", "pipeline")
# Operators a dynamically quantised ONNX model holds and an fp32 export does not.
INT8_OPS = frozenset(
    {"MatMulInteger", "DynamicQuantizeLinear", "DynamicQuantizeMatMul", "QLinearMatMul",
     "ConvInteger", "QLinearConv"}
)  # fmt: skip
# The flags only a Phase 16 run takes (each defaults to None, so a legacy run is unchanged).
PHASE16_FLAGS = ("threads", "backend", "onnx", "checkpoint", "batch", "padding", "device",
                 "precision", "out_json")  # fmt: skip


def read_batch(rec: TextRecognizer, imgs: Sequence[Image.Image], padding: str) -> list[str]:
    """Crops to text. ``pipeline``: `TextRecognizer.read`, the pipeline's own batching (sorted by
    width, split at its line and column caps, `PAD_MARGIN` padding), numerics and decoding.
    ``phase7``: one batch as `read_line` reads one crop: `prepare_crop`, right padding to the
    widest line rounded up to `PAD_MULTIPLE` with each line's last-column median, the forward pass
    under ``rec``'s numerics, greedy CTC decoding."""
    if padding == "pipeline":
        return [r.text for r in rec.read(imgs)]
    xs = [prepare_crop(img, rec.height, rec.max_width) for img in imgs]
    widths = [x.shape[1] for x in xs]
    cols = math.ceil(max(widths) / PAD_MULTIPLE) * PAD_MULTIPLE
    batch = torch.empty(len(xs), 1, rec.height, cols)
    for j, x in enumerate(xs):
        batch[j, 0, :, : x.shape[1]] = x
        batch[j, 0, :, x.shape[1] :] = x[:, -1].median()
    w = torch.tensor(widths)
    with torch.inference_mode(), rec._numerics():
        logits = rec.model(batch.to(rec.device), w.to(rec.device))
    lengths = rec.model.frames(w).clamp(max=logits.shape[0])
    return greedy_decode(logits, lengths, rec.charset)


def run_dir(run: str, runs: Path) -> Path:
    """``run`` as a directory: an absolute (or existing) path as given, else ``runs / run``."""
    p = Path(run)
    return p if p.is_absolute() or (p / "config.resolved.yaml").is_file() else runs / run


def onnx_int8_ops(path: Path) -> list[str]:
    """The quantised operators (`INT8_OPS`) an ONNX file holds, sorted and without repeats."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return sorted({node.op_type for node in model.graph.node} & INT8_OPS)


def _file(kind: str, path: Path) -> dict[str, Any]:
    return {"kind": kind, "name": path.name, "sha256": sha256_file(path),
            "mib": round(path.stat().st_size / 2**20, 2)}  # fmt: skip


def _live_threads(n: int | str) -> dict[str, Any]:
    try:
        return check_threads(n) if isinstance(n, int) else thread_record()
    except ThreadError as exc:
        raise SystemExit(str(exc)) from exc


def profile16(args: argparse.Namespace) -> dict[str, Any]:
    """One Phase 16 recognizer profile (module docstring); writes ``args.out_json``."""
    import psutil

    n: int | str = args.threads
    out = args.out_json
    if out.exists():
        # A void sitting is moved aside by hand and recorded in research/failures.md.
        raise SystemExit(f"{out.as_posix()} exists: move it aside first")
    try:
        if isinstance(n, int):
            threads = dict(_THREADS) if _THREADS is not None else set_threads(
                n, require_fresh=False)  # fmt: skip
        else:
            preloaded = [m for m in ("numpy", "torch") if m in sys.modules]
            threads = {"requested": "default", **thread_record(),
                       "env_before_torch_import": None, "preloaded": preloaded}  # fmt: skip
        pinned = assert_v1_code(".")  # before the model loads
    except (ThreadError, V1CodeError) as exc:
        raise SystemExit(str(exc)) from exc
    torch.set_grad_enabled(False)
    proc = psutil.Process()
    cond = ProfileConditions(cpu=lambda: run_conditions(), gpu=lambda: gpu_state())
    cond.start()  # before the model loads, so the CPU check sees an idle machine
    git = git_state()
    cuda = args.device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    directory = run_dir(args.run[0], args.runs)
    cfg = load_config(directory / "config.resolved.yaml")
    charset = Charset.load(cfg.data.charset or Path(cfg.paths.data_root) / "manifests/charset.json")
    name = cfg.model_name or "crnn"
    height, max_width = cfg.data.image_height, cfg.data.max_width
    ort_threads = n if isinstance(n, int) else None
    sessions = numerics = None
    if args.backend == "torch":
        ck = Path(args.checkpoint) if args.checkpoint else Path(cfg.checkpoint.dir) / "best.pt"
        model = build_recognizer(charset.num_classes, cfg.model, name)
        model.load_state_dict(load_checkpoint(ck)["model"])
        numerics = strict_inference(args.device)  # v1's numerics, as load_frozen sets them
        rec = TextRecognizer(model, charset, height, max_width, args.device, strict_fp32=True)
        model_file = _file("checkpoint", ck)
        runtime = f"torch {torch.__version__} ({torch.cuda.get_device_name() if cuda else 'CPU'})"
    else:
        import onnxruntime

        with torch.device("meta"):  # the stride and class count, without weights
            model = build_recognizer(charset.num_classes, cfg.model, name)
        session = ort_session(args.onnx, ort_threads)
        line_model = SessionLineModel(session, model.stride, model.num_classes)
        rec = TextRecognizer(line_model, charset, height, max_width, "cpu")
        try:
            sessions = check_session(session, ort_threads)
        except ThreadError as exc:
            raise SystemExit(str(exc)) from exc
        model_file = _file("onnx", Path(args.onnx))
        runtime = f"onnxruntime {onnxruntime.__version__} (CPU)"
    parameters = sum(p.numel() for p in model.parameters())
    before = _live_threads(n)
    batch, padding = args.batch, args.padding
    times, sizes = [], []
    samples = (img for _, _, img in rec_samples(args.data_root, EVAL_SET))
    started = datetime.now(UTC).isoformat(timespec="seconds")  # just before the first warm-up item
    for _ in range(WARMUP + PROFILE_SAMPLES):
        imgs = list(islice(samples, batch))
        if len(imgs) < batch:
            break
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        read_batch(rec, imgs, padding)
        if cuda:
            torch.cuda.synchronize()
        times.append(round((time.perf_counter() - t0) * 1000, 3))
        sizes.append(list(imgs[0].size) if batch == 1 else [list(im.size) for im in imgs])
    ended = datetime.now(UTC).isoformat(timespec="seconds")  # after the last timed item
    if len(times) < WARMUP + PROFILE_SAMPLES:
        raise SystemExit(f"{EVAL_SET} gave {len(times)} items of {batch} lines; profiling needs "
                         f"{WARMUP} warm-up + {PROFILE_SAMPLES} timed")  # fmt: skip
    after = _live_threads(n)
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
    int8_ops = onnx_int8_ops(Path(args.onnx)) if args.backend == "onnx" else []
    if (args.precision == "int8") != bool(int8_ops):
        raise SystemExit(f"--precision {args.precision}, but {model_file['name']} holds "
                         f"{int8_ops or 'no'} quantised operators")  # fmt: skip
    t = np.asarray(times[WARMUP:])
    res: dict[str, Any] = {
        "experiment": args.experiment,
        "run": directory.name,
        "model_name": name,
        "image_height": height,
        "max_width": max_width,
        "frame_stride": model.stride,
        "parameters": int(parameters),
        "model_file": model_file,
        "eval_set": EVAL_SET,
        "backend": args.backend,
        "device": args.device,
        "precision": args.precision,
        "int8_ops": int8_ops,
        "batch": batch,
        "padding": padding,
        "batching": (
            f"TextRecognizer.read: sorted by width, split at {MAX_BATCH_LINES} lines and "
            f"{MAX_BATCH_COLUMNS} columns, each line padded by at least {PAD_MARGIN} columns"
            if padding == "pipeline"
            else f"one tensor, the line padded to a multiple of {PAD_MULTIPLE} (read_line)"
        ),
        "warmup": WARMUP,
        "runtime": runtime,
        "settings": {
            "cpu_threads": n if isinstance(n, int) else None,  # the baselines' key
            "intra_op_threads": after.get("intra_op"),
            "inter_op_threads": after.get("inter_op"),
        },
        "torch_threads": torch.get_num_threads(),
        "threads": threads,
        "threads_check": {"before": before, "after": after},
        "session_options": sessions,
        "numerics": numerics,
        "started_utc": started,
        "ended_utc": ended,
        "warmup_times_ms": times[:WARMUP],
        "times_ms": times[WARMUP:],
        "per_line_ms": [round(v / batch, 4) for v in times[WARMUP:]],
        "input_sizes": sizes[WARMUP:],
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_line_ms": float(t.mean() / batch),
        "lines_per_second": float(batch * len(t) / (t.sum() / 1000)),
        "peak_mem_mib": peak,
        "peak_vram": vram,
        "v1_code": pinned,
        **cond.fields(),
        "cpu_busy_percent_after": cpu_busy_percent(),
        **git,
    }
    if out.exists():
        raise SystemExit(f"{out.as_posix()} exists: move it aside first")
    write_json_atomic(out, res, ensure_ascii=True)
    print(f"{args.experiment} {directory.name} {args.backend} {args.precision} batch {batch} at "
          f"{n} threads: mean {res['mean_ms']:.2f} ms, p95 {res['p95_ms']:.2f} ms per item; "
          f"peak {peak} MiB; written to {out.as_posix()}")  # fmt: skip
    return res


def _parse_threads(value: str) -> int | str:
    """``--threads``: a positive integer, or ``default``."""
    if value == "default":
        return value
    n = int(value)  # argparse reports the ValueError as a usage error
    if n < 1:
        raise ValueError(value)
    return n


def _check_phase16(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse a Phase 16 command line that is incomplete or inconsistent; fill the defaults."""
    if len(args.run) != 1 or args.table:
        parser.error("--experiment profiles exactly one --run, without --table")
    if args.threads is None or args.out_json is None:
        parser.error("--experiment needs --threads and --out-json")
    args.backend = args.backend or "torch"
    args.device = args.device or "cpu"
    args.batch = args.batch or 1
    args.padding = args.padding or "pipeline"  # the pipeline's own reading (module docstring)
    args.precision = args.precision or "fp32"
    if args.backend == "onnx" and (args.onnx is None or args.checkpoint is not None):
        parser.error("--backend onnx reads --onnx (an exported model file), not --checkpoint")
    if args.backend == "torch" and args.onnx is not None:
        parser.error("--onnx is for --backend onnx")
    if args.backend == "onnx" and args.device != "cpu":
        parser.error("ONNX Runtime runs on the CPU here")
    if args.threads == "default" and args.backend != "onnx":
        parser.error("--threads default is measured for ONNX only")
    if args.precision == "int8" and args.backend != "onnx":
        parser.error("--precision int8 is an ONNX model")
    if args.batch > 1 and args.padding != "pipeline":
        parser.error(f"--batch {args.batch} is read as the pipeline reads it: --padding pipeline")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: no CUDA device")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", default=[], help="experiment ID, e.g. GW-REC-003")
    parser.add_argument("--table", action="store_true", help="write cpu_latency.md")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=OUT)
    p16 = parser.add_argument_group("Phase 16 (with --experiment)")
    p16.add_argument("--experiment", default=None, help="a Phase 16 profile, e.g. GW-EFF-008")
    p16.add_argument(THREADS_FLAG, type=_parse_threads, default=None,
                     help="N, or 'default' (ONNX only: nothing set)")  # fmt: skip
    p16.add_argument("--backend", choices=["torch", "onnx"], default=None)
    p16.add_argument("--onnx", type=Path, default=None, help="the exported recognizer (ONNX)")
    p16.add_argument("--checkpoint", type=Path, default=None, help="default: the run's best.pt")
    p16.add_argument("--batch", type=int, choices=BATCHES, default=None)
    p16.add_argument("--padding", choices=PADDINGS, default=None)
    p16.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p16.add_argument("--precision", choices=["fp32", "int8"], default=None)
    p16.add_argument("--out-json", type=Path, default=None, help="the profile file")
    args = parser.parse_args(argv)
    if args.experiment is not None:
        _check_phase16(parser, args)
        profile16(args)
        return 0
    given = [f"--{f.replace('_', '-')}" for f in PHASE16_FLAGS if getattr(args, f) is not None]
    if given:
        parser.error(f"{', '.join(given)}: Phase 16 flags, which need --experiment")
    if len(args.run) > 1:
        raise SystemExit("one --run per process, so each peak memory is that run's own")
    for run in args.run:
        profile(run, args.runs, args.data_root, args.out)
    if args.table:
        print(table(args.out, args.data_root))
    if not args.run and not args.table:
        parser.error("give --run, --table or both")
    return 0


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    sys.exit(main())
