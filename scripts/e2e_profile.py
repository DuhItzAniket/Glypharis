"""CPU page time of the whole GLYPHARIS pipeline, measured like the baselines (GW-EFF-002).

    python scripts/e2e_profile.py --config configs/det/gw-det-001.yaml \
        --det-checkpoint checkpoints/GW-DET-001/best.pt --rec-run runs/GW-REC-024 \
        [--set det.post... --op ...] --out reports/efficiency/GW-EFF-002.json
    python scripts/e2e_profile.py --config configs/det/gw-det-001.yaml \
        --bundle runs/release/glypharis-0.1.0 --backend onnx --experiment GW-EFF-004 \
        --out reports/efficiency/GW-EFF-004.json

Same protocol and pages as `scripts/run_baselines.py --profile` for the baselines' page profiles:
the seeded Phase 5 HierText validation sample, one page at a time, 10 warm-up pages then 100 timed.
A timing covers a decoded page to its ordered lines: RGB conversion, detection, line crops,
recognition and reading order, on the CPU with default threads: PyTorch fp32 from the runs, or a
model bundle on either backend (``--bundle``, Phase 17), where ONNX Runtime is the build ADR-0011
is judged on. Peak memory is this process's. Run on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
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
from glypharis.export.bundle import load_bundle
from glypharis.models.det.detector import TextLineDetector
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer

EVAL_SET = "hiertext_val"
PAGE_MEAN_MS, PAGE_P95_MS, PAGE_RAM_MIB = 1500.0, 2700.0, 1836.0  # ADR-0011


def main(argv: list[str] | None = None) -> int:
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
    times, parts, n_lines, sizes = [], [], [], []
    for _, page in zip(range(WARMUP + PROFILE_SAMPLES), e2e_samples(root, EVAL_SET), strict=False):
        img = open_page_image(page, root)
        t0 = time.perf_counter()
        res = ocr.predict(np.asarray(img.convert("RGB")))
        times.append(round((time.perf_counter() - t0) * 1000, 3))
        parts.append({k: round(v * 1000, 3) for k, v in res.seconds.items()})
        n_lines.append(len(res.lines))
        sizes.append(list(img.size))
    if len(times) < WARMUP + PROFILE_SAMPLES:
        raise SystemExit(f"{EVAL_SET} gave {len(times)} pages; need {WARMUP + PROFILE_SAMPLES}")
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
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "mean_detect_ms": float(np.mean([p["detect"] for p in timed])),
        "mean_recognise_ms": float(np.mean([p["recognise"] for p in timed])),
        "mean_lines": float(np.mean(n_lines[WARMUP:])),
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(
        f"page mean {res['mean_ms']:.0f} ms, p95 {res['p95_ms']:.0f} ms (detect "
        f"{res['mean_detect_ms']:.0f}, recognise {res['mean_recognise_ms']:.0f}, "
        f"{res['mean_lines']:.1f} lines); peak {res['peak_mem_mib']} MiB; "
        f"meets ADR-0011: {res['meets_target']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
