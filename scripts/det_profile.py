"""CPU speed of GLYPHARIS-Det per page, measured like the baselines (Phase 9, GW-EFF-001).

    python scripts/det_profile.py --config configs/det/gw-det-001.yaml \
        --checkpoint checkpoints/GW-DET-001/best.pt --set det.infer.long_side=1536
    python scripts/det_profile.py --table      # the table from every profiled long side

Same protocol as `scripts/run_baselines.py --profile` (EXPERIMENT_PROTOCOL §3), on the same pages:
the seeded Phase 5 HierText validation page sample the baselines' page profiles used, one page at a
time (batch 1), the first 10 discarded as warm-up and the next 100 timed. A timing covers a decoded
page to boxes: RGB conversion, resize, normalise, the forward pass and post-processing. The model is
the checkpoint's, on the CPU in fp32 with PyTorch's default thread count, a stand-in for the ONNX
Runtime build of Phase 17 (ADR-0011); post-processing and input size come from the config with its
``--set`` overrides. The machine should be otherwise idle, with no training running.

Each long side runs in its own process, so its peak memory is its own. Results go to
`reports/efficiency/det_profile/<experiment>_ls<N>.json`; ``--table`` writes `det_latency.md` with
the estimated page time of plan item 5: detector time plus `REC_LINE_MS` per predicted box, against
ADR-0011's page target.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Final

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
from glypharis.data.download import sha256_file
from glypharis.data.pages import open_page_image
from glypharis.models.det.detector import TextLineDetector

OUT: Final = Path("reports/efficiency/det_profile")
EVAL_SET: Final = "hiertext_val"
REC_LINE_MS: Final = 24.4  # GW-REC-024's mean CPU time per line (reports/training/phase07)
PAGE_MEAN_MS, PAGE_P95_MS = 1500.0, 2700.0  # ADR-0011, page target


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import psutil

    cfg = load_config(args.config, overrides_from_args(args))
    if cfg.det is None:
        raise SystemExit(f"{args.config}: not a detector config")
    proc = psutil.Process()
    conditions = run_conditions()  # before the model loads, so the CPU check sees an idle machine
    detector = TextLineDetector.from_checkpoint(args.checkpoint, "cpu", args.op)
    spec = dataclasses.replace(detector.spec, infer=cfg.det.infer, post=cfg.det.post)
    detector = TextLineDetector(detector.backend, spec)
    times, sizes, n_boxes = [], [], []
    pages = e2e_samples(Path(cfg.paths.data_root), EVAL_SET)
    with torch.inference_mode():
        for _, page in zip(range(WARMUP + PROFILE_SAMPLES), pages, strict=False):
            img = open_page_image(page, Path(cfg.paths.data_root))
            t0 = time.perf_counter()
            boxes = detector.detect(np.asarray(img.convert("RGB")))
            times.append(round((time.perf_counter() - t0) * 1000, 3))
            sizes.append(list(img.size))
            n_boxes.append(len(boxes))
    if len(times) < WARMUP + PROFILE_SAMPLES:
        raise SystemExit(f"{EVAL_SET} gave {len(times)} pages; profiling needs {WARMUP} warm-up "
                         f"+ {PROFILE_SAMPLES} timed")  # fmt: skip
    t = np.asarray(times[WARMUP:])
    est = t + REC_LINE_MS * np.asarray(n_boxes[WARMUP:])
    res = {
        "experiment": args.experiment, "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "long_side": spec.infer.long_side, "operating_point": spec.operating_point,
        "post": dataclasses.asdict(spec.post), "eval_set": EVAL_SET, "batch": 1,
        "warmup": WARMUP, "precision": "fp32", "runtime": f"torch {torch.__version__} (CPU)",
        "torch_threads": torch.get_num_threads(), "warmup_times_ms": times[:WARMUP],
        "times_ms": times[WARMUP:], "input_sizes": sizes[WARMUP:], "boxes": n_boxes[WARMUP:],
        "mean_ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95)),
        "est_page_mean_ms": float(est.mean()), "est_page_p95_ms": float(np.percentile(est, 95)),
        "peak_mem_mib": round(peak_working_set(proc) / 2**20, 1),
        "conditions": conditions, "cpu_busy_percent_after": cpu_busy_percent(), **git_state(),
    }  # fmt: skip
    out = args.out / f"{args.experiment}_ls{spec.infer.long_side}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"long side {spec.infer.long_side}: mean {res['mean_ms']:.0f} ms, p95 "
          f"{res['p95_ms']:.0f} ms; estimated page {res['est_page_mean_ms']:.0f} / "
          f"{res['est_page_p95_ms']:.0f} ms; peak {res['peak_mem_mib']} MiB")  # fmt: skip
    return res


def eligible(r: dict[str, Any]) -> bool:
    return bool(r["est_page_mean_ms"] <= PAGE_MEAN_MS and r["est_page_p95_ms"] <= PAGE_P95_MS)


def table(out: Path) -> str:
    rows = sorted(
        (json.loads(p.read_text(encoding="utf-8")) for p in out.glob("*_ls*.json")),
        key=lambda r: (r["experiment"], r["long_side"]),
    )
    md = [
        "# Detector CPU latency by input resolution",
        "",
        "Generated by `scripts/det_profile.py`; do not edit by hand. The baselines' page protocol",
        f"and pages: HierText validation, batch 1, {WARMUP} warm-up then {PROFILE_SAMPLES} timed",
        "pages,",
        "on the CPU in PyTorch (fp32, default threads) until the ONNX export of Phase 17. A timing",
        "covers a decoded page to boxes. Estimated page time = detector time + "
        f"{REC_LINE_MS} ms (GW-REC-024's mean line time) per predicted box; ADR-0011's page target",
        f"is a mean of at most {PAGE_MEAN_MS / 1000:.1f} s and a p95 of at most "
        f"{PAGE_P95_MS / 1000:.1f} s.",
        "",
        "| Experiment | Long side | Det ms mean | Det ms p95 | Boxes per page | "
        "Est. page ms mean | Est. page ms p95 | Peak RAM MiB | Threads | "
        "CPU busy % before/after | Eligible |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        busy = f"{r['conditions'].get('cpu_busy_percent_before')} / {r['cpu_busy_percent_after']}"
        md.append(f"| {r['experiment']} | {r['long_side']} | {r['mean_ms']:.0f} | "
                  f"{r['p95_ms']:.0f} | {np.mean(r['boxes']):.1f} | {r['est_page_mean_ms']:.0f} | "
                  f"{r['est_page_p95_ms']:.0f} | {r['peak_mem_mib']:,.0f} | {r['torch_threads']} | "
                  f"{busy} | {'yes' if eligible(r) else 'no'} |")  # fmt: skip
    plans = sorted({str(r["conditions"].get("power_plan")) for r in rows})
    md += ["", f"Power plan: {'; '.join(plans) or '-'}.", ""]
    text = "\n".join(md)
    (out.parent / "det_latency.md").write_text(text, encoding="utf-8", newline="\n")
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--op", default=None, help="operating point (default: stored)")
    parser.add_argument("--experiment", default="GW-EFF-001")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.checkpoint:
        torch.set_grad_enabled(False)
        profile(args)
    if args.table:
        print(table(args.out))
    if not args.checkpoint and not args.table:
        parser.error("give --checkpoint (with --config), --table or both")
    return 0


if __name__ == "__main__":
    sys.exit(main())
