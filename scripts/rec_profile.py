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
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
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
from glypharis.data.rec_dataset import PAD_MULTIPLE, resized_width
from glypharis.data.text import Charset
from glypharis.models.rec.crnn import build_recognizer
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
    (out / f"{run}.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", default=[], help="experiment ID, e.g. GW-REC-003")
    parser.add_argument("--table", action="store_true", help="write cpu_latency.md")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
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
