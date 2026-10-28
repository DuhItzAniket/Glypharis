"""Line-confidence calibration on HierText validation lines (Phase 10, GW-E2E-003, H8).

    python scripts/rec_calibrate.py --rec-run runs/GW-REC-024 \
        --out reports/calibration/GW-E2E-003

Reads every line crop of the recognizer's validation pack with `TextRecognizer` (the pipeline's
own path) and records each line's confidence at every temperature of the grid
(`glypharis.pipeline.calibration.TEMPERATURES`) and whether it is right (exact match on the strict
track). Lines are split by page into Phase 9's halves (`calibration_halves` over the validation
pages): the temperature is fitted on half A (`fit_temperature`), and ECE (15 equal-mass bins) at
T = 1 and at the fitted T is reported on half B with a 95% line-bootstrap CI of the difference
(1,000 resamples). The fitted T is adopted if that CI lies below 0. Writes ``calibration.json``,
``calibration.md`` and a reliability diagram; per-line values go to ``--cache``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from glypharis.config.loader import load_config
from glypharis.data.det_dataset import PageIndex
from glypharis.data.linepack import LinePackReader
from glypharis.data.text import normalize_eval
from glypharis.eval.det_matching import calibration_halves
from glypharis.pipeline.calibration import BINS, TEMPERATURES, ece, fit_temperature, reliability
from glypharis.pipeline.recognizer import TextRecognizer

CHUNK = 512
RESAMPLES = 1000
SEED = 20261028
PAGES = "processed/hiertext/pages_validation.jsonl.gz"


def read_lines(rec: TextRecognizer, pack: Path) -> dict[str, Any]:
    reader = LinePackReader(pack)
    conf = np.zeros((len(reader), len(TEMPERATURES)), dtype=np.float32)
    correct = np.zeros(len(reader), dtype=bool)
    for start in range(0, len(reader), CHUNK):
        idx = range(start, min(start + CHUNK, len(reader)))
        reads = rec.read([reader.image(i) for i in idx], TEMPERATURES)
        for i, r in zip(idx, reads, strict=True):
            conf[i] = [r.at[t] for t in TEMPERATURES]
            label = reader.records[i].label
            correct[i] = normalize_eval(r.text, "strict") == normalize_eval(label, "strict")
        if (start // CHUNK) % 20 == 0:
            print(f"{start + len(idx)}/{len(reader)} lines", flush=True)
    return {
        "conf": conf,
        "correct": correct,
        "doc_ids": np.array([r.doc_id for r in reader.records]),
    }


def bootstrap_diff(c0: np.ndarray, c1: np.ndarray, y: np.ndarray) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    diffs = np.empty(RESAMPLES)
    for k in range(RESAMPLES):
        s = rng.integers(0, len(y), len(y))
        diffs[k] = ece(c1[s], y[s]) - ece(c0[s], y[s])
    return {
        "value": ece(c1, y) - ece(c0, y),
        "low": float(np.quantile(diffs, 0.025)),
        "high": float(np.quantile(diffs, 0.975)),
    }


def diagram(
    path: Path, before: list[dict[str, float]], after: list[dict[str, float]], t: float
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.4), dpi=150)
    ax.plot([0, 1], [0, 1], color="#9a9a9a", lw=1, ls="--", label="perfect calibration")
    for pts, label, colour in (
        (before, "T = 1 (raw)", "#c0392b"),
        (after, f"T = {t:g}", "#1f5fa8"),
    ):
        ax.plot(
            [p["confidence"] for p in pts],
            [p["accuracy"] for p in pts],
            marker="o",
            ms=3.5,
            lw=1.5,
            color=colour,
            label=label,
        )
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("share of lines read exactly right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability, validation half B ({BINS} equal-mass bins)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rec-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("runs/GW-E2E-003/lines.npz"))
    args = parser.parse_args(argv)
    cfg = load_config(args.rec_run / "config.resolved.yaml")
    root = Path(cfg.paths.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cache.exists():
        z = np.load(args.cache)
        data = {k: z[k] for k in ("conf", "correct", "doc_ids")}
    else:
        rec = TextRecognizer.from_run(args.rec_run, device=device)
        data = read_lines(rec, root / cfg.data.val[0])
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, **data)
    half_a, _ = calibration_halves(PageIndex.build(root / PAGES, "validation", root).doc_ids())
    in_a = np.isin(data["doc_ids"], np.array(half_a))
    conf, y = data["conf"], data["correct"].astype(np.float64)
    k1 = TEMPERATURES.index(1.0)
    fit = fit_temperature(lambda t: conf[in_a, TEMPERATURES.index(t)], y[in_a])
    t = fit["temperature"]
    kt = TEMPERATURES.index(t)
    b = ~in_a
    before, after = conf[b, k1].astype(np.float64), conf[b, kt].astype(np.float64)
    diff = bootstrap_diff(before, after, y[b])
    res = {
        "rec_run": args.rec_run.as_posix(),
        "pack": cfg.data.val[0],
        "lines": len(y),
        "lines_a": int(in_a.sum()),
        "lines_b": int(b.sum()),
        "accuracy_b": float(y[b].mean()),
        "temperature": t,
        "nll_a": fit["nll"],
        "nll_a_at_1": fit["nll_at_1"],
        "ece_b_at_1": ece(before, y[b]),
        "ece_b_at_t": ece(after, y[b]),
        "ece_diff_b": diff,
        "adopt": bool(diff["high"] < 0),
        "h8_raw_above_0_05": bool(ece(before, y[b]) > 0.05),
        "h8_halved": bool(ece(after, y[b]) <= ece(before, y[b]) / 2),
        "reliability_b_at_1": reliability(before, y[b]),
        "reliability_b_at_t": reliability(after, y[b]),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calibration.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    diagram(args.out / "reliability_B.png", res["reliability_b_at_1"], res["reliability_b_at_t"], t)
    d = res["ece_diff_b"]
    md = [
        "# Line-confidence calibration (GW-E2E-003)",
        "",
        "Generated by `scripts/rec_calibrate.py`; do not edit by hand. GW-REC-024 on every line of",
        f"`{res['pack']}` ({res['lines']:,} lines), read through the pipeline's recognizer. A line",
        "is right when its text equals the label exactly on the strict track. Temperature fitted",
        f"on half A ({res['lines_a']:,} lines) by binary NLL; ECE ({BINS} equal-mass bins) on",
        f"half B ({res['lines_b']:,} lines, accuracy {res['accuracy_b']:.3f}).",
        "",
        "| | T | ECE on half B |",
        "|---|---|---|",
        f"| raw | 1 | {res['ece_b_at_1']:.4f} |",
        f"| fitted | {t:g} | {res['ece_b_at_t']:.4f} |",
        "",
        f"Difference (fitted − raw): {d['value']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] "
        f"(95% line bootstrap). Adopted: **{'yes' if res['adopt'] else 'no'}**.",
        "",
        "![reliability](reliability_B.png)",
        "",
    ]
    (args.out / "calibration.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(
        f"T = {t}: ECE half B {res['ece_b_at_1']:.4f} -> {res['ece_b_at_t']:.4f} "
        f"({d['value']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]); adopt {res['adopt']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
