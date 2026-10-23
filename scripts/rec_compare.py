"""Compare two recognizer runs line by line on their shared validation set.

    python scripts/rec_compare.py runs/GW-REC-002 runs/GW-REC-001

Reads each run's per-line validation results for its selected checkpoint
(``val_best_samples.jsonl``) and pairs the lines by key. The two runs must have been validated on
the same lines with the same references. It then reports each run's CER and WER (strict track)
and the paired difference, first run minus second, with a 95% bootstrap CI over lines. Both runs
are scored on the same resampled lines (EXPERIMENT_PROTOCOL §4). It also counts the lines on which
each run makes fewer character errors. Writes ``reports/training/compare/<first>_vs_<second>.md``
and ``.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from glypharis.eval.stats import Interval, paired_ratio_diff_ci, ratio_ci
from glypharis.eval.text_metrics import score_pair

LOGGED_TOL = 1e-9  # the recomputed CER must be the one the run logged for its selected epoch
SUMMARY_FIELDS = ("experiment_id", "config_hash", "best_epoch", "steps", "train_seconds",
                  "train_lines", "parameters")  # fmt: skip


@dataclass
class Paired:
    """Two runs scored on the same validation lines; differences are first minus second."""

    first: dict[str, Any]  # run summaries
    second: dict[str, Any]
    lines: int
    cer: tuple[Interval, Interval]
    wer: tuple[Interval, Interval]
    d_cer: Interval
    d_wer: Interval
    fewer_errors: tuple[int, int]  # lines on which each run makes fewer character errors


def load(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    lines = (run_dir / "val_best_samples.jsonl").read_text(encoding="utf-8").splitlines()
    samples = {s["key"]: s for s in (json.loads(x) for x in lines if x.strip())}
    return summary, samples


def pair_runs(first: Path, second: Path) -> Paired:
    (sum_a, a), (sum_b, b) = load(first), load(second)
    ida, idb = sum_a["experiment_id"], sum_b["experiment_id"]
    if set(a) != set(b):
        raise SystemExit(f"{ida} and {idb} were validated on different lines")
    keys = sorted(a)
    if any(a[k]["label"] != b[k]["label"] for k in keys):
        raise SystemExit(f"{ida} and {idb} have different references for the same line")
    cols = np.zeros((6, len(keys)))  # chars, words, then errors of a and b for each
    for i, k in enumerate(keys):
        sa, sb = score_pair(a[k]["pred"], a[k]["label"]), score_pair(b[k]["pred"], b[k]["label"])
        cols[:, i] = (sa.ref_chars, sa.ref_words, sa.char_errors, sb.char_errors,
                      sa.word_errors, sb.word_errors)  # fmt: skip
    chars, words, ce_a, ce_b, we_a, we_b = cols
    cer_a, cer_b = ratio_ci(ce_a, chars, "lines"), ratio_ci(ce_b, chars, "lines")
    for s, cer in ((sum_a, cer_a), (sum_b, cer_b)):
        if abs(cer.value - s["best"]["val_cer"]) > LOGGED_TOL:
            raise SystemExit(
                f"{s['experiment_id']}: recomputed CER {cer.value} is not the logged "
                f"{s['best']['val_cer']}; are these the selected checkpoint's lines?"
            )
    return Paired(
        first=sum_a,
        second=sum_b,
        lines=len(keys),
        cer=(cer_a, cer_b),
        wer=(ratio_ci(we_a, words, "lines"), ratio_ci(we_b, words, "lines")),
        d_cer=paired_ratio_diff_ci(ce_a, ce_b, chars, "lines"),
        d_wer=paired_ratio_diff_ci(we_a, we_b, words, "lines"),
        fewer_errors=(int((ce_a < ce_b).sum()), int((ce_b < ce_a).sum())),
    )


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def diff(iv: Interval) -> str:
    return f"{100 * iv.value:+.2f} [{100 * iv.low:+.2f}, {100 * iv.high:+.2f}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("first", type=Path, help="run directory, e.g. runs/GW-REC-002")
    parser.add_argument("second", type=Path, help="run directory to compare against")
    parser.add_argument("--out", type=Path, default=Path("reports/training/compare"))
    args = parser.parse_args(argv)
    p = pair_runs(args.first, args.second)
    ida, idb = p.first["experiment_id"], p.second["experiment_id"]
    a_better, b_better = p.fewer_errors
    result = {
        "lines": p.lines, "track": "strict",
        "first": {k: p.first.get(k) for k in SUMMARY_FIELDS},
        "second": {k: p.second.get(k) for k in SUMMARY_FIELDS},
        "cer": {"first": vars(p.cer[0]), "second": vars(p.cer[1]), "difference": vars(p.d_cer)},
        "wer": {"first": vars(p.wer[0]), "second": vars(p.wer[1]), "difference": vars(p.d_wer)},
        "lines_fewer_char_errors": {"first": a_better, "second": b_better,
                                    "tied": p.lines - a_better - b_better},
    }  # fmt: skip

    def run_row(s: dict[str, Any], cer: Interval, wer: Interval) -> str:
        return (f"| {s['experiment_id']} | {s['best_epoch']} | {s['steps']:,} | "
                f"{s['train_lines']:,} | {s['train_seconds'] / 3600:.2f} | {pct(cer.value)} "
                f"[{pct(cer.low)}, {pct(cer.high)}] | {pct(wer.value)} "
                f"[{pct(wer.low)}, {pct(wer.high)}] |")  # fmt: skip

    md = [
        f"# {ida} vs {idb}",
        "",
        "Generated by `scripts/rec_compare.py`; do not edit by hand.",
        "",
        f"Validation lines: {p.lines:,}, the same lines and references for both runs; strict "
        "track; each run's selected checkpoint. Intervals are 95% bootstrap CIs over lines "
        f"({p.d_cer.resamples} resamples); the differences resample the same lines for both runs.",
        "",
        "| Run | Selected epoch | Steps | Training lines | GPU-hours | Val CER % | Val WER % |",
        "|---|---|---|---|---|---|---|",
        run_row(p.first, *(p.cer[0], p.wer[0])),
        run_row(p.second, *(p.cer[1], p.wer[1])),
        "",
        f"- CER difference ({ida} minus {idb}), percentage points: {diff(p.d_cer)}",
        f"- WER difference ({ida} minus {idb}), percentage points: {diff(p.d_wer)}",
        f"- Lines with fewer character errors: {ida} {a_better:,}, {idb} {b_better:,}, tied "
        f"{p.lines - a_better - b_better:,}",
        "",
    ]  # fmt: skip
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"{ida}_vs_{idb}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    stem.with_suffix(".md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print("\n".join(md[8:13]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
