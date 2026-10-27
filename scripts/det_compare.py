"""Paired comparison of two detector runs on the full validation set (H9, H10; spec §10).

    python scripts/det_compare.py --a runs/GW-DET-001 --b runs/GW-DET-002 \
        [--out reports/training/GW-DET-001_vs_GW-DET-002.md]

Reads each run's per-page full-validation records of best.pt (``val_full_best_pages.jsonl``) and
its ``eval_full_best.json``, and computes F1@0.5(A) − F1@0.5(B) with a 95% CI from resampling the
same pages for both arms (`paired_ratios_diff_ci`, 1,000 resamples; F1 = 2tp / (2tp + fp + fn)).
The first row compares each arm at its **stored** operating point, the one chosen on the per-epoch
subset and saved in best.pt: these are the H9 and H10 numbers. The other rows compare both arms at
each common operating point, for transparency. The full-set arg-max is never used.

Prints a markdown table for research/ablations.md (and writes it with ``--out``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from glypharis.eval.stats import paired_ratios_diff_ci
from glypharis.training.det_eval import DetPageRecord, interval_dict, read_records


def _arm(run_dir: Path) -> tuple[dict[str, Any], dict[str, DetPageRecord]]:
    ev_path = run_dir / "eval_full_best.json"
    if not ev_path.exists():
        raise SystemExit(f"{run_dir}: no eval_full_best.json (full validation of best.pt)")
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    recs = {
        r["doc_id"]: DetPageRecord.from_dict(r)
        for r in read_records(run_dir / "val_full_best_pages.jsonl")
    }
    if not recs:
        raise SystemExit(f"{run_dir}: no val_full_best_pages.jsonl")
    return ev, recs


def _f1_parts(recs: list[DetPageRecord], op: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([r.det[op][0.5] for r in recs], dtype=np.float64)
    return 2 * a[:, 0], 2 * a[:, 0] + a[:, 1] + a[:, 2]


def compare(a_dir: Path, b_dir: Path) -> dict[str, Any]:
    """Rows of paired F1@0.5 differences: stored operating points first, then common ones."""
    ev_a, ra = _arm(a_dir)
    ev_b, rb = _arm(b_dir)
    if set(ra) != set(rb):
        raise SystemExit(f"the runs were scored on different pages ({len(ra)} vs {len(rb)})")
    docs = sorted(ra)
    recs_a, recs_b = [ra[d] for d in docs], [rb[d] for d in docs]
    pairs = [("stored", ev_a["stored_op"], ev_b["stored_op"])]
    pairs += [(f"both at {op}", op, op) for op in ev_a["ops"] if op in ev_b["ops"]]
    rows = []
    for label, op_a, op_b in pairs:
        num_a, den_a = _f1_parts(recs_a, op_a)
        num_b, den_b = _f1_parts(recs_b, op_b)
        diff = paired_ratios_diff_ci(num_a, den_a, num_b, den_b, "pages")
        rows.append(
            {
                "comparison": label,
                "op_a": op_a,
                "op_b": op_b,
                "f1_a": float(num_a.sum() / den_a.sum()) if den_a.sum() else 0.0,
                "f1_b": float(num_b.sum() / den_b.sum()) if den_b.sum() else 0.0,
                "diff": interval_dict(diff),
            }
        )
    id_a = a_dir.name
    id_b = b_dir.name
    return {"a": id_a, "b": id_b, "pages": len(docs), "rows": rows}


def markdown(res: dict[str, Any]) -> str:
    a, b = res["a"], res["b"]
    lines = [
        f"Paired page-bootstrap comparison of `best.pt`, {a} minus {b}, on {res['pages']:,} "
        "validation pages (95% CI, 1,000 resamples). The first row uses each arm's stored "
        "operating point (the H9/H10 numbers); the others are for transparency.",
        "",
        f"| Comparison | Operating point {a} | Operating point {b} | F1@0.5 {a} | F1@0.5 {b} "
        "| Difference [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        d = r["diff"]
        lines.append(
            f"| {r['comparison']} | {r['op_a']} | {r['op_b']} | {r['f1_a']:.4f} | "
            f"{r['f1_b']:.4f} | {d['value']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}] |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, required=True, help="run directory of arm A")
    parser.add_argument("--b", type=Path, required=True, help="run directory of arm B")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    res = compare(args.a, args.b)
    text = markdown(res)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        args.out.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
