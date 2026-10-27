"""Paired F1@0.5 difference of two detector settings on the same pages (Phase 9 rule 2).

    python scripts/det_pair.py \
        --a runs/phase09/sweep_B.npz --setting-a '[0.25, 1, "inverse11", 0.5]' \
        --b runs/phase09/sweep_B.npz --setting-b '[0.3, 3, "inverse10", 0.6]' \
        --label-a GW-DET-004 --label-b record --out reports/error_cases/detector/gw-det-004_B.json

Each arm is either a sweep file from `scripts/det_sweep.py` (``.npz``, one grid setting, given as
its JSON list) or per-page counts from `scripts/det_eval.py --pages-out` (``.jsonl``, one
operating point, ``--op-a`` / ``--op-b``). Both arms must cover the same pages. F1 = 2tp / (2tp +
fp + fn), and the 95% CI of F1(A) − F1(B) comes from resampling those pages for both arms at
once (`paired_ratios_diff_ci`, 1,000 resamples).
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


def load_arm(path: Path, setting: str | None, op: str | None) -> dict[str, tuple[int, int, int]]:
    """doc_id -> (tp, fp, fn) at IoU 0.5."""
    if path.suffix == ".npz":
        if setting is None:
            raise SystemExit(f"{path}: give the grid setting of this arm")
        z = np.load(path)
        settings = [json.loads(str(s)) for s in z["settings"]]
        want = json.loads(setting)
        if want not in settings:
            raise SystemExit(f"{path}: setting {want} is not in the grid")
        k = settings.index(want)
        out: dict[str, tuple[int, int, int]] = {}
        for d, c in zip(z["doc_ids"], z["counts"], strict=True):
            out[str(d)] = (int(c[k][0]), int(c[k][1]), int(c[k][2]))
        return out
    if op is None:
        raise SystemExit(f"{path}: give the operating point of this arm")
    recs = [DetPageRecord.from_dict(r) for r in read_records(path)]
    if not recs:
        raise SystemExit(f"{path}: no page records")
    return {r.doc_id: r.det[op][0.5] for r in recs}


def pair(a: dict[str, tuple[int, int, int]], b: dict[str, tuple[int, int, int]]) -> dict[str, Any]:
    if set(a) != set(b):
        raise SystemExit(f"the arms cover different pages ({len(a)} vs {len(b)})")
    docs = sorted(a)
    ca = np.array([a[d] for d in docs], dtype=np.float64)
    cb = np.array([b[d] for d in docs], dtype=np.float64)

    def parts(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return 2 * c[:, 0], 2 * c[:, 0] + c[:, 1] + c[:, 2]

    na, da = parts(ca)
    nb, db = parts(cb)
    diff = paired_ratios_diff_ci(na, da, nb, db, "pages")

    def summary(c: np.ndarray) -> dict[str, float]:
        tp, fp, fn = c.sum(axis=0)
        return {"tp": int(tp), "fp": int(fp), "fn": int(fn),
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                "f1_50": float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0}  # fmt: skip

    return {"pages": len(docs), "a": summary(ca), "b": summary(cb), "diff": interval_dict(diff),
            "adopt": bool(diff.low > 0)}  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for arm in ("a", "b"):
        parser.add_argument(f"--{arm}", type=Path, required=True)
        parser.add_argument(f"--setting-{arm}", default=None, help="grid setting (.npz arm)")
        parser.add_argument(f"--op-{arm}", default=None, help="operating point (.jsonl arm)")
        parser.add_argument(f"--label-{arm}", default=arm.upper())
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    res = pair(load_arm(args.a, args.setting_a, args.op_a), load_arm(args.b, args.setting_b,
                                                                       args.op_b))  # fmt: skip
    res = {
        "label_a": args.label_a, "label_b": args.label_b,
        "arm_a": {"file": args.a.as_posix(), "setting": args.setting_a, "op": args.op_a},
        "arm_b": {"file": args.b.as_posix(), "setting": args.setting_b, "op": args.op_b},
        **res,
    }  # fmt: skip
    text = json.dumps(res, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    d = res["diff"]
    print(f"{args.label_a} {res['a']['f1_50']:.4f} vs {args.label_b} {res['b']['f1_50']:.4f} on "
          f"{res['pages']} pages: {d['value']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]"
          f"{' -> adopt' if res['adopt'] else ' -> keep B'}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
