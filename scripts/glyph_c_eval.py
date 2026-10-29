"""GLYPH-C robustness of recognizer runs, and the Phase 11 comparison (GW-DATA-009).

    python scripts/glyph_c_eval.py score --run runs/GW-REC-015          # one run, GPU if present
    python scripts/glyph_c_eval.py compare --arms phase11 --out reports/robustness

``score`` reads every crop of the frozen GLYPH-C v1 pack (6,000 validation lines, each clean and
under 7 corruptions at 5 levels) with the run's ``best.pt`` through `TextRecognizer` and stores, per
crop, its character errors and reference characters on the strict track
(``runs/<id>/glyph_c.npz``) and a summary (``reports/robustness/<id>.json``): CER on the clean
copies, CER per corruption and level, and the **mean corruption error (mCE)**: the unweighted mean,
over the 35 corruption-level cells, of each cell's micro CER (registered in docs/PHASE_STATUS.md,
Phase 11).

``compare`` applies Phase 11's registered decision rule to the five arms (three seeds each): for
each arm against A (no augmentation), the paired 95% CI of mCE(arm) − mCE(A), resampling the 6,000
source lines with each arm's per-line errors pooled over its three seeds, and whether all three of
the arm's seeds lie on the same side of all three of A's. Verdicts: helps (CI below 0 and every
seed better), hurts (the mirror), no clear effect otherwise. H3 adds the clean guard: arm D's mean
clean HierText validation CER at most 5% (relative) above A's. Validation-derived data only.
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
from glypharis.data.linepack import LinePackReader
from glypharis.data.text import normalize_eval
from glypharis.eval.text_metrics import char_distance
from glypharis.pipeline.recognizer import TextRecognizer

PACK = "processed/glyph_c/lines"
CHUNK = 1024
RESAMPLES = 1000
SEED = 20261028
ARMS: dict[str, dict[str, list[str]]] = {
    "phase11": {
        "A (none)": ["GW-REC-015", "GW-DATA-001", "GW-DATA-002"],
        "B (geometric)": ["GW-REC-016", "GW-DATA-003", "GW-DATA-004"],
        "C (photometric)": ["GW-REC-017", "GW-DATA-005", "GW-DATA-006"],
        "D (degradation)": ["GW-REC-018", "GW-DATA-007", "GW-DATA-008"],
        "E (full)": ["GW-REC-003", "GW-REC-004", "GW-REC-005"],
    }
}


def parse_key(key: str) -> tuple[str, str, int]:
    line, corruption, level = key.rsplit("|", 2)
    return line, corruption, int(level)


def score_run(run: Path, out: Path) -> dict[str, Any]:
    cfg = load_config(run / "config.resolved.yaml")
    root = Path(cfg.paths.data_root)
    reader = LinePackReader(root / PACK)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rec = TextRecognizer.from_run(run, device=device)
    n = len(reader)
    errors = np.zeros(n, dtype=np.int64)
    refs = np.zeros(n, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, n, CHUNK):
            idx = range(start, min(start + CHUNK, n))
            reads = rec.read([reader.image(i) for i in idx])
            for i, r in zip(idx, reads, strict=True):
                ref = normalize_eval(reader.records[i].label, "strict")
                errors[i] = char_distance(normalize_eval(r.text, "strict"), ref)
                refs[i] = len(ref)
    keys = [parse_key(r.key) for r in reader.records]
    np.savez_compressed(
        run / "glyph_c.npz",
        errors=errors,
        refs=refs,
        lines=np.array([k[0] for k in keys]),
        corruption=np.array([k[1] for k in keys]),
        level=np.array([k[2] for k in keys]),
    )
    res = summarise(errors, refs, keys)
    res = {
        "run": run.name,
        "pack_sha256": reader.meta["bin_sha256"],
        "glyph_c_version": reader.meta["glyph_c_version"],
        "crops": n,
        **res,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run.name}.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{run.name}: clean CER {res['clean_cer']:.4f}, mCE {res['mce']:.4f}")
    return res


def summarise(
    errors: np.ndarray, refs: np.ndarray, keys: list[tuple[str, str, int]]
) -> dict[str, Any]:
    cells: dict[str, dict[int, list[int]]] = {}
    clean = [i for i, (_, c, _) in enumerate(keys) if c == "clean"]
    for i, (_, c, lv) in enumerate(keys):
        if c != "clean":
            cells.setdefault(c, {}).setdefault(lv, []).append(i)
    per = {
        c: {
            str(lv): float(errors[ix].sum() / max(refs[ix].sum(), 1))
            for lv, ix in sorted(levels.items())
        }
        for c, levels in sorted(cells.items())
    }
    mce = float(np.mean([v for levels in per.values() for v in levels.values()]))
    return {
        "clean_cer": float(errors[clean].sum() / max(refs[clean].sum(), 1)),
        "cer_per_corruption_level": per,
        "mce": mce,
    }


def _per_line_cells(run: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(source lines, errors[line, cell], refs[line, cell]) over the 35 corrupted cells."""
    z = np.load(run / "glyph_c.npz")
    lines = sorted(set(z["lines"].tolist()))
    pos = {x: i for i, x in enumerate(lines)}
    cells = sorted(
        {(c, int(lv)) for c, lv in zip(z["corruption"], z["level"], strict=True) if c != "clean"}
    )
    cpos = {c: j for j, c in enumerate(cells)}
    err = np.zeros((len(lines), len(cells)))
    ref = np.zeros((len(lines), len(cells)))
    for x, c, lv, e, r in zip(
        z["lines"], z["corruption"], z["level"], z["errors"], z["refs"], strict=True
    ):
        if c != "clean":
            err[pos[x], cpos[(c, int(lv))]] += e
            ref[pos[x], cpos[(c, int(lv))]] += r
    return lines, err, ref


def mce_of(err: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(err.sum(axis=0) / np.maximum(ref.sum(axis=0), 1)))


def _cells(run: Path) -> list[tuple[str, int]]:
    """The corrupted cells in the column order of `_per_line_cells`."""
    z = np.load(run / "glyph_c.npz")
    return sorted(
        {(c, int(lv)) for c, lv in zip(z["corruption"], z["level"], strict=True) if c != "clean"}
    )


def per_corruption(err: np.ndarray, ref: np.ndarray, cells: list[tuple[str, int]]) -> dict:
    """Mean over levels of each corruption's micro CER, and the micro CER at level 5."""
    cer = err.sum(axis=0) / np.maximum(ref.sum(axis=0), 1)
    out: dict[str, dict[str, float]] = {}
    for c in sorted({c for c, _ in cells}):
        ix = [j for j, (cc, _) in enumerate(cells) if cc == c]
        top = [j for j in ix if cells[j][1] == max(cells[k][1] for k in ix)]
        out[c] = {"mean": float(cer[ix].mean()), "worst_level": float(cer[top].mean())}
    return out


def find_run(run_id: str, roots: list[Path]) -> Path:
    """The first ``<root>/<run_id>`` that has been scored (runs live in more than one checkout)."""
    for root in roots:
        if (root / run_id / "glyph_c.npz").exists():
            return root / run_id
    raise SystemExit(f"{run_id}: not scored under any of {[str(r) for r in roots]}")


def compare(arms: dict[str, list[str]], roots: list[Path], out: Path) -> dict[str, Any]:
    data: dict[str, dict[str, Any]] = {}
    lines: list[str] | None = None
    for arm, ids in arms.items():
        dirs = [find_run(i, roots) for i in ids]
        per_seed = [_per_line_cells(rd) for rd in dirs]
        lines = lines or per_seed[0][0]
        if any(p[0] != lines for p in per_seed):
            raise SystemExit(f"{arm}: the runs were scored on different lines")
        clean_val = []
        clean_glyph_c = []
        for rd in dirs:
            s = json.loads((rd / "summary.json").read_text(encoding="utf-8"))
            clean_val.append(float(s["best"]["val_cer"]))
            z = np.load(rd / "glyph_c.npz")
            keep = z["corruption"] == "clean"
            clean_glyph_c.append(float(z["errors"][keep].sum() / max(z["refs"][keep].sum(), 1)))
        data[arm] = {
            "ids": ids,
            "err": sum(p[1] for p in per_seed),
            "ref": sum(p[2] for p in per_seed),
            "seed_mce": [mce_of(p[1], p[2]) for p in per_seed],
            "clean_val": clean_val,
            "clean_glyph_c": clean_glyph_c,
            "cells": _cells(dirs[0]),
        }
    base = next(iter(arms))
    a = data[base]
    rng = np.random.default_rng(SEED)
    n = a["err"].shape[0]
    rows = []
    for arm, d in data.items():
        row: dict[str, Any] = {
            "arm": arm,
            "runs": d["ids"],
            "mce": mce_of(d["err"], d["ref"]),
            "seed_mce": d["seed_mce"],
            "clean_val_cer": d["clean_val"],
            "clean_val_cer_mean": float(np.mean(d["clean_val"])),
            "clean_glyph_c_cer_mean": float(np.mean(d["clean_glyph_c"])),
            "per_corruption": per_corruption(d["err"], d["ref"], d["cells"]),
        }
        if arm != base:
            diffs = np.empty(RESAMPLES)
            for k in range(RESAMPLES):
                s = rng.integers(0, n, n)
                diffs[k] = mce_of(d["err"][s], d["ref"][s]) - mce_of(a["err"][s], a["ref"][s])
            value = row["mce"] - mce_of(a["err"], a["ref"])
            lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
            better = max(d["seed_mce"]) < min(a["seed_mce"])
            worse = min(d["seed_mce"]) > max(a["seed_mce"])
            verdict = (
                "helps" if hi < 0 and better else "hurts" if lo > 0 and worse else "no clear effect"
            )
            row.update(diff={"value": value, "low": lo, "high": hi}, verdict=verdict)
        rows.append(row)
    d_row = next(r for r in rows if r["arm"].startswith("D"))
    guard = d_row["clean_val_cer_mean"] <= 1.05 * rows[0]["clean_val_cer_mean"]
    h3 = {
        "d_verdict": d_row["verdict"],
        "clean_guard_holds": bool(guard),
        "supported": bool(d_row["verdict"] == "helps" and guard),
    }
    res = {"base": base, "rows": rows, "h3": h3, "resamples": RESAMPLES}
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase11_comparison.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    md = [
        "# Phase 11: augmentation families on GLYPH-C (GW-DATA-009)",
        "",
        "Generated by `scripts/glyph_c_eval.py compare`; do not edit by hand. mCE: mean over the",
        "35 corruption-level cells of each cell's micro CER (strict track), GLYPH-C v1. Each arm",
        "has three seeds (1337, 2024, 7); differences against A pool the three seeds and resample",
        "the 6,000 source lines (95% CI, 1,000 resamples). Clean CER is the HierText validation",
        "CER of each run's best checkpoint.",
        "",
        "| Arm | mCE % (pooled) | mCE % per seed | Δ mCE vs A [95% CI] "
        "| Clean val CER % (mean) | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        seeds = ", ".join(f"{100 * x:.2f}" for x in r["seed_mce"])
        diff = (
            (
                f"{100 * r['diff']['value']:+.2f} [{100 * r['diff']['low']:+.2f}, "
                f"{100 * r['diff']['high']:+.2f}]"
            )
            if "diff" in r
            else "—"
        )
        md.append(
            f"| {r['arm']} | {100 * r['mce']:.2f} | {seeds} | {diff} | "
            f"{100 * r['clean_val_cer_mean']:.2f} | {r.get('verdict', 'control')} |"
        )
    md += [
        "",
        f"H3 (D lowers mCE against A, clean CER at most 5% higher): D's verdict "
        f"**{h3['d_verdict']}**, clean guard {'holds' if guard else 'fails'}: H3 "
        f"**{'supported' if h3['supported'] else 'not supported'}**.",
        "",
        "## CER per corruption (reported, not part of the decision rule)",
        "",
        "Micro CER % pooled over each arm's three seeds: the mean over the five levels, and in",
        "brackets level 5 alone. Clean GLYPH-C: the clean copies of the same 6,000 lines (mean of",
        "the three seeds).",
        "",
    ]
    corr = sorted(rows[0]["per_corruption"])
    md.append("| Arm | Clean GLYPH-C | " + " | ".join(corr) + " |")
    md.append("|---|---|" + "---|" * len(corr))
    for r in rows:
        pc = r["per_corruption"]
        cols = " | ".join(
            f"{100 * pc[c]['mean']:.2f} ({100 * pc[c]['worst_level']:.2f})" for c in corr
        )
        md.append(f"| {r['arm']} | {100 * r['clean_glyph_c_cer_mean']:.2f} | {cols} |")
    md.append("")
    (out / "phase11_comparison.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(
        f"{out / 'phase11_comparison.md'}: H3 {'supported' if h3['supported'] else 'not supported'}"
    )
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--out", type=Path, default=Path("reports/robustness"))
    c = sub.add_parser("compare")
    c.add_argument("--arms", choices=sorted(ARMS), default="phase11")
    c.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=[Path("runs")],
        help="run directories to look in, first match wins",
    )
    c.add_argument("--out", type=Path, default=Path("reports/robustness"))
    args = parser.parse_args(argv)
    if args.cmd == "score":
        score_run(args.run, args.out)
    else:
        compare(ARMS[args.arms], args.runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
