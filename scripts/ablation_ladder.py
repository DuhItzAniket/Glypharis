"""Phase 13 ablation ladder and final-model report (docs/PHASE_STATUS.md, Phase 13).

    python scripts/ablation_ladder.py ladder --runs runs ../Glypharis/runs --out reports/ablation
    python scripts/ablation_ladder.py decide --out reports/ablation      # prints A, B or ask
    python scripts/ablation_ladder.py final --option A --runs ... --out reports/ablation

``ladder`` compares each row of `LADDER` with the row its configuration is built on (`PARENT`):
HierText validation CER, paired over pages, each row pooling its three seeds, with Phase 7's rule
(CI on one side of zero and every seed beyond every parent seed). GLYPH-C mean corruption error is
compared the same way, paired over its 6,000 source lines; the document slice (Phase 12) is
reported. The registered decisions:

- a row is **kept** when it helps on HierText validation CER; row L5 (wider rotation) is kept when
  it helps on GLYPH-C mCE and does not hurt on HierText validation CER;
- the final configuration is L4's recipe (option A), with L5's rotation if L5 is kept (option B);
  if L4 does not help over L3, nothing is chosen automatically (``ask``: the owner decides).

``final`` reports the chosen recipe at the full budget (30 epochs, three seeds) against the CRNN
baseline at the same budget (GW-REC-001, one seed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glyph_c_eval
import synth_study as ss

LADDER: dict[str, list[str]] = {
    "L0 baseline (CRNN, no augmentation)": ["GW-REC-015", "GW-DATA-001", "GW-DATA-002"],
    "L1 + augmentation (full preset)": ["GW-REC-003", "GW-REC-004", "GW-REC-005"],
    "L2 + 25K GlyphSynth lines": ["GW-DATA-010", "GW-DATA-011", "GW-DATA-012"],
    "L3 + input height 48": ["GW-REC-006", "GW-REC-025", "GW-REC-026"],
    "L4 full model (+ SVTR encoder, weight decay 1e-2)": ["GW-REC-022", "GW-REC-027", "GW-REC-028"],
    "L5 + wider rotation": ["GW-REC-029", "GW-REC-030", "GW-REC-031"],
}
ROWS = list(LADDER)
PARENT = {ROWS[1]: ROWS[0], ROWS[2]: ROWS[1], ROWS[3]: ROWS[1], ROWS[4]: ROWS[3], ROWS[5]: ROWS[4]}
FINAL = {
    "A": ["GW-REC-024", "GW-REC-032", "GW-REC-033"],
    "B": ["GW-REC-034", "GW-REC-035", "GW-REC-036"],
}
BASELINE_FULL = "GW-REC-001"
SEED = 20261030


def mce_cells(ids: list[str], roots: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """(errors[seed, line, cell], refs[line, cell]) of GLYPH-C for ``ids``."""
    cells = [glyph_c_eval._per_line_cells(ss.find_run(i, roots, "glyph_c.npz")) for i in ids]
    lines = cells[0][0]
    if any(c[0] != lines for c in cells) or any(
        not np.array_equal(c[2], cells[0][2]) for c in cells
    ):
        raise SystemExit(f"{ids}: GLYPH-C scored on different lines or references")
    return np.array([c[1] for c in cells]), cells[0][2]


def _mce(err: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(err.sum(axis=0) / np.maximum(ref.sum(axis=0), 1)))


def mce_verdict(arm: list[str], ctl: list[str], roots: list[Path]) -> dict[str, Any]:
    a_err, ref = mce_cells(arm, roots)
    c_err, ref_c = mce_cells(ctl, roots)
    if not np.array_equal(ref, ref_c):
        raise SystemExit("GLYPH-C references differ between rows")
    a_seeds = [_mce(e, ref) for e in a_err]
    c_seeds = [_mce(e, ref) for e in c_err]
    a_mean, c_mean = a_err.mean(axis=0), c_err.mean(axis=0)
    rng = np.random.default_rng(SEED)
    n = ref.shape[0]
    diffs = []
    for _ in range(1000):
        s = rng.integers(0, n, n)
        diffs.append(_mce(a_mean[s], ref[s]) - _mce(c_mean[s], ref[s]))
    lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
    better, worse = max(a_seeds) < min(c_seeds), min(a_seeds) > max(c_seeds)
    v = "helps" if hi < 0 and better else "hurts" if lo > 0 and worse else "no clear effect"
    return {
        "value": float(np.mean(a_seeds)),
        "control": float(np.mean(c_seeds)),
        "seeds": a_seeds,
        "control_seeds": c_seeds,
        "diff": {"value": float(np.mean(a_seeds) - np.mean(c_seeds)), "low": lo, "high": hi},
        "verdict": v,
    }


def _hier(ids: list[str], roots: list[Path], pages: set[str] | None) -> tuple[Any, ...]:
    return ss.stack([ss.hiertext_units(ss.find_run(i, roots), pages) for i in ids])


def ladder(roots: list[Path], out: Path, data_root: Path) -> dict[str, Any]:
    pages = ss.doc_pages(data_root / f"{ss.HIERTEXT_VAL}.index.jsonl")
    val = {r: _hier(ids, roots, None) for r, ids in LADDER.items()}
    doc = {r: _hier(ids, roots, pages) for r, ids in LADDER.items()}
    res: dict[str, Any] = {"rows": {}}
    for r, ids in LADDER.items():
        _, ve, vr = val[r]
        _, de, dr = doc[r]
        a_err, ref = mce_cells(ids, roots)
        row: dict[str, Any] = {
            "runs": ids,
            "val_cer": float(np.mean(ve.sum(axis=1) / vr.sum())),
            "val_seeds": (ve.sum(axis=1) / vr.sum()).tolist(),
            "doc_cer": float(np.mean(de.sum(axis=1) / dr.sum())),
            "mce": float(np.mean([_mce(e, ref) for e in a_err])),
        }
        if r in PARENT:
            p = PARENT[r]
            row["parent"] = p
            row["val_vs_parent"] = ss.verdict(ve, val[p][1], vr, "pages")
            row["doc_vs_parent"] = ss.verdict(de, doc[p][1], dr, "pages")
            row["mce_vs_parent"] = mce_verdict(ids, LADDER[p], roots)
            if r == ROWS[5]:
                row["kept"] = bool(
                    row["mce_vs_parent"]["verdict"] == "helps"
                    and row["val_vs_parent"]["verdict"] != "hurts"
                )
            else:
                row["kept"] = row["val_vs_parent"]["verdict"] == "helps"
        res["rows"][r] = row
    l4, l5 = res["rows"][ROWS[4]], res["rows"][ROWS[5]]
    res["final_option"] = "ask" if not l4["kept"] else ("B" if l5["kept"] else "A")
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase13_ladder.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "phase13_ladder.md").write_text(render_ladder(res), encoding="utf-8", newline="\n")
    print(f"final option: {res['final_option']}")
    return res


def _p(x: float) -> str:
    return f"{100 * x:.2f}"


def _d(r: dict[str, Any]) -> str:
    d = r["diff"]
    return f"{100 * d['value']:+.2f} [{100 * d['low']:+.2f}, {100 * d['high']:+.2f}] {r['verdict']}"


def render_ladder(res: dict[str, Any]) -> str:
    md = [
        "# Phase 13: ablation ladder (GW-REC-037)",
        "",
        "Generated by `scripts/ablation_ladder.py ladder`; do not edit by hand. CRNN short budget",
        "(8 epochs) up to L3, the SVTR-style model at the same budget from L4; three seeds per",
        "row.",
        "Each row against the row its configuration is built on, pooled over seeds: HierText",
        "validation CER and the document slice paired over pages, GLYPH-C mCE over source lines,",
        "95% CI, 1,000 resamples; verdict: CI on one side of zero and every seed beyond every",
        "parent seed. CER % on the strict track.",
        "",
        "| Row | Runs | Val CER % | Δ val vs parent | Document slice % | Δ doc | GLYPH-C mCE % | "
        "Δ mCE | Kept |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r, row in res["rows"].items():
        if "parent" in row:
            md.append(
                f"| {r} | {', '.join(row['runs'])} | {_p(row['val_cer'])} | "
                f"{_d(row['val_vs_parent'])} | {_p(row['doc_cer'])} | "
                f"{_d(row['doc_vs_parent'])} | {_p(row['mce'])} | {_d(row['mce_vs_parent'])} | "
                f"{'yes' if row['kept'] else 'no'} |"
            )
        else:
            md.append(
                f"| {r} | {', '.join(row['runs'])} | {_p(row['val_cer'])} | — | "
                f"{_p(row['doc_cer'])} | — | {_p(row['mce'])} | — | start |"
            )
    md += [
        "",
        "Parents: L1 on L0, L2 on L1, L3 on L1, L4 on L3, L5 on L4. "
        f"Final configuration: option **{res['final_option']}** "
        "(A: L4's recipe; B: L4 with L5's rotation; ask: the owner decides).",
        "",
    ]
    return "\n".join(md)


def final(option: str, roots: list[Path], out: Path, data_root: Path) -> dict[str, Any]:
    ids = FINAL[option]
    pages = ss.doc_pages(data_root / f"{ss.HIERTEXT_VAL}.index.jsonl")
    _, ve, vr = _hier(ids, roots, None)
    _, be, _ = _hier([BASELINE_FULL], roots, None)
    _, de, dr = _hier(ids, roots, pages)
    _, bde, _ = _hier([BASELINE_FULL], roots, pages)
    a_err, ref = mce_cells(ids, roots)
    res = {
        "option": option,
        "runs": ids,
        "baseline": BASELINE_FULL,
        "val_seeds": (ve.sum(axis=1) / vr.sum()).tolist(),
        "val_cer": float(np.mean(ve.sum(axis=1) / vr.sum())),
        "val_vs_baseline": ss.verdict(ve, be, vr, "pages"),
        "doc_cer": float(np.mean(de.sum(axis=1) / dr.sum())),
        "doc_vs_baseline": ss.verdict(de, bde, dr, "pages"),
        "mce_seeds": [_mce(e, ref) for e in a_err],
    }
    res["mce"] = float(np.mean(res["mce_seeds"]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase13_final.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    md = [
        "# Phase 13: the final recognizer at the full budget (GW-REC-037)",
        "",
        "Generated by `scripts/ablation_ladder.py final`; do not edit by hand. 30 epochs, three",
        f"seeds, option {option}; against the CRNN baseline {BASELINE_FULL} at the same budget",
        "(one seed), paired over pages, 95% CI. CER % on the strict track.",
        "",
        "| Runs | Val CER % (per seed) | Val CER % (mean) | Δ vs baseline | Document slice % | "
        "Δ doc | GLYPH-C mCE % (per seed) |",
        "|---|---|---|---|---|---|---|",
        f"| {', '.join(ids)} | {', '.join(_p(x) for x in res['val_seeds'])} | "
        f"{_p(res['val_cer'])} | "
        f"{_d(res['val_vs_baseline'])} | {_p(res['doc_cer'])} | {_d(res['doc_vs_baseline'])} | "
        f"{', '.join(_p(x) for x in res['mce_seeds'])} |",
        "",
    ]
    (out / "phase13_final.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print(f"final {option}: val CER {res['val_cer']:.4f}")
    return res


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("ladder", "final"):
        s = sub.add_parser(name)
        s.add_argument("--runs", type=Path, nargs="+", default=[Path("runs")])
        s.add_argument("--out", type=Path, default=Path("reports/ablation"))
        s.add_argument("--data-root", type=Path, default=Path("data"))
        if name == "final":
            s.add_argument("--option", choices=sorted(FINAL), required=True)
    d = sub.add_parser("decide")
    d.add_argument("--out", type=Path, default=Path("reports/ablation"))
    args = parser.parse_args(argv)
    if args.cmd == "ladder":
        ladder(args.runs, args.out, args.data_root)
    elif args.cmd == "decide":
        print(
            json.loads((args.out / "phase13_ladder.json").read_text(encoding="utf-8"))[
                "final_option"
            ]
        )
    else:
        final(args.option, args.runs, args.out, args.data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
