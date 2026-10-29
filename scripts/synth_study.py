"""Phase 12 synthetic data study: scoring and the registered comparison (GW-DATA-022).

    python scripts/synth_study.py score --run runs/GW-DATA-010     # GlyphSynth validation lines
    python scripts/synth_study.py compare --runs runs ../Glypharis/runs --out reports/synthetic

``score`` reads the 10,000 GlyphSynth validation lines (held-out font families) with the run's
``best.pt`` and stores per-line character errors and reference characters on the strict track in
``runs/<id>/glyphsynth_val.npz`` and a summary in ``reports/synthetic/<id>.json``.

``compare`` applies the decision rules registered in docs/PHASE_STATUS.md (Phase 12):

- Readouts. **Document slice** (primary for H2'a and the font factor): the HierText validation
  lines on pages with at least `DOC_MIN_LINES` lines in the validation pack, from each run's
  per-line results for its selected checkpoint (``val_best_samples.jsonl``). **HierText
  validation**: all lines, the same way. **GlyphSynth validation**: from ``score``. **GLYPH-C mCE**
  (the capture factor): from ``scripts/glyph_c_eval.py score``.
- An arm's value pools its seeds (the mean of the seeds' micro CER). Differences pair the units:
  pages for the HierText readouts (lines on a page are not independent), lines for GlyphSynth,
  source lines for GLYPH-C; 95% bootstrap CI, 1,000 resamples.
- **Helps** (lower is better): CI entirely below 0 and every seed of the arm below every seed of
  its control; **hurts**: the mirror; otherwise **no clear effect**.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from glypharis.eval.stats import paired_ratio_diff_ci
from glypharis.eval.text_metrics import score_pair

DOC_MIN_LINES = 100
GLYPHSYNTH_VAL = "processed/glyphsynth/lines_val"
HIERTEXT_VAL = "processed/hiertext/lines_validation"
CHUNK = 1024
CONTROL = "HierText only"
ARMS: dict[str, list[str]] = {
    CONTROL: ["GW-REC-003", "GW-REC-004", "GW-REC-005"],
    "+25K": ["GW-DATA-010", "GW-DATA-011", "GW-DATA-012"],
    "+50K": ["GW-DATA-013"],
    "+100K": ["GW-DATA-014", "GW-DATA-015", "GW-DATA-016"],
    "+250K": ["GW-DATA-017", "GW-DATA-018", "GW-DATA-019"],
    "+100K, 6 fonts": ["GW-DATA-020"],
    "+100K, no capture": ["GW-DATA-021"],
}
SIZES = {"+25K": 25_000, "+50K": 50_000, "+100K": 100_000, "+250K": 250_000}


# --------------------------------------------------------------------------------- scoring
def score_run(run: Path, out: Path) -> dict[str, Any]:
    import torch

    from glypharis.config.loader import load_config
    from glypharis.data.linepack import LinePackReader
    from glypharis.pipeline.recognizer import TextRecognizer

    cfg = load_config(run / "config.resolved.yaml")
    reader = LinePackReader(Path(cfg.paths.data_root) / GLYPHSYNTH_VAL)
    rec = TextRecognizer.from_run(run, device="cuda" if torch.cuda.is_available() else "cpu")
    n = len(reader)
    errors = np.zeros(n, dtype=np.int64)
    refs = np.zeros(n, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, n, CHUNK):
            idx = range(start, min(start + CHUNK, n))
            for i, r in zip(idx, rec.read([reader.image(i) for i in idx]), strict=True):
                s = score_pair(r.text, reader.records[i].label)
                errors[i], refs[i] = s.char_errors, s.ref_chars
    keys = np.array([r.key for r in reader.records])
    np.savez_compressed(run / "glyphsynth_val.npz", errors=errors, refs=refs, keys=keys)
    res = {
        "run": run.name,
        "lines": n,
        "pack_sha256": reader.meta["bin_sha256"],
        "glyphsynth_val_cer": float(errors.sum() / max(refs.sum(), 1)),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{run.name}.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{run.name}: GlyphSynth validation CER {res['glyphsynth_val_cer']:.4f}")
    return res


# ------------------------------------------------------------------------------- readouts
def find_run(run_id: str, roots: list[Path], needs: str = "summary.json") -> Path:
    for root in roots:
        if (root / run_id / needs).exists():
            return root / run_id
    raise SystemExit(f"{run_id}: no {needs} under any of {[str(r) for r in roots]}")


def doc_pages(index_path: Path, min_lines: int = DOC_MIN_LINES) -> set[str]:
    """Pages (key prefix before the first ``/``) with at least ``min_lines`` validation lines."""
    counts: collections.Counter[str] = collections.Counter()
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                counts[json.loads(line)["key"].split("/", 1)[0]] += 1
    return {p for p, c in counts.items() if c >= min_lines}


def hiertext_units(run: Path, pages: set[str] | None) -> dict[str, tuple[int, int]]:
    """Per page: (character errors, reference characters) of the run's selected checkpoint,
    restricted to ``pages`` (all pages when None). The total must match the logged CER."""
    per: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    tot = [0, 0]
    with open(run / "val_best_samples.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            sc = score_pair(s["pred"], s["label"])
            tot[0] += sc.char_errors
            tot[1] += sc.ref_chars
            page = s["key"].split("/", 1)[0]
            if pages is None or page in pages:
                per[page][0] += sc.char_errors
                per[page][1] += sc.ref_chars
    logged = json.loads((run / "summary.json").read_text(encoding="utf-8"))["best"]["val_cer"]
    if abs(tot[0] / max(tot[1], 1) - logged) > 1e-9:
        raise SystemExit(f"{run.name}: per-line results do not reproduce the logged CER")
    return {k: (v[0], v[1]) for k, v in per.items()}


def line_units(path: Path) -> dict[str, tuple[int, int]]:
    z = np.load(path)
    return {
        str(k): (int(e), int(r)) for k, e, r in zip(z["keys"], z["errors"], z["refs"], strict=True)
    }


def glyph_c_units(run: Path) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import glyph_c_eval

    _, err, ref = glyph_c_eval._per_line_cells(run)
    return err, ref


# ---------------------------------------------------------------------------------- rules
def stack(units: list[dict[str, tuple[int, int]]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(unit keys, errors[seed, unit], refs[unit]); every seed must cover the same units."""
    keys = sorted(units[0])
    if any(sorted(u) != keys for u in units):
        raise SystemExit("the runs were scored on different units")
    err = np.array([[u[k][0] for k in keys] for u in units], dtype=np.float64)
    ref = np.array([units[0][k][1] for k in keys], dtype=np.float64)
    if any(np.any(np.array([u[k][1] for k in keys]) != ref) for u in units):
        raise SystemExit("the runs have different references for the same unit")
    return keys, err, ref


def verdict(arm_err: np.ndarray, ctl_err: np.ndarray, ref: np.ndarray, unit: str) -> dict[str, Any]:
    """Arm against control on shared units: pooled values, paired CI, seed ranges, verdict."""
    a_seeds = (arm_err.sum(axis=1) / ref.sum()).tolist()
    c_seeds = (ctl_err.sum(axis=1) / ref.sum()).tolist()
    iv = paired_ratio_diff_ci(arm_err.mean(axis=0), ctl_err.mean(axis=0), ref, unit)
    better = max(a_seeds) < min(c_seeds)
    worse = min(a_seeds) > max(c_seeds)
    v = (
        "helps"
        if iv.high < 0 and better
        else "hurts"
        if iv.low > 0 and worse
        else "no clear effect"
    )
    return {
        "value": float(np.mean(a_seeds)),
        "control": float(np.mean(c_seeds)),
        "seeds": a_seeds,
        "control_seeds": c_seeds,
        "diff": {"value": iv.value, "low": iv.low, "high": iv.high},
        "verdict": v,
    }


def _cer(err: np.ndarray, ref: np.ndarray) -> float:
    return float(err.sum() / max(ref.sum(), 1))


def compare(
    arms: dict[str, list[str]], roots: list[Path], out: Path, data_root: Path = Path("data")
) -> dict[str, Any]:
    pages = doc_pages(data_root / f"{HIERTEXT_VAL}.index.jsonl")
    dirs = {a: [find_run(i, roots) for i in ids] for a, ids in arms.items()}
    doc = {a: stack([hiertext_units(d, pages) for d in ds]) for a, ds in dirs.items()}
    full = {a: stack([hiertext_units(d, None) for d in ds]) for a, ds in dirs.items()}
    gs = {
        a: stack(
            [
                line_units(find_run(d.name, roots, "glyphsynth_val.npz") / "glyphsynth_val.npz")
                for d in ds
            ]
        )
        for a, ds in dirs.items()
    }
    res: dict[str, Any] = {
        "doc_min_lines": DOC_MIN_LINES,
        "doc_pages": len(pages),
        "doc_lines_chars": float(doc[CONTROL][2].sum()),
        "arms": {},
    }
    for a, ids in arms.items():
        row: dict[str, Any] = {"runs": ids}
        for name, data in (("doc", doc), ("hiertext_val", full), ("glyphsynth_val", gs)):
            _, err, ref = data[a]
            row[name] = {
                "value": float(np.mean(err.sum(axis=1) / ref.sum())),
                "seeds": (err.sum(axis=1) / ref.sum()).tolist(),
            }
        if a != CONTROL:
            _, ce, cr = doc[CONTROL]
            row["doc_vs_control"] = verdict(doc[a][1], ce, cr, "pages")
            row["hiertext_val_vs_control"] = verdict(
                full[a][1], full[CONTROL][1], full[a][2], "pages"
            )
            row["glyphsynth_val_vs_control"] = verdict(gs[a][1], gs[CONTROL][1], gs[a][2], "lines")
        res["arms"][a] = row

    sizes = [a for a in SIZES if a in arms]
    helps = [a for a in sizes if res["arms"][a]["doc_vs_control"]["verdict"] == "helps"]
    curve = {a: res["arms"][a]["doc"]["value"] for a in sizes}
    res["h2a"] = {"sizes_that_help": helps, "supported": bool(helps), "doc_cer_by_size": curve}
    if {"+25K", "+100K", "+250K"} <= set(curve):
        g1 = (curve["+25K"] - curve["+100K"]) / 75  # CER points gained per 1,000 added lines
        g2 = (curve["+100K"] - curve["+250K"]) / 150
        res["h2a"]["gain_per_1k_lines"] = {
            "25K_to_100K": g1,
            "100K_to_250K": g2,
            "diminishing": bool(g2 < g1),
        }

    h2b: dict[str, Any] = {}
    if "+100K, 6 fonts" in arms and "+100K" in arms:
        # many fonts help if the few-font arm is worse than +100K on the document slice
        r = verdict(doc["+100K, 6 fonts"][1], doc["+100K"][1], doc["+100K"][2], "pages")
        h2b["fonts"] = {**r, "many_fonts_help": r["verdict"] == "hurts"}
    if "+100K, no capture" in arms and "+100K" in arms:
        cells = [
            glyph_c_units(find_run(d.name, roots, "glyph_c.npz"))
            for d in dirs["+100K"] + dirs["+100K, no capture"]
        ]
        ref = cells[0][1]
        if any(not np.array_equal(c[1], ref) for c in cells):
            raise SystemExit("GLYPH-C references differ between runs")
        mce = lambda e: float(np.mean(e.sum(axis=0) / np.maximum(ref.sum(axis=0), 1)))  # noqa: E731
        cap = [mce(c[0]) for c in cells[:-1]]
        nocap = [mce(c[0]) for c in cells[-1:]]
        rng = np.random.default_rng(20261029)
        n = ref.shape[0]
        diffs = []
        cap_err = np.mean([c[0] for c in cells[:-1]], axis=0)
        noc_err = cells[-1][0]
        for _ in range(1000):
            s = rng.integers(0, n, n)
            den = np.maximum(ref[s].sum(axis=0), 1)
            diffs.append(
                np.mean(noc_err[s].sum(axis=0) / den) - np.mean(cap_err[s].sum(axis=0) / den)
            )
        lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
        value = float(np.mean(nocap) - np.mean(cap))
        capture_helps = lo > 0 and min(nocap) > max(cap)
        clean_cap = res["arms"]["+100K"]["hiertext_val"]["value"]
        clean_noc = res["arms"]["+100K, no capture"]["hiertext_val"]["value"]
        guard = clean_cap <= 1.05 * clean_noc
        h2b["capture"] = {
            "mce_with_capture": cap,
            "mce_without": nocap,
            "diff_without_minus_with": {"value": value, "low": lo, "high": hi},
            "capture_helps": bool(capture_helps),
            "clean_guard_holds": bool(guard),
            "clean_val_cer": {"with": clean_cap, "without": clean_noc},
        }
    if h2b:
        h2b["supported"] = bool(
            h2b.get("fonts", {}).get("many_fonts_help")
            and h2b.get("capture", {}).get("capture_helps")
            and h2b.get("capture", {}).get("clean_guard_holds")
        )
    res["h2b"] = h2b
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase12_comparison.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "phase12_comparison.md").write_text(render(res), encoding="utf-8", newline="\n")
    print(
        f"{out / 'phase12_comparison.md'}: H2'a "
        f"{'supported' if res['h2a']['supported'] else 'not supported'}"
    )
    return res


def _p(x: float) -> str:
    return f"{100 * x:.2f}"


def _d(r: dict[str, Any]) -> str:
    d = r["diff"]
    return f"{100 * d['value']:+.2f} [{100 * d['low']:+.2f}, {100 * d['high']:+.2f}]"


def render(res: dict[str, Any]) -> str:
    md = [
        "# Phase 12: synthetic data study (GW-DATA-022)",
        "",
        "Generated by `scripts/synth_study.py compare`; do not edit by hand. CER % on the strict",
        f"track. Document slice: HierText validation pages with at least {res['doc_min_lines']}",
        f"validation lines ({res['doc_pages']} pages). Arms pool their seeds; Δ against",
        "HierText only, paired over pages (HierText) or lines (GlyphSynth), 95% CI, 1,000",
        "resamples. Verdict: CI on one side of zero and every seed beyond every control seed.",
        "",
        "| Arm | Runs | Document slice | Δ vs HierText only | Verdict | HierText val | Δ | "
        "GlyphSynth val | Δ |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a, r in res["arms"].items():
        if a == CONTROL:
            md.append(
                f"| {a} | {', '.join(r['runs'])} | {_p(r['doc']['value'])} | — | control | "
                f"{_p(r['hiertext_val']['value'])} | — | {_p(r['glyphsynth_val']['value'])} | — |"
            )
        else:
            md.append(
                f"| {a} | {', '.join(r['runs'])} | {_p(r['doc']['value'])} | "
                f"{_d(r['doc_vs_control'])} | {r['doc_vs_control']['verdict']} | "
                f"{_p(r['hiertext_val']['value'])} | {_d(r['hiertext_val_vs_control'])} | "
                f"{_p(r['glyphsynth_val']['value'])} | {_d(r['glyphsynth_val_vs_control'])} |"
            )
    h = res["h2a"]
    md += [
        "",
        f"H2'a (an improvement on the document slice at one size or more): sizes that help: "
        f"{', '.join(h['sizes_that_help']) or 'none'}: "
        f"**{'supported' if h['supported'] else 'not supported'}**.",
    ]
    if "gain_per_1k_lines" in h:
        g = h["gain_per_1k_lines"]
        md.append(
            "Document-slice CER points gained per 1,000 added lines: "
            f"{100 * g['25K_to_100K']:+.4f} "
            f"(25K to 100K), {100 * g['100K_to_250K']:+.4f} (100K to 250K): "
            f"{'diminishing' if g['diminishing'] else 'not diminishing'} returns."
        )
    b = res["h2b"]
    if b:
        md.append("")
        if "fonts" in b:
            f = b["fonts"]
            md.append(
                f"Fonts: 6 families minus 85 on the document slice {_d(f)}; many fonts "
                f"{'help' if f['many_fonts_help'] else 'show no clear advantage'}."
            )
        if "capture" in b:
            c = b["capture"]
            dd = c["diff_without_minus_with"]
            md.append(
                f"Capture effects: GLYPH-C mCE without minus with {100 * dd['value']:+.2f} "
                f"[{100 * dd['low']:+.2f}, {100 * dd['high']:+.2f}]; capture "
                f"{'helps' if c['capture_helps'] else 'shows no clear effect'}; clean guard "
                f"{'holds' if c['clean_guard_holds'] else 'fails'}."
            )
        md.append(f"H2'b: **{'supported' if b.get('supported') else 'not supported'}**.")
    return "\n".join(md) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--out", type=Path, default=Path("reports/synthetic"))
    c = sub.add_parser("compare")
    c.add_argument("--runs", type=Path, nargs="+", default=[Path("runs")])
    c.add_argument("--out", type=Path, default=Path("reports/synthetic"))
    c.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(argv)
    if args.cmd == "score":
        score_run(args.run, args.out)
    else:
        compare(ARMS, args.runs, args.out, args.data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
