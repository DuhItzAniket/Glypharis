"""Score baseline predictions and write the Phase 5 baseline report.

Runs in the main environment. Reads every prediction file written by `scripts/run_baselines.py`
under data/processed/baselines/<experiment>/ and scores it with `glypharis.eval`, the same code that
will score GLYPHARIS. Writes three separate tables (EXPERIMENT_PROTOCOL §10): recognition only,
detection only, and end to end, each on both normalisation tracks, plus results.json with every
number and its experiment ID.

    python scripts/evaluate_baselines.py

Confidence intervals are 95% percentile bootstrap, resampling lines for recognition and pages for
detection and end to end. Latency: recognition is per line, averaged within each batch; end to end
is per page after the first 10 pages (warm-up). Peak RAM is the largest process RSS over the runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from run_baselines import EVAL_SETS, expected_count

from glypharis.data.pages import Page, read_pages
from glypharis.eval.page_metrics import PageResult, PredLine, evaluate_page
from glypharis.eval.stats import Interval, ratio_ci
from glypharis.eval.text_metrics import RecAccumulator, Track

TRACKS: tuple[Track, ...] = ("strict", "normalised")
WARMUP_PAGES = 10
SYSTEM_NAMES = {
    "tesseract": "Tesseract 5",
    "ppocr_mobile": "PP-OCRv5 mobile",
    "ppocr_server": "PP-OCRv5 server",
    "trocr": "TrOCR-small-printed",
    "ppocr6_tiny": "PP-OCRv6 tiny",
    "ppocr6_small": "PP-OCRv6 small",
    "ppocr6_medium": "PP-OCRv6 medium",
}
EXPERIMENTS = {"tesseract": "GW-BASE-001", "ppocr_mobile": "GW-BASE-002",
               "ppocr_server": "GW-BASE-003", "trocr": "GW-BASE-004",
               "ppocr6_tiny": "GW-BASE-005", "ppocr6_small": "GW-BASE-006",
               "ppocr6_medium": "GW-BASE-007"}  # fmt: skip

_REC_SETS = ("hiertext_val", "glyphsynth_val", "hiertext_test", "gdc_test")
_PAGE_SETS = ("hiertext_val", "hiertext_test", "gdc_test")
_SLOW = "not run: the recognizer needs about 1-3 s per line on this CPU"
_FULL = ("tesseract", "ppocr_mobile", "ppocr6_tiny", "ppocr6_small")
_REDUCED = ("ppocr_server", "ppocr6_medium")
# (system, task, eval set) -> None when planned, or the reason it is not run.
PLAN: dict[tuple[str, str, str], str | None] = {
    **{(s, "rec", e): None for s in (*_FULL, "trocr") for e in _REC_SETS},
    **{(s, "e2e", e): None for s in _FULL for e in _PAGE_SETS},
    **{("trocr", "e2e", e): "not applicable: TrOCR has no detector" for e in _PAGE_SETS},
    **{(s, "rec", e): None for s in _REDUCED for e in ("hiertext_val", "gdc_test")},
    **{(s, "e2e", "gdc_test"): None for s in _REDUCED},
    **{(s, "rec", e): _SLOW for s in _REDUCED for e in ("glyphsynth_val", "hiertext_test")},
    **{(s, "e2e", e): _SLOW for s in _REDUCED for e in ("hiertext_val", "hiertext_test")},
}


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def ci_dict(iv: Interval) -> dict[str, Any]:
    return {"value": iv.value, "low": iv.low, "high": iv.high, "unit": iv.unit}


def latency(times: Iterable[float], skip: int = 0) -> dict[str, float | int]:
    t = np.asarray(list(times), dtype=np.float64)[skip:]
    if t.size == 0:
        return {"mean_ms": float("nan"), "p95_ms": float("nan"), "n": 0}
    return {"mean_ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95)), "n": int(t.size)}


def score_rec(records: list[dict[str, Any]], track: Track) -> dict[str, Any]:
    acc = RecAccumulator(track)
    for r in records:
        acc.add(r["pred"], r["ref"])
    s = acc.summary()
    errs = np.array([x.char_errors for x in acc.scores], dtype=np.float64)
    chars = np.array([x.ref_chars for x in acc.scores], dtype=np.float64)
    werrs = np.array([x.word_errors for x in acc.scores], dtype=np.float64)
    words = np.array([x.ref_words for x in acc.scores], dtype=np.float64)
    return {
        "samples": s.samples,
        "empty_references": s.empty_references,
        "cer": ci_dict(ratio_ci(errs, chars, "lines")),
        "wer": ci_dict(ratio_ci(werrs, words, "lines")),
        "ned": s.ned,
        "sequence_accuracy": s.sequence_accuracy,
    }


def page_results(
    records: list[dict[str, Any]], pages: dict[str, Page], track: Track
) -> list[PageResult]:
    out = []
    for r in records:
        preds = [PredLine(ln["polygon"], ln["text"], ln.get("confidence")) for ln in r["lines"]]
        out.append(evaluate_page(pages[r["key"]], preds, track=track))
    return out


def _arr(results: list[PageResult], f: Any) -> np.ndarray:
    return np.array([f(r) for r in results], dtype=np.float64)


def score_det(results: list[PageResult]) -> dict[str, Any]:
    out: dict[str, Any] = {"pages": len(results)}
    for thr in (0.5, 0.7):
        tp, fp, fn = (_arr(results, lambda r, i=i, t=thr: r.det[t][i]) for i in range(3))
        out[f"iou_{thr}"] = {
            "precision": ci_dict(ratio_ci(tp, tp + fp, "pages")),
            "recall": ci_dict(ratio_ci(tp, tp + fn, "pages")),
            "f1": ci_dict(ratio_ci(2 * tp, 2 * tp + fp + fn, "pages")),
        }
    tp05 = sum(r.det[0.5][0] for r in results)
    out["mean_matched_iou"] = sum(r.matched_iou_sum for r in results) / tp05 if tp05 else 0.0
    return out


def score_e2e(results: list[PageResult]) -> dict[str, Any]:
    e2e_tp = _arr(results, lambda r: r.e2e_tp)
    return {
        "pages": len(results),
        "page_cer_order_aware": ci_dict(ratio_ci(_arr(results, lambda r: r.aware_errors),
                                                 _arr(results, lambda r: r.aware_ref_chars),
                                                 "pages")),
        "page_cer_bag_of_lines": ci_dict(ratio_ci(_arr(results, lambda r: r.bag_errors),
                                                  _arr(results, lambda r: r.bag_ref_chars),
                                                  "pages")),
        "e2e_f1": ci_dict(ratio_ci(2 * e2e_tp, _arr(results, lambda r: r.e2e_pred + r.e2e_gt),
                                   "pages")),
    }  # fmt: skip


def evaluate_file(path: Path, data: Path, page_cache: dict[str, dict[str, Page]]) -> dict[str, Any]:
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    records = read_records(path)
    task, eval_set = meta["task"], meta["eval_set"]
    spec = EVAL_SETS[eval_set]
    entry: dict[str, Any] = {
        "experiment": meta["experiment"],
        "system": meta["system"],
        "version": meta["version"],
        "settings": meta["settings"],
        "task": task,
        "eval_set": eval_set,
        "split": spec["split"],
        "samples": len(records),
        "expected": expected_count(data, eval_set, task),
        "peak_rss_mib": max(r["peak_rss_mib"] for r in meta["runs"]),
        "batch": meta["runs"][-1]["batch"],
    }
    if task == "rec":
        entry["latency"] = latency(r["time_ms"] for r in records)
        entry["tracks"] = {t: score_rec(records, t) for t in TRACKS}
        return entry
    if eval_set not in page_cache:
        page_cache[eval_set] = {p.doc_id: p for p in read_pages(data / spec["pages"])}
    pages = page_cache[eval_set]
    entry["latency"] = latency((r["time_ms"] for r in records), skip=WARMUP_PAGES)
    entry["tracks"] = {}
    for t in TRACKS:
        results = page_results(records, pages, t)
        if t == "strict":  # boxes do not depend on the text track
            entry["detection"] = score_det(results)
        entry["tracks"][t] = score_e2e(results)
    return entry


# ------------------------------------------------------------------------------------ report


def pct(d: dict[str, Any]) -> str:
    return f"{100 * d['value']:.2f} [{100 * d['low']:.2f}, {100 * d['high']:.2f}]"


def _row_head(e: dict[str, Any]) -> list[str]:
    n = f"{e['samples']:,}" + ("" if e["samples"] == e["expected"] else f" of {e['expected']:,}")
    return [e["experiment"], SYSTEM_NAMES.get(e["system"], e["system"]), e["eval_set"], n]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def render(entries: list[dict[str, Any]]) -> dict[str, str]:
    order = {s: i for i, s in enumerate(EVAL_SETS)}
    entries = sorted(entries, key=lambda e: (order[e["eval_set"]], e["experiment"]))
    rec = [e for e in entries if e["task"] == "rec"]
    e2e = [e for e in entries if e["task"] == "e2e"]
    note = "Generated by `scripts/evaluate_baselines.py`; do not edit by hand."
    out: dict[str, str] = {}

    lines = [
        "# Baselines: recognition only",
        "",
        f"Ground-truth line crops in, text out. {note} CER and WER in %, with the 95% bootstrap",
        "CI over lines. Latency is per line on CPU, averaged within each batch (last column).",
        "",
    ]
    head = ["ID", "System", "Set", "Lines", "CER %", "WER %", "NED", "Seq acc %",
            "ms/line mean / p95", "Peak RAM MiB", "Batch"]  # fmt: skip
    for t in TRACKS:
        rows = []
        for e in rec:
            s, lat = e["tracks"][t], e["latency"]
            rows.append([*_row_head(e), pct(s["cer"]), pct(s["wer"]), f"{s['ned']:.4f}",
                         f"{100 * s['sequence_accuracy']:.2f}",
                         f"{lat['mean_ms']:.1f} / {lat['p95_ms']:.1f}",
                         f"{e['peak_rss_mib']:,.0f}", str(e["batch"])])  # fmt: skip
        lines += [f"## Track: {t}", "", *_table(head, rows), ""]
    out["recognition.md"] = "\n".join(lines)

    lines = [
        "# Baselines: detection only",
        "",
        f"Page in, line polygons out. {note} Don't-care lines and ignore regions are handled as in",
        "EXPERIMENT_PROTOCOL §3. Values in %, with the 95% bootstrap CI over pages. Detection does",
        "not depend on the normalisation track.",
        "",
    ]
    head = ["ID", "System", "Set", "Pages", "P @0.5", "R @0.5", "F1 @0.5", "F1 @0.7", "Mean IoU"]
    rows = []
    for e in e2e:
        d = e["detection"]
        rows.append([*_row_head(e), pct(d["iou_0.5"]["precision"]), pct(d["iou_0.5"]["recall"]),
                     pct(d["iou_0.5"]["f1"]), pct(d["iou_0.7"]["f1"]),
                     f"{d['mean_matched_iou']:.3f}"])  # fmt: skip
    lines += [*_table(head, rows), ""]
    out["detection.md"] = "\n".join(lines)

    lines = [
        "# Baselines: end to end",
        "",
        f"Page in, text out. {note} Page CER in %, with the 95% bootstrap CI over pages.",
        "Order-aware joins predicted lines in the system's own order; bag-of-lines matches lines",
        "by box first, which separates reading-order errors from recognition errors. E2E F1: a",
        "line counts when IoU >= 0.5 and line CER <= 10%. Latency is per page on CPU after",
        f"{WARMUP_PAGES} warm-up pages.",
        "",
    ]
    head = ["ID", "System", "Set", "Pages", "Page CER % (order-aware)",
            "Page CER % (bag of lines)", "E2E F1 %", "ms/page mean / p95",
            "Peak RAM MiB"]  # fmt: skip
    for t in TRACKS:
        rows = []
        for e in e2e:
            s, lat = e["tracks"][t], e["latency"]
            rows.append([*_row_head(e), pct(s["page_cer_order_aware"]),
                         pct(s["page_cer_bag_of_lines"]), pct(s["e2e_f1"]),
                         f"{lat['mean_ms']:,.0f} / {lat['p95_ms']:,.0f}",
                         f"{e['peak_rss_mib']:,.0f}"])  # fmt: skip
        lines += [f"## Track: {t}", "", *_table(head, rows), ""]
    out["end_to_end.md"] = "\n".join(lines)
    out["README.md"] = render_summary(entries)
    return out


def render_summary(entries: list[dict[str, Any]]) -> str:
    """Systems, versions and settings as run, and the coverage of the planned run matrix."""
    lines = [
        "# Baselines (Phase 5)",
        "",
        "Generated by `scripts/evaluate_baselines.py`; do not edit by hand. Results are in",
        "[recognition.md](recognition.md), [detection.md](detection.md) and",
        "[end_to_end.md](end_to_end.md); every number with its experiment ID is in",
        "`results.json`. Protocol: `docs/EXPERIMENT_PROTOCOL.md` §3, §4 and §10.",
        "",
        "## Systems as run",
        "",
    ]
    seen: dict[str, dict[str, Any]] = {}
    for e in entries:
        seen.setdefault(e["system"], e)
    for system, e in sorted(seen.items(), key=lambda kv: EXPERIMENTS.get(kv[0], kv[0])):
        settings = ", ".join(f"{k}={v}" for k, v in e["settings"].items())
        lines.append(f"- **{e['experiment']} {SYSTEM_NAMES.get(system, system)}**: "
                     f"{e['version']}. Settings: {settings}.")  # fmt: skip
    lines += [
        "",
        "## Coverage",
        "",
        "Validation sets are seeded subsamples (10,000 crops, 400 pages); test sets are complete.",
        "A count below the expected one means the run is still in progress.",
        "",
    ]
    have = {(e["system"], e["task"], e["eval_set"]): e for e in entries}
    rows = []
    for key, reason in sorted(PLAN.items(), key=lambda kv: (EXPERIMENTS[kv[0][0]], kv[0][1:])):
        system, task, eval_set = key
        e = have.get(key)
        if e is not None:
            status = f"{e['samples']:,} of {e['expected']:,}"
        elif reason is not None:
            status = reason
        else:
            status = "planned, no predictions yet"
        rows.append([EXPERIMENTS[system], SYSTEM_NAMES[system], task, eval_set, status])
    lines += [*_table(["ID", "System", "Task", "Set", "Samples"], rows), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="reports/baselines")
    args = parser.parse_args(argv)

    data = Path(args.data_root)
    files = sorted((data / "processed" / "baselines").glob("*/*.jsonl"))
    if not files:
        print("no baseline predictions found")
        return 1
    cache: dict[str, dict[str, Page]] = {}
    entries = []
    for f in files:
        entries.append(evaluate_file(f, data, cache))
        print(f"scored {f.parent.name}/{f.name}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in render(entries).items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    (out / "results.json").write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8",
                                      newline="\n")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
