"""Score baseline predictions and write the Phase 5 baseline report.

Runs in the main environment. Reads every prediction file written by `scripts/run_baselines.py`
under data/processed/baselines/<experiment>/ and scores it with `glypharis.eval`, the same code that
will score GLYPHARIS. Writes separate tables (EXPERIMENT_PROTOCOL §10): recognition only, detection
only and end to end, on both normalisation tracks, GDC's vertical lines on their own (§3), and an
efficiency table from the profiling runs; plus results.json with every number and its experiment ID.

    python scripts/evaluate_baselines.py

Scoring a test set is an evaluation, so it goes through the test-access guard and is logged.
Prediction files are checked against the evaluation set they claim: a key outside it, or a key
twice, stops the run. Confidence intervals are 95% percentile bootstrap, resampling lines for
recognition and pages for detection and end to end. The charset ceiling is the share of reference
characters outside the GLYPHARIS charset (§2): errors GLYPHARIS cannot avoid, shown next to every
baseline so the comparison later is like for like. Order-aware page metrics are reported only where
the ground truth has a reading order (GDC); HierText's annotation order is not one.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from run_baselines import ACCESS_LOG, EVAL_SETS, EXPERIMENTS, SYSTEM_NAMES, expected_keys

from glypharis.data.pages import Page, read_pages
from glypharis.data.text import Charset
from glypharis.eval.access import open_test_access
from glypharis.eval.page_metrics import (
    PageResult,
    PredLine,
    evaluate_page,
    is_text_line,
    vertical_view,
)
from glypharis.eval.stats import Interval, ratio_ci
from glypharis.eval.text_metrics import RecAccumulator, Track, outside_charset, track_charset

TRACKS: tuple[Track, ...] = ("strict", "normalised")
THREAD_KEYS = ("cpu_threads", "torch_threads", "OMP_THREAD_LIMIT")

_REC_SETS = ("hiertext_val", "glyphsynth_val", "hiertext_test", "gdc_test")
_PAGE_SETS = ("hiertext_val", "hiertext_test", "gdc_test")
_SLOW = (
    "not run: on this CPU the recognizer needs about 1 s (PP-OCRv6 medium) to 3 s "
    "(PP-OCRv5 server) per long line"
)
LEGACY_NOTE = (
    " (settings as recorded by the first runner, which did not tell recognition and "
    "page runs apart)"
)
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
    ("ppocr_server", "e2e", "gdc_test"): (
        "not runnable on this laptop: both attempts died on the first page, most likely for "
        "lack of memory (research/failures.md, 2026-10-24)"
    ),
}


class Context:
    """What every file needs: data root, charset, page cache, access log."""

    def __init__(self, data: Path, access_log: Path) -> None:
        self.data = data
        self.access_log = access_log
        chars = Charset.load(data / "manifests" / "charset.json").characters
        self.allowed = {t: track_charset(chars, t) for t in TRACKS}
        self._pages: dict[str, dict[str, Page]] = {}
        self._keys: dict[tuple[str, str], list[str]] = {}

    def pages(self, eval_set: str) -> dict[str, Page]:
        if eval_set not in self._pages:
            path = self.data / EVAL_SETS[eval_set]["pages"]
            self._pages[eval_set] = {p.doc_id: p for p in read_pages(path)}
        return self._pages[eval_set]

    def keys(self, eval_set: str, task: str) -> list[str]:
        if (eval_set, task) not in self._keys:
            self._keys[eval_set, task] = expected_keys(self.data, eval_set, task)
        return self._keys[eval_set, task]


def read_records(path: Path) -> list[dict[str, Any]]:
    """Records of a prediction file. A last line cut off by a killed run is left out (the runner
    removes it when the run resumes); a broken line anywhere else is an error."""
    lines = [ln for ln in path.read_bytes().splitlines() if ln.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if i != len(lines) - 1:
                raise
            print(f"{path.name}: ignored a last record cut off by an interrupted run")
    return out


def check_keys(records: list[dict[str, Any]], expected: list[str], name: str) -> None:
    counts = Counter(r["key"] for r in records)
    twice = [k for k, c in counts.items() if c > 1]
    stray = set(counts) - set(expected)
    if twice or stray:
        msg = f"{len(twice)} keys repeated, {len(stray)} outside the evaluation set"
        raise SystemExit(f"{name}: {msg}; the file does not match its sample")


def ci_dict(iv: Interval) -> dict[str, Any]:
    return {"value": iv.value, "low": iv.low, "high": iv.high, "unit": iv.unit}


def ci(num: np.ndarray, den: np.ndarray, unit: str) -> dict[str, Any] | None:
    """Ratio with its bootstrap CI, or None (shown as n/a) when there is nothing to divide by."""
    return ci_dict(ratio_ci(num, den, unit)) if float(np.sum(den)) > 0 else None


def ceiling(texts: list[str], allowed: frozenset[str], track: Track) -> float:
    outside, total = outside_charset(texts, allowed, track)
    return outside / total if total else 0.0


def score_rec(
    records: list[dict[str, Any]], track: Track, allowed: frozenset[str]
) -> dict[str, Any]:
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
        "cer": ci(errs, chars, "lines"),
        "wer": ci(werrs, words, "lines"),
        "ned": s.ned,
        "sequence_accuracy": s.sequence_accuracy,
        "charset_ceiling": ceiling([r["ref"] for r in records], allowed, track),
    }


def _preds(record: dict[str, Any]) -> list[PredLine]:
    return [PredLine(ln["polygon"], ln["text"], ln.get("confidence")) for ln in record["lines"]]


def _arr(results: list[PageResult], f: Any) -> np.ndarray:
    return np.array([f(r) for r in results], dtype=np.float64)


def score_det(results: list[PageResult]) -> dict[str, Any]:
    out: dict[str, Any] = {"pages": len(results)}
    for thr in (0.5, 0.7):
        tp, fp, fn = (_arr(results, lambda r, i=i, t=thr: r.det[t][i]) for i in range(3))
        out[f"iou_{thr}"] = {
            "precision": ci(tp, tp + fp, "pages"),
            "recall": ci(tp, tp + fn, "pages"),
            "f1": ci(2 * tp, 2 * tp + fp + fn, "pages"),
        }
    tp05 = sum(r.det[0.5][0] for r in results)
    out["mean_matched_iou"] = sum(r.matched_iou_sum for r in results) / tp05 if tp05 else 0.0
    out["gt_lines"] = int(sum(r.det[0.5][0] + r.det[0.5][2] for r in results))
    return out


def score_e2e(results: list[PageResult], reading_order: bool) -> dict[str, Any]:
    e2e_tp = _arr(results, lambda r: r.e2e_tp)
    out: dict[str, Any] = {
        "pages": len(results),
        "order_aware_reported": reading_order,
        "page_cer_bag_of_lines": ci(_arr(results, lambda r: r.bag_errors),
                                    _arr(results, lambda r: r.bag_ref_chars), "pages"),
        "e2e_f1": ci(2 * e2e_tp, _arr(results, lambda r: r.e2e_pred + r.e2e_gt), "pages"),
    }  # fmt: skip
    if reading_order:
        out["page_cer_order_aware"] = ci(
            _arr(results, lambda r: r.aware_errors),
            _arr(results, lambda r: r.aware_ref_chars),
            "pages",
        )
        out["page_wer_order_aware"] = ci(
            _arr(results, lambda r: r.aware_word_errors),
            _arr(results, lambda r: r.aware_ref_words),
            "pages",
        )
    return out


def score_vertical(results: list[PageResult]) -> dict[str, Any]:
    """GDC vertical lines: recall, and the CER of those lines found or missed. Precision and
    insertions are left out: every spurious prediction on the page would count against them."""
    tp, fn = _arr(results, lambda r: r.det[0.5][0]), _arr(results, lambda r: r.det[0.5][2])
    errors = _arr(results, lambda r: r.bag_errors - r.bag_insert_chars)
    return {"lines": int(tp.sum() + fn.sum()), "recall": ci(tp, tp + fn, "pages"),
            "cer": ci(errors, _arr(results, lambda r: r.bag_ref_chars), "pages")}  # fmt: skip


def _no_meta_entry(path: Path) -> dict[str, Any] | None:
    """Coverage entry for predictions whose run never wrote metadata (killed before it could)."""
    for task in ("rec", "e2e"):
        system, sep, eval_set = path.stem.partition(f"_{task}_")
        if sep and system in SYSTEM_NAMES and eval_set in EVAL_SETS:
            return {
                "experiment": path.parent.name,
                "system": system,
                "task": task,
                "eval_set": eval_set,
                "samples": len(read_records(path)),
                "no_meta": True,
            }
    return None


def evaluate_file(path: Path, ctx: Context) -> dict[str, Any] | None:
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        print(
            f"not scored {path.parent.name}/{path.name}: no run metadata (killed before the "
            "runner wrote it); the next run of it completes it and writes the metadata"
        )
        return _no_meta_entry(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    task, eval_set, system = meta["task"], meta["eval_set"], meta["system"]
    spec = EVAL_SETS[eval_set]
    records = read_records(path)
    expected = ctx.keys(eval_set, task)
    check_keys(records, expected, f"{path.parent.name}/{path.name}")
    open_test_access(spec["benchmark"], spec["split"], meta["experiment"],
                     f"{system} ({task}), scoring", 5, "Phase 5 baseline scoring",
                     ctx.access_log)  # fmt: skip
    entry: dict[str, Any] = {
        "experiment": meta["experiment"],
        "system": system,
        "version": meta.get("version"),
        "settings": meta.get("settings"),
        "task": task,
        "eval_set": eval_set,
        "split": spec["split"],
        "samples": len(records),
        "expected": len(expected),
        "runs": len(meta.get("runs", [])),
        "legacy_meta": "expected" not in meta,
    }
    if task == "rec":
        entry["tracks"] = {t: score_rec(records, t, ctx.allowed[t]) for t in TRACKS}
        return entry
    pages = ctx.pages(eval_set)
    scored = [pages[r["key"]] for r in records]
    # Ceiling over every reference text line (the order-aware reference; it covers the bag-of-lines
    # reference, which leaves out GDC lines with an untrusted box).
    ref_texts = [ln.text for p in scored for ln in p.lines if is_text_line(ln)]
    entry["tracks"] = {}
    for t in TRACKS:
        results = [evaluate_page(pages[r["key"]], _preds(r), track=t) for r in records]
        if t == "strict":  # boxes do not depend on the text track
            entry["detection"] = score_det(results)
        entry["tracks"][t] = score_e2e(results, spec["reading_order"])
        entry["tracks"][t]["charset_ceiling"] = ceiling(ref_texts, ctx.allowed[t], t)
    if eval_set == "gdc_test":
        entry["vertical"] = {}
        vrefs = [
            ln.text
            for p in scored
            for ln in p.lines
            if ln.vertical and ln.legible and not ln.handwritten and ln.text.strip()
        ]
        for t in TRACKS:
            vres = [evaluate_page(vertical_view(pages[r["key"]]), _preds(r), track=t)
                    for r in records]  # fmt: skip
            entry["vertical"][t] = {
                **score_vertical(vres),
                "charset_ceiling": ceiling(vrefs, ctx.allowed[t], t),
            }
    return entry


def read_profile(path: Path) -> dict[str, Any]:
    p = json.loads(path.read_text(encoding="utf-8"))
    t = np.asarray(p["times_ms"], dtype=np.float64)
    sizes = np.asarray(p["input_sizes"], dtype=np.float64)
    cond = p["conditions"]
    return {
        "experiment": p["experiment"],
        "system": p["system"],
        "task": p["task"],
        "eval_set": p["eval_set"],
        "timed": int(t.size),
        "warmup": p["warmup"],
        "batch": p["batch"],
        "mean_ms": float(t.mean()),
        "p95_ms": float(np.percentile(t, 95)),
        "throughput_per_s": float(1000.0 / t.mean()),
        "peak_mem_mib": p["peak_mem_mib"],
        "child_peak_mib": p.get("child_peak_mib", 0.0),
        "median_input": [int(v) for v in np.median(sizes, axis=0)] if sizes.size else None,
        "invocation_overhead_ms": p.get("invocation_overhead_ms"),
        "threads": {k: v for k, v in (p.get("settings") or {}).items() if k in THREAD_KEYS},
        "on_ac_power": cond.get("on_ac_power"),
        "power_plan": cond.get("power_plan"),
        "cpu_busy_before": cond.get("cpu_busy_percent_before"),
        "cpu_busy_after": p.get("cpu_busy_percent_after"),
        "cpu": cond.get("cpu"),
        "model_mib": p.get("model_mib"),
    }


# ------------------------------------------------------------------------------------ report

NOTE = "Generated by `scripts/evaluate_baselines.py`; do not edit by hand."


def pct(d: dict[str, Any] | None) -> str:
    if d is None:
        return "n/a"
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


def render(entries: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> dict[str, str]:
    order = {s: i for i, s in enumerate(EVAL_SETS)}
    entries = sorted(entries, key=lambda e: (order[e["eval_set"]], e["experiment"]))
    scored = [e for e in entries if not e.get("no_meta")]
    rec = [e for e in scored if e["task"] == "rec"]
    e2e = [e for e in scored if e["task"] == "e2e"]
    out: dict[str, str] = {}

    lines = [
        "# Baselines: recognition only",
        "",
        f"Ground-truth line crops in, text out. {NOTE} CER and WER in %, with the 95% bootstrap",
        "CI over lines. The charset ceiling is the share of reference characters outside the",
        "GLYPHARIS charset (EXPERIMENT_PROTOCOL §2). Speed and memory are in",
        "[efficiency.md](efficiency.md).",
        "",
    ]
    head = ["ID", "System", "Set", "Lines", "CER %", "WER %", "NED", "Seq acc %",
            "Charset ceiling %"]  # fmt: skip
    for t in TRACKS:
        rows = []
        for e in rec:
            s = e["tracks"][t]
            rows.append([*_row_head(e), pct(s["cer"]), pct(s["wer"]), f"{s['ned']:.4f}",
                         f"{100 * s['sequence_accuracy']:.2f}",
                         f"{100 * s['charset_ceiling']:.2f}"])  # fmt: skip
        lines += [f"## Track: {t}", "", *_table(head, rows), ""]
    out["recognition.md"] = "\n".join(lines)

    lines = [
        "# Baselines: detection only",
        "",
        f"Page in, line polygons out. {NOTE} Don't-care lines and ignore regions are handled as in",
        "EXPERIMENT_PROTOCOL §3. Values in %, with the 95% bootstrap CI over pages. Detection does",
        "not depend on the normalisation track. GDC's vertical lines are scored separately in",
        "[end_to_end.md](end_to_end.md).",
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
        f"Page in, text out. {NOTE} Values in %, with the 95% bootstrap CI over pages.",
        "",
        "- **Order-aware** page CER/WER joins the predicted lines in the system's own order and",
        "  compares them with the reference text in reading order. Only GDC has a reading order",
        "  (from the PDF text layer); HierText's annotation order is not one, so its order-aware",
        "  columns are n/a. Predictions on don't-care lines and ignore regions are left out, as in",
        "  every other metric.",
        "- **Bag of lines** matches predicted lines to reference lines by box first, so it",
        "  measures recognition and detection without reading order.",
        "- **E2E F1**: a reference line counts as found when a prediction overlaps it at",
        "  IoU >= 0.5 and has line CER <= 10%.",
        "",
    ]
    head = ["ID", "System", "Set", "Pages", "Page CER % (order-aware)", "Page WER % (order-aware)",
            "Page CER % (bag of lines)", "E2E F1 %", "Charset ceiling %"]  # fmt: skip
    for t in TRACKS:
        rows = []
        for e in e2e:
            s = e["tracks"][t]
            rows.append([*_row_head(e), pct(s.get("page_cer_order_aware")),
                         pct(s.get("page_wer_order_aware")), pct(s["page_cer_bag_of_lines"]),
                         pct(s["e2e_f1"]), f"{100 * s['charset_ceiling']:.2f}"])  # fmt: skip
        lines += [f"## Track: {t}", "", *_table(head, rows), ""]
    vert = [e for e in e2e if "vertical" in e]
    if vert:
        lines += [
            "## GDC vertical lines",
            "",
            "Rotated lines (margin stamps, axis titles) are left out of the tables above and",
            "scored here on their own (EXPERIMENT_PROTOCOL §3). Only measures about these lines",
            "are shown: recall at IoU 0.5, and the CER of the vertical lines found or missed",
            "(a missed line counts all its characters). Precision and insertions are left out,",
            "because any spurious prediction on the page would count against these few lines.",
            "",
        ]
        head = ["ID", "System", "Lines", "Recall @0.5 %", "CER % (strict)", "CER % (normalised)",
                "Charset ceiling %"]  # fmt: skip
        rows = []
        for e in vert:
            v = e["vertical"]
            rows.append([e["experiment"], SYSTEM_NAMES.get(e["system"], e["system"]),
                         str(v["strict"]["lines"]), pct(v["strict"]["recall"]),
                         pct(v["strict"]["cer"]), pct(v["normalised"]["cer"]),
                         f"{100 * v['strict']['charset_ceiling']:.2f}"])  # fmt: skip
        lines += [*_table(head, rows), ""]
    out["end_to_end.md"] = "\n".join(lines)
    out["efficiency.md"] = render_efficiency(profiles)
    out["README.md"] = render_summary(entries, profiles)
    return out


def render_efficiency(profiles: list[dict[str, Any]]) -> str:
    lines = [
        "# Baselines: efficiency",
        "",
        f"{NOTE} Measured by `scripts/run_baselines.py --profile`, separately from the long",
        "prediction runs: one sample at a time (batch 1), the first 10 samples discarded as",
        "warm-up, then 100 timed (EXPERIMENT_PROTOCOL §3), on validation data, CPU only, each",
        "system at the precision it ships (PaddleOCR and TrOCR fp32; Tesseract's fast English",
        "model uses 8-bit integer weights). Peak memory is the peak working set of the Python",
        "process plus the peak of any engine it starts (tesseract.exe); the two peaks need not",
        "coincide, so for Tesseract it is an upper bound. It includes the harness (Python, NumPy,",
        "the data reader). Model size is the system's model files on disk. Peak VRAM: not",
        "applicable, no baseline uses a GPU.",
        "",
    ]
    if not profiles:
        return "\n".join([*lines, "No profiling runs yet.", ""])
    head = ["ID", "System", "Task", "Set", "Timed", "ms mean", "ms p95", "Items/s",
            "Peak RAM MiB", "Model MiB", "Median input (w x h)", "Threads", "On AC",
            "CPU busy % before/after"]  # fmt: skip
    rows = []
    for p in sorted(profiles, key=lambda p: (p["task"], p["eval_set"], p["experiment"])):
        size = "x".join(map(str, p["median_input"])) if p["median_input"] else "-"
        threads = ", ".join(f"{k}={v}" for k, v in p["threads"].items()) or "-"
        busy = f"{p['cpu_busy_before']} / {p['cpu_busy_after']}"
        model = f"{p['model_mib']:,.1f}" if p.get("model_mib") is not None else "-"
        rows.append([p["experiment"], SYSTEM_NAMES.get(p["system"], p["system"]), p["task"],
                     p["eval_set"], str(p["timed"]), f"{p['mean_ms']:,.1f}",
                     f"{p['p95_ms']:,.1f}", f"{p['throughput_per_s']:.2f}",
                     f"{p['peak_mem_mib']:,.0f}", model, size, threads, str(p["on_ac_power"]),
                     busy])  # fmt: skip
    lines += [*_table(head, rows), ""]
    tess = [p for p in profiles if p.get("invocation_overhead_ms") is not None]
    for p in tess:
        lines.append(f"Tesseract is called through its command line, so every call pays process "
                     f"start, model load and image file I/O: {p['invocation_overhead_ms']:.0f} ms "
                     f"on a blank image ({p['task']}, {p['eval_set']}).")  # fmt: skip
    plans = sorted({str(p["power_plan"]) for p in profiles})
    cpus = sorted({str(p["cpu"]) for p in profiles})
    lines += ["", f"CPU: {'; '.join(cpus)}. Power plan: {'; '.join(plans)}.", ""]
    return "\n".join(lines)


def render_summary(entries: list[dict[str, Any]], profiles: list[dict[str, Any]]) -> str:
    """Systems, versions and settings as run, and the coverage of the planned run matrix."""
    lines = [
        "# Baselines (Phase 5)",
        "",
        f"{NOTE} Results: [recognition.md](recognition.md), [detection.md](detection.md),",
        "[end_to_end.md](end_to_end.md) and [efficiency.md](efficiency.md); every number with its",
        "experiment ID is in `results.json`. Protocol: `docs/EXPERIMENT_PROTOCOL.md` §2-§4, §10.",
        "",
        "## Systems as run",
        "",
    ]
    for e in sorted(entries, key=lambda e: (e["experiment"], e["task"], e["eval_set"])):
        if e.get("settings") is None:
            continue
        settings = ", ".join(f"{k}={v}" for k, v in e["settings"].items())
        legacy = LEGACY_NOTE if e.get("legacy_meta") else ""
        lines.append(
            f"- **{e['experiment']} {SYSTEM_NAMES.get(e['system'], e['system'])}**, "
            f"{e['task']} on {e['eval_set']}: {e['version']}. {settings}.{legacy}"
        )
    lines += [
        "",
        "## Coverage",
        "",
        "HierText validation uses seeded subsamples (10,000 line crops, 400 pages); GlyphSynth",
        "validation (10,000 lines) and both test sets are used in full. Recognition sets keep",
        "lines with characters outside the GLYPHARIS charset. A count below the expected one",
        "means the run is still in progress or was interrupted.",
        "",
    ]
    have = {(e["system"], e["task"], e["eval_set"]): e for e in entries}
    prof = {(p["system"], p["task"], p["eval_set"]) for p in profiles}
    rows = []
    for key, reason in sorted(PLAN.items(), key=lambda kv: (EXPERIMENTS[kv[0][0]], kv[0][1:])):
        system, task, eval_set = key
        found = have.get(key)
        if found is not None and found.get("no_meta"):
            status = f"{found['samples']:,} predicted, no run metadata yet"
        elif found is not None:
            status = f"{found['samples']:,} of {found['expected']:,}"
        elif reason is not None:
            status = reason
        else:
            status = "planned, no predictions yet"
        rows.append([EXPERIMENTS[system], SYSTEM_NAMES[system], task, eval_set, status])
    lines += [*_table(["ID", "System", "Task", "Set", "Samples"], rows), ""]
    lines.append(f"Profiling runs present: {len(prof)}.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="reports/baselines")
    parser.add_argument("--access-log", default=str(ACCESS_LOG))
    args = parser.parse_args(argv)

    data = Path(args.data_root)
    root = data / "processed" / "baselines"
    files = sorted(root.glob("*/*.jsonl"))
    if not files:
        print("no baseline predictions found")
        return 1
    ctx = Context(data, Path(args.access_log))
    entries = []
    for f in files:
        entry = evaluate_file(f, ctx)
        if entry is not None:
            entries.append(entry)
            print(f"scored {f.parent.name}/{f.name}")
    profiles = [read_profile(p) for p in sorted(root.glob("*/profile_*.json"))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, text in render(entries, profiles).items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    results = {"predictions": entries, "efficiency": profiles}
    (out / "results.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8",
                                      newline="\n")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
