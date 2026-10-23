"""Run report for a detector training run (spec §10, EXPERIMENT_PROTOCOL §8), validation only.

Built only from what the run left on disk, so `scripts/det_report.py` can regenerate it at any time:
``metrics.jsonl``, ``summary.json``, ``eval_full_{best,last}.json``, the ``val_full_best_*`` files,
the sanity JSON, the slice cache and the validation images. A missing input is reported as "not
available" rather than failing, and a failed run (``exit_code`` != 0 in summary.json) opens with a
banner. A full-validation file is used only when summary.json's ``full_eval`` marks that checkpoint
done, so files from an earlier attempt of the run ID never pass as this run's. Writes
``report.md``, ``curves.png`` and ``overlays/<slice>.jpg`` into the report directory and full-size
per-page overlays into ``<run_dir>/overlays/`` (not committed).

The headline is best.pt on the full validation set at its stored operating point, the one chosen on
the per-epoch subset (spec §7.3). The full-set arg-max is printed beside it and never used.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from glypharis.config.schema import RunConfig
from glypharis.data.det_geometry import decode_page
from glypharis.data.download import sha256_file
from glypharis.data.pages import Page, read_pages
from glypharis.eval.det_matching import DetMatches, det_cis, detection_matches
from glypharis.eval.det_overlay import contact_sheet, draw_overlay
from glypharis.eval.det_slices import PAGE_SLICES, LineAttrs, load_or_build_slices, page_slices
from glypharis.eval.page_metrics import PredLine
from glypharis.eval.stats import Interval, ratio_ci
from glypharis.training.det_eval import DetPageRecord, box_from_dict, read_records
from glypharis.training.loop import read_jsonl

log = logging.getLogger(__name__)

OVERLAYS_PER_SLICE: Final = 6
OVERLAY_LONG_SIDE: Final = 1024
WORST_PAGES: Final = 8
OVERLAY_SEED: Final = 20261023  # overlay pages are drawn independently of any result
WORST_MIN_CARE: Final = 5
SHEET_TILE: Final = 512  # tile long side in the committed contact sheets
HEIGHT_BINS: Final = (0.0, 8.0, 12.0, 20.0, 40.0, math.inf)  # care-line height, px at 1280
ROTATION_BINS: Final = (0.0, 5.0, 15.0, 45.0, math.inf)  # abs(text angle), degrees
NA: Final = "not available"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _f(x: Any, digits: int = 4) -> str:
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) and math.isfinite(x) else "-"


def _ci(d: Mapping[str, float] | None) -> str:
    if not d:
        return "-"
    return f"{d['value']:.4f} [{d['low']:.4f}, {d['high']:.4f}]"


def _iv(iv: Interval) -> str:
    return iv.fmt(4)


# ---------------------------------------------------------------------------------- curves


def _series(rows: Sequence[Mapping[str, Any]], key: str) -> tuple[list[float], list[float]]:
    pts = [(r["step"], r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    return [p[0] for p in pts], [p[1] for p in pts]


def _curves(
    steps: list[dict[str, Any]], epochs: list[dict[str, Any]], ops: Sequence[str], out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    ax = axes.ravel()

    def plot(
        a: Any, rows: list[dict[str, Any]], key: str, fmt: str, label: str, lw: float = 1.0
    ) -> list[float]:
        xs, ys = _series(rows, key)
        if xs:
            a.plot(xs, ys, fmt, lw=lw, label=label)
        return ys

    loss_ys = plot(ax[0], steps, "loss", "-", "train loss (running)")
    for part in ("bce", "dice", "l1"):
        loss_ys += plot(ax[0], steps, part, "-", f"train {part} (step)", 0.5)
    for key in ("val_loss", "val_loss_bce", "val_loss_dice", "val_loss_l1"):
        loss_ys += plot(ax[0], epochs, key, "o-", key)
    if loss_ys and min(loss_ys) > 0:
        ax[0].set_yscale("log")
    for op in ops:
        plot(ax[1], epochs, f"val_{op}_f1_50", "o-", f"val F1@0.5 {op}")
    plot(ax[1], epochs, "train_f1_50", "s--", "train F1@0.5 (max rule)")
    plot(ax[2], steps, "lr", "-", "learning rate")
    norms = plot(ax[3], steps, "grad_norm", "-", "grad norm (before clipping)", 0.6)
    if norms and min(norms) > 0:
        ax[3].set_yscale("log")
    twin = ax[3].twinx()
    plot(twin, steps, "clipped_share", "r-", "clipped share", 0.6)
    plot(ax[4], steps, "vram_peak_alloc_mib", "-", "peak allocated MiB")
    plot(ax[4], steps, "vram_peak_reserved_mib", "-", "peak reserved MiB")
    plot(ax[5], steps, "gpu_temperature_c", "-", "GPU temperature C")
    plot(ax[5].twinx(), steps, "gpu_sm_clock_mhz", "g-", "SM clock MHz", 0.6)
    plot(ax[6], epochs, "pred_kernel_ratio", "o-", "predicted / kernel px")
    plot(ax[6], epochs, "kernel_excess_px", "s-", "kernel excess px")
    plot(ax[6], epochs, "thresh_out_of_range", "^-", "threshold out of range")
    plot(ax[7], epochs, "data_wait_share", "o-", "data-wait share")
    plot(ax[7], epochs, "empty_crop_share", "s-", "empty-crop share")
    titles = [
        "loss parts",
        "F1@0.5 (validation subset)",
        "learning rate",
        "gradient norm and clipped share",
        "VRAM",
        "GPU temperature and clock",
        "map statistics",
        "data wait and empty crops",
    ]
    for a, title in zip(ax, titles, strict=True):
        a.set_title(title)
        a.set_xlabel("optimizer step")
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------------------- sections


def _run_section(s: Mapping[str, Any], cfg: RunConfig) -> list[str]:
    det = cfg.det
    git = s.get("git") or {}
    data = s.get("data") or {}
    params = s.get("parameters") or {}
    versions = s.get("versions") or {}
    hours = (s.get("train_seconds") or 0.0) / 3600
    t = cfg.train
    crops = (s.get("steps") or 0) * t.batch_size * t.gradient_accumulation
    crops_per_s = crops / s["train_seconds"] if s.get("train_seconds") else 0.0
    gpu = s.get("gpu")
    out = [
        "## Run",
        "",
        f"- Config hash `{str(s.get('config_hash'))[:16]}`, git `{git.get('commit')}` "
        f"(dirty: {git.get('dirty')}); `train.deterministic`: {s.get('deterministic')}; "
        f"`det.post.box_type`: {s.get('box_type', det.post.box_type if det else NA)}",
        f"- Device {s.get('device')} ({gpu or 'CPU'}); precision: training {s.get('precision')}, "
        f"evaluation {s.get('precision_eval', 'fp32 (ieee)')}",
        "- Versions: " + (", ".join(f"{k} {v}" for k, v in versions.items()) or NA),
        f"- Parameters: {params.get('train', 0):,} (training), {params.get('inference', 0):,} "
        f"(inference, no threshold head); backbone {s.get('backbone')}, weights "
        f"{str(s.get('weights_sha256'))[:16]}",
        f"- Pages: `{data.get('train_pages')}` sha256 {str(data.get('train_pages_sha256'))[:12]}, "
        f"`{data.get('val_pages')}` sha256 {str(data.get('val_pages_sha256'))[:12]}; validation "
        f"subset of {s.get('val_pages_per_epoch')} pages, ID-list sha256 "
        f"{str(data.get('val_subset_ids_sha256'))[:12]}",
        f"- LABEL_VERSION {s.get('label_version')}, DET_DATA_VERSION {s.get('det_data_version')}",
        f"- Epochs run {s.get('epochs_run')}, optimizer steps {s.get('steps', 0):,} of "
        f"{s.get('total_steps', 0):,}; training time {hours:.2f} h "
        f"({f'{hours:.2f} GPU-hours' if gpu else 'CPU'}), {crops_per_s:.1f} crops/s",
        "",
    ]
    return out


def _epoch_row(name: str, p: Mapping[str, Any] | None) -> str:
    if not p:
        return f"| {name} | - | - | - | - | - | - | - | - | - | - |"
    op = p.get("val_op")
    cells = [
        p.get("epoch"),
        p.get("step"),
        op,
        _f(p.get(f"val_{op}_p50")),
        _f(p.get(f"val_{op}_r50")),
        _f(p.get("val_f1_50")),
        _f(p.get(f"val_{op}_f1_70")),
        _f(p.get(f"val_{op}_miou")),
        _f(p.get("val_loss")),
        _f(p.get("train_f1_50")),
    ]
    return f"| {name} | " + " | ".join(str(c) for c in cells) + " |"


def _full_section(
    full: Mapping[str, dict[str, Any] | None],
    s: Mapping[str, Any],
    sanity: Mapping[str, Any] | None,
) -> list[str]:
    out = ["## Full validation", ""]
    outcome = s.get("full_eval") or {}
    best = full.get("best")
    for tag in ("best", "last"):
        ev = full.get(tag)
        if ev is None:
            out += [f"`{tag}.pt`: {NA} ({outcome.get(tag, 'no evaluation on disk')}).", ""]
            continue
        out += [
            f"`{tag}.pt` on all {ev['pages']:,} validation pages ({ev['care_lines']:,} care "
            f"lines), checkpoint sha256 {ev['checkpoint_sha256'][:12]}, stored operating point "
            f"`{ev['stored_op']}`. Values with 95% page-bootstrap CIs (1,000 resamples).",
            "",
            "| Operating point | Precision@0.5 | Recall@0.5 | F1@0.5 | F1@0.7 | Mean IoU |",
            "|---|---|---|---|---|---|",
        ]
        for op, m in ev["ops"].items():
            mark = " (stored)" if op == ev["stored_op"] else ""
            out.append(
                f"| {op}{mark} | {_ci(m['precision'])} | {_ci(m['recall'])} | "
                f"{_ci(m['f1_50'])} | {_ci(m['f1_70'])} | {_f(m['mean_iou'])} |"
            )
        out.append("")
    if best is not None:
        h = best["headline"]
        out += [
            f"**Headline:** `best.pt` at its stored operating point `{best['stored_op']}`: "
            f"F1@0.5 {_ci(h['f1_50'])}, precision {_ci(h['precision'])}, recall "
            f"{_ci(h['recall'])}, F1@0.7 {_ci(h['f1_70'])}. The operating point was chosen on "
            "the per-epoch validation subset from two pre-registered settings; selection on "
            "validation, mildly optimistic.",
            "",
            f"Full-set arg-max operating point: `{best['full_set_max_op']}` (information only, "
            "not used for selection).",
            "",
        ]
        ceiling = (sanity or {}).get("label_ceiling")
        key = f"strict/{best['stored_op']}"
        if ceiling and key in ceiling.get("per_op", {}):
            c = ceiling["per_op"][key]
            sel = ceiling.get("selection", {}).get("strict", {})
            out += [
                f"Label ceiling at the same settings (sanity check 0, {ceiling['pages']:,} "
                f"pages): F1@0.5 {_ci(c['f1_50'])}; selection ceiling "
                f"{_f(sel.get('f1_50'))} at `{sel.get('op')}`.",
                "",
            ]
        else:
            out += [f"Label ceiling (sanity check 0): {NA}.", ""]
    trip = s.get("label_ceiling_tripwire")
    if trip:
        out += [
            f"Start-up tripwire: label ceiling F1@0.5 {_f(trip.get('f1_50'))} at "
            f"`{trip.get('op')}` on {trip.get('pages')} validation-subset pages.",
            "",
        ]
    out += ["## Baseline-comparable row", ""]
    base = (best or {}).get("baseline_pages")
    if base:
        out += [
            f"The {base['pages']} Phase 5 validation pages, detection only, `best.pt` at "
            f"`{base['op']}`: precision {_ci(base['precision'])}, recall "
            f"{_ci(base['recall'])}, F1@0.5 {_ci(base['f1_50'])}, F1@0.7 "
            f"{_ci(base['f1_70'])}.",
            "",
        ]
    else:
        out += [f"{NA}.", ""]
    return out


def _stats_section(
    best_point: Mapping[str, Any] | None, best_eval: Mapping[str, Any] | None, ops: Sequence[str]
) -> list[str]:
    out = ["## Label statistics and post-processing counters", ""]
    p = best_point or {}
    shares = [
        (k, p.get(f"label_{k}_share"))
        for k in ("demoted_small", "demoted_thin_kernel", "multipart_kernel", "dont_care")
    ]
    if any(v is not None for _, v in shares):
        out.append(
            "Training labels in the best epoch: "
            + ", ".join(f"{k.replace('_', ' ')} {100 * v:.1f}%" for k, v in shares if v is not None)
            + "."
        )
    else:
        out.append(f"Label statistics: {NA}.")
    out.append("")
    source = "full validation of best.pt" if best_eval else "the best epoch's validation subset"
    rows = []
    for op in ops:
        c = (best_eval or {}).get("ops", {}).get(op, {}).get("counters") or p.get(
            f"val_{op}_counters"
        )
        if c:
            rows.append(
                f"| {op} | "
                + " | ".join(
                    str(c.get(k, 0))
                    for k in (
                        "components",
                        "tiny",
                        "truncated",
                        "low_score",
                        "too_thin",
                        "outside",
                        "kept",
                        "poly_fallback",
                    )
                )
                + " |"
            )
    if rows:
        out += [
            f"Post-processing counters ({source}):",
            "",
            "| Operating point | Components | Tiny | Truncated | Low score | Too thin | "
            "Outside | Kept | Poly fallback |",
            "|---|---|---|---|---|---|---|---|---|",
            *rows,
        ]
        per_line = None
        if best_eval:
            h = best_eval["headline"]
            tp, _, fn = h.get("tp_fp_fn_50", (0, 0, 0))
            kept = (h.get("counters") or {}).get("kept")
            per_line = kept / (tp + fn) if tp + fn and kept is not None else None
        if per_line is not None:
            out += [
                "",
                f"Predictions per care line at the stored operating point: {per_line:.2f} "
                "(well above 1 hints at splits, well below at merges or misses).",
            ]
        elif p.get("val_boxes_per_page") is not None:
            out += [
                "",
                f"Boxes per validation page in the best epoch: {p['val_boxes_per_page']:.1f}.",
            ]
    else:
        out.append(f"Post-processing counters: {NA}.")
    return [*out, ""]


def _flags_section(
    s: Mapping[str, Any], epochs: list[dict[str, Any]], recs: list[dict[str, Any]]
) -> list[str]:
    flags = s.get("flags") or [r for r in recs if r.get("type") == "flag"]
    out = ["## Diagnostics", ""]
    if flags:
        out += ["| Flag | Epoch | Step | Evidence |", "|---|---|---|---|"]
        out += [f"| {f['name']} | {f['epoch']} | {f['step']} | {f['evidence']} |" for f in flags]
    else:
        out.append("No flags raised.")
    ooms = [r for r in recs if r.get("type") == "oom"]
    for r in ooms:
        out.append(
            f"\nOut of GPU memory in phase `{r.get('phase')}` at epoch {r.get('epoch')}, step "
            f"{r.get('step')} (batch shape {r.get('batch_shape')})."
        )
    skipped = sum(1 for r in recs if r.get("type") == "nonfinite_skipped")
    if skipped:
        out.append(f"\n{skipped} batch(es) with a non-finite loss were skipped.")
    no_update = sum(1 for r in recs if r.get("type") == "nonfinite_grad_skipped")
    if no_update:
        out.append(f"\n{no_update} optimizer step(s) with a non-finite gradient norm were skipped.")
    return [*out, ""]


# ------------------------------------------------------------------------ slices, overlays


def _preds(row: Mapping[str, Any] | None, op: str) -> list[PredLine]:
    if not row:
        return []
    return [box_from_dict(b).to_pred_line() for b in row["boxes"].get(op, [])]


def _bin_labels(edges: Sequence[float], unit: str) -> list[str]:
    return [
        f"{lo:g}–{hi:g} {unit}" if math.isfinite(hi) else f">= {lo:g} {unit}"
        for lo, hi in itertools.pairwise(edges)
    ]


def _bin_rows(
    lines: Sequence[tuple[str, float, bool]], edges: Sequence[float], labels: Sequence[str]
) -> list[str]:
    """Line-level recall per bin with CIs from resampling pages: (doc_id, value, found)."""
    docs = sorted({d for d, _, _ in lines})
    index = {d: i for i, d in enumerate(docs)}
    out = []
    for lo, hi, label in zip(edges[:-1], edges[1:], labels, strict=True):
        num, den = np.zeros(len(docs)), np.zeros(len(docs))
        for d, v, found in lines:
            if lo <= v < hi:
                den[index[d]] += 1
                num[index[d]] += found
        if den.sum():
            out.append(f"| {label} | {int(den.sum()):,} | {_iv(ratio_ci(num, den, 'pages'))} |")
        else:
            out.append(f"| {label} | 0 | - |")
    return out


def _slices_section(
    run_dir: Path, report_dir: Path, cfg: RunConfig, best: Mapping[str, Any]
) -> list[str]:
    """Slice table, line-level recall bins and the overlay sheets (all from disk)."""
    assert cfg.det is not None
    data_root = Path(cfg.paths.data_root)
    op = str(best["stored_op"])
    records = {
        r["doc_id"]: DetPageRecord.from_dict(r)
        for r in read_records(run_dir / "val_full_best_pages.jsonl")
    }
    boxes = {r["doc_id"]: r for r in read_records(run_dir / "val_full_best_boxes.jsonl.gz")}
    if not records:
        return [f"Slices: {NA} (no val_full_best_pages.jsonl).", ""]
    val_file = data_root / cfg.det.data.val_pages[0]
    pages = [p for p in read_pages(val_file) if p.doc_id in records]
    table = load_or_build_slices(pages, sha256_file(val_file), data_root, cfg.det.infer.long_side)
    members = page_slices(table)
    by_id = {p.doc_id: p for p in pages}
    matches: dict[str, DetMatches] = {
        d: detection_matches(by_id[d], _preds(boxes.get(d), op), 0.5) for d in by_id
    }

    out = [
        f"## Slices (`SLICE_VERSION` {table.version}, {table.long_side} px long side)",
        "",
        f"At the headline operating point `{op}`; P/R/F1@0.5 with 95% page-bootstrap CIs. "
        f"`noisy_background` threshold: edge density >= "
        f"{table.thresholds['noisy_background']:.4f}.",
        "",
        "| Slice | Pages | Care lines | Precision@0.5 | Recall@0.5 | F1@0.5 |",
        "|---|---|---|---|---|---|",
    ]
    for name in PAGE_SLICES:
        recs = [records[d] for d in members[name] if d in records]
        if not recs:
            out.append(f"| {name} | 0 | 0 | - | - | - |")
            continue
        a = np.array([r.det[op][0.5] for r in recs], dtype=np.float64)
        cis = det_cis(a[:, 0], a[:, 1], a[:, 2])
        out.append(
            f"| {name} | {len(recs):,} | {sum(r.n_care for r in recs):,} | "
            f"{_iv(cis['precision'])} | {_iv(cis['recall'])} | {_iv(cis['f1'])} |"
        )

    status = {
        (d, by_id[d].lines[i].line_id): s
        for d, m in matches.items()
        for i, s in enumerate(m.gt_status)
    }

    def found(a: LineAttrs) -> bool:
        return status.get((a.doc_id, a.line_id)) == "tp"

    heights = [(a.doc_id, a.height, found(a)) for a in table.lines]
    angles = [(a.doc_id, abs(a.angle), found(a)) for a in table.lines]
    tall = [(a.doc_id, 1.0 if a.tall_box else 0.0, found(a)) for a in table.lines]
    out += [
        "",
        "Line-level recall@0.5 of care lines, CIs from resampling pages:",
        "",
        "| Height (min-area-rectangle short side) | Lines | Recall@0.5 |",
        "|---|---|---|",
        *_bin_rows(heights, HEIGHT_BINS, _bin_labels(HEIGHT_BINS, "px")),
        "",
        "| abs(text angle) (GT TL→TR direction) | Lines | Recall@0.5 |",
        "|---|---|---|",
        *_bin_rows(angles, ROTATION_BINS, _bin_labels(ROTATION_BINS, "°")),
        "",
        "| Box shape | Lines | Recall@0.5 |",
        "|---|---|---|",
        *_bin_rows(
            tall,
            (0.0, 1.0, 2.0),
            ("other care lines", "tall_box (long axis > 45° from horizontal)"),
        ),
        "",
    ]

    out += _overlays(run_dir, report_dir, data_root, members, by_id, records, boxes, matches, op)
    return out


def _overlays(
    run_dir: Path,
    report_dir: Path,
    data_root: Path,
    members: Mapping[str, list[str]],
    by_id: Mapping[str, Page],
    records: Mapping[str, DetPageRecord],
    boxes: Mapping[str, Any],
    matches: Mapping[str, DetMatches],
    op: str,
) -> list[str]:
    sheet_dir, page_dir = report_dir / "overlays", run_dir / "overlays"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    page_dir.mkdir(parents=True, exist_ok=True)

    def tile(doc_id: str) -> Any:
        page = by_id[doc_id]
        rgb = decode_page(data_root / page.image, (page.width, page.height))
        img = draw_overlay(
            rgb, page, _preds(boxes.get(doc_id), op), matches[doc_id], OVERLAY_LONG_SIDE
        )
        img.save(page_dir / f"{doc_id}.jpg", quality=90)
        return img

    out = [
        "## Overlays",
        "",
        f"Six pages per slice drawn by `default_rng([{OVERLAY_SEED}, slice_index])` over the "
        "sorted members, independently of results, at the headline operating point. GT care: "
        "tp green, fn orange (thick); don't care grey; predictions: tp blue, fp red, ignored grey "
        "dashed. HierText annotations CC BY-SA 4.0, images CC BY 2.0.",
        "",
    ]
    for k, name in enumerate(PAGE_SLICES):
        ids = sorted(d for d in members[name] if d in by_id)
        n = min(OVERLAYS_PER_SLICE, len(ids))
        pick = np.sort(np.random.default_rng([OVERLAY_SEED, k]).choice(len(ids), n, replace=False))
        sheet = contact_sheet([tile(ids[int(i)]) for i in pick], cols=3, tile_long=SHEET_TILE)
        sheet.save(sheet_dir / f"{name}.jpg", quality=85)
        out.append(f"- `{name}` ({len(ids)} pages): ![{name}](overlays/{name}.jpg)")

    def page_f1(r: DetPageRecord) -> float:
        tp, fp, fn = r.det[op][0.5]
        return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0

    worst = sorted(
        (r for r in records.values() if r.n_care >= WORST_MIN_CARE and r.doc_id in by_id),
        key=lambda r: (page_f1(r), r.doc_id),
    )[:WORST_PAGES]
    sheet = contact_sheet([tile(r.doc_id) for r in worst], cols=4, tile_long=SHEET_TILE)
    sheet.save(sheet_dir / "worst_pages.jpg", quality=85)
    out += [
        f"- worst pages (lowest F1@0.5 with >= {WORST_MIN_CARE} care lines): "
        "![worst pages](overlays/worst_pages.jpg)",
        "",
    ]
    return out


# ---------------------------------------------------------------------------------- report


def write_det_report(run_dir: Path, report_dir: Path, cfg: RunConfig) -> Path:
    """Write report.md and its figures from the run's files on disk."""
    run_dir, report_dir = Path(run_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    recs = read_jsonl(run_dir / "metrics.jsonl")
    s = _read_json(run_dir / "summary.json") or {}
    steps = [r for r in recs if r.get("type") == "step"]
    epochs = [r for r in recs if r.get("type") == "epoch"]
    full = {tag: _read_json(run_dir / f"eval_full_{tag}.json") for tag in ("best", "last")}
    outcome = s.get("full_eval") or {}
    for tag in full:
        # Only an evaluation this run's finish reports as done belongs in its report.
        if outcome and outcome.get(tag) != "done":
            full[tag] = None
    sanity_dir = Path(cfg.paths.reports_dir) / "training" / "sanity" / str(cfg.experiment_id)
    sanity = _read_json(sanity_dir / "sanity.json")
    ops = [op.name for op in cfg.det.post.operating_points] if cfg.det else []
    _curves(steps, epochs, ops, report_dir / "curves.png")

    lines = [
        f"# Run report: {cfg.experiment_id}",
        "",
        "Generated by `glypharis.training.det_report`; do not edit by hand.",
        "",
    ]
    code = int(s.get("exit_code", 0) or 0)
    if code:
        reason = {2: "non-finite loss or gradient", 3: "out of GPU memory"}.get(
            code, "see the run log"
        )
        lines += [
            f"> **Run failed: exit code {code} ({reason}).** The numbers below are "
            "per-epoch validation-subset values from `metrics.jsonl`; full validation was "
            "skipped.",
            "",
        ]
    lines += _run_section(s, cfg) if s else ["## Run", "", f"summary.json: {NA}.", ""]

    best_point = s.get("best") or None
    final_point = s.get("final") or (epochs[-1] if epochs else None)
    lines += [
        "## Best and final epoch (validation subset)",
        "",
        "| | Epoch | Step | Operating point | P@0.5 | R@0.5 | F1@0.5 | F1@0.7 | Mean IoU | "
        "Val loss | Train F1@0.5 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        _epoch_row("best (selected)", best_point),
        _epoch_row("final", final_point),
        "",
        "`best.pt` is the arg-max of the selection metric, validation F1@0.5 maximised over "
        "the pre-registered operating points (protocol §7).",
        "",
    ]
    lines += _full_section(full, s, sanity)
    lines += _flags_section(s, epochs, recs)
    lines += ["## Curves", "", "![curves](curves.png)", ""]
    best_eval = full.get("best")
    lines += _stats_section(best_point, best_eval, ops)
    if best_eval is None:
        lines += [
            "## Slices",
            "",
            f"{NA} (no full validation of best.pt).",
            "",
            "## Overlays",
            "",
            f"{NA}.",
            "",
        ]
    else:
        try:
            lines += _slices_section(run_dir, report_dir, cfg, best_eval)
        except Exception as exc:  # the rest of the report is still useful
            log.exception("slices and overlays failed")
            lines += ["## Slices", "", f"{NA} ({type(exc).__name__}: {exc}).", ""]
    path = report_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return path
