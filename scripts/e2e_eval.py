"""End-to-end evaluation of the GLYPHARIS pipeline on HierText validation pages (Phase 10).

    python scripts/e2e_eval.py --config configs/det/gw-det-001.yaml \
        --det-checkpoint checkpoints/GW-DET-001/best.pt --rec-run runs/GW-REC-024 \
        --experiment GW-E2E-001 --pages all --mode pipeline --out runs/GW-E2E-001 \
        [--set det.post...]

``--mode pipeline`` runs `GlypharisOCR` on each page; ``--mode oracle`` reads the ground-truth care
lines instead of detected ones (cut as the training crops were), which leaves recognition error
only (EXPERIMENT_PROTOCOL §3, oracle analysis). Pages: ``all`` 1,715 validation pages,
``baseline`` the 400 Phase 5 pages, ``A``/``B`` Phase 9's halves. Predictions go to
``<out>/predictions.jsonl`` in the baselines' format, one page per line, and a rerun continues an
unfinished file. Scoring is `glypharis.eval.page_metrics.evaluate_page`, the code that scored the
baselines, on both normalisation tracks: detection P/R/F1, bag-of-lines page CER and E2E line F1
with page-bootstrap CIs; HierText has no reading order, so order-aware metrics are n/a. With
``--baselines`` the Phase 5 systems' validation predictions on the same pages are scored by the
same code and paired with GLYPHARIS. ``<out>/lines.jsonl`` holds every predicted line's confidence
and correctness (matched at IoU 0.5 and exactly right on the strict track) for calibration.
Validation only: no test split is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.data.det_dataset import PageIndex, read_gt_pages
from glypharis.data.det_geometry import decode_page
from glypharis.data.pages import Page
from glypharis.data.text import normalize_eval
from glypharis.eval.det_matching import baseline_doc_ids, calibration_halves, detection_matches
from glypharis.eval.page_metrics import PageResult, PredLine, evaluate_page, is_box_line
from glypharis.eval.stats import paired_ratios_diff_ci, ratio_ci
from glypharis.models.det.detector import TextLineDetector
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer

TRACKS = ("strict", "normalised")
BASELINES = {"GW-BASE-001": "tesseract", "GW-BASE-002": "ppocr_mobile",
             "GW-BASE-005": "ppocr6_tiny", "GW-BASE-006": "ppocr6_small"}  # fmt: skip
BASELINE_NAMES = {"GW-BASE-001": "Tesseract 5", "GW-BASE-002": "PP-OCRv5 mobile",
                  "GW-BASE-005": "PP-OCRv6 tiny", "GW-BASE-006": "PP-OCRv6 small"}  # fmt: skip


def page_ids(pages_file: Path, root: Path, which: str) -> list[str]:
    ids = PageIndex.build(pages_file, "validation", root).doc_ids()
    if which == "baseline":
        return baseline_doc_ids(ids)
    if which in ("A", "B"):
        return calibration_halves(ids)["AB".index(which)]
    return ids


def build_ocr(args: argparse.Namespace) -> GlypharisOCR:
    cfg = load_config(args.config, overrides_from_args(args))
    assert cfg.det is not None
    import dataclasses

    det = TextLineDetector.from_checkpoint(args.det_checkpoint, args.device)
    spec = dataclasses.replace(det.spec, infer=cfg.det.infer, post=cfg.det.post,
                               operating_point=args.op or det.spec.operating_point)  # fmt: skip
    rec = TextRecognizer.from_run(args.rec_run, device=args.device, temperature=args.temperature)
    return GlypharisOCR(TextLineDetector(det.backend, spec), rec)


def predict(args: argparse.Namespace, gt: dict[str, Page], ids: list[str], root: Path) -> Path:
    out = args.out / "predictions.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                break  # a cut-off last record: rewritten below
        keep = [ln for ln in out.read_text(encoding="utf-8").splitlines()
                if ln.strip() and _ok(ln)]  # fmt: skip
        out.write_text("".join(k + "\n" for k in keep), encoding="utf-8", newline="\n")
    todo = [d for d in ids if d not in done]
    if not todo:
        return out
    ocr = build_ocr(args)
    temps = tuple(args.record_temperatures)
    with out.open("a", encoding="utf-8", newline="\n") as fh, torch.inference_mode():
        for n, doc_id in enumerate(todo, 1):
            page = gt[doc_id]
            rgb = decode_page(root / page.image, (page.width, page.height))
            t0 = time.perf_counter()
            if args.mode == "oracle":
                care = [ln for ln in page.lines if is_box_line(ln)]
                reads = ocr.read_quads(rgb, [ln.polygon for ln in care], temps)
                lines = [{"polygon": ln.polygon, "text": r.text, "confidence": r.confidence,
                          "det_score": 1.0, "conf_at": {str(t): c for t, c in r.at.items()}}
                         for ln, r in zip(care, reads, strict=True)]  # fmt: skip
            else:
                res = ocr.predict(rgb, temps)
                lines = [{"polygon": ln.polygon, "text": ln.text, "confidence": ln.confidence,
                          "det_score": ln.det_score,
                          "conf_at": {str(t): c for t, c in ln.conf_at.items()}}
                         for ln in res.lines]  # fmt: skip
            ms = (time.perf_counter() - t0) * 1000
            fh.write(json.dumps({"key": doc_id, "lines": lines, "time_ms": round(ms, 1)},
                                ensure_ascii=False) + "\n")  # fmt: skip
            fh.flush()
            if n % 100 == 0:
                print(f"{n}/{len(todo)} pages", flush=True)
    return out


def _ok(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError:
        return False


def read_predictions(path: Path) -> dict[str, list[PredLine]]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["key"]] = [PredLine(p["polygon"], p["text"], p.get("confidence"))
                             for p in r["lines"]]  # fmt: skip
    return out


def score(preds: dict[str, list[PredLine]], gt: dict[str, Page], ids: list[str],
          track: str) -> list[PageResult]:  # fmt: skip
    return [evaluate_page(gt[d], preds.get(d, []), (0.5, 0.7), track) for d in ids]  # type: ignore[arg-type]


def _a(rs: list[PageResult], f: Any) -> np.ndarray:
    return np.array([f(r) for r in rs], dtype=np.float64)


def _iv(num: np.ndarray, den: np.ndarray) -> dict[str, float] | None:
    if not float(den.sum()):
        return None
    iv = ratio_ci(num, den, "pages")
    return {"value": iv.value, "low": iv.low, "high": iv.high}


def metrics(rs: list[PageResult]) -> dict[str, Any]:
    tp, fp, fn = (_a(rs, lambda r, i=i: r.det[0.5][i]) for i in range(3))
    e2e = _a(rs, lambda r: r.e2e_tp)
    return {
        "pages": len(rs),
        "det_precision": _iv(tp, tp + fp), "det_recall": _iv(tp, tp + fn),
        "det_f1": _iv(2 * tp, 2 * tp + fp + fn),
        "page_cer_bag_of_lines": _iv(
            _a(rs, lambda r: r.bag_errors), _a(rs, lambda r: r.bag_ref_chars)
        ),
        "e2e_f1": _iv(2 * e2e, _a(rs, lambda r: r.e2e_pred + r.e2e_gt)),
    }  # fmt: skip


def paired(a: list[PageResult], b: list[PageResult]) -> dict[str, Any]:
    """A minus B on the same pages: bag-of-lines CER and E2E F1."""

    def d(
        num_a: np.ndarray, den_a: np.ndarray, num_b: np.ndarray, den_b: np.ndarray
    ) -> dict[str, float]:
        iv = paired_ratios_diff_ci(num_a, den_a, num_b, den_b, "pages")
        return {"value": iv.value, "low": iv.low, "high": iv.high}

    return {
        "page_cer_bag_of_lines": d(_a(a, lambda r: r.bag_errors), _a(a, lambda r: r.bag_ref_chars),
                                   _a(b, lambda r: r.bag_errors), _a(b, lambda r: r.bag_ref_chars)),
        "e2e_f1": d(2 * _a(a, lambda r: r.e2e_tp), _a(a, lambda r: r.e2e_pred + r.e2e_gt),
                    2 * _a(b, lambda r: r.e2e_tp), _a(b, lambda r: r.e2e_pred + r.e2e_gt)),
    }  # fmt: skip


def line_records(preds: dict[str, list[PredLine]], raw: dict[str, list[dict[str, Any]]],
                 gt: dict[str, Page], ids: list[str]) -> list[dict[str, Any]]:  # fmt: skip
    """Every predicted line: its confidences and whether it is right (matched at IoU 0.5 to a care
    line and exactly equal to it on the strict track). Lines on don't-care text are left out."""
    out = []
    for d in ids:
        page, ps = gt[d], preds.get(d, [])
        m = detection_matches(page, ps, 0.5)
        for i, (p, status) in enumerate(zip(ps, m.pred_status, strict=True)):
            if status == "ignored":
                continue
            g = m.pred_gt[i]
            ok = bool(status == "tp" and g is not None and normalize_eval(p.text, "strict")
                      == normalize_eval(page.lines[g].text, "strict"))  # fmt: skip
            out.append({"doc_id": d, "confidence": p.confidence, "correct": ok,
                        "conf_at": raw[d][i].get("conf_at", {})})  # fmt: skip
    return out


def baseline_predictions(root: Path, exp: str) -> dict[str, list[PredLine]] | None:
    path = root / "processed" / "baselines" / exp / f"{BASELINES[exp]}_e2e_hiertext_val.jsonl"
    if not path.exists():
        return None
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["key"]] = [PredLine(p["polygon"], p["text"], p.get("confidence"))
                             for p in r["lines"]]  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--op", default=None)
    parser.add_argument("--rec-run", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--record-temperatures", type=float, nargs="*", default=[])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--pages", choices=["all", "baseline", "A", "B"], default="all")
    parser.add_argument("--mode", choices=["pipeline", "oracle"], default="pipeline")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.device = (
        "cuda" if torch.cuda.is_available() and args.device in (None, "auto", "cuda") else "cpu"
    )
    cfg = load_config(args.config, overrides_from_args(args))
    assert cfg.det is not None
    root = Path(cfg.paths.data_root)
    pages_file = root / cfg.det.data.val_pages[0]
    ids = page_ids(pages_file, root, args.pages)
    gt = read_gt_pages(pages_file, ids)
    path = predict(args, gt, ids, root)
    preds = read_predictions(path)
    raw = {json.loads(ln)["key"]: json.loads(ln)["lines"]
           for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}  # fmt: skip
    res: dict[str, Any] = {"experiment": args.experiment, "mode": args.mode, "pages": args.pages,
                           "n_pages": len(ids), "det_checkpoint": args.det_checkpoint,
                           "rec_run": args.rec_run.as_posix(), "temperature": args.temperature,
                           "det_overrides": overrides_from_args(args), "tracks": {}}  # fmt: skip
    times = [json.loads(ln)["time_ms"] for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip()]  # fmt: skip
    res["time_ms_mean_gpu"] = float(np.mean(times)) if times else None
    for track in TRACKS:
        ours = score(preds, gt, ids, track)
        entry: dict[str, Any] = {"glypharis": metrics(ours), "baselines": {}}
        if args.baselines:
            for exp in BASELINES:
                bp = baseline_predictions(root, exp)
                if bp is None or not set(ids) <= set(bp):
                    continue
                theirs = score(bp, gt, ids, track)
                entry["baselines"][exp] = {
                    "name": BASELINE_NAMES[exp],
                    **metrics(theirs),
                    "glypharis_minus_this": paired(ours, theirs),
                }
        res["tracks"][track] = entry
    lines = line_records(preds, raw, gt, ids)
    with (args.out / "lines.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in lines:
            fh.write(json.dumps(r) + "\n")
    (args.out / "e2e.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8",
                                       newline="\n")  # fmt: skip
    g = res["tracks"]["strict"]["glypharis"]
    print(f"{args.experiment} {args.mode} on {len(ids)} pages (strict): bag-of-lines CER "
          f"{g['page_cer_bag_of_lines']['value']:.4f}, E2E F1 {g['e2e_f1']['value']:.4f}, "
          f"det F1 {g['det_f1']['value']:.4f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
