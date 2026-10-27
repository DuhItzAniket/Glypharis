"""The installed package reads validation pages like the Phase 10 pipeline (Phase 18).

    python scripts/package_check.py --config configs/det/gw-det-001.yaml \
        --model runs/release/glypharis-0.1.0 --phase10 runs/GW-E2E-001-p5/predictions.jsonl \
        --out reports/package/check.json

Runs `GlypharisOCR.from_pretrained` (the shipped path: bundle, ONNX backend unless ``--backend``)
on the 400 Phase 5 validation pages and compares it with GW-E2E-001's predictions there (the
Phase 10 pipeline, built from the training runs): bag-of-lines page CER (strict) of both, scored
by `evaluate_page`, and the share of lines whose text and box agree. Passes when the page CER
differs by at most `CER_TOL` (registered in docs/PHASE_STATUS.md, Phase 18). Validation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from e2e_eval import read_predictions, score

from glypharis.config.loader import load_config
from glypharis.data.det_dataset import PageIndex, read_gt_pages
from glypharis.data.det_geometry import decode_page
from glypharis.eval.det_matching import baseline_doc_ids
from glypharis.eval.page_metrics import PredLine
from glypharis.pipeline import GlypharisOCR

CER_TOL = 0.0005
BOX_TOL = 1.0  # pixels: two boxes "agree" when every corner is within this


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--phase10", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    assert cfg.det is not None
    root = Path(cfg.paths.data_root)
    pages_file = root / cfg.det.data.val_pages[0]
    ids = baseline_doc_ids(PageIndex.build(pages_file, "validation", root).doc_ids())
    gt = read_gt_pages(pages_file, ids)
    ref = read_predictions(args.phase10)
    ocr = GlypharisOCR.from_pretrained(args.model, args.backend)
    ours: dict[str, list[PredLine]] = {}
    same_lines = total_lines = 0
    for d in ids:
        page = gt[d]
        res = ocr.predict(decode_page(root / page.image, (page.width, page.height)))
        ours[d] = [PredLine(ln.polygon, ln.text, ln.confidence) for ln in res.lines]
        theirs = ref.get(d, [])
        total_lines += max(len(theirs), len(ours[d]))
        for a, b in zip(ours[d], theirs, strict=False):
            close = len(a.polygon) == len(b.polygon) and bool(
                np.abs(np.asarray(a.polygon) - np.asarray(b.polygon)).max() <= BOX_TOL
            )
            same_lines += int(close and a.text == b.text)
    cer = {}
    for name, preds in (("package", ours), ("phase10", ref)):
        rs = score(preds, gt, ids, "strict")
        cer[name] = sum(r.bag_errors for r in rs) / sum(r.bag_ref_chars for r in rs)
    diff = cer["package"] - cer["phase10"]
    res = {
        "model": Path(args.model).name,  # the bundle name; never a local path
        "backend": args.backend,
        "model_version": ocr.model_version,
        "pages": len(ids),
        "page_cer_package": cer["package"],
        "page_cer_phase10": cer["phase10"],
        "difference": diff,
        "lines_same_text_and_box": same_lines,
        "lines": total_lines,
        "pass": abs(diff) <= CER_TOL,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(
        f"package {100 * cer['package']:.2f} vs Phase 10 {100 * cer['phase10']:.2f} page CER "
        f"({100 * diff:+.3f} points); {same_lines}/{total_lines} lines identical; "
        f"pass {res['pass']}"
    )
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
