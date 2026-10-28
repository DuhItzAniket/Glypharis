"""Build the v0.1 model bundle and check ONNX against PyTorch on validation data (Phase 17).

    python scripts/export_check.py --config configs/det/gw-det-001.yaml \
        --det-checkpoint checkpoints/GW-DET-001/best.pt --rec-run runs/GW-REC-024 \
        --temperature 1.0 [--set det.post... --op ...] --bundle runs/release/glypharis-0.1.0 \
        --out reports/export/GW-EFF-003

Writes the bundle (`glypharis.export.bundle.write_bundle`) with the detector spec the config and
its overrides give, then runs the registered equivalence checks (docs/PHASE_STATUS.md, Phase 17),
loading both backends from the bundle itself:

1. detector: probability maps on the 110 Phase 5 profile pages, max |ONNX - PyTorch|;
2. recognizer: logits and greedy text on a seeded sample of `LINES` validation lines; a line whose
   text differs is a near-tie when, at the first frame whose top class differs, the PyTorch top-2
   gap is below `NEAR_TIE`;
3. end to end: bag-of-lines page CER (strict) of the whole pipeline on the 400 Phase 5 pages, from
   each backend, scored by `evaluate_page`.

The export counts as successful only if every check passes. Validation data only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from e2e_eval import score
from run_baselines import e2e_samples

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.data.det_dataset import PageIndex, read_gt_pages
from glypharis.data.det_geometry import decode_page
from glypharis.data.linepack import LinePackReader
from glypharis.eval.det_matching import baseline_doc_ids
from glypharis.eval.page_metrics import PredLine
from glypharis.export.bundle import load_bundle, write_bundle
from glypharis.models.det.detector import TextLineDetector
from glypharis.models.det.preprocess import prepare_page
from glypharis.pipeline.recognizer import TextRecognizer

LINES = 5000
SEED = 20261028
PROB_TOL = 1e-4
LOGIT_TOL = 1e-3
NEAR_TIE = 1e-3
NEAR_TIE_SHARE = 0.001
CER_TOL = 0.0005  # 0.05 points of page CER
PROFILE_PAGES = 110


def det_check(torch_ocr: Any, onnx_ocr: Any, root: Path) -> dict[str, Any]:
    worst, n = 0.0, 0
    for _, page in zip(range(PROFILE_PAGES), e2e_samples(root, "hiertext_val"), strict=False):
        rgb = decode_page(root / page.image, (page.width, page.height))
        image, _ = prepare_page(rgb, torch_ocr.detector.spec.infer.long_side)
        a = torch_ocr.detector.backend(image)
        b = onnx_ocr.detector.backend(image)
        worst = max(worst, float(np.abs(a - b).max()))
        n += 1
    return {"pages": n, "prob_max_abs_diff": worst, "pass": worst <= PROB_TOL}


def rec_check(torch_rec: TextRecognizer, onnx_rec: TextRecognizer, pack: Path) -> dict[str, Any]:
    reader = LinePackReader(pack)
    idx = np.sort(
        np.random.default_rng(SEED).choice(len(reader), min(LINES, len(reader)), replace=False)
    )
    worst, same, near, other = 0.0, 0, 0, []
    for start in range(0, len(idx), 256):
        crops = [reader.image(int(i)) for i in idx[start : start + 256]]
        for la, lb in zip(torch_rec.logits(crops), onnx_rec.logits(crops), strict=True):
            worst = max(worst, float((la - lb).abs().max()))
            ta = torch_rec.charset.ctc_greedy_decode(la.argmax(-1).tolist())
            tb = torch_rec.charset.ctc_greedy_decode(lb.argmax(-1).tolist())
            if ta == tb:
                same += 1
                continue
            frame = int(torch.nonzero(la.argmax(-1) != lb.argmax(-1))[0])
            top2 = la[frame].topk(2).values
            gap = float(top2[0] - top2[1])
            if gap < NEAR_TIE:
                near += 1
            else:
                other.append({"frame": frame, "gap": gap, "torch": ta, "onnx": tb})
    n = len(idx)
    ok = worst <= LOGIT_TOL and not other and near <= NEAR_TIE_SHARE * n
    return {
        "lines": n,
        "logit_max_abs_diff": worst,
        "identical_text": same,
        "near_tie_differences": near,
        "other_differences": other,
        "pass": ok,
    }


def e2e_check(torch_ocr: Any, onnx_ocr: Any, root: Path, pages_file: Path) -> dict[str, Any]:
    ids = baseline_doc_ids(PageIndex.build(pages_file, "validation", root).doc_ids())
    gt = read_gt_pages(pages_file, ids)
    preds: dict[str, dict[str, list[PredLine]]] = {"torch": {}, "onnx": {}}
    for d in ids:
        page = gt[d]
        rgb = decode_page(root / page.image, (page.width, page.height))
        for name, ocr in (("torch", torch_ocr), ("onnx", onnx_ocr)):
            res = ocr.predict(rgb)
            preds[name][d] = [PredLine(ln.polygon, ln.text, ln.confidence) for ln in res.lines]
    cer = {}
    for name in preds:
        rs = score(preds[name], gt, ids, "strict")
        cer[name] = sum(r.bag_errors for r in rs) / sum(r.bag_ref_chars for r in rs)
    diff = cer["onnx"] - cer["torch"]
    return {
        "pages": len(ids),
        "page_cer_torch": cer["torch"],
        "page_cer_onnx": cer["onnx"],
        "difference": diff,
        "pass": abs(diff) <= CER_TOL,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--det-checkpoint", required=True)
    parser.add_argument("--op", default=None)
    parser.add_argument("--rec-run", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--version", default="0.1.0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_grad_enabled(False)
    cfg = load_config(args.config, overrides_from_args(args))
    assert cfg.det is not None
    root = Path(cfg.paths.data_root)
    det = TextLineDetector.from_checkpoint(args.det_checkpoint, "cpu")
    spec = dataclasses.replace(
        det.spec,
        infer=cfg.det.infer,
        post=cfg.det.post,
        operating_point=args.op or det.spec.operating_point,
    )
    rec = TextRecognizer.from_run(args.rec_run, temperature=args.temperature)
    sources = {
        "detector": {
            "experiment": cfg.experiment_id,
            "checkpoint": args.det_checkpoint,
            "settings": "Phase 9: GW-DET-004 post-processing, GW-DET-005 resolution",
        },
        "recognizer": {
            "run": args.rec_run.name,
            "temperature": "GW-E2E-003" if args.temperature != 1.0 else "1",
        },
    }
    write_bundle(args.det_checkpoint, spec, rec, args.bundle, args.version, sources)
    torch_ocr = load_bundle(args.bundle, "torch")
    onnx_ocr = load_bundle(args.bundle, "onnx")
    pack = root / load_config(args.rec_run / "config.resolved.yaml").data.val[0]
    pages_file = root / cfg.det.data.val_pages[0]
    res = {
        "bundle": args.bundle.name,
        "version": args.version,
        "detector": det_check(torch_ocr, onnx_ocr, root),
        "recognizer": rec_check(torch_ocr.recognizer, onnx_ocr.recognizer, pack),
        "end_to_end": e2e_check(torch_ocr, onnx_ocr, root, pages_file),
        "tolerances": {
            "prob": PROB_TOL,
            "logits": LOGIT_TOL,
            "near_tie": NEAR_TIE,
            "near_tie_share": NEAR_TIE_SHARE,
            "page_cer": CER_TOL,
        },
    }
    res["export_successful"] = all(res[k]["pass"] for k in ("detector", "recognizer", "end_to_end"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "equivalence.json").write_text(
        json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    d, r, e = res["detector"], res["recognizer"], res["end_to_end"]
    md = [
        "# ONNX equivalence of the v0.1 bundle (GW-EFF-003)",
        "",
        "Generated by `scripts/export_check.py`; do not edit by hand. Both backends loaded from",
        "the bundle itself; validation data only. Tolerances registered in `docs/PHASE_STATUS.md`",
        "(Phase 17) before this ran.",
        "",
        "| Check | Data | Measured | Tolerance | Pass |",
        "|---|---|---|---|---|",
        f"| Detector probability map, max abs diff | {d['pages']} pages | "
        f"{d['prob_max_abs_diff']:.2e} | {PROB_TOL:g} | {'yes' if d['pass'] else 'no'} |",
        f"| Recognizer logits, max abs diff | {r['lines']:,} lines | "
        f"{r['logit_max_abs_diff']:.2e} | {LOGIT_TOL:g} | {'yes' if r['pass'] else 'no'} |",
        f"| Recognizer text identical | {r['lines']:,} lines | {r['identical_text']:,} "
        f"(near ties {r['near_tie_differences']}, other {len(r['other_differences'])}) | other 0, "
        f"near ties at most {NEAR_TIE_SHARE:.1%} | {'yes' if r['pass'] else 'no'} |",
        f"| Page CER (bag of lines, strict), ONNX - PyTorch | {e['pages']} pages | "
        f"{100 * e['difference']:+.3f} points ({100 * e['page_cer_onnx']:.2f} vs "
        f"{100 * e['page_cer_torch']:.2f}) | ±{100 * CER_TOL:.2f} points | "
        f"{'yes' if e['pass'] else 'no'} |",
        "",
        f"Export successful: **{'yes' if res['export_successful'] else 'no'}**.",
        "",
    ]
    (args.out / "equivalence.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    print("\n".join(md[6:]))
    return 0 if res["export_successful"] else 1


if __name__ == "__main__":
    sys.exit(main())
