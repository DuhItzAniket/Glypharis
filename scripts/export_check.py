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

    python scripts/export_check.py --frozen --out runs/release/glypharis-v1 \
        [--version glypharis_v1] [--report reports/efficiency/phase16/v1_bundle.json]

``--frozen`` (Phase 16, GW-EFF-006) builds the v1 bundle from the freeze record instead: the
pipeline from `load_frozen("cpu")` (every setting the record's, hashes checked), the detector
release from the record's checkpoint, and the manifest's ``sources`` from the record and the
build's provenance. The v1 code paths must equal the ``v1-code`` tag's
(`glypharis.efficiency.v1.assert_v1_code`); GW-EFF-006 itself ran against ``phase14-eval``, as the
manifest's ``sources.v1_code`` records. Once written, the manifest's detector spec, the
recognizer's height, maximum width and temperature, and the source checkpoint hashes are compared
with the record (`bundle_problems`); a bundle that differs gets its manifest renamed to
``manifest.rejected.json``, so nothing loads it, and the run fails. An existing bundle or
``--report`` is never overwritten: the report names the bundle by its manifest SHA-256 and the
time it was written (``created_utc``), which the Phase 16 scorer requires to come before H1′'s
sitting. The equivalence checks against the PyTorch reference outputs are
`scripts/phase16_reference.py check`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from e2e_eval import score
from run_baselines import e2e_samples

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.data.det_dataset import PageIndex, read_gt_pages
from glypharis.data.det_geometry import decode_page
from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackReader
from glypharis.data.pages import Page
from glypharis.data.text import Charset
from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.v1 import V1CodeError, assert_v1_code
from glypharis.eval.det_matching import baseline_doc_ids
from glypharis.eval.page_metrics import PredLine
from glypharis.export.bundle import load_bundle, read_manifest, write_bundle
from glypharis.models.det.detector import TextLineDetector
from glypharis.models.det.preprocess import prepare_page
from glypharis.pipeline.frozen import FREEZE_RECORD, load_frozen
from glypharis.pipeline.recognizer import TextRecognizer

LINES = 5000
SEED = 20261028
PROB_TOL = 1e-4
LOGIT_TOL = 1e-3
NEAR_TIE = 1e-3
NEAR_TIE_SHARE = 0.001
CER_TOL = 0.0005  # 0.05 points of page CER
PROFILE_PAGES = 110
FROZEN_FLAG = "--frozen"


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
    argv = sys.argv[1:] if argv is None else list(argv)
    if FROZEN_FLAG in argv:
        return main_frozen(argv)
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


# ------------------------------------------------------------- Phase 16: v1 from the freeze record
# The item lists below are the ones det_check, rec_check and e2e_check read, written out so that
# scripts/phase16_reference.py reads exactly the same items (tests keep them equal).


def profile_pages(root: Path) -> list[Page]:
    """`det_check`'s pages: the first `PROFILE_PAGES` of the Phase 5 HierText validation sample."""
    samples = e2e_samples(root, "hiertext_val")
    return [page for _, page in zip(range(PROFILE_PAGES), samples, strict=False)]


def line_indices(n: int) -> np.ndarray:
    """`rec_check`'s lines of a pack of ``n``: `LINES` indices drawn with `SEED`, sorted."""
    return np.sort(np.random.default_rng(SEED).choice(n, min(LINES, n), replace=False))


def phase5_page_ids(root: Path, pages_file: Path) -> list[str]:
    """`e2e_check`'s pages: the 400 Phase 5 validation pages, in pages-file order."""
    return baseline_doc_ids(PageIndex.build(pages_file, "validation", root).doc_ids())


def detector_input(page: Page, root: Path, long_side: int) -> np.ndarray:
    """The padded network input `det_check` gives both backends for ``page``."""
    rgb = decode_page(root / page.image, (page.width, page.height))
    image, _ = prepare_page(rgb, long_side)
    return image


def text_difference(
    ref: torch.Tensor, other: torch.Tensor, charset: Charset
) -> dict[str, Any] | None:
    """None when the greedy texts of two ``(T, classes)`` logits agree; otherwise where they part,
    as `rec_check` judges it: the first frame whose top class differs and the reference's top-2 gap
    there, a near tie when the gap is below `NEAR_TIE`. Logits of different shapes are never a
    near tie."""
    ta = charset.ctc_greedy_decode(ref.argmax(-1).tolist())
    tb = charset.ctc_greedy_decode(other.argmax(-1).tolist())
    if ref.shape != other.shape:
        return {"frame": None, "gap": None, "near_tie": False, "torch": ta, "onnx": tb,
                "shapes": [list(ref.shape), list(other.shape)]}  # fmt: skip
    if ta == tb:
        return None
    frame = int(torch.nonzero(ref.argmax(-1) != other.argmax(-1))[0])
    top2 = ref[frame].topk(2).values
    gap = float(top2[0] - top2[1])
    return {"frame": frame, "gap": gap, "near_tie": gap < NEAR_TIE, "torch": ta, "onnx": tb}


def bundle_problems(manifest: dict[str, Any], record: dict[str, Any]) -> list[str]:
    """How a bundle manifest differs from the freeze record: the detector's resolution,
    post-processing, operating point and label version, the recognizer's input height, maximum
    width and temperature, and the source checkpoints' SHA-256 (empty when it is v1's)."""
    det, rec = record["detector"], record["recognizer"]
    spec = (manifest.get("detector") or {}).get("spec") or {}
    mrec = manifest.get("recognizer") or {}
    sources = manifest.get("sources") or {}
    pairs = {
        "detector resolution": (spec.get("infer"), det["infer"]),
        "detector post-processing": (spec.get("post"), det["post"]),
        "detector operating point": (spec.get("operating_point"), det["operating_point"]),
        "detector label version": (spec.get("label_version"), det["label_version"]),
        "recognizer height": (mrec.get("height"), int(rec["input"]["height"])),
        "recognizer maximum width": (mrec.get("max_width"), int(rec["input"]["max_width"])),
        "recognizer temperature": (mrec.get("temperature"), float(rec["temperature"])),
        "detector checkpoint SHA-256": (
            (sources.get("detector") or {}).get("checkpoint_sha256"),
            det["checkpoint_sha256"],
        ),
        "recognizer checkpoint SHA-256": (
            (sources.get("recognizer") or {}).get("checkpoint_sha256"),
            rec["checkpoint_sha256"],
        ),
    }
    return [f"{k}: {got!r} in the bundle, {want!r} in the record"
            for k, (got, want) in pairs.items() if got != want]  # fmt: skip


def frozen_sources(
    record: dict[str, Any], provenance: dict[str, Any], v1_code: dict[str, Any]
) -> dict[str, Any]:
    """The manifest's ``sources`` for the v1 bundle, from the record and the build's provenance."""
    det, rec = record["detector"], record["recognizer"]
    return {
        "exported_by": "scripts/export_check.py --frozen (GW-EFF-006)",
        "record": provenance["record"],
        "record_sha256": provenance["record_sha256"],
        "name": record.get("name"),
        "detector": {
            "experiment": det["experiment"],
            "config": det.get("config"),
            "checkpoint": det["checkpoint"],
            "checkpoint_sha256": det["checkpoint_sha256"],
            "settings": det.get("settings_from"),
        },
        "recognizer": {
            "experiment": rec["experiment"],
            "config": rec.get("config"),
            "checkpoint": rec["checkpoint"],
            "checkpoint_sha256": rec["checkpoint_sha256"],
            "charset": rec["charset"],
            "charset_sha256": provenance["charset_sha256"],
            "temperature": rec["temperature"],
            "temperature_from": rec.get("temperature_from"),
        },
        "build": {k: provenance.get(k) for k in ("torch", "numerics", "git_commit", "git_dirty")},
        "v1_code": v1_code,
    }


def main_frozen(argv: list[str]) -> int:
    """GW-EFF-006: the v1 bundle from the freeze record (the module docstring)."""
    parser = argparse.ArgumentParser(description="Build the v1 bundle from the freeze record")
    parser.add_argument(FROZEN_FLAG, action="store_true", help="build from the freeze record")
    parser.add_argument("--out", type=Path, required=True, help="the bundle directory")
    parser.add_argument("--version", default=None, help="manifest version (the record's name)")
    parser.add_argument("--report", type=Path, default=None,
                        help="also write the bundle's hashes and checks here (json)")  # fmt: skip
    parser.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    args = parser.parse_args(argv)
    root, out = args.root, args.out
    if (out / "manifest.json").exists() or (out / "manifest.rejected.json").exists():
        raise SystemExit(f"{out.as_posix()} already holds a bundle: move it aside first")
    if args.report is not None and args.report.exists():
        # The report is what names the GW-EFF-006 bundle; it is written once.
        raise SystemExit(f"{args.report.as_posix()} exists: move it aside first")
    try:
        v1_code = assert_v1_code(root)
    except V1CodeError as exc:
        raise SystemExit(f"{exc}; the v1 bundle is built from the tag's code") from exc
    torch.set_grad_enabled(False)
    frozen = load_frozen("cpu", root=root)
    record = yaml.safe_load((root / FREEZE_RECORD).read_text(encoding="utf-8"))
    version = args.version or str(record.get("name"))
    sources = frozen_sources(record, frozen.provenance, v1_code)
    det_checkpoint = root / frozen.provenance["detector_checkpoint"]
    write_bundle(det_checkpoint, frozen.ocr.detector.spec, frozen.recognizer, out, version,
                 sources)  # fmt: skip
    manifest = read_manifest(out)
    problems = bundle_problems(manifest, record)
    release = torch.load(out / "detector.pt", map_location="cpu", weights_only=True)
    if release.get("source_sha256") != record["detector"]["checkpoint_sha256"]:
        problems.append(f"detector.pt was written from SHA-256 {release.get('source_sha256')}, "
                        f"the record says {record['detector']['checkpoint_sha256']}")  # fmt: skip
    if problems:
        (out / "manifest.json").rename(out / "manifest.rejected.json")
        raise SystemExit("the bundle differs from the freeze record (manifest renamed to "
                         "manifest.rejected.json): " + "; ".join(problems))  # fmt: skip
    summary = {
        "experiment": "GW-EFF-006",
        "bundle": out.as_posix(),
        "version": version,
        "manifest_sha256": sha256_file(out / "manifest.json"),
        "files": manifest["files"],
        "record": frozen.provenance["record"],
        "record_sha256": frozen.provenance["record_sha256"],
        "equals_record": True,
        "v1_code": v1_code,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if args.report is not None:
        if args.report.exists():
            raise SystemExit(f"{args.report.as_posix()} exists: move it aside first")
        write_json_atomic(args.report, summary, ensure_ascii=True)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
