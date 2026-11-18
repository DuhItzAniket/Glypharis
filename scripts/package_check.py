"""The installed package reads validation pages like the Phase 10 pipeline (Phase 18).

    python scripts/package_check.py --config configs/det/gw-det-001.yaml \
        --model runs/release/glypharis-0.1.0 --phase10 runs/GW-E2E-001-p5/predictions.jsonl \
        --out reports/package/check.json \
        [--experiment ID] [--backend onnx|torch] [--threads N] [--v1] \
        [--reference-sha256 HEX] [--expect-manifest-sha256 HEX]

Runs `GlypharisOCR.from_pretrained` (the shipped path: bundle, ONNX backend unless ``--backend``)
on the 400 Phase 5 validation pages and compares it with GW-E2E-001's predictions there (the
Phase 10 pipeline, built from the training runs): bag-of-lines page CER (strict) of both, scored
by `evaluate_page`, and the share of lines whose text and box agree. Passes when the page CER
differs by at most `CER_TOL` (registered in docs/PHASE_STATUS.md, Phase 18). Validation only.

``--model`` is a bundle directory or a bundle name (looked up with
`glypharis.export.bundle.find_bundle`). The bundle is loaded with ``cache=False``, and with
``threads=N`` when ``--threads`` is given. The v0.1 command line (GW-E2E-004) means what it did.

It refuses, before anything loads: an existing ``--out`` (the check is written once); with
``--v1``, code that differs from v1's tag (`assert_v1_code`) or a bundle that differs from the
freeze record (`export_check.bundle_problems`); a bundle whose manifest SHA-256 is not
``--expect-manifest-sha256``; and a reference predictions file whose SHA-256 is not
``--reference-sha256``. The two hashes are checked only when given.

Besides the comparison, the output records where it came from: the experiment ID, the start and
end times (UTC), the git commit and dirty flag, the bundle's manifest SHA-256 and version, the
reference file's name (relative to the data root or the working directory when under one) and
SHA-256, the backend, the requested threads and torch's intra- and inter-op threads at the end,
the Python, torch and ONNX Runtime versions, and the v1 code check (null without ``--v1``). No
field holds a local path: the model is its bundle name. The output is written in one step.
Exit status 0 on a pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from e2e_eval import read_predictions, score
from export_check import bundle_problems
from run_baselines import git_state

from glypharis.config.loader import load_config
from glypharis.data.det_dataset import PageIndex, read_gt_pages
from glypharis.data.det_geometry import decode_page
from glypharis.data.download import sha256_file
from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.v1 import V1CodeError, assert_v1_code
from glypharis.errors import ModelLoadError
from glypharis.eval.det_matching import baseline_doc_ids
from glypharis.eval.page_metrics import PredLine
from glypharis.export.bundle import find_bundle, read_manifest
from glypharis.pipeline import GlypharisOCR
from glypharis.pipeline.frozen import FREEZE_RECORD

CER_TOL = 0.0005
BOX_TOL = 1.0  # pixels: two boxes "agree" when every corner is within this


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _versions() -> dict[str, Any]:
    try:
        import onnxruntime as ort

        ort_version: str | None = ort.__version__
    except ImportError:
        ort_version = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "onnxruntime": ort_version}  # fmt: skip


def portable_name(path: Path, bases: list[Path]) -> str:
    """``path`` relative to the first of ``bases`` it lies under, else its file name; never a
    local absolute path."""
    full = path.resolve()
    for base in bases:
        try:
            return full.relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return full.name


def _v1_checks(bundle: Path) -> dict[str, Any]:
    """`assert_v1_code` on the working directory, then the bundle against the freeze record."""
    try:
        v1_code = assert_v1_code(".")
    except V1CodeError as exc:
        raise SystemExit(f"{exc}; --v1 runs v1 as tagged") from exc
    record = yaml.safe_load(Path(FREEZE_RECORD).read_text(encoding="utf-8"))
    try:
        manifest = read_manifest(bundle)
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc
    problems = bundle_problems(manifest, record)
    if problems:
        raise SystemExit("the bundle is not v1's: " + "; ".join(problems))
    return v1_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True, help="a bundle directory or a bundle name")
    parser.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--phase10", type=Path, required=True,
                        help="the reference predictions (JSONL)")  # fmt: skip
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--experiment", default=None, help="the experiment ID, recorded")
    parser.add_argument("--threads", type=int, default=None,
                        help="threads passed to from_pretrained")  # fmt: skip
    parser.add_argument("--v1", action="store_true",
                        help="refuse code or a bundle that is not v1's")  # fmt: skip
    parser.add_argument("--reference-sha256", default=None,
                        help="refuse a reference file with another SHA-256")  # fmt: skip
    parser.add_argument("--expect-manifest-sha256", default=None,
                        help="refuse a bundle whose manifest has another SHA-256")  # fmt: skip
    args = parser.parse_args(argv)
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be a positive integer")
    if args.out.exists():
        raise SystemExit(f"{args.out.as_posix()} exists: the check is written once")
    started = _utc()

    try:
        bundle = find_bundle(args.model)
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc
    v1_code = _v1_checks(bundle) if args.v1 else None
    manifest_sha = sha256_file(bundle / "manifest.json")
    want = args.expect_manifest_sha256
    if want is not None and manifest_sha != want.lower():
        raise SystemExit(f"{Path(args.model).name}: manifest SHA-256 {manifest_sha}, not the "
                         f"expected {want}")  # fmt: skip
    try:
        bundle_version = read_manifest(bundle).get("version")
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc
    ref_sha = sha256_file(args.phase10)
    want = args.reference_sha256
    if want is not None and ref_sha != want.lower():
        raise SystemExit(f"{args.phase10.name}: SHA-256 {ref_sha}, not the expected {want}")

    cfg = load_config(args.config)
    assert cfg.det is not None
    root = Path(cfg.paths.data_root)
    pages_file = root / cfg.det.data.val_pages[0]
    ids = baseline_doc_ids(PageIndex.build(pages_file, "validation", root).doc_ids())
    gt = read_gt_pages(pages_file, ids)
    ref = read_predictions(args.phase10)
    ocr = GlypharisOCR.from_pretrained(bundle, args.backend, threads=args.threads, cache=False)
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
    ended = _utc()
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
        "experiment": args.experiment,
        "cer_tolerance": CER_TOL,
        "box_tolerance_px": BOX_TOL,
        "started_utc": started,
        "ended_utc": ended,
        **git_state(),
        "config": portable_name(args.config, [Path.cwd()]),
        "bundle_version": bundle_version,
        "bundle_manifest_sha256": manifest_sha,
        "loaded_with": "GlypharisOCR.from_pretrained(bundle, backend, threads, cache=False)",
        "reference": portable_name(args.phase10, [root, Path.cwd()]),
        "reference_sha256": ref_sha,
        "threads": args.threads,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "versions": _versions(),
        "v1_code": v1_code,
    }
    write_json_atomic(args.out, res)
    print(
        f"package {100 * cer['package']:.2f} vs Phase 10 {100 * cer['phase10']:.2f} page CER "
        f"({100 * diff:+.3f} points); {same_lines}/{total_lines} lines identical; "
        f"pass {res['pass']}; written to {args.out.as_posix()}"
    )
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
