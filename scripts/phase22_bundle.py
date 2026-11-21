"""Phase 22 item 16: freeze the phone candidate's bundle and check it (GW-EFF-014).

    python scripts/phase22_bundle.py freeze [--threads 10]
    python scripts/phase22_bundle.py check [--threads 10]

``freeze`` reads the selection (`SELECTION`, GW-EFF-013), loads the v1 bundle on ONNX Runtime
(`load_bundle_onnx`, its hashes checked) and writes ``glypharis-phone-bundle-1``
(`glypharis.efficiency.phone_bundle`) to `BUNDLE_OUT` once: v1's ``detector.onnx`` byte for byte,
the selected arm's recognizer (checked against GW-EFF-009's hash, or the INT8 build's record),
``charset.json`` and the manifest. The manifest is copied to `REPORT_DIR`.

``check`` loads the bundle with `load_phone_bundle` alone, checks that every setting it shares with
v1 (detector spec, recognizer input, charset, temperature, crop settings, reading order, pixel cap)
equals the v1 bundle's, and runs Phase 16's reference check unchanged
(`phase16_reference.check_references`, the GW-EFF-006 references, their files and manifest checked
first) with Phase 17's tolerances: map at most 1e-4, logits at most 1e-3, text identical except near
ties on at most 0.1% of lines, page CER within 0.05 points. The recognizer reads the 5,000 lines in
calls of 256 under the bundle's batching rule. The result goes to `REPORT_DIR`/equivalence.json
once; exit status 1 if a tolerance fails, and then the bundle is not changed to pass (item 16).

The v1 code must equal the ``v1-code`` tag's (`assert_v1_code`). Validation data only.
"""

# ruff: noqa: E402  (the thread variables are set before numpy and torch are imported)

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from glypharis.efficiency.threads import set_threads

DEFAULT_THREADS = 10
THREADS_FLAG = "--threads"


def requested_threads(argv: Sequence[str]) -> int | None:
    for i, arg in enumerate(argv):
        value = None
        if arg == THREADS_FLAG:
            value = argv[i + 1] if i + 1 < len(argv) else ""
        elif arg.startswith(THREADS_FLAG + "="):
            value = arg.split("=", 1)[1]
        if value is not None:
            try:
                n = int(value)
            except ValueError:
                return None
            return n if n > 0 else None
    return DEFAULT_THREADS


_THREADS: dict[str, Any] | None = None
if __name__ == "__main__":
    _n = requested_threads(sys.argv[1:])
    if _n is not None:
        _THREADS = set_threads(_n)

import argparse
import dataclasses
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch
from phase16_reference import REF_DIR, REF_REPORT, check_references, load_reference
from run_baselines import git_state

from glypharis.data.download import sha256_file
from glypharis.efficiency.candidate import (
    EXPERIMENT as SELECTION_EXPERIMENT,
)
from glypharis.efficiency.candidate import (
    FUSED_SHA256,
    WORK_ROOT,
    check_arm,
    check_fused,
    recognizer_file,
)
from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.onnx_build import check_sessions, load_bundle_onnx
from glypharis.efficiency.phone_bundle import (
    MANIFEST,
    PHONE_BUNDLE_FORMAT,
    load_phone_bundle,
    read_phone_manifest,
    write_phone_bundle,
)
from glypharis.efficiency.threads import check_threads
from glypharis.efficiency.v1 import assert_v1_code
from glypharis.efficiency.variants import charset_sha256
from glypharis.pipeline.frozen import FREEZE_RECORD
from glypharis.pipeline.ocr import GlypharisOCR

EXPERIMENT = "GW-EFF-014"
V1_BUNDLE = Path("runs/release/glypharis-v1")
BUNDLE_OUT = Path("runs/release/glypharis-phone-candidate")
SELECTION = Path("reports/final_validation/candidate/GW-EFF-013/selection.json")
REPORT_DIR = Path("reports/final_validation/candidate/GW-EFF-014")


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _threads(n: int) -> dict[str, Any]:
    return _THREADS if _THREADS is not None else set_threads(n, require_fresh=False)


def selected_arm(path: Path = SELECTION) -> tuple[str, dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"{path.as_posix()} is missing: item 14 selects the arm first")
    doc = json.loads(path.read_text(encoding="utf-8"))
    arm = doc.get("selected")
    if not isinstance(arm, str):
        raise SystemExit(f"{path.as_posix()} names no selected arm")
    check_arm(arm)
    return arm, {"file": path.as_posix(), "sha256": sha256_file(path), "selected": arm}


def arm_source(arm: str) -> dict[str, Any]:
    """The arm's recognizer file, checked: GW-EFF-009's fused file, or the INT8 build's."""
    path = recognizer_file(arm)
    if arm == "fused-int8":
        record = json.loads((WORK_ROOT / arm / "build.json").read_text(encoding="utf-8"))
        if sha256_file(path) != record["output"]["sha256"]:
            raise SystemExit(f"{path.as_posix()} is not the file its INT8 record names")
        if record["input"]["sha256"] != FUSED_SHA256:
            raise SystemExit("the INT8 file was not made from GW-EFF-009's fused recognizer")
        experiment = "GW-EFF-013 (H7' recipe on GW-EFF-009's fused recognizer)"
    else:
        check_fused(path)
        experiment = "GW-EFF-009 (fused-attention)"
    return {"path": path.as_posix(), "sha256": sha256_file(path), "experiment": experiment}


def cmd_freeze(args: argparse.Namespace) -> int:
    threads = _threads(args.threads)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / "manifest.json"
    for path in (args.out, report):
        if path.exists():
            raise SystemExit(f"{path.as_posix()} exists: move it aside first")
    v1_code = assert_v1_code(".")
    arm, selection = selected_arm()
    source = arm_source(arm)
    v1 = load_bundle_onnx(V1_BUNDLE, args.threads)
    manifest = write_phone_bundle(
        args.out,
        V1_BUNDLE,
        v1,
        arm,
        recognizer_file(arm),
        {
            "experiment": EXPERIMENT,
            "selection": selection,
            "recognizer": source,
            "detector": {"from": f"{V1_BUNDLE.as_posix()}/detector.onnx", "same_bytes": True},
            "v1_code": v1_code["v1_code"],
            "threads": threads["requested"],
            "created_utc": _utc(),
            **git_state(),
        },
    )
    shutil.copyfile(args.out / MANIFEST, report)
    print(
        f"{PHONE_BUNDLE_FORMAT} for {arm} written to {args.out.as_posix()}; manifest SHA-256 "
        f"{sha256_file(args.out / MANIFEST)}; files {manifest['files']}"
    )
    return 0


def shared_settings(candidate: GlypharisOCR, v1: GlypharisOCR) -> dict[str, Any]:
    """Every setting the candidate shares with v1, side by side, and whether each is equal."""

    def of(ocr: GlypharisOCR) -> dict[str, Any]:
        rec = ocr.recognizer
        return {
            "detector.spec": dataclasses.asdict(ocr.detector.spec),
            "recognizer.input": [rec.height, rec.max_width, int(rec.model.stride)],
            "recognizer.classes": int(rec.model.num_classes),
            "recognizer.arch": json.loads(json.dumps(rec.arch)),
            "recognizer.charset_sha256": charset_sha256(rec.charset),
            "recognizer.temperature": rec.temperature,
            "crop": dataclasses.asdict(ocr.crop),
            "reading_order": ocr.order,
            "max_pixels": ocr.max_pixels,
        }

    a, b = of(candidate), of(v1)
    return {k: {"equal": a[k] == b[k], "candidate": a[k], "v1": b[k]} for k in a}


def cmd_check(args: argparse.Namespace) -> int:
    threads = _threads(args.threads)
    out = REPORT_DIR / "equivalence.json"
    if out.exists():
        raise SystemExit(f"{out.as_posix()} exists: move it aside first")
    v1_code = assert_v1_code(".")
    manifest = read_phone_manifest(args.out)
    if sha256_file(args.out / MANIFEST) != sha256_file(REPORT_DIR / "manifest.json"):
        raise SystemExit("the bundle's manifest differs from the committed copy")
    reference = load_reference(REF_DIR, REF_REPORT)
    if reference["provenance"].get("record_sha256") != sha256_file(FREEZE_RECORD):
        raise SystemExit("the references were written from another freeze record")
    torch.set_grad_enabled(False)
    ocr = load_phone_bundle(args.out, args.threads)
    v1 = load_bundle_onnx(V1_BUNDLE, args.threads)
    settings = shared_settings(ocr, v1)
    differ = sorted(k for k, v in settings.items() if not v["equal"])
    if differ:
        raise SystemExit(f"the bundle differs from v1 in {differ}: it is not the selected arm")
    del v1
    sessions = check_sessions(ocr, args.threads)
    before = check_threads(args.threads)
    started = _utc()
    res = check_references(ocr, args.data_root, REF_DIR, reference)
    ended = _utc()
    after = check_threads(args.threads)
    doc = {
        "experiment": EXPERIMENT,
        "what": "the phone candidate's bundle against the PyTorch v1 references (GW-EFF-006), "
        "Phase 17's tolerances",
        "version_label": manifest["version_label"],
        "arm": manifest["arm"],
        "batching": manifest["batching"],
        "selection_experiment": SELECTION_EXPERIMENT,
        "bundle": args.out.as_posix(),
        "bundle_manifest_sha256": sha256_file(args.out / MANIFEST),
        "bundle_files": manifest["files"],
        "shared_settings_equal_v1": True,
        "shared_settings": sorted(settings),
        "reference": REF_DIR.as_posix(),
        "reference_manifest_sha256": sha256_file(REF_DIR / "manifest.json"),
        "complete": reference["complete"],
        "backend": "onnxruntime (CPU)",
        "started_utc": started,
        "ended_utc": ended,
        "threads": threads,
        "threads_checked": {"before": before, "after": after},
        "session_options": sessions,
        "v1_code": v1_code,
        **git_state(),
        **res,
    }
    write_json_atomic(out, doc)
    t = res["tolerances"]
    print(
        f"{EXPERIMENT}: map {t['map_max_abs']['measured']:.2e}, logits "
        f"{t['logits_max_abs']['measured']:.2e}, text {t['text']['measured']}, page CER "
        f"{100 * t['page_cer']['measured']:+.3f} points; pass: {res['pass']}; written to "
        f"{out.as_posix()}"
    )
    return 0 if res["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "check"):
        p = sub.add_parser(name)
        p.add_argument(THREADS_FLAG, type=int, default=DEFAULT_THREADS)
        p.add_argument("--out", type=Path, default=BUNDLE_OUT, help="the bundle directory")
        p.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    return {"freeze": cmd_freeze, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
