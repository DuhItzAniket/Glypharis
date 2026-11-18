"""The v1 bundle read through the PyTorch backend, checked against the Phase 16 PyTorch v1
references (GW-EFF-012, Phase 17 re-run for v1, item 4).

    python scripts/phase17_v1_check.py --bundle runs/release/glypharis-v1 [--threads 10]

v1 ships as the GW-EFF-006 bundle on the PyTorch backend (ADR-0011, owner decision of 2026-11-18).
GW-EFF-006 checked the bundle's ONNX files; this checks the path that ships: the bundle loaded with
`GlypharisOCR.from_pretrained(bundle, backend="torch", threads=N, cache=False)`, its release files
read by the PyTorch modules, run on the CPU at ``--threads`` torch threads (the OpenMP and MKL
variables set before numpy and torch load).

It refuses, before anything runs: v1 code that differs from its tag (`assert_v1_code`); a bundle
whose manifest SHA-256 is not ``--expect-manifest-sha256`` (the GW-EFF-006 bundle's by default)
or that differs from the freeze record (`export_check.bundle_problems`); references whose files
differ from their hashes, whose manifest differs from the committed copy, or that were written from
another freeze record; and an existing output. It then runs Phase 16's reference check unchanged
(`phase16_reference.check_references`): the same 110 profile pages, 5,000 seeded lines and 400
Phase 5 pages, and Phase 17's four tolerances (map max abs difference at most 1e-4; logits at most
1e-3; text identical except near ties on at most 0.1% of lines; page CER within 0.05 points).

Phase 16's check names the two sides ``torch`` (the references) and ``onnx`` (the bundle). Here
both are PyTorch, so the report renames them ``reference`` and ``bundle`` (`relabel`); nothing
else in the result changes. Output: reports/export/GW-EFF-012/torch_bundle_check.json, written
once. Exit status 0 when all four tolerances hold, 1 otherwise.
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
    """``--threads`` in ``argv`` (``--threads N`` or ``--threads=N``), `DEFAULT_THREADS` when it is
    absent, None when it is not a positive integer (argparse reports it later)."""
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


# Set by a command-line run before numpy and torch load; None when the module is imported.
_THREADS: dict[str, Any] | None = None
if __name__ == "__main__":
    _n = requested_threads(sys.argv[1:])
    if _n is not None:
        _THREADS = set_threads(_n)

import argparse
from datetime import UTC, datetime
from pathlib import Path

import phase16_reference as ref
import torch
import yaml
from export_check import bundle_problems
from run_baselines import git_state

from glypharis.data.download import sha256_file
from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.threads import check_threads
from glypharis.efficiency.v1 import V1CodeError, assert_v1_code
from glypharis.errors import ModelLoadError
from glypharis.export.bundle import read_manifest
from glypharis.pipeline.frozen import FREEZE_RECORD
from glypharis.pipeline.ocr import GlypharisOCR

EXPERIMENT = "GW-EFF-012"
# The GW-EFF-006 bundle, named before this check (docs/PHASE_STATUS.md, Phase 17 v1 re-run).
V1_BUNDLE_SHA256 = "875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037"
OUT = Path("reports/export/GW-EFF-012/torch_bundle_check.json")
# Phase 16's names for the two sides, and what they are here (both PyTorch).
LABELS = {
    "torch": "reference",
    "onnx": "bundle",
    "page_cer_torch": "page_cer_reference",
    "page_cer_onnx": "page_cer_bundle",
}


def relabel(obj: Any) -> Any:
    """``obj`` with every dict key in `LABELS` renamed, at any depth; values are untouched."""
    if isinstance(obj, dict):
        return {LABELS.get(k, k): relabel(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relabel(v) for v in obj]
    return obj


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _threads(n: int, require_fresh: bool) -> dict[str, Any]:
    if _THREADS is not None:
        if _THREADS["requested"] != n:
            raise SystemExit(f"threads were set to {_THREADS['requested']}, the run asks for {n}")
        return _THREADS
    if require_fresh:
        raise SystemExit("run this as a script: the thread variables are set before numpy and "
                         "torch load")  # fmt: skip
    return set_threads(n, require_fresh=False)


def _v1_code(root: Path) -> dict[str, Any]:
    try:
        return assert_v1_code(root)
    except V1CodeError as exc:
        raise SystemExit(f"{exc}; {EXPERIMENT} runs v1 as tagged") from exc


def run(args: argparse.Namespace, require_fresh: bool) -> int:
    if args.out.exists():
        raise SystemExit(f"{args.out.as_posix()} exists: the check is written once")
    threads = _threads(args.threads, require_fresh)
    v1_code = _v1_code(args.root)
    manifest_sha = sha256_file(args.bundle / "manifest.json")
    if manifest_sha != args.expect_manifest_sha256:
        raise SystemExit(f"{args.bundle.as_posix()}: manifest SHA-256 {manifest_sha}, not the "
                         f"registered bundle's {args.expect_manifest_sha256}")  # fmt: skip
    record_path = args.root / FREEZE_RECORD
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    try:
        bundle_manifest = read_manifest(args.bundle)
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc
    problems = bundle_problems(bundle_manifest, record)
    if problems:
        raise SystemExit("the bundle is not v1's: " + "; ".join(problems))
    reference = ref.load_reference(args.reference, args.manifest)
    if reference["provenance"].get("record_sha256") != sha256_file(record_path):
        raise SystemExit("the references were written from another freeze record")
    torch.set_grad_enabled(False)
    try:
        ocr = GlypharisOCR.from_pretrained(args.bundle, backend="torch", threads=args.threads,
                                           cache=False)  # fmt: skip
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc
    before = check_threads(args.threads)
    started = _utc()
    res = ref.check_references(ocr, args.data_root, args.reference, reference)
    ended = _utc()
    after = check_threads(args.threads)
    verdict = {
        "experiment": EXPERIMENT,
        "what": "the v1 bundle on the PyTorch backend against the PyTorch v1 references "
        "(Phase 17's tolerances)",
        "labels": "Phase 16's 'torch' and 'onnx' sides are named 'reference' and 'bundle' here",
        "loaded_with": "GlypharisOCR.from_pretrained(bundle, backend='torch', threads, "
        "cache=False)",
        "bundle": args.bundle.as_posix(),
        "bundle_version": bundle_manifest.get("version"),
        "model_version": ocr.model_version,
        "bundle_manifest_sha256": manifest_sha,
        "bundle_files": bundle_manifest["files"],
        "bundle_equals_record": True,
        "reference": args.reference.as_posix(),
        "reference_manifest_sha256": sha256_file(args.reference / "manifest.json"),
        "reference_files": reference["files"],
        "complete": reference["complete"],
        "limit": reference["limit"],
        "backend": f"torch {torch.__version__} (CPU)",
        "started_utc": started,
        "ended_utc": ended,
        "threads": threads,
        "threads_checked": {"before": before, "after": after},
        "v1_code": v1_code,
        "versions": ref._versions(),
        **git_state(),
        **relabel(res),
    }
    write_json_atomic(args.out, verdict)
    t = res["tolerances"]
    print(f"{EXPERIMENT} check of {args.bundle.as_posix()} (torch): map "
          f"{t['map_max_abs']['measured']:.2e}, logits {t['logits_max_abs']['measured']:.2e}, "
          f"text {t['text']['measured']}, page CER {100 * t['page_cer']['measured']:+.3f} points; "
          f"pass: {res['pass']}; written to {args.out.as_posix()}")  # fmt: skip
    return 0 if res["pass"] else 1


def main(argv: list[str] | None = None, *, require_fresh: bool = True) -> int:
    """``require_fresh=False`` lets a process that already holds torch (the tests) run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(THREADS_FLAG, type=int, default=DEFAULT_THREADS,
                        help="torch threads and the OpenMP/MKL variables")  # fmt: skip
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expect-manifest-sha256", default=V1_BUNDLE_SHA256,
                        help="the registered bundle's manifest SHA-256")  # fmt: skip
    parser.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--reference", type=Path, default=ref.REF_DIR)
    parser.add_argument("--manifest", type=Path, default=ref.REF_REPORT,
                        help="the committed copy of the references' manifest")  # fmt: skip
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    return run(args, require_fresh)


if __name__ == "__main__":
    sys.exit(main())
