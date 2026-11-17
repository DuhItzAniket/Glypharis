"""Reference outputs of PyTorch v1 on the CPU, and the v1 ONNX bundle checked against them
(GW-EFF-006, Phase 16 plan item 1).

    python scripts/phase16_reference.py write [--threads 10]
    python scripts/phase16_reference.py check --bundle runs/release/glypharis-v1 [--threads 10]

``write`` builds v1 from the freeze record (`load_frozen("cpu")`, every setting the record's)
after `assert_v1_code`, at ``--threads`` torch threads with the OpenMP and MKL variables set before
numpy and torch load, and writes once, under data/processed/phase16/reference/ (git-ignored):

- ``detector_maps.npz``: the detector's probability map (float32, the padded network input's size)
  on each of the 110 profile pages, keyed by position (``000``...), page ids in the manifest;
- ``recognizer_logits.npz`` and ``recognizer_lines.jsonl``: the ``(T, classes)`` logits (float32)
  of each of the 5,000 lines seeded 20261028 from processed/hiertext/lines_validation, read in
  batches of 256 in index order, with each line's pack index, key, frame count and greedy text;
- ``page_predictions.jsonl``: the pipeline's lines (polygon, text, confidence, detector score) on
  each of the 400 Phase 5 validation pages;
- ``manifest.json``: the SHA-256 of each file, the items, the build's provenance and settings, the
  v1 code check, the threads, the commit and the time each part took.

The manifest is copied to reports/efficiency/phase16/reference_manifest.json, which is committed.

``check`` loads the bundle on ONNX Runtime (`load_bundle_onnx`: sessions from `ort_session` at
``--threads`` intra-op threads, options asserted), refuses a bundle whose manifest differs from
the freeze record (`export_check.bundle_problems`) and references whose files differ from their
hashes or whose manifest differs from the committed copy, then runs the same items and applies
Phase 17's four tolerances (docs/PHASE_STATUS.md): map max abs difference at most 1e-4; logits at
most 1e-3; text identical except near ties (reference top-2 gap below 1e-3 at the first differing
frame) on at most 0.1% of lines; page CER (bag of lines, strict) within 0.05 points. It writes
every number, the bundle's file hashes and the session options to
reports/efficiency/phase16/reference_check.json.

The items are `scripts/export_check.py`'s (`profile_pages`, `line_indices`, `phase5_page_ids`),
read the way its checks read them; every page and line must be from the validation split.
``--limit N`` (smoke tests only, with ``--out`` and ``--report`` elsewhere) keeps the first N of
each list and marks the references incomplete; ``check`` then runs on the same items.
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
import json
import platform
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from e2e_eval import read_predictions, score
from export_check import (
    CER_TOL,
    LINES,
    LOGIT_TOL,
    NEAR_TIE,
    NEAR_TIE_SHARE,
    PROB_TOL,
    PROFILE_PAGES,
    SEED,
    bundle_problems,
    detector_input,
    line_indices,
    phase5_page_ids,
    profile_pages,
    text_difference,
)
from run_baselines import git_state

from glypharis.config.loader import load_config
from glypharis.data.det_dataset import read_gt_pages
from glypharis.data.det_geometry import decode_page
from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackReader
from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.onnx_build import check_sessions, load_bundle_onnx
from glypharis.efficiency.threads import check_threads
from glypharis.efficiency.v1 import V1CodeError, assert_v1_code
from glypharis.errors import ModelLoadError
from glypharis.eval.page_metrics import PageResult, PredLine
from glypharis.eval.sets import EVAL_SETS
from glypharis.export.bundle import read_manifest
from glypharis.pipeline.frozen import FREEZE_RECORD, load_frozen
from glypharis.pipeline.ocr import GlypharisOCR

EXPERIMENT = "GW-EFF-006"
REF_DIR = Path("data/processed/phase16/reference")
REF_REPORT = Path("reports/efficiency/phase16/reference_manifest.json")
CHECK_OUT = Path("reports/efficiency/phase16/reference_check.json")
PACK = "processed/hiertext/lines_validation"  # the export check's lines (Phase 16 plan, item 3)
PAGES = str(EVAL_SETS["hiertext_val"]["pages"])  # the Phase 5 HierText validation pages
SPLIT = "validation"
LINE_BATCH = 256  # lines per `TextRecognizer.logits` call, as export_check.rec_check
MAPS = "detector_maps.npz"
LOGITS = "recognizer_logits.npz"
LINE_TEXTS = "recognizer_lines.jsonl"
PREDICTIONS = "page_predictions.jsonl"
FILES = (MAPS, LOGITS, LINE_TEXTS, PREDICTIONS)
TRACK = "strict"


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    write_json_atomic(path, obj)


class NpzStream:
    """An ``.npz`` written one array at a time (`np.load` reads it), so 110 page maps or 5,000
    lines of logits never sit in memory together."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._zip = zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=1)

    def add(self, name: str, array: np.ndarray) -> None:
        with self._zip.open(f"{name}.npy", "w", force_zip64=True) as fh:
            np.lib.format.write_array(fh, np.ascontiguousarray(array), allow_pickle=False)

    def close(self) -> None:
        self._zip.close()

    def __enter__(self) -> NpzStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _key(i: int) -> str:
    return f"{i:04d}"


def _limited(items: list[Any], limit: int | None) -> list[Any]:
    return items if limit is None else items[:limit]


def _validation_only(what: str, splits: dict[str, str]) -> None:
    other = {k: s for k, s in splits.items() if s != SPLIT}
    if other:
        first = next(iter(other.items()))
        raise SystemExit(f"{len(other)} {what} are not from the {SPLIT} split (e.g. {first}); "
                         "Phase 16 reads validation data only")  # fmt: skip


# ------------------------------------------------------------------------------------- items


def map_pages(data: Path, limit: int | None) -> list[Any]:
    pages = _limited(profile_pages(data), limit)
    _validation_only("profile pages", {p.doc_id: p.split for p in pages})
    return pages


def line_items(data: Path, pack: str, limit: int | None) -> tuple[LinePackReader, list[int]]:
    reader = LinePackReader(data / pack)
    idx = _limited([int(i) for i in line_indices(len(reader))], limit)
    _validation_only("lines", {reader.records[i].key: reader.records[i].split for i in idx})
    return reader, idx


def e2e_pages(data: Path, pages_file: str, limit: int | None) -> tuple[list[str], dict[str, Any]]:
    ids = _limited(phase5_page_ids(data, data / pages_file), limit)
    gt = read_gt_pages(data / pages_file, ids)
    _validation_only("Phase 5 pages", {d: gt[d].split for d in ids})
    return ids, gt


def _page_lines(ocr: GlypharisOCR, data: Path, page: Any) -> list[dict[str, Any]]:
    rgb = decode_page(data / page.image, (page.width, page.height))
    res = ocr.predict(rgb)
    return [{"polygon": ln.polygon, "text": ln.text, "confidence": ln.confidence,
             "det_score": ln.det_score} for ln in res.lines]  # fmt: skip


# ------------------------------------------------------------------------------------- write


def write_references(
    ocr: GlypharisOCR, data: Path, pack: str, pages_file: str, out: Path, limit: int | None = None
) -> dict[str, Any]:
    """Write the four reference files under ``out`` from ``ocr`` (PyTorch v1); returns the
    manifest's item and file sections."""
    out.mkdir(parents=True, exist_ok=True)
    seconds: dict[str, float] = {}

    t0 = time.perf_counter()
    pages = map_pages(data, limit)
    long_side = ocr.detector.spec.infer.long_side
    shapes = []
    with NpzStream(out / MAPS) as z:
        for i, page in enumerate(pages):
            prob = np.asarray(ocr.detector.backend(detector_input(page, data, long_side)),
                              dtype=np.float32)  # fmt: skip
            z.add(_key(i), prob)
            shapes.append(list(prob.shape))
    seconds["detector_maps"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    reader, idx = line_items(data, pack, limit)
    charset = ocr.recognizer.charset
    classes = None
    with NpzStream(out / LOGITS) as z, (out / LINE_TEXTS).open("w", encoding="utf-8",
                                                               newline="\n") as fh:  # fmt: skip
        for start in range(0, len(idx), LINE_BATCH):
            batch = idx[start : start + LINE_BATCH]
            logits = ocr.recognizer.logits([reader.image(i) for i in batch])
            for k, (i, lg) in enumerate(zip(batch, logits, strict=True), start):
                z.add(_key(k), lg.numpy().astype(np.float32))
                classes = int(lg.shape[1])
                row = {"item": k, "index": i, "key": reader.records[i].key,
                       "frames": int(lg.shape[0]),
                       "text": charset.ctc_greedy_decode(lg.argmax(-1).tolist())}  # fmt: skip
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    seconds["recognizer_logits"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    ids, gt = e2e_pages(data, pages_file, limit)
    with (out / PREDICTIONS).open("w", encoding="utf-8", newline="\n") as fh:
        for d in ids:
            row = {"key": d, "lines": _page_lines(ocr, data, gt[d])}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    seconds["page_predictions"] = round(time.perf_counter() - t0, 1)

    return {
        "limit": limit,
        "complete": limit is None,
        "detector": {
            "file": MAPS,
            "pages": len(pages),
            "registered_pages": PROFILE_PAGES,
            "eval_set": "hiertext_val",
            "doc_ids": [p.doc_id for p in pages],
            "shapes": shapes,
            "long_side": long_side,
        },
        "recognizer": {
            "file": LOGITS,
            "texts": LINE_TEXTS,
            "lines": len(idx),
            "registered_lines": LINES,
            "pack": pack,
            "pack_lines": len(reader),
            "pack_bin_sha256": reader.meta.get("bin_sha256"),
            "seed": SEED,
            "batch": LINE_BATCH,
            "classes": classes,
            "indices": idx,
        },
        "end_to_end": {
            "file": PREDICTIONS,
            "pages": len(ids),
            "registered_pages": 400,
            "pages_file": pages_file,
            "pages_sha256": sha256_file(data / pages_file),
            "doc_ids": ids,
        },
        "seconds": seconds,
        "files": {name: sha256_file(out / name) for name in FILES},
    }


# ------------------------------------------------------------------------------------- check


def load_reference(ref_dir: Path, committed: Path | None) -> dict[str, Any]:
    """The reference manifest, after checking it equals the committed copy (``committed``; None
    skips this, for tests) and that every file has its recorded SHA-256."""
    try:
        manifest: dict[str, Any] = json.loads(
            (ref_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{ref_dir.as_posix()}: no readable reference manifest ({exc})") from exc
    if committed is not None:
        if not committed.is_file():
            raise SystemExit(f"{committed.as_posix()} is missing: the references' manifest is "
                             "committed before they are checked")  # fmt: skip
        if json.loads(committed.read_text(encoding="utf-8")) != manifest:
            raise SystemExit(f"{(ref_dir / 'manifest.json').as_posix()} differs from "
                             f"{committed.as_posix()}")  # fmt: skip
    if manifest.get("experiment") != EXPERIMENT or set(manifest.get("files", {})) != set(FILES):
        raise SystemExit(f"{ref_dir.as_posix()} does not hold {EXPERIMENT}'s reference files")
    for name, want in manifest["files"].items():
        got = sha256_file(ref_dir / name) if (ref_dir / name).is_file() else None
        if got != want:
            raise SystemExit(f"{(ref_dir / name).as_posix()}: SHA-256 {got} differs from the "
                             f"manifest's {want}")  # fmt: skip
    return manifest


def _same_items(what: str, got: list[Any], want: list[Any]) -> None:
    if got != want:
        raise SystemExit(f"the {what} differ from the references' ({len(got)} against "
                         f"{len(want)}); the data changed since they were written")  # fmt: skip


def check_detector(
    ocr: GlypharisOCR, data: Path, ref_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    pages = map_pages(data, manifest["limit"])
    _same_items("profile pages", [p.doc_id for p in pages], manifest["detector"]["doc_ids"])
    long_side = ocr.detector.spec.infer.long_side
    if long_side != manifest["detector"]["long_side"]:
        raise SystemExit(f"the bundle's detector long side {long_side} differs from the "
                         f"references' {manifest['detector']['long_side']}")  # fmt: skip
    per_page: list[float | None] = []
    wrong_shape = []
    with np.load(ref_dir / MAPS, allow_pickle=False) as ref:
        for i, page in enumerate(pages):
            a = ref[_key(i)]
            b = np.asarray(ocr.detector.backend(detector_input(page, data, long_side)))
            if a.shape != b.shape:
                wrong_shape.append({"page": page.doc_id, "reference": list(a.shape),
                                    "onnx": list(b.shape)})  # fmt: skip
                per_page.append(None)
                continue
            per_page.append(float(np.abs(a.astype(np.float64) - b).max()))
    worst = max((d for d in per_page if d is not None), default=0.0)
    return {
        "pages": len(pages),
        "prob_max_abs_diff": worst,
        "per_page_max_abs_diff": per_page,
        "shape_differences": wrong_shape,
        "tolerance": PROB_TOL,
        "pass": not wrong_shape and worst <= PROB_TOL,
    }


def check_recognizer(
    ocr: GlypharisOCR, data: Path, ref_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    section = manifest["recognizer"]
    reader, idx = line_items(data, section["pack"], manifest["limit"])
    _same_items("lines", idx, section["indices"])
    rows = [json.loads(ln) for ln in (ref_dir / LINE_TEXTS).read_text(encoding="utf-8").splitlines()
            if ln.strip()]  # fmt: skip
    _same_items("line keys", [reader.records[i].key for i in idx], [r["key"] for r in rows])
    charset = ocr.recognizer.charset
    worst, same, near, other, text_mismatch = 0.0, 0, 0, [], 0
    above = 0
    with np.load(ref_dir / LOGITS, allow_pickle=False) as ref:
        for start in range(0, len(idx), LINE_BATCH):
            batch = idx[start : start + LINE_BATCH]
            got = ocr.recognizer.logits([reader.image(i) for i in batch])
            for k, lb in enumerate(got, start):
                la = torch.from_numpy(ref[_key(k)])
                if charset.ctc_greedy_decode(la.argmax(-1).tolist()) != rows[k]["text"]:
                    text_mismatch += 1  # the bundle's charset does not decode the references
                if la.shape == lb.shape:
                    d = float((la - lb.float()).abs().max()) if la.numel() else 0.0
                    worst = max(worst, d)
                    above += d > LOGIT_TOL
                else:
                    worst, above = float("inf"), above + 1
                diff = text_difference(la, lb.float(), charset)
                if diff is None:
                    same += 1
                elif diff["near_tie"]:
                    near += 1
                else:
                    other.append({"item": k, "key": rows[k]["key"], **diff})
    n = len(idx)
    logit_ok = worst <= LOGIT_TOL
    text_ok = not other and near <= NEAR_TIE_SHARE * n and text_mismatch == 0
    return {
        "lines": n,
        "logit_max_abs_diff": worst,
        "lines_above_logit_tolerance": above,
        "logit_tolerance": LOGIT_TOL,
        "logit_pass": logit_ok,
        "identical_text": same,
        "near_tie_differences": near,
        "near_tie_share": near / n if n else 0.0,
        "near_tie_gap": NEAR_TIE,
        "near_tie_share_allowed": NEAR_TIE_SHARE,
        "other_differences": other,
        "reference_text_mismatches": text_mismatch,
        "text_pass": text_ok,
        "pass": logit_ok and text_ok,
    }


def _bag_cer(rs: list[PageResult]) -> float:
    ref = sum(r.bag_ref_chars for r in rs)
    return sum(r.bag_errors for r in rs) / ref if ref else 0.0


def check_end_to_end(
    ocr: GlypharisOCR, data: Path, ref_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    section = manifest["end_to_end"]
    if sha256_file(data / section["pages_file"]) != section["pages_sha256"]:
        raise SystemExit(f"{section['pages_file']} changed since the references were written")
    ids, gt = e2e_pages(data, section["pages_file"], manifest["limit"])
    _same_items("Phase 5 pages", ids, section["doc_ids"])
    ref_preds = read_predictions(ref_dir / PREDICTIONS)
    _same_items("predicted pages", list(ref_preds), ids)
    onnx_preds: dict[str, list[PredLine]] = {}
    same_text = 0
    for d in ids:
        got = _page_lines(ocr, data, gt[d])
        onnx_preds[d] = [PredLine(ln["polygon"], ln["text"], ln["confidence"]) for ln in got]
        same_text += [p.text for p in onnx_preds[d]] == [p.text for p in ref_preds[d]]
    cer_ref = _bag_cer(score(ref_preds, gt, ids, TRACK))
    cer_onnx = _bag_cer(score(onnx_preds, gt, ids, TRACK))
    diff = cer_onnx - cer_ref
    return {
        "pages": len(ids),
        "track": TRACK,
        "page_cer_torch": cer_ref,
        "page_cer_onnx": cer_onnx,
        "difference": diff,
        "pages_with_identical_text": same_text,
        "tolerance": CER_TOL,
        "pass": abs(diff) <= CER_TOL,
    }


def check_references(
    ocr: GlypharisOCR, data: Path, ref_dir: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Phase 17's four checks of ``ocr`` (the ONNX bundle) against the references."""
    seconds: dict[str, float] = {}
    out: dict[str, Any] = {}
    for name, fn in (("detector", check_detector), ("recognizer", check_recognizer),
                     ("end_to_end", check_end_to_end)):  # fmt: skip
        t0 = time.perf_counter()
        out[name] = fn(ocr, data, ref_dir, manifest)
        seconds[name] = round(time.perf_counter() - t0, 1)
    d, r, e = out["detector"], out["recognizer"], out["end_to_end"]
    out["tolerances"] = {
        "map_max_abs": {"limit": PROB_TOL, "measured": d["prob_max_abs_diff"], "pass": d["pass"]},
        "logits_max_abs": {"limit": LOGIT_TOL, "measured": r["logit_max_abs_diff"],
                           "pass": r["logit_pass"]},
        "text": {"limit": f"other differences 0, near ties (gap < {NEAR_TIE:g}) at most "
                          f"{NEAR_TIE_SHARE:.1%} of lines",
                 "measured": {"near_ties": r["near_tie_differences"],
                              "other": len(r["other_differences"])},
                 "pass": r["text_pass"]},
        "page_cer": {"limit": CER_TOL, "measured": e["difference"], "pass": e["pass"]},
    }  # fmt: skip
    out["pass"] = all(t["pass"] for t in out["tolerances"].values())
    out["seconds"] = seconds
    return out


# -------------------------------------------------------------------------------------- main


def record_sources(root: Path) -> dict[str, Any]:
    """The lines pack and pages file named by the freeze record's training configs, which must be
    the registered ones (`PACK`, `PAGES`)."""
    record = yaml.safe_load((root / FREEZE_RECORD).read_text(encoding="utf-8"))
    rec_cfg = load_config(root / record["recognizer"]["config"])
    det_cfg = load_config(root / record["detector"]["config"])
    if det_cfg.det is None:
        raise SystemExit(f"{record['detector']['config']} has no detector section")
    pack, pages = rec_cfg.data.val[0], det_cfg.det.data.val_pages[0]
    if (pack, pages) != (PACK, PAGES):
        raise SystemExit(f"the record's configs name {pack} and {pages}; the registered items "
                         f"are {PACK} and {PAGES}")  # fmt: skip
    return {"pack": pack, "pages_file": pages, "recognizer_config": record["recognizer"]["config"],
            "detector_config": record["detector"]["config"]}  # fmt: skip


def _versions() -> dict[str, Any]:
    try:
        import onnxruntime as ort

        ort_version: str | None = ort.__version__
    except ImportError:
        ort_version = None
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "onnxruntime": ort_version}  # fmt: skip


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


def _same_path(a: Path, b: Path, root: Path) -> bool:
    """``a`` is ``b`` (a registered path, relative to the working directory or to ``root``)."""
    return a.resolve() in {b.resolve(), (root / b).resolve()}


def main_write(args: argparse.Namespace, require_fresh: bool) -> int:
    out, report = args.out, args.report
    registered = _same_path(out, REF_DIR, args.root) or _same_path(report, REF_REPORT, args.root)
    if args.limit is not None and registered:
        raise SystemExit("--limit writes smoke-test references: give --out and --report "
                         "outside the registered paths")  # fmt: skip
    for path in (out / "manifest.json", report):
        if path.exists():
            raise SystemExit(f"{path.as_posix()} exists: the references are written once")
    threads = _threads(args.threads, require_fresh)
    v1_code = _v1_code(args.root)
    sources = record_sources(args.root)
    torch.set_grad_enabled(False)
    frozen = load_frozen("cpu", root=args.root)
    before = check_threads(args.threads)
    started = _utc()
    core = write_references(frozen.ocr, args.data_root, sources["pack"], sources["pages_file"],
                            out, args.limit)  # fmt: skip
    ended = _utc()
    after = check_threads(args.threads)
    manifest = {
        "experiment": EXPERIMENT,
        "what": "PyTorch v1 reference outputs on the CPU (Phase 16 plan, item 1)",
        "version": frozen.provenance.get("name"),
        "backend": f"torch {torch.__version__} (CPU)",
        "started_utc": started,
        "ended_utc": ended,
        "threads": threads,
        "threads_checked": {"before": before, "after": after},
        "v1_code": v1_code,
        "sources": sources,
        "settings": frozen.settings(),
        "provenance": frozen.provenance,
        "versions": _versions(),
        **git_state(),
        **core,
    }
    _write_json(out / "manifest.json", manifest)
    _write_json(report, manifest)
    s = core["seconds"]
    print(f"{EXPERIMENT} references: {core['detector']['pages']} maps ({s['detector_maps']} s), "
          f"{core['recognizer']['lines']} lines ({s['recognizer_logits']} s), "
          f"{core['end_to_end']['pages']} pages ({s['page_predictions']} s); manifest "
          f"{report.as_posix()}")  # fmt: skip
    return 0


def main_check(args: argparse.Namespace, require_fresh: bool) -> int:
    if args.out.exists():
        raise SystemExit(f"{args.out.as_posix()} exists: move it aside first")
    threads = _threads(args.threads, require_fresh)
    v1_code = _v1_code(args.root)
    record_path = args.root / FREEZE_RECORD
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    try:
        bundle_manifest = read_manifest(args.bundle)
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc
    problems = bundle_problems(bundle_manifest, record)
    if problems:
        raise SystemExit("the bundle is not v1's: " + "; ".join(problems))
    reference = load_reference(args.reference, args.manifest)
    if reference["provenance"].get("record_sha256") != sha256_file(record_path):
        raise SystemExit("the references were written from another freeze record")
    torch.set_grad_enabled(False)
    try:
        ocr = load_bundle_onnx(args.bundle, args.threads)
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc
    sessions = check_sessions(ocr, args.threads)
    before = check_threads(args.threads)
    started = _utc()
    res = check_references(ocr, args.data_root, args.reference, reference)
    ended = _utc()
    after = check_threads(args.threads)
    verdict = {
        "experiment": EXPERIMENT,
        "what": "the v1 fp32 ONNX bundle against the PyTorch v1 references (Phase 17's tolerances)",
        "bundle": args.bundle.as_posix(),
        "bundle_version": bundle_manifest.get("version"),
        "bundle_manifest_sha256": sha256_file(args.bundle / "manifest.json"),
        "bundle_files": bundle_manifest["files"],
        "bundle_equals_record": True,
        "reference": args.reference.as_posix(),
        "reference_manifest_sha256": sha256_file(args.reference / "manifest.json"),
        "reference_files": reference["files"],
        "complete": reference["complete"],
        "limit": reference["limit"],
        "backend": "onnxruntime (CPU)",
        "started_utc": started,
        "ended_utc": ended,
        "threads": threads,
        "threads_checked": {"before": before, "after": after},
        "session_options": sessions,
        "v1_code": v1_code,
        "versions": _versions(),
        **git_state(),
        **res,
    }
    _write_json(args.out, verdict)
    t = res["tolerances"]
    print(f"{EXPERIMENT} check of {args.bundle.as_posix()}: map {t['map_max_abs']['measured']:.2e}"
          f", logits {t['logits_max_abs']['measured']:.2e}, text {t['text']['measured']}, page "
          f"CER {100 * t['page_cer']['measured']:+.3f} points; pass: {res['pass']}; written to "
          f"{args.out.as_posix()}")  # fmt: skip
    return 0 if res["pass"] else 1


def main(argv: list[str] | None = None, *, require_fresh: bool = True) -> int:
    """``require_fresh=False`` lets a process that already holds torch (the tests) run it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("write", "check"):
        p = sub.add_parser(name)
        p.add_argument(
            THREADS_FLAG,
            type=int,
            default=DEFAULT_THREADS,
            help="torch threads, the OpenMP/MKL variables and ONNX Runtime intra-op",
        )
        p.add_argument("--root", type=Path, default=Path("."), help="the repository root")
        p.add_argument("--data-root", type=Path, default=Path("data"))
        if name == "write":
            p.add_argument("--out", type=Path, default=REF_DIR)
            p.add_argument("--report", type=Path, default=REF_REPORT)
            p.add_argument("--limit", type=int, default=None, help="smoke tests: first N items")
        else:
            p.add_argument("--bundle", type=Path, required=True)
            p.add_argument("--reference", type=Path, default=REF_DIR)
            p.add_argument("--manifest", type=Path, default=REF_REPORT,
                           help="the committed copy of the references' manifest")  # fmt: skip
            p.add_argument("--out", type=Path, default=CHECK_OUT)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.command == "write":
        return main_write(args, require_fresh)
    return main_check(args, require_fresh)


if __name__ == "__main__":
    sys.exit(main())
