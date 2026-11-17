"""Run the frozen GLYPHARIS v1 model on an evaluation set and write its predictions (Phase 14).

    python scripts/ext_eval.py rec --set gdc_test --experiment GW-REC-039 \
        --reason "Phase 14 GLYPHARIS recognition (GW-REC-039)"
    python scripts/ext_eval.py e2e --set hiertext_val --experiment GW-E2E-006 \
        --reason "Phase 14 validation reference (GW-E2E-006)"
    python scripts/ext_eval.py oracle --set gdc_test --experiment GW-E2E-008 \
        --reason "Phase 14 oracle run (GW-E2E-008)"

The model is built by `glypharis.pipeline.frozen.load_frozen` from the freeze record, never from
command-line settings. Modes (docs/PHASE_STATUS.md, Phase 14 items 1, 2 and 6):

- ``rec``: the recognizer on every crop of a set's line pack (no subsampling, validation
  included). One record per crop: ``key``, ``ref`` (the pack label), ``pred``, ``conf_t1`` and
  ``conf_t`` (line confidence at T = 1 and at the record's temperature), ``time_ms`` (the chunk's
  time divided by its crops; diagnostic only), and the per-line attributes Phase 15 reads:
  ``crop_height_px`` (the stored crop), ``line_height_px`` and ``rotation_deg`` (the line's box on
  its page; None for a set without pages, IIIT5K), ``blur`` (variance of the Laplacian of the grey
  crop) and ``contrast`` (standard deviation of the grey crop).
- ``e2e``: the pipeline on every page, written in the baselines' end-to-end schema: ``key``,
  ``lines`` (``polygon``, ``quad``, ``text``, ``confidence`` at the record's temperature,
  ``conf_t1``, ``det_score``) in reading order, and ``time_ms``.
- ``oracle``: the ground-truth box lines of each page (`is_box_line`) through
  `GlypharisOCR.read_quads`, as e2e plus each line's ``line_id``; ``det_score`` is None.

Which experiment runs which set is fixed by `PLAN`. Every run:

1. starts only on branch `PHASE14_BRANCH`, and runs `pinned_code_check` (the tree must equal the
   ``phase14-eval`` tag's apart from the run records). A test set is refused when the check fails;
   a validation run is refused too unless ``--allow-unpinned`` is given, which only tests use, and
   the meta then records the failure. Every pass runs on `TEST_DEVICE` (validation included), and
   the Phase 5 files must already be hashed (`require_phase5_hashes`);
2. takes the split from the records themselves (the pack index or the page file) and refuses if
   any record names another split than the set's;
3. hashes its input files and compares them with the hashes and counts recorded in
   data/manifests (eval_crops.json, benchmarks.json, hiertext.json, gdc_scan_v1.json; a pack's
   .bin and its .index.jsonl labels), refusing on any difference
   (`glypharis.eval.inputs.check_inputs`); in rec mode each crop's line is found on its page
   (`line_boxes`); in oracle mode the page line ids must be unique; in e2e and oracle mode every
   page image's header size must equal its page record's (`image_sizes_match`);
4. loads the frozen model, then logs a test access with `open_test_access` (phase 14, the log
   found from the repository root) before any image is decoded. A refused load logs nothing.

Outputs: data/processed/phase14/<ID>/<set>.jsonl and <set>.meta.json (provenance from `Frozen`,
the pinned-code check, the settings, the inputs' hashes and the counts; never a metric). Records
are appended as they are made, so an interrupted run resumes where it stopped; a resume must have
the same model, run environment (`RUN_ENV`), settings, recognition batch, inputs and tag, each
segment's provenance and batch are kept in its ``runs`` entry, and its log row says "(resumed: N
of M ...)". Once a file
is complete its SHA-256 and line count are added to data/manifests/processed/phase14_outputs.json.
The runner prints counts only and computes no metric; scoring is GW-E2E-009's one pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from run_baselines import done_keys

from glypharis.data.crops import line_quad
from glypharis.data.det_geometry import decode_page
from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackReader
from glypharis.data.pages import Page, read_pages
from glypharis.eval.access import is_test_split, open_test_access
from glypharis.eval.inputs import (
    RUN_ENV,
    check_inputs,
    duplicate_line_ids,
    image_sizes_match,
    line_boxes,
)
from glypharis.eval.page_metrics import is_box_line
from glypharis.eval.sets import (
    EVAL_SETS,
    PHASE14_BRANCH,
    access_log,
    repo_root,
    require_branch,
    require_phase5_hashes,
)
from glypharis.pipeline.frozen import Frozen, FrozenModelError, load_frozen, pinned_code_check
from glypharis.utils.gitinfo import git_info

MODES = ("rec", "e2e", "oracle")
# Registered with the Phase 14 plan (docs/PHASE_STATUS.md item 2): experiment -> its sets.
# fmt: off
PLAN: dict[str, dict[str, tuple[str, ...]]] = {
    "rec": {"GW-REC-039": ("hiertext_test", "gdc_test", "gdc_scan_test", "funsd_test",
                           "iiit5k_test"),
            "GW-REC-038": ("hiertext_val",)},
    "e2e": {"GW-E2E-007": ("hiertext_test", "gdc_test", "gdc_scan_test", "funsd_test",
                           "omnidocbench_test"),
            "GW-E2E-006": ("hiertext_val",)},
    "oracle": {"GW-E2E-008": ("gdc_test", "gdc_scan_test")},
}
# fmt: on
OUTPUT_ROOT = "phase14"  # under <data>/processed
OUTPUTS_MANIFEST = Path("manifests/processed/phase14_outputs.json")  # under <data>
# Every input must have a recorded hash, and every pack a recorded index hash (fail closed,
# `glypharis.eval.inputs.check_inputs`); the tests' tiny sets have none.
REQUIRE_RECORDED = True
ACCESS_LOG = Path("research/test_access_log.md")
TEST_DEVICE = "cuda"  # the plan's hardware for every pass, validation included (RTX 4050)
SYSTEM = "glypharis"  # the access log's name for this system, as the scorer logs it
T1 = 1.0
REC_CHUNK = 256  # crops per call; the recognizer reads a line the same in any batch


# ------------------------------------------------------------------------------------ paths


def log_path() -> Path:
    """The test-access log: `ACCESS_LOG` when absolute (tests point it at a temporary file),
    otherwise the repository's own log, found from the repository root."""
    return ACCESS_LOG if ACCESS_LOG.is_absolute() else access_log()


def output_path(data: Path, experiment: str, eval_set: str) -> Path:
    return data / "processed" / OUTPUT_ROOT / experiment / f"{eval_set}.jsonl"


def data_rel(path: Path, data: Path) -> str:
    """A file under the data root as the repository names it (``data/...``), never absolute."""
    return "data/" + path.relative_to(data).as_posix()


def system_name(frozen: Frozen) -> str:
    return str(frozen.provenance.get("name") or "glypharis_v1")


# ------------------------------------------------------------------------------- attributes


def line_geometry(polygon: Sequence[Sequence[float]]) -> tuple[float | None, float | None]:
    """A line's height in page pixels and the angle of its top edge in degrees (0 = level,
    positive = clockwise in image coordinates), as `cut_line` sees the box."""
    pts = [(float(x), float(y)) for x, y in polygon]
    n = len(pts) // 2
    if len(pts) > 4 and len(pts) % 2 == 0:
        top, bottom = pts[:n], pts[n:][::-1]
        height = sum(math.dist(t, b) for t, b in zip(top, bottom, strict=True)) / n
        (x0, y0), (x1, y1) = top[0], top[-1]
    elif len(pts) == 4:
        tl, tr, br, bl = line_quad([list(p) for p in pts])
        height = (math.dist(tl, bl) + math.dist(tr, br)) / 2
        (x0, y0), (x1, y1) = tl[:2], tr[:2]
    else:
        return None, None
    return round(height, 2), round(math.degrees(math.atan2(y1 - y0, x1 - x0)), 2)


def crop_attributes(img: Image.Image) -> dict[str, Any]:
    """Height of the crop as stored, blur (variance of the Laplacian) and contrast (grey std)."""
    grey = np.asarray(img.convert("L"), dtype=np.float64)
    return {
        "crop_height_px": int(img.height),
        "blur": round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 4),
        "contrast": round(float(grey.std()), 4),
    }


def _points(poly: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in poly]


# ------------------------------------------------------------------------------------ checks


def check_request(args: argparse.Namespace) -> None:
    """Refusals that need no data: the registered plan, the device and the unpinned escape."""
    sets_for = PLAN[args.mode].get(args.experiment)
    if sets_for is None:
        raise SystemExit(f"{args.experiment} is not a {args.mode} experiment; registered: "
                         f"{', '.join(sorted(PLAN[args.mode]))}")  # fmt: skip
    if args.set not in sets_for:
        raise SystemExit(f"{args.experiment} runs on {', '.join(sets_for)}, not {args.set}")
    spec = EVAL_SETS[args.set]
    if args.mode == "rec" and not spec["pack"]:
        raise SystemExit(f"{args.set} has no line pack")
    if args.mode != "rec" and not spec["pages"]:
        raise SystemExit(f"{args.set} has no pages")
    if not args.reason.strip():
        raise SystemExit("a reason is required (it goes into the test-access log)")
    if is_test_split(spec["split"]) and args.allow_unpinned:
        raise SystemExit("--allow-unpinned is for validation runs in tests; a test set "
                         "always needs the pinned code")  # fmt: skip
    # H5′ compares validation and test under the same code and precision, so validation runs
    # use the registered device too; only the tests' escape runs elsewhere. The device is
    # compared exactly as torch names it: "cuda:0" and "cuda" would record two environments
    # (`RUN_ENV`) for one GPU and void the runs.
    if str(torch.device(args.device)) != TEST_DEVICE and not args.allow_unpinned:
        raise SystemExit(f"Phase 14 passes run on {TEST_DEVICE} (the registered hardware, "
                         f"named exactly so), not {args.device}")  # fmt: skip


def pinned_state(args: argparse.Namespace, test: bool) -> dict[str, Any]:
    """The pinned-code check's result; a failure stops the run unless a validation run passed
    ``--allow-unpinned`` (tests), and is then recorded instead."""
    try:
        return pinned_code_check(root=repo_root())
    except RuntimeError as exc:
        if test or not args.allow_unpinned:
            raise SystemExit(f"pinned code check failed: {exc}") from exc
        return {"trees": "not pinned", "error": str(exc), "allow_unpinned": True}


MODEL_HASHES = ("record_sha256", "recognizer_checkpoint_sha256", "detector_checkpoint_sha256",
                "charset_sha256")  # fmt: skip


def identity(
    frozen: Frozen, pinned: dict[str, Any], inputs: dict[str, Any], batch: int | None
) -> dict[str, Any]:
    """What every segment of one evaluation must share: the model files, the run environment the
    scorer checks (`RUN_ENV`), settings, the recognition batch (None for page modes), inputs and
    the tag. HEAD may move between segments (commits of the access log, outside the trees)."""
    p = frozen.provenance
    doc = {"model": {k: p.get(k) for k in (*MODEL_HASHES, *RUN_ENV)},
           "settings": frozen.settings(), "batch": batch,
           "tag_commit": pinned.get("tag_commit"), "inputs": inputs}  # fmt: skip
    return dict(json.loads(json.dumps(doc)))


# -------------------------------------------------------------------------------- manifest


def count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def record_output(data: Path, out: Path, experiment: str) -> dict[str, Any]:
    """Add a complete output's hash and line count to `OUTPUTS_MANIFEST` (hashes only).

    An entry already there must match exactly: a file that changed after it was recorded stops
    here rather than being recorded twice.
    """
    manifest = data / OUTPUTS_MANIFEST
    if manifest.is_file():
        doc = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        doc = {"note": "Phase 14 per-sample outputs (git-ignored under data/processed/phase14/): "
                       "SHA-256 and line counts only, checked before Phases 15 and 22 read them",
               "outputs": []}  # fmt: skip
    meta = out.with_suffix(".meta.json")
    entry = {"file": data_rel(out, data), "sha256": sha256_file(out), "lines": count_lines(out),
             "experiment": experiment, "meta": data_rel(meta, data),
             "meta_sha256": sha256_file(meta)}  # fmt: skip
    same = [e for e in doc["outputs"] if e.get("file") == entry["file"]]
    if same:
        if same[0] != entry:
            where = f"data/{OUTPUTS_MANIFEST.as_posix()}"
            raise SystemExit(f"{entry['file']} is recorded in {where} with another hash or "
                             "count; it changed after it was recorded")  # fmt: skip
        return entry
    doc["outputs"] = sorted([*doc["outputs"], entry], key=lambda e: str(e.get("file")))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _write_json(manifest, doc)
    return entry


def close_meta(meta_path: Path, samples: int) -> None:
    """A run killed after its last record but before its metadata was updated: say so once."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    runs = meta.get("runs") or []
    if meta.get("samples") == samples and not (runs and runs[-1].get("status") == "running"):
        return
    if runs and runs[-1].get("status") == "running":
        runs[-1]["status"] = "stopped after its last record; closed by a later start"
    meta["samples"] = samples
    _write_json(meta_path, meta)


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


# ------------------------------------------------------------------------------------- runs


def set_records(
    data: Path, eval_set: str, mode: str
) -> tuple[list[str], list[Page] | None, set[str]]:
    """The keys a complete run writes, in run order (every crop or every page; no subsample),
    the pages for a page mode, and the splits the records name."""
    spec = EVAL_SETS[eval_set]
    if mode == "rec":
        records = LinePackReader(data / spec["pack"]).records
        return [r.key for r in records], None, {r.split for r in records}
    pages = list(read_pages(data / spec["pages"]))
    return [p.doc_id for p in pages], pages, {p.split for p in pages}


def _rec_records(
    frozen: Frozen,
    data: Path,
    eval_set: str,
    done: set[str],
    chunk: int,
    boxes: dict[str, list[list[float]]],
) -> Iterator[dict[str, Any]]:
    spec = EVAL_SETS[eval_set]
    reader = LinePackReader(data / spec["pack"])
    todo = [i for i, r in enumerate(reader.records) if r.key not in done]
    for start in range(0, len(todo), chunk):
        idx = todo[start : start + chunk]
        images = [reader.image(i) for i in idx]
        t0 = time.perf_counter()
        reads = frozen.recognizer.read(images, (T1,))
        per_ms = (time.perf_counter() - t0) * 1000 / len(idx)
        for i, img, r in zip(idx, images, reads, strict=True):
            rec = reader.records[i]
            height, angle = line_geometry(boxes[rec.key]) if spec["pages"] else (None, None)
            yield {"key": rec.key, "ref": rec.label, "pred": r.text, "conf_t1": r.at[T1],
                   "conf_t": r.confidence, "time_ms": round(per_ms, 3),
                   **crop_attributes(img), "line_height_px": height,
                   "rotation_deg": angle}  # fmt: skip


def _page_records(
    frozen: Frozen, data: Path, pages: list[Page], done: set[str], mode: str
) -> Iterator[dict[str, Any]]:
    for page in pages:
        if page.doc_id in done:
            continue
        rgb = decode_page(data / page.image, (page.width, page.height))
        t0 = time.perf_counter()
        if mode == "oracle":
            care = [ln for ln in page.lines if is_box_line(ln)]
            reads = frozen.ocr.read_quads(rgb, [ln.polygon for ln in care], (T1,))
            lines = [{"line_id": ln.line_id, "polygon": _points(ln.polygon),
                      "quad": _points(line_quad(ln.polygon)), "text": r.text,
                      "confidence": r.confidence, "conf_t1": r.at[T1], "det_score": None}
                     for ln, r in zip(care, reads, strict=True)]  # fmt: skip
        else:
            res = frozen.ocr.predict(rgb, (T1,))
            lines = [{"polygon": _points(ln.polygon), "quad": _points(ln.quad), "text": ln.text,
                      "confidence": ln.confidence, "conf_t1": ln.conf_at[T1],
                      "det_score": float(ln.det_score)} for ln in res.lines]  # fmt: skip
        ms = (time.perf_counter() - t0) * 1000
        yield {"key": page.doc_id, "lines": lines, "time_ms": round(ms, 1)}


def run(args: argparse.Namespace, data: Path) -> int:
    spec = EVAL_SETS[args.set]
    split = spec["split"]
    test = is_test_split(split)
    try:
        require_branch(PHASE14_BRANCH)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    pinned = pinned_state(args, test)  # before anything is read or loaded
    require_phase5_hashes(data)

    task = "rec" if args.mode == "rec" else "e2e"
    keys, pages, splits = set_records(data, args.set, args.mode)
    if splits != {split}:
        raise SystemExit(f"{args.set}: records name splits {sorted(splits)}, the set is "
                         f"registered as {split}")  # fmt: skip
    out = output_path(data, args.experiment, args.set)
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{args.set}: a key appears twice in its records")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    stray = done - set(keys)
    if stray:
        raise SystemExit(f"{data_rel(out, data)} holds {len(stray)} keys outside {args.set} "
                         f"(e.g. {sorted(stray)[0]}); move it aside")  # fmt: skip
    meta_path = out.with_suffix(".meta.json")
    if meta_path.exists() and not out.exists():
        meta_path.unlink()  # metadata of predictions that were moved aside: start afresh
    if len(done) >= len(keys):
        # Nothing to do: no model load and no row in the test-access log.
        if meta_path.exists():
            close_meta(meta_path, len(done))
            record_output(data, out, args.experiment)
        print(f"{args.experiment} {args.mode} {args.set}: already complete ({len(done)})")
        return 0

    inputs = check_inputs(data, args.set, args.mode, None if pages is None else len(pages),
                          require=REQUIRE_RECORDED)  # fmt: skip
    boxes = line_boxes(data, args.set) if args.mode == "rec" else {}
    if args.mode == "oracle" and (dup := duplicate_line_ids(pages or [])):
        raise SystemExit(f"{spec['pages']}: {len(dup)} line ids appear more than once (e.g. "
                         f"{dup[0]}); the oracle lines are joined by id")  # fmt: skip
    if args.mode != "rec" and (bad := image_sizes_match(data, pages or [])):
        raise SystemExit(f"{len(bad)} page images of {args.set} differ in size from their page "
                         f"records, refusing before the access: {'; '.join(bad[:5])}")  # fmt: skip
    batch = args.batch if args.mode == "rec" else None
    try:
        frozen = load_frozen(args.device, root=repo_root())
    except FrozenModelError as exc:
        raise SystemExit(f"frozen model refused: {exc}") from exc
    ident = identity(frozen, pinned, inputs, batch)
    meta: dict[str, Any] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    if done and meta.get("identity") != ident:
        raise SystemExit(f"{data_rel(out, data)} was started with another model, settings, "
                         "inputs or tag; a resume must be the same evaluation")  # fmt: skip

    name = system_name(frozen)
    reason = args.reason.strip()
    if done:
        reason += f" (resumed: {len(done)} of {len(keys)} samples already predicted)"
    open_test_access(spec["benchmark"], split, args.experiment, f"{SYSTEM} ({args.mode})", 14,
                     reason, log_path())  # fmt: skip

    runs = list(meta.get("runs", []))
    seg = len(runs)
    git = git_info(repo_root())
    run_info: dict[str, Any] = {
        "segment": seg, "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "running", "already_done": len(done), "git_commit": git.commit,
        "git_dirty": git.dirty, "pinned": pinned, "device": frozen.provenance.get("device"),
        "batch": batch, "provenance": frozen.provenance, "reason": reason,
    }  # fmt: skip
    runs.append(run_info)
    meta = {
        "system": name, "task": task, "mode": args.mode, "eval_set": args.set,
        "benchmark": spec["benchmark"], "split": split, "experiment": args.experiment,
        "phase": 14, "version": f"{name}: {frozen.provenance.get('recognizer_experiment')} + "
                                f"{frozen.provenance.get('detector_experiment')}",
        "expected": len(keys), "samples": len(done), "temperature": frozen.recognizer.temperature,
        "confidences_at": [T1, frozen.recognizer.temperature],
        "settings": frozen.settings(), "provenance": frozen.provenance, "pinned": pinned,
        "inputs": inputs, "identity": ident, "runs": runs,
        "note": "predictions only: this runner computes no metric (GW-E2E-009 scores once); "
                "time_ms is diagnostic, latency comes from GW-EFF-005",
    }  # fmt: skip
    _write_json(meta_path, meta)

    start = time.monotonic()
    written = 0
    records = (
        _rec_records(frozen, data, args.set, done, args.batch, boxes)
        if args.mode == "rec"
        else _page_records(frozen, data, pages or [], done, args.mode)
    )
    with out.open("a", encoding="utf-8", newline="\n") as fh, torch.inference_mode():
        for record in records:
            fh.write(json.dumps({**record, "seg": seg}, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            if written % (1000 if args.mode == "rec" else 100) == 0:
                print(f"{len(done) + written}/{len(keys)}", flush=True)

    total = len(done) + written
    run_info.update(status="done", written=written, wall_seconds=round(time.monotonic() - start, 1),
                    ended=datetime.now(UTC).isoformat(timespec="seconds"))  # fmt: skip
    meta["samples"] = total
    _write_json(meta_path, meta)
    if total != len(keys) or count_lines(out) != len(keys):
        raise SystemExit(f"{data_rel(out, data)}: {count_lines(out)} records for {len(keys)} "
                         "keys; not recorded")  # fmt: skip
    record_output(data, out, args.experiment)
    print(f"{args.experiment} {args.mode} {args.set}: {written} new, {total} total")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--set", required=True, choices=sorted(EVAL_SETS))
    common.add_argument("--experiment", required=True)
    common.add_argument("--reason", required=True, help="goes into the test-access log")
    common.add_argument("--device", default="cuda")
    common.add_argument("--data-root", default="data")
    common.add_argument("--allow-unpinned", action="store_true",
                        help="tests only: a validation run without the tag")  # fmt: skip
    sub = parser.add_subparsers(dest="mode", required=True)
    rec = sub.add_parser("rec", parents=[common], help="recognizer on a crop pack")
    rec.add_argument("--batch", type=int, default=REC_CHUNK, help="crops per recognizer call")
    sub.add_parser("e2e", parents=[common], help="the pipeline on pages")
    sub.add_parser("oracle", parents=[common], help="ground-truth box lines through read_quads")
    args = parser.parse_args(argv)
    if not hasattr(args, "batch"):
        args.batch = REC_CHUNK
    if args.batch < 1:
        raise SystemExit("--batch must be at least 1")
    check_request(args)
    return run(args, Path(args.data_root))


if __name__ == "__main__":
    sys.exit(main())
