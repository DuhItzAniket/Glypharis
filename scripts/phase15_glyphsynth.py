"""GW-REC-041: the frozen GLYPHARIS v1 recognizer on GlyphSynth v1 validation lines (Phase 15).

    python scripts/phase15_glyphsynth.py                       # the full run, on cuda
    python scripts/phase15_glyphsynth.py --limit 20 --out tmp/gs.jsonl   # a smoke test

GlyphSynth lines carry what the real test sets have no labels for: the font (family, file,
category, held out from training or not), italic, weight, inverted colours, neighbour fragments
(a stand-in for loose crops), underlines and the capture steps (ink spread for touching
characters, binarisation and low resolution for broken ones, paper texture, noise and salt and
pepper for background noise). Phase 15 slices the frozen recognizer's errors by them
(docs/PHASE_STATUS.md, Phase 15 item 6); this runner only reads and writes, it computes no metric.

The recognizer is the frozen one, built by `glypharis.pipeline.frozen.load_frozen` from the freeze
record in strict fp32, and reads the crops in chunks of `ext_eval.REC_CHUNK` with confidences at
T = 1 and at the record's temperature, as the Phase 14 rec mode does. Before the model is built:

- the set must be GlyphSynth validation (`SET`); any other set is refused, and so is a pack whose
  records name another split or source. No test data is read, so no test access is logged;
- the pack's .bin must match its own .meta.json and the hash and count in
  data/manifests/processed/glyphsynth.json;
- every pack record is joined by key with exactly one row of ``lines_val.render.jsonl`` (the
  render labels; no text in them), and every font file with its face in
  data/manifests/glyphsynth_fonts.json, whose category must equal the file path's first folder.

Output: data/processed/phase15/GW-REC-041/glyphsynth_val.jsonl, one record per crop in pack order
(``key``, ``ref`` = the pack label, ``pred``, ``conf_t1``, ``conf_t``, the crop's
``crop_height_px``, ``blur`` and ``contrast`` as ext_eval measures them, the render labels, and
``category`` and ``held_out`` from the font manifest), and a ``.meta.json`` beside it (commit,
checkpoint hashes, device, settings, input hashes, wall time, count). The file is written under a
temporary name and renamed when complete, and an existing output is never overwritten. ``--limit``
reads the first N crops and must write somewhere other than the registered output. ``--out`` is
refused under data/processed/phase14/, the baseline folders and data/manifests/processed/
(`glypharis.eval.errors.PROTECTED_OUTPUTS`), before anything is read.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from ext_eval import REC_CHUNK, T1, crop_attributes

from glypharis.data.download import sha256_file
from glypharis.data.linepack import CropRecord, LinePackReader
from glypharis.eval.access import is_test_split
from glypharis.eval.errors import protected_folder
from glypharis.eval.sets import EVAL_SETS, repo_root
from glypharis.pipeline.frozen import Frozen, FrozenModelError, load_frozen

EXPERIMENT = "GW-REC-041"
SET = "glyphsynth_val"  # the only set this runner reads (validation)
PACK_SPLIT = "val"  # what GlyphSynth's own records call validation
PACK_SOURCE = "glyphsynth"
OUTPUT = Path("processed/phase15") / EXPERIMENT / f"{SET}.jsonl"  # under the data root
GLYPHSYNTH_MANIFEST = Path("manifests/processed/glyphsynth.json")  # bin hash and count per split
FONTS_MANIFEST = Path("manifests/glyphsynth_fonts.json")  # faces: path, family, category, split
RENDER_FIELDS = ("family", "font_file", "size_px", "italic", "weight", "inverted", "neighbours",
                 "underline", "capture")  # fmt: skip
TRAIN_FONT_SPLIT = "train"  # a face in any other split was held out of training


class GlyphSynthInputError(SystemExit):
    """An input differs from its record or cannot be joined; nothing is run."""


# ---------------------------------------------------------------------------------- inputs


def check_set(eval_set: str) -> dict[str, Any]:
    """The set's registry entry, or a refusal for anything but GlyphSynth validation."""
    spec = EVAL_SETS.get(eval_set)
    if eval_set != SET or spec is None or is_test_split(str(spec["split"])):
        raise SystemExit(f"{EXPERIMENT} runs on GlyphSynth validation ({SET}) only, not "
                         f"{eval_set}")  # fmt: skip
    if spec["split"] != "validation" or spec["benchmark"] != PACK_SOURCE or not spec["pack"]:
        raise SystemExit(f"{SET} is not registered as GlyphSynth validation with a line pack")
    return spec


def render_path(prefix: Path) -> Path:
    """``lines_val.render.jsonl`` beside the pack (``with_suffix`` would eat ``lines_val``)."""
    return prefix.parent / f"{prefix.name}.render.jsonl"


def check_pack(data: Path, prefix: Path, reader: LinePackReader) -> dict[str, Any]:
    """Hash the .bin and compare it with its .meta.json and the GlyphSynth manifest; the records
    must all be GlyphSynth validation and their keys unique."""
    bad = [r.key for r in reader.records if r.split != PACK_SPLIT or r.source != PACK_SOURCE]
    if bad:
        raise GlyphSynthInputError(f"{len(bad)} pack records are not GlyphSynth validation "
                                   f"(e.g. {bad[0]}); refusing to run")  # fmt: skip
    keys = [r.key for r in reader.records]
    if len(set(keys)) != len(keys):
        raise GlyphSynthInputError("a key appears twice in the pack index")
    if reader.meta.get("split") not in (None, PACK_SPLIT):
        raise GlyphSynthInputError(f"the pack's .meta.json names split {reader.meta['split']}")
    bin_sha = sha256_file(prefix.with_suffix(".bin"))
    if bin_sha != reader.meta["bin_sha256"]:
        raise GlyphSynthInputError(f"{prefix.name}.bin does not match its own .meta.json hash")
    manifest = data / GLYPHSYNTH_MANIFEST
    if not manifest.is_file():
        raise GlyphSynthInputError(f"data/{GLYPHSYNTH_MANIFEST.as_posix()} is missing; the pack "
                                   "has no recorded hash")  # fmt: skip
    recorded = (json.loads(manifest.read_text(encoding="utf-8")).get("splits") or {}).get(
        PACK_SPLIT
    ) or {}
    if recorded.get("bin_sha256") != bin_sha:
        raise GlyphSynthInputError(f"{prefix.name}.bin has SHA-256 {bin_sha}, "
                                   f"data/{GLYPHSYNTH_MANIFEST.as_posix()} records "
                                   f"{recorded.get('bin_sha256')}; refusing to run")  # fmt: skip
    if recorded.get("count") is not None and int(recorded["count"]) != len(reader):
        raise GlyphSynthInputError(f"the pack holds {len(reader)} crops, "
                                   f"data/{GLYPHSYNTH_MANIFEST.as_posix()} records "
                                   f"{recorded['count']}; refusing to run")  # fmt: skip
    return {"bin_sha256": bin_sha, "count": len(reader),
            "recorded_in": f"data/{GLYPHSYNTH_MANIFEST.as_posix()}"}  # fmt: skip


def read_render_log(path: Path) -> dict[str, dict[str, Any]]:
    """The render labels by key; a repeated key or a row without every `RENDER_FIELDS` refuses."""
    if not path.is_file():
        raise GlyphSynthInputError(f"{path.name} is missing; the render labels cannot be joined")
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("key")
            if not isinstance(key, str):
                raise GlyphSynthInputError(f"{path.name} line {n} has no key")
            if key in out:
                raise GlyphSynthInputError(f"{path.name}: key {key} appears twice")
            missing = [f for f in RENDER_FIELDS if f not in row]
            if missing:
                raise GlyphSynthInputError(f"{path.name}: {key} lacks {', '.join(missing)}")
            out[key] = {f: row[f] for f in RENDER_FIELDS}
    return out


def font_faces(path: Path) -> dict[str, dict[str, Any]]:
    """The font manifest's faces by file path (relative to the font root, as the render log
    names them)."""
    if not path.is_file():
        raise GlyphSynthInputError(f"data/{FONTS_MANIFEST.as_posix()} is missing")
    faces = json.loads(path.read_text(encoding="utf-8")).get("faces") or []
    out: dict[str, dict[str, Any]] = {}
    for face in faces:
        if face["path"] in out:
            raise GlyphSynthInputError(f"the font manifest lists {face['path']} twice")
        out[face["path"]] = face
    return out


def font_category(font_file: str) -> str:
    """The category is the font file path's first folder (serif, sans, mono, ...)."""
    parts = PurePosixPath(font_file).parts
    if len(parts) < 2:
        raise GlyphSynthInputError(f"font file {font_file!r} has no category folder")
    return parts[0]


def font_labels(render: dict[str, Any], faces: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """``category`` and ``held_out`` for one line, from its font file and the font manifest.

    A family counts as held out when any of its faces is outside the training split; the
    manifest splits by family, so this equals the face's own split (checked)."""
    font_file = str(render["font_file"])
    face = faces.get(font_file)
    if face is None:
        raise GlyphSynthInputError(f"font file {font_file} is not in the font manifest")
    category = font_category(font_file)
    if face.get("category") != category:
        raise GlyphSynthInputError(f"{font_file}: the manifest says category "
                                   f"{face.get('category')}, the path says {category}")  # fmt: skip
    if face.get("family") != render["family"]:
        raise GlyphSynthInputError(f"{font_file}: the render log says family {render['family']}, "
                                   f"the manifest {face.get('family')}")  # fmt: skip
    return {"category": category, "held_out": face.get("split") != TRAIN_FONT_SPLIT}


def held_out_families(faces: dict[str, dict[str, Any]]) -> set[str]:
    """Families with a face outside the training split; refuses a family split across both."""
    splits: dict[str, set[bool]] = {}
    for face in faces.values():
        splits.setdefault(str(face["family"]), set()).add(face.get("split") != TRAIN_FONT_SPLIT)
    mixed = sorted(f for f, s in splits.items() if len(s) > 1)
    if mixed:
        raise GlyphSynthInputError(f"families with faces both held out and trained on: {mixed}")
    return {f for f, s in splits.items() if True in s}


def join(
    records: list[CropRecord],
    render: dict[str, dict[str, Any]],
    faces: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per pack record (in pack order): its render labels and font labels. The two key sets must
    be equal: a crop without labels, or labels without a crop, refuses."""
    keys = {r.key for r in records}
    no_labels = sorted(keys - set(render))
    no_crop = sorted(set(render) - keys)
    if no_labels or no_crop:
        raise GlyphSynthInputError(f"the pack and the render log disagree: {len(no_labels)} crops "
                                   f"without labels (e.g. {no_labels[:1]}), {len(no_crop)} labels "
                                   f"without a crop (e.g. {no_crop[:1]})")  # fmt: skip
    held = held_out_families(faces)
    out = []
    for rec in records:
        labels = {**render[rec.key], **font_labels(render[rec.key], faces)}
        if labels["held_out"] != (labels["family"] in held):
            raise GlyphSynthInputError(f"{rec.key}: face and family disagree on held out")
        out.append(labels)
    return out


# ------------------------------------------------------------------------------------- run


def recognise(
    frozen: Frozen,
    reader: LinePackReader,
    labels: list[dict[str, Any]],
    n: int,
    chunk: int,
) -> Iterator[dict[str, Any]]:
    """The first ``n`` crops through the frozen recognizer, ``chunk`` at a time (ext_eval's rec
    path without page geometry: GlyphSynth has no pages)."""
    for start in range(0, n, chunk):
        idx = list(range(start, min(start + chunk, n)))
        images = [reader.image(i) for i in idx]
        reads = frozen.recognizer.read(images, (T1,))
        for i, img, r in zip(idx, images, reads, strict=True):
            rec = reader.records[i]
            yield {"key": rec.key, "ref": rec.label, "pred": r.text, "conf_t1": r.at[T1],
                   "conf_t": r.confidence, **crop_attributes(img), **labels[i]}  # fmt: skip


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root().resolve()).as_posix()
    except (ValueError, RuntimeError):
        return path.as_posix()


def check_out(out: Path, data: Path) -> None:
    """Refuse an output under a protected folder of this data root or the repository's."""
    roots = [data.resolve()]
    with contextlib.suppress(RuntimeError):  # not in a git repository: this run's root only
        roots.append((repo_root() / "data").resolve())
    for root in roots:
        hit = protected_folder(out.resolve(), root)
        if hit is not None:
            raise SystemExit(f"{out} is under data/{hit}, which Phase 15 never writes")


def run(args: argparse.Namespace) -> int:
    data = Path(args.data_root)
    spec = check_set(args.set)
    default_out = data / OUTPUT
    out = Path(args.out) if args.out else default_out
    check_out(out, data)
    if args.limit is not None and out.resolve() == default_out.resolve():
        raise SystemExit("--limit writes a partial file; give --out somewhere else than "
                         f"data/{OUTPUT.as_posix()}")  # fmt: skip
    meta_path = out.with_name(out.name.removesuffix(".jsonl") + ".meta.json")
    if out.exists() or meta_path.exists():
        raise SystemExit(f"{out} (or its .meta.json) already exists; move it aside first")

    prefix = data / spec["pack"]
    reader = LinePackReader(prefix)
    pack = check_pack(data, prefix, reader)
    render_file = render_path(prefix)
    labels = join(reader.records, read_render_log(render_file), font_faces(data / FONTS_MANIFEST))
    n = len(reader) if args.limit is None else min(args.limit, len(reader))
    inputs = {
        "pack": {"path": f"data/{spec['pack']}", **pack,
                 "index_sha256": sha256_file(prefix.with_suffix(".index.jsonl"))},
        "render_log": {"path": f"data/{spec['pack']}.render.jsonl",
                       "sha256": sha256_file(render_file), "rows": len(labels)},
        "fonts": {"path": f"data/{FONTS_MANIFEST.as_posix()}",
                  "sha256": sha256_file(data / FONTS_MANIFEST)},
    }  # fmt: skip

    try:
        frozen = load_frozen(args.device, root=repo_root())
    except FrozenModelError as exc:
        raise SystemExit(f"frozen model refused: {exc}") from exc
    p = frozen.provenance
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    started = datetime.now(UTC).isoformat(timespec="seconds")
    t0 = time.monotonic()
    written = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as fh, torch.inference_mode():
        for record in recognise(frozen, reader, labels, n, args.batch):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"{written}/{n}", flush=True)
    reader.close()
    wall = round(time.monotonic() - t0, 1)
    if written != n:
        raise SystemExit(f"{written} records for {n} crops; {tmp} kept, not renamed")
    tmp.replace(out)
    meta = {
        "experiment": EXPERIMENT, "phase": 15, "eval_set": args.set,
        "benchmark": spec["benchmark"], "split": spec["split"],
        "system": str(p.get("name") or "glypharis_v1"),
        "version": f"{p.get('name')}: {p.get('recognizer_experiment')}",
        "commit": p.get("git_commit"), "git_dirty": p.get("git_dirty"),
        "runner": _rel(Path(__file__)), "runner_sha256": sha256_file(Path(__file__)),
        "recognizer_checkpoint": p.get("recognizer_checkpoint"),
        "recognizer_checkpoint_sha256": p.get("recognizer_checkpoint_sha256"),
        "record_sha256": p.get("record_sha256"), "device": p.get("device"),
        "device_name": p.get("device_name"), "batch": args.batch,
        "temperature": frozen.recognizer.temperature,
        "confidences_at": [T1, frozen.recognizer.temperature],
        "settings": frozen.settings()["recognizer"], "provenance": p, "inputs": inputs,
        "expected": len(reader), "limit": args.limit, "count": written,
        "started": started, "ended": datetime.now(UTC).isoformat(timespec="seconds"),
        "wall_seconds": wall, "output": _rel(out), "output_sha256": sha256_file(out),
        "note": "validation only (synthetic GlyphSynth v1 lines): no test data is read and no "
                "test access is logged; predictions only, Phase 15's analysis scores them",
    }  # fmt: skip
    _write_json(meta_path, meta)
    print(f"{EXPERIMENT} {args.set}: {written} of {len(reader)} crops in {wall} s -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--set", default=SET, help=f"must be {SET} (refused otherwise)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", help=f"default <data-root>/{OUTPUT.as_posix()}")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=REC_CHUNK, help="crops per recognizer call")
    parser.add_argument("--limit", type=int, help="the first N crops only (smoke tests)")
    args = parser.parse_args(argv)
    if args.batch < 1:
        raise SystemExit("--batch must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
