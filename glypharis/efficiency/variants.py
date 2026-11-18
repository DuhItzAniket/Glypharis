"""Phase 16 variants: one setting of v1 changed, every other one asserted equal (GW-EFF-008, 009).

A variant is a new version, never v1 (Phase 16 plan, item 1): it is built by changing exactly one
named setting (`SETTINGS`) of the v1 fp32 ONNX build, and every output says so
(``"version": VARIANT_LABEL``). The chain of checks, each refusing with `VariantError`:

1. **record -> v1 (PyTorch):** `glypharis.pipeline.frozen.load_frozen` builds v1 from the freeze
   record and checks the checkpoint hashes, the charset and the settings it holds;
2. **v1 (PyTorch) -> v1 (fp32 ONNX):** the GW-EFF-006 bundle is loaded with
   `onnx_build.load_bundle_onnx` (manifest hashes checked), its release weights must equal the
   record's checkpoints tensor for tensor (`state_sha256`), and its settings may differ from v1's
   only in the two runtimes;
3. **v1 (fp32 ONNX) -> variant:** the variant's settings may differ from the ONNX build's only in
   its own setting, and must differ there.

`prepare_variant` writes whatever model files a variant needs (the INT8 recognizer, the fused
attention export, another checkpoint's export with its release file or detector spec) into a work
directory once, with ``variant.json`` holding their hashes and sources (a failed preparation
removes what it wrote); `load_variant` rebuilds the pipeline from those files, checks every hash
again and runs the chain above, so the accuracy run and both latency runs use the same bytes. The
v1 builds themselves (``v1-torch``, ``v1-onnx``) go through the same path, labelled
``"version": "v1"``, so every Pareto point is measured by the same code.

`deployed_variant` builds the same pipeline the way a deployed build loads it: from the bundle
(`onnx_build.load_bundle_onnx`, as `load_bundle` does) and the variant's own files only, with no
training checkpoint and no second PyTorch copy of v1, so a profile's peak memory is the build's
own. A profile times that pipeline, reads its peak memory, and only then runs `load_variant`'s
chain and `same_build`, which refuses unless both pipelines hold the same settings and models.

Settings are read from the built objects (the detector spec, the recognizer's input settings,
temperature and charset, the crop settings and reading order); the model identity is the content
hash of the weights the part was built from, so a checkpoint, its release file and the export made
from it name the same model.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import torch

from glypharis.config.loader import from_dict
from glypharis.config.schema import ModelConfig
from glypharis.data.download import sha256_file
from glypharis.data.text import Charset
from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.fused_attention import export_fused_recognizer
from glypharis.efficiency.onnx_build import (
    load_bundle_onnx,
    onnx_detector,
    onnx_recognizer,
    with_parts,
)
from glypharis.efficiency.quant import quantize_recognizer
from glypharis.export.bundle import read_manifest
from glypharis.export.onnx import (
    OnnxLineModel,
    OnnxProbMap,
    export_detector_onnx,
    export_recognizer_onnx,
)
from glypharis.models.det.detector import DetectorSpec, TextLineDetector
from glypharis.models.rec.crnn import build_recognizer
from glypharis.pipeline.frozen import FREEZE_RECORD, Frozen, load_frozen, strict_inference
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer
from glypharis.training.checkpoint import load_checkpoint

VARIANT_LABEL: Final = "variant (new version, not v1)"
V1_LABEL: Final = "v1"
VARIANT_FILE: Final = "variant.json"
WORK_ROOT: Final = Path("data/processed/phase16/variants")  # git-ignored

# Runtimes, as recorded in the settings.
TORCH: Final = "torch fp32"
TORCH_STRICT: Final = "torch fp32 (strict)"
ORT_FP32: Final = "onnxruntime fp32"
ORT_INT8: Final = "onnxruntime int8 (dynamic, MatMul weights; glypharis.efficiency.quant.RECIPE)"
ORT_FUSED: Final = "onnxruntime fp32, fused attention (com.microsoft.MultiHeadAttention)"

SETTINGS: Final = (
    "detector.model",  # experiment, weights hash, architecture, label settings
    "detector.runtime",
    "detector.infer",  # resolution (long side)
    "detector.post",
    "detector.operating_point",
    "recognizer.model",  # experiment, weights hash, architecture, input height and width cap
    "recognizer.runtime",
    "recognizer.charset",
    "recognizer.temperature",
    "crop",
    "reading_order",
    "max_pixels",
)
RUNTIMES: Final = frozenset({"detector.runtime", "recognizer.runtime"})


class VariantError(RuntimeError):
    """A variant changes more or less than its one setting, or its files are not the recorded
    ones; nothing is measured."""


@dataclass(frozen=True)
class VariantSpec:
    name: str
    setting: str | None  # the one setting it changes; None for a v1 build
    about: str
    measure: tuple[str, ...]  # what GW-EFF-009 measures it on: "pages", "lines"
    experiment: str = "GW-EFF-009"
    checkpoint: str | None = None  # experiment of the checkpoint it needs (given by path)
    long_side: int | None = None


def _variants() -> dict[str, VariantSpec]:
    both = ("pages", "lines")
    specs = [
        VariantSpec("v1-torch", None, "v1 in PyTorch, built from the freeze record", both),
        VariantSpec("v1-onnx", None, "v1 in fp32 ONNX: the GW-EFF-006 bundle", both),
        *(
            VariantSpec(
                f"det-long-side-{n}",
                "detector.infer",
                f"detector long side {n}",
                ("pages",),
                long_side=n,
            )
            for n in (1024, 1280, 1792, 2048)
        ),
        VariantSpec(
            "onnx-det-torch-rec",
            "recognizer.runtime",
            "the ONNX detector with the PyTorch recognizer",
            ("pages",),
        ),
        VariantSpec(
            "int8-recognizer",
            "recognizer.runtime",
            "dynamic INT8 of the v1 fp32 ONNX recognizer (H7′)",
            both,
            experiment="GW-EFF-008",
        ),
        VariantSpec(
            "fused-attention",
            "recognizer.runtime",
            "the v1 recognizer exported with fused attention",
            both,
        ),
        VariantSpec(
            "det-GW-DET-003",
            "detector.model",
            "the smaller detector GW-DET-003",
            ("pages",),
            checkpoint="GW-DET-003",
        ),
        *(
            VariantSpec(
                f"rec-{e}",
                "recognizer.model",
                f"the recognizer {e} in fp32 ONNX",
                ("lines",),
                checkpoint=e,
            )
            for e in ("GW-REC-003", "GW-REC-006", "GW-REC-020", "GW-REC-022")
        ),
    ]
    return {s.name: s for s in specs}


VARIANTS: Final = _variants()

# The latency sittings and the (variant, set) arms each one measures, in the registered order
# (the runs go 1 ... k, then k ... 1, in one sitting). Each sitting has its own profile files, so
# a confirmatory sitting is never a Pareto run: GW-EFF-008 is H7′ (fp32, INT8), GW-EFF-011 is H6′
# (CRNN, SVTR), GW-EFF-009 every Pareto point of one set (the variants' order, `VARIANTS`).
SITTINGS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "GW-EFF-008": {"lines": ("v1-onnx", "int8-recognizer")},
    "GW-EFF-011": {"lines": ("rec-GW-REC-003", "rec-GW-REC-020")},
    "GW-EFF-009": {
        unit: tuple(v.name for v in VARIANTS.values() if unit in v.measure)
        for unit in ("pages", "lines")
    },
}


def sitting_order(
    experiment: str, unit: str, present: set[str] | None = None
) -> list[tuple[str, int]]:
    """The runs of one sitting in their registered order, ``(variant, run)``: run 1 of each arm,
    then run 2 in reverse; ``present`` keeps only those arms (a point cut from the plan)."""
    arms = [a for a in SITTINGS[experiment][unit] if present is None or a in present]
    return [(a, 1) for a in arms] + [(a, 2) for a in reversed(arms)]


def profile_file(unit: str, threads: int, experiment: str, run: int) -> str:
    """The file name of one latency run of a variant (under its variant folder)."""
    return f"profile_{unit}_{threads}t_{experiment}_run{run}.json"


def variant_spec(name: str) -> VariantSpec:
    try:
        return VARIANTS[name]
    except KeyError:
        raise VariantError(f"unknown variant {name!r}; one of {', '.join(VARIANTS)}") from None


def label(name: str) -> str:
    """``"v1"`` for a v1 build, `VARIANT_LABEL` for everything else."""
    return V1_LABEL if variant_spec(name).setting is None else VARIANT_LABEL


# ----------------------------------------------------------------------------------- settings


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 of a state dict's names, dtypes, shapes and bytes, in name order: the same weights
    give the same hash whether they came from a checkpoint, a release file or a built model."""
    h = hashlib.sha256()
    for key in sorted(state):
        t = state[key].detach().cpu().contiguous()
        h.update(f"{key}|{t.dtype}|{tuple(t.shape)}|".encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def charset_sha256(charset: Charset) -> str:
    return hashlib.sha256(json.dumps(charset.characters, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class Part:
    """Which weights a pipeline part was built from, and the runtime it runs on."""

    experiment: str
    weights_sha256: str
    runtime: str


def _runtime_ok(part: Part, onnx: bool, what: str) -> None:
    if part.runtime.startswith("onnxruntime") != onnx:
        raise VariantError(
            f"the {what} is recorded as {part.runtime!r} but runs on another backend"
        )


def pipeline_settings(ocr: GlypharisOCR, detector: Part, recognizer: Part) -> dict[str, Any]:
    """`SETTINGS` of a built pipeline, read from its objects; the parts give the model identity
    and runtime (checked against the backend each part actually has)."""
    _runtime_ok(detector, isinstance(ocr.detector.backend, OnnxProbMap), "detector")
    _runtime_ok(recognizer, isinstance(ocr.recognizer.model, OnnxLineModel), "recognizer")
    spec, rec = ocr.detector.spec, ocr.recognizer
    return {
        "detector.model": {
            "experiment": detector.experiment,
            "weights_sha256": detector.weights_sha256,
            "arch": dataclasses.asdict(spec.model),
            "labels": dataclasses.asdict(spec.labels),
            "label_version": spec.label_version,
        },
        "detector.runtime": detector.runtime,
        "detector.infer": dataclasses.asdict(spec.infer),
        "detector.post": dataclasses.asdict(spec.post),
        "detector.operating_point": spec.operating_point,
        "recognizer.model": {
            "experiment": recognizer.experiment,
            "weights_sha256": recognizer.weights_sha256,
            "arch": rec.arch,
            "height": rec.height,
            "max_width": rec.max_width,
            "stride": int(rec.model.stride),
            "classes": int(rec.model.num_classes),
        },
        "recognizer.runtime": recognizer.runtime,
        "recognizer.charset": charset_sha256(rec.charset),
        "recognizer.temperature": rec.temperature,
        "crop": dataclasses.asdict(ocr.crop),
        "reading_order": ocr.order,
        "max_pixels": ocr.max_pixels,
    }


def compare_settings(
    base: Mapping[str, Any], other: Mapping[str, Any], changed: frozenset[str], what: str
) -> dict[str, Any]:
    """`VariantError` unless ``other`` differs from ``base`` in exactly the settings ``changed``;
    returns ``{setting: {"from", "to"}}`` for those."""
    if set(base) != set(SETTINGS) or set(other) != set(SETTINGS):
        raise VariantError(f"{what}: the settings must be exactly {SETTINGS}")
    differ = {k for k in SETTINGS if base[k] != other[k]}
    extra, missing = differ - changed, changed - differ
    if extra:
        raise VariantError(
            f"{what}: {sorted(extra)} differ as well, so it changes more than "
            f"{sorted(changed)}: {[(k, base[k], other[k]) for k in sorted(extra)]}"
        )
    if missing:
        raise VariantError(f"{what}: {sorted(missing)} equal v1's, so it is not a variant")
    return {k: {"from": base[k], "to": other[k]} for k in sorted(changed)}


# ------------------------------------------------------------------------------- the v1 builds


def v1_parts(frozen: Frozen) -> tuple[Part, Part]:
    """The parts of v1 in PyTorch, by the freeze record's experiments and the loaded weights."""
    det_model = frozen.ocr.detector.backend.model  # type: ignore[attr-defined]
    det = Part(
        frozen.provenance["detector_experiment"], state_sha256(det_model.state_dict()), TORCH
    )
    rec = Part(
        frozen.provenance["recognizer_experiment"],
        state_sha256(frozen.recognizer.model.state_dict()),
        TORCH_STRICT if frozen.recognizer.strict_fp32 else TORCH,
    )
    return det, rec


def bundle_info(bundle: str | Path) -> dict[str, Any]:
    """The bundle's manifest hash and file hashes (after `read_manifest` checked every file)."""
    path = Path(bundle)
    manifest = read_manifest(path)
    return {
        "path": path.as_posix(),
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "version": manifest.get("version"),
        "files": dict(manifest["files"]),
    }


def v1_onnx(
    frozen: Frozen, bundle: str | Path, threads: int | None
) -> tuple[GlypharisOCR, Part, Part]:
    """v1 in fp32 ONNX from the bundle: its release weights must be the record's, tensor for
    tensor, and its settings v1's apart from the runtimes."""
    path = Path(bundle)
    ocr = load_bundle_onnx(path, threads)
    det_v1, rec_v1 = v1_parts(frozen)
    released_rec = TextRecognizer.from_release(path / "recognizer.pt", "cpu")
    released_det = TextLineDetector.from_release(path / "detector.pt")
    got_rec = state_sha256(released_rec.model.state_dict())
    got_det = state_sha256(released_det.backend.model.state_dict())  # type: ignore[attr-defined]
    if got_rec != rec_v1.weights_sha256:
        raise VariantError(f"{path}: recognizer.pt does not hold the freeze record's weights")
    if got_det != det_v1.weights_sha256:
        raise VariantError(f"{path}: detector.pt does not hold the freeze record's weights")
    det = Part(det_v1.experiment, got_det, ORT_FP32)
    rec = Part(rec_v1.experiment, got_rec, ORT_FP32)
    compare_settings(
        pipeline_settings(frozen.ocr, det_v1, rec_v1),
        pipeline_settings(ocr, det, rec),
        RUNTIMES,
        f"the bundle {path}",
    )
    return ocr, det, rec


# --------------------------------------------------------------------- checkpoints given by path


def checkpoint_recognizer(path: str | Path, temperature: float) -> tuple[TextRecognizer, str]:
    """A PyTorch recognizer from a training checkpoint (trusted local file) with its own charset,
    architecture and input settings, at ``temperature``; and the checkpoint's experiment."""
    ck = load_checkpoint(Path(path), "cpu")
    config = ck.get("config") or {}
    chars = list(ck.get("charset") or [])
    if not chars:
        raise VariantError(f"{path}: the checkpoint holds no character list")
    charset = Charset(chars)
    model_cfg = from_dict(ModelConfig, config["model"], "model")
    name = str(config.get("model_name") or "crnn")
    model = build_recognizer(charset.num_classes, model_cfg, name)
    model.load_state_dict(ck["model"])
    data = config.get("data") or {}
    arch = {"model_name": name, "model": dataclasses.asdict(model_cfg)}
    rec = TextRecognizer(
        model,
        charset,
        int(data["image_height"]),
        int(data["max_width"]),
        "cpu",
        temperature,
        arch,
        strict_fp32=True,
    )
    return rec, str(config.get("experiment_id"))


def checkpoint_detector(path: str | Path) -> tuple[TextLineDetector, str]:
    """A PyTorch detector from a training checkpoint, and the checkpoint's experiment."""
    ck = load_checkpoint(Path(path), "cpu")
    experiment = str((ck.get("config") or {}).get("experiment_id"))
    del ck
    return TextLineDetector.from_checkpoint(path, "cpu"), experiment


def _checkpoint(spec: VariantSpec, path: str | Path | None) -> dict[str, Any]:
    if spec.checkpoint is None:
        if path is not None:
            raise VariantError(f"{spec.name} takes no checkpoint")
        return {}
    if path is None:
        raise VariantError(f"{spec.name} needs the {spec.checkpoint} checkpoint (--checkpoint)")
    p = Path(path)
    if not p.is_file():
        raise VariantError(f"{p}: no such checkpoint")
    return {"path": p.as_posix(), "sha256": sha256_file(p), "experiment": spec.checkpoint}


def portable_path(path: str | Path) -> str:
    """A checkpoint path as a report records it: from its ``checkpoints`` folder on (or the file
    name alone), so no machine's folders or user name reach a committed file. The work folder's
    record keeps the full path, which the later steps read; the SHA-256 names the file."""
    parts = Path(path).as_posix().split("/")
    if "checkpoints" in parts:
        return "/".join(parts[len(parts) - 1 - parts[::-1].index("checkpoints") :])
    return parts[-1]


def report_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """A copy of a build record (`Build.record`) for a report, with the checkpoint path in its
    copy of the work folder's record made portable. ``variant_json_sha256`` still names the work
    folder's file, which keeps the full path."""
    out = json.loads(json.dumps(record))
    for holder in (out, out.get("variant_json") or {}):
        ck = holder.get("checkpoint")
        if ck and "path" in ck:
            ck["path"] = portable_path(ck["path"])
    return out


def _same_experiment(spec: VariantSpec, got: str) -> None:
    if got != spec.checkpoint:
        raise VariantError(f"{spec.name}: the checkpoint is {got}, not {spec.checkpoint}")


# --------------------------------------------------------------------------- prepare and load


def _file(work: Path, name: str) -> dict[str, Any]:
    p = work / name
    return {"sha256": sha256_file(p), "bytes": p.stat().st_size}


def prepare_variant(
    name: str,
    frozen: Frozen,
    bundle: str | Path | None,
    work: str | Path,
    threads: int | None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Write the variant's model files and ``variant.json`` into ``work`` (refused if it already
    holds one) and return what was written. Every variant but ``v1-torch`` needs the bundle."""
    spec = variant_spec(name)
    work = Path(work)
    if name == "v1-torch":
        bundle = None  # PyTorch from the record; no bundle is read
    if (work / VARIANT_FILE).exists():
        raise VariantError(f"{work / VARIANT_FILE} exists: {name} was prepared already")
    if bundle is None and name != "v1-torch":
        raise VariantError(f"{name} is built on the v1 fp32 ONNX bundle (--bundle)")
    ck = _checkpoint(spec, checkpoint)
    work.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in work.iterdir()}
    try:
        return _prepare(name, spec, frozen, bundle, work, threads, ck)
    except BaseException:
        # Leave the directory as it was, so a retry is not refused by a half-written variant.
        for p in work.iterdir():
            if p.name not in before and p.is_file():
                p.unlink()
        raise


def _plain(obj: Any) -> Any:
    """``obj`` as JSON reads it back (tuples become lists), for comparing with a stored copy."""
    return json.loads(json.dumps(obj))


def _variant_detector_spec(base: DetectorSpec, det: TextLineDetector) -> DetectorSpec:
    """v1's detector spec with another detector's model, labels and label version."""
    return dataclasses.replace(
        base, model=det.spec.model, labels=det.spec.labels, label_version=det.spec.label_version
    )


def _prepare(
    name: str,
    spec: VariantSpec,
    frozen: Frozen,
    bundle: str | Path | None,
    work: Path,
    threads: int | None,
    ck: dict[str, Any],
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "variant": name,
        "version": label(name),
        "setting": spec.setting,
        "about": spec.about,
        "record_sha256": frozen.provenance["record_sha256"],
        "bundle": bundle_info(bundle) if bundle is not None else None,
        "checkpoint": ck or None,
        "files": {},
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if name == "int8-recognizer":
        assert bundle is not None
        doc["quantization"] = quantize_recognizer(
            Path(bundle) / "recognizer.onnx", work / "recognizer.int8.onnx"
        )
        doc["files"]["recognizer.int8.onnx"] = _file(work, "recognizer.int8.onnx")
    elif name == "fused-attention":
        rec = frozen.recognizer
        doc["fused_attention"] = export_fused_recognizer(
            rec.model, rec.height, work / "recognizer.fused.onnx", threads
        )
        doc["files"]["recognizer.fused.onnx"] = _file(work, "recognizer.fused.onnx")
    elif spec.setting == "detector.model":
        det, experiment = checkpoint_detector(ck["path"])
        _same_experiment(spec, experiment)
        export_detector_onnx(det.backend.model, work / "detector.onnx")  # type: ignore[attr-defined]
        doc["files"]["detector.onnx"] = _file(work, "detector.onnx")
        # What a deployed build needs besides the network, so it never reads the checkpoint.
        doc["detector_spec"] = _plain(
            _variant_detector_spec(frozen.ocr.detector.spec, det).to_dict()
        )
    elif spec.setting == "recognizer.model":
        rec, experiment = checkpoint_recognizer(ck["path"], frozen.recognizer.temperature)
        _same_experiment(spec, experiment)
        export_recognizer_onnx(rec.model, rec.height, work / "recognizer.onnx")
        doc["files"]["recognizer.onnx"] = _file(work, "recognizer.onnx")
        # The release file (charset, input settings, architecture) a deployed build reads.
        torch.save(rec.release(), work / "recognizer.pt")
        doc["files"]["recognizer.pt"] = _file(work, "recognizer.pt")
    write_json_atomic(work / VARIANT_FILE, doc)
    return doc


@dataclass
class Build:
    """A built pipeline and everything that says what it is."""

    name: str
    ocr: GlypharisOCR
    record: dict[str, Any]  # goes into every output as "build"
    torch_reference: TextRecognizer | None  # the PyTorch model an ONNX recognizer was made from

    @property
    def version(self) -> str:
        return str(self.record["version"])


def _check_prepared(
    doc: dict[str, Any], name: str, frozen: Frozen, bundle: Path | None, work: Path
) -> None:
    _check_files(doc, name, bundle, work)
    if doc.get("record_sha256") != frozen.provenance["record_sha256"]:
        raise VariantError(f"{work / VARIANT_FILE} was prepared from another freeze record")


def _check_files(doc: dict[str, Any], name: str, bundle: Path | None, work: Path) -> None:
    """``variant.json`` is ``name``'s, made on ``bundle``, and every file it lists is unchanged."""
    if doc.get("variant") != name:
        raise VariantError(f"{work / VARIANT_FILE} is for {doc.get('variant')!r}, not {name!r}")
    want = doc.get("bundle")
    if (want is None) != (bundle is None):
        raise VariantError(f"{name}: the bundle given does not match {work / VARIANT_FILE}")
    if want is not None and bundle is not None:
        got = bundle_info(bundle)
        if (got["manifest_sha256"], got["files"]) != (want["manifest_sha256"], want["files"]):
            raise VariantError(f"{bundle}: not the bundle {name} was prepared from")
    for fname, meta in (doc.get("files") or {}).items():
        p = work / fname
        if not p.is_file() or sha256_file(p) != meta["sha256"]:
            raise VariantError(f"{p}: missing or changed since {name} was prepared")
    ck = doc.get("checkpoint")
    if ck and sha256_file(Path(ck["path"])) != ck["sha256"]:
        raise VariantError(f"{ck['path']}: changed since {name} was prepared")


def _read_variant(name: str, work: Path) -> dict[str, Any]:
    try:
        doc: dict[str, Any] = json.loads((work / VARIANT_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VariantError(f"{work / VARIANT_FILE}: not readable; prepare {name} first") from exc
    return doc


def _check_release(path: Path, torch_ref: TextRecognizer, name: str) -> None:
    """The release file a deployed build reads holds ``torch_ref``'s weights and settings."""
    rel = TextRecognizer.from_release(path, "cpu")
    got = (
        state_sha256(rel.model.state_dict()),
        charset_sha256(rel.charset),
        rel.height,
        rel.max_width,
        rel.temperature,
        _plain(rel.arch),
    )
    want = (
        state_sha256(torch_ref.model.state_dict()),
        charset_sha256(torch_ref.charset),
        torch_ref.height,
        torch_ref.max_width,
        torch_ref.temperature,
        _plain(torch_ref.arch),
    )
    if got != want:
        raise VariantError(f"{name}: {path.name} does not hold the checkpoint's recognizer")


def load_variant(
    name: str,
    frozen: Frozen,
    bundle: str | Path | None,
    work: str | Path,
    threads: int | None,
) -> Build:
    """The pipeline of ``name`` from its prepared files (see the module docstring for the
    checks). ONNX sessions get ``threads`` intra-op threads."""
    spec = variant_spec(name)
    work = Path(work)
    bundle_path = Path(bundle) if bundle is not None and name != "v1-torch" else None
    doc = _read_variant(name, work)
    _check_prepared(doc, name, frozen, bundle_path, work)
    det_v1, rec_v1 = v1_parts(frozen)
    v1_settings = pipeline_settings(frozen.ocr, det_v1, rec_v1)
    torch_ref: TextRecognizer | None = None
    changed: dict[str, Any] | None = None
    if name == "v1-torch":
        ocr, det, rec = frozen.ocr, det_v1, rec_v1
    else:
        assert bundle_path is not None
        base, det, rec = v1_onnx(frozen, bundle_path, threads)
        base_settings = pipeline_settings(base, det, rec)
        ocr, torch_ref = base, frozen.recognizer
        if spec.long_side is not None:
            dspec = base.detector.spec
            infer = dataclasses.replace(dspec.infer, long_side=spec.long_side)
            detector = TextLineDetector(
                base.detector.backend, dataclasses.replace(dspec, infer=infer)
            )
            ocr, torch_ref = with_parts(base, detector=detector), None
        elif name == "onnx-det-torch-rec":
            ocr, rec, torch_ref = with_parts(base, recognizer=frozen.recognizer), rec_v1, None
        elif name == "int8-recognizer":
            ocr = with_parts(
                base,
                recognizer=onnx_recognizer(work / "recognizer.int8.onnx", base.recognizer, threads),
            )
            rec, torch_ref = Part(rec.experiment, rec.weights_sha256, ORT_INT8), None
        elif name == "fused-attention":
            ocr = with_parts(
                base,
                recognizer=onnx_recognizer(
                    work / "recognizer.fused.onnx", base.recognizer, threads
                ),
            )
            rec = Part(rec.experiment, rec.weights_sha256, ORT_FUSED)
        elif spec.setting == "detector.model":
            torch_det, experiment = checkpoint_detector(doc["checkpoint"]["path"])
            _same_experiment(spec, experiment)
            dspec = _variant_detector_spec(base.detector.spec, torch_det)
            if doc.get("detector_spec") != _plain(dspec.to_dict()):
                raise VariantError(
                    f"{name}: {VARIANT_FILE}'s detector spec is not the checkpoint's"
                )
            model = torch_det.backend.model  # type: ignore[attr-defined]
            det = Part(experiment, state_sha256(model.state_dict()), ORT_FP32)
            ocr = with_parts(base, detector=onnx_detector(work / "detector.onnx", dspec, threads))
            torch_ref = None
        elif spec.setting == "recognizer.model":
            torch_ref, experiment = checkpoint_recognizer(
                doc["checkpoint"]["path"], base.recognizer.temperature
            )
            _same_experiment(spec, experiment)
            _check_release(work / "recognizer.pt", torch_ref, name)
            rec = Part(experiment, state_sha256(torch_ref.model.state_dict()), ORT_FP32)
            ocr = with_parts(
                base, recognizer=onnx_recognizer(work / "recognizer.onnx", torch_ref, threads)
            )
        if spec.setting is not None:
            changed = compare_settings(
                base_settings, pipeline_settings(ocr, det, rec), frozenset({spec.setting}), name
            )
    settings = pipeline_settings(ocr, det, rec)
    if spec.setting is not None:
        ocr.model_version = f"{frozen.provenance.get('name')} {name} ({VARIANT_LABEL})"
    record = {
        "variant": name,
        "version": label(name),
        "setting": spec.setting,
        "about": spec.about,
        "changed": changed,
        "settings": settings,
        "v1_settings": v1_settings,
        "variant_json": doc,
        "variant_json_sha256": sha256_file(work / VARIANT_FILE),
        "provenance": frozen.provenance,
    }
    return Build(name, ocr, record, torch_ref)


def _strict_like(rel: TextRecognizer) -> TextRecognizer:
    """``rel`` with strict fp32 numerics, as `load_frozen` and `checkpoint_recognizer` build it."""
    return TextRecognizer(
        rel.model, rel.charset, rel.height, rel.max_width, "cpu", rel.temperature, rel.arch,
        strict_fp32=True,
    )  # fmt: skip


def deployed_variant(
    name: str, bundle: str | Path, work: str | Path, threads: int | None
) -> tuple[GlypharisOCR, dict[str, Any]]:
    """The pipeline of ``name`` (any variant but ``v1-torch``) as a deployed build loads it: the
    bundle through `load_bundle_onnx` (manifest hashes checked) and the variant's files (hashes
    checked against ``variant.json``), nothing else. Returns it with what was loaded. Its
    identity is proven afterwards by `same_build` against `load_variant`'s pipeline."""
    spec = variant_spec(name)
    if spec.name == "v1-torch":
        raise VariantError("v1-torch is built from the freeze record (load_frozen)")
    work, path = Path(work), Path(bundle)
    doc = _read_variant(name, work)
    _check_files(doc, name, path, work)
    base = load_bundle_onnx(path, threads)
    ocr = base
    loaded = ["the bundle on ONNX Runtime (load_bundle_onnx)"]
    numerics = None
    if spec.long_side is not None:
        dspec = base.detector.spec
        infer = dataclasses.replace(dspec.infer, long_side=spec.long_side)
        detector = TextLineDetector(base.detector.backend, dataclasses.replace(dspec, infer=infer))
        ocr = with_parts(base, detector=detector)
    elif name == "onnx-det-torch-rec":
        rec = _strict_like(TextRecognizer.from_release(path / "recognizer.pt", "cpu"))
        numerics = strict_inference("cpu")  # as load_frozen sets them for the PyTorch recognizer
        ocr = with_parts(base, recognizer=rec)
        loaded.append("the bundle's recognizer.pt in PyTorch (strict fp32)")
    elif name in ("int8-recognizer", "fused-attention"):
        file = "recognizer.int8.onnx" if name == "int8-recognizer" else "recognizer.fused.onnx"
        ocr = with_parts(base, recognizer=onnx_recognizer(work / file, base.recognizer, threads))
        loaded.append(file)
    elif spec.setting == "detector.model":
        stored = doc.get("detector_spec")
        if not isinstance(stored, dict):
            raise VariantError(f"{work / VARIANT_FILE} holds no detector spec; prepare it again")
        detector = onnx_detector(work / "detector.onnx", DetectorSpec.from_dict(stored), threads)
        ocr = with_parts(base, detector=detector)
        loaded.append("detector.onnx with the spec in variant.json")
    elif spec.setting == "recognizer.model":
        if "recognizer.pt" not in (doc.get("files") or {}):
            raise VariantError(f"{work / VARIANT_FILE} lists no recognizer.pt; prepare it again")
        like = _strict_like(TextRecognizer.from_release(work / "recognizer.pt", "cpu"))
        ocr = with_parts(base, recognizer=onnx_recognizer(work / "recognizer.onnx", like, threads))
        loaded.append("recognizer.onnx with the charset and settings of recognizer.pt")
    return ocr, {"loaded": loaded, "numerics": numerics}


def _parts(settings: dict[str, Any]) -> tuple[Part, Part]:
    det, rec = settings["detector.model"], settings["recognizer.model"]
    return (
        Part(det["experiment"], det["weights_sha256"], settings["detector.runtime"]),
        Part(rec["experiment"], rec["weights_sha256"], settings["recognizer.runtime"]),
    )


def same_build(ocr: GlypharisOCR, build: Build) -> dict[str, Any]:
    """`VariantError` unless ``ocr`` (a `deployed_variant`) holds the settings of ``build`` (a
    `load_variant`, whose chain checked the model identities) on the same backends; returns
    them. The model identities are ``build``'s: both pipelines read the same hashed files."""
    settings = build.record["settings"]
    det, rec = _parts(settings)
    got = pipeline_settings(ocr, det, rec)
    differ = sorted(k for k in SETTINGS if _plain(got[k]) != _plain(settings[k]))
    if differ:
        raise VariantError(f"{build.name}: the deployed build differs from the checked one in "
                           f"{differ}")  # fmt: skip
    ocr.model_version = build.ocr.model_version
    return dict(settings)


def build_frozen(root: str | Path = ".", record: Path = FREEZE_RECORD) -> Frozen:
    """v1 in PyTorch on the CPU from the freeze record (`load_frozen`)."""
    return load_frozen("cpu", record, root=Path(root))
