"""The phone candidate's bundle, ``glypharis-phone-bundle-1`` (Phase 22, item 16, GW-EFF-014).

A new version, not v1. The directory holds ``detector.onnx`` (byte-identical to the v1 bundle's),
the selected arm's ONNX recognizer, ``charset.json`` and ``manifest.json``: the SHA-256 of every
file and every setting the loader needs (the detector spec; the recognizer's input height, maximum
width, stride, class count, architecture and temperature; the crop settings, reading order and
pixel cap; the ONNX opset, the ONNX Runtime version and session options the arm was measured with;
the batching rule; the v1 manifest's hash) and a version label that says it is not v1.

`load_phone_bundle` builds the pipeline from those files alone, on ONNX Runtime
(`threads.ort_session`), without any ``.pt`` file, so it cannot reuse `load_bundle_onnx`.
`glypharis.export.bundle.load_bundle` refuses this format, because its manifest is not
``glypharis-bundle-1``; ``from_pretrained`` therefore never loads it.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from glypharis.data.crops import CropSettings
from glypharis.data.download import sha256_file
from glypharis.data.text import Charset
from glypharis.efficiency.candidate import BATCHING, SingleLineRecognizer, check_arm
from glypharis.efficiency.onnx_build import SessionLineModel, onnx_detector
from glypharis.efficiency.threads import ort_session
from glypharis.export.bundle import read_manifest
from glypharis.models.det.detector import DetectorSpec
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer

PHONE_BUNDLE_FORMAT: Final = "glypharis-phone-bundle-1"
VERSION: Final = "glypharis_phone_candidate"
VERSION_LABEL: Final = "phone candidate (new version, not v1)"
DETECTOR_FILE: Final = "detector.onnx"
CHARSET_FILE: Final = "charset.json"
MANIFEST: Final = "manifest.json"
SESSION_OPTIONS: Final[Mapping[str, Any]] = {
    "providers": ["CPUExecutionProvider"],
    "execution_mode": "ORT_SEQUENTIAL",
    "graph_optimization_level": "ORT_ENABLE_ALL",
    "enable_cpu_mem_arena": False,
    "enable_mem_pattern": False,
    "allow_spinning": "default",
    "intra_op_num_threads": "the caller's thread count",
}


class PhoneBundleError(RuntimeError):
    """The directory is not a complete, unchanged phone bundle, or cannot be built."""


def _plain(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def write_phone_bundle(
    out: str | Path,
    v1_bundle: str | Path,
    v1_ocr: GlypharisOCR,
    arm: str,
    recognizer: str | Path,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the bundle of ``arm`` to ``out`` (refused if it exists) from the v1 bundle and the
    arm's ONNX recognizer, with the settings of ``v1_ocr`` (the v1 bundle loaded on ONNX Runtime).
    ``extra`` is recorded under ``sources``. Returns the manifest."""
    check_arm(arm)
    out, v1_bundle, recognizer = Path(out), Path(v1_bundle), Path(recognizer)
    if out.exists():
        raise PhoneBundleError(f"{out.as_posix()} exists; nothing is overwritten")
    v1 = read_manifest(v1_bundle)
    rec = v1_ocr.recognizer
    if dataclasses.asdict(v1_ocr.detector.spec) != v1["detector"]["spec"]:
        raise PhoneBundleError("the v1 pipeline does not hold the v1 bundle's detector spec")
    out.mkdir(parents=True)
    try:
        shutil.copyfile(v1_bundle / DETECTOR_FILE, out / DETECTOR_FILE)
        shutil.copyfile(recognizer, out / recognizer.name)
        (out / CHARSET_FILE).write_text(
            json.dumps(rec.charset.characters, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",  # the same bytes on every system
        )
        files = {n: sha256_file(out / n) for n in (DETECTOR_FILE, recognizer.name, CHARSET_FILE)}
        if files[DETECTOR_FILE] != v1["files"][DETECTOR_FILE]:
            raise PhoneBundleError("detector.onnx is not byte-identical to the v1 bundle's")
        import onnxruntime

        manifest = {
            "format": PHONE_BUNDLE_FORMAT,
            "version": VERSION,
            "version_label": VERSION_LABEL,
            "arm": arm,
            "batching": BATCHING[arm],
            "files": files,
            "recognizer_file": recognizer.name,
            "detector": {"spec": v1["detector"]["spec"]},
            "recognizer": {
                "height": rec.height,
                "max_width": rec.max_width,
                "stride": int(rec.model.stride),
                "classes": int(rec.model.num_classes),
                "arch": _plain(rec.arch),
                "temperature": rec.temperature,
            },
            "pipeline": {
                "crop": dataclasses.asdict(v1_ocr.crop),
                "reading_order": v1_ocr.order,
                "max_pixels": v1_ocr.max_pixels,
            },
            "onnx_opset": v1["onnx_opset"],
            "onnxruntime": onnxruntime.__version__,
            "session_options": dict(SESSION_OPTIONS),
            "v1": {
                "manifest_sha256": sha256_file(v1_bundle / MANIFEST),
                "version": v1["version"],
                "note": "the detector is v1's; the recognizer and its batching are not",
            },
            "sources": _plain(dict(extra or {})),
        }
        (out / MANIFEST).write_text(
            json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)  # only the directory this call created
        raise
    return manifest


def read_phone_manifest(bundle: str | Path) -> dict[str, Any]:
    """The manifest, after checking the format and that every listed file is there and
    unchanged."""
    path = Path(bundle)
    try:
        manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhoneBundleError(f"{path}: no readable {MANIFEST} ({exc})") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != PHONE_BUNDLE_FORMAT:
        raise PhoneBundleError(f"{path} is not a {PHONE_BUNDLE_FORMAT} directory")
    files = manifest.get("files") or {}
    want = {DETECTOR_FILE, CHARSET_FILE, str(manifest.get("recognizer_file"))}
    if set(files) != want:
        raise PhoneBundleError(f"{path}: the manifest lists {sorted(files)}, not {sorted(want)}")
    for name, sha in files.items():
        got = sha256_file(path / name) if (path / name).is_file() else None
        if got != sha:
            raise PhoneBundleError(f"{path / name}: missing or not the file the manifest names")
    check_arm(str(manifest.get("arm")))
    if manifest.get("batching") != BATCHING[manifest["arm"]]:
        raise PhoneBundleError(f"{path}: the batching rule is not {manifest['arm']}'s")
    return dict(manifest)


def load_phone_bundle(bundle: str | Path, threads: int | None) -> GlypharisOCR:
    """The candidate pipeline from a phone bundle, on ONNX Runtime with ``threads`` intra-op
    threads, the recognizer read under the bundle's batching rule."""
    path = Path(bundle)
    manifest = read_phone_manifest(path)
    try:
        spec = DetectorSpec.from_dict(manifest["detector"]["spec"])
        detector = onnx_detector(path / DETECTOR_FILE, spec, threads)
        r = manifest["recognizer"]
        charset = Charset(json.loads((path / CHARSET_FILE).read_text(encoding="utf-8")))
        if len(charset.characters) + 1 != r["classes"]:
            raise PhoneBundleError(f"{path}: the charset does not fit {r['classes']} classes")
        model = SessionLineModel(
            ort_session(path / manifest["recognizer_file"], threads), r["stride"], r["classes"]
        )
        kind = SingleLineRecognizer if manifest["arm"] == "fused-single-line" else TextRecognizer
        recognizer = kind(
            model, charset, r["height"], r["max_width"], "cpu", r["temperature"], r["arch"]
        )
        p = manifest["pipeline"]
        ocr = GlypharisOCR(
            detector, recognizer, CropSettings(**p["crop"]), p["reading_order"], p["max_pixels"]
        )
    except PhoneBundleError:
        raise
    except Exception as exc:  # a damaged file that still matches its hash, a runtime mismatch
        raise PhoneBundleError(f"{path}: cannot load ({type(exc).__name__}: {exc})") from exc
    if dataclasses.asdict(ocr.detector.spec) != manifest["detector"]["spec"]:
        raise PhoneBundleError(f"{path}: the detector spec did not survive loading")
    ocr.model_version = f"{VERSION} {manifest['arm']} ({VERSION_LABEL})"
    return ocr
