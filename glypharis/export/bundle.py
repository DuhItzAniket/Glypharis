"""The GLYPHARIS model bundle: one directory with everything inference needs (Phase 17).

    write_bundle(det_checkpoint, det_spec, recognizer, out_dir)   # once, from trained runs
    ocr = load_bundle(out_dir, backend="onnx")                     # GlypharisOCR

Files: ``detector.pt`` and ``recognizer.pt`` (release files, readable with
``torch.load(..., weights_only=True)``), ``detector.onnx`` and ``recognizer.onnx``, and
``manifest.json`` with the SHA-256 of each, the detector spec (resolution, post-processing,
operating point), the recognizer's input settings and temperature, and where they came from.
`load_bundle` checks every hash before it loads anything. The ONNX backend is the one ADR-0011's
latency target is judged on; the PyTorch backend is the reference it is checked against.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Final, Literal

import torch

from glypharis.data.download import sha256_file
from glypharis.export.onnx import (
    OPSET,
    OnnxLineModel,
    OnnxProbMap,
    export_detector_onnx,
    export_recognizer_onnx,
)
from glypharis.models.det.detector import DetectorSpec, TextLineDetector, export_detector
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer

BUNDLE_FORMAT: Final = "glypharis-bundle-1"
FILES: Final = ("detector.pt", "detector.onnx", "recognizer.pt", "recognizer.onnx")
Backend = Literal["onnx", "torch"]


def write_bundle(
    det_checkpoint: str | Path,
    det_spec: DetectorSpec,
    recognizer: TextRecognizer,
    out_dir: str | Path,
    version: str,
    sources: dict[str, Any] | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    export_detector(Path(det_checkpoint), out / "detector.pt", det_spec)
    det = TextLineDetector.from_release(out / "detector.pt")
    export_detector_onnx(det.backend.model, out / "detector.onnx")  # type: ignore[attr-defined]
    torch.save(recognizer.release(), out / "recognizer.pt")
    export_recognizer_onnx(recognizer.model, recognizer.height, out / "recognizer.onnx")
    manifest = {
        "format": BUNDLE_FORMAT,
        "version": version,
        "files": {f: sha256_file(out / f) for f in FILES},
        "detector": {"spec": det_spec.to_dict()},
        "recognizer": {
            "height": recognizer.height,
            "max_width": recognizer.max_width,
            "temperature": recognizer.temperature,
            "arch": recognizer.arch,
            "classes": recognizer.charset.num_classes,
        },
        "onnx_opset": OPSET,
        "torch": torch.__version__,
        "sources": sources or {},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return out


MODEL_DIRS_ENV: Final = "GLYPHARIS_MODELS"


def model_dirs() -> list[Path]:
    """Where bundles are looked up by name: ``$GLYPHARIS_MODELS`` (``os.pathsep``-separated),
    ``~/.glypharis/models`` and ``./models``."""
    dirs = [Path(d) for d in os.environ.get(MODEL_DIRS_ENV, "").split(os.pathsep) if d]
    return [*dirs, Path.home() / ".glypharis" / "models", Path("models")]


def find_bundle(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if (path / "manifest.json").is_file():
        return path
    for d in model_dirs():
        if (d / str(name_or_path) / "manifest.json").is_file():
            return d / str(name_or_path)
    looked = ", ".join(str(d) for d in model_dirs())
    raise FileNotFoundError(
        f"no GLYPHARIS model bundle {str(name_or_path)!r} (a directory with manifest.json); "
        f"looked in: {looked}"
    )


def read_manifest(bundle: str | Path) -> dict[str, Any]:
    path = Path(bundle)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"{path} is not a {BUNDLE_FORMAT} directory")
    for name, want in manifest["files"].items():
        got = sha256_file(path / name)
        if got != want:
            raise ValueError(f"{path / name}: SHA-256 {got[:12]} does not match the manifest")
    return dict(manifest)


def load_bundle(
    bundle: str | Path,
    backend: Backend = "onnx",
    threads: int | None = None,
    temperature: float | None = None,
) -> GlypharisOCR:
    """`GlypharisOCR` from a bundle directory, on the CPU, with the chosen backend."""
    path = Path(bundle)
    manifest = read_manifest(path)
    rec = TextRecognizer.from_release(path / "recognizer.pt", "cpu", temperature)
    det = TextLineDetector.from_release(path / "detector.pt")
    if backend == "onnx":
        det = TextLineDetector(OnnxProbMap(path / "detector.onnx", threads), det.spec)
        onnx_model = OnnxLineModel(
            path / "recognizer.onnx", rec.model.stride, rec.model.num_classes, threads
        )
        rec = TextRecognizer(
            onnx_model,
            rec.charset,
            rec.height,
            rec.max_width,
            "cpu",
            rec.temperature,
            rec.arch,
        )
    elif backend != "torch":
        raise ValueError(f"backend {backend!r}: onnx or torch")
    elif threads:
        torch.set_num_threads(threads)
    if dataclasses.asdict(det.spec) != manifest["detector"]["spec"]:
        raise ValueError("the detector release does not hold the manifest's spec")
    ocr = GlypharisOCR(det, rec)
    ocr.model_version = str(manifest["version"])
    return ocr
