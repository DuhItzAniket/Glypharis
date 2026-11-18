"""v1's pipeline objects on ONNX Runtime sessions made by `threads.ort_session`.

`glypharis.export.onnx.OnnxProbMap` and `OnnxLineModel` build their own sessions (default
sequential execution, threads optional) and `glypharis.export.bundle.load_bundle` uses them. Phase
16 needs the session options fixed and recorded (`threads.ort_session`, `threads.session_options`)
without editing those modules, so `SessionProbMap` and `SessionLineModel` subclass them and take a
ready session instead; the image checks, the call and the frame count are the v1 classes' own.

    ocr = load_bundle_onnx("runs/release/glypharis-v1", threads=10)
    profile["session_options"] = check_sessions(ocr, 10)

`load_bundle_onnx` is `load_bundle(..., backend="onnx")` with those sessions: the manifest's hashes
are checked before anything loads and the detector release must hold the manifest's spec.
`onnx_detector`, `onnx_recognizer` and `with_parts` build mixed pipelines (the ONNX detector with
the PyTorch recognizer, another recognizer file) from v1's objects, settings unchanged.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from glypharis.efficiency.threads import check_session, ort_session, session_options
from glypharis.errors import ModelLoadError
from glypharis.export.bundle import read_manifest
from glypharis.export.onnx import OnnxLineModel, OnnxProbMap
from glypharis.models.det.detector import DetectorSpec, TextLineDetector
from glypharis.models.rec.base import LineRecognizer
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer


class SessionProbMap(OnnxProbMap):
    """`OnnxProbMap` over a session made elsewhere (`threads.ort_session`)."""

    def __init__(self, session: Any) -> None:  # OnnxProbMap.__init__ would build its own session
        self.session = session


class SessionLineModel(OnnxLineModel):
    """`OnnxLineModel` over a session made elsewhere (`threads.ort_session`)."""

    def __init__(self, session: Any, stride: int, num_classes: int) -> None:
        LineRecognizer.__init__(self)  # skips OnnxLineModel.__init__, which builds a session
        self.session = session
        self.stride = stride
        self.num_classes = num_classes


def onnx_detector(path: str | Path, spec: DetectorSpec, threads: int | None) -> TextLineDetector:
    """A detector on the exported model at ``path`` with ``spec`` (resolution, post-processing,
    operating point) unchanged."""
    return TextLineDetector(SessionProbMap(ort_session(path, threads)), spec)


def onnx_recognizer(path: str | Path, like: TextRecognizer, threads: int | None) -> TextRecognizer:
    """A CPU recognizer on the exported model at ``path``, with ``like``'s charset, input height,
    maximum width, temperature, architecture record and strict-fp32 flag; ``like.model`` gives the
    stride and class count, so ``path`` must be an export of a model with the same ones."""
    model = SessionLineModel(ort_session(path, threads), like.model.stride, like.model.num_classes)
    return TextRecognizer(
        model,
        like.charset,
        like.height,
        like.max_width,
        "cpu",
        like.temperature,
        like.arch,
        strict_fp32=like.strict_fp32,
    )


def with_parts(
    ocr: GlypharisOCR,
    detector: TextLineDetector | None = None,
    recognizer: TextRecognizer | None = None,
) -> GlypharisOCR:
    """A new `GlypharisOCR` with ``ocr``'s crop settings, reading order, pixel cap and model
    version, and its detector or recognizer replaced where given."""
    out = GlypharisOCR(
        detector or ocr.detector,
        recognizer or ocr.recognizer,
        ocr.crop,  # a frozen dataclass
        ocr.order,
        ocr.max_pixels,
    )
    out.model_version = ocr.model_version
    return out


def load_bundle_onnx(
    bundle: str | Path, threads: int | None, temperature: float | None = None
) -> GlypharisOCR:
    """`glypharis.export.bundle.load_bundle(bundle, "onnx", threads, temperature)` with both
    sessions from `threads.ort_session`; a missing, damaged or mismatched bundle raises
    `ModelLoadError`."""
    path = Path(bundle)
    manifest = read_manifest(path)
    try:
        rec = TextRecognizer.from_release(path / "recognizer.pt", "cpu", temperature)
        det = TextLineDetector.from_release(path / "detector.pt")
        if dataclasses.asdict(det.spec) != manifest["detector"]["spec"]:
            raise ModelLoadError(f"{path}: the detector release does not hold the manifest's spec")
        ocr = GlypharisOCR(
            onnx_detector(path / "detector.onnx", det.spec, threads),
            onnx_recognizer(path / "recognizer.onnx", rec, threads),
        )
    except ModelLoadError:
        raise
    except Exception as exc:  # a damaged file that still matches its hash, a runtime mismatch
        raise ModelLoadError(
            f"{path}: cannot load the bundle ({type(exc).__name__}: {exc})"
        ) from exc
    ocr.model_version = str(manifest["version"])
    return ocr


def _sessions(ocr: GlypharisOCR) -> dict[str, Any]:
    backend, model = ocr.detector.backend, ocr.recognizer.model
    return {
        "detector": backend.session if isinstance(backend, OnnxProbMap) else None,
        "recognizer": model.session if isinstance(model, OnnxLineModel) else None,
    }


def sessions_options(ocr: GlypharisOCR) -> dict[str, Any]:
    """`threads.session_options` of the detector's and the recognizer's ONNX sessions (None for a
    part that runs in PyTorch or on another backend)."""
    return {k: None if s is None else session_options(s) for k, s in _sessions(ocr).items()}


def check_sessions(ocr: GlypharisOCR, threads: int | None) -> dict[str, Any]:
    """`threads.check_session` on every ONNX part of ``ocr`` (`threads.ThreadError` on a
    difference); returns `sessions_options`."""
    return {k: None if s is None else check_session(s, threads) for k, s in _sessions(ocr).items()}
