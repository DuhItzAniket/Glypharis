"""`GlypharisOCR`: a page image in, its text lines in reading order out (Phase 10).

    ocr = GlypharisOCR.from_pretrained("glypharis-0.1.0")   # a model bundle (Phase 17/18)
    result = ocr.predict(image)   # path, PIL image or HWC uint8 RGB array -> OCRResult
    result.text                   # the lines joined with newlines, in reading order
    result.to_json()              # the documented JSON (OCRResult.to_dict)

Steps: detect lines (`TextLineDetector`, its spec's resolution and post-processing); cut each line
out of the grey page along its quad and straighten it exactly as the training crops were made
(`glypharis.data.crops.cut_line`, 64 px high with a 15% margin, background-grey fill); read the
crops in width-sorted batches (`TextRecognizer`); order the lines (`reading_order`). A line keeps
its detection score and its recognition confidence apart; ``confidence`` is the recognizer's.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from glypharis.data.crops import CropSettings, background_grey, cut_line
from glypharis.models.det.detector import TextLineDetector
from glypharis.pipeline.reading_order import reading_order
from glypharis.pipeline.recognizer import LineReading, TextRecognizer

ImageInput = str | Path | Image.Image | np.ndarray


@dataclass
class OcrLine:
    polygon: list[list[float]]
    quad: list[list[float]]  # TL, TR, BR, BL in reading direction
    text: str
    confidence: float  # recognition confidence at the recognizer's temperature
    det_score: float
    conf_at: dict[float, float] = field(default_factory=dict)  # at other temperatures, if asked

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRResult:
    """One page's result. JSON (`to_dict`): ``glypharis_version``, ``model_version``, ``width``,
    ``height``, ``text`` (lines joined with newlines, in reading order), ``lines`` (each with
    ``text``, ``confidence``, ``det_score``, ``polygon`` and ``quad`` in page pixels, TL, TR, BR, BL
    along the reading direction) and ``seconds`` (time per step)."""

    width: int
    height: int
    lines: list[OcrLine] = field(default_factory=list)  # in reading order
    seconds: dict[str, float] = field(default_factory=dict)
    model_version: str = ""

    @property
    def text(self) -> str:
        return "\n".join(ln.text for ln in self.lines if ln.text)

    def to_dict(self) -> dict[str, Any]:
        from glypharis import __version__

        lines = []
        for ln in self.lines:
            d: dict[str, Any] = {
                "text": ln.text,
                "confidence": ln.confidence,
                "det_score": ln.det_score,
                "polygon": ln.polygon,
                "quad": ln.quad,
            }
            if ln.conf_at:
                d["confidence_at"] = {str(t): c for t, c in ln.conf_at.items()}
            lines.append(d)
        return {
            "glypharis_version": __version__,
            "model_version": self.model_version,
            "width": self.width,
            "height": self.height,
            "text": self.text,
            "lines": lines,
            "seconds": self.seconds,
        }

    def to_json(self, indent: int | None = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


OcrPage = OCRResult  # the Phase 10 name


def load_rgb(image: ImageInput) -> np.ndarray:
    """HWC uint8 RGB; a file is opened with its EXIF orientation applied, as the pages were."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return np.repeat(image[..., None], 3, axis=2).astype(np.uint8)
        return np.ascontiguousarray(image[..., :3]).astype(np.uint8)
    img = image if isinstance(image, Image.Image) else ImageOps.exif_transpose(Image.open(image))
    return np.asarray(img.convert("RGB"))


class GlypharisOCR:
    def __init__(
        self,
        detector: TextLineDetector,
        recognizer: TextRecognizer,
        crop: CropSettings | None = None,
        order: bool = True,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.crop = crop or CropSettings()
        self.order = order
        self.model_version = ""

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path = "glypharis-0.1.0",
        backend: str = "onnx",
        threads: int | None = None,
    ) -> GlypharisOCR:
        """A model bundle by path, or by name from the model directories
        (`glypharis.export.bundle.find_bundle`). Nothing is downloaded. ``backend`` is ``onnx``
        (the default, ONNX Runtime on the CPU) or ``torch``."""
        from glypharis.export.bundle import find_bundle, load_bundle

        return load_bundle(find_bundle(name_or_path), backend, threads)  # type: ignore[arg-type]

    def read_quads(
        self,
        rgb: np.ndarray,
        quads: Sequence[Sequence[Sequence[float]]],
        temperatures: Sequence[float] = (),
    ) -> list[LineReading]:
        """Readings of given line outlines on a page: detector quads, or ground-truth polygons for
        the oracle analysis (a polygon is cut as the training crops were)."""
        gray = Image.fromarray(rgb).convert("L")
        fill = background_grey(gray)
        crops = [cut_line(gray, [list(p) for p in q], self.crop, fill) for q in quads]
        return self.recognizer.read(crops, temperatures)

    def predict(self, image: ImageInput, temperatures: Sequence[float] = ()) -> OCRResult:
        t0 = time.perf_counter()
        rgb = load_rgb(image)
        t1 = time.perf_counter()
        boxes = self.detector.detect(rgb)
        t2 = time.perf_counter()
        reads = self.read_quads(rgb, [b.quad for b in boxes], temperatures)
        t3 = time.perf_counter()
        order = reading_order([b.quad for b in boxes]) if self.order else list(range(len(boxes)))
        lines = [
            OcrLine(
                boxes[i].polygon,
                boxes[i].quad,
                reads[i].text,
                reads[i].confidence,
                boxes[i].score,
                reads[i].at,
            )
            for i in order
        ]
        t4 = time.perf_counter()
        secs = {
            "load": t1 - t0,
            "detect": t2 - t1,
            "recognise": t3 - t2,
            "order": t4 - t3,
            "total": t4 - t0,
        }
        return OCRResult(rgb.shape[1], rgb.shape[0], lines, secs, self.model_version)

    def predict_batch(self, images: Iterable[ImageInput]) -> list[OCRResult]:
        """Pages one after another (each page already batches its lines)."""
        return [self.predict(im) for im in images]
