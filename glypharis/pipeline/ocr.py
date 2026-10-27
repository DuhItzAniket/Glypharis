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
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from glypharis.data.crops import CropSettings, background_grey, cut_line
from glypharis.errors import GlypharisError, ImageTooLargeError, InvalidImageError
from glypharis.models.det.detector import TextLineDetector
from glypharis.pipeline.reading_order import reading_order
from glypharis.pipeline.recognizer import LineReading, TextRecognizer

ImageInput = str | Path | Image.Image | np.ndarray
DEFAULT_MAX_PIXELS = 64_000_000  # about 8000 x 8000; a page scanned at 600 dpi is about 35 M
log = logging.getLogger(__name__)
_BUNDLE_CACHE: dict[tuple[str, str, int | None], GlypharisOCR] = {}


@dataclass
class OcrLine:
    polygon: list[list[float]]
    quad: list[list[float]]  # TL, TR, BR, BL in reading direction
    text: str
    confidence: float  # recognition confidence at the recognizer's temperature
    det_score: float
    conf_at: dict[float, float] = field(default_factory=dict)  # at other temperatures, if asked

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """x0, y0, x1, y1 of the polygon, in page pixels."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

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
                "bbox": list(ln.bbox),
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


def _check_size(w: int, h: int, max_pixels: int) -> None:
    if w < 1 or h < 1:
        raise InvalidImageError(f"the image is empty ({w}x{h} pixels)")
    if w * h > max_pixels:
        raise ImageTooLargeError(
            f"the image has {w}x{h} = {w * h:,} pixels, more than the limit of {max_pixels:,} "
            "(GlypharisOCR(max_pixels=...)); scale it down first"
        )


def _to_uint8(a: np.ndarray) -> np.ndarray:
    """Pixel values as uint8: uint8 as is, bool as 0/255, uint16 to its high byte, floats in [0, 1]
    or [0, 255] scaled; anything else is refused."""
    if a.dtype == np.uint8:
        return a
    if a.dtype == np.bool_:
        return a.astype(np.uint8) * 255
    if a.dtype == np.uint16:
        return (a >> 8).astype(np.uint8)
    if np.issubdtype(a.dtype, np.floating):
        if not np.isfinite(a).all():
            raise InvalidImageError("the array holds NaN or infinite values")
        hi = float(a.max())
        if float(a.min()) < 0 or hi > 255:
            raise InvalidImageError("float pixel values must lie in [0, 1] or [0, 255]")
        return np.round(a * 255 if hi <= 1.0 else a).astype(np.uint8)
    if np.issubdtype(a.dtype, np.integer) and a.min() >= 0 and a.max() <= 255:
        return a.astype(np.uint8)
    raise InvalidImageError(f"unsupported pixel type {a.dtype}")


def _array_rgb(a: np.ndarray, max_pixels: int) -> np.ndarray:
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (3, 4)):
        raise InvalidImageError(
            f"an image array must be (H, W), (H, W, 1), (H, W, 3) or (H, W, 4); got {a.shape}"
        )
    _check_size(a.shape[1], a.shape[0], max_pixels)
    a = _to_uint8(a)
    if a.ndim == 2:
        return np.repeat(a[..., None], 3, axis=2)
    return np.ascontiguousarray(a[..., :3])


def _pil_rgb(img: Image.Image, max_pixels: int) -> np.ndarray:
    _check_size(img.width, img.height, max_pixels)
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        a = np.asarray(img).astype(np.float64)
        top = float(np.nanmax(a)) if a.size else 1.0
        a = np.nan_to_num(a / top if top > 1.0 else a, nan=0.0, posinf=1.0, neginf=0.0)
        return _array_rgb(np.clip(a, 0.0, 1.0), max_pixels)
    return np.asarray(img.convert("RGB"))


def load_rgb(image: ImageInput, max_pixels: int = DEFAULT_MAX_PIXELS) -> np.ndarray:
    """HWC uint8 RGB from a path, a PIL image or an array; a file is opened with its EXIF
    orientation applied, as the pages were. Anything that is not a readable image of at most
    ``max_pixels`` pixels raises `InvalidImageError` (or `ImageTooLargeError`)."""
    if isinstance(image, np.ndarray):
        return _array_rgb(image, max_pixels)
    if isinstance(image, Image.Image):
        return _pil_rgb(image, max_pixels)
    if not isinstance(image, str | Path):
        raise InvalidImageError(
            f"unsupported input type {type(image).__name__}: give a file path, a PIL image or a "
            "numpy array"
        )
    path = Path(image)
    if not path.is_file():
        raise InvalidImageError(f"{path}: no such file")
    try:
        with Image.open(path) as opened:
            _check_size(opened.width, opened.height, max_pixels)  # from the header, before decoding
            opened.load()
            return _pil_rgb(ImageOps.exif_transpose(opened), max_pixels)
    except Image.DecompressionBombError as exc:
        raise ImageTooLargeError(f"{path}: {exc}") from exc
    except GlypharisError:
        raise
    except (OSError, SyntaxError, ValueError) as exc:  # unidentified, truncated or corrupt
        raise InvalidImageError(f"{path}: not a readable image ({exc})") from exc


class GlypharisOCR:
    def __init__(
        self,
        detector: TextLineDetector,
        recognizer: TextRecognizer,
        crop: CropSettings | None = None,
        order: bool = True,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.crop = crop or CropSettings()
        self.order = order
        self.max_pixels = max_pixels
        self.model_version = ""
        self._counts = {"pages": 0, "lines": 0, "failed": 0}
        self._seconds = 0.0

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path = "glypharis-0.1.0",
        backend: str = "onnx",
        threads: int | None = None,
        cache: bool = True,
    ) -> GlypharisOCR:
        """A model bundle by path, or by name from the model directories
        (`glypharis.export.bundle.find_bundle`). Nothing is downloaded. ``backend`` is ``onnx``
        (the default, ONNX Runtime on the CPU) or ``torch``. With ``cache`` a bundle already
        loaded in this process with the same backend and threads is returned again. A missing or
        damaged bundle raises `glypharis.errors.ModelLoadError`."""
        from glypharis.export.bundle import find_bundle, load_bundle

        path = find_bundle(name_or_path)
        key = (str(path.resolve()), backend, threads)
        if cache and key in _BUNDLE_CACHE:
            return _BUNDLE_CACHE[key]
        t0 = time.perf_counter()
        ocr = load_bundle(path, backend, threads)  # type: ignore[arg-type]
        log.info(
            "loaded GLYPHARIS model %s (%s backend) from %s in %.2f s",
            ocr.model_version,
            backend,
            path,
            time.perf_counter() - t0,
        )
        if cache:
            _BUNDLE_CACHE[key] = ocr
        return ocr

    def stats(self) -> dict[str, float]:
        """Pages read, lines found, inputs refused and mean seconds per page so far."""
        n = self._counts["pages"]
        return {
            **self._counts,
            "seconds": self._seconds,
            "mean_page_seconds": self._seconds / n if n else 0.0,
        }

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
        try:
            rgb = load_rgb(image, self.max_pixels)
        except GlypharisError:
            self._counts["failed"] += 1
            raise
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
        self._counts["pages"] += 1
        self._counts["lines"] += len(lines)
        self._seconds += secs["total"]
        log.debug(
            "page %dx%d: %d lines in %.3f s", rgb.shape[1], rgb.shape[0], len(lines), secs["total"]
        )
        return OCRResult(rgb.shape[1], rgb.shape[0], lines, secs, self.model_version)

    def predict_batch(self, images: Iterable[ImageInput]) -> list[OCRResult]:
        """Pages one after another (each page already batches its lines)."""
        return [self.predict(im) for im in images]
