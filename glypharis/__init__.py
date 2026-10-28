"""GLYPHARIS: lightweight OCR engine for document images.

    from glypharis import GlypharisOCR
    result = GlypharisOCR.from_pretrained("glypharis-0.1.0").predict("page.png")

``GlypharisOCR`` and ``OCRResult`` are imported on first use, so ``import glypharis`` stays light.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0.dev0"
__all__ = ["GlypharisOCR", "OCRResult", "__version__"]


def __getattr__(name: str) -> Any:
    if name in ("GlypharisOCR", "OCRResult"):
        from glypharis.pipeline import ocr

        return getattr(ocr, name)
    raise AttributeError(f"module 'glypharis' has no attribute {name!r}")
