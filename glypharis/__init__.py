"""GLYPHARIS: lightweight OCR engine for document images.

    from glypharis import GlypharisOCR
    result = GlypharisOCR.from_pretrained("glypharis-v1").predict("page.png")

``from_pretrained`` never downloads anything. ``download_model("glypharis-v1")`` fetches and
verifies a registered bundle into the models folder (`glypharis.hub`).

``GlypharisOCR``, ``OCRResult`` and ``download_model`` are imported on first use, so
``import glypharis`` stays light.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"
__all__ = ["GlypharisOCR", "OCRResult", "__version__", "download_model", "version_info"]


def __getattr__(name: str) -> Any:
    if name in ("GlypharisOCR", "OCRResult"):
        from glypharis.pipeline import ocr

        return getattr(ocr, name)
    if name == "download_model":
        from glypharis import hub

        return hub.download_model
    raise AttributeError(f"module 'glypharis' has no attribute {name!r}")


def version_info() -> dict[str, str]:
    """Versions of GLYPHARIS and of the libraries that decide its results, for bug reports and
    result metadata. A library that is not installed is reported as ``not installed``."""
    import importlib.metadata
    import platform

    out = {"glypharis": __version__, "python": platform.python_version()}
    for name in ("numpy", "torch", "onnxruntime", "opencv-contrib-python-headless", "pillow"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out
