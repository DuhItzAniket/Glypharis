"""GLYPHARIS: lightweight OCR engine for document images."""

from __future__ import annotations

__version__ = "0.1.0.dev0"


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
