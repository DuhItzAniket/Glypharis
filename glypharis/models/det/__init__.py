"""GLYPHARIS-Det text-line detector.

Re-exports load lazily: ``postprocess`` and ``preprocess`` are torch-free, and importing them (which
runs this file first) must not pull in torch and torchvision through the model modules.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from glypharis.models.det.dbnet import DBNet, build_detector
    from glypharis.models.det.detector import TextLineDetector
    from glypharis.models.det.postprocess import DetBox
    from glypharis.models.det.preprocess import prepare_page

_EXPORTS = {
    "DBNet": "glypharis.models.det.dbnet",
    "build_detector": "glypharis.models.det.dbnet",
    "TextLineDetector": "glypharis.models.det.detector",
    "DetBox": "glypharis.models.det.postprocess",
    "prepare_page": "glypharis.models.det.preprocess",
}

__all__ = ["DBNet", "DetBox", "TextLineDetector", "build_detector", "prepare_page"]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
