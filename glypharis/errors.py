"""The errors GLYPHARIS raises at its public edge (Phase 20).

Every failure a caller can cause or meet (a bad input, a missing or damaged model) is a
`GlypharisError` subclass with a stable ``code`` and a message that says what to do; anything else
reaching the caller is a bug. ``to_dict`` gives the structured form the CLI and services report.
"""

from __future__ import annotations

from typing import Any


class GlypharisError(Exception):
    code = "glypharis_error"

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self)}


class InvalidImageError(GlypharisError, ValueError):
    """The input is not an image GLYPHARIS can read: wrong type or shape, empty, or undecodable."""

    code = "invalid_image"


class ImageTooLargeError(InvalidImageError):
    """More pixels than the configured limit (``max_pixels``), or a decompression bomb."""

    code = "image_too_large"


class ModelLoadError(GlypharisError, RuntimeError):
    """A model bundle that is missing, incomplete, damaged or of an unknown format."""

    code = "model_load_error"
