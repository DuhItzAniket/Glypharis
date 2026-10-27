"""Photometric augmentation of detector training crops (§4.5).

It runs after the geometric warp, on the RGB crop, identically in every GW-DET arm. Geometry
(scale, rotation, crop placement) lives in `det_geometry.sample_view`. Every random draw comes from
the sample's own generator, so a crop is a pure function of (seed, epoch, item) whatever worker
makes it. Each step draws its on/off decision even when its probability is 0, so the draw order
never depends on the configuration. Augmentation is not a Phase 8 study variable (Phase 11 owns
it), so the recipe is deliberately plain.
"""

from __future__ import annotations

import cv2
import numpy as np

from glypharis.config.schema import DetAugmentConfig

# ITU-R BT.601 luma, the weights cv2.COLOR_RGB2GRAY uses.
_LUMA = np.array([0.299, 0.587, 0.114])


def _colour_jitter(
    img: np.ndarray, brightness: float, contrast: float, saturation: float
) -> np.ndarray:
    """Brightness, then contrast about the mean luma, then saturation about each pixel's luma.

    All three are linear in the pixel values, so they are applied as one 3x4 colour matrix and
    clipped once (`cv2.transform` rounds and saturates); per-step float passes cost 40 times more.
    """
    mean = brightness * float(_LUMA @ np.array(cv2.mean(img)[:3]))
    sat = saturation * np.eye(3) + (1.0 - saturation) * np.outer(np.ones(3), _LUMA)
    m = np.empty((3, 4))
    m[:, :3] = sat * (contrast * brightness)
    m[:, 3] = sat @ np.full(3, (1.0 - contrast) * mean)
    return cv2.transform(img, m)


def _jpeg_round_trip(img: np.ndarray, quality: int) -> np.ndarray:
    # The codec's colour transform assumes BGR input, so convert around it.
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else None
    if decoded is None:
        raise RuntimeError("JPEG round trip failed")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


class PhotometricAugmenter:
    """Colour jitter, grayscale, Gaussian blur, luminance noise and a JPEG round trip, in that
    order, each with its own probability."""

    def __init__(self, cfg: DetAugmentConfig) -> None:
        self.cfg = cfg

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """(S, S, 3) uint8 RGB -> the same shape and type."""
        c = self.cfg
        out = img
        if rng.random() < c.color_p:
            b, k, s = rng.uniform(1.0 - c.color_strength, 1.0 + c.color_strength, size=3)
            out = _colour_jitter(out, float(b), float(k), float(s))
        if rng.random() < c.grayscale_p:
            out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        if rng.random() < c.blur_p:
            sigma = float(rng.uniform(0.3, c.blur_sigma_max))
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if rng.random() < c.noise_p:
            sigma = float(rng.uniform(2.0, c.noise_sigma_max))
            noise = rng.standard_normal(out.shape[:2], dtype=np.float32)
            noise *= sigma
            # The same noise on every channel; cv2.add rounds and saturates to uint8.
            out = cv2.add(out, cv2.merge([noise, noise, noise]), dtype=cv2.CV_8U)
        if rng.random() < c.jpeg_p:
            out = _jpeg_round_trip(out, int(rng.integers(c.jpeg_quality_min, 96)))
        return np.ascontiguousarray(out)
