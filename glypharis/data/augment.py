"""Line-image augmentation for the recognizer, and the fixed corruptions behind GLYPH-C.

Every transform takes a grayscale ``PIL.Image`` (mode "L"), a ``numpy.random.Generator`` and a
severity in [0, 1], and returns a new image of the same height. All randomness comes from the
generator passed in, so a seeded worker reproduces its augmentations exactly.

Presets (``data.augmentation`` in the run config) group transforms into the families studied in
Phase 11: geometric, photometric, document degradation, and all of them together. Each entry has a
probability ``p`` and a ``severity``; ``data.augmentation_params`` can override either per
transform.
"""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageFilter

Transform = Callable[[Image.Image, np.random.Generator, float], Image.Image]


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


def _img(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), mode="L")


def _fill(img: Image.Image) -> int:
    """Border value for geometric transforms: the brighter of the image's two edge medians."""
    a = np.asarray(img)
    return int(max(np.median(a[:, :2]), np.median(a[:, -2:])))


def _same_height(img: Image.Image, h: int) -> Image.Image:
    if img.height == h:
        return img
    w = max(1, round(img.width * h / img.height))
    return img.resize((w, h), Image.Resampling.BILINEAR)


# ------------------------------------------------------------------------------- geometric


def _rotate_whole(img: Image.Image, angle: float) -> Image.Image:
    """Rotate onto an enlarged canvas so no text is cut off, then restore the line height."""
    out = img.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=_fill(img))
    return _same_height(out, img.height)


def _outward_quad(w: int, h: int, d: float, rng: np.random.Generator) -> tuple[float, ...]:
    """Source quad for a perspective warp whose corners only move outward by up to ``d``.

    The QUAD transform samples this quad into the output rectangle, so outward corners shrink and
    warp the text without cropping any of it.
    """
    m = rng.uniform(0, d, size=8)
    return (
        -m[0], -m[1],  # upper-left
        -m[2], h + m[3],  # lower-left
        w + m[4], h + m[5],  # lower-right
        w + m[6], -m[7],  # upper-right
    )  # fmt: skip


def rotate(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Small in-plane rotation, up to ±4° at full severity, without cropping the text."""
    return _rotate_whole(img, rng.uniform(-4, 4) * s)


def shear(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Horizontal shear (italic-like slant), up to ±0.35."""
    k = rng.uniform(-0.35, 0.35) * s
    w, h = img.size
    pad = int(abs(k) * h)
    out_w = w + pad
    offset = -pad if k > 0 else 0
    return img.transform(
        (out_w, h), Image.Transform.AFFINE, (1, k, offset, 0, 1, 0),
        resample=Image.Resampling.BILINEAR, fillcolor=_fill(img),
    )  # fmt: skip


def stretch(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Width scaled by a factor in [1 - 0.3s, 1 + 0.3s] (condensed or wide fonts)."""
    f = 1 + rng.uniform(-0.3, 0.3) * s
    return img.resize((max(1, round(img.width * f)), img.height), Image.Resampling.BILINEAR)


def perspective(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Corner jitter up to 12% of the height: mild camera tilt."""
    w, h = img.size
    quad = _outward_quad(w, h, 0.12 * h * s, rng)
    return img.transform(
        (w, h), Image.Transform.QUAD, quad, resample=Image.Resampling.BILINEAR, fillcolor=_fill(img)
    )


def margin_jitter(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Trim up to half of the stored context margin on each side (imprecise detector boxes)."""
    w, h = img.size
    m = 0.15 / 1.3 * h  # stored margin in pixels (CropSettings.margin of the unexpanded height)
    cut = rng.uniform(0, 0.5 * m * s, size=4)
    left, top = int(cut[0]), int(cut[1])
    right, bottom = w - int(cut[2]), h - int(cut[3])
    if right - left < 4 or bottom - top < 4:
        return img
    return _same_height(img.crop((left, top, right, bottom)), h)


# ----------------------------------------------------------------------------- photometric


def brightness_contrast(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    a = _arr(img)
    mean = a.mean()
    contrast = 1 + rng.uniform(-0.5, 0.5) * s
    shift = rng.uniform(-50, 50) * s
    return _img((a - mean) * contrast + mean + shift)


def gamma(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    g = math.exp(rng.uniform(-0.6, 0.6) * s)
    return _img(255.0 * (_arr(img) / 255.0) ** g)


def invert(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Light text on a dark background. Severity is not used; the preset probability decides."""
    return _img(255.0 - _arr(img))


def gaussian_blur(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    radius = rng.uniform(0.2, 0.2 + 1.8 * s) * img.height / 64
    return img.filter(ImageFilter.GaussianBlur(radius))


def gaussian_noise(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    sigma = rng.uniform(2, 2 + 25 * s)
    a = _arr(img)
    return _img(a + rng.normal(0, sigma, size=a.shape))


def jpeg(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    quality = round(95 - rng.uniform(0, 1) * 85 * s)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(5, quality))
    out = Image.open(io.BytesIO(buf.getvalue()))
    out.load()
    return out.convert("L")


# ---------------------------------------------------------------------- document degradation


def low_resolution(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Downscale (to as little as a quarter of the height) and scale back up."""
    f = 1 - rng.uniform(0, 0.75) * s
    w, h = img.size
    small = img.resize((max(1, round(w * f)), max(4, round(h * f))), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


def ink_spread(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Thicker (bleeding) or thinner (faded) strokes via a min or max filter."""
    if s < 0.3 or img.height < 24:
        return img
    if rng.random() < 0.5:
        return img.filter(ImageFilter.MinFilter(3))  # dark ink on light paper grows
    return img.filter(ImageFilter.MaxFilter(3))


def uneven_light(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """A linear illumination gradient across the line (lamp or scanner shading)."""
    w, h = img.size
    angle = rng.uniform(0, 2 * math.pi)
    amp = rng.uniform(0, 80) * s
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = (np.cos(angle) * xx / max(w, 1) + np.sin(angle) * yy / max(h, 1)) * amp
    return _img(_arr(img) + ramp - ramp.mean())


def paper_texture(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Low-frequency blotches multiplied into the image (paper fibres, stains)."""
    w, h = img.size
    gw, gh = max(2, w // 16), max(2, h // 16)
    field = rng.normal(0, 1, size=(gh, gw)).astype(np.float32)
    up = np.asarray(
        Image.fromarray(field, mode="F").resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32
    )
    return _img(_arr(img) * (1 + 0.12 * s * up))


def salt_pepper(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    a = _arr(img).copy()
    amount = rng.uniform(0, 0.03) * s
    mask = rng.random(a.shape)
    a[mask < amount / 2] = 0
    a[mask > 1 - amount / 2] = 255
    return _img(a)


def otsu_threshold(a: np.ndarray) -> float:
    """Otsu's threshold on a 0-255 image (maximises between-class variance)."""
    hist = np.bincount(np.clip(a, 0, 255).astype(np.int64).ravel(), minlength=256).astype(
        np.float64
    )
    p = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return float(np.nanargmax(between))


def binarize(img: Image.Image, rng: np.random.Generator, s: float) -> Image.Image:
    """Hard black-and-white threshold, as in old fax and photocopy scans.

    Otsu's threshold, jittered by up to ±10 grey levels; a threshold at the mean wipes out faint
    text when the background dominates the line.
    """
    a = _arr(img)
    t = otsu_threshold(a) + rng.uniform(-10, 10) * s
    return _img(np.where(a > t, 255.0, 0.0))


TRANSFORMS: dict[str, Transform] = {
    "rotate": rotate, "shear": shear, "stretch": stretch, "perspective": perspective,
    "margin_jitter": margin_jitter, "brightness_contrast": brightness_contrast, "gamma": gamma,
    "invert": invert, "gaussian_blur": gaussian_blur, "gaussian_noise": gaussian_noise,
    "jpeg": jpeg, "low_resolution": low_resolution, "ink_spread": ink_spread,
    "uneven_light": uneven_light, "paper_texture": paper_texture, "salt_pepper": salt_pepper,
    "binarize": binarize,
}  # fmt: skip


# ----------------------------------------------------------------------------------- presets


@dataclass(frozen=True)
class Step:
    name: str
    p: float
    severity: float


_GEOMETRIC = (
    Step("rotate", 0.3, 0.5), Step("shear", 0.3, 0.6), Step("stretch", 0.3, 0.6),
    Step("perspective", 0.2, 0.5), Step("margin_jitter", 0.5, 1.0),
)  # fmt: skip
_PHOTOMETRIC = (
    Step("brightness_contrast", 0.5, 0.6), Step("gamma", 0.3, 0.6), Step("invert", 0.1, 1.0),
    Step("gaussian_blur", 0.2, 0.5), Step("gaussian_noise", 0.2, 0.5), Step("jpeg", 0.2, 0.6),
)  # fmt: skip
_DEGRADATION = (
    Step("low_resolution", 0.25, 0.6), Step("ink_spread", 0.15, 1.0),
    Step("uneven_light", 0.25, 0.6), Step("paper_texture", 0.2, 0.7),
    Step("salt_pepper", 0.1, 0.5), Step("binarize", 0.05, 1.0),
)  # fmt: skip

PRESETS: dict[str, tuple[Step, ...]] = {
    "none": (),
    "geometric": _GEOMETRIC,
    "photometric": _PHOTOMETRIC,
    "degradation": _DEGRADATION,
    # Order: geometry first, then document wear, then capture effects (blur, noise, JPEG).
    "full": _GEOMETRIC + _DEGRADATION + _PHOTOMETRIC,
}


class Augmenter:
    """Apply a preset's steps in order, each with its own probability."""

    def __init__(
        self,
        preset: str = "none",
        overrides: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        if preset not in PRESETS:
            raise ValueError(f"unknown augmentation preset {preset!r} (known: {sorted(PRESETS)})")
        steps = list(PRESETS[preset])
        for name, params in (overrides or {}).items():
            idx = next((i for i, st in enumerate(steps) if st.name == name), None)
            if idx is None:
                raise ValueError(f"preset {preset!r} has no transform {name!r}")
            unknown = set(params) - {"p", "severity"}
            if unknown:
                raise ValueError(f"unknown parameters for {name}: {sorted(unknown)}")
            current = steps[idx]
            steps[idx] = replace(
                current,
                p=float(params.get("p", current.p)),
                severity=float(params.get("severity", current.severity)),
            )
        for st in steps:
            if not (0 <= st.p <= 1 and 0 <= st.severity <= 1):
                raise ValueError(f"{st.name}: p and severity must be in [0, 1]")
        self.preset = preset
        self.steps = tuple(steps)

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        h = img.height
        for st in self.steps:
            if rng.random() < st.p:
                img = TRANSFORMS[st.name](img, rng, st.severity)
        return _same_height(img, h)


# ------------------------------------------------------------------------- GLYPH-C corruptions

CORRUPTIONS = (
    "gaussian_blur", "gaussian_noise", "jpeg", "low_resolution", "contrast", "rotate",
    "perspective",
)  # fmt: skip
LEVELS = (1, 2, 3, 4, 5)


def corrupt(img: Image.Image, kind: str, level: int, seed: int) -> Image.Image:
    """Fixed corruption for the GLYPH-C suite: level 1-5 maps to severity 0.2-1.0.

    Unlike augmentation, the strength is not sampled: the random generator only drives noise
    patterns and directions, and the seed is fixed per sample, so the suite is reproducible.
    """
    if kind not in CORRUPTIONS or level not in LEVELS:
        raise ValueError(f"unknown corruption {kind!r} level {level}")
    s = level / 5
    rng = np.random.default_rng(seed)
    h = img.height
    if kind == "gaussian_blur":
        out = img.filter(ImageFilter.GaussianBlur(2.0 * s * h / 64))
    elif kind == "gaussian_noise":
        a = _arr(img)
        out = _img(a + rng.normal(0, 30 * s, size=a.shape))
    elif kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=round(60 - 55 * s))
        out = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    elif kind == "low_resolution":
        w = img.width
        f = 1 - 0.8 * s
        small = img.resize((max(1, round(w * f)), max(3, round(h * f))), Image.Resampling.BILINEAR)
        out = small.resize((w, h), Image.Resampling.BILINEAR)
    elif kind == "contrast":
        a = _arr(img)
        out = _img((a - a.mean()) * (1 - 0.85 * s) + a.mean())
    elif kind == "rotate":
        sign = 1 if rng.random() < 0.5 else -1
        out = _rotate_whole(img, sign * 6 * s)
    else:  # perspective: one side is pulled taller than the other (a camera tilted sideways)
        w = img.width
        d = 0.25 * h * s
        quad: tuple[float, ...]
        if rng.random() < 0.5:
            quad = (-d, -2 * d, -d, h + 2 * d, w + d, h, w + d, 0.0)
        else:
            quad = (-d, 0.0, -d, h, w + d, h + 2 * d, w + d, -2 * d)
        out = img.transform(
            (w, h),
            Image.Transform.QUAD,
            quad,
            resample=Image.Resampling.BILINEAR,
            fillcolor=_fill(img),
        )
    return _same_height(out, h)
