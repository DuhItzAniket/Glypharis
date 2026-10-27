"""Crop geometry for detector training: affine views of a page, page decoding and warping.

Every coordinate here is an *edge* coordinate: pixel (i, j) covers [j, j+1) x [i, i+1), so its
centre is (j + ½, i + ½), page polygons need no offset, and a scale s maps the image extent [0, W]
onto [0, sW] exactly. OpenCV puts pixel centres at integers instead; `Affine.cv2_matrix` does that
half-pixel conversion, and nothing else in the detector does it by hand.

A training crop is a single affine warp of the decoded page (`render_view`): scale and rotation
about the page centre, then a whole-pixel crop origin, all drawn by `sample_view` from the sample's
own generator. The polygons go through the same `View.affine`, so pixels and labels cannot drift
apart. Large downscales decode the JPEG at 1/2, 1/4 or 1/8 (libjpeg DCT scaling) and fold that
factor into the warp, which keeps the residual resampling at a factor of at least ½.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from glypharis.config.schema import DetDataConfig
from glypharis.models.det.preprocess import PAD_BGR

Reduce = Literal[1, 2, 4, 8]
_REDUCED_FLAGS: dict[int, int] = {
    1: cv2.IMREAD_COLOR,
    2: cv2.IMREAD_REDUCED_COLOR_2,
    4: cv2.IMREAD_REDUCED_COLOR_4,
    8: cv2.IMREAD_REDUCED_COLOR_8,
}
# libjpeg scales during decoding and rounds sizes up, which keeps each decoded pixel on a k x k
# block of the page. OpenCV resizes other formats after decoding and rounds sizes down, so those
# are decoded reduced only when the page size divides by k.
_JPEG_SUFFIXES = (".jpg", ".jpeg")


@dataclass(frozen=True, eq=False)
class Affine:
    """A 2x3 affine map ``x -> A x + b`` in edge coordinates (``m = [A | b]``)."""

    m: np.ndarray

    def __post_init__(self) -> None:
        m = np.array(self.m, dtype=np.float64)
        if m.shape != (2, 3):
            raise ValueError(f"an affine matrix is 2x3, got shape {m.shape}")
        object.__setattr__(self, "m", m)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        """Map (N, 2) points; returns float64 (N, 2)."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        out: np.ndarray = p @ self.m[:, :2].T + self.m[:, 2]
        return out

    def compose(self, other: Affine) -> Affine:
        """``self ∘ other``: apply ``other`` first."""
        a = self.m[:, :2] @ other.m[:, :2]
        b = self.m[:, :2] @ other.m[:, 2] + self.m[:, 2]
        return Affine(np.column_stack([a, b]))

    def cv2_matrix(self) -> np.ndarray:
        """The same map in OpenCV's index convention (pixel centres at integers).

        With x_edge = x_index + ½ on both sides, the offset b becomes b + A·(½, ½) − (½, ½).
        """
        a, b = self.m[:, :2], self.m[:, 2]
        out: np.ndarray = np.column_stack([a, b + a @ np.array([0.5, 0.5]) - 0.5])
        return out

    @staticmethod
    def scale(sx: float, sy: float) -> Affine:
        return Affine(np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0]]))


@dataclass(frozen=True)
class View:
    """One training crop of a page: the page -> crop map and the draws it came from."""

    scale: float  # f, the page -> crop scale
    angle_deg: float
    text_centred: bool  # the crop origin was placed around a care line
    affine: Affine  # page -> crop, edge coordinates


def min_area_rect(poly: np.ndarray) -> tuple[float, float, float]:
    """Long side, short side and long-axis angle in (-90, 90] (degrees, y down) of the
    minimum-area rectangle around the points.

    The angle comes from the rectangle's own edge vectors rather than OpenCV's angle output, whose
    convention has changed between OpenCV versions.
    """
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return 0.0, 0.0, 0.0
    corners = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float64)
    e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    long_edge = e1 if l1 >= l2 else e2
    angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    if angle <= -90.0:
        angle += 180.0
    elif angle > 90.0:
        angle -= 180.0
    return max(l1, l2), min(l1, l2), angle


def sample_view(
    rng: np.random.Generator,
    page_wh: tuple[int, int],
    care_centres: np.ndarray,
    data: DetDataConfig,
    long_side: int,
) -> View:
    """Draw one crop of a W x H page (§4.3). ``care_centres`` are (K, 2) page points, one per care
    line (K may be 0).

    Draws, in this order: scale, angle, crop mode, line choice (text-seeking crops only), one
    offset per axis. The scale is log-uniform around the inference scale ``long_side / max(W, H)``.
    The page is rotated about its centre into a frame of its rotated extent; along each axis the
    crop origin is then drawn so that the page lies fully inside the crop when it fits, or so that
    the chosen care line lands 10–90% of the way across the crop, or uniformly. The origin is
    rounded to whole pixels, so every crop of a sample is an integer shift of the same transformed
    page, and its labels are exact crops of the full-page labels.
    """
    w, h = page_wh
    size = data.crop_size
    f = (long_side / max(w, h)) * math.exp(
        rng.uniform(math.log(data.scale_min), math.log(data.scale_max))
    )
    angle = float(rng.uniform(-data.rotate_max_deg, data.rotate_max_deg))
    t = math.radians(angle)
    cos, sin = math.cos(t), math.sin(t)
    a = f * np.array([[cos, -sin], [sin, cos]])
    c0 = np.array([w / 2, h / 2])
    extent = f * np.array([abs(w * cos) + abs(h * sin), abs(w * sin) + abs(h * cos)])
    c1 = extent / 2

    centres = np.asarray(care_centres, dtype=np.float64).reshape(-1, 2)
    text_mode = bool(rng.random() < data.text_crop_p) and len(centres) > 0
    q = a @ (centres[int(rng.integers(len(centres)))] - c0) + c1 if text_mode else None
    origin = np.zeros(2)
    for ax in range(2):
        u = rng.random()
        room = extent[ax] - size
        if room <= 0:
            o = -u * (size - extent[ax])  # the page fits: place it anywhere inside the crop
        elif q is not None:
            o = min(max(q[ax] - (0.1 + 0.8 * u) * size, 0.0), room)
        else:
            o = u * room
        origin[ax] = math.floor(o + 0.5)
    affine = Affine(np.column_stack([a, c1 - a @ c0 - origin]))
    return View(f, angle, text_mode and bool((extent > size).any()), affine)


def _decode_bgr(path: Path, page_wh: tuple[int, int], reduce: Reduce) -> np.ndarray:
    """BGR uint8 decode at 1/``reduce`` with EXIF orientation applied (as `open_page_image` does).

    Raises ValueError when the decoded size is not (⌈H/reduce⌉, ⌈W/reduce⌉): a page record whose
    size does not match its image would otherwise warp to the wrong place without any error.
    """
    if reduce not in _REDUCED_FLAGS:
        raise ValueError(f"reduce must be 1, 2, 4 or 8, got {reduce}")
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), _REDUCED_FLAGS[reduce])
    if img is None:
        raise ValueError(f"cannot decode {path}")
    w, h = page_wh
    want = (-(-h // reduce), -(-w // reduce))
    if img.shape[:2] != want:
        raise ValueError(
            f"{path}: decoded {img.shape[1]}x{img.shape[0]} at 1/{reduce}, expected "
            f"{want[1]}x{want[0]} for a {w}x{h} page record"
        )
    return img


def decode_page(path: Path, page_wh: tuple[int, int], reduce: Reduce = 1) -> np.ndarray:
    """The page image as RGB uint8 (H/reduce, W/reduce, 3); the only public decode."""
    return cv2.cvtColor(_decode_bgr(path, page_wh, reduce), cv2.COLOR_BGR2RGB)


def _reduce_factor(path: Path, page_wh: tuple[int, int], scale: float, enabled: bool) -> Reduce:
    """The largest decode reduction k with k·scale <= 1 that keeps pixels on k x k page blocks."""
    if not enabled:
        return 1
    jpeg = Path(path).suffix.lower() in _JPEG_SUFFIXES
    w, h = page_wh
    ks: tuple[Reduce, ...] = (8, 4, 2)
    for k in ks:
        if k * scale <= 1.0 and (jpeg or (w % k == 0 and h % k == 0)):
            return k
    return 1


def render_view(
    path: Path, page_wh: tuple[int, int], view: View, size: int, reduced_decode: bool
) -> np.ndarray:
    """The (size, size, 3) uint8 RGB crop of ``view``, with `PAD_RGB` outside the page.

    The page is decoded as BGR (reduced when ``reduced_decode`` allows), warped with the BGR pad
    colour, and only the crop is converted to RGB.
    """
    k = _reduce_factor(path, page_wh, view.scale, reduced_decode)
    bgr = _decode_bgr(path, page_wh, k)
    # A reduced pixel covers k x k page pixels: reduced -> page is a scale by k.
    m = view.affine.compose(Affine.scale(k, k)).cv2_matrix()
    out = cv2.warpAffine(
        bgr,
        m,
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=PAD_BGR,
    )
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
