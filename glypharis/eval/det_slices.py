"""Validation slices for the detector report (spec §10.8). Phase 9 extends this file.

The definitions are fixed before any result (`SLICE_VERSION`). Line geometry is measured at the
inference scale (the long side ``long_side``, with `preprocess.scaled_size`'s exact per-axis
factors), on care lines only (`is_box_line`), the lines detection is scored on:

- ``height`` and ``length``: the short and long side of the line's minimum-area rectangle, the
  kernel-thickness view that ``min_text_size`` also uses;
- ``angle``: the ground-truth text direction, `text_angle(line_quad(polygon))`, the TL→TR
  direction folded to (−90°, 90°]. A polygon without a usable TL→TR direction falls back to the
  box rule of `order_quad`. The long axis is not used for this: it would count upright short words
  (taller than wide) as rotated by about 90°;
- ``tall_box``: the minimum-area rectangle's long axis is more than 45° from horizontal. Mostly
  upright short words, the boxes whose predicted direction comes from the edge nearest horizontal;
- ``stacked``: the line belongs to a stack of at least `STACK_MIN_LINES` lines (`_stack_flags`).

A page is in a slice if:

| slice | rule |
|---|---|
| small_text | median care-line height < `SMALL_TEXT_PX` |
| rotated | at least `ROTATED_SHARE` of care lines have abs(angle) >= `ROTATED_DEG` |
| dense | at least `DENSE_LINES` care lines |
| multi_line | at least `MULTI_LINE_SHARE` of care lines are stacked |
| noisy_background | background edge density in the top quartile of the table's pages |

Background edge density is the share of grey pixels with Sobel magnitude >= `EDGE_SOBEL_MIN` at a
long side of `EDGE_LONG_SIDE`, outside every ground-truth polygon and ignore region dilated by
`EDGE_DILATE_PX`. Pages without care lines take part in it (false positives only) and in no other
slice. ``blur_lapvar`` and ``contrast`` come from the Phase 2 image audit when it has the image and
are recomputed the same way otherwise, so Phase 9 can slice by them.

The table is cached per pages file and long side, since building it decodes every page.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np
from PIL import Image, ImageOps

from glypharis.data.audit import laplacian_variance
from glypharis.data.crops import line_quad
from glypharis.data.pages import Page
from glypharis.eval.page_metrics import is_box_line
from glypharis.models.det.postprocess import order_quad, text_angle
from glypharis.models.det.preprocess import scaled_size

SLICE_VERSION: Final = 1
PAGE_SLICES: Final = ("small_text", "rotated", "dense", "multi_line", "noisy_background")

SMALL_TEXT_PX: Final = 10.0
ROTATED_DEG: Final = 10.0
ROTATED_SHARE: Final = 0.25
DENSE_LINES: Final = 100
STACK_MIN_LINES: Final = 3
MULTI_LINE_SHARE: Final = 0.5
TALL_BOX_DEG: Final = 45.0
NOISY_QUANTILE: Final = 0.75
EDGE_LONG_SIDE: Final = 1024
EDGE_SOBEL_MIN: Final = 64.0
EDGE_DILATE_PX: Final = 8
AUDIT_FILE: Final = "interim/audit/hiertext_images.jsonl"  # under the data root


@dataclass(frozen=True)
class LineAttrs:  # inference scale
    doc_id: str
    line_id: str
    height: float  # minimum-area-rectangle short side
    length: float  # minimum-area-rectangle long side
    angle: float  # ground-truth TL->TR direction, (-90, 90]
    tall_box: bool  # minimum-area-rectangle long axis more than 45° from horizontal
    stacked: bool


@dataclass(frozen=True)
class PageAttrs:
    doc_id: str
    n_care: int
    density_per_mp: float  # care lines per megapixel of the original page
    median_height: float  # the shares and this are 0.0 on a page without care lines
    rotated_share: float
    tall_box_share: float
    stack_share: float
    bg_edge_density: float
    blur_lapvar: float
    contrast: float


@dataclass
class SliceTable:
    lines: list[LineAttrs]
    pages: list[PageAttrs]
    thresholds: dict[str, float]
    version: int
    pages_sha256: str
    long_side: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SliceTable:
        return cls(
            lines=[LineAttrs(**x) for x in d["lines"]],
            pages=[PageAttrs(**x) for x in d["pages"]],
            thresholds={k: float(v) for k, v in d["thresholds"].items()},
            version=int(d["version"]),
            pages_sha256=str(d["pages_sha256"]),
            long_side=int(d["long_side"]),
        )


# ------------------------------------------------------------------------------- line geometry


def _rect(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Corners, short side, long side and long-axis angle of the minimum-area rectangle."""
    corners = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32))).astype(np.float64)
    e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
    l1, l2 = math.hypot(*e1), math.hypot(*e2)
    long_edge = e1 if l1 >= l2 else e2
    angle = math.degrees(math.atan2(long_edge[1], long_edge[0]))
    angle = (angle + 90.0) % 180.0 - 90.0
    return corners, min(l1, l2), max(l1, l2), 90.0 if angle == -90.0 else angle


def _line_geometry(
    polygon: list[list[float]], sx: float, sy: float
) -> tuple[np.ndarray, float, float, float, bool]:
    """TL, TR, BR, BL quad, height, length, text angle and tall_box of one line, scaled."""
    pts = np.asarray(polygon, dtype=np.float64).reshape(-1, 2) * (sx, sy)
    corners, short, long, long_angle = _rect(pts)
    try:
        quad = np.asarray(line_quad(pts.tolist()), dtype=np.float64)
        angle = text_angle(quad)
    except ValueError:  # fewer than 4 points, or no TL->TR direction: the box rule
        quad = order_quad(corners)
        angle = text_angle(quad)
    return quad, short, long, angle, abs(long_angle) > TALL_BOX_DEG


def _stack_flags(quads: Sequence[np.ndarray], angles: Sequence[float]) -> list[bool]:
    """Lines in stacks of at least `STACK_MIN_LINES`.

    A pair of lines is stacked when, in the text frame of the earlier line in file order (u its
    text direction, v perpendicular): both have abs(angle) < `ROTATED_DEG`; their projections on u
    overlap by at least half the shorter projection; their centres along v are more than half the
    larger extent along v apart; and the gap between them along v is at most that half-extent.
    Stacks are the connected groups of stacked pairs.
    """
    n = len(quads)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    upright = [i for i in range(n) if abs(angles[i]) < ROTATED_DEG]
    if len(upright) >= 2:
        q = np.stack([quads[i] for i in upright])  # (m, 4, 2)
        for a, i in enumerate(upright[:-1]):
            t = math.radians(angles[i])
            u = np.array([math.cos(t), math.sin(t)])
            v = np.array([-u[1], u[0]])
            pu, pv = q @ u, q @ v  # (m, 4)
            ulo, uhi, vlo, vhi = pu.min(1), pu.max(1), pv.min(1), pv.max(1)
            j = np.arange(a + 1, len(upright))
            overlap = np.minimum(uhi[a], uhi[j]) - np.maximum(ulo[a], ulo[j])
            shorter = np.minimum(uhi[a] - ulo[a], uhi[j] - ulo[j])
            half_h = 0.5 * np.maximum(vhi[a] - vlo[a], vhi[j] - vlo[j])
            centre_gap = np.abs((vlo[a] + vhi[a]) / 2 - (vlo[j] + vhi[j]) / 2)
            gap = np.maximum(vlo[a], vlo[j]) - np.minimum(vhi[a], vhi[j])
            ok = (overlap >= 0.5 * shorter) & (centre_gap > half_h) & (gap <= half_h)
            for b in j[ok]:
                parent[find(upright[int(b)])] = find(i)
    roots = [find(i) for i in range(n)]
    sizes: dict[int, int] = {}
    for r in roots:
        sizes[r] = sizes.get(r, 0) + 1
    return [sizes[r] >= STACK_MIN_LINES for r in roots]


# ------------------------------------------------------------------------------ image measures


def _gray(path: Path) -> np.ndarray:
    """Grey image at `EDGE_LONG_SIDE`, decoded as the Phase 2 audit decodes it (JPEG draft, then
    a bilinear resize), with EXIF orientation applied so the polygons fit."""
    with Image.open(path) as img:
        w, h = img.size
        scale = max(w, h) / EDGE_LONG_SIDE
        if scale > 1 and img.format == "JPEG":
            img.draft("L", (max(1, int(w / scale)), max(1, int(h / scale))))
        gray = ImageOps.exif_transpose(img).convert("L")
    gw, gh = gray.size
    if max(gw, gh) != EDGE_LONG_SIDE:
        f = EDGE_LONG_SIDE / max(gw, gh)
        gray = gray.resize(
            (max(1, round(gw * f)), max(1, round(gh * f))), Image.Resampling.BILINEAR
        )
    return np.asarray(gray, dtype=np.float32)


def _edge_density(gray: np.ndarray, page: Page) -> float:
    gh, gw = gray.shape
    fx, fy = gw / page.width, gh / page.height
    text = np.zeros((gh, gw), dtype=np.uint8)
    shapes = [ln.polygon for ln in page.lines] + list(page.ignore)
    for poly in shapes:
        if len(poly) >= 3:
            # 4 fractional bits: polygons keep their sub-pixel position at this scale.
            pts = np.round(np.asarray(poly, dtype=np.float64) * (fx, fy) * 16).astype(np.int32)
            cv2.fillPoly(text, [pts], 1, lineType=cv2.LINE_8, shift=4)
    size = 2 * EDGE_DILATE_PX + 1
    near_text = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    mag = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    )
    background = near_text == 0
    if not background.any():
        return 0.0
    return float((mag[background] >= EDGE_SOBEL_MIN).mean())


def _read_audit(data_root: Path) -> dict[str, tuple[float, float]]:
    """(blur_lapvar, contrast) by image path relative to the data root."""
    path = data_root / AUDIT_FILE
    out: dict[str, tuple[float, float]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("ok"):
                key = str(rec["path"]).replace("\\", "/").removeprefix("data/")
                out[key] = (float(rec["blur_lapvar"]), float(rec["contrast"]))
    return out


# ---------------------------------------------------------------------------------- the table


def _page_attrs(
    page: Page, data_root: Path, long_side: int, audit: dict[str, tuple[float, float]]
) -> tuple[PageAttrs, list[LineAttrs]]:
    sw, sh = scaled_size(page.width, page.height, long_side)
    sx, sy = sw / page.width, sh / page.height
    care = [ln for ln in page.lines if is_box_line(ln)]
    geo = [_line_geometry(ln.polygon, sx, sy) for ln in care]
    stacked = _stack_flags([g[0] for g in geo], [g[3] for g in geo])
    lines = [
        LineAttrs(page.doc_id, ln.line_id, g[1], g[2], g[3], g[4], s)
        for ln, g, s in zip(care, geo, stacked, strict=True)
    ]
    n = len(lines)

    def share(flags: list[bool]) -> float:
        return sum(flags) / n if n else 0.0

    gray = _gray(data_root / page.image)
    blur, contrast = audit.get(page.image) or (laplacian_variance(gray), float(gray.std() / 255.0))
    attrs = PageAttrs(
        doc_id=page.doc_id,
        n_care=n,
        density_per_mp=n / (page.width * page.height / 1e6),
        median_height=float(np.median([a.height for a in lines])) if n else 0.0,
        rotated_share=share([abs(a.angle) >= ROTATED_DEG for a in lines]),
        tall_box_share=share([a.tall_box for a in lines]),
        stack_share=share([a.stacked for a in lines]),
        bg_edge_density=_edge_density(gray, page),
        blur_lapvar=blur,
        contrast=contrast,
    )
    return attrs, lines


def build_slices(
    pages: Sequence[Page], pages_sha256: str, data_root: Path, long_side: int
) -> SliceTable:
    """Measure every page (decodes each image once). `load_or_build_slices` caches the result."""
    for p in pages:
        # Slices read validation pages only; held-out data goes through eval.access.
        if p.split.startswith("test") or p.source == "gdc":
            raise ValueError(f"slices refuse held-out page {p.doc_id} ({p.source}/{p.split})")
    audit = _read_audit(Path(data_root))
    all_lines: list[LineAttrs] = []
    all_pages: list[PageAttrs] = []
    for page in pages:
        attrs, lines = _page_attrs(page, Path(data_root), long_side, audit)
        all_pages.append(attrs)
        all_lines.extend(lines)
    densities = [p.bg_edge_density for p in all_pages]
    noisy = float(np.quantile(densities, NOISY_QUANTILE)) if densities else 0.0
    thresholds = {
        "small_text_px": SMALL_TEXT_PX,
        "rotated_deg": ROTATED_DEG,
        "rotated_share": ROTATED_SHARE,
        "dense_lines": float(DENSE_LINES),
        "multi_line_share": MULTI_LINE_SHARE,
        "noisy_background": noisy,  # the top-quartile edge density of these pages
    }
    return SliceTable(all_lines, all_pages, thresholds, SLICE_VERSION, pages_sha256, long_side)


def slice_cache_path(data_root: Path, pages_sha256: str, long_side: int) -> Path:
    return (
        Path(data_root) / "cache" / "det" / "slices"
        / f"{pages_sha256[:12]}_{long_side}_v{SLICE_VERSION}.json"
    )  # fmt: skip


def load_or_build_slices(
    pages: Sequence[Page], pages_sha256: str, data_root: Path, long_side: int
) -> SliceTable:
    """The slice table for these pages, from the cache when it matches them exactly.

    Cache: ``<data_root>/cache/det/slices/<sha12>_<long_side>_v<SLICE_VERSION>.json``. A cached
    table is used only if its version, pages hash, long side and page list all match.
    """
    path = slice_cache_path(data_root, pages_sha256, long_side)
    doc_ids = [p.doc_id for p in pages]
    if path.is_file():
        table = SliceTable.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if (
            table.version == SLICE_VERSION
            and table.pages_sha256 == pages_sha256
            and table.long_side == long_side
            and [p.doc_id for p in table.pages] == doc_ids
        ):
            return table
    table = build_slices(pages, pages_sha256, data_root, long_side)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(table.to_dict()), encoding="utf-8", newline="\n")
    tmp.replace(path)
    return table


def page_slices(t: SliceTable) -> dict[str, list[str]]:
    """Slice name -> sorted doc_ids, by the rules in the module docstring."""
    th = t.thresholds
    out: dict[str, list[str]] = {name: [] for name in PAGE_SLICES}
    for p in t.pages:
        if p.n_care and p.median_height < th["small_text_px"]:
            out["small_text"].append(p.doc_id)
        if p.n_care and p.rotated_share >= th["rotated_share"]:
            out["rotated"].append(p.doc_id)
        if p.n_care >= th["dense_lines"]:
            out["dense"].append(p.doc_id)
        if p.n_care and p.stack_share >= th["multi_line_share"]:
            out["multi_line"].append(p.doc_id)
        if p.bg_edge_density >= th["noisy_background"]:
            out["noisy_background"].append(p.doc_id)
    return {name: sorted(ids) for name, ids in out.items()}
