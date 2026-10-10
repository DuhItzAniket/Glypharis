"""Data audit: per-line and per-image checks, image statistics, duplicate search.

Every problem becomes a `Flag` with a code from `FLAGS`, which fixes its severity and the default
decision. Nothing is dropped here; `scripts/audit_data.py` turns the flags into a report and a
versioned exclusion list (IMPLEMENTATION_PLAN Phase 3).
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from glypharis.data.hiertext import ImageAnnotation, Line

# -------------------------------------------------------------------------------- thresholds


@dataclass(frozen=True)
class Thresholds:
    """Audit rules. Recorded verbatim in the quality report and the exclusion list."""

    coord_tolerance_px: float = 2.0  # vertices may sit this far outside the image
    min_inside_fraction: float = 0.5  # less of the line's bbox inside the image → exclude
    tiny_line_height_px: float = 4.0  # legible lines below this height are flagged (kept)
    extreme_line_aspect: float = 60.0  # width / height above this is flagged (kept)
    extreme_image_aspect: float = 4.0  # long / short image side above this is flagged (kept)
    stats_long_side: int = 1024  # image statistics are computed at this long side
    dhash_max_distance: int = 6  # near-duplicate: dHash AND pHash distances both within limits
    phash_max_distance: int = 10
    # Text-overlap leakage: two images in different splits that share many distinctive lines.
    text_min_chars: int = 5  # a distinctive line has at least this many characters ...
    text_min_letters: int = 3  # ... and at least this many letters
    text_max_images: int = 20  # lines found in more images are common phrases, not copies
    text_min_shared: int = 5  # flag a pair sharing at least this many distinctive lines ...
    text_min_overlap: float = 0.5  # ... covering this share of the smaller image's lines

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------ flags

EXCLUDE_IMAGE = "exclude image"
EXCLUDE_LINE = "exclude line"
EXCLUDE_LINE_REC = "exclude line from recognizer; don't-care for detector"
FIX_CLIP = "fix: clip to image in Phase 4"
FIX_NORMALISE = "fix: text normalisation in Phase 4"
KEEP = "keep"
EXCLUDE_LOWER = "exclude copy in lower-priority split (test > validation > train)"
EXCLUDE_EXTRA = "exclude extra copy (train only)"


@dataclass(frozen=True)
class FlagSpec:
    severity: str  # error | warning | info
    decision: str
    meaning: str


FLAGS: dict[str, FlagSpec] = {
    "missing_image": FlagSpec("error", EXCLUDE_IMAGE, "annotation without an image file"),
    "corrupt_image": FlagSpec("error", EXCLUDE_IMAGE, "image file cannot be decoded"),
    "size_mismatch": FlagSpec("error", EXCLUDE_IMAGE, "decoded size differs from annotated size"),
    "unannotated_image": FlagSpec("info", KEEP, "image file without an annotation record"),
    "exif_orientation": FlagSpec(
        "info", KEEP, "EXIF orientation tag set; boxes refer to stored pixels, so it is ignored"
    ),
    "extreme_image_aspect": FlagSpec("info", KEEP, "very elongated image"),
    "no_lines": FlagSpec("info", KEEP, "no text annotated (negative example for the detector)"),
    "bad_polygon": FlagSpec("error", EXCLUDE_LINE, "line polygon does not have 4 vertices"),
    "degenerate_box": FlagSpec("error", EXCLUDE_LINE, "line polygon area below 1 px²"),
    "mostly_outside": FlagSpec("error", EXCLUDE_LINE, "most of the line box is outside the image"),
    "out_of_range": FlagSpec("warning", FIX_CLIP, "vertices outside the image beyond tolerance"),
    "empty_text_legible": FlagSpec(
        "error", EXCLUDE_LINE_REC, "line marked legible but its text is empty"
    ),
    "invalid_unicode": FlagSpec(
        "warning", FIX_NORMALISE, "control, replacement, private-use or surrogate characters"
    ),
    "not_nfc": FlagSpec("info", FIX_NORMALISE, "text is not in Unicode NFC form"),
    "tiny_line": FlagSpec("info", KEEP, "legible line shorter than the tiny-height threshold"),
    "extreme_line_aspect": FlagSpec("info", KEEP, "line much longer than it is tall"),
    "word_text_mismatch": FlagSpec(
        "info", KEEP, "line text differs from its words joined by single spaces"
    ),
    "exact_duplicate_cross_split": FlagSpec(
        "error", EXCLUDE_LOWER, "byte-identical image in two splits"
    ),
    "near_duplicate_cross_split": FlagSpec(
        "error", EXCLUDE_LOWER, "perceptually near-identical image in two splits"
    ),
    "text_overlap_cross_split": FlagSpec(
        "error", EXCLUDE_LOWER, "images in two splits share many distinctive text lines"
    ),
    "exact_duplicate_within_split": FlagSpec(
        "warning", EXCLUDE_EXTRA, "byte-identical image twice in one split"
    ),
    "near_duplicate_within_split": FlagSpec(
        "info", KEEP, "perceptually near-identical images in one split"
    ),
}

SPLIT_PRIORITY = {"test": 3, "validation": 2, "train": 1}


@dataclass(frozen=True)
class Flag:
    code: str
    split: str
    image_id: str
    line_id: str = ""
    detail: str = ""

    @property
    def spec(self) -> FlagSpec:
        return FLAGS[self.code]

    def to_row(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.spec.severity,
            "decision": self.spec.decision,
            "split": self.split,
            "image_id": self.image_id,
            "line_id": self.line_id,
            "detail": self.detail,
        }


# ------------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class LineGeometry:
    width: float
    height: float
    angle_deg: float  # direction of the reading axis, (-180, 180]
    area: float
    inside_fraction: float  # share of the axis-aligned bbox inside the image


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def polygon_area(vertices: tuple[tuple[float, float], ...]) -> float:
    n = len(vertices)
    s = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def line_geometry(
    vertices: tuple[tuple[float, float], ...], img_w: int, img_h: int
) -> LineGeometry:
    """Geometry of a 4-point line quadrilateral ordered top-left, top-right, bottom-right,
    bottom-left (HierText's convention)."""
    v0, v1, v2, v3 = vertices
    width = (_dist(v0, v1) + _dist(v3, v2)) / 2
    height = (_dist(v0, v3) + _dist(v1, v2)) / 2
    dx = (v1[0] - v0[0]) + (v2[0] - v3[0])
    dy = (v1[1] - v0[1]) + (v2[1] - v3[1])
    angle = math.degrees(math.atan2(dy, dx))
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    ix = max(0.0, min(max(xs), img_w) - max(min(xs), 0.0))
    iy = max(0.0, min(max(ys), img_h) - max(min(ys), 0.0))
    inside = (ix * iy) / (bw * bh) if bw > 0 and bh > 0 else 0.0
    return LineGeometry(width, height, angle, polygon_area(vertices), inside)


# ------------------------------------------------------------------------------------ text


def invalid_unicode_chars(text: str) -> list[str]:
    """Characters that should never appear in a transcription."""
    bad = []
    for ch in text:
        cat = unicodedata.category(ch)
        if (cat == "Cc" and ch not in "\t") or cat in ("Cs", "Co") or ch == "�":
            bad.append(f"U+{ord(ch):04X}")
    return bad


# ------------------------------------------------------------------------------ line checks


def check_lines(split: str, ann: ImageAnnotation, th: Thresholds) -> list[Flag]:
    flags: list[Flag] = []
    if not ann.lines:
        flags.append(Flag("no_lines", split, ann.image_id))
    for line in ann.lines:
        flags.extend(check_line(split, ann, line, th))
    return flags


def check_line(split: str, ann: ImageAnnotation, line: Line, th: Thresholds) -> list[Flag]:
    def flag(code: str, detail: str = "") -> Flag:
        return Flag(code, split, ann.image_id, line.line_id, detail)

    out: list[Flag] = []
    if len(line.vertices) != 4:
        return [flag("bad_polygon", f"{len(line.vertices)} vertices")]
    geo = line_geometry(line.vertices, ann.width, ann.height)
    if geo.area < 1.0:
        return [flag("degenerate_box", f"area {geo.area:.2f}")]
    if geo.inside_fraction < th.min_inside_fraction:
        return [flag("mostly_outside", f"{geo.inside_fraction:.2f} inside")]
    tol = th.coord_tolerance_px
    if any(
        x < -tol or y < -tol or x > ann.width + tol or y > ann.height + tol
        for x, y in line.vertices
    ):
        out.append(flag("out_of_range"))

    if line.legible:
        if not line.text.strip():
            out.append(flag("empty_text_legible"))
        if geo.height < th.tiny_line_height_px:
            out.append(flag("tiny_line", f"height {geo.height:.1f}px"))
        if geo.height > 0 and geo.width / geo.height > th.extreme_line_aspect:
            out.append(flag("extreme_line_aspect", f"{geo.width / geo.height:.0f}:1"))
    bad = invalid_unicode_chars(line.text)
    if bad:
        out.append(flag("invalid_unicode", " ".join(sorted(set(bad)))))
    if line.text != unicodedata.normalize("NFC", line.text):
        out.append(flag("not_nfc"))
    if line.words and line.legible:
        joined = " ".join(w.text for w in line.words)
        if joined != line.text:
            out.append(flag("word_text_mismatch"))
    return out


# --------------------------------------------------------------------------- image statistics

_DCT32: np.ndarray | None = None


def _dct_matrix(n: int = 32) -> np.ndarray:
    global _DCT32
    if _DCT32 is None:
        k = np.arange(n)[:, None]
        x = np.arange(n)[None, :]
        m = np.cos(np.pi * (2 * x + 1) * k / (2 * n)) * np.sqrt(2 / n)
        m[0] /= np.sqrt(2)
        _DCT32 = m
    return _DCT32


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for b in bits.ravel():
        value = (value << 1) | int(b)
    return value


def dhash(gray: Image.Image) -> int:
    """64-bit difference hash: horizontal gradient signs on a 9×8 thumbnail."""
    a = np.asarray(gray.resize((9, 8), Image.Resampling.BOX), dtype=np.float32)
    return _bits_to_int(a[:, 1:] > a[:, :-1])


def phash(gray: Image.Image) -> int:
    """64-bit perceptual hash: sign of the 8×8 low-frequency DCT block against its median."""
    a = np.asarray(gray.resize((32, 32), Image.Resampling.BOX), dtype=np.float64)
    c = _dct_matrix(32)
    d = (c @ a @ c.T)[:8, :8]
    flat = d.ravel()
    median = float(np.median(flat[1:]))  # ignore the DC term
    return _bits_to_int(d > median)


def laplacian_variance(a: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian; low values mean blur. Expects a 2-D float array."""
    lap = a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1] + a[2:, 1:-1] - 4 * a[1:-1, 1:-1]
    return float(lap.var())


def image_stats(path: str, long_side: int = 1024) -> dict[str, Any]:
    """Decode one image and measure it. Never raises; failures come back as ``ok=False``."""
    out: dict[str, Any] = {"path": path, "ok": False}
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        out["error"] = f"read: {exc}"
        return out
    out["sha256"] = hashlib.sha256(data).hexdigest()
    out["bytes"] = len(data)
    try:
        with Image.open(path) as img:
            w, h = img.size
            out.update(width=w, height=h, mode=img.mode, format=img.format)
            try:
                out["exif_orientation"] = int(img.getexif().get(0x0112, 1))
            except Exception:
                out["exif_orientation"] = 1
            scale = max(w, h) / long_side
            if scale > 1 and img.format == "JPEG":
                img.draft("L", (max(1, int(w / scale)), max(1, int(h / scale))))
            gray = img.convert("L")
            gray.load()
    except Exception as exc:  # truncated, unsupported or corrupt
        out["error"] = f"decode: {type(exc).__name__}: {exc}"
        return out
    gw, gh = gray.size
    if max(gw, gh) != long_side:
        f = long_side / max(gw, gh)
        gray = gray.resize(
            (max(1, round(gw * f)), max(1, round(gh * f))), Image.Resampling.BILINEAR
        )
    a = np.asarray(gray, dtype=np.float32)
    out.update(
        ok=True,
        brightness=float(a.mean() / 255.0),
        contrast=float(a.std() / 255.0),
        blur_lapvar=laplacian_variance(a),
        dhash=dhash(gray),
        phash=phash(gray),
    )
    return out


# ------------------------------------------------------------------------------- duplicates


@dataclass(frozen=True)
class DuplicatePair:
    a: tuple[str, str]  # (split, image_id)
    b: tuple[str, str]
    exact: bool
    dhash_distance: int
    phash_distance: int


def find_duplicates(
    items: list[tuple[str, str, str, int, int]], th: Thresholds
) -> list[DuplicatePair]:
    """Exact (sha256) and near (dHash and pHash) duplicate pairs.

    ``items`` are ``(split, image_id, sha256, dhash, phash)``. Near duplicates need both hash
    distances within their limits, which keeps blank or near-uniform images from matching
    each other on one hash alone.
    """
    pairs: list[DuplicatePair] = []
    if not items:
        return pairs
    d = np.array([it[3] for it in items], dtype=np.uint64)
    p = np.array([it[4] for it in items], dtype=np.uint64)
    shas = [it[2] for it in items]
    by_sha: dict[str, list[int]] = {}
    for idx, sha in enumerate(shas):
        by_sha.setdefault(sha, []).append(idx)
    for i in range(len(items) - 1):
        dd = np.bitwise_count(d[i + 1 :] ^ d[i])
        pd = np.bitwise_count(p[i + 1 :] ^ p[i])
        hits = np.nonzero((dd <= th.dhash_max_distance) & (pd <= th.phash_max_distance))[0]
        exact_hits = {j for j in by_sha[shas[i]] if j > i}
        for j in sorted(set((hits + i + 1).tolist()) | exact_hits):
            pairs.append(
                DuplicatePair(
                    a=(items[i][0], items[i][1]),
                    b=(items[j][0], items[j][1]),
                    exact=shas[i] == shas[j],
                    dhash_distance=int(dd[j - i - 1]),
                    phash_distance=int(pd[j - i - 1]),
                )
            )
    return pairs


def duplicate_flags(pairs: Iterable[DuplicatePair]) -> list[Flag]:
    """One flag per pair, attached to the copy that the decision rule removes."""
    flags = []
    for pr in pairs:
        cross = pr.a[0] != pr.b[0]
        kind = "exact" if pr.exact else "near"
        code = f"{kind}_duplicate_{'cross' if cross else 'within'}_split"
        if cross:
            drop, keep = sorted((pr.a, pr.b), key=lambda s: SPLIT_PRIORITY[s[0]])
        else:
            keep, drop = pr.a, pr.b
        detail = (
            f"other copy {keep[0]}/{keep[1]}; dhash {pr.dhash_distance}, phash {pr.phash_distance}"
        )
        flags.append(Flag(code, drop[0], drop[1], "", detail))
    return flags


# ----------------------------------------------------------------------------- text overlap


@dataclass(frozen=True)
class TextOverlapPair:
    a: tuple[str, str]
    b: tuple[str, str]
    shared: int
    overlap: float  # shared / size of the smaller image's distinctive-line set


def distinctive_lines(ann: ImageAnnotation, th: Thresholds) -> set[str]:
    out = set()
    for line in ann.lines:
        t = line.text.strip()
        if (
            line.legible
            and len(t) >= th.text_min_chars
            and sum(c.isalpha() for c in t) >= th.text_min_letters
        ):
            out.add(t)
    return out


def find_text_overlaps(
    texts: dict[tuple[str, str], set[str]], th: Thresholds
) -> list[TextOverlapPair]:
    """Cross-split image pairs that share many distinctive text lines.

    A recognizer that saw the same sign, product label or chart in training would be scored on
    memorised text, which pixel hashes miss when the photo or crop differs.
    """
    index: dict[str, list[tuple[str, str]]] = {}
    for key, lines in texts.items():
        for t in lines:
            index.setdefault(t, []).append(key)
    shared: dict[tuple[tuple[str, str], tuple[str, str]], int] = {}
    for keys in index.values():
        if len(keys) < 2 or len(keys) > th.text_max_images:
            continue
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = sorted((keys[i], keys[j]))
                if a[0] != b[0]:
                    shared[(a, b)] = shared.get((a, b), 0) + 1
    pairs = []
    for (a, b), n in shared.items():
        overlap = n / min(len(texts[a]), len(texts[b]))
        if n >= th.text_min_shared and overlap >= th.text_min_overlap:
            pairs.append(TextOverlapPair(a, b, n, overlap))
    return sorted(pairs, key=lambda p: (-p.shared, p.a, p.b))


def text_overlap_flags(pairs: Iterable[TextOverlapPair]) -> list[Flag]:
    flags = []
    for pr in pairs:
        drop, keep = sorted((pr.a, pr.b), key=lambda s: SPLIT_PRIORITY[s[0]])
        detail = f"other copy {keep[0]}/{keep[1]}; {pr.shared} shared lines ({pr.overlap:.0%})"
        flags.append(Flag("text_overlap_cross_split", drop[0], drop[1], "", detail))
    return flags


# --------------------------------------------------------------------------------- summaries


def summarise(values: Iterable[float]) -> dict[str, float | int]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    q = np.percentile(a, [1, 5, 25, 50, 75, 95, 99])
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "p50": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "p99": float(q[6]),
        "max": float(a.max()),
    }
