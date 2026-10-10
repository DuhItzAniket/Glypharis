"""GlyphSynth: document-style synthetic text lines (ADR-0003, ADR-0009).

A sample is fully determined by ``(seed, index)``: text, font, size, layout and capture effects are
all drawn from ``numpy.random.default_rng([seed, index])``. Output crops follow the HierText crop
convention (height 64 px, margin 0.15 x line height) so the recognizer sees one input format.

Train and validation use disjoint font families and disjoint source texts, so GlyphSynth-val
measures generalisation to unseen typefaces and unseen text.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from glypharis.data.augment import TRANSFORMS, Step
from glypharis.data.text import Charset, normalize_label

# ----------------------------------------------------------------------------------- fonts


@dataclass(frozen=True)
class FontFace:
    family: str
    category: str  # serif | sans | mono
    licence_dir: str  # ofl | apache | ufl
    path: str  # file path relative to the font root
    italic: bool
    variable_weight: tuple[float, float] | None  # (min, max) of the wght axis, if variable
    codepoints: frozenset[int] = field(repr=False, default=frozenset())

    def covers(self, text: str) -> bool:
        return all(ord(ch) in self.codepoints or ch == " " for ch in text)


def read_face(root: Path, rel: str, family: str, category: str, licence_dir: str) -> FontFace:
    from fontTools.ttLib import TTFont

    with TTFont(str(root / rel), lazy=True) as tt:
        cmap = tt.getBestCmap() or {}
        wght = None
        if "fvar" in tt:
            for axis in tt["fvar"].axes:
                if axis.axisTag == "wght":
                    wght = (float(axis.minValue), float(axis.maxValue))
    name = Path(rel).name.lower()
    return FontFace(
        family=family,
        category=category,
        licence_dir=licence_dir,
        path=rel,
        italic="italic" in name,
        variable_weight=wght,
        codepoints=frozenset(cmap),
    )


def split_families(families: dict[str, str], held_out_fraction: float, seed: int) -> dict[str, str]:
    """Assign each family to train or val, stratified by category, deterministically."""
    rng = np.random.default_rng(seed)
    out: dict[str, str] = {}
    by_cat: dict[str, list[str]] = {}
    for fam, cat in sorted(families.items()):
        by_cat.setdefault(cat, []).append(fam)
    for cat in sorted(by_cat):
        fams = by_cat[cat]
        n_val = max(1, round(len(fams) * held_out_fraction))
        val = set(rng.choice(fams, size=n_val, replace=False).tolist())
        for fam in fams:
            out[fam] = "val" if fam in val else "train"
    return out


# ------------------------------------------------------------------------------------ text

_PG_HEADER = re.compile(r"^\s*(\*\*\*|end of (the )?project gutenberg)", re.IGNORECASE)


def book_paragraphs(text: str, min_chars: int = 40) -> list[str]:
    """Paragraphs of a PG-19 book: hard-wrapped lines joined, labels normalised."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", text):
        if _PG_HEADER.match(block):
            continue
        para = normalize_label(" ".join(block.split()))
        if len(para) >= min_chars:
            paragraphs.append(para)
    return paragraphs


_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_WORDS = (
    "data",
    "report",
    "total",
    "section",
    "figure",
    "table",
    "page",
    "chapter",
    "note",
    "value",
    "price",
    "item",
    "order",
    "account",
    "invoice",
    "service",
    "review",
    "summary",
    "result",
    "method",
    "sample",
    "index",
)


def modern_snippet(rng: np.random.Generator) -> str:
    """Short modern content PG-19 lacks: dates, money, codes, contacts, references, lists."""

    def num(lo: int, hi: int) -> int:
        return int(rng.integers(lo, hi + 1))

    word = _WORDS[num(0, len(_WORDS) - 1)]
    kind = num(0, 13)
    if kind == 0:
        return f"{num(1, 28)} {_MONTHS[num(0, 11)]} {num(1950, 2035)}"
    if kind == 1:
        return f"{num(1990, 2035)}-{num(1, 12):02d}-{num(1, 28):02d}"
    if kind == 2:
        return f"{num(1, 12):02d}/{num(1, 28):02d}/{num(0, 99):02d}"
    if kind == 3:
        cur = ("$", "€", "£", "USD ", "Rs. ")[num(0, 4)]
        return f"{cur}{num(1, 99999):,}.{num(0, 99):02d}"
    if kind == 4:
        return f"{num(0, 100)}.{num(0, 9)}%"
    if kind == 5:
        return f"+{num(1, 99)} {num(100, 999)} {num(100, 999)} {num(1000, 9999)}"
    if kind == 6:
        domain = _WORDS[num(0, len(_WORDS) - 1)]
        return f"{word}{num(1, 99)}@{domain}.{('com', 'org', 'net', 'in')[num(0, 3)]}"
    if kind == 7:
        return f"https://www.{word}.org/{_WORDS[num(0, len(_WORDS) - 1)]}/{num(1, 9999)}"
    if kind == 8:
        return f"{('Fig.', 'Table', 'Eq.', 'Section', 'p.')[num(0, 4)]} {num(1, 20)}.{num(1, 20)}"
    if kind == 9:
        return f"{''.join(rng.choice(list(string.ascii_uppercase), size=3))}-{num(1000, 99999)}"
    if kind == 10:
        unit = ("kg", "mm", "cm", "km", "ml", "mg", "GB", "MHz")[num(0, 7)]
        return f"{num(1, 999)}.{num(0, 99)} {unit}"
    if kind == 11:
        bullet = ("•", "-", f"{num(1, 30)}.", f"({chr(97 + num(0, 7))})", f"{num(1, 9)})")[
            num(0, 4)
        ]
        return f"{bullet} {word.capitalize()} {_WORDS[num(0, len(_WORDS) - 1)]}"
    if kind == 12:
        return f"Page {num(1, 600)}" if rng.random() < 0.5 else str(num(1, 600))
    return f"{word.capitalize()} {num(1, 99)}: {num(1, 9999):,}"


@dataclass(frozen=True)
class TextConfig:
    modern_fraction: float = 0.2
    upper_fraction: float = 0.06  # ALL CAPS lines (headings)
    # Line length mixture: (probability, min chars, max chars)
    lengths: tuple[tuple[float, int, int], ...] = ((0.7, 30, 90), (0.22, 6, 30), (0.08, 1, 6))


class TextSampler:
    def __init__(self, paragraphs: list[str], charset: Charset, cfg: TextConfig | None = None):
        self.paragraphs = [p for p in paragraphs if p]
        if not self.paragraphs:
            raise ValueError("no paragraphs to sample from")
        self.charset = charset
        self.cfg = cfg or TextConfig()

    def _window(self, rng: np.random.Generator, n_chars: int) -> str:
        para = self.paragraphs[int(rng.integers(len(self.paragraphs)))]
        if len(para) <= n_chars:
            return para
        start = int(rng.integers(0, len(para) - n_chars + 1))
        # Snap to word boundaries so lines do not start or end mid-word (hyphenation aside).
        left = para.rfind(" ", 0, start) + 1
        right = para.find(" ", left + n_chars)
        return para[left : right if right != -1 else len(para)].strip()

    def sample(self, rng: np.random.Generator) -> str:
        for _ in range(20):
            if rng.random() < self.cfg.modern_fraction:
                text = modern_snippet(rng)
            else:
                probs = np.array([p for p, _, _ in self.cfg.lengths])
                i = int(rng.choice(len(probs), p=probs / probs.sum()))
                _, lo, hi = self.cfg.lengths[i]
                text = self._window(rng, int(rng.integers(lo, hi + 1)))
            if rng.random() < self.cfg.upper_fraction:
                text = text.upper()
            text = normalize_label(text)
            if text and not self.charset.oov(text):
                return text
        return "the"  # practically unreachable; keeps the sampler total


# -------------------------------------------------------------------------------- rendering


@dataclass(frozen=True)
class RenderConfig:
    out_height: int = 64
    margin: float = 0.15
    size_px: tuple[float, float] = (10.0, 48.0)  # log-uniform rendering size
    italic_p: float = 0.15
    bold_p: float = 0.15
    invert_p: float = 0.03
    neighbour_p: float = 0.35  # fragments of adjacent lines in the margins
    underline_p: float = 0.03
    capture: bool = True  # apply capture effects (Phase 12 factor)


CAPTURE_STEPS = (
    Step("rotate", 0.15, 0.3), Step("perspective", 0.1, 0.4),
    Step("paper_texture", 0.4, 0.6), Step("uneven_light", 0.3, 0.6),
    Step("ink_spread", 0.15, 1.0), Step("low_resolution", 0.2, 0.5),
    Step("gaussian_blur", 0.3, 0.5), Step("gaussian_noise", 0.3, 0.4),
    Step("salt_pepper", 0.05, 0.4), Step("binarize", 0.03, 1.0), Step("jpeg", 0.3, 0.6),
)  # fmt: skip


@dataclass
class RenderInfo:
    family: str
    font_file: str
    size_px: float
    italic: bool
    weight: float | None
    inverted: bool
    neighbours: bool
    underline: bool
    capture: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_font(
    root: Path, face: FontFace, size: int, weight: float | None
) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(root / face.path), size=size)
    if face.variable_weight is not None and weight is not None:
        # set_variation_by_axes needs a value for every axis, in the font's order: keep each
        # axis at its default and change only the weight.
        values: list[float] = []
        for axis in font.get_variation_axes():
            raw = axis.get("name") or b""
            name = raw.decode() if isinstance(raw, bytes) else str(raw)
            is_weight = name.lower() in ("weight", "wght")
            values.append(weight if is_weight else float(axis.get("default") or 0))
        font.set_variation_by_axes(values)
    return font


def render_line(
    text: str,
    face: FontFace,
    font_root: Path,
    rng: np.random.Generator,
    cfg: RenderConfig,
    neighbour_text: tuple[str, str] = ("", ""),
) -> tuple[Image.Image, RenderInfo]:
    """Render one line and return the crop (mode "L", height ``cfg.out_height``) and its record."""
    lo, hi = cfg.size_px
    size = math.exp(rng.uniform(math.log(lo), math.log(hi)))
    italic = face.italic
    weight = None
    if face.variable_weight is not None:
        wmin, wmax = face.variable_weight
        weight = float(min(max(700.0 if rng.random() < cfg.bold_p else 400.0, wmin), wmax))
    font = _load_font(font_root, face, max(6, round(size)), weight)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    text_w = max(1, math.ceil(font.getlength(text)))
    margin = max(1, round(cfg.margin * line_h))
    W, H = text_w + 2 * margin, line_h + 2 * margin

    paper = int(rng.integers(205, 256))
    ink = int(rng.integers(0, 70))
    canvas = Image.new("L", (W, H), paper)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), text, font=font, fill=ink)
    underline = bool(rng.random() < cfg.underline_p)
    if underline:
        y = margin + ascent + max(1, descent // 3)
        draw.line((margin, y, margin + text_w, y), fill=ink, width=max(1, line_h // 16))
    neighbours = bool(rng.random() < cfg.neighbour_p) and any(neighbour_text)
    if neighbours:
        gap = round(line_h * rng.uniform(1.05, 1.4))  # baseline-to-baseline spacing
        above, below = neighbour_text
        if above:
            draw.text(
                (margin + int(rng.integers(-margin, margin + 1)), margin - gap),
                above,
                font=font,
                fill=ink,
            )
        if below:
            draw.text(
                (margin + int(rng.integers(-margin, margin + 1)), margin + gap),
                below,
                font=font,
                fill=ink,
            )

    out_w = max(1, round(W * cfg.out_height / H))
    img = canvas.resize(
        (out_w, cfg.out_height),
        Image.Resampling.BICUBIC if cfg.out_height > H else Image.Resampling.BOX,
    )

    applied: list[str] = []
    if cfg.capture:
        for st in CAPTURE_STEPS:
            if rng.random() < st.p:
                img = TRANSFORMS[st.name](img, rng, st.severity)
                applied.append(st.name)
        if img.height != cfg.out_height:
            img = img.resize(
                (max(1, round(img.width * cfg.out_height / img.height)), cfg.out_height)
            )
    inverted = bool(rng.random() < cfg.invert_p)
    if inverted:
        img = Image.eval(img, lambda v: 255 - v)
    info = RenderInfo(
        face.family,
        face.path,
        round(size, 2),
        italic,
        weight,
        inverted,
        neighbours,
        underline,
        applied,
    )
    return img, info


class GlyphSynth:
    """Line generator over a font pool and a text sampler."""

    def __init__(
        self,
        faces: list[FontFace],
        font_root: Path,
        sampler: TextSampler,
        cfg: RenderConfig | None = None,
    ) -> None:
        if not faces:
            raise ValueError("no fonts")
        self.faces = faces
        self.font_root = font_root
        self.sampler = sampler
        self.cfg = cfg or RenderConfig()
        families = sorted({f.family for f in faces})
        self._by_family = {fam: [f for f in faces if f.family == fam] for fam in families}
        self._families = families

    def sample(self, seed: int, index: int) -> tuple[str, Image.Image, RenderInfo]:
        rng = np.random.default_rng([seed, index])
        family = self._families[int(rng.integers(len(self._families)))]
        candidates = self._by_family[family]
        italics = [f for f in candidates if f.italic]
        uprights = [f for f in candidates if not f.italic] or candidates
        pool = italics if italics and rng.random() < self.cfg.italic_p else uprights
        face = pool[int(rng.integers(len(pool)))]
        text = self.sampler.sample(rng)
        for _ in range(10):  # the font must have every glyph
            if face.covers(text):
                break
            text = self.sampler.sample(rng)
        else:
            text = "".join(ch for ch in text if face.covers(ch)).strip() or "the"
        neighbours = (self.sampler.sample(rng), self.sampler.sample(rng))
        neighbours = tuple(t if face.covers(t) else "" for t in neighbours)  # type: ignore[assignment]
        img, info = render_line(text, face, self.font_root, rng, self.cfg, neighbours)
        return text, img, info
