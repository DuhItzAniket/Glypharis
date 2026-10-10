"""Text normalisation and the recognizer character set.

Two kinds of normalisation, kept apart on purpose:

- `normalize_label` turns a ground-truth transcription into a training label. It only removes
  differences that are invisible on the page or that no OCR system should be asked to keep apart
  (Unicode composition, control characters, ligatures, odd spaces). Curly vs straight quotes, dashes
  and case are kept: they look different in print.
- `normalize_eval` implements the two evaluation tracks of EXPERIMENT_PROTOCOL §2.

Both original and normalised text are stored for every sample (brief §16).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

NORMALIZATION_VERSION = 1

# Spaces that render as a plain space: U+00A0 no-break, U+2000-U+200A typographic spaces,
# U+202F narrow no-break, U+205F medium mathematical, U+3000 ideographic.
_SPACES = {" ", *(chr(c) for c in range(0x2000, 0x200B)), " ", " ", "　"}
# Characters with no visible glyph: zero-width spaces and joiners, BOM, soft hyphen, and the
# variation selectors U+FE00-U+FE0F (e.g. the emoji presentation selector after a symbol).
_INVISIBLE = {
    "​", "‌", "‍", "⁠", "﻿", "­",
    *(chr(c) for c in range(0xFE00, 0xFE10)),
}  # fmt: skip
# Latin ligatures (U+FB00-U+FB06) are written out; they are one glyph for two or three letters.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
}  # fmt: skip
# Typographic map used only in the "normalised" evaluation track.
_TYPOGRAPHIC = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-",
    "−": "-", "…": "...",
}  # fmt: skip
_WS = re.compile(r"\s+")
_OTHER_SCRIPTS = {
    "ARABIC", "ARMENIAN", "BENGALI", "CJK", "CYRILLIC", "DEVANAGARI", "ETHIOPIC", "FULLWIDTH",
    "GEORGIAN", "GUJARATI", "GURMUKHI", "HALFWIDTH", "HANGUL", "HEBREW", "HIRAGANA", "IDEOGRAPHIC",
    "KANNADA", "KATAKANA", "KHMER", "LAO", "MALAYALAM", "MONGOLIAN", "MYANMAR", "ORIYA", "SINHALA",
    "TAMIL", "TELUGU", "THAI", "TIBETAN",
}  # fmt: skip


def _drop(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return ch in _INVISIBLE or cat in ("Cc", "Cs", "Co", "Cn") or ch == "�"


def normalize_label(text: str) -> str:
    """Training-label normalisation (version `NORMALIZATION_VERSION`)."""
    text = unicodedata.normalize("NFC", text)
    out = []
    for ch in text:
        if ch in _SPACES or ch in "\t\n\r\x0b\x0c":
            out.append(" ")
        elif ch in _LIGATURES:
            out.append(_LIGATURES[ch])
        elif not _drop(ch):
            out.append(ch)
    return _WS.sub(" ", "".join(out)).strip()


def normalize_eval(text: str, track: Literal["strict", "normalised"]) -> str:
    """Evaluation tracks (EXPERIMENT_PROTOCOL §2), applied to predictions and references alike."""
    text = normalize_label(text)
    if track == "strict":
        return text
    if track == "normalised":
        text = unicodedata.normalize("NFKC", text)
        text = "".join(_TYPOGRAPHIC.get(ch, ch) for ch in text)
        return _WS.sub(" ", text).strip().casefold()
    raise ValueError(f"unknown track {track!r}")


NORMALIZATION_RULES = [
    "Unicode NFC",
    "tabs, newlines and typographic spaces (U+00A0, U+2000-U+200A, U+202F, U+205F, U+3000)"
    " -> space",
    "zero-width characters, soft hyphen, variation selectors U+FE00-U+FE0F, control, surrogate,"
    " private-use, unassigned and U+FFFD removed",
    "Latin ligatures U+FB00-U+FB06 written out (e.g. U+FB01 -> 'fi')",
    "runs of whitespace collapsed to one space; leading and trailing spaces removed",
    "case, quotes, dashes and all other characters kept as written",
]


# ---------------------------------------------------------------------------------- charset


def in_scope(ch: str) -> bool:
    """Characters an English printed-text recognizer should know about, if frequent enough.

    Letters must be Latin or Greek (units, maths). Digits, punctuation, symbols and the space are
    allowed from any block; letters of other scripts are out of scope (ADR-0008: English only).
    """
    if ch == " ":
        return True
    cat = unicodedata.category(ch)
    name = unicodedata.name(ch, "")
    if cat.startswith("L"):
        return name.startswith(("LATIN", "GREEK"))
    # Punctuation and digits that belong to another script (e.g. the Devanagari danda).
    return cat[0] in "NPS" and name.split(" ", 1)[0] not in _OTHER_SCRIPTS


@dataclass
class Charset:
    """Ordered character vocabulary. Index 0 is the CTC blank; characters start at 1."""

    characters: list[str]
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(set(self.characters)) != len(self.characters):
            raise ValueError("charset has duplicate characters")
        if any(len(c) != 1 for c in self.characters):
            raise ValueError("charset entries must be single characters")
        self._index = {c: i + 1 for i, c in enumerate(self.characters)}

    BLANK = 0

    @property
    def num_classes(self) -> int:
        """Output size of the recognizer head: characters + blank."""
        return len(self.characters) + 1

    def __contains__(self, ch: object) -> bool:
        return ch in self._index

    def oov(self, text: str) -> set[str]:
        return {ch for ch in text if ch not in self._index}

    def encode(self, text: str) -> list[int]:
        missing = self.oov(text)
        if missing:
            raise KeyError(f"characters not in charset: {sorted(missing)}")
        return [self._index[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.characters[i - 1] for i in ids if i != self.BLANK)

    def ctc_greedy_decode(self, ids: Iterable[int]) -> str:
        """Collapse repeats, then drop blanks."""
        out, prev = [], None
        for i in ids:
            if i != prev and i != self.BLANK:
                out.append(self.characters[i - 1])
            prev = i
        return "".join(out)

    # --------------------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "vocabulary_size": self.num_classes,
            "special_tokens": {"blank": self.BLANK},
            "characters": self.characters,
            "codepoints": [f"U+{ord(c):04X}" for c in self.characters],
            **self.metadata,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: str | Path) -> Charset:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        meta = {
            k: v
            for k, v in data.items()
            if k not in ("version", "vocabulary_size", "special_tokens", "characters", "codepoints")
        }
        cs = cls(list(data["characters"]), int(data["version"]), meta)
        if cs.num_classes != data["vocabulary_size"]:
            raise ValueError("vocabulary_size does not match the character list")
        return cs


def build_charset(
    counts: dict[str, int],
    min_count: int,
    scope: Callable[[str], bool] = in_scope,
) -> tuple[list[str], dict[str, int], dict[str, int]]:
    """Select characters by frequency and scope.

    Returns ``(characters, rare, out_of_scope)``: the kept characters in a stable order (by code
    point, so a re-build with the same data gives the same indices), and the dropped characters with
    their counts, split by reason.
    """
    kept, rare, oos = [], {}, {}
    for ch, n in counts.items():
        if not scope(ch):
            oos[ch] = n
        elif n < min_count:
            rare[ch] = n
        else:
            kept.append(ch)
    return sorted(kept, key=ord), rare, oos


def coverage(counts: dict[str, int], charset: Charset) -> float:
    total = sum(counts.values())
    return sum(n for ch, n in counts.items() if ch in charset) / total if total else 1.0
