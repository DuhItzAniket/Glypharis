"""Reader for HierText ground truth (paragraph → line → word hierarchy).

The ``*.jsonl.gz`` files hold a single JSON document ``{"info": ..., "annotations": [...]}``. Lines
are 4-point quadrilaterals; words are polygons with 4 or more points. Every line gets a stable id
``<image_id>/p<paragraph>/l<line>`` so audit flags and exclusion lists can refer to it.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Word:
    vertices: tuple[tuple[float, float], ...]
    text: str
    legible: bool
    handwritten: bool
    vertical: bool


@dataclass(frozen=True)
class Line:
    line_id: str
    vertices: tuple[tuple[float, float], ...]
    text: str
    legible: bool
    handwritten: bool
    vertical: bool
    words: tuple[Word, ...]


@dataclass(frozen=True)
class ImageAnnotation:
    image_id: str
    width: int
    height: int
    paragraphs: int
    lines: tuple[Line, ...]


def _vertices(raw: Any) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in raw)


def parse_record(rec: dict[str, Any]) -> ImageAnnotation:
    image_id = str(rec["image_id"])
    lines: list[Line] = []
    paragraphs = rec.get("paragraphs", [])
    for p_idx, para in enumerate(paragraphs):
        for l_idx, line in enumerate(para.get("lines", [])):
            words = tuple(
                Word(
                    vertices=_vertices(w.get("vertices", [])),
                    text=str(w.get("text", "")),
                    legible=bool(w.get("legible", True)),
                    handwritten=bool(w.get("handwritten", False)),
                    vertical=bool(w.get("vertical", False)),
                )
                for w in line.get("words", [])
            )
            lines.append(
                Line(
                    line_id=f"{image_id}/p{p_idx}/l{l_idx}",
                    vertices=_vertices(line.get("vertices", [])),
                    text=str(line.get("text", "")),
                    legible=bool(line.get("legible", True)),
                    handwritten=bool(line.get("handwritten", False)),
                    vertical=bool(line.get("vertical", False)),
                    words=words,
                )
            )
    return ImageAnnotation(
        image_id=image_id,
        width=int(rec["image_width"]),
        height=int(rec["image_height"]),
        paragraphs=len(paragraphs),
        lines=tuple(lines),
    )


def load_split(gt_path: str | Path) -> tuple[dict[str, Any], list[ImageAnnotation]]:
    """Return ``(info, annotations)`` for one ground-truth file."""
    with gzip.open(gt_path, "rt", encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc.get("info", {}), [parse_record(r) for r in doc["annotations"]]


def iter_lines(annotations: list[ImageAnnotation]) -> Iterator[tuple[ImageAnnotation, Line]]:
    for ann in annotations:
        for line in ann.lines:
            yield ann, line


def image_path(raw_root: str | Path, split: str, image_id: str) -> Path:
    return Path(raw_root) / "images" / split / f"{image_id}.jpg"
