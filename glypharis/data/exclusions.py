"""Apply the versioned exclusion lists written by the Phase 3 audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Exclusions:
    dataset: str
    version: int
    images: frozenset[tuple[str, str]]  # (split, image_id)
    lines_all: frozenset[tuple[str, str]]  # (split, line_id) excluded everywhere
    lines_rec: frozenset[tuple[str, str]]  # (split, line_id) excluded from recognizer data only

    @classmethod
    def load(cls, path: str | Path) -> Exclusions:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        lines = doc.get("lines", [])
        return cls(
            dataset=doc["dataset"],
            version=int(doc["version"]),
            images=frozenset((e["split"], e["image_id"]) for e in doc.get("images", [])),
            lines_all=frozenset((e["split"], e["line_id"]) for e in lines if e["scope"] == "all"),
            lines_rec=frozenset(
                (e["split"], e["line_id"]) for e in lines if e["scope"] == "recognizer"
            ),
        )

    @classmethod
    def empty(cls, dataset: str) -> Exclusions:
        return cls(dataset, 0, frozenset(), frozenset(), frozenset())

    def image_excluded(self, split: str, image_id: str) -> bool:
        return (split, image_id) in self.images

    def line_excluded(self, split: str, line_id: str, *, for_recognizer: bool) -> bool:
        key = (split, line_id)
        return key in self.lines_all or (for_recognizer and key in self.lines_rec)
