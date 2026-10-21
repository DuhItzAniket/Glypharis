"""The internal page format shared by every dataset.

One record per page (image), one JSON line per record, gzipped per split::

    {"doc_id": ..., "image": "raw/hiertext/images/train/<id>.jpg", "width": W, "height": H,
     "source": "hiertext", "split": "train",
     "lines": [{"line_id": ..., "polygon": [[x, y] x4], "text_original": ..., "text": ...,
                "legible": bool, "handwritten": bool, "vertical": bool,
                "rec_ok": bool, "rec_skip": null | "<reason>"}]}

``doc_id`` is the grouping unit for splits: all pages of a document stay in one split
(EXPERIMENT_PROTOCOL §1). For HierText one image is one document. ``text`` is the label after
`normalize_label`; ``polygon`` is clipped to the image. ``rec_ok`` says whether the line may be used
to train the recognizer; when it is false, ``rec_skip`` gives the reason.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from glypharis.data.exclusions import Exclusions
from glypharis.data.hiertext import ImageAnnotation
from glypharis.data.text import Charset, normalize_label

PAGE_FORMAT_VERSION = 1

# Reasons a line is not used for recognizer training, in the order they are checked.
REC_SKIP_REASONS = (
    "excluded",  # exclusion list (Phase 3)
    "illegible",
    "handwritten",
    "vertical",  # stacked vertical text is out of scope for line recognition
    "empty",
    "out_of_charset",
    "too_long",  # width / height above the crop limit; squeezing would distort the text
)


@dataclass
class PageLine:
    line_id: str
    polygon: list[list[float]]
    text_original: str
    text: str
    legible: bool
    handwritten: bool
    vertical: bool
    rec_ok: bool
    rec_skip: str | None = None
    # GDC only: local agreement (NCC) between the source sheet and the aligned photo at this line.
    align_score: float | None = None


@dataclass
class Page:
    doc_id: str
    image: str
    width: int
    height: int
    source: str
    split: str
    lines: list[PageLine] = field(default_factory=list)
    # "Don't care" polygons (e.g. raster images whose text has no ground truth). Written only when
    # present, so pages without them serialise exactly as before.
    ignore: list[list[list[float]]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["ignore"]:
            del d["ignore"]
        for line in d["lines"]:
            if line["align_score"] is None:
                del line["align_score"]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Page:
        lines = [PageLine(**ln) for ln in d.get("lines", [])]
        return cls(
            doc_id=d["doc_id"],
            image=d["image"],
            width=int(d["width"]),
            height=int(d["height"]),
            source=d["source"],
            split=d["split"],
            lines=lines,
            ignore=d.get("ignore", []),
        )


def clip_polygon(vertices: Iterable[tuple[float, float]], w: int, h: int) -> list[list[float]]:
    return [
        [min(max(float(x), 0.0), float(w)), min(max(float(y), 0.0), float(h))] for x, y in vertices
    ]


def hiertext_to_page(
    ann: ImageAnnotation,
    split: str,
    image_rel: str,
    charset: Charset,
    excl: Exclusions,
    max_aspect: float = 64.0,
) -> Page | None:
    """Convert one HierText record. Returns None if the whole image is excluded."""
    if excl.image_excluded(split, ann.image_id):
        return None
    page = Page(ann.image_id, image_rel, ann.width, ann.height, "hiertext", split)
    for line in ann.lines:
        if excl.line_excluded(split, line.line_id, for_recognizer=False):
            continue  # broken geometry: dropped for every use, as recorded in the exclusion list
        label = normalize_label(line.text)
        reason: str | None = None
        if excl.line_excluded(split, line.line_id, for_recognizer=True):
            reason = "excluded"
        elif not line.legible:
            reason = "illegible"
        elif line.handwritten:
            reason = "handwritten"
        elif line.vertical:
            reason = "vertical"
        elif not label:
            reason = "empty"
        elif charset.oov(label):
            reason = "out_of_charset"
        elif _aspect(line.vertices) > max_aspect:
            reason = "too_long"
        page.lines.append(
            PageLine(
                line_id=line.line_id,
                polygon=clip_polygon(line.vertices, ann.width, ann.height),
                text_original=line.text,
                text=label,
                legible=line.legible,
                handwritten=line.handwritten,
                vertical=line.vertical,
                rec_ok=reason is None,
                rec_skip=reason,
            )
        )
    return page


def _aspect(v: tuple[tuple[float, float], ...]) -> float:
    if len(v) != 4:
        return 0.0
    h = (math.dist(v[0], v[3]) + math.dist(v[1], v[2])) / 2
    w = (math.dist(v[0], v[1]) + math.dist(v[3], v[2])) / 2
    return w / h if h > 0 else math.inf


def write_pages(path: str | Path, pages: Iterable[Page]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    # No name and mtime=0 in the gzip header: identical content gives identical bytes.
    with (
        open(path, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
    ):
        for page in pages:
            gz.write((json.dumps(page.to_dict(), ensure_ascii=False) + "\n").encode("utf-8"))
            n += 1
    return n


def read_pages(path: str | Path) -> Iterator[Page]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield Page.from_dict(json.loads(line))
