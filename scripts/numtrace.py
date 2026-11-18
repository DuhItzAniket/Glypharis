r"""Find the numbers in a Markdown document and in the reports (Phase 21, plan item 6).

    python scripts/numtrace.py DOCUMENT.md [DOCUMENT.md ...]

This is the library behind `experiment_summary.py`, `trace_numbers.py` and `derive_values.py`. It
uses only the standard library and is deterministic. Run directly, it lists what it sees in a
document (numbers, intervals, directives, directive errors), which is the quickest way for an author
to check a directive. Exit status 0 when every directive is valid, 1 otherwise.

Trace directives
================

A directive is an HTML comment, so it does not show when the document is rendered. There are three.
The spelling is frozen: lower-case keyword, a colon, single spaces, the whole comment on one line.
A comment that looks like a directive but is not spelled this way is an error, never ignored.
Directives inside code spans and fenced code are examples and do nothing.

    <!-- trace: GW-REC-040 reports/error_cases/phase15/recognition.md#gdc_test -->
    <!-- trace-table: GW-E2E-007 reports/external/end_to_end.md key=row|key=row,header -->
    <!-- derived: name -->

The ID is a registered experiment ID (``GW-REC-040``). Dataset counts have no experiment: for them,
write the manifest or report path in the ID's place, the same path as the source, without its
anchor::

    <!-- trace: data/manifests/charset.json data/manifests/charset.json#vocabulary_size -->

The source is a path from the repository root, with forward slashes, under ``reports/``,
``research/``, ``docs/``, ``configs/`` or ``data/manifests/`` (never ``docs/handoff/``), ending in
``.md``, ``.json``, ``.csv``, ``.yaml`` or ``.yml``. It may end in ``#anchor``, which narrows it:

- Markdown: one section, from its heading to the next heading of the same or a higher level. The
  anchor is the heading's GitHub slug, an ``<a id="...">`` inside the heading, or the start of the
  slug up to a hyphen, so ``#gdc_test`` finds ``## gdc_test (GW-REC-039)``. An anchor that fits no
  heading, or several, is an error.
- JSON and YAML: one subtree, written as keys joined by ``/`` (``#tracks/strict/totals``); list
  positions count from 0 in JSON. In a key, ``~1`` stands for ``/`` and ``~0`` for ``~``, so
  dots need no escape.
- CSV files take no anchor.

``trace`` stands alone on its line and applies to the next block: the paragraph, whole list or table
that follows, with only blank lines or other directives between. The numbers in that block are
looked up in that source first, under that ID. Several ``trace`` lines may stack on one block.

``trace-table`` stands alone on its line and the next block must be a table. After the source comes
one or more bindings separated by ``|``. A binding is ``key=row`` or ``key=row,header``:

- ``key`` picks what is bound in the document's table: the one row that holds this text in a cell.
  If no row does, ``key`` is a column header (not the first column's) and the binding covers that
  column, which is how a table transposed against its report is bound.
- ``row`` picks one row of the report's table: the one row holding this text in a cell. In a JSON or
  YAML source it is a key path, as in an anchor and relative to the anchor. ``L14`` means the table
  row on line 14 of the report.
- ``header`` is a column header of the report (in JSON, the last key). With it, every number the
  binding covers must come from that one cell.
- Without ``header``, a number is compared with the report column named like its own column (for a
  transposed binding: named like its row's first cell). If the report has no such column, any cell
  of the report row may match.

``key`` and ``row`` may list several texts separated by ``;``, all of which must be in the row
(``GW-E2E-007;gdc_test;strict``). Cells are compared exactly, after removing surrounding bold,
italics and backticks. Spaces around ``|``, ``=``, ``,`` and ``;`` do not count. Write ``\|``,
``\=``, ``\,``, ``\;`` or ``\\`` for those characters inside a text. A row selector that finds no
row or several is an error: add a text or an anchor. Rows of the table no binding names are treated
as under a ``trace`` with the same ID and source.

Example, for a document table with rows ``HierText`` and ``GDC``. The first row is compared column
by column; every number in the second must come from the report's ``CER %`` cell::

    <!-- trace-table: GW-REC-039 reports/external/recognition.md HierText=L15|GDC=L23,CER % -->

``derived`` names an entry of ``reports/paper_trace/derived.json``, and the number it applies to
must equal that entry. Inside a paragraph, list or table cell it applies to the next number after
it in the same block: ``about <!-- derived: fps_ratio --> 3.3 times``. Alone on its line it applies
to the first number of the next block; several in a row take the numbers of that block in order. A
name is letters, digits, ``_``, ``.`` and ``-``, starting with a letter or digit.

What counts as a number
=======================

A number may carry a sign (``-``, the Unicode minus or ``+``), thousands separators (a comma, or a
no-break or thin space, in groups of three), decimals, an exponent (``7.33e-06``, ``1E+3``) and a
unit (``%``, ``ms``, ``s``, ``h``, ``min``, ``x``, ``K``, ``M``, sizes like ``GiB``, ``px``, ``W``,
``fps``, ``points``, ``pp``). ``v [lo, hi]`` is read as an `Interval`, also when it wraps to a
second line. In ``5-10`` the hyphen is a range, so both numbers are positive.

These are skipped: fenced code, code spans, HTML comments and tags, link and image targets (the
link text is kept), list markers, and numbers at the start of a heading. So are all of these, which
are never quantities: GW and other hyphenated IDs (``E2E-001``), finding, hypothesis, question, ADR
and risk numbers (``F3``, ``H1′``, ``RQ8``, ``ADR-0013``, ``R-05``); ``Phase 7``, ``Phases 5, 14 and
22``, ``§13``, ``Table 2``, ``Figure 3``, ``Section 4``, ``item 6``, ``step 2``, ``stage 3``,
``#12``; dates; hashes and any zero-padded integer (``036``); URLs, DOIs, arXiv numbers, emails,
paths and file names; seeds (``seed 20261118``, ``seeds 1, 2 and 3``, ``seed [20261115, 1]``);
versions (``v1``, ``2.14.1``, ``version 1.1``, ``Python 3.11``, ``opset 17``, ``Tesseract 5``);
licences (``CC BY 4.0``, ``GPL-3.0``); digits inside a name (``fp32``, ``INT8``, ``p95``,
``PP-OCRv5``, ``MobileNetV3``, ``95th``); thresholds in a metric name (``F1@0.5``, ``P @0.5``); and
hardware names (``RTX 4050``, ``i7-14650HX``). A number the rules do not recognise as one of these
stays a number, so a mistake here shows up as a failed trace, not as a number nobody checked. To
state a number that is not a quantity, put it in a code span.

The report index
================

`ReportIndex` reads the numbers of a source file, each with its file and line. Markdown is read with
the tokenizer above; JSON, CSV and YAML give only values that are wholly numeric, so strings never
count. Free text is left out, because digits in it would let a wrong number pass:

- a Markdown table with a header cell exactly ``Reference`` or ``Prediction`` (``Reference class``
  tables stay);
- every ``words_*.md`` file;
- ``examples.md``, apart from its counts: the ``(N times on M lines`` of each edit type and the
  cells that are a single number in tables not excluded above.

Files are hashed after turning CRLF into LF, so a hash is the same on every checkout.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import fnmatch
import hashlib
import io
import json
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Final

ALLOWED_ROOTS: Final = ("reports/", "research/", "docs/", "configs/", "data/manifests/")
REFUSED_ROOTS: Final = ("docs/handoff/",)
SOURCE_SUFFIXES: Final = (".md", ".json", ".csv", ".yaml", ".yml")
FREE_TEXT_HEADERS: Final = frozenset({"Reference", "Prediction"})
FREE_TEXT_FILES: Final = ("words_*.md",)
COUNTS_ONLY_FILES: Final = ("examples.md",)
EXPERIMENT_ID: Final = re.compile(r"GW-[A-Z0-9]+-[0-9]{3}")

# Stand-ins written over text the number scan must not read; each keeps its line and column.
_CODE: Final = "\x00"
_COMMENT: Final = "\x01"
_FENCE: Final = "\x02"
_MASKS: Final = frozenset({_CODE, _COMMENT, _FENCE})
_SPACE: Final = " \t\u00a0"
_MINUS: Final = "\u2212"

_WS: Final = r"(?:[ \t\u00a0]*\n[ \t]*(?:>[ \t]*)*|[ \t\u00a0]+)"
_JOIN: Final = (
    rf"(?:{_WS}?[,/&\u2013\u2014-]{_WS}?(?:(?:and|or){_WS})?|{_WS}(?:and|or|to|through){_WS})"
)
_RANGE: Final = rf"(?:{_WS}?[\u2013\u2014-]{_WS}?|{_WS}to{_WS})"
_YEAR: Final = r"(?:19[0-9]{2}|20[0-3][0-9])"
_MONTH: Final = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?"
    r"|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?"
)
_DAY: Final = r"[0-9]{1,2}(?:st|nd|rd|th)?"
# Not the start of a larger number: "step 45,295" is a quantity.
_NOT_THOUSANDS: Final = r"(?![0-9])(?!,[0-9]{3}(?![0-9]))"
_REF: Final = rf"[0-9]{{1,3}}(?:\.[0-9]+)*[a-z]?{_NOT_THOUSANDS}"
_FILE_TYPES: Final = (
    "md|json|ya?ml|csv|tsv|txt|py|toml|png|jpe?g|pdf|onnx|pt|pth|ckpt|bib|tex|html?|log|zip|tar|gz"
    "|whl|safetensors|lock|sh|ps1|bin"
)


def _numbered(singular: str, plural: str, item: str = _REF) -> str:
    """A reference word and its number: one number or range, or a list after the plural."""
    return (
        rf"(?<![\w-])(?i:{plural}){_WS}{item}(?:{_JOIN}{item})*"
        rf"|(?<![\w-])(?i:{singular}){_WS}{item}(?:{_RANGE}{item})?"
    )


def _has_digit(text: str) -> bool:
    return any("0" <= c <= "9" for c in text)


def _is_path(text: str) -> bool:
    """A slash joins names, unless it is a rate or ratio like ``3.3ms/page`` or ``3/4``."""
    if not any(c.isalpha() for c in text) or not _has_digit(text):
        return False
    return not (text[0].isdigit() and len(re.split(r"[/\\]", text)) == 2)


# Applied over the whole text in this order, so a name broken across two lines is still one name.
_IDENTIFIERS: Final[tuple[tuple[re.Pattern[str], Callable[[str], bool] | None], ...]] = tuple(
    (re.compile(pattern), keep)
    for pattern, keep in (
        # Web and mail addresses, DOIs, arXiv numbers.
        (r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s|)\]>]+", None),
        (r"\bwww\.[^\s|)\]>]+", None),
        (r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", None),
        (r"\b10\.[0-9]{4,9}/[^\s|)\]>]+", None),
        (rf"(?i:\barxiv)(?::|{_WS})[ \t]*[0-9]{{4}}\.[0-9]{{4,5}}(?:v[0-9]+)?", None),
        # File names and paths.
        (rf"(?<![\w.-])[\w.+-]*\.(?i:{_FILE_TYPES})\b", None),
        (r"(?<![\w.~-])[\w.~-]+(?:[/\\][\w.~-]+)+", _is_path),
        # Dates.
        (
            r"\b(?:19|20)[0-9]{2}-[01][0-9](?:-[0-3][0-9](?:[T ][0-9]{2}:[0-9:.]+Z?)?)?(?![0-9])",
            None,
        ),
        (r"\b(?:19|20)[0-9]{2}/[01][0-9]/[0-3][0-9]\b", None),
        (r"\b(?:19|20)[0-9]{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])\b", None),
        (rf"\b{_DAY}{_WS}{_MONTH}(?![a-z])(?:,?{_WS}{_YEAR}\b)?", None),
        (rf"\b{_MONTH}{_WS}{_DAY}(?:,?{_WS}{_YEAR})?\b(?![.,]?[0-9])", None),
        (rf"\b{_MONTH}{_WS}{_YEAR}\b", None),
        (r"\b(?:19|20)[0-9]0s\b", None),
        (rf"\bet al\.?,?{_WS}{_YEAR}[a-z]?\b", None),
        (rf"\((?:[A-Z][\w-]+)(?:{_WS}(?:and|&){_WS}[A-Z][\w-]+)?,{_WS}{_YEAR}[a-z]?\)", None),
        (rf"\b(?:[A-Z]{{2,}}|NeurIPS){_WS}{_YEAR}\b(?![.,]?[0-9])", None),
        # Seeds.
        (rf"(?<![\w])(?:--)?(?i:seed)(?:{_WS}of)?(?:{_WS}|[ \t]*[=:][ \t]*)\[[^\]\n]*\]", None),
        (
            rf"(?<![\w])(?:--)?(?i:seeds)(?:{_WS}|[ \t]*[=:][ \t]*)[0-9]+(?:{_JOIN}[0-9]+)*",
            None,
        ),
        (rf"(?<![\w])(?:--)?(?i:seed)(?:{_WS}of)?(?:{_WS}|[ \t]*[=:][ \t]*)[0-9]+", None),
        # Licences, versions and products. A backend name alone is not here: "ONNX 9.82" is a value.
        (rf"\bCC(?:{_WS}|-)?BY(?:[ -](?:SA|NC|ND))*{_WS}[0-9]\.[0-9]", None),
        (rf"\bCC0{_WS}[0-9]\.[0-9]", None),
        (rf"\b(?:A?GPL|LGPL|FDL)(?:v|{_WS})[0-9](?:\.[0-9])?", None),
        (rf"\bApache(?:{_WS}License)?(?:{_WS}Version)?{_WS}[0-9]\.[0-9]", None),
        (r"(?<![0-9.])[0-9]+\.[0-9]+\.[0-9]+(?:[.+-]?[0-9A-Za-z]+)*", None),
        (rf"(?<![\w-])(?i:version|release){_WS}[vV]?[0-9]+(?:\.[0-9]+)*", None),
        (rf"\bPython{_WS}3(?:\.[0-9]+)*", None),
        (rf"\bopset{_WS}[0-9]+", None),
        (rf"\bTesseract{_WS}[0-9](?![0-9.,])", None),
        (rf"\bCUDA{_WS}[0-9]+\.[0-9]+", None),
        (rf"\bWindows{_WS}1[01]\b", None),
        (rf"\bUbuntu{_WS}[0-9]{{2}}\.[0-9]{{2}}", None),
        (rf"\bUnicode{_WS}[0-9]+(?:\.[0-9]+)*", None),
        (rf"\bISO{_WS}[0-9]+", None),
        (
            rf"\b(?:RTX|GTX|Quadro|Radeon|Ryzen|Snapdragon|Pixel|Galaxy){_WS}[A-Z]?[0-9]+[\w-]*",
            None,
        ),
        (rf"\bCore{_WS}(?:Ultra{_WS})?[0-9]\b", None),
        # A threshold in a metric name.
        (r"@[ \t]?\[?[0-9][0-9.:]*\]?", None),
        # References to parts of the project or the document.
        (_numbered("Phase", "Phases", rf"[0-9]{{1,2}}[a-z]?{_NOT_THOUSANDS}"), None),
        (_numbered("Section", "Sections"), None),
        (_numbered("Table", "Tables"), None),
        (_numbered("Figure", "Figures"), None),
        (_numbered("Item", "Items"), None),
        (_numbered("Step", "Steps"), None),
        (_numbered("Stage", "Stages"), None),
        (_numbered("Chapter", "Chapters"), None),
        (_numbered("Equation", "Equations"), None),
        (_numbered("Algorithm", "Algorithms"), None),
        (rf"\b(?i:fig|eq)\.{_WS}{_REF}", None),
        (rf"§§{_WS}?{_REF}(?:{_JOIN}§?{_REF})*", None),
        (rf"§{_WS}?{_REF}", None),
        (rf"\b(?:ADR|RQ){_WS}[0-9]+", None),
        (r"(?<![\w#])#[0-9]+\b", None),
        # A name with digits joined by hyphens: GW-REC-040, PP-OCRv5, i7-14650HX, top-5.
        (r"(?<![\w.-])[^\W\d_][\w]*(?:-\w+(?:\.[0-9]+)*)+", _has_digit),
    )
)

_LINK_TARGET: Final = re.compile(
    r"(?<=\])\((?:[^()\s]|\([^()\s]*\))*(?:\s+(?:\"[^\"\n]*\"|'[^'\n]*'))?\)"
)
_LINK_LABEL: Final = re.compile(r"(?<=\])\[[^\]\n]*\]")
_FOOTNOTE: Final = re.compile(r"\[\^[^\]\n]+\]")
_LINK_DEFINITION: Final = re.compile(r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*\S.*$", re.M)
_HTML_TAG: Final = re.compile(r"</?[a-zA-Z][^<>\n]*>")

_LITERAL: Final = re.compile(r"\\.|<!--|`+")
_BACKTICKS: Final = re.compile(r"`+")
_FENCE_OPEN: Final = re.compile(r"[ \t>]*(`{3,}(?=[^`\n]*$)|~{3,})", re.M)
_BLANK_LINE: Final = re.compile(r"\n[ \t]*\n")
_QUOTE: Final = re.compile(r"[ \t]{0,3}(?:>[ \t]?)+")
_ATX: Final = re.compile(r"[ ]{0,3}(#{1,6})(?:[ \t]+|$)")
_HEADING_NUMBER: Final = re.compile(r"[0-9]+(?:\.[0-9]+)*[.)]?(?=[ \t]|$)")
_THEMATIC: Final = re.compile(r"([-*_=])(?:[ \t]*\1){2,}")
_LIST_MARKER: Final = re.compile(r"[ \t]*(?:[-*+]|(?P<number>[0-9]{1,9})[.)])(?=[ \t]|$)")
_DELIMITER_CELL: Final = re.compile(r":?-+:?")

_NUMBER: Final = re.compile(
    r"(?P<sign>[+\-\u2212])?"
    r"(?P<body>(?:[1-9][0-9]{0,2}(?P<sep>[,\u00a0\u2009\u202f])[0-9]{3}(?:(?P=sep)[0-9]{3})*"
    r"(?![0-9])|[0-9]+)(?:\.[0-9]+)?|\.[0-9]+)"
    r"(?P<exp>[eE](?:[+\-\u2212][0-9]{1,3}|[0-9]{1,2})(?![0-9A-Za-z]))?"
)
_LETTERS: Final = re.compile(r"[^\W_]+")
_SIZE_UNITS: Final = ("KB", "KiB", "MB", "MiB", "GB", "GiB", "TB", "TiB")
_ATTACHED_UNITS: Final = frozenset(
    ("ms", "s", "h", "min", "x", "X", "K", "k", "M", "px", "W", "fps", *_SIZE_UNITS)
)
_SPACED_UNIT: Final = re.compile(
    r"[ \t\u00a0]?(?P<percent>%)"
    r"|[ \t\u00a0](?P<word>ms|s|h|min|px|W|fps|KB|KiB|MB|MiB|GB|GiB|TB|TiB|pp|points?"
    r"|percentage points|percent)(?![^\W_])"
)
_UNIT_NAMES: Final = {
    "X": "x",
    "\u00d7": "x",
    "percent": "%",
    "point": "points",
    "percentage points": "points",
}
_INTERVAL_JOIN: Final = re.compile(r"\s*(?:[,;]|\u2013|\u2014|to(?=\s))\s*")
_POINT_GAP: Final = re.compile(r"[\s*_]*\[\s*")
_INTERVAL_END: Final = re.compile(r"\s*\](?![(\x00])")  # not link text

_COMMENT_NEAR: Final = re.compile(r"<!--\s*(?:trace-table|trace|derived)(?:\s*:|\s+GW-)", re.I)
_TRACE: Final = re.compile(r"<!-- trace: (\S+) (\S+) -->")
_TRACE_TABLE: Final = re.compile(r"<!-- trace-table: (\S+) (\S+) (\S.*?) -->")
_DERIVED: Final = re.compile(r"<!-- derived: ([A-Za-z0-9][A-Za-z0-9_.-]*) -->")
_ID_ONLY: Final = re.compile(r"GW-[A-Z0-9]+-[0-9]{3}")
_SOURCE_CHARS: Final = re.compile(r"[A-Za-z0-9._/-]+")
_LINE_ROW: Final = re.compile(r"L([0-9]+)")
_SOURCE_MENTION: Final = re.compile(
    r"(?<![\w/.-])[\w./-]+\.(?:md|json|csv|ya?ml)(?:#[\w./~-]+)?(?![\w])"
)
_ANCHOR_ID: Final = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"\s*>", re.I)
_SLUG_DROP: Final = re.compile(r"[^\w\- ]")
_CELL_ESCAPE: Final = re.compile(r"\\([\\`*_|])")
_EXAMPLE_COUNTS: Final = re.compile(r"\*\* \(([0-9][0-9,]*) times on ([0-9][0-9,]*) lines?\b")
_GENERATED_BY: Final = re.compile(r"Generated by\b.*")
_JSON_TOKEN: Final = re.compile(
    r"\s+|\"(?:[^\"\\]|\\.)*\"|-?[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?"
    r"|true|false|null|-?Infinity|NaN|[{}\[\]:,]",
    re.S,
)
_YAML_LINE: Final = re.compile(
    r"(?P<indent> *)(?P<dash>- +)?(?:(?P<key>[\w.-]+):(?: +|$))?(?P<value>.*)$"
)


class TraceError(ValueError):
    """A source, anchor or binding cannot be resolved."""


class SourcePathError(TraceError):
    """A source path outside what the tools may read."""


class DirectiveError(TraceError):
    """A malformed or misplaced directive, with the document line it is on."""

    def __init__(self, line: int, message: str) -> None:
        super().__init__(f"line {line}: {message}")
        self.line = line
        self.message = message


def significant_digits(value: Decimal) -> int:
    """How many significant digits a number was printed with (``0.50`` has 2, ``100`` has 3)."""
    digits = "".join(map(str, value.as_tuple().digits)).lstrip("0")
    return max(len(digits), 1)


def half_unit(exponent: int) -> Fraction:
    """Half a unit of the last printed digit, for a number whose last digit is 10**exponent."""
    return Fraction(10) ** exponent / 2


def rounds_to(source: Decimal, printed: Decimal, exponent: int) -> bool:
    """True when ``source`` is within half a unit of ``printed``'s last digit, inclusive. Exact."""
    return abs(Fraction(source) - Fraction(printed)) <= half_unit(exponent)


@dataclass(frozen=True)
class CellRef:
    """Where a number sits in a table. ``row`` is -1 in the header row."""

    table: int
    row: int
    column: int
    header: str


@dataclass(frozen=True)
class NumberToken:
    """One number in a document. Lines and columns count from 1; ``end_column`` is exclusive.

    ``raw`` is the text as printed, without the unit; ``value`` is signed and exact; the last
    printed digit is worth 10**``exponent``. ``block`` indexes `Document.blocks`, ``derived`` is the
    name of the `derived` directive bound to this number.
    """

    line: int
    column: int
    end_column: int
    raw: str
    sign: str
    value: Decimal
    exponent: int
    unit: str
    block: int
    cell: CellRef | None = None
    derived: str | None = None

    @property
    def significant_digits(self) -> int:
        return significant_digits(self.value)


@dataclass(frozen=True)
class Interval:
    """``point [low, high]``; ``point`` is None when no number comes right before the bracket."""

    point: NumberToken | None
    low: NumberToken
    high: NumberToken


@dataclass(frozen=True)
class Binding:
    """One ``key=row`` or ``key=row,header`` of a `trace-table` directive."""

    key: tuple[str, ...]
    row: tuple[str, ...]
    header: str | None = None


@dataclass(frozen=True)
class Directive:
    """A parsed directive. ``kind`` is ``trace``, ``trace-table`` or ``derived``.

    ``experiment`` is a GW ID, or the source path where a path stands in for one. ``target`` indexes
    `Document.blocks`; ``inline`` is True for a `derived` written inside a block.
    """

    kind: str
    line: int
    target: int = -1
    inline: bool = False
    experiment: str = ""
    path: str = ""
    anchor: str | None = None
    bindings: tuple[Binding, ...] = ()
    name: str = ""


@dataclass(frozen=True)
class Heading:
    line: int
    level: int
    text: str
    slug: str
    ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Block:
    """A heading, paragraph, whole list, table or fenced code block.

    ``section`` indexes `Document.headings` (-1 before the first heading). ``ids`` and ``sources``
    are the experiment IDs and file names written in the block, code spans and links included.
    """

    index: int
    kind: str
    start_line: int
    end_line: int
    section: int
    ids: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class TableRow:
    index: int
    line: int
    cells: tuple[str, ...]


@dataclass(frozen=True)
class Table:
    index: int
    block: int
    line: int
    header: tuple[str, ...]
    rows: tuple[TableRow, ...]


@dataclass(frozen=True)
class Document:
    """What `parse_document` finds. ``errors`` holds every invalid directive."""

    path: str
    tokens: tuple[NumberToken, ...]
    intervals: tuple[Interval, ...]
    directives: tuple[Directive, ...]
    blocks: tuple[Block, ...]
    tables: tuple[Table, ...]
    headings: tuple[Heading, ...]
    errors: tuple[DirectiveError, ...]


@dataclass(frozen=True)
class ReportValue:
    """One number in a source file.

    ``where`` is the section slug in Markdown and the key path in JSON and YAML. ``header`` is the
    table column or the last key. ``table`` indexes `IndexedFile.tables` (-1 outside a table).
    """

    file: str
    line: int
    raw: str
    value: Decimal
    exponent: int
    unit: str = ""
    where: str = ""
    header: str = ""
    table: int = -1

    @property
    def significant_digits(self) -> int:
        return significant_digits(self.value)


@dataclass(frozen=True)
class ReportInterval:
    """A ``value [low, high]`` on one source line, or a JSON object with those three keys.

    ``unit`` is what a JSON interval was resampled over (``pages``, ``lines``, ...), else empty.
    """

    file: str
    line: int
    point: ReportValue | None
    low: ReportValue
    high: ReportValue
    unit: str = ""
    where: str = ""


@dataclass(frozen=True)
class IndexedFile:
    """A source file's numbers. ``generated_ids`` come from its header or JSON run field."""

    path: str
    sha256: str
    kind: str
    line_count: int
    values: tuple[ReportValue, ...]
    intervals: tuple[ReportInterval, ...] = ()
    tables: tuple[Table, ...] = ()
    headings: tuple[Heading, ...] = ()
    generated_ids: tuple[str, ...] = ()
    ids: tuple[str, ...] = ()

    def _in_scope(self, anchor: str | None) -> Callable[[int, str], bool]:
        if not anchor:
            return lambda _line, _where: True
        if self.kind == "md":
            first, last = section_lines(self.headings, anchor, self.line_count, self.path)
            return lambda line, _where: first <= line <= last
        if self.kind == "csv":
            raise TraceError(f"{self.path}: a CSV source takes no anchor")
        if not any(_under(v.where, anchor) for v in self.values):
            raise TraceError(f"{self.path}: no values under #{anchor}")
        return lambda _line, where: _under(where, anchor)

    def values_in(self, anchor: str | None = None) -> tuple[ReportValue, ...]:
        keep = self._in_scope(anchor)
        return tuple(v for v in self.values if keep(v.line, v.where))

    def intervals_in(self, anchor: str | None = None) -> tuple[ReportInterval, ...]:
        keep = self._in_scope(anchor)
        return tuple(i for i in self.intervals if keep(i.line, i.where))

    def tables_in(self, anchor: str | None = None) -> tuple[Table, ...]:
        keep = self._in_scope(anchor)
        return tuple(t for t in self.tables if keep(t.line, ""))


def _under(where: str, anchor: str) -> bool:
    return where == anchor or where.startswith(anchor + "/")


def slug(text: str) -> str:
    """GitHub's heading slug: lower case, punctuation dropped, spaces to hyphens."""
    return _SLUG_DROP.sub("", text.strip().lower()).replace(" ", "-")


def section_lines(
    headings: Sequence[Heading], anchor: str, line_count: int, path: str = ""
) -> tuple[int, int]:
    """First and last line of the section an anchor names; see the module docstring."""
    wanted = slug(anchor)
    found = [h for h in headings if h.slug == wanted or anchor in h.ids or wanted in h.ids]
    if not found:
        found = [h for h in headings if h.slug.startswith(wanted + "-")]
    if len(found) != 1:
        what = "no heading" if not found else f"{len(found)} headings"
        raise TraceError(f"{path}: #{anchor} fits {what}")
    heading = found[0]
    last = line_count
    for other in headings:
        if other.line > heading.line and other.level <= heading.level:
            last = other.line - 1
            break
    return heading.line, last


def clean_cell(text: str) -> str:
    """A table cell as compared by bindings: no surrounding bold, italics or backticks."""
    cell = "".join(c for c in text if c not in _MASKS).strip()
    changed = True
    while changed:
        changed = False
        for mark in ("**", "`", "*"):
            size = len(mark)
            if len(cell) > 2 * size and cell.startswith(mark) and cell.endswith(mark):
                cell = cell[size:-size].strip()
                changed = True
    return _CELL_ESCAPE.sub(r"\1", cell)


def _split_cells(line: str, base: int) -> list[tuple[int, int]]:
    """Offsets of a table row's cells; an escaped pipe does not end a cell."""
    edges = [i for i, c in enumerate(line) if c == "|" and (i == 0 or line[i - 1] != "\\")]
    stripped = line.strip()
    start = 0
    spans: list[tuple[int, int]] = []
    if stripped.startswith("|"):
        start = edges.pop(0) + 1
    closing = stripped.endswith("|") and not stripped.endswith("\\|")
    for edge in edges:
        spans.append((base + start, base + edge))
        start = edge + 1
    if not closing:
        spans.append((base + start, base + len(line)))
    return spans


def _is_delimiter(line: str) -> bool:
    if "-" not in line or "|" not in line:
        return False
    cells = [line[a:b].strip() for a, b in _split_cells(line, 0)]
    return bool(cells) and all(_DELIMITER_CELL.fullmatch(c) for c in cells)


def _plain_number(match: re.Match[str]) -> tuple[Decimal, int]:
    """The exact value and last-digit exponent of a match of `_NUMBER`."""
    body = re.sub(r"[,\u00a0\u2009\u202f]", "", match["body"])
    text = body + (match["exp"] or "").replace(_MINUS, "-")
    if (match["sign"] or "") in ("-", _MINUS):
        text = "-" + text
    value = Decimal(text)
    exponent = value.as_tuple().exponent
    assert isinstance(exponent, int)  # the pattern admits finite numbers only
    return value, exponent


def _whole_number(text: str) -> tuple[str, Decimal, int, str] | None:
    """(raw, value, exponent, unit) when a string is one number and nothing else."""
    stripped = text.strip()
    unit = ""
    if stripped.endswith("%"):
        stripped, unit = stripped[:-1].rstrip(), "%"
    match = _NUMBER.fullmatch(stripped)
    if match is None or re.match(r"0[0-9]", match["body"]):
        return None
    value, exponent = _plain_number(match)
    return stripped, value, exponent, unit


def _name_on_left(text: str, index: int) -> bool:
    """True when the character before ``index`` joins the number to a name."""
    if index == 0:
        return False
    left = text[index - 1]
    if left == _CODE or left == ".":  # a comment beside a number joins nothing
        return True
    if left == "_":  # emphasis when nothing word-like stands before it
        return index >= 2 and text[index - 2].isalnum()
    if left in "xX" and index >= 2 and text[index - 2].isdigit():
        return False  # the second side of 48x320
    return left.isalnum()


def _unit_on_right(text: str, end: int) -> tuple[str, int] | None:
    """The unit after a number and where it ends, or None when a name goes on from here."""
    next_char = text[end] if end < len(text) else ""
    after = text[end + 1] if end + 1 < len(text) else ""
    if next_char == _CODE or (next_char == "." and after.isdigit()):
        return None
    if next_char == "_" and after.isalnum():
        return None
    if next_char == "\u00d7":
        return ("", end) if after.isdigit() else ("x", end + 1)
    if next_char.isalnum():
        word = _LETTERS.match(text, end)
        assert word is not None
        if word[0] in _ATTACHED_UNITS:
            unit = word[0]
            return _UNIT_NAMES.get(unit, unit), word.end()
        if next_char in "xX" and after.isdigit():
            return "", end
        return None
    spaced = _SPACED_UNIT.match(text, end)
    if spaced is None:
        return "", end
    unit = "%" if spaced["percent"] else spaced["word"]
    return _UNIT_NAMES.get(unit, unit), spaced.end()


@dataclass(frozen=True)
class _Found:
    start: int
    end: int
    unit_end: int
    raw: str
    sign: str
    value: Decimal
    exponent: int
    unit: str


def _find_numbers(text: str) -> list[_Found]:
    found: list[_Found] = []
    for match in _NUMBER.finditer(text):
        start = match.start()
        sign = match["sign"] or ""
        if sign and start > 0 and (text[start - 1].isalnum() or text[start - 1] in ")]%'"):
            # A hyphen between two values is a range; after a name it was masked already.
            if text[start - 1].isalpha():
                continue
            start, sign = match.start("body"), ""
        elif sign and start > 0 and text[start - 1] == _CODE:
            continue
        if _name_on_left(text, start):
            continue
        body = match["body"]
        exp = match["exp"]
        if re.match(r"0[0-9]", body):
            continue  # zero-padded: an ID or a code, not a quantity
        if exp and exp[1].isdigit() and "." not in body and len(body) + len(exp) > 6:
            continue  # all hexadecimal digits with one e: part of a hash
        unit = _unit_on_right(text, match.end())
        if unit is None:
            continue
        value, exponent = _plain_number(match)
        if not sign and match["sign"]:
            value = abs(value)
        found.append(
            _Found(
                start,
                match.end(),
                unit[1],
                text[start : match.end()],
                sign.replace(_MINUS, "-"),
                value,
                exponent,
                unit[0],
            )
        )
    return found


def _mask_span(chars: list[str], start: int, end: int, mark: str) -> None:
    for i in range(start, end):
        if chars[i] != "\n":
            chars[i] = mark


def _mask_identifiers(text: str) -> str:
    chars = list(text)

    def apply(pattern: re.Pattern[str], keep: Callable[[str], bool] | None, source: str) -> None:
        for match in pattern.finditer(source):
            if keep is None or keep(match[0]):
                _mask_span(chars, match.start(), match.end(), _CODE)

    for pattern in (_LINK_TARGET, _LINK_LABEL, _FOOTNOTE, _LINK_DEFINITION, _HTML_TAG):
        apply(pattern, None, text)
    current = "".join(chars)
    for pattern, keep in _IDENTIFIERS:
        apply(pattern, keep, current)
        current = "".join(chars)
    return current


def _unescape_split(text: str, separator: str) -> list[str]:
    """Split on an unescaped separator and resolve the escapes in each part."""
    parts: list[str] = []
    current: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text) and text[i + 1] in "\\|=,;":
            current.append(c + text[i + 1])
            i += 2
            continue
        if c == separator:
            parts.append("".join(current))
            current = []
        else:
            current.append(c)
        i += 1
    parts.append("".join(current))
    return parts


def _unescape(text: str) -> str:
    return re.sub(r"\\([\\|=,;])", r"\1", text).strip()


def _terms(text: str) -> tuple[str, ...]:
    return tuple(_unescape(part) for part in _unescape_split(text, ";"))


def parse_bindings(text: str) -> tuple[Binding, ...]:
    """The bindings of a `trace-table` directive. Raises ValueError with the reason."""
    bindings: list[Binding] = []
    for part in _unescape_split(text, "|"):
        sides = _unescape_split(part, "=")
        if len(sides) != 2:
            raise ValueError(f"binding {part.strip()!r} is not key=row or key=row,header")
        target = _unescape_split(sides[1], ",")
        if len(target) > 2:
            raise ValueError(f"binding {part.strip()!r} has more than one comma; escape it as \\,")
        key, row = _terms(sides[0]), _terms(target[0])
        header = _unescape(target[1]) if len(target) == 2 else None
        if not all(key) or not all(row) or header == "":
            raise ValueError(f"binding {part.strip()!r} has an empty part")
        bindings.append(Binding(key, row, header))
    keys = [b.key for b in bindings]
    if len(set(keys)) != len(keys):
        raise ValueError("two bindings use the same key")
    return tuple(bindings)


def check_source_path(path: str) -> str:
    """Return ``path`` when the tools may read it; raise `SourcePathError` otherwise."""
    if not _SOURCE_CHARS.fullmatch(path):
        raise SourcePathError(f"{path!r}: use a repository path with forward slashes")
    parts = path.split("/")
    if path.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise SourcePathError(f"{path!r}: not a plain path from the repository root")
    if not path.startswith(ALLOWED_ROOTS) or path.startswith(REFUSED_ROOTS):
        raise SourcePathError(f"{path!r}: outside {', '.join(ALLOWED_ROOTS)}")
    if PurePosixPath(path).suffix not in SOURCE_SUFFIXES:
        raise SourcePathError(f"{path!r}: not one of {', '.join(SOURCE_SUFFIXES)}")
    return path


def _parse_source(experiment: str, source: str) -> tuple[str, str | None]:
    path, hash_mark, anchor = source.partition("#")
    if hash_mark and not anchor:
        raise ValueError(f"{source!r} has an empty anchor")
    try:
        check_source_path(path)
    except SourcePathError as error:
        raise ValueError(str(error)) from error
    if not _ID_ONLY.fullmatch(experiment) and experiment != path:
        raise ValueError(f"{experiment!r} is neither a GW ID nor the source path")
    return path, anchor or None


def parse_directive(comment: str, line: int = 1) -> Directive:
    """Parse one comment, ``<!-- ... -->``, into a `Directive` or raise `DirectiveError`."""
    try:
        if match := _TRACE.fullmatch(comment):
            path, anchor = _parse_source(match[1], match[2])
            return Directive("trace", line, experiment=match[1], path=path, anchor=anchor)
        if match := _TRACE_TABLE.fullmatch(comment):
            path, anchor = _parse_source(match[1], match[2])
            bindings = parse_bindings(match[3])
            return Directive(
                "trace-table",
                line,
                experiment=match[1],
                path=path,
                anchor=anchor,
                bindings=bindings,
            )
    except ValueError as error:
        raise DirectiveError(line, str(error)) from error
    if match := _DERIVED.fullmatch(comment):
        return Directive("derived", line, name=match[1])
    raise DirectiveError(line, f"not a valid directive: {comment[:80]}")


class _Parser:
    """One pass over a document; `parse_document` is the public entry."""

    def __init__(self, text: str, path: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.text = text.translate({ord(m): " " for m in _MASKS})
        self.path = path
        self.chars = list(self.text)
        self.visible = list(self.text)  # the text with only comments hidden
        self.comments: list[tuple[int, int]] = []
        self.errors: list[DirectiveError] = []
        self.starts = [0]
        self.starts += [i + 1 for i, c in enumerate(self.text) if c == "\n"]
        self.blocks: list[dict[str, int | str]] = []
        self.block_of: dict[int, int] = {}
        self.headings: list[Heading] = []
        self.tables: list[Table] = []
        self.cells: dict[int, tuple[int, int, list[tuple[int, int]]]] = {}

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self.starts, offset)

    def _fence_end(self, line_end: int, fence: str) -> int:
        mark, size = fence[0], len(fence)
        closing = re.compile(rf"[ \t>]*{re.escape(mark)}{{{size},}}[ \t]*$", re.M)
        pos = line_end + 1
        while pos <= len(self.text):
            end = self.text.find("\n", pos)
            end = len(self.text) if end < 0 else end
            if closing.match(self.text, pos, end):
                return end
            pos = end + 1
        return len(self.text)

    def _code_span_end(self, start: int, end: int, line_end: int) -> int | None:
        line_start = self.text.rfind("\n", 0, start) + 1
        if self.text[line_start:line_end].lstrip().startswith("|"):
            limit = line_end
        else:
            blank = _BLANK_LINE.search(self.text, end)
            limit = blank.start() if blank else len(self.text)
        for closing in _BACKTICKS.finditer(self.text, end, limit):
            if len(closing[0]) == end - start:
                return closing.end()
        return None

    def scan_literals(self) -> None:
        """Hide fenced code, comments and code spans, in the order they open."""
        text = self.text
        pos = 0
        while pos < len(text):
            line_end = text.find("\n", pos)
            line_end = len(text) if line_end < 0 else line_end
            if pos == 0 or text[pos - 1] == "\n":
                fence = _FENCE_OPEN.match(text, pos, line_end)
                if fence:
                    end = self._fence_end(line_end, fence[1])
                    _mask_span(self.chars, pos, end, _FENCE)
                    pos = end
                    continue
            hit = _LITERAL.search(text, pos, line_end)
            if hit is None:
                pos = line_end + 1
            elif hit[0].startswith("\\"):
                pos = hit.end()
            elif hit[0] == "<!--":
                close = text.find("-->", hit.end())
                if close < 0:
                    self.errors.append(
                        DirectiveError(self.line_of(hit.start()), "comment is never closed")
                    )
                end = len(text) if close < 0 else close + 3
                self.comments.append((hit.start(), end))
                _mask_span(self.chars, hit.start(), end, _COMMENT)
                _mask_span(self.visible, hit.start(), end, _COMMENT)
                pos = end
            else:
                end_span = self._code_span_end(hit.start(), hit.end(), line_end)
                if end_span is None:
                    pos = hit.end()
                else:
                    _mask_span(self.chars, hit.start(), end_span, _CODE)
                    pos = end_span

    def _open(self, kind: str, line: int) -> int:
        index = len(self.blocks)
        section = len(self.headings) - 1
        self.blocks.append({"kind": kind, "start": line, "end": line, "section": section})
        self.block_of[line] = index
        return index

    def _extend(self, index: int, line: int) -> None:
        self.blocks[index]["end"] = line
        self.block_of[line] = index

    def _kind(self, index: int | None) -> str:
        return "" if index is None else str(self.blocks[index]["kind"])

    def _cell_texts(self, spans: list[tuple[int, int]]) -> tuple[str, ...]:
        return tuple(clean_cell("".join(self.visible[a:b])) for a, b in spans)

    def _read_heading(self, number: int, base: int, body: str, marks: re.Match[str]) -> None:
        start = base + marks.end()
        rest = body[marks.end() :]
        leading = _HEADING_NUMBER.match(rest)
        if leading:
            _mask_span(self.chars, start, start + leading.end(), " ")
        title = "".join(self.visible[start : base + len(body)])
        title = "".join(c for c in title if c not in _MASKS).strip().rstrip("#").strip()
        ids = tuple(_ANCHOR_ID.findall(title))
        plain = _HTML_TAG.sub("", title).strip()
        block = self._open("heading", number)
        self.blocks[block]["section"] = len(self.headings)
        self.headings.append(Heading(number, len(marks[1]), plain, slug(plain), ids))

    def _read_table(self, first: int, lines: list[str]) -> int:
        """Record the table starting at line index ``first``; return the index after it."""
        block = self._open("table", first + 1)
        table_index = len(self.tables)
        header_spans = self._row_spans(first, lines)
        header = self._cell_texts(header_spans)
        self.cells[first + 1] = (table_index, -1, header_spans)
        self._extend(block, first + 2)
        rows: list[TableRow] = []
        i = first + 2
        while i < len(lines):
            body = self._body(lines[i])[1]
            if not body.strip() or "|" not in body or set(body.strip()) == {_COMMENT}:
                break
            spans = self._row_spans(i, lines)
            self.cells[i + 1] = (table_index, len(rows), spans)
            rows.append(TableRow(len(rows), i + 1, self._cell_texts(spans)))
            self._extend(block, i + 1)
            i += 1
        self.tables.append(Table(table_index, block, first + 1, header, tuple(rows)))
        return i

    @staticmethod
    def _body(line: str) -> tuple[int, str]:
        quote = _QUOTE.match(line)
        base = quote.end() if quote else 0
        return base, line[base:]

    def _row_spans(self, index: int, lines: list[str]) -> list[tuple[int, int]]:
        base, body = self._body(lines[index])
        return _split_cells(body, self.starts[index] + base)

    def scan_blocks(self) -> None:
        """Divide the lines into blocks; hide list markers, quote marks and heading numbers."""
        lines = "".join(self.chars).split("\n")
        current: int | None = None
        blank = False
        i = 0
        while i < len(lines):
            base, body = self._body(lines[i])
            offset = self.starts[i] + base
            _mask_span(self.chars, self.starts[i], offset, " ")
            stripped = body.strip()
            number = i + 1
            marks = set(stripped)
            if not stripped:
                current = current if self._kind(current) == "list" else None
                blank = True
            elif marks == {_COMMENT}:
                current = None
            elif marks == {_FENCE}:
                if self._kind(current) == "code" and current is not None:
                    self._extend(current, number)
                else:
                    current = self._open("code", number)
            elif heading := _ATX.match(body):
                self._read_heading(number, offset, body, heading)
                current = None
            elif _THEMATIC.fullmatch(stripped):
                current = None
            elif (
                "|" in body
                and i + 1 < len(lines)
                and _is_delimiter(self._body(lines[i + 1])[1])
                and len(_split_cells(body, 0)) == len(_split_cells(self._body(lines[i + 1])[1], 0))
            ):
                i = self._read_table(i, lines)
                current = None
                blank = False
                continue
            else:
                current = self._text_line(current, number, offset, body, blank)
                blank = False
                i += 1
                continue
            if stripped:
                blank = False
            i += 1

    def _text_line(
        self, current: int | None, number: int, offset: int, body: str, blank: bool
    ) -> int:
        kind = self._kind(current)
        marker = _LIST_MARKER.match(body)
        # An ordered marker can interrupt a paragraph only when it is 1 (CommonMark).
        if marker and marker["number"] and kind == "paragraph" and int(marker["number"]) != 1:
            marker = None
        if marker:
            _mask_span(self.chars, offset, offset + marker.end(), " ")
        continues_list = kind == "list" and (marker or body[:1] in (" ", "\t") or not blank)
        if current is not None and (continues_list or kind == "paragraph"):
            self._extend(current, number)
            return current
        return self._open("list" if marker else "paragraph", number)

    def _position(self, offset: int) -> tuple[int, int]:
        line = self.line_of(offset)
        return line, offset - self.starts[line - 1] + 1

    def _cell_at(self, offset: int) -> CellRef | None:
        line = self.line_of(offset)
        entry = self.cells.get(line)
        if entry is None:
            return None
        table, row, spans = entry
        for column, (start, end) in enumerate(spans):
            if start <= offset < end:
                header = self.tables[table].header
                return CellRef(table, row, column, header[column] if column < len(header) else "")
        return None

    def _finished_blocks(self) -> tuple[Block, ...]:
        visible = "".join(self.visible).split("\n")
        done: list[Block] = []
        for index, block in enumerate(self.blocks):
            start, end = int(block["start"]), int(block["end"])
            text = "\n".join(visible[start - 1 : end])
            done.append(
                Block(
                    index,
                    str(block["kind"]),
                    start,
                    end,
                    int(block["section"]),
                    tuple(sorted(set(EXPERIMENT_ID.findall(text)))),
                    tuple(sorted(set(_SOURCE_MENTION.findall(text)))),
                )
            )
        return tuple(done)

    def _next_block(self, line: int, blocks: Sequence[Block]) -> int:
        for block in blocks:
            if block.start_line > line:
                return block.index
        return -1

    def _read_directives(self, blocks: Sequence[Block]) -> list[tuple[Directive, int]]:
        lines = "".join(self.chars).split("\n")
        found: list[tuple[Directive, int]] = []
        for start, end in self.comments:
            comment = self.text[start:end]
            if not _COMMENT_NEAR.match(comment):
                continue
            number = self.line_of(start)
            try:
                directive = self._place(parse_directive(comment, number), lines, blocks)
            except DirectiveError as error:
                self.errors.append(error)
                continue
            found.append((directive, end))
        return found

    def _place(
        self, directive: Directive, lines: Sequence[str], blocks: Sequence[Block]
    ) -> Directive:
        number = directive.line
        alone = set(self._body(lines[number - 1])[1].strip()) == {_COMMENT}
        if not alone:
            if directive.kind != "derived":
                raise DirectiveError(number, f"{directive.kind} must stand alone on its line")
            return replace(directive, target=self.block_of.get(number, -1), inline=True)
        target = self._next_block(number, blocks)
        if target < 0:
            raise DirectiveError(number, f"{directive.kind} has no block after it")
        kind = blocks[target].kind
        if kind in ("heading", "code"):
            raise DirectiveError(number, f"{directive.kind} is followed by a {kind}")
        if directive.kind == "trace-table" and kind != "table":
            raise DirectiveError(number, f"trace-table is followed by a {kind}, not a table")
        return replace(directive, target=target)

    def _bind_derived(
        self,
        placed: Sequence[tuple[Directive, int]],
        found: Sequence[_Found],
        tokens: list[NumberToken],
    ) -> None:
        taken: set[int] = set()
        for directive, end in placed:
            if directive.kind != "derived":
                continue
            choice = next(
                (
                    i
                    for i, token in enumerate(tokens)
                    if i not in taken and token.block == directive.target and found[i].start >= end
                ),
                None,
            )
            if choice is None:
                self.errors.append(
                    DirectiveError(directive.line, f"derived {directive.name}: no number follows")
                )
                continue
            taken.add(choice)
            tokens[choice] = replace(tokens[choice], derived=directive.name)

    def run(self) -> Document:
        self.scan_literals()
        self.scan_blocks()
        masked = _mask_identifiers("".join(self.chars))
        found = _find_numbers(masked)
        tokens: list[NumberToken] = []
        for item in found:
            line, column = self._position(item.start)
            tokens.append(
                NumberToken(
                    line,
                    column,
                    column + item.end - item.start,
                    item.raw,
                    item.sign,
                    item.value,
                    item.exponent,
                    item.unit,
                    self.block_of.get(line, -1),
                    self._cell_at(item.start),
                )
            )
        blocks = self._finished_blocks()
        placed = self._read_directives(blocks)
        self._bind_derived(placed, found, tokens)
        return Document(
            self.path,
            tuple(tokens),
            _find_intervals(masked, found, tokens),
            tuple(d for d, _ in placed),
            blocks,
            tuple(self.tables),
            tuple(self.headings),
            tuple(sorted(self.errors, key=lambda e: e.line)),
        )


def _find_intervals(
    text: str, found: Sequence[_Found], tokens: Sequence[NumberToken]
) -> tuple[Interval, ...]:
    intervals: list[Interval] = []
    for i in range(len(found) - 1):
        low, high = found[i], found[i + 1]
        before = low.start - 1
        while before >= 0 and text[before].isspace():
            before -= 1
        if before < 0 or text[before] != "[":
            continue
        if not _INTERVAL_JOIN.fullmatch(text, low.unit_end, high.start):
            continue
        if not _INTERVAL_END.match(text, high.unit_end):
            continue
        point = None
        if i > 0 and _POINT_GAP.fullmatch(text, found[i - 1].unit_end, low.start):
            point = tokens[i - 1]
        intervals.append(Interval(point, tokens[i], tokens[i + 1]))
    return tuple(intervals)


def parse_document(text: str, path: str = "") -> Document:
    """Read a Markdown document: numbers, intervals, blocks, tables, headings and directives.

    Invalid directives are collected in ``errors``, so one run reports all of them.
    """
    return _Parser(text, path).run()


def tokenize(text: str) -> tuple[NumberToken, ...]:
    """The numbers of a Markdown text."""
    return parse_document(text).tokens


def select_rows(table: Table, terms: Sequence[str]) -> list[TableRow]:
    """The rows holding every term in a cell; a single ``L<n>`` selects the row on line n."""
    if len(terms) == 1 and (by_line := _LINE_ROW.fullmatch(terms[0])):
        return [row for row in table.rows if row.line == int(by_line[1])]
    return [row for row in table.rows if all(term in row.cells for term in terms)]


def bound_tokens(document: Document, directive: Directive, binding: Binding) -> list[NumberToken]:
    """The document numbers a binding covers: its row or, failing that, its column."""
    table = next(t for t in document.tables if t.block == directive.target)
    rows = select_rows(table, binding.key)
    label = binding.key[0]
    columns: list[int] = []
    if len(binding.key) == 1:
        columns = [i for i, name in enumerate(table.header) if i > 0 and name == label]
    if len(rows) > 1 or (rows and columns):
        raise DirectiveError(directive.line, f"key {';'.join(binding.key)!r} is ambiguous")
    if not rows and len(columns) != 1:
        raise DirectiveError(directive.line, f"key {';'.join(binding.key)!r} fits nothing")
    chosen: list[NumberToken] = []
    for token in document.tokens:
        cell = token.cell
        if cell is None or cell.table != table.index or cell.row < 0:
            continue
        if (rows and cell.row == rows[0].index) or (columns and cell.column == columns[0]):
            chosen.append(token)
    return chosen


def _json_escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")


def _value(
    path: str, line: int, parsed: tuple[str, Decimal, int, str], where: str, header: str
) -> ReportValue:
    raw, value, exponent, unit = parsed
    return ReportValue(path, line, raw, value, exponent, unit, where, header)


def _index_json(path: str, text: str) -> IndexedFile:
    data = json.loads(text)
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]
    frames: list[list[object]] = []  # [is_object, key or position, expecting a key]
    units: dict[str, str] = {}
    values: dict[str, ReportValue] = {}

    def pointer(frames_used: Sequence[list[object]]) -> str:
        return "/".join(_json_escape(str(frame[1])) for frame in frames_used)

    pos = 0
    while pos < len(text):
        match = _JSON_TOKEN.match(text, pos)
        if match is None:
            raise TraceError(f"{path}: cannot read JSON at offset {pos}")
        pos = match.end()
        token = match[0]
        top = frames[-1] if frames else None
        if token in ("{", "["):
            frames.append([token == "{", 0 if token == "[" else "", token == "{"])
        elif token in ("}", "]"):
            frames.pop()
        elif token == ",":
            if top is not None and top[0]:
                top[2] = True
            elif top is not None:
                position = top[1]
                assert isinstance(position, int)
                top[1] = position + 1
        elif token[0] == '"':
            if top is not None and top[0] and top[2]:
                top[1], top[2] = json.loads(token), False
            elif top is not None and top[0] and top[1] == "unit":
                units[pointer(frames[:-1])] = json.loads(token)
        elif token[0] in "-0123456789" and not token.endswith("Infinity"):
            parsed = _whole_json_number(token)
            where = pointer(frames)
            header = str(top[1]) if top is not None else ""
            line = bisect.bisect_right(starts, match.start())
            values[where] = _value(path, line, parsed, where, header)
    intervals: list[ReportInterval] = []
    for where, point in values.items():
        parent, _, leaf = where.rpartition("/")
        if leaf != "value":
            continue
        prefix = parent + "/" if parent else ""
        low, high = values.get(prefix + "low"), values.get(prefix + "high")
        if low is not None and high is not None:
            unit = units.get(parent, "")
            intervals.append(ReportInterval(path, point.line, point, low, high, unit, parent))
    generated: set[str] = set()
    if isinstance(data, dict):
        for key in ("experiment", "experiment_id", "run", "experiments"):
            field = data.get(key)
            names = field if isinstance(field, dict) else [field]
            for name in names:
                if isinstance(name, str):
                    generated.update(EXPERIMENT_ID.findall(name))
    return IndexedFile(
        path,
        "",
        "json",
        len(starts),
        tuple(values.values()),
        tuple(intervals),
        generated_ids=tuple(sorted(generated)),
    )


def _whole_json_number(token: str) -> tuple[str, Decimal, int, str]:
    value = Decimal(token)
    exponent = value.as_tuple().exponent
    assert isinstance(exponent, int)
    return token, value, exponent, ""


def _index_csv(path: str, text: str) -> IndexedFile:
    reader = csv.reader(io.StringIO(text, newline=""))
    header: tuple[str, ...] = ()
    rows: list[TableRow] = []
    values: list[ReportValue] = []
    for cells in reader:
        if not header:
            header = tuple(c.strip() for c in cells)
            continue
        line = reader.line_num
        rows.append(TableRow(len(rows), line, tuple(c.strip() for c in cells)))
        for column, cell in enumerate(cells):
            parsed = _whole_number(cell)
            if parsed is not None:
                name = header[column] if column < len(header) else ""
                values.append(replace(_value(path, line, parsed, "", name), table=0))
    table = Table(0, -1, 1, header, tuple(rows))
    return IndexedFile(path, "", "csv", text.count("\n") + 1, tuple(values), tables=(table,))


def _index_yaml(path: str, text: str) -> IndexedFile:
    """Plain scalars and flow lists of a block-style YAML file, by indentation."""
    values: list[ReportValue] = []
    stack: list[tuple[int, str]] = []
    lines = text.split("\n")
    for number, raw in enumerate(lines, 1):
        hash_at = raw.find(" #")
        line = raw if hash_at < 0 or "'" in raw[:hash_at] or '"' in raw[:hash_at] else raw[:hash_at]
        if raw.lstrip().startswith("#") or not line.strip() or line.startswith("---"):
            continue
        match = _YAML_LINE.fullmatch(line.rstrip())
        if match is None:
            continue
        indent = len(match["indent"]) + len(match["dash"] or "")
        while stack and stack[-1][0] >= indent:
            stack.pop()
        key, value = match["key"], match["value"].strip()
        where = "/".join([k for _, k in stack] + ([_json_escape(key)] if key else []))
        if key and not value:
            stack.append((indent, _json_escape(key)))
            continue
        items = value[1:-1].split(",") if value.startswith("[") and value.endswith("]") else [value]
        for item in items:
            parsed = _whole_number(item)
            if parsed is not None:
                values.append(_value(path, number, parsed, where, key or ""))
    return IndexedFile(path, "", "yaml", len(lines), tuple(values))


def _report_value(token: NumberToken, path: str, where: str, table: int) -> ReportValue:
    header = token.cell.header if token.cell else ""
    return ReportValue(
        path,
        token.line,
        token.raw,
        token.value,
        token.exponent,
        token.unit,
        where,
        header,
        table,
    )


def _index_markdown(path: str, text: str) -> IndexedFile:
    name = PurePosixPath(path).name
    document = parse_document(text, path)
    lines = text.split("\n")
    head = "\n".join(lines[:5])
    generated_line = _GENERATED_BY.search(head)
    generated = EXPERIMENT_ID.findall(generated_line[0]) if generated_line else []
    base = IndexedFile(
        path,
        "",
        "md",
        len(lines),
        (),
        headings=document.headings,
        generated_ids=tuple(sorted(set(generated))),
    )
    if any(fnmatch.fnmatch(name, pattern) for pattern in FREE_TEXT_FILES):
        return base
    counts_only = name in COUNTS_ONLY_FILES
    kept_tables: dict[int, int] = {}
    tables: list[Table] = []
    for table in document.tables:
        if not FREE_TEXT_HEADERS.intersection(table.header):
            kept_tables[table.index] = len(tables)
            tables.append(replace(table, index=len(tables)))

    def section_of(line: int) -> str:
        where = ""
        for heading in document.headings:
            if heading.line > line:
                break
            where = heading.slug
        return where

    def keep(token: NumberToken) -> bool:
        cell = token.cell
        if cell is None:
            return not counts_only
        if cell.table not in kept_tables or cell.row < 0:
            return False
        if not counts_only:
            return True
        row = document.tables[cell.table].rows[cell.row]
        printed = row.cells[cell.column] if cell.column < len(row.cells) else ""
        return printed in (token.raw, token.raw + token.unit, f"{token.raw} {token.unit}")

    def to_value(token: NumberToken) -> ReportValue:
        table = kept_tables[token.cell.table] if token.cell else -1
        return _report_value(token, path, section_of(token.line), table)

    values = [to_value(t) for t in document.tokens if keep(t)]
    intervals: list[ReportInterval] = []
    for item in document.intervals:
        members = [m for m in (item.point, item.low, item.high) if m is not None]
        if all(keep(m) for m in members) and item.low.line == item.high.line:
            point = item.point
            same_line = point is not None and point.line == item.low.line
            intervals.append(
                ReportInterval(
                    path,
                    item.low.line,
                    to_value(point) if point is not None and same_line else None,
                    to_value(item.low),
                    to_value(item.high),
                    where=section_of(item.low.line),
                )
            )
    if counts_only:
        for number, line in enumerate(lines, 1):
            for match in _EXAMPLE_COUNTS.finditer(line):
                for group in (1, 2):
                    parsed = _whole_number(match[group])
                    if parsed is not None:
                        values.append(_value(path, number, parsed, section_of(number), ""))
        values.sort(key=lambda v: v.line)
    return replace(base, values=tuple(values), intervals=tuple(intervals), tables=tuple(tables))


def index_text(path: str, text: str) -> IndexedFile:
    """Index the text of a source file; its type comes from the suffix of ``path``."""
    text = text.replace("\r\n", "\n")
    suffix = PurePosixPath(path).suffix
    if suffix == ".json":
        indexed = _index_json(path, text)
    elif suffix == ".csv":
        indexed = _index_csv(path, text)
    elif suffix in (".yaml", ".yml"):
        indexed = _index_yaml(path, text)
    else:
        indexed = _index_markdown(path, text)
    return replace(
        indexed,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        ids=tuple(sorted(set(EXPERIMENT_ID.findall(text)))),
    )


class ReportIndex:
    """Reads source files under a repository root, once each, on first use."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._files: dict[str, IndexedFile] = {}

    def file(self, path: str) -> IndexedFile:
        """The index of one source file. Raises `SourcePathError` or `TraceError`."""
        if path not in self._files:
            check_source_path(path)
            full = (self.root / path).resolve()
            allowed = [(self.root / root).resolve() for root in ALLOWED_ROOTS]
            if not any(full.is_relative_to(base) for base in allowed):
                raise SourcePathError(f"{path!r}: leaves the allowed folders through a link")
            if not full.is_file():
                raise TraceError(f"{path}: no such file")
            self._files[path] = index_text(path, full.read_bytes().decode("utf-8"))
        return self._files[path]

    def values(self, path: str, anchor: str | None = None) -> tuple[ReportValue, ...]:
        return self.file(path).values_in(anchor)

    def intervals(self, path: str, anchor: str | None = None) -> tuple[ReportInterval, ...]:
        return self.file(path).intervals_in(anchor)

    def binding_values(
        self, path: str, anchor: str | None, binding: Binding
    ) -> tuple[ReportValue, ...]:
        """The report values a binding's ``row`` (and ``header``) selects."""
        indexed = self.file(path)
        if indexed.kind in ("json", "yaml"):
            if len(binding.row) != 1:
                raise TraceError(f"{path}: a key path takes one term, not {binding.row}")
            where = "/".join(p for p in (anchor, binding.row[0]) if p)
            if binding.header is not None:
                where += "/" + _json_escape(binding.header)
            chosen = tuple(v for v in indexed.values if _under(v.where, where))
            if not chosen:
                raise TraceError(f"{path}: no values under {where}")
            return chosen
        matches = [
            (table, row)
            for table in indexed.tables_in(anchor)
            for row in select_rows(table, binding.row)
        ]
        if len(matches) != 1:
            raise TraceError(f"{path}: row {';'.join(binding.row)!r} fits {len(matches)} rows")
        table, row = matches[0]
        if binding.header is not None and binding.header not in table.header:
            raise TraceError(f"{path}: no column {binding.header!r} at line {table.line}")
        return tuple(
            v
            for v in indexed.values
            if v.table == table.index
            and v.line == row.line
            and (binding.header is None or v.header == binding.header)
        )

    def cell_candidates(
        self, document: Document, directive: Directive, binding: Binding, token: NumberToken
    ) -> tuple[ReportValue, ...]:
        """The report values one bound document number may match, as the module docstring says."""
        values = self.binding_values(directive.path, directive.anchor, binding)
        cell = token.cell
        if binding.header is not None or cell is None:
            return values
        table = document.tables[cell.table]
        wanted = cell.header
        if cell.column > 0 and wanted == binding.key[0] and cell.row >= 0:
            row = table.rows[cell.row]
            wanted = row.cells[0] if row.cells else ""
        return tuple(v for v in values if v.header == wanted) or values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List the numbers and directives of a document.")
    parser.add_argument("documents", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for path in args.documents:
        document = parse_document(path.read_bytes().decode("utf-8"), path.as_posix())
        for token in document.tokens:
            bound = f" derived={token.derived}" if token.derived else ""
            kind = document.blocks[token.block].kind if token.block >= 0 else "none"
            print(f"{path.as_posix()}:{token.line}:{token.column} {token.raw}{token.unit}", end="")
            print(f" [{kind}]{bound}")
        for interval in document.intervals:
            point = interval.point.raw if interval.point else "-"
            bounds = f"{interval.low.raw} {interval.high.raw}"
            print(f"interval line {interval.low.line}: {point} {bounds}")
        for directive in document.directives:
            subject = directive.name or f"{directive.experiment} {directive.path}"
            print(f"directive line {directive.line}: {directive.kind} {subject}")
        for error in document.errors:
            failed = True
            print(f"{path.as_posix()}: error: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
