"""Error-analysis building blocks for Phase 15 (GW-REC-040, GW-E2E-010); pure functions, no I/O.

Everything here works on strings and rows the caller has already read and checked:

- edit operations, character confusions and the class matrix, from the CER's own distance
  (`rapidfuzz.distance.Levenshtein`), so the operations of a line always add up to its
  char_errors. The alignment is one of several optimal ones: inside a run of repeated characters
  which copy counts as the error is arbitrary;
- word errors, the same alignment over whitespace tokens;
- bins with edges fixed from validation, the content flags and the primary error category;
- the page error budget, the order-inversion counts and the slice and share tables with their
  bootstrap intervals (`glypharis.eval.stats`, used unchanged).

GDC-scan is numbers only (Phase 15 plan, item 8): every builder that sees text and takes a set
name refuses it through `refuse_numbers_only`.
"""

from __future__ import annotations

import itertools
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Container, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Literal

import numpy as np
from rapidfuzz.distance import Levenshtein

from glypharis.eval import stats
from glypharis.eval.stats import Interval

NUMBERS_ONLY = ("gdc_scan_test",)  # sets whose text never leaves the data folder (Phase 14)

Op = Literal["sub", "ins", "del"]
OPS: tuple[Op, ...] = ("sub", "ins", "del")
_RF_OPS: dict[str, Op] = {"replace": "sub", "insert": "ins", "delete": "del"}

CLASSES = ("lower", "upper", "digit", "punct", "space", "other", "oov")
OUTCOMES = (*CLASSES, "deleted")

# Fixed bins (Phase 15 plan, item 3). Edges are lower-inclusive: `bin_of` puts v in bin i when
# edges[i-1] <= v < edges[i].
ROTATION_BINS = (2.0, 5.0, 15.0)  # on the folded angle |θ|: < 2, 2-5, 5-15, >= 15 degrees
IOU_BINS = (0.7, 0.85)  # matched lines (IoU >= 0.5 by construction): [0.5, 0.7), [0.7, 0.85), ...
UNMATCHED_BINS = ("0", "(0, 0.3)", "[0.3, 0.5)", "≥ 0.5")  # best IoU of an unmatched line
LONG_WORD = 10  # letters in a token for the "long word" flag

# Primary error categories, in precedence order (Phase 15 plan, item 4).
CATEGORIES = (
    "out_of_charset",
    "rotation_ge_5",
    "height_q1",
    "blur_q1",
    "contrast_q1",
    "digit",
    "punctuation",
    "long_word",
    "other",
)


# Folders under the data root that Phase 15 never writes into (plan item 8 and amendment 11):
# Phase 14's outputs, the Phase 5 and Phase 14 baseline files, and the processed manifests.
PROTECTED_OUTPUTS = (
    "processed/phase14",
    "processed/baselines",
    "processed/baselines_phase14",
    "manifests/processed",
)

# HierText line ids (`glypharis.data.hiertext`): <image>/p<paragraph>/l<line>.
_HIERTEXT_LINE_ID = re.compile(r"^([^/]+)/p\d+/l\d+$")


def refuse_numbers_only(eval_set: str, what: str) -> None:
    """Raise if ``eval_set`` is numbers only, so no text table is ever built for it."""
    if eval_set in NUMBERS_ONLY:
        raise ValueError(f"{eval_set} is numbers only; {what} is not built for it")


def protected_folder(path: PurePath, data: PurePath) -> str | None:
    """The `PROTECTED_OUTPUTS` folder that ``path`` lies in (or is), or None.

    Both paths are compared as given, so callers pass them resolved. Parts are compared, not
    strings: ``processed/baselines`` does not cover ``processed/baselines_phase14``."""
    for rel in PROTECTED_OUTPUTS:
        folder = data.joinpath(*rel.split("/"))
        if path == folder or folder in path.parents:
            return rel
    return None


def hiertext_page(line_id: str) -> str:
    """The page (image id) of a HierText line id ``<image>/p<i>/l<j>``; raises on another form."""
    m = _HIERTEXT_LINE_ID.match(line_id)
    if m is None:
        raise ValueError(f"{line_id!r} is not a HierText line id <image>/p<i>/l<j>")
    return m.group(1)


# -------------------------------------------------------------------------- character classes


def char_class(ch: str, charset: Container[str]) -> str:
    """The class of one character; out of charset is checked first."""
    if ch not in charset:
        return "oov"
    if ch.isspace():
        return "space"
    cat = unicodedata.category(ch)
    if cat == "Ll":
        return "lower"
    if cat == "Lu":
        return "upper"
    if cat == "Nd":
        return "digit"
    if cat[0] in "PS":
        return "punct"
    return "other"


# ---------------------------------------------------------------------------- edit operations


def edit_ops(ref: str, pred: str) -> list[tuple[str, str, str]]:
    """Levenshtein edit operations turning ``ref`` into ``pred``: (op, ref char, pred char).

    ``op`` is "sub", "ins" or "del"; the missing side is "". Their number is the CER's distance.
    Callers pass strings already normalised for the track.
    """
    return _aligned(ref, pred)[0]


def _aligned(ref: str, pred: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str]]]:
    """The edit operations and, for every reference character, what it became ("" if deleted)."""
    fate: dict[int, str] = {}
    out: list[tuple[str, str, str]] = []
    for e in Levenshtein.editops(ref, pred):
        op, i, j = _RF_OPS[e.tag], e.src_pos, e.dest_pos
        if op == "sub":
            fate[i] = pred[j]
            out.append((op, ref[i], pred[j]))
        elif op == "del":
            fate[i] = ""
            out.append((op, ref[i], ""))
        else:
            out.append((op, "", pred[j]))
    outcomes = [(c, fate.get(i, c)) for i, c in enumerate(ref)]
    return out, outcomes


@dataclass
class Confusions:
    """Character confusions of one set and track, accumulated line by line.

    Counts substitutions (ref -> pred), insertions and deletions, the space errors among them
    (a deleted space merges two words, an inserted one splits a word), and every reference
    character's outcome (itself when read correctly), which the class matrix needs.
    """

    eval_set: str
    lines: int = 0
    ref_counts: Counter[str] = field(default_factory=Counter)
    pairs: dict[str, Counter[tuple[str, str]]] = field(
        default_factory=lambda: {op: Counter() for op in OPS}
    )
    outcomes: Counter[tuple[str, str]] = field(default_factory=Counter)
    space: Counter[str] = field(default_factory=Counter)

    def __post_init__(self) -> None:
        refuse_numbers_only(self.eval_set, "a confusion table")

    def add(self, ref: str, pred: str, char_errors: int | None = None) -> int:
        """Add one normalised pair; returns its number of operations.

        With ``char_errors`` (the line's stored value) the count is checked against it.
        """
        ops, outcomes = _aligned(ref, pred)
        if char_errors is not None and len(ops) != char_errors:
            raise ValueError(f"{len(ops)} edit operations but char_errors = {char_errors}")
        self.lines += 1
        self.ref_counts.update(ref)
        self.outcomes.update(outcomes)
        for op, r, p in ops:
            self.pairs[op][(r, p)] += 1
            if op == "ins" and p.isspace():
                self.space["inserted"] += 1
            elif op == "del" and r.isspace():
                self.space["deleted"] += 1
            elif op == "sub" and (r.isspace() or p.isspace()):
                self.space["substituted"] += 1
        return len(ops)

    def top(self, kind: str, n: int = 30) -> list[tuple[str, str, int, float | None]]:
        """The ``n`` most frequent operations of ``kind`` ("sub", "ins" or "del").

        Rows are (ref char, pred char, count, rate per occurrence of the reference character),
        sorted by count descending, then by the characters; an insertion has no reference
        character, so its rate is None.
        """
        if kind not in OPS:
            raise ValueError(f"kind must be one of {OPS}, not {kind!r}")
        counter = self.pairs[kind]
        rows = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        out: list[tuple[str, str, int, float | None]] = []
        for (r, p), c in rows:
            rate = c / self.ref_counts[r] if r and self.ref_counts[r] else None
            out.append((r, p, c, rate))
        return out

    def class_matrix(self, charset: Container[str]) -> dict[str, Any]:
        """Reference class (rows) x outcome class (columns), correct characters included.

        ``counts[r][c]`` counts reference characters of class r that came out as a character of
        class c (or were deleted); ``rates`` divide by the row's reference characters, so each row
        sums to 1 (an empty row stays at 0). The same-class cell holds correct characters and
        same-class substitutions alike ("e" read as "c"), so ``correct`` counts the characters
        read as themselves and ``correct_rates`` their share of the row. Insertions belong to no
        reference character and are counted by the inserted character's class in ``inserted``.
        """
        counts = {r: dict.fromkeys(OUTCOMES, 0) for r in CLASSES}
        correct = dict.fromkeys(CLASSES, 0)
        for (r, p), c in self.outcomes.items():
            rc = char_class(r, charset)
            counts[rc][char_class(p, charset) if p else "deleted"] += c
            if p == r:
                correct[rc] += c
        totals = {r: sum(row.values()) for r, row in counts.items()}
        rates = {
            r: {c: (v / totals[r] if totals[r] else 0.0) for c, v in row.items()}
            for r, row in counts.items()
        }
        inserted = dict.fromkeys(CLASSES, 0)
        for (_, p), c in self.pairs["ins"].items():
            inserted[char_class(p, charset)] += c
        return {
            "rows": list(CLASSES),
            "columns": list(OUTCOMES),
            "ref_chars": totals,
            "counts": counts,
            "rates": rates,
            "correct": correct,
            "correct_rates": {r: (correct[r] / totals[r] if totals[r] else 0.0) for r in CLASSES},
            "inserted": inserted,
        }

    def totals(self) -> dict[str, Any]:
        """Lines, reference characters, operation counts and rates per reference character."""
        ref_chars = sum(self.ref_counts.values())
        n = {op: sum(self.pairs[op].values()) for op in OPS}
        errors = sum(n.values())

        def rate(x: int) -> float:
            return x / ref_chars if ref_chars else 0.0

        return {
            "lines": self.lines,
            "ref_chars": ref_chars,
            "errors": errors,
            "cer": rate(errors),
            **n,
            **{f"{op}_rate": rate(v) for op, v in n.items()},
            "space_inserted": self.space["inserted"],
            "space_deleted": self.space["deleted"],
            "space_substituted": self.space["substituted"],
        }


# -------------------------------------------------------------------------------- word errors


def word_ops(ref: str, pred: str) -> list[tuple[str, str, str]]:
    """Edit operations over whitespace tokens: (op, ref word, pred word), as `edit_ops`."""
    r, p = ref.split(), pred.split()
    out: list[tuple[str, str, str]] = []
    for e in Levenshtein.editops(r, p):
        op, i, j = _RF_OPS[e.tag], e.src_pos, e.dest_pos
        out.append((op, r[i] if op != "ins" else "", p[j] if op != "del" else ""))
    return out


@dataclass
class WordErrors:
    """Word substitutions, insertions and deletions of one set and track."""

    eval_set: str
    ref_words: int = 0
    subs: Counter[tuple[str, str]] = field(default_factory=Counter)
    ins: int = 0
    dels: int = 0

    def __post_init__(self) -> None:
        refuse_numbers_only(self.eval_set, "a word-error table")

    def add(self, ref: str, pred: str) -> int:
        """Add one normalised pair; returns its number of word operations."""
        ops = word_ops(ref, pred)
        self.ref_words += len(ref.split())
        for op, r, p in ops:
            if op == "sub":
                self.subs[(r, p)] += 1
            elif op == "ins":
                self.ins += 1
            else:
                self.dels += 1
        return len(ops)

    def top(self, n: int = 30) -> list[tuple[str, str, int]]:
        """The ``n`` most frequent (ref word, pred word, count) substitutions."""
        rows = sorted(self.subs.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        return [(r, p, c) for (r, p), c in rows]

    def one_char_share(self) -> float:
        """Share of substituted word pairs (by count) whose character distance is 1."""
        total = sum(self.subs.values())
        one = sum(c for (r, p), c in self.subs.items() if Levenshtein.distance(r, p) == 1)
        return one / total if total else 0.0

    def totals(self) -> dict[str, Any]:
        sub = sum(self.subs.values())
        errors = sub + self.ins + self.dels
        return {
            "ref_words": self.ref_words,
            "errors": errors,
            "wer": errors / self.ref_words if self.ref_words else 0.0,
            "sub": sub,
            "ins": self.ins,
            "del": self.dels,
            "one_char_share": self.one_char_share(),
        }


# ----------------------------------------------------------------------------------------- bins


def _missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def quantile_edges(
    values: Iterable[float | None], qs: Sequence[float] = (0.25, 0.5, 0.75)
) -> tuple[float, ...]:
    """Quantile edges of ``values`` (None and NaN left out), equal edges merged."""
    v = np.asarray([x for x in values if not _missing(x)], dtype=np.float64)
    if v.size == 0:
        raise ValueError("no values to take quantiles of")
    return tuple(float(e) for e in np.unique(np.quantile(v, list(qs))))


def bin_of(value: float | None, edges: Sequence[float]) -> int | None:
    """Index of the bin holding ``value`` (bin i is edges[i-1] <= v < edges[i]); None if missing."""
    if value is None or _missing(value):
        return None
    return int(np.searchsorted(np.asarray(edges, dtype=np.float64), float(value), side="right"))


def _num(x: float) -> str:
    return f"{x:g}"


def bin_labels(edges: Sequence[float], low: float | None = None) -> list[str]:
    """Labels of the ``len(edges) + 1`` bins; ``low`` names a known lower bound of bin 0."""
    if not edges:
        return ["all"]
    first = f"[{_num(low)}, {_num(edges[0])})" if low is not None else f"< {_num(edges[0])}"
    middle = [f"[{_num(a)}, {_num(b)})" for a, b in itertools.pairwise(edges)]
    return [first, *middle, f"≥ {_num(edges[-1])}"]


def fold_angle(deg: float) -> float:
    """A box angle folded to [0, 90]: how far the line is from level, either way round."""
    a = abs(deg) % 180
    return min(a, 180 - a)


def rotation_bin(deg: float | None) -> int | None:
    """`ROTATION_BINS` bin of a box angle, after folding; None if missing."""
    if deg is None or _missing(deg):
        return None
    return bin_of(fold_angle(float(deg)), ROTATION_BINS)


def unmatched_bin(best_iou: float) -> int:
    """`UNMATCHED_BINS` index of an unmatched line's best IoU."""
    if best_iou <= 0:
        return 0
    if best_iou < 0.3:
        return 1
    return 2 if best_iou < 0.5 else 3


# ------------------------------------------------------------------------------- content flags


def has_digit(text: str) -> bool:
    return any(unicodedata.category(c) == "Nd" for c in text)


def has_punct(text: str) -> bool:
    """Punctuation or symbol characters (Unicode categories P* and S*)."""
    return any(unicodedata.category(c)[0] in "PS" for c in text)


def all_upper(text: str) -> bool:
    """At least two letters, every one of them upper case."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    return len(letters) >= 2 and all(unicodedata.category(c) == "Lu" for c in letters)


def has_long_word(text: str) -> bool:
    """A whitespace token with at least `LONG_WORD` letters."""
    return any(sum(c.isalpha() for c in tok) >= LONG_WORD for tok in text.split())


def has_oov(text: str, charset: Container[str]) -> bool:
    return any(c not in charset for c in text)


def is_label_fragment(ref_norm: str, pred_norm: str) -> bool:
    """A likely HierText label fragment: a reference of at most 2 characters read as a much
    longer line (prediction at least 3 x ref + 3 characters)."""
    return len(ref_norm) <= 2 and len(pred_norm) >= 3 * len(ref_norm) + 3


def primary_category(attrs: Mapping[str, Any], edges: Mapping[str, Sequence[float]]) -> str:
    """The first `CATEGORIES` rule a wrong line meets; "other" if none.

    ``attrs`` may hold ``oov``, ``rotation_deg``, ``line_height_px``, ``blur``, ``contrast``
    (values) and ``digit``, ``punct``, ``long_word`` (flags); ``edges`` holds the validation
    quartile edges of ``line_height_px``, ``blur`` and ``contrast``. A missing field (or edge)
    skips its rule. "Lowest quartile" is bin 0 of those edges.
    """
    if attrs.get("oov"):
        return "out_of_charset"
    rot = attrs.get("rotation_deg")
    if rot is not None and not _missing(rot) and fold_angle(float(rot)) >= 5.0:
        return "rotation_ge_5"
    for name, cat in (("line_height_px", "height_q1"), ("blur", "blur_q1"),
                      ("contrast", "contrast_q1")):  # fmt: skip
        if name in edges and bin_of(attrs.get(name), edges[name]) == 0:
            return cat
    for name, cat in (("digit", "digit"), ("punct", "punctuation"), ("long_word", "long_word")):
        if attrs.get(name):
            return cat
    return "other"


# --------------------------------------------------------------------------------- page budget


def page_budget(pairs_row: Mapping[str, Any], bag_errors: int, track: str) -> dict[str, Any]:
    """Split one page's bag-of-lines errors into missed, matched and spurious characters.

    ``pairs_row`` is an ``e2e_pairs_<set>`` row; ``bag_errors`` the page's value on ``track`` from
    ``pages_<set>``. Missed = reference characters of lines no prediction matched, matched =
    errors on matched lines, spurious = the rest. Raises if spurious is negative, if it is not 0
    on a page with no spurious prediction, or if the unmatched bins do not add up to missed (an
    unmatched line without a best IoU).
    """
    key = pairs_row.get("key")
    missed = matched = 0
    unmatched = dict.fromkeys(UNMATCHED_BINS, 0)
    for p in pairs_row["pairs"]:
        t = p[track]
        if p["pred"] is None:
            missed += int(t["ref_chars"])
            best = p.get("best_iou")
            if best is not None and not _missing(
                best
            ):  # a line with no best IoU lands in no bin: the check fails
                unmatched[UNMATCHED_BINS[unmatched_bin(float(best))]] += int(t["ref_chars"])
        else:
            matched += int(t["char_errors"])
    spurious = int(bag_errors) - missed - matched
    if spurious < 0:
        raise ValueError(f"page {key}: spurious characters {spurious} < 0 ({track})")
    if spurious != 0 and not pairs_row.get("spurious"):
        raise ValueError(f"page {key}: {spurious} spurious characters but no spurious line")
    if sum(unmatched.values()) != missed:
        raise ValueError(f"page {key}: unmatched bins do not add up to the missed characters")
    return {
        "bag_errors": int(bag_errors),
        "missed": missed,
        "matched": matched,
        "spurious": spurious,
        "unmatched": unmatched,
    }


# -------------------------------------------------------------------------------- line order


def inversion_counts(pairs: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Consecutive matched reference lines (stored order) whose predictions come out reversed.

    Returns (inverted, consecutive pairs of matched lines).
    """
    preds = [int(p["pred"]) for p in pairs if p.get("pred") is not None]
    inverted = sum(b < a for a, b in itertools.pairwise(preds))
    return inverted, max(len(preds) - 1, 0)


def top_edge_order(polygons: Sequence[Sequence[Sequence[float]]]) -> tuple[int, int]:
    """Consecutive output lines whose top edge (min y) does not go up: (non-rising, pairs)."""
    tops = [min(float(pt[1]) for pt in poly) for poly in polygons]
    non_rising = sum(b >= a for a, b in itertools.pairwise(tops))
    return non_rising, max(len(tops) - 1, 0)


# ------------------------------------------------------------------------------ slice tables


def _sorted_labels(labels: Iterable[Hashable]) -> list[Hashable]:
    items = list(labels)
    try:
        return sorted(items)  # type: ignore[type-var]
    except TypeError:
        return sorted(items, key=str)


def _grouped_ci(
    ids: Sequence[Hashable],
    num: np.ndarray,
    den: np.ndarray,
    unit: str,
    resamples: int,
    seed: int,
) -> Interval:
    """`stats.ratio_ci` over per-group sums; groups with nothing in the slice stay in as 0/0."""
    order = _sorted_labels(set(ids))
    index = {g: i for i, g in enumerate(order)}
    inv = np.asarray([index[g] for g in ids], dtype=np.int64)
    m = len(order)
    n = np.bincount(inv, weights=num, minlength=m)
    d = np.bincount(inv, weights=den, minlength=m)
    return stats.ratio_ci(n, d, unit, resamples=resamples, seed=seed)


def slice_rows(
    units: Sequence[Mapping[str, Any]],
    key: Callable[[Mapping[str, Any]], Hashable | None],
    page_of: Callable[[Mapping[str, Any]], Hashable] | None = None,
    cluster: Callable[[Mapping[str, Any]], Hashable] | None = None,
    min_lines: int = 30,
    min_pages: int = 5,
    cluster_unit: str = "documents",
    resamples: int = stats.DEFAULT_RESAMPLES,
    seed: int = stats.DEFAULT_SEED,
    total: float | None = None,
    page_ci: bool = True,
) -> list[dict[str, Any]]:
    """One row per slice ``key(unit)`` (None leaves the unit out of every slice).

    Units are dicts with ``char_errors`` and ``ref_chars``. A row has label, lines, pages,
    clusters, ref_chars, char_errors, cer, share (of ``total``, the whole set's character
    errors; by default the units' own, for units that are the whole set), and the intervals:
    ``ci_lines`` over lines, ``ci_pages`` over per-page sums of every page the units cover (pages
    with no line in the slice kept as 0/0) when ``page_of`` is given and ``page_ci`` is true, and
    ``ci_clusters`` the same way over ``cluster``. Below ``min_lines`` lines or ``min_pages``
    pages a row carries counts only (intervals None).
    """
    err = np.asarray([float(u["char_errors"]) for u in units], dtype=np.float64)
    ref = np.asarray([float(u["ref_chars"]) for u in units], dtype=np.float64)
    if total is None:
        total = float(err.sum())
    elif total < float(err.sum()):
        raise ValueError(f"the set total {total:g} is below the units' {float(err.sum()):g} errors")
    labels = [key(u) for u in units]
    pages = [page_of(u) for u in units] if page_of is not None else None
    clusters = [cluster(u) for u in units] if cluster is not None else None
    rows: list[dict[str, Any]] = []
    for label in _sorted_labels({lb for lb in labels if lb is not None}):
        mask = np.asarray([lb == label for lb in labels])
        e, r = err[mask], ref[mask]
        n_pages = None if pages is None else len({p for p, m in zip(pages, mask, strict=True) if m})
        n_clusters = (
            None if clusters is None else len({c for c, m in zip(clusters, mask, strict=True) if m})
        )
        big = int(mask.sum()) >= min_lines and (n_pages is None or n_pages >= min_pages)
        ci_lines = ci_pages = ci_clusters = None
        if big:
            ci_lines = stats.ratio_ci(e, r, "lines", resamples=resamples, seed=seed)
            if pages is not None and page_ci:
                ci_pages = _grouped_ci(pages, err * mask, ref * mask, "pages", resamples, seed)
            if clusters is not None:
                ci_clusters = _grouped_ci(
                    clusters, err * mask, ref * mask, cluster_unit, resamples, seed
                )
        rows.append(
            {
                "label": label,
                "lines": int(mask.sum()),
                "pages": n_pages,
                "clusters": n_clusters,
                "ref_chars": int(r.sum()),
                "char_errors": int(e.sum()),
                "cer": float(e.sum() / r.sum()) if r.sum() else 0.0,
                "share": float(e.sum() / total) if total else 0.0,
                "ci_lines": ci_lines,
                "ci_pages": ci_pages,
                "ci_clusters": ci_clusters,
            }
        )
    return rows


def share_ci(
    num_by_page: Mapping[str, float],
    den_by_page: Mapping[str, float],
    cluster_of: Mapping[str, Hashable] | None = None,
    cluster_unit: str = "documents",
    resamples: int = stats.DEFAULT_RESAMPLES,
    seed: int = stats.DEFAULT_SEED,
    page_ci: bool = True,
    cluster_ci: bool = True,
) -> dict[str, Any]:
    """The oracle share Σnum / Σden over a set of pages (H4), with page and cluster intervals.

    Also reports the number of pages and clusters, and how many pages (largest ``den`` first)
    make up 80% of Σden. With ``page_ci`` (``cluster_ci``) false the page (cluster) interval
    is left out, for a cell that does not print it.
    """
    if set(num_by_page) != set(den_by_page):
        raise ValueError("numerator and denominator cover different pages")
    keys = sorted(den_by_page)
    num = np.asarray([float(num_by_page[k]) for k in keys], dtype=np.float64)
    den = np.asarray([float(den_by_page[k]) for k in keys], dtype=np.float64)
    out: dict[str, Any] = {
        "pages": len(keys),
        "num": float(num.sum()),
        "den": float(den.sum()),
        "value": float(num.sum() / den.sum()) if den.sum() else 0.0,
        "ci_pages": (
            stats.ratio_ci(num, den, "pages", resamples=resamples, seed=seed) if page_ci else None
        ),
        "clusters": None,
        "ci_clusters": None,
    }
    if cluster_of is not None:
        missing = [k for k in keys if k not in cluster_of]
        if missing:
            raise ValueError(f"pages without a cluster: {missing[:3]}")
        ids = [cluster_of[k] for k in keys]
        out["clusters"] = len(set(ids))
        if cluster_ci:
            out["ci_clusters"] = _grouped_ci(ids, num, den, cluster_unit, resamples, seed)
    cum = np.cumsum(np.sort(den)[::-1])
    target = 0.8 * float(den.sum())
    out["pages_80"] = int(np.searchsorted(cum, target, side="left") + 1) if target > 0 else 0
    return out
