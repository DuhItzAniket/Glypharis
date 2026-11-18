"""Every citation key written in the paper package resolves in ``references.bib`` (plan item 14)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BIB = ROOT / "research/paper/references.bib"
REVIEW = ROOT / "research/paper/trace/references_review.md"
PAPER_DIR = ROOT / "research/paper"
OTHER_DOCUMENTS = ("MODEL_CARD.md", "DATASETS.md", "research/approach_survey.md")

ENTRY = re.compile(r"^@\w+\{\s*([^,\s]+)", re.M)
CODE_SPAN = re.compile(r"`([^`\n]+)`")
# Keys look like ``smith2019onecycle`` or ``levenshtein1966``; these four have no year.
KEY_SHAPE = re.compile(r"[a-z][a-z_]*[0-9]{4}[a-z_]*")
KEYS_WITHOUT_YEAR = frozenset({"onnxruntime", "doctr", "easyocr", "paddleocr_ppocrv6"})
FROZEN_KEYS = frozenset(
    {
        "cui2025paddleocr",
        "paddleocr_ppocrv6",
        "kuznetsova2020openimages",
        "russakovsky2015imagenet",
        "harley2015rvlcdip",
        "lin2017fpn",
        "nagy1992xycut",
        "levenshtein1966",
        "naeini2015ece",
        "loshchilov2019adamw",
        "smith2019onecycle",
        "paszke2019pytorch",
        "onnxruntime",
        "doctr",
        "easyocr",
    }
)


def bib_keys(text: str) -> list[str]:
    return ENTRY.findall(text)


def cited_keys(text: str) -> set[str]:
    """The code spans of a document that have the shape of a citation key."""
    spans = CODE_SPAN.findall(text)
    return {s for s in spans if KEY_SHAPE.fullmatch(s) or s in KEYS_WITHOUT_YEAR}


def documents() -> list[Path]:
    return [*sorted(PAPER_DIR.glob("*.md")), *(ROOT / name for name in OTHER_DOCUMENTS)]


def read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def test_key_parsing() -> None:
    assert bib_keys("@article{a2020b,\n  x = 1\n}\n@misc{ doctr ,\n}\n") == ["a2020b", "doctr"]
    assert cited_keys("See `a2020b`, `doctr`, `hiertext_test` and `reports/x2020.md`.") == {
        "a2020b",
        "doctr",
    }


def test_keys_are_unique_and_the_frozen_ones_exist() -> None:
    keys = bib_keys(read(BIB))
    assert len(keys) == len(set(keys))
    assert not FROZEN_KEYS - set(keys)


@pytest.mark.parametrize("path", documents(), ids=lambda p: p.name)
def test_every_cited_key_resolves(path: Path) -> None:
    missing = cited_keys(read(path)) - set(bib_keys(read(BIB)))
    assert not missing, f"{path.name} cites keys not in references.bib: {sorted(missing)}"


def test_every_entry_has_a_row_in_the_review_log() -> None:
    review = read(REVIEW)
    missing = [key for key in bib_keys(read(BIB)) if f"`{key}`" not in review]
    assert not missing, f"no review row for {missing}"
