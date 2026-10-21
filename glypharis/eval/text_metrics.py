"""Recognition metrics (EXPERIMENT_PROTOCOL §3): CER, WER, NED, sequence accuracy.

Predictions and references are normalised with `glypharis.data.text.normalize_eval` for the
chosen track before scoring. Totals are micro-averaged over the evaluation set; per-sample values
are kept so `glypharis.eval.stats` can bootstrap confidence intervals over samples.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from rapidfuzz.distance import Levenshtein

from glypharis.data.text import normalize_eval

Track = Literal["strict", "normalised"]


def char_distance(pred: str, ref: str) -> int:
    return int(Levenshtein.distance(pred, ref))


def word_distance(pred: str, ref: str) -> int:
    return int(Levenshtein.distance(pred.split(), ref.split()))


@dataclass
class SampleScore:
    char_errors: int
    ref_chars: int
    word_errors: int
    ref_words: int
    ned: float  # 1 - d / max(len) ; 1 is perfect
    exact: bool


def score_pair(pred: str, ref: str, track: Track = "strict") -> SampleScore:
    p, r = normalize_eval(pred, track), normalize_eval(ref, track)
    d = char_distance(p, r)
    longest = max(len(p), len(r))
    return SampleScore(
        char_errors=d,
        ref_chars=len(r),
        word_errors=word_distance(p, r),
        ref_words=len(r.split()),
        ned=1.0 - d / longest if longest else 1.0,
        exact=p == r,
    )


@dataclass
class RecSummary:
    samples: int
    cer: float
    wer: float
    ned: float
    sequence_accuracy: float
    empty_references: int


@dataclass
class RecAccumulator:
    """Collects per-sample scores; empty references are counted but left out of CER/WER."""

    track: Track = "strict"
    scores: list[SampleScore] = field(default_factory=list)
    empty_refs: int = 0

    def add(self, pred: str, ref: str) -> SampleScore | None:
        if not normalize_eval(ref, self.track):
            self.empty_refs += 1
            return None
        s = score_pair(pred, ref, self.track)
        self.scores.append(s)
        return s

    def extend(self, pairs: Iterable[tuple[str, str]]) -> None:
        for pred, ref in pairs:
            self.add(pred, ref)

    def summary(self) -> RecSummary:
        return summarise_scores(self.scores, self.empty_refs)


def summarise_scores(scores: list[SampleScore], empty_refs: int = 0) -> RecSummary:
    n = len(scores)
    chars = sum(s.ref_chars for s in scores)
    words = sum(s.ref_words for s in scores)
    return RecSummary(
        samples=n,
        cer=sum(s.char_errors for s in scores) / chars if chars else 0.0,
        wer=sum(s.word_errors for s in scores) / words if words else 0.0,
        ned=sum(s.ned for s in scores) / n if n else 0.0,
        sequence_accuracy=sum(s.exact for s in scores) / n if n else 0.0,
        empty_references=empty_refs,
    )
