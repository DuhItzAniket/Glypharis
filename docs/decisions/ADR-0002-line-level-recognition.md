# ADR-0002: Line-level detection and CTC recognition

- **Status:** Accepted (2026-10-09, Phase 0)

## Context
The brief allows word- or line-level recognition. Documents are organised in lines; word segmentation
fails on tight kerning, justified text and degraded scans. CTC decoders are fast, non-autoregressive and
behave well on long sequences, while attention decoders are stronger on short irregular words but slower
and prone to drift on long inputs.

## Decision
- The detector predicts **text-line** polygons.
- The recognizer reads **whole lines** with a CTC head (CRNN baseline, SVTR-style encoder as a Phase 7
  candidate). Greedy decoding by default, optional lexicon-free beam search, no language model in v1.
- Overly long lines are split at the detector level, not inside the recognizer.
- Word-crop benchmarks (IIIT5K) are still evaluated, as an out-of-domain test.

## Consequences
- Spaces are recognised as characters, so word boundaries come from the model, not from geometry.
- HierText line annotations can be used directly; SynthText needs word-to-line grouping or word crops.
- Variable-width inputs need width bucketing to keep batches efficient.
- Word boxes are not produced in v1; they can be recovered later from CTC alignments.

## Revisit if
Line CER on validation is dominated by long-line failures that word-level recognition would avoid, as
measured in the Phase 9/15 error analysis.

**Note (2026-11-17, Phase 15, GW-REC-040):** the trigger is not met. Share of character errors by
strict-normalised reference length (`baseline_diagnostics.py` buckets), frozen recognizer on
ground-truth crops (`reports/error_cases/phase15/recognition.md`):

| Length | Validation (GW-REC-038): CER % | Share of errors % | HierText test (GW-REC-039): CER % | Share of errors % |
|---|---|---|---|---|
| 0-29 | 8.56 | 81.51 | 7.88 | 80.24 |
| 30-49 | 3.60 | 10.43 | 3.39 | 10.70 |
| 50-69 | 2.43 | 4.47 | 2.39 | 4.32 |
| 70+ | 3.22 | 3.59 | 3.31 | 4.74 |

Short lines, not long ones, carry most of the validation error. On GDC's documents the 70+ bucket
holds 61.80% of the errors at 6.37% CER, but the 20 worst GDC lines are table-of-contents lines
whose leader dots are dropped, a failure on punctuation runs, not on long text. No per-line
attribute labels leaders, so how much of the 70+ share they carry was not measured, nor whether
word-level recognition would read them better. The decision stands.
