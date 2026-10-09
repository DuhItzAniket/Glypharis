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
