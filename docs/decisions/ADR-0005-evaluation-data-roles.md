# ADR-0005: Fixed roles for every dataset: train / val / test / external

- **Status:** Accepted (2026-10-09, Phase 0). SynthText row superseded by
  [ADR-0009](ADR-0009-no-synthtext.md); HierText test labels became public, so the fallback is unused.

## Context
The brief requires strict separation of train, validation, test and external test data (§40) and
forbids tuning on test sets. Roles that drift during a project (a benchmark quietly becoming a tuning set)
are a common source of inflated results.

## Decision

| Dataset | Role |
|---|---|
| HierText train | train |
| HierText validation | validation (becomes test if HierText test labels are withheld; then a grouped validation split is carved from train) |
| HierText test | internal test (if labels are available) |
| SynthText | train only |
| GlyphSynth train / val | train / validation |
| GDC | internal test, frozen |
| IIIT5K test, FUNSD, OmniDocBench, SROIE | external test |
| DocVQA | reference only, never CER ground truth (machine-generated OCR labels) |

- Test and external sets are evaluated only in Phases 5, 14 and 22 and logged in
  `research/test_access_log.md`.
- A dataset's role can only change through a new ADR that supersedes this one.

## Consequences
External benchmarks stay truly external, so Phase 14 numbers measure generalisation. There is little
real document data for validation; document-domain validation comes from GlyphSynth-val and the
document-like slice of HierText-val, which is a limitation stated in the paper.

## Revisit if
A permissively licensed real document dataset with human line transcriptions becomes available that
could serve as a document-domain validation set.
