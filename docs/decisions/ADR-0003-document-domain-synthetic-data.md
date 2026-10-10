# ADR-0003: Add a document-style synthetic data source next to SynthText

- **Status:** Accepted (2026-10-09, Phase 0). The "keep SynthText" part is superseded by
  [ADR-0009](ADR-0009-no-synthtext.md) (2026-10-10).

## Context
The brief's primary training data, HierText and SynthText, is oriented towards text in natural scenes.
The target is document pages: dense printed lines, book and paper fonts, scanner and camera degradations.
Real document datasets with human line transcriptions are small or restrictively licensed. Without
document-style training data the model may learn the wrong prior (RISK_REGISTER R-06).

## Decision
- Build **GlyphSynth** (`glypharis.data.synth`, Phase 4): a renderer for document-style text lines and
  pages using open-licence fonts, licence-checked public-domain or permissive text, and document
  degradations, with exact labels and boxes.
- Keep SynthText as the brief requires. Phase 12 compares the two sources at equal budgets
  (25K → 250K), which turns RQ2 into a domain-vs-scale question.
- GlyphSynth-val uses held-out fonts and texts.

## Consequences
- Extra engineering in Phase 4, plus a licence manifest for fonts and texts.
- Synthetic-to-real gap is still a risk; it is measured on GDC and the external document benchmarks,
  not assumed away.
- The comparison is a candidate paper contribution (research/paper/introduction_notes.md).

## Revisit if
Phase 12 shows no benefit from GlyphSynth over SynthText on document validation slices at any budget.
