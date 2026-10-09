# ADR-0008: Ship a usable engine (v0.1) before the research studies; CPU-first inference

- **Status:** Accepted (2026-10-10, Phase 0)

## Context
The consumer application (BookWorm) is still in development and cannot handle image-only pages until
GLYPHARIS exists; the owner decided not to use an off-the-shelf OCR as a stopgap. In the original order,
export and packaging (Phases 17, 18, 20) come after all studies (Phases 11–16), so a usable engine would
only appear near v1.0, weeks after a working pipeline exists at Phase 10.

Where the consumer will run inference (CPU server or GPU) is not decided.

## Decision
1. **Execution order changes; phase numbers do not** (so brief traceability is kept):

   ```
   0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 17 → 18 → 20  ⇒  v0.1
     → 11 → 12 → 13 → 14 → 15 → 16 → (17, 18, 20 re-run for the final model) → 19 → 21 → 22 → 23  ⇒  v1.0
   ```

2. **v0.1 scope:** the Phase 10 pipeline with the Phase 7 recognizer and Phase 8 detector, exported to
   ONNX with equivalence tests (Phase 17), wrapped in `GlypharisOCR` (Phase 18), with input validation
   and structured errors (Phase 20). Quality numbers for v0.1 come from **validation sets only**; v0.1
   does not trigger a test-set evaluation, so the test discipline of Phases 5 / 14 / 22 is unchanged.
3. Phases 17, 18 and 20 pass their gates at v0.1. For v1.0 they are re-run on the final model with the
   already-tested tooling and a new completion record; the earlier record is kept.
4. **Data scope for v0.1:** HierText, GlyphSynth and only the SynthText subset needed for the recognizer
   mix. The full 25K–250K scaling study stays in Phase 12.
5. **CPU-first inference.** Model sizes, input resolutions and the export path are chosen so that CPU
   inference through ONNX Runtime is the primary deployment target; GPU inference stays supported.
   CPU latency is the primary efficiency number (EXPERIMENT_PROTOCOL §3). A concrete CPU latency target is
   set after Phase 5 from measured baseline latencies on this laptop, not guessed now.
6. English printed text only for v0.1 and v1.0 (charset derived from data per brief §15).

## Consequences
- The consumer can integrate after Phase 20 instead of after Phase 23.
- v0.1 is not the paper's model: robustness, synthetic-data and ablation studies may change it. The
  version number and model card make this explicit, and v0.1 metrics are never presented as final.
- Some export/packaging work is done twice (once for v0.1, once for the final model). Accepted, because
  the second pass reuses tested code.
- The roadmap milestones are renumbered (docs/ROADMAP.md).

## Revisit if
The deployment target is decided (a GPU server would relax the CPU-first constraint), or Phase 5 shows
that the v0.1 pipeline cannot reach a useful quality level without the Phase 11–13 studies.
