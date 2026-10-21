# ADR-0010: GDC capture runs as its own track; Phase 4 closes when the print pack is ready

- **Status:** Accepted (2026-10-10, Phase 4)

## Context
GDC (the document test set) is built by print-and-capture. The owner has no printer at home and will
print the pack once at a shop or library, then photograph and phone-scan every page. That takes days
of calendar time, during which every other Phase 4 deliverable is finished. Under the original gate,
Phase 4 (and therefore everything after it) would wait for the capture.

## Decision
1. Phase 4 closes when the GDC **print pack**, the capture checklist and the alignment tooling exist
   and are tested. Capturing, aligning, checking and freezing GDC becomes the tracked item
   **GDC-capture**, recorded in `docs/PHASE_STATUS.md`.
2. GDC-capture must be finished (GDC frozen with a hash manifest) before any GDC evaluation, i.e.
   before Phase 5 reports GDC numbers. Phase 5 may evaluate the other sets first.
3. Phase 6 (recognizer training on HierText + GlyphSynth, validation data only) may start while
   GDC-capture is open. It never reads GDC.
4. The GDC freeze still precedes any model being evaluated on GDC, so no model can influence its
   contents.

## Consequences
- Training is not blocked by printing logistics.
- Phase 4's exit criterion "GDC frozen with manifest" is met by the GDC-capture track instead of at
  the Phase 4 gate; PHASE_STATUS shows the open item until it closes.

## Revisit if
GDC-capture is still open when Phase 5 needs GDC numbers; then Phase 5 waits.
