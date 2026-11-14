# ADR-0013: GDC-scan is an internal test set, reported as a case study

- **Status:** Accepted (2026-11-15, Phase 14)

## Context
GDC's plan included a small set of the owner's own scans next to the print-and-capture pages
(IMPLEMENTATION_PLAN §4). It became a separate set, GDC-scan, built and frozen on its own
(`docs/PHASE_STATUS.md`, tracked item GDC-capture). What arrived on 2026-11-14 is one real document
of the owner's: a 28-page printed report, photographed page by page with a phone scanner app, whose
born-digital source exists. ADR-0005 fixes a role for every dataset and lets a role change only
through an ADR; GDC-scan did not exist when it was written.

## Decision
1. GDC-scan v1 is an **internal test** set under the same rules as GDC: frozen with a hash manifest
   before any model reads it (`data/manifests/gdc_scan/gdc_scan_v1.json`, 2026-11-14), never trained
   on, evaluated only in Phases 14 and 22, every access logged.
2. Its ground truth comes from the source's text layer, as for GDC, not from hand transcription; the
   owner confirmed that the printed document is that source with no edits after printing.
3. It is reported as a **case study**: one document, one printer and one font family, so its
   confidence intervals are conditional on that document and it carries no hypothesis verdict.
4. The document is personal. Its sources, photos, page file and per-line outputs stay local; every
   tracked file and commit message carries numbers only, never its text or images. Its source is on
   the training exclusion list by hash only.

## Consequences
- Phase 14 reports GDC-scan beside GDC with the same metrics, and Phase 15 uses it for numbers only.
- GDC's "old scans" composition target is still unmet: GDC-scan is a modern printed document.

## Revisit if
More documents of the owner's are added; that would be GDC-scan v2, frozen on its own.
