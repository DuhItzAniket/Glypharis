# Hypotheses

Written **before** the experiments that test them (pre-registration light, IMPLEMENTATION_PLAN §2).
A hypothesis is never edited after its experiment starts; if it needs to change, add a new one and mark
the old one superseded. Outcomes are filled in from generated reports with experiment IDs.

Status values: `open` → `testing` → `supported` | `not supported` | `inconclusive` | `superseded`.

Success criteria use the statistics in EXPERIMENT_PROTOCOL §4: "better" means the 95% paired-bootstrap
CI of the difference excludes zero, on validation unless stated otherwise.

---

### H1 — Lightweight pipeline is competitive on clean documents (RQ1, RQ4)

- **Statement:** On clean document pages, the GLYPHARIS pipeline reaches page CER within 2× of
  PaddleOCR PP-OCRv5 mobile while running at lower or equal CPU latency.
- **Test:** Phase 10 (validation), confirmed once in Phase 14 (GDC clean slice, external sets).
- **Success criterion:** stated ratio met on the GDC clean slice in Phase 14.
- **Status:** open

### H2 — Synthetic data helps, and its domain matters (RQ2)

- **H2a:** Adding synthetic data to HierText-train lowers validation CER on the document-style slice,
  with diminishing returns from 25K to 250K.
- **H2b:** At equal budget, GlyphSynth lowers document-slice CER more than SynthText; on HierText-val
  (scene-oriented) the difference is smaller or reversed.
- **Test:** Phase 12 scaling study (GW-DATA-*).
- **Success criterion:** H2a — CI-separated improvement at ≥ 1 budget vs HierText-only; H2b — CI-separated
  difference between sources at ≥ 2 of 4 budgets on the document slice.
- **Status:** superseded (2026-10-10) by H2′ — SynthText dropped for licence reasons (ADR-0009), so H2b
  cannot be tested. H2a carries over unchanged into H2′.

### H2′ — Document-style synthetic data helps and scales; diversity matters (RQ2, reframed)

- **H2′a:** Adding GlyphSynth to HierText-train lowers validation CER on the document-style slice, with
  diminishing returns from 25K to 250K lines.
- **H2′b:** At a fixed GlyphSynth size, rendering with many fonts lowers document-slice CER compared with
  a few fonts, and the degradation pipeline lowers CER on degraded slices without hurting clean ones.
- **Test:** Phase 12 (GW-DATA-*).
- **Success criterion:** H2′a — CI-separated improvement over HierText-only at ≥ 1 size; H2′b —
  CI-separated difference for each factor at the tested size.
- **Status:** open

### H3 — Document degradation augmentation improves robustness (RQ3)

- **Statement:** The document-degradation augmentation arm (D) lowers mean corruption CER on GLYPH-C
  compared with no augmentation (A), without raising clean validation CER by more than 5% relative.
- **Test:** Phase 11 (arms A–E, 3 seeds).
- **Status:** open

### H4 — Recognizer errors dominate on clean pages, detector errors on degraded ones (RQ5)

- **Statement:** In the oracle analysis, the recognizer accounts for the majority of page-level error on
  clean slices, while detection (missed / merged / split lines) accounts for the majority on photographed
  and heavily degraded slices.
- **Test:** Phase 10 (validation), Phase 15 (frozen model).
- **Status:** open

### H5 — Domain gap is large (RQ6)

- **Statement:** A model trained on HierText only shows a markedly higher CER on document-style data
  than on scene-style validation; adding GlyphSynth narrows that gap. (Wording updated 2026-10-10 before
  any experiment: the control was "HierText + SynthText", which ADR-0009 removed.)
- **Test:** Phase 12 / 14.
- **Status:** open

### H6 — SVTR-style encoder beats CRNN at matched budget (RQ7)

- **Statement:** GLYPHARIS-Rec-M (SVTR-style) reaches lower validation CER than CRNN at equal training
  steps, with CPU latency within 1.5× of CRNN.
- **Test:** Phase 7 (GW-REC-*).
- **Status:** open

### H7 — INT8 quantisation is nearly free (RQ7)

- **Statement:** Dynamic INT8 quantisation of the ONNX recognizer reduces CPU latency by ≥ 25% with an
  absolute validation CER increase ≤ 0.005.
- **Test:** Phase 16 (GW-EFF-*), possibly earlier during v0.1 export (Phase 17).
- **Status:** open

### H8 — Raw CTC confidence is over-confident; temperature scaling fixes most of it (RQ8)

- **Statement:** Raw line confidence has ECE above 0.05 on validation; temperature scaling fitted on
  validation reduces ECE by at least half, and the reduction transfers to GDC.
- **Test:** Phase 10 (validation), Phase 14 (transfer).
- **Status:** open

---

The numeric thresholds above are planning thresholds set before any data was seen. They may be refined
after Phase 5 baselines exist (brief §44: "refined after baseline results exist"); any refinement is a
new, dated hypothesis, and the original stays visible.
