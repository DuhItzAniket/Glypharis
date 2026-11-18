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
- **Status:** superseded (2026-11-15) by H1′, its operational form for Phase 14; the statement
  stays as written. Phase 10 evidence (HierText validation, not clean documents): page CER 0.48 of
  PP-OCRv5 mobile's on the Phase 5 pages, at 2.0 s per page in PyTorch against its 3.4 s
  (GW-E2E-001, GW-EFF-002).

### H1′ — H1 made testable for Phase 14 (RQ1, RQ4)

Written 2026-11-15, before the frozen model read any test set (`docs/PHASE_STATUS.md`, Phase 14,
item 5, has the full rule). Each half is **met** if its CI lies on the claimed side, **not met** if
on the other side, otherwise **not shown**; H1′ holds only if both halves are met.

- **CER half:** on GDC's clean slice (the `scan_app` and `photo_flat` captures, 53 pages), the CI of
  bag-of-lines page CER(GLYPHARIS) − 2 × CER(PP-OCRv5 mobile, GW-BASE-002), paired page bootstrap,
  strict track, lies below 0. Not shown if the normalised track or a source-document cluster
  bootstrap gives a different verdict.
- **Latency half:** on the Phase 5 validation profile pages (100 timed), the frozen pipeline in
  PyTorch on the CPU and a fresh GW-BASE-002 profile, one sitting, 10 threads each: the CI of the
  mean per-page latency difference lies below 0 (GW-EFF-005). The ONNX v1 build, when it exists, is
  judged by the same rule as a separate verdict.
- **Test:** Phase 14 (GW-E2E-007, GW-EFF-005, verdict in GW-E2E-009).
- **Status:** holds (2026-11-17, GW-E2E-009): both halves met.
  - CER half: −34.98 [−47.44, −24.08] points on the 53 clean-slice pages; normalised
    −34.73 [−47.17, −23.83]; source documents −34.98 [−50.22, −20.73]; all three met.
  - Latency half: −508.1 [−737.7, −256.5] ms per page (2,446.7 against 2,954.8 ms, 10 threads,
    one sitting); met. The ONNX v1 build is judged separately (next item).
  - ONNX build, separate verdict: not met (2026-11-18, GW-EFF-010) for the v1 fp32 ONNX bundle
    with manifest SHA-256 `875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`:
    +6,537.6 [+5,383.7, +7,806.3] ms per page (9,623.6 against 3,086.0 ms, 10 threads, one
    sitting). It applies to that bundle hash only and does not change H1′, which holds on the
    PyTorch build.

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
- **Status:** not supported on validation (2026-10-30), CRNN short budget at a fixed step budget
  (GW-DATA-022). H2′a: no size lowers document-slice CER against HierText only (+25K
  +0.04 [−0.03, +0.11], no clear effect); +50K, +100K and +250K raise it (+250K
  +0.39 [+0.25, +0.52] points). H2′b: 6 font families against 85, −0.09 [−0.21, +0.03] (no clear
  difference); capture effects lower GLYPH-C mCE by 0.31 points without a clean
  cost, so only that half holds. GDC (Phase 14) remains the document-domain test.
  Note (2026-11-15): the frozen model has no GlyphSynth (Phase 13 did not keep it), so H2′ cannot
  be tested on it in Phase 14.

### H3 — Document degradation augmentation improves robustness (RQ3)

- **Statement:** The document-degradation augmentation arm (D) lowers mean corruption CER on GLYPH-C
  compared with no augmentation (A), without raising clean validation CER by more than 5% relative.
- **Test:** Phase 11 (arms A–E, 3 seeds).
- **Status:** supported on validation (2026-10-29), CRNN short budget: mCE 10.04% against
  10.28%, −0.25 [−0.30, −0.19] points, all three D seeds below all three A seeds; clean validation
  CER 8.68% against 8.66% (+0.2%
  relative). The effect is small and comes from blur, low resolution and JPEG; geometric
  augmentation does far more (−1.67 [−1.73, −1.62]). Untested on the v0.1 recognizer (GW-DATA-009).

### H4 — Recognizer errors dominate on clean pages, detector errors on degraded ones (RQ5)

- **Statement:** In the oracle analysis, the recognizer accounts for the majority of page-level error on
  clean slices, while detection (missed / merged / split lines) accounts for the majority on photographed
  and heavily degraded slices.
- **Test:** Phase 10 (validation), Phase 15 (frozen model).
- **Status:** not supported on GDC (2026-11-17, post hoc, GW-E2E-010). From ground-truth boxes
  (GW-E2E-008) the recognizer makes 18.66% [11.55, 28.73] as many errors as the whole pipeline
  (GW-E2E-007) on all 106 GDC pages (source documents [11.51, 26.45]); 32.70% [16.64, 51.91] on
  the clean slice (`scan_app` + `photo_flat`), 45.14% [25.40, 58.41] on `scan_app` alone, 10.29%
  [5.24, 18.03] on the other five conditions and 1.03% [0.34, 2.93] on `rotated` (page CIs,
  strict bag of lines; `reports/error_cases/phase15/oracle_share.md`). The point estimate is
  below half in every slice, so on these numbers the recognizer does not account for the majority
  of page error on clean pages (the clean slice's page interval reaches 51.91% and `scan_app`'s
  58.41%); the rest is not all the detector's (it holds spurious predictions and crop shape too).
  GDC-scan, one document, information: 22.08% [12.65, 47.58]. No rule was registered, so this is
  a reading of the numbers, not a test.
  Before Phase 15: open; not supported by Phase 10's exploratory reading on HierText validation:
  recognition's share of page error is 0.20 to 0.29 on every blur quartile, the sharpest included
  (GW-E2E-002). HierText pages are mostly scene photographs; Phase 15 tests H4 on GDC.
  Note (2026-11-17, Phase 15 plan): no operational rule was registered before GW-E2E-008 was
  scored, and the published per-condition page CERs (GW-E2E-007, GW-E2E-008, scored in
  GW-E2E-009) already put the oracle below half the pipeline in every GDC condition, the
  clean slice included. No new version is written; H4 is closed post hoc from GW-E2E-010.

### H5 — Domain gap is large (RQ6)

- **Statement:** A model trained on HierText only shows a markedly higher CER on document-style data
  than on scene-style validation; adding GlyphSynth narrows that gap. (Wording updated 2026-10-10 before
  any experiment: the control was "HierText + SynthText", which ADR-0009 removed.)
- **Test:** Phase 12 / 14.
- **Status:** superseded (2026-11-15) by H5′; the statement stays as written.

### H5′ — H5 made testable for Phase 14 (RQ6)

Written 2026-11-15, before the frozen model read any test set.

- **Statement:** the frozen recognizer's strict CER on GDC's ground-truth crops (GW-REC-039) is at
  least 1.5 times its CER on the HierText validation evaluation crops under the same code and
  precision (GW-REC-038).
- **Rule:** the CI of CER_GDC − 1.5 × CER_val (independent bootstraps over lines) lies above 0:
  **met**; below 0: **not met**; otherwise **not shown**. Not shown, too, if a source-document
  cluster bootstrap on the GDC side gives a different verdict.
- **Not testable here:** H5's second half (GlyphSynth narrows the gap), since the frozen model has no
  GlyphSynth; Phase 12's validation evidence stands for it.
- **Known when written:** every Phase 5 baseline reads GDC's crops with a lower CER than HierText
  validation's (`reports/baselines/recognition.md`).
- **Test:** Phase 14 (verdict in GW-E2E-009).
- **Status:** not met (2026-11-17, GW-E2E-009): strict CER 4.17% on GDC's crops against 6.51% on
  the validation crops; CER_GDC − 1.5 × CER_val = −5.59 [−6.07, −5.01] points (lines) and
  −5.59 [−7.88, −2.52] (source documents). For this recognizer, GDC's printed documents are
  easier to read than HierText's scene-style lines, not harder.

### H6 — SVTR-style encoder beats CRNN at matched budget (RQ7)

- **Statement:** GLYPHARIS-Rec-M (SVTR-style) reaches lower validation CER than CRNN at equal training
  steps, with CPU latency within 1.5× of CRNN.
- **Test:** Phase 7 (GW-REC-*).
- **Status:** superseded for the latency half by H6′ (2026-11-17); the CER half stays
  exploratory, so H6 as a whole can at most be inconclusive. The statement stays as written.
  H6′, which supersedes H6's latency half, is not met (2026-11-18, GW-EFF-011); with its CER half
  exploratory, H6 as a whole stays at most inconclusive, as registered.
- **Evidence so far (exploratory):** Phase 13's row L4 (the SVTR-style model with its recipe,
  three seeds) lowers validation CER by 0.26 [0.19, 0.32] points against the CRNN at height 48
  (GW-REC-037), but it changes weight decay, learning rate and micro-batching together with the
  encoder, so it does not isolate the encoder. The latency half stays with Phase 16.

### H6′ — H6's latency half, made testable for Phase 16 (RQ7)

Written 2026-11-17, before any Phase 16 measurement. Known when written: Phase 7's PyTorch line
timings at 16 threads, SVTR 13.5 ms against CRNN 10.7 ms at height 32.

- **Statement:** the SVTR-style recognizer GW-REC-020 runs within 1.5 times the CPU latency of
  the CRNN GW-REC-003 (H6's matched-budget pair), in fp32 ONNX.
- **Rule:** both exported and each passing the logit and text checks against its own PyTorch
  outputs on the 5,000 seeded validation lines; `rec_samples`, 10 warm-up and 100 timed lines,
  batch 1, 10 threads, one sitting in the order CRNN, SVTR, SVTR, CRNN; met if the CI of
  mean(t_SVTR − 1.5 × t_CRNN) per line lies below 0, not met if above, otherwise not shown.
  PyTorch at 10 threads is information.
- **Test:** Phase 16 (GW-EFF-011).
- **Status:** not met (2026-11-18, GW-EFF-011): t_SVTR − 1.5 × t_CRNN = +16.78 [+12.02, +22.04] ms
  per line (32.01 against 10.16 ms, fp32 ONNX, 10 threads, one sitting); both exports passed their
  checks. PyTorch at 10 threads, information: 15.70 against 13.00 ms.

### H7 — INT8 quantisation is nearly free (RQ7)

- **Statement:** Dynamic INT8 quantisation of the ONNX recognizer reduces CPU latency by ≥ 25% with an
  absolute validation CER increase ≤ 0.005.
- **Test:** Phase 16 (GW-EFF-*), possibly earlier during v0.1 export (Phase 17).
- **Status:** superseded (2026-11-17) by H7′; the statement stays as written.

### H7′ — H7 made testable for Phase 16 (RQ7)

Written 2026-11-17, before any Phase 16 measurement.

- **Arms:** the v1 fp32 ONNX recognizer (GW-EFF-006) and its dynamic INT8 version: ONNX
  Runtime's `quant_pre_process` (defaults), then `quantize_dynamic(op_types_to_quantize=["MatMul"],
  per_channel=False, reduce_range=False, weight_type=QInt8, extra_options={"MatMulConstBOnly":
  True})`. Known in advance: the attention products have no constant input and stay fp32.
- **Latency half:** `rec_samples`, 10 warm-up and 100 timed lines, batch 1, 10 threads, one
  sitting in the order fp32, INT8, INT8, fp32: met if the CI of mean(t_INT8 − 0.75 × t_fp32)
  per line lies below 0.
- **Accuracy half:** the 5,000 seeded validation lines, strict track, greedy decoding:
  `paired_ratio_diff_ci(errors_INT8 − 0.005 × ref_chars, errors_fp32, ref_chars, "lines")`;
  met if the CI lies below 0.
- **Rule:** met, not met or not shown per half; H7′ holds only if both are met.
- **Test:** Phase 16 (GW-EFF-008).
- **Status:** holds (2026-11-18, GW-EFF-008): both halves met.
  - Latency half: t_INT8 − 0.75 × t_fp32 = −10.00 [−12.79, −7.37] ms per line (61.84 against
    95.78 ms, two runs averaged, one sitting); met.
  - Accuracy half: −0.512 [−0.541, −0.484] points (strict line CER 6.208% INT8 against 6.221%
    fp32 on the 5,000 lines); met.
  - The INT8 recognizer is a new version, not v1, measured on validation only; it does not
    replace v1.

### H8 — Raw CTC confidence is over-confident; temperature scaling fixes most of it (RQ8)

- **Statement:** Raw line confidence has ECE above 0.05 on validation; temperature scaling fitted on
  validation reduces ECE by at least half, and the reduction transfers to GDC.
- **Test:** Phase 10 (validation), Phase 14 (transfer).
- **Status:** superseded for the transfer by H8′ (2026-11-15). Validation part: supported on validation (2026-10-28): ECE 0.217 at T = 1, 0.089 at T = 2.35, a 59%
  reduction (GW-E2E-003). The transfer to GDC: see H8′ (Phase 14).

### H8′ — the calibration transfer, made testable for Phase 14 (RQ8)

Written 2026-11-15, before the frozen model read any test set.

- **Lines:** GDC's ground-truth crops through the frozen recognizer (GW-REC-039), vertical lines
  excluded as in every GDC metric, out-of-charset lines kept; GW-E2E-005's method (15 equal-mass
  bins, exact match on the strict track).
- **Transfers:** the CI of ECE(T = 2.35) − ECE(T = 1) lies below 0.
- **Transfers by half:** the CI of ECE(T = 2.35) − 0.5 × ECE(T = 1) lies below 0.
- **Rule:** line bootstrap as GW-E2E-005, with a page-cluster bootstrap that must give the same
  verdict, otherwise not shown. H8 counts as supported on GDC only if both are met. T stays 2.35
  whatever the result; dropping it would be a new model version.
- **Test:** Phase 14 (verdict in GW-E2E-009).
- **Status:** H8 not supported on GDC (2026-11-17, GW-E2E-009). GDC crops, ECE 0.234 at T = 1 and
  0.109 at T = 2.35.
  - Transfers: −0.125 [−0.132, −0.119] (lines), [−0.142, −0.101] (pages); met.
  - By half: −0.0081 [−0.0142, −0.0021] (lines, met), [−0.0168, +0.0044] (pages); the bootstraps
    disagree, so not shown.

### H9 — An ImageNet backbone helps the detector at equal steps (RQ1, ADR-0007)

- **Statement:** Initialising GLYPHARIS-Det's MobileNetV3-Large backbone from ImageNet weights raises
  validation F1@0.5 over random initialisation at an equal step budget.
- **Test:** Phase 8: GW-DET-001 (ImageNet) against GW-DET-002 (random initialisation, otherwise
  identical). One seed each, so exploratory.
- **Success criterion:** the paired page-bootstrap 95% CI of F1@0.5(GW-DET-001) − F1@0.5(GW-DET-002),
  on all HierText validation pages, lies entirely above 0 (`scripts/det_compare.py`).
- **Status:** supported, exploratory (2026-10-27). GW-DET-001 minus GW-DET-002: +0.0197
  [+0.0161, +0.0232] F1@0.5 over the 1,715 validation pages, one seed each
  (`reports/training/GW-DET-001_vs_GW-DET-002.md`). Written 2026-10-25, before the runs.

### H10 — MobileNetV3-Small is non-inferior to Large for detection (RQ7)

- **Statement:** With the same recipe and step budget, GLYPHARIS-Det with a MobileNetV3-Small backbone
  reaches validation F1@0.5 within 0.03 of the MobileNetV3-Large backbone (both ImageNet initialised).
- **Test:** Phase 8: GW-DET-001 (Large) against GW-DET-003 (Small). One seed each, so exploratory.
- **Success criterion:** the upper bound of the paired page-bootstrap 95% CI of F1@0.5(GW-DET-001) −
  F1@0.5(GW-DET-003), on all HierText validation pages, is ≤ 0.03 (`scripts/det_compare.py`).
- **Status:** not supported (2026-10-27). GW-DET-001 minus GW-DET-003: +0.0556 [+0.0505,
  +0.0616] F1@0.5, upper bound above the 0.03 margin; one seed each
  (`reports/training/GW-DET-001_vs_GW-DET-003.md`). Written 2026-10-25, before the runs.

**Operating point of record (H9, H10).** A detector run has exactly one operating point of record:
the one stored in its `best.pt` (`selected_op`). It is chosen from the operating points pre-registered
in the config (`dbnet15`, `inverse10`) as the one with the higher F1@0.5 on the per-epoch validation
subset at the best epoch (EXPERIMENT_PROTOCOL §7). Both hypotheses compare full validation of each
arm's `best.pt` at its stored operating point. The arg-max over operating points on the full
validation set is reported for information only and is never used for H9 or H10. Both differences are
also reported with each arm at each common fixed operating point, for transparency.

---

The numeric thresholds above are planning thresholds set before any data was seen. They may be refined
after Phase 5 baselines exist (brief §44: "refined after baseline results exist"); any refinement is a
new, dated hypothesis, and the original stays visible.
