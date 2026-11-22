# Results

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->

On HierText test GLYPHARIS v1 had a bag-of-lines page CER of 25.16% [23.44, 26.79], lower than that
of every baseline run there. On GDC's phone photographs it had <!-- derived: e2e_gly_gdc_bag --> 22.35% [15.87, 30.11], and no
difference is stated against the three PP-OCRv6 systems. On OmniDocBench its order-aware page
CER, <!-- derived: e2e_gly_odb_order --> 27.28% [23.95, 30.57], was higher than Tesseract 5's. All three intervals are over pages. From
ground-truth crops its CER on GDC was higher than that of three PaddleOCR recognizers. The
registered test H1′ holds on the PyTorch build, its separate ONNX verdict is not met, and no CPU
build of v1 reached ADR-0011's latency targets.

Section 2 reports v1 on HierText validation, Section 3 the Phase 5 baselines, Section 4 the test
sets read once in Phase 14 and Section 5 the official evaluators. Calibration, robustness, synthetic
data and ablations follow, and the last sections summarise `efficiency.md` and `error_analysis.md`.
Every number comes from a generated report and carries its experiment ID. Writing this section
needed no new run and no new access to test data.

Final validation (Phase 22) has been run; its record is `docs/FINAL_VALIDATION.md`. v1.0 is the
release of v1 (GW-REC-034 with GW-DET-001): the code under Apache-2.0, published on PyPI as
`glypharis` 1.0.0, and the v1 weights as the asset `glypharis-v1.tar.gz` of the GitHub release
`v1.0`, which `glypharis.download_model` fetches and checks against pinned hashes. Test results come
from the GPU strict-fp32 evaluation; equality with the bundle is shown on validation only. Phase 22
checked reproduction at the 0.05-point tolerance (`reports/final_validation/v1_reproduction.md`):
every prediction was the same, and it found 16 failures among the 1,381 point values checked, all
of them official OmniDocBench scores.

Conventions used throughout:

- v1 is the recognizer GW-REC-034 with the detector GW-DET-001 at Phase 9's post-processing and a
  long side of 1536 px, and line confidences at temperature T = 2.35 (GW-E2E-005), as frozen in
  `configs/final/glypharis_v1.yaml`. The code Phase 14 evaluated is the tag `phase14-eval`; the tag
  `v1-code` differs from it only in docstrings, comments and the default bundle name
  (`reports/export/v1-code/code_diff.json`).
- The track is strict unless a table says otherwise. CER, WER and F1 are percentages, and
  differences between them are percentage points.
- An interval is a 95% percentile bootstrap interval from 1,000 resamples. Each table or sentence
  says what was resampled: lines, words (IIIT5K), pages or source documents.
- A difference is GLYPHARIS minus the other system unless stated, so a negative CER difference
  means GLYPHARIS made fewer errors. Where the interval of a difference contains zero, no difference
  is stated.
- Bag-of-lines page CER matches predicted lines to reference lines by box and ignores order.
  Order-aware page CER also charges reading order. mCE is the mean corruption error over GLYPH-C,
  and ECE is expected calibration error.
- The words met, not met and holds are used only for the registered confirmatory tests in Section
  1.2. Every other interval is descriptive.

Data attribution. HierText is released under CC BY-SA 4.0; its images come from Open Images, which
lists them as CC BY 2.0 (`long2022hiertext`, `kuznetsova2020openimages`). The tables below that
are derived from HierText, including GLYPH-C, whose corruptions start from HierText validation lines
as well as GlyphSynth lines, are shared alike under CC BY-SA 4.0. GDC-scan is reported as numbers
only. The other data sets and their terms are listed in `DATASETS.md`.

## 1. Answers to the research questions

### 1.1 Summary

Hypothesis statuses are those recorded in `research/hypotheses.md`. The unit of each interval is
given in the cell.

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests-phase-14-plan-item-5 -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#iiit5k-alphanumeric-track-headline -->
<!-- trace: GW-E2E-009 reports/external/calibration.md -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx-not-met -->
<!-- trace: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-latency-met -->
<!-- trace: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-accuracy-met -->
<!-- trace: GW-EFF-011 reports/efficiency/phase16/verdicts.md#h6-not-met -->
<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages-one-page-at-a-time-lines-batched-as-the-pipeline-batches-them -->
<!-- trace: GW-DATA-022 reports/synthetic/phase12_comparison.md -->
<!-- trace: GW-DATA-009 reports/robustness/phase11_comparison.md -->
<!-- trace: GW-REC-037 reports/ablation/phase13_ladder.md -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#hiertext_test -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#gdc_test -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#funsd_test -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/oracle_share.md#gdc_test -->

| RQ | Question | Answer and evidence | Hypothesis | IDs |
|---|---|---|---|---|
| RQ1 | How well does a lightweight detector with a CTC line recognizer read documents? | Page CER was about a quarter on all three sets. GDC (<!-- derived: e2e_gly_gdc_pages --> 106 pages): bag of lines <!-- derived: e2e_gly_gdc_bag --> 22.35 [15.87, 30.11], order-aware <!-- derived: e2e_gly_gdc_order --> 25.37 [20.00, 30.80]. HierText test (<!-- derived: e2e_gly_ht_pages --> 1,634 pages): bag of lines <!-- derived: e2e_gly_ht_bag --> 25.16 [23.44, 26.79]. OmniDocBench (<!-- derived: e2e_gly_odb_pages --> 755 pages): order-aware <!-- derived: e2e_gly_odb_order --> 27.28 [23.95, 30.57]. Intervals over pages. | H1′ holds on the PyTorch build: CER half <!-- derived: h1_cer_strict_pages_v --> −34.98 [<!-- derived: h1_cer_strict_pages_lo --> −47.44, <!-- derived: h1_cer_strict_pages_hi --> −24.08] points (pages), latency half <!-- derived: h1_latency_v --> −508.11 [<!-- derived: h1_latency_lo --> −737.66, <!-- derived: h1_latency_hi --> −256.52] ms per page. The separate verdict for the ONNX bundle is not met: <!-- derived: h1_onnx_v --> +6,537.6 [<!-- derived: h1_onnx_lo --> +5,383.7, <!-- derived: h1_onnx_hi --> +7,806.3] ms per page. | GW-E2E-007, GW-E2E-009, GW-EFF-005, GW-EFF-010 |
| RQ2 | Does GlyphSynth help, how does it scale, and which generator factors matter? | No tested size lowered CER at a fixed step budget. On the HierText document slice arm `+25K` gave <!-- derived: p12_25k_doc --> +0.04 [−0.03, +0.11] and arm `+250K` <!-- derived: p12_250k_doc --> +0.39 [+0.25, +0.52] points (pages, CRNN short budget). Removing capture effects raised GLYPH-C mCE by <!-- derived: capture_mce_v --> +0.31 [<!-- derived: capture_mce_lo --> +0.27, <!-- derived: capture_mce_hi --> +0.35] points (lines); restricting fonts gave <!-- derived: fonts_doc_v --> −0.09 [<!-- derived: fonts_doc_lo --> −0.21, <!-- derived: fonts_doc_hi --> +0.03] on the slice (pages), so no difference is stated. v1 was trained without GlyphSynth. | H2′ not supported on validation. | GW-DATA-022 |
| RQ3 | Which degradation augmentations improve robustness? | Geometric augmentation had the largest point estimate of the three families. GLYPH-C mCE against no augmentation (lines, three seeds): geometric −1.67 [−1.73, −1.62], document degradation −0.25 [−0.30, −0.19], photometric −0.06 [−0.08, −0.02], all three <!-- derived: p11_full_mce --> −1.83 [−1.90, −1.75] points. Wider training rotation (ladder row L5, against L4) gave <!-- derived: ladder_l5_mce --> −1.49 [−1.53, −1.45] points. | H3 supported on validation (CRNN short budget). | GW-DATA-009, GW-REC-037 |
| RQ4 | How does GLYPHARIS compare with Tesseract, PaddleOCR and TrOCR-Small? | It depends on the set: its CER was lower than each baseline's on HierText test, higher than PP-OCRv6 medium's on GDC crops and higher than Tesseract's on OmniDocBench pages. HierText test crops: −1.03 [−1.14, −0.92] against PP-OCRv6 small and <!-- derived: cmp_v5m_rec_ht_cer --> −3.20 [−3.42, −2.99] against PP-OCRv5 mobile (lines). GDC crops: +0.49 [+0.23, +0.78] against PP-OCRv5 mobile and <!-- derived: cmp_v6m_rec_gdc_cer --> +0.97 [+0.76, +1.20] against PP-OCRv6 medium (lines). HierText test pages: <!-- derived: cmp_v6s_ht_cer --> −17.77 [−19.51, −16.01] against PP-OCRv6 small (pages). OmniDocBench order-aware page CER: <!-- derived: cmp_tess_odb_order --> +11.58 [+8.37, +14.88] against Tesseract and <!-- derived: cmp_v5m_odb_order --> −33.24 [−36.28, −30.01] against PP-OCRv5 mobile (pages). TrOCR-small-printed was run on crops only. Section 4 gives every pair. | H1′ as above. | GW-REC-039, GW-E2E-007, GW-BASE-001 to GW-BASE-007, GW-E2E-009 |
| RQ5 | Which failure modes dominate, and do they come from the detector or the recognizer? | Reference lines that no prediction matched were the largest part of page error on HierText test and had the largest point estimate on GDC, where the interval overlaps that of errors on matched lines. Their share of bag-of-lines page errors was <!-- derived: pipe_ht_missed --> 49.45% [47.47, 51.56] on HierText test, <!-- derived: pipe_gdc_missed --> 45.32% [36.14, 52.80] on GDC and <!-- derived: pipe_funsd_missed --> 40.22% [36.27, 42.88] on FUNSD (pages). From ground-truth boxes the recognizer made <!-- derived: oracle_gdc_all --> 18.66% [11.55, 28.73] as many errors as the whole pipeline on GDC, and 32.70% [16.64, 51.91] on its clean slice (pages). | H4 not supported on GDC, read post hoc; no rule was registered. | GW-E2E-010, GW-E2E-008, GW-REC-040 |
| RQ6 | How does document performance differ from scene-text performance? | The registered expectation that GDC's crops would be at least 1.5 times as hard as HierText validation's did not hold. Crop CER (lines): GDC <!-- derived: rec_gly_gdc_cer --> 4.17 [3.72, 4.67], FUNSD <!-- derived: rec_gly_funsd_cer --> 7.03 [5.71, 8.66], HierText validation <!-- derived: rec_gly_htval_cer --> 6.51 [6.36, 6.65], HierText test <!-- derived: rec_gly_ht_cer --> 6.03 [5.89, 6.18]. IIIT5K words, alphanumeric track: <!-- derived: rec_gly_iiit_cer --> 2.37 [2.06, 2.70]. No interval was computed for the difference between sets. | H5′ not met: CER_GDC − 1.5 × CER_val = −5.59 [−6.07, −5.01] points (lines), <!-- derived: h5_docs_v --> −5.59 [<!-- derived: h5_docs_lo --> −7.88, <!-- derived: h5_docs_hi --> −2.52] (documents). | GW-REC-038, GW-REC-039, GW-E2E-009 |
| RQ7 | What is the trade-off between accuracy, latency and memory for local use? | No CPU build of v1 reached ADR-0011's latency targets; they are recorded as missed and v1 keeps PyTorch as its default. v1 pages on the laptop CPU at 10 threads: PyTorch 2,566 [2,327, 2,829] ms, fp32 ONNX <!-- derived: cost_onnx_ms --> 8,574 [7,516, 9,766] ms (pages, one run each, HierText validation). See `efficiency.md`. Phones are the target hardware. Phase 22 measured the phone candidate's networks on two phones (GW-EFF-015): network time and memory only, without decoding, post-processing, cropping or reading order; v1 itself was not measured on a phone. | H7′ holds for the INT8 recognizer, a new version: <!-- derived: h7_latency_v --> −10.00 [<!-- derived: h7_latency_lo --> −12.79, <!-- derived: h7_latency_hi --> −7.37] ms per line and <!-- derived: h7_accuracy_v --> −0.512 [<!-- derived: h7_accuracy_lo --> −0.541, <!-- derived: h7_accuracy_hi --> −0.484] points. H6′ not met: <!-- derived: h6_v --> +16.78 [<!-- derived: h6_lo --> +12.02, <!-- derived: h6_hi --> +22.04] ms per line. | GW-EFF-006 to GW-EFF-011 |
| RQ8 | Is raw CTC confidence calibrated, and does temperature scaling fix it? | No: temperature scaling lowered ECE on HierText and GDC crops and raised it on IIIT5K. ECE of line confidence at T = 1 and T = 2.35: HierText validation crops <!-- derived: cal_htval_ece_t1 --> 0.2278 and <!-- derived: cal_htval_ece_t235 --> 0.0904, GDC crops <!-- derived: cal_gdc_ece_t1 --> 0.2345 and <!-- derived: cal_gdc_ece_t235 --> 0.1092. On IIIT5K words ECE rose from <!-- derived: cal_iiit_ece_t1 --> 0.0546 to <!-- derived: cal_iiit_ece_t235 --> 0.1118, <!-- derived: cal_iiit_ece_diff --> +0.0572 [+0.0431, +0.0716] (words). | H8′ transfers met; transfers by half not shown, so H8 is not supported on GDC. | GW-E2E-005, GW-REC-038, GW-REC-039, GW-E2E-009 |

### 1.2 Confirmatory tests

These tests were registered before the frozen model read any test set or, for Phase 16, before any
Phase 16 measurement. The H1′ latency half and ONNX verdict, the validation side of H5′, H6′ and H7′
use HierText validation data, which earlier phases had already used, and H6′ compares two other
recognizers, not the frozen model. A test is met when its interval lies on the claimed side of zero,
not met when it lies on the other side, and otherwise not shown.

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests-phase-14-plan-item-5 -->
<!-- trace: GW-E2E-009 reports/external/results.json#calibration/sets/gdc_test -->

The H1′ CER half used GDC's clean slice (the `scan_app` and `photo_flat` captures): 53 pages from 21
source documents. H8′ used GDC's <!-- derived: cal_gdc_lines --> 8,372 crop lines. H7′ accuracy was scored on <!-- derived: h7_accuracy_lines --> 5,000 validation
lines.

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests-phase-14-plan-item-5 -->
<!-- trace: GW-E2E-009 reports/external/results.json#calibration/sets/gdc_test -->
<!-- trace: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-latency-met -->
<!-- trace: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-accuracy-met -->
<!-- trace: GW-EFF-011 reports/efficiency/phase16/verdicts.md#h6-not-met -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx-not-met -->

| Test | Data and unit of resampling | Quantity | Result | Verdict |
|---|---|---|---|---|
| H1′, CER half | GDC clean slice | `CER(GLYPHARIS) − 2 × CER(PP-OCRv5 mobile)`, bag of lines, points | strict pages −34.98 [−47.44, −24.08]; normalised pages −34.73 [−47.17, −23.83]; strict documents <!-- derived: h1_cer_strict_docs_v --> −34.98 [<!-- derived: h1_cer_strict_docs_lo --> −50.22, <!-- derived: h1_cer_strict_docs_hi --> −20.73] | met |
| H1′, latency half | Phase 5 validation profile pages, PyTorch build, 10 threads, one sitting | GLYPHARIS − PP-OCRv5 mobile, ms per page, over pages | −508.11 [−737.66, −256.52] (means 2,446.7 and 2,954.8 ms) | met |
| H1′ | both halves | | | holds |
| H5′ | GDC crops against HierText validation crops | `CER_GDC − 1.5 × CER_val`, points | lines: −5.59 [−6.07, −5.01]; documents: −5.59 [−7.88, −2.52] | not met |
| H8′, transfers | GDC crops | `ECE(T = 2.35) − ECE(T = 1)` | lines: <!-- derived: cal_gdc_diff_lines --> −0.1253 [−0.1319, −0.1190]; pages: <!-- derived: cal_gdc_diff_pages --> −0.1253 [−0.1423, −0.1013] | met |
| H8′, transfers by half | GDC crops | `ECE(T = 2.35) − 0.5 × ECE(T = 1)` | lines: −0.0081 [−0.0142, −0.0021]; pages: <!-- derived: cal_gdc_half_pages --> −0.0081 [−0.0168, +0.0044] | not shown |
| H7′, latency | INT8 recognizer against v1 fp32 ONNX, validation lines | `t_INT8 − 0.75 × t_fp32`, ms per line, over lines | <!-- derived: h7_latency_v --> −10.00 [<!-- derived: h7_latency_lo --> −12.79, <!-- derived: h7_latency_hi --> −7.37] | met |
| H7′, accuracy | the same recognizers, validation lines | CER difference against a margin of `0.005`, strict track, points, over lines | <!-- derived: h7_accuracy_v --> −0.512 [<!-- derived: h7_accuracy_lo --> −0.541, <!-- derived: h7_accuracy_hi --> −0.484] | met |
| H7′ | both halves | | | holds |
| H6′ | GW-REC-020 against GW-REC-003, fp32 ONNX, validation lines | `t_SVTR − 1.5 × t_CRNN`, ms per line, over lines | <!-- derived: h6_v --> +16.78 [<!-- derived: h6_lo --> +12.02, <!-- derived: h6_hi --> +22.04] | not met |
| H1′, ONNX bundle | v1 fp32 ONNX against PP-OCRv5 mobile, 10 threads, one sitting | difference in ms per page, over pages | <!-- derived: h1_onnx_v --> +6,537.6 [<!-- derived: h1_onnx_lo --> +5,383.7, <!-- derived: h1_onnx_hi --> +7,806.3] (means <!-- derived: h1_onnx_mean_v1 --> 9,623.6 and <!-- derived: h1_onnx_mean_base --> 3,086.0 ms) | not met |

H1′ holds on the PyTorch build; the ONNX verdict applies to the bundle it names and does not change
that. PP-OCRv5 mobile ran PaddleOCR's two orientation models in the latency half, as it did for its
accuracy numbers. The H1′ latency half and the ONNX verdict come from different sittings, so their
latencies are not compared with each other.

## 2. Accuracy of v1 on validation

Three validation line CERs and two validation page CERs appear in this project. They measure
different sets of lines and pages and are not interchangeable.

<!-- trace: GW-REC-034 reports/training/GW-REC-034/report.md -->
<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#lines-line-cer-on-the-5000-seeded-lines-strict-greedy-one-line-at-a-time-line-time-on-100-rec_samples-lines -->
<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages-page-cer-and-e2e-f1-on-the-400-phase-5-pages-strict-page-time-on-the-110-profile-pages -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->

| Quantity | Items | Value % [95% CI] | ID |
|---|---|---|---|
| Line CER of the selected checkpoint, recognizer validation pack | 73,888 lines | 6.30 [6.17, 6.44] | GW-REC-034 |
| Line CER on the evaluation crops, the figure placed beside test results | 74,500 crops | 6.51 [6.36, 6.65] | GW-REC-038 |
| Line CER on the seeded subset used for runtime checks | <!-- derived: h7_accuracy_lines --> 5,000 lines | 6.22 [5.73, 6.71] | GW-EFF-009 |
| Page CER, bag of lines, all validation pages | <!-- derived: e2e_gly_htval_pages --> 1,715 pages | 27.57 [25.11, 30.36] | GW-E2E-006 |
| Page CER, bag of lines, the Phase 5 subset of pages | <!-- derived: phase5_pages --> 400 pages | <!-- derived: pareto_torch_cer --> 24.32 [20.87, 28.15] | GW-EFF-009 |

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#detection -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#track-strict -->
<!-- trace: GW-REC-034 reports/training/GW-REC-034/report.md -->
<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages-page-cer-and-e2e-f1-on-the-400-phase-5-pages-strict-page-time-on-the-110-profile-pages -->
<!-- trace: GW-E2E-011 reports/package/GW-E2E-011.json -->

The line intervals are over lines and the page intervals over pages. On all 1,715 pages E2E F1
was <!-- derived: e2e_gly_htval_e2e_f1 --> 58.24% [56.64, 59.91] and detection F1 at IoU 0.5 was 77.79% [76.21, 79.19] (GW-E2E-006); on the
Phase 5 subset of <!-- derived: phase5_pages --> 400 pages E2E F1 was <!-- derived: pareto_torch_e2e_f1 --> 60.50% [56.81, 63.96] (GW-EFF-009). The <!-- derived: rec_gly_htval_lines --> 74,500 HierText
validation crops are cut from the validation pages by the rule used for the test sets, and
the <!-- derived: val_pack_lines --> 73,888 validation lines are the recognizer's training-time pack; `DATASETS.md` reconciles the
two counts. The v1 package matched the frozen pipeline on those pages: its page CER differed by 0.0
points and 16,835 of 16,835 lines were identical in text and box (GW-E2E-011, PyTorch backend).

<!-- trace: GW-REC-037 reports/ablation/phase13_final.md -->

Across the three final seeds the recognizer's validation CER was 6.30, 6.28 and 6.30% (mean 6.29%),
GW-REC-034 to GW-REC-036. Against the CRNN baseline GW-REC-001 at the same budget of 30 epochs, the
three seeds gave −1.24 [−1.40, −1.11] points on validation CER and −1.08 [−1.41, −0.78] points on
the document slice, paired over pages (GW-REC-037). The baseline has one seed.

## 3. Baselines on validation (Phase 5)

<!-- trace: GW-BASE-002 reports/baselines/recognition.md#track-strict -->
<!-- trace: GW-BASE-002 reports/baselines/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->

The baselines were scored on seeded subsamples of HierText validation (<!-- derived: rec_v5m_htval_lines --> 10,000 lines and <!-- derived: e2e_v5m_htval_pages --> 400 pages)
and GLYPHARIS on the full evaluation sets in the broad tables (<!-- derived: rec_gly_htval_lines --> 74,500 crops and 1,715 pages). Those
rows do not cover the same items, so no difference is stated from these two tables. Differences with
paired intervals come from the test sets in Section 4. Latency is left out of this section: it was
measured in different sittings (`reports/benchmark_results/comparison_table.md`).

Line crops, strict track; intervals over lines.

<!-- trace-table: GW-REC-038 reports/external/recognition.md#track-strict GLYPHARIS v1=GW-REC-038;hiertext_val -->
<!-- trace-table: GW-BASE-001 reports/baselines/recognition.md#track-strict Tesseract 5=GW-BASE-001;hiertext_val -->
<!-- trace-table: GW-BASE-002 reports/baselines/recognition.md#track-strict PP-OCRv5 mobile=GW-BASE-002;hiertext_val -->
<!-- trace-table: GW-BASE-003 reports/baselines/recognition.md#track-strict PP-OCRv5 server=GW-BASE-003;hiertext_val -->
<!-- trace-table: GW-BASE-004 reports/baselines/recognition.md#track-strict TrOCR-small-printed=GW-BASE-004;hiertext_val -->
<!-- trace-table: GW-BASE-005 reports/baselines/recognition.md#track-strict PP-OCRv6 tiny=GW-BASE-005;hiertext_val -->
<!-- trace-table: GW-BASE-006 reports/baselines/recognition.md#track-strict PP-OCRv6 small=GW-BASE-006;hiertext_val -->
<!-- trace-table: GW-BASE-007 reports/baselines/recognition.md#track-strict PP-OCRv6 medium=GW-BASE-007;hiertext_val -->

| System | ID | Lines | CER % | WER % | Seq acc % |
|---|---|---|---|---|---|
| GLYPHARIS v1 | GW-REC-038 | 74,500 | 6.51 [6.36, 6.65] | 19.78 [19.49, 20.05] | 70.62 |
| Tesseract 5 | GW-BASE-001 | 10,000 | 24.35 [23.40, 25.42] | 44.67 [43.43, 46.09] | 40.94 |
| PP-OCRv5 mobile | GW-BASE-002 | 10,000 | 9.27 [8.63, 9.96] | 24.34 [23.39, 25.39] | 66.30 |
| PP-OCRv5 server | GW-BASE-003 | 10,000 | 14.65 [13.79, 15.52] | 43.80 [42.56, 45.01] | 56.08 |
| TrOCR-small-printed | GW-BASE-004 | 10,000 | 62.58 [60.97, 64.50] | 77.90 [76.60, 79.32] | 29.45 |
| PP-OCRv6 tiny | GW-BASE-005 | 10,000 | 11.10 [10.58, 11.65] | 31.12 [30.10, 32.26] | 57.44 |
| PP-OCRv6 small | GW-BASE-006 | 10,000 | 7.53 [7.01, 8.06] | 22.60 [21.72, 23.56] | 68.47 |
| PP-OCRv6 medium | GW-BASE-007 | 10,000 | 7.54 [6.94, 8.16] | 21.80 [20.83, 22.81] | 70.26 |

Pages, strict track; intervals over pages. PP-OCRv5 server and PP-OCRv6 medium were not run end to
end on validation because their recognizers were too slow on this CPU, and TrOCR-small-printed has
no detector.

<!-- trace-table: GW-E2E-006 reports/external/end_to_end.md#track-strict GLYPHARIS v1=GW-E2E-006;hiertext_val -->
<!-- trace-table: GW-BASE-001 reports/baselines/end_to_end.md#track-strict Tesseract 5=GW-BASE-001;hiertext_val -->
<!-- trace-table: GW-BASE-002 reports/baselines/end_to_end.md#track-strict PP-OCRv5 mobile=GW-BASE-002;hiertext_val -->
<!-- trace-table: GW-BASE-005 reports/baselines/end_to_end.md#track-strict PP-OCRv6 tiny=GW-BASE-005;hiertext_val -->
<!-- trace-table: GW-BASE-006 reports/baselines/end_to_end.md#track-strict PP-OCRv6 small=GW-BASE-006;hiertext_val -->

| System | ID | Pages | Page CER % (bag of lines) | E2E F1 % |
|---|---|---|---|---|
| GLYPHARIS v1 | GW-E2E-006 | 1,715 | 27.57 [25.11, 30.36] | 58.24 [56.64, 59.91] |
| Tesseract 5 | GW-BASE-001 | 400 | 94.50 [88.86, 100.55] | 12.65 [10.17, 15.41] |
| PP-OCRv5 mobile | GW-BASE-002 | 400 | 50.84 [45.61, 56.11] | 38.75 [34.89, 42.57] |
| PP-OCRv6 tiny | GW-BASE-005 | 400 | 49.65 [44.36, 54.88] | 38.69 [34.84, 42.43] |
| PP-OCRv6 small | GW-BASE-006 | 400 | 42.44 [37.58, 47.30] | 45.68 [42.16, 49.14] |

<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages-page-cer-and-e2e-f1-on-the-400-phase-5-pages-strict-page-time-on-the-110-profile-pages -->

On the same <!-- derived: phase5_pages --> 400 pages v1's page CER was 24.32% [20.87, 28.15] (Section 2). No paired interval
against the baselines exists for v1 on these pages, so no difference is stated.

## 4. External test sets

Each set was read once by the frozen model in Phase 14 (GW-REC-039 for crops, GW-E2E-007 for pages,
GW-E2E-009 for scoring). The baselines were scored on the same crops and pages with the same
metrics. `reports/benchmark_results/comparison_table.md` has the same rows with WER, sizes and
parameter counts; its latency columns come from HierText validation and other sittings and are not
used here.

<!-- trace: GW-E2E-009 reports/external/recognition.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#iiit5k-alphanumeric-track-headline -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#detection -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->

| Set | What it is | Pages | Crops |
|---|---|---|---|
| HierText test | scene-oriented pages, line annotations | 1,634 | <!-- derived: rec_gly_ht_lines --> 67,574 |
| GDC | phone captures of printed documents | 106 | <!-- derived: rec_gly_gdc_lines --> 8,372 |
| FUNSD | scanned forms | <!-- derived: funsd_pages --> 50 | <!-- derived: rec_gly_funsd_lines --> 2,647 |
| GDC-scan | one document, a case study; its intervals are conditional on it | 28 | <!-- derived: rec_gly_scan_lines --> 776 |
| OmniDocBench | document pages with layout-block references | 755 | not applicable |
| IIIT5K | cropped scene words | not applicable | 3,000 |

### 4.1 Recognition from ground-truth crops

GLYPHARIS v1 on each set, strict track, intervals over lines (words for IIIT5K).

<!-- trace-table: GW-REC-039 reports/external/recognition.md#track-strict HierText test=GW-REC-039;hiertext_test|GDC=GW-REC-039;gdc_test|FUNSD=GW-REC-039;funsd_test|GDC-scan=GW-REC-039;gdc_scan_test|IIIT5K (strict)=GW-REC-039;iiit5k_test -->

| Set | Lines | CER % | WER % | Seq acc % | Charset ceiling % |
|---|---|---|---|---|---|
| HierText test | 67,574 | 6.03 [5.89, 6.18] | 19.26 [18.95, 19.56] | 71.37 | 0.08 |
| GDC | 8,372 | 4.17 [3.72, 4.67] | 14.56 [13.44, 15.81] | 72.67 | 0.10 |
| FUNSD | 2,647 | 7.03 [5.71, 8.66] | 25.88 [23.83, 27.90] | 61.54 | 0.11 |
| GDC-scan | 776 | 1.02 [0.85, 1.20] | 6.08 [5.15, 7.00] | 77.06 | 0.33 |
| IIIT5K (strict) | 3,000 | 44.27 [42.46, 46.16] | 53.67 [51.90, 55.57] | 47.23 | 0.00 |

IIIT5K's headline is its alphanumeric track, on which both sides are lower-cased and characters
outside `0-9a-z` removed; the strict row charges case against the upper-case labels and is shown
only for completeness. The next two tables use the strict track for every other set and the
alphanumeric track for IIIT5K.

CER by system, in percent, with intervals over lines (words for IIIT5K).

<!-- trace: GW-E2E-009 reports/external/recognition.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#iiit5k-alphanumeric-track-headline -->

| System (ID) | HierText test | GDC | FUNSD | GDC-scan | IIIT5K, alphanumeric |
|---|---|---|---|---|---|
| GLYPHARIS v1 (GW-REC-039) | 6.03 [5.89, 6.18] | 4.17 [3.72, 4.67] | 7.03 [5.71, 8.66] | 1.02 [0.85, 1.20] | 2.37 [2.06, 2.70] |
| Tesseract 5 (GW-BASE-001) | 24.28 [23.89, 24.64] | 6.42 [5.91, 6.95] | 18.48 [16.10, 21.11] | 2.48 [2.05, 2.94] | 24.68 [23.22, 26.45] |
| PP-OCRv5 mobile (GW-BASE-002) | 9.23 [8.98, 9.50] | 3.68 [3.40, 3.95] | 18.34 [15.77, 21.13] | 1.41 [1.06, 1.88] | 1.22 [1.00, 1.42] |
| PP-OCRv5 server (GW-BASE-003) | not run | 11.24 [10.71, 11.80] | 29.20 [26.38, 32.03] | 11.89 [10.87, 13.00] | 1.49 [1.21, 1.77] |
| TrOCR-small-printed (GW-BASE-004) | 64.00 [63.01, 65.11] | 72.34 [71.65, 73.16] | 48.13 [45.75, 50.45] | 75.09 [73.85, 76.66] | 20.11 [15.62, 27.30] |
| PP-OCRv6 tiny (GW-BASE-005) | 10.82 [10.61, 11.02] | 4.00 [3.71, 4.34] | 10.35 [8.69, 12.47] | 0.86 [0.67, 1.09] | 3.05 [2.71, 3.40] |
| PP-OCRv6 small (GW-BASE-006) | 7.06 [6.88, 7.26] | 3.46 [3.13, 3.84] | 9.30 [7.26, 11.65] | 0.37 [0.27, 0.49] | 0.82 [0.65, 0.99] |
| PP-OCRv6 medium (GW-BASE-007) | not run | 3.20 [2.91, 3.56] | 17.91 [15.26, 20.55] | 0.48 [0.37, 0.60] | 0.67 [0.53, 0.83] |

GLYPHARIS minus each baseline, in points, with paired intervals over the same lines (words for
IIIT5K).

<!-- trace: GW-E2E-009 reports/external/comparisons.md -->

| Baseline (ID) | HierText test | GDC | FUNSD | GDC-scan | IIIT5K, alphanumeric |
|---|---|---|---|---|---|
| Tesseract 5 (GW-BASE-001) | −18.25 [−18.56, −17.92] | −2.25 [−2.50, −1.98] | −11.45 [−13.38, −9.64] | −1.46 [−1.84, −1.12] | <!-- derived: cmp_tess_rec_iiit_cer --> −22.31 [−23.99, −20.84] |
| PP-OCRv5 mobile (GW-BASE-002) | −3.20 [−3.42, −2.99] | +0.49 [+0.23, +0.78] | −11.31 [−13.55, −9.05] | −0.39 [−0.78, −0.09] | <!-- derived: cmp_v5m_rec_iiit_cer --> +1.15 [+0.88, +1.45] |
| PP-OCRv5 server (GW-BASE-003) | not run | −7.06 [−7.42, −6.73] | −22.18 [−24.65, −19.61] | −10.87 [−11.93, −9.88] | <!-- derived: cmp_v5s_rec_iiit_cer --> +0.88 [+0.54, +1.20] |
| TrOCR-small-printed (GW-BASE-004) | −57.96 [−59.05, −56.96] | −68.16 [−69.10, −67.29] | −41.10 [−43.29, −38.77] | −74.07 [−75.66, −72.81] | <!-- derived: cmp_trocr_rec_iiit_cer --> −17.74 [−25.04, −13.35] |
| PP-OCRv6 tiny (GW-BASE-005) | −4.79 [−4.89, −4.68] | +0.18 [−0.04, +0.40] | −3.32 [−3.95, −2.76] | +0.16 [−0.00, +0.32] | <!-- derived: cmp_v6t_rec_iiit_cer --> −0.67 [−0.96, −0.39] |
| PP-OCRv6 small (GW-BASE-006) | −1.03 [−1.14, −0.92] | +0.71 [+0.53, +0.91] | −2.27 [−3.43, −1.27] | +0.65 [+0.51, +0.78] | <!-- derived: cmp_v6s_rec_iiit_cer --> +1.55 [+1.28, +1.85] |
| PP-OCRv6 medium (GW-BASE-007) | not run | +0.97 [+0.76, +1.20] | −10.88 [−12.98, −8.70] | +0.54 [+0.39, +0.68] | <!-- derived: cmp_v6m_rec_iiit_cer --> +1.70 [+1.43, +1.98] |

<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#iiit5k-alphanumeric-track-headline -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#track-normalised -->

GLYPHARIS had the lowest crop CER of the systems run on HierText test and on FUNSD, and every paired
interval there lies below zero. On GDC's crops its CER was higher than that of PP-OCRv5 mobile
(<!-- derived: cmp_v5m_rec_gdc_cer --> +0.49 [+0.23, +0.78]), PP-OCRv6 small (<!-- derived: cmp_v6s_rec_gdc_cer --> +0.71 [+0.53, +0.91]) and PP-OCRv6 medium (<!-- derived: cmp_v6m_rec_gdc_cer --> +0.97 [+0.76,
+1.20]); against PP-OCRv6 tiny the interval <!-- derived: cmp_v6t_rec_gdc_cer --> +0.18 [−0.04, +0.40] contains zero, so no difference is
stated. On IIIT5K's alphanumeric track its CER was higher than that of PP-OCRv5 mobile, PP-OCRv5
server, PP-OCRv6 small and PP-OCRv6 medium, by <!-- derived: cmp_v5s_rec_iiit_cer_0_88 --> +0.88 to <!-- derived: cmp_v6m_rec_iiit_cer_1_70 --> +1.70 points. Its word accuracy there
was <!-- derived: rec_gly_iiit_word_acc --> 91.47% [90.43, 92.47] against 97.13% [96.57, 97.73] for PP-OCRv6 medium, a paired difference
of <!-- derived: cmp_v6m_rec_iiit_order --> −5.67 [−6.53, −4.77] points. On GDC-scan, one document, its CER was higher than that of PP-OCRv6
small (+0.65 [+0.51, +0.78]) and PP-OCRv6 medium (+0.54 [+0.39, +0.68]) and lower than that of
PP-OCRv5 mobile (−0.39 [−0.78, −0.09]).

<!-- trace: GW-E2E-009 reports/external/recognition.md#track-normalised -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->

TrOCR-small-printed's strict CER is high on every set. Its normalised CER on GDC was 13.07% [12.33,
13.98], and the paired difference on that track was <!-- derived: cmp_trocr_rec_gdc_normalised_cer --> −9.13 [−9.97, −8.37] points.

### 4.2 Pages

Bag-of-lines page CER matches predicted lines to reference lines by box and ignores order.
Order-aware page CER joins each system's lines in its own output order and compares them with the
reference in reading order; it exists only where the set has a reading order. On OmniDocBench the
reference unit is a layout block, so only the order-aware figure is defined, and it is this
project's metric, not that benchmark's protocol (its official scores are in Section 5). All
intervals are over pages.

Bag-of-lines page CER, %.

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->

| System (ID) | HierText test | GDC | FUNSD | GDC-scan |
|---|---|---|---|---|
| GLYPHARIS v1 (GW-E2E-007) | 25.16 [23.44, 26.79] | 22.35 [15.87, 30.11] | 32.05 [24.28, 39.85] | 4.60 [2.10, 8.21] |
| Tesseract 5 (GW-BASE-001) | 92.93 [90.39, 95.53] | 55.21 [44.70, 66.70] | 111.81 [100.43, 121.89] | 31.39 [18.97, 48.32] |
| PP-OCRv5 mobile (GW-BASE-002) | 50.95 [48.45, 53.58] | 27.83 [20.42, 36.44] | 30.61 [25.08, 36.35] | 12.94 [7.96, 18.51] |
| PP-OCRv5 server (GW-BASE-003) | not run | not run | 29.62 [24.78, 35.01] | not run |
| PP-OCRv6 tiny (GW-BASE-005) | 49.51 [46.92, 52.17] | 24.18 [16.80, 33.21] | 31.37 [24.54, 38.22] | 6.83 [3.85, 11.23] |
| PP-OCRv6 small (GW-BASE-006) | 42.93 [40.48, 45.29] | 22.29 [15.07, 30.91] | 29.51 [23.37, 35.98] | 6.68 [3.46, 10.55] |
| PP-OCRv6 medium (GW-BASE-007) | not run | 20.96 [13.57, 29.78] | 27.82 [22.69, 33.08] | 4.94 [2.24, 8.92] |

Order-aware page CER, %.

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->

| System (ID) | GDC | GDC-scan | OmniDocBench |
|---|---|---|---|
| GLYPHARIS v1 (GW-E2E-007) | 25.37 [20.00, 30.80] | 11.34 [7.10, 16.70] | 27.28 [23.95, 30.57] |
| Tesseract 5 (GW-BASE-001) | 31.90 [25.66, 38.51] | 17.81 [11.17, 26.88] | 15.70 [14.23, 17.37] |
| PP-OCRv5 mobile (GW-BASE-002) | 44.74 [38.40, 50.74] | 19.07 [10.56, 28.55] | 60.52 [58.54, 62.43] |
| PP-OCRv6 tiny (GW-BASE-005) | 44.06 [37.48, 50.43] | 16.40 [7.48, 26.82] | 60.62 [58.68, 62.55] |
| PP-OCRv6 small (GW-BASE-006) | 42.86 [36.16, 49.20] | 16.88 [8.20, 27.17] | 60.29 [58.23, 62.26] |
| PP-OCRv6 medium (GW-BASE-007) | 42.87 [36.18, 49.36] | 15.32 [6.64, 25.35] | 60.35 [58.32, 62.29] |

GLYPHARIS minus each baseline, points, with paired intervals over the same pages.

<!-- trace: GW-E2E-009 reports/external/comparisons.md -->

| Baseline (IDs compared) | HierText test, bag of lines | GDC, bag of lines | GDC, order-aware | FUNSD, bag of lines | GDC-scan, bag of lines | OmniDocBench, order-aware |
|---|---|---|---|---|---|---|
| Tesseract 5 (GW-E2E-007 minus GW-BASE-001) | −67.77 [−70.43, −65.10] | −32.86 [−43.33, −23.03] | −6.53 [−13.51, +0.08] | −79.76 [−93.34, −66.51] | −26.80 [−43.03, −15.44] | +11.58 [+8.37, +14.88] |
| PP-OCRv5 mobile (GW-E2E-007 minus GW-BASE-002) | −25.79 [−27.69, −24.00] | −5.48 [−9.94, −0.29] | −19.37 [−27.28, −12.16] | +1.44 [−2.32, +5.42] | −8.35 [−13.38, −4.32] | −33.24 [−36.28, −30.01] |
| PP-OCRv5 server (GW-E2E-007 minus GW-BASE-003) | not run | not run | not run | +2.44 [−3.75, +9.07] | not run | not run |
| PP-OCRv6 tiny (GW-E2E-007 minus GW-BASE-005) | −24.35 [−26.22, −22.47] | −1.83 [−5.91, +3.31] | −18.69 [−26.48, −11.47] | +0.69 [−2.98, +4.71] | −2.24 [−4.05, −0.87] | −33.34 [−36.37, −30.09] |
| PP-OCRv6 small (GW-E2E-007 minus GW-BASE-006) | −17.77 [−19.51, −16.01] | +0.06 [−4.02, +5.65] | −17.50 [−25.44, −10.15] | +2.55 [−0.98, +6.81] | −2.09 [−4.44, −0.31] | −33.01 [−36.11, −29.80] |
| PP-OCRv6 medium (GW-E2E-007 minus GW-BASE-007) | not run | +1.39 [−2.71, +6.88] | −17.50 [−25.52, −10.10] | +4.23 [−0.52, +8.76] | −0.34 [−1.40, +0.29] | −33.07 [−36.12, −29.81] |

Detection F1 at IoU 0.5 and E2E F1, %, over pages. E2E F1 counts a reference line as found when a
prediction overlaps it at IoU 0.5 or more and has a line CER of at most 10%.

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#detection -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->

| System (ID) | HierText test detection | HierText test E2E | GDC detection | GDC E2E | FUNSD detection | FUNSD E2E | GDC-scan detection | GDC-scan E2E |
|---|---|---|---|---|---|---|---|---|
| GLYPHARIS v1 (GW-E2E-007) | 79.13 [77.98, 80.26] | 60.18 [58.71, 61.72] | 80.80 [72.63, 88.07] | 69.36 [60.69, 77.48] | 79.07 [75.56, 82.16] | 62.81 [59.06, 66.48] | 95.92 [92.97, 98.16] | 89.84 [85.88, 93.62] |
| Tesseract 5 (GW-BASE-001) | 18.95 [17.43, 20.44] | 13.46 [12.12, 14.75] | 55.18 [47.15, 63.64] | 46.11 [38.03, 55.25] | 32.16 [27.27, 38.27] | 19.77 [16.19, 24.24] | 62.84 [50.80, 73.92] | 55.02 [44.86, 64.74] |
| PP-OCRv5 mobile (GW-BASE-002) | 54.95 [53.23, 56.73] | 40.47 [38.80, 42.10] | 66.04 [57.03, 75.01] | 59.80 [50.80, 68.56] | 81.81 [78.59, 84.88] | 62.65 [59.13, 66.13] | 78.63 [72.68, 83.65] | 74.02 [67.63, 79.79] |
| PP-OCRv5 server (GW-BASE-003) | not run | not run | not run | not run | 85.03 [82.10, 87.72] | 61.73 [58.26, 64.97] | not run | not run |
| PP-OCRv6 tiny (GW-BASE-005) | 55.75 [54.08, 57.44] | 39.50 [37.84, 41.06] | 68.05 [55.60, 79.75] | 60.94 [49.18, 72.24] | 81.87 [78.39, 84.92] | 59.29 [55.95, 62.56] | 89.64 [86.18, 92.76] | 84.53 [79.92, 88.73] |
| PP-OCRv6 small (GW-BASE-006) | 59.13 [57.42, 60.84] | 46.33 [44.69, 47.97] | 67.26 [55.51, 78.43] | 63.13 [51.70, 73.98] | 83.93 [81.08, 86.76] | 64.79 [61.69, 67.59] | 87.76 [83.85, 91.37] | 84.95 [81.52, 88.39] |
| PP-OCRv6 medium (GW-BASE-007) | not run | not run | 69.99 [57.83, 81.10] | 65.47 [53.61, 76.54] | 84.97 [82.18, 87.60] | 67.70 [64.63, 70.74] | 89.67 [86.23, 92.89] | 87.11 [82.76, 91.26] |

The cells marked not run were not scored. PP-OCRv5 server failed on GDC's photographs in Phase 5 and
could not process OmniDocBench's largest pages, and PP-OCRv5 server and PP-OCRv6 medium were too
slow to run on the HierText test set. TrOCR-small-printed has no detector, so it has no page
results.

<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#detection -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->

On HierText test the pipeline's bag-of-lines page CER was lower than that of every baseline run
there, by <!-- derived: cmp_v6s_ht_cer --> −17.77 [−19.51, −16.01] points against PP-OCRv6 small, the closest. On GDC it was lower
than Tesseract's (<!-- derived: cmp_tess_gdc_cer --> −32.86 [−43.33, −23.03]) and PP-OCRv5 mobile's (<!-- derived: cmp_v5m_gdc_cer --> −5.48 [−9.94, −0.29]); against the
three PP-OCRv6 systems every bag-of-lines interval contains zero, so no difference is stated. In
reading order on GDC it was lower than each of the four PaddleOCR systems run there by <!-- derived: cmp_v6m_gdc_order_17_50 --> 17.50
to <!-- derived: cmp_v5m_gdc_order_19_37 --> 19.37 points, each interval below zero, and against Tesseract the interval <!-- derived: cmp_tess_gdc_order --> −6.53 [−13.51, +0.08]
contains zero. On FUNSD every bag-of-lines interval against the five PaddleOCR systems contains
zero. Detection F1 on FUNSD was 79.07% [75.56, 82.16] for GLYPHARIS. The five PaddleOCR systems'
point estimates were all higher, from 81.81% [78.59, 84.88] for PP-OCRv5 mobile to <!-- derived: det_v5s_funsd_f1 --> 85.03% [82.10,
87.72] for PP-OCRv5 server, but no paired interval was computed for detection F1, so no difference
is stated.

<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->

On OmniDocBench the pipeline's order-aware page CER was higher than Tesseract's, <!-- derived: cmp_tess_odb_order --> +11.58 [+8.37,
+14.88] points, and lower than that of each of the four PaddleOCR systems run there (table above).
Those four read OmniDocBench at <!-- derived: e2e_v6s_odb_order_60_29 --> 60.29 to <!-- derived: e2e_v6t_odb_order_60_62 --> 60.62% whichever recognizer they used. The cause is not
established: a top-to-bottom box sort was proposed from their code (F26), and the measure registered
for it in Phase 15 did not decide it (F33, `error_analysis.md`, Section 3).

### 4.3 GDC by capture condition

GLYPHARIS v1 on GDC's capture conditions, strict track, intervals over pages. The oracle column
(GW-E2E-008) feeds ground-truth boxes to the same recognizer. These figures are information only:
conditions go with source documents, so no contrast between conditions is drawn.

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#gdc-per-capture-condition-information -->

| Condition | Pages | Pipeline (GW-E2E-007) page CER % (bag of lines) | Oracle boxes (GW-E2E-008) page CER % (bag of lines) | Pipeline (GW-E2E-007) page CER % (order-aware) |
|---|---|---|---|---|
| scan_app | <!-- derived: pages_scan_app --> 37 | 14.49 [7.73, 22.89] | 6.54 [2.21, 11.98] | 21.56 [13.27, 30.48] |
| photo_flat | <!-- derived: pages_photo_flat --> 16 | 20.67 [3.07, 45.41] | 3.05 [1.03, 6.23] | 19.57 [8.22, 29.53] |
| photo_angled | <!-- derived: pages_photo_angled --> 16 | 22.55 [5.82, 52.78] | 2.89 [0.81, 6.30] | 33.62 [18.22, 46.72] |
| distant | <!-- derived: pages_distant --> 11 | 14.21 [6.47, 25.90] | 3.14 [1.23, 7.65] | 18.63 [4.57, 37.50] |
| low_light | <!-- derived: pages_low_light --> 11 | 8.13 [3.52, 14.83] | 1.33 [0.92, 2.00] | 11.46 [4.78, 21.01] |
| curved | <!-- derived: pages_curved --> 8 | 64.60 [47.03, 82.96] | 8.05 [0.45, 23.10] | 36.80 [24.83, 50.24] |
| rotated | <!-- derived: pages_rotated --> 7 | 80.53 [62.13, 92.43] | 0.83 [0.26, 2.19] | 70.38 [49.17, 79.50] |

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#gdc-per-capture-condition-information -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#vertical-lines-gdc-gdc-scan -->

On the <!-- derived: pages_rotated --> 7 rotated pages the pipeline's bag-of-lines page CER was <!-- derived: cond_gly_rotated --> 80.53% [62.13, 92.43], the same
recognizer on ground-truth boxes reached <!-- derived: cond_oracle_rotated --> 0.83% [0.26, 2.19], and PP-OCRv6 small, which classifies
page orientation first, reached <!-- derived: cond_v6s_rotated --> 1.70% [0.42, 5.79] (pages). GLYPHARIS has no page-orientation step.
No paired interval was computed for this slice, so no difference is stated, and the effect of such a
step was not measured. GDC's 32 vertical lines are scored separately: GLYPHARIS found 18.75% [5.26,
40.00] of them at IoU 0.5 and PP-OCRv6 tiny 43.75% [20.00, 70.59]; the intervals overlap and no
paired interval was computed, so no difference is stated.

## 5. Official evaluators

The pinned official evaluators ran in their own environments (`reports/external/official.md`,
GW-E2E-009). They report no intervals, so no difference is stated from these tables.

HierText, line level, end to end, on all 1,634 test pages. Each predicted line was given to the
evaluator as a line holding one word, because no system produces word output.

<!-- trace-table: GW-E2E-009 reports/external/official.md#hiertext-line-level-end-to-end GLYPHARIS v1=GW-E2E-007|Tesseract 5=GW-BASE-001|PP-OCRv5 mobile=GW-BASE-002|PP-OCRv6 tiny=GW-BASE-005|PP-OCRv6 small=GW-BASE-006 -->

| System | Det-Fscore | Det-Precision | Det-Recall | E2E-Fscore | E2E-PQ | E2E-Precision | E2E-Recall | E2E-Tightness |
|---|---|---|---|---|---|---|---|---|
| GLYPHARIS v1 | 0.7837 | 0.8475 | 0.7288 | 0.5038 | 0.4037 | 0.5449 | 0.4685 | 0.8012 |
| Tesseract 5 | 0.1683 | 0.4901 | 0.1016 | 0.0857 | 0.0677 | 0.2496 | 0.0518 | 0.7903 |
| PP-OCRv5 mobile | 0.5057 | 0.5707 | 0.4540 | 0.3155 | 0.2011 | 0.3560 | 0.2832 | 0.6373 |
| PP-OCRv6 tiny | 0.5240 | 0.6203 | 0.4536 | 0.2961 | 0.1883 | 0.3505 | 0.2563 | 0.6360 |
| PP-OCRv6 small | 0.5555 | 0.6378 | 0.4920 | 0.3735 | 0.2341 | 0.4289 | 0.3308 | 0.6269 |

OmniDocBench, text block and reading order, on all <!-- derived: off_all_pages --> 755 pages (`quick_match`, `Edit_dist`; formula
and table metrics were off, so its Overall score cannot be computed). Lower is better.

<!-- trace-table: GW-E2E-009 reports/external/official.md#omnidocbench-text-block-and-reading-order-edit_dist-quick_match GLYPHARIS v1=GW-E2E-007;all|Tesseract 5=GW-BASE-001;all|PP-OCRv5 mobile=GW-BASE-002;all|PP-OCRv6 tiny=GW-BASE-005;all|PP-OCRv6 small=GW-BASE-006;all|PP-OCRv6 medium=GW-BASE-007;all -->

| System | Count | text_block page avg | text_block whole | reading_order page avg |
|---|---|---|---|---|
| GLYPHARIS v1 | 755 | 0.1007 | 0.1859 | 0.2443 |
| Tesseract 5 | 755 | 0.0966 | 0.0847 | 0.2352 |
| PP-OCRv5 mobile | 755 | 0.4217 | 0.6491 | 0.4490 |
| PP-OCRv6 tiny | 755 | 0.4039 | 0.6479 | 0.4383 |
| PP-OCRv6 small | 755 | 0.4044 | 0.6463 | 0.4374 |
| PP-OCRv6 medium | 755 | 0.4048 | 0.6464 | 0.4386 |

The same scores on the benchmark's <!-- derived: off_subset_pages --> 100-page subset, the set on which PP-OCRv5 server was tried and
could not finish.

<!-- trace-table: GW-E2E-009 reports/external/official.md#omnidocbench-text-block-and-reading-order-edit_dist-quick_match GLYPHARIS v1=GW-E2E-007;subset100|Tesseract 5=GW-BASE-001;subset100|PP-OCRv5 mobile=GW-BASE-002;subset100|PP-OCRv6 tiny=GW-BASE-005;subset100|PP-OCRv6 small=GW-BASE-006;subset100|PP-OCRv6 medium=GW-BASE-007;subset100 -->

| System | Count | text_block page avg | text_block whole | reading_order page avg |
|---|---|---|---|---|
| GLYPHARIS v1 | 100 | 0.1063 | 0.1881 | 0.2523 |
| Tesseract 5 | 100 | 0.1108 | 0.0623 | 0.2525 |
| PP-OCRv5 mobile | 100 | 0.3561 | 0.5670 | 0.4434 |
| PP-OCRv6 tiny | 100 | 0.3457 | 0.5615 | 0.4216 |
| PP-OCRv6 small | 100 | 0.3421 | 0.5669 | 0.4264 |
| PP-OCRv6 medium | 100 | 0.3525 | 0.5736 | 0.4243 |

On all <!-- derived: off_all_pages --> 755 pages Tesseract's point values were the lowest in every column, with GLYPHARIS next. On
the subset GLYPHARIS had the lowest point values for the two page averages and Tesseract for
`text_block whole`. No difference is stated.

<!-- trace: GW-E2E-018 reports/final_validation/v1_reproduction.md -->
These official OmniDocBench scores carry a disclosed limitation. In Phase 22 a re-run of v1 gave the
same predictions, but the strict reproduction verdict, which judged GLYPHARIS v1's values only, found
16 failures among the 1,381 point values checked, all of them GLYPHARIS's official OmniDocBench
scores, because the evaluator's per-page time limit depends on machine speed (GW-E2E-018,
`reports/final_validation/v1_reproduction.md`). Re-scoring the same saved Phase 14 files (GW-E2E-015,
`reports/final_validation/GW-E2E-015/verdict.json`) also changed Tesseract 5's official OmniDocBench
scores, through the same time-limit fallback, so the official GLYPHARIS-versus-Tesseract comparison
and ranking on these scores carry the same limitation. The owner accepted this as a disclosed
limitation of the official evaluator; the tables above are Phase 14's and stay as recorded.

## 6. Confidence calibration

The frozen recognizer's line confidences were scored at T = 1 and at T = 2.35, the temperature
fitted on half of HierText validation (GW-E2E-005). ECE uses 15 equal-mass bins, and a line counts
as right when it matches its label exactly on the strict track (alphanumeric track for IIIT5K). The
interval of the difference is over lines (words for IIIT5K), and only the GDC crops carry a
registered verdict (Section 1.2).

<!-- trace-table: GW-REC-038 reports/external/calibration.md hiertext_val=GW-REC-038;hiertext_val -->
<!-- trace-table: GW-REC-039 reports/external/calibration.md hiertext_test=GW-REC-039;hiertext_test|gdc_test=GW-REC-039;gdc_test|funsd_test=GW-REC-039;funsd_test|gdc_scan_test=GW-REC-039;gdc_scan_test|iiit5k_test=GW-REC-039;iiit5k_test -->
<!-- trace-table: GW-E2E-007 reports/external/calibration.md gdc_test (pipeline lines)=GW-E2E-007;gdc_test (pipeline lines) -->

| Set | Lines | Track | Accuracy | ECE T=1 | ECE T=2.35 | ECE(2.35) - ECE(1) [CI] |
|---|---|---|---|---|---|---|
| hiertext_val | 74,500 | strict | 0.7062 | 0.2278 | 0.0904 | −0.1374 [−0.1409, −0.1344] |
| hiertext_test | 67,574 | strict | 0.7137 | 0.2231 | 0.0920 | −0.1311 [−0.1345, −0.1277] |
| gdc_test | 8,372 | strict | 0.7267 | 0.2345 | 0.1092 | −0.1253 [−0.1319, −0.1190] |
| funsd_test | 2,647 | strict | 0.6154 | 0.3382 | 0.1227 | −0.2155 [−0.2181, −0.2052] |
| gdc_scan_test | 776 | strict | 0.7706 | 0.2019 | 0.1248 | −0.0771 [−0.0986, −0.0518] |
| iiit5k_test | 3,000 | alnum | 0.9147 | 0.0546 | 0.1118 | +0.0572 [+0.0431, +0.0716] |
| gdc_test (pipeline lines) | 9,057 | strict | 0.5315 | 0.4150 | 0.2489 | −0.1660 [−0.1678, −0.1643] |

<!-- trace: GW-E2E-005 reports/calibration/GW-E2E-005/calibration.md -->
<!-- trace: GW-E2E-009 reports/external/results.json#calibration/sets/gdc_test -->
<!-- trace: GW-E2E-009 reports/external/calibration.md -->

On the half of HierText validation that was not used for fitting (<!-- derived: cal_half_b_lines --> 37,912 lines), the raw confidences
had an ECE of 0.2140 at T = 1 and the fitted temperature gave <!-- derived: cal_val_fitted_ece --> 0.0886, a difference of −0.1254
[−0.1296, −0.1217] (lines). On GDC's crops the temperature lowered ECE as well: <!-- derived: cal_gdc_diff_lines --> −0.1253 [−0.1319,
−0.1190] over lines and <!-- derived: cal_gdc_diff_pages --> −0.1253 [−0.1423, −0.1013] over pages. Halving it there was not shown: the
line bootstrap of `ECE(T = 2.35) − 0.5 × ECE(T = 1)` gave <!-- derived: cal_gdc_half_lines --> −0.0081 [−0.0142, −0.0021], while the page
bootstrap gave <!-- derived: cal_gdc_half_pages --> −0.0081 [−0.0168, +0.0044]. H8 therefore counts as not supported on GDC. On IIIT5K's
words, where ECE at T = 1 was already <!-- derived: cal_iiit_ece_t1 --> 0.0546, T = 2.35 raised it to <!-- derived: cal_iiit_ece_t235 --> 0.1118.

## 7. Robustness

<!-- trace: data/manifests/processed/glyph_c.json data/manifests/processed/glyph_c.json -->

GLYPH-C v1 has <!-- derived: glyphc_lines --> 6,000 validation-derived source lines, 3,000 from HierText validation and 3,000 from
GlyphSynth validation, each copied clean and under 7 corruptions at 5 levels, 216,000 crops in all.
mCE is the mean over the corruption-level cells of each cell's micro CER, strict track.

Augmentation families on the CRNN at the short budget, three seeds per arm (GW-DATA-009). The
difference against arm A pools the seeds and resamples the source lines. Clean validation CER is the
HierText validation CER of each run's selected checkpoint.

<!-- trace-table: GW-DATA-009 reports/robustness/phase11_comparison.md A (none)=A (none);control|B (geometric)=B (geometric);helps|C (photometric)=C (photometric);no clear effect|D (degradation)=D (degradation);helps|E (full)=E (full);helps -->

| Arm | mCE % (pooled) | Δ mCE vs A [95% CI] | Clean val CER % (mean) | Report's label |
|---|---|---|---|---|
| A (none) | 10.28 | — | 8.66 | control |
| B (geometric) | 8.61 | −1.67 [−1.73, −1.62] | 8.34 | helps |
| C (photometric) | 10.23 | −0.06 [−0.08, −0.02] | 8.57 | no clear effect |
| D (degradation) | 10.04 | −0.25 [−0.30, −0.19] | 8.68 | helps |
| E (full) | 8.46 | −1.83 [−1.90, −1.75] | 8.31 | helps |

<!-- trace: GW-DATA-009 reports/robustness/phase11_comparison.md#cer-per-corruption-reported-not-part-of-the-decision-rule -->

The label column follows the report's rule (interval on one side of zero and every seed beyond every
control seed), which is why photometric augmentation is listed as no clear effect although its
interval excludes zero. H3 is recorded as supported on validation: arm D lowered mCE and its clean
CER stayed within the guard. Rotation had the highest CER of the seven corruptions in every arm.
With no augmentation, CER under rotation was <!-- derived: p11_a_none_rotate_38 --> 38.89% averaged over the five levels and <!-- derived: p11_a_none_rotate_61 --> 61.59% at
level 5, against <!-- derived: p11_a_none_clean --> 4.14% on the clean copies of the same lines. With the full preset it was 33.31%
and <!-- derived: p11_e_full_rotate_56 --> 56.49%, against <!-- derived: p11_e_full_clean --> 3.41% on the clean copies.

v1's recognizer on GLYPH-C, CER per corruption at each level (GW-REC-034). The columns are the
corruption levels, mildest first.

<!-- trace-table: GW-REC-034 reports/robustness/GW-REC-034.json#cer_per_corruption_level contrast=contrast|gaussian_blur=gaussian_blur|gaussian_noise=gaussian_noise|jpeg=jpeg|low_resolution=low_resolution|perspective=perspective|rotate=rotate -->

| Corruption | `1` | `2` | `3` | `4` | `5` |
|---|---|---|---|---|---|
| contrast | 1.93% | 1.93% | 1.95% | 2.01% | 2.31% |
| gaussian_blur | 1.93% | 1.96% | 2.06% | 2.26% | 2.63% |
| gaussian_noise | 1.99% | 2.16% | 2.39% | 2.74% | 3.12% |
| jpeg | 1.95% | 1.96% | 2.01% | 2.11% | 4.08% |
| low_resolution | 1.94% | 1.97% | 2.03% | 2.28% | 5.72% |
| perspective | 2.00% | 2.21% | 2.50% | 2.97% | 3.55% |
| rotate | 2.38% | 3.76% | 7.29% | 14.35% | 23.57% |

<!-- trace: GW-REC-034 reports/robustness/GW-REC-034.json -->

v1's CER on GLYPH-C's clean copies was 1.93% and its mCE was 3.54%. The clean figure is on a pack
that is half GlyphSynth lines, so it is not comparable with CER on HierText sets. Rotation again had
the highest CER: `rotate` at the top level gave 23.57%, against 5.72% for `low_resolution`, the
highest cell of any other corruption.

## 8. Synthetic data

The study (GW-DATA-022) added GlyphSynth lines to HierText training at a fixed step budget, with the
CRNN at the short budget, and scored three validation views. The document slice is the <!-- derived: p12_doc_pages --> 104 HierText
validation pages with at least 100 validation lines. Arms pool their seeds. Differences are against
HierText only, paired over pages.

<!-- trace-table: GW-DATA-022 reports/synthetic/phase12_comparison.md HierText only=HierText only;control|+25K=+25K;no clear effect|+50K=+50K;hurts|+100K=+100K;hurts|+250K=+250K;hurts|+100K\, 6 fonts=+100K\, 6 fonts;no clear effect|+100K\, no capture=+100K\, no capture;no clear effect -->

| Arm | Document slice | Δ vs HierText only | Report's label | HierText val | GlyphSynth val |
|---|---|---|---|---|---|
| HierText only | 11.06 | — | control | 8.31 | 1.97 |
| +25K | 11.09 | +0.04 [−0.03, +0.11] | no clear effect | 8.29 | 0.94 |
| +50K | 11.18 | +0.12 [+0.03, +0.22] | hurts | 8.30 | 0.73 |
| +100K | 11.22 | +0.16 [+0.06, +0.26] | hurts | 8.38 | 0.57 |
| +250K | 11.44 | +0.39 [+0.25, +0.52] | hurts | 8.56 | 0.38 |
| +100K, 6 fonts | 11.13 | +0.07 [−0.04, +0.18] | no clear effect | 8.40 | 0.61 |
| +100K, no capture | 11.09 | +0.04 [−0.07, +0.15] | no clear effect | 8.38 | 0.65 |

<!-- trace: GW-DATA-022 reports/synthetic/phase12_comparison.md -->

No size of added GlyphSynth lowered CER on the document slice, and each arm from `+50K` on had a
higher CER than HierText only, so H2′a is recorded as not supported on validation. On GlyphSynth's
own validation lines CER was <!-- derived: p12_control_gs --> 1.97 for HierText only and <!-- derived: p12_250k_gs --> 0.38 for arm `+250K`, <!-- derived: glyphsynth_250k_v --> −1.58 [<!-- derived: glyphsynth_250k_lo --> −1.66, <!-- derived: glyphsynth_250k_hi --> −1.50]
points over lines. Two generator factors were tested on arm `+100K`. Restricting the fonts to `6`
families changed document-slice CER by <!-- derived: fonts_doc_v --> −0.09 [<!-- derived: fonts_doc_lo --> −0.21, <!-- derived: fonts_doc_hi --> +0.03], so no difference is stated. Removing
the capture effects raised GLYPH-C mCE by +0.31 [+0.27, +0.35] points. H2′b needed a difference for
each factor, so H2′ is recorded as not supported. No delivered model was trained on GlyphSynth, so
H2′ could not be tested on v1 in Phase 14.

## 9. Ablations

The ladder (GW-REC-037) adds one change at a time to the row it is built on, with three seeds per
row at the short budget. Differences are pooled over seeds and paired over pages (validation and
document slice) or over GLYPH-C's source lines (mCE). The labels `helps` and `no clear effect`
follow the Phase 7 rule: the interval on one side of zero and every seed beyond every parent seed.
Row L3 is built on L1, not L2, because L2 was not kept.

<!-- trace-table: GW-REC-037 reports/ablation/phase13_ladder.md L0;start=L0 baseline (CRNN\, no augmentation)|L1;yes=L1 + augmentation (full preset)|L2;no=L2 + 25K GlyphSynth lines|L3;yes=L3 + input height 48|L4;yes=L4 full model (+ SVTR encoder\, weight decay 1e-2)|L5;yes=L5 + wider rotation -->

| Row | Change | Val CER % | Δ val vs parent | Document slice % | Δ doc | GLYPH-C mCE % | Δ mCE | Kept |
|---|---|---|---|---|---|---|---|---|
| L0 | CRNN, no augmentation | 8.66 | — | 11.46 | — | 10.28 | — | start |
| L1 | + augmentation (full preset) | 8.31 | −0.36 [−0.40, −0.31] helps | 11.06 | −0.40 [−0.54, −0.26] helps | 8.46 | −1.83 [−1.91, −1.75] helps | yes |
| L2 | + GlyphSynth lines | 8.29 | −0.02 [−0.05, +0.01] no clear effect | 11.09 | +0.04 [−0.03, +0.11] no clear effect | 7.19 | −1.27 [−1.32, −1.21] helps | no |
| L3 | + input height | 7.83 | −0.48 [−0.54, −0.42] helps | 10.66 | −0.39 [−0.54, −0.25] helps | 6.83 | −1.63 [−1.68, −1.58] helps | yes |
| L4 | full model: SVTR-style encoder, weight decay, its learning rate and micro-batching | 7.57 | −0.26 [−0.32, −0.19] helps | 10.36 | −0.31 [−0.53, −0.11] helps | 6.29 | −0.54 [−0.59, −0.49] helps | yes |
| L5 | + wider rotation | 7.45 | −0.12 [−0.16, −0.08] helps | 10.27 | −0.09 [−0.17, −0.01] helps | 4.80 | −1.49 [−1.53, −1.45] helps | yes |

<!-- trace: GW-REC-037 reports/ablation/phase13_ladder.md -->
<!-- trace: GW-REC-022 reports/training/compare/GW-REC-022_vs_GW-REC-006.md -->

L1 added the full augmentation preset, L2 added `25K` GlyphSynth lines, L3 raised the input height
to `48` and L5 widened training rotation. L4's step cannot be credited to the encoder alone: it
changes the encoder, the weight decay, the learning rate and the micro-batching together, as does
the one-seed comparison of GW-REC-022 with GW-REC-006 (−0.27 [−0.32, −0.22] points, paired over
lines). H6's CER half stays exploratory for that reason.

<!-- trace: GW-REC-037 reports/ablation/phase13_final.md -->

The final recognizer used L4's recipe with L5's rotation at the full 30-epoch budget and three
seeds; its validation results are in Section 2. Its GLYPH-C mCE per seed was 3.54, 3.53 and 3.53%.

The ablation design called for a baseline, then augmentation, synthetic data, recognizer changes,
detector changes and preprocessing, one controlled change at a time. The ladder covers the
recognizer side: L0 is the baseline, L1 augmentation, L2 synthetic data, L4 the recognizer change,
and the input height of L3 is the preprocessing row. Detector changes are not in the ladder, which
is recognizer-only. They were tested separately, one seed each, on HierText validation pages, with
intervals over pages. GW-DET-002, GW-DET-003 and GW-DET-007 used all of the <!-- derived: det_val_pages --> 1,715 validation pages;
GW-DET-004 and GW-DET-005 were checked on half B, the <!-- derived: det_half_b_pages --> 858 pages not used for tuning.

<!-- trace: GW-DET-002 reports/training/GW-DET-001_vs_GW-DET-002.md -->
<!-- trace: GW-DET-003 reports/training/GW-DET-001_vs_GW-DET-003.md -->
<!-- trace: GW-DET-004 reports/error_cases/detector/GW-DET-004/pair_B.json -->
<!-- trace: GW-DET-005 reports/error_cases/detector/GW-DET-005/pair_B.json -->
<!-- trace: GW-DET-007 reports/error_cases/detector/GW-DET-007/pair.json -->

| ID | Change | Comparison | Difference in F1 at IoU 0.5 | Outcome |
|---|---|---|---|---|
| GW-DET-002 | backbone not initialised from ImageNet | GW-DET-001 minus GW-DET-002 | <!-- derived: det002_diff --> +0.0197 [+0.0161, +0.0232] | H9 supported, exploratory |
| GW-DET-003 | MobileNetV3-Small backbone | GW-DET-001 minus GW-DET-003 | <!-- derived: det003_diff --> +0.0556 [+0.0505, +0.0616] | H10 not supported |
| GW-DET-004 | post-processing tuned on half A | tuned minus the stored setting | <!-- derived: det004_diff --> +0.0316 [+0.0247, +0.0386] | adopted |
| GW-DET-005 | larger long side | `1536` minus `1280` | <!-- derived: det005_diff --> +0.0177 [+0.0131, +0.0221] | adopted |
| GW-DET-007 | labels for smaller text, wider training rotation | GW-DET-007 minus GW-DET-001 | <!-- derived: det007_diff --> −0.0111 [−0.0146, −0.0075] | not adopted |

## 10. Efficiency

`efficiency.md` has the full account (GW-EFF-006 to GW-EFF-011). All of it is HierText validation
on one laptop.

<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages-one-page-at-a-time-lines-batched-as-the-pipeline-batches-them -->
<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages-page-cer-and-e2e-f1-on-the-400-phase-5-pages-strict-page-time-on-the-110-profile-pages -->
<!-- trace: GW-EFF-007 reports/efficiency/phase16/README.md#adr-0011s-targets-against-each-v1-build -->

- v1 in PyTorch on the CPU at 10 threads took <!-- derived: cost_torch_ms --> 2,566 [2,327, 2,829] ms per page and the fp32 ONNX
  build <!-- derived: cost_onnx_ms --> 8,574 [7,516, 9,766] ms (over pages, one run each, GW-EFF-007). In the mirrored sitting of
  GW-EFF-009 the two builds took <!-- derived: pareto_torch_ms --> 2,637.3 [2,395.4, 2,909.1] and <!-- derived: pareto_onnx_ms --> 9,739.4 [8,510.8, 11,113.1] ms. Page
  latencies from different sittings are not compared here.
- ADR-0011's laptop targets are a line mean of at most 25 ms and p95 of at most 70 ms, a page mean
  of at most 1.5 s and p95 of at most 2.7 s, and peak RAM on pages below 1,836 MiB. v1 in PyTorch at
  10 threads measured 33.9 and 77.6 ms per line, <!-- derived: adr_torch_page_mean --> 2,565.7 and <!-- derived: adr_torch_page_p95 --> 5,127.1 ms per page and 1,641.8 MiB, so
  it is over all four latency targets and within the RAM target. Both ONNX builds are over all five.
  These targets are recorded as missed, and v1 keeps PyTorch as its default (ADR-0011, decision of
  2026-11-18).
- Phones are the target hardware. Phase 22 measured the phone candidate's networks on two phones
  (GW-EFF-015): network time and memory only, without decoding, post-processing, cropping or reading
  order; v1 itself was not measured on a phone.
- The all-ONNX fused-attention build took <!-- derived: pareto_fused_ms --> 4,020.8 [3,604.2, 4,479.8] ms per page in the GW-EFF-009
  sitting, at v1's page CER of 24.32% on the <!-- derived: phase5_pages --> 400 pages. The phone candidate chosen in Phase 22 is
  the all-ONNX build with fused attention, reading one line per recognizer run (`fused-single-line`,
  GW-EFF-013). A new version, not v1. Phase 22 evaluated it once per test set, and it met H11's six
  non-inferiority tests against v1 (GW-E2E-021, `reports/final_validation/candidate/h11.md`); its
  results sit beside v1's and replace none of them.

## 11. Error analysis

`error_analysis.md` has the full account (GW-E2E-010, GW-REC-040, GW-REC-041). In brief, for the
frozen pipeline:

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#hiertext_test -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#gdc_test -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/oracle_share.md#gdc_test -->

- Missed reference lines were the largest share of bag-of-lines page error on HierText test, <!-- derived: pipe_ht_missed --> 49.45%
  [47.47, 51.56]. On GDC they had the largest point estimate, <!-- derived: pipe_gdc_missed --> 45.32% [36.14, 52.80], with an
  interval that overlaps the share of errors on matched lines, <!-- derived: pipe_gdc_matched --> 27.55% [15.80, 42.07] (pages).
- From ground-truth boxes the recognizer made <!-- derived: oracle_gdc_all --> 18.66% [11.55, 28.73] as many errors as the pipeline
  on GDC (pages). The rest is not a pure detector share, since it also includes spurious predictions
  and crop shape.

## 12. What these results do not show

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests-phase-14-plan-item-5 -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#detection -->

Final validation (Phase 22) has been run; its record is `docs/FINAL_VALIDATION.md`. Configuration
frozen and recorded. Phase 22 re-ran it: the predictions equal Phase 14's in every test file, and the
strict verdict failed only on official OmniDocBench scores, through the evaluator's time limit; the
owner accepted this as a disclosed limitation (`reports/final_validation/v1_reproduction.md`). The
test results describe the frozen model on these sets. Set sizes limit them:
GDC has <!-- derived: det_gly_gdc_pages --> 106 pages from <!-- derived: gdc_documents --> 21 documents, GDC-scan is one document and FUNSD has <!-- derived: funsd_pages --> 50 forms. The baselines
were run as published with the settings recorded in `reports/baselines/README.md`; they were not
retrained for these tasks.
