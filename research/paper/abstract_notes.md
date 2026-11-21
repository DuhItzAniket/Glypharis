# Abstract notes

Status: these results are Phase 14's, and Phase 22 left them as recorded. Final validation (Phase 22)
has been run; its record is `docs/FINAL_VALIDATION.md`. No weights are distributed; release is
Phase 23.

HierText is released under CC BY-SA 4.0; its images come from Open Images, which lists them as
CC BY 2.0. Figures derived from HierText are shared alike.

## Abstract

GLYPHARIS is a two-stage OCR engine for English printed text in page images, built to be called by a
document-reading application. A DBNet-style detector with a MobileNetV3-Large backbone finds text
lines, and an SVTR-style encoder with CTC decoding reads each rectified line. Both models were
trained on one laptop GPU, on HierText only. The detector backbone started from ImageNet weights; no
OCR weights from another system were used.

The frozen model, v1, was evaluated once on each held-out set. Results are character error rates
(CER) on the strict track, where case, punctuation and symbols all count. Bag-of-lines page CER
matches predicted lines to reference lines by box and ignores their order; order-aware CER also
charges reading order. Brackets are 95% bootstrap intervals, over pages
unless stated, and a difference is GLYPHARIS minus the other system, in percentage points.

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
HierText test holds <!-- derived: ht_pages --> 1,634 mostly scene-style pages. There GLYPHARIS had a
bag-of-lines CER of <!-- derived: ht_glyph --> 25.16% [23.44, 26.79]. PP-OCRv6 small, the lowest of
the four baselines run end to end on that set, had <!-- derived: ht_small --> 42.93% [40.48, 45.29],
and the paired difference was <!-- derived: d_ht_small --> −17.77 [−19.51, −16.01] points.

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
The Glypharis Document Challenge (GDC) is a smaller test on documents: <!-- derived: gdc_pages -->
106 phone captures of pages from <!-- derived: gdc_documents --> 21 printed documents. Its
bag-of-lines CER was <!-- derived: gdc_glyph --> 22.35% [15.87, 30.11]. PP-OCRv6 medium, the lowest
baseline on this measure, had <!-- derived: gdc_medium --> 20.96% [13.57, 29.78], and no difference
is stated (<!-- derived: d_gdc_medium --> +1.39 [−2.71, +6.88] points). Order-aware CER on GDC
was <!-- derived: gdc_glyph_oa --> 25.37% [20.00, 30.80]. Tesseract 5 was the lowest baseline on that
measure, at <!-- derived: gdc_tess_oa --> 31.90% [25.66, 38.51], and no difference from it is stated
(<!-- derived: d_gdc_tess_oa --> −6.53 [−13.51, +0.08] points). PP-OCRv6 medium
had <!-- derived: gdc_medium_oa --> 42.87% [36.18, 49.36], a difference
of <!-- derived: d_gdc_medium_oa --> −17.50 [−25.52, −10.10] points.

<!-- trace: GW-E2E-009 reports/external/recognition.md -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
Several comparisons went against GLYPHARIS. Reading GDC's ground-truth line crops, its CER
was <!-- derived: crop_glyph --> 4.17% [3.72, 4.67] against <!-- derived: crop_medium --> 3.20%
[2.91, 3.56] for PP-OCRv6 medium, a difference of <!-- derived: d_crop_medium --> +0.97
[+0.76, +1.20] points, with intervals over lines. On IIIT5K's cropped words, alphanumeric track, its
word accuracy was <!-- derived: iiit_glyph --> 91.47% [90.43, 92.47]
against <!-- derived: iiit_medium --> 97.13% [96.57, 97.73] for the same baseline, a difference
of <!-- derived: d_iiit_medium --> −5.67 [−6.53, −4.77] points, with intervals over words. On
OmniDocBench, with the project's order-aware CER and not the official protocol, it
had <!-- derived: odb_glyph --> 27.28% [23.95, 30.57] against <!-- derived: odb_tess --> 15.70%
[14.23, 17.37] for Tesseract 5 (<!-- derived: d_odb_tess --> +11.58 [+8.37, +14.88]
points).

<!-- trace: GW-E2E-009 reports/external/official.md#omnidocbench -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#detection -->
Two further results point the same way, but no difference is stated. OmniDocBench's official
text-block edit distance, where lower is better and no interval exists,
was <!-- derived: off_tess_whole --> 0.0847 for Tesseract 5 against <!-- derived: off_glyph_whole -->
0.1859 for GLYPHARIS over the whole set, and <!-- derived: off_tess_page --> 0.0966
against <!-- derived: off_glyph_page --> 0.1007 as a page average. FUNSD detection F1 at IoU
0.5 was <!-- derived: funsd_glyph --> 79.07% [75.56, 82.16], below the point
estimates of all five PaddleOCR systems, which ran from <!-- derived: funsd_mobile --> 81.81% for
PP-OCRv5 mobile to <!-- derived: funsd_server --> 85.03% for PP-OCRv5 server.

<!-- trace: GW-E2E-018 reports/final_validation/v1_reproduction.md -->
In Phase 22 a re-run of v1 gave the same predictions as Phase 14, but the strict reproduction
verdict, which judged GLYPHARIS v1's values only, found 16 failures among the 1,381 point values
checked, all of them GLYPHARIS's official OmniDocBench scores, because the evaluator's per-page time
limit depends on machine speed (GW-E2E-018, `reports/final_validation/v1_reproduction.md`).
Re-scoring the same saved Phase 14 files (GW-E2E-015,
`reports/final_validation/GW-E2E-015/verdict.json`) also changed Tesseract 5's official OmniDocBench
scores, through the same time-limit fallback, so the official GLYPHARIS-versus-Tesseract comparison
and ranking on these scores carry the same limitation. The owner accepted this as a disclosed
limitation of the official evaluator.

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests -->
<!-- trace: GW-E2E-009 reports/external/results.json#hypotheses -->
Three hypotheses were registered as tests before the frozen model read any test set. The first, H1′, required v1's
bag-of-lines CER on GDC's <!-- derived: clean_pages --> 53 clean pages (the `scan_app` and
`photo_flat` captures) to stay below twice that of PP-OCRv5 mobile, and its page time to be lower
than that system's. Both halves were met on PyTorch, so H1′ holds for that build. CER minus twice
the baseline's was <!-- derived: h1_cer --> −34.98 [<!-- derived: h1_cer_lo -->−47.44, <!-- derived: h1_cer_hi --> −24.08] points. Page time minus the baseline's
was <!-- derived: h1_lat --> −508.11 [−737.66, −256.52] ms on HierText validation pages, in one sitting,
with the baseline running PaddleOCR's two orientation models.

<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx -->
The ONNX build of v1 did not meet the same latency rule, tested in its own sitting: its page time
minus that of PP-OCRv5 mobile was <!-- derived: h1_onnx --> +6,537.6 [+5,383.7, +7,806.3] ms.

<!-- trace: GW-E2E-009 reports/external/results.json#hypotheses -->
The second, H5′, expected GDC crops to read at least 1.5 times worse
than HierText validation crops, and it was not met: GDC's CER minus that multiple of the validation
CER was <!-- derived: h5 --> −5.59 [<!-- derived: h5_lo -->−6.07, <!-- derived: h5_hi --> −5.01]
points, with intervals over lines. The third, H8′, decides whether H8 holds on GDC, and it does not:
temperature scaling lowered calibration error there, but the halving H8 predicted was not shown.

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#hiertext_test -->
<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages -->
Missed lines made up <!-- derived: missed_ht --> 49.45% [47.47, 51.56] of the page errors on
HierText test. In PyTorch on the laptop CPU at 10 threads, v1
took <!-- derived: cost_page --> 2,566 ms [2,327, 2,829] per HierText validation page, and the project's
laptop latency targets (ADR-0011) were missed. Phones are the target hardware. Phase 22 measured the
phone candidate's networks on two phones (GW-EFF-015): network time and memory only, without
decoding, post-processing, cropping or reading order; v1 itself was not measured on a phone.

Test results come from the GPU strict-fp32 evaluation; equality with the bundle is shown on
validation only. Final validation (Phase 22) has been run; its record is `docs/FINAL_VALIDATION.md`.
No weights are distributed; release is Phase 23.

## What the abstract rests on

- Accuracy and comparisons: F21, F22, F23, GW-E2E-007, GW-E2E-009, GW-REC-039, GW-BASE-001 to 007.
- Error analysis: F24, F27, F28, GW-E2E-008, GW-E2E-010.
- Efficiency and the owner decisions of 2026-11-18: F34, F35, F38, F40, ADR-0011. v1 keeps PyTorch
as its default backend.
- The phone candidate is the all-ONNX build with fused attention, reading one line per recognizer run
(`fused-single-line`, GW-EFF-013). A new version, not v1. Phase 22 evaluated it once per test set,
and it met H11's six non-inferiority tests against v1 (GW-E2E-021,
`reports/final_validation/candidate/h11.md`); its results sit beside v1's and replace none of them.
- Reproduction (GW-E2E-018, `reports/final_validation/v1_reproduction.md`): identical predictions;
the strict verdict failed only on official OmniDocBench scores, through the evaluator's time limit,
and the owner accepted this as a disclosed limitation.
- Registered tests: H1′ (holds for the PyTorch build, not met for the ONNX bundle), H5′ (not met),
H8′ (H8 not supported on GDC). Exploratory statuses: H4 (not supported on GDC, closed post hoc), H2′
(not supported on validation).

Not for the abstract: any statement that the model is the most accurate system, and the validation
CERs measured on other samples, which are not comparable with the test results
above: <!-- derived: val_line_034 --> 6.30% on GW-REC-034's validation
lines, <!-- derived: val_line_eff --> 6.22% on GW-EFF-009's seeded lines
and <!-- derived: val_page_400 --> 24.32% on the Phase 5 validation pages.
