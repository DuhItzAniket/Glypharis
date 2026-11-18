# Conclusion notes

Final validation (Phase 22) has not been run, so everything below is a statement about the evidence
recorded so far. No weights are distributed; release is Phase 23.

HierText is released under CC BY-SA 4.0; its images come from Open Images, which lists them as
CC BY 2.0. Figures derived from HierText are shared alike. Tracks are strict unless stated.

## What was built

GLYPHARIS v1 is a two-stage OCR engine for English printed text in page images: a DBNet-style line
detector, an SVTR-style CTC recognizer and a heuristic reading order. It was trained on one laptop
GPU on HierText only, frozen from validation data, and evaluated once on each held-out set.
Configuration frozen and recorded; reproduction pending Phase 22.

## What the evidence shows

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
On HierText test the pipeline's bag-of-lines page CER was <!-- derived: ht_glyph --> 25.16%
[23.44, 26.79] (interval over pages). That is <!-- derived: d_ht_small --> −17.77 [−19.51, −16.01]
points against PP-OCRv6 small, the lowest of the four baselines run end to end on that set. On GDC's
phone captures its bag-of-lines CER was <!-- derived: gdc_glyph --> 22.35% [15.87, 30.11], and no
difference is stated from PP-OCRv6 small (<!-- derived: d_gdc_small --> +0.06 [−4.02, +5.65]) or
PP-OCRv6 medium (<!-- derived: d_gdc_medium --> +1.39 [−2.71, +6.88]).

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
Its order-aware CER on GDC was <!-- derived: gdc_glyph_oa --> 25.37% [20.00, 30.80]. That
is <!-- derived: d_gdc_medium_oa --> −17.50 [−25.52, −10.10] points against PP-OCRv6 medium. No
difference is stated against the lowest baseline on this measure, Tesseract 5
(<!-- derived: d_gdc_tess_oa --> −6.53 [−13.51, +0.08]). The cause of the gap to the PaddleOCR
systems is not measured.

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests -->
<!-- trace: GW-E2E-009 reports/external/results.json#hypotheses -->
H1′ holds for the PyTorch build. On the CER half, measured on
GDC's <!-- derived: clean_pages --> 53 clean-slice pages, CER minus twice that of PP-OCRv5 mobile
was <!-- derived: h1_cer --> −34.98 [<!-- derived: h1_cer_lo -->−47.44, <!-- derived: h1_cer_hi --> −24.08] points. On the latency half, measured on HierText validation
pages in one sitting, with the baseline running its two orientation models, page time minus the
baseline's was <!-- derived: h1_lat --> −508.11 [−737.66, −256.52] ms. H5′, which expected GDC crops
to read at least 1.5 times worse than HierText validation crops, was not
met: <!-- derived: h5 --> −5.59 [<!-- derived: h5_lo -->−6.07, <!-- derived: h5_hi --> −5.01] points
over lines. H8 was not supported on GDC.

## Where it fell short

<!-- trace: GW-E2E-009 reports/external/recognition.md -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
GLYPHARIS read GDC crops worse than PP-OCRv6 medium, <!-- derived: crop_glyph --> 4.17%
[3.72, 4.67] against <!-- derived: crop_medium --> 3.20% [2.91, 3.56]
(<!-- derived: d_crop_medium --> +0.97 [+0.76, +1.20] points, intervals over lines). Its IIIT5K word
accuracy was <!-- derived: iiit_glyph --> 91.47% [90.43, 92.47]
against <!-- derived: iiit_medium --> 97.13% [96.57, 97.73] for the same baseline,
or <!-- derived: d_iiit_medium --> −5.67 [−6.53, −4.77] points (alphanumeric track, intervals over
words). On OmniDocBench its order-aware CER was <!-- derived: odb_glyph --> 27.28% [23.95, 30.57]
against <!-- derived: odb_tess --> 15.70% [14.23, 17.37] for Tesseract 5
(<!-- derived: d_odb_tess --> +11.58 [+8.37, +14.88] points). FUNSD detection F1 was below the point
estimate of every PaddleOCR system run on it; no difference is stated.

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#hiertext_test -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/oracle_share.md#gdc_test -->
Most page error does not come from misreading well-found lines. On HierText test, missed lines
were <!-- derived: missed_ht --> 49.45% [47.47, 51.56] of the page errors and errors on matched
lines <!-- derived: matched_ht --> 25.79% [23.51, 28.00]. On GDC the recognizer working from ground-truth
boxes made <!-- derived: oracle_all --> 18.66% [11.55, 28.73] as many errors as the whole pipeline
(both sets, intervals over pages). The remainder also holds spurious predictions and crop shape, so
it is not a pure detector share. These shares point to line finding and page handling, and v1 has no
page-orientation step.

<!-- trace: GW-EFF-007 reports/efficiency/phase16/README.md#adr-0011s -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx -->
The ADR-0011 laptop targets were missed. PyTorch v1 took <!-- derived: adr_page --> 2,565.7 ms per
HierText validation page on the CPU against a target of 1.5 s. The ONNX bundle, in its own sitting,
was slower than PP-OCRv5 mobile by <!-- derived: h1_onnx --> +6,537.6 [+5,383.7, +7,806.3] ms per page.

At a fixed step budget, adding GlyphSynth lines did not lower the CRNN's CER on the document slice of
HierText validation, a proxy and not a document set (H2′), and v1 contains none. These results are
in `limitations.md` with their conditions.

## What remains open

- Final validation (Phase 22) has not been run. Test results come from the GPU strict-fp32
evaluation; equality with the bundle is shown on validation only. Phase 22 checks reproduction
within the 0.05-point tolerance.
- Phones are the target hardware; no phone has been measured.
- The all-ONNX fused-attention build is the phone candidate. A new version, not v1; validation and
laptop figures only; to be evaluated in Phase 22.
- Page orientation, small text and steep lines are the failures the error analysis points to first.
- The cause of the PaddleOCR reading-order error on OmniDocBench remains unmeasured (F26, F33).
- Weights: intended terms under ADR-0014, effective only if released: CC BY-SA 4.0 with attribution to
HierText and Open Images, with GW-DET-001's ImageNet initialisation kept as a disclosed exception to
ADR-0009's commercial-use rule ([ADR-0014](../../docs/decisions/ADR-0014-weights-and-backbone-terms.md),
owner decision of 2026-11-18; not a legal conclusion). Code licence and citation: to be settled in
Phase 23.
