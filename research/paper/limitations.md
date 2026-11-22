# Limitations

HierText is released under CC BY-SA 4.0; its images come from Open Images, which lists them as
CC BY 2.0. Figures derived from HierText are shared alike. Tracks are strict unless stated. GDC-scan
appears as numbers only.

## Status of the evidence

Final validation (Phase 22) has been run; its record is `docs/FINAL_VALIDATION.md`. Configuration
frozen and recorded. Phase 22 re-ran it: the predictions equal Phase 14's in every test file, and the
strict verdict failed only on official OmniDocBench scores, through the evaluator's time limit; the
owner accepted this as a disclosed limitation (`reports/final_validation/v1_reproduction.md`). v1.0
is the release of v1 (GW-REC-034 with GW-DET-001): the code under Apache-2.0, published on PyPI as
`glypharis` 1.0.0, and the v1 weights as the asset `glypharis-v1.tar.gz` of the GitHub release
`v1.0`, which `glypharis.download_model` fetches and checks against pinned hashes.

<!-- trace: GW-E2E-018 reports/final_validation/v1_reproduction.md -->
- Reproduction of the official OmniDocBench scores. The strict reproduction verdict (GW-E2E-018,
  `reports/final_validation/v1_reproduction.md`), which judged GLYPHARIS v1's values only, found 16
  failures among the 1,381 point values checked, all of them GLYPHARIS's official OmniDocBench
  scores. The predictions were identical to Phase 14's; the official evaluator gives up on a page
  after a time limit and scores it by a fallback, and how many pages hit that limit depends on
  machine speed. Re-scoring the same saved Phase 14 files (GW-E2E-015,
  `reports/final_validation/GW-E2E-015/verdict.json`) also changed Tesseract 5's official
  OmniDocBench scores, through the same time-limit fallback, so the official GLYPHARIS-versus-Tesseract
  comparison and ranking on these scores carry the same limitation. The owner accepted this as a
  disclosed limitation; the strict verdict stays as recorded, and so do Phase 14's results. Any
  official OmniDocBench score in this package carries this limitation.
- Phone figures are network-only. The phone measurement (GW-EFF-015) covers the phone candidate's
  network time and memory; decoding, post-processing, cropping and reading order were not ported, so
  a full page on a phone was not measured, and v1 itself was not measured on a phone.

<!-- trace: GW-E2E-011 reports/package/GW-E2E-011.json -->
Test results come from the GPU strict-fp32 evaluation; equality with the bundle is shown on
validation only. Phase 22 checked reproduction at the 0.05-point tolerance (see above). The check on
validation covered <!-- derived: pkg_pages --> 400 HierText pages through the bundle's PyTorch
backend: all <!-- derived: pkg_lines --> 16,835 lines kept their text and box (GW-E2E-011). The same
bundle's agreement with the PyTorch references is recorded in GW-EFF-012, and the ONNX export's in
GW-EFF-006.

## Size and make-up of the test sets

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#detection -->
<!-- trace: GW-E2E-009 reports/external/recognition.md#iiit5k -->
GDC has <!-- derived: gdc_pages --> 106 pages from <!-- derived: gdc_documents --> 21 documents,
FUNSD <!-- derived: funsd_pages --> 50 forms and GDC-scan <!-- derived: gdc_scan_pages --> 28 pages from
one document. GDC-scan is a case study: its intervals are conditional on that document and it
carries no verdict (ADR-0013). HierText test has <!-- derived: ht_pages --> 1,634 pages and is mostly
scene-style. OmniDocBench has <!-- derived: odb_pages --> 755 pages. Its text blocks and FUNSD's
fields include content that v1 does not try to read. GDC and FUNSD leave wide page intervals. IIIT5K
has <!-- derived: iiit_words --> 3,000 cropped words, which is not the line task v1 was built for.

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#gdc-per-capture -->
GDC's capture conditions are confounded with its documents, and some are small:
only <!-- derived: gdc_rotated_pages --> 7 pages are rotated and <!-- derived: gdc_curved_pages --> 8
curved. Per-condition figures such as the bag-of-lines CER
of <!-- derived: gdc_curved_cer --> 64.60% [47.03, 82.96] on curved pages are descriptive and are not
contrasts between conditions.

No old or degraded archival scans were evaluated. GDC's target for them is unmet, and GDC-scan is a
modern printed document (ADR-0013). SROIE was skipped, so receipts were not tested.

## What v1 does not handle

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#gdc-per-capture -->
There is no page-orientation step (F24). On GDC's <!-- derived: gdc_rotated_pages --> 7 rotated
pages the bag-of-lines CER of v1 was <!-- derived: rot_glyph --> 80.53% [62.13, 92.43]. From
ground-truth boxes the same recognizer reached <!-- derived: rot_oracle --> 0.83% [0.26, 2.19]
(GW-E2E-008), and PP-OCRv6 small, which classifies page orientation
first, <!-- derived: rot_small --> 1.70% [0.42, 5.79] (GW-BASE-006). Intervals are over pages.

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#vertical-lines -->
Vertical text is not read. Of GDC's <!-- derived: gdc_vertical_lines --> 32 vertical lines, the
detector matched <!-- derived: vert_recall --> 18.75% [5.26, 40.00] (interval over pages).

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#hiertext_test -->
The detector loses small and steep lines. On HierText test its recall
was <!-- derived: recall_small --> 57.92% [54.64, 61.17] for lines
under <!-- derived: small_height --> 9.35 px high and <!-- derived: recall_steep --> 49.53% [45.80, 53.19]
for lines in the steepest rotation bin, at <!-- derived: rotation_bin --> 15 degrees or more. CER on
matched lines was <!-- derived: iou_low --> 29.24% [27.83, 30.64] in the lowest IoU bin
and <!-- derived: iou_high --> 3.63% [2.99, 4.49] in the highest. These slices are marginal and
confounded, so they do not isolate the effect of the box.

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#gdc_test -->
On GDC the detector found none of the <!-- derived: gdc_small_lines --> 113 lines
under <!-- derived: small_height_gdc --> 9.35 px high.

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#reading-order -->
Reading order is a heuristic. On GDC 14.56% [9.93, 20.36] of consecutive matched lines came out in
the opposite order (page interval). Tables, sidebars
and other complex layouts are not solved, and OmniDocBench's overall score could not be computed
because tables and formulas were not produced.

<!-- trace: reports/charset/charset_report.md reports/charset/charset_report.md -->
The character set is English-oriented. The charset file has 274 characters. Handwritten lines were
excluded from recognizer training, and other scripts are out of scope.

## Seeds

<!-- trace: GW-REC-037 reports/ablation/phase13_final.md -->
<!-- trace: GW-REC-034 reports/training/GW-REC-034/report.md -->
The v1 recognizer was trained with three seeds, GW-REC-034 to 036, with validation CER
of <!-- derived: seed_cer_a --> 6.30, <!-- derived: seed_cer_b --> 6.28
and <!-- derived: seed_cer_a --> 6.30%, over <!-- derived: rec034_lines --> 73,888 lines. The test sets
were read by GW-REC-034 alone, so they do not show the spread between seeds. The v0.1 recognizer,
GW-REC-024, stayed at one seed. Two more seeds were registered for it as GW-REC-032 and GW-REC-033
and were not run, because option B was chosen in GW-REC-037. The promise of extra v0.1 seeds is
closed for v1 by the three v1 seeds and is not met for v0.1.

The detector GW-DET-001 has one seed, and so does each detector compared with it (GW-DET-002, 003,
007). The detector variations therefore show no seed spread.

<!-- trace: GW-REC-022 reports/training/compare/GW-REC-022_vs_GW-REC-006.md -->
GW-REC-022's gain over GW-REC-006, −0.27 [−0.32, −0.22] points of validation CER over lines, also
changes weight decay, learning rate and micro-batching, so it cannot be credited to the encoder
alone.

## Comparisons with other systems

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests -->
Only H1′, H5′ and H8′ (and the Phase 16 tests) are confirmatory; every other interval is
descriptive. PP-OCRv5 mobile ran PaddleOCR's two orientation models in the H1′ latency half, as it
did for its accuracy numbers.

<!-- trace: GW-E2E-009 reports/external/recognition.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
Several comparisons went against GLYPHARIS. On GDC's line crops, read from ground-truth boxes, its
CER was <!-- derived: crop_glyph --> 4.17% [3.72, 4.67] against <!-- derived: crop_medium --> 3.20%
[2.91, 3.56] for PP-OCRv6 medium, a difference of <!-- derived: d_crop_medium --> +0.97
[+0.76, +1.20] points. Its CER was also higher than that of PP-OCRv6 small,
by <!-- derived: d_crop_small --> +0.71 [+0.53, +0.91], and of PP-OCRv5 mobile,
by <!-- derived: d_crop_mobile --> +0.49 [+0.23, +0.78] (intervals over lines).

<!-- trace: GW-E2E-009 reports/external/recognition.md#iiit5k -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
On IIIT5K's alphanumeric track its word accuracy was <!-- derived: iiit_glyph --> 91.47%
[90.43, 92.47] against <!-- derived: iiit_medium --> 97.13% [96.57, 97.73] for PP-OCRv6 medium, a
difference of <!-- derived: d_iiit_medium --> −5.67 [−6.53, −4.77] points (intervals over words).

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
<!-- trace: GW-E2E-009 reports/external/official.md#omnidocbench -->
On OmniDocBench its order-aware CER was <!-- derived: odb_glyph --> 27.28% [23.95, 30.57]
against <!-- derived: odb_tess --> 15.70% [14.23, 17.37] for Tesseract 5 (a difference
of <!-- derived: d_odb_tess --> +11.58 [+8.37, +14.88] points, intervals over pages). This is the
project's metric, not the official protocol. The official text-block edit distance, where lower is
better and no interval exists, was <!-- derived: off_glyph_whole --> 0.1859 for GLYPHARIS
against <!-- derived: off_tess_whole --> 0.0847 for Tesseract 5 over the whole set,
and <!-- derived: off_glyph_page --> 0.1007 against <!-- derived: off_tess_page --> 0.0966 as a page
average; no difference is stated. In Phase 22 these official scores of GLYPHARIS did not pass the
strict reproduction verdict, which judged GLYPHARIS v1's values only, although the predictions were
identical, because the evaluator's per-page time limit depends on machine speed (GW-E2E-018,
`reports/final_validation/v1_reproduction.md`). Re-scoring the same saved Phase 14 files (GW-E2E-015,
`reports/final_validation/GW-E2E-015/verdict.json`) also changed Tesseract 5's official scores,
through the same time-limit fallback, so this comparison and its ranking carry the same limitation.
The owner accepted this as a disclosed limitation (see the status section above).

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#detection -->
FUNSD detection F1 at IoU 0.5 was <!-- derived: funsd_glyph --> 79.07%
[75.56, 82.16], below the point estimates of all five PaddleOCR systems,
from <!-- derived: funsd_mobile --> 81.81% for PP-OCRv5 mobile to <!-- derived: funsd_server --> 85.03%
for PP-OCRv5 server. There is no paired interval for it, so no difference is stated.

<!-- trace: GW-BASE-002 reports/baselines/recognition.md -->
<!-- trace: GW-BASE-002 reports/baselines/end_to_end.md -->
<!-- trace: GW-REC-038 reports/external/recognition.md#track-strict -->
<!-- trace: GW-E2E-006 reports/external/end_to_end.md#track-strict -->
Validation results are not all on the same sample. The Phase 5 baselines
cover <!-- derived: base_val_lines --> 10,000 HierText validation lines
and <!-- derived: base_val_pages --> 400 pages, while the broad GLYPHARIS tables
cover <!-- derived: val_crops --> 74,500 crops and <!-- derived: val_pages --> 1,715 pages. Baseline
coverage of the test sets also differs: PP-OCRv5 server and PP-OCRv6 medium were not run end to end
on HierText test.

<!-- trace: GW-BASE-004 reports/external/recognition.md -->
TrOCR-small-printed was run on line crops only, so it has no end-to-end result. On GDC crops its CER
was <!-- derived: trocr_strict --> 72.34% [71.65, 73.16] strict
and <!-- derived: trocr_norm --> 13.07% [12.33, 13.98] normalised (intervals over lines), so its
standing depends on the track.

<!-- trace: GW-E2E-009 reports/external/end_to_end.md#track-strict -->
On OmniDocBench the PaddleOCR systems had order-aware CER
from <!-- derived: odb_small --> 60.29% [58.23, 62.26] to <!-- derived: odb_tiny --> 60.62%
[58.68, 62.55]. A top-to-bottom box sort was proposed as the cause (F26), but the measure registered
for it could not decide the question (F33). The cause is unmeasured and none is claimed.

H4 was closed post hoc: no decision rule was registered before the oracle was scored, so the
statement that it is not supported on GDC is a reading of the numbers and not a test.

## Efficiency

<!-- trace: GW-EFF-007 reports/efficiency/phase16/README.md#adr-0011s -->
ADR-0011's laptop targets were missed and are recorded as missed. The PyTorch build at
10 threads took <!-- derived: adr_line --> 33.9 ms per line against a target
of 25 and <!-- derived: adr_page --> 2,565.7 ms per page against 1.5 s. It stayed within the page
memory target, at <!-- derived: adr_ram --> 1,641.8 MiB against 1,836. The ONNX builds were over on
all five targets.

<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx -->
The v1 ONNX bundle took <!-- derived: h1_onnx --> +6,537.6 [+5,383.7, +7,806.3] ms per page against PP-OCRv5 mobile, so H1′ is
not met for it. H1′ holds for the PyTorch build only.

<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages -->
All latencies come from one laptop (Intel Core i7-14650HX, 10 threads) on
HierText validation pages, which average <!-- derived: lines_per_page --> 40.9 lines, and not on
document pages. Phones are the target hardware. Phase 22 measured the phone candidate's networks on
two phones (GW-EFF-015): network time and memory only, without decoding, post-processing, cropping
or reading order; v1 itself was not measured on a phone.

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests -->
<!-- trace: GW-EFF-007 reports/benchmark_results/efficiency_comparison.md#pages -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx -->
Page latencies from different sittings are not compared. The mean page time of PP-OCRv5 mobile
(GW-BASE-002) was <!-- derived: mobile_p5 --> 3,408.2 ms in Phase 5, <!-- derived: mobile_p14 -->
2,954.8 ms in Phase 14 and, in Phase 16, <!-- derived: mobile_p16 --> 3,086.0 ms. PyTorch v1
took <!-- derived: torch_p14 --> 2,446.7 ms in Phase 14 and <!-- derived: torch_p16 --> 2,565.7 ms in
Phase 16. Thermal drift within a sitting is limited by the mirrored run order of the compared arms,
not removed.

<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages -->
The fused-attention build took <!-- derived: fused_ms --> 4,020.8 ms [3,604.2, 4,479.8] per page on
the laptop with <!-- derived: fused_ram --> 2,393.0 MiB at peak. The phone candidate chosen in
Phase 22 is the all-ONNX build with fused attention, reading one line per recognizer run
(`fused-single-line`, GW-EFF-013). A new version, not v1. Phase 22 evaluated it once per test set,
and it met H11's six non-inferiority tests against v1 (GW-E2E-021); its results sit beside v1's and
replace none of them. H11 is a non-inferiority test and does not show line-for-line equality on test
data. Its official OmniDocBench scores are not comparable with v1's, because the evaluator's time
limit sent a different number of pages to fallback scoring
(`reports/final_validation/candidate/results.json`, `match_debug`).

The compute ledger counts only queue-logged jobs. Experiments it does not cover are listed by ID in
`reports/experiment_summary.md`, so its totals are not the total compute of the project.

## Calibration

<!-- trace: GW-REC-039 reports/external/calibration.md -->
One temperature, T = 2.35, fitted on HierText validation, does not suit
every kind of text. On IIIT5K words the change in ECE was <!-- derived: ece_iiit --> +0.0572
[+0.0431, +0.0716], so calibration became worse (interval over words). On GDC's pipeline lines ECE
was still <!-- derived: ece_pipe --> 0.2489 after scaling. H8 is not supported on GDC: the reduction
transferred, but the halving was not shown, because the line and page bootstraps disagreed. Line
confidence should not be read as a calibrated probability on document pages.

## Synthetic data and robustness

<!-- trace: GW-DATA-022 reports/synthetic/phase12_comparison.md -->
v1 contains no GlyphSynth. H2′ was not supported at the tested budget: with the CRNN at a fixed step
budget, no amount of added lines lowered CER on the document slice of HierText validation, and
+250K lines raised it by <!-- derived: synth_250 --> +0.39
[+0.25, +0.52] points (paired over pages). The slice is a proxy, <!-- derived: doc_slice_pages -->
104 validation pages with many lines and not a document set, and the study was not repeated on the
SVTR-style model.

<!-- trace: GW-DATA-009 reports/robustness/phase11_comparison.md -->
H3 was tested on the CRNN at the short budget only: degradation augmentation lowered GLYPH-C mCE
by <!-- derived: h3_delta --> −0.25 [−0.30, −0.19] points. It was not tested on the SVTR-style
recognizer.

<!-- trace: GW-REC-037 reports/ablation/phase13_final.md -->
GLYPH-C is derived from validation lines, half of them rendered by GlyphSynth, so its mCE for v1
(<!-- derived: mce_a --> 3.54, <!-- derived: mce_b --> 3.53 and <!-- derived: mce_b --> 3.53% over
the three seeds) and its clean CER are not comparable with HierText CER. SynthText was not compared
with GlyphSynth, because its licence is incompatible with the intended use (ADR-0009).

## Engineering limits

The default pixel limit, `max_pixels`, of 64 million is a judgement and was not measured. A float
image file holding NaN values is read as a blank page without an error, although an array with NaN
values is refused (GW-E2E-012).

## Licence and release position

v1.0 is the release of v1 (GW-REC-034 with GW-DET-001): the code under Apache-2.0, published on PyPI
as `glypharis` 1.0.0, and the v1 weights as the asset `glypharis-v1.tar.gz` of the GitHub release
`v1.0`, which `glypharis.download_model` fetches and checks against pinned hashes. The Apache
License covers the code and the project's own text (`LICENSE`, `NOTICE`), with the exceptions that
`NOTICE` lists, and not the weights.

The weights are distributed under CC BY-SA 4.0 with attribution to HierText and Open Images
([ADR-0014](../../docs/decisions/ADR-0014-weights-and-backbone-terms.md)). The detector was
initialised from ImageNet weights, a disclosed exception to the project's commercial-use rule
(ADR-0009), because the ImageNet terms do not clearly allow commercial use, and the weights are not
described as cleared for commercial use. The owner decided on 2026-11-22 to publish them under these
terms without the qualified advice ADR-0014 had asked for; this is the project's chosen handling,
not a legal conclusion. The phone candidate is not released. Cite with `CITATION.cff` (DOI
10.5281/zenodo.24433864). FUNSD, IIIT5K and OmniDocBench were used for evaluation only, under
their own terms (`DATASETS.md`).
