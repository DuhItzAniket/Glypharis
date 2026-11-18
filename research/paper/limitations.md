# Limitations

HierText annotations are CC BY-SA 4.0 and its images CC BY 2.0 through Open Images; figures derived
from them are shared alike. Tracks are strict unless stated. GDC-scan appears as numbers only.

## Status of the evidence

Final validation (Phase 22) has not been run. Configuration frozen and recorded; reproduction
pending Phase 22. No weights are distributed; release is Phase 23.

<!-- trace: GW-E2E-011 reports/package/GW-E2E-011.json -->
Test results come from the GPU strict-fp32 evaluation; equality with the bundle is shown on
validation only. Phase 22 checks reproduction within the 0.05-point tolerance. The check on
validation covered 400 HierText pages through the bundle's PyTorch backend: all 16,835 lines kept
their text and box (GW-E2E-011). The same bundle's agreement with the PyTorch references is recorded
in GW-EFF-012, and the ONNX export's in GW-EFF-006.

## Size and make-up of the test sets

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#detection -->
<!-- trace: GW-REC-039 reports/external/recognition.md#track-strict -->
GDC has 106 pages from <!-- derived: gdc_documents --> 21 documents, FUNSD
<!-- derived: funsd_pages --> 50 forms and GDC-scan <!-- derived: gdc_scan_pages --> 28 pages from
one document. GDC-scan is a case study: its intervals are conditional on that document and it
carries no verdict (ADR-0013). HierText test has 1,634 pages and is mostly scene-style. OmniDocBench
has 755 pages. Its text blocks and FUNSD's fields include content that v1 does not try to read. GDC
and FUNSD leave wide page intervals. IIIT5K has 3,000 cropped words, which is not the line task v1
was built for.

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#gdc-per-capture -->
GDC's capture conditions are confounded with its documents, and some are small:
<!-- derived: gdc_rotated_pages --> 7 rotated and
<!-- derived: gdc_curved_pages --> 8 curved pages. Per-condition figures such as
64.60% [47.03, 82.96] bag-of-lines CER on curved pages are descriptive and are not contrasts
between conditions.

No old or degraded archival scans were evaluated. GDC's target for them is unmet, and GDC-scan is a
modern printed document (ADR-0013). SROIE was skipped, so receipts were not tested.

## What v1 does not handle

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#gdc-per-capture -->
There is no page-orientation step (F24). On GDC's <!-- derived: gdc_rotated_pages --> 7 rotated
pages the bag-of-lines CER of v1 was 80.53% [62.13, 92.43]. From ground-truth boxes the same
recognizer reached 0.83% [0.26, 2.19] (GW-E2E-008), and PP-OCRv6 small, which classifies page
orientation first, 1.70% [0.42, 5.79] (GW-BASE-006). Intervals are over pages.

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#vertical-lines -->
Vertical text is not read. Of GDC's <!-- derived: gdc_vertical_lines --> 32 vertical lines, the
detector matched 18.75% [5.26, 40.00] (interval over pages).

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#hiertext_test -->
The detector loses small and steep lines. On HierText test its recall was 57.92% [54.64, 61.17] for
lines under 9.35 px high and 49.53% [45.80, 53.19] for lines in the steepest rotation bin, at
<!-- derived: rotation_bin --> 15 degrees or more. Loose boxes cost the recognizer: CER on matched
lines was 29.24% [27.83, 30.64] in the lowest IoU bin and 3.63% [2.99, 4.49] in the highest.

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#gdc_test -->
On GDC the detector found none of the 113 lines under 9.35 px high.

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#reading-order -->
Reading order is a heuristic. On GDC 14.56% [9.93, 20.36] of consecutive matched lines came out in
the opposite order (page interval). Tables, sidebars and other complex layouts are not solved, and
OmniDocBench's overall score could not be computed because tables and formulas were not produced.

<!-- trace: reports/charset/charset_report.md reports/charset/charset_report.md -->
The character set is English-oriented, with 274 characters. Handwritten lines were excluded from
recognizer training, and other scripts are out of scope.

## Seeds

<!-- trace: GW-REC-037 reports/ablation/phase13_final.md -->
<!-- trace: GW-REC-034 reports/training/GW-REC-034/report.md -->
The v1 recognizer was trained with three seeds, GW-REC-034 to 036, with validation CER
<!-- derived: seed_cer_a --> 6.30, <!-- derived: seed_cer_b --> 6.28 and
<!-- derived: seed_cer_a --> 6.30% over 73,888 lines. The test sets were read by GW-REC-034
alone, so they do not show the spread between seeds. The v0.1 recognizer, GW-REC-024, stayed at one seed. Two more seeds were
registered for it as GW-REC-032 and GW-REC-033 and were not run, because option B was chosen in
GW-REC-037. The promise of extra v0.1 seeds is closed for v1 by the three v1 seeds and is not met
for v0.1.

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

<!-- trace: GW-BASE-002 reports/baselines/recognition.md -->
<!-- trace: GW-BASE-002 reports/baselines/end_to_end.md -->
<!-- trace: GW-REC-038 reports/external/recognition.md#track-strict -->
<!-- trace: GW-E2E-006 reports/external/end_to_end.md#track-strict -->
Validation results are not all on the same sample. The Phase 5 baselines cover 10,000 HierText
validation lines and 400 pages, while the broad GLYPHARIS tables cover 74,500 crops and 1,715 pages.
Baseline coverage of the test sets also differs: PP-OCRv5 server and PP-OCRv6 medium were not run
end to end on HierText test.

<!-- trace: GW-BASE-004 reports/external/recognition.md -->
TrOCR-small-printed was run on line crops only, so it has no end-to-end result. On GDC crops its CER
was 72.34% [71.65, 73.16] strict and 13.07% [12.33, 13.98] normalised (intervals over lines), so its
standing depends on the track.

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#track-strict -->
On OmniDocBench the PaddleOCR systems had order-aware CER from 60.29% [58.23, 62.26] to 60.62%
[58.68, 62.55]. A top-to-bottom box sort was proposed as the cause (F26), but the measure registered
for it could not decide the question (F33). The cause is unmeasured and none is claimed. The
order-aware CER on this set is the project's metric, not the official protocol.

H4 was closed post hoc: no decision rule was registered before the oracle was scored, so the
statement that it is not supported on GDC is a reading of the numbers and not a test.

## Efficiency

<!-- trace: GW-EFF-007 reports/efficiency/phase16/README.md#adr-0011s -->
ADR-0011's laptop targets were missed and are recorded as missed. The PyTorch build at 10 threads
took 33.9 ms per line against a target of 25 and 2,565.7 ms per page against 1.5 s. It stayed within
the page memory target, at 1,641.8 MiB against 1,836. The ONNX builds were over on all five targets.

<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx -->
The v1 ONNX bundle took +6,537.6 [+5,383.7, +7,806.3] ms per page against PP-OCRv5 mobile, so H1′ is
not met for it. H1′ holds for the PyTorch build only.

<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages -->
All latencies come from one laptop (Intel Core i7-14650HX, 10 threads) on HierText validation pages,
which average 40.9 lines, and not on document pages. Phones are the target hardware; no phone has
been measured.

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests -->
<!-- trace: GW-EFF-007 reports/benchmark_results/efficiency_comparison.md#pages -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx -->
Page latencies from different sittings are not compared. The mean page time of PP-OCRv5 mobile
(GW-BASE-002) was 3,408.2 ms in Phase 5, 2,954.8 ms in Phase 14 and, for GW-BASE-002 in Phase 16,
3,086.0 ms. PyTorch v1 took 2,446.7 ms in Phase 14 and 2,565.7 ms in Phase 16. Thermal drift within
a sitting is limited by the mirrored run order of the compared arms, not removed.

<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages -->
The fused-attention build took 4,020.8 ms [3,604.2, 4,479.8] per page on the laptop with 2,393.0 MiB
at peak. A new version, not v1; validation and laptop figures only; to be evaluated in Phase 22.

The compute ledger counts only queue-logged jobs. Experiments it does not cover are listed by ID in
`reports/experiment_summary.md`, so its totals are not the total compute of the project.

## Calibration

<!-- trace: GW-REC-039 reports/external/calibration.md -->
One temperature, T = 2.35, fitted on HierText validation, does not suit every kind of text. On
IIIT5K words the change in ECE was +0.0572 [+0.0431, +0.0716], so calibration became worse (interval
over words). On GDC's pipeline lines ECE stayed at 0.2489 after scaling. H8 is not supported on GDC:
the reduction transferred, but the halving was not shown, because the line and page bootstraps
disagreed. Line confidence should not be read as a calibrated probability on document pages.

## Synthetic data and robustness

<!-- trace: GW-DATA-022 reports/synthetic/phase12_comparison.md -->
v1 contains no GlyphSynth. H2′ was not supported at the tested budget: with the CRNN at a fixed step
budget, no amount of added lines lowered CER on the document slice of HierText validation, and +250K
lines raised it by +0.39 [+0.25, +0.52] points (paired over pages). The slice is a proxy, 104
validation pages with many lines and not a document set, and the study was not repeated on the
SVTR-style model.

<!-- trace: GW-DATA-009 reports/robustness/phase11_comparison.md -->
H3 was tested on the CRNN at the short budget only: degradation augmentation lowered GLYPH-C mCE by
−0.25 [−0.30, −0.19] points. It was not tested on the SVTR-style recognizer.

<!-- trace: GW-REC-037 reports/ablation/phase13_final.md -->
GLYPH-C is derived from validation lines, half of them rendered by GlyphSynth, so its mCE for v1
(3.54, 3.53 and 3.53% over the three seeds) and its clean CER are not comparable with HierText CER.
SynthText was not compared with GlyphSynth, because its licence is incompatible with the intended
use (ADR-0009).

## Engineering limits

The default pixel limit, `max_pixels`, of 64 million is a judgement and was not measured. A float
image file holding NaN values is read as a blank page without an error, although an array with NaN
values is refused (GW-E2E-012).

## Licence and release position

To be settled in Phase 23: the code licence, the project citation and any public package. No weights
are distributed; release is Phase 23. Intended terms under ADR-0014, effective only if released. The
owner decided on 2026-11-18 ([ADR-0014](../../docs/decisions/ADR-0014-weights-and-backbone-terms.md))
that GW-DET-001, initialised from ImageNet weights, stays v1's detector as a disclosed exception to
ADR-0009's commercial-use rule, because the ImageNet terms do not clearly allow commercial use, and
that the intended terms for the weights are CC BY-SA 4.0 with attribution to HierText and Open Images.
This is the project's chosen handling, not a legal conclusion, and the v1 weights are not described
as cleared for commercial use. FUNSD, IIIT5K and OmniDocBench were used for evaluation only, under
their own terms (`DATASETS.md`).
