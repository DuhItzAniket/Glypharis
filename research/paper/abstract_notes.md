# Abstract notes

Status: these results predate final validation. Final validation (Phase 22) has not been run. No
weights are distributed; release is Phase 23.

HierText annotations are CC BY-SA 4.0 and its images CC BY 2.0 through Open Images; figures derived
from them are shared alike.

## Abstract

GLYPHARIS is a two-stage OCR engine for English printed text in page images, built to be called by a
document-reading application. A DBNet-style detector with a MobileNetV3-Large backbone finds text
lines, and an SVTR-style encoder with CTC decoding reads each rectified line. Both models were
trained on one laptop GPU, on HierText only. The detector backbone started from ImageNet weights; no
OCR weights from another system were used.

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
The frozen model, v1, was evaluated once on each held-out set, strict track, with intervals over
pages unless stated. On HierText test (1,634 pages) bag-of-lines page CER was 25.16% [23.44, 26.79],
against 42.93% [40.48, 45.29] for PP-OCRv6 small, the lowest of the four baselines run end to end
there; the paired difference was −17.77 [−19.51, −16.01] points. On GDC, 106 phone captures of pages
from <!-- derived: gdc_documents --> 21 printed documents, bag-of-lines CER was 22.35% [15.87,
30.11] and no difference from PP-OCRv6 medium is stated (+1.39 [−2.71, +6.88] points). Order-aware
CER on GDC was 25.37% [20.00, 30.80] against 42.87% [36.18, 49.36], −17.50 [−25.52, −10.10] points.

<!-- trace: GW-REC-039 reports/external/recognition.md -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
<!-- trace: GW-E2E-007 reports/external/end_to_end.md -->
<!-- trace: GW-E2E-009 reports/external/official.md#omnidocbench -->
Several results were worse than a baseline. From ground-truth crops, GDC line CER was 4.17% [3.72,
4.67] against 3.20% [2.91, 3.56] for PP-OCRv6 medium, +0.97 [+0.76, +1.20] points (intervals over
lines). IIIT5K word accuracy, alphanumeric track, was 91.47% [90.43, 92.47] against 97.13% [96.57,
97.73], −5.67 [−6.53, −4.77] points (intervals over words). On OmniDocBench Tesseract 5 had the
lower order-aware CER, 15.70% [14.23, 17.37] against 27.28% [23.95, 30.57] (the project's metric,
not the official protocol), and the official text-block edit distance was 0.0966 for Tesseract
against 0.1007 for GLYPHARIS. FUNSD detection F1 at IoU 0.5 was 79.07% [75.56, 82.16], below the
point estimates of all five PaddleOCR systems (81.81% for PP-OCRv5 mobile to 85.03% for PP-OCRv5
server); no difference is stated.

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx -->
Of the registered tests, H1′ holds: on GDC's 53 clean-slice pages, CER minus twice the CER of
PP-OCRv5 mobile was −34.98 [−47.44, −24.08] points, and the PyTorch pipeline was −508.11 [−737.66,
−256.52] ms per page against that baseline on HierText validation pages. The ONNX build of v1 did
not meet the same latency rule, +6,537.6 [+5,383.7, +7,806.3] ms per page. H5′, which expected GDC
crops to read at least 1.5 times worse than HierText validation crops, was not met: −5.59 [−6.07,
−5.01] points over lines. H8 was not supported on GDC: temperature scaling lowered calibration error
there, but the halving it predicted was not shown.

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#hiertext_test -->
<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages -->
Missed lines were 49.45% [47.47, 51.56] of the page errors on HierText test. In PyTorch on the
laptop CPU, v1 took 2,566 ms [2,327, 2,829] per HierText validation page at 10 threads; the ADR-0011
laptop targets were missed. Phones are the target hardware; no phone has been measured.

## What the abstract rests on

- Accuracy and comparisons: F21, F22, F23, GW-E2E-007, GW-E2E-009, GW-REC-039, GW-BASE-001 to 007.
- Error analysis: F24, F27, F28, GW-E2E-008, GW-E2E-010.
- Efficiency and the owner decisions of 2026-11-18: F34, F35, F38, F40, ADR-0011. v1 keeps PyTorch
as its default backend.
- The all-ONNX fused-attention build is the phone candidate. A new version, not v1; validation and
laptop figures only; to be evaluated in Phase 22.
- Hypotheses: H1′ (holds), H5′ (not met), H8 (not supported on GDC), H4 (not supported, closed post
hoc), H2′ (not supported on validation).

Not for the abstract: any statement that the model is the most accurate system, and the validation
figures 6.30, 6.22 and 24.32, which belong to other line and page samples.
