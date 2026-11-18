# Conclusion notes

Final validation (Phase 22) has not been run, so everything below is a statement about the evidence
recorded so far. No weights are distributed; release is Phase 23.

HierText annotations are CC BY-SA 4.0 and its images CC BY 2.0 through Open Images; figures derived
from them are shared alike. Tracks are strict unless stated.

## What was built

GLYPHARIS v1 is a two-stage OCR engine for English printed text in page images: a DBNet-style line
detector, an SVTR-style CTC recognizer and a heuristic reading order. It was trained on one laptop
GPU on HierText only, frozen from validation data, and evaluated once on each held-out set.
Configuration frozen and recorded; reproduction pending Phase 22.

## What the evidence shows

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#track-strict -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
On HierText test the pipeline's bag-of-lines page CER was 25.16% [23.44, 26.79] (interval over
pages). That is −17.77 [−19.51, −16.01] points against PP-OCRv6 small, the lowest of the four
baselines run end to end on that set. On GDC's phone captures its bag-of-lines CER was 22.35%
[15.87, 30.11], and no difference from PP-OCRv6 small (+0.06 [−4.02, +5.65]) or PP-OCRv6 medium
(+1.39 [−2.71, +6.88]) is stated. Its order-aware CER on GDC was 25.37% [20.00, 30.80], which was
−17.50 [−25.52, −10.10] points against PP-OCRv6 medium. The cause of that order gap is not measured.

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests -->
H1′ holds for the PyTorch build: −34.98 [−47.44, −24.08] points on the CER half and −508.11
[−737.66, −256.52] ms per page on the latency half. H5′, which expected GDC crops to read at least
1.5 times worse than HierText validation crops, was not met: −5.59 [−6.07, −5.01] points over lines.
H8 was not supported on GDC.

## Where it fell short

<!-- trace: GW-REC-039 reports/external/recognition.md -->
<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
<!-- trace: GW-E2E-007 reports/external/end_to_end.md#track-strict -->
GLYPHARIS read GDC crops worse than PP-OCRv6 medium, 4.17% [3.72, 4.67] against 3.20% [2.91, 3.56]
(+0.97 [+0.76, +1.20] points, intervals over lines). Its IIIT5K word accuracy was 91.47% [90.43,
92.47] against 97.13% [96.57, 97.73] (alphanumeric track, intervals over words). On OmniDocBench
Tesseract 5 had the lower order-aware CER, 15.70% [14.23, 17.37] against 27.28% [23.95, 30.57].
FUNSD detection F1 was below the point estimate of every PaddleOCR system run on it; no difference
is stated.

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#hiertext_test -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/oracle_share.md -->
Most page error does not come from misreading well-found lines. Missed lines were 49.45% [47.47,
51.56] of HierText test's page errors. On GDC the recognizer working from ground-truth boxes made
18.66% [11.55, 28.73] as many errors as the whole pipeline (both intervals over pages). The error
analysis points first to the detector and to page handling, and v1 has no page-orientation step.

<!-- trace: GW-EFF-007 reports/efficiency/phase16/README.md#adr-0011s -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx -->
The ADR-0011 laptop targets were missed. PyTorch v1 took 2,565.7 ms per HierText validation page on
the CPU against a target of 1.5 s, and the ONNX bundle was slower than PP-OCRv5 mobile by +6,537.6
[+5,383.7, +7,806.3] ms per page.

Document-style synthetic data did not help the CRNN at a fixed step budget (H2′), and v1 contains
none. These results are in `limitations.md` with their conditions.

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
