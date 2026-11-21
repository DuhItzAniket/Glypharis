# Introduction notes

Status: Final validation (Phase 22) has been run; its record is `docs/FINAL_VALIDATION.md`. No
weights are distributed; release is Phase 23.

HierText is released under CC BY-SA 4.0; its images come from Open Images, which lists them as
CC BY 2.0. Figures derived from HierText are shared alike.

## Motivation

Many PDF files hold page images and no text layer: scans, photographed pages and image exports. A
document-reading application that searches, splits or quotes such a file needs OCR first, and it
needs to know what the OCR output can be trusted for. Large OCR systems and vision-language models
can read such pages, but they are heavy to run. Compact two-stage toolkits also exist
(`du2020ppocr`, `cui2025paddleocr`, `paddleocr_ppocrv6`). This project built a small engine and
measured it beside those toolkits, on sets it had not seen, with the same metrics for every system.

## Scope

GLYPHARIS takes a page image and returns lines of English printed text with boxes and confidence. A
DBNet-style detector (`liao2020dbnet`) with a MobileNetV3-Large backbone (`howard2019mobilenetv3`)
finds lines. Each line is cropped and rectified, read by an SVTR-style encoder (`du2022svtr`) with
CTC decoding (`graves2006ctc`), and placed in reading order by a recursive XY-cut
(`nagy1992xycut`). The models were trained on one laptop with an RTX 4050 GPU, on HierText
(`long2022hiertext`) only. The detector backbone started from ImageNet weights
(`russakovsky2015imagenet`). No OCR weights from another system were used (ADR-0007).

Handwriting, other scripts, tables, formulas and layout analysis are outside the scope.

## Decisions that shape the claims

The owner made these decisions on 2026-11-18, after the efficiency measurements (ADR-0011); the
last two items also say what Phase 22 then found:

- v1 keeps PyTorch as its default backend, as it was evaluated in Phase 14.
- The laptop targets of ADR-0011 are recorded as missed for v1. They are not restated.
- Phones are the target hardware. Phase 22 measured the phone candidate's networks on two phones
  (GW-EFF-015): network time and memory only, without decoding, post-processing, cropping or reading
  order; v1 itself was not measured on a phone.
- The phone candidate is the all-ONNX build with fused attention, reading one line per recognizer
  run (`fused-single-line`, GW-EFF-013). A new version, not v1. Phase 22 evaluated it once per test
  set, and it met H11's six non-inferiority tests against v1 (GW-E2E-021); its results sit beside
  v1's and replace none of them.

## Research questions

The questions are those of `docs/IMPLEMENTATION_PLAN.md`. The answers and their evidence are in
`results.md`.

| Question | Topic | Main evidence |
|---|---|---|
| RQ1 | How effective a lightweight detector and CTC line recognizer are for document OCR | GW-E2E-007, GW-E2E-009 |
| RQ2 | How much document-style synthetic data helps recognition, how that scales, and which generator factors matter | GW-DATA-022, H2′ |
| RQ3 | Which degradation augmentations improve robustness | GW-DATA-009, H3 |
| RQ4 | How the engine compares with Tesseract, PaddleOCR and TrOCR-small | GW-E2E-009, GW-EFF-005, GW-BASE-001 to 007 |
| RQ5 | Which failures dominate, and whether they come from the detector or the recognizer | GW-E2E-010, GW-REC-040, H4 |
| RQ6 | How document performance differs from scene-text performance | GW-REC-038, GW-REC-039, H5′ |
| RQ7 | The accuracy, latency and memory trade-off for local use | GW-EFF-007 to 011 |
| RQ8 | Whether raw CTC confidence is calibrated and whether temperature scaling repairs it | GW-E2E-005, GW-E2E-009, H8 |

RQ2 was first framed as a comparison of scene-text and document-style synthetic data. SynthText was
dropped for licence reasons before any run (ADR-0009), so that comparison was not made and this work
does not address it.

## What this work reports

1. A two-stage engine trained on one laptop GPU, frozen as v1 from validation data alone and then
   evaluated once per held-out set (GW-REC-037, GW-E2E-007, GW-REC-039).
2. A recognizer ablation ladder over three seeds per row (GW-REC-037). One of its rows, the
   SVTR-style model with its recipe, changes the encoder together with weight decay, learning rate
   and micro-batching, so the ladder does not isolate the encoder.
3. A negative result on document-style synthetic data: at a fixed step budget, with the CRNN, no
   amount of GlyphSynth tried lowered CER on the document slice of HierText validation (GW-DATA-022,
   H2′). v1 contains no GlyphSynth.
4. An augmentation study on a corruption suite derived from validation lines, GLYPH-C (GW-DATA-009,
   H3, F14).
5. External comparisons with Tesseract, PP-OCRv5 and PP-OCRv6 on detection, recognition and whole
   pages, and with TrOCR-small-printed on line crops only, since it has no detector here
   (GW-BASE-001 to 007, GW-E2E-009). They include the official HierText and OmniDocBench evaluators.
   Several comparisons went against GLYPHARIS (`limitations.md`, `results.md`).
6. An error analysis that splits page errors into missed lines, errors on matched lines and spurious
   predictions, and compares the pipeline with the recognizer on ground-truth boxes (GW-E2E-008,
   GW-E2E-010, GW-REC-040).
7. Efficiency measurements on one laptop, with the missed targets recorded (GW-EFF-005 to 011), and
   a calibration study of line confidence (GW-E2E-005, GW-E2E-009).

The work makes no claim of a new method. The detector and recognizer follow published designs, and
the contribution is the record: what was run, on which data, and what came out, favourable or not.

## How to read the numbers

Every number in these notes traces to an experiment ID and a generated report. Tracks are strict
unless stated, and each interval says whether it is over lines, pages or documents. Only the
registered confirmatory tests (H1′, H5′, H8′ and the Phase 16 tests) carry the verdicts met, not met
and holds. Other hypotheses keep the exploratory status recorded in `research/hypotheses.md`, and
every other interval is descriptive.
