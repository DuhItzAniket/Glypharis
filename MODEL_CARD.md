# Model card: GLYPHARIS v1

Status: evaluated and packaged. Final validation (Phase 22) has been run; its record is
[`docs/FINAL_VALIDATION.md`](docs/FINAL_VALIDATION.md). v1.0 is the release of v1 (GW-REC-034 with
GW-DET-001): the code under Apache-2.0, published on PyPI as `glypharis` 1.0.0, and the v1 weights as
the asset `glypharis-v1.tar.gz` of the GitHub release `v1.0`, which `glypharis.download_model`
fetches and checks against pinned hashes. The weights' terms are under
[licence information](#licence-information). The phone candidate is not released.

This card describes `glypharis_v1`, a line-based OCR model for English printed text in document
images. It is written for someone deciding whether the model suits their use, and it gives the
source of each number so that the number can be checked. A document-reading application is the
intended consumer.

Contents: [summary](#summary), [how to read the numbers](#how-to-read-the-numbers),
[model description](#model-description), [model version](#model-version),
[intended use](#intended-use), [not intended use](#not-intended-use),
[architecture](#architecture), [training datasets](#training-datasets),
[evaluation datasets](#evaluation-datasets), [metrics](#metrics),
[limitations](#limitations), [known failure modes](#known-failure-modes),
[hardware](#hardware), [licence information](#licence-information),
[citation information](#citation-information),
[final configuration record](#final-configuration-record), [not v1](#not-v1).

## Summary

CER is the character error rate in percent, and lower is better. Square brackets hold the interval.
GDC is this project's test set of printed pages captured with a phone, FUNSD is a set of forms,
OmniDocBench a set of document pages, and HierText a public dataset of mostly scene photographs.
Terms are defined under [how to read the numbers](#how-to-read-the-numbers), and each value appears
again with its context under [metrics](#metrics).

Source: GW-REC-039 [`reports/external/recognition.md`](reports/external/recognition.md); GW-E2E-007
[`reports/external/end_to_end.md`](reports/external/end_to_end.md); GW-EFF-007
[`reports/efficiency/phase16/cost_table.md`](reports/efficiency/phase16/cost_table.md).

<!-- trace-table: GW-REC-039 reports/external/recognition.md#track-strict GDC line CER from ground-truth crops=GW-REC-039;gdc_test,CER %|HierText test line CER from ground-truth crops=GW-REC-039;hiertext_test,CER % -->
<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md#detection GDC line detection F1 @0.5=GW-E2E-007;gdc_test,F1 @0.5|HierText test line detection F1 @0.5=GW-E2E-007;hiertext_test,F1 @0.5 -->
<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md#track-strict GDC page CER, detector and recognizer together=GW-E2E-007;gdc_test,Page CER % (bag of lines)|FUNSD page CER, detector and recognizer together=GW-E2E-007;funsd_test,Page CER % (bag of lines)|OmniDocBench page CER, order-aware=GW-E2E-007;omnidocbench_test,Page CER % (order-aware)|HierText test page CER, detector and recognizer together=GW-E2E-007;hiertext_test,Page CER % (bag of lines) -->
<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages Mean page time, PyTorch on the CPU=PyTorch\, CPU\, 10 threads,Mean ms -->
<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/cost_table.md#parameters Parameters=total,Parameters -->
| Measure | Value | Set and unit | Experiment |
|---|---|---|---|
| GDC line CER from ground-truth crops | 4.17 [3.72, 4.67] % | phone captures of printed pages, CI by line | GW-REC-039 |
| GDC line detection F1 @0.5 | 80.80 [72.63, 88.07] % | phone captures of printed pages, CI by page | GW-E2E-007 |
| GDC page CER, detector and recognizer together | 22.35 [15.87, 30.11] % | phone captures of printed pages, bag of lines, CI by page | GW-E2E-007 |
| FUNSD page CER, detector and recognizer together | 32.05 [24.28, 39.85] % | forms, bag of lines, CI by page | GW-E2E-007 |
| OmniDocBench page CER, order-aware | 27.28 [23.95, 30.57] % | English document pages, this project's metric, CI by page | GW-E2E-007 |
| HierText test line CER from ground-truth crops | 6.03 [5.89, 6.18] % | mostly scene photographs, CI by line | GW-REC-039 |
| HierText test line detection F1 @0.5 | 79.13 [77.98, 80.26] % | mostly scene photographs, CI by page | GW-E2E-007 |
| HierText test page CER, detector and recognizer together | 25.16 [23.44, 26.79] % | mostly scene photographs, bag of lines, CI by page | GW-E2E-007 |
| Mean page time, PyTorch on the CPU | 2,566 [2,327, 2,829] ms | HierText validation pages, CI by page | GW-EFF-007 |
| Parameters | 11,116,044 | detector and recognizer | GW-EFF-007 |

Page CER also counts detection failures, which line CER from ground-truth crops does not. On
HierText test the largest part of the page error is lines the detector missed. On GDC missed lines
have the largest point estimate, but their interval overlaps that of the errors on matched lines
([known failure modes](#known-failure-modes)).

Curved pages and pages photographed sideways are not handled well. GDC's <!-- derived: pages_rotated --> 7
sideways pages have a bag-of-lines CER of <!-- derived: cap_rotated --> 80.53 [62.13, 92.43] % and
its <!-- derived: pages_curved --> 8 curved pages <!-- derived: cap_curved --> 64.60 [47.03, 82.96] %
(CI by page). These groups are small, and capture conditions are confounded with documents.

Some baselines do better on some sets: PP-OCRv6 medium has a lower error on GDC crops and IIIT5K
words, and Tesseract 5 a lower order-aware page CER on OmniDocBench, each with a paired interval that
excludes zero. On FUNSD the PaddleOCR systems have higher detection F1 point values; no difference is
stated. See [comparison with baselines](#comparison-with-baselines). Every line and page latency
target of ADR-0011 was missed.

Final validation (Phase 22) has been run; its record is
[`docs/FINAL_VALIDATION.md`](docs/FINAL_VALIDATION.md). A re-run of the frozen v1 gave the same
predictions as Phase 14 in every test file. The strict reproduction verdict failed only on official
OmniDocBench scores, through that evaluator's per-page time limit, and the owner accepted this as a
disclosed limitation (GW-E2E-018,
[`reports/final_validation/v1_reproduction.md`](reports/final_validation/v1_reproduction.md)).

The code and the project's own text are under Apache-2.0 ([`LICENSE`](LICENSE),
[`NOTICE`](NOTICE)), with the exceptions that `NOTICE` lists; the Apache License does not cover the
weights, the release asset `glypharis-v1.tar.gz`. The weights are distributed under CC BY-SA 4.0
with attribution to HierText and Open Images (ADR-0014). The detector was initialised from ImageNet
weights, a disclosed exception to the project's commercial-use rule (ADR-0009), and the weights are
not described as cleared for commercial use. The owner decided on 2026-11-22 to publish them under
these terms without the qualified advice ADR-0014 had asked for; this is the project's chosen
handling, not a legal conclusion. Cite with [`CITATION.cff`](CITATION.cff) (DOI
[10.5281/zenodo.24433864](https://doi.org/10.5281/zenodo.24433864)).

Not done yet:

- Phones are the target hardware, and no full page has been timed on a phone. Phase 22 measured
  only the networks of the phone candidate, a new version, on two phones (GW-EFF-015, see
  [not v1](#not-v1)); v1 itself was not measured on a phone.
- No set of old scans was evaluated ([limitations](#limitations)).

## How to read the numbers

<!-- trace: GW-E2E-009 reports/external/README.md -->
<!-- trace: GW-E2E-007 reports/external/end_to_end.md#detection -->
- CER and WER are character and word error rates in percent, lower is better. Results are on the
  strict track unless a table says otherwise.
- Intervals are 95% percentile bootstrap intervals with 1,000 resamples and seed 20261010. Each
  table says whether it resamples lines, pages or documents. Intervals outside the confirmatory
  tests are descriptive.
- `[cited CI]` is the error-analysis reports' name for their main interval. The text next to each
  such table gives what it resamples.
- Each table is preceded by a source line with the experiment ID and the report, from the
  repository root. Experiments are registered in
  [`research/experiments.md`](research/experiments.md).
- Comparisons with other systems either carry a paired interval or say that no difference is
  stated. Latencies from different sittings are never compared.
- HierText is released under CC BY-SA 4.0; its images come from Open Images, which lists them as
  CC BY 2.0. Tables derived from HierText are shared alike under CC BY-SA 4.0.
- GDC-scan is one 28-page printed document. It is a case study, its intervals are conditional on
  that document, and it is reported as numbers only.

Terms used in this card:

| Term | Meaning |
|---|---|
| Phase | A numbered stage of the project, tracked in [`docs/PHASE_STATUS.md`](docs/PHASE_STATUS.md) |
| Sitting | One measurement session on the laptop |
| Track | How text is prepared before scoring, defined in [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) §2. On the strict track case, punctuation and symbols all count. The normalised track also folds case and maps typographic variants such as curly quotes |
| GDC | Glypharis Document Challenge, this project's test set of printed pages captured with a phone |
| Clean slice | GDC's `scan_app` and `photo_flat` captures |
| GlyphSynth | This project's generator of synthetic document-style text lines |
| GLYPH-C | This project's corruption benchmark: validation lines, half of them from GlyphSynth, under seven corruptions at five levels |
| mCE | Mean corruption error, the unweighted mean of CER over GLYPH-C's corruption levels |
| Document slice | Validation pages with many lines. It is a density proxy and not a labelled set of documents |
| Bag-of-lines page CER | Predicted lines are matched to reference lines by box. Matched pairs count their edit errors and unmatched lines count in full, whatever their order |
| Order-aware page CER | The lines are joined in predicted reading order and compared with the reference page text, so reading order is charged too |
| F1 @0.5 | Detection F1 where a predicted box matches a reference line at IoU 0.5; `P` and `R` are precision and recall |
| E2E F1 | Line F1 where a reference line counts as found only if a prediction overlaps it at IoU 0.5 or more and its line CER is at most 0.1 |
| ECE | Expected calibration error, the gap between a line's confidence and how often such lines are right |
| fp32, strict fp32, bf16 | fp32 is single-precision floating point. Strict fp32 means fp32 with TF32 off, and has nothing to do with the strict text track. bf16 is bfloat16 |
| H and F numbers | Hypotheses in [`research/hypotheses.md`](research/hypotheses.md) and findings in [`research/findings.md`](research/findings.md) |

## Model description

GLYPHARIS v1 is a two-stage OCR pipeline:

1. a line detector finds text lines on the page as four-point boxes;
2. each line is cropped and rectified;
3. a line recognizer reads the crop into text with CTC greedy decoding and gives a confidence;
4. a reading-order step sorts the lines with a recursive XY-cut after deskewing;
5. the result is returned as structured output: text, and lines with box, confidence and timing.

The recognizer was trained from random initialisation. The detector's backbone started from
ImageNet-1K classification weights. No weights trained for text detection or recognition were used
([ADR-0007](docs/decisions/ADR-0007-pretrained-weights-policy.md)). The configuration was frozen by
GW-REC-037 using validation data only, before the frozen model read any test set
([`configs/final/glypharis_v1.yaml`](configs/final/glypharis_v1.yaml)).

The input is an image path, image file bytes, a PIL image or a NumPy array. The default backend of
v1 is PyTorch ([ADR-0011](docs/decisions/ADR-0011-cpu-latency-target.md), owner decision of
2026-11-18). Usage is described in [`README.md`](README.md).

## Model version

Source: GW-EFF-006 [`reports/efficiency/phase16/v1_bundle.json`](reports/efficiency/phase16/v1_bundle.json),
[`configs/final/glypharis_v1.yaml`](configs/final/glypharis_v1.yaml),
[`glypharis/__init__.py`](glypharis/__init__.py), [`reports/export/v1-code/code_diff.json`](reports/export/v1-code/code_diff.json).

| Item | Value |
|---|---|
| Model version | `glypharis_v1` |
| Package version | `1.0.0` (code, published on PyPI as `glypharis`) |
| Bundle format | `glypharis-bundle-1`, with ONNX files at opset 17 |
| Bundle manifest SHA-256 | `875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037` |
| `detector.pt` SHA-256 | `caafdce9402de07a775102558eeffa6f48d4f54cd370ac63c09ab0be2a929c99` |
| `recognizer.pt` SHA-256 | `785409f75f1d31e4933a42ddcae8c9ecb84960d975083f3f81ebc47d6070f9ca` |
| `detector.onnx` SHA-256 | `378c1feda480c1bb1dde838dd4a5f4579a904710a7f7b222d978c583b5bddcb7` |
| `recognizer.onnx` SHA-256 | `930329b4bf635a935975823d2004b47371d9123b2504cfe590d52f44db84f438` |
| Freeze record SHA-256 | `0aa61f9ef4843aaf6c1db0d0f3378b0c5f519fe375b1bf70f3b858509dfb4f32` |
| Code evaluated in Phase 14 | tag `phase14-eval`, commit `6795186be166d6b60238c701753aad32e44d0e98` |
| Code at the end of the v1 freeze | tag `v1-code`, commit `efb197a5cd17570d810d39fe88a6895e20c4c639` |

The tag `v1-code` differs from `phase14-eval` only in docstrings, comments and the default bundle
name of `from_pretrained`, so it is not a new model version. The earlier version v0.1 used the
recognizer GW-REC-024. Its numbers are not v1's and are not used here.

## Intended use

- Reading English printed text from document images: scans, photographs of flat, upright pages and
  image-only pages, one image at a time, with lines, boxes and confidences as output. Curved pages,
  such as an open book, and pages photographed sideways are not handled well; see
  [not intended use](#not-intended-use).
- Use on a laptop or desktop through the PyTorch backend, the configuration that was evaluated. One
  laptop was measured.
- Use where the output is searched, indexed or read by a person who can tolerate errors. The page
  error rates under [metrics](#metrics) show how many errors to expect.

Phones are the target hardware. Phase 22 measured the phone candidate's networks on two phones
(GW-EFF-015): network time and memory only, without decoding, post-processing, cropping or reading
order; v1 itself was not measured on a phone ([not v1](#not-v1)). The laptop figures in this card
rank builds on that laptop and say nothing about phone speed
([ADR-0011](docs/decisions/ADR-0011-cpu-latency-target.md)).

## Not intended use

Source: as named in each row. Intervals by page.

<!-- trace-table: data/manifests/processed/hiertext.json data/manifests/processed/hiertext.json Handwriting=splits/train/rec_skip/handwritten -->
<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md#gdc-per-capture Pages photographed sideways=GW-E2E-007;e2e;rotated|Curved pages, such as an open book=GW-E2E-007;e2e;curved -->
<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md#vertical-lines Vertical text=GW-E2E-007;e2e;gdc_test -->
<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md#track-strict Unreviewed transcription where errors matter=GW-E2E-007;hiertext_test -->
| Use | Why not | Evidence |
|---|---|---|
| Handwriting | Handwritten lines were left out of recognizer training: 25,122 lines of the training split | `data/manifests/processed/hiertext.json` |
| Scripts other than English | The character set has <!-- derived: charset_chars --> 274 characters plus the CTC blank; <!-- derived: charset_out --> 331 out-of-scope characters were dropped when it was built | `data/manifests/charset.json`, `reports/charset/charset_report.md` |
| Pages photographed sideways | No page-orientation step. GDC's 7 rotated pages have a bag-of-lines CER of 80.53 [62.13, 92.43] % | GW-E2E-007, `reports/external/end_to_end.md` |
| Curved pages, such as an open book | GDC's 8 curved pages have a bag-of-lines CER of 64.60 [47.03, 82.96] %. Capture conditions are confounded with documents, so this is not a contrast with other conditions | GW-E2E-007, `reports/external/end_to_end.md` |
| Vertical text | On GDC's 32 vertical lines, recall @0.5 is 18.75 [5.26, 40.00] % | GW-E2E-007, `reports/external/end_to_end.md` |
| Tables, formulas and layout analysis | The model returns text lines only. Formula and table metrics were off, so OmniDocBench's overall score could not be computed | GW-E2E-009, `reports/external/official.md` |
| On-device use on phones | v1 itself was not measured on a phone, and v1 keeps PyTorch as its default backend. Only the networks of the phone candidate, a new version, were measured on phones, without decoding, post-processing, cropping or reading order ([not v1](#not-v1)) | ADR-0011, GW-EFF-015 |
| A service open to a network | The optional HTTP server is for local use and has no authentication or TLS | `README.md` |
| Unreviewed transcription where errors matter | On the 1,634 HierText test pages, page CER is 25.16 [23.44, 26.79] % | GW-E2E-007, `reports/external/end_to_end.md` |

## Architecture

Source: GW-REC-034 and GW-DET-001 in [`configs/final/glypharis_v1.yaml`](configs/final/glypharis_v1.yaml),
[`configs/rec/gw-rec-020.yaml`](configs/rec/gw-rec-020.yaml) (encoder sizes),
[`configs/det/gw-det-001.yaml`](configs/det/gw-det-001.yaml) (detector model),
[`reports/efficiency/phase16/profiles/lines_onnx_cpu_10t.json`](reports/efficiency/phase16/profiles/lines_onnx_cpu_10t.json)
(frame stride), [`reports/charset/charset_report.md`](reports/charset/charset_report.md), and GW-EFF-007
[`reports/efficiency/phase16/cost_table.md`](reports/efficiency/phase16/cost_table.md) (sizes).

Recognizer, GW-REC-034: an SVTR-style encoder with a CTC head.

<!-- trace-table: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#recognizer Input height, pixels=input/height|Maximum input width, pixels=input/max_width|Parameters=parameters|Confidence temperature T=temperature -->
<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/profiles/lines_onnx_cpu_10t.json Frame stride, input columns per output frame=frame_stride -->
<!-- trace-table: data/manifests/charset.json data/manifests/charset.json Output classes, with the CTC blank=vocabulary_size -->
<!-- trace-table: configs/rec/gw-rec-020.yaml configs/rec/gw-rec-020.yaml#model/svtr Encoder stage widths=dims|Local mixing blocks per stage=local_blocks|Global mixing blocks per stage=global_blocks|Attention heads per stage=heads|MLP ratio=mlp_ratio|Local mixing kernel=local_kernel -->
<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/cost_table.md#parameters Weights file, release .pt / fp32 ONNX, MiB=recognizer -->
| Setting | Value |
|---|---|
| Input height, pixels | 48 |
| Maximum input width, pixels | 1536 |
| Frame stride, input columns per output frame | 4 |
| Output classes, with the CTC blank | 275 |
| Encoder stage widths | 96, 192, 256 |
| Local mixing blocks per stage | 3, 3, 0 |
| Global mixing blocks per stage | 0, 3, 6 |
| Attention heads per stage | 3, 6, 8 |
| MLP ratio | 4.0 |
| Local mixing kernel | 3 × 7 |
| Parameters | 7,928,835 |
| Weights file, release .pt / fp32 ONNX, MiB | 30.31 / 30.40 |
| Confidence temperature T | 2.35 |

A line's confidence is the geometric mean of the top-class probability over the frames that emit a
character, after dividing the logits by T. Decoding does not depend on T.

Detector, GW-DET-001: DBNet-style, with a MobileNetV3-Large backbone.

<!-- trace-table: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#det/model Inner channels=inner_channels -->
<!-- trace-table: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector Parameters at inference=parameters|Input long side, pixels=infer/long_side|Binarisation threshold=post/bin_thresh|Box score threshold=post/box_thresh|Minimum component area, pixels=post/min_component_area|Minimum box side, pixels=post/min_box_side|Box expansion ratio, operating point `inverse09`=post/operating_points/unclip_ratio -->
<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/cost_table.md#parameters Weights file, release .pt / fp32 ONNX, MiB=detector -->
| Setting | Value |
|---|---|
| Inner channels | 96 |
| Parameters at inference | 3,187,209 |
| Input long side, pixels | 1536 |
| Binarisation threshold | 0.3 |
| Box score threshold | 0.5 |
| Minimum component area, pixels | 1 |
| Minimum box side, pixels | 3.0 |
| Box expansion ratio, operating point `inverse09` | 0.9 |
| Weights file, release .pt / fp32 ONNX, MiB | 12.44 / 12.15 |

The post-processing settings come from GW-DET-004 and the input size from GW-DET-005, both tuned on
validation data. The threshold head used in training is not part of the inference model.

## Training datasets

<!-- trace: GW-REC-037 reports/ablation/phase13_ladder.md -->
Both models were trained on HierText only. No GlyphSynth line is in v1: adding 25K GlyphSynth lines
(row L2 of GW-REC-037's ablation ladder,
[`reports/ablation/phase13_ladder.md`](reports/ablation/phase13_ladder.md)) showed no clear effect
on validation CER and was not kept.

Source: [`data/manifests/processed/hiertext.json`](data/manifests/processed/hiertext.json), training
split. Dataset counts have no experiment ID; the manifest stands in for one.

<!-- trace-table: data/manifests/processed/hiertext.json data/manifests/processed/hiertext.json#splits/train Lines in the training split=lines|Illegible=rec_skip/illegible|Handwritten=rec_skip/handwritten|Vertical=rec_skip/vertical|Characters outside the charset=rec_skip/out_of_charset|Too long=rec_skip/too_long|Lines used to train the recognizer=lines_rec_ok|Pages used to train the detector=pages|Images excluded from training=images_excluded -->
| Item | Count |
|---|---|
| Lines in the training split | 493,464 |
| Illegible | 115,764 |
| Handwritten | 25,122 |
| Vertical | 2,039 |
| Characters outside the charset | 1,457 |
| Too long | 147 |
| Lines used to train the recognizer | 348,935 |
| Pages used to train the detector | 8,213 |
| Images excluded from training | 68 |

<!-- trace: data/manifests/processed/hiertext.json data/manifests/processed/hiertext.json#crop_settings -->
The recognizer was validated on <!-- derived: val_lines_manifest --> 73,888 HierText validation
lines and the detector on <!-- derived: val_pages_manifest --> 1,715 validation pages. The test
splits were not used for training or tuning. HierText line crops for training were cut at height 64
with a margin of 0.15, a maximum aspect ratio of 64 and JPEG quality 95.

The detector's backbone was initialised from torchvision's MobileNetV3-Large ImageNet-1K weights,
SHA-256 `8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04`.

### Recognizer training recipe

Source: GW-REC-034 [`reports/training/GW-REC-034/report.md`](reports/training/GW-REC-034/report.md),
[`configs/final/glypharis_v1.yaml`](configs/final/glypharis_v1.yaml) and the config chain
`configs/rec/gw-rec-034.yaml`, `gw-rec-024.yaml`, `gw-rec-022.yaml`, `gw-rec-020.yaml`,
`gw-rec-003.yaml`, `gw-rec-001.yaml`, `configs/base.yaml`.

<!-- trace-table: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#recognizer/training Epochs=epochs|Optimizer steps=steps|Selected epoch, numbered from zero=selected_epoch|GPU-hours=gpu_hours -->
<!-- trace-table: configs/rec/gw-rec-020.yaml configs/rec/gw-rec-020.yaml#train Learning rate=learning_rate -->
<!-- trace-table: configs/rec/gw-rec-022.yaml configs/rec/gw-rec-022.yaml Weight decay=train/weight_decay|Lines per micro-batch=train/batch_size|Micro-batches per optimizer step=train/gradient_accumulation|Maximum columns per micro-batch=data/max_batch_columns -->
<!-- trace-table: configs/rec/gw-rec-001.yaml configs/rec/gw-rec-001.yaml#train Gradient clipping norm=grad_clip_norm -->
| Setting | Value |
|---|---|
| Epochs | 30 |
| Optimizer steps | 169,860 |
| Selected epoch, numbered from zero | 29 |
| GPU-hours | 20.11 |
| Optimizer and schedule | AdamW, OneCycle |
| Learning rate | 5e-4 |
| Weight decay | 1e-2 |
| Lines per micro-batch | 8 |
| Micro-batches per optimizer step | 8 |
| Maximum columns per micro-batch | 6144 |
| Gradient clipping norm | 5.0 |
| Precision | bf16, deterministic mode |
| Augmentation | preset `full`, with `rotate: {p: 0.5, severity: 1.0}` from `configs/rec/gw-rec-034.yaml` |

The run was resumed four times from its last checkpoint; the resumes are listed in the run report.
The checkpoint with the lowest validation CER on the strict track was selected.

### Detector training recipe

Source: GW-DET-001 [`reports/training/GW-DET-001/report.md`](reports/training/GW-DET-001/report.md),
[`configs/det/gw-det-001.yaml`](configs/det/gw-det-001.yaml),
[`configs/final/glypharis_v1.yaml`](configs/final/glypharis_v1.yaml).

<!-- trace-table: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/training Epochs=epochs|Optimizer steps=steps|Selected epoch, numbered from zero=selected_epoch -->
<!-- trace-table: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#train Batch size=batch_size|Learning rate=learning_rate|Weight decay=weight_decay|Warm-up steps=warmup_steps|Gradient clipping norm=grad_clip_norm -->
<!-- trace-table: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#det/data Crop size, pixels=crop_size|Crops per page=crops_per_page|Minimum crop scale=scale_min|Maximum crop scale=scale_max|Maximum rotation, degrees=rotate_max_deg -->
| Setting | Value |
|---|---|
| Epochs | 24 |
| Optimizer steps | 98,544 |
| Selected epoch, numbered from zero | 22 |
| GPU-hours | <!-- derived: det_gpu_hours --> 4.86 |
| Optimizer and schedule | AdamW, cosine |
| Batch size | 8 |
| Learning rate | 5e-3 |
| Weight decay | 1e-4 |
| Warm-up steps | 2,000 |
| Gradient clipping norm | 10 |
| Crop size, pixels | 640 |
| Crops per page | 4 |
| Minimum crop scale | 0.5 |
| Maximum crop scale | 2.0 |
| Maximum rotation, degrees | 10 |

The detector was trained once, with one seed. GW-DET-007, a variant with labels for smaller text,
was tried and not adopted.

Its validation F1 @0.5 differs from GW-DET-001's by <!-- derived: det007_f1_diff --> -0.0111
[-0.0146, -0.0075] (CI by page, [`pair.json`](reports/error_cases/detector/GW-DET-007/pair.json)).

### Selection evidence

All of it is on validation data.

Source: GW-REC-037 [`reports/ablation/phase13_final.md`](reports/ablation/phase13_final.md).

<!-- trace-table: GW-REC-037 reports/ablation/phase13_final.md Three seeds of the final recipe=GW-REC-034\, GW-REC-035\, GW-REC-036 -->
| Runs | Val CER % (per seed) | Val CER % (mean) | Δ vs baseline | Document slice % | Δ doc | GLYPH-C mCE % (per seed) |
|---|---|---|---|---|---|---|
| Three seeds of the final recipe | 6.30, 6.28, 6.30 | 6.29 | -1.24 [-1.40, -1.11] | 9.15 | -1.08 [-1.41, -0.78] | 3.54, 3.53, 3.53 |

The differences are in points against the CRNN baseline GW-REC-001 at the same budget, paired over
pages. They measure the whole recipe, which changes the encoder, input height, learning rate,
weight decay, micro-batching and rotation range together, so they are not the effect of the encoder
alone.

## Evaluation datasets

Source: GW-E2E-009 [`reports/external/README.md`](reports/external/README.md),
[`reports/external/run_record.md`](reports/external/run_record.md),
[`research/test_access_log.md`](research/test_access_log.md), and the set counts in
[`recognition.md`](reports/external/recognition.md) and
[`end_to_end.md`](reports/external/end_to_end.md). GDC's document count is in GW-E2E-010
[`oracle_share.md`](reports/error_cases/phase15/oracle_share.md).

<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md#detection HierText test=GW-E2E-007;hiertext_test,Pages|GDC=GW-E2E-007;gdc_test,Pages|GDC-scan=GW-E2E-007;gdc_scan_test,Pages|FUNSD=GW-E2E-007;funsd_test,Pages|OmniDocBench (English pages)=GW-E2E-007;omnidocbench_test,Pages -->
| Set | Role | Pages | Crops or words | Note |
|---|---|---|---|---|
| HierText test | held-out split of the training dataset; mixed scene and document images | 1,634 | <!-- derived: n_ht --> 67,574 lines | |
| GDC | internal test set: printed pages captured with a phone under seven conditions | 106 | <!-- derived: n_gdc --> 8,372 lines | from <!-- derived: n_gdc_docs --> 21 source documents |
| GDC-scan | internal case study | 28 | <!-- derived: n_scan --> 776 lines | one printed document |
| FUNSD | external test set of forms | 50 | <!-- derived: n_funsd --> 2,647 lines | includes handwritten fill-ins |
| IIIT5K | external test set of word crops | n/a | <!-- derived: n_iiit --> 3,000 words | scene-text words, recognition only |
| OmniDocBench (English pages) | external test set of document pages | 755 | n/a | reference unit is a layout block |

The test sets were evaluated once in Phase 14 with the frozen model, in PyTorch on the RTX 4050 at
strict fp32, from code tag `phase14-eval`. Each evaluation is a row in the access log, which holds
Phase 5 and Phase 14 only. The Phase 15 error analysis evaluated nothing: it read hashed saved
outputs, recorded in
[`reports/error_cases/phase15/results.json`](reports/error_cases/phase15/results.json). HierText
validation is the reference set (GW-REC-038, GW-E2E-006). The scorer's record shows a dirty working
tree because it reads the git state after writing its own access-log rows; see the Phase 14 notes in
[`docs/PHASE_STATUS.md`](docs/PHASE_STATUS.md).

Test results come from the GPU strict-fp32 evaluation; equality with the bundle is shown on
validation only. Phase 22 checked reproduction at the 0.05-point tolerance (GW-E2E-018,
[`reports/final_validation/v1_reproduction.md`](reports/final_validation/v1_reproduction.md)): every
prediction was the same, and the check found 16 failures among the 1,381 point values checked, all of
them official OmniDocBench scores. The cause is the official evaluator's per-page time limit, which sent one more page to its
fallback scoring than in Phase 14; no count, flag or hypothesis verdict changed, and the owner
accepted the result as a disclosed limitation. Phase 14's results stay as recorded. The tolerance is
defined under Phase 22 in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

## Metrics

### Validation reference values

Some quantities have more than one recorded value because they were measured on different sets.
Only the values marked as the reference belong beside test results.

Source: GW-REC-034 [report](reports/training/GW-REC-034/report.md); GW-REC-038 and GW-E2E-006
[`reports/external/recognition.md`](reports/external/recognition.md) and
[`end_to_end.md`](reports/external/end_to_end.md); GW-EFF-009
[`reports/efficiency/phase16/pareto.md`](reports/efficiency/phase16/pareto.md); GW-REC-034
[`reports/robustness/GW-REC-034.json`](reports/robustness/GW-REC-034.json).

<!-- trace-table: GW-REC-038 reports/external/recognition.md#track-strict Line CER, evaluation crops=GW-REC-038;hiertext_val -->
<!-- trace-table: GW-E2E-006 reports/external/end_to_end.md#track-strict Page CER, all validation pages=GW-E2E-006;hiertext_val -->
| Quantity | Value % | Evaluated on | Experiment | Use |
|---|---|---|---|---|
| Line CER, training run | <!-- derived: val_line_run --> 6.30 | <!-- derived: val_line_run_n --> 73,888 in-charset validation lines | GW-REC-034 | model selection |
| Line CER, evaluation crops | 6.51 [6.36, 6.65] | 74,500 validation crops, CI by line | GW-REC-038 | reference beside test results |
| Line CER, seeded sample | <!-- derived: val_line_sample --> 6.22 | <!-- derived: val_line_sample_n --> 5,000 validation lines | GW-EFF-009 | build comparisons only |
| Page CER, all validation pages | 27.57 [25.11, 30.36] | 1,715 pages, CI by page | GW-E2E-006 | reference beside test results |
| Page CER, Phase 5 page sample | <!-- derived: val_page_sample --> 24.32 | <!-- derived: val_page_sample_n --> 400 validation pages | GW-EFF-009 | build comparisons only |
| GLYPH-C clean CER | <!-- derived: glyphc_clean --> 1.93 | <!-- derived: glyphc_lines --> 6,000 lines, half of them GlyphSynth | GW-REC-034 | not comparable with HierText CER |

### Recognition from ground-truth crops

Ground-truth line crops in, text out. Strict track, CI by line.

Source: GW-REC-039 and GW-REC-038, [`reports/external/recognition.md`](reports/external/recognition.md).

<!-- trace-table: GW-REC-038 reports/external/recognition.md#track-strict HierText validation (reference)=GW-REC-038;hiertext_val -->
<!-- trace-table: GW-REC-039 reports/external/recognition.md#track-strict HierText test=GW-REC-039;hiertext_test|GDC=GW-REC-039;gdc_test|FUNSD=GW-REC-039;funsd_test|GDC-scan=GW-REC-039;gdc_scan_test -->
| Set | Experiment | Lines | CER % | WER % | Seq acc % |
|---|---|---|---|---|---|
| HierText validation (reference) | GW-REC-038 | 74,500 | 6.51 [6.36, 6.65] | 19.78 [19.49, 20.05] | 70.62 |
| HierText test | GW-REC-039 | 67,574 | 6.03 [5.89, 6.18] | 19.26 [18.95, 19.56] | 71.37 |
| GDC | GW-REC-039 | 8,372 | 4.17 [3.72, 4.67] | 14.56 [13.44, 15.81] | 72.67 |
| FUNSD | GW-REC-039 | 2,647 | 7.03 [5.71, 8.66] | 25.88 [23.83, 27.90] | 61.54 |
| GDC-scan | GW-REC-039 | 776 | 1.02 [0.85, 1.20] | 6.08 [5.15, 7.00] | 77.06 |

IIIT5K is reported on the alphanumeric track, which lower-cases both sides and keeps only `0-9a-z`.
Its strict track charges case against upper-case labels and is not the headline. CI by word.

<!-- trace-table: GW-REC-039 reports/external/recognition.md#iiit5k-alphanumeric-track-headline Alphanumeric (headline)=GW-REC-039 -->
<!-- trace-table: GW-REC-039 reports/external/recognition.md#track-strict Strict (not the headline)=GW-REC-039;iiit5k_test -->
| IIIT5K track | Words | Word acc % | CER % |
|---|---|---|---|
| Alphanumeric (headline) | 3,000 | 91.47 [90.43, 92.47] | 2.37 [2.06, 2.70] |
| Strict (not the headline) | 3,000 | 47.23 | 44.27 [42.46, 46.16] |

### Detection and end to end

The whole pipeline on test pages. Strict track, CI by page. Detection columns are scored at
IoU @0.5.

Source: GW-E2E-007, GW-E2E-006 (validation) and GW-E2E-008 (ground-truth boxes),
[`reports/external/end_to_end.md`](reports/external/end_to_end.md).

<!-- trace-table: GW-E2E-006 reports/external/end_to_end.md#detection HierText validation (reference)=GW-E2E-006;hiertext_val -->
<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md#detection HierText test=GW-E2E-007;hiertext_test|GDC=GW-E2E-007;gdc_test|FUNSD=GW-E2E-007;funsd_test|GDC-scan=GW-E2E-007;gdc_scan_test -->
| Set | Experiment | Pages | P @0.5 | R @0.5 | F1 @0.5 |
|---|---|---|---|---|---|
| HierText validation (reference) | GW-E2E-006 | 1,715 | 85.19 [83.81, 86.44] | 71.57 [69.53, 73.38] | 77.79 [76.21, 79.19] |
| HierText test | GW-E2E-007 | 1,634 | 85.58 [84.48, 86.65] | 73.58 [72.05, 75.03] | 79.13 [77.98, 80.26] |
| GDC | GW-E2E-007 | 106 | 77.74 [66.93, 88.31] | 84.10 [78.88, 88.70] | 80.80 [72.63, 88.07] |
| FUNSD | GW-E2E-007 | 50 | 79.14 [75.96, 82.27] | 79.00 [75.09, 82.91] | 79.07 [75.56, 82.16] |
| GDC-scan | GW-E2E-007 | 28 | 96.36 [93.35, 98.86] | 95.49 [92.31, 98.17] | 95.92 [92.97, 98.16] |

Order-aware CER exists where a set has a reading order. On OmniDocBench it is this project's metric
and not the benchmark's official protocol.

<!-- trace-table: GW-E2E-006 reports/external/end_to_end.md#track-strict HierText validation (reference)=GW-E2E-006;hiertext_val -->
<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md#track-strict HierText test=GW-E2E-007;hiertext_test|GDC=GW-E2E-007;gdc_test|FUNSD=GW-E2E-007;funsd_test|GDC-scan=GW-E2E-007;gdc_scan_test|OmniDocBench=GW-E2E-007;omnidocbench_test -->
<!-- trace-table: GW-E2E-008 reports/external/end_to_end.md#track-strict GDC, ground-truth boxes=GW-E2E-008;gdc_test|GDC-scan, ground-truth boxes=GW-E2E-008;gdc_scan_test -->
| Set | Experiment | Pages | Page CER % (order-aware) | Page CER % (bag of lines) | E2E F1 % |
|---|---|---|---|---|---|
| HierText validation (reference) | GW-E2E-006 | 1,715 | n/a | 27.57 [25.11, 30.36] | 58.24 [56.64, 59.91] |
| HierText test | GW-E2E-007 | 1,634 | n/a | 25.16 [23.44, 26.79] | 60.18 [58.71, 61.72] |
| GDC | GW-E2E-007 | 106 | 25.37 [20.00, 30.80] | 22.35 [15.87, 30.11] | 69.36 [60.69, 77.48] |
| GDC, ground-truth boxes | GW-E2E-008 | 106 | 5.72 [3.41, 8.69] | 4.17 [2.49, 6.28] | 88.34 [84.20, 91.73] |
| FUNSD | GW-E2E-007 | 50 | n/a | 32.05 [24.28, 39.85] | 62.81 [59.06, 66.48] |
| GDC-scan | GW-E2E-007 | 28 | 11.34 [7.10, 16.70] | 4.60 [2.10, 8.21] | 89.84 [85.88, 93.62] |
| GDC-scan, ground-truth boxes | GW-E2E-008 | 28 | 1.02 [0.80, 1.28] | 1.01 [0.78, 1.30] | 94.97 [91.85, 97.63] |
| OmniDocBench | GW-E2E-007 | 755 | 27.28 [23.95, 30.57] | n/a | n/a |

### Official evaluators

The benchmarks' own evaluation code, pinned in
[`data/manifests/evaluators.json`](data/manifests/evaluators.json). No intervals are reported.

Source: GW-E2E-007, scored in GW-E2E-009, [`reports/external/official.md`](reports/external/official.md).

<!-- trace-table: GW-E2E-007 reports/external/official.md#hiertext-line-level GLYPHARIS v1=GW-E2E-007 -->
<!-- trace-table: GW-BASE-006 reports/external/official.md#hiertext-line-level PP-OCRv6 small=GW-BASE-006 -->
| HierText, line level, 1,634 pages | Det-Fscore | Det-Precision | Det-Recall | E2E-Fscore | E2E-Precision | E2E-Recall |
|---|---|---|---|---|---|---|
| GLYPHARIS v1 | 0.7837 | 0.8475 | 0.7288 | 0.5038 | 0.5449 | 0.4685 |
| PP-OCRv6 small | 0.5555 | 0.6378 | 0.4920 | 0.3735 | 0.4289 | 0.3308 |

On OmniDocBench, lower is better. Formula and table metrics were off, so the overall score was not
computed. No difference is stated, because the evaluator gives no interval.

<!-- trace-table: GW-E2E-007 reports/external/official.md#omnidocbench-text-block GLYPHARIS v1, all pages=GW-E2E-007;all|GLYPHARIS v1, 100-page subset=GW-E2E-007;subset100 -->
<!-- trace-table: GW-BASE-001 reports/external/official.md#omnidocbench-text-block Tesseract, all pages=GW-BASE-001;all -->
| OmniDocBench | Count | text_block page avg | text_block whole | reading_order page avg |
|---|---|---|---|---|
| GLYPHARIS v1, all pages | 755 | 0.1007 | 0.1859 | 0.2443 |
| GLYPHARIS v1, 100-page subset | 100 | 0.1063 | 0.1881 | 0.2523 |
| Tesseract, all pages | 755 | 0.0966 | 0.0847 | 0.2352 |

These official OmniDocBench scores are Phase 14's and stay as recorded. In Phase 22 the strict
reproduction verdict failed on GLYPHARIS's official OmniDocBench scores, and re-scoring the saved
Phase 14 files (GW-E2E-015) changed official OmniDocBench scores of Tesseract and GLYPHARIS, both
times because the evaluator's per-page time limit sent more pages to its fallback scoring than in
Phase 14. The
owner accepted this as a disclosed limitation of the official evaluator (GW-E2E-018,
[`reports/final_validation/v1_reproduction.md`](reports/final_validation/v1_reproduction.md)).

### Comparison with baselines

<!-- trace: GW-E2E-009 reports/external/comparisons.md -->
For each set and measure, the row names the baseline with the lowest error among those run on that
set, whichever side it falls on. The difference is GLYPHARIS minus the baseline in points, with a
95% paired interval over the same lines, words or pages. A negative difference in CER favours
GLYPHARIS; for word accuracy a negative difference favours the baseline. These intervals are
descriptive. All baselines are in
[`reports/benchmark_results/comparison_table.md`](reports/benchmark_results/comparison_table.md).

Source: GW-E2E-009 [`reports/external/comparisons.md`](reports/external/comparisons.md), values from
[`recognition.md`](reports/external/recognition.md) and [`end_to_end.md`](reports/external/end_to_end.md).

<!-- trace-table: GW-REC-039 reports/external/comparisons.md HierText test crops, CER=GW-REC-039;GW-BASE-006;hiertext_test;strict,CER (crops or bag of lines) pp|GDC crops, CER=GW-REC-039;GW-BASE-007;gdc_test;strict,CER (crops or bag of lines) pp|GDC-scan crops, CER=GW-REC-039;GW-BASE-006;gdc_scan_test;strict,CER (crops or bag of lines) pp|FUNSD crops, CER=GW-REC-039;GW-BASE-006;funsd_test;strict,CER (crops or bag of lines) pp|IIIT5K alphanumeric, word accuracy=GW-REC-039;GW-BASE-007;iiit5k_test;alnum,Order-aware CER / word acc pp|IIIT5K alphanumeric, CER=GW-REC-039;GW-BASE-007;iiit5k_test;alnum,CER (crops or bag of lines) pp -->
<!-- trace-table: GW-E2E-007 reports/external/comparisons.md HierText test pages, bag of lines=GW-E2E-007;GW-BASE-006;hiertext_test;strict,CER (crops or bag of lines) pp|GDC pages, bag of lines=GW-E2E-007;GW-BASE-007;gdc_test;strict,CER (crops or bag of lines) pp|GDC pages, order-aware=GW-E2E-007;GW-BASE-001;gdc_test;strict,Order-aware CER / word acc pp|GDC-scan pages, bag of lines=GW-E2E-007;GW-BASE-007;gdc_scan_test;strict,CER (crops or bag of lines) pp|GDC-scan pages, order-aware=GW-E2E-007;GW-BASE-007;gdc_scan_test;strict,Order-aware CER / word acc pp|FUNSD pages, bag of lines=GW-E2E-007;GW-BASE-007;funsd_test;strict,CER (crops or bag of lines) pp|OmniDocBench pages, order-aware=GW-E2E-007;GW-BASE-001;omnidocbench_test;strict,Order-aware CER / word acc pp -->
| Set and measure | Interval over | Baseline | Baseline value % | GLYPHARIS value % | GLYPHARIS minus baseline, points |
|---|---|---|---|---|---|
| HierText test crops, CER | lines | PP-OCRv6 small | <!-- derived: b_ht_crop --> 7.06 | <!-- derived: g_ht_crop --> 6.03 | -1.03 [-1.14, -0.92] |
| GDC crops, CER | lines | PP-OCRv6 medium | <!-- derived: b_gdc_crop --> 3.20 | <!-- derived: g_gdc_crop --> 4.17 | +0.97 [+0.76, +1.20] |
| GDC-scan crops, CER | lines | PP-OCRv6 small | <!-- derived: b_scan_crop --> 0.37 | <!-- derived: g_scan_crop --> 1.02 | +0.65 [+0.51, +0.78] |
| FUNSD crops, CER | lines | PP-OCRv6 small | <!-- derived: b_funsd_crop --> 9.30 | <!-- derived: g_funsd_crop --> 7.03 | -2.27 [-3.43, -1.27] |
| IIIT5K alphanumeric, word accuracy | words | PP-OCRv6 medium | <!-- derived: b_iiit_acc --> 97.13 | <!-- derived: g_iiit_acc --> 91.47 | -5.67 [-6.53, -4.77] |
| IIIT5K alphanumeric, CER | words | PP-OCRv6 medium | <!-- derived: b_iiit_cer --> 0.67 | <!-- derived: g_iiit_cer --> 2.37 | +1.70 [+1.43, +1.98] |
| HierText test pages, bag of lines | pages | PP-OCRv6 small | <!-- derived: b_ht_page --> 42.93 | <!-- derived: g_ht_page --> 25.16 | -17.77 [-19.51, -16.01] |
| GDC pages, bag of lines | pages | PP-OCRv6 medium | <!-- derived: b_gdc_bag --> 20.96 | <!-- derived: g_gdc_bag --> 22.35 | +1.39 [-2.71, +6.88] |
| GDC pages, order-aware | pages | Tesseract 5 | <!-- derived: b_gdc_ord --> 31.90 | <!-- derived: g_gdc_ord --> 25.37 | -6.53 [-13.51, +0.08] |
| GDC-scan pages, bag of lines | pages | PP-OCRv6 medium | <!-- derived: b_scan_bag --> 4.94 | <!-- derived: g_scan_bag --> 4.60 | -0.34 [-1.40, +0.29] |
| GDC-scan pages, order-aware | pages | PP-OCRv6 medium | <!-- derived: b_scan_ord --> 15.32 | <!-- derived: g_scan_ord --> 11.34 | -3.98 [-14.50, +4.40] |
| FUNSD pages, bag of lines | pages | PP-OCRv6 medium | <!-- derived: b_funsd_bag --> 27.82 | <!-- derived: g_funsd_bag --> 32.05 | +4.23 [-0.52, +8.76] |
| OmniDocBench pages, order-aware | pages | Tesseract 5 | <!-- derived: b_omni_ord --> 15.70 | <!-- derived: g_omni_ord --> 27.28 | +11.58 [+8.37, +14.88] |

Detection on FUNSD has no paired interval. Its F1 @0.5 is <!-- derived: funsd_f1_glypharis --> 79.07
[75.56, 82.16] % for GLYPHARIS and lies below the point value of every PaddleOCR system run on
FUNSD, from <!-- derived: funsd_f1_v5_mobile --> 81.81 [78.59, 84.88] % (PP-OCRv5 mobile) to
PP-OCRv5 server's <!-- derived: funsd_f1_v5_server --> 85.03 [82.10, 87.72] % (CI by page); no
difference is stated ([`end_to_end.md`](reports/external/end_to_end.md)).

On OmniDocBench the four PaddleOCR systems run there have order-aware page CERs from PP-OCRv6
small's <!-- derived: omni_v6_small --> 60.29 [58.23, 62.26] % to PP-OCRv6 tiny's <!-- derived: omni_v6_tiny --> 60.62
[58.68, 62.55] %. GLYPHARIS has <!-- derived: omni_glypharis --> 27.28 [23.95, 30.57] % and
Tesseract 5 has <!-- derived: omni_tesseract --> 15.70 [14.23, 17.37] % (CI by page). The cause of
the PaddleOCR results was not established.

### Confirmatory tests

These tests were registered before the frozen model read any test set or, for Phase 16, before any
Phase 16 measurement. Their rules are in [`research/hypotheses.md`](research/hypotheses.md).

Source: GW-E2E-009 [`reports/external/README.md`](reports/external/README.md) and
[`results.json`](reports/external/results.json); GW-EFF-010 and GW-EFF-011
[`reports/efficiency/phase16/verdicts.md`](reports/efficiency/phase16/verdicts.md).

<!-- trace: GW-E2E-009 reports/external/README.md -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md -->
| Test | What was tested | Result | Verdict |
|---|---|---|---|
| H1′, CER half | Page CER (bag of lines) minus twice PP-OCRv5 mobile's, in points, on GDC's <!-- derived: h1_pages --> 53 clean pages | Strict track over pages: <!-- derived: h1_cer_pages --> -34.98 [<!-- derived: h1_cer_pages_low --> -47.44, <!-- derived: h1_cer_pages_high --> -24.08]. Strict track over documents: <!-- derived: h1_cer_docs --> -34.98 [<!-- derived: h1_cer_docs_low --> -50.22, <!-- derived: h1_cer_docs_high --> -20.73] | met |
| H1′, latency half | PyTorch page time minus PP-OCRv5 mobile's, one sitting, 10 threads; means <!-- derived: gl_p14 --> 2,446.7 and <!-- derived: v5m_p14 --> 2,954.8 ms | <!-- derived: h1_latency --> -508.11 [-737.66, -256.52] ms per page | met |
| H1′ | Both halves | | holds |
| H1′ for the ONNX bundle | The same latency test for the fp32 ONNX bundle, another sitting; means <!-- derived: onnx_p16 --> 9,623.6 and <!-- derived: v5m_p16 --> 3,086.0 ms | <!-- derived: h1_onnx --> +6,537.6 [+5,383.7, +7,806.3] ms per page | not met |
| H5′ | GDC crop CER at least <!-- derived: h5_factor --> 1.5 times the validation crop CER, by line and by document. The value is the GDC CER minus that multiple of the validation CER | Over lines: -5.59 [-6.07, -5.01] points. Over documents: -5.59 [-7.88, -2.52] points | not met |
| H6′ | The SVTR-style GW-REC-020 within <!-- derived: h6_factor --> 1.5 times the CPU latency of the CRNN GW-REC-003, fp32 ONNX | <!-- derived: h6_test --> +16.78 [+12.02, +22.04] ms per line | not met |
| H8′, transfers | ECE at T = 2.35 minus ECE at T = 1 on GDC crops, below zero by line and by page | By line <!-- derived: h8_lines --> -0.1253 [-0.1319, -0.1190]; by page <!-- derived: h8_pages --> -0.1253 [-0.1423, -0.1013] | met |
| H8′, transfers by half | ECE at T = 2.35 minus half of ECE at T = 1 on GDC crops, below zero by line and by page | By line <!-- derived: h8_half_lines --> -0.0081 [-0.0142, -0.0021]; by page <!-- derived: h8_half_pages --> -0.0081 [-0.0168, +0.0044]. The two bootstraps disagree | not shown |

H8 is therefore not supported on GDC. The H1′ latency verdict applies to the PyTorch build on
validation pages, with PP-OCRv5 mobile running its two orientation models as it does for its
accuracy numbers. The ONNX verdict applies to this bundle hash only.

### Confidence calibration

<!-- trace: GW-E2E-005 reports/calibration/GW-E2E-005/calibration.md -->
The temperature T = 2.35 was fitted on half of the validation lines. On the other half, ECE fell
from <!-- derived: ece_raw --> 0.2140 to <!-- derived: ece_fit --> 0.0886, a difference (fitted
minus raw) of -0.1254 [-0.1296, -0.1217] by line (GW-E2E-005,
[`reports/calibration/GW-E2E-005/calibration.md`](reports/calibration/GW-E2E-005/calibration.md)).

Source: GW-REC-039, GW-REC-038 and GW-E2E-007, [`reports/external/calibration.md`](reports/external/calibration.md).
CI by line; 15 equal-mass bins.

<!-- trace-table: GW-REC-038 reports/external/calibration.md HierText validation=GW-REC-038;hiertext_val -->
<!-- trace-table: GW-REC-039 reports/external/calibration.md HierText test=GW-REC-039;hiertext_test|GDC crops=GW-REC-039;gdc_test|GDC-scan crops=GW-REC-039;gdc_scan_test|FUNSD crops=GW-REC-039;funsd_test|IIIT5K (alphanumeric)=GW-REC-039;iiit5k_test -->
<!-- trace-table: GW-E2E-007 reports/external/calibration.md GDC pipeline lines=GW-E2E-007 -->
| Set | Lines | Accuracy | ECE T=1 | ECE T=2.35 | ECE(2.35) - ECE(1) [CI] |
|---|---|---|---|---|---|
| HierText validation | 74,500 | 0.7062 | 0.2278 | 0.0904 | -0.1374 [-0.1409, -0.1344] |
| HierText test | 67,574 | 0.7137 | 0.2231 | 0.0920 | -0.1311 [-0.1345, -0.1277] |
| GDC crops | 8,372 | 0.7267 | 0.2345 | 0.1092 | -0.1253 [-0.1319, -0.1190] |
| GDC-scan crops | 776 | 0.7706 | 0.2019 | 0.1248 | -0.0771 [-0.0986, -0.0518] |
| FUNSD crops | 2,647 | 0.6154 | 0.3382 | 0.1227 | -0.2155 [-0.2181, -0.2052] |
| IIIT5K (alphanumeric) | 3,000 | 0.9147 | 0.0546 | 0.1118 | +0.0572 [+0.0431, +0.0716] |
| GDC pipeline lines | 9,057 | 0.5315 | 0.4150 | 0.2489 | -0.1660 [-0.1678, -0.1643] |

The confidence should be read as a ranking aid and not as a probability. Calibration became worse on
IIIT5K and remained at <!-- derived: ece_gdc_pipeline --> 0.2489 on the GDC pipeline's lines.

On HierText test, the lowest confidence quartile holds most of the errors. This interval is by page,
unlike the by-line intervals above.

Source: GW-REC-040 [`reports/error_cases/phase15/recognition.md`](reports/error_cases/phase15/recognition.md).

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md#hiertext_test Lowest confidence quartile=L83 -->
| Bin | Lines | Share of the set's errors % | CER % [cited CI] |
|---|---|---|---|
| Lowest confidence quartile | 16,297 | 78.51 | 24.01 [22.88, 25.25] |

### Synthetic corruptions and capture conditions

GLYPH-C v1 applies seven corruptions at five levels to <!-- derived: glyphc_lines --> 6,000
validation-derived lines, half from HierText and half from GlyphSynth, in all <!-- derived: glyphc_crops --> 216,000
crops. Values are CER % without intervals, because the report stores none. They are not comparable
with HierText CER. The clean CER is <!-- derived: glyphc_clean --> 1.93 % and the mean corruption
error (mCE) is <!-- derived: glyphc_mce --> 3.54 %.

Source: GW-REC-034, scored under GW-REC-037, [`reports/robustness/GW-REC-034.json`](reports/robustness/GW-REC-034.json);
pack counts in [`data/manifests/processed/glyph_c.json`](data/manifests/processed/glyph_c.json).
Corruption definitions: `glypharis/data/augment.py`.

<!-- trace-table: GW-REC-034 reports/robustness/GW-REC-034.json#cer_per_corruption_level rotate=rotate|low_resolution=low_resolution|jpeg=jpeg|perspective=perspective|gaussian_noise=gaussian_noise|gaussian_blur=gaussian_blur|contrast=contrast -->
| Corruption | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| rotate | 2.38% | 3.76% | 7.29% | 14.35% | 23.57% |
| low_resolution | 1.94% | 1.97% | 2.03% | 2.28% | 5.72% |
| jpeg | 1.95% | 1.96% | 2.01% | 2.11% | 4.08% |
| perspective | 2.00% | 2.21% | 2.50% | 2.97% | 3.55% |
| gaussian_noise | 1.99% | 2.16% | 2.39% | 2.74% | 3.12% |
| gaussian_blur | 1.93% | 1.96% | 2.06% | 2.26% | 2.63% |
| contrast | 1.93% | 1.93% | 1.95% | 2.01% | 2.31% |

Real capture conditions on GDC: bag-of-lines page CER of the whole pipeline, with the same pages from
ground-truth boxes in the second table. Each condition covers different documents, so the rows are
descriptive and are not contrasts between conditions. Intervals by page.

Source: GW-E2E-007 and GW-E2E-008, [`reports/external/end_to_end.md`](reports/external/end_to_end.md),
section "GDC per capture condition".

<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md#gdc-per-capture scan_app=GW-E2E-007;e2e;scan_app,CER % strict (crops or bag of lines)|photo_flat=GW-E2E-007;e2e;photo_flat,CER % strict (crops or bag of lines)|photo_angled=GW-E2E-007;e2e;photo_angled,CER % strict (crops or bag of lines)|distant=GW-E2E-007;e2e;distant,CER % strict (crops or bag of lines)|low_light=GW-E2E-007;e2e;low_light,CER % strict (crops or bag of lines)|curved=GW-E2E-007;e2e;curved,CER % strict (crops or bag of lines)|rotated=GW-E2E-007;e2e;rotated,CER % strict (crops or bag of lines) -->
| Condition | Pipeline bag-of-lines CER % |
|---|---|
| scan_app | 14.49 [7.73, 22.89] |
| photo_flat | 20.67 [3.07, 45.41] |
| photo_angled | 22.55 [5.82, 52.78] |
| distant | 14.21 [6.47, 25.90] |
| low_light | 8.13 [3.52, 14.83] |
| curved | 64.60 [47.03, 82.96] |
| rotated | 80.53 [62.13, 92.43] |

<!-- trace-table: GW-E2E-008 reports/external/end_to_end.md#gdc-per-capture scan_app, ground-truth boxes=GW-E2E-008;oracle;scan_app,CER % strict (crops or bag of lines)|photo_flat, ground-truth boxes=GW-E2E-008;oracle;photo_flat,CER % strict (crops or bag of lines)|photo_angled, ground-truth boxes=GW-E2E-008;oracle;photo_angled,CER % strict (crops or bag of lines)|distant, ground-truth boxes=GW-E2E-008;oracle;distant,CER % strict (crops or bag of lines)|low_light, ground-truth boxes=GW-E2E-008;oracle;low_light,CER % strict (crops or bag of lines)|curved, ground-truth boxes=GW-E2E-008;oracle;curved,CER % strict (crops or bag of lines)|rotated, ground-truth boxes=GW-E2E-008;oracle;rotated,CER % strict (crops or bag of lines) -->
| Condition | Bag-of-lines CER % from ground-truth boxes |
|---|---|
| scan_app, ground-truth boxes | 6.54 [2.21, 11.98] |
| photo_flat, ground-truth boxes | 3.05 [1.03, 6.23] |
| photo_angled, ground-truth boxes | 2.89 [0.81, 6.30] |
| distant, ground-truth boxes | 3.14 [1.23, 7.65] |
| low_light, ground-truth boxes | 1.33 [0.92, 2.00] |
| curved, ground-truth boxes | 8.05 [0.45, 23.10] |
| rotated, ground-truth boxes | 0.83 [0.26, 2.19] |

GlyphSynth validation lines through the frozen recognizer give a CER of <!-- derived: glyphsynth_all --> 0.71
[0.63, 0.79] %, by line (GW-REC-041,
[`reports/error_cases/phase15/glyphsynth.md`](reports/error_cases/phase15/glyphsynth.md)). This is a
synthetic set and says little about real pages.

### Efficiency

<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md -->
Measured on the laptop under [hardware](#hardware), on HierText validation, 10 warm-up and 100
timed items, one process per run, on mains power. CI by line or page, descriptive. The page time
starts from the decoded image.

Source: GW-EFF-007 [`reports/efficiency/phase16/cost_table.md`](reports/efficiency/phase16/cost_table.md).

<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/cost_table.md#lines PyTorch\, CPU\, 10 threads=PyTorch\, CPU\, 10 threads|fp32 ONNX\, CPU\, 10 threads=fp32 ONNX\, CPU\, 10 threads|PyTorch\, GPU\, batch 1=PyTorch\, GPU\, batch 1 -->
| Build, lines | Mean ms per item | p95 ms | Lines/s | Peak RAM MiB |
|---|---|---|---|---|
| PyTorch, CPU, 10 threads | 33.90 [29.92, 38.13] | 77.60 | 29.5 | 825.7 |
| fp32 ONNX, CPU, 10 threads | 92.25 [70.69, 116.28] | 304.39 | 10.8 | 1,028.0 |
| PyTorch, GPU, batch 1 | 12.22 [11.24, 13.33] | 19.01 | 81.8 | 1,180.6 |

<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages PyTorch\, CPU\, 10 threads=PyTorch\, CPU\, 10 threads|fp32 ONNX\, CPU\, 10 threads=fp32 ONNX\, CPU\, 10 threads|PyTorch\, GPU=PyTorch\, GPU -->
| Build, pages | Mean ms | p95 ms | Peak RAM MiB | Lines per page |
|---|---|---|---|---|
| PyTorch, CPU, 10 threads | 2,566 [2,327, 2,829] | 5,127 | 1,641.8 | 40.9 |
| fp32 ONNX, CPU, 10 threads | 8,574 [7,516, 9,766] | 19,011 | 6,529.6 | 40.9 |
| PyTorch, GPU | 1,049 [825, 1,300] | 3,288 | 2,450.0 | 40.9 |

On the GPU, the peak memory allocated by PyTorch on pages was <!-- derived: gpu_vram --> 432.4 MiB.
On the CPU with PyTorch, the recognizer's network took <!-- derived: rec_ms --> 2,148.2 ms of the mean
page time of <!-- derived: page_ms --> 2,565.7 ms.

ADR-0011's targets were set for this laptop's CPU and are recorded as missed for v1. Only peak RAM
of the PyTorch build is within its target. The targets were not restated
([ADR-0011](docs/decisions/ADR-0011-cpu-latency-target.md), owner decision of 2026-11-18).

Source: GW-EFF-007 [`reports/efficiency/phase16/README.md`](reports/efficiency/phase16/README.md).

<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/README.md#adr-0011s-targets v1 PyTorch\, CPU\, 10 threads=v1 PyTorch\, CPU\, 10 threads|v1 fp32 ONNX\, CPU\, 10 threads=v1 fp32 ONNX\, CPU\, 10 threads -->
| Build | line mean (≤ 25) | line p95 (≤ 70) | page mean (≤ 1,500) | page p95 (≤ 2,700) | peak RAM on pages (< 1,836) |
|---|---|---|---|---|---|
| v1 PyTorch, CPU, 10 threads | 33.9, over | 77.6, over | 2,565.7, over | 5,127.1, over | 1,641.8, within |
| v1 fp32 ONNX, CPU, 10 threads | 92.2, over | 304.4, over | 8,573.7, over | 19,011.5, over | 6,529.6, over |

Baselines were profiled in Phase 5 and v1 in Phase 16, under the same protocol on HierText
validation. These are different sittings, so no difference is stated; the rows are for orientation
only. One system's recorded page time is not the same from sitting to sitting. In Phase 5 PP-OCRv5
mobile took <!-- derived: v5m_p5 --> 3,408.2 ms, in the Phase 14 sitting <!-- derived: v5m_p14 --> 2,954.8
ms and in the Phase 16 sitting <!-- derived: v5m_p16 --> 3,086.0 ms. GLYPHARIS on PyTorch took <!-- derived: gl_p14 --> 2,446.7
ms in Phase 14 and <!-- derived: page_ms --> 2,565.7 ms in Phase 16.

Sources for these times: Phase 5, the table below; Phase 14, GW-E2E-009
[`reports/external/README.md`](reports/external/README.md); Phase 16, GW-EFF-010
[`verdicts.md`](reports/efficiency/phase16/verdicts.md) for PP-OCRv5 mobile and GW-EFF-007
[`cost_table.md`](reports/efficiency/phase16/cost_table.md) for GLYPHARIS.

Source: GW-EFF-007 and GW-BASE-001 to 007,
[`reports/benchmark_results/efficiency_comparison.md`](reports/benchmark_results/efficiency_comparison.md).

<!-- trace-table: GW-EFF-007 reports/benchmark_results/efficiency_comparison.md#pages GLYPHARIS v1 PyTorch\, CPU\, 10 threads=GLYPHARIS v1 PyTorch\, CPU\, 10 threads -->
<!-- trace-table: GW-BASE-001 reports/benchmark_results/efficiency_comparison.md#pages Tesseract 5=Tesseract 5 -->
<!-- trace-table: GW-BASE-002 reports/benchmark_results/efficiency_comparison.md#pages PP-OCRv5 mobile=PP-OCRv5 mobile -->
<!-- trace-table: GW-BASE-005 reports/benchmark_results/efficiency_comparison.md#pages PP-OCRv6 tiny=PP-OCRv6 tiny -->
<!-- trace-table: GW-BASE-006 reports/benchmark_results/efficiency_comparison.md#pages PP-OCRv6 small=PP-OCRv6 small -->
<!-- trace-table: GW-BASE-007 reports/benchmark_results/efficiency_comparison.md#pages PP-OCRv6 medium=PP-OCRv6 medium -->
| System, pages | Mean ms | p95 ms | Peak RAM MiB | Model MiB | Sitting |
|---|---|---|---|---|---|
| GLYPHARIS v1 PyTorch, CPU, 10 threads | 2,565.7 | 5,127.1 | 1,641.8 | 42.75 | Phase 16 |
| Tesseract 5 | 430.8 | 807.1 | 281.3 | 3.92 | Phase 5 |
| PP-OCRv5 mobile | 3,408.2 | 5,802.8 | 1,959.6 | 25.41 | Phase 5 |
| PP-OCRv6 tiny | 1,726.6 | 2,729.0 | 1,836.0 | 19.31 | Phase 5 |
| PP-OCRv6 small | 3,714.2 | 6,308.3 | 2,051.0 | 43.10 | Phase 5 |
| PP-OCRv6 medium | 32,528.5 | 56,305.4 | 3,015.5 | 145.74 | Phase 5 |

## Limitations

Sources for the counts below: GW-E2E-007 [`end_to_end.md`](reports/external/end_to_end.md), GW-E2E-010
[`oracle_share.md`](reports/error_cases/phase15/oracle_share.md) and GW-EFF-007
[`cost_table.md`](reports/efficiency/phase16/cost_table.md).

<!-- trace: GW-E2E-009 reports/external/README.md -->
<!-- trace: GW-E2E-007 reports/external/end_to_end.md#detection -->
<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages -->
- One frozen recognizer seed and one detector seed. Three recognizer seeds were trained and their
  validation CER was close (see [selection evidence](#selection-evidence)); the detector has none
  beyond GW-DET-001.
- Test results come from the GPU strict-fp32 evaluation; equality with the bundle is shown on
  validation only. Phase 22 checked reproduction at the 0.05-point tolerance (GW-E2E-018,
  [`reports/final_validation/v1_reproduction.md`](reports/final_validation/v1_reproduction.md)):
  every prediction was the same, and the check found 16 failures among the 1,381 point values
  checked, all of them official OmniDocBench scores, through the evaluator's per-page time limit. The owner accepted this as a
  disclosed limitation.
- Latency was measured on HierText validation pages, on one laptop, with a mean of <!-- derived: lines_per_page --> 40.9
  lines per page. Page latency on other document pages was not measured.
- GDC has <!-- derived: n_gdc_docs --> 21 source documents and <!-- derived: n_gdc_pages --> 106
  pages, GDC-scan one document, FUNSD <!-- derived: n_funsd_pages --> 50 forms. Intervals for those
  sets are wide or conditional, and GDC's capture conditions are confounded with its documents.
- No set of old scans was evaluated: GDC's old-scan target is unmet
  ([ADR-0013](docs/decisions/ADR-0013-gdc-scan-role.md)).
- FUNSD and OmniDocBench include text v1 does not try to read, such as handwritten fill-ins on forms
  and formulas in document blocks. SROIE was skipped.
- The pixel limit `max_pixels` is an unmeasured default.
- A float image file holding NaN values is read as a blank page; an array with such values is
  refused (GW-E2E-012, known issue in [`docs/PHASE_STATUS.md`](docs/PHASE_STATUS.md)).
- The confidence is not a probability; see [confidence calibration](#confidence-calibration).

Evidence that the bundle behaves as the frozen model, on validation only:

<!-- trace: GW-E2E-011 reports/package/GW-E2E-011.json -->
- GW-E2E-011 read the validation sample of <!-- derived: package_pages --> 400 pages through the
  package and the v1 bundle on the PyTorch backend. The page CER difference from the frozen pipeline was 0.0, and 16,835 of 16,835
  lines had the same text and box
  ([`reports/package/GW-E2E-011.json`](reports/package/GW-E2E-011.json)).
- GW-EFF-012 ([`torch_bundle_check.json`](reports/export/GW-EFF-012/torch_bundle_check.json)) checks
  the bundle's PyTorch files, and GW-EFF-006
  ([`reference_check.json`](reports/efficiency/phase16/reference_check.json)) the fp32 ONNX files,
  against the PyTorch references. Both passed their tolerances.
- GW-E2E-012 ([`reports/hardening/GW-E2E-012.json`](reports/hardening/GW-E2E-012.json)) passed: bad
  files, wrong inputs, oversized images, damaged bundles and fuzzed inputs raised only the package's
  own errors on both backends.

## Known failure modes

### Pipeline

Bag-of-lines errors on a page split into characters of ground-truth lines that no prediction matched
(missed), errors on matched lines (matched), and characters of predictions that matched no line
(spurious). Shares in %, CI by page.

Source: GW-E2E-010 [`reports/error_cases/phase15/pipeline.md`](reports/error_cases/phase15/pipeline.md).

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md HierText test;missed=L17,Share % [page CI]|HierText test;matched=L18,Share % [page CI]|HierText test;spurious=L19,Share % [page CI]|GDC;missed=L85,Share % [page CI]|GDC;matched=L86,Share % [page CI]|GDC;spurious=L87,Share % [page CI]|FUNSD;missed=L264,Share % [page CI]|FUNSD;matched=L265,Share % [page CI]|FUNSD;spurious=L266,Share % [page CI] -->
| Set | Part of the page error | Share % |
|---|---|---|
| HierText test | missed | 49.45 [47.47, 51.56] |
| HierText test | matched | 25.79 [23.51, 28.00] |
| HierText test | spurious | 24.77 [23.24, 26.38] |
| GDC | missed | 45.32 [36.14, 52.80] |
| GDC | matched | 27.55 [15.80, 42.07] |
| GDC | spurious | 27.13 [20.89, 33.57] |
| FUNSD | missed | 40.22 [36.27, 42.88] |
| FUNSD | matched | 19.39 [14.64, 27.08] |
| FUNSD | spurious | 40.39 [36.32, 43.00] |

Where detection fails. Recall of ground-truth lines by line height in pixels and by rotation in
degrees, strict track, CI by page, from the same report. Bins are written as in the report.

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md HierText test;< 9.35=L61|HierText test;≥ 15=L73|GDC;< 9.35=L155|GDC;[9.35\, 13.52)=L156 -->
| Set | Slice | Bin | Lines | Pages | Matched | Recall % [cited CI] |
|---|---|---|---|---|---|---|
| HierText test | line height | `< 9.35` | 16,712 | 1099 | 9,680 | 57.92 [54.64, 61.17] |
| HierText test | rotation | `≥ 15` | 10,256 | 834 | 5,080 | 49.53 [45.80, 53.19] |
| GDC | line height | `< 9.35` | 113 | 18 | 0 | 0.00 [0.00, 0.00] |
| GDC | line height | `[9.35, 13.52)` | 471 | 23 | 174 | 36.94 [10.52, 45.00] |

Matched lines with lower IoU have higher CER. These are marginal slices, confounded with line height
and rotation, so the bins do not isolate the effect of the box. CER of matched lines by the IoU of
the match, strict track, CI by page, from the same report.

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md HierText test;[0.5\, 0.7)=L47|HierText test;≥ 0.85=L49|GDC;[0.5\, 0.7)=L139|GDC;≥ 0.85=L141 -->
| Set | Matched IoU | Lines | CER % [cited CI] |
|---|---|---|---|
| HierText test | `[0.5, 0.7)` | 7,010 | 29.24 [27.83, 30.64] |
| HierText test | `≥ 0.85` | 24,317 | 3.63 [2.99, 4.49] |
| GDC | `[0.5, 0.7)` | 624 | 29.36 [20.83, 38.02] |
| GDC | `≥ 0.85` | 3,889 | 4.31 [1.10, 8.76] |

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#reading-order -->
Reading order: on GDC, 1,010 of 6,935 consecutive matched ground-truth lines come out in the
opposite order, 14.56 [9.93, 20.36] % by page.

With ground-truth boxes the recognizer makes <!-- derived: oracle_all --> 18.66 [11.55, 28.73] % as
many errors on GDC as the whole pipeline, and <!-- derived: oracle_clean --> 32.70 [16.64, 51.91] %
on the clean slice (by page, GW-E2E-010,
[`oracle_share.md`](reports/error_cases/phase15/oracle_share.md)). H4 expected the recognizer to
account for most of the page error on clean pages. [`research/hypotheses.md`](research/hypotheses.md)
records it as not supported on GDC, post hoc: no rule was registered, so this is a reading of the
numbers and not a test, and the clean slice's interval reaches past one half. The rest of the error
also holds spurious predictions and crop shape, so it is not a pure detector share.

Curved pages and pages photographed sideways have high page CER; the values are under
[capture conditions](#synthetic-corruptions-and-capture-conditions). From ground-truth boxes the
recognizer makes <!-- derived: oracle_rotated --> 1.03 [0.34, 2.93] % as many errors as the whole
pipeline on the sideways pages and <!-- derived: oracle_curved --> 12.47 [0.71, 35.21] % on the
curved ones (by page, same report), so on these pages most of the errors do not occur when the line
boxes are right. The pipeline has no page-orientation step, and finding F24 in
[`research/findings.md`](research/findings.md) ties the sideways failure to that design. No cause was
measured for curved pages.

### Recognizer

Source: GW-REC-040 [`reports/error_cases/phase15/recognition.md`](reports/error_cases/phase15/recognition.md),
ground-truth crops, strict track, CI by page. These are slices of a marginal analysis and are
confounded with each other; no slice is called worse than another from two marginal intervals.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md line height;< 9.35=L47,CER % [cited CI]|rotation;< 2=L92,CER % [cited CI]|rotation;≥ 15=L95,CER % [cited CI] -->
| HierText test crops | Bin | CER % [cited CI] |
|---|---|---|
| line height | `< 9.35` | 11.01 [9.90, 12.22] |
| rotation | `< 2` | 4.46 [4.01, 4.94] |
| rotation | `≥ 15` | 10.82 [9.60, 12.07] |

Share of each set's character errors by the primary category of a wrong line. Each wrong line gets
one category by a fixed precedence; the category says which rule the line met first, not what caused
its errors. Category names are the report's: `rotation_ge_5` covers rotation of at least five
degrees, and `height_q1` the lowest quartile of line height.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md HierText test;rotation_ge_5=L137,Share of errors %|HierText test;height_q1=L138,Share of errors %|GDC;digit=L396,Share of errors %|GDC;contrast_q1=L395,Share of errors %|GDC;out_of_charset=L391,Share of errors % -->
| Set | Primary category | Share of errors % |
|---|---|---|
| HierText test | `rotation_ge_5` | 34.92 |
| HierText test | `height_q1` | 21.30 |
| GDC | `digit` | 49.21 |
| GDC | `contrast_q1` | 20.41 |
| GDC | `out_of_charset` | 13.62 |

On GDC crops, deletions are most of the error: <!-- derived: gdc_del --> 3.19 of the <!-- derived: gdc_cer --> 4.17
CER points. The recognizer deleted <!-- derived: gdc_space --> 4,941 spaces, <!-- derived: gdc_space_rate --> 10.165 % of them, and <!-- derived: gdc_stop --> 4,153 full
stops, <!-- derived: gdc_stop_rate --> 44.176 % of them
([`characters_gdc_test.md`](reports/confusion_analysis/GW-REC-040/characters_gdc_test.md)). Lines in the
`70+` length bucket hold <!-- derived: gdc_long_share --> 61.80 % of GDC's character errors at a
CER of <!-- derived: gdc_long_cer --> 6.37 %
([`recognition.md`](reports/error_cases/phase15/recognition.md), which also gives the deletion
points). Dot leaders appear among the worst lines, but their share of the errors was not measured.

On GlyphSynth validation, lines with touching characters from ink spread have a CER of <!-- derived: glyphsynth_touching --> 1.41
[1.17, 1.72] %, by line (GW-REC-041, [`glyphsynth.md`](reports/error_cases/phase15/glyphsynth.md)).

## Hardware

Training ran on an NVIDIA GeForce RTX 4050 Laptop GPU. All of v1's latency numbers were
measured on an Intel Core i7-14650HX laptop under Windows on mains power, with the Balanced power
plan and power mode "best performance". v1 was not measured on a phone, an ARM device or a second
machine. In Phase 22 the networks of the phone candidate, a new version, were measured on two
phones (GW-EFF-015, see [not v1](#not-v1)).

Source: [`reports/environment/env_report.md`](reports/environment/env_report.md) and GW-EFF-007
[`pages_torch_cpu_10t.json`](reports/efficiency/phase16/profiles/pages_torch_cpu_10t.json) (conditions block).

<!-- trace-table: reports/environment/env_report.md reports/environment/env_report.md RAM, GiB=RAM,Value|GPU memory, MiB=VRAM,Value -->
<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/profiles/pages_torch_cpu_10t.json#conditions Physical CPU cores=physical_cores|Logical CPUs=logical_cpus -->
| Item | Value |
|---|---|
| Physical CPU cores | 16 |
| Logical CPUs | 24 |
| RAM, GiB | 15.71 |
| GPU memory, MiB | 6140 |

## Licence information

The weights are distributed under CC BY-SA 4.0 with attribution to HierText and Open Images
([ADR-0014](docs/decisions/ADR-0014-weights-and-backbone-terms.md)). The detector was initialised
from ImageNet weights, a disclosed exception to the project's commercial-use rule (ADR-0009), and the
weights are not described as cleared for commercial use. The owner decided the positions below on
2026-11-18, and on 2026-11-22 decided to publish the weights under these terms without the qualified
advice ADR-0014 had asked for; this is the project's chosen handling, not a legal conclusion.

| Part | Position |
|---|---|
| Weights | CC BY-SA 4.0 with attribution to HierText and Open Images, subject to the detector exception below; distributed as the asset `glypharis-v1.tar.gz` of the GitHub release `v1.0`, which carries `LICENSE-WEIGHTS` and `NOTICE.md` |
| Detector initialised from ImageNet-1K weights | GW-DET-001 stays v1's detector as a disclosed exception to ADR-0009's commercial-use rule: its backbone was initialised from ImageNet-1K weights, and the ImageNet terms do not clearly allow commercial use |
| Code | Apache-2.0 for the code and the project's own text ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)), with the exceptions that `NOTICE` lists, among them the HierText-derived tables in this card; the code is published on PyPI as `glypharis` 1.0.0, and the licence does not cover the weights, which are distributed separately |
| Phone candidate | not released; v1.0 ships no phone bundle or asset |

Terms of the data and weights that influenced the model:

| Source | Terms | Role |
|---|---|---|
| HierText dataset | CC BY-SA 4.0 | training and evaluation |
| Images in HierText, from Open Images | listed as CC BY 2.0 | training and evaluation |
| MobileNetV3-Large ImageNet-1K weights (torchvision) | code BSD-3-Clause; the weights carry ImageNet-1K terms | detector initialisation only |

HierText-derived tables in this card are shared alike under CC BY-SA 4.0, with the attribution given
under [how to read the numbers](#how-to-read-the-numbers).

The evaluation-only sets never entered training, augmentation or tuning: FUNSD (non-commercial
research and education), OmniDocBench (research only), IIIT5K (no licence stated), GDC (print-and-capture
pages from openly licensed and public-domain sources) and GDC-scan (private, numbers only). Details
and attributions are in [`DATASETS.md`](DATASETS.md).

## Citation information

Cite GLYPHARIS with [`CITATION.cff`](CITATION.cff) (DOI
[10.5281/zenodo.24433864](https://doi.org/10.5281/zenodo.24433864)).

The methods and data behind the model are cited with these keys of
[`research/paper/references.bib`](research/paper/references.bib): `graves2006ctc` (CTC),
`shi2017crnn` (the CRNN baseline), `du2022svtr` (SVTR), `liao2020dbnet` (DBNet),
`howard2019mobilenetv3` (MobileNetV3), `russakovsky2015imagenet` (ImageNet),
`long2022hiertext` (HierText), `kuznetsova2020openimages` (Open Images), `jaume2019funsd` (FUNSD),
`mishra2012iiit5k` (IIIT5K), `ouyang2025omnidocbench` (OmniDocBench), `guo2017calibration`
(temperature scaling), `hendrycks2019benchmarking` (corruption benchmarks), `efron1994bootstrap`
(bootstrap), `smith2007tesseract` (Tesseract), `du2020ppocr` and `cui2025paddleocr` (PaddleOCR).

## Final configuration record

Configuration frozen and recorded. Phase 22 re-ran it: the predictions equal Phase 14's in every
test file, and the strict verdict failed only on official OmniDocBench scores, through the
evaluator's time limit; the owner accepted this as a disclosed limitation
(`reports/final_validation/v1_reproduction.md`). The record is
[`configs/final/glypharis_v1.yaml`](configs/final/glypharis_v1.yaml), and this table says where each
part of it is described.

| Field | Value or location |
|---|---|
| Model architecture | [Architecture](#architecture); `configs/rec/gw-rec-034.yaml` chain and `configs/det/gw-det-001.yaml` |
| Dataset versions | Recognizer packs `b8cfff844ed2a3be027e28ff821c8e50f793756c1840624153935cb04a51a620` (train) and `7db3ec6782eaf919da95194629b35574fbedd4c3361f4b8579400782d16f4f2a` (validation); detector pages `6f5f3871a557771e03ad57a774b5bc45e92dc8be52313700837973f5783562a8` and `0655c5f795a644666777a072714d039d48231878b7707889b1e770703ddfd180`; `data/manifests/charset.json`, version 1 |
| Preprocessing | Label normalisation and line crops in [training datasets](#training-datasets); recognizer input as in [architecture](#architecture) |
| Augmentation | Recognizer: preset `full` with the rotation in [the recipe](#recognizer-training-recipe). Detector: crops, scale and rotation in [the recipe](#detector-training-recipe) |
| Hyperparameters | [Recognizer recipe](#recognizer-training-recipe), [detector recipe](#detector-training-recipe) |
| Checkpoint | Recognizer `c7e745403d01bc23c66e570e3e9596755c7d8fe913daad5dc34f8392c8681a4d`, detector `016b2b69d257d9165afd9aa4a7ef5ce36cdf3ad5d2c5926aa6b1b60c43705cd5`. Checkpoints are not in the repository |
| Git commit | Recognizer training `d75e3591a416383859e180d0429df3c8a6b41b96`, detector training `0a29dddbdfa9bf7c4507cef2c668f1601f697a97`, evaluation tag `phase14-eval` |
| Hardware | [Hardware](#hardware) |
| Software versions | Python 3.11.9, PyTorch 2.14.1+cu126, torchvision 0.29.1+cu126, OpenCV 5.0.0, Shapely 2.2.0, GEOS 3.14.1, ONNX Runtime 1.31.1. Sources: `configs/final/glypharis_v1.yaml` (PyTorch, torchvision, OpenCV, Shapely, GEOS), `reports/environment/env_report.md` (Python), `requirements/lock.txt` (ONNX Runtime) |
| Random seed | Recognizer seed 1337; detector seed 1337 from `configs/base.yaml` |

## Not v1

A new version, not v1. Phase 22 evaluated it once per test set, and it met H11's six non-inferiority
tests against v1 (GW-E2E-021); its results sit beside v1's and replace none of them. Its figures
must not be read as v1's. It is not released: v1.0 ships no phone bundle or asset.

The phone candidate is the all-ONNX build with fused attention, reading one line per recognizer run
(`fused-single-line`, GW-EFF-013,
[`selection.md`](reports/final_validation/candidate/GW-EFF-013/selection.md)): v1's weights, with
each global mixer of the recognizer exported as one attention operator. It is frozen as bundle
format `glypharis-phone-bundle-1` with manifest SHA-256
`97a4116884f93d0d0293d0910e37dee4dbd0aacfb249c3f2c3819e289ee06e2e`, and the bundle passed Phase 17's
tolerances (GW-EFF-014,
[`equivalence.json`](reports/final_validation/candidate/GW-EFF-014/equivalence.json)). Its
validation gate (GW-REC-044, GW-E2E-019,
[`validation_gate.json`](reports/final_validation/candidate/validation_gate.json)) was met before
any test set was opened.

H11 ([`research/hypotheses.md`](research/hypotheses.md)) was met on all six tests. The values below
are fractions, as the report prints them; strict track. On GDC the three tests were also met with
source documents as the unit. H11 is a non-inferiority test: equal values at this precision do not
show that the candidate reads every test line as v1 does.

Source: GW-E2E-021 [`reports/final_validation/candidate/h11.md`](reports/final_validation/candidate/h11.md).

<!-- trace-table: GW-E2E-021 reports/final_validation/candidate/h11.md HierText test;line CER=hiertext_test;line_cer|HierText test;page CER, bag of lines=hiertext_test;page_cer|HierText test;E2E F1=hiertext_test;e2e_f1|GDC;line CER=gdc_test;line_cer|GDC;page CER, bag of lines=gdc_test;page_cer|GDC;E2E F1=gdc_test;e2e_f1 -->
| Set | Measure | Units | Candidate | v1 | Verdict |
|---|---|---|---|---|---|
| HierText test | line CER | 67,574 | 0.0603 | 0.0603 | met |
| HierText test | page CER, bag of lines | 1,634 | 0.2516 | 0.2516 | met |
| HierText test | E2E F1 | 1,634 | 0.6018 | 0.6018 | met |
| GDC | line CER | 8,372 | 0.0417 | 0.0417 | met |
| GDC | page CER, bag of lines | 106 | 0.2235 | 0.2235 | met |
| GDC | E2E F1 | 106 | 0.6936 | 0.6936 | met |

The candidate's other test figures are descriptive and are in
[`reports/final_validation/candidate/results.md`](reports/final_validation/candidate/results.md).
Its official OmniDocBench scores are not comparable with v1's, because the evaluator's time limit
sent more of the candidate's pages to its fallback scoring than of v1's in Phase 14; they are not
compared here.

Phone measurement, GW-EFF-015
([`phone_sm-m526b.md`](reports/final_validation/GW-EFF-015/phone_sm-m526b.md),
[`phone_moto-g35-5g.md`](reports/final_validation/GW-EFF-015/phone_moto-g35-5g.md)): network time
and memory only. Decoding, post-processing, cropping and reading order were not ported, so the time
of a full page on a phone was not measured. Each run timed 100 validation pages with 4 intra-op
threads, and each arm ran twice; times are the mean network time per page over both runs, and
memory is the app's peak `VmHWM`. The Samsung Galaxy M52 5G (SM-M526B) was the first phone and decided the
selection; the Moto G35 5G is a second named device and changes no decision. On both phones the
judged arms' logits were within 1e-3 of the laptop's and their greedy text was identical. v1 itself
(PyTorch) was not measured on a phone. No phone target is set, and ADR-0011's laptop CPU targets
stay recorded as missed for v1.

<!-- trace-table: GW-EFF-015 reports/final_validation/GW-EFF-015/phone_sm-m526b.md SM-M526B;fused-single-line=L9|SM-M526B;fused (not selected)=L7|SM-M526B;fused-int8 (not selected)=L8 -->
<!-- trace-table: GW-EFF-015 reports/final_validation/GW-EFF-015/phone_moto-g35-5g.md Moto G35 5G;fused-single-line=L7 -->
| Phone | Arm | Mean ms, both runs | p95 ms, both runs | Peak VmHWM MiB |
|---|---|---|---|---|
| SM-M526B | fused-single-line | 8,111.9 | 18,298.9 | 919.9 |
| SM-M526B | fused (not selected) | 12,824.4 | 25,756.3 | 1,645.3 |
| SM-M526B | fused-int8 (not selected) | 8,522.2 | 17,568.2 | 1,814.9 |
| Moto G35 5G | fused-single-line | 10,987.5 | 23,283.8 | 409.0 |

The figures below are Phase 16's, on validation data and on the laptop, for the `fused` build, from
which the candidate differs in reading one line per recognizer run. On these samples its CER equals
v1's to two decimals, and its pages and lines take less time than v1's ONNX bundle in the same
sittings. No difference against v1 on PyTorch is stated, because the report has no paired interval
for it.

<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md -->
Source: GW-EFF-009 [`reports/efficiency/phase16/pareto.md`](reports/efficiency/phase16/pareto.md).
Validation only: page CER and E2E F1 on <!-- derived: val_page_sample_n --> 400 pages (CI by page),
page time on <!-- derived: pareto_profile_pages --> 110 profile pages, line CER on <!-- derived: val_line_sample_n --> 5,000
lines (CI by line), line time on <!-- derived: pareto_time_lines --> 100 lines. Latency is fp32 ONNX
at 10 threads unless the row says PyTorch. The differences are against the v1 ONNX bundle in the
same sittings.

<!-- trace-table: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages v1 PyTorch=v1-torch|v1 fp32 ONNX=v1-onnx|Fused attention (not v1)=fused-attention|INT8 recognizer (not v1)=int8-recognizer -->
| Pages | CER % | E2E F1 % | Mean ms | p95 ms | Peak RAM MiB | Δ CER (points) | Δ ms |
|---|---|---|---|---|---|---|---|
| v1 PyTorch | 24.32 [20.87, 28.15] | 60.50 [56.81, 63.96] | 2,637.3 [2,395.4, 2,909.1] | 5,327.7 | 1,665.0 | -0.00 [-0.00, +0.00] | -7,102.1 [-8,195.7, -6,115.5] |
| v1 fp32 ONNX | 24.32 [20.87, 28.15] | 60.50 [56.81, 63.96] | 9,739.4 [8,510.8, 11,113.1] | 22,474.1 | 6,525.5 | n/a | n/a |
| Fused attention (not v1) | 24.32 [20.87, 28.15] | 60.50 [56.81, 63.96] | 4,020.8 [3,604.2, 4,479.8] | 8,267.1 | 2,393.0 | +0.00 [+0.00, +0.00] | -5,718.6 [-6,630.9, -4,883.7] |
| INT8 recognizer (not v1) | 24.32 [20.87, 28.14] | 60.46 [56.71, 63.97] | 6,890.9 [6,010.1, 7,872.0] | 16,071.7 | 6,896.5 | +0.00 [-0.02, +0.02] | -2,848.5 [-3,233.6, -2,505.1] |

<!-- trace-table: GW-EFF-009 reports/efficiency/phase16/pareto.md#lines v1 PyTorch=v1-torch|v1 fp32 ONNX=v1-onnx|Fused attention (not v1)=fused-attention|INT8 recognizer (not v1)=int8-recognizer -->
| Lines | CER % | Mean ms | p95 ms | Peak RAM MiB | Δ CER (points) | Δ ms |
|---|---|---|---|---|---|---|
| v1 PyTorch | 6.22 [5.73, 6.71] | 35.55 [31.32, 40.10] | 80.82 | 841.3 | +0.00 [+0.00, +0.00] | -60.04 [-80.33, -42.17] |
| v1 fp32 ONNX | 6.22 [5.73, 6.71] | 95.59 [73.55, 120.41] | 340.04 | 1,075.0 | n/a | n/a |
| Fused attention (not v1) | 6.22 [5.73, 6.71] | 39.53 [32.12, 47.52] | 112.10 | 887.4 | +0.00 [+0.00, +0.00] | -56.06 [-72.68, -41.23] |
| INT8 recognizer (not v1) | 6.21 [5.73, 6.69] | 63.35 [48.10, 80.25] | 220.31 | 1,097.4 | -0.01 [-0.04, +0.02] | -32.24 [-39.97, -24.87] |

The fused build is within none of ADR-0011's page targets on this laptop. Combining it with the INT8
recognizer was not measured in Phase 16. Phase 22 measured it (`fused-int8`, GW-EFF-013 and
GW-EFF-015): on the first phone it met the speed condition but used more memory than `fused`, so it
was not selected. The INT8 recognizer is also a new version; its confirmatory
test H7′, which limits its latency and its CER cost against the fp32 recognizer, holds
([`reports/efficiency/phase16/int8.md`](reports/efficiency/phase16/int8.md)).
