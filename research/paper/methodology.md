# Methodology

This section describes GLYPHARIS v1 as it was built and frozen: the data it was trained and
evaluated on, the two networks, the steps that join them into a page reader, the confidence
calibration, the metrics, and the record of the final configuration. Each subsection ends with the
code and the configuration that implement it. Results are in `results.md`, and the way the
experiments were organised is in `experiments.md`.

Three conventions hold throughout. The strict text track is the default, and the normalised track
is named where it is used. Every interval is a 95% percentile bootstrap interval, and the text says
whether lines, pages or documents were resampled. Every number comes from a generated report, a
manifest or a configuration file, and carries its experiment ID where a run produced it.

Final validation (Phase 22) has been run; its record is `docs/FINAL_VALIDATION.md`. Configuration
frozen and recorded. Phase 22 re-ran it: the predictions equal Phase 14's in every test file, and the
strict verdict failed only on official OmniDocBench scores, through the evaluator's time limit; the
owner accepted this as a disclosed limitation (`reports/final_validation/v1_reproduction.md`). No
weights are distributed; release is Phase 23.

HierText is released under CC BY-SA 4.0; its images come from Open Images, which lists them as
CC BY 2.0 (`long2022hiertext`, `kuznetsova2020openimages`). Tables derived from HierText are shared
alike.

## 1. System overview

GLYPHARIS is an optical character recognition engine for page images, built to be called by a
document-reading application. It takes one image and returns text lines in reading order, each with
a polygon, a text and a confidence. It knows nothing about PDF files or about what the caller does
with the text.

<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#recognizer/input -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/infer -->
The system has two trained parts. The detector finds text lines in the whole page, resized so that
its long side is 1536 pixels. Each detected line is cut out of the page along its box, straightened
and given to the recognizer, which reads it at a height of 48 pixels and a width of at most 1536
pixels. The lines are then put in reading order from their geometry alone.

<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#recognizer/parameters -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/parameters -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#recognizer/temperature -->
<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#parameters-and-file-sizes -->
"v1" means one specific build: the recognizer GW-REC-034 (7,928,835 parameters), the detector
GW-DET-001 (3,187,209 parameters at inference) at the post-processing and resolution chosen in
Phase 9, and the confidence temperature 2.35 from GW-E2E-005. The parameter total of the two
networks is 11,116,044. The configuration was frozen by GW-REC-037 from validation data only, before
any GLYPHARIS model read a test set. Anything else measured in this project, such as an INT8
recognizer, is a variant and not v1. The candidate for phones is the all-ONNX build with fused
attention, reading one line per recognizer run (`fused-single-line`, GW-EFF-013). A new version, not
v1. Phase 22 evaluated it once per test set, and it met H11's six non-inferiority tests against v1
(GW-E2E-021); its results sit beside v1's and replace none of them.

Phones are the target hardware. Phase 22 measured the phone candidate's networks on two phones
(GW-EFF-015): network time and memory only, without decoding, post-processing, cropping or reading
order; v1 itself was not measured on a phone. Every other latency in this package was measured on
one laptop.

Code: `glypharis/pipeline/ocr.py` (`GlypharisOCR`), `glypharis/pipeline/frozen.py` (builds v1 from
the freeze record). Configuration: `configs/final/glypharis_v1.yaml`.

## 2. Data

### 2.1 Roles and splits

Dataset roles were fixed before training (ADR-0005). Training data gave gradients, the character
set and augmentation statistics. Validation data was used for early stopping, checkpoint selection,
tuning, calibration and any error analysis that fed back into design. Test data has been read only
in Phases 5, 14 and 22, and every read went through a guard that writes a row to
`research/test_access_log.md` before it returns data. Phases 5 and 14 read each set once per frozen
system. Phase 22 re-ran v1 on each set to check its reproduction, and the phone candidate read each
set once, after its validation gate was met (GW-REC-044, GW-E2E-019).

The unit of splitting is the source document, or the source image for scene datasets, so all lines
of a document stay in one split. Official splits were kept. Duplicates across splits were searched
for by exact hash, by perceptual hash and by shared text, and the copy in the lower-priority split
was removed, in the order test, validation, train, so a test image was never the one removed. No
model was trained on any GDC source, and those sources were also kept out of the text corpus of the
synthetic generator.

Code: `glypharis/data/pages.py`, `glypharis/data/audit.py`, `glypharis/data/exclusions.py`,
`glypharis/eval/access.py`. Configuration: `data/manifests/exclusions/`,
`docs/EXPERIMENT_PROTOCOL.md` §1.

### 2.2 HierText, the training source

The recognizer and the detector were trained on HierText (`long2022hiertext`) and nothing else.
Lines came with their polygons and transcriptions. A line was usable for recognizer training only
if it was legible, printed, not vertical, inside the character set and not extremely long.

<!-- trace-table: data/manifests/processed/hiertext.json data/manifests/processed/hiertext.json train=splits/train|validation=splits/validation|test=splits/test -->
| Split | Pages | Images excluded | Lines | Lines for recognizer training | Illegible | Handwritten | Vertical | Out of charset | Too long |
|---|---|---|---|---|---|---|---|---|---|
| train | 8,213 | 68 | 493,464 | 348,935 | 115,764 | 25,122 | 2,039 | 1,457 | 147 |
| validation | 1,715 | 9 | 103,810 | 73,888 | 24,019 | 4,920 | 371 | 590 | 22 |
| test | 1,634 | 0 | 94,564 | 67,277 | 20,827 | 5,738 | 424 | 282 | 15 |

Table 1. HierText after the Phase 3 exclusion list and the Phase 4 preparation
(`data/manifests/processed/hiertext.json`). The last five columns count the reasons a line was not
used for recognizer training. The test row is a count from the manifest; no test image was opened
to write it.

<!-- trace: data/manifests/processed/hiertext.json data/manifests/processed/hiertext.json#crop_settings -->
Line crops were cut once and stored. Each line was warped along its quadrilateral to a strip <!-- derived: crop_height --> 64
pixels high, with a margin of 0.15 of the line height on every side, and saved as JPEG at quality 95.
Lines longer than <!-- derived: crop_max_aspect --> 64 times their height were not cut for the training packs.

<!-- trace: data/manifests/processed/hiertext.json data/manifests/processed/hiertext.json#splits/validation/rec_skip -->
<!-- trace: data/manifests/processed/hiertext.json data/manifests/processed/hiertext.json#splits/validation/lines_rec_ok -->
<!-- trace: data/manifests/processed/eval_crops.json data/manifests/processed/eval_crops.json#hiertext_val/count -->
Two validation line sets exist, and they give different numbers. The training pack holds the 73,888
lines the recognizer can represent, and it was used for checkpoint selection and calibration. The
evaluation pack adds back the 590 out-of-charset lines and the <!-- derived: val_too_long --> 22 lines that were too long, which
gives <!-- derived: val_eval_crops --> 74,500 crops, so that an evaluation set is not shaped by one system's limits.

Code: `glypharis/data/hiertext.py`, `glypharis/data/crops.py`, `glypharis/data/linepack.py`,
`scripts/prepare_hiertext.py`, `scripts/build_eval_crops.py`. Configuration:
`data/manifests/processed/hiertext.json`, `data/manifests/processed/eval_crops.json`.

### 2.3 Text normalisation and character set

Labels went through one normalisation: Unicode NFC, typographic spaces turned into plain spaces,
runs of whitespace collapsed, Latin ligatures written out, and invisible, control and replacement
characters removed. Case, quotation marks and dashes were kept as written, because they look
different in print.

<!-- trace: data/manifests/charset.json data/manifests/charset.json#vocabulary_size -->
<!-- trace: data/manifests/charset.json data/manifests/charset.json#min_count -->
<!-- trace: data/manifests/charset.json data/manifests/charset.json#special_tokens -->
The character set was derived from HierText training labels only. It holds every Latin or Greek
letter, digit, punctuation mark and symbol seen at least 10 times, plus the space. With the CTC
blank at index 0 the recognizer has <!-- derived: charset_classes --> 275 output classes. Lines containing other characters were left
out of recognizer training but stay in the evaluation sets, where such characters count as errors
no model with this character set can avoid. Their share of the reference is reported next to each
result as the charset ceiling.

Code: `glypharis/data/text.py`, `scripts/build_charset.py`. Configuration:
`data/manifests/charset.json`, `reports/charset/charset_report.md`.

### 2.4 GlyphSynth

<!-- trace: data/manifests/processed/glyphsynth.json data/manifests/processed/glyphsynth.json#splits/train -->
<!-- trace: data/manifests/processed/glyphsynth.json data/manifests/processed/glyphsynth.json#splits/val/count -->
GlyphSynth is a generator of document-style text lines written for this project. A sample is fully
determined by a seed and an index. It takes text from public-domain books (PG-19,
`rae2020compressive`) and from generated modern snippets, renders it in fonts from a pinned
collection, and adds capture effects such as paper texture, uneven light, blur and noise. The
training pack has <!-- derived: synth_train_lines --> 100,000 lines from 85 font families. The validation pack has 10,000 lines from
other font families and other source texts, so it measures reading of unseen typefaces and text.

GlyphSynth was used in the Phase 12 study, in the Phase 13 ladder row that tested it, as half of the
GLYPH-C corruption suite, and as a validation set. No delivered model was trained on it.

Code: `glypharis/data/synth.py`, `scripts/build_glyphsynth.py`,
`scripts/build_glyphsynth_variants.py`. Configuration: `data/manifests/processed/glyphsynth.json`,
`data/manifests/glyphsynth_fonts.json`.

### 2.5 GLYPH-C

<!-- trace: data/manifests/processed/glyph_c.json data/manifests/processed/glyph_c.json#per_source -->
<!-- trace: data/manifests/processed/glyph_c.json data/manifests/processed/glyph_c.json#samples -->
<!-- trace: data/manifests/processed/glyph_c.json data/manifests/processed/glyph_c.json#crops -->
GLYPH-C is the project's frozen corruption suite, built from validation data in the manner of
`hendrycks2019benchmarking`. It takes 3,000 lines from HierText validation and 3,000 from GlyphSynth
validation, <!-- derived: glyph_c_samples --> 6,000 source lines in all. Each is stored clean and under seven corruptions (Gaussian
blur, Gaussian noise, JPEG, low resolution, contrast, rotation, perspective) at five levels, which
gives 216,000 crops. The mean corruption error (mCE) is the unweighted mean of the micro CER over
the 35 corruption-level cells.

<!-- trace: GW-REC-034 reports/robustness/GW-REC-034.json#clean_cer -->
Because half of the suite is synthetic, its clean CER is not comparable with a CER on HierText. The
frozen recognizer's clean CER of 1.93% is a figure for this mixed pack only.

Code: `scripts/build_glyph_c.py`, `scripts/glyph_c_eval.py`, `glypharis/data/augment.py`
(`CORRUPTIONS`). Configuration: `data/manifests/processed/glyph_c.json`.

### 2.6 Evaluation sets

<!-- trace-table: data/manifests/processed/eval_crops.json data/manifests/processed/eval_crops.json HierText validation=hiertext_val/count|HierText test=hiertext_test/count|GDC v1.1=gdc_test/count|GDC-scan v1=gdc_scan_test/count|FUNSD=funsd_test/count -->
| Set | Role | Line crops |
|---|---|---|
| HierText validation | selection, calibration, reference | 74,500 |
| HierText test | internal test | 67,574 |
| GDC v1.1 | external test, printed and photographed documents | 8,372 |
| GDC-scan v1 | external test, case study of one document | 776 |
| FUNSD | external test, forms | 2,647 |

Table 2. Recognition packs. They keep lines with characters outside the character set and very long
lines, and they are cut with the crop settings of the training packs.

<!-- trace: data/manifests/processed/benchmarks.json data/manifests/processed/benchmarks.json#iiit5k_test/count -->
<!-- trace: data/manifests/processed/benchmarks.json data/manifests/processed/benchmarks.json#omnidocbench_en/page_count -->
<!-- trace: data/manifests/processed/benchmarks.json data/manifests/processed/benchmarks.json#omnidocbench_en/block_count -->
<!-- trace: data/manifests/processed/benchmarks.json data/manifests/processed/benchmarks.json#funsd_test/page_count -->
Two other external sets have no line crops of this kind or are read differently. IIIT5K
(`mishra2012iiit5k`) is 3,000 cropped words, scored as published without an added margin.
OmniDocBench (`ouyang2025omnidocbench`, English pages) has 755 pages and 11,369 text blocks; its
blocks are paragraphs, so it is scored at page level only. FUNSD (`jaume2019funsd`) has 50 forms, its
lines are this project's grouping of the dataset's entity words, and handwritten fill-ins count as
text.

<!-- trace: data/manifests/gdc/gdc_v1.json data/manifests/gdc/gdc_v1.json -->
<!-- trace: data/manifests/gdc_scan/gdc_scan_v1.json data/manifests/gdc_scan/gdc_scan_v1.json -->
GDC (Glypharis Document Challenge) was built for this project: born-digital documents were printed
and photographed, and ground truth was projected from the PDF text layer through markers on each
sheet, so no OCR engine produced any of it. Version 1.1 has <!-- derived: gdc_sheets --> 106 sheets and 8,606 ground-truth lines.
GDC-scan v1 has 28 pages and 779 ground-truth lines from one document. It is a case study: its
intervals are conditional on that document, and only its numbers are published.

Code: `glypharis/eval/sets.py`, `glypharis/data/gdc.py`, `scripts/prepare_benchmarks.py`,
`scripts/gdc_align.py`, `scripts/gdc_scan_build.py`. Configuration:
`data/manifests/processed/eval_crops.json`, `data/manifests/processed/benchmarks.json`,
`data/manifests/gdc/gdc_v1.json`, `data/manifests/gdc_scan/gdc_scan_v1.json`.

## 3. Recognizer

### 3.1 Architecture

The recognizer maps a grey line image to per-frame character probabilities and is trained with CTC
(`graves2006ctc`). The encoder follows SVTR (`du2022svtr`), adapted to lines of any width.

<!-- trace: configs/rec/gw-rec-020.yaml configs/rec/gw-rec-020.yaml#model/svtr -->
A stem of two strided convolutions reduces height and width by a factor of 4, so one output frame
covers 4 input columns. Three stages of mixing blocks follow, with embedding widths <!-- derived: svtr_dim_1 --> 96, <!-- derived: svtr_dim_2 --> 192
and <!-- derived: svtr_dim_3 --> 256. The first stage has 3 local blocks, the second 3 local and 3 global, the third 6 global.

<!-- trace: configs/rec/gw-rec-020.yaml configs/rec/gw-rec-020.yaml#model/svtr/local_kernel -->
A local block is a depthwise convolution with a local kernel of 3 by 7.

<!-- trace: configs/rec/gw-rec-020.yaml configs/rec/gw-rec-020.yaml#model/svtr -->
A global block is multi-head self-attention over all tokens of the line, with 6 heads in the second
stage and 8 in the third. Every block ends in a feed-forward layer with an MLP ratio of 4. Between stages a convolution halves
the height and changes the embedding width. The remaining height is averaged away and a linear layer
gives the logits. Padded columns are never attended to, so a line reads the same alone or in a batch.

<!-- trace: GW-REC-034 reports/training/GW-REC-034/report.md#run -->
<!-- trace: data/manifests/charset.json data/manifests/charset.json#vocabulary_size -->
With <!-- derived: charset_classes --> 275 output classes the model has 7,928,835 parameters (30.2 MiB of weights and buffers).
Decoding is greedy CTC, with no language model and no lexicon.

The baseline against which it was developed is a CRNN (`shi2017crnn`), a convolutional stack and a
two-layer bidirectional LSTM trained with the same data pipeline and CTC head.

Code: `glypharis/models/rec/svtr.py`, `glypharis/models/rec/base.py`,
`glypharis/models/rec/crnn.py`. Configuration: `configs/rec/gw-rec-034.yaml`, which inherits
`gw-rec-024.yaml`, `gw-rec-022.yaml`, `gw-rec-020.yaml`, `gw-rec-003.yaml`, `gw-rec-001.yaml` and
`configs/base.yaml`.

### 3.2 Input

<!-- trace: configs/rec/gw-rec-022.yaml configs/rec/gw-rec-022.yaml#data -->
A crop was converted to grey, resized to a height of 48 pixels with its aspect ratio kept, squeezed
only if it would be wider than <!-- derived: rec_max_width --> 1536 pixels, scaled to the range from minus one to one, and padded
on the right. In training, a line too narrow for CTC (fewer frames than its label needs) was padded with
background instead of being stretched. Batches were formed by width buckets, with a cap of 6,144 on
lines times the widest line of the batch.

Code: `glypharis/data/rec_dataset.py` (`WidthBucketBatchSampler`, `collate`),
`glypharis/pipeline/recognizer.py` (`prepare_crop`). Configuration: `configs/rec/gw-rec-022.yaml`.

### 3.3 Training

<!-- trace: configs/rec/gw-rec-001.yaml configs/rec/gw-rec-001.yaml#train/grad_clip_norm -->
<!-- trace: configs/rec/gw-rec-020.yaml configs/rec/gw-rec-020.yaml#train/learning_rate -->
<!-- trace: configs/rec/gw-rec-022.yaml configs/rec/gw-rec-022.yaml#train -->
<!-- trace: configs/rec/gw-rec-001.yaml configs/rec/gw-rec-001.yaml#train/batch_size -->
The optimizer was AdamW (`loshchilov2019adamw`) with learning rate 5e-4 and weight decay 1e-2, under
a OneCycle schedule (`smith2019onecycle`) sized from all optimizer steps of the run. Gradients were
clipped at a global norm of 5.0. Each optimizer step accumulated <!-- derived: rec_accumulation --> 8 micro-batches of <!-- derived: rec_micro_batch --> 8 lines, which
keeps the 64 lines per step of the CRNN control. The forward pass ran under bf16 autocast and the
CTC loss in fp32.

<!-- trace: configs/rec/gw-rec-024.yaml configs/rec/gw-rec-024.yaml#train/epochs -->
<!-- trace: GW-REC-034 reports/training/GW-REC-034/report.md#run -->
<!-- trace: GW-REC-034 reports/training/GW-REC-034/report.md#best-and-final-epoch -->
GW-REC-034 trained on the 348,935 HierText lines for <!-- derived: rec_epochs --> 30 epochs, 169,860 optimizer steps and 20.11
GPU-hours on the laptop's RTX 4050, with seed 1337. It was validated at the end of each epoch on the
73,888 validation lines. The selected checkpoint is the one with the lowest validation CER on the
strict track, and it came from the last epoch (index <!-- derived: rec_best_epoch --> 29). Its validation CER is 6.30% [6.17, 6.44]
(by line) and its WER 19.37%. Early stopping was configured and did not fire, and no diagnostic flag
was raised.

Code: `glypharis/training/rec_trainer.py`, `glypharis/training/loop.py`,
`glypharis/training/optim.py`, `train.py`. Configuration: `configs/rec/gw-rec-034.yaml`; the record
of the run is `configs/final/glypharis_v1.yaml`, section `recognizer`.

### 3.4 Augmentation

Training lines were augmented on the fly with the `full` preset, which applies three families in a
fixed order: geometric (rotation, shear, stretch, perspective, margin jitter), document degradation
(low resolution, ink spread, uneven light, paper texture, salt-and-pepper noise, binarisation) and
photometric or capture effects (brightness and contrast, gamma, inversion, blur, noise, JPEG). Each
transform has its own probability and severity in `PRESETS`. All randomness is a function of the
seed, the epoch and the sample index, so a resumed run reproduces its augmentation.

The final recipe changed one thing in that preset. Rotation went up to 4 degrees either way on half
of the lines, instead of up to 2 degrees on 30% of them. Row L5 of the Phase 13 ladder tested this
change (`experiments.md`, section 5).

Code: `glypharis/data/augment.py`. Configuration: `configs/rec/gw-rec-001.yaml`
(`data.augmentation`), `configs/rec/gw-rec-034.yaml` (`data.augmentation_params`).

## 4. Detector

### 4.1 Architecture and labels

<!-- trace: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#det/model/inner_channels -->
<!-- trace: GW-DET-001 reports/training/GW-DET-001/report.md#run -->
The detector follows DBNet (`liao2020dbnet`). Its backbone is MobileNetV3-Large
(`howard2019mobilenetv3`), tapped at four depths. A feature pyramid neck (`lin2017fpn`) with 96
channels fuses the maps at a quarter of the input resolution, and two heads upsample the result to a
probability map and a threshold map. The threshold head is used only in training, so the model has
3,210,466 parameters in training and 3,187,209 at inference.

The backbone was initialised from ImageNet weights (`russakovsky2015imagenet`), loaded from a
registered file whose hash is recorded, under the policy of ADR-0007. It is the only pretrained
component of v1; the recognizer was trained from scratch. The owner decided on 2026-11-18
([ADR-0014](../../docs/decisions/ADR-0014-weights-and-backbone-terms.md)) that GW-DET-001 stays v1's
detector as a disclosed exception to ADR-0009's commercial-use rule, because the ImageNet terms do not
clearly allow commercial use, and that the intended terms for the v1 weights are CC BY-SA 4.0 with
attribution to HierText and Open Images. Intended terms under ADR-0014, effective only if released.
This is the project's chosen handling, not a legal conclusion.

Targets follow DBNet. Each line polygon is shrunk with shrink ratio 0.4 to give the text kernel, and
a threshold map from 0.3 to 0.7 is drawn around each line. A line whose short side is under a minimum
text size of 6 pixels at the training scale becomes "don't care" for that sample. The loss has three
parts: binary cross-entropy on the probability map with online hard example mining (OHEM) at 3
negatives per positive, a Dice loss on the approximate binary map with amplification k = 50, and an
L1 loss on the threshold map with weight 10. It was computed in fp32.

Code: `glypharis/models/det/dbnet.py`, `glypharis/models/det/backbone.py`,
`glypharis/models/det/loss.py`, `glypharis/data/det_labels.py`. Configuration:
`configs/det/gw-det-001.yaml`; decisions in ADR-0012.

### 4.2 Training

<!-- trace: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#det/data/crop_size -->
<!-- trace: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#det/data/crops_per_page -->
<!-- trace: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#det/data/scale_min -->
<!-- trace: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#det/data/scale_max -->
<!-- trace: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#det/data/rotate_max_deg -->
<!-- trace: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#det/data/text_crop_p -->
The detector was trained on crops of <!-- derived: det_crop_size --> 640 pixels square, 4 per page per epoch, with the scale drawn
between 0.5 and 2.0 and rotation up to <!-- derived: det_rotate --> 10 degrees, followed by photometric augmentation. A crop was
placed around a text line with probability 0.8.

<!-- trace: configs/det/gw-det-001.yaml configs/det/gw-det-001.yaml#train -->
The optimizer was AdamW with learning rate 5e-3 and weight decay 1e-4, under a cosine schedule with
2,000 warm-up steps. The batch size was 8 crops, gradients were clipped at a norm of 10.0, and the
forward pass ran under bf16. The learning rate was fixed from the sanity sequence before the main run.

<!-- trace: GW-DET-001 reports/training/GW-DET-001/report.md#run -->
<!-- trace: GW-DET-001 reports/training/GW-DET-001/report.md#best-and-final-epoch -->
GW-DET-001 ran for 24 epochs, 98,544 optimizer steps and 4.86 GPU-hours. Each epoch was scored on a
fixed validation subset of 400 pages at the two pre-registered operating points. The best checkpoint
by that F1 was selected from epoch 22, together with its operating point. One seed was trained, so
the spread between detector seeds was not measured.

Code: `glypharis/training/det_trainer.py`, `glypharis/training/det_eval.py`,
`glypharis/data/det_dataset.py`, `glypharis/data/det_geometry.py`,
`glypharis/data/det_augment.py`. Configuration: `configs/det/gw-det-001.yaml`, `configs/base.yaml`
(seed).

### 4.3 Post-processing and input size

The probability map is binarised, each connected component becomes a candidate with its mean
probability as score, candidates below the box threshold are dropped, and each remaining kernel is
grown back to a text line and returned as a rotated rectangle.

<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/post/min_component_area -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/post/bin_thresh -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/post/box_thresh -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/post/operating_points -->
These settings did not come from the training run. In Phase 9 the validation pages were split into
two fixed halves. GW-DET-004 scored a grid of post-processing settings on half A and tested the best
one on half B against the settings of record. The adopted setting has a binarisation threshold of <!-- derived: det_bin_thresh --> 0.3,
a box threshold of 0.5, a minimum component area of 1 and the `inverse09` unclip with ratio 0.9.

<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/infer -->
<!-- trace: GW-DET-005 reports/error_cases/detector/GW-DET-005/resolution_A.md -->
GW-DET-005 then chose the long side among the sizes whose estimated page time fitted ADR-0011's page
target at that point, again on half A with a test on half B. It adopted a long side of 1536 in place
of the long side 1280 of the training configuration.

<!-- trace: GW-DET-001 reports/error_cases/detector/phase09_adopted/eval_full.json#headline -->
<!-- trace: GW-DET-001 reports/error_cases/detector/phase09_adopted/analysis.md -->
With both changes, on all <!-- derived: det_val_pages --> 1,715 validation pages (<!-- derived: det_care_lines --> 74,500 care lines) and at the primary IoU
threshold, the detector has F1 <!-- derived: det_f1 --> 0.7779 [0.7621, 0.7919] (by page), precision <!-- derived: det_precision --> 0.8519 [0.8381, 0.8644]
and recall <!-- derived: det_recall --> 0.7157 [0.6953, 0.7338]. The result of each step is in `experiments.md`, section 5.6.

Code: `glypharis/models/det/postprocess.py`, `glypharis/models/det/preprocess.py`,
`glypharis/models/det/detector.py`, `scripts/det_sweep.py`, `scripts/det_choose.py`. Configuration:
`configs/final/glypharis_v1.yaml`, section `detector`.

## 5. From boxes to text

### 5.1 Cropping and reading

Each detected line is cut out of the grey page along its quadrilateral, with the crop code and
settings of the training packs, and read in batches sorted by width. Every line keeps enough padding
before the edge of its batch tensor that it reads the same alone or in any batch.

Code: `glypharis/pipeline/ocr.py`, `glypharis/data/crops.py` (`cut_line`),
`glypharis/pipeline/recognizer.py` (`TextRecognizer`). Configuration:
`configs/final/glypharis_v1.yaml`, section `recognizer`.

### 5.2 Reading order

The order is a geometric heuristic. The page's dominant text angle, the length-weighted median of the
line directions, is removed first. The lines are then split recursively in the manner of the XY-cut
(`nagy1992xycut`): a group is cut at its widest empty band, vertically when the band is a column
gutter and otherwise horizontally. A group with no empty band left is read row by row. There is no
learned layout model and no page-orientation step, so a page photographed sideways is not corrected.

HierText has no reading order, so the method could not be tuned or checked on the training data. It
is measured on GDC, GDC-scan and OmniDocBench, which have one.

Code: `glypharis/pipeline/reading_order.py`. Configuration: none; the method has no learned or
configured parameters beyond the constants in that file.

### 5.3 Confidence

The confidence of a line comes from the per-frame logits divided by a temperature T. It is the
geometric mean of the top-class probability over the frames whose top class is not the CTC blank, and
for a line read as empty it is the mean blank probability. Decoding does not depend on T. The
recognizer's confidence and the detector's score are kept apart in the output.

Code: `glypharis/pipeline/recognizer.py` (`line_confidence`). Configuration:
`configs/final/glypharis_v1.yaml`, `recognizer.temperature`.

## 6. Calibration

<!-- trace: GW-E2E-005 reports/calibration/GW-E2E-005/calibration.md -->
One temperature was fitted for the frozen recognizer (GW-E2E-005). All 73,888 validation lines were
read through the pipeline's recognizer and split by page into the two halves used in Phase 9. A line
counts as right when its text equals the label exactly on the strict track. The temperature was
chosen on half A (35,976 lines) by minimising the binary negative log-likelihood of correctness, and
the expected calibration error (`guo2017calibration`, `naeini2015ece`) was measured on half B (37,912
lines, accuracy <!-- derived: cal_accuracy --> 0.724) with 15 equal-mass bins.

<!-- trace: GW-E2E-005 reports/calibration/GW-E2E-005/calibration.md -->
The fitted temperature is <!-- derived: cal_temperature --> 2.35. On half B the ECE was 0.2140 for the raw confidences and <!-- derived: cal_ece_fitted --> 0.0886 for
the fitted ones, a difference of -0.1254 [-0.1296, -0.1217] (by line). The rule fixed beforehand was
to adopt the temperature if this interval lay below zero, so it was adopted, and it was kept for
every later evaluation whatever those evaluations showed. Whether it transfers to other data was
tested in Phase 14 (H8′; `experiments.md`, section 6).

Code: `glypharis/pipeline/calibration.py`, `scripts/rec_calibrate.py`. Configuration:
`configs/final/glypharis_v1.yaml`, sections `recognizer` (temperature) and `calibration`.

## 7. Metrics and statistics

### 7.1 Text tracks

All text metrics were computed on two tracks. The strict track applies only the label normalisation
of section 2.3, so case, punctuation and symbols count. The normalised track adds NFKC, maps
typographic characters to plain ones and folds case; it exists for systems whose output conventions
differ. IIIT5K's labels are upper case, so its headline uses the customary case-insensitive
alphanumeric track, and its strict row, which charges case against those labels, is reported beside
it. Characters outside the character set are never removed from a reference.

Code: `glypharis/data/text.py` (`normalize_eval`), `glypharis/eval/text_metrics.py`. Configuration:
`docs/EXPERIMENT_PROTOCOL.md` §2.

### 7.2 Recognition

CER is the summed Levenshtein distance (`levenshtein1966`) between predictions and references over
the summed reference length, a micro average over the set, and it can exceed one. WER is the same on
whitespace-separated words. Normalised edit distance and sequence accuracy are reported as well.
Empty references are left out of the CER denominator and counted separately.

Code: `glypharis/eval/text_metrics.py`. Configuration: `docs/EXPERIMENT_PROTOCOL.md` §3.

### 7.3 Detection and end to end

Predicted and reference lines are matched one to one by polygon IoU. The matching first maximises the
number of pairs that pass the threshold and then their summed IoU, and lines that count are matched
before don't-care lines. Precision, recall and F1 are reported at IoU 0.5 (primary) and 0.7.

The page metric that needs no reading order is the bag-of-lines page CER. Predicted lines are matched
to reference lines by box IoU, CER is computed over matched pairs, and unmatched lines count as full
deletions or insertions, so it measures detection and recognition together. Where the ground truth
has a reading order (GDC, GDC-scan, OmniDocBench), the order-aware page CER and WER compare the text
joined in predicted order with the reference page. The E2E line F1 counts a reference line as found
if a prediction overlaps it at IoU 0.5 or more and its line CER is at most 0.1, a threshold fixed in
the protocol. The oracle analysis feeds ground-truth boxes to the recognizer, to separate the error
of the two stages.

Ground-truth lines marked illegible, handwritten or vertical are don't care. From Phase 14,
HierText's own line-level score and OmniDocBench's text-block and reading-order edit distances were
also computed with the official code at pinned commits.

Code: `glypharis/eval/page_metrics.py`, `glypharis/eval/det_matching.py`,
`scripts/official_hiertext.py`, `scripts/official_omnidocbench.py`. Configuration:
`docs/EXPERIMENT_PROTOCOL.md` §3, `data/manifests/evaluators.json`.

### 7.4 Latency and memory

Latency is wall-clock time per item after 10 warm-up items, over 100 timed items, in a profiling pass
separate from the accuracy runs and on validation data. Peak RAM is the peak working set of the
process, plus that of any engine process it starts. CPU latency is the primary deployment number. The
rules for sittings are in `experiments.md`, section 7.

Code: `scripts/rec_profile.py`, `scripts/e2e_profile.py`, `glypharis/efficiency/conditions.py`.
Configuration: `docs/EXPERIMENT_PROTOCOL.md` §3.

### 7.5 Intervals and comparisons

Confidence intervals are percentile bootstrap intervals (`efron1994bootstrap`) at 95% with 1,000
resamples, resampling lines for recognition and pages for detection and end-to-end results. For a
ratio such as CER the numerator and denominator are summed afresh on each resample. Two models are
compared on the same resampled indices, and a difference counts only if its interval excludes zero.
Source-document cluster bootstraps were added for the Phase 14 tests on GDC. Ablation tables report
raw intervals and state how many comparisons were made; no correction is applied.

Code: `glypharis/eval/stats.py`, `glypharis/eval/calib_stats.py`. Configuration:
`docs/EXPERIMENT_PROTOCOL.md` §4.

## 8. Training control and reproducibility

Every run used one configuration file, whose resolved form and hash are saved with the run. The
trainers refuse to start if training and validation sample identifiers overlap. Each run writes its
report automatically: configuration hash, commit, data hashes, best and final epoch, curves, worst
validation lines and any diagnostic flag. The flags cover over- and underfitting, blank and prediction
collapse, gradient explosion or vanishing, non-finite losses and memory errors, and for the detector
the map collapse, threshold-branch and box-size failures listed in the protocol.

Random choices of the data pipeline are functions of the seed, the epoch and the sample index.
Checkpoints are written at optimizer-step boundaries and hold every random state, so an interrupted
run continues on the same schedule. On the CPU a resumed run is bit-identical. On CUDA, CTC backward
has no deterministic kernel, so two runs agree only within the tolerance recorded in the protocol.

Code: `glypharis/diagnostics/__init__.py`, `glypharis/diagnostics/det.py`,
`glypharis/training/checkpoint.py`, `glypharis/training/report.py`,
`glypharis/training/det_report.py`, `glypharis/config/`. Configuration: `configs/base.yaml`,
`docs/EXPERIMENT_PROTOCOL.md` §6 to §8.

## 9. The final configuration

One configuration was identified as final, and `configs/final/glypharis_v1.yaml` was generated from
run summaries, checkpoints and reports by `scripts/freeze_final.py`. Table 3 is its record, field by
field. Configuration frozen and recorded. Phase 22 re-ran it: the predictions equal Phase 14's in
every test file, and the strict verdict failed only on official OmniDocBench scores, through the
evaluator's time limit; the owner accepted this as a disclosed limitation
(`reports/final_validation/v1_reproduction.md`).

<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#recognizer/parameters -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/parameters -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#recognizer/input -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/infer -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#recognizer/temperature -->
<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#detector/post/box_thresh -->
| Field | Recognizer | Detector |
|---|---|---|
| Model architecture | SVTR-style encoder with CTC head, 7,928,835 parameters (`configs/rec/gw-rec-034.yaml`) | DBNet-style on MobileNetV3-Large, 3,187,209 parameters at inference (`configs/det/gw-det-001.yaml`) |
| Dataset versions | `lines_train` `b8cfff84…`, `lines_validation` `7db3ec67…`; charset, normalisation and exclusion list at their first versions | `pages_train` `6f5f3871…`, `pages_validation` `0655c5f7…`; first label version |
| Preprocessing | grey, height 48, width at most 1536, range minus one to one, right padding | long side 1536, padded to a multiple of the network stride, ImageNet normalisation inside the model |
| Augmentation | `full` preset with the rotation override (section 3.4) | crop, scale, rotation and photometric steps (section 4.2) |
| Hyperparameters | sections 3.1 to 3.3; temperature 2.35 | sections 4.1 to 4.3; `inverse09`, box threshold 0.5 |
| Checkpoint | `checkpoints/GW-REC-034/best.pt`, SHA-256 `c7e74540…` (last epoch) | `checkpoints/GW-DET-001/best.pt`, SHA-256 `016b2b69…` (section 4.2) |
| Git commit | `d75e3591` | `0a29dddb` |
| Hardware | NVIDIA GeForce RTX 4050 Laptop GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Software versions | `torch 2.14.1+cu126`, bf16, deterministic mode | `torch 2.14.1+cu126`, `torchvision 0.29.1+cu126`, `opencv 5.0.0`, `shapely 2.2.0`, `geos 3.14.1`, bf16, deterministic mode |
| Random seed | seed 1337 | seed 1337, the default of the base configuration |

Table 3. The record of the final configuration. Hashes are shortened here and given in full in the
configuration file.

The checkpoints are not in the repository, and later phases check them by these hashes. v1's code is
the tree tagged `phase14-eval`, which Phase 14 evaluated. The tag `v1-code` differs from it only in
docstrings, comments and the default bundle name (`reports/export/v1-code/code_diff.json`).

Code: `scripts/freeze_final.py`, `glypharis/pipeline/frozen.py`, `glypharis/efficiency/v1.py`.
Configuration: `configs/final/glypharis_v1.yaml`.
