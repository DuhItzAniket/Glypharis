# GLYPHARIS — Architecture

Status: as built for v1. This file replaces the Phase 0 design draft; §10 lists what changed from it.

v1 is the frozen model recorded in `configs/final/glypharis_v1.yaml` (frozen by GW-REC-037):
recognizer GW-REC-034, detector GW-DET-001, read from the model bundle with manifest SHA-256
`875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037` (GW-EFF-006). The code Phase 14
evaluated is tag `phase14-eval`; tag `v1-code` differs from it only in docstrings, comments and the
default bundle name (`reports/export/v1-code/code_diff.json`).

Final validation (Phase 22) has not been run. No weights are distributed; release is Phase 23.

Settings below come from the freeze record and the configs it names. Measured values carry their
experiment ID and report. Results on the evaluation sets are in [MODEL_CARD.md](../MODEL_CARD.md) and
`research/paper/results.md`, not here. Unless stated otherwise, CER is on the strict track.

## 1. Scope

GLYPHARIS is a standalone OCR engine: image in, structured text out. It knows nothing about PDFs,
chunking, retrieval or question answering. A document-reading application renders a page to an image
and calls the engine. The repository contains no consumer-specific code.

```
        INPUT IMAGE  (path | image file bytes | PIL.Image | numpy array)
                        │
                        ▼
        ┌──────────── load_rgb ──────────────────┐
        │ decode, EXIF orientation, RGB uint8,   │
        │ pixel limit checked before decoding    │
        └────────────────────────────────────────┘
                        │
                        ▼
              GLYPHARIS-Det  (text-line detector)
                 long side 1536, pad to a multiple of 32
                        │  line polygons, quads, detector scores
                        ▼
           cut_line  (perspective warp of each quad from the grey page)
                        │  line crops, 64 px high, 15% margin
                        ▼
              GLYPHARIS-Rec  (SVTR-style encoder + CTC head)
                 height 48, width-sorted batches, greedy decoding
                        │  line text + confidence at temperature 2.35
                        ▼
        reading order  (deskew, recursive XY-cut, rows)
                        │
                        ▼
                 OCRResult (JSON-serialisable)
```

The detector and recognizer are trained, evaluated and versioned independently. They are combined in
`glypharis/pipeline/ocr.py` and shipped together as one bundle (§8). The pipeline has no page
orientation step: pages photographed sideways are not read correctly (F24,
`reports/external/end_to_end.md`).

## 2. Granularity decision: lines, not words

The detector localises text lines and the recognizer reads whole lines
([ADR-0002](decisions/ADR-0002-line-level-recognition.md)).

- Document text is organised in lines. Line-level recognition keeps inter-word spacing and avoids a
  separate word-segmentation step.
- CTC decoding handles long sequences without the drift that autoregressive decoders can show on long
  inputs (literature evidence in `research/approach_survey.md`).
- HierText provides line annotations directly, and GLYPH-C and GlyphSynth are built at line level.
- Word boxes are not produced. A `word_boxes` field could later be derived from CTC alignments without
  changing the training granularity (ROADMAP §5).

Word crops (IIIT5K) are used as an out-of-domain test of the recognizer (RQ6), with the alphanumeric
track as the headline row (EXPERIMENT_PROTOCOL §11, 2026-11-15).

## 3. GLYPHARIS-Det

A DBNet-style differentiable-binarisation detector, built in Phase 8, with post-processing and input
size set in Phase 9. Decisions on labels and operating points are in
[ADR-0012](decisions/ADR-0012-detector-labels-and-operating-points.md).

| Part | As built (v1) | Source |
|---|---|---|
| Backbone | torchvision MobileNetV3-Large, initialised from ImageNet weights (file SHA-256 `8738ca79…`); feature taps at strides 4, 8, 16 and 32. No pretrained OCR weights | `configs/det/gw-det-001.yaml`, `glypharis/models/det/backbone.py`, [ADR-0007](decisions/ADR-0007-pretrained-weights-policy.md) |
| Size | 3,187,209 parameters | GW-EFF-007, `reports/efficiency/phase16/cost_table.md` |
| Input | uint8 RGB. The page is scaled so its long side is 1536 (enlarged as well as shrunk), then padded right and bottom to multiples of 32. ImageNet normalisation is inside the model | `configs/final/glypharis_v1.yaml`, GW-DET-005, `glypharis/models/det/preprocess.py` |
| Neck | Lightweight FPN fused to 1/4 input resolution, 96 inner channels | `configs/det/gw-det-001.yaml`, `glypharis/models/det/dbnet.py` |
| Head | Probability map and threshold map. The threshold head runs only in training. The last transposed convolution of each head runs in fp32 with autocast off | `glypharis/models/det/dbnet.py` |
| Labels | Line polygons shrunk by D = A(1 − r²)/L with r = 0.4, minimum text size 6 px, threshold map 0.3 to 0.7 | `configs/det/gw-det-001.yaml`, ADR-0012 |
| Loss | L_prob (BCE with OHEM 3:1, BCE weight 1) + α·L_bin (Dice) + β·L_thresh (L1), α = 1, β = 10, k = 50 | `configs/det/gw-det-001.yaml` |
| Training | HierText pages, 640×640 crops, 4 crops per page, scale 0.5 to 2.0, rotation up to 10°. Batch 8, AdamW, learning rate 5e-3, weight decay 1e-4, cosine schedule with 2,000 warm-up steps, gradient clip 10, bf16. 24 epochs, 98,544 steps, selected epoch 22. One seed | `configs/det/gw-det-001.yaml`, `configs/final/glypharis_v1.yaml`, `reports/training/GW-DET-001/report.md` |
| Post-processing | Probability map thresholded at 0.3; 8-connected components kept when their mean probability is at least 0.5, area at least 1 pixel, box side at least 3.0, at most 5,000 candidates; `inverse` unclip with ratio 0.9 (`inverse09`); output shape `quad` | `configs/final/glypharis_v1.yaml`, GW-DET-004, `glypharis/models/det/postprocess.py` |

Outputs per line: polygon clipped to the image, quad (TL, TR, BR, BL of its minimum-area rectangle,
with the text direction from the aspect-2 rule in ADR-0012), score, aspect and long-axis angle.

The training run selected its checkpoint at long side 1280 and operating point `inverse10`
(`reports/training/GW-DET-001/report.md`). Phase 9 then changed two things on the same network, each
chosen on validation half A and checked on half B (858 pages, paired page bootstrap, 95% interval):

| Change | F1@0.5 gain on half B | Source |
|---|---|---|
| Post-processing (GW-DET-004) | +0.032 [+0.025, +0.039] | `reports/error_cases/detector/GW-DET-004/pair_B.json` |
| Long side 1536 against 1280 (GW-DET-005) | +0.018 [+0.013, +0.022] | `reports/error_cases/detector/GW-DET-005/pair_B.json` |

At these settings, on all 1,715 HierText validation pages (74,500 care lines), F1@0.5 is
0.7779 [0.7621, 0.7919], precision 0.8519 [0.8381, 0.8644] and recall 0.7157 [0.6953, 0.7338], with 95%
page-bootstrap intervals (`reports/error_cases/detector/phase09_adopted/eval_full.json`). The retrained
GW-DET-007 was not adopted: its F1@0.5 differs by −0.0111 [−0.0146, −0.0075], paired over the same pages
(`reports/error_cases/detector/GW-DET-007/pair.json`).

The output shape is set by `det.post.box_type`. `quad` gives the rotated rectangle around the unclipped
kernel and is what v1 uses. `poly` gives the kernel outline offset by the same distance and is not used
by v1. The comparison planned for it (GW-DET-006) could not decide, because HierText validation holds no
curved care lines (`reports/error_cases/detector/GW-DET-006/`).

## 4. GLYPHARIS-Rec

An SVTR-style encoder with a CTC head (`glypharis/models/rec/svtr.py`, model name `svtr`). The recipe
is GW-REC-024's with Phase 13's wider rotation (`configs/rec/gw-rec-034.yaml`, which inherits
`gw-rec-024`, `gw-rec-022`, `gw-rec-020`, `gw-rec-003` and `gw-rec-001`).

| Part | As built (v1) | Source |
|---|---|---|
| Input | Grey line image resized to height 48 with its aspect ratio kept, width capped at 1536 (a wider line is squeezed), scaled to [−1, 1], padded on the right with the median of its last column | `configs/final/glypharis_v1.yaml`, `glypharis/pipeline/recognizer.py` |
| Stem | Two strided convolutions with batch norm and GELU; height and width divided by 4, so T = W // 4 frames | `glypharis/models/rec/svtr.py` |
| Stage 1 | 96 dimensions, 3 local mixing blocks (depthwise convolution, kernel 3×7) | `configs/rec/gw-rec-020.yaml` |
| Stage 2 | 192 dimensions, 3 local blocks and 3 global blocks (self-attention, 6 heads) | `configs/rec/gw-rec-020.yaml` |
| Stage 3 | 256 dimensions, 6 global blocks (8 heads) | `configs/rec/gw-rec-020.yaml` |
| Between stages | A convolution halves the grid height and changes the embedding width. The remaining height is averaged away at the end. MLP ratio 4, dropout 0.1 | `glypharis/models/rec/svtr.py`, `configs/rec/gw-rec-001.yaml` |
| Padding | Attention never attends to a padded column, and padded tokens are zeroed after every block | `glypharis/models/rec/svtr.py` |
| Output | Linear layer to 275 classes: 274 characters and the CTC blank | `reports/charset/charset_report.md` |
| Size | 7,928,835 parameters | GW-EFF-007, `reports/efficiency/phase16/cost_table.md` |
| Loss | `nn.CTCLoss(zero_infinity=True)`, computed in fp32 outside autocast | `glypharis/training/rec_trainer.py`, `reports/environment/env_report.md` |
| Decoding | Greedy CTC. Beam search and language models are not implemented | `glypharis/pipeline/recognizer.py` |
| Training | 348,935 HierText training lines, no GlyphSynth. Full augmentation preset with rotation up to ±4° on half the lines. Micro-batches of 8 with 8-step accumulation (64 lines per optimizer step), at most 6,144 batch columns. AdamW, learning rate 5e-4, weight decay 1e-2, OneCycle, gradient clip 5, bf16. 30 epochs, 169,860 steps, selected epoch 29, seed 1337 | `configs/rec/gw-rec-034.yaml` chain, `configs/final/glypharis_v1.yaml`, `reports/training/GW-REC-034/report.md` |

Validation CER of the shipped checkpoint is 6.30% [6.17, 6.44] on 73,888 lines, with a 95% line-bootstrap
interval (GW-REC-034, `reports/training/GW-REC-034/report.md`). This pack holds the lines the recognizer
can represent. It is not the 74,500-crop reference that sits beside the test results (GW-REC-038,
`reports/external/recognition.md`).

The recipe was chosen by the Phase 13 ladder (GW-REC-037, `reports/ablation/phase13_ladder.md`): short
budget, three seeds per row, HierText validation CER, each row against its parent, paired over pages,
95% interval.

| Row | Change | Δ CER (points) | Kept |
|---|---|---|---|
| L1 | Full augmentation preset, on L0 | −0.36 [−0.40, −0.31] | yes |
| L2 | 25,000 GlyphSynth lines, on L1 | −0.02 [−0.05, +0.01] | no |
| L3 | Input height 48, on L1 | −0.48 [−0.54, −0.42] | yes |
| L4 | SVTR-style encoder, weight decay 1e-2, its learning rate and micro-batching, on L3 | −0.26 [−0.32, −0.19] | yes |
| L5 | Wider rotation, on L4 | −0.12 [−0.16, −0.08] | yes |

L4 changes the encoder together with weight decay, learning rate and micro-batching, so its gain is not
the encoder's alone. The same holds for GW-REC-022 against GW-REC-006.

At the full budget, three seeds of the final recipe (GW-REC-034, 035, 036) read 6.30, 6.28 and 6.30%
validation CER. Against the CRNN baseline GW-REC-001 (one seed) the mean differs by −1.24 [−1.40, −1.11]
points, paired over pages (GW-REC-037, `reports/ablation/phase13_final.md`). That difference covers every
change from L1 to L5 taken together.

The CRNN stays in the code as the baseline (`glypharis/models/rec/crnn.py`, 7,796,435 parameters,
`reports/training/GW-REC-001/report.md`). TrOCR-small-printed is an external baseline on line crops only
(GW-BASE-004) and is not part of GLYPHARIS.

## 5. Pipeline components

Code is in `glypharis/pipeline/`.

- Crop and rectify (`glypharis.data.crops.cut_line`): each quad is expanded by a 15% margin and warped
  from the grey page to a strip 64 px high, with the page's background grey as fill. These are the
  settings the training crops were made with.
- Batching (`TextRecognizer`): crops are sorted by width and read in batches of at most 32 lines and
  16,384 columns. Each line keeps at least 64 padding columns before the tensor edge, so a line reads
  the same alone or in any batch.
- Reading order (`reading_order.py`): the page's dominant text angle is removed when it is more than 1°
  from horizontal, then a recursive XY-cut splits at the widest empty band. A vertical band counts as a
  column gutter when it is at least one median line height wide and no narrower than the widest
  horizontal band. Lines that still overlap are read row by row. Only geometry is used, so the order is
  a heuristic.
- Confidence (`recognizer.py`): softmax over the per-frame logits divided by a temperature; the geometric
  mean of the top-class probability over frames whose top class is not blank, or for a line read as empty,
  the mean blank probability. Decoding does not depend on the temperature.
- Temperature: T = 2.35, fitted by binary negative log-likelihood on validation half A (35,976 lines).
  On half B (37,912 lines) ECE with 15 equal-mass bins goes from 0.2140 at T = 1 to 0.0886, a difference
  of −0.1254 [−0.1296, −0.1217] with a 95% line bootstrap (GW-E2E-005,
  `reports/calibration/GW-E2E-005/calibration.md`). The temperature does not help on every set: on IIIT5K
  word crops (alphanumeric track, 3,000 words) ECE goes from 0.0546 to 0.1118, a difference of
  +0.0572 [+0.0431, +0.0716] with a 95% interval by word (GW-REC-039,
  `reports/external/calibration.md`).

## 6. Public interface

```python
from glypharis import GlypharisOCR

# A bundle by name or path; nothing is downloaded.
ocr = GlypharisOCR.from_pretrained("glypharis-v1")
# A path, image file bytes, a PIL image or a numpy array.
result = ocr.predict("page.png")
results = ocr.predict_batch([...])

print(result.text)  # the lines joined with newlines, in reading order
result.to_dict()  # JSON-serialisable; result.to_json() gives the string
```

Fields of `OCRResult.to_dict` (`glypharis/pipeline/ocr.py`):

| Field | Meaning |
|---|---|
| `glypharis_version`, `model_version` | Package version (`0.1.0.dev0` until Phase 23, `pyproject.toml`) and model version (`glypharis_v1`) |
| `width`, `height` | Page size in pixels |
| `text` | Non-empty lines joined with newlines, in reading order |
| `lines[].text` | The line's text |
| `lines[].confidence` | Recognition confidence at the bundle's temperature (§5) |
| `lines[].det_score` | The detector's score for the line, kept apart from `confidence` |
| `lines[].bbox` | x0, y0, x1, y1 of the polygon, in page pixels |
| `lines[].polygon`, `lines[].quad` | Outline, and TL, TR, BR, BL along the reading direction, in page pixels |
| `lines[].confidence_at` | Only when other temperatures are requested |
| `seconds` | Time per step: `load`, `detect`, `recognise`, `order`, `total` |

Failures a caller can cause or meet are `GlypharisError` subclasses with a stable code (`invalid_image`,
`image_too_large`, `model_load_error`; `glypharis/errors.py`). The default pixel limit is 64,000,000
(`glypharis/pipeline/ocr.py`); it is a set value and has not been measured against memory use
(`docs/PHASE_STATUS.md`, Phase 20 v1 re-run). On the v1 bundle, on both backends, the hardening cases
ended in a result or a `GlypharisError`, with no other exception (GW-E2E-012,
`reports/hardening/GW-E2E-012.json`).

Command line: `glypharis-ocr --image x.png` or `--input-dir ./images` (`python infer.py` is the same)
writes `results/<name>.json` and `results/<name>.txt`. Exit code 1 means an image failed and 2 a usage
error.

HTTP (optional, Phase 19): `glypharis-serve` offers `POST /ocr` (the image file as the request body) and
`GET /health`. It uses the standard library, one process and one request at a time, listens on
127.0.0.1 by default and has no authentication or TLS (`glypharis/server.py`). It is an integration
boundary for local use.

## 7. Repository layout

```
glypharis/                 # importable package
  config/                  # typed config loading and validation (YAML to dataclasses)
  data/                    # dataset readers, manifests, normalisation, charset, augmentation, GlyphSynth, GDC
  models/det/              # GLYPHARIS-Det
  models/rec/              # GLYPHARIS-Rec (SVTR-style) and the CRNN baseline
  training/                # trainers, checkpointing, run reports
  diagnostics/             # over- and underfitting flags, CTC collapse, gradient health
  eval/                    # metrics, slices, bootstrap intervals, test-access guard
  pipeline/                # GlypharisOCR, recognizer wrapper, reading order, calibration, frozen v1 builder
  export/                  # release files, ONNX export, the model bundle
  efficiency/              # Phase 16 tooling: thread control, variants, fused attention, v1 code check
  utils/                   # seeding, logging, device and VRAM monitoring, git information
  cli.py  server.py  errors.py
configs/                   # hyperparameters; configs/final/glypharis_v1.yaml is the freeze record
scripts/                   # data preparation, evaluation, reports, checks
tests/                     # pytest suite; tiny fixtures and tiny models only
data/                      # raw/ and processed/ (ignored), manifests/ (committed)
reports/                   # generated analysis, curves and benchmark tables
research/                  # research log, hypotheses, registry, findings, paper notes
docs/                      # plan, roadmap, protocol, architecture, decisions
requirements/              # lock files, one per environment
train.py  infer.py         # training entry point; same command as glypharis-ocr
MODEL_CARD.md  DATASETS.md  EXPERIMENTS.md
```

There is no `evaluate.py`: evaluation runs through named scripts such as `scripts/e2e_eval.py`,
`scripts/det_eval.py` and `scripts/phase14_score.py`.

Python package names are lowercase per PEP 8 (`glypharis`), while the project and model names stay
GLYPHARIS, GLYPHARIS-Det and GLYPHARIS-Rec ([ADR-0004](decisions/ADR-0004-package-naming.md)).

## 8. Model bundle and runtimes

A bundle is one directory (`glypharis/export/bundle.py`, format `glypharis-bundle-1`): `detector.pt`,
`recognizer.pt`, `detector.onnx`, `recognizer.onnx` and `manifest.json`, which holds the SHA-256 of each
file, the detector settings, the recognizer's input settings and temperature, and their provenance.
`read_manifest` checks every hash before anything loads. `from_pretrained` looks in `$GLYPHARIS_MODELS`,
`~/.glypharis/models` and `./models`.

| File | Parameters | Release `.pt` (MiB) | fp32 ONNX (MiB) |
|---|---|---|---|
| Detector | 3,187,209 | 12.44 | 12.15 |
| Recognizer | 7,928,835 | 30.31 | 30.40 |

Source: GW-EFF-007, `reports/efficiency/phase16/cost_table.md`.

PyTorch on the CPU is v1's default backend (owner decision of 2026-11-18,
[ADR-0011](decisions/ADR-0011-cpu-latency-target.md)). The ONNX backend (`backend="onnx"`, TorchScript
exporter, opset 17, dynamic axes) reads all four files; the PyTorch backend does not open the `.onnx`
files.

Both backends were compared with hashed references from the PyTorch v1 build on validation data:

| Check (limit) | Items | ONNX, GW-EFF-006 | PyTorch bundle, GW-EFF-012 |
|---|---|---|---|
| Detector map, max abs difference (1e-4) | 110 pages | 7.33e-06 | 0.0 |
| Logits, max abs difference (1e-3) | 5,000 lines | 3.09e-04 | 0.0 |
| Text | 5,000 lines | no differences | 5,000 identical |
| Page CER difference, as a fraction (5e-04, that is 0.05 points) | 400 pages | 1.49e-05 | 0.0 |

Sources: `reports/efficiency/phase16/README.md`, `reports/export/GW-EFF-012/torch_bundle_check.json`.

The installed package on the PyTorch backend read the same 400 pages with 24.32% page CER, the value of
the frozen pipeline run on the GPU in strict fp32, and 16,835 of 16,835 lines had identical text and box
corners within 1 px (GW-E2E-011 against GW-E2E-006, `reports/package/GW-E2E-011.json`). These 400 pages
are the Phase 5 subset; on all 1,715 validation pages the frozen pipeline's page CER is
27.57% [25.11, 30.36], with a 95% page-bootstrap interval (GW-E2E-006, `reports/external/end_to_end.md`).

Test results come from the GPU strict-fp32 evaluation; equality with the bundle is shown on validation
only. Phase 22 checks reproduction within the 0.05-point tolerance.

## 9. Efficiency and target hardware

All figures in this section are from one laptop (`reports/environment/env_report.md`) on HierText
validation data.

Phones are the target hardware; no phone has been measured. The laptop figures only rank candidate
builds (ADR-0011, owner decision of 2026-11-18).

ADR-0011's laptop targets are recorded as missed for v1. Its limits and v1's values at 10 CPU threads
(GW-EFF-007, `reports/efficiency/phase16/README.md`):

| Build | Line mean (≤ 25 ms) | Line p95 (≤ 70 ms) | Page mean (≤ 1,500 ms) | Page p95 (≤ 2,700 ms) | Peak RAM on pages (< 1,836 MiB) |
|---|---|---|---|---|---|
| v1 PyTorch | 33.9, over | 77.6, over | 2,565.7, over | 5,127.1, over | 1,641.8, within |
| v1 fp32 ONNX | 92.2, over | 304.4, over | 8,573.7, over | 19,011.5, over | 6,529.6, over |

Most of the page time is the recognizer's network: 2,148.2 of 2,565.7 ms on PyTorch and 8,167.7 of
8,573.7 ms on ONNX (GW-EFF-007, `reports/efficiency/phase16/cost_table.md`).

Builds can be compared only within one sitting. The GW-EFF-009 sittings give, at 10 threads, each item
the mean of its two runs, with descriptive 95% intervals by page (`reports/efficiency/phase16/pareto.md`).
Page CER is on the 400 Phase 5 validation pages and page time on the 110 profile pages. These are other
sittings than the table above, so v1's times differ from it.

| Point | Version | Page CER % | Mean ms per page | p95 ms | Peak RAM MiB |
|---|---|---|---|---|---|
| `v1-torch` | v1 | 24.32 [20.87, 28.15] | 2,637.3 [2,395.4, 2,909.1] | 5,327.7 | 1,665.0 |
| `v1-onnx` | v1 | 24.32 [20.87, 28.15] | 9,739.4 [8,510.8, 11,113.1] | 22,474.1 | 6,525.5 |
| `fused-attention` | new version, not v1 | 24.32 [20.87, 28.15] | 4,020.8 [3,604.2, 4,479.8] | 8,267.1 | 2,393.0 |

The `fused-attention` build is the phone candidate (ADR-0011). It exports each of the recognizer's nine
global mixers as one `com.microsoft.MultiHeadAttention` node (`glypharis/efficiency/fused_attention.py`).
Against `v1-onnx` its page time differs by −5,718.6 [−6,630.9, −4,883.7] ms. On 5,000 validation lines
its logits are within 3.01e-04 of PyTorch's and its text is identical
(`reports/efficiency/phase16/variants/fused-attention/lines.json`). Whether phone builds of ONNX Runtime
include this operator has not been checked. A new version, not v1; validation and laptop figures only; to
be evaluated in Phase 22.

Two confirmatory latency tests were not met (`reports/efficiency/phase16/verdicts.md`):

| Test | Interval | Verdict |
|---|---|---|
| H1′ ONNX: v1 fp32 ONNX bundle against GW-BASE-002 (PP-OCRv5 mobile), one sitting (GW-EFF-010) | +6,537.6 [+5,383.7, +7,806.3] ms per page | not met |
| H6′: SVTR-style GW-REC-020 against 1.5 × CRNN GW-REC-003, fp32 ONNX, one sitting (GW-EFF-011) | +16.78 [+12.02, +22.04] ms per line | not met |

## 10. What changed from the Phase 0 draft

| Item | Phase 0 draft | As built | Source |
|---|---|---|---|
| Recognizer | CRNN, with an SVTR-style encoder as a Phase 7 candidate | SVTR-style encoder; the CRNN is the baseline | GW-REC-034, GW-REC-037 |
| Recognizer input height | 32 or 48 | 48 | `configs/final/glypharis_v1.yaml` |
| Recognizer sequence model | 2-layer BiLSTM, hidden 256 | Global attention blocks (the BiLSTM remains in the baseline) | `configs/rec/gw-rec-020.yaml` |
| Decoding | Greedy, optional beam search | Greedy only | `glypharis/pipeline/recognizer.py` |
| Over-long lines | Split at detector level | Squeezed to the 1536 width cap | `glypharis/data/rec_dataset.py` |
| Detector input | Long side 1280 | Long side 1536 | GW-DET-005 |
| Detector box threshold | 0.6 | 0.5 | GW-DET-004 |
| Detector unclip | `dbnet15` and `inverse10` candidates | `inverse09` | GW-DET-004 |
| Preprocessing | Optional page deskew | No page deskew before detection. The angle is used only for reading order | `glypharis/pipeline/ocr.py`, `reading_order.py` |
| Orientation | Optional 180° check | None | F24 |
| Confidence | Mean of per-frame maxima | Geometric mean at T = 2.35 | `glypharis/pipeline/recognizer.py`, GW-E2E-005 |
| Output timing | `processing_time_ms` | `seconds` with one value per step | `glypharis/pipeline/ocr.py` |
| Output fields | `page_number`, `raw_confidence` planned | Not present; `det_score`, `quad`, `width`, `height` added | `glypharis/pipeline/ocr.py` |
| API alias | `GLYPHARISOCR` | Not exported; only `GlypharisOCR` | `glypharis/__init__.py` |
| Package layout | `glypharis/inference/`, `evaluate.py` | `glypharis/pipeline/`, `glypharis/export/`, `glypharis/efficiency/`; evaluation through scripts | §7 |
