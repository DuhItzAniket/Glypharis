# GLYPHARIS — Target Architecture

Status: **design draft (Phase 0)**. Concrete hyperparameters marked _(tunable)_ are starting points to be
confirmed or replaced by measured results in Phases 6–8. Nothing here has been trained yet.

## 1. Scope

GLYPHARIS is a standalone OCR engine: **image in → structured text out**. It knows nothing about PDFs,
chunking, retrieval or QA. A consumer such as BookWorm renders a page to an image and calls the
engine. The repository contains no consumer-specific code.

```
                  INPUT IMAGE  (path | PIL.Image | numpy array)
                        │
                        ▼
        ┌──────────── preprocessing ─────────────┐
        │ decode, EXIF orientation, colour space, │
        │ optional page deskew, resize to /32     │
        └────────────────────────────────────────┘
                        │
                        ▼
              GLYPHARIS-Det  (text-line detector)
                        │  line polygons + detector scores
                        ▼
           crop + rectify (perspective warp per line)
                        │  line images, height-normalised
                        ▼
              GLYPHARIS-Rec  (CTC line recognizer)
                        │  line text + per-frame probabilities
                        ▼
        reading order (block grouping → top-to-bottom, left-to-right)
                        │
                        ▼
        confidence (raw → calibrated, only if validated)
                        │
                        ▼
                 OCRResult (JSON-serialisable)
```

The detector and recognizer are trained, evaluated, and versioned **independently** and are only
combined in Phase 10.

## 2. Granularity decision: lines, not words

The detector localises **text lines** and the recognizer reads **whole lines**.

- Document text is organised in lines; line-level recognition keeps inter-word spacing information and
  avoids a separate word-segmentation step that fails on tight kerning or justified text.
- CTC decoding handles long sequences without the attention-drift problem autoregressive decoders show on
  long inputs (literature evidence collected in `research/approach_survey.md`).
- HierText provides line annotations directly. Synthetic document data can be generated at line level with
  exact boxes.
- Word boxes can be recovered later from CTC alignments (a future-compatible `word_boxes` field) without
  changing the training granularity.

Word-crop benchmarks (IIIT5K) are still used, as an *out-of-domain recognizer test* (RQ6).

## 3. GLYPHARIS-Det (baseline)

DBNet-style differentiable-binarisation segmentation detector, as built in Phase 8 (decisions in
[ADR-0012](decisions/ADR-0012-detector-labels-and-operating-points.md)).

| Part | Baseline choice | Notes |
|---|---|---|
| Backbone | MobileNetV3 (torchvision; Large default, Small for the `-S` variant), ImageNet-pretrained | Pretrained-vs-scratch is a planned ablation (GW-DET-002). Only generic ImageNet weights are used; no pretrained OCR weights. |
| Input | uint8 RGB, H and W multiples of 32 | ImageNet normalisation happens inside the model, so callers pass raw pixels |
| Neck | Lightweight FPN, fused to 1/4 input resolution | 96 inner channels (`det.model.inner_channels`) |
| Head | Probability map + threshold map → approximate binary map | Threshold branch used only in training. The final transposed convolutions of both heads run in fp32 with autocast off, so the logits are always fp32 |
| Labels | Line polygons shrunk by D = A(1 − r²)/L (DBNet shrink ratio r = 0.4) | A mitre inward offset with shapely, which for a convex polygon equals the Vatti offset the reference code computes with `pyclipper`. Pixels are labelled by exact centre sampling; a line whose kernel has no pixel centre becomes don't care for that sample |
| Loss | L = L_prob(BCE + OHEM 3:1) + α·L_bin(Dice) + β·L_thresh(L1), BCE weight 1, α=1, β=10 | Checked: the DBNet reference code scales the BCE term by 5 (`bce_scale`); kept at 1 as pre-registered. `det.loss.bce_weight` is configurable for a later ablation under its own ID |
| Train input | Random crops 640×640 with scale/rotate augmentation | Peak memory is gated per run by the sanity dry run (≤ 4.0 GiB reserved) |
| Inference | Long side resized to 1280 (a multiple of 32) → threshold the probability map (0.3) → components scored by their mean probability (≥ 0.6) → unclip → boxes | Two pre-registered unclip operating points, `dbnet15` (DBNet's ratio 1.5) and `inverse10` (the exact inverse of the training shrink for a rectangle); training selects on the better of the two (ADR-0012) |

**Deviation from the original plan (unclip → polygons).** The output shape is set by
`det.post.box_type`: `quad` (default, all GW-DET runs) gives the rotated rectangle around the
unclipped kernel; `poly` gives the kernel's outline offset by the same unclip distance. Candidates
keep the simplified contour, so Phase 9 can compare the two on validation before anything is
evaluated on GDC, whose phone-captured lines can be curved (RISK_REGISTER R-22).

Outputs, per line: polygon (N×2, clipped to the image), quad (TL, TR, BR, BL of its minimum-area
rectangle, with the text direction from the aspect-2 rule in ADR-0012), axis-aligned bbox, score,
aspect (long/short side) and long-axis angle.

## 4. GLYPHARIS-Rec (baseline: CRNN)

| Part | Baseline choice | Notes |
|---|---|---|
| Input | Grayscale line image, height H = 32 or 48 _(ablation)_, aspect ratio preserved, variable width (bucketed), right-padded | Max width capped _(tunable)_; overly long lines split at detector level |
| CNN | VGG/ResNet-lite stack collapsing height to 1, horizontal stride 4 | Gives T = W/4 frames, enough for CTC on dense lines |
| Sequence | 2-layer BiLSTM, hidden 256 per direction _(tunable)_ | |
| Output | Linear → |charset| + 1 (CTC blank) | Charset from `charset.json` (Phase 4) |
| Loss | `nn.CTCLoss(zero_infinity=True)` computed in **fp32** even under autocast | Avoids fp16 overflow in log-space sums |
| Decoding | Greedy (default), beam search (optional, lexicon-free) | No language model in v1 |
| Size target | ≈ 5–10 M parameters | Measured in Phase 6 |

**GLYPHARIS-Rec-M (Phase 7 candidate):** an SVTR-style mixing-block encoder with CTC head, a drop-in
replacement behind the same interface. Promoted only if it beats CRNN on validation CER under the same
data and budget.

**TrOCR-Small** is used only as an external comparison (fine-tune and/or zero-shot), never as the primary
model.

## 5. Pipeline components

- **Crop and rectify:** min-area rectangle or polygon → perspective warp to a horizontal strip; height
  normalisation; optional 180° flip check (a small orientation classifier is a later, optional addition).
- **Reading order:** v1 heuristic, which groups lines into blocks/columns (recursive XY-cut on line boxes),
  then orders blocks top-to-bottom, left-to-right and lines within each block top-to-bottom. Explicit
  tests cover single-column, two-column, mixed blocks, ragged line lengths and rotated regions. Known to
  be imperfect; documented as a limitation.
- **Confidence:** raw line confidence = mean of max per-frame probabilities over non-blank frames
  (alternatives evaluated). A *calibrated* confidence (temperature scaling fit on validation) is exposed
  only after its calibration error is measured and reported.

## 6. Public interface (target, Phase 18)

```python
from glypharis import GlypharisOCR  # alias GLYPHARISOCR kept for the original spec

ocr = GlypharisOCR.from_pretrained("path/or/version")
result = ocr.predict("page.png")  # path | PIL.Image | np.ndarray
results = ocr.predict_batch([...])

print(result.text)
result.to_dict()  # JSON-serialisable
```

```json
{
  "text": "Recognized text...",
  "lines": [
    {"text": "Recognized line", "confidence": 0.96, "bbox": [12, 40, 610, 72],
     "polygon": [[12,40],[610,41],[610,72],[12,71]]}
  ],
  "processing_time_ms": 142,
  "model_version": "0.1.0",
  "page_number": null
}
```

`confidence` is documented as **raw** until calibration is validated; then `confidence` becomes the
calibrated value and `raw_confidence` is added. Internal details (feature maps, logits) are not exposed.

CLI (Phase 18): `python infer.py --image x.png` / `python infer.py --input-dir ./images` →
`results/<name>.json` and `results/<name>.txt`. Optional HTTP boundary (Phase 19): `POST /ocr`.

## 7. Repository layout (target)

```
glypharis/                 # importable package
  config/                  # typed config loading + validation (YAML → dataclasses)
  data/                    # dataset readers, manifests, normalisation, charset, augmentation, synth
  models/det/              # GLYPHARIS-Det
  models/rec/              # GLYPHARIS-Rec (CRNN, SVTR-style)
  training/                # trainer, AMP, checkpointing, early stopping, resume
  diagnostics/             # over/underfitting, CTC collapse, grad health, leakage checks
  eval/                    # metrics (CER/WER/NED, det P/R/F1/IoU), bootstrap CIs, reports
  inference/               # pipeline, reading order, confidence, GlypharisOCR API
  utils/                   # seeding, logging, device/VRAM monitor, git info
configs/                   # all hyperparameters live here (no magic numbers in code)
scripts/                   # download_datasets.py, audit_data.py, env_report.py, run_baselines.py …
tests/                     # pytest; tiny fixtures only
data/                      # raw/ interim/ processed/ (ignored) + manifests/ (committed)
reports/                   # generated analysis, training curves, benchmark tables
research/                  # research log, hypotheses, registry, paper notes
docs/                      # plan, roadmap, protocol, architecture, decisions
train.py  evaluate.py  infer.py
```

Python package names are lowercase per PEP 8 (`glypharis`), while the project and model names stay
**GLYPHARIS**, **GLYPHARIS-Det** and **GLYPHARIS-Rec**. See `docs/decisions/ADR-0004-package-naming.md`.
