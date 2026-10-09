# Approach Survey

Phase 0 survey of methods, systems and datasets, and why the GLYPHARIS design was chosen. This is a
qualitative survey: it records *published* properties and design trade-offs, not numbers measured here.
Any performance figure quoted from a paper is attributed to that paper and is not comparable with our
results until reproduced under [EXPERIMENT_PROTOCOL](../docs/EXPERIMENT_PROTOCOL.md). Citation keys refer to
[`paper/references.bib`](paper/references.bib); entries are to be checked against the publisher pages in
Phase 21.

## 1. Problem framing

Input: an image of a document page (scanned, photographed, degraded, rendered from an image-only PDF).
Output: text with line boxes, confidence and timing. Constraints: train on one 6 GB laptop GPU, run cheaply
at inference (CPU-first, [ADR-0008](../docs/decisions/ADR-0008-usable-engine-first.md)), English printed
text, and a research record that can explain errors.

## 2. System families

| Family | Examples | Strengths | Weaknesses for this project |
|---|---|---|---|
| Classical / LSTM engines | Tesseract 4/5 (`smith2007tesseract`) | Mature, CPU-only, good on clean scans | Weak on photos, skew, low contrast, complex layouts; not trainable in our framework. Kept as a **baseline**. |
| Two-stage det + rec toolkits | PaddleOCR PP-OCR series (`du2020ppocr`), docTR, EasyOCR | Small, fast, strong; same design family as ours | Using their weights would make GLYPHARIS a fine-tune ([ADR-0007](../docs/decisions/ADR-0007-pretrained-weights-policy.md)). PP-OCRv5 is the main **baseline**. |
| Transformer line recognizers | TrOCR (`li2023trocr`), PARSeq (`bautista2022parseq`), ABINet (`fang2021abinet`) | High accuracy on crops | Autoregressive / iterative decoding is slower; TrOCR-Small is still large for CPU. TrOCR-Small kept as a recognition-only **baseline**. |
| Page-level image-to-sequence | Donut (`kim2022donut`), Nougat (`blecher2023nougat`), OCR-capable VLMs | Handle layout and reading order jointly | Hundreds of millions of parameters or more; not trainable on 6 GB; slow on CPU; hallucination risk; no line boxes by default. Out of scope ([ADR-0001](../docs/decisions/ADR-0001-two-stage-pipeline.md)). |

## 3. Text detection

| Method | Idea | Assessment |
|---|---|---|
| EAST (`zhou2017east`) | Direct per-pixel box regression | Simple and fast, weaker on long, curved or dense lines. |
| CRAFT (`baek2019craft`) | Character region + affinity heatmaps | Strong on scene text; needs character-level (often weak) supervision; heavier post-processing. |
| PSENet (`wang2019psenet`) | Progressive scale expansion of kernels | Good separation of close lines; slower post-processing. |
| **DBNet / DBNet++** (`liao2020dbnet`, `liao2022dbnetpp`) | Segmentation with a differentiable binarisation threshold map | Light backbones work well, simple post-processing, arbitrary shapes, used in production toolkits. **Chosen** for GLYPHARIS-Det with MobileNetV3 (`howard2019mobilenetv3`). |

Documents differ from scene text: many small, dense, nearly horizontal lines. Line-level DBNet targets
with a shrink ratio and an unclip step separate adjacent lines reasonably; Phase 9 measures merge/split
errors explicitly.

## 4. Text recognition

| Method | Idea | Assessment |
|---|---|---|
| **CRNN + CTC** (`shi2017crnn`, `graves2006ctc`) | CNN features → BiLSTM → CTC | Small, fast, well understood, trains from scratch on modest hardware, good on horizontal printed lines. **Baseline** GLYPHARIS-Rec. |
| Attention encoder-decoder (e.g. ASTER `shi2019aster`) | Autoregressive decoding with rectification | Better on curved/irregular words; slower; drift on long lines. Not needed for document lines. |
| **SVTR** (`du2022svtr`) | Vision-transformer-style mixing blocks + CTC, no RNN | Strong accuracy at small size, CTC speed. **Phase 7 candidate** (GLYPHARIS-Rec-M). |
| PARSeq / ABINet | Permuted / iterative language-aware decoding | Strong on scene words; more complex and slower; language modelling is out of scope for v1. |
| TrOCR | Pretrained ViT encoder + text decoder | Strong, but pretraining scale is far beyond the laptop; comparison only. |

Conclusion: CTC on lines is the right baseline for printed documents under a CPU-first constraint
([ADR-0002](../docs/decisions/ADR-0002-line-level-recognition.md)); the SVTR-style encoder is the main
upgrade path.

## 5. Training data

See [DATASETS.md](../DATASETS.md) for cards. Key observation: **domain gap.** HierText and SynthText are
scene-text-oriented, the target is document pages. Options considered:

| Option | Assessment |
|---|---|
| Train on HierText + SynthText only (as briefed) | Kept as the control; likely to under-serve dense document text. |
| Add real document datasets with human line transcriptions | Few exist at scale with permissive licences; those that exist are used as external tests to keep them independent. |
| Pseudo-label scans with another OCR engine | Would distil that engine's errors and violate the spirit of ADR-0007; out of scope for v1. |
| **Render document-style synthetic data (GlyphSynth)** | Exact labels, unlimited volume, matches the target domain, licence-controllable. **Adopted** ([ADR-0003](../docs/decisions/ADR-0003-document-domain-synthetic-data.md)); its value is measured, not assumed (RQ2, Phase 12). |

Degradation augmentation for documents: Augraphy-style pipelines (`groleau2023augraphy`) and our own
transforms. Robustness is measured with a fixed corruption suite in the style of ImageNet-C
(`hendrycks2019benchmarking`).

## 6. Evaluation practice

- CER / WER / NED as the core recognition metrics, with explicit normalisation tracks, because published
  numbers often differ in case and punctuation handling.
- Detection by IoU-matched precision / recall / F1, plus dataset-native metrics where defined.
- Calibration via ECE and temperature scaling (`guo2017calibration`).
- Bootstrap confidence intervals (`efron1994bootstrap`) and multiple seeds, because single-run differences
  in OCR papers are often within noise.

## 7. Decisions taken from this survey

| Decision | Record |
|---|---|
| Two-stage detector + recognizer | ADR-0001 |
| Lines + CTC | ADR-0002 |
| GlyphSynth next to SynthText | ADR-0003 |
| Fixed dataset roles | ADR-0005 |
| ImageNet backbones only, no OCR weights | ADR-0007 |
| Usable engine first, CPU-first inference | ADR-0008 |

## 8. Open questions for later phases

- Does an SVTR-style encoder beat CRNN at the same CPU latency? (Phase 7, RQ7)
- How much do HierText's document-like images help compared with GlyphSynth? (Phase 12)
- Is XY-cut reading order good enough for GDC's multi-column pages? (Phase 10, R-14)
