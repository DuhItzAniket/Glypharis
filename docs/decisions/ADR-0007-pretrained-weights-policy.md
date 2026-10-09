# ADR-0007: Generic ImageNet backbones allowed; no pretrained OCR weights inside GLYPHARIS

- **Status:** Accepted (2026-10-09, Phase 0)

## Context
The goal is a model trained on this machine. Detector backbones such as MobileNetV3 converge much faster
from ImageNet initialisation, and that initialisation carries no OCR-specific knowledge. On the other
hand, starting from PP-OCR or TrOCR weights would make GLYPHARIS a fine-tune of someone else's OCR model
and muddy every comparison against those systems.

## Decision
- **Allowed:** generic ImageNet-pretrained classification backbones from torchvision (license recorded
  in the model card).
- **Not allowed inside GLYPHARIS-Det / GLYPHARIS-Rec:** any weights trained for text detection or
  recognition (PP-OCR, TrOCR, PARSeq, docTR, EasyOCR, etc.).
- External OCR models are used only as **baselines**, and only in `.venv-baselines`.
- ImageNet-pretrained vs random initialisation of the detector backbone is a planned ablation
  (Phase 8), so the effect of that prior is measured rather than assumed.
- The recognizer is trained from random initialisation (standard for CRNN).

## Consequences
Comparisons against external systems stay clean, and the model card can honestly say "trained from
scratch except for an ImageNet backbone". Convergence will be slower than fine-tuning PP-OCR, which is
acceptable.

## Revisit if
The detector fails to reach a usable F1 within the Phase 8 compute budget even with the ImageNet backbone.
