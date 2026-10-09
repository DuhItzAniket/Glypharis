# ADR-0001: Two-stage detector + recognizer instead of an end-to-end/VLM model

- **Status:** Accepted (2026-10-09, Phase 0)

## Context
Document OCR can be built as (a) a text detector followed by a recognizer on crops, (b) a single
end-to-end spotting network, or (c) an image-to-sequence model that reads a whole page (Donut / Nougat
style encoder-decoders and vision-language models). The constraints are a 6 GB laptop GPU for training,
an engine that must run cheaply at inference time, and a research goal of explaining *where* errors come
from. See `research/approach_survey.md` for the alternatives in detail.

## Decision
GLYPHARIS is a two-stage pipeline: GLYPHARIS-Det (DBNet-style line detector) → crop and rectify →
GLYPHARIS-Rec (CTC line recognizer) → reading order → structured result. The two models are trained,
evaluated and versioned independently and combined in Phase 10.

## Consequences
- Each stage fits comfortably in the VRAM budget and can be trained from scratch on the laptop.
- Detector and recognizer errors can be separated with an oracle analysis (GT boxes → recognizer), which
  directly answers RQ5.
- Line boxes and per-line confidence come for free, matching the output contract.
- Reading order is a separate, heuristic step, and page-level layout understanding is out of scope.
- Errors compound across stages; the E2E metrics capture this.

## Revisit if
A page-level model of comparable size and latency becomes trainable within the laptop budget and clearly
outperforms the pipeline on GDC validation slices.
