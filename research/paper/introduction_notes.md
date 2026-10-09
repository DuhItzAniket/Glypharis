# Introduction — working notes

## Motivation
- Many PDFs in the wild (scans, photographed pages, exported images) carry no text layer, so
  downstream text pipelines (search, chunking, QA) see nothing.
- Large OCR / VLM systems are accurate but heavy; a small, well-measured model that runs and
  trains on a laptop is useful and under-documented.

## Gap (to be validated against related_work.md)
- Few works report a complete, reproducible *laptop-scale* training recipe with honest
  diagnostics (over/underfitting, calibration) and controlled ablations.
- The value of scene-text synthetic data (e.g. SynthText) for *document* OCR is usually assumed,
  not measured against a document-style synthetic alternative at equal budget.

## Contributions (candidate list; keep only what the evidence supports)
1. A reproducible laptop-scale detector + recognizer pipeline with full experiment records.
2. A controlled study of synthetic-data domain and scale for document OCR.
3. A degradation-augmentation ablation on document conditions.
4. Error analysis separating detector vs recognizer failures.
5. Accuracy / latency / memory comparison against Tesseract, PaddleOCR and TrOCR-Small.
