# GLYPHARIS

A lightweight OCR engine for document images. It recovers readable text from scanned, photographed,
degraded, or image-only document pages and returns it as structured output: lines, boxes, confidence,
and timing.

GLYPHARIS is built as a research project. Every model is trained and evaluated on a single laptop GPU
(RTX 4050, 6 GB), every experiment is registered before it runs, and every reported number traces back to
a generated report.

> **Status: Phase 2 — dataset acquisition.** No model has been trained yet. Nothing below is a result.
> See [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md). First usable version (v0.1) is planned right after the
> end-to-end pipeline works ([ADR-0008](docs/decisions/ADR-0008-usable-engine-first.md)).

## What it will do

```python
from glypharis import GlypharisOCR

ocr = GlypharisOCR.from_pretrained("glypharis-0.1")
result = ocr.predict("page.png")

print(result.text)
for line in result.lines:
    print(line.text, line.confidence, line.bbox)
```

```
image ─▶ preprocessing ─▶ GLYPHARIS-Det (line detector) ─▶ crop + rectify
      ─▶ GLYPHARIS-Rec (CTC line recognizer) ─▶ reading order ─▶ structured result
```

- **GLYPHARIS-Det**: DBNet-style text-line detector with a MobileNetV3 backbone.
- **GLYPHARIS-Rec**: CNN + BiLSTM + CTC line recognizer (baseline), with an SVTR-style variant under study.
- Stands alone: any application can call it with an image. It contains no application-specific code.

## Research questions

1. How effective is a lightweight detector + CTC recognizer for document OCR?
2. Does synthetic data help, and does its *domain* matter (scene-text vs document-style) at equal budget?
3. Which degradation augmentations improve robustness on degraded documents?
4. How does it compare with Tesseract, PaddleOCR (PP-OCRv5) and TrOCR-Small under one protocol?
5. Which failure modes dominate, detector or recognizer?
6. How large is the gap between document-domain and scene-text performance?
7. What is the accuracy / latency / memory trade-off for local deployment?
8. Is the model's confidence calibrated?

## Documentation

| Document | What's in it |
|---|---|
| [Implementation plan](docs/IMPLEMENTATION_PLAN.md) | 24 phases with tasks, deliverables and exit criteria |
| [Roadmap](docs/ROADMAP.md) | Milestones, versions, dependency graph, success criteria |
| [Architecture](docs/ARCHITECTURE.md) | System design and public interface |
| [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) | Metrics, splits, statistics, diagnostics, leakage rules |
| [Datasets](DATASETS.md) | Provenance, licences and roles of every dataset |
| [Approach survey](research/approach_survey.md) | Alternatives considered and why this design was chosen |
| [Risk register](docs/RISK_REGISTER.md) | Known risks and mitigations |
| [Decisions](docs/decisions/) | Architecture decision records |
| [Workflow](docs/WORKFLOW.md) | Branching, commits, phase gate, experiment IDs |
| [Environment](docs/ENVIRONMENT.md) | Development machine, as measured |
| [Research log](research/) | Hypotheses, experiment registry, ablations, failures, findings, paper notes |

## Repository layout

```
glypharis/      package (from Phase 1)
configs/        all hyperparameters (YAML)
scripts/        dataset download, audit, baselines, env report
tests/          pytest suite
data/           manifests are committed; raw/processed data is not
reports/        generated analysis, curves and benchmark tables
research/       research log and paper notes
docs/           plans, protocol, decisions
```

## Hardware target

| | |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop, 6 GB (≈ 4.5–5 GB usable) |
| CPU | Intel Core i7-14650HX |
| RAM | 16 GB |
| OS | Windows 11; CI on Linux (CPU) |

Training supports CUDA with automatic CPU fallback, mixed precision (bf16/fp16), gradient accumulation,
checkpoint resume, and VRAM monitoring. These arrive in Phases 1 and 6.

## Licence

Code licence to be decided. Model weights will be released only on terms compatible with the training
datasets' licences (see [DATASETS.md](DATASETS.md)).
