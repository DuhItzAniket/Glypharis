# GLYPHARIS

A lightweight OCR engine for document images. It recovers readable text from scanned, photographed,
degraded, or image-only document pages and returns it as structured output: lines, boxes, confidence,
and timing.

GLYPHARIS is built as a research project. Every model is trained and evaluated on a single laptop GPU
(see [Hardware](#hardware)), every experiment is registered before it runs, and every reported number
traces back to a generated report.

> **Status: v1 evaluated and packaged, not released.** Phases 0 to 20 are complete, including the export,
> package and hardening re-runs on the final model. The research package (Phase 21) is in progress.
> Final validation (Phase 22) has not been run. No weights are distributed; release is Phase 23.
> `v0.1` is the earlier usable engine ([ADR-0008](docs/decisions/ADR-0008-usable-engine-first.md)).
> See [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md) for what is done and every number behind it, and
> [MODEL_CARD.md](MODEL_CARD.md) for what v1 is, how it measured and where it fails.

## Using it

```python
from glypharis import GlypharisOCR

ocr = GlypharisOCR.from_pretrained("glypharis-v1")  # a model bundle by name or path
result = ocr.predict("page.png")  # path, image file bytes, PIL image or numpy array

print(result.text)  # lines in reading order
for line in result.lines:
    print(line.text, line.confidence, line.bbox)
print(result.to_json())  # the same as structured JSON
```

From the command line (`pip install -e .` provides `glypharis-ocr`; `python infer.py` is the same):

```
glypharis-ocr --image page.png --output results
glypharis-ocr --input-dir scans/ --recursive --format json txt
```

Each image gives `results/<name>.json` and `results/<name>.txt`. Model bundles are not downloaded:
`from_pretrained` looks in `$GLYPHARIS_MODELS`, `~/.glypharis/models` and `./models` (a bundle is a
directory with `manifest.json`, two release files and two ONNX files; Phase 17). No bundle ships with
this repository (`runs/` is git-ignored), so `glypharis-v1` resolves only once that directory is in one
of those places; on the machine that built it, set `GLYPHARIS_MODELS=runs/release` or pass the path
`runs/release/glypharis-v1`. The default backend is PyTorch on the CPU; `backend="onnx"` uses ONNX
Runtime, which matches PyTorch within Phase 17's tolerances (GW-EFF-006) but is slower on the CPU (F34).

Over HTTP (optional, Phase 19; `glypharis-serve`, standard library only, one request at a time):

```
glypharis-serve --model glypharis-v1 --port 8080
curl --data-binary @page.png http://127.0.0.1:8080/ocr
```

`POST /ocr` takes the image file as the request body and answers with the same JSON; `GET /health`
reports the versions. It is an integration boundary for local use, without authentication or TLS;
the limits and error codes are in `glypharis/server.py`.

```
image ─▶ preprocessing ─▶ GLYPHARIS-Det (line detector) ─▶ crop + rectify
      ─▶ GLYPHARIS-Rec (CTC line recognizer) ─▶ reading order ─▶ structured result
```

- **GLYPHARIS-Det**: DBNet-style text-line detector with a MobileNetV3 backbone.
- **GLYPHARIS-Rec**: SVTR-style CTC line recognizer in v1 (a CNN + BiLSTM + CTC recognizer is the baseline).
- Stands alone: any application can call it with an image. It contains no application-specific code.

## Research questions

1. How effective is a lightweight detector + CTC recognizer for document OCR?
2. How much does document-style synthetic data help, how does it scale, and which generator factors matter?
3. Which degradation augmentations improve robustness on degraded documents?
4. How does it compare with Tesseract, PaddleOCR (PP-OCRv5) and TrOCR-Small under one protocol?
5. Which failure modes dominate, detector or recognizer?
6. How large is the gap between document-domain and scene-text performance?
7. What is the accuracy / latency / memory trade-off for local deployment?
8. Is the model's confidence calibrated?

## Documentation

| Document | What's in it |
|---|---|
| [Model card](MODEL_CARD.md) | What v1 is, its intended use, results and limitations |
| [Datasets](DATASETS.md) | Provenance, licences and roles of every dataset |
| [Experiments](EXPERIMENTS.md) | Generated summary of every registered experiment |
| [Comparison table](reports/benchmark_results/comparison_table.md) | Generated table of accuracy, latency and size by model and dataset |
| [Paper notes](research/paper/) | Methodology, experiments, results, error analysis, efficiency, limitations |
| [Phase status](docs/PHASE_STATUS.md) | Live record of every phase, with its gate and evidence |
| [Implementation plan](docs/IMPLEMENTATION_PLAN.md) | 24 phases with tasks, deliverables, exit criteria and deviations |
| [Roadmap](docs/ROADMAP.md) | Milestones, versions, dependency graph, success criteria |
| [Architecture](docs/ARCHITECTURE.md) | The system as built, and its public interface |
| [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) | Metrics, splits, statistics, diagnostics, leakage rules |
| [Approach survey](research/approach_survey.md) | Alternatives considered and why this design was chosen |
| [Risk register](docs/RISK_REGISTER.md) | Known risks and mitigations |
| [Decisions](docs/decisions/) | Architecture decision records |
| [Workflow](docs/WORKFLOW.md) | Branching, commits, phase gate, experiment IDs |
| [Environment](docs/ENVIRONMENT.md) | Development machine and where its generated record lives |
| [Reports](reports/README.md) | What each generated report folder holds |
| [Research log](research/) | Hypotheses, experiment registry, ablations, failures, findings |

## Repository layout

```
glypharis/      the package: models, pipeline, training, evaluation, export
configs/        all hyperparameters (YAML); configs/final/ holds the v1 freeze record
scripts/        data preparation, evaluation, reports and checks
tests/          pytest suite
data/           manifests are committed; raw/processed data is not
reports/        generated analysis, curves and benchmark tables
research/       research log and paper notes
docs/           plans, protocol, decisions
requirements/   lock files, one per environment
```

## Hardware

Development and training machine, from the generated
[environment report](reports/environment/env_report.md) (the CPU name is in the `conditions` field of
`reports/efficiency/phase16/profiles/pages_torch_cpu_10t.json`):

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6,140 MiB |
| CPU | Intel Core i7-14650HX, 16 physical cores, 24 logical |
| RAM | 15.71 GiB |
| OS | Windows 11; CI on Linux (CPU) |

Training supports CUDA with automatic CPU fallback, mixed precision (bf16/fp16), gradient accumulation,
checkpoint resume and VRAM monitoring.

Phones are the target hardware; no phone has been measured. The laptop's figures only rank candidate
builds, and v1 misses the laptop CPU latency targets of
[ADR-0011](docs/decisions/ADR-0011-cpu-latency-target.md) (GW-EFF-007,
`reports/efficiency/phase16/README.md`). The phone candidate is the all-ONNX build with fused attention
(GW-EFF-009). A new version, not v1; validation and laptop figures only; to be evaluated in Phase 22.

## Licence

Code licence to be decided. Model weights will be released only on terms compatible with the training
datasets' licences (see [DATASETS.md](DATASETS.md)).
