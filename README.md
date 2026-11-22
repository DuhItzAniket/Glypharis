# GLYPHARIS

A lightweight OCR engine for document images. It recovers readable text from scanned, photographed,
degraded, or image-only document pages and returns it as structured output: lines, boxes, confidence,
and timing.

GLYPHARIS is built as a research project. Every model is trained and evaluated on a single laptop GPU
(see [Hardware](#hardware)), every experiment is registered before it runs, and every reported number
traces back to a generated report.

> **Status: v1.0.** v1.0 is the release of v1 (GW-REC-034 with GW-DET-001): the code under
> Apache-2.0, published on PyPI as `glypharis` 1.0.0, and the v1 weights as the asset
> `glypharis-v1.tar.gz` of the GitHub release `v1.0`, which `glypharis.download_model` fetches and
> checks against pinned hashes. Phases 0 to 22 are complete, including the export, package and
> hardening re-runs on the final model, the research package (Phase 21) and final validation
> (Phase 22, tag `phase-22`; its record is `docs/FINAL_VALIDATION.md`). Phase 23 is the release.
> The phone candidate is not released.
> `v0.1` is the earlier usable engine ([ADR-0008](docs/decisions/ADR-0008-usable-engine-first.md)).
> See [docs/PHASE_STATUS.md](docs/PHASE_STATUS.md) for what is done and every number behind it,
> [MODEL_CARD.md](MODEL_CARD.md) for what v1 is, how it measured and where it fails, and
> [docs/RELEASE_NOTES_v1.0.md](docs/RELEASE_NOTES_v1.0.md) for what v1.0 ships.

## Install

The package needs Python 3.11 (3.11.4 or later, below 3.12):

```
pip install glypharis==1.0.0
```

pip installs the default PyTorch build for your platform; for a particular build (CPU only, or a CUDA
version), install `torch` and `torchvision` from the PyTorch index first
([requirements/README.md](requirements/README.md) has the project's pins). From a checkout:

```
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install .                # core: reads pages with either backend
pip install ".[export]"      # adds onnx, for writing bundles and ONNX files
pip install ".[train]"       # adds plotting, font, PDF and monitoring packages for training
```

For development, install the exact versions instead: `pip install -r requirements/lock.txt`, then
`pip install -e . --no-deps` ([docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)). The package holds no
weights; they are fetched separately (next section).

## Using it

```python
import glypharis
from glypharis import GlypharisOCR

glypharis.download_model("glypharis-v1")  # once: fetches and checks the release asset
ocr = GlypharisOCR.from_pretrained("glypharis-v1")  # a model bundle by name or path
result = ocr.predict("page.png")  # path, image file bytes, PIL image or numpy array

print(result.text)  # lines in reading order
for line in result.lines:
    print(line.text, line.confidence, line.bbox)
print(result.to_json())  # the same as structured JSON
```

From the command line (the install provides `glypharis-ocr`; in a checkout, `python infer.py` is the
same):

```
glypharis-ocr --image page.png --output results
glypharis-ocr --input-dir scans/ --recursive --format json txt
```

Each image gives `results/<name>.json` and `results/<name>.txt`. `from_pretrained` never downloads; it
looks in `$GLYPHARIS_MODELS`, `~/.glypharis/models` and `./models` (a bundle is a
directory with `manifest.json`, two release files and two ONNX files; Phase 17). No bundle ships with
this repository or the package. `glypharis.download_model("glypharis-v1")` fetches the asset
`glypharis-v1.tar.gz` of the GitHub release `v1.0`, checks its size and SHA-256 and the bundle's
manifest hash against the pins in `glypharis/hub.py`, and installs it where `from_pretrained` finds
it (the first `GLYPHARIS_MODELS` entry, else `~/.glypharis/models`). `glypharis-ocr` and
`glypharis-serve` do the same for a registered name that is not installed, unless `--no-download` is
given or `GLYPHARIS_OFFLINE` is set; `glypharis-check` checks an installed bundle against the pins
([docs/INTEGRATION.md](docs/INTEGRATION.md)). The default backend is PyTorch on the CPU;
`backend="onnx"` uses ONNX Runtime, which matches PyTorch within Phase 17's tolerances (GW-EFF-006) but
is slower on the CPU (F34).

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
| [Integration guide](docs/INTEGRATION.md) | Embedding GLYPHARIS in an application: install, model lookup, threads, server, errors |
| [Release notes](docs/RELEASE_NOTES_v1.0.md) | What v1.0 ships, its terms, how to install, fetch and verify it |
| [Changelog](CHANGELOG.md) | Changes by version |
| [Release runbook](docs/RELEASE_RUNBOOK.md) | The release-day steps, in order, and who does each |
| [Release evidence](docs/RELEASE_EVIDENCE.md) | Each v1.0 release precondition mapped to its evidence, and the gaps still open |
| [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md) | Metrics, splits, statistics, diagnostics, leakage rules |
| [Approach survey](research/approach_survey.md) | Alternatives considered and why this design was chosen |
| [Risk register](docs/RISK_REGISTER.md) | Known risks and mitigations |
| [Decisions](docs/decisions/) | Architecture decision records |
| [Workflow](docs/WORKFLOW.md) | Branching, commits, phase gate, experiment IDs |
| [Environment](docs/ENVIRONMENT.md) | Development machine and where its generated record lives |
| [Reports](reports/README.md) | What each generated report folder holds |
| [Research log](research/) | Hypotheses, experiment registry, ablations, failures, findings |

The numbers in these documents trace to the generated reports under `reports/`. An archive of the
repository made with `git archive`, such as the Zenodo archive of the release, holds the code and
documentation only: `reports/` and the evaluation records under `data/manifests` are left out
(`export-ignore` in [.gitattributes](.gitattributes), checked by
[scripts/check_git_archive.py](scripts/check_git_archive.py)). The numbers stay in the documents, so
tracing every one of them needs the tagged repository on GitHub.

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

Phones are the target hardware. Phase 22 measured the phone candidate's networks on two phones
(GW-EFF-015, `reports/final_validation/GW-EFF-015/`): network time and memory only, without decoding,
post-processing, cropping or reading order; v1 itself was not measured on a phone. The laptop's figures
only rank candidate builds, and v1 misses the laptop CPU latency targets of
[ADR-0011](docs/decisions/ADR-0011-cpu-latency-target.md) (GW-EFF-007,
`reports/efficiency/phase16/README.md`). No phone target is set. The phone candidate is the all-ONNX
build with fused attention, reading one line per recognizer run (`fused-single-line`, GW-EFF-013). A new
version, not v1. Phase 22 evaluated it once per test set, and it met H11's six non-inferiority tests
against v1 (GW-E2E-021, `reports/final_validation/candidate/h11.md`); its results sit beside v1's and
replace none of them.

## Licence

The code and the project's own text are under the Apache License, Version 2.0 ([LICENSE](LICENSE),
[NOTICE](NOTICE)), with the exceptions that NOTICE lists. The weights are not covered by it.

The weights are distributed under CC BY-SA 4.0 with attribution to HierText and Open Images
([ADR-0014](docs/decisions/ADR-0014-weights-and-backbone-terms.md)); the archive
`glypharis-v1.tar.gz` carries `LICENSE-WEIGHTS` and a `NOTICE.md` with the attribution, the changes
made and the disclosure below ([release/weights/](release/weights/)). The detector was initialised
from ImageNet weights, a disclosed exception to the project's commercial-use rule
([ADR-0009](docs/decisions/ADR-0009-no-synthtext.md)): GW-DET-001's backbone was initialised from
ImageNet-1K weights, and the ImageNet terms do not clearly allow commercial use. The weights are not
described as cleared for commercial use. The owner decided on 2026-11-22 to publish them under these
terms without the qualified advice ADR-0014 had asked for; this is the project's chosen handling, not
a legal conclusion. The terms of each dataset are in [DATASETS.md](DATASETS.md).

## Citing

Cite with [CITATION.cff](CITATION.cff); a Zenodo DOI is added after the deposit. The methods and
datasets behind the model are cited in the model card's
[citation information](MODEL_CARD.md#citation-information).
