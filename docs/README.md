# GLYPHARIS documentation

An index of the project's documents, grouped by who they are for. Each document is listed once, under
the audience that needs it first. The project overview, installation and a first example are in the
[repository README](../README.md).

## Release v1.0 at a glance

| What | Where |
|---|---|
| Python package (code, Apache-2.0) | [`glypharis` 1.0.0 on PyPI](https://pypi.org/project/glypharis/) |
| Model weights (CC BY-SA 4.0, attribution to HierText and Open Images) | `glypharis-v1.tar.gz`, an asset of the [GitHub release v1.0](https://github.com/DuhItzAniket/Glypharis/releases/tag/v1.0); fetched and checked by `glypharis.download_model` |
| Mirror of the weights (same files) | [Hugging Face](https://huggingface.co/DuhItzAniket/glypharis-v1) |
| Archived code and documentation | Zenodo, [doi:10.5281/zenodo.24433864](https://doi.org/10.5281/zenodo.24433864) |

The weights' terms, including the disclosed exception for the ImageNet-initialised detector, are in
the [model card](../MODEL_CARD.md#licence-information) and
[ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md).

## Users

For reading pages with the released package.

| Document | What it covers |
|---|---|
| [README](../README.md) | What GLYPHARIS is, how to install it, a first example in Python and on the command line |
| [Model card](../MODEL_CARD.md) | What v1 is, its intended and not intended uses, results, limitations and known failure modes |
| [Release notes v1.0](RELEASE_NOTES_v1.0.md) | What v1.0 ships, its terms, and how to install, fetch and verify it |
| [Changelog](../CHANGELOG.md) | Changes by version, with the experiment IDs behind each |
| [PyPI description](PYPI.md) | The project description shown on the PyPI page |
| [Weights terms](../release/weights/) | `LICENSE-WEIGHTS` and the `NOTICE.md` shipped inside the weights archive |
| [Hugging Face model card](../release/huggingface/README.md) | The card of the Hugging Face mirror of the weights |
| [Licence](../LICENSE) and [NOTICE](../NOTICE) | Apache-2.0 for the code and the project's text, and the material it does not cover |
| [Citation](../CITATION.cff) | How to cite GLYPHARIS (Citation File Format) |
| [Security policy](../SECURITY.md) | Supported versions, how to report a vulnerability privately, and how model files are verified |

## Integrators

For embedding GLYPHARIS in an application.

| Document | What it covers |
|---|---|
| [Integration guide](INTEGRATION.md) | Install, model lookup and download, threads, the HTTP server, errors and result fields |
| [Architecture](ARCHITECTURE.md) | The system as built for v1: detector, recognizer, reading order, bundle format and public interface |

## Reproducers

For rebuilding the environment, the data and the model from a fresh clone.

| Document | What it covers |
|---|---|
| [Reproducing](REPRODUCING.md) | The order of work from a fresh clone to an evaluated model |
| [Environment](ENVIRONMENT.md) | The development machine, the virtual environments and where the generated environment record lives |
| [Requirements](../requirements/README.md) | The lock files, the torch and torchvision pins, and the dependency groups of the package |
| [Datasets](../DATASETS.md) | Provenance, licence and role of every dataset and of the third-party weights and evaluators |
| [Final validation](FINAL_VALIDATION.md) | The record of Phase 22: the fresh-clone walkthrough, the v1 re-run and the phone candidate's evaluation |

## Researchers

For the methods, the evaluation protocol and the results.

| Document | What it covers |
|---|---|
| [Experiment protocol](EXPERIMENT_PROTOCOL.md) | Data roles and splits, metrics, statistics, diagnostics, leakage and test-access rules |
| [Experiments](../EXPERIMENTS.md) | Generated summary of every registered experiment |
| [Experiment registry](../research/experiments.md) | Every experiment ID with its hypothesis, method and result, allocated before the run |
| [Hypotheses](../research/hypotheses.md) | The hypotheses, written before the experiments that test them |
| [Findings](../research/findings.md) | Conclusions supported by completed experiments, with their confidence |
| [Ablations](../research/ablations.md) | Controlled one-variable comparisons against named controls |
| [Failures](../research/failures.md) | Failed, abandoned and negative results |
| [Research log](../research/research_log.md) | The chronological notebook around the experiments |
| [Test-access log](../research/test_access_log.md) | Every evaluation on a held-out test split, written by the access check |
| [Approach survey](../research/approach_survey.md) | Methods and systems considered, and why this design was chosen |
| [Paper notes](../research/paper/) | Methodology, experiments, results, error analysis, efficiency and limitations |
| [Comparison table](../reports/benchmark_results/comparison_table.md) | Generated table of accuracy, latency and size by model and dataset |
| [Reports](../reports/README.md) | What each folder of generated reports holds |

## Maintainers

For contributing, running phases and making releases.

| Document | What it covers |
|---|---|
| [Contributing](../CONTRIBUTING.md) | Setup, the checks CI runs, the project rules for contributions and commit style |
| [Workflow](WORKFLOW.md) | Branches, commit messages, the phase gate, experiment IDs, test-set discipline |
| [Phase status](PHASE_STATUS.md) | The live record of every phase, with its gate and evidence |
| [Implementation plan](IMPLEMENTATION_PLAN.md) | The phases with their tasks, deliverables, exit criteria and deviations |
| [Roadmap](ROADMAP.md) | Milestones, versions, the dependency graph and success criteria |
| [Risk register](RISK_REGISTER.md) | Known risks, their triggers and mitigations |
| [Release runbook](RELEASE_RUNBOOK.md) | The release-day steps for v1.0, in order, and who does each |
| [Release evidence](RELEASE_EVIDENCE.md) | Each v1.0 release precondition mapped to its evidence |
| [Adviser brief](ADVISER_BRIEF.md) | The open questions on the weights' terms, kept as written; they have not been answered by an adviser |

## Decision records

Architecture decision records, one per decision; a changed decision gets a new record and both stay.
The [index](decisions/README.md) gives each one's status.

| ADR | Decision |
|---|---|
| [0001](decisions/ADR-0001-two-stage-pipeline.md) | Two-stage detector and recognizer instead of an end-to-end or vision-language model |
| [0002](decisions/ADR-0002-line-level-recognition.md) | Line-level detection and CTC recognition |
| [0003](decisions/ADR-0003-document-domain-synthetic-data.md) | A document-style synthetic data source (partly superseded by 0009) |
| [0004](decisions/ADR-0004-package-naming.md) | Lowercase Python package name |
| [0005](decisions/ADR-0005-evaluation-data-roles.md) | Fixed roles for every dataset: train, validation, test, external |
| [0006](decisions/ADR-0006-isolated-environments.md) | Separate virtual environments for the project and for the baselines |
| [0007](decisions/ADR-0007-pretrained-weights-policy.md) | Generic ImageNet backbones allowed; no pretrained OCR weights |
| [0008](decisions/ADR-0008-usable-engine-first.md) | A usable engine (v0.1) before the research studies; CPU-first inference |
| [0009](decisions/ADR-0009-no-synthtext.md) | No SynthText; every training source commercially usable |
| [0010](decisions/ADR-0010-gdc-capture-track.md) | The GDC capture as its own track |
| [0011](decisions/ADR-0011-cpu-latency-target.md) | The CPU latency target, accepted for v1 as a missed target |
| [0012](decisions/ADR-0012-detector-labels-and-operating-points.md) | Detector labels and unclip operating points |
| [0013](decisions/ADR-0013-gdc-scan-role.md) | GDC-scan as an internal test set, reported as a case study |
| [0014](decisions/ADR-0014-weights-and-backbone-terms.md) | Terms for the v1 weights; the ImageNet-initialised detector as a disclosed exception |

## How numbers are kept

The project's rule is that no number is typed into a document by hand: each figure is copied from a
generated report under [`reports/`](../reports/README.md) together with the experiment ID that
produced it ([Workflow §6](WORKFLOW.md#6-numbers)). CI checks
that the generated summaries are current and that the numbers in the model card, the dataset page
and the paper notes trace to reports ([Contributing](../CONTRIBUTING.md#running-the-checks)). An
archive made with `git archive`, such as the Zenodo deposit, leaves `reports/` out, so tracing a
number needs the tagged repository on GitHub.
