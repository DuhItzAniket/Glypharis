# Architecture Decision Records

Short records of decisions that shape the project, written when the decision is made.
A decision is never edited away. If it changes, a new ADR supersedes it and both stay.

Format: `ADR-NNNN-short-name.md` with Status, Context, Decision, Consequences, and Revisit-if.

| ADR | Title | Status |
|---|---|---|
| [0001](ADR-0001-two-stage-pipeline.md) | Two-stage detector + recognizer instead of an end-to-end/VLM model | Accepted |
| [0002](ADR-0002-line-level-recognition.md) | Line-level detection and CTC recognition | Accepted |
| [0003](ADR-0003-document-domain-synthetic-data.md) | Add a document-style synthetic data source next to SynthText | Accepted |
| [0004](ADR-0004-package-naming.md) | Lowercase Python package name | Accepted |
| [0005](ADR-0005-evaluation-data-roles.md) | Fixed roles for every dataset: train / val / test / external | Accepted |
| [0006](ADR-0006-isolated-environments.md) | Separate virtual environments for the project and for baselines | Accepted |
| [0007](ADR-0007-pretrained-weights-policy.md) | Generic ImageNet backbones allowed; no pretrained OCR weights inside GLYPHARIS | Accepted |
| [0008](ADR-0008-usable-engine-first.md) | Ship a usable engine (v0.1) before the research studies; CPU-first inference | Accepted |
