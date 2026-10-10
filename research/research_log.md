# Research Log

Chronological notebook: what was done, what was observed, what was decided. Short entries, newest at
the bottom. Experiments get their full record in `experiments.md`; this log is the narrative around them.

---

### 2026-10-09 — Phase 0: discovery and planning

- Inspected the machine: RTX 4050 Laptop (6 GiB, ~4.96 GiB free at idle), bf16 supported, torch
  2.5.1+cu121 working on driver 566.24. Details in `docs/ENVIRONMENT.md`.
- Surveyed methods and datasets (`approach_survey.md`). Main observation: the brief's training data is
  scene-text-oriented while the target is document pages. Decided to add a document-style synthetic
  source (GlyphSynth, ADR-0003) and a frozen document test set (GDC).
- Wrote the implementation plan, roadmap, architecture, risk register and workflow.

### 2026-10-10 — Phase 0 closed; execution order revised

- Found that part of the Phase 0 documentation had not been written and nothing had been committed.
  Wrote the missing documents (experiment protocol, dataset cards, ADR-0001/0002/0003/0005, survey,
  hypotheses, related work, references).
- Owner decisions: English printed text only; deployment target (CPU vs GPU) undecided; no off-the-shelf
  stopgap in the consumer application, so a usable engine is needed early.
- Recorded ADR-0008: ship v0.1 after Phases 10 → 17 → 18 → 20, then run the studies; CPU-first
  inference. Phase numbers unchanged for traceability.
- Hypotheses H1–H8 written before any experiment (`hypotheses.md`).

### 2026-10-10 — Phase 1: environment

- Project `.venv` with torch 2.14.1+cu126 (driver 566.24 supports CUDA ≤ 12.7). Chose it over the global
  2.5.1+cu121 after the AMP, CTC-in-fp32 and determinism checks passed
  (`reports/environment/env_report.md`).
- Seeded CPU toy training (CNN-BiLSTM-CTC, 50 steps) is bit-identical across runs. On CUDA the same
  check also came out identical, although PyTorch warns that CTC backward on GPU has no deterministic
  kernel; the protocol keeps a 1e-4 relative tolerance rather than assuming exactness.
- CTC computed in an fp32 region under bf16 autocast stays finite with finite gradients.

### 2026-10-10 — Phase 2: licences and SynthText

- Checked every dataset's licence on its official page (DATASETS.md). SynthText's VGG terms are
  non-commercial, 12-month and no-redistribution, and the download links have been withdrawn. The consumer
  application's commercial status is undecided, so SynthText was dropped (ADR-0009). GlyphSynth becomes
  the only synthetic source; RQ2 and H2 reframed before any experiment (H2′).
- HierText test labels have been public since 2024-12, so the planned fallback test split is unnecessary.
- Research-only sets (FUNSD, OmniDocBench) and sets without a stated licence (IIIT5K, SROIE, DocVQA) are
  evaluation-only.
