# Phase Status

Live record of every phase. Values are measured or quoted from generated reports, never invented.
The gate checklist is defined in [WORKFLOW.md §3](WORKFLOW.md#3-the-phase-gate).
Phases run in the order set by [ADR-0008](decisions/ADR-0008-usable-engine-first.md):
0–10 → 17 → 18 → 20 (v0.1) → 11–16 → 17/18/20 again → 19 → 21–23 (v1.0).

| Phase | Name | Status | Tag |
|---|---|---|---|
| 0 | Repository discovery and planning | **COMPLETE** | `phase-00` |
| 1 | Environment and reproducibility | **IN GATE** (awaiting CI) | — |
| 2 | Dataset acquisition | not started | — |
| 3 | Data audit | not started | — |
| 4 | Preprocessing, charset, GlyphSynth, GDC | not started | — |
| 5 | Baseline systems | not started | — |
| 6 | Recognition baseline | not started | — |
| 7 | Recognizer improvement | not started | — |
| 8 | Text detector | not started | — |
| 9 | Detector error analysis | not started | — |
| 10 | Full OCR pipeline | not started | — |
| 11 | Robustness training | not started | — |
| 12 | Synthetic data study | not started | — |
| 13 | Ablation study | not started | — |
| 14 | External evaluation | not started | — |
| 15 | Error analysis | not started | — |
| 16 | Efficiency optimisation | not started | — |
| 17 | Model export | not started | — |
| 18 | Inference package | not started | — |
| 19 | Optional HTTP API | not started | — |
| 20 | Production engineering | not started | — |
| 21 | Research package | not started | — |
| 22 | Final validation | not started | — |
| 23 | Release v1.0 | not started | — |

---

## Phase 00 — Repository discovery and planning

**Status:** COMPLETE (opened 2026-10-09, closed 2026-10-10)

**Objective:** Inspect the machine, toolchain and (empty) repository; research approaches and datasets;
produce the implementation plan, roadmap and research scaffolding. No training.

**Implemented**
- Environment inspection recorded in `docs/ENVIRONMENT.md` (GPU, driver, CUDA runtime, VRAM free at idle,
  bf16 support, CPU, RAM, disk, Python stack, missing tools).
- Literature, method and dataset survey: `research/approach_survey.md`, `research/paper/related_work.md`,
  `DATASETS.md`, `research/paper/references.bib`.
- Plan set: `docs/IMPLEMENTATION_PLAN.md`, `docs/ROADMAP.md`, `docs/ARCHITECTURE.md`,
  `docs/EXPERIMENT_PROTOCOL.md`, `docs/RISK_REGISTER.md`, `docs/WORKFLOW.md`, ADR-0001…0007.
- Research scaffolding: hypotheses, experiment registry, ablations, failures, findings, test access log,
  paper-section notes.
- Hypotheses H1–H8 pre-registered (`research/hypotheses.md`); research log started.
- ADR-0008 (2026-10-10): usable engine (v0.1) after Phases 10 → 17 → 18 → 20, CPU-first inference,
  English printed text only. Roadmap and plan updated.

**Tests:** none applicable (no code yet). Documentation reviewed for internal consistency (phase numbers,
file references, experiment-ID scheme); every relative link in the Markdown files checked by script and
resolves.

**Results (measured)**
- GPU: RTX 4050 Laptop, CC 8.9, 6.0 GiB total, **4.96 GiB free at idle**, bf16 supported.
- Torch 2.5.1+cu121 works with CUDA on driver 566.24.
- Tesseract and `gh` not installed; global Python env not suitable for reproducible work (322 packages).

**Known issues**
- The first session (2026-10-09) ended before part of the documentation was written and before anything
  was committed. The second session (2026-10-10) wrote the missing files, renamed `DATA/` to `data/`, and
  made the commits. The phase is recorded as closed on 2026-10-10.
- Dataset licences and published counts in `DATASETS.md` are unverified until Phase 2.
- Bibliography entries were written from memory and must be checked before Phase 21.
- No CI yet. CI arrives in Phase 1, so this phase's gate step 9 is N/A.

**Git:** commits on `main`, annotated tag `phase-00` on the final Phase 0 commit.

**CI status:** N/A (introduced in Phase 1).

**Research notes:** The main finding of the survey is a **domain gap**. The brief's primary training data
(HierText, SynthText) is scene-text-oriented, while the target is document pages. Addressed by ADR-0003
(GlyphSynth) and the GDC test set, and turned into RQ2/RQ6.

---

## Phase 01 — Environment and reproducibility

**Status:** IN GATE — implementation, tests and diagnostics done; waiting for push and CI (gate steps 8–10).

**Objective:** a reproducible, tested foundation that later phases build on.

**Implemented**
- `.venv` (Python 3.11.9) with torch 2.14.1+cu126 / torchvision 0.29.1; every other dependency pinned in
  `requirements/lock.txt`; `pyproject.toml` with ruff, mypy and pytest settings.
- `glypharis.utils`: `seed_everything` (Python, NumPy, torch, CUDA, `PYTHONHASHSEED`, cuDNN flags,
  deterministic algorithms with `CUBLAS_WORKSPACE_CONFIG`), DataLoader `worker_init_fn` and generator;
  `select_device` with CPU fallback; AMP policy bf16 → fp16 → fp32 and an fp32 region for CTC; resource
  snapshots (RAM, VRAM, peak VRAM, GPU temperature / clock / utilisation / power via NVML); git commit
  capture; structured JSON-lines logging.
- `glypharis.config`: YAML → typed dataclasses, `inherits:` merging, `${...}` references, `--config
  --seed --device --resume --set` CLI, strict validation of keys, types and values.
- `scripts/env_report.py` → `reports/environment/env_report.{md,json}` (no host names, user names or
  absolute paths); `scripts/validate_configs.py`; `configs/base.yaml`.
- CI: `.github/workflows/ci.yml` (Ubuntu, CPU torch of the same version): ruff check + format check →
  mypy → pytest → config validation → environment report as an artifact.

**Tests:** 67 passed locally, including the GPU-marked tests (`pytest`; ruff and mypy clean).

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| CPU determinism: two seeded runs, bit-identical losses for 50 steps | pass | `test_cpu_toy_training_is_bit_identical_for_50_steps`; env report "CPU determinism" |
| CUDA determinism within a documented tolerance | pass (tolerance 1e-4 relative, EXPERIMENT_PROTOCOL §11) | `test_cuda_toy_training_within_tolerance`; env report "CUDA determinism" |
| Config validator rejects an unknown key and a wrong type | pass | `tests/test_config.py` |
| `env_report.py` runs on the laptop and in CI | laptop: pass; CI: pending | `reports/environment/env_report.md` |
| CI green on `main` | pending | — |

**Results (measured):** see `reports/environment/env_report.md` — AMP (bf16, fp16), CTC in fp32 under
autocast, CPU and CUDA determinism checks all pass on the RTX 4050 Laptop with driver 566.24.

**Known issues**
- PyTorch warns that `ctc_loss_backward_gpu` has no deterministic implementation. The toy check showed
  no difference between runs, but CUDA reproducibility is only claimed within the stated tolerance.
- The GPU-marked tests are skipped in CI (no GPU runner); they run locally before every phase gate.
- ruff also formats Python code blocks inside Markdown files.

**Git:** branch `phase/01-environment`.

**CI status:** pending.
