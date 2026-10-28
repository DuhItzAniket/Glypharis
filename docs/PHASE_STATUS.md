# Phase Status

Live record of every phase. Values are measured or quoted from generated reports, never invented.
The gate checklist is defined in [WORKFLOW.md §3](WORKFLOW.md#3-the-phase-gate).
Phases run in the order set by [ADR-0008](decisions/ADR-0008-usable-engine-first.md):
0–10 → 17 → 18 → 20 (v0.1) → 11–16 → 17/18/20 again → 19 → 21–23 (v1.0).

| Phase | Name | Status | Tag |
|---|---|---|---|
| 0 | Repository discovery and planning | **COMPLETE** | `phase-00` |
| 1 | Environment and reproducibility | **COMPLETE** | `phase-01` |
| 2 | Dataset acquisition | **COMPLETE** | `phase-02` |
| 3 | Data audit | **COMPLETE** | `phase-03` |
| 4 | Preprocessing, charset, GlyphSynth, GDC | **COMPLETE** | `phase-04` |
| 5 | Baseline systems | **COMPLETE** | `phase-05` |
| 6 | Recognition baseline | **COMPLETE** | `phase-06` |
| 7 | Recognizer improvement | **COMPLETE** | `phase-07` |
| 8 | Text detector | **COMPLETE** | `phase-08` |
| 9 | Detector error analysis | **COMPLETE** | `phase-09` |
| 10 | Full OCR pipeline | **IN PROGRESS** (gate: push, CI, merge; after Phase 9) | — |
| 11 | Robustness training | not started | — |
| 12 | Synthetic data study | not started | — |
| 13 | Ablation study | not started | — |
| 14 | External evaluation | not started | — |
| 15 | Error analysis | not started | — |
| 16 | Efficiency optimisation | not started | — |
| 17 | Model export | **IN PROGRESS** (gate: push, CI, merge; after Phase 10) | — |
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

**Status:** COMPLETE (2026-10-10)

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
| `env_report.py` runs on the laptop and in CI | pass | `reports/environment/env_report.md`; CI artifact `env-report-ci` |
| CI green on `main` | pass | GitHub Actions run 38025056305 on merge commit `030049c` |

**Results (measured):** see `reports/environment/env_report.md` — AMP (bf16, fp16), CTC in fp32 under
autocast, CPU and CUDA determinism checks all pass on the RTX 4050 Laptop with driver 566.24.

**Known issues**
- PyTorch warns that `ctc_loss_backward_gpu` has no deterministic implementation. The toy check showed
  no difference between runs, but CUDA reproducibility is only claimed within the stated tolerance.
- The GPU-marked tests are skipped in CI (no GPU runner); they run locally before every phase gate.
- ruff also formats Python code blocks inside Markdown files.

**Git:** branch `phase/01-environment`, merged through pull request #1. Merge commit `030049c`, tag
`phase-01`. (The merge commit was re-created once with the noreply identity; same tree and parents.)

**CI status:** green on the branch (run 38024668422) and on `main` (run 38025056305). The GPU-marked tests
are skipped in CI and passed locally.

---

## Phase 02 — Dataset acquisition

**Status:** COMPLETE (2026-10-10)

**Objective:** reproducible download and provenance for every dataset.

**Implemented**
- `data/sources.yaml` (source registry, validated by `glypharis.data.sources`).
- `glypharis.data.download`: resumable HTTP download via `.part` files and Range requests, retries,
  SHA-256 verification (published hash, or the hash recorded on first download), safe archive extraction
  with an idempotent marker, free-disk check with a 20 GiB margin, per-dataset manifest written only when
  its content changes.
- `glypharis.data.counters` (measured counts) and `glypharis.data.registry` →
  `data/DATASET_MANIFEST.yaml` (brief §13 fields).
- `scripts/download_datasets.py` (`--list`, `--dry-run`, `--files`, `--verify`).
- Licence check of every candidate dataset on its official page, recorded in `DATASETS.md`.
- ADR-0009: SynthText dropped; delivered models train only on commercially usable data.

**Descoped (documented):** external test sets (IIIT5K, FUNSD, OmniDocBench, SROIE) are registered and
licence-checked but downloaded in Phase 14, when they are used; GlyphSynth inputs (Google Fonts at a
pinned commit, PG-19 text) are added to the registry in Phase 4, where the font list and text sample are
designed. SynthText is not downloaded (ADR-0009).

**Tests:** 92 passed locally (downloader tested against a local HTTP server with Range support: resume,
server ignoring Range, size and hash mismatch, corruption detected by `--verify`, path traversal, disk
check, no-op rerun, manifest content).

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| Every downloaded dataset has a manifest with file counts and hashes | pass | `data/manifests/hiertext.json`, `data/DATASET_MANIFEST.yaml` |
| A re-run of the downloader is a no-op | pass | second run and `--verify` run: manifest content and mtime unchanged, nothing fetched |
| DATASETS.md has no "TBD" for downloaded sets | pass | HierText card has measured counts |
| CI green; downloader unit-tested against a fake source | pass | `tests/test_download.py`; GitHub Actions run 38059995516 on `main` |

**Results (measured):** HierText 11,639 images (8,281 / 1,724 / 1,634), counts as in DATASETS.md;
3.72 GiB of archives plus 79 MiB of ground truth. Ground-truth files match GitHub's git blob SHA-1.

**Known issues**
- S3 throughput was 0.3–0.8 MB/s per connection from this machine; the downloader uses one connection.
- Counting walks all extracted images and parses the ground truth, so each run takes about a minute.

**Git:** branch `phase/02-datasets`, merged through pull request #2. Merge commit `937206b`, tag
`phase-02`. (The merge commit was re-created with the noreply identity; same tree and parents.)

**CI status:** green on the branch (run 38059683845), the pull request, and `main` (run 38059995516).

---

## Phase 03 — Data audit

**Status:** COMPLETE (2026-10-10)

**Objective:** know the data before training on it.

**Implemented**
- `glypharis.data.hiertext`: typed reader with stable line ids (`<image>/p<n>/l<n>`), reused in Phase 4.
- `glypharis.data.audit`: line checks (polygon, area, inside fraction, out-of-range, empty legible text,
  invalid Unicode, NFC, tiny and elongated lines, line/word text mismatch), image statistics at a fixed
  1024 px long side (brightness, contrast, Laplacian-variance blur), dHash + pHash, exact / near
  duplicate search, **text-overlap leakage search**, flag catalogue with severity and decision.
- `scripts/audit_data.py`: statistics (`stats.json`, `chars.json`), plots, flags CSV, duplicate and
  text-overlap contact sheets, `reports/data_quality_report.md`, versioned exclusion list
  `data/manifests/exclusions/hiertext.json` (v1); per-image measurements cached in
  `data/interim/audit/`.

**Tests:** 113 passed locally (21 new: reader ids, geometry, every line flag, hash robustness to
resizing, corrupt images, duplicate priority, text-overlap rules, end-to-end run with an exclusion list
that stays at the same version when nothing changes).

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| Quality report for every downloaded dataset | pass | `reports/data_quality_report.md` (HierText) |
| Zero unexplained exclusions | pass | every exclusion carries a flag code with a documented decision |
| Zero cross-split near-duplicates left unhandled | pass | 4 hash pairs + 87 text-overlap pairs, lower-priority copy excluded; contact sheets checked by eye |
| Plots committed | pass | `reports/data_analysis/hiertext/*.png`, `reports/data_quality/*.png` |
| CI green | pass | GitHub Actions run 38063919422 on `main` |

**Results (measured, `reports/data_quality_report.md`):** 68 train and 9 validation
images excluded for duplication / text overlap; 3 train lines excluded (broken boxes); 1 test line
excluded from recognizer use (empty legible text); 5,632 lines to clip; 30 lines with invalid characters
to normalise. Median legible line: 1 word, 7 characters,
14 px high.

**Known issues**
- Text-overlap thresholds were chosen by inspecting HierText pairs (an audit heuristic, not model
  selection); borderline pairs were checked by eye and all were genuine overlaps.
- HierText line boxes are mostly single words, so long-line recognition needs GlyphSynth (R-06).

**Git:** branch `phase/03-data-audit`, merged through pull request #3. Merge commit `7cb784d`, tag
`phase-03` (re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 38063919422).

---

## Phase 04 — Preprocessing, charset, synthetic data, GDC

**Status:** COMPLETE (2026-10-21). GDC capture ran as the tracked item **GDC-capture** (ADR-0010) and
finished inside the phase: GDC v1 was frozen before the merge.

**Objective:** training-ready data with documented transformations.

**Implemented**
- `glypharis.data.text`: label normalisation v1 (original and normalised text both kept), evaluation
  tracks, `Charset`; `scripts/build_charset.py` → `data/manifests/charset.json` (274 characters + blank),
  `reports/charset/charset_report.md`.
- `glypharis.data.pages` (internal page format, grouped by document), `crops` (rectified line crops,
  64 px, 0.15 margin), `linepack` (binary pack + index, Windows-safe); `scripts/prepare_hiertext.py`
  applies the Phase 3 exclusion list and packs train and validation crops.
- `glypharis.data.augment`: presets none / geometric / photometric / degradation / full with
  per-transform probability and severity (config `data.augmentation`, `data.augmentation_params`),
  and the fixed GLYPH-C corruptions; previews in `reports/data_analysis/`.
- GlyphSynth v1 (`glypharis.data.synth`, `scripts/build_glyphsynth.py`) from 100 pinned Google Fonts
  families and PG-19 text, with held-out fonts and texts for validation.
- GLYPH-C v1 frozen (`scripts/build_glyph_c.py`, `data/manifests/processed/glyph_c.json`).
- GDC tooling (`glypharis.data.gdc`, `scripts/gdc_select_pages.py`, `gdc_print_pack.py`,
  `gdc_align.py`), 21 source documents, 106-sheet print pack, capture checklist, training exclusion
  list.
- Downloader: published MD5 support and parallel transfers.

**Descoped / moved (documented):** GDC capture and freeze → GDC-capture track (ADR-0010).
Detector-side (page-level) augmentation → Phase 8. Real-scan GDC slice (owner's own scans) → part of
GDC-capture.

**Tests:** 233 passed locally (CPU and GPU-marked).

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| Round-trip test (convert → read back → identical text and boxes) | pass | `test_pages_round_trip_is_exact_and_deterministic`; linepack round trip |
| Crops visually spot-checked | pass | `reports/data_analysis/hiertext/line_crops.png`, `glyphsynth/samples.png` |
| Charset covers ≥ 99.9% of validation characters | pass on in-scope characters (99.943%); 99.809% of all characters, the rest being letters of other scripts excluded by the English-only scope | `reports/charset/charset_report.md` |
| GlyphSynth reproducible from seed | pass | validation pack rebuilt byte-identical; HierText packs likewise |
| GDC frozen with manifest | pass: GDC v1 frozen 2026-10-21 through the GDC-capture track (ADR-0010) | `data/manifests/gdc/gdc_v1.json` |
| CI green | pass | GitHub Actions run 39625823839 on `main` |

**Known issues**
- HierText labels contain some noise (visible on the crop sheet); recorded, not corrected.
- GDC text-layer ground truth cannot cover text inside raster images (handled as ignore regions) and
  merges some side-by-side form headers into one line.

**Git:** branch `phase/04-preprocessing`, merged through pull request #4. Merge commit `b2e3053`, tag
`phase-04` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 39625823839).

### Tracked item: GDC-capture (ADR-0010)

| Step | Owner | Status |
|---|---|---|
| Print `data/processed/gdc/print_pack.pdf` (A4, 100 % scale) | owner | done |
| Capture all 106 sheets per `data/manifests/gdc/capture_checklist.md` into `data/raw/gdc/captures/` | owner | done (2026-10-21) |
| Run `scripts/gdc_align.py`, check overlays, request retakes | maintainer | done: 106/106 aligned, no retakes; all overlays checked |
| Freeze (`scripts/gdc_align.py --freeze`) | maintainer | **done: GDC v1 frozen 2026-10-21** (`data/manifests/gdc/gdc_v1.json`) |
| Real-scan slice | owner + maintainer | moved to a separate set, **GDC-scan v1**, built and frozen on its own when the owner's scans arrive |

Alignment (`data/manifests/gdc/alignment.json`): marker homography refined by smooth optical flow,
because the curved captures were off by up to a line height with the homography alone. Refined
sheet/photo agreement: median NCC 0.924, minimum 0.757. 200 of 8,606 lines (2.3 %) have an alignment
score below 0.3 (box not trusted; text ground truth unaffected), mostly small figure text in distant
shots.

---

## Phase 05 — Baseline systems

**Status:** COMPLETE (2026-10-25).

**Objective:** scientific reference points before training anything.

**Implemented**
- `.venv-baselines` (ADR-0006) with Tesseract 5 (UB Mannheim build), PaddleOCR 3.7 (PP-OCRv5 mobile
  and server, PP-OCRv6 tiny, small and medium) and TrOCR-small-printed.
- `scripts/run_baselines.py`: resumable prediction runs in the GLYPHARIS output schema, run metadata
  written first, every test access logged before an image is read; `--profile` measures speed and
  memory (batch 1, 10 warm-up and 100 timed samples, run conditions recorded).
- `scripts/build_eval_crops.py`: evaluation line packs that keep lines with characters outside the
  GLYPHARIS charset (the charset ceiling is reported next to every result).
- `scripts/evaluate_baselines.py`: scores everything with `glypharis.eval`, the code that will score
  GLYPHARIS; both normalisation tracks, bootstrap CIs, one table per comparison type, GDC's vertical
  lines on their own, and the coverage of the planned run matrix.
- `scripts/baseline_diagnostics.py`: validation-only diagnostics behind three results.
- Seven baselines registered before running and run: GW-BASE-001 to GW-BASE-007.

**Results** (strict track; from `reports/baselines/`, where every number has its CI and ID)

| ID | System | HierText val lines, CER % | GDC test lines, CER % | HierText val pages, E2E F1 % | ms per line | ms per page |
|---|---|---|---|---|---|---|
| GW-BASE-001 | Tesseract 5 | 24.35 | 6.42 | 12.65 | 141.8 | 430.8 |
| GW-BASE-002 | PP-OCRv5 mobile | 9.27 | 3.68 | 38.75 | 61.4 | 3,408.2 |
| GW-BASE-003 | PP-OCRv5 server | 14.65 | 11.24 | not run | 2,266.7 | 80,861.6 |
| GW-BASE-004 | TrOCR-small-printed | 62.58 (19.14 normalised) | 72.34 (13.07 normalised) | no detector | 133.3 | — |
| GW-BASE-005 | PP-OCRv6 tiny | 11.10 | 4.00 | 38.69 | 21.6 | 1,726.6 |
| GW-BASE-006 | PP-OCRv6 small | 7.53 | 3.46 | 45.68 | 46.0 | 3,714.2 |
| GW-BASE-007 | PP-OCRv6 medium | 7.54 | 3.20 | not run | 552.5 | 32,528.5 |

- PP-OCRv6 small is the most accurate full-scope system on every set. PP-OCRv6 medium (PaddleOCR's
  English default) matches it on HierText lines at about twelve times the line time.
- Every system is weak on HierText pages, mostly from detection: the best recall at IoU 0.5 is
  50.67% (GW-BASE-006). Tesseract finds about one HierText line in ten.
- On GDC, Tesseract has the best reading order (order-aware page CER 31.90% against 42.86–44.74%
  for the PaddleOCR pipelines, which return lines in detection order).
- PP-OCRv6 medium and PP-OCRv5 server are impractical on a laptop CPU: 32.5 s and 80.9 s per page.
- Three results are explained by `reports/baselines/diagnostics.md`: PP-OCRv5 server drops word
  gaps (WER three to four times its CER), PP-OCRv5 mobile cuts long lines short (its GlyphSynth CER),
  and TrOCR-small-printed writes capitals (its strict-track CER).

**Not run (documented with the reason):** PP-OCRv5 server and PP-OCRv6 medium ran at a reduced
scope registered before the runs (about 1–3 s per line on this CPU); the PP-OCRv5 server GDC page run
is not runnable on the laptop (`research/failures.md`, 2026-10-24); TrOCR has no detector. Peak VRAM
is not applicable: no baseline uses a GPU.

**CPU latency target (ADR-0008):** set in ADR-0011 from these measurements and accepted by the
owner on 2026-10-25. For v0.1 through ONNX Runtime on this laptop: line recognition mean at most
25 ms (p95 70 ms), pages mean at most 1.5 s (p95 2.7 s), peak RAM on pages below PP-OCRv6 tiny's
1,836 MiB.

**Tests:** 264 passed locally (CPU; the 4 GPU tests were deselected while the GPU was training).

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| One table per comparison type in `reports/baselines/` | pass: recognition, detection and end to end, plus efficiency | `reports/baselines/*.md` |
| Normalisation track stated | pass: strict and normalised tables; detection does not depend on the track | same |
| Every number traceable to a `GW-BASE-NNN` ID | pass: every row carries its ID; every number is in `results.json` | `reports/baselines/results.json` |
| Baselines that could not be run documented with the reason | pass | coverage table in `reports/baselines/README.md`; `research/failures.md` |
| CER, WER, NED, accuracy, mean and p95 latency, peak RAM, peak VRAM | pass (peak VRAM not applicable) | `recognition.md`, `efficiency.md` |
| Test sets used only as logged | pass: every prediction and scoring access logged | `research/test_access_log.md` |
| CI green | pass | GitHub Actions runs 40130891767 (branch) and 40130913098 (pull request); `main` after the merge: run 40132573309 |

**Known issues**
- The profiling pass ran on the Windows Balanced power plan (recorded in `efficiency.md`); GLYPHARIS
  must be timed the same way.
- On its largest pages PP-OCRv5 server needed more memory than the laptop had free, so some of its
  page times include paging to disk. Windows was also holding an unusually large amount of kernel
  memory during the pass (a leak whose cause was not found), which left less RAM free.
- The PP-OCRv5 detectors and the v6 detectors run without oneDNN, which crashes on them here (speed
  only; outputs checked identical).

**Git:** branch `phase/05-baselines`, merged through pull request #5. Merge commit `cdc823c`, tag
`phase-05` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main`.

---

## Phase 06 — Recognition baseline (GLYPHARIS-Rec, CRNN)

**Status:** COMPLETE (2026-10-25). Run in parallel with Phase 5 (owner approved, 2026-10-22) and
merged after it.

**Objective:** a first stable recognizer with a fully instrumented training loop.

**Implemented**
- `glypharis.models.rec.crnn`: CNN → 2-layer BiLSTM → linear → CTC (ARCHITECTURE §4); 7,796,435
  parameters, 29.8 MiB of weights and buffers. The BiLSTM reads each line only up to its own end
  (packed sequences), and its layers run as separate single-layer BiLSTMs with ordinary dropout
  between them (EXPERIMENT_PROTOCOL §11, 2026-10-23).
- `glypharis.training.rec_trainer`: bf16 autocast with CTC in fp32, gradient accumulation and
  clipping, OneCycle and cosine schedules, a width-bucketed sampler with a cap on columns per batch,
  and augmentation that depends only on seed, epoch and index. Checkpoints (`best.pt`, `last.pt`,
  `epoch_XXX.pt`, mid-epoch `last.pt`) carry model, optimizer, scheduler and scaler state, position,
  history, config, git hash and every RNG state, so a run resumes exactly. `best.pt` holds the lowest
  validation CER; `min_delta` only drives early stopping.
- `glypharis.diagnostics`: overfitting (revised rule, §11), underfitting, blank and prediction
  collapse, gradient explosion and vanishing. Hard stops on a non-finite loss or gradient, with the
  state and batch keys saved; an OOM record; and the train/validation leakage assertion.
- `glypharis.training.report`: curves, best vs final epoch, flags, bootstrap CI, worst lines and
  notes, built only from what the run leaves on disk.
- `train.py` (exit codes 0 / 2 non-finite / 3 out of memory). Scripts: `rec_sanity.py` (overfit 64
  lines, LR range test, 10-minute dry run), `rec_resume_check.py`, `rec_reselect.py`,
  `rec_report.py`, `rec_compare.py`.

**Descoped / moved (documented):** on out-of-memory, the plan's "halve the batch" option is not
implemented. The run takes the other permitted path: it saves `oom.pt`, writes the report and exits
with 3.

**Tests:** 293 passed locally (CPU and GPU-marked). With Phase 5 merged in, 290 CPU tests pass; the 7
GPU tests were deselected while the GPU was training a Phase 7 run.

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| All sanity checks pass and are recorded | pass: overfit 64 lines to 0.39% CER (target ≤ 1%); LR range test, configured 1e-3 within 3x of the suggestion; 10-minute dry run, peak VRAM 3825.5 MiB, loss finite | `reports/training/sanity/GW-REC-001/sanity.md` |
| Resume test: kill at step N, resume, metrics match an uninterrupted run within tolerance | pass. CPU: identical (unit test). CUDA: first step after the resume identical, learning rates identical, drift 1.51e-2 against 2.90e-2 between two uninterrupted runs. The first CUDA attempt failed a tolerance that CUDA cannot meet; that result stays recorded | `reports/training/sanity/GW-REC-001/resume_check.json`, `sanity.md`; EXPERIMENT_PROTOCOL §11 |
| First full run finishes without NaN | pass | `reports/training/GW-REC-001/report.md` |
| Run report generated automatically | pass | `reports/training/GW-REC-001/`, `GW-REC-002/` |
| Validation CER with a bootstrap CI | pass: GW-REC-001 7.53% [7.38, 7.69]; GW-REC-002 7.49% [7.34, 7.64] | run reports; `reports/training/compare/GW-REC-002_vs_GW-REC-001.md` |
| Parameters, model size and training time recorded | pass: 7,796,435 parameters, 29.8 MiB; 4.71 and 5.32 GPU-hours | run reports |
| CI green, with a tiny CPU training smoke test | pass: GitHub Actions runs 40132754557 (branch) and 40132758219 (pull request) | `test_training_smoke_writes_metrics_checkpoints_and_summary` |

**Known issues**
- Two CUDA runs of one config drift apart because CTC backward has no deterministic kernel. Two
  uninterrupted runs differed by up to 2.90% in per-step loss over 400 steps.
- GW-REC-002's `best.pt` was re-selected (epoch 24 → 27) and its config hash corrected after the
  run. Both are recorded in its report's notes and in §11.
- GW-REC-001 and GW-REC-002 aborted at exit (0xC0000409) after writing their outputs. The cause is
  fixed in `c58c063`, and the results are unaffected.
- On this laptop `CUDA_VISIBLE_DEVICES=` (empty) does not hide the GPU, and `-1` crashes torch a few
  seconds later; the Phase 8 build found this. GPU-free test runs need another way to keep the GPU
  out.
- HierText annotation noise shows up in the worst-line tables. It is recorded, not corrected, and
  left for Phase 7's error analysis to count.

**Git:** branch `phase/06-rec-baseline` (Phase 5 merged in first), merged through pull request #6.
Merge commit `608ee3f`, tag `phase-06` (merge re-created with the noreply identity; same tree and
parents).

**CI status:** green on the branch, the pull request and `main` (run 40133080942).

---

## Phase 07 — Recognizer improvement

**Status:** COMPLETE (2026-10-27). Opened in parallel with Phases 5 and 6 at the owner's request
(2026-10-23), on `phase/07-rec-improvement`.

**Objective:** improve the recognizer under controlled conditions and settle its architecture and
hyperparameters. Validation data only.

**Plan.** About 4.7 GPU-hours per full run, so Phase 7 ranks configurations on a short budget and
confirms the winner on the full one:
1. Control (GW-REC-003, 004, 005): GW-REC-001's configuration for 8 epochs (OneCycle over those 8
   epochs), seeds 1337, 2024 and 7. Their spread is the noise floor.
2. One-variable ablations against that control, each with an ID and expectation registered before it
   runs: input height 48; a narrower CNN; BiLSTM size; coarser sequence downsampling; optimizer,
   schedule, learning rate and weight decay (at most 12 short runs together); augmentation presets;
   and the SVTR-style encoder, GLYPHARIS-Rec-M (H6), at a matched step budget, with CPU latency
   measured. Each arm is compared line by line with the control (`scripts/rec_compare.py`) and read
   against the seed spread.
3. The chosen configuration at the full budget with three seeds against GW-REC-001's configuration,
   which gives the evidence table.

The optional TrOCR-Small fine-tune comes last and only if the budget allows.

**Decision rule (fixed 2026-10-23, before any ablation ran).** Each arm is paired line by line with
GW-REC-003, the control with the same seed. It counts as **better** only if the 95% CI of its CER
difference lies below zero *and* its CER is below all three control seeds. It counts as **worse**
only if that CI lies above zero *and* its CER is above all three. Otherwise it shows no clear
difference. The table states how many comparisons were made; no multiplicity correction is
applied (EXPERIMENT_PROTOCOL §4). The combination step only takes changes judged better, plus
changes with no clear difference that make the model smaller or faster. The evidence table is
`reports/training/phase07/ablation_table.md` (`scripts/rec_ablation_table.py`).


**Combination step (registered 2026-10-25, before it ran).** Two changes were judged better on the
short budget: input height 48 (GW-REC-006) and the SVTR-style encoder (GW-REC-020); GW-REC-014
(weight decay 1e-2) finished later and was judged better too. GW-REC-022 combines all three at the
same budget and seed (amended before it ran). GW-REC-023, registered while GW-REC-022 ran and
before its result, is the same without the encoder change (CRNN, input height 48, weight decay
1e-2): the cheaper candidate, and the run that shows whether the SVTR encoder still adds anything. The changes with no clear difference so far (geometric
augmentation only, learning rate 2e-3) make the model neither smaller nor faster, so the rule leaves
them out; GW-REC-014 (weight decay) and GW-REC-021 (frame stride 8, which would make it faster) are
still running and are added under the same rule if they qualify. Every candidate is timed on the
CPU against ADR-0011's line target (`scripts/rec_profile.py`, `reports/training/phase07/cpu_latency.md`)
before one configuration is chosen for the full-budget confirmation.

**Results so far (2026-10-25).** All 15 short-budget arms are done (evidence table and
`research/ablations.md`). Better: input height 48 (GW-REC-006, −0.49 points), the SVTR-style encoder
(GW-REC-020, −0.13) and weight decay 1e-2 (GW-REC-014, −0.04). No clear difference: geometric
augmentation only (GW-REC-016) and learning rate 2e-3 (GW-REC-013). Worse: the other ten, the
largest loss being frame stride 8 (GW-REC-021, +2.61). Every architecture tried meets ADR-0011's
line target in PyTorch on the CPU, from 6.3 ms (GW-REC-007) to 15.7 ms (GW-REC-006) per line
(`reports/training/phase07/cpu_latency.md`). Compute so far: 30.0 GPU-hours (`reports/compute/ledger.md`).
Open: GW-REC-022 and GW-REC-023, then the choice of configuration and its full-budget confirmation.

**Choice and confirmation (2026-10-26).** GW-REC-022 (SVTR-style encoder, input height 48, weight
decay 1e-2) reached 7.56% on the short budget, −0.76 points against the control and −0.27 against
input height 48 alone (`reports/training/compare/GW-REC-022_vs_GW-REC-006.md`); it meets ADR-0011's
line target at 22.8 ms mean and 51.5 ms p95 per line on the CPU. It is the chosen configuration.
GW-REC-023 was abandoned before it ran (GW-REC-022 answered its question). GW-REC-024 confirms the
choice at the full budget with **one seed** instead of the planned three, to stay within the Phase 7
budget; this deviation is recorded in its registry entry. The owner decided (2026-10-26) to add
the two missing seeds later, on a college A100 if access is granted, before the paper.

**Chosen configuration.** GLYPHARIS-Rec-M (the SVTR-style encoder) at input height 48 with weight
decay 1e-2, trained with GW-REC-001's recipe otherwise (`configs/rec/gw-rec-024.yaml`). At the full
budget (GW-REC-024) it reaches 6.46% validation CER [6.33, 6.60], −1.07 points [−1.13, −1.01]
against GW-REC-001, with 7,928,835 parameters, and reads a line in 24.4 ms mean, 60.3 ms p95 on the
CPU (PyTorch), inside ADR-0011's line target with little margin on the mean. One seed; the owner
decided (2026-10-26) to add the two missing seeds later, on a college A100 if access is granted.
The A100 slot is still booked (2026-10-27), so the seeds wait; nothing before the paper depends on them.

**Implemented**
- `glypharis/models/rec/svtr.py` (GLYPHARIS-Rec-M), `glypharis/models/rec/base.py`, the CRNN frame
  stride option, CTC-aware batching (`data.ctc_aware_batching`), config fields that keep earlier
  config hashes unchanged at their defaults.
- `scripts/rec_ablation_table.py` (evidence table with the decision rule), `scripts/rec_profile.py`
  (CPU latency like the baselines), `scripts/rec_memory_probe.py` (worst-case batches before a run),
  `scripts/compute_ledger.py` (`reports/compute/ledger.md`).
- Runs GW-REC-003 to 024: three control seeds, 15 one-change ablations, two combinations, one
  full-budget confirmation; GW-REC-019 failed (out of memory) and was re-run as 021; GW-REC-023 was
  abandoned before it ran.

**Tests:** 314 passed locally (CPU; the 7 GPU tests passed on 2026-10-25 and were deselected now
because the GPU was training the detector).

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| A chosen recognizer configuration with the evidence table | pass: GW-REC-022's recipe, confirmed at the full budget by GW-REC-024 | `reports/training/phase07/ablation_table.md`, `research/ablations.md`, `reports/training/compare/GW-REC-024_vs_GW-REC-001.md` |
| All runs have diagnostic reports | pass: every run's report has its diagnostics section (no flags raised in Phase 7) | `reports/training/GW-REC-0*/report.md` |
| Negative results recorded | pass: ten arms judged worse, the stride-8 OOM, the abandoned GW-REC-023 | `research/ablations.md`, `research/failures.md`, `research/experiments.md` |
| CI green | pass | runs 40363336641 (branch), 40363358399 (pull request), 40365154402 (`main`) |

**Compute:** 55.3 GPU-hours in Phase 7, inside the plan's 30–60 (`reports/compute/ledger.md`; the
hours of jobs that did not finish include the parts of crashed runs that later resumed).

**Known issues**
- The chosen recognizer's CPU latency (24.4 ms mean per line in PyTorch) is close to ADR-0011's 25 ms
  target; the Phase 17 ONNX export has to keep it inside.
- One seed at the full budget (deviation recorded in GW-REC-024's entry).
- On this laptop, training crashed with CUDA host-allocation and "unknown" errors under memory
  pressure; two data-loading workers and the resume path kept every run going.
- The optional TrOCR-Small fine-tune was not run (budget).

**Git:** branch `phase/07-rec-improvement`, merged through pull request #7. Merge commit `9efb1f9`,
tag `phase-07` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 40365154402).

---

## Phase 08 — Text detector (GLYPHARIS-Det)

**Status:** COMPLETE (2026-10-27). The code was built in parallel with Phase 7 (owner approved, 2026-10-23) on `phase/08-detector`, which now
includes `main` with Phases 5 to 7. The runs used the GPU only after Phase 7's runs had finished.

**Objective:** a lightweight line detector, trained and evaluated on its own.

**Implemented**
- Config: one optional `det:` section (`DetConfig` in `glypharis.config.schema`) for data,
  augmentation, labels, model, loss, inference, post-processing and diagnostics. For `task: det` the
  recognizer's `data` and `model` sections must keep their defaults and are left out of the config
  hash, so the recognizer hashes did not move (pinned by a test). A missing required key is a config
  error, not a crash. Configs `configs/det/gw-det-001.yaml` to `003`.
- The MobileNetV3 Large and Small ImageNet weights, registered as a pinned source
  (`data/sources.yaml`, DATASETS.md): size and SHA-256 are checked before loading, and nothing is
  downloaded at run time.
- Data (`glypharis.data.det_labels`, `det_geometry`, `det_augment`, `det_dataset`): DBNet shrink and
  threshold maps sampled exactly at pixel centres, with the shrink done by a shapely mitre offset
  instead of pyclipper; don't-care lines handled as the evaluator handles them; lines too small at
  the training scale, or with too thin a kernel, become don't care in that sample; one affine warp
  per crop, deterministic per seed, epoch and index; a memory-mapped page index for spawned workers;
  start-up guards against test or GDC pages, a page-file hash check and a train/validation leakage
  check.
- Model (`glypharis.models.det`): DBNet on MobileNetV3 features (FPN neck at 1/4 resolution,
  probability and threshold heads with their final layers in fp32), normalisation inside the model
  on uint8 RGB input, and the SE pooling replaced by a plain mean so that its CUDA backward is
  deterministic. The DB loss (BCE with sort-based OHEM 3:1, Dice, masked L1) runs in fp32.
- Post-processing (`postprocess.py`, no torch): connected components scored by their mean
  probability, two pre-registered unclip operating points (`dbnet15`, `inverse10`) from one forward
  pass, `quad` (default) or `poly` box output, and a text-direction rule for the TL/TR/BR/BL quad.
  `TextLineDetector` is the API for Phases 10, 17 and 18, with a release-file export.
- Evaluation (`glypharis.eval.det_matching`, `det_slices`, `det_overlay`): per-box match details for
  merge and split counts; pruned matching that gives the same counts as `evaluate_page` at bounded
  cost (a test checks the parity); the Phase 5 baseline page set; the five plan slices, defined
  before any result; overlays. `glypharis.eval.stats` gains a paired page-bootstrap CI for the
  difference of two F1 scores.
- Training (`glypharis.training.loop`, `det_trainer`, `det_eval`, `det_report`): a task-agnostic
  loop with exact resume, which the recognizer moves onto after Phase 7 merges; the detector
  trainer, with a GPU headroom check at start-up (exit 4 when another process holds the GPU),
  validation on a seeded subset every epoch and full validation of `best.pt` and `last.pt` at the
  end; a run report written on every exit path, with a failure banner, CIs, curves, slices, overlays
  and the label ceiling. `glypharis.diagnostics.det` adds the detection flags (EXPERIMENT_PROTOCOL
  §6).
- Scripts: `det_sanity.py` (label ceiling, overfit one batch, LR range test, dry run),
  `det_resume_check.py`, `det_eval.py`, `det_compare.py`, `det_preview.py`, `det_report.py`.
  `train.py` runs `task: det`.

**Plan** (registered 2026-10-25, before any GPU run)
1. Sanity sequence under the planned IDs: label ceiling (once, CPU only), the label preview, overfit
   one batch (GW-DET-001), LR range test (001 and 002), a 10-minute dry run with its VRAM gate for
   each of 001, 002 and 003, and the resume check (001). Reports go to
   `reports/training/sanity/<ID>/`.
2. Freeze: the pre-declared decision rules in the GW-DET-001 entry of `research/experiments.md` turn
   the measurements into `epochs` and any fallback, the same for all three configs. A dated
   amendment to each entry cites its sanity report and final config hash before any main run starts.
3. Main runs GW-DET-001, 002 and 003 at equal step budgets.
4. `scripts/det_compare.py` at each arm's stored operating point: H9 (001 vs 002) and H10 (001 vs
   003), into `research/ablations.md`.

**Sanity sequence and freeze (2026-10-26).** Label ceiling F1@0.5 0.948, overfit one batch and the
resume check passed (`reports/training/sanity/`). The pre-declared rules then changed the plan
before any main run, in a dated amendment to each registry entry: learning rate 1e-3 → 5e-3 (the
range test suggested 6.0e-3), 24 epochs, a 400-page validation subset per epoch instead of 800
(validation took 32–37% of an epoch), and 4 loader workers instead of 6 (peak RSS 8.13 GB against
an 8 GB gate). Final config hashes `ebd82071f0a36782`, `26340926cb03c343`, `2ba0a429ed5ba49c`.

**Results (2026-10-27).** All three runs finished their 24 epochs without a crash. On all 1,715
HierText validation pages, each `best.pt` at its stored operating point (`inverse10` for all three):

| Run | Backbone | Init | Parameters (inference) | F1@0.5 [95% CI] | Precision | Recall | Δ vs GW-DET-001 [95% CI] |
|---|---|---|---|---|---|---|---|
| GW-DET-001 | MobileNetV3-Large | ImageNet | 3,187,209 | 0.724 [0.708, 0.739] | 0.924 | 0.595 | — |
| GW-DET-002 | MobileNetV3-Large | random | 3,187,209 | 0.704 [0.689, 0.719] | 0.911 | 0.574 | −0.020 [−0.023, −0.016] |
| GW-DET-003 | MobileNetV3-Small | ImageNet | 1,096,953 | 0.668 [0.651, 0.685] | 0.914 | 0.527 | −0.056 [−0.062, −0.051] |

- H9 (ImageNet initialisation helps) is supported, exploratory with one seed; H10 (Small within 0.03
  of Large) is not: the whole CI lies beyond the margin. `research/hypotheses.md`,
  `research/ablations.md`.
- **Chosen detector:** GW-DET-001, MobileNetV3-Large from ImageNet weights, `inverse10`, `quad`
  boxes (ADR-0012). On the 400 Phase 5 validation pages it reaches F1@0.5 0.726 [0.688, 0.760],
  against 0.579 [0.544, 0.614] for the best baseline detector there (PP-OCRv6 small); only
  GLYPHARIS-Det was trained on HierText's line definition, so part of that gap is the definition.
- Its errors are misses (finding F5): recall 0.20 for lines under 8 px high and 0.18 beyond 45° of
  rotation, against 0.72 for 12–20 px lines; slice F1@0.5 from 0.536 (rotated) to 0.859
  (multi-line). This is Phase 9's starting point.

**Descoped / moved (documented)**
- CLEval-style detection metrics: descoped from Phase 8, with the reason, in the IMPLEMENTATION_PLAN
  §8 deviations table and EXPERIMENT_PROTOCOL §11 (2026-10-25). Signed off by the owner at the
  Phase 8 gate (2026-10-27).
- HierText's native PQ-style metric: computed from Phase 14 with the official code, for the detector
  as for every system (EXPERIMENT_PROTOCOL §11, 2026-10-22 and 2026-10-25).

**Tests:** 569 pass locally with `main` merged in, the GPU-marked tests included (2026-10-27, GPU
idle). `scripts/compute_ledger.py` now counts the Phase 8 jobs and has a test for the detector logs.

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| Overfit-one-batch passes | pass: Dice 0.973, threshold L1 0.018, F1@0.5 0.944 against a batch ceiling of 0.990 | `reports/training/sanity/GW-DET-001/sanity.md` |
| Validation F1 reported with CI | pass: full validation with page-bootstrap CIs for all three runs | `reports/training/GW-DET-00N/report.md` |
| Overlays saved for each slice (small text, rotated, dense, multi-line, noisy background) | pass: one sheet per slice plus the worst pages, for each run | `reports/training/GW-DET-00N/overlays/` |
| Diagnostics report generated | pass: each run report has its diagnostics section, slice and line-level recall tables and post-processing counters | `reports/training/GW-DET-00N/report.md` |
| CI green (post-processing unit tests on synthetic masks) | pass | runs 40553993755 (branch), 40554005155 (pull request), 40556174553 (`main`) |

**Compute:** 16.2 GPU-hours in Phase 8, inside the plan's 20–40 (`reports/compute/ledger.md`): 13.9 h
of training, the rest the sanity sequence, the resume check and full-validation scoring. The 0.8 h
counted as jobs that did not finish are the three sanity runs that reported FAIL because a
pre-declared rule fired; they finished, and the rules were applied as planned.

**Known issues**
- The ImageNet weight terms are an owner decision before v0.1 ships a detector (RISK_REGISTER R-05).
  GW-DET-002 measured the fallback: training from scratch costs about 0.02 F1@0.5.
- The recognizer trainer is not on the shared loop yet. Phase 7 has merged, but the move is left out
  of this gate to keep it to the detector. The owner approved it (2026-10-27) for its own commit at
  the start of Phase 10, accepted only if the recognizer tests pass unchanged and an existing GW-REC
  `last.pt` still resumes.
- CI first hung in `tests/test_det_training.py`: on Linux the DataLoader forked its workers while
  the trainer's post-processing threads ran. Detector loaders now always spawn their workers
  (`c73451c`), and pytest dumps every thread's stack after two minutes (`faulthandler_timeout`).
- `LABEL_VERSION` 2, the fallback for a slow data loader, was not needed: the Large runs waited for
  data 2–3% of each epoch. The Small run waited 21–31%, which cost wall-clock time, not steps.
- Detector CPU latency is not measured yet. ADR-0011's page target is checked when the pipeline
  exists (Phase 10) and again after export (Phase 17).
- One seed per arm; the paired CIs cover page sampling, not training randomness.

**Git:** branch `phase/08-detector` (`main` with Phases 5 to 7 merged in first), merged through pull
request #8. Merge commit `60659f3`, tag `phase-08` (merge re-created with the noreply identity; same
tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 40556174553).

---

## Phase 09 — Detector error analysis

**Status:** COMPLETE (2026-10-28). Opened 2026-10-27 on `phase/09-det-analysis`. Validation data
only; the owner approved working through the night (2026-10-27).

**Objective:** understand GW-DET-001's failures before the pipeline is built, and fix what can be
fixed without retraining: post-processing, input resolution and box shape.

**Plan** (registered 2026-10-27, before any Phase 9 number was computed)

1. **Error analysis of GW-DET-001** (`best.pt`, operating point of record `inverse10`, all 1,715
   validation pages; descriptive, nothing is selected on it). Written to
   `reports/error_cases/detector/` by `scripts/det_analysis.py` from the run's saved boxes and
   candidates:
   - *Slices:* recall by line height and text angle (as in the run report), and page-level
     precision, recall and F1 by quartile of care-line density, blur (variance of the Laplacian)
     and contrast, from the Phase 3 audit, with page-bootstrap CIs.
   - *Merges and splits* at IoU 0.5, counting care lines only. A **merge** is a prediction that
     covers at least two care lines, each with intersection / line area ≥ 0.5. A **split** line is
     a care line with at least two predictions, each with intersection / prediction area ≥ 0.5.
   - *Missed lines (fn)*, each in the first category it fits: merged; split; poorly localised (best
     IoU with any prediction in [0.3, 0.5)); partly found (some prediction covers ≥ 0.1 of the
     line); missed. Missed lines are further split by whether a post-processing candidate existed
     (a component whose area-rectangle centre lies inside the line, dropped by `box_thresh` or the
     size filter: recoverable without retraining) or not (the network did not fire).
   - *False alarms (fp)*, likewise: merge; fragment (a prediction mostly inside one care line,
     intersection / prediction ≥ 0.5); poorly localised (best IoU with a care line in [0.3, 0.5));
     background (intersection / prediction < 0.1 with every line, don't care included); other.
   - *Galleries:* up to 24 crops per fn and fp category, drawn with `default_rng([20261027, k])`
     over the sorted members, independent of results, plus the 12 worst pages.
   - *Kernel excess* (RISK_REGISTER R-21): which unclip regime the trained map is in.
2. **Two halves of validation for every choice.** The 1,715 pages are split once, seeded
   (`det_matching.calibration_halves`, seed 20261027): half A selects, half B checks. A setting is
   adopted only if, on half B, the paired page-bootstrap 95% CI of its F1@0.5 minus the current
   setting's lies above 0. The B number is the one reported.
3. **GW-DET-004: post-processing calibration** of GW-DET-001, at long side 1280, from cached 8-bit
   probability maps on the CPU. Grid: `bin_thresh` 0.20–0.40 (step 0.05), `box_thresh` 0.4, 0.5,
   0.6, 0.7, `min_component_area` 1 or 3, and unclip `inverse` 0.8–1.3 (step 0.1) or `dbnet`
   1.3–2.1 (step 0.2): 440 settings. The arg-max of F1@0.5 on half A (ties: first in grid order)
   against the operating point of record, by rule 2, then re-checked once on float maps.
4. **GW-EFF-001: detector CPU latency** at long side 1024, 1280, 1536, 1792 and 2048:
   `scripts/det_profile.py`, PyTorch fp32 with default threads, batch 1, on the 110 Phase 5
   validation pages the baselines' page profiles used (10 warm-up, 100 timed), timing a decoded
   page to boxes (resize, normalise, forward, post-processing).
5. **GW-DET-005: input resolution.** The same five long sides with GW-DET-004's adopted
   post-processing (or the record's). A long side is **eligible** if the estimated page time on
   the profiled pages, its detector time plus 24.4 ms (GW-REC-024's mean line time) per predicted
   box, meets ADR-0011's page target (mean ≤ 1.5 s, p95 ≤ 2.7 s). Among eligible long sides the
   arg-max of F1@0.5 on half A is adopted against 1280 by rule 2. If no long side of 1280 or more
   is eligible, 1280 stays and the gap goes to Phases 16 and 17 as a risk: a lower resolution is
   not adopted on PyTorch timings, since ADR-0011 is judged on the ONNX Runtime build.
6. **GW-DET-006: box shape** (ADR-0012 point 3, RISK_REGISTER R-22). `quad` against `poly` at the
   adopted settings, on all validation pages and on the curved lines: care lines with at least six
   polygon vertices whose polygon area is below 0.8 of their minimum-area rectangle's. Phase 14 uses
   `poly` if, on the curved lines, the paired CI of recall@0.5 (poly minus quad) lies above 0 and,
   on all pages, the CI of F1@0.5 does not lie below 0; otherwise `quad`. Recall is used on the
   curved subset because precision cannot be restricted to a subset of lines. Phase 10 rectifies
   the quad in either case.
7. **Fixes that need retraining** (for example training labels for lines under 6 px, which
   `min_text_size` turns into don't care) are not run in Phase 9. Each is either registered with
   its cost for the owner to approve, or deferred to Phase 11 or 13 with the reason.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 9): `reports/error_cases/detector/` and written findings
in `research/findings.md`; fixes either applied under a new experiment ID or deferred with a reason.

**Results (2026-10-28)**

| Step | What was found | Evidence |
|---|---|---|
| Error analysis, GW-DET-001 | 30,184 missed care lines: 46% had a candidate the 0.6 threshold removed, 37% no candidate (mostly under 8 px), 11% merged; 3,631 false alarms, mostly fragments and merges. Recall 0.20 under 8 px, 0.18 beyond 45°. Kernels as wide as the labels (median ratio 1.00): the `inverse` regime | `reports/error_cases/detector/GW-DET-001/` |
| GW-DET-004, post-processing | half A arg-max `bin_thresh` 0.3, `min_component_area` 1, `inverse09`, `box_thresh` 0.5; half B +0.032 [+0.025, +0.039] F1@0.5 (float maps: +0.032 [+0.025, +0.039]): **adopted** | `reports/error_cases/detector/GW-DET-004/` |
| GW-EFF-001, detector CPU time | 132 to 615 ms per page from long side 1024 to 2048; with 24.4 ms per line for recognition, long sides up to 1536 fit ADR-0011's page target | `reports/efficiency/det_latency.md` |
| GW-DET-005, resolution | half A: F1 rises with resolution up to 2048; 1536 is the best eligible; half B +0.018 [+0.013, +0.022] over 1280: **adopted** | `reports/error_cases/detector/GW-DET-005/` |
| Both changes, all validation pages | F1@0.5 0.778 [0.762, 0.792], precision 0.852, recall 0.716, F1@0.7 0.671 (GW-DET-001 at its record point: 0.724, 0.924, 0.595, 0.631) | `reports/error_cases/detector/phase09_adopted/` |
| GW-DET-006, box shape | no curved care lines in HierText validation (all four-point), so the check is empty; `quad` kept | `reports/error_cases/detector/GW-DET-006/` |

**Fixes that need retraining** (plan item 7), deferred with their reasons:
- *Small text.* Lines under 8 px at the inference scale are still found only 34% of the time.
  Labels below `min_text_size` (6 px at the training scale) are don't care, so the network was never
  asked to find them. Lowering it, or training at a larger scale, is a new GW-DET run of about five
  GPU-hours; deferred to Phase 11 (robustness training), which retrains anyway, so one run tests it.
- *Rotated and vertical lines.* Recall 0.48 at 15–45° and 0.38 beyond 45°. Training rotation is at
  most 10°; a wider rotation range belongs to the same Phase 11 run.
- *Merges and fragments* grew with the looser threshold (1,982 merge boxes, 3,396 fragments); the
  shrink ratio and threshold-map weight that would change them are Phase 13 ablations.

**Compute:** about one GPU-hour of forward passes and three CPU-hours of sweeps and timing (chain log
`runs/phase09/chain.log`, local; the compute ledger counts training queues only).

**Tests:** the CPU suite passes locally (`scripts/det_sweep.py` is checked against `score_page` at
all 440 settings; the error categories, the curved-line rule and the pairing have tests).

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| `reports/error_cases/detector/` | pass: analysis, galleries per category, worst pages, before and after the Phase 9 settings | `reports/error_cases/detector/` |
| Written findings in `research/findings.md` | pass: F7, F8, F9 | `research/findings.md` |
| Fixes applied under a new experiment ID or deferred with a reason | pass: GW-DET-004 and 005 applied; retraining fixes deferred above; GW-DET-006 could not decide | `research/experiments.md` |
| CI green | pass | runs 40570430877 (branch), 40619874396 (pull request), 40620241350 (`main`) |

**Known issues**
- RISK_REGISTER R-22 stays open: no validation set has curved line annotations, so whether `poly`
  boxes would score better on GDC's curved lines is unknown before Phase 14. Phase 14 uses `quad`,
  as fixed here. The owner may decide before Phase 14 to also report `poly` there, labelled
  exploratory.
- The looser post-processing finds more of everything: false alarms rose 2.5 times (3,631 to 9,269),
  a quarter of them on background. Phase 10 measures what that does to page text.

**Git:** branch `phase/09-det-analysis`, merged through pull request #9. Merge commit `def522b`, tag
`phase-09` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 40620241350, on the merge as GitHub made it; the re-created merge has the same tree).

---

## Phase 10 — Full OCR pipeline

**Status:** write-up done (2026-10-28). Gate open: push, pull request, CI and merge, after Phase 9.
Opened 2026-10-28 on `phase/10-pipeline`, branched from Phase 9 while its gate was open (the owner
approved working through the night). Validation data only.

**Objective:** `GlypharisOCR.predict(image)` end to end, measured on validation.

**Implemented so far**
- `glypharis.pipeline`: `TextRecognizer` (width-sorted batches with a fixed padding margin, so a
  line reads the same alone or in any batch; tested on GW-REC-024 too), `GlypharisOCR` (detect, cut
  each line exactly as the training crops were cut, read, order), `reading_order` (deskew, then
  recursive XY-cut; tests for single column, two columns, mixed blocks, ragged lines, rotated pages
  and blocks), `calibration` (ECE with 15 equal-mass bins, temperature fit).
- `scripts/e2e_eval.py` (pipeline or oracle on validation pages, scored by the code that scored the
  baselines, with paired comparisons) and `scripts/rec_calibrate.py`.
- An inference smoke test on tiny untrained models built from a fixed seed (no weight files in the
  repository), in CI with the rest of the tests.

**Plan** (registered 2026-10-28, before any Phase 10 number was computed)

1. **Pipeline settings.** Detector GW-DET-001 at Phase 9's adopted settings (GW-DET-004's
   post-processing and GW-DET-005's input resolution); recognizer GW-REC-024 at temperature 1 for
   text (temperature changes confidence only). Line crops: `cut_line`, 64 px high with a 15%
   margin, the training-crop settings.
2. **Line confidence** (`glypharis.pipeline.recognizer`): the geometric mean of the top-class
   probability over the frames whose top class is not blank, at temperature T; for a line read as
   empty, the mean blank probability. A line is right when its text equals the label exactly on the
   strict track.
3. **GW-E2E-003, calibration (H8, RQ8).** Every line of the recognizer's validation pack, read by
   the pipeline's recognizer. T is fitted on the lines of Phase 9's validation half A (binary NLL,
   grid 0.50 to 4.00 in steps of 0.05, ties to the T nearest 1). ECE at T = 1 and at the fitted T on
   half B, with a 95% line-bootstrap CI of the difference; the fitted T is adopted if that CI lies
   below 0, otherwise confidence is documented as raw. H8's Phase 10 reading: raw ECE above 0.05,
   and the fitted T at least halving it. The fitted T is also applied to the pipeline's own lines
   (GW-E2E-001), as information.
4. **GW-E2E-001, the pipeline on validation (H1, RQ1).** All 1,715 HierText validation pages, and
   the 400 Phase 5 pages with the baselines' validation predictions scored by the same code and
   paired page by page (bag-of-lines page CER and E2E line F1, strict track primary, normalised
   secondary). HierText has no reading order, so order-aware metrics are n/a. H1 is judged on GDC's
   clean slice in Phase 14; here its ratio (GLYPHARIS page CER over PP-OCRv5 mobile's, on the 400
   pages) is reported as information.
5. **GW-E2E-002, oracle analysis (H4, RQ5).** The ground-truth care lines of all 1,715 pages, cut
   and read like the pipeline's. Recognition's share of page error = oracle bag-of-lines CER over
   pipeline bag-of-lines CER; detection's share is the rest. Reported on all pages and per blur
   quartile (Phase 9's quartiles: Q4 sharpest, Q1 blurriest). H4's Phase 10 reading, exploratory:
   recognition's share above one half on Q4 and below one half on Q1.
6. **GW-EFF-002, page time on the CPU.** The whole pipeline, decoded page to ordered text, on the
   110 Phase 5 profile pages (10 warm-up, 100 timed), PyTorch fp32 with default threads, batch 1
   page, on an otherwise idle machine: mean and p95 page time and peak RAM against ADR-0011's page
   target (mean at most 1.5 s, p95 at most 2.7 s, RAM below 1,836 MiB). This measurement replaces
   Phase 9's estimate.
7. **Reading order** is checked by its unit tests only: no validation set here has a reading order;
   GDC measures it in Phase 14.
8. **Moving the recognizer trainer onto the shared loop** (owner approved 2026-10-27) is deferred
   to the start of Phase 11, the next phase that trains the recognizer: nothing in Phases 10 to 20
   trains it, and `train.py` and five recognizer scripts depend on the current trainer, so the move
   is best made and checked (recognizer tests unchanged, an existing GW-REC `last.pt` resumes) when
   it is next used.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 10): pipeline unit- and integration-tested on
fixtures; validation E2E metrics reported; calibration either validated (ECE reduced, with numbers)
or confidence documented as raw; CI green with an inference smoke test on a tiny bundled model.

**Results (2026-10-28)** (strict track; percent with 95% page-bootstrap CIs)

| Experiment | Result | Evidence |
|---|---|---|
| GW-E2E-003, calibration | T = 2.35 fitted on half A; ECE on half B 0.217 → 0.089, −0.128 [−0.133, −0.124]: **adopted**; H8 supported on validation | `reports/calibration/GW-E2E-003/` |
| GW-E2E-001, all 1,715 pages | bag-of-lines page CER 27.56 [25.14, 30.36], E2E line F1 57.96, detection F1 77.79 | `reports/e2e/GW-E2E-001/` |
| GW-E2E-001, the 400 Phase 5 pages | page CER 24.33 against 42.44 (PP-OCRv6 small), 49.65 (PP-OCRv6 tiny), 50.84 (PP-OCRv5 mobile), 94.50 (Tesseract 5); paired against PP-OCRv6 small −18.11 [−21.43, −15.08] CER, +14.50 [+11.93, +17.05] E2E F1 | `reports/e2e/GW-E2E-001/` |
| GW-E2E-002, oracle | page CER 6.67 with ground-truth boxes; recognition's share 0.24 overall, 0.20 to 0.29 per blur quartile: detection dominates; H4 not supported on HierText | `reports/e2e/GW-E2E-002/` |
| GW-EFF-002, CPU page time (PyTorch) | mean 1,998 ms, p95 3,955 ms (recognition 1,721 ms for 40.9 lines), peak RAM 1,624 MiB: misses ADR-0011 on time, meets it on memory | `reports/efficiency/GW-EFF-002.json` |

**Compute:** about 0.6 GPU-hours (calibration and the end-to-end and oracle runs) and 0.2 CPU-hours
(chain log `runs/phase_progress.log`, local).

**Tests:** the CPU suite passes locally with Phase 9 merged in; the pipeline, reading order,
calibration and the tiny-model smoke test are covered (`tests/test_pipeline.py`,
`tests/test_reading_order.py`).

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| Pipeline unit- and integration-tested on fixtures | pass: batching, confidence, calibration, end to end on known lines, a smoke run on tiny models; reading order on six layouts | `tests/test_pipeline.py`, `tests/test_reading_order.py` |
| Validation E2E metrics reported | pass: GW-E2E-001 and GW-E2E-002 | `reports/e2e/` |
| Calibration validated or confidence documented as raw | pass: validated (ECE 0.217 → 0.089 on the held-out half) | `reports/calibration/GW-E2E-003/` |
| CI green with an inference smoke test on a tiny bundled model | pending (after the push) | `tests/test_pipeline.py::test_smoke_tiny_bundled_models_run_end_to_end` |

**Known issues**
- The PyTorch pipeline misses ADR-0011's page time (2.0 s mean against 1.5 s; recognition is 86%
  of it). ADR-0011 is judged on the ONNX Runtime build: Phase 17 measures it (GW-EFF-004). Speed
  work, such as a smaller padding margin, fewer padded columns per batch or INT8, belongs to
  Phase 16 under registered IDs.
- Reading order is tested on synthetic layouts only; GDC measures it in Phase 14.
- Detection is the bottleneck on HierText (three quarters of page error). The retraining fixes
  Phase 9 deferred to Phase 11 are where it can improve.
- The recognizer trainer's move onto the shared loop is deferred to Phase 11 (plan item 8).

**Git:** branch `phase/10-pipeline`, with Phase 9 merged in; push and merge pending (after Phase 9).

---

## Phase 17 — Model export

**Status:** write-up done (2026-10-28). Gate open: push, pull request, CI and merge, after Phase 10.
Opened 2026-10-28 on `phase/17-export`, branched from Phase 10; Phases 11 to 16 come after v0.1
(ADR-0008). The owner approved installing `onnx` 1.23.2 and `onnxruntime`
1.31.1 from PyPI (2026-10-28); both are pinned in `requirements/lock.txt`, so CI runs the ONNX
checks too.

**Objective:** a release of the v0.1 models that runs without the training code, and an ONNX build
proven equivalent to PyTorch.

**Implemented**
- Release files: the detector's (`export_detector`, now with the spec chosen later, here Phase 9's)
  and the recognizer's (`TextRecognizer.release` / `from_release`), both readable with
  `torch.load(..., weights_only=True)`.
- `glypharis.export.onnx`: both models exported with dynamic axes (TorchScript exporter, opset 17),
  and ONNX Runtime backends that slot into `TextLineDetector` and `TextRecognizer`.
- `glypharis.export.bundle`: one directory with both release files, both ONNX files and a manifest
  of their SHA-256s and settings; loading checks every hash and picks the backend.
- `scripts/export_check.py` (equivalence) and `scripts/e2e_profile.py --bundle` (page time).
- CI's equivalence on tiny models: ONNX against PyTorch for both models and the whole bundle.

**Plan** (registered 2026-10-28, before the export of the trained models was checked)
1. **Bundle v0.1.0:** GW-DET-001 with Phase 9's settings (GW-DET-004's post-processing, GW-DET-005's
   resolution) and GW-REC-024 at GW-E2E-003's temperature if adopted, else 1.
2. **GW-EFF-003, equivalence** (`scripts/export_check.py`), both backends loaded from the bundle:
   detector probability maps on the 110 Phase 5 profile pages, max abs difference at most 1e-4;
   recognizer logits on 5,000 seeded validation lines, max abs difference at most 1e-3; greedy text
   identical on every line except near ties (a PyTorch top-2 gap below 1e-3 at the first frame that
   differs), at most 0.1% of lines; bag-of-lines page CER (strict) of the whole pipeline on the 400
   Phase 5 pages within 0.05 points between the backends. The export counts as successful only if
   all four pass.
3. **GW-EFF-004, page time on ONNX Runtime:** `scripts/e2e_profile.py --bundle ... --backend onnx`,
   the protocol of GW-EFF-002, against ADR-0011's page target. This is the measurement ADR-0011 is
   judged on.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 17): export declared successful only after the
equivalence tests pass and are recorded; CI runs ONNX equivalence on the tiny model.

**Results (2026-10-28)**

| Experiment | Result | Evidence |
|---|---|---|
| GW-EFF-003, equivalence | maps 7.3e-6, logits 3.7e-4, text identical on 5,000 lines, page CER 24.33% on both backends: **export successful** | `reports/export/GW-EFF-003/` |
| GW-EFF-004, ONNX Runtime page time | mean 5,515 ms, p95 12,691 ms, peak RAM 11,221 MiB: misses ADR-0011 on time and memory | `reports/efficiency/GW-EFF-004.json` |

The first bundle was built with the detector's old post-processing (a slip in the local script that
ran the checks); it passed GW-EFF-003 but failed the Phase 18 check against Phase 10, and was rebuilt
and rechecked (`research/failures.md`; its outputs are kept in
`reports/export/GW-EFF-003/attempt1_record_post/`).

**Decision taken here** (reversible, for the owner to confirm): PyTorch is the default backend of
v0.1 (`load_bundle`, `from_pretrained`, the CLI), and ONNX Runtime runs without its memory arena,
which kept a buffer per input shape. ONNX stays available (`backend="onnx"`).

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| Export declared successful only after the equivalence tests pass and are recorded | pass | `reports/export/GW-EFF-003/equivalence.md` |
| CI runs ONNX equivalence on the tiny model | pass locally; CI pending the push | `tests/test_export.py` |

**Known issues**
- **Owner decision needed: ADR-0011.** Its revisit condition is met (the exported v0.1 misses the page
  target by a wide margin). Options are listed in ADR-0011.
- The exported recognizer is slow because the SVTR's global attention is exported as plain matrix
  products; a fused attention export (or a rewritten attention) is the first Phase 16 task. The
  exported detector is already twice PyTorch's speed and could be used on its own then.
- The newer `torch.export`-based exporter needs `onnxscript`, which is not installed; the
  TorchScript exporter works but is deprecated upstream.
- Model bundles are not distributed: `runs/release/glypharis-0.1.0` exists on the laptop only. How
  BookWorm gets the bundle (a release asset, a shared drive) is an owner decision.

**Git:** branch `phase/17-export`; push and merge pending (after Phase 10).

