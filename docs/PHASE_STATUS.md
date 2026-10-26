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
| 7 | Recognizer improvement | **IN PROGRESS** (gate: push, CI, merge) | — |
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

**Status:** write-up done (2026-10-26). Gate open: push, pull request, CI and merge. Opened in
parallel with Phases 5 and 6 at the owner's request (2026-10-23), on `phase/07-rec-improvement`.

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
| CI green | pending (after the push) | — |

**Compute:** 55.3 GPU-hours in Phase 7, inside the plan's 30–60 (`reports/compute/ledger.md`; the
hours of jobs that did not finish include the parts of crashed runs that later resumed).

**Known issues**
- The chosen recognizer's CPU latency (24.4 ms mean per line in PyTorch) is close to ADR-0011's 25 ms
  target; the Phase 17 ONNX export has to keep it inside.
- One seed at the full budget (deviation recorded in GW-REC-024's entry).
- On this laptop, training crashed with CUDA host-allocation and "unknown" errors under memory
  pressure; two data-loading workers and the resume path kept every run going.
- The optional TrOCR-Small fine-tune was not run (budget).

**Git:** branch `phase/07-rec-improvement`; push and merge pending.
