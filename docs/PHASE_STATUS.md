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
| 10 | Full OCR pipeline | **COMPLETE** | `phase-10` |
| 11 | Robustness training | **COMPLETE** | `phase-11` |
| 12 | Synthetic data study | **COMPLETE** | `phase-12` |
| 13 | Ablation study | **COMPLETE** | `phase-13` |
| 14 | External evaluation | **COMPLETE** | `phase-14` |
| 15 | Error analysis | **COMPLETE** | `phase-15` |
| 16 | Efficiency optimisation | **COMPLETE** | `phase-16` |
| 17 (v1) | Model export, v1 re-run | **COMPLETE** | `phase-17-v1` |
| 18 (v1) | Inference package, v1 re-run | **COMPLETE** | `phase-18-v1` |
| 20 (v1) | Production engineering, v1 re-run | **COMPLETE** | `phase-20-v1` |
| 17 | Model export | **COMPLETE** | `phase-17` |
| 18 | Inference package | **COMPLETE** | `phase-18` |
| 19 | Optional HTTP API | **COMPLETE** | `phase-19` |
| 20 | Production engineering | **COMPLETE** | `phase-20`, `v0.1` |
| 21 | Research package | **COMPLETE** | `phase-21` |
| 22 | Final validation | in progress (plan registered) | — |
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
(Correction, 2026-11-18: 3.72 GiB is the total. The archives are 3.64 GiB and the ground truth 79 MiB,
as `DATASETS.md` now states.)

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
| Real-scan slice | owner + maintainer | moved to a separate set: **done, GDC-scan v1 frozen 2026-11-14** (`data/manifests/gdc_scan/gdc_scan_v1.json`; 28 pages, 779 lines; ground truth from the born-digital source's text layer instead of hand transcription) |

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
**Note (2026-11-18, Phase 21):** these two seeds (GW-REC-032, 033) were not run, because Phase 13 chose
option B (GW-REC-037). The promise is closed for v1, which has three seeds of its own (GW-REC-034 to
036); GW-REC-024 stays at one seed.

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
  (Decided 2026-11-18: ADR-0014 keeps GW-DET-001 as a disclosed exception to ADR-0009's rule.)
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

**Status:** COMPLETE (2026-10-28).
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
| CI green with an inference smoke test on a tiny bundled model | pass | runs 40620770935 (branch), 40620789850 (pull request), 40621254147 (`main`) |

**Known issues**
- The PyTorch pipeline misses ADR-0011's page time (2.0 s mean against 1.5 s; recognition is 86%
  of it). ADR-0011 is judged on the ONNX Runtime build: Phase 17 measures it (GW-EFF-004). Speed
  work, such as a smaller padding margin, fewer padded columns per batch or INT8, belongs to
  Phase 16 under registered IDs.
- Reading order is tested on synthetic layouts only; GDC measures it in Phase 14.
- Detection is the bottleneck on HierText (three quarters of page error). The retraining fixes
  Phase 9 deferred to Phase 11 are where it can improve.
- The recognizer trainer's move onto the shared loop is deferred to Phase 11 (plan item 8).

**Git:** branch `phase/10-pipeline` (Phase 9 merged in first), merged through pull request #11
(#10 had been merged into the Phase 9 branch instead of `main`). Merge commit `f060380`, tag
`phase-10` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 40621254147).

---

## Phase 17 — Model export

**Status:** COMPLETE (2026-10-28). Opened 2026-10-28 on `phase/17-export`, branched from Phase 10; Phases 11 to 16 come after v0.1
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
| CI runs ONNX equivalence on the tiny model | pass: runs 40621722608 (branch), 40621740954 (pull request), 40622745727 (`main`) | `tests/test_export.py` |

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

**Git:** branch `phase/17-export` (`main` with Phase 10 merged in first), merged through pull request
#12. Merge commit `99805d8`, tag `phase-17` (merge re-created with the noreply identity; same tree
and parents).

**CI status:** green on the branch, the pull request and `main` (run 40622745727, on the merge as
GitHub made it; the re-created merge has the same tree).

---

## Phase 17 — Model export (v1 re-run)

**Status:** COMPLETE (2026-11-18). Opened 2026-11-18 on `phase/17-v1-rerun`, branched from `main` after Phase 16 (merge `0af7fc4`). Validation data only; `research/test_access_log.md` gets no row. The
v0.1 record above stays as it is (ADR-0008 item 3: Phases 17, 18 and 20 are re-run on the final
model with the already-tested tooling and a new completion record).

**Objective:** declare the export of v1 successful for one named bundle, the GW-EFF-006 bundle
(`runs/release/glypharis-v1`, manifest SHA-256 `875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`),
and check the path v1 ships on, that bundle read with the PyTorch backend, against the same PyTorch
v1 references.

**Owner decisions (2026-11-18, before this plan)**
- v1 keeps PyTorch as its default backend; ADR-0011's laptop targets are recorded as missed for v1.
- BookWorm's target hardware is lightweight devices such as smartphones, which triggers ADR-0011's
  revisit; the phone candidate is the all-ONNX build with fused attention, a new version evaluated
  in Phase 22 (ADR-0011, owner decision).
- The re-runs of 17, 18 and 20 go on three branches in that order, each with its own gate and
  record. Phase 18's package check runs the PyTorch backend only; its clean-venv install test may
  download CPU torch 2.14.1, torchvision 0.29.1 and `requirements/lock.txt` into a throwaway venv.
- The v1 code freeze (`assert_v1_code` against `phase14-eval`) ends with one commit after the
  Phase 20 re-run, tagged and made the new reference for the check; until then `glypharis/pipeline`,
  `models`, `export`, `data` and `configs` stay byte for byte as at the tag.

**Plan** (registered 2026-11-18, before any Phase 17 re-run measurement)

1. **Preflight.** `assert_v1_code`; the bundle's manifest SHA-256 is the one above and
   `read_manifest` re-hashes its four files; the Phase 16 reference outputs load against their
   committed manifest (`reports/efficiency/phase16/reference_manifest.json`). The bundle is never
   rebuilt or written to: a rebuild changes its hash, and the H1′ ONNX verdict (GW-EFF-010) is for
   this hash only.
2. **Release files and the ONNX export: cited.** The bundle's `detector.pt` and `recognizer.pt`
   with the manifest's provenance, and its two ONNX files with dynamic axes, from GW-EFF-006
   (`reports/efficiency/phase16/v1_bundle.json`). The export is the TorchScript exporter at opset
   17, as at v0.1.
3. **The four equivalence tolerances for the ONNX backend: cited** from GW-EFF-006
   (`reference_check.json`): detector map max abs difference ≤ 1e-4, logits ≤ 1e-3, decoded text
   identical except near ties (top-2 gap < 1e-3) on at most 0.1% of lines, page CER within 0.05
   points. The comparison is against the hashed PyTorch v1 references, as the 2026-11-17 change row
   in EXPERIMENT_PROTOCOL §11 records, not against the bundle's own PyTorch backend.
4. **GW-EFF-012: the bundle on the PyTorch backend against the same references.** A new script,
   `scripts/phase17_v1_check.py` (outside the frozen paths, with tests on tiny models), loads the
   bundle through `GlypharisOCR.from_pretrained(..., backend="torch")`, the shipped path, at 10
   threads, and runs Phase 16's reference check unchanged: the same four tolerances on the same 110
   profile pages, 5,000 seeded lines and 400 pages. It refuses a bundle with another manifest
   SHA-256, bundle problems, references written from another freeze record, and an existing
   output. Pass rule: all four tolerances hold. Expected about 40 minutes on the CPU.
5. **ADR-0011: cited.** GW-EFF-007, 009 and 010 measured this bundle with the same tooling on
   2026-11-17 and 18; they are this re-run's measurement against ADR-0011, not repeated.
6. **Default backend: PyTorch**, as at v0.1 and by the owner's decision; the code already defaults
   to it (`load_bundle`, `from_pretrained`, the CLI and the server).
7. **CI.** The tiny-model ONNX equivalence tests (`tests/test_export.py`, and the tiny SVTR tests in
   `tests/test_phase16_reference.py`) run in CI; the full suite passes in a clone without tags.
8. **Records.** GW-EFF-012's entry; GW-EFF-006 to 010's "next experiment" lines; the export is
   declared successful only once GW-EFF-012 passes and is recorded.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 17): the export declared successful only after the
equivalence tests pass and are recorded; CI runs ONNX equivalence on the tiny model.

**Known limitations, stated in advance**
- One laptop, HierText validation only; the phone candidate is not measured here.
- The ONNX evidence is Phase 16's (GW-EFF-006); only the PyTorch path is run again.

**Results (2026-11-18)** (GW-EFF-012; `reports/export/GW-EFF-012/torch_bundle_check.json`, written
by `scripts/phase17_v1_check.py` at `8c70747`, clean tree; validation only, no test data read)

- *Preflight.* `assert_v1_code` passed (the v1 paths equal `phase14-eval`); the bundle's manifest
  SHA-256 was `875357a6…` and `bundle_problems` found nothing; the Phase 16 references matched
  their hashes and the committed manifest. The bundle was not rebuilt or written to.
- *GW-EFF-012* (2026-11-18T07:29:27Z to 07:50:27Z, 10 threads, `model_version` `glypharis_v1`):
  all four tolerances pass with no difference at all.

  | Check | Limit | Measured |
  |---|---|---|
  | Detector map max abs difference (110 pages) | 1e-4 | 0.0 |
  | Logits max abs difference (5,000 lines) | 1e-3 | 0.0 |
  | Text (5,000 lines) | near ties ≤ 0.1%, no other difference | 5,000 identical, no near ties |
  | Page CER (400 pages, strict) | 0.05 points | 24.32% on both, difference 0.0; 400 pages identical |

  The shipped path, the bundle read with the PyTorch backend through `from_pretrained`, gives
  exactly the outputs of the PyTorch v1 build that Phase 14 evaluated.
- *ONNX* (cited, item 3): GW-EFF-006 passed the same four tolerances for the bundle's ONNX files
  (map 7.33e-06, logits 3.09e-04, no text differences, page CER within 1.49e-05;
  `reports/efficiency/phase16/reference_check.json`).
- *ADR-0011* (cited, item 5): GW-EFF-007, 009 and 010; every v1 CPU build misses the latency
  targets (F38), and by the owner's decision v1 keeps PyTorch as its default.

**The v1 export is declared successful** for bundle `875357a6…`: the ONNX equivalence tests
(GW-EFF-006) and the PyTorch path (GW-EFF-012) both pass and are recorded.

**Exit criteria check** (IMPLEMENTATION_PLAN Phase 17)

| Criterion | Status | Evidence |
|---|---|---|
| Export declared successful only after the equivalence tests pass and are recorded | pass | GW-EFF-006, GW-EFF-012 (above) |
| CI runs ONNX equivalence on the tiny model | pass | `tests/test_export.py`; the tiny SVTR tests in `tests/test_phase16_reference.py` |
| CI green | pass | runs 44263370636 (branch), 44263400419 (pull request), 44267047137 (`main`) |

**Tests:** the CPU suite passes in a clone without tags (1,218 passed, 4 skipped, at `8c70747`);
the new tests cover the check on tiny models: a pass with no difference on the same model, and
refusal of another bundle hash, a bundle unlike the record, changed references and an existing
output.

**Compute:** GW-EFF-012 ran by hand for 21 minutes on the CPU, outside the queue logs, so the
compute ledger does not count it.

**Known issues**
- The `glypharis/export/onnx.py` docstring still gives F13's old figures; it is corrected in the
  commit that ends the v1 code freeze, after the Phase 20 re-run (owner decision, 2026-11-18).
- `from_pretrained`'s default name is still `glypharis-0.1.0`, which does not resolve to the v1
  bundle; v1 is loaded by path. Changed in the same end-of-freeze commit.

**Git:** branch `phase/17-v1-rerun`, merged through pull request #22. Merge commit `e2cd8f6`, tag
`phase-17-v1` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 44267047137).

## Phase 18 — Inference package

**Status:** COMPLETE (2026-10-28). Opened 2026-10-28 on `phase/18-package`, branched from Phase 17.

**Objective:** GLYPHARIS installable and usable without the training code.

**Implemented**
- `GlypharisOCR.from_pretrained(name_or_path, backend="onnx")`: a model bundle by path, or by name
  from `$GLYPHARIS_MODELS`, `~/.glypharis/models` or `./models`; every file's hash is checked;
  nothing is downloaded.
- `predict` and `predict_batch` on file paths, PIL images and arrays (grey or RGB); `OCRResult`
  with `to_dict` / `to_json` (version, model version, size, text in reading order, lines with text,
  confidence, detection score, polygon and quad, and the time per step).
- `infer.py --image ... | --input-dir ... [--recursive]` and the `glypharis-ocr` command: one JSON
  and one text file per image; unreadable images are reported and skipped (exit code 1); a missing
  model is exit code 2. Package version `0.1.0.dev0`.
- Tests on a tiny bundle: loading by path and name, every input type, the JSON fields, the CLI's
  outputs and exit codes, and `infer.py` run as a program. CI installs the package into a fresh
  environment on every run and now also runs the installed command: that is the clean-install test.

**Plan** (registered 2026-10-28, before the check ran)
1. **GW-E2E-004, the package against Phase 10:** `scripts/package_check.py` runs
   `GlypharisOCR.from_pretrained` on the v0.1 bundle (ONNX backend, as shipped) over the 400 Phase 5
   validation pages and compares it with GW-E2E-001's predictions on those pages. It passes when the
   bag-of-lines page CER (strict) differs by at most 0.05 points; the share of lines with identical
   text and box (corners within 1 px) is reported.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 18): API tests on fixtures; clean-venv install test;
outputs match the Phase 10 pipeline on the validation sample.

**Results (2026-10-28):** GW-E2E-004 passes: the package reads the 400 Phase 5 pages with 24.33% page
CER, as Phase 10's pipeline did (+0.001 points), with 16,827 of 16,835 lines identical
(`reports/package/GW-E2E-004.json`). It ran on the ONNX backend as registered; since Phase 17 the
default backend is PyTorch, whose reading GW-EFF-003 showed to be identical.

**Exit criteria**

| Criterion | Result | Evidence |
|---|---|---|
| API tests on fixtures | pass | `tests/test_api.py` |
| Clean-venv install test | pass in CI: every run installs the package into a fresh environment and runs the installed `glypharis-ocr`; locally the editable install and the command were checked | `.github/workflows/ci.yml` |
| Outputs match the Phase 10 pipeline on the validation sample | pass (GW-E2E-004) | `reports/package/GW-E2E-004.json` |

**Known issues**
- The default backend is PyTorch (Phase 17's decision, for the owner to confirm with ADR-0011).
- Model bundles are not distributed; `from_pretrained` finds one only where it is placed.

**Git:** branch `phase/18-package` (`main` with Phase 17 merged in first), merged through pull request
#13. Merge commit `feda915`, tag `phase-18` (merge re-created with the noreply identity; same tree
and parents).

**CI status:** green on the branch (run 40623224823) and the pull request (run 40623231007). On
`main`, the run for this merge (40624284890) was cancelled when Phase 20 merged a minute later; the
run for that merge, which contains Phase 18, passed (40624323452).

---

## Phase 18 — Inference package (v1 re-run)

**Status:** COMPLETE (2026-11-18). Opened 2026-11-18 on `phase/18-v1-rerun`, branched from `main` after the Phase 17 re-run (merge
`e2cd8f6`). Validation data only; `research/test_access_log.md` gets no row.
The v0.1 record above stays as it is (ADR-0008 item 3).

**Objective:** the installed package, loading the v1 bundle the way it ships (PyTorch backend),
reads the validation pages like the frozen v1 pipeline Phase 14 evaluated, and installs and runs
in a clean environment.

**Plan** (registered 2026-11-18, before any Phase 18 re-run measurement; the owner's decisions of
2026-11-18 are in the Phase 17 re-run record: PyTorch only, and the clean-venv download approved)

1. **Preflight.** `assert_v1_code`; the bundle's manifest SHA-256 is `875357a6…` (GW-EFF-006,
   declared successful in the Phase 17 re-run); the reference predictions,
   `data/processed/phase14/GW-E2E-006/hiertext_val.jsonl`, have SHA-256 `c8520576…`, the value in
   `data/manifests/processed/phase14_outputs.json`.
2. **Tooling.** `scripts/package_check.py` gains `--experiment`, `--v1` (the v1 code check and
   the bundle against the freeze record), `--expect-manifest-sha256`, `--reference-sha256`,
   `--threads`, a refusal to overwrite its output, and provenance in the report (git state, the
   bundle and reference hashes, versions, times); tests on a tiny bundle. The comparison itself is
   unchanged: the 400 Phase 5 pages, strict bag-of-lines page CER, the 0.05-point tolerance and
   the 1 px box agreement.
3. **GW-E2E-011, the package against the frozen v1 pipeline:** `GlypharisOCR.from_pretrained` on
   the v1 bundle with the PyTorch backend, on the CPU, over the 400 Phase 5 validation pages,
   compared with GW-E2E-006's predictions on those pages (the frozen pipeline on the GPU in strict
   fp32, Phase 14's validation reference). Passes when the page CER differs by at most 0.05
   points; the share of lines with identical text and box (corners within 1 px) is reported. As at
   v0.1 (GW-E2E-004 against GW-E2E-001), the reference ran on the GPU and the package runs on the
   CPU. About 30 minutes.
4. **API tests:** `tests/test_api.py`, `tests/test_export.py`, `tests/test_server.py` and the full
   suite, in a clone without tags.
5. **Smoke on the real bundle** (a check, not a result; no numbers in the docs): `infer.py
   --image` and `--input-dir` on two validation pages write their JSON and text files;
   `model_version` is `glypharis_v1`; a path, a PIL image, an array and `predict_batch` give the
   same lines.
6. **Clean-venv install:** a new virtual environment outside the repository; CPU `torch` 2.14.1 and
   `torchvision` 0.29.1 from the PyTorch CPU index, then `requirements/lock.txt`, then the package
   from this branch (`pip install <worktree> --no-deps`, not editable). From a folder outside the
   repository: `glypharis-ocr --version`, and one validation page read with the v1 bundle given by
   its path. The environment is deleted afterwards. CI's install into a fresh environment on every
   run is cited beside it.
7. **Packaging:** `pyproject.toml` stays at `0.1.0.dev0`; the version for v1.0 is Phase 23's.
8. **Records:** GW-E2E-011's entry; the completion record beside the v0.1 one.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 18): API tests on fixtures; clean-venv install test;
outputs match the pipeline on the validation sample (here the frozen v1 pipeline, GW-E2E-006).

**Known limitations, stated in advance**
- `from_pretrained`'s default name `glypharis-0.1.0` does not resolve to the v1 bundle; v1 is
  loaded by path until the end-of-freeze commit after the Phase 20 re-run.
- The PyTorch backend only (owner decision); the ONNX files' equivalence is GW-EFF-006's.
- Model bundles are still not distributed; `from_pretrained` finds one only where it is placed.

**Results (2026-11-18)** (GW-E2E-011; validation only, no test data read)

- *Preflight* (item 1): `assert_v1_code` passed; the bundle's manifest SHA-256 was `875357a6…` and
  the reference predictions' `c8520576…`, both checked by the script before anything loaded.
- *Tooling* (item 2): `scripts/package_check.py` at `c5d7011`, with its first tests
  (`tests/test_package_check.py`); the comparison is unchanged.
- *GW-E2E-011* (item 3; `reports/package/GW-E2E-011.json`, 2026-11-18T08:33:54Z to 08:52:03Z,
  PyTorch backend, 10 torch threads, clean tree): **pass**. The package reads the 400 Phase 5
  validation pages with 24.32% page CER, as the frozen pipeline did in Phase 14 (GW-E2E-006),
  difference 0.0 points (limit 0.05), and 16,835 of 16,835 lines have identical text and box
  (corners within 1 px), though the reference ran on the GPU in strict fp32 and the package on the
  CPU. The report names the reference by its file name and SHA-256 only.
- *Smoke on the real bundle* (item 5; a check, no numbers): `infer.py --input-dir` over two
  validation pages wrote a JSON and a text file for each; `model_version` was `glypharis_v1`; a
  path, a PIL image, an array, the image's bytes and `predict_batch` gave identical lines.
- *Clean-venv install* (item 6, 2026-11-18): a new environment outside the repository with CPU
  `torch` 2.14.1, `torchvision` 0.29.1 and `requirements/lock.txt`, then the package from
  `c5d7011` (not editable). From a folder outside the repository the installed
  `glypharis-ocr --version` printed its versions, and `glypharis-ocr --image` read a validation
  page with the v1 bundle given by its path and wrote its JSON and text; the package was imported
  from the new environment. The environment was then deleted. CI installs the package into a fresh
  environment on every run as well.

**Exit criteria check** (IMPLEMENTATION_PLAN Phase 18)

| Criterion | Status | Evidence |
|---|---|---|
| API tests on fixtures | pass | `tests/test_api.py`, `tests/test_export.py`, `tests/test_server.py`, `tests/test_package_check.py` |
| Clean-venv install test | pass | item 6 above; CI's fresh-environment install |
| Outputs match the pipeline on the validation sample | pass (GW-E2E-011) | `reports/package/GW-E2E-011.json` |
| CI green | pass | runs 44273719841 (branch), 44273747616 (pull request), 44285968356 (`main`) |

**Tests:** the CPU suite passes in a clone without tags (1,226 passed, 4 skipped, at `c5d7011`).
A first run at the same commit had one failure that did not repeat; its name was not captured, and
the tests of this phase's code passed on their own in between, so which test it was is not known.
The new tests cover the package check on a tiny bundle: a pass with every provenance field, a
bundle by name, the refusals (existing output, wrong hashes, `--v1` with other code or a bundle
unlike the record) and references that fail.

**Compute:** GW-E2E-011 ran by hand for 18 minutes on the CPU, outside the queue logs, so the
compute ledger does not count it.

**Known issues**
- `from_pretrained`'s default name `glypharis-0.1.0` still does not resolve to v1 (frozen code;
  changed in the end-of-freeze commit after the Phase 20 re-run).
- The package version stays `0.1.0.dev0` until Phase 23.

**Git:** branch `phase/18-v1-rerun`, merged through pull request #23. Merge commit `0dd8f19`, tag
`phase-18-v1` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 44285968356).

---

## Phase 19 — Optional HTTP API

**Status:** COMPLETE (2026-10-29). Opened
2026-10-29 on `phase/19-http-api`, branched from `main` after Phase 11, at the owner's request
(2026-10-29) to do it while Phase 12 waits; it uses no model results and no evaluation data.

**Objective:** a minimal HTTP endpoint around the package, documented as an integration boundary.

**Implemented**
- `glypharis.server` and the `glypharis-serve` command: `POST /ocr` (the image file as the request
  body; the answer is `OCRResult.to_dict`, which carries `model_version` and `glypharis_version`)
  and `GET /health`. One process, one request at a time, Python's standard-library HTTP server, so
  no new dependency; it listens on 127.0.0.1 by default.
- Limits: request body at most 25 MiB by default (`--max-bytes`, 413 `request_too_large`), and the
  library's pixel limit (`--max-pixels`, 413 `image_too_large`). Unreadable bodies are 400
  `invalid_image`, a missing Content-Length 411, unknown paths and methods 404 and 405, anything
  unexpected 500 `internal_error`, logged with its traceback but never sent to the client.
- Each connection carries one request and closes after the answer, and a client silent for 30
  seconds is dropped. A first smoke run found that, with keep-alive, one idle client blocked the
  single-request server; this is the fix, and a test covers it.
- `GlypharisOCR.predict` (and `load_rgb`) now also accept the bytes of an image file, checked like a
  file on disk (pixel limit from the header before decoding).

**Plan check:** the IMPLEMENTATION_PLAN asks for "FastAPI or similar, single process". FastAPI is not
installed and would add several packages for one endpoint, so the standard library serves it; the
schema and limits are the same either way.

**Smoke check (validation data, not a result):** with the v0.1 bundle on the CPU, one HierText
validation page through `POST /ocr` returned the same 47 lines as `predict` on the same bytes.

**Tests:** `tests/test_server.py` (a real socket on a free port, a tiny bundle from untrained models):
health, the documented JSON matching `predict`, unreadable and empty bodies, both size limits, the
missing length, unknown paths and methods, a 500 without internals, an idle client not blocking the
next, and a missing model at start-up. CI also runs `glypharis-serve --help` in its fresh
environment.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 19)

| Criterion | Result | Evidence |
|---|---|---|
| API tests with a test client | pass | `tests/test_server.py` |
| Documented as an integration boundary only | pass | `glypharis/server.py` docstring, README |
| CI green | pass | runs 40822321310 (branch), 40822344940 (pull request), 40824345618 (`main`) |

**Known issues**
- Not for exposure to a network as is: no authentication, no TLS, one request at a time. A
  multi-worker deployment needs one model per worker process.
- The 25 MiB body limit is a round number above a 600 dpi A4 PNG, not a measurement.

**Git:** branch `phase/19-http-api`, merged through pull request #16. Merge commit `8dfb634`, tag
`phase-19` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 40824345618; the run on the
merge as GitHub made it was cancelled by the re-created merge's run, which has the same tree).

---

## Phase 20 — Production engineering

**Status:** COMPLETE (2026-10-28). Opened 2026-10-28 on `phase/20-hardening`, branched from Phase 18.
With it the v0.1 path (ADR-0008: Phases 10, 17, 18, 20) is complete: tag `v0.1`.

**Objective:** a library BookWorm can call without surprises: every failure it can cause or meet is
a clear, typed error.

**Implemented**
- `glypharis.errors`: `GlypharisError` (stable `code`, `to_dict`), `InvalidImageError`,
  `ImageTooLargeError`, `ModelLoadError`.
- Input checks (`load_rgb`): paths that do not exist or are not files; files that are not images,
  empty, truncated or corrupt; arrays of the wrong shape, NaN or infinite values, negative or
  out-of-range values, unsupported types; empty images. Pixel limit `max_pixels` (default 64 million,
  about 8000 by 8000), checked from the file header before decoding, and PIL's decompression-bomb
  guard mapped to `ImageTooLargeError`. Accepted: grey, RGB, RGBA, palette and CMYK images, 16-bit
  and float images, booleans; a blank page gives no lines.
- Model loading: a missing bundle, a missing or changed file, an unreadable or foreign manifest, or
  a file that fails to load all raise `ModelLoadError`; loaded bundles are cached per process
  (`from_pretrained(..., cache=True)`).
- Logging through the `glypharis` logger (model load at INFO, per page at DEBUG); `stats()` on a
  `GlypharisOCR` (pages, lines, refused inputs, mean seconds per page); `glypharis.version_info()`
  and `glypharis-ocr --version`; the CLI reports each refused image with its error code.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 20)

| Criterion | Result | Evidence |
|---|---|---|
| Failure-path tests: corrupt image, huge image, empty page, non-image | pass | `tests/test_hardening.py` |
| No unhandled exceptions in the fuzz-lite test (120 seeded cases: random arrays, random bytes, damaged PNGs) | pass | `tests/test_hardening.py::test_fuzz_lite_never_raises_anything_but_glypharis_errors` |
| CI green | pass | runs 40623490172 (branch), 40623496316 (pull request), 40624323452 (`main`) |

**Known issues**
- `max_pixels` (64 million) is a guess at a safe upper bound for this laptop, not a measurement;
  BookWorm may want its own.
- The fuzz test is seeded and small (120 cases); it guards the input edge, not the models.
- Owner decisions carried into v0.1: ADR-0011's page time (Phase 17) and how bundles are
  distributed.

**Git:** branch `phase/20-hardening` (Phase 18 merged in first), merged through pull request #14.
Merge commit `1f910c9`, tags `phase-20` and `v0.1` (merge re-created with the noreply identity;
same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 40624323452, on the merge as
GitHub made it; the re-created merge has the same tree).

---

## Phase 20 — Production engineering (v1 re-run)

**Status:** COMPLETE (2026-11-18). Opened 2026-11-18 on `phase/20-v1-rerun` after the Phase 18 re-run.
Validation and made-up inputs only; no test data is read and `research/test_access_log.md` gets no
row. The v0.1 record above stays as it is (ADR-0008 item 3).

**Objective:** the hardening holds on the v1 bundle itself, not only on the tiny models the tests
use: every input BookWorm can pass and every damaged bundle ends in a result or a typed
`GlypharisError`, never another exception.

**Plan** (registered 2026-11-18, before any Phase 20 re-run measurement)

1. **Failure-path tests: cited and re-run.** `tests/test_hardening.py` (corrupt, truncated and
   non-image files; huge images refused before decoding; empty and unusual pages; wrong types and
   values; the 120 seeded fuzz-lite cases; damaged bundles; caching, stats, logging and versions)
   in the full suite and in CI. The hardening code is unchanged since `phase14-eval` apart from
   Phase 19's bytes input, which is part of the frozen v1 code.
2. **GW-E2E-012, the same cases on the v1 bundle.** A new script, `scripts/hardening_check.py`
   (outside the frozen paths, with a test on a tiny bundle so CI covers it), loads the v1 bundle
   (manifest SHA-256 `875357a6…`, refused otherwise; never written to) with
   `GlypharisOCR.from_pretrained` on each backend, PyTorch and ONNX, at 10 threads, and runs:
   - the 120 seeded fuzz-lite cases of `tests/test_hardening.py` (the generator shared, not
     copied), the unusual-page and wrong-type lists, and the same inputs as bytes;
   - corrupt, truncated, empty and non-image files, a missing path, a directory, and an image over
     the pixel limit;
   - damaged bundles, made only in temporary copies: each file removed, one byte changed in an ONNX
     and a `.pt` file, an unreadable manifest, a foreign format, and a file that fails to load
     although its hash matches (the copy's manifest re-hashed). Each must raise `ModelLoadError`;
   - caching (the same object with `cache`, a new one without), the load log line naming
     `glypharis_v1`, `version_info()` and `glypharis-ocr --version`;
   - the CLI on a folder holding a good page and a bad file (exit code 1, the bad file reported
     with its error code) and the HTTP API's `POST /ocr` (Phase 19) through its test client.

   It records, per backend and case kind, how many inputs gave a result and how many a
   `GlypharisError` with which code, and every other exception with its type; `assert_v1_code`,
   the bundle hash, the git state and the times. **Pass rule:** no exception other than a
   `GlypharisError` anywhere, every damaged bundle a `ModelLoadError`, and the caching, version,
   CLI and API checks as stated. Times are not measured (the latency numbers are Phase 16's).
   **Amendment (2026-11-18, after the harness was written and before it ran on the v1 bundle).**
   (a) The PyTorch backend never opens the two `.onnx` files (the ONNX backend reads all four: the
   `.pt` files give it the settings and the character set), so junk in an `.onnx` file, with its
   hash re-written in the copy's manifest, loads on PyTorch; for those cases "loaded" is the
   expected outcome, and every other damaged-bundle case must raise `ModelLoadError`. Each of the
   four files gets one changed byte, and a removed manifest and a missing bundle are added
   (16 cases per backend). (b) Two checks join the pass rule: the good page gives a result on every
   backend, with the same text from its path and its bytes; and the bundle's files still match its
   manifest afterwards. The other outcomes per input kind (bad files refused as `invalid_image`,
   oversized images as `image_too_large`, and so on) are recorded but not part of the rule, since a
   few wrong arrays become readable images once encoded as bytes.
3. **Records.** GW-E2E-012's entry; the completion record beside the v0.1 one; then the commit that
   ends the v1 code freeze (owner decision of 2026-11-18), as its own step after this phase.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 20): failure-path tests (corrupt image, huge image,
empty page, non-image); no unhandled exceptions in the fuzz-lite test.

**Known limitations, stated in advance**
- The fuzz cases are seeded and few; most are refused before they reach the models.
- `max_pixels` (64 million) is still not measured; this phase does not change it.
- `glypharis_version` stays `0.1.0.dev0` beside `model_version` `glypharis_v1` until Phase 23.

**Results (2026-11-18)** (GW-E2E-012; `reports/hardening/GW-E2E-012.json`, written by
`scripts/hardening_check.py` at `958f24c`, clean tree, 2026-11-18T10:36:16Z to 10:38:01Z, 10 threads;
no test data read). **Pass** on every check, for bundle `875357a6…` on both backends.

| Inputs, per backend (the same on PyTorch and ONNX) | Inputs | Results | `GlypharisError` |
|---|---|---|---|
| Fuzz-lite: random arrays | 40 | 6 | 34 `invalid_image` |
| Fuzz-lite: random bytes behind image suffixes | 40 | 0 | 40 `invalid_image` |
| Fuzz-lite: damaged PNGs | 40 | 0 | 40 `invalid_image` |
| The fuzz-lite inputs as bytes (arrays with a bytes form) | 93 | 13 | 80 `invalid_image` |
| Unusual pages, and as bytes | 22 | 21 | 1 `invalid_image` (the empty page) |
| Wrong types and values, and as bytes | 14 | 3 | 11 `invalid_image` |
| Bad files, and as bytes | 10 | 0 | 10 `invalid_image` |
| Over the pixel limit | 6 | 0 | 6 `image_too_large` |
| The good page (path, bytes, PIL image, array) | 4 | 4 | 0 |
| **All** | **269** | **47** | **222** |

- *No other exception* on either backend. The three wrong inputs that give a result as bytes are
  arrays that become ordinary images once encoded, as the amended plan says.
- *Damaged bundles* (16 cases per backend, in temporary copies): each file or the manifest removed,
  the bundle missing, one byte changed in each file, an unreadable manifest, a foreign format, and
  each file replaced by junk with its hash re-written. All 32 ended as registered: `ModelLoadError`,
  except junk in the two `.onnx` files on PyTorch, which never opens them, and which loads.
- *The good page* gave 47 lines from its path, its bytes, a PIL image and an array, with the same
  text from the path and the bytes.
- *Cache, log, versions:* the same object with `cache`, a new one without; the load log line names
  `glypharis_v1` and the backend; `version_info()` and `--version` agree (`glypharis` `0.1.0.dev0`).
- *CLI* (from a folder outside the repository, PyTorch): a folder with the good page and a bad file
  gives exit code 1, the bad file reported with `[invalid_image]`, and the good page's JSON written.
- *HTTP API* (`POST /ocr`, each backend): 200 with 47 lines for the good page, 400 `invalid_image`
  for the bad file.
- The bundle's files matched its manifest afterwards.

**Exit criteria check** (IMPLEMENTATION_PLAN Phase 20)

| Criterion | Status | Evidence |
|---|---|---|
| Failure-path tests: corrupt image, huge image, empty page, non-image | pass | `tests/test_hardening.py`; on the v1 bundle GW-E2E-012 |
| No unhandled exceptions in the fuzz-lite test | pass | `tests/test_hardening.py`; the same 120 cases on the v1 bundle, both backends, GW-E2E-012 |
| CI green | pass | runs 44288817497 (branch), 44288826149 (pull request), 44291099138 (`main`) |

**Tests:** the CPU suite passes in a clone without tags (1,230 passed, 4 skipped, at `958f24c`). The
new tests run the check on a tiny bundle on both backends (a full pass, the report's structure, no
local path in it) and cover its refusals (an existing output, another manifest hash, other v1
code). `tests/test_hardening.py` now takes its cases from `scripts/hardening_cases.py`, unchanged.

**Compute:** GW-E2E-012 ran by hand for under 2 minutes on the CPU, outside the queue logs, so the
compute ledger does not count it.

**Known issues**
- An array with NaN values is refused, but the same values in a float image file are not: the file
  is read as a blank page, with NumPy's "All-NaN slice" `RuntimeWarning` from `_pil_rgb` when every
  value is NaN. No exception escapes; the code is frozen, so this stays as it is for v1.
- `max_pixels` is still not measured, and the default model name still does not resolve to v1; the
  name changes in the end-of-freeze commit.

**Git:** branch `phase/20-v1-rerun`, merged through pull request #25. Merge commit `fdd6b5d`, tag
`phase-20-v1` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 44291099138, on `4574576`, which
also holds pull request #24, the server's 413 fix).

### The end of the v1 code freeze (2026-11-18)

The owner decided on 2026-11-18 that the freeze of the v1 code ends with one tagged commit after
the Phase 20 re-run (Phase 17 re-run, owner decisions). That commit is `efb197a`, tag `v1-code`.

- **What changed in the v1 paths** (`glypharis/pipeline`, `models`, `export`, `data`, `configs`):
  three files. `glypharis/export/onnx.py` and `bundle.py` changed in their docstrings only, which
  closes the deviation Phase 16 recorded: the export docstring's old figures are replaced by F34's
  and F35's. `glypharis/pipeline/ocr.py` changed in its docstrings and in one default, the bundle
  name of `from_pretrained`, from `glypharis-0.1.0` to `glypharis-v1`.
- **The proof:** `scripts/v1_code_diff.py` compares `phase14-eval` with `v1-code` from git's raw
  blobs, as syntax trees with the docstrings removed, and passes
  (`reports/export/v1-code/code_diff.json`). Weights, settings and every code path that produces a
  reading are unchanged, so this is not a new version of v1, and Phase 14's results and GW-EFF-006,
  GW-EFF-012, GW-E2E-011 and GW-E2E-012 still apply. The bundle was not rebuilt (manifest SHA-256
  `875357a6…`).
- **The check:** `assert_v1_code` now compares the v1 paths with `v1-code` (`V1_TAG`). `PIN_TAG`
  stays `phase14-eval`, the code Phase 14 evaluated. Reports written before this commit record
  `phase14-eval` in their `v1_code`, as they should. The change is in EXPERIMENT_PROTOCOL §11.
- **The default name:** `glypharis-v1` is the v1 bundle's directory name. It resolves once that
  directory is in a model directory (`$GLYPHARIS_MODELS`, `~/.glypharis/models` or `./models`); on
  the laptop, set `GLYPHARIS_MODELS=runs/release`, which is not searched by default. No bundle ships
  with the repository yet.
- **Tests:** the CPU suite passes in a clone without tags (1,252 passed, 4 skipped, at `efb197a`).
- **Git:** branch `phase/20-v1-end-of-freeze`, merged through pull request #26. Merge commit
  `6caf327` (re-created with the noreply identity; same tree and parents); `v1-code` is in `main`'s
  history. CI green on the branch (run 44295660367), the pull request (44295705382) and `main`
  (44298874690).

---

## Phase 11 — Robustness training

**Status:** COMPLETE (2026-10-29). Opened 2026-10-28 on `phase/11-robustness`, branched from Phase
20; plan registered 2026-10-28 (below) and run on the laptop GPU while the college GPU slot is
queued. Validation data only.

**Objective:** find out which degradation augmentations help (RQ3, H3).

**Done**
- The recognizer trainer now runs on the shared training loop (owner approved 2026-10-27; deferred
  here from Phase 10). Both acceptance conditions hold: the recognizer tests pass unchanged (48,
  GPU tests included), and a GW-REC-024 checkpoint written by the old trainer resumes on the new one
  with identical learning rates and losses within CUDA's run-to-run noise; the A/B/C resume check
  passes too (`reports/training/phase11/trainer_move/`).

**Plan** (registered 2026-10-28, before any Phase 11 number was computed; the owner chose the
CRNN recipe and the reuse of Phase 7's runs on 2026-10-28)

1. **Arms and runs.** The CRNN short budget of Phase 7 (GW-REC-003's configuration: 8 epochs,
   height 32, batch 64, HierText training lines), changing only `data.augmentation`. Three seeds
   (1337, 2024, 7) per arm, 15 runs:

   | Arm | Augmentation | Seed 1337 | Seed 2024 | Seed 7 |
   |---|---|---|---|---|
   | A | none | GW-REC-015 | GW-DATA-001 | GW-DATA-002 |
   | B | geometric | GW-REC-016 | GW-DATA-003 | GW-DATA-004 |
   | C | photometric | GW-REC-017 | GW-DATA-005 | GW-DATA-006 |
   | D | document degradation | GW-REC-018 | GW-DATA-007 | GW-DATA-008 |
   | E | full (all three) | GW-REC-003 | GW-REC-004 | GW-REC-005 |

   The seed-1337 runs of A to D and all of E already exist (Phase 7); the 8 GW-DATA runs are new
   (`configs/rec/gw-data-001.yaml` to `008`, each Phase 7's arm config with the seed changed).
   Every arm sees the same lines for the same number of optimizer steps. About 1.3 GPU-hours per
   run on the RTX 4050, so about 11 GPU-hours.
2. **Robustness metric.** Each run's `best.pt` (chosen on clean validation CER, as in Phase 7) reads
   all 216,000 crops of the frozen GLYPH-C v1 pack (`scripts/glyph_c_eval.py score`, GW-DATA-009):
   CER on the strict track per corruption and level, and the **mean corruption error (mCE)**, the
   unweighted mean over the 35 corruption-level cells (7 corruptions, 5 levels) of each cell's
   micro CER. The clean copies are scored too and reported, not used in the rule.
3. **Decision rule** (Phase 7's rule, moved to mCE). For each of B to E against A: the paired 95%
   CI of mCE(arm) − mCE(A), pooling each arm's three seeds and resampling the 6,000 GLYPH-C source
   lines (1,000 resamples), and the seed ranges. **Helps**: CI entirely below 0 and all three of
   the arm's seeds below all three of A's. **Hurts**: the mirror. Otherwise **no clear effect**.
4. **H3.** Supported if arm D helps under rule 3 and its mean clean validation CER (HierText
   validation pack, the three runs' best checkpoints) is at most 5% (relative) above arm A's.
   Otherwise not supported, with the reason (no clear effect, or the clean guard).
5. **Reported but not decided on:** CER per corruption and level for every arm (where each family
   helps), clean validation CER per arm, and the clean GLYPH-C CER.
6. **Data:** training and validation splits and GLYPH-C (built from validation lines) only. No test
   set is read in this phase.

**Known limitations, stated in advance**
- The seed-1337 runs and arm E were trained by Phase 7's recognizer trainer; the new runs by the
  shared training loop it moved onto. The move was checked (same learning rates, losses within
  run-to-run noise), but for arms A to D one seed in three comes from the older code.
- GLYPH-C's HierText half comes from the validation lines that picked each run's best epoch. Every
  arm is selected the same way, so the comparison is fair, but the absolute CERs are slightly
  optimistic.
- The CRNN short budget is Phase 7's ablation recipe, not the v0.1 recognizer (SVTR, height 48, 30
  epochs). The verdicts are about the augmentation families on this recipe; whether they carry
  over to the shipped model is a later check on the college GPU.

**Results (2026-10-29)** (GW-DATA-009, `reports/robustness/phase11_comparison.md`; strict track)

| Arm | Runs (seeds 1337, 2024, 7) | mCE % (pooled) | mCE % per seed | Δ mCE vs A [95% CI] | Clean val CER % (mean) | Verdict |
|---|---|---|---|---|---|---|
| A (none) | GW-REC-015, GW-DATA-001, GW-DATA-002 | 10.28 | 10.43, 10.14, 10.28 | — | 8.66 | control |
| B (geometric) | GW-REC-016, GW-DATA-003, GW-DATA-004 | 8.61 | 8.56, 8.51, 8.76 | −1.67 [−1.73, −1.62] | 8.34 | helps |
| C (photometric) | GW-REC-017, GW-DATA-005, GW-DATA-006 | 10.23 | 10.17, 10.19, 10.32 | −0.06 [−0.08, −0.02] | 8.57 | no clear effect |
| D (degradation) | GW-REC-018, GW-DATA-007, GW-DATA-008 | 10.04 | 10.04, 10.03, 10.04 | −0.25 [−0.30, −0.19] | 8.68 | helps |
| E (full) | GW-REC-003, GW-REC-004, GW-REC-005 | 8.46 | 8.39, 8.45, 8.53 | −1.83 [−1.90, −1.75] | 8.31 | helps |

- **H3: supported** (on validation, this recipe). Arm D helps under the registered rule: CI below
  zero and all three D seeds (10.04, 10.03, 10.04) below all three A seeds
  (10.43, 10.14, 10.28); clean validation CER 8.68%
  against 8.66%, +0.2% relative, inside the 5% guard.
- **Verdicts:** geometric helps, photometric has no clear effect (CI just below zero, but its seeds
  overlap A's), document degradation helps, full helps.

**Per corruption** (reported, not decided on; mean over the five levels, pooled over seeds):
- Rotation dominates: 38.89% for A, 32.93% for B, 39.00%
  for D and 33.31% for E, against at most 7.90% for any other
  corruption. GLYPH-C rotates whole lines by 1.2° to 6°; training rotates by at most ±2° (geometric
  preset), and only 30% of lines. Rotation is more than half of every arm's mCE.
- Geometric augmentation's gain is mostly rotation and perspective (7.90 →
  5.55%); degradation's is blur (4.93 →
  4.53%), low resolution (6.14 →
  5.55%) and JPEG (4.68 → 4.48%), with nothing on
  rotation. Photometric augmentation includes blur, noise and JPEG, yet barely moves them.
- Clean GLYPH-C CER: 4.14% (A), 3.50% (B),
  4.13% (C), 3.95% (D),
  3.41% (E).

**Run notes**
- The 8 new runs trained with no diagnostic flags; each landed close to Phase 7's run of the same
  arm (per-seed mCE above), so the trainer change and the seeds do not stand out. The new runs took
  45,367 or 45,368 optimizer steps against 45,360 for the Phase 7 runs: one more batch per epoch
  (5,671 against 5,670), 0.02% more steps.
- GW-DATA-003 and GW-DATA-005 were each stopped once by a power cut and resumed from their last
  checkpoint (steps 40,000 and 11,342), losing under 1,000 steps each; the GLYPH-C scoring chain
  was also killed once when its terminal closed (GW-REC-018 was scored again from scratch).

**Deferred from Phase 9:** the detector retraining for small and rotated text was to share Phase 11's
retraining. Phase 11 retrained only the recognizer (the owner chose the CRNN study that fits the
laptop), so it is deferred again, to Phase 13 or the college GPU, where it is one GW-DET run of
about five GPU-hours.

**Compute:** 9.5 GPU-hours in the compute ledger (1.3 on jobs that did not finish:
the two power-cut stops and the killed scoring job), of which 1.26 for 15 GLYPH-C
scoring jobs; the first scoring run (GW-REC-015, about six minutes) was a timing check run by hand
and is not in the ledger (`reports/compute/ledger.md`).

**Tests:** the CPU suite passes locally; new tests cover the GLYPH-C scorer (mCE, per-line pooling,
verdicts, the H3 guard, the per-corruption table) and the ledger's Phase 11 lines.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 11)

| Criterion | Result | Evidence |
|---|---|---|
| Results table with CIs and a paired comparison against A | pass | `reports/robustness/phase11_comparison.md` |
| A clear verdict per family (helps / no effect / hurts) | pass: geometric helps, photometric no clear effect, degradation helps, full helps | same |
| Recorded in `research/ablations.md` | pass | `research/ablations.md`, Phase 11 |
| CI green | pass | runs 40817268680 (branch), 40817332338 (pull request), 40819192658 (`main`) |

**Known issues**
- The verdicts hold for the CRNN short budget. The v0.1 recognizer (SVTR, height 48, 30 epochs)
  was not retrained; checking that they carry over needs the college GPU.
- GLYPH-C's corruptions are close relatives of the training transforms, so it measures robustness
  to familiar corruptions at higher strength; Phase 14 (GDC phone photos) is the unseen test.
- Rotation within a line box is the recognizer's largest weakness (F15). A wider rotation range in
  the geometric preset is a Phase 13 candidate.

**Git:** branch `phase/11-robustness`, merged through pull request #15. Merge commit `456b095`, tag
`phase-11` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 40819192658).

---

## Phase 12 — Synthetic data study

**Status:** COMPLETE (2026-10-30). Opened
2026-10-29 on `phase/12-synthetic`, branched from `main` after Phase 19; the owner approved running it
on the laptop GPU in batches that can pause (2026-10-29). Validation data only.

**Objective:** measure how much document-style synthetic data (GlyphSynth) helps the recognizer,
how it scales, and which generator factors matter (RQ2, H2′).

**Done before registration**
- `data.epoch_lines`: each epoch draws that many lines afresh from the whole training pool (no line
  twice in an epoch), so a larger pool does not mean more steps. Earlier configs keep their hashes;
  at the pool size or above nothing changes.
- `scripts/build_glyphsynth_variants.py`: the extra GlyphSynth packs. Its first lines reproduce the
  v1 training pack byte for byte, so the extension continues v1 exactly.
- `scripts/synth_study.py`: GlyphSynth validation scoring and the comparison below, with tests.

**Plan** (registered 2026-10-29, before any Phase 12 run was trained or scored)

1. **Recipe and control.** The CRNN short budget of Phases 7 and 11 (GW-REC-003's configuration:
   8 epochs, height 32, full augmentation). The control is HierText only: GW-REC-003, 004 and 005
   (seeds 1337, 2024, 7), already trained. Every synthetic arm adds GlyphSynth v1 training lines
   to HierText and sets `data.epoch_lines` to 348,935, the HierText training set's size, so all
   arms see as many lines per epoch, and take as many steps, as the control. The cost of this
   equal budget: the more synthetic lines in the pool, the fewer HierText lines per epoch.
2. **Runs** (12 new, about 11 GPU-hours on the RTX 4050):

   | Arm | GlyphSynth lines added | Seeds | Runs |
   |---|---|---|---|
   | +25K | 25,000 (seeded subset of v1) | 1337, 2024, 7 | GW-DATA-010, 011, 012 |
   | +50K | 50,000 (seeded subset of v1) | 1337 | GW-DATA-013 |
   | +100K | 100,000 (all of v1) | 1337, 2024, 7 | GW-DATA-014, 015, 016 |
   | +250K | 250,000 (v1 and its extension) | 1337, 2024, 7 | GW-DATA-017, 018, 019 |
   | +100K, 6 fonts | 100,000 from 6 of the 85 training families | 1337 | GW-DATA-020 |
   | +100K, no capture | the v1 lines without capture effects | 1337 | GW-DATA-021 |

   The plan asked for three seeds at 25K and 250K; +100K gets three as well, because it is the
   control for both generator factors.
3. **Readouts** (GW-DATA-022, `scripts/synth_study.py`), each run's best checkpoint (selected on
   HierText validation CER, as before):
   - **Document slice** (primary): the HierText validation lines on pages with at least 100 lines
     in the validation pack, 104 pages and 16,782 lines (22.7%). HierText has no document label,
     and pages this dense are likely to be printed text; this was chosen from line counts only and
     not checked page by page, so it is a proxy.
   - HierText validation, all lines; GlyphSynth validation (10,000 lines in held-out font
     families, optimistic for synthetic arms: same generator); GLYPH-C mean corruption error for
     the capture factor (the +100K and no-capture runs).
4. **Decision rule** (Phase 7's, as in Phase 11). An arm's value pools its seeds. Differences are
   paired over pages (HierText readouts: lines on a page are not independent) or lines
   (GlyphSynth, GLYPH-C source lines), 95% bootstrap CI, 1,000 resamples. **Helps**: CI below 0 and
   every seed of the arm below every seed of its control; **hurts**: the mirror; otherwise **no
   clear effect**.
5. **H2′a** is supported if at least one size helps on the document slice against HierText only.
   Diminishing returns are reported, not decided: document-slice CER points gained per 1,000
   added lines from 25K to 100K against 100K to 250K.
6. **H2′b.** Fonts: many fonts help if +100K, 6 fonts is worse than +100K on the document slice
   under rule 4. Capture: capture effects help if the no-capture run's GLYPH-C mCE is higher than
   +100K's (CI above 0, its seed above all three +100K seeds), and the clean guard holds (+100K's
   mean HierText validation CER at most 5% relative above the no-capture run's). H2′b is supported
   if both factors help (and the guard holds).
7. **Data:** HierText training and validation, GlyphSynth training and validation, GLYPH-C. No
   test set is read.

**Known limitations, stated in advance**
- Equal steps, not equal epochs: the synthetic arms see fewer HierText lines per epoch, so a gain
  has to beat the loss of real data, and a loss may come from that displacement, not from the
  synthetic lines being harmful.
- The document slice is a density proxy, not a labelled domain; GDC (Phase 14) is the real
  document test. GlyphSynth validation flatters the synthetic arms.
- The factor arms have one seed each; their control (+100K) has three.
- The CRNN short budget, not the v0.1 recognizer; the control runs were trained by Phase 7's
  trainer, the new ones by the shared loop (as in Phase 11, where the two agreed).

**Results (2026-10-30)** (GW-DATA-022, `reports/synthetic/phase12_comparison.md`; strict track;
scaling plot `reports/synthetic/phase12_scaling.png`)

| Arm | Runs | Document slice CER % | Δ vs HierText only [95% CI] | Verdict | HierText val CER % | GlyphSynth val CER % | WER % (document slice) |
|---|---|---|---|---|---|---|---|
| HierText only | GW-REC-003, GW-REC-004, GW-REC-005 | 11.06 | — | control | 8.31 | 1.97 | 27.39 |
| +25K | GW-DATA-010, GW-DATA-011, GW-DATA-012 | 11.09 | +0.04 [−0.03, +0.11] | no clear effect | 8.29 | 0.94 | 27.34 |
| +50K | GW-DATA-013 | 11.18 | +0.12 [+0.03, +0.22] | hurts | 8.30 | 0.73 | 27.39 |
| +100K | GW-DATA-014, GW-DATA-015, GW-DATA-016 | 11.22 | +0.16 [+0.06, +0.26] | hurts | 8.38 | 0.57 | 27.49 |
| +250K | GW-DATA-017, GW-DATA-018, GW-DATA-019 | 11.44 | +0.39 [+0.25, +0.52] | hurts | 8.56 | 0.38 | 28.08 |
| +100K, 6 fonts | GW-DATA-020 | 11.13 | +0.07 [−0.04, +0.18] | no clear effect | 8.40 | 0.61 | 27.48 |
| +100K, no capture | GW-DATA-021 | 11.09 | +0.04 [−0.07, +0.15] | no clear effect | 8.38 | 0.65 | 27.43 |

- **H2′a: not supported.** No size helps on the document slice; +50K, +100K and +250K hurt, and the
  loss grows steadily with the size (CER rises by 0.0017 and 0.0015 points per 1,000 lines
  added, from 25K to 100K and from 100K to 250K). HierText validation as a whole tells the same
  story (+0.25 [+0.20, +0.30] at +250K).
- **H2′b: not supported.** Font variety: 6 families against 85 at 100K, −0.09 [−0.21, +0.03] on the document
  slice, no clear difference. Capture effects: without them GLYPH-C mCE is +0.31 [+0.27, +0.35] points higher
  (helps; the clean guard holds: 8.38% with, 8.38%
  without). One factor of two.
- The synthetic lines are learned: GlyphSynth validation CER falls from 1.97%
  to 0.38%. They do not carry over to photographed text, and each one
  displaces a HierText line from an epoch.

**Deviation from the plan: steps were not exactly equal.** Every arm saw 348,935 lines per epoch,
as registered, but batches are capped by padded width (`data.max_batch_columns`), and GlyphSynth
lines are wider (369 against 169 pixels on average at height 32, first 200,000 lines of each
pack), so the larger arms made more batches: 45,368 steps at 25K, 45,380 at 50K, 46,133 to 46,141 at 100K, 46,561 to 46,586 at
250K, against 45,360 to 45,368 for the control (up to 2.7% more). This favours the synthetic arms, so their
losses are if anything understated; the verdicts stand.

**Run notes**
- Every run trained with no diagnostic flags. Interruptions, all resumed or restarted cleanly:
  GW-DATA-011 (paused by the owner before its first checkpoint, restarted), GW-DATA-017 (a blocked
  checkpoint rename, `research/failures.md`; fixed with a retry), GW-DATA-020 (the chain was killed
  when the desktop app that launched it restarted; resumed from step 1,000). Several owner-requested
  pauses for power cuts fell between runs.

**Compute:** 17.1 GPU-hours in the compute ledger (0.3 on jobs that did not finish), plus
about 17 CPU-minutes for the GlyphSynth packs (`reports/compute/ledger.md`).

**Tests:** the CPU suite passes locally; new tests cover `data.epoch_lines` (no change at or above
the pool size, a fresh sample per epoch, old config hashes unchanged), the GlyphSynth variant
builder (byte-identical to v1), the study's readouts and rules, the checkpoint-rename retry and the
ledger's Phase 12 lines.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 12)

| Criterion | Result | Evidence |
|---|---|---|
| Scaling plots and table (CER, WER; latency unchanged by design) | pass | `reports/synthetic/phase12_comparison.md`, `phase12_scaling.png` |
| Verdict on "more synthetic data ⇒ better?" with CIs | pass: no; it hurts from 50K lines at this budget | same |
| Verdict on each generator factor with CIs | pass: fonts no clear effect; capture effects help robustness | same |
| CI green | pass | runs 41032780845 (branch), 41032809450 (pull request), 41034715295 (`main`) |

**Known issues**
- The document slice is a density proxy inside HierText; GDC (Phase 14) is the real test of
  whether document-style synthetic data helps on documents.
- The factor arms have one seed each.
- The verdicts are for the CRNN short budget at a fixed step budget. A larger model, or more steps,
  could use synthetic data differently; Phase 13 decides the final recipe, and on this evidence it
  should not add GlyphSynth beyond 25K lines.

**Git:** branch `phase/12-synthetic`, merged through pull request #17. Merge commit `c7c709d`, tag
`phase-12` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 41034715295).

---

## Phase 13 — Ablation study

**Status:** COMPLETE (2026-11-14). Opened
2026-10-30 on `phase/13-ablation`, branched from `main` after Phase 12. The owner chose to run all of it on the laptop GPU (2026-10-30), knowing the cost below (about
75 to 90 GPU-hours, two weeks with pauses), and a short-budget ladder with a full-budget final
model. Validation data only.

**Objective:** the main ablation table, and the final recognizer and detector configuration chosen
on validation and frozen (`configs/final/glypharis_v1.yaml`).

**Plan** (registered 2026-10-30. Rows L0, L1 and L2 and the seed-1337 runs of L3 and L4 were trained
in Phases 7, 11 and 12, so some of their numbers are already known; the rules below are applied to
them as written, and every other run is new.)

1. **The ladder** (short budget: 8 epochs, HierText training lines, 64 lines per step; three seeds,
   1337, 2024 and 7, per row). Each row changes its parent's configuration:

   | Row | Change | Parent | Runs (1337, 2024, 7) |
   |---|---|---|---|
   | L0 | baseline: CRNN, height 32, no augmentation | — | GW-REC-015, GW-DATA-001, 002 (Phase 11) |
   | L1 | + augmentation (full preset) | L0 | GW-REC-003, 004, 005 (Phase 7) |
   | L2 | + 25,000 GlyphSynth lines (the only size Phase 12 found harmless) | L1 | GW-DATA-010, 011, 012 (Phase 12) |
   | L3 | + input height 48 (input preprocessing) | L1 | GW-REC-006 (Phase 7), GW-REC-025, 026 (new) |
   | L4 | full model: + SVTR-style encoder, weight decay 1e-2 (and its learning rate 5e-4 and micro-batching) | L3 | GW-REC-022 (Phase 7), GW-REC-027, 028 (new) |
   | L5 | + wider rotation: up to ±4° on half the lines, instead of ±2° on 30% (Phase 11, F15) | L4 | GW-REC-029, 030, 031 (new) |

   L3 builds on L1, not L2: Phase 12 already found 25K GlyphSynth lines to make no clear
   difference on HierText validation (−0.02 [−0.05, +0.01], `reports/synthetic/`), which under
   rule 2 means L2 is not kept.
2. **Rules.** Each row against its parent, pooling each row's three seeds: HierText validation CER
   (the selection metric), paired over pages; Phase 7's verdict (CI on one side of zero and every
   seed beyond every parent seed). A row is **kept** when it helps. L5 is kept when it helps on
   GLYPH-C mean corruption error (paired over its 6,000 source lines) and does not hurt on HierText
   validation CER. The document slice (Phase 12) and GLYPH-C mCE are reported for every row.
3. **Final recognizer** (full budget: 30 epochs, three seeds): L4's recipe (option A: GW-REC-024,
   already trained, plus GW-REC-032 and 033), or L4 with L5's rotation if L5 is kept (option B:
   GW-REC-034, 035, 036). If L4 does not help over L3, nothing is trained until the owner decides.
   The final runs are reported against the CRNN baseline at the same budget (GW-REC-001, one seed),
   with GLYPH-C. The shipped checkpoint is the seed-1337 run of the chosen option, fixed now so that
   no seed is picked on its result.
4. **Detector** (GW-DET-007): the retraining Phase 9 deferred, GW-DET-001 with labels for smaller
   text (`min_text_size` 3 px instead of 6) and training rotation up to 30° instead of 10°, one
   seed. Both detectors are scored on all 1,715 validation pages at Phase 9's adopted
   post-processing and resolution; GW-DET-007 is **adopted** if its F1@0.5 is higher with a paired
   page-bootstrap CI above zero. Otherwise GW-DET-001 stays.
5. **Freezing.** `configs/final/glypharis_v1.yaml` records the chosen recognizer and detector
   configurations, their checkpoints and hashes, the post-processing and the calibration step,
   from generated reports only. The test sets stay closed until Phase 14.

**Cost on the laptop** (from the compute ledger's earlier runs): L3 2 × 2.4 h, L4 and L5 5 × 4.7 h,
the detector about 6 h, the final runs 2 or 3 × 19.3 h: about 75 to 90 GPU-hours.

**Known limitations, stated in advance**
- The ladder runs at the short budget; only the final configuration is trained at full length.
- L4 bundles the encoder with weight decay, learning rate and micro-batching (Phase 7's chosen
  recipe), so it is not a clean test of the encoder alone (H6 stays exploratory).
- The detector change is one seed against one seed, and the post-processing was tuned on
  GW-DET-001, which favours it slightly.
- Seeds 1337 of L3 and L4 and all of L0 to L2 were trained by the earlier recognizer trainer or
  in earlier phases (as in Phases 11 and 12).

**Results (2026-11-14)** (GW-REC-037, `reports/ablation/phase13_ladder.md` and `phase13_final.md`;
strict track; three seeds per row; Δ against each row's parent, paired, 95% CI)

| Row | Runs | Val CER % | Δ val | Document slice % | Δ doc | GLYPH-C mCE % | Δ mCE | Kept |
|---|---|---|---|---|---|---|---|---|
| L0 baseline | GW-REC-015, GW-DATA-001, GW-DATA-002 | 8.66 | — | 11.46 | — | 10.28 | — | start |
| L1 + augmentation | GW-REC-003, GW-REC-004, GW-REC-005 | 8.31 | −0.36 [−0.40, −0.31] | 11.06 | −0.40 [−0.54, −0.26] | 8.46 | −1.83 [−1.91, −1.75] | yes |
| L2 + 25K GlyphSynth | GW-DATA-010, GW-DATA-011, GW-DATA-012 | 8.29 | −0.02 [−0.05, +0.01] | 11.09 | +0.04 [−0.03, +0.11] | 7.19 | −1.27 [−1.32, −1.21] | no |
| L3 + height 48 | GW-REC-006, GW-REC-025, GW-REC-026 | 7.83 | −0.48 [−0.54, −0.42] | 10.66 | −0.39 [−0.54, −0.25] | 6.83 | −1.63 [−1.68, −1.58] | yes |
| L4 full model | GW-REC-022, GW-REC-027, GW-REC-028 | 7.57 | −0.26 [−0.32, −0.19] | 10.36 | −0.31 [−0.53, −0.11] | 6.29 | −0.54 [−0.59, −0.49] | yes |
| L5 + wider rotation | GW-REC-029, GW-REC-030, GW-REC-031 | 7.45 | −0.12 [−0.16, −0.08] | 10.27 | −0.09 [−0.17, −0.01] | 4.80 | −1.49 [−1.53, −1.45] | yes |

- Every "kept" row helps on all three measures. L2 makes no clear difference on validation or the
  document slice, so it is not kept (the rule), although it helps on GLYPH-C.
- L5 is kept under its own rule: it helps on GLYPH-C mCE and does not hurt validation (it helps
  there too). **Final option: B.**
- **Final recognizer** (30 epochs, option B): GW-REC-034, 035 and 036 read 6.30, 6.28 and 6.30%
  validation CER (mean 6.29%), against 7.53% for the CRNN baseline GW-REC-001 at the same budget:
  **−1.24 [−1.40, −1.11]** points; document slice 9.15% against 10.23%, −1.08 [−1.41, −0.78];
  GLYPH-C mCE 3.54, 3.53, 3.53%. The shipped checkpoint is GW-REC-034, as fixed with the plan.
- **Detector:** GW-DET-007 F1@0.5 0.7668 against GW-DET-001's 0.7779 on all 1,715 validation pages
  at Phase 9's settings, **−0.0111 [−0.0146, −0.0075]**; recall 0.7006 against 0.7157. **Not
  adopted**; GW-DET-001 stays (`reports/error_cases/detector/GW-DET-007/pair.json`).
- **Calibration** (GW-E2E-005, `reports/calibration/GW-E2E-005/calibration.md`): GW-E2E-003's method
  on GW-REC-034. Fitted T = 2.35; ECE on validation half B 0.214 at T = 1 and 0.089 at the
  fitted T (−0.125 [−0.130, −0.122]): adopted. It is the same grid value as GW-E2E-003's, so the temperature does not change.

**Frozen configuration:** `configs/final/glypharis_v1.yaml`, written by `scripts/freeze_final.py`
from the run summaries, the checkpoints and the reports above. It records the recognizer
(GW-REC-034: config, config hash, commit, seed, checkpoint SHA-256 `c7e745403d01…`, input height 48 and
maximum width 1536, temperature 2.35 from GW-E2E-005, training data hashes, library versions) and
the detector (GW-DET-001: checkpoint SHA-256 `016b2b69d257…`, long side 1536, GW-DET-004's
post-processing, operating point `inverse09`), with the validation results behind each choice.
The script refuses to freeze anything the rules did not choose. The checkpoints stay local and are
identified by their hashes; the test sets are still closed.

**Run notes**
- Every run trained with no diagnostic flags, except one `POSSIBLE OVERFITTING` flag for
  GW-DET-007 at epoch 9, which later epochs recovered from.
- The owner paused the chain many times, between steps or in the middle of a run; a stopped run
  resumed from its `last.pt`, and the run reports list each resume.
- Two incidents, both in `research/failures.md`: a stray `uv run` half-replaced the environment
  on 2026-11-07 (repaired to the lock, checked, resumed), and GW-REC-034 ran out of GPU memory in
  epoch 25 on 2026-11-10, most likely because another program took memory (resumed from step
  142,000).
- GW-REC-032 and 033 (option A) were not trained, as the plan says for option B.

**Compute:** 107.6 GPU-hours in the compute ledger (51.3 on jobs that did not finish: almost all
of it runs stopped by hand that then resumed, so only the steps after each checkpoint were
repeated), against the 75 to 90 estimated; the training time inside the runs is 97.2 GPU-hours
(`reports/compute/ledger.md`). The estimate left out the GLYPH-C scoring (3.4 hours) and the time
lost to stops.

**Tests:** the CPU suite passes locally; new tests cover the freeze script (the record is built
from the results, worktree paths stay out, and it refuses a detector or a seed the rules did not
choose), the validator's check of freeze records, and the ledger's Phase 13 lines.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 13)

| Criterion | Result | Evidence |
|---|---|---|
| Ablation table generated (baseline → augmentation → synthetic data → preprocessing → full model; one change per row; 3 seeds) | pass | `reports/ablation/phase13_ladder.md`, `research/ablations.md` |
| Final configuration chosen on validation | pass: option B, GW-REC-034 with GW-DET-001 | `reports/ablation/phase13_final.md`, `reports/error_cases/detector/GW-DET-007/pair.json` |
| Final configuration frozen with a full reproducibility record | pass | `configs/final/glypharis_v1.yaml` |
| CI green | pass | runs 43603968022 (branch), 43603989084 (pull request), 43606228642 (`main`) |

**Known issues**
- The ladder runs at the short budget; only the final configuration was trained at full length,
  and the CRNN baseline at the full budget has one seed.
- The pipeline's end-to-end numbers (GW-E2E-001) and the v0.1 bundle still use GW-REC-024. The v1
  bundle is built and checked when Phases 17, 18 and 20 are run again (ADR-0008's order).
- Whether GW-DET-007 found more small or rotated lines at the cost of others was not measured;
  Phase 15's error analysis can look.

**Git:** branch `phase/13-ablation`, merged through pull request #18. Merge commit `8659b69`, tag
`phase-13` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 43606228642).

---

## Phase 14 — External evaluation

**Status:** COMPLETE (2026-11-17). Opened
2026-11-14 on `phase/14-external-eval`, branched from `main` after
Phase 13 (merge `8659b69`). GDC-scan v1 was frozen on the branch before this plan (`5f5a3b5`); the
owner confirmed on 2026-11-14 that the printed document is the same PDF, with no edits after
printing. The external benchmarks prepared early on `phase/14-data` (`f354ba9`, 2026-10-29) are
merged into this branch (`3377c40`). The owner approved the plan and the download of both official
evaluators on 2026-11-15 (`data/manifests/evaluators.json`). Test data: one evaluation per frozen
model per set, every access logged in `research/test_access_log.md`.

**Objective:** one-shot evaluation of the frozen model (`configs/final/glypharis_v1.yaml`) on the
held-out and external benchmarks, against the Phase 5 baselines (RQ1, RQ4, RQ6, RQ8).

**Data prepared** (conversion only, done early on `phase/14-data`; nothing scored, so the
test-access log was untouched; `data/manifests/processed/benchmarks.json`)

| Benchmark | What is ready | Size |
|---|---|---|
| IIIT5K (test) | line pack of the published word crops and labels (upper case) | 3,000 words |
| FUNSD (test) | page file; entity words grouped into lines | 50 forms, 2,647 lines |
| OmniDocBench (English, pinned commit) | page file of text blocks in reading order, plus "don't care" regions | 755 pages, 11,369 blocks |
| HierText (test) | prepared in Phase 4; evaluation crop pack in Phase 5 | 1,634 pages |
| GDC v1.1 | frozen 2026-10-21 (106 phone captures) | 8,606 lines |
| GDC-scan v1 | frozen 2026-11-14 (28 phone photos of one document) | 779 lines |

**Plan** (registered 2026-11-15, before the frozen model or any GLYPHARIS model read a test or
external set. Already known when it was written: the baselines' Phase 5 numbers on HierText test
and GDC (`reports/baselines/`); the v0.1 pipeline's validation latency in PyTorch (GW-EFF-002) and
ONNX (GW-EFF-004); and that every baseline reads GDC's crops with a lower CER than HierText
validation's. No GLYPHARIS number exists on any test set.)

1. **What is evaluated.** The frozen configuration, built from the freeze record by code, not by
   hand: the recognizer GW-REC-034 (height 48, maximum width 1536, T = 2.35 from GW-E2E-005) and the
   detector GW-DET-001 at Phase 9's settings (long side 1536, GW-DET-004's post-processing,
   operating point `inverse09`, `quad` boxes), in PyTorch on the RTX 4050.
   - **Builder.** Reads `configs/final/glypharis_v1.yaml`, verifies both checkpoint SHA-256 values,
     checks that the character list stored in the GW-REC-034 checkpoint equals
     `data/manifests/charset.json` (SHA-256 `bbdfab42…`, recorded with this plan), and refuses to
     run on any difference. Both checkpoints are copied into this worktree at the recorded paths.
   - **Numerics.** The recognizer runs in strict fp32 (TF32 off, as the detector already does),
     with cuDNN benchmark off and deterministic algorithms on. T = 2.35 was fitted with TF32
     convolutions; GW-REC-038 reports validation ECE at T = 2.35 under the strict setting, as
     description only (its crop pack is not GW-E2E-005's half B). T stays 2.35 whatever any test
     result shows.
   - **Pinned code.** Once the code is complete and the preprocessing checks pass, the commit is
     tagged `phase14-eval` and recorded here, before any of the validation runs below. The pinned
     code is the tree of `glypharis/`, `scripts/`, `configs/` and `requirements/`. Every GLYPHARIS
     run in this phase, validation and test, records its commit and refuses to run unless those
     trees equal the tag's and nothing outside the tracked records (the access log, the outputs
     manifest, reports) is changed. A validation run made at another commit is run again at the
     tag; a pipeline change after the tag is not allowed.
   - `poly` boxes are not evaluated. The v0.1 bundle (GW-REC-024) is not the frozen model and is not
     used.

2. **Experiments** (full entries in `research/experiments.md` with this plan).

   | ID | What | Data | Test access |
   |---|---|---|---|
   | GW-E2E-006 | v1 pipeline, validation reference | HierText validation, all 1,715 pages | no |
   | GW-REC-038 | frozen recognizer on validation crops: H5′'s reference and the fp32 ECE check | HierText validation evaluation pack (74,500 crops, Phase 5) | no |
   | GW-EFF-005 | v1 CPU latency for H1′, back to back with a fresh GW-BASE-002 profile | Phase 5 validation profile pages (10 warm-up, 100 timed) | no |
   | GW-REC-039 | frozen recognizer on ground-truth crops; confidences at T = 1 and 2.35 | HierText test, GDC, GDC-scan, FUNSD, IIIT5K | yes |
   | GW-E2E-007 | frozen pipeline end to end; confidences at T = 1 and 2.35 | HierText test, GDC, GDC-scan, FUNSD, OmniDocBench (English) | yes |
   | GW-E2E-008 | oracle: ground-truth lines through the pipeline's recognizer (H4, Phase 15) | GDC, GDC-scan | yes |
   | GW-E2E-009 | the single scoring pass, comparisons, hypothesis verdicts | all prediction files above and the baselines' | via the scoring rows |
   | GW-BASE-001 to 007 | the baselines on the new sets (dated amendments to their entries) | IIIT5K, FUNSD, OmniDocBench, GDC-scan | yes |

3. **The sets and how each is scored.**

   | Set | Record | Unit | Metrics | Reading order | CI unit |
   |---|---|---|---|---|---|
   | HierText test | pages: `data/manifests/processed/hiertext.json` (Phase 4); crops: `eval_crops.json` (Phase 5) | line | recognition; detection; bag-of-lines, E2E F1; HierText's official line-level score | none | lines; pages |
   | GDC v1.1 | `data/manifests/gdc/gdc_v1.json`; crops `eval_crops.json` | line | recognition; detection; order-aware and bag-of-lines; E2E F1; vertical lines apart; per capture condition | PDF text layer | lines; pages |
   | GDC-scan v1 | `data/manifests/gdc_scan/gdc_scan_v1.json`; new crop pack | line | as GDC, without conditions | PDF text layer, footer last | lines; pages |
   | FUNSD | `data/manifests/funsd.json`, `data/manifests/processed/benchmarks.json`; new crop pack | our entity lines | recognition; detection; bag-of-lines, E2E F1 | none | lines; pages |
   | OmniDocBench, English | `data/manifests/omnidocbench.json` (data commit `aa1ee96d`) | layout block | order-aware page CER/WER; the official text-block and reading-order edit distances | published order, unordered blocks last | pages |
   | IIIT5K | `data/manifests/iiit5k.json` | word | word accuracy and CER, lexicon-free | n/a | words |

   - **Both normalisation tracks** everywhere, with the charset ceiling beside every metric.
   - **New crop packs** (GDC-scan, FUNSD) use `build_eval_crops.py`'s Phase 5 settings: lines
     selected by `is_rec_line` (legible, not handwritten, not vertical, trusted box; out-of-charset
     and too-long lines kept). FUNSD's page file marks 57 of its 2,647 lines out of charset, which is
     why its record says 2,590 recognition lines; its pack must hold 2,647. GDC-scan's must hold 779
     minus its vertical lines minus the 3 lines with untrusted boxes. Both counts are asserted.
   - **IIIT5K:** the labels are upper case, so the headline is the customary case-insensitive
     alphanumeric track (both sides lower-cased, then every character outside `0-9a-z` removed). The
     strict and normalised tracks are reported too, the strict one marked as charging case against
     upper-case labels. The 50-word and 1,000-word lexicons are not used. Images are scored as
     published (no margin added).
   - **OmniDocBench:** its blocks are paragraphs, so detection, bag-of-lines and E2E line F1 are
     n/a. The project's order-aware page CER/WER is reported, labelled "not the official protocol";
     beside it, the official evaluation's `text_block` and `reading_order` edit distances from the
     pinned code (`quick_match`, formula and table metrics switched off; its Overall score needs
     tables and formulas and is not computable). Differences between the two, stated now: the
     project's reference keeps headers, footers, page numbers and footnotes; blocks without a
     numeric order are put last; inline formulas stay as LaTeX source in the reference, and the
     charset ceiling cannot see that; the project matches nothing (one page string), the official
     code matches blocks. Rotated and handwritten blocks count as text.
   - **FUNSD:** lines are our grouping of FUNSD's entity words, not FUNSD's own unit; a question and
     its answer on one printed row are two reference lines. Handwritten fill-ins count as text. No
     reading order, so the order-aware metric is n/a.
   - **HierText's official score:** the official evaluator at commit `70b6620b` in its line-level,
     end-to-end mode (`--eval_lines --e2e`), for GLYPHARIS and every baseline with HierText test
     pages. That mode builds each line's mask from its words and compares line texts
     (`eval.py:85-118`, `:146-175`, `:205-224`, `:315-318`), so each predicted line is given to it as
     a line holding one word with the line's polygon and text. Its word-level score needs word
     output, which no system here produces, and is not reported.
   - **GDC-scan:** one document, so a case study; its CIs are conditional on that document. 87 of
     its 779 lines contain characters outside the charset; results are also given without those
     lines, as information only. Every tracked file carries numbers only, never its text or images.

4. **Baselines.**
   - **HierText test and GDC: reused, not run again.** The baselines are fixed third-party models
     already evaluated once on these sets in Phase 5. The baseline lock, runner, scorer and metric
     code are unchanged since `phase-05` (the main lock gained export packages only). Phase 14
     re-scores their files once, logged as Phase 14. Before any other number is computed, the
     re-scored totals (predictions and the efficiency block) must equal
     `git show phase-05:reports/baselines/results.json` exactly, and every stored reference must
     equal the current pack's label; if not, scoring stops, and reading files to find out why counts
     as a logged access. The Phase 5 files are hashed before any Phase 14 run and never written.
   - **New sets:** same runner and settings, no tuning; before any new-set run, every field of the
     runner's settings must equal those recorded in each system's Phase 5 run metadata (the
     `*.meta.json` and profile files only, never the test predictions). Phase 14 predictions and
     profiles go to `data/processed/baselines_phase14/<ID>/`. Coverage, fixed now:

     | System | IIIT5K | GDC-scan rec | GDC-scan e2e | FUNSD rec | FUNSD e2e | OmniDocBench e2e |
     |---|---|---|---|---|---|---|
     | GW-BASE-001 Tesseract | yes | yes | yes | yes | yes | all 755 |
     | GW-BASE-002 PP-OCRv5 mobile | yes | yes | yes | yes | yes | all 755 |
     | GW-BASE-003 PP-OCRv5 server | yes | yes | no: died on GDC's 3072 × 4080 photos in Phase 5 | yes | yes | the 100-page subset |
     | GW-BASE-004 TrOCR-small | yes | yes | no detector | yes | no detector | no detector |
     | GW-BASE-005 PP-OCRv6 tiny | yes | yes | yes | yes | yes | all 755 |
     | GW-BASE-006 PP-OCRv6 small | yes | yes | yes | yes | yes | all 755 |
     | GW-BASE-007 PP-OCRv6 medium | yes | yes | yes | yes | yes | all 755 |

     The OmniDocBench subset is `random.Random(20261115).sample(sorted(image names), 100)` over the
     755 English pages in `data/manifests/omnidocbench.json`; the sorted list is committed with the
     plan. Every system is also reported on it, so PP-OCRv5 server is compared on equal pages. If a
     system dies on a set, as PP-OCRv5 server did on GDC, it is recorded as not runnable there, with
     no substitute.

     **Outcome (2026-11-17):** PP-OCRv5 server died on the subset as well. Its detector fails on
     any input above about 8 million pixels, and 5 of the 100 pages are that large
     (`research/failures.md`, 2026-11-17). Under the rule above it is recorded as not runnable on
     OmniDocBench, with no substitute (the owner's decision, 2026-11-17); the subset view
     compares GLYPHARIS and the other five baselines on the same 100 pages.

5. **Hypotheses judged here.** New dated versions in `research/hypotheses.md`; H1, H5 and H8 are
   marked superseded and stay as written. H2′ gets a note: it cannot be tested on the frozen model,
   which has no GlyphSynth. The confirmatory tests are exactly five (H1′ CER, H1′ latency, H5′, H8′
   transfer, H8′ by half); every other interval is descriptive and carries no verdict wording.
   Each verdict is **met** if its CI lies on the claimed side, **not met** if on the other side,
   otherwise **not shown**.
   - **H1′ (RQ1, RQ4).**
     - *CER half.* The GDC clean slice is the `scan_app` and `photo_flat` captures (53 pages; the two
       conditions with no deliberate degradation), joined to pages by `Page.image` =
       `alignment.json` `file`, with the condition counts asserted (37, 16, 16, 11, 11, 8, 7).
       Bag-of-lines page CER, strict track: `paired_ratio_diff_ci(bag_errors_G, 2 × bag_errors_P,
       bag_ref_chars, "pages")`, 1,000 resamples, default seed, all 53 pages; met if the CI lies
       below 0. Judged also on the normalised track and with a source-document cluster bootstrap
       (21 documents); if any of the three verdicts differs, the result is not shown. A page missing
       from either file blocks the comparison until the pass is completed; empty output counts as all
       deletions. The order-aware CER, every capture condition and the external page sets are
       reported with the same ratio, as information.
     - *Latency half.* GW-EFF-005 (frozen pipeline, PyTorch CPU, the v1 default backend under
       ADR-0011) and a fresh GW-BASE-002 profile in one sitting: the baseline first, then
       GLYPHARIS, on mains power, the Balanced power plan, the power mode recorded, no other user
       program running. Both at 10 threads (PaddleOCR's default, as in Phase 5; asserted from each
       file's settings), batch 1, the same 10 warm-up and 100 timed validation pages in the same
       order; pairing by position, after asserting that the input sizes match page by page.
       `mean_ci` of the per-page differences, unit pages; met if the CI lies below 0. A sitting is
       void only if a recorded condition above fails or the input sizes differ; the first sitting
       that is not void is the one of record, and a void one goes to `research/failures.md`. Both
       profile files are hashed and both `times_ms` arrays go into the report. Disclosed beside the
       result: the baseline runs PaddleOCR's two orientation models, as it did for its accuracy
       numbers. The Phase 5 profile is shown as a consistency check, and GLYPHARIS at 16 threads as
       information. This half largely re-confirms a known validation result (GW-EFF-002). The ONNX
       build of v1 does not exist yet; when Phases 16 and 17 build it, it is judged by the same rule
       and reported as a separate verdict.
     - H1′ holds only if both halves are met.
   - **H5′ (RQ6).** "Markedly higher": the frozen recognizer's strict CER on GDC's crops
     (GW-REC-039) is at least 1.5 times its CER on the HierText validation evaluation pack under the
     same code and precision (GW-REC-038). Test: CI of CER_GDC − 1.5 × CER_val (independent
     bootstraps, lines, new `unpaired_ratio_diff_ci`); met if above 0. A cluster bootstrap on the
     GDC side (source documents) must give the same verdict, otherwise not shown. FUNSD, GDC-scan and IIIT5K, and HierText
     test, are reported for information. The GlyphSynth half cannot be tested on the frozen model;
     Phase 12's validation evidence stands for it.
   - **H8′ (RQ8).** On GDC's crops (GW-REC-039; vertical lines excluded as in every GDC metric,
     out-of-charset lines kept), GW-E2E-005's method (15 equal-mass bins, exact match on the strict
     track):
     - *transfers:* the CI of ECE(T = 2.35) − ECE(T = 1) lies below 0;
     - *transfers by half:* the CI of ECE(T = 2.35) − 0.5 × ECE(T = 1) lies below 0.
     Line bootstrap as GW-E2E-005, with a page-cluster bootstrap that must give the same verdict,
     otherwise not shown. H8 counts as
     supported on GDC only if both are met. Reliability diagrams for every set. GDC pipeline lines,
     GDC-scan, HierText test and FUNSD are information; IIIT5K on the alphanumeric track.
   - H4 is tested in Phase 15 from GW-E2E-008.

6. **Rules fixed in advance.**
   - **One evaluation** per system, set and task: one prediction pass and one scoring pass. A
     resume after a crash belongs to the same evaluation and is logged as a resume.
   - **Order.**
     1. This plan, the code and its tests are committed and pushed, and CI is green. Before the
        push, the diff is searched for the strings that would identify GDC-scan's author, taken from
        what is already known, not by opening its files.
     2. Preprocessing: the new crop packs are cut (counts asserted); every prepared file is checked
        against its recorded hash; page line ids are checked for uniqueness; GDC-scan's photos are
        checked for a rotation tag and their size against the alignment record; GDC's
        `alignment.json` capture hashes are checked against `gdc_v1.json`. Any failure is fixed and
        the check repeated here, before the tag; nothing in this step looks at a model output.
     3. The `phase14-eval` tag is set.
     4. GW-E2E-006, GW-REC-038 and the GW-EFF-005 sitting, on validation, at the tag. The GPU
        estimate is updated from GW-E2E-006.
     5. Every prediction pass, GLYPHARIS and baselines, from this worktree on this branch. The
        Phase 15 per-line attributes are computed inside these logged passes. The access log is
        committed after each batch.
     6. One scoring pass, with the scorer commit recorded in the report. No score is computed or
        printed before every prediction file is complete: the test runners write predictions only.
   - **Not an access:** the preprocessing in step 2 (as in Phase 5).
   - **Log rows.** Prediction rows carry the producing ID; scoring rows follow Phase 5: one row per
     scored file under the producing ID, checkpoint column `<system> (<task>), scoring`, phase 14,
     reason `Phase 14 scoring (GW-E2E-009)`. The runners take the split from the set's records,
     resolve the log from the repository root, write repo-relative paths only, and refuse to start
     outside `phase/14-external-eval`.
   - **Void runs.** A run is void only on a mismatch found mechanically, before the scoring pass,
     between its provenance (commit, hashes, settings) and this plan. A mismatch found after scoring
     is reported beside the original result, never in its place.
   - **Scorer fixes** after scoring: only through a protocol change row written first; every system
     and set is re-scored, and both versions are reported.
   - **No change after scoring:** no metric, slice, threshold or setting changes once a score has
     been seen. Any change to the model or its settings after this point is a new model version,
     evaluated again and disclosed.
   - **Single seed.** The frozen model is one model by design (the seed-1337 run, fixed before
     training); running the other seeds on test would break the one-evaluation rule. Its
     comparisons with the baselines are between fixed systems. The CIs leave out training variance;
     the final recipe's seed spread is 0.02 points of validation CER, recognizer only (GW-REC-037);
     the detector has one seed.
   - **Outputs.** Per-sample files (predictions, per-line confidences, the oracle run, matched pairs,
     the Phase 15 hand-off, logs) go under `data/processed/phase14/<ID>/`, git-ignored and in the
     shared data folder, so removing a worktree cannot delete them. Their SHA-256 and line counts
     are committed in `data/manifests/processed/phase14_outputs.json`, and Phases 15 and 22 check
     them before reading. Any backup is the owner's own and private, because the files hold
     GDC-scan text. Only aggregates go under `reports/external/`. GDC-scan contributes numbers only
     to every tracked file and commit message.
   - **Phase 15 reads only what Phase 14 saves** and runs no model on test data: per-line
     predictions with references; per-line attributes computed now (crop height, blur as variance
     of the Laplacian, contrast as grey-level standard deviation, box rotation, detector match and
     IoU, capture condition); the oracle run; matched pairs; and, for examples, GDC line crops of
     every line with an error plus 200 correct lines drawn with seed 20261115. Nothing from
     GDC-scan beyond numbers.
   - **Phase 22 tolerance, fixed now.** Re-scoring the saved files gives identical numbers. A re-run
     of the frozen model from the prepared files (each matching its recorded hash) with the same
     flags gives every aggregate metric within 0.05 percentage points (near-tie flips on the GPU
     allowed within that bound). An fp32 ONNX bundle within Phase 17's tolerance on validation is the
     same model in another runtime; an INT8 bundle is a new model version and is evaluated in Phase
     22 if it ships.

7. **Code needed before any access** (each with tests):
   - merge `f354ba9` (conflicts in `DATASETS.md` and `docs/PHASE_STATUS.md`; its line calling
     GDC-scan "transcribed" is corrected);
   - the frozen-config builder (hash and charset checks, strict fp32 recognizer, pinned code);
   - the profiler: built from the freeze record, a `--threads` option (torch intra- and inter-op and
     the OpenMP variable) recorded and asserted in its output; the baseline profile's thread count
     asserted at 10 from its settings;
   - a guarded GLYPHARIS test runner (recognition, pipeline, oracle via `read_quads`) that writes
     predictions in the baselines' schema plus confidences and computes no metric;
   - the baseline runner and scorer: required `--phase`, no default reason, set filters, the new
     sets (reading order on for GDC-scan and OmniDocBench, off for FUNSD; the vertical-line view
     for GDC-scan), a separate output directory, refusal to write under `reports/baselines` or to
     score unselected files when the phase is not 5;
   - crop-pack entries for GDC-scan and FUNSD;
   - the alphanumeric track and word accuracy; `unpaired_ratio_diff_ci` and a cluster bootstrap in
     `stats.py`; an ECE scorer for test crops; the GDC condition join;
   - wrappers for the official HierText and OmniDocBench evaluators in their own environments;
   - the Phase 15 hand-off writer; the Phase 14 report, which writes no GDC-scan text (a test plants a
     marker string and checks it never reaches a tracked output);
   - `.gitignore` rules for `*.jsonl`, `*.jsonl.gz` and `*.log` under `reports/` and `research/`.

8. **Records updated with this plan:** experiment entries (GW-E2E-006 to 009, GW-REC-038 and 039,
   GW-EFF-005; dated Phase 14 amendments to GW-BASE-001 to 007, their rows back to running);
   H1′, H5′, H8′ and the H2′ note; EXPERIMENT_PROTOCOL change rows (order-aware metric on
   OmniDocBench and GDC-scan; OmniDocBench at page level with the official components; HierText's
   official score at line level only, replacing the earlier word-level plan; the IIIT5K tracks; the
   single-seed exception; the access unit, void and re-scoring rules; Phase 15's data; the
   sub-0.3-alignment wording; the real access guard instead of `--test-access`; the split table);
   ADR-0013 giving GDC-scan its role (internal test, case study); the IMPLEMENTATION_PLAN Phase 14
   task list and its old-scans line; the risk register (R-17, R-18, R-22); the GDC-scan source added
   to the training exclusions by hash only; the owner's same-PDF confirmation in its dataset card;
   `data/sources.yaml` entries for both official evaluators; a research-log entry.

9. **Not run (documented with the reason):**
   - HierText's word-level official score (no system outputs words).
   - OmniDocBench's Overall score (needs tables and formulas).
   - SROIE: skipped by the owner on 2026-10-29 (needs an ICDAR account; licence not public).
   - PP-OCRv5 server end to end on GDC-scan (failed on photos of this size in Phase 5) and on all but
     100 OmniDocBench pages (cost: about 81 s per page in Phase 5, `reports/baselines/efficiency.md`).

10. **Amendment (2026-11-15, from two reviews of the Phase 14 code; written before the tag and
    before any Phase 14 run, with nothing predicted or scored).** Where it differs, it replaces
    the text above.
    - **Records after the tag.** Besides the access log, the outputs manifest and `reports/`,
      `research/failures.md` may change after the tag (a void sitting is recorded there). Every
      other path is pinned, renames included: a file moved out of a pinned path counts as changed.
      The tag commit and the GPU re-estimate from GW-E2E-006 are written to
      `reports/external/run_record.md`, not to this file; this section is brought up to date after
      the scoring pass.
    - **Order, step 2,** in this order: cut the GDC-scan and FUNSD packs (the cutter itself asserts
      FUNSD's 2,647 and GDC-scan's count, with the untrusted lines equal to `alignment.json`'s 3,
      and records each pack's label-file hash); `scripts/phase14_prepare.py record-index-hashes`
      (adds the label-file hash to the packs prepared earlier); `scripts/phase14_score.py
      hash-phase5`; commit those records; then `scripts/phase14_prepare.py check` must pass. It
      checks every input against its recorded hash and count (GDC's page count now from
      `gdc_v1.json`), page and line id uniqueness, every page image's header size against its
      record (JPEG, PNG and WebP only, orientation from the EXIF block only), GDC-scan's photos
      (no rotation tag, size and hash as `alignment.json` records), GDC's capture hashes, the
      batch each planned baseline run must use against Phase 5's run metadata, and that the
      records under `data/manifests` are committed. Both record commands refuse once the tag
      exists.
    - **Runners.** Every run, GLYPHARIS and baseline, refuses before logging an access when an input
      has no recorded hash or differs from it, and a GLYPHARIS page run also when an image's size
      differs from its record. Every GLYPHARIS pass runs on device `cuda`, validation included; its
      recognition batch is recorded. A baseline run uses the inference batch its Phase 5 runs used.
      Tesseract's temporary crops stay under the git-ignored `data/cache/`.
    - **Latency sitting.** Both 10-thread profiles record when they started and ended. The sitting
      is also void if GW-EFF-005 starts before the baseline profile ends, more than 2 hours after
      it ends, or if either time is missing. The 16-thread profile is information and cannot make
      the sitting void.
    - **The scoring pass** checks before its first row everything that needs no score: completeness,
      the Phase 5 and Phase 14 hashes, provenance (one tag, the frozen files, one device and
      software stack), the latency sitting, every input against its record and against each
      GLYPHARIS run's record of it, the official evaluators' environments and ground-truth hashes,
      and a conversion of every prediction file into the evaluators' formats. A refusal writes
      nothing. If the pass stops after rows are logged, it is finished with `score --resume`, the
      same evaluation, and every row of the resumed pass is logged with "(resumed)"; a finished
      pass refuses any further run.
    - **GDC-scan without its out-of-charset lines** (item 3) covers recognition and the page metrics
      (bag-of-lines CER, E2E F1, with those reference lines treated as don't-care), information
      only.
    - **Phase 5 is closed** once its files are hashed: the Phase 5 scoring command and Phase 5
      baseline runs refuse, and Phase 14 files are scored only by `phase14_score.py score`.

**Cost on the laptop** (estimates from `reports/baselines/efficiency.md`, not records). Baselines:
OmniDocBench pages for the four full systems about 2 CPU-hours, PP-OCRv6 medium on all 755 about 7,
PP-OCRv5 server on 100 about 2; the slow recognizers on the roughly 6,400 IIIT5K, FUNSD and
GDC-scan crops about 5; the rest under 2: about 18 CPU-hours in total. GLYPHARIS: about 1 GPU-hour
(Phase 10 measured 449 ms per validation page with the v0.1 recognizer, GW-E2E-001; re-estimated
from GW-E2E-006 before any test run). Validation runs and profiles about 2 hours. Two new
environments for the official evaluators.

**Known limitations, stated in advance**
- GDC-scan is one document and FUNSD 50 forms; their CIs are wide and say nothing about other
  documents.
- Only components of OmniDocBench's official protocol are run, and HierText's only at line level.
- OmniDocBench and FUNSD count rotated and handwritten text that GLYPHARIS does not try to read.
- H1′'s latency half is measured on HierText validation pages (GLYPHARIS detected 40.9 lines per
  page there, GW-EFF-002) because no document validation set exists; GDC pages hold about 81
  ground-truth lines. How either
  system's page time grows with line count is not measured, so the verdict may not carry over to
  dense pages. The PyTorch default was set after the v0.1 ONNX latency was known (GW-EFF-004).
- The H5′ threshold was set knowing the baselines' direction (lower CER on GDC than on HierText
  validation for every baseline).
- Training-time validation ran with bf16 autocast, so GW-REC-034's logged 6.30% is not the H5′
  reference; GW-REC-038 is.

**Results (2026-11-17)** (GW-E2E-009, `reports/external/`; one scoring pass, finished as a logged
resume; strict track unless stated; 95% bootstrap CI, 1,000 resamples, seed 20261010)

*Phase 5 reproduction.* The 37 Phase 5 entries and 13 profiles re-scored in this pass equal
`phase-05:reports/baselines/results.json` exactly, and all 516,474 stored references equal the
current pack labels.

*The five confirmatory tests* (plan item 5)

| Test | Rule (CI must lie…) | Result | Verdict |
|---|---|---|---|
| H1′, CER half | CER(GLYPHARIS) − 2 × CER(PP-OCRv5 mobile) below 0; GDC clean slice, bag of lines, 53 pages | −34.98 [−47.44, −24.08] points; normalised −34.73 [−47.17, −23.83]; 21 source documents −34.98 [−50.22, −20.73] | **met** (all three agree) |
| H1′, latency half | GLYPHARIS − PP-OCRv5 mobile per page below 0; CPU, 10 threads, 100 validation pages, one sitting (GW-EFF-005 against a fresh GW-BASE-002 profile) | −508.1 [−737.7, −256.5] ms; means 2,446.7 and 2,954.8 ms | **met** |
| H5′ | CER_GDC − 1.5 × CER_val above 0 | 4.17% on GDC's crops against 6.51% on validation's: −5.59 [−6.07, −5.01] points (lines), −5.59 [−7.88, −2.52] (documents) | **not met** |
| H8′, transfers | ECE(T = 2.35) − ECE(T = 1) below 0; GDC crops | −0.125 [−0.132, −0.119] (lines), [−0.142, −0.101] (pages) | **met** |
| H8′, by half | ECE(T = 2.35) − 0.5 × ECE(T = 1) below 0 | −0.0081 [−0.0142, −0.0021] (lines, met); [−0.0168, +0.0044] (pages, not shown) | **not shown** |

- **H1′ holds**: both halves are met. Disclosed with the latency half: the baseline runs PaddleOCR's
  two orientation models, as it does for its accuracy numbers. The Phase 5 profile of the same
  baseline read 3,408.2 ms per page; GLYPHARIS at 16 threads 2,107.1 ms (information).
- **H5′ is not met**: on ground-truth crops the frozen recognizer reads GDC's documents with a
  lower CER than HierText validation, as every Phase 5 baseline already did.
- **H8 is not supported on GDC**: the temperature lowers GDC's ECE (0.234 to 0.109), but the page
  bootstrap cannot show that it halves it.

*Validation references* (no test access). GW-E2E-006, all 1,715 validation pages: bag-of-lines
page CER 27.57% [25.11, 30.36], E2E F1 58.24% [56.64, 59.91], detection F1@0.5 77.79%
[76.21, 79.19] (GW-DET-001's Phase 13 value, 0.7779). GW-REC-038, 74,500 validation crops: 6.51%
[6.36, 6.65] strict (6.23% normalised), against the 6.30% logged in training with bf16 autocast on
the training pack; ECE 0.228 at T = 1 and 0.090 at T = 2.35.

*Descriptive comparisons* (no verdicts; GLYPHARIS crops GW-REC-039 and pages GW-E2E-007 minus the
baseline in percentage points, paired over the same lines or pages,
`reports/external/comparisons.md`)

| Set | Unit | GLYPHARIS | Strongest baseline on that row | Difference |
|---|---|---|---|---|
| HierText test, crops | CER, 67,574 lines | 6.03% [5.89, 6.18] | PP-OCRv6 small 7.06% | −1.03 [−1.14, −0.92] |
| GDC, crops | CER, 8,372 lines | 4.17% [3.72, 4.67] | PP-OCRv6 medium 3.20% | +0.97 [+0.76, +1.20] |
| GDC-scan, crops | CER, 776 lines | 1.02% [0.85, 1.20] | PP-OCRv6 small 0.37% | +0.65 [+0.51, +0.78] |
| FUNSD, crops | CER, 2,647 lines | 7.03% [5.71, 8.66] | PP-OCRv6 small 9.30% | −2.27 [−3.43, −1.27] |
| IIIT5K | word accuracy, alphanumeric, 3,000 words | 91.47% [90.43, 92.47] | PP-OCRv6 medium 97.13% | −5.67 [−6.53, −4.77] |
| HierText test, pages | bag-of-lines CER, 1,634 pages | 25.16% [23.44, 26.79] | PP-OCRv6 small 42.93% | −17.77 [−19.51, −16.01] |
| GDC, pages | bag-of-lines CER, 106 pages | 22.35% [15.87, 30.11] | PP-OCRv6 medium 20.96% | +1.39 [−2.71, +6.88] |
| GDC, pages | order-aware CER | 25.37% [20.00, 30.80] | Tesseract 31.90%; PaddleOCR 42.86 to 44.74% | −6.53 [−13.51, +0.08] against Tesseract |
| GDC-scan, pages | bag-of-lines CER, 28 pages | 4.60% [2.10, 8.21] | PP-OCRv6 medium 4.94% | −0.34 [−1.40, +0.29] |
| GDC-scan, pages | order-aware CER | 11.34% [7.10, 16.70] | PP-OCRv6 medium 15.32% | −3.98 [−14.50, +4.40] |
| FUNSD, pages | bag-of-lines CER, 50 pages | 32.05% [24.28, 39.85] | PP-OCRv6 medium 27.82% | +4.23 [−0.52, +8.76] |
| OmniDocBench, pages | order-aware CER, 755 pages | 27.28% [23.95, 30.57] | Tesseract 15.70% | +11.58 [+8.37, +14.88] |

- **Official evaluators** (`reports/external/official.md`). HierText test, line level, end to end:
  E2E F-score 0.5038 for GLYPHARIS against 0.3735 for PP-OCRv6 small, the best baseline there
  (detection F-score 0.7837 against 0.5555). OmniDocBench, all 755 pages: text-block edit distance
  (page average) 0.1007 (Tesseract 0.0966, the PaddleOCR systems 0.4039 to 0.4217) and reading-order edit distance
  0.2443 (Tesseract 0.2352, PaddleOCR 0.4374 to 0.4490); lower is better.
- **HierText** carries over from validation: page CER 25.16% on test against 27.57% on validation,
  crops 6.03% against 6.51%. GLYPHARIS is ahead of every baseline run there on crops, pages and
  the official score.
- **Documents.** On GDC's crops PP-OCRv6 small and medium and PP-OCRv5 mobile read better than
  GLYPHARIS, and on GDC-scan's PP-OCRv6 small and medium do; PP-OCRv6 tiny is level on both (paired
  CIs across 0). On FUNSD's crops GLYPHARIS reads best. End to end on GDC, its bag-of-lines page CER
  is level with the three PP-OCRv6 systems (paired differences −1.83 to +1.39 points, every CI
  across 0), and its order-aware CER is 17.5 to 19.4 points lower than each of the four PaddleOCR
  systems run there, which return lines sorted top to bottom (see OmniDocBench below). On GDC-scan
  it is level with PP-OCRv6 medium and below tiny and small. On FUNSD forms its bag-of-lines page
  CER is level with every PaddleOCR system's (every paired CI across 0), but its detection F1@0.5,
  79.07% [75.56, 82.16], is below all of theirs (81.81 to 85.03%): with the better crops, it finds
  the lines less well.
- **GDC rotated pages** (7 pages, information): end to end GLYPHARIS reads them at 80.53%
  bag-of-lines CER, against 1.70% for PP-OCRv6 small, which runs a document-orientation classifier.
  With ground-truth boxes (GW-E2E-008) GLYPHARIS's recognizer reads them at 0.83%. GLYPHARIS has no
  page-orientation step.
- **OmniDocBench.** The four PaddleOCR systems, with recognizers of very different strength, all
  read about 60% order-aware CER (60.29 to 60.62%) and 0.40 to 0.42 text-block edit distance. That fits
  their line order: PaddleOCR sorts boxes by top edge, then left edge, so the columns of a
  multi-column page interleave. This explanation is read from PaddleOCR's code and was not
  measured. Tesseract, which analyses the page layout, reads all 755 OmniDocBench pages best on
  every measure. On the 100-page subset GLYPHARIS's page-average text-block edit distance is
  slightly lower (0.1063 against 0.1108) and its reading order level (0.2523 against 0.2525).
- **Calibration** (`reports/external/calibration.md`): T = 2.35 lowers ECE on HierText test (0.223
  to 0.092), FUNSD (0.338 to 0.123), GDC-scan (0.202 to 0.125) and GDC's pipeline lines (0.415 to
  0.249). On IIIT5K, where ECE at T = 1 is 0.055, it rises to 0.112 (+0.057 [+0.043, +0.072]).
- **Oracle** (GW-E2E-008, for H4 in Phase 15): with ground-truth boxes the recognizer reads GDC at
  4.17% bag-of-lines page CER and GDC-scan at 1.01%, against 22.35% and 4.60% end to end.
- **Not run**, as planned or by the registered rule (results.json `not_run`): TrOCR end to end
  (no detector); PP-OCRv5 server end to end on GDC-scan (Phase 5 failure) and on OmniDocBench (not
  runnable, above). PP-OCRv5 server has no GDC page result either: its Phase 5 file holds no
  predictions (`run_record.md`). Not computable: HierText's word-level score and OmniDocBench's Overall.

**Run notes** (detail in `reports/external/run_record.md`)
- The tag `phase14-eval` is commit `6795186`. Every GLYPHARIS run had its pinned code trees equal
  to the tag's: the validation runs at the tag commit itself, the test runs at later run-record
  commits, which record `git_dirty` for uncommitted run records.
- GW-BASE-007 on OmniDocBench was suspended for two hours when a power cut put the laptop to sleep,
  then died out of virtual memory on 2026-11-16 after 513 pages; after the owner restarted the
  laptop it resumed from those 513 pages, logged as a resume, and finished all 755.
- PP-OCRv5 server died twice on the OmniDocBench subset; the second death, with memory to spare,
  led to the size diagnosis on validation images and the not-runnable record above.
- The scoring pass stopped in HierText's official evaluator at Apache Beam's five-minute job
  deadline, after its scoring rows were logged and before any score was written or read. The
  evaluator is now started through `scripts/hiertext_eval_launch.py`, which lifts only that
  deadline (identical score files on the first 300 validation pages, and the full validation
  split finishes in 543 s), and the pass was finished with `score
  --resume`, every row logged "(resumed)". Both scoring-side changes are in the protocol change
  log (2026-11-17).
- results.json records the scorer commit `087252e` as dirty because the scorer reads the git state
  after writing its own access-log rows and outputs manifest.

**Compute:** GLYPHARIS 1.17 GPU-hours of run time (validation 0.31, test 0.86), against about half
a GPU-hour re-estimated from GW-E2E-006. The baselines' completed run segments took 21.8 hours
on the CPU from their run metadata; on top of that, GW-BASE-007's first OmniDocBench segment ran
about 15.7 hours before it died after 513 pages (about 2 of them asleep), and PP-OCRv5 server's
second subset run about 5.3 hours before its death. The estimate was about 18 CPU-hours. The scoring pass took 4 hours 16 minutes after the resume.
Inference only, so none of it is in the training compute ledger.

**Tests:** the CPU suite passes in a clone without tags (890 passed, 4 skipped, at `a392238`); new tests cover the not-runnable entry and the evaluator launcher, and
the evaluator tests pass against the real HierText and OmniDocBench evaluators on the laptop. In a checkout that carries the `phase14-eval` tag, the scorer's tests fail
at their fixture, because `hash-phase5` refuses once the tag exists (known issue below); CI's
checkout has no tags.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 14)

| Criterion | Result | Evidence |
|---|---|---|
| Benchmark tables with CIs: IIIT5K, FUNSD, OmniDocBench (official components where compatible), HierText test, GDC, GDC-scan; identical runs for the baselines | pass; SROIE skipped by the owner; PP-OCRv5 server not runnable on OmniDocBench | `reports/external/` |
| Incompatibilities explained (granularity, normalisation, charset) | pass | plan item 3; the notes in each report; the charset ceiling beside every metric |
| Each benchmark evaluated once per frozen model, logged | pass; resumes are logged as resumes | `research/test_access_log.md`, `reports/external/run_record.md` |
| No tuning after seeing results | pass: no model, setting, metric or slice changed; the two scoring-side code changes came before any score was read | protocol change log, 2026-11-17 |
| CI green | pass | runs 44039110860 (branch), 44039180576 (pull request), 44042627595 (`main`) |

**Known issues**
- GDC-scan is one document and FUNSD 50 forms; their intervals say nothing about other documents.
- `calibration.md` prints the H8′ difference on lines only; the by-half and page-bootstrap
  intervals are in results.json (`calibration.sets.gdc_test`) and in the table above. A scorer fix
  after scoring would mean a protocol change and a full re-score, so the report stays as generated.
- The scorer's tests need a checkout without the `phase14-eval` tag (see Tests).
- The PaddleOCR line-order explanation on OmniDocBench is not measured.
- GLYPHARIS has no page-orientation step, which costs it the rotated GDC pages; that is for a later
  model version, not this one.
- H1′'s latency half was measured on HierText validation pages; the ONNX build of v1 is judged by
  the same rule when Phases 16 and 17 build it. H4 is tested in Phase 15 from the saved files.

**Git:** branch `phase/14-external-eval`, merged through pull request #19. Merge commit `eb15025`, tag
`phase-14` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 44042627595).

---

## Phase 15 — Error analysis

**Status:** COMPLETE (2026-11-17). Opened
2026-11-17 on `phase/15-error-analysis`, branched from `main` after Phase 14 (merge `eb15025`). No
model runs on test data in this phase.

**Objective:** which failure modes dominate the frozen v1 pipeline, and whether they come from the
detector or the recognizer (RQ5): a categorised error report from the files Phase 14 saved, and the
paper's error-analysis section.

**Plan** (registered 2026-11-17, before any Phase 15 code reads a per-line file. Already known when
it was written: every aggregate in `reports/external/` and its results.json, among them the
per-condition GDC page CERs of the pipeline, GW-E2E-007, and of the oracle, GW-E2E-008, scored in
GW-E2E-009 (`reports/external/end_to_end.md`, "GDC per capture condition"), which put the oracle's
page CER below half the pipeline's in every condition; the rotated-page failure (F24); and that the
hand-off stores no character alignments, no merge or split labels, and no text for GDC-scan.)

1. **Inputs, and only these.** Each file is checked before it is read: Phase 14 files against
   `data/manifests/processed/phase14_outputs.json` (`file_entry` and `check_outputs` from
   `scripts/phase14_score.py`, a non-empty result refusing), the Phase 5 files with the read-only
   `verify_phase5`, and each example crop against the SHA-256 in `gdc_crops/index.jsonl`. Any
   mismatch stops the run before a row is read.
   - the hand-off, `data/processed/phase14/handoff/`: `rec_lines_<set>`, `e2e_pairs_<set>`,
     `pages_<set>`, `gdc_crops/index.jsonl` and its crops;
   - the Phase 5 baseline recognition files on GDC (GW-BASE-001 to 007) and HierText test
     (GW-BASE-001, 002, 004, 005, 006), keyed like the crop packs;
   - the baselines' OmniDocBench end-to-end files (GW-BASE-002, 005, 006, 007), for the line order
     of their output only (item 5): polygons, never compared with any reference;
   - validation: GW-REC-038's `hiertext_val.jsonl` (74,500 crops), and GW-REC-041 below.
   Never read: test images or page files, any GDC-scan text, and the GLYPHARIS prediction files
   (`data/processed/phase14/GW-*/`; what Phase 15 needs from them is in the hand-off). No model,
   detector or scorer pass touches a test set. Reads are recorded in the Phase 15 results.json
   (file, verified SHA-256, lines, time), not in the test access log: Phase 15 evaluates nothing
   (protocol change log, 2026-11-15 and 2026-11-17).

2. **Experiments** (new IDs; Phase 9 filed its analysis under the analysed run's ID instead).

   | ID | What | Data |
   |---|---|---|
   | GW-REC-040 | recognition errors of the frozen recognizer: edit operations, confusions, word errors, slices, categories, examples | `rec_lines_*` (HierText test, GDC, FUNSD, IIIT5K; GDC-scan numbers only); the Phase 5 baseline rec files; GW-REC-038 (validation) |
   | GW-E2E-010 | pipeline errors: the page error budget, unmatched lines, cropping error, detection slices, reading order, the oracle share (H4) | `pages_*`, `e2e_pairs_*`, `rec_lines_*`; GDC and GDC-scan with the oracle, HierText test, FUNSD and OmniDocBench without |
   | GW-REC-041 | the frozen recognizer on GlyphSynth validation lines, which carry font, italic, inverted, neighbour and capture-step labels: the "unusual font", "touching", "broken" and "background noise" slices the real sets have no labels for | GlyphSynth v1 validation (synthetic; validation only) |

3. **Shared rules for every table.**
   - Strict track first, normalised beside it, the charset ceiling beside every CER; IIIT5K on its
     alphanumeric track. Lines whose normalised reference is empty are left out (as Phase 14).
   - Every slice row carries lines, pages, reference characters, character errors, the CER and the
     slice's **share of the set's character errors** (what "dominates" means here).
   - **CIs:** 95% percentile bootstrap, 1,000 resamples, seed 20261010, from `stats.py` unchanged.
     On page sets the cited CI resamples pages (`ratio_ci` over per-page sums; pages with no line in
     the slice stay in as 0/0), with the line bootstrap beside it; GDC condition rows resample source
     documents; IIIT5K and GlyphSynth resample lines. A cell with fewer than 30 lines or 5 pages shows
     its counts only. Document groups get per-document CERs and no CI (2 to 9 documents each).
   - **Bin edges come from validation and are fixed before any test row is read**: quartiles of
     GW-REC-038's values (`np.quantile`, duplicate edges merged) for line height in page pixels,
     crop blur, crop contrast, line length in characters and line confidence at T = 2.35. The same
     edges apply to every set; the report prints them with the counts. Box rotation uses
     `rotation_deg` (the top-edge angle), folded to |θ|: < 2°, 2–5°, 5–15°, ≥ 15°, shown with and
     without GDC's `rotated` pages. Matched IoU: [0.5, 0.7), [0.7, 0.85), ≥ 0.85.
   - All intervals are descriptive. The report states how many it prints, applies no correction,
     and makes no "worse than" claim from two marginal intervals: a contrast the text states is a
     paired difference over the same page resample (`paired_ratios_diff_ci`). Slices are marginal
     and confounded (blur on a resized crop goes with line height; condition with document); the
     text says so.

4. **Recognition errors (GW-REC-040).**
   - **Consistency first:** each set's CER from the hand-off equals results.json's GW-REC-039 value,
     and GDC's per-condition crop CER its `gdc_conditions` row; otherwise the run stops.
   - **Edit operations:** Levenshtein edit operations (`rapidfuzz.distance.Levenshtein.editops`, the
     CER's own distance) on the normalised pair; substitutions, insertions and deletions per set,
     their sum asserted equal to each line's char_errors. The alignment is one of several optimal
     ones, so a confusion inside a run of repeated characters is arbitrary; the report says so.
   - **Character confusions:** the 30 most frequent substitutions (ref → pred), insertions and
     deletions per set, each with its count, its rate per occurrence of the reference character,
     and whether it disappears on the normalised track; a class matrix (lower, upper, digit,
     punctuation, space, other, out of charset) as a PNG for GDC and HierText test; space errors
     (merged and split words) on their own.
   - **Word errors:** whitespace tokens aligned the same way; the 30 most frequent wrong
     (ref word → pred word) pairs, and the share of word errors one character away.
   - **Slices:** line height, blur, contrast, line length, confidence, rotation (bins above);
     content: digit, punctuation, all upper case, a word of 10 or more letters, out of charset,
     each against its complement; on GDC, capture condition, document group (from
     `data/manifests/gdc/selection.yaml`, the stand-in for fonts), and the condition × group counts
     with CER by group inside `scan_app` and by condition inside `paper` (the only group in every
     condition). Which slices exist per set: IIIT5K has no height, rotation or detection fields;
     GDC-scan gets the numeric slices only.
   - **Categories** (RQ5's "which failure modes"): every wrong line gets one primary category by a
     fixed precedence, and the shares of wrong lines and of character errors sum to 100%:
     out of charset → rotation ≥ 5° → line height in the lowest quartile → blur in the lowest
     quartile → contrast in the lowest quartile → digit → punctuation → long word → other.
   - **Annotation noise (HierText):** a line counts as a likely label fragment when its normalised
     reference has at most 2 characters and the prediction is at least 3 × ref + 3 characters long
     (Phase 6 left this count open). Counted on GW-REC-038 and on HierText test, with their share of
     insertions and of character errors; the HierText confusion tables are shown with and without
     those lines.
   - **Baselines beside GLYPHARIS** (GDC, and HierText test for its five systems): substitution,
     insertion and deletion rates and class-matrix rates per reference character, every system
     scored with the GLYPHARIS charset classes; numbers only, no "own" or "shared" label.
   - **Validation counterpart:** the same tables on GW-REC-038 where its fields allow (no detection
     fields, no condition); ADR-0002's revisit trigger answered there: the share of character errors
     by line-length bucket (`baseline_diagnostics.py` BUCKETS), test beside it.
   - **Examples and difficult cases:**
     - the 20 most frequent GDC confusion types, up to 3 crops each, drawn with
       `np.random.default_rng([20261115, k])` over the sorted keys of the lines holding type k;
     - 40 error crops drawn with seed 20261115 from the sorted error keys, and 20 of the hand-off's
       200 correct crops (a draw from its draw), seed 20261115;
     - per set, the 20 worst lines (char_errors descending, key ascending) and the 10 worst pages
       (bag errors descending, key ascending).
     Galleries go to `data/processed/phase15/galleries/` (git-ignored; GDC capture images stay out
     of git). The committed report lists the drawn GDC lines with reference, prediction and source
     (title, licence, URL from `data/manifests/gdc_sources.json`; CC BY and CC BY-SA sources
     attributed). HierText tables carry `HIERTEXT_ATTRIBUTION` and its share-alike note. FUNSD and
     IIIT5K: character and class tables and counts only, no word pairs or line text. GDC-scan:
     numbers only.

5. **Pipeline errors (GW-E2E-010).**
   - **Consistency first:** each page's bag errors and reference characters from `pages_` match the
     e2e_pairs sums used below; the set totals equal results.json.
   - **Page error budget** (bag of lines, both tracks, per set and per GDC condition):
     (a) characters of ground-truth lines no prediction matched, (b) errors on matched lines, and
     (c) characters of spurious predictions, computed as `bag_errors − (a) − (b)`; asserted (c) ≥ 0
     on every page and (c) = 0 on every page whose `spurious` list is empty. Shares with page CIs.
   - **Unmatched lines** by `best_iou`: 0, (0, 0.3), [0.3, 0.5) ("poorly localised"), and ≥ 0.5
     ("its best box went to another line"); the four sum to (a) per page (asserted). Merges and
     splits cannot be told apart from IoU alone; GW-DET-001's validation categories (Phase 9:
     merged 20.0% and split 3.5% of misses) are cited for them.
   - **Cropping error:** matched-line CER by matched IoU bin. On GDC, after asserting per line that
     the pack label equals the page line's text, the matched line's end-to-end errors minus the
     oracle's errors on the same line, signed and never clipped, with the counts of lines where it
     is negative, zero and positive.
   - **Detection slices:** recall at IoU 0.5 by line height and rotation (pack lines, per page set),
     and by GDC condition (document clusters); matched-IoU distribution and the share of matches at
     0.7 or more per set (FUNSD's localisation, F22).
   - **Reading order:** per page, order-aware minus bag-of-lines errors for the pipeline, the oracle
     and each baseline on GDC, GDC-scan and OmniDocBench (descriptive; not an exact decomposition);
     on GDC, the share of consecutive matched ground-truth lines (in reference order) whose
     predictions come out in the opposite order. For F26: on OmniDocBench, the share of consecutive
     output lines of each PaddleOCR system whose top edge does not go up, which a top-to-bottom sort
     makes 1 by construction (polygons only, no reference).
   - **H4, the oracle share** on GDC, strict bag of lines: oracle errors / pipeline errors over a set
     of pages (the reference characters are the same for both, asserted per page). Per condition,
     for H1′'s clean slice (`scan_app` + `photo_flat`, 53 pages, 21 documents), for the other five
     conditions (53 pages, 18 documents) and those without `rotated` (46 pages, 18 documents); page
     and document-cluster CIs, the number of clusters printed, and the number of pages that make up
     80% of the pipeline's errors in each slice. Wording: "from ground-truth boxes the recognizer
     makes X% as many errors as the whole pipeline"; 1 − share is "errors the oracle does not
     reproduce" (missed and poorly boxed lines, spurious predictions, and crop shape), not a pure
     detector share.
     **H4 has no registered operational rule, and the published per-condition numbers already
     decide it** (oracle CER below half the pipeline's everywhere, clean slice included). So no new
     version of H4 is written: H4 is closed as "not supported on GDC (post hoc)", with these
     numbers as its record. GDC-scan: the same numbers, information only (ADR-0013).

6. **GW-REC-041** (validation only, a model run). The frozen recognizer, built from the freeze
   record in strict fp32 as in Phase 14, reads GlyphSynth v1 validation's lines; slices by font
   family and category (serif, sans, mono, from `data/manifests/glyphsynth_fonts.json`), held-out
   fonts, italic, inverted, neighbour fragments (a cropping stand-in), and capture steps (ink spread
   for touching characters, binarisation and low resolution for broken ones, paper texture, noise and
   salt-and-pepper for background noise). Line bootstrap. The paper also cites the frozen
   recognizer's GLYPH-C results (`reports/robustness/GW-REC-034.json`) for blur, noise, contrast,
   JPEG, perspective and in-box rotation. Both are synthetic or derived from validation, and say so.

7. **Descoped, with the reason:**
   - GW-DET-007's small and rotated lines (Phase 13): it needs a detector pass, and GW-DET-007 was
     not adopted, so the answer cannot change v1.
   - A v1 detector-against-recognizer split on HierText: GW-E2E-002's oracle cut and precision
     differ from the frozen pipeline's, so a new validation oracle would not pair cleanly with
     GW-E2E-006. GDC's GW-E2E-008 is v1's oracle; F10 stays the HierText reference.
   - Touching or broken characters, background noise and fonts on the real test sets: no labels
     exist; GW-REC-041 and GLYPH-C stand in.

8. **Rules fixed in advance.**
   - Every number in a tracked file comes from the Phase 15 scripts' output, with its experiment ID:
     `reports/error_cases/phase15/` (results.json, Markdown, PNG plots) and
     `reports/confusion_analysis/GW-REC-040/` (character and word tables).
   - **GDC-scan:** numbers only. The text, word, confusion and example builders refuse a
     numbers-only set (a test asserts the refusal), and a test plants a marker in every made-up
     GDC-scan input and checks no output contains it.
   - **Nothing flows back** into v1 (EXPERIMENT_PROTOCOL §9); a later model that uses these findings
     is a new version, evaluated again and disclosed.
   - **Phase 22:** nothing is written under `data/processed/phase14/` or the baseline folders,
     `phase14_outputs.json` is untouched, and `glypharis/eval/stats.py` stays byte-identical (the
     pinned scorer imports it). Per-line joins and galleries go to `data/processed/phase15/`.
   - **No cherry-picking:** examples come only from the draws above.

9. **Code needed** (tests on made-up data): `glypharis/eval/errors.py` (edit operations, confusion
   and class tables, word alignment, slice binning, the error budget, the order-inversion rate,
   numbers-only refusal); `scripts/phase15_errors.py` (input checks and the read record, every table
   above, plots, the paper tables); a GW-REC-041 runner on validation; the GDC-scan tests.

10. **Records updated with this plan:** experiment entries GW-REC-040, GW-E2E-010 and GW-REC-041;
   H4's status note (closed post hoc once GW-E2E-010 runs); a protocol change row (inputs and how
   reads are recorded; the bounded per-line text; H4 post hoc; validation-derived bin edges);
   `research/paper/error_analysis.md`'s outline, with Phase 9's validation categories moved in;
   RISK_REGISTER R-22 (curved lines) after the run; a research-log entry.

11. **Amendment (2026-11-17, from three reviews of the Phase 15 code; written before any Phase 15
    code read a per-line test file).** Where it differs, it replaces the text above.
    - **OmniDocBench** gets the reading-order rows only: its reference unit is a layout block, so the
      page budget, unmatched lines, cropping error and detection rows are n/a there, as in Phase 14.
    - **Intervals** are computed and printed on the primary track (strict; IIIT5K alphanumeric);
      the other tracks show values only. The report counts the intervals it prints, as it prints
      them.
    - **Validation** (GW-REC-038) resamples pages, the page taken from the line id
      (`<image>/p<i>/l<j>`).
    - **Line length** is binned on the strict-normalised reference length on every track.
    - **The one-character-away share** is over substituted word pairs (an inserted or deleted word
      has no partner).
    - **Example types** are the most frequent edit operations of any kind (substitution, insertion,
      deletion), ranked by count, then by operation and characters; type k is the k-th in that
      ranking, counted from 1, and seeds its own draw.
    - **Inside a GDC document group** (conditions within `paper`, groups within `scan_app`): counts
      and CER only, no CI. Every share column is a share of the whole set's character errors.
    - **GDC quotes:** lines from the two CC BY-SA 3.0 textbooks (`hefferon_linalg`, `mit_mcs`) are
      listed by id and numbers only, never quoted; every other quoted line carries its document key,
      source URL and licence (arXiv CC BY 4.0, whose abstract page names title and authors; US
      federal and Gutenberg works public domain). HierText-derived files (Markdown, JSON, PNG) carry
      `HIERTEXT_ATTRIBUTION` and the share-alike note.
    - **GlyphSynth's held-out-font slice** is dropped: every validation line is rendered with a
      validation font, so the slice has one value.
    - **GDC-scan gets no category table** (most rules read text).
    - **Writes:** the scripts refuse any output folder under `data/processed/phase14/`, the baseline
      folders or `data/manifests/processed/`, and the gallery folder must be git-ignored.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 15): `reports/error_cases/phase15/`,
`reports/confusion_analysis/GW-REC-040/` and `research/paper/error_analysis.md` drafted from
generated numbers: character and word errors, slices and categories, the detector against
recognizer proportion, representative examples; findings rows for RQ5; CI green.

**Known limitations, stated in advance**
- No merge, split or candidate-level detector analysis on test.
- GDC is 106 pages from 21 documents and GDC-scan one document; FUNSD 50 forms; the document
  cluster intervals with 18 to 21 clusters cover less than nominal.
- The blur attribute is measured on 64-px crops, so it mixes source resolution with optical blur.
- GlyphSynth and GLYPH-C are synthetic or derived from validation.
- H4 is read post hoc.

**Results (2026-11-17)** (GW-REC-040, GW-E2E-010, GW-REC-041; `reports/error_cases/phase15/` and
`reports/confusion_analysis/GW-REC-040/`; strict track unless stated; 95% percentile bootstrap,
1,000 resamples, seed 20261010. Every interval is descriptive: no correction, and no slice is
called worse than another from two marginal intervals.)

*Where page error comes from* (GW-E2E-010, `pipeline.md`): the frozen pipeline's bag-of-lines
errors (GW-E2E-007) split into characters of ground-truth lines no prediction matched, errors on
matched lines, and characters of spurious predictions. Shares of the set's errors, page CI.

| Set | Pages | Bag errors | Missed lines % | Matched lines % | Spurious % |
|---|---|---|---|---|---|
| HierText test | 1,634 | 203,171 | 49.45 [47.47, 51.56] | 25.79 [23.51, 28.00] | 24.77 [23.24, 26.38] |
| GDC | 106 | 71,886 | 45.32 [36.14, 52.80] | 27.55 [15.80, 42.07] | 27.13 [20.89, 33.57] |
| GDC-scan | 28 | 1,490 | 34.36 [14.87, 42.85] | 32.28 [16.36, 71.27] | 33.36 [14.01, 40.79] |
| FUNSD | 50 | 16,074 | 40.22 [36.27, 42.88] | 19.39 [14.64, 27.08] | 40.39 [36.32, 43.00] |

- GDC with source documents as the bootstrap unit: missed 45.32 [33.27, 53.44], matched 27.55
  [14.61, 45.81], spurious 27.13 [19.61, 32.97]. OmniDocBench has no budget: its reference unit is
  a layout block (amendment, item 11).
- Per GDC condition (document CI) the missed share runs from 40.83% (`low_light`) to 54.95%
  (`curved`, `photo_angled`), except on `rotated` pages, where matched lines carry 78.16%
  [45.80, 97.26] of the errors: the recognizer gets sideways crops (F24).
- Missed lines by best IoU: on HierText test 10,996 lines (45,721 characters) got no box at all;
  on GDC most missed characters are in lines a box barely touched (17,721 characters, 521 lines)
  or poorly localised (12,240, 284 lines), and only 2,508 in the 525 lines with no box.
- Box quality on matched lines: matched-line CER is 29.24% at IoU [0.5, 0.7) and 3.63% at 0.85 or
  more on HierText test, 29.36% and 4.31% on GDC. On the same 7,041 matched GDC lines the pipeline
  makes 2.048 [0.753, 3.817] more errors per line than the oracle (1,160 lines more, 5,567 equal,
  314 fewer).
- Recall at IoU 0.5 on HierText test: 57.92% [54.64, 61.17] for lines under 9.35 px high, 49.53%
  [45.80, 53.19] for |θ| ≥ 15°. On GDC none of the 113 lines under 9.35 px is found, and 36.94%
  [10.52, 45.00] of those from 9.35 to 13.52 px. FUNSD's matches are loose: 47.11% [42.48, 51.39]
  at IoU 0.7 or more (HierText test 85.90%, GDC 91.14%).
- Phase 9's validation categories for GW-DET-001 stand in for merges and splits, which IoU alone
  cannot separate: merged 20.0% and split 3.5% of its missed lines.

*The oracle share (H4)* (GW-E2E-010, `oracle_share.md`): oracle errors (GW-E2E-008, ground-truth
boxes) over pipeline errors (GW-E2E-007), strict bag of lines. "From ground-truth boxes the
recognizer makes X% as many errors as the whole pipeline."

| GDC slice | Pages | Documents | Share % [page CI] | Share % [document CI] | Pages for 80% |
|---|---|---|---|---|---|
| all pages | 106 | 21 | 18.66 [11.55, 28.73] | 18.66 [11.51, 26.45] | 27 |
| clean (`scan_app` + `photo_flat`) | 53 | 21 | 32.70 [16.64, 51.91] | 32.70 [24.43, 44.01] | 11 |
| the other five | 53 | 18 | 10.29 [5.24, 18.03] | 10.29 [5.24, 17.73] | 16 |
| the other five without `rotated` | 46 | 18 | 14.86 [8.46, 23.93] | 14.86 [8.22, 26.43] | 14 |
| `scan_app` | 37 | 19 | 45.14 [25.40, 58.41] | 45.14 [24.95, 58.56] | 10 |
| `photo_flat` | 16 | 13 | 14.75 [5.80, 43.65] | 14.75 [5.53, 43.95] | 3 |
| `photo_angled` | 16 | 10 | 12.81 [9.84, 22.51] | 12.81 [11.29, 19.44] | 4 |
| `distant` | 11 | 10 | 22.09 [8.24, 46.34] | 22.09 [8.57, 50.93] | 5 |
| `low_light` | 11 | 8 | 16.39 [8.99, 40.69] | 16.39 [7.59, 45.03] | 4 |
| `curved` | 8 | 8 | 12.47 [0.71, 35.21] | 12.47 [0.71, 35.21] | 5 |
| `rotated` | 7 | 6 | 1.03 [0.34, 2.93] | 1.03 [0.36, 2.94] | 4 |
| GDC-scan, all pages (information) | 28 | - | 22.08 [12.65, 47.58] | - | 7 |

- **H4: not supported on GDC (post hoc).** The share's point estimate is below half in every
  slice; the clean slice's page interval reaches 51.91% and `scan_app`'s 58.41%. 1 − share is the
  errors the oracle does not reproduce (missed and poorly boxed lines, spurious predictions, crop
  shape), not a pure detector share.

*Recognition* (GW-REC-040, `recognition.md`; ground-truth crops through the frozen recognizer,
GW-REC-039 on test, GW-REC-038 on validation; page CI, IIIT5K line CI)

| Set | Lines | CER % [CI] | Sub % | Ins % | Del % | Merged words | Split words | WER % |
|---|---|---|---|---|---|---|---|---|
| HierText test | 67,574 | 6.03 [5.60, 6.48] | 3.27 | 0.89 | 1.88 | 2,512 | 1,110 | 19.26 |
| HierText validation | 74,500 | 6.51 [5.98, 7.12] | 3.41 | 0.97 | 2.12 | 2,860 | 1,319 | 19.78 |
| GDC | 8,372 | 4.17 [2.48, 6.22] | 0.86 | 0.12 | 3.19 | 4,941 | 115 | 14.56 |
| GDC-scan (numbers only) | 776 | 1.02 [0.79, 1.30] | n/a | n/a | n/a | n/a | n/a | n/a |
| FUNSD | 2,647 | 7.03 [5.59, 8.61] | 2.12 | 1.45 | 3.46 | 640 | 83 | 25.88 |
| IIIT5K (alphanumeric) | 3,000 | 2.37 [2.06, 2.70] | 1.39 | 0.21 | 0.77 | 0 | 0 | 8.53 |

The page CIs are wider than Phase 14's line CIs for the same CERs (GDC 4.17% [3.72, 4.67] by
lines). The top primary categories, as shares of the set's character errors (one category per
wrong line, by the fixed precedence of plan item 4, so "digit" is any wrong line with a digit that
meets no earlier rule):

| Set | First | Second | Third |
|---|---|---|---|
| HierText test | rotation ≥ 5°, 34.92 | height, lowest quartile, 21.30 | other, 9.75 |
| HierText validation | rotation ≥ 5°, 37.69 | height, lowest quartile, 18.31 | other, 11.59 |
| GDC | digit, 49.21 | contrast, lowest quartile, 20.41 | out of charset, 13.62 |
| FUNSD | digit, 43.21 | punctuation, 31.32 | long word, 11.46 |
| IIIT5K | other, 87.85 | long word, 7.73 | contrast, lowest quartile, 2.76 |

- **GDC:** deletions are 3.19 of the 4.17% CER. Spaces are deleted 4,941 times (10.165% of
  reference spaces: merged words) and full stops 4,153 times (44.176% of reference full stops), on
  583 and 391 lines. All of the 20 worst GDC lines are table-of-contents lines whose leader dots
  are dropped, from `census_income` and `hilbert_geometry`; lines of 70 characters or more carry
  61.80% of GDC's errors (CER 6.37%). Typographic quotes and dashes become ASCII (’ to ' 198 times,
  “ and ” to " 111 and 109 times, — and – to - 66 and 46 times); those pairs are gone on the
  normalised track (CER 3.94%).
- **GDC document groups** (no CI): book 15.98% (37.46% of the errors), report 7.56%, textbook 5.05%,
  form 2.96%, paper 1.02%. By condition (document CI) `scan_app` holds 53.40% of the errors at 6.55%
  [2.42, 11.30]; inside the paper group (paper and book are both in every condition; the report
  breaks out paper), the condition CERs run from 0.26 to 1.42%. Conditions go with documents (the
  condition × group table), so no per-condition contrast is drawn.
- **Beside the baselines on GDC** (same lines, no CI): PP-OCRv5 mobile and the three PP-OCRv6
  systems read 86.90 to 89.95% of reference punctuation as itself, GLYPHARIS 73.02%; spaces 87.11
  to 88.54% against 89.61%; deletion rates 2.29 to 2.55% against 3.19%.
- **HierText test:** CER 4.46% for |θ| < 2° and 10.82% for |θ| ≥ 15°; 11.01% for lines under
  9.35 px (30.24% of the errors). The lowest confidence quartile (below 0.626682) holds 78.51% of
  the errors at 24.01% CER, the top quartile 0.89% at 0.39%. Out-of-charset lines: 282, with 2.65%
  of the errors. Likely label fragments: 15 lines, 145 errors (0.30%) and 139 insertions (1.94% of
  insertions); without them CER 6.02%. On GDC the two lowest confidence quartiles hold 22.79% and
  53.64% of the errors.
- **FUNSD:** deletions are 3.46 of the 7.03%; 9.56% of reference digits and 10.52% of reference
  spaces are deleted; the 170 lines of 21.485 px or more read at 27.74% [17.30, 38.51] and carry
  37.62% of the errors.
- **ADR-0002's trigger** (error share by reference length): on validation, lines of 0 to 29
  characters carry 81.51% of the errors and lines of 70 or more 3.59% (HierText test 80.24% and
  4.74%). Long lines do not dominate. The 20 worst GDC lines are dot leaders, but no per-line
  attribute labels leaders, so their share of GDC's 70+ bucket (61.80%) is not measured.

*GlyphSynth validation* (GW-REC-041, `glyphsynth.md`; synthetic, line CI)

| Slice | Lines | CER % [CI] | Share of errors % |
|---|---|---|---|
| all | 10,000 | 0.71 [0.63, 0.79] | 100.00 |
| touching (ink spread) | 1,543 | 1.41 [1.17, 1.72] | 31.11 |
| no ink spread | 8,457 | 0.58 [0.50, 0.67] | 68.89 |
| underline | 283 | 1.37 [1.05, 1.76] | 5.16 |
| binarised | 307 | 1.18 [0.81, 1.67] | 4.92 |
| inverted | 318 | 1.06 [0.62, 1.72] | 5.02 |
| italic | 1,203 | 0.86 [0.67, 1.12] | 14.59 |
| broken (binarisation, low resolution) | 2,226 | 0.76 [0.63, 0.90] | 23.02 |
| background noise (texture, noise, salt and pepper) | 6,012 | 0.76 [0.65, 0.89] | 64.32 |
| neighbour fragments | 3,517 | 0.66 [0.56, 0.78] | 32.69 |

- Font families run from 0.38% (`worksans`) to 1.48% [1.09, 1.93] (`cormorantgaramond`); serif
  0.83%, sans 0.60%, mono 0.62%. GLYPH-C (`reports/robustness/GW-REC-034.json`, CERs as the
  fractions it stores): clean 0.0193; in-box rotation 0.0238 at level 1 to 0.2357 at level 5; every
  other corruption 0.0572 or less (low resolution, level 5); mCE 3.54%
  (`reports/ablation/phase13_final.md`).

*Reading order* (GW-E2E-010): on GDC 1,010 of 6,935 consecutive matched ground-truth lines come out
reversed, 14.56% [9.93, 20.36]. Order-aware minus bag-of-lines errors per page on GDC: GLYPHARIS
+125.509 [−76.729, 333.847], the four PaddleOCR systems +573.019 to +721.915. The OmniDocBench
bag-of-lines CERs in the same table (about 179 to 181%) are not meaningful: its reference is blocks,
not lines (Phase 14 marks bag of lines n/a there). For F26: 76.66 to 77.74% of the four PaddleOCR
systems' consecutive output lines on OmniDocBench have a top edge that does not rise. PaddleOCR's
own sort, read from its installed code (`SortQuadBoxes` in
`paddlex/inference/pipelines/components/common/sort_boxes.py`, used by the OCR pipeline), orders
boxes by the top-left corner's y and then swaps neighbours whose y differs by less than 10 px so
that they run left to right, so a following line can sit up to 10 px higher and the measure would
not be 1 under that sort. It cannot tell the sort apart from other orders, so it neither supports
nor refutes F26's cause, which stays unmeasured (F26 revised, F33).

**Run notes**
- Code `ba44a84`, reports `6cb1b9c`; results.json records commit `ba44a848f9eb`, not dirty. Phase
  15 evaluates nothing, so its reads are recorded in results.json (`inputs`), not in the test
  access log.
- Every input was checked before a row was read: 34 of the 38 files in the README's table against
  their manifests or metadata (the other 4 hashed as committed), the Phase 5 files with
  `verify_phase5` (93 files), the 2,488 GDC crops against `gdc_crops/index.jsonl`, and the 120
  drawn crops again before they were copied into the gallery.
- GW-REC-041 read 10,000 GlyphSynth validation lines in 79.3 s on the GPU (strict fp32; wall time
  from its run metadata, `data/processed/phase15/GW-REC-041/glyphsynth_val.meta.json`). The pack's
  hash was checked against its metadata and the GlyphSynth manifest before the run, and the
  output's hash and count against the run's `.meta.json` before the analysis read it.
- 583 intervals printed; 49 CERs from the hand-off equal results.json's, and the hand-off's pair
  sums agree with the page files on all 2,573 pages (the five page sets, OmniDocBench included).

**Tests:** the CPU suite passes in a clone without tags (1,011 passed, 4 skipped, at `6cb1b9c`);
new tests cover the error module, the analysis script (including the GDC-scan marker, the write
guards, the hash and consistency refusals and atomic output) and the GlyphSynth runner.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 15)

| Criterion | Result | Evidence |
|---|---|---|
| Character confusion matrix; most common character and word errors | pass | class matrices (PNG for GDC and HierText test) in `recognition.md`; `reports/confusion_analysis/GW-REC-040/` |
| Slices: blur, rotation, small font, low contrast, punctuation, numbers, long words, cropping error | pass, on the real sets | `recognition.md`, `pipeline.md` |
| Slices: touching or broken characters, unusual fonts, background noise | pass, on GlyphSynth validation and GLYPH-C only: the test sets have no labels for them (plan item 7) | `glyphsynth.md`, `reports/robustness/GW-REC-034.json` |
| Detector against recognizer proportion (oracle analysis) | pass: the page budget on every page set but OmniDocBench (layout blocks; reading order only), the oracle share on GDC and GDC-scan; H4 not supported on GDC (post hoc) | `pipeline.md`, `oracle_share.md`, `research/hypotheses.md` |
| Representative examples | pass: seeded draws only | `examples.md` (crops in the git-ignored gallery) |
| `reports/error_cases/` and `research/paper/error_analysis.md` drafted from generated numbers | pass | `reports/error_cases/phase15/`, `research/paper/error_analysis.md` |
| Findings rows for RQ5 | pass | `research/findings.md` F27 to F32; F26 revised by F33 |
| CI green | pass | runs 44065877523 (branch), 44065813870 (pull request), 44068353082 (`main`) |

**Known issues**
- GDC is 106 pages from 21 documents, GDC-scan one document and FUNSD 50 forms. The
  document-cluster intervals with 6 to 21 clusters (the per-condition ones fewest) cover less than
  nominal, and GDC's conditions go with its documents, so per-condition numbers are not contrasts.
- Merges and splits cannot be told apart from IoU on test; Phase 9's validation categories stand
  in for them.
- The oracle share is not a pure detector share.
- `pipeline.md`'s reading-order table prints bag-of-lines CERs for OmniDocBench (about 179 to
  181%). They are not meaningful, because the reference unit is a layout block; only the
  order-aware columns and the F26 table are read. The report stays as generated.
- The 20 worst GDC lines are dot leaders from `census_income` and `hilbert_geometry`, and the
  seeded draws show leaders in other documents too; no per-line attribute labels them, so their
  share of the 70+ bucket and of the full-stop deletions is not measured.
- The registered F26 measure did not account for PaddleOCR's 10 px row tolerance: the plan expected
  a top-to-bottom sort to make it 1 by construction, but PaddleOCR's sort lets a following line sit
  up to 10 px higher. So the measure does not decide F26's cause, and what causes PaddleOCR's order
  error on OmniDocBench is still open.
- The blur attribute mixes source resolution with optical blur; GlyphSynth and GLYPH-C are
  synthetic or derived from validation; H4 is read post hoc.

**Git:** branch `phase/15-error-analysis`, merged through pull request #20. Merge commit `ed372db`, tag
`phase-15` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 44068353082).

---

## Phase 16 — Efficiency optimisation

**Status:** COMPLETE (2026-11-18).
Opened 2026-11-17 on `phase/16-efficiency`, branched from `main` after Phase 15 (merge `ed372db`).
Validation data only.

**Objective:** the frozen v1 system's cost on this laptop (parameters, size, CPU and GPU latency,
throughput, memory, page time, preprocessing) in PyTorch and in ONNX Runtime, and the accuracy and
latency of variants that would be new versions (INT8, other input sizes, other model sizes, a fused
attention): an efficiency table and an accuracy/latency Pareto plot (RQ7), with H7, H6's latency
half and H1′'s ONNX latency judged by rules fixed here.

**Plan** (registered 2026-11-17, before any Phase 16 measurement. Already known: GW-EFF-001 to
005; F9, F12, F13 (the TorchScript-exported SVTR recognizer about 4.5 times slower than PyTorch on
the CPU; ONNX Runtime's arena growing with input shapes; the exported detector's "twice as fast"
came from an arena-on side test, and arena-off it ran at about PyTorch's speed, GW-EFF-004's entry);
H1′'s PyTorch latency verdict (GW-EFF-005); GW-DET-005's detector F1 by input size on validation
half A; the Phase 7 line timings, PyTorch at 16 threads: SVTR 13.5 ms against CRNN 10.7 ms at
height 32 (`reports/training/phase07/cpu_latency.md`). The code under `glypharis/pipeline`,
`glypharis/models`, `glypharis/export`, `glypharis/data` and `configs` is unchanged since
`phase14-eval`.)

1. **v1 and its variants.**
   - **v1 is the evaluated build.** The frozen weights, every recorded setting of
     `configs/final/glypharis_v1.yaml`, and the model, pipeline and export code as it is at
     `phase14-eval`. The Phase 16 scripts assert that those code paths still equal the tag's before
     they build v1; no Phase 16 change touches them.
   - **The same model in another runtime:** only the plain fp32 ONNX export of v1 (TorchScript
     exporter, as in Phase 17), and only if it passes Phase 17's registered tolerances against
     **reference outputs written once from the PyTorch v1 build on the CPU** (detector probability
     maps on the 110 profile pages, recognizer logits on the 5,000 seeded validation lines, page
     predictions on the 400 Phase 5 pages; their SHA-256 committed): map max abs difference ≤ 1e-4;
     logits ≤ 1e-3; text identical except near ties (top-2 gap < 1e-3) on at most 0.1% of lines;
     page CER within 0.05 points (Phase 14 item 6). This is the bundle the Phase 17 re-run cites.
   - **Variants** (new versions; opt-in code paths outside the v1 modules; measured on validation
     as Pareto points; none replaces v1; one that ships is evaluated in Phase 22, as Phase 14 item
     6 fixes for INT8): dynamic INT8; detector long side 1024, 1280, 1792, 2048; the ONNX detector
     with the PyTorch recognizer; a fused attention export; the smaller detector GW-DET-003 and the
     recognizers GW-REC-003, 006, 020, 022 (other recipes and budgets, not a clean size ladder).
     Choosing a default backend or shipping a variant is the owner's decision (ADR-0011); Phase 16
     changes no default.

2. **Experiments** (validation only).

   | ID | What |
   |---|---|
   | GW-EFF-006 | the reference outputs from PyTorch v1 on the CPU; the v1 fp32 ONNX bundle built from the freeze record; Phase 17's four checks against the references |
   | GW-EFF-007 | the cost table of v1 (PyTorch and fp32 ONNX): parameters; file size per format; CPU line latency (batch 1) and page latency (one page at a time, lines batched as the pipeline does), mean and p95, peak RAM, at 10 threads; the same at the default threads for fp32 ONNX only; GPU (PyTorch only; ONNX Runtime here is CPU-only) line latency at batch 1 and line throughput at batch 32, page latency, peak VRAM; page time split into decode, detector preprocessing, detector network, post-processing, cropping, recognizer batch build, recognizer network, decoding and reading order, summing to the page time; page time against lines per page (least-squares slope in ms per line, page-bootstrap CI) |
   | GW-EFF-008 | dynamic INT8 of the v1 fp32 ONNX recognizer (H7′) |
   | GW-EFF-009 | Pareto points (item 1's variants): page CER and E2E F1 on the 400 Phase 5 pages on the CPU, page latency and peak RAM on the 110 profile pages, fp32 ONNX at 10 threads unless the variant says otherwise; recognizer variants by line CER on the 5,000 lines and line latency |
   | GW-EFF-010 | H1′'s ONNX latency: a fresh GW-BASE-002 profile and the GW-EFF-006 bundle (by its hash) in one sitting, by Phase 14's rule |
   | GW-EFF-011 | H6′'s latency: CRNN GW-REC-003 against SVTR GW-REC-020 (H6's matched-budget pair, height 32), fp32 ONNX, per line |

3. **Measurement rules.**
   - Mains power, Balanced plan, the power mode recorded, no other user program; `run_conditions`
     and the GPU's clocks and temperature (nvidia-smi) at the start and end of every profile, CPU
     runs included.
   - 10 warm-up and 100 timed items (protocol §3): lines from `rec_samples` (hiertext_val), pages
     from the Phase 5 profile pages (the first 110 of the 400-page sample). Accuracy samples: the
     5,000 lines seeded 20261028 from `processed/hiertext/lines_validation` (the export check's) and
     the 400 Phase 5 pages; accuracy for CPU builds is measured on the CPU.
   - Threads: torch intra and inter 10 with the OpenMP and MKL variables set before the import; each
     ONNX Runtime session intra-op 10, sequential execution (inter-op unused), spinning at the
     default; each session's options and the torch pools written into the profile and asserted.
   - One process per run. When k arms are compared, they run in the order 1 … k, then k … 1, in
     one sitting; each item's time is the mean of its two runs, and both runs' times are kept.
     GW-EFF-010 follows Phase 14's order instead (baseline, then GLYPHARIS, within 2 hours).
   - A sitting is void only if a recorded condition fails (mains, plan, threads, input sizes, a
     missing time); a void sitting goes to `research/failures.md` and is repeated, and the first
     sitting that is not void is the one of record.
   - p95 from 100 items is reported as a point value, no CI.
   - **Confirmatory tests, exactly four:** H7′ latency, H7′ accuracy, H6′, H1′ ONNX. Every other
     interval is descriptive and carries no verdict word; the report counts its intervals.
   - **Single model:** the comparisons are between fixed builds of one frozen model; the seed rule
     of protocol §4 does not apply (Phase 14's exception, extended by a change row).

4. **H7′** (new dated version; H7 superseded and kept).
   - Arms: the GW-EFF-006 fp32 ONNX recognizer and its dynamic INT8 version, made with ONNX
     Runtime's `quant_pre_process` (defaults) then `quantize_dynamic(op_types_to_quantize=
     ["MatMul"], per_channel=False, reduce_range=False, weight_type=QInt8,
     extra_options={"MatMulConstBOnly": True})`. Known in advance: the attention products have no
     constant input, so they stay fp32. No other configuration is tested.
   - *Latency:* `rec_samples`, 10 warm-up and 100 timed lines, batch 1, 10 threads, one sitting in
     the order fp32, INT8, INT8, fp32: met if the CI of mean(t_INT8 − 0.75 × t_fp32) per line lies
     below 0.
   - *Accuracy:* the 5,000 lines, strict track, greedy decoding:
     `paired_ratio_diff_ci(errors_INT8 − 0.005 × ref_chars, errors_fp32, ref_chars, "lines")`;
     met if the CI lies below 0.
   - Met, not met or not shown per half; H7′ holds only if both are met.

5. **H6′, latency half** (new dated version; H6 superseded, its CER half stays exploratory (Phase
   13), so H6 as a whole can at most be "inconclusive"): GW-REC-020 and GW-REC-003, each exported
   to fp32 ONNX and each passing the logit and text checks against its own PyTorch outputs on the
   5,000 lines first; `rec_samples` 100 timed lines, batch 1, 10 threads, one sitting in the order
   CRNN, SVTR, SVTR, CRNN: met if the CI of mean(t_SVTR − 1.5 × t_CRNN) lies below 0. PyTorch at 10
   threads is information (Phase 7's table re-confirmed).

6. **H1′ ONNX:** Phase 14's rule (items 5 and 10) with the GW-EFF-006 bundle, named by its SHA-256
   before the sitting; no build is selected by timing. The verdict applies to that hash only; if the
   Phase 17 re-run ships another, it says the verdict does not carry over.
   - The fresh profile is a dated Phase 16 amendment to GW-BASE-002: `scripts/run_baselines.py`
     gains `--phase 16` for validation profiles only (branch-guarded; output
     `data/processed/baselines_phase16/GW-BASE-002/`; every settings field asserted equal to the
     Phase 14 profile's), run in `.venv-baselines`.
   - The sitting checks and the verdict live in a new Phase 16 script that imports Phase 14's
     helpers; `scripts/phase14_score.py` is not edited (Phase 22 re-scores with it).

7. **Code needed** (each with tests on made-up data or tiny models): an export path from the
   freeze record with the recorded provenance (`scripts/export_check.py --frozen`), the reference
   writer and the check against it; an ONNX profile path for the frozen build with `--threads`,
   `--experiment` and start and end times (`scripts/e2e_profile.py`); `--threads` and an ONNX
   backend for `scripts/rec_profile.py`; GPU timing with CUDA synchronisation and peak VRAM; the
   stage split; the variant builder (overrides one named setting, asserts the rest, labels the
   output "new version"); INT8 (a smoke test on a tiny SVTR); a tiny-SVTR ONNX export test at widths
   other than the trace width; the fused attention export (a test that it equals the reference
   attention with the padding mask); `--phase 16` in the baseline runner; the Phase 16 scorer
   (efficiency table, Pareto plot, the four verdicts) writing `reports/efficiency/phase16/` and
   `reports/benchmark_results/efficiency_comparison.md` (v1 and the Phase 5 baselines side by side,
   labelled "different sittings, no difference stated").

8. **Order of work and cost** (CPU hours are estimates from GW-EFF-001 to 005 and the export
   checks, not records): tooling; GW-EFF-006 (about 1.5 h); GW-EFF-007 (about 2 h, GPU minutes);
   GW-EFF-008 (about 1 h); GW-EFF-010 (about 0.5 h); GW-EFF-009 (about 6 to 8 h: 8 page variants
   × 400 pages); GW-EFF-011 (about 1 h). About 12 to 14 CPU-hours. The cut line, if time runs out:
   GW-EFF-009's recognizer and detector-size points beyond 1024 and 2048, then GW-EFF-011.

9. **Records:** experiment entries GW-EFF-006 to 011 and the GW-BASE-002 amendment; H7′ and H6′;
   change rows (the reference-based same-model check; Phase 16's baseline profile; the single-model
   exception; GPU conditions on every profile); ADR-0011's five targets reported for v1's builds,
   with an options table for the owner (not tightened; the owner's call); F13 and the
   `glypharis/export/onnx.py` docstring corrected from arena-off numbers; the compute ledger; the
   risk register if ADR-0011 is still missed.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 16): the efficiency table and the accuracy/latency
Pareto plot, measured on this laptop with documented conditions; INT8 measured with its accuracy
loss; the four verdicts; CI green.

**Known limitations, stated in advance**
- One laptop; latency on HierText validation pages (about 41 lines per page), not documents.
- Variants are new versions, measured on validation only; the recognizer and detector size points
  differ in recipe and budget, not only in size.
- Thermal drift within a sitting is limited by the mirrored order, not removed.
- Height sweeps need retraining and are not run; the width cap is left at 1536 (a scope choice).

**Results (2026-11-18)** (GW-EFF-006 to 011 and the GW-BASE-002 amendment;
`reports/efficiency/phase16/` and `reports/benchmark_results/efficiency_comparison.md`, written by
`scripts/phase16_score.py` at `1253dfe`. Validation data only, on this laptop; no test data was
read, so `research/test_access_log.md` gets no row. Means carry a 95% bootstrap interval over the
timed items, 1,000 resamples; p95 is a point value. 95 intervals printed: 4 confirmatory, 91
descriptive. Only the four confirmatory tests carry a verdict. Every point but v1 is a new version,
not v1, measured on validation only; none replaces v1, and Phase 16 changes no default.)

*The run.* One detached chain of 94 steps, 2026-11-17T13:44:59Z to 2026-11-18T05:36:52Z (15.7
CPU-hours and 0.1 GPU-hours in the compute ledger, against the plan's estimate of about 12 to 14
CPU-hours; see Compute below). Every profile recorded mains power, the Balanced plan and power mode
"best performance" at its start and end; every GLYPHARIS profile also recorded the thread settings
(asserted before and after) and the GPU's clocks and temperature (nvidia-smi), CPU runs included
(the GW-BASE-002 profile, run by the Phase 5 runner, records no GPU state). No sitting was void, and
the cut line was not needed: every variant ran.

- Mirrored orders, one sitting each, for the compared arms of GW-EFF-008, 009 and 011: the page
  points (20 runs, v1-torch to det-GW-DET-003 and back, 2026-11-18T00:59:56Z to 05:29:38Z); the
  line points (16 runs, 05:29:50Z to 05:34:57Z); H7′ (fp32, INT8, INT8, fp32, 2026-11-17T16:38:41Z
  to 16:39:58Z); H6′ (CRNN, SVTR, SVTR, CRNN, 2026-11-18T05:35:10Z to 05:36:07Z). GW-EFF-010
  followed Phase 14's order, one run each: GW-BASE-002 from 16:40:45Z to 16:46:30Z, then the bundle
  from 16:46:42Z to 17:04:02Z. GW-EFF-007's rows are one run each, in chain order (Deviations).
- Step 10 (GW-EFF-007's PyTorch GPU page profile) stopped at 2026-11-17T15:40:20Z after 110 s. The
  profiler's own check that the stages sum to the page time within 1% failed on a short page ("the
  stages sum to 75.922 ms, the call took 78.329 ms"): on CUDA every stage boundary adds a
  synchronisation, so a fixed 1% is too tight on short pages. `a8e8bb6` allows the larger of 1% and
  5 ms; the step was rerun from the start and finished (134 s). The failed attempt wrote no output,
  and no reported number depends on the tolerance; it only guards the stage split. In the profiles
  of record the largest relative difference is 3.03% on the GPU and at most 0.70% in the CPU page
  profiles (`stage_sum_max_rel_diff`). Logged in `research/failures.md`, with the path rewrite
  below.
- Every v1 build recorded its v1 code paths equal to `phase14-eval` (`6795186`) and a clean tree.
  The reports of steps 1 to 9 (the bundle, the PyTorch reference outputs, the reference check,
  GW-EFF-007's five line profiles and its PyTorch CPU page profile) record `6950a5a`; the 68 later
  ones record `a8e8bb6`. The only code difference between the two is that tolerance in
  `scripts/e2e_profile.py` and its test.
  results.json records the scorer commit `1253dfe`.

*The v1 fp32 ONNX bundle* (GW-EFF-006, `README.md`, `reference_check.json`): built by
`scripts/export_check.py --frozen` from the freeze record (record SHA-256 equal), manifest SHA-256
`875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`. Against the references written
once from PyTorch v1 on the CPU (manifest `4ff7f155…`, committed as `reference_manifest.json`),
Phase 17's four checks pass: map max abs difference 7.33e-06 on the 110 pages (limit 1e-4), logits
3.09e-04 on the 5,000 lines (limit 1e-3), no text differences and no near ties, page CER 1.49e-05
apart on the 400 pages (limit 5e-4, 0.05 points). This is the bundle the Phase 17 re-run cites.

*What v1 costs* (GW-EFF-007, `cost_table.md`; HierText validation, 10 warm-up and 100 timed items,
CPU at 10 threads unless stated). Parameters: detector 3,187,209, recognizer 7,928,835, total
11,116,044. Files: detector 12.44 MiB as the release .pt and 12.15 MiB as fp32 ONNX, recognizer
30.31 and 30.40 MiB (dynamic INT8, a new version: 9.82 MiB).

| Build | Line mean ms | Line p95 ms | Page mean ms | Page p95 ms | Page peak RAM MiB | ms per line on pages |
|---|---|---|---|---|---|---|
| PyTorch, CPU | 33.90 [29.92, 38.13] | 77.60 | 2,566 [2,327, 2,829] | 5,127 | 1,641.8 | 28.6 [19.2, 42.9] |
| fp32 ONNX, CPU | 92.25 [70.69, 116.28] | 304.39 | 8,574 [7,516, 9,766] | 19,011 | 6,529.6 | 72.6 [32.9, 142.7] |
| fp32 ONNX, CPU, default threads | 104.73 [79.87, 132.05] | 392.09 | 9,925 [8,744, 11,261] | 22,072 | 6,541.5 | 85.5 [41.6, 165.1] |
| PyTorch, GPU (lines at batch 1) | 12.22 [11.24, 13.33] | 19.01 | 1,049 [825, 1,300] | 3,288 | 2,450.0 | 4.8 [−1.4, 16.7] |

- The profile pages average 40.9 lines. "ms per line" is the least-squares slope of page time on
  lines per page, with a page-bootstrap interval. Peak RAM on lines: 825.7 MiB (PyTorch), 1,028.0
  (ONNX), 1,043.6 (ONNX, default threads), 1,180.6 (GPU, batch 1; 1,216.2 at batch 32).
- GPU: 160.1 lines/s at batch 32 (199.89 [187.57, 213.04] ms per 32-line item, batched as the
  pipeline batches) against 81.8 at batch 1. Peak VRAM by torch's `max_memory_allocated`: 77.3 MiB
  (batch 1), 282.2 (batch 32), 432.4 (pages). nvidia-smi gives no per-process figure under the
  Windows driver; the whole GPU used 529.0, 903.0 and 1,335.0 MiB at the end, display included.
- Stage split (mean ms per page; the stages sum to the page time): the recognizer network takes
  2,148.2 of 2,565.7 ms (84%) in PyTorch and 8,167.7 of 8,573.7 ms (95%) in ONNX, while the
  detector network takes about the same in both (296.4 and 297.5 ms). On the GPU the detector
  network is the largest stage (694.5 ms, 66%) and the recognizer network takes 261.0 ms (25%).
  Decoding the image file (13.3 to 17.5 ms) happens before the clock starts, as in GW-EFF-005.
- The same builds in later sittings: v1 fp32 ONNX lines 95.78 ms (GW-EFF-008) and 95.59 ms, pages
  9,739.4 ms; PyTorch lines 35.55 ms, pages 2,637.3 ms (GW-EFF-009).
- `efficiency_comparison.md` puts these rows beside the Phase 5 baseline profiles. They are
  different sittings, so no difference is stated.

*The four confirmatory tests* (`verdicts.md`; each is met if its interval lies below 0)

| Test | Quantity | Interval | Verdict |
|---|---|---|---|
| H7′ latency (GW-EFF-008) | mean(t_INT8 − 0.75 × t_fp32) per line | −10.00 [−12.79, −7.37] ms | met |
| H7′ accuracy (GW-EFF-008) | paired ratio difference, errors_INT8 − 0.005 × ref_chars against errors_fp32 | −0.512 [−0.541, −0.484] points | met |
| H6′ (GW-EFF-011) | mean(t_SVTR − 1.5 × t_CRNN) per line, fp32 ONNX | +16.78 [+12.02, +22.04] ms | not met |
| H1′ ONNX (GW-EFF-010) | per page, time(v1 fp32 ONNX) − time(GW-BASE-002) | +6,537.6 [+5,383.7, +7,806.3] ms | not met |

- **H7′ holds** (both halves met). fp32 95.78 ms and INT8 61.84 ms per line, two runs averaged (run
  means 96.61 and 94.95; 63.21 and 60.46); strict line CER 6.221% and 6.208% on the 5,000 lines.
  Recognizer file 30.40 to 9.82 MiB; 49 of the 67 MatMuls became MatMulInteger, and the 18 left
  include the attention products, which have no constant input and stay fp32.
- **H6′ not met.** Both exports pass their checks against their own PyTorch outputs on the 5,000
  lines (max logit difference 1.56e-04 for GW-REC-003 and 2.89e-04 for GW-REC-020, every text
  identical, no near ties, height 32). CRNN 10.16 ms and SVTR 32.01 ms per line. PyTorch at 10
  threads (information): GW-REC-003 13.00 ms, GW-REC-020 15.70 ms.
- **H1′ ONNX not met.** GW-BASE-002 3,086.0 ms per page, v1 fp32 ONNX 9,623.6 ms, same pages in the
  same order. The verdict applies to bundle `875357a6…` only. H1′ itself still holds from Phase 14,
  whose latency half is the PyTorch build (GW-EFF-005).

*Accuracy against latency* (GW-EFF-009, `pareto.md`, `pareto.png`; fp32 ONNX at 10 threads unless
the point says otherwise, each item the mean of its two runs; differences are the point minus v1's
fp32 ONNX build on the same items, descriptive)

| Point | Page CER % | E2E F1 % | Page mean ms | Page p95 ms | Peak RAM MiB | Δ CER (points) | Δ ms |
|---|---|---|---|---|---|---|---|
| v1-torch (v1) | 24.32 [20.87, 28.15] | 60.50 | 2,637.3 | 5,327.7 | 1,665.0 | −0.00 [−0.00, +0.00] | −7,102.1 [−8,195.7, −6,115.5] |
| v1-onnx (v1) | 24.32 [20.87, 28.15] | 60.50 | 9,739.4 | 22,474.1 | 6,525.5 | n/a | n/a |
| det-long-side-1024 | 31.62 [27.74, 35.79] | 55.95 | 8,491.3 | 20,331.8 | 5,983.6 | +7.29 [+5.55, +9.51] | −1,248.1 [−1,761.3, −734.3] |
| det-long-side-1280 | 26.63 [23.18, 30.38] | 58.98 | 9,537.2 | 22,553.2 | 5,963.5 | +2.31 [+1.66, +2.98] | −202.2 [−451.2, +32.6] |
| det-long-side-1792 | 24.21 [20.38, 28.30] | 60.67 | 10,168.1 | 22,592.5 | 6,576.8 | −0.11 [−0.93, +0.92] | +428.7 [+230.7, +647.8] |
| det-long-side-2048 | 23.75 [19.91, 28.17] | 61.10 | 10,400.0 | 21,713.0 | 6,556.8 | −0.57 [−1.80, +0.83] | +660.6 [+393.3, +949.6] |
| onnx-det-torch-rec | 24.32 [20.87, 28.15] | 60.50 | 2,579.1 | 5,186.9 | 1,550.0 | +0.00 [+0.00, +0.00] | −7,160.4 [−8,258.3, −6,171.7] |
| int8-recognizer | 24.32 [20.87, 28.14] | 60.46 | 6,890.9 | 16,071.7 | 6,896.5 | +0.00 [−0.02, +0.02] | −2,848.5 [−3,233.6, −2,505.1] |
| fused-attention | 24.32 [20.87, 28.15] | 60.50 | 4,020.8 | 8,267.1 | 2,393.0 | +0.00 [+0.00, +0.00] | −5,718.6 [−6,630.9, −4,883.7] |
| det-GW-DET-003 | 27.33 [23.60, 31.43] | 57.75 | 9,065.7 | 20,756.4 | 6,107.0 | +3.00 [+1.52, +4.33] | −673.7 [−888.5, −443.8] |

Page CER and E2E F1 (strict) on the 400 Phase 5 pages, on the CPU; time and memory on the 110
profile pages. The F1 and page-time intervals are in `pareto.md`.

| Point | Line CER % | Line mean ms | Line p95 ms | Δ CER (points) | Δ ms |
|---|---|---|---|---|---|
| v1-torch (v1) | 6.22 [5.73, 6.71] | 35.55 | 80.82 | +0.00 [+0.00, +0.00] | −60.04 [−80.33, −42.17] |
| v1-onnx (v1) | 6.22 [5.73, 6.71] | 95.59 | 340.04 | n/a | n/a |
| int8-recognizer | 6.21 [5.73, 6.69] | 63.35 | 220.31 | −0.01 [−0.04, +0.02] | −32.24 [−39.97, −24.87] |
| fused-attention | 6.22 [5.73, 6.71] | 39.53 | 112.10 | +0.00 [+0.00, +0.00] | −56.06 [−72.68, −41.23] |
| rec-GW-REC-003 | 8.34 [7.74, 8.92] | 9.07 | 21.15 | +2.12 [+1.82, +2.40] | −86.52 [−110.28, −65.59] |
| rec-GW-REC-006 | 7.85 [7.29, 8.41] | 16.67 | 42.32 | +1.63 [+1.39, +1.87] | −78.91 [−101.36, −58.96] |
| rec-GW-REC-020 | 8.21 [7.61, 8.78] | 29.03 | 97.18 | +1.99 [+1.72, +2.23] | −66.56 [−84.64, −50.34] |
| rec-GW-REC-022 | 7.57 [7.00, 8.14] | 100.57 | 387.76 | +1.35 [+1.13, +1.55] | +4.98 [+3.12, +6.86] |

Line CER on the 5,000 seeded lines (strict, greedy, one line at a time); time on the 100
`rec_samples` lines.

- The recognizer sets the page time. Three points keep v1's page CER (Δ CER +0.00 [+0.00, +0.00]
  or −0.00 [−0.00, +0.00]) and F1: v1-torch (2,637.3 ms), the ONNX detector with the PyTorch
  recognizer (2,579.1 ms) and the fused attention export (4,020.8 ms). INT8 is at +0.00 [−0.02,
  +0.02] points at 6,890.9 ms.
- Detector input size moves accuracy more than page time: from long side 1024 to 2048 the page
  time runs from −1,248.1 to +660.6 ms against v1-onnx's 9,739.4 ms.
- The recognizer points differ in recipe and budget, not only in size; GW-REC-003 and GW-REC-020
  read at height 32, v1 at 48.

*ADR-0011's targets* (`README.md`, GW-EFF-007; ms and MiB; not tightened here, the owner's call)

| Build | Line mean (≤ 25) | Line p95 (≤ 70) | Page mean (≤ 1,500) | Page p95 (≤ 2,700) | Peak RAM on pages (< 1,836) |
|---|---|---|---|---|---|
| v1 PyTorch, CPU, 10 threads | 33.9: over | 77.6: over | 2,565.7: over | 5,127.1: over | 1,641.8: within |
| v1 fp32 ONNX, CPU, 10 threads | 92.2: over | 304.4: over | 8,573.7: over | 19,011.5: over | 6,529.6: over |
| v1 fp32 ONNX, CPU, default threads | 104.7: over | 392.1: over | 9,924.9: over | 22,071.7: over | 6,541.5: over |

No CPU build of v1 meets the latency targets; only PyTorch's page RAM is within. The points closest
to the page targets, from the README's options table (GW-EFF-009; every row but v1 is a new
version, measured on validation only, and one that ships is evaluated in Phase 22):

| Point | Page mean ms | Page p95 ms | Peak RAM MiB | Page targets within | Page CER % |
|---|---|---|---|---|---|
| v1-torch (v1) | 2,637 | 5,328 | 1,665.0 | 1 of 3 | 24.32 |
| v1-onnx (v1) | 9,739 | 22,474 | 6,525.5 | 0 of 3 | 24.32 |
| onnx-det-torch-rec | 2,579 | 5,187 | 1,550.0 | 1 of 3 | 24.32 |
| fused-attention | 4,021 | 8,267 | 2,393.0 | 0 of 3 | 24.32 |
| int8-recognizer | 6,891 | 16,072 | 6,896.5 | 0 of 3 | 24.32 |

The detector-size points and GW-DET-003 are within none of the three (README). The options for the
owner, restated in ADR-0011's Phase 16 section (not a decision): keep v1 with PyTorch as its
default backend, as now; adopt a variant as a new version (onnx-det-torch-rec is the closest at
v1's accuracy), evaluated in Phase 22 if it ships; restate the target for the PyTorch build; or set
a GPU target as a separate decision. PyTorch on the RTX 4050 reads a page in 1,049 ms mean
(GW-EFF-007), but ADR-0011's targets are for the CPU. The miss is R-23 in the risk register.

*Findings* (`research/findings.md`)
- F13 revised: its detector half came from an arena-on side test; with the arena off, as the
  backends run, the exported detector runs at about PyTorch's speed and the recognizer about 2.7
  times slower per line (F34).
- F34: v1's plain fp32 ONNX export is the same model but slower on this CPU, almost all of it in the
  recognizer network (GW-EFF-006, GW-EFF-007).
- F35: the plain export's unfused attention is the main reason the exported recognizer is slow:
  one com.microsoft.MultiHeadAttention node per global mixer keeps v1's page CER and F1 and cuts
  the ONNX page time from 9,739.4 to 4,020.8 ms, still slower than PyTorch (GW-EFF-009).
  (Correction, 2026-11-18: no interval covers that pair. F35 now gives both values, 4,020.8 and
  2,637.3 ms, and states no difference.)
- F36: dynamic INT8 of the constant-weight MatMuls (H7′ holds); the attention products stay fp32,
  and on pages INT8 removes about half as much time as fusing the attention (GW-EFF-008,
  GW-EFF-009).
- F37: detector input size and the smaller detector move page accuracy but change the ONNX page
  time by at most about 13%; the recognizer sets the page time (GW-EFF-009).
- F38: no CPU build of v1 meets ADR-0011's latency targets; the lowest mean page time at v1's
  accuracy is the ONNX detector with the PyTorch recognizer, still over the 1,500 ms and 2,700 ms
  page targets (within the RAM target) (GW-EFF-007, GW-EFF-009).
- F39: H6′ not met; the plain export's unfused attention (F35) is the likely reason the ONNX gap is
  wider than PyTorch's, not measured for GW-REC-020 (GW-EFF-011).
- F40: H1′ ONNX not met for bundle `875357a6…`; H1′ still holds from Phase 14 (GW-EFF-010,
  GW-BASE-002 amendment).

**Records:** experiment entries GW-EFF-006 to 011 and the GW-BASE-002 amendment
(`research/experiments.md`); H7′, H6′ and H1′ ONNX (`research/hypotheses.md`); F13 revised, F34
to F40 and F9's status pointing to F37 (`research/findings.md`); the two run problems
(`research/failures.md`); ADR-0011's Phase 16
section; R-23 in `docs/RISK_REGISTER.md`; the compute ledger (`reports/compute/ledger.md`); the
change rows registered with the plan (`docs/EXPERIMENT_PROTOCOL.md` §11, the three rows of
2026-11-17: the reference-based same-model check, GPU conditions on every profile with the mirrored
order, the single-model exception and the four confirmatory tests, and Phase 16's baseline
profile), unchanged by the results; the `glypharis/export/onnx.py` docstring is deferred
(Deviations).

**Deviations from the plan**
- *The stage-check tolerance changed mid-chain* (`a8e8bb6`, after step 10 failed; see the run
  above). The fixed 1% could not hold on short GPU pages. Only the guard on the stage split changed,
  and no number depends on it.
- *Variant reports rewritten and re-scored* (`1253dfe`). After scoring, a privacy scan found that 19
  variant reports (det-GW-DET-003 and the rec-GW-REC-* variants) recorded the training checkpoint's
  absolute local path. The variant reports now record it from the checkpoints folder on
  (`report_record` and `portable_path` in `glypharis/efficiency/variants.py`; the work-folder
  records keep the full path). In the 19 files exactly that one line was rewritten to
  `checkpoints/<ID>/best.pt`, with the originals kept outside the repository, and scoring was
  re-run at `1253dfe`: only those 19 input hashes and the scorer commit changed in results.json,
  and every number is the same. The 19 files still record `a8e8bb6` and a clean tree from their
  runs.
- *The `glypharis/export/onnx.py` docstring is not corrected yet* (item 9). It still says the
  exported detector runs about twice as fast as PyTorch and the recognizer about 4.5 times slower.
  The file is inside the frozen v1 paths that `assert_v1_code` (`glypharis/efficiency/v1.py`)
  compares byte for byte against `phase14-eval`, and the Phase 17 re-run builds v1 through that
  check, so editing it now would break the freeze. F13 is corrected now; the docstring is corrected
  when the v1 code is next allowed to change, after the Phase 17 re-run, in the phase that changes
  it.
- *Two run commits.* Steps 1 to 9 record `6950a5a`, every later step `a8e8bb6`, and the scorer
  `1253dfe`; the tolerance is the only code difference between the two run commits.
- *GW-EFF-007's rows are one run each*, in chain order, not mirrored pairs. The plan's mirrored rule
  is written for compared arms, and the cost table is read as a table; the mirrored comparison of
  PyTorch and ONNX is GW-EFF-009's (lines 35.55 against 95.59 ms, pages 2,637.3 against 9,739.4
  ms). F34 quotes the GW-EFF-007 values.
- *No GPU state in the GW-BASE-002 profile* (item 3 and the §11 change row that asks for GPU
  conditions on every profile). The fresh baseline profile is written by the Phase 5 runner,
  unchanged apart from `--phase 16`, which records mains, the plan and the power mode but not
  nvidia-smi's clocks and temperature. Every GLYPHARIS profile records them.
- *No per-process peak VRAM from nvidia-smi* (item 2). The Windows driver does not report it, so
  the cost table gives torch's peak and the whole GPU's memory at the end instead.

The rest of items 2 and 3 is in the reports: the default-threads rows for fp32 ONNX only, GPU line
latency at batch 1 and throughput at batch 32, the stage split, the slope with its page-bootstrap
interval, the mirrored orders, the conditions in every profile and the GPU state in every
GLYPHARIS profile, and the interval count.

**Compute:** 15.7 CPU-hours and 0.1 GPU-hours in the compute ledger, 95 jobs, of which one did not
finish: the 110 s GPU attempt of step 10. That is against the plan's estimate of about 12 to 14
CPU-hours; GW-EFF-009 took 12.51 hours against the 6 to 8 estimated. The re-score at `1253dfe` ran
by hand and is not in the ledger (`reports/compute/ledger.md`).

**Tests:** the CPU suite passes in a clone without tags (1,207 passed, 4 skipped, at `7f13926`);
new tests cover the reference writer and check, the frozen export path, the ONNX and GPU profile
paths and the stage split (with the 5 ms slack), the variant builder and its portable report
record, INT8 on a tiny SVTR, the fused attention against the reference attention with the padding
mask, `--phase 16` in the baseline runner, the Phase 16 scorer, and the ledger's Phase 16 lines.
CI is checked on the PR.

**Exit criteria check** (IMPLEMENTATION_PLAN Phase 16)

| Criterion | Result | Evidence |
|---|---|---|
| Efficiency table, measured on this laptop with documented conditions | pass | `cost_table.md` (GW-EFF-007), conditions in every profile; `efficiency_comparison.md` |
| Accuracy/latency Pareto plot | pass, on validation; every point but v1 a new version | `pareto.md`, `pareto.png` (GW-EFF-009) |
| INT8 measured with its accuracy loss | pass: strict line CER 6.221% (fp32) and 6.208% (INT8); page CER +0.00 [−0.02, +0.02] points (descriptive) | `int8.md`, `verdicts.md` (GW-EFF-008, GW-EFF-009) |
| The four verdicts | pass: H7′ latency met, H7′ accuracy met (H7′ holds), H6′ not met, H1′ ONNX not met | `verdicts.md` |
| CI green | pass | runs 44249945309 (branch), 44249996705 (pull request), 44253277519 (`main`) |

**Known issues**
- `cost_table.md` says each page's stage sum was checked within 1% of a second clock. Since
  `a8e8bb6` the check allows the larger of 1% and 5 ms, and the GPU page profile's largest relative
  difference is 3.03%. The report stays as generated.
- ADR-0011's latency targets are still missed by every CPU build of v1 (R-23).
- One laptop; latency on HierText validation pages (40.9 lines per profile page), not documents.
  The recognizer and detector size points differ in recipe and budget, not only in size.
- The H1′ ONNX verdict applies to bundle `875357a6…` only; if the Phase 17 re-run ships another, it
  does not carry over.
- `glypharis/export/onnx.py`'s docstring still gives F13's old figures (the exported detector about
  twice as fast as PyTorch, the recognizer about 4.5 times slower); it is corrected when the frozen
  v1 code may next change, after the Phase 17 re-run (Deviations).
- Batch size: lines are timed at batch 1 on the CPU and at batch 1 and 32 on the GPU, pages with
  lines batched as the pipeline batches them; no CPU batch-size sweep was run (the registered
  plan's scope, item 2).

**Git:** branch `phase/16-efficiency`, merged through pull request #21. Merge commit `0af7fc4`, tag
`phase-16` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 44253277519).

---

## Phase 21 — Research package

**Status:** COMPLETE (2026-11-18). The plan was registered before any Phase 21 deliverable. Opened
2026-11-18 on `phase/21-research-package`, branched from `main` after the end of the v1 code freeze
(merge `6caf327`). It reads recorded reports only: no training, evaluation or test data, and no
GPU. Line numbers below are for `034b867`.

**Objective:** turn the recorded evidence into a package a reader can check:

- `MODEL_CARD.md`
- a corrected `DATASETS.md`
- a generated `EXPERIMENTS.md`
- `research/paper/*`
- a complete, verified `references.bib`

Exit criterion (`docs/IMPLEMENTATION_PLAN.md:398-402`): every number in the paper notes traces to an experiment ID and a report file. This plan applies the same check to `MODEL_CARD.md` and `DATASETS.md`, because `docs/WORKFLOW.md` §6 applies to every doc.

**Fixed rules**

- No training, evaluation, re-scoring or model loading.
- The tools read only `reports/`, `research/`, `docs/`, `configs/` and `data/manifests/`. They never open `data/processed`, `data/raw`, `runs/` or `checkpoints/`.
- No change under `glypharis/pipeline`, `glypharis/models`, `glypharis/export`, `glypharis/data` or `configs` (`V1_PATHS`, `glypharis/efficiency/v1.py:21-27`). An untracked file there also fails `assert_v1_code`, so no new tool input goes in `configs/`.
- No change to `glypharis/eval/` or `scripts/phase14_*.py`: Phase 22 re-scores with them.
- No existing generated report is edited or regenerated. New generated files only.
- GDC-scan appears as numbers only (ADR-0013 item 4). Lines from the two CC BY-SA textbooks are never quoted. No personal paths, names or emails.
- Commits follow `docs/WORKFLOW.md` §2: the owner's identity, no tool or co-author trailers.

**Conventions decided here (no owner input needed)**

| Topic | Decision |
|---|---|
| Dataset counts | A tracked manifest or report path stands in for an experiment ID, because acquisition and preparation have no GW IDs. Anything measured by a run keeps its ID. |
| Default track | Strict, unless stated. |
| Intervals | Every interval says whether it is by line, page or document. |
| Derived values | Only through the generated derived-values file (item 9). |
| Paper depth | Full prose for methodology, experiments, results, limitations and conclusion. Abstract and introduction are complete but short, and the abstract says results predate final validation. |
| Consumer name | `MODEL_CARD.md` and the paper say "a document-reading application" (plan §2, brief §66). |
| Code licence and citation | "To be settled in Phase 23" (plan, Phase 23 Publishing). |
| Compute statement | Limited to `reports/compute/ledger.md`, with uncounted experiments listed by ID. |
| `data/DATASET_MANIFEST.yaml` | Not regenerated. `DATASETS.md` gets a table mapping each brief §13 field to the manifest that holds it, and the deviation goes in plan §8. |

**Work items**

**A. Decisions and registration**

**1. Register this plan.** Add the Phase 21 section to `docs/PHASE_STATUS.md` and set the status row to in progress, before any deliverable is committed.
*Check:* the plan commit comes first in `git log main..HEAD`.

**2. ADR-0014, weights and backbone terms.** Record the owner's answers to the questions at the end of this plan. Update:
- `docs/decisions/README.md`
- `docs/RISK_REGISTER.md` R-05
- `README.md:118-121`
- `DATASETS.md:33,240-243,264-271`

ADR-0009's rule is that sources must allow commercial use, and `DATASETS.md:241` says the ImageNet terms do not clearly allow it. The ADR must state how v1 relates to that rule.
*Check:* `git grep "before v0.1"` finds no open licence wording.

*Note, 2026-11-18.* The owner's answers are the "Owner decisions" paragraph at the end of this
section.

*Note, 2026-11-18.* The check still finds the licence note written at download in `data/sources.yaml`,
`data/manifests/imagenet_backbones.json` and `data/DATASET_MANIFEST.yaml`, and they keep it on
purpose. They are source records: `data/sources.yaml` is the download registry the manifests come
from, and the number trace cites `imagenet_backbones.json` by SHA-256, so an edit would change a cited
hash. `DATASETS.md` (MobileNetV3 card, known issues) and ADR-0014 carry the correct statement, and
`docs/IMPLEMENTATION_PLAN.md` §8 records the exception. A later data pass may regenerate these files.
The hit at line 685 of this file is the Phase 8 record, and the one above is this check.

**B. Registry repair (`research/experiments.md`, `research/failures.md`)**

**3. GW-REC-022** (row L38, entry L670-688): set status done, run date 2026-10-25.
- Key result: val CER 7.56% [7.40, 7.71]; −0.76 [−0.81, −0.71] against GW-REC-003.
- Training time 4.65 GPU-hours; 45,295 steps; 7,928,835 parameters.
- Interpretation: the hypothesis of a gain below the sum did not hold (F3).
- Sources: `reports/training/phase07/ablation_table.md:36`, `reports/training/GW-REC-022/report.md`, `reports/training/compare/GW-REC-022_vs_GW-REC-006.md`, ledger.
- State that the interval is the ablation table's; the run report prints [7.41, 7.72].

**4. Other registry fixes**
- Report cell becomes `—` for GW-REC-023, 032, 033 (L39, L84, L85).
- Add `not run` to the status list (L4).
- Add GW-REC-023 to `research/failures.md`, as `docs/WORKFLOW.md` §4 requires. `docs/PHASE_STATUS.md:547` already cites it there.
- Fill run dates at L308, L327, L612 from the ledger.
- Fill blank fields at L624-628 and L703-707 with `—` or "not applicable".
- Reword GW-EFF-001's key result (L45): the detector alone meets the page target and the whole pipeline does not.
- Add second report paths:
  - 15 Phase 7 rows: `reports/training/phase07/ablation_table.md`
  - GW-REC-002, 024: their `reports/training/compare/` files
  - GW-DET-002, 003: `reports/training/GW-DET-001_vs_GW-DET-00N.md`
  - GW-DET-007: its `pair.json`
  - GW-EFF-005: `reports/external/results.json`
  - 18 rows: `reports/robustness/<ID>.json`
  - GW-DATA-010 to 021: `reports/synthetic/phase12_comparison.json`
- Add an optional `Superseded by` field for v0.1 rows, confirming each against its entry: E2E-001→006, E2E-003→005, E2E-004→011, EFF-002 and 004→007, EFF-003→006.
- Put explicit anchors inside headings: `### <a id="gw-rec-022"></a>GW-REC-022 — …`. This keeps line numbers and all 101 `#gw-xxx-nnn` links, with no slug logic.

**5. Close the open seeds promise.** `docs/PHASE_STATUS.md:517,525` says two extra GW-REC-024 seeds would be added "before the paper". They are GW-REC-032 and 033, not run because option B was chosen (GW-REC-037), and v1 has three seeds (GW-REC-034 to 036). Record this as closed for v1. In `limitations.md`, say the v0.1 recognizer stayed at one seed.

*Check for B:* `python scripts/experiment_summary.py --check` passes.

**C. Tooling (standard library, `main(argv)`, LF output, no timestamps)**

**6. `scripts/numtrace.py` and `tests/test_numtrace.py`.** Shared by both checkers, landed first.

*Tokenizer.* Markdown-aware. It skips code spans, link targets, list markers, heading numbers and comments. It handles thousands separators, Unicode minus, explicit plus and exponents. It never treats these as numbers: GW IDs, F/H/RQ/ADR/R numbers, `Phase N`, dates, hashes, file names, seeds, versions, or digits inside names (`fp32`, `INT8`, `p95`, `F1@0.5`, `PP-OCRv5`, `MobileNetV3`, `CC BY 4.0`, `RTX 4050`).

*Report index.* Excludes free-text content:
- tables with a header cell exactly `Reference` or `Prediction` (`Reference class` tables stay indexed)
- `words_*.md`
- `examples.md`, apart from its counts

*Directive syntax, frozen before parallel work starts:*

```
<!-- trace: GW-REC-040 reports/error_cases/phase15/recognition.md#gdc_test -->
<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md key=row|key=row,header -->
<!-- derived: name -->
```

**7. `scripts/experiment_summary.py`, `tests/test_experiment_summary.py`.** Implement the third reader's design.

*Outputs:* `EXPERIMENTS.md`, `reports/experiment_summary.csv`, `reports/experiment_summary.md`.

*Validation rules R1-R11.* Writes nothing on failure; `--check` compares after LF normalisation. Adjustments to that design:
- Link with the new anchors, so no GitHub slug code is needed.
- `research/failures.md` headings can hold several IDs (for example GW-EFF-003 / GW-EFF-004 / GW-E2E-004).
- R7 stays a loose presence check for registry cells. It does not count as evidence for the paper.
- Merge the ledger's `GW-EFF-007 (GPU jobs)` and `GW-BASE-002 (Phase 16 amendment)` into their IDs. Show `Phase 16 scoring` as unattributed.
- List the 28 registered IDs without a ledger row. Update the ledger's "Not in these logs" note only if the logs allow it; otherwise the summary states the gap.

*Tests:* one failing case per rule, determinism, `--check` stale and current, and no file written on failure.

*Note, 2026-11-18.* The third reader's design is recorded in the docstring of
`scripts/experiment_summary.py` (rules R1-R11).

**8. `scripts/experiment_comparison.py`, `tests/test_experiment_comparison.py`.** Brief §63 asks for a generated table of Model, Dataset, CER, WER, F1, Latency, Params, Model Size and VRAM. No report has it, and no reader scoped it.

*Outputs:* `reports/benchmark_results/comparison_table.md` and `.csv`, built from:
- `reports/external/results.json`
- `reports/baselines/results.json`
- `reports/efficiency/phase16/results.json`
- `reports/benchmark_results/efficiency_comparison.json`

Missing cells are `n/a`, never imputed. Each row names its accuracy ID and its latency ID. A header note says latency is HierText validation, from different sittings, so no difference may be stated.

**9. `scripts/trace_numbers.py`, `scripts/derive_values.py`, tests.**

*Inputs and outputs*
- Scope: `research/paper/*.md`, `MODEL_CARD.md`, `DATASETS.md`.
- Each document has its own `research/paper/trace/<doc>.json` holding its protocol constants and derived formulas, so parallel authors do not share a file.
- Outputs: `reports/paper_trace/derived.json`, `trace.json`, `trace.md`. Each number gets document line, class, ID, source file, source line and source SHA-256.

*Matching rules*
- Scope resolves nearest first: directive, block, caption or preceding paragraph, section, file header.
- The ID must be registered as done and must match the report's `Generated by` header or JSON run field. A small explicit map covers reports without one.
- Rounding: within half a unit of the printed last digit, inclusive.
- A printed value may not be more precise than its source.
- `v [lo, hi]` must match as a triple on one source line.
- ×100 is allowed only for JSON known to store fractions. Other unit changes need a derived entry.
- Numbers with fewer than 3 significant digits must match in narrow scope.
- `DATASETS.md` may use `data/manifests/` and a `published` class for authors' figures.

*Lints*
- Fail on personal paths and emails.
- Fail on claims that Phases 22 or 23 happened.
- Fail on SOTA wording outside quoted literature.
- List `best` for human review, without failing.

*Mutation audit.* `--mutation-audit --seed 20261118` changes sampled values and writes the escape rate to `trace.md`. Target: at most 5% per document. Above that, add anchors or table bindings.

*Tests:* changed digit, swapped interval, missing ID, undeclared derived value, rounding boundary, free-text exclusion, and a guard that rejects paths outside the allowed roots.

**10. CI.** Add named steps to `.github/workflows/ci.yml` after `Validate configs`:
- `experiment_summary.py --check`
- `experiment_comparison.py --check`
- `trace_numbers.py --check`

**D. Deliverables**

**11. `MODEL_CARD.md`.** Status line: evaluated and packaged; final validation (Phase 22) and release (Phase 23) not done; no weights distributed.

| Brief §53 field | Sources |
|---|---|
| Model description, version | `configs/final/glypharis_v1.yaml`, `reports/efficiency/phase16/v1_bundle.json`, `pyproject.toml:7` |
| Intended / not intended use | ADR-0008, ADR-0011, `reports/charset/charset_report.md`, `reports/external/end_to_end.md` (rotated, vertical), `reports/external/official.md` |
| Architecture | `configs/rec/gw-rec-020.yaml` chain, `configs/det/gw-det-001.yaml`, `reports/efficiency/phase16/cost_table.md:11-13`; frame stride from `reports/efficiency/phase16/profiles/lines_onnx_cpu_10t.json:7` |
| Training data, recipe | `data/manifests/processed/hiertext.json`, `reports/training/GW-REC-034/report.md`, `reports/training/GW-DET-001/report.md`, `reports/ablation/phase13_final.md` |
| Evaluation data | `reports/external/README.md`, `run_record.md`, `research/test_access_log.md` |
| Metrics | `reports/external/{recognition,end_to_end,official,comparisons,calibration}.md`, `reports/robustness/GW-REC-034.json`, `reports/efficiency/phase16/{cost_table,README,verdicts}.md`, new comparison table |
| Limitations, failure modes | `reports/error_cases/phase15/{pipeline,recognition,glyphsynth,oracle_share}.md`, GW-EFF-012, GW-E2E-011, GW-E2E-012 |
| Hardware | `reports/environment/env_report.md`, conditions block of `reports/efficiency/phase16/profiles/pages_torch_cpu_10t.json` |
| Licence | ADR-0014, `DATASETS.md`, ADR-0007, ADR-0009 |
| Citation | Method keys now; project citation pending Phase 23 |

Add a short section mapping brief §64's ten fields to their sources, and a separate "Not v1" section for the fused-attention candidate.

Numbers that must not be confused:
- Validation line CER: 6.30 (GW-REC-034, 73,888 lines), 6.51 (GW-REC-038, 74,500 crops), 6.22 (GW-EFF-009, 5,000 lines). Only 6.51 sits beside test results.
- Validation page CER: 27.57 (GW-E2E-006, 1,715 pages) against 24.32 (400 pages).
- GLYPH-C clean CER 1.93 is on a pack half made of GlyphSynth lines, so it is not comparable with HierText CER.
- IIIT5K headline is the alphanumeric row; the strict row is not.
- `page_cer_phase10` in `reports/package/GW-E2E-011.json` holds the 400-page package reference (24.32%), despite the key name.
- Unfavourable comparisons stay in: GDC crops, IIIT5K, OmniDocBench, FUNSD detection, H1′ ONNX, missed ADR-0011 targets.

*Check:* trace passes, and a reviewer ticks all 13 fields.

**12. `DATASETS.md`.**

*Corrections*
- L65: 3.64 GiB archives plus 79 MiB ground truth (3.72 GiB in total).
- L23, L266: no delivered model trained on GlyphSynth.
- L26, L123-133: GDC version 1.1, frozen 2026-10-22; 4 of 106 captures are landscape.
- L110-115: no transcribed old-scan subset exists.
- L92, L173-174, L191: remove stale future tense.
- Update L9 check dates.

*Additions*
- cards or rows for GLYPH-C v1, the Phase 12 packs, evaluation crop packs and subsets, GDC-scan, baseline weights and official evaluators
- per-source GDC attribution table
- usage matrix by dataset, phase and experiment
- known issues and the §9 and §54 fields for every card
- redistribution section
- §13 field map

*Crop counts.* Publish HierText's reconciliation:
- 73,888 + 590 + 22 = 74,500
- 67,277 + 282 + 15 = 67,574

For GDC, publish 8,606 lines, 8,372 crops and the rule `is_rec_line` (`glypharis/eval/page_metrics.py:61-70`), with no breakdown. The 200 low-score and 32 vertical lines explain at most 232 of the 234 difference. Do not open the page file to explain the other 2.

*Licence wording*
- Hefferon: GNU FDL or CC BY-SA 3.0 United States, at the user's choice. No election is needed, because no line is redistributed.
- Re-confirm the wording for each licence checked, with the date.

**13. Attribution and privacy** (`DATASETS.md`, `reports/README.md`, `glypharis/training/report.py`, `scripts/gdc_scan_build.py`).
- Add a HierText and Open Images notice to `reports/README.md`, covering the 53 training reports with worst-line tables and the other unattributed crops and previews. The reports themselves are not edited.
- Make `glypharis/training/report.py` write the notice in future reports, with a test.
- Cut `DATASETS.md` L134-150 to hashes, method, counts and the case-study limit. Remove the physical description, authoring-software remark, ignore-region contents and out-of-charset character names.
- Remove matching detail from the docstring in `scripts/gdc_scan_build.py` and the identifier list at `docs/PHASE_STATUS.md:2176-2177`.
- Repeat the Phase 14 pre-push search of the diff for known identifying strings, kept local and never written to the repository.

*Note, 2026-11-18.* Done: the search ran over every file changed on this branch at `955c7a9`,
including `reports/paper_trace/`, and found nothing. The search terms are kept local.

**14. `research/paper/references.bib`.**
- Verify all 29 entries against publisher or arXiv pages.
- Add DOI or eprint, volume, number and pages where they exist.
- Fix `liao2022dbnetpp`, `ouyang2025omnidocbench`, `efron1994bootstrap`, and the two arXiv `@article` entries.
- Remove the "written from memory" header only when finished. Update `research/approach_survey.md:8`.
- Citation keys are frozen before drafting: `cui2025paddleocr`, `paddleocr_ppocrv6`, `kuznetsova2020openimages`, `russakovsky2015imagenet`, `harley2015rvlcdip`, `lin2017fpn`, `nagy1992xycut`, `levenshtein1966`, `naeini2015ece`, `loshchilov2019adamw`, `smith2019onecycle`, `paszke2019pytorch`, `onnxruntime`, `doctr`, `easyocr`.
- If no citable PP-OCRv6 paper exists, use `@misc` for the pinned model pages in `data/sources.yaml`.

*Check:* a small test that every backticked key resolves, plus a review log of the source page per entry.

**15. `research/paper/*`.**

| File | Work | Main sources |
|---|---|---|
| `methodology.md` | Pipeline, data, metrics, training method, calibration, §64 record; a code path and config per subsection | `configs/final/glypharis_v1.yaml`, `docs/EXPERIMENT_PROTOCOL.md`, GW-REC-034, GW-DET-001, GW-E2E-005 |
| `experiments.md` | Setup, baselines, ablation design, compute, failures, v1 re-runs; replace L6 | `reports/baselines/README.md`, `research/ablations.md`, ledger, `research/failures.md` |
| `results.md` | RQ1-RQ8 answer table first, then accuracy, baselines, external sets, official evaluators, calibration, robustness, synthetic data, ablations; add the attribution note | `reports/external/*`, `reports/robustness/phase11_comparison.md`, `reports/synthetic/phase12_comparison.md`, `reports/ablation/*` |
| `efficiency.md` | Fix L58 (use or explain the mirrored 3.69 ratio), L129, L131 through derived values; L181-182 and L201-202 for the owner decision and unchanged bundle; source L200 | `reports/efficiency/phase16/*`, ADR-0011, GW-EFF-012 |
| `error_analysis.md` | Trace directives; add GW-BASE-004 or say why it is left out (`recognition.md:422-423`) | `reports/error_cases/phase15/*` |
| `abstract_notes.md`, `introduction_notes.md` | Remove the scene-text domain study; narrow TrOCR to crops; state the owner decisions | F21, F22, F27, F28, F34, F35, F38, `research/hypotheses.md` |
| `related_work.md` | Prose; mark SynthText, SROIE, DocVQA as not used; fix L38-39 | `research/approach_survey.md`, verified bibliography |
| `limitations.md`, `conclusion_notes.md` | F24, F26, F33, no phone measurement, set sizes, single detector seed, post hoc H4 | findings, hypotheses, known issues for Phases 14-16 |

The ablation text must say how brief §45 is met: the ladder is recognizer-only, input height is the preprocessing row (L3), and detector changes are GW-DET-002, 003, 004, 005, 007.

**16. `docs/` and `research/` finalisation** (brief Phase 21 names both).
- `docs/ARCHITECTURE.md`: replace the Phase 0 draft values with as-built facts and sources.
- `docs/ROADMAP.md` §4: tick criteria whose phase gates have passed. Leave model card and dataset documentation for this gate. Leave "Reproducible final experiment" open for Phase 22.
- `docs/IMPLEMENTATION_PLAN.md`: add the fused-attention evaluation to Phase 22 and record new deviations in §8.
- `docs/ENVIRONMENT.md`: point to `reports/environment/env_report.md`; add no new measured values by hand.
- `README.md`: link the new documents and keep its status accurate.
- `reports/README.md`: note where brief §62's training curves live.
- `research/findings.md`, `research/hypotheses.md`: consistency pass only.
- `research/research_log.md`: Phase 21 entry.

**Claims only Phases 22 and 23 can support**

| Do not claim | Required wording |
|---|---|
| Final, validated, reproduced from a clean machine | "Final validation (Phase 22) has not been run." |
| Test metrics reproduced through the shipped CPU bundle | "Test results come from the GPU strict-fp32 evaluation; equality with the bundle is shown on validation only. Phase 22 checks reproduction within the 0.05-point tolerance." |
| Fused-attention accuracy on test sets, or that it is v1 | "A new version, not v1; validation and laptop figures only; to be evaluated in Phase 22." |
| Phone speed or phone suitability | "Phones are the target hardware; no phone has been measured." |
| Released, v1.0, available weights | "No weights are distributed; release is Phase 23." |
| Effective code licence, PyPI, Zenodo DOI, `CITATION.cff` | "To be settled in Phase 23." |
| Weight licence in force | "Intended terms under ADR-0014, effective only if released." |
| "Reproducible final experiment" complete | "Configuration frozen and recorded; reproduction pending Phase 22." |

**Order of work**

| Stage | Package | Owns exactly these files | Depends on |
|---|---|---|---|
| 0, serial | Integrator | `docs/PHASE_STATUS.md`, `research/research_log.md` | none |
| 1, parallel | P1 Registry | `research/experiments.md`, `research/failures.md` | Stage 0 |
| 1, parallel | P2 Number core | `scripts/numtrace.py`, its test, directive spec | Stage 0 |
| 2, parallel | P3 Summary and comparison | `scripts/experiment_summary.py`, `scripts/experiment_comparison.py`, their tests | P2 |
| 2, parallel | P4 Trace | `scripts/trace_numbers.py`, `scripts/derive_values.py`, their tests | P2 |
| 2, parallel | P5 Card | `MODEL_CARD.md`, `research/paper/trace/MODEL_CARD.json` | directive spec |
| 2, parallel | P6 Data and privacy | `DATASETS.md`, its trace file, `reports/README.md`, `glypharis/training/report.py` and test, `scripts/gdc_scan_build.py` | directive spec |
| 2, parallel | P7 Methods | `methodology.md`, `experiments.md`, their trace files | directive spec |
| 2, parallel | P8 Results | `results.md`, its trace file | directive spec |
| 2, parallel | P9 Existing drafts | `efficiency.md`, `error_analysis.md`, their trace files | directive spec |
| 2, parallel | P10 Framing and bibliography | `abstract_notes.md`, `introduction_notes.md`, `related_work.md`, `limitations.md`, `conclusion_notes.md`, `references.bib`, `research/approach_survey.md`, their trace files | frozen keys |
| 2, parallel | P11 Docs | `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/ENVIRONMENT.md`, `README.md` except its licence section | none |
| 3, serial | Integrator | ADR-0014, `docs/decisions/README.md`, `docs/RISK_REGISTER.md`, `README.md` licence section, `.github/workflows/ci.yml`, all generated outputs, `docs/PHASE_STATUS.md:2176-2177` | owner answers, all packages |
| 4, serial | Review and gate | Fixes go back to the file's owner package | Stage 3 |

Licence wording is drafted against the placeholder `ADR-0014` and completed in Stage 3.

**Review plan**

1. **Tool review:** read both checkers against their tests, and inspect the mutation audit.
2. **Number review:** an independent seeded sample of 50 numbers per long document and 25 per short one, checked by hand against the labelled report cell.
3. **Claim review:** every sentence with a comparison, cause or status in `MODEL_CARD.md`, `abstract_notes.md`, `results.md` and `limitations.md`, against the Phase 22/23 table.
4. **Licence and privacy review:** attribution notes, redistribution statement, GDC-scan numbers-only, no textbook lines, no personal data, local diff search.
5. **Completeness review:** tick every line of the coverage table below.
6. **Fresh-reader review:** starting only from `MODEL_CARD.md`, reach the source report of ten numbers.

*Note, 2026-11-18.* No separate coverage table was written. The coverage list for review 5 is items
1-16, the Fixed rules, the Conventions, the Phase 22/23 claims table, the Phase 21 entry in
`docs/IMPLEMENTATION_PLAN.md`, and brief §§9, 13, 43, 45, 53, 54, 62-65 and Phase 21.

**Gate checks**
- Ruff, `ruff format --check`, mypy, pytest in a clone without tags, `scripts/validate_configs.py`, all three `--check` commands, `trace_numbers.py` exit 0.
- `assert_v1_code` passes.
- `git diff main...HEAD -- research/test_access_log.md configs glypharis/pipeline glypharis/models glypharis/export glypharis/data glypharis/eval scripts/phase14_score.py` is empty.
- `git diff --name-status main...HEAD -- reports` shows only added generated files and the `reports/README.md` edit.
- Render `EXPERIMENTS.md` on GitHub and test several anchors.
- CI green on the branch; the owner merges.



**Owner decisions (2026-11-18, ADR-0014):** GW-DET-001, initialised from ImageNet weights, stays
v1's detector as a disclosed exception to ADR-0009's commercial-use rule; the intended terms for the
v1 weights are CC BY-SA 4.0 with HierText and Open Images attribution, effective only on a Phase 23
release. Neither is a legal conclusion.

**Exit criteria** (IMPLEMENTATION_PLAN Phase 21): every number in the paper notes traces to an
experiment ID and a report file.

**Results (2026-11-18)** (recorded evidence only: no model was run, no test data was read, and
`research/test_access_log.md` has no new row)

- *The package:* `MODEL_CARD.md`, a corrected `DATASETS.md`, the generated `EXPERIMENTS.md` and
  `reports/experiment_summary.*`, the generated comparison table
  (`reports/benchmark_results/comparison_table.*`), the paper notes in `research/paper/`, a
  bibliography of 44 entries with its review log (`research/paper/trace/references_review.md`), and
  ADR-0014.
- *The exit criterion:* all 5,128 numbers in the 12 traced documents resolve to an experiment ID and
  a report or recorded source (`reports/paper_trace/trace.md`, written by
  `scripts/trace_numbers.py`). Dataset counts trace to tracked manifests, and authors' published
  figures in `DATASETS.md` are marked as such.
- *How far the check can be trusted:* two seeded audits (seed 20261118) change numbers and count the
  changes that still pass. The highest rates are 1.8% for changed digits (`methodology.md`) and 1.0%
  for values taken from another row of the same source (`efficiency.md`); the target is at most 5%
  per document. A passing trace shows that a number is in the cited place. It does not show that the
  sentence interprets it correctly, which is why the claims were reviewed separately.
- *Reviews:* six independent reviews (tools, a seeded sample of 350 numbers checked by hand against
  their reports, claims, licences and privacy, completeness, a fresh reader) raised 54 issues. The
  sample found no wrong value, though several numbers were tied to the wrong row or experiment. All
  54 are closed. The first checker passed a correct interval taken from the wrong row and an interval
  with its minus signs removed; it was rebuilt to refuse both. The abstract and conclusion had left
  out Tesseract 5, the closest baseline for GDC order-aware page CER, where no difference can be
  stated; they now give it.
- *Findings revised:* F22 overstated one comparison (PP-OCRv5 mobile's bag-of-lines interval on GDC
  excludes zero) and F35 claimed a difference no interval supports.
- *Privacy (item 13):* the search of this phase's changes for the strings that would identify
  GDC-scan's author found none. The search terms are kept local.

**Exit criteria check** (IMPLEMENTATION_PLAN Phase 21)

| Criterion | Status | Evidence |
|---|---|---|
| `MODEL_CARD.md`, `DATASETS.md`, `EXPERIMENTS.md` (generated) finalised | pass | the files; `scripts/experiment_summary.py --check` |
| `research/paper/*` finalised from recorded evidence | pass | `research/paper/`, `reports/paper_trace/trace.md` |
| `references.bib` complete | pass, with three fields on secondary records | `research/paper/trace/references_review.md`, `tests/test_references.py` |
| Every number in the paper notes traces to an experiment ID and a report file | pass | `scripts/trace_numbers.py --check`, in CI |
| CI green | pass | runs 44339800784 (branch), 44339885181 (pull request), 44343507005 (`main`) |

**Tests:** the CPU suite passes in a clone without tags (2,072 passed, 5 skipped, at `955c7a9`), with
the three generated-file checks and `scripts/validate_configs.py`.

**Compute:** none beyond tests and report parsing; no GPU.

**Known issues**
- `data/sources.yaml` and the manifests made from it keep two old licence wordings (item 2's note).
  Older generated reports keep "HierText annotations CC BY-SA 4.0" and Hefferon as plain CC BY-SA
  3.0; they are not edited, and `DATASETS.md` has the current statements.
- Some licence pages could not be opened on 2026-11-18; `DATASETS.md` lists them. Attribution for
  Open Images is at dataset level, not per image.
- Three bibliography fields rest on secondary records and need a check in a browser.
- The older, longer description of GDC-scan stays in the public git history. It contains none of the
  searched identifying strings. Rewriting history is the owner's decision.
- Nothing here has been validated from a clean machine, and no phone has been measured (Phase 22).

**Git:** branch `phase/21-research-package`, merged through pull request #27. Merge commit `420e095`,
tag `phase-21` (merge re-created with the noreply identity; same tree and parents).

**CI status:** green on the branch, the pull request and `main` (run 44343507005).

---

## Phase 22 — Final validation

**Status:** in progress (plan registered 2026-11-18). Nothing in this plan has been run: no test file
has been read for Phase 22, nothing has been downloaded and no number below is a Phase 22 result.
Opened on `phase/22-final-validation` from `e8e0b55`, the Phase 21 completion record on top of `main`
at `420e095`. New scripts and flags named here are proposals and do not exist yet.

**Objective:** close the four tasks of `docs/IMPLEMENTATION_PLAN.md`, Phase 22, without changing any
v1 record:

1. a clean-machine-style walkthrough, with every manual step recorded in `docs/FINAL_VALIDATION.md`;
2. a reproduction of v1 at the tolerance fixed before Phase 14 (Phase 14 item 6): re-scoring the saved
   files gives identical numbers, and a re-run of the frozen model gives every aggregate metric within
   0.05 points;
3. one evaluation per test set of the fused-attention phone candidate, as a new version under its own
   IDs, with its results beside v1's and replacing none of them;
4. the phone measurement on a named device, or a record that the gap stays open.

**Known when this was written.** v1's Phase 14 results on every test set (GW-REC-039, GW-E2E-007,
GW-E2E-009). For the fused-attention build, the validation and laptop results of GW-EFF-009 only
(`reports/efficiency/phase16/pareto.md`). No candidate build has read a test set, no phone has been
measured, and whether the stock Android runtime holds the `com.microsoft.MultiHeadAttention` kernel
has not been checked.

**Owner decisions (2026-11-18).** All as the plan recommended.

| # | Decision |
|---|---|
| 1 | The walkthrough downloads are approved: HierText 3.72 GiB, MobileNetV3 weights 31 MiB, FUNSD 16 MiB, IIIT5K 101 MiB, packages 2.60 GiB, the clone 39 MiB. For Android, the `onnxruntime-android` package and the build tools. The other registered datasets are not fetched. |
| 2 | Downloading, hashing and converting test files in the walkthrough is preprocessing, written into EXPERIMENT_PROTOCOL §9 first. |
| 3 | v1 is backed up: a verified private backup exists. |
| 4 | v1 is reproduced through a detached `phase14-eval` worktree and an external wrapper. |
| 5 | "Every aggregate metric" means all point values in the GLYPHARIS blocks, including small slices, calibration and official scores. Confidence bounds are shown but not judged. |
| 6 | If OmniDocBench's official number and its timeout-fallback pages differ, "identical" is recorded as failed, with the cause. The evaluator is not changed. |
| 7 | Pass B covers all official evaluators. |
| 8 | Two phones are available: Samsung Galaxy M52 5G first, Moto G35 5G second. Each device's SoC, RAM and Android build are read from the device and registered before measuring. The owner enables USB debugging when asked. Validation tensors may be stored on the phone temporarily. |
| 9 | The Android operator check runs early, on a phone rather than an emulator, before the candidate's selection and evaluation work. |
| 10 | If the stock Android package lacks the `MultiHeadAttention` kernel, the candidate's test evaluation stops and the reason is recorded. |
| 11 | Fused attention with the INT8 recognizer is measured on validation, and adopted only under the selection rule. |
| 12 | The one-line-at-a-time batching arm is included, and adopted only under the selection rule. |
| 13 | H11 is accepted: six tests, all required, margin 0.05 points (0.5 points only for an INT8 candidate). |
| 14 | Candidate test scope: all Phase 14 sets, all 755 OmniDocBench pages, no oracle. |
| 15 | Order: the operator check early, then sittings S1, S2, S3 and onward. Every long sitting waits for the owner's go-ahead and a PC restart. |

**Owner confirmation (2026-11-18), after the review of this plan and before it was committed:** the
owner kept H11's source-document check on GDC and the 0.05-point margin, knowing that margin is about
four lines there. The candidate work continues whichever way the v1 reproduction goes. If `fused` fails
the phone's logit or text check, part b is recorded as failed, no other arm is selected, `fused` stays
the laptop candidate and the phone gap stays open. If the operator check fails, items 14 to 20 stay
unstarted and their IDs and H11 close as `not run`.

**Rules added while reviewing this plan (2026-11-18); the owner confirmed those above.** Each applies unless the
owner changes it before the step it governs: a failed operator check closes items 14 to 20 (item 3);
the candidate work does not depend on the reproduction verdicts (item 13); if `fused` fails the phone
check it stays the laptop candidate and no other arm can be selected (item 15); H11's GDC tests also
carry the source-document check of H1′ and H5′ (`research/hypotheses.md`).

**Fixed rules**

- Test data is read only through logged paths. Reasons and IDs are written before the first read.
- The shipped v1 bundle is **not** run on test data. Test results stay the GPU strict-fp32 results, and
  bundle equality stays validation-only (`docs/IMPLEMENTATION_PLAN.md`, Phase 22 task 2).
- No edits to `glypharis/eval/`, `scripts/ext_eval.py`, `scripts/phase14_*.py`,
  `scripts/evaluate_baselines.py`, `scripts/official_*.py`, `scripts/hiertext_eval_launch.py`,
  `scripts/run_baselines.py`, or the `V1_PATHS` of `glypharis/efficiency/v1.py`.
- No per-line test text is viewed or written to a tracked file. GDC-scan contributes numbers only.
- No test set and no GDC-scan content goes on a phone.
- No latency claim comes from a test run.
- A failed check is reported as failed. No threshold, setting or metric changes after a result is seen.
- No personal paths, names or emails in tracked files. Phone records hold the model and build, never a
  serial number or account.
- Commits follow `docs/WORKFLOW.md` §2: the owner's identity, no tool or co-author trailers. The owner
  merges the pull request.

**Experiments** (full entries in `research/experiments.md`, all `planned`), in the order of work. The
v1 re-run keeps the old folder names because the scorer hard-codes them. Log rows, run metadata and
`v1_reproduction.*` carry the new IDs. Pass B's raw scorer report keeps the folder IDs and says so
(item 6), and documents cite only the mapped report.

| ID | What | Data | Test access |
|---|---|---|---|
| GW-EFF-015 | Part a: Android operator check on the phones. Part b: phone measurement (`not run` if impossible) | validation tensors | no |
| GW-E2E-013 | Walkthrough: the installed package on the v1 bundle against GW-E2E-006 | 400 validation pages | no |
| GW-REC-042 | v1 re-run, rehearsal (folder `GW-REC-038`) | HierText validation crops | no |
| GW-E2E-014 | v1 re-run, rehearsal (folder `GW-E2E-006`) | HierText validation pages | no |
| GW-E2E-015 | Pass A: re-score the saved Phase 14 files | all Phase 14 files | scoring rows |
| GW-REC-043 | v1 re-run on test crops (folder `GW-REC-039`) | HierText test, GDC, GDC-scan, FUNSD, IIIT5K | yes |
| GW-E2E-016 | v1 re-run on test pages (folder `GW-E2E-007`) | HierText test, GDC, GDC-scan, FUNSD, OmniDocBench | yes |
| GW-E2E-017 | v1 re-run, oracle (folder `GW-E2E-008`) | GDC, GDC-scan | yes |
| GW-E2E-018 | Pass B: score the re-run; tolerance verdict | the re-run and the baselines | scoring rows |
| GW-EFF-013 | Candidate selection | validation, laptop and first phone | no |
| GW-EFF-014 | Candidate bundle and equivalence | validation | no |
| GW-REC-044 | Candidate validation gate, crops | 74,500 crops | no |
| GW-E2E-019 | Candidate validation gate, pages | 1,715 pages | no |
| GW-REC-045 | Candidate test crops | HierText test, GDC, GDC-scan, FUNSD, IIIT5K | yes |
| GW-E2E-020 | Candidate test pages | HierText test, GDC, GDC-scan, FUNSD, OmniDocBench | yes |
| GW-E2E-021 | Candidate scoring pass, H11 | candidate and v1 files | scoring rows |

**Work items**

**A. Protect and register**

**1. Backup of v1.** A verified private backup exists. No worktree is removed.

**2. Register this plan (this commit).**
- this section and the status row;
- `research/experiments.md`: index rows and entries for every ID above;
- `research/hypotheses.md`: H11 and the selection rule for the arms;
- `docs/EXPERIMENT_PROTOCOL.md` §9 and §11: the reproduction re-reads test files but is not a new
  evaluation and cannot change v1; walkthrough downloading, hashing and converting of test files is
  preprocessing; the test re-run is done once per set; the candidate gets one prediction pass and one
  scoring pass; the count-only line comparison of item 12; preflight checks and hash lists add no row;
- a dated note in ADR-0011, and the `docs/FINAL_VALIDATION.md` skeleton;
- `EXPERIMENTS.md`, the experiment summary and the number trace regenerated.

*Pass rule:* `experiment_summary.py --check`, `experiment_comparison.py --check` and
`trace_numbers.py --check` pass; pushed; CI green.
*Stop if:* any of these fails. Nothing below starts until they pass.

**B. Android operator check (early, before any candidate work)**

**3. Android operator go/no-go, GW-EFF-015 part a (about 4 h of session work).** Build
`mobile/android-bench` (Kotlin, `onnxruntime-android:1.31.1`, the version in `requirements/lock.txt`).

Before the first build:
- commit `mobile/android-bench/.gitignore` covering `.gradle/`, `.kotlin/`, `.idea/`, `build/`,
  `captures/`, `local.properties`, `*.apk`, `*.aab` and any tensor or model file, because untracked
  build files would make `pinned_code_check` refuse the later runs;
- give the JDK and SDK through `JAVA_HOME` (JDK 21) and `ANDROID_HOME`, set for the build command only,
  and use a pinned Gradle wrapper. The JDK, Gradle and Android Gradle Plugin versions go in
  `docs/FINAL_VALIDATION.md`. No machine path is committed;
- confirm that `adb devices` lists each phone as authorized. If the Moto is not listed, the owner decides
  whether to install its driver; until then it is recorded as not connected and the M52 work continues.

On the Samsung Galaxy M52 5G first and then the Moto G35 5G, with USB debugging enabled by the owner
when asked:
- read the SoC, RAM, Android build and ABI from the device and register them in
  `docs/FINAL_VALIDATION.md`, with the package's SHA-256, before the check;
- create a session on `recognizer.fused.onnx` (GW-EFF-009, SHA-256 `b0e2a510…`, 31,810,737 bytes, 9
  `com.microsoft.MultiHeadAttention` nodes), with item 15's session options, and run one validation
  tensor.

No emulator is used. No model or data file enters the repository, and tensors are deleted from the
phone when the phone work ends. The result goes to
`reports/final_validation/GW-EFF-015/operator_check.json`.

*Pass rule (go):* `git status --porcelain --untracked-files=all` is empty straight after the build, and
on the phone the session is created and the tensor returns finite logits of the shape the laptop gives.
The largest logit difference from the laptop is recorded here and judged in item 15. If it exceeds
`1e-3`, or the greedy text differs, the owner is told before S4 starts; the formal verdict stays in item
15 and no threshold changes.
*Stop if (no-go):* the kernel is missing on the stock package. The candidate's test evaluation stops
and the reason is recorded (owner decision 10). Because this check comes first, items 14 to 20 have not
started; they stay unstarted, and their IDs and H11 are closed as `not run` with that reason, unless
the owner decides otherwise. A custom runtime build would be a different build, needing its own
registration. Items 4 to 13 go on, and the phone gap is recorded as still open.
If the first phone cannot be used for a reason other than the kernel, the second takes its place, and
that is recorded.

**C. Task 1: walkthrough**

**4. Run from a fresh public clone at `420e095` (sitting S1).** `$CLEAN` is a clean folder outside the
worktrees, with no links to existing data; `$REPO_URL` is the public repository.

```bash
mkdir -p $CLEAN/models $CLEAN/ref $CLEAN/evidence $CLEAN/out
git clone $REPO_URL $CLEAN/repo; cd $CLEAN/repo
git checkout --detach 420e095
unset PYTHONPATH; export PIP_NO_CACHE_DIR=1; py -3.11 -m venv .venv; P=.venv/Scripts/python
$P -m pip install --upgrade pip
$P -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
$P -m pip install -r requirements/lock.txt; $P -m pip install -e . --no-deps; $P -m pip check
$P scripts/env_report.py --out "$(cygpath -m $CLEAN/evidence/env)"
$P -m pytest -q; $P scripts/validate_configs.py

$P scripts/download_datasets.py --dataset hiertext --dataset imagenet_backbones --dataset funsd --dataset iiit5k --dry-run
$P scripts/download_datasets.py --dataset hiertext --parallel 3
$P scripts/download_datasets.py --dataset imagenet_backbones --dataset funsd --dataset iiit5k
# keep the diff of the four raw manifests: only downloaded_utc may change
git checkout -- data/manifests data/DATASET_MANIFEST.yaml
$P scripts/download_datasets.py --dataset hiertext --dataset imagenet_backbones --dataset funsd --dataset iiit5k --verify

$P scripts/prepare_hiertext.py --workers 12 --reports "$(cygpath -m $CLEAN/evidence/reports)"
git diff --exit-code -- data/manifests/processed/hiertext.json
$P scripts/build_eval_crops.py --eval-set hiertext_val --expect-count 74500
$P scripts/prepare_benchmarks.py funsd; $P scripts/prepare_benchmarks.py iiit5k
$P scripts/build_eval_crops.py --eval-set funsd_test --expect-count 2647
git diff --exit-code -- data/manifests/processed/benchmarks.json data/manifests/processed/eval_crops.json
```

Then, as logged manual steps, copy the bundle to `$CLEAN/models/` and
`data/processed/phase14/GW-E2E-006/hiertext_val.jsonl` (SHA-256 `c8520576…`) to `$CLEAN/ref/`. Set
`GLYPHARIS_MODELS`, run `glypharis-ocr` on one validation page with both backends from outside the
repository, and evaluate:

```bash
$P scripts/package_check.py --config configs/det/gw-det-001.yaml --model glypharis-v1 \
  --backend torch --threads 10 --v1 --phase10 "$(cygpath -m $CLEAN/ref/hiertext_val.jsonl)" \
  --reference-sha256 c8520576f44f8b2ee436f9103556556b2f5e6685379b95bee3ef2403eb73eb4b \
  --expect-manifest-sha256 875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037 \
  --experiment GW-E2E-013 --out reports/package/GW-E2E-013.json
```

*Pass rule:* all of these hold: every raw file matches its committed hash; no derived manifest
differs, apart from the expected `iiit5k_test` line below, which item 5 clears; `package_check` exits 0
(page CER difference at most 0.05 points, manifest hash matched, `--v1` check passed).
*Stop if:* an upstream hash differs. That is a finding and the manifest is not updated. Any retry,
deviation or hand action becomes a row in `docs/FINAL_VALIDATION.md`.
*Expected findings:*
- `pytest` in this clone will fail at the scorer test fixture, because `hash-phase5` refuses once the
  `phase14-eval` tag exists. Phase 14 recorded this as a known issue. The failing test names are
  recorded.
- `prepare_benchmarks.py iiit5k` does not write `index_sha256`, so `benchmarks.json` will differ by that
  one line. The failing diff is recorded, and the steps after it go on.

**5. Fix what the walkthrough finds (about 1 h).**
- Add one document giving the order clone, install, download, prepare, obtain model, infer, evaluate.
  No current document gives it.
- Patch `tag_exists` in the scorer test helper; the scorer itself stays unedited. Add a test.
- Make `prepare_iiit5k` in `scripts/prepare_benchmarks.py` return `index_sha256` (the SHA-256 of
  `words_test.index.jsonl`) directly after `bin_sha256`. Add a test.
- Fetch the fixed commit into the clean clone and re-run only the affected steps, including
  `prepare_benchmarks.py iiit5k` followed by
  `git diff --exit-code -- data/manifests/processed/benchmarks.json`.

Write up the walkthrough; the clean folder is deleted only with the owner's agreement. The write-up
says that this is clean-machine-style, with the same OS, driver, base Python and git config, and that
the weights came by hand, so the weight download path is untested until Phase 23.

*Pass rule:* the affected steps pass in the clone, and the write-up is committed.
*Stop if:* a fix would need a change under the paths the Fixed rules protect.

**D. Task 2: reproduce v1**

**6. Wrapper, comparer and tests (about 3 h).** Add `scripts/phase22_v1_repro.py` and
`scripts/phase22_v1_compare.py`, and generate `reports/final_validation/metric_inventory.json` from
`reports/external/results.json` before any re-run.

*`rerun` mode* (working directory in the tag worktree, code imported from there):
- assert that `ext_eval` and `glypharis` import from the tag worktree;
- set `ext_eval.PHASE14_BRANCH = "HEAD"` and an absolute `ext_eval.ACCESS_LOG`;
- wrap `ext_eval.open_test_access` to write phase 22 and the new ID;
- wrap `ext_eval._write_json` so metadata says phase 22 and records the wrapper commit;
- refuse a test set if the access log already holds a prediction row for that benchmark, split and ID,
  unless the set's output exists and is incomplete (a resume), because `ext_eval` starts afresh once an
  output has been moved aside;
- leave `pinned_code_check` active.

*`rescore` mode* (working directory in the Phase 22 worktree):
- set `ps.PHASE14_BRANCH`, `ps.EXPERIMENT`, `ps.REASON`;
- replace `ps.Scorer` with a subclass that sets only `ctx.phase` and `gctx.phase` to 22 (`ps.PHASE`
  stays 14 because it routes baseline paths);
- in Pass B only, wrap `eb.open_test_access` to map GW-REC-039 to 043, GW-E2E-007 to 016 and GW-E2E-008
  to 017;
- require `pinned_code_check("phase22-repro", root=worktree)` before scoring;
- `check` also asserts that every pack and page file the Phase 5 re-score needs exists, including
  `processed/glyphsynth/lines_val`;
- after Pass B's scoring, add to its `results.json` and each Markdown report an `id_map` block and a
  header line saying these are re-run values: GW-REC-039 = GW-REC-043, GW-E2E-007 = GW-E2E-016,
  GW-E2E-008 = GW-E2E-017. The unchanged scorer labels them with the folder IDs.

*`phase22_v1_compare.py`:*
- verifies the Phase 14 files against `data/manifests/processed/phase14_outputs.json` before it reads
  them, and stops on any difference;
- `rehearsal` scores only GW-REC-038 and GW-E2E-006 under `$ROOTS/rerun`, with
  `eb.score_file(path, ps.FullContext(data, log, 22, reason), meta=ps.glypharis_meta(job))`, writes no
  log row because the split is validation, and compares every point value with the same two entries in
  `reports/external/results.json` at 0.0005. The scorer's own preflight cannot do this, because it
  requires every test prediction file.

From the `phase22-repro` tag until Pass B has finished, only `RUN_RECORDS` may change. Rehearsal
results, Pass A verdicts and any owner ruling on a disclosed difference go under
`reports/final_validation/`; `docs/` and `research/experiments.md` are updated after Pass B. A code fix
needs a new tag and a recorded reason.

*Pass rule:* tests on made-up data show that rows say phase 22 with the right IDs; nothing is written
under the Phase 14 paths; imports outside the tag worktree are refused; a second attempt at a test set
is refused and a resume is allowed; a Phase 14 file that differs from its recorded hash stops the
comparer; `rehearsal` scores the two validation entries and logs nothing; Pass B's files carry the ID
map and header; a 0.0004 change passes and a 0.0006 change fails; a missing metric fails; provenance is
ignored; a planted GDC-scan marker reaches no tracked output. Commit, push, CI green, then tag
`phase22-repro`.
*Stop if:* the wrapper cannot work without editing a protected file.

**7. Isolated roots (about 15 min).** `$TAG_WT` is a detached worktree beside the others, and `$ROOTS`
a folder outside every worktree.

```bash
git worktree add --detach $TAG_WT phase14-eval
# copy the two checkpoints to their recorded paths in that worktree
```

Under `$ROOTS/rescore` and `$ROOTS/rerun`:
- `manifests/` is copied from the Phase 22 worktree. `git diff phase14-eval HEAD -- data/manifests`
  lists only `phase14_outputs.json`.
- In `rescore`, drop the hand-off entries from `phase14_outputs.json`. In `rerun`, drop every
  `data/processed/phase14/**` entry, so baseline hashes are still checked.
- Junctions: `raw` and `processed/{hiertext,gdc,gdc_scan,funsd,iiit5k,omnidocbench,glyphsynth,baselines,baselines_phase14}`.
- `rescore` also gets junctions for
  `processed/phase14/{GW-REC-038,GW-REC-039,GW-E2E-006,GW-E2E-007,GW-E2E-008}` inside a real
  `processed/phase14`.
- `rerun` gets an empty real `processed/phase14`.

*Pass rule:* the manifest difference is as stated, and the first hash list under "How v1's records stay
untouched" has been taken, written to `reports/final_validation/v1_files_before.json` and committed
before item 8, after a check that its paths identify no GDC-scan content.
*Stop if:* `processed/phase14` itself, `handoff`, `official` or `GW-E2E-009` would be junctioned:
`score --resume` runs `shutil.rmtree` on hand-off and official folders under its data root.

**8. Validation rehearsal on the GPU, GW-REC-042 and GW-E2E-014 (sitting S2).** `$PY` is the project
environment's interpreter and `$P22` the Phase 22 worktree.

```bash
cd $TAG_WT
PYTHONPATH='.;scripts' $PY -P $P22/scripts/phase22_v1_repro.py \
  rerun rec --set hiertext_val --folder-id GW-REC-038 --experiment GW-REC-042 \
  --reason "Phase 22 v1 re-run rehearsal (GW-REC-042)" --data-root $ROOTS/rerun
# then: rerun e2e --set hiertext_val --folder-id GW-E2E-006 --experiment GW-E2E-014 ...
```

Verify the Phase 14 validation files against `data/manifests/processed/phase14_outputs.json`; the
comparison stops on any difference. Then run `phase22_v1_compare.py rehearsal` (item 6), ignoring
`time_ms`, and record every differing line count. No log rows, because this is validation.
*Pass rule:* every point value is within 0.0005 of Phase 14's.
*Stop if:* any value differs by more. No test file is read.

**9. Pass A, GW-E2E-015 (sitting S2).**

```bash
cd $P22
PYTHONPATH='.;scripts' $PY -P scripts/phase22_v1_repro.py \
  rescore check --data-root $ROOTS/rescore
# then: rescore score --experiment GW-E2E-015 --out reports/final_validation/GW-E2E-015 ...
```

`rescore check` hashes files, reads keys and converts predictions without ground truth. It adds no log
row (see the log plan).
*Pass rule:* `rescore check` passes before `score` starts, and the pass finishes with its rows logged.
*Stop if:* a check fails. A stopped pass resumes as the same evaluation, its rows marked `(resumed)`.

**10. Pass A verdict.** Compare the output with `reports/external/results.json`, and the OmniDocBench
fallback page lists separately. Commit the log and the report.
*Pass rule:* every number outside provenance and path strings is exactly equal.
*Stop if:* a difference is unexplained. If OmniDocBench's official number differs and its fallback
pages differ too, "identical" is recorded as failed with that cause, and the evaluator is not changed
(owner decision 6). The re-run waits for the owner's ruling on any disclosed difference.

**11. Test re-run on the GPU, GW-REC-043, GW-E2E-016, GW-E2E-017 (sitting S3).** Twelve runs in Phase
14's order: rec on `hiertext_test`, `gdc_test`, `gdc_scan_test`, `funsd_test`, `iiit5k_test`; oracle on
`gdc_test`, `gdc_scan_test`; e2e on `gdc_scan_test`, `funsd_test`, `gdc_test`, `omnidocbench_test`,
`hiertext_test`, the order Phase 14's run records show (corrected 2026-11-19, see the amendment after
item 13). Same device, batch 256, same tag. Commit the log after the batch.

The test re-run is done once per set. A resume only finishes a crashed run. A failed tolerance is not
answered by running again; any further attempt needs a new registered ID, written before it starts, and
is reported beside the first.

*Pass rule:* all 12 outputs are complete and their hashes are in a tracked report.
*Stop if:* a guard refuses (branch, tag, pinned code, input hash, image size). A crash resumes with a
logged resume row.

**12. Pass B, GW-E2E-018 (sitting S3).** The same unchanged scorer on the re-run root, with all
official evaluators and `--id-map rerun`. Then a count-only comparison of re-run and Phase 14
predictions: how many crops and pages differ, with no text output. The Phase 14 files are first verified
against `data/manifests/processed/phase14_outputs.json`, and the comparison stops on any difference.
*Pass rule:* the pass finishes, every row carries a re-run or baseline ID, Pass B's files carry the ID
map, and the comparison writes counts only.
*Stop if:* scoring would overlap a GPU run: `open_test_access` rewrites the whole log file.

**13. Tolerance verdict.** Judge every path in the metric inventory. Write
`reports/final_validation/v1_reproduction.json` and `.md`, listing the largest differences and any
changed verdict.
*Pass rule:* every point value in the GLYPHARIS blocks, including small slices, calibration and official
scores, differs by at most `0.0005` as a fraction, and counts are equal. Confidence bounds are shown
and not judged.
*If it fails:* it is reported as failed, beside Phase 14's results, which stay as recorded. The exit
criterion is then unmet and the phase stays open for the owner.

Items 14 to 20 do not depend on the verdicts of items 10 and 13. H11 reads the Phase 14 files, which item
20 checks by hash, so the candidate work continues whichever way those verdicts go, and neither verdict
can add, remove or reorder a candidate step.

**E. Tasks 3 and 4: phone candidate**

**14. Candidate code and selection, GW-EFF-013 (sitting S4, closed by the phone part of S5).** New
`glypharis/efficiency/candidate.py` and `scripts/phase22_candidate.py`, outside `V1_PATHS`. No arm is
added to `VARIANTS`, because `SITTINGS` derives from it and Phase 16's records depend on it.

Arms:
- `fused`, the default: GW-EFF-009's `fused-attention`;
- `fused-int8`: the fused recognizer with H7′'s dynamic INT8 settings;
- `fused-single-line`: the fused recognizer, one line per recognizer run.

On the laptop, accuracy on the 400 Phase 5 pages and the 5,000 seeded lines, then latency and RAM on 10
warm-up plus 100 items, mirrored order, one sitting, Phase 16 conditions. Each arm reads the 5,000 lines
under the batching rule its bundle would ship: `fused` and `fused-int8` through the recognizer's
width-sorted batches with 256 crops per call, `fused-single-line` one line per recognizer run. (Phase
16's evaluator reads every line at batch 1, where `fused-single-line` would equal `fused` by
construction.) The laptop's latency and RAM are recorded and decide nothing while a phone can be used.

*Selection rule, fixed now* (exact statistics in `research/hypotheses.md`, under H11). Stay with `fused`
unless another arm meets all three conditions:
- accuracy: non-inferior to `fused` on line CER, page CER and E2E F1, with margin 0.005 for
  `fused-int8` and 0.0005 for `fused-single-line`;
- speed: the CI of `mean(t_arm - 0.9 * t_fused)` per page lies below 0 on the first phone, over both
  runs and on each arm's first and second runs taken alone;
- memory: peak RAM not higher, on the same phone, with each arm and run in a new process.

An arm that fails the accuracy condition does not go to the phone. If two arms qualify, the one with the
larger page-time saving is taken. Exactly one candidate leaves this item, and only after the phone part
of item 15.
*Stop if:* `recognizer.fused.onnx` differs from the hash GW-EFF-009 recorded. The laptop times serve for
the speed and memory conditions only if neither phone can be used, which is recorded.

**15. Phone measurement, GW-EFF-015 part b (sitting S5; length unknown, planned as up to 6 h for the
first phone).** Register in `docs/FINAL_VALIDATION.md`, and commit, before the first timed run: model,
SoC, RAM, Android build, package hash, ABI, session options, thread count, spinning setting, charge
state, starting battery temperature, airplane mode, screen and foreground state.

Session options are those of `ort_session` (`glypharis/efficiency/threads.py`), so that phone figures
describe the configuration measured on the laptop: sequential execution, all graph optimisations, CPU
arena off, memory patterns off, and the registered intra-op thread count and spinning value, read back
or logged by the app.

Replay the laptop's exact validation network inputs: 10 plus 100 Phase 5 profile pages, then 1,000
seeded lines at batch 1. Each arm and run is a new app process, force-stopped between runs. A timing run
loads one input tensor at a time, times only `session.run`, reads `VmHWM` immediately after the last
timed item and holds no reference logits. The logit and greedy-text check is a separate pass in its own
process and contributes no time or memory figure. Report mean and p95 network time, peak `VmHWM`, each
run's mean time, and thermal state at start and end of each run.

- *Samsung Galaxy M52 5G:* `fused` and every arm that passed item 14's accuracy condition, in mirrored
  order. These runs decide item 14's speed and memory conditions.
- *Moto G35 5G:* the selected candidate, after selection. This is a second named device and changes no
  decision.

*Pass rule:* for each arm measured, the session loads, logits are within `1e-3` of the laptop's for the
same arm, and greedy text is identical. An INT8 arm's differences are reported, and the selection margin
decides.
*Stop if:* `fused` fails the logit or text check. Part b is then recorded as failed, its phone times are
not reported as the candidate's, no other arm can be selected, `fused` remains the candidate for the
laptop evaluation, and the phone gap stays open. Another arm that fails to load or fails the check drops
out.

The report says plainly that this is network time and memory only: decoding, post-processing, cropping
and reading order are not ported. Tensors are deleted from the phones afterwards.

**16. Freeze the bundle, GW-EFF-014 (about 2 h).** New format `glypharis-phone-bundle-1`: `detector.onnx`
byte-identical to v1's, the chosen recognizer, `charset.json`, and a manifest of hashes and every
setting the loader needs (detector spec, height 48, maximum width 1536, stride 4, class count,
recognizer arch, T = 2.35, opset, ONNX Runtime version, session options as in item 15, batching rule, v1
manifest hash, and a version label saying not v1). The loader lives outside `from_pretrained`. `load_bundle_onnx` cannot be reused,
because it needs the `.pt` files. A test asserts that `load_bundle` refuses this format.

*Pass rule:* Phase 17's tolerances against the GW-EFF-006 references: map difference at most `1e-4`,
logits at most `1e-3`, text identical except near ties on at most 0.1% of lines, page CER within 0.05
points. If `fused-int8` was selected, the logit and text checks are reported and the selection margin
decides.
*Stop if:* a tolerance fails. The bundle is not changed to pass, no validation gate runs and no other
arm is tried. The IDs of the steps not reached, and H11, close as `not run` with the reason.

**17. Runner, scorer and tag (about 8 h).**

*`scripts/phase22_eval.py`:*
- guarded to the phase branch; `pinned_code_check("phase22-candidate-eval")`, `assert_v1_code`, bundle
  hash, `check_inputs`, `image_sizes_match`;
- before logging a test access, reads `reports/final_validation/GW-EFF-015/operator_check.json` and
  `reports/final_validation/candidate/validation_gate.json`, and refuses unless the check is `go`, all
  three gate tests are `met`, and the gate's bundle hash and tag commit equal this run's;
- refuses a test set if the access log already holds a prediction row for that benchmark, split and ID,
  unless the set's output exists, is incomplete and has the same identity (a resume). Moving or deleting
  an output cannot start a second pass;
- `open_test_access(..., 22, ...)` before reading; reuses `ext_eval.set_records`, `_rec_records` and
  `_page_records` through a small adapter exposing `.ocr` and `.recognizer`, with crops read under the
  bundle's batching rule; resumable; writes predictions only, to `data/processed/phase22/<ID>/`.

*`scripts/phase22_score.py`:*
- verifies v1's files against `phase14_outputs.json` first; reuses `ps.Scorer.score_phase14`,
  `official_hiertext`, `official_omnidocbench`, `rec_units` and `page_units`, with every `Scorer` logging
  phase 22;
- keeps `data/processed/phase22/GW-E2E-021/pass.json` under the rules of `ps.start_pass`: a finished pass
  or an existing `results.json` is refused, and a started pass needs `--resume`;
- scores the candidate's 10 files in one `Scorer`, with `ps.GLYPHARIS_VERSION` set to the bundle's not-v1
  label, so that no candidate entry says v1. The official methods only process files scored in that
  instance, so neither v1 nor the baselines reach the official evaluators;
- puts the 4 v1 files H11 needs (GW-REC-039 and GW-E2E-007 on HierText test and GDC) through
  `eb.score_file` in a second `Scorer`, under v1's own label, so each is logged. v1's other figures are
  copied from `reports/external/results.json`, and its other test files are not read.

*Isolation.* Candidate work files go under `data/processed/phase22/GW-E2E-021/`. The runner and scorer do
not call `ext_eval.record_output`, `ps.run_score`, `ps.preflight`, `ps.dry_convert`, `ps.record_outputs`
or `ps.drop_handoff_outputs`. The SHA-256 and line counts of the 10 prediction files and their metadata
go to `reports/final_validation/candidate/prediction_hashes.json`, committed before scoring.

After the tag, only `RUN_RECORDS` may change, and of those Phase 22 uses the access log,
`research/failures.md` and `reports/`. The shared check also allows `phase14_outputs.json`; that allowance
is not used, and the file is not written in Phase 22. Results are written into `docs/` after scoring.

*Pass rule:* tests pass, H11 and the set list are committed, CI is green, and the commit is tagged
`phase22-candidate-eval`. The tests show each refusal above: a no-go operator check, an unmet or
mismatched gate, a second prediction pass after an output is moved or deleted, a second scoring pass and
an unmarked resume. On made-up data they also show that nothing changes under `processed/phase14`, the
baseline folders or `phase14_outputs.json`; that no candidate entry or report says v1; and that a marker
planted in made-up GDC-scan references, predictions and file names appears in no report, log row, console
output or other tracked file written by `phase22_eval.py` or `phase22_score.py`.
*Stop if:* the runner or scorer would need a change to a protected file.

**18. Full validation gate, GW-REC-044 and GW-E2E-019 (sitting S6).** The same runner and scorer at the
tag, paired against GW-REC-038 and GW-E2E-006, using H11's rules on line CER, bag-of-lines page CER and
E2E F1 on HierText validation. The result, with the bundle hash and tag commit, goes to
`reports/final_validation/candidate/validation_gate.json`.
*Pass rule:* all three are `met`.
*Stop if:* any is not met. No test set is opened and no other arm is tried. The IDs of the steps not
reached, and H11, close as `not run` with the reason.

**19. Test evaluation, GW-REC-045 and GW-E2E-020 (sitting S7).** One pass per set, 10 threads, on the
CPU: crops on HierText test, GDC, GDC-scan, FUNSD and IIIT5K; pages on HierText test, GDC, GDC-scan,
FUNSD and all 755 OmniDocBench pages. The oracle is not run for the candidate, since it served only H4.
Commit the log after each batch.
*Pass rule:* all 10 prediction files are complete, and the SHA-256 and line counts of each file and its
metadata are in `reports/final_validation/candidate/prediction_hashes.json`.
*Stop if:* a guard refuses. A crash resumes and logs a resume row. Nothing is changed and re-run after a
test file has been read.

**20. One scoring pass, GW-E2E-021 (sitting S8).** Both tracks, charset ceiling, the Phase 14 metrics
for each set, official HierText and OmniDocBench scores for the candidate only, calibration at T = 2.35
unchanged, and H11, including its source-document check on GDC. Fourteen logged rows: the candidate's 10
files and v1's 4 (item 17). Reports go under `reports/final_validation/candidate/`, beside v1's.
*Rule:* H11 as registered in `research/hypotheses.md`: six tests, all required. Every other set and
metric is descriptive. If H11 does not hold, the candidate is not shown non-inferior; a fix would be
another new version and is outside this phase.
*Stop if:* v1's files do not match `phase14_outputs.json`.

**F. Close**

**21. Records and gate (about 3 h).**
- Check the records of v1 (below).
- `docs/FINAL_VALIDATION.md`, `MODEL_CARD.md` "Not v1", README and paper status sentences, ROADMAP
  "Reproducible final experiment", ADR-0011, R-23, `research/failures.md` and the session log, all from
  generated reports, with `trace_numbers.py --check` passing.
- WORKFLOW's ten-step gate, a push, the owner's merge, then tag `phase-22`.
- Junctions are removed with `cmd //c rmdir` before any root or the tag worktree is deleted.

**Order of work**

| Step | Items | Sitting |
|---|---|---|
| 1 | 1, 2: backup, registration, CI green | session work |
| 2 | 3: Android operator check on the phones | session work, phones attached |
| 3 | 4, 5: walkthrough and its fixes | S1, then session work |
| 4 | 6, 7: v1 tooling, tag, isolated roots | session work |
| 5 | 8, 9, 10: rehearsal, Pass A, verdict | S2 |
| 6 | 11, 12, 13: test re-run, Pass B, tolerance verdict | S3 |
| 7 | 14: candidate code, laptop part of selection | session work, S4 |
| 8 | 15: phones; selection closes | S5 |
| 9 | 16, 17: bundle, runner, scorer, tag | session work |
| 10 | 18: validation gate | S6 |
| 11 | 19: candidate test runs | S7 |
| 12 | 20: candidate scoring | S8 |
| 13 | 21: records and gate | session work |

**What must be committed first**

| Before | Must be committed, pushed and passed |
|---|---|
| Android operator check | Item 2; CI green; `mobile/android-bench/.gitignore`; USB debugging enabled by the owner |
| Walkthrough downloads | Item 2; operator check recorded; PC restart |
| First logged test access in the phase (Pass A) | Walkthrough write-up; wrapper, comparer, metric inventory and tests; CI green; tag `phase22-repro`; `v1_files_before.json`; rehearsal passed; `rescore check` passed; owner go; restart |
| v1 test re-run | Pass A report and access log committed; exact check passed, or the owner has ruled on a disclosed difference |
| Pass B | All 12 re-run outputs complete; hashes in a tracked report; log committed |
| Candidate selection | Operator check `go` recorded; H11 and the selection rule committed. The verdicts of items 10 and 13 are not a condition, whichever way they went |
| Phone timing runs | Device records, session options and process rule committed for that phone |
| Candidate validation gate | Selection report, including the phone part; bundle built, hashed and checked; runner and scorer tested; CI green; tag `phase22-candidate-eval` |
| Candidate test runs | `validation_gate.json` with all three tests met; phone work finished wherever it could change the candidate; owner go; restart |
| Candidate scoring | All 10 prediction files complete; `prediction_hashes.json` and the log committed |

Scoring and GPU runs never overlap, because `open_test_access` rewrites the whole log file.

**Test-access log plan** (`research/test_access_log.md`)

| When | ID in the row | Rows | Phase | Reason text |
|---|---|---|---|---|
| Android check and phone runs | none (no test data) | 0 | — | — |
| Walkthrough download and conversion | none (preprocessing, EXPERIMENT_PROTOCOL §9) | 0 | — | — |
| Validation runs and gates | none (`open_test_access` logs test splits only) | 0 | — | — |
| Preflight checks and the v1 hash lists | none: they hash files, read keys, or convert predictions without ground truth; no model runs, no score is computed and no text is shown, as in Phase 14 | 0 | — | — |
| Pass A | the producing IDs | 70 | 22 | `Phase 22 v1 re-score (GW-E2E-015)` |
| v1 test re-run | GW-REC-043, GW-E2E-016, GW-E2E-017 | 12 | 22 | one reason per run, before the images are read |
| Pass B | re-run IDs and baseline IDs | 70 | 22 | `Phase 22 v1 re-run scoring (GW-E2E-018)` |
| Count-only comparison | GW-E2E-018 | 12 | 22 | `Phase 22 v1 line comparison, counts only (GW-E2E-018)` |
| Candidate test runs | GW-REC-045, GW-E2E-020 | 10 | 22 | one per set |
| Candidate scoring | GW-REC-045, GW-E2E-020 (10 rows); GW-REC-039 and GW-E2E-007 on HierText test and GDC (4 rows) | 14 | 22 | `Phase 22 candidate scoring (GW-E2E-021)` |

188 rows in total, plus any `(resumed)` rows. The log is committed after each batch.

**Cost by step (estimates).** Every figure here is a planning estimate, not a result. Those for Pass A,
Pass B and the re-run are based on Phase 14's recorded durations, and those for the candidate on
GW-EFF-009's validation runs; OmniDocBench is the weakest of them, and the phone's run time is unknown.

| Items | CPU | GPU | Network | Session work | Download | Disk |
|---|---|---|---|---|---|---|
| 1–2 Backup, registration | — | — | — | 3 h | — | — |
| 3 Android check | — | — | setup | 4 h | package 55,138,794 bytes; the existing Android SDK is reused, and only missing Gradle or Maven artifacts are fetched | a few GB |
| 4 Walkthrough | about 1.2 h | smoke checks only | about 2.6 h | 2 to 3 h | 3.87 GiB data, 2.60 GiB packages, 39 MiB clone | about 15.5 GiB, temporary |
| 5 Fixes | 0.5 h | — | — | 1 h | — | — |
| 6–7 v1 tooling, roots | tests | — | — | 3.3 h | — | about 155 MB |
| 8 Rehearsal | — | 0.35 h | — | check | — | about 55 MB |
| 9–10 Pass A | 4.3 h | — | — | check | — | about 290 MB |
| 11 Test re-run | — | 0.9 h | — | check | — | about 105 MB |
| 12–13 Pass B and verdict | about 4.5 h | — | — | check | — | about 290 MB |
| 14 Selection | 2 to 3 h | — | — | 5 h | — | under 100 MB |
| 15 Phones | laptop packing | — | — | up to 6 h for the first phone; the second is not estimated | — | about 1 GB on laptop and phone |
| 16 Bundle | about 1 h | — | — | 2 h | — | 45 MB |
| 17 Runner, scorer | tests | — | — | 8 h | — | — |
| 18 Validation gate | about 3 h | — | — | check | — | about 55 MB |
| 19 Candidate test | 7 to 11 h | — | — | check | — | about 100 MB |
| 20 Scoring | 1.5 to 3 h | — | — | 1 h | — | tens of MB, plus official work files up to about 150 MB |
| 21 Close | checks | — | — | 3 h | — | frees temporaries if approved |
| **Total** | **about 25 to 32 h** | **about 1.3 h** | **about 2.6 h** | **about 40 h** | **about 6.5 GiB, plus Android** | well within the free disk |

**Long sittings.** Each one waits for the owner's go-ahead and a PC restart, and needs mains power, no
sleep and other applications closed. Phase 14 lost runs to a power-cut sleep and to the commit limit.

| Sitting | Contents | Length (estimate) |
|---|---|---|
| S1 | Walkthrough | about 5 h, about 2 h of it unattended download |
| S2 | Rehearsal, Pass A | about 4.7 h |
| S3 | v1 test re-run, Pass B | about 5.3 h |
| S4 | Candidate selection, laptop part | 2 to 3 h |
| S5 | Phones: the first with every remaining arm, the second with the selected candidate | up to 6 h for the first phone, attended; the second not estimated |
| S6 | Candidate validation gate | about 3 h |
| S7 | Candidate test runs | 7 to 11 h, overnight |
| S8 | Candidate scoring | 1.5 to 3 h |

S2 and S3 are not chained. S3 starts only after the Pass A verdict and log are committed, the owner has
given the go-ahead, and the PC has been restarted.

**How v1's records stay untouched**

- All re-run and re-score writes go to the two separate roots and to new folders under
  `reports/final_validation/`. The one exception is the OmniDocBench evaluator, which writes temporarily
  in its own `result/` folder under the shared raw store before its files are moved out. That folder's
  file list must be the same before and after each pass, candidate scoring included.
- Candidate writes go under `data/processed/phase22/` and `reports/final_validation/candidate/` only.
- Before the first run and at closure, hash every file under `data/processed/phase14`, `baselines` and
  `baselines_phase14`. `official/` and `logs/` are not covered by `phase14_outputs.json`, so the
  manifest alone is not enough. The first list is committed as
  `reports/final_validation/v1_files_before.json` (item 7), and the two lists must match.
- At closure, verify the Phase 14 outputs against `phase14_outputs.json`, and check that the access log
  at the start of Phase 22 is an exact prefix of the current log.
- `git diff --exit-code phase-21 HEAD --` must be empty for `reports/external`,
  `data/manifests/processed/phase14_outputs.json`, `data/manifests/processed/phase5_baseline_files.json`,
  `configs`, `glypharis/eval`, `glypharis/pipeline`, `glypharis/models`, `glypharis/export`,
  `glypharis/data`, and the pinned scripts of the Fixed rules.
- Candidate results never overwrite or edit a v1 number. A failed reproduction is reported beside Phase
  14's results, which stay as recorded.

**Exit criteria** (`docs/IMPLEMENTATION_PLAN.md`, Phase 22)

| Criterion | Items | Note |
|---|---|---|
| Walkthrough log in `docs/FINAL_VALIDATION.md`, every manual step recorded | 4, 5 | "Evaluate" is the validation package check; it is not a second machine |
| Re-scoring gives identical numbers with the scorer as it stands | 6, 7, 9, 10 | OmniDocBench's wall-clock limits can break exact equality (owner decision 6) |
| Re-run within 0.05 points, from hash-matching files with the same flags | 8, 11, 12, 13 | Passing is not guaranteed. If it fails, this criterion is unmet and the phase stays open for the owner |
| Candidate fixed from validation; bundle built and hashed; own IDs; sets, metrics and rules registered | 2, 3, 14, 16, 17, 18 | |
| Candidate results under its own IDs, once per test set, logged, beside v1's; or the reason it was not evaluated | 3, 19, 20 | |
| Phone gap recorded as measured on a named device or as still open | 3, 15 | Device, operating system, runtime build, threads, thermal and power conditions registered first |

**Still unverified**

- That the Android package contains the `MultiHeadAttention` CPU kernel. Only a session on an Android
  build settles this (item 3).
- That logits on an ARM phone agree with the laptop's within `1e-3`. That limit has only been used
  between runtimes on one x86 machine (items 3 and 15).
- That the Moto G35 5G appears in `adb` on this laptop without a further USB driver (item 3).
- That `quant_pre_process` succeeds on the fused graph, which `fused-int8` needs (item 14).
- The wrapper design. Its first real proof is the tests and the rehearsal.
- That every point value in the metric inventory is stored as a fraction, which the 0.0005 limit
  assumes. Item 6 checks each value's scale when it generates the inventory, before any re-run.
- Install and test-suite durations, phone run time and candidate test times.
- Whether fused attention with INT8, or one line per run, helps on the real model.
- Whether a re-run on the present GPU driver reproduces Phase 14's lines. The driver has changed since
  the environment report, and Phase 14's run metadata records none, so a driver effect could not be
  ruled out.

**Limits this plan cannot remove**

- GDC, GDC-scan and the weights cannot be re-downloaded by a stranger.
- The walkthrough shares the OS, driver and base Python with the original work.
- Phone pages stay validation-only, while document pages hold about twice as many lines.
- Phone figures are network time and memory, not a full page on the phone.
- ADR-0014's ImageNet exception was written for v1 and does not automatically cover a released candidate.

**Progress (2026-11-18 UTC): item 3 passed.** The Android operator check is go on both phones
(GW-EFF-015 part a, `reports/final_validation/GW-EFF-015/operator_check.json`), so the candidate work
stays open. It covers one validation line and says nothing about speed or memory.

**Progress: items 4 and 5 passed (GW-E2E-013).** The walkthrough from a fresh clone passed, and its three
findings were fixed at `8a4b3a3` and re-checked in the clean clone (`docs/FINAL_VALIDATION.md`, section 1).

**Amendment to items 6 to 13 (2026-11-19, before `phase22-repro` was tagged and before any reproduction
result existed).** The review of the reproduction tools settled these points:

- *Order of the page runs.* Item 11 first listed them in the wrong order. Phase 14's recorded start times
  give `gdc_scan_test`, `funsd_test`, `gdc_test`, `omnidocbench_test`, `hiertext_test`, and the wrapper
  enforces that.
- *Metric inventory.* `reports/final_validation/metric_inventory.json` lists 1,381 point values, 413
  counts, 22 flags and 16 verdicts from `reports/external/results.json`. As owner decision 5 says, every
  point value is judged, including small slices, calibration bins and H1' latency. Bin failures are
  counted separately in the verdict, so that one line crossing a bin edge can be seen for what it is, but
  they still fail the check.
- *Same flags.* The rehearsal and the verdict also fail if the run's identity differs from Phase 14's:
  model hashes, settings, batch, inputs, tag, or the torch, CUDA and cuDNN versions. The GPU driver is
  recorded but cannot be compared, because Phase 14 did not record it.
- *Pass A.* `phase22_v1_compare.py pass-a` makes the exact comparison of item 10. Strings that are paths
  are skipped; every other value must match, and OmniDocBench's fallback pages are compared row by row.
- *A crash.* A set whose records are all written but which never reached the manifest may be closed with
  no model load, image read or log row. A finished set can never run again.
- *Pass identity.* Each root belongs to one pass. Pass A refuses re-run files, Pass B requires all 14
  re-run files, and a stopped pass resumes only as itself.
- *After the tag,* only the run records listed in `RUN_RECORDS` may change until Pass B has finished, so
  this section is not edited again before then.

**Progress (2026-11-19 UTC): items 6 to 13 done; the reproduction verdict failed.**

- *Item 8 passed (GW-REC-042, GW-E2E-014).* The rehearsal reproduced both validation runs exactly:
  largest difference 0.0, every crop and page record identical (`reports/final_validation/rehearsal.json`).
- *Item 10 failed (GW-E2E-015).* Pass A is not identical: 40 official OmniDocBench values of Tesseract
  and GLYPHARIS differ, because more pages reached the evaluator's time limit, and none is unexplained.
  The owner ruled to continue under the rule unchanged and decide on acceptance at the end.
- *Item 11 passed (GW-REC-043, GW-E2E-016, GW-E2E-017).* All 12 test runs finished once, in Phase 14's
  order; their hashes and the access log were committed before scoring.
- *Items 12 and 13 (GW-E2E-018).* The predictions are identical to Phase 14's in all 12 files. The
  tolerance verdict failed on 16 of 1,381 point values, all of them GLYPHARIS's official OmniDocBench
  scores, from one more fallback page than in Phase 14 (the same page as in Pass A); no count, flag or
  verdict changed (`reports/final_validation/v1_reproduction.md`).

The v1 files are unchanged (8,062 of 8,062). Both failures are in `research/failures.md` and in
`docs/FINAL_VALIDATION.md`, section 2. The exit criterion for the reproduction stays unmet until the
owner rules; items 14 to 20 go ahead, as they do not depend on it.


**Progress (2026-11-19 UTC): item 14, laptop part (GW-EFF-013).** Both `fused-int8` and
`fused-single-line` met all three accuracy tests against `fused`, so all three arms go to the phone
(`reports/final_validation/candidate/GW-EFF-013/accuracy.md`). The laptop sitting's times and peaks are
recorded as information (`laptop.md` in the same folder). The phone part of item 14 and item 15 come
next: the timing build of the Android app is not written yet.
