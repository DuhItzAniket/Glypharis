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
| 5 | Baseline systems | **IN PROGRESS** | — |
| 6 | Recognition baseline | **IN PROGRESS** (in parallel with 5, owner approved) | — |
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
