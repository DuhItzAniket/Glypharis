# Changelog

All notable changes to GLYPHARIS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and package versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Results are not repeated here. Each entry names its experiment IDs; the numbers are in the reports
those IDs point to (`research/experiments.md`) and in [`MODEL_CARD.md`](MODEL_CARD.md). Every phase
has its full record, gate and known issues in [`docs/PHASE_STATUS.md`](docs/PHASE_STATUS.md).

## [1.0.0] — unreleased

v1 is the recognizer GW-REC-034 with the detector GW-DET-001, frozen in
`configs/final/glypharis_v1.yaml` and packed as the bundle `glypharis-v1` (model version
`glypharis_v1`). Nothing in this version has been published. The weights are held until the
qualified advice that ADR-0014 asks for (Phase 23, D4).

### Added

- **Phase 23, release preparation (in progress).** `LICENSE` (Apache-2.0 for the code and the
  project's own text), `NOTICE` (what that licence does not cover), `CITATION.cff`, this changelog,
  a release-notes draft (`docs/RELEASE_NOTES_v1.0.md`) and a brief for a qualified adviser on the
  weights' terms (`docs/ADVISER_BRIEF.md`). `scripts/check_citation.py` checks `CITATION.cff`
  against the package version.
- **Phase 23, model download.** `glypharis.download_model` and `glypharis/hub.py`: a pinned
  registry of bundles, each archive checked for its size and SHA-256 and each bundle's manifest
  against its pinned SHA-256 before it is put in place. The `glypharis-v1` entry stays unpublished,
  and so refuses to download, until the weights go out. Failures raise `ModelDownloadError`;
  `glypharis-ocr --no-download` and the `GLYPHARIS_OFFLINE` environment variable keep it from
  reaching the network.
- **Phase 23, packaging.** The package is version 1.0.0, with the version read dynamically from
  `glypharis.__version__`, an Apache-2.0 SPDX licence with `license-files` (`LICENSE`, `NOTICE`),
  and Python >=3.11.4. `.gitattributes` leaves `reports/` and the records of the evaluation-only
  sets out of `git archive` (export-ignore), and `scripts/check_git_archive.py` checks that
  exclusion.
- **Phase 23, release tooling.** `.github/workflows/release.yml`, which checks, builds and drafts a
  release holding the code files only, and uploads to PyPI only after the owner publishes the
  draft; `scripts/pack_release.py`, which packs the v1 bundle and its weight notices
  (`release/weights/`) into a reproducible archive; the `glypharis-check` command with its sample
  page; and `scripts/release_rehearsal.py`. The release rehearsal (GW-E2E-022) and the clean-clone
  training smoke runs (GW-REC-046, GW-DET-008) passed. The verification of the published release
  (GW-E2E-023) is planned.
- **Phase 22, final validation.** A walkthrough from a fresh clone (GW-E2E-013), re-scoring of the
  saved Phase 14 files (GW-E2E-015) and a re-run of the frozen v1 on every test set (GW-REC-043,
  GW-E2E-016, GW-E2E-017, scored in GW-E2E-018); record in `docs/FINAL_VALIDATION.md`. The re-run
  gave Phase 14's predictions in every test file. The strict reproduction verdict failed only on
  official OmniDocBench scores, through that evaluator's per-page time limit, and the owner accepted
  this as a disclosed limitation.
- **Phase 22, phone candidate.** The all-ONNX build with fused attention, reading one line per
  recognizer run, chosen on validation (GW-EFF-013), bundled and checked (GW-EFF-014), gated on
  validation (GW-REC-044, GW-E2E-019) and evaluated once per test set as a new version (GW-REC-045,
  GW-E2E-020, scored in GW-E2E-021): it met H11's non-inferiority tests against v1. Its networks
  were timed on two named phones (GW-EFF-015). It is not v1 and is not part of this release.
- **Phase 21, research package.** `MODEL_CARD.md`, a corrected `DATASETS.md`, a generated
  `EXPERIMENTS.md`, the paper notes under `research/paper/` and a checked `references.bib`.
  `scripts/trace_numbers.py --check` ties every number in those documents to an experiment ID and a
  report, in CI. ADR-0014 records the intended terms for the v1 weights and the ImageNet-initialised
  detector as a disclosed exception to ADR-0009.
- **Phase 16, efficiency.** The v1 cost table (GW-EFF-007), INT8 of the ONNX recognizer
  (GW-EFF-008), an accuracy/latency Pareto study of variants that would be new versions
  (GW-EFF-009), and the ONNX and encoder latency comparisons (GW-EFF-010, GW-EFF-011), with the
  PyTorch references of v1 (GW-EFF-006). Every line and page latency target of ADR-0011 was missed
  (`reports/efficiency/phase16/`).
- **Phase 15, error analysis.** Recognition errors (GW-REC-040), pipeline errors and the detector
  against recognizer split (GW-E2E-010), and GlyphSynth slices (GW-REC-041), under
  `reports/error_cases/phase15/` and `reports/confusion_analysis/GW-REC-040/`; findings F27 to F33.
- **Phase 14, external evaluation.** One evaluation of the frozen v1 per test set: recognition from
  ground-truth crops (GW-REC-039), end to end (GW-E2E-007), the oracle run (GW-E2E-008), and scoring
  with the official HierText and OmniDocBench evaluators (GW-E2E-009), against the baselines
  GW-BASE-001 to GW-BASE-007 under one protocol. Every test access is in
  `research/test_access_log.md`.
- **Phase 13, ablation and freeze.** The ablation ladder L0 to L5 and the full-budget final
  recognizer (GW-REC-037; seeds GW-REC-034 to GW-REC-036), the detector variant GW-DET-007, and the
  frozen configuration `configs/final/glypharis_v1.yaml`, chosen on validation only.
- **Phase 12, synthetic data study.** GlyphSynth arms GW-DATA-010 to GW-DATA-021, compared in
  GW-DATA-022: more synthetic lines did not help at this budget, and GlyphSynth is in no delivered
  model (`reports/synthetic/`).
- **Phase 11, robustness training.** Augmentation families compared on GLYPH-C v1 (GW-DATA-009, with
  the arm runs GW-REC-003 to GW-REC-005, GW-REC-015 to GW-REC-018 and GW-DATA-001 to GW-DATA-008):
  geometric, document
  degradation and the full preset helped, photometric had no clear effect
  (`reports/robustness/phase11_comparison.md`).
- **Phase 19, HTTP API.** `glypharis-serve`, a one-request-at-a-time local endpoint (`POST /ocr`,
  `GET /health`) on the standard library, with size and pixel limits. It is an integration boundary
  without authentication or TLS.
- `GlypharisOCR.predict` also accepts the bytes of an image file (Phase 19).

### Changed

- **Phase 23, dependencies.** The base install keeps only what inference needs; training, data
  preparation and reports, ONNX export and development tools move to the `train`, `export` and
  `dev` extras.
- **Phases 17, 18 and 20 re-run on v1.** The ONNX export of the v1 bundle `glypharis-v1` checked
  against PyTorch (GW-EFF-006) and the shipped PyTorch path checked against its references
  (GW-EFF-012); the package on the v1 bundle against the frozen pipeline (GW-E2E-011); failure
  paths, fuzz-lite cases and damaged bundles on both backends (GW-E2E-012). The v1 code is tagged
  `v1-code`.
- The recognizer trainer runs on the shared training loop (Phase 11).
- v1 (GW-REC-034 with GW-DET-001) replaces v0.1's models (GW-REC-024 with GW-DET-001 at the Phase 9
  settings) as the engine's model.

### Fixed

- `glypharis-serve` no longer loses its answer when it refuses a body it has not read, such as an
  upload over the size limit: it answers first, closes the write side, then discards the rest within
  a byte and a time limit. Status codes and error bodies are unchanged.

## [0.1] — 2026-10-28

The first usable engine ([ADR-0008](docs/decisions/ADR-0008-usable-engine-first.md)), tag `v0.1`,
covering Phases 0 to 10, 17, 18 and 20. Not published as a package; no weights were distributed.

- Planning, environment, dataset acquisition and audit, preprocessing, GlyphSynth, GLYPH-C and the
  GDC test set (Phases 0 to 4).
- Baselines under one protocol: Tesseract, PaddleOCR and TrOCR-Small (GW-BASE-001 to GW-BASE-007,
  Phase 5).
- The CRNN recognizer baseline (GW-REC-001, Phase 6), the improved recognizer GW-REC-024 (Phase 7),
  the detector GW-DET-001 (Phase 8) and its error analysis and settings (GW-DET-004 to GW-DET-006,
  Phase 9).
- The end-to-end pipeline with reading order and calibrated line confidence (GW-E2E-001 to
  GW-E2E-003, GW-EFF-002, Phase 10).
- ONNX export with equivalence checks (GW-EFF-003, GW-EFF-004, Phase 17), the `GlypharisOCR` package
  and `glypharis-ocr` command (GW-E2E-004, Phase 18), and input hardening (Phase 20).

[1.0.0]: https://github.com/DuhItzAniket/Glypharis/compare/v0.1...HEAD
[0.1]: https://github.com/DuhItzAniket/Glypharis/releases/tag/v0.1
