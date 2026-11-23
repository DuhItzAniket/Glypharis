# Test-Set Access Log

Append one row **every time** a held-out test split or external benchmark is evaluated.
Allowed phases: 5 (baselines), 14 (external evaluation), 22 (final validation).
Any other access must be justified in the "Reason" column and discussed in the paper's limitations.

## Reads by companion projects

Test-set reads made by other projects with GLYPHARIS output. They are not GLYPHARIS experiments, change
nothing in this repository and are listed here because the GDC protocol logs every GDC read in both
projects. GLYPHARIS's own reads are the table below, which `glypharis.eval.access` appends to.

- **2026-11-23, LECTRIS LX-TEST-001** (LECTRIS Phase 13, its registered test of hypotheses H1, H2 and
  H3, run once; LECTRIS code at `7527e49`).
  - Set: GDC 1.1, the frozen `data/processed/gdc/pages_test.jsonl.gz` (106 pages, 8,606 lines),
    SHA-256 `c4227f1ae6d0aa856c3e64a2bcf55325840343df48dc6810b104e62a6f092af5`, equal to `pages_sha256`
    in `data/manifests/gdc/gdc_v1.json` (frozen 2026-10-22).
  - Engine: GLYPHARIS `glypharis_v1` from the clean 1.0 install, PyTorch backend on the CPU, run once over
    the 106 captures with `glypharis-ocr --format json`; both LECTRIS reads used that one output.
  - Ground truth as in `glypharis/eval/page_metrics.py`: 8,372 lines count, 234 lines and 455 ignore
    regions are don't care.
  - Read-only: nothing in this repository changed, and no GDC text, image or output went into the LECTRIS
    repository (page ids, SHA-256 hashes and aggregate scores only).
  - Outcome: the run exposed a bug in LECTRIS's own scoring (a hyphen-rejoining step in how its
    reference text is built); the run is kept but marked invalid for LECTRIS's corrected-output rows.
    Report: LECTRIS `reports/LX-TEST-001/`.
- **2026-11-23, LECTRIS LX-TEST-002** (LECTRIS's registered full re-run after that bug fix, which its
  implementation plan allows for Phase 13; LECTRIS code at `e722231`).
  - Same set, the same GLYPHARIS output (GLYPHARIS was not run again), pairs and frozen model packs.
  - Report: LECTRIS `reports/LX-TEST-002/` (LECTRIS pull request #34). Headline, as LECTRIS scores it:
    GLYPHARIS's CER on GDC is 21.35% (normalised track, lines matched by IoU >= 0.5), and
    LECTRIS-standard lowers it by 0.02 points [0.01, 0.04] at a 0.012% harm rate. This is LECTRIS's
    metric, not GLYPHARIS's page CER.

| Date | Split / benchmark | Experiment ID | Checkpoint | Phase | Reason |
|---|---|---|---|---|---|
| 2026-10-21 | gdc test | GW-BASE-002 | ppocr_mobile (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | gdc test | GW-BASE-002 | ppocr_mobile (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | hiertext test | GW-BASE-002 | ppocr_mobile (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | hiertext test | GW-BASE-002 | ppocr_mobile (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | gdc test | GW-BASE-002 | ppocr_mobile (rec) | 5 | Phase 5 baseline (resumed: 8106 of 8372 samples already predicted) |
| 2026-10-22 | hiertext test | GW-BASE-002 | ppocr_mobile (rec) | 5 | Phase 5 baseline (resumed: 67277 of 67574 samples already predicted) |
| 2026-10-22 | gdc test | GW-BASE-002 | ppocr_mobile (e2e) | 5 | Phase 5 baseline |
| 2026-10-22 | hiertext test | GW-BASE-002 | ppocr_mobile (e2e) | 5 | Phase 5 baseline |
| 2026-10-22 | gdc test | GW-BASE-001 | tesseract (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | hiertext test | GW-BASE-001 | tesseract (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | gdc test | GW-BASE-001 | tesseract (e2e) | 5 | Phase 5 baseline |
| 2026-10-22 | hiertext test | GW-BASE-001 | tesseract (e2e) | 5 | Phase 5 baseline |
| 2026-10-22 | gdc test | GW-BASE-005 | ppocr6_tiny (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | hiertext test | GW-BASE-005 | ppocr6_tiny (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | gdc test | GW-BASE-005 | ppocr6_tiny (e2e) | 5 | Phase 5 baseline |
| 2026-10-22 | hiertext test | GW-BASE-005 | ppocr6_tiny (e2e) | 5 | Phase 5 baseline |
| 2026-10-22 | gdc test | GW-BASE-006 | ppocr6_small (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | hiertext test | GW-BASE-006 | ppocr6_small (rec) | 5 | Phase 5 baseline |
| 2026-10-22 | gdc test | GW-BASE-006 | ppocr6_small (e2e) | 5 | Phase 5 baseline |
| 2026-10-22 | hiertext test | GW-BASE-006 | ppocr6_small (e2e) | 5 | Phase 5 baseline |
| 2026-10-23 | gdc test | GW-BASE-004 | trocr (rec) | 5 | Phase 5 baseline |
| 2026-10-23 | hiertext test | GW-BASE-004 | trocr (rec) | 5 | Phase 5 baseline |
| 2026-10-23 | hiertext test | GW-BASE-004 | trocr (rec) | 5 | Phase 5 baseline (resumed: 50576 of 67574 samples already predicted) |
| 2026-10-23 | gdc test | GW-BASE-007 | ppocr6_medium (rec) | 5 | Phase 5 baseline |
| 2026-10-23 | gdc test | GW-BASE-007 | ppocr6_medium (e2e) | 5 | Phase 5 baseline |
| 2026-10-24 | gdc test | GW-BASE-003 | ppocr_server (rec) | 5 | Phase 5 baseline |
| 2026-10-24 | gdc test | GW-BASE-003 | ppocr_server (e2e) | 5 | Phase 5 baseline |
| 2026-10-24 | gdc test | GW-BASE-003 | ppocr_server (e2e) | 5 | Phase 5 baseline |
| 2026-10-24 | gdc test | GW-BASE-001 | tesseract (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-001 | tesseract (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-001 | tesseract (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-001 | tesseract (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-002 | ppocr_mobile (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-002 | ppocr_mobile (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-003 | ppocr_server (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-003 | ppocr_server (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-004 | trocr (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-004 | trocr (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-006 | ppocr6_small (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-006 | ppocr6_small (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-006 | ppocr6_small (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-006 | ppocr6_small (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-007 | ppocr6_medium (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-001 | tesseract (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-001 | tesseract (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-001 | tesseract (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-001 | tesseract (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-002 | ppocr_mobile (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-002 | ppocr_mobile (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-003 | ppocr_server (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-004 | trocr (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-004 | trocr (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-006 | ppocr6_small (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-006 | ppocr6_small (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-006 | ppocr6_small (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | hiertext test | GW-BASE-006 | ppocr6_small (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 5 | Phase 5 baseline scoring |
| 2026-10-24 | gdc test | GW-BASE-007 | ppocr6_medium (rec), scoring | 5 | Phase 5 baseline scoring |
| 2026-11-14 | iiit5k test | GW-BASE-001 | tesseract (rec) | 14 | Phase 14 baseline test pass (GW-BASE-001) |
| 2026-11-14 | hiertext test | GW-REC-039 | glypharis (rec) | 14 | Phase 14 GLYPHARIS test pass (GW-REC-039) |
| 2026-11-14 | iiit5k test | GW-BASE-002 | ppocr_mobile (rec) | 14 | Phase 14 baseline test pass (GW-BASE-002) |
| 2026-11-14 | iiit5k test | GW-BASE-005 | ppocr6_tiny (rec) | 14 | Phase 14 baseline test pass (GW-BASE-005) |
| 2026-11-14 | iiit5k test | GW-BASE-006 | ppocr6_small (rec) | 14 | Phase 14 baseline test pass (GW-BASE-006) |
| 2026-11-14 | gdc test | GW-REC-039 | glypharis (rec) | 14 | Phase 14 GLYPHARIS test pass (GW-REC-039) |
| 2026-11-14 | iiit5k test | GW-BASE-007 | ppocr6_medium (rec) | 14 | Phase 14 baseline test pass (GW-BASE-007) |
| 2026-11-14 | gdc_scan test | GW-REC-039 | glypharis (rec) | 14 | Phase 14 GLYPHARIS test pass (GW-REC-039) |
| 2026-11-14 | funsd test | GW-REC-039 | glypharis (rec) | 14 | Phase 14 GLYPHARIS test pass (GW-REC-039) |
| 2026-11-14 | iiit5k test | GW-REC-039 | glypharis (rec) | 14 | Phase 14 GLYPHARIS test pass (GW-REC-039) |
| 2026-11-14 | gdc test | GW-E2E-008 | glypharis (oracle) | 14 | Phase 14 GLYPHARIS test pass (GW-E2E-008) |
| 2026-11-14 | gdc_scan test | GW-E2E-008 | glypharis (oracle) | 14 | Phase 14 GLYPHARIS test pass (GW-E2E-008) |
| 2026-11-14 | gdc_scan test | GW-E2E-007 | glypharis (e2e) | 14 | Phase 14 GLYPHARIS test pass (GW-E2E-007) |
| 2026-11-14 | funsd test | GW-E2E-007 | glypharis (e2e) | 14 | Phase 14 GLYPHARIS test pass (GW-E2E-007) |
| 2026-11-14 | gdc test | GW-E2E-007 | glypharis (e2e) | 14 | Phase 14 GLYPHARIS test pass (GW-E2E-007) |
| 2026-11-14 | omnidocbench test | GW-E2E-007 | glypharis (e2e) | 14 | Phase 14 GLYPHARIS test pass (GW-E2E-007) |
| 2026-11-14 | iiit5k test | GW-BASE-003 | ppocr_server (rec) | 14 | Phase 14 baseline test pass (GW-BASE-003) |
| 2026-11-14 | hiertext test | GW-E2E-007 | glypharis (e2e) | 14 | Phase 14 GLYPHARIS test pass (GW-E2E-007) |
| 2026-11-15 | iiit5k test | GW-BASE-004 | trocr (rec) | 14 | Phase 14 baseline test pass (GW-BASE-004) |
| 2026-11-15 | gdc_scan test | GW-BASE-001 | tesseract (rec) | 14 | Phase 14 baseline test pass (GW-BASE-001) |
| 2026-11-15 | gdc_scan test | GW-BASE-002 | ppocr_mobile (rec) | 14 | Phase 14 baseline test pass (GW-BASE-002) |
| 2026-11-15 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (rec) | 14 | Phase 14 baseline test pass (GW-BASE-005) |
| 2026-11-15 | gdc_scan test | GW-BASE-006 | ppocr6_small (rec) | 14 | Phase 14 baseline test pass (GW-BASE-006) |
| 2026-11-15 | gdc_scan test | GW-BASE-007 | ppocr6_medium (rec) | 14 | Phase 14 baseline test pass (GW-BASE-007) |
| 2026-11-15 | gdc_scan test | GW-BASE-003 | ppocr_server (rec) | 14 | Phase 14 baseline test pass (GW-BASE-003) |
| 2026-11-15 | gdc_scan test | GW-BASE-004 | trocr (rec) | 14 | Phase 14 baseline test pass (GW-BASE-004) |
| 2026-11-15 | funsd test | GW-BASE-001 | tesseract (rec) | 14 | Phase 14 baseline test pass (GW-BASE-001) |
| 2026-11-15 | funsd test | GW-BASE-002 | ppocr_mobile (rec) | 14 | Phase 14 baseline test pass (GW-BASE-002) |
| 2026-11-15 | funsd test | GW-BASE-005 | ppocr6_tiny (rec) | 14 | Phase 14 baseline test pass (GW-BASE-005) |
| 2026-11-15 | funsd test | GW-BASE-006 | ppocr6_small (rec) | 14 | Phase 14 baseline test pass (GW-BASE-006) |
| 2026-11-15 | funsd test | GW-BASE-007 | ppocr6_medium (rec) | 14 | Phase 14 baseline test pass (GW-BASE-007) |
| 2026-11-15 | funsd test | GW-BASE-003 | ppocr_server (rec) | 14 | Phase 14 baseline test pass (GW-BASE-003) |
| 2026-11-15 | funsd test | GW-BASE-004 | trocr (rec) | 14 | Phase 14 baseline test pass (GW-BASE-004) |
| 2026-11-15 | gdc_scan test | GW-BASE-001 | tesseract (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-001) |
| 2026-11-15 | gdc_scan test | GW-BASE-002 | ppocr_mobile (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-002) |
| 2026-11-15 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-005) |
| 2026-11-15 | gdc_scan test | GW-BASE-006 | ppocr6_small (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-006) |
| 2026-11-15 | gdc_scan test | GW-BASE-007 | ppocr6_medium (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-007) |
| 2026-11-15 | funsd test | GW-BASE-001 | tesseract (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-001) |
| 2026-11-15 | funsd test | GW-BASE-002 | ppocr_mobile (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-002) |
| 2026-11-15 | funsd test | GW-BASE-005 | ppocr6_tiny (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-005) |
| 2026-11-15 | funsd test | GW-BASE-006 | ppocr6_small (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-006) |
| 2026-11-15 | funsd test | GW-BASE-007 | ppocr6_medium (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-007) |
| 2026-11-15 | funsd test | GW-BASE-003 | ppocr_server (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-003) |
| 2026-11-15 | omnidocbench test | GW-BASE-001 | tesseract (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-001) |
| 2026-11-15 | omnidocbench test | GW-BASE-002 | ppocr_mobile (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-002) |
| 2026-11-15 | omnidocbench test | GW-BASE-005 | ppocr6_tiny (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-005) |
| 2026-11-15 | omnidocbench test | GW-BASE-006 | ppocr6_small (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-006) |
| 2026-11-15 | omnidocbench test | GW-BASE-007 | ppocr6_medium (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-007) |
| 2026-11-16 | omnidocbench test | GW-BASE-003 | ppocr_server (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-003) |
| 2026-11-16 | omnidocbench test | GW-BASE-007 | ppocr6_medium (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-007) (resumed: 513 of 755 samples already predicted) |
| 2026-11-16 | omnidocbench test | GW-BASE-003 | ppocr_server (e2e) | 14 | Phase 14 baseline test pass (GW-BASE-003), started again: the first start died out of memory before any page |
| 2026-11-17 | gdc test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-003 | ppocr_server (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-007 | ppocr6_medium (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | omnidocbench test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-007 | ppocr6_medium (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-007 | ppocr6_medium (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | iiit5k test | GW-BASE-007 | ppocr6_medium (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | omnidocbench test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | iiit5k test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | omnidocbench test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | iiit5k test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | omnidocbench test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | iiit5k test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-003 | ppocr_server (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-003 | ppocr_server (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-003 | ppocr_server (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | iiit5k test | GW-BASE-003 | ppocr_server (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | omnidocbench test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | iiit5k test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | iiit5k test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | iiit5k test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | hiertext test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | funsd test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | omnidocbench test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-E2E-008 | glypharis (oracle), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc_scan test | GW-E2E-008 | glypharis (oracle), scoring | 14 | Phase 14 scoring (GW-E2E-009) |
| 2026-11-17 | gdc test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-003 | ppocr_server (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-BASE-007 | ppocr6_medium (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | omnidocbench test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-007 | ppocr6_medium (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-007 | ppocr6_medium (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | iiit5k test | GW-BASE-007 | ppocr6_medium (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | omnidocbench test | GW-BASE-006 | ppocr6_small (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | iiit5k test | GW-BASE-006 | ppocr6_small (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | omnidocbench test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | iiit5k test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | omnidocbench test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | iiit5k test | GW-BASE-002 | ppocr_mobile (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-003 | ppocr_server (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-003 | ppocr_server (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-003 | ppocr_server (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | iiit5k test | GW-BASE-003 | ppocr_server (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | omnidocbench test | GW-BASE-001 | tesseract (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | iiit5k test | GW-BASE-001 | tesseract (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | iiit5k test | GW-BASE-004 | trocr (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | iiit5k test | GW-REC-039 | glypharis (rec), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | hiertext test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | funsd test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | omnidocbench test | GW-E2E-007 | glypharis (e2e), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc test | GW-E2E-008 | glypharis (oracle), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-17 | gdc_scan test | GW-E2E-008 | glypharis (oracle), scoring | 14 | Phase 14 scoring (GW-E2E-009) (resumed) |
| 2026-11-18 | gdc test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-003 | ppocr_server (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-BASE-007 | ppocr6_medium (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | omnidocbench test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-007 | ppocr6_medium (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-007 | ppocr6_medium (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | iiit5k test | GW-BASE-007 | ppocr6_medium (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | omnidocbench test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | iiit5k test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | omnidocbench test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | iiit5k test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | omnidocbench test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | iiit5k test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-003 | ppocr_server (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-003 | ppocr_server (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-003 | ppocr_server (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | iiit5k test | GW-BASE-003 | ppocr_server (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | omnidocbench test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | iiit5k test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | iiit5k test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-REC-039 | glypharis (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-REC-039 | glypharis (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-REC-039 | glypharis (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-REC-039 | glypharis (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | iiit5k test | GW-REC-039 | glypharis (rec), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | hiertext test | GW-E2E-007 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-E2E-007 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-E2E-007 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | funsd test | GW-E2E-007 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | omnidocbench test | GW-E2E-007 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc test | GW-E2E-008 | glypharis (oracle), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-18 | gdc_scan test | GW-E2E-008 | glypharis (oracle), scoring | 22 | Phase 22 v1 re-score (GW-E2E-015) |
| 2026-11-19 | hiertext test | GW-REC-043 | glypharis (rec) | 22 | Phase 22 v1 re-run (GW-REC-043, hiertext_test) |
| 2026-11-19 | gdc test | GW-REC-043 | glypharis (rec) | 22 | Phase 22 v1 re-run (GW-REC-043, gdc_test) |
| 2026-11-19 | gdc_scan test | GW-REC-043 | glypharis (rec) | 22 | Phase 22 v1 re-run (GW-REC-043, gdc_scan_test) |
| 2026-11-19 | funsd test | GW-REC-043 | glypharis (rec) | 22 | Phase 22 v1 re-run (GW-REC-043, funsd_test) |
| 2026-11-19 | iiit5k test | GW-REC-043 | glypharis (rec) | 22 | Phase 22 v1 re-run (GW-REC-043, iiit5k_test) |
| 2026-11-19 | gdc test | GW-E2E-017 | glypharis (oracle) | 22 | Phase 22 v1 re-run (GW-E2E-017, gdc_test) |
| 2026-11-19 | gdc_scan test | GW-E2E-017 | glypharis (oracle) | 22 | Phase 22 v1 re-run (GW-E2E-017, gdc_scan_test) |
| 2026-11-19 | gdc_scan test | GW-E2E-016 | glypharis (e2e) | 22 | Phase 22 v1 re-run (GW-E2E-016, gdc_scan_test) |
| 2026-11-19 | funsd test | GW-E2E-016 | glypharis (e2e) | 22 | Phase 22 v1 re-run (GW-E2E-016, funsd_test) |
| 2026-11-19 | gdc test | GW-E2E-016 | glypharis (e2e) | 22 | Phase 22 v1 re-run (GW-E2E-016, gdc_test) |
| 2026-11-19 | omnidocbench test | GW-E2E-016 | glypharis (e2e) | 22 | Phase 22 v1 re-run (GW-E2E-016, omnidocbench_test) |
| 2026-11-19 | hiertext test | GW-E2E-016 | glypharis (e2e) | 22 | Phase 22 v1 re-run (GW-E2E-016, hiertext_test) |
| 2026-11-19 | gdc test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-003 | ppocr_server (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-BASE-007 | ppocr6_medium (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | omnidocbench test | GW-BASE-007 | ppocr6_medium (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-007 | ppocr6_medium (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-007 | ppocr6_medium (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | iiit5k test | GW-BASE-007 | ppocr6_medium (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | omnidocbench test | GW-BASE-006 | ppocr6_small (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | iiit5k test | GW-BASE-006 | ppocr6_small (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | omnidocbench test | GW-BASE-005 | ppocr6_tiny (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | iiit5k test | GW-BASE-005 | ppocr6_tiny (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | omnidocbench test | GW-BASE-002 | ppocr_mobile (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | iiit5k test | GW-BASE-002 | ppocr_mobile (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-003 | ppocr_server (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-003 | ppocr_server (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-003 | ppocr_server (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | iiit5k test | GW-BASE-003 | ppocr_server (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | omnidocbench test | GW-BASE-001 | tesseract (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | iiit5k test | GW-BASE-001 | tesseract (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | iiit5k test | GW-BASE-004 | trocr (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-REC-043 | glypharis (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-REC-043 | glypharis (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-REC-043 | glypharis (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-REC-043 | glypharis (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | iiit5k test | GW-REC-043 | glypharis (rec), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-E2E-016 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-E2E-016 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-E2E-016 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-E2E-016 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | omnidocbench test | GW-E2E-016 | glypharis (e2e), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-E2E-017 | glypharis (oracle), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-E2E-017 | glypharis (oracle), scoring | 22 | Phase 22 v1 re-run scoring (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-E2E-018 | glypharis (rec), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-E2E-018 | glypharis (rec), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-E2E-018 | glypharis (rec), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-E2E-018 | glypharis (rec), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | iiit5k test | GW-E2E-018 | glypharis (rec), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-E2E-018 | glypharis (oracle), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-E2E-018 | glypharis (oracle), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | gdc_scan test | GW-E2E-018 | glypharis (e2e), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | funsd test | GW-E2E-018 | glypharis (e2e), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | gdc test | GW-E2E-018 | glypharis (e2e), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | omnidocbench test | GW-E2E-018 | glypharis (e2e), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-19 | hiertext test | GW-E2E-018 | glypharis (e2e), line comparison | 22 | Phase 22 v1 line comparison, counts only (GW-E2E-018) |
| 2026-11-20 | hiertext test | GW-REC-045 | glypharis_phone_candidate (rec) | 22 | Phase 22 candidate test run (GW-REC-045) |
| 2026-11-20 | gdc test | GW-REC-045 | glypharis_phone_candidate (rec) | 22 | Phase 22 candidate test run (GW-REC-045) |
| 2026-11-20 | gdc_scan test | GW-REC-045 | glypharis_phone_candidate (rec) | 22 | Phase 22 candidate test run (GW-REC-045) |
| 2026-11-20 | funsd test | GW-REC-045 | glypharis_phone_candidate (rec) | 22 | Phase 22 candidate test run (GW-REC-045) |
| 2026-11-20 | iiit5k test | GW-REC-045 | glypharis_phone_candidate (rec) | 22 | Phase 22 candidate test run (GW-REC-045) |
| 2026-11-20 | gdc_scan test | GW-E2E-020 | glypharis_phone_candidate (e2e) | 22 | Phase 22 candidate test run (GW-E2E-020) |
| 2026-11-20 | funsd test | GW-E2E-020 | glypharis_phone_candidate (e2e) | 22 | Phase 22 candidate test run (GW-E2E-020) |
| 2026-11-20 | gdc test | GW-E2E-020 | glypharis_phone_candidate (e2e) | 22 | Phase 22 candidate test run (GW-E2E-020) |
| 2026-11-20 | omnidocbench test | GW-E2E-020 | glypharis_phone_candidate (e2e) | 22 | Phase 22 candidate test run (GW-E2E-020) |
| 2026-11-21 | hiertext test | GW-E2E-020 | glypharis_phone_candidate (e2e) | 22 | Phase 22 candidate test run (GW-E2E-020) |
| 2026-11-21 | hiertext test | GW-REC-045 | glypharis (rec), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | gdc test | GW-REC-045 | glypharis (rec), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | gdc_scan test | GW-REC-045 | glypharis (rec), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | funsd test | GW-REC-045 | glypharis (rec), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | iiit5k test | GW-REC-045 | glypharis (rec), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | hiertext test | GW-E2E-020 | glypharis (e2e), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | gdc test | GW-E2E-020 | glypharis (e2e), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | gdc_scan test | GW-E2E-020 | glypharis (e2e), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | funsd test | GW-E2E-020 | glypharis (e2e), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | omnidocbench test | GW-E2E-020 | glypharis (e2e), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | hiertext test | GW-REC-039 | glypharis (rec), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | gdc test | GW-REC-039 | glypharis (rec), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | hiertext test | GW-E2E-007 | glypharis (e2e), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
| 2026-11-21 | gdc test | GW-E2E-007 | glypharis (e2e), scoring | 22 | Phase 22 candidate scoring (GW-E2E-021) |
