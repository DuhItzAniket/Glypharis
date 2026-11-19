# Test-Set Access Log

Append one row **every time** a held-out test split or external benchmark is evaluated.
Allowed phases: 5 (baselines), 14 (external evaluation), 22 (final validation).
Any other access must be justified in the "Reason" column and discussed in the paper's limitations.

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
