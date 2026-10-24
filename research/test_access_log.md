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
