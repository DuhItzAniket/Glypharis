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
