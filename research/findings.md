# Findings

Conclusions that are supported by at least one completed, recorded experiment.
Each finding cites the experiment IDs behind it and states its confidence.
A finding can be revised or withdrawn later; keep the history and say why.

| # | Finding | Evidence (experiment IDs) | Confidence | Date | Status |
|---|---|---|---|---|---|
| F1 | A CTC recognizer's validation loss can creep up late in training while its validation CER keeps improving, so validation-loss divergence alone is not evidence of overfitting. A rule built on it raised eight false alarms. | GW-REC-001 | medium (one run) | 2026-10-23 | current |
| F2 | Adding 35,000 GlyphSynth lines to HierText training did not hurt HierText validation CER (paired difference −0.05 points [−0.09, −0.01]). The gain is not attributable to the synthetic data: one seed, and 10% more optimizer steps. | GW-REC-002 vs GW-REC-001 | low | 2026-10-23 | current |
| F3 | For HierText line recognition at this model size, input height 48 helps most of any single change (−0.49 points on the short budget), and it combines with an SVTR-style encoder and weight decay 1e-2 to more than the sum of the three (−0.76 against −0.66). At the full budget the combination reaches 6.46% validation CER, -1.07 [-1.13, -1.01] points against the CRNN baseline. | GW-REC-006, 014, 020, 022, 024 vs GW-REC-001, 003 | medium (one seed per configuration) | 2026-10-26 | current |
| F4 | Halving the frames per line (frame stride 8) costs 2.6 points of CER on HierText lines: many lines are too dense for one frame per 8 input columns. Capacity cuts to the CRNN cost 0.46 to 0.51 points each. | GW-REC-021, 007, 008, 009 | medium (one seed each) | 2026-10-26 | current |
| F5 | On HierText, the trained detector's errors are mostly misses, not false alarms: precision 0.92, recall 0.59 against a 0.95 label ceiling. Recall falls to 0.20 for lines under 8 px high and to 0.18 for lines rotated beyond 45°, against 0.72 for 12–20 px horizontal lines. | GW-DET-001 | medium (one run, 74,500 lines) | 2026-10-27 | current |
| F6 | For the DBNet-style detector at 640² crops and a 24-epoch budget, ImageNet initialisation is worth about 0.02 F1@0.5 and MobileNetV3-Large over Small about 0.056, both mostly through recall; Small loses most on small text. | GW-DET-002, 003 vs GW-DET-001 | medium (one seed each) | 2026-10-27 | current |
| F7 | Of GW-DET-001's 30,184 missed validation lines, 46% had a candidate region that only the confidence threshold removed, 37% produced no candidate at all (mostly lines under 8 px high) and 11% were merged with a neighbour. Its predicted kernels match the label kernels' width (median ratio 1.00), so the exact-inverse unclip is the right regime. | GW-DET-001, `reports/error_cases/detector/GW-DET-001/` | medium (one run, 74,500 lines) | 2026-10-28 | current |
| F8 | Post-processing chosen on half of validation and checked on the other half adds 0.032 F1@0.5 [0.025, 0.039] to the same network: a lower box threshold (0.5) and slightly smaller boxes trade seven points of precision for eight of recall. | GW-DET-004 | medium (one network, two halves of one set) | 2026-10-28 | current |
