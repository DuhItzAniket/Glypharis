# Findings

Conclusions that are supported by at least one completed, recorded experiment.
Each finding cites the experiment IDs behind it and states its confidence.
A finding can be revised or withdrawn later; keep the history and say why.

| # | Finding | Evidence (experiment IDs) | Confidence | Date | Status |
|---|---|---|---|---|---|
| F1 | A CTC recognizer's validation loss can creep up late in training while its validation CER keeps improving, so validation-loss divergence alone is not evidence of overfitting. A rule built on it raised eight false alarms. | GW-REC-001 | medium (one run) | 2026-10-23 | current |
| F2 | Adding 35,000 GlyphSynth lines to HierText training did not hurt HierText validation CER (paired difference −0.05 points [−0.09, −0.01]). The gain is not attributable to the synthetic data: one seed, and 10% more optimizer steps. | GW-REC-002 vs GW-REC-001 | low | 2026-10-23 | current |
