# Error Analysis — paper section notes

Populated from Phases 9 and 15 (automatic difficult-example reports).
Planned categories: blur, rotation, small font, low contrast, touching characters, broken
characters, punctuation, numbers, long words, unusual fonts, background noise, cropping error,
detector failure, recognizer failure.

_No analysis yet._ Phase 15 (registered 2026-11-17, `docs/PHASE_STATUS.md`) fills this from
`reports/error_cases/phase15/` and `reports/confusion_analysis/GW-REC-040/`, with Phase 9's
validation detector categories (`reports/error_cases/detector/phase09_adopted/analysis.md`).

Outline: 1. where page error comes from (budget, oracle share, H4); 2. detection failures
(unmatched lines, cropping error, recall slices, rotated pages, forms); 3. reading order;
4. recognition failures (operations, confusions, words, slices, categories, annotation noise);
5. labelled synthetic slices (GW-REC-041, GLYPH-C); 6. what the baselines share; 7. limits.
