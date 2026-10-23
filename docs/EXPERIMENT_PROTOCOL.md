# Experiment Protocol

How every GLYPHARIS number is produced. If a result was not produced under this protocol, it is
labelled as such wherever it appears. Changes to this file after Phase 5 are listed in §11 with the
date and the reason, because they can change the meaning of earlier numbers.

## 1. Data roles and splits

Dataset roles are fixed by [ADR-0005](decisions/ADR-0005-evaluation-data-roles.md); provenance is in
[DATASETS.md](../DATASETS.md).

| Role | Used for | Never used for |
|---|---|---|
| train | Gradient updates, charset derivation, augmentation statistics | Model selection |
| validation | Early stopping, checkpoint selection, hyperparameter search, calibration fitting, error analysis that feeds back into design | Final claims on their own |
| test (internal) | One evaluation per frozen model in Phases 5, 14, 22 | Anything that feeds back into training or design |
| external test | Same as test, for out-of-distribution claims | Training, tuning, augmentation sources |

### Splitting rules

1. **Grouped splits.** The unit of splitting is the *source document* (or source image for scene
   datasets). All pages, lines and crops of one document land in exactly one split. Line crops inherit
   the split of their page.
2. **Official splits first.** Where a dataset ships official splits they are kept. A validation split
   carved out of an official train split is grouped as above and recorded with a hash of its ID list.
3. **Missing test labels.** If a benchmark withholds test ground truth, its official validation split
   becomes our test split and a new validation split is carved from train (RISK_REGISTER R-08). The
   change is recorded in DATASETS.md and in the split statistics table below.
4. **Cross-split deduplication.** Three checks, thresholds fixed in Phase 3 (`glypharis.data.audit.Thresholds`,
   recorded in `reports/data_quality_report.md`): exact copies (SHA-256), near copies (dHash ≤ 6 **and**
   pHash ≤ 10 of 64 bits), and **text overlap** (images sharing ≥ 5 distinctive lines covering ≥ 50% of
   the smaller image's distinctive lines; a distinctive line has ≥ 5 characters, ≥ 3 letters and occurs
   in ≤ 20 images). The copy in the lower-priority split is removed (test > validation > train), so
   test images are never removed for duplication. Removals are listed in
   `data/manifests/exclusions/<dataset>.json`.
5. **GDC sources are excluded from all training corpora,** including the text corpus used by GlyphSynth.
   The exclusion list lives in `data/manifests/exclusions/` and is checked by a test.
6. **GlyphSynth validation** uses held-out fonts *and* held-out source texts, so it measures
   generalisation to unseen typography and content rather than memorisation.

Split files (`data/manifests/splits/*.txt`) are committed with their SHA-256 in the dataset manifest.

### Split statistics

Copied from the generated manifests named in the last column (Phase 4).

| Dataset | Split | Images / pages | Lines (recognizer-usable) | Source |
|---|---|---|---|---|
| HierText | train | 8,213 | 493,464 (348,935) | `data/manifests/processed/hiertext.json` |
| HierText | validation | 1,715 | 103,810 (73,888) | same |
| HierText | test | 1,634 | 94,564 (67,277) | same |
| GlyphSynth v1 | train | — | 100,000 | `data/manifests/processed/glyphsynth.json` |
| GlyphSynth v1 | val (held-out fonts and texts) | — | 10,000 | same |
| GLYPH-C v1 | validation-derived | 6,000 samples | 216,000 crops (clean + 7 corruptions x 5 levels) | `data/manifests/processed/glyph_c.json` |
| GDC v1 | test | 106 sheets | 8,606 ground-truth lines | `data/manifests/gdc/pack.json` (capture pending, ADR-0010) |

HierText image counts are after the Phase 3 exclusion list (v1).

## 2. Text normalisation tracks

All metrics are computed on two tracks and both are reported. Normalisation code lives in
`glypharis.data.text` (Phase 4) and is shared by training, evaluation and the baselines.

| Track | Applied to prediction and reference | Purpose |
|---|---|---|
| **strict** | The label normalisation `glypharis.data.text.normalize_label` (v1): Unicode NFC; typographic spaces turned into spaces, whitespace runs collapsed, leading and trailing whitespace stripped; Latin ligatures U+FB00–U+FB06 written out; invisible, control, private-use, unassigned and U+FFFD characters removed | Primary track. Case, punctuation and symbols all count. |
| **normalised** | strict + NFKC + typographic map (curly quotes → straight, en/em dash → `-`, ligatures expanded) + case-folded | Comparison with systems whose output conventions differ (e.g. baselines that emit straight quotes or a reduced charset) |

Characters outside the model charset in a reference are **not** removed; they count as errors the model
cannot avoid. Their share of the reference is reported next to the metric (the "charset ceiling").

Hyphenation at line ends is preserved as written; the evaluator never re-joins hyphenated words.

## 3. Metrics

### Recognition (line or word crops in, text out)

| Metric | Definition |
|---|---|
| **CER** (primary) | Σ Levenshtein(pred, ref) / Σ len(ref), summed over the evaluation set (micro average). Can exceed 1. |
| **WER** | Same, on whitespace-tokenised word sequences. |
| **NED** | Mean over samples of 1 − Levenshtein(pred, ref) / max(len(pred), len(ref)) (ICDAR-style normalised edit distance; 1 is perfect). |
| **Sequence accuracy** | Share of samples where pred == ref exactly. |
| **Word accuracy** | For word-crop benchmarks (IIIT5K): exact match per word, on the benchmark's customary case-insensitive alphanumeric track **and** our strict track. |

Empty references are excluded from CER denominators and counted separately.

### Detection (page in, line polygons out)

| Metric | Definition |
|---|---|
| **Precision / Recall / F1 @ IoU 0.5** (primary) | One-to-one matching by polygon IoU at threshold 0.5: Hungarian assignment that maximises the number of pairs passing the threshold, then their summed IoU. Lines that count are matched before don't-care lines. |
| F1 @ IoU 0.7 | Stricter localisation. |
| Mean matched IoU | Over true positives. |
| Merge / split counts | One GT ↔ many predictions and vice versa (Phase 9). |
| Dataset-native metric | Where a benchmark defines its own (e.g. HierText's PQ-style metric), it is reported in addition, with the official code where available. |

"Don't care" regions (illegible text) are excluded from both sides where the dataset marks them.
GDC pages also carry *ignore regions*: raster images inside the source PDFs, whose printed text has
no text layer and therefore no ground truth. Ground-truth lines are always scored; an ignore region
only excuses predictions inside it that match no ground-truth line. These exclusions apply to every
metric, the order-aware page text included. GDC lines flagged `vertical`
(rotated text such as margin stamps and axis titles) are reported separately and excluded from the
main line metrics.

### End to end (page in, text out)

| Metric | Definition |
|---|---|
| **Page CER / WER, order-aware** | Lines joined in predicted reading order with `\n`, compared with the reference page text. Reported only where the ground truth has a reading order: GDC (PDF text-layer order). HierText's annotation order is not a reading order, so the metric is n/a there. |
| **Page CER, order-agnostic** | Bag-of-lines: predicted lines matched to GT lines by box IoU, CER over matched pairs, unmatched lines count as full deletions / insertions. Measures recognition and detection without reading order. |
| E2E line F1 | A GT line counts as found if a prediction overlaps it at IoU ≥ 0.5 *and* its line CER ≤ 0.1 (threshold fixed here, not tuned). |
| **Oracle analysis** | GT boxes → recognizer, compared with detector boxes → recognizer, to split error between the two stages. |

### Confidence calibration (RQ8)

Expected calibration error (ECE, 15 equal-mass bins) of line confidence against line correctness
(exact match on the strict track), plus a reliability diagram. Reported before and after temperature
scaling; the temperature is fitted on validation only.

### Efficiency

| Metric | Measurement rule |
|---|---|
| Latency mean / p95 | Wall clock per image, after 10 warm-up images, over ≥ 100 images; batch size and threads stated |
| Throughput | Images / second at the stated batch size |
| Peak RAM | Peak working set (RSS high-water mark) of the process, plus that of any engine process it starts (e.g. `tesseract.exe`) |
| Peak VRAM | `torch.cuda.max_memory_allocated` plus `nvidia-smi` process memory |
| Model size | Parameters and on-disk bytes per format (PyTorch, ONNX, INT8 ONNX) |

Efficiency is measured in a dedicated profiling pass (for the baselines `scripts/run_baselines.py
--profile`), one sample at a time on validation data, not taken from long accuracy runs, which share
the machine with other work over hours. Conditions recorded with every efficiency number: power plan
(AC, performance mode), CPU load before and after, CPU thread count,
GPU clocks and temperature at start and end, input resolution, precision. **CPU latency is the primary
deployment number** ([ADR-0008](decisions/ADR-0008-usable-engine-first.md)).

## 4. Statistics

- **Confidence intervals:** 95% percentile bootstrap, 1,000 resamples, resampling *lines* for recognition
  and *pages* for detection and end to end. The resampling unit is stated next to every CI.
- **Model vs model:** paired bootstrap on the same resampled indices; a difference is called significant
  only if the 95% CI of the difference excludes zero.
- **Seeds:** 3 seeds (1337, 2024, 7) for any comparison that goes into a main paper table; mean ± std
  across seeds plus the bootstrap CI of the seed-averaged prediction set. Single-seed runs are labelled
  *exploratory*.
- **Multiple comparisons:** ablation tables report raw CIs and say how many comparisons were made; no
  correction is applied silently.

## 5. Budget matching

Compared arms get the same number of optimiser steps, the same effective batch size and the same LR
schedule unless the variable under study is one of those. Wall-clock time is reported but never used as
the matching criterion, because laptop thermals make it noisy.

## 6. Training diagnostics (every run)

Implemented in `glypharis.diagnostics` (Phase 6). Thresholds come from the run config, defaults below.
Each flag is written into the run report with the epoch it was raised and the evidence.

| Flag | Rule (defaults) |
|---|---|
| `POSSIBLE OVERFITTING` | Over a window of 5 validation points, train loss slope < 0 while val loss rises ≥ 2% above its lowest point in the window **and** val CER sets no new best in the window, **or** val CER rises ≥ 10% relative above its best while train CER keeps falling |
| `POSSIBLE UNDERFITTING` | Train CER > 0.5 (`underfit_cer_threshold`) after 20% of the step budget, **or** both train and val loss improve < 1% relative over the last 5 validation points while val CER is still above that threshold (both poor, both flat) |
| `BLANK COLLAPSE` | Share of blank argmax frames > 0.95 on validation, or greedy decode empty for > 50% of validation lines |
| `PREDICTION COLLAPSE` | > 50% of validation predictions identical |
| `GRADIENT EXPLOSION` / `VANISHING` | Global grad-norm > 1e3, or < 1e-7 for 100 consecutive steps |
| `NAN/INF` | Any non-finite loss: stop, save the offending batch IDs and state, exit non-zero |
| `OOM` | Caught; state saved; run aborts with a report (optional automatic batch halving, logged) |
| `LEAKAGE` | Train/val (and train/test) sample-ID overlap ≠ 0 at start-up: hard error |

The underfitting rule above is a starting point; it is reviewed against the first real runs in Phase 6
and any change is logged in §11.

## 7. Early stopping and checkpoint selection

- Monitor: validation CER (recognizer), validation F1 @ 0.5 (detector), never a training metric.
- Patience and min-delta come from config; defaults 8 validation points and 0.001.
- `best.pt` is the checkpoint with the best monitored metric; `last.pt` is the last one. The run report
  always prints both, with their epochs, and states which one was selected and why. They are often
  different.

## 8. Run reports

Every training run writes `reports/training/<run_id>/report.md` automatically: config hash, git commit,
dataset manifest versions, hardware, best vs final epoch and metrics, all flags from §6, curves (loss,
CER, WER, LR, grad norm, VRAM, GPU temperature), worst-predicted validation samples, and wall-clock /
GPU-hours. Numbers in docs are copied from these reports together with the experiment ID.

## 9. Leakage and test-access rules

- No test-split file is opened by training code; the data loader refuses split names starting with
  `test` unless the evaluation entry point is called with `--test-access <experiment-id>`, which appends
  a row to `research/test_access_log.md` automatically.
- Test evaluations happen only in Phases 5, 14 and 22. Each frozen model is evaluated **once** per test
  set. A model changed after seeing test results is a new model version and is disclosed as such.
- Looking at individual test errors is allowed only in Phase 15 (error analysis), after the model is
  frozen; findings from it go into the paper, not back into that model.

## 10. Baselines

Baselines run through `scripts/run_baselines.py` in `.venv-baselines`
([ADR-0006](decisions/ADR-0006-isolated-environments.md)) and produce the same output schema, so the same
evaluator scores them. Three comparisons are kept separate and never mixed in one table: detection only,
recognition only (GT crops in), end to end. Baseline versions, model names and settings are recorded in
the baseline report. Baselines run with default settings plus at most the documented language/model
choice; no per-benchmark tuning.

## 11. Protocol change log

| Date | Change | Reason | Affects |
|---|---|---|---|
| 2026-10-10 | Initial version (Phase 0) | — | — |
| 2026-10-10 | Ignore regions and vertical lines for GDC (§3); split statistics filled (§1) | Phase 4 built GDC from PDF text layers; raster images inside PDFs have no text layer | GDC metrics |
| 2026-10-10 | CUDA reproducibility tolerance fixed at a maximum **relative** loss difference of 1e-4 between two seeded runs (toy check in `scripts/env_report.py`); CPU runs must be bit-identical | Phase 1 exit criterion needed a documented tolerance. CUDA CTC backward has no deterministic kernel, so exact equality is not guaranteed even though the Phase 1 measurement showed none (reports/environment/env_report.md) | All CUDA runs |
| 2026-10-22 | §2: strict track described as implemented since Phase 4 (`normalize_label`: ligatures written out, invisible and replacement characters removed); the code did not change | The table described only NFC and whitespace | Wording only; no number changes |
| 2026-10-22 | §2 applied to recognition benchmarks: their crops keep lines with out-of-charset characters and very long lines, and the charset ceiling is reported next to every result | The first Phase 5 crops reused the training filter (`rec_ok`), which dropped those lines | Recognition-only benchmarks |
| 2026-10-22 | §3: matching maximises valid pairs before IoU and matches lines that count before don't-care lines; don't-care exclusions apply to the order-aware page text too; order-aware page CER/WER is reported only where a reading order exists (GDC) | Phase 5 code review: plain IoU-sum matching could drop valid pairs, the order-aware text counted reads of don't-care regions as insertions, and HierText has no reading order | Detection, end to end; no results had been reported yet |
| 2026-10-22 | §3: efficiency comes from a dedicated profiling pass, and peak RAM includes engine processes the harness starts | Long accuracy runs share the machine with other work; a command-line engine's memory is outside the Python process | Efficiency numbers |
| 2026-10-22 | §3: dataset-native metrics (HierText's PQ-style metric) are computed from Phase 14, with the official evaluation code | Phase 5 scores every system with the project's own line metrics; the native metric needs the official code and word-level output | HierText reporting |
| 2026-10-23 | §6: the loss branch of `POSSIBLE OVERFITTING` now needs val loss to rise at least 2% above its window low and val CER to set no new best in the window (before: any positive val-loss slope) | GW-REC-001 raised it 8 times while val CER improved every epoch to the end: a CTC model's validation loss creeps up as it grows more confident, so a loss slope alone is not overfitting. Replayed on GW-REC-001's history the revised rule raises nothing; the original flags stay in that run's record | Diagnostics of later runs |
| 2026-10-23 | §7 as implemented: `best.pt` is the strictly lowest validation CER; early stopping's `min_delta` only drives the patience counter (before: one rule did both, so an improvement smaller than `min_delta` never became `best.pt`). The wording of §7 did not change; the code now follows it | GW-REC-002 kept epoch 24 (7.58%) although epoch 27 reached 7.49%. It was re-selected from its own validation records with `scripts/rec_reselect.py`: the epoch-27 checkpoint re-scored to exactly the logged CER, nothing was retrained, no test data was read, and the superseded files are kept (reports/training/GW-REC-002/report.md, Notes). GW-REC-001 was unaffected: its lowest epoch was the selected one | Checkpoint selection of all runs; GW-REC-002's selected checkpoint |
