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
| Dataset-native metric | Where a benchmark defines its own (e.g. HierText's PQ-style metric), it is reported in addition, with the official code, from Phase 14 (§11). |

"Don't care" regions (illegible text) are excluded from both sides where the dataset marks them.
GDC pages also carry *ignore regions*: raster images inside the source PDFs, whose printed text has
no text layer and therefore no ground truth, and (from GDC v1.1) the sheet outside the document
area, where the corner markers and the printed sheet label sit. Ground-truth lines that count are
always scored; an ignore region only excuses predictions inside it that match no ground-truth line.
GDC lines whose alignment score is below 0.3 keep their text in the order-aware reference but
their box is not trusted, so for detection, bag-of-lines and E2E F1 they are don't care. These
exclusions apply to every metric, the order-aware page text included. GDC lines flagged `vertical`
(rotated text such as margin stamps and axis titles) are reported separately and excluded from the
main line metrics: recall and the CER of those lines only, since precision would charge every
spurious prediction on the page to them.

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

**Detector runs** (`glypharis.diagnostics.det`, Phase 8) apply the over- and underfitting rules above
with error = 1 − F1@0.5 in place of CER (underfit threshold `det.diagnostics.underfit_error_threshold`,
default 0.5), and the gradient, NaN/Inf, OOM and LEAKAGE rules unchanged; blank and prediction
collapse do not apply. They add the flags below. Thresholds come from `det.diagnostics`; `val_op` is
the epoch's winning operating point (§7).

| Flag | Rule (defaults) |
|---|---|
| `ALL-BACKGROUND COLLAPSE` | Predicted text pixels / ground-truth kernel pixels < 0.1 on validation, **or** more than 50% of pages with lines that count get no box at `val_op` |
| `ALL-TEXT COLLAPSE` | Predicted text pixels > 50% of valid pixels, **or** predicted / kernel pixels > 10 |
| `THRESHOLD DIVERGENCE` | After 20% of the step budget, validation threshold-map L1 above that of a constant T = 0.5 on the same labels, **or** that L1 ≥ 1.5× its best for 3 consecutive validation points while the probability-map loss stays within 1% of its best |
| `THRESHOLD OUT OF RANGE` | More than 5% of threshold-mask pixels have T outside [0.2, 0.8] (the labels lie in [0.3, 0.7]) |
| `BOX SIZE MISMATCH` | At `val_op`, F1@0.3 − F1@0.5 > 0.25: lines are found but mis-sized |
| `POSTPROCESS DROPS` | At `val_op`, boxes dropped as too thin or outside the image exceed 50% of the components that passed the box score threshold |
| `EMPTY CROPS` | More than 35% of an epoch's training crops have no positive pixel |
| `DATA LOADER BOTTLENECK` (resource note) | The training loop waited for data more than 20% of the epoch |
| `SLOW VALIDATION` (resource note) | A validation pass takes more than 25% of the epoch's training time |
| `LABEL CEILING LOW` (start-up) | The label-ceiling F1@0.5 on 20 validation pages is below 0.85 |

## 7. Early stopping and checkpoint selection

- Monitor: validation CER (recognizer), validation F1 @ 0.5 (detector), never a training metric.
- Patience and min-delta come from config; defaults 8 validation points and 0.001.
- `best.pt` is the checkpoint with the best monitored metric; `last.pt` is the last one. The run report
  always prints both, with their epochs, and states which one was selected and why. They are often
  different.
- **Detector operating points.** A detector config pre-registers one or more post-processing
  operating points (unclip settings). Every epoch scores a fixed seeded validation subset at each of
  them from one forward pass. The monitored metric is the maximum over them of micro F1@0.5 (ties go
  to the first listed), and the winner is that epoch's `val_op`, which each checkpoint stores as
  `selected_op`. The one stored in `best.pt` is the run's **operating point of record**: full
  validation of `best.pt` and `last.pt` reports it as the headline, with no second selection on the
  full set. The arg-max over operating points on the full set is printed for information only and
  is never used for a claim or a comparison. A later phase may move to another operating point,
  chosen on validation, only as its own registered change. F1@0.3 is a diagnostic, never a result.
  The label ceiling (the same post-processing applied to the ground-truth kernels) is reported next
  to detector F1.

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
| 2026-10-22 | §3: GDC v1.1 adds ignore regions for the sheet outside the document area (markers and the printed sheet label); every ground-truth line is unchanged (checked by the alignment script). The GDC vertical-line view reports recall and the CER of those lines only; GDC lines with an untrusted box are stated as don't care for box-based metrics | Second Phase 5 code review: every system reads the sheet label, which has no ground truth, and was charged for it; precision in the vertical view counted every spurious prediction on the page | GDC end-to-end and detection metrics; no results had been reported yet |
| 2026-10-23 | §6: the loss branch of `POSSIBLE OVERFITTING` now needs val loss to rise at least 2% above its window low and val CER to set no new best in the window (before: any positive val-loss slope) | GW-REC-001 raised it 8 times while val CER improved every epoch to the end: a CTC model's validation loss creeps up as it grows more confident, so a loss slope alone is not overfitting. Replayed on GW-REC-001's history the revised rule raises nothing; the original flags stay in that run's record | Diagnostics of later runs |
| 2026-10-23 | §7 as implemented: `best.pt` is the strictly lowest validation CER; early stopping's `min_delta` only drives the patience counter (before: one rule did both, so an improvement smaller than `min_delta` never became `best.pt`). The wording of §7 did not change; the code now follows it | GW-REC-002 kept epoch 24 (7.58%) although epoch 27 reached 7.49%. It was re-selected from its own validation records with `scripts/rec_reselect.py`: the epoch-27 checkpoint re-scored to exactly the logged CER, nothing was retrained, no test data was read, and the superseded files are kept (reports/training/GW-REC-002/report.md, Notes). GW-REC-001 was unaffected: its lowest epoch was the selected one | Checkpoint selection of all runs; GW-REC-002's selected checkpoint |
| 2026-10-23 | Recognizer implementation: the 2-layer BiLSTM runs as two single-layer BiLSTMs with ordinary dropout between them, instead of one stacked `nn.LSTM` with cuDNN's built-in dropout. Same architecture (ARCHITECTURE §4), parameters and parameter order; checkpoints of the old layout load through a key rename | cuDNN draws that dropout from its own random state, which no checkpoint holds, so a resumed CUDA run used different dropout masks from the first step after the resume. On Windows, freeing that cached state at exit also aborted every training process with 0xC0000409 (Git Bash reports 127) after all output was written: the "FAIL" entries of GW-REC-001 and GW-REC-002 in the GPU queue log. With the new code both runs' `best.pt` re-score to exactly their logged validation CER (7.5340% and 7.4870%) | CUDA training runs from this change on draw their LSTM dropout from PyTorch's generator; GW-REC-001/002 results unchanged |
| 2026-10-23 | Resume check on CUDA (Phase 6 exit criterion): two uninterrupted runs A and B, plus C resumed from B's mid-epoch checkpoint. The first step after the resume must match B's to 1e-6, the learning rates must be identical, and C's largest relative loss difference from B may not exceed max(1e-4, 2 × the A-vs-B difference). On the CPU, results must still be identical | The old check required A and C to agree to 1e-4 over 400 steps, which assumes that two CUDA runs reproduce each other. With CTC backward nondeterministic, they do not at real scale: two uninterrupted runs differed by up to 2.90% in loss. The resumed run differed from its own uninterrupted run by 1.51% and matched it exactly on the first step after the resume (reports/training/sanity/GW-REC-001/resume_check.json). The old check's FAIL (2.0e-2) stays in sanity.md | Resume checks on CUDA. The 1e-4 tolerance of 2026-10-10 still applies to the toy check in `scripts/env_report.py` |
| 2026-10-25 | §6: detector runs apply the shared over- and underfitting rules with error = 1 − F1@0.5 in place of CER, plus ten detection flags: map collapse both ways, threshold-branch divergence and range, box-size mismatch, post-processing drops, empty crops, data-loader bottleneck, slow validation and a low label ceiling | Phase 8 adds the detector. The §6 rules were written for the CTC recognizer (CER, blank collapse); a DBNet detector fails in other ways: an all-background or all-text map, a threshold branch that stops learning, lines found but boxes mis-sized | Detector runs (GW-DET-*); recognizer runs unchanged |
| 2026-10-25 | §7: detector selection is the maximum over the pre-registered operating points of validation F1@0.5 on the per-epoch subset; the operating point stored in `best.pt` is the one reported on full validation, with no second selection; F1@0.3 is diagnostic only; the label ceiling is reported next to detector F1 | The right unclip distance depends on how much thicker the trained probability region is than the label kernel, which is not known before training, so two operating points (`dbnet15`, `inverse10`) are pre-registered and scored from the same forward pass. Choosing between them again on the full validation set would be a second selection | Detector selection and reporting, H9 and H10; written before any GW-DET run |
| 2026-10-25 | §3: CLEval-style detection metrics are not computed in Phase 8 (descoped; IMPLEMENTATION_PLAN §8), signed off by the owner at the Phase 8 gate (2026-10-27) | §3 never defined a CLEval metric, so none was pre-registered. That gap is not a reason on its own: CLEval's detection mode would be feasible, since it needs only the ground-truth transcriptions, which HierText has. CLEval asks whether granularity errors (merges, splits, partial lines), which one-to-one IoU matching scores all or nothing, hide real detection quality. Phase 9 answers that from the merge and split counts (§3), and defining a new metric after the detector exists would add a metric chosen after the fact. If the owner declines the descope, CLEval's detection mode is defined in §3 and computed on the CPU from the saved boxes of the three GW-DET runs, without retraining, and labelled as defined after the runs | Detection reporting in Phase 8 |
| 2026-10-25 | §3 for the detector: in Phase 8, GLYPHARIS-Det is scored with the §3 line metrics only; HierText's native PQ-style metric comes from Phase 14 with the official code, as for every system (row of 2026-10-22) | The Phase 8 task in the implementation plan names HierText-style metrics; the 2026-10-22 change had already moved the native metric to Phase 14 | HierText detection reporting in Phase 8; no number changes |
