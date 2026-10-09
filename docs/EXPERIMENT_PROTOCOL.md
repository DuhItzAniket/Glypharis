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
4. **Cross-split deduplication.** Exact (SHA-256) and near-duplicate (perceptual hash, Hamming distance
   threshold set and recorded in Phase 3) images across splits are reported. Duplicates are removed
   from the *training* side, never from evaluation splits.
5. **GDC sources are excluded from all training corpora,** including the text corpus used by GlyphSynth.
   The exclusion list lives in `data/manifests/exclusions/` and is checked by a test.
6. **GlyphSynth validation** uses held-out fonts *and* held-out source texts, so it measures
   generalisation to unseen typography and content rather than memorisation.

Split files (`data/manifests/splits/*.txt`) are committed with their SHA-256 in the dataset manifest.

### Split statistics

Filled in Phase 4 from `scripts/audit_data.py` output. Values are copied from
`reports/data_analysis/split_stats.md`, not typed by hand.

| Dataset | Split | Images / pages | Lines | Characters | Source |
|---|---|---|---|---|---|
| — | — | — | — | — | pending Phase 4 |

## 2. Text normalisation tracks

All metrics are computed on two tracks and both are reported. Normalisation code lives in
`glypharis.data.text` (Phase 4) and is shared by training, evaluation and the baselines.

| Track | Applied to prediction and reference | Purpose |
|---|---|---|
| **strict** | Unicode NFC; strip leading/trailing whitespace; collapse internal whitespace runs to one space | Primary track. Case, punctuation and symbols all count. |
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
| **Precision / Recall / F1 @ IoU 0.5** (primary) | One-to-one matching by polygon IoU (Hungarian on IoU, threshold 0.5). |
| F1 @ IoU 0.7 | Stricter localisation. |
| Mean matched IoU | Over true positives. |
| Merge / split counts | One GT ↔ many predictions and vice versa (Phase 9). |
| Dataset-native metric | Where a benchmark defines its own (e.g. HierText's PQ-style metric), it is reported in addition, with the official code where available. |

"Don't care" regions (illegible text) are excluded from both sides where the dataset marks them.

### End to end (page in, text out)

| Metric | Definition |
|---|---|
| **Page CER / WER, order-aware** | Lines joined in predicted reading order with `\n`, compared with the reference page text. |
| **Page CER, order-agnostic** | Bag-of-lines: predicted lines matched to GT lines by box IoU, CER over matched pairs, unmatched lines count as full deletions / insertions. Separates reading-order errors from recognition errors. |
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
| Peak RAM | Process RSS high-water mark |
| Peak VRAM | `torch.cuda.max_memory_allocated` plus `nvidia-smi` process memory |
| Model size | Parameters and on-disk bytes per format (PyTorch, ONNX, INT8 ONNX) |

Conditions recorded with every efficiency number: power plan (AC, performance mode), CPU thread count,
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
| `POSSIBLE OVERFITTING` | Over a window of 5 validation points, train loss slope < 0 while val loss slope > 0, **or** val CER rises ≥ 10% relative above its best while train CER keeps falling |
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
| 2026-10-10 | CUDA reproducibility tolerance fixed at a maximum **relative** loss difference of 1e-4 between two seeded runs (toy check in `scripts/env_report.py`); CPU runs must be bit-identical | Phase 1 exit criterion needed a documented tolerance. CUDA CTC backward has no deterministic kernel, so exact equality is not guaranteed even though the Phase 1 measurement showed none (reports/environment/env_report.md) | All CUDA runs |
