# Experiment Registry

Every experiment ID is allocated here **before** the run starts (see `docs/WORKFLOW.md` §4).
Status values: `planned` → `running` → `done` | `failed` | `abandoned`.

Rows link to a full entry below. Numbers are copied from generated reports only, never typed by hand.

| ID | Date allocated | Phase | Question / hypothesis | Status | Key result | Report |
|---|---|---|---|---|---|---|
| [GW-BASE-001](#gw-base-001) | 2026-10-21 | 05 | Tesseract 5 (LSTM, eng): recognition and end to end | planned | — | `reports/baselines/` |
| [GW-BASE-002](#gw-base-002) | 2026-10-21 | 05 | PaddleOCR PP-OCRv5 mobile (det + English rec): recognition and end to end | planned | — | `reports/baselines/` |
| [GW-BASE-003](#gw-base-003) | 2026-10-21 | 05 | PaddleOCR PP-OCRv5 server (det + rec): recognition and end to end | planned | — | `reports/baselines/` |
| [GW-BASE-004](#gw-base-004) | 2026-10-21 | 05 | TrOCR-small-printed: recognition only | planned | — | `reports/baselines/` |
| [GW-BASE-005](#gw-base-005) | 2026-10-21 | 05 | PaddleOCR PP-OCRv6 tiny (det + rec): recognition and end to end | planned | — | `reports/baselines/` |
| [GW-BASE-006](#gw-base-006) | 2026-10-21 | 05 | PaddleOCR PP-OCRv6 small (det + rec): recognition and end to end | planned | — | `reports/baselines/` |
| [GW-BASE-007](#gw-base-007) | 2026-10-21 | 05 | PaddleOCR PP-OCRv6 medium (PaddleOCR 3.7's English default): reduced scope | planned | — | `reports/baselines/` |
| [GW-REC-001](#gw-rec-001) | 2026-10-22 | 06 | CRNN baseline on HierText lines | done | val CER 7.53% [7.38, 7.69] | `reports/training/GW-REC-001/` |
| [GW-REC-002](#gw-rec-002) | 2026-10-22 | 06 | GW-REC-001 plus 35,000 GlyphSynth lines (exploratory look at H2′a) | done | val CER 7.49% [7.34, 7.64]; −0.05 points [−0.09, −0.01] vs GW-REC-001 (one seed, not budget-matched) | `reports/training/GW-REC-002/` |

### GW-BASE-001 — Tesseract 5 (LSTM, eng): recognition and end to end

- **System:** Tesseract 5 (UB Mannheim build), `-l eng`, default engine; line crops with `--psm 7`, pages with the default page mode (`--psm 3`).
- **Date allocated / run:** 2026-10-21 / —
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** see `reports/baselines/` (filled in after the run)
- **Training diagnostics:** not applicable (no training)
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** not applicable

### GW-BASE-002 — PaddleOCR PP-OCRv5 mobile (det + English rec): recognition and end to end

- **System:** PaddleOCR 3.7 with PP-OCRv5_mobile_det and en_PP-OCRv5_mobile_rec, CPU. Pipeline defaults except: document unwarping off (it moves the output polygons into the unwarped image's frame), and the text detector runs without oneDNN because Paddle 3.3's oneDNN path crashes on the PP-OCRv5 detectors on this machine. Recognizer outputs were checked to be identical with oneDNN on and off (64 of 64 GlyphSynth-val lines), so this is a speed-only change.
- **Date allocated / run:** 2026-10-21 / —
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** see `reports/baselines/` (filled in after the run)
- **Training diagnostics:** not applicable (no training)
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** not applicable

### GW-BASE-003 — PaddleOCR PP-OCRv5 server (det + rec): recognition and end to end

- **System:** PaddleOCR 3.7 with PP-OCRv5_server_det and PP-OCRv5_server_rec (no English-only server recognizer exists), CPU, same pipeline settings as GW-BASE-002. **Reduced scope:** the server recognizer needs about 1–3 s per line on this CPU, so it runs on GDC test (recognition and end to end) and HierText validation (recognition) only. HierText test and GlyphSynth val would add roughly 45 CPU-hours and are recorded as not run.
- **Date allocated / run:** 2026-10-21 / —
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** see `reports/baselines/` (filled in after the run)
- **Training diagnostics:** not applicable (no training)
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** not applicable

### GW-BASE-004 — TrOCR-small-printed: recognition only

- **System:** microsoft/trocr-small-printed, transformers on CPU, greedy decoding as in the checkpoint, with `max_new_tokens=256` because the checkpoint sets no length and transformers would otherwise stop at 20 tokens and cut long lines, and with the decoder cache on: the checkpoint ships `use_cache: false`, which re-runs the whole prefix at every step. Outputs were checked identical on 64 validation lines; decoding is 2.3x faster. It has no detector, so it is run on recognition only.
- **Date allocated / run:** 2026-10-21 / —
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** see `reports/baselines/` (filled in after the run)
- **Training diagnostics:** not applicable (no training)
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** not applicable

### GW-BASE-005 — PaddleOCR PP-OCRv6 tiny (det + rec): recognition and end to end

- **System:** PaddleOCR 3.7 with PP-OCRv6_tiny_det and PP-OCRv6_tiny_rec (multilingual recognizer: Chinese, English, Latin scripts), CPU, same pipeline settings and deviations as GW-BASE-002 (the v6 detectors crash under oneDNN too).
- **Date allocated / run:** 2026-10-21 / —
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** see `reports/baselines/` (filled in after the run)
- **Training diagnostics:** not applicable (no training)
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** not applicable

### GW-BASE-006 — PaddleOCR PP-OCRv6 small (det + rec): recognition and end to end

- **System:** PaddleOCR 3.7 with PP-OCRv6_small_det and PP-OCRv6_small_rec, CPU, same settings as GW-BASE-005.
- **Date allocated / run:** 2026-10-21 / —
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** see `reports/baselines/` (filled in after the run)
- **Training diagnostics:** not applicable (no training)
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** not applicable

### GW-BASE-007 — PaddleOCR PP-OCRv6 medium (PaddleOCR 3.7's English default): reduced scope

- **System:** PaddleOCR 3.7 with PP-OCRv6_medium_det and PP-OCRv6_medium_rec, which is what `PaddleOCR(lang="en")` loads by default in 3.7. CPU, same settings as GW-BASE-005. **Reduced scope** for the same reason as GW-BASE-003 (about 1 s per line): GDC test recognition and end to end, HierText validation recognition.
- **Date allocated / run:** 2026-10-21 / —
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** see `reports/baselines/` (filled in after the run)
- **Training diagnostics:** not applicable (no training)
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** not applicable

### GW-REC-001 — CRNN baseline on HierText lines

- **Date allocated / run:** 2026-10-22 / 2026-10-22 to 2026-10-23
- **Phase:** 06
- **Git commit:** `9d07ea2` (run summary)   **Config:** `configs/rec/gw-rec-001.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** HierText line packs from Phase 4 (`data/manifests/processed/hiertext.json`); charset v1
- **Hardware:** RTX 4050 Laptop 6 GB, bf16 autocast, CTC in fp32
- **Research question:** how well does the ARCHITECTURE §4 CRNN (about 7.8M parameters, H = 32) learn HierText lines, and is the training loop healthy?
- **Hypothesis (written before the run):** none tested; this is the reference run that Phase 7 improves on. Expectation, not a claim: no diagnostic flags other than possible late overfitting.
- **Method:** `python train.py --config configs/rec/gw-rec-001.yaml` after the sanity sequence (`scripts/rec_sanity.py`: overfit 64 lines, LR range test, 10-minute dry run). Up to 30 epochs, AdamW 1e-3 with OneCycle, batch 64, augmentation preset `full`, early stopping on validation CER (patience 8). Checkpoint selection on validation CER (strict track).
- **Variables:** none (reference run)
- **Control:** —
- **Result:** (from `reports/training/GW-REC-001/report.md`) selected checkpoint = final epoch 29 (step 170,116): validation CER **7.53% [7.38, 7.69]** (95% bootstrap CI over 73,888 lines), validation WER 21.59%, train CER 3.57% on a fixed un-augmented sample of training lines. 7,796,435 parameters (29.8 MiB of weights and buffers).
- **Training diagnostics:** best epoch = final epoch; 30 epochs, no early stop. Eight `POSSIBLE OVERFITTING` flags (epochs 18 to 29) from the original rule while validation CER kept improving to the last epoch. They were false alarms, the rule was revised (EXPERIMENT_PROTOCOL §11, 2026-10-23), and the flags stay in the run record. No NaN, collapse or gradient flags. After writing its report the process aborted with 0xC0000409, caused by cuDNN's LSTM dropout state at exit. This had no effect on results and is fixed in `c58c063`; under that code `best.pt` re-scores to exactly the same validation CER (§11).
- **Interpretation:** the training loop is healthy and the ARCHITECTURE §4 CRNN learns HierText lines. The selected epoch is the last one (`curves.png`), so the 30-epoch budget may not be saturated. The worst-line table is dominated by one- or two-character references on crops that hold more text, which is annotation noise in HierText. It is recorded, not corrected, and Phase 7's error analysis should count it.
- **Limitations:** single seed; validation is the in-charset HierText validation pack; the train CER comes from a fixed un-augmented sample, not from the augmented batches.
- **Next experiment:** Phase 7: a longer schedule (the best epoch is the last), then the planned ablations.
- **Training time:** 4.71 GPU-hours (RTX 4050 Laptop), 30 epochs.

### GW-REC-002 — GW-REC-001 plus a small GlyphSynth mix

- **Date allocated / run:** 2026-10-22 / 2026-10-23
- **Phase:** 06
- **Git commit:** epochs 0 to 16 at `13cab15`; after a planned stop at the epoch 16/17 boundary, resumed at `5124b16` (the commit in the run summary). The commits in between change only the overfitting diagnostic and the report layout, not training.   **Config:** `configs/rec/gw-rec-002.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001, plus GlyphSynth v1 train (`data/manifests/processed/glyphsynth.json`), a fixed seeded subset of 35,000 lines
- **Hardware:** as GW-REC-001
- **Research question:** does a small amount of document-style synthetic text change HierText validation CER?
- **Hypothesis (written before the run):** exploratory look at [H2′a](hypotheses.md); the real test, with several sizes and seeds, is Phase 12. Expectation: similar or slightly lower HierText validation CER; a clear gain is more likely on long, document-like lines than on HierText's short scene text.
- **Method:** identical to GW-REC-001 except the training data. Same seed, schedule, steps per epoch scale with the larger training set.
- **Variables:** independent: training data (HierText vs HierText + 35,000 GlyphSynth lines); controlled: everything else
- **Control:** GW-REC-001
- **Result:** (from `reports/training/GW-REC-002/report.md` and `reports/training/compare/GW-REC-002_vs_GW-REC-001.md`) selected checkpoint epoch 27 (step 174,692): validation CER **7.49% [7.34, 7.64]**, validation WER 21.26%, train CER 2.58%; final epoch 29: validation CER 7.53%. Against GW-REC-001 on the same 73,888 lines, paired: CER **−0.05 points [−0.09, −0.01]**, WER −0.33 points [−0.47, −0.19].
- **Training diagnostics:** one `GRADIENT EXPLOSION` flag at epoch 1 (step 6,477, norm 1275 > 1000), absorbed by clipping; no other flags. Epochs 0 to 16 ran under the original overfitting rule and 17 to 29 under the revised one. A selection bug first left `best.pt` at epoch 24 (7.58%). It was re-selected at epoch 27 from the run's own validation records: same CER on re-scoring, no retraining, no test data (report Notes; EXPERIMENT_PROTOCOL §11). The process aborted at exit, as GW-REC-001 did, with no effect on results.
- **Interpretation:** this matches the expectation written before the run: similar or slightly lower HierText validation CER. The paired CI excludes zero, but it covers line sampling only, not training randomness. There is one seed per arm, and two CUDA runs of one config already differ (the resume check measured up to 2.90% per-step loss difference over 400 steps; `reports/training/sanity/GW-REC-001/resume_check.json`). GW-REC-002 also took 10% more optimizer steps (187,170 against 170,116). The difference therefore cannot be credited to GlyphSynth. This is not a test of H2′a either, because H2′a concerns the document-style slice.
- **Limitations:** single seed, one mix size; more steps per epoch than GW-REC-001 (not budget-matched, EXPERIMENT_PROTOCOL §5), so a difference is not attributable to the data alone; scored on all HierText validation lines, not the document-style slice.
- **Next experiment:** Phase 12 (GW-DATA-*): several GlyphSynth sizes and seeds, budget-matched, on the document-style slice.
- **Training time:** 5.32 GPU-hours (RTX 4050 Laptop), 30 epochs, with one planned stop and resume.

---

## Entry template

```markdown
### GW-XXX-NNN — <short title>

- **Date allocated / run:** YYYY-MM-DD / YYYY-MM-DD
- **Phase:** NN
- **Git commit:** <hash>   **Config:** configs/....yaml   **Seed(s):** ...
- **Dataset version(s):** <manifest hash or version tag>
- **Hardware:** RTX 4050 Laptop 6 GB (driver ...), precision ...
- **Research question:**
- **Hypothesis (written before the run):** link to research/hypotheses.md#Hn
- **Method:**
- **Variables:** independent / controlled
- **Control:** <experiment ID of the control run>
- **Result:** (from reports/... — paste table, include CI where available)
- **Training diagnostics:** best epoch, final epoch, over/underfitting flags
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** wall clock, GPU-hours
```
