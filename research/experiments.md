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
| [GW-REC-003](#gw-rec-003) | 2026-10-23 | 07 | Phase 7 control: GW-REC-001 on the 8-epoch short budget, seed 1337 (noise floor) | planned | — | `reports/training/GW-REC-003/` |
| [GW-REC-004](#gw-rec-004) | 2026-10-23 | 07 | Phase 7 control: GW-REC-001 on the 8-epoch short budget, seed 2024 (noise floor) | planned | — | `reports/training/GW-REC-004/` |
| [GW-REC-005](#gw-rec-005) | 2026-10-23 | 07 | Phase 7 control: GW-REC-001 on the 8-epoch short budget, seed 7 (noise floor) | planned | — | `reports/training/GW-REC-005/` |
| [GW-REC-006](#gw-rec-006) | 2026-10-23 | 07 | Phase 7 ablation: input height 48 (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-006/` |
| [GW-REC-007](#gw-rec-007) | 2026-10-23 | 07 | Phase 7 ablation: CNN half as wide (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-007/` |
| [GW-REC-008](#gw-rec-008) | 2026-10-23 | 07 | Phase 7 ablation: BiLSTM hidden 128 (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-008/` |
| [GW-REC-009](#gw-rec-009) | 2026-10-23 | 07 | Phase 7 ablation: one BiLSTM layer (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-009/` |
| [GW-REC-010](#gw-rec-010) | 2026-10-23 | 07 | Phase 7 ablation: Adam (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-010/` |
| [GW-REC-011](#gw-rec-011) | 2026-10-23 | 07 | Phase 7 ablation: cosine schedule (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-011/` |
| [GW-REC-012](#gw-rec-012) | 2026-10-23 | 07 | Phase 7 ablation: learning rate 5e-4 (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-012/` |
| [GW-REC-013](#gw-rec-013) | 2026-10-23 | 07 | Phase 7 ablation: learning rate 2e-3 (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-013/` |
| [GW-REC-014](#gw-rec-014) | 2026-10-23 | 07 | Phase 7 ablation: weight decay 1e-2 (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-014/` |
| [GW-REC-015](#gw-rec-015) | 2026-10-23 | 07 | Phase 7 ablation: no augmentation (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-015/` |
| [GW-REC-016](#gw-rec-016) | 2026-10-23 | 07 | Phase 7 ablation: geometric augmentation only (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-016/` |
| [GW-REC-017](#gw-rec-017) | 2026-10-23 | 07 | Phase 7 ablation: photometric augmentation only (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-017/` |
| [GW-REC-018](#gw-rec-018) | 2026-10-23 | 07 | Phase 7 ablation: degradation augmentation only (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-018/` |
| [GW-REC-019](#gw-rec-019) | 2026-10-23 | 07 | Phase 7 ablation: frame stride 8 (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-019/` |
| [GW-REC-020](#gw-rec-020) | 2026-10-23 | 07 | GLYPHARIS-Rec-M (SVTR-style) vs CRNN at matched budget (H6, exploratory) | planned | — | `reports/training/GW-REC-020/` |

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

### GW-REC-003 — Phase 7 control, short budget, seed 1337

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-003.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** how far apart do three seeds of the GW-REC-001 configuration land on the Phase 7 short budget (8 epochs, OneCycle over those 8 epochs)?
- **Hypothesis (written before the run):** none tested. GW-REC-003, 004 and 005 are the Phase 7 control: every short-budget ablation is compared with them, and their spread is the noise floor a difference has to clear. Expectation, not a claim: the three validation CERs lie within a few tenths of a point of each other, above GW-REC-001's 7.53% because the budget is shorter.
- **Method:** `python train.py --config configs/rec/gw-rec-003.yaml`. Identical to GW-REC-001 except `train.epochs: 8` and the seed. Selection on validation CER (strict track).
- **Variables:** independent: seed; controlled: everything else
- **Control:** —
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** a short budget ranks configurations, it does not show their full-budget accuracy; the winner is confirmed at the full budget at the end of Phase 7.
- **Next experiment:** the Phase 7 ablations against this control
- **Training time:**

### GW-REC-004 — Phase 7 control, short budget, seed 2024

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-004.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** how far apart do three seeds of the GW-REC-001 configuration land on the Phase 7 short budget (8 epochs, OneCycle over those 8 epochs)?
- **Hypothesis (written before the run):** none tested. GW-REC-003, 004 and 005 are the Phase 7 control: every short-budget ablation is compared with them, and their spread is the noise floor a difference has to clear. Expectation, not a claim: the three validation CERs lie within a few tenths of a point of each other, above GW-REC-001's 7.53% because the budget is shorter.
- **Method:** `python train.py --config configs/rec/gw-rec-004.yaml`. Identical to GW-REC-001 except `train.epochs: 8` and the seed. Selection on validation CER (strict track).
- **Variables:** independent: seed; controlled: everything else
- **Control:** —
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** a short budget ranks configurations, it does not show their full-budget accuracy; the winner is confirmed at the full budget at the end of Phase 7.
- **Next experiment:** the Phase 7 ablations against this control
- **Training time:**

### GW-REC-005 — Phase 7 control, short budget, seed 7

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-005.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** how far apart do three seeds of the GW-REC-001 configuration land on the Phase 7 short budget (8 epochs, OneCycle over those 8 epochs)?
- **Hypothesis (written before the run):** none tested. GW-REC-003, 004 and 005 are the Phase 7 control: every short-budget ablation is compared with them, and their spread is the noise floor a difference has to clear. Expectation, not a claim: the three validation CERs lie within a few tenths of a point of each other, above GW-REC-001's 7.53% because the budget is shorter.
- **Method:** `python train.py --config configs/rec/gw-rec-005.yaml`. Identical to GW-REC-001 except `train.epochs: 8` and the seed. Selection on validation CER (strict track).
- **Variables:** independent: seed; controlled: everything else
- **Control:** —
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** a short budget ranks configurations, it does not show their full-budget accuracy; the winner is confirmed at the full budget at the end of Phase 7.
- **Next experiment:** the Phase 7 ablations against this control
- **Training time:**

### GW-REC-006 — Phase 7 ablation: input height 48

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-006.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, input height 48 instead of 32 (max width scaled to 1536; micro-batches of 32 with 2-step accumulation, so 64 lines per step as in the control), do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: lower validation CER than the control, because small glyphs keep more detail, at about 1.5 times the compute per line.
- **Method:** `python train.py --config configs/rec/gw-rec-006.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: input height 48 instead of 32 (max width scaled to 1536; micro-batches of 32 with 2-step accumulation, so 64 lines per step as in the control); controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget. BatchNorm sees micro-batches of 32 instead of 64.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-007 — Phase 7 ablation: CNN half as wide

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-007.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, CNN channels [32, 64, 128, 128, 256, 256] instead of [64, 128, 256, 256, 512, 512], do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: somewhat higher validation CER with far fewer parameters and faster inference; it shows how much CNN width the task needs.
- **Method:** `python train.py --config configs/rec/gw-rec-007.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: CNN channels [32, 64, 128, 128, 256, 256] instead of [64, 128, 256, 256, 512, 512]; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-008 — Phase 7 ablation: BiLSTM hidden 128

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-008.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, BiLSTM hidden size 128 per direction instead of 256, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: slightly higher validation CER with a much smaller sequence model.
- **Method:** `python train.py --config configs/rec/gw-rec-008.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: BiLSTM hidden size 128 per direction instead of 256; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-009 — Phase 7 ablation: one BiLSTM layer

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-009.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, one BiLSTM layer instead of two, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: slightly higher validation CER.
- **Method:** `python train.py --config configs/rec/gw-rec-009.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: one BiLSTM layer instead of two; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-010 — Phase 7 ablation: Adam

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-010.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, Adam instead of AdamW, same learning rate and weight decay, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: no difference beyond the seed spread, since the weight decay is small.
- **Method:** `python train.py --config configs/rec/gw-rec-010.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: Adam instead of AdamW, same learning rate and weight decay; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-011 — Phase 7 ablation: cosine schedule

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-011.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, cosine schedule with 2,000 warm-up steps instead of OneCycle (30% warm-up), do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: no difference beyond the seed spread.
- **Method:** `python train.py --config configs/rec/gw-rec-011.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: cosine schedule with 2,000 warm-up steps instead of OneCycle (30% warm-up); controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-012 — Phase 7 ablation: learning rate 5e-4

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-012.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, peak learning rate 5e-4 instead of 1e-3, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: slightly higher validation CER on the short budget (slower progress).
- **Method:** `python train.py --config configs/rec/gw-rec-012.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: peak learning rate 5e-4 instead of 1e-3; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-013 — Phase 7 ablation: learning rate 2e-3

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-013.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, peak learning rate 2e-3 instead of 1e-3, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: similar validation CER, possibly less stable; the Phase 6 LR range test put the lowest smoothed loss at 3.72e-3 (`reports/training/sanity/GW-REC-001/sanity.md`).
- **Method:** `python train.py --config configs/rec/gw-rec-013.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: peak learning rate 2e-3 instead of 1e-3; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-014 — Phase 7 ablation: weight decay 1e-2

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-014.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, weight decay 1e-2 instead of 1e-4, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: similar or slightly lower validation CER.
- **Method:** `python train.py --config configs/rec/gw-rec-014.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: weight decay 1e-2 instead of 1e-4; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-015 — Phase 7 ablation: no augmentation

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-015.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, augmentation preset none instead of full, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: lower train CER and somewhat higher validation CER than the control.
- **Method:** `python train.py --config configs/rec/gw-rec-015.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: augmentation preset none instead of full; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget. First pass only; the full augmentation study is Phase 11 (H3).
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-016 — Phase 7 ablation: geometric augmentation only

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-016.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, augmentation preset geometric instead of full, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: between none (GW-REC-015) and full.
- **Method:** `python train.py --config configs/rec/gw-rec-016.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: augmentation preset geometric instead of full; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget. First pass only; the full augmentation study is Phase 11 (H3).
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-017 — Phase 7 ablation: photometric augmentation only

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-017.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, augmentation preset photometric instead of full, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: between none (GW-REC-015) and full.
- **Method:** `python train.py --config configs/rec/gw-rec-017.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: augmentation preset photometric instead of full; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget. First pass only; the full augmentation study is Phase 11 (H3).
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-018 — Phase 7 ablation: degradation augmentation only

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-018.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, augmentation preset degradation instead of full, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: between none (GW-REC-015) and full.
- **Method:** `python train.py --config configs/rec/gw-rec-018.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: augmentation preset degradation instead of full; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget. First pass only; the full augmentation study is Phase 11 (H3).
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-019 — Phase 7 ablation: frame stride 8

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-019.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does halving the number of frames per line (a frame covers 8 input columns instead of 4) do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: similar or slightly higher validation CER, with the BiLSTM reading half as many frames (faster). Lines dense in characters may lose most, because CTC needs at least one frame per character.
- **Method:** `python train.py --config configs/rec/gw-rec-019.yaml`; the control GW-REC-003 with `model.frame_stride: 8`. Compared line by line with GW-REC-003 and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: frame stride (8 instead of 4); controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-020 — GLYPHARIS-Rec-M vs CRNN at matched budget (H6)

- **Date allocated / run:** 2026-10-23 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-020.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** does the SVTR-style encoder reach lower validation CER than the CRNN at the same training budget?
- **Hypothesis (written before the run):** [H6](hypotheses.md): GLYPHARIS-Rec-M reaches lower validation CER than the CRNN at equal training steps, with CPU latency within 1.5 times the CRNN's. This run is the exploratory single-seed test on the short budget; H6's latency half is measured separately on a quiet machine.
- **Method:** the sanity sequence first (`scripts/rec_sanity.py --config configs/rec/gw-rec-020.yaml`: overfit 64 lines, LR range test, dry run with peak VRAM), then `python train.py --config configs/rec/gw-rec-020.yaml`. Same data, batches, steps, schedule and learning rate as the control GW-REC-003; about as many parameters as the CRNN (both counts are in the run reports). If the LR range test puts the control's learning rate far outside the encoder's useful range, the learning rate is changed here, with the reason, before the main run.
- **Variables:** independent: encoder (SVTR-style vs CRNN); controlled: data, budget, schedule, optimizer
- **Amendment (2026-10-24, before the main run):** the first sanity attempt (`reports/training/sanity/GW-REC-020/attempt1/`) found three things. The LR range test put the lowest smoothed loss at 3.24e-3 (suggestion 3.24e-4), so the control's 1e-3 was outside the 3x band the CRNN met. The overfit check stopped at 12.5% CER after 1,500 steps, still falling fast after a long all-blank plateau. The dry run ran out of GPU memory at the control's batch. The run therefore uses learning rate 5e-4, which is inside the band, chosen by the same procedure that checked the CRNN's rate. It uses micro-batches of 32 lines with 2-step accumulation and a column cap of 16,384: still 64 lines per optimizer step, with wide batches split at the same widths as the control's. The sanity sequence is re-run with these settings and a 4,000-step overfit allowance before the main run. The learning rate now differs between the arms, so the comparison is CRNN at its checked rate against Rec-M at its checked rate.
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget; a transformer may need a longer schedule than a CRNN, so a short-budget loss for Rec-M is weaker evidence against H6 than a win would be for it.
- **Next experiment:** if it is competitive, three seeds at the full budget
- **Training time:**

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
