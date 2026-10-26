# Experiment Registry

Every experiment ID is allocated here **before** the run starts (see `docs/WORKFLOW.md` §4).
Status values: `planned` → `running` → `done` | `failed` | `abandoned`.

Rows link to a full entry below. Numbers are copied from generated reports only, never typed by hand.

| ID | Date allocated | Phase | Question / hypothesis | Status | Key result | Report |
|---|---|---|---|---|---|---|
| [GW-BASE-001](#gw-base-001) | 2026-10-21 | 05 | Tesseract 5 (LSTM, eng): recognition and end to end | done | lines 24.35% CER (HierText val); 0.43 s per page | `reports/baselines/` |
| [GW-BASE-002](#gw-base-002) | 2026-10-21 | 05 | PaddleOCR PP-OCRv5 mobile (det + English rec): recognition and end to end | done | lines 9.27% CER (HierText val); 3.4 s per page | `reports/baselines/` |
| [GW-BASE-003](#gw-base-003) | 2026-10-21 | 05 | PaddleOCR PP-OCRv5 server (det + rec): recognition and end to end | done | lines 14.65% CER (HierText val); 80.9 s per page; GDC pages not runnable | `reports/baselines/` |
| [GW-BASE-004](#gw-base-004) | 2026-10-21 | 05 | TrOCR-small-printed: recognition only | done | lines 62.58% strict / 19.14% normalised CER (HierText val); 133 ms per line | `reports/baselines/` |
| [GW-BASE-005](#gw-base-005) | 2026-10-21 | 05 | PaddleOCR PP-OCRv6 tiny (det + rec): recognition and end to end | done | lines 11.10% CER (HierText val); 21.6 ms per line, 1.7 s per page | `reports/baselines/` |
| [GW-BASE-006](#gw-base-006) | 2026-10-21 | 05 | PaddleOCR PP-OCRv6 small (det + rec): recognition and end to end | done | lines 7.53% CER (HierText val); 46 ms per line, 3.7 s per page | `reports/baselines/` |
| [GW-BASE-007](#gw-base-007) | 2026-10-21 | 05 | PaddleOCR PP-OCRv6 medium (PaddleOCR 3.7's English default): reduced scope | done | lines 7.54% CER (HierText val); 553 ms per line, 32.5 s per page | `reports/baselines/` |
| [GW-REC-001](#gw-rec-001) | 2026-10-22 | 06 | CRNN baseline on HierText lines | done | val CER 7.53% [7.38, 7.69] | `reports/training/GW-REC-001/` |
| [GW-REC-002](#gw-rec-002) | 2026-10-22 | 06 | GW-REC-001 plus 35,000 GlyphSynth lines (exploratory look at H2′a) | done | val CER 7.49% [7.34, 7.64]; −0.05 points [−0.09, −0.01] vs GW-REC-001 (one seed, not budget-matched) | `reports/training/GW-REC-002/` |
| [GW-REC-003](#gw-rec-003) | 2026-10-23 | 07 | Phase 7 control: GW-REC-001 on the 8-epoch short budget, seed 1337 (noise floor) | done | val CER 8.32% [8.16, 8.47] (control) | `reports/training/GW-REC-003/` |
| [GW-REC-004](#gw-rec-004) | 2026-10-23 | 07 | Phase 7 control: GW-REC-001 on the 8-epoch short budget, seed 2024 (noise floor) | done | val CER 8.30% [8.15, 8.47] (control) | `reports/training/GW-REC-004/` |
| [GW-REC-005](#gw-rec-005) | 2026-10-23 | 07 | Phase 7 control: GW-REC-001 on the 8-epoch short budget, seed 7 (noise floor) | done | val CER 8.31% [8.15, 8.47] (control) | `reports/training/GW-REC-005/` |
| [GW-REC-006](#gw-rec-006) | 2026-10-23 | 07 | Phase 7 ablation: input height 48 (short budget, vs GW-REC-003) | done | val CER 7.83% [7.66, 7.98]; -0.49 [-0.54, -0.44] vs GW-REC-003 (better) | `reports/training/GW-REC-006/` |
| [GW-REC-007](#gw-rec-007) | 2026-10-23 | 07 | Phase 7 ablation: CNN half as wide (short budget, vs GW-REC-003) | done | val CER 8.83% [8.65, 9.00]; +0.51 [+0.47, +0.56] vs GW-REC-003 (worse) | `reports/training/GW-REC-007/` |
| [GW-REC-008](#gw-rec-008) | 2026-10-23 | 07 | Phase 7 ablation: BiLSTM hidden 128 (short budget, vs GW-REC-003) | done | val CER 8.78% [8.60, 8.94]; +0.46 [+0.42, +0.51] vs GW-REC-003 (worse) | `reports/training/GW-REC-008/` |
| [GW-REC-009](#gw-rec-009) | 2026-10-23 | 07 | Phase 7 ablation: one BiLSTM layer (short budget, vs GW-REC-003) | done | val CER 8.78% [8.60, 8.94]; +0.46 [+0.42, +0.51] vs GW-REC-003 (worse) | `reports/training/GW-REC-009/` |
| [GW-REC-010](#gw-rec-010) | 2026-10-23 | 07 | Phase 7 ablation: Adam (short budget, vs GW-REC-003) | done | val CER 9.17% [8.99, 9.34]; +0.85 [+0.81, +0.90] vs GW-REC-003 (worse) | `reports/training/GW-REC-010/` |
| [GW-REC-011](#gw-rec-011) | 2026-10-23 | 07 | Phase 7 ablation: cosine schedule (short budget, vs GW-REC-003) | done | val CER 8.44% [8.26, 8.60]; +0.12 [+0.08, +0.17] vs GW-REC-003 (worse) | `reports/training/GW-REC-011/` |
| [GW-REC-012](#gw-rec-012) | 2026-10-23 | 07 | Phase 7 ablation: learning rate 5e-4 (short budget, vs GW-REC-003) | done | val CER 8.51% [8.34, 8.67]; +0.19 [+0.15, +0.23] vs GW-REC-003 (worse) | `reports/training/GW-REC-012/` |
| [GW-REC-013](#gw-rec-013) | 2026-10-23 | 07 | Phase 7 ablation: learning rate 2e-3 (short budget, vs GW-REC-003) | done | val CER 8.34% [8.17, 8.50]; +0.02 [-0.02, +0.06] vs GW-REC-003 (no clear difference) | `reports/training/GW-REC-013/` |
| [GW-REC-014](#gw-rec-014) | 2026-10-23 | 07 | Phase 7 ablation: weight decay 1e-2 (short budget, vs GW-REC-003) | done | val CER 8.28% [8.11, 8.44]; -0.04 [-0.08, -0.00] vs GW-REC-003 (better) | `reports/training/GW-REC-014/` |
| [GW-REC-015](#gw-rec-015) | 2026-10-23 | 07 | Phase 7 ablation: no augmentation (short budget, vs GW-REC-003) | done | val CER 8.68% [8.50, 8.84]; +0.36 [+0.32, +0.41] vs GW-REC-003 (worse) | `reports/training/GW-REC-015/` |
| [GW-REC-016](#gw-rec-016) | 2026-10-23 | 07 | Phase 7 ablation: geometric augmentation only (short budget, vs GW-REC-003) | done | val CER 8.34% [8.16, 8.50]; +0.02 [-0.02, +0.06] vs GW-REC-003 (no clear difference) | `reports/training/GW-REC-016/` |
| [GW-REC-017](#gw-rec-017) | 2026-10-23 | 07 | Phase 7 ablation: photometric augmentation only (short budget, vs GW-REC-003) | done | val CER 8.59% [8.42, 8.75]; +0.28 [+0.23, +0.32] vs GW-REC-003 (worse) | `reports/training/GW-REC-017/` |
| [GW-REC-018](#gw-rec-018) | 2026-10-23 | 07 | Phase 7 ablation: degradation augmentation only (short budget, vs GW-REC-003) | done | val CER 8.68% [8.50, 8.85]; +0.36 [+0.32, +0.41] vs GW-REC-003 (worse) | `reports/training/GW-REC-018/` |
| [GW-REC-019](#gw-rec-019) | 2026-10-23 | 07 | Phase 7 ablation: frame stride 8 (short budget, vs GW-REC-003) | failed | out of GPU memory at step 1,444 (see failures.md); re-run as GW-REC-021 | `reports/training/GW-REC-019/` |
| [GW-REC-020](#gw-rec-020) | 2026-10-23 | 07 | GLYPHARIS-Rec-M (SVTR-style) vs CRNN at matched budget (H6, exploratory) | done | val CER 8.18% [8.01, 8.35]; -0.13 [-0.18, -0.09] vs GW-REC-003 (better) | `reports/training/GW-REC-020/` |
| [GW-REC-021](#gw-rec-021) | 2026-10-24 | 07 | Phase 7 ablation: frame stride 8 with CTC-aware batching (re-run of GW-REC-019; short budget, vs GW-REC-003) | done | val CER 10.93% [10.70, 11.16]; +2.61 [+2.48, +2.74] vs GW-REC-003 (worse) | `reports/training/GW-REC-021/` |
| [GW-REC-022](#gw-rec-022) | 2026-10-25 | 07 | Phase 7 combination: GLYPHARIS-Rec-M (SVTR) at input height 48 with weight decay 1e-2 (short budget, vs GW-REC-003) | planned | — | `reports/training/GW-REC-022/` |
| [GW-REC-023](#gw-rec-023) | 2026-10-25 | 07 | Phase 7 combination without the encoder change: CRNN at input height 48 with weight decay 1e-2 (short budget, vs GW-REC-003) | abandoned | not run: GW-REC-022 settled the question (see entry) | `reports/training/GW-REC-023/` |
| [GW-REC-024](#gw-rec-024) | 2026-10-26 | 07 | Phase 7 full-budget confirmation: GW-REC-022's recipe at GW-REC-001's budget (30 epochs, one seed), vs GW-REC-001 | done | val CER 6.46% [6.33, 6.60]; -1.07 [-1.13, -1.01] vs GW-REC-001 | `reports/training/GW-REC-024/` |

### GW-BASE-001 — Tesseract 5 (LSTM, eng): recognition and end to end

- **System:** Tesseract 5 (UB Mannheim build), `-l eng`, default engine; line crops with `--psm 7`, pages with the default page mode (`--psm 3`).
- **Date allocated / run:** 2026-10-21 / 2026-10-22 to 2026-10-24
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** Strict track unless stated, 95% CI in brackets; every table in `reports/baselines/`. Lines: HierText val 24.35% CER [23.40, 25.42], GlyphSynth val 1.55% [1.39, 1.71], GDC test 6.42% [5.91, 6.95], HierText test 24.28% [23.89, 24.64]. Pages: HierText val detection recall 10.29% at IoU 0.5 and E2E F1 12.65% [10.17, 15.41]; GDC test E2E F1 46.11% [38.03, 55.25], order-aware page CER 31.90% [25.66, 38.51]. Speed (HierText val, batch 1): 141.8 ms per line (p95 303.7), 430.8 ms per page (p95 807.1); peak RAM 113 MiB on lines, 281 MiB on pages; model 3.9 MiB.
- **Training diagnostics:** not applicable (no training)
- **Interpretation:** Accurate on clean rendered text (GlyphSynth, GDC) and the fastest page pipeline measured, but its page analysis finds about one HierText line in ten, so its HierText page scores mostly count missed lines. On GDC it has the best reading order of all systems: order-aware page CER 31.90% against 42.86–44.74% for the PaddleOCR pipelines, which return lines in detection order.
- **Limitations:** Called through its command line: about 96 ms of every call is process start and model load (`efficiency.md`), so its line times overstate the engine's own cost. Its peak memory adds two peaks that need not coincide, so it is an upper bound.
- **Next experiment:** reference point for GLYPHARIS-Rec (Phases 6–7) and the full pipeline (Phase 10).
- **Training time:** not applicable

### GW-BASE-002 — PaddleOCR PP-OCRv5 mobile (det + English rec): recognition and end to end

- **System:** PaddleOCR 3.7 with PP-OCRv5_mobile_det and en_PP-OCRv5_mobile_rec, CPU. Pipeline defaults except: document unwarping off (it moves the output polygons into the unwarped image's frame), and the text detector runs without oneDNN because Paddle 3.3's oneDNN path crashes on the PP-OCRv5 detectors on this machine. Recognizer outputs were checked to be identical with oneDNN on and off (64 of 64 GlyphSynth-val lines), so this is a speed-only change.
- **Date allocated / run:** 2026-10-21 / 2026-10-21 to 2026-10-24
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** Strict track unless stated, 95% CI in brackets; every table in `reports/baselines/`. Lines: HierText val 9.27% [8.63, 9.96], GlyphSynth val 18.06% [17.24, 18.90], GDC test 3.68% [3.40, 3.95], HierText test 9.23% [8.98, 9.50]. Pages: HierText val E2E F1 38.75% [34.89, 42.57], bag-of-lines page CER 50.84%; GDC test E2E F1 59.80% [50.80, 68.56]. Speed: 61.4 ms per line (p95 184.3), 3,408.2 ms per page (p95 5,802.8); peak RAM 924 / 1,960 MiB (lines / pages); model 7.6 MiB (recognizer), 25.4 MiB (pipeline).
- **Training diagnostics:** not applicable (no training)
- **Interpretation:** Good on real lines (HierText, GDC) but poor on GlyphSynth, because it fails on long lines: its output keeps full length below 30 characters and drops to 0.76 of the reference length at 70 characters or more, with 1,201 of 10,000 outputs cut short (`diagnostics.md`); the other recognizers stay at 0.98–1.00. GlyphSynth lines are long by construction. The cause was not investigated.
- **Limitations:** Detector run without oneDNN (speed only; outputs checked identical). Lines come back in detection order, which costs it on GDC's order-aware metrics.
- **Next experiment:** reference point for GLYPHARIS-Rec (Phases 6–7) and the full pipeline (Phase 10).
- **Training time:** not applicable

### GW-BASE-003 — PaddleOCR PP-OCRv5 server (det + rec): recognition and end to end

- **System:** PaddleOCR 3.7 with PP-OCRv5_server_det and PP-OCRv5_server_rec (no English-only server recognizer exists), CPU, same pipeline settings as GW-BASE-002. **Reduced scope:** the server recognizer needs about 1–3 s per line on this CPU, so it runs on GDC test (recognition and end to end) and HierText validation (recognition) only. HierText test and GlyphSynth val would add roughly 45 CPU-hours and are recorded as not run.
- **Date allocated / run:** 2026-10-21 / 2026-10-24 to 2026-10-25
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** Strict track unless stated, 95% CI in brackets; every table in `reports/baselines/`. Lines: HierText val 14.65% [13.79, 15.52] (WER 43.80%), GDC test 11.24% [10.71, 11.80] (WER 48.68%). Pages: not run on HierText (reduced scope), not runnable on GDC (see Limitations). Speed: 2,266.7 ms per line (p95 3,199.7), 80,861.6 ms per page (p95 148,634.4) on HierText val; peak RAM 1,914 / 9,790 MiB; model 81.3 / 178.6 MiB.
- **Training diagnostics:** not applicable (no training)
- **Interpretation:** Less accurate than the smaller English mobile model on both line sets. Its WER is three to four times its CER because the multilingual recognizer drops many word gaps: 70.3% of its multi-word outputs keep a space (85.0–92.4% for the other systems) and 760 of 10,000 HierText lines are wrong only in their spaces (`diagnostics.md`). It is also by far the slowest system, at 81 s per page on average.
- **Limitations:** the end-to-end run on the GDC test pages could not be run on the laptop: two attempts died on the first page (out of memory next to a GPU job; an access violation alone). See `research/failures.md`, 2026-10-24. Page speed was measured on HierText validation although its page accuracy was not (too slow to run in full). On the largest pages it needed more memory than the laptop had free and part of it was paged to disk, so some page times include paging; the 9,790 MiB peak is the working set, below what the process reserved.
- **Next experiment:** reference point for GLYPHARIS-Rec (Phases 6–7) and the full pipeline (Phase 10).
- **Training time:** not applicable

### GW-BASE-004 — TrOCR-small-printed: recognition only

- **System:** microsoft/trocr-small-printed, transformers on CPU, greedy decoding as in the checkpoint, with `max_new_tokens=256` because the checkpoint sets no length and transformers would otherwise stop at 20 tokens and cut long lines, and with the decoder cache on: the checkpoint ships `use_cache: false`, which re-runs the whole prefix at every step. A one-off setup check found identical outputs on 64 validation lines and about 2.3x faster decoding (an informal check, not a reported result; the profiling pass measures speed). It has no detector, so it is run on recognition only.
- **Date allocated / run:** 2026-10-21 / 2026-10-23 to 2026-10-25
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** Strict track unless stated, 95% CI in brackets; every table in `reports/baselines/`. Lines, strict / normalised (case-folded): HierText val 62.58% / 19.14%, GlyphSynth val 70.75% / 7.27%, GDC test 72.34% / 13.07%, HierText test 64.00% / 20.21%. Speed: 133.3 ms per line (p95 183.6); peak RAM 701 MiB; model 235.8 MiB.
- **Training diagnostics:** not applicable (no training)
- **Interpretation:** The strict scores mostly measure letter case: it writes 96.4% of lines without a lowercase letter where the reference has some (`diagnostics.md`); the checkpoint is fine-tuned on receipts. Case-folded, it is still behind every PaddleOCR system on real lines (19.14% on HierText val) though competitive on GlyphSynth (7.27%), and it has the largest model of the line recognizers.
- **Limitations:** Recognition only (no detector).
- **Next experiment:** reference point for GLYPHARIS-Rec (Phases 6–7) and the full pipeline (Phase 10).
- **Training time:** not applicable

### GW-BASE-005 — PaddleOCR PP-OCRv6 tiny (det + rec): recognition and end to end

- **System:** PaddleOCR 3.7 with PP-OCRv6_tiny_det and PP-OCRv6_tiny_rec (multilingual recognizer: Chinese, English, Latin scripts), CPU, same pipeline settings and deviations as GW-BASE-002 (the v6 detectors crash under oneDNN too).
- **Date allocated / run:** 2026-10-21 / 2026-10-22 to 2026-10-24
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** Strict track unless stated, 95% CI in brackets; every table in `reports/baselines/`. Lines: HierText val 11.10% [10.58, 11.65], GlyphSynth val 1.63% [1.52, 1.74], GDC test 4.00% [3.71, 4.34], HierText test 10.82% [10.61, 11.02]. Pages: HierText val E2E F1 38.69% [34.84, 42.43]; GDC test E2E F1 60.94% [49.18, 72.24]. Speed: 21.6 ms per line (p95 66.4), 1,726.6 ms per page (p95 2,729.0); peak RAM 819 / 1,836 MiB; model 4.4 / 19.3 MiB.
- **Training diagnostics:** not applicable (no training)
- **Interpretation:** The fastest recognizer measured and the fastest PaddleOCR page pipeline, with accuracy between PP-OCRv5 mobile and PP-OCRv6 small. It is the existing system closest to what GLYPHARIS aims at (small, CPU first), so it is the speed reference for the CPU latency target.
- **Limitations:** Multilingual recognizer (Chinese, English, Latin scripts) scored on English data.
- **Next experiment:** reference point for GLYPHARIS-Rec (Phases 6–7) and the full pipeline (Phase 10).
- **Training time:** not applicable

### GW-BASE-006 — PaddleOCR PP-OCRv6 small (det + rec): recognition and end to end

- **System:** PaddleOCR 3.7 with PP-OCRv6_small_det and PP-OCRv6_small_rec, CPU, same settings as GW-BASE-005.
- **Date allocated / run:** 2026-10-21 / 2026-10-23 to 2026-10-24
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** Strict track unless stated, 95% CI in brackets; every table in `reports/baselines/`. Lines: HierText val 7.53% [7.01, 8.06], GlyphSynth val 0.99% [0.86, 1.12], GDC test 3.46% [3.13, 3.84], HierText test 7.06% [6.88, 7.26]. Pages: HierText val E2E F1 45.68% [42.16, 49.14]; GDC test E2E F1 63.13% [51.70, 73.98]. Speed: 46.0 ms per line (p95 114.4), 3,714.2 ms per page (p95 6,308.3); peak RAM 1,000 / 2,051 MiB; model 20.4 / 43.1 MiB.
- **Training diagnostics:** not applicable (no training)
- **Interpretation:** The most accurate full-scope system on every set. On HierText val lines it matches PP-OCRv6 medium (7.53% against 7.54%) at about a twelfth of the line time, so it is the accuracy reference for GLYPHARIS.
- **Limitations:** Multilingual recognizer, as GW-BASE-005.
- **Next experiment:** reference point for GLYPHARIS-Rec (Phases 6–7) and the full pipeline (Phase 10).
- **Training time:** not applicable

### GW-BASE-007 — PaddleOCR PP-OCRv6 medium (PaddleOCR 3.7's English default): reduced scope

- **System:** PaddleOCR 3.7 with PP-OCRv6_medium_det and PP-OCRv6_medium_rec, which is what `PaddleOCR(lang="en")` loads by default in 3.7. CPU, same settings as GW-BASE-005. **Reduced scope** for the same reason as GW-BASE-003 (about 1 s per line): GDC test recognition and end to end, HierText validation recognition.
- **Date allocated / run:** 2026-10-21 / 2026-10-23 to 2026-10-24
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Recognition crops come from the evaluation packs of `scripts/build_eval_crops.py`, which keep lines with characters outside the GLYPHARIS charset (EXPERIMENT_PROTOCOL §2; the charset ceiling is reported). HierText validation: 10,000 seeded line crops and 400 seeded pages; GlyphSynth validation and the test sets in full (test access logged in `research/test_access_log.md`, scoring included). Speed and memory come from a separate `--profile` pass on validation data (10 warm-up, 100 timed, batch 1).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** Strict track unless stated, 95% CI in brackets; every table in `reports/baselines/`. Lines: HierText val 7.54% [6.94, 8.16], GDC test 3.20% [2.91, 3.56]. Pages: GDC test E2E F1 65.47% [53.61, 76.54], bag-of-lines page CER 20.96%. Speed: 552.5 ms per line (p95 971.7), 32,528.5 ms per page (p95 56,305.4); peak RAM 1,575 / 3,016 MiB; model 73.3 / 145.7 MiB.
- **Training diagnostics:** not applicable (no training)
- **Interpretation:** The most accurate system on GDC, but its CIs overlap PP-OCRv6 small's on every set both ran, and it is about 12 times slower per line and 9 times slower per page. PaddleOCR's English default is a poor fit for CPU use.
- **Limitations:** Reduced scope: no HierText pages, HierText test or GlyphSynth.
- **Next experiment:** reference point for GLYPHARIS-Rec (Phases 6–7) and the full pipeline (Phase 10).
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
- **Result:** val CER 8.32% [8.16, 8.47] (95% bootstrap CI over 73,888 lines). The three control seeds: 8.32%, 8.30%, 8.31%; mean 8.31%, standard deviation 0.01 points. 8 epochs, 45,360 optimizer steps, the final epoch was the best.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-003/report.md`)
- **Interpretation:** The three control seeds land within 0.02 points of each other, a much tighter spread than the expected few tenths, so the decision rule's "outside every control seed" condition is a strict bar only when combined with the paired CI. As expected, the short budget ends above GW-REC-001's full-budget 7.53%.
- **Limitations:** a short budget ranks configurations, it does not show their full-budget accuracy; the winner is confirmed at the full budget at the end of Phase 7.
- **Next experiment:** the Phase 7 ablations against this control
- **Training time:** 1.34 GPU-hours (RTX 4050 Laptop, bf16)

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
- **Result:** val CER 8.30% [8.15, 8.47] (95% bootstrap CI over 73,888 lines). The three control seeds: 8.32%, 8.30%, 8.31%; mean 8.31%, standard deviation 0.01 points. 8 epochs, 45,368 optimizer steps, the final epoch was the best.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-004/report.md`)
- **Interpretation:** Second control seed; see GW-REC-003.
- **Limitations:** a short budget ranks configurations, it does not show their full-budget accuracy; the winner is confirmed at the full budget at the end of Phase 7.
- **Next experiment:** the Phase 7 ablations against this control
- **Training time:** 1.31 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-005 — Phase 7 control, short budget, seed 7

- **Date allocated / run:** 2026-10-23 / 2026-10-23 to 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-005.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** how far apart do three seeds of the GW-REC-001 configuration land on the Phase 7 short budget (8 epochs, OneCycle over those 8 epochs)?
- **Hypothesis (written before the run):** none tested. GW-REC-003, 004 and 005 are the Phase 7 control: every short-budget ablation is compared with them, and their spread is the noise floor a difference has to clear. Expectation, not a claim: the three validation CERs lie within a few tenths of a point of each other, above GW-REC-001's 7.53% because the budget is shorter.
- **Method:** `python train.py --config configs/rec/gw-rec-005.yaml`. Identical to GW-REC-001 except `train.epochs: 8` and the seed. Selection on validation CER (strict track).
- **Variables:** independent: seed; controlled: everything else
- **Control:** —
- **Result:** val CER 8.31% [8.15, 8.47] (95% bootstrap CI over 73,888 lines). The three control seeds: 8.32%, 8.30%, 8.31%; mean 8.31%, standard deviation 0.01 points. 8 epochs, 45,367 optimizer steps, the final epoch was the best.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-005/report.md`)
- **Interpretation:** Third control seed; see GW-REC-003.
- **Limitations:** a short budget ranks configurations, it does not show their full-budget accuracy; the winner is confirmed at the full budget at the end of Phase 7.
- **Next experiment:** the Phase 7 ablations against this control
- **Training time:** 1.31 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-006 — Phase 7 ablation: input height 48

- **Date allocated / run:** 2026-10-23 / 2026-10-23 to 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-006.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, input height 48 instead of 32 (max width scaled to 1536; micro-batches of 32 with 2-step accumulation, so 64 lines per step as in the control), do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: lower validation CER than the control, because small glyphs keep more detail, at about 1.5 times the compute per line.
- **Method:** `python train.py --config configs/rec/gw-rec-006.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: input height 48 instead of 32 (max width scaled to 1536; micro-batches of 32 with 2-step accumulation, so 64 lines per step as in the control); controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 7.83% [7.66, 7.98]; against GW-REC-003 -0.49 points [-0.54, -0.44] (paired bootstrap over lines, 1,000 resamples): **better** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,339 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-006/report.md`)
- **Interpretation:** The largest gain of any single change: a taller input gives small and dense text more pixels per character. It costs about twice the training time of the control at the same parameter count.
- **Limitations:** single seed; short budget. BatchNorm sees micro-batches of 32 instead of 64.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 2.44 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-007 — Phase 7 ablation: CNN half as wide

- **Date allocated / run:** 2026-10-23 / 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-007.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, CNN channels [32, 64, 128, 128, 256, 256] instead of [64, 128, 256, 256, 512, 512], do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: somewhat higher validation CER with far fewer parameters and faster inference; it shows how much CNN width the task needs.
- **Method:** `python train.py --config configs/rec/gw-rec-007.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: CNN channels [32, 64, 128, 128, 256, 256] instead of [64, 128, 256, 256, 512, 512]; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.83% [8.65, 9.00]; against GW-REC-003 +0.51 points [+0.47, +0.56] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 3,897,075 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-007/report.md`)
- **Interpretation:** Halving the CNN width halves the parameters but costs half a point; with GW-REC-008 and 009, every capacity reduction tried costs 0.46 to 0.51 points at this budget.
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 0.95 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-008 — Phase 7 ablation: BiLSTM hidden 128

- **Date allocated / run:** 2026-10-23 / 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-008.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, BiLSTM hidden size 128 per direction instead of 256, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: slightly higher validation CER with a much smaller sequence model.
- **Method:** `python train.py --config configs/rec/gw-rec-008.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: BiLSTM hidden size 128 per direction instead of 256; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.78% [8.60, 8.94]; against GW-REC-003 +0.46 points [+0.42, +0.51] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 5,624,787 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-008/report.md`)
- **Interpretation:** A smaller BiLSTM costs about as much as a narrower CNN or a single layer (GW-REC-007, 009): the CRNN has no obviously redundant capacity at this budget.
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 1.52 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-009 — Phase 7 ablation: one BiLSTM layer

- **Date allocated / run:** 2026-10-23 / 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-009.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, one BiLSTM layer instead of two, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: slightly higher validation CER.
- **Method:** `python train.py --config configs/rec/gw-rec-009.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: one BiLSTM layer instead of two; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.78% [8.60, 8.94]; against GW-REC-003 +0.46 points [+0.42, +0.51] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 6,219,475 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-009/report.md`)
- **Interpretation:** One BiLSTM layer instead of two costs as much as halving its width (GW-REC-008).
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 1.45 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-010 — Phase 7 ablation: Adam

- **Date allocated / run:** 2026-10-23 / 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-010.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, Adam instead of AdamW, same learning rate and weight decay, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: no difference beyond the seed spread, since the weight decay is small.
- **Method:** `python train.py --config configs/rec/gw-rec-010.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: Adam instead of AdamW, same learning rate and weight decay; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 9.17% [8.99, 9.34]; against GW-REC-003 +0.85 points [+0.81, +0.90] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-010/report.md`)
- **Interpretation:** The worst arm: Adam with coupled L2 weight decay is clearly behind AdamW at the same learning rate and schedule.
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 1.53 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-011 — Phase 7 ablation: cosine schedule

- **Date allocated / run:** 2026-10-23 / 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-011.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, cosine schedule with 2,000 warm-up steps instead of OneCycle (30% warm-up), do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: no difference beyond the seed spread.
- **Method:** `python train.py --config configs/rec/gw-rec-011.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: cosine schedule with 2,000 warm-up steps instead of OneCycle (30% warm-up); controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.44% [8.26, 8.60]; against GW-REC-003 +0.12 points [+0.08, +0.17] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-011/report.md`)
- **Interpretation:** Cosine with warm-up is slightly behind OneCycle at this budget; the difference is small but its CI excludes zero.
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 1.40 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-012 — Phase 7 ablation: learning rate 5e-4

- **Date allocated / run:** 2026-10-23 / 2026-10-25
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-012.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, peak learning rate 5e-4 instead of 1e-3, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: slightly higher validation CER on the short budget (slower progress).
- **Method:** `python train.py --config configs/rec/gw-rec-012.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: peak learning rate 5e-4 instead of 1e-3; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.51% [8.34, 8.67]; against GW-REC-003 +0.19 points [+0.15, +0.23] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-012/report.md`)
- **Interpretation:** Half the learning rate undertrains in 8 epochs.
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 1.00 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-013 — Phase 7 ablation: learning rate 2e-3

- **Date allocated / run:** 2026-10-23 / 2026-10-25
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-013.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, peak learning rate 2e-3 instead of 1e-3, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: similar validation CER, possibly less stable; the Phase 6 LR range test put the lowest smoothed loss at 3.72e-3 (`reports/training/sanity/GW-REC-001/sanity.md`).
- **Method:** `python train.py --config configs/rec/gw-rec-013.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: peak learning rate 2e-3 instead of 1e-3; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.34% [8.17, 8.50]; against GW-REC-003 +0.02 points [-0.02, +0.06] (paired bootstrap over lines, 1,000 resamples): **no clear difference** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-013/report.md`)
- **Interpretation:** Twice the learning rate lands on the control: together with GW-REC-012, the configured 1e-3 sits on a flat stretch between 1e-3 and 2e-3. It makes the model neither smaller nor faster, so the combination step leaves it out.
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 0.96 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-014 — Phase 7 ablation: weight decay 1e-2

- **Date allocated / run:** 2026-10-23 / 2026-10-25
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-014.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, weight decay 1e-2 instead of 1e-4, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: similar or slightly lower validation CER.
- **Method:** `python train.py --config configs/rec/gw-rec-014.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: weight decay 1e-2 instead of 1e-4; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.28% [8.11, 8.44]; against GW-REC-003 -0.04 points [-0.08, -0.00] (paired bootstrap over lines, 1,000 resamples): **better** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-014/report.md`)
- **Interpretation:** Weight decay 1e-2 instead of 1e-4 is a small gain whose CI only just excludes zero, but it lies below all three control seeds, so the rule counts it as better and the combination (GW-REC-022) takes it.
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 0.97 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-015 — Phase 7 ablation: no augmentation

- **Date allocated / run:** 2026-10-23 / 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-015.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, augmentation preset none instead of full, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: lower train CER and somewhat higher validation CER than the control.
- **Method:** `python train.py --config configs/rec/gw-rec-015.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: augmentation preset none instead of full; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.68% [8.50, 8.84]; against GW-REC-003 +0.36 points [+0.32, +0.41] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-015/report.md`)
- **Interpretation:** Training without augmentation costs 0.36 points, so augmentation helps even on the validation split of the same dataset.
- **Limitations:** single seed; short budget. First pass only; the full augmentation study is Phase 11 (H3).
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 1.16 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-016 — Phase 7 ablation: geometric augmentation only

- **Date allocated / run:** 2026-10-23 / 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-016.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, augmentation preset geometric instead of full, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: between none (GW-REC-015) and full.
- **Method:** `python train.py --config configs/rec/gw-rec-016.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: augmentation preset geometric instead of full; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.34% [8.16, 8.50]; against GW-REC-003 +0.02 points [-0.02, +0.06] (paired bootstrap over lines, 1,000 resamples): **no clear difference** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-016/report.md`)
- **Interpretation:** Geometric augmentation alone matches the full preset: it carries the benefit (compare GW-REC-015, 017, 018). It does not make the model smaller or faster, so the combination step leaves it out; the full augmentation study is Phase 11.
- **Limitations:** single seed; short budget. First pass only; the full augmentation study is Phase 11 (H3).
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 1.39 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-017 — Phase 7 ablation: photometric augmentation only

- **Date allocated / run:** 2026-10-23 / 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-017.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, augmentation preset photometric instead of full, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: between none (GW-REC-015) and full.
- **Method:** `python train.py --config configs/rec/gw-rec-017.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: augmentation preset photometric instead of full; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.59% [8.42, 8.75]; against GW-REC-003 +0.28 points [+0.23, +0.32] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-017/report.md`)
- **Interpretation:** Photometric augmentation alone recovers only part of the benefit of the full preset.
- **Limitations:** single seed; short budget. First pass only; the full augmentation study is Phase 11 (H3).
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 1.16 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-018 — Phase 7 ablation: degradation augmentation only

- **Date allocated / run:** 2026-10-23 / 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-018.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** what does changing one thing, augmentation preset degradation instead of full, do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** expectation, not a claim: between none (GW-REC-015) and full.
- **Method:** `python train.py --config configs/rec/gw-rec-018.yaml`; the control GW-REC-003 with only this change. Compared line by line with GW-REC-003 (`scripts/rec_compare.py`) and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: augmentation preset degradation instead of full; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.68% [8.50, 8.85]; against GW-REC-003 +0.36 points [+0.32, +0.41] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,360 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-018/report.md`)
- **Interpretation:** Degradation augmentation alone is no better than no augmentation (GW-REC-015): on HierText validation it adds nothing by itself.
- **Limitations:** single seed; short budget. First pass only; the full augmentation study is Phase 11 (H3).
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 1.18 GPU-hours (RTX 4050 Laptop, bf16)

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
- **Result:** failed. Out of GPU memory at step 1,444 of epoch 1, three times at the same batch (64 lines padded to 3,264 columns); see `research/failures.md`. No result; re-run as GW-REC-021.
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:**

### GW-REC-020 — GLYPHARIS-Rec-M vs CRNN at matched budget (H6)

- **Date allocated / run:** 2026-10-23 / 2026-10-23 to 2026-10-24
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-020.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** does the SVTR-style encoder reach lower validation CER than the CRNN at the same training budget?
- **Hypothesis (written before the run):** [H6](hypotheses.md): GLYPHARIS-Rec-M reaches lower validation CER than the CRNN at equal training steps, with CPU latency within 1.5 times the CRNN's. This run is the exploratory single-seed test on the short budget; H6's latency half is measured separately on a quiet machine.
- **Method:** the sanity sequence first (`scripts/rec_sanity.py --config configs/rec/gw-rec-020.yaml`: overfit 64 lines, LR range test, dry run with peak VRAM), then `python train.py --config configs/rec/gw-rec-020.yaml`. Same data, batches, steps, schedule and learning rate as the control GW-REC-003; about as many parameters as the CRNN (both counts are in the run reports). If the LR range test puts the control's learning rate far outside the encoder's useful range, the learning rate is changed here, with the reason, before the main run.
- **Variables:** independent: encoder (SVTR-style vs CRNN); controlled: data, budget, schedule, optimizer
- **Amendment (2026-10-24, before the main run):** the first sanity attempt (`reports/training/sanity/GW-REC-020/attempt1/`) found three things. The LR range test put the lowest smoothed loss at 3.24e-3 (suggestion 3.24e-4), so the control's 1e-3 was outside the 3x band the CRNN met. The overfit check stopped at 12.5% CER after 1,500 steps, still falling fast after a long all-blank plateau. The dry run ran out of GPU memory at the control's batch. The run therefore uses learning rate 5e-4, which is inside the band, chosen by the same procedure that checked the CRNN's rate. It uses micro-batches of 32 lines with 2-step accumulation and a column cap of 16,384: still 64 lines per optimizer step, with wide batches split at the same widths as the control's. The sanity sequence is re-run with these settings and a 4,000-step overfit allowance before the main run. The learning rate now differs between the arms, so the comparison is CRNN at its checked rate against Rec-M at its checked rate.
- **Second amendment (2026-10-24, before the main run):** the second sanity attempt (`reports/training/sanity/GW-REC-020/attempt2/`) used learning rate 5e-4. The overfit check passed, reaching 0% CER after 300 steps, and the LR range test put 5e-4 inside the 3x band (lowest smoothed loss at 2.07e-3). The dry run still ran out of GPU memory at micro-batches of 32 (4.73 GiB allocated under the 85% cap). The run therefore uses micro-batches of 16 with 4-step accumulation and a column cap of 8,192: still 64 lines per optimizer step, with wide batches split at the same widths as the control's. BatchNorm in the stem now sees 16 lines per micro-batch. The sanity sequence is re-run with these settings before the main run.
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 8.18% [8.01, 8.35]; against GW-REC-003 -0.13 points [-0.18, -0.09] (paired bootstrap over lines, 1,000 resamples): **better** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,301 optimizer steps, the final epoch was the best; 7,928,835 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-020/report.md`)
- **Interpretation:** The SVTR-style encoder beats the CRNN at the same step budget and a similar parameter count, a smaller gain than input height 48, and it takes about four times the control's training time. CPU latency is measured separately (`scripts/rec_profile.py`). Its gain and GW-REC-006's are combined in GW-REC-022.
- **Limitations:** single seed; short budget; a transformer may need a longer schedule than a CRNN, so a short-budget loss for Rec-M is weaker evidence against H6 than a win would be for it.
- **Next experiment:** if it is competitive, three seeds at the full budget
- **Training time:** 4.64 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-021 — Phase 7 ablation: frame stride 8, CTC-aware batching (re-run of GW-REC-019)

- **Date allocated / run:** 2026-10-24 / 2026-10-25
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-021.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** as GW-REC-019: what does halving the number of frames per line do to validation CER on the Phase 7 short budget?
- **Hypothesis (written before the run):** as GW-REC-019: similar or slightly higher validation CER with a faster BiLSTM. Lines denser than 8 columns per character lose most; 3% of the training lines are, against 0.27% at stride 4.
- **Method:** `python train.py --config configs/rec/gw-rec-021.yaml`; the control GW-REC-003 with `model.frame_stride: 8` and `data.ctc_aware_batching: true`, so the column cap bounds the padded batches. Compared line by line with GW-REC-003 and read against the spread of GW-REC-003, 004 and 005.
- **Variables:** independent: frame stride (8 instead of 4); batching follows from it (only lines padded beyond their image change buckets); controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005)
- **Result:** val CER 10.93% [10.70, 11.16]; against GW-REC-003 +2.61 points [+2.48, +2.74] (paired bootstrap over lines, 1,000 resamples): **worse** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). 8 epochs, 45,371 optimizer steps, the final epoch was the best; 7,796,435 parameters.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-021/report.md`)
- **Interpretation:** Halving the frames per line is by far the costliest change: about 2.6 points worse, although it is the fastest CRNN on the CPU after the half-width one (7.1 ms per line against the control's 10.7 ms, `reports/training/phase07/cpu_latency.md`). At stride 8 a frame covers 8 input columns, and many HierText lines are too dense for that. Not a candidate.
- **Limitations:** single seed; short budget; batches are not identical to the control's.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** 0.74 GPU-hours (RTX 4050 Laptop, bf16)

### GW-REC-022 — Phase 7 combination: GLYPHARIS-Rec-M (SVTR) at input height 48, weight decay 1e-2

- **Date allocated / run:** 2026-10-25 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-022.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** do the two changes judged better on the short budget, input height 48 (GW-REC-006, −0.49 points) and the SVTR-style encoder (GW-REC-020, −0.13 points), add up when combined?
- **Hypothesis (written before the run):** lower validation CER than either change alone, but by less than the sum of the two gains: both help with small and dense text, so part of their effect overlaps. Expected CPU latency above GW-REC-020's (1.5 times the columns per line), possibly above ADR-0011's line target.
- **Method:** `python train.py --config configs/rec/gw-rec-022.yaml`; GW-REC-020 with `data.image_height: 48` and `data.max_width: 1536` as in GW-REC-006, micro-batches of 8 with 8-step accumulation (64 lines per optimizer step) and a column cap that splits wide batches at the same scaled widths. Learning rate as GW-REC-020 (5e-4); no new LR range test, because only the input height changes. Before the run, one forward and backward pass on the widest batches the sampler can form checks that it fits in GPU memory. Compared line by line with GW-REC-003 (the evidence table), with GW-REC-006 and GW-REC-020 (`scripts/rec_compare.py`), and timed on the CPU with `scripts/rec_profile.py`.
- **Variables:** independent: input height, encoder and weight decay together; batching follows from the height; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005); GW-REC-006, GW-REC-014 and GW-REC-020 for the parts
- **Amendment (2026-10-25, before the run):** GW-REC-014 (weight decay 1e-2 instead of 1e-4) finished after this entry was written and was judged better under the decision rule (−0.04 points [−0.08, −0.00], below all three control seeds; `reports/training/phase07/ablation_table.md`), so the combination includes it. Its gain is small and its CI touches zero at the second decimal, so this run cannot separate its share from the other two changes. GW-REC-021 (frame stride 8) cannot join an SVTR combination whatever its result: the SVTR encoder's frame stride is fixed at 4 (`model.frame_stride must be 4 for svtr`).
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget; micro-batches differ from the control's.
- **Next experiment:** the full-budget confirmation of the chosen configuration
- **Training time:**

### GW-REC-023 — Phase 7 combination without the encoder change: CRNN at input height 48, weight decay 1e-2

- **Date allocated / run:** 2026-10-25 / —
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-023.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** with the CRNN kept, do input height 48 and weight decay 1e-2 add up, and does GW-REC-022's SVTR encoder still help on top of them?
- **Hypothesis (written before the run, while GW-REC-022 was running and before its result):** validation CER close to GW-REC-006's or slightly lower, since weight decay's gain was small (GW-REC-014, −0.04 points). If GW-REC-022 does not beat this run by a clear margin, the SVTR encoder's gain does not survive the combination, and this cheaper, faster candidate is preferred for the full-budget confirmation.
- **Method:** `python train.py --config configs/rec/gw-rec-023.yaml`; GW-REC-006 with `train.weight_decay: 1e-2`. Compared line by line with GW-REC-003 (the evidence table), with GW-REC-006 and with GW-REC-022 (`scripts/rec_compare.py`), and timed on the CPU with `scripts/rec_profile.py`.
- **Variables:** independent: input height and weight decay together; batching as GW-REC-006; controlled: everything else
- **Control:** GW-REC-003 (seed spread: GW-REC-004, 005); GW-REC-006 and GW-REC-014 for the parts; GW-REC-022 for the encoder
- **Result:** not run (abandoned 2026-10-26, before it started). GW-REC-022 beat GW-REC-006 by −0.27 points [−0.32, −0.22] (`reports/training/compare/GW-REC-022_vs_GW-REC-006.md`), so the SVTR encoder's gain survives the combination; weight decay alone was worth −0.04 (GW-REC-014), so this run could only have isolated that small share. Its GPU time went to the full-budget confirmation instead.
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; short budget.
- **Next experiment:** the full-budget confirmation of the chosen configuration
- **Training time:**

### GW-REC-024 — Phase 7 full-budget confirmation: GW-REC-022's recipe at 30 epochs

- **Date allocated / run:** 2026-10-26 / 2026-10-26
- **Phase:** 07
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-024.yaml`   **Seed(s):** 1337 (single seed)
- **Dataset version(s):** as GW-REC-001
- **Hardware:** as GW-REC-001
- **Research question:** does the configuration chosen on the short budget (SVTR-style encoder, input height 48, weight decay 1e-2; GW-REC-022, −0.76 points against the control) also beat GW-REC-001's configuration at the full budget?
- **Hypothesis (written before the run):** validation CER below GW-REC-001's 7.53% with a paired 95% CI entirely below zero. The gap may be smaller than on the short budget, since longer training narrows differences between configurations.
- **Method:** `python train.py --config configs/rec/gw-rec-024.yaml`; GW-REC-022 with `train.epochs: 30` (OneCycle over 30 epochs, early stopping as GW-REC-001). Compared line by line with GW-REC-001 (`scripts/rec_compare.py`) and timed on the CPU (`scripts/rec_profile.py`). The memory probe of GW-REC-022 covers it (same micro-batches).
- **Variables:** independent: the configuration (encoder, input height, weight decay together) at the full budget; controlled: data, budget, seed, schedule shape
- **Control:** GW-REC-001
- **Deviation from the plan:** the Phase 7 plan confirms the chosen configuration with three seeds. This run uses one, to stay within the 30–60 GPU-hour Phase 7 budget (about 35 GPU-hours used before it, about 17 for it). The full-budget seed spread is therefore not measured; the short-budget control seeds (8.30 to 8.32%) are the only seed-noise estimate. Owner decision (2026-10-26): keep one seed for v0.1; the two missing seeds (1337 is this run; 2024 and 7 as in the control) are added later, on a college A100 if access is granted, before the paper (Phase 21), each under its own ID and with the hardware recorded.
- **Result:** val CER 6.46% [6.33, 6.60] (95% bootstrap CI over 73,888 lines); against GW-REC-001 (7.53%) -1.07 [-1.13, -1.01] points CER and -1.94 [-2.10, -1.79] points WER (paired, `reports/training/compare/GW-REC-024_vs_GW-REC-001.md`). 30 epochs, 169,860 optimizer steps; 7,928,835 parameters. CPU latency 24.4 ms mean and 60.3 ms p95 per line (PyTorch, idle machine; `reports/training/phase07/cpu_latency.md`): within ADR-0011's line target, with little margin on the mean.
- **Training diagnostics:** No flags raised. One crash mid-run (CUDA "unknown error" at epoch 23 under memory pressure); the queue resumed it from its last checkpoint the same minute (`reports/training/GW-REC-024/report.md`, Notes).
- **Interpretation:** the hypothesis holds: the configuration chosen on the short budget also wins at the full budget, by about 1.1 points of CER, more than its short-budget gain against the control (−0.76). It is the Phase 7 recognizer configuration for v0.1. Its CPU latency is the main cost: about 2.3 times the CRNN control's per line, just inside the target.
- **Limitations:** one seed; compared with a single GW-REC-001 run.
- **Next experiment:** Phase 8 detector runs; the recognizer is exported in Phase 17.
- **Training time:** 19.31 GPU-hours (RTX 4050 Laptop, bf16)

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
