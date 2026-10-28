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
| [GW-DET-001](#gw-det-001) | 2026-10-25 | 08 | GLYPHARIS-Det baseline: DBNet-style, MobileNetV3-Large, ImageNet init, HierText pages (control for H9 and H10) | done | full val F1@0.5 0.724 [0.708, 0.739] (P 0.924, R 0.595) | `reports/training/GW-DET-001/` |
| [GW-DET-002](#gw-det-002) | 2026-10-25 | 08 | GW-DET-001 with a randomly initialised backbone (H9: ImageNet vs scratch, exploratory) | done | full val F1@0.5 0.704 [0.689, 0.719]; −0.020 [−0.023, −0.016] vs GW-DET-001: H9 supported | `reports/training/GW-DET-002/` |
| [GW-DET-003](#gw-det-003) | 2026-10-25 | 08 | GW-DET-001 with a MobileNetV3-Small backbone (H10: Small within 0.03 F1@0.5 of Large, exploratory) | done | full val F1@0.5 0.668 [0.651, 0.685]; −0.056 [−0.062, −0.051] vs GW-DET-001: H10 not supported | `reports/training/GW-DET-003/` |
| [GW-DET-004](#gw-det-004) | 2026-10-27 | 09 | GW-DET-001 with post-processing calibrated on validation half A, checked on half B (no retraining) | done | half B F1@0.5 0.754 [0.728, 0.775]; +0.032 [+0.025, +0.039] vs the record: adopted | `reports/error_cases/detector/` |
| [GW-EFF-001](#gw-eff-001) | 2026-10-27 | 09 | GW-DET-001 detector CPU latency at five input resolutions (PyTorch, batch 1) | done | detector 203 ms mean per page at 1280; long sides up to 1536 fit the page target | `reports/efficiency/` |
| [GW-DET-005](#gw-det-005) | 2026-10-27 | 09 | GW-DET-001 at the best input resolution that fits ADR-0011's page target (no retraining) | done | half B +0.018 [+0.013, +0.022] vs 1280: 1536 adopted; full val F1@0.5 0.778 [0.762, 0.792] | `reports/error_cases/detector/` |
| [GW-DET-006](#gw-det-006) | 2026-10-27 | 09 | Box shape `quad` vs `poly` on validation and its curved lines (RISK_REGISTER R-22) | done | no curved lines in HierText validation (all 4-point); quad kept; R-22 stays open | `reports/error_cases/detector/` |

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

### GW-DET-001 — GLYPHARIS-Det baseline: DBNet-style, MobileNetV3-Large, ImageNet init

- **Date allocated / run:** 2026-10-25 / 2026-10-26 to 2026-10-27
- **Phase:** 08
- **Git commit:** recorded in the run summary   **Config:** `configs/det/gw-det-001.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** HierText pages from Phase 4 (`data/manifests/processed/hiertext.json`; each page file's SHA-256 is checked against it at start-up); MobileNetV3-Large ImageNet weights (`data/manifests/imagenet_backbones.json`; SHA-256 checked before loading)
- **Hardware:** RTX 4050 Laptop 6 GB. Training in bf16 autocast with the final head layers and the loss in fp32; validation in fp32 with TF32 off, batch 1
- **Research question:** how well does the ARCHITECTURE §3 detector (DBNet-style, MobileNetV3-Large backbone initialised from ImageNet) find HierText text lines, measured as validation F1@0.5, and is its training loop healthy?
- **Hypothesis (written before the run):** none tested on its own. This is the Phase 8 reference run and the control for [H9](hypotheses.md) (against GW-DET-002) and [H10](hypotheses.md) (against GW-DET-003). Expectation, not a claim: validation F1@0.5 below the label ceiling, and no collapse or threshold flags.
- **Method:** the sanity sequence first, under this ID: `scripts/det_preview.py` (training crops and their labels, checked by eye), `scripts/det_sanity.py --config configs/det/gw-det-001.yaml` (label ceiling, overfit one batch, LR range test, 10-minute dry run) and `scripts/det_resume_check.py`. The decision rules below are then applied, and a dated amendment to this entry freezes the measured values, citing `reports/training/sanity/GW-DET-001/sanity.md` and the final config hash, before any main run starts. Then `python train.py --config configs/det/gw-det-001.yaml`: HierText train pages, 4 random 640×640 crops per page per epoch, batch 8, AdamW 1e-3 with 2,000 warm-up steps and a cosine schedule, up to 24 epochs (rule 4), early stopping off so the three arms keep equal step budgets (EXPERIMENT_PROTOCOL §5). Every epoch is scored on a fixed seeded subset of 800 validation pages (plus 200 training pages) at the two pre-registered operating points, `dbnet15` (DBNet unclip, ratio 1.5) and `inverse10` (the exact inverse of the label shrink). `best.pt` is the epoch with the highest F1@0.5 over the two, and it stores that operating point, the run's operating point of record (EXPERIMENT_PROTOCOL §7). At the end `best.pt` and `last.pt` are scored on every validation page: P, R and F1@0.5, F1@0.7 and mean IoU with page-bootstrap CIs at the stored operating point (the headline) and at each fixed one, per slice, with overlays, next to the label ceiling and a row on the Phase 5 baseline page set. GW-DET-002 and 003 are compared with this run by `scripts/det_compare.py` at each arm's stored operating point. GW-DET runs start only when no GW-REC run holds the GPU; the trainer refuses to start with less than 4.0 GiB of GPU memory free (exit 4).
- **Decision rules (fixed 2026-10-25, before the sanity sequence):** each rule names the measurement that triggers it. A fallback triggered by any arm's dry run applies to all three arms, so they keep one recipe. Every outcome goes into the amendment.
  1. **Label ceiling** (once, model-free, the same for all arms): the configured post-processing on every validation page, with the ground-truth label kernels as the probability map. Gate: selection-metric ceiling F1@0.5 ≥ 0.90, and zero counting mismatches between `fast_det_counts` and `evaluate_page`. Below the gate the runs stop and the owner decides; a change such as inference long side 1600 or shrink ratio 0.6 would be a new registered experiment.
  2. **Overfit one batch** (plan exit criterion): 8 fixed training crops, up to 2,000 steps. Pass: kernel Dice ≥ 0.90, threshold-map L1 ≤ 0.02 and F1@0.5 ≥ 0.95 × the batch's own label ceiling. A failure is resolved before any main run.
  3. **Learning rate:** if the configured 1e-3 is not within 3x of GW-DET-001's LR range-test suggestion (the LR at the lowest smoothed loss, divided by 10), the learning rate of all three arms changes before any main run to a value inside that band, stated with the reason in the amendment. GW-DET-002's own range test is recorded but does not change its learning rate (one variable).
  4. **Epochs (9-hour rule):** `epochs = min(24, ⌊9 h / t_epoch⌋)`, where t_epoch is GW-DET-001's dry-run estimate of one epoch's training time plus one per-epoch validation pass. GW-DET-002 and 003 inherit the result (equal step budgets).
  5. **Validation share:** if a dry run's validation pass takes more than 15% of the epoch's training time, `det.data.val_subset` drops from 800 to 400.
  6. **Data wait:** if the training loop waits for data more than 10% of a dry run's training time, `det.data.num_workers` rises from 6 to 8 and the dry run is repeated. If the wait is still above 10%, labels move to `LABEL_VERSION` 2 (a threshold map from `cv2.distanceTransform`), registered before any main run and never changed mid-arm.
  7. **VRAM:** each arm's own dry run must keep reserved GPU memory ≤ 4.0 GiB in training and in validation. If any arm exceeds it, all arms train with batch 4 and gradient accumulation 2 (the same effective batch of 8).
  8. **Other dry-run gates,** with no fallback (a failure stops that arm until it is understood): finite loss, and main-process plus worker RSS ≤ 8 GB.
  9. **Resume check** (GW-DET-001), with the protocol's CUDA design of 2026-10-23 (EXPERIMENT_PROTOCOL §11): two uninterrupted runs A and B, and C resumed from B's mid-epoch checkpoint. On the CPU, C equals B exactly. On CUDA, the first step after the resume matches B's to 1e-6, the learning rates are identical, and C's largest relative loss difference from B is at most max(1e-4, 2 × the A-vs-B difference).
- **Variables:** none (reference run; the control for GW-DET-002 and GW-DET-003)
- **Control:** —
- **Amendment (2026-10-26, before the main runs):** values measured in the sanity sequence (`reports/training/sanity/GW-DET-001/sanity.md`, `GW-DET-002/`, `GW-DET-003/`; preview checked by eye: kernels lie inside the text lines, including rotated and perspective text, and don't-care masks cover illegible and handwritten text). Label ceiling F1@0.5 0.948 at `inverse10`, zero counting mismatches: pass. Overfit one batch: Dice 0.973, threshold L1 0.018, F1@0.5 0.944 against a batch ceiling of 0.990: pass. Resume check (A/B/C): identical on CUDA, pass. Rules applied: (3) GW-DET-001's LR range test suggests 6.03e-3, so the configured 1e-3 was outside the 3x band; the learning rate of all three arms becomes 5e-3, inside that band and inside GW-DET-002's (suggestion 1.51e-2). (4) 24 epochs by the 9-hour rule. (5) validation took 36%, 32% and 37% of an epoch (limit 15%), so `det.data.val_subset` becomes 400. (7) peak reserved VRAM 2.57, 2.57 and 1.55 GiB: within the 4.0 GiB gate. (8) GW-DET-001's dry-run RSS was 8.13 GB against the 8 GB gate, which has no pre-declared fallback; with data wait at 2%, `det.data.num_workers` drops from 6 to 4 for all arms, a change made here before any main run, and the main runs' RSS is checked against the same gate. Final config hash `ebd82071f0a36782`.
- **Result:** full validation (all 1,715 pages, 74,500 care lines), `best.pt` (epoch 22 of 24) at its stored operating point `inverse10`: F1@0.5 0.724 [0.708, 0.739], precision 0.924 [0.917, 0.932], recall 0.595 [0.574, 0.614], F1@0.7 0.631 [0.615, 0.646] (95% page-bootstrap CIs). At `dbnet15`: F1@0.5 0.706 [0.690, 0.721]. Label ceiling at the same settings 0.948. On the 400 Phase 5 validation pages, detection only: F1@0.5 0.726 [0.688, 0.760], against 0.579 [0.544, 0.614] for the best baseline detector there, PP-OCRv6 small (GW-BASE-006, `reports/baselines/detection.md`). Recall is the weak side: 0.20 for lines under 8 px high, 0.30 for lines rotated 15–45° and 0.18 beyond 45°, against 0.72 for 12–20 px lines; per slice F1@0.5 runs from 0.536 (rotated) to 0.859 (multi-line). 3,210,466 parameters (3,187,209 at inference). Report: `reports/training/GW-DET-001/report.md`.
- **Training diagnostics:** one `SLOW VALIDATION` flag at epoch 0 (the 400-page subset took 29% of the epoch against a 25% alarm; it was 36% on 800 pages before rule 5 halved the subset); from epoch 1 validation took 6–7% of an epoch. Data wait 2–3% per epoch. No collapse, threshold or post-processing flags; no crash.
- **Interpretation:** the detector trains cleanly and, on the baselines' own pages, finds HierText lines far better than any baseline (+0.15 F1@0.5 over PP-OCRv6 small), with the caveat that it was trained on HierText's own line definition and the baselines were not. Precision is high (0.92); recall (0.59) is the gap to the 0.95 label ceiling, concentrated in small, rotated and vertical text. Post-processing drops many components as too small (23,596 of 128,100 on full validation), which Phase 9 should test against the small-text misses. GW-DET-001 is the Phase 8 detector and the control for H9 and H10.
- **Limitations:** single seed; the operating point of record is chosen on validation (the per-epoch subset), so the headline is mildly optimistic; HierText is mostly scene text with short lines (RISK_REGISTER R-06), so document pages are measured later, on GDC in Phase 14.
- **Next experiment:** GW-DET-002 and GW-DET-003 (H9, H10); then Phase 9, which works from this run's saved candidates and probability maps.
- **Training time:** 4.86 GPU-hours of training (RTX 4050 Laptop, bf16); 5.74 h of GPU wall-clock including the sanity sequence, the resume check and the full-validation scoring (`reports/compute/ledger.md`).

### GW-DET-002 — GW-DET-001 with a randomly initialised backbone (H9)

- **Date allocated / run:** 2026-10-25 / 2026-10-27
- **Phase:** 08
- **Git commit:** recorded in the run summary   **Config:** `configs/det/gw-det-002.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-DET-001, without the ImageNet weights
- **Hardware:** as GW-DET-001
- **Research question:** does ImageNet initialisation of the MobileNetV3-Large backbone raise validation F1@0.5 at an equal step budget? This is the ImageNet-vs-scratch ablation planned in ADR-0007, so the effect of the prior is measured rather than assumed.
- **Hypothesis (written before the run):** [H9](hypotheses.md): GW-DET-001 beats this run. Success if the paired page-bootstrap 95% CI of F1@0.5(GW-DET-001) − F1@0.5(GW-DET-002) lies entirely above 0, with each arm at the operating point stored in its `best.pt`.
- **Method:** identical to GW-DET-001 except `det.model.init: scratch` (torchvision's random initialisation, no weights file); a test checks that the resolved config differs from GW-DET-001's only there and in the run's name. Sanity sequence under this ID: an LR range test (recorded only; the learning rate stays GW-DET-001's) and its own 10-minute dry run with the VRAM gate, because a scratch model's early probability maps are noisier and its validation can cost more. GW-DET-001's decision rules apply, including `epochs` from GW-DET-001's dry run, and a fallback triggered here applies to all three arms. A dated amendment to this entry freezes the values before the main run, citing `reports/training/sanity/GW-DET-002/sanity.md` and the final config hash. Then `python train.py --config configs/det/gw-det-002.yaml` and `python scripts/det_compare.py --a runs/GW-DET-001 --b runs/GW-DET-002`.
- **Variables:** independent: backbone initialisation (ImageNet vs random); controlled: data, crops, augmentation, labels, loss, optimizer, learning rate, schedule, step budget, operating points, seed
- **Control:** GW-DET-001
- **Amendment (2026-10-26, before the main runs):** values measured in the sanity sequence (`reports/training/sanity/GW-DET-001/sanity.md`, `GW-DET-002/`, `GW-DET-003/`; preview checked by eye: kernels lie inside the text lines, including rotated and perspective text, and don't-care masks cover illegible and handwritten text). Label ceiling F1@0.5 0.948 at `inverse10`, zero counting mismatches: pass. Overfit one batch: Dice 0.973, threshold L1 0.018, F1@0.5 0.944 against a batch ceiling of 0.990: pass. Resume check (A/B/C): identical on CUDA, pass. Rules applied: (3) GW-DET-001's LR range test suggests 6.03e-3, so the configured 1e-3 was outside the 3x band; the learning rate of all three arms becomes 5e-3, inside that band and inside GW-DET-002's (suggestion 1.51e-2). (4) 24 epochs by the 9-hour rule. (5) validation took 36%, 32% and 37% of an epoch (limit 15%), so `det.data.val_subset` becomes 400. (7) peak reserved VRAM 2.57, 2.57 and 1.55 GiB: within the 4.0 GiB gate. (8) GW-DET-001's dry-run RSS was 8.13 GB against the 8 GB gate, which has no pre-declared fallback; with data wait at 2%, `det.data.num_workers` drops from 6 to 4 for all arms, a change made here before any main run, and the main runs' RSS is checked against the same gate. Final config hash `26340926cb03c343`.
- **Result:** full validation, `best.pt` (epoch 16 of 24) at `inverse10`: F1@0.5 0.704 [0.689, 0.719], precision 0.911 [0.902, 0.919], recall 0.574 [0.554, 0.593]. Against GW-DET-001, paired over the 1,715 pages: **−0.0197 [−0.0232, −0.0161] F1@0.5** (GW-DET-001 minus GW-DET-002: +0.0197 [+0.0161, +0.0232]; `reports/training/GW-DET-001_vs_GW-DET-002.md`); +0.0169 [+0.0129, +0.0205] with both at `dbnet15`. Lower than GW-DET-001 on every slice, by 0.007 (multi-line) to 0.035 (dense). Report: `reports/training/GW-DET-002/report.md`.
- **Training diagnostics:** one `POSSIBLE OVERFITTING` flag at epoch 9 (subset validation error 10% above its best while training error kept falling); validation recovered and peaked at epoch 16, and the gap to training F1 at the best epoch is 0.008. No crash.
- **Interpretation:** **H9 supported** (exploratory, one seed): at an equal step budget the ImageNet initialisation is worth about 0.02 F1@0.5, the CI clear of zero at both operating points. The scratch backbone still reaches 0.70, so RISK_REGISTER R-05's fallback (no ImageNet weights) would cost about two points of F1, not the detector. ImageNet initialisation stays in the v0.1 recipe.
- **Limitations:** single seed, and the paired CI covers page sampling only, not training randomness, so part of a difference between two single runs may be seed noise; the learning rate is GW-DET-001's, and if this arm's own range test suggests another one, that is reported as a limitation of the comparison; the two arms may end at different operating points of record.
- **Next experiment:** if the owner rules out the ImageNet weight terms for a delivered detector (RISK_REGISTER R-05), this arm is the measured fallback.
- **Training time:** 4.72 GPU-hours of training; 5.42 h GPU wall-clock with its sanity runs and full-validation scoring.

### GW-DET-003 — GW-DET-001 with a MobileNetV3-Small backbone (H10)

- **Date allocated / run:** 2026-10-25 / 2026-10-27
- **Phase:** 08
- **Git commit:** recorded in the run summary   **Config:** `configs/det/gw-det-003.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-DET-001, with the MobileNetV3-Small ImageNet weights (`data/manifests/imagenet_backbones.json`)
- **Hardware:** as GW-DET-001
- **Research question:** does the smaller MobileNetV3-Small backbone (the `-S` variant of ARCHITECTURE §3) keep validation F1@0.5 within 0.03 of MobileNetV3-Large at the same recipe and step budget?
- **Hypothesis (written before the run):** [H10](hypotheses.md): Small is non-inferior to Large within 0.03 F1@0.5. Success if the upper bound of the paired page-bootstrap 95% CI of F1@0.5(GW-DET-001) − F1@0.5(GW-DET-003) is ≤ 0.03, with each arm at the operating point stored in its `best.pt`.
- **Method:** identical to GW-DET-001 except `det.model.backbone: mobilenet_v3_small` and its ImageNet weights file (a test checks that nothing else differs): same neck width, heads, data, recipe and learning rate. Sanity sequence under this ID: its own 10-minute dry run with the VRAM gate, because its memory use and throughput differ from Large. GW-DET-001's decision rules apply, including `epochs` from GW-DET-001's dry run, and a fallback triggered here applies to all three arms. A dated amendment to this entry freezes the values before the main run, citing `reports/training/sanity/GW-DET-003/sanity.md` and the final config hash. Then `python train.py --config configs/det/gw-det-003.yaml` and `python scripts/det_compare.py --a runs/GW-DET-001 --b runs/GW-DET-003`. Both arms' parameter counts come from their run summaries.
- **Variables:** independent: backbone size (MobileNetV3-Large vs Small, both ImageNet initialised); controlled: neck and heads, data, crops, augmentation, labels, loss, optimizer, learning rate, schedule, step budget, operating points, seed
- **Control:** GW-DET-001
- **Amendment (2026-10-26, before the main runs):** values measured in the sanity sequence (`reports/training/sanity/GW-DET-001/sanity.md`, `GW-DET-002/`, `GW-DET-003/`; preview checked by eye: kernels lie inside the text lines, including rotated and perspective text, and don't-care masks cover illegible and handwritten text). Label ceiling F1@0.5 0.948 at `inverse10`, zero counting mismatches: pass. Overfit one batch: Dice 0.973, threshold L1 0.018, F1@0.5 0.944 against a batch ceiling of 0.990: pass. Resume check (A/B/C): identical on CUDA, pass. Rules applied: (3) GW-DET-001's LR range test suggests 6.03e-3, so the configured 1e-3 was outside the 3x band; the learning rate of all three arms becomes 5e-3, inside that band and inside GW-DET-002's (suggestion 1.51e-2). (4) 24 epochs by the 9-hour rule. (5) validation took 36%, 32% and 37% of an epoch (limit 15%), so `det.data.val_subset` becomes 400. (7) peak reserved VRAM 2.57, 2.57 and 1.55 GiB: within the 4.0 GiB gate. (8) GW-DET-001's dry-run RSS was 8.13 GB against the 8 GB gate, which has no pre-declared fallback; with data wait at 2%, `det.data.num_workers` drops from 6 to 4 for all arms, a change made here before any main run, and the main runs' RSS is checked against the same gate. Final config hash `2ba0a429ed5ba49c`.
- **Result:** full validation, `best.pt` (epoch 22 of 24) at `inverse10`: F1@0.5 0.668 [0.651, 0.685], precision 0.914 [0.904, 0.923], recall 0.527 [0.507, 0.547]. Against GW-DET-001, paired: GW-DET-001 minus GW-DET-003 **+0.0556 [+0.0505, +0.0616] F1@0.5** (`reports/training/GW-DET-001_vs_GW-DET-003.md`); +0.0495 [+0.0445, +0.0555] with both at `dbnet15`. The loss is in recall, largest on small text (0.11 recall under 8 px, against 0.20 for Large). 1,120,210 parameters (1,096,953 at inference), 35% of Large's; 50.6 crops/s in training against 45.0. Report: `reports/training/GW-DET-003/report.md`.
- **Training diagnostics:** four flags at epoch 0 only, none later: `THRESHOLD OUT OF RANGE` (5.7% of threshold pixels outside [0.2, 0.8], alarm 5%), `POSTPROCESS DROPS` (at `dbnet15`, 22,450 of 27,401 components dropped as too thin), `DATA LOADER BOTTLENECK` (24.8% data wait, alarm 20%) and `SLOW VALIDATION` (32%). Each flag is raised once per run, at its first occurrence, so the per-epoch metrics were checked: the threshold, post-processing and validation-time conditions were gone from epoch 1 (threshold pixels out of range 0.0%, thin drops 0%, validation 7–8% of an epoch). The data wait was not: it stayed at 21–31% of every epoch, because the smaller model outruns four loader workers. It cost wall-clock time, not steps, so the step budget, and with it the H10 comparison, is unaffected. No crash.
- **Interpretation:** **H10 not supported**: the upper CI bound, 0.062, is twice the 0.03 margin, and the whole CI lies above it, so Small is clearly worse at this recipe and budget, not merely unproven. MobileNetV3-Large stays the v0.1 backbone. Small's case rested on CPU latency, which this run did not measure; Phase 16 can revisit it with a latency number if Large misses ADR-0011's page target.
- **Limitations:** single seed, and the paired CI covers page sampling only, not training randomness; no LR range test of its own (it uses GW-DET-001's learning rate); CPU latency, the reason to prefer a smaller backbone, is not measured by this run.
- **Next experiment:** with H9, the evidence for choosing the detector backbone of v0.1.
- **Training time:** 4.33 GPU-hours of training; 5.02 h GPU wall-clock with its dry run and full-validation scoring.

### GW-DET-004 — Post-processing calibrated on validation

- **Date allocated / run:** 2026-10-27 / 2026-10-27 to 2026-10-28
- **Phase:** 09
- **Git commit:** recorded in the result file   **Config:** `configs/det/gw-det-001.yaml` with overrides   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages as in GW-DET-001 (`processed/hiertext/pages_validation.jsonl.gz`, sha256 0655c5f795a6)
- **Hardware:** RTX 4050 Laptop 6 GB for forward passes; i7-14650HX CPU for post-processing
- **Research question:** how much F1@0.5 does GW-DET-001 gain from post-processing chosen on validation instead of the pre-registered operating points?
- **Hypothesis (written before the run):** none formal. Expectation, not a claim: a small gain (under 0.02) from a lower `bin_thresh` or `box_thresh`, since the run's errors are misses (precision 0.92, recall 0.59).
- **Method:** `docs/PHASE_STATUS.md` Phase 09, plan items 2 and 3. GW-DET-001 `best.pt`, long side 1280; 8-bit probability maps cached once by `scripts/det_eval.py --save-prob-maps`; 440 settings scored on half A (`scripts/det_sweep.py`); the arg-max is compared with `inverse10` at the default thresholds on half B (paired page bootstrap, 1,000 resamples) and adopted only if that CI lies above 0; the adopted setting is re-scored once on float maps for half B, which gives the reported number.
- **Variables:** independent: `bin_thresh`, `box_thresh`, `min_component_area`, unclip mode and ratio; controlled: checkpoint, input resolution, box type, pages
- **Control:** GW-DET-001 at its operating point of record (`inverse10`, `bin_thresh` 0.3, `box_thresh` 0.6, `min_component_area` 3)
- **Result:** half A (857 pages, 8-bit maps): the arg-max of the 440 settings is `bin_thresh` 0.3, `min_component_area` 1, unclip `inverse09`, `box_thresh` 0.5, F1@0.5 0.7626 against 0.7252 for the record (`min_component_area` 1 and 3 tie: the filter never changes a box). Half B (858 pages): 0.7543 against 0.7227, **+0.0316 [+0.0247, +0.0386]** (`reports/error_cases/detector/GW-DET-004/pair_B.json`): adopted. Float re-check on half B: F1@0.5 0.754 [0.728, 0.775], precision 0.856, recall 0.674, F1@0.7 0.645, against the record's 0.723 [0.695, 0.745], precision 0.924, recall 0.593; paired +0.0317 [+0.0247, +0.0387] (`pair_B_float.json`, `eval_B_adopted.json`, `eval_B_record.json`). The chosen `box_thresh` (0.5) and `bin_thresh` (0.3) lie inside the grid.
- **Interpretation:** the pre-registered `box_thresh` 0.6 was too strict for this network: lowering it to 0.5 and growing boxes slightly less (`inverse09`) recovers about eight points of recall for seven of precision, a net gain of about 0.03 F1@0.5 on the check half. The error analysis predicted it: 46% of the missed lines had a candidate below the old threshold. Phase 10 uses these settings.
- **Limitations:** chosen and checked on two halves of one validation set from one checkpoint; the uint8 maps differ slightly from float maps, hence the float re-check.
- **Next experiment:** GW-DET-005 uses the adopted setting.

### GW-EFF-001 — Detector CPU latency by input resolution

- **Date allocated / run:** 2026-10-27 / 2026-10-28
- **Phase:** 09
- **Git commit:** recorded in the result file   **Config:** `configs/det/gw-det-001.yaml` with overrides   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages as in GW-DET-001 (`processed/hiertext/pages_validation.jsonl.gz`, sha256 0655c5f795a6)
- **Hardware:** RTX 4050 Laptop 6 GB for forward passes; i7-14650HX CPU for post-processing
- **Research question:** how long does GW-DET-001 take per page on the CPU at long side 1024 to 2048, and does any of them leave room for recognition inside ADR-0011's page target?
- **Hypothesis (written before the run):** none. Expectation: time grows roughly with pixel count, so 2048 costs about 2.5 times 1280.
- **Method:** `scripts/det_profile.py --long-side N`, one process per long side, on an otherwise idle machine with no training running: the 110 Phase 5 validation pages the baselines' page profiles used (10 warm-up, 100 timed), batch 1, PyTorch fp32 with default threads, timing a decoded page to boxes (resize, normalise, forward, post-processing at GW-DET-004's adopted setting or the record's). Records mean and p95 time, peak RAM and the number of boxes per page, with run conditions as for the baselines.
- **Variables:** independent: long side; controlled: checkpoint, post-processing, pages, threads
- **Control:** long side 1280
- **Result:** PyTorch fp32, 16 threads, at GW-DET-004's post-processing (`reports/efficiency/det_latency.md`): detector mean / p95 per page 132 / 176 ms at long side 1024, 203 / 301 at 1280, 326 / 435 at 1536, 452 / 595 at 1792, 615 / 872 at 2048; peak RAM 1,235 to 1,779 MiB. Estimated page time (plus 24.4 ms per predicted box): 954 / 1,812 ms, 1,130 / 2,175, 1,323 / 2,523, 1,514 / 2,788 and 1,708 / 3,025 ms. Eligible under ADR-0011's page target: 1024, 1280 and 1536.
- **Interpretation:** recognition dominates the page time at every resolution (34 to 45 lines at 24.4 ms each), so the detector's input size can grow to 1536 before the page target binds; 1792 misses the mean by 14 ms.
- **Limitations:** PyTorch stands in for the Phase 17 ONNX Runtime build; recognition time is estimated from GW-REC-024's mean line time, not measured on the detected crops (Phase 10 measures the pipeline).
- **Next experiment:** GW-DET-005's eligibility rule.

### GW-DET-005 — Input resolution

- **Date allocated / run:** 2026-10-27 / 2026-10-28
- **Phase:** 09
- **Git commit:** recorded in the result file   **Config:** `configs/det/gw-det-001.yaml` with overrides   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages as in GW-DET-001 (`processed/hiertext/pages_validation.jsonl.gz`, sha256 0655c5f795a6)
- **Hardware:** RTX 4050 Laptop 6 GB for forward passes; i7-14650HX CPU for post-processing
- **Research question:** does a larger input resolution recover small text (recall 0.20 for lines under 8 px at 1280) within the page-time budget?
- **Hypothesis (written before the run):** none formal. Expectation: recall on small lines rises with resolution, F1 overall rises less, and time limits how far it can go.
- **Method:** `docs/PHASE_STATUS.md` Phase 09, plan items 2 and 5. Float forward passes of GW-DET-001 at long side 1024, 1280, 1536, 1792 and 2048 on both halves (`scripts/det_eval.py --set det.infer.long_side=N --pages-half A|B`) with GW-DET-004's adopted post-processing. Eligibility from GW-EFF-001 (estimated page time within ADR-0011's page target); the eligible arg-max on half A is adopted against 1280 if its paired CI on half B lies above 0. If no long side of 1280 or more is eligible, 1280 stays and the gap is recorded for Phases 16 and 17.
- **Variables:** independent: `det.infer.long_side`; controlled: checkpoint, post-processing, box type, pages
- **Control:** long side 1280 with the same post-processing
- **Result:** half A, float maps with GW-DET-004's post-processing (`reports/error_cases/detector/GW-DET-005/resolution_A.md`): F1@0.5 0.718 at long side 1024, 0.763 at 1280, 0.784 at 1536, 0.793 at 1792 and 0.796 at 2048; eligible under the page target (GW-EFF-001): 1024, 1280 and 1536, so the choice is 1536. Half B: 0.772 [0.746, 0.793] against 0.754 at 1280, **+0.0177 [+0.0131, +0.0221]** (`pair_B.json`): adopted. With both Phase 9 changes, all 1,715 validation pages: F1@0.5 0.778 [0.762, 0.792], precision 0.852, recall 0.716, F1@0.7 0.671 (`reports/error_cases/detector/phase09_adopted/eval_full.json` and `analysis.md`), against GW-DET-001's 0.724 at its operating point of record.
- **Interpretation:** resolution is the cheapest lever left for small text: recall keeps rising up to 2048 while precision slips slowly, and the page-time budget, not accuracy, stops it at 1536. Lines under 8 px (now measured at 1536) are found 34% of the time, against 20% under 8 px at 1280. The detector keeps its training-time scale range, so this is within what it was trained for (scale jitter 0.5 to 2.0).
- **Limitations:** the network was trained with scale jitter 0.5–2.0 around the 1280 scale, so other resolutions test its scale tolerance as much as resolution; post-processing is not re-tuned per resolution.
- **Next experiment:** GW-DET-006 at the adopted resolution.

### GW-DET-006 — Box shape: `quad` or `poly`

- **Date allocated / run:** 2026-10-27 / 2026-10-28
- **Phase:** 09
- **Git commit:** recorded in the result file   **Config:** `configs/det/gw-det-001.yaml` with overrides   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages as in GW-DET-001 (`processed/hiertext/pages_validation.jsonl.gz`, sha256 0655c5f795a6)
- **Hardware:** RTX 4050 Laptop 6 GB for forward passes; i7-14650HX CPU for post-processing
- **Research question:** do outline boxes (`poly`) match curved HierText lines better than rotated rectangles (`quad`), enough to change the setting Phase 14 uses on GDC (RISK_REGISTER R-22)?
- **Hypothesis (written before the run):** none formal. Expectation: `poly` helps only on the curved lines, which are few, and costs nothing elsewhere.
- **Method:** `docs/PHASE_STATUS.md` Phase 09, plan item 6. Both box types from the same candidates at the adopted settings, on all validation pages. Curved lines: care lines with at least six polygon vertices whose polygon area is below 0.8 of their minimum-area rectangle's. `poly` is adopted for Phase 14 if, on the curved lines, the paired CI of recall@0.5 (poly minus quad) lies above 0 and, on all pages, the CI of F1@0.5 (poly minus quad) does not lie below 0.
- **Variables:** independent: `det.post.box_type`; controlled: everything else
- **Control:** `quad` at the same settings
- **Result:** every one of the 74,500 HierText validation care lines is a four-point polygon, so the registered curved-line subset is empty and its recall comparison is undefined. On all pages, at the adopted settings: `poly` minus `quad` F1@0.5 +0.0004 [-0.0001, +0.0009] (`reports/error_cases/detector/GW-DET-006/box_shape.md`). The rule therefore keeps `quad` for Phase 14.
- **Interpretation:** the check could not answer its question: HierText's line annotations carry no curvature, so validation data cannot say whether `poly` helps on GDC's curved phone-captured lines. RISK_REGISTER R-22 stays open (Phase 9 known issues).
- **Limitations:** HierText's curved lines stand in for GDC's phone-captured ones, which cannot be looked at before Phase 14.
- **Next experiment:** Phase 10 uses the adopted detector settings.

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
