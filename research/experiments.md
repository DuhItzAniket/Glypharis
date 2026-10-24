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
| [GW-DET-001](#gw-det-001) | 2026-10-25 | 08 | GLYPHARIS-Det baseline: DBNet-style, MobileNetV3-Large, ImageNet init, HierText pages (control for H9 and H10) | planned | — | `reports/training/GW-DET-001/` |
| [GW-DET-002](#gw-det-002) | 2026-10-25 | 08 | GW-DET-001 with a randomly initialised backbone (H9: ImageNet vs scratch, exploratory) | planned | — | `reports/training/GW-DET-002/` |
| [GW-DET-003](#gw-det-003) | 2026-10-25 | 08 | GW-DET-001 with a MobileNetV3-Small backbone (H10: Small within 0.03 F1@0.5 of Large, exploratory) | planned | — | `reports/training/GW-DET-003/` |

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

### GW-DET-001 — GLYPHARIS-Det baseline: DBNet-style, MobileNetV3-Large, ImageNet init

- **Date allocated / run:** 2026-10-25 / —
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
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed; the operating point of record is chosen on validation (the per-epoch subset), so the headline is mildly optimistic; HierText is mostly scene text with short lines (RISK_REGISTER R-06), so document pages are measured later, on GDC in Phase 14.
- **Next experiment:** GW-DET-002 and GW-DET-003 (H9, H10); then Phase 9, which works from this run's saved candidates and probability maps.
- **Training time:**

### GW-DET-002 — GW-DET-001 with a randomly initialised backbone (H9)

- **Date allocated / run:** 2026-10-25 / —
- **Phase:** 08
- **Git commit:** recorded in the run summary   **Config:** `configs/det/gw-det-002.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-DET-001, without the ImageNet weights
- **Hardware:** as GW-DET-001
- **Research question:** does ImageNet initialisation of the MobileNetV3-Large backbone raise validation F1@0.5 at an equal step budget? This is the ImageNet-vs-scratch ablation planned in ADR-0007, so the effect of the prior is measured rather than assumed.
- **Hypothesis (written before the run):** [H9](hypotheses.md): GW-DET-001 beats this run. Success if the paired page-bootstrap 95% CI of F1@0.5(GW-DET-001) − F1@0.5(GW-DET-002) lies entirely above 0, with each arm at the operating point stored in its `best.pt`.
- **Method:** identical to GW-DET-001 except `det.model.init: scratch` (torchvision's random initialisation, no weights file); a test checks that the resolved config differs from GW-DET-001's only there and in the run's name. Sanity sequence under this ID: an LR range test (recorded only; the learning rate stays GW-DET-001's) and its own 10-minute dry run with the VRAM gate, because a scratch model's early probability maps are noisier and its validation can cost more. GW-DET-001's decision rules apply, including `epochs` from GW-DET-001's dry run, and a fallback triggered here applies to all three arms. A dated amendment to this entry freezes the values before the main run, citing `reports/training/sanity/GW-DET-002/sanity.md` and the final config hash. Then `python train.py --config configs/det/gw-det-002.yaml` and `python scripts/det_compare.py --a runs/GW-DET-001 --b runs/GW-DET-002`.
- **Variables:** independent: backbone initialisation (ImageNet vs random); controlled: data, crops, augmentation, labels, loss, optimizer, learning rate, schedule, step budget, operating points, seed
- **Control:** GW-DET-001
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed, and the paired CI covers page sampling only, not training randomness, so part of a difference between two single runs may be seed noise; the learning rate is GW-DET-001's, and if this arm's own range test suggests another one, that is reported as a limitation of the comparison; the two arms may end at different operating points of record.
- **Next experiment:** if the owner rules out the ImageNet weight terms for a delivered detector (RISK_REGISTER R-05), this arm is the measured fallback.
- **Training time:**

### GW-DET-003 — GW-DET-001 with a MobileNetV3-Small backbone (H10)

- **Date allocated / run:** 2026-10-25 / —
- **Phase:** 08
- **Git commit:** recorded in the run summary   **Config:** `configs/det/gw-det-003.yaml`   **Seed(s):** 1337 (single seed: exploratory)
- **Dataset version(s):** as GW-DET-001, with the MobileNetV3-Small ImageNet weights (`data/manifests/imagenet_backbones.json`)
- **Hardware:** as GW-DET-001
- **Research question:** does the smaller MobileNetV3-Small backbone (the `-S` variant of ARCHITECTURE §3) keep validation F1@0.5 within 0.03 of MobileNetV3-Large at the same recipe and step budget?
- **Hypothesis (written before the run):** [H10](hypotheses.md): Small is non-inferior to Large within 0.03 F1@0.5. Success if the upper bound of the paired page-bootstrap 95% CI of F1@0.5(GW-DET-001) − F1@0.5(GW-DET-003) is ≤ 0.03, with each arm at the operating point stored in its `best.pt`.
- **Method:** identical to GW-DET-001 except `det.model.backbone: mobilenet_v3_small` and its ImageNet weights file (a test checks that nothing else differs): same neck width, heads, data, recipe and learning rate. Sanity sequence under this ID: its own 10-minute dry run with the VRAM gate, because its memory use and throughput differ from Large. GW-DET-001's decision rules apply, including `epochs` from GW-DET-001's dry run, and a fallback triggered here applies to all three arms. A dated amendment to this entry freezes the values before the main run, citing `reports/training/sanity/GW-DET-003/sanity.md` and the final config hash. Then `python train.py --config configs/det/gw-det-003.yaml` and `python scripts/det_compare.py --a runs/GW-DET-001 --b runs/GW-DET-003`. Both arms' parameter counts come from their run summaries.
- **Variables:** independent: backbone size (MobileNetV3-Large vs Small, both ImageNet initialised); controlled: neck and heads, data, crops, augmentation, labels, loss, optimizer, learning rate, schedule, step budget, operating points, seed
- **Control:** GW-DET-001
- **Result:**
- **Training diagnostics:**
- **Interpretation:**
- **Limitations:** single seed, and the paired CI covers page sampling only, not training randomness; no LR range test of its own (it uses GW-DET-001's learning rate); CPU latency, the reason to prefer a smaller backbone, is not measured by this run.
- **Next experiment:** with H9, the evidence for choosing the detector backbone of v0.1.
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
