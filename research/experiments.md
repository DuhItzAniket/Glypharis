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

### GW-BASE-001 — Tesseract 5 (LSTM, eng): recognition and end to end

- **System:** Tesseract 5 (UB Mannheim build), `-l eng`, default engine; line crops with `--psm 7`, pages with the default page mode (`--psm 3`).
- **Date allocated / run:** 2026-10-21 / —
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Validation: 10,000 seeded line crops per set and 400 seeded pages; test sets in full (logged in `research/test_access_log.md`).
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
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Validation: 10,000 seeded line crops per set and 400 seeded pages; test sets in full (logged in `research/test_access_log.md`).
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
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Validation: 10,000 seeded line crops per set and 400 seeded pages; test sets in full (logged in `research/test_access_log.md`).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** see `reports/baselines/` (filled in after the run)
- **Training diagnostics:** not applicable (no training)
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** not applicable

### GW-BASE-004 — TrOCR-small-printed: recognition only

- **System:** microsoft/trocr-small-printed, transformers on CPU, greedy decoding as in the checkpoint, with `max_new_tokens=256` because the checkpoint sets no length and transformers would otherwise stop at 20 tokens and cut long lines. It has no detector, so it is run on recognition only.
- **Date allocated / run:** 2026-10-21 / —
- **Phase:** 05
- **Git commit:** recorded in the run metadata   **Config:** `scripts/run_baselines.py` defaults   **Seed(s):** subsample seed 5; bootstrap seed 20261010
- **Dataset version(s):** HierText (Phase 4 manifests), GlyphSynth v1 val, GDC v1; test crops in `data/manifests/processed/eval_crops.json`
- **Hardware:** laptop CPU only (`.venv-baselines`, ADR-0006); no GPU
- **Research question:** where do existing systems stand on our evaluation sets, under our metrics?
- **Hypothesis (written before the run):** none; this is a reference measurement, not a test of an idea.
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Validation: 10,000 seeded line crops per set and 400 seeded pages; test sets in full (logged in `research/test_access_log.md`).
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
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Validation: 10,000 seeded line crops per set and 400 seeded pages; test sets in full (logged in `research/test_access_log.md`).
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
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Validation: 10,000 seeded line crops per set and 400 seeded pages; test sets in full (logged in `research/test_access_log.md`).
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
- **Method:** `scripts/run_baselines.py` then `scripts/evaluate_baselines.py`. Validation: 10,000 seeded line crops per set and 400 seeded pages; test sets in full (logged in `research/test_access_log.md`).
- **Variables:** independent: system; controlled: evaluation sets, crops, metrics and normalisation tracks
- **Control:** —
- **Result:** see `reports/baselines/` (filled in after the run)
- **Training diagnostics:** not applicable (no training)
- **Interpretation:**
- **Limitations:**
- **Next experiment:**
- **Training time:** not applicable

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
