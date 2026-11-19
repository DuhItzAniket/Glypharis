# Experiment Registry

Every experiment ID is allocated here **before** the run starts (see `docs/WORKFLOW.md` §4).
Status values: `planned` → `running` → `done` | `failed` | `abandoned`, and `not run` for a conditional run whose condition was not met.

Rows link to a full entry below. Numbers are copied from generated reports only, never typed by hand. The Report column names the files or folders a row's key result comes from; `—` means the run never started, so it has no report. An entry whose result was replaced by a v1 re-run names that run in an optional `Superseded by` field, and its own result stays as recorded.

| ID | Date allocated | Phase | Question / hypothesis | Status | Key result | Report |
|---|---|---|---|---|---|---|
| [GW-BASE-001](#gw-base-001) | 2026-10-21 | 05 | Tesseract 5 (LSTM, eng): recognition and end to end | done (Phase 14 amendment run) | lines 24.35% CER (HierText val); 0.43 s per page | `reports/baselines/` |
| [GW-BASE-002](#gw-base-002) | 2026-10-21 | 05 | PaddleOCR PP-OCRv5 mobile (det + English rec): recognition and end to end | done (Phase 14 and Phase 16 amendments run) | lines 9.27% CER (HierText val); 3.4 s per page | `reports/baselines/` |
| [GW-BASE-003](#gw-base-003) | 2026-10-21 | 05 | PaddleOCR PP-OCRv5 server (det + rec): recognition and end to end | done (Phase 14 amendment run) | lines 14.65% CER (HierText val); 80.9 s per page; GDC pages not runnable | `reports/baselines/` |
| [GW-BASE-004](#gw-base-004) | 2026-10-21 | 05 | TrOCR-small-printed: recognition only | done (Phase 14 amendment run) | lines 62.58% strict / 19.14% normalised CER (HierText val); 133 ms per line | `reports/baselines/` |
| [GW-BASE-005](#gw-base-005) | 2026-10-21 | 05 | PaddleOCR PP-OCRv6 tiny (det + rec): recognition and end to end | done (Phase 14 amendment run) | lines 11.10% CER (HierText val); 21.6 ms per line, 1.7 s per page | `reports/baselines/` |
| [GW-BASE-006](#gw-base-006) | 2026-10-21 | 05 | PaddleOCR PP-OCRv6 small (det + rec): recognition and end to end | done (Phase 14 amendment run) | lines 7.53% CER (HierText val); 46 ms per line, 3.7 s per page | `reports/baselines/` |
| [GW-BASE-007](#gw-base-007) | 2026-10-21 | 05 | PaddleOCR PP-OCRv6 medium (PaddleOCR 3.7's English default): reduced scope | done (Phase 14 amendment run) | lines 7.54% CER (HierText val); 553 ms per line, 32.5 s per page | `reports/baselines/` |
| [GW-REC-001](#gw-rec-001) | 2026-10-22 | 06 | CRNN baseline on HierText lines | done | val CER 7.53% [7.38, 7.69] | `reports/training/GW-REC-001/` |
| [GW-REC-002](#gw-rec-002) | 2026-10-22 | 06 | GW-REC-001 plus 35,000 GlyphSynth lines (exploratory look at H2′a) | done | val CER 7.49% [7.34, 7.64]; −0.05 points [−0.09, −0.01] vs GW-REC-001 (one seed, not budget-matched) | `reports/training/GW-REC-002/`, `reports/training/compare/GW-REC-002_vs_GW-REC-001.md` |
| [GW-REC-003](#gw-rec-003) | 2026-10-23 | 07 | Phase 7 control: GW-REC-001 on the 8-epoch short budget, seed 1337 (noise floor) | done | val CER 8.32% [8.16, 8.47] (control) | `reports/training/GW-REC-003/` |
| [GW-REC-004](#gw-rec-004) | 2026-10-23 | 07 | Phase 7 control: GW-REC-001 on the 8-epoch short budget, seed 2024 (noise floor) | done | val CER 8.30% [8.15, 8.47] (control) | `reports/training/GW-REC-004/` |
| [GW-REC-005](#gw-rec-005) | 2026-10-23 | 07 | Phase 7 control: GW-REC-001 on the 8-epoch short budget, seed 7 (noise floor) | done | val CER 8.31% [8.15, 8.47] (control) | `reports/training/GW-REC-005/` |
| [GW-REC-006](#gw-rec-006) | 2026-10-23 | 07 | Phase 7 ablation: input height 48 (short budget, vs GW-REC-003) | done | val CER 7.83% [7.66, 7.98]; -0.49 [-0.54, -0.44] vs GW-REC-003 (better) | `reports/training/GW-REC-006/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-007](#gw-rec-007) | 2026-10-23 | 07 | Phase 7 ablation: CNN half as wide (short budget, vs GW-REC-003) | done | val CER 8.83% [8.65, 9.00]; +0.51 [+0.47, +0.56] vs GW-REC-003 (worse) | `reports/training/GW-REC-007/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-008](#gw-rec-008) | 2026-10-23 | 07 | Phase 7 ablation: BiLSTM hidden 128 (short budget, vs GW-REC-003) | done | val CER 8.78% [8.60, 8.94]; +0.46 [+0.42, +0.51] vs GW-REC-003 (worse) | `reports/training/GW-REC-008/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-009](#gw-rec-009) | 2026-10-23 | 07 | Phase 7 ablation: one BiLSTM layer (short budget, vs GW-REC-003) | done | val CER 8.78% [8.60, 8.94]; +0.46 [+0.42, +0.51] vs GW-REC-003 (worse) | `reports/training/GW-REC-009/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-010](#gw-rec-010) | 2026-10-23 | 07 | Phase 7 ablation: Adam (short budget, vs GW-REC-003) | done | val CER 9.17% [8.99, 9.34]; +0.85 [+0.81, +0.90] vs GW-REC-003 (worse) | `reports/training/GW-REC-010/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-011](#gw-rec-011) | 2026-10-23 | 07 | Phase 7 ablation: cosine schedule (short budget, vs GW-REC-003) | done | val CER 8.44% [8.26, 8.60]; +0.12 [+0.08, +0.17] vs GW-REC-003 (worse) | `reports/training/GW-REC-011/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-012](#gw-rec-012) | 2026-10-23 | 07 | Phase 7 ablation: learning rate 5e-4 (short budget, vs GW-REC-003) | done | val CER 8.51% [8.34, 8.67]; +0.19 [+0.15, +0.23] vs GW-REC-003 (worse) | `reports/training/GW-REC-012/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-013](#gw-rec-013) | 2026-10-23 | 07 | Phase 7 ablation: learning rate 2e-3 (short budget, vs GW-REC-003) | done | val CER 8.34% [8.17, 8.50]; +0.02 [-0.02, +0.06] vs GW-REC-003 (no clear difference) | `reports/training/GW-REC-013/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-014](#gw-rec-014) | 2026-10-23 | 07 | Phase 7 ablation: weight decay 1e-2 (short budget, vs GW-REC-003) | done | val CER 8.28% [8.11, 8.44]; -0.04 [-0.08, -0.00] vs GW-REC-003 (better) | `reports/training/GW-REC-014/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-015](#gw-rec-015) | 2026-10-23 | 07 | Phase 7 ablation: no augmentation (short budget, vs GW-REC-003) | done | val CER 8.68% [8.50, 8.84]; +0.36 [+0.32, +0.41] vs GW-REC-003 (worse) | `reports/training/GW-REC-015/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-016](#gw-rec-016) | 2026-10-23 | 07 | Phase 7 ablation: geometric augmentation only (short budget, vs GW-REC-003) | done | val CER 8.34% [8.16, 8.50]; +0.02 [-0.02, +0.06] vs GW-REC-003 (no clear difference) | `reports/training/GW-REC-016/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-017](#gw-rec-017) | 2026-10-23 | 07 | Phase 7 ablation: photometric augmentation only (short budget, vs GW-REC-003) | done | val CER 8.59% [8.42, 8.75]; +0.28 [+0.23, +0.32] vs GW-REC-003 (worse) | `reports/training/GW-REC-017/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-018](#gw-rec-018) | 2026-10-23 | 07 | Phase 7 ablation: degradation augmentation only (short budget, vs GW-REC-003) | done | val CER 8.68% [8.50, 8.85]; +0.36 [+0.32, +0.41] vs GW-REC-003 (worse) | `reports/training/GW-REC-018/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-019](#gw-rec-019) | 2026-10-23 | 07 | Phase 7 ablation: frame stride 8 (short budget, vs GW-REC-003) | failed | out of GPU memory at step 1,444 (see failures.md); re-run as GW-REC-021 | `reports/training/GW-REC-019/` |
| [GW-REC-020](#gw-rec-020) | 2026-10-23 | 07 | GLYPHARIS-Rec-M (SVTR-style) vs CRNN at matched budget (H6, exploratory) | done | val CER 8.18% [8.01, 8.35]; -0.13 [-0.18, -0.09] vs GW-REC-003 (better) | `reports/training/GW-REC-020/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-021](#gw-rec-021) | 2026-10-24 | 07 | Phase 7 ablation: frame stride 8 with CTC-aware batching (re-run of GW-REC-019; short budget, vs GW-REC-003) | done | val CER 10.93% [10.70, 11.16]; +2.61 [+2.48, +2.74] vs GW-REC-003 (worse) | `reports/training/GW-REC-021/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-022](#gw-rec-022) | 2026-10-25 | 07 | Phase 7 combination: GLYPHARIS-Rec-M (SVTR) at input height 48 with weight decay 1e-2 (short budget, vs GW-REC-003) | done | val CER 7.56% [7.40, 7.71]; -0.76 [-0.81, -0.71] vs GW-REC-003 (better) | `reports/training/GW-REC-022/`, `reports/training/phase07/ablation_table.md` |
| [GW-REC-023](#gw-rec-023) | 2026-10-25 | 07 | Phase 7 combination without the encoder change: CRNN at input height 48 with weight decay 1e-2 (short budget, vs GW-REC-003) | abandoned | not run: GW-REC-022 settled the question (see entry) | — |
| [GW-REC-024](#gw-rec-024) | 2026-10-26 | 07 | Phase 7 full-budget confirmation: GW-REC-022's recipe at GW-REC-001's budget (30 epochs, one seed), vs GW-REC-001 | done | val CER 6.46% [6.33, 6.60]; -1.07 [-1.13, -1.01] vs GW-REC-001 | `reports/training/GW-REC-024/`, `reports/training/compare/GW-REC-024_vs_GW-REC-001.md` |
| [GW-DET-001](#gw-det-001) | 2026-10-25 | 08 | GLYPHARIS-Det baseline: DBNet-style, MobileNetV3-Large, ImageNet init, HierText pages (control for H9 and H10) | done | full val F1@0.5 0.724 [0.708, 0.739] (P 0.924, R 0.595) | `reports/training/GW-DET-001/` |
| [GW-DET-002](#gw-det-002) | 2026-10-25 | 08 | GW-DET-001 with a randomly initialised backbone (H9: ImageNet vs scratch, exploratory) | done | full val F1@0.5 0.704 [0.689, 0.719]; −0.020 [−0.023, −0.016] vs GW-DET-001: H9 supported | `reports/training/GW-DET-002/`, `reports/training/GW-DET-001_vs_GW-DET-002.md` |
| [GW-DET-003](#gw-det-003) | 2026-10-25 | 08 | GW-DET-001 with a MobileNetV3-Small backbone (H10: Small within 0.03 F1@0.5 of Large, exploratory) | done | full val F1@0.5 0.668 [0.651, 0.685]; −0.056 [−0.062, −0.051] vs GW-DET-001: H10 not supported | `reports/training/GW-DET-003/`, `reports/training/GW-DET-001_vs_GW-DET-003.md` |
| [GW-DET-004](#gw-det-004) | 2026-10-27 | 09 | GW-DET-001 with post-processing calibrated on validation half A, checked on half B (no retraining) | done | half B F1@0.5 0.754 [0.728, 0.775]; +0.032 [+0.025, +0.039] vs the record: adopted | `reports/error_cases/detector/` |
| [GW-EFF-001](#gw-eff-001) | 2026-10-27 | 09 | GW-DET-001 detector CPU latency at five input resolutions (PyTorch, batch 1) | done | detector alone 203 ms mean per page at 1280, inside the page target; long sides up to 1536 fit it only on an estimate (+24.4 ms per box), and the measured whole pipeline misses it (GW-EFF-002) | `reports/efficiency/` |
| [GW-DET-005](#gw-det-005) | 2026-10-27 | 09 | GW-DET-001 at the best input resolution that fits ADR-0011's page target (no retraining) | done | half B +0.018 [+0.013, +0.022] vs 1280: 1536 adopted; full val F1@0.5 0.778 [0.762, 0.792] | `reports/error_cases/detector/` |
| [GW-DET-006](#gw-det-006) | 2026-10-27 | 09 | Box shape `quad` vs `poly` on validation and its curved lines (RISK_REGISTER R-22) | done | no curved lines in HierText validation (all 4-point); quad kept; R-22 stays open | `reports/error_cases/detector/` |
| [GW-E2E-003](#gw-e2e-003) | 2026-10-28 | 10 | Line-confidence calibration: temperature fitted on validation half A, ECE on half B (H8) | done | ECE half B 0.217 → 0.089 at T = 2.35 (−0.128 [−0.133, −0.124]): adopted; H8 supported on validation | `reports/calibration/GW-E2E-003/` |
| [GW-E2E-001](#gw-e2e-001) | 2026-10-28 | 10 | GLYPHARIS pipeline (GW-DET-001 at Phase 9 settings + GW-REC-024) on HierText validation, with the baselines on the Phase 5 pages (H1) | done | page CER 27.6% [25.1, 30.4] on 1,715 pages; 24.3% vs 42.4% for PP-OCRv6 small on the Phase 5 pages (−18.1 [−21.4, −15.1]) | `reports/e2e/GW-E2E-001/` |
| [GW-E2E-002](#gw-e2e-002) | 2026-10-28 | 10 | Oracle analysis: ground-truth lines into the recognizer, to split page error between detection and recognition (H4) | done | oracle page CER 6.7% vs 27.6%: recognition's share 0.24 (0.20–0.29 per blur quartile); H4 not supported on HierText | `reports/e2e/GW-E2E-002/` |
| [GW-EFF-002](#gw-eff-002) | 2026-10-28 | 10 | Whole-pipeline CPU page time and memory on the Phase 5 profile pages (ADR-0011) | done | PyTorch page mean 2.0 s, p95 4.0 s: misses ADR-0011's page target (recognition 1.7 s); RAM 1,624 MiB meets it | `reports/efficiency/` |
| [GW-EFF-003](#gw-eff-003) | 2026-10-28 | 17 | ONNX equivalence of the v0.1 bundle against PyTorch on validation data | done | maps 7.3e-6, logits 3.7e-4, text identical on 5,000 lines, page CER equal: export successful | `reports/export/GW-EFF-003/` |
| [GW-EFF-004](#gw-eff-004) | 2026-10-28 | 17 | Whole-pipeline CPU page time on ONNX Runtime (ADR-0011's measurement) | done | ONNX page mean 5.5 s, p95 12.7 s, 11.2 GB: misses ADR-0011 by far (recognizer); PyTorch default for v0.1 | `reports/efficiency/` |
| [GW-E2E-004](#gw-e2e-004) | 2026-10-28 | 18 | The installed package (v0.1 bundle, ONNX) against the Phase 10 pipeline on the Phase 5 validation pages | done | page CER 24.33% vs 24.33% (+0.001 points), 16,827/16,835 lines identical: pass | `reports/package/` |
| [GW-DATA-001](#gw-data-001) | 2026-10-28 | 11 | Phase 11 arm A (no augmentation), seed 2024: GW-REC-015 with the seed changed | done | val CER 8.62% [8.46, 8.78]; GLYPH-C mCE 10.14% (arm A: control) | `reports/training/GW-DATA-001/`, `reports/robustness/GW-DATA-001.json` |
| [GW-DATA-002](#gw-data-002) | 2026-10-28 | 11 | Phase 11 arm A (no augmentation), seed 7: GW-REC-015 with the seed changed | done | val CER 8.69% [8.53, 8.86]; GLYPH-C mCE 10.28% (arm A: control) | `reports/training/GW-DATA-002/`, `reports/robustness/GW-DATA-002.json` |
| [GW-DATA-003](#gw-data-003) | 2026-10-28 | 11 | Phase 11 arm B (geometric augmentation), seed 2024: GW-REC-016 with the seed changed | done | val CER 8.29% [8.13, 8.45]; GLYPH-C mCE 8.51% (arm B: helps) | `reports/training/GW-DATA-003/`, `reports/robustness/GW-DATA-003.json` |
| [GW-DATA-004](#gw-data-004) | 2026-10-28 | 11 | Phase 11 arm B (geometric augmentation), seed 7: GW-REC-016 with the seed changed | done | val CER 8.38% [8.23, 8.55]; GLYPH-C mCE 8.76% (arm B: helps) | `reports/training/GW-DATA-004/`, `reports/robustness/GW-DATA-004.json` |
| [GW-DATA-005](#gw-data-005) | 2026-10-28 | 11 | Phase 11 arm C (photometric augmentation), seed 2024: GW-REC-017 with the seed changed | done | val CER 8.55% [8.39, 8.71]; GLYPH-C mCE 10.19% (arm C: no clear effect) | `reports/training/GW-DATA-005/`, `reports/robustness/GW-DATA-005.json` |
| [GW-DATA-006](#gw-data-006) | 2026-10-28 | 11 | Phase 11 arm C (photometric augmentation), seed 7: GW-REC-017 with the seed changed | done | val CER 8.56% [8.40, 8.73]; GLYPH-C mCE 10.32% (arm C: no clear effect) | `reports/training/GW-DATA-006/`, `reports/robustness/GW-DATA-006.json` |
| [GW-DATA-007](#gw-data-007) | 2026-10-28 | 11 | Phase 11 arm D (degradation augmentation), seed 2024: GW-REC-018 with the seed changed | done | val CER 8.66% [8.50, 8.82]; GLYPH-C mCE 10.03% (arm D: helps) | `reports/training/GW-DATA-007/`, `reports/robustness/GW-DATA-007.json` |
| [GW-DATA-008](#gw-data-008) | 2026-10-28 | 11 | Phase 11 arm D (degradation augmentation), seed 7: GW-REC-018 with the seed changed | done | val CER 8.70% [8.54, 8.87]; GLYPH-C mCE 10.04% (arm D: helps) | `reports/training/GW-DATA-008/`, `reports/robustness/GW-DATA-008.json` |
| [GW-DATA-009](#gw-data-009) | 2026-10-28 | 11 | Phase 11 robustness: the 15 arm runs on GLYPH-C v1 (mCE), paired comparison against arm A, verdict per family (H3) | done | ΔmCE vs no augmentation: geometric −1.67 [−1.73, −1.62], photometric −0.06 [−0.08, −0.02], degradation −0.25 [−0.30, −0.19], full −1.83 [−1.90, −1.75] points; H3 supported | `reports/robustness/` |
| [GW-DATA-010](#gw-data-010) | 2026-10-29 | 12 | Phase 12 arm +25K: HierText + 25,000 GlyphSynth lines at the control's step budget, seed 1337 | done | val CER 8.29% [8.14, 8.46]; document slice 11.18% (arm +25K: no clear effect) | `reports/training/GW-DATA-010/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-011](#gw-data-011) | 2026-10-29 | 12 | Phase 12 arm +25K: HierText + 25,000 GlyphSynth lines at the control's step budget, seed 2024 | done | val CER 8.29% [8.12, 8.46]; document slice 11.05% (arm +25K: no clear effect) | `reports/training/GW-DATA-011/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-012](#gw-data-012) | 2026-10-29 | 12 | Phase 12 arm +25K: HierText + 25,000 GlyphSynth lines at the control's step budget, seed 7 | done | val CER 8.28% [8.12, 8.44]; document slice 11.05% (arm +25K: no clear effect) | `reports/training/GW-DATA-012/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-013](#gw-data-013) | 2026-10-29 | 12 | Phase 12 arm +50K: HierText + 50,000 GlyphSynth lines at the control's step budget, seed 1337 | done | val CER 8.30% [8.15, 8.46]; document slice 11.18% (arm +50K: hurts) | `reports/training/GW-DATA-013/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-014](#gw-data-014) | 2026-10-29 | 12 | Phase 12 arm +100K: HierText + 100,000 GlyphSynth lines at the control's step budget, seed 1337 | done | val CER 8.39% [8.24, 8.56]; document slice 11.22% (arm +100K: hurts) | `reports/training/GW-DATA-014/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-015](#gw-data-015) | 2026-10-29 | 12 | Phase 12 arm +100K: HierText + 100,000 GlyphSynth lines at the control's step budget, seed 2024 | done | val CER 8.38% [8.22, 8.54]; document slice 11.22% (arm +100K: hurts) | `reports/training/GW-DATA-015/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-016](#gw-data-016) | 2026-10-29 | 12 | Phase 12 arm +100K: HierText + 100,000 GlyphSynth lines at the control's step budget, seed 7 | done | val CER 8.36% [8.21, 8.53]; document slice 11.21% (arm +100K: hurts) | `reports/training/GW-DATA-016/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-017](#gw-data-017) | 2026-10-29 | 12 | Phase 12 arm +250K: HierText + 250,000 GlyphSynth lines at the control's step budget, seed 1337 | done | val CER 8.58% [8.42, 8.75]; document slice 11.46% (arm +250K: hurts) | `reports/training/GW-DATA-017/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-018](#gw-data-018) | 2026-10-29 | 12 | Phase 12 arm +250K: HierText + 250,000 GlyphSynth lines at the control's step budget, seed 2024 | done | val CER 8.57% [8.40, 8.74]; document slice 11.49% (arm +250K: hurts) | `reports/training/GW-DATA-018/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-019](#gw-data-019) | 2026-10-29 | 12 | Phase 12 arm +250K: HierText + 250,000 GlyphSynth lines at the control's step budget, seed 7 | done | val CER 8.52% [8.36, 8.69]; document slice 11.38% (arm +250K: hurts) | `reports/training/GW-DATA-019/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-020](#gw-data-020) | 2026-10-29 | 12 | Phase 12 arm +100K, 6 fonts: HierText + 100,000 (6 font families) GlyphSynth lines at the control's step budget, seed 1337 | done | val CER 8.40% [8.24, 8.56]; document slice 11.13% (arm +100K, 6 fonts: no clear effect) | `reports/training/GW-DATA-020/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-021](#gw-data-021) | 2026-10-29 | 12 | Phase 12 arm +100K, no capture: HierText + 100,000 (no capture effects) GlyphSynth lines at the control's step budget, seed 1337 | done | val CER 8.38% [8.22, 8.54]; document slice 11.09% (arm +100K, no capture: no clear effect) | `reports/training/GW-DATA-021/`, `reports/synthetic/phase12_comparison.json` |
| [GW-DATA-022](#gw-data-022) | 2026-10-29 | 12 | Phase 12 comparison: the synthetic arms against HierText only on the document slice (H2′a), and the font and capture factors (H2′b) | done | document slice vs HierText only: +25K +0.04 [−0.03, +0.11], +100K +0.16 [+0.06, +0.26], +250K +0.39 [+0.25, +0.52]; H2′a and H2′b not supported | `reports/synthetic/` |
| [GW-REC-025](#gw-rec-025) | 2026-10-30 | 13 | Phase 13 L3 (+ input height 48), seed 2024: GW-REC-006 with the seed changed | done | val CER 7.82% [7.67, 7.98]; GLYPH-C mCE 6.78% | `reports/training/GW-REC-025/`, `reports/robustness/GW-REC-025.json` |
| [GW-REC-026](#gw-rec-026) | 2026-10-30 | 13 | Phase 13 L3 (+ input height 48), seed 7: GW-REC-006 with the seed changed | done | val CER 7.83% [7.68, 7.99]; GLYPH-C mCE 6.84% | `reports/training/GW-REC-026/`, `reports/robustness/GW-REC-026.json` |
| [GW-REC-027](#gw-rec-027) | 2026-10-30 | 13 | Phase 13 L4 (full model), seed 2024: GW-REC-022 with the seed changed | done | val CER 7.59% [7.44, 7.74]; GLYPH-C mCE 6.25% | `reports/training/GW-REC-027/`, `reports/robustness/GW-REC-027.json` |
| [GW-REC-028](#gw-rec-028) | 2026-10-30 | 13 | Phase 13 L4 (full model), seed 7: GW-REC-022 with the seed changed | done | val CER 7.56% [7.41, 7.72]; GLYPH-C mCE 6.33% | `reports/training/GW-REC-028/`, `reports/robustness/GW-REC-028.json` |
| [GW-REC-029](#gw-rec-029) | 2026-10-30 | 13 | Phase 13 L5 (+ wider rotation), seed 1337: GW-REC-022 with rotation up to ±4° on half the lines | done | val CER 7.48% [7.34, 7.64]; GLYPH-C mCE 4.84% | `reports/training/GW-REC-029/`, `reports/robustness/GW-REC-029.json` |
| [GW-REC-030](#gw-rec-030) | 2026-10-30 | 13 | Phase 13 L5 (+ wider rotation), seed 2024: GW-REC-022 with rotation up to ±4° on half the lines | done | val CER 7.43% [7.28, 7.58]; GLYPH-C mCE 4.79% | `reports/training/GW-REC-030/`, `reports/robustness/GW-REC-030.json` |
| [GW-REC-031](#gw-rec-031) | 2026-10-30 | 13 | Phase 13 L5 (+ wider rotation), seed 7: GW-REC-022 with rotation up to ±4° on half the lines | done | val CER 7.44% [7.30, 7.60]; GLYPH-C mCE 4.78% | `reports/training/GW-REC-031/`, `reports/robustness/GW-REC-031.json` |
| [GW-REC-032](#gw-rec-032) | 2026-10-30 | 13 | Phase 13 final, option A, seed 2024: GW-REC-024 with the seed changed (trained only for option A) | not run | option B was chosen (GW-REC-037) | — |
| [GW-REC-033](#gw-rec-033) | 2026-10-30 | 13 | Phase 13 final, option A, seed 7: GW-REC-024 with the seed changed (trained only for option A) | not run | option B was chosen (GW-REC-037) | — |
| [GW-REC-034](#gw-rec-034) | 2026-10-30 | 13 | Phase 13 final, option B, seed 1337: GW-REC-024 with L5's rotation (trained only for option B) | done | val CER 6.30% [6.17, 6.44]; GLYPH-C mCE 3.54% | `reports/training/GW-REC-034/`, `reports/robustness/GW-REC-034.json` |
| [GW-REC-035](#gw-rec-035) | 2026-10-30 | 13 | Phase 13 final, option B, seed 2024: GW-REC-024 with L5's rotation (trained only for option B) | done | val CER 6.28% [6.15, 6.42]; GLYPH-C mCE 3.53% | `reports/training/GW-REC-035/`, `reports/robustness/GW-REC-035.json` |
| [GW-REC-036](#gw-rec-036) | 2026-10-30 | 13 | Phase 13 final, option B, seed 7: GW-REC-024 with L5's rotation (trained only for option B) | done | val CER 6.30% [6.17, 6.44]; GLYPH-C mCE 3.53% | `reports/training/GW-REC-036/`, `reports/robustness/GW-REC-036.json` |
| [GW-DET-007](#gw-det-007) | 2026-10-30 | 13 | GW-DET-001 with labels for smaller text (min_text_size 3) and training rotation up to 30° (Phase 9's deferred fixes) | done | F1@0.5 0.7668 vs 0.7779 (−0.0111 [−0.0146, −0.0075]): not adopted | `reports/training/GW-DET-007/`, `reports/error_cases/detector/GW-DET-007/pair.json` |
| [GW-REC-037](#gw-rec-037) | 2026-10-30 | 13 | Phase 13 ablation ladder (L0 to L5) and the final recognizer at the full budget | done | ladder L0 to L5 8.66 to 7.45%, L2 not kept; option B; final 6.29% vs 7.53% (−1.24 [−1.40, −1.11]) | `reports/ablation/` |
| [GW-E2E-005](#gw-e2e-005) | 2026-11-14 | 13 | Line-confidence calibration of the frozen recognizer GW-REC-034, as GW-E2E-003 | done | ECE half B 0.214 → 0.089 at T = 2.35 (−0.125 [−0.130, −0.122]): adopted | `reports/calibration/GW-E2E-005/` |
| [GW-E2E-006](#gw-e2e-006) | 2026-11-15 | 14 | The frozen v1 pipeline on HierText validation: Phase 14's reference | done | bag-of-lines page CER 27.57%, E2E F1 58.24% (all 1,715 validation pages) | `reports/external/` |
| [GW-REC-038](#gw-rec-038) | 2026-11-15 | 14 | The frozen recognizer on the HierText validation evaluation crops (H5′'s reference) | done | 6.51% CER strict, 74,500 crops | `reports/external/` |
| [GW-EFF-005](#gw-eff-005) | 2026-11-15 | 14 | v1 CPU page latency against PP-OCRv5 mobile in one sitting (H1′ latency) | done | 2,446.7 against 2,954.8 ms per page, −508.1 [−737.7, −256.5] ms | `reports/external/GW-EFF-005/`, `reports/external/results.json` |
| [GW-REC-039](#gw-rec-039) | 2026-11-15 | 14 | The frozen recognizer on the test crop packs (H5′, H8′) | done | CER 6.03% HierText test, 4.17% GDC, 7.03% FUNSD; IIIT5K 91.47% words | `reports/external/` |
| [GW-E2E-007](#gw-e2e-007) | 2026-11-15 | 14 | The frozen pipeline end to end on the test pages (H1′ CER) | done | bag-of-lines page CER 25.16% HierText test, 22.35% GDC | `reports/external/` |
| [GW-E2E-008](#gw-e2e-008) | 2026-11-15 | 14 | Oracle run on GDC and GDC-scan, for H4 in Phase 15 | done | GDC bag-of-lines page CER 4.17% from ground-truth boxes | `reports/external/` |
| [GW-E2E-009](#gw-e2e-009) | 2026-11-15 | 14 | Phase 14 scoring, comparisons and hypothesis verdicts | done | H1′ holds; H5′ not met; H8′ transfers met, by half not shown | `reports/external/` |
| [GW-REC-040](#gw-rec-040) | 2026-11-17 | 15 | Recognition errors of the frozen recognizer on the Phase 14 hand-off (RQ5) | done | GDC: deletions 3.19 of 4.17% CER, full stops 44.176% deleted, the 20 worst lines dot leaders; HierText test: rotation ≥ 5° 34.92% of errors | `reports/error_cases/phase15/`, `reports/confusion_analysis/GW-REC-040/` |
| [GW-E2E-010](#gw-e2e-010) | 2026-11-17 | 15 | Pipeline errors: page error budget, detection, reading order, the oracle share (RQ5, H4) | done | missed lines 49.45% of HierText test and 45.32% of GDC page error; GDC oracle share 18.66% (clean 32.70%); H4 not supported on GDC (post hoc) | `reports/error_cases/phase15/` |
| [GW-E2E-011](#gw-e2e-011) | 2026-11-18 | 18 (v1) | The package on the v1 bundle (PyTorch backend) against the frozen v1 pipeline (GW-E2E-006) on the 400 Phase 5 validation pages | done | 24.32% page CER on both, difference 0.0; 16,835 of 16,835 lines with identical text and box: pass | `reports/package/` |
| [GW-E2E-012](#gw-e2e-012) | 2026-11-18 | 20 (v1) | Failure paths, fuzz-lite and damaged bundles on the v1 bundle, both backends | done | pass on both backends: 269 inputs each, 47 results and 222 typed errors, no other exception; all 32 damaged-bundle cases as registered | `reports/hardening/` |
| [GW-REC-041](#gw-rec-041) | 2026-11-17 | 15 | The frozen recognizer on GlyphSynth validation, sliced by font and capture labels | done | CER 0.71% [0.63, 0.79]; ink spread 1.41% against 0.58% without | `reports/error_cases/phase15/` |
| [GW-EFF-006](#gw-eff-006) | 2026-11-17 | 16 | Reference outputs of PyTorch v1 on the CPU; the v1 fp32 ONNX bundle from the freeze record, checked against them | done | Phase 17's four checks pass: maps 7.33e-6, logits 3.09e-4, text identical, page CER within 1.49e-5; bundle `875357a6…` | `reports/efficiency/phase16/` |
| [GW-EFF-007](#gw-eff-007) | 2026-11-17 | 16 | The v1 cost table: parameters, sizes, CPU and GPU latency, throughput, memory, page-time split | done | 11,116,044 parameters; CPU pages at 10 threads: PyTorch 2,566 ms, 1,641.8 MiB; fp32 ONNX 8,574 ms, 6,529.6 MiB (recognizer network 95%); GPU 1,049 ms | `reports/efficiency/phase16/`, `reports/benchmark_results/efficiency_comparison.md` |
| [GW-EFF-008](#gw-eff-008) | 2026-11-17 | 16 | Dynamic INT8 of the v1 ONNX recognizer (H7′) | done | 61.84 against 95.78 ms per line, −10.00 [−12.79, −7.37]; CER 6.208 against 6.221%, −0.512 [−0.541, −0.484]; H7′ holds | `reports/efficiency/phase16/` |
| [GW-EFF-009](#gw-eff-009) | 2026-11-17 | 16 | Pareto points: input sizes, INT8, mixed backends, fused attention, other model sizes (new versions) | done | lowest mean page time at v1's accuracy: ONNX detector with PyTorch recognizer, 2,579.1 ms per page; fused attention 4,020.8 against 9,739.4 ms; detector size moves CER, ONNX page time by at most about 13% | `reports/efficiency/phase16/` |
| [GW-EFF-010](#gw-eff-010) | 2026-11-17 | 16 | H1′'s ONNX latency: a fresh GW-BASE-002 profile and the v1 ONNX bundle in one sitting | done | 9,623.6 against 3,086.0 ms per page, +6,537.6 [+5,383.7, +7,806.3] ms: H1′ ONNX not met (bundle `875357a6…`) | `reports/efficiency/phase16/` |
| [GW-EFF-011](#gw-eff-011) | 2026-11-17 | 16 | H6′'s latency: CRNN GW-REC-003 against SVTR GW-REC-020 in ONNX, per line | done | SVTR 32.01 against CRNN 10.16 ms per line, +16.78 [+12.02, +22.04] ms: H6′ not met | `reports/efficiency/phase16/` |
| [GW-EFF-012](#gw-eff-012) | 2026-11-18 | 17 (v1) | The v1 bundle on the PyTorch backend against the Phase 16 PyTorch v1 references (the shipped path) | done | all four tolerances pass with no difference: map and logits 0.0, 5,000 of 5,000 lines identical, page CER 24.32% on both | `reports/export/GW-EFF-012/` |
| [GW-E2E-013](#gw-e2e-013) | 2026-11-18 | 22 | Clean-machine-style walkthrough: the installed package on the v1 bundle against GW-E2E-006 | done | passes: page CER 24.32% on both, difference 0.0 points; 16,835 of 16,835 lines identical | `reports/package/GW-E2E-013.json` |
| [GW-E2E-014](#gw-e2e-014) | 2026-11-18 | 22 | v1 re-run rehearsal on HierText validation pages | done | passes: 13 point values, largest difference 0.0; 1,715 of 1,715 pages identical | `reports/final_validation/rehearsal.json` |
| [GW-E2E-015](#gw-e2e-015) | 2026-11-18 | 22 | Pass A: re-score the saved Phase 14 files with the unchanged scorer | done | not identical: 40 values differ, all official OmniDocBench scores of Tesseract and GLYPHARIS (more pages hit the evaluator's time limit); 0 unexplained | `reports/final_validation/GW-E2E-015/` |
| [GW-E2E-016](#gw-e2e-016) | 2026-11-18 | 22 | v1 re-run on the test pages | done | predictions identical to GW-E2E-007 in all 5 sets; scored by GW-E2E-018 | `reports/final_validation/v1_rerun_outputs.json`, `reports/final_validation/GW-E2E-018/line_comparison.json` |
| [GW-E2E-017](#gw-e2e-017) | 2026-11-18 | 22 | v1 re-run of the oracle on GDC and GDC-scan | done | predictions identical to GW-E2E-008 on GDC and GDC-scan; scored by GW-E2E-018 | `reports/final_validation/v1_rerun_outputs.json`, `reports/final_validation/GW-E2E-018/line_comparison.json` |
| [GW-E2E-018](#gw-e2e-018) | 2026-11-18 | 22 | Pass B: score the v1 re-run; the 0.05-point tolerance verdict | done | tolerance verdict failed: 16 of 1,381 point values outside 0.0005, all GLYPHARIS's official OmniDocBench scores (largest 0.0208, three-column slice); no count, flag or verdict changed | `reports/final_validation/v1_reproduction.md`, `reports/final_validation/GW-E2E-018/` |
| [GW-E2E-019](#gw-e2e-019) | 2026-11-18 | 22 | Phone candidate: validation gate on pages, against GW-E2E-006 | planned | — | — |
| [GW-E2E-020](#gw-e2e-020) | 2026-11-18 | 22 | Phone candidate on the test pages | planned | — | — |
| [GW-E2E-021](#gw-e2e-021) | 2026-11-18 | 22 | Phone candidate: the scoring pass and H11 | planned | — | — |
| [GW-REC-042](#gw-rec-042) | 2026-11-18 | 22 | v1 re-run rehearsal on HierText validation crops | done | passes: 10 point values, largest difference 0.0; 74,500 of 74,500 crops identical | `reports/final_validation/rehearsal.json` |
| [GW-REC-043](#gw-rec-043) | 2026-11-18 | 22 | v1 re-run on the test crops | done | predictions identical to GW-REC-039 in all 5 sets; scored by GW-E2E-018 | `reports/final_validation/v1_rerun_outputs.json`, `reports/final_validation/GW-E2E-018/line_comparison.json` |
| [GW-REC-044](#gw-rec-044) | 2026-11-18 | 22 | Phone candidate: validation gate on crops, against GW-REC-038 | planned | — | — |
| [GW-REC-045](#gw-rec-045) | 2026-11-18 | 22 | Phone candidate on the test crops | planned | — | — |
| [GW-EFF-013](#gw-eff-013) | 2026-11-18 | 22 | Phone candidate selection: `fused`, `fused-int8`, `fused-single-line` | planned | — | — |
| [GW-EFF-014](#gw-eff-014) | 2026-11-18 | 22 | Phone candidate bundle and its equivalence check | planned | — | — |
| [GW-EFF-015](#gw-eff-015) | 2026-11-18 | 22 | Android operator check, then the phone measurement on named devices | planned | — | — |

### <a id="gw-base-001"></a>GW-BASE-001 — Tesseract 5 (LSTM, eng): recognition and end to end

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
- **Amendment (2026-11-15, Phase 14, before the runs):** the same system, runner, settings and scorer run on the Phase 14 sets as fixed in `docs/PHASE_STATUS.md` (Phase 14, item 4): IIIT5K, FUNSD and GDC-scan recognition; FUNSD and GDC-scan end to end; OmniDocBench end to end on all 755 English pages. Predictions and profiles go to `data/processed/baselines_phase14/GW-BASE-001/`; the Phase 5 files are re-scored once in GW-E2E-009, never re-run or rewritten.
- **Result (Phase 14, 2026-11-17, GW-E2E-009):** strict track, `reports/external/`. IIIT5K 66.17% words (alphanumeric); FUNSD crops 18.48%, GDC-scan crops 2.48%; pages: FUNSD E2E F1 19.77% (bag-of-lines CER 111.81%), GDC-scan bag-of-lines CER 31.39% (order-aware 17.81%), OmniDocBench order-aware 15.70% [14.23, 17.37] and official text-block edit distance 0.0966, the best of all systems there.

### <a id="gw-base-002"></a>GW-BASE-002 — PaddleOCR PP-OCRv5 mobile (det + English rec): recognition and end to end

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
- **Amendment (2026-11-15, Phase 14, before the runs):** the same system, runner, settings and scorer run on the Phase 14 sets as fixed in `docs/PHASE_STATUS.md` (Phase 14, item 4): IIIT5K, FUNSD and GDC-scan recognition; FUNSD and GDC-scan end to end; OmniDocBench end to end on all 755 English pages; and a fresh validation profile for H1′ in one sitting with GW-EFF-005. Predictions and profiles go to `data/processed/baselines_phase14/GW-BASE-002/`; the Phase 5 files are re-scored once in GW-E2E-009, never re-run or rewritten.
- **Result (Phase 14, 2026-11-17, GW-E2E-009):** strict track, `reports/external/`. IIIT5K 94.93% words; FUNSD crops 18.34%, GDC-scan crops 1.41%; pages: FUNSD E2E F1 62.65%, GDC-scan bag-of-lines CER 12.94%, OmniDocBench order-aware 60.52% (text-block edit distance 0.4217). Fresh 10-thread profile for H1′: 2,954.8 ms per page (GW-EFF-005).

- **Amendment (2026-11-17, Phase 16, before the run):** one fresh validation profile for H1′'s ONNX verdict (GW-EFF-010), with the same runner, settings and profiling path as the Phase 14 profile (every settings field asserted equal), through `scripts/run_baselines.py --phase 16` in `.venv-baselines`; output `data/processed/baselines_phase16/GW-BASE-002/`. No test data.
- **Result (Phase 16, 2026-11-18, GW-EFF-010):** fresh 10-thread validation profile, the sitting's first (2026-11-17, 16:40:45 to 16:46:30Z), mains, Balanced plan, power mode "best performance" at start and end, every settings field equal to the Phase 14 profile's: 3,086.0 ms per page mean (`reports/efficiency/phase16/verdicts.md`, with the sitting times). The profile itself (`data/processed/baselines_phase16/GW-BASE-002/profile_ppocr_mobile_e2e_hiertext_val.json`) is not committed; its SHA-256 is in `reports/efficiency/phase16/results.json`.

### <a id="gw-base-003"></a>GW-BASE-003 — PaddleOCR PP-OCRv5 server (det + rec): recognition and end to end

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
- **Amendment (2026-11-15, Phase 14, before the runs):** the same system, runner, settings and scorer run on the Phase 14 sets as fixed in `docs/PHASE_STATUS.md` (Phase 14, item 4): IIIT5K, FUNSD and GDC-scan recognition; FUNSD end to end; OmniDocBench end to end on the fixed 100-page subset (`data/manifests/processed/omnidocbench_subset100.json`). Not run: GDC-scan end to end (it died on GDC's photos of the same size, `research/failures.md`). Predictions and profiles go to `data/processed/baselines_phase14/GW-BASE-003/`; the Phase 5 files are re-scored once in GW-E2E-009, never re-run or rewritten.
- **Result (Phase 14, 2026-11-17, GW-E2E-009):** strict track, `reports/external/`. IIIT5K 94.80% words; FUNSD crops 29.20%, GDC-scan crops 11.89%; FUNSD E2E F1 61.73%. OmniDocBench subset: not runnable (its detector dies on inputs above about 8 megapixels; `research/failures.md`, 2026-11-17), no substitute.

### <a id="gw-base-004"></a>GW-BASE-004 — TrOCR-small-printed: recognition only

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
- **Amendment (2026-11-15, Phase 14, before the runs):** the same system, runner, settings and scorer run on the Phase 14 sets as fixed in `docs/PHASE_STATUS.md` (Phase 14, item 4): IIIT5K, FUNSD and GDC-scan recognition (no detector). Predictions and profiles go to `data/processed/baselines_phase14/GW-BASE-004/`; the Phase 5 files are re-scored once in GW-E2E-009, never re-run or rewritten.
- **Result (Phase 14, 2026-11-17, GW-E2E-009):** strict track, `reports/external/`. IIIT5K 71.13% words; crops FUNSD 48.13% strict (12.34% normalised), GDC-scan 75.09% strict (16.78% normalised).

### <a id="gw-base-005"></a>GW-BASE-005 — PaddleOCR PP-OCRv6 tiny (det + rec): recognition and end to end

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
- **Amendment (2026-11-15, Phase 14, before the runs):** the same system, runner, settings and scorer run on the Phase 14 sets as fixed in `docs/PHASE_STATUS.md` (Phase 14, item 4): IIIT5K, FUNSD and GDC-scan recognition; FUNSD and GDC-scan end to end; OmniDocBench end to end on all 755 English pages. Predictions and profiles go to `data/processed/baselines_phase14/GW-BASE-005/`; the Phase 5 files are re-scored once in GW-E2E-009, never re-run or rewritten.
- **Result (Phase 14, 2026-11-17, GW-E2E-009):** strict track, `reports/external/`. IIIT5K 89.13% words; FUNSD crops 10.35%, GDC-scan crops 0.86%; pages: FUNSD E2E F1 59.29%, GDC-scan bag-of-lines CER 6.83%, OmniDocBench order-aware 60.62%.

### <a id="gw-base-006"></a>GW-BASE-006 — PaddleOCR PP-OCRv6 small (det + rec): recognition and end to end

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
- **Amendment (2026-11-15, Phase 14, before the runs):** the same system, runner, settings and scorer run on the Phase 14 sets as fixed in `docs/PHASE_STATUS.md` (Phase 14, item 4): IIIT5K, FUNSD and GDC-scan recognition; FUNSD and GDC-scan end to end; OmniDocBench end to end on all 755 English pages. Predictions and profiles go to `data/processed/baselines_phase14/GW-BASE-006/`; the Phase 5 files are re-scored once in GW-E2E-009, never re-run or rewritten.
- **Result (Phase 14, 2026-11-17, GW-E2E-009):** strict track, `reports/external/`. IIIT5K 96.43% words; FUNSD crops 9.30%, GDC-scan crops 0.37%; pages: FUNSD E2E F1 64.79%, GDC-scan bag-of-lines CER 6.68%, OmniDocBench order-aware 60.29%.

### <a id="gw-base-007"></a>GW-BASE-007 — PaddleOCR PP-OCRv6 medium (PaddleOCR 3.7's English default): reduced scope

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
- **Amendment (2026-11-15, Phase 14, before the runs):** the same system, runner, settings and scorer run on the Phase 14 sets as fixed in `docs/PHASE_STATUS.md` (Phase 14, item 4): IIIT5K, FUNSD and GDC-scan recognition; FUNSD and GDC-scan end to end; OmniDocBench end to end on all 755 English pages. Predictions and profiles go to `data/processed/baselines_phase14/GW-BASE-007/`; the Phase 5 files are re-scored once in GW-E2E-009, never re-run or rewritten.
- **Result (Phase 14, 2026-11-17, GW-E2E-009):** strict track, `reports/external/`. IIIT5K 97.13% words; FUNSD crops 17.91%, GDC-scan crops 0.48%; pages: FUNSD E2E F1 67.70%, GDC-scan bag-of-lines CER 4.94%, OmniDocBench order-aware 60.35%.

### <a id="gw-rec-001"></a>GW-REC-001 — CRNN baseline on HierText lines

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

### <a id="gw-rec-002"></a>GW-REC-002 — GW-REC-001 plus a small GlyphSynth mix

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

### <a id="gw-rec-003"></a>GW-REC-003 — Phase 7 control, short budget, seed 1337

- **Date allocated / run:** 2026-10-23 / 2026-10-23
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

### <a id="gw-rec-004"></a>GW-REC-004 — Phase 7 control, short budget, seed 2024

- **Date allocated / run:** 2026-10-23 / 2026-10-23
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

### <a id="gw-rec-005"></a>GW-REC-005 — Phase 7 control, short budget, seed 7

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

### <a id="gw-rec-006"></a>GW-REC-006 — Phase 7 ablation: input height 48

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

### <a id="gw-rec-007"></a>GW-REC-007 — Phase 7 ablation: CNN half as wide

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

### <a id="gw-rec-008"></a>GW-REC-008 — Phase 7 ablation: BiLSTM hidden 128

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

### <a id="gw-rec-009"></a>GW-REC-009 — Phase 7 ablation: one BiLSTM layer

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

### <a id="gw-rec-010"></a>GW-REC-010 — Phase 7 ablation: Adam

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

### <a id="gw-rec-011"></a>GW-REC-011 — Phase 7 ablation: cosine schedule

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

### <a id="gw-rec-012"></a>GW-REC-012 — Phase 7 ablation: learning rate 5e-4

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

### <a id="gw-rec-013"></a>GW-REC-013 — Phase 7 ablation: learning rate 2e-3

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

### <a id="gw-rec-014"></a>GW-REC-014 — Phase 7 ablation: weight decay 1e-2

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

### <a id="gw-rec-015"></a>GW-REC-015 — Phase 7 ablation: no augmentation

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

### <a id="gw-rec-016"></a>GW-REC-016 — Phase 7 ablation: geometric augmentation only

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

### <a id="gw-rec-017"></a>GW-REC-017 — Phase 7 ablation: photometric augmentation only

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

### <a id="gw-rec-018"></a>GW-REC-018 — Phase 7 ablation: degradation augmentation only

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

### <a id="gw-rec-019"></a>GW-REC-019 — Phase 7 ablation: frame stride 8

- **Date allocated / run:** 2026-10-23 / 2026-10-24
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
- **Training diagnostics:** not applicable (no epoch finished; `reports/training/GW-REC-019/report.md`)
- **Interpretation:** not applicable (no result; the cause and the fix are in `research/failures.md`)
- **Limitations:** single seed; short budget.
- **Next experiment:** the Phase 7 combination and full-budget confirmation
- **Training time:** not applicable (no epoch finished)

### <a id="gw-rec-020"></a>GW-REC-020 — GLYPHARIS-Rec-M vs CRNN at matched budget (H6)

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

### <a id="gw-rec-021"></a>GW-REC-021 — Phase 7 ablation: frame stride 8, CTC-aware batching (re-run of GW-REC-019)

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

### <a id="gw-rec-022"></a>GW-REC-022 — Phase 7 combination: GLYPHARIS-Rec-M (SVTR) at input height 48, weight decay 1e-2

- **Date allocated / run:** 2026-10-25 / 2026-10-25
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
- **Result:** val CER 7.56% [7.40, 7.71]; against GW-REC-003 -0.76 points [-0.81, -0.71] (paired bootstrap over lines, 1,000 resamples): **better** under the Phase 7 decision rule (`reports/training/phase07/ablation_table.md`). The interval is the ablation table's, as for the other Phase 7 arms; the run report prints [7.41, 7.72] for the same 7.56% (`reports/training/GW-REC-022/report.md`). Against GW-REC-006 -0.27 points [-0.32, -0.22] (`reports/training/compare/GW-REC-022_vs_GW-REC-006.md`) and against GW-REC-020 -0.62 [-0.67, -0.58] (`reports/training/compare/GW-REC-022_vs_GW-REC-020.md`). 8 epochs, 45,295 optimizer steps, the final epoch was the best; 7,928,835 parameters. CPU latency 22.8 ms mean and 51.5 ms p95 per line (PyTorch; `reports/training/phase07/cpu_latency.md`), inside ADR-0011's line target.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-022/report.md`)
- **Interpretation:** the hypothesis of a gain below the sum did not hold (finding F3). The combination gains −0.76 points against the control, while input height 48 (−0.49), the SVTR-style encoder (−0.13) and weight decay (−0.04) add up to −0.66, and its CI lies below that sum. The parts are separate single-seed runs, so the excess over the sum has no interval of its own. Latency rose above GW-REC-020's 13.5 ms per line as expected, but stayed inside ADR-0011's line target. This is the configuration taken to the full-budget confirmation (GW-REC-024).
- **Limitations:** single seed; short budget; micro-batches differ from the control's.
- **Next experiment:** the full-budget confirmation of the chosen configuration
- **Training time:** 4.65 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-023"></a>GW-REC-023 — Phase 7 combination without the encoder change: CRNN at input height 48, weight decay 1e-2

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
- **Result:** not run (abandoned 2026-10-26, before it started). GW-REC-022 beat GW-REC-006 by −0.27 points [−0.32, −0.22] (`reports/training/compare/GW-REC-022_vs_GW-REC-006.md`), so the SVTR encoder's gain survives the combination; weight decay alone was worth −0.04 (GW-REC-014), so this run could only have isolated that small share. Its GPU time went to the full-budget confirmation instead. Recorded in `research/failures.md`.
- **Training diagnostics:** —
- **Interpretation:** —
- **Limitations:** single seed; short budget.
- **Next experiment:** the full-budget confirmation of the chosen configuration
- **Training time:** —

### <a id="gw-rec-024"></a>GW-REC-024 — Phase 7 full-budget confirmation: GW-REC-022's recipe at 30 epochs

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
- **Amendment (2026-11-18, Phase 21):** the promise above is closed for v1, without those runs. The two seeds were registered as GW-REC-032 (seed 2024) and GW-REC-033 (seed 7), Phase 13's option A. They were not run, because the ladder kept row L5 and option B was chosen (GW-REC-037). The final recognizer has three seeds at the full budget instead, GW-REC-034 to 036 (seeds 1337, 2024 and 7; validation CER 6.30, 6.28 and 6.30%, `reports/ablation/phase13_final.md`). GW-REC-024, the v0.1 recognizer, stays at one seed, and no registered run used an A100.
- **Result:** val CER 6.46% [6.33, 6.60] (95% bootstrap CI over 73,888 lines); against GW-REC-001 (7.53%) -1.07 [-1.13, -1.01] points CER and -1.94 [-2.10, -1.79] points WER (paired, `reports/training/compare/GW-REC-024_vs_GW-REC-001.md`). 30 epochs, 169,860 optimizer steps; 7,928,835 parameters. CPU latency 24.4 ms mean and 60.3 ms p95 per line (PyTorch, idle machine; `reports/training/phase07/cpu_latency.md`): within ADR-0011's line target, with little margin on the mean.
- **Training diagnostics:** No flags raised. One crash mid-run (CUDA "unknown error" at epoch 23 under memory pressure); the queue resumed it from its last checkpoint the same minute (`reports/training/GW-REC-024/report.md`, Notes).
- **Interpretation:** the hypothesis holds: the configuration chosen on the short budget also wins at the full budget, by about 1.1 points of CER, more than its short-budget gain against the control (−0.76). It is the Phase 7 recognizer configuration for v0.1. Its CPU latency is the main cost: about 2.3 times the CRNN control's per line, just inside the target.
- **Limitations:** one seed; compared with a single GW-REC-001 run.
- **Next experiment:** Phase 8 detector runs; the recognizer is exported in Phase 17.
- **Training time:** 19.31 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-det-001"></a>GW-DET-001 — GLYPHARIS-Det baseline: DBNet-style, MobileNetV3-Large, ImageNet init

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

### <a id="gw-det-002"></a>GW-DET-002 — GW-DET-001 with a randomly initialised backbone (H9)

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

### <a id="gw-det-003"></a>GW-DET-003 — GW-DET-001 with a MobileNetV3-Small backbone (H10)

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

### <a id="gw-det-004"></a>GW-DET-004 — Post-processing calibrated on validation

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

### <a id="gw-eff-001"></a>GW-EFF-001 — Detector CPU latency by input resolution

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

### <a id="gw-det-005"></a>GW-DET-005 — Input resolution

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

### <a id="gw-det-006"></a>GW-DET-006 — Box shape: `quad` or `poly`

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

### <a id="gw-e2e-003"></a>GW-E2E-003 — Line-confidence calibration

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 10
- **Superseded by:** GW-E2E-005, for the temperature v1 ships with (this run fitted it for GW-REC-024). H8's validation verdict stays with this run.
- **Git commit:** recorded in the result file   **Config:** `configs/det/gw-det-001.yaml` with Phase 9's overrides; `runs/GW-REC-024/config.resolved.yaml`   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages and lines as in GW-DET-001 and GW-REC-024
- **Hardware:** RTX 4050 Laptop 6 GB for the validation runs; i7-14650HX CPU for GW-EFF-002
- **Research question:** is the recognizer's raw line confidence calibrated, and does one temperature fix it (RQ8)?
- **Hypothesis (written before the run):** [H8](hypotheses.md): raw ECE above 0.05 on validation; temperature scaling at least halves it.
- **Method:** `docs/PHASE_STATUS.md` Phase 10, plan items 2 and 3; `scripts/rec_calibrate.py`.
- **Variables:** independent: temperature; controlled: model, lines, confidence definition
- **Control:** T = 1
- **Result:** 73,888 validation lines (half A 35,976, half B 37,912; accuracy on half B 0.721). Fitted T = 2.35 (binary NLL on half A 0.412 against 0.784 at T = 1). ECE on half B: 0.2172 at T = 1, 0.0888 at T = 2.35, difference **−0.128 [−0.133, −0.124]** (`reports/calibration/GW-E2E-003/calibration.md`, reliability diagram there): adopted. On the pipeline's own 62,591 validation lines (GW-E2E-001), ECE 0.344 at T = 1 and 0.134 at T = 2.35, as information.
- **Interpretation:** **H8 supported on validation**: the raw confidence is strongly over-confident (ECE 0.22, above 0.05) and one temperature cuts the error by 59%, more than half. The remaining ECE (0.09) and the larger one on pipeline lines (0.13, where detection errors also make lines wrong) say a single temperature is not the whole story. The v0.1 bundle carries T = 2.35. Transfer to GDC is H8's Phase 14 test.
- **Limitations:** fitted and checked on two halves of one validation set; exact match is a strict notion of right.
- **Next experiment:** GW-E2E-001 records confidence at the fitted T too.

### <a id="gw-e2e-001"></a>GW-E2E-001 — The pipeline on HierText validation

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 10
- **Superseded by:** GW-E2E-006, for v1's page numbers on the same 1,715 validation pages (this run used GW-REC-024). GW-E2E-006 does not repeat the comparison with the baselines on the 400 Phase 5 pages.
- **Git commit:** recorded in the result file   **Config:** `configs/det/gw-det-001.yaml` with Phase 9's overrides; `runs/GW-REC-024/config.resolved.yaml`   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages and lines as in GW-DET-001 and GW-REC-024
- **Hardware:** RTX 4050 Laptop 6 GB for the validation runs; i7-14650HX CPU for GW-EFF-002
- **Research question:** how well does the whole GLYPHARIS pipeline read HierText validation pages, and how does it compare with the Phase 5 baselines on the same pages (RQ1)?
- **Hypothesis (written before the run):** [H1](hypotheses.md) is judged in Phase 14; here its ratio is information. Expectation, not a claim: lower bag-of-lines page CER than every baseline on the 400 Phase 5 pages, since both stages beat the baselines on their own.
- **Method:** `docs/PHASE_STATUS.md` Phase 10, plan items 1 and 4; `scripts/e2e_eval.py --mode pipeline --pages all` and `--pages baseline --baselines`.
- **Variables:** independent: system; controlled: pages, scoring code, normalisation track
- **Control:** the Phase 5 baselines (GW-BASE-001, 002, 005, 006) on the same 400 pages
- **Result:** all 1,715 validation pages, strict track (`reports/e2e/GW-E2E-001/report.md`): bag-of-lines page CER **27.56% [25.14, 30.36]**, E2E line F1 57.96% [56.37, 59.64], detection P 85.19%, R 71.57%, F1 77.79%; normalised track page CER 27.37%. On the 400 Phase 5 pages (strict): page CER 24.33% [20.90, 28.17] and E2E F1 60.18% [56.52, 63.67], against PP-OCRv6 small 42.44% and 45.68% (paired differences **−18.11 [−21.43, −15.08]** points CER, +14.50 [+11.93, +17.05] F1), PP-OCRv5 mobile 50.84% and 38.75% (−26.51 [−30.46, −22.76], +21.42), PP-OCRv6 tiny 49.65% and 38.69%, Tesseract 5 94.50% and 12.65%. H1's ratio (information): 0.48 of PP-OCRv5 mobile's page CER.
- **Interpretation:** the expectation holds and by a wide margin: on HierText validation GLYPHARIS reads pages with about 18 points less page CER than the best baseline and wins every paired comparison on both tracks. The page CER of 24 to 28% is mostly detection (GW-E2E-002). For H1 the CER half of the criterion is already far inside its 2× bound on these pages; its latency half is GW-EFF-002 and Phase 17's ONNX measurement, and its test is Phase 14's GDC clean slice.
- **Limitations:** HierText pages are mostly scene photographs, not the documents GLYPHARIS targets; no reading order can be scored on them.
- **Next experiment:** GW-E2E-002 splits its error.

### <a id="gw-e2e-002"></a>GW-E2E-002 — Oracle analysis

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 10
- **Git commit:** recorded in the result file   **Config:** `configs/det/gw-det-001.yaml` with Phase 9's overrides; `runs/GW-REC-024/config.resolved.yaml`   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages and lines as in GW-DET-001 and GW-REC-024
- **Hardware:** RTX 4050 Laptop 6 GB for the validation runs; i7-14650HX CPU for GW-EFF-002
- **Research question:** how much of the pipeline's page error comes from detection and how much from recognition (RQ5)?
- **Hypothesis (written before the run):** [H4](hypotheses.md), exploratory here: recognition's share of bag-of-lines CER above one half on the sharpest quartile and below one half on the blurriest.
- **Method:** `docs/PHASE_STATUS.md` Phase 10, plan item 5; `scripts/e2e_eval.py --mode oracle --pages all`.
- **Variables:** independent: where the line outlines come from (detector or ground truth); controlled: recognizer, crops, pages, scoring
- **Control:** GW-E2E-001
- **Result:** all 1,715 pages, strict track (`reports/e2e/GW-E2E-002/report.md`): oracle page CER **6.67% [6.11, 7.34]** against the pipeline's 27.56%; recognition's share of page error 0.24. Per blur quartile (Q1 blurriest to Q4 sharpest): pipeline 36.5, 29.5, 29.7, 21.2%; oracle 8.8, 8.5, 7.4, 4.3%; recognition's share 0.24, 0.29, 0.25, 0.20. Oracle E2E line F1 76.9%.
- **Interpretation:** detection causes about three quarters of the page error on HierText, on sharp pages as much as on blurry ones. **H4's Phase 10 reading is not supported**: recognition's share is below one half on every quartile, the sharpest included. HierText's pages are mostly photographs of scenes and signs, not the clean documents H4 is about, so this is a statement about this data; Phase 15 tests H4 on the frozen model with GDC's slices. The oracle's 6.7% matches the recognizer's own validation CER (6.46%), so cutting lines from page polygons reproduces the training crops.
- **Limitations:** ground-truth outlines are cut exactly like training crops, so the oracle also removes any crop-shape mismatch between detector quads and training crops.
- **Next experiment:** Phase 15 repeats it on the frozen model.

### <a id="gw-eff-002"></a>GW-EFF-002 — Whole-pipeline CPU page time

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 10
- **Superseded by:** GW-EFF-007, for v1's CPU page time and memory in PyTorch. That table uses 10 threads and this run 16, so the two are not a like-for-like pair.
- **Git commit:** recorded in the result file   **Config:** `configs/det/gw-det-001.yaml` with Phase 9's overrides; `runs/GW-REC-024/config.resolved.yaml`   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages and lines as in GW-DET-001 and GW-REC-024
- **Hardware:** RTX 4050 Laptop 6 GB for the validation runs; i7-14650HX CPU for GW-EFF-002
- **Research question:** does the PyTorch pipeline meet ADR-0011's page target on the CPU?
- **Hypothesis (written before the run):** none. Expectation from GW-EFF-001's estimate: inside the mean target, close to the p95 one.
- **Method:** `docs/PHASE_STATUS.md` Phase 10, plan item 6; `scripts/e2e_profile.py`.
- **Variables:** none; conditions recorded as for the baselines
- **Control:** ADR-0011's page target and the baselines' page profiles
- **Result:** PyTorch fp32, 16 threads, long side 1536, idle machine (CPU busy 6.3% before, 6.6% after) (`reports/efficiency/GW-EFF-002.json`): page mean **1,998 ms**, p95 3,955 ms; detection 269 ms and recognition 1,721 ms per page on average (40.9 lines); peak RAM 1,624 MiB. ADR-0011's page target (mean 1.5 s, p95 2.7 s, RAM below 1,836 MiB): **missed** on time, met on memory. Baselines on the same pages: PP-OCRv6 tiny 1,727 ms, PP-OCRv5 mobile 3,408 ms, PP-OCRv6 small 3,714 ms (`reports/baselines/efficiency.md`).
- **Interpretation:** the PyTorch build is faster than PP-OCRv5 mobile and PP-OCRv6 small but misses ADR-0011's page target. The miss is recognition: about 42 ms per line inside the pipeline against 24.4 ms for one line at a time (GW-REC-024), so Phase 9's estimate (1.32 s) was too low. Cutting the lines costs under 1 ms each; the model is the cost, and the padding that keeps every line's reading independent of its batch (64 columns, about 40% of a typical line) is a likely part of it. ADR-0011 is judged on the ONNX Runtime build (GW-EFF-004, Phase 17); speed work belongs to Phase 16.
- **Limitations:** PyTorch stands in for the Phase 17 ONNX Runtime build.
- **Next experiment:** Phase 17 measures the exported pipeline.

### <a id="gw-eff-003"></a>GW-EFF-003 — ONNX equivalence of the v0.1 bundle

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 17
- **Superseded by:** GW-EFF-006, for the ONNX equivalence of the v1 bundle under the same four tolerances (this run checked the v0.1 bundle).
- **Git commit:** recorded in the result file   **Config:** the v0.1 bundle's manifest   **Seed(s):** 20261028 (line sample)
- **Dataset version(s):** HierText validation pages and lines as in GW-E2E-001
- **Hardware:** i7-14650HX CPU, ONNX Runtime 1.31.1 and PyTorch 2.14.1
- **Research question:** does the ONNX build of the v0.1 models read validation data like the PyTorch models?
- **Hypothesis (written before the run):** none; the tolerances are pass criteria.
- **Method:** `docs/PHASE_STATUS.md` Phase 17, plan item 2.
- **Variables:** independent: backend; controlled: weights, settings, data
- **Control:** the PyTorch backend of the same bundle
- **Result:** bundle `glypharis-0.1.0` (GW-DET-001 at long side 1536 with GW-DET-004's post-processing; GW-REC-024 at T = 2.35), `reports/export/GW-EFF-003/equivalence.md`: detector probability maps on 110 pages, max abs difference 7.3e-6 (tolerance 1e-4); recognizer logits on 5,000 lines, 3.7e-4 (1e-3); greedy text identical on all 5,000 lines, no near ties; page CER on the 400 Phase 5 pages 24.33% on both backends, difference 0.000 points (0.05). **Export successful.** A first bundle built with the wrong post-processing also passed these checks and was caught by GW-E2E-004 (`research/failures.md`).
- **Interpretation:** the ONNX build reads exactly like the PyTorch models: the two backends are interchangeable for accuracy, and every Phase 10 number holds for either.
- **Limitations:** CPU only; one ONNX Runtime version.
- **Next experiment:** GW-EFF-004.

### <a id="gw-eff-004"></a>GW-EFF-004 — Page time on ONNX Runtime

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 17
- **Superseded by:** GW-EFF-007, for v1's page time and memory on ONNX Runtime (this run measured the v0.1 bundle).
- **Git commit:** recorded in the result file   **Config:** the v0.1 bundle's manifest   **Seed(s):** 20261028 (line sample)
- **Dataset version(s):** HierText validation pages and lines as in GW-E2E-001
- **Hardware:** i7-14650HX CPU, ONNX Runtime 1.31.1 and PyTorch 2.14.1
- **Research question:** does the ONNX build meet ADR-0011's page target on the CPU?
- **Hypothesis (written before the run):** none. Expectation: faster than PyTorch (GW-EFF-002).
- **Method:** `docs/PHASE_STATUS.md` Phase 17, plan item 3.
- **Variables:** none; conditions recorded as for the baselines
- **Control:** ADR-0011's page target; GW-EFF-002
- **Result:** ONNX Runtime 1.31.1, default threads, the bundle's ONNX backend (`reports/efficiency/GW-EFF-004.json`): page mean **5,515 ms**, p95 12,691 ms (detection 442 ms, recognition 5,063 ms for 40.9 lines), peak RAM 11,221 MiB: **misses ADR-0011's page target on time and memory**, by far. PyTorch on the same pages (GW-EFF-002): 1,998 ms, p95 3,955 ms, 1,624 MiB. A direct comparison on 40 recognizer batches of varied shape: PyTorch 37 s, ONNX Runtime 170 s with its memory arena (+8.4 GB) and 219 s without it (+8 MB); detector on 12 pages: PyTorch 243 ms per page, ONNX Runtime 113 ms with the arena (+398 MB) and 265 ms without it.
- **Interpretation:** the exported recognizer is the problem: its SVTR global attention becomes plain matrix products over every pair of tokens (12 rows by up to 384 columns per line), where PyTorch runs a fused attention kernel, and ONNX Runtime's arena keeps a buffer for every new input shape. The exported detector is fine (twice PyTorch's speed). v0.1 therefore ships with PyTorch as the default backend and ONNX Runtime without its arena; ADR-0011's revisit condition is met and the trade-off goes to the owner (ADR-0011, Phase 17 measurement). Making the exported recognizer fast is Phase 16 work.
- **Limitations:** one machine; default threads.
- **Next experiment:** Phase 18 packages the bundle.

### <a id="gw-e2e-004"></a>GW-E2E-004 — The package against the Phase 10 pipeline

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 18
- **Superseded by:** GW-E2E-011, for the package check on the v1 bundle. That run uses the PyTorch backend, the path v1 ships on; this one used the v0.1 bundle's ONNX backend.
- **Git commit:** recorded in the result file   **Config:** the v0.1 bundle's manifest   **Seed(s):** none
- **Dataset version(s):** the 400 Phase 5 HierText validation pages
- **Hardware:** i7-14650HX CPU (package); GW-E2E-001 ran on the RTX 4050
- **Research question:** does the shipped path (bundle, `from_pretrained`, ONNX Runtime) read validation pages like the pipeline Phase 10 measured?
- **Hypothesis (written before the run):** none; pass criterion: page CER within 0.05 points.
- **Method:** `docs/PHASE_STATUS.md` Phase 18, plan item 1; `scripts/package_check.py`.
- **Variables:** independent: how the pipeline is built and run; controlled: weights, settings, pages
- **Control:** GW-E2E-001 on the same pages
- **Result:** `GlypharisOCR.from_pretrained` on the corrected v0.1 bundle, ONNX backend, 400 Phase 5 pages (`reports/package/GW-E2E-004.json`): page CER 24.33% against GW-E2E-001's 24.33%, difference +0.001 points (tolerance 0.05); 16,827 of 16,835 lines with the same text and a box within 1 px: **pass**. The first bundle (old post-processing) failed this check: 26.28% against 24.33%, 2,891 lines identical (`reports/package/attempt1_record_post/`, `research/failures.md`).
- **Interpretation:** the shipped path reproduces the pipeline Phase 10 measured, on a different device (CPU instead of GPU) and runtime; the 8 lines that differ are numeric noise. The check is also the one that caught a mis-built bundle the equivalence checks could not see.
- **Limitations:** GW-E2E-001 ran on the GPU in PyTorch; small numeric differences can move a box or a character.
- **Next experiment:** Phase 20.

### <a id="gw-data-001"></a>GW-DATA-001 — Phase 11 arm A (no augmentation), seed 2024

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 11
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-001.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for the robustness score
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ3: which augmentation families make the recognizer robust to degraded input?
- **Hypothesis (written before the run):** [H3](hypotheses.md#h3--document-degradation-augmentation-improves-robustness-rq3) (tested on arm D against A, three seeds each)
- **Method:** `python train.py --config configs/rec/gw-data-001.yaml`; one of three seeds of arm A (with GW-REC-015 and the other GW-DATA seed); scored on GLYPH-C by GW-DATA-009. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 11.
- **Variables:** independent: the seed (the arm's augmentation is fixed); controlled: everything else
- **Control:** arm A (GW-REC-015, GW-DATA-001, 002)
- **Result:** val CER 8.62% [8.46, 8.78] (95% bootstrap CI over the 73,888 validation lines); GLYPH-C mCE 10.14%, clean GLYPH-C CER 4.03% (`reports/robustness/GW-DATA-001.json`). 8 epochs, 45,368, optimizer steps; the final epoch was the best. Arm A as a whole: see GW-DATA-009.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-001/report.md`)
- **Interpretation:** one seed of arm A; it lies within the arm's seed range reported in GW-DATA-009, and the arm, not this run, carries the verdict.
- **Limitations:** trained by the shared training loop; GW-REC-015 (seed 1337) by Phase 7's trainer.
- **Next experiment:** GW-DATA-009
- **Training time:** 0.80 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-002"></a>GW-DATA-002 — Phase 11 arm A (no augmentation), seed 7

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 11
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-002.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for the robustness score
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ3: which augmentation families make the recognizer robust to degraded input?
- **Hypothesis (written before the run):** [H3](hypotheses.md#h3--document-degradation-augmentation-improves-robustness-rq3) (tested on arm D against A, three seeds each)
- **Method:** `python train.py --config configs/rec/gw-data-002.yaml`; one of three seeds of arm A (with GW-REC-015 and the other GW-DATA seed); scored on GLYPH-C by GW-DATA-009. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 11.
- **Variables:** independent: the seed (the arm's augmentation is fixed); controlled: everything else
- **Control:** arm A (GW-REC-015, GW-DATA-001, 002)
- **Result:** val CER 8.69% [8.53, 8.86] (95% bootstrap CI over the 73,888 validation lines); GLYPH-C mCE 10.28%, clean GLYPH-C CER 4.18% (`reports/robustness/GW-DATA-002.json`). 8 epochs, 45,367, optimizer steps; the final epoch was the best. Arm A as a whole: see GW-DATA-009.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-002/report.md`)
- **Interpretation:** one seed of arm A; it lies within the arm's seed range reported in GW-DATA-009, and the arm, not this run, carries the verdict.
- **Limitations:** trained by the shared training loop; GW-REC-015 (seed 1337) by Phase 7's trainer.
- **Next experiment:** GW-DATA-009
- **Training time:** 0.81 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-003"></a>GW-DATA-003 — Phase 11 arm B (geometric augmentation), seed 2024

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 11
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-003.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for the robustness score
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ3: which augmentation families make the recognizer robust to degraded input?
- **Hypothesis (written before the run):** [H3](hypotheses.md#h3--document-degradation-augmentation-improves-robustness-rq3) (tested on arm D against A, three seeds each)
- **Method:** `python train.py --config configs/rec/gw-data-003.yaml`; one of three seeds of arm B (with GW-REC-016 and the other GW-DATA seed); scored on GLYPH-C by GW-DATA-009. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 11.
- **Variables:** independent: the seed (the arm's augmentation is fixed); controlled: everything else
- **Control:** arm A (GW-REC-015, GW-DATA-001, 002)
- **Result:** val CER 8.29% [8.13, 8.45] (95% bootstrap CI over the 73,888 validation lines); GLYPH-C mCE 8.51%, clean GLYPH-C CER 3.48% (`reports/robustness/GW-DATA-003.json`). 8 epochs, 45,368, optimizer steps; the final epoch was the best. Arm B as a whole: see GW-DATA-009.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-003/report.md`)
- **Interpretation:** one seed of arm B; it lies within the arm's seed range reported in GW-DATA-009, and the arm, not this run, carries the verdict.
- **Limitations:** trained by the shared training loop; GW-REC-016 (seed 1337) by Phase 7's trainer.
- **Next experiment:** GW-DATA-009
- **Training time:** 1.04 GPU-hours (RTX 4050 Laptop, bf16); stopped once by a power cut and resumed from its last checkpoint (step 40,000)

### <a id="gw-data-004"></a>GW-DATA-004 — Phase 11 arm B (geometric augmentation), seed 7

- **Date allocated / run:** 2026-10-28 / 2026-10-28
- **Phase:** 11
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-004.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for the robustness score
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ3: which augmentation families make the recognizer robust to degraded input?
- **Hypothesis (written before the run):** [H3](hypotheses.md#h3--document-degradation-augmentation-improves-robustness-rq3) (tested on arm D against A, three seeds each)
- **Method:** `python train.py --config configs/rec/gw-data-004.yaml`; one of three seeds of arm B (with GW-REC-016 and the other GW-DATA seed); scored on GLYPH-C by GW-DATA-009. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 11.
- **Variables:** independent: the seed (the arm's augmentation is fixed); controlled: everything else
- **Control:** arm A (GW-REC-015, GW-DATA-001, 002)
- **Result:** val CER 8.38% [8.23, 8.55] (95% bootstrap CI over the 73,888 validation lines); GLYPH-C mCE 8.76%, clean GLYPH-C CER 3.53% (`reports/robustness/GW-DATA-004.json`). 8 epochs, 45,367, optimizer steps; the final epoch was the best. Arm B as a whole: see GW-DATA-009.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-004/report.md`)
- **Interpretation:** one seed of arm B; it lies within the arm's seed range reported in GW-DATA-009, and the arm, not this run, carries the verdict.
- **Limitations:** trained by the shared training loop; GW-REC-016 (seed 1337) by Phase 7's trainer.
- **Next experiment:** GW-DATA-009
- **Training time:** 0.94 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-005"></a>GW-DATA-005 — Phase 11 arm C (photometric augmentation), seed 2024

- **Date allocated / run:** 2026-10-28 / 2026-10-29
- **Phase:** 11
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-005.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for the robustness score
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ3: which augmentation families make the recognizer robust to degraded input?
- **Hypothesis (written before the run):** [H3](hypotheses.md#h3--document-degradation-augmentation-improves-robustness-rq3) (tested on arm D against A, three seeds each)
- **Method:** `python train.py --config configs/rec/gw-data-005.yaml`; one of three seeds of arm C (with GW-REC-017 and the other GW-DATA seed); scored on GLYPH-C by GW-DATA-009. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 11.
- **Variables:** independent: the seed (the arm's augmentation is fixed); controlled: everything else
- **Control:** arm A (GW-REC-015, GW-DATA-001, 002)
- **Result:** val CER 8.55% [8.39, 8.71] (95% bootstrap CI over the 73,888 validation lines); GLYPH-C mCE 10.19%, clean GLYPH-C CER 4.08% (`reports/robustness/GW-DATA-005.json`). 8 epochs, 45,368, optimizer steps; the final epoch was the best. Arm C as a whole: see GW-DATA-009.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-005/report.md`)
- **Interpretation:** one seed of arm C; it lies within the arm's seed range reported in GW-DATA-009, and the arm, not this run, carries the verdict.
- **Limitations:** trained by the shared training loop; GW-REC-017 (seed 1337) by Phase 7's trainer.
- **Next experiment:** GW-DATA-009
- **Training time:** 0.83 GPU-hours (RTX 4050 Laptop, bf16); stopped once by a power cut and resumed from its last checkpoint (step 11,342)

### <a id="gw-data-006"></a>GW-DATA-006 — Phase 11 arm C (photometric augmentation), seed 7

- **Date allocated / run:** 2026-10-28 / 2026-10-29
- **Phase:** 11
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-006.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for the robustness score
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ3: which augmentation families make the recognizer robust to degraded input?
- **Hypothesis (written before the run):** [H3](hypotheses.md#h3--document-degradation-augmentation-improves-robustness-rq3) (tested on arm D against A, three seeds each)
- **Method:** `python train.py --config configs/rec/gw-data-006.yaml`; one of three seeds of arm C (with GW-REC-017 and the other GW-DATA seed); scored on GLYPH-C by GW-DATA-009. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 11.
- **Variables:** independent: the seed (the arm's augmentation is fixed); controlled: everything else
- **Control:** arm A (GW-REC-015, GW-DATA-001, 002)
- **Result:** val CER 8.56% [8.40, 8.73] (95% bootstrap CI over the 73,888 validation lines); GLYPH-C mCE 10.32%, clean GLYPH-C CER 4.20% (`reports/robustness/GW-DATA-006.json`). 8 epochs, 45,367, optimizer steps; the final epoch was the best. Arm C as a whole: see GW-DATA-009.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-006/report.md`)
- **Interpretation:** one seed of arm C; it lies within the arm's seed range reported in GW-DATA-009, and the arm, not this run, carries the verdict.
- **Limitations:** trained by the shared training loop; GW-REC-017 (seed 1337) by Phase 7's trainer.
- **Next experiment:** GW-DATA-009
- **Training time:** 0.81 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-007"></a>GW-DATA-007 — Phase 11 arm D (degradation augmentation), seed 2024

- **Date allocated / run:** 2026-10-28 / 2026-10-29
- **Phase:** 11
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-007.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for the robustness score
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ3: which augmentation families make the recognizer robust to degraded input?
- **Hypothesis (written before the run):** [H3](hypotheses.md#h3--document-degradation-augmentation-improves-robustness-rq3) (tested on arm D against A, three seeds each)
- **Method:** `python train.py --config configs/rec/gw-data-007.yaml`; one of three seeds of arm D (with GW-REC-018 and the other GW-DATA seed); scored on GLYPH-C by GW-DATA-009. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 11.
- **Variables:** independent: the seed (the arm's augmentation is fixed); controlled: everything else
- **Control:** arm A (GW-REC-015, GW-DATA-001, 002)
- **Result:** val CER 8.66% [8.50, 8.82] (95% bootstrap CI over the 73,888 validation lines); GLYPH-C mCE 10.03%, clean GLYPH-C CER 3.93% (`reports/robustness/GW-DATA-007.json`). 8 epochs, 45,368, optimizer steps; the final epoch was the best. Arm D as a whole: see GW-DATA-009.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-007/report.md`)
- **Interpretation:** one seed of arm D; it lies within the arm's seed range reported in GW-DATA-009, and the arm, not this run, carries the verdict.
- **Limitations:** trained by the shared training loop; GW-REC-018 (seed 1337) by Phase 7's trainer.
- **Next experiment:** GW-DATA-009
- **Training time:** 0.82 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-008"></a>GW-DATA-008 — Phase 11 arm D (degradation augmentation), seed 7

- **Date allocated / run:** 2026-10-28 / 2026-10-29
- **Phase:** 11
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-008.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for the robustness score
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ3: which augmentation families make the recognizer robust to degraded input?
- **Hypothesis (written before the run):** [H3](hypotheses.md#h3--document-degradation-augmentation-improves-robustness-rq3) (tested on arm D against A, three seeds each)
- **Method:** `python train.py --config configs/rec/gw-data-008.yaml`; one of three seeds of arm D (with GW-REC-018 and the other GW-DATA seed); scored on GLYPH-C by GW-DATA-009. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 11.
- **Variables:** independent: the seed (the arm's augmentation is fixed); controlled: everything else
- **Control:** arm A (GW-REC-015, GW-DATA-001, 002)
- **Result:** val CER 8.70% [8.54, 8.87] (95% bootstrap CI over the 73,888 validation lines); GLYPH-C mCE 10.04%, clean GLYPH-C CER 3.98% (`reports/robustness/GW-DATA-008.json`). 8 epochs, 45,367, optimizer steps; the final epoch was the best. Arm D as a whole: see GW-DATA-009.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-008/report.md`)
- **Interpretation:** one seed of arm D; it lies within the arm's seed range reported in GW-DATA-009, and the arm, not this run, carries the verdict.
- **Limitations:** trained by the shared training loop; GW-REC-018 (seed 1337) by Phase 7's trainer.
- **Next experiment:** GW-DATA-009
- **Training time:** 0.80 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-009"></a>GW-DATA-009 — Phase 11: augmentation families on GLYPH-C

- **Date allocated / run:** 2026-10-28 / 2026-10-28 to 2026-10-29
- **Phase:** 11
- **Git commit:** recorded in the result files   **Config:** the 15 runs' resolved configs   **Seed(s):** 1337, 2024, 7 per arm; bootstrap seed 20261028
- **Dataset version(s):** GLYPH-C v1 (`data/processed/glyph_c/lines`, sha256 in each result file); HierText validation pack for clean CER
- **Hardware:** RTX 4050 Laptop 6 GB
- **Research question:** RQ3: which augmentation families (geometric, photometric, document degradation, all) lower the recognizer's error on corrupted lines, and at what cost on clean lines?
- **Hypothesis (written before the run):** [H3](hypotheses.md#h3--document-degradation-augmentation-improves-robustness-rq3): arm D lowers mCE against A without raising clean validation CER by more than 5% relative.
- **Method:** `scripts/glyph_c_eval.py score` on each run's best checkpoint, then `scripts/glyph_c_eval.py compare`; plan, metric and decision rule registered in `docs/PHASE_STATUS.md`, Phase 11.
- **Variables:** independent: `data.augmentation` (none, geometric, photometric, degradation, full); controlled: recipe, data, steps; seeds 1337, 2024, 7 in every arm
- **Control:** arm A (no augmentation)
- **Result:** (`reports/robustness/phase11_comparison.md`)

  | Arm | mCE % (pooled) | mCE % per seed (1337, 2024, 7) | Δ vs A [95% CI] | Clean val CER % | Verdict |
  |---|---|---|---|---|---|
  | A (none) | 10.28 | 10.43, 10.14, 10.28 | — | 8.66 | control |
  | B (geometric) | 8.61 | 8.56, 8.51, 8.76 | −1.67 [−1.73, −1.62] | 8.34 | helps |
  | C (photometric) | 10.23 | 10.17, 10.19, 10.32 | −0.06 [−0.08, −0.02] | 8.57 | no clear effect |
  | D (degradation) | 10.04 | 10.04, 10.03, 10.04 | −0.25 [−0.30, −0.19] | 8.68 | helps |
  | E (full) | 8.46 | 8.39, 8.45, 8.53 | −1.83 [−1.90, −1.75] | 8.31 | helps |

  H3: arm D helps (all three D seeds below all three A seeds, CI below zero) and its clean validation CER is +0.2% relative to A's, inside the 5% guard: **supported**. Per corruption (mean over levels, pooled), rotation dominates: A 38.89%, B 32.93%, D 39.00%, E 33.31%; the other six corruptions stay between 3.69% and 7.90%.
- **Interpretation:** geometric augmentation does most of the robustness work (−1.67 [−1.73, −1.62] points), on rotation (38.89 → 32.93%) and perspective (7.90 → 5.55%) above all, and it also lowers clean CER. Degradation helps a little (−0.25 [−0.30, −0.19]), through blur (4.93 → 4.53%), low resolution (6.14 → 5.55%) and JPEG (4.68 → 4.48%), not rotation, at no clean cost. Photometric augmentation shows no clear effect, although its preset includes blur, noise and JPEG. The full preset is best on robustness and on clean lines. Rotation beyond the training range (at most ±2°) is the recognizer's largest weakness.
- **Limitations:** stated with the plan in `docs/PHASE_STATUS.md` (trainer versions across seeds; GLYPH-C's HierText half overlaps the selection set; CRNN short budget, not the v0.1 recognizer). Also: several GLYPH-C corruptions are close relatives of training transforms (rotation and perspective in B, blur, noise and JPEG in C, low resolution in D), so the suite measures robustness to familiar corruptions at higher strength, not to unseen ones; the unweighted mCE is dominated by rotation.
- **Next experiment:** a wider training rotation range is a Phase 13 candidate; whether the verdicts carry over to the v0.1 recognizer (SVTR, height 48) needs the college GPU.

### <a id="gw-data-010"></a>GW-DATA-010 — Phase 12 arm +25K, seed 1337

- **Date allocated / run:** 2026-10-29 / 2026-10-29
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-010.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-010.yaml`; GW-REC-003's recipe with 25,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.29% [8.14, 8.46] (95% bootstrap CI over lines); document slice CER 11.18%; GlyphSynth validation CER 0.96% (`reports/synthetic/GW-DATA-010.json`). 8 epochs, 45,368 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +25K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-010/report.md`)
- **Interpretation:** one run of arm +25K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.02 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-011"></a>GW-DATA-011 — Phase 12 arm +25K, seed 2024

- **Date allocated / run:** 2026-10-29 / 2026-10-29
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-011.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-011.yaml`; GW-REC-003's recipe with 25,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.29% [8.12, 8.46] (95% bootstrap CI over lines); document slice CER 11.05%; GlyphSynth validation CER 0.93% (`reports/synthetic/GW-DATA-011.json`). 8 epochs, 45,368 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +25K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-011/report.md`)
- **Interpretation:** one run of arm +25K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.02 GPU-hours (RTX 4050 Laptop, bf16); stopped at the owner's request 1.5 minutes in, before its first checkpoint, and restarted from the beginning

### <a id="gw-data-012"></a>GW-DATA-012 — Phase 12 arm +25K, seed 7

- **Date allocated / run:** 2026-10-29 / 2026-10-29
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-012.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-012.yaml`; GW-REC-003's recipe with 25,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.28% [8.12, 8.44] (95% bootstrap CI over lines); document slice CER 11.05%; GlyphSynth validation CER 0.93% (`reports/synthetic/GW-DATA-012.json`). 8 epochs, 45,368 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +25K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-012/report.md`)
- **Interpretation:** one run of arm +25K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.03 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-013"></a>GW-DATA-013 — Phase 12 arm +50K, seed 1337

- **Date allocated / run:** 2026-10-29 / 2026-10-29
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-013.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-013.yaml`; GW-REC-003's recipe with 50,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.30% [8.15, 8.46] (95% bootstrap CI over lines); document slice CER 11.18%; GlyphSynth validation CER 0.73% (`reports/synthetic/GW-DATA-013.json`). 8 epochs, 45,380 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +50K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-013/report.md`)
- **Interpretation:** one run of arm +50K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.08 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-014"></a>GW-DATA-014 — Phase 12 arm +100K, seed 1337

- **Date allocated / run:** 2026-10-29 / 2026-10-29
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-014.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-014.yaml`; GW-REC-003's recipe with 100,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.39% [8.24, 8.56] (95% bootstrap CI over lines); document slice CER 11.22%; GlyphSynth validation CER 0.59%, GLYPH-C mCE 6.45% (`reports/synthetic/GW-DATA-014.json`). 8 epochs, 46,141 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +100K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-014/report.md`)
- **Interpretation:** one run of arm +100K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.24 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-015"></a>GW-DATA-015 — Phase 12 arm +100K, seed 2024

- **Date allocated / run:** 2026-10-29 / 2026-10-29
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-015.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-015.yaml`; GW-REC-003's recipe with 100,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.38% [8.22, 8.54] (95% bootstrap CI over lines); document slice CER 11.22%; GlyphSynth validation CER 0.55%, GLYPH-C mCE 6.46% (`reports/synthetic/GW-DATA-015.json`). 8 epochs, 46,134 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +100K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-015/report.md`)
- **Interpretation:** one run of arm +100K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.18 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-016"></a>GW-DATA-016 — Phase 12 arm +100K, seed 7

- **Date allocated / run:** 2026-10-29 / 2026-10-30
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-016.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-016.yaml`; GW-REC-003's recipe with 100,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.36% [8.21, 8.53] (95% bootstrap CI over lines); document slice CER 11.21%; GlyphSynth validation CER 0.56%, GLYPH-C mCE 6.39% (`reports/synthetic/GW-DATA-016.json`). 8 epochs, 46,133 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +100K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-016/report.md`)
- **Interpretation:** one run of arm +100K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.16 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-017"></a>GW-DATA-017 — Phase 12 arm +250K, seed 1337

- **Date allocated / run:** 2026-10-29 / 2026-10-30
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-017.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-017.yaml`; GW-REC-003's recipe with 250,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.58% [8.42, 8.75] (95% bootstrap CI over lines); document slice CER 11.46%; GlyphSynth validation CER 0.36% (`reports/synthetic/GW-DATA-017.json`). 8 epochs, 46,586 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +250K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-017/report.md`)
- **Interpretation:** one run of arm +250K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.41 GPU-hours (RTX 4050 Laptop, bf16); stopped at step 8,000 by a blocked checkpoint rename (`research/failures.md`) and resumed from step 7,000

### <a id="gw-data-018"></a>GW-DATA-018 — Phase 12 arm +250K, seed 2024

- **Date allocated / run:** 2026-10-29 / 2026-10-30
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-018.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-018.yaml`; GW-REC-003's recipe with 250,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.57% [8.40, 8.74] (95% bootstrap CI over lines); document slice CER 11.49%; GlyphSynth validation CER 0.38% (`reports/synthetic/GW-DATA-018.json`). 8 epochs, 46,577 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +250K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-018/report.md`)
- **Interpretation:** one run of arm +250K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.37 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-019"></a>GW-DATA-019 — Phase 12 arm +250K, seed 7

- **Date allocated / run:** 2026-10-29 / 2026-10-30
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-019.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-019.yaml`; GW-REC-003's recipe with 250,000 GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005)
- **Result:** HierText validation CER 8.52% [8.36, 8.69] (95% bootstrap CI over lines); document slice CER 11.38%; GlyphSynth validation CER 0.40% (`reports/synthetic/GW-DATA-019.json`). 8 epochs, 46,561 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +250K as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-019/report.md`)
- **Interpretation:** one run of arm +250K; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.40 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-020"></a>GW-DATA-020 — Phase 12 arm +100K, 6 fonts, seed 1337

- **Date allocated / run:** 2026-10-29 / 2026-10-30
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-020.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-020.yaml`; GW-REC-003's recipe with 100,000 (6 font families) GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005); for the factor, +100K (GW-DATA-014, 015, 016)
- **Result:** HierText validation CER 8.40% [8.24, 8.56] (95% bootstrap CI over lines); document slice CER 11.13%; GlyphSynth validation CER 0.61% (`reports/synthetic/GW-DATA-020.json`). 8 epochs, 46,198 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +100K, 6 fonts as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-020/report.md`)
- **Interpretation:** one run of arm +100K, 6 fonts; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.20 GPU-hours (RTX 4050 Laptop, bf16); stopped at step 1,950 when the desktop app that launched the chain restarted, and resumed from step 1,000

### <a id="gw-data-021"></a>GW-DATA-021 — Phase 12 arm +100K, no capture, seed 1337

- **Date allocated / run:** 2026-10-29 / 2026-10-30
- **Phase:** 12
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-data-021.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-REC-001; GlyphSynth v1 (`data/manifests/processed/glyphsynth.json`, variants in `glyphsynth_variants.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** RQ2: does document-style synthetic data help, how does it scale, which generator factors matter?
- **Hypothesis (written before the run):** [H2′](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed), tested by GW-DATA-022
- **Method:** `python train.py --config configs/rec/gw-data-021.yaml`; GW-REC-003's recipe with 100,000 (no capture effects) GlyphSynth lines added and `data.epoch_lines` 348,935. Plan and decision rule: `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: the GlyphSynth lines in the training pool; controlled: recipe, step budget, validation set
- **Control:** HierText only (GW-REC-003, 004, 005); for the factor, +100K (GW-DATA-014, 015, 016)
- **Result:** HierText validation CER 8.38% [8.22, 8.54] (95% bootstrap CI over lines); document slice CER 11.09%; GlyphSynth validation CER 0.65%, GLYPH-C mCE 6.75% (`reports/synthetic/GW-DATA-021.json`). 8 epochs, 46,222 optimizer steps (the HierText-only control: 45,360 to 45,368); best epoch 7. Arm +100K, no capture as a whole: GW-DATA-022.
- **Training diagnostics:** No flags raised. (`reports/training/GW-DATA-021/report.md`)
- **Interpretation:** one run of arm +100K, no capture; the arm, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-DATA-022
- **Training time:** 1.29 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-data-022"></a>GW-DATA-022 — Phase 12: synthetic data against HierText only

- **Date allocated / run:** 2026-10-29 / 2026-10-30
- **Phase:** 12
- **Git commit:** recorded in the result files   **Config:** the runs' resolved configs   **Seed(s):** as the runs
- **Dataset version(s):** HierText validation pack; GlyphSynth v1 validation; GLYPH-C v1
- **Hardware:** RTX 4050 Laptop 6 GB
- **Research question:** RQ2 (reframed by ADR-0009)
- **Hypothesis (written before the run):** [H2′a and H2′b](hypotheses.md#h2--document-style-synthetic-data-helps-and-scales-diversity-matters-rq2-reframed)
- **Method:** `scripts/synth_study.py score` (GlyphSynth validation) and `scripts/glyph_c_eval.py score` (factor runs), then `scripts/synth_study.py compare`; readouts and rules registered in `docs/PHASE_STATUS.md`, Phase 12.
- **Variables:** independent: GlyphSynth lines added (0, 25K, 50K, 100K, 250K), font families (85 or 6), capture effects (on or off); controlled: recipe and step budget
- **Control:** HierText only (GW-REC-003, 004, 005); +100K for the factors
- **Result:** (`reports/synthetic/phase12_comparison.md`, `phase12_scaling.png`)

  | Arm | Document slice CER % | Δ vs HierText only [95% CI] | Verdict | HierText val CER % | GlyphSynth val CER % |
  |---|---|---|---|---|---|
  | HierText only | 11.06 | — | control | 8.31 | 1.97 |
  | +25K | 11.09 | +0.04 [−0.03, +0.11] | no clear effect | 8.29 | 0.94 |
  | +50K | 11.18 | +0.12 [+0.03, +0.22] | hurts | 8.30 | 0.73 |
  | +100K | 11.22 | +0.16 [+0.06, +0.26] | hurts | 8.38 | 0.57 |
  | +250K | 11.44 | +0.39 [+0.25, +0.52] | hurts | 8.56 | 0.38 |
  | +100K, 6 fonts | 11.13 | +0.07 [−0.04, +0.18] | no clear effect | 8.40 | 0.61 |
  | +100K, no capture | 11.09 | +0.04 [−0.07, +0.15] | no clear effect | 8.38 | 0.65 |

  H2′a: no size helps on the document slice; 50K, 100K and 250K hurt: **not supported**. H2′b: 6 font families against 85 on the document slice −0.09 [−0.21, +0.03] (no clear difference); without capture effects, GLYPH-C mCE +0.31 [+0.27, +0.35] points higher (capture helps, clean guard holds): one factor of two, **not supported**.
- **Interpretation:** at a fixed step budget, GlyphSynth lines do not help HierText's dense pages, and the more of them, the worse (CER rises roughly in proportion to the lines added). They help a great deal on GlyphSynth's own validation lines (held-out fonts), so the model learns them; the rendered lines are simply not close enough to photographed text to stand in for real lines. Each synthetic line in the pool displaces a HierText line from an epoch, and that trade does not pay. Font variety made no measurable difference at 100K; the capture effects (rotation, blur, noise, lighting) do make the model more robust to corrupted lines, at no clean cost.
- **Limitations:** stated with the plan (equal steps displace real lines; the document slice is a proxy; one seed per factor arm). Also: the arms did not take exactly equal steps (see GW-DATA-010 to 021: up to 2.7% more for 250K, which favours the synthetic arms, so the losses are if anything understated).
- **Next experiment:** GDC (Phase 14, phone photos of printed documents) is the real document test of whether GlyphSynth helps; for the final model, Phase 13 should not add GlyphSynth beyond 25K lines on the strength of this phase.

### <a id="gw-rec-025"></a>GW-REC-025 — Phase 13 L3 (+ input height 48), seed 2024

- **Date allocated / run:** 2026-10-30 / 2026-10-30
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-025.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-025.yaml`; GW-REC-006 with the seed changed; 8 epochs (short budget). Compared by GW-REC-037.
- **Variables:** independent: the seed; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 7.82% [7.67, 7.98] (95% bootstrap CI over 73,888 lines), WER 22.25%; document slice 10.73%; GLYPH-C clean CER 2.80%, mCE 6.78% (`reports/robustness/GW-REC-025.json`). 8 epochs, 45,343 optimizer steps; best epoch 7, the last. The row as a whole: GW-REC-037.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-025/report.md`)
- **Interpretation:** one seed of row L3 (7.83, 7.82, 7.83% over its three seeds); the seeds agree to within 0.05 points, so the row, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 2.02 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-026"></a>GW-REC-026 — Phase 13 L3 (+ input height 48), seed 7

- **Date allocated / run:** 2026-10-30 / 2026-10-31
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-026.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-026.yaml`; GW-REC-006 with the seed changed; 8 epochs (short budget). Compared by GW-REC-037.
- **Variables:** independent: the seed; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 7.83% [7.68, 7.99] (95% bootstrap CI over 73,888 lines), WER 22.20%; document slice 10.62%; GLYPH-C clean CER 2.81%, mCE 6.84% (`reports/robustness/GW-REC-026.json`). 8 epochs, 45,346 optimizer steps; best epoch 7, the last. The row as a whole: GW-REC-037.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-026/report.md`)
- **Interpretation:** one seed of row L3 (7.83, 7.82, 7.83% over its three seeds); the seeds agree to within 0.05 points, so the row, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 2.10 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-027"></a>GW-REC-027 — Phase 13 L4 (full model), seed 2024

- **Date allocated / run:** 2026-10-30 / 2026-10-31
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-027.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-027.yaml`; GW-REC-022 with the seed changed; 8 epochs (short budget). Compared by GW-REC-037.
- **Variables:** independent: the seed; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 7.59% [7.44, 7.74] (95% bootstrap CI over 73,888 lines), WER 22.13%; document slice 10.32%; GLYPH-C clean CER 2.51%, mCE 6.25% (`reports/robustness/GW-REC-027.json`). 8 epochs, 45,297 optimizer steps; best epoch 7, the last. The row as a whole: GW-REC-037.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-027/report.md`)
- **Interpretation:** one seed of row L4 (7.56, 7.59, 7.56%); the seeds agree to within 0.05 points, so the row, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 5.34 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-028"></a>GW-REC-028 — Phase 13 L4 (full model), seed 7

- **Date allocated / run:** 2026-10-30 / 2026-11-01
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-028.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-028.yaml`; GW-REC-022 with the seed changed; 8 epochs (short budget). Compared by GW-REC-037.
- **Variables:** independent: the seed; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 7.56% [7.41, 7.72] (95% bootstrap CI over 73,888 lines), WER 22.04%; document slice 10.39%; GLYPH-C clean CER 2.62%, mCE 6.33% (`reports/robustness/GW-REC-028.json`). 8 epochs, 45,298 optimizer steps; best epoch 7, the last. The row as a whole: GW-REC-037.
- **Training diagnostics:** No flags raised. Stopped by hand 2 times at the owner's request and resumed from `last.pt` each time (report, Notes). (`reports/training/GW-REC-028/report.md`)
- **Interpretation:** one seed of row L4 (7.56, 7.59, 7.56%); the seeds agree to within 0.05 points, so the row, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 5.37 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-029"></a>GW-REC-029 — Phase 13 L5 (+ wider rotation), seed 1337

- **Date allocated / run:** 2026-10-30 / 2026-11-02
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-029.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-029.yaml`; GW-REC-022 with rotation up to ±4° on half the lines; 8 epochs (short budget). Compared by GW-REC-037.
- **Variables:** independent: the seed and the change of its row; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 7.48% [7.34, 7.64] (95% bootstrap CI over 73,888 lines), WER 22.05%; document slice 10.31%; GLYPH-C clean CER 2.44%, mCE 4.84% (`reports/robustness/GW-REC-029.json`). 8 epochs, 45,295 optimizer steps; best epoch 7, the last. The row as a whole: GW-REC-037.
- **Training diagnostics:** No flags raised. (`reports/training/GW-REC-029/report.md`)
- **Interpretation:** one seed of row L5 (7.48, 7.43, 7.44%); the seeds agree to within 0.05 points, so the row, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 5.05 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-030"></a>GW-REC-030 — Phase 13 L5 (+ wider rotation), seed 2024

- **Date allocated / run:** 2026-10-30 / 2026-11-03
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-030.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-030.yaml`; GW-REC-022 with rotation up to ±4° on half the lines; 8 epochs (short budget). Compared by GW-REC-037.
- **Variables:** independent: the seed and the change of its row; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 7.43% [7.28, 7.58] (95% bootstrap CI over 73,888 lines), WER 21.99%; document slice 10.24%; GLYPH-C clean CER 2.40%, mCE 4.79% (`reports/robustness/GW-REC-030.json`). 8 epochs, 45,297 optimizer steps; best epoch 7, the last. The row as a whole: GW-REC-037.
- **Training diagnostics:** No flags raised. Stopped by hand once at the owner's request and resumed from `last.pt` each time (report, Notes). (`reports/training/GW-REC-030/report.md`)
- **Interpretation:** one seed of row L5 (7.48, 7.43, 7.44%); the seeds agree to within 0.05 points, so the row, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 5.87 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-031"></a>GW-REC-031 — Phase 13 L5 (+ wider rotation), seed 7

- **Date allocated / run:** 2026-10-30 / 2026-11-03
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-031.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-031.yaml`; GW-REC-022 with rotation up to ±4° on half the lines; 8 epochs (short budget). Compared by GW-REC-037.
- **Variables:** independent: the seed and the change of its row; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 7.44% [7.30, 7.60] (95% bootstrap CI over 73,888 lines), WER 22.03%; document slice 10.26%; GLYPH-C clean CER 2.39%, mCE 4.78% (`reports/robustness/GW-REC-031.json`). 8 epochs, 45,298 optimizer steps; best epoch 7, the last. The row as a whole: GW-REC-037.
- **Training diagnostics:** No flags raised. Stopped by hand 2 times at the owner's request and resumed from `last.pt` each time (report, Notes). (`reports/training/GW-REC-031/report.md`)
- **Interpretation:** one seed of row L5 (7.48, 7.43, 7.44%); the seeds agree to within 0.05 points, so the row, not this run, carries the verdict.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 5.34 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-032"></a>GW-REC-032 — Phase 13 final, option A, seed 2024

- **Date allocated / run:** 2026-10-30 / —
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-032.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-032.yaml`; GW-REC-024 with the seed changed (trained only for option A); 30 epochs (full budget). Compared by GW-REC-037.
- **Variables:** independent: the seed; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** not trained: the ladder kept row L5, so the final runs are option B (GW-REC-034 to 036), as the plan fixed in advance.
- **Amendment (2026-11-18, Phase 21):** this is one of the two extra GW-REC-024 seeds (seed 2024) promised in that entry for before the paper. It stays not run; GW-REC-024's entry records the promise as closed for v1.
- **Training diagnostics:** —
- **Interpretation:** —
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** —

### <a id="gw-rec-033"></a>GW-REC-033 — Phase 13 final, option A, seed 7

- **Date allocated / run:** 2026-10-30 / —
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-033.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-033.yaml`; GW-REC-024 with the seed changed (trained only for option A); 30 epochs (full budget). Compared by GW-REC-037.
- **Variables:** independent: the seed; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** not trained: the ladder kept row L5, so the final runs are option B (GW-REC-034 to 036), as the plan fixed in advance.
- **Amendment (2026-11-18, Phase 21):** this is one of the two extra GW-REC-024 seeds (seed 7) promised in that entry for before the paper. It stays not run; GW-REC-024's entry records the promise as closed for v1.
- **Training diagnostics:** —
- **Interpretation:** —
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** —

### <a id="gw-rec-034"></a>GW-REC-034 — Phase 13 final, option B, seed 1337

- **Date allocated / run:** 2026-10-30 / 2026-11-10
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-034.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-034.yaml`; GW-REC-024 with L5's rotation (trained only for option B); 30 epochs (full budget). Compared by GW-REC-037.
- **Variables:** independent: the seed and the change of its row; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 6.30% [6.17, 6.44] (95% bootstrap CI over 73,888 lines), WER 19.37%; document slice 9.13%; GLYPH-C clean CER 1.93%, mCE 3.54% (`reports/robustness/GW-REC-034.json`). 30 epochs, 169,860 optimizer steps; best epoch 29 (the last). The three seeds against the baseline: GW-REC-037. This is the shipped checkpoint, fixed with the plan (`configs/final/glypharis_v1.yaml`).
- **Training diagnostics:** No flags raised. Stopped by hand three times at the owner's request and resumed from `last.pt`; one crash in epoch 25 (CUDA out of memory in the backward pass, most likely because another program took GPU memory; `research/failures.md`), resumed from step 142,000. (`reports/training/GW-REC-034/report.md`)
- **Interpretation:** one seed of the final recognizer (option B); the three seeds read 6.28 to 6.30%, so the spread at the full budget is about 0.02 points, smaller than any difference the ladder acted on.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 20.11 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-035"></a>GW-REC-035 — Phase 13 final, option B, seed 2024

- **Date allocated / run:** 2026-10-30 / 2026-11-12
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-035.yaml`   **Seed(s):** 2024
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-035.yaml`; GW-REC-024 with L5's rotation (trained only for option B); 30 epochs (full budget). Compared by GW-REC-037.
- **Variables:** independent: the seed and the change of its row; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 6.28% [6.15, 6.42] (95% bootstrap CI over 73,888 lines), WER 19.26%; document slice 9.12%; GLYPH-C clean CER 1.91%, mCE 3.53% (`reports/robustness/GW-REC-035.json`). 30 epochs, 169,860 optimizer steps; best epoch 28; the final epoch reads the same to two decimals. The three seeds against the baseline: GW-REC-037.
- **Training diagnostics:** No flags raised. Stopped by hand twice at the owner's request and resumed from `last.pt`. (`reports/training/GW-REC-035/report.md`)
- **Interpretation:** one seed of the final recognizer (option B); the three seeds read 6.28 to 6.30%, so the spread at the full budget is about 0.02 points, smaller than any difference the ladder acted on.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 19.09 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-rec-036"></a>GW-REC-036 — Phase 13 final, option B, seed 7

- **Date allocated / run:** 2026-10-30 / 2026-11-14
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/rec/gw-rec-036.yaml`   **Seed(s):** 7
- **Dataset version(s):** as GW-REC-001; GLYPH-C v1 for robustness
- **Hardware:** RTX 4050 Laptop 6 GB, bf16
- **Research question:** the Phase 13 ablation table and final configuration
- **Hypothesis (written before the run):** none of its own; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13
- **Method:** `python train.py --config configs/rec/gw-rec-036.yaml`; GW-REC-024 with L5's rotation (trained only for option B); 30 epochs (full budget). Compared by GW-REC-037.
- **Variables:** independent: the seed and the change of its row; controlled: everything else
- **Control:** the row's parent (GW-REC-037)
- **Result:** val CER 6.30% [6.17, 6.44] (95% bootstrap CI over 73,888 lines), WER 19.24%; document slice 9.21%; GLYPH-C clean CER 1.90%, mCE 3.53% (`reports/robustness/GW-REC-036.json`). 30 epochs, 169,860 optimizer steps; best epoch 29 (the last). The three seeds against the baseline: GW-REC-037.
- **Training diagnostics:** No flags raised. Stopped by hand once at the owner's request and resumed from `last.pt`. (`reports/training/GW-REC-036/report.md`)
- **Interpretation:** one seed of the final recognizer (option B); the three seeds read 6.28 to 6.30%, so the spread at the full budget is about 0.02 points, smaller than any difference the ladder acted on.
- **Limitations:** stated with the plan.
- **Next experiment:** GW-REC-037
- **Training time:** 19.37 GPU-hours (RTX 4050 Laptop, bf16)

### <a id="gw-det-007"></a>GW-DET-007 — Phase 13 detector: smaller text and wider rotation

- **Date allocated / run:** 2026-10-30 / 2026-11-04
- **Phase:** 13
- **Git commit:** recorded in the run summary   **Config:** `configs/det/gw-det-007.yaml`   **Seed(s):** 1337
- **Dataset version(s):** as GW-DET-001
- **Hardware:** RTX 4050 Laptop 6 GB
- **Research question:** do labels for smaller text and wider training rotation fix the misses Phase 9 found (lines under 8 px, rotated lines)?
- **Hypothesis (written before the run):** expectation, not a claim: higher recall on small and rotated lines, and higher F1@0.5 overall.
- **Method:** `python train.py --config configs/det/gw-det-007.yaml`; GW-DET-001 and GW-DET-007 scored on all validation pages at Phase 9's adopted settings (`scripts/det_eval.py`), paired with `scripts/det_pair.py`. Adopted if the F1@0.5 difference has a CI above zero (`docs/PHASE_STATUS.md`, Phase 13).
- **Variables:** independent: `labels.min_text_size` 6 → 3, `data.rotate_max_deg` 10 → 30; controlled: everything else
- **Control:** GW-DET-001
- **Result:** all 1,715 validation pages (74,500 care lines) at Phase 9's settings (long side 1536, GW-DET-004's post-processing, operating point `inverse09`; `reports/error_cases/detector/GW-DET-007/`): F1@0.5 0.7668 [0.7518, 0.7810], precision 0.8468, recall 0.7006 (TP 52,196, FP 9,444, FN 22,304), against GW-DET-001 0.7779 [0.7621, 0.7919], 0.8519, 0.7157 (53,322, 9,269, 21,178). Paired difference in F1@0.5 **−0.0111 [−0.0146, −0.0075]**: **not adopted**; GW-DET-001 stays. 24 epochs, 98,544 optimizer steps, best epoch 19 on the 400-page subset; 3,187,209 parameters at inference, as GW-DET-001.
- **Training diagnostics:** one `POSSIBLE OVERFITTING` flag at epoch 9 on the 400-page subset (`reports/training/GW-DET-007/report.md`), which later epochs recovered from; no other flags.
- **Interpretation:** the expectation fails: with labels for text down to 3 px and rotation up to 30°, the detector finds fewer lines, not more (recall −1.5 points), and is no more precise. Small and rotated lines were not scored as slices here, so whether those groups gained at the cost of the rest is not known; Phase 15's error analysis can look. Most of the page error is detection (GW-E2E-002, Phase 10), and these two label and augmentation changes do not reduce it.
- **Limitations:** one seed each; the post-processing was tuned on GW-DET-001, which favours it slightly (both stated in advance).
- **Next experiment:** the frozen configuration
- **Training time:** 7.51 GPU-hours (RTX 4050 Laptop)

### <a id="gw-rec-037"></a>GW-REC-037 — Phase 13: ablation ladder and final recognizer

- **Date allocated / run:** 2026-10-30 / 2026-11-14
- **Phase:** 13
- **Git commit:** recorded in the result files   **Config:** the runs' resolved configs   **Seed(s):** 1337, 2024, 7 per row
- **Dataset version(s):** HierText validation pack; GLYPH-C v1
- **Hardware:** RTX 4050 Laptop 6 GB
- **Research question:** what does each step from the baseline to the full model contribute, and which configuration is final?
- **Hypothesis (written before the run):** none; the ladder rules in `docs/PHASE_STATUS.md`, Phase 13 (H6 is touched by L4, exploratory)
- **Method:** `scripts/ablation_ladder.py ladder`, `decide` and `final`; GLYPH-C via `scripts/glyph_c_eval.py score`.
- **Variables:** independent: the ladder's changes; controlled: data, budget, seeds
- **Control:** each row's parent; GW-REC-001 for the final model
- **Result:** `reports/ablation/phase13_ladder.md` and `phase13_final.md`. Ladder (8 epochs, three seeds per row, each row against its parent): L0 baseline 8.66%; L1 + augmentation 8.31% (−0.36 [−0.40, −0.31], helps; kept); L2 + 25K GlyphSynth 8.29% (−0.02 [−0.05, +0.01], no clear effect; not kept, though GLYPH-C mCE −1.27 [−1.32, −1.21]); L3 + height 48 on L1 7.83% (−0.48 [−0.54, −0.42], helps; kept); L4 full model 7.57% (−0.26 [−0.32, −0.19], helps; kept); L5 + wider rotation 7.45% (−0.12 [−0.16, −0.08] on validation, helps, and GLYPH-C mCE 4.80% against 6.29%, −1.49 [−1.53, −1.45]; kept). Every kept row also helps on the document slice and on GLYPH-C mCE. Final option: **B**. Final recognizer (30 epochs, GW-REC-034 to 036): val CER 6.30, 6.28, 6.30% (mean 6.29%) against GW-REC-001's 7.53%, **−1.24 [−1.40, −1.11]** points; document slice 9.15% against 10.23%, −1.08 [−1.41, −0.78]; GLYPH-C mCE 3.54, 3.53, 3.53%.
- **Interpretation:** from the plain CRNN to the final model, validation CER falls from 8.66 to 7.45% at the short budget, and every step that was kept helps on all three measures. The largest single steps on validation are the input height (−0.48) and augmentation (−0.36); the wider rotation adds little on clean validation lines (−0.12) but cuts the corruption error by almost a quarter, as Phase 11's finding F15 pointed to. GlyphSynth lines leave validation unchanged but make the model more robust, as in Phase 12. At the full budget the final model gains 1.24 points over the CRNN baseline, and its three seeds agree to 0.02 points.
- **Limitations:** stated with the plan (short-budget ladder; L4 bundles the encoder with its training recipe; earlier seeds come from earlier phases). The baseline at the full budget is one seed, so the final comparison's CI covers page sampling, not the baseline's seed noise; the short-budget seed spread (about 0.05 points) is small next to −1.24.
- **Next experiment:** Phase 14

### <a id="gw-e2e-005"></a>GW-E2E-005 — Line-confidence calibration of the frozen recognizer

- **Date allocated / run:** 2026-11-14 / 2026-11-14
- **Phase:** 13
- **Git commit:** recorded in the result file   **Config:** `runs/GW-REC-034/config.resolved.yaml`   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages and lines as in GW-E2E-003
- **Hardware:** RTX 4050 Laptop 6 GB
- **Research question:** the temperature the frozen configuration ships with. GW-E2E-003's T = 2.35 was fitted for GW-REC-024, and a retrained recognizer needs its own.
- **Hypothesis (written before the run):** none of its own; GW-E2E-003's method and adoption rule unchanged (fit on validation half A, adopt if the ECE difference on half B has a CI below zero). H8 was settled by GW-E2E-003; this run is reported as a check of it on a second model.
- **Method:** `python scripts/rec_calibrate.py --rec-run runs/GW-REC-034 --experiment-id GW-E2E-005 --out reports/calibration/GW-E2E-005`
- **Variables:** independent: temperature; controlled: model, lines, confidence definition
- **Control:** T = 1
- **Result:** 73,888 validation lines (half A 35,976, half B 37,912; accuracy on half B 0.724). Fitted T = 2.35, the same grid value as GW-E2E-003 (binary NLL on half A 0.410 against 0.772 at T = 1). ECE on half B: 0.2140 at T = 1, 0.0886 at T = 2.35, difference **−0.125 [−0.130, −0.122]** (`reports/calibration/GW-E2E-005/calibration.md`, reliability diagram there): adopted.
- **Interpretation:** the final recognizer is over-confident in the same way and by about as much as GW-REC-024 (raw ECE 0.21 against 0.22), and the same temperature fixes most of it (59% lower), so H8's validation result holds on a second model. The frozen configuration carries T = 2.35 from this run.
- **Limitations:** as GW-E2E-003.
- **Next experiment:** the frozen configuration (`configs/final/glypharis_v1.yaml`)

### <a id="gw-e2e-006"></a>GW-E2E-006 — The frozen pipeline on HierText validation (Phase 14 reference)

- **Date allocated / run:** 2026-11-15 / 2026-11-15
- **Phase:** 14
- **Git commit:** the `phase14-eval` tag   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none (no training)
- **Dataset version(s):** HierText validation pages (Phase 4, `data/manifests/processed/hiertext.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, recognizer in strict fp32 (TF32 off), detector in strict fp32, cuDNN deterministic
- **Research question:** the frozen v1 pipeline's end-to-end numbers on validation, which do not exist yet (GW-E2E-001 used GW-REC-024), and its GPU time per page for the Phase 14 estimate.
- **Hypothesis (written before the run):** none; a reference measurement.
- **Method:** the frozen-config builder; all 1,715 validation pages; scored with the §3 page metrics, both tracks; confidences at T = 1 and 2.35.
- **Variables:** none (reference)
- **Control:** GW-E2E-001 (v0.1 pipeline, same pages)
- **Result:** all 1,715 validation pages, strict: bag-of-lines page CER 27.57% [25.11, 30.36], E2E F1 58.24% [56.64, 59.91], detection P/R/F1@0.5 85.19 / 71.57 / 77.79%; 810.2 s on the GPU, 0.47 s per page (`reports/external/end_to_end.md`, `run_record.md`).
- **Interpretation:** the reference the test pages are read against; detection equals GW-DET-001's Phase 13 value. HierText test came out close to it (25.16%, GW-E2E-007).
- **Limitations:** validation pages only; not a test result.
- **Next experiment:** GW-E2E-007

### <a id="gw-rec-038"></a>GW-REC-038 — The frozen recognizer on the HierText validation evaluation crops

- **Date allocated / run:** 2026-11-15 / 2026-11-15
- **Phase:** 14
- **Git commit:** the `phase14-eval` tag   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** `processed/hiertext/lines_validation_eval` (74,500 crops, `data/manifests/processed/eval_crops.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32 as GW-E2E-006
- **Research question:** H5′'s reference: the frozen recognizer's CER on validation crops under exactly the code, precision and crop selection the test crops get (the logged 6.30% was computed with bf16 autocast on the 73,888-line training pack, which drops out-of-charset and long lines).
- **Hypothesis (written before the run):** feeds [H5′](hypotheses.md); no verdict of its own.
- **Method:** the guarded runner's recognition path on a validation pack (no test access); per-line edit counts saved; strict and normalised tracks; ECE at T = 2.35 under strict fp32, as description only.
- **Variables:** none (reference)
- **Control:** GW-REC-034's training-time validation (6.30%)
- **Result:** 74,500 crops: CER 6.51% [6.36, 6.65] strict, 6.23% [6.09, 6.37] normalised; sequence accuracy 70.62%; ECE 0.228 at T = 1, 0.090 at T = 2.35 (`reports/external/recognition.md`, `calibration.md`).
- **Interpretation:** 0.21 points above the 6.30% logged in training, which used bf16 autocast and the training pack without out-of-charset and long lines. It is H5′'s reference.
- **Limitations:** validation only.
- **Next experiment:** GW-REC-039

### <a id="gw-eff-005"></a>GW-EFF-005 — v1 CPU page latency against PP-OCRv5 mobile, one sitting

- **Date allocated / run:** 2026-11-15 / 2026-11-15
- **Phase:** 14
- **Git commit:** the `phase14-eval` tag   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** the Phase 5 validation profile pages (`e2e_samples` order for `hiertext_val`: 10 warm-up, 100 timed)
- **Hardware:** laptop CPU (i7-14650HX), mains power, Balanced power plan, power mode recorded, no other user program running
- **Research question:** H1′'s latency half.
- **Hypothesis (written before the run):** [H1′](hypotheses.md), latency half.
- **Method:** a fresh GW-BASE-002 profile first, then the frozen pipeline in PyTorch on the CPU, both at 10 threads, batch 1, the same pages in the same order; pairs by position after checking that the input sizes match; `mean_ci` of the per-page differences (pages). GLYPHARIS also at 16 threads, as information. The sitting is void only if a recorded condition fails or the input sizes differ; as amended before the tag (`docs/PHASE_STATUS.md`, Phase 14 item 10), both profiles are timed and GLYPHARIS must start after the baseline ends and within 2 hours, and the 16-thread run cannot void the sitting.
- **Variables:** independent: system; controlled: pages, order, threads, power plan
- **Control:** GW-BASE-002 (fresh profile in the same sitting)
- **Result:** one sitting, baseline 22:28:50 to 22:34:22Z, GLYPHARIS 22:34:40 to 22:39:09Z (2026-11-14 UTC), power mode "best performance" for both: mean 2,446.7 ms per page for GLYPHARIS against 2,954.8 ms for PP-OCRv5 mobile, difference −508.1 [−737.7, −256.5] ms; GLYPHARIS at 16 threads 2,107.1 ms (information); the Phase 5 baseline profile 3,408.2 ms (consistency check). Both `times_ms` arrays are in results.json.
- **Interpretation:** H1′'s latency half is met. The baseline's fresh profile is faster than its Phase 5 one, so the margin is smaller than the Phase 5 numbers suggested.
- **Limitations:** HierText validation pages (no document validation set exists); PyTorch build only (the ONNX v1 build does not exist yet).
- **Next experiment:** the H1′ verdict in GW-E2E-009

### <a id="gw-rec-039"></a>GW-REC-039 — The frozen recognizer on the test crop packs

- **Date allocated / run:** 2026-11-15 / 2026-11-15
- **Phase:** 14
- **Git commit:** the `phase14-eval` tag   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** crop packs in `data/manifests/processed/eval_crops.json` (HierText test 67,574; GDC 8,372; GDC-scan and FUNSD, cut in Phase 14) and `data/manifests/processed/benchmarks.json` (IIIT5K 3,000)
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32
- **Research question:** recognition on ground-truth crops across the test sets (RQ4, RQ6, RQ8).
- **Hypothesis (written before the run):** [H5′](hypotheses.md), [H8′](hypotheses.md).
- **Method:** the guarded test runner (phase 14, split `test`, predictions only; confidences at T = 1 and 2.35; the Phase 15 per-line attributes). Scored once in GW-E2E-009.
- **Variables:** none (one frozen model)
- **Control:** the Phase 5 baselines on the same crops
- **Result:** strict CER: HierText test 6.03% [5.89, 6.18], GDC 4.17% [3.72, 4.67], GDC-scan 1.02% [0.85, 1.20], FUNSD 7.03% [5.71, 8.66]; IIIT5K alphanumeric word accuracy 91.47% [90.43, 92.47]. ECE at T = 1 and 2.35: HierText test 0.223 and 0.092, GDC 0.234 and 0.109, GDC-scan 0.202 and 0.125, FUNSD 0.338 and 0.123, IIIT5K 0.055 and 0.112 (`reports/external/recognition.md`, `calibration.md`).
- **Interpretation:** best of the systems run on HierText test crops (PP-OCRv5 server and PP-OCRv6 medium were not run there) and of all systems on FUNSD crops; behind PP-OCRv6 small and medium and PP-OCRv5 mobile on GDC crops (3.20 to 3.68% against 4.17%) and behind PP-OCRv6 small and medium on GDC-scan, level with PP-OCRv6 tiny on both (paired CIs cross 0); behind both PP-OCRv5 systems and PP-OCRv6 small and medium on IIIT5K. H5′ not met; H8′ transfers met, by half not shown (GW-E2E-009).
- **Limitations:** one frozen model (single seed by design; Phase 14 plan).
- **Next experiment:** GW-E2E-009

### <a id="gw-e2e-007"></a>GW-E2E-007 — The frozen pipeline end to end on the test pages

- **Date allocated / run:** 2026-11-15 / 2026-11-15
- **Phase:** 14
- **Git commit:** the `phase14-eval` tag   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** HierText test pages (Phase 4), GDC v1.1, GDC-scan v1, FUNSD and OmniDocBench English (`data/manifests/processed/benchmarks.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32
- **Research question:** end-to-end page quality across the test sets (RQ1, RQ4, RQ6).
- **Hypothesis (written before the run):** [H1′](hypotheses.md), CER half.
- **Method:** the guarded test runner, pipeline mode; predictions only, scored once in GW-E2E-009.
- **Variables:** none (one frozen model)
- **Control:** the Phase 5 baselines on the same pages
- **Result:** strict: bag-of-lines page CER HierText test 25.16% [23.44, 26.79], GDC 22.35% [15.87, 30.11], GDC-scan 4.60% [2.10, 8.21], FUNSD 32.05% [24.28, 39.85]; order-aware GDC 25.37%, GDC-scan 11.34%, OmniDocBench 27.28% [23.95, 30.57]; E2E F1 HierText test 60.18%, GDC 69.36%, FUNSD 62.81%. HierText official E2E F-score 0.5038; OmniDocBench text-block edit distance 0.1007 (`reports/external/end_to_end.md`, `official.md`).
- **Interpretation:** ahead of every baseline on HierText test; on GDC level with PP-OCRv6 on bag-of-lines error (every paired CI crosses 0) and lower on order-aware error than every PaddleOCR system; on FUNSD bag-of-lines CER level with every PaddleOCR system (every paired CI crosses 0) but lower detection F1 than all of them (79.07% against 81.81 to 85.03%); behind Tesseract on OmniDocBench; rotated GDC pages fail (no orientation step). H1′'s CER half is met (GW-E2E-009).
- **Limitations:** one frozen model; `quad` boxes only.
- **Next experiment:** GW-E2E-009

### <a id="gw-e2e-008"></a>GW-E2E-008 — Oracle run: ground-truth lines through the frozen recognizer

- **Date allocated / run:** 2026-11-15 / 2026-11-15
- **Phase:** 14
- **Git commit:** the `phase14-eval` tag   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** GDC v1.1, GDC-scan v1
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32
- **Research question:** the split of page error between detection and recognition on documents, saved for Phase 15 (H4).
- **Hypothesis (written before the run):** [H4](hypotheses.md), tested in Phase 15.
- **Method:** the guarded test runner, oracle mode (`read_quads` on the box lines); predictions only.
- **Variables:** detector boxes against ground-truth boxes
- **Control:** GW-E2E-007 on the same pages
- **Result:** ground-truth boxes, strict: GDC bag-of-lines page CER 4.17% [2.49, 6.28], order-aware 5.72%, E2E F1 88.34%; GDC-scan 1.01%, 1.02% and 94.97% (`reports/external/end_to_end.md`).
- **Interpretation:** saved for H4 in Phase 15, which reads it from the hand-off files.
- **Limitations:** as GW-E2E-002.
- **Next experiment:** Phase 15

### <a id="gw-e2e-009"></a>GW-E2E-009 — Phase 14 scoring, comparisons and hypothesis verdicts

- **Date allocated / run:** 2026-11-15 / 2026-11-17
- **Phase:** 14
- **Git commit:** recorded in the report   **Config:** —   **Seed(s):** bootstrap seed 20261010, 1,000 resamples
- **Dataset version(s):** every set above; Phase 5 baseline files (hashed before any Phase 14 run)
- **Hardware:** laptop CPU; the official evaluators in `.venv-hiertext` and `.venv-omnidocbench`
- **Research question:** RQ1, RQ4, RQ6, RQ8 on held-out and external data.
- **Hypothesis (written before the run):** [H1′](hypotheses.md), [H5′](hypotheses.md), [H8′](hypotheses.md): five confirmatory tests, everything else descriptive (Phase 14 plan, item 5).
- **Method:** one scoring pass over every complete prediction file, GLYPHARIS and baselines, logged per file (`<system> (<task>), scoring`, phase 14). First the Phase 5 reproduction check; then the §3 metrics on both tracks with the charset ceiling, the official HierText line-level score and OmniDocBench components, paired comparisons, the verdicts, and the Phase 15 hand-off.
- **Variables:** system
- **Control:** GW-BASE-002 for H1′; GW-REC-038 for H5′; T = 1 for H8′
- **Result:** Phase 5 reproduced exactly (37 entries, 13 profiles, 516,474 references). H1′ CER half met, −34.98 [−47.44, −24.08] points; H1′ latency half met, −508.1 [−737.7, −256.5] ms; H1′ holds. H5′ not met, −5.59 [−6.07, −5.01] points. H8′ transfers met, lines −0.125 [−0.132, −0.119], pages [−0.142, −0.101]; by half not shown (lines −0.0081 [−0.0142, −0.0021], pages [−0.0168, +0.0044]); H8 not supported on GDC. Every table in `reports/external/`, every number in `results.json`.
- **Interpretation:** see `docs/PHASE_STATUS.md`, Phase 14 results, and findings F21 to F26. The pass stopped once in HierText's official evaluator and was finished with `score --resume`, logged as a resume (`reports/external/run_record.md`).
- **Limitations:** stated with the plan (`docs/PHASE_STATUS.md`, Phase 14).
- **Next experiment:** Phase 15


### <a id="gw-rec-040"></a>GW-REC-040 — Recognition errors of the frozen recognizer (Phase 15)

- **Date allocated / run:** 2026-11-17 / 2026-11-17
- **Phase:** 15
- **Git commit:** `ba44a84` (reports `6cb1b9c`)   **Config:** `configs/final/glypharis_v1.yaml` (no model run)   **Seed(s):** bootstrap 20261010; example draws 20261115
- **Dataset version(s):** the Phase 14 hand-off (`data/manifests/processed/phase14_outputs.json`), the Phase 5 baseline rec files (`phase5_baseline_files.json`), GW-REC-038's validation predictions
- **Hardware:** laptop CPU
- **Research question:** RQ5: which recognition failure modes dominate, on documents and scene text?
- **Hypothesis (written before the run):** none; a categorised description (`docs/PHASE_STATUS.md`, Phase 15 plan, items 3 and 4).
- **Method:** edit operations on the normalised (ref, pred) pairs; character and word confusions; slices with validation-derived bins; one primary category per wrong line; annotation-noise count; baselines' operation rates; seeded example draws.
- **Variables:** none (one frozen model)
- **Control:** the Phase 5 baselines on the same crops; GW-REC-038 on validation
- **Result:** strict CER with page CIs (IIIT5K alphanumeric, line CI) and substitution, insertion and deletion rates: HierText test 6.03% [5.60, 6.48] (3.27, 0.89, 1.88), validation 6.51% [5.98, 7.12] (3.41, 0.97, 2.12), GDC 4.17% [2.48, 6.22] (0.86, 0.12, 3.19), FUNSD 7.03% [5.59, 8.61] (2.12, 1.45, 3.46), GDC-scan 1.02% [0.79, 1.30] (numbers only), IIIT5K 2.37% [2.06, 2.70]. Largest primary categories, share of character errors: HierText test rotation ≥ 5° 34.92%, height in the lowest quartile 21.30% (validation 37.69%, 18.31%); GDC digit 49.21%, contrast in the lowest quartile 20.41%, out of charset 13.62%; FUNSD digit 43.21%, punctuation 31.32%. GDC deletions: 4,941 spaces (10.165% of reference spaces) and 4,153 full stops (44.176% of reference full stops); the 20 worst GDC lines are all table-of-contents dot leaders (`census_income`, `hilbert_geometry`); document groups book 15.98% to paper 1.02%. HierText test: the lowest confidence quartile holds 78.51% of the errors; 15 likely label fragments, 0.30% of the errors. Error share of lines of 70 characters or more: validation 3.59%, HierText test 4.74%, GDC 61.80% (`reports/error_cases/phase15/recognition.md`, `examples.md`, `reports/confusion_analysis/GW-REC-040/`).
- **Interpretation:** the two domains fail differently. Scene text fails on geometry (rotation and small lines), which fits the rotation weakness F15 found on GLYPH-C. GDC's documents lose characters rather than misread them: merged words and dropped full stops, with table-of-contents dot leaders as the worst lines. On the same lines PP-OCRv5 mobile and the three PP-OCRv6 systems (GW-BASE-002, 005 to 007) read more of the punctuation as itself (86.90 to 89.95% against 73.02%; point values); PP-OCRv5 server (GW-BASE-003) reads 67.62%. ADR-0002's revisit trigger is not met: long lines do not dominate validation error. Nothing flows back into v1. Findings F30 and F31.
- **Limitations:** stated with the plan (no labels for touching, broken, noise or fonts on test; GDC-scan numbers only).
- **Next experiment:** Phase 16

### <a id="gw-e2e-010"></a>GW-E2E-010 — Pipeline errors of the frozen pipeline (Phase 15)

- **Date allocated / run:** 2026-11-17 / 2026-11-17
- **Phase:** 15
- **Git commit:** `ba44a84` (reports `6cb1b9c`)   **Config:** `configs/final/glypharis_v1.yaml` (no model run)   **Seed(s):** bootstrap 20261010
- **Dataset version(s):** the Phase 14 hand-off; the baselines' OmniDocBench end-to-end files (line order only)
- **Hardware:** laptop CPU
- **Research question:** RQ5: does page error come from the detector or the recognizer, and where?
- **Hypothesis (written before the run):** H4 has no registered operational rule and the published per-condition numbers decide it; it is closed post hoc from this experiment (Phase 15 plan, item 5).
- **Method:** the page error budget (missed lines, matched-line errors, spurious predictions), unmatched lines by best overlap, matched-line CER by IoU and against the oracle, recall slices, reading-order excess and inversion rate, the oracle share per GDC slice with page and document-cluster CIs.
- **Variables:** none (one frozen model)
- **Control:** GW-E2E-008 (oracle) on GDC and GDC-scan
- **Result:** page error budget (missed lines, matched lines, spurious predictions; shares of bag-of-lines errors, page CI): HierText test 49.45% [47.47, 51.56], 25.79% [23.51, 28.00], 24.77% [23.24, 26.38]; GDC 45.32% [36.14, 52.80], 27.55% [15.80, 42.07], 27.13% [20.89, 33.57]; GDC-scan 34.36%, 32.28%, 33.36%; FUNSD 40.22%, 19.39%, 40.39%; GDC `rotated` pages 78.16% [45.80, 97.26] on matched lines (document CI). Oracle share on GDC: 18.66% [11.55, 28.73] (documents [11.51, 26.45]); clean slice 32.70% [16.64, 51.91]; other five conditions 10.29% [5.24, 18.03]; `scan_app` 45.14% [25.40, 58.41]; `rotated` 1.03%; GDC-scan 22.08% [12.65, 47.58]. Recall at IoU 0.5 on HierText test 57.92% for lines under 9.35 px and 49.53% for |θ| ≥ 15°; on GDC 0 of 113 lines under 9.35 px. Matched-line CER 29.24% at IoU [0.5, 0.7) and 3.63% at 0.85 or more (HierText test); GDC pipeline minus oracle on matched lines +2.048 [0.753, 3.817] errors per line. GDC reading order: 14.56% [9.93, 20.36] of consecutive matched lines reversed. OmniDocBench: 76.66 to 77.74% of the PaddleOCR systems' consecutive output lines have a top edge that does not rise (`reports/error_cases/phase15/pipeline.md`, `oracle_share.md`). The OmniDocBench bag-of-lines CERs in `pipeline.md` (about 180%) are not meaningful: its reference is blocks, not lines.
- **Interpretation:** on HierText test, GDC and FUNSD, 40.22 to 49.45% of the page error is ground-truth lines no prediction matched at IoU 0.5 (on GDC mostly lines a box barely touched or poorly localised), and errors on matched lines are 19.39 to 27.55%, on documents as on scene text (F10 on validation). H4 is not supported on GDC (post hoc): the oracle share's point estimate is under half in every slice, the clean one included. The OmniDocBench measure does not decide F26's proposed cause: PaddleOCR's own sort (`SortQuadBoxes`, read from the installed code) lets a following line sit up to 10 px higher, so the measure would not be 1 under it either and cannot tell that sort apart from other orders. F26's cause stays unmeasured. Findings F27 to F29 and F33.
- **Limitations:** no merge or split labels and no detector candidates on test; the oracle share is not a pure detector share.
- **Next experiment:** Phase 16

### <a id="gw-rec-041"></a>GW-REC-041 — The frozen recognizer on GlyphSynth validation lines

- **Date allocated / run:** 2026-11-17 / 2026-11-17
- **Phase:** 15
- **Git commit:** `ba44a84` (as recorded in the run metadata, `data/processed/phase15/GW-REC-041/glyphsynth_val.meta.json`; reports `6cb1b9c`)   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none (no training); bootstrap 20261010
- **Dataset version(s):** GlyphSynth v1 validation (`data/processed/glyphsynth/lines_val.*`, with its render log)
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32 as in Phase 14
- **Research question:** the failure modes the real sets have no labels for: unusual fonts, touching and broken characters, background noise, neighbour fragments.
- **Hypothesis (written before the run):** none; description.
- **Method:** the frozen recognizer from the freeze record reads every validation line; CER by font family and category, held-out fonts, italic, inverted, neighbours and each capture step, line bootstrap.
- **Variables:** independent: the generator's labels
- **Control:** the same model's clean-line CER on GlyphSynth validation
- **Result:** 10,000 lines read in 79.3 s on the GPU (`wall_seconds` in the run metadata, `glyphsynth_val.meta.json`); the pack's hash checked against its metadata and the GlyphSynth manifest before the run, and the output's hash and count against the run metadata before the analysis. Strict CER 0.71% [0.63, 0.79] (normalised 0.51%). Ink spread (touching) 1.41% [1.17, 1.72] on 1,543 lines with 31.11% of the errors, 0.58% [0.50, 0.67] without; underline 1.37% [1.05, 1.76]; binarisation 1.18% [0.81, 1.67]; inverted 1.06% [0.62, 1.72]; italic 0.86% [0.67, 1.12]; broken group 0.76% [0.63, 0.90]; background-noise group 0.76% [0.65, 0.89]; neighbour fragments 0.66% [0.56, 0.78]; font families 0.38% (`worksans`) to 1.48% [1.09, 1.93] (`cormorantgaramond`); serif 0.83%, sans 0.60%, mono 0.62%. The held-out-font slice was dropped by the amendment (every validation line uses a validation font) (`reports/error_cases/phase15/glyphsynth.md`).
- **Interpretation:** on rendered lines every labelled slice reads at 1.48% CER or less; the 1,543 ink-spread lines carry 31.11% of the errors. Synthetic lines say little about photographed text (F16), so these slices describe the recognizer, not the test sets. Finding F32.
- **Limitations:** synthetic lines; validation only.
- **Next experiment:** Phase 16


### <a id="gw-eff-006"></a>GW-EFF-006 — Reference outputs and the v1 fp32 ONNX bundle

- **Date allocated / run:** 2026-11-17 / 2026-11-17
- **Phase:** 16
- **Git commit:** `9f58c4b` with the fixes `6950a5a`, `a8e8bb6` and `1253dfe`; the references, the bundle and the check record `6950a5a` (reports `e04f54e`, scored at `1253dfe`)   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none (no training); bootstrap 20261010
- **Dataset version(s):** HierText validation: the Phase 5 profile pages and 400-page sample, `rec_samples`, the 5,000 lines seeded 20261028 from `processed/hiertext/lines_validation`
- **Hardware:** laptop CPU (and the RTX 4050 for GW-EFF-007's GPU rows); mains, Balanced plan, power mode recorded
- **Research question:** is the plain fp32 ONNX export of v1 the same model (RQ7)?
- **Hypothesis (written before the run):** none; Phase 17's four registered tolerances decide it (Phase 16 plan, item 1).
- **Method:** write detector maps (110 profile pages), recognizer logits (5,000 lines) and page predictions (400 pages) from PyTorch v1 on the CPU, hash them; build the bundle from the freeze record; check it against the saved outputs.
- **Variables:** see `docs/PHASE_STATUS.md`, Phase 16 plan
- **Control:** the saved PyTorch v1 outputs
- **Result:** the first steps of the Phase 16 chain (94 steps in one detached chain, 2026-11-17T13:44:59Z to 2026-11-18T05:36:52Z). References written once from PyTorch v1 on the CPU, their SHA-256 in `reference_manifest.json` (SHA-256 `4ff7f155…`). Bundle `runs/release/glypharis-v1`, built from the freeze record (record SHA-256 `0aa61f9e…`, that of `configs/final/glypharis_v1.yaml`), manifest SHA-256 `875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`, named at 13:45:10Z before any timed run. Phase 17's four checks against the references (14:22:06 to 15:31:40Z), limit and measured: detector map max abs difference 1e-4, 7.33e-6 on 110 pages; recognizer logits 1e-3, 3.09e-4 on 5,000 lines; greedy text, no differences except near ties (top-2 gap < 1e-3) on at most 0.1% of lines, 0 other differences and 0 near ties; page CER within 0.05 points (5e-4), 24.32% on both backends, difference 1.49e-5, on 400 pages. All four pass (`reports/efficiency/phase16/README.md`, `reference_check.json`, `v1_bundle.json`).
- **Interpretation:** by Phase 17's registered tolerances the plain fp32 ONNX export (TorchScript exporter) is the same model as PyTorch v1, so its timings in GW-EFF-007 to 010 are v1's in ONNX Runtime, not a variant's. This is the bundle the Phase 17 re-run cites, and the H1′ ONNX verdict (GW-EFF-010) applies to its hash only. Finding F34.
- **Limitations:** TorchScript exporter only (as Phase 17).
- **Next experiment:** Phase 17 re-run (the v1 bundle; GW-EFF-012)

### <a id="gw-eff-007"></a>GW-EFF-007 — The v1 cost table

- **Date allocated / run:** 2026-11-17 / 2026-11-17
- **Phase:** 16
- **Git commit:** `9f58c4b` with the fixes `6950a5a`, `a8e8bb6` and `1253dfe` (reports `e04f54e`, scored at `1253dfe`); the five line profiles and the PyTorch CPU page profile record `6950a5a`, the GPU page profile and the two ONNX page profiles `a8e8bb6` (their bundle `6950a5a`)   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none (no training); bootstrap 20261010
- **Dataset version(s):** HierText validation: the Phase 5 profile pages and 400-page sample, `rec_samples`, the 5,000 lines seeded 20261028 from `processed/hiertext/lines_validation`
- **Hardware:** laptop CPU (and the RTX 4050 for GW-EFF-007's GPU rows); mains, Balanced plan, power mode recorded
- **Research question:** what does v1 cost on this laptop (RQ7, RQ4)?
- **Hypothesis (written before the run):** none; description.
- **Method:** profiles of PyTorch v1 and the GW-EFF-006 bundle at 10 threads (ONNX also at the default), GPU PyTorch rows, the page-time split and the lines-per-page slope; mirrored run order in one sitting.
- **Variables:** see `docs/PHASE_STATUS.md`, Phase 16 plan
- **Control:** —
- **Result:** one sitting, 2026-11-17, 15:31:49 to 16:20:07Z, each row one run; mains, Balanced plan and power mode "best performance" at the start and end of every profile, no sitting void. Parameters: detector 3,187,209, recognizer 7,928,835, total 11,116,044; files: detector 12.44 MiB as release .pt and 12.15 MiB as fp32 ONNX, recognizer 30.31 and 30.40 MiB (the INT8 recognizer 9.82 MiB, GW-EFF-008). Lines, batch 1 (mean, p95, peak RAM): PyTorch CPU 10 threads 33.90 [29.92, 38.13] ms, 77.60 ms, 825.7 MiB; fp32 ONNX 10 threads 92.25 [70.69, 116.28] ms, 304.39 ms, 1,028.0 MiB; fp32 ONNX default threads 104.73 [79.87, 132.05] ms, 392.09 ms, 1,043.6 MiB; PyTorch GPU 12.22 [11.24, 13.33] ms, 19.01 ms, 81.8 lines/s, torch peak VRAM 77.3 MiB; GPU batch 32 199.89 [187.57, 213.04] ms per batch, 160.1 lines/s, 282.2 MiB. Pages (110 profile pages, 40.9 lines per page): PyTorch CPU 2,566 [2,327, 2,829] ms, p95 5,127 ms, 1,641.8 MiB; fp32 ONNX 10 threads 8,574 [7,516, 9,766] ms, p95 19,011 ms, 6,529.6 MiB; fp32 ONNX default threads 9,925 [8,744, 11,261] ms, p95 22,072 ms, 6,541.5 MiB; PyTorch GPU 1,049 [825, 1,300] ms, p95 3,288 ms, 2,450.0 MiB RAM, torch peak VRAM 432.4 MiB (whole GPU 1,335 MiB at the end, display included; nvidia-smi gives no per-process figure under the Windows driver). Stage split, mean ms per page: PyTorch CPU detector network 296.4 (12%), recognizer network 2,148.2 (84%), detector post-processing 43.6, cropping 38.8; fp32 ONNX 10 threads detector network 297.5 (3%), recognizer network 8,167.7 (95%); at default threads 484.3 (5%) and 9,289.5 (94%); PyTorch GPU detector network 694.5 (66%), recognizer network 261.0 (25%); every other stage under 60 ms on every build. Page time against lines per page (least squares, page bootstrap): 28.6 [19.2, 42.9] ms per line for PyTorch CPU, 72.6 [32.9, 142.7] for fp32 ONNX at 10 threads, 85.5 [41.6, 165.1] at default threads, 4.8 [−1.4, 16.7] on the GPU. ADR-0011's targets: every CPU build is over on line mean and p95 and page mean and p95; PyTorch's page RAM (1,641.8 MiB) is within, fp32 ONNX's is over (`reports/efficiency/phase16/cost_table.md`, `README.md`; beside the Phase 5 baselines, different sittings, in `reports/benchmark_results/efficiency_comparison.md`). The GPU page profile stopped once on the profiler's own stage-sum check and was run again from its start after `a8e8bb6` (`research/failures.md`); no reported number depends on that tolerance.
- **Interpretation:** exported as it is, v1 is slower in ONNX Runtime than in PyTorch on this CPU: about 2.7 times per line and about 3.3 times per page. The detector network takes the same time on both (297.5 against 296.4 ms per page), so F13's "twice as fast" exported detector, an arena-on side test, does not hold with the arena off, as the backends run; the recognizer network is the whole difference (8,167.7 against 2,148.2 ms). At ONNX Runtime's default threads its means are higher (104.73 ms per line, 9,925 ms per page; one run each, not compared in a mirrored order). No CPU build of v1 meets ADR-0011's latency targets. On the GPU the detector network takes longer per page than on the CPU (694.5 against 296.4 ms); this is not explained here. Findings F34 and F38.
- **Limitations:** one laptop; HierText validation pages. Each row is one run, in chain order rather than the mirrored order the Method names (a deviation, `docs/PHASE_STATUS.md`), so the builds are not compared here in a mirrored order (GW-EFF-009's sittings do that for v1's two CPU builds); the GPU rows are PyTorch only.
- **Next experiment:** Phase 17 re-run (the v1 bundle; GW-EFF-012)

### <a id="gw-eff-008"></a>GW-EFF-008 — Dynamic INT8 of the v1 ONNX recognizer

- **Date allocated / run:** 2026-11-17 / 2026-11-17
- **Phase:** 16
- **Git commit:** `9f58c4b` with the fixes `6950a5a`, `a8e8bb6` and `1253dfe`; every run records `a8e8bb6` (reports `e04f54e`, scored at `1253dfe`)   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none (no training); bootstrap 20261010
- **Dataset version(s):** HierText validation: the Phase 5 profile pages and 400-page sample, `rec_samples`, the 5,000 lines seeded 20261028 from `processed/hiertext/lines_validation`
- **Hardware:** laptop CPU (and the RTX 4050 for GW-EFF-007's GPU rows); mains, Balanced plan, power mode recorded
- **Research question:** does dynamic INT8 make the recognizer faster at little cost (RQ7)?
- **Hypothesis (written before the run):** [H7′](hypotheses.md).
- **Method:** the registered quantisation recipe; latency on `rec_samples` (batch 1, 10 threads, order fp32, INT8, INT8, fp32); CER on the 5,000 lines, strict.
- **Variables:** see `docs/PHASE_STATUS.md`, Phase 16 plan
- **Control:** the GW-EFF-006 fp32 recognizer
- **Result:** the registered recipe (onnxruntime 1.31.1, onnx 1.23.2) turns 49 of the 67 fp32 MatMuls into MatMulInteger with 49 DynamicQuantizeLinear nodes; 18 MatMuls, the attention products among them, stay fp32; file 30.40 → 9.82 MiB (SHA-256 `e7a4ee3b…`). Latency sitting, 2026-11-17, 16:38:41 to 16:39:58Z, order fp32, INT8, INT8, fp32: fp32 95.78 ms per line mean (runs 96.61 and 94.95), p95 346.50 ms; INT8 61.84 ms (63.21 and 60.46), p95 232.02 ms; t_INT8 − 0.75 × t_fp32 = −10.00 [−12.79, −7.37] ms: **met**. Accuracy on the 5,000 lines, strict, greedy: line CER 6.221% fp32, 6.208% INT8; the registered margin test −0.512 [−0.541, −0.484] points: **met**. **H7′ holds.** On the pages (GW-EFF-009, descriptive), INT8 minus v1's fp32 ONNX build: page time −2,848.5 [−3,233.6, −2,505.1] ms (6,890.9 against 9,739.4 ms), page CER +0.00 [−0.02, +0.02] points, peak RAM 6,896.5 against 6,525.5 MiB (`reports/efficiency/phase16/int8.md`, `verdicts.md`, `pareto.md`).
- **Interpretation:** dynamic INT8 of the constant-weight MatMuls takes about a third off the recognizer's line time, at a line CER of 6.208% against 6.221%, and shrinks the file to about a third. The attention products stay fp32, and on pages it removes less time than fusing the attention (GW-EFF-009: 6,890.9 against 4,020.8 ms per page). The INT8 recognizer is a new version, not v1, measured on validation only; it does not replace v1, and shipping it is the owner's decision (ADR-0011), with a Phase 22 evaluation if it ships. Finding F36.
- **Limitations:** the attention products stay fp32; an INT8 bundle is a new version.
- **Next experiment:** Phase 17 re-run (the v1 bundle; GW-EFF-012)

### <a id="gw-eff-009"></a>GW-EFF-009 — Pareto points (new versions)

- **Date allocated / run:** 2026-11-17 / 2026-11-17 to 2026-11-18
- **Phase:** 16
- **Git commit:** `9f58c4b` with the fixes `6950a5a`, `a8e8bb6` and `1253dfe`; every run records `a8e8bb6`; `1253dfe` changed only how the 19 variant reports of `det-GW-DET-003` and the `rec-GW-REC-*` points record their checkpoint (`checkpoints/<ID>/best.pt`), and the re-scoring changed no number (`research/failures.md`) (reports `e04f54e`, scored at `1253dfe`)   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none (no training); bootstrap 20261010
- **Dataset version(s):** HierText validation: the Phase 5 profile pages and 400-page sample, `rec_samples`, the 5,000 lines seeded 20261028 from `processed/hiertext/lines_validation`
- **Hardware:** laptop CPU (and the RTX 4050 for GW-EFF-007's GPU rows); mains, Balanced plan, power mode recorded
- **Research question:** where do the variants sit on accuracy against latency (RQ7)?
- **Hypothesis (written before the run):** none; description.
- **Method:** each variant built by overriding one named setting; page CER and E2E F1 on the 400 pages (CPU), latency and peak RAM on the 110 profile pages; recognizer variants by line CER and line latency.
- **Variables:** see `docs/PHASE_STATUS.md`, Phase 16 plan
- **Control:** v1 (GW-EFF-006, GW-EFF-007)
- **Result:** accuracy runs 2026-11-17, 16:20:50Z to 2026-11-18, 00:59:49Z; page sitting 00:59:56 to 05:29:38Z and line sitting 05:29:50 to 05:34:57Z (2026-11-18), each in the mirrored order, each item the mean of its two runs; no sitting void, every variant run (the cut line was not needed). Every point but v1 is a new version, not v1; intervals are descriptive, differences are the point minus v1's fp32 ONNX build. Pages (page CER and E2E F1 on the 400 pages, strict; mean, p95 and peak RAM on the 110 profile pages): v1 PyTorch 24.32% [20.87, 28.15], 60.50%, 2,637.3 [2,395.4, 2,909.1] ms, 5,327.7 ms, 1,665.0 MiB; v1 fp32 ONNX 24.32%, 60.50%, 9,739.4 [8,510.8, 11,113.1] ms, 22,474.1 ms, 6,525.5 MiB; detector long side 1024 31.62%, 55.95%, 8,491.3 ms (+7.29 [+5.55, +9.51] points, −1,248.1 [−1,761.3, −734.3] ms); 1280 26.63%, 58.98%, 9,537.2 ms (+2.31 [+1.66, +2.98], −202.2 [−451.2, +32.6]); 1792 24.21%, 60.67%, 10,168.1 ms (−0.11 [−0.93, +0.92], +428.7 [+230.7, +647.8]); 2048 23.75%, 61.10%, 10,400.0 ms (−0.57 [−1.80, +0.83], +660.6 [+393.3, +949.6]); ONNX detector with the PyTorch recognizer 24.32%, 60.50%, 2,579.1 [2,340.0, 2,844.7] ms, p95 5,186.9 ms, 1,550.0 MiB (+0.00 [+0.00, +0.00], −7,160.4 [−8,258.3, −6,171.7]); INT8 recognizer 24.32%, 60.46%, 6,890.9 ms, 6,896.5 MiB (+0.00 [−0.02, +0.02], −2,848.5 [−3,233.6, −2,505.1]); fused attention (each global mixer exported as one `com.microsoft.MultiHeadAttention` node) 24.32%, 60.50%, 4,020.8 [3,604.2, 4,479.8] ms, p95 8,267.1 ms, 2,393.0 MiB (+0.00 [+0.00, +0.00], −5,718.6 [−6,630.9, −4,883.7]); detector GW-DET-003 27.33%, 57.75%, 9,065.7 ms (+3.00 [+1.52, +4.33], −673.7 [−888.5, −443.8]). Lines (line CER on the 5,000 lines, strict, greedy; mean and p95 on 100 `rec_samples` lines): v1 PyTorch 6.22% [5.73, 6.71], 35.55 [31.32, 40.10] ms, 80.82 ms; v1 fp32 ONNX 6.22%, 95.59 [73.55, 120.41] ms, 340.04 ms; INT8 6.21%, 63.35 ms (−0.01 [−0.04, +0.02] points, −32.24 [−39.97, −24.87] ms); fused attention 6.22%, 39.53 [32.12, 47.52] ms, 112.10 ms (−56.06 [−72.68, −41.23] ms); GW-REC-003 8.34%, 9.07 ms (+2.12 [+1.82, +2.40] points); GW-REC-006 7.85%, 16.67 ms (+1.63 [+1.39, +1.87]); GW-REC-020 8.21%, 29.03 ms (+1.99 [+1.72, +2.23]); GW-REC-022 7.57%, 100.57 ms (+1.35 [+1.13, +1.55] points, +4.98 [+3.12, +6.86] ms). (`reports/efficiency/phase16/pareto.md`, `pareto.png`; the options table for the owner in `README.md`.)
- **Interpretation:** the recognizer sets the page time. Fusing the attention keeps v1's page CER, E2E F1 and line CER and takes the ONNX page time from 9,739.4 to 4,020.8 ms and peak RAM from 6,525.5 to 2,393.0 MiB, so the plain export's attention is where the ONNX build loses most of its time. PyTorch takes 2,637.3 ms per page and 35.55 ms per line in the same sittings; no interval covers that pair, so no difference is stated. Among the points at v1's accuracy, the ONNX detector with the PyTorch recognizer has the lowest mean page time (2,579.1 ms mean, 5,186.9 ms p95, 1,550.0 MiB; about 58 ms below v1-torch's 2,637.3 ms in the same sittings, and no interval for that difference is reported), still over both of ADR-0011's page time targets (within the RAM target). Detector input size and the smaller detector move page accuracy but change the ONNX page time by at most about 13%: 1024 costs +7.29 points for −1,248.1 ms, and GW-DET-003 +3.00 points for −673.7 ms. Every other recognizer reads with a higher line CER than v1's; GW-REC-003 has the lowest mean line time (9.07 ms) at +2.12 points. These are new versions on validation; none replaces v1, and choosing a default backend or shipping a variant is the owner's decision (ADR-0011); Phase 16 changes no default. Findings F35, F37 and F38.
- **Limitations:** variants are new versions on validation only; the size points differ in recipe and budget.
- **Next experiment:** Phase 17 re-run (the v1 bundle; GW-EFF-012)

### <a id="gw-eff-010"></a>GW-EFF-010 — H1′'s ONNX latency verdict

- **Date allocated / run:** 2026-11-17 / 2026-11-17
- **Phase:** 16
- **Git commit:** `9f58c4b` with the fixes `6950a5a`, `a8e8bb6` and `1253dfe`; both profiles record `a8e8bb6` (the bundle `6950a5a`) (reports `e04f54e`, scored at `1253dfe`)   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none (no training); bootstrap 20261010
- **Dataset version(s):** HierText validation: the Phase 5 profile pages and 400-page sample, `rec_samples`, the 5,000 lines seeded 20261028 from `processed/hiertext/lines_validation`
- **Hardware:** laptop CPU; mains, Balanced plan, power mode recorded
- **Research question:** is the v1 ONNX build faster per page than PP-OCRv5 mobile on the CPU (RQ1)?
- **Hypothesis (written before the run):** [H1′](hypotheses.md), latency half, the ONNX build's separate verdict.
- **Method:** Phase 14's rule (items 5 and 10) with the GW-EFF-006 bundle named by its SHA-256 before the sitting.
- **Variables:** see `docs/PHASE_STATUS.md`, Phase 16 plan
- **Control:** a fresh GW-BASE-002 profile in the same sitting (amendment to GW-BASE-002)
- **Result:** bundle named before the sitting (2026-11-17T13:45:10Z), manifest SHA-256 `875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`. One sitting, 2026-11-17: GW-BASE-002 16:40:45 to 16:46:30Z, then the v1 fp32 ONNX bundle 16:46:42 to 17:04:02Z, both at 10 threads, power mode "best performance" for both. Mean 9,623.6 ms per page for the bundle against 3,086.0 ms for PP-OCRv5 mobile, difference +6,537.6 [+5,383.7, +7,806.3] ms: **not met** (`reports/efficiency/phase16/verdicts.md`; the bundle's per-page times in `reports/efficiency/phase16/profiles/pages_onnx_cpu_10t_GW-EFF-010.json`, the baseline profile's SHA-256 in `results.json`).
- **Interpretation:** H1′'s ONNX verdict is not met: v1's plain fp32 ONNX build, in this bundle, is slower per page than PP-OCRv5 mobile on this CPU, by about 3.1 times. It applies to this bundle hash only; if the Phase 17 re-run ships another bundle, the verdict does not carry over. H1′ itself still holds from Phase 14, its latency half judged on the PyTorch build (GW-EFF-005). The same bundle's page means in other sittings, as information: 8,573.7 ms (GW-EFF-007) and 9,739.4 ms (GW-EFF-009). Finding F40.
- **Limitations:** HierText validation pages.
- **Next experiment:** Phase 17 re-run (the v1 bundle; GW-EFF-012)

### <a id="gw-eff-011"></a>GW-EFF-011 — H6′'s latency: SVTR against CRNN in ONNX

- **Date allocated / run:** 2026-11-17 / 2026-11-18
- **Phase:** 16
- **Git commit:** `9f58c4b` with the fixes `6950a5a`, `a8e8bb6` and `1253dfe`; every run records `a8e8bb6` (reports `e04f54e`, scored at `1253dfe`)   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none (no training); bootstrap 20261010
- **Dataset version(s):** HierText validation: the Phase 5 profile pages and 400-page sample, `rec_samples`, the 5,000 lines seeded 20261028 from `processed/hiertext/lines_validation`
- **Hardware:** laptop CPU; mains, Balanced plan, power mode recorded
- **Research question:** is the SVTR-style recognizer within 1.5 times the CRNN's CPU latency (RQ7)?
- **Hypothesis (written before the run):** [H6′](hypotheses.md).
- **Method:** GW-REC-003 and GW-REC-020 exported to fp32 ONNX, each checked against its own PyTorch outputs; `rec_samples`, batch 1, 10 threads, order CRNN, SVTR, SVTR, CRNN.
- **Variables:** see `docs/PHASE_STATUS.md`, Phase 16 plan
- **Control:** GW-REC-003
- **Result:** export checks on the 5,000 lines, height 32: GW-REC-003 max logit difference 1.56e-04, text identical on 5,000 lines, 0 near ties; GW-REC-020 2.89e-04, 5,000, 0; both pass. Sitting 2026-11-18, 05:35:10 to 05:36:07Z, order CRNN, SVTR, SVTR, CRNN: CRNN 10.16 ms per line mean (runs 10.83 and 9.48), SVTR 32.01 ms (31.23 and 32.80); t_SVTR − 1.5 × t_CRNN = +16.78 [+12.02, +22.04] ms: **not met**. PyTorch at 10 threads, information: GW-REC-003 13.00 ms (p95 27.52), GW-REC-020 15.70 ms (p95 32.40) (`reports/efficiency/phase16/verdicts.md`).
- **Interpretation:** H6′ is not met: in fp32 ONNX the SVTR-style recognizer takes about 3.2 times the CRNN's time per line, against about 1.2 times in PyTorch at 10 threads, which matches Phase 7's 16-thread timings (SVTR slower than CRNN by less than 1.5 times; information, not a test). The plain export's unfused attention (F35) is the likely reason the ONNX gap is wider than PyTorch's; a fused export of GW-REC-020 was not measured. H6′, which supersedes H6's latency half, is not met (2026-11-18, GW-EFF-011); with its CER half exploratory, H6 as a whole stays at most inconclusive, as registered. Finding F39.
- **Limitations:** height 32 pair (H6's matched budget); v1 is height 48.
- **Next experiment:** Phase 17 re-run (the v1 bundle; GW-EFF-012)

### <a id="gw-eff-012"></a>GW-EFF-012 — The v1 bundle on the PyTorch backend (Phase 17 re-run)

- **Date allocated / run:** 2026-11-18 / 2026-11-18
- **Phase:** 17 (v1 re-run)
- **Git commit:** `8c70747` (recorded in the report, clean tree)   **Config:** `configs/final/glypharis_v1.yaml` (through the bundle)   **Seed(s):** none (no training); the 5,000 lines seeded 20261028
- **Dataset version(s):** HierText validation: the Phase 5 profile pages (110), the 5,000 lines seeded 20261028 from `processed/hiertext/lines_validation`, the 400 Phase 5 pages; the Phase 16 reference outputs (`reports/efficiency/phase16/reference_manifest.json`)
- **Hardware:** laptop CPU, 10 threads (an equivalence check, not a timing: power conditions are not recorded)
- **Research question:** does the path v1 ships on, the GW-EFF-006 bundle read with the PyTorch backend, give the PyTorch v1 references' outputs?
- **Hypothesis (written before the run):** none; Phase 17's four tolerances are the pass rule (`docs/PHASE_STATUS.md`, Phase 17 v1 re-run, item 4).
- **Method:** `GlypharisOCR.from_pretrained(bundle, backend="torch")` on the bundle with manifest SHA-256 `875357a6…`; Phase 16's reference check unchanged (detector maps, recognizer logits and text, page CER).
- **Variables:** none (one frozen model, one backend)
- **Control:** the Phase 16 PyTorch v1 reference outputs (GW-EFF-006)
- **Result:** pass on all four tolerances, with no difference at all (`reports/export/GW-EFF-012/torch_bundle_check.json`, 2026-11-18T07:29:27Z to 07:50:27Z, 10 threads, bundle manifest SHA-256 `875357a6…`, `model_version` `glypharis_v1`): detector map max abs difference 0.0 on the 110 profile pages (limit 1e-4); logits 0.0 on the 5,000 lines (limit 1e-3), 5,000 of 5,000 texts identical, no near ties; page CER 24.32% for the references and the bundle, difference 0.0 (limit 0.05 points), all 400 pages with identical text.
- **Interpretation:** the path v1 ships on, the bundle's release files read by the PyTorch modules through `from_pretrained`, gives exactly the outputs of the PyTorch v1 build Phase 14 evaluated, so Phase 14's results apply to it unchanged. With GW-EFF-006 for the ONNX files, the v1 export is declared successful (Phase 17 v1 re-run).
- **Limitations:** HierText validation; CPU only.
- **Next experiment:** Phase 18 re-run (the package on the v1 bundle)

### <a id="gw-e2e-011"></a>GW-E2E-011 — The package on the v1 bundle against the frozen pipeline (Phase 18 re-run)

- **Date allocated / run:** 2026-11-18 / 2026-11-18
- **Phase:** 18 (v1 re-run)
- **Git commit:** `c5d7011` (recorded in the report, clean tree)   **Config:** `configs/det/gw-det-001.yaml` (pages); the model from the bundle   **Seed(s):** none (no training)
- **Dataset version(s):** the 400 Phase 5 HierText validation pages; GW-E2E-006's predictions (`data/processed/phase14/GW-E2E-006/hiertext_val.jsonl`, SHA-256 `c8520576…`)
- **Hardware:** laptop CPU (the package); GW-E2E-006 ran on the RTX 4050 in strict fp32
- **Research question:** does the installed package, loading the v1 bundle as it ships, read the validation pages like the frozen v1 pipeline Phase 14 evaluated?
- **Hypothesis (written before the run):** none; the pass rule is Phase 18's (page CER within 0.05 points; `docs/PHASE_STATUS.md`, Phase 18 v1 re-run, item 3).
- **Method:** `scripts/package_check.py --v1 --backend torch` on the bundle with manifest SHA-256 `875357a6…`; strict bag-of-lines page CER of both, and the share of lines with identical text and box (corners within 1 px).
- **Variables:** none
- **Control:** GW-E2E-006 (the frozen pipeline on the same pages)
- **Result:** pass (`reports/package/GW-E2E-011.json`, 2026-11-18T08:33:54Z to 08:52:03Z, PyTorch backend, 10 torch threads, `--v1` checks passed, bundle manifest SHA-256 `875357a6…`, reference SHA-256 `c8520576…`): the package reads the 400 Phase 5 validation pages with 24.32% page CER, as GW-E2E-006 did, difference 0.0 points (limit 0.05); 16,835 of 16,835 lines have identical text and box (corners within 1 px).
- **Interpretation:** the installed package on the v1 bundle reads these pages line for line as the frozen pipeline Phase 14 evaluated, though the reference ran on the GPU in strict fp32 and the package on the CPU. The Phase 18 gate's output match holds for v1.
- **Limitations:** HierText validation; the reference ran on the GPU, the package on the CPU.
- **Next experiment:** Phase 20 re-run (hardening on the v1 bundle)

### <a id="gw-e2e-012"></a>GW-E2E-012 — The hardening cases on the v1 bundle (Phase 20 re-run)

- **Date allocated / run:** 2026-11-18 / 2026-11-18
- **Phase:** 20 (v1 re-run)
- **Git commit:** `958f24c` (recorded in the report, clean tree)   **Config:** the v1 bundle   **Seed(s):** the fuzz-lite generator's (`tests/test_hardening.py`)
- **Dataset version(s):** made-up inputs; one HierText validation page for the good-page cases
- **Hardware:** laptop CPU, 10 threads
- **Research question:** do the typed errors hold on the v1 bundle itself, on both backends?
- **Hypothesis (written before the run):** none; the pass rule is Phase 20's (`docs/PHASE_STATUS.md`, Phase 20 v1 re-run, item 2).
- **Method:** `scripts/hardening_check.py` on the bundle with manifest SHA-256 `875357a6…`: fuzz-lite, unusual and wrong inputs (also as bytes), bad files, damaged bundle copies, caching, versions, the CLI and the HTTP API.
- **Variables:** backend (PyTorch, ONNX)
- **Control:** `tests/test_hardening.py` on the tiny models
- **Result:** pass on every check (`reports/hardening/GW-E2E-012.json`, 2026-11-18T10:36:16Z to 10:38:01Z, 10 threads, bundle manifest SHA-256 `875357a6…`). On each backend, PyTorch and ONNX, 269 inputs gave 47 results and 222 `GlypharisError`s and no other exception: the 120 fuzz-lite cases gave 6 results (random arrays) and 114 `invalid_image`; bad files all `invalid_image`; inputs over the pixel limit all `image_too_large`. 16 damaged-bundle cases per backend: every one raised `ModelLoadError` except junk in the two `.onnx` files with the hash re-written, which loads on PyTorch as the amended plan expects. The good page gave 47 lines from a path, bytes, a PIL image and an array. Caching, the load log line, the version output, the CLI (exit code 1, the bad file reported as `invalid_image`) and `POST /ocr` (200 for the good page, 400 `invalid_image` for the bad file) behaved as stated; the bundle's files were unchanged afterwards.
- **Interpretation:** the typed errors hold on the v1 bundle itself, on both backends, not only on the tiny models the tests use. The Phase 20 gate's failure-path and fuzz-lite criteria hold for v1.
- **Limitations:** seeded, small case set; no latency.
- **Next experiment:** Phase 21

### <a id="gw-e2e-013"></a>GW-E2E-013 — Clean-machine-style walkthrough: the installed package on the v1 bundle

- **Date allocated / run:** 2026-11-18 / 2026-11-18
- **Phase:** 22
- **Git commit:** a fresh public clone at `420e095`; fixes recorded in `docs/FINAL_VALIDATION.md`   **Config:** `configs/det/gw-det-001.yaml`, the v1 bundle   **Seed(s):** none
- **Dataset version(s):** HierText, MobileNetV3 weights, FUNSD and IIIT5K downloaded again and checked against the committed manifests; reference `hiertext_val.jsonl` from GW-E2E-006, SHA-256 `c8520576…`
- **Hardware:** this laptop, CPU, 10 threads, in a clean folder outside the worktrees
- **Research question:** can the project be installed, its data fetched and prepared, the v1 bundle loaded and validation pages evaluated from a fresh clone, and which steps need a hand?
- **Hypothesis (written before the run):** none; the pass rule is in `docs/PHASE_STATUS.md`, Phase 22, item 4.
- **Method:** the commands of item 4, then `scripts/package_check.py --v1 --backend torch` on the 400 Phase 5 validation pages; the report goes to `reports/package/GW-E2E-013.json`, and every retry, deviation and hand action to `docs/FINAL_VALIDATION.md`.
- **Variables:** none
- **Control:** GW-E2E-006 (the frozen pipeline on the same pages)
- **Result:** pass (`reports/package/GW-E2E-013.json`, steps in `reports/final_validation/GW-E2E-013/steps.txt`). From a fresh public clone at `420e095`, every downloaded file matched its recorded hash and the prepared HierText and FUNSD manifests were unchanged. The installed package read the 400 Phase 5 validation pages on the PyTorch backend with 24.32% page CER, as GW-E2E-006 did, difference 0.0 points (limit 0.05), and 16,835 of 16,835 lines had identical text and box. Three findings were fixed at `8a4b3a3` and the affected steps repeated: 23 scorer tests depended on the real tag, the IIIT5K manifest entry lacked `index_sha256`, and no document gave the order of work.
- **Interpretation:** the project installs and its public data can be fetched and prepared from the repository alone. The model and the reference file cannot yet: both were placed by hand (`docs/FINAL_VALIDATION.md`, section 1.3).
- **Limitations:** same OS, driver, base Python and git config as the original work; the weights come by hand, so their download path is untested until Phase 23; GDC and GDC-scan cannot be fetched by a stranger.
- **Next experiment:** GW-REC-042 and GW-E2E-014

### <a id="gw-e2e-014"></a>GW-E2E-014 — v1 re-run rehearsal on validation pages

- **Date allocated / run:** 2026-11-18 / 2026-11-18
- **Phase:** 22
- **Git commit:** the `phase14-eval` code, with the wrapper at tag `phase22-repro`   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** HierText validation, 1,715 pages (`data/manifests/processed/hiertext.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32, as Phase 14
- **Research question:** does the frozen pipeline, run through the Phase 22 wrapper, reproduce its own validation run before any test file is read?
- **Hypothesis (written before the run):** none; pass rule in `docs/PHASE_STATUS.md`, Phase 22, item 8: every point value within 0.0005.
- **Method:** `scripts/phase22_v1_repro.py rerun e2e` in a detached `phase14-eval` worktree, folder `GW-E2E-006` under a separate data root; `scripts/phase22_v1_compare.py rehearsal` checks Phase 14's file against its recorded hash, scores the output and compares every point value with Phase 14's entry, `time_ms` ignored, every differing line counted.
- **Variables:** none
- **Control:** GW-E2E-006
- **Result:** pass (`reports/final_validation/rehearsal.json`). All 13 point values, 6 counts and 2 flags equal Phase 14's (largest difference 0.0), no run identity differs, and 1,715 of 1,715 page records are identical outside `time_ms` and `seg`. Run time 799 s.
- **Interpretation:** the frozen pipeline reproduces itself exactly on validation pages through the wrapper.
- **Limitations:** validation only; the GPU driver may differ from Phase 14's, which its run metadata does not record.
- **Next experiment:** GW-E2E-015

### <a id="gw-e2e-015"></a>GW-E2E-015 — Pass A: re-score the saved Phase 14 files

- **Date allocated / run:** 2026-11-18 / 2026-11-18 to 2026-11-19
- **Phase:** 22
- **Git commit:** tag `phase22-repro`, with the scorer unedited   **Config:** `scripts/phase14_score.py` as it stands   **Seed(s):** bootstrap 20261010
- **Dataset version(s):** the Phase 14 prediction files (`data/manifests/processed/phase14_outputs.json`) and the baseline files, in a separate root
- **Hardware:** laptop CPU
- **Research question:** does the unchanged scorer give the numbers of `reports/external/results.json` again from the saved files?
- **Hypothesis (written before the run):** none; Phase 14 item 6 fixed the rule: re-scoring gives identical numbers.
- **Method:** `scripts/phase22_v1_repro.py rescore check`, then `rescore score`, with the official evaluators; every number outside provenance and path strings is compared exactly, and OmniDocBench's fallback page lists separately. Reports go to `reports/final_validation/GW-E2E-015/`.
- **Variables:** none
- **Control:** GW-E2E-009
- **Result:** not identical (`reports/final_validation/GW-E2E-015/verdict.json`). 40 values differ and the comparer explains all of them: they are the official OmniDocBench scores of Tesseract (GW-BASE-001) and GLYPHARIS (GW-E2E-007), where more pages reached the evaluator's quick-match time limit and were scored by its fallback (Tesseract 1 page in Phase 14, 4 now; GLYPHARIS 1, now 2). Every other number is identical, and all 8,062 v1 files still match `reports/final_validation/v1_files_before.json`. Recorded in `research/failures.md`; the owner ruled on 2026-11-19 to continue under the rule unchanged and decide on acceptance at the end. Scoring time 14,976 s.
- **Interpretation:** the scorer and the saved files reproduce Phase 14; only the official OmniDocBench evaluator does not, because its fallback depends on the wall clock.
- **Limitations:** OmniDocBench's wall-clock limit can change which pages fall back; if the number and those pages differ, "identical" is recorded as failed, with the cause. Logged test access, 70 scoring rows; not a new evaluation (EXPERIMENT_PROTOCOL §9).
- **Next experiment:** GW-REC-043, GW-E2E-016 and GW-E2E-017

### <a id="gw-e2e-016"></a>GW-E2E-016 — v1 re-run on the test pages

- **Date allocated / run:** 2026-11-18 / 2026-11-19
- **Phase:** 22
- **Git commit:** the `phase14-eval` code, with the wrapper at tag `phase22-repro`   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** HierText test, GDC, GDC-scan, FUNSD, OmniDocBench (English), each matching its recorded hash
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32, batch 256, as Phase 14
- **Research question:** does a re-run of the frozen pipeline reproduce GW-E2E-007 within the tolerance?
- **Hypothesis (written before the run):** none; the tolerance is Phase 14 item 6's.
- **Method:** `scripts/phase22_v1_repro.py rerun e2e`, Phase 14's order, folder `GW-E2E-007` under a separate data root; one logged access per set; scored by GW-E2E-018.
- **Variables:** none
- **Control:** GW-E2E-007
- **Result:** done in Phase 14's order, 1,943 s in all, one logged access per set; output hashes in `reports/final_validation/v1_rerun_outputs.json`. Every page record equals Phase 14's outside `time_ms` and `seg`: GDC-scan 28, FUNSD 50, GDC 106, OmniDocBench 755, HierText test 1,634 (`reports/final_validation/GW-E2E-018/line_comparison.json`). The scores are GW-E2E-018's.
- **Interpretation:** the frozen pipeline is deterministic on this machine, detector included.
- **Limitations:** does not replace GW-E2E-007; no latency claim; GDC-scan numbers only. Done once per set: a resume only finishes a crashed run, and any further attempt needs a new registered ID.
- **Next experiment:** GW-E2E-018

### <a id="gw-e2e-017"></a>GW-E2E-017 — v1 re-run of the oracle on GDC and GDC-scan

- **Date allocated / run:** 2026-11-18 / 2026-11-19
- **Phase:** 22
- **Git commit:** the `phase14-eval` code, with the wrapper at tag `phase22-repro`   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** GDC v1.1, GDC-scan v1, each matching its recorded hash
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32, batch 256, as Phase 14
- **Research question:** does a re-run of the oracle reproduce GW-E2E-008 within the tolerance?
- **Hypothesis (written before the run):** none; the tolerance is Phase 14 item 6's.
- **Method:** `scripts/phase22_v1_repro.py rerun oracle`, folder `GW-E2E-008` under a separate data root; one logged access per set; scored by GW-E2E-018.
- **Variables:** none
- **Control:** GW-E2E-008
- **Result:** done, 140 s in all, one logged access per set; output hashes in `reports/final_validation/v1_rerun_outputs.json`. Every page record equals Phase 14's outside `time_ms` and `seg`: GDC 106, GDC-scan 28 (`reports/final_validation/GW-E2E-018/line_comparison.json`). The scores are GW-E2E-018's.
- **Interpretation:** the oracle path reproduces exactly.
- **Limitations:** does not replace GW-E2E-008; GDC-scan numbers only. Done once per set: a resume only finishes a crashed run, and any further attempt needs a new registered ID.
- **Next experiment:** GW-E2E-018

### <a id="gw-e2e-018"></a>GW-E2E-018 — Pass B: score the v1 re-run; the tolerance verdict

- **Date allocated / run:** 2026-11-18 / 2026-11-19
- **Phase:** 22
- **Git commit:** tag `phase22-repro`, with the scorer unedited   **Config:** `scripts/phase14_score.py` as it stands   **Seed(s):** bootstrap 20261010
- **Dataset version(s):** the outputs of GW-REC-043, GW-E2E-016 and GW-E2E-017, and the baseline files
- **Hardware:** laptop CPU
- **Research question:** is every aggregate metric of the re-run within 0.05 points of Phase 14's?
- **Hypothesis (written before the run):** none; the tolerance is Phase 14 item 6's. Every point value in the GLYPHARIS blocks, including small slices, calibration and official scores, is judged at 0.0005 as a fraction; counts must be equal; confidence bounds are shown and not judged.
- **Method:** the same scorer on the re-run root with all official evaluators, then `scripts/phase22_v1_compare.py` over `reports/final_validation/metric_inventory.json`. It also counts how many crops and pages differ between the re-run and Phase 14, with no text output, after checking Phase 14's files against their recorded hashes. Reports: `reports/final_validation/GW-E2E-018/`, `reports/final_validation/v1_reproduction.json` and `.md`. The raw scorer files keep the folder IDs, so they carry an `id_map` block and a header saying these are re-run values (GW-REC-039 = GW-REC-043, GW-E2E-007 = GW-E2E-016, GW-E2E-008 = GW-E2E-017); documents cite only `v1_reproduction`.
- **Variables:** none
- **Control:** GW-E2E-009
- **Result:** failed (`reports/final_validation/v1_reproduction.json` and `.md`). Of the 1,381 point values, 1,365 are within 0.0005 (largest 0.00034); all 413 counts, 22 flags and 16 verdicts are unchanged, and no run identity differs. The 16 failures are GLYPHARIS's official OmniDocBench text-block and reading-order scores, the largest 0.0208 on the three-column slice and 0.0013 on the overall reading-order average. The predictions are identical to Phase 14's in all 12 files (`reports/final_validation/GW-E2E-018/line_comparison.json`), and one more page reached the evaluator's quick-match time limit than in Phase 14: the same page as in GW-E2E-015, whose fallback lists Pass B repeats exactly. All 8,062 v1 files still match `reports/final_validation/v1_files_before.json`. Recorded in `research/failures.md`; Phase 14's results stay as recorded. Scoring time 16,004 s.
- **Interpretation:** v1 reproduces line for line and every scorer number but the official OmniDocBench scores is within the tolerance; those depend on the evaluator's wall-clock limit, not on the model. Under the registered rule the verdict fails, and the exit criterion stays unmet until the owner rules at the end of the phase.
- **Limitations:** a failure is reported beside Phase 14's results, which stay as recorded, and leaves the phase open for the owner. The candidate work does not depend on this verdict. Logged access: 70 scoring rows and 12 comparison rows.
- **Next experiment:** GW-EFF-013

### <a id="gw-e2e-019"></a>GW-E2E-019 — Phone candidate: validation gate on pages

- **Date allocated / run:** 2026-11-18 / —
- **Phase:** 22
- **Git commit:** tag `phase22-candidate-eval`   **Config:** the GW-EFF-014 bundle   **Seed(s):** bootstrap 20261010
- **Dataset version(s):** HierText validation, 1,715 pages
- **Hardware:** laptop CPU, 10 threads
- **Research question:** is the candidate non-inferior to v1 on the full validation pages?
- **Hypothesis (written before the run):** the validation gate of [H11](hypotheses.md#h11--the-phone-candidate-is-non-inferior-to-v1-rq7).
- **Method:** `scripts/phase22_eval.py`, then `scripts/phase22_score.py`: H11's bag-of-lines page CER and E2E F1 tests, paired by page. Reports go to `reports/final_validation/candidate/`, the gate's verdict to `validation_gate.json` there.
- **Variables:** build (candidate, v1)
- **Control:** GW-E2E-006
- **Result:** — (planned)
- **Interpretation:** —
- **Limitations:** validation only. Not run if the operator check or the bundle check fails. If a test is not met, no test set is opened, no other arm is tried, and the later IDs and H11 close as `not run`.
- **Next experiment:** GW-E2E-020, if the gate is met

### <a id="gw-e2e-020"></a>GW-E2E-020 — Phone candidate on the test pages

- **Date allocated / run:** 2026-11-18 / —
- **Phase:** 22
- **Git commit:** tag `phase22-candidate-eval`   **Config:** the GW-EFF-014 bundle   **Seed(s):** none
- **Dataset version(s):** HierText test, GDC, GDC-scan, FUNSD, OmniDocBench (English, all 755 pages)
- **Hardware:** laptop CPU, 10 threads
- **Research question:** how does the candidate read the test pages? This is the single prediction pass per set.
- **Hypothesis (written before the run):** [H11](hypotheses.md#h11--the-phone-candidate-is-non-inferior-to-v1-rq7), scored by GW-E2E-021.
- **Method:** `scripts/phase22_eval.py`, which refuses to start without a `go` operator check and a met validation gate for this bundle and tag, and refuses a second pass over a set; one logged access per set before its images are read; predictions only, to `data/processed/phase22/GW-E2E-020/`, with their hashes in `reports/final_validation/candidate/prediction_hashes.json`. No oracle run.
- **Variables:** none
- **Control:** GW-E2E-007
- **Result:** — (planned)
- **Interpretation:** —
- **Limitations:** a new version, not v1. No latency claim comes from this run. GDC-scan numbers only. Not run if the operator check, the bundle check or the validation gate fails.
- **Next experiment:** GW-E2E-021

### <a id="gw-e2e-021"></a>GW-E2E-021 — Phone candidate: the scoring pass and H11

- **Date allocated / run:** 2026-11-18 / —
- **Phase:** 22
- **Git commit:** tag `phase22-candidate-eval`   **Config:** `scripts/phase22_score.py`, reusing the Phase 14 scorer unedited   **Seed(s):** bootstrap 20261010
- **Dataset version(s):** the 10 outputs of GW-REC-045 and GW-E2E-020; v1's GW-REC-039 and GW-E2E-007 on HierText test and GDC (4 files), checked against `phase14_outputs.json`
- **Hardware:** laptop CPU
- **Research question:** is the candidate non-inferior to v1 on HierText test and GDC, and how does it score on every Phase 14 set?
- **Hypothesis (written before the run):** [H11](hypotheses.md#h11--the-phone-candidate-is-non-inferior-to-v1-rq7): six tests, all required, the three on GDC also checked with source documents as the unit.
- **Method:** both tracks, the charset ceiling, the Phase 14 metrics for each set, official HierText and OmniDocBench scores for the candidate only, calibration at T = 2.35 unchanged. The 4 v1 files are scored in a second `Scorer`, so each is logged and none reaches the official evaluators; v1's other figures are copied from `reports/external/results.json`, and its other test files are not read. Work files go to `data/processed/phase22/GW-E2E-021/`, reports to `reports/final_validation/candidate/`.
- **Variables:** build (candidate, v1)
- **Control:** GW-E2E-009
- **Result:** — (planned)
- **Interpretation:** —
- **Limitations:** every set and metric outside H11's six tests is descriptive; the results sit beside v1's and replace none of them; one scoring pass, enforced by its `pass.json`. Logged access: 14 scoring rows (10 candidate, 4 v1).
- **Next experiment:** Phase 22 closure

### <a id="gw-rec-042"></a>GW-REC-042 — v1 re-run rehearsal on validation crops

- **Date allocated / run:** 2026-11-18 / 2026-11-18
- **Phase:** 22
- **Git commit:** the `phase14-eval` code, with the wrapper at tag `phase22-repro`   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** HierText validation evaluation pack, 74,500 crops (`data/manifests/processed/eval_crops.json`)
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32, as Phase 14
- **Research question:** does the frozen recognizer, run through the Phase 22 wrapper, reproduce its own validation run before any test file is read?
- **Hypothesis (written before the run):** none; pass rule in `docs/PHASE_STATUS.md`, Phase 22, item 8: every point value within 0.0005.
- **Method:** `scripts/phase22_v1_repro.py rerun rec` in a detached `phase14-eval` worktree, folder `GW-REC-038` under a separate data root; `scripts/phase22_v1_compare.py rehearsal` checks Phase 14's file against its recorded hash, scores the output and compares every point value with Phase 14's entry, `time_ms` ignored, every differing line counted.
- **Variables:** none
- **Control:** GW-REC-038
- **Result:** pass (`reports/final_validation/rehearsal.json`). All 10 point values and 6 counts equal Phase 14's (largest difference 0.0), no run identity differs, and 74,500 of 74,500 crop records are identical outside `time_ms` and `seg`. The test-access log was unchanged by the rehearsal. Run time 303 s.
- **Interpretation:** the wrapper and the tag worktree reproduce the frozen recognizer exactly on validation data, so the test re-run could start.
- **Limitations:** validation only.
- **Next experiment:** GW-E2E-014

### <a id="gw-rec-043"></a>GW-REC-043 — v1 re-run on the test crops

- **Date allocated / run:** 2026-11-18 / 2026-11-19
- **Phase:** 22
- **Git commit:** the `phase14-eval` code, with the wrapper at tag `phase22-repro`   **Config:** `configs/final/glypharis_v1.yaml`   **Seed(s):** none
- **Dataset version(s):** HierText test, GDC, GDC-scan, FUNSD and IIIT5K crop packs, each matching its recorded hash
- **Hardware:** RTX 4050 Laptop 6 GB, strict fp32, batch 256, as Phase 14
- **Research question:** does a re-run of the frozen recognizer reproduce GW-REC-039 within the tolerance?
- **Hypothesis (written before the run):** none; the tolerance is Phase 14 item 6's.
- **Method:** `scripts/phase22_v1_repro.py rerun rec`, Phase 14's order, folder `GW-REC-039` under a separate data root; one logged access per set; scored by GW-E2E-018.
- **Variables:** none
- **Control:** GW-REC-039
- **Result:** done in Phase 14's order, 477 s in all, one logged access per set; output hashes in `reports/final_validation/v1_rerun_outputs.json`. Every crop record equals Phase 14's outside `time_ms` and `seg`: HierText test 67,574, GDC 8,372, GDC-scan 776, FUNSD 2,647, IIIT5K 3,000 (`reports/final_validation/GW-E2E-018/line_comparison.json`). The scores are GW-E2E-018's.
- **Interpretation:** the frozen recognizer is deterministic on this machine: the re-run reads every test crop as Phase 14 did.
- **Limitations:** does not replace GW-REC-039; GDC-scan numbers only. Done once per set: a resume only finishes a crashed run, and any further attempt needs a new registered ID.
- **Next experiment:** GW-E2E-018

### <a id="gw-rec-044"></a>GW-REC-044 — Phone candidate: validation gate on crops

- **Date allocated / run:** 2026-11-18 / —
- **Phase:** 22
- **Git commit:** tag `phase22-candidate-eval`   **Config:** the GW-EFF-014 bundle   **Seed(s):** bootstrap 20261010
- **Dataset version(s):** HierText validation evaluation pack, 74,500 crops
- **Hardware:** laptop CPU, 10 threads
- **Research question:** is the candidate's recognizer non-inferior to v1's on the full validation crops?
- **Hypothesis (written before the run):** the validation gate of [H11](hypotheses.md#h11--the-phone-candidate-is-non-inferior-to-v1-rq7).
- **Method:** `scripts/phase22_eval.py`, with crops read under the bundle's batching rule, then `scripts/phase22_score.py`: H11's line CER test, paired by line. Reports go to `reports/final_validation/candidate/`, the gate's verdict to `validation_gate.json` there.
- **Variables:** build (candidate, v1)
- **Control:** GW-REC-038
- **Result:** — (planned)
- **Interpretation:** —
- **Limitations:** validation only. Not run if the operator check or the bundle check fails. If the test is not met, no test set is opened, no other arm is tried, and the later IDs and H11 close as `not run`.
- **Next experiment:** GW-REC-045, if the gate is met

### <a id="gw-rec-045"></a>GW-REC-045 — Phone candidate on the test crops

- **Date allocated / run:** 2026-11-18 / —
- **Phase:** 22
- **Git commit:** tag `phase22-candidate-eval`   **Config:** the GW-EFF-014 bundle   **Seed(s):** none
- **Dataset version(s):** HierText test, GDC, GDC-scan, FUNSD and IIIT5K crop packs
- **Hardware:** laptop CPU, 10 threads
- **Research question:** how does the candidate's recognizer read the test crops? This is the single prediction pass per set.
- **Hypothesis (written before the run):** [H11](hypotheses.md#h11--the-phone-candidate-is-non-inferior-to-v1-rq7), scored by GW-E2E-021.
- **Method:** `scripts/phase22_eval.py`, which refuses to start without a `go` operator check and a met validation gate for this bundle and tag, and refuses a second pass over a set; crops read under the bundle's batching rule; one logged access per set before its crops are read; predictions only, to `data/processed/phase22/GW-REC-045/`, with their hashes in `reports/final_validation/candidate/prediction_hashes.json`.
- **Variables:** none
- **Control:** GW-REC-039
- **Result:** — (planned)
- **Interpretation:** —
- **Limitations:** a new version, not v1. No latency claim comes from this run. GDC-scan numbers only. Not run if the operator check, the bundle check or the validation gate fails.
- **Next experiment:** GW-E2E-021

### <a id="gw-eff-013"></a>GW-EFF-013 — Phone candidate selection

- **Date allocated / run:** 2026-11-18 / —
- **Phase:** 22
- **Git commit:** recorded in the report   **Config:** `glypharis/efficiency/candidate.py`, `scripts/phase22_candidate.py` (to be written)   **Seed(s):** Phase 16's line sample; bootstrap 20261010
- **Dataset version(s):** HierText validation: the 400 Phase 5 pages, the 5,000 seeded lines, and the Phase 5 profile pages (10 warm-up, 100 timed)
- **Hardware:** laptop CPU, 10 threads, Phase 16 conditions, then the phone registered under GW-EFF-015
- **Research question:** does fused attention with INT8, or one recognizer run per line, beat the default `fused` build on a phone without losing accuracy or using more memory?
- **Hypothesis (written before the run):** none; the selection rule is written with [H11](hypotheses.md#h11--the-phone-candidate-is-non-inferior-to-v1-rq7).
- **Method:** accuracy of the three arms on the laptop, each reading the 5,000 lines under the batching rule its bundle would ship; the arms that pass go to the phone, in mirrored order, in the sitting of GW-EFF-015, where the speed condition must hold over both runs and on each run alone. The rule leaves exactly one candidate. Reports go to `reports/final_validation/GW-EFF-013/`.
- **Variables:** independent: arm (`fused`, `fused-int8`, `fused-single-line`); controlled: pages, lines, thread count, order, device
- **Control:** the `fused` arm (GW-EFF-009's `fused-attention`)
- **Result:** — (planned)
- **Interpretation:** —
- **Limitations:** phone times are network time only; validation data; a new version, not v1. Not started if the operator check of GW-EFF-015 fails. If `fused` fails the phone's logit and text check, no other arm can be selected and `fused` stays the candidate for the laptop evaluation.
- **Next experiment:** GW-EFF-014

### <a id="gw-eff-014"></a>GW-EFF-014 — Phone candidate bundle and its equivalence check

- **Date allocated / run:** 2026-11-18 / —
- **Phase:** 22
- **Git commit:** recorded in the report   **Config:** format `glypharis-phone-bundle-1`   **Seed(s):** Phase 16's line sample
- **Dataset version(s):** the GW-EFF-006 reference outputs (`reports/efficiency/phase16/reference_manifest.json`)
- **Hardware:** laptop CPU, 10 threads
- **Research question:** is the frozen candidate bundle the selected build, within Phase 17's tolerances?
- **Hypothesis (written before the run):** none; pass rule in `docs/PHASE_STATUS.md`, Phase 22, item 16.
- **Method:** build the bundle (v1's `detector.onnx` byte for byte, the chosen recognizer, `charset.json`, a manifest of hashes and settings, session options included) and check it against the references: map difference at most 1e-4, logits at most 1e-3, text identical except near ties on at most 0.1% of lines, page CER within 0.05 points. Reports go to `reports/final_validation/GW-EFF-014/`.
- **Variables:** none
- **Control:** GW-EFF-006
- **Result:** — (planned)
- **Interpretation:** —
- **Limitations:** if `fused-int8` was selected, the logit and text checks are reported and the selection margin decides. The bundle is a new version, not v1. If a tolerance fails, no other arm is tried, and the later IDs and H11 close as `not run`.
- **Next experiment:** GW-REC-044 and GW-E2E-019

### <a id="gw-eff-015"></a>GW-EFF-015 — Android operator check and phone measurement

- **Date allocated / run:** 2026-11-18 / —
- **Phase:** 22
- **Git commit:** recorded in the report   **Config:** `mobile/android-bench` (to be written), `onnxruntime-android:1.31.1`   **Seed(s):** Phase 16's line sample
- **Dataset version(s):** validation network inputs only: one tensor for part a; 10 plus 100 Phase 5 profile pages and 1,000 seeded lines for part b
- **Hardware:** Samsung Galaxy M52 5G, then Moto G35 5G; each device's SoC, RAM and Android build read from it and registered in `docs/FINAL_VALIDATION.md` before measuring
- **Research question:** a: does the stock Android package run the candidate's `com.microsoft.MultiHeadAttention` nodes? b: what network time and peak memory does the candidate have on a named phone?
- **Hypothesis (written before the run):** none; pass rules in `docs/PHASE_STATUS.md`, Phase 22, items 3 and 15.
- **Method:** part a, before any candidate work: a session on `recognizer.fused.onnx` and one tensor, on each phone. Part b: the laptop's exact inputs replayed under the session options of `ort_session`, each arm and run in a new app process, with mean and p95 network time, peak `VmHWM`, each run's mean time and thermal state at its start and end; logits and greedy text are compared with the laptop's in a separate pass that gives no time or memory figure. On the first phone, every arm that passed selection's accuracy condition; on the second, the selected candidate. Reports go to `reports/final_validation/GW-EFF-015/`.
- **Variables:** device; arm on the first phone; controlled: inputs, package, session options, thread count, spinning, charge state, airplane mode, screen and foreground state, all registered first
- **Control:** the laptop's outputs for the same arm and inputs
- **Part a result (2026-11-18 UTC):** go on both phones (`reports/final_validation/GW-EFF-015/operator_check.json`). On the Samsung Galaxy M52 5G and the Moto G35 5G the stock `onnxruntime-android` 1.31.1 created a session on the fused recognizer, with its 9 `com.microsoft.MultiHeadAttention` nodes, at 4 threads. One validation line gave finite logits of the laptop's shape [72, 1, 275], the largest difference from the laptop over the line's 48 frames was 1.14e-05, and the greedy text was the same. No time or memory figure comes from this check. An earlier accidental run is in `research/failures.md`.
- **Result:** — (planned)
- **Interpretation:** —
- **Limitations:** network time and memory only; decoding, post-processing, cropping and reading order are not ported. Validation data only; no test set and no GDC-scan content goes on a phone. If the operator check fails or no phone can be used, part b is recorded as not run and R-23 stays open. If `fused` fails the phone's logit and text check, part b is recorded as failed and R-23 stays open.
- **Next experiment:** part a: GW-E2E-013; part b: GW-EFF-014

---

## Entry template

```markdown
### <a id="gw-xxx-nnn"></a>GW-XXX-NNN — <short title>

- **Date allocated / run:** YYYY-MM-DD / YYYY-MM-DD
- **Phase:** NN
- **Superseded by:** (optional) <ID of the v1 re-run that replaces this result, and what it replaces>
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
