# Ablation Studies

Controlled comparisons where exactly one major variable changes against a named control.
The planned ablation matrix lives in `docs/IMPLEMENTATION_PLAN.md` (Phases 7, 11, 12, 13).
Results are filled in only from generated reports.

| Ablation | Control ID | Variant ID(s) | Variable changed | Metric (val) | Δ vs control (95% CI) | Seeds | Verdict |
|---|---|---|---|---|---|---|---|
| Input height | GW-REC-003 | GW-REC-006 | height 32 → 48 | CER 7.83% | -0.49 [-0.54, -0.44] | 1 (control: 3) | better |
| CNN width | GW-REC-003 | GW-REC-007 | CNN channels halved | CER 8.83% | +0.51 [+0.47, +0.56] | 1 (control: 3) | worse |
| Encoder (H6) | GW-REC-003 | GW-REC-020 | CRNN → SVTR-style (GLYPHARIS-Rec-M) | CER 8.18% | -0.13 [-0.18, -0.09] | 1 (control: 3) | better |
| BiLSTM size | GW-REC-003 | GW-REC-008 | hidden 256 → 128 | CER 8.78% | +0.46 [+0.42, +0.51] | 1 (control: 3) | worse |
| BiLSTM depth | GW-REC-003 | GW-REC-009 | 2 layers → 1 | CER 8.78% | +0.46 [+0.42, +0.51] | 1 (control: 3) | worse |
| Augmentation | GW-REC-003 | GW-REC-015 | full preset → none | CER 8.68% | +0.36 [+0.32, +0.41] | 1 (control: 3) | worse |
| Augmentation | GW-REC-003 | GW-REC-016 | full preset → geometric only | CER 8.34% | +0.02 [-0.02, +0.06] | 1 (control: 3) | no clear difference |
| Augmentation | GW-REC-003 | GW-REC-017 | full preset → photometric only | CER 8.59% | +0.28 [+0.23, +0.32] | 1 (control: 3) | worse |
| Augmentation | GW-REC-003 | GW-REC-018 | full preset → degradation only | CER 8.68% | +0.36 [+0.32, +0.41] | 1 (control: 3) | worse |
| Optimizer | GW-REC-003 | GW-REC-010 | AdamW → Adam | CER 9.17% | +0.85 [+0.81, +0.90] | 1 (control: 3) | worse |
| LR schedule | GW-REC-003 | GW-REC-011 | OneCycle → cosine with warm-up | CER 8.44% | +0.12 [+0.08, +0.17] | 1 (control: 3) | worse |
| Learning rate | GW-REC-003 | GW-REC-012 | 1e-3 → 5e-4 | CER 8.51% | +0.19 [+0.15, +0.23] | 1 (control: 3) | worse |
| Learning rate | GW-REC-003 | GW-REC-013 | 1e-3 → 2e-3 | CER 8.34% | +0.02 [-0.02, +0.06] | 1 (control: 3) | no clear difference |
| Weight decay | GW-REC-003 | GW-REC-014 | 1e-4 → 1e-2 | CER 8.28% | -0.04 [-0.08, -0.00] | 1 (control: 3) | better |
| Sequence downsampling | GW-REC-003 | GW-REC-021 | frame stride 4 → 8 | CER 10.93% | +2.61 [+2.48, +2.74] | 1 (control: 3) | worse |

**Phase 7 (short budget).** One change at a time against GW-REC-003 (8 epochs, seed 1337), paired
line by line; Δ in percentage points with a 95% bootstrap CI over the 73,888 validation lines. The
decision rule (`docs/PHASE_STATUS.md`, Phase 07) also requires the arm's CER to lie outside all
three control seeds (8.30 to 8.32%). The generated table makes 16 comparisons, with no multiplicity
correction: the arms above and GW-REC-022, which changes several variables at once and so has no row
here. Source: `reports/training/phase07/ablation_table.md`. GW-REC-019 (stride 8 without CTC-aware batching)
failed and was re-run as GW-REC-021 (`research/failures.md`).

**Phase 11 (robustness, CRNN short budget).** Augmentation families against no augmentation (arm A), three seeds per arm (1337, 2024, 7), scored on the frozen GLYPH-C v1 suite. Metric: mean corruption error (mCE), the unweighted mean of the micro CER over 7 corruptions × 5 levels. Δ pools each arm's three seeds, paired over the 6,000 GLYPH-C source lines (1,000 resamples). Decision rule (`docs/PHASE_STATUS.md`, Phase 11): CI on one side of zero and all three seeds beyond all three of A's. Source: `reports/robustness/phase11_comparison.md`.

| Ablation | Control | Variant runs | Variable changed | mCE (GLYPH-C) | Δ vs control (95% CI) | Clean val CER | Seeds | Verdict |
|---|---|---|---|---|---|---|---|---|
| Augmentation family | A, none (10.28%) | GW-REC-016, GW-DATA-003, GW-DATA-004 | none → geometric | 8.61% | −1.67 [−1.73, −1.62] | 8.34% (A 8.66%) | 3 | helps |
| Augmentation family | A, none (10.28%) | GW-REC-017, GW-DATA-005, GW-DATA-006 | none → photometric | 10.23% | −0.06 [−0.08, −0.02] | 8.57% (A 8.66%) | 3 | no clear effect |
| Augmentation family | A, none (10.28%) | GW-REC-018, GW-DATA-007, GW-DATA-008 | none → document degradation | 10.04% | −0.25 [−0.30, −0.19] | 8.68% (A 8.66%) | 3 | helps (H3 supported) |
| Augmentation family | A, none (10.28%) | GW-REC-003, GW-REC-004, GW-REC-005 | none → full (all three) | 8.46% | −1.83 [−1.90, −1.75] | 8.31% (A 8.66%) | 3 | helps |

**Phase 12 (synthetic data, CRNN short budget).** GlyphSynth lines added to HierText, every arm at the HierText-only control's lines per epoch (`data.epoch_lines` 348,935). Metric: CER on the document slice (HierText validation pages with at least 100 lines, 104 pages); arms pool their seeds; Δ paired over pages, 1,000 resamples; Phase 7's rule. Source: `reports/synthetic/phase12_comparison.md`.

| Ablation | Control | Variant runs | Variable changed | Document-slice CER | Δ vs control (95% CI) | Seeds | Verdict |
|---|---|---|---|---|---|---|---|
| Synthetic data | HierText only (11.06%) | GW-DATA-010, GW-DATA-011, GW-DATA-012 | +25,000 GlyphSynth lines | 11.09% | +0.04 [−0.03, +0.11] | 3 (control: 3) | no clear effect |
| Synthetic data | HierText only (11.06%) | GW-DATA-013 | +50,000 | 11.18% | +0.12 [+0.03, +0.22] | 1 (control: 3) | hurts |
| Synthetic data | HierText only (11.06%) | GW-DATA-014, GW-DATA-015, GW-DATA-016 | +100,000 | 11.22% | +0.16 [+0.06, +0.26] | 3 (control: 3) | hurts |
| Synthetic data | HierText only (11.06%) | GW-DATA-017, GW-DATA-018, GW-DATA-019 | +250,000 | 11.44% | +0.39 [+0.25, +0.52] | 3 (control: 3) | hurts |
| Synthetic data | HierText only (11.06%) | GW-DATA-020 | +100,000 from 6 font families | 11.13% | +0.07 [−0.04, +0.18] | 1 (control: 3) | no clear effect |
| Synthetic data | HierText only (11.06%) | GW-DATA-021 | +100,000 without capture effects | 11.09% | +0.04 [−0.07, +0.15] | 1 (control: 3) | no clear effect |

Generator factors at 100K against +100K (three seeds): 6 font families −0.09 [−0.21, +0.03] on the document slice (no clear effect); no capture effects +0.31 [+0.27, +0.35] GLYPH-C mCE points (capture helps).

**Phase 13 (the ablation ladder, short budget).** Each row adds one change to the row it builds on, three seeds per row (1337, 2024, 7), 8 epochs; CRNN up to L3, the SVTR-style model from L4. Metric: HierText validation CER (the selection metric) and the document slice, paired over pages; GLYPH-C mCE, paired over its 6,000 source lines; rows pool their seeds, 1,000 resamples; Phase 7's rule (CI on one side of zero, every seed beyond every parent seed). Rule for keeping a row in `docs/PHASE_STATUS.md`, Phase 13. Source: `reports/ablation/phase13_ladder.md` (GW-REC-037).

| Row | Builds on | Runs | Change | Val CER | Δ val (95% CI) | Document slice, Δ | GLYPH-C mCE, Δ | Kept |
|---|---|---|---|---|---|---|---|---|
| L0 | — | GW-REC-015, GW-DATA-001, GW-DATA-002 | CRNN, height 32, no augmentation | 8.66% | — | 11.46% | 10.28% | start |
| L1 | L0 | GW-REC-003, GW-REC-004, GW-REC-005 | + augmentation (full preset) | 8.31% | −0.36 [−0.40, −0.31] | 11.06%, −0.40 [−0.54, −0.26] | 8.46%, −1.83 [−1.91, −1.75] | yes |
| L2 | L1 | GW-DATA-010, GW-DATA-011, GW-DATA-012 | + 25,000 GlyphSynth lines | 8.29% | −0.02 [−0.05, +0.01] | 11.09%, +0.04 [−0.03, +0.11] | 7.19%, −1.27 [−1.32, −1.21] | no (no clear effect on validation) |
| L3 | L1 | GW-REC-006, GW-REC-025, GW-REC-026 | + input height 48 | 7.83% | −0.48 [−0.54, −0.42] | 10.66%, −0.39 [−0.54, −0.25] | 6.83%, −1.63 [−1.68, −1.58] | yes |
| L4 | L3 | GW-REC-022, GW-REC-027, GW-REC-028 | full model: + SVTR-style encoder, weight decay 1e-2 (and its learning rate and micro-batching) | 7.57% | −0.26 [−0.32, −0.19] | 10.36%, −0.31 [−0.53, −0.11] | 6.29%, −0.54 [−0.59, −0.49] | yes |
| L5 | L4 | GW-REC-029, GW-REC-030, GW-REC-031 | + rotation up to ±4° on half the lines (was ±2° on 30%) | 7.45% | −0.12 [−0.16, −0.08] | 10.27%, −0.09 [−0.17, −0.01] | 4.80%, −1.49 [−1.53, −1.45] | yes |

Every verdict above is "helps" except L2's two "no clear effect" on validation and the document slice. From L0 to L5: −1.21 points of validation CER and −5.48 points of mCE at the short budget.

**Phase 13 (the final recognizer, full budget).** Option B (L4 with L5's rotation, chosen by the ladder), 30 epochs, three seeds, against the CRNN baseline GW-REC-001 at the same budget (one seed), paired over pages. Source: `reports/ablation/phase13_final.md` (GW-REC-037).

| Runs | Val CER per seed | Mean | Δ vs GW-REC-001 (7.53%) | Document slice, Δ | GLYPH-C mCE per seed |
|---|---|---|---|---|---|
| GW-REC-034, GW-REC-035, GW-REC-036 | 6.30, 6.28, 6.30% | 6.29% | −1.24 [−1.40, −1.11] | 9.15%, −1.08 [−1.41, −0.78] | 3.54, 3.53, 3.53% |

**Phase 13 (detector).** GW-DET-007 (labels down to 3 px, rotation up to 30°) against GW-DET-001, both at Phase 9's settings on all 1,715 validation pages; paired page bootstrap. Adopted only if the F1@0.5 difference has a CI above zero. Source: `reports/error_cases/detector/GW-DET-007/pair.json`.

| Ablation | Control ID | Variant ID | Variable changed | F1@0.5 (val) | Δ vs control (95% CI) | Seeds | Verdict |
|---|---|---|---|---|---|---|---|
| Small-text labels and wide rotation | GW-DET-001 (0.7779) | GW-DET-007 | `min_text_size` 6 → 3, `rotate_max_deg` 10 → 30 | 0.7668 | −0.0111 [−0.0146, −0.0075] | 1 | worse; not adopted |

**Phase 8 (detector, full budget).** One change at a time against GW-DET-001 (MobileNetV3-Large,
ImageNet initialisation, 24 epochs, seed 1337), each arm at the operating point stored in its
`best.pt`; paired page bootstrap over all 1,715 HierText validation pages (1,000 resamples),
`scripts/det_compare.py`. Decision rules are the pre-registered hypotheses H9 and H10.

| Ablation | Control ID | Variant ID | Variable changed | F1@0.5 (val) | Δ vs control (95% CI) | Seeds | Verdict |
|---|---|---|---|---|---|---|---|
| Backbone initialisation (H9) | GW-DET-001 (0.724) | GW-DET-002 | ImageNet → random | 0.704 | -0.0197 [-0.0232, -0.0161] | 1 | worse (H9 supported) |
| Backbone size (H10) | GW-DET-001 (0.724) | GW-DET-003 | MobileNetV3-Large → Small | 0.668 | -0.0556 [-0.0616, -0.0505] | 1 | worse, outside the 0.03 margin (H10 not supported) |

Sources: `reports/training/GW-DET-001_vs_GW-DET-002.md`, `reports/training/GW-DET-001_vs_GW-DET-003.md`.

CPU latency of the architectures tried (PyTorch, fp32, batch 1, the baselines' 100 HierText
validation lines; `reports/training/phase07/cpu_latency.md`), against ADR-0011's line target of
25 ms mean and 70 ms p95:

| Run | Model | Height | Frame stride | ms mean | ms p95 |
|---|---|---|---|---|---|
| GW-REC-003 | crnn | 32 | 4 | 10.7 | 22.5 |
| GW-REC-006 | crnn | 48 | 4 | 15.7 | 38.5 |
| GW-REC-007 | crnn | 32 | 4 | 6.3 | 12.4 |
| GW-REC-008 | crnn | 32 | 4 | 8.6 | 18.9 |
| GW-REC-009 | crnn | 32 | 4 | 8.3 | 18.2 |
| GW-REC-020 | svtr | 32 | 4 | 13.5 | 26.6 |
| GW-REC-021 | crnn | 32 | 8 | 7.1 | 15.0 |
