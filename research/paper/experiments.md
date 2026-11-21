# Experiments

This section describes how the experiments were set up and run: the machine, the protocol, the
training runs, the baselines, the ablation design, the external evaluation, the efficiency
measurements, the compute used, what failed, and the re-runs on the v1 bundle. `methodology.md`
describes the system itself and `results.md` gives the results in full. Numbers appear here where
they define an experiment or decided what happened next. The registry of every run is
`research/experiments.md`, and the generated summary of it is `EXPERIMENTS.md`.

The strict text track is the default. Every interval is a 95% bootstrap interval, and the text says
whether lines, pages or documents were resampled. Final validation (Phase 22) has been run; its
record is `docs/FINAL_VALIDATION.md`.

HierText is released under CC BY-SA 4.0; its images come from Open Images, which lists them as
CC BY 2.0 (`long2022hiertext`, `kuznetsova2020openimages`). Tables derived from HierText are shared
alike.

## 1. Setup

<!-- trace: reports/environment/env_report.md reports/environment/env_report.md#system -->
All work ran on one laptop: an Intel Core i7-14650HX with 16 physical cores and 24 logical CPUs, an
NVIDIA GeForce RTX 4050 Laptop GPU with <!-- derived: gpu_memory --> 6140 MiB of memory, and 15.71 GiB of RAM at the time of the
environment report, under Windows 11. There was no cluster and no cloud run. The same machine
trained the models, ran the baselines and took every latency measurement, and it was also in use for
other work during long training runs. For that reason latency was never taken from a training or
accuracy run.

Training and evaluation used Python 3.11 and `torch 2.14.1+cu126`, with bf16 autocast for training.
Evaluation of the frozen model ran in strict fp32. Export and ONNX measurements used
`onnxruntime 1.31.1`. The baselines ran in a separate environment (ADR-0006), so their dependencies
could not change the main one. Package versions are pinned in `requirements/`.

Code: `scripts/env_report.py`. Configuration: `requirements/lock.txt`, `configs/base.yaml`; the
generated record is `reports/environment/env_report.md`.

## 2. Protocol

### 2.1 Order of work and registration

The work ran in numbered phases, each closed by a review gate. Baselines came first (Phase 5), then
the recognizer (Phases 6 and 7), the detector (Phases 8 and 9), the pipeline (Phase 10), the
corruption, synthetic-data and ablation studies (Phases 11 to 13), the external evaluation (Phase
14), error analysis (Phase 15), efficiency (Phase 16), and export, packaging and hardening (Phases 17
to 20). Every experiment received an ID in `research/experiments.md` before it ran, together with its
question, its control and, where it had one, its hypothesis or decision rule. Hypotheses are in
`research/hypotheses.md` and were not edited after their experiment started; a changed hypothesis
was added as a new one, marked with a prime.

Code: `glypharis/training/report.py`, `scripts/experiment_summary.py`. Configuration:
`docs/WORKFLOW.md`, `docs/EXPERIMENT_PROTOCOL.md`.

### 2.2 Comparisons

<!-- trace: configs/rec/gw-rec-003.yaml configs/rec/gw-rec-003.yaml#train/epochs -->
<!-- trace: configs/rec/gw-rec-001.yaml configs/rec/gw-rec-001.yaml#train/epochs -->
<!-- trace: configs/rec/gw-rec-001.yaml configs/rec/gw-rec-001.yaml#train/batch_size -->
Two training budgets were used for the recognizer. The short budget of 8 epochs served the ablations,
and the full budget of 30 epochs the baseline and final models. Compared arms had the same epoch
budget, the same <!-- derived: lines_per_step --> 64 lines per optimizer step and the same schedule unless one of those was the
variable under study. Wall-clock time was recorded but never used for matching, because laptop
thermals make it noisy.

The rule for a recipe comparison was fixed in advance. In Phase 7, a change counted as better or worse
only if the paired interval of the CER difference excluded zero and the arm's CER also lay outside the
range of the control's seeds, in the same direction. Later phases pooled three seeds per arm (seeds
1337, 2024 and 7) and required every seed of the arm to lie beyond every seed of the control. No
correction for multiple comparisons was applied; each table says how many comparisons it holds.

Code: `scripts/rec_ablation_table.py`, `scripts/rec_compare.py`, `glypharis/eval/stats.py`.
Configuration: `docs/EXPERIMENT_PROTOCOL.md` §4 and §5.

### 2.3 Test data

Test sets were read in Phases 5 and 14 only. Phase 5 ran the baselines. Phase 14 ran the frozen model
once per set, from code tagged `phase14-eval` before the first validation or test run of that phase.
Every read wrote a row to `research/test_access_log.md` first. Phase 15 did not read test images: it
worked from the hashed prediction files saved by Phase 14. A model changed after seeing test results
would count as a new version; none was changed.

Code: `glypharis/eval/access.py`, `glypharis/pipeline/frozen.py`, `scripts/ext_eval.py`,
`scripts/phase14_prepare.py`, `scripts/phase14_score.py`. Configuration:
`docs/EXPERIMENT_PROTOCOL.md` §9, `configs/final/glypharis_v1.yaml`.

### 2.4 Confirmatory and descriptive results

A few tests were written down as confirmatory before their data were scored: H1′, H5′ and H8′ in
Phase 14, and H6′, H7′ and the ONNX half of H1′ in Phase 16. Each has a rule that gives "met" if the
interval lies on the claimed side of zero, "not met" if it lies on the other side, and otherwise "not
shown". They are listed in section 6.2. Every other interval in the package is descriptive and carries
no verdict.

Code: `glypharis/eval/stats.py` (`verdict`), `scripts/phase14_score.py`, `scripts/phase16_score.py`.
Configuration: `research/hypotheses.md`.

## 3. Training runs

<!-- trace-table: GW-REC-001 reports/training/GW-REC-001/report.md#best-and-final-epoch GW-REC-001=best (selected) -->
<!-- trace-table: GW-REC-003 reports/training/GW-REC-003/report.md#best-and-final-epoch GW-REC-003=best (selected) -->
<!-- trace-table: GW-REC-006 reports/training/GW-REC-006/report.md#best-and-final-epoch GW-REC-006=best (selected) -->
<!-- trace-table: GW-REC-020 reports/training/GW-REC-020/report.md#best-and-final-epoch GW-REC-020=best (selected) -->
<!-- trace-table: GW-REC-022 reports/training/GW-REC-022/report.md#best-and-final-epoch GW-REC-022=best (selected) -->
<!-- trace-table: GW-REC-024 reports/training/GW-REC-024/report.md#best-and-final-epoch GW-REC-024=best (selected) -->
<!-- trace-table: GW-REC-034 reports/training/GW-REC-034/report.md#best-and-final-epoch GW-REC-034=best (selected) -->
<!-- trace-table: GW-REC-035 reports/training/GW-REC-035/report.md#best-and-final-epoch GW-REC-035=best (selected) -->
<!-- trace-table: GW-REC-036 reports/training/GW-REC-036/report.md#best-and-final-epoch GW-REC-036=best (selected) -->
| Run | Recipe | Budget | Seed | Epoch | Step | Val CER % | Val WER % |
|---|---|---|---|---|---|---|---|
| GW-REC-001 | CRNN, the baseline | full | seed 1337 | 29 | 170,116 | 7.53 | 21.59 |
| GW-REC-003 | CRNN, control of the ablations | short | seed 1337 | 7 | 45,360 | 8.32 | 23.30 |
| GW-REC-006 | CRNN at a larger input height | short | seed 1337 | 7 | 45,339 | 7.83 | 22.17 |
| GW-REC-020 | SVTR-style encoder | short | seed 1337 | 7 | 45,301 | 8.18 | 23.19 |
| GW-REC-022 | SVTR-style encoder, larger height, weight decay | short | seed 1337 | 7 | 45,295 | 7.56 | 22.08 |
| GW-REC-024 | the GW-REC-022 recipe | full | seed 1337 | 29 | 169,860 | 6.46 | 19.65 |
| GW-REC-034 | final recipe with wider rotation (v1) | full | seed 1337 | 29 | 169,860 | 6.30 | 19.37 |
| GW-REC-035 | final recipe | full | seed 2024 | 28 | 164,197 | 6.28 | 19.26 |
| GW-REC-036 | final recipe | full | seed 7 | 29 | 169,860 | 6.30 | 19.24 |

Table 1. The main recognizer runs: the selected checkpoint of each, on the 73,888 validation lines.
All recognizer runs are in the registry. Epochs count from zero.

The seed that would ship was fixed at seed 1337 before the final runs, so no seed was picked for its
result. GW-REC-035 and GW-REC-036 measure the spread of the recipe and are not part of v1.

<!-- trace: GW-DET-001 reports/training/GW-DET-001/report.md#full-validation -->
<!-- trace: GW-DET-001 reports/error_cases/detector/phase09_adopted/eval_full.json#headline/f1_50 -->
The detector GW-DET-001 is described in `methodology.md`, section 4. Its result at the settings of
training, on all 1,715 validation pages (74,500 care lines) at the operating point stored in the
checkpoint, was F1 <!-- derived: det_f1_training --> 0.7238 [0.7075, 0.7390] (by page) at the primary IoU threshold. After the Phase 9
changes of section 5.6 the same network has F1 <!-- derived: det_f1_phase9 --> 0.7779 [0.7621, 0.7919].

Code: `train.py`, `glypharis/training/rec_trainer.py`, `glypharis/training/det_trainer.py`.
Configuration: `configs/rec/`, `configs/det/gw-det-001.yaml`; the run reports are under
`reports/training/`.

### 3.1 Validation figures that differ on purpose

<!-- trace: GW-REC-034 reports/training/GW-REC-034/report.md -->
<!-- trace: GW-REC-038 reports/external/recognition.md -->
<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md -->
<!-- trace: GW-E2E-006 reports/external/end_to_end.md -->
| Figure | Value % | Experiment | Set | Resampled by |
|---|---|---|---|---|
| Line CER, training-time validation | <!-- derived: train_val_cer --> 6.30 [<!-- derived: train_val_cer_low --> 6.17, <!-- derived: train_val_cer_high --> 6.44] | GW-REC-034 | the 73,888 lines of the training validation pack | line |
| Line CER, evaluation pack | <!-- derived: eval_pack_cer --> 6.51 [6.36, 6.65] | GW-REC-038 | <!-- derived: eval_pack_crops --> 74,500 crops, with out-of-charset and long lines kept | line |
| Line CER, efficiency sample | <!-- derived: efficiency_line_cer --> 6.22 [5.73, 6.71] | GW-EFF-009 | <!-- derived: efficiency_lines --> 5,000 seeded lines from the training validation pack | line |
| Page CER, bag of lines, all pages | 27.57 [25.11, 30.36] | GW-E2E-006 | all <!-- derived: val_pages_all --> 1,715 validation pages | page |
| Page CER, bag of lines, Phase 5 pages | <!-- derived: phase5_page_cer --> 24.32 [20.87, 28.15] | GW-EFF-009 | the <!-- derived: baseline_pages --> 400 pages the baselines were run on | page |

Table 2. The same model on different validation sets. Only the evaluation-pack figure, <!-- derived: eval_pack_cer_point --> 6.51, is placed
beside the test results of Phase 14, because it was produced by the same code and precision on crops
cut the same way. The <!-- derived: baseline_pages --> 400-page figure is the one to compare with the Phase 5 baselines.

Code: `scripts/ext_eval.py`, `scripts/phase14_score.py`, `scripts/phase16_score.py`. Configuration:
`configs/final/glypharis_v1.yaml`, `data/manifests/processed/eval_crops.json`.

## 4. Baselines

### 4.1 Systems

Seven systems were registered before they ran. They used default settings plus the documented choice
of language and model, with no tuning for any benchmark, and each was run through the same harness
and scored by the same code as GLYPHARIS.

| ID | System | Model | Runtime | Tasks |
|---|---|---|---|---|
| GW-BASE-001 | Tesseract 5 (`smith2007tesseract`) | LSTM engine, English | command line, `tesseract v5.4.0` | recognition, end to end |
| GW-BASE-002 | PP-OCRv5 mobile (`cui2025paddleocr`) | mobile detector, English mobile recognizer | `paddleocr 3.7.0`, CPU | recognition, end to end |
| GW-BASE-003 | PP-OCRv5 server (`cui2025paddleocr`) | server detector and recognizer | `paddleocr 3.7.0`, CPU | reduced scope (section 4.2) |
| GW-BASE-004 | TrOCR-small-printed (`li2023trocr`) | small printed-text model | `transformers 5.20.2`, CPU | recognition only (no detector) |
| GW-BASE-005 | PP-OCRv6 tiny (`paddleocr_ppocrv6`) | tiny detector and recognizer | `paddleocr 3.7.0`, CPU | recognition, end to end |
| GW-BASE-006 | PP-OCRv6 small (`paddleocr_ppocrv6`) | small detector and recognizer | `paddleocr 3.7.0`, CPU | recognition, end to end |
| GW-BASE-007 | PP-OCRv6 medium (`paddleocr_ppocrv6`) | medium detector and recognizer | `paddleocr 3.7.0`, CPU | reduced scope (section 4.2) |

Table 3. The baseline systems as run (`reports/baselines/README.md`).

The PaddleOCR pipelines kept their document-orientation and text-line-orientation models on, and
document unwarping was turned off, because it moves the polygons. These orientation models are part
of what the baselines' accuracy and latency include. TrOCR was given a larger limit on generated
tokens than its default, which would stop at short lines.

Code: `scripts/run_baselines.py`, `scripts/evaluate_baselines.py`. Configuration:
`data/manifests/baseline_models.json`, `requirements/`, `docs/EXPERIMENT_PROTOCOL.md` §10.

### 4.2 Sets and coverage

<!-- trace: GW-BASE-001 reports/baselines/README.md#coverage -->
<!-- trace: GW-REC-038 reports/external/recognition.md#track-strict -->
<!-- trace: GW-E2E-006 reports/external/end_to_end.md#detection -->
In Phase 5 the baselines read a seeded subsample of HierText validation (10,000 line crops and 400
pages), all 10,000 GlyphSynth validation lines, and both test sets in full: HierText test (<!-- derived: base_test_crops --> 67,574
crops, <!-- derived: base_test_pages --> 1,634 pages) and GDC (<!-- derived: base_gdc_crops --> 8,372 crops, <!-- derived: base_gdc_pages --> 106 pages). On validation they therefore cover fewer lines
and pages than the GLYPHARIS figures in the broad tables, which use <!-- derived: eval_pack_crops --> 74,500 crops and <!-- derived: val_pages_all --> 1,715 pages.
Where a baseline stands beside GLYPHARIS on validation, the table names the set.

PP-OCRv5 server and PP-OCRv6 medium ran at a scope reduced before the runs, because on this CPU their
recognizers need between about one and three seconds for a long line. PP-OCRv5 server could not run
the GDC pages on the laptop (section 9). TrOCR has no detector, so it has no end-to-end rows.

In Phase 14 the same runner added IIIT5K, FUNSD, OmniDocBench and GDC-scan. Predictions on HierText
test and GDC were not produced again. The Phase 5 files were re-scored once, and the scorer stopped
unless the re-scored totals equalled the Phase 5 report exactly. Phase 14 kept the Phase 5 rule that a
system which cannot run a set gets a missing cell with its reason and no substitute.

Code: `scripts/build_eval_crops.py`, `glypharis/eval/sets.py`, `scripts/phase14_score.py`.
Configuration: `data/manifests/processed/phase5_baseline_files.json`,
`data/manifests/processed/eval_crops.json`.

### 4.3 Latency sittings

Latency differs between sittings on this machine, so a difference is stated only between two systems
timed in one sitting, one after the other, at the same thread count.

| System | Phase 5 sitting, ms per page | Phase 14 sitting, ms per page | Phase 16 sitting, ms per page |
|---|---|---|---|
| PP-OCRv5 mobile (GW-BASE-002) | <!-- derived: base2_page_phase5 --> 3,408.2 | <!-- derived: base2_page_phase14 --> 2,954.8 | <!-- derived: base2_page_phase16 --> 3,086.0 |
| GLYPHARIS v1, PyTorch (GW-EFF-005, GW-EFF-007) | not measured | <!-- derived: v1_page_phase14 --> 2,446.7 | <!-- derived: v1_page_phase16 --> 2,565.7 |

Table 4. Mean CPU time per page on HierText validation profile pages. No difference is stated across
columns. In the Phase 14 column the two systems were timed in one sitting (GW-EFF-005). The two values
of the Phase 16 column come from different runs (GW-EFF-010 and GW-EFF-007), and GW-EFF-010 compared
PP-OCRv5 mobile with the ONNX build, so no difference is stated between them either.

Code: `scripts/run_baselines.py` (`--profile`), `scripts/e2e_profile.py`. Configuration:
`docs/EXPERIMENT_PROTOCOL.md` §3.

## 5. Ablation design

### 5.1 What the study had to cover

The project's requirement for the ablation study names six steps: a baseline, then augmentation,
synthetic data, detector improvements, recognizer improvements and preprocessing, changing one major
variable at a time where a causal reading is wanted. They were covered as follows.

| Step | Where it was tested | Experiments |
|---|---|---|
| Baseline | Ladder row L0: CRNN, no augmentation | GW-REC-015, GW-DATA-001, GW-DATA-002 |
| Augmentation | Ladder row L1; families in Phases 7 and 11 | GW-REC-003 to GW-REC-005, GW-REC-015 to GW-REC-018, GW-DATA-003 to GW-DATA-008 |
| Synthetic data | Ladder row L2; sizes and generator factors in Phase 12 | GW-DATA-010 to GW-DATA-022 |
| Preprocessing | Ladder row L3: input height | GW-REC-006, GW-REC-025, GW-REC-026 |
| Recognizer improvements | Ladder rows L4 and L5; single changes in Phase 7 | GW-REC-006 to GW-REC-014, GW-REC-020 to GW-REC-022, GW-REC-027 to GW-REC-031 |
| Detector improvements | Outside the ladder (section 5.6) | GW-DET-002, GW-DET-003, GW-DET-004, GW-DET-005, GW-DET-007 |

Table 5. How the required steps map to experiments.

The Phase 13 ladder is about the recognizer only. It cannot hold the detector steps, since a detector
change does not alter a recognizer's line CER, so those were run as their own paired comparisons on
pages. The preprocessing row of the ladder is the input height, the one preprocessing choice that was
varied for the recognizer.

Code: `scripts/ablation_ladder.py`. Configuration: `research/ablations.md` holds the tables, and
`docs/PHASE_STATUS.md` (Phase 13) the rules.

### 5.2 Phase 7: single changes against a control

<!-- trace: GW-REC-003 reports/training/phase07/ablation_table.md#control-and-seed-spread -->
The control was GW-REC-001's configuration at the short budget, trained at three seeds (GW-REC-003 to
GW-REC-005). Their validation CER ranged from <!-- derived: control_cer_low --> 8.30 to <!-- derived: control_cer_high --> 8.32%, and that range was the noise floor each
arm was read against.

<!-- trace: GW-REC-003 reports/training/phase07/ablation_table.md#arms-against-gw-rec-003 -->
Each arm changed one setting from the control at seed 1337: input height, CNN width, the encoder,
LSTM size or depth, the augmentation family, optimizer, schedule, learning rate, weight decay, or the
frame stride. The table holds 16 comparisons, paired over the validation lines, without correction
for multiplicity. Arms that needed a different batch configuration to fit the GPU kept the
control's lines per optimizer step through smaller micro-batches and gradient accumulation.

<!-- trace: configs/rec/gw-rec-022.yaml configs/rec/gw-rec-022.yaml#data/image_height -->
<!-- trace: configs/rec/gw-rec-022.yaml configs/rec/gw-rec-022.yaml#train/weight_decay -->
The combination run GW-REC-022 is not a single change. It takes the SVTR-style encoder at image height
48 with weight decay 1e-2, as the Phase 7 decision rule selected them.

<!-- trace: GW-REC-022 reports/training/compare/GW-REC-022_vs_GW-REC-006.md -->
Its gain over GW-REC-006, the CRNN at the same height, is -0.27 [-0.32, -0.22] points of CER (by
line), but GW-REC-022 also differs from GW-REC-006 in weight decay, learning rate and micro-batching.
The gain therefore belongs to the combination and is not credited to the encoder alone. The run that
would have isolated the encoder's share, GW-REC-023, was abandoned before it started (section 9).

Code: `scripts/rec_ablation_table.py`, `scripts/rec_compare.py`, `train.py`. Configuration:
`configs/rec/gw-rec-003.yaml` to `configs/rec/gw-rec-022.yaml`.

### 5.3 Phase 11: augmentation families on corrupted lines

<!-- trace: GW-DATA-009 reports/robustness/phase11_comparison.md -->
Five arms were trained at the short budget, each with three seeds: no augmentation, geometric only,
photometric only, document degradation only, and all three. They were scored on the frozen GLYPH-C
suite. The differences in mCE against the arm without augmentation pool each arm's seeds and resample
the 6,000 source lines. The decision rule was the one of section 2.2.

Code: `scripts/glyph_c_eval.py`. Configuration: `configs/rec/gw-data-001.yaml` to
`configs/rec/gw-data-008.yaml`.

### 5.4 Phase 12: synthetic data

<!-- trace: GW-DATA-022 reports/synthetic/phase12_comparison.md -->
GlyphSynth lines were added to HierText in four arms: +25K (GW-DATA-010 to GW-DATA-012), <!-- derived: synth_arm_50k --> +50K
(GW-DATA-013), +100K (GW-DATA-014 to GW-DATA-016) and +250K (GW-DATA-017 to GW-DATA-019). Every arm was
kept at the HierText-only control's lines per epoch, so that added lines replace HierText lines in each
epoch rather than lengthen it. Two more arms changed one generator factor each: +100K, 6 fonts
(GW-DATA-020), against the 85 families of the standard generator, and +100K, no capture (GW-DATA-021).
The primary measure was the CER on the document slice, the <!-- derived: doc_slice_pages --> 104 HierText validation pages with at least
100 validation lines, paired over pages.

Code: `scripts/synth_study.py`, `scripts/build_glyphsynth_variants.py`. Configuration:
`configs/rec/gw-data-010.yaml` to `configs/rec/gw-data-021.yaml`,
`data/manifests/processed/glyphsynth_variants.json`.

### 5.5 Phase 13: the ladder

Each row of the ladder adds one change to the row it builds on, with three seeds per row at the short
budget. A row was kept if it helped on HierText validation CER under the rule of section 2.2, paired
over pages. L5 had its own rule: kept if it helped on GLYPH-C mCE and did not hurt validation CER. The
document slice and GLYPH-C mCE are reported for every row.

<!-- trace-table: GW-REC-037 reports/ablation/phase13_ladder.md L0;none=L0 baseline (CRNN\, no augmentation)|L1;L0=L1 + augmentation (full preset)|L2;L1=L2 + 25K GlyphSynth lines|L3;L1=L3 + input height 48|L4;L3=L4 full model (+ SVTR encoder\, weight decay 1e-2)|L5;L4=L5 + wider rotation -->
| Row | Change | Builds on | Val CER % | Δ val vs parent | Document slice % | Δ doc | GLYPH-C mCE % | Δ mCE | Kept |
|---|---|---|---|---|---|---|---|---|---|
| L0 | baseline: CRNN, no augmentation | none | 8.66 | — | 11.46 | — | 10.28 | — | start |
| L1 | + augmentation, full preset | L0 | 8.31 | -0.36 [-0.40, -0.31] | 11.06 | -0.40 [-0.54, -0.26] | 8.46 | -1.83 [-1.91, -1.75] | yes |
| L2 | + 25K GlyphSynth lines | L1 | 8.29 | -0.02 [-0.05, +0.01] | 11.09 | +0.04 [-0.03, +0.11] | 7.19 | -1.27 [-1.32, -1.21] | no |
| L3 | + input height 48 | L1 | 7.83 | -0.48 [-0.54, -0.42] | 10.66 | -0.39 [-0.54, -0.25] | 6.83 | -1.63 [-1.68, -1.58] | yes |
| L4 | full model: SVTR encoder, weight decay 1e-2 | L3 | 7.57 | -0.26 [-0.32, -0.19] | 10.36 | -0.31 [-0.53, -0.11] | 6.29 | -0.54 [-0.59, -0.49] | yes |
| L5 | + wider rotation | L4 | 7.45 | -0.12 [-0.16, -0.08] | 10.27 | -0.09 [-0.17, -0.01] | 4.80 | -1.49 [-1.53, -1.45] | yes |

Table 6. The ablation ladder (GW-REC-037), CER in percent on the strict track, three seeds per row.
Differences are against the row in "Builds on", with 95% intervals; validation CER and the document
slice are paired over pages, and mCE over the GLYPH-C source lines.

Three limits belong with this table. The ladder ran at the short budget, and only the final recipe was
trained at full length. L3 builds on L1 and not on L2, because L2 was not kept. And L4 bundles the
encoder with weight decay, learning rate and micro-batching, so it does not measure the encoder alone.
Rows L0 to L2 and the seed-1337 runs of L3 and L4 were trained in earlier phases, so their outcomes
were partly known when the rules were written.

<!-- trace: GW-REC-037 reports/ablation/phase13_final.md -->
The chosen recipe was trained at the full budget with three seeds (GW-REC-034 to GW-REC-036). Their
validation CER is 6.30, 6.28 and 6.30%, mean 6.29%, against the CRNN baseline GW-REC-001 at the same
budget: <!-- derived: final_vs_baseline --> -1.24 [-1.40, -1.11] points, paired over pages. The baseline has one seed, so this interval
covers page sampling and not its seed variation.

Code: `scripts/ablation_ladder.py`, `scripts/freeze_final.py`. Configuration:
`configs/rec/gw-rec-025.yaml` to `configs/rec/gw-rec-036.yaml`, `configs/final/glypharis_v1.yaml`.

### 5.6 Detector steps

Each step was compared with its control on validation pages by a paired page bootstrap. GW-DET-004
and GW-DET-005 were chosen on half A of the validation pages and checked on half B.

<!-- trace: GW-DET-002 reports/training/GW-DET-001_vs_GW-DET-002.md -->
<!-- trace: GW-DET-003 reports/training/GW-DET-001_vs_GW-DET-003.md -->
<!-- trace: GW-DET-004 reports/error_cases/detector/GW-DET-004/pair_B.json -->
<!-- trace: GW-DET-005 reports/error_cases/detector/GW-DET-005/pair_B.json -->
<!-- trace: GW-DET-007 reports/error_cases/detector/GW-DET-007/pair.json -->
<!-- trace: GW-DET-005 reports/error_cases/detector/GW-DET-005/resolution_A.md -->
| Experiment | Change | Pages | Difference in F1 (95% interval, by page) | What followed |
|---|---|---|---|---|
| GW-DET-002 | random backbone initialisation instead of ImageNet | all validation pages | GW-DET-001 minus this run: <!-- derived: det_002_diff --> +0.0197 [+0.0161, +0.0232] | ImageNet kept |
| GW-DET-003 | MobileNetV3-Small instead of Large | all validation pages | GW-DET-001 minus this run: <!-- derived: det_003_diff --> +0.0556 [+0.0505, +0.0616] | Large kept |
| GW-DET-004 | post-processing chosen on half A | half B | this setting minus the setting of record: <!-- derived: det_004_diff --> +0.0316 [+0.0247, +0.0386] | adopted |
| GW-DET-005 | long side 1536 instead of 1280 | half B | <!-- derived: det_005_diff --> +0.0177 [+0.0131, +0.0221] | adopted |
| GW-DET-007 | labels for smaller text, wider training rotation | all validation pages | this run minus GW-DET-001: <!-- derived: det_007_diff --> -0.0111 [-0.0146, -0.0075] | not adopted |

Table 7. The detector steps, with F1 at the primary IoU threshold. GW-DET-002 and GW-DET-003 are
measured at the settings of training, the others at the Phase 9 settings as they became available.

All detector runs have one seed each. GW-DET-004's post-processing was tuned on GW-DET-001, which
favours that network slightly in the GW-DET-007 comparison.

Code: `scripts/det_compare.py`, `scripts/det_sweep.py`, `scripts/det_choose.py`, `scripts/det_pair.py`.
Configuration: `configs/det/gw-det-001.yaml`, `configs/det/gw-det-002.yaml`,
`configs/det/gw-det-003.yaml`, `configs/det/gw-det-007.yaml`.

## 6. External evaluation

### 6.1 Design

The frozen pipeline was evaluated once per set in Phase 14: the recognizer on ground-truth crops
(GW-REC-039) and the pipeline on pages (GW-E2E-007), with an oracle run on GDC and GDC-scan
(GW-E2E-008). Validation references came first (GW-REC-038 and GW-E2E-006). One scoring pass
(GW-E2E-009) then computed every metric, the paired comparisons with the baselines and the verdicts.
The model ran on the GPU in strict fp32, with deterministic settings. The latency comparison
(GW-EFF-005) ran on the CPU on validation pages. Phase 15 then analysed errors from the saved
predictions without reading any test image.

Phase 14 used one model, the seed fixed before training, as the frozen system, so the intervals leave
out training variance. The seed spread of the recipe on validation is the one in Table 1.

Code: `scripts/ext_eval.py`, `scripts/phase14_score.py`, `scripts/phase15_errors.py`. Configuration:
`configs/final/glypharis_v1.yaml`, `docs/EXPERIMENT_PROTOCOL.md` §11 (entries of 2026-11-15 to
2026-11-17).

### 6.2 Confirmatory tests

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests-phase-14-plan-item-5 -->
<!-- trace: GW-E2E-009 reports/external/calibration.md -->
<!-- trace: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-latency-met -->
<!-- trace: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-accuracy-met -->
<!-- trace: GW-EFF-011 reports/efficiency/phase16/verdicts.md#h6-not-met -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx-not-met -->
| Test | Quantity whose interval decides | Experiment | Result (95% interval) | Verdict |
|---|---|---|---|---|
| H1′, CER half | page CER of GLYPHARIS minus 2 x PP-OCRv5 mobile, 53 clean GDC pages, by page | GW-E2E-009 | <!-- derived: h1_cer --> -34.98 [<!-- derived: h1_cer_low --> -47.44, <!-- derived: h1_cer_high --> -24.08] points | met |
| H1′, latency half | GLYPHARIS (PyTorch) minus PP-OCRv5 mobile, ms per page, one sitting, by page | GW-EFF-005 | <!-- derived: h1_latency --> -508.11 [-737.66, -256.52] ms | met |
| H1′, ONNX build | v1 fp32 ONNX minus PP-OCRv5 mobile, ms per page, one sitting | GW-EFF-010 | <!-- derived: h1_onnx --> +6,537.6 [+5,383.7, +7,806.3] ms | not met |
| H5′ | CER on GDC crops minus 1.5 x CER on validation crops, by line | GW-E2E-009 | <!-- derived: h5 --> -5.59 [<!-- derived: h5_low --> -6.07, <!-- derived: h5_high --> -5.01] points | not met |
| H6′ | SVTR minus 1.5 x CRNN, ONNX, ms per line | GW-EFF-011 | <!-- derived: h6 --> +16.78 [+12.02, +22.04] ms | not met |
| H7′, latency half | INT8 minus 0.75 x fp32, ms per line | GW-EFF-008 | <!-- derived: h7_latency --> -10.00 [-12.79, -7.37] ms | met |
| H7′, accuracy half | line CER of INT8 minus fp32, with an allowance of 0.005 errors per reference character | GW-EFF-008 | <!-- derived: h7_accuracy --> -0.512 [<!-- derived: h7_accuracy_low --> -0.541, <!-- derived: h7_accuracy_high --> -0.484] points | met |
| H8′, transfers | ECE at T = 2.35 minus ECE at T = 1, GDC crops, by line | GW-E2E-009 | <!-- derived: h8_transfers --> -0.1253 [-0.1319, -0.1190] | met |

Table 8. The confirmatory tests. H1′ holds as a whole on the PyTorch build, with both of its halves
met; the verdict on the ONNX build is separate and applies to that bundle only. H7′ holds. H7′ concerns
an INT8 recognizer, which is a new version and not v1. H1′'s CER half was also met on the normalised
track and with a source-document bootstrap. H5′ was also not met with a source-document bootstrap.

H8′ had a second part: that the temperature removes at least half of the ECE on GDC. It was not shown,
because the line and page bootstraps gave different answers, so H8 is not supported on GDC. The figures
are in `results.md`.

Code: `scripts/phase14_score.py`, `scripts/phase16_score.py`, `glypharis/eval/calib_stats.py`.
Configuration: `research/hypotheses.md`.

### 6.3 Results that went against GLYPHARIS

The descriptive comparisons are reported whichever way they fall, and the full tables are in
`results.md`. The cases below are those where a baseline read better than v1, or where the interval
does not separate them.

<!-- trace-table: GW-E2E-009 reports/external/comparisons.md GDC crops\, PP-OCRv5 mobile=GW-REC-039;GW-BASE-002;gdc_test;strict|GDC crops\, PP-OCRv6 tiny=GW-REC-039;GW-BASE-005;gdc_test;strict|GDC crops\, PP-OCRv6 small=GW-REC-039;GW-BASE-006;gdc_test;strict|GDC crops\, PP-OCRv6 medium=GW-REC-039;GW-BASE-007;gdc_test;strict|IIIT5K\, PP-OCRv5 mobile=GW-REC-039;GW-BASE-002;iiit5k_test;alnum|IIIT5K\, PP-OCRv6 small=GW-REC-039;GW-BASE-006;iiit5k_test;alnum|IIIT5K\, PP-OCRv6 medium=GW-REC-039;GW-BASE-007;iiit5k_test;alnum|OmniDocBench\, Tesseract 5=GW-E2E-007;GW-BASE-001;omnidocbench_test;strict|FUNSD pages\, PP-OCRv6 small=GW-E2E-007;GW-BASE-006;funsd_test;strict -->
| Comparison | Track | CER (crops or bag of lines) pp | Order-aware CER / word acc pp |
|---|---|---|---|
| GDC crops, PP-OCRv5 mobile | strict | +0.49 [+0.23, +0.78] | n/a |
| GDC crops, PP-OCRv6 tiny | strict | +0.18 [-0.04, +0.40] | n/a |
| GDC crops, PP-OCRv6 small | strict | +0.71 [+0.53, +0.91] | n/a |
| GDC crops, PP-OCRv6 medium | strict | +0.97 [+0.76, +1.20] | n/a |
| IIIT5K, PP-OCRv5 mobile | alphanumeric | +1.15 [+0.88, +1.45] | -3.47 [-4.43, -2.57] |
| IIIT5K, PP-OCRv6 small | alphanumeric | +1.55 [+1.28, +1.85] | -4.97 [-5.87, -4.07] |
| IIIT5K, PP-OCRv6 medium | alphanumeric | +1.70 [+1.43, +1.98] | -5.67 [-6.53, -4.77] |
| OmniDocBench, Tesseract 5 | strict | +11.58 [+8.37, +14.88] | +11.58 [+8.37, +14.88] |
| FUNSD pages, PP-OCRv6 small | strict | +2.55 [-0.98, +6.81] | n/a |

Table 9. GLYPHARIS v1 minus the baseline in percentage points, paired over the same lines, words or
pages (GW-E2E-009). For CER a positive value means v1 made more errors. The last column is word
accuracy on IIIT5K, where a negative value means v1 read fewer words, and order-aware page CER on
OmniDocBench. Intervals that include zero (the PP-OCRv6 tiny and FUNSD rows) state no difference.

<!-- trace: GW-E2E-007 reports/external/end_to_end.md#detection -->
On FUNSD, detection by GLYPHARIS v1 has F1 79.07 [75.56, 82.16] (by page) and PP-OCRv6 small has
83.93 [81.08, 86.76]. No paired difference was computed for detection, so no difference is stated.

<!-- trace: GW-E2E-009 reports/external/official.md#omnidocbench-text-block-and-reading-order-edit_dist-quick_match -->
On OmniDocBench, the official text-block edit distance over the whole set is 0.1859 for GLYPHARIS v1
and 0.0847 for Tesseract 5 (lower is better).

<!-- trace: GW-E2E-018 reports/final_validation/v1_reproduction.md -->
These official OmniDocBench scores did not reproduce exactly in Phase 22. A re-run of v1 gave the
same predictions, but the strict reproduction verdict, which judged GLYPHARIS v1's values only, found
16 failures among the 1,381 point values checked, all of them GLYPHARIS's official OmniDocBench
scores, because the evaluator's per-page time limit depends on machine speed (GW-E2E-018,
`reports/final_validation/v1_reproduction.md`). Re-scoring the same saved Phase 14 files (GW-E2E-015,
`reports/final_validation/GW-E2E-015/verdict.json`) also changed Tesseract 5's official OmniDocBench
scores, through the same time-limit fallback, so the official GLYPHARIS-versus-Tesseract comparison
and ranking on these scores carry the same limitation. The owner accepted this as a disclosed
limitation of the official evaluator; Phase 14's scores stay as recorded.

Code: `scripts/phase14_score.py`, `scripts/official_omnidocbench.py`. Configuration:
`data/manifests/evaluators.json`.

## 7. Efficiency measurements

### 7.1 Rules

<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#detector/pages -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#recognizer/lines -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#end_to_end/pages -->
Phase 16 measured the frozen v1 and its variants on HierText validation only. Latency used the 110
profile pages, the first 10 as warm-up and 100 timed, and the same split for lines. Accuracy used the
400 pages of Phase 5 and 5,000 seeded validation lines, read on the CPU for CPU builds. Every profile
ran at 10 threads unless stated, one process per run, on mains power under the Balanced plan, with the
power mode and the state of the CPU and GPU recorded at its start and end. When arms were compared in
one sitting they ran in one order and then in the reverse order, and each item's time is the mean of
its two runs. The exception is GW-EFF-010, which followed Phase 14's order: the baseline first and the
bundle within two hours.

Each variant is a new version. It is measured on validation, and none replaces v1.

| Experiment | What it measured |
|---|---|
| GW-EFF-006 | reference outputs of PyTorch v1 on the CPU, and the fp32 ONNX bundle built from the freeze record and checked against them |
| GW-EFF-007 | parameters, sizes, line and page latency, throughput, memory and page time by stage |
| GW-EFF-008 | dynamic INT8 of the ONNX recognizer (H7′) |
| GW-EFF-009 | accuracy against latency for input sizes, INT8, mixed backends, fused attention and other model sizes |
| GW-EFF-010 | the ONNX half of H1′, with a new GW-BASE-002 profile in one sitting |
| GW-EFF-011 | SVTR against CRNN in ONNX (H6′) |

Table 10. The Phase 16 experiments. The results are in `efficiency.md`.

Code: `scripts/phase16_reference.py`, `scripts/phase16_variants.py`, `scripts/phase16_score.py`,
`scripts/e2e_profile.py`, `scripts/rec_profile.py`, `glypharis/efficiency/`. Configuration:
`configs/final/glypharis_v1.yaml`, `docs/EXPERIMENT_PROTOCOL.md` §3 and §11.

### 7.2 The laptop targets and the owner's decisions

ADR-0011 set targets for this laptop's CPU, after the Phase 5 baselines were measured.

<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/README.md v1 PyTorch\, CPU\, 10 threads=v1 PyTorch\, CPU\, 10 threads|v1 fp32 ONNX\, CPU\, 10 threads=v1 fp32 ONNX\, CPU\, 10 threads -->
| Build | Line mean, ms | Line p95, ms | Page mean, ms | Page p95, ms | Peak RAM on pages, MiB |
|---|---|---|---|---|---|
| Target of ADR-0011 | 25 | 70 | 1.5 s | 2.7 s | 1,836 |
| v1 PyTorch, CPU, 10 threads | 33.9 | 77.6 | 2,565.7 | 5,127.1 | 1,641.8 |
| v1 fp32 ONNX, CPU, 10 threads | 92.2 | 304.4 | 8,573.7 | 19,011.5 | 6,529.6 |

Table 11. ADR-0011's targets against the two main v1 builds (GW-EFF-007). Every target is an upper
limit.

v1 misses the line and page targets on this laptop in both builds; the PyTorch build is within the
memory limit. The project owner decided on 2026-11-18 that v1 keeps PyTorch as its default backend and
that the targets are recorded as missed, not restated. The owner also set the target hardware to
lightweight devices such as phones. Phones are the target hardware. Phase 22 measured the phone
candidate's networks on two phones (GW-EFF-015): network time and memory only, without decoding,
post-processing, cropping or reading order; v1 itself was not measured on a phone. The laptop
numbers only rank candidate builds. The candidate for phones is the all-ONNX build with fused
attention, reading one line per recognizer run (`fused-single-line`, GW-EFF-013). A new version, not
v1. Phase 22 evaluated it once per test set, and it met H11's six non-inferiority tests against v1
(GW-E2E-021, `reports/final_validation/candidate/h11.md`); its results sit beside v1's and replace
none of them.

Code: `scripts/phase16_score.py`, `glypharis/efficiency/fused_attention.py`. Configuration:
`docs/decisions/ADR-0011-cpu-latency-target.md`.

## 8. Compute

This statement is limited to `reports/compute/ledger.md`, which is generated from the queue logs. It
counts the wall-clock hours each queued job occupied its device, whether it finished or not. A job
that did not finish is not all waste, since a training job that stopped resumed from its last
checkpoint.

<!-- trace-table: reports/compute/ledger.md reports/compute/ledger.md Phase 5=L13|Phase 6=L14|Phase 7=L15|Phase 8=L16|Phase 11=L17|Phase 12=L18|Phase 13=L19|Phase 16=L20|All counted phases=L21 -->
| Phase | GPU-h (all) | GPU-h (jobs that did not finish) | CPU-h (all) | CPU-h (jobs that did not finish) |
|---|---|---|---|---|
| Phase 5 | 0.0 | 0.0 | 45.9 | 1.9 |
| Phase 6 | 11.5 | 3.8 | 0.0 | 0.0 |
| Phase 7 | 55.3 | 17.9 | 0.0 | 0.0 |
| Phase 8 | 16.2 | 0.8 | 0.0 | 0.0 |
| Phase 11 | 9.5 | 1.3 | 0.0 | 0.0 |
| Phase 12 | 17.1 | 0.3 | 0.0 | 0.0 |
| Phase 13 | 107.6 | 51.3 | 0.0 | 0.0 |
| Phase 16 | 0.1 | 0.0 | 15.7 | 0.0 |
| All counted phases | 217.2 | 75.6 | 61.7 | 1.9 |

Table 12. Compute by phase, in hours, as the ledger prints it. These totals cover many experiments, so
they cite the ledger itself and no single ID. Phases not listed have no hours in the ledger.

<!-- trace-table: GW-REC-034 reports/compute/ledger.md GW-REC-034=GW-REC-034 -->
<!-- trace-table: GW-REC-035 reports/compute/ledger.md GW-REC-035=GW-REC-035 -->
<!-- trace-table: GW-REC-036 reports/compute/ledger.md GW-REC-036=GW-REC-036 -->
<!-- trace-table: GW-REC-024 reports/compute/ledger.md GW-REC-024=GW-REC-024 -->
<!-- trace-table: GW-DET-001 reports/compute/ledger.md GW-DET-001=GW-DET-001 -->
<!-- trace-table: GW-DET-007 reports/compute/ledger.md GW-DET-007=GW-DET-007 -->
<!-- trace-table: GW-BASE-003 reports/compute/ledger.md GW-BASE-003=GW-BASE-003 -->
<!-- trace-table: GW-EFF-009 reports/compute/ledger.md GW-EFF-009=GW-EFF-009 -->
| Experiment | Device | Jobs | Failed jobs | Wall-clock h | Training h |
|---|---|---|---|---|---|
| GW-REC-034 | GPU | 5 | 4 | 21.58 | 20.11 |
| GW-REC-035 | GPU | 3 | 2 | 20.38 | 19.09 |
| GW-REC-036 | GPU | 2 | 1 | 20.63 | 19.37 |
| GW-REC-024 | GPU | 2 | 1 | 20.37 | 19.31 |
| GW-DET-001 | GPU | 3 | 1 | 5.74 | 4.86 |
| GW-DET-007 | GPU | 3 | 0 | 8.18 | 7.51 |
| GW-BASE-003 | CPU | 6 | 2 | 19.22 | |
| GW-EFF-009 | CPU | 64 | 0 | 12.51 | |

Table 13. The largest single entries of the ledger. A failed job here includes planned stops, and
training hours exclude validation and scoring.

The ledger leaves out work that was not run from a queue, so the totals are lower bounds. It does not
count Phases 9, 10, 14 and 15, jobs run by hand (the Phase 6 sanity sequence, the redesigned GPU resume
check, the exit-crash probes, the re-scores of GW-REC-001 and GW-REC-002, the Phase 8 preview and
label-ceiling checks, and the Phase 16 re-score after the report-path fix), or any test run. CPU steps
inside the GPU chains and the building of the GlyphSynth packs are left out as well.

These registered experiments have no row in the ledger:

- Phase 9: GW-DET-004, GW-DET-005, GW-DET-006, GW-EFF-001.
- Phase 10: GW-E2E-001, GW-E2E-002, GW-E2E-003, GW-EFF-002.
- Phase 13: GW-E2E-005.
- Phase 14: GW-E2E-006, GW-E2E-007, GW-E2E-008, GW-E2E-009, GW-EFF-005, GW-REC-038, GW-REC-039.
- Phase 15: GW-E2E-010, GW-REC-040, GW-REC-041.
- Phases 17 and 18 and their v1 re-runs: GW-EFF-003, GW-EFF-004, GW-E2E-004, GW-EFF-012, GW-E2E-011,
  and GW-E2E-012 from Phase 20.
- Never run: GW-REC-023 (abandoned), GW-REC-032 and GW-REC-033 (not run, because option B was chosen).

Code: `scripts/compute_ledger.py`. Configuration: the queue logs named in that script, which stay
local under `runs/`.

## 9. Failures

The log of failures is `research/failures.md`. The cases that bear on how to read the results are these.

The first frame-stride ablation, GW-REC-019, ran out of GPU memory during its first epoch. CTC needs
at least one frame per label symbol, so at a coarser stride some lines are padded far beyond their
image width, and the sampler did not account for that. It was re-run as GW-REC-021 with batching that
uses the padded widths.

PP-OCRv5 server could not read the GDC pages on this laptop in Phase 5. In Phase 14 it died on
OmniDocBench pages of about eight million pixels or more, with memory not short, and the same happened
on large pages built from validation images. It is recorded as not runnable on those sets. No smaller
input was tried, because that would be a different configuration, and its finished pages were not
scored, since they exclude exactly the large ones.

GW-REC-023, a CRNN combination run, was abandoned before it started once GW-REC-022 had finished, so
no run separates the encoder's share (section 5.2).

<!-- trace: GW-E2E-004 reports/package/attempt1_record_post/GW-E2E-004.json -->
The first v0.1 bundle carried the detector's old post-processing, because a helper in a local build
script failed silently. Its own export checks passed, and the package check against the Phase 10
pipeline caught it: page CER 26.28% against <!-- derived: bundle_reference_cer --> 24.33% on the <!-- derived: bundle_check_pages --> 400 pages. The bundle was rebuilt. An export
can match itself exactly and still be the wrong model, which is why v1 has a check against the measured
pipeline (section 10).

<!-- trace: GW-REC-034 reports/training/GW-REC-034/report.md#notes -->
The final recognizer stopped once with CUDA out of memory in its epoch <!-- derived: resume_epoch --> 25, most likely because another
program took GPU memory, and resumed from step <!-- derived: resume_step --> 142,000. Two other interruptions had causes outside
the method: a file lock on a checkpoint rename (GW-DATA-017) and an unrelated `uv run` that
half-replaced the environment during GW-REC-034, after which the packages were reinstalled at their
locked versions. Neither changed a result.

In Phase 14, HierText's official evaluator stopped after five minutes because of a job deadline in its
runner, and it was started again through a launcher that lifts only that deadline. On validation pages
the launcher and a direct start gave identical score files. In Phase 16 a timing check on page stages
was too tight for short pages on the GPU and was widened, and some variant reports had recorded a local
checkpoint path, which was corrected before they were committed. No reported number depends on either.

Negative results are kept with the rest. H2′ on synthetic data was not supported, H4 was not supported
and was closed post hoc, and several confirmatory tests were not met (section 6.2).

Code: the scripts named in `research/failures.md`. Configuration: `data.ctc_aware_batching` in
`configs/rec/gw-rec-021.yaml`.

## 10. Re-runs on the v1 bundle

Phases 17 and 18 were first carried out on an earlier v0.1 bundle, and Phase 20's failure-path tests on
tiny untrained models, before the frozen v1 existed. All three were repeated on the v1 bundle with
manifest SHA-256 `875357a6…`. PyTorch is its default backend.

<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#tolerances/map_max_abs/measured -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#tolerances/logits_max_abs/measured -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#tolerances/map_max_abs/limit -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#tolerances/logits_max_abs/limit -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#recognizer/identical_text -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#detector/pages -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#end_to_end/pages_with_identical_text -->
GW-EFF-012 checked the bundle on the PyTorch backend against the reference outputs that Phase 16 wrote
once from the PyTorch v1 build on the CPU. The detector probability maps on the <!-- derived: bundle_profile_pages --> 110 profile pages
differ by <!-- derived: bundle_map_difference --> 0.0 (limit 0.0001). The recognizer logits on the <!-- derived: efficiency_lines --> 5,000 lines differ by <!-- derived: bundle_logit_difference --> 0.0 (limit 0.001),
with all <!-- derived: bundle_identical_lines --> 5,000 lines giving identical text. On the 400 pages all page texts are identical.

<!-- trace: GW-E2E-011 reports/package/GW-E2E-011.json -->
GW-E2E-011 ran the installed package on the bundle over the <!-- derived: baseline_pages --> 400 Phase
5 validation pages and compared it with GW-E2E-006's predictions for the same pages, which came from
the GPU in strict fp32. The page CER is 24.32% for both, a difference of 0.0 points against a limit of
0.05, and 16,835 of 16,835 lines have identical text and box. In that report the key
`page_cer_phase10` holds this <!-- derived: baseline_pages --> 400-page reference, despite its name.

<!-- trace: GW-E2E-012 reports/hardening/GW-E2E-012.json#fuzz_cases -->
<!-- trace: GW-E2E-012 reports/hardening/GW-E2E-012.json#inputs/torch/good_page_lines -->
GW-E2E-012 repeated the failure-path checks on both backends: seeded fuzz-lite inputs (120 cases),
unusual pages, wrong types, bad files, images over the pixel limit, damaged copies of the bundle, the
cache, the command line and the HTTP API. On each backend <!-- derived: hardening_inputs --> 269 inputs
gave <!-- derived: hardening_results --> 47 results. All the others raised typed errors, and no other
exception occurred. The good page read as 47 lines from a path, bytes, an image object and an array.

The code of v1 is the tree tagged `phase14-eval`. The later tag `v1-code` changes only docstrings,
comments and the default bundle name in `from_pretrained`, as `scripts/v1_code_diff.py` records in
`reports/export/v1-code/code_diff.json`. All of these re-runs are on validation data or made-up inputs.

Test results come from the GPU strict-fp32 evaluation; equality with the bundle is shown on validation
only. Phase 22 checked reproduction at the 0.05-point tolerance
(`reports/final_validation/v1_reproduction.md`): every prediction was the same, and it found 16
failures among the 1,381 point values checked, all of them official OmniDocBench scores.

Code: `scripts/phase17_v1_check.py`, `scripts/package_check.py`, `scripts/hardening_check.py`,
`scripts/hardening_cases.py`, `scripts/v1_code_diff.py`, `glypharis/export/bundle.py`. Configuration:
`configs/final/glypharis_v1.yaml`; the plans are in `docs/PHASE_STATUS.md` (the v1 re-run sections of
Phases 17, 18 and 20).

## 11. Final validation and release

Final validation (Phase 22) has been run; its record is `docs/FINAL_VALIDATION.md`. Configuration
frozen and recorded. Phase 22 re-ran it: the predictions equal Phase 14's in every test file, and the
strict verdict failed only on official OmniDocBench scores, through the evaluator's time limit; the
owner accepted this as a disclosed limitation (`reports/final_validation/v1_reproduction.md`). No
weights are distributed; release is Phase 23.

Intended terms under ADR-0014, effective only if released. The owner decided on 2026-11-18
([ADR-0014](../../docs/decisions/ADR-0014-weights-and-backbone-terms.md)) that the intended terms for
the v1 weights are CC BY-SA 4.0 with attribution to HierText and Open Images, and that GW-DET-001 stays
v1's detector as a disclosed exception to ADR-0009's commercial-use rule. This is the project's chosen
handling, not a legal conclusion, and the v1 weights are not described as cleared for commercial use.
For the code licence and the project citation: To be settled in Phase 23.
