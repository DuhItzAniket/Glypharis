# Research Log

Chronological notebook: what was done, what was observed, what was decided. Short entries, newest at
the bottom. Experiments get their full record in `experiments.md`; this log is the narrative around them.

---

### 2026-10-09 — Phase 0: discovery and planning

- Inspected the machine: RTX 4050 Laptop (6 GiB, ~4.96 GiB free at idle), bf16 supported, torch
  2.5.1+cu121 working on driver 566.24. Details in `docs/ENVIRONMENT.md`.
- Surveyed methods and datasets (`approach_survey.md`). Main observation: the brief's training data is
  scene-text-oriented while the target is document pages. Decided to add a document-style synthetic
  source (GlyphSynth, ADR-0003) and a frozen document test set (GDC).
- Wrote the implementation plan, roadmap, architecture, risk register and workflow.

### 2026-10-10 — Phase 0 closed; execution order revised

- Found that part of the Phase 0 documentation had not been written and nothing had been committed.
  Wrote the missing documents (experiment protocol, dataset cards, ADR-0001/0002/0003/0005, survey,
  hypotheses, related work, references).
- Owner decisions: English printed text only; deployment target (CPU vs GPU) undecided; no off-the-shelf
  stopgap in the consumer application, so a usable engine is needed early.
- Recorded ADR-0008: ship v0.1 after Phases 10 → 17 → 18 → 20, then run the studies; CPU-first
  inference. Phase numbers unchanged for traceability.
- Hypotheses H1–H8 written before any experiment (`hypotheses.md`).

### 2026-10-10 — Phase 1: environment

- Project `.venv` with torch 2.14.1+cu126 (driver 566.24 supports CUDA ≤ 12.7). Chose it over the global
  2.5.1+cu121 after the AMP, CTC-in-fp32 and determinism checks passed
  (`reports/environment/env_report.md`).
- Seeded CPU toy training (CNN-BiLSTM-CTC, 50 steps) is bit-identical across runs. On CUDA the same
  check also came out identical, although PyTorch warns that CTC backward on GPU has no deterministic
  kernel; the protocol keeps a 1e-4 relative tolerance rather than assuming exactness.
- CTC computed in an fp32 region under bf16 autocast stays finite with finite gradients.

### 2026-10-10 — Phase 2: licences and SynthText

- Checked every dataset's licence on its official page (DATASETS.md). SynthText's VGG terms are
  non-commercial, 12-month and no-redistribution, and the download links have been withdrawn. The consumer
  application's commercial status is undecided, so SynthText was dropped (ADR-0009). GlyphSynth becomes
  the only synthetic source; RQ2 and H2 reframed before any experiment (H2′).
- HierText test labels have been public since 2024-12, so the planned fallback test split is unnecessary.
- Research-only sets (FUNSD, OmniDocBench) and sets without a stated licence (IIIT5K, SROIE, DocVQA) are
  evaluation-only.
- Downloaded HierText (3.72 GiB). Image and annotation counts match the published split sizes; the
  published word count corresponds to legible words only. S3 throughput was the bottleneck (about
  0.3–0.8 MB/s per connection).

### 2026-10-10 — Phase 3: data audit

- HierText is technically clean: no missing, corrupt or mis-sized images; 3 broken line boxes; 30 lines
  with control or private-use characters; 5.6K lines with vertices slightly outside the image (to clip).
- **Leakage across the official splits.** Pixel hashes found only 4 near-identical cross-split pairs,
  all "same template, different text". Comparing text found 87 cross-split pairs that share most of
  their distinctive lines: the same product, sign, chart or map photographed again. Checked by eye.
  Removed the lower-priority copy (68 train, 9 validation images); test untouched.
  Lesson for the paper: for OCR, text-level overlap is the leakage that matters, and pixel hashes alone
  underestimate it.
- **Domain evidence (R-06).** HierText "lines" are short: median 1 word,
  7 characters; 95th percentile 7 words and
  40 characters. Book-page lines are usually much longer (assumption, to be measured on GDC), so HierText alone
  under-represents long lines. This supports GlyphSynth and is a design input for the recognizer's
  maximum width.
- Characters: 955 distinct in train legible text, 99.4% of occurrences ASCII; 74 / 227 / 596 distinct
  characters cover 99% / 99.9% / 99.99% of occurrences. Input for the Phase 4 charset.
- Train, validation and test distributions (image size, brightness, contrast, blur, line geometry, text
  length) are visually indistinguishable.

### 2026-10-10 — Phase 4: preprocessing, synthetic data and GDC

- Charset v1: 274 characters seen at least 10 times in HierText train labels, letters limited to Latin
  and Greek. It covers 99.943% of in-scope validation characters; 0.13% of validation characters are
  letters of other scripts, out of scope by design.
- HierText line crops: 348,935 train and 73,888 validation; the sample sheet also shows label noise in
  HierText itself (e.g. a crop reading "KI ONOS" labelled "KFONOS").
- An out-of-memory crash in the crop job traced to every worker process importing torch through
  `glypharis.utils`; data tools now import it lazily.
- GlyphSynth v1: the first sample sheet showed Gutenberg plain-text markup (_italics_, "--") in labels;
  fixed before generating. Median line 39 characters against 7 in HierText.
- Augmentation previews caught three defects before any training (rotation and perspective cutting
  text off, mean-threshold binarisation wiping faint text).
- GDC ground truth from PDF text layers needed four corrections, each found on QA overlays: PDFium
  joins hyphenated lines (it marks the printed hyphen as U+FFFE), splits rotated text into single
  glyphs (fixed with per-character angles), compares tiny punctuation boxes as rows, and leaves
  small-caps text in lowercase. Raster images inside PDFs have no text layer and became ignore
  regions.
- Test-set discipline: GDC ground truth contains a few characters outside the charset (en dash,
  minus sign, section sign, Greek letters; 0.11% of characters). The charset was **not** changed in
  response; any change must be justified on training or validation data.
- Decision (ADR-0010): GDC capture runs as its own track because printing is on the owner's side.

### 2026-10-21 — GDC v1 captured, aligned and frozen

- The owner printed the pack and captured all 106 sheets with a phone. Every sheet was identified from
  its corner markers.
- The marker homography alone was fine for flat captures (RMS 2–6 px) but missed by up to a line
  height on the eight curved sheets, and even flat captures had ~1 mm of residual misalignment (paper
  curl, lens distortion). Added a smooth optical-flow refinement with forward-backward consistency;
  median sheet/photo NCC rose from about 0.09 to 0.92.
- A first synthetic test of the refinement used near-identical rows of fake words and locked onto the
  wrong row: repetitive text is a real risk. On real pages the per-line score (source line image vs
  aligned photo) detects such shifts; a test now checks that it does.
- GDC v1 frozen with 106 pages and 8,606 lines. The real-scan slice becomes a separate set (GDC-scan),
  so the frozen set does not wait on it.

### 2026-10-23 — Phase 6: the recognizer baseline

- GW-REC-001 (the CRNN on HierText lines) reached a validation CER of 7.53% [7.38, 7.69] after 30
  epochs, with the best epoch the last one. GW-REC-002 added 35,000 GlyphSynth lines: 7.49%
  [7.34, 7.64], paired −0.05 points [−0.09, −0.01]. That is one seed per arm and not budget-matched,
  so it says nothing yet about the synthetic data (F2).
- The sanity sequence caught two bugs before the full run. The BiLSTM read the right padding of
  shorter lines (fixed with packed sequences), and unbounded batch widths fragmented the 6 GB of VRAM
  (fixed with a cap on the columns per batch).
- The first overfitting rule flagged GW-REC-001 eight times while its CER improved every epoch: a
  CTC model's validation loss creeps up as it grows confident (F1). The rule now also needs CER to
  stall.
- `best.pt` was chosen with early stopping's `min_delta`, so improvements smaller than 0.001 CER
  never became the best. GW-REC-002 had kept epoch 24 (7.58%) over epoch 27 (7.49%). It was
  re-selected from its own records, and the trainer now keeps the lowest CER outright.
- Every GPU training process aborted at exit (0xC0000409, which Git Bash shows as 127). Tiny probe
  scripts traced it to cuDNN's built-in LSTM dropout, whose cached state is freed after the CUDA
  driver has gone. That same state lives outside PyTorch's random generators, so a resumed CUDA run
  drew different dropout masks. The two LSTM layers now run separately with ordinary dropout, and
  both trained models re-score to exactly the same CER.
- The first GPU resume check failed by design. It asked two CUDA runs to agree to 1e-4 over 400
  steps, which nondeterministic CTC backward rules out: two uninterrupted runs differed by up to
  2.90%. The redesigned check passes. The first step after the resume matches exactly, and the
  resumed run drifts less than two uninterrupted runs do (1.51% against 2.90%).
- The Phase 8 code review found that a NaN gradient with a finite loss got through every check in
  the recognizer trainer, because bf16 has no gradient scaler to skip it. The trainer now stops
  before the weights change.

### 2026-10-25 — Phase 5: baselines

- Seven baselines on our evaluation sets and metrics (`reports/baselines/`). PP-OCRv6 small is the
  most accurate full-scope system; PaddleOCR's English default (PP-OCRv6 medium) matches it on
  HierText lines at about twelve times the CPU time per line.
- HierText pages are hard for every system, mostly because detection misses lines; Tesseract's page
  analysis finds about one line in ten. On GDC, Tesseract still has the best reading order, since the
  PaddleOCR pipelines return lines in detection order.
- Three surprising numbers were checked on validation predictions before being written up
  (`scripts/baseline_diagnostics.py`): PP-OCRv5 server drops word gaps, PP-OCRv5 mobile cuts long
  lines short, and TrOCR-small-printed writes capitals. None of them is a scoring bug.
- PP-OCRv5 server could not run the GDC pages on the 16 GB laptop (two attempts, both dead on the
  first page) and takes 81 s per HierText page; recorded as a deployment finding, not retried with
  a non-default configuration.
- The scorer showed the failed run as "0 of 106" because the attempts had left an empty prediction
  file; fixed so a file with no predictions is skipped and the coverage table gives the reason.
- Set a CPU latency target for v0.1 from the measured speeds (ADR-0011, accepted by the owner),
  anchored on PP-OCRv6 tiny, the fastest modern pipeline; Tesseract is faster on pages but finds too
  few HierText lines to be the bar.

### 2026-10-26 — Phase 7: recognizer improvement

- Three seeds of the short-budget control landed within 0.02 points of each other (8.30 to 8.32%),
  so even small differences could be judged; the decision rule was fixed before any arm ran.
- Of 15 one-change arms, three helped: input height 48 (−0.49), the SVTR-style encoder (−0.13) and
  weight decay 1e-2 (−0.04). Frame stride 8 was by far the worst (+2.61): HierText lines are too
  dense for one frame per 8 columns. Its first attempt ran out of GPU memory because CTC padding was
  not counted in the batch cap, which led to CTC-aware batching and a memory probe before runs.
- The three helpful changes combined to more than their sum (GW-REC-022, −0.76), and the combination
  held at the full budget: GW-REC-024 reached 6.46%, −1.07 against the CRNN baseline, at 24.4 ms per
  line on the CPU, just inside the v0.1 target. One seed for now (owner decision).
- Training on the laptop crashed several times from memory pressure (Windows commit limit, a kernel
  token leak, other apps); two loader workers and resuming from checkpoints carried every run through.

### 2026-10-27 — Phase 8: text detector

- The DBNet-style detector (MobileNetV3-Large, ImageNet start) reached F1@0.5 0.724 [0.708, 0.739]
  on all HierText validation pages (GW-DET-001), against a label ceiling of 0.948. On the Phase 5
  pages it beats the best baseline detector, PP-OCRv6 small, by about 0.15, though only it was
  trained on HierText's line definition.
- Its errors are misses: precision 0.92, recall 0.59. Small text (under 8 px, recall 0.20) and
  rotated or vertical lines (recall 0.18 beyond 45°) account for most of them; that is where Phase 9
  starts.
- H9 held: starting from ImageNet weights is worth 0.020 F1@0.5 [0.016, 0.023] over a random start
  at the same budget (GW-DET-002). H10 did not: MobileNetV3-Small lost 0.056 [0.051, 0.062], about
  twice the margin allowed (GW-DET-003). v0.1 keeps Large with ImageNet weights.
- The sanity sequence changed the plan before any main run: the learning rate rose from 1e-3 to
  5e-3 after the range test, and the per-epoch validation subset fell to 400 pages. All three runs
  finished without a crash; 16.2 GPU-hours in all.

### 2026-10-28 — Phase 9: detector error analysis

- The detector's misses were mostly not the network's: 46% of GW-DET-001's missed validation lines
  had a candidate that only the confidence threshold removed. A post-processing sweep chosen on one
  half of validation and checked on the other (GW-DET-004) and a larger input, 1536 px (GW-DET-005),
  raised F1@0.5 from 0.724 to 0.778 without retraining; recall from 0.595 to 0.716, precision down
  from 0.924 to 0.852.
- Timing the detector on the CPU (GW-EFF-001) showed recognition, not detection, fills most of the
  page budget; 1536 px was the largest input inside ADR-0011's page target.
- What is left needs training: lines under 8 px and steeply rotated or vertical lines are still the
  weakest (recall 0.34 and 0.38), and merges and fragments grew with the looser threshold.
- One registered check had nothing to measure: HierText's line annotations are all four-point
  polygons, so whether curved outlines help on GDC (R-22) stays open.

### 2026-10-28 — Phase 10: the full pipeline

- `GlypharisOCR` now reads a page end to end: detect at 1536 px with Phase 9's settings, cut each
  line exactly as the training crops were cut, read them in width-sorted batches, and order them
  by XY-cut. A fixed padding margin makes a line read the same alone or in any batch.
- On the 400 HierText pages the baselines were scored on, GLYPHARIS's page CER is 24.3% against
  42.4% for PP-OCRv6 small, the best of them; on all 1,715 validation pages it is 27.6%.
- With ground-truth boxes the page CER would be 6.7%: detection is now the bottleneck, on sharp and
  blurry pages alike, so H4's expectation that recognition dominates on clean pages does not hold
  on HierText.
- Raw confidence is over-confident; a temperature of 2.35 fitted on half of validation cuts ECE
  from 0.22 to 0.09 on the other half (H8). The v0.1 model carries it.
- The PyTorch pipeline misses the page-time target: 2.0 s per page, mostly recognition. Phase 17
  measures the ONNX build, which is what the target is judged on.

### 2026-10-28 — Phase 17: model export

- The v0.1 model now ships as one bundle: release files, ONNX files and a manifest of their hashes.
  Its ONNX build reads exactly like PyTorch on validation data (identical text on 5,000 lines,
  identical page CER on 400 pages).
- The first bundle was built with the old detector settings because of a scripting slip; its
  equivalence checks passed anyway, and only the check against Phase 10's pipeline caught it.
- ONNX Runtime turned out slower, not faster: the exported recognizer takes 4.5 times as long as in
  PyTorch, and the runtime's memory arena grew to 11 GB. v0.1 uses PyTorch by default; ADR-0011's
  speed target is now an owner decision, and speed work is Phase 16's.

### 2026-10-28 — Phases 18 and 20: the package and its edge

- GLYPHARIS installs as a package with `from_pretrained`, `predict`, a JSON result and a
  `glypharis-ocr` command; the shipped path reads validation pages exactly as Phase 10 measured
  (GW-E2E-004).
- Every input the caller can get wrong (missing, corrupt, truncated, huge, empty or odd images,
  wrong types, damaged model files) now ends in a typed error with a clear message; a seeded fuzz
  test found no other exception.
- That completes the v0.1 path: on HierText validation it reads pages at 24.3% page CER against
  42.4% for the best baseline, at about 2 s per page on the laptop's CPU, which is above ADR-0011's
  target and is the owner's call.

### 2026-10-29 — Phase 11: which augmentations make the recognizer robust

- Five augmentation arms on the CRNN short budget, three seeds each, scored on GLYPH-C (6,000
  validation lines under 7 corruptions at 5 strengths). Phase 7 had already trained 7 of the 15
  runs; the other 8 ran overnight on the laptop, through two power cuts.
- Geometric augmentation does most of the work (−1.67 [−1.73, −1.62] points of mean corruption error); all
  three families together are best (−1.83 [−1.90, −1.75]) and also best on clean lines. Document degradation
  helps a little (−0.25 [−0.30, −0.19]) at no clean cost, so H3 is supported; photometric augmentation shows
  no clear effect.
- The surprise is rotation: tilting a line by a few degrees costs more than any blur, noise or
  compression in the suite, and training only ever tilts by 2°.

### 2026-10-30 — Phase 12: does synthetic data help?

- Not on this evidence. Twelve CRNN runs added 25K to 250K GlyphSynth lines to HierText at the same
  step budget. None improved the document-like HierText pages, and from 50K lines on they got worse,
  more so the more lines were added (+0.39 [+0.25, +0.52] points at 250K).
- The model does learn the rendered lines (their own validation error falls from
  1.97% to 0.38%); they just do not stand
  in for photographed text, and every synthetic line costs a real one at a fixed budget.
- Of the generator's factors, the camera-style capture effects matter for robustness; font variety
  made no difference that three seeds against one could see.
- The real document test is still to come: GDC's phone photos of printed pages, in Phase 14.
