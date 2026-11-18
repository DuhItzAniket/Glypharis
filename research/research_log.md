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

### 2026-11-14 — Phase 13: what each step is worth

- The ablation ladder, three seeds per row, takes the recognizer from the plain CRNN at 8.66%
  validation CER to 7.45% at the short budget. Every step that stayed in helped on all three
  measures: validation, the document slice and the corruption suite. The biggest single gains
  came from a taller input (48 px) and from augmentation.
- Rotating training lines a little more, up to 4° on half of them, did almost nothing for clean
  lines but cut the corruption error by about a quarter. That is the weakness Phase 11 pointed at,
  and the cure was simple.
- 25K GlyphSynth lines again left validation unchanged, so they stay out of the final recipe,
  although they did make the model more robust.
- The final model at the full budget: 6.29% over three seeds that agree to 0.02 points, 1.24
  points better than the CRNN baseline. The seed-1337 run ships, as fixed before training.
- The detector retraining Phase 9 kept deferring (smaller-text labels, wider rotation) came out
  slightly worse and was not adopted. Detection is still most of the page error; label and
  augmentation tweaks are not what will fix it.
- About 108 GPU-hours on the laptop, over two weeks with pauses; one crash, probably another
  program taking GPU memory, cost a resume and nothing else.

### 2026-11-15 — Phase 14 opens: the plan for the one real test

- The owner gave one of their own documents for a second document test set: a 28-page printed
  report, photographed page by page. Its Word source exists, so its text is exact without any
  transcription. GDC-scan v1 was frozen on 2026-11-14, before any model saw it. It is personal, so
  only numbers from it ever leave the laptop.
- The Phase 14 plan is registered before the frozen model reads any test set: six sets, the same
  seven baselines, one prediction pass and one scoring pass each, and nobody looks at a score until
  every prediction file is written.
- Three claims get a yes, no or "not shown" rule fixed now: GLYPHARIS within twice PP-OCRv5
  mobile's page error on GDC's clean photos and faster on the CPU; a markedly higher error on
  documents than on scene text; and the confidence fix carrying over to documents.
- HierText's and OmniDocBench's official scorers are used where they fit line output: HierText at
  line level, OmniDocBench's text and reading-order parts.


### 2026-11-17 — Phase 14: the frozen model on held-out data

- Every planned prediction pass ran, except PP-OCRv5 server on OmniDocBench, which is recorded
  as not runnable. One scoring pass, resumed once, scored every complete prediction file. The
  Phase 5 baselines re-scored to exactly their Phase 5 numbers first, so the scorer had not
  drifted.
- H1′ holds. On GDC's clean photos GLYPHARIS's page error is well inside twice PP-OCRv5 mobile's,
  on both tracks and with documents as the bootstrap unit. On 100 HierText validation pages it is
  about half a second faster per page on the CPU (−508 ms at 10 threads).
- H5′ is not met: the recognizer reads GDC's printed documents better than HierText's scene text,
  not worse. As H5′ noted when it was written, every Phase 5 baseline did the same.
- The temperature carries over to GDC, but whether it halves the calibration error (ECE) there
  depends on the bootstrap, so H8 is not supported on GDC. On IIIT5K's single words it makes calibration worse.
- HierText test is in line with validation, slightly better, and GLYPHARIS leads every baseline
  run there, by HierText's official evaluator too. On documents the picture is mixed: PP-OCRv6
  small and medium read the crops better, but
  GLYPHARIS keeps the reading order and matches them on bag-of-lines page error.
- Two weaknesses stand out. Sideways photos fail, because there is no page-orientation step. Forms
  lose lines to the detector, although their crops are read best.
- The laptop was the hard part: a power cut, two out-of-memory deaths, PP-OCRv5 server's detector
  dying on large pages, and Apache Beam ending HierText's evaluator after five minutes. Each is
  recorded in the run record or `research/failures.md`; the detector failure and the Beam deadline
  were diagnosed on validation data only. PP-OCRv6 medium and the scoring pass were resumed under
  the registered rules, and PP-OCRv5 server was recorded as not runnable on OmniDocBench.

### 2026-11-17 — Phase 15 opens: what goes wrong, and where

- The plan is registered before any per-line Phase 14 file is read. Phase 15 runs no model on test
  data: it reads the hashed hand-off Phase 14 saved, and checks every file before opening it.
- Three experiments: the recognizer's errors (confusions, words, slices, one category per wrong
  line), the pipeline's (where a page's errors come from, missed lines, box quality, reading
  order), and one validation run on GlyphSynth's labelled lines for the failure modes the real
  sets have no labels for (fonts, touching or broken characters, background noise).
- H4 never got an operational rule, and the published per-condition numbers already decide it:
  even on clean pages the recognizer from ground-truth boxes makes well under half the pipeline's
  errors. It is closed post hoc rather than dressed up as a test.
- Bin edges come from validation, examples from seeded draws, and GDC-scan stays numbers only.

### 2026-11-17 — Phase 15: where the errors are

- The analysis ran once on the hashed Phase 14 files (code `ba44a84`, reports `6cb1b9c`). Every
  checked input matched its recorded hash, every CER matched Phase 14's, and the hand-off's pair
  sums agreed with the page files on all 2,573 pages. No model touched a test set; the one model run
  was on GlyphSynth validation lines.
- Most page error still comes from outside the recognizer. Ground-truth lines that no prediction
  matched at IoU 0.5 are about half of the page error on HierText test and GDC (49.45% and 45.32%,
  GW-E2E-010), and misread matched lines about a quarter. The detector finds 57.92% of HierText
  test's lines under 9.35 px high (recall at IoU 0.5).
- H4 is closed as not supported on GDC, post hoc. Even on clean pages, from ground-truth boxes the
  recognizer makes about a third as many errors as the whole pipeline (32.70%, GW-E2E-010).
- The recognizer fails differently by domain (GW-REC-040). On scene text it is rotation and small
  lines. On documents it drops characters rather than misreading them: merged words and full stops,
  and the worst lines are all table-of-contents dot leaders. Typographic quotes come out as ASCII.
- ADR-0002's worry did not come true: long lines carry little of the validation error.
- F26's explanation for PaddleOCR's reading order on OmniDocBench was not decided by its registered
  measure (GW-E2E-010): 76.66 to 77.74% of the four PaddleOCR systems' consecutive output lines have
  a top edge that does not rise, but PaddleOCR's own sort, read from its code, lets a following line
  sit up to 10 px higher, so the measure cannot tell that sort apart from other orders. F26 is
  revised (F33); its cause is still open.
- On rendered GlyphSynth lines every labelled slice (fonts, touching or broken characters,
  background noise) stays at 1.48% CER or less (GW-REC-041), which says more about the recognizer
  than about photographed pages.

### 2026-11-17 — Phase 16 opens: what v1 costs, and what a faster version would cost in accuracy

- The plan is registered before any timing. v1 stays the build that was evaluated on test; the
  only thing that counts as the same model is its plain fp32 ONNX export, checked against outputs
  saved from the PyTorch build, not against the changed code.
- Everything faster (INT8, smaller inputs, a fused attention, a mixed backend) is a variant: a new
  version, measured on validation as a point on the accuracy and latency plot. Which one ships, if
  any, is the owner's call.
- Four claims get rules: INT8 cuts recognizer latency by a quarter for at most half a point of CER;
  the SVTR-style recognizer is within 1.5 times the CRNN's CPU latency; and the ONNX build of v1
  beats PP-OCRv5 mobile per page, by Phase 14's own rule.

### 2026-11-18 — Phase 16 results: the export is the same model, and slower

- Every planned run finished in one detached chain of 94 steps (2026-11-17T13:44:59Z to
  2026-11-18T05:36:52Z), the compared arms of GW-EFF-008, 009 and 011 in mirrored order (GW-EFF-010
  in Phase 14's order, GW-EFF-007's rows one run each), on mains power with the Balanced plan; no
  sitting was void and the cut line was not needed. Validation data only; no test set was read.
- One step stopped on its own guard. On a short page of the GPU profile the stages summed to
  75.922 ms against 78.329 ms for the call, outside the 1% check, because on CUDA every stage
  boundary adds a synchronisation. The check now allows the larger of 1% and 5 ms (`a8e8bb6`) and
  the step was rerun from the start; no reported number depends on it. After scoring, 19 variant
  reports turned out to record the training checkpoint's local path; they now record
  `checkpoints/<ID>/best.pt` (`1253dfe`), and re-scoring changed only those input hashes and the
  scorer commit.
- v1's plain fp32 ONNX export is the same model: it passes all four of Phase 17's checks against
  outputs saved from the PyTorch build, with no text differences (GW-EFF-006). It is also about 3.3
  times slower per page on the CPU (8,574 against 2,566 ms at 10 threads) and needs about four
  times the RAM (6,529.6 against 1,641.8 MiB, GW-EFF-007).
- The detector is not the problem: its network takes 297.5 ms per page exported and 296.4 ms in
  PyTorch. The recognizer network takes 8,167.7 of the 8,573.7 ms (GW-EFF-007). F13's "exported
  detector twice as fast" came from an arena-on side test; F13 is marked revised and F34 gives the
  arena-off numbers. The same claim in `glypharis/export/onnx.py`'s docstring waits until the
  frozen v1 code may change again, after the Phase 17 re-run.
- Most of the gap is the attention. The plain exporter writes each global mixer as separate
  MatMul, Softmax and mask nodes. Exported with one fused MultiHeadAttention node per mixer, the
  same weights keep v1's page CER and F1 and line text, and the ONNX page time falls from 9,739.4
  to 4,020.8 ms, most of the way to PyTorch's 2,637.3 ms in the same sittings (GW-EFF-009, F35).
- H7′ holds. Dynamic INT8 of the recognizer's constant-weight MatMuls takes a line from 95.78 to
  61.84 ms, with strict line CER 6.208% against 6.221% and the file down from 30.40 to 9.82 MiB
  (GW-EFF-008). The attention products stay fp32; on pages INT8 saves less than fusing:
  −2,848.5 against −5,718.6 ms (GW-EFF-009, descriptive; F36).
- Detector input size moves accuracy more than time. A long side of 1024 adds 7.29 points of page
  CER and saves 1,248.1 ms of 9,739.4; 2048 adds 660.6 ms for a CER change whose interval crosses
  zero (GW-EFF-009, F37). The recognizer sets the page time.
- H6′ is not met: in fp32 ONNX the SVTR-style GW-REC-020 takes 32.01 ms per line against the
  CRNN's 10.16 ms, about 3.2 times; in PyTorch the two are closer, 15.70 against 13.00 ms
  (GW-EFF-011, F39). The unfused attention is the likely reason, not measured for GW-REC-020.
- H1′ ONNX is not met: the v1 fp32 ONNX bundle takes 9,623.6 ms per page against 3,086.0 ms for a
  fresh PP-OCRv5 mobile profile in the same sitting (GW-EFF-010, F40). H1′ itself still holds
  from Phase 14 on the PyTorch build; this verdict is for bundle `875357a6…` only.
- No CPU build of v1 meets ADR-0011's latency targets; the PyTorch build is within the RAM target
  only. The lowest mean page time at v1's accuracy is a new version's, the ONNX detector with the
  PyTorch recognizer, at 2,579.1 ms per page, still over both page time targets but within the RAM
  target (GW-EFF-009, F38). Which backend is the default, and whether any variant ships, is the
  owner's call; Phase 16 changes no default.

### 2026-11-18 — Phase 17 re-run: the shipped path is the evaluated model

- After Phase 16 the owner kept PyTorch as v1's backend and named phones as the target hardware;
  the all-ONNX build with fused attention becomes the phone candidate for Phase 22 (ADR-0011).
- The v1 bundle read through the PyTorch backend, the way it ships, gives exactly the PyTorch v1
  build's outputs: no difference in the maps, the logits, the line texts or page CER (GW-EFF-012).
  With Phase 16's ONNX check (GW-EFF-006) the v1 export is declared successful.

### 2026-11-18 — Phase 18 re-run: the installed package reads like the evaluated pipeline

- Installed from the branch, the package on the v1 bundle reads the 400 Phase 5 validation pages
  line for line as the frozen pipeline did in Phase 14: 16,835 of 16,835 lines with the same text
  and box and the same 24.32% page CER, CPU against the GPU reference (GW-E2E-011).
- A clean environment with only the locked dependencies installs it and runs `glypharis-ocr` on a
  page from outside the repository.

### 2026-11-18 — Phase 20 re-run: the typed errors hold on the real model

- The hardening tests had only ever run on tiny untrained models. On the v1 bundle, on both
  backends, 269 inputs per backend ended in a result or a typed error and nothing else, and every
  damaged copy of the bundle was refused at load, except junk in the `.onnx` files on PyTorch,
  which never opens them (GW-E2E-012).
- With this the export, the package and the hardening have all been re-run on the final model.

### 2026-11-18 — The v1 code freeze ends

- With the export, package and hardening re-run on the final model, the frozen code could finally
  say true things about itself: the docstrings no longer claim the exported detector is twice as
  fast, and the default bundle name is `glypharis-v1`.
- `v1-code` is that commit. A syntax-tree comparison against `phase14-eval` shows nothing else
  changed, so the Phase 14 results still describe this code.

