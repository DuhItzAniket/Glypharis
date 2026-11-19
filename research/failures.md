# Failures, Dead Ends and Negative Results

Negative results are research results. Record every failed, diverged, abandoned or
disappointing experiment here, including bugs that invalidated a run.

## Template

```markdown
### YYYY-MM-DD — <experiment ID or component> — <one-line summary>

- **What we tried:**
- **What happened:** (symptoms, metrics, logs under reports/...)
- **Root cause (if known):**
- **What we changed:**
- **Lesson / implication for the paper:**
```

---

### 2026-10-24 — GW-REC-019 — the stride-8 ablation ran out of GPU memory

- **What we tried:** the control GW-REC-003 with `model.frame_stride: 8` (a frame covers 8 input columns instead of 4).
- **What happened:** out of GPU memory at step 1,444 of epoch 1, on a batch of 64 lines padded to 3,264 columns (`reports/training/GW-REC-019/report.md`). The queue resumed it twice from step 1,000 and it failed at the same batch each time, since batch order is fixed by the seed.
- **Root cause:** CTC needs at least one frame per label symbol, so collate pads a line to stride x the frames its label needs. The sampler buckets and caps batches by image width only. At stride 8, 10,331 of the 348,935 training lines (3%) need more columns than their image, up to 3,256; the column cap (32,768) never saw those widths. At stride 4 the same is true for 948 lines (0.27%, at most 1,628 columns), which the earlier runs absorbed.
- **What we changed:** `data.ctc_aware_batching` makes the sampler use the padded widths. It is opt-in, so the stride-4 runs keep the batches they were compared on. The ablation is re-run as GW-REC-021 with it on.
- **Lesson / implication for the paper:** a coarser frame stride cannot represent dense lines without padding, and that padding has to be part of batching, or memory can blow up on a few batches; the share of such lines (3% at stride 8) is itself part of the stride-8 result.

### 2026-10-24 — GW-BASE-003 — PP-OCRv5 server cannot run the GDC pages on the laptop

- **What we tried:** the PP-OCRv5 server end-to-end run on the GDC test pages, the last run of its reduced scope, after its two recognition runs (HierText validation lines and GDC test lines) had finished.
- **What happened:** two attempts, both dead within five minutes on the first page and with no page written. The first (2026-10-24 21:26) shared the machine with a GPU training run; the system reached its commit limit (53.2 of 54.2 GB) and the process died without a message, taking the GPU run down with a host-allocation failure. The second (21:50) ran alone and ended with an access violation (exit 139). The last log line before both: PaddleOCR resizing a 3,072 x 4,080 page to fit its 4,000-pixel side limit.
- **Root cause (probable):** the server-size text detector at near-full page resolution needs more memory on the CPU than this 16 GB laptop can give it. PP-OCRv6 medium ran the same pages without trouble.
- **What we changed:** nothing in the runner. The run is recorded as not runnable on this hardware, as the Phase 5 exit criteria allow; running it with a smaller detector input would be a different, non-default configuration.
- **Lesson / implication for the paper:** the server pipeline's page-level memory is itself a deployment finding for a laptop-class target; the report states the missing cell and why.

### 2026-10-26 — GW-REC-023 — the CRNN combination was abandoned before it ran

- **What we tried:** nothing was trained. GW-REC-023 was registered on 2026-10-25, while GW-REC-022
  was running and before its result: GW-REC-006 (CRNN, input height 48) with weight decay 1e-2
  (`configs/rec/gw-rec-023.yaml`). It was to be the cheaper candidate for the full-budget
  confirmation, and the run that shows whether the SVTR-style encoder still adds anything on top of
  the other two changes.
- **What happened:** it was abandoned on 2026-10-26 without starting, so it has no run report and no
  job in `reports/compute/ledger.md`. GW-REC-022 had finished at -0.76 points [-0.81, -0.71] against
  the control (`reports/training/phase07/ablation_table.md`) and -0.27 [-0.32, -0.22] against
  GW-REC-006 (`reports/training/compare/GW-REC-022_vs_GW-REC-006.md`).
- **Root cause:** a decision, not a fault. The entry would have preferred this run if GW-REC-022 did
  not beat it by a clear margin. GW-REC-006 was used in its place for that reading: GW-REC-022 beat
  it clearly, and weight decay alone was worth only -0.04 (GW-REC-014), so this run could only have
  isolated that small share.
- **What we changed:** the GPU time went to the full-budget confirmation, GW-REC-024. The registry
  keeps the ID and the config, with no report path.
- **Lesson / implication for the paper:** the -0.27 is not the encoder's effect alone. GW-REC-022
  differs from GW-REC-006 in the encoder, weight decay, and the encoder's learning rate and
  micro-batches, and row L4 of the Phase 13 ladder adds them together too (`research/ablations.md`).
  No run measures the encoder's share with the other changes held equal, so the paper should credit
  the combination.

### 2026-10-28 — GW-EFF-003 / GW-EFF-004 / GW-E2E-004 — the first v0.1 bundle had the wrong post-processing

- **What we tried:** building the v0.1 bundle with Phase 9's settings and running the Phase 17 and 18
  checks on it, from a local script that runs them in order.
- **What happened:** the bundle carried long side 1536 and temperature 2.35 but the detector's old
  post-processing (`inverse10`, `box_thresh` 0.6, `min_component_area` 3). Its own checks passed
  (ONNX matched PyTorch: probability maps within 7e-6, logits within 4e-4, all 5,000 lines' text and
  the page CER identical), but the package check against Phase 10 failed: page CER 26.28% against
  24.33%, 32 lines per page against 41. Outputs kept in
  `reports/export/GW-EFF-003/attempt1_record_post/` and `reports/package/attempt1_record_post/`.
- **Root cause:** the script built the post-processing overrides before changing into the Phase 17
  checkout; the helper it called failed silently there, the override list came out empty, and the
  config's defaults were used.
- **What we changed:** the script now builds the overrides in the right checkout and stops if they
  are missing; the bundle was rebuilt and every check rerun as registered. The package check did
  its job: it is what caught this.
- **Lesson / implication for the paper:** an export can be perfectly equivalent and still be the
  wrong model; the release needs a check against the measured pipeline, not only against itself.


### 2026-10-30 — GW-DATA-017 — a checkpoint rename was blocked by another process

- **What we tried:** the first 250K run of Phase 12, in the overnight chain.
- **What happened:** 15 minutes in, at step 8,000, saving `last.pt` failed: Windows refused to
  rename `last.pt.tmp` over `last.pt` because another process held the file (`WinError 32`). The
  run stopped, and the chain with it, for about four hours until it was noticed.
- **Root cause:** most likely the virus scanner reading the 93 MB file just written; the rename
  had no retry.
- **What we changed:** `save_checkpoint` retries the rename for about 20 seconds on this error
  (tested). The run resumed from its previous `last.pt` (step 7,000), losing about 1,000 steps;
  the half-renamed temporary file was deleted.
- **Lesson / implication for the paper:** none for the results (the resumed run follows the same
  schedule); long unattended runs on Windows need retries around file renames.

### 2026-11-07 — environment — a stray `uv run` half-replaced the project environment mid-run

- **What we tried:** nothing on purpose: GW-REC-034 was training in the shared `.venv`.
- **What happened:** a `uv run` command for another project was started by mistake in this
  checkout. uv began syncing `.venv` to its own resolution: it removed fontTools and half of
  Pillow, then stopped on a Pillow extension file the training process had open, and left an
  untracked `uv.lock`. Training itself was not affected (both packages were already loaded), but
  the report and scoring steps after it would have failed.
- **Root cause:** uv treats any folder with a `pyproject.toml` as its project and syncs the
  environment it finds there; this project's environment is managed with pip from
  `requirements/lock.txt`.
- **What we changed:** the run was paused at an epoch end, the two packages were reinstalled at
  their locked versions (binary wheels only, no dependency changes), `pip freeze` was compared with
  the lock (identical apart from the separately installed torch), `pip check` and the full test
  suite passed, and the run resumed from `last.pt`. `uv.lock` is not committed.
- **Lesson / implication for the paper:** none for the results; the run's checkpoints and
  schedule are unchanged.

### 2026-11-10 — GW-REC-034 — CUDA out of memory in the 26th epoch

- **What we tried:** the final recognizer, 30 epochs.
- **What happened:** at 00:12 the run failed in the backward pass with CUDA out of memory, in
  epoch 25 (counting from 0). The run had used about 2.5 GB of the 6 GB card until then, and
  nothing in its batches changes between epochs (the sampler is seeded and fixed).
- **Root cause:** most likely another program on the laptop took GPU memory while the run was
  going; that cannot be checked after the fact.
- **What we changed:** nothing in the code. The run resumed from its last checkpoint (step
  142,000) and finished with no further trouble; the steps after that checkpoint were repeated
  (`reports/training/GW-REC-034/report.md`, Notes). The remaining runs were kept clear of other
  GPU work.
- **Lesson / implication for the paper:** none for the results (a resumed run follows the same
  schedule; Phase 6's resume check); on a shared laptop GPU, long runs need the card to
  themselves.

### 2026-11-17 — GW-BASE-003 — PP-OCRv5 server's detector dies above about 8 megapixels

- **What we tried:** the PP-OCRv5 server end-to-end pass on the 100-page OmniDocBench subset in
  Phase 14, started again after the first start died with the machine out of memory
  (`reports/external/run_record.md`, step 5).
- **What happened:** it died with an access violation (exit 139) after 62 pages, starting a
  5723 x 8614 newspaper page. Memory was not short this time (36.7 of 62.3 GB committed, no
  low-memory event).
- **Root cause:** a fault inside the server-size text detector (PaddleX's `paddle_static` runner,
  CPU) for any input of about 8 million pixels or more. Repeated on validation images only: a
  page of the same size built from HierText validation pages died the same way with under 4 GiB
  in use; on mostly white W x 4000 pages, W = 1984 (7.94 M) passed and W = 2080 (8.32 M) and
  every larger width died within two minutes. Why the detector faults there is not known; it
  does not depend on memory or page content. The Phase 5 GDC failure (2026-10-24) fits the same
  limit: those 3072 x 4080 photos reach the detector at 3008 x 4000 (12.0 M), and the second
  attempt there, run alone, ended with the same access violation. Its first attempt did hit the
  commit limit, so memory may have played a part then too.
- **What we changed:** nothing in the runner or the configuration (a smaller detector input would
  be a different, non-registered configuration). PP-OCRv5 server is recorded as not runnable on
  OmniDocBench, with no substitute, under the registered rule (PHASE_STATUS, Phase 14 item 4):
  one `NOT_RUNNABLE` entry in `scripts/phase14_score.py`. Its 62 pages are kept, unscored.
- **Lesson / implication for the paper:** the server pipeline at its default settings cannot read
  large pages on a CPU (61 of OmniDocBench's 755 English pages, 5 of the subset); this is a
  limit of that configuration, reported as a missing cell with its reason, not scored on the
  pages it did finish (those exclude exactly the large pages).

### 2026-11-17 — GW-E2E-009 — HierText's official evaluator stops after five minutes

- **What we tried:** the Phase 14 scoring pass. It scored all 51 prediction files and passed the
  Phase 5 reproduction check, then started HierText's official evaluator on the first system with
  HierText test pages (Tesseract, 1,634 pages).
- **What happened:** the evaluator exited with an error five minutes after it started. Its output
  ends in gRPC stream errors (DEADLINE_EXCEEDED, then connection resets). The pass stopped there,
  as designed, and stays "started". Repeated on HierText validation pages (Tesseract's Phase 5
  validation predictions): the same error at five minutes, after 1,103 of 1,715 pages.
- **Root cause:** Apache Beam 2.77's DirectRunner hands the evaluator's pipeline to its Prism
  runner. Prism's job-state and message streams carry a gRPC deadline of `--job_server_timeout`,
  300 s by default, so any run longer than five minutes is cut off. `eval.py` builds its own
  pipeline options and takes no extra flags. The evaluator's tests only use a few pages, which
  finish well inside the deadline, so the limit never showed before.
- **What we changed:** `scripts/hiertext_eval_launch.py` starts the pinned `eval.py` unchanged,
  with `--job_server_timeout` added to its pipeline options, and `scripts/official_hiertext.py`
  starts the evaluator through it. Checked on validation pages only: on the first 300, a direct
  start and the launcher wrote identical score files; through the launcher, the full validation
  split finished in 543 s. The pass is finished with `score --resume`, as the same evaluation.
- **Lesson / implication for the paper:** none for the numbers (the evaluator's code, flags and
  output are unchanged). An evaluator's own environment can change how it runs; a run on a
  full-size validation split before the test pass would have shown this.

### 2026-11-17 — GW-EFF-007 — the GPU page profile stops on its own stage-sum check

- **What we tried:** step 10 of the Phase 16 chain (94 steps run as one detached chain, the
  compared arms of GW-EFF-008, 009 and 011 in mirrored order, 2026-11-17T13:44:59Z to
  2026-11-18T05:36:52Z): GW-EFF-007's PyTorch GPU page profile, which splits each page's time
  into the nine stages and checks that they sum to the time of a second clock around the call
  within 1%.
- **What happened:** the step stopped at 2026-11-17T15:40:20Z, after 110 s, on that check, on a
  short page: "the stages sum to 75.922 ms, the call took 78.329 ms". The failed attempt wrote no
  output. The nine earlier steps had finished and were kept.
- **Root cause:** on CUDA every stage boundary adds a synchronisation, so a fixed 1% is too tight
  on short pages: here the two clocks differed by about 2.4 ms, about 3% of the page time.
- **What we changed:** the check allows the larger of 1% of the page time and 5 ms
  (`scripts/e2e_profile.py` and its test, a8e8bb6). The step was run again from its start and
  finished in 134 s. The nine earlier steps' reports record git commit 6950a5a and all later ones
  a8e8bb6; that tolerance is the only code difference between the two.
- **Lesson / implication for the paper:** none for the numbers: the check only guards the stage
  split, and no reported number depends on its tolerance. A relative tolerance on a timer needs an
  absolute floor when the thing timed is short and timing it adds a fixed cost.

### 2026-11-18 — GW-EFF-009 — variant reports recorded the checkpoint's local path

- **What we tried:** scoring Phase 16 (`scripts/phase16_score.py`) and a privacy scan of the
  reports before committing them.
- **What happened:** 19 variant reports under `reports/efficiency/phase16/variants/`
  (`det-GW-DET-003` and the `rec-GW-REC-*` variants) recorded the training checkpoint's full local
  path, user folder included. None had been committed.
- **Root cause:** the variant reports copied the build record as it was, and these variants load
  a checkpoint given by path, so the record held the path as it was on this machine. The work
  folder's record needs the full path, since the later steps read it; a committed report does not.
- **What we changed:** reports record a checkpoint path from its `checkpoints` folder on
  (`report_record` and `portable_path` in `glypharis/efficiency/variants.py`, with tests, 1253dfe);
  the work folder's records keep the full path, and the SHA-256 still names the file. The 19 files
  had exactly that one line rewritten to `checkpoints/<ID>/best.pt`; the originals are kept outside
  the repository. Scoring was run again at 1253dfe: in `results.json` only those 19 input hashes and
  the scorer commit changed, and every number is the same.
- **Lesson / implication for the paper:** none for the results. A record written for the machine
  and a record written for the repository are different things; a report should be built for the
  repository from the start, not cleaned afterwards.

### 2026-11-19 — GW-EFF-015 — a code test reached the owner's phone before the registered run

- **What we tried:** testing the `build` and `run` commands of `scripts/phase22_android.py` from a
  throwaway repository, expecting `run` to stop with "no authorized device".
- **What happened:** the owner had connected the Samsung M52 for the coming check while the test was
  under way, so `run` went through: it installed the app, copied the fused recognizer and one
  validation tensor to the phone, ran them and fetched the result. No test data or GDC-scan content
  was involved. The session was created and the logits had the expected shape.
- **Root cause:** the test relied on no phone being attached instead of faking the device.
- **What we changed:** the app and its files were removed from the phone straight away, and the
  run's files were kept out of the official record. The official run is made from the committed build.
  Tests of this script use a fake `adb`.
- **Lesson / implication for the paper:** the operator check's outcome was seen once before the official
  run. Its pass rule is mechanical and was registered first, and nothing about the candidate was chosen
  from this run.

### 2026-11-19 — GW-E2E-015 — the re-score of v1's saved files is not identical

- **What we tried:** Pass A of the v1 reproduction: Phase 14's unchanged scorer on Phase 14's saved
  prediction files, from the `phase22-repro` wrapper, then the exact comparison with
  `reports/external/results.json` (`scripts/phase22_v1_compare.py pass-a`).
- **What happened:** 40 values differ, and all of them are official OmniDocBench scores of two systems,
  Tesseract (GW-BASE-001) and GLYPHARIS (GW-E2E-007): the comparer found no unexplained difference
  (`reports/final_validation/GW-E2E-015/verdict.json`). In both, more pages reached the evaluator's
  per-page time limit and used its fallback than in Phase 14, and the page lists differ. Every other
  value is identical. The validation rehearsal before it reproduced Phase 14 with no difference, and all
  8,062 of v1's Phase 14 files still match their recorded hashes.
- **Root cause:** the official OmniDocBench evaluator limits how long its text matching may take on a
  page and switches to a fallback when that time runs out, so which pages fall back depends on how fast
  the machine is at that moment. The files being scored and the evaluator are the same as in Phase 14.
- **What we changed:** nothing in the evaluator or the rules. As registered (owner decision 6),
  "identical" is recorded as failed with this cause. The owner ruled on 2026-11-19 to continue with the
  test re-run and Pass B under the registered rule unchanged, and to decide on acceptance at the end.
- **Lesson / implication for the paper:** official scores from an evaluator with wall-clock limits are
  not exactly reproducible on the same machine, and the paper says so wherever it cites them.

### 2026-11-19 — GW-E2E-018 — the v1 re-run misses the tolerance on 16 official OmniDocBench scores

- **What we tried:** the v1 test re-run from the `phase14-eval` code (GW-REC-043, GW-E2E-016,
  GW-E2E-017), scored by Phase 14's unchanged scorer with all official evaluators (Pass B), then every
  path of the metric inventory judged at 0.0005 (`scripts/phase22_v1_compare.py verdict`).
- **What happened:** the re-run's predictions are the same as Phase 14's: all 12 files compared, with no
  crop or page whose record differs outside the timings (`reports/final_validation/GW-E2E-018/line_comparison.json`).
  Of the 1,381 point values, 1,365 are within the tolerance, no count, flag or verdict changed, and no run
  identity differs. The 16 that fail are all GLYPHARIS's official OmniDocBench scores (text block and
  reading order), the largest 0.0208 on the three-column slice
  (`reports/final_validation/v1_reproduction.json`). All 8,062 v1 files still match
  `reports/final_validation/v1_files_before.json`.
- **Root cause:** the same as in GW-E2E-015. One more page reached the evaluator's quick-match time limit
  and was scored by its fallback than in Phase 14; Pass A and Pass B fell back on exactly the same pages,
  so on this machine today the result is stable, but it is not Phase 14's. The predictions themselves did
  not change. Two baseline values of the official HierText evaluator also moved, by about 1e-8 (float32
  rounding); baselines are not judged.
- **What we changed:** nothing in the evaluator, the rule or Phase 14's results, which stay as recorded.
  As item 13 says, the verdict is reported as failed and the exit criterion stays unmet until the owner
  rules at the end of the phase. The candidate work does not depend on this verdict.
- **Lesson / implication for the paper:** the model reproduces line for line; scores from an evaluator
  with wall-clock limits do not, and the paper reports them with that caveat and the pages involved.

