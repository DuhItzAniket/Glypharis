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
