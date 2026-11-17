# Phase 14 run record

Written during the Phase 14 runs, because `docs/PHASE_STATUS.md` is pinned by the `phase14-eval`
tag until the scoring pass (Phase 14, item 10). Its Phase 14 section is brought up to date from
this file afterwards.

## Pinned code

- Tag `phase14-eval` on commit `6795186be166d6b60238c701753aad32e44d0e98`, set 2026-11-15 after
  the step-2 checks passed (`scripts/phase14_prepare.py check`: every set ok, 13 planned baseline
  runs' batches ok, records committed) and before any validation or test run.
- Crop packs cut in step 2: GDC-scan 776 and FUNSD 2,647 crops, both as registered
  (`data/manifests/processed/eval_crops.json`).

## Validation runs (step 4)

From each run's metadata (`data/processed/phase14/<ID>/hiertext_val.meta.json`; hashes in
`data/manifests/processed/phase14_outputs.json`). Both at the tag, strict fp32, RTX 4050 Laptop.

| ID | Set | Items | Wall time | Per item |
|---|---|---|---|---|
| GW-REC-038 | HierText validation evaluation crops | 74,500 crops | 300.9 s | 4.0 ms per crop |
| GW-E2E-006 | HierText validation pages | 1,715 pages | 810.2 s | 0.47 s per page |

No metric is computed here; both are scored in GW-E2E-009.

## GPU estimate for the test passes (re-estimated from GW-E2E-006)

At the validation rates above: GW-REC-039's 82,369 test crops (67,574 + 8,372 + 776 + 2,647 +
3,000) about 6 minutes; GW-E2E-007's 2,573 pages (1,634 + 106 + 28 + 50 + 755) about 20 minutes,
more for GDC's and GDC-scan's 3072 × 4080 photos; GW-E2E-008's 134 oracle pages a few minutes.
About half a GPU-hour in total, below the plan's estimate of about 1 GPU-hour (which used the
v0.1 recognizer's 449 ms per page, GW-E2E-001). An estimate, not a record.

## Interruptions during the baseline test passes (step 5)

- GW-BASE-007 PP-OCRv6 medium, OmniDocBench pages: a mains power cut left the laptop on battery,
  and Windows put it to sleep at 2026-11-15 20:48Z (Kernel-Power event 42, sleep reason: battery).
  It woke at 22:46Z once power was back. The run's process was suspended, not stopped: it carried
  on from page 263 of 755 without a restart, so there is no resume and no extra access-log row.
  That page's `time_ms` (7,288,872 ms) includes the two hours asleep; `time_ms` in the prediction
  records is diagnostic only, and no reported speed comes from it (speed is GW-EFF-005).
- GW-BASE-007 PP-OCRv6 medium, OmniDocBench pages: died at 2026-11-16 04:41:33Z with an access
  violation (exit 139) after 513 of 755 pages, while starting page 514 (5723 x 8614, resized to
  the 4,000-pixel side limit). One second earlier Windows logged that the system was out of
  virtual memory (System event 26, "Virtual Memory Minimum Too Low"). The predictions written so
  far are intact.
- GW-BASE-003 PP-OCRv5 server, OmniDocBench subset: started by the chain at 04:41:33Z and died
  the same way at 04:42:30Z while loading its models, before any page; its access-log row is the
  one dated 2026-11-16.
- Cause: the commit limit (15.7 GB RAM plus a 16 GB page file Windows grows on demand, 31.4 GB)
  was used up. PP-OCRv6 medium needs about 9 to 10 GB on large pages (peak 9,766 MiB on the
  GDC-scan pages, 8,693 MiB on GDC in Phase 5); the rest of the laptop held about 24 GB at the
  time (other applications, and a 4.2 GB kernel non-paged pool). Neither run is resumed until the
  owner has freed memory; a resume continues the same evaluation and is logged as one.
- After the owner restarted the laptop (commit limit 36.7 GB, about 10 GB in use), GW-BASE-007
  resumed at 2026-11-16 12:39:34Z from its 513 pages, logged as a resume ("resumed: 513 of 755
  samples already predicted"). It finished at 20:17:51Z with all 755 pages (242 new; peak
  10,350 MiB). Page 514, the one it died on, took 294.8 s this time.
- GW-BASE-003 PP-OCRv5 server, OmniDocBench subset: started again at 20:17:51Z (access-log row
  dated 2026-11-16). It died at 2026-11-17 01:36:34Z with an access violation (exit 139) after 62
  of 100 pages, while starting page 63 (5723 x 8614, resized to the 4,000-pixel side limit). This
  time memory was not short: the last memory sample, two seconds earlier, had 36.7 GB committed
  of a 62.3 GB limit, and Windows logged no low-memory or resource-exhaustion event.
- Diagnosis, on validation images only (no test data), with the runner's own pipeline
  construction (`make_system("ppocr_server", "e2e")`):
  - a 5723 x 8614 page tiled from the 12 densest HierText validation pages died the same way,
    inside the text detector (PaddleX `paddle_static` runner), about 2.5 minutes after the model
    load, with the process's private memory never above 4 GiB;
  - mostly white W x 4000 pages with one validation page in a corner (the detector sees them
    unchanged): W = 1984 passed (678 s, peak working set 10.1 GiB); W = 2080, 2144, 2240 and 2656
    each died within 80 to 109 s.
  So the detector dies once its input reaches a size between 7.94 and 8.32 million pixels,
  whatever the page holds and however much memory is free. 5 of the subset's 100 pages (61 of the
  755) reach the detector at 8.32 million pixels or more; the largest page it had finished was
  1984 x 4000 at the detector.
- Recorded as not runnable on OmniDocBench, with no substitute, under the registered rule
  (PHASE_STATUS, Phase 14 item 4), at the owner's decision on 2026-11-17: one `NOT_RUNNABLE`
  entry in `scripts/phase14_score.py`; the runner's plan is unchanged. Its partial file (62
  records, SHA-256 `538baba2b3453920fb432b74f753be2f68125401e45109b0806690699be06348`) stays on
  disk, unscored. The OmniDocBench subset view compares the other systems on the same 100 pages.

## The scoring pass (GW-E2E-009, step 6)

- `phase14_score.py check` passed on 2026-11-17: 51 prediction files complete, the Phase 5 files
  matching their hashes.
- `score` started at 02:13:47Z. It scored every prediction file and reproduced Phase 5, then
  stopped in HierText's official evaluator at Beam's five-minute job deadline
  (`research/failures.md`, 2026-11-17). The rows it wrote are in the access log, and the pass stays
  "started".
- Fix: the evaluator is started through `scripts/hiertext_eval_launch.py`, which lifts only that
  deadline. On validation pages, a direct start and the launcher gave identical score files.
- Code differing from `phase14-eval` at the resume, all scoring-side (the prediction runs were
  made at the tag): `scripts/phase14_score.py` (the `NOT_RUNNABLE` entry and the `not_run` block),
  `scripts/official_hiertext.py` and the new `scripts/hiertext_eval_launch.py`. results.json records
  the scorer commit and the tree comparison.
- `score --resume` started at 02:52:43Z from commit 087252e and ended at 07:08:48Z with exit 0;
  `pass.json` is "done". It re-ran the whole scoring as the same evaluation, so its 70 access-log
  rows carry "(resumed)". Both official evaluators finished on every converted system: HierText on
  5 systems, OmniDocBench on 6 (all pages and the 100-page subset).
- The scorer reads the git state after it has written its own access-log rows and outputs manifest,
  so results.json says `dirty: True`; those two run records were the only uncommitted changes.
- One prediction file was not scored because it holds no predictions:
  `GW-BASE-003/ppocr_server_e2e_gdc_test.jsonl`, the Phase 5 run that died on GDC's photos
  (`research/failures.md`, 2026-10-24). It is a Phase 5 file; the matching Phase 14 cell
  (PP-OCRv5 server end to end on GDC-scan) is listed under `not_run` in results.json.
