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
