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
