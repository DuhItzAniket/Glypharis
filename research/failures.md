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

### 2026-10-24 — GW-BASE-003 — PP-OCRv5 server cannot run the GDC pages on the laptop

- **What we tried:** the PP-OCRv5 server end-to-end run on the GDC test pages, the last run of its reduced scope, after its two recognition runs (HierText validation lines and GDC test lines) had finished.
- **What happened:** two attempts, both dead within five minutes on the first page and with no page written. The first (2026-10-24 21:26) shared the machine with a GPU training run; the system reached its commit limit (53.2 of 54.2 GB) and the process died without a message, taking the GPU run down with a host-allocation failure. The second (21:50) ran alone and ended with an access violation (exit 139). The last log line before both: PaddleOCR resizing a 3,072 x 4,080 page to fit its 4,000-pixel side limit.
- **Root cause (probable):** the server-size text detector at near-full page resolution needs more memory on the CPU than this 16 GB laptop can give it. PP-OCRv6 medium ran the same pages without trouble.
- **What we changed:** nothing in the runner. The run is recorded as not runnable on this hardware, as the Phase 5 exit criteria allow; running it with a smaller detector input would be a different, non-default configuration.
- **Lesson / implication for the paper:** the server pipeline's page-level memory is itself a deployment finding for a laptop-class target; the report states the missing cell and why.
