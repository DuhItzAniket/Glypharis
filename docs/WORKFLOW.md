# Engineering Workflow

How work moves through this repository. These rules apply to every contributor, human or otherwise.

## 1. Branches

| Branch | Purpose |
|---|---|
| `main` | Stable. Only phase-complete work lands here. Every commit on `main` must pass CI (from Phase 1 on). |
| `phase/NN-short-name` | One branch per phase, e.g. `phase/01-environment`, `phase/06-rec-baseline`. Merged into `main` with `--no-ff` when the phase gate passes. |
| `exp/<experiment-id>` | Optional, short-lived. Only for experiments that need code changes not ready for the phase branch. Deleted or merged after the experiment is recorded. |

Phase 0 was committed straight to `main` because the repository was empty.

Each completed phase gets an annotated tag: `phase-00`, `phase-01`, … The tag message summarises the
phase completion record from `docs/PHASE_STATUS.md`.

## 2. Commit messages

Write messages the way an engineer would in a lab notebook: what changed and, if not obvious, why.

- Imperative mood, ≤ 72 characters on the first line: `Fix CTC padding in recognition batches`.
- Body (optional) explains *why*, wrapped at 72 columns. Mention experiment IDs where relevant
  (`Results for GW-REC-004 in research/experiments.md`).
- One logical change per commit. Don't bundle a refactor with a bug fix.
- Never commit just to commit. Never write filler like "enhance", "comprehensive improvements",
  "various fixes", "update code".
- No generated-by / co-authored-by trailers for tools.

Good:

```
Add width bucketing to the recognition dataloader
Handle empty HierText line annotations instead of dropping them
Train first stable CRNN baseline (GW-REC-001)
```

## 3. The phase gate

A phase is not complete until **all ten** steps below are done and recorded in
`docs/PHASE_STATUS.md`. "The code looks right" does not count as verification.

1. **Implementation**: all planned tasks for the phase, or an explicit, documented descope.
2. **Testing**: unit tests for new logic, and the existing suite still passes locally.
3. **Review**: re-read the diff end to end and look for leakage, hard-coded paths, and silent data drops.
4. **Diagnostics**: run the phase's diagnostic scripts (data audit, training diagnostics, benchmark…) and save their output under `reports/`.
5. **Documentation**: update `PHASE_STATUS.md`, the research log, and any affected docs/cards.
6. **Git diff inspection**: `git diff main...HEAD --stat` plus a read-through, with no stray data, checkpoints, or secrets.
7. **Commit**: meaningful commits plus a final checkpoint commit for the phase.
8. **Push**: push the phase branch, then merge to `main`.
9. **CI verification**: the GitHub Actions run on the pushed commit is green (from Phase 1 on).
10. **Phase completion record**: status, measured results, known issues, commit hash, CI status, tag.

If a step fails, the phase stays open. Serious unresolved failures block the next phase.

## 4. Experiment IDs

Every training or evaluation run that produces a number someone might quote gets an ID:

| Prefix | Scope |
|---|---|
| `GW-BASE-NNN` | External baselines (Tesseract, PaddleOCR, TrOCR) |
| `GW-REC-NNN` | Recognizer experiments |
| `GW-DET-NNN` | Detector experiments |
| `GW-E2E-NNN` | Full pipeline experiments |
| `GW-DATA-NNN` | Data studies (synthetic mix, augmentation ablations) |
| `GW-EFF-NNN` | Efficiency / export / quantisation measurements |

IDs are allocated in `research/experiments.md` **before** the run starts, together with the
hypothesis being tested. A failed or abandoned run keeps its ID and is recorded in
`research/failures.md`. IDs are never reused.

## 5. Test-set discipline

- The test splits and external benchmarks are evaluated **only** at the points the
  implementation plan allows (Phase 5 baselines, Phase 14 external evaluation, Phase 22 final validation).
- Every test-set evaluation is appended to `research/test_access_log.md` with the date, experiment ID,
  checkpoint, and reason. If that log grows outside those phases, something is wrong.
- Model selection, early stopping, and hyperparameter search use validation splits only.

## 6. Numbers

- Never write a metric into a doc by hand. Numbers come from generated reports under `reports/` and are
  copied together with the experiment ID that produced them.
- Negative and null results are recorded with the same care as positive ones.
- No "SOTA", "best", or "state-of-the-art" wording unless a properly controlled comparison supports it.
