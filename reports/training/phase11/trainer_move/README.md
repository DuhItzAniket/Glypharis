# The recognizer trainer on the shared loop: acceptance checks (2026-10-28)

The move of `RecTrainer` onto `glypharis.training.loop.TrainerCore` (owner approved 2026-10-27) was
accepted only if both conditions held. Both were run on the RTX 4050 from a scratch directory, so no
committed run or report was touched.

1. **The recognizer tests pass unchanged.** `tests/test_rec_training.py` and `tests/test_rec_svtr.py`:
   48 passed, GPU tests included, with no change to either file (the Phase 6 names `TrainState`,
   `record_validation` and `state_from_checkpoint` are kept as thin aliases).
2. **An existing GW-REC checkpoint still resumes.** A copy of GW-REC-024's `epoch_028.pt` (written by
   the Phase 7 trainer) resumed at epoch 29, step 164,196 and trained to step 164,300 under its own
   resolved config. Against the original run's log at the same steps:

   | Step | Step loss, resumed | Step loss, original run | Relative difference | Learning rate |
   |---|---|---|---|---|
   | 164,200 | 0.275357 | 0.275449 | 3.3e-4 | 2.791e-06 in both |
   | 164,250 | 0.173295 | 0.172719 | 3.3e-3 | 2.742e-06 in both |
   | 164,300 | 0.191374 | 0.191704 | 1.7e-3 | 2.694e-06 in both |

   Identical learning rates show the optimizer and schedule state came back; the loss differences
   are of the size CUDA's non-deterministic CTC backward produces between any two runs.

In addition, the A/B/C resume check (`scripts/rec_resume_check.py`, the Phase 6 exit criterion) on
the new trainer with GW-REC-024's config passes (`resume_check.json`): the first step after the
resume matches exactly (relative difference 0.0), the learning rate is identical at every step, and
the resumed run's drift (5.9e-5) stays within twice the drift of two uninterrupted runs (5.5e-5;
allowed 1.1e-4).
