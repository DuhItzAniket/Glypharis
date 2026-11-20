# Final Validation

**Status: skeleton, written when the Phase 22 plan was registered (2026-11-18). Nothing recorded here
has been done yet.** Every row says `not run` until the step it describes has happened, and each result
is then copied from a generated report with its experiment ID. The plan, with every pass rule, is in
[PHASE_STATUS](PHASE_STATUS.md), Phase 22.

## 1. Walkthrough (GW-E2E-013)

A clean-machine-style run from a fresh clone at `420e095`, in a clean folder outside the worktrees.

### 1.1 Environment

| Item | Value |
|---|---|
| Date | 2026-11-18, 19:26:04Z to 20:49:46Z (`reports/final_validation/GW-E2E-013/steps.txt`) |
| Commit of the clone | `420e095`; the affected steps repeated at `8a4b3a3` |
| OS, Python, GPU driver | Windows 11 (10.0.26300), Python 3.11.9, PyTorch 2.14.1 for CUDA 12.6, NVIDIA driver 616.56 |
| Environment report | written by `scripts/env_report.py` in the clean folder; its GPU checks passed |

### 1.2 Steps in order

| Step | Command or action | Result | Evidence |
|---|---|---|---|
| Clone | `git clone`, then `git checkout --detach 420e095` | pass; clean tree | `steps.txt` step 1 |
| Install | virtual environment, PyTorch, `requirements/lock.txt`, editable package, `pip check` | pass | steps 2 to 7 |
| Environment and tests | `env_report.py`, `pytest`, `validate_configs.py` | 2,049 passed, 23 failed, 5 skipped at `420e095`, all 23 in `tests/test_phase14_score.py` (finding 1); 62 of 62 configs valid | steps 8 to 10 |
| Download | HierText, MobileNetV3 weights, FUNSD, IIIT5K | pass after one resumed download (manual step 5) | steps 11 to 13 |
| Verify | `download_datasets.py --verify`; only `downloaded_utc` may change | pass: every file matches its recorded hash | steps 14, 15 |
| Prepare | `prepare_hiertext.py`, `prepare_benchmarks.py`, `build_eval_crops.py`; no derived manifest differs | pass, except the expected IIIT5K line (finding 2), cleared at `8a4b3a3` | steps 16 to 23 |
| Load the model | the v1 bundle, placed by hand; manifest SHA-256 `875357a6…` matched | pass | step 24 |
| Infer | `glypharis-ocr` on one validation page, both backends, from outside the repository | pass: 47 lines, JSON and text written, on each backend | steps 26, 27 |
| Evaluate | `package_check.py --v1`, 400 validation pages | **pass:** page CER 24.32% on both, difference 0.0 points (limit 0.05), 16,835 of 16,835 lines with identical text and box | `reports/package/GW-E2E-013.json` |

The report records `git_dirty: true`: the only tracked difference at that moment was the expected IIIT5K
manifest line (finding 2).

### 1.3 Manual steps

One row for every action not done by a listed command, and for every retry or deviation.

| # | Manual step | Why it was needed | Done by | Outcome | Retry or deviation |
|---|---|---|---|---|---|
| 1 | Copy the v1 bundle into the clean folder | The weights are not distributed before Phase 23 | script, from the worktree's copy | hash matched | none |
| 2 | Copy the GW-E2E-006 reference file (`hiertext_val.jsonl`, SHA-256 `c8520576…`) | It is an evaluation output and is not in git | script | hash matched | none |
| 3 | Set `GLYPHARIS_MODELS` | The package has to find the copied bundle | script | bundle found by name | none |
| 4 | Run `glypharis-ocr` on one page with each backend | Smoke check of the installed command | script | pass | none |
| 5 | Run the HierText download again | `test.tgz` stopped at 348,371,966 of 537,066,598 bytes when the connection dropped | resumed the same command | finished; the hash matched | one retry, from the kept partial file |
| 6 | Restore `data/manifests` after downloading | Downloading rewrites `downloaded_utc` | `git checkout`, as listed in the plan | only those lines had changed | none |

### 1.4 Findings and fixes

| Finding | Fix | Commit | Steps repeated |
|---|---|---|---|
| 1. In a clone with the `phase14-eval` tag, 23 scorer tests failed: they reached the real tag through `hash-phase5` | the tests no longer depend on the real tag; the scorer is unchanged | `8a4b3a3` | full suite in the clean clone: 2,154 passed, 5 skipped |
| 2. `prepare_benchmarks.py iiit5k` left out `index_sha256`, so the fresh manifest differed from the recorded one by that line (`reports/final_validation/GW-E2E-013/derived_manifests_before_fix.diff`) | the script writes it | `8a4b3a3` | `prepare_benchmarks.py iiit5k`; `git diff --exit-code` on the derived manifests passes |
| 3. No document gave the order of work from a fresh clone | `docs/REPRODUCING.md` | `8a4b3a3` | not applicable |

### 1.5 What the walkthrough does not show

It is clean-machine-style, not a second machine: it shares the OS, driver, base Python and git config with
the original work. The weights and the reference file came by hand, so the weight download path is untested
until Phase 23. GDC and GDC-scan cannot be fetched by anyone else. The evaluation is on validation pages.

## 2. Reproduction of v1

| Check | ID | Rule | Result | Report |
|---|---|---|---|---|
| Rehearsal, validation crops | GW-REC-042 | every point value within 0.0005 | passed: largest difference 0.0; 74,500 of 74,500 crops identical | `reports/final_validation/rehearsal.json` |
| Rehearsal, validation pages | GW-E2E-014 | every point value within 0.0005 | passed: largest difference 0.0; 1,715 of 1,715 pages identical | `reports/final_validation/rehearsal.json` |
| Pass A, re-score of the saved files | GW-E2E-015 | identical numbers | failed: 40 official OmniDocBench values differ, 0 unexplained | `reports/final_validation/GW-E2E-015/verdict.json` |
| Re-run, test crops | GW-REC-043 | once per set; outputs complete and hashed | passed: 5 sets, one access each | `reports/final_validation/v1_rerun_outputs.json` |
| Re-run, test pages | GW-E2E-016 | once per set; outputs complete and hashed | passed: 5 sets, one access each | `reports/final_validation/v1_rerun_outputs.json` |
| Re-run, oracle | GW-E2E-017 | once per set; outputs complete and hashed | passed: 2 sets, one access each | `reports/final_validation/v1_rerun_outputs.json` |
| Pass B and tolerance verdict | GW-E2E-018 | every point value within 0.05 points, counts equal | failed: 16 of 1,381 point values outside, all official OmniDocBench | `reports/final_validation/v1_reproduction.md` |

The re-run's predictions are the same as Phase 14's in all 12 test files: no crop or page record
differs outside its timings (`reports/final_validation/GW-E2E-018/line_comparison.json`). No count,
flag or hypothesis verdict changed, and no run differs from Phase 14's recorded identity (model hashes,
settings, batch, inputs, tag, torch, CUDA and cuDNN). After both passes, all 8,062 v1 files still match
`reports/final_validation/v1_files_before.json` byte for byte, and the OmniDocBench evaluator's
`result/` folder held the same files before and after each pass.

### 2.1 Differences found

Every difference is in the official OmniDocBench evaluator. It gives each page a time limit for its
quick text matching and scores the page with a fallback when the time runs out, so which pages fall back
depends on the machine's speed at that moment.

- *Pass A (GW-E2E-015).* On the same saved files, Tesseract fell back on 4 pages against Phase 14's 1,
  and GLYPHARIS on 2 against 1. The 40 differing values are those two systems' official OmniDocBench
  scores; every other number is identical. Recorded in `research/failures.md`. The owner ruled on
  2026-11-19 to keep the rule unchanged, continue, and decide on acceptance at the end of the phase.
- *Pass B (GW-E2E-018).* The 16 values outside the tolerance are GLYPHARIS's official OmniDocBench
  text-block and reading-order scores. The largest is 0.0208, on the three-column slice; on the overall
  reading-order average it is 0.0013. The pages that fell back are exactly Pass A's, one more than in
  Phase 14, and the predictions did not change. Recorded in `research/failures.md`.
- *HierText official evaluator.* Two baseline values (Tesseract E2E-PQ, PP-OCRv6 small Det-PQ) moved
  by about 1e-8 between the passes, float32 rounding. Baselines are not judged.

**Status.** Under the registered rule the reproduction is reported as failed, beside Phase 14's results,
which stay as recorded. The candidate work does not depend on this verdict.

**Owner ruling (2026-11-21).** The owner accepts the reproduction as an explained limitation of the
official OmniDocBench evaluator, disclosed: the strict verdicts above stay as written, the paper reports
them with their cause wherever it cites official OmniDocBench scores, and the reproduction exit criterion
is closed by this ruling. Nothing in the rule, the evaluator or Phase 14's results was changed.

## 3. Phone candidate

A new version, not v1. Its results sit beside v1's and replace none of them.

| Step | ID | Rule | Result | Report |
|---|---|---|---|---|
| Android operator check | GW-EFF-015, part a | session and one tensor on the phone | go on both phones | `reports/final_validation/GW-EFF-015/operator_check.json` |
| Selection | GW-EFF-013 | selection rule in `research/hypotheses.md` | accuracy part done: `fused-int8` and `fused-single-line` both met all three tests, so all three arms go to the phone; the phone part is due | `reports/final_validation/candidate/GW-EFF-013/accuracy.md` |
| Bundle and equivalence | GW-EFF-014 | Phase 17's tolerances | not run | |
| Validation gate, crops | GW-REC-044 | H11's test met | not run | |
| Validation gate, pages | GW-E2E-019 | H11's two tests met | not run | |
| Test crops | GW-REC-045 | once per set, logged | not run | |
| Test pages | GW-E2E-020 | once per set, logged | not run | |
| Scoring and H11 | GW-E2E-021 | six tests, all required; the three on GDC also by source document | not run | |

On the laptop, `fused-single-line` gave `fused`'s text on all 5,000 lines (its `lines.json`) and
`fused`'s page CER and E2E F1 on the 400 pages, and `fused-int8` stayed within its margin on all three
tests. The laptop sitting's page times and peaks are recorded in
`reports/final_validation/candidate/GW-EFF-013/laptop.md` as information: the selection rule judges
speed and memory on the phone. The pages and profile reports carry the recognizer's batching label; on a
page the pipeline hands the recognizer all of that page's crops in one call, and the 256 crops per call
apply to the line evaluation only.

## 4. Phone measurement (GW-EFF-015)

Each column is filled from the device and committed before that phone is measured. No serial number or
account is recorded.

| Condition | Samsung Galaxy M52 5G | Moto G35 5G |
|---|---|---|
| SoC | Qualcomm SM7325 (platform `lahaina`) | Spreadtrum UMS9620 (platform `ums9620`) |
| RAM | 5,500,000 kB reported by `/proc/meminfo` | 3,464,688 kB reported by `/proc/meminfo` |
| Android build | Android 13, SDK 33, `TP1A.220624.014.M526BXXS7CYD2` | Android 15, SDK 35, `VUOAS35.116-151-3` |
| ABI | `arm64-v8a` | `arm64-v8a` |
| Listed by `adb devices` as authorized | yes | yes, with no extra driver |
| `onnxruntime-android` package and SHA-256 | 1.31.1, `d4d50660300ed2c4f38ffdfd5ab5bfa3a22f995f282dda24f42ea3ea55467b49` | same package |
| JDK, Gradle and Android Gradle Plugin versions | Temurin 21.0.12.1, Gradle 8.14.7, plugin 8.13.2 (Kotlin 2.3.21) | same build |
| Session options (sequential execution, all graph optimisations, CPU arena off, memory patterns off) | set as `glypharis.efficiency.threads.ort_session` sets them (`BenchSession.options`), recorded by the app in every result and checked by `phone-report` | same |
| Thread count, spinning setting | 4 intra-op threads; spinning at ONNX Runtime's default, as on the laptop | same |
| Process rule (a new app process per arm and run, force-stopped between runs) | confirmed in the tool: `am force-stop` before and after every pass, one pass per process (`BenchSession.claimProcess`), a reused `pid` refused | same |
| Charge state, starting battery temperature | 91%, charging over USB from the laptop; battery 39.1 °C, processor sensor 46.1 °C, thermal status none (read 2026-11-20 15:04 UTC, `conditions --label registration`) | 74%, charging (the phone reports AC power over the laptop's USB port; at the first reading it was not charging, and the cable was reseated); battery 29.2 °C, SoC sensor 38.6 °C, thermal status light (read 2026-11-20 17:48 UTC) |
| Airplane mode, screen and foreground state | airplane mode on; screen on and unlocked (wakefulness Awake), the home screen in front; the bench app keeps the screen on during its passes. Each run's state before and after is recorded with it. Developer option Stay awake was turned on around the start of the first timed run (the screen timeout was 30 minutes) | airplane mode on; screen on (wakefulness Awake), Stay awake on; the settings screen in front |
| Operator check | go (`reports/final_validation/GW-EFF-015/operator_check.json`) | go (same report) |
| Mean and p95 network time, peak `VmHWM` | not run | not run |
| Each run's mean time | not run | not run |
| Thermal state, start and end of each run | not run | not run |
| Logits and greedy text against the laptop (separate pass) | not run | not run |

The phone app uses the stock `onnxruntime-android` package, unchanged and checked by its SHA-256. The
package's network permissions and its telemetry start-up provider are removed at manifest merge, and
the app turns telemetry off, so the APK asks for no permissions. `scripts/phase22_android.py run`
refuses an APK that asks for any.

**Unplanned run before registration (2026-11-19).** While the check's code was being tested, a test run
reached the M52, which had just been connected, although the tester expected no phone. It installed
the app from a throwaway commit, ran the fused recognizer on the one validation tensor, and the app and
its files were then removed. The session was created and the output had the expected shape. It is not
the result of GW-EFF-015: the official run below is made from the committed build, and the pass rule was
fixed before it and is not changed. See `research/failures.md`.

## 5. Closure checks

| Check | Result |
|---|---|
| Hash lists of the Phase 14 and baseline folders match, before (`reports/final_validation/v1_files_before.json`) and after | not run |
| OmniDocBench evaluator's `result/` file list unchanged across each pass | not run |
| Phase 14 outputs match `phase14_outputs.json` | not run |
| The access log at the start of Phase 22 is an exact prefix of the current log | not run |
| No difference from `phase-21` under the protected paths | not run |
