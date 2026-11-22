# Final Validation

**Status: complete (2026-11-21, tag `phase-22`).** Every step of the Phase 22 plan has run, and each result
below is copied from a generated report with its experiment ID. The plan, with every pass rule, is in
[PHASE_STATUS](PHASE_STATUS.md), Phase 22. In short: the walkthrough passed; the v1 re-run gave Phase
14's predictions in every test file, and the strict reproduction verdict failed only on official
OmniDocBench scores, which the owner accepted as a disclosed limitation (section 2); the phone
candidate met H11 against v1 (section 3); its networks were measured on two named phones (section 4);
and v1's records are unchanged (section 5).

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
| 1 | Copy the v1 bundle into the clean folder | The weights were not distributed at the time; v1.0 distributes them as a release asset (Phase 23) | script, from the worktree's copy | hash matched | none |
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
the original work. The weights and the reference file came by hand, so the walkthrough did not test the weight
download. Phase 23 tested it against a local server in the release rehearsal (GW-E2E-022,
`reports/package/GW-E2E-022_fetch.json`); the fetch from the published asset is GW-E2E-023, planned
for after publication. GDC and GDC-scan cannot be fetched by anyone else. The evaluation is on validation pages.

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
| Selection | GW-EFF-013 | selection rule in `research/hypotheses.md` | `fused-single-line` selected, the only arm to meet all three conditions: on the SM-M526B 8,111.9 ms per page and 919.9 MiB peak against `fused`'s 12,824.4 ms and 1,645.3 MiB; `fused-int8` met speed (8,522.2 ms) but not memory (1,814.9 MiB) | `reports/final_validation/candidate/GW-EFF-013/selection.md`, `accuracy.md` |
| Bundle and equivalence | GW-EFF-014 | Phase 17's tolerances | passed: map 7.33e-06, logits 2.78e-04, 5,000 of 5,000 lines identical, page CER 24.321% against 24.320%; manifest SHA-256 `97a41168…` | `reports/final_validation/candidate/GW-EFF-014/equivalence.json` |
| Validation gate, crops | GW-REC-044 | H11's test met | met: line CER 6.507% for the candidate and for v1 on 74,500 lines | `reports/final_validation/candidate/validation_gate.json` |
| Validation gate, pages | GW-E2E-019 | H11's two tests met | met: page CER 27.575% and E2E F1 0.5824 for both on 1,715 pages | `reports/final_validation/candidate/validation_gate.json` |
| Test crops | GW-REC-045 | once per set, logged | done: 5 sets, one logged access each, no resume; hashes committed before scoring | `reports/final_validation/candidate/prediction_hashes.json` |
| Test pages | GW-E2E-020 | once per set, logged | done: 5 sets, one logged access each, no resume; hashes committed before scoring | `reports/final_validation/candidate/prediction_hashes.json` |
| Scoring and H11 | GW-E2E-021 | six tests, all required; the three on GDC also by source document | **met**, six of six, and the GDC source-document check | `reports/final_validation/candidate/h11.md` |

On the laptop, `fused-single-line` gave `fused`'s text on all 5,000 lines (its `lines.json`) and
`fused`'s page CER and E2E F1 on the 400 pages, and `fused-int8` stayed within its margin on all three
tests. The laptop sitting's page times and peaks are recorded in
`reports/final_validation/candidate/GW-EFF-013/laptop.md` as information: the selection rule judges
speed and memory on the phone. The pages and profile reports carry the recognizer's batching label; on a
page the pipeline hands the recognizer all of that page's crops in one call, and the 256 crops per call
apply to the line evaluation only.

### 3.1 H11 (GW-E2E-021)

From `reports/final_validation/candidate/h11.md`. Margin 0.05 points (`fused-single-line` is not an
INT8 arm), strict track. The interval is in points, for the candidate less the margin against v1;
CER tests are met below 0 and E2E F1 above 0.

| Set | Test | Units | Candidate | v1 | Interval (points) | Verdict |
|---|---|---|---|---|---|---|
| HierText test | line CER | 67,574 lines | 0.0603 | 0.0603 | -0.050 [-0.050, -0.050] | met |
| HierText test | page CER | 1,634 pages | 0.2516 | 0.2516 | -0.049 [-0.050, -0.048] | met |
| HierText test | E2E F1 | 1,634 pages | 0.6018 | 0.6018 | +0.050 [+0.048, +0.050] | met |
| GDC | line CER | 8,372 lines | 0.0417 | 0.0417 | -0.050 [-0.050, -0.050]; by document the same | met |
| GDC | page CER | 106 pages | 0.2235 | 0.2235 | -0.050 [-0.050, -0.050]; by document the same | met |
| GDC | E2E F1 | 106 pages | 0.6936 | 0.6936 | +0.050 [+0.050, +0.050]; by document the same | met |

On all six the candidate's value equals v1's at the printed precision. H11 is a non-inferiority rule:
it does not say the two builds give the same text on every line, and that was not compared on test
data.

### 3.2 The candidate on every test set (GW-REC-045, GW-E2E-020)

From `reports/final_validation/candidate/results.md`, strict track, 95% bootstrap intervals. Every
figure here is descriptive; only the six tests above decide. v1's figures stay in
`reports/external/` and are not repeated here.

| Set | Units | Line CER % | Bag-of-lines page CER % | Order-aware page CER % | E2E F1 % |
|---|---|---|---|---|---|
| HierText test | 67,574 lines, 1,634 pages | 6.03 [5.89, 6.18] | 25.16 [23.44, 26.79] | n/a | 60.18 [58.71, 61.72] |
| GDC | 8,372 lines, 106 pages | 4.17 [3.72, 4.67] | 22.35 [15.87, 30.11] | 25.37 [20.00, 30.80] | 69.36 [60.69, 77.48] |
| FUNSD | 2,647 lines, 50 pages | 7.03 [5.71, 8.66] | 32.05 [24.28, 39.85] | n/a | 62.81 [59.06, 66.48] |
| GDC-scan | 776 lines, 28 pages | 1.02 [0.85, 1.20] | 4.60 [2.10, 8.21] | 11.34 [7.10, 16.70] | 89.84 [85.88, 93.62] |
| OmniDocBench | 755 pages | n/a | n/a | 27.28 [23.95, 30.57] | n/a |

IIIT5K: strict CER 44.27 [42.46, 46.16]% on 3,000 words and alphanumeric word accuracy 91.47 [90.43,
92.47]%. Official scores for the candidate only: HierText line level Det-Fscore 0.7837 and E2E-Fscore
0.5038 on 1,634 pages; OmniDocBench text-block edit distance over the whole set 0.1892 on 755 pages.
That OmniDocBench figure is not compared with v1's: the evaluator's time limit sent 2 of the
candidate's pages to its fallback scoring (`results.json`, `match_debug`), against 1 of v1's in Phase
14, the cause described in section 2.1. `results.md` also prints other official OmniDocBench fields
than Phase 14's `official.md` does (its page average is `edit_sample_avg`, and its reading-order
column is the whole-set edit); `results.json` holds every field. Calibration at T = 2.35, unchanged:
HierText test ECE 0.0920 against 0.2231 at T = 1. GDC-scan contributes numbers only.

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
| Mean and p95 network time per page over both runs, peak `VmHWM` | `fused` 12,824.4 ms (p95 25,756.3), 1,645.3 MiB; `fused-int8` 8,522.2 ms (17,568.2), 1,814.9 MiB; `fused-single-line` 8,111.9 ms (18,298.9), 919.9 MiB | `fused-single-line` 10,987.5 ms (p95 23,283.8), 409.0 MiB |
| Each run's mean time per page, runs 1 and 2 | `fused` 13,515.6 and 12,133.2 ms; `fused-int8` 8,395.7 and 8,648.7; `fused-single-line` 8,236.9 and 7,986.8 | 10,474.4 and 11,500.6 ms |
| Thermal state, start and end of each run | none to none in every run | light to moderate in run 1, moderate to moderate in run 2 |
| Logits and greedy text against the laptop (separate pass, 1,488 lines) | `fused` 1.28e-04, 1,488 of 1,488 identical; `fused-single-line` 1.03e-04, 1,488 of 1,488; `fused-int8` 5.27e+00, 1,444 of 1,488, reported and not judged | `fused-single-line` 1.18e-04, 1,488 of 1,488 |
| Part b | passed; these runs decided the selection | passed; a second named device, it changes no decision |
| Report | `reports/final_validation/GW-EFF-015/phone_sm-m526b.md` | `reports/final_validation/GW-EFF-015/phone_moto-g35-5g.md` |

The phone app uses the stock `onnxruntime-android` package, unchanged and checked by its SHA-256. The
package's network permissions and its telemetry start-up provider are removed at manifest merge, and
the app turns telemetry off, so the APK asks for no permissions. `scripts/phase22_android.py run`
refuses an APK that asks for any.

**What the phone figures are.** Network time and memory only: a page's time is the sum of its
`session.run` calls over 100 timed validation pages per run, and the memory is the app's `VmHWM`.
Decoding, post-processing, cropping and reading order were not ported, so a full page on a phone takes
longer than these figures, by an amount not measured. The 1,000 seeded lines at batch 1 are in the JSON
reports. Laptop figures were used only to rank the arms (`reports/final_validation/candidate/GW-EFF-013/laptop.md`).
The validation tensors were deleted from both phones after the runs. No phone target is set:
ADR-0011's targets are for the laptop CPU and stay missed for v1.

**Unplanned run before registration (2026-11-19).** While the check's code was being tested, a test run
reached the M52, which had just been connected, although the tester expected no phone. It installed
the app from a throwaway commit, ran the fused recognizer on the one validation tensor, and the app and
its files were then removed. The session was created and the output had the expected shape. It is not
the result of GW-EFF-015: the official run below is made from the committed build, and the pass rule was
fixed before it and is not changed. See `research/failures.md`.

## 5. Closure checks

From `reports/final_validation/closure.md` (`scripts/phase22_closure.py`), after every pass of the
phase. All five passed.

| Check | Result |
|---|---|
| Hash lists of the Phase 14 and baseline folders match, before (`reports/final_validation/v1_files_before.json`) and after | passed: 8,062 of 8,062 files, none missing, changed or added |
| OmniDocBench evaluator's `result/` file list unchanged across each pass | passed: the same 15 file names as in the lists taken before Pass A and Pass B, after the candidate's scoring too |
| Phase 14 outputs match `phase14_outputs.json` | passed: 68 of 68 entries, 14 of them with a run-metadata hash |
| The access log at the start of Phase 22 is an exact prefix of the current log | passed: 263 rows at `phase-21`, 188 added in Phase 22 with no resumed row, the number the plan expected |
| No difference from `phase-21` under the protected paths | passed: `git diff --exit-code` exit code 0 |

## 6. Exit criteria

| Criterion (IMPLEMENTATION_PLAN, Phase 22) | Status | Where |
|---|---|---|
| Walkthrough log, every manual step recorded | met | section 1 |
| Reproduced metrics within the stated tolerance of the reported ones | not met under the strict rule (official OmniDocBench only); closed by the owner's ruling of 2026-11-21, disclosed | section 2 |
| The fused-attention build's results under its own IDs | met: GW-EFF-013 to 015, GW-REC-044 and 045, GW-E2E-019 to 021 | sections 3, 4 |
| The phone gap recorded as measured on a named device or as still open | measured for the candidate's networks on the Samsung Galaxy M52 5G and the Moto G35 5G; the full page on a phone is not measured | section 4 |
