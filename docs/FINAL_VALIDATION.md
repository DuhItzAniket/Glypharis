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
| Date | not run |
| Commit of the clone | not run |
| OS, Python, GPU driver | not run |
| Environment report | not run |

### 1.2 Steps in order

| Step | Command or action | Result | Evidence |
|---|---|---|---|
| Clone | `git clone`, then `git checkout --detach 420e095` | not run | |
| Install | virtual environment, PyTorch, `requirements/lock.txt`, editable package, `pip check` | not run | |
| Environment and tests | `env_report.py`, `pytest`, `validate_configs.py` | not run | |
| Download | HierText, MobileNetV3 weights, FUNSD, IIIT5K | not run | |
| Verify | `download_datasets.py --verify`; only `downloaded_utc` may change | not run | |
| Prepare | `prepare_hiertext.py`, `prepare_benchmarks.py`, `build_eval_crops.py`; no derived manifest differs | not run | |
| Load the model | the v1 bundle, placed by hand | not run | |
| Infer | `glypharis-ocr` on one validation page, both backends, from outside the repository | not run | |
| Evaluate | `package_check.py --v1`, 400 validation pages | not run | |

### 1.3 Manual steps

One row for every action not done by a listed command, and for every retry or deviation. The first four
are known in advance.

| # | Manual step | Why it was needed | Done by | Outcome | Retry or deviation |
|---|---|---|---|---|---|
| 1 | Copy the v1 bundle into the clean folder | The weights are not distributed before Phase 23 | | not run | |
| 2 | Copy the GW-E2E-006 reference file (`hiertext_val.jsonl`, SHA-256 `c8520576…`) | It is an evaluation output and is not in git | | not run | |
| 3 | Set `GLYPHARIS_MODELS` | The package has to find the copied bundle | | not run | |
| 4 | Run `glypharis-ocr` on one page with each backend | Smoke check of the installed command | | not run | |

### 1.4 Findings and fixes

| Finding | Fix | Commit | Steps repeated |
|---|---|---|---|
| not run | | | |

### 1.5 What the walkthrough does not show

To be confirmed in the write-up: it shares the OS, driver, base Python and git config with the original
work, and the weights came by hand, so the weight download path is untested until Phase 23.

## 2. Reproduction of v1

| Check | ID | Rule | Result | Report |
|---|---|---|---|---|
| Rehearsal, validation crops | GW-REC-042 | every point value within 0.0005 | not run | |
| Rehearsal, validation pages | GW-E2E-014 | every point value within 0.0005 | not run | |
| Pass A, re-score of the saved files | GW-E2E-015 | identical numbers | not run | |
| Re-run, test crops | GW-REC-043 | once per set; outputs complete and hashed | not run | |
| Re-run, test pages | GW-E2E-016 | once per set; outputs complete and hashed | not run | |
| Re-run, oracle | GW-E2E-017 | once per set; outputs complete and hashed | not run | |
| Pass B and tolerance verdict | GW-E2E-018 | every point value within 0.05 points, counts equal | not run | |

### 2.1 Differences found

Not run.

## 3. Phone candidate

A new version, not v1. Its results sit beside v1's and replace none of them.

| Step | ID | Rule | Result | Report |
|---|---|---|---|---|
| Android operator check | GW-EFF-015, part a | session and one tensor on the phone | not run | |
| Selection | GW-EFF-013 | selection rule in `research/hypotheses.md` | not run | |
| Bundle and equivalence | GW-EFF-014 | Phase 17's tolerances | not run | |
| Validation gate, crops | GW-REC-044 | H11's test met | not run | |
| Validation gate, pages | GW-E2E-019 | H11's two tests met | not run | |
| Test crops | GW-REC-045 | once per set, logged | not run | |
| Test pages | GW-E2E-020 | once per set, logged | not run | |
| Scoring and H11 | GW-E2E-021 | six tests, all required; the three on GDC also by source document | not run | |

## 4. Phone measurement (GW-EFF-015)

Each column is filled from the device and committed before that phone is measured. No serial number or
account is recorded.

| Condition | Samsung Galaxy M52 5G | Moto G35 5G |
|---|---|---|
| SoC | not read | not read |
| RAM | not read | not read |
| Android build | not read | not read |
| ABI | not read | not read |
| Listed by `adb devices` as authorized | not checked | not checked |
| `onnxruntime-android` package and SHA-256 | not recorded | not recorded |
| JDK, Gradle and Android Gradle Plugin versions | not recorded | not recorded |
| Session options (sequential execution, all graph optimisations, CPU arena off, memory patterns off) | not set | not set |
| Thread count, spinning setting | not set | not set |
| Process rule (a new app process per arm and run, force-stopped between runs) | not confirmed | not confirmed |
| Charge state, starting battery temperature | not recorded | not recorded |
| Airplane mode, screen and foreground state | not recorded | not recorded |
| Operator check | not run | not run |
| Mean and p95 network time, peak `VmHWM` | not run | not run |
| Each run's mean time | not run | not run |
| Thermal state, start and end of each run | not run | not run |
| Logits and greedy text against the laptop (separate pass) | not run | not run |

## 5. Closure checks

| Check | Result |
|---|---|
| Hash lists of the Phase 14 and baseline folders match, before (`reports/final_validation/v1_files_before.json`) and after | not run |
| OmniDocBench evaluator's `result/` file list unchanged across each pass | not run |
| Phase 14 outputs match `phase14_outputs.json` | not run |
| The access log at the start of Phase 22 is an exact prefix of the current log | not run |
| No difference from `phase-21` under the protected paths | not run |
