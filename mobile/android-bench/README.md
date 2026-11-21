# Android bench

A single-purpose Android app for Phase 22's phone work (GW-EFF-015). Part a is the operator check:
create an ONNX Runtime session on the fused recognizer, run one validation tensor, and write what
happened. Part b replays the laptop's exact validation network inputs for each candidate arm: a
timing run times every `session.run` call and reads the process's peak memory, and an output check,
in a process of its own, writes the recognizer's logits for comparison with the laptop. The APK asks
for no permissions, so it has no network access, and it has no account and no launcher entry. It
reads and writes only its own external files folder.

The `onnxruntime-android` package is used as published. Its manifest asks for `INTERNET` and
`ACCESS_NETWORK_STATE` and starts a telemetry client through `ai.onnxruntime.TelemetryInitializer`;
`AndroidManifest.xml` removes all three at manifest merge and the app also calls
`setTelemetry(false)`. `phase22_android.py run` refuses an APK that still has any of them. The activity
needs `android.permission.DUMP`, which the adb shell holds and ordinary apps do not.

`scripts/phase22_android.py` drives it; see that script's docstring for the steps.

## Build

JDK 21, the Android SDK and every cache are given for the build command only and stay outside the
repository. No machine path is committed, and no `local.properties` is written.

```bash
JAVA_HOME=<JDK 21> ANDROID_HOME=<Android SDK> \
  GRADLE_USER_HOME=<folder outside the repository> \
  ANDROID_USER_HOME=<folder outside the repository> \
  python scripts/phase22_android.py build
```

From the repository root, on a tree with nothing changed or untracked, this runs
`gradlew --no-daemon clean :app:assembleDebug` here, checks `git status` again and records the commit,
the result and the APK's hash for `run`.

The APK is `app/build/outputs/apk/debug/app-debug.apk`, for `arm64-v8a` only. The build stops unless
it runs on JDK 21 and `onnxruntime-android` has the SHA-256 in `gradle.properties`.

## Starting it

The app is started by adb with an explicit component, one mode per start:

| `mode` | Extras | What it does |
|---|---|---|
| `ready` (default) | none | creates the external files folder and writes `ready.json` |
| `operator_check` | `model`, `input`, `output`, `result` (`--es`), `threads` (`--ei`) | part a, the operator check |
| `timing` | `plan`, `result` (`--es`), `threads`, `run` (`--ei`) | part b, one timing run of one arm |
| `check` | `plan`, `result` (`--es`), `threads` (`--ei`) | part b, the output check of one arm |

String extras are plain file names. A missing string extra, or a thread count or run number that is
missing or below 1, gives `bad_request`.

```bash
adb shell am start -W -n org.glypharis.bench/.BenchActivity --es mode ready
adb shell am start -W -n org.glypharis.bench/.BenchActivity --es mode operator_check \
  --es model recognizer.fused.onnx --es input input.gwt --es output logits.gwt \
  --es result result.json --ei threads 4
adb shell am start -W -n org.glypharis.bench/.BenchActivity --es mode timing \
  --es plan plan_fused.json --es result timing_fused_run1.json --ei threads 4 --ei run 1
adb shell am start -W -n org.glypharis.bench/.BenchActivity --es mode check \
  --es plan check_fused.json --es result checkresult_fused.json --ei threads 4
```

Files live in `/sdcard/Android/data/org.glypharis.bench/files`, in one flat folder. A name with a
path separator or a leading dot is refused. The activity closes itself when the result is written. A
further mode is one more branch in `BenchActivity.dispatch`.

| File | Written by | Holds |
|---|---|---|
| `detector.onnx`, `recognizer.fused.onnx`, `recognizer.fused.int8.onnx` | the script | the models |
| `t_<first 40 hex digits of the file's SHA-256>.gwt` | the script | the inputs of one `session.run` call; a file shared by arms is stored once |
| `plan_<arm>.json`, `check_<arm>.json` | the script | an arm's timing plan and check plan |
| `timing_<arm>_run<r>.json`, `checkresult_<arm>.json` | the app | results |
| `c_<arm>_<i>.gwt` | the app | the logits of check call `i` |

Every result is JSON with `schema` `glypharis-android-bench-1`, written atomically through a `.tmp`
file; a failure is a result like any other. `bad_request` means the app could not read its request
or its files, or the plan names an output the model does not have, so it says nothing about the
model.

## Session options

Every session of every mode is created with one function, `BenchSession.options`: the options of
`glypharis.efficiency.threads.ort_session`, which are sequential execution, all graph
optimisations, CPU memory arena off, memory patterns off, the requested intra-op thread count,
spinning at ONNX Runtime's default, CPU provider only. Each result records them under
`session_options`, in the names `threads.session_options` reports: `providers`,
`intra_op_num_threads`, `execution_mode`, `graph_optimization_level`, `enable_cpu_mem_arena`,
`enable_mem_pattern` and `allow_spinning` (`"default"`).

## The operator check (part a)

The result file:

| Field | Meaning |
|---|---|
| `schema`, `mode` | `glypharis-android-bench-1`, `operator_check` |
| `status` | `ok`, `session_failed`, `runtime_failed`, `run_failed` or `bad_request` |
| `onnxruntime_version` | the version the native library reports |
| `model_sha256`, `input_sha256` | hashes of the files as the app read them |
| `session_created` | whether the session was created |
| `session_error`, `run_error`, `request_error` | `type`, `message` (exact text, with the app's folder shown as `<app files>`) and `code`, or null |
| `session_options` | the options the app set |
| `input_names`, `output_names` | as the session reports them |
| `output_shape`, `logits_finite` | the output's shape and whether every value is finite |

No time is measured or recorded. The input file holds `images` (float32, `(1, 1, height, width)`)
and `widths` (int64, `(1,)`); the output file holds `logits` (float32, `(T, 1, classes)`).

## Timing runs (part b)

One start times one arm once. Each arm and run is a new process, force-stopped between runs, because
`VmHWM` never falls within a process; the app refuses a second timing or check pass in a process
that has already started one.

Before any session exists the app checks the request and every file it will read: the thread count
is at least 1 and equal to the plan's, the run number is at least 1, the plan fits its schema, each
model file is present with the plan's SHA-256, and each tensor file a call names is present with the
plan's size and SHA-256. It then creates the detector and recognizer sessions and runs the items in
the plan's order. Before the first item it allocates one direct little-endian input buffer of the
plan's `buffer_bytes`, writes every page of it once so that it is resident, collects the heap
(`System.gc()`) and records `VmHWM` and `VmRSS`; nothing is timed there. For each call it reads the
call's GWT1 file into that buffer (the header field by field, then each payload straight from the
file's channel at the next 8-byte boundary, with no copy on the heap), wraps the payloads as ONNX
Runtime tensors without a copy, and takes `System.nanoTime()` right before and right after
`session.run(inputs, setOf(output))`. Nothing else is timed. The output and the input tensors are
closed straight after the second time stamp, so the run holds one call's inputs at a time and no
logits. `buffer_bytes` is the same in every arm's plan (the largest call of any arm), so the buffer
adds the same memory to every arm, and a call leaves no garbage behind. Right after the last item
the app reads `VmHWM` and `VmRSS` from `/proc/self/status`.

The timing plan, `plan_<arm>.json`, is written by the script:

```json
{
  "schema": "glypharis-phone-plan-1",
  "experiment": "GW-EFF-015",
  "arm": "fused",
  "threads": 4,
  "models": {
    "detector":   {"file": "detector.onnx", "sha256": "...", "output": "prob"},
    "recognizer": {"file": "recognizer.fused.onnx", "sha256": "...", "output": "logits"}
  },
  "buffer_bytes": 7077888,
  "warmup_pages": 10, "timed_pages": 100, "lines": 1000,
  "items": [
    {"kind": "page", "key": "<doc id>", "warmup": true,
     "calls": [{"model": "detector", "tensor": "t_....gwt"}, {"model": "recognizer", "tensor": "t_....gwt"}]},
    {"kind": "line", "key": "<line key>", "warmup": false,
     "calls": [{"model": "recognizer", "tensor": "t_....gwt"}]}
  ],
  "tensors": {"t_....gwt": {"sha256": "<full 64 hex>", "bytes": 123}}
}
```

| Field | Meaning |
|---|---|
| `schema` | `glypharis-phone-plan-1` |
| `experiment`, `arm` | copied into the result |
| `threads` | the intra-op thread count; the request's must equal it |
| `models` | exactly `detector` and `recognizer`, each with its `file`, the file's `sha256` and the `output` to ask for |
| `buffer_bytes` | the size of the one input buffer every call is read into, each payload 8-byte aligned; the same in every arm's plan |
| `warmup_pages`, `timed_pages`, `lines` | the item counts, which must add up to the number of items |
| `items` | in run order: the warm-up pages (`warmup: true`), the timed pages, then the lines (both `warmup: false`). A page's `calls` are its detector call, then its recognizer calls; a line's is one recognizer call |
| `tensors` | every tensor file a call names, with its `sha256` (64 lower-case hex digits) and `bytes` |

A field of the wrong type (a thread count written as text, say) or an item out of this order refuses
the whole plan.

The timing result, `timing_<arm>_run<r>.json`:

| Field | Meaning |
|---|---|
| `schema`, `mode` | `glypharis-android-bench-1`, `timing` |
| `status` | `ok`, `bad_request`, `runtime_failed`, `session_failed` or `run_failed` |
| `failed_model` | `detector` or `recognizer` when that model's file, session or call failed, else null |
| `request_error`, `session_error`, `run_error` | `type`, `message` and, from ONNX Runtime, `code`, or null; a failure during the items adds `item` and `call` (indices from 0) and `tensor` |
| `pid` | `android.os.Process.myPid()` |
| `experiment`, `arm` | from the plan |
| `run`, `threads`, `plan` | from the request |
| `plan_sha256` | the plan file's SHA-256 |
| `model_sha256` | per model (`detector`, `recognizer`), the SHA-256 of the file as the app read it |
| `tensors_checked` | how many tensor files were checked against the plan |
| `onnxruntime_version`, `session_options` | the native library's version, the options set |
| `session_created` | whether both sessions were created |
| `input_names`, `output_names` | per model, as its session reports them |
| `warmup_pages`, `timed_pages`, `lines` | the plan's counts |
| `interactive_start`, `thermal_start`, `battery_start` | the phone's state right before the first item |
| `items_done` | how many items finished |
| `buffer_bytes` | from the plan |
| `vm_hwm_kb_before_items`, `vm_rss_kb_before_items` | `VmHWM` and `VmRSS` in kB after the buffer is allocated and the heap collected, before the first item |
| `vm_hwm_kb`, `vm_rss_kb` | `VmHWM` and `VmRSS` in kB, read right after the last item |
| `interactive_end`, `thermal_end`, `battery_end` | the phone's state after `VmHWM` was read |
| `items` | per finished item, in run order: `kind`, `key`, `warmup`, `network_ms` (the sum of its calls), `detector_ms` and `recognizer_ms` (the sums of that model's calls, 0 when it has none) and `calls_ms` (each call, in the plan's order) |

Times are in milliseconds, from `System.nanoTime()` differences; a whole number may be written
without a decimal point. The phone's state is read with no permission:

- `thermal_*`: `status` and `status_name` from `PowerManager.currentThermalStatus` (`none`,
  `light`, `moderate`, `severe`, `critical`, `emergency` or `shutdown`; API 29 and later), and
  `headroom` from `getThermalHeadroom(0)` (API 30 and later; null when the phone gives NaN);
- `battery_*`, from the sticky `ACTION_BATTERY_CHANGED` intent: `temperature_c`
  (`EXTRA_TEMPERATURE` / 10), `level_percent`, and `plugged` and `status` as `BatteryManager`'s
  codes;
- `interactive_*`: `PowerManager.isInteractive`.

A value the phone does not give is null, and `thermal_*` and `battery_*` carry an `error` that says
why a reading failed (null otherwise).

## Output checks (part b)

The logit and greedy-text check is a separate pass, one new process per arm, and gives no time or
memory figure. The app checks the request and every file as a timing run does (and that each output
name is a plain name, used once, and not a file the check reads), creates the recognizer session
only, runs each call in order, and writes each call's logits to the file the call names: a GWT1 file
holding one float32 tensor named `logits`. The laptop compares them.

The check plan, `check_<arm>.json`, is written by the script:

```json
{
  "schema": "glypharis-phone-check-1",
  "experiment": "GW-EFF-015",
  "arm": "fused",
  "threads": 4,
  "model": {"file": "recognizer.fused.onnx", "sha256": "...", "output": "logits"},
  "calls": [{"tensor": "t_....gwt", "output": "c_fused_0.gwt", "source": "page <doc id> call 1"}],
  "tensors": {"t_....gwt": {"sha256": "...", "bytes": 123}}
}
```

Its calls are every recognizer call of the 10 warm-up pages, then the 1,000 lines at batch 1.

The check result, `checkresult_<arm>.json`, has `schema`, `mode` (`check`), `status`,
`failed_model` (`recognizer` or null), `request_error`, `session_error`, `run_error` (a failure
during the calls adds `call` and `tensor`; an output file that cannot be written is a `request_error`
with status `bad_request`, not a model failure), `pid`, `experiment`, `arm`, `threads`, `plan`,
`plan_sha256`, `model_sha256` (`recognizer`), `tensors_checked`, `onnxruntime_version`,
`session_options`, `session_created`, `input_names` and `output_names` (`recognizer`) as a timing
result does, and:

| Field | Meaning |
|---|---|
| `calls_done` | how many calls finished |
| `all_finite` | whether every finished call's logits are all finite |
| `calls` | per finished call, in plan order: `tensor`, `output` and `source` from the plan, `output_shape` and `logits_finite` |

## GWT1 tensor files

Inputs and outputs use one small format. All integers and values are little-endian.

| Field | Type |
|---|---|
| magic | 4 bytes, `GWT1` |
| tensor count | u32 |
| *for each tensor:* | |
| name length, name | u32, UTF-8 bytes |
| element type | u32: 1 = float32, 2 = int64, 3 = uint8 (one byte per element) |
| rank | u32 |
| dimensions | rank x i64 |
| payload length in bytes | u64 |
| payload | elements in C order |

One file holds every input tensor of one `session.run` call: the detector's `image` (uint8,
`(1, 3, H, W)`, about 5 MB for a page), or the recognizer's `images` (float32, `(N, 1, 48, W)`) and
`widths` (int64, `(N,)`). An output file holds `logits` (float32, time-major `(T, N, classes)`). The
detector's output is `prob`.
