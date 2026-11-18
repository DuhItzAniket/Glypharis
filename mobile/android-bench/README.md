# Android bench

A single-purpose Android app for Phase 22's phone work (GW-EFF-015). It implements part a, the
operator check: create an ONNX Runtime session on the fused recognizer, run one validation tensor,
and write what happened. The APK asks for no permissions, so it has no network access, and it has no
account and no launcher entry. It reads and writes only its own external files folder.

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
| `operator_check` | `model`, `input`, `output`, `result` (`--es`, plain file names), `threads` (`--ei`) | runs the check below |

```bash
adb shell am start -W -n org.glypharis.bench/.BenchActivity --es mode ready
adb shell am start -W -n org.glypharis.bench/.BenchActivity --es mode operator_check \
  --es model recognizer.fused.onnx --es input input.gwt --es output logits.gwt \
  --es result result.json --ei threads 4
```

Files live in `/sdcard/Android/data/org.glypharis.bench/files`. A name with a path separator or a
leading dot is refused. The activity closes itself when the result is written. A further mode is one
more branch in `BenchActivity.dispatch`.

## The check

The session uses the options of `glypharis.efficiency.threads.ort_session`: sequential execution,
all graph optimisations, CPU memory arena off, memory patterns off, the requested intra-op thread
count, spinning at ONNX Runtime's default, CPU provider only.

The result file is JSON, written atomically:

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

No time is measured or recorded. `bad_request` means the app could not read its request or files and
never reached the session, so it says nothing about the model.

## GWT1 tensor files

Inputs and outputs use one small format. All integers and values are little-endian.

| Field | Type |
|---|---|
| magic | 4 bytes, `GWT1` |
| tensor count | u32 |
| *for each tensor:* | |
| name length, name | u32, UTF-8 bytes |
| element type | u32: 1 = float32, 2 = int64 |
| rank | u32 |
| dimensions | rank x i64 |
| payload length in bytes | u64 |
| payload | elements in C order |

The input file holds `images` (float32, `(1, 1, height, width)`) and `widths` (int64, `(1,)`), the
recognizer's two inputs. The output file holds `logits` (float32, `(T, 1, classes)`).
