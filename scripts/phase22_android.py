"""GW-EFF-015 on a phone (Phase 22): the operator check, part a, and the arms' timing, part b.

    python scripts/phase22_android.py prepare [--threads 4]
    python scripts/phase22_android.py device [--serial S]
    python scripts/phase22_android.py build
    python scripts/phase22_android.py run [--serial S] [--apk APK] [--wait-seconds 900]
    python scripts/phase22_android.py report [--device SLUG ...] [--unavailable TEXT ...]
    python scripts/phase22_android.py prepare-timing [--threads 4]
    python scripts/phase22_android.py conditions [--serial S] --label L
    python scripts/phase22_android.py push-timing [--serial S] [--apk APK]
    python scripts/phase22_android.py time-run --arm A --run R [--serial S] [--wait-seconds 7200]
    python scripts/phase22_android.py check-run --arm A [--serial S] [--wait-seconds 7200]
    python scripts/phase22_android.py phone-report --device SLUG
    python scripts/phase22_android.py select --device SLUG
    python scripts/phase22_android.py cleanup [--serial S]

Run from the repository root. Validation data only; nothing here reads a test set.

Part a (item 3): does the stock ``onnxruntime-android`` package hold the
``com.microsoft.MultiHeadAttention`` kernel the fused recognizer (GW-EFF-009) needs, and does the
phone give the laptop's output for one validation line?

- ``prepare`` takes the first registered validation line (index 0 of the export check's sorted
  seeded sample, `export_check.line_indices`, over the recognizer's validation pack named by the
  freeze record), runs it through the v1 recognizer's own input code on an ONNX Runtime session
  with the options of `glypharis.efficiency.threads.ort_session`, and writes the exact input
  tensors and the laptop's logits to `PREPARED` (git-ignored). The greedy text must equal Phase
  16's for that line when its file is present.
- ``device`` reads model, SoC, RAM, Android build and ABI from named properties and
  ``/proc/meminfo`` and writes them under `RUNS` (git-ignored). It never reads the serial number.
- ``build`` refuses a tree with anything changed or untracked, runs ``gradlew --no-daemon clean
  :app:assembleDebug`` with the caller's ``JAVA_HOME``, ``ANDROID_HOME`` and ``GRADLE_USER_HOME``,
  runs ``git status --porcelain --untracked-files=all`` again and writes `BUILD_RECORD`.
- ``run`` needs that record to name the current commit and the APK's hash, an empty status straight
  after the build and still an empty status now (``--allow-dirty`` is for rehearsals and its report
  says no-go). The APK must ask for no permissions and hold no telemetry provider. It installs the
  APK, lets the app create its folder, pushes the model and tensors, checks their hashes on the
  phone when ``sha256sum`` is there, starts the app in a new process, waits for ``result.json`` and
  pulls the results.
- ``report`` compares each phone with the laptop and writes `REPORT` once. Rule (Phase 22, item
  3): **go** when the build was clean, the session was created and the tensor gave finite logits
  of the laptop's shape, on every phone listed. A session error is **no-go**, recorded as
  ``missing_kernel`` when ONNX Runtime says a kernel or operator is missing. The largest logit
  difference over the line's own frames and the greedy text are recorded, not judged here (item 15
  judges them): above `LOGIT_NOTICE` or with different text, ``tell_owner_before_s4`` is set. A run
  in which the app could not read its files gives no verdict and no report. A phone seen by
  ``device`` or ``run`` that is not in the report must be named with ``--unavailable "<slug>:
  reason"``, and a phone whose run recorded a missing kernel cannot be left out at all.

Part b (item 15): the arms that GW-EFF-013's ``accuracy.json`` sends to the phone (``to_phone``),
timed and checked on the laptop's exact validation network inputs. Each arm is v1's detector and
the arm's recognizer file (`glypharis.efficiency.candidate`); only ``session.run`` is ported.

- ``prepare-timing`` builds each arm as its bundle loads it (`phase22_candidate.deployed_arm`,
  hashes checked) and replaces the ``run`` of its two ONNX Runtime sessions with a recorder that
  copies the feeds and then calls the original (`RunRecorder`). It replays 10 warm-up and 100 timed
  Phase 5 profile pages as `phase16_variants.profile_pages` hands them to `GlypharisOCR.predict`,
  then the first 1,000 of the export check's seeded validation lines, each alone through
  ``recognizer.logits([crop])``, and groups the calls into items: a page's calls start with its one
  detector call, a line is one recognizer call. Each call's feeds are one GWT1 file named by its
  content, ``t_<first 40 hex of the SHA-256>.gwt``, so a tensor shared by arms is written once.
  Per arm it writes ``plan_<arm>.json`` (the timing plan), ``check_<arm>.json`` (every recognizer
  call of the warm-up pages, then the lines) and ``laptop_<arm>.json`` (each check line's frames
  and greedy text, from the recorded outputs), then ``prepare.json`` with the hashes, to `TIMING`
  (git-ignored). Every timing plan names one ``buffer_bytes``, the largest call of any arm with
  each payload 8-byte aligned (`arena_bytes`): the app reads every call into that one buffer, so
  it adds the same memory to every arm. Nothing is overwritten. ``--limit-pages N`` (the first N
  pages, one of them a warm-up page) and ``--limit-lines M`` are for a smoke test: they need
  ``--out`` and mark the output ``"smoke": true``, and no later command reads it, nor one made
  from a tree with changes.
- ``conditions --label L`` reads the battery (``dumpsys battery``), the thermal status (``dumpsys
  thermalservice``), airplane mode, screen state and wakefulness (``dumpsys power``) and the
  resumed activity (``dumpsys activity activities``) and writes ``conditions_<L>.json`` in the
  phone's folder under `RUNS`. Only named fields are kept.
- ``push-timing`` needs `BUILD_RECORD` to name the current commit and the APK, installs the APK as
  ``run`` does (`install_app`), pushes the models, every tensor file and the plans, checks every
  pushed file's hash on the phone when ``sha256sum`` is there and writes ``push.json``. Before
  ``select`` the first phone is the registered M52 (`FIRST_PHONE`); another phone takes its place
  only with ``--first-phone-reason``, which goes into the phone report and the selection.
- ``time-run --arm A --run R`` refuses any run but the next of the phone's registered order: on
  the first phone, the arms going to the phone in `candidate.ARMS` order and then reversed
  (`candidate.arm_order`); on a second phone, after ``select``, the selected arm twice. It needs
  the clean build ``run`` needs and this preparation pushed with this APK. It writes the
  conditions, starts the app in a new process with ``mode=timing``, pulls the result, force-stops
  the app and writes the conditions again. A result whose ``pid`` is an earlier run's on that
  phone is refused, and so is a ``bad_request``, which says nothing about the model: the step
  stays due, and its pulled ``<result>.part`` must be moved aside (with the conditions) before the
  step runs again. When the app's process ends before it writes a result (watched with
  ``pidof``), the tool writes a ``process_died`` result itself: the arm is not measured, or for
  ``fused`` part b fails, and the order goes on.
- ``check-run --arm A`` comes after the phone's timing runs: a new process with ``mode=check``,
  then the result and every check output are pulled. It gives no time or memory figure.
- ``phone-report --device SLUG`` writes ``phone_<slug>.json`` and ``.md`` under `PHONE_REPORTS`
  once: per arm the mean and p95 network time of the timed pages per run and over both runs (a
  page's time is the mean of its two runs), the lines' times, the peak ``VmHWM`` (the larger run)
  and the thermal state at the start and end of each run; the check against the laptop's logits,
  computed on the same tensor files by the arm's own laptop session (`laptop_session`, the plan's
  thread count), over each line's own frames (``model.frames(widths)`` clamped to T, as
  `TextRecognizer.logits` takes them): within `LOGIT_TOL` and with identical greedy text, or for
  ``fused-int8`` finite outputs of the laptop's shape with the differences reported; and on the
  first phone the speed and memory conditions of each other arm against ``fused`` on the timed
  pages (`candidate.speed_tests`, `candidate.memory_test`). This is network time and memory only:
  decoding, post-processing, cropping and reading order are not ported. Every run record must
  name the current ``push.json``: a sitting has one install, one APK and one preparation. A
  second phone's part b is judged on the arm it timed.
- ``select --device SLUG`` passes GW-EFF-013's accuracy verdicts and that first phone's report to
  `candidate.select` and writes `SELECTION` and its ``.md`` once.

``cleanup`` deletes the pushed files, the timing files among them, and uninstalls the app.

With no phone or with several, every command that needs one refuses; ``--serial`` picks one for
that command only and is never written or printed. All adb calls go through `run_adb`. The app, its
modes and its tensor file format (GWT1) are described in ``mobile/android-bench/README.md``. Part a
measures no time.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from glypharis.data.download import sha256_file
from glypharis.data.text import Charset
from glypharis.efficiency.files import write_json_atomic, write_text_atomic

EXPERIMENT: Final = "GW-EFF-015"

MODEL: Final = Path("data/processed/phase16/variants/fused-attention/recognizer.fused.onnx")
MODEL_SHA256: Final = "b0e2a5100cb465f72d2abca97fc30a47892c75d069e556791021b7b74dfc172c"
MODEL_BYTES: Final = 31_810_737
BUNDLE: Final = Path("runs/release/glypharis-v1")

WORK: Final = Path("data/processed/phase22/android")  # git-ignored
PREPARED: Final = WORK / "operator"
RUNS: Final = WORK / "runs"
BUILD_RECORD: Final = WORK / "build.json"
REPORT: Final = Path("reports/final_validation/GW-EFF-015/operator_check.json")

APP_DIR: Final = Path("mobile/android-bench")
APK: Final = APP_DIR / "app/build/outputs/apk/debug/app-debug.apk"
ORT_PACKAGE: Final = "com.microsoft.onnxruntime:onnxruntime-android"
ORT_VERSION: Final = "1.31.1"
TELEMETRY_PROVIDER: Final = "ai.onnxruntime.TelemetryInitializer"
GRADLE_ARGS: Final = ("--no-daemon", "clean", ":app:assembleDebug")
BUILD_ENV: Final = ("JAVA_HOME", "ANDROID_HOME", "GRADLE_USER_HOME")

APP_ID: Final = "org.glypharis.bench"
ACTIVITY: Final = f"{APP_ID}/.BenchActivity"
PHONE_DIR: Final = f"/sdcard/Android/data/{APP_ID}/files"
PHONE_MODEL: Final = "recognizer.fused.onnx"
PHONE_INPUT: Final = "input.gwt"
PHONE_LOGITS: Final = "logits.gwt"
PHONE_RESULT: Final = "result.json"
PHONE_READY: Final = "ready.json"
PHONE_FILES: Final = (PHONE_MODEL, PHONE_INPUT, PHONE_LOGITS, PHONE_RESULT, PHONE_READY)
REQUIRED_ABI: Final = "arm64-v8a"
APK_PACKAGE_LINE: Final = f"package: {APP_ID}"

DEFAULT_THREADS: Final = 4  # intra-op threads on both sides; they do not change the outputs
LOGIT_NOTICE: Final = 1e-3  # item 3: above this the owner is told before S4; item 15 judges it
READY_SECONDS: Final = 60.0
PUSH_SECONDS: Final = 600.0
ADB_SECONDS: Final = 60.0
POLL_SECONDS: float = 1.0

GO: Final = "go"
NO_GO: Final = "no-go"
RULE: Final = (
    "go: the tree was clean straight after the build, and on every phone listed the session was "
    "created and the tensor gave finite logits of the laptop's shape. A session error is no-go."
)
MISSING_KERNEL: Final = re.compile(
    r"could not find an implementation|not a registered function/op|kernel\b.*\bnot found"
    r"|no kernel",
    re.IGNORECASE,
)
DEVICE_PATH: Final = re.compile(r"(?<![\w./-])/(?:storage|sdcard|mnt|data)/[^\s'\"]+")
SESSION_OPTION_KEYS: Final = (
    "providers",
    "intra_op_num_threads",
    "execution_mode",
    "graph_optimization_level",
    "enable_cpu_mem_arena",
    "enable_mem_pattern",
    "allow_spinning",
)

GWT_MAGIC: Final = b"GWT1"
GWT_TYPES: Final[dict[int, np.dtype[Any]]] = {
    1: np.dtype("<f4"),
    2: np.dtype("<i8"),
    3: np.dtype("u1"),
}
GWT_CODES: Final = {"f": 1, "i": 2, "u": 3}

# Part b: the phone timing and check (item 15).
TIMING: Final = WORK / "timing"  # git-ignored: tensor files, plans and the laptop's texts
ACCURACY: Final = Path("reports/final_validation/candidate/GW-EFF-013/accuracy.json")
SELECTION: Final = Path("reports/final_validation/candidate/GW-EFF-013/selection.json")
PHONE_REPORTS: Final = Path("reports/final_validation/GW-EFF-015")
DETECTOR: Final = BUNDLE / "detector.onnx"
DETECTOR_SHA256: Final = "378c1feda480c1bb1dde838dd4a5f4579a904710a7f7b222d978c583b5bddcb7"
PHONE_DETECTOR: Final = "detector.onnx"
PLAN_SCHEMA: Final = "glypharis-phone-plan-1"
CHECK_SCHEMA: Final = "glypharis-phone-check-1"
TIMING_LINES: Final = 1_000  # the first 1,000 of the export check's seeded lines, at batch 1
LOGIT_TOL: Final = 1e-3  # item 15: the phone's logits against the laptop's, each line's frames
UNJUDGED_ARMS: Final = ("fused-int8",)  # item 15: differences reported; the margin decided
LOAD_FAILURES: Final = ("session_failed", "runtime_failed")
# The app's process ended before it wrote its result (a low-memory kill or a crash). The tool
# writes that result itself; the arm is not measured or not checked, and the order goes on.
PROCESS_DIED: Final = "process_died"
DEATH_POLLS: Final = 3  # polls in a row with no app process and no result before it counts
FIRST_PHONE: Final = "sm-m526b"  # Samsung Galaxy M52 5G, the registered first phone (H11)
ARENA_ALIGN: Final = 8  # the app reads each call's payloads into one buffer, 8-byte aligned
TIMING_SECONDS: Final = 7200.0
PUSH_BATCH: Final = 64  # files per adb push or pull, and names per sha256sum call
PUSH_RECORD: Final = "push.json"
CHECK_OUTPUTS: Final = "check"  # the pulled check outputs, in the phone's folder under RUNS
TIMING_RESULT: Final = re.compile(r"timing_(?P<arm>[A-Za-z0-9-]+)_run(?P<run>\d+)\.json")
CHECK_RESULT: Final = re.compile(r"checkresult_(?P<arm>[A-Za-z0-9-]+)\.json")
TIMING_PHONE_FILES: Final = (
    "*.onnx",
    "t_*.gwt",
    "c_*.gwt",
    "plan_*.json",
    "check_*.json",
    "timing_*.json",
    "checkresult_*.json",
    "*.tmp",
)
DEVICE_STATE_KEYS: Final = (
    "thermal_start",
    "thermal_end",
    "battery_start",
    "battery_end",
    "interactive_start",
    "interactive_end",
)
LABEL: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
THERMAL_NAMES: Final = ("none", "light", "moderate", "severe", "critical", "emergency", "shutdown")
SCOPE: Final = (
    "Network time and memory only: a page's or a line's time is the sum of its session.run calls "
    "and the memory is the app's VmHWM; decoding, post-processing, cropping and reading order are "
    "not ported."
)
CHECK_RULE: Final = (
    "item 15: the sessions load, the logits are within 1e-3 of the laptop's for the same arm over "
    "each line's own frames and the greedy text is identical; for fused-int8 the differences are "
    "reported and not judged, and the check passes when its sessions load and every output is "
    "finite and of the laptop's shape"
)


# ------------------------------------------------------------------------------ tensor files


def gwt_bytes(tensors: Mapping[str, np.ndarray]) -> bytes:
    """`tensors` as the bytes of a GWT1 file (``mobile/android-bench/README.md``): floats as
    float32, signed integers as int64 and unsigned bytes as uint8."""
    parts = [GWT_MAGIC, len(tensors).to_bytes(4, "little")]
    for name, array in tensors.items():
        code = GWT_CODES[array.dtype.kind]
        if code == 3 and array.dtype.itemsize != 1:
            raise ValueError(
                f"{name}: unsigned integers are stored as uint8 only, not {array.dtype}"
            )
        raw = array.astype(GWT_TYPES[code]).tobytes(order="C")
        encoded = name.encode("utf-8")
        parts += [len(encoded).to_bytes(4, "little"), encoded, code.to_bytes(4, "little")]
        parts.append(array.ndim.to_bytes(4, "little"))
        parts += [int(d).to_bytes(8, "little", signed=True) for d in array.shape]
        parts += [len(raw).to_bytes(8, "little"), raw]
    return b"".join(parts)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_gwt(path: Path, tensors: Mapping[str, np.ndarray]) -> None:
    """`tensors` as a GWT1 file, moved into place whole."""
    _write_bytes(path, gwt_bytes(tensors))


def parse_gwt(data: bytes, label: str = "the tensor file") -> dict[str, np.ndarray]:
    """The tensors of GWT1 bytes; `ValueError` when they are not a complete file."""
    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        if n < 0 or pos + n > len(data):
            raise ValueError(f"{label} is cut short")
        chunk = data[pos : pos + n]
        pos += n
        return chunk

    def number(n: int, signed: bool = False) -> int:
        return int.from_bytes(take(n), "little", signed=signed)

    if take(4) != GWT_MAGIC:
        raise ValueError(f"{label} is not a GWT1 file")
    out: dict[str, np.ndarray] = {}
    for _ in range(number(4)):
        name = take(number(4)).decode("utf-8")
        code = number(4)
        if code not in GWT_TYPES:
            raise ValueError(f"{label}: unknown element type {code}")
        shape = [number(8, signed=True) for _ in range(number(4))]
        raw = take(number(8))
        dtype = GWT_TYPES[code]
        if any(d < 0 for d in shape) or len(raw) != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"{label}: {name} does not fit its shape {shape}")
        out[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if pos != len(data):
        raise ValueError(f"{label} has trailing bytes")
    return out


def read_gwt(path: Path) -> dict[str, np.ndarray]:
    """The tensors of a GWT1 file; `ValueError` when it is not a complete one."""
    return parse_gwt(path.read_bytes(), path.name)


# ------------------------------------------------------------------------------------- common


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


def _refuse_existing(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise SystemExit(f"{path.as_posix()} exists: move it aside first")


def _git(*args: str) -> str:
    """Output of a git command, stripped. A failure stops the script: an unreadable status must
    not pass for a clean tree."""
    proc = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(f"git {args[0]} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def tree_changes() -> str:
    return _git("status", "--porcelain", "--untracked-files=all")


def build_state(apk: Path) -> dict[str, Any]:
    """Item 3's build condition for ``apk``, from what `cmd_build` recorded. The record counts only
    when it names the current commit and this APK's hash."""
    record = _read_json(BUILD_RECORD) if BUILD_RECORD.is_file() else {}
    head = _git("rev-parse", "HEAD")
    apk_sha256 = sha256_file(apk)
    matches = record.get("git_commit") == head and record.get("apk_sha256") == apk_sha256
    return {
        "git_commit": head,
        "apk_sha256": apk_sha256,
        "build_record_matches": matches,
        "tree_clean_after_build": matches and record.get("tree_clean_after_build") is True,
        "tree_clean_at_run": not tree_changes(),
    }


def build_tools(root: Path = APP_DIR) -> dict[str, Any]:
    """The pinned versions, read from the app's build files."""

    def find(pattern: str, text: str, what: str) -> str:
        match = re.search(pattern, text)
        if match is None:
            raise SystemExit(f"{what} not found under {root.as_posix()}")
        return match.group(1)

    wrapper = (root / "gradle/wrapper/gradle-wrapper.properties").read_text(encoding="utf-8")
    settings = (root / "settings.gradle.kts").read_text(encoding="utf-8")
    app = (root / "app/build.gradle.kts").read_text(encoding="utf-8")
    props = (root / "gradle.properties").read_text(encoding="utf-8")
    aar_sha256 = find(r"glypharis\.ortAarSha256=([0-9a-f]{64})", props, "package hash")
    if set(aar_sha256) == {"0"}:
        raise SystemExit("gradle.properties still holds the placeholder package hash")
    return {
        "gradle": find(r"gradle-([0-9.]+)-bin\.zip", wrapper, "Gradle version"),
        "android_gradle_plugin": find(
            r'"com\.android\.application"\) version "([^"]+)"', settings, "AGP version"
        ),
        "kotlin": find(
            r'"org\.jetbrains\.kotlin\.android"\) version "([^"]+)"', settings, "Kotlin version"
        ),
        "jdk_major": int(find(r"glypharis\.jdkMajor=(\d+)", props, "JDK version")),
        "compile_sdk": int(find(r"compileSdk = (\d+)", app, "compileSdk")),
        "target_sdk": int(find(r"targetSdk = (\d+)", app, "targetSdk")),
        "min_sdk": int(find(r"minSdk = (\d+)", app, "minSdk")),
        "onnxruntime_android": find(r"glypharis\.ortVersion=(\S+)", props, "package version"),
        "onnxruntime_android_aar_sha256": aar_sha256,
    }


def check_model(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"{path.as_posix()} is missing")
    size = path.stat().st_size
    got = sha256_file(path)
    if size != MODEL_BYTES or got != MODEL_SHA256:
        raise SystemExit(
            f"{path.as_posix()} is not the GW-EFF-009 model: {size} bytes, SHA-256 {got}"
        )


def load_prepared(folder: Path = PREPARED) -> dict[str, Any]:
    """``prepared.json`` with its two tensor files checked against the hashes it records."""
    path = folder / "prepared.json"
    if not path.is_file():
        raise SystemExit(f"{path.as_posix()} is missing: run `prepare` first")
    doc = _read_json(path)
    for part in ("input", "logits"):
        file = folder / doc[part]["file"]
        if not file.is_file() or sha256_file(file) != doc[part]["sha256"]:
            raise SystemExit(f"{file.as_posix()} differs from prepared.json")
    return doc


# ------------------------------------------------------------------------------------ prepare


def capture_line(recognizer: Any, crop: Any) -> dict[str, Any]:
    """Run one crop through ``recognizer`` (`TextRecognizer.logits`, the laptop's own input code)
    and capture what its model received and returned: ``images``, ``widths``, the raw ``logits``
    ``(T, 1, classes)`` and the line's own ``line_logits`` ``(frames, classes)``."""
    import torch

    from glypharis.models.rec.base import LineRecognizer

    class Capture(LineRecognizer):
        def __init__(self, inner: LineRecognizer) -> None:
            super().__init__()
            self.inner = inner
            self.stride = inner.stride
            self.num_classes = inner.num_classes
            self.seen: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def frames(self, widths: torch.Tensor) -> torch.Tensor:
            return self.inner.frames(widths)

        def forward(self, x: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
            out = self.inner(x, widths)
            self.seen.append((x.detach().clone(), widths.detach().clone(), out.detach().clone()))
            return out

    inner = recognizer.model
    capture = Capture(inner)
    recognizer.model = capture
    try:
        line_logits = recognizer.logits([crop])[0]
    finally:
        recognizer.model = inner
    if len(capture.seen) != 1:
        raise SystemExit(f"expected one network call for one line, got {len(capture.seen)}")
    images, widths, logits = capture.seen[0]
    return {
        "images": images.numpy().astype(np.float32),
        "widths": widths.numpy().astype(np.int64),
        "logits": logits.numpy().astype(np.float32),
        "line_logits": line_logits.numpy().astype(np.float32),
    }


def check_laptop_logits(logits: np.ndarray, line_logits: np.ndarray) -> None:
    """The laptop's output must be a usable reference: one line, its frames first, all finite."""
    if logits.ndim != 3 or logits.shape[1] != 1:
        raise SystemExit(f"unexpected logits shape {logits.shape}")
    if not np.array_equal(logits[: line_logits.shape[0], 0], line_logits):
        raise SystemExit("the line's logits are not the head of the network's output")
    if not np.isfinite(logits).all():
        raise SystemExit("the laptop's logits are not all finite")


def phase16_text(key: str, model: Path) -> str | None:
    """Phase 16's greedy text for line ``key``, from the file beside the model, if it is there."""
    path = model.parent / "lines_outputs.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["key"] == key:
                return str(row["text"])
    return None


def cmd_prepare(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise SystemExit(f"{args.out.as_posix()} exists: move it aside first")
    check_model(args.model)
    variant = _read_json(args.model.parent / "variant.json")
    manifest_sha = sha256_file(args.bundle / "manifest.json")
    if manifest_sha != variant["bundle"]["manifest_sha256"]:
        raise SystemExit(f"{args.bundle.as_posix()} is not the bundle GW-EFF-009 was built from")
    release_sha = sha256_file(args.bundle / "recognizer.pt")
    if release_sha != variant["bundle"]["files"]["recognizer.pt"]:
        raise SystemExit(f"{args.bundle.as_posix()}/recognizer.pt differs from its manifest")

    import onnxruntime
    from export_check import line_indices
    from phase16_variants import eval_paths

    from glypharis.data.linepack import LinePackReader
    from glypharis.efficiency.onnx_build import onnx_recognizer
    from glypharis.efficiency.threads import check_session
    from glypharis.pipeline.frozen import FREEZE_RECORD
    from glypharis.pipeline.recognizer import TextRecognizer

    _, pack = eval_paths(FREEZE_RECORD, args.data_root)
    reader = LinePackReader(pack)
    index = int(line_indices(len(reader))[0])
    record = reader.records[index]
    if record.split != "validation":
        raise SystemExit(f"line {record.key} is not a validation line")

    like = TextRecognizer.from_release(args.bundle / "recognizer.pt", "cpu")
    recognizer = onnx_recognizer(args.model, like, args.threads)
    options = check_session(recognizer.model.session, args.threads)
    got = capture_line(recognizer, reader.image(index))
    reader.close()

    logits, line_logits = got["logits"], got["line_logits"]
    check_laptop_logits(logits, line_logits)
    text = recognizer.charset.ctc_greedy_decode(line_logits.argmax(-1).tolist())
    reference = phase16_text(record.key, args.model)
    if reference is not None and reference != text:
        raise SystemExit(f"line {record.key}: the text differs from Phase 16's")

    args.out.mkdir(parents=True)
    write_gwt(args.out / PHONE_INPUT, {"images": got["images"], "widths": got["widths"]})
    write_gwt(args.out / "laptop_logits.gwt", {"logits": logits})
    doc: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "part": "a",
        "created_utc": _utc(),
        "git_commit": _git("rev-parse", "HEAD"),
        "line": {
            "rule": "index 0 of export_check.line_indices over the validation pack",
            "index": index,
            "key": record.key,
            "split": record.split,
            "pack": pack.relative_to(args.data_root).as_posix(),
        },
        "model": {"file": PHONE_MODEL, "sha256": MODEL_SHA256, "bytes": MODEL_BYTES},
        "threads": args.threads,
        "session_options": options,
        "onnxruntime": onnxruntime.__version__,
        "input": {
            "file": PHONE_INPUT,
            "sha256": sha256_file(args.out / PHONE_INPUT),
            "images_shape": list(got["images"].shape),
            "widths": got["widths"].tolist(),
        },
        "logits": {
            "file": "laptop_logits.gwt",
            "sha256": sha256_file(args.out / "laptop_logits.gwt"),
            "shape": list(logits.shape),
            "frames": int(line_logits.shape[0]),
            "finite": bool(np.isfinite(logits).all()),
        },
        "text": text,
        "matches_phase16_text": None if reference is None else True,
        "characters": list(recognizer.charset.characters),
    }
    write_json_atomic(args.out / "prepared.json", doc)
    print(
        json.dumps(
            {
                "line": doc["line"],
                "images_shape": doc["input"]["images_shape"],
                "widths": doc["input"]["widths"],
                "logits_shape": doc["logits"]["shape"],
                "frames": doc["logits"]["frames"],
                "finite": doc["logits"]["finite"],
                "matches_phase16_text": doc["matches_phase16_text"],
                "session_options": options,
            },
            indent=1,
        )
    )
    return 0


# ---------------------------------------------------------------------------------------- adb


@dataclass(frozen=True)
class AdbResult:
    returncode: int
    stdout: str
    stderr: str


def sdk_roots() -> list[Path]:
    """The SDK folders to look in: ``ANDROID_HOME``, ``ANDROID_SDK_ROOT``, the default folder."""
    roots = [os.environ.get("ANDROID_HOME"), os.environ.get("ANDROID_SDK_ROOT")]
    local = os.environ.get("LOCALAPPDATA")
    roots.append(str(Path(local) / "Android/Sdk") if local else None)
    return [Path(root) for root in roots if root]


def adb_executable() -> str:
    """``adb`` from the SDK, or from the PATH."""
    name = "adb.exe" if os.name == "nt" else "adb"
    for root in sdk_roots():
        if (root / "platform-tools" / name).is_file():
            return str(root / "platform-tools" / name)
    found = shutil.which("adb")
    if found is None:
        raise SystemExit("adb not found: set ANDROID_HOME for this command")
    return found


def run_adb(
    args: Sequence[str], serial: str | None = None, timeout: float = ADB_SECONDS
) -> AdbResult:
    """Every adb call of this script. Tests replace this one function."""
    command = [adb_executable()]
    if serial is not None:
        command += ["-s", serial]
    command += list(args)
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return AdbResult(124, "", f"adb {args[0]} timed out after {timeout:.0f} s")
    return AdbResult(proc.returncode, proc.stdout, proc.stderr)


@dataclass(frozen=True)
class Phone:
    """The chosen device. Its serial number stays in memory and is cut out of every message."""

    serial: str | None
    hidden: tuple[str, ...]

    def scrub(self, text: str) -> str:
        for value in self.hidden:
            text = text.replace(value, "<device>")
        return text

    def call(self, args: Sequence[str], timeout: float = ADB_SECONDS) -> AdbResult:
        return run_adb(args, self.serial, timeout)

    def adb(self, args: Sequence[str], timeout: float = ADB_SECONDS) -> str:
        """stdout of an adb command that must succeed."""
        got = self.call(args, timeout)
        if got.returncode != 0:
            detail = self.scrub((got.stderr or got.stdout).strip())
            raise SystemExit(f"adb {args[0]} failed: {detail}")
        return got.stdout

    def shell(self, command: str, timeout: float = ADB_SECONDS) -> str:
        return self.adb(["shell", command], timeout)

    def test(self, command: str) -> bool:
        return self.call(["shell", command]).returncode == 0


def select_phone(serial: str | None) -> Phone:
    """The one authorized device, or ``serial`` among several; refuses otherwise."""
    listed = run_adb(["devices"])
    if listed.returncode != 0:
        raise SystemExit("adb devices failed")
    rows = [
        line.split()[:2]
        for line in listed.stdout.splitlines()[1:]
        if len(line.split()) >= 2 and not line.startswith("*")
    ]
    hidden = tuple(row[0] for row in rows)
    authorized = [row[0] for row in rows if row[1] == "device"]
    if serial is not None:
        if serial not in authorized:
            raise SystemExit("the device named by --serial is not listed as authorized")
        return Phone(serial, (*hidden, serial))
    if not authorized:
        states = sorted(row[1] for row in rows)
        detail = f" ({len(rows)} listed: {', '.join(states)})" if rows else ""
        raise SystemExit(
            f"no authorized device{detail}: connect the phone, enable USB debugging and accept "
            "the prompt on it"
        )
    if len(authorized) > 1:
        raise SystemExit(
            f"{len(authorized)} authorized devices: pass --serial to pick one (it is used for "
            "this command only and never recorded)"
        )
    return Phone(None, hidden)


DEVICE_PROPS: Final = {
    "manufacturer": "ro.product.manufacturer",
    "model": "ro.product.model",
    "soc_manufacturer": "ro.soc.manufacturer",
    "soc_model": "ro.soc.model",
    "board_platform": "ro.board.platform",
    "android_release": "ro.build.version.release",
    "sdk": "ro.build.version.sdk",
    "build": "ro.build.display_id",
    "abi": "ro.product.cpu.abi",
    "abi_list": "ro.product.cpu.abilist",
}


def read_device(phone: Phone) -> dict[str, Any]:
    """Model, SoC, RAM, Android build and ABI, from `DEVICE_PROPS` and ``/proc/meminfo`` only."""
    info: dict[str, Any] = {
        key: phone.scrub(phone.shell(f"getprop {prop}").strip())
        for key, prop in DEVICE_PROPS.items()
    }
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", phone.shell("cat /proc/meminfo"), re.MULTILINE)
    info["ram_total_kb"] = int(match.group(1)) if match else None
    if not info["model"]:
        raise SystemExit("the phone did not report its model")
    return info


def device_slug(info: Mapping[str, Any]) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(info["model"]).lower()).strip("-")


def cmd_device(args: argparse.Namespace) -> int:
    info = read_device(select_phone(args.serial))
    write_json_atomic(RUNS / device_slug(info) / "device.json", info)
    print(json.dumps({"slug": device_slug(info), **info}, indent=1))
    return 0


# -------------------------------------------------------------------------------------- build


def run_gradle(args: Sequence[str]) -> int:
    """The app's Gradle wrapper with the caller's environment. Tests replace this one function."""
    name = "gradlew.bat" if os.name == "nt" else "gradlew"
    return subprocess.run([str((APP_DIR / name).resolve()), *args], cwd=APP_DIR).returncode


def cmd_build(args: argparse.Namespace) -> int:
    if tree_changes():
        raise SystemExit("the tree is not clean: commit first, then build")
    missing = [name for name in BUILD_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"set {', '.join(missing)} for this command")
    if Path(os.environ["GRADLE_USER_HOME"]).resolve().is_relative_to(Path.cwd().resolve()):
        raise SystemExit("GRADLE_USER_HOME must be outside the repository")
    BUILD_RECORD.unlink(missing_ok=True)  # a failed build leaves no record to trust
    if run_gradle(GRADLE_ARGS) != 0:
        raise SystemExit("the build failed")
    if not APK.is_file():
        raise SystemExit(f"the build left no {APK.as_posix()}")
    changes = tree_changes()
    doc = {
        "experiment": EXPERIMENT,
        "part": "a",
        "built_utc": _utc(),
        "git_commit": _git("rev-parse", "HEAD"),
        "tree_clean_after_build": not changes,
        "apk_sha256": sha256_file(APK),
        "apk_bytes": APK.stat().st_size,
    }
    write_json_atomic(BUILD_RECORD, doc)
    print(json.dumps(doc, indent=1))
    if changes:
        raise SystemExit("git status is not empty straight after the build: no-go for this build")
    return 0


def run_aapt2(args: Sequence[str]) -> str:
    """stdout of the newest ``aapt2`` in the SDK. Tests replace this one function."""
    name = "aapt2.exe" if os.name == "nt" else "aapt2"
    found = [path for root in sdk_roots() for path in (root / "build-tools").glob(f"*/{name}")]
    if not found:
        raise SystemExit("aapt2 not found: set ANDROID_HOME for this command")
    tool = max(found, key=lambda path: [int(n) for n in re.findall(r"\d+", path.parent.name)])
    proc = subprocess.run(
        [str(tool), *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise SystemExit(f"aapt2 {args[0]} {args[1]} failed")
    return proc.stdout


def check_apk_manifest(apk: Path) -> list[str]:
    """The APK's permissions, which must be none, with the telemetry provider gone."""
    lines = run_aapt2(["dump", "permissions", str(apk)]).splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    if lines != [APK_PACKAGE_LINE]:
        raise SystemExit(f"the APK must ask for no permissions; aapt2 lists {lines}")
    tree = run_aapt2(["dump", "xmltree", "--file", "AndroidManifest.xml", str(apk)])
    if TELEMETRY_PROVIDER in tree:
        raise SystemExit(f"the APK still declares {TELEMETRY_PROVIDER}")
    return []


# ---------------------------------------------------------------------------------------- run


def phone_path(name: str) -> str:
    return f"{PHONE_DIR}/{name}"


def start_app(phone: Phone, extras: Mapping[str, str | int]) -> None:
    """Start the activity in a new process (`am start`), with `extras` as intent extras."""
    phone.shell(f"am force-stop {APP_ID}")
    parts = ["am", "start", "-W", "-n", ACTIVITY]
    for key, value in extras.items():
        flag = "--ei" if isinstance(value, int) else "--es"
        parts += [flag, key, str(value)]
    got = phone.call(["shell", shlex.join(parts)], timeout=120.0)
    output = f"{got.stdout}\n{got.stderr}".strip()
    if got.returncode != 0 or "Error" in output or "Exception" in output:
        raise SystemExit(f"the app did not start: {phone.scrub(output)}")


def wait_for(phone: Phone, name: str, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not phone.test(f"test -f {shlex.quote(phone_path(name))}"):
        if time.monotonic() >= deadline:
            raise SystemExit(f"the app wrote no {name} within {seconds:.0f} s; nothing recorded")
        time.sleep(POLL_SECONDS)


def wait_for_result(phone: Phone, name: str, seconds: float) -> bool:
    """Wait for the app's result ``name``: True once it is written, False when the app's process
    has been gone for `DEATH_POLLS` polls in a row without writing it (a low-memory kill or a
    crash). A phone without ``pidof`` is only waited on. Past ``seconds`` with no result and the
    process still there, nothing is recorded."""
    path = shlex.quote(phone_path(name))
    watch = phone.test("command -v pidof >/dev/null")
    deadline = time.monotonic() + seconds
    gone = 0
    while True:
        if phone.test(f"test -f {path}"):
            return True
        if watch:
            gone = 0 if phone.test(f"pidof {APP_ID} >/dev/null") else gone + 1
            if gone >= DEATH_POLLS:
                return phone.test(f"test -f {path}")
        if time.monotonic() >= deadline:
            raise SystemExit(f"the app wrote no {name} within {seconds:.0f} s; nothing recorded")
        time.sleep(POLL_SECONDS)


def died_result(mode: str, arm: str, run: int | None) -> dict[str, Any]:
    """The result of a pass whose process ended before the app wrote one, written by this tool:
    the arm is not measured (timing) or not checked (check), and the registered order goes on."""
    return {
        "schema": "glypharis-android-bench-1",
        "mode": mode,
        "status": PROCESS_DIED,
        "written_by": "phase22_android.py: the app's process ended before it wrote a result",
        "pid": None,
        "arm": arm,
        "run": run,
        "detected_utc": _utc(),
    }


def install_app(phone: Phone, apk: Path) -> None:
    """Install ``apk`` and let the app create its folder (``mode=ready``)."""
    phone.adb(["install", "-r", str(apk)], timeout=PUSH_SECONDS)
    start_app(phone, {"mode": "ready"})
    wait_for(phone, PHONE_READY, READY_SECONDS)


def verify_phone_hashes(phone: Phone, expected: Mapping[str, str]) -> str:
    """Compare ``sha256sum`` on the phone with ``expected`` (phone file name to hash). Returns the
    method used: ``sha256sum`` or ``unavailable``; the app hashes what it reads either way."""
    if not phone.test("command -v sha256sum"):
        return "unavailable"
    for name, sha in expected.items():
        output = phone.shell(f"sha256sum {shlex.quote(phone_path(name))}", timeout=120.0)
        got = output.split()[0] if output.split() else ""
        if got != sha:
            raise SystemExit(f"{name} on the phone has SHA-256 {got or 'unknown'}, expected {sha}")
    return "sha256sum"


def cmd_run(args: argparse.Namespace) -> int:
    prepared = load_prepared()
    check_model(args.model)
    if not args.apk.is_file():
        raise SystemExit(f"{args.apk.as_posix()} is missing: build the app first")
    tree = build_state(args.apk)
    if not args.allow_dirty:
        if not tree["build_record_matches"]:
            raise SystemExit("no build record for this commit and APK: run `build` first")
        if not tree["tree_clean_after_build"]:
            raise SystemExit("git status was not empty straight after the build: commit and build")
        if not tree["tree_clean_at_run"]:
            raise SystemExit("the tree is not clean: commit, run `build` again and re-run")
    apk_permissions = check_apk_manifest(args.apk)
    phone = select_phone(args.serial)
    info = read_device(phone)
    if REQUIRED_ABI not in str(info["abi_list"]).split(","):
        raise SystemExit(f"the phone does not list {REQUIRED_ABI}")
    folder = RUNS / device_slug(info)
    if (folder / "run.json").exists():
        raise SystemExit(f"{folder.as_posix()} already holds a run: move it aside first")
    folder.mkdir(parents=True, exist_ok=True)
    write_json_atomic(folder / "device.json", info)

    install_app(phone, args.apk)
    stale = " ".join(shlex.quote(phone_path(n)) for n in (PHONE_LOGITS, PHONE_RESULT))
    phone.shell(f"rm -f {stale}")
    phone.adb(["push", str(args.model), phone_path(PHONE_MODEL)], timeout=PUSH_SECONDS)
    phone.adb(["push", str(PREPARED / PHONE_INPUT), phone_path(PHONE_INPUT)], timeout=PUSH_SECONDS)
    hash_check = verify_phone_hashes(
        phone, {PHONE_MODEL: MODEL_SHA256, PHONE_INPUT: prepared["input"]["sha256"]}
    )

    start_app(
        phone,
        {
            "mode": "operator_check",
            "model": PHONE_MODEL,
            "input": PHONE_INPUT,
            "output": PHONE_LOGITS,
            "result": PHONE_RESULT,
            "threads": int(prepared["threads"]),
        },
    )
    wait_for(phone, PHONE_RESULT, args.wait_seconds)
    phone.adb(["pull", phone_path(PHONE_RESULT), str(folder / PHONE_RESULT)])
    result = _read_json(folder / PHONE_RESULT)
    if result.get("status") == "ok":
        phone.adb(["pull", phone_path(PHONE_LOGITS), str(folder / PHONE_LOGITS)])
    write_json_atomic(
        folder / "run.json",
        {
            "experiment": EXPERIMENT,
            "part": "a",
            "run_utc": _utc(),
            **tree,
            "apk_bytes": args.apk.stat().st_size,
            "apk_permissions": apk_permissions,
            "phone_hash_check": hash_check,
        },
    )
    print(
        json.dumps(
            {
                "device": device_slug(info),
                "status": result.get("status"),
                "folder": folder.as_posix(),
            }
        )
    )
    return 0


# ------------------------------------------------------------------------------------- report


def _session_failure_reason(error: Mapping[str, Any] | None) -> str:
    error = error or {}
    text = str(error.get("message") or "")
    if error.get("code") == "ORT_NOT_IMPLEMENTED" or MISSING_KERNEL.search(text):
        return "missing_kernel"
    return "session_failed"


def _has_missing_kernel(folder: Path) -> bool:
    """Whether the phone's last pulled result says a kernel or operator is missing."""
    path = folder / PHONE_RESULT
    if not path.is_file():
        return False
    result = _read_json(path)
    return (
        result.get("status") in ("session_failed", "runtime_failed")
        and _session_failure_reason(result.get("session_error")) == "missing_kernel"
    )


def _scrub_error(error: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The app's error record without device paths; the app cuts its own folder out first."""
    if error is None:
        return None
    return {
        key: DEVICE_PATH.sub("<app files>", value) if isinstance(value, str) else value
        for key, value in error.items()
    }


def compare_device(folder: Path, prepared: Mapping[str, Any], laptop: np.ndarray) -> dict[str, Any]:
    """One phone's record and verdict against the laptop's logits."""
    run = _read_json(folder / "run.json")
    result = _read_json(folder / PHONE_RESULT)
    status = result.get("status")
    if status == "bad_request":
        detail = (result.get("request_error") or {}).get("message")
        raise SystemExit(f"{folder.name}: the app could not start the check ({detail}); no verdict")

    reasons: list[str] = []
    if not (run["tree_clean_after_build"] and run["tree_clean_at_run"]):
        reasons.append("build_not_clean")
    if run["apk_permissions"] != []:
        reasons.append("apk_has_permissions")
    if result.get("model_sha256") != MODEL_SHA256:
        reasons.append("wrong_model_file")
    if result.get("input_sha256") != prepared["input"]["sha256"]:
        reasons.append("wrong_input_file")
    if status in ("session_failed", "runtime_failed"):
        reasons.append(_session_failure_reason(result.get("session_error")))
    elif status == "run_failed":
        reasons.append("run_failed")
    elif status != "ok":
        reasons.append(f"status_{status}")
    if status == "ok" and result.get("session_created") is not True:
        reasons.append("session_not_created")
    options = result.get("session_options") or {}
    if any(options.get(k) != prepared["session_options"].get(k) for k in SESSION_OPTION_KEYS):
        reasons.append("session_options_differ")

    record: dict[str, Any] = {
        "device": _read_json(folder / "device.json"),
        "apk_sha256": run["apk_sha256"],
        "apk_bytes": run["apk_bytes"],
        "git_commit": run["git_commit"],
        "build_record_matches": run["build_record_matches"],
        "tree_clean_after_build": run["tree_clean_after_build"],
        "tree_clean_at_run": run["tree_clean_at_run"],
        "apk_permissions": run["apk_permissions"],
        "phone_hash_check": run["phone_hash_check"],
        "app_model_sha256": result.get("model_sha256"),
        "app_input_sha256": result.get("input_sha256"),
        "status": status,
        "onnxruntime_version": result.get("onnxruntime_version"),
        "session_created": bool(result.get("session_created")),
        "session_error": _scrub_error(result.get("session_error")),
        "run_error": _scrub_error(result.get("run_error")),
        "session_options": result.get("session_options"),
        "output_shape": None,
        "shape_equal": None,
        "logits_finite": None,
        "logit_max_abs_diff": None,
        "logit_max_abs_diff_with_padding": None,
        "greedy_text_equal": None,
    }
    if status != "runtime_failed" and result.get("onnxruntime_version") != ORT_VERSION:
        reasons.append("wrong_onnxruntime_version")

    path = folder / PHONE_LOGITS
    if status == "ok":
        phone = read_gwt(path)["logits"] if path.is_file() else None
        if phone is None:
            reasons.append("logits_missing")
        else:
            shape = [int(d) for d in phone.shape]
            same_shape = shape == list(laptop.shape) and result.get("output_shape") == shape
            finite = bool(np.isfinite(phone).all()) and result.get("logits_finite") is True
            record.update(output_shape=shape, shape_equal=same_shape, logits_finite=finite)
            if not same_shape:
                reasons.append("wrong_shape")
            if not finite:
                reasons.append("non_finite_logits")
            if same_shape and finite:
                charset = Charset(list(prepared["characters"]))
                frames = int(prepared["logits"]["frames"])
                text = charset.ctc_greedy_decode(phone[:frames, 0].argmax(-1).tolist())
                # The line's own frames, as every laptop comparison of this name; the rest of the
                # output comes from padding and is recorded apart, driving nothing.
                diff = float(np.abs(phone[:frames] - laptop[:frames]).max())
                padded = float(np.abs(phone - laptop).max())
                if np.isfinite(diff):
                    record["logit_max_abs_diff"] = diff
                else:
                    reasons.append("logit_difference_not_finite")
                if np.isfinite(padded):
                    record["logit_max_abs_diff_with_padding"] = padded
                record["greedy_text_equal"] = text == prepared["text"]
                record["phone_text_sha256"] = _sha256_text(text)

    diff_seen = record["logit_max_abs_diff"]
    record["tell_owner_before_s4"] = (
        (diff_seen is not None and diff_seen > LOGIT_NOTICE)
        or record["greedy_text_equal"] is False
        or "logit_difference_not_finite" in reasons
    )
    record["verdict"] = NO_GO if reasons else GO
    record["reasons"] = reasons
    return record


def check_nothing_left_out(slugs: Sequence[str], unavailable: Sequence[str]) -> None:
    """A phone that was seen is in the report or named as unavailable, so a failure cannot vanish.
    A missing kernel cannot be named away: item 3 replaces a phone only for other reasons."""
    seen = sorted(p.parent.name for p in RUNS.glob("*/device.json"))
    left_out = [slug for slug in seen if slug not in slugs]
    kernel = [slug for slug in left_out if _has_missing_kernel(RUNS / slug)]
    if kernel:
        raise SystemExit(f"{kernel}: the run recorded a missing kernel and must be in the report")
    named = {text.split(":", 1)[0].strip() for text in unavailable}
    unnamed = [slug for slug in left_out if slug not in named]
    if unnamed:
        raise SystemExit(
            f"{unnamed}: seen but not in this report; finish the run or pass --unavailable "
            '"<slug>: reason"'
        )


def cmd_report(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise SystemExit(f"{args.out.as_posix()} exists: move it aside first")
    prepared = load_prepared()
    slugs = args.device or sorted(p.parent.name for p in RUNS.glob("*/run.json"))
    check_nothing_left_out(slugs, args.unavailable)
    if not slugs:
        raise SystemExit("no phone has a finished run: nothing to report")
    laptop = read_gwt(PREPARED / prepared["logits"]["file"])["logits"]
    devices = []
    for slug in slugs:
        folder = RUNS / slug
        if not (folder / "run.json").is_file() or not (folder / PHONE_RESULT).is_file():
            raise SystemExit(f"{slug} has no finished run")
        devices.append({"slug": slug, **compare_device(folder, prepared, laptop)})
    tools = build_tools()
    verdict = GO if all(d["verdict"] == GO for d in devices) else NO_GO
    report = {
        "experiment": EXPERIMENT,
        "part": "a: Android operator check",
        "generated_utc": _utc(),
        "git_commit": _git("rev-parse", "HEAD"),
        "data": "one validation line's network input; no test data",
        "verdict": verdict,
        "rule": RULE,
        "no_go_reasons": sorted({r for d in devices for r in d["reasons"]}),
        "tell_owner_before_s4": any(d["tell_owner_before_s4"] for d in devices),
        "logit_notice_threshold": LOGIT_NOTICE,
        "package": {
            "name": ORT_PACKAGE,
            "version": tools["onnxruntime_android"],
            "aar_sha256": tools["onnxruntime_android_aar_sha256"],
            "merged_manifest": (
                "the package's files are unchanged; its INTERNET and ACCESS_NETWORK_STATE "
                "permissions and telemetry provider are removed at manifest merge"
            ),
        },
        "build_tools": {k: v for k, v in tools.items() if not k.startswith("onnxruntime_android")},
        "model": {
            **prepared["model"],
            "source": "GW-EFF-009 fused-attention",
            "fused_nodes": "com.microsoft.MultiHeadAttention, 9",
        },
        "line": {k: prepared["line"][k] for k in ("index", "key", "split", "pack")},
        "laptop": {
            "onnxruntime": prepared["onnxruntime"],
            "threads": prepared["threads"],
            "session_options": prepared["session_options"],
            "input_images_shape": prepared["input"]["images_shape"],
            "input_widths": prepared["input"]["widths"],
            "input_sha256": prepared["input"]["sha256"],
            "logits_shape": prepared["logits"]["shape"],
            "logits_sha256": prepared["logits"]["sha256"],
            "frames": prepared["logits"]["frames"],
            "text_sha256": _sha256_text(prepared["text"]),
        },
        "devices": devices,
        "unavailable": list(args.unavailable),
        "not_measured": "no time or memory figure comes from this check (part b measures them)",
    }
    write_json_atomic(args.out, report)
    print(
        json.dumps(
            {
                "verdict": verdict,
                "reasons": report["no_go_reasons"],
                "tell_owner_before_s4": report["tell_owner_before_s4"],
                "report": args.out.as_posix(),
            }
        )
    )
    return 0


# ------------------------------------------------------------------- phone timing: recording


def tensor_name(sha256: str) -> str:
    return f"t_{sha256[:40]}.gwt"


def plan_name(arm: str) -> str:
    return f"plan_{arm}.json"


def check_name(arm: str) -> str:
    return f"check_{arm}.json"


def laptop_name(arm: str) -> str:
    return f"laptop_{arm}.json"


def timing_name(arm: str, run: int) -> str:
    return f"timing_{arm}_run{run}.json"


def checkresult_name(arm: str) -> str:
    return f"checkresult_{arm}.json"


def output_name(arm: str, index: int) -> str:
    return f"c_{arm}_{index}.gwt"


@dataclass
class Call:
    """One ``session.run`` of a pipeline being replayed: the model, the tensor file of its feeds
    and, for the recognizer, its output's shape and each line's frames and greedy text."""

    model: str
    tensor: str
    shape: list[int] | None = None
    frames: list[int] | None = None
    texts: list[str] | None = None


@dataclass(frozen=True)
class Mark:
    """Where an item, a page or a line, starts among the recorded calls."""

    kind: str
    key: str
    warmup: bool
    start: int


Decode = Callable[[Mapping[str, np.ndarray], np.ndarray], tuple[list[int], list[str]]]


class RunRecorder:
    """Records every ``session.run`` of the sessions it wraps, in call order.

    `wrapping` replaces each session instance's ``run`` with a recorder that copies the feeds,
    hands the copy to ``store`` (which writes the tensor file and returns its name) and then calls
    the original with the feeds it was given. For the recognizer, ``decode`` turns the copy and the
    returned logits into each line's frames and greedy text. `mark` starts an item."""

    def __init__(self, store: Callable[[Mapping[str, np.ndarray]], str], decode: Decode) -> None:
        self.store = store
        self.decode = decode
        self.calls: list[Call] = []
        self.marks: list[Mark] = []

    def mark(self, kind: str, key: str, warmup: bool) -> None:
        self.marks.append(Mark(kind, key, warmup, len(self.calls)))

    def _recorder(self, model: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def run(output_names: Any, input_feed: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
            feeds = {name: np.array(value, copy=True) for name, value in input_feed.items()}
            call = Call(model, self.store(feeds))
            outputs = original(output_names, input_feed, *args, **kwargs)
            if model == "recognizer":
                logits = np.asarray(outputs[0])
                call.shape = [int(d) for d in logits.shape]
                call.frames, call.texts = self.decode(feeds, logits)
            self.calls.append(call)
            return outputs

        return run

    @contextlib.contextmanager
    def wrapping(self, sessions: Mapping[str, Any]) -> Iterator[RunRecorder]:
        """Each session of ``{"detector": ..., "recognizer": ...}`` recorded for the block, its
        own ``run`` put back afterwards."""
        saved = {model: vars(session).get("run") for model, session in sessions.items()}
        for model, session in sessions.items():
            session.run = self._recorder(model, session.run)
        try:
            yield self
        finally:
            for model, session in sessions.items():
                if saved[model] is None:
                    del session.run
                else:
                    session.run = saved[model]


def group_items(calls: Sequence[Call], marks: Sequence[Mark]) -> list[dict[str, Any]]:
    """The recorded calls grouped into their items, in run order. A page's calls start with its
    detector call, its only one; a line is exactly one recognizer call."""
    if calls and (not marks or marks[0].start != 0):
        raise SystemExit("a session ran before the first item started")
    items = []
    for n, mark in enumerate(marks):
        end = marks[n + 1].start if n + 1 < len(marks) else len(calls)
        own = list(calls[mark.start : end])
        models = [c.model for c in own]
        if mark.kind == "page":
            if models[:1] != ["detector"] or models.count("detector") != 1:
                raise SystemExit(
                    f"page {mark.key}: its calls are {models}, not its detector call and then "
                    "the recognizer's"
                )
        elif mark.kind == "line":
            if models != ["recognizer"]:
                raise SystemExit(
                    f"line {mark.key}: its calls are {models}, not one recognizer call"
                )
        else:
            raise SystemExit(f"{mark.key}: unknown item kind {mark.kind!r}")
        items.append({"kind": mark.kind, "key": mark.key, "warmup": mark.warmup, "calls": own})
    return items


class TensorStore:
    """Tensor files named by their content, ``t_<first 40 hex of the file's SHA-256>.gwt``, in one
    folder: a tensor fed to several calls or arms is written once."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.tensors: dict[str, dict[str, Any]] = {}
        self.arena: dict[str, int] = {}  # bytes of the app's input buffer each file needs

    def put(self, tensors: Mapping[str, np.ndarray]) -> str:
        data = gwt_bytes(tensors)
        sha = hashlib.sha256(data).hexdigest()
        name = tensor_name(sha)
        known = self.tensors.get(name)
        if known is None:
            if (self.folder / name).exists():
                raise SystemExit(f"{name} exists already: nothing is overwritten")
            _write_bytes(self.folder / name, data)
            self.tensors[name] = {"sha256": sha, "bytes": len(data)}
            self.arena[name] = arena_bytes(tensors)
        elif known["sha256"] != sha:
            raise SystemExit(f"{name}: two tensor files share the first 40 hex of their SHA-256")
        return name


def arena_bytes(tensors: Mapping[str, np.ndarray]) -> int:
    """The bytes the app's one input buffer needs for a call: each payload starts on an
    `ARENA_ALIGN`-byte boundary, in the file's order."""
    total = 0
    for value in tensors.values():
        total += -(-int(np.asarray(value).nbytes) // ARENA_ALIGN) * ARENA_ALIGN
    return total


def line_outputs(
    model: Any, charset: Charset, widths: np.ndarray, logits: np.ndarray
) -> tuple[list[int], list[str]]:
    """Each line's frames in one recognizer call, ``model.frames(widths)`` clamped to the output's
    length as `TextRecognizer.logits` takes them, and its greedy text from the logits ``(T, N,
    classes)``."""
    import torch

    w = torch.from_numpy(np.asarray(widths, dtype=np.int64))
    frames = [int(f) for f in model.frames(w).clamp(max=logits.shape[0])]
    texts = [
        charset.ctc_greedy_decode(logits[:f, j].argmax(-1).tolist()) for j, f in enumerate(frames)
    ]
    return frames, texts


def record_arm(
    ocr: Any,
    pages: Iterable[tuple[str, np.ndarray]],
    lines: Iterable[tuple[str, Any]],
    warmup: int,
    store: TensorStore,
) -> list[dict[str, Any]]:
    """Replay ``pages`` (key and RGB array) through ``ocr.predict``, the first ``warmup`` of them
    marked as warm-up pages, then each line (key and crop) alone through
    ``ocr.recognizer.logits([crop])``, recording every session call; returns the items."""
    rec = ocr.recognizer

    def decode(feeds: Mapping[str, np.ndarray], logits: np.ndarray) -> tuple[list[int], list[str]]:
        return line_outputs(rec.model, rec.charset, feeds["widths"], logits)

    recorder = RunRecorder(store.put, decode)
    sessions = {"detector": ocr.detector.backend.session, "recognizer": rec.model.session}
    with recorder.wrapping(sessions):
        for n, (key, rgb) in enumerate(pages):
            recorder.mark("page", key, n < warmup)
            ocr.predict(rgb)
        for key, crop in lines:
            recorder.mark("line", key, False)
            rec.logits([crop])
    return group_items(recorder.calls, recorder.marks)


def _tensor_map(names: Iterable[str], tensors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {name: dict(tensors[name]) for name in sorted(set(names))}


def timing_plan(
    arm: str,
    threads: int,
    models: Mapping[str, Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
    buffer_bytes: int,
) -> dict[str, Any]:
    """One arm's timing plan (``mobile/android-bench/README.md``): its two models (``models``
    maps ``detector`` and ``recognizer`` to the phone file and its hash), the size of the app's
    one input buffer (the same for every arm, so it adds the same to every arm's VmHWM) and every
    item in run order with its calls' tensor files."""
    pages = [i for i in items if i["kind"] == "page"]
    return {
        "schema": PLAN_SCHEMA,
        "experiment": EXPERIMENT,
        "arm": arm,
        "threads": threads,
        "models": {
            role: {"file": models[role]["file"], "sha256": models[role]["sha256"], "output": out}
            for role, out in (("detector", "prob"), ("recognizer", "logits"))
        },
        "buffer_bytes": buffer_bytes,
        "warmup_pages": sum(1 for i in pages if i["warmup"]),
        "timed_pages": sum(1 for i in pages if not i["warmup"]),
        "lines": sum(1 for i in items if i["kind"] == "line"),
        "items": [
            {
                "kind": i["kind"],
                "key": i["key"],
                "warmup": i["warmup"],
                "calls": [{"model": c.model, "tensor": c.tensor} for c in i["calls"]],
            }
            for i in items
        ],
        "tensors": _tensor_map((c.tensor for i in items for c in i["calls"]), tensors),
    }


def check_plan(
    arm: str,
    threads: int,
    model: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One arm's check plan, every recognizer call of the warm-up pages and then the lines, each
    with its own output file; and the laptop's record of each call (output shape, each line's
    frames and greedy text) for the comparison."""
    calls: list[dict[str, Any]] = []
    laptop: list[dict[str, Any]] = []
    for item in items:
        if item["kind"] == "page" and not item["warmup"]:
            continue
        for k, call in enumerate(item["calls"]):
            if call.model != "recognizer":
                continue
            page = item["kind"] == "page"
            source = f"page {item['key']} call {k}" if page else f"line {item['key']}"
            entry = {
                "tensor": call.tensor,
                "output": output_name(arm, len(calls)),
                "source": source,
            }
            calls.append(entry)
            laptop.append(
                {**entry, "shape": call.shape, "frames": call.frames, "texts": call.texts}
            )
    plan = {
        "schema": CHECK_SCHEMA,
        "experiment": EXPERIMENT,
        "arm": arm,
        "threads": threads,
        "model": {"file": model["file"], "sha256": model["sha256"], "output": "logits"},
        "calls": calls,
        "tensors": _tensor_map((c["tensor"] for c in calls), tensors),
    }
    return plan, laptop


def _write_plan(path: Path, doc: Mapping[str, Any]) -> None:
    """A plan for the phone: ASCII, without indentation, so the app reads it with little memory."""
    write_text_atomic(path, json.dumps(doc, ensure_ascii=True, separators=(",", ":")) + "\n")


def write_arm(
    folder: Path,
    arm: str,
    threads: int,
    models: Mapping[str, Mapping[str, Any]],
    items: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
    buffer_bytes: int,
) -> dict[str, Any]:
    """Write one arm's timing plan, check plan and laptop record; returns what ``prepare.json``
    keeps of them."""
    plan = timing_plan(arm, threads, models, items, tensors, buffer_bytes)
    check, laptop = check_plan(arm, threads, models["recognizer"], items, tensors)
    files = {"plan": plan_name(arm), "check": check_name(arm), "laptop": laptop_name(arm)}
    _refuse_existing(*(folder / name for name in files.values()))
    _write_plan(folder / files["plan"], plan)
    _write_plan(folder / files["check"], check)
    write_json_atomic(
        folder / files["laptop"],
        {"experiment": EXPERIMENT, "arm": arm, "check": files["check"], "calls": laptop},
    )
    return {
        **{
            part: {"file": name, "sha256": sha256_file(folder / name)}
            for part, name in files.items()
        },
        "items": len(items),
        "calls": sum(len(i["calls"]) for i in items),
        "check_calls": len(check["calls"]),
        "check_lines": sum(len(c["frames"] or []) for c in laptop),
        "tensors": len(plan["tensors"]),
        "tensor_bytes": sum(t["bytes"] for t in plan["tensors"].values()),
    }


def timing_record(
    arms: Mapping[str, Mapping[str, Any]],
    tensors: Mapping[str, Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
    threads: int,
    characters: Sequence[str],
    smoke: bool,
    **extra: Any,
) -> dict[str, Any]:
    """``prepare.json``: what was prepared, with the hashes the later commands check. ``models``
    maps each phone model file to its repository path, hash and size."""
    return {
        "experiment": EXPERIMENT,
        "part": "b",
        "kind": "phase22-phone-timing",
        "smoke": smoke,
        "threads": threads,
        "to_phone": list(arms),
        **extra,
        "models": {name: dict(m) for name, m in models.items()},
        "arms": {arm: dict(r) for arm, r in arms.items()},
        "unique_tensors": len(tensors),
        "tensor_bytes": sum(t["bytes"] for t in tensors.values()),
        "tensors": {name: dict(t) for name, t in sorted(tensors.items())},
        "characters": list(characters),
    }


def model_record(path: Path, sha256: str | None = None) -> dict[str, Any]:
    """A model file the phone gets: its repository path, hash and size; ``sha256`` must match
    when given."""
    if not path.is_file():
        raise SystemExit(f"{path.as_posix()} is missing")
    got = sha256_file(path)
    if sha256 is not None and got != sha256:
        raise SystemExit(f"{path.as_posix()} has SHA-256 {got}, not {sha256}")
    return {"source": path.as_posix(), "sha256": got, "bytes": path.stat().st_size}


def read_to_phone(path: Path = ACCURACY) -> list[str]:
    """The arms GW-EFF-013's accuracy part sends to the phone, in `candidate.ARMS` order."""
    from glypharis.efficiency.candidate import ARMS, DEFAULT_ARM

    if not path.is_file():
        raise SystemExit(f"{path.as_posix()} is missing: run phase22_candidate.py accuracy first")
    listed = list(_read_json(path).get("to_phone") or [])
    if DEFAULT_ARM not in listed or any(a not in ARMS for a in listed):
        raise SystemExit(f"{path.as_posix()}: to_phone must name fused and arms only, not {listed}")
    return [a for a in ARMS if a in listed]


def profile_page_inputs(data_root: Path, total: int) -> Iterator[tuple[str, np.ndarray]]:
    """The first ``total`` Phase 5 profile pages as `phase16_variants.profile_pages` hands them to
    `GlypharisOCR.predict`: opened with `open_page_image`, as an RGB array."""
    from phase16_variants import EVAL_SET
    from run_baselines import e2e_samples

    from glypharis.data.pages import open_page_image

    n = 0
    for _, page in zip(range(total), e2e_samples(data_root, EVAL_SET), strict=False):
        if page.split != "validation":
            raise SystemExit(f"page {page.doc_id} is from the {page.split} split")
        img = open_page_image(page, data_root)
        yield page.doc_id, np.asarray(img.convert("RGB"))
        n += 1
    if n < total:
        raise SystemExit(f"{EVAL_SET} gave {n} pages; need {total}")


def seeded_line_inputs(pack: Path, indices: Sequence[int]) -> Iterator[tuple[str, Any]]:
    """The lines at ``indices`` of the validation pack, key and crop."""
    from glypharis.data.linepack import LinePackReader

    reader = LinePackReader(pack)
    try:
        for i in indices:
            record = reader.records[i]
            if record.split != "validation":
                raise SystemExit(f"line {record.key} is not a validation line")
            yield record.key, reader.image(i)
    finally:
        reader.close()


def cmd_prepare_timing(args: argparse.Namespace) -> int:
    limits = {"pages": args.limit_pages, "lines": args.limit_lines}
    for name, value in limits.items():
        if value is not None and value < 1:
            raise SystemExit(f"--limit-{name} must be a positive integer")
    if args.threads < 1:
        raise SystemExit("--threads must be a positive integer")
    smoke = any(value is not None for value in limits.values())
    if smoke and args.out is None:
        raise SystemExit("a smoke test (--limit-pages, --limit-lines) needs --out")
    out: Path = TIMING if args.out is None else args.out
    if smoke and out.resolve() == TIMING.resolve():
        raise SystemExit(f"a smoke test is not written to {TIMING.as_posix()}")
    _refuse_existing(out)
    to_phone = read_to_phone()
    detector = model_record(DETECTOR, DETECTOR_SHA256)

    import onnxruntime
    from export_check import SEED
    from phase16_variants import EVAL_SET, eval_paths, line_sample
    from phase22_candidate import deployed_arm
    from run_baselines import PROFILE_SAMPLES, WARMUP

    from glypharis.data.linepack import LinePackReader
    from glypharis.efficiency import candidate
    from glypharis.efficiency.onnx_build import check_sessions
    from glypharis.efficiency.v1 import assert_v1_code
    from glypharis.pipeline.frozen import FREEZE_RECORD

    v1_code = assert_v1_code(".")
    warmup, timed = (WARMUP, PROFILE_SAMPLES)
    if args.limit_pages is not None:
        warmup, timed = 1, args.limit_pages - 1
    _, pack = eval_paths(FREEZE_RECORD, args.data_root)
    reader = LinePackReader(pack)
    pack_lines = len(reader)
    reader.close()
    count = TIMING_LINES if args.limit_lines is None else min(args.limit_lines, TIMING_LINES)
    indices = [int(i) for i in line_sample(pack_lines)[:count]]

    models: dict[str, dict[str, Any]] = {PHONE_DETECTOR: detector}
    arms: dict[str, dict[str, Any]] = {}
    recorded: dict[str, tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = {}
    sessions: dict[str, Any] = {}
    characters: list[str] = []
    out.mkdir(parents=True)
    store = TensorStore(out)
    for arm in to_phone:
        path = candidate.recognizer_file(arm)
        ocr, loaded = deployed_arm(arm, BUNDLE, args.threads, candidate.WORK_ROOT)
        sessions[arm] = check_sessions(ocr, args.threads)
        chars = list(ocr.recognizer.charset.characters)
        if characters and chars != characters:
            raise SystemExit(f"{arm} reads another charset than {to_phone[0]}")
        characters = chars
        models.setdefault(path.name, model_record(path))
        if loaded["recognizer_sha256"] != models[path.name]["sha256"]:
            raise SystemExit(f"{arm} loaded another recognizer than {path.as_posix()}")
        roles = {
            "detector": {"file": PHONE_DETECTOR, "sha256": detector["sha256"]},
            "recognizer": {"file": path.name, "sha256": models[path.name]["sha256"]},
        }
        items = record_arm(
            ocr,
            profile_page_inputs(args.data_root, warmup + timed),
            seeded_line_inputs(pack, indices),
            warmup,
            store,
        )
        recorded[arm] = (roles, items, loaded)
        del ocr
        print(f"{arm}: {len(items)} items, {sum(len(i['calls']) for i in items)} calls", flush=True)
    # One buffer size for every arm: the largest call of any arm, so the app's input buffer adds
    # the same to each arm's VmHWM.
    buffer = max(
        store.arena[c.tensor]
        for _, items, _ in recorded.values()
        for i in items
        for c in i["calls"]
    )
    for arm, (roles, items, loaded) in recorded.items():
        arms[arm] = {
            **write_arm(out, arm, args.threads, roles, items, store.tensors, buffer),
            "recognizer": roles["recognizer"]["file"],
            "batching": loaded["batching"],
            "loaded": loaded["loaded"],
        }
    doc = timing_record(
        arms,
        store.tensors,
        models,
        args.threads,
        characters,
        smoke,
        created_utc=_utc(),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=bool(tree_changes()),
        v1_code=v1_code,
        onnxruntime=onnxruntime.__version__,
        session_options=sessions,
        buffer_bytes=buffer,
        accuracy={
            "file": ACCURACY.as_posix(),
            "sha256": sha256_file(ACCURACY),
            "to_phone": to_phone,
        },
        limits=limits,
        pages={
            "eval_set": EVAL_SET,
            "rule": "the first pages of run_baselines.e2e_samples, opened and passed to "
            "GlypharisOCR.predict as phase16_variants.profile_pages does",
            "warmup": warmup,
            "timed": timed,
        },
        lines={
            "pack": pack.relative_to(args.data_root).as_posix(),
            "seed": SEED,
            "rule": "the first indices of phase16_variants.line_sample over the pack, each line "
            "alone through TextRecognizer.logits",
            "count": len(indices),
            "indices": indices,
        },
    )
    write_json_atomic(out / "prepare.json", doc)
    summary = {
        "smoke": smoke,
        "pages": {"warmup": warmup, "timed": timed},
        "lines": len(indices),
        "arms": {
            arm: {k: r[k] for k in ("items", "calls", "check_calls", "tensors", "tensor_bytes")}
            for arm, r in arms.items()
        },
        "unique_tensors": doc["unique_tensors"],
        "tensor_bytes": doc["tensor_bytes"],
        "folder": out.as_posix(),
    }
    print(json.dumps(summary, indent=1))
    return 0


# ------------------------------------------------------------------ phone timing: conditions


def _match(pattern: str, text: str) -> str | None:
    found = re.search(pattern, text, re.MULTILINE)
    return found.group(1) if found else None


def _number(value: str | None) -> int | None:
    return None if value is None else int(value)


def _flag(value: str | None) -> bool | None:
    return None if value is None else value.lower() == "true"


def parse_battery(text: str) -> dict[str, Any]:
    """Level, power sources, status and temperature from ``dumpsys battery``."""
    tenths = _number(_match(r"^\s*temperature:\s*(-?\d+)", text))
    return {
        "level_percent": _number(_match(r"^\s*level:\s*(\d+)", text)),
        "scale": _number(_match(r"^\s*scale:\s*(\d+)", text)),
        "ac_powered": _flag(_match(r"^\s*AC powered:\s*(\w+)", text)),
        "usb_powered": _flag(_match(r"^\s*USB powered:\s*(\w+)", text)),
        "wireless_powered": _flag(_match(r"^\s*Wireless powered:\s*(\w+)", text)),
        "status": _number(_match(r"^\s*status:\s*(\d+)", text)),
        "temperature_c": None if tenths is None else tenths / 10,
    }


def parse_thermal(text: str) -> dict[str, Any]:
    """The thermal status and the temperatures the HAL reports, from ``dumpsys thermalservice``."""
    status = _number(_match(r"^\s*Thermal Status:\s*(\d+)", text))
    hal = text.split("Current temperatures from HAL:", 1)
    body = hal[1].split("Current cooling devices from HAL:", 1)[0] if len(hal) == 2 else ""
    temperatures = [
        {"name": m[3], "type": int(m[2]), "value_c": float(m[1]), "status": int(m[4])}
        for m in re.finditer(
            r"Temperature\{mValue=(-?[\d.]+|NaN), mType=(-?\d+), mName=([^,}]*), "
            r"mStatus=(-?\d+)\}",
            body,
        )
    ]
    name = None
    if status is not None and 0 <= status < len(THERMAL_NAMES):
        name = THERMAL_NAMES[status]
    return {"status": status, "status_name": name, "temperatures": temperatures}


def parse_power(text: str) -> dict[str, Any]:
    """Wakefulness and the display's power state, from ``dumpsys power``."""
    return {
        "wakefulness": _match(r"\bmWakefulness=(\w+)", text),
        "display_state": _match(r"Display Power: state=(\w+)", text),
    }


def parse_resumed(text: str) -> str | None:
    """The resumed activity's component, from ``dumpsys activity activities``."""
    return _match(
        r"\b(?:mResumedActivity|ResumedActivity|topResumedActivity)[:=]\s*"
        r"ActivityRecord\{[0-9a-f]+ u\d+ ([^\s}]+)",
        text,
    )


def read_conditions(phone: Phone) -> dict[str, Any]:
    """The phone's conditions now; only named fields are kept."""

    def shell(command: str) -> str:
        return phone.scrub(phone.shell(command))

    airplane = shell("settings get global airplane_mode_on").strip()
    return {
        "read_utc": _utc(),
        "battery": parse_battery(shell("dumpsys battery")),
        "thermal": parse_thermal(shell("dumpsys thermalservice")),
        "airplane_mode": {"1": True, "0": False}.get(airplane),
        "power": parse_power(shell("dumpsys power")),
        "resumed_activity": parse_resumed(shell("dumpsys activity activities")),
    }


def write_conditions(phone: Phone, folder: Path, label: str) -> Path:
    if not LABEL.fullmatch(label):
        raise SystemExit(f"{label!r} is not a plain label (letters, digits, '.', '_', '-')")
    path = folder / f"conditions_{label}.json"
    _refuse_existing(path)
    write_json_atomic(path, {"experiment": EXPERIMENT, "label": label, **read_conditions(phone)})
    return path


def cmd_conditions(args: argparse.Namespace) -> int:
    phone = select_phone(args.serial)
    info = read_device(phone)
    path = write_conditions(phone, RUNS / device_slug(info), args.label)
    print(json.dumps({"device": device_slug(info), "conditions": path.as_posix()}))
    return 0


# ------------------------------------------------------------- phone timing: push and runs


def load_timing(folder: Path = TIMING) -> dict[str, Any]:
    """``prepare.json`` of the phone timing, with every arm's plan, check plan and laptop record
    checked against the hashes it records. A smoke test, a preparation made from a tree with
    changes (the arms' calls come from the candidate code, which `assert_v1_code` does not cover)
    or another thread count than the registered one is refused."""
    path = folder / "prepare.json"
    if not path.is_file():
        raise SystemExit(f"{path.as_posix()} is missing: run `prepare-timing` first")
    doc = _read_json(path)
    if doc.get("smoke") is not False:
        raise SystemExit(f"{path.as_posix()} is a smoke test, not the registered inputs")
    if doc.get("git_dirty") is not False:
        raise SystemExit(f"{path.as_posix()} was made from a tree with changes: commit and prepare")
    if doc.get("threads") != DEFAULT_THREADS:
        raise SystemExit(
            f"{path.as_posix()} was made for {doc.get('threads')} threads, not {DEFAULT_THREADS}"
        )
    for rec in doc["arms"].values():
        for part in ("plan", "check", "laptop"):
            file = folder / rec[part]["file"]
            if not file.is_file() or sha256_file(file) != rec[part]["sha256"]:
                raise SystemExit(f"{file.as_posix()} differs from prepare.json")
    return doc


def timing_files(prepared: Mapping[str, Any], folder: Path = TIMING) -> dict[str, tuple[Path, str]]:
    """Every file the phone needs, by its name there, with its local path and hash: the models,
    the tensor files and each arm's two plans."""
    files = {name: (Path(m["source"]), m["sha256"]) for name, m in prepared["models"].items()}
    files.update({name: (folder / name, t["sha256"]) for name, t in prepared["tensors"].items()})
    for rec in prepared["arms"].values():
        for part in ("plan", "check"):
            files[rec[part]["file"]] = (folder / rec[part]["file"], rec[part]["sha256"])
    return files


def push_files(phone: Phone, files: Mapping[str, Path]) -> None:
    """Push each file to the phone folder under its name; files whose local name is that name go
    `PUSH_BATCH` to an adb call."""
    batch = [path for name, path in files.items() if path.name == name]
    for name, path in files.items():
        if path.name != name:
            phone.adb(["push", str(path), phone_path(name)], timeout=PUSH_SECONDS)
    for start in range(0, len(batch), PUSH_BATCH):
        chunk = [str(p) for p in batch[start : start + PUSH_BATCH]]
        phone.adb(["push", *chunk, PHONE_DIR], timeout=PUSH_SECONDS)


def pull_files(phone: Phone, names: Sequence[str], folder: Path) -> None:
    """Pull the phone files ``names`` into ``folder``, `PUSH_BATCH` to an adb call."""
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(names), PUSH_BATCH):
        chunk = [phone_path(n) for n in names[start : start + PUSH_BATCH]]
        phone.adb(["pull", *chunk, str(folder)], timeout=PUSH_SECONDS)


def verify_phone_files(phone: Phone, expected: Mapping[str, str]) -> str:
    """`verify_phone_hashes` for many files: ``sha256sum`` run in the phone folder on
    `PUSH_BATCH` names at a time. Returns ``sha256sum`` or ``unavailable``."""
    if not phone.test("command -v sha256sum"):
        return "unavailable"
    names = sorted(expected)
    for start in range(0, len(names), PUSH_BATCH):
        chunk = names[start : start + PUSH_BATCH]
        listed = " ".join(shlex.quote(n) for n in chunk)
        output = phone.shell(
            f"cd {shlex.quote(PHONE_DIR)} && sha256sum {listed}", timeout=PUSH_SECONDS
        )
        got: dict[str, str] = {}
        for line in output.splitlines():
            digest, _, name = line.strip().partition(" ")
            got[name.strip().lstrip("*")] = digest
        for name in chunk:
            if got.get(name) != expected[name]:
                raise SystemExit(
                    f"{name} on the phone has SHA-256 {got.get(name) or 'unknown'}, "
                    f"expected {expected[name]}"
                )
    return "sha256sum"


def cmd_push_timing(args: argparse.Namespace) -> int:
    prepared = load_timing()
    files = timing_files(prepared)
    for path, sha in files.values():
        if not path.is_file() or sha256_file(path) != sha:
            raise SystemExit(f"{path.as_posix()} is missing or differs from prepare.json")
    if not args.apk.is_file():
        raise SystemExit(f"{args.apk.as_posix()} is missing: build the app first")
    tree = build_state(args.apk)
    if not tree["build_record_matches"]:
        raise SystemExit("no build record for this commit and APK: run `build` first")
    if not tree["tree_clean_after_build"]:
        raise SystemExit("git status was not empty straight after the build: commit and build")
    apk_permissions = check_apk_manifest(args.apk)
    phone = select_phone(args.serial)
    info = read_device(phone)
    if REQUIRED_ABI not in str(info["abi_list"]).split(","):
        raise SystemExit(f"the phone does not list {REQUIRED_ABI}")
    slug = device_slug(info)
    folder = RUNS / slug
    _refuse_existing(folder / PUSH_RECORD)
    reason = (args.first_phone_reason or "").strip() or None
    selected_on = _read_json(SELECTION).get("device") if SELECTION.is_file() else None
    if selected_on is None and slug != FIRST_PHONE and reason is None:
        raise SystemExit(
            f"{slug} is not the registered first phone ({FIRST_PHONE}); give "
            "--first-phone-reason to record why it takes the M52's place"
        )
    if reason is not None and (slug == FIRST_PHONE or selected_on is not None):
        raise SystemExit("--first-phone-reason is only for a phone replacing the M52 before select")
    folder.mkdir(parents=True, exist_ok=True)

    install_app(phone, args.apk)
    push_files(phone, {name: path for name, (path, _) in files.items()})
    method = verify_phone_files(phone, {name: sha for name, (_, sha) in files.items()})
    doc = {
        "experiment": EXPERIMENT,
        "part": "b",
        "pushed_utc": _utc(),
        "prepare_sha256": sha256_file(TIMING / "prepare.json"),
        **tree,
        "apk_bytes": args.apk.stat().st_size,
        "apk_permissions": apk_permissions,
        "device": info,
        "first_phone_reason": reason,
        "phone_hash_check": method,
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path, _ in files.values()),
        "files": {name: sha for name, (_, sha) in sorted(files.items())},
    }
    write_json_atomic(folder / PUSH_RECORD, doc)
    print(
        json.dumps(
            {
                "device": device_slug(info),
                "files": len(files),
                "bytes": doc["bytes"],
                "phone_hash_check": method,
            }
        )
    )
    return 0


def first_phone_order(to_phone: Sequence[str]) -> list[tuple[str, int]]:
    """The first phone's registered order: run 1 of the arms going to the phone in
    `candidate.ARMS` order, then run 2 in reverse (`candidate.arm_order`)."""
    from glypharis.efficiency.candidate import arm_order

    return [(arm, run) for run in (1, 2) for arm in arm_order(run) if arm in to_phone]


def timing_results(folder: Path) -> list[tuple[str, int]]:
    """The (arm, run) of every timing result pulled into ``folder``."""
    found = [TIMING_RESULT.fullmatch(p.name) for p in sorted(folder.glob("timing_*.json"))]
    return [(m["arm"], int(m["run"])) for m in found if m is not None]


def first_phone_reason(slug: str) -> str | None:
    """Why ``slug`` takes the registered first phone's place (H11: the Moto G35 5G only if the M52
    cannot be used), as ``push-timing --first-phone-reason`` recorded it; None for the M52. Any
    other phone without a recorded reason is refused as the first phone."""
    if slug == FIRST_PHONE:
        return None
    path = RUNS / slug / PUSH_RECORD
    reason = _read_json(path).get("first_phone_reason") if path.is_file() else None
    if not reason:
        raise SystemExit(
            f"{slug} is not the registered first phone ({FIRST_PHONE}): it takes its place only "
            "with the reason recorded (push-timing --first-phone-reason)"
        )
    return str(reason)


def device_order(slug: str, to_phone: Sequence[str]) -> tuple[str, list[tuple[str, int]]]:
    """The phone's role and its registered (arm, run) order. Before ``select`` the first phone is
    the registered M52, or another phone whose push recorded why it takes the M52's place, and
    only one phone may hold timing results; after it, the phone the selection came from stays the
    first, and any other is a second phone, which times the selected arm twice and decides
    nothing."""
    if SELECTION.is_file():
        selection = _read_json(SELECTION)
        if selection.get("device") == slug:
            return "first", first_phone_order(to_phone)
        return "second", [(str(selection["selected"]), 1), (str(selection["selected"]), 2)]
    first_phone_reason(slug)
    others = sorted(
        {p.parent.name for p in RUNS.glob("*/timing_*.json") if TIMING_RESULT.fullmatch(p.name)}
        - {slug}
    )
    if others:
        raise SystemExit(
            f"{others[0]} holds the first phone's timing runs: another phone times the selected "
            "arm only, after `select`"
        )
    return "first", first_phone_order(to_phone)


def _order_text(steps: Iterable[tuple[str, int]]) -> str:
    return ", ".join(f"{arm} run {run}" for arm, run in steps)


def next_step(
    order: Sequence[tuple[str, int]], done: Iterable[tuple[str, int]]
) -> tuple[str, int] | None:
    """The first (arm, run) of ``order`` without a result, None when every one has one. The done
    steps must be the head of the order."""
    done = set(done)
    unknown = sorted(done - set(order))
    if unknown:
        raise SystemExit(f"results outside the registered order: {_order_text(unknown)}")
    for k, step in enumerate(order):
        if step not in done:
            later = [s for s in order[k + 1 :] if s in done]
            if later:
                raise SystemExit(
                    f"{_order_text(later)} ran before {_order_text([step])}: the order was not kept"
                )
            return step
    return None


def earlier_pids(folder: Path) -> dict[int, str]:
    """The process ID of every timing and check result pulled into ``folder``."""
    out: dict[int, str] = {}
    for path in sorted(folder.glob("*.json")):
        if TIMING_RESULT.fullmatch(path.name) or CHECK_RESULT.fullmatch(path.name):
            pid = _read_json(path).get("pid")
            if isinstance(pid, int) and not isinstance(pid, bool):
                out[pid] = path.name
    return out


def checked_result(pulled: Path, mode: str, folder: Path) -> dict[str, Any]:
    """The pulled result, refused unless written by ``mode`` and, when it ran, by a process no
    earlier timing or check result on this phone came from. A ``bad_request`` is refused too: the
    app could not read its request or files, so the result says nothing about the model and the
    same step runs again once the files are right."""
    try:
        doc = _read_json(pulled)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{pulled.name} is not a result: {exc}") from exc
    if doc.get("mode") != mode:
        raise SystemExit(f"{pulled.name} is a {doc.get('mode')!r} result, not a {mode} result")
    if doc.get("status") == "bad_request":
        detail = (doc.get("request_error") or {}).get("message")
        raise SystemExit(
            f"the app could not start the {mode} pass ({detail}); {pulled.name} is kept as it was "
            "pulled and not taken as a result"
        )
    pid = doc.get("pid")
    if isinstance(pid, bool) or (pid is not None and not isinstance(pid, int)):
        raise SystemExit(f"{pulled.name}: pid {pid!r} is not a process ID")
    if doc.get("status") == "ok" and pid is None:
        raise SystemExit(f"{pulled.name} names no pid: it cannot show a new process")
    seen = earlier_pids(folder)
    if pid is not None and pid in seen:
        raise SystemExit(
            f"{pulled.name}: pid {pid} is that of {seen[pid]}, so it is not a new process; it is "
            "kept as it was pulled and not taken as a result"
        )
    return doc


def timing_phone(args: argparse.Namespace) -> tuple[dict[str, Any], Phone, dict[str, Any], Path]:
    """What a timing or check run needs: this preparation, the clean build ``run`` needs, the
    phone, and this preparation pushed to it with this APK. Returns the preparation, the phone,
    its record and its folder."""
    prepared = load_timing()
    if not args.apk.is_file():
        raise SystemExit(f"{args.apk.as_posix()} is missing: build the app first")
    tree = build_state(args.apk)
    if not tree["build_record_matches"]:
        raise SystemExit("no build record for this commit and APK: run `build` first")
    if not tree["tree_clean_after_build"]:
        raise SystemExit("git status was not empty straight after the build: commit and build")
    if not tree["tree_clean_at_run"]:
        raise SystemExit("the tree is not clean: commit, run `build` again and push again")
    check_apk_manifest(args.apk)
    phone = select_phone(args.serial)
    info = read_device(phone)
    folder = RUNS / device_slug(info)
    if not (folder / PUSH_RECORD).is_file():
        raise SystemExit(f"{device_slug(info)}: nothing pushed: run `push-timing` first")
    push = _read_json(folder / PUSH_RECORD)
    if push.get("prepare_sha256") != sha256_file(TIMING / "prepare.json"):
        raise SystemExit("the phone holds another preparation: move push.json aside and push again")
    if push.get("apk_sha256") != tree["apk_sha256"]:
        raise SystemExit("the phone runs another APK: move push.json aside and push again")
    return prepared, phone, info, folder


def _run_record(folder: Path, name: str, doc: Mapping[str, Any], **fields: Any) -> None:
    write_json_atomic(
        folder / name,
        {
            "experiment": EXPERIMENT,
            "part": "b",
            **fields,
            "git_commit": _git("rev-parse", "HEAD"),
            "push_sha256": sha256_file(folder / PUSH_RECORD),
            "status": doc.get("status"),
            "pid": doc.get("pid"),
        },
    )


def cmd_time_run(args: argparse.Namespace) -> int:
    prepared, phone, info, folder = timing_phone(args)
    slug = device_slug(info)
    if args.arm not in prepared["to_phone"]:
        raise SystemExit(f"{args.arm} is not an arm going to the phone ({prepared['to_phone']})")
    role, order = device_order(slug, prepared["to_phone"])
    step = next_step(order, timing_results(folder))
    if step is None:
        raise SystemExit(f"{slug} holds every registered run: {_order_text(order)}")
    if step != (args.arm, args.run):
        raise SystemExit(
            f"the next run on {slug} is {_order_text([step])}, not {args.arm} run {args.run} "
            f"(order: {_order_text(order)})"
        )
    result = timing_name(args.arm, args.run)
    label = f"{args.arm}_run{args.run}"
    pulled = folder / f"{result}.part"
    # A refused attempt's pulled result stays as it was: move it aside to try the step again.
    _refuse_existing(pulled, folder / result)
    write_conditions(phone, folder, f"{label}_before")
    stale = (shlex.quote(phone_path(n)) for n in (result, f"{result}.tmp"))
    phone.shell(f"rm -f {' '.join(stale)}")
    started = _utc()
    written = False
    try:
        start_app(
            phone,
            {
                "mode": "timing",
                "plan": plan_name(args.arm),
                "result": result,
                "threads": int(prepared["threads"]),
                "run": int(args.run),
            },
        )
        written = wait_for_result(phone, result, args.wait_seconds)
        if written:
            phone.adb(["pull", phone_path(result), str(pulled)])
    finally:
        phone.call(["shell", f"am force-stop {APP_ID}"])
    ended = _utc()
    if written:
        doc = checked_result(pulled, "timing", folder)
        os.replace(pulled, folder / result)
    else:
        doc = died_result("timing", args.arm, args.run)
        write_json_atomic(folder / result, doc)
    write_conditions(phone, folder, f"{label}_after")
    _run_record(
        folder,
        f"record_{label}.json",
        doc,
        arm=args.arm,
        run=args.run,
        role=role,
        started_utc=started,
        ended_utc=ended,
        plan_sha256=prepared["arms"][args.arm]["plan"]["sha256"],
        result_sha256=sha256_file(folder / result),
    )
    print(
        json.dumps(
            {
                "device": slug,
                "arm": args.arm,
                "run": args.run,
                "status": doc.get("status"),
                "pid": doc.get("pid"),
            }
        )
    )
    return 0


def cmd_check_run(args: argparse.Namespace) -> int:
    prepared, phone, info, folder = timing_phone(args)
    slug = device_slug(info)
    role, order = device_order(slug, prepared["to_phone"])
    step = next_step(order, timing_results(folder))
    if step is not None:
        raise SystemExit(f"{slug}: the timing runs come first; the next is {_order_text([step])}")
    if args.arm not in {arm for arm, _ in order}:
        raise SystemExit(f"{args.arm} was not timed on {slug}")
    result = checkresult_name(args.arm)
    pulled = folder / f"{result}.part"
    # A refused attempt's pulled result stays as it was: move it aside to try the step again.
    _refuse_existing(folder / result, pulled)
    plan = _read_json(TIMING / prepared["arms"][args.arm]["check"]["file"])
    stale = [shlex.quote(phone_path(n)) for n in (result, f"{result}.tmp")]
    stale.append(f"{shlex.quote(PHONE_DIR)}/c_{args.arm}_*.gwt*")
    phone.shell(f"rm -f {' '.join(stale)}")
    started = _utc()
    written = False
    try:
        start_app(
            phone,
            {
                "mode": "check",
                "plan": check_name(args.arm),
                "result": result,
                "threads": int(prepared["threads"]),
            },
        )
        written = wait_for_result(phone, result, args.wait_seconds)
        if written:
            phone.adb(["pull", phone_path(result), str(pulled)])
    finally:
        phone.call(["shell", f"am force-stop {APP_ID}"])
    ended = _utc()
    if written:
        doc = checked_result(pulled, "check", folder)
        if doc.get("status") == "ok":
            pull_files(phone, [c["output"] for c in plan["calls"]], folder / CHECK_OUTPUTS)
        os.replace(pulled, folder / result)
    else:
        doc = died_result("check", args.arm, None)
        write_json_atomic(folder / result, doc)
    _run_record(
        folder,
        f"record_check_{args.arm}.json",
        doc,
        arm=args.arm,
        role=role,
        started_utc=started,
        ended_utc=ended,
        plan_sha256=prepared["arms"][args.arm]["check"]["sha256"],
        result_sha256=sha256_file(folder / result),
    )
    print(json.dumps({"device": slug, "arm": args.arm, "status": doc.get("status")}))
    return 0


# ---------------------------------------------------------- phone timing: report and selection


def laptop_session(path: Path, threads: int) -> Any:
    """The laptop's session over ``path`` with the registered options (`threads.ort_session`,
    checked by `threads.check_session`). Tests replace this one function."""
    from glypharis.efficiency.threads import check_session, ort_session

    session = ort_session(path, threads)
    check_session(session, threads)
    return session


def registered_options(threads: int) -> dict[str, Any]:
    """The session options of `threads.ort_session` in the names the app records them."""
    return {
        "providers": ["CPUExecutionProvider"],
        "intra_op_num_threads": threads,
        "execution_mode": "ORT_SEQUENTIAL",
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "enable_cpu_mem_arena": False,
        "enable_mem_pattern": False,
        "allow_spinning": "default",
    }


def _is_time(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _stats(values: Iterable[float]) -> dict[str, Any]:
    t = np.asarray(list(values), dtype=np.float64)
    if t.size == 0:
        return {"count": 0, "mean_ms": None, "p95_ms": None}
    return {"count": int(t.size), "mean_ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95))}


def result_problems(
    result: Mapping[str, Any], plan_sha256: str, model_sha256: Mapping[str, str], threads: int
) -> list[str]:
    """What makes a finished result not the run of its plan: another plan or model file (the app
    records each model's hash by its role), another ONNX Runtime or other session options."""
    problems = []
    if result.get("plan_sha256") != plan_sha256:
        problems.append("the plan's hash differs")
    if result.get("model_sha256") != dict(model_sha256):
        problems.append("the model files' hashes differ")
    if result.get("onnxruntime_version") != ORT_VERSION:
        problems.append(f"ONNX Runtime {result.get('onnxruntime_version')}, not {ORT_VERSION}")
    options = result.get("session_options") or {}
    want = registered_options(threads)
    if any(options.get(k) != want[k] for k in SESSION_OPTION_KEYS):
        problems.append("the session options differ")
    return problems


def _item_problem(got: Any, want: Mapping[str, Any]) -> str | None:
    """Why one timed item does not fit its plan's item, or None."""
    if not isinstance(got, Mapping):
        return f"item {want['key']}: not a record"
    calls = got.get("calls_ms")
    parts = [got.get("network_ms"), got.get("detector_ms"), got.get("recognizer_ms")]
    if got.get("key") != want["key"]:
        return f"item {got.get('key')!r} where the plan has {want['key']!r}"
    if not isinstance(calls, list) or len(calls) != len(want["calls"]):
        return f"item {want['key']}: not one time per call"
    if not all(_is_time(t) for t in [*parts, *calls]):
        return f"item {want['key']}: a time is missing or not a finite number"
    if abs(sum(calls) - got["network_ms"]) > 1e-3 * max(1.0, got["network_ms"]):
        return f"item {want['key']}: network_ms is not the sum of its calls"
    return None


def _no_report(arm: str, what: str, result: Mapping[str, Any]) -> SystemExit:
    detail = (result.get("request_error") or {}).get("message")
    return SystemExit(f"{arm}: the app could not start the {what} ({detail}); no report")


def timing_summary(
    result: Mapping[str, Any], plan: Mapping[str, Any], plan_sha256: str, threads: int
) -> dict[str, Any]:
    """One timing run, checked against its plan: its status and errors, the device states the app
    recorded and, when it finished, the times and peaks. A run the app could not start, or a
    finished one that is not its plan's, gives no report."""
    arm = plan["arm"]
    status = result.get("status")
    if status == "bad_request":
        raise _no_report(arm, "timing run", result)
    summary: dict[str, Any] = {
        "status": status,
        "pid": result.get("pid"),
        "failed_model": result.get("failed_model"),
        "onnxruntime_version": result.get("onnxruntime_version"),
        "session_options": result.get("session_options"),
        "session_error": _scrub_error(result.get("session_error")),
        "run_error": _scrub_error(result.get("run_error")),
        **{key: result.get(key) for key in DEVICE_STATE_KEYS},
        "measured": False,
    }
    if status != "ok":
        return summary
    models = {role: plan["models"][role]["sha256"] for role in ("detector", "recognizer")}
    problems = result_problems(result, plan_sha256, models, threads)
    want = plan["items"]
    got = result.get("items")
    if not isinstance(got, list) or len(got) != len(want):
        problems.append(f"{len(got) if isinstance(got, list) else 'no'} items, not {len(want)}")
        got = []
    item = next((p for g, w in zip(got, want, strict=False) if (p := _item_problem(g, w))), None)
    if item is not None:
        problems.append(item)
    hwm = result.get("vm_hwm_kb")
    if not isinstance(hwm, int) or isinstance(hwm, bool) or hwm <= 0:
        problems.append("no VmHWM")
    if problems:
        raise SystemExit(f"{arm}: the run is not its plan's ({'; '.join(problems)}); no report")
    pairs = list(zip(want, got, strict=True))
    timed = [g for w, g in pairs if w["kind"] == "page" and not w["warmup"]]
    warm = [g for w, g in pairs if w["kind"] == "page" and w["warmup"]]
    lines = [g for w, g in pairs if w["kind"] == "line"]
    summary.update(
        measured=True,
        vm_hwm_kb=hwm,
        vm_rss_kb=result.get("vm_rss_kb"),
        vm_hwm_kb_before_items=result.get("vm_hwm_kb_before_items"),
        vm_rss_kb_before_items=result.get("vm_rss_kb_before_items"),
        pages={
            **_stats(g["network_ms"] for g in timed),
            "detector_mean_ms": _stats(g["detector_ms"] for g in timed)["mean_ms"],
            "recognizer_mean_ms": _stats(g["recognizer_ms"] for g in timed)["mean_ms"],
        },
        warmup_pages=_stats(g["network_ms"] for g in warm),
        lines=_stats(g["network_ms"] for g in lines),
        page_keys=[w["key"] for w, _ in pairs if w["kind"] == "page" and not w["warmup"]],
        page_network_ms=[float(g["network_ms"]) for g in timed],
        line_network_ms=[float(g["network_ms"]) for g in lines],
    )
    return summary


def compare_call(
    reference: np.ndarray,
    phone: np.ndarray | None,
    frames: Sequence[int],
    texts: Sequence[str],
    charset: Charset,
    app: Mapping[str, Any],
) -> dict[str, Any]:
    """One check call: the phone's logits ``(T, N, classes)`` against the laptop's over each
    line's own frames (``frames`` clamped to T), and each line's greedy text against the laptop's
    recorded one. ``app`` is the app's record of the call (``output_shape``, ``logits_finite``)."""
    out: dict[str, Any] = {
        "present": phone is not None,
        "shape_equal": False,
        "finite": False,
        "max_abs_diff": None,
        "diff_finite": True,
        "lines": len(frames),
        "identical_text": 0,
        "laptop_identical_text": 0,
    }
    if phone is None:
        return out
    shape = [int(d) for d in phone.shape]
    out["shape_equal"] = shape == list(reference.shape) and app.get("output_shape") == shape
    out["finite"] = bool(np.isfinite(phone).all()) and app.get("logits_finite") is True
    if not out["shape_equal"]:
        return out
    worst = 0.0
    for j, (f, text) in enumerate(zip(frames, texts, strict=True)):
        n = min(int(f), reference.shape[0])
        got, want = phone[:n, j], reference[:n, j]
        if n:
            diff = float(np.abs(got - want).max())
            if math.isfinite(diff):
                worst = max(worst, diff)
            else:
                out["diff_finite"] = False
        out["identical_text"] += charset.ctc_greedy_decode(got.argmax(-1).tolist()) == text
        out["laptop_identical_text"] += charset.ctc_greedy_decode(want.argmax(-1).tolist()) == text
    out["max_abs_diff"] = worst if out["diff_finite"] else None
    return out


def check_verdict(per_call: Sequence[Mapping[str, Any]], judged: bool) -> dict[str, Any]:
    """The check rule over an arm's calls (item 15). A judged arm needs its logits within
    `LOGIT_TOL` and identical text on every line; ``fused-int8`` needs its outputs present, finite
    and of the laptop's shape, its differences being reported."""
    present = [c for c in per_call if c["present"]]
    compared = len(present) == len(per_call) and all(c["shape_equal"] for c in per_call)
    reasons = []
    if len(present) != len(per_call):
        reasons.append("outputs_missing")
    if not all(c["shape_equal"] for c in present):
        reasons.append("wrong_shape")
    if not all(c["finite"] for c in present):
        reasons.append("non_finite_logits")
    finite_diff = all(c["diff_finite"] for c in per_call)
    worst = None
    if compared and finite_diff:
        worst = max((c["max_abs_diff"] for c in per_call), default=0.0)
    lines = sum(c["lines"] for c in per_call)
    identical = sum(c["identical_text"] for c in per_call)
    if judged and compared:
        if worst is None or worst > LOGIT_TOL:
            reasons.append("logits_differ")
        if identical != lines:
            reasons.append("text_differs")
    return {
        "lines": lines,
        "logit_max_abs_diff": worst,
        "identical_text": identical if compared else None,
        "laptop_rerun_identical_text": sum(c["laptop_identical_text"] for c in per_call),
        "passed": not reasons,
        "reasons": reasons,
    }


def read_tensor(folder: Path, name: str, sha256: str) -> dict[str, np.ndarray]:
    """A tensor file, refused unless it has the hash its plan names."""
    data = (folder / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise SystemExit(f"{name} differs from the hash its plan names")
    return parse_gwt(data, name)


def check_on_phone(
    arm: str, folder: Path, prepared: Mapping[str, Any], charset: Charset
) -> dict[str, Any]:
    """The arm's check result against the laptop: the laptop's logits come from the arm's own
    laptop session on the same tensor files, its texts from what ``prepare-timing`` recorded."""
    rec = prepared["arms"][arm]
    plan = _read_json(TIMING / rec["check"]["file"])
    laptop = _read_json(TIMING / rec["laptop"]["file"])["calls"]
    result = _read_json(folder / checkresult_name(arm))
    status = result.get("status")
    if status == "bad_request":
        raise _no_report(arm, "check", result)
    out: dict[str, Any] = {
        "status": status,
        "pid": result.get("pid"),
        "judged": arm not in UNJUDGED_ARMS,
        "calls": len(plan["calls"]),
        "session_error": _scrub_error(result.get("session_error")),
        "run_error": _scrub_error(result.get("run_error")),
    }
    if status != "ok":
        return {
            **out,
            "lines": sum(len(c["frames"]) for c in laptop),
            "logit_max_abs_diff": None,
            "identical_text": None,
            "laptop_rerun_identical_text": None,
            "passed": False,
            "reasons": [f"status_{status}"],
        }
    threads = int(prepared["threads"])
    models = {"recognizer": plan["model"]["sha256"]}
    problems = result_problems(result, rec["check"]["sha256"], models, threads)
    app_calls = result.get("calls")
    if not isinstance(app_calls, list) or len(app_calls) != len(plan["calls"]):
        problems.append("not one record per call")
    if problems:
        raise SystemExit(f"{arm}: the check is not its plan's ({'; '.join(problems)}); no report")
    model = prepared["models"][plan["model"]["file"]]
    source = Path(model["source"])
    model_record(source, model["sha256"])
    session = laptop_session(source, threads)
    per_call = []
    for call, lap, app in zip(plan["calls"], laptop, app_calls, strict=True):
        if (call["tensor"], call["output"]) != (lap["tensor"], lap["output"]):
            raise SystemExit(f"{arm}: the laptop record does not follow the check plan")
        feeds = read_tensor(TIMING, call["tensor"], plan["tensors"][call["tensor"]]["sha256"])
        reference = np.asarray(session.run(["logits"], feeds)[0])
        if [int(d) for d in reference.shape] != lap["shape"]:
            raise SystemExit(
                f"{arm}, {call['source']}: the laptop's session gives {list(reference.shape)}, "
                f"not the {lap['shape']} recorded"
            )
        path = folder / CHECK_OUTPUTS / call["output"]
        phone = read_gwt(path).get("logits") if path.is_file() else None
        per_call.append(compare_call(reference, phone, lap["frames"], lap["texts"], charset, app))
    return {**out, **check_verdict(per_call, out["judged"])}


def _json_or_none(path: Path) -> dict[str, Any] | None:
    return _read_json(path) if path.is_file() else None


def arm_entry(
    arm: str, folder: Path, prepared: Mapping[str, Any], charset: Charset
) -> dict[str, Any]:
    """One arm on one phone: both timing runs, their means and peaks, and the check."""
    rec = prepared["arms"][arm]
    plan = _read_json(TIMING / rec["plan"]["file"])
    runs = []
    for run in (1, 2):
        result = _read_json(folder / timing_name(arm, run))
        summary = timing_summary(result, plan, rec["plan"]["sha256"], int(prepared["threads"]))
        label = f"{arm}_run{run}"
        runs.append(
            {
                "run": run,
                **summary,
                "conditions_before": _json_or_none(folder / f"conditions_{label}_before.json"),
                "conditions_after": _json_or_none(folder / f"conditions_{label}_after.json"),
            }
        )
    check = check_on_phone(arm, folder, prepared, charset)
    measured = all(r["measured"] for r in runs)
    loaded = all(r["status"] not in LOAD_FAILURES for r in runs)
    loaded = loaded and check["status"] not in LOAD_FAILURES
    both = np.mean([r["page_network_ms"] for r in runs], axis=0) if measured else None
    lines = np.mean([r["line_network_ms"] for r in runs], axis=0) if measured else None
    peak = max(r["vm_hwm_kb"] for r in runs) if measured else None
    return {
        "recognizer": plan["models"]["recognizer"]["file"],
        "recognizer_sha256": plan["models"]["recognizer"]["sha256"],
        "plan_sha256": rec["plan"]["sha256"],
        "runs": runs,
        "measured": measured,
        "loaded": loaded,
        "run_mean_ms": [r["pages"]["mean_ms"] if r["measured"] else None for r in runs],
        "run_p95_ms": [r["pages"]["p95_ms"] if r["measured"] else None for r in runs],
        "pages_both_runs": None if both is None else _stats(both),
        "lines_both_runs": None if lines is None else _stats(lines),
        "peak_vm_hwm_kb": peak,
        "peak_vm_hwm_mib": None if peak is None else round(peak / 1024, 1),
        "check": check,
        "check_passed": bool(check["passed"]),
    }


def phone_conditions(entries: Mapping[str, dict[str, Any]]) -> None:
    """The speed and memory conditions of each arm against fused, added to its entry: on the
    timed pages' network times of both runs (`candidate.speed_tests`) and on the peaks
    (`candidate.memory_test`). An arm without two full runs, or with a fused that has none, does
    not meet them."""
    from glypharis.efficiency.candidate import DEFAULT_ARM, memory_test, speed_tests

    fused = entries[DEFAULT_ARM]
    for arm, entry in entries.items():
        if arm == DEFAULT_ARM:
            continue
        if not (entry["measured"] and fused["measured"]):
            note = "not measured: fused and the arm need two finished timing runs each"
            entry["speed"] = {"met": False, "note": note}
            entry["memory"] = {"met": False, "note": note}
            continue
        for run, ref in zip(entry["runs"], fused["runs"], strict=True):
            if run["page_keys"] != ref["page_keys"]:
                raise SystemExit(f"{arm} run {run['run']} timed other pages than fused")
        entry["speed"] = speed_tests(
            [r["page_network_ms"] for r in entry["runs"]],
            [r["page_network_ms"] for r in fused["runs"]],
        )
        entry["memory"] = memory_test(
            [r["vm_hwm_kb"] for r in entry["runs"]], [r["vm_hwm_kb"] for r in fused["runs"]]
        )


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _thermal(run: Mapping[str, Any]) -> str:
    def name(state: Any) -> str:
        if not isinstance(state, Mapping):
            return "n/a"
        return str(state.get("status_name") or state.get("status"))

    return f"{name(run.get('thermal_start'))} to {name(run.get('thermal_end'))}"


def _phone_md(doc: Mapping[str, Any]) -> str:
    device = doc["device"]
    rows = [
        f"# {EXPERIMENT} part b: {device.get('model')} ({device['slug']}), phone timing and check",
        "",
        "Generated by `scripts/phase22_android.py phone-report`; do not edit by hand. "
        f"{doc['scope']} {doc['threads']} intra-op threads; this phone is the {doc['role']} "
        f"phone and {doc['decides']}.",
        "",
        "| Arm | Mean ms, run 1 | Mean ms, run 2 | Mean ms, both runs | p95 ms, both runs | "
        "Peak VmHWM MiB | Thermal, run 1 | Thermal, run 2 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm, e in doc["arms"].items():
        both = e["pages_both_runs"] or {}
        rows.append(
            f"| {arm} | {_ms(e['run_mean_ms'][0])} | {_ms(e['run_mean_ms'][1])} "
            f"| {_ms(both.get('mean_ms'))} | {_ms(both.get('p95_ms'))} "
            f"| {_ms(e['peak_vm_hwm_mib'])} | {_thermal(e['runs'][0])} | {_thermal(e['runs'][1])} |"
        )
    pages = "n/a" if doc["pages"] is None else doc["pages"]
    rows += [
        "",
        f"Times are per timed page ({pages} per run). Check against the laptop over each line's "
        f"own frames ({doc['check_rule']}):",
        "",
        "| Arm | Lines | Max logit difference | Identical text | Judged | Passed |",
        "|---|---|---|---|---|---|",
    ]
    for arm, e in doc["arms"].items():
        c = e["check"]
        diff = "n/a" if c["logit_max_abs_diff"] is None else f"{c['logit_max_abs_diff']:.2e}"
        same = "n/a" if c["identical_text"] is None else f"{c['identical_text']} of {c['lines']}"
        rows.append(
            f"| {arm} | {c['lines']} | {diff} | {same} | {'yes' if c['judged'] else 'no'} "
            f"| {'yes' if c['passed'] else 'no'} |"
        )
    if doc["role"] == "first":
        rows += [
            "",
            "Speed and memory conditions against fused (research/hypotheses.md, H11): "
            "`mean_ci(t_a - 0.9 * t_f, 'pages')` below 0 over both runs and on each run alone, "
            "and the peak VmHWM not above fused's.",
            "",
            "| Arm | mean(t_a - 0.9 t_f) ms, both runs | Speed met | Memory met |",
            "|---|---|---|---|",
        ]
        for arm, e in doc["arms"].items():
            if "speed" not in e:
                continue
            tests = e["speed"].get("tests")
            cell = "n/a"
            if tests:
                iv = tests["both_runs"]["interval"]
                cell = f"{iv['value']:+.1f} [{iv['low']:+.1f}, {iv['high']:+.1f}]"
            rows.append(
                f"| {arm} | {cell} | {'yes' if e['speed']['met'] else 'no'} "
                f"| {'yes' if e['memory']['met'] else 'no'} |"
            )
    if doc.get("first_phone_reason"):
        rows += ["", f"This phone took the M52's place: {doc['first_phone_reason']}."]
    rows += ["", f"Part b: {doc['part_b']}."]
    return "\n".join(rows) + "\n"


def sitting_records(
    folder: Path, order: Sequence[tuple[str, int]], arms: Sequence[str]
) -> dict[str, str]:
    """The run record of every timing and check run of the phone's sitting, each refused unless
    it was made under the current ``push.json``: one install, one APK and one preparation for
    the whole sitting. Returns each record's commit, by record name."""
    push_sha = sha256_file(folder / PUSH_RECORD)
    names = [f"record_{arm}_run{run}.json" for arm, run in order]
    names += [f"record_check_{arm}.json" for arm in arms]
    commits = {}
    for name in names:
        path = folder / name
        if not path.is_file():
            raise SystemExit(f"{folder.name}: {name} is missing; no report")
        record = _read_json(path)
        if record.get("push_sha256") != push_sha:
            raise SystemExit(
                f"{folder.name}: {name} was made under another push.json; a sitting has one "
                "install, one APK and one preparation"
            )
        commits[name] = str(record.get("git_commit"))
    return commits


def cmd_phone_report(args: argparse.Namespace) -> int:
    from glypharis.efficiency.candidate import DEFAULT_ARM

    slug = args.device
    folder = RUNS / slug
    out = PHONE_REPORTS / f"phone_{slug}.json"
    _refuse_existing(out, out.with_suffix(".md"))
    prepared = load_timing()
    if not (folder / PUSH_RECORD).is_file():
        raise SystemExit(f"{slug}: nothing was pushed to this phone")
    push = _read_json(folder / PUSH_RECORD)
    if push.get("prepare_sha256") != sha256_file(TIMING / "prepare.json"):
        raise SystemExit(f"{slug} was pushed another preparation")
    role, order = device_order(slug, prepared["to_phone"])
    step = next_step(order, timing_results(folder))
    if step is not None:
        raise SystemExit(f"{slug}: the timing runs are not finished; next: {_order_text([step])}")
    arms = list(dict.fromkeys(arm for arm, _ in order))
    missing = [a for a in arms if not (folder / checkresult_name(a)).is_file()]
    if missing:
        raise SystemExit(f"{slug}: no check result for {', '.join(missing)}; run check-run first")
    commits = sitting_records(folder, order, arms)
    charset = Charset(list(prepared["characters"]))
    entries = {arm: arm_entry(arm, folder, prepared, charset) for arm in arms}
    fused_failed: bool | None = None
    if role == "first":
        phone_conditions(entries)
        fused = entries[DEFAULT_ARM]
        fused_failed = not (fused["loaded"] and fused["check_passed"])
        if fused_failed:
            part_b = "failed: fused failed the phone check"
        elif not fused["measured"]:
            part_b = "failed: fused did not finish both timing runs"
        else:
            part_b = "passed"
    else:
        arm = arms[0]  # a second phone times the selected arm only
        e = entries[arm]
        if e["loaded"] and e["measured"] and e["check_passed"]:
            part_b = f"passed: {arm} loaded, finished both runs and passed its check"
        else:
            part_b = f"failed on this phone: {arm} did not load, finish or pass its check"
        part_b += "; this phone changes no decision"
    first = entries[arms[0]]
    doc = {
        "note": "Generated by scripts/phase22_android.py phone-report; do not edit by hand.",
        "experiment": EXPERIMENT,
        "part": "b: phone timing and check",
        "scope": SCOPE,
        "device": {"slug": slug, **push["device"]},
        "role": role,
        "first_phone_reason": push.get("first_phone_reason") if role == "first" else None,
        "decides": (
            "its runs decide item 14's speed and memory conditions"
            if role == "first"
            else "it changes no decision (a second named device)"
        ),
        "order": [f"{arm} run {run}" for arm, run in order],
        "threads": prepared["threads"],
        "session_options": registered_options(int(prepared["threads"])),
        "onnxruntime_version": ORT_VERSION,
        "pages": first["runs"][0]["pages"]["count"] if first["measured"] else None,
        "check_rule": CHECK_RULE,
        "logit_tolerance": LOGIT_TOL,
        "prepare_sha256": push["prepare_sha256"],
        "apk_sha256": push["apk_sha256"],
        "build_commit": push["git_commit"],
        "run_commits": commits,
        "phone_hash_check": push["phone_hash_check"],
        "arms": entries,
        "fused_failed_phone_check": fused_failed,
        "part_b": part_b,
        "generated_utc": _utc(),
        "git_commit": _git("rev-parse", "HEAD"),
    }
    write_json_atomic(out, doc)
    write_text_atomic(out.with_suffix(".md"), _phone_md(doc))
    means = {a: (e["pages_both_runs"] or {}).get("mean_ms") for a, e in entries.items()}
    checks = {a: e["check_passed"] for a, e in entries.items()}
    print(
        json.dumps(
            {
                "device": slug,
                "role": role,
                "part_b": doc["part_b"],
                "mean_ms": means,
                "check_passed": checks,
                "report": out.as_posix(),
            }
        )
    )
    return 0


def _selection_md(doc: Mapping[str, Any]) -> str:
    qualified = ", ".join(doc.get("qualified") or []) or "none"
    rows = [
        f"# {doc['experiment']}: the phone candidate's selection",
        "",
        "Generated by `scripts/phase22_android.py select`; do not edit by hand. Rule: "
        "research/hypotheses.md, H11, selection rule (`candidate.select`).",
        "",
        f"Selected: **{doc['selected']}**; {doc['reason']}.",
        "",
        f"To the phone: {', '.join(doc['to_phone'])}. Qualified: {qualified}. First phone: "
        f"{doc['device']} ({doc['phone_report']['file']}).",
    ]
    if doc.get("first_phone_reason"):
        rows += ["", f"This phone took the M52's place: {doc['first_phone_reason']}."]
    return "\n".join(rows) + "\n"


def cmd_select(args: argparse.Namespace) -> int:
    from glypharis.efficiency.candidate import EXPERIMENT as SELECTION_EXPERIMENT
    from glypharis.efficiency.candidate import VERSION_LABEL, select

    report_path = PHONE_REPORTS / f"phone_{args.device}.json"
    if not report_path.is_file():
        raise SystemExit(f"{report_path.as_posix()} is missing: run phone-report first")
    _refuse_existing(SELECTION, SELECTION.with_suffix(".md"))
    report = _read_json(report_path)
    if report.get("role") != "first":
        raise SystemExit(f"{args.device} is not the first phone: its runs decide nothing")
    prepared = load_timing()
    if sha256_file(ACCURACY) != prepared["accuracy"]["sha256"]:
        raise SystemExit(f"{ACCURACY.as_posix()} changed after prepare-timing read it")
    accuracy = _read_json(ACCURACY)
    phone = {
        arm: {key: entry.get(key) for key in ("loaded", "check_passed", "speed", "memory")}
        for arm, entry in report["arms"].items()
    }
    result = select(accuracy["arms"], phone, bool(report["fused_failed_phone_check"]))
    doc = {
        "note": "Generated by scripts/phase22_android.py select; do not edit by hand.",
        "experiment": SELECTION_EXPERIMENT,
        "version": VERSION_LABEL,
        "rule": "research/hypotheses.md, H11, selection rule",
        "device": args.device,
        "first_phone_reason": report.get("first_phone_reason"),
        "phone_report": {"file": report_path.as_posix(), "sha256": sha256_file(report_path)},
        "accuracy": {"file": ACCURACY.as_posix(), "sha256": sha256_file(ACCURACY)},
        **result,
        "phone": phone,
        "date": datetime.now(UTC).date().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
    }
    write_json_atomic(SELECTION, doc)
    write_text_atomic(SELECTION.with_suffix(".md"), _selection_md(doc))
    print(json.dumps({"selected": doc["selected"], "reason": doc["reason"]}))
    return 0


# ------------------------------------------------------------------------------------ cleanup


def cmd_cleanup(args: argparse.Namespace) -> int:
    phone = select_phone(args.serial)
    info = read_device(phone)
    names = (*PHONE_FILES, *(f"{n}.tmp" for n in PHONE_FILES))
    phone.shell("rm -f " + " ".join(shlex.quote(phone_path(n)) for n in names))
    # The timing files go by pattern; the app's folder holds nothing else of these names.
    phone.shell("rm -f " + " ".join(f"{shlex.quote(PHONE_DIR)}/{p}" for p in TIMING_PHONE_FILES))
    removed = phone.call(["uninstall", APP_ID], timeout=120.0)
    installed = phone.shell(f"pm list packages {APP_ID}").strip()
    phone.shell(f"rm -rf {shlex.quote(f'/sdcard/Android/data/{APP_ID}')}")
    gone = not phone.test(f"test -e {shlex.quote(PHONE_DIR)}")
    if APP_ID in installed or not gone:
        raise SystemExit("the app or its folder is still on the phone")
    doc = {
        "cleaned_utc": _utc(),
        "app_was_installed": removed.returncode == 0,
        "phone_files_removed": True,
        "timing_files_removed": True,
        "app_uninstalled": True,
    }
    write_json_atomic(RUNS / device_slug(info) / "cleanup.json", doc)
    print(json.dumps(doc))
    return 0


# --------------------------------------------------------------------------------------- main


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="write the input tensors and the laptop's logits")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--bundle", type=Path, default=BUNDLE)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=PREPARED)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("build", help="build the app from a clean tree and record the result")
    p.set_defaults(func=cmd_build)

    for name, func, about in (
        ("device", cmd_device, "read the phone's model, SoC, RAM, Android build and ABI"),
        ("run", cmd_run, "install, push, run the app and pull its results"),
        ("cleanup", cmd_cleanup, "remove the pushed files and uninstall the app"),
        ("conditions", cmd_conditions, "record the phone's battery, thermal and screen state"),
        ("push-timing", cmd_push_timing, "install the app and push the models, tensors, plans"),
        ("time-run", cmd_time_run, "the next timing run of the phone's registered order"),
        ("check-run", cmd_check_run, "one arm's logit and text check after the timing runs"),
    ):
        p = sub.add_parser(name, help=about)
        p.add_argument("--serial", default=None, help="only when several phones are connected")
        p.set_defaults(func=func)
        if name == "run":
            p.add_argument("--apk", type=Path, default=APK)
            p.add_argument("--model", type=Path, default=MODEL)
            p.add_argument("--wait-seconds", type=float, default=900.0)
            p.add_argument(
                "--allow-dirty", action="store_true", help="rehearsal: an unclean build gives no-go"
            )
        if name == "conditions":
            p.add_argument("--label", required=True, help="names the file conditions_<label>.json")
        if name in ("push-timing", "time-run", "check-run"):
            p.add_argument("--apk", type=Path, default=APK)
        if name == "push-timing":
            p.add_argument(
                "--first-phone-reason",
                default=None,
                help=f"why this phone takes the place of the first phone, {FIRST_PHONE}",
            )
        if name in ("time-run", "check-run"):
            p.add_argument("--arm", required=True)
            p.add_argument("--wait-seconds", type=float, default=TIMING_SECONDS)
        if name == "time-run":
            p.add_argument("--run", type=int, required=True, choices=[1, 2])

    p = sub.add_parser("report", help="compare with the laptop and write the report once")
    p.add_argument("--device", action="append", default=[], help="a run's slug (default: all)")
    p.add_argument("--unavailable", action="append", default=[], help="a phone that was not used")
    p.add_argument("--out", type=Path, default=REPORT)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("prepare-timing", help="record the arms' network inputs and write the plans")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument(
        "--limit-pages", type=int, default=None, help="smoke test: the first N pages, one warm-up"
    )
    p.add_argument("--limit-lines", type=int, default=None, help="smoke test: the first M lines")
    p.add_argument("--out", type=Path, default=None, help=f"default {TIMING.as_posix()}")
    p.set_defaults(func=cmd_prepare_timing)

    for name, func, about in (
        ("phone-report", cmd_phone_report, "a phone's times, peaks, check and conditions"),
        ("select", cmd_select, "the arm that leaves item 14, from accuracy and the first phone"),
    ):
        p = sub.add_parser(name, help=about)
        p.add_argument("--device", required=True, help="the phone's slug under the runs folder")
        p.set_defaults(func=func)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
