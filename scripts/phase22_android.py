"""Android operator check on a phone, GW-EFF-015 part a (Phase 22, item 3).

    python scripts/phase22_android.py prepare [--threads 4]
    python scripts/phase22_android.py device [--serial S]
    python scripts/phase22_android.py build
    python scripts/phase22_android.py run [--serial S] [--apk APK] [--wait-seconds 900]
    python scripts/phase22_android.py report [--device SLUG ...] [--unavailable TEXT ...]
    python scripts/phase22_android.py cleanup [--serial S]

Does the stock ``onnxruntime-android`` package hold the ``com.microsoft.MultiHeadAttention`` kernel
the fused recognizer (GW-EFF-009) needs, and does the phone give the laptop's output for one
validation line? Run from the repository root. Validation data only; nothing here reads a test set.

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
- ``cleanup`` deletes the pushed files and uninstalls the app.

With no phone or with several, every command that needs one refuses; ``--serial`` picks one for
that command only and is never written or printed. All adb calls go through `run_adb`. The app and
its tensor file format (GWT1) are described in ``mobile/android-bench/README.md``. No time is
measured: phone timing is part b.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

from glypharis.data.download import sha256_file
from glypharis.data.text import Charset
from glypharis.efficiency.files import write_json_atomic

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
GWT_TYPES: Final[dict[int, np.dtype[Any]]] = {1: np.dtype("<f4"), 2: np.dtype("<i8")}
GWT_CODES: Final = {"f": 1, "i": 2}


# ------------------------------------------------------------------------------ tensor files


def write_gwt(path: Path, tensors: Mapping[str, np.ndarray]) -> None:
    """`tensors` as a GWT1 file (``mobile/android-bench/README.md``), moved into place whole."""
    parts = [GWT_MAGIC, len(tensors).to_bytes(4, "little")]
    for name, array in tensors.items():
        code = GWT_CODES[array.dtype.kind]
        raw = array.astype(GWT_TYPES[code]).tobytes(order="C")
        encoded = name.encode("utf-8")
        parts += [len(encoded).to_bytes(4, "little"), encoded, code.to_bytes(4, "little")]
        parts.append(array.ndim.to_bytes(4, "little"))
        parts += [int(d).to_bytes(8, "little", signed=True) for d in array.shape]
        parts += [len(raw).to_bytes(8, "little"), raw]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(b"".join(parts))
    os.replace(tmp, path)


def read_gwt(path: Path) -> dict[str, np.ndarray]:
    """The tensors of a GWT1 file; `ValueError` when it is not a complete one."""
    data = path.read_bytes()
    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        if n < 0 or pos + n > len(data):
            raise ValueError(f"{path.name} is cut short")
        chunk = data[pos : pos + n]
        pos += n
        return chunk

    def number(n: int, signed: bool = False) -> int:
        return int.from_bytes(take(n), "little", signed=signed)

    if take(4) != GWT_MAGIC:
        raise ValueError(f"{path.name} is not a GWT1 file")
    out: dict[str, np.ndarray] = {}
    for _ in range(number(4)):
        name = take(number(4)).decode("utf-8")
        code = number(4)
        if code not in GWT_TYPES:
            raise ValueError(f"{path.name}: unknown element type {code}")
        shape = [number(8, signed=True) for _ in range(number(4))]
        raw = take(number(8))
        dtype = GWT_TYPES[code]
        if any(d < 0 for d in shape) or len(raw) != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError(f"{path.name}: {name} does not fit its shape {shape}")
        out[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    if pos != len(data):
        raise ValueError(f"{path.name} has trailing bytes")
    return out


# ------------------------------------------------------------------------------------- common


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return doc


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

    phone.adb(["install", "-r", str(args.apk)], timeout=PUSH_SECONDS)
    start_app(phone, {"mode": "ready"})
    wait_for(phone, PHONE_READY, READY_SECONDS)
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


# ------------------------------------------------------------------------------------ cleanup


def cmd_cleanup(args: argparse.Namespace) -> int:
    phone = select_phone(args.serial)
    info = read_device(phone)
    names = (*PHONE_FILES, *(f"{n}.tmp" for n in PHONE_FILES))
    phone.shell("rm -f " + " ".join(shlex.quote(phone_path(n)) for n in names))
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

    p = sub.add_parser("report", help="compare with the laptop and write the report once")
    p.add_argument("--device", action="append", default=[], help="a run's slug (default: all)")
    p.add_argument("--unavailable", action="append", default=[], help="a phone that was not used")
    p.add_argument("--out", type=Path, default=REPORT)
    p.set_defaults(func=cmd_report)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
