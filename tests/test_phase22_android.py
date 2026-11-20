"""The Android operator check (GW-EFF-015 part a): tensor files, device choice, the run on a fake
phone and the go/no-go report; and part b: the recorded network inputs, the timing and check
plans, the runs in their registered order, the phone report and the selection. No phone, adb,
model or real data is used."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import phase22_android as pa
import pytest
import torch
from PIL import Image

from glypharis.data.text import Charset
from glypharis.efficiency import candidate as cand
from glypharis.models.rec.base import LineRecognizer
from glypharis.pipeline.recognizer import TextRecognizer, _padded, prepare_crop

REPO = Path(__file__).resolve().parents[1]
SERIAL = "SERIAL0123456789"
OTHER_SERIAL = "OTHERSERIAL98765"
CHARACTERS = ["a", "b", "c"]
MODEL_BYTES = b"not a real model"
KERNEL_ERROR = (
    "Failed to find kernel for com.microsoft.MultiHeadAttention(1). Could not find an "
    "implementation for MultiHeadAttention(1) node with name 'mha'"
)
PROPS = {
    "ro.product.manufacturer": "samsung",
    "ro.product.model": "SM-TEST 5G",
    "ro.soc.manufacturer": "QTI",
    "ro.soc.model": "SM7325",
    "ro.board.platform": "lahaina",
    "ro.build.version.release": "14",
    "ro.build.version.sdk": "34",
    "ro.build.display_id": "TEST.BUILD.1",
    "ro.product.cpu.abi": "arm64-v8a",
    "ro.product.cpu.abilist": "arm64-v8a,armeabi-v7a",
}
SLUG = "sm-test-5g"
COMMIT = "c" * 40
SESSION_OPTIONS = {
    "providers": ["CPUExecutionProvider"],
    "intra_op_num_threads": 4,
    "execution_mode": "ORT_SEQUENTIAL",
    "graph_optimization_level": "ORT_ENABLE_ALL",
    "enable_cpu_mem_arena": False,
    "enable_mem_pattern": False,
    "allow_spinning": "default",
}


def laptop_logits() -> np.ndarray:
    logits = np.zeros((6, 1, 4), dtype=np.float32)
    for frame, cls in enumerate([1, 1, 0, 2, 0, 3]):
        logits[frame, 0, cls] = 5.0
    return logits


class FakePhone:
    """Stands in for `run_adb`: a device list, a file folder and the app's behaviour."""

    def __init__(self, devices: list[tuple[str, str]] | None = None) -> None:
        self.devices = [(SERIAL, "device")] if devices is None else devices
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str | None, list[str]]] = []
        self.installed = False
        self.uninstall_works = True
        self.has_sha256sum = True
        self.corrupt_push = False
        self.start_refused = False
        self.writes_result = True
        self.props = dict(PROPS)
        self.logits = laptop_logits()
        self.outcome: dict[str, Any] = {"status": "ok"}
        self.git = {"status": "", "rev-parse": COMMIT}
        self.apk_permissions = f"package: {pa.APP_ID}\n"
        self.apk_manifest = "E: manifest\n  E: application\n"
        self.gradle_calls: list[list[str]] = []

    def aapt2(self, args: list[str]) -> str:
        return {"permissions": self.apk_permissions, "xmltree": self.apk_manifest}[args[1]]

    def gradle(self, args: list[str]) -> int:
        self.gradle_calls.append(list(args))
        pa.APK.write_bytes(b"apk")
        return 0

    def __call__(
        self, args: list[str], serial: str | None = None, timeout: float = 60.0
    ) -> pa.AdbResult:
        self.calls.append((serial, list(args)))
        head = args[0]
        if head == "devices":
            rows = "".join(f"{s}\t{state}\n" for s, state in self.devices)
            return pa.AdbResult(0, "List of devices attached\n" + rows, "")
        if head == "install":
            self.installed = True
            return pa.AdbResult(0, "Success\n", "")
        if head == "uninstall":
            was = self.installed
            if self.uninstall_works:
                self.installed = False
                self.files.clear()
            return pa.AdbResult(0 if was else 1, "Success\n" if was else "", "")
        if head == "push":
            data = Path(args[1]).read_bytes()
            self.files[args[2]] = b"x" + data if self.corrupt_push else data
            return pa.AdbResult(0, "1 file pushed\n", "")
        if head == "pull":
            if args[1] not in self.files:
                return pa.AdbResult(1, "", f"remote object {args[1]} does not exist")
            Path(args[2]).write_bytes(self.files[args[1]])
            return pa.AdbResult(0, "1 file pulled\n", "")
        assert head == "shell", args
        return self.shell(args[1])

    def shell(self, command: str) -> pa.AdbResult:
        words = shlex.split(command)
        if words[0] == "getprop":
            assert "serial" not in words[1], "the serial number must never be read"
            return pa.AdbResult(0, self.props[words[1]] + "\n", "")
        if words[:2] == ["cat", "/proc/meminfo"]:
            return pa.AdbResult(0, "MemTotal:        5799999 kB\nMemFree: 1 kB\n", "")
        if words[:2] == ["am", "force-stop"]:
            return pa.AdbResult(0, "", "")
        if words[:2] == ["am", "start"]:
            return self.start(words)
        if words[0] == "test":
            found = words[2] in self.files or (
                words[1] == "-e" and any(k.startswith(words[2]) for k in self.files)
            )
            return pa.AdbResult(0 if found else 1, "", "")
        if words[0] == "rm":
            for name in words[2:]:
                for key in [k for k in self.files if k == name or k.startswith(name + "/")]:
                    del self.files[key]
            return pa.AdbResult(0, "", "")
        if words[:2] == ["command", "-v"]:
            return pa.AdbResult(0 if self.has_sha256sum else 1, "/system/bin/sha256sum\n", "")
        if words[0] == "sha256sum":
            digest = hashlib.sha256(self.files[words[1]]).hexdigest()
            return pa.AdbResult(0, f"{digest}  {words[1]}\n", "")
        if words[:3] == ["pm", "list", "packages"]:
            return pa.AdbResult(0, f"package:{pa.APP_ID}\n" if self.installed else "", "")
        raise AssertionError(f"unexpected shell command: {command}")

    def start(self, words: list[str]) -> pa.AdbResult:
        extras = {words[i + 1]: words[i + 2] for i, w in enumerate(words) if w in ("--es", "--ei")}
        if self.start_refused:
            refusal = "Exception occurred while executing 'start':\nSecurityException: denied\n"
            return pa.AdbResult(255, "", refusal)
        if extras["mode"] == "ready":
            self.files[pa.phone_path(pa.PHONE_READY)] = b"{}"
            return pa.AdbResult(0, "Status: ok\n", "")
        if not self.writes_result:  # the process died before it wrote anything
            return pa.AdbResult(0, "Status: ok\n", "")
        model = self.files[pa.phone_path(extras["model"])]
        tensor = self.files[pa.phone_path(extras["input"])]
        result = {
            "schema": "glypharis-android-bench-1",
            "mode": "operator_check",
            "session_created": True,
            "session_error": None,
            "run_error": None,
            "onnxruntime_version": pa.ORT_VERSION,
            "model_sha256": hashlib.sha256(model).hexdigest(),
            "input_sha256": hashlib.sha256(tensor).hexdigest(),
            "session_options": {**SESSION_OPTIONS, "intra_op_num_threads": int(extras["threads"])},
            "output_shape": [int(d) for d in self.logits.shape],
            "logits_finite": bool(np.isfinite(self.logits).all()),
            **self.outcome,
        }
        if result["status"] == "ok":
            local = Path("phone_logits.tmp")
            pa.write_gwt(local, {"logits": self.logits})
            self.files[pa.phone_path(extras["output"])] = local.read_bytes()
            local.unlink()
        self.files[pa.phone_path(extras["result"])] = json.dumps(result).encode()
        return pa.AdbResult(0, "Status: ok\n", "")


def record_build(**changes: Any) -> None:
    """The record `build` leaves for the current APK at `COMMIT`, with ``changes`` applied."""
    doc = {
        "git_commit": COMMIT,
        "tree_clean_after_build": True,
        "apk_sha256": pa.sha256_file(pa.APK),
        **changes,
    }
    pa.BUILD_RECORD.parent.mkdir(parents=True, exist_ok=True)
    pa.BUILD_RECORD.write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def phone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakePhone:
    """A prepared repository in ``tmp_path`` with a fake phone behind `run_adb`."""
    monkeypatch.chdir(tmp_path)
    fake = FakePhone()
    monkeypatch.setattr(pa, "run_adb", fake)
    monkeypatch.setattr(pa, "run_aapt2", fake.aapt2)
    monkeypatch.setattr(pa, "run_gradle", fake.gradle)
    monkeypatch.setattr(pa, "POLL_SECONDS", 0.0)
    monkeypatch.setenv("JAVA_HOME", "jdk")
    monkeypatch.setenv("ANDROID_HOME", "sdk")
    monkeypatch.setenv("GRADLE_USER_HOME", str(tmp_path.parent / "gradle-home-outside"))

    def no_subprocess(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"a process was started outside run_adb: {args}")

    monkeypatch.setattr(pa.subprocess, "run", no_subprocess)
    monkeypatch.setattr(pa, "_git", lambda *args: fake.git[args[0]])
    monkeypatch.setattr(
        pa,
        "build_tools",
        lambda: {
            "gradle": "1",
            "android_gradle_plugin": "2",
            "kotlin": "3",
            "jdk_major": 21,
            "compile_sdk": 35,
            "target_sdk": 35,
            "min_sdk": 26,
            "onnxruntime_android": pa.ORT_VERSION,
            "onnxruntime_android_aar_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(pa, "MODEL_SHA256", hashlib.sha256(MODEL_BYTES).hexdigest())
    monkeypatch.setattr(pa, "MODEL_BYTES", len(MODEL_BYTES))
    pa.MODEL.parent.mkdir(parents=True)
    pa.MODEL.write_bytes(MODEL_BYTES)
    pa.APK.parent.mkdir(parents=True)
    pa.APK.write_bytes(b"apk")
    record_build()

    logits = laptop_logits()
    images = np.zeros((1, 1, 48, 64), dtype=np.float32)
    pa.write_gwt(pa.PREPARED / pa.PHONE_INPUT, {"images": images, "widths": np.array([24])})
    pa.write_gwt(pa.PREPARED / "laptop_logits.gwt", {"logits": logits})
    doc = {
        "line": {"index": 0, "key": "doc/p0/l0", "split": "validation", "pack": "pack"},
        "model": {"file": pa.PHONE_MODEL, "sha256": pa.MODEL_SHA256, "bytes": len(MODEL_BYTES)},
        "threads": 4,
        "session_options": {"onnxruntime": pa.ORT_VERSION, **SESSION_OPTIONS},
        "onnxruntime": pa.ORT_VERSION,
        "input": {
            "file": pa.PHONE_INPUT,
            "sha256": pa.sha256_file(pa.PREPARED / pa.PHONE_INPUT),
            "images_shape": [1, 1, 48, 64],
            "widths": [24],
        },
        "logits": {
            "file": "laptop_logits.gwt",
            "sha256": pa.sha256_file(pa.PREPARED / "laptop_logits.gwt"),
            "shape": [6, 1, 4],
            "frames": 6,
        },
        "text": Charset(CHARACTERS).ctc_greedy_decode([1, 1, 0, 2, 0, 3]),
        "characters": CHARACTERS,
    }
    (pa.PREPARED / "prepared.json").write_text(json.dumps(doc), encoding="utf-8")
    return fake


def run_and_report(*run_args: str) -> dict[str, Any]:
    assert pa.main(["run", "--wait-seconds", "5", *run_args]) == 0
    assert pa.main(["report"]) == 0
    return json.loads(pa.REPORT.read_text(encoding="utf-8"))


def set_laptop(logits: np.ndarray, frames: int | None = None) -> None:
    """Replace the prepared laptop logits; the line keeps ``frames`` of them (default all)."""
    frames = logits.shape[0] if frames is None else frames
    path = pa.PREPARED / "laptop_logits.gwt"
    pa.write_gwt(path, {"logits": logits})
    doc = json.loads((pa.PREPARED / "prepared.json").read_text(encoding="utf-8"))
    doc["logits"].update(sha256=pa.sha256_file(path), shape=list(logits.shape), frames=frames)
    doc["text"] = Charset(CHARACTERS).ctc_greedy_decode(logits[:frames, 0].argmax(-1).tolist())
    (pa.PREPARED / "prepared.json").write_text(json.dumps(doc), encoding="utf-8")


def edit_json(path: Path, **changes: Any) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc.update(changes)
    path.write_text(json.dumps(doc), encoding="utf-8")


def written_text(root: Path) -> str:
    """Everything the commands wrote, to search for values that must not be there."""
    out = [
        p.read_text(encoding="utf-8")
        for p in root.rglob("*")
        if p.is_file() and p.suffix in (".json", ".txt", ".md")
    ]
    return "\n".join(out)


# ------------------------------------------------------------------------------ tensor files


def test_gwt_round_trip_and_layout(tmp_path: Path) -> None:
    path = tmp_path / "t.gwt"
    images = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    widths = np.array([7], dtype=np.int64)
    pa.write_gwt(path, {"images": images, "widths": widths})
    got = pa.read_gwt(path)
    assert list(got) == ["images", "widths"]
    assert got["images"].dtype == np.float32 and got["widths"].dtype == np.int64
    assert np.array_equal(got["images"], images) and np.array_equal(got["widths"], widths)
    raw = path.read_bytes()
    assert raw[:8] == b"GWT1" + (2).to_bytes(4, "little")
    assert raw[8:12] == (6).to_bytes(4, "little") and raw[12:18] == b"images"


@pytest.mark.parametrize("cut", [3, 12, -1])
def test_gwt_refuses_damaged_files(tmp_path: Path, cut: int) -> None:
    path = tmp_path / "t.gwt"
    pa.write_gwt(path, {"x": np.zeros((2, 2), dtype=np.float32)})
    path.write_bytes(path.read_bytes()[:cut])
    with pytest.raises(ValueError):
        pa.read_gwt(path)


def test_gwt_refuses_trailing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "t.gwt"
    pa.write_gwt(path, {"x": np.zeros(2, dtype=np.float32)})
    path.write_bytes(path.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="trailing"):
        pa.read_gwt(path)


# ------------------------------------------------------------------------------------ prepare


class StubLineModel(LineRecognizer):
    stride = 4
    num_classes = 4

    def forward(self, x: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        frames = x.shape[3] // self.stride
        out = torch.zeros(frames, x.shape[0], self.num_classes)
        out[:, :, 1] = x.mean()
        return out


def test_capture_line_records_the_recognizer_input() -> None:
    crop = Image.fromarray(np.tile(np.arange(40, dtype=np.uint8) * 5, (16, 1)), mode="L")
    recognizer = TextRecognizer(StubLineModel(), Charset(CHARACTERS), 32, 200)
    model = recognizer.model
    got = pa.capture_line(recognizer, crop)
    assert recognizer.model is model
    line = prepare_crop(crop, 32, 200)
    width = line.shape[1]
    assert got["widths"].tolist() == [width]
    assert got["images"].shape == (1, 1, 32, _padded(width))
    assert np.array_equal(got["images"][0, 0, :, :width], line.numpy())
    assert np.all(got["images"][0, 0, :, width:] == float(line[:, -1].median()))
    assert got["logits"].shape[:2] == (_padded(width) // 4, 1)
    assert got["line_logits"].shape == (width // 4, 4)


# ------------------------------------------------------------------------------- device choice


def test_no_device_is_refused(phone: FakePhone) -> None:
    phone.devices = []
    with pytest.raises(SystemExit, match="no authorized device"):
        pa.main(["device"])


def test_unauthorized_device_is_refused_without_its_serial(
    phone: FakePhone, capsys: pytest.CaptureFixture[str]
) -> None:
    phone.devices = [(SERIAL, "unauthorized")]
    with pytest.raises(SystemExit) as caught:
        pa.main(["device"])
    assert "unauthorized" in str(caught.value) and SERIAL not in str(caught.value)
    assert SERIAL not in capsys.readouterr().out


def test_two_devices_need_a_serial(phone: FakePhone) -> None:
    phone.devices = [(SERIAL, "device"), (OTHER_SERIAL, "device")]
    with pytest.raises(SystemExit, match="--serial") as caught:
        pa.main(["run"])
    assert SERIAL not in str(caught.value) and OTHER_SERIAL not in str(caught.value)
    assert not phone.installed


def test_unknown_serial_is_refused(phone: FakePhone) -> None:
    with pytest.raises(SystemExit, match="not listed") as caught:
        pa.main(["device", "--serial", OTHER_SERIAL])
    assert OTHER_SERIAL not in str(caught.value)


def test_device_reads_named_properties_only(
    phone: FakePhone, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert pa.main(["device"]) == 0
    info = json.loads((pa.RUNS / SLUG / "device.json").read_text(encoding="utf-8"))
    assert info["model"] == "SM-TEST 5G" and info["soc_model"] == "SM7325"
    assert info["ram_total_kb"] == 5799999 and info["abi"] == "arm64-v8a"
    assert info["sdk"] == "34" and info["android_release"] == "14"
    assert SERIAL not in capsys.readouterr().out
    assert SERIAL not in written_text(tmp_path)


def test_adb_failure_message_hides_the_serial(phone: FakePhone) -> None:
    chosen = pa.Phone(SERIAL, (SERIAL,))
    with pytest.raises(SystemExit) as caught:
        chosen.adb(["pull", f"/sdcard/{SERIAL}", "out"])
    assert SERIAL not in str(caught.value) and "<device>" in str(caught.value)


# ----------------------------------------------------------------------------------- go path


def test_go_path(phone: FakePhone, capsys: pytest.CaptureFixture[str]) -> None:
    report = run_and_report()
    assert report["verdict"] == "go" and report["no_go_reasons"] == []
    assert report["experiment"] == "GW-EFF-015"
    assert {k: v for k, v in report["package"].items() if k != "merged_manifest"} == {
        "name": pa.ORT_PACKAGE,
        "version": pa.ORT_VERSION,
        "aar_sha256": "a" * 64,
    }
    assert report["model"]["sha256"] == pa.MODEL_SHA256
    assert report["tell_owner_before_s4"] is False
    device = report["devices"][0]
    assert device["slug"] == SLUG and device["verdict"] == "go"
    assert device["device"]["model"] == "SM-TEST 5G"
    assert device["apk_sha256"] == hashlib.sha256(b"apk").hexdigest()
    assert device["output_shape"] == [6, 1, 4] and device["shape_equal"] is True
    assert device["logits_finite"] is True and device["logit_max_abs_diff"] == 0.0
    assert device["greedy_text_equal"] is True
    assert device["phone_hash_check"] == "sha256sum"
    assert device["tree_clean_after_build"] is True and device["apk_permissions"] == []
    assert device["session_options"]["execution_mode"] == "ORT_SEQUENTIAL"
    assert phone.installed
    # The run started the app in a fresh process for the folder and again for the check.
    stops = [c for _, c in phone.calls if c[0] == "shell" and "force-stop" in c[1]]
    assert len(stops) == 2
    assert not any("elapsed" in k or k.endswith(("_ms", "_seconds")) for k in device)


def test_serial_is_not_recorded_anywhere(
    phone: FakePhone, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    phone.devices = [(SERIAL, "device"), (OTHER_SERIAL, "device")]
    assert pa.main(["run", "--serial", SERIAL, "--wait-seconds", "5"]) == 0
    assert pa.main(["report"]) == 0
    assert pa.main(["cleanup", "--serial", SERIAL]) == 0
    out = capsys.readouterr()
    text = written_text(tmp_path)
    assert all(serial == SERIAL for serial, c in phone.calls if c != ["devices"])
    for value in (SERIAL, OTHER_SERIAL):
        assert value not in text and value not in out.out and value not in out.err


def test_serial_inside_a_device_property_is_cut_out(phone: FakePhone, tmp_path: Path) -> None:
    phone.props["ro.build.display_id"] = f"BUILD-{SERIAL}"
    assert pa.main(["device"]) == 0
    assert SERIAL not in written_text(tmp_path)
    info = json.loads((pa.RUNS / SLUG / "device.json").read_text(encoding="utf-8"))
    assert info["build"] == "BUILD-<device>"


def test_logit_difference_is_recorded_and_flagged_but_does_not_decide(phone: FakePhone) -> None:
    phone.logits = laptop_logits() + 0.01
    report = run_and_report()
    device = report["devices"][0]
    assert report["verdict"] == "go"
    assert device["logit_max_abs_diff"] == pytest.approx(0.01, abs=1e-5)
    assert device["logit_max_abs_diff_with_padding"] == pytest.approx(0.01, abs=1e-5)
    assert device["greedy_text_equal"] is True
    assert report["tell_owner_before_s4"] is True


def test_different_text_is_flagged(phone: FakePhone) -> None:
    logits = laptop_logits()
    logits[5, 0, 3] = 0.0
    logits[5, 0, 1] = 5.0
    phone.logits = logits
    report = run_and_report()
    assert report["verdict"] == "go"
    assert report["devices"][0]["greedy_text_equal"] is False
    assert report["tell_owner_before_s4"] is True


def test_text_difference_is_flagged_when_the_logits_are_close(phone: FakePhone) -> None:
    laptop = laptop_logits()
    laptop[3, 0, 1] = 4.9995
    set_laptop(laptop)
    close = laptop.copy()
    close[3, 0, 1], close[3, 0, 2] = 5.0, 4.9995
    phone.logits = close
    device = run_and_report()["devices"][0]
    assert device["logit_max_abs_diff"] < pa.LOGIT_NOTICE
    assert device["greedy_text_equal"] is False and device["tell_owner_before_s4"] is True


def test_only_the_lines_own_frames_are_compared(phone: FakePhone) -> None:
    laptop = laptop_logits()
    set_laptop(laptop, frames=4)
    tail = laptop.copy()
    tail[4:] += 1.0
    tail[5, 0, 1] = 99.0  # the padded frames change their top class too
    phone.logits = tail
    report = run_and_report()
    device = report["devices"][0]
    assert report["verdict"] == "go" and device["logit_max_abs_diff"] == 0.0
    assert device["logit_max_abs_diff_with_padding"] == pytest.approx(99.0)
    assert device["greedy_text_equal"] is True and report["tell_owner_before_s4"] is False


def test_phone_without_sha256sum_still_runs(phone: FakePhone) -> None:
    phone.has_sha256sum = False
    report = run_and_report()
    assert report["verdict"] == "go"
    assert report["devices"][0]["phone_hash_check"] == "unavailable"
    assert report["devices"][0]["app_model_sha256"] == pa.MODEL_SHA256


# -------------------------------------------------------------------------------------- no-go


def test_missing_kernel_is_no_go(phone: FakePhone) -> None:
    phone.outcome = {
        "status": "session_failed",
        "session_created": False,
        "session_error": {
            "type": "ai.onnxruntime.OrtException",
            "code": "ORT_FAIL",
            "message": KERNEL_ERROR,
        },
        "output_shape": None,
        "logits_finite": None,
    }
    report = run_and_report()
    device = report["devices"][0]
    assert report["verdict"] == "no-go" and report["no_go_reasons"] == ["missing_kernel"]
    assert device["session_created"] is False
    assert device["session_error"]["message"] == KERNEL_ERROR
    assert device["logit_max_abs_diff"] is None


def test_other_session_error_is_no_go_but_not_a_missing_kernel(phone: FakePhone) -> None:
    phone.outcome = {
        "status": "session_failed",
        "session_created": False,
        "session_error": {"type": "OrtException", "code": "ORT_FAIL", "message": "out of memory"},
    }
    report = run_and_report()
    assert report["verdict"] == "no-go" and report["no_go_reasons"] == ["session_failed"]


def test_runtime_that_does_not_load_is_no_go(phone: FakePhone) -> None:
    phone.outcome = {
        "status": "runtime_failed",
        "session_created": False,
        "onnxruntime_version": None,
        "session_error": {"type": "UnsatisfiedLinkError", "message": "no library"},
    }
    report = run_and_report()
    assert report["verdict"] == "no-go" and report["no_go_reasons"] == ["session_failed"]


def test_not_implemented_code_counts_as_a_missing_kernel(phone: FakePhone) -> None:
    phone.outcome = {
        "status": "session_failed",
        "session_created": False,
        "session_error": {"code": "ORT_NOT_IMPLEMENTED", "message": "something else"},
    }
    assert run_and_report()["no_go_reasons"] == ["missing_kernel"]


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ({"model_sha256": "0" * 64}, "wrong_model_file"),
        ({"input_sha256": "0" * 64}, "wrong_input_file"),
        ({"onnxruntime_version": "0"}, "wrong_onnxruntime_version"),
        ({"onnxruntime_version": None}, "wrong_onnxruntime_version"),
        ({"status": "weird"}, "status_weird"),
        ({"output_shape": [1, 2, 3]}, "wrong_shape"),
        ({"logits_finite": False}, "non_finite_logits"),
        ({"session_created": False}, "session_not_created"),
        (
            {"session_options": {**SESSION_OPTIONS, "enable_mem_pattern": True}},
            "session_options_differ",
        ),
        ({"session_options": None}, "session_options_differ"),
    ],
)
def test_what_the_app_reports_can_give_no_go(
    phone: FakePhone, outcome: dict[str, Any], reason: str
) -> None:
    phone.outcome = {"status": "ok", **outcome}
    report = run_and_report()
    assert report["verdict"] == "no-go" and report["no_go_reasons"] == [reason]


def test_deleted_logits_are_no_go(phone: FakePhone) -> None:
    assert pa.main(["run", "--wait-seconds", "5"]) == 0
    (pa.RUNS / SLUG / pa.PHONE_LOGITS).unlink()
    assert pa.main(["report"]) == 0
    report = json.loads(pa.REPORT.read_text(encoding="utf-8"))
    assert report["verdict"] == "no-go" and report["no_go_reasons"] == ["logits_missing"]


def test_an_apk_recorded_with_permissions_is_no_go(phone: FakePhone) -> None:
    assert pa.main(["run", "--wait-seconds", "5"]) == 0
    edit_json(pa.RUNS / SLUG / "run.json", apk_permissions=["android.permission.INTERNET"])
    assert pa.main(["report"]) == 0
    report = json.loads(pa.REPORT.read_text(encoding="utf-8"))
    assert report["no_go_reasons"] == ["apk_has_permissions"]


def test_non_finite_laptop_logits_are_no_go_and_flagged(phone: FakePhone) -> None:
    laptop = laptop_logits()
    laptop[2, 0, 1] = np.nan
    set_laptop(laptop)
    report = run_and_report()
    device = report["devices"][0]
    assert report["no_go_reasons"] == ["logit_difference_not_finite"]
    assert device["logit_max_abs_diff"] is None and device["tell_owner_before_s4"] is True


def test_prepare_refuses_non_finite_laptop_logits() -> None:
    logits = laptop_logits()
    logits[5, 0, 1] = np.inf
    with pytest.raises(SystemExit, match="not all finite"):
        pa.check_laptop_logits(logits, logits[:4, 0])


def test_device_paths_stay_out_of_the_report(phone: FakePhone) -> None:
    folder = f"/storage/emulated/0/Android/data/{pa.APP_ID}/files"
    phone.outcome = {
        "status": "session_failed",
        "session_created": False,
        "session_error": {
            "code": "ORT_FAIL",
            "message": f"Load model from {folder}/m.onnx failed: /sdcard/x bad at core/data/y.cc",
        },
    }
    report = run_and_report()
    text = pa.REPORT.read_text(encoding="utf-8")
    assert "/storage/" not in text and "/sdcard/" not in text
    message = report["devices"][0]["session_error"]["message"]
    assert message == "Load model from <app files> failed: <app files> bad at core/data/y.cc"


def test_wrong_shape_is_no_go(phone: FakePhone) -> None:
    phone.logits = np.zeros((5, 1, 4), dtype=np.float32)
    report = run_and_report()
    assert report["verdict"] == "no-go" and report["no_go_reasons"] == ["wrong_shape"]
    assert report["devices"][0]["shape_equal"] is False
    assert report["devices"][0]["logit_max_abs_diff"] is None


def test_non_finite_logits_are_no_go(phone: FakePhone) -> None:
    logits = laptop_logits()
    logits[2, 0, 1] = np.nan
    phone.logits = logits
    report = run_and_report()
    assert report["verdict"] == "no-go" and report["no_go_reasons"] == ["non_finite_logits"]
    assert report["devices"][0]["logits_finite"] is False


def test_run_failure_is_no_go(phone: FakePhone) -> None:
    phone.outcome = {"status": "run_failed", "run_error": {"message": "boom"}}
    report = run_and_report()
    assert report["verdict"] == "no-go" and report["no_go_reasons"] == ["run_failed"]


def test_rehearsal_on_a_dirty_tree_is_no_go(phone: FakePhone) -> None:
    phone.git["status"] = "?? new.txt"
    report = run_and_report("--allow-dirty")
    assert report["verdict"] == "no-go" and report["no_go_reasons"] == ["build_not_clean"]
    assert report["devices"][0]["tree_clean_after_build"] is True
    assert report["devices"][0]["tree_clean_at_run"] is False


def test_rehearsal_without_a_build_record_is_no_go(phone: FakePhone) -> None:
    pa.BUILD_RECORD.unlink()
    report = run_and_report("--allow-dirty")
    assert report["no_go_reasons"] == ["build_not_clean"]
    assert report["devices"][0]["tree_clean_after_build"] is False


def test_every_phone_must_pass(phone: FakePhone) -> None:
    assert pa.main(["run", "--wait-seconds", "5"]) == 0
    second = pa.RUNS / "second-phone"
    shutil.copytree(pa.RUNS / SLUG, second)
    result = json.loads((second / pa.PHONE_RESULT).read_text(encoding="utf-8"))
    result.update(status="session_failed", session_created=False, session_error={"message": "x"})
    (second / pa.PHONE_RESULT).write_text(json.dumps(result), encoding="utf-8")
    assert pa.main(["report", "--unavailable", "Third phone: not connected"]) == 0
    report = json.loads(pa.REPORT.read_text(encoding="utf-8"))
    assert report["verdict"] == "no-go"
    assert [d["verdict"] for d in report["devices"]] == ["no-go", "go"]
    assert report["unavailable"] == ["Third phone: not connected"]


# ------------------------------------------------------------------- phones left out of a report


def test_phone_without_a_finished_run_cannot_vanish(phone: FakePhone) -> None:
    phone.writes_result = False
    with pytest.raises(SystemExit, match=r"no result\.json"):
        pa.main(["run", "--wait-seconds", "0"])
    crashed = pa.RUNS / SLUG
    second = pa.RUNS / "second-phone"
    phone.writes_result = True
    shutil.move(crashed, second)
    assert pa.main(["run", "--wait-seconds", "5"]) == 0
    with pytest.raises(SystemExit, match="seen but not in this report") as caught:
        pa.main(["report"])
    assert "second-phone" in str(caught.value) and not pa.REPORT.exists()
    assert pa.main(["report", "--unavailable", "second-phone: the app died in the session"]) == 0
    report = json.loads(pa.REPORT.read_text(encoding="utf-8"))
    assert report["verdict"] == "go" and [d["slug"] for d in report["devices"]] == [SLUG]
    assert report["unavailable"] == ["second-phone: the app died in the session"]


@pytest.mark.parametrize("named", [False, True])
def test_a_missing_kernel_cannot_be_left_out(phone: FakePhone, named: bool) -> None:
    assert pa.main(["run", "--wait-seconds", "5"]) == 0
    second = pa.RUNS / "second-phone"
    shutil.copytree(pa.RUNS / SLUG, second)
    edit_json(
        second / pa.PHONE_RESULT,
        status="session_failed",
        session_created=False,
        session_error={"code": "ORT_FAIL", "message": KERNEL_ERROR},
    )
    extra = ["--unavailable", "second-phone: not used"] if named else []
    with pytest.raises(SystemExit, match="missing kernel"):
        pa.main(["report", "--device", SLUG, *extra])
    assert not pa.REPORT.exists()


# -------------------------------------------------------------------------------------- build


def test_build_records_the_commit_the_clean_tree_and_the_apk(phone: FakePhone) -> None:
    pa.BUILD_RECORD.unlink()
    pa.APK.unlink()
    assert pa.main(["build"]) == 0
    assert phone.gradle_calls == [["--no-daemon", "clean", ":app:assembleDebug"]]
    doc = json.loads(pa.BUILD_RECORD.read_text(encoding="utf-8"))
    assert doc["git_commit"] == COMMIT and doc["tree_clean_after_build"] is True
    assert doc["apk_sha256"] == hashlib.sha256(b"apk").hexdigest()
    assert run_and_report()["verdict"] == "go"


def test_build_refuses_a_dirty_tree(phone: FakePhone) -> None:
    phone.git["status"] = " M scripts/x.py"
    with pytest.raises(SystemExit, match="not clean"):
        pa.main(["build"])
    assert phone.gradle_calls == []


@pytest.mark.parametrize("name", pa.BUILD_ENV)
def test_build_needs_its_environment(
    phone: FakePhone, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.delenv(name)
    with pytest.raises(SystemExit, match=name):
        pa.main(["build"])
    assert phone.gradle_calls == []


def test_build_keeps_the_gradle_cache_out_of_the_repository(
    phone: FakePhone, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GRADLE_USER_HOME", str(tmp_path / "cache"))
    with pytest.raises(SystemExit, match="outside the repository"):
        pa.main(["build"])
    assert phone.gradle_calls == []


def test_failed_build_leaves_no_record(phone: FakePhone, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pa, "run_gradle", lambda args: 1)
    with pytest.raises(SystemExit, match="build failed"):
        pa.main(["build"])
    assert not pa.BUILD_RECORD.exists()
    with pytest.raises(SystemExit, match="run `build`"):
        pa.main(["run"])


def test_files_left_by_the_build_are_recorded_and_stop_the_run(
    phone: FakePhone, monkeypatch: pytest.MonkeyPatch
) -> None:
    def untidy(args: list[str]) -> int:
        phone.git["status"] = "?? mobile/android-bench/stray"
        return 0

    monkeypatch.setattr(pa, "run_gradle", untidy)
    with pytest.raises(SystemExit, match="straight after the build"):
        pa.main(["build"])
    doc = json.loads(pa.BUILD_RECORD.read_text(encoding="utf-8"))
    assert doc["tree_clean_after_build"] is False
    phone.git["status"] = ""  # removing the file afterwards does not repair the build
    with pytest.raises(SystemExit, match="straight after the build"):
        pa.main(["run"])
    assert phone.calls == []


@pytest.mark.parametrize(
    "change",
    [
        lambda fake: fake.git.update({"rev-parse": "d" * 40}),
        lambda fake: pa.APK.write_bytes(b"built from something else"),
        lambda fake: pa.BUILD_RECORD.unlink(),
    ],
    ids=["later-commit", "other-apk", "no-record"],
)
def test_run_needs_the_record_of_this_commit_and_apk(
    phone: FakePhone, change: Callable[[FakePhone], Any]
) -> None:
    change(phone)
    with pytest.raises(SystemExit, match="run `build`"):
        pa.main(["run"])
    assert phone.calls == []


def test_a_failing_git_status_is_not_a_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 128, "", "fatal: not a git repository")

    monkeypatch.setattr(pa.subprocess, "run", failed)
    with pytest.raises(SystemExit, match="git status failed"):
        pa.tree_changes()


# ---------------------------------------------------------------------- the app's own settings


def test_app_manifest_removes_network_and_telemetry_and_guards_the_activity() -> None:
    tools = "{http://schemas.android.com/tools}"
    android = "{http://schemas.android.com/apk/res/android}"
    manifest = REPO / pa.APP_DIR / "app/src/main/AndroidManifest.xml"
    root = ET.parse(manifest).getroot()
    removed = {
        e.get(f"{android}name")
        for e in root.iter()
        if e.tag in ("uses-permission", "provider") and e.get(f"{tools}node") == "remove"
    }
    assert removed == {
        "android.permission.INTERNET",
        "android.permission.ACCESS_NETWORK_STATE",
        pa.TELEMETRY_PROVIDER,
    }
    activity = root.find("application/activity")
    assert activity is not None
    assert activity.get(f"{android}permission") == "android.permission.DUMP"


def test_app_turns_telemetry_off_before_any_session() -> None:
    sources = sorted((REPO / pa.APP_DIR / "app/src/main/kotlin").rglob("*.kt"))
    creating = [s for s in sources if "env.createSession" in s.read_text(encoding="utf-8")]
    assert {s.name for s in creating} >= {"OperatorCheck.kt", "Timing.kt", "Check.kt"}
    for source in creating:
        text = source.read_text(encoding="utf-8")
        assert 0 < text.index("env.setTelemetry(false)") < text.index("env.createSession"), source


def test_build_tools_reads_the_registered_package_hash() -> None:
    tools = pa.build_tools(REPO / pa.APP_DIR)
    assert tools["onnxruntime_android"] == pa.ORT_VERSION
    assert set(tools["onnxruntime_android_aar_sha256"]) != {"0"}


def test_build_tools_refuses_the_placeholder_hash(tmp_path: Path) -> None:
    app = REPO / pa.APP_DIR
    for name in (
        "gradle/wrapper/gradle-wrapper.properties",
        "settings.gradle.kts",
        "app/build.gradle.kts",
    ):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(app / name, tmp_path / name)
    props = "glypharis.jdkMajor=21\nglypharis.ortVersion=1\nglypharis.ortAarSha256=" + "0" * 64
    (tmp_path / "gradle.properties").write_text(props, encoding="utf-8")
    with pytest.raises(SystemExit, match="placeholder"):
        pa.build_tools(tmp_path)


# ---------------------------------------------------------------------------------- refusals


def test_existing_report_is_refused(phone: FakePhone) -> None:
    run_and_report()
    before = pa.REPORT.read_bytes()
    with pytest.raises(SystemExit, match="exists"):
        pa.main(["report"])
    assert pa.REPORT.read_bytes() == before


def test_a_second_run_on_the_same_phone_is_refused(phone: FakePhone) -> None:
    assert pa.main(["run", "--wait-seconds", "5"]) == 0
    with pytest.raises(SystemExit, match="already holds a run"):
        pa.main(["run", "--wait-seconds", "5"])


def test_dirty_tree_is_refused_before_the_phone_is_touched(phone: FakePhone) -> None:
    phone.git["status"] = " M scripts/x.py"
    with pytest.raises(SystemExit, match="not clean"):
        pa.main(["run"])
    assert phone.calls == []


@pytest.mark.parametrize("allow_dirty", [False, True])
def test_apk_with_a_permission_is_refused(phone: FakePhone, allow_dirty: bool) -> None:
    phone.apk_permissions += "uses-permission: name='android.permission.INTERNET'\n"
    with pytest.raises(SystemExit, match="no permissions"):
        pa.main(["run", *(["--allow-dirty"] if allow_dirty else [])])
    assert phone.calls == []


def test_apk_with_the_telemetry_provider_is_refused(phone: FakePhone) -> None:
    phone.apk_manifest += f'A: android:name="{pa.TELEMETRY_PROVIDER}"\n'
    with pytest.raises(SystemExit, match="TelemetryInitializer"):
        pa.main(["run"])
    assert phone.calls == []


def test_phone_without_the_required_abi_is_refused_before_install(phone: FakePhone) -> None:
    phone.props["ro.product.cpu.abilist"] = "armeabi-v7a,armeabi"
    with pytest.raises(SystemExit, match=pa.REQUIRED_ABI):
        pa.main(["run"])
    assert not phone.installed


def test_changed_prepared_file_is_refused(phone: FakePhone) -> None:
    (pa.PREPARED / pa.PHONE_INPUT).write_bytes(b"changed")
    with pytest.raises(SystemExit, match="differs from prepared"):
        pa.main(["run"])
    assert phone.calls == []


def test_refused_start_records_nothing(phone: FakePhone) -> None:
    phone.start_refused = True
    with pytest.raises(SystemExit, match="did not start"):
        pa.main(["run", "--wait-seconds", "0"])
    assert not (pa.RUNS / SLUG / "run.json").exists()


def test_stale_phone_files_are_not_taken_for_this_run(phone: FakePhone) -> None:
    phone.files[pa.phone_path(pa.PHONE_RESULT)] = b'{"status": "ok"}'
    phone.files[pa.phone_path(pa.PHONE_LOGITS)] = b"old"
    phone.writes_result = False
    with pytest.raises(SystemExit, match=r"no result\.json"):
        pa.main(["run", "--wait-seconds", "0"])
    assert pa.phone_path(pa.PHONE_LOGITS) not in phone.files
    assert not (pa.RUNS / SLUG / pa.PHONE_RESULT).exists()


def test_damaged_push_is_refused(phone: FakePhone) -> None:
    phone.corrupt_push = True
    with pytest.raises(SystemExit, match="SHA-256"):
        pa.main(["run", "--wait-seconds", "5"])
    assert not (pa.RUNS / SLUG / "run.json").exists()


def test_wrong_model_is_refused(phone: FakePhone) -> None:
    pa.MODEL.write_bytes(b"something else")
    with pytest.raises(SystemExit, match="not the GW-EFF-009 model"):
        pa.main(["run"])
    assert phone.calls == []


def test_app_that_cannot_read_its_files_gives_no_report(phone: FakePhone) -> None:
    phone.outcome = {
        "status": "bad_request",
        "session_created": False,
        "request_error": {"message": "input.gwt is missing or cannot be read"},
    }
    assert pa.main(["run", "--wait-seconds", "5"]) == 0
    with pytest.raises(SystemExit, match="no verdict"):
        pa.main(["report"])
    assert not pa.REPORT.exists()


def test_no_result_in_time_records_nothing(phone: FakePhone, monkeypatch) -> None:
    original = phone.start

    def silent(words: list[str]) -> pa.AdbResult:
        out = original(words)
        phone.files.pop(pa.phone_path(pa.PHONE_RESULT), None)
        return out

    monkeypatch.setattr(phone, "start", silent)
    with pytest.raises(SystemExit, match=r"no result\.json"):
        pa.main(["run", "--wait-seconds", "0"])
    assert not (pa.RUNS / SLUG / "run.json").exists()


def test_report_without_a_run_is_refused(phone: FakePhone) -> None:
    with pytest.raises(SystemExit, match="no phone has a finished run"):
        pa.main(["report"])


# ------------------------------------------------------------------------------------ cleanup


def test_cleanup_removes_files_and_app(phone: FakePhone) -> None:
    assert pa.main(["run", "--wait-seconds", "5"]) == 0
    assert any(k.endswith(pa.PHONE_MODEL) for k in phone.files)
    assert pa.main(["cleanup"]) == 0
    assert phone.files == {} and not phone.installed
    doc = json.loads((pa.RUNS / SLUG / "cleanup.json").read_text(encoding="utf-8"))
    assert doc["phone_files_removed"] is True and doc["app_uninstalled"] is True


def test_cleanup_refuses_when_the_app_stays(phone: FakePhone) -> None:
    assert pa.main(["run", "--wait-seconds", "5"]) == 0
    phone.uninstall_works = False
    with pytest.raises(SystemExit, match="still on the phone"):
        pa.main(["cleanup"])
    assert not (pa.RUNS / SLUG / "cleanup.json").exists()


# ========================================================================= part b: phone timing
# The pipelines, the laptop's sessions and the timing app are fakes.

WARMUP_PAGES = 3
MODEL_FILES = {
    "detector.onnx": b"detector",
    "recognizer.fused.onnx": b"fused",
    "recognizer.fused.int8.onnx": b"int8",
}
BATTERY = """Current Battery Service state:
  AC powered: false
  USB powered: true
  Wireless powered: false
  status: 2
  health: 2
  level: 81
  scale: 100
  temperature: 287
"""
THERMAL = """IsStatusOverride: false
Thermal Status: 1
Cached temperatures:
\tTemperature{mValue=30.0, mType=3, mName=skin, mStatus=0}
HAL Ready: true
Current temperatures from HAL:
\tTemperature{mValue=36.5, mType=0, mName=cpu-0-0, mStatus=0}
\tTemperature{mValue=31.2, mType=2, mName=battery, mStatus=0}
Current cooling devices from HAL:
\tCoolingDevice{mValue=0, mType=2, mName=cpu}
"""
POWER = f"""POWER MANAGER (dumpsys power)
  mWakefulness=Awake
  mWakefulnessChanging=false
  mLastWakeReason={SERIAL}
Display Power: state=ON
"""
ACTIVITIES = """ACTIVITY MANAGER ACTIVITIES (dumpsys activity activities)
Display #0 (activities from top to bottom):
    mResumedActivity: ActivityRecord{4d5e6f u0 org.glypharis.bench/.BenchActivity t12}
  ResumedActivity: ActivityRecord{4d5e6f u0 org.glypharis.bench/.BenchActivity t12}
"""


def fake_logits(feeds: Mapping[str, np.ndarray]) -> np.ndarray:
    """A recognizer's logits ``(T, N, classes)``, one frame per 4 columns, that depend on each
    line's width and pixels."""
    images, widths = feeds["images"], feeds["widths"]
    classes = len(CHARACTERS) + 1
    out = np.zeros((images.shape[3] // 4, images.shape[0], classes), dtype=np.float32)
    for j, width in enumerate(widths):
        for t in range(out.shape[0]):
            out[t, j, (int(width) // 3 + t // 2) % classes] = 5.0 + float(images[j, 0, 0, 0])
    return out


def fake_prob(feeds: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.zeros((1, 1, *feeds["image"].shape[2:]), dtype=np.float32)


class FakeSession:
    """An ONNX Runtime session: ``run`` gives ``fn(feeds)`` and keeps the feeds it was given."""

    def __init__(self, fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]) -> None:
        self.fn = fn
        self.seen: list[dict[str, np.ndarray]] = []

    def run(
        self, output_names: list[str], feeds: Mapping[str, np.ndarray], run_options: Any = None
    ) -> list[np.ndarray]:
        self.seen.append(dict(feeds))
        return [self.fn(feeds)]


class FakeLineModel:
    stride = 4

    def __init__(self) -> None:
        self.session = FakeSession(fake_logits)

    def frames(self, widths: torch.Tensor) -> torch.Tensor:
        return torch.div(widths, self.stride, rounding_mode="floor")


class FakeRecognizer:
    """Reads crops (here, their widths) ``per_call`` to a network call, padded past the widest."""

    def __init__(self, per_call: int) -> None:
        self.model = FakeLineModel()
        self.charset = Charset(CHARACTERS)
        self.per_call = per_call

    def logits(self, crops: list[int]) -> None:
        for start in range(0, len(crops), self.per_call):
            widths = crops[start : start + self.per_call]
            images = np.full((len(widths), 1, 4, max(widths) + 6), 0.25, dtype=np.float32)
            for j, width in enumerate(widths):
                images[j, 0, :, :width] = width / 100
            feeds = {"images": images, "widths": np.array(widths, dtype=np.int64)}
            self.model.session.run(["logits"], feeds)


class FakeOCR:
    """A pipeline: a page calls the detector once, then the recognizer on the page's lines. Pixel
    (0, 0, 1) of a page holds its line count and row 0 of channel 0 their widths."""

    def __init__(self, per_call: int) -> None:
        self.detector = SimpleNamespace(backend=SimpleNamespace(session=FakeSession(fake_prob)))
        self.recognizer = FakeRecognizer(per_call)

    def predict(self, rgb: np.ndarray) -> None:
        image = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        self.detector.backend.session.run(["prob"], {"image": image})
        count = int(rgb[0, 0, 1])
        self.recognizer.logits([int(w) for w in rgb[0, :count, 0]])


def fake_page(k: int) -> np.ndarray:
    rgb = np.full((8, 8, 3), k, dtype=np.uint8)
    count = k % 3  # 0, 1 or 2 lines
    rgb[0, 0, 1] = count
    rgb[0, :count, 0] = [10 + 2 * k + 3 * j for j in range(count)]
    return rgb


FAKE_PAGES = [(f"doc{k}", fake_page(k)) for k in range(7)]  # 3 warm-up and 4 timed pages
FAKE_LINES = [(f"doc{k}/p0/l0", 12 + 5 * k) for k in range(3)]


def make_timing(arms: Sequence[str] = cand.ARMS) -> dict[str, Any]:
    """What `prepare-timing` leaves, made from the fake pipelines: the model files it names, the
    accuracy report it read, the tensor files, plans and laptop records, and ``prepare.json``."""
    models = {}
    for name, data in MODEL_FILES.items():
        path = Path("models") / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        models[name] = {"source": path.as_posix(), "sha256": sha, "bytes": len(data)}
    accuracy = {
        "to_phone": list(arms),
        "arms": {a: {"met": a in arms} for a in cand.ARMS if a != cand.DEFAULT_ARM},
    }
    pa.ACCURACY.parent.mkdir(parents=True, exist_ok=True)
    pa.ACCURACY.write_text(json.dumps(accuracy), encoding="utf-8")
    pa.TIMING.mkdir(parents=True)
    store = pa.TensorStore(pa.TIMING)
    recorded = {}
    for arm in arms:
        name = "recognizer.fused.int8.onnx" if arm == "fused-int8" else "recognizer.fused.onnx"
        roles = {
            "detector": {"file": "detector.onnx", "sha256": models["detector.onnx"]["sha256"]},
            "recognizer": {"file": name, "sha256": models[name]["sha256"]},
        }
        ocr = FakeOCR(1 if arm == "fused-single-line" else 8)
        recorded[arm] = (roles, pa.record_arm(ocr, FAKE_PAGES, FAKE_LINES, WARMUP_PAGES, store))
    buffer = max(
        store.arena[c.tensor] for _, items in recorded.values() for i in items for c in i["calls"]
    )
    records = {
        arm: pa.write_arm(pa.TIMING, arm, 4, roles, items, store.tensors, buffer)
        for arm, (roles, items) in recorded.items()
    }
    accuracy_record = {
        "file": pa.ACCURACY.as_posix(),
        "sha256": pa.sha256_file(pa.ACCURACY),
        "to_phone": list(arms),
    }
    doc = pa.timing_record(
        records,
        store.tensors,
        models,
        4,
        CHARACTERS,
        False,
        accuracy=accuracy_record,
        git_dirty=False,
        buffer_bytes=buffer,
    )
    pa.write_json_atomic(pa.TIMING / "prepare.json", doc)
    return doc


class TimingPhone(FakePhone):
    """`FakePhone` with the timing app (plans in; timing results, check results and outputs out),
    several files to a push or pull, ``sha256sum`` in the app's folder, patterns in ``rm`` and
    the ``dumpsys`` and ``settings`` readings."""

    def __init__(self) -> None:
        super().__init__()
        self.pid = 4000
        self.reuse_pid = False
        self.base_ms = {"fused": 100.0, "fused-int8": 50.0, "fused-single-line": 120.0}
        self.hwm_kb = {"fused": 300_000, "fused-int8": 250_000, "fused-single-line": 310_000}
        self.check_offset: dict[str, float] = {}
        self.timing_status: dict[tuple[str, int], str] = {}
        self.started: list[str] = []
        self.at_uninstall: list[str] | None = None
        # (arm, run) of a timing run, or (arm, None) of a check, whose process dies unwritten
        self.dies: set[tuple[str, int | None]] = set()
        self.alive = False

    def __call__(
        self, args: list[str], serial: str | None = None, timeout: float = 60.0
    ) -> pa.AdbResult:
        head = args[0]
        if head == "push" and args[-1] == pa.PHONE_DIR:
            self.calls.append((serial, list(args)))
            for source in args[1:-1]:
                data = Path(source).read_bytes()
                self.files[pa.phone_path(Path(source).name)] = (
                    b"x" + data if self.corrupt_push else data
                )
            return pa.AdbResult(0, f"{len(args) - 2} files pushed\n", "")
        if head == "pull" and Path(args[-1]).is_dir():
            self.calls.append((serial, list(args)))
            for remote in args[1:-1]:
                if remote not in self.files:
                    return pa.AdbResult(1, "", f"remote object {remote} does not exist")
                (Path(args[-1]) / remote.rsplit("/", 1)[1]).write_bytes(self.files[remote])
            return pa.AdbResult(0, f"{len(args) - 2} files pulled\n", "")
        if head == "uninstall":
            self.at_uninstall = sorted(self.files)
        return super().__call__(args, serial, timeout)

    def shell(self, command: str) -> pa.AdbResult:
        hashing = f"cd {pa.PHONE_DIR} && sha256sum "
        if command.startswith(hashing):
            rows = []
            for name in shlex.split(command[len(hashing) :]):
                data = self.files.get(pa.phone_path(name))
                if data is None:
                    return pa.AdbResult(1, "", f"sha256sum: {name}: No such file or directory\n")
                rows.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
            return pa.AdbResult(0, "\n".join(rows) + "\n", "")
        readings = {
            "dumpsys battery": BATTERY,
            "dumpsys thermalservice": THERMAL,
            "dumpsys power": POWER,
            "dumpsys activity activities": ACTIVITIES,
            "settings get global airplane_mode_on": "1\n",
        }
        if command in readings:
            return pa.AdbResult(0, readings[command], "")
        if command == f"pidof {pa.APP_ID} >/dev/null":
            return pa.AdbResult(0 if self.alive else 1, "", "")
        words = shlex.split(command)
        if words[0] == "rm":
            for pattern in words[2:]:
                for key in [k for k in self.files if fnmatch.fnmatchcase(k, pattern)]:
                    del self.files[key]
        if words[:2] == ["am", "start"]:
            self.started.append(command)
        return super().shell(command)

    def start(self, words: list[str]) -> pa.AdbResult:
        extras = {words[i + 1]: words[i + 2] for i, w in enumerate(words) if w in ("--es", "--ei")}
        self.alive = True
        if extras.get("mode") in ("timing", "check"):
            plan = json.loads(self.files[pa.phone_path(extras["plan"])])
            run = int(extras["run"]) if extras.get("mode") == "timing" else None
            if (plan["arm"], run) in self.dies:
                self.alive = False  # killed before it wrote anything
                return pa.AdbResult(0, "Status: ok\n", "")
        if extras.get("mode") == "timing":
            return self.app_timing(extras)
        if extras.get("mode") == "check":
            return self.app_check(extras)
        return super().start(words)

    def next_pid(self) -> int:
        if not self.reuse_pid:
            self.pid += 1
        return self.pid

    def sha(self, name: str) -> str:
        return hashlib.sha256(self.files[pa.phone_path(name)]).hexdigest()

    def call_ms(self, arm: str, run: int, key: str, model: str) -> float:
        """The arm's base time, the item's own share and a little drift by run."""
        share = int(hashlib.sha256(f"{key}/{model}".encode()).hexdigest()[:4], 16) / 0xFFFF
        scale = 0.3 if model == "detector" else 1.0
        return round(self.base_ms[arm] * scale * (1 + share) * (1 + 0.01 * run), 3)

    def app_timing(self, extras: dict[str, str]) -> pa.AdbResult:
        plan = json.loads(self.files[pa.phone_path(extras["plan"])])
        arm, run = plan["arm"], int(extras["run"])
        status = self.timing_status.get((arm, run), "ok")
        items = []
        for item in plan["items"]:
            calls_ms = [self.call_ms(arm, run, item["key"], c["model"]) for c in item["calls"]]
            detector = sum(
                t for t, c in zip(calls_ms, item["calls"], strict=True) if c["model"] == "detector"
            )
            items.append(
                {
                    "kind": item["kind"],
                    "key": item["key"],
                    "warmup": item["warmup"],
                    "network_ms": sum(calls_ms),
                    "detector_ms": detector,
                    "recognizer_ms": sum(calls_ms) - detector,
                    "calls_ms": calls_ms,
                }
            )
        refused = {"message": "t_0.gwt cannot be read"} if status == "bad_request" else None
        result = {
            "schema": "glypharis-android-bench-1",
            "mode": "timing",
            "status": status,
            "failed_model": None if status == "ok" else "recognizer",
            "request_error": refused,
            "pid": self.next_pid(),
            "onnxruntime_version": pa.ORT_VERSION,
            "session_options": {**SESSION_OPTIONS, "intra_op_num_threads": int(extras["threads"])},
            "plan_sha256": self.sha(extras["plan"]),
            "model_sha256": {role: self.sha(m["file"]) for role, m in plan["models"].items()},
            "items": items if status == "ok" else [],
            "vm_hwm_kb": self.hwm_kb[arm] + run,
            "vm_rss_kb": 200_000,
            "thermal_start": {"status": 0, "status_name": "none", "headroom": 0.3},
            "thermal_end": {"status": 1, "status_name": "light", "headroom": 0.6},
            "battery_start": {"temperature_c": 29.0, "level_percent": 81, "plugged": 2},
            "battery_end": {"temperature_c": 33.5, "level_percent": 80, "plugged": 2},
            "interactive_start": True,
            "interactive_end": True,
        }
        self.files[pa.phone_path(extras["result"])] = json.dumps(result).encode()
        return pa.AdbResult(0, "Status: ok\n", "")

    def app_check(self, extras: dict[str, str]) -> pa.AdbResult:
        plan = json.loads(self.files[pa.phone_path(extras["plan"])])
        offset = np.float32(self.check_offset.get(plan["arm"], 0.0))
        calls = []
        for call in plan["calls"]:
            logits = fake_logits(pa.parse_gwt(self.files[pa.phone_path(call["tensor"])])) + offset
            self.files[pa.phone_path(call["output"])] = pa.gwt_bytes({"logits": logits})
            finite = bool(np.isfinite(logits).all())
            calls.append({"output_shape": list(logits.shape), "logits_finite": finite})
        result = {
            "schema": "glypharis-android-bench-1",
            "mode": "check",
            "status": "ok",
            "pid": self.next_pid(),
            "onnxruntime_version": pa.ORT_VERSION,
            "session_options": {**SESSION_OPTIONS, "intra_op_num_threads": int(extras["threads"])},
            "plan_sha256": self.sha(extras["plan"]),
            "model_sha256": {"recognizer": self.sha(plan["model"]["file"])},
            "calls": calls,
        }
        self.files[pa.phone_path(extras["result"])] = json.dumps(result).encode()
        return pa.AdbResult(0, "Status: ok\n", "")


@pytest.fixture
def timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TimingPhone:
    """A repository in ``tmp_path`` with a clean build and the prepared timing files, and a fake
    phone running the timing app behind `run_adb`."""
    monkeypatch.chdir(tmp_path)
    fake = TimingPhone()
    monkeypatch.setattr(pa, "run_adb", fake)
    monkeypatch.setattr(pa, "run_aapt2", fake.aapt2)
    monkeypatch.setattr(pa, "POLL_SECONDS", 0.0)
    monkeypatch.setattr(pa, "FIRST_PHONE", SLUG)  # the fake phone stands in for the M52
    monkeypatch.setattr(pa, "_git", lambda *args: fake.git[args[0]])
    monkeypatch.setattr(pa, "laptop_session", lambda path, threads: FakeSession(fake_logits))
    pa.APK.parent.mkdir(parents=True)
    pa.APK.write_bytes(b"apk")
    record_build()
    make_timing()

    def no_subprocess(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"a process was started outside run_adb: {args}")

    monkeypatch.setattr(pa.subprocess, "run", no_subprocess)
    return fake


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_sitting() -> None:
    """Push, then the six timing runs in the registered order, then the three checks."""
    assert pa.main(["push-timing"]) == 0
    for arm, run in cand.sitting_order():
        assert pa.main(["time-run", "--arm", arm, "--run", str(run)]) == 0
    for arm in cand.ARMS:
        assert pa.main(["check-run", "--arm", arm]) == 0


# ---------------------------------------------------------------------- tensor files, uint8


def test_gwt_uint8_round_trip_and_layout(tmp_path: Path) -> None:
    path = tmp_path / "t.gwt"
    image = np.arange(24, dtype=np.uint8).reshape(1, 3, 2, 4)
    pa.write_gwt(path, {"image": image, "widths": np.array([3], dtype=np.int64)})
    got = pa.read_gwt(path)
    assert got["image"].dtype == np.uint8 and np.array_equal(got["image"], image)
    assert got["widths"].dtype == np.int64 and got["widths"].tolist() == [3]
    raw = path.read_bytes()
    # name length and name, element type 3, rank 4, four dimensions, then one byte per element
    assert raw[8:12] == (5).to_bytes(4, "little") and raw[12:17] == b"image"
    assert raw[17:21] == (3).to_bytes(4, "little") and raw[21:25] == (4).to_bytes(4, "little")
    assert raw[57:65] == (24).to_bytes(8, "little") and raw[65:89] == image.tobytes()


def test_gwt_holds_unsigned_integers_as_bytes_only(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="uint8"):
        pa.write_gwt(tmp_path / "t.gwt", {"x": np.zeros(2, dtype=np.uint16)})


# --------------------------------------------------------------------------- prepare-timing


def test_recorder_copies_the_feeds_then_calls_the_original() -> None:
    session = FakeSession(lambda feeds: feeds["x"] * 2)
    stored: list[dict[str, np.ndarray]] = []

    def store(feeds: Mapping[str, np.ndarray]) -> str:
        stored.append(dict(feeds))
        return f"t{len(stored)}"

    recorder = pa.RunRecorder(store, lambda feeds, logits: ([], []))
    x = np.arange(4, dtype=np.float32)
    with recorder.wrapping({"detector": session}):
        recorder.mark("page", "p", True)
        out = session.run(["y"], {"x": x})
        x[:] = -1  # the caller changes its array after the call
    assert np.array_equal(out[0], np.arange(4) * 2)
    assert np.array_equal(stored[0]["x"], np.arange(4)) and stored[0]["x"] is not x
    assert session.seen[0]["x"] is x  # the original ran on the caller's own feeds
    assert "run" not in vars(session)  # and the session has its own run back
    assert [(c.model, c.tensor) for c in recorder.calls] == [("detector", "t1")]


def test_calls_are_grouped_into_pages_and_lines(tmp_path: Path) -> None:
    store = pa.TensorStore(tmp_path)
    fused = pa.record_arm(FakeOCR(8), FAKE_PAGES, FAKE_LINES, WARMUP_PAGES, store)
    single = pa.record_arm(FakeOCR(1), FAKE_PAGES, FAKE_LINES, WARMUP_PAGES, store)
    expected = [("page", key, k < WARMUP_PAGES) for k, (key, _) in enumerate(FAKE_PAGES)]
    expected += [("line", key, False) for key, _ in FAKE_LINES]
    for items in (fused, single):
        assert [(i["kind"], i["key"], i["warmup"]) for i in items] == expected
        for item in items:
            models = [c.model for c in item["calls"]]
            if item["kind"] == "page":
                assert models[0] == "detector" and "detector" not in models[1:]
            else:
                assert models == ["recognizer"]
    # page k holds k % 3 lines: one recognizer call for them, or one per line
    assert [len(i["calls"]) for i in fused[:7]] == [1, 2, 2, 1, 2, 2, 1]
    assert [len(i["calls"]) for i in single[:7]] == [1, 2, 3, 1, 2, 3, 1]
    call = fused[-1]["calls"][0]
    feeds = pa.read_gwt(tmp_path / call.tensor)
    assert feeds["widths"].tolist() == [FAKE_LINES[-1][1]]
    frames = FAKE_LINES[-1][1] // 4
    text = Charset(CHARACTERS).ctc_greedy_decode(fake_logits(feeds)[:frames, 0].argmax(-1).tolist())
    assert call.frames == [frames] and call.texts == [text]
    assert call.shape == list(fake_logits(feeds).shape)


def test_grouping_refuses_calls_that_do_not_fit_their_items() -> None:
    det, rec = pa.Call("detector", "t1"), pa.Call("recognizer", "t2")
    with pytest.raises(SystemExit, match="page p"):
        pa.group_items([rec, det], [pa.Mark("page", "p", True, 0)])
    with pytest.raises(SystemExit, match="page p"):
        pa.group_items([det, det], [pa.Mark("page", "p", True, 0)])
    with pytest.raises(SystemExit, match="line l"):
        pa.group_items([rec, rec], [pa.Mark("line", "l", False, 0)])
    with pytest.raises(SystemExit, match="before the first item"):
        pa.group_items([det, rec], [pa.Mark("page", "p", True, 1)])


def test_tensor_files_are_named_by_content_and_written_once(tmp_path: Path) -> None:
    store = pa.TensorStore(tmp_path)
    image = np.arange(24, dtype=np.uint8).reshape(1, 3, 2, 4)
    name = store.put({"image": image})
    data = (tmp_path / name).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    assert name == f"t_{sha[:40]}.gwt" and store.tensors == {name: {"sha256": sha, "bytes": 89}}
    assert store.put({"image": image.copy()}) == name
    assert [p.name for p in tmp_path.iterdir()] == [name]
    assert store.put({"image": image + 1}) != name and len(store.tensors) == 2


def test_arms_share_their_common_tensor_files(timing: TimingPhone) -> None:
    prepared = read(pa.TIMING / "prepare.json")
    plans = {arm: read(pa.TIMING / f"plan_{arm}.json") for arm in cand.ARMS}
    # fused and fused-int8 batch alike, so they feed the same tensors
    assert plans["fused"]["tensors"] == plans["fused-int8"]["tensors"]

    def firsts(plan: dict[str, Any], kind: str) -> list[str]:
        return [i["calls"][0]["tensor"] for i in plan["items"] if i["kind"] == kind]

    for kind in ("page", "line"):  # the detector's inputs and the lines are the same everywhere
        assert firsts(plans["fused"], kind) == firsts(plans["fused-single-line"], kind)
    files = sorted(p.name for p in pa.TIMING.glob("t_*.gwt"))
    assert files == sorted(prepared["tensors"]) and prepared["unique_tensors"] == len(files)
    assert prepared["unique_tensors"] < sum(len(p["tensors"]) for p in plans.values())
    assert prepared["tensor_bytes"] == sum(t["bytes"] for t in prepared["tensors"].values())


def test_timing_and_check_plans(timing: TimingPhone) -> None:
    prepared = read(pa.TIMING / "prepare.json")
    plan = read(pa.TIMING / "plan_fused.json")
    sha = {name: hashlib.sha256(data).hexdigest() for name, data in MODEL_FILES.items()}
    assert (plan["schema"], plan["experiment"], plan["arm"], plan["threads"]) == (
        "glypharis-phone-plan-1",
        "GW-EFF-015",
        "fused",
        4,
    )
    assert plan["models"] == {
        "detector": {"file": "detector.onnx", "sha256": sha["detector.onnx"], "output": "prob"},
        "recognizer": {
            "file": "recognizer.fused.onnx",
            "sha256": sha["recognizer.fused.onnx"],
            "output": "logits",
        },
    }
    assert (plan["warmup_pages"], plan["timed_pages"], plan["lines"]) == (3, 4, 3)
    assert [i["key"] for i in plan["items"]] == [k for k, _ in FAKE_PAGES + FAKE_LINES]
    assert [i["warmup"] for i in plan["items"]] == [True] * 3 + [False] * 7
    assert set(plan["tensors"]) == {c["tensor"] for i in plan["items"] for c in i["calls"]}
    for name, record in plan["tensors"].items():
        assert record == prepared["tensors"][name]
        assert hashlib.sha256((pa.TIMING / name).read_bytes()).hexdigest() == record["sha256"]
    image = pa.read_gwt(pa.TIMING / plan["items"][0]["calls"][0]["tensor"])["image"]
    assert image.dtype == np.uint8 and image.shape == (1, 3, 8, 8)
    int8 = read(pa.TIMING / "plan_fused-int8.json")["models"]["recognizer"]
    assert int8["file"] == "recognizer.fused.int8.onnx"

    check = read(pa.TIMING / "check_fused.json")
    assert check["schema"] == "glypharis-phone-check-1" and check["arm"] == "fused"
    assert check["model"] == plan["models"]["recognizer"] and check["threads"] == 4
    lines = [f"line {key}" for key, _ in FAKE_LINES]
    assert [c["source"] for c in check["calls"]] == ["page doc1 call 1", "page doc2 call 1", *lines]
    assert [c["output"] for c in check["calls"]] == [f"c_fused_{i}.gwt" for i in range(5)]
    assert set(check["tensors"]) == {c["tensor"] for c in check["calls"]}
    single = read(pa.TIMING / "check_fused-single-line.json")
    assert [c["source"] for c in single["calls"]][:3] == [
        "page doc1 call 1",
        "page doc2 call 1",
        "page doc2 call 2",
    ]
    assert prepared["arms"]["fused"]["check_calls"] == 5
    assert prepared["arms"]["fused-single-line"]["check_calls"] == 6
    assert prepared["arms"]["fused"]["check_lines"] == 6

    laptop = read(pa.TIMING / "laptop_fused.json")["calls"]
    charset = Charset(CHARACTERS)
    for call, record in zip(check["calls"], laptop, strict=True):
        feeds = pa.read_gwt(pa.TIMING / call["tensor"])
        logits = fake_logits(feeds)
        frames = [int(w) // 4 for w in feeds["widths"]]
        texts = [
            charset.ctc_greedy_decode(logits[:f, j].argmax(-1).tolist())
            for j, f in enumerate(frames)
        ]
        assert record["shape"] == list(logits.shape) and record["frames"] == frames
        assert record["texts"] == texts and record["tensor"] == call["tensor"]


def test_prepare_timing_refuses_before_loading_anything(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="needs --out"):
        pa.main(["prepare-timing", "--limit-pages", "3"])
    with pytest.raises(SystemExit, match="exists"):
        pa.main(["prepare-timing", "--limit-lines", "5", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="positive"):
        pa.main(["prepare-timing", "--limit-lines", "0", "--out", str(tmp_path / "new")])


# ------------------------------------------------------------------------------- conditions


def test_conditions_keep_named_fields_only(timing: TimingPhone, tmp_path: Path) -> None:
    assert pa.main(["conditions", "--label", "start"]) == 0
    doc = read(pa.RUNS / SLUG / "conditions_start.json")
    assert doc["battery"] == {
        "level_percent": 81,
        "scale": 100,
        "ac_powered": False,
        "usb_powered": True,
        "wireless_powered": False,
        "status": 2,
        "temperature_c": 28.7,
    }
    assert doc["thermal"]["status"] == 1 and doc["thermal"]["status_name"] == "light"
    assert doc["thermal"]["temperatures"] == [
        {"name": "cpu-0-0", "type": 0, "value_c": 36.5, "status": 0},
        {"name": "battery", "type": 2, "value_c": 31.2, "status": 0},
    ]
    assert doc["airplane_mode"] is True
    assert doc["power"] == {"wakefulness": "Awake", "display_state": "ON"}
    assert doc["resumed_activity"] == "org.glypharis.bench/.BenchActivity"
    assert SERIAL not in written_text(tmp_path)
    with pytest.raises(SystemExit, match="exists"):
        pa.main(["conditions", "--label", "start"])
    with pytest.raises(SystemExit, match="plain label"):
        pa.main(["conditions", "--label", "../start"])


# -------------------------------------------------------------------------- push and runs


def test_push_timing_installs_pushes_and_checks_every_file(timing: TimingPhone) -> None:
    assert pa.main(["push-timing"]) == 0
    prepared = read(pa.TIMING / "prepare.json")
    push = read(pa.RUNS / SLUG / "push.json")
    plans = [r[part]["file"] for r in prepared["arms"].values() for part in ("plan", "check")]
    names = {*prepared["models"], *prepared["tensors"], *plans}
    assert set(push["files"]) == names and push["file_count"] == len(names)
    assert push["phone_hash_check"] == "sha256sum" and timing.installed
    assert push["prepare_sha256"] == pa.sha256_file(pa.TIMING / "prepare.json")
    for name in names:
        assert hashlib.sha256(timing.files[pa.phone_path(name)]).hexdigest() == push["files"][name]
    assert not any(name.startswith("laptop_") for name in names)
    with pytest.raises(SystemExit, match="exists"):
        pa.main(["push-timing"])


def test_damaged_timing_push_is_refused(timing: TimingPhone) -> None:
    timing.corrupt_push = True
    with pytest.raises(SystemExit, match="SHA-256"):
        pa.main(["push-timing"])
    assert not (pa.RUNS / SLUG / "push.json").exists()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"smoke": True}, "smoke test"),
        ({"threads": 2}, "threads"),
    ],
)
def test_push_timing_refuses_an_unregistered_preparation(
    timing: TimingPhone, change: dict[str, Any], message: str
) -> None:
    edit_json(pa.TIMING / "prepare.json", **change)
    with pytest.raises(SystemExit, match=message):
        pa.main(["push-timing"])
    assert timing.calls == []


def test_time_runs_follow_the_registered_order(timing: TimingPhone) -> None:
    with pytest.raises(SystemExit, match="push-timing"):
        pa.main(["time-run", "--arm", "fused", "--run", "1"])
    assert pa.main(["push-timing"]) == 0
    with pytest.raises(SystemExit, match="next run on sm-test-5g is fused run 1"):
        pa.main(["time-run", "--arm", "fused-int8", "--run", "1"])
    with pytest.raises(SystemExit, match="timing runs come first"):
        pa.main(["check-run", "--arm", "fused"])
    for arm, run in cand.sitting_order():
        assert pa.main(["time-run", "--arm", arm, "--run", str(run)]) == 0
    with pytest.raises(SystemExit, match="every registered run"):
        pa.main(["time-run", "--arm", "fused", "--run", "2"])
    folder = pa.RUNS / SLUG
    assert sorted(pa.timing_results(folder)) == sorted(cand.sitting_order())
    starts = [s for s in timing.started if "--es mode timing" in s]
    assert len(starts) == 6
    assert "--es plan plan_fused.json --es result timing_fused_run1.json" in starts[0]
    assert "--ei threads 4 --ei run 1" in starts[0] and "--ei run 2" in starts[-1]
    for arm, run in cand.sitting_order():
        assert (folder / f"conditions_{arm}_run{run}_before.json").is_file()
        assert (folder / f"conditions_{arm}_run{run}_after.json").is_file()
        record = read(folder / f"record_{arm}_run{run}.json")
        assert record["status"] == "ok" and record["git_commit"] == COMMIT
    pids = [read(folder / f"timing_{a}_run{r}.json")["pid"] for a, r in cand.sitting_order()]
    assert len(set(pids)) == 6
    # every run is a new process: the app is stopped before it starts and after it ends
    stops = [c for _, c in timing.calls if c[0] == "shell" and c[1].startswith("am force-stop")]
    assert len(stops) >= 2 * 6


def test_time_run_needs_a_clean_build_of_this_commit(timing: TimingPhone) -> None:
    assert pa.main(["push-timing"]) == 0
    timing.git["status"] = " M scripts/x.py"
    with pytest.raises(SystemExit, match="not clean"):
        pa.main(["time-run", "--arm", "fused", "--run", "1"])
    timing.git.update({"status": "", "rev-parse": "d" * 40})
    with pytest.raises(SystemExit, match="run `build`"):
        pa.main(["time-run", "--arm", "fused", "--run", "1"])


def test_a_result_from_an_earlier_process_is_refused(timing: TimingPhone) -> None:
    assert pa.main(["push-timing"]) == 0
    assert pa.main(["time-run", "--arm", "fused", "--run", "1"]) == 0
    timing.reuse_pid = True
    with pytest.raises(SystemExit, match="not a new process"):
        pa.main(["time-run", "--arm", "fused-int8", "--run", "1"])
    folder = pa.RUNS / SLUG
    assert not (folder / "timing_fused-int8_run1.json").exists()
    assert (folder / "timing_fused-int8_run1.json.part").is_file()
    assert pa.timing_results(folder) == [("fused", 1)]


def test_a_run_the_app_could_not_start_leaves_the_step_due(timing: TimingPhone) -> None:
    assert pa.main(["push-timing"]) == 0
    timing.timing_status[("fused", 1)] = "bad_request"
    with pytest.raises(SystemExit, match=r"could not start the timing pass \(t_0.gwt"):
        pa.main(["time-run", "--arm", "fused", "--run", "1"])
    folder = pa.RUNS / SLUG
    assert pa.timing_results(folder) == [] and (folder / "timing_fused_run1.json.part").is_file()
    del timing.timing_status[("fused", 1)]
    with pytest.raises(SystemExit, match="exists"):  # the attempt's pulled result is kept
        pa.main(["time-run", "--arm", "fused", "--run", "1"])
    part = folder / "timing_fused_run1.json.part"
    part.rename(folder / "attempt1_timing_fused_run1.json.part")
    with pytest.raises(SystemExit, match="exists"):  # and so are its conditions
        pa.main(["time-run", "--arm", "fused", "--run", "1"])
    before = folder / "conditions_fused_run1_before.json"
    before.rename(folder / "attempt1_conditions_fused_run1_before.json")
    assert pa.main(["time-run", "--arm", "fused", "--run", "1"]) == 0
    assert pa.timing_results(folder) == [("fused", 1)]
    assert read(folder / "attempt1_timing_fused_run1.json.part")["status"] == "bad_request"


def test_the_order_holds_over_results_already_pulled() -> None:
    order = [("fused", 1), ("fused-int8", 1), ("fused-int8", 2), ("fused", 2)]
    assert pa.next_step(order, []) == ("fused", 1)
    assert pa.next_step(order, [("fused", 1), ("fused-int8", 1)]) == ("fused-int8", 2)
    assert pa.next_step(order, order) is None
    with pytest.raises(SystemExit, match="order was not kept"):
        pa.next_step(order, [("fused", 1), ("fused-int8", 2)])
    with pytest.raises(SystemExit, match="outside the registered order"):
        pa.next_step(order, [("fused-single-line", 1)])
    assert pa.first_phone_order(["fused", "fused-int8"]) == order


def test_a_second_phone_times_the_selected_arm_only_after_selection(timing: TimingPhone) -> None:
    assert pa.main(["push-timing"]) == 0
    assert pa.main(["time-run", "--arm", "fused", "--run", "1"]) == 0
    timing.props["ro.product.model"] = "moto g35 5G"
    with pytest.raises(SystemExit, match="not the registered first phone"):
        pa.main(["push-timing"])
    assert pa.main(["push-timing", "--first-phone-reason", "the M52 was not available"]) == 0
    with pytest.raises(SystemExit, match="sm-test-5g holds the first phone's timing runs"):
        pa.main(["time-run", "--arm", "fused", "--run", "1"])
    pa.SELECTION.write_text(json.dumps({"device": SLUG, "selected": "fused"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="next run on moto-g35-5g is fused run 1"):
        pa.main(["time-run", "--arm", "fused-int8", "--run", "1"])
    assert pa.main(["time-run", "--arm", "fused", "--run", "1"]) == 0
    assert pa.main(["time-run", "--arm", "fused", "--run", "2"]) == 0
    with pytest.raises(SystemExit, match="every registered run"):
        pa.main(["time-run", "--arm", "fused", "--run", "2"])
    assert pa.device_order("moto-g35-5g", cand.ARMS) == ("second", [("fused", 1), ("fused", 2)])


# -------------------------------------------------------------------------- the check rule


def two_lines() -> np.ndarray:
    """Laptop logits of two lines over six frames: line 0 reads "a", line 1 reads "b"."""
    logits = np.zeros((6, 2, len(CHARACTERS) + 1), dtype=np.float32)
    logits[:, 0, 1] = 5.0
    logits[:, 1, 2] = 5.0
    return logits


APP_CALL = {"output_shape": [6, 2, 4], "logits_finite": True}


def test_check_compares_each_lines_own_frames() -> None:
    charset = Charset(CHARACTERS)
    laptop = two_lines()
    phone = laptop.copy()
    phone[4:, 0] += 3.0  # line 0's padding frames
    phone[2:, 1, 3] = 9.0  # line 1's padding frames even change their top class
    got = pa.compare_call(laptop, phone, [4, 2], ["a", "b"], charset, APP_CALL)
    assert got["max_abs_diff"] == 0.0 and got["identical_text"] == 2
    assert got["laptop_identical_text"] == 2
    verdict = pa.check_verdict([got], judged=True)
    assert verdict["passed"] and verdict["logit_max_abs_diff"] == 0.0 and verdict["lines"] == 2
    # frames are clamped to the output's length
    longer = pa.compare_call(laptop, phone, [9, 2], ["a", "b"], charset, APP_CALL)
    assert longer["max_abs_diff"] == pytest.approx(3.0)


@pytest.mark.parametrize(("delta", "passed"), [(5e-4, True), (2e-3, False)])
def test_check_rule_on_the_logits(delta: float, passed: bool) -> None:
    laptop = two_lines()
    phone = laptop.copy()
    phone[1, 0, 0] += delta
    got = pa.compare_call(laptop, phone, [6, 6], ["a", "b"], Charset(CHARACTERS), APP_CALL)
    verdict = pa.check_verdict([got], judged=True)
    assert verdict["logit_max_abs_diff"] == pytest.approx(delta, abs=1e-6)
    assert verdict["passed"] is passed
    assert verdict["reasons"] == ([] if passed else ["logits_differ"])


def test_check_rule_on_the_text() -> None:
    laptop = two_lines()
    laptop[2, 0, 2] = 4.9998  # a near tie on line 0
    phone = laptop.copy()
    phone[2, 0, 2] = 5.0001
    got = pa.compare_call(laptop, phone, [6, 6], ["a", "b"], Charset(CHARACTERS), APP_CALL)
    verdict = pa.check_verdict([got], judged=True)
    assert verdict["logit_max_abs_diff"] < pa.LOGIT_TOL
    assert verdict["identical_text"] == 1 and verdict["reasons"] == ["text_differs"]


def test_int8_differences_are_reported_not_judged() -> None:
    charset = Charset(CHARACTERS)
    laptop = two_lines()
    phone = laptop + np.float32(0.5)
    phone[2, 0, 2] = 9.0  # the text differs too
    got = pa.compare_call(laptop, phone, [6, 6], ["a", "b"], charset, APP_CALL)
    verdict = pa.check_verdict([got], judged=False)
    assert verdict["passed"] and verdict["reasons"] == []
    assert verdict["logit_max_abs_diff"] == pytest.approx(9.0)
    assert verdict["identical_text"] == 1
    nan = phone.copy()
    nan[0, 1, 0] = np.nan
    app = {**APP_CALL, "logits_finite": False}
    bad = pa.check_verdict([pa.compare_call(laptop, nan, [6, 6], ["a", "b"], charset, app)], False)
    assert not bad["passed"] and "non_finite_logits" in bad["reasons"]
    short = phone[:5]
    app = {**APP_CALL, "output_shape": [5, 2, 4]}
    shape = pa.compare_call(laptop, short, [6, 6], ["a", "b"], charset, app)
    assert pa.check_verdict([shape], False)["reasons"] == ["wrong_shape"]
    missing = pa.compare_call(laptop, None, [6, 6], ["a", "b"], charset, APP_CALL)
    assert pa.check_verdict([missing], False)["reasons"] == ["outputs_missing"]


# ------------------------------------------------------------------- phone report and select


def test_phone_report_numbers_and_conditions(timing: TimingPhone, tmp_path: Path) -> None:
    timing.check_offset = {"fused-int8": 0.01, "fused-single-line": 0.002}
    run_sitting()
    with pytest.raises(SystemExit, match="phone-report"):
        pa.main(["select", "--device", SLUG])
    assert pa.main(["phone-report", "--device", SLUG]) == 0
    report = read(pa.PHONE_REPORTS / f"phone_{SLUG}.json")
    folder = pa.RUNS / SLUG
    plan = read(pa.TIMING / "plan_fused.json")["items"]
    timed = [k for k, item in enumerate(plan) if item["kind"] == "page" and not item["warmup"]]

    def page_times(arm: str) -> list[list[float]]:
        runs = [read(folder / f"timing_{arm}_run{run}.json")["items"] for run in (1, 2)]
        return [[items[k]["network_ms"] for k in timed] for items in runs]

    assert report["role"] == "first" and report["pages"] == len(timed) == 4
    assert report["order"] == [f"{a} run {r}" for a, r in cand.sitting_order()]
    for arm in cand.ARMS:
        entry = report["arms"][arm]
        runs = page_times(arm)
        both = np.mean(runs, axis=0)
        assert entry["run_mean_ms"] == pytest.approx([np.mean(r) for r in runs])
        assert entry["run_p95_ms"] == pytest.approx([np.percentile(r, 95) for r in runs])
        assert entry["pages_both_runs"]["mean_ms"] == pytest.approx(both.mean())
        assert entry["pages_both_runs"]["p95_ms"] == pytest.approx(np.percentile(both, 95))
        assert entry["peak_vm_hwm_kb"] == timing.hwm_kb[arm] + 2 and entry["loaded"]
        assert entry["runs"][0]["thermal_start"]["status_name"] == "none"
        assert entry["runs"][1]["thermal_end"]["status_name"] == "light"
        assert entry["runs"][1]["conditions_before"]["thermal"]["status"] == 1
        assert entry["lines_both_runs"]["count"] == len(FAKE_LINES)
        if arm != cand.DEFAULT_ARM:
            assert entry["speed"] == cand.speed_tests(runs, page_times(cand.DEFAULT_ARM))
            fused = timing.hwm_kb[cand.DEFAULT_ARM]
            peaks = [timing.hwm_kb[arm] + 1, timing.hwm_kb[arm] + 2]
            assert entry["memory"] == cand.memory_test(peaks, [fused + 1, fused + 2])
    assert report["arms"]["fused-int8"]["speed"]["met"]
    assert report["arms"]["fused-int8"]["memory"]["met"]
    assert not report["arms"]["fused-single-line"]["speed"]["met"]
    assert "speed" not in report["arms"]["fused"]

    checks = {arm: report["arms"][arm]["check"] for arm in cand.ARMS}
    assert checks["fused"]["passed"] and checks["fused"]["logit_max_abs_diff"] == 0.0
    assert checks["fused"]["identical_text"] == checks["fused"]["lines"] == 6
    assert checks["fused"]["laptop_rerun_identical_text"] == 6
    assert checks["fused-int8"]["passed"] and not checks["fused-int8"]["judged"]
    assert checks["fused-int8"]["logit_max_abs_diff"] == pytest.approx(0.01, abs=1e-5)
    assert not checks["fused-single-line"]["passed"]
    assert checks["fused-single-line"]["reasons"] == ["logits_differ"]
    assert report["fused_failed_phone_check"] is False and report["part_b"] == "passed"
    md = (pa.PHONE_REPORTS / f"phone_{SLUG}.md").read_text(encoding="utf-8")
    assert "Network time and memory only" in md and "not ported" in md
    assert SERIAL not in written_text(tmp_path)
    with pytest.raises(SystemExit, match="exists"):
        pa.main(["phone-report", "--device", SLUG])


def test_select_takes_the_first_phones_report(timing: TimingPhone) -> None:
    timing.check_offset = {"fused-int8": 0.01, "fused-single-line": 0.002}
    run_sitting()
    assert pa.main(["phone-report", "--device", SLUG]) == 0
    assert pa.main(["select", "--device", SLUG]) == 0
    selection = read(pa.SELECTION)
    report = read(pa.PHONE_REPORTS / f"phone_{SLUG}.json")
    keys = ("loaded", "check_passed", "speed", "memory")
    phone = {arm: {k: e.get(k) for k in keys} for arm, e in report["arms"].items()}
    expected = cand.select(read(pa.ACCURACY)["arms"], phone, False)
    assert {k: selection[k] for k in expected} == expected
    assert selection["selected"] == "fused-int8" and selection["qualified"] == ["fused-int8"]
    assert selection["device"] == SLUG and selection["experiment"] == cand.EXPERIMENT
    assert "**fused-int8**" in pa.SELECTION.with_suffix(".md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="exists"):
        pa.main(["select", "--device", SLUG])


def test_fused_failing_its_check_keeps_fused(timing: TimingPhone) -> None:
    timing.check_offset = {"fused": 0.002}
    run_sitting()
    assert pa.main(["phone-report", "--device", SLUG]) == 0
    report = read(pa.PHONE_REPORTS / f"phone_{SLUG}.json")
    assert report["fused_failed_phone_check"] is True and report["part_b"].startswith("failed")
    assert pa.main(["select", "--device", SLUG]) == 0
    selection = read(pa.SELECTION)
    assert selection["selected"] == "fused" and "fused failed" in selection["reason"]


def test_an_arm_whose_session_fails_drops_out(timing: TimingPhone) -> None:
    timing.timing_status[("fused-int8", 2)] = "session_failed"
    run_sitting()
    assert pa.main(["phone-report", "--device", SLUG]) == 0
    entry = read(pa.PHONE_REPORTS / f"phone_{SLUG}.json")["arms"]["fused-int8"]
    assert entry["loaded"] is False and entry["measured"] is False
    assert entry["runs"][1]["failed_model"] == "recognizer"
    assert entry["speed"]["met"] is False and entry["peak_vm_hwm_kb"] is None
    assert pa.main(["select", "--device", SLUG]) == 0
    assert read(pa.SELECTION)["selected"] == "fused"


def test_a_result_that_is_not_its_plans_gives_no_report(timing: TimingPhone) -> None:
    run_sitting()
    path = pa.RUNS / SLUG / "timing_fused-int8_run1.json"
    edit_json(path, plan_sha256="0" * 64)
    with pytest.raises(SystemExit, match="not its plan's"):
        pa.main(["phone-report", "--device", SLUG])
    assert not (pa.PHONE_REPORTS / f"phone_{SLUG}.json").exists()


# -------------------------------------------------------------------------- cleanup, part b


def test_cleanup_deletes_the_timing_files(timing: TimingPhone) -> None:
    assert pa.main(["push-timing"]) == 0
    assert pa.main(["time-run", "--arm", "fused", "--run", "1"]) == 0
    assert any(name.endswith(".gwt") for name in timing.files)
    assert pa.main(["cleanup"]) == 0
    assert timing.at_uninstall == [] and not timing.installed
    assert read(pa.RUNS / SLUG / "cleanup.json")["timing_files_removed"] is True


# ------------------------------------------------------- part b: review fixes (process, pushes)


def test_arena_bytes_align_each_payload() -> None:
    call = {"images": np.zeros((1, 1, 2, 3), np.float32), "widths": np.array([3], np.int64)}
    assert pa.arena_bytes(call) == 24 + 8
    assert pa.arena_bytes({"image": np.zeros(5, np.uint8)}) == 8


def test_every_plan_holds_one_buffer_size_the_largest_call(timing: TimingPhone) -> None:
    prepared = read(pa.TIMING / "prepare.json")
    plans = [read(pa.TIMING / rec["plan"]["file"]) for rec in prepared["arms"].values()]
    assert {p["buffer_bytes"] for p in plans} == {prepared["buffer_bytes"]}
    largest = max(
        pa.arena_bytes(pa.parse_gwt((pa.TIMING / name).read_bytes(), name))
        for name in prepared["tensors"]
    )
    assert prepared["buffer_bytes"] == largest


def test_a_preparation_from_a_tree_with_changes_is_refused(timing: TimingPhone) -> None:
    edit_json(pa.TIMING / "prepare.json", git_dirty=True)
    with pytest.raises(SystemExit, match="tree with changes"):
        pa.main(["push-timing"])


def test_a_timing_process_that_dies_is_recorded_and_the_order_goes_on(
    timing: TimingPhone,
) -> None:
    timing.dies.add(("fused-int8", 1))
    run_sitting()
    folder = pa.RUNS / SLUG
    died = read(folder / "timing_fused-int8_run1.json")
    assert died["status"] == pa.PROCESS_DIED and died["pid"] is None
    assert read(folder / "record_fused-int8_run1.json")["status"] == pa.PROCESS_DIED
    assert (folder / "conditions_fused-int8_run1_after.json").is_file()
    assert pa.main(["phone-report", "--device", SLUG]) == 0
    report = read(pa.PHONE_REPORTS / f"phone_{SLUG}.json")
    entry = report["arms"]["fused-int8"]
    assert entry["measured"] is False and entry["speed"]["met"] is False
    assert report["part_b"] == "passed"
    assert pa.main(["select", "--device", SLUG]) == 0
    assert read(pa.SELECTION)["selected"] != "fused-int8"


def test_a_check_whose_process_dies_fails_that_arm(timing: TimingPhone) -> None:
    timing.dies.add(("fused", None))
    run_sitting()
    assert pa.main(["phone-report", "--device", SLUG]) == 0
    report = read(pa.PHONE_REPORTS / f"phone_{SLUG}.json")
    assert report["arms"]["fused"]["check"]["reasons"] == [f"status_{pa.PROCESS_DIED}"]
    assert report["fused_failed_phone_check"] is True
    assert report["part_b"] == "failed: fused failed the phone check"


def test_the_report_needs_one_push_for_the_whole_sitting(timing: TimingPhone) -> None:
    run_sitting()
    edit_json(pa.RUNS / SLUG / "record_fused-int8_run1.json", push_sha256="0" * 64)
    with pytest.raises(SystemExit, match=r"another push.json"):
        pa.main(["phone-report", "--device", SLUG])


def test_another_first_phone_needs_and_records_a_reason(
    timing: TimingPhone, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pa, "FIRST_PHONE", "sm-m526b")
    with pytest.raises(SystemExit, match="not the registered first phone"):
        pa.main(["push-timing"])
    reason = "the M52 would not stay connected"
    assert pa.main(["push-timing", "--first-phone-reason", reason]) == 0
    for arm, run in cand.sitting_order():
        assert pa.main(["time-run", "--arm", arm, "--run", str(run)]) == 0
    for arm in cand.ARMS:
        assert pa.main(["check-run", "--arm", arm]) == 0
    assert pa.main(["phone-report", "--device", SLUG]) == 0
    assert read(pa.PHONE_REPORTS / f"phone_{SLUG}.json")["first_phone_reason"] == reason
    assert pa.main(["select", "--device", SLUG]) == 0
    assert read(pa.SELECTION)["first_phone_reason"] == reason
    assert reason in pa.SELECTION.with_suffix(".md").read_text(encoding="utf-8")


def test_a_second_phone_reports_on_its_own_arm(timing: TimingPhone) -> None:
    run_sitting()
    assert pa.main(["phone-report", "--device", SLUG]) == 0
    assert pa.main(["select", "--device", SLUG]) == 0
    selected = read(pa.SELECTION)["selected"]
    timing.props["ro.product.model"] = "moto g35 5G"
    timing.check_offset[selected] = float("inf")  # its outputs are not finite on this phone
    assert pa.main(["push-timing"]) == 0
    for run in (1, 2):
        assert pa.main(["time-run", "--arm", selected, "--run", str(run)]) == 0
    assert pa.main(["check-run", "--arm", selected]) == 0
    assert pa.main(["phone-report", "--device", "moto-g35-5g"]) == 0
    report = read(pa.PHONE_REPORTS / "phone_moto-g35-5g.json")
    assert report["role"] == "second" and report["fused_failed_phone_check"] is None
    assert report["part_b"].startswith(f"failed on this phone: {selected}")
    assert report["part_b"].endswith("this phone changes no decision")
