"""The Android operator check (GW-EFF-015 part a): tensor files, device choice, the run on a fake
phone and the go/no-go report. No phone, adb, model or real data is used."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import phase22_android as pa
import pytest
import torch
from PIL import Image

from glypharis.data.text import Charset
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
    source = REPO / pa.APP_DIR / "app/src/main/kotlin/org/glypharis/bench/OperatorCheck.kt"
    text = source.read_text(encoding="utf-8")
    assert 0 < text.index("env.setTelemetry(false)") < text.index("env.createSession")


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
