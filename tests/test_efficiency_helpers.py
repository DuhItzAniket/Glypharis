"""Phase 16 shared helpers: the v1 code check, thread counts and ONNX sessions, run conditions, and
v1's pipeline objects on recorded sessions (tiny models and made-up repositories only)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from det_fixtures import rect_polygon
from test_pipeline import KernelMap, _page, tiny_detector, tiny_recognizer

from glypharis.efficiency import conditions, threads
from glypharis.efficiency.v1 import V1_PATHS, V1_TAG, V1CodeError, assert_v1_code

REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_ID = ("-c", "user.name=test", "-c", "user.email=test@example.invalid",
          "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false")  # fmt: skip


# ------------------------------------------------------------------------------------ v1 code


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *GIT_ID, *args], cwd=repo, capture_output=True, text=True,
                          check=True)  # fmt: skip
    return done.stdout.strip()


def _write(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with v1 paths tagged `V1_TAG` and one later commit outside them."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, ".gitignore", "*.pyc\n")
    for rel in V1_PATHS:
        _write(repo, f"{rel}/a.txt", f"{rel}\n")
    _write(repo, "scripts/s.py", "x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "v1")
    _git(repo, "tag", V1_TAG)
    _write(repo, "scripts/s.py", "x = 2\n")
    _write(repo, "glypharis/efficiency/new.py", "y = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "phase 16 work")
    return repo


def test_v1_tag_moved_and_the_freeze_pin_did_not():
    from glypharis.pipeline.frozen import PIN_TAG

    assert V1_TAG == "v1-code" and PIN_TAG == "phase14-eval"  # the check moved, the pin did not
    assert V1_PATHS == (
        "glypharis/pipeline",
        "glypharis/models",
        "glypharis/export",
        "glypharis/data",
        "configs",
    )


def test_v1_code_passes_when_only_other_paths_changed(repo):
    _write(repo, "scripts/untracked.py", "z = 1\n")  # outside the v1 paths
    _write(repo, "glypharis/models/cache.pyc", "ignored")  # ignored files are not the tree
    got = assert_v1_code(repo)
    assert got["tag"] == V1_TAG
    assert got["tag_commit"] == _git(repo, "rev-parse", f"{V1_TAG}^{{commit}}")
    assert got["head"] == _git(repo, "rev-parse", "HEAD")
    assert got["head"] != got["tag_commit"]
    assert got["paths"] == list(V1_PATHS)
    assert got["v1_code"] == "equal"


def test_v1_code_refuses_uncommitted_staged_and_committed_changes(repo):
    _write(repo, "configs/a.txt", "changed\n")
    with pytest.raises(V1CodeError, match=re.escape("configs/a.txt")):
        assert_v1_code(repo)
    _git(repo, "add", "configs/a.txt")
    with pytest.raises(V1CodeError, match="differs"):
        assert_v1_code(repo)
    _git(repo, "commit", "-q", "-m", "edit a config")
    with pytest.raises(V1CodeError, match="differs"):
        assert_v1_code(repo)


def test_v1_code_refuses_untracked_and_removed_files(repo):
    _write(repo, "glypharis/pipeline/extra.py", "new\n")
    with pytest.raises(V1CodeError, match="untracked"):
        assert_v1_code(repo)
    (repo / "glypharis/pipeline/extra.py").unlink()
    (repo / "glypharis/data/a.txt").unlink()
    with pytest.raises(V1CodeError, match=re.escape("glypharis/data/a.txt")):
        assert_v1_code(repo)


def test_v1_code_refuses_a_missing_tag(repo):
    _git(repo, "tag", "-d", V1_TAG)
    with pytest.raises(V1CodeError, match="does not exist"):
        assert_v1_code(repo)


# ------------------------------------------------------------------------------------ threads


def test_thread_env_is_the_one_the_baselines_record():
    import run_baselines

    assert threads.THREAD_ENV == run_baselines.THREAD_ENV


def test_set_thread_env_and_check_threads(monkeypatch):
    for key in threads.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    with pytest.raises(ValueError):
        threads.set_thread_env(0)
    assert threads.set_thread_env(3) == {k: "3" for k in threads.THREAD_ENV}
    assert all(os.environ[k] == "3" for k in threads.THREAD_ENV)
    other = torch.get_num_threads() + 1  # variables right, torch's pools not
    threads.set_thread_env(other)
    with pytest.raises(threads.ThreadError, match="torch runs"):
        threads.check_threads(other)
    intra, inter = torch.get_num_threads(), torch.get_num_interop_threads()
    if intra == inter:
        threads.set_thread_env(intra)
        assert threads.check_threads(intra)["intra_op"] == intra
    threads.set_thread_env(3)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    with pytest.raises(threads.ThreadError, match="thread variables"):
        threads.check_threads(3)


def test_thread_record_reads_live_values():
    rec = threads.thread_record()
    assert rec["intra_op"] == torch.get_num_threads()
    assert rec["inter_op"] == torch.get_num_interop_threads()
    assert set(rec["env"]) == set(threads.THREAD_ENV)


def test_set_threads_refuses_after_numpy_or_torch_loaded(monkeypatch):
    for key in threads.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    with pytest.raises(threads.ThreadError, match="already imported"):
        threads.set_threads(2)
    assert all(os.environ[k] == "1" for k in threads.THREAD_ENV)  # nothing was changed


def test_set_threads_in_a_fresh_process():
    code = (
        "import json, sys\n"
        "from glypharis.efficiency.threads import ThreadError, check_threads, set_threads\n"
        "first = set_threads(3)\n"
        "check_threads(3)\n"
        "try:\n"
        "    set_threads(4, require_fresh=False)\n"
        "    second = 'no error'\n"
        "except ThreadError as exc:\n"
        "    second = str(exc)\n"
        "print(json.dumps({'first': first, 'second': second}))\n"
    )
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "scripts")]),
    }
    for key in threads.THREAD_ENV:
        env.pop(key, None)
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=300, check=True)  # fmt: skip
    got = json.loads(done.stdout.strip().splitlines()[-1])
    first = got["first"]
    assert first["requested"] == 3
    assert (first["intra_op"], first["inter_op"]) == (3, 3)
    assert first["env"] == {k: "3" for k in threads.THREAD_ENV}
    assert first["env_before_torch_import"] is True and first["preloaded"] == []
    assert "fresh process" in got["second"]  # the inter-op pool is fixed once per process


# ------------------------------------------------------------------------------- ONNX sessions


@pytest.fixture
def identity_onnx(tmp_path: Path) -> Path:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "identity",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = tmp_path / "identity.onnx"
    onnx.save(model, str(path))
    return path


def test_ort_session_options_are_recorded_and_checked(identity_onnx):
    session = threads.ort_session(identity_onnx, 4)
    x = np.arange(5, dtype=np.float32)
    assert np.array_equal(session.run(["y"], {"x": x})[0], x)
    got = threads.check_session(session, 4)
    assert got == threads.session_options(session)
    assert got["intra_op_num_threads"] == 4
    assert got["execution_mode"] == "ORT_SEQUENTIAL"
    assert got["graph_optimization_level"] == "ORT_ENABLE_ALL"
    assert got["enable_cpu_mem_arena"] is False and got["enable_mem_pattern"] is False
    assert got["allow_spinning"] == "default"
    assert got["providers"] == ["CPUExecutionProvider"]
    json.dumps(got)
    with pytest.raises(threads.ThreadError, match="intra_op_num_threads"):
        threads.check_session(session, 5)
    default = threads.ort_session(identity_onnx, None)
    assert threads.check_session(default, None)["intra_op_num_threads"] == 0
    with pytest.raises(ValueError):
        threads.ort_session(identity_onnx, 0)


def test_check_session_refuses_the_arena(identity_onnx):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    plain = ort.InferenceSession(str(identity_onnx), opts, providers=["CPUExecutionProvider"])
    with pytest.raises(threads.ThreadError, match="enable_cpu_mem_arena"):
        threads.check_session(plain, 4)


def test_ort_session_sets_what_the_v1_export_sets(identity_onnx):
    """Same options as `glypharis.export.onnx` (whose sequential mode is ONNX Runtime's default)."""
    from glypharis.export.onnx import _session

    assert threads.session_options(_session(identity_onnx, 4)) == threads.session_options(
        threads.ort_session(identity_onnx, 4)
    )


# ---------------------------------------------------------------------------------- conditions

SMI_LINE = (
    "0, NVIDIA GeForce RTX 4050 Laptop GPU, 616.56, P0, 2310, 8201, 3105, 8001, 49, 173, 6141, "
    "23, 15.32, 0x0000000000000001"
)


def test_parse_gpu_query():
    [gpu] = conditions.parse_gpu_query(SMI_LINE + "\n")
    assert gpu == {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "driver": "616.56",
        "pstate": "P0",
        "clock_sm_mhz": 2310,
        "clock_mem_mhz": 8201,
        "clock_max_sm_mhz": 3105,
        "clock_max_mem_mhz": 8001,
        "temperature_c": 49,
        "memory_used_mib": 173,
        "memory_total_mib": 6141,
        "utilization_percent": 23,
        "power_draw_w": 15.32,
        "throttle_reasons": "0x0000000000000001",
    }
    [gpu] = conditions.parse_gpu_query(SMI_LINE.replace("15.32", "[N/A]"))
    assert gpu["power_draw_w"] is None
    with pytest.raises(ValueError):
        conditions.parse_gpu_query("0, name, 1")


class _Done:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode, self.stdout = returncode, stdout


def test_gpu_state_without_and_with_nvidia_smi(monkeypatch):
    monkeypatch.setattr(conditions.shutil, "which", lambda name: None)
    assert conditions.gpu_state() is None
    monkeypatch.setattr(conditions.shutil, "which", lambda name: "nvidia-smi")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return _Done(0, SMI_LINE + "\n")

    monkeypatch.setattr(conditions.subprocess, "run", fake_run)
    state = conditions.gpu_state()
    assert state is not None and state["gpus"][0]["temperature_c"] == 49 and state["utc"]
    assert calls[0][1].startswith("--query-gpu=index,name,")
    monkeypatch.setattr(conditions.subprocess, "run", lambda cmd, **kw: _Done(9, ""))
    assert conditions.gpu_state() is None


def test_run_conditions_is_the_baselines_function(monkeypatch):
    import run_baselines

    monkeypatch.setattr(run_baselines, "run_conditions", lambda: {"on_ac_power": True})
    assert conditions.run_conditions() == {"on_ac_power": True}


def test_profile_conditions_at_start_and_end():
    cpu = iter([{"on_ac_power": True, "n": 1}, {"on_ac_power": True, "n": 2}])
    gpu = iter([{"gpus": [{"temperature_c": 40}]}, {"gpus": [{"temperature_c": 60}]}])
    rec = conditions.ProfileConditions(lambda: next(cpu), lambda: next(gpu))
    with pytest.raises(RuntimeError):
        rec.end()
    rec.start()
    with pytest.raises(RuntimeError):
        rec.fields()
    rec.end()
    got = rec.fields()
    assert got["conditions"] == {"on_ac_power": True, "n": 1}  # Phase 14's key: the start
    assert got["conditions_end"] == {"on_ac_power": True, "n": 2}
    assert got["gpu_state"]["start"]["gpus"][0]["temperature_c"] == 40
    assert got["gpu_state"]["end"]["gpus"][0]["temperature_c"] == 60
    assert set(got["conditions_utc"]) == {"start", "end"}
    no_gpu = conditions.ProfileConditions(lambda: {}, lambda: None)
    no_gpu.start()
    no_gpu.end()
    assert no_gpu.fields()["gpu_state"] == {"start": None, "end": None}


# ------------------------------------------------------------------------------ ONNX pipeline


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    pytest.importorskip("onnxruntime")
    from glypharis.export.bundle import write_bundle

    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, tmp_path / "det_ck.pt")
    return write_bundle(tmp_path / "det_ck.pt", det.spec, rec, tmp_path / "bundle", "0.0.0-test",
                        {"detector": "tiny", "recognizer": "tiny"})  # fmt: skip


def test_load_bundle_onnx_equals_load_bundle_and_records_sessions(bundle):
    from glypharis.efficiency.onnx_build import (
        SessionLineModel,
        SessionProbMap,
        check_sessions,
        load_bundle_onnx,
        sessions_options,
    )
    from glypharis.export.bundle import load_bundle

    ours = load_bundle_onnx(bundle, 2)
    v1 = load_bundle(bundle, "onnx", 2)
    torch_build = load_bundle(bundle, "torch")
    assert isinstance(ours.detector.backend, SessionProbMap)
    assert isinstance(ours.recognizer.model, SessionLineModel)
    assert ours.model_version == v1.model_version == "0.0.0-test"
    assert ours.detector.spec == v1.detector.spec
    for attr in ("height", "max_width", "temperature", "arch", "strict_fp32"):
        assert getattr(ours.recognizer, attr) == getattr(v1.recognizer, attr)
    got = check_sessions(ours, 2)
    assert got == sessions_options(ours)
    assert got["detector"]["intra_op_num_threads"] == got["recognizer"]["intra_op_num_threads"] == 2
    with pytest.raises(threads.ThreadError):
        check_sessions(ours, 3)

    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (96, 160, 3), np.uint8)
    assert np.array_equal(ours.detector.backend(img), v1.detector.backend(img))
    widths = torch.tensor([7, 64, 200])
    assert torch.equal(ours.recognizer.model.frames(widths), v1.recognizer.model.frames(widths))

    lines = [rect_polygon(200, 60, 300, 24), rect_polygon(200, 150, 300, 24)]
    page = _page(lines)
    results = {}
    for name, ocr in (("ours", ours), ("v1", v1), ("torch", torch_build)):
        ocr.detector = type(ocr.detector)(KernelMap(lines, 160 / 400), ocr.detector.spec)
        results[name] = ocr.predict(page)
    texts = {k: [ln.text for ln in r.lines] for k, r in results.items()}
    assert texts["ours"] == texts["v1"] == texts["torch"] and len(texts["ours"]) == 2
    for a, b in zip(results["ours"].lines, results["v1"].lines, strict=True):
        assert a.confidence == b.confidence


def test_mixed_builds_keep_the_settings(bundle):
    from glypharis.efficiency.onnx_build import (
        onnx_detector,
        onnx_recognizer,
        sessions_options,
        with_parts,
    )
    from glypharis.export.bundle import load_bundle

    base = load_bundle(bundle, "torch")
    base.model_version = "v1-test"
    mixed = with_parts(
        base, detector=onnx_detector(bundle / "detector.onnx", base.detector.spec, 2)
    )
    assert mixed.recognizer is base.recognizer and mixed.detector.spec == base.detector.spec
    assert (mixed.crop, mixed.order, mixed.max_pixels, mixed.model_version) == (
        base.crop,
        base.order,
        base.max_pixels,
        "v1-test",
    )
    opts = sessions_options(mixed)
    assert opts["recognizer"] is None and opts["detector"]["intra_op_num_threads"] == 2

    rec = onnx_recognizer(bundle / "recognizer.onnx", base.recognizer, 2)
    rng = np.random.default_rng(1)
    from PIL import Image

    crops = [Image.fromarray(rng.integers(0, 255, (40, w), np.uint8)) for w in (30, 300, 75)]
    for a, b in zip(rec.logits(crops), base.recognizer.logits(crops), strict=True):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-4
    assert sessions_options(with_parts(base, recognizer=rec))["detector"] is None


def test_load_bundle_onnx_checks_hashes(bundle):
    from glypharis.efficiency.onnx_build import load_bundle_onnx
    from glypharis.errors import ModelLoadError

    m = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    m["files"]["detector.onnx"] = "0" * 64
    (bundle / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        load_bundle_onnx(bundle, 2)


def test_process_gpu_memory_is_this_processes_or_none():
    text = "11332, [N/A]\n4242, 300\n4242, 12\n77, 999\n"
    assert conditions.parse_process_memory(text, 4242) == 312
    assert conditions.parse_process_memory(text, 11332) is None  # WDDM: not reported
    assert conditions.parse_process_memory(text, 5) is None  # not listed


def test_atomic_writes_leave_no_partial_file(tmp_path, monkeypatch):
    from glypharis.efficiency import files

    path = tmp_path / "out" / "p.json"
    files.write_json_atomic(path, {"a": "é"})
    assert path.read_text(encoding="utf-8") == '{\n "a": "é"\n}\n'

    def crash(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(files.os, "replace", crash)
    with pytest.raises(OSError, match="disk full"):
        files.write_text_atomic(tmp_path / "out" / "q.json", "x")
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["p.json"]
