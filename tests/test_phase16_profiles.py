"""Phase 16 profilers: `scripts/e2e_profile.py --v1` (the v1 page profile with its stage split) and
`scripts/rec_profile.py --experiment` (line profiles with threads, ONNX, batches and devices).
Tiny models, made-up validation pages and lines in tmp_path only; no real data is read."""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import e2e_profile
import numpy as np
import pytest
import rec_profile
import torch
import yaml
from det_fixtures import rect_polygon
from PIL import Image, ImageDraw
from test_rec_training import WORDS, _pack

from glypharis.config.loader import build_config, save_resolved
from glypharis.config.schema import (
    DetInferConfig,
    DetLabelConfig,
    DetModelConfig,
    DetPostConfig,
    ModelConfig,
)
from glypharis.data.det_labels import LABEL_VERSION
from glypharis.data.pages import Page, write_pages
from glypharis.data.text import Charset
from glypharis.efficiency.threads import THREAD_ENV
from glypharis.efficiency.v1 import V1CodeError
from glypharis.eval.sets import EVAL_SETS
from glypharis.models.det.dbnet import build_detector
from glypharis.models.det.detector import DetectorSpec, TextLineDetector
from glypharis.models.det.runtime import TorchProbMap
from glypharis.models.rec.crnn import build_recognizer
from glypharis.pipeline.frozen import FREEZE_RECORD, Frozen
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer, _padded
from glypharis.training.checkpoint import save_checkpoint

REPO = Path(__file__).resolve().parents[1]
CHARSET = REPO / "data/manifests/charset.json"
PAGE = (400, 300)
LINES = [rect_polygon(200, 60, 300, 24), rect_polygon(200, 150, 260, 24),
         rect_polygon(180, 240, 200, 24)]  # fmt: skip
PROVENANCE = {"name": "glypharis_v1", "record_sha256": "r" * 64,
              "recognizer_checkpoint_sha256": "c" * 64, "detector_checkpoint_sha256": "d" * 64,
              "charset_sha256": "s" * 64, "device_name": "cpu"}  # fmt: skip
V1_CODE = {"tag": "phase14-eval", "tag_commit": "t" * 40, "head": "h" * 40,
           "paths": ["glypharis/pipeline"], "v1_code": "equal"}  # fmt: skip


class KernelMap:
    """A detector backend whose map fires on `LINES` (at the spec's network scale)."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        prob = np.zeros((h, w), np.float32)
        for poly in LINES:
            pts = np.asarray(poly, np.float32) * self.scale
            c = pts.mean(axis=0)
            cv2.fillPoly(prob, [np.round(c + 0.6 * (pts - c)).astype(np.int32)], 0.95)
        return prob


def tiny_recognizer(seed: int = 0) -> TextRecognizer:
    torch.manual_seed(seed)
    charset = Charset.load(CHARSET)
    cfg = ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=16, lstm_layers=1,
                      dropout=0.0)  # fmt: skip
    arch = {"model_name": "crnn", "model": dataclasses.asdict(cfg)}
    return TextRecognizer(build_recognizer(charset.num_classes, cfg), charset, 32, 512,
                          temperature=2.35, arch=arch, strict_fp32=True)  # fmt: skip


def tiny_detector(seed: int = 0) -> TextLineDetector:
    torch.manual_seed(seed)
    mcfg = DetModelConfig(backbone="mobilenet_v3_small", init="scratch", inner_channels=16)
    model = build_detector(mcfg, Path("."), load_pretrained=False)
    spec = DetectorSpec(mcfg, DetLabelConfig(), DetInferConfig(long_side=160), DetPostConfig(),
                        "inverse10", LABEL_VERSION)  # fmt: skip
    return TextLineDetector(TorchProbMap(model, torch.device("cpu")), spec)


def tiny_ocr(order: bool = True) -> GlypharisOCR:
    """A pipeline that finds `LINES` on a `PAGE`-sized page and reads them with a tiny CRNN."""
    det = tiny_detector()
    ocr = GlypharisOCR(TextLineDetector(KernelMap(160 / PAGE[0]), det.spec), tiny_recognizer(),
                       order=order)  # fmt: skip
    ocr.model_version = "glypharis_v1"
    return ocr


def page_image(seed: int) -> Image.Image:
    img = Image.new("RGB", PAGE, (235, 235, 225))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    for poly in LINES:
        d.polygon([tuple(p) for p in poly], fill=tuple(int(v) for v in rng.integers(0, 80, 3)))
    return img


@pytest.fixture
def restore_threads(monkeypatch: pytest.MonkeyPatch) -> Iterator[int]:
    """A thread count this process can be set to (the inter-op pool may already be fixed); the
    thread variables, torch's intra-op count and grad mode are put back afterwards."""
    for key in THREAD_ENV:
        monkeypatch.delenv(key, raising=False)
    threads, grad = torch.get_num_threads(), torch.is_grad_enabled()
    yield torch.get_num_interop_threads()
    torch.set_num_threads(threads)
    torch.set_grad_enabled(grad)


def _stub_machine(module: Any, monkeypatch: pytest.MonkeyPatch, gpu: Any = None) -> None:
    """No powercfg, no CPU sampling, no git and no nvidia-smi; the v1 code check passes."""
    cpu = itertools.cycle([{"cpu_busy_percent_before": 2.0, "at": "start"},
                           {"cpu_busy_percent_before": 4.0, "at": "end"}])  # fmt: skip
    monkeypatch.setattr(module, "run_conditions", lambda: dict(next(cpu)))
    monkeypatch.setattr(module, "gpu_state", lambda: gpu)
    monkeypatch.setattr(module, "cpu_busy_percent", lambda interval=2.0: 3.0)
    monkeypatch.setattr(module, "git_state", lambda: {"git_commit": "g" * 40, "git_dirty": True})
    monkeypatch.setattr(module, "assert_v1_code", lambda root=".": dict(V1_CODE))


# ------------------------------------------------------------------------- the staged pipeline


@pytest.mark.parametrize("order", [True, False])
def test_the_staged_pipeline_reads_what_predict_reads(order):
    ocr = tiny_ocr(order)
    img = page_image(0)
    staged, ms = e2e_profile.staged_predict(ocr, img)
    plain = ocr.predict(np.asarray(img.convert("RGB")))
    assert len(plain.lines) == len(LINES)
    assert [ln.to_dict() for ln in staged.lines] == [ln.to_dict() for ln in plain.lines]
    assert (staged.width, staged.height, staged.model_version) == (
        plain.width, plain.height, "glypharis_v1")  # fmt: skip
    assert list(ms) == list(e2e_profile.STAGES)
    assert all(v >= 0 for v in ms.values()) and ms["rec_network"] > 0


def test_the_staged_pipeline_on_a_page_without_lines():
    det = tiny_detector()
    ocr = GlypharisOCR(TextLineDetector(lambda image: np.zeros(image.shape[:2], np.float32),
                                        det.spec), tiny_recognizer())  # fmt: skip
    staged, ms = e2e_profile.staged_predict(ocr, page_image(1))
    assert staged.lines == [] and ocr.predict(np.asarray(page_image(1))).lines == []
    assert list(ms) == list(e2e_profile.STAGES)


def test_the_staged_pipeline_reads_the_real_detector_and_long_lines_alike():
    """A real (tiny, random) detector network, and crops wide enough to make several batches."""
    ocr = GlypharisOCR(tiny_detector(), tiny_recognizer())
    img = page_image(2)
    staged, _ = e2e_profile.staged_predict(ocr, img)
    plain = ocr.predict(np.asarray(img))
    assert [ln.to_dict() for ln in staged.lines] == [ln.to_dict() for ln in plain.lines]


def test_every_stage_boundary_is_synchronised():
    calls: list[int] = []
    e2e_profile.staged_predict(tiny_ocr(), page_image(0), sync=lambda: calls.append(1))
    # once at the start, once per stage, once more per recognizer batch (one here)
    assert len(calls) == 1 + len(e2e_profile.STAGES) + 1


# --------------------------------------------------------------------------- e2e --v1 profiles


@pytest.fixture
def pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_threads: int) -> SimpleNamespace:
    """A repository-like cwd with three validation pages, the profile shortened to 1 + 2 and the
    machine probes stubbed; ``loads`` lists the calls to the (stubbed) frozen-model builder."""
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    rows = []
    for i in range(3):
        page_image(i).save(data / f"raw/p{i}.png")
        rows.append(Page(f"p{i}", f"raw/p{i}.png", *PAGE, "hiertext", "validation", []))
    write_pages(data / EVAL_SETS["hiertext_val"]["pages"], rows)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(e2e_profile, "WARMUP", 1)
    monkeypatch.setattr(e2e_profile, "PROFILE_SAMPLES", 2)
    _stub_machine(e2e_profile, monkeypatch)
    calls: list[tuple[Any, ...]] = []

    def fake_load(*args: Any, **kwargs: Any) -> Frozen:
        calls.append((args, kwargs))
        ocr = tiny_ocr()
        return Frozen(ocr, ocr.recognizer, dict(PROVENANCE))

    monkeypatch.setattr(e2e_profile, "load_frozen", fake_load)
    return SimpleNamespace(root=tmp_path, loads=calls, threads=restore_threads)


def _v1(*extra: str, n: int | str, out: Path, backend: str = "torch") -> list[str]:
    return ["--v1", "--backend", backend, "--threads", str(n), "--experiment", "GW-EFF-007",
            "--out", str(out), *extra]  # fmt: skip


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_v1_torch_profile_records_times_stages_threads_and_conditions(pages):
    n = pages.threads
    out = pages.root / "reports/p16/e2e_torch.json"
    assert e2e_profile.main(_v1(n=n, out=out)) == 0
    assert pages.loads == [(("cpu",), {})]
    p = _read(out)
    assert (p["system"], p["task"], p["eval_set"], p["experiment"]) == (
        "glypharis", "e2e", "hiertext_val", "GW-EFF-007")  # fmt: skip
    assert (p["build"], p["backend"], p["device"], p["version"]) == ("v1", "torch", "cpu",
                                                                     "glypharis_v1")  # fmt: skip
    s = p["settings"]
    assert s["cpu_threads"] == s["intra_op_threads"] == s["inter_op_threads"] == n
    assert s["recognizer"] == {"input": {"height": 32, "max_width": 512}, "temperature": 2.35,
                               "strict_fp32": True}  # fmt: skip
    assert (s["backend"], s["precision"]) == ("torch", "fp32") and s["runtime"].startswith("torch")
    t = p["threads"]
    assert (t["requested"], t["intra_op"], t["inter_op"]) == (n, n, n)
    assert t["env"] == {k: str(n) for k in THREAD_ENV}
    assert t["env_before_torch_import"] is False and "torch" in t["preloaded"]
    for when in ("before", "after"):
        assert p["threads_check"][when] == {"intra_op": n, "inter_op": n,
                                            "env": {k: str(n) for k in THREAD_ENV}}  # fmt: skip
    assert p["session_options"] == {"detector": None, "recognizer": None}
    assert (p["batch"], p["warmup"]) == (1, 1)
    assert len(p["warmup_times_ms"]) == 1 and len(p["times_ms"]) == 2
    assert p["stages"] == list(e2e_profile.STAGES)
    assert len(p["warmup_stages_ms"]) == 1 and len(p["stages_ms"]) == 2
    for total, parts in zip(p["times_ms"], p["stages_ms"], strict=True):
        assert list(parts) == p["stages"]
        assert sum(parts.values()) == pytest.approx(total, rel=0.01)
    assert 0 <= p["stage_sum_max_rel_diff"] <= 0.01
    assert p["mean_stages_ms"]["rec_network"] == pytest.approx(
        np.mean([s["rec_network"] for s in p["stages_ms"]]))  # fmt: skip
    assert p["lines"] == [3, 3] and p["mean_lines"] == 3.0
    assert p["input_sizes"] == [list(PAGE), list(PAGE)] and p["page_ids"] == ["p1", "p2"]
    assert p["mean_ms"] == pytest.approx(sum(p["times_ms"]) / 2)
    assert p["p95_ms"] == pytest.approx(float(np.percentile(p["times_ms"], 95)))
    assert p["peak_mem_mib"] > 0 and p["peak_vram"] is None
    check = p["staged_check"]
    assert {k: check[k] for k in ("page_id", "lines", "same_as_predict")} == {
        "page_id": "p1", "lines": 3, "same_as_predict": True}  # fmt: skip
    assert check["staged_ms"] > 0 and check["predict_ms"] > 0
    # The second clock around each call, and the image file's decoding outside the page time.
    assert len(p["call_ms"]) == len(p["file_decode_ms"]) == 2
    for total, wall in zip(p["times_ms"], p["call_ms"], strict=True):
        assert wall >= total
    assert all(v > 0 for v in p["file_decode_ms"])
    assert p["mean_file_decode_ms"] == pytest.approx(np.mean(p["file_decode_ms"]))
    assert p["stages"][0] == "to_rgb" and "decode" not in p["stages"]
    assert p["provenance"] == PROVENANCE and p["bundle"] is None and p["v1_code"] == V1_CODE
    assert p["conditions"] == {"cpu_busy_percent_before": 2.0, "at": "start"}
    assert p["conditions_end"] == {"cpu_busy_percent_before": 4.0, "at": "end"}
    assert set(p["conditions_utc"]) == {"start", "end"}
    assert p["gpu_state"] == {"start": None, "end": None}
    assert p["started_utc"] <= p["ended_utc"] and p["started_utc"].endswith("+00:00")
    assert p["cpu_busy_percent_after"] == 3.0 and p["git_commit"] == "g" * 40
    text = out.read_text(encoding="utf-8")
    assert str(pages.root) not in text and pages.root.as_posix() not in text
    assert torch.is_grad_enabled() is False


def test_phase_14s_scorer_reads_the_v1_profiles_thread_fields(pages):
    """`phase14_score.profile_problems` (reused by the Phase 16 scorer) finds the thread count, the
    torch pools and the batch and counts where it looks for them."""
    import phase14_score as ps

    n = pages.threads
    out = pages.root / "p.json"
    assert e2e_profile.main(_v1(n=n, out=out)) == 0
    prof = _read(out)
    prof["threads"]["env_before_torch_import"] = True  # set by a command-line start only
    problems = ps.profile_problems("v1", prof, {}, threads=n)
    assert not [p for p in problems if "thread" in p or "pool" in p]
    assert any("timed pages" in p for p in problems)  # 1 + 2 pages here, not 10 + 100


def _record(ocr: GlypharisOCR, **change: Any) -> dict[str, Any]:
    got = Frozen(ocr, ocr.recognizer, {}).settings()
    rec = {"input": got["recognizer"]["input"], "temperature": got["recognizer"]["temperature"]}
    rec.update(change)
    return {"name": "glypharis_v1", "detector": got["detector"], "recognizer": rec}


@pytest.fixture
def bundle(pages: SimpleNamespace) -> Path:
    """The tiny pipeline as a bundle under the cwd, and a freeze record that matches it."""
    pytest.importorskip("onnxruntime")
    from glypharis.export.bundle import load_bundle, write_bundle

    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, pages.root / "det_ck.pt")
    path = write_bundle(pages.root / "det_ck.pt", det.spec, rec, pages.root / "bundle", "1.0.0",
                        {"detector": "tiny", "recognizer": "tiny"})  # fmt: skip
    FREEZE_RECORD.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_RECORD.write_text(yaml.safe_dump(_record(load_bundle(path))), encoding="utf-8")
    return path


def test_the_v1_onnx_profile_records_the_bundle_and_its_sessions(pages, bundle):
    import hashlib

    n = pages.threads
    out = pages.root / "e2e_onnx.json"
    assert e2e_profile.main(_v1("--bundle", str(bundle), n=n, out=out, backend="onnx")) == 0
    assert pages.loads == []  # ONNX is built from the bundle, not the freeze record's checkpoints
    p = _read(out)
    assert (p["backend"], p["settings"]["backend"], p["version"]) == ("onnx", "onnx", "1.0.0")
    assert p["settings"]["runtime"].startswith("onnxruntime")
    for part in ("detector", "recognizer"):
        o = p["session_options"][part]
        assert o["intra_op_num_threads"] == n and o["providers"] == ["CPUExecutionProvider"]
        assert (o["execution_mode"], o["enable_cpu_mem_arena"], o["allow_spinning"]) == (
            "ORT_SEQUENTIAL", False, "default")  # fmt: skip
    manifest = (bundle / "manifest.json").read_bytes()
    b = p["bundle"]
    assert b["path"] == "bundle" and b["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert b["files"] == json.loads(manifest)["files"] and b["record_settings"] == "equal"
    assert p["provenance"]["record"] == FREEZE_RECORD.as_posix()
    assert p["provenance"]["bundle_manifest_sha256"] == b["manifest_sha256"]
    assert len(p["times_ms"]) == 2 and p["staged_check"]["same_as_predict"] is True
    for total, parts in zip(p["times_ms"], p["stages_ms"], strict=True):
        assert sum(parts.values()) == pytest.approx(total, rel=0.01)


def test_the_v1_onnx_profile_at_the_default_threads(pages, bundle):
    out = pages.root / "e2e_onnx_default.json"
    argv = _v1("--bundle", str(bundle), n="default", out=out, backend="onnx")
    assert e2e_profile.main(argv) == 0
    p = _read(out)
    assert p["threads"]["requested"] == "default" and p["settings"]["cpu_threads"] is None
    assert p["threads"]["env"] == {k: None for k in THREAD_ENV}
    assert p["session_options"]["recognizer"]["intra_op_num_threads"] == 0


def test_a_bundle_that_is_not_the_records_is_refused(pages, bundle):
    from glypharis.export.bundle import load_bundle

    FREEZE_RECORD.write_text(yaml.safe_dump(_record(load_bundle(bundle), temperature=1.0)),
                             encoding="utf-8")  # fmt: skip
    out = pages.root / "x.json"
    with pytest.raises(SystemExit, match="not the v1 build: recognizer temperature"):
        e2e_profile.main(_v1("--bundle", str(bundle), n=pages.threads, out=out, backend="onnx"))
    assert not out.exists()


def test_a_damaged_bundle_is_refused(pages, bundle):
    from glypharis.errors import ModelLoadError

    (bundle / "recognizer.onnx").write_bytes(b"not a model")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        e2e_profile.main(_v1("--bundle", str(bundle), n=pages.threads, out=pages.root / "x.json",
                             backend="onnx"))  # fmt: skip


def test_changed_v1_code_is_refused_before_the_model_loads(pages, monkeypatch):
    def changed(root: str = ".") -> dict[str, Any]:
        raise V1CodeError("v1 code differs from phase14-eval: glypharis/models/rec/svtr.py")

    monkeypatch.setattr(e2e_profile, "assert_v1_code", changed)
    out = pages.root / "x.json"
    with pytest.raises(SystemExit, match="differs from phase14-eval"):
        e2e_profile.main(_v1(n=pages.threads, out=out))
    assert pages.loads == [] and not out.exists()


def test_stages_that_do_not_sum_to_the_page_time_are_refused(pages, monkeypatch):
    staged = e2e_profile.staged_predict

    def padded(*args: Any, **kwargs: Any) -> Any:
        res, ms = staged(*args, **kwargs)
        return res, {**ms, "to_rgb": ms["to_rgb"] + 1000.0}

    monkeypatch.setattr(e2e_profile, "staged_predict", padded)
    out = pages.root / "x.json"
    with pytest.raises(SystemExit, match="more than 1% and 5 ms apart"):
        e2e_profile.main(_v1(n=pages.threads, out=out))
    assert not out.exists()


def test_time_outside_the_stages_clock_is_caught(pages, monkeypatch):
    """The check reads a second clock: time the stages' clock never sees fails it too."""
    staged = e2e_profile.staged_predict

    def slow_return(*args: Any, **kwargs: Any) -> Any:
        got = staged(*args, **kwargs)
        time.sleep(0.2)  # after the stages' last reading
        return got

    monkeypatch.setattr(e2e_profile, "staged_predict", slow_return)
    with pytest.raises(SystemExit, match="the call took"):
        e2e_profile.main(_v1(n=pages.threads, out=pages.root / "x.json"))


def test_a_staged_reading_that_differs_from_predict_is_refused(pages, monkeypatch):
    ocr = tiny_ocr()
    real = ocr.predict

    def other(image: Any, temperatures: Any = ()) -> Any:
        res = real(image, temperatures)
        res.lines[0].text += "x"
        return res

    monkeypatch.setattr(ocr, "predict", other)
    page = Page("p0", "raw/p0.png", *PAGE, "hiertext", "validation", [])
    with pytest.raises(SystemExit, match=r"other lines than GlypharisOCR\.predict"):
        e2e_profile.staged_check(ocr, page, pages.root / "data")


def test_threads_changed_during_the_run_are_refused(pages, monkeypatch):
    n = pages.threads
    staged = e2e_profile.staged_predict

    def resetting(*args: Any, **kwargs: Any) -> Any:
        torch.set_num_threads(n + 1)
        return staged(*args, **kwargs)

    monkeypatch.setattr(e2e_profile, "staged_predict", resetting)
    out = pages.root / "x.json"
    with pytest.raises(SystemExit, match="intra-op"):
        e2e_profile.main(_v1(n=n, out=out))
    assert not out.exists()


def test_only_validation_pages_are_profiled(pages):
    write_pages(pages.root / "data" / EVAL_SETS["hiertext_val"]["pages"],
                [Page("p0", "raw/p0.png", *PAGE, "hiertext", "test", [])])  # fmt: skip
    with pytest.raises(SystemExit, match="validation pages only"):
        e2e_profile.main(_v1(n=pages.threads, out=pages.root / "x.json"))


def test_too_few_pages_are_refused(pages, monkeypatch):
    monkeypatch.setattr(e2e_profile, "PROFILE_SAMPLES", 3)
    with pytest.raises(SystemExit, match="gave 3 pages; need 4"):
        e2e_profile.main(_v1(n=pages.threads, out=pages.root / "x.json"))


def test_an_existing_v1_profile_is_never_overwritten(pages):
    out = pages.root / "x.json"
    out.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists: move it aside"):
        e2e_profile.main(_v1(n=pages.threads, out=out))
    assert out.read_text(encoding="utf-8") == "{}\n" and pages.loads == []


@pytest.mark.parametrize(
    ("argv", "n"),
    [
        (["--backend", "onnx"], 4),  # no bundle
        (["--backend", "torch", "--bundle", "b"], 4),
        (["--backend", "torch"], "default"),  # default threads: ONNX only
        (["--backend", "onnx", "--bundle", "b", "--device", "cuda"], 4),
        (["--backend", "torch"], 0),
        (["--backend", "torch"], "x"),
    ],
)
def test_inconsistent_v1_command_lines_are_refused(pages, argv, n):
    full = ["--v1", *argv, "--threads", str(n), "--experiment", "E", "--out", "x.json"]
    with pytest.raises(SystemExit) as exc:
        e2e_profile.main(full)
    assert exc.value.code == 2 and pages.loads == []


def test_frozen_and_v1_together_are_refused(pages):
    with pytest.raises(SystemExit, match="separate runs"):
        e2e_profile.main(["--frozen", *_v1(n=pages.threads, out=pages.root / "x.json")])


@pytest.mark.gpu
def test_the_v1_profile_on_cuda_synchronises_and_records_vram(pages, monkeypatch):
    def cuda_load(device: str) -> Frozen:
        assert device == "cuda"
        det = tiny_detector()
        rec = tiny_recognizer()
        rec = TextRecognizer(rec.model, rec.charset, 32, 512, "cuda", 2.35, rec.arch, True)
        backend = TorchProbMap(det.backend.model, torch.device("cuda"))
        ocr = GlypharisOCR(TextLineDetector(backend, det.spec), rec)
        return Frozen(ocr, rec, dict(PROVENANCE))

    monkeypatch.setattr(e2e_profile, "load_frozen", cuda_load)
    gpu = {"utc": "t", "gpus": [{"memory_used_mib": 512}]}
    monkeypatch.setattr(e2e_profile, "gpu_state", lambda: gpu)
    monkeypatch.setattr(e2e_profile, "gpu_process_memory_mib", lambda pid: 200)
    out = pages.root / "e2e_cuda.json"
    assert e2e_profile.main(_v1("--device", "cuda", n=pages.threads, out=out)) == 0
    p = _read(out)
    assert p["device"] == p["settings"]["device"] == "cuda"
    v = p["peak_vram"]
    assert v["max_memory_allocated_mib"] > 0 and v["nvidia_smi_whole_gpu_used_mib_end"] == 512
    assert v["nvidia_smi_process_mib_end"] == 200
    assert p["gpu_state"] == {"start": gpu, "end": gpu}


def test_a_v1_command_line_run_sets_the_threads_before_torch_loads(tmp_path):
    """Run as a script, ``--v1 --threads 1`` reaches the variables and both torch pools before
    torch is imported; the run then stops because its output exists, before any model, page or git
    command is touched."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("one core: one thread is torch's default anyway")
    out = tmp_path / "exists.json"
    out.write_text("{}\n", encoding="utf-8")
    script = REPO / "scripts/e2e_profile.py"
    code = (
        "import os, runpy, sys\n"
        f"sys.argv = [{str(script)!r}, '--v1', '--backend', 'torch', '--threads', '1',"
        f" '--experiment', 'E', '--out', {str(out)!r}]\n"
        "try:\n"
        f"    g = runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    print('exit:', exc)\n"
        "import torch\n"
        "print('env:', [os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',"
        " 'MKL_NUM_THREADS')])\n"
        "print('threads:', torch.get_num_threads(), torch.get_num_interop_threads())\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "scripts")])
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert "exists: move it aside" in proc.stdout
    assert "env: ['1', '1', '1']" in proc.stdout
    assert "threads: 1 1" in proc.stdout


# ------------------------------------------------------------------ rec_profile --experiment

N_LINES = 100  # enough for 1 + 2 items of 32 lines


@pytest.fixture
def rec_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_threads: int
) -> SimpleNamespace:
    """A trained-run directory (resolved config, a checkpoint and its summary), its ONNX export,
    `N_LINES` made-up validation crops, the profile shortened to 1 + 2 items and the machine
    probes stubbed. The cwd is tmp_path."""
    data = tmp_path / "data"
    Charset(list(" abcdefghijklmnopqrstuvwxyz")).save(data / "manifests/charset.json")
    rng = np.random.default_rng(0)
    lines = [(f"hv{i}", f"hdoc{i // 4}", " ".join(rng.choice(WORDS, 1 + i % 3)))
             for i in range(N_LINES)]  # fmt: skip
    _pack(data / "processed/hiertext/lines_validation_eval", lines, "validation")
    cfg = build_config({
        "experiment_id": "GW-REC-999", "task": "rec", "model_name": "crnn", "device": "cpu",
        "paths": {"data_root": str(data), "runs_dir": str(tmp_path / "runs"),
                  "reports_dir": str(tmp_path / "reports")},
        "data": {"train": ["processed/x"], "val": ["processed/y"], "max_width": 256},
        "model": {"cnn_channels": [8, 8, 16, 16, 16, 16], "lstm_hidden": 16, "dropout": 0.0},
        "checkpoint": {"dir": str(tmp_path / "ckpt")},
    }, [])  # fmt: skip
    run = tmp_path / "runs/GW-REC-999"
    save_resolved(cfg, run / "config.resolved.yaml")
    torch.manual_seed(0)
    charset = Charset.load(data / "manifests/charset.json")
    model = build_recognizer(charset.num_classes, cfg.model, "crnn").eval()
    save_checkpoint(tmp_path / "ckpt/best.pt", {"model": model.state_dict()})
    (run / "summary.json").write_text(json.dumps({"parameters": 1, "model_bytes": 1,
                                                  "best_epoch": 1}), encoding="utf-8")  # fmt: skip
    onnx_path = None
    try:
        import onnxruntime  # noqa: F401

        from glypharis.export.onnx import export_recognizer_onnx

        onnx_path = export_recognizer_onnx(model, cfg.data.image_height, tmp_path / "rec.onnx")
    except ImportError:
        pass
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rec_profile, "WARMUP", 1)
    monkeypatch.setattr(rec_profile, "PROFILE_SAMPLES", 2)
    _stub_machine(rec_profile, monkeypatch)
    return SimpleNamespace(root=tmp_path, run=run, ck=tmp_path / "ckpt/best.pt",
                           onnx=onnx_path, model=model, charset=charset, cfg=cfg,
                           threads=restore_threads)  # fmt: skip


def _rec(r: SimpleNamespace, out: Path, *extra: str, n: int | str | None = None) -> list[str]:
    return ["--experiment", "GW-EFF-011", "--run", str(r.run), "--threads",
            str(r.threads if n is None else n), "--out-json", str(out), *extra]  # fmt: skip


def _crops(seed: int, widths: tuple[int, ...]) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    return [Image.fromarray(rng.integers(0, 255, (24, w), np.uint8)) for w in widths]


def test_read_batch_at_batch_1_with_phase7_padding_is_read_line():
    """The same input tensor as Phase 7's `read_line`, so the same text."""
    rec = tiny_recognizer()
    seen: list[torch.Tensor] = []
    rec.model.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    for crop in _crops(0, (5, 40, 64, 300, 900)):
        text = rec_profile.read_line(rec.model, rec.charset, crop, rec.height, rec.max_width)
        assert rec_profile.read_batch(rec, [crop], "phase7") == [text]
        assert torch.equal(seen[-1], seen[-2])


def test_read_batch_with_pipeline_padding_reads_as_the_recognizer():
    rec = tiny_recognizer()
    crops = _crops(1, (30, 300, 75, 120, 64))
    seen: list[torch.Tensor] = []
    rec.model.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    got = rec_profile.read_batch(rec, crops, "pipeline")
    assert got == [r.text for r in rec.read(crops)]
    widths = [rec_profile.prepare_crop(c, rec.height, rec.max_width).shape[1] for c in crops]
    assert seen[0].shape == (5, 1, 32, max(_padded(w) for w in widths))


def test_the_torch_line_profile_records_threads_numerics_and_times(rec_run):
    r = rec_run
    out = r.root / "lines_torch.json"
    assert rec_profile.main(_rec(r, out, "--checkpoint", str(r.ck))) == 0
    p = _read(out)
    n = r.threads
    assert (p["experiment"], p["run"], p["model_name"], p["eval_set"]) == (
        "GW-EFF-011", "GW-REC-999", "crnn", "hiertext_val")  # fmt: skip
    assert (p["backend"], p["device"], p["precision"], p["batch"], p["padding"]) == (
        "torch", "cpu", "fp32", 1, "pipeline")  # fmt: skip
    assert p["batching"].startswith("TextRecognizer.read")
    assert (p["image_height"], p["max_width"], p["frame_stride"]) == (32, 256, 4)
    assert p["parameters"] == sum(v.numel() for v in r.model.parameters())
    assert p["model_file"]["kind"] == "checkpoint" and p["model_file"]["name"] == "best.pt"
    assert p["settings"] == {"cpu_threads": n, "intra_op_threads": n, "inter_op_threads": n}
    assert p["threads"]["requested"] == n and p["threads"]["env"] == {k: str(n) for k in THREAD_ENV}
    assert p["threads_check"]["after"]["intra_op"] == n and p["session_options"] is None
    assert p["numerics"]["recognizer_strict_fp32"] is True
    assert len(p["warmup_times_ms"]) == 1 and len(p["times_ms"]) == 2 and p["warmup"] == 1
    assert p["per_line_ms"] == pytest.approx(p["times_ms"], abs=1e-3)
    assert len(p["input_sizes"]) == 2 and len(p["input_sizes"][0]) == 2  # [w, h] per line
    assert p["mean_ms"] == pytest.approx(np.mean(p["times_ms"]))
    assert p["lines_per_second"] == pytest.approx(2 / (sum(p["times_ms"]) / 1000))
    assert p["peak_mem_mib"] > 0 and p["peak_vram"] is None and p["v1_code"] == V1_CODE
    assert p["conditions"]["at"] == "start" and p["conditions_end"]["at"] == "end"
    assert p["gpu_state"] == {"start": None, "end": None}
    assert p["started_utc"] <= p["ended_utc"]
    text = out.read_text(encoding="utf-8")
    assert str(r.root) not in text and r.root.as_posix() not in text


def test_a_run_is_found_by_its_id_under_runs(rec_run):
    r = rec_run
    out = r.root / "by_id.json"
    argv = _rec(r, out, "--runs", str(r.root / "runs"))
    argv[argv.index(str(r.run))] = "GW-REC-999"
    assert rec_profile.main(argv) == 0
    assert _read(out)["model_file"]["name"] == "best.pt"


def test_a_batch_of_32_is_padded_as_the_pipeline_pads(rec_run):
    r = rec_run
    out = r.root / "lines_b32.json"
    assert rec_profile.main(_rec(r, out, "--batch", "32")) == 0
    p = _read(out)
    assert (p["batch"], p["padding"]) == (32, "pipeline") and len(p["times_ms"]) == 2
    assert all(len(item) == 32 for item in p["input_sizes"])
    assert p["per_line_ms"] == pytest.approx([t / 32 for t in p["times_ms"]], abs=1e-3)
    assert p["lines_per_second"] == pytest.approx(64 / (sum(p["times_ms"]) / 1000))
    with pytest.raises(SystemExit) as exc:  # a batch is read as the pipeline reads it
        rec_profile.main(_rec(r, r.root / "y.json", "--batch", "32", "--padding", "phase7"))
    assert exc.value.code == 2


def test_a_batch_is_sorted_and_split_as_the_pipeline_batches_it():
    """32 lines as one item: sorted by width and split at the column cap, never one tensor of 32
    lines padded to the widest."""
    from glypharis.pipeline.recognizer import MAX_BATCH_COLUMNS

    rec = tiny_recognizer()
    rec = TextRecognizer(rec.model, rec.charset, rec.height, 1536, "cpu", 1.0, rec.arch)
    crops = _crops(3, tuple(40 + 47 * (i % 9) * (1 + i % 4) for i in range(32)))
    seen: list[torch.Tensor] = []
    rec.model.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    got = rec_profile.read_batch(rec, crops, "pipeline")
    assert got == [rd.text for rd in rec.read(crops)]
    seen = seen[: len(seen) // 2]  # the reading inside read_batch, not the check above
    assert len(seen) > 1 and all(x.shape[0] * x.shape[3] <= MAX_BATCH_COLUMNS for x in seen)
    widths = [x.shape[3] for x in seen]
    assert widths == sorted(widths)


def test_too_few_lines_for_the_batches_are_refused(rec_run, monkeypatch):
    monkeypatch.setattr(rec_profile, "PROFILE_SAMPLES", 3)  # 4 x 32 > 100 lines
    with pytest.raises(SystemExit, match="gave 3 items of 32 lines"):
        rec_profile.main(_rec(rec_run, rec_run.root / "x.json", "--batch", "32", "--padding",
                              "pipeline"))  # fmt: skip


def test_the_onnx_line_profile_reads_as_pytorch_and_records_its_session(rec_run):
    r = rec_run
    if r.onnx is None:
        pytest.skip("no onnxruntime")
    out = r.root / "lines_onnx.json"
    assert rec_profile.main(_rec(r, out, "--backend", "onnx", "--onnx", str(r.onnx))) == 0
    p = _read(out)
    assert (p["backend"], p["precision"], p["int8_ops"], p["numerics"]) == ("onnx", "fp32", [],
                                                                            None)  # fmt: skip
    assert p["model_file"]["kind"] == "onnx" and p["model_file"]["name"] == "rec.onnx"
    assert p["parameters"] == sum(v.numel() for v in r.model.parameters())  # from the meta build
    o = p["session_options"]
    assert o["intra_op_num_threads"] == r.threads and o["execution_mode"] == "ORT_SEQUENTIAL"
    assert p["runtime"].startswith("onnxruntime")

    from glypharis.efficiency.onnx_build import SessionLineModel
    from glypharis.efficiency.threads import ort_session

    torch_rec = TextRecognizer(r.model, r.charset, 32, 256)
    onnx_rec = TextRecognizer(SessionLineModel(ort_session(r.onnx, 1), 4, r.charset.num_classes),
                              r.charset, 32, 256)  # fmt: skip
    crops = _crops(2, (20, 90, 250))
    for padding in rec_profile.PADDINGS:
        assert rec_profile.read_batch(onnx_rec, crops, padding) == rec_profile.read_batch(
            torch_rec, crops, padding)  # fmt: skip


def test_the_onnx_line_profile_at_the_default_threads(rec_run):
    r = rec_run
    if r.onnx is None:
        pytest.skip("no onnxruntime")
    out = r.root / "lines_default.json"
    assert rec_profile.main(_rec(r, out, "--backend", "onnx", "--onnx", str(r.onnx),
                                 n="default")) == 0  # fmt: skip
    p = _read(out)
    assert p["threads"]["requested"] == "default" and p["settings"]["cpu_threads"] is None
    assert p["session_options"]["intra_op_num_threads"] == 0


def test_an_int8_model_is_labelled_by_its_operators(rec_run):
    r = rec_run
    if r.onnx is None:
        pytest.skip("no onnxruntime")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = r.root / "rec_int8.onnx"
    quantize_dynamic(str(r.onnx), str(int8), op_types_to_quantize=["MatMul"],
                     weight_type=QuantType.QInt8)  # fmt: skip
    assert rec_profile.onnx_int8_ops(int8) and rec_profile.onnx_int8_ops(r.onnx) == []
    out = r.root / "lines_int8.json"
    args = ("--backend", "onnx", "--onnx", str(int8))
    assert rec_profile.main(_rec(r, out, *args, "--precision", "int8")) == 0
    p = _read(out)
    assert p["precision"] == "int8" and p["int8_ops"] == rec_profile.onnx_int8_ops(int8)
    with pytest.raises(SystemExit, match="quantised operators"):  # an INT8 file called fp32
        rec_profile.main(_rec(r, r.root / "y.json", *args))
    with pytest.raises(SystemExit, match="quantised operators"):  # an fp32 file called INT8
        rec_profile.main(_rec(r, r.root / "z.json", "--backend", "onnx", "--onnx", str(r.onnx),
                              "--precision", "int8"))  # fmt: skip


@pytest.mark.parametrize(
    "extra",
    [
        ["--backend", "onnx"],  # no model file
        ["--onnx", "m.onnx"],  # an ONNX file for the torch backend
        ["--backend", "onnx", "--onnx", "m.onnx", "--checkpoint", "c.pt"],
        ["--backend", "onnx", "--onnx", "m.onnx", "--device", "cuda"],
        ["--precision", "int8"],  # INT8 is an ONNX model
        ["--batch", "8"],
        ["--table"],
    ],
)
def test_inconsistent_phase16_command_lines_are_refused(rec_run, extra):
    with pytest.raises(SystemExit) as exc:
        rec_profile.main(_rec(rec_run, rec_run.root / "x.json", *extra))
    assert exc.value.code == 2


def test_default_threads_are_for_onnx_only_and_every_run_needs_its_file(rec_run):
    r = rec_run
    with pytest.raises(SystemExit) as exc:
        rec_profile.main(_rec(r, r.root / "x.json", n="default"))
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        rec_profile.main(["--experiment", "E", "--run", str(r.run), "--threads", "2"])
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", [["--threads", "4"], ["--backend", "onnx"], ["--batch", "32"],
                                  ["--out-json", "x.json"], ["--device", "cpu"]])  # fmt: skip
def test_phase16_flags_need_an_experiment(rec_run, flag):
    with pytest.raises(SystemExit) as exc:
        rec_profile.main(["--run", "GW-REC-999", *flag])
    assert exc.value.code == 2


def test_an_existing_line_profile_is_never_overwritten(rec_run):
    out = rec_run.root / "x.json"
    out.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists: move it aside"):
        rec_profile.main(_rec(rec_run, out))
    assert out.read_text(encoding="utf-8") == "{}\n"


def test_the_line_profile_refuses_changed_v1_code(rec_run, monkeypatch):
    def changed(root: str = ".") -> dict[str, Any]:
        raise V1CodeError("untracked files under the v1 paths: configs/x.yaml")

    monkeypatch.setattr(rec_profile, "assert_v1_code", changed)
    out = rec_run.root / "x.json"
    with pytest.raises(SystemExit, match="untracked files"):
        rec_profile.main(_rec(rec_run, out))
    assert not out.exists()


@pytest.mark.gpu
def test_the_line_profile_on_cuda_records_vram(rec_run, monkeypatch):
    gpu = {"utc": "t", "gpus": [{"memory_used_mib": 300}]}
    monkeypatch.setattr(rec_profile, "gpu_state", lambda: gpu)
    monkeypatch.setattr(rec_profile, "gpu_process_memory_mib", lambda pid: None)
    out = rec_run.root / "lines_cuda.json"
    assert rec_profile.main(_rec(rec_run, out, "--device", "cuda", "--batch", "32",
                                 "--padding", "pipeline")) == 0  # fmt: skip
    p = _read(out)
    assert p["device"] == "cuda" and p["runtime"].startswith("torch")
    assert p["peak_vram"]["max_memory_allocated_mib"] > 0
    assert p["peak_vram"]["nvidia_smi_whole_gpu_used_mib_end"] == 300
    assert p["peak_vram"]["nvidia_smi_process_mib_end"] is None


def test_a_line_profile_command_line_sets_the_threads_before_torch_loads(tmp_path):
    if (os.cpu_count() or 1) < 2:
        pytest.skip("one core: one thread is torch's default anyway")
    out = tmp_path / "exists.json"
    out.write_text("{}\n", encoding="utf-8")
    script = REPO / "scripts/rec_profile.py"
    code = (
        "import os, runpy, sys\n"
        f"sys.argv = [{str(script)!r}, '--experiment', 'E', '--run', 'R', '--threads', '1',"
        f" '--out-json', {str(out)!r}]\n"
        "try:\n"
        f"    runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    print('exit:', exc)\n"
        "import torch\n"
        "print('env:', [os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',"
        " 'MKL_NUM_THREADS')])\n"
        "print('threads:', torch.get_num_threads(), torch.get_num_interop_threads())\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "scripts")])
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert "exists: move it aside" in proc.stdout
    assert "env: ['1', '1', '1']" in proc.stdout
    assert "threads: 1 1" in proc.stdout
