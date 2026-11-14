"""Phase 14 page profiler (GW-EFF-005): built from the frozen record, threads set and asserted,
pinned-code refusal, one file per thread count, and the earlier command line still working.
Tiny models and made-up validation pages in tmp_path only; no real data is read."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import e2e_profile
import pytest
import run_baselines
import torch
from PIL import Image

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
from glypharis.eval.sets import EVAL_SETS, PHASE5_HASHES
from glypharis.models.det.dbnet import build_detector
from glypharis.models.det.detector import DetectorSpec, TextLineDetector
from glypharis.models.det.runtime import TorchProbMap
from glypharis.models.rec.crnn import build_recognizer
from glypharis.pipeline.frozen import Frozen
from glypharis.pipeline.ocr import GlypharisOCR
from glypharis.pipeline.recognizer import TextRecognizer

REPO = Path(__file__).resolve().parents[1]
SIZES = [(320, 200), (300, 240), (360, 180)]  # warm-up page first, then the two timed pages
PROVENANCE = {
    "name": "glypharis_v1",
    "record": "configs/final/glypharis_v1.yaml",
    "recognizer_checkpoint_sha256": "c" * 64,
    "detector_checkpoint_sha256": "d" * 64,
}
PINNED = {"tag": "phase14-eval", "tag_commit": "t" * 40, "head": "t" * 40, "trees": "equal"}


def tiny_frozen() -> Frozen:
    torch.manual_seed(0)
    charset = Charset.load(REPO / "data/manifests/charset.json")
    cfg = ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=16, lstm_layers=1,
                      dropout=0.0)  # fmt: skip
    arch = {"model_name": "crnn", "model": dataclasses.asdict(cfg)}
    rec = TextRecognizer(build_recognizer(charset.num_classes, cfg), charset, 32, 512,
                         temperature=2.35, arch=arch, strict_fp32=True)  # fmt: skip
    mcfg = DetModelConfig(backbone="mobilenet_v3_small", init="scratch", inner_channels=16)
    model = build_detector(mcfg, Path("."), load_pretrained=False)
    spec = DetectorSpec(mcfg, DetLabelConfig(), DetInferConfig(long_side=160), DetPostConfig(),
                        "inverse10", LABEL_VERSION)  # fmt: skip
    det = TextLineDetector(TorchProbMap(model, torch.device("cpu")), spec)
    return Frozen(GlypharisOCR(det, rec), rec, dict(PROVENANCE))


@pytest.fixture
def layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """A repository-like cwd with three validation pages and the profile shortened to 1 + 2;
    ``loads`` lists the calls to the (stubbed) frozen-model builder.

    Thread counts, the thread variables and grad mode are process-wide; they are put back after
    each test. Machine-state probes are stubbed (no powercfg, no 2 s CPU sampling)."""
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    pages = []
    for i, size in enumerate(SIZES):
        Image.new("RGB", size, (235, 235, 225)).save(data / f"raw/p{i}.png")
        pages.append(Page(f"p{i}", f"raw/p{i}.png", *size, "hiertext", "validation", []))
    write_pages(data / EVAL_SETS["hiertext_val"]["pages"], pages)
    (data / PHASE5_HASHES).parent.mkdir(parents=True, exist_ok=True)
    (data / PHASE5_HASHES).write_text('{"files": []}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for key in e2e_profile.THREAD_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(e2e_profile, "WARMUP", 1)
    monkeypatch.setattr(e2e_profile, "PROFILE_SAMPLES", 2)
    monkeypatch.setattr(e2e_profile, "run_conditions", lambda: {"cpu_busy_percent_before": 2.0})
    monkeypatch.setattr(e2e_profile, "cpu_busy_percent", lambda interval=2.0: 3.0)
    monkeypatch.setattr(e2e_profile, "git_state", lambda: {"git_commit": "t" * 40,
                                                           "git_dirty": False})  # fmt: skip
    monkeypatch.setattr(e2e_profile, "pinned_code_check", lambda: dict(PINNED))
    calls: list[tuple[Any, ...]] = []

    def fake_load(*args: Any, **kwargs: Any) -> Frozen:
        calls.append((args, kwargs))
        return tiny_frozen()

    monkeypatch.setattr(e2e_profile, "load_frozen", fake_load)
    threads, grad = torch.get_num_threads(), torch.is_grad_enabled()
    yield SimpleNamespace(root=tmp_path, loads=calls)
    torch.set_num_threads(threads)
    torch.set_grad_enabled(grad)


def _threads() -> int:
    """A thread count this process can always be set to: the inter-op pool may already be fixed."""
    return torch.get_num_interop_threads()


def _profile(n: int) -> dict[str, Any]:
    return json.loads(e2e_profile.frozen_out_path(n).read_text(encoding="utf-8"))


def test_the_frozen_profile_needs_the_phase_5_hashes_first(layout):
    (layout.root / "data" / PHASE5_HASHES).unlink()
    with pytest.raises(SystemExit, match="hash-phase5"):
        e2e_profile.main(["--frozen", "--threads", str(_threads())])
    assert layout.loads == []


def test_the_scorer_reads_the_profilers_thread_and_pin_fields(layout):
    """The profile GW-EFF-005 writes, read by GW-E2E-009's void check: its thread count, torch
    pools, pin and model hashes pass; only the conditions this fixture stubs are reported."""
    import phase14_score as ps

    n = _threads()
    assert e2e_profile.main(["--frozen", "--threads", str(n)]) == 0
    prof = _profile(n)
    prof["threads"]["env_before_torch_import"] = True  # set by a command-line start only
    runs = {"tag_commit": PINNED["tag_commit"],
            **{k: prof["provenance"].get(k) for k in ps.MODEL_HASHES}}  # fmt: skip
    problems = ps.profile_problems("GW-EFF-005", prof, runs, threads=n)
    assert not [p for p in problems if "thread" in p or "pool" in p or "pinned" in p
                or "differs" in p or "tag" in p]  # fmt: skip
    assert problems  # the stubbed conditions (no power record, 1 + 2 pages) are still caught


def test_thread_variables_match_the_baseline_runner():
    assert e2e_profile.THREAD_ENV == run_baselines.THREAD_ENV


@pytest.mark.parametrize(
    ("argv", "want"),
    [
        (["--frozen", "--threads", "10"], 10),
        (["--threads=16", "--frozen"], 16),
        (["--frozen", "--threads", "x"], None),
        (["--frozen", "--threads", "0"], None),
        (["--frozen", "--threads"], None),
        (["--config", "c.yaml"], None),
    ],
)
def test_threads_are_read_from_the_command_line_before_imports(argv, want):
    assert e2e_profile.requested_threads(argv) == want


def test_frozen_profile_records_threads_settings_and_per_page_times(layout):
    n = _threads()
    assert e2e_profile.main(["--frozen", "--threads", str(n)]) == 0
    assert layout.loads == [(("cpu",), {})]
    path = layout.root / "reports/external/GW-EFF-005" / f"profile_glypharis_{n}t.json"
    assert path.is_file()
    prof = _profile(n)
    assert (prof["system"], prof["task"], prof["eval_set"]) == ("glypharis", "e2e", "hiertext_val")
    assert prof["experiment"] == "GW-EFF-005" and prof["version"] == "glypharis_v1"
    s = prof["settings"]
    assert s["cpu_threads"] == s["intra_op_threads"] == s["inter_op_threads"] == n
    assert s["recognizer"] == {"input": {"height": 32, "max_width": 512}, "temperature": 2.35,
                               "strict_fp32": True}  # fmt: skip
    assert s["detector"]["infer"] == {"long_side": 160} and s["device"] == "cpu"
    t = prof["threads"]
    assert (t["requested"], t["intra_op"], t["inter_op"]) == (n, n, n)
    assert t["env"] == {k: str(n) for k in run_baselines.THREAD_ENV}
    assert t["env_before_torch_import"] is False  # torch was loaded before this test imported it
    assert all(os.environ[k] == str(n) for k in run_baselines.THREAD_ENV)
    assert prof["batch"] == 1 and prof["warmup"] == 1
    assert len(prof["warmup_times_ms"]) == 1 and len(prof["times_ms"]) == 2
    assert prof["input_sizes"] == [list(SIZES[1]), list(SIZES[2])]  # same order as the baseline
    assert len(prof["parts_ms"]) == len(prof["lines"]) == 2
    assert prof["mean_ms"] == pytest.approx(sum(prof["times_ms"]) / 2)
    assert prof["provenance"] == PROVENANCE and prof["pinned"] == PINNED
    assert prof["conditions"] == {"cpu_busy_percent_before": 2.0}
    assert prof["cpu_busy_percent_after"] == 3.0 and prof["git_commit"] == "t" * 40
    assert prof["peak_mem_mib"] > 0
    text = path.read_text(encoding="utf-8")
    assert str(layout.root) not in text and layout.root.as_posix() not in text  # repo-relative only
    assert torch.is_grad_enabled() is False


class _Clock:
    """A `datetime` stand-in that notes each ``now()`` in ``events``."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def now(self, tz: Any = None) -> datetime:
        self.events.append("now")
        return datetime.now(tz)


def test_the_frozen_profile_records_when_its_sitting_ran(layout, monkeypatch):
    """``started_utc`` just before the first warm-up page, ``ended_utc`` after the last timed
    one, with their UTC offset, so the scorer can order the two profiles of a sitting."""
    events: list[str] = []
    read = e2e_profile.open_page_image

    def logged(page: Page, root: Path) -> Image.Image:
        events.append("page")
        return read(page, root)

    monkeypatch.setattr(e2e_profile, "open_page_image", logged)
    monkeypatch.setattr(e2e_profile, "datetime", _Clock(events))
    n = _threads()
    before = datetime.now(UTC).replace(microsecond=0)
    assert e2e_profile.main(["--frozen", "--threads", str(n)]) == 0
    assert events == ["now", "page", "page", "page", "now"]  # 1 warm-up + 2 timed pages
    prof = _profile(n)
    start = datetime.fromisoformat(prof["started_utc"])
    end = datetime.fromisoformat(prof["ended_utc"])
    assert start.utcoffset() == end.utcoffset() == timedelta(0)
    assert before <= start <= end <= datetime.now(UTC)


def test_an_existing_profile_is_never_overwritten(layout):
    n = _threads()
    assert e2e_profile.main(["--frozen", "--threads", str(n)]) == 0
    before = e2e_profile.frozen_out_path(n).read_bytes()
    with pytest.raises(SystemExit, match="exists: move it aside"):
        e2e_profile.main(["--frozen", "--threads", str(n)])
    assert e2e_profile.frozen_out_path(n).read_bytes() == before
    assert len(layout.loads) == 1


def test_unpinned_code_is_refused_before_the_model_loads(layout, monkeypatch):
    def unpinned() -> dict[str, Any]:
        raise RuntimeError("uncommitted changes under glypharis, scripts: ['scripts/x.py']")

    monkeypatch.setattr(e2e_profile, "pinned_code_check", unpinned)
    n = _threads()
    with pytest.raises(SystemExit, match="GW-EFF-005 runs at the tag"):
        e2e_profile.main(["--frozen", "--threads", str(n)])
    assert layout.loads == []
    assert not e2e_profile.frozen_out_path(n).exists()
    # The test-only escape hatch runs and records why the check failed.
    assert e2e_profile.main(["--frozen", "--threads", str(n), "--allow-unpinned"]) == 0
    assert _profile(n)["pinned"] == {"error": "uncommitted changes under glypharis, scripts: "
                                              "['scripts/x.py']"}  # fmt: skip


def test_a_thread_count_torch_cannot_take_is_refused(layout, monkeypatch):
    n = _threads()

    def fixed(_: int) -> None:
        raise RuntimeError("cannot set number of interop threads after parallel work has started")

    monkeypatch.setattr(torch, "set_num_interop_threads", fixed)
    with pytest.raises(SystemExit, match="run in a fresh process"):
        e2e_profile.main(["--frozen", "--threads", str(n + 1)])
    assert layout.loads == []
    assert not e2e_profile.frozen_out_path(n + 1).exists()


def test_threads_changed_by_the_model_load_are_refused(layout, monkeypatch):
    n = _threads()

    def resetting(*args: Any, **kwargs: Any) -> Frozen:
        torch.set_num_threads(n + 1)
        return tiny_frozen()

    monkeypatch.setattr(e2e_profile, "load_frozen", resetting)
    with pytest.raises(SystemExit, match="thread counts changed"):
        e2e_profile.main(["--frozen", "--threads", str(n)])
    assert not e2e_profile.frozen_out_path(n).exists()


def test_only_validation_pages_are_profiled(layout):
    data = layout.root / "data"
    pages = [Page("p0", "raw/p0.png", *SIZES[0], "hiertext", "test", [])]
    write_pages(data / EVAL_SETS["hiertext_val"]["pages"], pages)
    n = _threads()
    with pytest.raises(SystemExit, match="validation pages only"):
        e2e_profile.main(["--frozen", "--threads", str(n)])
    assert not e2e_profile.frozen_out_path(n).exists()


def test_too_few_pages_are_refused(layout, monkeypatch):
    monkeypatch.setattr(e2e_profile, "PROFILE_SAMPLES", 3)  # 1 + 3 > the three pages
    n = _threads()
    with pytest.raises(SystemExit, match="gave 3 pages; need 4"):
        e2e_profile.main(["--frozen", "--threads", str(n)])
    assert not e2e_profile.frozen_out_path(n).exists()


@pytest.mark.parametrize("argv", [["--frozen"], ["--frozen", "--threads", "0"]])
def test_the_frozen_path_requires_a_positive_thread_count(layout, argv):
    with pytest.raises(SystemExit) as exc:
        e2e_profile.main(argv)
    assert exc.value.code == 2  # argparse
    assert layout.loads == []


def test_the_earlier_command_line_still_works(layout, monkeypatch):
    cfg = SimpleNamespace(det=object(), paths=SimpleNamespace(data_root="data"))
    monkeypatch.setattr(e2e_profile, "load_config", lambda path, overrides: cfg)
    monkeypatch.setattr(e2e_profile, "load_bundle", lambda path, backend: tiny_frozen().ocr)
    out = layout.root / "reports/efficiency/x.json"
    argv = ["--config", "c.yaml", "--bundle", "runs/release/b", "--backend", "torch",
            "--experiment", "GW-EFF-004", "--out", str(out)]  # fmt: skip
    assert e2e_profile.main(argv) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["experiment"] == "GW-EFF-004" and res["bundle"] == "b"
    assert res["runtime"] == "torch bundle b" and res["long_side"] == 160
    assert len(res["times_ms"]) == 2 and res["input_sizes"] == [list(SIZES[1]), list(SIZES[2])]
    assert "meets_target" in res and "threads" not in res
    assert layout.loads == []
    assert not (layout.root / "reports/external").exists()


def test_a_command_line_run_sets_the_variables_before_torch_loads(tmp_path):
    """Run as a script, --threads reaches the environment before torch is imported: torch then
    starts with that many intra-op threads without being told. The run stops at once because its
    output exists, before any model, page or git command is touched."""
    out = tmp_path / e2e_profile.frozen_out_path(3)
    out.parent.mkdir(parents=True)
    out.write_text("{}\n", encoding="utf-8")
    script = REPO / "scripts/e2e_profile.py"
    code = (
        "import os, runpy, sys\n"
        f"sys.argv = [{str(script)!r}, '--frozen', '--threads', '3']\n"
        "try:\n"
        f"    runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    print('exit:', exc)\n"
        "import torch\n"
        "print('env:', [os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OMP_THREAD_LIMIT',"
        " 'MKL_NUM_THREADS')])\n"
        "print('threads:', torch.get_num_threads())\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in e2e_profile.THREAD_ENV}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "scripts")])
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert "exists: move it aside" in proc.stdout
    assert "env: ['3', '3', '3']" in proc.stdout
    assert "threads: 3" in proc.stdout
