"""Phase 16 rules of the baseline runner: GW-BASE-002's validation profile for GW-EFF-010.

Everything runs on a small synthetic data root in tmp_path; the Phase 14 profile it checks against
is a made-up copy of the real one's settings block, and the system is a fake.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest
import run_baselines
from PIL import Image

from glypharis.data.pages import Page, PageLine, write_pages
from glypharis.eval import sets

# The settings block a Phase 14 PP-OCRv5 mobile page profile records (field names and values).
P14_SETTINGS = {
    "device": "cpu",
    "cpu_threads": 10,
    "precision": "fp32",
    "module": "PaddleOCR pipeline",
    "det_model": "PP-OCRv5_mobile_det",
    "rec_model": "en_PP-OCRv5_mobile_rec",
    "use_doc_unwarping": False,
    "use_doc_orientation_classify": True,
    "use_textline_orientation": True,
    "detector_run_mode": "paddle (oneDNN crashes on this machine)",
    "deviation": "doc unwarping off (it moves polygons)",
}
GPU = {"utc": "2026-11-20T08:00:00+00:00", "gpus": [{"name": "fake gpu", "temperature_c": 41}]}
PROFILE = "processed/baselines_phase16/GW-BASE-002/profile_ppocr_mobile_e2e_hiertext_val.json"
P14_PROFILE = "processed/baselines_phase14/GW-BASE-002/profile_ppocr_mobile_e2e_hiertext_val.json"


class FakePPOCR:
    version = "paddleocr fake, PP-OCRv5_mobile_det + en_PP-OCRv5_mobile_rec"

    def __init__(self, settings=None):
        self._settings = dict(P14_SETTINGS if settings is None else settings)
        self.child_peak = 0

    def settings(self):
        return dict(self._settings)

    def inference_batch(self, runner_batch):
        return 1

    def model_bytes(self):
        return 2**20

    def page(self, img):
        return [{"polygon": [[0, 0], [10, 0], [10, 5], [0, 5]], "text": "a", "confidence": 1.0}]


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A data root with three validation pages, the Phase 14 profile's settings, the Phase 5 files
    closed (as they are after Phase 14), and the branch check passing."""
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    Image.new("RGB", (120, 80), "white").save(data / "raw/page.png")
    line = PageLine("hv/l0", _box(0, 0, 50, 20), "alpha", "alpha", True, False, False, True)
    pages = [Page(f"hv{i}", "raw/page.png", 120, 80, "hiertext", "validation", [line])
             for i in range(3)]  # fmt: skip
    write_pages(data / sets.EVAL_SETS["hiertext_val"]["pages"], pages)
    p14 = data / P14_PROFILE
    p14.parent.mkdir(parents=True)
    p14.write_text(json.dumps({"system": "ppocr_mobile", "task": "e2e", "eval_set": "hiertext_val",
                               "settings": P14_SETTINGS}), encoding="utf-8")  # fmt: skip
    (data / sets.PHASE5_HASHES).parent.mkdir(parents=True, exist_ok=True)
    (data / sets.PHASE5_HASHES).write_text('{"files": []}', encoding="utf-8")
    branches = []
    monkeypatch.setattr(
        run_baselines, "require_branch", lambda name, start=None: branches.append(name)
    )
    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: FakePPOCR())
    monkeypatch.setattr(run_baselines, "cpu_busy_percent", lambda interval=2.0: 3.0)
    monkeypatch.setattr(run_baselines, "phase16_gpu_state", lambda: GPU)
    monkeypatch.setattr(run_baselines, "WARMUP", 1)
    monkeypatch.setattr(run_baselines, "PROFILE_SAMPLES", 2)
    return data, branches


def _run(data, *extra, system="ppocr_mobile", task="e2e", eval_set="hiertext_val", profile=True):
    args = ["--phase", "16", "--system", system, "--task", task, "--eval-set", eval_set,
            "--experiment", run_baselines.EXPERIMENTS[system], "--reason", "GW-EFF-010 sitting",
            "--data-root", str(data), *extra]  # fmt: skip
    return run_baselines.main([*args, "--profile"] if profile else args)


def test_phase16_profile_writes_apart_and_records_the_sitting(root):
    data, branches = root
    before = sorted(p.relative_to(data).as_posix() for p in data.rglob("*"))
    assert _run(data) == 0
    assert branches == [run_baselines.PHASE16_BRANCH] == ["phase/16-efficiency"]
    prof = json.loads((data / PROFILE).read_text(encoding="utf-8"))
    assert prof["phase"] == 16 and prof["experiment"] == "GW-BASE-002"
    assert prof["settings"] == P14_SETTINGS
    check = prof["phase14_settings_check"]
    assert check["result"] == "equal" and check["reference"] == P14_PROFILE
    assert check["fields"] == sorted(P14_SETTINGS)
    assert len(prof["times_ms"]) == 2 and prof["input_sizes"] == [[120, 80], [120, 80]]
    started, ended = (datetime.fromisoformat(prof[k]) for k in ("started_utc", "ended_utc"))
    assert started.utcoffset() == ended.utcoffset() == timedelta(0) and started <= ended
    # Conditions at the start (Phase 14's key) and at the end, and the GPU at both.
    assert prof["conditions"]["cpu_busy_percent_before"] == 3.0
    assert prof["conditions_end"]["cpu_busy_percent_before"] == 3.0
    assert prof["gpu_state"] == {"start": GPU, "end": GPU}
    assert set(prof["conditions_utc"]) == {"start", "end"}
    # Only the new profile was written; the Phase 5 and Phase 14 folders are untouched.
    after = sorted(p.relative_to(data).as_posix() for p in data.rglob("*"))
    assert set(after) - set(before) == {PROFILE, PROFILE.rsplit("/", 1)[0],
                                        "processed/baselines_phase16"}  # fmt: skip
    with pytest.raises(SystemExit, match="move it aside"):
        _run(data)


def test_phase16_profile_passes_phase_14s_profile_checks(root):
    """The scorer reads the profile with Phase 14's `profile_problems`; with the made-up times it
    must fail only on the counts the test shortened, never on a missing field."""
    import phase14_score

    data, _ = root
    _run(data)
    prof = json.loads((data / PROFILE).read_text(encoding="utf-8"))
    probs = phase14_score.profile_problems("GW-BASE-002", prof, None)
    assert all("warm-up" in p or "mains" in p or "Balanced" in p or "power mode" in p
               for p in probs), probs  # fmt: skip


@pytest.mark.parametrize(
    "settings",
    [
        {**P14_SETTINGS, "use_textline_orientation": False},  # a value changed
        {**P14_SETTINGS, "batch_size": 1},  # a field added
        {k: v for k, v in P14_SETTINGS.items() if k != "deviation"},  # a field lost
    ],
)
def test_phase16_refuses_settings_that_differ_from_phase_14s_profile(root, monkeypatch, settings):
    data, _ = root
    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: FakePPOCR(settings))
    with pytest.raises(SystemExit, match="settings differ from Phase 14's"):
        _run(data)
    assert not (data / PROFILE).exists()


def test_phase16_needs_the_phase_14_profile_and_reads_only_its_settings(root):
    data, _ = root
    p14 = data / P14_PROFILE
    # Other fields of the Phase 14 file play no part.
    doc = json.loads(p14.read_text(encoding="utf-8"))
    p14.write_text(json.dumps({**doc, "version": "older", "times_ms": [1.0]}), encoding="utf-8")
    assert _run(data) == 0
    (data / PROFILE).unlink()
    p14.write_text(json.dumps({**doc, "task": "rec"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="is a profile of"):
        _run(data)
    p14.unlink()
    with pytest.raises(SystemExit, match="no Phase 14 profile"):
        _run(data)
    assert not (data / PROFILE).exists()


def test_phase16_asserts_ten_threads(root, monkeypatch):
    data, _ = root
    monkeypatch.setattr(run_baselines, "make_system",
                        lambda n, t: FakePPOCR({**P14_SETTINGS, "cpu_threads": 16}))  # fmt: skip
    with pytest.raises(SystemExit, match="Phase 16 profiles run at 10"):
        _run(data)


def test_phase16_is_one_profile_on_its_branch_only(root, monkeypatch):
    data, _ = root
    with pytest.raises(SystemExit, match="validation profiles only"):
        _run(data, profile=False)
    with pytest.raises(SystemExit, match="not tesseract e2e"):
        _run(data, system="tesseract")
    with pytest.raises(SystemExit, match="not ppocr_mobile rec"):
        _run(data, task="rec")

    def off_branch(name, start=None):
        raise RuntimeError(f"this run must start on branch {name}; HEAD is on main")

    monkeypatch.setattr(run_baselines, "require_branch", off_branch)
    with pytest.raises(SystemExit, match="phase/16-efficiency"):
        _run(data)
    assert not (data / PROFILE).exists()


def test_phase16_leaves_the_phase_5_and_14_rules_as_they_were(root):
    data, _ = root
    # Phase 5 stays closed once its files are hashed; Phase 16 writes elsewhere and is allowed.
    with pytest.raises(SystemExit, match="Phase 5 files are closed"):
        run_baselines.main(["--phase", "5", "--system", "ppocr_mobile", "--task", "e2e",
                            "--eval-set", "hiertext_val", "--experiment", "GW-BASE-002",
                            "--reason", "x", "--data-root", str(data), "--profile"])  # fmt: skip
    for phase, folder in ((5, "baselines"), (14, "baselines_phase14"), (16, "baselines_phase16")):
        path = run_baselines.output_path(data, "GW-BASE-002", "ppocr_mobile", "e2e",
                                         "hiertext_val", phase)  # fmt: skip
        assert path.relative_to(data).parts[:2] == ("processed", folder)
    assert run_baselines.PHASES == (5, 14, 16)
    assert run_baselines.phase14_profile_path(data, "ppocr_mobile", "e2e", "hiertext_val") == (
        data / P14_PROFILE
    )


@pytest.mark.parametrize("profile", [True, False])
def test_phase16_never_reads_a_test_set(root, monkeypatch, profile):
    """A test split is refused before any data, model or access-log call, with or without
    --profile; the refusal comes from the request checks alone."""
    data, _ = root

    def no_access(*args, **kwargs):
        raise AssertionError("the test-access log was touched")

    monkeypatch.setattr(run_baselines, "make_system", no_access)
    monkeypatch.setattr(run_baselines, "open_test_access", no_access)
    with pytest.raises(SystemExit) as exc:
        _run(data, eval_set="hiertext_test", profile=profile)
    want = "profiling e2e uses ('hiertext_val',)" if profile else "validation profiles only"
    assert want in str(exc.value)
    assert not (data / "processed/baselines_phase16").exists()


def test_phase16_files_its_profile_under_gw_base_002_only(root):
    """Another experiment ID would file the profile elsewhere with a wrong amendment record."""
    data, _ = root
    args = ["--phase", "16", "--system", "ppocr_mobile", "--task", "e2e", "--eval-set",
            "hiertext_val", "--experiment", "GW-BASE-099", "--reason", "x", "--data-root",
            str(data), "--profile"]  # fmt: skip
    with pytest.raises(SystemExit, match="registered as GW-BASE-002, not GW-BASE-099"):
        run_baselines.main(args)
    assert not (data / "processed/baselines_phase16").exists()
