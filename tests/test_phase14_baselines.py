"""Phase 14 rules of the baseline runner, the scorer, the evaluation sets and the crop packs.

Everything here runs on small synthetic data roots in tmp_path; no prepared benchmark is read.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from datetime import datetime, timedelta

import build_eval_crops
import evaluate_baselines
import pytest
import run_baselines
from PIL import Image

from glypharis.data.crops import encode_jpeg
from glypharis.data.linepack import LinePackReader, LinePackWriter
from glypharis.data.pages import Page, PageLine, write_pages
from glypharis.data.text import Charset
from glypharis.eval import sets
from glypharis.eval.access import HeldOutAccessError, open_test_access

LOG_HEADER = "| Date | Benchmark | ID | Checkpoint | Phase | Reason |\n|---|---|---|---|---|---|\n"
PLACEHOLDER = "| — | — | — | — | — | No test access yet. |\n"
OOC = (True, False, False, False, "out_of_charset")
VERTICAL = (True, False, True, False, "vertical")
SECRET = "GDCSCAN-MARKER"


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _lines(prefix, vertical=True):
    lines = [
        PageLine(f"{prefix}/l0", _box(0, 0, 100, 20), "alpha", "alpha", True, False, False, True),
        PageLine(f"{prefix}/l1", _box(0, 30, 100, 50), "beta", "beta", True, False, False, True),
        PageLine(f"{prefix}/l2", _box(0, 60, 100, 80), "naïve", "naïve", *OOC),
    ]
    if vertical:
        lines.append(PageLine(f"{prefix}/v0", _box(150, 0, 170, 90), "STAMP", "STAMP", *VERTICAL))
    return lines


def _pack(prefix, keys_labels, split, source):
    crop = encode_jpeg(Image.new("L", (40, 16), 255), 95)
    with LinePackWriter(prefix) as w:
        for key, label in keys_labels:
            w.add(key, key.split("/")[0], split, source, label, crop, (40, 16))


SETTINGS = {"cpu_threads": 10, "module": "fake"}


class FakeSystem:
    version = "fake 1.0"

    def __init__(self, settings=None):
        self._settings = dict(SETTINGS if settings is None else settings)
        self.child_peak = 0

    def settings(self):
        return dict(self._settings)

    def inference_batch(self, runner_batch):
        return runner_batch

    def model_bytes(self):
        return 2**20

    def recognize(self, images):
        return ["alpha"] * len(images)

    def page(self, img):
        return [{"polygon": _box(0, 0, 100, 20), "text": "alpha", "confidence": 1.0}]


PHASE5_RUNS = [{"status": "loading"}, {"status": "done", "runner_batch": 2, "inference_batch": 2}]


def _phase5_meta(data, system, task, eval_set, settings, **extra):
    exp = run_baselines.EXPERIMENTS[system]
    path = data / "processed/baselines" / exp / f"{system}_{task}_{eval_set}.meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"system": system, "task": task, "eval_set": eval_set, "experiment": exp,
            "expected": 3, "settings": settings, "runs": PHASE5_RUNS, **extra}  # fmt: skip
    path.write_text(json.dumps(meta), encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A data root with every Phase 14 set, a validation page set, Phase 5 run metadata for the
    fake systems, and an empty access log; the branch check passes."""
    data = tmp_path / "data"
    Charset(list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")).save(
        data / "manifests/charset.json"
    )
    (data / "raw").mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(data / "raw/page.png")
    for name, source, vertical in (("funsd_test", "funsd", False),
                                   ("gdc_scan_test", "gdc_scan", True)):  # fmt: skip
        spec = sets.EVAL_SETS[name]
        lines = _lines(name, vertical)
        write_pages(data / spec["pages"], [Page(name, "raw/page.png", 200, 100, source, "test",
                                                lines)])  # fmt: skip
        _pack(data / spec["pack"], [(ln.line_id, ln.text) for ln in lines[:3]], "test", source)
    _pack(data / sets.EVAL_SETS["iiit5k_test"]["pack"],
          [("iiit5k/0", "HELLO"), ("iiit5k/1", "WORLD"), ("iiit5k/2", "A-1")], "test",
          "iiit5k")  # fmt: skip
    # OmniDocBench: three pages of blocks, two of them in the subset.
    odb = []
    for i in range(3):
        blocks = [
            PageLine(f"odb-{i}/b0", _box(0, 0, 190, 40), "first block", "first block", True,
                     False, False, True),
            PageLine(f"odb-{i}/b1", _box(0, 50, 190, 90), "second block", "second block", True,
                     False, False, True),
        ]  # fmt: skip
        odb.append(Page(f"odb-{i}", "raw/page.png", 200, 100, "omnidocbench", "test", blocks))
    write_pages(data / sets.EVAL_SETS["omnidocbench_test"]["pages"], odb)
    subset = data / sets.OMNIDOCBENCH_SUBSET.relative_to("data")
    subset.parent.mkdir(parents=True, exist_ok=True)
    subset.write_text(json.dumps({"pages": ["odb-0", "odb-2"]}), encoding="utf-8")
    hv = sets.EVAL_SETS["hiertext_val"]
    lines = _lines("hiertext_val", False)
    write_pages(data / hv["pages"], [Page("hv", "raw/page.png", 200, 100, "hiertext",
                                          "validation", lines)])  # fmt: skip
    _pack(data / hv["pack"], [(ln.line_id, ln.text) for ln in lines], "validation", "hiertext")
    for system in ("tesseract", "ppocr_mobile", "ppocr_server", "trocr"):
        for task in ("rec", "e2e"):
            _phase5_meta(data, system, task, "gdc_test", SETTINGS)
    log = tmp_path / "access.md"
    log.write_text(LOG_HEADER + PLACEHOLDER, encoding="utf-8")
    monkeypatch.setattr(run_baselines, "ACCESS_LOG", log)
    monkeypatch.setattr(run_baselines, "cpu_busy_percent", lambda interval=2.0: 3.0)
    monkeypatch.setattr(run_baselines, "require_branch", lambda name, start=None: None)
    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: FakeSystem())
    monkeypatch.setattr(run_baselines, "REQUIRE_RECORDED", False)  # the tiny sets have no record
    (data / sets.PHASE5_HASHES).write_text('{"files": []}', encoding="utf-8")
    return data, log


def test_phase_14_runs_need_the_phase_5_hashes_and_close_phase_5(root):
    data, log = root
    with pytest.raises(SystemExit, match="Phase 5 files are closed"):
        run_baselines.main(["--phase", "5", "--reason", "x", "--system", "tesseract",
                            "--experiment", "GW-BASE-001", "--data-root", str(data), "--task",
                            "rec", "--eval-set", "gdc_test"])  # fmt: skip
    (data / sets.PHASE5_HASHES).unlink()
    with pytest.raises(SystemExit, match="hash-phase5"):
        _run(data, "--task", "rec", "--eval-set", "funsd_test")
    assert not _rows(log)


def _run(data, *extra, system="tesseract", reason="Phase 14 baseline"):
    return run_baselines.main(["--phase", "14", "--reason", reason, "--system", system,
                               "--experiment", run_baselines.EXPERIMENTS[system], "--data-root",
                               str(data), "--batch", "2", *extra])  # fmt: skip


def _rows(log):
    return [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.startswith("| 20")]


# --------------------------------------------------------------------------------- the sets


def test_eval_sets_are_shared_and_complete():
    assert run_baselines.EVAL_SETS is sets.EVAL_SETS
    assert set(sets.PHASE5_SETS) | set(sets.PHASE14_SETS) == set(sets.EVAL_SETS)
    for name, spec in sets.EVAL_SETS.items():
        assert {"benchmark", "split", "pack", "pages", "reading_order", "vertical_view",
                "block_level"} <= set(spec), name  # fmt: skip
    vertical = {k for k, v in sets.EVAL_SETS.items() if v["vertical_view"]}
    assert vertical == {"gdc_test", "gdc_scan_test"}
    assert {k for k, v in sets.EVAL_SETS.items() if v["block_level"]} == {"omnidocbench_test"}
    assert sets.EVAL_SETS["omnidocbench_test"]["pack"] is None
    assert sets.EVAL_SETS["iiit5k_test"]["pages"] is None
    assert all(sets.EVAL_SETS[k]["split"] == "test" for k in sets.PHASE14_SETS)
    # The crop builder cuts exactly the packs the runners read.
    for name, (pages, pack) in build_eval_crops.SETS.items():
        assert (sets.EVAL_SETS[name]["pages"], sets.EVAL_SETS[name]["pack"]) == (pages, pack)
    assert {"gdc_scan_test", "funsd_test"} <= set(build_eval_crops.SETS)


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
                   cwd=repo, check=True, capture_output=True)  # fmt: skip


def test_repo_root_access_log_and_branch_on_a_tmp_repository(tmp_path):
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", sets.PHASE14_BRANCH)
    (repo / "f.txt").write_text("x", encoding="utf-8")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "first")
    assert sets.repo_root(repo / "sub").resolve() == repo.resolve()
    log = sets.access_log(repo / "sub")
    assert log.resolve() == (repo / "research/test_access_log.md").resolve()
    sets.require_branch(sets.PHASE14_BRANCH, repo)
    _git(repo, "checkout", "-q", "-b", "other")
    with pytest.raises(RuntimeError, match="HEAD is on other"):
        sets.require_branch(sets.PHASE14_BRANCH, repo)
    _git(repo, "checkout", "-q", "--detach")
    with pytest.raises(RuntimeError, match="HEAD is on HEAD"):
        sets.require_branch(sets.PHASE14_BRANCH, repo)
    with pytest.raises(RuntimeError, match="git rev-parse"):
        sets.repo_root(tmp_path)  # not a repository


def test_access_log_refuses_rows_that_break_the_table_and_a_missing_log(tmp_path):
    log = tmp_path / "log.md"
    log.write_text(LOG_HEADER + PLACEHOLDER, encoding="utf-8")
    with pytest.raises(HeldOutAccessError, match="reason"):
        open_test_access("gdc_scan", "test", "GW-BASE-001", "x", 14, "a | b", log)
    with pytest.raises(HeldOutAccessError, match="checkpoint"):
        open_test_access("gdc_scan", "test", "GW-BASE-001", "x\ny", 14, "why", log)
    with pytest.raises(HeldOutAccessError, match="not found"):
        open_test_access("gdc_scan", "test", "GW-BASE-001", "x", 14, "why", tmp_path / "no.md")
    assert PLACEHOLDER in log.read_text(encoding="utf-8")
    open_test_access("gdc_scan", "test", "GW-BASE-001", "tesseract (rec)", 14, "why", log)
    assert "| gdc_scan test | GW-BASE-001 | tesseract (rec) | 14 | why |" in log.read_text("utf-8")


def test_subset_pages_reads_the_committed_list_under_the_data_root(tmp_path):
    path = tmp_path / sets.OMNIDOCBENCH_SUBSET.relative_to("data")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"pages": ["a", "b"]}), encoding="utf-8")
    assert sets.subset_pages(tmp_path) == ["a", "b"]
    path.write_text(json.dumps({"pages": ["a", "a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        sets.subset_pages(tmp_path)


def test_the_committed_subset_lists_100_distinct_pages():
    path = sets.OMNIDOCBENCH_SUBSET
    if not path.exists():
        pytest.skip("run from the repository root")
    pages = sets.subset_pages(path.parents[2])
    assert len(pages) == 100 and pages == sorted(pages)


# ------------------------------------------------------------------------------ the runner


def test_phase14_writes_only_under_its_own_folder_and_logs_phase_14(root):
    data, log = root
    assert _run(data, "--task", "rec", "--eval-set", "funsd_test") == 0
    out = data / "processed/baselines_phase14/GW-BASE-001/tesseract_rec_funsd_test.jsonl"
    keys = [json.loads(x)["key"] for x in out.read_text(encoding="utf-8").splitlines()]
    assert keys == ["funsd_test/l0", "funsd_test/l1", "funsd_test/l2"]
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["phase"] == 14 and meta["subset"] is None and meta["settings"] == SETTINGS
    check = meta["phase5_settings_check"]
    assert check["result"] == "equal"
    assert check["references"] == [
        "processed/baselines/GW-BASE-001/tesseract_rec_gdc_test.meta.json"
    ]
    assert not list((data / "processed/baselines").glob("*/*.jsonl"))  # Phase 5 never written
    pack = sets.EVAL_SETS["funsd_test"]["pack"]
    assert meta["inputs"]["pack"] == {
        "path": f"data/{pack}", "bin_sha256": build_eval_crops.sha256(data / f"{pack}.bin"),
        "index_sha256": build_eval_crops.sha256(data / f"{pack}.index.jsonl"), "count": 3,
        "recorded_in": None,
    }  # fmt: skip
    assert meta["inputs"]["pages"]["count"] is None  # a rec run does not read the pages
    assert meta["batch"] == {"runner_batch": 2, "inference_batch": 2}
    assert meta["phase5_batch_check"] == {
        "result": "equal", "inference_batch": 2,
        "references": ["processed/baselines/GW-BASE-001/tesseract_rec_gdc_test.meta.json"],
    }  # fmt: skip
    (row,) = _rows(log)
    assert "| funsd test | GW-BASE-001 | tesseract (rec) | 14 | Phase 14 baseline |" in row
    # complete: no further row
    assert _run(data, "--task", "rec", "--eval-set", "funsd_test") == 0
    assert len(_rows(log)) == 1
    # a page run counts the pages of the file it reads
    assert _run(data, "--task", "e2e", "--eval-set", "funsd_test") == 0
    out = run_baselines.output_path(data, "GW-BASE-001", "tesseract", "e2e", "funsd_test", 14)
    pages = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))["inputs"]["pages"]
    assert pages["count"] == 1 and pages["path"] == f"data/{sets.EVAL_SETS['funsd_test']['pages']}"


def test_phase14_refuses_inputs_without_or_against_their_record_before_logging(root, monkeypatch):
    data, log = root
    monkeypatch.setattr(run_baselines, "REQUIRE_RECORDED", True)
    with pytest.raises(SystemExit, match="no recorded hash in data/manifests"):
        _run(data, "--task", "rec", "--eval-set", "funsd_test")
    monkeypatch.setattr(run_baselines, "REQUIRE_RECORDED", False)
    man = data / "manifests/processed/eval_crops.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    pack = sets.EVAL_SETS["funsd_test"]["pack"]
    man.write_text(json.dumps({"funsd_test": {"pack": pack, "bin_sha256": "0" * 64, "count": 3}}),
                   encoding="utf-8")  # fmt: skip
    with pytest.raises(SystemExit, match=r"records 0{64}; refusing to run"):  # still compared
        _run(data, "--task", "rec", "--eval-set", "funsd_test")
    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: pytest.fail("model loaded"))
    with pytest.raises(SystemExit, match="refusing to run"):  # before the model loads
        _run(data, "--task", "rec", "--eval-set", "funsd_test")
    assert _rows(log) == []
    assert not run_baselines.output_path(data, "GW-BASE-001", "tesseract", "rec", "funsd_test",
                                         14).exists()  # fmt: skip


def test_phase14_runs_phase_5s_inference_batch(root, monkeypatch):
    data, log = root
    with pytest.raises(SystemExit, match=r"inference batch 3, Phase 5 ran \[2\]"):
        _run(data, "--task", "rec", "--eval-set", "funsd_test", "--batch", "3")
    # no Phase 5 run recorded a batch: refused for a system whose batch follows --batch ...
    _phase5_meta(data, "tesseract", "rec", "gdc_test", SETTINGS, runs=[{"status": "loading"}])
    with pytest.raises(SystemExit, match="no Phase 5 run metadata records an inference batch"):
        _run(data, "--task", "rec", "--eval-set", "funsd_test")
    assert _rows(log) == []

    class OneAtATime(FakeSystem):
        def inference_batch(self, runner_batch):
            return 1

    # ... and run for one that fixes its batch, so Phase 5 could not have run another
    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: OneAtATime())
    assert _run(data, "--task", "rec", "--eval-set", "funsd_test") == 0
    out = run_baselines.output_path(data, "GW-BASE-001", "tesseract", "rec", "funsd_test", 14)
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["phase5_batch_check"]["result"] == "fixed by the system"
    assert meta["batch"] == {"runner_batch": 2, "inference_batch": 1}
    assert meta["runs"][-1]["inference_batch"] == 1
    assert len(_rows(log)) == 1


def test_phase14_resume_logs_a_resume_row(root, monkeypatch):
    data, log = root
    out = run_baselines.output_path(data, "GW-BASE-001", "tesseract", "rec", "iiit5k_test", 14)
    out.parent.mkdir(parents=True)
    out.write_text('{"key": "iiit5k/0", "ref": "HELLO", "pred": "x"}\n', encoding="utf-8")
    assert _run(data, "--task", "rec", "--eval-set", "iiit5k_test") == 0
    (row,) = _rows(log)
    assert "| 14 | Phase 14 baseline (resumed: 1 of 3 samples already predicted) |" in row
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


@pytest.mark.parametrize("field", ["inputs", "batch"])
def test_phase14_resume_must_read_the_same_inputs_and_batch(root, field):
    data, log = root
    out = run_baselines.output_path(data, "GW-BASE-001", "tesseract", "rec", "iiit5k_test", 14)
    out.parent.mkdir(parents=True)
    out.write_text('{"key": "iiit5k/0", "ref": "HELLO", "pred": "x"}\n', encoding="utf-8")
    meta = {"phase": 14, "runs": [{"status": "running"}], field: {"other": 1}}
    out.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit, match=f"recorded other {field}; a resume must be the same"):
        _run(data, "--task", "rec", "--eval-set", "iiit5k_test")
    assert _rows(log) == []
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_phase14_refuses_off_branch_and_the_reused_sets(root, monkeypatch):
    data, log = root

    def off_branch(name, start=None):
        raise RuntimeError("this run must start on branch phase/14-external-eval; HEAD is on main")

    monkeypatch.setattr(run_baselines, "require_branch", off_branch)
    with pytest.raises(SystemExit, match="HEAD is on main"):
        _run(data, "--task", "rec", "--eval-set", "funsd_test")
    monkeypatch.setattr(run_baselines, "require_branch", lambda name, start=None: None)
    with pytest.raises(SystemExit, match="reuses the Phase 5 predictions"):
        _run(data, "--task", "rec", "--eval-set", "gdc_test")
    with pytest.raises(SystemExit, match="is a Phase 14 set"):
        run_baselines.main(["--phase", "5", "--reason", "x", "--system", "tesseract",
                            "--experiment", "GW-BASE-001", "--data-root", str(data), "--task",
                            "rec", "--eval-set", "funsd_test"])  # fmt: skip
    with pytest.raises(SystemExit):  # --reason and --phase are required
        run_baselines.main(["--system", "tesseract", "--experiment", "GW-BASE-001", "--task",
                            "rec", "--eval-set", "funsd_test"])  # fmt: skip
    with pytest.raises(SystemExit, match="no line pack"):
        _run(data, "--task", "rec", "--eval-set", "omnidocbench_test")
    assert _rows(log) == []


def test_phase14_refuses_settings_that_differ_from_phase_5_before_logging(root, monkeypatch):
    data, log = root
    monkeypatch.setattr(run_baselines, "make_system",
                        lambda n, t: FakeSystem({**SETTINGS, "cpu_threads": 4}))  # fmt: skip
    with pytest.raises(SystemExit, match=r"differ from Phase 5's .* in cpu_threads"):
        _run(data, "--task", "rec", "--eval-set", "funsd_test")
    monkeypatch.setattr(run_baselines, "make_system",
                        lambda n, t: FakeSystem({**SETTINGS, "extra": 1}))  # fmt: skip
    with pytest.raises(SystemExit, match="in extra"):  # a field added counts as a difference
        _run(data, "--task", "rec", "--eval-set", "funsd_test")
    assert _rows(log) == []
    out = run_baselines.output_path(data, "GW-BASE-001", "tesseract", "rec", "funsd_test", 14)
    assert not out.exists()
    with pytest.raises(SystemExit, match="no Phase 5 run metadata"):
        _run(data, "--task", "rec", "--eval-set", "funsd_test", system="ppocr6_tiny")


def test_settings_check_reads_settings_only_skips_legacy_and_applies_the_wording_alias(tmp_path):
    data = tmp_path
    old = {**SETTINGS, "detector_run_mode": "paddle (no oneDNN; speed only)"}
    new = {**SETTINGS, "detector_run_mode": "paddle (oneDNN crashes on this machine)"}
    _phase5_meta(data, "ppocr_mobile", "e2e", "gdc_test", old)
    prof = data / "processed/baselines/GW-BASE-002/profile_ppocr_mobile_e2e_hiertext_val.json"
    prof.write_text(json.dumps({"system": "ppocr_mobile", "task": "e2e", "settings": new}), "utf-8")
    legacy = _phase5_meta(data, "ppocr_mobile", "e2e", "glyphsynth_val", {"other": 1})
    meta = json.loads(legacy.read_text(encoding="utf-8"))
    del meta["expected"]
    legacy.write_text(json.dumps(meta), encoding="utf-8")
    _phase5_meta(data, "ppocr_mobile", "e2e", "hiertext_test", None)
    # a prediction file next to them is never opened
    (data / "processed/baselines/GW-BASE-002/ppocr_mobile_e2e_gdc_test.jsonl").write_bytes(b"\xff")
    check = run_baselines.check_phase5_settings(data, "ppocr_mobile", "e2e", new)
    assert check["result"] == "equal" and len(check["references"]) == 2
    assert check["wording_aliases"] == [
        "processed/baselines/GW-BASE-002/ppocr_mobile_e2e_gdc_test.meta.json: detector_run_mode"
    ]
    assert set(check["skipped"]) == {
        "processed/baselines/GW-BASE-002/ppocr_mobile_e2e_glyphsynth_val.meta.json",
        "processed/baselines/GW-BASE-002/ppocr_mobile_e2e_hiertext_test.meta.json",
    }
    with pytest.raises(SystemExit, match="detector_run_mode"):  # the alias is one way only
        run_baselines.check_phase5_settings(data, "ppocr_mobile", "e2e", old)


def test_phase14_coverage_and_the_omnidocbench_subset(root):
    data, log = root
    with pytest.raises(SystemExit, match="subset only: pass --subset"):
        _run(data, "--task", "e2e", "--eval-set", "omnidocbench_test", system="ppocr_server")
    with pytest.raises(SystemExit, match="same pages twice"):
        _run(data, "--task", "e2e", "--eval-set", "omnidocbench_test", "--subset")
    with pytest.raises(SystemExit, match="died on GDC"):
        _run(data, "--task", "e2e", "--eval-set", "gdc_scan_test", system="ppocr_server")
    with pytest.raises(SystemExit, match="no detector"):
        _run(data, "--task", "e2e", "--eval-set", "funsd_test", system="trocr")
    with pytest.raises(SystemExit, match="omnidocbench_test only"):
        _run(data, "--task", "e2e", "--eval-set", "funsd_test", "--subset")
    assert _rows(log) == []
    assert _run(data, "--task", "e2e", "--eval-set", "omnidocbench_test", "--subset",
                system="ppocr_server") == 0  # fmt: skip
    out = data / ("processed/baselines_phase14/GW-BASE-003/"
                  "ppocr_server_e2e_omnidocbench_test_subset100.jsonl")  # fmt: skip
    keys = [json.loads(x)["key"] for x in out.read_text(encoding="utf-8").splitlines()]
    assert keys == ["odb-0", "odb-2"]
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["subset"] == "subset100" and meta["expected"] == 2
    assert len(_rows(log)) == 1
    # a subset page missing from the pages file means the subset came from another file
    subset = data / sets.OMNIDOCBENCH_SUBSET.relative_to("data")
    subset.write_text(json.dumps({"pages": ["odb-0", "odb-9"]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not in omnidocbench_test"):
        run_baselines.expected_keys(data, "omnidocbench_test", "e2e", subset=True)


def test_phase14_takes_the_split_from_the_records(root):
    data, log = root
    _pack(data / sets.EVAL_SETS["iiit5k_test"]["pack"], [("iiit5k/0", "HELLO")], "train",
          "iiit5k")  # fmt: skip
    with pytest.raises(SystemExit, match="records name splits"):
        _run(data, "--task", "rec", "--eval-set", "iiit5k_test")
    assert _rows(log) == []


def test_phase14_profile_writes_apart_asserts_ten_threads_and_never_overwrites(root, monkeypatch):
    data, _ = root
    monkeypatch.setattr(run_baselines, "WARMUP", 1)
    monkeypatch.setattr(run_baselines, "PROFILE_SAMPLES", 2)
    _phase5_meta(data, "ppocr_mobile", "rec", "hiertext_val", SETTINGS)
    assert _run(data, "--task", "rec", "--eval-set", "hiertext_val", "--profile",
                system="ppocr_mobile") == 0  # fmt: skip
    path = (
        data / "processed/baselines_phase14/GW-BASE-002/profile_ppocr_mobile_rec_hiertext_val.json"
    )
    prof = json.loads(path.read_text(encoding="utf-8"))
    assert prof["phase"] == 14 and prof["settings"]["cpu_threads"] == 10
    assert prof["input_sizes"] == [[40, 16], [40, 16]] and len(prof["times_ms"]) == 2
    assert prof["phase5_settings_check"]["result"] == "equal"
    started, ended = (datetime.fromisoformat(prof[k]) for k in ("started_utc", "ended_utc"))
    assert started.utcoffset() == ended.utcoffset() == timedelta(0) and started <= ended
    assert not (
        data / "processed/baselines/GW-BASE-002/profile_ppocr_mobile_rec_hiertext_val.json"
    ).exists()
    with pytest.raises(SystemExit, match="move it aside"):
        _run(data, "--task", "rec", "--eval-set", "hiertext_val", "--profile",
             system="ppocr_mobile")  # fmt: skip
    path.unlink()
    monkeypatch.setattr(run_baselines, "make_system",
                        lambda n, t: FakeSystem({**SETTINGS, "cpu_threads": 16}))  # fmt: skip
    with pytest.raises(SystemExit, match="Phase 14 profiles run at 10"):
        _run(data, "--task", "rec", "--eval-set", "hiertext_val", "--profile",
             system="ppocr_mobile")  # fmt: skip


# ------------------------------------------------------------------------------ the scorer


def _write_run(data, system, task, eval_set, records, subset=False):
    exp = run_baselines.EXPERIMENTS[system]
    out = run_baselines.output_path(data, exp, system, task, eval_set, 14, subset)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    meta = {"system": system, "task": task, "eval_set": eval_set, "experiment": exp,
            "expected": len(records), "version": "x 1.0", "settings": SETTINGS, "phase": 14,
            "subset": "subset100" if subset else None, "runs": [{"status": "done"}]}  # fmt: skip
    out.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return out


def _page(key, texts):
    return {"key": key, "time_ms": 1.0, "lines": [
        {"polygon": _box(0, 50 * i, 190, 50 * i + 40), "text": t, "confidence": 0.9}
        for i, t in enumerate(texts)]}  # fmt: skip


def test_the_scorer_cli_is_phase_5_only_and_closes_once_phase_5_is_hashed(root):
    data, log = root
    with pytest.raises(SystemExit, match="scored once, by `scripts/phase14_score"):
        evaluate_baselines.main(["--phase", "14", "--data-root", str(data)])
    (data / sets.PHASE5_HASHES).parent.mkdir(parents=True, exist_ok=True)
    (data / sets.PHASE5_HASHES).write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="Phase 5 scoring is closed"):
        evaluate_baselines.main(["--phase", "5", "--data-root", str(data), "--access-log",
                                 str(log)])  # fmt: skip
    assert not _rows(log)
    with pytest.raises(ValueError, match="its own reason"):
        evaluate_baselines.Context(data, log, 14)


REASON14 = "Phase 14 scoring (GW-E2E-009)"


def test_score_file_scores_a_complete_phase14_file_and_logs_one_row(root):
    data, log = root
    ctx = evaluate_baselines.Context(data, log, 14, REASON14)
    iiit = _write_run(data, "tesseract", "rec", "iiit5k_test",
                      [{"key": "iiit5k/0", "ref": "HELLO", "pred": "hello"},
                       {"key": "iiit5k/1", "ref": "WORLD", "pred": "W0RLD"},
                       {"key": "iiit5k/2", "ref": "A-1", "pred": "a1"}])  # fmt: skip
    entry = evaluate_baselines.score_file(iiit, ctx)
    assert entry["samples"] == entry["expected"] == 3
    assert entry["tracks"]["alnum"]["word_accuracy"]["value"] == pytest.approx(2 / 3)  # W0RLD
    assert entry["tracks"]["alnum"]["word_accuracy"]["unit"] == "words"
    assert entry["tracks"]["strict"]["sequence_accuracy"] == 0.0  # case counts on strict
    (row,) = _rows(log)
    assert "| iiit5k test | GW-BASE-001 | tesseract (rec), scoring | 14 | " + REASON14 in row

    scan_lines = [
        {"polygon": _box(0, 0, 100, 20), "text": "alpha", "confidence": 0.9},
        {"polygon": _box(150, 0, 170, 90), "text": SECRET, "confidence": 0.9},
    ]
    scan = evaluate_baselines.score_file(
        _write_run(data, "tesseract", "e2e", "gdc_scan_test",
                   [{"key": "gdc_scan_test", "time_ms": 1.0, "lines": scan_lines}]), ctx
    )  # fmt: skip
    assert scan["vertical"]["strict"]["lines"] == 1
    assert "page_cer_order_aware" in scan["tracks"]["strict"]
    assert SECRET not in json.dumps(scan)  # no text of a prediction reaches an entry

    pages = [_page(f"odb-{i}", ["first block", "second blick"]) for i in range(3)]
    odb = evaluate_baselines.score_file(
        _write_run(data, "tesseract", "e2e", "omnidocbench_test", pages), ctx
    )
    assert odb["detection"] is None
    for t in ("strict", "normalised"):
        assert odb["tracks"][t]["page_cer_bag_of_lines"] is None
        assert odb["tracks"][t]["e2e_f1"] is None
        assert odb["tracks"][t]["page_cer_order_aware"]["value"] > 0
    rows = _rows(log)
    assert len(rows) == 3 and all(r.endswith(f"| 14 | {REASON14} |") for r in rows)


def test_score_file_refuses_an_incomplete_file_before_logging(root):
    data, log = root
    ctx = evaluate_baselines.Context(data, log, 14, REASON14)
    path = _write_run(data, "tesseract", "rec", "iiit5k_test",
                      [{"key": "iiit5k/0", "ref": "HELLO", "pred": "hello"}])  # fmt: skip
    with path.open("ab") as fh:  # and a record cut off by a killed run
        fh.write(b'{"key": "iiit5k/1", "ref": "WOR')
    with pytest.raises(ValueError, match=r"1 of 3 expected keys; an incomplete file") as exc:
        evaluate_baselines.score_file(path, ctx)
    assert "HELLO" not in str(exc.value) and "iiit5k/" not in str(exc.value)  # counts only
    assert _rows(log) == []
    # a subset run missing a subset page, and a whole-set file scored on the subset
    sub = _write_run(data, "ppocr_server", "e2e", "omnidocbench_test",
                     [_page("odb-0", ["first block"])], subset=True)  # fmt: skip
    with pytest.raises(ValueError, match="1 of 2 expected keys"):
        evaluate_baselines.score_file(sub, ctx)
    whole = _write_run(data, "tesseract", "e2e", "omnidocbench_test",
                       [_page("odb-0", ["first block"]), _page("odb-1", ["x"])])  # fmt: skip
    with pytest.raises(ValueError, match="1 of 2 expected keys"):
        evaluate_baselines.score_file(whole, ctx, subset=True)
    assert _rows(log) == []
    # the Phase 5 path keeps scoring a partial file as it stands
    entry = evaluate_baselines.evaluate_file(path, ctx)
    assert entry is not None and entry["samples"] == 1 and entry["expected"] == 3
    assert len(_rows(log)) == 1


def test_the_report_writer_is_phase_5_only():
    assert not hasattr(evaluate_baselines, "PLAN14")
    params = inspect.signature(evaluate_baselines.score_folder).parameters
    assert list(params) == ["data", "root", "out", "access"]


def test_score_file_restricts_a_whole_set_file_to_the_subset(root):
    data, log = root
    pages = [_page(f"odb-{i}", ["first block", "second block"]) for i in range(3)]
    path = _write_run(data, "tesseract", "e2e", "omnidocbench_test", pages)
    ctx = evaluate_baselines.Context(data, log, 14, "Phase 14 scoring (GW-E2E-009)")
    entry = evaluate_baselines.score_file(path, ctx, subset=True)
    assert entry["samples"] == 2 and entry["expected"] == 2 and entry["subset"] == "subset100"
    assert entry["tracks"]["strict"]["page_cer_order_aware"]["value"] == 0.0
    full = evaluate_baselines.score_file(path, ctx)
    assert full["samples"] == 3 and "subset" not in full
    assert len(_rows(log)) == 2
    sub = _write_run(data, "ppocr_server", "e2e", "omnidocbench_test",
                     [_page("odb-0", ["first block"]), _page("odb-2", ["second block"])],
                     subset=True)  # fmt: skip
    entry = evaluate_baselines.score_file(sub, ctx)
    assert entry["subset"] == "subset100" and entry["expected"] == 2
    # explicit metadata (GW-E2E-009 scores its own files this way) and a stray key
    meta = {"system": "glypharis", "task": "e2e", "eval_set": "omnidocbench_test",
            "experiment": "GW-E2E-007"}  # fmt: skip
    entry = evaluate_baselines.score_file(path, ctx, meta=meta)
    assert entry["experiment"] == "GW-E2E-007" and entry["samples"] == 3
    assert "glypharis (e2e), scoring" in _rows(log)[-1]
    with pytest.raises(SystemExit, match="outside the evaluation set"):
        evaluate_baselines.score_file(sub, ctx, meta={**meta, "eval_set": "funsd_test"})
    path.with_suffix(".meta.json").unlink()
    with pytest.raises(ValueError, match="no run metadata"):
        evaluate_baselines.score_file(path, ctx)


# ------------------------------------------------------------------------------ crop packs


@pytest.fixture
def crop_root(tmp_path, monkeypatch):
    """FUNSD, GDC-scan and GDC pages of 3 crops each; GDC-scan also has one untrusted box, and
    its freeze and alignment records; the registered sizes are scaled down to these pages."""
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    (data / "manifests/processed").mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(data / "raw/page.png")
    for name in ("funsd_test", "gdc_scan_test", "gdc_test"):
        lines = _lines(name)
        if name == "gdc_scan_test":  # a box the alignment did not trust: no crop
            lines.append(PageLine(f"{name}/u0", _box(0, 84, 100, 98), SECRET, SECRET, True,
                                  False, False, True, None, 0.1))  # fmt: skip
        write_pages(data / sets.EVAL_SETS[name]["pages"],
                    [Page(name, "raw/page.png", 200, 100, name.split("_test")[0], "test",
                          lines)])  # fmt: skip
    (data / "manifests/gdc_scan").mkdir(parents=True)
    (data / build_eval_crops.GDC_SCAN_RECORD).write_text('{"lines": 5}', encoding="utf-8")
    (data / build_eval_crops.GDC_SCAN_ALIGNMENT).write_text(
        json.dumps({"results": [{"file": "raw/page.png", "lines": 5, "lines_low_score": 1}]}),
        encoding="utf-8")  # fmt: skip
    monkeypatch.setattr(build_eval_crops, "FUNSD_LINES", 3)
    monkeypatch.setattr(build_eval_crops, "GDC_SCAN_LINES", 5)
    return data


def _build(data, eval_set, *extra):
    return build_eval_crops.main(["--eval-set", eval_set, "--data-root", str(data),
                                  "--workers", "1", *extra])  # fmt: skip


def test_crop_packs_for_the_new_sets_assert_their_size(crop_root, monkeypatch, capsys):
    data = crop_root
    man = data / "manifests/processed/eval_crops.json"
    with pytest.raises(SystemExit, match="3 lines selected, expected 4; nothing written"):
        _build(data, "gdc_scan_test", "--expect-count", "4")
    assert not man.exists()
    assert not (data / "processed/gdc_scan/lines_test.bin").exists()
    # The registered size is asserted without --expect-count: 5 lines - 1 vertical - 1 untrusted.
    assert _build(data, "gdc_scan_test") == 0
    reader = LinePackReader(data / "processed/gdc_scan/lines_test")
    # the vertical and the untrusted lines are left out; the out-of-charset line stays in
    assert [r.key for r in reader.records] == [f"gdc_scan_test/l{i}" for i in range(3)]
    monkeypatch.setattr(build_eval_crops, "FUNSD_LINES", 2647)
    with pytest.raises(SystemExit, match="3 lines selected, expected 2647"):
        _build(data, "funsd_test")
    monkeypatch.setattr(build_eval_crops, "FUNSD_LINES", 3)
    assert _build(data, "funsd_test") == 0
    manifest = json.loads(man.read_text(encoding="utf-8"))
    for name, pack in (("gdc_scan_test", "processed/gdc_scan/lines_test"),
                       ("funsd_test", "processed/funsd/lines_test")):  # fmt: skip
        entry = manifest[name]
        assert entry["pack"] == pack and entry["count"] == entry["expected_count"] == 3
        assert entry["index_sha256"] == build_eval_crops.sha256(data / f"{pack}.index.jsonl")
    assert manifest["funsd_test"]["crop_settings"] == manifest["gdc_scan_test"]["crop_settings"]
    assert _build(data, "funsd_test", "--verify", "--expect-count", "3") == 0
    with pytest.raises(SystemExit, match="expected 4"):
        _build(data, "funsd_test", "--verify", "--expect-count", "4")
    # A Phase 5 set keeps its behaviour: no registered size, and its index hash is recorded.
    assert _build(data, "gdc_test") == 0
    gdc = json.loads(man.read_text(encoding="utf-8"))["gdc_test"]
    assert "expected_count" not in gdc and gdc["count"] == 3 and gdc["index_sha256"]
    assert SECRET not in man.read_text(encoding="utf-8") and SECRET not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("fault", "message"),
    [("alignment", "1 lines with untrusted boxes in the page records, .* records 2"),
     ("record", "records 6; the plan registers 5"),
     ("missing", "alignment.json is missing")],
)  # fmt: skip
def test_the_gdc_scan_pack_size_is_checked_against_its_records(crop_root, fault, message):
    data = crop_root
    alignment = data / build_eval_crops.GDC_SCAN_ALIGNMENT
    if fault == "alignment":
        alignment.write_text(json.dumps({"results": [{"lines_low_score": 2}]}), encoding="utf-8")
    elif fault == "record":
        (data / build_eval_crops.GDC_SCAN_RECORD).write_text('{"lines": 6}', encoding="utf-8")
    else:
        alignment.unlink()
    with pytest.raises(SystemExit, match=message):
        _build(data, "gdc_scan_test")
    assert not (data / "manifests/processed/eval_crops.json").exists()
    assert not (data / "processed/gdc_scan/lines_test.bin").exists()
