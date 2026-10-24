from __future__ import annotations

import json

import evaluate_baselines
import numpy as np
import pytest
import run_baselines
from PIL import Image

from glypharis.data.crops import encode_jpeg
from glypharis.data.linepack import LinePackWriter
from glypharis.data.pages import Page, PageLine, write_pages
from glypharis.data.text import Charset

TSV = "\n".join(
    [
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
        "1\t1\t0\t0\t0\t0\t0\t0\t500\t300\t-1\t",
        "4\t1\t1\t1\t1\t0\t10\t10\t200\t20\t-1\t",
        "5\t1\t1\t1\t1\t1\t10\t10\t80\t20\t90\tHello",
        "5\t1\t1\t1\t1\t2\t100\t12\t110\t22\t70\tworld",
        "5\t1\t1\t1\t2\t1\t10\t40\t60\t20\t95\tNext",
        "5\t1\t1\t1\t2\t2\t80\t40\t60\t20\t-1\t ",
    ]
)
LOG_HEADER = "| Date | Benchmark | ID | Checkpoint | Phase | Reason |\n|---|---|---|---|---|---|\n"
PLACEHOLDER = "| — | — | — | — | — | No test access yet. |\n"
OOC = (True, False, False, False, "out_of_charset")  # legible, handwritten, vertical, rec_ok, skip
VERTICAL = (True, False, True, False, "vertical")


# ------------------------------------------------------------------------- runner helpers


def test_tesseract_tsv_groups_words_into_lines():
    lines = run_baselines.tesseract_tsv_lines(TSV)
    assert [ln["text"] for ln in lines] == ["Hello world", "Next"]
    assert lines[0]["polygon"] == [[10, 10], [210, 10], [210, 34], [10, 34]]
    assert lines[0]["confidence"] == 0.8
    assert run_baselines.tesseract_tsv_lines("") == []


def test_unrotate_inverts_quarter_turns():
    h, w = 30, 50
    page = np.arange(h * w).reshape(h, w)
    for angle, k in ((90, 1), (180, 2), (270, 3)):
        rotated = np.rot90(page, k)  # counter-clockwise, k quarter turns
        ry, rx = 7, 11
        x, y = run_baselines.unrotate(rx + 0.5, ry + 0.5, angle, w, h)
        assert page[int(y), int(x)] == rotated[ry, rx], angle
    assert run_baselines.unrotate(3, 4, 0, w, h) == [3, 4]


def test_validation_is_subsampled_and_test_is_complete():
    val = run_baselines.rec_indices(50_000, "validation")
    assert len(val) == run_baselines.REC_VAL_SAMPLE and len(set(val)) == len(val)
    assert (np.diff(val) > 0).all()
    assert np.array_equal(val, run_baselines.rec_indices(50_000, "validation"))
    assert len(run_baselines.rec_indices(50_000, "test")) == 50_000
    assert len(run_baselines.page_indices(100, "validation")) == 100


def test_resume_repairs_the_end_of_an_interrupted_file(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"key": "a"}\n{"key": "b"}\n{"key": "c", "pre', encoding="utf-8")
    assert run_baselines.done_keys(path) == {"a", "b"}
    assert path.read_text(encoding="utf-8") == '{"key": "a"}\n{"key": "b"}\n'
    # a complete record that lost only its newline keeps its key and gets the newline back
    path.write_text('{"key": "a"}\n{"key": "b"}', encoding="utf-8")
    assert run_baselines.done_keys(path) == {"a", "b"}
    assert path.read_text(encoding="utf-8").endswith("}\n")
    # a cut inside a multi-byte character
    path.write_bytes(b'{"key": "a"}\n{"key": "b", "pred": "caf' + "é".encode()[:1])
    assert run_baselines.done_keys(path) == {"a"}
    assert path.read_bytes() == b'{"key": "a"}\n'
    assert run_baselines.done_keys(tmp_path / "missing.jsonl") == set()


# --------------------------------------------------------------------------- fake data root


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _lines(prefix):
    return [
        PageLine(f"{prefix}/l0", _box(0, 0, 100, 20), "alpha", "alpha", True, False, False, True),
        PageLine(f"{prefix}/l1", _box(0, 30, 100, 50), "beta", "beta", True, False, False, True),
        PageLine(f"{prefix}/l2", _box(0, 60, 100, 80), "naïve", "naïve", *OOC),
        PageLine(f"{prefix}/v0", _box(150, 0, 170, 90), "STAMP", "STAMP", *VERTICAL),
    ]


def _pack(prefix, lines, split, source):
    crop = encode_jpeg(Image.new("L", (40, 16), 255), 95)
    with LinePackWriter(prefix) as w:
        for ln in lines:
            if ln.line_id.endswith(("l0", "l1", "l2")):  # the recognition lines
                w.add(ln.line_id, ln.line_id.split("/")[0], split, source, ln.text, crop, (40, 16))


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A data root with one page per page-based set, its line pack and an empty access log."""
    data = tmp_path / "data"
    Charset(list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")).save(
        data / "manifests/charset.json"
    )
    (data / "raw").mkdir(parents=True)
    Image.new("RGB", (200, 100), "white").save(data / "raw/page.png")
    for name, (pages, pack, split, source) in {
        "gdc_test": ("processed/gdc/pages_test.jsonl.gz", "processed/gdc/lines_test", "test",
                     "gdc"),
        "hiertext_test": ("processed/hiertext/pages_test.jsonl.gz", "processed/hiertext/lines_test",
                          "test", "hiertext"),
        "hiertext_val": ("processed/hiertext/pages_validation.jsonl.gz",
                         "processed/hiertext/lines_validation_eval", "validation", "hiertext"),
    }.items():  # fmt: skip
        lines = _lines(name)
        write_pages(data / pages, [Page(name, "raw/page.png", 200, 100, source, split, lines)])
        _pack(data / pack, lines, split, source)
    log = tmp_path / "access.md"
    log.write_text(LOG_HEADER + PLACEHOLDER, encoding="utf-8")
    monkeypatch.setattr(run_baselines, "ACCESS_LOG", log)
    monkeypatch.setattr(run_baselines, "cpu_busy_percent", lambda interval=2.0: 3.0)
    return data, log


class FakeSystem:
    version = "fake 1.0"

    def __init__(self, fail_after=None):
        self.calls, self.fail_after, self.child_peak = 0, fail_after, 0

    def settings(self):
        return {"cpu_threads": 2}

    def inference_batch(self, runner_batch):
        return runner_batch

    def model_bytes(self):
        return 2**20

    def recognize(self, images):
        self.calls += len(images)
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("interrupted")
        return ["alpha"] * len(images)

    def page(self, img):
        return [{"polygon": _box(0, 0, 100, 20), "text": "alpha", "confidence": 1.0}]


def _run(data, *extra, system="tesseract", experiment="GW-BASE-001"):
    return run_baselines.main(["--system", system, "--experiment", experiment, "--data-root",
                               str(data), "--batch", "1", *extra])  # fmt: skip


# ---------------------------------------------------------------------------- runner main


def test_recognition_set_keeps_out_of_charset_lines(root):
    data, _ = root
    keys = run_baselines.expected_keys(data, "gdc_test", "rec")
    assert keys == ["gdc_test/l0", "gdc_test/l1", "gdc_test/l2"]
    assert run_baselines.expected_keys(data, "gdc_test", "e2e") == ["gdc_test"]


def test_runner_writes_metadata_first_resumes_and_logs_each_access(root, monkeypatch):
    data, log = root
    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: FakeSystem(fail_after=1))
    with pytest.raises(RuntimeError, match="interrupted"):
        _run(data, "--task", "rec", "--eval-set", "gdc_test")
    out = data / "processed/baselines/GW-BASE-001/tesseract_rec_gdc_test.jsonl"
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["runs"][0]["status"] == "running" and meta["expected"] == 3
    assert meta["runs"][0]["git_commit"] is not None

    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: FakeSystem())
    assert _run(data, "--task", "rec", "--eval-set", "gdc_test") == 0
    records = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["key"] for r in records) == ["gdc_test/l0", "gdc_test/l1", "gdc_test/l2"]
    assert [r["seg"] for r in records] == [0, 1, 1]
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in meta["runs"]] == ["running", "done"] and meta["samples"] == 3
    rows = [ln for ln in log.read_text(encoding="utf-8").splitlines() if "GW-BASE-001" in ln]
    assert len(rows) == 2 and "resumed: 1 of 3" in rows[1]

    # complete: no model load and no further access-log row
    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: pytest.fail("model loaded"))
    assert _run(data, "--task", "rec", "--eval-set", "gdc_test") == 0
    assert sum("GW-BASE-001" in ln for ln in log.read_text(encoding="utf-8").splitlines()) == 2


def test_runner_refuses_wrong_ids_and_stale_files(root, monkeypatch):
    data, _ = root
    with pytest.raises(SystemExit, match="registered as GW-BASE-001"):
        _run(data, "--task", "rec", "--eval-set", "gdc_test", experiment="GW-BASE-002")
    out = data / "processed/baselines/GW-BASE-001/tesseract_rec_gdc_test.jsonl"
    out.parent.mkdir(parents=True)
    out.write_text('{"key": "old-sample/l9"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="move it aside"):
        _run(data, "--task", "rec", "--eval-set", "gdc_test")


def test_profile_times_after_warmup_on_validation_only(root, monkeypatch):
    data, _ = root
    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: FakeSystem())
    monkeypatch.setattr(run_baselines, "WARMUP", 1)
    monkeypatch.setattr(run_baselines, "PROFILE_SAMPLES", 2)
    with pytest.raises(SystemExit, match="profiling rec uses"):
        _run(data, "--task", "rec", "--eval-set", "gdc_test", "--profile")
    with pytest.raises(SystemExit, match="profiling needs"):  # one page cannot fill 1 + 2
        _run(data, "--task", "e2e", "--eval-set", "hiertext_val", "--profile")
    assert _run(data, "--task", "rec", "--eval-set", "hiertext_val", "--profile") == 0
    path = data / "processed/baselines/GW-BASE-001/profile_tesseract_rec_hiertext_val.json"
    prof = json.loads(path.read_text(encoding="utf-8"))
    assert len(prof["warmup_times_ms"]) == 1 and len(prof["times_ms"]) == 2
    assert prof["batch"] == 1 and prof["input_sizes"] == [[40, 16], [40, 16]]
    assert prof["conditions"]["cpu_busy_percent_before"] == 3.0
    assert prof["peak_mem_mib"] >= prof["process_peak_mib"] > 0 and prof["model_mib"] == 1.0


# ------------------------------------------------------------------------------- scoring


def _write_run(data, system, experiment, task, eval_set, records):
    out = run_baselines.output_path(data, experiment, system, task, eval_set)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    meta = {"system": system, "task": task, "eval_set": eval_set, "experiment": experiment,
            "version": "x 1.0", "settings": {"cpu_threads": 2},
            "runs": [{"status": "done"}]}  # fmt: skip
    out.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return out


def _page_record(key):
    return {"key": key, "time_ms": 1.0, "lines": [
        {"polygon": _box(0, 0, 100, 20), "text": "alpha", "confidence": 0.9},
        {"polygon": _box(150, 0, 170, 90), "text": "STAMP", "confidence": 0.9},
    ]}  # fmt: skip


def test_scorer_writes_every_table_and_logs_test_scoring(root):
    data, log = root
    _write_run(data, "tesseract", "GW-BASE-001", "rec", "gdc_test",
               [{"key": "gdc_test/l0", "ref": "alpha", "pred": "alpha"},
                {"key": "gdc_test/l1", "ref": "beta", "pred": "bota"},
                {"key": "gdc_test/l2", "ref": "naïve", "pred": "naive"}])  # fmt: skip
    _write_run(data, "tesseract", "GW-BASE-001", "e2e", "gdc_test", [_page_record("gdc_test")])
    _write_run(data, "tesseract", "GW-BASE-001", "e2e", "hiertext_val",
               [_page_record("hiertext_val")])  # fmt: skip
    prof = {"system": "tesseract", "task": "rec", "eval_set": "hiertext_val",
            "experiment": "GW-BASE-001", "settings": {"OMP_THREAD_LIMIT": None}, "batch": 1,
            "warmup": 10, "times_ms": [10.0] * 100, "input_sizes": [[40, 16]] * 100,
            "peak_mem_mib": 80.0, "child_peak_mib": 50.0, "invocation_overhead_ms": 120.0,
            "cpu_busy_percent_after": 4.0,
            "conditions": {"on_ac_power": True, "power_plan": "Balanced", "cpu": "test cpu",
                           "cpu_busy_percent_before": 3.0}}  # fmt: skip
    (data / "processed/baselines/GW-BASE-001/profile_tesseract_rec_hiertext_val.json").write_text(
        json.dumps(prof), encoding="utf-8"
    )
    out = data.parent / "reports"
    args = ["--data-root", str(data), "--out", str(out), "--access-log", str(log)]
    assert evaluate_baselines.main(args) == 0

    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    entries = {(e["task"], e["eval_set"]): e for e in results["predictions"]}
    rec = entries["rec", "gdc_test"]["tracks"]
    assert rec["strict"]["cer"]["value"] == 2 / 14  # "bota" (1) and "naive" (1) over 14 chars
    assert rec["strict"]["charset_ceiling"] == pytest.approx(1 / 14)  # the "ï"
    gdc = entries["e2e", "gdc_test"]
    assert gdc["detection"]["iou_0.5"]["recall"]["value"] == pytest.approx(1 / 3)
    assert gdc["tracks"]["strict"]["order_aware_reported"] is True
    assert "page_cer_order_aware" in gdc["tracks"]["strict"]
    vert = gdc["vertical"]["strict"]
    assert vert["recall"]["value"] == 1.0 and vert["lines"] == 1
    assert vert["cer"]["value"] == 0.0  # the read "alpha" elsewhere is not charged to the stamp
    ht = entries["e2e", "hiertext_val"]["tracks"]["strict"]
    assert ht["order_aware_reported"] is False and "page_cer_order_aware" not in ht
    assert results["efficiency"][0]["mean_ms"] == 10.0

    rows = [ln for ln in log.read_text(encoding="utf-8").splitlines() if "scoring" in ln]
    assert len(rows) == 2  # the two gdc test files; the validation file is not logged
    e2e_md = (out / "end_to_end.md").read_text(encoding="utf-8")
    assert "## GDC vertical lines" in e2e_md and "| n/a | n/a |" in e2e_md
    eff = (out / "efficiency.md").read_text(encoding="utf-8")
    assert "no baseline uses a GPU" in eff and "120 ms on a blank image" in eff
    summary = (out / "README.md").read_text(encoding="utf-8")
    assert "| GW-BASE-001 | Tesseract 5 | rec | gdc_test | 3 of 3 |" in summary
    assert "planned, no predictions yet" in summary and "TrOCR has no detector" in summary


def test_scorer_rejects_predictions_for_another_sample(root):
    data, log = root
    _write_run(data, "tesseract", "GW-BASE-001", "rec", "gdc_test",
               [{"key": "gdc_test/l0", "ref": "alpha", "pred": "alpha"},
                {"key": "gdc_test/l0", "ref": "alpha", "pred": "alpha"}])  # fmt: skip
    args = ["--data-root", str(data), "--out", str(data.parent / "r"), "--access-log", str(log)]
    with pytest.raises(SystemExit, match="1 keys repeated"):
        evaluate_baselines.main(args)


def test_scorer_tolerates_a_cut_off_last_record_and_lists_files_without_metadata(root):
    data, log = root
    out = _write_run(data, "tesseract", "GW-BASE-001", "rec", "gdc_test",
                     [{"key": "gdc_test/l0", "ref": "alpha", "pred": "alpha"}])  # fmt: skip
    with out.open("ab") as fh:
        fh.write(b'{"key": "gdc_test/l1", "ref": "be')
    no_meta = run_baselines.output_path(data, "GW-BASE-002", "ppocr_mobile", "rec", "gdc_test")
    no_meta.parent.mkdir(parents=True, exist_ok=True)
    no_meta.write_text('{"key": "gdc_test/l0", "ref": "alpha", "pred": "alpha"}\n', "utf-8")
    # a run that died before its first page: metadata, no predictions
    _write_run(data, "ppocr_server", "GW-BASE-003", "e2e", "gdc_test", [])
    report = data.parent / "rep"
    args = ["--data-root", str(data), "--out", str(report), "--access-log", str(log)]
    assert evaluate_baselines.main(args) == 0
    summary = (report / "README.md").read_text(encoding="utf-8")
    assert "| GW-BASE-001 | Tesseract 5 | rec | gdc_test | 1 of 3 |" in summary
    assert "1 predicted, no run metadata yet" in summary
    assert (
        "| GW-BASE-003 | PP-OCRv5 server | e2e | gdc_test | not runnable on this laptop" in summary
    )
    assert "GW-BASE-003" not in (report / "end_to_end.md").read_text(encoding="utf-8")
    assert "ppocr_server" not in log.read_text(encoding="utf-8")  # nothing scored, nothing logged


def test_diagnostics_read_validation_predictions_only(root):
    import baseline_diagnostics

    data, _ = root
    _write_run(data, "ppocr_server", "GW-BASE-003", "rec", "hiertext_val",
               [{"key": "a", "ref": "two words", "pred": "twowords"},
                {"key": "b", "ref": "Mixed case", "pred": "MIXED CASE"},
                {"key": "c", "ref": "one", "pred": "one"}])  # fmt: skip
    long_ref = "x" * 80
    _write_run(data, "ppocr_mobile", "GW-BASE-002", "rec", "glyphsynth_val",
               [{"key": "a", "ref": long_ref, "pred": "x" * 20},
                {"key": "b", "ref": "short", "pred": "short"}])  # fmt: skip
    _write_run(data, "tesseract", "GW-BASE-001", "rec", "gdc_test",
               [{"key": "gdc_test/l0", "ref": "secret", "pred": "secret"}])  # fmt: skip
    out = data.parent / "diag"
    assert baseline_diagnostics.main(["--data-root", str(data), "--out", str(out)]) == 0
    res = json.loads((out / "diagnostics.json").read_text(encoding="utf-8"))
    gaps = res["word_gaps"]["ppocr_server"]
    assert gaps["multi_word"] == 2 and gaps["keeps_a_space"] == 0.5
    assert gaps["wrong_only_in_spaces"] == 1
    # three references have lowercase letters; one output has none
    assert res["letter_case"]["ppocr_server"]["no_lowercase_output"] == pytest.approx(1 / 3)
    buckets = res["long_lines"]["ppocr_mobile"]["buckets"]
    assert buckets[0]["length_ratio"] == 1.0 and buckets[-1]["length_ratio"] == 0.25
    assert res["long_lines"]["ppocr_mobile"]["cut_short"] == 1
    assert "tesseract" not in res["word_gaps"]  # the test-set file is never read
    assert "GW-BASE-003 | PP-OCRv5 server | 2 | 50.0 | 1 |" in (out / "diagnostics.md").read_text(
        encoding="utf-8"
    )


def test_runner_refuses_bad_requests_and_handles_metadata(root, monkeypatch):
    data, _ = root
    with pytest.raises(SystemExit, match="no pages"):
        _run(data, "--task", "e2e", "--eval-set", "glyphsynth_val")
    with pytest.raises(SystemExit, match="reason is required"):
        _run(data, "--task", "rec", "--eval-set", "gdc_test", "--reason", " ")
    out = run_baselines.output_path(data, "GW-BASE-001", "tesseract", "rec", "gdc_test")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = out.with_suffix(".meta.json")
    meta.write_text('{"runs": [{"status": "done"}], "settings": {"old": 1}}', encoding="utf-8")
    monkeypatch.setattr(run_baselines, "make_system", lambda n, t: FakeSystem())
    assert _run(data, "--task", "rec", "--eval-set", "gdc_test") == 0  # predictions moved aside
    fresh = json.loads(meta.read_text(encoding="utf-8"))
    assert len(fresh["runs"]) == 1 and "old" not in fresh["settings"]
    meta.unlink()  # complete predictions whose run never wrote metadata get a minimal record
    assert _run(data, "--task", "rec", "--eval-set", "gdc_test") == 0
    assert "not recorded" in json.loads(meta.read_text(encoding="utf-8"))["note"]
