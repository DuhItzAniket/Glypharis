"""Phase 14 GLYPHARIS runner (scripts/ext_eval.py): guards, schema, resume, output manifest.

Everything runs on a synthetic data root in tmp_path with a tiny untrained model standing in for
the frozen one; no prepared benchmark, checkpoint or the real access log is touched.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import ext_eval
import numpy as np
import pytest
import torch
from det_fixtures import rect_polygon
from PIL import Image, ImageDraw

from glypharis.config.schema import (
    DetInferConfig,
    DetLabelConfig,
    DetModelConfig,
    DetPostConfig,
    ModelConfig,
)
from glypharis.data.crops import encode_jpeg
from glypharis.data.det_labels import LABEL_VERSION
from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackWriter
from glypharis.data.pages import Page, PageLine, read_pages, write_pages
from glypharis.data.text import Charset
from glypharis.eval.inputs import GDC_SCAN_MANIFEST
from glypharis.eval.sets import EVAL_SETS, PHASE5_HASHES
from glypharis.models.det.detector import DetectorSpec, TextLineDetector
from glypharis.models.rec.crnn import build_recognizer
from glypharis.pipeline import GlypharisOCR, TextRecognizer
from glypharis.pipeline.frozen import Frozen, FrozenModelError

REPO = Path(__file__).resolve().parents[1]
LOG_HEADER = "| Date | Benchmark | ID | Checkpoint | Phase | Reason |\n|---|---|---|---|---|---|\n"
PLACEHOLDER = "| — | — | — | — | — | No test access yet. |\n"
MARKER = "GDCSCAN-MARKER"
PAGE_SIZE = (400, 300)
LINES = [rect_polygon(200, 60, 300, 24), rect_polygon(200, 150, 300, 24),
         rect_polygon(200, 240, 300, 24)]  # fmt: skip
PINNED = {"tag": "phase14-eval", "tag_commit": "a" * 40, "head": "b" * 40, "trees": "equal"}
# Names a metric would carry; none may appear as a key anywhere in what the runner writes.
METRIC_KEYS = {"cer", "wer", "accuracy", "word_accuracy", "f1", "precision", "recall", "ece",
               "errors", "correct", "edit_distance", "score", "e2e_f1"}  # fmt: skip


class KernelMap:
    """A detector backend that fires on known lines (at the spec's network scale)."""

    def __init__(self, lines: list[list[list[float]]], scale: float) -> None:
        self.lines, self.scale = lines, scale

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        prob = np.zeros((math.ceil(h / 32) * 32, math.ceil(w / 32) * 32), np.float32)
        for poly in self.lines:
            pts = np.asarray(poly, np.float32) * self.scale
            c = pts.mean(axis=0)
            cv2.fillPoly(prob, [np.round(c + 0.6 * (pts - c)).astype(np.int32)], 0.95)
        return prob


def tiny_frozen() -> Frozen:
    torch.manual_seed(0)
    charset = Charset.load(REPO / "data/manifests/charset.json")
    cfg = ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=16, lstm_layers=1,
                      dropout=0.0)  # fmt: skip
    arch = {"model_name": "crnn", "model": dataclasses.asdict(cfg)}
    rec = TextRecognizer(build_recognizer(charset.num_classes, cfg), charset, 32, 512, "cpu",
                         2.35, arch, strict_fp32=True)  # fmt: skip
    mcfg = DetModelConfig(backbone="mobilenet_v3_small", init="scratch", inner_channels=16)
    spec = DetectorSpec(mcfg, DetLabelConfig(), DetInferConfig(long_side=160), DetPostConfig(),
                        "inverse10", LABEL_VERSION)  # fmt: skip
    det = TextLineDetector(KernelMap(LINES, 160 / max(PAGE_SIZE)), spec)
    ocr = GlypharisOCR(det, rec)
    ocr.model_version = "glypharis_v1"
    provenance = {
        "record": "configs/final/glypharis_v1.yaml", "record_sha256": "1" * 64,
        "name": "glypharis_v1", "recognizer_experiment": "GW-REC-034",
        "detector_experiment": "GW-DET-001",
        "recognizer_checkpoint": "checkpoints/GW-REC-034/best.pt",
        "recognizer_checkpoint_sha256": "2" * 64,
        "detector_checkpoint": "checkpoints/GW-DET-001/best.pt",
        "detector_checkpoint_sha256": "3" * 64, "charset": "data/manifests/charset.json",
        "charset_sha256": "4" * 64, "device": "cpu", "device_name": "cpu",
        "torch": torch.__version__, "cuda": None, "cudnn": None,
        "numerics": {"cudnn_benchmark": False, "recognizer_strict_fp32": True},
        "git_commit": "c" * 40, "git_dirty": False,
    }  # fmt: skip
    return Frozen(ocr, rec, provenance)


def _page_image() -> Image.Image:
    img = Image.new("RGB", PAGE_SIZE, (235, 235, 225))
    d = ImageDraw.Draw(img)
    for poly in LINES:
        d.polygon([tuple(p) for p in poly], fill=(30, 30, 30))
    return img


def _line(line_id: str, poly: list[list[float]], text: str, **kw: Any) -> PageLine:
    fields = {"legible": True, "handwritten": False, "vertical": False, "rec_ok": True,
              "rec_skip": None, **kw}  # fmt: skip
    return PageLine(line_id, poly, text, text, **fields)


def _pack(prefix: Path, keys_labels: list[tuple[str, str]], split: str, source: str,
          size: tuple[int, int] = (60, 64)) -> None:  # fmt: skip
    img = Image.new("L", size, 255)
    ImageDraw.Draw(img).rectangle([10, 20, 40, 40], fill=0)
    crop = encode_jpeg(img, 95)
    with LinePackWriter(prefix) as w:
        for key, label in keys_labels:
            w.add(key, key.split("/")[0], split, source, label, crop, size)


def _set(data: Path, name: str, source: str, split: str, pages: list[Page]) -> None:
    spec = EVAL_SETS[name]
    write_pages(data / spec["pages"], pages)
    if spec["pack"]:
        kl = [(ln.line_id, ln.text) for p in pages for ln in p.lines
              if not ln.vertical and ln.align_score is None]  # fmt: skip
        _pack(data / spec["pack"], kl, split, source)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A data root with GDC, GDC-scan, IIIT5K and HierText validation; an empty access log; the
    branch and pinned-code checks pass and the frozen model is the tiny stand-in."""
    data = tmp_path / "data"
    (data / "raw").mkdir(parents=True)
    _page_image().save(data / "raw/page.png")
    w, h = PAGE_SIZE

    def lines(prefix: str, text: str = "alpha") -> list[PageLine]:
        return [_line(f"{prefix}/l{i}", poly, f"{text} {i}") for i, poly in enumerate(LINES)]

    gdc = [Page(f"gdc{i}", "raw/page.png", w, h, "gdc", "test", lines(f"gdc{i}"))
           for i in range(2)]  # fmt: skip
    gdc[0].lines.append(_line("gdc0/untrusted", rect_polygon(100, 280, 80, 16), "x",
                              align_score=0.1))  # fmt: skip
    _set(data, "gdc_test", "gdc", "test", gdc)
    scan = Page("scan0", "raw/page.png", w, h, "gdc_scan", "test", lines("scan0", MARKER))
    scan.lines.append(_line("scan0/v0", rect_polygon(380, 150, 200, 16, 90), MARKER,
                            vertical=True, rec_ok=False, rec_skip="vertical"))  # fmt: skip
    _set(data, "gdc_scan_test", "gdc_scan", "test", [scan])
    _pack(data / EVAL_SETS["iiit5k_test"]["pack"], [("iiit5k/0", "HELLO"), ("iiit5k/1", "A1")],
          "test", "iiit5k", size=(90, 33))  # fmt: skip
    hv = [Page(f"hv{i}", "raw/page.png", w, h, "hiertext", "validation", lines(f"hv{i}"))
          for i in range(2)]  # fmt: skip
    _set(data, "hiertext_val", "hiertext", "validation", hv)

    (data / PHASE5_HASHES).parent.mkdir(parents=True, exist_ok=True)
    (data / PHASE5_HASHES).write_text('{"files": []}\n', encoding="utf-8")
    log = tmp_path / "access.md"
    log.write_text(LOG_HEADER + PLACEHOLDER, encoding="utf-8")
    calls = {"load": 0}

    def load(device, **kw):
        calls["load"] += 1
        return tiny_frozen()

    monkeypatch.setattr(ext_eval, "ACCESS_LOG", log)
    monkeypatch.setattr(ext_eval, "TEST_DEVICE", "cpu")
    monkeypatch.setattr(ext_eval, "REQUIRE_RECORDED", False)  # the tiny sets have no manifests
    monkeypatch.setattr(ext_eval, "require_branch", lambda name, start=None: None)
    monkeypatch.setattr(ext_eval, "pinned_code_check", lambda tag=None, root=None: dict(PINNED))
    monkeypatch.setattr(ext_eval, "load_frozen", load)
    return data, log, calls


def _run(data, mode, eval_set, experiment, *extra, reason="Phase 14 GLYPHARIS"):
    return ext_eval.main([mode, "--set", eval_set, "--experiment", experiment, "--reason", reason,
                          "--device", "cpu", "--data-root", str(data), *extra])  # fmt: skip


def _rows(log):
    return [ln for ln in log.read_text(encoding="utf-8").splitlines() if ln.startswith("| 20")]


def _records(path):
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def _out(data, experiment, eval_set):
    return data / "processed/phase14" / experiment / f"{eval_set}.jsonl"


# ------------------------------------------------------------------------------- the modes


def test_rec_writes_predictions_attributes_one_log_row_and_the_manifest(root, capsys):
    data, log, _ = root
    assert _run(data, "rec", "gdc_test", "GW-REC-039", "--batch", "4") == 0
    out = _out(data, "GW-REC-039", "gdc_test")
    recs = _records(out)
    assert [r["key"] for r in recs] == [f"gdc{p}/l{i}" for p in range(2) for i in range(3)]
    for r in recs:
        assert set(r) == {"key", "ref", "pred", "conf_t1", "conf_t", "time_ms", "crop_height_px",
                          "line_height_px", "rotation_deg", "blur", "contrast", "seg"}  # fmt: skip
        assert 0.0 <= r["conf_t1"] <= 1.0 and 0.0 <= r["conf_t"] <= 1.0
        assert r["crop_height_px"] == 64 and r["line_height_px"] == pytest.approx(24, abs=0.01)
        assert r["rotation_deg"] == pytest.approx(0.0, abs=0.01)
        assert r["blur"] > 0 and r["contrast"] > 0 and r["seg"] == 0
    assert recs[0]["ref"] == "alpha 0"
    (row,) = _rows(log)
    assert row.endswith("| gdc test | GW-REC-039 | glypharis (rec) | 14 | Phase 14 GLYPHARIS |")
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert (meta["system"], meta["task"], meta["mode"], meta["eval_set"], meta["split"]) == (
        "glypharis_v1",
        "rec",
        "rec",
        "gdc_test",
        "test",
    )
    assert meta["expected"] == meta["samples"] == 6 and meta["phase"] == 14
    assert meta["pinned"] == PINNED and meta["confidences_at"] == [1.0, 2.35]
    assert meta["settings"]["recognizer"] == {
        "input": {"height": 32, "max_width": 512},
        "temperature": 2.35,
        "strict_fp32": True,
    }
    assert meta["provenance"]["recognizer_checkpoint_sha256"] == "2" * 64
    assert meta["inputs"]["pack"]["path"] == "data/processed/gdc/lines_test"
    assert meta["inputs"]["pages"]["path"] == "data/processed/gdc/pages_test.jsonl.gz"
    assert meta["runs"][0]["status"] == "done" and meta["runs"][0]["written"] == 6
    manifest = json.loads((data / ext_eval.OUTPUTS_MANIFEST).read_text(encoding="utf-8"))
    (entry,) = manifest["outputs"]
    assert entry["file"] == "data/processed/phase14/GW-REC-039/gdc_test.jsonl"
    assert entry["sha256"] == sha256_file(out) and entry["lines"] == 6
    assert entry["meta_sha256"] == sha256_file(out.with_suffix(".meta.json"))
    printed = capsys.readouterr().out
    assert "6 new, 6 total" in printed and "alpha" not in printed
    # Nothing written to disk carries an absolute path.
    for text in (out.with_suffix(".meta.json").read_text(encoding="utf-8"),
                 (data / ext_eval.OUTPUTS_MANIFEST).read_text(encoding="utf-8")):  # fmt: skip
        assert str(data.parent) not in text and data.parent.as_posix() not in text


def test_rec_on_a_set_without_pages_records_the_crop_only(root):
    data, log, _ = root
    _run(data, "rec", "iiit5k_test", "GW-REC-039")
    recs = _records(_out(data, "GW-REC-039", "iiit5k_test"))
    assert [r["ref"] for r in recs] == ["HELLO", "A1"]
    assert all(r["crop_height_px"] == 33 and r["line_height_px"] is None
               and r["rotation_deg"] is None for r in recs)  # fmt: skip
    assert "| iiit5k test | GW-REC-039 | glypharis (rec) |" in _rows(log)[0]


def test_e2e_writes_the_baselines_schema_in_reading_order(root):
    data, log, _ = root
    _run(data, "e2e", "gdc_test", "GW-E2E-007")
    recs = _records(_out(data, "GW-E2E-007", "gdc_test"))
    assert [r["key"] for r in recs] == ["gdc0", "gdc1"]
    for r in recs:
        assert set(r) == {"key", "lines", "time_ms", "seg"} and len(r["lines"]) == 3
        for ln in r["lines"]:
            assert set(ln) == {"polygon", "quad", "text", "confidence", "conf_t1", "det_score"}
            assert len(ln["quad"]) == 4 and 0.0 <= ln["det_score"] <= 1.0
        ys = [np.mean([p[1] for p in ln["quad"]]) for ln in r["lines"]]
        assert ys == sorted(ys)
    assert "| gdc test | GW-E2E-007 | glypharis (e2e) |" in _rows(log)[0]


def test_oracle_reads_box_lines_only_and_no_scan_text_leaves_the_prediction_file(root, capsys):
    data, log, _ = root
    _run(data, "oracle", "gdc_scan_test", "GW-E2E-008")
    out = _out(data, "GW-E2E-008", "gdc_scan_test")
    (rec,) = _records(out)
    assert [ln["line_id"] for ln in rec["lines"]] == ["scan0/l0", "scan0/l1", "scan0/l2"]
    assert all(ln["det_score"] is None and len(ln["quad"]) == 4 for ln in rec["lines"])
    assert "glypharis (oracle)" in _rows(log)[0]
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["task"] == "e2e" and meta["mode"] == "oracle"
    # GDC-scan text stays in the git-ignored prediction file's references at most: never in the
    # metadata, the committed manifest, the access log or the console.
    _run(data, "rec", "gdc_scan_test", "GW-REC-039")
    assert MARKER in _out(data, "GW-REC-039", "gdc_scan_test").read_text(encoding="utf-8")
    for path in (out.with_suffix(".meta.json"), data / ext_eval.OUTPUTS_MANIFEST, log,
                 _out(data, "GW-REC-039", "gdc_scan_test").with_suffix(".meta.json")):  # fmt: skip
        assert MARKER not in path.read_text(encoding="utf-8")
    assert MARKER not in capsys.readouterr().out


def test_validation_runs_log_nothing_and_no_metric_is_written_anywhere(root):
    data, log, _ = root
    before = log.read_text(encoding="utf-8")
    _run(data, "rec", "hiertext_val", "GW-REC-038")
    _run(data, "e2e", "hiertext_val", "GW-E2E-006")
    assert log.read_text(encoding="utf-8") == before
    _run(data, "oracle", "gdc_test", "GW-E2E-008")
    written = list((data / "processed/phase14").rglob("*.json*"))
    assert len(written) == 6
    for path in written:
        docs = (_records(path) if path.suffix == ".jsonl"
                else [json.loads(path.read_text(encoding="utf-8"))])  # fmt: skip
        assert not METRIC_KEYS & set(_keys(docs)), path.name
    manifest = json.loads((data / ext_eval.OUTPUTS_MANIFEST).read_text(encoding="utf-8"))
    assert not METRIC_KEYS & set(_keys(manifest))
    assert [e["file"].split("/")[-2] for e in manifest["outputs"]] == [
        "GW-E2E-006", "GW-E2E-008", "GW-REC-038"]  # fmt: skip


# ---------------------------------------------------------------------------- resume, rerun


def test_a_crash_resumes_as_the_same_evaluation_and_a_rerun_does_nothing(root, monkeypatch):
    data, log, _ = root
    frozen = tiny_frozen()
    read = frozen.recognizer.read
    n = {"calls": 0}

    def crashing(crops, temperatures=()):
        n["calls"] += 1
        if n["calls"] == 2:
            raise RuntimeError("simulated crash")
        return read(crops, temperatures)

    monkeypatch.setattr(frozen.recognizer, "read", crashing)
    monkeypatch.setattr(ext_eval, "load_frozen", lambda device, **kw: frozen)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(data, "rec", "gdc_test", "GW-REC-039", "--batch", "2")
    out = _out(data, "GW-REC-039", "gdc_test")
    assert len(_records(out)) == 2 and not (data / ext_eval.OUTPUTS_MANIFEST).exists()
    _run(data, "rec", "gdc_test", "GW-REC-039", "--batch", "2")
    recs = _records(out)
    assert len({r["key"] for r in recs}) == len(recs) == 6
    assert [r["seg"] for r in recs] == [0, 0, 1, 1, 1, 1]
    rows = _rows(log)
    assert len(rows) == 2 and "(resumed: 2 of 6 samples already predicted) |" in rows[1]
    meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert [r["status"] for r in meta["runs"]] == ["running", "done"] and meta["samples"] == 6
    # Each segment keeps its own provenance and batch; the identity holds the scorer's RUN_ENV.
    assert all(r["provenance"] == frozen.provenance and r["batch"] == 2 for r in meta["runs"])
    assert set(ext_eval.RUN_ENV) <= set(meta["identity"]["model"])
    assert meta["identity"]["batch"] == 2

    # Complete: no model load, no log row, the manifest entry is kept as it is.
    monkeypatch.setattr(ext_eval, "load_frozen", lambda *a, **k: pytest.fail("model loaded"))
    manifest = (data / ext_eval.OUTPUTS_MANIFEST).read_text(encoding="utf-8")
    assert _run(data, "rec", "gdc_test", "GW-REC-039") == 0
    assert len(_rows(log)) == 2
    assert (data / ext_eval.OUTPUTS_MANIFEST).read_text(encoding="utf-8") == manifest
    # A recorded file that changed afterwards is refused rather than recorded again.
    out.write_text(out.read_text(encoding="utf-8").replace("gdc0/l0", "gdc0/l0", 1) + "\n",
                   encoding="utf-8")  # fmt: skip
    with out.open("a", encoding="utf-8") as fh:
        fh.write(" \n")
    with pytest.raises(SystemExit, match="changed after it was recorded"):
        _run(data, "rec", "gdc_test", "GW-REC-039")


def test_a_resume_with_another_model_or_setting_is_refused_before_logging(root, monkeypatch):
    data, log, _ = root
    frozen = tiny_frozen()

    def crash(crops, temperatures=()):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(frozen.recognizer, "read", crash)
    monkeypatch.setattr(ext_eval, "load_frozen", lambda device, **kw: frozen)
    out = _out(data, "GW-REC-039", "gdc_test")
    out.parent.mkdir(parents=True)
    with pytest.raises(RuntimeError):
        _run(data, "rec", "gdc_test", "GW-REC-039")
    with out.open("a", encoding="utf-8") as fh:  # one record, as if the first chunk had landed
        fh.write(json.dumps({"key": "gdc0/l0", "ref": "alpha 0", "pred": "", "seg": 0}) + "\n")
    other = tiny_frozen()
    other.provenance["recognizer_checkpoint_sha256"] = "9" * 64
    monkeypatch.setattr(ext_eval, "load_frozen", lambda device, **kw: other)
    rows = len(_rows(log))
    with pytest.raises(SystemExit, match="a resume must be the same evaluation"):
        _run(data, "rec", "gdc_test", "GW-REC-039")
    assert len(_rows(log)) == rows


@pytest.mark.parametrize("change", ["batch", "cudnn", "device"])
def test_a_resume_with_another_batch_or_environment_is_refused(root, monkeypatch, change):
    data, log, _ = root
    frozen = tiny_frozen()
    read = frozen.recognizer.read
    n = {"calls": 0}

    def crashing(crops, temperatures=()):
        n["calls"] += 1
        if n["calls"] == 2:
            raise RuntimeError("simulated crash")
        return read(crops, temperatures)

    monkeypatch.setattr(frozen.recognizer, "read", crashing)
    monkeypatch.setattr(ext_eval, "load_frozen", lambda device, **kw: frozen)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(data, "rec", "gdc_test", "GW-REC-039", "--batch", "2")
    other = tiny_frozen()
    batch = "2"
    if change == "batch":
        batch = "3"
    else:
        other.provenance[change] = "9.1" if change == "cudnn" else "cpu:1"
    monkeypatch.setattr(ext_eval, "load_frozen", lambda device, **kw: other)
    rows = len(_rows(log))
    with pytest.raises(SystemExit, match="a resume must be the same evaluation"):
        _run(data, "rec", "gdc_test", "GW-REC-039", "--batch", batch)
    assert len(_rows(log)) == rows


# --------------------------------------------------------------------------------- refusals


@pytest.mark.parametrize(
    ("mode", "eval_set", "experiment", "message"),
    [
        ("rec", "gdc_test", "GW-REC-038", "runs on hiertext_val"),
        ("rec", "hiertext_val", "GW-REC-039", "not hiertext_val"),
        ("rec", "gdc_test", "GW-E2E-007", "not a rec experiment"),
        ("e2e", "iiit5k_test", "GW-E2E-007", "not iiit5k_test"),
        ("e2e", "hiertext_val", "GW-E2E-007", "not hiertext_val"),
        ("oracle", "hiertext_test", "GW-E2E-008", "not hiertext_test"),
        ("oracle", "gdc_test", "GW-E2E-007", "not a oracle experiment"),
    ],
)
def test_runs_outside_the_registered_plan_are_refused(root, mode, eval_set, experiment, message):
    data, log, calls = root
    with pytest.raises(SystemExit, match=message):
        _run(data, mode, eval_set, experiment)
    assert not _rows(log) and calls["load"] == 0


def test_test_sets_need_the_tag_the_branch_and_the_registered_device(root, monkeypatch):
    data, log, calls = root
    with pytest.raises(SystemExit, match="--allow-unpinned is for validation runs"):
        _run(data, "rec", "gdc_test", "GW-REC-039", "--allow-unpinned")
    monkeypatch.setattr(ext_eval, "TEST_DEVICE", "cuda")
    with pytest.raises(SystemExit, match="Phase 14 passes run on cuda"):
        _run(data, "rec", "gdc_test", "GW-REC-039")
    with pytest.raises(SystemExit, match="Phase 14 passes run on cuda"):  # validation too (H5′)
        _run(data, "rec", "hiertext_val", "GW-REC-038")
    # The device is named exactly as the record will name it: "cuda:0" would be a second
    # environment for the same GPU, and the scorer would void every run.
    with pytest.raises(SystemExit, match="named exactly so"):
        ext_eval.main(["rec", "--set", "gdc_test", "--experiment", "GW-REC-039", "--reason", "r",
                       "--device", "cuda:0", "--data-root", str(data)])  # fmt: skip
    monkeypatch.setattr(ext_eval, "TEST_DEVICE", "cpu")

    def off_branch(name, start=None):
        raise RuntimeError(f"this run must start on branch {name}; HEAD is on main")

    monkeypatch.setattr(ext_eval, "require_branch", off_branch)
    with pytest.raises(SystemExit, match="HEAD is on main"):
        _run(data, "rec", "gdc_test", "GW-REC-039")
    monkeypatch.setattr(ext_eval, "require_branch", lambda name, start=None: None)

    def unpinned(tag=None, root=None):
        raise RuntimeError("HEAD differs from phase14-eval under glypharis: ['x.py']")

    monkeypatch.setattr(ext_eval, "pinned_code_check", unpinned)
    with pytest.raises(SystemExit, match="pinned code check failed"):
        _run(data, "e2e", "gdc_test", "GW-E2E-007")
    assert not _rows(log) and calls["load"] == 0
    assert not (data / "processed/phase14").exists()


def test_validation_runs_need_the_tag_unless_a_test_allows_it(root, monkeypatch):
    data, _, _ = root

    def unpinned(tag=None, root=None):
        raise RuntimeError("the tag phase14-eval does not exist; the code is not pinned")

    monkeypatch.setattr(ext_eval, "pinned_code_check", unpinned)
    with pytest.raises(SystemExit, match="pinned code check failed"):
        _run(data, "rec", "hiertext_val", "GW-REC-038")
    _run(data, "rec", "hiertext_val", "GW-REC-038", "--allow-unpinned")
    meta = json.loads(_out(data, "GW-REC-038", "hiertext_val").with_suffix(".meta.json")
                      .read_text(encoding="utf-8"))  # fmt: skip
    assert meta["pinned"]["trees"] == "not pinned" and meta["pinned"]["allow_unpinned"]
    assert "does not exist" in meta["pinned"]["error"]


def test_the_split_comes_from_the_records(root):
    data, log, calls = root
    spec = EVAL_SETS["gdc_test"]
    _pack(data / spec["pack"], [("gdc0/l0", "a"), ("gdc0/l1", "b")], "validation", "gdc")
    with pytest.raises(SystemExit, match=r"records name splits \['validation'\]"):
        _run(data, "rec", "gdc_test", "GW-REC-039")
    assert not _rows(log) and calls["load"] == 0


def test_inputs_must_match_their_recorded_hash_and_count(root):
    data, log, calls = root
    manifest = data / "manifests/processed/eval_crops.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    spec = EVAL_SETS["gdc_test"]
    good = json.loads((data / f"{spec['pack']}.meta.json").read_text(encoding="utf-8"))
    entry = {"pages": spec["pages"], "pages_sha256": sha256_file(data / spec["pages"]),
             "pack": spec["pack"], "bin_sha256": "0" * 64, "count": good["count"]}  # fmt: skip
    manifest.write_text(json.dumps({"gdc_test": entry}), encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to run"):
        _run(data, "rec", "gdc_test", "GW-REC-039")
    manifest.write_text(json.dumps({"gdc_test": {**entry, "bin_sha256": good["bin_sha256"],
                                                 "count": 99}}), encoding="utf-8")  # fmt: skip
    with pytest.raises(SystemExit, match="holds 6 items"):
        _run(data, "rec", "gdc_test", "GW-REC-039")
    assert not _rows(log) and calls["load"] == 0
    manifest.write_text(json.dumps({"gdc_test": {**entry, "bin_sha256": good["bin_sha256"]}}),
                        encoding="utf-8")  # fmt: skip
    _run(data, "rec", "gdc_test", "GW-REC-039")
    meta = json.loads(_out(data, "GW-REC-039", "gdc_test").with_suffix(".meta.json")
                      .read_text(encoding="utf-8"))  # fmt: skip
    assert meta["inputs"]["pack"]["recorded_in"] == "data/manifests/processed/eval_crops.json"


def test_an_input_without_a_recorded_hash_is_refused(root, monkeypatch):
    data, log, calls = root
    monkeypatch.setattr(ext_eval, "REQUIRE_RECORDED", True)
    with pytest.raises(SystemExit, match="no recorded hash"):
        _run(data, "rec", "iiit5k_test", "GW-REC-039")
    assert not _rows(log) and calls["load"] == 0


def test_gdc_scan_pages_are_checked_against_their_freeze_record(root, monkeypatch):
    data, log, calls = root
    monkeypatch.setattr(ext_eval, "REQUIRE_RECORDED", True)
    pages = EVAL_SETS["gdc_scan_test"]["pages"]
    scan = data / GDC_SCAN_MANIFEST
    scan.parent.mkdir(parents=True, exist_ok=True)
    scan.write_text(json.dumps({"pages": 1, "pages_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="refusing to run"):
        _run(data, "e2e", "gdc_scan_test", "GW-E2E-007")
    assert not _rows(log) and calls["load"] == 0
    doc = {"pages": 1, "pages_sha256": sha256_file(data / pages)}
    scan.write_text(json.dumps(doc), encoding="utf-8")
    _run(data, "e2e", "gdc_scan_test", "GW-E2E-007")
    assert len(_rows(log)) == 1


def test_a_phase_14_run_needs_the_phase_5_hashes_first(root):
    data, log, calls = root
    (data / PHASE5_HASHES).unlink()
    with pytest.raises(SystemExit, match="hash-phase5"):
        _run(data, "rec", "gdc_test", "GW-REC-039")
    assert not _rows(log) and calls["load"] == 0


@pytest.mark.parametrize("fault", ["missing", "twice"])
def test_a_crop_without_one_page_line_is_refused_before_the_access(root, fault):
    data, log, calls = root
    spec = EVAL_SETS["gdc_test"]
    pages = list(read_pages(data / spec["pages"]))
    if fault == "missing":
        pages[0].lines = [ln for ln in pages[0].lines if ln.line_id != "gdc0/l1"]
    else:
        pages[1].lines.append(dataclasses.replace(pages[1].lines[0]))
    write_pages(data / spec["pages"], pages)
    with pytest.raises(SystemExit, match="have no line" if fault == "missing" else "twice"):
        _run(data, "rec", "gdc_test", "GW-REC-039")
    assert not _rows(log) and calls["load"] == 0
    assert not _out(data, "GW-REC-039", "gdc_test").with_suffix(".meta.json").exists()


def test_oracle_refuses_a_duplicated_line_id_before_the_access(root):
    data, log, calls = root
    spec = EVAL_SETS["gdc_test"]
    pages = list(read_pages(data / spec["pages"]))
    pages[1].lines.append(dataclasses.replace(pages[0].lines[0]))
    write_pages(data / spec["pages"], pages)
    with pytest.raises(SystemExit, match="1 line ids appear more than once"):
        _run(data, "oracle", "gdc_test", "GW-E2E-008")
    assert not _rows(log) and calls["load"] == 0
    assert not _out(data, "GW-E2E-008", "gdc_test").with_suffix(".meta.json").exists()


@pytest.mark.parametrize(("mode", "experiment"), [("e2e", "GW-E2E-007"), ("oracle", "GW-E2E-008")])
def test_a_page_image_of_another_size_is_refused_before_the_access(root, mode, experiment):
    data, log, calls = root
    spec = EVAL_SETS["gdc_scan_test"]
    (page,) = read_pages(data / spec["pages"])
    Image.new("RGB", (PAGE_SIZE[1], PAGE_SIZE[0])).save(data / "raw/other.png")
    write_pages(data / spec["pages"], [dataclasses.replace(page, image="raw/other.png")])
    with pytest.raises(SystemExit, match=r"raw/other.png: image is 300x400") as err:
        _run(data, mode, "gdc_scan_test", experiment)
    assert MARKER not in str(err.value)
    assert not _rows(log) and calls["load"] == 0
    assert not _out(data, experiment, "gdc_scan_test").with_suffix(".meta.json").exists()


def test_a_pack_without_a_recorded_index_hash_is_refused(root, monkeypatch):
    data, log, calls = root
    spec = EVAL_SETS["iiit5k_test"]
    manifest = data / "manifests/processed/benchmarks.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    good = json.loads((data / f"{spec['pack']}.meta.json").read_text(encoding="utf-8"))
    entry = {"pack": spec["pack"], "bin_sha256": good["bin_sha256"], "count": good["count"]}
    manifest.write_text(json.dumps({"iiit5k_test": entry}), encoding="utf-8")
    monkeypatch.setattr(ext_eval, "REQUIRE_RECORDED", True)
    with pytest.raises(SystemExit, match="no recorded index_sha256"):
        _run(data, "rec", "iiit5k_test", "GW-REC-039")
    index = sha256_file(data / f"{spec['pack']}.index.jsonl")
    manifest.write_text(json.dumps({"iiit5k_test": {**entry, "index_sha256": "0" * 64}}),
                        encoding="utf-8")  # fmt: skip
    with pytest.raises(SystemExit, match=r"\.index\.jsonl has SHA-256"):
        _run(data, "rec", "iiit5k_test", "GW-REC-039")
    assert not _rows(log) and calls["load"] == 0
    manifest.write_text(json.dumps({"iiit5k_test": {**entry, "index_sha256": index}}),
                        encoding="utf-8")  # fmt: skip
    _run(data, "rec", "iiit5k_test", "GW-REC-039")
    assert len(_rows(log)) == 1


def test_a_refused_model_load_logs_nothing(root, monkeypatch):
    data, log, _ = root

    def refused(device, **kw):
        raise FrozenModelError("recognizer checkpoint has SHA-256 x, the record says y")

    monkeypatch.setattr(ext_eval, "load_frozen", refused)
    with pytest.raises(SystemExit, match="frozen model refused"):
        _run(data, "e2e", "gdc_test", "GW-E2E-007")
    assert not _rows(log)
    assert not _out(data, "GW-E2E-007", "gdc_test").with_suffix(".meta.json").exists()


# ------------------------------------------------------------------------------- attributes


def test_line_geometry_and_crop_attributes():
    h, a = ext_eval.line_geometry(rect_polygon(100, 50, 120, 20))
    assert h == pytest.approx(20) and a == pytest.approx(0)
    h, a = ext_eval.line_geometry(rect_polygon(100, 50, 120, 20, angle_deg=10))
    assert h == pytest.approx(20) and a == pytest.approx(10)
    # A densified line (top edge left to right, bottom edge back) of height 10.
    top = [[x, 0.0] for x in (0, 50, 100)]
    bottom = [[x, 10.0] for x in (100, 50, 0)]
    assert ext_eval.line_geometry(top + bottom) == (10.0, 0.0)
    assert ext_eval.line_geometry([[0, 0], [1, 1], [2, 0]]) == (None, None)
    flat = ext_eval.crop_attributes(Image.new("L", (40, 16), 200))
    assert flat == {"crop_height_px": 16, "blur": 0.0, "contrast": 0.0}
    stripes = Image.fromarray(np.tile(np.array([0, 255], np.uint8), (16, 20)))
    sharp = ext_eval.crop_attributes(stripes)
    assert sharp["blur"] > 1000 and sharp["contrast"] == pytest.approx(127.5)
