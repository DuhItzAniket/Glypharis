from __future__ import annotations

import json

import evaluate_baselines
import numpy as np
import run_baselines

from glypharis.data.pages import Page, PageLine, write_pages

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


def test_resume_drops_a_half_written_last_line(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text('{"key": "a"}\n{"key": "b"}\n{"key": "c", "pre', encoding="utf-8")
    assert run_baselines.done_keys(path) == {"a", "b"}
    assert path.read_text(encoding="utf-8") == '{"key": "a"}\n{"key": "b"}\n'
    assert run_baselines.done_keys(tmp_path / "missing.jsonl") == set()


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _write(path, records, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    run = {"written": len(records), "wall_seconds": 1.0, "peak_rss_mib": 300.0, "batch": 2}
    meta = {**meta, "version": "x 1.0", "settings": {}, "runs": [run]}
    path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_evaluate_baselines_writes_three_tables(tmp_path):
    data = tmp_path / "data"
    lines = [
        PageLine("d/l0", _box(0, 0, 100, 20), "alpha", "alpha", True, False, False, True),
        PageLine("d/l1", _box(0, 30, 100, 50), "beta", "beta", True, False, False, True),
    ]
    write_pages(data / "processed/gdc/pages_test.jsonl.gz",
                [Page("d", "img.jpg", 200, 100, "gdc", "test", lines)])  # fmt: skip
    pack_meta = data / "processed/gdc/lines_test.meta.json"
    pack_meta.write_text(json.dumps({"count": 2}), encoding="utf-8")
    exp = data / "processed/baselines/GW-BASE-001"
    _write(exp / "tesseract_rec_gdc_test.jsonl",
           [{"key": "d/l0", "ref": "alpha", "pred": "alpha", "time_ms": 5.0},
            {"key": "d/l1", "ref": "beta", "pred": "bota", "time_ms": 5.0}],
           {"system": "tesseract", "task": "rec", "eval_set": "gdc_test",
            "experiment": "GW-BASE-001"})  # fmt: skip
    _write(exp / "tesseract_e2e_gdc_test.jsonl",
           [{"key": "d", "time_ms": 900.0,
             "lines": [{"polygon": _box(0, 0, 100, 20), "text": "alpha", "confidence": 0.9}]}],
           {"system": "tesseract", "task": "e2e", "eval_set": "gdc_test",
            "experiment": "GW-BASE-001"})  # fmt: skip
    out = tmp_path / "reports"
    assert evaluate_baselines.main(["--data-root", str(data), "--out", str(out)]) == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    rec = next(r for r in results if r["task"] == "rec")
    assert rec["tracks"]["strict"]["cer"]["value"] == 1 / 9  # one substitution in 9 characters
    e2e = next(r for r in results if r["task"] == "e2e")
    assert e2e["detection"]["iou_0.5"]["recall"]["value"] == 0.5
    assert e2e["tracks"]["strict"]["e2e_f1"]["value"] == 2 / 3
    for name in ("recognition.md", "detection.md", "end_to_end.md"):
        text = (out / name).read_text(encoding="utf-8")
        assert "GW-BASE-001" in text and "Tesseract 5" in text
    assert "## Track: normalised" in (out / "recognition.md").read_text(encoding="utf-8")
