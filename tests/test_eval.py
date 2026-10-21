from __future__ import annotations

import numpy as np
import pytest

from glypharis.data.pages import Page, PageLine
from glypharis.eval.access import HeldOutAccessError, open_test_access
from glypharis.eval.page_metrics import PredLine, det_summary, evaluate_page
from glypharis.eval.stats import mean_ci, paired_ratio_diff_ci, ratio_ci
from glypharis.eval.text_metrics import RecAccumulator, score_pair

# ------------------------------------------------------------------------------ text metrics


def test_score_pair_hand_computed():
    s = score_pair("helo world", "hello world")
    assert (s.char_errors, s.ref_chars, s.word_errors, s.ref_words) == (1, 11, 1, 2)
    assert s.ned == pytest.approx(1 - 1 / 11)
    assert not s.exact
    assert score_pair("same", "same").exact


def test_tracks_differ_on_case_and_quotes():
    assert score_pair("“Hello”", '"hello"', "strict").char_errors == 3
    assert score_pair("“Hello”", '"hello"', "normalised").char_errors == 0


def test_accumulator_micro_average_and_empty_refs():
    acc = RecAccumulator()
    acc.add("abc", "abd")  # 1 / 3
    acc.add("", "xyzw")  # 4 / 4
    acc.add("anything", "   ")  # empty reference: counted, not scored
    s = acc.summary()
    assert s.samples == 2 and s.empty_references == 1
    assert s.cer == pytest.approx(5 / 7)
    assert s.sequence_accuracy == 0.0


# ------------------------------------------------------------------------------- statistics


def test_ratio_ci_brackets_the_value_and_is_reproducible():
    rng = np.random.default_rng(0)
    den = rng.integers(5, 50, size=400).astype(float)
    num = rng.binomial(den.astype(int), 0.1).astype(float)
    a = ratio_ci(num, den, "lines")
    assert a.low <= a.value <= a.high and a.value == pytest.approx(num.sum() / den.sum())
    assert a == ratio_ci(num, den, "lines")
    assert mean_ci(np.ones(10), "x").low == 1.0


def test_paired_diff_detects_a_real_difference():
    den = np.full(300, 20.0)
    a = np.full(300, 4.0)  # CER 0.2
    b = np.full(300, 2.0)  # CER 0.1
    d = paired_ratio_diff_ci(a, b, den, "lines")
    assert d.value == pytest.approx(0.1) and d.low > 0


# ------------------------------------------------------------------------------ page metrics


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _line(i, box, text, legible=True, vertical=False, score=None):
    return PageLine(f"p/l{i}", box, text, text, legible, False, vertical, True, None, score)


def _page():
    lines = [
        _line(0, _box(0, 0, 100, 20), "first line"),
        _line(1, _box(0, 30, 100, 50), "second line"),
        _line(2, _box(0, 60, 100, 80), "smudge", legible=False),  # don't care
        _line(3, _box(0, 90, 100, 110), "wobbly", score=0.1),  # box not trusted
    ]
    return Page("p", "img.jpg", 200, 200, "test", "test", lines, ignore=[_box(150, 0, 200, 200)])


def test_perfect_prediction():
    page = _page()
    preds = [PredLine(ln.polygon, ln.text) for ln in page.lines if ln.legible]
    r = evaluate_page(page, preds)
    assert r.det[0.5] == (2, 0, 0)  # two trusted lines found; "wobbly" is don't care for boxes
    assert r.aware_errors == 0 and r.bag_errors == 0
    assert (r.e2e_tp, r.e2e_gt) == (2, 2)


def test_misses_false_positives_and_ignore_regions():
    page = _page()
    preds = [
        PredLine(_box(0, 0, 100, 20), "first lime"),  # TP, one character wrong
        PredLine(_box(0, 60, 100, 80), "smudge"),  # matches a don't-care line: ignored
        PredLine(_box(160, 10, 190, 30), "logo"),  # inside the ignore region: ignored
        PredLine(_box(0, 150, 100, 170), "ghost"),  # a real false positive
    ]
    r = evaluate_page(page, preds)
    assert r.det[0.5] == (1, 1, 1)
    # bag of lines: 1 substitution + 5 inserted ("ghost") + 11 deleted ("second line")
    assert r.bag_errors == 1 + 5 + 11 and r.bag_ref_chars == len("first line") + len("second line")
    assert r.e2e_tp == 1 and r.e2e_pred == 2
    s = det_summary([r], 0.5)
    assert s.precision == 0.5 and s.recall == 0.5


def test_iou_threshold_matters():
    page = _page()
    shifted = PredLine(_box(0, 8, 100, 28), "first line")  # IoU 12/28 ~ 0.43 with line 0
    r = evaluate_page(page, [shifted], thresholds=(0.3, 0.5))
    assert r.det[0.3][0] == 1 and r.det[0.5][0] == 0


# ------------------------------------------------------------------------------ test access


def test_test_access_is_guarded_and_logged(tmp_path):
    log = tmp_path / "log.md"
    log.write_text(
        "| Date | ... |\n|---|\n| — | — | — | — | — | No test access yet. |\n", encoding="utf-8"
    )
    open_test_access("gdc", "validation", None, "-", None, "", log)  # not a test split: no-op
    assert "No test access yet" in log.read_text(encoding="utf-8")
    with pytest.raises(HeldOutAccessError, match="experiment ID"):
        open_test_access("gdc", "test", None, "-", 5, "why", log)
    with pytest.raises(HeldOutAccessError, match="phases"):
        open_test_access("gdc", "test", "GW-BASE-001", "-", 6, "why", log)
    with pytest.raises(HeldOutAccessError, match="reason"):
        open_test_access("gdc", "test", "GW-BASE-001", "-", 5, " ", log)
    open_test_access("gdc", "test", "GW-BASE-001", "tesseract-5", 5, "baseline", log)
    open_test_access("hiertext", "test", "GW-BASE-002", "ppocr", 5, "baseline", log)
    text = log.read_text(encoding="utf-8")
    assert "No test access yet" not in text
    assert "| gdc test | GW-BASE-001 | tesseract-5 | 5 | baseline |" in text
    assert text.count("GW-BASE-00") == 2
