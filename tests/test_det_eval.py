from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import run_baselines
from det_fixtures import make_line, rect_polygon
from PIL import Image

from glypharis.data.pages import Page, PageLine, write_pages
from glypharis.eval import det_slices
from glypharis.eval.det_matching import (
    BASELINE_VAL_PAGES,
    baseline_doc_ids,
    det_cis,
    detection_matches,
    fast_det_counts,
)
from glypharis.eval.det_overlay import GT_FN, PRED_FP, PRED_TP, contact_sheet, draw_overlay
from glypharis.eval.det_slices import (
    PAGE_SLICES,
    load_or_build_slices,
    page_slices,
    slice_cache_path,
)
from glypharis.eval.page_metrics import PredLine, evaluate_page, is_box_line
from glypharis.eval.stats import paired_ratios_diff_ci

THRESHOLDS = (0.5, 0.7, 0.3)

# --------------------------------------------------------------------------- synthetic pages


def _jitter(rng: np.random.Generator, poly: list[list[float]]) -> list[list[float]]:
    """A prediction near a line: shifted, scaled and turned, so IoUs straddle 0.3, 0.5 and 0.7."""
    pts = np.array(poly)
    c = pts.mean(axis=0)
    a = np.radians(rng.uniform(-6, 6))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = (pts - c) @ rot.T * rng.uniform(0.75, 1.25, 2) + c + rng.uniform(-6, 6, 2)
    return out.tolist()


def _random_page(rng: np.random.Generator, i: int) -> tuple[Page, list[PredLine]]:
    """Care and don't-care lines, sometimes an ignore region, and predictions of every kind:
    near lines, duplicated, on don't-care lines, inside ignore regions and spurious."""
    w, h = 800, 600
    lines: list[PageLine] = []
    for k in range(int(rng.integers(0, 25))):
        poly = rect_polygon(rng.uniform(0, w), rng.uniform(0, h), rng.uniform(20, 200),
                            rng.uniform(6, 30), rng.uniform(-30, 30))  # fmt: skip
        kind = rng.choice(["care"] * 4 + ["illegible", "handwritten", "vertical", "empty", "align"])
        flags: dict[str, bool | float | None] = {
            "illegible": {"legible": False},
            "handwritten": {"handwritten": True},
            "vertical": {"vertical": True},
            "align": {"align_score": 0.1},
        }.get(str(kind), {})
        lines.append(make_line(f"l{k}", poly, "" if kind == "empty" else "text", **flags))
    ignore = []
    if rng.uniform() < 0.3:
        ignore.append(rect_polygon(rng.uniform(0, w), rng.uniform(0, h), 150, 80))
    preds: list[PredLine] = []
    for ln in lines:
        if rng.uniform() < 0.85:
            preds.append(PredLine(_jitter(rng, ln.polygon)))
        if rng.uniform() < 0.2:  # duplicates: exact and near
            preds.append(PredLine(ln.polygon if rng.uniform() < 0.5 else _jitter(rng, ln.polygon)))
    for region in ignore:
        preds.append(PredLine(_jitter(rng, rect_polygon(*np.mean(region, axis=0), 60, 20))))
    for _ in range(int(rng.integers(0, 6))):
        preds.append(PredLine(rect_polygon(rng.uniform(0, w), rng.uniform(0, h),
                                           rng.uniform(10, 150), rng.uniform(5, 30))))  # fmt: skip
    order = rng.permutation(len(preds))
    page = Page(f"p{i}", "x.jpg", w, h, "detsynth", "validation", lines, ignore)
    return page, [preds[int(j)] for j in order]


@pytest.fixture(scope="module")
def pages() -> list[tuple[Page, list[PredLine]]]:
    rng = np.random.default_rng(20261023)
    return [_random_page(rng, i) for i in range(200)]


# -------------------------------------------------------------------------- counting parity


def test_pages_cover_every_case(pages):
    lines = [ln for page, _ in pages for ln in page.lines]
    assert any(is_box_line(ln) for ln in lines) and any(not is_box_line(ln) for ln in lines)
    assert sum(1 for page, _ in pages if page.ignore) > 20
    assert any(not page.lines for page, _ in pages)


def test_detection_matches_counts_equal_evaluate_page(pages):
    for page, preds in pages:
        ref = evaluate_page(page, preds, (0.5, 0.7), score_text=False)
        for thr in (0.5, 0.7):
            m = detection_matches(page, preds, thr)
            assert m.counts() == ref.det[thr], (page.doc_id, thr)
            assert len(m.pred_status) == len(preds) and len(m.gt_status) == len(page.lines)


def test_detection_matches_details_are_consistent(pages):
    for page, preds in pages:
        m = detection_matches(page, preds, 0.5)
        pairs = {(p, g): iou for p, g, iou, _, _ in m.overlaps}
        for p, (status, g, iou) in enumerate(zip(m.pred_status, m.pred_gt, m.pred_iou,
                                                 strict=True)):  # fmt: skip
            if g is None:
                assert status in ("fp", "ignored") and iou == 0.0
                continue
            assert iou >= 0.5 and pairs[(p, g)] == iou
            if status == "tp":
                assert is_box_line(page.lines[g]) and m.gt_status[g] == "tp"
            else:
                assert status == "ignored" and m.gt_status[g] == "dont_care"
        assert m.gt_status.count("tp") == m.pred_status.count("tp")
        for ln, status in zip(page.lines, m.gt_status, strict=True):
            assert (status == "dont_care") == (not is_box_line(ln))
        for _, _, iou, in_pred, in_gt in m.overlaps:
            assert 0 < iou <= min(in_pred, in_gt) + 1e-12 and in_pred <= 1 + 1e-9


def test_fast_det_counts_equal_evaluate_page(pages):
    for page, preds in pages:
        ref = evaluate_page(page, preds, THRESHOLDS, score_text=False)
        fast = fast_det_counts(page, preds, THRESHOLDS)
        assert fast.det == ref.det, page.doc_id
        assert abs(fast.matched_iou_sum - ref.matched_iou_sum) <= 1e-9
        assert fast.bag_errors == fast.e2e_tp == fast.aware_errors == 0


def test_fast_det_counts_with_duplicates_dont_care_and_ignore_regions():
    care = rect_polygon(100, 50, 120, 20)
    dont = rect_polygon(100, 150, 120, 20)
    page = Page("d", "x.jpg", 400, 300, "detsynth", "validation",
                [make_line("c", care), make_line("u", dont, legible=False)],
                [rect_polygon(300, 250, 100, 60)])  # fmt: skip
    in_ignore, stray = rect_polygon(300, 250, 40, 20), rect_polygon(300, 50, 40, 20)
    preds = [PredLine(p) for p in (care, care, dont, dont, in_ignore, stray)]
    ref = evaluate_page(page, preds, THRESHOLDS, score_text=False)
    # fp: the second care copy and the stray box. The spare don't-care copy lies on don't-care
    # text and the small box inside the ignore region: both ignored.
    assert ref.det[0.5] == (1, 2, 0)
    assert fast_det_counts(page, preds, THRESHOLDS).det == ref.det
    m = detection_matches(page, preds, 0.5)
    assert sorted(m.pred_status[:2]) == ["fp", "tp"]
    assert m.pred_status[2:] == ["ignored", "ignored", "ignored", "fp"]
    assert m.gt_status == ["tp", "dont_care"]


def test_pruned_counting_is_fast_on_noisy_maps():
    rng = np.random.default_rng(7)
    lines, preds = [], []
    for k in range(500):
        r, c = divmod(k, 20)
        cx, cy = 50 + c * 100, 30 + r * 40
        lines.append(make_line(f"l{k}", rect_polygon(cx, cy, 80, 16)))
        for _ in range(2):  # a few boxes per line that overlap it well
            preds.append(PredLine(rect_polygon(cx + rng.uniform(-4, 4), cy + rng.uniform(-2, 2),
                                               80 * rng.uniform(0.9, 1.1), 16)))  # fmt: skip
    for x, y in rng.uniform(0, 1, (5000, 2)) * (2000, 1000):
        preds.append(PredLine([[x, y], [x + 3, y], [x + 3, y + 3], [x, y + 3]]))
    page = Page("noisy", "x.jpg", 2000, 1000, "detsynth", "validation", lines)
    start = time.perf_counter()
    res = fast_det_counts(page, preds, THRESHOLDS)
    assert time.perf_counter() - start < 2.0
    tp, fp, fn = res.det[0.5]
    assert tp + fn == 500 and tp >= 490 and fp >= 5000


# ----------------------------------------------------------------- confidence intervals, CIs


def test_det_cis_are_the_micro_ratios():
    rng = np.random.default_rng(8)
    tp = rng.integers(0, 30, 200)
    fp, fn = rng.integers(0, 5, 200), rng.integers(0, 5, 200)
    cis = det_cis(tp, fp, fn)
    assert set(cis) == {"precision", "recall", "f1"}
    assert cis["precision"].value == pytest.approx(tp.sum() / (tp.sum() + fp.sum()))
    assert cis["recall"].value == pytest.approx(tp.sum() / (tp.sum() + fn.sum()))
    p, r = cis["precision"].value, cis["recall"].value
    assert cis["f1"].value == pytest.approx(2 * p * r / (p + r))
    for ci in cis.values():
        assert ci.low <= ci.value <= ci.high and ci.unit == "pages"


def test_paired_ratios_diff_ci_brackets_a_known_difference():
    rng = np.random.default_rng(9)
    den = rng.integers(10, 60, 300).astype(float)
    num_b = rng.binomial(den.astype(int), 0.6).astype(float)
    exact = paired_ratios_diff_ci(num_b + 0.1 * den, den, num_b, den, "pages")
    assert exact.value == pytest.approx(0.1) and exact.low == pytest.approx(0.1)
    assert exact.high == pytest.approx(0.1)
    num_a = rng.binomial(den.astype(int), 0.8).astype(float)
    ci = paired_ratios_diff_ci(num_a, den, num_b, den * 1.1, "pages")
    expected = num_a.sum() / den.sum() - num_b.sum() / (1.1 * den.sum())
    assert ci.value == pytest.approx(expected) and 0 < ci.low <= ci.value <= ci.high
    same = paired_ratios_diff_ci(num_a, den, num_a, den, "pages")
    assert same.value == 0.0 and same.low <= 0.0 <= same.high


# -------------------------------------------------------------------------- Phase 5 page set


@pytest.mark.parametrize("n", [1715, 100])
def test_baseline_doc_ids_match_run_baselines(n):
    ids = [f"doc{v:05d}" for v in np.random.default_rng(n).permutation(n)]  # file order
    expected = [ids[int(i)] for i in run_baselines.page_indices(n, "validation")]
    assert baseline_doc_ids(ids) == expected
    assert len(expected) == min(n, BASELINE_VAL_PAGES)


def test_baseline_doc_ids_match_expected_keys(tmp_path):
    order = np.random.default_rng(10).permutation(450)
    pages = [Page(f"d{v:04d}", f"raw/hiertext/images/validation/d{v:04d}.jpg", 10, 10,
                  "hiertext", "validation") for v in order]  # fmt: skip
    write_pages(tmp_path / "processed/hiertext/pages_validation.jsonl.gz", pages)
    keys = run_baselines.expected_keys(tmp_path, "hiertext_val", "e2e")
    assert keys == baseline_doc_ids([p.doc_id for p in pages])
    assert len(keys) == BASELINE_VAL_PAGES


# ---------------------------------------------------------------------------------- slices


# Slice pages are laid out at 400x300, the inference scale of the tests (long side 400), and
# stored at 1024x768, the edge-density scale, so the background noise is measured unresampled.
LAYOUT_TO_PAGE = 1024 / 400


def _write_page_image(path: Path, page: Page, noise_sigma: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = (page.height, page.width)
    canvas = np.clip(235.0 + rng.normal(0, noise_sigma, shape), 0, 255).astype(np.uint8)
    for ln in page.lines:
        cv2.fillPoly(canvas, [np.round(np.array(ln.polygon)).astype(np.int32)], 30)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).convert("RGB").save(path)


def _slice_pages(root: Path) -> list[Page]:
    def page(doc_id: str, polys: list[list[list[float]]], sigma: float, seed: int,
             extra: list[PageLine] | None = None) -> Page:  # fmt: skip
        lines = [make_line(f"{doc_id}-{k}", (np.array(p) * LAYOUT_TO_PAGE).tolist())
                 for k, p in enumerate(polys)] + (extra or [])  # fmt: skip
        pg = Page(doc_id, f"raw/synth/{doc_id}.png", 1024, 768, "detsynth", "validation", lines)
        _write_page_image(root / pg.image, pg, sigma, seed)
        return pg

    stack = [rect_polygon(150, 40 + 16 * k, 200, 12) for k in range(4)]
    stack.append(rect_polygon(330, 250, 60, 12))  # a lone line
    upright = [rect_polygon(60, 50 + 20 * k, 10, 16) for k in range(3)]  # taller than wide
    rotated = [rect_polygon(200, 60 + 80 * k, 160, 20, 30.0) for k in range(3)]
    dense = [rect_polygon(20 + 38 * c, 15 + 28 * r, 30, 8) for r in range(10) for c in range(10)]
    blank = [make_line("blank-0", rect_polygon(512, 384, 256, 50), legible=False)]
    return [
        page("stack", stack, 2.0, 1),
        page("upright", upright, 4.0, 2),
        page("rotated", rotated, 6.0, 3),
        page("dense", dense, 8.0, 4),
        page("blank", [], 10.0, 5, blank),
        page("noisy", [rect_polygon(200, 150, 200, 20)], 40.0, 6),
    ]


def test_slice_rules_on_constructed_pages(tmp_path):
    pages = _slice_pages(tmp_path)
    table = load_or_build_slices(pages, "ab" * 32, tmp_path, 400)
    slices = page_slices(table)
    assert set(slices) == set(PAGE_SLICES)
    assert slices["multi_line"] == ["stack", "upright"]
    assert slices["rotated"] == ["rotated"]
    assert slices["dense"] == ["dense"]
    assert slices["small_text"] == ["dense"]
    assert "noisy" in slices["noisy_background"] and "stack" not in slices["noisy_background"]
    densities = [p.bg_edge_density for p in table.pages]
    assert table.thresholds["noisy_background"] == pytest.approx(np.quantile(densities, 0.75))

    by_page: dict[str, list[det_slices.LineAttrs]] = {}
    for la in table.lines:
        by_page.setdefault(la.doc_id, []).append(la)
    # Upright short boxes: horizontal text, tall boxes, and they can stack.
    assert all(la.tall_box and la.stacked and la.angle == pytest.approx(0.0, abs=1e-6)
               for la in by_page["upright"])  # fmt: skip
    assert [la.stacked for la in by_page["stack"]] == [True] * 4 + [False]
    assert all(la.angle == pytest.approx(30.0, abs=0.1) and not la.stacked
               for la in by_page["rotated"])  # fmt: skip
    assert "blank" not in by_page
    blank = next(p for p in table.pages if p.doc_id == "blank")
    assert blank.n_care == 0 and blank.median_height == 0.0
    stack = next(p for p in table.pages if p.doc_id == "stack")
    assert stack.stack_share == pytest.approx(0.8) and stack.median_height == pytest.approx(12.0)


def test_slices_are_cached_and_stable_across_rebuilds(tmp_path, monkeypatch):
    pages = _slice_pages(tmp_path)
    sha = "cd" * 32
    first = load_or_build_slices(pages, sha, tmp_path, 400)
    path = slice_cache_path(tmp_path, sha, 400)
    assert path.name == f"{sha[:12]}_400_v1.json"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    rebuilt = det_slices.build_slices(pages, sha, tmp_path, 400)
    assert rebuilt == first

    def refuse(*args: object) -> None:
        raise AssertionError("the cache should have been used")

    monkeypatch.setattr(det_slices, "build_slices", refuse)
    assert load_or_build_slices(pages, sha, tmp_path, 400) == first
    monkeypatch.undo()
    # A different page list does not reuse the cache.
    fewer = load_or_build_slices(pages[:3], sha, tmp_path, 400)
    assert [p.doc_id for p in fewer.pages] == ["stack", "upright", "rotated"]


def test_slices_take_blur_and_contrast_from_the_audit(tmp_path):
    pages = _slice_pages(tmp_path)
    audit = tmp_path / "interim/audit/hiertext_images.jsonl"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({"path": "data/raw/synth/stack.png", "ok": True,
                                 "blur_lapvar": 123.0, "contrast": 0.5}) + "\n",
                     encoding="utf-8")  # fmt: skip
    table = det_slices.build_slices(pages, "ef" * 32, tmp_path, 400)
    attrs = {p.doc_id: p for p in table.pages}
    assert (attrs["stack"].blur_lapvar, attrs["stack"].contrast) == (123.0, 0.5)
    assert attrs["noisy"].blur_lapvar > attrs["rotated"].blur_lapvar > 0


def test_slices_refuse_held_out_pages(tmp_path):
    held_out = Page("t", "raw/x/t.png", 10, 10, "hiertext", "test_set")
    with pytest.raises(ValueError, match="held-out"):
        det_slices.build_slices([held_out], "00" * 32, tmp_path, 400)


# --------------------------------------------------------------------------------- overlays


def test_overlay_draws_statuses_and_contact_sheet_lays_out_tiles():
    care_hit, care_missed = rect_polygon(100, 50, 120, 20), rect_polygon(100, 150, 120, 20)
    dont = rect_polygon(300, 100, 100, 20)
    page = Page("o", "x.jpg", 400, 300, "hiertext", "validation",
                [make_line("a", care_hit), make_line("b", care_missed),
                 make_line("c", dont, handwritten=True)])  # fmt: skip
    preds = [PredLine(care_hit), PredLine(rect_polygon(300, 250, 60, 20)), PredLine(dont)]
    m = detection_matches(page, preds, 0.5)
    assert m.pred_status == ["tp", "fp", "ignored"]
    img = draw_overlay(np.full((300, 400, 3), 255, np.uint8), page, preds, m, long_side=800)
    assert img.width == 800 and img.height > 600
    colours = {tuple(int(v) for v in c) for c in np.unique(np.asarray(img).reshape(-1, 3), axis=0)}
    assert {GT_FN, PRED_FP, PRED_TP} <= colours
    with pytest.raises(ValueError):
        draw_overlay(np.zeros((300, 400, 3), np.uint8), page, preds[:2], m)

    sheet = contact_sheet([img] * 4, cols=3, tile_long=200)
    assert sheet.size == (3 * 200 + 4 * 4, 2 * 200 + 3 * 4)
    assert contact_sheet([], cols=3, tile_long=200).width == 200


def test_calibration_halves_are_disjoint_complete_and_order_free():
    from glypharis.eval.det_matching import calibration_halves

    ids = [f"p{i:04d}" for i in range(101)]
    a, b = calibration_halves(ids)
    assert len(a) == 50 and len(b) == 51
    assert not set(a) & set(b) and set(a) | set(b) == set(ids)
    assert calibration_halves(list(reversed(ids))) == (a, b)
    assert a == sorted(a) and b == sorted(b)
    with pytest.raises(ValueError):
        calibration_halves(["x", "x"])
