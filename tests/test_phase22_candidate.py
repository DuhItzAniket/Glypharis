"""The phone candidate's arms and the selection rule (Phase 22, item 14)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from glypharis.efficiency import candidate as cand
from glypharis.efficiency.candidate import CandidateError


def test_arms_and_mirrored_order() -> None:
    assert cand.ARMS[0] == cand.DEFAULT_ARM == "fused"
    assert set(cand.MARGINS) == set(cand.ARMS) - {"fused"}
    assert cand.MARGINS == {"fused-int8": 0.005, "fused-single-line": 0.0005}
    order = cand.sitting_order()
    assert order[:3] == [(a, 1) for a in cand.ARMS]
    assert order[3:] == [(a, 2) for a in reversed(cand.ARMS)]
    with pytest.raises(CandidateError):
        cand.arm_order(3)
    with pytest.raises(CandidateError):
        cand.check_arm("int8-recognizer")


def test_check_fused_refuses_another_file(tmp_path: Path) -> None:
    other = tmp_path / "recognizer.fused.onnx"
    other.write_bytes(b"not the fused model")
    with pytest.raises(CandidateError, match="not GW-EFF-009"):
        cand.check_fused(other)
    with pytest.raises(CandidateError, match="no fused recognizer"):
        cand.check_fused(tmp_path / "missing.onnx")


def test_recognizer_files(tmp_path: Path) -> None:
    assert cand.recognizer_file("fused", tmp_path) == cand.PHASE16_FUSED
    assert cand.recognizer_file("fused-single-line", tmp_path) == cand.PHASE16_FUSED
    assert cand.recognizer_file("fused-int8", tmp_path) == tmp_path / "fused-int8" / cand.INT8_FILE


def test_single_line_batches_one_line_in_width_order() -> None:
    widths = [40, 8, 40, 16]
    assert cand.SingleLineRecognizer._batches(None, widths) == [[1], [3], [0], [2]]  # type: ignore[arg-type]


def _lines(errors: list[int], ref: list[int], keys: list[str] | None = None) -> dict[str, Any]:
    return {
        "per_line": {
            "keys": keys or [f"k{i}" for i in range(len(errors))],
            "strict": {"errors": errors, "ref_chars": ref},
        }
    }


def _pages(
    bag: list[int],
    ref: list[int],
    tp: list[int],
    pred: list[int],
    gt: list[int],
    ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "per_page": {
            "doc_ids": ids or [f"p{i}" for i in range(len(bag))],
            "strict": {
                "bag_errors": bag,
                "bag_ref_chars": ref,
                "e2e_tp": tp,
                "e2e_pred": pred,
                "e2e_gt": gt,
            },
        }
    }


def _units(seed: int, n: int = 400) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    ref = rng.integers(20, 60, n)
    err = rng.binomial(ref, 0.06)
    return err.tolist(), ref.tolist()


def test_identical_arm_is_non_inferior() -> None:
    e, n = _units(1)
    b, r = _units(2)
    tp = [x // 2 for x in r]
    lines, pages = _lines(e, n), _pages(b, r, tp, r, r)
    res = cand.accuracy_tests("fused-single-line", lines, pages, lines, pages)
    assert res["met"]
    assert {t["verdict"] for t in res["tests"].values()} == {"met"}
    assert res["tests"]["line_cer"]["interval"]["value"] == pytest.approx(-0.0005)
    assert res["tests"]["e2e_f1"]["side"] == "above"


def test_worse_arm_is_not_non_inferior() -> None:
    e, n = _units(1)
    b, r = _units(2)
    tp = [x // 2 for x in r]
    worse = [x + 3 for x in e]
    res = cand.accuracy_tests(
        "fused-int8", _lines(worse, n), _pages(b, r, tp, r, r), _lines(e, n), _pages(b, r, tp, r, r)
    )
    assert res["tests"]["line_cer"]["verdict"] == "not met"
    assert not res["met"]


def test_accuracy_refuses_mismatched_items() -> None:
    e, n = _units(1)
    b, r = _units(2)
    pages = _pages(b, r, b, r, r)
    other = _lines(e, n, keys=[f"x{i}" for i in range(len(e))])
    with pytest.raises(CandidateError, match="line keys"):
        cand.accuracy_tests("fused-int8", other, pages, _lines(e, n), pages)
    moved = _pages(b, [x + 1 for x in r], b, r, r)
    with pytest.raises(CandidateError, match="page reference lengths"):
        cand.accuracy_tests("fused-int8", _lines(e, n), moved, _lines(e, n), pages)


def _runs(scale: float, seed: int = 0, n: int = 100) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    base = rng.uniform(800, 1200, n)
    return [(base * scale).tolist(), (base * scale * 1.01).tolist()]


def test_speed_condition() -> None:
    fast = cand.speed_tests(_runs(0.5), _runs(1.0))
    assert fast["met"] and fast["saving_ms"] > 0
    same = cand.speed_tests(_runs(1.0), _runs(1.0))
    assert not same["met"]  # equal time is not a 10% saving
    late = _runs(0.5)
    late[1] = _runs(1.0)[1]  # the arm throttles in its second run
    assert not cand.speed_tests(late, _runs(1.0))["met"]
    with pytest.raises(CandidateError):
        cand.speed_tests(_runs(0.5)[:1], _runs(1.0))


def test_memory_condition() -> None:
    assert cand.memory_test([100.0, 120.0], [110.0, 120.0])["met"]
    assert not cand.memory_test([100.0, 121.0], [110.0, 120.0])["met"]


def _acc(met: bool) -> dict[str, Any]:
    return {"met": met}


def _phone(saving: float, ok: bool = True) -> dict[str, Any]:
    return {
        "loaded": True,
        "check_passed": True,
        "speed": {"met": ok, "saving_ms": saving},
        "memory": {"met": True},
    }


def test_select() -> None:
    acc = {"fused-int8": _acc(True), "fused-single-line": _acc(False)}
    before = cand.select(acc, None)
    assert before["selected"] is None and before["to_phone"] == ["fused", "fused-int8"]
    both = {"fused-int8": _acc(True), "fused-single-line": _acc(True)}
    got = cand.select(both, {"fused-int8": _phone(100.0), "fused-single-line": _phone(300.0)})
    assert got["selected"] == "fused-single-line"
    slow = cand.select(acc, {"fused-int8": _phone(100.0, ok=False)})
    assert slow["selected"] == "fused"
    failed = cand.select(acc, {"fused-int8": _phone(100.0)}, fused_failed_phone_check=True)
    assert failed["selected"] == "fused"
    with pytest.raises(CandidateError, match="no phone result"):
        cand.select(acc, {})
    with pytest.raises(CandidateError):
        cand.select({"fused-int8": _acc(True)}, None)


def _profile(arm: str, run: int, start: str, sizes: list[list[int]]) -> dict[str, Any]:
    return {
        "experiment": cand.EXPERIMENT,
        "arm": arm,
        "run": run,
        "smoke": False,
        "git_dirty": False,
        "started_utc": start,
        "input_sizes": sizes,
        "times_ms": [1000.0, 1100.0],
        "mean_ms": 1050.0,
        "p95_ms": 1095.0,
        "peak_mem_mib": 900.0,
        "build": {"recognizer_sha256": "0" * 64},
    }


def _write_sitting(root: Path, swap: bool = False, other_pages: bool = False) -> None:
    order = cand.sitting_order()
    if swap:
        order[0], order[1] = order[1], order[0]
    for k, (arm, run) in enumerate(order):
        sizes = [[640, 480], [800, 600]]
        if other_pages and arm == "fused-int8":
            sizes = [[640, 480], [800, 601]]
        doc = _profile(arm, run, f"2026-11-19T20:{k:02d}:00+00:00", sizes)
        path = root / cand.EXPERIMENT / arm / f"profile_pages_t10_run{run}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc), encoding="utf-8")


def test_read_profiles_checks_order_and_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phase22_candidate as pc

    monkeypatch.setattr(pc, "REPORTS", tmp_path)
    _write_sitting(tmp_path)
    got = pc.read_profiles()
    assert set(got) == set(cand.ARMS) and all(len(r) == 2 for r in got.values())
    _write_sitting(tmp_path, swap=True)
    with pytest.raises(SystemExit, match="mirrored order"):
        pc.read_profiles()
    _write_sitting(tmp_path, other_pages=True)
    with pytest.raises(SystemExit, match="other pages"):
        pc.read_profiles()
