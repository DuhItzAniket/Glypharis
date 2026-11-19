"""The phone candidate's arms and the selection rule (Phase 22, item 14, GW-EFF-013).

Three arms, each a new version and not v1, all built on the v1 fp32 ONNX bundle (GW-EFF-006) with
v1's detector and settings unchanged:

- ``fused``, the default: GW-EFF-009's ``fused-attention`` recognizer, the file Phase 16 wrote,
  checked against the hash GW-EFF-009 recorded (`FUSED_SHA256`);
- ``fused-int8``: that file with H7′'s dynamic INT8 recipe (`glypharis.efficiency.quant`);
- ``fused-single-line``: the fused file, one line per recognizer run (`SingleLineRecognizer`).

Each arm reads lines under the batching rule its bundle would ship (`BATCHING`): ``fused`` and
``fused-int8`` through the recognizer's own width-sorted batches with `CROPS_PER_CALL` crops per
call, ``fused-single-line`` one line per run. No arm is added to Phase 16's ``VARIANTS``, because
its ``SITTINGS`` derive from it and Phase 16's records depend on both.

The selection rule is fixed in `research/hypotheses.md`, under H11. `accuracy_tests` gives the
three laptop accuracy verdicts of an arm against ``fused``; `speed_tests` and `memory_test` the
phone conditions; `select` takes the results and returns the one arm that leaves item 14. The
statistics are `glypharis.eval.stats` with its defaults (1,000 resamples, seed 20261010).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from glypharis.data.download import sha256_file
from glypharis.eval.stats import mean_ci, paired_ratio_diff_ci, paired_ratios_diff_ci, verdict
from glypharis.pipeline.recognizer import TextRecognizer

EXPERIMENT: Final = "GW-EFF-013"
DEFAULT_ARM: Final = "fused"
ARMS: Final = ("fused", "fused-int8", "fused-single-line")
VERSION_LABEL: Final = "phone candidate arm (new version, not v1)"
# Non-inferiority margins against fused, as fractions (research/hypotheses.md, selection rule).
MARGINS: Final[Mapping[str, float]] = {"fused-int8": 0.005, "fused-single-line": 0.0005}
SPEED_FACTOR: Final = 0.9  # the arm must save more than 10% of fused's page time

# GW-EFF-009's fused recognizer: data/processed/phase16/variants/fused-attention/variant.json and
# every report of that variant record this hash.
PHASE16_FUSED: Final = Path("data/processed/phase16/variants/fused-attention/recognizer.fused.onnx")
FUSED_SHA256: Final = "b0e2a5100cb465f72d2abca97fc30a47892c75d069e556791021b7b74dfc172c"
FUSED_FILE: Final = "recognizer.fused.onnx"
INT8_FILE: Final = "recognizer.fused.int8.onnx"

WORK_ROOT: Final = Path("data/processed/phase22/candidate")
REPORTS: Final = Path("reports/final_validation/candidate")
BUILD_FILE: Final = "build.json"

CROPS_PER_CALL: Final = 256
BATCHING: Final[Mapping[str, str]] = {
    "fused": f"width-sorted batches (TextRecognizer), {CROPS_PER_CALL} crops per call",
    "fused-int8": f"width-sorted batches (TextRecognizer), {CROPS_PER_CALL} crops per call",
    "fused-single-line": "one line per recognizer run",
}


class CandidateError(RuntimeError):
    """An arm's files are not the recorded ones, or the selection inputs do not fit together;
    nothing is measured or selected."""


def check_arm(arm: str) -> str:
    if arm not in ARMS:
        raise CandidateError(f"{arm!r} is not an arm; the arms are {', '.join(ARMS)}")
    return arm


def arm_order(run: int) -> tuple[str, ...]:
    """The arms of one profile run, in the sitting's mirrored order: run 1 forward, run 2 back."""
    if run == 1:
        return ARMS
    if run == 2:
        return tuple(reversed(ARMS))
    raise CandidateError(f"run must be 1 or 2, not {run}")


def sitting_order() -> list[tuple[str, int]]:
    """Every (arm, run) of the laptop sitting, in the order they run."""
    return [(a, r) for r in (1, 2) for a in arm_order(r)]


def check_fused(path: str | Path = PHASE16_FUSED) -> str:
    """`CandidateError` unless ``path`` is GW-EFF-009's fused recognizer; returns its hash."""
    p = Path(path)
    if not p.is_file():
        raise CandidateError(f"{p.as_posix()}: no fused recognizer; GW-EFF-009 built it")
    got = sha256_file(p)
    if got != FUSED_SHA256:
        raise CandidateError(f"{p.as_posix()} has SHA-256 {got}, not GW-EFF-009's {FUSED_SHA256}")
    return got


def recognizer_file(arm: str, work_root: str | Path = WORK_ROOT) -> Path:
    """The ONNX recognizer an arm runs: the Phase 16 fused file, or its INT8 version."""
    if check_arm(arm) == "fused-int8":
        return Path(work_root) / arm / INT8_FILE
    return PHASE16_FUSED


class SingleLineRecognizer(TextRecognizer):
    """A `TextRecognizer` that runs its network on one line at a time, in the same width order.

    Everything else (preparation, padding margin, decoding, confidence) is the parent's, so a
    line's reading differs from the batched one only by what the network does with a batch."""

    def _batches(self, widths: Sequence[int]) -> list[list[int]]:
        return [[i] for i in sorted(range(len(widths)), key=lambda i: (widths[i], i))]


def single_line(rec: TextRecognizer) -> SingleLineRecognizer:
    """``rec``'s model, charset and settings, read one line per network run."""
    return SingleLineRecognizer(
        rec.model,
        rec.charset,
        rec.height,
        rec.max_width,
        str(rec.device),
        rec.temperature,
        rec.arch,
        strict_fp32=rec.strict_fp32,
    )


def batching_of(rec: TextRecognizer) -> str:
    """The batching rule a recognizer applies, in `BATCHING`'s words."""
    if isinstance(rec, SingleLineRecognizer):
        return BATCHING["fused-single-line"]
    return BATCHING["fused"]


# --------------------------------------------------------------------------- selection rule


def _iv(iv: Any) -> dict[str, Any]:
    return {
        "value": iv.value,
        "low": iv.low,
        "high": iv.high,
        "resamples": iv.resamples,
        "unit": iv.unit,
    }


def _same(a: Sequence[Any], b: Sequence[Any], what: str) -> None:
    if list(a) != list(b):
        raise CandidateError(
            f"the two arms' {what} differ; they must cover the same items in order"
        )


def accuracy_tests(
    arm: str,
    lines_a: Mapping[str, Any],
    pages_a: Mapping[str, Any],
    lines_f: Mapping[str, Any],
    pages_f: Mapping[str, Any],
) -> dict[str, Any]:
    """The three laptop accuracy verdicts of ``arm`` against ``fused`` (strict track), from the
    ``per_line`` and ``per_page`` blocks of each arm's lines and pages results."""
    if check_arm(arm) == DEFAULT_ARM:
        raise CandidateError("fused is the reference, not an arm to test")
    s = MARGINS[arm]
    la, lf = lines_a["per_line"], lines_f["per_line"]
    _same(la["keys"], lf["keys"], "line keys")
    e_a = np.asarray(la["strict"]["errors"], dtype=np.float64)
    e_f = np.asarray(lf["strict"]["errors"], dtype=np.float64)
    n = np.asarray(lf["strict"]["ref_chars"], dtype=np.float64)
    _same(la["strict"]["ref_chars"], lf["strict"]["ref_chars"], "line reference lengths")
    pa, pf = pages_a["per_page"], pages_f["per_page"]
    _same(pa["doc_ids"], pf["doc_ids"], "page IDs")
    b_a = np.asarray(pa["strict"]["bag_errors"], dtype=np.float64)
    b_f = np.asarray(pf["strict"]["bag_errors"], dtype=np.float64)
    r = np.asarray(pf["strict"]["bag_ref_chars"], dtype=np.float64)
    _same(pa["strict"]["bag_ref_chars"], pf["strict"]["bag_ref_chars"], "page reference lengths")
    tp_a = np.asarray(pa["strict"]["e2e_tp"], dtype=np.float64)
    tp_f = np.asarray(pf["strict"]["e2e_tp"], dtype=np.float64)
    d_a = np.asarray(pa["strict"]["e2e_pred"], dtype=np.float64) + np.asarray(
        pa["strict"]["e2e_gt"], dtype=np.float64
    )
    d_f = np.asarray(pf["strict"]["e2e_pred"], dtype=np.float64) + np.asarray(
        pf["strict"]["e2e_gt"], dtype=np.float64
    )
    tests = {
        "line_cer": (
            paired_ratio_diff_ci(e_a - s * n, e_f, n, "lines"),
            "below",
            "paired_ratio_diff_ci(e_a - s * n, e_f, n, 'lines')",
        ),
        "page_cer": (
            paired_ratio_diff_ci(b_a - s * r, b_f, r, "pages"),
            "below",
            "paired_ratio_diff_ci(b_a - s * r, b_f, r, 'pages')",
        ),
        "e2e_f1": (
            paired_ratios_diff_ci(2 * tp_a + s * d_a, d_a, 2 * tp_f, d_f, "pages"),
            "above",
            "paired_ratios_diff_ci(2 * tp_a + s * d_a, d_a, 2 * tp_f, d_f, 'pages')",
        ),
    }
    out = {
        name: {"interval": _iv(iv), "side": side, "verdict": verdict(iv, side), "formula": f}
        for name, (iv, side, f) in tests.items()
    }
    return {
        "arm": arm,
        "against": DEFAULT_ARM,
        "margin": s,
        "lines": int(n.size),
        "zero_length_lines": int((n == 0).sum()),
        "pages": int(r.size),
        "tests": out,
        "met": all(t["verdict"] == "met" for t in out.values()),
    }


def _page_times(runs: Sequence[Sequence[float]], what: str) -> np.ndarray:
    if len(runs) != 2:
        raise CandidateError(f"{what}: need exactly two runs, got {len(runs)}")
    a, b = (np.asarray(r, dtype=np.float64) for r in runs)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise CandidateError(f"{what}: the two runs must time the same pages")
    return np.stack([a, b])


def speed_tests(
    arm_runs: Sequence[Sequence[float]], fused_runs: Sequence[Sequence[float]]
) -> dict[str, Any]:
    """The speed condition: `mean_ci(t_a - 0.9 * t_f, "pages")` below 0 on the mean of the two
    runs, and on each arm's first and second runs alone. Each run lists the timed pages' network
    times in one page order; run 1 and run 2 are the arm's own first and second runs."""
    ta = _page_times(arm_runs, "arm")
    tf = _page_times(fused_runs, "fused")
    if ta.shape != tf.shape:
        raise CandidateError("the arm and fused must time the same pages")
    cases = {
        "both_runs": ta.mean(axis=0) - SPEED_FACTOR * tf.mean(axis=0),
        "first_runs": ta[0] - SPEED_FACTOR * tf[0],
        "second_runs": ta[1] - SPEED_FACTOR * tf[1],
    }
    out = {}
    for name, x in cases.items():
        iv = mean_ci(x, "pages")
        out[name] = {"interval": _iv(iv), "side": "below", "verdict": verdict(iv, "below")}
    return {
        "factor": SPEED_FACTOR,
        "pages": int(ta.shape[1]),
        "saving_ms": float(tf.mean() - ta.mean()),
        "tests": out,
        "met": all(t["verdict"] == "met" for t in out.values()),
    }


def memory_test(arm_peaks: Sequence[float], fused_peaks: Sequence[float]) -> dict[str, Any]:
    """The memory condition: the arm's peak (the larger of its two runs) is not above fused's."""
    if len(arm_peaks) != 2 or len(fused_peaks) != 2:
        raise CandidateError("need the peak of each of the two runs, for the arm and for fused")
    a, f = max(arm_peaks), max(fused_peaks)
    return {"arm_peak": a, "fused_peak": f, "met": a <= f}


def select(
    accuracy: Mapping[str, Mapping[str, Any]],
    phone: Mapping[str, Mapping[str, Any]] | None,
    fused_failed_phone_check: bool = False,
) -> dict[str, Any]:
    """The arm that leaves item 14.

    ``accuracy`` maps each non-default arm to its `accuracy_tests`. ``phone`` maps each arm that
    was measured to ``{"loaded": bool, "check_passed": bool, "speed": speed_tests(...),
    "memory": memory_test(...)}``; None before the phone runs, which gives no selection. If fused
    fails its own phone check, no other arm can be selected and fused stays the candidate."""
    if set(accuracy) != set(ARMS) - {DEFAULT_ARM}:
        raise CandidateError("accuracy results are needed for every arm but fused")
    to_phone = [a for a in ARMS if a == DEFAULT_ARM or accuracy[a]["met"]]
    if phone is None:
        return {"selected": None, "to_phone": to_phone, "reason": "the phone runs are still due"}
    if fused_failed_phone_check:
        return {
            "selected": DEFAULT_ARM,
            "to_phone": to_phone,
            "qualified": [],
            "reason": "fused failed the phone's logit and text check: no other arm can be "
            "selected, fused stays the candidate and the phone gap stays open",
        }
    qualified = []
    for a in to_phone:
        if a == DEFAULT_ARM:
            continue
        got = phone.get(a)
        if got is None:
            raise CandidateError(f"{a} passed accuracy but has no phone result")
        if got["loaded"] and got["check_passed"] and got["speed"]["met"] and got["memory"]["met"]:
            qualified.append(a)
    if not qualified:
        return {
            "selected": DEFAULT_ARM,
            "to_phone": to_phone,
            "qualified": [],
            "reason": "no other arm met all three conditions",
        }
    best = max(qualified, key=lambda a: (phone[a]["speed"]["saving_ms"], -ARMS.index(a)))
    return {
        "selected": best,
        "to_phone": to_phone,
        "qualified": qualified,
        "reason": "met all three conditions"
        + (", with the larger page-time saving" if len(qualified) > 1 else ""),
    }
