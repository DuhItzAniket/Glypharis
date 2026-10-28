"""Confidence calibration of line readings (Phase 10, EXPERIMENT_PROTOCOL §3, RQ8).

- `ece`: expected calibration error with 15 equal-mass bins: lines sorted by confidence and cut
  into 15 groups of (almost) equal size; the size-weighted mean of |accuracy − mean confidence|.
- `fit_temperature`: the temperature on the grid `TEMPERATURES` that minimises the binary negative
  log-likelihood of correctness (exact match on the strict track) given the line confidence at
  that temperature. Ties go to the temperature nearest 1. Fitted on validation only.
- `reliability`: the per-bin points of a reliability diagram.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

BINS = 15
TEMPERATURES = tuple(round(0.5 + 0.05 * k, 2) for k in range(71))  # 0.50 .. 4.00


def _bins(conf: np.ndarray, n_bins: int) -> list[np.ndarray]:
    order = np.argsort(conf, kind="stable")
    return [b for b in np.array_split(order, n_bins) if b.size]


def ece(conf: Sequence[float] | np.ndarray, correct: Sequence[bool] | np.ndarray,
        n_bins: int = BINS) -> float:  # fmt: skip
    c = np.asarray(conf, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if c.size == 0:
        return 0.0
    return float(sum(b.size * abs(y[b].mean() - c[b].mean()) for b in _bins(c, n_bins)) / c.size)


def reliability(conf: Sequence[float] | np.ndarray, correct: Sequence[bool] | np.ndarray,
                n_bins: int = BINS) -> list[dict[str, float]]:  # fmt: skip
    c = np.asarray(conf, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    return [{"lines": int(b.size), "confidence": float(c[b].mean()), "accuracy": float(y[b].mean())}
            for b in _bins(c, n_bins)]  # fmt: skip


def binary_nll(conf: np.ndarray, correct: np.ndarray) -> float:
    p = np.clip(conf, 1e-6, 1 - 1e-6)
    return float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1 - p)))


def fit_temperature(
    confidence_at: Callable[[float], np.ndarray], correct: Sequence[bool] | np.ndarray
) -> dict[str, Any]:
    """``confidence_at(T)`` gives every line's confidence at temperature T."""
    y = np.asarray(correct, dtype=np.float64)
    nll = {t: binary_nll(confidence_at(t), y) for t in TEMPERATURES}
    best = min(nll, key=lambda t: (nll[t], abs(t - 1.0)))
    return {"temperature": best, "nll": nll[best], "nll_at_1": nll[1.0], "grid": list(nll.items())}
