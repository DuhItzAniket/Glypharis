"""Bootstrap confidence intervals (EXPERIMENT_PROTOCOL §4).

Ratio metrics such as CER are recomputed on each resample from summed numerators and denominators
(not averaged per sample), so a resample reproduces how the metric itself is defined.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 1000
DEFAULT_SEED = 20261010


@dataclass(frozen=True)
class Interval:
    value: float
    low: float
    high: float
    resamples: int
    unit: str  # what was resampled: "lines", "pages", ...

    def fmt(self, digits: int = 4) -> str:
        return f"{self.value:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def _indices(n: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(resamples, n))


def ratio_ci(
    numerators: np.ndarray,
    denominators: np.ndarray,
    unit: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval:
    """CI for sum(numerators) / sum(denominators), resampling units with replacement."""
    num = np.asarray(numerators, dtype=np.float64)
    den = np.asarray(denominators, dtype=np.float64)
    if num.shape != den.shape or num.ndim != 1:
        raise ValueError("numerators and denominators must be 1-D arrays of the same length")
    n = num.size
    value = float(num.sum() / den.sum()) if den.sum() else 0.0
    if n == 0:
        return Interval(value, value, value, 0, unit)
    idx = _indices(n, resamples, seed)
    d = den[idx].sum(axis=1)
    stats = np.divide(num[idx].sum(axis=1), d, out=np.zeros(resamples), where=d > 0)
    a = (1 - level) / 2
    lo, hi = np.quantile(stats, [a, 1 - a])
    return Interval(value, float(lo), float(hi), resamples, unit)


def mean_ci(values: np.ndarray, unit: str, resamples: int = DEFAULT_RESAMPLES,
            seed: int = DEFAULT_SEED, level: float = 0.95) -> Interval:  # fmt: skip
    v = np.asarray(values, dtype=np.float64)
    return ratio_ci(v, np.ones_like(v), unit, resamples, seed, level)


def paired_ratio_diff_ci(
    num_a: np.ndarray,
    num_b: np.ndarray,
    den: np.ndarray,
    unit: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval:
    """CI for ratio_a - ratio_b on the same units (same resampled indices for both systems)."""
    a = np.asarray(num_a, dtype=np.float64)
    b = np.asarray(num_b, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    total = d.sum()
    value = float((a.sum() - b.sum()) / total) if total else 0.0
    idx = _indices(d.size, resamples, seed)
    dd = d[idx].sum(axis=1)
    diff = np.divide(a[idx].sum(axis=1) - b[idx].sum(axis=1), dd, out=np.zeros(resamples),
                     where=dd > 0)  # fmt: skip
    q = (1 - level) / 2
    lo, hi = np.quantile(diff, [q, 1 - q])
    return Interval(value, float(lo), float(hi), resamples, unit)
