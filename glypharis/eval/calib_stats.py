"""Bootstrap intervals for calibration comparisons (Phase 14, H8′).

`ece_diff_ci` gives the percentile CI of ECE(conf_a) - factor * ECE(conf_b) on the same lines:
each resample draws one set of lines (or of whole clusters, such as pages, with all their lines)
and recomputes both ECEs on it with `glypharis.pipeline.calibration.ece`, so the 15 equal-mass bins
are rebuilt on every resample exactly as the metric is defined. The line bootstrap draws one row
of indices per resample from ``default_rng(seed)``, the scheme GW-E2E-005 used
(`scripts/rec_calibrate.py`), so with that seed and ``factor`` 1 it reproduces that interval.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from glypharis.eval.stats import DEFAULT_RESAMPLES, DEFAULT_SEED, Interval
from glypharis.pipeline.calibration import BINS, ece


def _cluster_members(clusters: np.ndarray | Sequence[object], n: int) -> list[np.ndarray]:
    """Line indices of each cluster, clusters in sorted id order."""
    ids = np.asarray(clusters)
    if ids.ndim != 1 or ids.size != n:
        raise ValueError("clusters must give one id per line")
    _, inverse = np.unique(ids, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind="stable")
    return np.split(order, np.cumsum(np.bincount(inverse))[:-1])


def ece_diff_ci(
    conf_a: np.ndarray | Sequence[float],
    conf_b: np.ndarray | Sequence[float],
    correct: np.ndarray | Sequence[bool],
    factor: float,
    unit: str,
    clusters: np.ndarray | Sequence[object] | None = None,
    n_bins: int = BINS,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval:
    """CI of ECE(conf_a, correct) - factor * ECE(conf_b, correct).

    ``conf_a`` and ``conf_b`` are two confidences for the same lines (for H8′, T = 2.35 and T = 1)
    and ``correct`` the lines' exact-match flags. Without ``clusters`` lines are resampled; with
    them (one id per line), whole clusters are drawn with replacement and keep all their lines.
    ``unit`` names what was resampled ("lines", "pages").
    """
    a = np.asarray(conf_a, dtype=np.float64)
    b = np.asarray(conf_b, dtype=np.float64)
    y = np.asarray(correct, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or a.shape != y.shape:
        raise ValueError("conf_a, conf_b and correct must be 1-D arrays of the same length")
    n = a.size
    value = ece(a, y, n_bins) - factor * ece(b, y, n_bins)
    if n == 0:
        return Interval(value, value, value, 0, unit)
    rng = np.random.default_rng(seed)
    members = None if clusters is None else _cluster_members(clusters, n)
    diffs = np.empty(resamples)
    for k in range(resamples):
        if members is None:
            s = rng.integers(0, n, n)
        else:
            picks = rng.integers(0, len(members), len(members))
            s = np.concatenate([members[i] for i in picks])
        diffs[k] = ece(a[s], y[s], n_bins) - factor * ece(b[s], y[s], n_bins)
    q = (1 - level) / 2
    lo, hi = np.quantile(diffs, [q, 1 - q])
    return Interval(value, float(lo), float(hi), resamples, unit)
