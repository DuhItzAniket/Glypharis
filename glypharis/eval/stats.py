"""Bootstrap confidence intervals (EXPERIMENT_PROTOCOL §4).

Ratio metrics such as CER are recomputed on each resample from summed numerators and denominators
(not averaged per sample), so a resample reproduces how the metric itself is defined.

Phase 14 adds unpaired comparisons between two sets (independent resampling), cluster bootstraps
(whole source documents or pages resampled with all their units), and `verdict`, the registered
"met" / "not met" / "not shown" reading of an interval against 0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

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


def paired_ratios_diff_ci(
    num_a: np.ndarray,
    den_a: np.ndarray,
    num_b: np.ndarray,
    den_b: np.ndarray,
    unit: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval:
    """CI for sum(num_a)/sum(den_a) - sum(num_b)/sum(den_b) on the same resampled units.

    Unlike `paired_ratio_diff_ci`, each system has its own denominator, as for detection F1
    (num = 2tp, den = 2tp + fp + fn per page): two detectors predict different boxes on a page.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in (num_a, den_a, num_b, den_b)]
    if arrays[0].ndim != 1 or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("numerators and denominators must be 1-D arrays of the same length")
    na, da, nb, db = arrays

    def ratio(num: float, den: float) -> float:
        return num / den if den else 0.0

    value = ratio(float(na.sum()), float(da.sum())) - ratio(float(nb.sum()), float(db.sum()))
    n = na.size
    if n == 0:
        return Interval(value, value, value, 0, unit)
    idx = _indices(n, resamples, seed)
    sums = [x[idx].sum(axis=1) for x in arrays]
    ra = np.divide(sums[0], sums[1], out=np.zeros(resamples), where=sums[1] > 0)
    rb = np.divide(sums[2], sums[3], out=np.zeros(resamples), where=sums[3] > 0)
    q = (1 - level) / 2
    lo, hi = np.quantile(ra - rb, [q, 1 - q])
    return Interval(value, float(lo), float(hi), resamples, unit)


# ------------------------------------------------------------------------ Phase 14 additions
#
# The functions below draw their resamples a block of rows at a time, so memory stays small on
# large packs (74,500 validation crops would need 600 MB of indices in a single draw). numpy's
# Generator yields the same index stream whether the rows are drawn at once or in blocks, so with
# the same seed they resample exactly as `ratio_ci` and `paired_ratio_diff_ci` do.

_BLOCK_CELLS = 2_000_000  # indices drawn per block (rows x units)

Side = Literal["below", "above"]
ArrayLike: TypeAlias = np.ndarray | Sequence[float]


def _second_rng(seed: int) -> np.random.Generator:
    """An independent stream for the second side of an unpaired comparison.

    The first side uses ``default_rng(seed)``, the stream `ratio_ci` uses, so on its own it
    resamples exactly as `ratio_ci` would.
    """
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])


def _resampled_sums(
    arrays: Sequence[np.ndarray], resamples: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Sum of each array over ``resamples`` draws of its n entries with replacement."""
    n = arrays[0].size
    out = [np.empty(resamples) for _ in arrays]
    rows = max(1, _BLOCK_CELLS // max(n, 1))
    for start in range(0, resamples, rows):
        k = min(rows, resamples - start)
        idx = rng.integers(0, n, size=(k, n))
        for x, o in zip(arrays, out, strict=True):
            o[start : start + k] = x[idx].sum(axis=1)
    return out


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def _ratios(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros(num.size), where=den > 0)


def _vectors(*xs: ArrayLike) -> list[np.ndarray]:
    arrays = [np.asarray(x, dtype=np.float64) for x in xs]
    if arrays[0].ndim != 1 or any(x.shape != arrays[0].shape for x in arrays):
        raise ValueError("per-unit arrays must be 1-D and of the same length")
    return arrays


def _cluster_totals(
    clusters: np.ndarray | Sequence[object], arrays: Sequence[np.ndarray]
) -> list[np.ndarray]:
    """Per-cluster sums of each per-unit array, clusters in sorted id order."""
    ids = np.asarray(clusters)
    if ids.ndim != 1 or ids.size != arrays[0].size:
        raise ValueError("clusters must give one id per unit")
    _, inverse = np.unique(ids, return_inverse=True)
    inverse = inverse.reshape(-1)
    m = int(inverse.max()) + 1
    return [np.bincount(inverse, weights=x, minlength=m) for x in arrays]


def _interval(value: float, stats: np.ndarray, resamples: int, unit: str, level: float) -> Interval:
    q = (1 - level) / 2
    lo, hi = np.quantile(stats, [q, 1 - q])
    return Interval(value, float(lo), float(hi), resamples, unit)


def unpaired_ratio_diff_ci(
    err_a: ArrayLike,
    den_a: ArrayLike,
    err_b: ArrayLike,
    den_b: ArrayLike,
    factor: float,
    unit: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval:
    """CI for sum(err_a)/sum(den_a) - factor * sum(err_b)/sum(den_b) on two different sets.

    The sets share no units (H5′: GDC crops against HierText validation crops), so each is
    resampled on its own from independent streams, and the percentile interval is taken over the
    differences of the k-th resample ratios.
    """
    ea, da = _vectors(err_a, den_a)
    eb, db = _vectors(err_b, den_b)
    rb = _ratio(float(eb.sum()), float(db.sum()))
    value = _ratio(float(ea.sum()), float(da.sum())) - factor * rb
    if ea.size == 0 or eb.size == 0:
        return Interval(value, value, value, 0, unit)
    sa = _resampled_sums([ea, da], resamples, np.random.default_rng(seed))
    sb = _resampled_sums([eb, db], resamples, _second_rng(seed))
    stats = _ratios(sa[0], sa[1]) - factor * _ratios(sb[0], sb[1])
    return _interval(value, stats, resamples, unit, level)


def cluster_paired_ratio_diff_ci(
    err_a: ArrayLike,
    err_b: ArrayLike,
    den: ArrayLike,
    clusters: np.ndarray | Sequence[object],
    unit: str,
    factor: float = 1.0,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval:
    """CI for sum(err_a)/sum(den) - factor * sum(err_b)/sum(den), resampling whole clusters.

    ``clusters`` holds one id per unit (for example each GDC page's source document). A resample
    draws clusters with replacement and keeps all their units, for both systems alike, so units
    that share a source are not treated as independent. ``unit`` names the clusters
    ("documents"). With every unit its own cluster (ids 0..n-1) this resamples exactly as
    `paired_ratio_diff_ci` does on ``err_a`` and ``factor * err_b``.
    """
    a, b, d = _vectors(err_a, err_b, den)
    value = _ratio(float(a.sum()) - factor * float(b.sum()), float(d.sum()))
    if d.size == 0:
        return Interval(value, value, value, 0, unit)
    ca, cb, cd = _cluster_totals(clusters, [a, b, d])
    sa, sb, sd = _resampled_sums([ca, cb, cd], resamples, np.random.default_rng(seed))
    return _interval(value, _ratios(sa - factor * sb, sd), resamples, unit, level)


def cluster_unpaired_ratio_diff_ci(
    err_a: ArrayLike,
    den_a: ArrayLike,
    clusters_a: np.ndarray | Sequence[object],
    err_b: ArrayLike,
    den_b: ArrayLike,
    factor: float,
    unit: str,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = 0.95,
) -> Interval:
    """As `unpaired_ratio_diff_ci`, but side a is resampled by cluster and side b by unit.

    H5′'s cluster check: the GDC side by source document, the validation side by line.
    """
    ea, da = _vectors(err_a, den_a)
    eb, db = _vectors(err_b, den_b)
    rb = _ratio(float(eb.sum()), float(db.sum()))
    value = _ratio(float(ea.sum()), float(da.sum())) - factor * rb
    if ea.size == 0 or eb.size == 0:
        return Interval(value, value, value, 0, unit)
    ca = _cluster_totals(clusters_a, [ea, da])
    sa = _resampled_sums(ca, resamples, np.random.default_rng(seed))
    sb = _resampled_sums([eb, db], resamples, _second_rng(seed))
    stats = _ratios(sa[0], sa[1]) - factor * _ratios(sb[0], sb[1])
    return _interval(value, stats, resamples, unit, level)


def verdict(iv: Interval, side: Side | str) -> str:
    """The registered verdict wording (Phase 14 plan, item 5) for a CI against 0.

    ``side`` is the side the hypothesis claims, "below" or "above" 0. The verdict is "met" if the
    whole interval lies strictly on that side, "not met" if it lies strictly on the other side,
    and "not shown" otherwise (an interval that touches or straddles 0).
    """
    if side == "below":
        claimed, other = iv.high < 0, iv.low > 0
    elif side == "above":
        claimed, other = iv.low > 0, iv.high < 0
    else:
        raise ValueError(f"side must be 'below' or 'above', not {side!r}")
    if claimed:
        return "met"
    return "not met" if other else "not shown"
