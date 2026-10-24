"""Training diagnostics (EXPERIMENT_PROTOCOL §6): rules that raise named flags with evidence.

Each check reads the run's history of validation points (`EpochPoint`) or per-step values and
returns a `Flag` or None. The trainer records every flag in the metrics log and the run report.
NaN/Inf losses and leakage are not flags but hard stops (`NonFiniteLoss`,
`glypharis.data.rec_dataset.assert_no_leakage`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from glypharis.config.schema import DiagnosticsConfig

OVERFIT_CER_RISE = 0.10  # relative rise of val CER above its best
VAL_LOSS_RISE = 0.02  # relative rise of val loss within the window that counts as divergence
UNDERFIT_AFTER = 0.20  # share of the step budget after which a high train CER is flagged
FLAT_LOSS = 0.01  # relative improvement over the window below which a loss counts as flat
EMPTY_ALARM = 0.5  # share of empty greedy decodes
IDENTICAL_ALARM = 0.5  # share of identical predictions
VANISH_NORM = 1e-7
VANISH_STEPS = 100


class NonFiniteLoss(RuntimeError):
    """A NaN or Inf loss: the run stops after saving the offending batch and its state."""


@dataclass
class Flag:
    name: str
    epoch: int
    step: int
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochPoint:
    """One validation point (end of an epoch)."""

    epoch: int
    step: int
    train_loss: float
    val_loss: float
    train_cer: float
    val_cer: float
    extra: dict[str, float] = field(default_factory=dict)


def _slope(values: Sequence[float]) -> float:
    y = np.asarray(values, dtype=np.float64)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def _rel_gain(values: Sequence[float]) -> float:
    """Relative improvement (decrease) from the first to the last value of a window."""
    first, last = values[0], values[-1]
    return (first - last) / abs(first) if first else 0.0


def check_overfitting(history: Sequence[EpochPoint], cfg: DiagnosticsConfig) -> Flag | None:
    w = cfg.overfit_window
    if len(history) < w:
        return None
    win = history[-w:]
    now = history[-1]
    train_loss = [p.train_loss for p in win]
    val_loss = [p.val_loss for p in win]
    ts = _slope(train_loss)
    low = min(val_loss)
    rise = (val_loss[-1] - low) / low if low > 0 else 0.0
    # A CTC model's validation loss can creep up while its CER still improves (it grows more
    # confident), so loss divergence alone is not overfitting: it also needs CER to have stopped
    # setting new bests within the window (rule revised after GW-REC-001, protocol §11).
    before = [p.val_cer for p in history[:-w]]
    reference = min(before) if before else win[0].val_cer
    cer_improving = min(p.val_cer for p in (win if before else win[1:])) < reference
    if ts < 0 and rise >= VAL_LOSS_RISE and not cer_improving:
        msg = (f"over the last {w} validation points train loss slope {ts:.4g} < 0, val loss "
               f"rose {rise:.1%} above its window low, val CER set no new best")  # fmt: skip
        return Flag("POSSIBLE OVERFITTING", now.epoch, now.step, msg)
    best = min(p.val_cer for p in history)
    train_cer = [p.train_cer for p in win]
    if best > 0 and now.val_cer >= best * (1 + OVERFIT_CER_RISE) and _slope(train_cer) < 0:
        return Flag("POSSIBLE OVERFITTING", now.epoch, now.step,
                    f"val CER {now.val_cer:.4f} is >= {OVERFIT_CER_RISE:.0%} above its best "
                    f"{best:.4f} while train CER keeps falling")  # fmt: skip
    return None


def check_underfitting(
    history: Sequence[EpochPoint], cfg: DiagnosticsConfig, budget_fraction: float
) -> Flag | None:
    if not history:
        return None
    now, thr = history[-1], cfg.underfit_cer_threshold
    if budget_fraction >= UNDERFIT_AFTER and now.train_cer > thr:
        return Flag("POSSIBLE UNDERFITTING", now.epoch, now.step,
                    f"train CER {now.train_cer:.4f} > {thr} after {budget_fraction:.0%} of the "
                    "step budget")  # fmt: skip
    w = cfg.overfit_window
    if len(history) >= w and now.val_cer > thr:
        win = history[-w:]
        tg = _rel_gain([p.train_loss for p in win])
        vg = _rel_gain([p.val_loss for p in win])
        if tg < FLAT_LOSS and vg < FLAT_LOSS:
            return Flag("POSSIBLE UNDERFITTING", now.epoch, now.step,
                        f"train and val loss improved {tg:.2%} and {vg:.2%} over {w} points while "
                        f"val CER {now.val_cer:.4f} > {thr}")  # fmt: skip
    return None


def check_collapse(
    blank_ratio: float,
    empty_ratio: float,
    identical_ratio: float,
    cfg: DiagnosticsConfig,
    epoch: int,
    step: int,
) -> list[Flag]:
    flags = []
    if blank_ratio > cfg.blank_ratio_alarm or empty_ratio > EMPTY_ALARM:
        flags.append(Flag("BLANK COLLAPSE", epoch, step,
                          f"blank argmax share {blank_ratio:.3f} (alarm {cfg.blank_ratio_alarm}), "
                          f"empty decodes {empty_ratio:.3f} (alarm {EMPTY_ALARM})"))  # fmt: skip
    if identical_ratio > IDENTICAL_ALARM:
        msg = f"{identical_ratio:.3f} of validation predictions are the same string"
        flags.append(Flag("PREDICTION COLLAPSE", epoch, step, msg))
    return flags


class GradientMonitor:
    """Explosion: one global norm above the alarm. Vanishing: below 1e-7 for 100 steps in a row."""

    def __init__(self, cfg: DiagnosticsConfig) -> None:
        self.alarm = cfg.grad_norm_alarm
        self.small_run = 0
        self.raised: set[str] = set()

    def update(self, norm: float, epoch: int, step: int) -> Flag | None:
        if norm > self.alarm and "explosion" not in self.raised:
            self.raised.add("explosion")
            return Flag("GRADIENT EXPLOSION", epoch, step,
                        f"global grad norm {norm:.4g} > {self.alarm:g}")  # fmt: skip
        self.small_run = self.small_run + 1 if norm < VANISH_NORM else 0
        if self.small_run >= VANISH_STEPS and "vanishing" not in self.raised:
            self.raised.add("vanishing")
            return Flag("GRADIENT VANISHING", epoch, step,
                        f"global grad norm < {VANISH_NORM:g} for {VANISH_STEPS} steps")  # fmt: skip
        return None

    def state_dict(self) -> dict[str, Any]:
        return {"small_run": self.small_run, "raised": sorted(self.raised)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.small_run = int(state["small_run"])
        self.raised = set(state["raised"])
