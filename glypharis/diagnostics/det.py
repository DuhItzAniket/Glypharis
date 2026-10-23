"""Detection diagnostics (EXPERIMENT_PROTOCOL §6, spec §9), on top of the shared rules.

The shared over- and underfitting rules (`glypharis.diagnostics`) run unchanged on detection
points, with error = 1 − F1@0.5 in place of CER: `check_det_fitting` builds the underfit threshold
from ``det.diagnostics.underfit_error_threshold`` and `relabel` rewrites the evidence. The other
checks read one validation pass (`DetEvalResult`, `MapStats`) or one epoch's training sums. Each
returns flags with evidence; the trainer raises each flag name once per run.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from glypharis.config.schema import DetDiagnosticsConfig, DiagnosticsConfig
from glypharis.diagnostics import (
    EpochPoint,
    Flag,
    check_overfitting,
    check_underfitting,
)

if TYPE_CHECKING:
    from glypharis.training.det_eval import DetEvalResult, MapStats

THRESH_DIVERGENCE_AFTER: Final = 0.20  # share of the step budget before L_t > l1_const counts
THRESH_DIVERGENCE_POINTS: Final = 3  # consecutive validation points for the L_t rise rule
BCE_FLAT: Final = 0.01  # L_s within 1% of its best counts as flat


def relabel(flag: Flag | None) -> Flag | None:
    """The shared rules speak of CER; for the detector the error is 1 − F1@0.5."""
    if flag is None:
        return None
    return dataclasses.replace(flag, evidence=flag.evidence.replace("CER", "error (1 - F1@0.5)"))


def check_det_fitting(
    history: Sequence[EpochPoint],
    diag: DiagnosticsConfig,
    det: DetDiagnosticsConfig,
    budget_fraction: float,
) -> list[Flag]:
    """Overfitting and underfitting on points whose ``*_cer`` fields hold 1 − F1@0.5."""
    d = dataclasses.replace(diag, underfit_cer_threshold=det.underfit_error_threshold)
    flags = [
        relabel(check_overfitting(history, d)),
        relabel(check_underfitting(history, d, budget_fraction)),
    ]
    return [f for f in flags if f is not None]


def check_map_collapse(m: MapStats, cfg: DetDiagnosticsConfig, epoch: int, step: int) -> list[Flag]:
    flags = []
    ratio = m.pred_text_px / m.gt_kernel_px if m.gt_kernel_px else None
    empty = m.pages_with_care_no_box / m.pages_with_care if m.pages_with_care else 0.0
    share = m.pred_text_px / m.valid_px if m.valid_px else 0.0
    ratio_txt = f"{ratio:.3f}" if ratio is not None else "n/a"
    if (ratio is not None and ratio < cfg.collapse_low_ratio) or empty > cfg.empty_page_alarm:
        flags.append(
            Flag(
                "ALL-BACKGROUND COLLAPSE",
                epoch,
                step,
                f"predicted / kernel px {ratio_txt} (alarm < {cfg.collapse_low_ratio}); "
                f"{empty:.1%} of pages with care lines got no box (alarm "
                f"{cfg.empty_page_alarm:.0%})",
            )
        )
    if share > cfg.text_fraction_alarm or (ratio is not None and ratio > cfg.collapse_high_ratio):
        flags.append(
            Flag(
                "ALL-TEXT COLLAPSE",
                epoch,
                step,
                f"predicted text is {share:.1%} of valid px (alarm "
                f"{cfg.text_fraction_alarm:.0%}); predicted / kernel px {ratio_txt} "
                f"(alarm > {cfg.collapse_high_ratio})",
            )
        )
    return flags


def check_threshold(
    history: Sequence[Mapping[str, Any]],
    m: MapStats,
    cfg: DetDiagnosticsConfig,
    budget_fraction: float,
    epoch: int,
    step: int,
) -> list[Flag]:
    """THRESHOLD DIVERGENCE and THRESHOLD OUT OF RANGE. ``history`` holds the epoch points up to
    and including this one, with ``val_loss_l1``, ``val_l1_const`` and ``val_loss_bce``."""
    flags = []
    pts = [h for h in history if h.get("val_loss_l1") is not None]
    if pts:
        now = pts[-1]
        if budget_fraction >= THRESH_DIVERGENCE_AFTER and now["val_loss_l1"] > now["val_l1_const"]:
            flags.append(
                Flag(
                    "THRESHOLD DIVERGENCE",
                    epoch,
                    step,
                    f"val L_t {now['val_loss_l1']:.4f} is above the {now['val_l1_const']:.4f} "
                    f"of a constant T = 0.5 after {budget_fraction:.0%} of the step "
                    "budget",
                )
            )
        elif len(pts) > THRESH_DIVERGENCE_POINTS:
            rising = True
            for i in range(len(pts) - THRESH_DIVERGENCE_POINTS, len(pts)):
                best_l1 = min(p["val_loss_l1"] for p in pts[: i + 1])
                best_bce = min(p["val_loss_bce"] for p in pts[: i + 1])
                rising &= pts[i]["val_loss_l1"] >= cfg.thresh_divergence_rise * best_l1
                rising &= pts[i]["val_loss_bce"] <= (1 + BCE_FLAT) * best_bce
            if rising:
                flags.append(
                    Flag(
                        "THRESHOLD DIVERGENCE",
                        epoch,
                        step,
                        f"val L_t >= {cfg.thresh_divergence_rise}x its best for "
                        f"{THRESH_DIVERGENCE_POINTS} points while val L_s stayed within "
                        f"{BCE_FLAT:.0%} of its best",
                    )
                )
    oor = m.thresh_out_of_range_px / m.thresh_px if m.thresh_px else 0.0
    if oor > cfg.thresh_out_of_range_alarm:
        flags.append(
            Flag(
                "THRESHOLD OUT OF RANGE",
                epoch,
                step,
                f"{oor:.1%} of threshold-mask pixels have T outside [0.2, 0.8] (alarm "
                f"{cfg.thresh_out_of_range_alarm:.0%}; labels lie in [0.3, 0.7])",
            )
        )
    return flags


def check_box_size(
    res: DetEvalResult, cfg: DetDiagnosticsConfig, epoch: int, step: int
) -> Flag | None:
    m = res.ops[res.selected_op]
    gap = m.f1_30 - m.f1_50
    if gap <= cfg.box_size_gap:
        return None
    px = f"{res.kernel_excess_px:.2f}" if res.kernel_excess_px is not None else "n/a"
    ratio = f"{res.kernel_excess_ratio:.2f}" if res.kernel_excess_ratio is not None else "n/a"
    return Flag(
        "BOX SIZE MISMATCH",
        epoch,
        step,
        f"at {res.selected_op}: F1@0.3 {m.f1_30:.3f} - F1@0.5 {m.f1_50:.3f} = {gap:.3f} > "
        f"{cfg.box_size_gap}; kernel excess {px} px, ratio {ratio}",
    )


def check_postprocess(
    res: DetEvalResult, cfg: DetDiagnosticsConfig, epoch: int, step: int
) -> Flag | None:
    c = res.ops[res.selected_op].counters
    passed = c.too_thin + c.outside + c.kept  # components that passed box_thresh
    dropped = c.too_thin + c.outside
    if not passed or dropped <= cfg.postprocess_drop_alarm * passed:
        return None
    return Flag(
        "POSTPROCESS DROPS",
        epoch,
        step,
        f"at {res.selected_op}: {dropped} of {passed} components above box_thresh were "
        f"dropped (too thin {c.too_thin}, outside {c.outside}; alarm "
        f"{cfg.postprocess_drop_alarm:.0%})",
    )


def check_epoch_data(
    empty_crop_share: float,
    data_wait_share: float,
    cfg: DetDiagnosticsConfig,
    epoch: int,
    step: int,
) -> list[Flag]:
    flags = []
    if empty_crop_share > cfg.empty_crop_alarm:
        flags.append(
            Flag(
                "EMPTY CROPS",
                epoch,
                step,
                f"{empty_crop_share:.1%} of this epoch's training crops have no positive "
                f"pixel (alarm {cfg.empty_crop_alarm:.0%})",
            )
        )
    if data_wait_share > cfg.data_wait_alarm:
        flags.append(
            Flag(
                "DATA LOADER BOTTLENECK",
                epoch,
                step,
                f"the loop waited for data {data_wait_share:.1%} of the epoch (alarm "
                f"{cfg.data_wait_alarm:.0%})",
            )
        )
    return flags


def check_slow_validation(
    val_seconds: Mapping[str, float],
    epoch_train_seconds: float,
    cfg: DetDiagnosticsConfig,
    epoch: int,
    step: int,
) -> Flag | None:
    """``val_seconds`` holds data_wait, forward, post, match and total, and may hold
    ``boxes_per_page`` for the evidence."""
    total = val_seconds.get("total", 0.0)
    if epoch_train_seconds <= 0 or total <= cfg.slow_validation_share * epoch_train_seconds:
        return None
    parts = ", ".join(
        f"{k} {val_seconds[k]:.1f} s"
        for k in ("data_wait", "forward", "post", "match")
        if k in val_seconds
    )
    boxes = val_seconds.get("boxes_per_page")
    extra = f"; {boxes:.1f} boxes per page" if boxes is not None else ""
    return Flag(
        "SLOW VALIDATION",
        epoch,
        step,
        f"validation took {total:.1f} s, {total / epoch_train_seconds:.0%} of the epoch's "
        f"{epoch_train_seconds:.1f} training seconds (alarm "
        f"{cfg.slow_validation_share:.0%}): {parts}{extra}",
    )
