"""Phase 8 sanity sequence for a detector config, run before its real training run (spec §11).

    python scripts/det_sanity.py --config configs/det/gw-det-001.yaml
    python scripts/det_sanity.py --config configs/det/gw-det-002.yaml --checks lr_range dry_run

0. label_ceiling (CPU only, model-free): every validation page at the inference scale, its own
   label kernels used as the probability map, through the real post-processing and counting, for
   each operating point. Also the representational variant (min_text_size 0), the poly box type
   and a counting-parity check of fast_det_counts against evaluate_page. Gate: selection ceiling
   F1@0.5 >= 0.90 and zero parity mismatches.
1. overfit: one fixed batch of 8 training crops (no photometric augmentation) kept on the device,
   AdamW at the configured LR held constant; evaluated in eval mode on the same batch. Pass: kernel
   Dice >= 0.90, threshold L1 <= 0.02 and F1@0.5 >= 0.95 x the batch's own label ceiling.
2. lr_range: LR raised exponentially from 1e-6 to 1 on real augmented batches; the LR at the lowest
   smoothed loss, divided by 10, is the suggestion. Pass: configured LR within 3x of it.
3. dry_run: the real DetTrainer (persistent loaders, configured precision, channels-last) for a
   fixed wall time, then one per-epoch validation pass. Records throughput, data wait, peak VRAM in
   both phases, RSS, GPU temperature, and the epoch and GPU-hour estimates. Pass: finite loss,
   reserved VRAM <= 4 GiB in both phases, data wait <= 10%, RSS <= 8 GB, validation <= 15% of the
   epoch's training time.

Each check builds its own model or trainer and, before it returns, copies its results to the CPU,
closes it and frees CUDA memory, so the next check's GPU preflight sees the memory returned.
Trainers used here write to a temporary directory, never to the run's own directories.

Results go to reports/training/sanity/<experiment_id>/ (sanity.json, sanity.md, lr_range.png).
The overfit check deliberately evaluates on its training batch; it is a sanity check, not a result.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import shapely
import torch
from torch import nn

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.config.schema import RunConfig
from glypharis.data.det_dataset import (
    MAP_KEYS,
    OVERFIT_STREAM,
    DetTrainDataset,
    PageIndex,
    care_centres,
    collate_train,
    split_polygons,
)
from glypharis.data.det_geometry import sample_view
from glypharis.data.det_labels import make_labels
from glypharis.data.pages import Page, PageLine, read_pages
from glypharis.eval.det_matching import fast_det_counts
from glypharis.models.det.dbnet import build_detector
from glypharis.models.det.loss import DBLoss, masked_l1
from glypharis.models.det.postprocess import boxes_from_candidates, find_candidates
from glypharis.models.det.preprocess import ResizeInfo
from glypharis.models.det.runtime import strict_fp32
from glypharis.training.det_eval import label_ceiling_page, summarise_ceiling
from glypharis.training.det_trainer import DetTrainer
from glypharis.training.loop import free_cuda
from glypharis.utils.device import resolve_amp, select_device
from glypharis.utils.gitinfo import git_info
from glypharis.utils.monitor import reset_peak, snapshot
from glypharis.utils.seed import seed_everything

CHECKS = ("label_ceiling", "overfit", "lr_range", "dry_run")

LABEL_CEILING_PAGES: int | None = None  # None: every validation page
CEILING_WORKERS = 4
CEILING_INLINE_MAX = 64  # fewer pages than this are scored in-process (no spawn start-up)
CEILING_GATE = 0.90

OVERFIT_PAGES = 8
OVERFIT_CARE_RANGE = (10, 60)  # care lines per page eligible for the overfit batch
OVERFIT_MAX_STEPS = 2000
OVERFIT_EVAL_FROM = 500  # BatchNorm momentum 0.01: running statistics need a few hundred steps
OVERFIT_EVAL_EVERY = 100
OVERFIT_DICE, OVERFIT_L1, OVERFIT_F1_SHARE, OVERFIT_CEILING_MIN = 0.90, 0.02, 0.95, 0.90

LR_START, LR_END, LR_STEPS = 1e-6, 1.0, 300

VRAM_GATE_GIB = 4.0
DATA_WAIT_GATE = 0.10
RSS_GATE_GB = 8.0
VAL_SHARE_GATE = 0.15
RUN_HOURS = 9.0  # epochs = min(configured, floor(9 h / epoch time)) from GW-DET-001's dry run
VARIANT_MINUTES = 2.0
VARIANT_KEYS = {"channels_last": "det.model.channels_last", "deterministic": "train.deterministic"}


# ------------------------------------------------------------------------------ helpers


def _sanity_trainer(args: argparse.Namespace, work: Path, extra: Sequence[str] = ()) -> DetTrainer:
    """A DetTrainer on the config whose run, checkpoint and report directories are temporary."""
    overrides = [
        *overrides_from_args(args),
        f"paths.runs_dir={work / 'runs'}",
        f"checkpoint.dir={work / 'ckpt'}",
        f"paths.reports_dir={work / 'reports'}",
        *extra,
    ]
    return DetTrainer(load_config(args.config, overrides))


def _step(
    trainer: DetTrainer, opt: torch.optim.Optimizer, batch: dict[str, Any]
) -> tuple[float, float]:
    """One plain optimizer step through the trainer's own forward and loss: (loss, grad norm)."""
    out = trainer.train_step(batch)
    opt.zero_grad(set_to_none=True)
    trainer.scaler.scale(out.loss).backward()
    if trainer.amp.needs_grad_scaler:
        trainer.scaler.unscale_(opt)
    clip = trainer.cfg.train.grad_clip_norm
    norm = float(nn.utils.clip_grad_norm_(trainer.model.parameters(), clip or math.inf))
    trainer.scaler.step(opt)
    trainer.scaler.update()
    return float(out.loss.detach()), norm


def _batches(trainer: DetTrainer) -> Any:
    """Training batches across epoch boundaries, from the trainer's persistent loader."""
    epoch = 0
    while True:
        yield from trainer.train_batches(epoch, 0)
        epoch += 1


def _rss_gb() -> float:
    proc = psutil.Process()
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            continue
    return total / 1e9


def _release(trainer: DetTrainer | None) -> None:
    if trainer is not None:
        trainer.close()
    free_cuda()


# ------------------------------------------------------------------------ 0. label ceiling


def label_ceiling(cfg: RunConfig) -> dict[str, Any]:
    assert cfg.det is not None
    det = cfg.det
    pages = list(read_pages(Path(cfg.paths.data_root) / det.data.val_pages[0]))
    for p in pages:  # validation only: test splits never reach this check
        if p.split != "validation" or p.source == "gdc":
            raise ValueError(
                f"label_ceiling reads validation pages only, got {p.doc_id} ({p.source}/{p.split})"
            )
    if LABEL_CEILING_PAGES is not None:
        pages = pages[:LABEL_CEILING_PAGES]
    t0 = time.perf_counter()
    if len(pages) < CEILING_INLINE_MAX or CEILING_WORKERS <= 1:
        results = [label_ceiling_page(p, det) for p in pages]
    else:
        with ProcessPoolExecutor(CEILING_WORKERS) as ex:
            results = list(ex.map(label_ceiling_page, pages, [det] * len(pages), chunksize=16))
    names = [op.name for op in det.post.operating_points]
    out = summarise_ceiling(results, names)
    strict = out["selection"].get("strict", {"op": None, "f1_50": 0.0})
    out.update(
        long_side=det.infer.long_side,
        box_type=det.post.box_type,
        gate=CEILING_GATE,
        selection_f1_50=strict["f1_50"],
        passed=strict["f1_50"] >= CEILING_GATE and out["mismatches"] == 0,
        seconds=round(time.perf_counter() - t0, 1),
    )
    return out


# -------------------------------------------------------------------------------- 1. overfit


def crop_geometry(
    ds: DetTrainDataset, epoch: int, t: int
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray]]:
    """The crop's polygons, care flags and ignore regions, drawn exactly as the dataset draws
    them (sample_view is the first consumer of the item's generator)."""
    data = ds.det.data
    rng = np.random.default_rng([ds.seed, epoch, t])
    p = t // data.crops_per_page
    rec = ds.index.record(p)
    pts, offs, care = ds.index.line_points(p)
    centres = care_centres(pts, offs, care)
    view = sample_view(rng, (rec.width, rec.height), centres, data, ds.det.infer.long_side)
    polys = split_polygons(view.affine.apply(pts), offs)
    ignore = [view.affine.apply(q) for q in ds.index.ignore_regions(p)]
    return polys, care, ignore


def crop_page(ds: DetTrainDataset, epoch: int, t: int, item: dict[str, Any]) -> Page:
    """A page in crop coordinates for scoring the overfit batch: lines that are text targets in
    the labels and at least half inside the crop count, clipped to it; every other line is don't
    care."""
    size = ds.det.data.crop_size
    polys, care, ignore = crop_geometry(ds, epoch, t)
    labels = make_labels(polys, care, ignore, (size, size), ds.det.labels)
    if not np.array_equal(labels.shrink_map, item["shrink_map"]):
        raise AssertionError("crop_geometry no longer matches DetTrainDataset; update it")
    box = shapely.box(0, 0, size, size)
    lines = []
    for k, poly in enumerate(polys):
        geom = shapely.make_valid(shapely.Polygon(poly))
        clipped = shapely.intersection(geom, box)
        if clipped.is_empty or clipped.area <= 0:
            continue
        if clipped.geom_type != "Polygon":
            clipped = max(getattr(clipped, "geoms", []), key=lambda g: g.area, default=None)
            if clipped is None or clipped.geom_type != "Polygon":
                continue
        counts = bool(labels.care_kept[k]) and clipped.area >= 0.5 * geom.area
        ring = [[float(x), float(y)] for x, y in list(clipped.exterior.coords)[:-1]]
        lines.append(PageLine(f"l{k}", ring, "text", "text", counts, False, False, True))
    ignored = [[[float(x), float(y)] for x, y in q] for q in ignore]
    return Page(f"crop{t}", "", size, size, "crop", "train", lines, ignored)


def _score_maps(
    maps: Sequence[np.ndarray], pages: Sequence[Page], cfg: RunConfig
) -> dict[str, float]:
    """F1@0.5 per operating point of crop-sized probability maps against crop pages."""
    assert cfg.det is not None
    post, size = cfg.det.post, cfg.det.data.crop_size
    info = ResizeInfo(size, size, size, size, size, size)
    sums = {op.name: np.zeros(3) for op in post.operating_points}
    for prob, page in zip(maps, pages, strict=True):
        cands, _ = find_candidates(prob.astype(np.float32), info, post)
        for op in post.operating_points:
            boxes, _ = boxes_from_candidates(cands, op, post, cfg.det.labels.shrink_ratio, info)
            sums[op.name] += fast_det_counts(page, [b.to_pred_line() for b in boxes], (0.5,)).det[
                0.5
            ]
    return {
        n: float(2 * s[0] / (2 * s[0] + s[1] + s[2])) if s.any() else 0.0 for n, s in sums.items()
    }


def overfit(cfg: RunConfig) -> dict[str, Any]:
    """One fixed batch, the same tensor every step, so BatchNorm's running statistics converge to
    exactly the statistics evaluation uses."""
    assert cfg.det is not None
    det, root = cfg.det, Path(cfg.paths.data_root)
    seed_everything(cfg.seed, cfg.train.deterministic)
    device = select_device(cfg.device)
    amp = resolve_amp(cfg.train.mixed_precision, device)
    index = PageIndex.build(root / det.data.train_pages[0], "train", root)
    lo, hi = OVERFIT_CARE_RANGE
    eligible = [i for i in range(len(index)) if lo <= int(index.line_points(i)[2].sum()) <= hi]
    if len(eligible) < OVERFIT_PAGES:
        raise ValueError(f"only {len(eligible)} training pages have {lo}-{hi} care lines")
    rng = np.random.default_rng([cfg.seed, OVERFIT_STREAM])
    pick = np.sort(rng.choice(eligible, OVERFIT_PAGES, replace=False))
    ds = DetTrainDataset(index, root, det, cfg.seed, photometric=False)
    keys = [int(p) * det.data.crops_per_page for p in pick]
    items = [ds[(0, t)] for t in keys]
    pages = [crop_page(ds, 0, t, it) for t, it in zip(keys, items, strict=True)]
    batch = collate_train(items)
    images = batch["images"].to(device).permute(0, 3, 1, 2)
    shrink, smask, tmap, tmask = (batch[k].to(device).float().unsqueeze(1) for k in MAP_KEYS)
    ceiling = _score_maps([it["shrink_map"] for it in items], pages, cfg)
    ceiling_best = max(ceiling.values())

    model = build_detector(det.model, root).to(device)
    loss_fn = DBLoss(det.loss)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    curve: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    step = 0
    last: dict[str, Any] = {}
    while step < OVERFIT_MAX_STEPS:
        model.train()
        with amp.autocast():
            logits = model.forward_train(images)
        loss = loss_fn(logits, shrink, smask, tmap, tmask).total
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm or math.inf)
        opt.step()
        step += 1
        due = step >= OVERFIT_EVAL_FROM and step % OVERFIT_EVAL_EVERY == 0
        if not (due or step == OVERFIT_MAX_STEPS):
            continue
        model.eval()
        with torch.no_grad(), strict_fp32(), torch.autocast(device.type, enabled=False):
            out = model.forward_train(images)
            prob, thresh = torch.sigmoid(out.prob), torch.sigmoid(out.thresh)
            pred = (prob > 0.5).float()
            dice = float(
                2
                * (pred * shrink * smask).sum()
                / ((pred * smask).sum() + (shrink * smask).sum()).clamp(min=1e-6)
            )
            l1 = float(masked_l1(thresh, tmap, tmask, det.loss.eps))
        f1 = _score_maps([p[0].cpu().numpy() for p in prob], pages, cfg)
        last = {
            "step": step,
            "loss": float(loss.detach()),
            "kernel_dice": dice,
            "l1": l1,
            "f1_50": max(f1.values()),
            "f1_per_op": f1,
        }
        curve.append(last)
        if (
            dice >= OVERFIT_DICE
            and l1 <= OVERFIT_L1
            and last["f1_50"] >= OVERFIT_F1_SHARE * ceiling_best
        ):
            break
    result = {
        "pages": [index.record(int(p)).doc_id for p in pick],
        "steps": step,
        "seconds": round(time.perf_counter() - t0, 1),
        "batch_ceiling_f1_50": ceiling_best,
        "batch_ceiling_per_op": ceiling,
        "final": last,
        "mis_specified": ceiling_best < OVERFIT_CEILING_MIN,
        "passed": bool(last)
        and last["kernel_dice"] >= OVERFIT_DICE
        and last["l1"] <= OVERFIT_L1
        and last["f1_50"] >= OVERFIT_F1_SHARE * ceiling_best,
        "curve": curve,
    }
    del model, opt, images, shrink, smask, tmap, tmask
    free_cuda()
    return result


# ------------------------------------------------------------------------------- 2. lr range


def lr_range(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="det_lr_range_"))
    trainer = None
    try:
        trainer = _sanity_trainer(args, work)
        cfg = trainer.cfg
        opt = torch.optim.AdamW(
            trainer.model.parameters(), lr=LR_START, weight_decay=cfg.train.weight_decay
        )
        factor = (LR_END / LR_START) ** (1 / LR_STEPS)
        lrs: list[float] = []
        losses: list[float] = []
        smooth, best = None, math.inf
        trainer.model.train()
        for i, batch in enumerate(_batches(trainer)):
            lr = LR_START * factor**i
            for g in opt.param_groups:
                g["lr"] = lr
            loss, _ = _step(trainer, opt, batch)
            if not math.isfinite(loss):
                break
            smooth = loss if smooth is None else 0.95 * smooth + 0.05 * loss
            lrs.append(lr)
            losses.append(smooth)
            best = min(best, smooth)
            if i + 1 >= LR_STEPS or (i > 20 and smooth > 4 * best):
                break
        configured = cfg.train.learning_rate
    finally:
        _release(trainer)
        del trainer
        free_cuda()
        shutil.rmtree(work, ignore_errors=True)
    if not lrs:
        return {
            "steps": 0,
            "configured_lr": configured,
            "configured_within_3x_of_suggestion": False,
        }
    k = int(np.argmin(losses))
    suggested = lrs[k] / 10
    _plot_lr(lrs, losses, suggested, configured, out / "lr_range.png")
    return {
        "steps": len(lrs),
        "lr_at_min_loss": lrs[k],
        "suggested_lr": suggested,
        "configured_lr": configured,
        "configured_within_3x_of_suggestion": suggested / 3 <= configured <= suggested * 3,
    }


def _plot_lr(
    lrs: list[float], losses: list[float], suggested: float, configured: float, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lrs, losses)
    ax.axvline(suggested, ls="--", c="g", label=f"suggested {suggested:.1e}")
    ax.axvline(configured, ls=":", c="r", label=f"configured {configured:.1e}")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed DB loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# -------------------------------------------------------------------------------- 3. dry run


def dry_run(args: argparse.Namespace, minutes: float, extra: Sequence[str] = ()) -> dict[str, Any]:
    work = Path(tempfile.mkdtemp(prefix="det_dry_run_"))
    trainer = None
    try:
        trainer = _sanity_trainer(args, work, extra)
        result = _dry_run(trainer, minutes)
    finally:
        _release(trainer)
        del trainer
        free_cuda()
        shutil.rmtree(work, ignore_errors=True)
    return result


def _dry_run(trainer: DetTrainer, minutes: float) -> dict[str, Any]:
    cfg, dev = trainer.cfg, trainer.device
    cuda = dev.type == "cuda"
    reset_peak(dev)
    start, wait, crops, steps, clipped = time.perf_counter(), 0.0, 0, 0, 0
    losses: list[float] = []
    temps: list[int] = []
    rss = _rss_gb()
    stats: dict[str, float] = {}
    clip = cfg.train.grad_clip_norm
    batches = _batches(trainer)
    trainer.model.train()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        while time.perf_counter() - start < minutes * 60:
            t0 = time.perf_counter()
            batch = next(batches)
            wait += time.perf_counter() - t0
            loss, norm = _step(trainer, trainer.optimizer, batch)
            if trainer.scheduler is not None:
                trainer.scheduler.step()
            losses.append(loss)
            clipped += int(clip is not None and norm > clip)
            crops += len(batch["ids"])
            steps += 1
            for f in dataclasses.fields(batch["stats"]):
                stats[f.name] = stats.get(f.name, 0) + getattr(batch["stats"], f.name)
            if steps % 50 == 0:
                rss = max(rss, _rss_gb())
                temp = snapshot(dev).gpu_temperature_c
                if temp is not None:
                    temps.append(temp)
        if cuda:
            torch.cuda.synchronize()
        train_s = time.perf_counter() - start
        train_alloc = torch.cuda.max_memory_allocated(dev) / 2**30 if cuda else None
        train_reserved = torch.cuda.max_memory_reserved(dev) / 2**30 if cuda else None
        reset_peak(dev)
        t0 = time.perf_counter()
        val = trainer.validate()  # records its peak VRAM in the point, then resets the counter
        val_s = time.perf_counter() - t0
        gib = 1024.0
        val_alloc = val.point["vram_peak_alloc_mib"] / gib if cuda else None
        val_reserved = val.point["vram_peak_reserved_mib"] / gib if cuda else None
        rss = max(rss, _rss_gb())
    temp = snapshot(dev).gpu_temperature_c
    temps += [temp] if temp is not None else []
    steps_per_epoch = trainer.steps_per_epoch[0]
    epoch_s = steps_per_epoch * train_s / max(steps, 1)
    run_epochs = min(cfg.train.epochs, math.floor(RUN_HOURS * 3600 / (epoch_s + val_s)))
    val_share = val_s / epoch_s if epoch_s else math.inf
    vram_ok = None  # not measured on the CPU
    if train_reserved is not None and val_reserved is not None:
        vram_ok = max(train_reserved, val_reserved) <= VRAM_GATE_GIB
    finite = bool(losses) and all(math.isfinite(x) for x in losses)
    wait_share = wait / train_s if train_s else 0.0
    return {
        "minutes": round(train_s / 60, 2),
        "steps": steps,
        "crops": crops,
        "crops_per_s": round(crops / train_s, 2) if train_s else 0.0,
        "ms_per_step": round(1000 * train_s / max(steps, 1), 1),
        "data_wait_share": round(wait_share, 3),
        "clipped_share": round(clipped / max(steps, 1), 3),
        "train_peak_alloc_gib": train_alloc,
        "train_peak_reserved_gib": train_reserved,
        "val_peak_alloc_gib": val_alloc,
        "val_peak_reserved_gib": val_reserved,
        "rss_gb_max": round(rss, 2),
        "gpu_temperature_c_max": max(temps) if temps else None,
        "loss_first_last": [losses[0], float(np.mean(losses[-50:]))] if losses else None,
        "finite": finite,
        "label_stats": stats,
        "nondeterminism_warnings": sorted(
            {str(w.message)[:200] for w in caught if "determinis" in str(w.message)}
        ),
        "val_seconds": round(val_s, 1),
        "val_pages": len(trainer.eval_ds),
        "val_seconds_parts": {
            k: val.point.get(f"val_seconds_{k}")
            for k in ("data_wait", "forward", "post", "match", "total")
        },
        "steps_per_epoch": steps_per_epoch,
        "estimated_minutes_per_epoch": round(epoch_s / 60, 1),
        "validation_share": round(val_share, 3),
        "epochs_by_9h_rule": run_epochs,
        "estimated_gpu_hours": round(cfg.train.epochs * (epoch_s + val_s) / 3600, 2),
        "gates": {
            "vram_reserved_le_4gib": vram_ok,
            "data_wait_le_10pct": wait_share <= DATA_WAIT_GATE,
            "rss_le_8gb": rss <= RSS_GATE_GB,
            "validation_le_15pct": val_share <= VAL_SHARE_GATE,
        },
        "passed": finite
        and vram_ok is not False
        and wait_share <= DATA_WAIT_GATE
        and rss <= RSS_GATE_GB
        and val_share <= VAL_SHARE_GATE,
    }


# ------------------------------------------------------------------------------- reporting


def write_markdown(results: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Sanity sequence: {results['experiment_id']}",
        "",
        "Generated by `scripts/det_sanity.py`; do not edit by hand.",
        "",
        f"Device {results['device']}, precision {results['precision']}, git `{results['git']}`.",
        "",
    ]
    if "label_ceiling" in results:
        c = results["label_ceiling"]
        sel = c["selection"]
        lines += [
            f"- **Label ceiling** ({c['pages']:,} validation pages at {c['long_side']} px): "
            f"selection ceiling F1@0.5 {c['selection_f1_50']:.4f} at "
            f"`{sel.get('strict', {}).get('op')}` (gate {c['gate']}), representational "
            f"{sel.get('representational', {}).get('f1_50', float('nan')):.4f}, poly "
            f"{sel.get('poly', {}).get('f1_50', float('nan')):.4f}; demoted care lines "
            f"{100 * c['demoted_share']:.1f}%; counting mismatches {c['mismatches']}: "
            f"{'PASS' if c['passed'] else 'FAIL'}"
        ]
    if "overfit" in results:
        o = results["overfit"]
        f = o["final"] or {}
        verdict = "MIS-SPECIFIED" if o["mis_specified"] else ("PASS" if o["passed"] else "FAIL")
        lines += [
            f"- **Overfit one batch** ({len(o['pages'])} crops): kernel Dice "
            f"{f.get('kernel_dice', float('nan')):.3f}, threshold L1 "
            f"{f.get('l1', float('nan')):.4f}, F1@0.5 {f.get('f1_50', float('nan')):.3f} "
            f"against a batch ceiling of {o['batch_ceiling_f1_50']:.3f} after {o['steps']} "
            f"steps: {verdict}"
        ]
    if "lr_range" in results:
        r = results["lr_range"]
        if r.get("steps"):
            lines += [
                f"- **LR range test**: lowest smoothed loss at {r['lr_at_min_loss']:.2e}; "
                f"suggested {r['suggested_lr']:.2e}, configured {r['configured_lr']:.2e} "
                f"({'within' if r['configured_within_3x_of_suggestion'] else 'outside'} "
                "3x). ![lr range](lr_range.png)"
            ]
        else:
            lines += ["- **LR range test**: no finite step"]
    if "dry_run" in results:
        d = results["dry_run"]
        lines += [
            f"- **Dry run** ({d['minutes']} min): {d['crops_per_s']} crops/s, "
            f"{d['ms_per_step']} ms/step, data wait {100 * d['data_wait_share']:.0f}%, peak "
            f"reserved VRAM {d['train_peak_reserved_gib']} GiB (train) / "
            f"{d['val_peak_reserved_gib']} GiB (validation), RSS {d['rss_gb_max']} GB, GPU "
            f"max {d['gpu_temperature_c_max']} C, loss finite: {d['finite']}; about "
            f"{d['estimated_minutes_per_epoch']} min per epoch, validation "
            f"{100 * d['validation_share']:.0f}% of it, {d['epochs_by_9h_rule']} epochs by "
            f"the 9-hour rule, {d['estimated_gpu_hours']} GPU-hours at the configured "
            f"epochs: {'PASS' if d['passed'] else 'FAIL'}"
        ]
    for name, d in results.get("dry_run_variants", {}).items():
        lines += [
            f"- Dry-run variant `{name}` (information only): {d['crops_per_s']} crops/s, "
            f"{d['ms_per_step']} ms/step"
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--checks", nargs="+", default=list(CHECKS), choices=CHECKS)
    parser.add_argument("--dry-run-minutes", type=float, default=10.0)
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=f"extra information-only dry run, KEY in {sorted(VARIANT_KEYS)}",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides_from_args(args))
    cfg.require_run_identity()
    if cfg.task != "det" or cfg.det is None:
        raise SystemExit(f"{args.config}: not a detector config")
    out = Path(cfg.paths.reports_dir) / "training" / "sanity" / str(cfg.experiment_id)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sanity.json"
    results: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    device = select_device(cfg.device)
    results.update(
        experiment_id=cfg.experiment_id,
        device=str(device),
        precision=resolve_amp(cfg.train.mixed_precision, device).name,
        git=git_info().commit,
    )
    for check in args.checks:
        if check == "label_ceiling":
            results[check] = label_ceiling(cfg)
        elif check == "overfit":
            results[check] = overfit(cfg)
        elif check == "lr_range":
            results[check] = lr_range(args, out)
        else:
            results[check] = dry_run(args, args.dry_run_minutes)
            variants = {}
            for item in args.variant:
                key, _, value = item.partition("=")
                if key not in VARIANT_KEYS:
                    raise SystemExit(
                        f"--variant {item!r}: KEY must be one of {sorted(VARIANT_KEYS)}"
                    )
                variants[item] = dry_run(args, VARIANT_MINUTES, [f"{VARIANT_KEYS[key]}={value}"])
            if variants:
                results["dry_run_variants"] = variants
        path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(
            check,
            json.dumps({k: v for k, v in results[check].items() if k not in ("curve", "per_op")}),
        )
    write_markdown(results, out / "sanity.md")
    ok = all(
        results.get(c, {}).get("passed", True) for c in ("label_ceiling", "overfit", "dry_run")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
