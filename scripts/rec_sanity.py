"""Phase 6 sanity sequence for a recognizer config, run before the real training run.

    python scripts/rec_sanity.py --config configs/rec/gw-rec-001.yaml
    python scripts/rec_sanity.py --config configs/rec/gw-rec-001.yaml --checks overfit lr_range

1. overfit: 64 training lines, no augmentation, constant learning rate; the model must drive its CER
   on those same lines to <= 1% (shows the model, loss and decoding can learn at all).
2. lr_range: learning rate raised exponentially from 1e-6 to 1 over a few hundred steps on the real
   training data; the loss curve is saved and the rate at the lowest smoothed loss, divided by 10,
   is reported as a guide for the configured rate.
3. dry_run: the real config for a fixed wall time; records throughput, step time, the share of time
   spent waiting for data, peak VRAM and GPU temperature.

Results go to reports/training/sanity/<experiment_id>/ (sanity.json, sanity.md, figures). The
overfit check deliberately validates on its training lines; it is a sanity check, not a result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from glypharis.config.cli import add_config_args, config_from_args
from glypharis.config.schema import RunConfig
from glypharis.data.augment import Augmenter
from glypharis.data.rec_dataset import RecLineDataset, WidthBucketBatchSampler, collate
from glypharis.data.text import Charset
from glypharis.models.rec.crnn import build_recognizer, count_parameters
from glypharis.training.rec_trainer import ctc_loss, evaluate
from glypharis.utils.device import AmpPolicy, resolve_amp, select_device
from glypharis.utils.monitor import reset_peak, snapshot
from glypharis.utils.seed import make_generator, seed_everything

OVERFIT_LINES = 64
OVERFIT_MAX_STEPS = 3000
OVERFIT_TARGET_CER = 0.01
LR_START, LR_END, LR_STEPS = 1e-6, 1.0, 400


class Parts:
    def __init__(self, cfg: RunConfig) -> None:
        seed_everything(cfg.seed, cfg.train.deterministic)
        self.cfg = cfg
        self.device = select_device(cfg.device)
        self.amp: AmpPolicy = resolve_amp(cfg.train.mixed_precision, self.device)
        root = Path(cfg.paths.data_root)
        self.charset = Charset.load(cfg.data.charset or root / "manifests" / "charset.json")
        self.packs = [root / p for p in cfg.data.train]
        self.limits = {str(root / k): v for k, v in cfg.data.train_limits.items()}
        self.ctc = nn.CTCLoss(blank=Charset.BLANK, zero_infinity=True)

    def model(self) -> nn.Module:
        return build_recognizer(self.charset.num_classes, self.cfg.model).to(self.device)

    def dataset(self, augment: bool, subset: int | None = None) -> RecLineDataset:
        d = self.cfg.data
        aug = Augmenter(d.augmentation, d.augmentation_params) if augment else None
        return RecLineDataset(self.packs, self.charset, d.image_height, d.max_width, aug,
                              self.cfg.seed, self.limits, subset)  # fmt: skip

    def loader(self, ds: RecLineDataset, batch: int, shuffle: bool, epoch: int = 0) -> Any:
        sampler = WidthBucketBatchSampler(ds.widths, batch, self.cfg.seed, shuffle=shuffle)
        sampler.set_epoch(epoch)
        w = self.cfg.data.num_workers
        gen = make_generator(self.cfg.seed + epoch)
        return torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate,
                                           num_workers=w, pin_memory=self.device.type == "cuda",
                                           generator=gen)  # fmt: skip


def _step(p: Parts, model: nn.Module, opt: torch.optim.Optimizer, batch: dict[str, Any]) -> float:
    loss, _, _ = ctc_loss(model, batch, p.amp, p.device, p.ctc)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), p.cfg.train.grad_clip_norm or math.inf)
    opt.step()
    return float(loss.detach())


def overfit(p: Parts, out: Path) -> dict[str, Any]:
    ds = p.dataset(augment=False, subset=OVERFIT_LINES)
    model = p.model()
    opt = torch.optim.AdamW(model.parameters(), lr=p.cfg.train.learning_rate)
    curve, step, cer = [], 0, 1.0
    t0 = time.monotonic()
    while step < OVERFIT_MAX_STEPS:
        for batch in p.loader(ds, 16, shuffle=True, epoch=step):
            _step(p, model, opt, batch)
            step += 1
        if step % 40 < 4:  # about every 40 steps (4 batches per pass)
            res = evaluate(model, p.loader(ds, 32, shuffle=False), p.charset, p.amp, p.device,
                           p.ctc)  # fmt: skip
            cer = res.cer
            curve.append({"step": step, "cer": cer, "loss": res.loss})
            if cer <= OVERFIT_TARGET_CER:
                break
    return {"lines": len(ds), "steps": step, "final_cer": cer, "target_cer": OVERFIT_TARGET_CER,
            "passed": cer <= OVERFIT_TARGET_CER, "seconds": round(time.monotonic() - t0, 1),
            "curve": curve}  # fmt: skip


def lr_range(p: Parts, out: Path) -> dict[str, Any]:
    ds = p.dataset(augment=True)
    model = p.model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR_START,
                            weight_decay=p.cfg.train.weight_decay)  # fmt: skip
    factor = (LR_END / LR_START) ** (1 / LR_STEPS)
    lrs, losses, smooth, best = [], [], None, math.inf
    for i, batch in enumerate(p.loader(ds, p.cfg.train.batch_size, shuffle=True)):
        lr = LR_START * factor**i
        for g in opt.param_groups:
            g["lr"] = lr
        loss = _step(p, model, opt, batch)
        if not math.isfinite(loss):
            break
        smooth = loss if smooth is None else 0.95 * smooth + 0.05 * loss
        lrs.append(lr)
        losses.append(smooth)
        best = min(best, smooth)
        if i >= LR_STEPS or (i > 20 and smooth > 4 * best):
            break
    k = int(np.argmin(losses))
    suggested = lrs[k] / 10
    _plot_lr(lrs, losses, suggested, p.cfg.train.learning_rate, out / "lr_range.png")
    return {"steps": len(lrs), "lr_at_min_loss": lrs[k], "suggested_lr": suggested,
            "configured_lr": p.cfg.train.learning_rate,
            "configured_within_3x_of_suggestion":
                suggested / 3 <= p.cfg.train.learning_rate <= suggested * 3}  # fmt: skip


def _plot_lr(lrs: list[float], losses: list[float], suggested: float, configured: float,
             path: Path) -> None:  # fmt: skip
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lrs, losses)
    ax.axvline(suggested, ls="--", c="g", label=f"suggested {suggested:.1e}")
    ax.axvline(configured, ls=":", c="r", label=f"configured {configured:.1e}")
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("smoothed CTC loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def dry_run(p: Parts, out: Path, minutes: float) -> dict[str, Any]:
    ds = p.dataset(augment=True)
    model = p.model()
    opt = torch.optim.AdamW(model.parameters(), lr=p.cfg.train.learning_rate,
                            weight_decay=p.cfg.train.weight_decay)  # fmt: skip
    reset_peak(p.device)
    start, wait, lines, steps, losses = time.monotonic(), 0.0, 0, 0, []
    temps: list[float] = []
    it = iter(p.loader(ds, p.cfg.train.batch_size, shuffle=True))
    while time.monotonic() - start < minutes * 60:
        t0 = time.monotonic()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(p.loader(ds, p.cfg.train.batch_size, shuffle=True, epoch=steps))
            batch = next(it)
        wait += time.monotonic() - t0
        losses.append(_step(p, model, opt, batch))
        lines += len(batch["keys"])
        steps += 1
        if steps % 200 == 0:
            snap = snapshot(p.device).to_dict()
            if snap.get("gpu_temperature_c") is not None:
                temps.append(snap["gpu_temperature_c"])
    if p.device.type == "cuda":
        torch.cuda.synchronize()
    total = time.monotonic() - start
    snap = snapshot(p.device).to_dict()
    steps_per_epoch = len(WidthBucketBatchSampler(ds.widths, p.cfg.train.batch_size, p.cfg.seed))
    return {
        "minutes": round(total / 60, 2), "steps": steps, "lines": lines,
        "lines_per_s": round(lines / total, 1), "ms_per_step": round(1000 * total / steps, 1),
        "data_wait_share": round(wait / total, 3),
        "peak_vram_mib": round(snap.get("cuda_peak_allocated_mib") or 0, 1),
        "gpu_temperature_c_max": max(temps) if temps else snap.get("gpu_temperature_c"),
        "loss_first_last": [losses[0], float(np.mean(losses[-50:]))],
        "finite": all(math.isfinite(x) for x in losses),
        "estimated_minutes_per_epoch": round(steps_per_epoch * total / steps / 60, 1),
        "parameters": count_parameters(model),
    }  # fmt: skip


def write_markdown(results: dict[str, Any], path: Path) -> None:
    lines = [f"# Sanity sequence: {results['experiment_id']}", "",
             "Generated by `scripts/rec_sanity.py`; do not edit by hand.", "",
             f"Device {results['device']}, precision {results['precision']}, git "
             f"`{results['git']}`.", ""]  # fmt: skip
    if "overfit" in results:
        o = results["overfit"]
        lines += [f"- **Overfit {o['lines']} lines**: CER {100 * o['final_cer']:.2f}% after "
                  f"{o['steps']} steps (target <= {100 * o['target_cer']:.0f}%): "
                  f"{'PASS' if o['passed'] else 'FAIL'}"]  # fmt: skip
    if "lr_range" in results:
        r = results["lr_range"]
        lines += [f"- **LR range test**: lowest smoothed loss at {r['lr_at_min_loss']:.2e}; "
                  f"suggested {r['suggested_lr']:.2e}, configured {r['configured_lr']:.2e} "
                  f"({'within' if r['configured_within_3x_of_suggestion'] else 'outside'} 3x). "
                  "![lr range](lr_range.png)"]  # fmt: skip
    if "dry_run" in results:
        d = results["dry_run"]
        lines += [f"- **Dry run** ({d['minutes']} min): {d['lines_per_s']} lines/s, "
                  f"{d['ms_per_step']} ms/step, data wait {100 * d['data_wait_share']:.0f}%, "
                  f"peak VRAM {d['peak_vram_mib']} MiB, GPU max {d['gpu_temperature_c_max']} C, "
                  f"loss finite: {d['finite']}; about {d['estimated_minutes_per_epoch']} min "
                  "per epoch"]  # fmt: skip
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--checks", nargs="+", default=["overfit", "lr_range", "dry_run"],
                        choices=["overfit", "lr_range", "dry_run"])  # fmt: skip
    parser.add_argument("--dry-run-minutes", type=float, default=10.0)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    cfg.require_run_identity()
    assert cfg.experiment_id is not None
    out = Path(cfg.paths.reports_dir) / "training" / "sanity" / cfg.experiment_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sanity.json"
    results: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    p = Parts(cfg)
    from glypharis.utils.gitinfo import git_info

    results.update(experiment_id=cfg.experiment_id, device=str(p.device), precision=p.amp.name,
                   git=git_info().commit)  # fmt: skip
    for check in args.checks:
        if check == "overfit":
            results["overfit"] = overfit(p, out)
        elif check == "lr_range":
            results["lr_range"] = lr_range(p, out)
        else:
            results["dry_run"] = dry_run(p, out, args.dry_run_minutes)
        path.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
        print(check, json.dumps({k: v for k, v in results[check].items() if k != "curve"}))
    write_markdown(results, out / "sanity.md")
    ok = results.get("overfit", {}).get("passed", True) and results.get("dry_run", {}).get(
        "finite", True)  # fmt: skip
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
