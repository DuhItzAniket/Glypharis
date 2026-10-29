"""Optimizer and learning-rate schedule from a run config, shared by every trainer.

A step is an optimizer step; the schedule is sized from the whole run's step budget, so a run cut
short by ``train.max_steps`` follows the full schedule (`glypharis.training.loop`).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from glypharis.config.schema import ConfigError, RunConfig


def make_optimizer(cfg: RunConfig, model: nn.Module) -> torch.optim.Optimizer:
    t = cfg.train
    params, lr, wd = model.parameters(), t.learning_rate, t.weight_decay
    if t.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if t.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)


def make_scheduler(
    cfg: RunConfig, optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    t = cfg.train
    if t.scheduler == "none":
        return None
    if t.scheduler == "onecycle":
        pct = t.warmup_steps / total_steps if t.warmup_steps else 0.3
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=t.learning_rate, total_steps=total_steps, pct_start=min(pct, 0.5)
        )
    if t.scheduler == "cosine":
        warm = t.warmup_steps

        def factor(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            progress = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    raise ConfigError(f"train.scheduler {t.scheduler!r} is not supported by the recognizer trainer")
