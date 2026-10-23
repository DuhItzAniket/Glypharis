"""DBNet loss (ARCHITECTURE §3): OHEM BCE on P, Dice on the approximate binary map, masked L1 on T.

    L = bce_weight * L_s + alpha * L_b + beta * L_t          (1, 1, 10 as pre-registered)

The forward pass may run under bf16 autocast; everything here runs with autocast disabled on fp32
copies of the logits and targets, because the binary map sigmoid(k * (P - T)) with k = 50 magnifies
any rounding in P - T. Sums run over the whole batch, as in DBNet.

OHEM keeps every positive and the ``ohem_ratio * n_pos`` hardest negatives. The selection is a sort
and a tensor comparison, never ``.item()`` or ``topk(k)`` with a Python k, so a training step does
not wait on the GPU, and gradients reach only the selected pixels.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from glypharis.config.schema import DetLossConfig
from glypharis.models.det.dbnet import DetLogits
from glypharis.utils.device import fp32_region


class DetLossParts(NamedTuple):
    total: torch.Tensor
    bce: torch.Tensor
    dice: torch.Tensor
    l1: torch.Tensor
    n_pos: torch.Tensor  # positive pixels counted by OHEM (scalar, fp32)
    n_neg: torch.Tensor  # hardest negatives kept by OHEM (scalar, fp32)


def ohem_bce(
    logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, ratio: float, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BCE over all positives and the ``floor(ratio * n_pos)`` hardest negatives in the batch.

    Returns ``(loss, n_pos, n_neg)``. With no positives no negative is kept either and the loss is
    0, as in DBNet.
    """
    logits, gt, mask = logits.float(), gt.float(), mask.float()
    # BCE-with-logits, never binary_cross_entropy on probabilities: it is autocast-safe and stable.
    loss = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    pos = gt * mask
    neg = (1.0 - gt) * mask
    n_pos = pos.sum()
    n_neg = torch.minimum(neg.sum(), torch.floor(ratio * n_pos))
    hardest = torch.sort((loss * neg).flatten(), descending=True).values
    keep = torch.arange(hardest.numel(), device=hardest.device) < n_neg.long()
    total = ((loss * pos).sum() + (hardest * keep).sum()) / (n_pos + n_neg + eps)
    return total, n_pos, n_neg


def dice_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    """``1 - 2 * sum(pred * gt * mask) / (sum(pred * mask) + sum(gt * mask) + eps)``; 0 when the
    mask is empty, so a fully don't-care batch adds nothing."""
    pred, gt, mask = pred.float(), gt.float(), mask.float()
    inter = (pred * gt * mask).sum()
    union = (pred * mask).sum() + (gt * mask).sum() + eps
    loss = 1.0 - 2.0 * inter / union
    # torch.where instead of a Python branch keeps this free of host syncs.
    return torch.where(mask.sum() > 0, loss, torch.zeros_like(loss))


def masked_l1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    """Mean absolute error over the mask; 0 when the mask is empty."""
    pred, gt, mask = pred.float(), gt.float(), mask.float()
    return ((pred - gt).abs() * mask).sum() / (mask.sum() + eps)


class DBLoss(nn.Module):
    def __init__(self, cfg: DetLossConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        logits: DetLogits,
        shrink_map: torch.Tensor,
        shrink_mask: torch.Tensor,
        thresh_map: torch.Tensor,
        thresh_mask: torch.Tensor,
    ) -> DetLossParts:
        """Loss parts for one batch. Targets are ``(N, 1, H, W)`` like the logits (any dtype)."""
        shape = logits.prob.shape
        for name, target in (
            ("thresh logits", logits.thresh),
            ("shrink_map", shrink_map),
            ("shrink_mask", shrink_mask),
            ("thresh_map", thresh_map),
            ("thresh_mask", thresh_mask),
        ):
            # A (N, H, W) target would broadcast against (N, 1, H, W) logits without an error.
            if target.shape != shape:
                raise ValueError(f"{name} has shape {tuple(target.shape)}, expected {tuple(shape)}")
        c = self.cfg
        with fp32_region(logits.prob.device.type):
            p, t = logits.prob.float(), logits.thresh.float()
            gt, mask = shrink_map.float(), shrink_mask.float()
            bce, n_pos, n_neg = ohem_bce(p, gt, mask, c.ohem_ratio, c.eps)
            prob, thresh = torch.sigmoid(p), torch.sigmoid(t)
            binary = torch.sigmoid(c.k * (prob - thresh))
            dice = dice_loss(binary, gt, mask, c.eps)
            l1 = masked_l1(thresh, thresh_map.float(), thresh_mask.float(), c.eps)
            total = c.bce_weight * bce + c.alpha * dice + c.beta * l1
        return DetLossParts(total, bce, dice, l1, n_pos, n_neg)
