from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from glypharis.config.schema import DetLossConfig
from glypharis.models.det.dbnet import DetLogits
from glypharis.models.det.loss import DBLoss, dice_loss, masked_l1, ohem_bce

EPS = 1e-6


def _bce(logit: float, target: float) -> float:
    p = 1.0 / (1.0 + math.exp(-logit))
    return -(target * math.log(p) + (1.0 - target) * math.log(1.0 - p))


def _topk_reference(logits, gt, mask, ratio, eps):
    """OHEM written the obvious way, with a host sync and topk."""
    loss = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    pos = (gt * mask).bool()
    neg = ((1 - gt) * mask).bool()
    n_pos = int(pos.sum())
    n_neg = min(int(neg.sum()), int(n_pos * ratio))
    hard = loss[neg].topk(n_neg).values.sum() if n_neg else loss.new_zeros(())
    return (loss[pos].sum() + hard) / (n_pos + n_neg + eps), n_pos, n_neg


def test_ohem_hand_built_case():
    # One positive, six candidate negatives, one masked-out pixel with the largest loss of all.
    logits = torch.tensor([[[[2.0, 3.0, -1.0, 0.5], [1.5, -2.0, 0.0, 9.0]]]], requires_grad=True)
    gt = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]]]])
    loss, n_pos, n_neg = ohem_bce(logits, gt, mask, ratio=3.0, eps=EPS)
    assert float(n_pos) == 1.0 and float(n_neg) == 3.0  # min(6 negatives, 3 * 1)
    # The three hardest negatives are the logits 3.0, 1.5 and 0.5; 9.0 is masked out.
    expected = (_bce(2.0, 1) + _bce(3.0, 0) + _bce(1.5, 0) + _bce(0.5, 0)) / (4 + EPS)
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-6)
    loss.backward()
    chosen = logits.grad.flatten() != 0
    assert chosen.tolist() == [True, True, False, True, True, False, False, False]


def test_ohem_negatives_capped_by_what_exists():
    logits = torch.tensor([[[[0.3, -0.2, 1.0]]]])
    gt = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    mask = torch.ones_like(gt)
    loss, n_pos, n_neg = ohem_bce(logits, gt, mask, ratio=3.0, eps=EPS)
    assert (float(n_pos), float(n_neg)) == (2.0, 1.0)
    expected = (_bce(0.3, 1) + _bce(-0.2, 1) + _bce(1.0, 0)) / (3 + EPS)
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-6)


def test_ohem_without_positives_is_zero():
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    gt = torch.zeros(2, 1, 8, 8)
    loss, n_pos, n_neg = ohem_bce(logits, gt, torch.ones_like(gt), ratio=3.0, eps=EPS)
    assert float(loss.detach()) == 0.0 and float(n_pos) == 0.0 and float(n_neg) == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ratio", [1.0, 3.0, 50.0])
def test_sort_based_ohem_equals_topk(seed, ratio):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(3, 1, 16, 24, generator=g) * 3
    gt = (torch.rand(3, 1, 16, 24, generator=g) < 0.1).float()
    mask = (torch.rand(3, 1, 16, 24, generator=g) < 0.9).float()
    loss, n_pos, n_neg = ohem_bce(logits, gt, mask, ratio, EPS)
    ref, ref_pos, ref_neg = _topk_reference(logits, gt, mask, ratio, EPS)
    assert (int(n_pos), int(n_neg)) == (ref_pos, ref_neg)
    torch.testing.assert_close(loss, ref, rtol=1e-6, atol=0)


def test_dice_value_and_mask():
    pred = torch.tensor([0.9, 0.2, 0.6, 0.7]).view(1, 1, 2, 2)
    gt = torch.tensor([1.0, 0.0, 1.0, 1.0]).view(1, 1, 2, 2)
    mask = torch.tensor([1.0, 1.0, 1.0, 0.0]).view(1, 1, 2, 2)
    inter, union = 0.9 + 0.6, (0.9 + 0.2 + 0.6) + 2.0
    assert float(dice_loss(pred, gt, mask, EPS)) == pytest.approx(1 - 2 * inter / (union + EPS))
    # Whatever sits under mask 0 does not count.
    pred2, gt2 = pred.clone(), gt.clone()
    pred2[0, 0, 1, 1], gt2[0, 0, 1, 1] = 0.0, 0.0
    assert torch.equal(dice_loss(pred2, gt2, mask, EPS), dice_loss(pred, gt, mask, EPS))
    assert float(dice_loss(pred, gt, torch.zeros_like(mask), EPS)) == 0.0
    assert float(dice_loss(gt, gt, torch.ones_like(gt), EPS)) == pytest.approx(0.0, abs=1e-6)


def test_masked_l1_value_and_mask():
    pred = torch.tensor([0.5, 0.2, 0.9, 0.1]).view(1, 1, 2, 2)
    gt = torch.tensor([0.3, 0.7, 0.9, 0.6]).view(1, 1, 2, 2)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0]).view(1, 1, 2, 2)
    assert float(masked_l1(pred, gt, mask, EPS)) == pytest.approx((0.2 + 0.5) / (2 + EPS))
    pred2 = pred.clone()
    pred2[0, 0, 1] = 0.0
    assert torch.equal(masked_l1(pred2, gt, mask, EPS), masked_l1(pred, gt, mask, EPS))
    assert float(masked_l1(pred, gt, torch.zeros_like(mask), EPS)) == 0.0


def _targets(n: int = 2, h: int = 32, w: int = 32, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    shrink = torch.zeros(n, 1, h, w, dtype=torch.uint8)
    shrink[:, :, 10:14, 4:28] = 1
    shrink_mask = (torch.rand(n, 1, h, w, generator=g) < 0.95).to(torch.uint8)
    thresh = 0.3 + 0.4 * torch.rand(n, 1, h, w, generator=g)
    thresh_mask = torch.zeros(n, 1, h, w, dtype=torch.uint8)
    thresh_mask[:, :, 6:18, 0:32] = 1
    return shrink, shrink_mask, thresh, thresh_mask


def test_dbloss_combines_the_parts_with_the_configured_weights():
    cfg = DetLossConfig(bce_weight=2.0, alpha=0.5, beta=10.0)
    g = torch.Generator().manual_seed(1)
    logits = DetLogits(
        torch.randn(2, 1, 32, 32, generator=g), torch.randn(2, 1, 32, 32, generator=g)
    )
    shrink, shrink_mask, thresh, thresh_mask = _targets()
    parts = DBLoss(cfg)(logits, shrink, shrink_mask, thresh, thresh_mask)
    bce, n_pos, n_neg = ohem_bce(logits.prob, shrink, shrink_mask, cfg.ohem_ratio, cfg.eps)
    binary = torch.sigmoid(cfg.k * (torch.sigmoid(logits.prob) - torch.sigmoid(logits.thresh)))
    dice = dice_loss(binary, shrink, shrink_mask, cfg.eps)
    l1 = masked_l1(torch.sigmoid(logits.thresh), thresh, thresh_mask, cfg.eps)
    assert (
        torch.equal(parts.bce, bce) and torch.equal(parts.dice, dice) and torch.equal(parts.l1, l1)
    )
    assert torch.equal(parts.n_pos, n_pos) and torch.equal(parts.n_neg, n_neg)
    torch.testing.assert_close(parts.total, 2.0 * bce + 0.5 * dice + 10.0 * l1)


def test_dbloss_parts_are_fp32_for_bf16_logits_and_gradients_are_finite():
    conv = torch.nn.Conv2d(3, 2, 3, padding=1)
    x = torch.randn(2, 3, 32, 32)
    shrink, shrink_mask, thresh, thresh_mask = _targets()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = conv(x)
        assert out.dtype == torch.bfloat16
        parts = DBLoss(DetLossConfig())(
            DetLogits(out[:, :1], out[:, 1:]), shrink, shrink_mask, thresh, thresh_mask
        )
    for name, value in parts._asdict().items():
        assert value.dtype == torch.float32, name
        assert value.dim() == 0, name
    parts.total.backward()
    assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()
    assert float(conv.weight.grad.abs().sum()) > 0


def test_dbloss_refuses_targets_that_would_broadcast():
    logits = DetLogits(torch.zeros(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))
    good = torch.zeros(2, 1, 8, 8)
    with pytest.raises(ValueError, match="shrink_map"):
        DBLoss(DetLossConfig())(logits, torch.zeros(2, 8, 8), good, good, good)
