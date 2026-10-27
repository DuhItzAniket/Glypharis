"""GLYPHARIS-Det: a DBNet-style text-line detector on MobileNetV3 features (ARCHITECTURE §3).

Input: uint8 (or float in [0, 255]) RGB images ``(N, 3, H, W)`` with H and W multiples of 32.
Normalisation happens inside the model, so training, validation, `TextLineDetector` and an ONNX
export all feed the same raw pixels and cannot drift apart; ``PAD_RGB`` padding normalises to ~0.

Backbone taps at strides 4-32 go through DBNet's FPN neck (no BatchNorm) into a fused map at H/4,
then two identical heads upsample it back to H: one for the probability map P and one for the
threshold map T. Both heads end in a transposed convolution that runs in fp32 with autocast off:
the binary map multiplies (P - T) by k = 50, so bf16 logits would add visible jitter.

`DBNet.forward` returns probabilities and never runs the threshold head; it is the only path used
at inference. `DBNet.forward_train` returns both logit maps for the loss.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from glypharis.config.schema import DetModelConfig
from glypharis.models.det.backbone import (
    MobileNetV3Features,
    WeightsError,
    load_imagenet_state_dict,
)

IMAGENET_MEAN_255: Final = (123.675, 116.28, 103.53)
IMAGENET_STD_255: Final = (58.395, 57.12, 57.375)


class DetLogits(NamedTuple):
    prob: torch.Tensor  # (N, 1, H, W) float32 logits
    thresh: torch.Tensor  # (N, 1, H, W) float32 logits


def _up(x: torch.Tensor, factor: int) -> torch.Tensor:
    # Nearest upsampling by an integer factor is exact here because H and W are multiples of 32.
    return F.interpolate(x, scale_factor=factor, mode="nearest")


class DBFPN(nn.Module):
    """DBNet's SegDetector neck: 1x1 laterals, a nearest top-down path, 3x3 smoothing to
    ``inner / 4`` channels per level, then everything upsampled to H/4 and concatenated."""

    def __init__(self, in_channels: Sequence[int], inner: int) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"DBFPN takes 4 feature maps, got {len(in_channels)}")
        if inner <= 0 or inner % 4:
            raise ValueError(f"inner channels must be a positive multiple of 4, got {inner}")
        self.lateral = nn.ModuleList(nn.Conv2d(c, inner, 1, bias=False) for c in in_channels)
        self.smooth = nn.ModuleList(
            nn.Conv2d(inner, inner // 4, 3, padding=1, bias=False) for _ in in_channels
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        l2, l3, l4, l5 = (conv(f) for conv, f in zip(self.lateral, feats, strict=True))
        o4 = l4 + _up(l5, 2)
        o3 = l3 + _up(o4, 2)
        o2 = l2 + _up(o3, 2)
        s2, s3, s4, s5 = self.smooth
        return torch.cat([_up(s5(l5), 8), _up(s4(o4), 4), _up(s3(o3), 2), s2(o2)], dim=1)


class DBHead(nn.Module):
    """One DBNet head: 3x3 conv, then two stride-2 transposed convolutions back to full size.

    The final layer runs in fp32 with autocast disabled, so the logits are always fp32.
    """

    def __init__(self, inner: int) -> None:
        super().__init__()
        mid = inner // 4
        self.body = nn.Sequential(
            nn.Conv2d(inner, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, mid, 2, stride=2),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.out = nn.ConvTranspose2d(mid, 1, 2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.out(x.float())


class DBNet(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, backbone: MobileNetV3Features, inner_channels: int = 96) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = DBFPN(backbone.out_channels, inner_channels)
        self.prob_head = DBHead(inner_channels)
        self.thresh_head = DBHead(inner_channels)
        # Non-persistent: the constants are code, not weights, and stay out of checkpoints.
        mean = torch.tensor(IMAGENET_MEAN_255, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD_255, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)
        for part in (self.neck, self.prob_head, self.thresh_head):
            part.apply(_init_dbnet)

    def _fused(self, images: torch.Tensor) -> torch.Tensor:
        x = (images.float() - self.mean) / self.std
        return self.neck(self.backbone(x))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Text probabilities ``(N, 1, H, W)`` in fp32. The threshold head is never executed.

        No shape checks and no data-dependent control flow, so an export keeps N, H and W dynamic.
        """
        return torch.sigmoid(self.prob_head(self._fused(images)))

    def forward_train(self, images: torch.Tensor) -> DetLogits:
        """Probability and threshold logits for the loss (training, and validation loss)."""
        fused = self._fused(images)
        return DetLogits(self.prob_head(fused), self.thresh_head(fused))


def _init_dbnet(m: nn.Module) -> None:
    # DBNet's reference initialisation for the neck and heads; the backbone keeps its own.
    if isinstance(m, nn.Conv2d | nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.constant_(m.bias, 1e-4)


def build_detector(cfg: DetModelConfig, data_root: Path, load_pretrained: bool = True) -> DBNet:
    """Build GLYPHARIS-Det from its config.

    With ``init: imagenet`` and ``load_pretrained`` the backbone loads the pinned file at
    ``data_root / cfg.weights`` after its hash check. ``load_pretrained=False`` skips that file (for
    resume, `TextLineDetector.from_checkpoint` and tests, where a checkpoint then sets every
    weight). The model is converted to channels-last when ``cfg.channels_last`` is set.
    """
    state = None
    if cfg.init == "imagenet" and load_pretrained:
        if not cfg.weights or not cfg.weights_sha256:
            raise WeightsError(
                "init 'imagenet' needs det.model.weights and det.model.weights_sha256"
            )
        state = load_imagenet_state_dict(Path(data_root) / cfg.weights, cfg.weights_sha256)
    model = DBNet(MobileNetV3Features(cfg.backbone, state), cfg.inner_channels)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def count_parameters(model: nn.Module, inference_only: bool = False) -> int:
    """Number of parameters; ``inference_only`` leaves out the threshold head, which only the loss
    uses and which `DBNet.forward` (and so an export) never runs."""
    skip: set[int] = set()
    if inference_only:
        head = getattr(model, "thresh_head", None)
        if not isinstance(head, nn.Module):
            raise ValueError("inference_only needs a model with a thresh_head")
        skip = {id(p) for p in head.parameters()}
    return sum(p.numel() for p in model.parameters() if id(p) not in skip)
