"""MobileNetV3 trunk for GLYPHARIS-Det (ARCHITECTURE §3, ADR-0007).

The torchvision model is always built with ``weights=None``, so nothing is ever downloaded. ImageNet
initialisation reads a registered file (``data/sources.yaml`` → ``imagenet_backbones``) whose
SHA-256 is checked before ``torch.load`` touches it; a trained detector checkpoint never needs it.

Four feature maps are tapped at strides 4, 8, 16 and 32 (`BACKBONE_TAPS`). Their channel counts are
read from the built modules, never from a table, so a torchvision change cannot silently mismatch
the neck. Every squeeze-excitation ``AdaptiveAvgPool2d(1)`` is replaced by `GlobalMean`: the maths
is identical and the state-dict keys are unchanged, but its CUDA backward is deterministic and it
exports to ONNX as ReduceMean.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, Literal

import torch
from torch import nn
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
from torchvision.ops.misc import SqueezeExcitation

from glypharis.data.download import sha256_file

BackboneName = Literal["mobilenet_v3_large", "mobilenet_v3_small"]

# Indices into ``features`` whose outputs feed the neck, at strides 4, 8, 16 and 32.
BACKBONE_TAPS: Final[dict[str, tuple[int, int, int, int]]] = {
    "mobilenet_v3_large": (3, 6, 12, 16),
    "mobilenet_v3_small": (1, 3, 8, 12),
}

_BUILDERS: Final[dict[str, Callable[..., nn.Module]]] = {
    "mobilenet_v3_large": mobilenet_v3_large,
    "mobilenet_v3_small": mobilenet_v3_small,
}

DOWNLOAD_HINT = "python scripts/download_datasets.py --dataset imagenet_backbones"


class WeightsError(RuntimeError):
    """Raised when backbone weights are missing, fail their hash check or do not fit the trunk."""


def load_imagenet_state_dict(path: Path, sha256: str) -> dict[str, torch.Tensor]:
    """Read a torchvision MobileNetV3 checkpoint after checking its SHA-256.

    Returns the ``features.*`` tensors with the prefix stripped, ready for a strict load into
    `MobileNetV3Features.features`. The hash is checked before ``torch.load`` so an unexpected file
    is never unpickled, and ``weights_only=True`` limits what a matching file can contain anyway.
    """
    path = Path(path)
    if not path.is_file():
        raise WeightsError(
            f"backbone weights not found: {path}; install them with `{DOWNLOAD_HINT}`"
        )
    actual = sha256_file(path)
    if actual != sha256.lower():
        raise WeightsError(f"backbone weights {path} have SHA-256 {actual}, expected {sha256}")
    state = torch.load(path, weights_only=True, map_location="cpu")
    prefix = "features."
    features = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not features:
        raise WeightsError(f"{path} holds no '{prefix}*' tensors; not a MobileNetV3 checkpoint")
    return features


class GlobalMean(nn.Module):
    """Global average pool as a plain mean: same result as ``AdaptiveAvgPool2d(1)``, but with a
    deterministic CUDA backward."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean((2, 3), keepdim=True)


class MobileNetV3Features(nn.Module):
    """The MobileNetV3 ``features`` trunk returning the four tapped maps (C2, C3, C4, C5).

    ``state_dict`` (from `load_imagenet_state_dict`) is loaded strictly; without it the trunk keeps
    torchvision's initialisation, which callers seed beforehand. BatchNorm keeps torchvision's eps
    0.001 and momentum 0.01.
    """

    out_channels: tuple[int, int, int, int]

    def __init__(
        self, name: BackboneName, state_dict: Mapping[str, torch.Tensor] | None = None
    ) -> None:
        super().__init__()
        if name not in BACKBONE_TAPS:
            raise ValueError(f"unknown backbone {name!r}; expected one of {sorted(BACKBONE_TAPS)}")
        self.name = name
        self.taps = BACKBONE_TAPS[name]
        # The classifier is discarded; both trunks end exactly on their last tap.
        trunk = _BUILDERS[name](weights=None).features
        if not isinstance(trunk, nn.Sequential):
            raise TypeError(
                f"torchvision {name}.features is {type(trunk).__name__}, not Sequential"
            )
        self.features = trunk
        if state_dict is not None:
            try:
                self.features.load_state_dict(state_dict, strict=True)
            except RuntimeError as exc:
                raise WeightsError(f"weights do not fit {name}: {exc}") from exc
        for module in self.features.modules():
            if isinstance(module, SqueezeExcitation):
                module.avgpool = GlobalMean()
        c2, c3, c4, c5 = (_out_channels(self.features[i]) for i in self.taps)
        self.out_channels = (c2, c3, c4, c5)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outs: list[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.taps:
                outs.append(x)
        return outs[0], outs[1], outs[2], outs[3]


def _out_channels(layer: nn.Module) -> int:
    channels = getattr(layer, "out_channels", None)
    if not isinstance(channels, int):
        raise TypeError(f"{type(layer).__name__} has no integer out_channels")
    return channels
