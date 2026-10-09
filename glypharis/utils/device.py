"""Device selection and the mixed-precision policy.

Precision policy (brief §23, ARCHITECTURE §4): bf16 → fp16 → fp32. bf16 needs no GradScaler; fp16
does. CPU runs use fp32: CPU autocast exists but is slow for the conv/LSTM mix we train, and CPU
runs are for tests and CI, where exact numerics matter more than speed.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import torch

DeviceRequest = Literal["auto", "cuda", "cpu"]
PrecisionRequest = Literal["bf16", "fp16", "none"]

log = logging.getLogger(__name__)


def select_device(request: DeviceRequest = "auto") -> torch.device:
    """Resolve a device request.

    ``auto`` picks CUDA when available and falls back to CPU with a warning. An explicit ``cuda``
    request fails loudly instead of silently training on the CPU for hours.
    """
    if request == "cpu":
        return torch.device("cpu")
    if request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available")
        return torch.device("cuda")
    if request == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("CUDA not available; falling back to CPU")
        return torch.device("cpu")
    raise ValueError(f"unknown device request {request!r}")


@dataclass(frozen=True)
class AmpPolicy:
    """Resolved mixed-precision settings for one run."""

    requested: str
    dtype: torch.dtype
    device_type: str

    @property
    def enabled(self) -> bool:
        return self.dtype != torch.float32

    @property
    def needs_grad_scaler(self) -> bool:
        return self.dtype == torch.float16

    @property
    def name(self) -> str:
        return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[self.dtype]

    def autocast(self) -> contextlib.AbstractContextManager[None]:
        """Autocast context for the forward pass; a no-op when running in fp32."""
        if not self.enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device_type, dtype=self.dtype)


def resolve_amp(request: PrecisionRequest, device: torch.device) -> AmpPolicy:
    """Apply the bf16 → fp16 → fp32 fallback for the given device."""
    if request not in ("bf16", "fp16", "none"):
        raise ValueError(f"unknown precision {request!r}")
    if device.type != "cuda" or request == "none":
        return AmpPolicy(request, torch.float32, device.type)
    if request == "bf16":
        if torch.cuda.is_bf16_supported():
            return AmpPolicy(request, torch.bfloat16, device.type)
        log.warning("bf16 not supported on this GPU; falling back to fp16")
    return AmpPolicy(request, torch.float16, device.type)


@contextlib.contextmanager
def fp32_region(device_type: str) -> Iterator[None]:
    """Disable autocast inside the block, e.g. around CTC loss (EXPERIMENT_PROTOCOL, R-13).

    Inputs created under autocast must still be cast with ``.float()`` by the caller.
    """
    with torch.autocast(device_type=device_type, enabled=False):
        yield
