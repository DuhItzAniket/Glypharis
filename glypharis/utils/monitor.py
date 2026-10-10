"""Resource snapshots: VRAM, RAM, GPU temperature and clocks.

Snapshots are cheap enough to take every logging step. GPU telemetry comes from NVML when it is
available; every field degrades to ``None`` instead of raising, so the same code runs in CI.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch

log = logging.getLogger(__name__)

_MIB = 1024**2


@dataclass(frozen=True)
class ResourceSnapshot:
    process_rss_mib: float
    system_ram_used_mib: float
    system_ram_total_mib: float
    cuda_allocated_mib: float | None = None
    cuda_reserved_mib: float | None = None
    cuda_peak_allocated_mib: float | None = None
    cuda_free_mib: float | None = None
    cuda_total_mib: float | None = None
    gpu_temperature_c: int | None = None
    gpu_sm_clock_mhz: int | None = None
    gpu_utilization_pct: int | None = None
    gpu_power_w: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Nvml:
    """Lazy NVML handle; disables itself on the first failure."""

    def __init__(self) -> None:
        self._handle: Any = None
        self._failed = False

    def handle(self, index: int) -> Any:
        if self._failed:
            return None
        if self._handle is None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception as exc:  # NVML missing, no driver, no GPU
                log.debug("NVML unavailable: %s", exc)
                self._failed = True
                return None
        return self._handle


_nvml = _Nvml()


def _gpu_telemetry(index: int) -> dict[str, Any]:
    handle = _nvml.handle(index)
    if handle is None:
        return {}
    import pynvml

    out: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        out["gpu_temperature_c"] = int(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        )
    with contextlib.suppress(Exception):
        out["gpu_sm_clock_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
    with contextlib.suppress(Exception):
        out["gpu_utilization_pct"] = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    with contextlib.suppress(Exception):
        out["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    return out


def snapshot(device: torch.device | None = None) -> ResourceSnapshot:
    """Take one resource snapshot. CUDA fields are filled only for a CUDA device."""
    vm = psutil.virtual_memory()
    fields: dict[str, Any] = {
        "process_rss_mib": psutil.Process().memory_info().rss / _MIB,
        "system_ram_used_mib": (vm.total - vm.available) / _MIB,
        "system_ram_total_mib": vm.total / _MIB,
    }
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(index)
        fields.update(
            cuda_allocated_mib=torch.cuda.memory_allocated(index) / _MIB,
            cuda_reserved_mib=torch.cuda.memory_reserved(index) / _MIB,
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(index) / _MIB,
            cuda_free_mib=free / _MIB,
            cuda_total_mib=total / _MIB,
        )
        fields.update(_gpu_telemetry(index))
    return ResourceSnapshot(**fields)


def reset_peak(device: torch.device) -> None:
    """Reset the CUDA peak-memory counter (start of a dry run or an epoch)."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
