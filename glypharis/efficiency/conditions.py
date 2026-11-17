"""Machine state at the start and end of every Phase 16 profile (Phase 16 plan, item 3).

Every profile, CPU runs included, records `run_baselines.run_conditions` (power source, plan and
mode, how busy the CPU was, the thread variables) and the GPU's state from nvidia-smi (clocks,
temperature, memory, throttle reasons) once before the warm-up items and once after the timed ones,
so thermal state and a power change during the sitting can be seen.

    conditions = ProfileConditions()
    conditions.start()                 # before the model loads: the CPU check wants an idle machine
    ...                                # warm-up and timed items
    conditions.end()
    profile.update(conditions.fields())

`fields` keeps Phase 14's shape: ``conditions`` is the start's `run_conditions` dict, which
`phase14_score.profile_problems` reads; the end and the GPU sit beside it.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Final

# nvidia-smi field -> recorded key (values without units: MHz, degrees C, MiB, %, W).
GPU_FIELDS: Final = (
    ("index", "index"),
    ("name", "name"),
    ("driver_version", "driver"),
    ("pstate", "pstate"),
    ("clocks.sm", "clock_sm_mhz"),
    ("clocks.mem", "clock_mem_mhz"),
    ("clocks.max.sm", "clock_max_sm_mhz"),
    ("clocks.max.mem", "clock_max_mem_mhz"),
    ("temperature.gpu", "temperature_c"),
    ("memory.used", "memory_used_mib"),
    ("memory.total", "memory_total_mib"),
    ("utilization.gpu", "utilization_percent"),
    ("power.draw", "power_draw_w"),
    ("clocks_throttle_reasons.active", "throttle_reasons"),
)
_TEXT_FIELDS: Final = frozenset({"name", "driver", "pstate", "throttle_reasons"})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _value(key: str, raw: str) -> Any:
    raw = raw.strip()
    if not raw or raw.startswith("[") or raw in ("N/A", "Not Supported"):
        return None
    if key in _TEXT_FIELDS:
        return raw
    try:
        number = float(raw)
    except ValueError:
        return raw
    return int(number) if number.is_integer() else number


def parse_gpu_query(text: str) -> list[dict[str, Any]]:
    """Rows of ``nvidia-smi --query-gpu=<GPU_FIELDS> --format=csv,noheader,nounits``."""
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        cells = line.split(",")
        if len(cells) != len(GPU_FIELDS):
            raise ValueError(f"nvidia-smi gave {len(cells)} fields, expected {len(GPU_FIELDS)}")
        gpus.append({key: _value(key, c) for (_, key), c in zip(GPU_FIELDS, cells, strict=True)})
    return gpus


def gpu_state() -> dict[str, Any] | None:
    """Every GPU's clocks, temperature and memory from nvidia-smi, with the query time; None when
    nvidia-smi is not installed or does not answer (a machine without an NVIDIA GPU)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    query = ",".join(field for field, _ in GPU_FIELDS)
    try:
        done = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    try:
        gpus = parse_gpu_query(done.stdout)
    except ValueError:
        return None
    return {"utc": _utc_now(), "gpus": gpus} if gpus else None


def parse_process_memory(text: str, pid: int) -> int | float | None:
    """MiB ``pid`` uses, from ``nvidia-smi --query-compute-apps=pid,used_memory
    --format=csv,noheader,nounits``; None when the process is not listed or its memory is not
    reported (Windows drivers in WDDM mode report ``[N/A]``)."""
    total: float | None = None
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != 2 or cells[0] != str(pid):
            continue
        value = _value("used_memory", cells[1])
        if isinstance(value, int | float):
            total = (total or 0) + value
    if total is None:
        return None
    return int(total) if float(total).is_integer() else total


def gpu_process_memory_mib(pid: int) -> int | float | None:
    """The GPU memory nvidia-smi reports for process ``pid`` now (MiB, summed over GPUs), not the
    whole GPU's, which includes the display and other programs; None without nvidia-smi or when
    the driver does not report it per process."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        done = subprocess.run(
            [exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return parse_process_memory(done.stdout, pid)


def run_conditions() -> dict[str, Any]:
    """`run_baselines.run_conditions` (scripts/ must be on the path, as it is for every script).

    Imported when called: run_baselines loads numpy, which must come after `threads.set_threads`.
    """
    module = importlib.import_module("run_baselines")
    conditions: dict[str, Any] = module.run_conditions()
    return conditions


def snapshot(
    cpu: Callable[[], dict[str, Any]] = run_conditions,
    gpu: Callable[[], dict[str, Any] | None] = gpu_state,
) -> dict[str, Any]:
    """``{"utc", "cpu", "gpu"}``: the CPU conditions and the GPU state now."""
    return {"utc": _utc_now(), "cpu": cpu(), "gpu": gpu()}


class ProfileConditions:
    """The conditions at a profile's start and end (`snapshot` twice)."""

    def __init__(
        self,
        cpu: Callable[[], dict[str, Any]] = run_conditions,
        gpu: Callable[[], dict[str, Any] | None] = gpu_state,
    ) -> None:
        self._cpu, self._gpu = cpu, gpu
        self.at_start: dict[str, Any] | None = None
        self.at_end: dict[str, Any] | None = None

    def start(self) -> dict[str, Any]:
        self.at_start = snapshot(self._cpu, self._gpu)
        return self.at_start

    def end(self) -> dict[str, Any]:
        if self.at_start is None:
            raise RuntimeError("end() before start()")
        self.at_end = snapshot(self._cpu, self._gpu)
        return self.at_end

    def fields(self) -> dict[str, Any]:
        """The profile's fields: ``conditions`` (the start's CPU conditions, Phase 14's key),
        ``conditions_end``, and ``gpu_state`` with ``start`` and ``end`` (None without a GPU)."""
        if self.at_start is None or self.at_end is None:
            raise RuntimeError("fields() needs start() and end()")
        return {
            "conditions": self.at_start["cpu"],
            "conditions_end": self.at_end["cpu"],
            "conditions_utc": {"start": self.at_start["utc"], "end": self.at_end["utc"]},
            "gpu_state": {"start": self.at_start["gpu"], "end": self.at_end["gpu"]},
        }
