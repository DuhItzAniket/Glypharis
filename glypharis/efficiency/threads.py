"""Thread counts for Phase 16 profiles, set before torch loads, and ONNX Runtime sessions.

    from glypharis.efficiency.threads import set_threads
    threads = set_threads(10)          # at the top of a script, before numpy and torch
    import torch

The OpenMP and MKL variables (`THREAD_ENV`, the ones `run_baselines` records) are read once, when
the runtime that uses them loads, so they must be in place before numpy or torch is imported;
`set_threads` refuses to run after that unless told the run is not a measurement. It then sets
torch's intra-op and inter-op pools and checks what torch reports. The inter-op pool can be sized
only once per process, before it starts, so a profile runs in a fresh process.

`ort_session` builds a CPU session the way `glypharis.export.onnx` does (all graph optimisations,
memory arena and memory patterns off, GW-EFF-004) with ``threads`` intra-op threads and sequential
execution, so the inter-op pool is unused; spinning is left at ONNX Runtime's default.
`session_options` reads the options back from the session, for the profile and for
`check_session`.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Any, Final

THREAD_ENV: Final = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS")
# Modules that load an OpenMP or BLAS runtime, which reads THREAD_ENV when it loads.
EARLY_MODULES: Final = ("numpy", "torch")
SPINNING_KEY: Final = "session.intra_op.allow_spinning"


class ThreadError(RuntimeError):
    """A thread count is not what the run asked for."""


def set_thread_env(n: int) -> dict[str, str]:
    """Put ``n`` into every `THREAD_ENV` variable; returns what was set. Imports nothing."""
    if n < 1:
        raise ValueError(f"threads must be a positive integer, got {n}")
    for key in THREAD_ENV:
        os.environ[key] = str(n)
    return {key: os.environ[key] for key in THREAD_ENV}


def set_torch_threads(n: int) -> dict[str, int]:
    """Set torch's intra-op and inter-op pools to ``n``; `ThreadError` if torch reports other
    counts (the inter-op pool was already fixed in this process)."""
    import torch

    torch.set_num_threads(n)
    with contextlib.suppress(RuntimeError):  # already fixed in this process; checked below
        torch.set_num_interop_threads(n)
    got = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads()}
    if got != {"intra_op": n, "inter_op": n}:
        raise ThreadError(
            f"asked for {n} threads, torch runs {got['intra_op']} intra-op and "
            f"{got['inter_op']} inter-op; run in a fresh process"
        )
    return got


def set_threads(n: int, *, require_fresh: bool = True) -> dict[str, Any]:
    """`set_thread_env` then `set_torch_threads`; returns `thread_record` with ``requested``,
    ``preloaded`` (the `EARLY_MODULES` already loaded when it was called) and
    ``env_before_torch_import`` (none was), the shape Phase 14's profiles record as ``threads``.

    With ``require_fresh`` (the default, for measurements) a preloaded module is refused with
    `ThreadError`, since the variables would not reach its runtime; tests pass False.
    """
    preloaded = [m for m in EARLY_MODULES if m in sys.modules]
    if preloaded and require_fresh:
        raise ThreadError(
            f"{', '.join(preloaded)} already imported: set the threads at the top of the script, "
            "before numpy and torch"
        )
    set_thread_env(n)
    set_torch_threads(n)
    return {
        "requested": n,
        **thread_record(),
        "env_before_torch_import": not preloaded,
        "preloaded": preloaded,
    }


def thread_record() -> dict[str, Any]:
    """The live values: torch's ``intra_op`` and ``inter_op`` counts (None when torch is not
    imported; this never imports it) and every `THREAD_ENV` variable under ``env`` (None when
    unset)."""
    torch = sys.modules.get("torch")
    intra = torch.get_num_threads() if torch is not None else None
    inter = torch.get_num_interop_threads() if torch is not None else None
    return {
        "intra_op": intra,
        "inter_op": inter,
        "env": {key: os.environ.get(key) for key in THREAD_ENV},
    }


def check_threads(n: int) -> dict[str, Any]:
    """`ThreadError` unless every `THREAD_ENV` variable is ``n`` and torch runs ``n`` intra-op and
    ``n`` inter-op threads; returns `thread_record`. Called before and after the timed items."""
    got = thread_record()
    want_env = {key: str(n) for key in THREAD_ENV}
    if got["env"] != want_env:
        raise ThreadError(f"thread variables {got['env']}, expected {want_env}")
    if (got["intra_op"], got["inter_op"]) != (n, n):
        raise ThreadError(
            f"torch runs {got['intra_op']} intra-op and {got['inter_op']} inter-op threads, "
            f"expected {n} and {n}"
        )
    return got


def ort_session(path: str | Path, threads: int | None) -> Any:
    """An ONNX Runtime CPU session over ``path``: ``threads`` intra-op threads (None leaves ONNX
    Runtime's default, recorded as 0), sequential execution, all graph optimisations, the CPU
    memory arena and memory patterns off as in `glypharis.export.onnx`, spinning at the default."""
    import onnxruntime as ort

    if threads is not None and threads < 1:
        raise ValueError(f"threads must be a positive integer or None, got {threads}")
    opts = ort.SessionOptions()
    if threads is not None:
        opts.intra_op_num_threads = threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def session_options(session: Any) -> dict[str, Any]:
    """The options ``session`` runs with, read back from it, as a JSON-ready dict."""
    import onnxruntime as ort

    opts = session.get_session_options()
    try:
        spinning: str = opts.get_session_config_entry(SPINNING_KEY)
    except Exception:  # not set: ONNX Runtime's default applies
        spinning = "default"
    return {
        "onnxruntime": ort.__version__,
        "providers": list(session.get_providers()),
        "intra_op_num_threads": opts.intra_op_num_threads,
        "inter_op_num_threads": opts.inter_op_num_threads,
        "execution_mode": opts.execution_mode.name,
        "graph_optimization_level": opts.graph_optimization_level.name,
        "enable_cpu_mem_arena": opts.enable_cpu_mem_arena,
        "enable_mem_pattern": opts.enable_mem_pattern,
        "allow_spinning": spinning,
    }


def check_session(session: Any, threads: int | None) -> dict[str, Any]:
    """`ThreadError` unless ``session`` holds the options `ort_session` sets for ``threads``;
    returns `session_options`."""
    got = session_options(session)
    want = {
        "providers": ["CPUExecutionProvider"],
        "intra_op_num_threads": threads or 0,
        "execution_mode": "ORT_SEQUENTIAL",
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "enable_cpu_mem_arena": False,
        "enable_mem_pattern": False,
        "allow_spinning": "default",
    }
    wrong = {k: got[k] for k, v in want.items() if got[k] != v}
    if wrong:
        raise ThreadError(f"session options {wrong} differ from {want}")
    return got
