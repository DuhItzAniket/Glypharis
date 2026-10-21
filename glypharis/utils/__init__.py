"""Seeding, device / precision policy, resource monitoring, git capture and logging.

Submodules are imported lazily: `glypharis.utils.gitinfo` and `logsetup` must stay importable
without PyTorch, because data-preparation worker processes import them and should not each
load torch (about 500 MB per process on Windows).
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "AmpPolicy": "device",
    "fp32_region": "device",
    "resolve_amp": "device",
    "select_device": "device",
    "GitInfo": "gitinfo",
    "git_info": "gitinfo",
    "JsonFormatter": "logsetup",
    "setup_logging": "logsetup",
    "ResourceSnapshot": "monitor",
    "reset_peak": "monitor",
    "snapshot": "monitor",
    "make_generator": "seed",
    "seed_everything": "seed",
    "worker_init_fn": "seed",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'glypharis.utils' has no attribute {name!r}")
    return getattr(importlib.import_module(f"glypharis.utils.{module}"), name)
