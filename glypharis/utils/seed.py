"""Global seeding and determinism switches.

`seed_everything` must be called once, before any model, data loader or random draw is created.
Data loader workers get their own derived seeds through `worker_init_fn` and `make_generator`.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

# Needed for deterministic cuBLAS on CUDA >= 10.2. Must be set before the first cuBLAS call.
_CUBLAS_WORKSPACE = ":4096:8"


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch (CPU and all CUDA devices).

    With ``deterministic=True`` cuDNN autotuning is disabled and PyTorch is asked for deterministic
    kernels. Ops that have no deterministic implementation (e.g. CTC loss backward on CUDA) only
    warn instead of failing, so CUDA runs are reproducible up to a documented tolerance
    (docs/EXPERIMENT_PROTOCOL.md), while CPU runs are bit-identical.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")

    # Only affects subprocesses started from here on (e.g. spawned data loader workers).
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def worker_init_fn(worker_id: int) -> None:
    """Seed `random` and NumPy inside a DataLoader worker.

    PyTorch already gives each worker a distinct torch seed (base_seed + worker_id); this derives
    the other generators from it, so augmentations differ per worker but repeat across runs.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """A torch.Generator for DataLoader shuffling, independent of the global RNG state."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
