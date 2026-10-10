from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from glypharis.utils.determinism import run_toy_training
from glypharis.utils.seed import make_generator, seed_everything, worker_init_fn

# Must match EXPERIMENT_PROTOCOL §6 and scripts/env_report.py.
CUDA_REL_TOLERANCE = 1e-4


def _draws() -> tuple[float, float, list[float]]:
    return random.random(), float(np.random.rand()), torch.rand(3).tolist()


def test_seed_everything_repeats_all_generators():
    seed_everything(42)
    first = _draws()
    seed_everything(42)
    assert _draws() == first
    seed_everything(43)
    assert _draws() != first


@pytest.mark.parametrize("bad", [-1, 1.5, True, "1"])
def test_seed_everything_rejects_bad_seed(bad):
    with pytest.raises(ValueError):
        seed_everything(bad)


def test_make_generator_is_independent_of_global_state():
    a = torch.randperm(10, generator=make_generator(7)).tolist()
    torch.manual_seed(0)
    torch.rand(5)
    b = torch.randperm(10, generator=make_generator(7)).tolist()
    assert a == b


def test_worker_init_fn_derives_from_torch_seed():
    torch.manual_seed(123)
    worker_init_fn(0)
    first = (random.random(), float(np.random.rand()))
    torch.manual_seed(123)
    worker_init_fn(0)
    assert (random.random(), float(np.random.rand())) == first


# ------------------------------------------------------------- Phase 1 exit criterion (CPU)


def test_cpu_toy_training_is_bit_identical_for_50_steps():
    a = run_toy_training(seed=1337, steps=50)
    b = run_toy_training(seed=1337, steps=50)
    assert len(a) == 50
    assert a == b  # exact equality, not approx


def test_different_seed_gives_different_losses():
    assert run_toy_training(seed=1, steps=5) != run_toy_training(seed=2, steps=5)


def test_toy_training_learns_something():
    losses = run_toy_training(seed=1337, steps=50)
    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]


# ---------------------------------------------------------- Phase 1 exit criterion (CUDA)


@pytest.mark.gpu
def test_cuda_toy_training_within_tolerance():
    dev = torch.device("cuda")
    a = run_toy_training(seed=1337, steps=50, device=dev)
    b = run_toy_training(seed=1337, steps=50, device=dev)
    rel = max(abs(x - y) / max(abs(x), 1e-12) for x, y in zip(a, b, strict=True))
    assert rel <= CUDA_REL_TOLERANCE


@pytest.mark.gpu
def test_cuda_bf16_toy_training_is_finite():
    losses = run_toy_training(seed=1337, steps=20, device=torch.device("cuda"), precision="bf16")
    assert all(np.isfinite(losses))
