from __future__ import annotations

import torch

from glypharis.utils import device

GPU = torch.device("cuda")


def test_the_gpu_memory_cap_defaults_to_085_and_can_be_overridden_or_switched_off(monkeypatch):
    calls: list[float] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", calls.append)
    monkeypatch.delenv("GLYPHARIS_GPU_MEMORY_FRACTION", raising=False)
    assert device.limit_gpu_memory(GPU) == 0.85 and calls == [0.85]
    monkeypatch.setenv("GLYPHARIS_GPU_MEMORY_FRACTION", "0.7")
    assert device.limit_gpu_memory(GPU) == 0.7 and calls[-1] == 0.7
    monkeypatch.setenv("GLYPHARIS_GPU_MEMORY_FRACTION", "1")
    assert device.limit_gpu_memory(GPU) == 1.0 and len(calls) == 2  # 1 means no cap


def test_without_cuda_there_is_nothing_to_cap(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device.limit_gpu_memory(GPU) is None


def test_a_cpu_run_never_touches_the_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f: 1 / 0)
    assert device.limit_gpu_memory(torch.device("cpu")) is None
