from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# Scripts and the top-level entry points (train.py) are not part of the package; make them
# importable for tests.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _restore_torch_numerics():
    """`glypharis.pipeline.frozen.strict_inference` sets process-wide flags; put them back after
    each test so one test's deterministic mode does not leak into the next."""
    import os

    saved = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
        os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )
    yield
    (bench, det, algos, warn_only, mm_tf32, cudnn_tf32, prec, cublas) = saved
    torch.backends.cudnn.benchmark = bench
    torch.backends.cudnn.deterministic = det
    torch.use_deterministic_algorithms(algos, warn_only=warn_only)
    torch.backends.cuda.matmul.allow_tf32 = mm_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.set_float32_matmul_precision(prec)
    if cublas is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def write_yaml(tmp_path: Path):
    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    return _write
