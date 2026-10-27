"""Running the detector network on one prepared page.

Validation and inference run in true fp32: on CUDA, cuDNN convolutions default to TF32, which would
make the selected validation F1 a number the CPU (and Phase 17's export) cannot reproduce.
`strict_fp32` switches convolutions and matmuls to IEEE fp32 for the block and restores the previous
settings afterwards. On the CPU it changes nothing.

`ProbMapBackend` is the seam `TextLineDetector` calls: a padded uint8 page in, a probability map
out. `TorchProbMap` is the PyTorch implementation; an exported runtime can provide another one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Protocol

import numpy as np
import torch

from glypharis.models.det.dbnet import DBNet
from glypharis.models.det.preprocess import PAD_MULTIPLE


@contextlib.contextmanager
def strict_fp32() -> Iterator[None]:
    """IEEE fp32 (no TF32) for cuDNN convolutions and CUDA matmuls inside the block."""
    # cudnn.conv is added to the module object at run time, so the type checker cannot see it.
    conv = torch.backends.cudnn.conv  # type: ignore[attr-defined]
    matmul = torch.backends.cuda.matmul
    saved = conv.fp32_precision, matmul.fp32_precision
    conv.fp32_precision = "ieee"
    matmul.fp32_precision = "ieee"
    try:
        yield
    finally:
        conv.fp32_precision, matmul.fp32_precision = saved


class ProbMapBackend(Protocol):
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """``(pad_h, pad_w, 3)`` uint8 RGB from `prepare_page` -> ``(pad_h, pad_w)`` float32."""
        ...


class TorchProbMap:
    """`ProbMapBackend` over a `DBNet`: batch 1, eval mode, fp32 with autocast off, strict fp32.

    The model is moved to ``device`` and put in eval mode once. A model switched back to training
    mode (a trainer between validations) is refused rather than run with batch statistics.
    """

    def __init__(self, model: DBNet, device: torch.device) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected an (H, W, 3) uint8 image, got {image.shape} {image.dtype}")
        if image.shape[0] % PAD_MULTIPLE or image.shape[1] % PAD_MULTIPLE:
            raise ValueError(
                f"image size {image.shape[1]}x{image.shape[0]} is not a multiple of {PAD_MULTIPLE};"
                " pass it through prepare_page"
            )
        if self.model.training:
            raise RuntimeError("TorchProbMap needs the model in eval mode")
        # HWC -> NCHW as a view: the batch is channels-last in memory, which suits the model.
        x = torch.from_numpy(np.ascontiguousarray(image)).to(self.device).permute(2, 0, 1)[None]
        with (
            torch.inference_mode(),
            strict_fp32(),
            torch.autocast(device_type=self.device.type, enabled=False),
        ):
            prob = self.model(x)
        return np.ascontiguousarray(prob[0, 0].float().cpu().numpy(), dtype=np.float32)
