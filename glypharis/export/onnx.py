"""ONNX export of the detector and the recognizer, and ONNX Runtime backends for them (Phase 17).

Exports use PyTorch's TorchScript-based exporter (``dynamo=False``, opset 17) with dynamic axes:
the detector takes ``image`` ``(N, 3, H, W)`` uint8 (normalisation is inside the model) and gives
``prob`` ``(N, 1, H, W)``; the recognizer takes ``images`` ``(N, 1, height, W)`` float32 in
[-1, 1] and ``widths`` ``(N,)`` int64, and gives ``logits`` ``(T, N, classes)``. Nothing is
traced through data-dependent branches: both models were written for this.

`OnnxProbMap` is a `ProbMapBackend` for `TextLineDetector`; `OnnxLineModel` stands in for the
recognizer module inside `TextRecognizer` (same call, same ``frames``). ONNX Runtime runs on the
CPU with its default threads unless ``threads`` is given, and without its memory arena.

Measured (Phase 17): the exported detector runs about twice as fast as PyTorch, but the exported
recognizer about four and a half times slower, because the SVTR's global attention is exported as
plain matrix products over every token pair, where PyTorch uses a fused kernel. Until that is
fixed (Phase 16), PyTorch is the default backend.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from glypharis.models.det.dbnet import DBNet
from glypharis.models.det.preprocess import PAD_MULTIPLE
from glypharis.models.rec.base import LineRecognizer

OPSET = 17


def _session(path: str | Path, threads: int | None) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Every page and batch has its own shape; with the memory arena on, ONNX Runtime kept the
    # buffers of each one and grew past 10 GB on the 110 profile pages (GW-EFF-004).
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def export_detector_onnx(
    model: DBNet, out: str | Path, example_hw: tuple[int, int] = (640, 640)
) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    x = torch.zeros(1, 3, *example_hw, dtype=torch.uint8)
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (x,),
            str(out),
            input_names=["image"],
            output_names=["prob"],
            dynamic_axes={"image": {0: "n", 2: "h", 3: "w"}, "prob": {0: "n", 2: "h", 3: "w"}},
            opset_version=OPSET,
            dynamo=False,
        )
    return out


def export_recognizer_onnx(model: LineRecognizer, height: int, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    x = torch.zeros(2, 1, height, 256)
    w = torch.tensor([256, 128])
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (x, w),
            str(out),
            input_names=["images", "widths"],
            output_names=["logits"],
            dynamic_axes={
                "images": {0: "n", 3: "w"},
                "widths": {0: "n"},
                "logits": {0: "t", 1: "n"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
    return out


class OnnxProbMap:
    """`ProbMapBackend` over an exported detector."""

    def __init__(self, path: str | Path, threads: int | None = None) -> None:
        self.session = _session(path, threads)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"expected an (H, W, 3) uint8 image, got {image.shape} {image.dtype}")
        if image.shape[0] % PAD_MULTIPLE or image.shape[1] % PAD_MULTIPLE:
            raise ValueError(
                f"image size {image.shape[1]}x{image.shape[0]} is not a multiple of "
                f"{PAD_MULTIPLE}; pass it through prepare_page"
            )
        x = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
        prob = self.session.run(["prob"], {"image": x})[0]
        return np.ascontiguousarray(prob[0, 0], dtype=np.float32)


class OnnxLineModel(LineRecognizer):
    """An exported recognizer behind the module interface `TextRecognizer` uses."""

    def __init__(
        self, path: str | Path, stride: int, num_classes: int, threads: int | None = None
    ) -> None:
        super().__init__()
        self.session = _session(path, threads)
        self.stride = stride
        self.num_classes = num_classes

    def frames(self, widths: torch.Tensor) -> torch.Tensor:
        return torch.div(widths, self.stride, rounding_mode="floor")

    def forward(self, x: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        out = self.session.run(
            ["logits"],
            {
                "images": x.detach().cpu().numpy().astype(np.float32),
                "widths": widths.detach().cpu().numpy().astype(np.int64),
            },
        )[0]
        return torch.from_numpy(out)
