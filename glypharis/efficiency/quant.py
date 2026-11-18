"""Dynamic INT8 of an exported recognizer with H7′'s registered recipe, nothing else (GW-EFF-008).

    record = quantize_recognizer("bundle/recognizer.onnx", "work/recognizer.int8.onnx")

The recipe (`RECIPE`, registered in research/hypotheses.md under H7′ before any Phase 16
measurement): ONNX Runtime's `quant_pre_process` with its defaults, then `quantize_dynamic` on
MatMul nodes only, per-tensor, full range, signed 8-bit weights, and only where the second input
is a constant (``MatMulConstBOnly``). The attention products multiply two activations, so they
have no constant input and stay fp32; convolutions, LayerNorm and everything else stay fp32 too.
No other configuration is offered: a second one tried after seeing data would be a choice made on
the result.

The record holds the recipe, the library versions, the SHA-256 and size of the input, of the
pre-processed graph and of the output, and node counts before and after, so the scorer can tie a
latency or accuracy number to the exact file and see how many products were quantised.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Final

from glypharis.data.download import sha256_file

RECIPE: Final[dict[str, Any]] = {
    "pre_process": "onnxruntime.quantization.quant_pre_process, default arguments",
    "quantize": "onnxruntime.quantization.quantize_dynamic",
    "op_types_to_quantize": ["MatMul"],
    "per_channel": False,
    "reduce_range": False,
    "weight_type": "QInt8",
    "extra_options": {"MatMulConstBOnly": True},
}
# Node types counted in the record: the products, and what dynamic quantisation puts around them.
COUNTED_OPS: Final = ("MatMul", "MatMulInteger", "DynamicQuantizeLinear", "Gemm", "Conv")


class QuantizationError(RuntimeError):
    """The recognizer could not be quantised with the registered recipe."""


def node_counts(path: str | Path) -> dict[str, int]:
    """How many nodes of each `COUNTED_OPS` type the graph at ``path`` has."""
    import onnx

    counts = Counter(n.op_type for n in onnx.load(str(path), load_external_data=False).graph.node)
    return {op: int(counts.get(op, 0)) for op in COUNTED_OPS}


def _file(path: Path) -> dict[str, Any]:
    return {"path": path.as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def quantize_recognizer(
    fp32_path: str | Path, out_path: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    """Write the dynamic INT8 version of the fp32 recognizer at ``fp32_path`` to ``out_path`` with
    `RECIPE`, and return the record (module docstring). An existing ``out_path`` is refused unless
    ``overwrite``; the pre-processed intermediate graph is hashed and deleted."""
    import onnx
    import onnxruntime
    from onnxruntime.quantization import QuantType, quant_pre_process, quantize_dynamic

    src, out = Path(fp32_path), Path(out_path)
    if not src.is_file():
        raise QuantizationError(f"{src}: no such fp32 model")
    if out.exists() and not overwrite:
        raise QuantizationError(f"{out} exists; refusing to overwrite it")
    out.parent.mkdir(parents=True, exist_ok=True)
    before = node_counts(src)
    with tempfile.TemporaryDirectory(prefix="glypharis-int8-") as tmp:
        pre = Path(tmp) / "preprocessed.onnx"
        try:
            quant_pre_process(str(src), str(pre))
            pre_sha = sha256_file(pre)
            quantize_dynamic(
                str(pre),
                str(out),
                op_types_to_quantize=list(RECIPE["op_types_to_quantize"]),
                per_channel=RECIPE["per_channel"],
                reduce_range=RECIPE["reduce_range"],
                weight_type=QuantType.QInt8,
                extra_options=dict(RECIPE["extra_options"]),
            )
        except Exception as exc:  # onnxruntime raises plain Exceptions from shape inference
            raise QuantizationError(
                f"{src}: quantisation failed ({type(exc).__name__}: {exc})"
            ) from exc
    after = node_counts(out)
    if after["MatMulInteger"] == 0:
        raise QuantizationError(f"{src}: no MatMul with a constant weight was quantised")
    return {
        "recipe": RECIPE,
        "onnxruntime": onnxruntime.__version__,
        "onnx": onnx.__version__,
        "input": _file(src),
        "preprocessed_sha256": pre_sha,
        "output": _file(out),
        "nodes_fp32": before,
        "nodes_int8": after,
    }
