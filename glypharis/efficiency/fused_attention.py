"""The SVTR recognizer exported with its attention as one fused ONNX Runtime node (GW-EFF-009).

The plain export (`glypharis.export.onnx.export_recognizer_onnx`) writes each global mixer's
``scaled_dot_product_attention`` as separate MatMul, Softmax and masking nodes over every token
pair, where PyTorch runs a fused kernel; that is why the exported v1 recognizer is slower than
PyTorch on the CPU (F9). This module swaps each `_GlobalMixer` of a copy of the model for
`FusedGlobalMixer` at export time, which computes the same thing and exports it as one
``com.microsoft.MultiHeadAttention`` node (ONNX Runtime's contrib op, CPU kernel included):

- query, key and value are the mixer's own ``qkv`` projection split in three, each
  ``(N, L, C)`` with the heads laid out as the original reshape lays them out;
- ``key_padding_mask`` is the line's key mask as int32 ``(N, L)``, 1 for a column inside the
  line, so padded columns are never attended to, as in the original;
- the scale is ONNX Runtime's default ``1 / sqrt(C / heads)``, PyTorch's default too; masked
  scores get ONNX Runtime's default filter value (-10000), whose softmax weight is exactly 0 in
  fp32 next to any real score.

Nothing is approximated: no FastGelu or other transformer-optimizer pass runs, and the session
is the usual `glypharis.efficiency.threads.ort_session`. In PyTorch the fused mixer runs the
original attention, so the swapped model gives the original's outputs exactly. The tests check
the fused ONNX graph against the reference within `FUSED_TOL` at several widths and batch sizes;
on the 5,000 validation lines `scripts/phase16_variants.py eval --set lines` applies the export
check's logit and text rules (`export_check.LOGIT_TOL`, 1e-3, as for any ONNX export) against
PyTorch v1. `glypharis/models/rec/svtr.py` is not touched: the swap acts on a copy.

The fused recognizer is a variant (a new version), not v1.
"""

from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import nn

from glypharis.efficiency.threads import ort_session
from glypharis.export.onnx import export_recognizer_onnx
from glypharis.models.rec.base import LineRecognizer
from glypharis.models.rec.svtr import Grid, _GlobalMixer

FUSED_OP: Final = "MultiHeadAttention"
FUSED_DOMAIN: Final = "com.microsoft"
FUSED_TOL: Final = 1e-4  # max abs difference of the logits, fused ONNX against the reference


class FusedAttentionError(RuntimeError):
    """The model has no attention to fuse, or ONNX Runtime cannot build or run the fused graph."""


class _MultiHeadAttention(torch.autograd.Function):
    """Self-attention with a key-padding mask: the original computation in PyTorch, one
    ``com.microsoft.MultiHeadAttention`` node in an ONNX export."""

    @staticmethod
    def forward(
        ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, heads: int
    ) -> torch.Tensor:
        n, length, c = q.shape

        def split(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(n, length, heads, c // heads).transpose(1, 2)

        keys = mask.to(torch.bool)[:, None, None, :]
        y = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=keys)
        return y.transpose(1, 2).reshape(n, length, c)

    @staticmethod
    def symbolic(
        g: Any, q: Any, k: Any, v: Any, mask: Any, heads: int
    ) -> Any:  # pragma: no cover - called by the exporter
        from torch.onnx import symbolic_helper

        no_bias = symbolic_helper._optional_input_placeholder_tensor(g)
        return g.op(f"{FUSED_DOMAIN}::{FUSED_OP}", q, k, v, no_bias, mask, num_heads_i=heads)


class FusedGlobalMixer(nn.Module):
    """`_GlobalMixer` with the attention as `_MultiHeadAttention`; shares the original's weights."""

    def __init__(self, mixer: _GlobalMixer) -> None:
        super().__init__()
        self.heads = mixer.heads
        self.qkv = mixer.qkv
        self.proj = mixer.proj

    def forward(self, x: torch.Tensor, grid: Grid, keys: torch.Tensor) -> torch.Tensor:
        c = x.shape[-1]
        q, k, v = self.qkv(x).split(c, dim=-1)
        y = _MultiHeadAttention.apply(q, k, v, keys.to(torch.int32), self.heads)
        return self.proj(y)


def fuse_attention(model: LineRecognizer) -> LineRecognizer:
    """A copy of ``model`` (in eval mode, on the CPU) with every `_GlobalMixer` replaced by a
    `FusedGlobalMixer`; ``model`` itself is unchanged. `FusedAttentionError` if it has none
    (a CRNN, or an SVTR without global blocks)."""
    fused = copy.deepcopy(model).cpu().eval()
    swapped = 0
    for module in fused.modules():
        mixer = getattr(module, "mixer", None)
        if isinstance(mixer, _GlobalMixer):
            module.mixer = FusedGlobalMixer(mixer)
            swapped += 1
    if not swapped:
        raise FusedAttentionError(
            f"{type(model).__name__} has no global attention blocks; there is nothing to fuse"
        )
    return fused


def count_global_mixers(model: nn.Module) -> int:
    return sum(isinstance(m, _GlobalMixer) for m in model.modules())


def graph_ops(path: str | Path) -> dict[str, int]:
    """Node counts of an ONNX graph by ``domain::op`` (the default domain as ``ai.onnx``)."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    counts = Counter(f"{n.domain or 'ai.onnx'}::{n.op_type}" for n in graph.node)
    return dict(sorted(counts.items()))


def export_fused_recognizer(
    model: LineRecognizer, height: int, out: str | Path, threads: int | None = None
) -> dict[str, Any]:
    """Export ``model`` with fused attention to ``out`` (inputs, outputs and dynamic axes as
    `export_recognizer_onnx`), check that every global mixer became one fused node and that no
    Softmax is left, and open it in an `ort_session` with ``threads``. Returns a record with the
    node counts; `FusedAttentionError` if any step fails, with ONNX Runtime's message."""
    out = Path(out)
    expected = count_global_mixers(model)
    fused = fuse_attention(model)
    try:
        export_recognizer_onnx(fused, height, out)
    except Exception as exc:  # the exporter's own errors carry no common base class
        raise FusedAttentionError(f"the fused export failed ({type(exc).__name__}: {exc})") from exc
    ops = graph_ops(out)
    got = ops.get(f"{FUSED_DOMAIN}::{FUSED_OP}", 0)
    if got != expected or ops.get("ai.onnx::Softmax", 0):
        raise FusedAttentionError(
            f"{out}: {got} {FUSED_OP} nodes for {expected} global mixers and "
            f"{ops.get('ai.onnx::Softmax', 0)} Softmax nodes left; the attention was not fused"
        )
    try:
        session = ort_session(out, threads)
        n = 2
        x = torch.zeros(n, 1, height, 64).numpy()
        w = torch.tensor([64, 32], dtype=torch.int64).numpy()
        session.run(["logits"], {"images": x, "widths": w})
    except Exception as exc:  # onnxruntime raises its own exception types without a common base
        raise FusedAttentionError(
            f"ONNX Runtime cannot run the fused recognizer {out} ({type(exc).__name__}: {exc})"
        ) from exc
    return {
        "op": f"{FUSED_DOMAIN}.{FUSED_OP}",
        "global_mixers": expected,
        "fused_nodes": got,
        "mask": "key_padding_mask int32 (N, L), 1 inside the line",
        "scale": "default, 1/sqrt(head size)",
        "mask_filter_value": "default (-10000)",
        "approximations": "none",
        "graph_ops": ops,
    }
