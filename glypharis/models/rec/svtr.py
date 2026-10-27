"""GLYPHARIS-Rec-M: an SVTR-style encoder with a CTC head (ARCHITECTURE §4; Phase 7, H6).

After SVTR (Du et al., 2022), adapted to lines of any width:

- A stem of two strided convolutions cuts height and width by 4, so a frame covers 4 input
  columns as in the CRNN; the data pipeline and the CTC lengths are shared.
- Three stages of mixing blocks over the 2-D token grid. Local mixing is a depthwise convolution
  (SVTR's convolutional local mixer); global mixing is multi-head self-attention over every token
  of the line. Between stages a convolution halves the height and changes the embedding width.
- The remaining height is averaged away, and a linear layer gives per-frame logits.

Lines of different widths share a batch. Attention never attends to a padded column, and padded
tokens are zeroed after every block, so a line's frames do not depend on the rest of its batch
(away from its right edge, where the stem sees a few padded columns, as the CRNN's CNN does).
The default sizes give about as many parameters as the default CRNN.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from glypharis.config.schema import ModelConfig, SvtrConfig
from glypharis.models.rec.base import LineRecognizer

Grid = tuple[int, int]  # token grid height, width


class _LocalMixer(nn.Module):
    """Depthwise convolution over the token grid."""

    def __init__(self, dim: int, kernel: list[int]) -> None:
        super().__init__()
        kh, kw = kernel
        self.conv = nn.Conv2d(dim, dim, (kh, kw), padding=(kh // 2, kw // 2), groups=dim)

    def forward(self, x: torch.Tensor, grid: Grid, keys: torch.Tensor) -> torch.Tensor:
        n, _, c = x.shape
        y = self.conv(x.transpose(1, 2).reshape(n, c, *grid))
        return y.flatten(2).transpose(1, 2)


class _GlobalMixer(nn.Module):
    """Multi-head self-attention over all tokens of a line; padded tokens are never attended to."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, grid: Grid, keys: torch.Tensor) -> torch.Tensor:
        n, length, c = x.shape
        qkv = self.qkv(x).reshape(n, length, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each N, heads, L, C/heads
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=keys[:, None, None, :])
        return self.proj(y.transpose(1, 2).reshape(n, length, c))


class _Block(nn.Module):
    """Pre-norm residual block: mixer, then MLP. Padded tokens leave the block as zeros."""

    def __init__(self, dim: int, mixer: nn.Module, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = mixer
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))  # fmt: skip

    def forward(
        self, x: torch.Tensor, grid: Grid, keys: torch.Tensor, keep: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x), grid, keys)
        x = x + self.mlp(self.norm2(x))
        return x * keep


class _Merge(nn.Module):
    """Halve the grid height and change the embedding width (SVTR's merging)."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=(2, 1), padding=1)
        self.norm = nn.LayerNorm(c_out)

    def forward(self, x: torch.Tensor, grid: Grid) -> tuple[torch.Tensor, Grid]:
        n, _, c = x.shape
        y = self.conv(x.transpose(1, 2).reshape(n, c, *grid))
        return self.norm(y.flatten(2).transpose(1, 2)), (y.shape[2], y.shape[3])


class SVTR(LineRecognizer):
    def __init__(self, num_classes: int, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ModelConfig(svtr=SvtrConfig())
        s = cfg.svtr or SvtrConfig()
        self.stride = 4
        self.num_classes = num_classes
        d0 = s.dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, d0 // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d0 // 2),
            nn.GELU(),
            nn.Conv2d(d0 // 2, d0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d0),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        for i, dim in enumerate(s.dims):
            mixers: list[nn.Module] = [_LocalMixer(dim, s.local_kernel)
                                       for _ in range(s.local_blocks[i])]  # fmt: skip
            mixers += [_GlobalMixer(dim, s.heads[i]) for _ in range(s.global_blocks[i])]
            self.stages.append(
                nn.ModuleList(_Block(dim, m, s.mlp_ratio, cfg.dropout) for m in mixers)
            )
        self.merges = nn.ModuleList(_Merge(a, b) for a, b in zip(s.dims, s.dims[1:], strict=False))
        self.norm = nn.LayerNorm(s.dims[-1])
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(s.dims[-1], num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, widths: torch.Tensor | None = None) -> torch.Tensor:
        """Per-frame logits ``(T, N, classes)`` with ``T = W // 4``."""
        y = self.stem(x)  # N, C, H/4, W/4
        n, _, h, w = y.shape
        tokens = y.flatten(2).transpose(1, 2)  # N, H*W, C, row by row
        cols = torch.arange(w, device=x.device)
        if widths is None:
            valid = torch.ones(n, w, dtype=torch.bool, device=x.device)
        else:
            frames = self.frames(widths.to(x.device)).clamp(min=1, max=w)
            valid = cols[None, :] < frames[:, None]  # N, W: columns inside each line
        grid: Grid = (h, w)
        for i, blocks in enumerate(self.stages):
            if i:
                tokens, grid = self.merges[i - 1](tokens, grid)
            keys = valid.repeat(1, grid[0])  # N, H*W, same row-by-row order as the tokens
            keep = keys[..., None].to(tokens.dtype)
            tokens = tokens * keep
            for block in cast(nn.ModuleList, blocks):
                tokens = block(tokens, grid, keys, keep)
        f = self.norm(tokens).reshape(n, grid[0], grid[1], -1).mean(dim=1)  # N, W, C
        return self.head(self.dropout(f)).transpose(0, 1)  # T, N, classes
