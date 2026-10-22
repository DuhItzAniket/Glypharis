"""CRNN line recognizer (ARCHITECTURE §4): CNN -> BiLSTM -> linear -> CTC.

Input: grayscale line images ``(N, 1, H, W)``, normalised to [-1, 1], right-padded to the widest
line in the batch. The CNN halves height and width twice, then the height twice more, so a frame
covers 4 columns (``STRIDE``); whatever height is left is averaged away, so the same network works
for H = 32 and H = 48 (Phase 7 ablation). About 7.8M parameters with the default config.
Output: per-frame logits ``(T, N, C)`` with ``T = W // 4`` and ``C = |charset| + 1`` (index 0 is the
CTC blank).
"""

from __future__ import annotations

import torch
from torch import nn

from glypharis.config.schema import ModelConfig

STRIDE = 4  # input columns per output frame


def _conv(c_in: int, c_out: int) -> list[nn.Module]:
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)]


class CRNN(nn.Module):
    def __init__(self, num_classes: int, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        c1, c2, c3, c4, c5, c6 = cfg.cnn_channels
        self.cnn = nn.Sequential(
            *_conv(1, c1),
            nn.MaxPool2d(2, 2),  # H/2, W/2
            *_conv(c1, c2),
            nn.MaxPool2d(2, 2),  # H/4, W/4
            *_conv(c2, c3),
            *_conv(c3, c4),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, W/4
            *_conv(c4, c5),
            *_conv(c5, c6),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/16, W/4
        )
        self.rnn = nn.LSTM(
            c6,
            cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            bidirectional=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(2 * cfg.lstm_hidden, num_classes)
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def frames(widths: torch.Tensor) -> torch.Tensor:
        """Output frames for input widths (each conv keeps width; two 2x pools floor it)."""
        return torch.div(torch.div(widths, 2, rounding_mode="floor"), 2, rounding_mode="floor")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average away the remaining height (2 rows at H = 32, 3 at H = 48). A plain mean, unlike
        # adaptive pooling, has a deterministic CUDA backward.
        f = self.cnn(x).mean(dim=2)  # N, C, T
        f = f.permute(2, 0, 1)  # T, N, C
        f, _ = self.rnn(f)
        return self.head(self.dropout(f))  # T, N, classes


def build_recognizer(num_classes: int, cfg: ModelConfig) -> CRNN:
    return CRNN(num_classes, cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
