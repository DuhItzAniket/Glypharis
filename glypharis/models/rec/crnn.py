"""CRNN line recognizer (ARCHITECTURE §4): CNN -> BiLSTM -> linear -> CTC.

Input: grayscale line images ``(N, 1, H, W)``, normalised to [-1, 1], right-padded to the widest
line in the batch. The CNN halves height and width twice, then the height twice more, so a frame
covers 4 columns (``STRIDE``); whatever height is left is averaged away, so the same network works
for H = 32 and H = 48 (Phase 7 ablation). About 7.8M parameters with the default config.
Output: per-frame logits ``(T, N, C)`` with ``T = W // 4`` and ``C = |charset| + 1`` (index 0 is the
CTC blank).
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from glypharis.config.schema import ModelConfig

STRIDE = 4  # input columns per output frame
# A stacked nn.LSTM names layer k's tensors rnn.weight_ih_l{k}[_reverse]; the per-layer modules
# name the same tensors rnn.{k}.weight_ih_l0[_reverse].
_STACKED_KEY = re.compile(r"^rnn\.((?:weight|bias)_(?:ih|hh))_l(\d+)(_reverse)?$")


def _stacked_lstm_keys(module: nn.Module, state_dict: dict[str, Any], prefix: str,
                       *_: Any) -> None:  # fmt: skip
    """Load-time hook: rename a stacked-LSTM checkpoint's keys (GW-REC-001/002) to this layout."""
    for key in [k for k in state_dict if k.startswith(prefix)]:
        m = _STACKED_KEY.match(key[len(prefix) :])
        if m:
            name, layer, rev = m.group(1), m.group(2), m.group(3) or ""
            state_dict[f"{prefix}rnn.{layer}.{name}_l0{rev}"] = state_dict.pop(key)


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
        # One single-layer BiLSTM per layer, with ordinary dropout between them, instead of a
        # stacked nn.LSTM with its built-in dropout. cuDNN draws that dropout from its own random
        # state, which a checkpoint cannot save (so a resumed CUDA run drew different masks), and
        # on Windows freeing that state at exit aborted every training process (0xC0000409).
        # Same architecture, parameters and parameter order; old checkpoints load unchanged.
        h = cfg.lstm_hidden
        self.rnn = nn.ModuleList(
            nn.LSTM(c6 if i == 0 else 2 * h, h, bidirectional=True) for i in range(cfg.lstm_layers)
        )
        self.rnn_dropout = nn.Dropout(cfg.dropout)  # between BiLSTM layers
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(2 * h, num_classes)
        self.num_classes = num_classes
        self._init_weights()
        self.register_load_state_dict_pre_hook(_stacked_lstm_keys)

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

    def forward(self, x: torch.Tensor, widths: torch.Tensor | None = None) -> torch.Tensor:
        """Per-frame logits. ``widths`` (input columns of each line before padding) makes the
        BiLSTM read each line only up to its own end: without it, the backward direction starts in
        the padding, so a line's output would depend on the other lines in its batch."""
        # Average away the remaining height (2 rows at H = 32, 3 at H = 48). A plain mean, unlike
        # adaptive pooling, has a deterministic CUDA backward.
        f = self.cnn(x).mean(dim=2).permute(2, 0, 1)  # T, N, C
        t = f.shape[0]
        # The recurrent part runs in fp32: cuDNN's bf16 LSTM path is slow and less exact.
        with torch.autocast(device_type=x.device.type, enabled=False):
            f = f.float()
            if widths is None:
                for i, rnn in enumerate(self.rnn):
                    f, _ = rnn(self.rnn_dropout(f) if i else f)
            else:
                lengths = self.frames(widths.cpu()).clamp(min=1, max=t)
                seq = nn.utils.rnn.pack_padded_sequence(f, lengths, enforce_sorted=False)
                for i, rnn in enumerate(self.rnn):
                    if i:  # dropout on the packed values only, as the stacked LSTM did
                        seq = PackedSequence(self.rnn_dropout(seq.data), seq.batch_sizes,
                                             seq.sorted_indices, seq.unsorted_indices)  # fmt: skip
                    seq, _ = rnn(seq)
                f, _ = nn.utils.rnn.pad_packed_sequence(seq, total_length=t)
            return self.head(self.dropout(f))  # T, N, classes


def build_recognizer(num_classes: int, cfg: ModelConfig) -> CRNN:
    return CRNN(num_classes, cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
