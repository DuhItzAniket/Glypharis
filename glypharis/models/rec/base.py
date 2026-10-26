"""What the trainer needs from a CTC line recognizer, whatever its encoder."""

from __future__ import annotations

import torch
from torch import nn


class LineRecognizer(nn.Module):
    """``forward(x, widths)`` maps right-padded lines ``(N, 1, H, W)`` to per-frame logits
    ``(T, N, classes)``; ``stride`` input columns make one frame. ``widths`` (each line's columns
    before padding) keeps a line's output independent of the rest of its batch."""

    stride: int
    num_classes: int

    def frames(self, widths: torch.Tensor) -> torch.Tensor:
        """Output frames that a line's own width gives (the encoder floors at each downsampling)."""
        return torch.div(widths, self.stride, rounding_mode="floor")
