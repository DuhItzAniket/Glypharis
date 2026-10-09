"""A tiny CNN + BiLSTM + CTC training loop used to verify reproducibility.

It mirrors the shape of GLYPHARIS-Rec (conv features → BiLSTM → CTC) at toy scale, so the
determinism check exercises the same kinds of kernels the real recognizer will use. It is used by
the Phase 1 determinism tests and by ``scripts/env_report.py``; it is not a model.
"""

from __future__ import annotations

import torch
from torch import nn

from glypharis.utils.device import AmpPolicy, fp32_region, resolve_amp
from glypharis.utils.seed import seed_everything

NUM_CLASSES = 11  # 10 symbols + CTC blank (index 0)
IMG_H, IMG_W = 16, 64
TARGET_LEN = 6


class ToyCRNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H 16→8, W 64→32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((8, 2)),  # H 8→1, W 32→16
        )
        self.rnn = nn.LSTM(32, hidden, bidirectional=True, batch_first=False)
        self.head = nn.Linear(2 * hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features(x)  # (B, C, 1, T)
        f = f.squeeze(2).permute(2, 0, 1)  # (T, B, C)
        out, _ = self.rnn(f)
        logits: torch.Tensor = self.head(out)  # (T, B, classes)
        return logits


def _synthetic_batch(
    generator: torch.Generator, batch: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.rand(batch, 1, IMG_H, IMG_W, generator=generator)
    targets = torch.randint(1, NUM_CLASSES, (batch, TARGET_LEN), generator=generator)
    target_lengths = torch.full((batch,), TARGET_LEN, dtype=torch.long)
    return images, targets, target_lengths


def run_toy_training(
    seed: int,
    steps: int = 50,
    device: torch.device | None = None,
    precision: str = "none",
    batch: int = 8,
) -> list[float]:
    """Train ToyCRNN for ``steps`` steps on seeded synthetic data and return the loss per step.

    Two calls with the same arguments must return identical lists on CPU. On CUDA the protocol
    allows a small documented tolerance (non-deterministic CTC backward kernels).
    """
    device = device or torch.device("cpu")
    seed_everything(seed, deterministic=True)
    amp: AmpPolicy = resolve_amp(precision, device)  # type: ignore[arg-type]

    model = ToyCRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    data_gen = torch.Generator().manual_seed(seed)

    losses: list[float] = []
    for _ in range(steps):
        images, targets, target_lengths = _synthetic_batch(data_gen, batch)
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast():
            logits = model(images)
        with fp32_region(device.type):
            log_probs = logits.float().log_softmax(-1)
            input_lengths = torch.full((batch,), log_probs.shape[0], dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses
