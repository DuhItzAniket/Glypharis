"""Checkpoints: atomic writes and the random-number state needed for exact resume."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    """Every global random-number state a training step can touch."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore the states saved by `rng_state`.

    Generator states must be CPU byte tensors, but a checkpoint loaded with ``map_location`` set to
    a GPU moves every tensor there, so they are brought back to the CPU first.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


REPLACE_RETRIES = 8  # waits of 0.5, 1, 2, 4, 5, 5, 5 s: about 22 s in all


def save_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    """Write via a temporary file and rename, so a crash never leaves half a checkpoint.

    On Windows the rename fails while another process holds the target open, as a virus scanner
    does for a moment after a large file is written (Phase 12: one run lost to this). The rename is
    retried for about 20 seconds before the error is raised.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    for attempt in range(REPLACE_RETRIES):
        try:
            tmp.replace(path)
            return path
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(min(0.5 * 2**attempt, 5.0))
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    # The payload holds numpy RNG state and plain Python objects as well as tensors.
    return dict(torch.load(path, map_location=device, weights_only=False))


def prune_epoch_checkpoints(directory: Path, keep_last: int) -> None:
    """Keep only the newest ``keep_last`` epoch_XXX.pt files (best.pt and last.pt are separate)."""
    files = sorted(directory.glob("epoch_*.pt"))
    for old in files[:-keep_last]:
        old.unlink(missing_ok=True)
