"""Line recognizer for inference: crops in, text and a confidence out (Phase 10).

`TextRecognizer` wraps a trained CTC recognizer the way it was trained and validated: each crop is
turned grey, resized to the model's input height (width capped at ``max_width``), scaled to
[-1, 1] and padded on the right with the median of its last column, as `scripts/rec_profile.py`
times it. Crops are sorted by width and read in batches capped by `MAX_BATCH_COLUMNS`; each
line's own width goes to the model, and every line gets at least `PAD_MARGIN` padding columns
before the edge of the batch tensor. Without that margin the last frames, whose receptive field
reaches the tensor edge, change with the batch width; with it a line reads the same alone or in
any batch (tested).

Confidence of a line (registered in docs/PHASE_STATUS.md, Phase 10): softmax over the per-frame
logits divided by the temperature ``T``; the geometric mean of the top-class probability over the
frames whose top class is not blank, or, for a line read as empty, the mean blank probability.
Decoding is greedy CTC and does not depend on ``T``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from glypharis.config.loader import from_dict, load_config
from glypharis.config.schema import ModelConfig
from glypharis.data.rec_dataset import PAD_MULTIPLE, resized_width
from glypharis.data.text import Charset
from glypharis.models.rec.base import LineRecognizer
from glypharis.models.rec.crnn import build_recognizer
from glypharis.training.checkpoint import load_checkpoint

REC_RELEASE_FORMAT = "glypharis-rec-release-1"
MAX_BATCH_LINES = 32
MAX_BATCH_COLUMNS = 16384
PAD_MARGIN = 64  # padding columns every line keeps before the tensor edge (32 already suffice)


@dataclass(frozen=True)
class LineReading:
    text: str
    confidence: float  # at the recognizer's temperature
    frames: int
    at: dict[float, float] = field(default_factory=dict)  # confidence at other temperatures


def line_confidence(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """Confidence of one line from its ``(T, classes)`` logits (module docstring)."""
    if logits.shape[0] == 0:
        return 0.0
    p = torch.softmax(logits.float() / temperature, dim=-1)
    top, idx = p.max(dim=-1)
    emitted = idx != Charset.BLANK
    if bool(emitted.any()):
        return float(torch.exp(torch.log(top[emitted].clamp_min(1e-12)).mean()))
    return float(p[:, Charset.BLANK].mean())


def _padded(width: int) -> int:
    return math.ceil((width + PAD_MARGIN) / PAD_MULTIPLE) * PAD_MULTIPLE


def prepare_crop(img: Image.Image, height: int, max_width: int) -> torch.Tensor:
    """One crop as the model's ``(height, width)`` input in [-1, 1], unpadded."""
    g = img.convert("L")
    w = resized_width(g.width, g.height, height, max_width)
    g = g.resize((w, height), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(g, dtype=np.float32) / 127.5 - 1.0)


class TextRecognizer:
    """Crops in, `LineReading` out, on one device."""

    def __init__(
        self,
        model: LineRecognizer,
        charset: Charset,
        height: int,
        max_width: int,
        device: str | torch.device = "cpu",
        temperature: float = 1.0,
        arch: dict[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.arch = arch or {}  # model_name and model config, for a release file
        self.charset = charset
        self.height = height
        self.max_width = max_width
        self.device = torch.device(device)
        self.temperature = temperature

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        checkpoint: str | Path | None = None,
        device: str | torch.device = "cpu",
        temperature: float = 1.0,
    ) -> TextRecognizer:
        """A training run: its resolved config and ``best.pt`` (or ``checkpoint``)."""
        cfg = load_config(Path(run_dir) / "config.resolved.yaml")
        charset = Charset.load(
            cfg.data.charset or Path(cfg.paths.data_root) / "manifests/charset.json"
        )
        model = build_recognizer(charset.num_classes, cfg.model, cfg.model_name or "crnn")
        path = Path(checkpoint) if checkpoint else Path(cfg.checkpoint.dir) / "best.pt"
        model.load_state_dict(load_checkpoint(path, "cpu")["model"])
        arch = {"model_name": cfg.model_name or "crnn", "model": dataclasses.asdict(cfg.model)}
        return cls(model, charset, cfg.data.image_height, cfg.data.max_width, device, temperature,
                   arch)  # fmt: skip

    def release(self) -> dict[str, Any]:
        """Plain data for a release file (`REC_RELEASE_FORMAT`): weights, architecture, charset
        and input settings; ``torch.load(..., weights_only=True)`` reads it."""
        if not self.arch:
            raise ValueError("this recognizer was built without its architecture; no release")
        return {
            "format": REC_RELEASE_FORMAT,
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "arch": self.arch,
            "characters": self.charset.characters,
            "height": self.height,
            "max_width": self.max_width,
            "temperature": self.temperature,
        }

    @classmethod
    def from_release(
        cls, path: str | Path, device: str | torch.device = "cpu", temperature: float | None = None
    ) -> TextRecognizer:
        """A release file written from `release`; anything else is refused."""
        obj = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(obj, dict) or obj.get("format") != REC_RELEASE_FORMAT:
            raise ValueError(f"{path} is not a {REC_RELEASE_FORMAT} file")
        charset = Charset(list(obj["characters"]))
        cfg = from_dict(ModelConfig, obj["arch"]["model"], "arch.model")
        model = build_recognizer(charset.num_classes, cfg, obj["arch"]["model_name"])
        model.load_state_dict(obj["state_dict"])
        t = obj["temperature"] if temperature is None else temperature
        return cls(model, charset, int(obj["height"]), int(obj["max_width"]), device, float(t),
                   dict(obj["arch"]))  # fmt: skip

    def _batches(self, widths: Sequence[int]) -> list[list[int]]:
        order = sorted(range(len(widths)), key=lambda i: (widths[i], i))
        out: list[list[int]] = []
        cur: list[int] = []
        for i in order:
            cols = _padded(widths[i])
            if cur and (len(cur) >= MAX_BATCH_LINES or cols * (len(cur) + 1) > MAX_BATCH_COLUMNS):
                out.append(cur)
                cur = []
            cur.append(i)
        if cur:
            out.append(cur)
        return out

    @torch.inference_mode()
    def logits(self, crops: Sequence[Image.Image]) -> list[torch.Tensor]:
        """Per-line ``(T, classes)`` logits on the CPU, in input order."""
        xs = [prepare_crop(c, self.height, self.max_width) for c in crops]
        widths = [x.shape[1] for x in xs]
        out: list[torch.Tensor | None] = [None] * len(xs)
        for idx in self._batches(widths):
            width = max(_padded(widths[i]) for i in idx)
            batch = torch.empty(len(idx), 1, self.height, width)
            for j, i in enumerate(idx):
                x = xs[i]
                batch[j, 0, :, : x.shape[1]] = x
                batch[j, 0, :, x.shape[1] :] = x[:, -1].median()
            w = torch.tensor([widths[i] for i in idx])
            logits = self.model(batch.to(self.device), w.to(self.device))
            lengths = self.model.frames(w).clamp(max=logits.shape[0])
            for j, i in enumerate(idx):
                out[i] = logits[: int(lengths[j]), j].float().cpu()
        return [o for o in out if o is not None]

    def read(
        self, crops: Sequence[Image.Image], temperatures: Sequence[float] = ()
    ) -> list[LineReading]:
        """Greedy text and confidence per crop; ``temperatures`` also fills `LineReading.at`."""
        result = []
        for lg in self.logits(crops):
            ids = lg.argmax(dim=-1).tolist()
            result.append(
                LineReading(
                    self.charset.ctc_greedy_decode(ids),
                    line_confidence(lg, self.temperature),
                    int(lg.shape[0]),
                    {t: line_confidence(lg, t) for t in temperatures},
                )
            )
        return result
