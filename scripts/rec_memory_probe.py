"""Does a recognizer config fit in GPU memory? One training step on its worst-case batches.

    python scripts/rec_memory_probe.py --config configs/rec/gw-rec-022.yaml

The width-bucket sampler packs at most `train.batch_size` lines and at most
`data.max_batch_columns` padded columns into a micro-batch, so its most memory-hungry batches are
(a) as many lines of the widest width (`data.max_width`) as the column cap allows, and (b) a full
batch at the widest width the cap allows for it. The probe runs a forward pass, the CTC loss and a
backward pass on each, with random images and labels, at the run's precision and under the same GPU
memory cap as training (`limit_gpu_memory`). It reports the peak memory and exits 1 if either batch
runs out of memory: a run that would fail mid-epoch (GW-REC-019) fails here in seconds instead.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any

import torch
from torch import nn

from glypharis.config.cli import add_config_args, config_from_args
from glypharis.config.schema import RunConfig
from glypharis.data.rec_dataset import PAD_MULTIPLE
from glypharis.data.text import Charset
from glypharis.models.rec.crnn import build_recognizer
from glypharis.training.rec_trainer import ctc_loss
from glypharis.utils.device import limit_gpu_memory, resolve_amp, select_device


def worst_batches(cfg: RunConfig) -> list[tuple[int, int]]:
    """(lines, padded width) of the micro-batches that need the most memory."""
    d, n = cfg.data, cfg.train.batch_size
    cap = d.max_batch_columns or n * d.max_width

    def pad(w: int) -> int:
        return math.ceil(w / PAD_MULTIPLE) * PAD_MULTIPLE

    widest = (max(1, min(n, cap // d.max_width)), pad(d.max_width))
    fullest = (n, pad(min(d.max_width, cap // n)))
    return list(dict.fromkeys([widest, fullest]))


def probe(cfg: RunConfig, device: torch.device, num_classes: int) -> list[dict[str, Any]]:
    torch.manual_seed(0)
    model = build_recognizer(num_classes, cfg.model, cfg.model_name or "crnn").to(device)
    model.train()
    amp = resolve_amp(cfg.train.mixed_precision, device)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    out = []
    for lines, width in worst_batches(cfg):
        h = cfg.data.image_height
        widths = torch.full((lines,), width)
        frames = int(model.frames(widths)[0])
        label_len = max(1, frames // 3)
        batch = {
            "images": torch.rand(lines, 1, h, width) * 2 - 1,
            "widths": widths,
            "targets": torch.randint(1, num_classes, (lines * label_len,)),
            "target_lengths": torch.full((lines,), label_len, dtype=torch.long),
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        row: dict[str, Any] = {"lines": lines, "width": width, "height": h}
        try:
            loss, _, _ = ctc_loss(model, batch, amp, device, ctc)
            loss.backward()
            row["fits"] = True
        except torch.OutOfMemoryError:
            row["fits"] = False
        model.zero_grad(set_to_none=True)
        if device.type == "cuda":
            row["peak_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    cfg = config_from_args(parser.parse_args(argv))
    device = select_device(cfg.device)
    fraction = limit_gpu_memory(device)
    charset = Charset.load(cfg.data.charset or f"{cfg.paths.data_root}/manifests/charset.json")
    rows = probe(cfg, device, charset.num_classes)
    total = (torch.cuda.get_device_properties(device).total_memory / 2**20
             if device.type == "cuda" else None)  # fmt: skip
    for r in rows:
        peak = f", peak {r['peak_mib']:,.0f} MiB" if "peak_mib" in r else ""
        print(f"{cfg.experiment_id}: {r['lines']} lines x {r['width']} columns at height "
              f"{r['height']}: {'fits' if r['fits'] else 'OUT OF MEMORY'}{peak}")  # fmt: skip
    if total and fraction:
        print(f"GPU memory cap: {fraction:.0%} of {total:,.0f} MiB = {fraction * total:,.0f} MiB")
    return 0 if all(r["fits"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
