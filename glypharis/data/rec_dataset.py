"""Line-recognizer data: line packs -> normalised tensors in width-bucketed batches.

Every random choice is a pure function of (seed, epoch, sample index):

- the batch order comes from `WidthBucketBatchSampler`, which rebuilds an epoch's batches from
  ``(seed, epoch)`` and can start part-way through an epoch (exact resume);
- augmentation uses ``numpy.random.default_rng([seed, epoch, index])``; the sampler passes the epoch
  with every index, so results do not depend on which DataLoader worker loads a sample or on how
  many workers there are.

Images are resized to the model height with the aspect ratio kept, squeezed only when wider than
``max_width``, normalised to [-1, 1] and right-padded. A line whose image is too narrow for CTC
(fewer frames than the label needs) is padded on the right with background, which adds blank frames
instead of distorting the text.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from glypharis.data.augment import Augmenter
from glypharis.data.linepack import LinePackReader
from glypharis.data.text import Charset

STRIDE = 4  # default input columns per frame; the trainer passes its model's stride to collate
PAD_MULTIPLE = 32  # padded batch widths are multiples of this


@dataclass(frozen=True)
class LineRef:
    pack: int
    index: int
    key: str
    doc_id: str
    label: str
    width: int  # stored crop size
    height: int


def ctc_frames_needed(target: Sequence[int]) -> int:
    """Minimum frames CTC needs for a label: one per symbol plus a blank between repeats."""
    repeats = sum(1 for a, b in itertools.pairwise(target) if a == b)
    return len(target) + repeats


def resized_width(width: int, height: int, target_height: int, max_width: int) -> int:
    return max(1, min(max_width, round(width * target_height / max(height, 1))))


class RecLineDataset(Dataset[dict[str, Any]]):
    """Lines from one or more packs. Items are addressed by ``(epoch, index)``."""

    def __init__(
        self,
        packs: Sequence[str | Path],
        charset: Charset,
        height: int,
        max_width: int,
        augmenter: Augmenter | None = None,
        seed: int = 0,
        limits: Mapping[str, int] | None = None,
        subset: int | None = None,
    ) -> None:
        self.readers = [LinePackReader(p) for p in packs]
        self.charset, self.height, self.max_width = charset, height, max_width
        self.augmenter = augmenter if augmenter and augmenter.steps else None
        self.seed = seed
        refs: list[LineRef] = []
        for p, (name, reader) in enumerate(zip(packs, self.readers, strict=True)):
            idx = np.arange(len(reader))
            cap = (limits or {}).get(str(name))
            if cap is not None and cap < len(idx):
                rng = np.random.default_rng([seed, p, 1])
                idx = np.sort(rng.choice(len(idx), cap, replace=False))
            for i in idx:
                r = reader.records[int(i)]
                refs.append(LineRef(p, int(i), r.key, r.doc_id, r.label, r.width, r.height))
        if subset is not None and subset < len(refs):
            rng = np.random.default_rng([seed, 2])
            keep = np.sort(rng.choice(len(refs), subset, replace=False))
            refs = [refs[int(i)] for i in keep]
        self.refs = refs
        self.widths = np.array(
            [resized_width(r.width, r.height, height, max_width) for r in refs], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, item: tuple[int, int]) -> dict[str, Any]:
        epoch, i = item
        ref = self.refs[i]
        img = self.readers[ref.pack].image(ref.index).convert("L")
        if self.augmenter is not None:
            img = self.augmenter(img, np.random.default_rng([self.seed, epoch, i]))
        w = resized_width(img.width, img.height, self.height, self.max_width)
        img = img.resize((w, self.height), Image.Resampling.BILINEAR)
        target = self.charset.encode(ref.label)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
        return {"image": x, "target": target, "key": ref.key, "label": ref.label, "index": i}


def collate(batch: list[dict[str, Any]], stride: int = STRIDE) -> dict[str, Any]:
    """Right-pad to the widest line (and to what CTC needs at ``stride`` columns per frame), stack,
    and concatenate targets.

    The padded width is rounded up to a multiple of ``PAD_MULTIPLE``: far fewer distinct tensor
    shapes reach the GPU, which keeps the CUDA memory cache from fragmenting.
    """
    need = [max(b["image"].shape[1], stride * ctc_frames_needed(b["target"])) for b in batch]
    width = int(math.ceil(max(need) / PAD_MULTIPLE) * PAD_MULTIPLE)
    height = batch[0]["image"].shape[0]
    images = torch.empty(len(batch), 1, height, width)
    for k, b in enumerate(batch):
        img = b["image"]
        w = img.shape[1]
        images[k, 0, :, :w] = img
        # Pad with the line's own background (median of its right-most column).
        images[k, 0, :, w:] = img[:, -1].median()
    targets = [t for b in batch for t in b["target"]]
    return {
        "images": images,
        "widths": torch.tensor(need),
        "targets": torch.tensor(targets, dtype=torch.long),
        "target_lengths": torch.tensor([len(b["target"]) for b in batch], dtype=torch.long),
        "keys": [b["key"] for b in batch],
        "labels": [b["label"] for b in batch],
        "indices": [b["index"] for b in batch],
    }


def _split(
    order: np.ndarray, widths: np.ndarray, bs: int, max_columns: int | None
) -> list[list[int]]:
    """Cut width-sorted indices into batches of at most ``bs`` lines and, when ``max_columns`` is
    set, at most that many padded columns (lines x widest line), which bounds GPU memory."""
    out: list[list[int]] = []
    cur: list[int] = []
    for i in order.tolist():
        w = int(widths[i])
        if cur and (len(cur) >= bs or (max_columns and (len(cur) + 1) * w > max_columns)):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


class WidthBucketBatchSampler(Sampler[list[tuple[int, int]]]):
    """Batches of similar width, rebuilt deterministically from ``(seed, epoch)``.

    Training: shuffle all indices, cut them into chunks of ``batch_size * chunk`` lines, sort each
    chunk by width, split it into batches (also capped at ``max_columns`` padded columns), and
    shuffle the batch order. Evaluation (``shuffle=False``): all lines sorted by width. The number
    of batches can differ from epoch to epoch when ``max_columns`` is set; `num_batches` gives it.
    ``start_batch`` skips the batches an interrupted epoch already used. With ``epoch_lines``, each
    training epoch uses only the first ``epoch_lines`` lines of its shuffled order: a fresh sample
    every epoch, no line twice within one. At ``epoch_lines >= len(widths)`` nothing changes.
    """

    def __init__(
        self,
        widths: np.ndarray,
        batch_size: int,
        seed: int,
        chunk: int = 50,
        shuffle: bool = True,
        max_columns: int | None = None,
        epoch_lines: int | None = None,
    ) -> None:
        self.widths, self.batch_size, self.seed = np.asarray(widths), batch_size, seed
        self.chunk, self.shuffle, self.max_columns = chunk, shuffle, max_columns
        self.epoch_lines = epoch_lines
        self.epoch, self.start_batch = 0, 0

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch, self.start_batch = epoch, start_batch

    def batches(self, epoch: int) -> list[list[int]]:
        n, bs = len(self.widths), self.batch_size
        if not self.shuffle:
            order = np.argsort(self.widths, kind="stable")
            return _split(order, self.widths, bs, self.max_columns)
        rng = np.random.default_rng([self.seed, epoch])
        perm = rng.permutation(n)
        if self.epoch_lines is not None and self.epoch_lines < n:
            perm = perm[: self.epoch_lines]
            n = self.epoch_lines
        out: list[list[int]] = []
        step = bs * self.chunk
        for c in range(0, n, step):
            part = perm[c : c + step]
            part = part[np.argsort(self.widths[part], kind="stable")]
            out.extend(_split(part, self.widths, bs, self.max_columns))
        order = rng.permutation(len(out))
        return [out[int(i)] for i in order]

    def num_batches(self, epoch: int) -> int:
        return len(self.batches(epoch))

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for b in self.batches(self.epoch)[self.start_batch :]:
            yield [(self.epoch, i) for i in b]

    def __len__(self) -> int:
        return self.num_batches(self.epoch)


def assert_no_leakage(train: RecLineDataset, val: RecLineDataset) -> None:
    """Train and validation must share no line and no source document (EXPERIMENT_PROTOCOL §6)."""
    keys = {r.key for r in train.refs} & {r.key for r in val.refs}
    docs = {r.doc_id for r in train.refs} & {r.doc_id for r in val.refs}
    if keys or docs:
        raise RuntimeError(f"LEAKAGE: {len(keys)} line keys and {len(docs)} documents are in both "
                           f"training and validation, e.g. {sorted(keys or docs)[:3]}")  # fmt: skip
