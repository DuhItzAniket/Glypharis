"""Contact sheet of seeded detector training crops, for a visual check of the labels (spec §11).

    python scripts/det_preview.py --config configs/det/gw-det-001.yaml [--count 16]

Each tile shows one training crop exactly as the model sees it (augmented), three panels side by
side: the image; the image with the shrunk kernels (green) and the don't-care mask (red) on top;
the threshold map T* (bright near line edges) inside its mask. Crops are drawn from epoch 0 by
``default_rng([seed, 21])``, independently of any result. Run it before the sanity sequence.

Writes reports/training/sanity/<experiment_id>/preview.jpg.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from glypharis.config.cli import add_config_args, config_from_args
from glypharis.data.det_dataset import DetTrainDataset, PageIndex

PREVIEW_STREAM = 21
KERNEL_RGB = np.array([40, 200, 60], dtype=np.float32)
DONT_CARE_RGB = np.array([230, 30, 30], dtype=np.float32)


def tile(item: dict[str, Any], thresh_min: float, thresh_max: float) -> Image.Image:
    img = item["image"].astype(np.float32)
    marked = img.copy()
    kernel = item["shrink_map"].astype(bool)
    dont = ~item["shrink_mask"].astype(bool)
    marked[kernel] = 0.45 * marked[kernel] + 0.55 * KERNEL_RGB
    marked[dont] = 0.45 * marked[dont] + 0.55 * DONT_CARE_RGB
    t = (item["thresh_map"] - thresh_min) / max(thresh_max - thresh_min, 1e-6)
    t = np.where(item["thresh_mask"].astype(bool), 40 + 215 * np.clip(t, 0, 1), 0)
    thresh = np.repeat(t[..., None], 3, axis=2)
    gap = np.full((img.shape[0], 4, 3), 255.0, np.float32)
    row = np.concatenate([img, gap, marked, gap, thresh], axis=1)
    return Image.fromarray(np.clip(row, 0, 255).astype(np.uint8))


def grid(tiles: list[Image.Image], cols: int = 2, gap: int = 8) -> Image.Image:
    """Tiles at native resolution, row by row (a square-cell contact sheet would shrink them)."""
    w, h = max(t.width for t in tiles), max(t.height for t in tiles)
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * w + (cols + 1) * gap, rows * h + (rows + 1) * gap), "white")
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (gap + c * (w + gap), gap + r * (h + gap)))
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    cfg.require_run_identity()
    if cfg.task != "det" or cfg.det is None:
        raise SystemExit(f"{args.config}: not a detector config")
    det, root = cfg.det, Path(cfg.paths.data_root)
    index = PageIndex.build(root / det.data.train_pages[0], "train", root)
    ds = DetTrainDataset(index, root, det, cfg.seed)
    n = min(args.count, len(ds))
    items = np.sort(
        np.random.default_rng([cfg.seed, PREVIEW_STREAM]).choice(len(ds), n, replace=False)
    )
    lab = det.labels
    tiles = [tile(ds[(0, int(t))], lab.thresh_min, lab.thresh_max) for t in items]
    sheet = grid(tiles)
    out = args.out or (
        Path(cfg.paths.reports_dir) / "training" / "sanity" / str(cfg.experiment_id) / "preview.jpg"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
