"""Render augmentation presets and GLYPH-C corruptions on real train crops for a visual check.

Usage:
    python scripts/augment_preview.py [--seed 0]

Writes reports/data_analysis/augmentation_preview.png and glyph_c_preview.png.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

from glypharis.data.augment import CORRUPTIONS, LEVELS, PRESETS, Augmenter, corrupt
from glypharis.data.linepack import LinePackReader


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Preview augmentations on real crops.")
    parser.add_argument("--pack", default="data/processed/hiertext/lines_train")
    parser.add_argument("--out", default="reports/data_analysis")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    reader = LinePackReader(args.pack)
    pick = random.Random(args.seed)
    # Prefer multi-word crops so the effects are visible.
    candidates = [i for i, r in enumerate(reader.records) if 12 <= len(r.label) <= 40]
    idx = pick.sample(candidates, 5)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    presets = [p for p in PRESETS if p != "none"]
    cols = 1 + 2 * len(presets)
    fig, axes = plt.subplots(len(idx), cols, figsize=(3.2 * cols, 1.0 * len(idx) + 0.6))
    for r, i in enumerate(idx):
        img = reader.image(i)
        panels = [("original", img)]
        for preset in presets:
            aug = Augmenter(preset)
            for k in range(2):
                rng = np.random.default_rng(args.seed * 1000 + r * 10 + k)
                panels.append((f"{preset} #{k + 1}", aug(img, rng)))
        for c, (title, im) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.asarray(im), cmap="gray", vmin=0, vmax=255)
            if r == 0:
                ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "augmentation_preview.png", dpi=60)
    plt.close(fig)

    img = reader.image(idx[0])
    fig, axes = plt.subplots(
        len(CORRUPTIONS), len(LEVELS), figsize=(3.2 * len(LEVELS), 0.9 * len(CORRUPTIONS) + 0.6)
    )
    for r, kind in enumerate(CORRUPTIONS):
        for c, level in enumerate(LEVELS):
            ax = axes[r, c]
            ax.imshow(np.asarray(corrupt(img, kind, level, seed=0)), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{kind} {level}", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "glyph_c_preview.png", dpi=60)
    plt.close(fig)
    reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
