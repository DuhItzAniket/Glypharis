"""Build GLYPH-C, the frozen corruption suite for robustness evaluation (RQ3, Phase 11).

Usage:
    python scripts/build_glyph_c.py [--per-source 3000] [--seed 7]

Samples a fixed set of validation crops from HierText-val and GlyphSynth-val, then writes each one
clean and under every corruption in `glypharis.data.augment.CORRUPTIONS` at levels 1-5. The noise
seed of each corrupted crop depends only on its key, so the suite does not change if it is rebuilt.

Outputs: data/processed/glyph_c/lines.{bin,index.jsonl,meta.json} and
data/manifests/processed/glyph_c.json (sample list hash and pack hash: the frozen record).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from glypharis.data.augment import CORRUPTIONS, LEVELS, corrupt
from glypharis.data.crops import encode_jpeg
from glypharis.data.linepack import LinePackReader, LinePackWriter

GLYPH_C_VERSION = 1
SOURCES = {
    "hiertext": "processed/hiertext/lines_validation",
    "glyphsynth": "processed/glyphsynth/lines_val",
}


def sample_keys(reader: LinePackReader, n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    n = min(n, len(reader))
    return sorted(rng.choice(len(reader), size=n, replace=False).tolist())


def corruption_seed(key: str) -> int:
    return zlib.crc32(key.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--per-source", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    root = Path(args.data_root)
    prefix = root / "processed" / "glyph_c" / "lines"
    sample_list: list[dict[str, str]] = []
    meta: dict[str, Any] = {
        "glyph_c_version": GLYPH_C_VERSION,
        "corruptions": list(CORRUPTIONS),
        "levels": list(LEVELS),
    }
    with LinePackWriter(prefix, meta) as writer:
        for source, rel in SOURCES.items():
            reader = LinePackReader(root / rel)
            for i in sample_keys(reader, args.per_source, args.seed):
                rec = reader.records[i]
                img = reader.image(i)
                sample_list.append({"source": source, "key": rec.key})
                writer.add(f"{rec.key}|clean|0", rec.doc_id, "glyph_c", source, rec.label,
                           reader.read_bytes(i), (rec.width, rec.height))  # fmt: skip
                for kind in CORRUPTIONS:
                    for level in LEVELS:
                        key = f"{rec.key}|{kind}|{level}"
                        out = corrupt(img, kind, level, corruption_seed(key))
                        writer.add(key, rec.doc_id, "glyph_c", source, rec.label,
                                   encode_jpeg(out, 95), out.size)  # fmt: skip
            reader.close()
        pack = writer.close()

    listing = json.dumps(sample_list, sort_keys=True).encode("utf-8")
    manifest = {
        "glyph_c_version": GLYPH_C_VERSION,
        "seed": args.seed,
        "per_source": args.per_source,
        "sources": SOURCES,
        "corruptions": list(CORRUPTIONS),
        "levels": list(LEVELS),
        "samples": len(sample_list),
        "sample_list_sha256": hashlib.sha256(listing).hexdigest(),
        "crops": pack["count"],
        "bytes": pack["bytes"],
        "bin_sha256": pack["bin_sha256"],
        "frozen": True,
    }
    out = root / "manifests" / "processed" / "glyph_c.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    (root / "manifests" / "processed" / "glyph_c_samples.json").write_text(
        json.dumps(sample_list, indent=0) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"GLYPH-C v{GLYPH_C_VERSION}: {len(sample_list)} samples, {pack['count']} crops, "
          f"{pack['bytes'] / 2**20:.0f} MiB")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
