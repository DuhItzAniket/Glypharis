"""GlyphSynth training packs for the Phase 12 synthetic data study.

    python scripts/build_glyphsynth_variants.py ext         # train lines 100,000 to 249,999
    python scripts/build_glyphsynth_variants.py fewfonts    # 100,000 lines, 6 font families
    python scripts/build_glyphsynth_variants.py nocapture   # 100,000 lines, no capture effects
    python scripts/build_glyphsynth_variants.py ext --count 2000 --out-name probe   # a timing probe

Every variant is GlyphSynth v1 (`scripts/build_glyphsynth.py`): the same renderer, text sampler,
training font families and seed (20261010), so a line is fully determined by its index.

- ``ext`` continues the v1 training pack: indices 100,000 to 249,999, rendered exactly as v1 would
  have rendered them, so ``lines_train`` + ``lines_train_ext`` is the 250,000-line pack.
- ``fewfonts`` renders indices 0 to 99,999 from `FEW_FAMILIES` training families (chosen with a
  fixed seed, in proportion to the serif, sans and mono families) instead of all 85.
- ``nocapture`` renders indices 0 to 99,999 with the capture effects (rotation, perspective, paper
  texture, uneven light, blur, noise, JPEG) switched off. Text, font and layout are those of
  ``lines_train``: only the capture step differs.

Outputs (git-ignored): ``data/processed/glyphsynth/lines_train_<variant>.*``; settings, counts and
hashes go to ``data/manifests/processed/glyphsynth_variants.json`` (committed).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from glypharis.data.crops import encode_jpeg
from glypharis.data.linepack import LinePackWriter
from glypharis.data.synth import GlyphSynth, RenderConfig, TextConfig, TextSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_glyphsynth as v1

SEED = 20261010
FEW_FAMILIES = 6
FEW_SEED = 12
VARIANTS = {
    "ext": {"start": 100_000, "count": 150_000, "capture": True, "few": False},
    "fewfonts": {"start": 0, "count": 100_000, "capture": True, "few": True},
    "nocapture": {"start": 0, "count": 100_000, "capture": False, "few": False},
}
MANIFEST = Path("data/manifests/processed/glyphsynth_variants.json")

_GEN: GlyphSynth | None = None


def few_families(
    records: list[dict[str, Any]], n: int = FEW_FAMILIES, seed: int = FEW_SEED
) -> list[str]:
    """``n`` training families, drawn per category in proportion to the category sizes."""
    by_cat: dict[str, list[str]] = collections.defaultdict(list)
    for fam, cat in sorted(
        {(r["family"], r["category"]) for r in records if r["usable"] and r["split"] == "train"}
    ):
        by_cat[cat].append(fam)
    total = sum(len(v) for v in by_cat.values())
    quota = {c: max(1, round(n * len(v) / total)) for c, v in by_cat.items()}
    while sum(quota.values()) > n:  # rounding up can overshoot: trim the largest category
        quota[max(quota, key=lambda c: (quota[c], c))] -= 1
    rng = np.random.default_rng(seed)
    out = []
    for cat in sorted(by_cat):
        out += sorted(rng.choice(by_cat[cat], quota[cat], replace=False).tolist())
    return out


def _init_worker(
    font_root: str, records: list[dict[str, Any]], pg_root: str, charset_path: str, capture: bool
) -> None:
    global _GEN
    charset = v1.Charset.load(charset_path)
    faces = v1.load_faces(Path(font_root), records, "train")
    sampler = TextSampler(
        v1.load_paragraphs(Path(pg_root), v1.TEXT_PARTS["train"]), charset, TextConfig()
    )
    _GEN = GlyphSynth(faces, Path(font_root), sampler, RenderConfig(capture=capture))


def _render(args: tuple[int, int, int]) -> list[tuple[int, str, bytes, tuple[int, int], dict]]:
    seed, start, stop = args
    assert _GEN is not None
    out = []
    for i in range(start, stop):
        text, img, info = _GEN.sample(seed, i)
        out.append((i, text, encode_jpeg(img, 95), img.size, info.to_dict()))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant", choices=sorted(VARIANTS))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=None, help="fewer lines (timing probe)")
    parser.add_argument("--out-name", default=None, help="pack name suffix (default: the variant)")
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    spec = VARIANTS[args.variant]
    root = args.data_root
    fonts_doc = json.loads((root / "manifests/glyphsynth_fonts.json").read_text(encoding="utf-8"))
    records = fonts_doc["faces"]
    families = None
    if spec["few"]:
        families = few_families(records)
        keep = set(families)
        records = [r for r in records if r["family"] in keep]
    start, count = spec["start"], args.count or spec["count"]
    name = args.out_name or args.variant
    prefix = root / "processed/glyphsynth" / f"lines_train_{name}"
    key_base = "train" if args.variant == "ext" else args.variant
    chunks = [(SEED, s, min(s + 500, start + count)) for s in range(start, start + count, 500)]
    init = (
        str(root / "raw/google_fonts"),
        records,
        str(root / "raw/pg19"),
        str(root / "manifests/charset.json"),
        spec["capture"],
    )
    meta = {
        "split": "train",
        "source": "glyphsynth",
        "glyphsynth_version": v1.GLYPHSYNTH_VERSION,
        "variant": args.variant,
    }
    used: collections.Counter[str] = collections.Counter()
    t0 = time.monotonic()
    with (
        ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=init) as pool,
        LinePackWriter(prefix, meta) as writer,
        prefix.with_suffix(".render.jsonl").open("w", encoding="utf-8", newline="\n") as rl,
    ):
        for chunk in pool.map(_render, chunks):
            for i, text, data, size, info in chunk:
                key = f"glyphsynth-{key_base}-{i:07d}"
                writer.add(key, key, "train", "glyphsynth", text, data, size)
                rl.write(json.dumps({"key": key, **info}, ensure_ascii=False) + "\n")
                used[info["family"]] += 1
        pack = writer.close()
    secs = time.monotonic() - t0
    print(
        f"{name}: {pack['count']} lines in {secs:.0f} s ({pack['count'] / secs:.0f} lines/s), "
        f"{len(used)} families"
    )
    if args.count is None:
        manifest = (
            json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {}
        )
        manifest[args.variant] = {
            "pack": f"processed/glyphsynth/lines_train_{name}",
            "seed": SEED,
            "indices": [start, start + count],
            "capture": spec["capture"],
            "families": sorted(used) if spec["few"] else len(used),
            "count": pack["count"],
            "bytes": pack["bytes"],
            "bin_sha256": pack["bin_sha256"],
            "seconds": round(secs),
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
