"""Generate GlyphSynth line packs.

Usage:
    python scripts/build_glyphsynth.py [--train 100000] [--val 10000] [--seed 20261010]

Inputs: data/raw/google_fonts (pinned commit), data/raw/pg19, data/manifests/charset.json.
Outputs:
    data/manifests/glyphsynth_fonts.json        font faces, licences, charset coverage, split
    data/processed/glyphsynth/lines_{train,val}.{bin,index.jsonl,meta.json,render.jsonl}
    data/manifests/processed/glyphsynth.json    settings, versions and hashes
    reports/data_analysis/glyphsynth/report.md, samples.png
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import string
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from glypharis.data.crops import encode_jpeg
from glypharis.data.linepack import LinePackReader, LinePackWriter
from glypharis.data.synth import (
    CAPTURE_STEPS,
    FontFace,
    GlyphSynth,
    RenderConfig,
    TextConfig,
    TextSampler,
    book_paragraphs,
    read_face,
    split_families,
)
from glypharis.data.text import Charset
from glypharis.utils.gitinfo import git_info
from glypharis.utils.logsetup import setup_logging

GLYPHSYNTH_VERSION = 1
HELD_OUT_FRACTION = 0.15
FONT_SPLIT_SEED = 4
TEXT_PARTS = {"train": ("train",), "val": ("validation", "test")}
log = logging.getLogger("build_glyphsynth")

_GEN: GlyphSynth | None = None


# ------------------------------------------------------------------------------------ fonts


def scan_fonts(font_root: Path, charset: Charset) -> list[dict[str, Any]]:
    ascii_needed = set(string.ascii_letters + string.digits + ".,;:!?'\"()-")
    out = []
    for path in sorted(font_root.rglob("*.ttf")):
        rel = path.relative_to(font_root).as_posix()
        category, licence_dir, family = rel.split("/")[:3]
        face = read_face(font_root, rel, family, category, licence_dir)
        cov = sum(1 for ch in charset.characters if ch == " " or ord(ch) in face.codepoints)
        missing_ascii = sorted(ch for ch in ascii_needed if ord(ch) not in face.codepoints)
        out.append(
            {
                "family": family,
                "category": category,
                "licence_dir": licence_dir,
                "path": rel,
                "italic": face.italic,
                "variable_weight": face.variable_weight,
                "charset_coverage": round(cov / len(charset.characters), 4),
                "usable": not missing_ascii,
                "missing_ascii": "".join(missing_ascii),
            }
        )
    return out


def load_faces(font_root: Path, records: list[dict[str, Any]], split: str) -> list[FontFace]:
    return [
        read_face(font_root, r["path"], r["family"], r["category"], r["licence_dir"])
        for r in records
        if r["usable"] and r["split"] == split
    ]


# ------------------------------------------------------------------------------------- text


def load_paragraphs(pg_root: Path, parts: tuple[str, ...]) -> list[str]:
    paras: list[str] = []
    for part in parts:
        for book in sorted((pg_root / part).glob("*.txt")):
            paras.extend(book_paragraphs(book.read_text(encoding="utf-8")))
    return paras


# ---------------------------------------------------------------------------------- workers


def _init_worker(
    font_root: str, font_records: list[dict[str, Any]], pg_root: str, charset_path: str,
    split: str, capture: bool,
) -> None:  # fmt: skip
    global _GEN
    charset = Charset.load(charset_path)
    faces = load_faces(Path(font_root), font_records, split)
    sampler = TextSampler(load_paragraphs(Path(pg_root), TEXT_PARTS[split]), charset, TextConfig())
    _GEN = GlyphSynth(faces, Path(font_root), sampler, RenderConfig(capture=capture))


def _render_chunk(
    args: tuple[int, int, int],
) -> list[tuple[int, str, bytes, tuple[int, int], dict[str, Any]]]:
    seed, start, stop = args
    assert _GEN is not None
    out = []
    for i in range(start, stop):
        text, img, info = _GEN.sample(seed, i)
        out.append((i, text, encode_jpeg(img, 95), img.size, info.to_dict()))
    return out


# ----------------------------------------------------------------------------------- report


def _sheet(reader: LinePackReader, path: Path, n: int = 40) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(n, 1, figsize=(10, 0.75 * n))
    for i, ax in enumerate(axes):
        ax.imshow(np.asarray(reader.image(i)), cmap="gray", vmin=0, vmax=255)
        ax.set_title(repr(reader.records[i].label), fontsize=7, loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--train", type=int, default=100_000)
    parser.add_argument("--val", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--no-capture", action="store_true", help="render without capture effects")
    parser.add_argument("--workers", type=int, default=min(12, max(1, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args(argv)
    root = Path(args.data_root)
    setup_logging(root / "cache" / "logs", run_id="build_glyphsynth")
    font_root = root / "raw" / "google_fonts"
    pg_root = root / "raw" / "pg19"
    charset_path = root / "manifests" / "charset.json"
    charset = Charset.load(charset_path)

    fonts = scan_fonts(font_root, charset)
    families = {r["family"]: r["category"] for r in fonts if r["usable"]}
    split_of = split_families(families, HELD_OUT_FRACTION, FONT_SPLIT_SEED)
    for r in fonts:
        r["split"] = split_of.get(r["family"], "unused")
    (root / "manifests" / "glyphsynth_fonts.json").write_text(
        json.dumps({"held_out_fraction": HELD_OUT_FRACTION, "split_seed": FONT_SPLIT_SEED,
                    "faces": fonts}, indent=1) + "\n",
        encoding="utf-8", newline="\n",
    )  # fmt: skip
    n_val = sum(v == "val" for v in split_of.values())
    log.info("fonts: %d faces, %d usable families (%d train, %d val)", len(fonts), len(families),
             len(families) - n_val, n_val)  # fmt: skip

    out_dir = root / "processed" / "glyphsynth"
    manifest: dict[str, Any] = {
        "glyphsynth_version": GLYPHSYNTH_VERSION,
        "seed": args.seed,
        "capture": not args.no_capture,
        "render_config": asdict(RenderConfig(capture=not args.no_capture)),
        "text_config": asdict(TextConfig()),
        "capture_steps": [asdict(s) for s in CAPTURE_STEPS],
        "charset_version": charset.version,
        "font_commit": "bd8f81ddb5c74d5c8897b36ad88b440266245103",
        "splits": {},
    }
    stats: dict[str, Any] = {}
    for split, n in (("train", args.train), ("val", args.val)):
        if n <= 0:
            continue
        seed = args.seed if split == "train" else args.seed + 1
        prefix = out_dir / f"lines_{split}"
        start = time.monotonic()
        chunks = [(seed, s, min(s + 500, n)) for s in range(0, n, 500)]
        init = (str(font_root), fonts, str(pg_root), str(charset_path), split, not args.no_capture)
        families_used: collections.Counter[str] = collections.Counter()
        effects: collections.Counter[str] = collections.Counter()
        lengths = []
        meta = {"split": split, "source": "glyphsynth", "glyphsynth_version": GLYPHSYNTH_VERSION}
        render_log = prefix.with_suffix(".render.jsonl")
        with (
            ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=init) as pool,
            LinePackWriter(prefix, meta) as writer,
            render_log.open("w", encoding="utf-8", newline="\n") as rl,
        ):
            for chunk in pool.map(_render_chunk, chunks):
                for i, text, data, size, info in chunk:
                    key = f"glyphsynth-{split}-{i:07d}"
                    writer.add(key, key, split, "glyphsynth", text, data, size)
                    rl.write(json.dumps({"key": key, **info}, ensure_ascii=False) + "\n")
                    families_used[info["family"]] += 1
                    effects.update(info["capture"])
                    lengths.append(len(text))
            pack = writer.close()
        log.info(
            "%s: %d lines in %.0fs (%.0f MiB)",
            split,
            n,
            time.monotonic() - start,
            pack["bytes"] / 2**20,
        )
        manifest["splits"][split] = {
            "count": pack["count"], "bytes": pack["bytes"], "bin_sha256": pack["bin_sha256"],
            "seed": seed, "families": len(families_used),
        }  # fmt: skip
        stats[split] = {
            "families": len(families_used),
            "length_p50": float(np.percentile(lengths, 50)),
            "length_p95": float(np.percentile(lengths, 95)),
            "effects": {k: v / n for k, v in effects.most_common()},
        }
        if split == "train":
            rep_dir = Path(args.reports) / "data_analysis" / "glyphsynth"
            rep_dir.mkdir(parents=True, exist_ok=True)
            reader = LinePackReader(prefix)
            _sheet(reader, rep_dir / "samples.png")
            reader.close()

    mp = root / "manifests" / "processed" / "glyphsynth.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")

    g = git_info()
    unusable = [r for r in fonts if not r["usable"]]
    by_split = collections.Counter((split_of[f], c) for f, c in families.items())
    lines = [
        "# GlyphSynth v1",
        "",
        f"Generated by `scripts/build_glyphsynth.py` on {datetime.now(UTC).date().isoformat()} "
        f"from commit `{(g.commit or 'unknown')[:7]}`. Do not edit by hand. Hashes and settings: "
        "`data/manifests/processed/glyphsynth.json`; fonts: "
        "`data/manifests/glyphsynth_fonts.json`.",
        "",
        "## Fonts",
        "",
        f"{len(fonts)} faces from 100 Google Fonts families (commit bd8f81d). Families are split "
        f"{1 - HELD_OUT_FRACTION:.0%} / {HELD_OUT_FRACTION:.0%} into train / held-out per category "
        f"(seed {FONT_SPLIT_SEED}).",
        "",
        "| Category | Train families | Held-out families |",
        "|---|---|---|",
        *[
            f"| {c} | {by_split[('train', c)]} | {by_split[('val', c)]} |"
            for c in ("serif", "sans", "mono")
        ],
        "",
        f"Faces excluded for missing basic ASCII glyphs: {len(unusable)}"
        + (
            ": " + ", ".join(f"{r['path']} (missing {r['missing_ascii']!r})" for r in unusable)
            if unusable
            else ""
        ),
        "",
        "## Text",
        "",
        "Train lines use the 300 sampled PG-19 train books; held-out lines use the 150 PG-19 "
        "validation and test books. About "
        f"{TextConfig().modern_fraction:.0%} of lines are generated modern snippets (dates, "
        "money, codes, contacts, references, list items).",
        "",
        "## Output",
        "",
        "| Split | Lines | Families used | Median / p95 characters | Size |",
        "|---|---|---|---|---|",
    ]
    for split, st in stats.items():
        sp = manifest["splits"][split]
        lines.append(f"| {split} | {sp['count']:,} | {st['families']} | {st['length_p50']:.0f} / "
                     f"{st['length_p95']:.0f} | {sp['bytes'] / 2**20:,.0f} MiB |")  # fmt: skip
    if "train" in stats:
        lines += ["", "Capture effects (share of train lines): "
                  + ", ".join(f"{k} {v:.2f}" for k, v in stats["train"]["effects"].items()),
                  "", "First 40 train lines: `samples.png`.", ""]  # fmt: skip
    rp = Path(args.reports) / "data_analysis" / "glyphsynth" / "report.md"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
