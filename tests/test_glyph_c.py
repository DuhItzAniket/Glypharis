from __future__ import annotations

import json

import build_glyph_c
from PIL import Image

from glypharis.data.crops import encode_jpeg
from glypharis.data.linepack import LinePackReader, LinePackWriter


def _pack(prefix, n):
    with LinePackWriter(prefix) as w:
        for i in range(n):
            img = Image.new("L", (80 + i, 64), 200)
            w.add(f"k{i}", f"d{i}", "val", "src", f"text {i}", encode_jpeg(img, 95), img.size)


def test_glyph_c_build_is_frozen_and_reproducible(tmp_path):
    root = tmp_path / "data"
    _pack(root / "processed/hiertext/lines_validation", 6)
    _pack(root / "processed/glyphsynth/lines_val", 5)
    args = ["--data-root", str(root), "--per-source", "3", "--seed", "1"]
    assert build_glyph_c.main(args) == 0
    m1 = json.loads((root / "manifests/processed/glyph_c.json").read_text(encoding="utf-8"))
    per_sample = 1 + len(m1["corruptions"]) * len(m1["levels"])
    assert m1["samples"] == 6 and m1["crops"] == 6 * per_sample and m1["frozen"] is True
    r = LinePackReader(root / "processed/glyph_c/lines")
    keys = [rec.key for rec in r.records]
    assert keys[0].endswith("|clean|0") and any(k.endswith("|rotate|5") for k in keys)
    assert all(rec.height == 64 for rec in r.records)
    r.close()
    assert build_glyph_c.main(args) == 0
    m2 = json.loads((root / "manifests/processed/glyph_c.json").read_text(encoding="utf-8"))
    assert m2 == m1  # same samples, same bytes


def test_corruption_seed_depends_only_on_key():
    assert build_glyph_c.corruption_seed("a|jpeg|3") == build_glyph_c.corruption_seed("a|jpeg|3")
    assert build_glyph_c.corruption_seed("a|jpeg|3") != build_glyph_c.corruption_seed("a|jpeg|4")
