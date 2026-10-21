from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

from glypharis.data.synth import (
    GlyphSynth,
    RenderConfig,
    TextSampler,
    book_paragraphs,
    modern_snippet,
    read_face,
    split_families,
)
from glypharis.data.text import Charset

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONT = "DejaVuSans.ttf"  # Bitstream Vera / DejaVu licence; ships with matplotlib


@pytest.fixture(scope="module")
def charset(repo_root_module):
    return Charset.load(repo_root_module / "data" / "manifests" / "charset.json")


@pytest.fixture(scope="module")
def repo_root_module():
    return Path(__file__).resolve().parents[1]


BOOK = """The Project Gutenberg EBook of Something

*** START OF THIS PROJECT GUTENBERG EBOOK ***

It was the best of times, it was the worst of times, it was the age of
wisdom, it was the age of foolishness, it was the epoch of belief.

Short.

There were a king with a large jaw and a queen with a plain face, on the
throne of England; there were a king with a large jaw and a queen with
a fair face, on the throne of France.
"""


def test_book_paragraphs_joins_hard_wraps_and_skips_short_blocks():
    paras = book_paragraphs(BOOK)
    assert len(paras) == 3  # title line, the two prose paragraphs; "Short." and *** are dropped
    assert paras[1].startswith("It was the best of times") and "\n" not in paras[1]
    assert not any(p.startswith("***") for p in paras)


def test_modern_snippets_fit_the_charset(charset):
    rng = np.random.default_rng(0)
    for _ in range(500):
        s = modern_snippet(rng)
        assert s and not charset.oov(s), s


def test_text_sampler_is_seeded_and_in_charset(charset):
    sampler = TextSampler(book_paragraphs(BOOK) * 5, charset)
    a = [sampler.sample(np.random.default_rng([1, i])) for i in range(200)]
    b = [sampler.sample(np.random.default_rng([1, i])) for i in range(200)]
    assert a == b
    assert all(t and not charset.oov(t) for t in a)
    assert max(len(t) for t in a) > 30  # body-text lines occur
    assert any(len(t) <= 6 for t in a)  # very short lines occur


def test_split_families_is_stratified_and_deterministic():
    fams = {f"s{i}": "serif" for i in range(20)} | {f"m{i}": "mono" for i in range(4)}
    a = split_families(fams, 0.15, seed=4)
    assert a == split_families(fams, 0.15, seed=4)
    assert sum(v == "val" for k, v in a.items() if k.startswith("s")) == 3
    assert sum(v == "val" for k, v in a.items() if k.startswith("m")) == 1


def test_read_face():
    face = read_face(FONT_DIR, FONT, "dejavusans", "sans", "other")
    assert ord("A") in face.codepoints and face.covers("Hello, world 42")
    assert not face.italic and face.variable_weight is None


def _gen(charset, capture=True):
    face = read_face(FONT_DIR, FONT, "dejavusans", "sans", "other")
    sampler = TextSampler(book_paragraphs(BOOK) * 5, charset)
    return GlyphSynth([face], FONT_DIR, sampler, RenderConfig(capture=capture))


def test_glyphsynth_samples_are_reproducible(charset):
    gen = _gen(charset)
    t1, img1, info1 = gen.sample(7, 3)
    t2, img2, info2 = gen.sample(7, 3)
    assert t1 == t2 and info1 == info2
    assert np.array_equal(np.asarray(img1), np.asarray(img2))
    assert img1.mode == "L" and img1.height == 64
    t3, img3, _ = gen.sample(7, 4)
    assert (t3, img3.size) != (t1, img1.size) or not np.array_equal(
        np.asarray(img3), np.asarray(img1)
    )


def test_capture_switch(charset):
    gen = _gen(charset, capture=False)
    for i in range(30):
        _, _, info = gen.sample(1, i)
        assert info.capture == []
    effects = {e for i in range(60) for e in _gen(charset).sample(1, i)[2].capture}
    assert effects  # with capture on, some effects are applied


def test_rendered_text_is_dark_on_light_unless_inverted(charset):
    gen = _gen(charset, capture=False)
    for i in range(20):
        _, img, info = gen.sample(2, i)
        a = np.asarray(img, dtype=np.float32)
        border = np.concatenate([a[:2].ravel(), a[-2:].ravel()])
        if not info.inverted and not info.neighbours:
            assert border.mean() > a.mean() - 1  # background is the brightest part


def test_gutenberg_markup_is_cleaned(charset):
    paras = book_paragraphs(
        "He said _never_ again--and meant it, as everyone in the room knew well."
    )
    assert paras == ["He said never again--and meant it, as everyone in the room knew well."]
    sampler = TextSampler(paras * 3, charset)
    outs = {sampler.sample(np.random.default_rng([0, i])) for i in range(200)}
    assert any("—" in t for t in outs) and not any("_" in t for t in outs)
