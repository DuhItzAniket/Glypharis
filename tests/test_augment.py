from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from glypharis.data.audit import laplacian_variance
from glypharis.data.augment import (
    CORRUPTIONS,
    LEVELS,
    PRESETS,
    TRANSFORMS,
    Augmenter,
    corrupt,
)


def _line_image(w: int = 240, h: int = 64) -> Image.Image:
    img = Image.new("L", (w, h), 230)
    draw = ImageDraw.Draw(img)
    for x in range(10, w - 20, 22):  # dark "glyph" strokes
        draw.rectangle((x, 16, x + 6, 48), fill=20)
        draw.rectangle((x, 16, x + 14, 22), fill=20)
    return img


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_transform_keeps_mode_and_is_seeded(name):
    img = _line_image()
    a = TRANSFORMS[name](img, np.random.default_rng(1), 1.0)
    b = TRANSFORMS[name](img, np.random.default_rng(1), 1.0)
    assert a.mode == "L" and a.height == img.height
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_presets_keep_height_and_are_reproducible(preset):
    aug = Augmenter(preset)
    img = _line_image()
    outs = [aug(img, np.random.default_rng(7)) for _ in range(2)]
    assert all(o.height == 64 and o.mode == "L" for o in outs)
    assert np.array_equal(np.asarray(outs[0]), np.asarray(outs[1]))


def test_none_is_identity_and_full_changes_something():
    img = _line_image()
    assert np.array_equal(
        np.asarray(Augmenter("none")(img, np.random.default_rng(0))), np.asarray(img)
    )
    full = Augmenter("full", {name: {"p": 1.0} for name in ("gaussian_noise", "rotate")})
    out = full(img, np.random.default_rng(0))
    assert out.size != img.size or not np.array_equal(np.asarray(out), np.asarray(img))


def test_overrides_are_validated():
    assert Augmenter("photometric", {"jpeg": {"p": 0.9}}).steps[-1].p == 0.9
    with pytest.raises(ValueError, match="unknown augmentation preset"):
        Augmenter("strong")
    with pytest.raises(ValueError, match="no transform"):
        Augmenter("geometric", {"jpeg": {"p": 1.0}})
    with pytest.raises(ValueError, match="unknown parameters"):
        Augmenter("photometric", {"jpeg": {"quality": 10}})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Augmenter("photometric", {"jpeg": {"p": 2.0}})


@pytest.mark.parametrize("kind", CORRUPTIONS)
def test_corruptions_are_deterministic(kind):
    img = _line_image()
    for level in LEVELS:
        a, b = corrupt(img, kind, level, seed=3), corrupt(img, kind, level, seed=3)
        assert a.height == 64 and np.array_equal(np.asarray(a), np.asarray(b))


def test_corruption_strength_increases_with_level():
    img = _line_image()
    sharp = [laplacian_variance(np.asarray(corrupt(img, "gaussian_blur", lv, 0), dtype=np.float32))
             for lv in LEVELS]  # fmt: skip
    assert sharp == sorted(sharp, reverse=True)
    spread = [
        float(np.asarray(corrupt(img, "contrast", lv, 0), dtype=np.float32).std()) for lv in LEVELS
    ]
    assert spread == sorted(spread, reverse=True)


def test_unknown_corruption():
    with pytest.raises(ValueError):
        corrupt(_line_image(), "fog", 1, 0)
    with pytest.raises(ValueError):
        corrupt(_line_image(), "jpeg", 6, 0)
