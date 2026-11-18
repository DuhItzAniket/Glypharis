"""The Phase 20 hardening inputs, shared by `tests/test_hardening.py` and
`scripts/hardening_check.py` (GW-E2E-012), so both run the same cases.

- `fuzz_cases`: the 120 seeded fuzz-lite cases (seed `FUZZ_SEED`): random arrays, random bytes
  behind an image suffix, and real PNGs damaged in place. The file cases are written into the
  folder given; the random draws are made in one fixed order, so the cases depend on the seed only.
- `unusual_pages`: images that are odd but readable (a 1x1 page, an extreme aspect, RGBA, one
  channel, booleans, 16-bit, floats, and PIL's I;16, P and CMYK modes); `blank_page` and
  `empty_page` beside them.
- `wrong_inputs`: inputs of the wrong type or with wrong values, each refused as an invalid image.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FUZZ_SEED = 20261028
FUZZ_CASES = 120
FUZZ_KINDS = ("random_array", "random_bytes", "damaged_png")
_DTYPES = [np.uint8, np.uint16, np.float32, np.float64, np.int32, np.bool_, np.int8]


def fuzz_cases(
    directory: Path, n: int = FUZZ_CASES, seed: int = FUZZ_SEED
) -> Iterator[tuple[str, Any]]:
    """``(kind, input)`` for each case, in order: an array for ``random_array``, the path of a file
    written into ``directory`` for ``random_bytes`` and ``damaged_png``."""
    rng = np.random.default_rng(seed)
    for k in range(n):
        kind = k % 3
        x: Any
        if kind == 0:  # random arrays
            shape = tuple(int(s) for s in rng.integers(0, 70, size=int(rng.integers(1, 5))))
            dt = _DTYPES[int(rng.integers(len(_DTYPES)))]
            x = (rng.random(shape) * rng.choice([1, 255, 70000])).astype(dt)
        elif kind == 1:  # random bytes behind an image suffix
            x = directory / f"f{k}.{['png', 'jpg', 'tif', 'bmp'][k % 4]}"
            x.write_bytes(rng.integers(0, 256, int(rng.integers(0, 400)), np.uint8).tobytes())
        else:  # a real image, then damaged in place
            x = directory / f"d{k}.png"
            Image.fromarray(rng.integers(0, 255, (30, 50, 3), np.uint8)).save(x)
            b = bytearray(x.read_bytes())
            for i in rng.integers(0, len(b), 8):
                b[int(i)] = int(rng.integers(0, 256))
            x.write_bytes(bytes(b))
        yield FUZZ_KINDS[kind], x


def blank_page() -> np.ndarray:
    """A white page: read, with no lines."""
    return np.full((300, 400, 3), 255, np.uint8)


def empty_page() -> np.ndarray:
    """No pixels at all: refused as empty."""
    return np.zeros((0, 10, 3), np.uint8)


def unusual_pages() -> list[tuple[str, Any]]:
    """``(name, image)`` pairs that must each give a result."""
    return [
        ("1x1", np.zeros((1, 1), np.uint8)),
        ("extreme_aspect", np.zeros((5, 3000), np.uint8)),
        ("rgba", np.full((40, 60, 4), 200, np.uint8)),
        ("one_channel", np.full((40, 60, 1), 200, np.uint8)),
        ("bool", np.ones((40, 60), bool)),
        ("uint16", np.full((40, 60), 40000, np.uint16)),
        ("float_0_1", np.random.default_rng(0).random((40, 60, 3))),  # floats in [0, 1]
        ("pil_i16", Image.new("I;16", (60, 40), 1000)),
        ("pil_p", Image.new("P", (60, 40))),
        ("pil_cmyk", Image.new("CMYK", (60, 40))),
    ]


def wrong_inputs() -> list[tuple[str, Any]]:
    """``(name, input)`` pairs that must each raise `InvalidImageError`."""
    return [
        ("none", None),
        ("int", 42),
        ("non_image_bytes", b"bytes"),
        ("nested_list", [[0, 1], [2, 3]]),
        ("4d_array", np.zeros((2, 3, 4, 5), np.uint8)),
        ("2_channels", np.zeros((10, 10, 2), np.uint8)),
        ("nan", np.full((10, 10), np.nan)),
        ("negative_float", np.full((10, 10), -1.0)),
        ("int64_over_255", np.full((10, 10), 300, np.int64)),
        ("complex", np.zeros((10, 10), np.complex64)),
    ]
