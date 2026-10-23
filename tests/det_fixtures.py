"""Synthetic pages and images shared by the detector tests (no real data, CPU only).

`make_synthetic_pages` writes a tiny HierText-shaped dataset under ``<tmp>/data``: page files
written with `write_pages`, one JPEG per page (rotated dark bars on textured paper), and the
processed manifest whose hashes `PageIndex.build` checks. Some lines are illegible or handwritten,
so every page mixes text and don't-care lines; the last training page has no text line at all
(HierText has a few such pages). `tiny_det_config` is the CPU smoke configuration of §12.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from glypharis.config.loader import build_config
from glypharis.config.schema import RunConfig
from glypharis.data.download import sha256_file
from glypharis.data.pages import Page, PageLine, write_pages

SOURCE = "detsynth"
WORDS = ("the", "of", "and", "page", "line", "text", "ocr", "print")


@dataclass
class SyntheticPages:
    data_root: Path
    train_pages: str  # page file relative to data_root, as det.data.train_pages lists it
    val_pages: str
    train: list[Page]
    val: list[Page]

    @property
    def train_file(self) -> Path:
        return self.data_root / self.train_pages

    @property
    def val_file(self) -> Path:
        return self.data_root / self.val_pages


def rect_polygon(
    cx: float, cy: float, length: float, height: float, angle_deg: float = 0.0
) -> list[list[float]]:
    """TL, TR, BR, BL corners of a rotated rectangle (clockwise on screen, y down)."""
    a = math.radians(angle_deg)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-math.sin(a), math.cos(a)])
    c = np.array([cx, cy])
    hl, hh = length / 2, height / 2
    corners = [c - u * hl - v * hh, c + u * hl - v * hh, c + u * hl + v * hh, c - u * hl + v * hh]
    return [[float(x), float(y)] for x, y in corners]


def random_quad(
    rng: np.random.Generator,
    canvas_wh: tuple[int, int],
    height: tuple[float, float] = (4.0, 30.0),
    aspect: tuple[float, float] = (1.0, 15.0),
    max_angle: float = 45.0,
    skew: float = 0.0,
) -> np.ndarray:
    """A random rotated rectangle (or, with ``skew`` > 0, a sheared convex quad) as (4, 2)."""
    w, h = canvas_wh
    hgt = rng.uniform(*height)
    poly = np.array(
        rect_polygon(rng.uniform(0, w), rng.uniform(0, h), hgt * rng.uniform(*aspect), hgt,
                     rng.uniform(-max_angle, max_angle))
    )  # fmt: skip
    if skew:
        shift = rng.uniform(-skew, skew) * hgt
        poly[[0, 1]] += (poly[1] - poly[0]) / np.linalg.norm(poly[1] - poly[0]) * shift
    return poly


def make_line(
    line_id: str, polygon: list[list[float]], text: str = "text", **flags: bool | float | None
) -> PageLine:
    """A page line, a box line unless ``flags`` (legible, handwritten, vertical, align_score)
    say otherwise."""
    line = PageLine(line_id, polygon, text, text, True, False, False, True)
    for key, value in flags.items():
        setattr(line, key, value)
    return line


def _paper(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Light, slightly warm paper with a soft gradient and grain (RGB float)."""
    yy, xx = np.mgrid[0:h, 0:w]
    shade = 12 * np.sin(xx / w * rng.uniform(1, 4) + rng.uniform(0, 6)) * np.cos(yy / h * 2)
    base = np.array([228.0, 222.0, 208.0]) + rng.uniform(-10, 10, 3)
    grain = rng.normal(0, 6, (h, w, 1))
    out: np.ndarray = base + (shade[..., None] + grain)
    return out


def _draw_bar(img: np.ndarray, polygon: list[list[float]], ink: float) -> None:
    pts = np.round(np.asarray(polygon) * 16).astype(np.int32)  # 4 bits of sub-pixel precision
    cv2.fillPoly(img, [pts], (ink, ink, ink), lineType=cv2.LINE_8, shift=4)


def _page_lines(
    rng: np.random.Generator, doc_id: str, w: int, h: int, no_care: bool
) -> list[PageLine]:
    """3–6 bars fully inside the page, stacked top to bottom, some of them don't care."""
    n = int(rng.integers(3, 7))
    lines: list[PageLine] = []
    rows = np.linspace(0.12, 0.88, n) * h
    for k, cy in enumerate(rows):
        for _ in range(50):
            height = rng.uniform(6.0, 13.0)
            length = rng.uniform(0.25, 0.7) * w
            angle = rng.uniform(-8, 8) if rng.random() < 0.8 else rng.uniform(-25, 25)
            poly = rect_polygon(rng.uniform(0.3, 0.7) * w, cy, length, height, angle)
            p = np.asarray(poly)
            if p.min() > 1 and p[:, 0].max() < w - 1 and p[:, 1].max() < h - 1:
                break
        u = rng.random()
        flags: dict[str, bool] = {}
        if no_care or u < 0.15:
            flags["legible"] = False
        elif u < 0.25:
            flags["handwritten"] = True
        words = " ".join(rng.choice(WORDS, int(rng.integers(1, 4))))
        lines.append(make_line(f"{doc_id}_l{k}", poly, words, **flags))
    return lines


def _write_jpeg(path: Path, rgb: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])  # fmt: skip
    assert ok
    buf.tofile(path)


def write_processed_manifest(data_root: Path, source: str, files: dict[str, Path]) -> Path:
    """``manifests/processed/<source>.json`` with ``splits.<split>.pages_sha256`` per page file."""
    out = data_root / "manifests" / "processed" / f"{source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    splits = {
        split: {"pages_file": path.relative_to(data_root).as_posix(),
                "pages_sha256": sha256_file(path)}
        for split, path in files.items()
    }  # fmt: skip
    out.write_text(json.dumps({"dataset": source, "splits": splits}, indent=1), encoding="utf-8")
    return out


def make_synthetic_pages(
    tmp: Path,
    n_train: int = 12,
    n_val: int = 4,
    size: tuple[int, int] = (160, 128),
    seed: int = 0,
    ignore_regions: bool = False,
) -> SyntheticPages:
    """Write the synthetic dataset under ``tmp / "data"``; ``size`` is (W, H).

    With ``ignore_regions`` the first training page also gets one ignore polygon.
    """
    rng = np.random.default_rng(seed)
    data_root = Path(tmp) / "data"
    w, h = size
    out: dict[str, list[Page]] = {}
    files: dict[str, Path] = {}
    for split, n in (("train", n_train), ("validation", n_val)):
        pages: list[Page] = []
        for i in range(n):
            doc_id = f"{SOURCE}_{split}_{i:03d}"
            image = f"raw/{SOURCE}/images/{split}/{doc_id}.jpg"
            no_care = split == "train" and i == n - 1
            lines = _page_lines(rng, doc_id, w, h, no_care)
            rgb = _paper(rng, w, h)
            for line in lines:
                _draw_bar(rgb, line.polygon, rng.uniform(20, 70))
            _write_jpeg(data_root / image, np.clip(rgb, 0, 255).astype(np.uint8))
            page = Page(doc_id, image, w, h, SOURCE, split, lines)
            if ignore_regions and split == "train" and i == 0:
                page.ignore = [[[2.0, 2.0], [30.0, 2.0], [30.0, 20.0], [2.0, 20.0]]]
            pages.append(page)
        files[split] = data_root / "processed" / SOURCE / f"pages_{split}.jsonl.gz"
        write_pages(files[split], pages)
        out[split] = pages
    write_processed_manifest(data_root, SOURCE, files)
    return SyntheticPages(
        data_root,
        files["train"].relative_to(data_root).as_posix(),
        files["validation"].relative_to(data_root).as_posix(),
        out["train"],
        out["validation"],
    )


def tiny_det_config(
    synth: SyntheticPages, root: Path, tag: str = "a", overrides: Iterable[str] = ()
) -> RunConfig:
    """The §12 CPU smoke configuration: MobileNetV3-Small from scratch with a 16-channel neck,
    64 px crops, inference long side 160, batch 2, one crop per page, two epochs, no workers."""
    d = {
        "experiment_id": "GW-DET-999",
        "task": "det",
        "model_name": "dbnet_smoke",
        "device": "cpu",
        "paths": {"data_root": str(synth.data_root), "runs_dir": str(root / tag / "runs"),
                  "reports_dir": str(root / tag / "reports")},
        "logging": {"log_every_n_steps": 1},
        "train": {"epochs": 2, "batch_size": 2, "learning_rate": 1.0e-3, "scheduler": "cosine",
                  "warmup_steps": 0, "mixed_precision": "none"},
        "checkpoint": {"dir": str(root / tag / "ckpt")},
        "early_stopping": {"enabled": False, "monitor": "val_f1_50", "mode": "max"},
        "det": {
            "data": {"train_pages": [synth.train_pages], "val_pages": [synth.val_pages],
                     "val_subset": None, "train_eval_subset": 4, "crops_per_page": 1,
                     "crop_size": 64, "num_workers": 0, "eval_workers": 0},
            "model": {"backbone": "mobilenet_v3_small", "init": "scratch", "weights": None,
                      "weights_sha256": None, "inner_channels": 16},
            "infer": {"long_side": 160},
        },
    }  # fmt: skip
    return build_config(d, overrides)


def write_colour_png(path: Path, size: tuple[int, int] = (48, 36)) -> tuple[int, int]:
    """Distinct red, green and blue patches on a non-grey background; returns (W, H)."""
    w, h = size
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[:] = (200, 170, 40)
    rgb[2 : h // 2, 2 : w // 3] = (255, 0, 0)
    rgb[2 : h // 2, w // 3 : 2 * w // 3] = (0, 255, 0)
    rgb[h // 2 : h - 2, w // 3 : 2 * w // 3] = (0, 0, 255)
    rgb[h // 2 :, 2 * w // 3 :] = np.arange(w - 2 * w // 3, dtype=np.uint8)[None, :, None] * 7
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return w, h


def write_exif6_jpeg(path: Path, stored_size: tuple[int, int] = (64, 40)) -> tuple[int, int]:
    """A JPEG stored W x H with EXIF orientation 6 (rotate 90° clockwise to display), with an
    asymmetric pattern so a wrong rotation shows; returns the displayed (W, H) = (H, W)."""
    w, h = stored_size
    rgb = np.full((h, w, 3), 230, np.uint8)
    rgb[4:12, 4 : w // 2] = (30, 30, 30)  # a bar near the stored top-left corner
    rgb[h - 10 : h - 4, w - 12 : w - 4] = (200, 30, 30)
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=95, exif=exif.tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return h, w
