"""Phase 10 pipeline: recognizer batching, confidence, calibration, and an end-to-end smoke test on
tiny bundled models (built here from a fixed seed, so the repository carries no weight files)."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from det_fixtures import rect_polygon
from PIL import Image, ImageDraw

from glypharis.config.schema import (
    DetInferConfig,
    DetLabelConfig,
    DetModelConfig,
    DetPostConfig,
    ModelConfig,
)
from glypharis.data.det_labels import LABEL_VERSION
from glypharis.data.text import Charset
from glypharis.models.det.dbnet import build_detector
from glypharis.models.det.detector import DetectorSpec, TextLineDetector
from glypharis.models.det.runtime import TorchProbMap
from glypharis.models.rec.crnn import build_recognizer
from glypharis.pipeline import GlypharisOCR, TextRecognizer
from glypharis.pipeline.calibration import TEMPERATURES, ece, fit_temperature, reliability
from glypharis.pipeline.recognizer import line_confidence

CHARSET = Path("data/manifests/charset.json")


def tiny_recognizer(seed: int = 0) -> TextRecognizer:
    torch.manual_seed(seed)
    charset = Charset.load(CHARSET)
    cfg = ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=16, lstm_layers=1,
                      dropout=0.0)  # fmt: skip
    arch = {"model_name": "crnn", "model": dataclasses.asdict(cfg)}
    return TextRecognizer(build_recognizer(charset.num_classes, cfg), charset, 32, 512, arch=arch)


def tiny_detector(seed: int = 0) -> TextLineDetector:
    torch.manual_seed(seed)
    mcfg = DetModelConfig(backbone="mobilenet_v3_small", init="scratch", inner_channels=16)
    model = build_detector(mcfg, Path("."), load_pretrained=False)
    spec = DetectorSpec(mcfg, DetLabelConfig(), DetInferConfig(long_side=160), DetPostConfig(),
                        "inverse10", LABEL_VERSION)  # fmt: skip
    return TextLineDetector(TorchProbMap(model, torch.device("cpu")), spec)


class KernelMap:
    """A stand-in backend whose map fires on known lines (at the spec's network scale)."""

    def __init__(self, lines: list[list[list[float]]], scale: float) -> None:
        self.lines, self.scale = lines, scale

    def __call__(self, image: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        prob = np.zeros((math.ceil(h / 32) * 32, math.ceil(w / 32) * 32), np.float32)
        for poly in self.lines:
            pts = np.asarray(poly, np.float32) * self.scale
            c = pts.mean(axis=0)
            cv2.fillPoly(prob, [np.round(c + 0.6 * (pts - c)).astype(np.int32)], 0.95)
        return prob


def _page(lines: list[list[list[float]]], size: tuple[int, int] = (400, 300)) -> Image.Image:
    img = Image.new("RGB", size, (235, 235, 225))
    d = ImageDraw.Draw(img)
    for poly in lines:
        d.polygon([tuple(p) for p in poly], fill=(30, 30, 30))
    return img


def test_batched_reading_equals_one_line_at_a_time():
    rec = tiny_recognizer()
    rng = np.random.default_rng(1)
    crops = [Image.fromarray(rng.integers(0, 255, (40, int(w)), np.uint8)) for w in
             (30, 300, 75, 512, 120, 64, 900)]  # fmt: skip
    together = rec.logits(crops)
    for crop, lg in zip(crops, together, strict=True):
        alone = rec.logits([crop])[0]
        assert lg.shape == alone.shape
        assert torch.allclose(lg, alone, atol=1e-5)
    assert rec.logits([]) == []


def test_line_confidence_definition():
    logits = torch.full((4, 5), -10.0)
    logits[0, 0] = logits[1, 2] = logits[2, 0] = logits[3, 3] = 10.0  # blank, c, blank, c
    p = torch.softmax(logits, -1)
    want = math.sqrt(float(p[1, 2] * p[3, 3]))
    assert line_confidence(logits) == pytest.approx(want)
    blank = torch.full((3, 5), -10.0)
    blank[:, 0] = 2.0
    assert line_confidence(blank) == pytest.approx(float(torch.softmax(blank, -1)[:, 0].mean()))
    assert line_confidence(logits, 4.0) < line_confidence(logits)  # a higher T softens
    assert line_confidence(torch.zeros((0, 5))) == 0.0


def test_ece_and_temperature_fit():
    assert ece([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0], n_bins=2) == pytest.approx(0.1)
    assert ece([], []) == 0.0
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.95, 4000)
    correct = rng.uniform(size=4000) < true_p

    def at(t: float) -> np.ndarray:  # over-confident at T = 1, calibrated at T = 2
        logit = np.log(true_p / (1 - true_p)) * 2.0 / t
        return 1 / (1 + np.exp(-logit))

    fit = fit_temperature(at, correct)
    assert fit["temperature"] in TEMPERATURES and abs(fit["temperature"] - 2.0) <= 0.25
    assert ece(at(fit["temperature"]), correct) < ece(at(1.0), correct) / 2
    rel = reliability(at(1.0), correct)
    assert len(rel) == 15 and sum(r["lines"] for r in rel) == 4000


def test_pipeline_finds_reads_and_orders_known_lines():
    lines = [rect_polygon(200, 60, 300, 24), rect_polygon(200, 150, 300, 24),
             rect_polygon(200, 240, 300, 24)]  # fmt: skip
    det = tiny_detector()
    det = TextLineDetector(KernelMap(lines, 160 / 400), det.spec)
    ocr = GlypharisOCR(det, tiny_recognizer())
    page = ocr.predict(_page(list(reversed(lines))), temperatures=(2.0,))
    assert (page.width, page.height) == (400, 300) and len(page.lines) == 3
    ys = [np.mean([p[1] for p in ln.quad]) for ln in page.lines]
    assert ys == sorted(ys)  # reading order: top to bottom
    for ln in page.lines:
        assert 0.0 <= ln.confidence <= 1.0 and 2.0 in ln.conf_at
        assert isinstance(ln.text, str)
    assert set(page.seconds) == {"load", "detect", "recognise", "order", "total"}
    d = page.to_dict()
    assert len(d["lines"]) == 3 and set(d["lines"][0]) >= {"polygon", "quad", "text", "confidence"}


def test_smoke_tiny_bundled_models_run_end_to_end(tmp_path):
    """CI's inference smoke test: real (untrained) networks through the public entry points."""
    ocr = GlypharisOCR(tiny_detector(), tiny_recognizer())
    img = _page([rect_polygon(150, 80, 200, 20)])
    path = tmp_path / "page.png"
    img.save(path)
    for source in (path, img, np.asarray(img)):
        page = ocr.predict(source)
        assert (page.width, page.height) == (400, 300)
        assert all(isinstance(ln.text, str) for ln in page.lines)
    assert len(ocr.predict_batch([img, img])) == 2
