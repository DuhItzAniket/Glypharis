"""Phase 17: release files, ONNX export and the model bundle, on tiny models (CI's equivalence)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from det_fixtures import rect_polygon
from test_pipeline import KernelMap, _page, tiny_detector, tiny_recognizer

from glypharis.errors import ModelLoadError
from glypharis.export.bundle import load_bundle, read_manifest, write_bundle
from glypharis.export.onnx import (
    OnnxLineModel,
    OnnxProbMap,
    export_detector_onnx,
    export_recognizer_onnx,
)
from glypharis.models.det.detector import TextLineDetector
from glypharis.pipeline import GlypharisOCR, TextRecognizer

pytest.importorskip("onnxruntime")


def test_detector_onnx_matches_pytorch(tmp_path):
    det = tiny_detector()
    path = export_detector_onnx(det.backend.model, tmp_path / "det.onnx")
    ort_map = OnnxProbMap(path)
    rng = np.random.default_rng(0)
    for h, w in ((160, 128), (96, 160), (64, 64)):
        img = rng.integers(0, 255, (h, w, 3), np.uint8)
        assert np.abs(ort_map(img) - det.backend(img)).max() < 1e-4
    with pytest.raises(ValueError):
        ort_map(rng.integers(0, 255, (100, 64, 3), np.uint8))  # not padded to 32


def test_recognizer_onnx_matches_pytorch_at_any_batch_and_width(tmp_path):
    rec = tiny_recognizer()
    path = export_recognizer_onnx(rec.model, rec.height, tmp_path / "rec.onnx")
    onnx_rec = TextRecognizer(
        OnnxLineModel(path, rec.model.stride, rec.model.num_classes),
        rec.charset,
        rec.height,
        rec.max_width,
        arch=rec.arch,
    )
    rng = np.random.default_rng(1)
    from PIL import Image

    crops = [
        Image.fromarray(rng.integers(0, 255, (40, int(w)), np.uint8))
        for w in (30, 300, 75, 512, 900)
    ]
    for a, b in zip(rec.logits(crops), onnx_rec.logits(crops), strict=True):
        assert a.shape == b.shape and torch.allclose(a, b, atol=1e-4)
    assert [r.text for r in rec.read(crops)] == [r.text for r in onnx_rec.read(crops)]


def test_recognizer_release_round_trip(tmp_path):
    rec = tiny_recognizer()
    torch.save(rec.release(), tmp_path / "rec.pt")
    back = TextRecognizer.from_release(tmp_path / "rec.pt")
    assert back.charset.characters == rec.charset.characters
    assert (back.height, back.max_width, back.temperature) == (rec.height, rec.max_width, 1.0)
    x = torch.zeros(1, 1, rec.height, 96)
    w = torch.tensor([96])
    assert torch.equal(back.model(x, w), rec.model.eval()(x, w))
    torch.save({"format": "something else"}, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        TextRecognizer.from_release(tmp_path / "bad.pt")


def test_bundle_backends_agree_and_hashes_are_checked(tmp_path):
    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, tmp_path / "det_ck.pt")
    out = write_bundle(
        tmp_path / "det_ck.pt",
        det.spec,
        rec,
        tmp_path / "bundle",
        "0.0.0-test",
        {"detector": "tiny", "recognizer": "tiny"},
    )
    manifest = read_manifest(out)
    assert set(manifest["files"]) == {
        "detector.pt",
        "detector.onnx",
        "recognizer.pt",
        "recognizer.onnx",
    }
    lines = [rect_polygon(200, 60, 300, 24), rect_polygon(200, 150, 300, 24)]
    img = _page(lines)
    pages = {}
    for backend in ("torch", "onnx"):
        ocr = load_bundle(out, backend)  # type: ignore[arg-type]
        assert isinstance(ocr, GlypharisOCR)
        # A known map, so both backends cut the same lines; the recognizers then differ only in
        # their runtime.
        ocr.detector = TextLineDetector(KernelMap(lines, 160 / 400), ocr.detector.spec)
        pages[backend] = ocr.predict(img)
    assert [ln.text for ln in pages["torch"].lines] == [ln.text for ln in pages["onnx"].lines]
    for a, b in zip(pages["torch"].lines, pages["onnx"].lines, strict=True):
        assert abs(a.confidence - b.confidence) < 1e-4
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    m["files"]["recognizer.onnx"] = "0" * 64
    (out / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="SHA-256"):
        load_bundle(out)
