"""Phase 20: failure paths and a fuzz-lite test. Bad inputs and damaged bundles must raise a
`GlypharisError` with a clear message, never anything else."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest
import torch
from det_fixtures import rect_polygon
from hardening_cases import blank_page, empty_page, fuzz_cases, unusual_pages, wrong_inputs
from PIL import Image
from test_pipeline import _page, tiny_detector, tiny_recognizer

from glypharis import __version__, version_info
from glypharis.errors import GlypharisError, ImageTooLargeError, InvalidImageError, ModelLoadError
from glypharis.export.bundle import load_bundle, write_bundle
from glypharis.pipeline import GlypharisOCR
from glypharis.pipeline.ocr import load_rgb


@pytest.fixture(scope="module")
def ocr() -> GlypharisOCR:
    return GlypharisOCR(tiny_detector(), tiny_recognizer(), max_pixels=2_000_000)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    pytest.importorskip("onnxruntime")
    root = tmp_path_factory.mktemp("bundles")
    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, root / "ck.pt")
    return write_bundle(root / "ck.pt", det.spec, rec, root / "tiny", "0.0.1-test")


def test_corrupt_truncated_and_non_image_files(ocr, tmp_path):
    good = tmp_path / "good.jpg"
    _page([rect_polygon(150, 80, 200, 20)]).save(good, quality=90)
    data = good.read_bytes()
    (tmp_path / "truncated.jpg").write_bytes(data[: len(data) // 3])
    (tmp_path / "garbage.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    (tmp_path / "text.jpg").write_text("this is not an image", encoding="utf-8")
    (tmp_path / "empty.png").write_bytes(b"")
    for name in ("truncated.jpg", "garbage.png", "text.jpg", "empty.png", "missing.png"):
        with pytest.raises(InvalidImageError) as info:
            ocr.predict(tmp_path / name)
        assert name in str(info.value) and info.value.code == "invalid_image"
    with pytest.raises(InvalidImageError):
        ocr.predict(tmp_path)  # a directory


def test_huge_images_are_refused_before_decoding(ocr, tmp_path):
    with pytest.raises(ImageTooLargeError, match="limit"):
        ocr.predict(np.zeros((2000, 1500), np.uint8))  # 3 MP over the 2 MP limit
    big = tmp_path / "big.png"
    Image.new("L", (2000, 1500)).save(big)
    with pytest.raises(ImageTooLargeError):
        ocr.predict(big)
    assert ImageTooLargeError("x").to_dict() == {"error": "image_too_large", "message": "x"}


def test_empty_blank_tiny_and_unusual_pages(ocr):
    assert ocr.predict(blank_page()).lines == []  # blank page
    with pytest.raises(InvalidImageError, match="empty"):
        ocr.predict(empty_page())
    for _, img in unusual_pages():
        res = ocr.predict(img)
        assert res.width >= 1 and isinstance(res.text, str)


@pytest.mark.parametrize("bad", [bad for _, bad in wrong_inputs()])
def test_wrong_types_and_values_say_what_is_wrong(ocr, bad):
    with pytest.raises(InvalidImageError):
        ocr.predict(bad)


def test_fuzz_lite_never_raises_anything_but_glypharis_errors(ocr, tmp_path):
    outcomes = {"ok": 0, "refused": 0}
    for _, x in fuzz_cases(tmp_path):  # 120 seeded cases (hardening_cases.FUZZ_SEED)
        try:
            ocr.predict(x)
            outcomes["ok"] += 1
        except GlypharisError:
            outcomes["refused"] += 1
    assert outcomes["ok"] + outcomes["refused"] == 120 and outcomes["refused"] > 0


def test_damaged_bundles_raise_model_load_errors(bundle, tmp_path):
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(bundle, copy)
    (copy / "recognizer.onnx").unlink()
    with pytest.raises(ModelLoadError, match="missing"):
        load_bundle(copy)
    shutil.rmtree(copy)
    shutil.copytree(bundle, copy)
    (copy / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="manifest"):
        load_bundle(copy)
    m = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    m["format"] = "something-else"
    (copy / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="not a"):
        load_bundle(copy)
    with pytest.raises(ModelLoadError):
        GlypharisOCR.from_pretrained(tmp_path / "nothing-here")


def test_cache_stats_logging_and_versions(bundle, caplog):
    with caplog.at_level(logging.INFO, logger="glypharis"):
        a = GlypharisOCR.from_pretrained(bundle)
    assert any("loaded GLYPHARIS model" in r.getMessage() for r in caplog.records)
    assert GlypharisOCR.from_pretrained(bundle) is a
    assert GlypharisOCR.from_pretrained(bundle, cache=False) is not a
    fresh = GlypharisOCR.from_pretrained(bundle, cache=False)
    fresh.predict(_page([rect_polygon(150, 80, 200, 20)]))
    with pytest.raises(InvalidImageError):
        fresh.predict(None)  # type: ignore[arg-type]
    st = fresh.stats()
    assert st["pages"] == 1 and st["failed"] == 1 and st["mean_page_seconds"] > 0
    v = version_info()
    assert v["glypharis"] == __version__ and "numpy" in v and "python" in v


def test_load_rgb_conversions():
    assert load_rgb(np.ones((4, 5), bool))[0, 0].tolist() == [255, 255, 255]
    assert load_rgb(np.full((4, 5), 0.5, np.float32))[0, 0, 0] == 128
    assert load_rgb(np.full((4, 5), 65535, np.uint16))[0, 0, 0] == 255
    assert load_rgb(np.zeros((4, 5, 4), np.uint8)).shape == (4, 5, 3)
