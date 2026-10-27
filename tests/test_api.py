"""Phase 18: the public API (`GlypharisOCR.from_pretrained`, `OCRResult`) and the CLI, on a tiny
bundle built from untrained models."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from det_fixtures import rect_polygon
from test_pipeline import _page, tiny_detector, tiny_recognizer

from glypharis import __version__
from glypharis.cli import main
from glypharis.errors import ModelLoadError
from glypharis.export.bundle import write_bundle
from glypharis.pipeline import GlypharisOCR, OCRResult

pytest.importorskip("onnxruntime")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, root / "det_ck.pt")
    return write_bundle(root / "det_ck.pt", det.spec, rec, root / "tiny-0.0.1", "0.0.1-test")


def test_from_pretrained_by_path_and_by_name(bundle, monkeypatch, tmp_path):
    ocr = GlypharisOCR.from_pretrained(bundle)
    assert ocr.model_version == "0.0.1-test"
    monkeypatch.setenv("GLYPHARIS_MODELS", str(bundle.parent))
    assert GlypharisOCR.from_pretrained("tiny-0.0.1", backend="torch").model_version == "0.0.1-test"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GLYPHARIS_MODELS", str(tmp_path / "nowhere"))
    with pytest.raises(ModelLoadError, match="looked in"):
        GlypharisOCR.from_pretrained("tiny-0.0.1")


def test_predict_accepts_paths_pil_and_arrays_and_gives_the_documented_json(bundle, tmp_path):
    ocr = GlypharisOCR.from_pretrained(bundle)
    img = _page([rect_polygon(150, 80, 200, 20)])
    path = tmp_path / "page.png"
    img.save(path)
    gray = np.asarray(img.convert("L"))
    results = [ocr.predict(x) for x in (path, str(path), img, np.asarray(img), gray)]
    for r in results:
        assert isinstance(r, OCRResult) and (r.width, r.height) == (400, 300)
    d = json.loads(results[0].to_json())
    assert set(d) == {
        "glypharis_version",
        "model_version",
        "width",
        "height",
        "text",
        "lines",
        "seconds",
    }
    assert d["glypharis_version"] == __version__ and d["model_version"] == "0.0.1-test"
    for ln in d["lines"]:
        assert set(ln) >= {"text", "confidence", "det_score", "polygon", "quad"}
        assert len(ln["quad"]) == 4
    assert d["text"] == "\n".join(x["text"] for x in d["lines"] if x["text"])
    assert len(ocr.predict_batch([img, path])) == 2


def test_cli_writes_json_and_text_and_reports_bad_files(bundle, tmp_path, capsys):
    pages = tmp_path / "in"
    (pages / "sub").mkdir(parents=True)
    img = _page([rect_polygon(150, 80, 200, 20)])
    img.save(pages / "a.png")
    img.save(pages / "sub" / "b.jpg")
    (pages / "broken.png").write_bytes(b"not an image")
    (pages / "notes.txt").write_text("ignored", encoding="utf-8")
    out = tmp_path / "results"
    code = main(
        ["--input-dir", str(pages), "--recursive", "--output", str(out), "--model", str(bundle)]
    )
    assert code == 1  # broken.png failed, the others were written
    assert "FAILED" in capsys.readouterr().err
    assert sorted(p.name for p in out.iterdir()) == ["a.json", "a.txt", "sub__b.json", "sub__b.txt"]
    assert json.loads((out / "a.json").read_text(encoding="utf-8"))["width"] == 400
    code = main(
        [
            "--image",
            str(pages / "a.png"),
            "--output",
            str(out / "one"),
            "--format",
            "txt",
            "--model",
            str(bundle),
        ]
    )
    assert code == 0 and [p.name for p in (out / "one").iterdir()] == ["a.txt"]
    assert main(["--image", str(pages / "a.png"), "--model", str(tmp_path / "missing")]) == 2
    with pytest.raises(SystemExit):
        main([])


def test_infer_script_runs_as_a_program(bundle, tmp_path):
    img = _page([rect_polygon(150, 80, 200, 20)])
    img.save(tmp_path / "p.png")
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run(
        [
            sys.executable,
            str(root / "infer.py"),
            "--image",
            str(tmp_path / "p.png"),
            "--output",
            str(tmp_path / "r"),
            "--model",
            str(bundle),
        ],
        capture_output=True,
        text=True,
        cwd=root,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "r" / "p.json").is_file() and (tmp_path / "r" / "p.txt").is_file()
