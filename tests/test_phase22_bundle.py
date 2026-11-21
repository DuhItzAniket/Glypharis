"""The phone candidate's bundle format (Phase 22, item 16)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from glypharis.data.download import sha256_file
from glypharis.efficiency import phone_bundle as pb
from glypharis.efficiency.candidate import BATCHING
from glypharis.errors import ModelLoadError
from glypharis.export.bundle import load_bundle, read_manifest

REAL = Path(__file__).resolve().parents[1] / "runs/release/glypharis-phone-candidate"


def make_bundle(root: Path, arm: str = "fused-single-line") -> Path:
    """A phone bundle's files and manifest, with made-up contents."""
    path = root / "bundle"
    path.mkdir()
    names = {pb.DETECTOR_FILE: b"detector", "recognizer.fused.onnx": b"rec", pb.CHARSET_FILE: b"[]"}
    for name, data in names.items():
        (path / name).write_bytes(data)
    manifest: dict[str, Any] = {
        "format": pb.PHONE_BUNDLE_FORMAT,
        "version": pb.VERSION,
        "version_label": pb.VERSION_LABEL,
        "arm": arm,
        "batching": BATCHING[arm],
        "files": {name: sha256_file(path / name) for name in names},
        "recognizer_file": "recognizer.fused.onnx",
    }
    (path / pb.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return path


def edit(path: Path, **fields: Any) -> None:
    doc = json.loads((path / pb.MANIFEST).read_text(encoding="utf-8"))
    doc.update(fields)
    (path / pb.MANIFEST).write_text(json.dumps(doc), encoding="utf-8")


def test_the_v1_loader_refuses_a_phone_bundle(tmp_path: Path) -> None:
    path = make_bundle(tmp_path)
    with pytest.raises(ModelLoadError, match="glypharis-bundle-1"):
        load_bundle(path)
    with pytest.raises(ModelLoadError, match="glypharis-bundle-1"):
        read_manifest(path)


def test_the_manifest_is_read_with_its_files_checked(tmp_path: Path) -> None:
    path = make_bundle(tmp_path)
    assert pb.read_phone_manifest(path)["arm"] == "fused-single-line"
    (path / "recognizer.fused.onnx").write_bytes(b"changed")
    with pytest.raises(pb.PhoneBundleError, match="not the file the manifest names"):
        pb.read_phone_manifest(path)


def test_a_manifest_of_another_format_or_rule_is_refused(tmp_path: Path) -> None:
    path = make_bundle(tmp_path)
    edit(path, format="glypharis-bundle-1")
    with pytest.raises(pb.PhoneBundleError, match="not a glypharis-phone-bundle-1"):
        pb.read_phone_manifest(path)
    edit(path, format=pb.PHONE_BUNDLE_FORMAT, batching=BATCHING["fused"])
    with pytest.raises(pb.PhoneBundleError, match="batching rule"):
        pb.read_phone_manifest(path)
    edit(path, batching=BATCHING["fused-single-line"], recognizer_file="other.onnx")
    with pytest.raises(pb.PhoneBundleError, match="lists"):
        pb.read_phone_manifest(path)


def test_the_version_label_says_not_v1() -> None:
    assert "not v1" in pb.VERSION_LABEL and pb.VERSION != "glypharis_v1"


@pytest.mark.skipif(not REAL.is_dir(), reason="the frozen bundle is not on this machine")
def test_the_frozen_bundle_loads_and_reads_one_line_per_run() -> None:
    from glypharis.efficiency.candidate import SingleLineRecognizer

    manifest = pb.read_phone_manifest(REAL)
    ocr = pb.load_phone_bundle(REAL, 2)
    if manifest["arm"] == "fused-single-line":
        assert isinstance(ocr.recognizer, SingleLineRecognizer)
    assert "not v1" in ocr.model_version
