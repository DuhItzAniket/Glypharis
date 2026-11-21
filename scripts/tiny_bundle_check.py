"""Build a tiny untrained bundle with the installed package and read a blank page with it.

Used by the clean-wheel CI job (Phase 23, item 3) after installing the built wheel in a fresh
environment, and by ``tests/test_packaging.py``. It uses only the package's own modules, so it
shows that the installed code can write a bundle (``write_bundle``), load it with
``GlypharisOCR.from_pretrained`` and run a page through both backends. The models are random and
tiny (no weights are read or downloaded), so it says nothing about accuracy.

    python tiny_bundle_check.py [--out DIR] [--build-only | --bundle DIR] [--backend torch onnx]
                                [--require-installed] [--block MODULE ...]

Writing a bundle exports ONNX files, which needs the ``onnx`` package (the ``export`` extra);
reading one needs only the core dependencies. So the clean-wheel job builds with ``--build-only``
after installing the extra, then reads with ``--bundle`` and ``--block onnx ...``.
``--require-installed`` fails unless ``glypharis`` was imported from the interpreter's
site-packages (not from a checkout on ``sys.path``). ``--block`` makes the named top-level modules
fail to import, as if they were not installed. Prints a JSON summary and exits 0 when every step
passed and no blocked module was imported.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

BUNDLE_NAME = "tiny-check"
MODEL_VERSION = "0.0.0-tiny"


def _build(out: Path) -> Path:
    import dataclasses

    import torch

    from glypharis.config.schema import (
        DetInferConfig,
        DetLabelConfig,
        DetModelConfig,
        DetPostConfig,
        ModelConfig,
    )
    from glypharis.data.det_labels import LABEL_VERSION
    from glypharis.data.text import Charset
    from glypharis.export.bundle import write_bundle
    from glypharis.models.det.dbnet import build_detector
    from glypharis.models.det.detector import DetectorSpec
    from glypharis.models.rec.crnn import build_recognizer
    from glypharis.pipeline import TextRecognizer

    torch.manual_seed(0)
    charset = Charset(list(string.ascii_letters + string.digits + " .,:;!?'-"))
    rcfg = ModelConfig(
        cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=16, lstm_layers=1, dropout=0.0
    )
    arch = {"model_name": "crnn", "model": dataclasses.asdict(rcfg)}
    rec = TextRecognizer(build_recognizer(charset.num_classes, rcfg), charset, 32, 512, arch=arch)
    dcfg = DetModelConfig(backbone="mobilenet_v3_small", init="scratch", inner_channels=16)
    det = build_detector(dcfg, out, load_pretrained=False)
    spec = DetectorSpec(
        dcfg,
        DetLabelConfig(),
        DetInferConfig(long_side=160),
        DetPostConfig(),
        "inverse10",
        LABEL_VERSION,
    )
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": det.state_dict()}, out / "det_ck.pt")
    return write_bundle(out / "det_ck.pt", spec, rec, out / BUNDLE_NAME, MODEL_VERSION)


def _where() -> tuple[Path, bool]:
    import glypharis

    location = Path(glypharis.__file__).resolve().parent
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    return location, location.parent == purelib


def read(bundle: Path, backends: Sequence[str]) -> dict[str, Any]:
    import numpy as np

    import glypharis
    from glypharis import GlypharisOCR

    page = np.full((240, 320, 3), 255, np.uint8)  # a white page: nothing to find
    reads: dict[str, Any] = {}
    for backend in backends:
        ocr = GlypharisOCR.from_pretrained(bundle, backend=backend, cache=False)
        d = json.loads(ocr.predict(page).to_json())
        if (d["width"], d["height"]) != (320, 240):
            raise SystemExit(f"{backend}: page size {d['width']}x{d['height']}, expected 320x240")
        if d["glypharis_version"] != glypharis.__version__ or d["model_version"] != MODEL_VERSION:
            raise SystemExit(
                f"{backend}: the result says glypharis {d['glypharis_version']!r}, "
                f"model {d['model_version']!r}"
            )
        reads[backend] = {"lines": len(d["lines"])}
    return reads


def run(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    import glypharis

    location, installed = _where()
    if args.require_installed and not installed:
        raise SystemExit(f"glypharis was imported from {location}, not from site-packages")
    bundle = args.bundle if args.bundle is not None else _build(out)
    summary: dict[str, Any] = {
        "glypharis_version": glypharis.__version__,
        "imported_from": str(location),
        "installed": installed,
        "bundle": str(bundle),
        "bundle_files": sorted(p.name for p in bundle.iterdir()),
    }
    if not args.build_only:
        summary["reads"] = read(bundle, args.backend)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, help="folder for the tiny bundle (default: a temp dir)")
    step = parser.add_mutually_exclusive_group()
    step.add_argument("--build-only", action="store_true", help="write the bundle, read nothing")
    step.add_argument("--bundle", type=Path, help="read this bundle instead of building one")
    parser.add_argument("--backend", nargs="+", default=["torch", "onnx"],
                        choices=["torch", "onnx"])  # fmt: skip
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--block", nargs="*", default=[], metavar="MODULE")
    args = parser.parse_args(argv)
    if args.build_only and args.out is None:
        parser.error("--build-only needs --out")
    for name in args.block:
        # A None entry makes `import name` raise ImportError and `importlib.util.find_spec(name)`
        # return None, as for a module that is not installed (torch probes some this way).
        sys.modules[name] = None  # type: ignore[assignment]
    if args.out is not None or args.bundle is not None:
        summary = run(args, args.out or Path("."))
    else:
        with tempfile.TemporaryDirectory(prefix="glypharis-tiny-") as tmp:
            summary = run(args, Path(tmp))
    summary["blocked"] = sorted(args.block)
    summary["blocked_imported"] = sorted(
        n
        for n in args.block
        if any(m.split(".", 1)[0] == n and mod is not None for m, mod in sys.modules.items())
    )
    summary["passed"] = not summary["blocked_imported"]
    print(json.dumps(summary, indent=1))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
