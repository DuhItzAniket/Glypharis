"""Command-line OCR: ``glypharis-ocr`` (or ``python infer.py``) (Phase 18).

    glypharis-ocr --image page.jpg [--image other.png ...] [--output results]
    glypharis-ocr --input-dir scans/ [--recursive] [--format json txt] [--model glypharis-0.1.0]

Each image gets ``<output>/<name>.json`` (`OCRResult.to_dict`) and/or ``<output>/<name>.txt`` (the
text in reading order). Images that cannot be read are reported and skipped; the exit code is 1 if
any image failed, 2 for a usage error, 0 otherwise. The model is a bundle by name or path
(`GlypharisOCR.from_pretrained`); nothing is downloaded.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})


def images_in(directory: Path, recursive: bool) -> Iterator[Path]:
    pattern = "**/*" if recursive else "*"
    for p in sorted(directory.glob(pattern)):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def _out_name(path: Path, root: Path | None) -> str:
    """Output stem: the path below ``--input-dir`` with separators as ``__``, else the file stem."""
    if root is None:
        return path.stem
    rel = path.relative_to(root).with_suffix("")
    return "__".join(rel.parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glypharis-ocr", description=__doc__.splitlines()[0])
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--format", nargs="+", choices=["json", "txt"], default=["json", "txt"])
    parser.add_argument("--model", default="glypharis-0.1.0", help="bundle name or directory")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.image and args.input_dir is None:
        parser.error("give --image (one or more) or --input-dir")
    if args.input_dir is not None and not args.input_dir.is_dir():
        parser.error(f"--input-dir {args.input_dir} is not a directory")
    jobs: list[tuple[Path, Path | None]] = [(p, None) for p in args.image]
    if args.input_dir is not None:
        jobs += [(p, args.input_dir) for p in images_in(args.input_dir, args.recursive)]
    if not jobs:
        print(f"no images found in {args.input_dir}", file=sys.stderr)
        return 1

    from glypharis.pipeline.ocr import GlypharisOCR

    try:
        ocr = GlypharisOCR.from_pretrained(args.model, args.backend, args.threads)
    except (FileNotFoundError, ValueError) as exc:
        print(f"cannot load the model: {exc}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path, root in jobs:
        try:
            result = ocr.predict(path)
        except (OSError, ValueError) as exc:  # unreadable or not an image
            failed += 1
            print(f"FAILED {path}: {exc}", file=sys.stderr)
            continue
        name = _out_name(path, root)
        if "json" in args.format:
            (args.output / f"{name}.json").write_text(result.to_json() + "\n", encoding="utf-8")
        if "txt" in args.format:
            (args.output / f"{name}.txt").write_text(result.text + "\n", encoding="utf-8")
        print(f"{path}: {len(result.lines)} lines in {result.seconds.get('total', 0.0):.2f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
