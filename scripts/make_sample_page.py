"""Make the sample page that ``glypharis-check`` reads, and its expected output (Phase 23, item 9).

    python scripts/make_sample_page.py render --font <ofl/opensans/OpenSans[wdth,wght].ttf>
        [--out glypharis/sample/page.png]
    python scripts/make_sample_page.py expected --model <bundle dir>
        [--page glypharis/sample/page.png] [--name glypharis-v1] [--threads 4]
        [--expect-manifest-sha256 SHA] [--out PATH]
    python scripts/make_sample_page.py registry --model <bundle dir> [--name NAME] --out PATH

``render`` draws `LINES` (the project's own text) in black on a white 1240 x 1754 grey-scale page
(A4 at 150 dpi) with Pillow only: one font at a fixed size, fixed positions, Pillow's basic layout
engine, no randomness. The shipped page was rendered with Open Sans Regular (Google Fonts, SIL OFL
1.1); only the PNG is kept, never the font. The output can differ with another Pillow or FreeType
build, which is why the PNG itself is shipped and its SHA-256 is pinned in ``expected.json``.

``expected`` reads the page with a bundle (``GlypharisOCR.from_pretrained``, the torch backend on
the CPU) and writes ``expected.json`` (`glypharis.check.expected_record`): the lines as rendered,
the predicted line texts in reading order, the joined text, the model version and the bundle's
manifest SHA-256. No timings. The shipped file comes from the v1 bundle (manifest SHA-256
``875357a6...``) with 4 threads.

``registry`` writes a registry file for ``glypharis-check --registry`` that pins a bundle's
manifest (unpublished, no archive pin): for tests and the clean-wheel job on a tiny bundle only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "glypharis" / "sample"

# The page's text: written for this page, quoted from nowhere.
LINES = (
    "Optical character recognition turns a picture of text",
    "into text that a computer can search, copy and edit.",
    "GLYPHARIS reads a page in two steps. First a detector",
    "finds each line of print and marks its outline.",
    "Then a recognizer reads the line from left to right,",
    "one character at a time, and the lines are put back",
    "in reading order. This page was rendered by the",
    "project itself so that anyone can check the engine.",
)
PAGE_SIZE = (1240, 1754)
FONT_SIZE = 38
LEFT = 130
TOP = 220
LINE_STEP = 76


def render(font_path: Path, out: Path) -> str:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(font_path), FONT_SIZE, layout_engine=ImageFont.Layout.BASIC)
    page = Image.new("L", PAGE_SIZE, 255)
    draw = ImageDraw.Draw(page)
    for i, line in enumerate(LINES):
        draw.text((LEFT, TOP + i * LINE_STEP), line, fill=0, font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out, format="PNG", optimize=True)
    return hashlib.sha256(out.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    r = sub.add_parser("render", help="draw the page")
    r.add_argument("--font", type=Path, required=True)
    r.add_argument("--out", type=Path, default=SAMPLE / "page.png")
    e = sub.add_parser("expected", help="read the page with a bundle and write expected.json")
    e.add_argument("--model", type=Path, required=True, help="bundle directory")
    e.add_argument("--page", type=Path, default=None, help="default: the sample page")
    e.add_argument("--name", default="glypharis-v1", help="the registered name it is checked as")
    e.add_argument("--threads", type=int, default=4)
    e.add_argument("--expect-manifest-sha256", default=None)
    e.add_argument("--out", type=Path, default=SAMPLE / "expected.json")
    g = sub.add_parser("registry", help="write a test registry pinning a bundle's manifest")
    g.add_argument("--model", type=Path, required=True, help="bundle directory")
    g.add_argument("--name", default=None, help="default: the bundle directory's name")
    g.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.step == "render":
        sha = render(args.font, args.out)
        print(f"{args.out}: {args.out.stat().st_size} bytes, SHA-256 {sha}")
        return 0

    from glypharis.check import expected_record, manifest_sha256, read_sample
    from glypharis.hub import ModelEntry

    if args.step == "registry":
        name = args.name or args.model.name
        entry = ModelEntry(
            name=name,
            url=f"https://example.invalid/{name}.tar.gz",  # never fetched: not published
            archive=f"{name}.tar.gz",
            bytes=None,
            archive_sha256=None,
            manifest_sha256=manifest_sha256(args.model),
            top_folder=name,
            published=False,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"models": [dataclasses.asdict(entry)]}, indent=1) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"{args.out}: test registry for {name}")
        return 0

    sha = manifest_sha256(args.model)
    if args.expect_manifest_sha256 and sha != args.expect_manifest_sha256:
        print(
            f"{args.model}: manifest SHA-256 {sha} is not {args.expect_manifest_sha256}",
            file=sys.stderr,
        )
        return 1
    page = args.page
    if page is None:
        # The repository's page, or the installed package's when the script runs from a copy.
        page = SAMPLE / "page.png"
        if not page.is_file():
            from glypharis.check import SAMPLE_PAGE

            page = SAMPLE_PAGE
    result = read_sample(args.model, page, args.threads)
    record = expected_record(result, args.name, sha, page, args.threads, LINES)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{args.out}: {len(record['lines'])} lines, model {record['model_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
