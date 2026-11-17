"""Convert the Phase 14 external benchmarks into the project's evaluation formats.

    python scripts/prepare_benchmarks.py iiit5k        # test words -> line pack
    python scripts/prepare_benchmarks.py funsd         # test forms -> page file (words -> lines)
    python scripts/prepare_benchmarks.py omnidocbench  # English pages -> page file (text blocks)
    python scripts/prepare_benchmarks.py all

Conversion is preprocessing, not evaluation: no model sees the data and nothing is scored, so the
test-access log is not touched (as for `scripts/build_eval_crops.py`). Only the test splits are
converted. Outputs are git-ignored; their counts and hashes go to
``data/manifests/processed/benchmarks.json`` (committed).

- **IIIT5K** (`raw/iiit5k`): the 3,000 test words of ``testdata.mat`` into
  ``processed/iiit5k/words_test``, each word image stored byte for byte as published. The label is
  the published ground truth, which is upper case: the benchmark is scored case-insensitively.
- **FUNSD** (`raw/funsd`): the 50 test forms into ``processed/funsd/pages_test.jsonl.gz``. FUNSD
  annotates entities (a question, an answer, a header) with their words. Each entity's words are
  grouped into lines by vertical overlap and joined left to right with single spaces; the line box
  is the union of its word boxes. Empty words and entities are dropped.
- **OmniDocBench** (`raw/omnidocbench`): the English pages (``page_attribute.language ==
  "english"``) into ``processed/omnidocbench/pages_test.jsonl.gz``. Its units are layout blocks,
  not lines: every block of a text category (`TEXT_CATEGORIES`) becomes one entry, in the
  published reading order, with ``line_id`` ``<page>/<anno_id>/<category>``; its text can span
  several printed lines. Blocks marked ``ignore``, empty text blocks and non-text blocks (figures,
  tables, equations, masks) become "don't care" regions. Entries are marked ``rec_ok = False``, so
  no recognizer crops are cut from blocks; Phase 14 scores this benchmark at page level.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from glypharis.data.linepack import LinePackWriter
from glypharis.data.pages import Page, PageLine, write_pages
from glypharis.data.text import Charset, normalize_label

MANIFEST = Path("data/manifests/processed/benchmarks.json")
TEXT_CATEGORIES = frozenset(
    {
        "text_block",
        "title",
        "header",
        "footer",
        "page_number",
        "figure_caption",
        "table_caption",
        "table_footnote",
        "figure_footnote",
        "page_footnote",
        "equation_caption",
        "reference",
        "code_txt",
        "equation_explanation",
        "list_group",
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _png_size(data: bytes) -> tuple[int, int]:
    from io import BytesIO

    from PIL import Image

    with Image.open(BytesIO(data)) as im:
        return im.width, im.height


# ------------------------------------------------------------------------------------- IIIT5K
def iiit5k_records(mat_path: Path) -> list[tuple[str, str]]:
    """(image name under the dataset folder, label) for every test word, in published order."""
    import scipy.io

    data = scipy.io.loadmat(mat_path)["testdata"][0]
    return [(str(r["ImgName"][0]), str(r["GroundTruth"][0])) for r in data]


def prepare_iiit5k(root: Path) -> dict[str, Any]:
    base = root / "raw/iiit5k/extracted/IIIT5K"
    recs = iiit5k_records(base / "testdata.mat")
    prefix = root / "processed/iiit5k/words_test"
    meta = {
        "benchmark": "iiit5k",
        "split": "test",
        "label_case": "upper (score case-insensitively)",
    }
    with LinePackWriter(prefix, meta) as w:
        for name, label in recs:
            data = (base / name).read_bytes()
            w.add(
                f"iiit5k/{name}", f"iiit5k/{name}", "test", "iiit5k", label, data, _png_size(data)
            )
        out = w.close()
    return {
        "pack": "processed/iiit5k/words_test",
        "count": out["count"],
        "bin_sha256": out["bin_sha256"],
        "source": "testdata.mat",
        "source_sha256": sha256(base / "testdata.mat"),
    }


# -------------------------------------------------------------------------------------- FUNSD
def group_lines(words: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Words of one entity grouped into printed lines: a word joins the current line when its
    vertical extent overlaps the line's by at least half of the smaller height."""
    words = sorted(
        (w for w in words if w["text"].strip()), key=lambda w: (w["box"][1] + w["box"][3]) / 2
    )
    lines: list[list[dict[str, Any]]] = []
    top = bottom = 0.0
    for w in words:
        y0, y1 = w["box"][1], w["box"][3]
        overlap = min(bottom, y1) - max(top, y0)
        if lines and overlap >= 0.5 * min(bottom - top, y1 - y0):
            lines[-1].append(w)
            top, bottom = min(top, y0), max(bottom, y1)
        else:
            lines.append([w])
            top, bottom = y0, y1
    return [sorted(ln, key=lambda w: w["box"][0]) for ln in lines]


def funsd_page(
    ann: dict[str, Any], image: str, size: tuple[int, int], doc_id: str, charset: Charset | None
) -> Page:
    page = Page(doc_id, image, size[0], size[1], "funsd", "test")
    for ent in ann["form"]:
        for j, ln in enumerate(group_lines(ent.get("words", []))):
            text = " ".join(w["text"].strip() for w in ln)
            x0 = min(w["box"][0] for w in ln)
            y0 = min(w["box"][1] for w in ln)
            x1 = max(w["box"][2] for w in ln)
            y1 = max(w["box"][3] for w in ln)
            label = normalize_label(text)
            reason = "out_of_charset" if charset is not None and charset.oov(label) else None
            page.lines.append(
                PageLine(
                    line_id=f"{doc_id}/e{ent['id']}/l{j}",
                    polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    text_original=text,
                    text=label,
                    legible=True,
                    handwritten=False,
                    vertical=False,
                    rec_ok=reason is None,
                    rec_skip=reason,
                )
            )
    return page


def prepare_funsd(root: Path, charset: Charset | None) -> dict[str, Any]:
    from PIL import Image

    base = root / "raw/funsd/extracted/dataset/testing_data"
    pages = []
    for ann_path in sorted((base / "annotations").glob("*.json")):
        stem = ann_path.stem
        img = base / "images" / f"{stem}.png"
        with Image.open(img) as im:
            size = (im.width, im.height)
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        pages.append(
            funsd_page(ann, img.relative_to(root).as_posix(), size, f"funsd-{stem}", charset)
        )
    out = root / "processed/funsd/pages_test.jsonl.gz"
    write_pages(out, pages)
    return {
        "pages": "processed/funsd/pages_test.jsonl.gz",
        "pages_sha256": sha256(out),
        "page_count": len(pages),
        "line_count": sum(len(p.lines) for p in pages),
        "rec_lines": sum(p_l.rec_ok for p in pages for p_l in p.lines),
        "line_grouping": "entity words grouped by vertical overlap >= 0.5 of the smaller height",
    }


# ------------------------------------------------------------------------------- OmniDocBench
def _poly(flat: Any) -> list[list[float]]:
    v = json.loads(flat) if isinstance(flat, str) else flat
    return [[float(v[i]), float(v[i + 1])] for i in range(0, len(v) - 1, 2)]


def _flag(x: Any) -> bool:
    return x is True or str(x).strip().lower() == "true"


def omnidocbench_page(pg: dict[str, Any], image: str) -> Page:
    info = pg["page_info"]
    stem = Path(info["image_path"]).stem
    page = Page(
        f"odb-{stem}", image, int(info["width"]), int(info["height"]), "omnidocbench", "test"
    )
    keep = []
    for det in pg["layout_dets"]:
        poly = _poly(det["poly"])
        text = str(det.get("text") or "")
        if (
            det["category_type"] in TEXT_CATEGORIES
            and not _flag(det.get("ignore"))
            and text.strip()
        ):
            order = det.get("order")
            keep.append(
                (float(order) if str(order).replace(".", "", 1).isdigit() else 1e9, det, poly, text)
            )
        else:
            page.ignore.append(poly)
    for _, det, poly, text in sorted(keep, key=lambda k: k[0]):
        page.lines.append(
            PageLine(
                line_id=f"{page.doc_id}/{det.get('anno_id', '')}/{det['category_type']}",
                polygon=poly,
                text_original=text,
                text=normalize_label(text),
                legible=True,
                handwritten=False,
                vertical=False,
                rec_ok=False,
                rec_skip=None,
            )
        )
    return page


def prepare_omnidocbench(root: Path) -> dict[str, Any]:
    base = root / "raw/omnidocbench"
    ann = json.loads((base / "OmniDocBench.json").read_text(encoding="utf-8"))
    pages = [
        omnidocbench_page(pg, f"raw/omnidocbench/images/{pg['page_info']['image_path']}")
        for pg in ann
        if pg["page_info"]["page_attribute"].get("language") == "english"
    ]
    missing = [p.image for p in pages if not (root / p.image).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} English page images are missing, e.g. {missing[0]}")
    out = root / "processed/omnidocbench/pages_test.jsonl.gz"
    write_pages(out, pages)
    return {
        "pages": "processed/omnidocbench/pages_test.jsonl.gz",
        "pages_sha256": sha256(out),
        "page_count": len(pages),
        "block_count": sum(len(p.lines) for p in pages),
        "ignored_regions": sum(len(p.ignore) for p in pages),
        "source_sha256": sha256(base / "OmniDocBench.json"),
        "unit": "layout block (text categories), not line",
        "text_categories": sorted(TEXT_CATEGORIES),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("which", choices=["iiit5k", "funsd", "omnidocbench", "all"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    root = args.data_root
    charset_path = root / "manifests/charset.json"
    charset = Charset.load(charset_path) if charset_path.exists() else None
    todo = ["iiit5k", "funsd", "omnidocbench"] if args.which == "all" else [args.which]
    manifest = (
        json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else {}
    )
    keys = {"iiit5k": "iiit5k_test", "funsd": "funsd_test", "omnidocbench": "omnidocbench_en"}
    for name in todo:
        if name == "iiit5k":
            entry = prepare_iiit5k(root)
        elif name == "funsd":
            entry = prepare_funsd(root, charset)
        else:
            entry = prepare_omnidocbench(root)
        manifest[keys[name]] = entry
        print(name, {k: v for k, v in entry.items() if "count" in k or k.endswith("lines")})
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(dict(sorted(manifest.items())), indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
