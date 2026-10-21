"""Choose GDC pages and capture conditions, deterministically.

Usage:
    python scripts/gdc_select_pages.py [--seed 11]

For every source document, pages are eligible when their text layer is usable as ground truth:
at least ``MIN_CHARS`` characters, at most ``MAX_OOV`` of them outside the charset, and at most
``MAX_FRAGMENT_LINES`` of lines with two or fewer characters (scattered maths and figure labels,
whose reading order in a text layer is unreliable; forms are exempt, their short tokens are field
numbers). From the eligible pages a fixed number per
document is drawn with the seed; for papers, page 1 (title and abstract) is always included.
Pages listed in ``data/manifests/gdc/rejected.yaml`` (with a reason, after visual checking) are
skipped. Capture conditions are then assigned with fixed quotas, shuffled with the seed.

Writes data/manifests/gdc/selection.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from glypharis.data.gdc import extract_lines
from glypharis.data.text import Charset

MIN_CHARS = 600
MAX_OOV = 0.03
MAX_FRAGMENT_LINES = 0.25
MAX_VERTICAL = 0.3  # pages mostly in rotated text (landscape tables) are left out
QUOTAS = {
    "scan_app": 0.35, "photo_flat": 0.15, "photo_angled": 0.15, "low_light": 0.10,
    "distant": 0.10, "rotated": 0.07, "curved": 0.08,
}  # fmt: skip

# key, file under data/raw/gdc_sources, group, pages to draw, always-include pages (1-based)
DOCUMENTS: list[tuple[str, str, str, int, tuple[int, ...]]] = [
    ("llama", "arxiv/2302.13971v1.pdf", "paper", 5, (1,)),
    ("blip2", "arxiv/2301.12597v3.pdf", "paper", 5, (1,)),
    ("llava15", "arxiv/2310.03744v2.pdf", "paper", 5, (1,)),
    ("smoothquant", "arxiv/2211.10438v7.pdf", "paper", 5, (1,)),
    ("dpo", "arxiv/2305.18290v3.pdf", "paper", 5, (1,)),
    ("mamba", "arxiv/2312.00752v2.pdf", "paper", 5, (1,)),
    ("mistral7b", "arxiv/2310.06825v1.pdf", "paper", 4, (1,)),
    ("cot", "arxiv/2201.11903v6.pdf", "paper", 5, (1,)),
    ("qlora", "arxiv/2305.14314v1.pdf", "paper", 5, (1,)),
    ("nasa_tm", "gov/nasa-tm-20220000420.pdf", "report", 6, ()),
    ("census_income", "gov/census-p60-282.pdf", "report", 6, ()),
    ("frb_fsr", "gov/frb-fsr-20240419.pdf", "report", 6, ()),
    ("calculus_made_easy", "books/pg-33283.pdf", "book", 5, ()),
    ("hilbert_geometry", "books/pg-17384.pdf", "book", 5, ()),
    ("science_hypothesis", "books/pg-37157.pdf", "book", 5, ()),
    ("whitehead_maths", "books/pg-41568.pdf", "book", 5, ()),
    ("hefferon_linalg", "books/hefferon-linear-algebra.pdf", "textbook", 6, ()),
    ("mit_mcs", "books/mit-mcs-2018.pdf", "textbook", 6, ()),
    ("irs_1040", "forms/irs-f1040-2024.pdf", "form", 2, ()),
    ("irs_w9", "forms/irs-fw9-2024.pdf", "form", 4, ()),
    ("irs_1040_instructions", "forms/irs-i1040gi-2024.pdf", "form", 6, ()),
]


def page_quality(pdf: Path, index: int, charset: Charset) -> dict[str, float]:
    lines = extract_lines(pdf, index, 1.0)
    text = "".join(ln.text for ln in lines)
    n = len(text)
    oov = sum(1 for ch in text if ch not in charset) / n if n else 1.0
    frag = sum(1 for ln in lines if len(ln.text) <= 2) / len(lines) if lines else 1.0
    vert = sum(1 for ln in lines if ln.vertical) / len(lines) if lines else 0.0
    return {"chars": n, "oov": oov, "fragments": frag, "vertical": vert}


def main(argv: list[str] | None = None) -> int:
    import pypdfium2 as pdfium

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    root = Path(args.data_root)
    charset = Charset.load(root / "manifests" / "charset.json")
    rej_path = root / "manifests" / "gdc" / "rejected.yaml"
    rejected: dict[tuple[str, int], str] = {}
    if rej_path.exists():
        for r in yaml.safe_load(rej_path.read_text(encoding="utf-8")) or []:
            rejected[(r["doc"], int(r["page"]))] = r["reason"]
    rng = np.random.default_rng(args.seed)

    docs: list[dict[str, Any]] = []
    stats = {"eligible": 0, "pages": 0}
    for key, rel, group, n_pages, always in DOCUMENTS:
        pdf = root / "raw" / "gdc_sources" / rel
        doc = pdfium.PdfDocument(str(pdf))
        total = len(doc)
        doc.close()
        stats["pages"] += total
        eligible = []
        for p in range(1, total + 1):
            if (key, p) in rejected:
                continue
            q = page_quality(pdf, p - 1, charset)
            if (
                q["chars"] >= MIN_CHARS
                and q["oov"] <= MAX_OOV
                and (q["fragments"] <= MAX_FRAGMENT_LINES or group == "form")
                and q["vertical"] <= MAX_VERTICAL
            ):
                eligible.append(p)
        stats["eligible"] += len(eligible)
        chosen = [p for p in always if p in eligible]
        rest = [p for p in eligible if p not in chosen]
        k = min(n_pages - len(chosen), len(rest))
        chosen += sorted(rng.choice(rest, size=k, replace=False).tolist()) if k > 0 else []
        docs.append({"key": key, "file": rel, "group": group, "pages": sorted(chosen),
                     "eligible": len(eligible), "total_pages": total})  # fmt: skip
        print(f"{key:24s} {len(eligible):4d}/{total:4d} eligible -> pages {sorted(chosen)}")

    slots = [(i, p) for i, d in enumerate(docs) for p in d["pages"]]
    n = len(slots)
    conditions: list[str] = []
    for cond, share in QUOTAS.items():
        conditions += [cond] * round(share * n)
    while len(conditions) < n:
        conditions.append("scan_app")
    conditions = conditions[:n]
    rng.shuffle(conditions)
    out_docs = []
    for i, d in enumerate(docs):
        pages = [
            {"page": p, "condition": conditions[j]} for j, (di, p) in enumerate(slots) if di == i
        ]
        out_docs.append({"key": d["key"], "file": d["file"], "group": d["group"], "pages": pages})
    out = {
        "seed": args.seed,
        "rules": {
            "min_chars": MIN_CHARS,
            "max_oov": MAX_OOV,
            "max_fragment_lines": MAX_FRAGMENT_LINES,
            "max_vertical": MAX_VERTICAL,
            "quotas": QUOTAS,
            "rejected_after_review": len(rejected),
        },
        "documents": out_docs,
    }
    path = root / "manifests" / "gdc" / "selection.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(out, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )
    print(f"{n} pages selected from {stats['eligible']} eligible of {stats['pages']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
