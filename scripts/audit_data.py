"""Audit a downloaded dataset: statistics, plots, quality flags and the exclusion list.

Usage:
    python scripts/audit_data.py --dataset hiertext [--workers 8] [--no-cache]

Writes (all generated, never edited by hand):
    reports/data_analysis/<dataset>/stats.json, chars.json, *.png
    reports/data_quality/<dataset>_flags.csv          error and warning flags, one row each
    reports/data_quality_report.md                     summary, decisions, duplicates, examples
    data/manifests/exclusions/<dataset>.json          versioned exclusion list (committed)
Per-image measurements are cached in data/interim/audit/ (git-ignored) keyed by file size and mtime.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import os
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from glypharis.data.audit import (
    FLAGS,
    DuplicatePair,
    Flag,
    Thresholds,
    check_lines,
    distinctive_lines,
    duplicate_flags,
    find_duplicates,
    find_text_overlaps,
    image_stats,
    line_geometry,
    summarise,
    text_overlap_flags,
)
from glypharis.data.hiertext import SPLITS, ImageAnnotation, image_path, load_split
from glypharis.utils.gitinfo import git_info
from glypharis.utils.logsetup import setup_logging
from glypharis.utils.monitor import snapshot

log = logging.getLogger("audit_data")

FLAG_COLUMNS = ["code", "severity", "decision", "split", "image_id", "line_id", "detail"]
EXCLUDING = {code for code, spec in FLAGS.items() if spec.decision.startswith("exclude")}


# ------------------------------------------------------------------------------- image pass


def _cached_image_stats(
    paths: list[Path], cache_file: Path, workers: int, long_side: int, use_cache: bool
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if use_cache and cache_file.exists():
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cache[rec["key"]] = rec
    results: dict[str, dict[str, Any]] = {}
    todo: list[tuple[str, str]] = []
    for p in paths:
        try:
            st = p.stat()
            key = f"{p.as_posix()}|{st.st_size}|{st.st_mtime_ns}|{long_side}"
        except OSError:
            key = f"{p.as_posix()}|missing"
        if key in cache:
            results[p.as_posix()] = cache[key]
        else:
            todo.append((key, p.as_posix()))
    log.info("image stats: %d cached, %d to compute", len(results), len(todo))
    if todo:
        start = time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = pool.map(image_stats, [t[1] for t in todo], [long_side] * len(todo),
                               chunksize=16)  # fmt: skip
            for n, ((key, path), rec) in enumerate(zip(todo, futures, strict=True), 1):
                rec["key"] = key
                results[path] = rec
                if n % 1000 == 0:
                    log.info("  %d / %d images (%.0fs)", n, len(todo), time.monotonic() - start)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in results.values():
                fh.write(json.dumps(rec) + "\n")
    return results


# ------------------------------------------------------------------------------------ plots


def _plots(
    out_dir: Path, per_split: dict[str, dict[str, list[float]]], chars: collections.Counter[str]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("image_aspect", "image long/short side", False),
        ("brightness", "image mean brightness (0-1)", False),
        ("contrast", "image contrast (std, 0-1)", False),
        ("blur_lapvar", "Laplacian variance at 1024 px (log10)", True),
        ("line_height", "legible line height px (log10)", True),
        ("line_aspect", "legible line width/height (log10)", True),
        ("line_angle", "line angle (degrees)", False),
        ("text_len", "characters per legible line", False),
        ("words_per_line", "words per legible line", False),
        ("chars_per_word", "characters per word", False),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    for ax, (key, title, log10) in zip(axes.ravel(), panels, strict=True):
        for split, data in per_split.items():
            v = np.asarray(data.get(key, []), dtype=np.float64)
            if log10:
                v = np.log10(v[v > 0])
            v = v[np.isfinite(v)]
            if v.size:
                lo, hi = np.percentile(v, [0.5, 99.5])
                if hi <= lo:  # constant data: widen the range so the bins have width
                    lo, hi = lo - 0.5, hi + 0.5
                counts, edges = np.histogram(v, bins=60, range=(lo, hi))
                if counts.sum():
                    ax.stairs(counts / (counts.sum() * np.diff(edges)), edges, label=split)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("HierText distributions (density; 0.5-99.5 percentile range)")
    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=80)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for split, data in per_split.items():
        w, h = np.asarray(data["image_width"]), np.asarray(data["image_height"])
        ax.scatter(w, h, s=2, alpha=0.3, label=split)
    ax.set_xlabel("width px")
    ax.set_ylabel("height px")
    ax.set_title("Image sizes")
    ax.legend()
    fig.savefig(out_dir / "image_sizes.png", dpi=80)
    plt.close(fig)

    top = chars.most_common(100)
    total = sum(chars.values())
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(20, 9))
    a1.bar(range(len(top)), [c for _, c in top])
    a1.set_xticks(range(len(top)), [repr(ch)[1:-1] for ch, _ in top], fontsize=7)
    a1.set_yscale("log")
    a1.set_title("100 most frequent characters (train, legible lines)")
    counts = np.array(sorted(chars.values(), reverse=True), dtype=np.float64)
    a2.plot(np.arange(1, counts.size + 1), np.cumsum(counts) / total)
    a2.set_xscale("log")
    a2.set_xlabel("number of distinct characters (most frequent first)")
    a2.set_ylabel("coverage of character occurrences")
    a2.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "char_frequency.png", dpi=80)
    plt.close(fig)


def _duplicate_sheet(
    path: Path, raw: Path, pairs: list[DuplicatePair], notes: list[str] | None = None
) -> None:
    """Side-by-side thumbnails of every duplicate pair, for a visual check of the hash rule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    if not pairs:
        return
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 3 * len(pairs)), squeeze=False)
    for row, pr in enumerate(pairs):
        for col, (split, image_id) in enumerate((pr.a, pr.b)):
            with Image.open(image_path(raw, split, image_id)) as img:
                img.thumbnail((400, 400))
                axes[row, col].imshow(img.convert("RGB"))
            if notes:
                kind = notes[row]
            else:
                kind = "exact" if pr.exact else f"d={pr.dhash_distance} p={pr.phash_distance}"
            axes[row, col].set_title(f"{split}/{image_id} ({kind})", fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# ----------------------------------------------------------------------------- char analysis


def _script_of(ch: str) -> str:
    """ASCII, the script of a non-ASCII letter (LATIN, CYRILLIC, ...), or a category group."""
    if ch.isascii():
        return "ASCII"
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        return unicodedata.name(ch, "UNNAMED").split()[0]
    return {
        "N": "non-ASCII digits/numerals",
        "P": "non-ASCII punctuation",
        "S": "non-ASCII symbols",
        "Z": "non-ASCII spaces",
        "M": "combining marks",
    }.get(cat[0], "other")


def char_summary(chars: collections.Counter[str]) -> dict[str, Any]:
    total = sum(chars.values())
    ordered = chars.most_common()
    cum, needed = 0, {}
    for i, (_, c) in enumerate(ordered, 1):
        cum += c
        for target in (0.99, 0.999, 0.9999):
            if target not in needed and cum / total >= target:
                needed[target] = i
    scripts: collections.Counter[str] = collections.Counter()
    for ch, c in chars.items():
        scripts[_script_of(ch)] += c
    return {
        "occurrences": total,
        "distinct": len(chars),
        "distinct_for_coverage": {str(k): v for k, v in needed.items()},
        "ascii_share": scripts["ASCII"] / total if total else 0.0,
        "by_script": dict(scripts.most_common(15)),
    }


# ------------------------------------------------------------------------- exclusion list


def build_exclusions(flags: list[Flag], th: Thresholds, dataset: str, out: Path) -> dict[str, Any]:
    images: dict[tuple[str, str], list[str]] = {}
    lines: dict[tuple[str, str], tuple[str, list[str]]] = {}
    for f in flags:
        if f.code not in EXCLUDING:
            continue
        if f.line_id:
            scope = "recognizer" if f.code == "empty_text_legible" else "all"
            key = (f.split, f.line_id)
            prev_scope, reasons = lines.get(key, (scope, []))
            lines[key] = ("all" if "all" in (prev_scope, scope) else scope, [*reasons, f.code])
        else:
            images.setdefault((f.split, f.image_id), []).append(f.code)
    body = {
        "dataset": dataset,
        "rules": th.to_dict(),
        "images": [
            {"split": s, "image_id": i, "reasons": sorted(set(r))}
            for (s, i), r in sorted(images.items())
        ],
        "lines": [
            {"split": s, "line_id": lid, "scope": sc, "reasons": sorted(set(r))}
            for (s, lid), (sc, r) in sorted(lines.items())
        ],
    }
    version = 1
    if out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        old_body = {k: old[k] for k in body if k in old}
        version = old.get("version", 1) if old_body == body else old.get("version", 0) + 1
    doc = {"version": version, "generated_by": "scripts/audit_data.py", **body}
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if not out.exists() or out.read_text(encoding="utf-8") != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8", newline="\n")
    return doc


# ------------------------------------------------------------------------------------- report


def _fmt(x: float | int, digits: int = 1) -> str:
    return f"{x:,.{digits}f}" if isinstance(x, float) else f"{x:,}"


def write_report(
    path: Path,
    dataset: str,
    th: Thresholds,
    counts: dict[str, dict[str, int]],
    stats: dict[str, Any],
    flags: list[Flag],
    excl: dict[str, Any],
    chars: dict[str, Any],
) -> None:
    g = git_info()
    by_code: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for f in flags:
        by_code[f.code][f.split] += 1
    L: list[str] = [
        "# Data quality report",
        "",
        f"Generated by `scripts/audit_data.py` on {datetime.now(UTC).date().isoformat()} "
        "from commit "
        f"`{(g.commit or 'unknown')[:7]}`. Do not edit by hand; re-run the script.",
        "",
        "Every flagged sample is listed with a decision. Nothing is dropped silently: removals are",
        f"recorded in `data/manifests/exclusions/{dataset}.json` "
        f"(version {excl['version']}), which",
        "Phase 4 applies when it builds training data. Evaluation splits lose samples only when",
        "they are unusable (corrupt, missing, wrong size) or duplicate a test image.",
        "",
        f"## {dataset}",
        "",
        "### Contents",
        "",
        "| Split | Images | Image files | Lines | Legible lines with text | Words |",
        "|---|---|---|---|---|---|",
    ]
    for s in SPLITS:
        c = counts[s]
        L.append(f"| {s} | {c['images']:,} | {c['image_files']:,} | {c['lines']:,} | "
                 f"{c['legible_lines']:,} | {c['words']:,} |")  # fmt: skip
    L += [
        "",
        "### Flags and decisions",
        "",
        "| Flag | Severity | Meaning | Decision | " + " | ".join(SPLITS) + " |",
        "|---|---|---|---|" + "---|" * len(SPLITS),
    ]
    for code, spec in FLAGS.items():
        cnt = by_code.get(code, collections.Counter())
        L.append(f"| `{code}` | {spec.severity} | {spec.meaning} | {spec.decision} | "
                 + " | ".join(f"{cnt[s]:,}" for s in SPLITS) + " |")  # fmt: skip
    n_img = collections.Counter(e["split"] for e in excl["images"])
    n_line = collections.Counter((e["split"], e["scope"]) for e in excl["lines"])
    L += [
        "",
        "### Exclusions",
        "",
        "| Split | Images excluded | Lines excluded (all uses) | "
        "Lines excluded from recognizer only |",
        "|---|---|---|---|",
    ]
    for s in SPLITS:
        L.append(f"| {s} | {n_img[s]:,} | {n_line[(s, 'all')]:,} | {n_line[(s, 'recognizer')]:,} |")
    dup = [f for f in flags if "duplicate" in f.code or "overlap" in f.code]
    L += [
        "",
        "### Duplicates and text overlap",
        "",
        "Pixel duplicates use dHash and pHash together. Text overlap catches the same sign, label,",
        "chart or map photographed differently: images in different splits that share at least "
        f"{th.text_min_shared} distinctive",
        f"lines (≥ {th.text_min_chars} characters, ≥ {th.text_min_letters} letters, found in ≤ "
        f"{th.text_max_images} images) covering ≥ {th.text_min_overlap:.0%} of the smaller image's",
        "distinctive lines. Pairs were checked by eye:",
        "`reports/data_quality/hiertext_duplicates.png` and `hiertext_text_overlap.png`",
        "(the 12 largest overlaps).",
        "",
    ]
    if dup:
        L += ["| Flag | Removed copy | Detail |", "|---|---|---|"]
        L += [f"| `{f.code}` | {f.split}/{f.image_id} | {f.detail} |" for f in dup[:200]]
        if len(dup) > 200:
            L.append(f"| … | {len(dup) - 200} more in the flags CSV | |")
    else:
        L.append("No exact or near duplicates found within or across splits.")
    L += [
        "",
        "### Statistics (median, with 5th–95th percentile)",
        "",
        "| Measure | " + " | ".join(SPLITS) + " |",
        "|---|" + "---|" * len(SPLITS),
    ]
    for key, label in [
        ("image_width", "image width (px)"), ("image_height", "image height (px)"),
        ("brightness", "brightness (0-1)"), ("contrast", "contrast (0-1)"),
        ("blur_lapvar", "Laplacian variance @1024 px"), ("line_height", "legible line height (px)"),
        ("line_aspect", "legible line width/height"), ("line_angle", "line angle (deg)"),
        ("text_len", "characters per legible line"), ("words_per_line", "words per legible line"),
        ("chars_per_word", "characters per word"), ("lines_per_image", "lines per image"),
    ]:  # fmt: skip
        cells = []
        nd = 2 if key in ("brightness", "contrast") else 1
        for s in SPLITS:
            st = stats[s].get(key, {"n": 0})
            if not st.get("n"):
                cells.append("—")
                continue
            p50, p05, p95 = (_fmt(st[k], nd) for k in ("p50", "p05", "p95"))
            cells.append(f"{p50} ({p05}–{p95})")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    cov = chars["distinct_for_coverage"]
    L += [
        "",
        "### Characters (train, legible lines)",
        "",
        f"- {chars['occurrences']:,} character occurrences, "
        f"{chars['distinct']:,} distinct characters.",
        f"- ASCII share: {chars['ascii_share']:.4f}. Distinct characters needed for 99% / 99.9% / "
        f"99.99% coverage: {cov.get('0.99')} / {cov.get('0.999')} / {cov.get('0.9999')}.",
        "- By script (occurrences): "
        + ", ".join(f"{k} {v:,}" for k, v in chars["by_script"].items())
        + ".",
        "",
        f"Plots: `reports/data_analysis/{dataset}/distributions.png`, `image_sizes.png`, "
        "`char_frequency.png`. Full numbers: `stats.json`, `chars.json`.",
        "",
        "### Examples (up to 5 per flag)",
        "",
    ]
    seen: collections.Counter[str] = collections.Counter()
    for f in flags:
        if seen[f.code] < 5:
            seen[f.code] += 1
            ref = f.line_id or f.image_id
            L.append(f"- `{f.code}` {f.split} {ref} {f.detail}".rstrip())
    L += ["", "### Rules", "", "```json", json.dumps(th.to_dict(), indent=2), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------------------- main


def audit_hiertext(root: Path, reports: Path, workers: int, use_cache: bool) -> int:
    th = Thresholds()
    raw = root / "raw" / "hiertext"
    flags: list[Flag] = []
    counts: dict[str, dict[str, int]] = {}
    per_split: dict[str, dict[str, list[float]]] = {}
    chars_train: collections.Counter[str] = collections.Counter()
    annotations: dict[str, list[ImageAnnotation]] = {}

    for split in SPLITS:
        log.info("reading %s ground truth", split)
        _, anns = load_split(raw / "gt" / f"{split}.jsonl.gz")
        annotations[split] = anns
        data: dict[str, list[float]] = collections.defaultdict(list)
        n_lines = n_leg = n_words = 0
        for ann in anns:
            flags.extend(check_lines(split, ann, th))
            data["lines_per_image"].append(len(ann.lines))
            for line in ann.lines:
                n_lines += 1
                n_words += len(line.words)
                if len(line.vertices) != 4:
                    continue
                geo = line_geometry(line.vertices, ann.width, ann.height)
                data["line_angle"].append(geo.angle_deg)
                if not line.legible or not line.text.strip():
                    continue
                n_leg += 1
                data["line_height"].append(geo.height)
                data["line_aspect"].append(geo.width / geo.height if geo.height > 0 else 0.0)
                data["text_len"].append(len(line.text))
                words = line.text.split()
                data["words_per_line"].append(len(words))
                data["chars_per_word"].extend(len(w) for w in words)
                if split == "train":
                    chars_train.update(line.text)
        counts[split] = {
            "images": len(anns),
            "lines": n_lines,
            "legible_lines": n_leg,
            "words": n_words,
        }
        per_split[split] = data
        log.info("  %s: %d images, %d lines; RSS %.0f MiB", split, len(anns), n_lines,
                 snapshot().process_rss_mib)  # fmt: skip

    # images: presence, decoding, statistics, duplicates
    paths: list[Path] = []
    owner: dict[str, tuple[str, ImageAnnotation | None]] = {}
    for split in SPLITS:
        annotated = {a.image_id: a for a in annotations[split]}
        on_disk = {p.stem for p in (raw / "images" / split).glob("*.jpg")}
        counts[split]["image_files"] = len(on_disk)
        for image_id in sorted(set(annotated) - on_disk):
            flags.append(Flag("missing_image", split, image_id))
        for image_id in sorted(on_disk - set(annotated)):
            flags.append(Flag("unannotated_image", split, image_id))
        for image_id in sorted(on_disk):
            p = image_path(raw, split, image_id)
            paths.append(p)
            owner[p.as_posix()] = (split, annotated.get(image_id))
    results = _cached_image_stats(
        paths, root / "interim" / "audit" / "hiertext_images.jsonl", workers, th.stats_long_side,
        use_cache,
    )  # fmt: skip

    dup_items = []
    for path_str, rec in results.items():
        split, ann = owner[path_str]
        image_id = Path(path_str).stem
        if not rec.get("ok"):
            flags.append(Flag("corrupt_image", split, image_id, "", rec.get("error", "")))
            continue
        data = per_split[split]
        data["image_width"].append(rec["width"])
        data["image_height"].append(rec["height"])
        data["image_aspect"].append(
            max(rec["width"], rec["height"]) / min(rec["width"], rec["height"])
        )
        for k in ("brightness", "contrast", "blur_lapvar"):
            data[k].append(rec[k])
        if rec.get("exif_orientation", 1) != 1:
            flags.append(
                Flag("exif_orientation", split, image_id, "", f"tag {rec['exif_orientation']}")
            )
        if data["image_aspect"][-1] > th.extreme_image_aspect:
            flags.append(
                Flag(
                    "extreme_image_aspect", split, image_id, "", f"{data['image_aspect'][-1]:.1f}:1"
                )
            )
        if ann is not None and (rec["width"], rec["height"]) != (ann.width, ann.height):
            detail = f"file {rec['width']}x{rec['height']}, gt {ann.width}x{ann.height}"
            flags.append(Flag("size_mismatch", split, image_id, "", detail))
        dup_items.append((split, image_id, rec["sha256"], int(rec["dhash"]), int(rec["phash"])))
    log.info("duplicate search over %d images", len(dup_items))
    pairs = find_duplicates(sorted(dup_items), th)
    flags.extend(duplicate_flags(pairs))
    texts = {
        (split, ann.image_id): distinctive_lines(ann, th)
        for split in SPLITS
        for ann in annotations[split]
    }
    text_pairs = find_text_overlaps(texts, th)
    flags.extend(text_overlap_flags(text_pairs))
    log.info("%d hash duplicate pairs, %d text-overlap pairs", len(pairs), len(text_pairs))
    (reports / "data_quality").mkdir(parents=True, exist_ok=True)
    _duplicate_sheet(reports / "data_quality" / "hiertext_duplicates.png", raw, pairs)
    _duplicate_sheet(
        reports / "data_quality" / "hiertext_text_overlap.png",
        raw,
        [DuplicatePair(t.a, t.b, False, -1, -1) for t in text_pairs[:12]],
        [f"{t.shared} shared lines ({t.overlap:.0%})" for t in text_pairs[:12]],
    )

    # outputs
    out_dir = reports / "data_analysis" / "hiertext"
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {s: {k: summarise(v) for k, v in sorted(per_split[s].items())} for s in SPLITS}
    (out_dir / "stats.json").write_text(
        json.dumps({"rules": th.to_dict(), "counts": counts, "stats": stats}, indent=1) + "\n",
        encoding="utf-8", newline="\n",
    )  # fmt: skip
    chars = char_summary(chars_train)
    (out_dir / "chars.json").write_text(
        json.dumps({"summary": chars, "counts": dict(chars_train.most_common())}, indent=0,
                   ensure_ascii=False) + "\n",
        encoding="utf-8", newline="\n",
    )  # fmt: skip
    _plots(out_dir, per_split, chars_train)

    flags.sort(key=lambda f: (f.code, f.split, f.image_id, f.line_id))
    flag_csv = reports / "data_quality" / "hiertext_flags.csv"
    flag_csv.parent.mkdir(parents=True, exist_ok=True)
    with flag_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FLAG_COLUMNS)
        writer.writeheader()
        for f in flags:
            if f.spec.severity != "info":
                writer.writerow(f.to_row())
    excl = build_exclusions(
        flags, th, "hiertext", root / "manifests" / "exclusions" / "hiertext.json"
    )
    write_report(
        reports / "data_quality_report.md", "hiertext", th, counts, stats, flags, excl, chars
    )

    by_sev = collections.Counter(f.spec.severity for f in flags)
    log.info("flags: %s; excluded images %d, lines %d (exclusion list v%d)", dict(by_sev),
             len(excl["images"]), len(excl["lines"]), excl["version"])  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a downloaded dataset.")
    parser.add_argument("--dataset", required=True, choices=["hiertext"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--reports", default="reports")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    setup_logging(Path(args.data_root) / "cache" / "logs", run_id="audit_data")
    return audit_hiertext(Path(args.data_root), Path(args.reports), args.workers, not args.no_cache)


if __name__ == "__main__":
    sys.exit(main())
