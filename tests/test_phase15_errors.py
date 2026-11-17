"""Phase 15's error analysis (scripts/phase15_errors.py) on a made-up repository in tmp_path.

The Phase 14 hand-off is written by Phase 14's own code (`phase14_score.Scorer` scoring and
`Scorer.handoff`) from made-up pages, packs and prediction files whose records carry
`scripts/ext_eval.py`'s fields, so every field the analysis reads is named as the writer names it.
No prepared benchmark, real prediction file, real results.json or real access log is read.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import phase14_score as ps
import phase15_errors as pe
import pytest
import run_baselines
import yaml
from ext_eval import line_geometry
from PIL import Image

from glypharis.data.crops import encode_jpeg
from glypharis.data.linepack import LinePackWriter
from glypharis.data.pages import Page, PageLine, write_pages
from glypharis.data.text import Charset
from glypharis.eval import errors as er
from glypharis.eval.sets import EVAL_SETS
from glypharis.eval.slices import CLEAN_SLICE

LOG_HEADER = "| Date | Benchmark | ID | Checkpoint | Phase | Reason |\n|---|---|---|---|---|---|\n"
PLACEHOLDER = "| — | — | — | — | — | No test access yet. |\n"
MARKER = "Zqxjvmarker"  # stands for GDC-scan's text; must never reach an output
FUNSD_TEXT = "Qfunsdword"  # FUNSD and IIIT5K text: counts only, never a line in an output
IIIT_TEXT = "QIIITWORD"
HEFFERON = "Hefferonq"  # a CC BY-SA 3.0 GDC source: its lines are never quoted
CHARSET = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;-%=+()")
GDC_TEXT = [
    "The quick brown fox jumps",
    "Table 3: results 42.5%",
    "INTRODUCTION AND SCOPE",
    "characteristically long words here",
    "naïve café prices rose",
    "alpha beta gamma delta",
    "x = y + 2 (approx)",
    "Section 4.1 Methods used",
]
HIER_TEXT = ["Open daily", "SALE 50%", "Main Street", "welcome home", "No parking", "Exit"]
# GDC: seven capture conditions, five pages each (so condition rows carry cluster CIs), every
# page from one source document (selection keys), each condition from five documents.
GDC_CONDITIONS = ["scan_app", "photo_flat", "photo_angled", "low_light", "distant", "rotated",
                  "curved"]  # fmt: skip
DOCS = [
    ("llama", "arxiv/2302.13971v1.pdf", "paper"),
    ("blip2", "arxiv/2301.12597v3.pdf", "paper"),
    ("irs_w9", "forms/irs-fw9-2024.pdf", "form"),
    ("hefferon_linalg", "books/hefferon-linear-algebra.pdf", "textbook"),
    ("nasa_tm", "gov/nasa-tm-20220000420.pdf", "report"),
    ("calculus_made_easy", "books/pg-33283.pdf", "book"),
]
GDC_PAGES = 5 * len(GDC_CONDITIONS)
CONDITIONS = [GDC_CONDITIONS[i % len(GDC_CONDITIONS)] for i in range(GDC_PAGES)]
SOURCES = [DOCS[(i % len(GDC_CONDITIONS) + i // len(GDC_CONDITIONS)) % len(DOCS)][0]
           for i in range(GDC_PAGES)]  # fmt: skip
VAL_PAGES = 5
P5_REC = {"gdc_test": ("GW-BASE-001", "GW-BASE-002"), "hiertext_test": ("GW-BASE-002",)}
ODB = ("GW-BASE-002", "GW-BASE-005")


def _h4_sizes() -> dict[str, tuple[int, int]]:
    """The fixture's H4 slice sizes (pages, documents), as `phase15_errors.H4_SIZES` states the
    real ones."""

    def size(keep: Any) -> tuple[int, int]:
        idx = [i for i in range(GDC_PAGES) if keep(CONDITIONS[i])]
        return len(idx), len({SOURCES[i] for i in idx})

    return {
        "clean": size(lambda c: c in CLEAN_SLICE),
        "the other five": size(lambda c: c not in CLEAN_SLICE),
        "the other five without rotated": size(lambda c: c not in CLEAN_SLICE and c != "rotated"),
    }


def _h(*parts: Any) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest(), 16)


def _box(x0: float, y0: float, x1: float, y1: float, angle: float = 0.0) -> list[list[float]]:
    pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if not angle:
        return pts
    cx, cy, a = (x0 + x1) / 2, (y0 + y1) / 2, math.radians(angle)
    return [
        [cx + (x - cx) * math.cos(a) - (y - cy) * math.sin(a),
         cy + (x - cx) * math.sin(a) + (y - cy) * math.cos(a)]
        for x, y in pts
    ]  # fmt: skip


def _page(doc_id: str, image: str, texts: list[str], source: str, split: str = "test",
          angle: float = 0.0, line_id: str = "{doc}/l{i}") -> Page:  # fmt: skip
    lines = [
        PageLine(line_id.format(doc=doc_id, i=i), _box(0, 30 * i, 120, 30 * i + 20, angle), t, t,
                 True, False, False, True)
        for i, t in enumerate(texts)
    ]  # fmt: skip
    return Page(doc_id, image, 200, 260, source, split, lines)


def _gdc_texts(i: int) -> list[str]:
    if SOURCES[i] == "hefferon_linalg":
        return [f"{HEFFERON} row {j} of matrix {i}" for j in range(6)]
    return [GDC_TEXT[(i + j) % len(GDC_TEXT)] for j in range(6)]


def corrupt(text: str, key: str, salt: str) -> str:
    """A deterministic reading with the error kinds the tables count."""
    h = _h(salt, key) % 7
    if h == 0:
        return text
    if h == 1:
        return text[:-1]  # deletion
    if h == 2:
        return text.replace("e", "c", 1)  # substitution
    if h == 3:
        return text + "l"  # insertion
    if h == 4:
        return text.replace(" ", "", 1)  # merged words
    if h == 5:
        return text[:2] + " " + text[2:]  # split word
    return text.replace("o", "0", 1).replace("a", "o", 1)


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), "utf-8")


def _meta(path: Path, **fields: Any) -> None:
    path.with_suffix(".meta.json").write_text(json.dumps(fields), encoding="utf-8")


def _pack(prefix: Path, rows: list[tuple[str, str, str]], split: str, source: str) -> None:
    with LinePackWriter(prefix) as w:
        for i, (key, doc_id, label) in enumerate(rows):
            crop = encode_jpeg(Image.new("L", (40, 16), 60 + (i * 7) % 180), 95)
            w.add(key, doc_id, split, source, label, crop, (40, 16))


def _crop_rows(pages: list[Page]) -> list[tuple[str, str, str]]:
    return [(ln.line_id, p.doc_id, ln.text) for p in pages for ln in p.lines]


def rec_records(crops: list[tuple[str, str, str]], polygons: dict[str, Any], salt: str,
                glypharis: bool = True) -> list[dict[str, Any]]:  # fmt: skip
    """`ext_eval._rec_records`'s fields (Phase 5 baseline files: key, ref, pred only)."""
    rows = []
    for key, _, label in crops:
        row: dict[str, Any] = {"key": key, "ref": label, "pred": corrupt(label, key, salt)}
        if glypharis:
            h = _h("attr", key)
            height, angle = line_geometry(polygons[key]) if key in polygons else (None, None)
            row.update(conf_t1=0.5 + (h % 50) / 100, conf_t=0.4 + (h % 60) / 100, time_ms=1.5,
                       crop_height_px=64, blur=round(5 + h % 400 / 3, 4),
                       contrast=round(20 + h % 70, 4), line_height_px=height,
                       rotation_deg=angle, seg=0)  # fmt: skip
        rows.append(row)
    return rows


def e2e_records(pages: list[Page], salt: str, glypharis: bool = True,
                oracle: bool = False) -> list[dict[str, Any]]:  # fmt: skip
    """Pipeline (or oracle) records as `ext_eval._page_records` writes them: matched, shrunk
    (IoU 0.75), missed and poorly placed lines, spurious lines on some pages, one swapped pair."""
    rows = []
    for pi, p in enumerate(pages):
        lines = []
        for i, ln in enumerate(p.lines):
            poly = ln.polygon
            if not oracle:
                if i == 2 and pi % 2 == 0:
                    continue  # missed
                if i == 1:
                    poly = _box(0, 30 * i, 90, 30 * i + 20)  # IoU 0.75
                if i == 3 and pi % 3 == 1:
                    poly = _box(60, 30 * i, 180, 30 * i + 20)  # IoU 1/3: unmatched, spurious
            out: dict[str, Any] = {"polygon": poly, "text": corrupt(ln.text, ln.line_id, salt),
                                   "confidence": 0.8}  # fmt: skip
            if glypharis:
                out.update(quad=poly, conf_t1=0.9, det_score=None if oracle else 0.95)
            if oracle:
                out["line_id"] = ln.line_id
            lines.append(out)
        if not oracle and pi == 1 and len(lines) > 1:
            lines[0], lines[1] = lines[1], lines[0]  # out of reading order
        if not oracle and pi % 3 == 0:
            lines.append({"polygon": _box(10, 235, 60, 255), "text": "zz", "confidence": 0.3,
                          "conf_t1": 0.6, "det_score": 0.5,
                          "quad": _box(10, 235, 60, 255)})  # fmt: skip
        rows.append({"key": p.doc_id, "lines": lines, "time_ms": 3.0})
    return rows


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Every input of the analysis, the hand-off written by Phase 14's code, in a git work tree
    whose .gitignore covers the gallery folder."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("work/\n", encoding="utf-8")
    data = tmp_path / "data"
    Charset(CHARSET).save(data / "manifests/charset.json")
    pages: dict[str, list[Page]] = {
        "gdc_test": [
            _page(f"gdc{i}", f"raw/gdc/captures/{i:03d}.jpg", _gdc_texts(i), "gdc",
                  angle=8.0 if CONDITIONS[i] == "rotated" else 0.0)
            for i in range(GDC_PAGES)
        ],
        "gdc_scan_test": [
            _page(f"scan{i}", f"raw/gdc_scan/p{i}.jpg",
                  [f"{MARKER} one", f"two {MARKER}", f"{MARKER} été", f"{MARKER}"], "gdc_scan")
            for i in range(2)
        ],
        "funsd_test": [
            _page(f"funsd{i}", f"raw/funsd/{i}.png", [f"{FUNSD_TEXT} {j}: total" for j in range(4)],
                  "funsd")
            for i in range(2)
        ],
        "hiertext_test": [
            _page(f"ht{i}", f"raw/hiertext/{i}.jpg", HIER_TEXT, "hiertext") for i in range(3)
        ],
        "omnidocbench_test": [
            _page(f"odb-odb_{i}", f"raw/omnidocbench/images/odb_{i}.png", HIER_TEXT[i:i + 3],
                  "omnidocbench")
            for i in range(3)
        ],
        "hiertext_val": [
            _page(f"val{i:02x}", f"raw/hiertext/v{i}.jpg", [*HIER_TEXT, "a", "b"], "hiertext",
                  split="validation", line_id="{doc}/p0/l{i}")
            for i in range(VAL_PAGES)
        ],
    }  # fmt: skip
    crops: dict[str, list[tuple[str, str, str]]] = {}
    polygons: dict[str, Any] = {}
    for name, ps_ in pages.items():
        spec = EVAL_SETS[name]
        write_pages(data / spec["pages"], ps_)
        polygons.update({ln.line_id: ln.polygon for p in ps_ for ln in p.lines})
        if spec["pack"]:
            crops[name] = _crop_rows(ps_)
            _pack(data / spec["pack"], crops[name], spec["split"], spec["benchmark"])
    crops["iiit5k_test"] = [
        (f"iiit5k/{i}", "iiit5k", w)
        for i, w in enumerate((IIIT_TEXT, "WORLD", "A-1", "OK", "HELLO", "PARK"))
    ]
    _pack(data / EVAL_SETS["iiit5k_test"]["pack"], crops["iiit5k_test"], "test", "iiit5k")
    # The validation file has one line whose prediction is a long label fragment.
    val_rows = rec_records(crops["hiertext_val"], polygons, "val")
    val_rows[-1]["pred"] = "Main Street open"

    # GLYPHARIS prediction files, with the runner's metadata.
    p14 = data / "processed/phase14"
    for exp, (task, sets) in ps.GLYPHARIS_RUNS.items():
        if exp == "GW-E2E-006":
            continue
        for s in sets:
            path = p14 / exp / f"{s}.jsonl"
            if task == "rec":
                rows = val_rows if exp == "GW-REC-038" else rec_records(crops[s], polygons, exp)
            else:
                rows = e2e_records(pages[s], exp, oracle=task == "oracle")
            _jsonl(path, rows)
            _meta(path, experiment=exp, eval_set=s, expected=len(rows),
                  settings={"temperature": 2.35}, runs=[])  # fmt: skip
    # Phase 5 baseline files (recognition on GDC and HierText test, GDC end to end).
    for exp in sorted({e for v in P5_REC.values() for e in v}):
        system = pe.SYSTEM_OF[exp]
        for s, exps in P5_REC.items():
            if exp in exps:
                path = run_baselines.output_path(data, exp, system, "rec", s)
                _jsonl(path, rec_records(crops[s], polygons, system, glypharis=False))
                _meta(path, system=system, task="rec", eval_set=s, experiment=exp)
    p5_e2e = run_baselines.output_path(data, "GW-BASE-002", "ppocr_mobile", "e2e", "gdc_test")
    _jsonl(p5_e2e, e2e_records(pages["gdc_test"], "ppocr_mobile", glypharis=False))
    _meta(p5_e2e, system="ppocr_mobile", task="e2e", eval_set="gdc_test", experiment="GW-BASE-002")
    # Phase 14 baselines: OmniDocBench end to end (F26) and GDC-scan end to end.
    p14_base = []
    for exp, s in [*((e, "omnidocbench_test") for e in ODB), ("GW-BASE-002", "gdc_scan_test")]:
        system = pe.SYSTEM_OF[exp]
        path = run_baselines.output_path(data, exp, system, "e2e", s, phase=14)
        _jsonl(path, e2e_records(pages[s], system, glypharis=False))
        _meta(path, system=system, task="e2e", eval_set=s, experiment=exp, phase=14)
        p14_base.append(ps.Job(path, exp, system, "e2e", s, 14))

    # Phase 14's scoring and hand-off code writes the hand-off; its entries make results.json.
    log = tmp_path / "access.md"
    log.write_text(LOG_HEADER + PLACEHOLDER, encoding="utf-8")
    images = [p.image for p in pages["gdc_test"]]
    conditions = dict(zip(images, CONDITIONS, strict=True))
    sources = dict(zip(images, SOURCES, strict=True))
    sc = ps.Scorer(data, log)
    jobs = [j for j in ps.glypharis_jobs(data) if j.experiment != "GW-E2E-006"]
    p5_jobs = [ps.Job(p5_e2e, "GW-BASE-002", "ppocr_mobile", "e2e", "gdc_test", 5)]
    sc.score_phase14([*jobs, *p5_jobs, *p14_base])
    written = sc.handoff(data / ps.HANDOFF, conditions, sources)
    results = {
        "predictions": [s.entry for s in sc.scored.values()],
        "gdc_conditions": sc.gdc_condition_tables(conditions),
    }
    results_path = tmp_path / "reports/external/results.json"
    results_path.parent.mkdir(parents=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    recorded = [*written, p14 / "GW-REC-038/hiertext_val.jsonl", *(j.path for j in p14_base[:2])]
    ps.record_outputs(data, [ps.file_entry(p, data) for p in recorded])
    (data / ps.PHASE5_HASHES).write_text(
        json.dumps({"files": [ps.file_entry(p, data) for p in ps._phase5_files(data)]}),
        encoding="utf-8",
    )
    # GDC manifests: the selection's documents and groups, and the sources' URLs.
    sel = {"documents": [{"key": k, "file": f, "group": g, "pages": []} for k, f, g in DOCS]}
    (data / "manifests/gdc").mkdir(parents=True)
    (data / "manifests/gdc/selection.yaml").write_text(yaml.safe_dump(sel), encoding="utf-8")
    files = {
        f"src_{k}": {"path": f"raw/gdc_sources/{f}", "url": f"https://example.org/{k}.pdf"}
        for k, f, _ in DOCS
    }
    (data / "manifests/gdc_sources.json").write_text(json.dumps({"files": files}), "utf-8")
    monkeypatch.setattr(pe, "PHASE5_REC", P5_REC)
    monkeypatch.setattr(pe, "ODB_BASELINES", ODB)
    monkeypatch.setattr(pe, "H4_SIZES", _h4_sizes())
    out = tmp_path / "reports/error_cases/phase15"
    conf = tmp_path / "reports/confusion_analysis/GW-REC-040"
    work = tmp_path / "work"
    args = ["--data", str(data), "--out", str(out), "--confusion-out", str(conf), "--work",
            str(work), "--results", str(results_path)]  # fmt: skip
    return {"data": data, "out": out, "conf": conf, "work": work, "results": results_path,
            "args": args, "tmp": tmp_path, "pages": pages, "crops": crops}  # fmt: skip


def _handoff(tree: dict[str, Any], name: str) -> Path:
    return tree["data"] / ps.HANDOFF / name


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text("utf-8").splitlines() if x.strip()]


def _rerecord(tree: dict[str, Any], *paths: Path) -> None:
    """Record changed hand-off files again, as if Phase 14 had written them so."""
    data = tree["data"]
    doc = json.loads((data / ps.OUTPUTS).read_text(encoding="utf-8"))
    names = {ps.data_rel(p, data) for p in paths}
    doc["outputs"] = [e for e in doc["outputs"] if e["file"] not in names]
    (data / ps.OUTPUTS).write_text(json.dumps(doc), encoding="utf-8")
    ps.record_outputs(data, [ps.file_entry(p, data) for p in paths])


def _outputs(tree: dict[str, Any]) -> list[Path]:
    return [
        p for d in (tree["out"], tree["conf"], tree["work"]) for p in d.rglob("*") if p.is_file()
    ]


def _nothing_written(tree: dict[str, Any]) -> None:
    for d in (tree["out"], tree["conf"], tree["work"]):
        assert not d.exists(), d
        assert not list(d.parent.glob(f".{d.name}.*")), d  # no staging folder left behind


def _synth(data: Path, **meta: Any) -> Path:
    """A GW-REC-041 file as `scripts/phase15_glyphsynth.py` writes it, with its .meta.json."""
    rows = []
    for i in range(40):
        ref = HIER_TEXT[i % len(HIER_TEXT)] + ("é" if i % 10 == 0 else "")
        rows.append({"key": f"glyphsynth-val-{i:07d}", "ref": ref,
                     "pred": corrupt(ref, str(i), "gs"), "conf_t1": 0.9, "conf_t": 0.8,
                     "crop_height_px": 64, "blur": 10.0, "contrast": 30.0,
                     "family": ("inter", "cousine")[i % 2],
                     "font_file": ("sans/ofl/inter/Inter.ttf", "mono/apache/cousine/C.ttf")[i % 2],
                     "size_px": 20.0, "italic": i % 3 == 0, "weight": None, "inverted": i == 4,
                     "neighbours": i % 2 == 0, "underline": False,
                     "capture": ["ink_spread"] if i % 4 == 0 else ["gaussian_noise"],
                     "category": ("sans", "mono")[i % 2], "held_out": True})  # fmt: skip
    path = data / pe.SYNTH_FILE
    _jsonl(path, rows)
    doc = {
        **pe.SYNTH_META,
        "output_sha256": ps.sha256_file(path),
        "count": len(rows),
        "limit": None,
        **meta,
    }
    path.with_name("glyphsynth_val.meta.json").write_text(json.dumps(doc), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------- end to end


def test_the_analysis_runs_end_to_end_and_writes_every_file(tree: dict[str, Any]) -> None:
    _synth(tree["data"])
    assert pe.main(tree["args"]) == 0
    out, conf, work = tree["out"], tree["conf"], tree["work"]
    for name in ("results.json", "README.md", "recognition.md", "pipeline.md", "oracle_share.md",
                 "examples.md", "glyphsynth.md", "class_matrix_gdc_test.png",
                 "class_matrix_hiertext_test.png"):  # fmt: skip
        assert (out / name).is_file(), name
    for s in ("hiertext_test", "gdc_test", "funsd_test", "iiit5k_test", "hiertext_val"):
        for kind in ("characters", "words"):
            for ext in ("md", "json"):
                assert (conf / f"{kind}_{s}.{ext}").is_file(), (kind, s, ext)
    assert not list(conf.glob("*gdc_scan*"))
    for p in [*out.glob("*.md"), *conf.glob("*.md")]:
        text = p.read_text(encoding="utf-8")
        assert text.startswith(pe.NOTE), p.name
        assert str(tree["tmp"]) not in text and tree["tmp"].as_posix() not in text, p.name
    assert not [p for p in tree["tmp"].rglob(".*tmp-*")]  # no staging folder left
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))

    # Inputs: every Phase 14 and Phase 5 file, each crop and the manifests, with time and hash.
    files = {e["file"] for e in res["inputs"]}
    assert "data/processed/phase14/handoff/rec_lines_gdc_scan_test.jsonl" in files
    assert "data/processed/phase14/GW-REC-038/hiertext_val.jsonl" in files
    assert "data/processed/baselines/GW-BASE-001/tesseract_rec_gdc_test.jsonl" in files
    assert "data/processed/phase15/GW-REC-041/glyphsynth_val.jsonl" in files
    assert all(e["checked_utc"] for e in res["inputs"])
    index = _rows(_handoff(tree, "gdc_crops/index.jsonl"))
    crop_reads = [e for e in res["inputs"] if e.get("crop")]
    first = {e["file"]: e["sha256"] for e in crop_reads[: len(index)]}
    assert len(first) == len(index) == res["crops_verified"]
    for row in index:
        assert first[f"data/processed/phase14/handoff/gdc_crops/{row['file']}"] == row["sha256"]
    assert len(crop_reads) == len(index) + res["galleries"]["crops"]  # drawn crops hashed again
    assert res["consistency"]["result"] == "equal"
    # The edges come from validation; the README prints them with counts and the intervals.
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert set(res["edges"]) == set(pe.EDGE_FIELDS)
    for v in res["edges"].values():
        assert sum(v["validation_lines"]) + v["missing"] == 8 * VAL_PAGES
    # Every interval computed is printed, and the README counts them as printed.
    assert res["intervals_printed"] == res["intervals_computed"] > 50
    assert f"Intervals printed: {res['intervals_printed']:,}" in readme
    assert "reading-order rows" in readme and "OmniDocBench" in readme
    assert "omnidocbench_test" not in res["pipeline"]

    # Recognition: operations add up, categories cover every wrong line, shares sum to 1.
    gdc = res["recognition"]["gdc_test"]["tracks"]["strict"]
    tot = gdc["characters"]["totals"]
    assert tot["sub"] + tot["ins"] + tot["del"] == tot["errors"] == gdc["totals"][0]["char_errors"]
    cats = gdc["categories"]
    assert sum(r["lines"] for r in cats["rows"]) == cats["wrong_lines"]
    assert math.isclose(sum(r["share_errors"] for r in cats["rows"]), 1.0)
    assert {r["category"] for r in cats["rows"] if r["lines"]} >= {"out_of_charset"}
    assert {"condition", "rotation", "rotation_without_rotated_pages", "digit", "oov"} <= set(
        gdc["slices"]
    )
    # Condition rows: source-document CIs (cited) beside line CIs, no page CI.
    cond = {r["label"]: r for r in gdc["slices"]["condition"]}
    assert set(cond) == set(GDC_CONDITIONS)
    scan_docs = {SOURCES[i] for i in range(GDC_PAGES) if CONDITIONS[i] == "scan_app"}
    assert cond["scan_app"]["clusters"] == len(scan_docs) and cond["scan_app"]["pages"] == 5
    for r in cond.values():
        assert r["ci_clusters"]["unit"] == "documents" and r["ci_lines"] is not None
        assert r["ci_pages"] is None
    rec_md = (out / "recognition.md").read_text(encoding="utf-8")
    assert pe.pct(cond["scan_app"]["ci_clusters"]) in rec_md
    assert pe.pct(cond["scan_app"]["ci_lines"]) in rec_md
    # Shares inside subsets are shares of the whole set's errors.
    total = gdc["totals"][0]["char_errors"]
    for r in gdc["slices"]["rotation_without_rotated_pages"]:
        assert math.isclose(r["share"], r["char_errors"] / total)
    assert sum(r["share"] for r in gdc["slices"]["rotation_without_rotated_pages"]) < 1
    docs = gdc["documents"]
    assert docs["groups_in_scan_app"] and docs["conditions_in_paper"]
    for key in ("groups_in_scan_app", "conditions_in_paper"):
        for r in docs[key]:
            assert math.isclose(r["share"], r["char_errors"] / total)
            assert r["ci_lines"] is r["ci_pages"] is r["ci_clusters"] is None
    for rows in gdc["slices"].values():
        if rows and "bin" in rows[0]:
            assert all(isinstance(r["label"], str) for r in rows)
    # Line length is the strict length on every track.
    norm = res["recognition"]["gdc_test"]["tracks"]["normalised"]
    assert [r["lines"] for r in norm["slices"]["length"]] == [
        r["lines"] for r in gdc["slices"]["length"]
    ]
    matrix = gdc["characters"]["class_matrix"]
    for c in er.CLASSES:
        row = sum(matrix["rates"][c].values())
        assert row == 0 or math.isclose(row, 1.0)
        assert matrix["correct_rates"][c] <= matrix["rates"][c][c] + 1e-12
    # The whole GDC set carries its CIs; the normalised track values only.
    assert gdc["totals"][0]["ci_pages"] is not None and gdc["totals"][0]["ci_lines"] is not None
    assert norm["totals"][0]["ci_pages"] is None and norm["totals"][0]["ci_lines"] is None
    assert "Charset ceiling %" in rec_md
    iiit = res["recognition"]["iiit5k_test"]
    assert iiit["primary"] == "alnum" and "line_height_px" not in iiit["tracks"]["alnum"]["slices"]
    scan = res["recognition"]["gdc_scan_test"]["tracks"]["strict"]
    assert "characters" not in scan and "digit" not in scan["slices"] and "blur" in scan["slices"]
    assert "categories" not in scan
    assert all("key" not in r for r in scan["worst_lines"])
    # Validation: pages from the line ids, page CIs.
    val = res["validation"]
    assert val["has_pages"]
    vt = val["tracks"]["strict"]["totals"][0]
    assert vt["pages"] == VAL_PAGES and vt["ci_pages"]["unit"] == "pages"
    frag = val["tracks"]["strict"]["label_fragments"]
    assert frag["lines"] >= 1 and frag["insertions"] > 0
    assert "top" not in res["recognition"]["funsd_test"]["tracks"]["strict"]["words"]
    base = res["baselines"]["gdc_test"]
    assert len(base) == 3 * 2  # GLYPHARIS and two baselines, two tracks
    assert all(set(x["correct_rates"]) == set(er.CLASSES) for x in base)
    assert "lower correct %" in rec_md

    # Examples: the registered draws (type k counted from 1), each crop copied into the gallery.
    ex = res["examples"]
    assert len(ex["types"]) <= pe.CONFUSION_TYPES and ex["types"]
    assert [t["rank"] for t in ex["types"]] == list(range(1, len(ex["types"]) + 1))
    assert all(t["seed"] == [pe.EXAMPLE_SEED, t["rank"]] for t in ex["types"])
    assert all(len(t["drawn"]) <= pe.PER_TYPE for t in ex["types"])
    assert len(ex["correct"]) == min(pe.CORRECT_DRAW, ex["correct_crops"])
    gallery = json.loads((work / "galleries/index.json").read_text(encoding="utf-8"))
    assert len(gallery) == res["galleries"]["crops"] > 0
    for g in gallery:
        raw = (work / "galleries" / g["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == g["sha256"]
    examples_md = (out / "examples.md").read_text(encoding="utf-8")
    assert "https://example.org/llama.pdf" in examples_md and "CC BY 4.0" in examples_md
    assert "public domain" in examples_md
    assert all("title" not in v for v in ex["sources"].values())
    assert "default_rng([20261115, k])" in examples_md and "counted from 1" in examples_md
    assert res["licences"]["hiertext"]["attribution"] == pe.HIERTEXT_ATTRIBUTION

    # Pipeline: the budget adds up, per page and per set.
    for s in pe.LINE_PAGE_SETS:
        p = res["pipeline"][s]
        for b in p["budget"].values():
            assert b["missed"] + b["matched"] + b["spurious"] == b["bag_errors"]
            assert sum(b["unmatched"].values()) == b["missed"]
        assert p["budget"]["normalised"]["shares"]["missed"]["ci_pages"] is None
        assert set(p["ceiling"]) == set(pe.TRACKS)
        bag = p["budget"]["strict"]["bag_errors"]
        for r in p["cropping"]["strict"]:
            assert math.isclose(r["share"], r["char_errors"] / bag)
    g = res["pipeline"]["gdc_test"]
    assert g["unmatched_lines"]["[0.3, 0.5)"] > 0 and g["budget"]["strict"]["spurious"] > 0
    by_cond = g["budget_by_condition"]["strict"]
    assert set(by_cond) == set(GDC_CONDITIONS)
    for b in by_cond.values():  # condition rows: document CIs only
        assert b["shares"]["missed"]["ci_clusters"] is not None
        assert b["shares"]["missed"]["ci_pages"] is None
    assert g["inversions"]["inverted"] >= 1 and g["inversions"]["share"] is not None
    od = g["oracle_difference"]
    assert od["negative"] + od["zero"] + od["positive"] == od["lines"] > 0
    assert od["mean"]["unit"] == "pages"
    labels = [r["label"] for r in g["cropping"]["strict"]]
    assert {"[0.7, 0.85)", "≥ 0.85"} <= set(labels) <= {"[0.5, 0.7)", "[0.7, 0.85)", "≥ 0.85"}
    det = g["detection"]
    assert det["condition"] and det["rotation_without_rotated_pages"] and g["iou"]["matched"] > 0
    assert all(r["ci_pages"] is None and r["ci_clusters"] for r in det["condition"])
    pipe_md = (out / "pipeline.md").read_text(encoding="utf-8")
    assert "rotation_without_rotated_pages" in pipe_md and "Charset ceiling" in pipe_md
    order = res["reading_order"]
    assert pe.PIPELINE in order["sets"]["gdc_test"] and pe.ORACLE in order["sets"]["gdc_test"]
    assert "GW-BASE-002 ppocr_mobile e2e" in order["sets"]["gdc_test"]
    assert order["sets"]["gdc_test"][pe.PIPELINE]["normalised"]["mean_per_page"] is None
    assert "gdc_inversions" not in order  # stored once, under the pipeline
    assert [x["experiment"] for x in order["omnidocbench_top_edges"]] == list(ODB)
    # H4: the oracle share per condition, the clean slice and the other five.
    h4 = res["oracle_share"]["gdc_test"]
    sizes = _h4_sizes()
    clean = h4["clean (scan_app + photo_flat)"]
    assert (clean["pages"], clean["clusters"]) == sizes["clean"]
    rest = h4["the other five without rotated"]
    assert (rest["pages"], rest["clusters"]) == sizes["the other five without rotated"]
    assert h4["all pages"]["clusters"] == len(DOCS)
    assert "gdc_scan_test" in res["oracle_share"]
    # GW-REC-041: no held-out slice; the ceiling counts the reference characters outside.
    gs = res["glyphsynth"]["tracks"]["strict"]
    assert {"family", "category", "capture_steps", "capture_groups"} <= set(gs)
    assert "held_out" not in gs
    assert set(gs["capture_steps"]) == {"ink_spread", "gaussian_noise"}
    assert gs["ceiling"] > 0 and gs["totals"][0]["ceiling"] == gs["ceiling"]
    gs_md = (out / "glyphsynth.md").read_text(encoding="utf-8")
    assert "capture groups: touching" in gs_md and "Charset ceiling %" in gs_md
    # HierText-derived files carry the attribution and the share-alike note.
    for name in ("recognition.md", "pipeline.md"):
        assert pe.HIERTEXT_ATTRIBUTION in (out / name).read_text(encoding="utf-8")
    assert "share-alike" in (conf / "characters_hiertext_test.md").read_text(encoding="utf-8")
    for s in ("hiertext_test", "hiertext_val"):
        for kind in ("characters", "words"):
            doc = json.loads((conf / f"{kind}_{s}.json").read_text(encoding="utf-8"))
            assert doc["licence"]["attribution"] == pe.HIERTEXT_ATTRIBUTION
            assert "share-alike" in doc["licence"]["note"]
    assert "licence" not in json.loads((conf / "characters_gdc_test.json").read_text("utf-8"))
    # Label-fragment "without" tables include deletions.
    chars_val = (conf / "characters_hiertext_val.md").read_text(encoding="utf-8")
    assert "Without them: deletions" in chars_val


def test_cc_by_sa_gdc_lines_are_never_quoted(tree: dict[str, Any]) -> None:
    assert pe.main(tree["args"]) == 0
    rows = [r for r in _rows(_handoff(tree, "rec_lines_gdc_test.jsonl"))
            if r["source"] == "hefferon_linalg"]  # fmt: skip
    assert rows
    texts = {str(r["ref"]) for r in rows} | {str(r["pred"]) for r in rows}
    texts = {t for t in texts if len(t.split()) > 1}  # whole lines (word tables hold tokens)
    assert texts
    for p in _outputs(tree):
        raw = p.read_bytes()
        for t in texts:
            assert t.encode() not in raw, (p.name, t)
    # The lines are there, listed by id and numbers only.
    res = json.loads((tree["out"] / "results.json").read_text(encoding="utf-8"))
    listed = [x for x in res["recognition"]["gdc_test"]["tracks"]["strict"]["worst_lines"]
              if x["source"] == "hefferon_linalg"]  # fmt: skip
    ex = res["examples"]
    listed += [x for t in ex["types"] for x in t["drawn"] if x["source"] == "hefferon_linalg"]
    listed += [x for x in ex["errors"] + ex["correct"] if x["source"] == "hefferon_linalg"]
    assert listed and all(not x["quoted"] and "ref" not in x and "pred" not in x for x in listed)
    assert "not quoted (CC BY-SA 3.0)" in (tree["out"] / "examples.md").read_text("utf-8")
    assert res["examples"]["sources"]["hefferon_linalg"]["quoted"] is False


def test_funsd_and_iiit5k_text_and_gdc_scan_text_never_reach_an_output(
    tree: dict[str, Any],
) -> None:
    # Plant the marker in every GDC-scan hand-off file, as text fields its writer never wrote,
    # so the analysis must refuse the set by name rather than rely on missing fields.
    rec = _handoff(tree, "rec_lines_gdc_scan_test.jsonl")
    pairs = _handoff(tree, "e2e_pairs_gdc_scan_test.jsonl")
    pages = _handoff(tree, "pages_gdc_scan_test.jsonl")
    rows = _rows(rec)
    for r in rows:
        r.update(ref=f"{MARKER} ref", pred=f"{MARKER} pred", oracle_pred=MARKER)
    _jsonl(rec, rows)
    rows = _rows(pairs)
    for r in rows:
        for p in r["pairs"]:
            p.update(ref=MARKER, text=MARKER)
    _jsonl(pairs, rows)
    rows = _rows(pages)
    for r in rows:
        r["note"] = MARKER
    _jsonl(pages, rows)
    _rerecord(tree, rec, pairs, pages)
    assert pe.main(tree["args"]) == 0
    files = _outputs(tree)
    assert any(p.name == "results.json" for p in files) and len(files) > 10
    for p in files:
        raw = p.read_bytes().lower()
        assert MARKER.lower().encode() not in raw, p.name
        assert FUNSD_TEXT.lower().encode() not in raw, p.name
        assert IIIT_TEXT.lower().encode() not in raw, p.name
    # The markers are in the inputs, so the test would see a leak.
    assert MARKER.encode() in rec.read_bytes()
    assert FUNSD_TEXT.encode() in _handoff(tree, "rec_lines_funsd_test.jsonl").read_bytes()


def test_without_gw_rec_041_its_section_says_so(tree: dict[str, Any]) -> None:
    assert pe.main(tree["args"]) == 0
    text = (tree["out"] / "glyphsynth.md").read_text(encoding="utf-8")
    assert "Not run:" in text and "phase15_glyphsynth.py" in text


def test_the_crop_index_is_read_only_after_the_edges(
    tree: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    edges, crops = pe.validation_edges, pe.verify_crops

    def spy_edges(rows: Any) -> Any:
        calls.append("edges")
        return edges(rows)

    def spy_crops(data: Path) -> Any:
        calls.append("crops")
        return crops(data)

    monkeypatch.setattr(pe, "validation_edges", spy_edges)
    monkeypatch.setattr(pe, "verify_crops", spy_crops)
    assert pe.main(tree["args"]) == 0
    assert calls == ["edges", "crops"]


def test_an_existing_output_is_replaced_whole(tree: dict[str, Any]) -> None:
    assert pe.main(tree["args"]) == 0
    stale = tree["out"] / "stale.md"
    stale.write_text("old", encoding="utf-8")
    assert pe.main(tree["args"]) == 0
    assert not stale.exists() and (tree["out"] / "results.json").is_file()


# ------------------------------------------------------------------------------ refusals


def test_a_changed_handoff_file_refuses_before_any_output(tree: dict[str, Any]) -> None:
    path = _handoff(tree, "rec_lines_gdc_test.jsonl")
    path.write_bytes(path.read_bytes().replace(b'"blur": ', b'"blur":  ', 1))
    with pytest.raises(SystemExit, match="differs from its hash"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_an_unrecorded_input_refuses(tree: dict[str, Any]) -> None:
    data = tree["data"]
    doc = json.loads((data / ps.OUTPUTS).read_text(encoding="utf-8"))
    doc["outputs"] = [e for e in doc["outputs"] if "pages_funsd_test" not in e["file"]]
    (data / ps.OUTPUTS).write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="not recorded"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_a_changed_crop_refuses_before_any_output(tree: dict[str, Any]) -> None:
    index = _handoff(tree, "gdc_crops/index.jsonl")
    first = json.loads(index.read_text("utf-8").splitlines()[0])
    (index.parent / first["file"]).write_bytes(b"not the crop")
    with pytest.raises(SystemExit, match="crop"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_a_crop_changed_after_the_check_leaves_no_output(
    tree: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    analyse = pe.analyse

    def then_change(opts: pe.Options) -> Any:
        res, draws = analyse(opts)
        (tree["data"] / ps.HANDOFF / "gdc_crops" / draws[-1].file).write_bytes(b"changed")
        return res, draws

    monkeypatch.setattr(pe, "analyse", then_change)
    with pytest.raises(SystemExit, match="changed since it was checked"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_a_failure_while_writing_leaves_no_partial_output(
    tree: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_: Any) -> None:
        raise RuntimeError("the plot failed")

    monkeypatch.setattr(pe, "class_matrix_png", broken)
    with pytest.raises(RuntimeError, match="the plot failed"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_a_failure_keeps_the_previous_outputs(
    tree: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert pe.main(tree["args"]) == 0
    before = {p: p.read_bytes() for p in _outputs(tree)}

    def broken(*_: Any) -> str:
        raise RuntimeError("rendering failed")

    monkeypatch.setattr(pe, "render_glyphsynth", broken)
    with pytest.raises(RuntimeError):
        pe.main(tree["args"])
    assert {p: p.read_bytes() for p in _outputs(tree)} == before


def test_a_changed_phase5_file_refuses(tree: dict[str, Any]) -> None:
    path = run_baselines.output_path(tree["data"], "GW-BASE-001", "tesseract", "rec", "gdc_test")
    path.write_text(path.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Phase 5"):
        pe.main(tree["args"])
    _nothing_written(tree)


@pytest.mark.parametrize("flag", ["--out", "--confusion-out", "--work"])
@pytest.mark.parametrize("folder", er.PROTECTED_OUTPUTS)
def test_outputs_under_a_protected_folder_are_refused(
    tree: dict[str, Any], flag: str, folder: str
) -> None:
    args = list(tree["args"])
    target = tree["data"] / folder / "phase15"
    args[args.index(flag) + 1] = str(target)
    with pytest.raises(SystemExit, match="never writes"):
        pe.main(args)
    assert not target.exists()
    _nothing_written(tree)


@pytest.mark.parametrize(
    ("flag", "where"),
    [
        ("--out", lambda t: t["data"] / "processed"),
        ("--out", lambda t: t["data"]),
        ("--confusion-out", lambda t: t["results"].parent),
        ("--confusion-out", lambda t: t["tmp"]),
    ],
)
def test_an_output_folder_holding_protected_files_is_refused(
    tree: dict[str, Any], flag: str, where: Any
) -> None:
    args = list(tree["args"])
    target = where(tree)
    before = sorted(str(p) for p in target.rglob("*"))
    args[args.index(flag) + 1] = str(target)
    with pytest.raises(SystemExit, match=r"must not replace|never writes"):
        pe.main(args)
    assert sorted(str(p) for p in target.rglob("*")) == before


def test_nested_output_folders_are_refused(tree: dict[str, Any]) -> None:
    args = list(tree["args"])
    args[args.index("--confusion-out") + 1] = str(tree["out"] / "conf")
    with pytest.raises(SystemExit, match="overlap"):
        pe.main(args)
    _nothing_written(tree)


def test_an_existing_folder_not_written_by_phase15_is_refused(tree: dict[str, Any]) -> None:
    tree["out"].mkdir(parents=True)
    (tree["out"] / "keep.txt").write_text("someone else's", encoding="utf-8")
    with pytest.raises(SystemExit, match="not written by this script"):
        pe.main(tree["args"])
    assert (tree["out"] / "keep.txt").read_text(encoding="utf-8") == "someone else's"


def test_a_gallery_folder_git_does_not_ignore_is_refused(tree: dict[str, Any]) -> None:
    args = list(tree["args"])
    args[args.index("--work") + 1] = str(tree["tmp"] / "tracked")
    with pytest.raises(SystemExit, match="not git-ignored"):
        pe.main(args)
    assert not (tree["tmp"] / "tracked").exists()
    _nothing_written(tree)


@pytest.mark.parametrize(
    ("experiment", "eval_set", "metric"),
    [("GW-REC-039", "gdc_test", "cer"), ("GW-E2E-007", "funsd_test", "page_cer_bag_of_lines"),
     ("GW-E2E-008", "gdc_test", "page_cer_order_aware")],
)  # fmt: skip
def test_the_consistency_check_refuses_a_planted_mismatch(
    tree: dict[str, Any], experiment: str, eval_set: str, metric: str
) -> None:
    res = json.loads(tree["results"].read_text(encoding="utf-8"))
    entry = next(
        e for e in res["predictions"]
        if (e["experiment"], e["eval_set"]) == (experiment, eval_set)
    )  # fmt: skip
    entry["tracks"]["strict"][metric]["value"] += 1e-6
    tree["results"].write_text(json.dumps(res), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not reproduce"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_a_gdc_condition_mismatch_refuses(tree: dict[str, Any]) -> None:
    res = json.loads(tree["results"].read_text(encoding="utf-8"))
    row = next(r for r in res["gdc_conditions"]["rows"] if r["experiment"] == "GW-REC-039")
    row["tracks"]["normalised"]["cer"]["value"] += 0.01
    tree["results"].write_text(json.dumps(res), encoding="utf-8")
    with pytest.raises(SystemExit, match="does not reproduce"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_a_pages_file_that_disagrees_with_the_pairs_refuses(tree: dict[str, Any]) -> None:
    pages = _handoff(tree, "pages_hiertext_test.jsonl")
    rows = _rows(pages)
    rows[0]["systems"][pe.PIPELINE]["strict"]["bag_ref_chars"] += 1
    _jsonl(pages, rows)
    _rerecord(tree, pages)
    with pytest.raises(SystemExit, match="bag reference characters differ"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_the_budget_asserts_fire(tree: dict[str, Any]) -> None:
    pairs = _rows(_handoff(tree, "e2e_pairs_gdc_test.jsonl"))
    pages = _rows(_handoff(tree, "pages_gdc_test.jsonl"))
    assert pe.budget("gdc_test", pairs, pages, "strict")  # as written, it holds
    key = next(r["key"] for r in pairs if not r["spurious"])
    quiet = next(i for i, r in enumerate(pages) if r["key"] == key)
    v = pages[quiet]["systems"][pe.PIPELINE]["strict"]
    v["bag_errors"] += 1  # spurious characters on a page without a spurious line
    with pytest.raises(SystemExit, match="budget does not hold"):
        pe.budget("gdc_test", pairs, pages, "strict")
    v["bag_errors"] = -1  # fewer bag errors than missed + matched
    with pytest.raises(SystemExit, match="budget does not hold"):
        pe.budget("gdc_test", pairs, pages, "strict")
    # Through the run: a planted budget error stops it before anything is written.
    path = _handoff(tree, "pages_gdc_test.jsonl")
    rows = _rows(path)
    rows[quiet]["systems"][pe.PIPELINE]["strict"]["bag_errors"] += 1
    _jsonl(path, rows)
    _rerecord(tree, path)
    res = json.loads(tree["results"].read_text(encoding="utf-8"))
    entry = next(e for e in res["predictions"]
                 if (e["experiment"], e["eval_set"]) == ("GW-E2E-007", "gdc_test"))  # fmt: skip
    b = entry["tracks"]["strict"]["page_cer_bag_of_lines"]
    errs = sum(r["systems"][pe.PIPELINE]["strict"]["bag_errors"] for r in rows)
    refs = sum(r["systems"][pe.PIPELINE]["strict"]["bag_ref_chars"] for r in rows)
    b["value"] = errs / refs  # results.json agrees, so only the budget can refuse
    tree["results"].write_text(json.dumps(res), encoding="utf-8")
    with pytest.raises(SystemExit, match="budget does not hold"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_wrong_h4_slice_sizes_refuse(tree: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = _h4_sizes()
    sizes["clean"] = (sizes["clean"][0], sizes["clean"][1] + 1)
    monkeypatch.setattr(pe, "H4_SIZES", sizes)
    with pytest.raises(SystemExit, match="H4's clean slice"):
        pe.main(tree["args"])
    _nothing_written(tree)
    assert pe.H4_SIZES is sizes and _h4_sizes() != sizes


def test_the_registered_h4_sizes() -> None:
    assert pe.H4_SIZES == {"clean": (53, 21), "the other five": (53, 18),
                           "the other five without rotated": (46, 18)}  # fmt: skip


def test_gdc_sources_outside_the_selection_refuse(tree: dict[str, Any]) -> None:
    path = tree["data"] / "manifests/gdc/selection.yaml"
    sel = yaml.safe_load(path.read_text(encoding="utf-8"))
    sel["documents"] = [d for d in sel["documents"] if d["key"] != "irs_w9"]
    path.write_text(yaml.safe_dump(sel), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"outside selection\.yaml"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_a_gdc_source_without_a_known_licence_refuses(tree: dict[str, Any]) -> None:
    path = tree["data"] / "manifests/gdc/selection.yaml"
    sel = yaml.safe_load(path.read_text(encoding="utf-8"))
    sel["documents"][0]["file"] = "misc/unknown.pdf"
    path.write_text(yaml.safe_dump(sel), encoding="utf-8")
    sources = tree["data"] / "manifests/gdc_sources.json"
    doc = json.loads(sources.read_text(encoding="utf-8"))
    doc["files"]["src_llama"]["path"] = "raw/gdc_sources/misc/unknown.pdf"
    sources.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="no known licence"):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_gdc_licences_by_folder() -> None:
    assert pe.gdc_licence("arxiv/2302.13971v1.pdf").startswith("CC BY 4.0")
    assert pe.gdc_licence("gov/census-p60-282.pdf").startswith("public domain")
    assert pe.gdc_licence("forms/irs-fw9-2024.pdf").startswith("public domain")
    assert pe.gdc_licence("books/pg-17384.pdf") == pe.GUTENBERG
    for f in ("books/hefferon-linear-algebra.pdf", "books/mit-mcs-2018.pdf"):
        assert pe.gdc_licence(f) == pe.CC_BY_SA
    with pytest.raises(SystemExit, match="no known licence"):
        pe.gdc_licence("books/other.pdf")


@pytest.mark.parametrize(
    ("meta", "match"),
    [({"experiment": "GW-REC-040"}, "names"), ({"eval_set": "hiertext_val"}, "names"),
     ({"split": "test"}, "names"), ({"output_sha256": "0" * 64}, "hash or count"),
     ({"count": 39}, "hash or count"), ({"limit": 20}, "--limit")],
)  # fmt: skip
def test_a_gw_rec_041_file_that_differs_from_its_meta_refuses(
    tree: dict[str, Any], meta: dict[str, Any], match: str
) -> None:
    _synth(tree["data"], **meta)
    with pytest.raises(SystemExit, match=match):
        pe.main(tree["args"])
    _nothing_written(tree)


def test_a_gw_rec_041_file_without_meta_refuses(tree: dict[str, Any]) -> None:
    path = _synth(tree["data"])
    path.with_name("glyphsynth_val.meta.json").unlink()
    with pytest.raises(SystemExit, match=r"no \.meta\.json"):
        pe.main(tree["args"])
    _nothing_written(tree)


# ---------------------------------------------------------------------- unit-level refusals


def _ctx() -> pe.Ctx:
    allowed = frozenset(CHARSET)
    edges = {k: {"edges": [1.0]} for k in ("line_height_px", "blur", "contrast")}
    return pe.Ctx({"strict": allowed, "normalised": allowed}, edges)


def test_text_builders_refuse_the_numbers_only_set() -> None:
    allowed = frozenset(CHARSET)
    ctx = _ctx()
    with pytest.raises(ValueError, match="numbers only"):
        pe.content_flags("gdc_scan_test", "a", "b", "strict", allowed)
    unit = {"key": "k", "ref_norm": "ab", "pred_norm": "ac", "char_errors": 1, "source": "s"}
    with pytest.raises(ValueError, match="numbers only"):
        pe.char_tables("gdc_scan_test", [unit], "strict", allowed)
    with pytest.raises(ValueError, match="numbers only"):
        pe.word_tables("gdc_scan_test", [unit])
    with pytest.raises(ValueError, match="numbers only"):
        pe.baseline_rates("gdc_scan_test", [], [], {"strict": allowed})
    with pytest.raises(ValueError, match="numbers only"):
        pe.categories("gdc_scan_test", [unit], ctx)
    with pytest.raises(ValueError, match="numbers only"):
        pe.gdc_examples("gdc_scan_test", [unit], [], ctx)
    with pytest.raises(ValueError, match="numbers only"):
        pe.oracle_difference("gdc_scan_test", [], [])
    numbers = {"char_errors": 1, "ref_chars": 2, "exact": False}
    rows = [{"key": "k", "doc_id": "p", "strict": numbers, "ref": MARKER, "pred": MARKER}]
    units = pe.line_units("gdc_scan_test", rows, "strict", ctx, True)
    assert all(MARKER not in json.dumps(u) for u in units)  # the text fields are never read


def test_line_length_is_the_strict_length_on_every_track() -> None:
    row = {"key": "k", "doc_id": "p", "ref": "A-1x", "pred": "A1x",
           "strict": {"char_errors": 1, "ref_chars": 4, "exact": False},
           "alnum": {"char_errors": 0, "ref_chars": 3, "exact": True}}  # fmt: skip
    ctx = _ctx()
    ctx.allowed["alnum"] = frozenset(CHARSET)
    (u,) = pe.line_units("iiit5k_test", [row], "alnum", ctx, False)
    assert u["length"] == 4 and u["ref_chars"] == 3


def test_a_line_whose_operations_differ_from_char_errors_refuses() -> None:
    unit = {"key": "k", "ref_norm": "ab", "pred_norm": "ac", "char_errors": 2}
    with pytest.raises(SystemExit, match="edit operations but char_errors"):
        pe.char_tables("gdc_test", [unit], "strict", frozenset(CHARSET))


def test_categories_that_do_not_add_up_refuse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(er, "primary_category", lambda *_: "not_a_category")
    with pytest.raises(SystemExit, match="do not add up"):
        pe.categories("gdc_test", [{"char_errors": 1}], _ctx())


def test_two_sources_on_one_page_refuse() -> None:
    rows = [{"doc_id": "p1", "source": "llama"}, {"doc_id": "p1", "source": "blip2"}]
    with pytest.raises(SystemExit, match="two source documents"):
        pe.page_sources(rows)
    assert pe.page_sources(rows[:1]) == {"p1": "llama"}


def test_the_oracle_and_pipeline_must_share_reference_characters() -> None:
    def sys_(ref: int) -> dict[str, Any]:
        return {"strict": {"bag_ref_chars": ref, "bag_errors": 1}}

    rows: list[dict[str, Any]] = [
        {"key": "p", "systems": {pe.ORACLE: sys_(3), pe.PIPELINE: sys_(4)}}
    ]
    with pytest.raises(SystemExit, match="differ in reference characters"):
        pe.oracle_share("gdc_scan_test", rows, [], [])
    rows[0]["systems"][pe.ORACLE] = sys_(4)
    assert pe.oracle_share("gdc_scan_test", rows, [], [])["all pages"]["value"] == 1.0


def _pair(line: str, ref: str, errors: int, ref_chars: int = 3) -> dict[str, Any]:
    return {"pred": 0, "line_id": line, "ref": ref,
            "strict": {"char_errors": errors, "ref_chars": ref_chars}}  # fmt: skip


def test_the_oracle_difference() -> None:
    rec = [{"key": f"l{i}", "ref": "abc", "oracle_strict": {"char_errors": i % 2}}
           for i in range(36)]  # fmt: skip
    pairs: list[dict[str, Any]] = [
        {"key": f"p{j}", "pairs": [_pair(f"l{6 * j + i}", "abc", 1) for i in range(6)]}
        for j in range(6)
    ]
    pairs[0]["pairs"].append(_pair("l99", "", 0, ref_chars=0))  # empty reference: left out
    out = pe.oracle_difference("gdc_test", pairs, rec)
    assert out["lines"] == 36 and out["pages"] == 6 and out["empty_references"] == 1
    assert (out["negative"], out["zero"], out["positive"]) == (0, 18, 18)
    assert out["mean"].unit == "pages" and out["mean_value"] == pytest.approx(0.5)
    pairs[1]["pairs"][0]["ref"] = "abd"  # the pack label differs from the page line's text
    with pytest.raises(SystemExit, match="pack label differs"):
        pe.oracle_difference("gdc_test", pairs, rec)


def _baseline(tmp_path: Path, rows: list[dict[str, Any]]) -> list[tuple[str, Path]]:
    path = tmp_path / "base.jsonl"
    _jsonl(path, rows)
    return [("GW-BASE-001", path)]


def test_baseline_rates_need_the_same_lines(tmp_path: Path) -> None:
    allowed: dict[str, frozenset[str]] = {t: frozenset(CHARSET) for t in pe.TRACKS}
    glyph = [{"key": f"k{i}", "ref": "abc", "pred": "abd"} for i in range(3)]
    same = [{"key": f"k{i}", "ref": "abc", "pred": "abc"} for i in range(3)]
    rows = pe.baseline_rates("gdc_test", glyph, _baseline(tmp_path, same), allowed)
    assert [r["lines"] for r in rows] == [3, 3, 3, 3]
    assert rows[0]["correct_rates"]["lower"] == pytest.approx(2 / 3)
    assert rows[0]["class_rates"]["lower"]["lower"] == pytest.approx(1.0)  # c -> d same class
    for bad, match in (
        ([*same, same[0]], "twice"),
        (same[:2], "other lines"),
        ([*same, {"key": "k9", "ref": "x", "pred": "x"}], "other lines"),
        ([{**same[0], "ref": "abx"}, *same[1:]], "stored reference"),
    ):
        with pytest.raises(SystemExit, match=match):
            pe.baseline_rates("gdc_test", glyph, _baseline(tmp_path, bad), allowed)
    with pytest.raises(SystemExit, match="twice"):
        pe.baseline_rates("gdc_test", [*glyph, glyph[0]], _baseline(tmp_path, same), allowed)


def test_a_validation_key_of_another_form_refuses() -> None:
    rows = pe.val_rows_as_lines([{"key": "img1/p0/l3", "ref": "ab", "pred": "ab"}])
    assert rows[0]["doc_id"] == "img1"
    with pytest.raises(SystemExit, match="HierText line id"):
        pe.val_rows_as_lines([{"key": "img1/l3", "ref": "ab", "pred": "ab"}])


def test_draws_are_reproducible_and_registered() -> None:
    assert pe._draw(10, 3, [pe.EXAMPLE_SEED, 1]) == pe._draw(10, 3, [pe.EXAMPLE_SEED, 1])
    want = np.sort(np.random.default_rng(pe.EXAMPLE_SEED).choice(50, 40, replace=False))
    assert pe._draw(50, 40, pe.EXAMPLE_SEED) == [int(i) for i in want]
    assert pe._draw(2, 3, 1) == [0, 1] and pe._draw(0, 3, 1) == []


def test_validation_edges_and_gone_characters() -> None:
    rows = [{"ref": "x" * n, "line_height_px": float(n), "blur": 1.0, "contrast": float(n % 3),
             "conf_t": n / 10} for n in range(1, 9)]  # fmt: skip
    e = pe.validation_edges(rows)
    assert e["length"]["edges"] == [2.75, 4.5, 6.25]
    assert e["blur"]["edges"] == [1.0] and e["blur"]["validation_lines"] == [0, 8]
    assert pe._norm_char("’") == pe._norm_char("'") and pe._norm_char(" ") == " "
    assert pe._norm_char("A") == "a"


def test_printed_intervals_are_counted_once() -> None:
    pe._PRINTED.clear()
    iv = {"value": 0.1, "low": 0.05, "high": 0.2}
    row = {"cer": 0.1, "ci_lines": iv, "ci_pages": None, "ci_clusters": None}
    assert pe._cer_cell(row) == pe.pct(iv) and pe._line_cell(row) == "cited"
    pe.pct(iv)
    assert len(pe._PRINTED) == 1
    assert pe._cer_cell({"cer": 0.25}) == "25.00" and len(pe._PRINTED) == 1


def test_a_work_folder_ignored_only_for_its_gallery_is_refused(tree: dict[str, Any]) -> None:
    (tree["tmp"] / ".gitignore").write_text("galleries/\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not git-ignored"):
        pe.main(tree["args"])
    _nothing_written(tree)
