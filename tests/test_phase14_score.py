"""GW-E2E-009's scoring pass on a small synthetic repository layout (tmp_path only).

Every set, prediction file, Phase 5 report and evaluator is made up here: no prepared benchmark,
no real prediction file and no real access log is read, and the official evaluators are replaced
by fakes that check what they are given.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import evaluate_baselines
import numpy as np
import official_hiertext
import official_omnidocbench
import phase14_score as ps
import pytest
import run_baselines
from PIL import Image

from glypharis.data.crops import encode_jpeg
from glypharis.data.linepack import LinePackWriter
from glypharis.data.pages import Page, PageLine, read_pages, write_pages
from glypharis.data.text import Charset
from glypharis.eval.inputs import check_inputs
from glypharis.eval.page_metrics import PredLine, evaluate_page
from glypharis.eval.sets import EVAL_SETS, OMNIDOCBENCH_SUBSET, SUBSET_NAME
from glypharis.eval.stats import Interval

LOG_HEADER = "| Date | Benchmark | ID | Checkpoint | Phase | Reason |\n|---|---|---|---|---|---|\n"
PLACEHOLDER = "| — | — | — | — | — | No test access yet. |\n"
MARKER = "Zqxjvmarker"  # stands for GDC-scan's text; must never reach a report
SETTINGS = {"cpu_threads": 10, "module": "fake"}
CONDITIONS = ["scan_app", "photo_flat", "scan_app", "photo_flat", "curved", "low_light"]
SOURCES = ["doc-a", "doc-a", "doc-b", "doc-b", "doc-c", "doc-c"]
WORDS = ["alpha beta", "gamma delta", "epsilon zeta", "theta iota", "kappa lambda"]


def _box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _line(line_id, i, text, vertical=False, x=0):
    box = _box(150, 0, 170, 90) if vertical else _box(x, 30 * i, x + 120, 30 * i + 20)
    skip = "vertical" if vertical else None
    return PageLine(line_id, box, text, text, True, False, vertical, not vertical, skip)


def _page(doc_id, image, texts, source, vertical=False):
    lines = [_line(f"{doc_id}/l{i}", i, t) for i, t in enumerate(texts)]
    if vertical:
        lines.append(_line(f"{doc_id}/v0", 0, "STAMP", vertical=True))
    return Page(
        doc_id, image, 200, 200, source, "test" if "val" not in doc_id else "validation", lines
    )


def _pack(prefix, rows, split, source):
    crop = encode_jpeg(Image.new("L", (40, 16), 255), 95)
    with LinePackWriter(prefix) as w:
        for key, doc_id, label in rows:
            w.add(key, doc_id, split, source, label, crop, (40, 16))


def _corrupt(text, key, system):
    """A deterministic reading with some errors, different per system."""
    h = int(hashlib.sha256(f"{system}:{key}".encode()).hexdigest(), 16)
    if h % 3 == 0:
        return text
    if h % 3 == 1:
        return text[:-1]
    return text.replace("a", "o", 1) + ("x" if system != "glypharis" else "")


def _jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def _meta(path, **fields):
    path.with_suffix(".meta.json").write_text(json.dumps(fields), encoding="utf-8")


def _rec_rows(pages_or_pack, system, glypharis=False):
    rows = []
    for key, _, label in pages_or_pack:
        row = {"key": key, "ref": label, "pred": _corrupt(label, key, system)}
        if glypharis:
            h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 100
            row.update(
                conf_t1=0.5 + h / 250,
                conf_t=0.4 + h / 200,
                time_ms=1.5,
                height=16,
                blur=12.5,
                contrast=30.0,
            )
        rows.append(row)
    return rows


def _e2e_rows(pages, system, glypharis=False, oracle=False):
    rows = []
    for p in pages:
        lines = []
        for i, ln in enumerate(p.lines):
            if ln.vertical or (not oracle and i == 1 and system == "ppocr_mobile"):
                continue  # the baseline misses a line
            out = {
                "polygon": ln.polygon,
                "text": _corrupt(ln.text, ln.line_id, system),
                "confidence": 0.8,
            }
            if glypharis:
                out.update(conf_t1=0.9, det_score=0.95, quad=ln.polygon)
            if oracle:
                out["line_id"] = ln.line_id
            lines.append(out)
        if not oracle and system == "glypharis":
            lines.append(
                {
                    "polygon": _box(10, 180, 60, 195),
                    "text": "zz",
                    "confidence": 0.3,
                    "conf_t1": 0.6,
                    "det_score": 0.5,
                    "quad": _box(10, 180, 60, 195),
                }
            )
        rows.append({"key": p.doc_id, "lines": lines, "time_ms": 3.0})
    return rows


def _crop_rows(pages):
    return [(ln.line_id, p.doc_id, ln.text) for p in pages for ln in p.lines if not ln.vertical]


def _inputs(data, eval_set, task):
    """The "inputs" block `scripts/ext_eval.py` records: `check_inputs` at run time."""
    pages = EVAL_SETS[eval_set]["pages"]
    n = None if task == "rec" or not pages else len(list(read_pages(data / pages)))
    return check_inputs(data, eval_set, task, n, require=False)


def _recorded(data, path, experiment):
    """A GLYPHARIS output's phase14_outputs.json entry as `ext_eval.record_output` writes it."""
    meta = path.with_suffix(".meta.json")
    return {
        **ps.file_entry(path, data),
        "experiment": experiment,
        "meta": "data/" + meta.relative_to(data).as_posix(),
        "meta_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(),
    }


def _rerecord(data, *, restamp=False):
    """Record every GLYPHARIS output again, as if its runner had written the files as they are
    now (after a test changed one); ``restamp`` also re-takes each run's inputs block."""
    entries = []
    for exp, (task, sets) in ps.GLYPHARIS_RUNS.items():
        for eval_set in sets:
            path = data / "processed/phase14" / exp / f"{eval_set}.jsonl"
            if restamp:
                meta = path.with_suffix(".meta.json")
                doc = json.loads(meta.read_text(encoding="utf-8"))
                doc["inputs"] = _inputs(data, eval_set, task)
                meta.write_text(json.dumps(doc), encoding="utf-8")
            entries.append(_recorded(data, path, exp))
    (data / ps.OUTPUTS).write_text(
        json.dumps({"note": "test", "outputs": sorted(entries, key=lambda e: e["file"])}),
        encoding="utf-8",
    )


def _glyph_fields(threads, pin, provenance):
    """The fields `scripts/e2e_profile.py --frozen` writes about threads, the pin and the model."""
    return {
        "settings": {"cpu_threads": threads, "intra_op_threads": threads,
                     "inter_op_threads": threads},
        "threads": {"requested": threads, "intra_op": threads, "inter_op": threads,
                    "env": {"OMP_NUM_THREADS": str(threads)}, "env_before_torch_import": True},
        "pinned": dict(pin),
        "provenance": dict(provenance),
    }  # fmt: skip


def _sitting(start, minutes=20):
    """The ``started_utc`` / ``ended_utc`` pair both profilers write, from 10:00 UTC."""
    t0 = datetime(2026, 11, 16, 10, 0, tzinfo=UTC) + timedelta(minutes=start)
    return {
        "started_utc": t0.isoformat(),
        "ended_utc": (t0 + timedelta(minutes=minutes)).isoformat(),
    }


def _profile(path, times, sizes, threads_field, **extra):
    cond = {
        "on_ac_power": True,
        "power_plan": "GUID x  (Balanced)",
        "power_mode": "balanced",
        "cpu": "fake cpu",
        "cpu_busy_percent_before": 2.0,
    }
    doc = {
        "times_ms": times,
        "input_sizes": sizes,
        "warmup": 10,
        "batch": 1,
        "conditions": cond,
        **threads_field,
        **extra,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A repository layout with every set, the Phase 5 baseline files and their report, the
    Phase 14 files of a reduced baseline plan, every GLYPHARIS file and the latency profiles."""
    data = tmp_path / "data"
    Charset(list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")).save(
        data / "manifests/charset.json"
    )
    pages: dict[str, list[Page]] = {}
    pages["gdc_test"] = [
        _page(
            f"gdc{i}",
            f"raw/gdc/captures/{i:03d}.jpg",
            WORDS[i % 3 : i % 3 + 3],
            "gdc",
            vertical=i == 0,
        )
        for i in range(6)
    ]
    pages["gdc_scan_test"] = [
        _page(
            f"scan{i}",
            f"raw/gdc_scan/p{i}.jpg",
            # The third line holds a character outside the charset (é).
            [f"{MARKER} one", f"two {MARKER}", f"{MARKER} été"],
            "gdc_scan",
            vertical=True,
        )
        for i in range(2)
    ]
    pages["funsd_test"] = [
        _page(f"funsd{i}", f"raw/funsd/{i}.png", WORDS[:2], "funsd") for i in range(2)
    ]
    pages["hiertext_test"] = [
        _page(f"ht{i}", f"raw/hiertext/{i}.jpg", WORDS[1:4], "hiertext") for i in range(3)
    ]
    pages["hiertext_val"] = [
        _page(f"val{i}", f"raw/hiertext/v{i}.jpg", WORDS[:3], "hiertext") for i in range(3)
    ]
    pages["omnidocbench_test"] = [
        _page(
            f"odb-odb_{i}", f"raw/omnidocbench/images/odb_{i}.png", WORDS[i : i + 2], "omnidocbench"
        )
        for i in range(3)
    ]
    for name, ps_ in pages.items():
        spec = EVAL_SETS[name]
        write_pages(data / spec["pages"], ps_)
        if spec["pack"]:
            _pack(data / spec["pack"], _crop_rows(ps_), spec["split"], spec["benchmark"])
    iiit = [(f"iiit5k/{i}", "iiit5k", w) for i, w in enumerate(("HELLO", "WORLD", "A-1", "OK"))]
    _pack(data / EVAL_SETS["iiit5k_test"]["pack"], iiit, "test", "iiit5k")
    subset = data / OMNIDOCBENCH_SUBSET.relative_to("data")
    subset.parent.mkdir(parents=True, exist_ok=True)
    subset.write_text(json.dumps({"pages": ["odb-odb_0", "odb-odb_2"]}), encoding="utf-8")
    crops = {k: _crop_rows(v) for k, v in pages.items()}
    crops["iiit5k_test"] = iiit

    # Phase 5: PP-OCRv5 mobile everywhere it ran, Tesseract on GDC crops, one profile.
    base5 = data / "processed/baselines"
    for system, task, eval_set in [
        ("ppocr_mobile", "rec", "gdc_test"),
        ("ppocr_mobile", "e2e", "gdc_test"),
        ("ppocr_mobile", "rec", "hiertext_test"),
        ("ppocr_mobile", "e2e", "hiertext_test"),
        ("ppocr_mobile", "rec", "hiertext_val"),
        ("tesseract", "rec", "gdc_test"),
    ]:
        exp = run_baselines.EXPERIMENTS[system]
        path = base5 / exp / f"{system}_{task}_{eval_set}.jsonl"
        rows = (
            _rec_rows(crops[eval_set], system)
            if task == "rec"
            else _e2e_rows(pages[eval_set], system)
        )
        _jsonl(path, rows)
        _meta(
            path,
            system=system,
            task=task,
            eval_set=eval_set,
            experiment=exp,
            expected=len(rows),
            samples=len(rows),
            settings=SETTINGS,
            version="fake 1.0",
            runs=[{"written": len(rows)}],
        )
    sizes = [[100 + i, 80] for i in range(100)]
    _profile(
        base5 / "GW-BASE-002/profile_ppocr_mobile_e2e_hiertext_val.json",
        [50.0 + i % 7 for i in range(100)],
        sizes,
        {"settings": SETTINGS},
        system="ppocr_mobile",
        task="e2e",
        eval_set="hiertext_val",
        experiment="GW-BASE-002",
        peak_mem_mib=100.0,
        model_mib=5.0,
    )
    p5log = tmp_path / "phase5_log.md"
    p5log.write_text(LOG_HEADER + PLACEHOLDER, encoding="utf-8")
    p5report = tmp_path / "phase5_report"
    assert (
        evaluate_baselines.main(
            [
                "--phase",
                "5",
                "--data-root",
                str(data),
                "--out",
                str(p5report),
                "--access-log",
                str(p5log),
            ]
        )
        == 0
    )
    phase5 = json.loads((p5report / "results.json").read_text(encoding="utf-8"))

    # Phase 14 baselines, a reduced plan.
    plan = {
        ("ppocr_mobile", "rec", "gdc_scan_test"): None,
        ("ppocr_mobile", "e2e", "gdc_scan_test"): None,
        ("ppocr_mobile", "rec", "iiit5k_test"): None,
        ("ppocr_mobile", "e2e", "funsd_test"): None,
        ("ppocr_mobile", "e2e", "omnidocbench_test"): None,
        ("ppocr_server", "e2e", "omnidocbench_test"): SUBSET_NAME,
        ("trocr", "e2e", "funsd_test"): "not applicable: TrOCR has no detector",
    }
    monkeypatch.setattr(ps, "BASELINE_PLAN", plan)
    for (system, task, eval_set), planned in plan.items():
        if planned not in (None, SUBSET_NAME):
            continue
        sub = planned == SUBSET_NAME
        exp = run_baselines.EXPERIMENTS[system]
        path = run_baselines.output_path(data, exp, system, task, eval_set, 14, sub)
        src = pages.get(eval_set, [])
        if sub:
            src = [p for p in src if p.doc_id in ("odb-odb_0", "odb-odb_2")]
        rows = _rec_rows(crops[eval_set], system) if task == "rec" else _e2e_rows(src, system)
        _jsonl(path, rows)
        _meta(
            path,
            system=system,
            task=task,
            eval_set=eval_set,
            experiment=exp,
            expected=len(rows),
            samples=len(rows),
            settings=SETTINGS,
            version="fake 1.0",
            runs=[],
            phase=14,
            subset=SUBSET_NAME if sub else None,
        )
    _profile(
        run_baselines.output_path(
            data, "GW-BASE-002", "ppocr_mobile", "e2e", "hiertext_val", 14
        ).with_name("profile_ppocr_mobile_e2e_hiertext_val.json"),
        [60.0 + i % 5 for i in range(100)],
        sizes,
        {"settings": SETTINGS},
        system="ppocr_mobile",
        task="e2e",
        eval_set="hiertext_val",
        experiment="GW-BASE-002",
        **_sitting(0),
    )

    # The freeze record and every GLYPHARIS file.
    record = tmp_path / "configs/glypharis_v1.yaml"
    record.parent.mkdir(parents=True)
    record.write_text(
        "recognizer:\n  checkpoint_sha256: aaa\ndetector:\n  checkpoint_sha256: bbb\n",
        encoding="utf-8",
    )
    provenance = {
        "record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        "recognizer_checkpoint_sha256": "aaa",
        "detector_checkpoint_sha256": "bbb",
        "charset_sha256": hashlib.sha256(
            (data / "manifests/charset.json").read_bytes()
        ).hexdigest(),
    }
    pin = {"tag": "phase14-eval", "tag_commit": "1234", "head": "1234", "trees": "equal"}
    recorded = []  # phase14_outputs.json as scripts/ext_eval.py writes it
    for exp, (task, sets) in ps.GLYPHARIS_RUNS.items():
        for eval_set in sets:
            path = data / "processed/phase14" / exp / f"{eval_set}.jsonl"
            if task == "rec":
                rows = _rec_rows(crops[eval_set], "glypharis", glypharis=True)
            else:
                rows = _e2e_rows(pages[eval_set], "glypharis", True, oracle=task == "oracle")
            _jsonl(path, rows)
            _meta(
                path,
                experiment=exp,
                eval_set=eval_set,
                expected=len(rows),
                provenance=provenance,
                pinned=pin,
                settings={"temperature": 2.35},
                inputs=_inputs(data, eval_set, task),
            )
            recorded.append(_recorded(data, path, exp))
    (data / ps.OUTPUTS).parent.mkdir(parents=True, exist_ok=True)
    (data / ps.OUTPUTS).write_text(
        json.dumps({"note": "test", "outputs": sorted(recorded, key=lambda e: e["file"])}),
        encoding="utf-8",
    )
    eff = tmp_path / "reports/external/GW-EFF-005"
    _profile(
        eff / "profile_glypharis_10t.json",
        [40.0 + i % 3 for i in range(100)],
        sizes,
        _glyph_fields(10, pin, provenance),
        **_sitting(30),
    )
    _profile(
        eff / "profile_glypharis_16t.json",
        [30.0] * 100,
        sizes,
        _glyph_fields(16, pin, provenance),
        **_sitting(60),
    )

    # The official evaluators' ground truth, and their manifests.
    gt = data / "raw/hiertext/gt/test.jsonl.gz"
    gt.parent.mkdir(parents=True)
    with gzip.open(gt, "wt", encoding="utf-8") as fh:
        json.dump({"annotations": [{"image_id": p.doc_id} for p in pages["hiertext_test"]]}, fh)
    odb = data / "raw/omnidocbench/OmniDocBench.json"
    odb.parent.mkdir(parents=True)
    odb.write_text(
        json.dumps([{"page_info": {"image_path": f"images/odb_{i}.png"}} for i in range(4)]),
        encoding="utf-8",
    )
    for name, key, path in (("hiertext", "gt_test", gt), ("omnidocbench", "annotations", odb)):
        (data / f"manifests/{name}.json").write_text(
            json.dumps(
                {
                    "files": {
                        key: {
                            "path": path.relative_to(data).as_posix(),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    calls: dict[str, list] = {"hiertext": [], "omnidocbench": []}

    def fake_hiertext(gt_json, pred_json, out_dir):
        pred = json.loads(Path(pred_json).read_text(encoding="utf-8"))
        assert [a["image_id"] for a in pred["annotations"]] == ["ht0", "ht1", "ht2"]
        calls["hiertext"].append(out_dir.name)
        return {"word": {"E2E-Fscore": 0.1}, "line": {"E2E-Fscore": 0.5, "Det-Fscore": 0.9}}

    def fake_omnidocbench(gt_json, pred_dir, work_dir):
        stems = [
            Path(p["page_info"]["image_path"]).stem
            for p in json.loads(Path(gt_json).read_text(encoding="utf-8"))
        ]
        assert stems and all((pred_dir / f"{s}.md").is_file() for s in stems)
        calls["omnidocbench"].append((pred_dir.name, len(stems)))
        el = {"ALL_page_avg": 0.2, "edit_whole": 0.25, "edit_sample_avg": 0.3, "by_attribute": {}}
        return {
            "pages": len(stems),
            "save_name": pred_dir.name,
            "text_block": el,
            "reading_order": el,
            "match_debug": {},
        }

    monkeypatch.setattr(official_hiertext, "run_hiertext", fake_hiertext)
    monkeypatch.setattr(official_omnidocbench, "run_omnidocbench", fake_omnidocbench)
    # The evaluators' code folders and pinned entries under a made-up root; their environments
    # are "present" unless a test removes one.
    evalroot = tmp_path / "evalroot"
    for _name, (_venv, code, inner) in ps.EVALUATORS.items():
        (evalroot / code).mkdir(parents=True)
        if inner is not None:
            (evalroot / code / inner).parent.mkdir(parents=True, exist_ok=True)
            (evalroot / code / inner).write_text("end2end_eval: {}\n", encoding="utf-8")
    (evalroot / official_hiertext.EVALUATORS).parent.mkdir(parents=True)
    (evalroot / official_hiertext.EVALUATORS).write_text(
        json.dumps({n: {"commit": "abc"} for n in ps.EVALUATORS}), encoding="utf-8"
    )
    monkeypatch.setattr(ps, "EVALUATOR_ROOT", evalroot)
    venvs = {v for v, _, _ in ps.EVALUATORS.values()}

    def fake_venv(name, root=None):
        if name not in venvs:
            raise FileNotFoundError(f"evaluator environment {name} not found under fake")
        return Path(name) / "python"

    monkeypatch.setattr(official_hiertext, "venv_python", fake_venv)
    monkeypatch.setattr(ps, "REQUIRE_RECORDED", False)  # the tiny sets have no preparation record
    images = [p.image for p in pages["gdc_test"]]
    monkeypatch.setattr(ps, "gdc_conditions", lambda: dict(zip(images, CONDITIONS, strict=True)))
    monkeypatch.setattr(ps, "gdc_sources", lambda: dict(zip(images, SOURCES, strict=True)))
    monkeypatch.setattr(ps, "require_branch", lambda name, start=None: None)
    monkeypatch.setattr(ps, "pinned_state", lambda: {"trees": "equal", "tag": "phase14-eval"})
    monkeypatch.setattr(ps, "load_phase5_results", lambda tag=ps.PHASE5_TAG: (phase5, "5555"))
    log = tmp_path / "access.md"
    log.write_text(LOG_HEADER + PLACEHOLDER, encoding="utf-8")
    monkeypatch.setattr(ps, "ACCESS_LOG", log)
    out = tmp_path / "reports/external"
    args = [
        "--data-root",
        str(data),
        "--out",
        str(out),
        "--eff-dir",
        str(eff),
        "--record",
        str(record),
    ]
    return {
        "data": data,
        "out": out,
        "log": log,
        "args": args,
        "phase5": phase5,
        "calls": calls,
        "pages": pages,
        "tmp": tmp_path,
        "eff": eff,
        "pin": pin,
        "provenance": provenance,
        "sizes": sizes,
    }


def _rows(log):
    return [ln for ln in log.read_text(encoding="utf-8").splitlines() if "Phase 14 scoring" in ln]


def _score(layout):
    assert ps.main(["hash-phase5", "--data-root", str(layout["data"])]) == 0
    return ps.main(["score", *layout["args"]])


def test_scoring_pass_end_to_end(layout, capsys):
    assert _score(layout) == 0
    out, data = layout["out"], layout["data"]
    # Everything printed survives a cp1252 console (no prime); results.json keeps it.
    capsys.readouterr().out.encode("cp1252")
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert json.loads((data / ps.PASS_RECORD).read_text(encoding="utf-8"))["status"] == "done"
    assert res["provenance"]["pass"]["resumed_utc"] == []
    assert res["provenance"]["converted"] == {"hiertext": 2, "omnidocbench": 3}
    assert set(res["official"]["evaluators"]) == {"hiertext", "omnidocbench"}
    # One row per scored test file: 5 Phase 5 test files, 6 Phase 14 baseline files, 12 GLYPHARIS
    # test files; validation files log nothing.
    rows = _rows(layout["log"])
    assert len(rows) == 5 + 6 + 12
    assert all("| 14 | Phase 14 scoring (GW-E2E-009) |" in r and ", scoring |" in r for r in rows)
    assert sum("glypharis (oracle), scoring" in r for r in rows) == 2
    assert res["reproduction"]["result"] == "equal"
    assert res["reproduction"]["tag_commit"] == "5555"
    h = res["hypotheses"]
    for v in (
        h["H1′ CER"]["verdict"],
        h["H5′"]["verdict"],
        h["H8′"]["transfers"]["verdict"],
        h["H8′"]["by_half"]["verdict"],
        h["H1′ latency"]["verdict"],
    ):
        assert v in ("met", "not met", "not shown")
    assert h["H1′ CER"]["pages"] == 4 and h["H1′ CER"]["documents"] == 2
    assert set(h["H1′ CER"]["verdicts"]) == {"strict_pages", "normalised_pages", "strict_documents"}
    # GLYPHARIS is 40-42 ms against the baseline's 60-64 ms on the same pages.
    assert h["H1′ latency"]["verdict"] == "met"
    assert h["H1′ latency"]["information"]["glypharis_16t_mean_ms"] == 30.0
    assert h["H1′ latency"]["information"]["glypharis_16t_problems"] == []
    assert h["H1′ latency"]["sitting"]["GW-EFF-005"]["started_utc"].startswith("2026-11-16T10:30")
    # GDC-scan without its out-of-charset lines: crops and pages, every system, numbers only.
    assert {r["left_out"] for r in res["gdc_scan_in_charset"]} == {2}
    scan = {(r["experiment"], r["task"]): r for r in res["gdc_scan_in_charset_pages"]}
    assert set(scan) == {("GW-BASE-002", "e2e"), ("GW-E2E-007", "e2e"), ("GW-E2E-008", "oracle")}
    for r in scan.values():
        assert (r["pages"], r["lines"], r["left_out"]) == (2, 4, 2)
        assert {"page_cer_bag_of_lines", "e2e_f1"} <= set(r["tracks"]["strict"])
    full = next(
        e
        for e in res["predictions"]
        if (e["experiment"], e["eval_set"]) == ("GW-E2E-007", "gdc_scan_test")
    )
    # GLYPHARIS reads every line, the é line with an error; leaving it out changes the CER.
    assert (
        scan[("GW-E2E-007", "e2e")]["tracks"]["strict"]["page_cer_bag_of_lines"]["value"]
        != full["tracks"]["strict"]["page_cer_bag_of_lines"]["value"]
    )
    assert "GDC-scan without out-of-charset lines" in (out / "end_to_end.md").read_text("utf-8")
    assert set(h["H8′"]["transfers"]["verdicts"]) == {"lines", "pages"}
    # Every GLYPHARIS and baseline file is scored, the subset view covers every OmniDocBench
    # system, and the block-level set reports n/a for the line measures.
    keys = {(e["experiment"], e["task"], e["eval_set"]) for e in res["predictions"]}
    assert ("GW-REC-038", "rec", "hiertext_val") in keys
    assert ("GW-E2E-008", "oracle", "gdc_scan_test") in keys
    assert ("GW-BASE-002", "rec", "iiit5k_test") in keys
    assert ("GW-BASE-002", "rec", "gdc_test") in keys  # Phase 5 test files are reported
    assert ("GW-BASE-002", "rec", "hiertext_val") not in keys  # their validation ones are not
    sub = {(e["experiment"], e["samples"]) for e in res["subset"]}
    assert sub == {("GW-E2E-007", 2), ("GW-BASE-002", 2), ("GW-BASE-003", 2)}
    odb = next(
        e
        for e in res["predictions"]
        if e["experiment"] == "GW-E2E-007" and e["eval_set"] == "omnidocbench_test"
    )
    assert odb["detection"] is None and odb["tracks"]["strict"]["page_cer_bag_of_lines"] is None
    iiit = next(e for e in res["predictions"] if e["eval_set"] == "iiit5k_test")
    assert "word_accuracy" in iiit["tracks"]["alnum"]
    # Comparisons pair GLYPHARIS with every baseline file of its sets, the subset one included.
    pairs = {(c["baseline"], c["eval_set"], c["subset"]) for c in res["comparisons"]}
    assert ("GW-BASE-003", "omnidocbench_test", SUBSET_NAME) in pairs
    assert ("GW-BASE-001", "gdc_test", None) in pairs
    # Official evaluators: HierText for GLYPHARIS and GW-BASE-002; OmniDocBench all pages for
    # the two full systems, the subset for all three.
    assert sorted(layout["calls"]["hiertext"]) == [
        "GW-BASE-002_ppocr_mobile",
        "GW-E2E-007_glypharis",
    ]
    odb_calls = sorted(layout["calls"]["omnidocbench"])
    assert [n for _, n in odb_calls].count(3) == 2 and [n for _, n in odb_calls].count(2) == 3
    # The hand-off and its hashes.
    handoff = data / ps.HANDOFF
    crops = [
        json.loads(x) for x in (handoff / "gdc_crops/index.jsonl").read_text("utf-8").splitlines()
    ]
    assert {c["kind"] for c in crops} <= {"error", "correct"} and crops
    for c in crops:
        raw = (handoff / "gdc_crops" / c["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == c["sha256"]
    rec_gdc = [
        json.loads(x)
        for x in (handoff / "rec_lines_gdc_test.jsonl").read_text("utf-8").splitlines()
    ]
    assert all(
        {
            "condition",
            "source",
            "det_matched",
            "det_iou",
            "rotation",
            "conf_t",
            "blur",
            "oracle_strict",
        }
        <= set(r)
        for r in rec_gdc
    )
    manifest = json.loads((data / ps.OUTPUTS).read_text(encoding="utf-8"))
    files = {e["file"] for e in manifest["outputs"]}
    assert [e["file"] for e in manifest["outputs"]] == sorted(files)  # the runner's file order
    assert "data/processed/phase14/handoff/gdc_crops/index.jsonl" in files
    assert "data/processed/phase14/GW-REC-039/gdc_test.jsonl" in files
    assert (
        "data/processed/baselines_phase14/GW-BASE-003/ppocr_server_e2e_omnidocbench_test_subset100.jsonl"
        in files
    )
    for name in (
        "README.md",
        "recognition.md",
        "end_to_end.md",
        "official.md",
        "comparisons.md",
        "calibration.md",
        "reliability_gdc_test.png",
    ):
        assert (out / name).is_file(), name


def test_gdc_scan_text_never_reaches_a_report(layout):
    assert _score(layout) == 0
    data = layout["data"]
    checked = [p for p in layout["out"].rglob("*") if p.is_file()]
    checked += [data / ps.OUTPUTS, data / ps.PHASE5_HASHES]
    checked += sorted((data / ps.HANDOFF).glob("*gdc_scan_test*"))
    assert any("gdc_scan_test" in p.name for p in checked)
    for p in checked:
        assert MARKER.encode() not in p.read_bytes(), p.name
        assert MARKER.lower().encode() not in p.read_bytes().lower(), p.name
    # No absolute path reaches a tracked file either.
    tmp = str(layout["tmp"])
    for p in checked:
        if p.suffix in (".json", ".md"):
            text = p.read_text(encoding="utf-8")
            assert tmp not in text and Path(tmp).as_posix() not in text, p.name
    # The marker is in the inputs, so the test would see it if it leaked.
    path = data / "processed/phase14/GW-REC-039/gdc_scan_test.jsonl"
    assert MARKER.encode() in path.read_bytes()


def test_reproduction_difference_stops_the_pass(layout, monkeypatch):
    phase5 = json.loads(json.dumps(layout["phase5"]))
    entry = next(
        e
        for e in phase5["predictions"]
        if e["eval_set"] == "gdc_test" and e["task"] == "rec" and e["system"] == "ppocr_mobile"
    )
    entry["tracks"]["strict"]["cer"]["value"] += 1e-9
    monkeypatch.setattr(ps, "load_phase5_results", lambda tag=ps.PHASE5_TAG: (phase5, "5555"))
    with pytest.raises(SystemExit, match="does not reproduce"):
        _score(layout)
    assert not (layout["out"] / "results.json").exists()
    # Only the Phase 5 re-score was logged: no Phase 14 file was scored.
    assert len(_rows(layout["log"])) == 5
    assert not (layout["data"] / ps.HANDOFF).exists()


def test_stored_reference_differing_from_the_pack_stops_the_pass(layout):
    data = layout["data"]
    spec = EVAL_SETS["gdc_test"]
    rows = [
        (ln.line_id, p.doc_id, ln.text + ("!" if j == 0 else ""))
        for p in layout["pages"]["gdc_test"]
        for j, ln in enumerate(x for x in p.lines if not x.vertical)
    ]
    _pack(data / spec["pack"], rows, "test", "gdc")
    before = (data / ps.OUTPUTS).read_bytes()
    # The GLYPHARIS runs recorded the old pack: void before anything is logged or written.
    with pytest.raises(SystemExit, match="recorded pack index_sha256 differs"):
        _score(layout)
    assert _rows(layout["log"]) == [] and (data / ps.OUTPUTS).read_bytes() == before
    assert not (data / ps.PASS_RECORD).exists()
    # Had they read the new pack, the Phase 5 references would still disagree with it.
    _rerecord(data, restamp=True)
    with pytest.raises(SystemExit, match="stored ref differs"):
        _score(layout)
    assert not (layout["out"] / "results.json").exists()


def test_inputs_are_checked_against_the_preparation_record(layout, monkeypatch):
    monkeypatch.setattr(ps, "REQUIRE_RECORDED", True)  # the tiny sets have no record
    with pytest.raises(SystemExit, match="no recorded hash"):
        _score(layout)
    assert _rows(layout["log"]) == []


def test_changed_phase5_file_is_refused_before_any_score(layout):
    data = layout["data"]
    assert ps.main(["hash-phase5", "--data-root", str(data)]) == 0
    meta = data / "processed/baselines/GW-BASE-002/ppocr_mobile_rec_gdc_test.meta.json"
    meta.write_text(meta.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(SystemExit, match="differ from their recorded hashes"):
        ps.main(["score", *layout["args"]])
    assert _rows(layout["log"]) == []
    # The record itself is never rewritten.
    with pytest.raises(SystemExit, match="never rewritten"):
        ps.main(["hash-phase5", "--data-root", str(data)])


def test_incomplete_prediction_file_is_refused_before_any_score(layout):
    path = layout["data"] / "processed/phase14/GW-E2E-007/funsd_test.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not every prediction file is complete"):
        _score(layout)
    assert _rows(layout["log"]) == []


def test_unpinned_glypharis_run_is_void(layout):
    meta = layout["data"] / "processed/phase14/GW-REC-039/gdc_test.meta.json"
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["pinned"]["trees"] = "not pinned"
    meta.write_text(json.dumps(doc), encoding="utf-8")
    # Changed after its runner recorded it: the metadata hash stops the pass first.
    with pytest.raises(SystemExit, match=r"run metadata of .* differs from its hash"):
        _score(layout)
    _rerecord(layout["data"])  # as the runner recorded it
    with pytest.raises(SystemExit, match="void GLYPHARIS runs"):
        _score(layout)
    assert _rows(layout["log"]) == []


def test_changed_phase14_output_is_refused(layout):
    data = layout["data"]
    before = (data / ps.OUTPUTS).read_bytes()
    assert ps.main(["hash-phase5", "--data-root", str(data)]) == 0
    assert ps.main(["check", *layout["args"]]) == 0
    assert (data / ps.OUTPUTS).read_bytes() == before  # check writes nothing
    path = data / "processed/phase14/GW-REC-039/gdc_test.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")  # the same records, different bytes
    with pytest.raises(SystemExit, match="differs from its hash"):
        _score(layout)
    assert _rows(layout["log"]) == []


def test_glypharis_output_its_runner_did_not_record_is_refused(layout):
    data = layout["data"]
    doc = json.loads((data / ps.OUTPUTS).read_text(encoding="utf-8"))
    oracle = "data/processed/phase14/GW-E2E-008/gdc_test.jsonl"
    doc["outputs"] = [e for e in doc["outputs"] if e["file"] != oracle]
    (data / ps.OUTPUTS).write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="GLYPHARIS outputs are not in"):
        _score(layout)
    assert _rows(layout["log"]) == []


def test_second_pass_is_refused(layout):
    assert _score(layout) == 0
    with pytest.raises(SystemExit, match="one-shot"):
        ps.main(["score", *layout["args"]])


def test_outputs_manifest_layouts(tmp_path):
    data = tmp_path
    f = data / "processed/x.jsonl"
    f.parent.mkdir(parents=True)
    f.write_text('{"key": 1}\n{"key": 2}\n', encoding="utf-8")
    entry = ps.file_entry(f, data)
    # Written repository-relative, as scripts/ext_eval.py writes its entries.
    assert entry["lines"] == 2 and entry["file"] == "data/processed/x.jsonl"
    # Created in the GLYPHARIS runner's layout when absent; a second record is a no-op.
    assert ps.record_outputs(data, [entry]) == 1
    assert ps.record_outputs(data, [entry]) == 0
    doc = json.loads((data / ps.OUTPUTS).read_text(encoding="utf-8"))
    assert doc["outputs"] == [entry] and "note" in doc
    # Entries the runner wrote keep their extra fields; a new one is added in file order.
    first = {**entry, "file": "data/processed/a.jsonl", "experiment": "GW-REC-039"}
    (data / ps.OUTPUTS).write_text(json.dumps({"note": "n", "outputs": [first]}), "utf-8")
    assert ps.record_outputs(data, [entry]) == 1
    doc = json.loads((data / ps.OUTPUTS).read_text(encoding="utf-8"))
    assert doc["outputs"] == [first, entry] and doc["note"] == "n"
    # A recorded bare "processed/..." path is matched too, and a different hash or count stops.
    for changed in ({"sha256": "0"}, {"lines": 3}):
        bad = {**entry, "file": "processed/x.jsonl", **changed}
        (data / ps.OUTPUTS).write_text(json.dumps({"outputs": [bad]}), encoding="utf-8")
        with pytest.raises(SystemExit, match="differs from its hash"):
            ps.record_outputs(data, [entry])
    # A recorded metadata hash is checked too, from its "meta" path or beside the file.
    meta = f.with_suffix(".meta.json")
    meta.write_text("{}", encoding="utf-8")
    good = hashlib.sha256(meta.read_bytes()).hexdigest()
    for extra in ({"meta": "data/processed/x.meta.json"}, {}):
        rec = {**entry, **extra, "meta_sha256": good}
        (data / ps.OUTPUTS).write_text(json.dumps({"outputs": [rec]}), encoding="utf-8")
        assert ps.check_outputs(data, [entry]) == []
        (data / ps.OUTPUTS).write_text(
            json.dumps({"outputs": [{**rec, "meta_sha256": "0"}]}), encoding="utf-8"
        )
        with pytest.raises(SystemExit, match=r"run metadata of data/processed/x.jsonl differs"):
            ps.check_outputs(data, [entry])
    (data / ps.OUTPUTS).write_text(json.dumps([entry]), encoding="utf-8")
    with pytest.raises(SystemExit, match="outputs"):
        ps.record_outputs(data, [entry])


def test_phase5_record_holds_repository_paths(layout):
    data = layout["data"]
    assert ps.main(["hash-phase5", "--data-root", str(data)]) == 0
    doc = json.loads((data / ps.PHASE5_HASHES).read_text(encoding="utf-8"))
    assert doc["files"] and all(e["file"].startswith("data/processed/") for e in doc["files"])
    # A record in the bare form is read as the same files.
    for e in doc["files"]:
        e["file"] = e["file"].removeprefix("data/")
    (data / ps.PHASE5_HASHES).write_text(json.dumps(doc), encoding="utf-8")
    assert ps.main(["hash-phase5", "--data-root", str(data)]) == 0
    assert ps.verify_phase5(data)["files"] == len(doc["files"])


RUNS = {"tag_commit": "1234", **{k: k[:3] for k in ps.MODEL_HASHES}}
PIN = {"tag": "phase14-eval", "tag_commit": "1234", "head": "1234", "trees": "equal"}


def test_latency_sitting_verdicts_and_void_conditions(tmp_path):
    sizes = [[100, 80]] * 100
    paths = {
        k: tmp_path / f"{k}.json"
        for k in ("baseline", "glypharis", "glypharis_16t", "phase5_baseline")
    }
    prov = {k: RUNS[k] for k in ps.MODEL_HASHES}
    base = {"settings": {"cpu_threads": 10}}
    _profile(paths["baseline"], [60.0] * 100, sizes, base, **_sitting(0))
    glyph = _sitting(30)

    def glypharis(times, fields=None, sitting=None):
        _profile(paths["glypharis"], times, sizes, fields or _glyph_fields(10, PIN, prov),
                 **(glyph if sitting is None else sitting))  # fmt: skip

    glypharis([40.0] * 100)
    assert ps.latency_void(paths, RUNS) == []
    assert ps.h1_latency(paths, RUNS)["verdict"] == "met"
    glypharis([40.0, 80.0] * 50)
    res = ps.h1_latency(paths, RUNS)
    assert res["verdict"] == "not shown" and res["test"]["value"] == pytest.approx(0.0)
    assert len(res["times_ms"]["GW-EFF-005"]) == 100

    def void_with(change, field=None):
        g = _glyph_fields(10, PIN, prov)
        change(g)
        _profile(paths["glypharis"], [40.0] * 100, sizes if field is None else field, g, **glyph)
        res = ps.h1_latency(paths, RUNS)
        assert res["verdict"] == "void"
        return " / ".join(res["void"])

    assert "input sizes differ" in void_with(lambda g: None, [[101, 80], *sizes[1:]])
    assert "16 threads" in void_with(lambda g: g["settings"].update(cpu_threads=16))
    assert "torch pools" in void_with(lambda g: g["threads"].update(inter_op=1))
    # A bool is not a thread count, even where Python treats it as an int.
    assert "None threads" in void_with(lambda g: g["settings"].update(cpu_threads=True))
    assert "before torch was imported" in void_with(
        lambda g: g["threads"].update(env_before_torch_import=False)
    )
    assert "pinned code" in void_with(lambda g: g.update(pinned={"error": "no tag"}))
    assert "another tag commit" in void_with(lambda g: g["pinned"].update(tag_commit="9999"))
    assert "recognizer_checkpoint_sha256" in void_with(
        lambda g: g["provenance"].update(recognizer_checkpoint_sha256="zzz")
    )

    # One sitting, the baseline first: GW-EFF-005 starts after the baseline ended, within 2 hours.
    def sitting_void(sitting):
        glypharis([40.0] * 100, sitting=sitting)
        return " / ".join(ps.latency_void(paths, RUNS))

    assert sitting_void(_sitting(20)) == ""  # straight after the baseline: fine
    assert sitting_void(_sitting(20 + 120)) == ""  # exactly the gap allowed
    assert "the baseline runs first" in sitting_void(_sitting(10))
    assert "not one sitting" in sitting_void(_sitting(20 + 121))
    assert "not one sitting" in sitting_void(_sitting(3 * 24 * 60))
    assert "not recorded" in sitting_void({"started_utc": glyph["started_utc"]})
    assert "not recorded" in sitting_void({**glyph, "started_utc": "2026-11-16T10:30:00"})
    assert "not recorded" in sitting_void({**glyph, "ended_utc": "yesterday"})
    assert "ended_utc is before started_utc" in sitting_void(
        {"started_utc": glyph["ended_utc"], "ended_utc": glyph["started_utc"]}
    )
    _profile(paths["baseline"], [60.0] * 100, sizes, base)  # no times at all
    assert "GW-BASE-002: started_utc and ended_utc" in sitting_void(glyph)
    _profile(paths["baseline"], [60.0] * 100, sizes, base, **_sitting(0))

    # The 16-thread file is information: its problems are reported beside it, never a void.
    glypharis([40.0] * 100)
    _profile(paths["glypharis_16t"], [30.0] * 100, sizes, _glyph_fields(16, {"error": "x"}, prov))
    assert ps.latency_void(paths, RUNS) == []
    res = ps.h1_latency(paths, RUNS)
    assert res["verdict"] == "met"
    assert res["information"]["glypharis_16t_problems"] == [
        "GW-EFF-005 (16 threads): not run at the pinned code"
    ]
    assert res["information"]["glypharis_16t_mean_ms"] == 30.0


def test_a_void_latency_sitting_is_refused_before_any_score(layout):
    data = layout["data"]
    path = layout["eff"] / "profile_glypharis_10t.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["conditions"]["on_ac_power"] = False
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert ps.main(["hash-phase5", "--data-root", str(data)]) == 0
    before = (data / ps.OUTPUTS).read_bytes()
    with pytest.raises(SystemExit, match="latency sitting is void"):
        ps.main(["check", *layout["args"]])
    with pytest.raises(SystemExit, match="GW-EFF-005: not on mains power"):
        ps.main(["score", *layout["args"]])
    assert _rows(layout["log"]) == []
    assert not (layout["out"] / "results.json").exists()
    # Nothing was written: the baseline profile is not recorded, so the repeat sitting (new
    # bytes at the same path) is accepted.
    assert (data / ps.OUTPUTS).read_bytes() == before
    assert not (data / ps.PASS_RECORD).exists()
    base = run_baselines.output_path(
        data, "GW-BASE-002", "ppocr_mobile", "e2e", "hiertext_val", 14
    ).with_name("profile_ppocr_mobile_e2e_hiertext_val.json")
    repeat = json.loads(base.read_text(encoding="utf-8"))
    repeat.update(times_ms=[61.0 + i % 4 for i in range(100)], **_sitting(24 * 60))
    base.write_text(json.dumps(repeat), encoding="utf-8")
    doc["conditions"]["on_ac_power"] = True
    doc.update(_sitting(24 * 60 + 30))
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert ps.main(["score", *layout["args"]]) == 0
    files = {e["file"] for e in json.loads((data / ps.OUTPUTS).read_text("utf-8"))["outputs"]}
    assert ps.data_rel(base, data) in files


def test_glypharis_runs_on_different_hardware_are_void(layout):
    meta = layout["data"] / "processed/phase14/GW-REC-038/hiertext_val.meta.json"
    doc = json.loads(meta.read_text(encoding="utf-8"))
    doc["provenance"]["device_name"] = "cpu"
    meta.write_text(json.dumps(doc), encoding="utf-8")
    _rerecord(layout["data"])  # as its runner recorded it
    with pytest.raises(SystemExit, match="GLYPHARIS runs differ in device"):
        _score(layout)
    assert _rows(layout["log"]) == []


def test_a_glypharis_run_whose_inputs_changed_is_void(layout):
    data = layout["data"]
    # The same pages in another order: other bytes, the same keys.
    write_pages(data / EVAL_SETS["funsd_test"]["pages"], layout["pages"]["funsd_test"][::-1])
    with pytest.raises(SystemExit, match="recorded pages sha256 differs from data/processed"):
        _score(layout)
    assert _rows(layout["log"]) == []
    # A run without an inputs block is void too.
    _rerecord(data, restamp=True)
    meta = data / "processed/phase14/GW-REC-039/iiit5k_test.meta.json"
    doc = json.loads(meta.read_text(encoding="utf-8"))
    del doc["inputs"]
    meta.write_text(json.dumps(doc), encoding="utf-8")
    _rerecord(data)
    with pytest.raises(SystemExit, match="GW-REC-039 glypharis rec iiit5k_test: recorded pack"):
        _score(layout)
    assert _rows(layout["log"]) == []


def test_official_evaluator_problems_are_refused_before_any_score(layout, monkeypatch):
    data = layout["data"]
    before = (data / ps.OUTPUTS).read_bytes()

    def refused(match):
        with pytest.raises(SystemExit, match=match):
            _score(layout)
        assert _rows(layout["log"]) == [] and (data / ps.OUTPUTS).read_bytes() == before
        assert not (data / ps.PASS_RECORD).exists()

    # A missing environment.
    real = official_hiertext.venv_python

    def no_omnidocbench(name, root=None):
        if name == official_omnidocbench.OMNIDOCBENCH_VENV:
            raise FileNotFoundError(f"evaluator environment {name} not found under fake")
        return real(name, root)

    monkeypatch.setattr(official_hiertext, "venv_python", no_omnidocbench)
    refused("omnidocbench: evaluator environment .venv-omnidocbench not found")
    monkeypatch.setattr(official_hiertext, "venv_python", real)
    # Missing evaluator code.
    code = ps.EVALUATOR_ROOT / official_hiertext.HIERTEXT_DIR
    code.rename(code.with_name("moved"))
    refused("hiertext: evaluator code .* is missing")
    code.with_name("moved").rename(code)
    # Ground truth that differs from its manifest hash.
    gt = data / "raw/omnidocbench/OmniDocBench.json"
    gt.write_bytes(gt.read_bytes() + b" ")
    refused("omnidocbench: raw/omnidocbench/OmniDocBench.json differs from its hash")


def test_a_polygon_the_evaluator_cannot_read_is_refused_before_any_score(layout):
    data = layout["data"]
    path = data / "processed/phase14/GW-E2E-007/hiertext_test.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["lines"][0]["polygon"] = [[0, 0], [10, 10]]  # two points
    _jsonl(path, rows)
    _rerecord(data)  # as its runner recorded it
    before = (data / ps.OUTPUTS).read_bytes()
    with pytest.raises(SystemExit, match=r"(?s)cannot be converted.*at least 3 points"):
        _score(layout)
    assert _rows(layout["log"]) == [] and (data / ps.OUTPUTS).read_bytes() == before
    with pytest.raises(SystemExit, match="cannot be converted"):
        ps.main(["check", *layout["args"]])
    # The temporary conversion folder is gone.
    assert not (data / ps.GLYPHARIS_ROOT / ps.EXPERIMENT).exists()


def test_a_stopped_pass_resumes_as_one_evaluation(layout, monkeypatch):
    data = layout["data"]
    fake = official_hiertext.run_hiertext

    def crash(gt_json, pred_json, out_dir):
        raise RuntimeError("evaluator exited with 1")

    monkeypatch.setattr(official_hiertext, "run_hiertext", crash)
    with pytest.raises(RuntimeError, match="evaluator exited"):
        _score(layout)
    first = _rows(layout["log"])
    assert len(first) == 5 + 6 + 12 and not any(ps.RESUMED in r for r in first)
    record = json.loads((data / ps.PASS_RECORD).read_text(encoding="utf-8"))
    assert record["status"] == "started" and record["reason"] == ps.REASON
    assert not (layout["out"] / "results.json").exists()
    # A plain score refuses: the stopped pass is finished with --resume, not started again.
    monkeypatch.setattr(official_hiertext, "run_hiertext", fake)
    with pytest.raises(SystemExit, match="did not finish: run `score --resume`"):
        ps.main(["score", *layout["args"]])
    assert len(_rows(layout["log"])) == len(first)
    (data / ps.OFFICIAL / "stale.txt").write_text("left by the stopped run", encoding="utf-8")
    assert ps.main(["score", "--resume", *layout["args"]]) == 0
    resumed = _rows(layout["log"])[len(first) :]
    assert len(resumed) == len(first)
    assert all("| 14 | Phase 14 scoring (GW-E2E-009) (resumed) |" in r for r in resumed)
    assert not (data / ps.OFFICIAL / "stale.txt").exists()  # the folder it owns was cleared
    record = json.loads((data / ps.PASS_RECORD).read_text(encoding="utf-8"))
    assert record["status"] == "done" and len(record["resumed_utc"]) == 1
    res = json.loads((layout["out"] / "results.json").read_text(encoding="utf-8"))
    assert res["provenance"]["access_log_reason"] == ps.REASON + ps.RESUMED
    # Done is done: neither a plain score nor a resume runs again.
    for extra in ([], ["--resume"]):
        with pytest.raises(SystemExit, match="one-shot"):
            ps.main(["score", *extra, *layout["args"]])
    (layout["out"] / "results.json").unlink()
    with pytest.raises(SystemExit, match="records a finished pass"):
        ps.main(["score", "--resume", *layout["args"]])


def test_resume_needs_a_stopped_pass_and_refuses_a_void_sitting_first(layout):
    data = layout["data"]
    with pytest.raises(SystemExit, match="no pass to resume"):
        ps.main(["score", "--resume", *layout["args"]])
    # A stopped pass whose hand-off was recorded: a resume records the hand-off afresh.
    ps.write_pass(data, {"status": "started", "started_utc": "2026-11-16T12:00:00+00:00"})
    doc = json.loads((data / ps.OUTPUTS).read_text(encoding="utf-8"))
    stale = {"file": "data/processed/phase14/handoff/rec_lines_gdc_test.jsonl",
             "sha256": "0", "lines": 1}  # fmt: skip
    doc["outputs"] = sorted([*doc["outputs"], stale], key=lambda e: e["file"])
    (data / ps.OUTPUTS).write_text(json.dumps(doc), encoding="utf-8")
    # Preflight still comes first: a void sitting refuses the resume with nothing cleared.
    (data / ps.HANDOFF).mkdir(parents=True)
    (data / ps.HANDOFF / "x.jsonl").write_text("{}\n", encoding="utf-8")
    path = layout["eff"] / "profile_glypharis_10t.json"
    good = path.read_text(encoding="utf-8")
    path.write_text(json.dumps({**json.loads(good), **_sitting(-30)}), encoding="utf-8")
    with pytest.raises(SystemExit, match="the baseline runs first"):
        _score_resume(layout)
    assert (data / ps.HANDOFF / "x.jsonl").exists() and _rows(layout["log"]) == []
    path.write_text(good, encoding="utf-8")
    assert _score_resume(layout) == 0
    files = {e["file"] for e in json.loads((data / ps.OUTPUTS).read_text("utf-8"))["outputs"]}
    assert stale["file"] in files and not (data / ps.HANDOFF / "x.jsonl").exists()
    entry = next(
        e
        for e in json.loads((data / ps.OUTPUTS).read_text("utf-8"))["outputs"]
        if e["file"] == stale["file"]
    )
    assert entry["sha256"] != "0"
    res = json.loads((layout["out"] / "results.json").read_text(encoding="utf-8"))
    assert res["provenance"]["pass"]["started_utc"] == "2026-11-16T12:00:00+00:00"


def _score_resume(layout):
    assert ps.main(["hash-phase5", "--data-root", str(layout["data"])]) == 0
    return ps.main(["score", "--resume", *layout["args"]])


def test_combined_verdicts_and_rotation():
    assert ps.combined(["met", "met", "met"]) == "met"
    assert ps.combined(["met", "not shown", "met"]) == "not shown"
    assert ps.combined(["not met", "not met"]) == "not met"
    assert ps.box_rotation(_box(0, 0, 100, 20)) == 0.0
    assert ps.box_rotation([[0, 0], [20, 0], [20, 100], [0, 100]]) == 90.0
    assert ps.box_rotation([[0, 0], [1, 1]]) is None
    iv = ps.iv(Interval(-0.1, -0.2, -0.05, 1000, "pages"))
    assert ps._interval(iv) == Interval(-0.1, -0.2, -0.05, 1000, "pages")


def test_match_page_follows_bag_of_lines():
    page = _page("p", "x.png", ["alpha beta", "gamma delta", "theta"], "gdc")
    preds = [
        PredLine(page.lines[0].polygon, "alpha betx"),
        PredLine(_box(0, 150, 50, 170), "spurious"),
    ]
    m = ps.match_page(page, preds)
    assert m.status == ["matched", "spurious"]
    assert set(m.matched) == {0} and m.matched[0][1] == pytest.approx(1.0)
    r = evaluate_page(page, preds)
    misses = sum(len(ln.text) for i, ln in enumerate(m.box_lines) if i not in m.matched)
    assert r.bag_errors == 1 + len("spurious") + misses
    assert np.isclose(m.best_iou[1], 0.0)


def test_hash_phase5_is_a_before_the_tag_step(layout, monkeypatch):
    monkeypatch.setattr(ps, "tag_exists", lambda *a, **k: True)
    with pytest.raises(SystemExit, match="tag exists"):
        ps.main(["hash-phase5", "--data-root", str(layout["data"])])
    assert not (layout["data"] / ps.PHASE5_HASHES).exists()


def test_a_missing_access_log_is_refused_before_the_pass_starts(layout, monkeypatch):
    monkeypatch.setattr(ps, "ACCESS_LOG", layout["tmp"] / "nowhere.md")
    assert ps.main(["hash-phase5", "--data-root", str(layout["data"])]) == 0
    with pytest.raises(SystemExit, match="does not exist"):
        ps.main(["score", *layout["args"]])
    assert ps.read_pass(layout["data"]) is None


@pytest.mark.parametrize("text", ['{"truncated": ', '{"settings": {}}', '{"times_ms": ["x"]}'])
def test_an_unreadable_16_thread_profile_is_refused_before_any_row(layout, text):
    (layout["eff"] / "profile_glypharis_16t.json").write_text(text, encoding="utf-8")
    assert ps.main(["hash-phase5", "--data-root", str(layout["data"])]) == 0
    with pytest.raises(SystemExit, match="not a readable profile"):
        ps.main(["score", *layout["args"]])
    assert _rows(layout["log"]) == [] and ps.read_pass(layout["data"]) is None
