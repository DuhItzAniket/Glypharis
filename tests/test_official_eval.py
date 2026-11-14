"""Tests for the official HierText and OmniDocBench evaluator wrappers (Phase 14).

Everything here is made up: tiny synthetic pages and predictions in tmp folders. The runner tests
call the real pinned evaluators in their own environments and are skipped where those environments
do not exist (CI has none).
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import official_hiertext as oh
import official_omnidocbench as oo
import pytest
import yaml

needs_hiertext = pytest.mark.skipif(
    not oh.venv_available(oh.HIERTEXT_VENV), reason="no .venv-hiertext"
)
needs_omnidocbench = pytest.mark.skipif(
    not oh.venv_available(oo.OMNIDOCBENCH_VENV), reason="no .venv-omnidocbench"
)


def box(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# ----------------------------------------------------------------------------- HierText


def hiertext_gt() -> dict[str, Any]:
    def word(b: list[list[float]], text: str) -> dict[str, Any]:
        return {"vertices": b, "text": text, "legible": True, "vertical": False}

    lines = [
        {
            "vertices": box(10, 10, 190, 30),
            "text": "Hello world",
            "legible": True,
            "words": [word(box(10, 10, 90, 30), "Hello"), word(box(100, 10, 190, 30), "world")],
        },
        {
            "vertices": box(10, 50, 120, 70),
            "text": "Next",
            "legible": True,
            "words": [word(box(10, 50, 120, 70), "Next")],
        },
    ]
    para = {"vertices": box(5, 5, 195, 95), "legible": True, "lines": lines}
    page = {"image_id": "img1", "image_width": 200, "image_height": 100, "paragraphs": [para]}
    return {"annotations": [page]}


def test_hiertext_predictions_one_paragraph_one_word_per_line():
    recs = [
        {
            "key": "a",
            "lines": [
                {"polygon": box(10.4, 9.6, 190.2, 30), "text": "Hello world", "confidence": 0.9},
                {"polygon": box(1, 2, 3, 4), "text": ""},
            ],
            "time_ms": 3.0,
        },
        {"key": "b", "lines": []},
    ]
    out = oh.to_hiertext_predictions(recs)
    a, b = out["annotations"]
    assert a["image_id"] == "a" and len(a["paragraphs"]) == 1
    first, second = a["paragraphs"][0]["lines"]
    assert first == {
        "text": "Hello world",
        "words": [{"vertices": [[10, 10], [190, 10], [190, 30], [10, 30]], "text": "Hello world"}],
    }
    assert all(isinstance(v, int) for pt in first["words"][0]["vertices"] for v in pt)
    assert second["text"] == "" and second["words"][0]["text"] == ""
    assert b == {"image_id": "b", "paragraphs": []}


def test_hiertext_predictions_pad_to_ground_truth_and_refuse_strays():
    recs = [{"key": "b", "lines": [{"polygon": box(0, 0, 5, 5), "text": "x"}]}]
    out = oh.to_hiertext_predictions(recs, image_ids=["a", "b", "c"])
    assert [a["image_id"] for a in out["annotations"]] == ["a", "b", "c"]
    assert out["annotations"][0]["paragraphs"] == []
    with pytest.raises(ValueError, match="not in the ground truth"):
        oh.to_hiertext_predictions(recs, image_ids=["a"])


def test_hiertext_predictions_refuse_duplicates_and_degenerate_polygons():
    line = {"polygon": box(0, 0, 5, 5), "text": "x"}
    with pytest.raises(ValueError, match="twice"):
        oh.to_hiertext_predictions([{"key": "a", "lines": [line]}, {"key": "a", "lines": []}])
    with pytest.raises(ValueError, match="at least 3 points"):
        oh.to_hiertext_predictions([{"key": "a", "lines": [{"polygon": [[0, 0], [5, 5]]}]}])


def test_parse_scores_reads_every_block():
    text = (
        "========= line =========\nDet-Fscore: 1.0\nE2E-Fscore: 0.5\n\n"
        "========= word =========\nDet-Fscore: 0.25\nE2E-PQ: 0.0\n"
    )
    assert oh.parse_scores(text) == {
        "line": {"Det-Fscore": 1.0, "E2E-Fscore": 0.5},
        "word": {"Det-Fscore": 0.25, "E2E-PQ": 0.0},
    }
    with pytest.raises(ValueError, match="unexpected line"):
        oh.parse_scores("Det-Fscore: 1.0")


def test_image_ids_reads_plain_and_gzipped_ground_truth(tmp_path):
    gt = hiertext_gt()
    (tmp_path / "gt.json").write_text(json.dumps(gt), encoding="utf-8")
    with gzip.open(tmp_path / "gt.json.gz", "wt", encoding="utf-8") as fh:
        json.dump(gt, fh)
    assert oh.image_ids(tmp_path / "gt.json") == ["img1"]
    assert oh.image_ids(tmp_path / "gt.json.gz") == ["img1"]


def test_run_hiertext_refuses_before_running(tmp_path, monkeypatch):
    def no_venv(name: str, root: Path | None = None) -> Path:
        raise AssertionError("the evaluator must not be reached")

    monkeypatch.setattr(oh, "venv_python", no_venv)
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(hiertext_gt()), encoding="utf-8")
    pred = tmp_path / "pred.json"
    oh.write_predictions({"annotations": []}, pred)
    with pytest.raises(ValueError, match="1 pages missing"):
        oh.run_hiertext(gt, pred, tmp_path / "out")
    oh.write_predictions(oh.to_hiertext_predictions([], image_ids=["img1", "zzz"]), pred)
    with pytest.raises(ValueError, match="1 not in the ground truth"):
        oh.run_hiertext(gt, pred, tmp_path / "out")
    oh.write_predictions(oh.to_hiertext_predictions([], image_ids=["img1"]), pred)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "scores-00000-of-00001").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError):
        oh.run_hiertext(gt, pred, tmp_path / "out")


def test_venv_python_honours_the_override(tmp_path, monkeypatch):
    monkeypatch.setenv(oh.VENV_ROOT_ENV, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        oh.venv_python(".venv-x")
    assert not oh.venv_available(".venv-x")
    for sub in ("Scripts/python.exe", "bin/python"):
        (tmp_path / ".venv-x" / sub).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / ".venv-x" / sub).write_text("", encoding="utf-8")
    assert oh.venv_python(".venv-x").parent.parent == tmp_path / ".venv-x"


def test_main_checkout_of_a_worktree_is_the_main_folder(tmp_path):
    main = tmp_path / "main"
    main.mkdir()

    def git(*args: str, cwd: Path = main) -> None:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q",
        "--allow-empty", "-m", "init")  # fmt: skip
    git("worktree", "add", "-q", str(tmp_path / "wt"))
    assert oh.main_checkout(main).resolve() == main.resolve()
    assert oh.main_checkout(tmp_path / "wt").resolve() == main.resolve()


def test_evaluator_folders_match_the_pinned_commits():
    hiertext = oh.evaluator_info("hiertext")
    odb = oh.evaluator_info("omnidocbench")
    assert hiertext["commit"].startswith(oh.HIERTEXT_DIR.name.split("-")[-1])
    assert odb["commit"].startswith(oo.OMNIDOCBENCH_DIR.name.split("-")[-1])
    with pytest.raises(KeyError):
        oh.evaluator_info("nope")


@needs_hiertext
def test_run_hiertext_on_a_made_up_page(tmp_path):
    gt = tmp_path / "gt.json.gz"
    with gzip.open(gt, "wt", encoding="utf-8") as fh:
        json.dump(hiertext_gt(), fh)
    recs = [
        {
            "key": "img1",
            "lines": [
                {"polygon": box(10, 10, 190, 30), "text": "Hello world"},
                {"polygon": box(10, 50, 120, 70), "text": "Nxt"},
            ],
        }
    ]
    pred = tmp_path / "pred.json"
    oh.write_predictions(oh.to_hiertext_predictions(recs, image_ids=oh.image_ids(gt)), pred)
    scores = oh.run_hiertext(gt, pred, tmp_path / "out")
    line = scores["line"]
    # Both lines are found; one of the two texts is right.
    assert line["Det-Recall"] == 1.0 and line["Det-Precision"] == 1.0
    assert line["E2E-Recall"] == 0.5 and line["E2E-Precision"] == 0.5
    assert 0.9 < line["Det-Tightness"] <= 1.0
    assert "Det-Fscore" in scores["word"]
    assert not (tmp_path / "out" / "_gt_unpacked.json").exists()
    assert list((tmp_path / "out").glob("scores-*-of-*"))


# ----------------------------------------------------------------------------- OmniDocBench


def odb_page(image: str, texts: list[str]) -> dict[str, Any]:
    dets = []
    for order, text in enumerate(texts, start=1):
        y = 100 * order
        dets.append(
            {
                "category_type": "text_block",
                "poly": [10, y, 500, y, 500, y + 40, 10, y + 40],
                "ignore": False,
                "order": order,
                "anno_id": order,
                "text": text,
                "attribute": {
                    "text_language": "text_english",
                    "text_background": "white",
                    "text_rotate": "normal",
                },
            }
        )
    info = {
        "page_attribute": {"data_source": "book", "language": "english", "layout": "single_column"},
        "page_no": 0,
        "height": 800,
        "width": 600,
        "image_path": image,
    }
    return {"layout_dets": dets, "extra": {"relation": []}, "page_info": info}


def test_to_markdown_one_paragraph_per_line():
    assert oo.to_markdown(["First line", " second ", "", "a\nb", "# not escaped"]) == (
        "First line\n\nsecond\n\na b\n\n# not escaped\n"
    )
    assert oo.to_markdown([]) == "\n"


def test_page_stem_from_key_or_image():
    assert oo.page_stem({"key": "odb-doc.pdf_3"}) == "doc.pdf_3"
    assert oo.page_stem({"key": "x", "image": "raw/omnidocbench/images/doc.pdf_3.jpg"}) == (
        "doc.pdf_3"
    )
    with pytest.raises(ValueError, match="not an OmniDocBench page key"):
        oo.page_stem({"key": "img1"})


def test_write_predictions_one_file_per_page(tmp_path):
    recs = [
        {"key": "odb-doc.pdf_3", "lines": [{"text": "B"}, {"text": "A", "polygon": []}]},
        {"key": "odb-empty", "lines": []},
    ]
    out = tmp_path / "pred"
    oo.write_predictions(recs, out)
    assert sorted(p.name for p in out.iterdir()) == ["doc.pdf_3.md", "empty.md"]
    assert (out / "doc.pdf_3.md").read_text(encoding="utf-8") == "B\n\nA\n"
    assert (out / "empty.md").read_text(encoding="utf-8") == "\n"
    with pytest.raises(FileExistsError):
        oo.write_predictions(recs, out)
    with pytest.raises(ValueError, match="twice"):
        oo.write_predictions(recs + recs[:1], tmp_path / "pred2")


def test_filtered_ground_truth_keeps_only_the_given_pages(tmp_path):
    pages = [odb_page("a.jpg", ["x"]), odb_page("b.pdf_2.png", ["y"]), odb_page("c.jpg", ["z"])]
    src = tmp_path / "all.json"
    src.write_text(json.dumps(pages), encoding="utf-8")
    oo.filtered_ground_truth(src, {"c", "b.pdf_2"}, tmp_path / "gt.json")
    kept = json.loads((tmp_path / "gt.json").read_text(encoding="utf-8"))
    assert [p["page_info"]["image_path"] for p in kept] == ["b.pdf_2.png", "c.jpg"]
    assert kept[0] == pages[1]
    with pytest.raises(ValueError, match="1 pages are not in the ground truth"):
        oo.filtered_ground_truth(src, {"a", "missing"}, tmp_path / "gt2.json")


def fake_evaluator(root: Path) -> Path:
    """A folder laid out like the pinned evaluator, holding its real pinned config."""
    code = root / oo.OMNIDOCBENCH_DIR
    (code / "configs").mkdir(parents=True)
    (code / "result").mkdir()
    shutil.copy(oh.ROOT / oo.OMNIDOCBENCH_DIR / oo.PIPELINE_CONFIG, code / oo.PIPELINE_CONFIG)
    return code


def test_write_config_switches_off_formulas_and_tables(tmp_path):
    code = fake_evaluator(tmp_path / "repo")
    (tmp_path / "pred").mkdir()
    cfg_path = oo.write_config(tmp_path / "gt.json", tmp_path / "pred", tmp_path / "work", code)
    text = cfg_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in text and str(tmp_path.as_posix()) not in text
    cfg = yaml.safe_load(text)["end2end_eval"]
    assert list(cfg["metrics"]) == ["text_block", "reading_order"]
    assert all(m == {"metric": ["Edit_dist"]} for m in cfg["metrics"].values())
    ds = cfg["dataset"]
    assert ds["match_method"] == "quick_match" and ds["match_workers"] == 4
    assert (code / ds["ground_truth"]["data_path"]).resolve() == (tmp_path / "gt.json").resolve()
    assert (code / ds["prediction"]["data_path"]).resolve() == (tmp_path / "pred").resolve()
    pinned = yaml.safe_load((code / oo.PIPELINE_CONFIG).read_text(encoding="utf-8"))
    assert ds["match_timeout_sec"] == pinned["end2end_eval"]["dataset"]["match_timeout_sec"]


def write_results(result_dir: Path, name: str) -> None:
    metric = {
        "text_block": {
            "all": {"Edit_dist": {"ALL_page_avg": 0.1, "edit_whole": 0.2, "edit_sample_avg": 0.3}},
            "page": {"Edit_dist": {"ALL": 0.1}},
        },
        "reading_order": {
            "all": {"Edit_dist": {"ALL_page_avg": 0.5, "edit_whole": 0.6, "edit_sample_avg": 0.5}}
        },
        "match_debug": {"page_count": 2},
    }
    (result_dir / f"{name}_metric_result.json").write_text(json.dumps(metric), encoding="utf-8")
    per_page = {"a.jpg": 0.0, "b.jpg": 0.2}
    (result_dir / f"{name}_text_block_per_page_edit.json").write_text(
        json.dumps(per_page), encoding="utf-8"
    )


def test_collect_and_parse_results(tmp_path):
    code = fake_evaluator(tmp_path / "repo")
    write_results(code / "result", "mine_quick_match")
    write_results(code / "result", "other_quick_match")
    moved = oo.collect_results(code, "mine_quick_match", tmp_path / "work")
    assert len(moved) == 2 and all(m.startswith("mine_quick_match_") for m in moved)
    left = sorted(p.name for p in (code / "result").iterdir())
    assert left and all(n.startswith("other_") for n in left)
    res = oo.parse_results(tmp_path / "work" / "result", "mine_quick_match")
    assert res["text_block"]["ALL_page_avg"] == 0.1
    assert res["text_block"]["edit_whole"] == 0.2
    assert res["text_block"]["per_page"] == {"a.jpg": 0.0, "b.jpg": 0.2}
    assert res["text_block"]["by_attribute"] == {"ALL": 0.1}
    assert res["reading_order"]["edit_sample_avg"] == 0.5
    assert "per_page" not in res["reading_order"]
    assert res["match_debug"] == {"page_count": 2}


def test_run_omnidocbench_refuses_before_running(tmp_path, monkeypatch):
    def no_venv(name: str, root: Path | None = None) -> Path:
        raise AssertionError("the evaluator must not be reached")

    monkeypatch.setattr(oo, "venv_python", no_venv)
    monkeypatch.setattr(oo, "ROOT", tmp_path / "repo")
    code = fake_evaluator(tmp_path / "repo")
    gt = tmp_path / "gt.json"
    gt.write_text(
        json.dumps([odb_page("a.jpg", ["x"]), odb_page("b.jpg", ["y"])]), encoding="utf-8"
    )
    pred = tmp_path / "preds"
    oo.write_predictions([{"key": "odb-a", "lines": [{"text": "x"}]}], pred)
    with pytest.raises(ValueError, match="1 ground-truth pages have no prediction"):
        oo.run_omnidocbench(gt, pred, tmp_path / "work")
    (pred / "b.md").write_text("y\n", encoding="utf-8")
    write_results(code / "result", oo.save_name(pred))
    with pytest.raises(FileExistsError):
        oo.run_omnidocbench(gt, pred, tmp_path / "work")
    gt.write_text(json.dumps([odb_page("a.jpeg", ["x"])]), encoding="utf-8")
    with pytest.raises(ValueError, match="three-letter extension"):
        oo.run_omnidocbench(gt, pred, tmp_path / "work")


@needs_omnidocbench
def test_run_omnidocbench_on_made_up_pages(tmp_path):
    src = tmp_path / "all.json"
    pages = [
        odb_page("synth_a.pdf_1.jpg", ["The first paragraph of text.", "A second one follows."]),
        odb_page("synth_b.jpg", ["Only block on page b."]),
        odb_page("other.jpg", ["Not kept."]),
    ]
    src.write_text(json.dumps(pages), encoding="utf-8")
    gt = tmp_path / "gt.json"
    oo.filtered_ground_truth(src, {"synth_a.pdf_1", "synth_b"}, gt)
    recs = [
        # Right text, wrong order on page a; page b exact.
        {
            "key": "odb-synth_a.pdf_1",
            "lines": [{"text": "A second one follows."}, {"text": "The first paragraph of text."}],
        },
        {"key": "odb-synth_b", "lines": [{"text": "Only block on page b."}]},
    ]
    pred = tmp_path / f"a4test_{tmp_path.name}"
    oo.write_predictions(recs, pred)
    res = oo.run_omnidocbench(gt, pred, tmp_path / "work")
    assert res["pages"] == 2
    assert res["text_block"]["ALL_page_avg"] == 0.0
    assert res["reading_order"]["per_page"] == {"synth_a.pdf_1.jpg": 1.0, "synth_b.jpg": 0.0}
    assert res["reading_order"]["ALL_page_avg"] == 0.5
    assert res["match_debug"]["text_match_fallback_counts"] == {
        "quick_match_timeout": 0,
        "page_timeout": 0,
    }
    code = oh.ROOT / oo.OMNIDOCBENCH_DIR
    assert not list((code / "result").glob(f"{res['save_name']}_*"))
    assert (tmp_path / "work" / "result" / f"{res['save_name']}_metric_result.json").is_file()
