from __future__ import annotations

import gzip
import io
import json

import numpy as np
import prepare_benchmarks as pb
import pytest
from PIL import Image

from glypharis.data.linepack import LinePackReader
from glypharis.data.pages import read_pages


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (w, h), 255).save(buf, format="PNG")
    return buf.getvalue()


def word(text, x0, y0, x1, y1):
    return {"text": text, "box": [x0, y0, x1, y1]}


def test_group_lines_splits_by_vertical_overlap_and_orders_left_to_right():
    words = [
        word("world", 60, 10, 100, 30),
        word("hello", 0, 12, 50, 32),
        word("next", 0, 40, 40, 60),
        word("", 50, 40, 60, 60),
    ]
    lines = pb.group_lines(words)
    assert [[w["text"] for w in ln] for ln in lines] == [["hello", "world"], ["next"]]


def test_funsd_page_builds_lines_from_entity_words():
    ann = {
        "form": [
            {"id": 3, "words": [word("Date:", 10, 10, 60, 30), word("1998", 70, 11, 110, 31)]},
            {"id": 4, "words": [word("", 0, 0, 1, 1)]},
            {"id": 5, "words": [word("Two", 10, 50, 40, 70), word("lines", 10, 80, 50, 100)]},
        ]
    }
    page = pb.funsd_page(ann, "raw/funsd/x.png", (200, 120), "funsd-x", None)
    assert [(ln.line_id, ln.text) for ln in page.lines] == [
        ("funsd-x/e3/l0", "Date: 1998"),
        ("funsd-x/e5/l0", "Two"),
        ("funsd-x/e5/l1", "lines"),
    ]
    assert page.lines[0].polygon == [[10, 10], [110, 10], [110, 31], [10, 31]]
    assert all(ln.rec_ok for ln in page.lines)


def test_omnidocbench_page_keeps_text_blocks_in_order_and_ignores_the_rest():
    pg = {
        "page_info": {
            "image_path": "p1.png",
            "width": 100,
            "height": 200,
            "page_attribute": {"language": "english"},
        },
        "layout_dets": [
            {
                "category_type": "text_block",
                "poly": "[0, 50, 90, 50, 90, 60, 0, 60]",
                "ignore": "False",
                "order": "2",
                "anno_id": "b2",
                "text": "second",
            },
            {
                "category_type": "title",
                "poly": [0, 0, 90, 0, 90, 10, 0, 10],
                "ignore": "False",
                "order": "1",
                "anno_id": "b1",
                "text": "First",
            },
            {"category_type": "figure", "poly": [0, 100, 50, 100, 50, 150, 0, 150], "order": "3"},
            {
                "category_type": "text_block",
                "poly": [0, 160, 9, 160, 9, 170, 0, 170],
                "ignore": "True",
                "order": "4",
                "anno_id": "b4",
                "text": "skip me",
            },
            {
                "category_type": "text_block",
                "poly": [0, 180, 9, 180, 9, 190, 0, 190],
                "ignore": "False",
                "order": None,
                "anno_id": "b5",
                "text": "  ",
            },
        ],
    }
    page = pb.omnidocbench_page(pg, "raw/omnidocbench/images/p1.png")
    assert [ln.text for ln in page.lines] == ["First", "second"]
    assert page.lines[0].line_id == "odb-p1/b1/title"
    assert page.lines[1].polygon == [[0, 50], [90, 50], [90, 60], [0, 60]]
    assert not any(ln.rec_ok for ln in page.lines)
    assert len(page.ignore) == 3  # the figure, the ignored block, the empty block


def test_main_writes_packs_pages_and_manifest(tmp_path):
    import scipy.io

    root = tmp_path / "data"
    base = root / "raw/iiit5k/extracted/IIIT5K"
    (base / "test").mkdir(parents=True)
    names = ["test/1_1.png", "test/2_1.png"]
    for i, n in enumerate(names):
        (base / n).write_bytes(_png(40 + i, 20))
    rec = np.zeros((1, 2), dtype=[("ImgName", "O"), ("GroundTruth", "O")])
    rec[0, 0] = (np.array([names[0]]), np.array(["PRIVATE"]))
    rec[0, 1] = (np.array([names[1]]), np.array(["SALE"]))
    scipy.io.savemat(base / "testdata.mat", {"testdata": rec})

    fbase = root / "raw/funsd/extracted/dataset/testing_data"
    (fbase / "annotations").mkdir(parents=True)
    (fbase / "images").mkdir(parents=True)
    (fbase / "images/f1.png").write_bytes(_png(200, 100))
    (fbase / "annotations/f1.json").write_text(
        json.dumps({"form": [{"id": 0, "words": [word("Name", 5, 5, 40, 20)]}]}), encoding="utf-8"
    )

    obase = root / "raw/omnidocbench"
    (obase / "images").mkdir(parents=True)
    (obase / "images/p1.png").write_bytes(_png(100, 200))
    pages = [
        {
            "page_info": {
                "image_path": "p1.png",
                "width": 100,
                "height": 200,
                "page_attribute": {"language": "english"},
            },
            "layout_dets": [
                {
                    "category_type": "text_block",
                    "poly": [0, 0, 9, 0, 9, 9, 0, 9],
                    "ignore": "False",
                    "order": "1",
                    "anno_id": "a",
                    "text": "Hi",
                }
            ],
        },
        {
            "page_info": {
                "image_path": "zh.png",
                "width": 1,
                "height": 1,
                "page_attribute": {"language": "simplified_chinese"},
            },
            "layout_dets": [],
        },
    ]
    (obase / "OmniDocBench.json").write_text(json.dumps(pages), encoding="utf-8")

    manifest = tmp_path / "benchmarks.json"
    assert pb.main(["all", "--data-root", str(root), "--manifest", str(manifest)]) == 0
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["iiit5k_test"]["count"] == 2
    assert m["funsd_test"]["page_count"] == 1 and m["funsd_test"]["line_count"] == 1
    assert m["omnidocbench_en"]["page_count"] == 1 and m["omnidocbench_en"]["block_count"] == 1
    reader = LinePackReader(root / "processed/iiit5k/words_test")
    assert [r.label for r in reader.records] == ["PRIVATE", "SALE"]
    assert reader.image(1).size == (41, 20)
    assert reader.read_bytes(0) == (base / names[0]).read_bytes()  # stored as published
    odb = list(read_pages(root / "processed/omnidocbench/pages_test.jsonl.gz"))
    assert [p.doc_id for p in odb] == ["odb-p1"]
    with gzip.open(root / "processed/funsd/pages_test.jsonl.gz", "rt", encoding="utf-8") as f:
        assert json.loads(f.readline())["lines"][0]["text"] == "Name"


def test_missing_omnidocbench_images_stop_the_conversion(tmp_path):
    obase = tmp_path / "raw/omnidocbench"
    obase.mkdir(parents=True)
    pages = [
        {
            "page_info": {
                "image_path": "gone.png",
                "width": 1,
                "height": 1,
                "page_attribute": {"language": "english"},
            },
            "layout_dets": [],
        }
    ]
    (obase / "OmniDocBench.json").write_text(json.dumps(pages), encoding="utf-8")
    with pytest.raises(SystemExit, match="missing"):
        pb.prepare_omnidocbench(tmp_path)
