"""GW-REC-041 runner (scripts/phase15_glyphsynth.py): the join, the refusals and the output.

A made-up GlyphSynth validation pack, render log and font manifest in tmp_path, with a stub
recognizer in place of the frozen one; no checkpoint, GPU or prepared data is touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import phase15_glyphsynth as gs
import pytest
from PIL import Image, ImageDraw

from glypharis.data.crops import encode_jpeg
from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackWriter
from glypharis.pipeline.recognizer import LineReading

LABELS = ["alpha beta", "Gamma 42", "delta.", "EPSILON", "zeta-eta"]
KEYS = [f"glyphsynth-val-{i:07d}" for i in range(len(LABELS))]
FACES = [
    {"family": "robotoslab", "category": "serif", "path": "serif/apache/robotoslab/RS.ttf",
     "split": "val"},
    {"family": "inter", "category": "sans", "path": "sans/ofl/inter/Inter.ttf", "split": "train"},
    {"family": "inter", "category": "sans", "path": "sans/ofl/inter/Inter-Italic.ttf",
     "split": "train"},
    {"family": "cousine", "category": "mono", "path": "mono/apache/cousine/Cousine.ttf",
     "split": "val"},
]  # fmt: skip
FONT_OF = [0, 1, 2, 3, 0]  # face index per line
PROVENANCE = {
    "name": "glypharis_v1", "recognizer_experiment": "GW-REC-034", "record_sha256": "1" * 64,
    "recognizer_checkpoint": "checkpoints/GW-REC-034/best.pt",
    "recognizer_checkpoint_sha256": "2" * 64, "device": "cpu", "device_name": "cpu",
    "git_commit": "c" * 40, "git_dirty": False,
}  # fmt: skip


def _render_row(i: int) -> dict[str, Any]:
    face = FACES[FONT_OF[i]]
    return {"key": KEYS[i], "family": face["family"], "font_file": face["path"],
            "size_px": 20.0 + i, "italic": "Italic" in face["path"],
            "weight": None if i % 2 else 400.0, "inverted": i == 3, "neighbours": i % 2 == 0,
            "underline": False, "capture": ["ink_spread"] if i == 1 else []}  # fmt: skip


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def build(data: Path, *, split: str = "val", render: list[dict[str, Any]] | None = None) -> Path:
    """A complete tiny data root; returns the pack prefix."""
    prefix = data / "processed/glyphsynth/lines_val"
    meta = {"split": "val", "source": "glyphsynth", "glyphsynth_version": 1}
    with LinePackWriter(prefix, meta) as w:
        for i, (key, label) in enumerate(zip(KEYS, LABELS, strict=True)):
            img = Image.new("L", (60 + 10 * i, 64), 255)
            ImageDraw.Draw(img).rectangle([5, 20, 30 + i, 40], fill=0)
            w.add(key, key, split, "glyphsynth", label, encode_jpeg(img, 95), img.size)
    rows = render if render is not None else [_render_row(i) for i in range(len(KEYS))]
    _write_jsonl(gs.render_path(prefix), rows)
    manifests = data / "manifests"
    (manifests / "processed").mkdir(parents=True, exist_ok=True)
    sha = sha256_file(prefix.with_suffix(".bin"))
    (manifests / "processed/glyphsynth.json").write_text(
        json.dumps({"splits": {"val": {"count": len(KEYS), "bin_sha256": sha}}}), encoding="utf-8"
    )
    (manifests / "glyphsynth_fonts.json").write_text(
        json.dumps({"held_out_fraction": 0.15, "faces": FACES}), encoding="utf-8"
    )
    return prefix


class StubRecognizer:
    """Reads every crop as its width in pixels, and records the batches it was given."""

    temperature = 2.35

    def __init__(self) -> None:
        self.batches: list[int] = []

    def read(
        self, crops: list[Image.Image], temperatures: tuple[float, ...] = ()
    ) -> list[LineReading]:
        self.batches.append(len(crops))
        return [LineReading(f"w{c.width}", 0.5, c.width // 4, {t: 0.9 for t in temperatures})
                for c in crops]  # fmt: skip


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    rec = StubRecognizer()
    frozen = SimpleNamespace(
        recognizer=rec,
        provenance=dict(PROVENANCE),
        settings=lambda: {"recognizer": {"input": {"height": 64}, "temperature": 2.35,
                                         "strict_fp32": True}},
    )  # fmt: skip
    calls: list[str] = []

    def load(device: str, **kw: Any) -> SimpleNamespace:
        calls.append(device)
        return frozen

    monkeypatch.setattr(gs, "load_frozen", load)
    return SimpleNamespace(rec=rec, calls=calls)


def _run(data: Path, *extra: str) -> int:
    return gs.main(["--data-root", str(data), "--device", "cpu", *extra])


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_joins_pack_and_render_log_by_key_in_pack_order(
    tmp_path: Path, stub: SimpleNamespace
) -> None:
    data = tmp_path / "data"
    prefix = build(data, render=[_render_row(i) for i in reversed(range(len(KEYS)))])
    assert _run(data, "--batch", "2") == 0
    out = data / gs.OUTPUT
    recs = _records(out)
    assert [r["key"] for r in recs] == KEYS
    assert [r["ref"] for r in recs] == LABELS
    assert [r["pred"] for r in recs] == [f"w{60 + 10 * i}" for i in range(len(KEYS))]
    assert stub.rec.batches == [2, 2, 1] and stub.calls == ["cpu"]
    for i, r in enumerate(recs):
        row = _render_row(i)
        assert {f: r[f] for f in gs.RENDER_FIELDS} == {f: row[f] for f in gs.RENDER_FIELDS}
        assert r["conf_t1"] == 0.9 and r["conf_t"] == 0.5
        assert r["crop_height_px"] == 64 and r["blur"] >= 0 and r["contrast"] > 0
    assert [r["category"] for r in recs] == ["serif", "sans", "sans", "mono", "serif"]
    assert [r["held_out"] for r in recs] == [True, False, False, True, True]

    meta = json.loads((out.parent / "glyphsynth_val.meta.json").read_text(encoding="utf-8"))
    assert meta["experiment"] == "GW-REC-041" and meta["split"] == "validation"
    assert meta["count"] == meta["expected"] == len(KEYS) and meta["limit"] is None
    assert meta["commit"] == "c" * 40 and meta["recognizer_checkpoint_sha256"] == "2" * 64
    assert meta["device"] == "cpu" and meta["wall_seconds"] >= 0
    assert meta["output_sha256"] == sha256_file(out)
    assert meta["inputs"]["pack"]["bin_sha256"] == sha256_file(prefix.with_suffix(".bin"))
    assert meta["inputs"]["render_log"]["sha256"] == sha256_file(gs.render_path(prefix))
    assert not list(out.parent.glob("*.tmp"))


def test_limit_reads_the_first_crops_and_needs_another_output(
    tmp_path: Path, stub: SimpleNamespace
) -> None:
    data = tmp_path / "data"
    build(data)
    with pytest.raises(SystemExit, match="--limit"):
        _run(data, "--limit", "2")
    assert not stub.calls
    out = tmp_path / "smoke" / "gs.jsonl"
    assert _run(data, "--limit", "2", "--out", str(out)) == 0
    assert [r["key"] for r in _records(out)] == KEYS[:2]
    meta = json.loads((out.parent / "gs.meta.json").read_text(encoding="utf-8"))
    assert meta["count"] == 2 and meta["limit"] == 2 and meta["expected"] == len(KEYS)


@pytest.mark.parametrize("eval_set", ["hiertext_val", "hiertext_test", "gdc_test",
                                      "gdc_scan_test", "iiit5k_test", "no_such_set"])  # fmt: skip
def test_any_other_set_is_refused_before_anything_is_read(
    tmp_path: Path, stub: SimpleNamespace, eval_set: str
) -> None:
    with pytest.raises(SystemExit, match="GlyphSynth validation"):
        _run(tmp_path / "missing", "--set", eval_set)
    assert not stub.calls


def test_a_pack_record_of_another_split_is_refused(tmp_path: Path, stub: SimpleNamespace) -> None:
    data = tmp_path / "data"
    build(data, split="test")
    with pytest.raises(SystemExit, match="not GlyphSynth validation"):
        _run(data)
    assert not stub.calls and not (data / gs.OUTPUT).exists()


@pytest.mark.parametrize("change", ["drop", "extra", "twice", "field"])
def test_a_render_log_that_does_not_join_is_refused(
    tmp_path: Path, stub: SimpleNamespace, change: str
) -> None:
    rows = [_render_row(i) for i in range(len(KEYS))]
    if change == "drop":
        rows = rows[:-1]
    elif change == "extra":
        rows.append({**rows[0], "key": "glyphsynth-val-9999999"})
    elif change == "twice":
        rows.append(dict(rows[0]))
    else:
        del rows[2]["capture"]
    data = tmp_path / "data"
    build(data, render=rows)
    with pytest.raises(SystemExit):
        _run(data)
    assert not stub.calls and not (data / gs.OUTPUT).exists()


def test_a_font_whose_path_and_manifest_disagree_is_refused(
    tmp_path: Path, stub: SimpleNamespace
) -> None:
    data = tmp_path / "data"
    build(data)
    fonts = data / "manifests/glyphsynth_fonts.json"
    doc = json.loads(fonts.read_text(encoding="utf-8"))
    doc["faces"][1]["category"] = "serif"
    fonts.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(SystemExit, match="category"):
        _run(data)
    assert not stub.calls


def test_a_font_missing_from_the_manifest_is_refused(tmp_path: Path, stub: SimpleNamespace) -> None:
    rows = [_render_row(i) for i in range(len(KEYS))]
    rows[0]["font_file"] = "serif/ofl/unknown/Unknown.ttf"
    data = tmp_path / "data"
    build(data, render=rows)
    with pytest.raises(SystemExit, match="not in the font manifest"):
        _run(data)
    assert not stub.calls


def test_a_pack_that_differs_from_its_record_is_refused(
    tmp_path: Path, stub: SimpleNamespace
) -> None:
    data = tmp_path / "data"
    build(data)
    manifest = data / "manifests/processed/glyphsynth.json"
    manifest.write_text(json.dumps({"splits": {"val": {"count": len(KEYS),
                                                       "bin_sha256": "0" * 64}}}),
                        encoding="utf-8")  # fmt: skip
    with pytest.raises(SystemExit, match="refusing to run"):
        _run(data)
    manifest.unlink()
    with pytest.raises(SystemExit, match="no recorded hash"):
        _run(data)
    assert not stub.calls


def test_an_existing_output_is_never_overwritten(tmp_path: Path, stub: SimpleNamespace) -> None:
    data = tmp_path / "data"
    build(data)
    assert _run(data) == 0
    before = sha256_file(data / gs.OUTPUT)
    with pytest.raises(SystemExit, match="already exists"):
        _run(data)
    assert sha256_file(data / gs.OUTPUT) == before and len(stub.calls) == 1


def test_font_category_is_the_first_folder() -> None:
    assert gs.font_category("serif/ofl/newsreader/Newsreader[opsz,wght].ttf") == "serif"
    with pytest.raises(SystemExit):
        gs.font_category("Loose.ttf")


def test_a_family_split_across_trained_and_held_out_faces_is_refused() -> None:
    faces = {f["path"]: f for f in FACES}
    faces["sans/ofl/inter/Inter.ttf"] = {**faces["sans/ofl/inter/Inter.ttf"], "split": "val"}
    with pytest.raises(SystemExit, match="both held out and trained on"):
        gs.held_out_families(faces)


@pytest.mark.parametrize(
    "folder",
    ["processed/phase14", "processed/baselines", "processed/baselines_phase14",
     "manifests/processed"],
)  # fmt: skip
def test_an_output_under_a_protected_folder_is_refused_before_anything_is_read(
    tmp_path: Path, stub: SimpleNamespace, folder: str
) -> None:
    data = tmp_path / "data"
    build(data)
    out = data / folder / "GW-REC-041" / "glyphsynth_val.jsonl"
    with pytest.raises(SystemExit, match="never writes"):
        _run(data, "--out", str(out))
    assert not stub.calls and not out.parent.exists()


def test_the_runner_writes_what_the_analysis_checks(tmp_path: Path, stub: SimpleNamespace) -> None:
    """The .meta.json carries the fields `phase15_errors` requires of GW-REC-041's file."""
    import phase15_errors as pe

    data = tmp_path / "data"
    build(data)
    assert _run(data) == 0
    meta = json.loads((data / gs.OUTPUT).with_name("glyphsynth_val.meta.json").read_text("utf-8"))
    assert {k: meta[k] for k in pe.SYNTH_META} == pe.SYNTH_META
    assert gs.OUTPUT == pe.SYNTH_FILE
