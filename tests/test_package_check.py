"""Phase 18's package check (`scripts/package_check.py`) on made-up data: a tiny bundle from
untrained models, three synthetic validation pages and reference predictions written from the same
bundle. The detector's probability head is set to fire everywhere, so each page reads one line and
the line comparison has something to compare."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import package_check
import pytest
import torch
import yaml
from det_fixtures import make_synthetic_pages, write_processed_manifest
from test_pipeline import tiny_detector, tiny_recognizer

from glypharis.data.det_dataset import read_gt_pages
from glypharis.data.det_geometry import decode_page
from glypharis.data.download import sha256_file
from glypharis.data.pages import write_pages
from glypharis.efficiency.v1 import V1CodeError
from glypharis.export.bundle import read_manifest, write_bundle
from glypharis.pipeline import GlypharisOCR

pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parents[1]
PAGES = "processed/hiertext/pages_validation.jsonl.gz"  # gw-det-001's det.data.val_pages
GIT = {"git_commit": "0123456789abcdef", "git_dirty": False}


def _write_predictions(path: Path, pages: dict[str, list[dict[str, Any]]]) -> Path:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for key, lines in pages.items():
            fh.write(json.dumps({"key": key, "lines": lines}) + "\n")
    return path


@pytest.fixture(scope="module")
def env(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("package_check")
    synth = make_synthetic_pages(root, n_train=1, n_val=3)
    data = synth.data_root
    pages_file = data / PAGES
    write_pages(pages_file, synth.val)
    write_processed_manifest(data, "detsynth", {"validation": pages_file})
    config = root / "det.yaml"
    config.write_text(
        yaml.safe_dump({
            "inherits": (REPO / "configs" / "det" / "gw-det-001.yaml").as_posix(),
            "paths": {"data_root": data.as_posix()},
            "det": {"data": {"val_pages": [PAGES]}},
        }),
        encoding="utf-8",
    )  # fmt: skip
    det, rec = tiny_detector(), tiny_recognizer()
    state = det.backend.model.state_dict()
    state["prob_head.out.weight"].zero_()
    state["prob_head.out.bias"].fill_(10.0)  # the whole page is text: one line per page
    torch.save({"model": state}, root / "det_ck.pt")
    bundle = write_bundle(root / "det_ck.pt", det.spec, rec, root / "models" / "tiny-0.0.1",
                          "0.0.1-test")  # fmt: skip
    ocr = GlypharisOCR.from_pretrained(bundle, "onnx", cache=False)
    gt = read_gt_pages(pages_file, [p.doc_id for p in synth.val])
    same: dict[str, list[dict[str, Any]]] = {}
    other: dict[str, list[dict[str, Any]]] = {}
    truth: dict[str, list[dict[str, Any]]] = {}
    for d, page in gt.items():
        res = ocr.predict(decode_page(data / page.image, (page.width, page.height)))
        same[d] = [{"polygon": ln.polygon, "text": ln.text, "confidence": ln.confidence}
                   for ln in res.lines]  # fmt: skip
        other[d] = [dict(ln, text="~" * 60) for ln in same[d]]
        truth[d] = [{"polygon": ln.polygon, "text": ln.text or "x", "confidence": 1.0}
                    for ln in page.lines]  # fmt: skip
    return SimpleNamespace(
        root=root,
        data=data,
        config=config,
        bundle=bundle,
        same=_write_predictions(data / "same.jsonl", same),
        other=_write_predictions(root / "other.jsonl", other),
        truth=_write_predictions(root / "truth.jsonl", truth),
    )


@pytest.fixture(autouse=True)
def fixed_git(monkeypatch):
    monkeypatch.setattr(package_check, "git_state", lambda: dict(GIT))


def _argv(env: SimpleNamespace, out: Path, *extra: str, ref: Path | None = None) -> list[str]:
    return ["--config", str(env.config), "--model", str(env.bundle),
            "--phase10", str(ref or env.same), "--out", str(out), *extra]  # fmt: skip


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in (k, *_strings(v))]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return [obj] if isinstance(obj, str) else []


def _no_local_paths(out: Path, *roots: Path) -> None:
    text = out.read_text(encoding="utf-8")
    for r in roots:
        for form in (str(r), r.as_posix(), json.dumps(str(r))[1:-1]):
            assert form not in text
    for s in _strings(json.loads(text)):
        assert not re.match(r"^([A-Za-z]:[\\/]|/|\\\\)", s), s


def test_passes_against_its_own_predictions_and_records_provenance(env, tmp_path):
    out = tmp_path / "reports" / "check.json"
    assert package_check.main(_argv(env, out, "--experiment", "GW-E2E-099")) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    # the v0.1 fields, unchanged in name and meaning
    assert res["model"] == "tiny-0.0.1" and res["backend"] == "onnx"
    assert res["model_version"] == "0.0.1-test" and res["pages"] == 3
    assert res["difference"] == 0.0 and res["page_cer_package"] == res["page_cer_phase10"]
    assert res["lines_same_text_and_box"] == res["lines"] == 3 and res["pass"] is True
    # provenance
    assert res["experiment"] == "GW-E2E-099"
    assert res["cer_tolerance"] == package_check.CER_TOL == 0.0005
    assert res["box_tolerance_px"] == package_check.BOX_TOL == 1.0
    for k in ("started_utc", "ended_utc"):
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", res[k])
    assert res["started_utc"] <= res["ended_utc"]
    assert res["git_commit"] == GIT["git_commit"] and res["git_dirty"] is False
    assert res["bundle_manifest_sha256"] == sha256_file(env.bundle / "manifest.json")
    assert res["bundle_version"] == "0.0.1-test"
    assert res["reference"] == "same.jsonl"  # relative to the data root
    assert res["reference_sha256"] == sha256_file(env.same)
    assert res["config"] == "det.yaml"  # outside the working directory: the file name only
    assert res["threads"] is None
    assert res["torch_threads"] == torch.get_num_threads()
    assert res["torch_interop_threads"] == torch.get_num_interop_threads()
    assert set(res["versions"]) == {"python", "torch", "onnxruntime"}
    assert res["versions"]["torch"] == torch.__version__
    assert res["v1_code"] is None
    assert not list(out.parent.glob("*.tmp"))
    _no_local_paths(out, env.root, tmp_path)


def test_a_bundle_name_is_resolved_and_threads_are_recorded(env, tmp_path, monkeypatch):
    monkeypatch.setenv("GLYPHARIS_MODELS", str(env.bundle.parent))
    out = tmp_path / "check.json"
    argv = _argv(env, out, "--threads", "1", "--reference-sha256", sha256_file(env.same).upper(),
                 "--expect-manifest-sha256", sha256_file(env.bundle / "manifest.json"))  # fmt: skip
    argv[argv.index("--model") + 1] = "tiny-0.0.1"
    assert package_check.main(argv) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["model"] == "tiny-0.0.1" and res["threads"] == 1 and res["experiment"] is None
    assert res["bundle_manifest_sha256"] == sha256_file(env.bundle / "manifest.json")
    _no_local_paths(out, env.root, tmp_path)


def test_an_existing_output_is_refused_before_anything_loads(env, tmp_path, monkeypatch):
    out = tmp_path / "check.json"
    out.write_text("{}\n", encoding="utf-8")

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("loaded before the output was checked")

    monkeypatch.setattr(package_check, "find_bundle", boom)
    monkeypatch.setattr(package_check, "load_config", boom)
    with pytest.raises(SystemExit, match=r"check\.json exists"):
        package_check.main(_argv(env, out))
    assert out.read_text(encoding="utf-8") == "{}\n"


def test_wrong_hashes_are_refused(env, tmp_path):
    out = tmp_path / "check.json"
    with pytest.raises(SystemExit, match=r"same\.jsonl: SHA-256 .* not the expected 00"):
        package_check.main(_argv(env, out, "--reference-sha256", "00" * 32))
    with pytest.raises(SystemExit, match=r"tiny-0\.0\.1: manifest SHA-256 .* not the expected"):
        package_check.main(_argv(env, out, "--expect-manifest-sha256", "ff" * 32))
    assert not out.exists()


def _record(manifest: dict[str, Any], temperature: float | None = None) -> dict[str, Any]:
    spec, rec = manifest["detector"]["spec"], manifest["recognizer"]
    return {
        "detector": {k: spec[k] for k in ("infer", "post", "operating_point", "label_version")}
        | {"checkpoint_sha256": None},
        "recognizer": {
            "input": {"height": rec["height"], "max_width": rec["max_width"]},
            "temperature": rec["temperature"] if temperature is None else temperature,
            "checkpoint_sha256": None,
        },
    }


def test_v1_refuses_code_that_is_not_v1s(env, tmp_path, monkeypatch):
    def differs(root: Any = ".") -> dict[str, Any]:
        raise V1CodeError("the v1 code differs from phase14-eval")

    monkeypatch.setattr(package_check, "assert_v1_code", differs)
    out = tmp_path / "check.json"
    with pytest.raises(SystemExit, match="differs from phase14-eval; --v1 runs v1 as tagged"):
        package_check.main(_argv(env, out, "--v1"))
    assert not out.exists()


def test_v1_records_the_code_check_and_refuses_a_bundle_unlike_the_record(
    env, tmp_path, monkeypatch
):
    v1 = {"tag": "phase14-eval", "tag_commit": "aa", "head": "bb", "v1_code": "equal"}
    monkeypatch.setattr(package_check, "assert_v1_code", lambda root=".": dict(v1))
    manifest = read_manifest(env.bundle)
    record = tmp_path / "glypharis_v1.yaml"
    monkeypatch.setattr(package_check, "FREEZE_RECORD", record)

    record.write_text(yaml.safe_dump(_record(manifest, temperature=0.5)), encoding="utf-8")
    out = tmp_path / "check.json"
    with pytest.raises(SystemExit, match="the bundle is not v1's: recognizer temperature"):
        package_check.main(_argv(env, out, "--v1"))
    assert not out.exists()

    record.write_text(yaml.safe_dump(_record(manifest)), encoding="utf-8")
    assert package_check.main(_argv(env, out, "--v1")) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["v1_code"] == v1 and res["pass"] is True
    _no_local_paths(out, env.root, tmp_path)


@pytest.mark.parametrize("which", ["other", "truth"])
def test_a_reference_with_other_text_fails(env, tmp_path, which):
    out = tmp_path / "check.json"
    assert package_check.main(_argv(env, out, ref=getattr(env, which))) == 1
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["pass"] is False and abs(res["difference"]) > package_check.CER_TOL
    if which == "other":  # the same boxes, other text: no line agrees
        assert res["lines_same_text_and_box"] == 0 and res["lines"] == 3
    else:  # the ground truth itself
        assert res["page_cer_phase10"] == 0.0
    assert res["reference"] == f"{which}.jsonl"  # outside the data root and working directory
    _no_local_paths(out, env.root, tmp_path)
