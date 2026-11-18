"""GW-E2E-012's script (`scripts/hardening_check.py`) on a tiny bundle built from untrained models
and a made-up page; the git checks are stubbed."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hardening_check as hc
import pytest
import torch
from det_fixtures import rect_polygon
from test_pipeline import _page, tiny_detector, tiny_recognizer

from glypharis.data.download import sha256_file
from glypharis.export.bundle import write_bundle

pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parents[1]
V1_CODE = {"tag": "phase14-eval", "tag_commit": "t" * 40, "head": "h" * 40, "v1_code": "equal"}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> SimpleNamespace:
    root = tmp_path_factory.mktemp("hardening")
    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, root / "det_ck.pt")
    bundle = write_bundle(root / "det_ck.pt", det.spec, rec, root / "tiny-v1", hc.MODEL_VERSION)
    page = root / "page.png"
    _page([rect_polygon(150, 80, 200, 20)]).save(page)
    return SimpleNamespace(root=root, bundle=bundle, page=page,
                           sha=sha256_file(bundle / "manifest.json"))  # fmt: skip


@pytest.fixture
def stubbed(monkeypatch) -> None:
    monkeypatch.setattr(hc, "assert_v1_code", lambda root: dict(V1_CODE))
    monkeypatch.setattr(hc, "git_state", lambda: {"git_commit": "c" * 40, "git_dirty": False})


def _argv(t: SimpleNamespace, out: Path, sha: str | None = None) -> list[str]:
    return ["--bundle", str(t.bundle), "--page", str(t.page), "--threads", "2", "--root",
            str(REPO), "--expect-manifest-sha256", sha or t.sha, "--out", str(out)]  # fmt: skip


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in [k, *_strings(v)]]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def test_the_check_passes_on_both_backends_and_names_no_local_path(tiny, stubbed, tmp_path):
    out = tmp_path / "reports" / "GW-E2E-012.json"
    assert hc.main(_argv(tiny, out)) == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["experiment"] == "GW-E2E-012" and r["pass"] is True
    assert r["checks"] == {k: True for k in ("no_other_exceptions",
                           "damaged_bundles_model_load_error", "cache",
                           "load_log_names_model_version", "versions", "cli", "api",
                           "good_page", "bundle_unchanged")}  # fmt: skip
    assert r["other_exceptions"] == [] and r["bundle_unchanged"]
    assert r["backends"] == ["torch", "onnx"] and r["threads"] == 2
    assert (r["bundle"], r["page"]) == ("tiny-v1", "page.png")
    assert r["bundle_manifest_sha256"] == tiny.sha and r["bundle_version"] == hc.MODEL_VERSION
    assert r["model_version"] == hc.MODEL_VERSION and r["v1_code"] == V1_CODE
    assert r["git_commit"] == "c" * 40 and r["fuzz_seed"] == 20261028
    assert {"started_utc", "ended_utc", "versions", "cli_version"} <= set(r)
    for backend in ("torch", "onnx"):
        kinds = r["inputs"][backend]["kinds"]
        assert all(r["expected_outcomes"][backend].values())
        assert sum(t["inputs"] for k, t in kinds.items() if k.startswith("fuzz_lite/")) == 120
        fuzz_b = [t for k, t in kinds.items() if k.startswith("fuzz_lite_bytes/")]
        assert sum(t["inputs"] + t["no_bytes_form"] for t in fuzz_b) == 120
        assert kinds["bad_files"]["inputs"] == 6 and kinds["bad_files_bytes"]["no_bytes_form"] == 2
        assert kinds["over_pixel_limit"]["glypharis_errors"] == {"image_too_large": 6}
        assert kinds["good_page"] == {"inputs": 4, "results": 4, "glypharis_errors": {},
                                      "no_bytes_form": 0, "other_exceptions": []}  # fmt: skip
        assert r["inputs"][backend]["good_page_path_and_bytes_same_text"]
        assert r["inputs"][backend]["max_pixels"] == hc.DEFAULT_MAX_PIXELS
        c = r["cache_and_log"][backend]
        assert c["cached_twice_same_object"] and c["uncached_new_object"]
        assert f"{hc.MODEL_VERSION} ({backend} backend) from <bundle>" in c["load_log_line"]
        assert "in <seconds> s" in c["load_log_line"]
        a = r["api"][backend]
        assert a["good_page"]["status"] == 200 and a["good_page"]["error"] is None
        assert a["bad_file"] == {"status": 400, "error": "invalid_image"}
        assert "seconds" not in json.dumps(r["stats"][backend])
    damaged = r["damaged_bundles"]
    assert len(damaged) == 2 * (4 + 1 + 1 + 4 + 2 + 4)
    loaded = {(d["case"], d["backend"]) for d in damaged if d["outcome"] == "loaded"}
    assert loaded == {("junk_rehashed:detector.onnx", "torch"),
                      ("junk_rehashed:recognizer.onnx", "torch")}  # fmt: skip
    for d in damaged:
        assert d["ok"]
        if d["outcome"] == "model_load_error":
            assert len(d["message"]) <= hc.MESSAGE_CHARS and "<copy>" in d["message"]
    cli = r["cli"]
    assert cli["exit_code"] == 1 and cli["runs_this_code"] and cli["good_page_json_written"]
    assert cli["failed_line"].startswith("FAILED <input>/not_an_image.png: [invalid_image]")
    assert cli["good_page_line"].startswith("<input>/page.png: ")
    assert r["cli_version"]["printed"] == r["versions"]
    # No local path anywhere: not the temporary folders, not the home folder, no drive letter.
    needles = {str(p) for p in (tmp_path, tiny.root, Path.home(), Path(tempfile.gettempdir()))}
    needles |= {Path(n).as_posix() for n in needles} | {str(REPO), REPO.as_posix()}
    for s in _strings(r):
        assert not any(n in s for n in needles), s
        assert not hc._WIN_PATH.search(s), s


def test_an_existing_output_is_refused(tiny, stubbed, tmp_path):
    out = tmp_path / "GW-E2E-012.json"
    out.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="written once"):
        hc.main(_argv(tiny, out))
    assert out.read_text(encoding="utf-8") == "{}\n"


def test_another_manifest_hash_is_refused_before_anything_loads(tiny, stubbed, tmp_path):
    out = tmp_path / "GW-E2E-012.json"
    with pytest.raises(SystemExit, match="not the registered bundle"):
        hc.main(_argv(tiny, out, sha="0" * 64))
    with pytest.raises(SystemExit, match="not the registered bundle"):
        hc.main([a for a in _argv(tiny, out) if a not in ("--expect-manifest-sha256", tiny.sha)])
    assert not out.exists()


def test_v1_code_that_differs_is_refused(tiny, monkeypatch, tmp_path):
    def differs(root: Any) -> dict[str, Any]:
        raise hc.V1CodeError("the v1 code differs from phase14-eval")

    monkeypatch.setattr(hc, "assert_v1_code", differs)
    with pytest.raises(SystemExit, match="runs v1 as tagged"):
        hc.main(_argv(tiny, tmp_path / "v.json"))
