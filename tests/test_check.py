"""Phase 23, item 9: the outsider check (`glypharis.check`, ``glypharis-check``) on a tiny bundle
built from untrained models, through a test registry. Home, the working directory and
GLYPHARIS_MODELS are redirected to the test's temporary folder, and every socket connection fails
the test, so nothing reaches the network or the real ~/.glypharis/models or ./models."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.resources
import json
import shutil
import socket
import tomllib
from pathlib import Path
from typing import Any

import pytest
import torch
from test_pipeline import tiny_detector, tiny_recognizer

from glypharis import check, hub
from glypharis.export.bundle import write_bundle

pytest.importorskip("onnx")  # writing the tiny bundle exports ONNX files

NAME = "tiny-check"
ROOT = Path(__file__).resolve().parents[1]
V1_MANIFEST = "875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, Any]:
    """The tiny bundle and its expected output on the shipped sample page, made once."""
    import make_sample_page as msp

    root = tmp_path_factory.mktemp("check-src")
    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, root / "det_ck.pt")
    bundle = write_bundle(root / "det_ck.pt", det.spec, rec, root / "src" / NAME, "0.0.1-check")
    sha = check.manifest_sha256(bundle)
    result = check.read_sample(bundle, check.SAMPLE_PAGE, threads=1)
    record = check.expected_record(result, NAME, sha, check.SAMPLE_PAGE, 1, msp.LINES)
    expected = root / "expected.json"
    expected.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    registry = root / "registry.json"
    assert msp.main(["registry", "--model", str(bundle), "--out", str(registry)]) == 0
    return {"bundle": bundle, "sha": sha, "expected": expected, "registry": registry}


@pytest.fixture
def env(tmp_path, monkeypatch, built) -> dict[str, Any]:
    """Home, cwd and GLYPHARIS_MODELS in ``tmp_path``, the tiny bundle installed there by hand,
    and no network: any socket connection or call to download_model is recorded and refused."""
    home, cwd, models = tmp_path / "home", tmp_path / "cwd", tmp_path / "models"
    for d in (home, cwd, models):
        d.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("GLYPHARIS_MODELS", str(models))
    monkeypatch.delenv(hub.OFFLINE_ENV, raising=False)
    monkeypatch.chdir(cwd)
    connects: list[Any] = []

    def no_network(self: socket.socket, address: Any) -> None:
        connects.append(address)
        raise AssertionError(f"the check tried to connect to {address}")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    downloads: list[str] = []
    real_download = hub.download_model

    def download(name: str, **kw: Any) -> Path:
        downloads.append(name)
        return real_download(name, **kw)

    monkeypatch.setattr(hub, "download_model", download)
    shutil.copytree(built["bundle"], models / NAME)
    entry = hub.ModelEntry(
        name=NAME,
        url="https://example.invalid/tiny-check.tar.gz",
        archive="tiny-check.tar.gz",
        bytes=None,
        archive_sha256=None,
        manifest_sha256=built["sha"],
        top_folder=NAME,
    )
    yield {
        "models": models,
        "entry": entry,
        "connects": connects,
        "downloads": downloads,
        **built,
    }
    assert connects == []


def _run(env: dict[str, Any], entry: hub.ModelEntry | None = None, **kw: Any) -> tuple[bool, str]:
    out: list[str] = []
    kw.setdefault("expected_path", env["expected"])
    with hub.override_registry(entry or env["entry"]):
        passed = check.run_check(NAME, threads=1, say=out.append, **kw)
    return passed, "\n".join(out)


def _cli(env: dict[str, Any], *extra: str, expected: Path | None = None) -> list[str]:
    exp = str(expected or env["expected"])
    return ["--model", NAME, "--threads", "1", "--registry", str(env["registry"]),
            "--expected", exp, *extra]  # fmt: skip


# ---------------------------------------------------------------------------------- passing


def test_passes_when_the_bundle_reproduces_the_expected_output(env):
    passed, out = _run(env)
    assert passed, out
    assert "PASSED" in out and "manifest.json SHA-256" in out
    assert "identical to the expected output" in out
    assert "no download record" in out
    assert "says nothing about accuracy" in out
    assert env["downloads"] == []


def test_the_command_passes_with_a_test_registry_and_says_so(env, capsys):
    assert check.main(_cli(env)) == 0
    out = capsys.readouterr().out
    assert "PASSED" in out
    assert "test registry" in out and "does not show that the weights are the released ones" in out
    assert NAME not in hub.REGISTRY and "glypharis-v1" in hub.REGISTRY  # restored
    assert env["downloads"] == []


def test_a_matching_download_record_passes(env):
    entry = dataclasses.replace(env["entry"], bytes=1234, archive_sha256="ab" * 32)
    record = {"name": NAME, "archive_bytes": 1234, "archive_sha256": "ab" * 32,
              "manifest_sha256": env["sha"], "url": "http://127.0.0.1:1/x"}  # fmt: skip
    (env["models"] / NAME / hub.RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")
    passed, out = _run(env, entry)
    assert passed, out
    assert "download record: archive SHA-256" in out


def test_models_dir_is_used_and_a_shadowing_bundle_is_noted(env, tmp_path):
    other = tmp_path / "elsewhere"
    shutil.copytree(env["models"] / NAME, other / NAME)
    shutil.rmtree(env["models"] / NAME)
    passed, out = _run(env, models_dir=other)
    assert passed, out
    assert f"glypharis-ocr --model {NAME} would load nothing" in out


# ---------------------------------------------------------------------------------- failing


@pytest.mark.parametrize("change", ["first_line", "extra_line", "text", "model_version"])
def test_fails_when_the_output_differs(env, tmp_path, capsys, change):
    data = json.loads(env["expected"].read_text(encoding="utf-8"))
    if change == "first_line" and data["lines"]:
        data["lines"][0] += "x"
    elif change in ("first_line", "extra_line"):
        data["lines"].append("a line the model never read")
    elif change == "text":
        data["text"] += " and more"
    else:
        data["model_version"] = "glypharis_v0"
    changed = tmp_path / "expected.json"
    changed.write_text(json.dumps(data), encoding="utf-8")
    passed, out = _run(env, expected_path=changed)
    assert not passed
    assert "FAILED" in out and "says nothing about accuracy" not in out
    assert check.main(_cli(env, expected=changed)) == 1
    assert "FAIL" in capsys.readouterr().out


def test_fails_on_a_wrong_manifest_pin(env):
    entry = dataclasses.replace(env["entry"], manifest_sha256="0" * 64)
    passed, out = _run(env, entry)
    assert not passed
    assert f"is not the registry's {'0' * 64}" in out


def test_fails_when_a_model_file_does_not_match_the_manifest(env):
    with (env["models"] / NAME / "recognizer.pt").open("ab") as fh:
        fh.write(b"\0")
    passed, out = _run(env)
    assert not passed
    assert "do not match its manifest.json" in out


@pytest.mark.parametrize(
    ("pins", "record_change", "expected"),
    [
        ((1234, "ab" * 32), {"archive_sha256": "cd" * 32}, "archive_sha256"),
        ((1234, "ab" * 32), {"archive_bytes": 999}, "archive_bytes"),
        ((1234, "ab" * 32), {"manifest_sha256": "0" * 64}, "manifest_sha256"),
        ((None, None), {}, "pins no archive"),
    ],
)
def test_fails_on_a_wrong_download_record(env, pins, record_change, expected):
    entry = dataclasses.replace(env["entry"], bytes=pins[0], archive_sha256=pins[1])
    record = {"name": NAME, "archive_bytes": 1234, "archive_sha256": "ab" * 32,
              "manifest_sha256": env["sha"], **record_change}  # fmt: skip
    (env["models"] / NAME / hub.RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")
    passed, out = _run(env, entry)
    assert not passed
    assert expected in out


def test_fails_on_an_unreadable_download_record(env):
    (env["models"] / NAME / hub.RECORD_FILE).write_text("{not json", encoding="utf-8")
    passed, out = _run(env)
    assert not passed
    assert "unreadable download record" in out


def test_fails_when_the_expected_output_is_for_another_model(env, tmp_path):
    data = json.loads(env["expected"].read_text(encoding="utf-8"))
    data["manifest_sha256"] = V1_MANIFEST
    changed = tmp_path / "expected.json"
    changed.write_text(json.dumps(data), encoding="utf-8")
    passed, out = _run(env, expected_path=changed)
    assert not passed
    assert "was made with manifest 875357a6346e" in out


def test_fails_when_the_page_is_not_the_pinned_one(env, tmp_path):
    page = tmp_path / "page.png"
    page.write_bytes(check.SAMPLE_PAGE.read_bytes() + b"\0")
    passed, out = _run(env, page=page)
    assert not passed
    assert "is not the expected" in out


# ------------------------------------------------------------------------- cannot run, no network


def test_a_missing_model_is_not_downloaded(env, capsys):
    shutil.rmtree(env["models"] / NAME)
    assert check.main(_cli(env)) == 2
    err = capsys.readouterr().err
    assert "is not installed" in err and "nothing was downloaded" in err
    assert env["downloads"] == []


def test_download_goes_through_download_model_and_respects_its_rules(env, capsys):
    # Installed: nothing to fetch, so download_model is not even called.
    assert check.main(_cli(env, "--download")) == 0
    assert env["downloads"] == []
    capsys.readouterr()
    shutil.rmtree(env["models"] / NAME)
    # Missing and not published: refused before any network call.
    assert check.main(_cli(env, "--download")) == 2
    assert "not yet published" in capsys.readouterr().err
    # Missing and offline: refused too.
    assert check.main(_cli(env, "--download", "--offline")) == 2
    assert "offline" in capsys.readouterr().err
    assert env["downloads"] == [NAME, NAME]


def test_an_unknown_model_or_a_bad_registry_file_cannot_run(env, tmp_path, capsys):
    assert check.main(["--model", "nope", "--registry", str(env["registry"])]) == 2
    assert "unknown model 'nope'" in capsys.readouterr().err
    bad = tmp_path / "registry.json"
    bad.write_text(json.dumps({"models": [{"name": NAME, "surprise": 1}]}), encoding="utf-8")
    assert check.main(["--model", NAME, "--registry", str(bad)]) == 2
    assert "not a readable registry file" in capsys.readouterr().err


def test_a_bad_expected_file_cannot_run(env, tmp_path, capsys):
    bad = tmp_path / "expected.json"
    bad.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")
    assert check.main(_cli(env, expected=bad)) == 2
    assert "[sample_error]" in capsys.readouterr().err


# ------------------------------------------------------------------------------- shipped files


def test_the_sample_ships_as_package_data_and_pins_v1():
    import make_sample_page as msp

    sample = importlib.resources.files("glypharis").joinpath("sample")
    for name in ("page.png", "expected.json", "README.md"):
        assert sample.joinpath(name).is_file(), name
    assert check.SAMPLE_PAGE.is_file() and check.SAMPLE_EXPECTED.is_file()
    expected = check.load_expected(check.SAMPLE_EXPECTED)
    assert (
        expected["manifest_sha256"] == V1_MANIFEST == hub.REGISTRY["glypharis-v1"].manifest_sha256
    )
    assert expected["model"] == "glypharis-v1" and expected["model_version"] == "glypharis_v1"
    assert expected["backend"] == "torch" and expected["threads"] == check.DEFAULT_THREADS
    page = check.SAMPLE_PAGE.read_bytes()
    assert hashlib.sha256(page).hexdigest() == expected["page_sha256"]
    assert len(page) < 300_000
    assert expected["rendered_lines"] == list(msp.LINES)
    assert expected["text"] == "\n".join(t for t in expected["lines"] if t)
    assert not any("second" in k or "time" in k for k in expected)  # no timings

    from PIL import Image

    with Image.open(check.SAMPLE_PAGE) as im:
        assert im.size == msp.PAGE_SIZE == (expected["width"], expected["height"])

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["glypharis-check"] == "glypharis.check:main"
    data = project["tool"]["setuptools"]["package-data"]["glypharis"]
    assert {"sample/page.png", "sample/expected.json"} <= set(data)
    assert "Open Sans" in (ROOT / "NOTICE").read_text(encoding="utf-8")


def test_check_wheel_requires_the_command_and_sample_with_the_module(tmp_path):
    import check_wheel as cw
    from test_packaging import _source, _wheel

    root = _source(tmp_path)
    (root / "glypharis" / "check.py").write_text("def main(): ...\n", "utf-8")
    whl = _wheel(tmp_path / "glypharis-1.0.0-py3-none-any.whl", root)
    problems = cw.check_wheel(whl, "1.0.0", None)
    assert any("entry_points.txt has no glypharis-check" in p for p in problems), problems
    assert any("no glypharis/sample/page.png" in p for p in problems), problems
    info = "glypharis-1.0.0.dist-info/"
    (tmp_path / "ok").mkdir()
    whl = _wheel(
        tmp_path / "ok" / "glypharis-1.0.0-py3-none-any.whl",
        root,
        extra={
            info + "entry_points.txt": b"[console_scripts]\nglypharis-ocr = glypharis.cli:main\n"
            b"glypharis-serve = glypharis.server:main\nglypharis-check = glypharis.check:main\n",
            "glypharis/sample/page.png": b"png",
            "glypharis/sample/expected.json": b"{}",
        },
    )
    assert cw.check_wheel(whl, "1.0.0", None) == []
