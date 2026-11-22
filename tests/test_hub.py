"""Phase 23: model download (`glypharis.hub`) against a local HTTP server, with a tiny bundle built
from untrained models. Home, the working directory and GLYPHARIS_MODELS are redirected to the test's
temporary folder, so nothing reaches the real ~/.glypharis/models or ./models."""

from __future__ import annotations

import dataclasses
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from det_fixtures import rect_polygon
from test_pipeline import _page, tiny_detector, tiny_recognizer

import glypharis
from glypharis import hub
from glypharis import server as server_module
from glypharis.cli import main as cli_main
from glypharis.errors import ModelDownloadError, ModelLoadError
from glypharis.export.bundle import read_manifest, write_bundle

pytest.importorskip("onnxruntime")

NAME = "tiny-hub"
REPO_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------------------- fixtures


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pack(bundle: Path, out: Path, top: str = NAME) -> bytes:
    with tarfile.open(out, "w:gz") as tar:
        tar.add(bundle, arcname=top)
    return out.read_bytes()


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, Any]:
    """The tiny bundle, its archive and variants of it, built once."""
    root = tmp_path_factory.mktemp("hub-src")
    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, root / "det_ck.pt")
    bundle = write_bundle(root / "det_ck.pt", det.spec, rec, root / "bundle" / NAME, "0.0.1-hub")
    good = _pack(bundle, root / "good.tar.gz")

    other = root / "other" / NAME  # the same files, a different manifest
    shutil.copytree(bundle, other)
    manifest = json.loads((other / "manifest.json").read_text(encoding="utf-8"))
    manifest["version"] = "something-else"
    (other / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    damaged = root / "damaged" / NAME  # a model file that no longer matches the manifest
    shutil.copytree(bundle, damaged)
    with (damaged / "detector.onnx").open("ab") as fh:
        fh.write(b"\0")

    return {
        "bundle": bundle,
        "manifest_sha": _sha((bundle / "manifest.json").read_bytes()),
        "good": good,
        "other": other,
        "other_manifest_sha": _sha((other / "manifest.json").read_bytes()),
        "damaged_archive": _pack(damaged, root / "damaged.tar.gz"),
        "wrong_top_archive": _pack(bundle, root / "wrong-top.tar.gz", top="not-the-name"),
    }


class LocalServer:
    """Serves ``files`` by name, records every request, and can cut a response short or wait."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[dict[str, str | None]] = []
        self.cut = 0  # how many of the next responses stop half way
        self.delay = 0.0  # seconds before each response body
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                outer.requests.append({"path": self.path, "range": self.headers.get("Range")})
                data = outer.files.get(self.path.lstrip("/"))
                if data is None:
                    self.send_error(404)
                    return
                rng = self.headers.get("Range")
                start = int(rng.removeprefix("bytes=").split("-")[0]) if rng else 0
                if start >= len(data) > 0:
                    self.send_error(416)
                    return
                body = data[start:]
                self.send_response(206 if rng else 200)
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                time.sleep(outer.delay)
                if outer.cut > 0:
                    outer.cut -= 1
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/{name}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def places(tmp_path, monkeypatch) -> dict[str, Path]:
    """Home, the working directory and GLYPHARIS_MODELS, all under ``tmp_path``."""
    home, cwd, models = tmp_path / "home", tmp_path / "cwd", tmp_path / "models"
    for d in (home, cwd, models):
        d.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("GLYPHARIS_MODELS", str(models))
    monkeypatch.delenv(hub.OFFLINE_ENV, raising=False)
    monkeypatch.chdir(cwd)
    assert Path.home() == home
    monkeypatch.setattr(hub, "DOWNLOAD_RETRIES", 1)
    monkeypatch.setattr(hub, "DOWNLOAD_BACKOFF", 0.01)
    monkeypatch.setattr(hub, "DOWNLOAD_TIMEOUT", 10.0)
    monkeypatch.setattr(hub, "LOCK_POLL_SECONDS", 0.02)
    return {"home": home, "cwd": cwd, "models": models, "user": home / ".glypharis" / "models"}


@pytest.fixture
def server(built) -> Iterator[LocalServer]:
    srv = LocalServer({"tiny-hub.tar.gz": built["good"]})
    yield srv
    srv.close()


def _entry(built: dict[str, Any], server: LocalServer, **kw: Any) -> hub.ModelEntry:
    fields: dict[str, Any] = dict(
        name=NAME,
        url=server.url("tiny-hub.tar.gz"),
        archive="tiny-hub.tar.gz",
        bytes=len(built["good"]),
        archive_sha256=_sha(built["good"]),
        manifest_sha256=built["manifest_sha"],
        top_folder=NAME,
        published=True,
    )
    fields.update(kw)
    return hub.ModelEntry(**fields)


@pytest.fixture
def registered(built, server, places) -> Iterator[hub.ModelEntry]:
    entry = _entry(built, server)
    with hub.override_registry(entry):
        yield entry


def _downloads(models: Path) -> list[str]:
    d = models / hub.DOWNLOADS_DIR
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


def _assert_installed(path: Path, built: dict[str, Any]) -> None:
    read_manifest(path)
    assert _sha((path / "manifest.json").read_bytes()) == built["manifest_sha"]


# ------------------------------------------------------------------------- the shipped entry


def test_the_shipped_entry_is_v1_and_published(places, monkeypatch):
    from glypharis.data.download import DownloadError

    entry = hub.REGISTRY["glypharis-v1"]
    assert entry.url == (
        "https://github.com/DuhItzAniket/Glypharis/releases/download/v1.0/glypharis-v1.tar.gz"
    )
    assert entry.archive == "glypharis-v1.tar.gz" and entry.top_folder == "glypharis-v1"
    assert entry.manifest_sha256 == (
        "875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037"
    )
    # The archive packed by scripts/pack_release.py (Phase 23, item 7), the asset of release v1.0.
    assert entry.bytes == 82975077
    assert entry.archive_sha256 == (
        "3b90c1c74147747d7474a86b550e198f0df83d447961bd6aec88d4c24846541d"
    )
    assert entry.published is True
    assert hub.can_fetch("glypharis-v1", offline=False)
    assert not hub.can_fetch("glypharis-v1", offline=True)

    requested: list[str] = []

    def no_network(url: str, *args: object, **kw: object) -> None:
        requested.append(url)
        raise DownloadError("no network in tests")

    monkeypatch.setattr("glypharis.data.download.download_file", no_network)
    # Missing and published: the fetch goes to the registered address (and is stopped here).
    with pytest.raises(ModelDownloadError, match="cannot download 'glypharis-v1'"):
        glypharis.download_model()
    assert requested == [entry.url]
    # Offline refuses before any network call.
    with pytest.raises(ModelDownloadError, match="offline"):
        glypharis.download_model(offline=True)
    # An entry that is not published is refused before any network call, unless url= is given.
    with (
        hub.override_registry(dataclasses.replace(entry, published=False)),
        pytest.raises(ModelDownloadError, match="not yet published"),
    ):
        glypharis.download_model("glypharis-v1")
    # With an address but no archive pins there is still nothing to verify against.
    unpinned = dataclasses.replace(entry, bytes=None, archive_sha256=None)
    with (
        hub.override_registry(unpinned),
        pytest.raises(ModelDownloadError, match="no pinned archive"),
    ):
        glypharis.download_model("glypharis-v1", url="http://127.0.0.1:9/x.tar.gz")
    assert requested == [entry.url]
    assert _downloads(places["models"]) == []
    assert hub.REGISTRY["glypharis-v1"] is entry  # restored after each override


def test_download_model_is_exported_lazily():
    assert "download_model" in glypharis.__all__
    assert glypharis.download_model is hub.download_model
    assert issubclass(ModelDownloadError, ModelLoadError)
    assert ModelDownloadError("x").to_dict() == {"error": "model_download_error", "message": "x"}


# -------------------------------------------------------------------------- happy path, record


def test_fetch_installs_a_verified_bundle_with_a_record(registered, server, places, built):
    path = hub.download_model(NAME)
    assert path == places["models"] / NAME  # the first GLYPHARIS_MODELS entry
    _assert_installed(path, built)
    record = hub.read_record(path)
    assert record is not None
    assert record["network_fetch"] is True and record["url"] == server.url("tiny-hub.tar.gz")
    assert record["archive_sha256"] == registered.archive_sha256
    assert record["manifest_sha256"] == built["manifest_sha"]
    assert len(server.requests) == 1
    assert _downloads(places["models"]) == []  # archive, lock and staging all gone
    # Installed: the next call, and the CLI's check, make no request.
    assert hub.download_model(NAME) == path and not hub.needs_download(NAME)
    assert len(server.requests) == 1


def test_default_folder_without_glypharis_models_is_the_user_folder(
    registered, places, monkeypatch
):
    monkeypatch.delenv("GLYPHARIS_MODELS")
    path = hub.download_model(NAME)
    assert path == places["user"] / NAME
    assert not (places["cwd"] / "models").exists()


def test_models_dir_must_be_a_lookup_folder(registered, places, server, tmp_path):
    with pytest.raises(ModelDownloadError, match="not a folder bundles are looked up in"):
        hub.download_model(NAME, models_dir=tmp_path / "elsewhere")
    assert server.requests == []
    path = hub.download_model(NAME, models_dir=places["user"])
    assert path == places["user"] / NAME


def test_unknown_name_is_refused(registered, server):
    with pytest.raises(ModelDownloadError, match="unknown model 'nope'"):
        hub.download_model("nope")
    assert server.requests == []


# ------------------------------------------------------------------ refusals before the network


def test_offline_refuses_with_no_request(registered, server, places, monkeypatch):
    with pytest.raises(ModelDownloadError, match="offline"):
        hub.download_model(NAME, offline=True)
    monkeypatch.setenv(hub.OFFLINE_ENV, "1")
    with pytest.raises(ModelDownloadError, match="offline"):
        hub.download_model(NAME)
    assert server.requests == [] and _downloads(places["models"]) == []
    assert hub.download_model(NAME, offline=False).is_dir()  # the argument wins over the variable


def test_not_published_is_refused_unless_a_url_is_given(built, server, places):
    entry = _entry(built, server, published=False)
    with hub.override_registry(entry):
        with pytest.raises(ModelDownloadError, match="not yet published"):
            hub.download_model(NAME)
        assert server.requests == []
        path = hub.download_model(NAME, url=server.url("tiny-hub.tar.gz"))
    _assert_installed(path, built)


def test_only_http_and_https_addresses_are_accepted(registered, server, places):
    for bad in ("notaurl", "", "file:///x.tar.gz", "ftp://127.0.0.1/tiny-hub.tar.gz"):
        with pytest.raises(ModelDownloadError, match="only http and https"):
            hub.download_model(NAME, url=bad)
    assert server.requests == []
    assert not (places["models"] / hub.DOWNLOADS_DIR).exists()


def test_missing_pins_are_refused_with_no_request(built, server, places):
    for kw in ({"bytes": None}, {"archive_sha256": None}):
        entry = _entry(built, server, **kw)
        with hub.override_registry(entry), pytest.raises(ModelDownloadError, match="no pinned"):
            hub.download_model(NAME)
    assert server.requests == []


# ------------------------------------------------------------------------ installed bundles


def test_a_pinned_bundle_already_installed_is_returned_with_no_request(
    registered, server, places, built
):
    shutil.copytree(built["bundle"], places["user"] / NAME)  # not in models_dir, still found
    assert hub.download_model(NAME) == places["user"] / NAME
    assert server.requests == []


def test_a_shadowing_bundle_with_another_hash_is_refused_and_kept(
    registered, server, places, built
):
    shadow = places["models"] / NAME  # first in the lookup order
    shutil.copytree(built["other"], shadow)
    before = (shadow / "manifest.json").read_bytes()
    for kw in ({}, {"force": True, "models_dir": places["user"]}):
        with pytest.raises(ModelDownloadError, match="not the registered one"):
            hub.download_model(NAME, **kw)
    assert (shadow / "manifest.json").read_bytes() == before
    assert not (places["user"] / NAME).exists() and server.requests == []


def test_force_replaces_only_the_bundle_in_models_dir_and_keeps_the_old_one(
    registered, server, places, built
):
    target = places["models"] / NAME
    shutil.copytree(built["other"], target)
    path = hub.download_model(NAME, force=True)
    assert path == target
    _assert_installed(path, built)
    kept = [n for n in _downloads(places["models"]) if n.startswith(f"replaced-{NAME}-")]
    assert len(kept) == 1
    moved = places["models"] / hub.DOWNLOADS_DIR / kept[0] / "manifest.json"
    assert _sha(moved.read_bytes()) == built["other_manifest_sha"]


def test_a_failed_force_replacement_puts_the_old_bundle_back_or_says_where_it_is(
    registered, server, places, built, monkeypatch
):
    target = places["models"] / NAME
    shutil.copytree(built["other"], target)

    def fail(src: Path, dst: Path, attempts: int = 5) -> None:
        raise PermissionError("in use")

    monkeypatch.setattr(hub, "_rename", fail)  # the new bundle cannot be moved in
    with pytest.raises(ModelDownloadError, match="restored to"):
        hub.download_model(NAME, force=True)
    assert _sha((target / "manifest.json").read_bytes()) == built["other_manifest_sha"]
    assert not [n for n in _downloads(places["models"]) if n.startswith("replaced-")]

    real_rename = os.rename

    def rename(src: Any, dst: Any) -> None:  # and the old one cannot be moved back either
        if Path(src).name.startswith(f"replaced-{NAME}-"):
            raise PermissionError("in use")
        real_rename(src, dst)

    monkeypatch.setattr(hub.os, "rename", rename)
    with pytest.raises(ModelDownloadError, match="still at") as info:
        hub.download_model(NAME, force=True)
    assert not target.exists()
    kept = [n for n in _downloads(places["models"]) if n.startswith(f"replaced-{NAME}-")]
    assert len(kept) == 1 and kept[0] in str(info.value)
    moved = places["models"] / hub.DOWNLOADS_DIR / kept[0] / "manifest.json"
    assert _sha(moved.read_bytes()) == built["other_manifest_sha"]


def test_a_folder_in_the_way_is_refused(registered, server, places):
    (places["models"] / NAME).mkdir()
    (places["models"] / NAME / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ModelDownloadError, match="not a bundle"):
        hub.download_model(NAME)
    assert (places["models"] / NAME / "notes.txt").read_text(encoding="utf-8") == "mine"


# --------------------------------------------------------------------- transfer and hash checks


def test_a_truncated_transfer_keeps_its_part_file_and_the_next_call_resumes(
    registered, server, places, built
):
    server.cut = 1
    with pytest.raises(ModelDownloadError, match="cannot download"):
        hub.download_model(NAME)
    assert not (places["models"] / NAME).exists()
    parts = [n for n in _downloads(places["models"]) if n.endswith(".part")]
    assert len(parts) == 1
    size = (places["models"] / hub.DOWNLOADS_DIR / parts[0]).stat().st_size
    assert 0 < size < len(built["good"])
    path = hub.download_model(NAME)
    _assert_installed(path, built)
    assert server.requests[-1]["range"] == f"bytes={size}-"


def test_a_short_file_on_the_server_is_refused(built, places):
    srv = LocalServer({"tiny-hub.tar.gz": built["good"][:-100]})
    try:
        with (
            hub.override_registry(_entry(built, srv)),
            pytest.raises(ModelDownloadError, match="expected"),
        ):
            hub.download_model(NAME)
    finally:
        srv.close()
    assert not (places["models"] / NAME).exists()


def test_resume_from_an_existing_part_file(registered, server, places, built):
    downloads = places["models"] / hub.DOWNLOADS_DIR
    downloads.mkdir()
    sha12 = registered.archive_sha256[:12]  # type: ignore[index]
    part = downloads / f"{NAME}-{sha12}-tiny-hub.tar.gz.part"
    half = len(built["good"]) // 3
    part.write_bytes(built["good"][:half])
    path = hub.download_model(NAME)
    _assert_installed(path, built)
    assert [r["range"] for r in server.requests] == [f"bytes={half}-"]


def test_a_corrupted_archive_is_deleted_and_nothing_installed(built, places):
    bad = bytearray(built["good"])
    bad[len(bad) // 2] ^= 0xFF  # same size, other bytes
    srv = LocalServer({"tiny-hub.tar.gz": bytes(bad)})
    try:
        with (
            hub.override_registry(_entry(built, srv)),
            pytest.raises(ModelDownloadError, match="does not match the registered"),
        ):
            hub.download_model(NAME)
    finally:
        srv.close()
    assert not (places["models"] / NAME).exists()
    assert _downloads(places["models"]) == []


def test_url_override_still_enforces_size_and_hashes(registered, server, places, built):
    server.files["evil.tar.gz"] = built["damaged_archive"]
    with pytest.raises(ModelDownloadError):
        hub.download_model(NAME, url=server.url("evil.tar.gz"))
    assert not (places["models"] / NAME).exists()
    server.files["copy.tar.gz"] = built["good"]
    path = hub.download_model(NAME, url=server.url("copy.tar.gz"))
    record = hub.read_record(path)
    assert record is not None and record["url"] == server.url("copy.tar.gz")
    assert record["registry_url"] == registered.url


def test_a_wrong_manifest_hash_inside_the_archive_is_refused(built, server, places):
    entry = _entry(built, server, manifest_sha256="0" * 64)
    with hub.override_registry(entry), pytest.raises(ModelDownloadError, match="manifest SHA"):
        hub.download_model(NAME)
    assert not (places["models"] / NAME).exists()
    assert not [n for n in _downloads(places["models"]) if n.startswith(".staging")]


def test_an_archive_whose_files_do_not_match_its_manifest_is_refused(built, places):
    data = built["damaged_archive"]
    srv = LocalServer({"tiny-hub.tar.gz": data})
    entry = _entry(built, srv, bytes=len(data), archive_sha256=_sha(data))
    try:
        with hub.override_registry(entry), pytest.raises(ModelDownloadError, match="not a valid"):
            hub.download_model(NAME)
    finally:
        srv.close()
    assert not (places["models"] / NAME).exists()


def test_an_archive_with_another_top_folder_is_refused(built, places):
    data = built["wrong_top_archive"]
    srv = LocalServer({"tiny-hub.tar.gz": data})
    entry = _entry(built, srv, bytes=len(data), archive_sha256=_sha(data))
    try:
        with hub.override_registry(entry), pytest.raises(ModelDownloadError, match="top-level"):
            hub.download_model(NAME)
    finally:
        srv.close()
    assert not (places["models"] / NAME).exists()


# ----------------------------------------------------------------------------- lock and races


def test_two_threads_end_with_one_bundle_and_one_transfer(registered, server, places, built):
    server.delay = 0.5  # the second caller waits on the lock while the first downloads
    results: list[Path | BaseException] = []

    def run() -> None:
        try:
            results.append(hub.download_model(NAME))
        except BaseException as exc:
            results.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert results == [places["models"] / NAME] * 2
    assert len(server.requests) == 1
    _assert_installed(places["models"] / NAME, built)
    assert _downloads(places["models"]) == []


def test_two_processes_end_with_one_bundle_and_one_transfer(registered, server, places, built):
    worker = places["home"] / "worker.py"
    worker.write_text(
        "import json, sys\n"
        "from glypharis import hub\n"
        "hub.LOCK_POLL_SECONDS = 0.05\n"
        "entry = hub.ModelEntry(**json.loads(sys.argv[1]))\n"
        "with hub.override_registry(entry):\n"
        "    print(hub.download_model(entry.name))\n",
        encoding="utf-8",
    )
    server.delay = 1.0
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONIOENCODING": "utf-8"}
    arg = json.dumps(dataclasses.asdict(registered))
    procs = [
        subprocess.Popen(
            [sys.executable, str(worker), arg],
            cwd=places["cwd"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outs = [p.communicate(timeout=300) for p in procs]
    for p, (_, err) in zip(procs, outs, strict=True):
        assert p.returncode == 0, err
    assert {Path(out.strip()) for out, _ in outs} == {places["models"] / NAME}
    assert len(server.requests) == 1
    _assert_installed(places["models"] / NAME, built)
    assert _downloads(places["models"]) == []


def test_a_stale_lock_is_taken_over(registered, places, built):
    downloads = places["models"] / hub.DOWNLOADS_DIR
    downloads.mkdir()
    lock = downloads / f"{NAME}.lock"
    lock.write_text("{}", encoding="utf-8")
    old = time.time() - hub.LOCK_STALE_SECONDS - 10
    os.utime(lock, (old, old))
    _assert_installed(hub.download_model(NAME), built)
    assert not lock.exists()


def test_a_live_lock_times_out(registered, server, places, monkeypatch):
    downloads = places["models"] / hub.DOWNLOADS_DIR
    downloads.mkdir()
    (downloads / f"{NAME}.lock").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hub, "LOCK_WAIT_SECONDS", 0.2)
    with pytest.raises(ModelDownloadError, match="another download holds"):
        hub.download_model(NAME)
    assert server.requests == []


def test_a_fresh_lock_that_replaces_a_stale_one_is_not_taken_over(places, monkeypatch):
    downloads = places["models"] / hub.DOWNLOADS_DIR
    downloads.mkdir()
    lock = downloads / f"{NAME}.lock"
    lock.write_text(json.dumps({"pid": 1, "token": "dead"}) + "\n", encoding="utf-8")
    old = time.time() - hub.LOCK_STALE_SECONDS - 10
    os.utime(lock, (old, old))
    fresh = json.dumps({"pid": 2, "token": "fresh"}) + "\n"
    real_rename = os.rename

    def rename(src: Any, dst: Any) -> None:
        # Another waiter took the stale lock over and a new holder created its own lock, all
        # between this waiter's look at the file and its rename.
        if Path(src) == lock:
            lock.unlink()
            lock.write_text(fresh, encoding="utf-8")
        real_rename(src, dst)

    monkeypatch.setattr(hub.os, "rename", rename)
    assert hub._break_stale(lock) is False
    assert lock.read_text(encoding="utf-8") == fresh
    assert _downloads(places["models"]) == [lock.name]  # nothing left aside


def test_the_holder_refreshes_and_removes_only_its_own_lock(places, monkeypatch):
    monkeypatch.setattr(hub, "LOCK_HEARTBEAT_SECONDS", 0.02)
    lock = places["models"] / f"{NAME}.lock"
    with hub._lock(lock):
        assert hub._lock_token(lock) is not None
        lock.write_text(json.dumps({"token": "someone-else"}) + "\n", encoding="utf-8")
        old = time.time() - hub.LOCK_STALE_SECONDS - 10
        os.utime(lock, (old, old))
        time.sleep(0.2)  # several heartbeats
        assert time.time() - lock.stat().st_mtime > hub.LOCK_STALE_SECONDS
    assert hub._lock_token(lock) == "someone-else"
    lock.unlink()
    with hub._lock(lock):
        assert lock.is_file()
    assert not lock.exists()


# --------------------------------------------------------------------------- CLI and server


def _page_file(tmp_path: Path) -> Path:
    path = tmp_path / "page.png"
    _page([rect_polygon(150, 80, 200, 20)]).save(path)
    return path


def test_cli_fetches_a_missing_registered_model(registered, server, places, tmp_path, capsys):
    page = _page_file(tmp_path)
    args = ["--image", str(page), "--output", str(tmp_path / "out"), "--model", NAME]
    assert cli_main([*args, "--no-download"]) == 2
    assert "cannot load the model" in capsys.readouterr().err
    assert server.requests == [] and not (places["models"] / NAME).exists()
    assert cli_main(args) == 0
    assert "downloading it" in capsys.readouterr().err
    assert (tmp_path / "out" / "page.json").is_file() and len(server.requests) == 1
    assert cli_main(args) == 0 and len(server.requests) == 1  # installed now


def test_cli_offline_refuses_with_no_request(registered, server, places, tmp_path, monkeypatch,
                                             capsys):  # fmt: skip
    monkeypatch.setenv(hub.OFFLINE_ENV, "true")
    page = _page_file(tmp_path)
    assert cli_main(["--image", str(page), "--output", str(tmp_path / "o"), "--model", NAME]) == 2
    err = capsys.readouterr().err
    assert "offline" in err and "downloading it" not in err and server.requests == []
    assert not hub.can_fetch(NAME) and hub.can_fetch(NAME, offline=False)


def test_cli_never_fetches_a_path_or_an_unregistered_name(registered, server, tmp_path):
    page = _page_file(tmp_path)
    for model in (str(tmp_path / NAME), "not-registered"):
        assert cli_main(["--image", str(page), "--model", model]) == 2
    assert not hub.needs_download(tmp_path / NAME) and not hub.needs_download("not-registered")
    assert server.requests == []


class _FakeHTTPServer:
    server_port = 0

    def serve_forever(self) -> None:
        raise KeyboardInterrupt

    def server_close(self) -> None:
        pass


def test_server_fetches_a_missing_registered_model(registered, server, places, monkeypatch):
    loaded: list[Any] = []

    def fake_make_server(ocr: Any, *args: Any) -> _FakeHTTPServer:
        loaded.append(ocr)
        return _FakeHTTPServer()

    monkeypatch.setattr(server_module, "make_server", fake_make_server)
    assert server_module.main(["--model", NAME, "--no-download"]) == 2
    assert server.requests == [] and loaded == []
    assert server_module.main(["--model", NAME]) == 0
    assert len(server.requests) == 1 and loaded[0].model_version == "0.0.1-hub"
    _assert_installed(places["models"] / NAME, {"manifest_sha": registered.manifest_sha256})
