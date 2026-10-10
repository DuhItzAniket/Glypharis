"""Downloader tests against a local fake HTTP server (no network access)."""

from __future__ import annotations

import gzip
import hashlib
import http.server
import io
import json
import shutil
import tarfile
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import download_datasets
import pytest
import yaml

from glypharis.data import download as dl
from glypharis.data.counters import count_hiertext
from glypharis.data.registry import build_dataset_manifest, write_dataset_manifest
from glypharis.data.sources import DatasetSpec, FileSpec, SourceError, load_sources, parse_sources

# ------------------------------------------------------------------------------ fake server


class _Handler(http.server.BaseHTTPRequestHandler):
    root: Path
    honour_range = True
    requests: list[dict[str, str | None]]

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass

    def do_GET(self) -> None:
        self.requests.append({"path": self.path, "range": self.headers.get("Range")})
        path = self.root / self.path.lstrip("/")
        if not path.is_file():
            self.send_error(404)
            return
        data = path.read_bytes()
        rng = self.headers.get("Range")
        if rng and self.honour_range:
            start = int(rng.removeprefix("bytes=").split("-")[0])
            if start >= len(data):
                self.send_error(416)
                return
            body = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            body = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class FakeServer:
    def __init__(self, root: Path, honour_range: bool = True) -> None:
        handler = type(
            "H", (_Handler,), {"root": root, "honour_range": honour_range, "requests": []}
        )
        self.handler = handler
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def requests(self) -> list[dict[str, str | None]]:
        return self.handler.requests

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/{name}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _make_tgz(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[FakeServer, Path]]:
    files = tmp_path / "server"
    files.mkdir()
    (files / "blob.bin").write_bytes(bytes(range(256)) * 400)  # 102,400 bytes
    _make_tgz(files / "imgs.tgz", {"train/a.jpg": b"a" * 10, "train/b.jpg": b"b" * 20})
    server = FakeServer(files)
    yield server, files
    server.close()


def _spec(server: FakeServer, server_dir: Path, **overrides: object) -> DatasetSpec:
    blob = FileSpec(
        "blob", server.url("blob.bin"), "blob.bin", size=(server_dir / "blob.bin").stat().st_size
    )
    arch = FileSpec(
        "imgs",
        server.url("imgs.tgz"),
        "archives/imgs.tgz",
        size=(server_dir / "imgs.tgz").stat().st_size,
        extract_to="images",
    )
    fields: dict[str, object] = dict(
        key="fake", title="Fake", version="1", homepage="https://example.org", licence="CC0",
        citation="none", role="test", phase=2, files=(blob, arch),
    )  # fmt: skip
    fields.update(overrides)
    return DatasetSpec(**fields)  # type: ignore[arg-type]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------- download


def test_fetch_downloads_extracts_and_writes_manifest(served, tmp_path):
    server, files = served
    root = tmp_path / "data"
    manifest = dl.fetch_dataset(_spec(server, files), root, counter=lambda raw: {"n": 1})

    assert (root / "raw/fake/blob.bin").read_bytes() == (files / "blob.bin").read_bytes()
    assert (root / "raw/fake/images/train/a.jpg").read_bytes() == b"a" * 10
    rec = manifest["files"]["blob"]
    assert rec["sha256"] == _sha(files / "blob.bin")
    assert rec["sha256_source"] == "computed"
    assert rec["path"] == "raw/fake/blob.bin"
    assert manifest["files"]["imgs"]["extracted_files"] == 2
    assert manifest["counts"] == {"n": 1}
    on_disk = json.loads((root / "manifests/fake.json").read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert str(tmp_path) not in json.dumps(on_disk)  # no absolute paths in manifests


def test_second_run_is_a_no_op(served, tmp_path):
    server, files = served
    root = tmp_path / "data"
    spec = _spec(server, files)
    dl.fetch_dataset(spec, root)
    text = (root / "manifests/fake.json").read_text(encoding="utf-8")
    mtime = (root / "manifests/fake.json").stat().st_mtime_ns
    n_requests = len(server.requests)

    dl.fetch_dataset(spec, root)

    assert len(server.requests) == n_requests  # nothing fetched
    assert (root / "manifests/fake.json").read_text(encoding="utf-8") == text
    assert (root / "manifests/fake.json").stat().st_mtime_ns == mtime  # not even rewritten


def test_resume_uses_range_request(served, tmp_path):
    server, files = served
    root = tmp_path / "data"
    spec = _spec(server, files)
    part = root / "raw/fake/blob.bin.part"
    part.parent.mkdir(parents=True)
    part.write_bytes((files / "blob.bin").read_bytes()[:1000])

    dl.fetch_dataset(spec, root, only={"blob"})

    assert {"path": "/blob.bin", "range": "bytes=1000-"} in server.requests
    assert _sha(root / "raw/fake/blob.bin") == _sha(files / "blob.bin")
    assert not part.exists()


def test_server_ignoring_range_restarts_cleanly(tmp_path):
    files = tmp_path / "server"
    files.mkdir()
    (files / "blob.bin").write_bytes(b"x" * 5000)
    server = FakeServer(files, honour_range=False)
    try:
        dest = tmp_path / "out/blob.bin"
        dest.parent.mkdir()
        (tmp_path / "out/blob.bin.part").write_bytes(b"y" * 100)  # stale partial content
        dl.download_file(server.url("blob.bin"), dest, 5000)
        assert dest.read_bytes() == b"x" * 5000
    finally:
        server.close()


def test_size_mismatch_keeps_partial_file(served, tmp_path):
    server, _ = served
    dest = tmp_path / "out/blob.bin"
    with pytest.raises(dl.DownloadError, match="expected"):
        dl.download_file(server.url("blob.bin"), dest, 999_999)
    assert dest.with_name("blob.bin.part").exists()
    assert not dest.exists()


def test_published_sha256_mismatch_fails(served, tmp_path):
    server, files = served
    spec = _spec(server, files)
    wrong = FileSpec("blob", spec.files[0].url, "blob.bin", spec.files[0].size, sha256="0" * 64)
    with pytest.raises(dl.DownloadError, match="published"):
        dl.fetch_dataset(_spec(server, files, files=(wrong,)), tmp_path / "data")


def test_verify_detects_corruption(served, tmp_path):
    server, files = served
    root = tmp_path / "data"
    spec = _spec(server, files)
    dl.fetch_dataset(spec, root)
    blob = root / "raw/fake/blob.bin"
    blob.write_bytes(b"\0" * blob.stat().st_size)  # same size, different content

    dl.fetch_dataset(spec, root)  # size-only check passes silently
    with pytest.raises(dl.DownloadError, match="recorded on first download"):
        dl.fetch_dataset(spec, root, verify=True)


def test_unknown_file_name_rejected(served, tmp_path):
    server, files = served
    with pytest.raises(dl.DownloadError, match="unknown file names"):
        dl.fetch_dataset(_spec(server, files), tmp_path / "data", only={"nope"})


def test_partial_selection_keeps_other_records(served, tmp_path):
    server, files = served
    root = tmp_path / "data"
    spec = _spec(server, files)
    dl.fetch_dataset(spec, root)
    manifest = dl.fetch_dataset(spec, root, only={"blob"})
    assert set(manifest["files"]) == {"blob", "imgs"}


def test_disk_space_check(monkeypatch, tmp_path):
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        dl.shutil, "disk_usage", lambda _p: usage._replace(free=dl.DISK_MARGIN_BYTES + 10)
    )
    with pytest.raises(dl.DownloadError, match="not enough disk space"):
        dl.check_disk_space(tmp_path, needed=100)


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.tgz"
    _make_tgz(archive, {"../escape.txt": b"x"})
    with pytest.raises(dl.DownloadError, match="escapes"):
        dl.extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_zip(tmp_path):
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("x/1.jpg", b"1")
        zf.writestr("x/2.jpg", b"2")
    assert dl.extract_archive(archive, tmp_path / "out") == 2


def test_plan_reports_status(served, tmp_path):
    server, files = served
    root = tmp_path / "data"
    spec = _spec(server, files)
    assert {p.status for p in dl.plan_dataset(spec, root)} == {"missing"}
    dl.fetch_dataset(spec, root, only={"blob"})
    status = {p.spec.name: p.status for p in dl.plan_dataset(spec, root)}
    assert status == {"blob": "present", "imgs": "missing"}


# ---------------------------------------------------------------------------------- sources


def test_repo_sources_file_is_valid(repo_root):
    sources = load_sources(repo_root / "data" / "sources.yaml")
    assert "hiertext" in sources
    assert "synthtext" not in sources  # ADR-0009
    for spec in sources.values():
        assert all(f.url.startswith("https://") for f in spec.files)


def _one(**file_overrides: object) -> dict:
    f = {"name": "a", "url": "https://x.org/a", "filename": "a.bin"}
    f.update(file_overrides)
    return {
        "d": {
            "title": "t", "version": "1", "homepage": "h", "licence": "l", "citation": "c",
            "role": "r", "phase": 2, "files": [f],
        }
    }  # fmt: skip


@pytest.mark.parametrize(
    ("overrides", "msg"),
    [
        ({"url": "http://x.org/a"}, "https"),
        ({"filename": "../a.bin"}, "relative path"),
        ({"filename": "/abs/a.bin"}, "relative path"),
        ({"size": -1}, "positive int"),
        ({"sha256": "abc"}, "64 lowercase hex"),
        ({"md5": "x"}, "unknown keys"),
        ({"extract_to": "../up"}, "relative path"),
    ],
)
def test_bad_sources_rejected(overrides, msg):
    with pytest.raises(SourceError, match=msg):
        parse_sources(_one(**overrides))


def test_duplicate_file_names_rejected():
    data = _one()
    data["d"]["files"].append(dict(data["d"]["files"][0]))
    with pytest.raises(SourceError, match="duplicate"):
        parse_sources(data)


# ------------------------------------------------------------------- counters and registry


def test_count_hiertext_both_layouts(tmp_path):
    lines = [
        {"legible": True, "words": [{"legible": True}, {"legible": False}]},
        {"legible": False, "handwritten": True, "words": [{}]},
    ]
    rec = {"image_id": "1", "paragraphs": [{"lines": lines}]}
    raw = tmp_path / "hiertext"
    (raw / "gt").mkdir(parents=True)
    (raw / "images/train").mkdir(parents=True)
    (raw / "images/train/1.jpg").write_bytes(b"x")
    with gzip.open(raw / "gt/train.jsonl.gz", "wt", encoding="utf-8") as fh:
        json.dump({"info": {}, "annotations": [rec, rec]}, fh)
    with gzip.open(raw / "gt/validation.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")

    counts = count_hiertext(raw)
    assert counts["train"] == {
        "image_files": 1,
        "annotations": {
            "images": 2,
            "paragraphs": 2,
            "lines": 4,
            "lines_legible": 2,
            "lines_handwritten": 2,
            "words": 6,
            "words_legible": 4,
        },
    }
    assert counts["validation"]["annotations"]["images"] == 1
    assert counts["test"] == {"image_files": 0}


def test_dataset_manifest_yaml(served, tmp_path):
    server, files = served
    root = tmp_path / "data"
    dl.fetch_dataset(_spec(server, files), root, counter=lambda raw: {"images": 2})
    (root / "manifests/charset.json").write_text("{}", encoding="utf-8")  # ignored

    records = build_dataset_manifest(root)
    assert len(records) == 1
    assert records[0]["dataset"] == "fake"
    assert records[0]["samples"] == {"images": 2}
    assert set(records[0]) >= {"version", "source", "license", "download_date", "split"}
    out = write_dataset_manifest(root)
    assert yaml.safe_load(out.read_text(encoding="utf-8"))[0]["license"] == "CC0"


# ---------------------------------------------------------------------------------------- CLI


def test_cli_dry_run_and_full_run(served, tmp_path, capsys):
    server, files = served
    spec = _spec(server, files)
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        yaml.safe_dump(
            {
                "fake": {
                    "title": spec.title, "version": spec.version, "homepage": spec.homepage,
                    "licence": spec.licence, "citation": spec.citation, "role": spec.role,
                    "phase": 2,
                    "files": [
                        {"name": "blob", "url": server.url("blob.bin").replace("http", "https"),
                         "filename": "blob.bin"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )  # fmt: skip
    root = tmp_path / "data"
    args = ["--sources", str(sources), "--data-root", str(root)]

    assert download_datasets.main([*args, "--list"]) == 0
    assert download_datasets.main([*args, "--dataset", "fake", "--dry-run"]) == 0
    assert "missing" in capsys.readouterr().out
    assert download_datasets.main([*args, "--dataset", "nope"]) == 2
    assert not (root / "raw").exists()  # dry run fetched nothing
