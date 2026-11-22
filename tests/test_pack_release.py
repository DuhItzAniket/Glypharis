"""Phase 23, item 7: the release packer (`scripts/pack_release.py`) on a tiny bundle built from
untrained models, and the weight notices under ``release/weights``. Home, the working directory and
GLYPHARIS_MODELS are redirected to the test's temporary folder (the ``places`` fixture of
``test_hub``), so the install test never reaches the real ~/.glypharis/models or ./models."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import shutil
import tarfile
import zlib
from pathlib import Path
from typing import Any

import pack_release as pr
import pytest
import torch
from test_hub import LocalServer, places  # noqa: F401  (places is a fixture)
from test_pipeline import tiny_detector, tiny_recognizer

from glypharis import hub
from glypharis.data.download import extract_archive
from glypharis.export.bundle import read_manifest, write_bundle

pytest.importorskip("onnxruntime")

COMMIT = "0123456789abcdef0123456789abcdef01234567"
EXPECTED_MEMBERS = (
    "glypharis-v1/",
    "glypharis-v1/LICENSE-WEIGHTS",
    "glypharis-v1/NOTICE.md",
    "glypharis-v1/detector.onnx",
    "glypharis-v1/detector.pt",
    "glypharis-v1/manifest.json",
    "glypharis-v1/model_info.json",
    "glypharis-v1/recognizer.onnx",
    "glypharis-v1/recognizer.pt",
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("pack-src")
    det, rec = tiny_detector(), tiny_recognizer()
    torch.save({"model": det.backend.model.state_dict()}, root / "det_ck.pt")
    bundle = write_bundle(root / "det_ck.pt", det.spec, rec, root / "glypharis-v1", "0.0.1-pack")
    return {"bundle": bundle, "manifest_sha": _sha((bundle / "manifest.json").read_bytes())}


def _pack(tiny: dict[str, Any], bundle: Path | None = None) -> pr.Packed:
    return pr.pack(bundle or tiny["bundle"], manifest_sha256=tiny["manifest_sha"], commit=COMMIT)


def _cli(tiny: dict[str, Any], out: Path, *extra: str, bundle: Path | None = None) -> int:
    return pr.main(
        [
            "--bundle",
            str(bundle or tiny["bundle"]),
            "--out",
            str(out),
            "--manifest-sha256",
            tiny["manifest_sha"],
            "--code-commit",
            COMMIT,
            *extra,
        ]
    )


def _copy(tiny: dict[str, Any], tmp_path: Path) -> Path:
    return Path(shutil.copytree(tiny["bundle"], tmp_path / "copy" / "glypharis-v1"))


# ------------------------------------------------------------------------- reproducible bytes


def test_two_packs_are_byte_identical(tiny, tmp_path, capsys):
    a, b = _pack(tiny), _pack(tiny)
    assert a.data == b.data and a.sha256 == b.sha256

    assert _cli(tiny, tmp_path / "one") == 0
    assert _cli(tiny, tmp_path / "two") == 0
    one = (tmp_path / "one" / pr.ARCHIVE).read_bytes()
    assert one == (tmp_path / "two" / pr.ARCHIVE).read_bytes() == a.data
    sums = (tmp_path / "one" / f"{pr.ARCHIVE}.sha256").read_bytes()
    assert sums == f"{_sha(one)}  glypharis-v1.tar.gz\n".encode("ascii")
    assert sums == (tmp_path / "two" / f"{pr.ARCHIVE}.sha256").read_bytes()
    out = capsys.readouterr().out
    assert f"{len(one)} bytes, SHA-256 {_sha(one)} (written)" in out

    # Packing again into the same folder leaves the archive as it is.
    assert _cli(tiny, tmp_path / "one") == 0
    assert "(unchanged)" in capsys.readouterr().out


def test_members_order_and_contents(tiny):
    packed = _pack(tiny)
    assert packed.members == EXPECTED_MEMBERS
    with tarfile.open(fileobj=io.BytesIO(packed.data), mode="r:gz") as tar:
        members = tar.getmembers()
        assert [m.name + ("/" if m.isdir() else "") for m in members] == list(EXPECTED_MEMBERS)
        bundle = tiny["bundle"]
        for name in pr.BUNDLE_FILES:
            fh = tar.extractfile(f"glypharis-v1/{name}")
            assert fh is not None and fh.read() == (bundle / name).read_bytes()
        for name in pr.NOTICE_FILES:
            fh = tar.extractfile(f"glypharis-v1/{name}")
            assert fh is not None and fh.read() == (pr.NOTICES_DIR / name).read_bytes()
        assert "MODEL_CARD.md" not in {Path(m.name).name for m in members}


def test_fixed_metadata(tiny):
    packed = _pack(tiny)
    head = packed.data[:10]
    assert head[:3] == b"\x1f\x8b\x08"  # gzip, deflate
    assert head[3] == 0  # no file name, comment or extra field
    assert head[4:8] == b"\0\0\0\0"  # mtime 0
    with tarfile.open(fileobj=io.BytesIO(packed.data), mode="r:gz") as tar:
        assert tar.format == tarfile.PAX_FORMAT or tar.pax_headers == {}
        for m in tar.getmembers():
            assert m.isfile() or (m.isdir() and m.name == "glypharis-v1")
            assert (m.mtime, m.uid, m.gid, m.uname, m.gname) == (0, 0, 0, "", "")
            assert m.mode == (0o755 if m.isdir() else 0o644)
            assert m.pax_headers == {}


def test_model_info_is_deterministic_and_traced(tiny):
    packed = _pack(tiny)
    raw = packed.sources[pr.INFO_FILE]
    assert b"\r" not in raw and raw.endswith(b"\n")
    info = json.loads(raw)
    assert raw.decode("utf-8") == json.dumps(info, indent=1, sort_keys=True) + "\n"
    assert info["name"] == "glypharis-v1"
    assert info["model_version"] == "0.0.1-pack"  # the manifest's version
    assert info["manifest_sha256"] == tiny["manifest_sha"]
    assert info["files"] == {f: _sha((tiny["bundle"] / f).read_bytes()) for f in pr.BUNDLE_FILES}
    assert info["config"] == "configs/final/glypharis_v1.yaml"
    assert info["experiments"] == {"detector": "GW-DET-001", "recognizer": "GW-REC-034"}
    assert info["code"] == {
        "repository": "https://github.com/DuhItzAniket/Glypharis",
        "tag": "v1-code",
        "commit": COMMIT,
    }
    assert info["model_card"] == "https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md"
    assert info["terms"] == "CC BY-SA 4.0 (see LICENSE-WEIGHTS and NOTICE.md)"
    assert info["training_data"]["recognizer"]["data"] == "HierText train lines"
    assert info["training_data"]["detector"]["data"] == "HierText train pages"
    # No time stamps and no metric values.
    text = raw.decode("utf-8").lower()
    for word in ("utc", "date", "time", "cer", "wer", "f1"):
        assert f'"{word}' not in text


def test_the_v1_manifest_must_match_model_info():
    manifest = {
        "version": "glypharis_v1",
        "detector": {"spec": {"model": {"init": "imagenet", "weights_sha256": pr.BACKBONE_SHA256}}},
        "sources": {
            "record": "configs/final/glypharis_v1.yaml",
            "detector": {"experiment": "GW-DET-001"},
            "recognizer": {"experiment": "GW-REC-034"},
        },
    }
    pr.check_v1_manifest(manifest)
    manifest["sources"]["recognizer"]["experiment"] = "GW-REC-035"
    with pytest.raises(pr.PackError, match=r"sources\.recognizer\.experiment"):
        pr.check_v1_manifest(manifest)


# --------------------------------------------------------------------------------- refusals


def test_refuses_files_that_do_not_match_the_manifest(tiny, tmp_path):
    bundle = _copy(tiny, tmp_path)
    with (bundle / "recognizer.onnx").open("ab") as fh:
        fh.write(b"\0")
    with pytest.raises(pr.PackError, match=r"recognizer\.onnx.*does not match manifest\.json"):
        _pack(tiny, bundle)
    assert _cli(tiny, tmp_path / "out", bundle=bundle) == 1
    assert not (tmp_path / "out").exists()


def test_refuses_a_bundle_with_other_files(tiny, tmp_path):
    bundle = _copy(tiny, tmp_path)
    (bundle / hub.RECORD_FILE).write_text("{}\n", encoding="utf-8")
    with pytest.raises(pr.PackError, match="nothing else"):
        _pack(tiny, bundle)
    (bundle / hub.RECORD_FILE).unlink()
    (bundle / "detector.pt").unlink()
    with pytest.raises(pr.PackError, match="expected exactly"):
        _pack(tiny, bundle)


def test_refuses_a_wrong_manifest_pin(tiny, tmp_path):
    with pytest.raises(pr.PackError, match="is not the pinned"):
        pr.pack(tiny["bundle"], commit=COMMIT)  # the v1 pin, not the tiny bundle's
    assert pr.main(["--bundle", str(tiny["bundle"]), "--out", str(tmp_path / "o")]) == 1
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize("kind", ["dir", "file"])
def test_refuses_an_out_folder_inside_a_git_worktree(tiny, tmp_path, kind):
    repo = tmp_path / "repo"
    repo.mkdir()
    if kind == "dir":
        (repo / ".git").mkdir()
    else:  # a linked worktree has a .git file
        (repo / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    out = repo / "nested" / "out"
    with pytest.raises(pr.PackError, match="inside the git worktree"):
        pr.write(_pack(tiny), out)
    assert _cli(tiny, out) == 1
    assert not (repo / "nested").exists()
    assert pr.git_worktree_of(pr.REPO_ROOT / "runs") is not None  # this checkout is one too


def test_refuses_to_replace_an_archive_with_other_bytes(tiny, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / pr.ARCHIVE).write_bytes(b"something else")
    assert _cli(tiny, out) == 1
    assert (out / pr.ARCHIVE).read_bytes() == b"something else"
    assert _cli(tiny, out, "--overwrite") == 0
    assert (out / pr.ARCHIVE).read_bytes() == _pack(tiny).data


def test_check_only_writes_nothing_and_compares(tiny, tmp_path):
    out = tmp_path / "out"
    assert _cli(tiny, out, "--check-only") == 0
    assert not out.exists()
    assert _cli(tiny, out) == 0
    assert _cli(tiny, out, "--check-only") == 0
    data = (out / pr.ARCHIVE).read_bytes()
    (out / pr.ARCHIVE).write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    assert _cli(tiny, out, "--check-only") == 1


# ---------------------------------------------------------------------------- the check report


def _report(tiny: dict[str, Any], out: Path, path: Path) -> tuple[int, dict[str, Any], str]:
    code = _cli(tiny, out, "--check-only", "--report", str(path))
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    return code, json.loads(raw), path.with_suffix(".md").read_text(encoding="utf-8")


def test_check_only_report_records_the_comparison(tiny, tmp_path):
    out = tmp_path / "out"
    path = tmp_path / "reports" / "release" / "pack_check.json"

    # Nothing in --out yet: the report says so, and --out stays untouched.
    code, report, _ = _report(tiny, out, path)
    assert code == 0 and not out.exists()
    assert report["rebuilt_identical"] is None and report["sha256_file_matches"] is None
    assert report["passed"] is True

    assert _cli(tiny, out) == 0
    packed = _pack(tiny)
    code, report, md = _report(tiny, out, path)
    assert code == 0
    assert report["format"] == "glypharis-pack-check-1"
    assert report["archive"] == "glypharis-v1.tar.gz"
    assert report["bytes"] == packed.size == (out / pr.ARCHIVE).stat().st_size
    assert report["sha256"] == packed.sha256
    assert report["members"] == list(EXPECTED_MEMBERS)
    assert report["rebuilt_identical"] is True and report["sha256_file_matches"] is True
    assert report["manifest_sha256"] == tiny["manifest_sha"]
    assert report["manifest_is_v1_pin"] is False
    # The registry pins the v1 manifest only, so the tiny pack has no pin to match.
    assert report["registry_manifest_sha256"] == hub.REGISTRY["glypharis-v1"].manifest_sha256
    for key in (
        "registry_bytes",
        "registry_sha256",
        "registry_bytes_match",
        "registry_sha256_match",
    ):
        assert report[key] is None, key
    assert report["notices_sha256"] == {
        name: _sha((pr.NOTICES_DIR / name).read_bytes()) for name in pr.NOTICE_FILES
    }
    assert report["model_info_sha256"] == _sha(packed.sources[pr.INFO_FILE])
    assert report["code"] == {"tag": "v1-code", "commit": COMMIT}
    assert report["python"] == platform.python_version()
    assert report["zlib"] == zlib.ZLIB_RUNTIME_VERSION
    assert report["passed"] is True

    # One line of Markdown, and the same inputs give the same files.
    assert md.endswith("\n") and md.count("\n") == 1
    assert f"{packed.size:,} bytes" in md and packed.sha256 in md and COMMIT in md
    assert "out folder: yes" in md and "passed" in md and "`pack_check.json`" in md
    before = path.read_bytes(), md
    assert _report(tiny, out, path)[0] == 0
    assert (path.read_bytes(), path.with_suffix(".md").read_text(encoding="utf-8")) == before

    # No local path: not the temporary folders, the bundle, --out or home.
    for text in (path.read_text(encoding="utf-8"), md):
        assert pr.LOCAL_PATH.search(text) is None
        for place in (tmp_path, tiny["bundle"], out, Path.home()):
            assert str(place) not in text and place.as_posix() not in text

    # A changed archive is recorded, and the check fails; the report is still written.
    data = (out / pr.ARCHIVE).read_bytes()
    (out / pr.ARCHIVE).write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    code, report, md = _report(tiny, out, path)
    assert code == 1
    assert report["rebuilt_identical"] is False and report["sha256_file_matches"] is True
    assert report["passed"] is False and "FAILED" in md


def test_check_only_report_compares_the_registry_pins(tiny, tmp_path):
    packed = _pack(tiny)
    path = tmp_path / "report" / "pack_check.json"

    def entry(size: int, sha256: str) -> hub.ModelEntry:
        return hub.ModelEntry(
            name="glypharis-v1",
            url="https://example.invalid/glypharis-v1.tar.gz",
            archive=pr.ARCHIVE,
            bytes=size,
            archive_sha256=sha256,
            manifest_sha256=tiny["manifest_sha"],
            top_folder="glypharis-v1",
        )

    with hub.override_registry(entry(packed.size, packed.sha256)):
        code, report, md = _report(tiny, tmp_path / "out", path)
    assert code == 0
    assert (report["registry_bytes"], report["registry_sha256"]) == (packed.size, packed.sha256)
    assert report["registry_manifest_sha256"] == tiny["manifest_sha"]
    assert report["registry_bytes_match"] is True and report["registry_sha256_match"] is True
    assert "registry pins match: bytes yes, SHA-256 yes" in md

    with hub.override_registry(entry(packed.size + 1, packed.sha256)):
        code, report, md = _report(tiny, tmp_path / "out", path)
    assert code == 1
    assert report["registry_bytes"] == packed.size + 1
    assert report["registry_bytes_match"] is False and report["registry_sha256_match"] is True
    assert report["passed"] is False and "bytes no, SHA-256 yes" in md


def test_check_only_report_refusals(tiny, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        _cli(tiny, out, "--report", str(tmp_path / "r.json"))  # needs --check-only
    assert _cli(tiny, out, "--check-only", "--report", str(tmp_path / "r.md")) == 1
    out.mkdir()
    assert _cli(tiny, out, "--check-only", "--report", str(out / "r.json")) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]
    assert list(out.iterdir()) == []

    # A refused pack writes no report.
    bundle = _copy(tiny, tmp_path)
    (bundle / "detector.pt").unlink()
    assert _cli(tiny, out, "--check-only", "--report", str(tmp_path / "r.json"), bundle=bundle) == 1
    assert not (tmp_path / "r.json").exists()

    # The last guard: a report that would hold a local path is not written.
    packed = _pack(tiny)
    report = pr.check_report(packed, pr.compare_existing(packed, out), tiny["manifest_sha"])
    for place in ("C:/Users/someone/models", "D:\\runs", "/home/someone/models", "/tmp/x/"):
        with pytest.raises(pr.PackError, match="local path"):
            pr.write_report({**report, "note": place}, tmp_path / "x" / "r.json", out)
    assert not (tmp_path / "x").exists()


def test_notice_line_endings_are_checked(tiny, tmp_path):
    notices = tmp_path / "notices"
    notices.mkdir()
    for name in pr.NOTICE_FILES:
        (notices / name).write_bytes((pr.NOTICES_DIR / name).read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(pr.PackError, match="LF"):
        pr.pack(
            tiny["bundle"], manifest_sha256=tiny["manifest_sha"], notices_dir=notices, commit=COMMIT
        )


# ---------------------------------------------------------------------------- extract, install


def test_the_archive_extracts_and_installs_through_download_model(tiny, tmp_path, places):  # noqa: F811
    assert _cli(tiny, tmp_path / "out") == 0
    archive = tmp_path / "out" / pr.ARCHIVE
    data = archive.read_bytes()

    dest = tmp_path / "extracted"
    assert extract_archive(archive, dest) == len(EXPECTED_MEMBERS) - 1
    read_manifest(dest / "glypharis-v1")
    assert _sha((dest / "glypharis-v1" / "manifest.json").read_bytes()) == tiny["manifest_sha"]

    srv = LocalServer({pr.ARCHIVE: data})
    try:
        entry = hub.ModelEntry(
            name="glypharis-v1",
            url=srv.url(pr.ARCHIVE),
            archive=pr.ARCHIVE,
            bytes=len(data),
            archive_sha256=_sha(data),
            manifest_sha256=tiny["manifest_sha"],
            top_folder="glypharis-v1",
            published=True,
        )
        with hub.override_registry(entry):
            path = hub.download_model("glypharis-v1")
    finally:
        srv.close()
    assert path == places["models"] / "glypharis-v1"
    read_manifest(path)
    for name in (*pr.BUNDLE_FILES, *pr.NOTICE_FILES):
        assert (path / name).read_bytes() == (dest / "glypharis-v1" / name).read_bytes()
    record = hub.read_record(path)
    assert record is not None and record["archive_sha256"] == _sha(data)
    assert record["network_fetch"] is True


# ------------------------------------------------------------------------------- the notices


def test_the_weight_notices_are_outside_the_package_and_say_what_they_must():
    assert pr.NOTICES_DIR == pr.REPO_ROOT / "release" / "weights"
    assert not pr.NOTICES_DIR.is_relative_to(pr.REPO_ROOT / "glypharis")
    licence = (pr.NOTICES_DIR / "LICENSE-WEIGHTS").read_text(encoding="utf-8")
    notice = (pr.NOTICES_DIR / "NOTICE.md").read_text(encoding="utf-8")
    for text in (licence, notice):
        assert "\r" not in text
        assert "https://creativecommons.org/licenses/by-sa/4.0/legalcode" in text
        assert "never described as cleared for commercial use" in " ".join(text.split())
        assert "Apache License, Version 2.0" in text
    assert "not a\nlegal conclusion" in licence or "not a legal conclusion" in " ".join(
        licence.split()
    )
    assert "NOTICE.md" in licence
    flat = " ".join(notice.split())
    for phrase in (
        "HierText is released under CC BY-SA 4.0",
        "which lists them as CC BY 2.0",
        "ImageNet-1K",
        "ADR-0009",
        "ADR-0014",
        "tag `v1.0`",
        "without warranties",
        "GW-DET-001",
        "GW-REC-034",
    ):
        assert phrase in flat, phrase


def test_the_code_tag_resolves_when_present():
    try:
        commit = pr.code_commit()
    except pr.PackError:
        pytest.skip("the v1-code tag is not in this clone")
    assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
