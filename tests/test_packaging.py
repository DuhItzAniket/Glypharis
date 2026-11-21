"""Phase 23, items 3 and 5: the package metadata, the core/extras split, the wheel and sdist checks
(`scripts/check_wheel.py`) and the `git archive` exclusion check (`scripts/check_git_archive.py`).

Tests that write a model bundle redirect home, the working directory and ``GLYPHARIS_MODELS`` to
``tmp_path``, so nothing reaches ``~/.glypharis/models`` or ``./models``."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import check_git_archive as cga
import check_wheel as cw
import pytest
from packaging.version import Version

import glypharis

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
PROJECT = PYPROJECT["project"]

# Distribution -> top-level module, for the packages that are extras rather than core.
EXTRAS = {
    "train": {
        "matplotlib": "matplotlib",
        "fonttools": "fontTools",
        "pypdfium2": "pypdfium2",
        "psutil": "psutil",
        "nvidia-ml-py": "pynvml",
    },
    "export": {"onnx": "onnx"},
}
EXTRA_MODULES = sorted(m for group in EXTRAS.values() for m in group.values())
INFERENCE_MODULES = [
    "glypharis",
    "glypharis.pipeline",
    "glypharis.export.bundle",
    "glypharis.models",
    "glypharis.cli",
    "glypharis.server",
]
if (ROOT / "glypharis" / "hub.py").is_file():
    INFERENCE_MODULES.append("glypharis.hub")


def _requirement(spec: str) -> tuple[str, str]:
    """(normalised name, the part after the name) of a requirement string."""
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(.*)$", spec)
    assert match, spec
    return re.sub(r"[-_.]+", "-", match[1]).lower(), match[2].strip()


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),
        GLYPHARIS_MODELS=str(tmp_path / "models"),
        PYTHONIOENCODING="utf-8",
        PYTHONPATH=os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep),
    )
    return env


# --------------------------------------------------------------------------- the core/extras split


def test_inference_imports_need_none_of_the_extras(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = (
        "import importlib, json, sys\n"
        f"for m in {INFERENCE_MODULES!r}:\n"
        "    importlib.import_module(m)\n"
        "print(json.dumps(sorted({n.split('.')[0] for n, v in sys.modules.items() if v})))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_isolated_env(tmp_path),
        cwd=tmp_path,
        check=True,
    )
    imported = set(json.loads(done.stdout.strip().splitlines()[-1]))
    assert "glypharis" in imported
    # torch.cuda tries `import pynvml` for itself when it is installed; glypharis does not need it,
    # and the next test reads a page with pynvml blocked.
    assert imported & (set(EXTRA_MODULES) - {"pynvml"}) == set()


def test_a_page_reads_with_core_dependencies_only(tmp_path, monkeypatch):
    pytest.importorskip("onnx")  # writing the tiny bundle exports ONNX files
    pytest.importorskip("onnxruntime")
    monkeypatch.chdir(tmp_path)
    env = _isolated_env(tmp_path)
    script = str(ROOT / "scripts" / "tiny_bundle_check.py")
    out = tmp_path / "work"
    subprocess.run([sys.executable, script, "--out", str(out), "--build-only"], env=env,
                   cwd=tmp_path, check=True, capture_output=True)  # fmt: skip
    done = subprocess.run(
        [sys.executable, script, "--bundle", str(out / "tiny-check"), "--block", *EXTRA_MODULES],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    summary = json.loads(done.stdout[done.stdout.index("{") :])
    assert summary["passed"] and summary["blocked_imported"] == []
    assert set(summary["reads"]) == {"torch", "onnx"}
    assert summary["glypharis_version"] == glypharis.__version__
    assert not (tmp_path / ".glypharis").exists() and not (tmp_path / "models").exists()


def test_extras_hold_only_what_inference_does_not_need():
    core = {_requirement(r)[0] for r in PROJECT["dependencies"]}
    optional = PROJECT["optional-dependencies"]
    for group, packages in EXTRAS.items():
        assert {_requirement(r)[0] for r in optional[group]} == set(packages)
        assert core.isdisjoint(packages)
    assert "onnxruntime" in core  # --backend onnx works with the core install
    assert set(optional) == {"train", "export", "dev"}


def test_every_dependency_has_a_lower_bound_at_the_locked_major_minor():
    lock = {}
    for line in (ROOT / "requirements" / "lock.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==", 1)
            lock[_requirement(name)[0]] = version.strip()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for name, key in (("torch", "TORCH_VERSION"), ("torchvision", "TORCHVISION_VERSION")):
        match = re.search(rf'^\s*{key}:\s*"([^"]+)"', ci, re.M)
        assert match, key
        lock[name] = match[1]
    specs = [*PROJECT["dependencies"]]
    for group in PROJECT["optional-dependencies"].values():
        specs += group
    for spec in specs:
        name, rest = _requirement(spec)
        assert rest.startswith(">="), f"{spec}: no lower bound"
        assert name in lock, f"{spec}: not in requirements/lock.txt"
        release = Version(lock[name]).release
        assert Version(rest[2:]) == Version(".".join(map(str, release[:2]))), spec


# --------------------------------------------------------------------------- metadata


def test_version_is_read_from_the_package():
    assert "version" not in PROJECT and PROJECT["dynamic"] == ["version"]
    assert PYPROJECT["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "glypharis.__version__"
    }
    assert str(Version(glypharis.__version__)) == glypharis.__version__  # canonical PEP 440
    assert cw.source_version(ROOT) == glypharis.__version__


def _installed_from_this_tree() -> importlib.metadata.Distribution | None:
    """The installed distribution whose code is this tree, if any. A ``glypharis.egg-info`` in
    the checkout itself is a build by-product (and may be stale), not an installation."""
    here = Path(glypharis.__file__).resolve()
    for dist in importlib.metadata.distributions(name="glypharis"):
        if Path(str(dist.locate_file(""))).resolve() == ROOT:
            continue
        direct = dist.read_text("direct_url.json")
        if direct:
            url = json.loads(direct).get("url", "")
            if url.startswith("file:") and Path(url2pathname(urlparse(url).path)).resolve() == ROOT:
                return dist
        for f in dist.files or []:
            name = str(f).replace("\\", "/")
            if name == "glypharis/__init__.py" and Path(str(dist.locate_file(f))).resolve() == here:
                return dist
    return None


def test_installed_metadata_matches_the_version():
    dist = _installed_from_this_tree()
    if dist is None:
        pytest.skip("glypharis is not installed from this tree")
    assert importlib.metadata.version("glypharis") == glypharis.__version__
    assert dist.metadata["Version"] == glypharis.__version__


def test_licence_python_and_project_fields():
    assert PROJECT["license"] == "Apache-2.0"
    assert PROJECT["license-files"] == ["LICENSE", "NOTICE"]
    assert PROJECT["requires-python"] == ">=3.11.4,<3.12"
    assert not [c for c in PROJECT["classifiers"] if c.startswith("License ::")]
    assert PROJECT["authors"] == [{"name": "DuhItzAniket"}]
    assert PYPROJECT["build-system"]["requires"] == ["setuptools>=77"]
    assert set(PROJECT["scripts"]) == set(cw.COMMANDS)
    for url in PROJECT["urls"].values():
        assert url.startswith("https://github.com/DuhItzAniket/Glypharis")
    assert (ROOT / PROJECT["readme"]).is_file()


def test_pypi_description_links_are_absolute_and_pinned_to_the_tag():
    text = (ROOT / PROJECT["readme"]).read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)\s]+)\)", text)
    assert links
    repo = "https://github.com/DuhItzAniket/Glypharis/"
    for link in links:
        assert link.startswith("https://"), f"relative link on the PyPI page: {link}"
        if link.startswith(repo + "blob/") or link.startswith(repo + "tree/"):
            rest = link[len(repo) :].split("/", 1)[1]
            tag, _, path = rest.partition("/")
            assert tag == "v1.0", link
            if path and path != "CHANGELOG.md":  # the changelog is item 4's file
                assert (ROOT / path).exists(), link
    assert "not yet published" in text and "download_model" in text


# --------------------------------------------------------------------------- check_wheel


def _metadata(version: str = "1.0.0", **override: str | list[str] | None) -> bytes:
    fields: dict[str, str | list[str] | None] = {
        "Metadata-Version": "2.4",
        "Name": "glypharis",
        "Version": version,
        "Requires-Python": "<3.12,>=3.11.4",
        "License-Expression": "Apache-2.0",
        "License-File": ["LICENSE", "NOTICE"],
        "Classifier": ["Programming Language :: Python :: 3.11"],
    }
    fields.update(override)
    lines = []
    for key, value in fields.items():
        for v in [value] if isinstance(value, str) else value or []:
            lines.append(f"{key}: {v}")
    return ("\n".join(lines) + "\n\nThe description.\n").encode("utf-8")


def _source(tmp_path: Path, version: str = "1.0.0") -> Path:
    root = tmp_path / "src"
    (root / "glypharis" / "sub").mkdir(parents=True)
    (root / "glypharis" / "__init__.py").write_text(f'__version__ = "{version}"\n', "utf-8")
    (root / "glypharis" / "sub" / "__init__.py").write_text("x = 1\n", "utf-8")
    return root


def _wheel(path: Path, root: Path, extra: dict[str, bytes] | None = None,
           drop: tuple[str, ...] = (), meta: bytes | None = None) -> Path:  # fmt: skip
    info = "glypharis-1.0.0.dist-info/"
    files = {
        p.relative_to(root).as_posix(): p.read_bytes() for p in (root / "glypharis").rglob("*.py")
    }
    files |= {
        info + "METADATA": meta or _metadata(),
        info + "WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        info + "entry_points.txt": b"[console_scripts]\nglypharis-ocr = glypharis.cli:main\n"
        b"glypharis-serve = glypharis.server:main\n",
        info + "top_level.txt": b"glypharis\n",
        info + "licenses/LICENSE": b"Apache License\n",
        info + "licenses/NOTICE": b"GLYPHARIS\n",
        **(extra or {}),
    }
    for name in drop:
        files.pop(name)
    record = io.StringIO()
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.write(f"{name},sha256={digest},{len(data)}\n")
    record.write(info + "RECORD,,\n")
    files[info + "RECORD"] = record.getvalue().encode("utf-8")
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path


def test_check_wheel_accepts_a_clean_wheel(tmp_path):
    root = _source(tmp_path)
    whl = _wheel(tmp_path / "glypharis-1.0.0-py3-none-any.whl", root)
    assert cw.check_wheel(whl, "1.0.0", cw.source_tree(root)) == []
    assert cw.main([str(whl), "--root", str(root)]) == 0


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"extra": {"glypharis/detector.onnx": b"w"}}, "forbidden file"),
        ({"extra": {"glypharis/NOTICE.md": b"w"}}, "forbidden file"),
        ({"extra": {"LICENSE-WEIGHTS": b"w"}}, "forbidden file"),
        ({"extra": {"tests/test_x.py": b"x"}}, "stray file"),
        ({"extra": {"glypharis/new.py": b"x"}}, "not in the source tree"),
        ({"drop": ("glypharis/sub/__init__.py",)}, "not in the package"),
        ({"drop": ("glypharis-1.0.0.dist-info/licenses/NOTICE",)}, "dist-info has no"),
        ({"meta": _metadata(**{"Requires-Python": ">=3.11"})}, "Requires-Python"),
        ({"meta": _metadata(**{"License-Expression": "MIT"})}, "License-Expression"),
        ({"meta": _metadata(**{"License-File": ["LICENSE"]})}, "License-File"),
        ({"meta": _metadata(Version="1.0.1")}, "Version is"),
        ({"meta": _metadata(**{"Metadata-Version": "2.1"})}, "below 2.4"),
        (
            {"meta": _metadata(Classifier=["License :: OSI Approved :: Apache Software License"])},
            "licence classifiers",
        ),
    ],
)
def test_check_wheel_finds_each_problem(tmp_path, change, expected):
    root = _source(tmp_path)
    whl = _wheel(tmp_path / "glypharis-1.0.0-py3-none-any.whl", root, **change)
    problems = cw.check_wheel(whl, "1.0.0", cw.source_tree(root))
    assert any(expected in p for p in problems), problems
    assert cw.main([str(whl), "--root", str(root)]) == 1


def test_check_wheel_finds_a_changed_file_and_a_bad_record(tmp_path):
    root = _source(tmp_path)
    whl = _wheel(tmp_path / "glypharis-1.0.0-py3-none-any.whl", root)
    (root / "glypharis" / "sub" / "__init__.py").write_text("x = 2\n", "utf-8")
    assert any("differs" in p for p in cw.check_wheel(whl, "1.0.0", cw.source_tree(root)))
    with zipfile.ZipFile(whl, "a") as zf:
        zf.writestr("glypharis/late.py", b"y = 1\n")
    assert any("not in RECORD" in p for p in cw.check_wheel(whl, "1.0.0", None))


def _sdist(path: Path, root: Path, extra: dict[str, bytes] | None = None) -> Path:
    files = {
        p.relative_to(root).as_posix(): p.read_bytes() for p in (root / "glypharis").rglob("*.py")
    }
    files |= {"PKG-INFO": _metadata(), "pyproject.toml": b"[project]\n", "LICENSE": b"L\n",
              "NOTICE": b"N\n", **(extra or {})}  # fmt: skip
    with tarfile.open(path, "w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(f"glypharis-1.0.0/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_check_wheel_checks_the_sdist(tmp_path):
    root = _source(tmp_path)
    good = _sdist(tmp_path / "glypharis-1.0.0.tar.gz", root)
    assert cw.check_sdist(good, "1.0.0", cw.source_tree(root)) == []
    (tmp_path / "bad").mkdir()
    bad = _sdist(tmp_path / "bad" / "glypharis-1.0.0.tar.gz", root,
                 {"weights/recognizer.pt": b"w"})  # fmt: skip
    assert any("forbidden" in p for p in cw.check_sdist(bad, "1.0.0", cw.source_tree(root)))


# --------------------------------------------------------------------------- check_git_archive


def test_gitattributes_export_ignore_matches_the_check():
    patterns = set()
    for line in (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and "export-ignore" in parts[1:]:
            patterns.add(parts[0].strip("/"))
    assert patterns == set(cga.EXCLUDED)
    try:
        tracked = cga.commit_files(ROOT, "HEAD")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    for prefix in cga.EXCLUDED:
        assert any(cga._under(f, prefix) for f in tracked), prefix
    for kept in ("data/manifests/processed/hiertext.json", "data/manifests/charset.json"):
        assert not any(cga._under(kept, p) for p in cga.EXCLUDED)


def test_check_finds_leaks_missing_files_and_stale_exclusions():
    committed = ["glypharis/a.py", "docs/x.md", "README.md", "pyproject.toml", "reports/r.json"]
    archived = [f for f in committed if not f.startswith("reports/")]
    assert cga.check(committed, archived, excluded=("reports",)) == []
    assert any("in the archive" in p for p in cga.check(committed, committed, ("reports",)))
    assert any("left out" in p for p in cga.check(committed, archived[1:], ("reports",)))
    assert any("not in the commit" in p for p in cga.check(committed, archived, ("gone",)))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.mark.parametrize("ignored", [True, False])
def test_git_archive_check_on_a_repository(tmp_path, ignored):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    for path in (*cga.KEPT, *cga.EXCLUDED, "data/manifests/charset.json"):
        file = repo / path if Path(path).suffix else repo / path / "f.txt"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("x\n", encoding="utf-8")
    if ignored:
        lines = [f"/{p} export-ignore" for p in cga.EXCLUDED]
        (repo / ".gitattributes").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "files")
    assert cga.main(["--root", str(repo)]) == (0 if ignored else 1)
    archived = cga.archive_files(repo, "HEAD")
    assert ("data/manifests/charset.json" in archived) and ("README.md" in archived)
    assert any(f.startswith("reports/") for f in archived) is not ignored
