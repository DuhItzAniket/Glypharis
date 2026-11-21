"""Check the built wheel and sdist before anything is uploaded (Phase 23, item 3).

    python scripts/check_wheel.py dist/glypharis-1.0.0-py3-none-any.whl
        [dist/glypharis-1.0.0.tar.gz] [--root .] [--version 1.0.0] [--no-tree]

For a wheel:

- every member is under ``glypharis/`` or ``glypharis-<version>.dist-info/``, and the dist-info
  holds only the standard files and ``licenses/LICENSE`` and ``licenses/NOTICE``;
- no weight file (``.pt``, ``.onnx``, ...), weight notice (``LICENSE-WEIGHTS``, ``NOTICE.md``),
  archive or bytecode anywhere;
- ``glypharis/`` equals the source tree it was built from, file for file and byte for byte (the
  files ``git ls-files glypharis`` lists under ``--root``, or every file there outside
  ``__pycache__`` when ``--root`` is not a git checkout);
- every member matches its line in ``RECORD``;
- the metadata: ``Name`` glypharis, ``Version`` equal to ``glypharis/__init__.py`` (or
  ``--version``), ``Requires-Python`` ``>=3.11.4,<3.12``, ``License-Expression`` ``Apache-2.0``
  with ``License-File`` LICENSE and NOTICE, no ``License ::`` classifier, and both commands in
  ``entry_points.txt``.

For an sdist (``.tar.gz``): one top folder ``glypharis-<version>/``, regular files only, no weight
file or weight notice, ``PKG-INFO`` with the same metadata rules, ``LICENSE``, ``NOTICE`` and
``pyproject.toml`` present, and ``glypharis/`` equal to the source tree.

Exits 1 and lists every problem when a check fails.
"""

from __future__ import annotations

import argparse
import base64
import csv
import email.parser
import hashlib
import io
import re
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Iterable, Sequence
from email.message import Message
from pathlib import Path, PurePosixPath

NAME = "glypharis"
REQUIRES_PYTHON = frozenset({">=3.11.4", "<3.12"})
LICENSE_EXPRESSION = "Apache-2.0"
LICENSE_FILES = frozenset({"LICENSE", "NOTICE"})
COMMANDS = ("glypharis-ocr", "glypharis-serve")
# Nothing with these suffixes or names may enter a code package.
FORBIDDEN_SUFFIXES = frozenset(
    {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".npz", ".lmdb", ".pyc", ".pyo"}
    | {".tar", ".gz", ".tgz", ".zip", ".whl"}
)
FORBIDDEN_NAMES = frozenset({"LICENSE-WEIGHTS", "NOTICE.md", "manifest.json", "__pycache__"})
DIST_INFO_FILES = frozenset(
    {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt"}
    | {f"licenses/{f}" for f in LICENSE_FILES}
)
DIST_INFO_REQUIRED = frozenset({"METADATA", "WHEEL", "RECORD", "entry_points.txt"}) | {
    f"licenses/{f}" for f in LICENSE_FILES
}
_VERSION = re.compile(r'^__version__\s*=\s*"([^"]+)"\s*$', re.M)


def source_version(root: Path) -> str:
    match = _VERSION.search((root / NAME / "__init__.py").read_text(encoding="utf-8"))
    if not match:
        raise SystemExit(f"no __version__ literal in {root / NAME / '__init__.py'}")
    return match[1]


def source_tree(root: Path) -> dict[str, bytes]:
    """The package files the wheel should hold: the files under ``glypharis/`` that git tracks or
    would track (not ignored), or every file outside ``__pycache__`` when ``root`` is not a git
    checkout. In a clean checkout these are the tracked files."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "-co", "--exclude-standard", "--", NAME],
            check=True,
            capture_output=True,
        ).stdout.decode("utf-8")
        names = [n for n in out.split("\0") if n]
    except (OSError, subprocess.CalledProcessError):
        names = [
            p.relative_to(root).as_posix()
            for p in sorted((root / NAME).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        ]
    return {n: (root / n).read_bytes() for n in names if (root / n).is_file()}


def forbidden(names: Iterable[str]) -> list[str]:
    bad = []
    for name in names:
        path = PurePosixPath(name)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or FORBIDDEN_NAMES & set(path.parts):
            bad.append(f"forbidden file in the package: {name}")
    return bad


def check_metadata(meta: Message, version: str, label: str) -> list[str]:
    problems = []
    if meta.get("Name") != NAME:
        problems.append(f"{label}: Name is {meta.get('Name')!r}, expected {NAME!r}")
    if meta.get("Version") != version:
        problems.append(f"{label}: Version is {meta.get('Version')!r}, expected {version!r}")
    meta_version = tuple(int(p) for p in str(meta.get("Metadata-Version", "0")).split("."))
    if meta_version < (2, 4):
        problems.append(f"{label}: Metadata-Version {meta.get('Metadata-Version')} is below 2.4")
    requires = {s.strip() for s in str(meta.get("Requires-Python", "")).split(",") if s.strip()}
    if requires != REQUIRES_PYTHON:
        problems.append(f"{label}: Requires-Python is {meta.get('Requires-Python')!r}, expected "
                        f"{','.join(sorted(REQUIRES_PYTHON))!r}")  # fmt: skip
    if meta.get("License-Expression") != LICENSE_EXPRESSION:
        problems.append(f"{label}: License-Expression is {meta.get('License-Expression')!r}")
    if meta.get("License") is not None:
        problems.append(f"{label}: has a License field beside License-Expression")
    files = set(meta.get_all("License-File") or [])
    if files != LICENSE_FILES:
        problems.append(f"{label}: License-File is {sorted(files)}, expected LICENSE and NOTICE")
    licence_classifiers = [
        c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")
    ]
    if licence_classifiers:
        problems.append(
            f"{label}: licence classifiers beside an SPDX licence: {licence_classifiers}"
        )
    return problems


def _compare_tree(members: dict[str, bytes], tree: dict[str, bytes], label: str) -> list[str]:
    problems = []
    for name in sorted(set(tree) - set(members)):
        problems.append(f"{label}: {name} is in the source tree but not in the package")
    for name in sorted(set(members) - set(tree)):
        problems.append(f"{label}: {name} is in the package but not in the source tree")
    for name in sorted(set(members) & set(tree)):
        if members[name] != tree[name]:
            problems.append(f"{label}: {name} differs from the source tree")
    return problems


def _record_hash(data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return "sha256=" + digest.decode("ascii")


def check_wheel(path: Path, version: str, tree: dict[str, bytes] | None) -> list[str]:
    label = path.name
    want_name = f"{NAME}-{version}-py3-none-any.whl"
    problems = [] if path.name == want_name else [f"{label}: expected the file name {want_name}"]
    dist_info = f"{NAME}-{version}.dist-info/"
    with zipfile.ZipFile(path) as zf:
        data = {i.filename: zf.read(i) for i in zf.infolist() if not i.is_dir()}
    problems += forbidden(data)
    package: dict[str, bytes] = {}
    info: dict[str, bytes] = {}
    for name, content in data.items():
        if name.startswith(f"{NAME}/"):
            package[name] = content
        elif name.startswith(dist_info):
            info[name[len(dist_info) :]] = content
        else:
            problems.append(f"{label}: stray file outside the package: {name}")
    for name in sorted(set(info) - DIST_INFO_FILES):
        problems.append(f"{label}: unexpected dist-info file {name}")
    for name in sorted(DIST_INFO_REQUIRED - set(info)):
        problems.append(f"{label}: dist-info has no {name}")
    if "METADATA" in info:
        meta = email.parser.BytesParser().parsebytes(info["METADATA"])
        problems += check_metadata(meta, version, label)
    entry_points = info.get("entry_points.txt", b"").decode("utf-8")
    for command in COMMANDS:
        if not re.search(rf"^{re.escape(command)}\s*=", entry_points, re.M):
            problems.append(f"{label}: entry_points.txt has no {command}")
    if "RECORD" in info:
        rows = {r[0]: r[1] for r in csv.reader(io.StringIO(info["RECORD"].decode("utf-8"))) if r}
        for name, content in data.items():
            if name == dist_info + "RECORD":
                continue
            if name not in rows:
                problems.append(f"{label}: {name} is not in RECORD")
            elif rows[name] != _record_hash(content):
                problems.append(f"{label}: {name} does not match its RECORD hash")
        for name in sorted(set(rows) - set(data)):
            problems.append(f"{label}: RECORD lists {name}, which is not in the wheel")
    if tree is not None:
        problems += _compare_tree(package, tree, label)
    return problems


def check_sdist(path: Path, version: str, tree: dict[str, bytes] | None) -> list[str]:
    label = path.name
    want_name = f"{NAME}-{version}.tar.gz"
    problems = [] if path.name == want_name else [f"{label}: expected the file name {want_name}"]
    top = f"{NAME}-{version}/"
    files: dict[str, bytes] = {}
    with tarfile.open(path, "r:gz") as tf:
        for member in tf.getmembers():
            if member.isdir():
                continue
            if not member.isfile():
                problems.append(f"{label}: {member.name} is not a regular file")
                continue
            if not member.name.startswith(top):
                problems.append(f"{label}: {member.name} is outside {top}")
                continue
            extracted = tf.extractfile(member)
            files[member.name[len(top) :]] = extracted.read() if extracted else b""
    problems += forbidden(files)
    for name in ("PKG-INFO", "pyproject.toml", *sorted(LICENSE_FILES)):
        if name not in files:
            problems.append(f"{label}: has no {name}")
    if "PKG-INFO" in files:
        meta = email.parser.BytesParser().parsebytes(files["PKG-INFO"])
        problems += check_metadata(meta, version, label)
    if tree is not None:
        package = {n: c for n, c in files.items() if n.startswith(f"{NAME}/")}
        problems += _compare_tree(package, tree, label)
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dists", nargs="+", type=Path, help="wheel (.whl) and sdist (.tar.gz)")
    parser.add_argument("--root", type=Path, default=Path("."), help="the source checkout")
    parser.add_argument("--version", help="expected version (default: glypharis/__init__.py)")
    parser.add_argument("--no-tree", action="store_true", help="skip the source-tree comparison")
    args = parser.parse_args(argv)
    version = args.version or source_version(args.root)
    tree = None if args.no_tree else source_tree(args.root)
    problems: list[str] = []
    for dist in args.dists:
        if dist.suffix == ".whl":
            found = check_wheel(dist, version, tree)
        elif dist.name.endswith(".tar.gz"):
            found = check_sdist(dist, version, tree)
        else:
            found = [f"{dist}: neither a wheel nor an sdist"]
        print(f"{dist.name}: {'ok' if not found else f'{len(found)} problem(s)'}")
        problems += found
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    if not problems:
        files = f"{len(tree)} package files compared" if tree is not None else "tree not compared"
        print(f"version {version}; {files}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
