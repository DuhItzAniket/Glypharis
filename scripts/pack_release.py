"""Pack the v1 bundle into its release archive, reproducibly (Phase 23, item 7).

    python scripts/pack_release.py --bundle runs/release/glypharis-v1 --out <folder> [--check-only]

writes ``<out>/glypharis-v1.tar.gz`` and ``<out>/glypharis-v1.tar.gz.sha256`` (``sha256sum``
format, the archive's file name only) and prints the archive's size and SHA-256, which go into the
``glypharis-v1`` entry of ``glypharis.hub.REGISTRY``. Nothing is uploaded or published.

The archive holds one top folder ``glypharis-v1/`` with

- the five bundle files byte for byte: ``manifest.json``, ``detector.pt``, ``detector.onnx``,
  ``recognizer.pt``, ``recognizer.onnx``. The bundle's ``manifest.json`` must have the pinned
  SHA-256 (`V1_MANIFEST_SHA256`; ``--manifest-sha256`` replaces the pin for tests), every file it
  lists must match its hash, and the bundle folder must hold nothing else;
- the weight notices ``LICENSE-WEIGHTS`` and ``NOTICE.md`` from ``release/weights/``, which sits
  outside the ``glypharis`` package so the notices never enter the code wheel;
- ``model_info.json``: what the weights were trained on, the frozen config, the experiment IDs,
  the code tag and commit, the terms and a link to the model card at the tag. It holds no metric
  values (the model card does) and no copy of the model card, so the archive's hash does not
  depend on later edits of the documents. The bytes are deterministic (sorted keys, LF, no time).

Reproducible: the members are regular files and the one folder, in a fixed sorted order, with
mtime `MTIME`, uid and gid 0, empty user and group names, mode 0o644 (files) and 0o755 (the
folder), in PAX format; the gzip header has mtime 0 and no file name. Two packs of the same inputs
with the same Python and zlib give the same bytes.

The archive is checked after it is built: it is extracted with
`glypharis.data.download.extract_archive` into a temporary folder, the extracted bundle must pass
`glypharis.export.bundle.read_manifest` and the manifest pin, and every member must equal its
source byte for byte.

The out folder must lie outside every git worktree (no parent of it may hold a ``.git`` entry), so
an archive with the weights can never be committed by accident. An archive already there with the
same bytes is left as it is; one with other bytes is refused unless ``--overwrite`` is given.

``--check-only`` writes nothing: it builds and checks the archive in memory, compares it with the
archive and ``.sha256`` file in ``--out`` if they are there, and with the registry's pins once
they are set. Exits 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
NOTICES_DIR: Final = REPO_ROOT / "release" / "weights"

NAME: Final = "glypharis-v1"
ARCHIVE: Final = f"{NAME}.tar.gz"
V1_MANIFEST_SHA256: Final = "875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037"
BUNDLE_FILES: Final = (
    "manifest.json",
    "detector.pt",
    "detector.onnx",
    "recognizer.pt",
    "recognizer.onnx",
)
NOTICE_FILES: Final = ("LICENSE-WEIGHTS", "NOTICE.md")
INFO_FILE: Final = "model_info.json"
INFO_FORMAT: Final = "glypharis-model-info-1"

MTIME: Final = 0  # every member's modification time (1970-01-01T00:00:00Z)
FILE_MODE: Final = 0o644
DIR_MODE: Final = 0o755
TAR_FORMAT: Final = tarfile.PAX_FORMAT
GZIP_LEVEL: Final = 9

REPOSITORY: Final = "https://github.com/DuhItzAniket/Glypharis"
RELEASE_TAG: Final = "v1.0"
CODE_TAG: Final = "v1-code"
MODEL_CARD_URL: Final = f"{REPOSITORY}/blob/{RELEASE_TAG}/MODEL_CARD.md"
TERMS: Final = "CC BY-SA 4.0 (see LICENSE-WEIGHTS and NOTICE.md)"
CONFIG: Final = "configs/final/glypharis_v1.yaml"
MODEL_VERSION: Final = "glypharis_v1"
EXPERIMENTS: Final = {"detector": "GW-DET-001", "recognizer": "GW-REC-034"}
BACKBONE_SHA256: Final = "8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04"

# Copied from MODEL_CARD.md ("Training datasets", "Licence information") and DATASETS.md ("What
# trained the delivered models"); nothing here is new.
TRAINING_DATA: Final[dict[str, Any]] = {
    "detector": {
        "experiment": "GW-DET-001",
        "data": "HierText train pages",
        "initialisation": "backbone from MobileNetV3-Large ImageNet-1K weights",
    },
    "recognizer": {
        "experiment": "GW-REC-034",
        "data": "HierText train lines",
        "initialisation": "random",
    },
    "sources": [
        {
            "name": "HierText dataset",
            "split": "train",
            "terms": "CC BY-SA 4.0",
            "role": "training and evaluation",
        },
        {
            "name": "Images in HierText, from Open Images",
            "terms": "listed as CC BY 2.0",
            "role": "training and evaluation",
        },
        {
            "name": "MobileNetV3-Large ImageNet-1K weights (torchvision)",
            "sha256": BACKBONE_SHA256,
            "terms": "code BSD-3-Clause; the weights carry ImageNet-1K terms",
            "role": "detector initialisation only",
        },
    ],
    "not_used": [
        "No GlyphSynth line is in v1.",
        "The evaluation-only sets (FUNSD, OmniDocBench, IIIT5K, GDC, GDC-scan) never entered "
        "training, augmentation or tuning.",
    ],
}


class PackError(Exception):
    """A refusal: the inputs or the out folder are not what the release archive needs."""


@dataclass(frozen=True)
class Packed:
    data: bytes  # the archive
    sha256: str
    members: tuple[str, ...]
    sources: dict[str, bytes]  # member path inside the top folder -> its bytes

    @property
    def size(self) -> int:
        return len(self.data)

    def sha256_line(self, archive: str = ARCHIVE) -> str:
        return f"{self.sha256}  {archive}\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------------------------------------- inputs


def read_bundle(bundle: Path, manifest_sha256: str) -> tuple[dict[str, bytes], dict[str, Any]]:
    """The five bundle files' bytes, after checking the manifest pin and every file's hash."""
    if not bundle.is_dir():
        raise PackError(f"{bundle}: not a folder")
    found = sorted(p.name for p in bundle.iterdir())
    if found != sorted(BUNDLE_FILES):
        raise PackError(
            f"{bundle}: expected exactly {sorted(BUNDLE_FILES)}, found {found}; the bundle "
            f"folder must hold the five bundle files and nothing else"
        )
    data: dict[str, bytes] = {}
    for name in BUNDLE_FILES:
        path = bundle / name
        if path.is_symlink() or not path.is_file():
            raise PackError(f"{path}: not a regular file")
        data[name] = path.read_bytes()
    got = _sha(data["manifest.json"])
    if got != manifest_sha256:
        raise PackError(
            f"{bundle / 'manifest.json'}: SHA-256 {got} is not the pinned {manifest_sha256}"
        )
    try:
        manifest = json.loads(data["manifest.json"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackError(f"{bundle / 'manifest.json'}: not valid JSON ({exc})") from exc
    listed = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(listed, dict) or sorted(listed) != sorted(BUNDLE_FILES[1:]):
        raise PackError(
            f"{bundle / 'manifest.json'}: its files are "
            f"{sorted(listed) if isinstance(listed, dict) else listed}, expected "
            f"{sorted(BUNDLE_FILES[1:])}"
        )
    for name, want in sorted(listed.items()):
        have = _sha(data[name])
        if have != want:
            raise PackError(
                f"{bundle / name}: SHA-256 {have} does not match manifest.json ({want})"
            )
    return data, manifest


def check_v1_manifest(manifest: dict[str, Any]) -> None:
    """The constants written into model_info.json must agree with the v1 manifest they describe."""
    sources = manifest.get("sources") or {}
    det_model = (manifest.get("detector") or {}).get("spec", {}).get("model", {})
    checks = {
        "version": (manifest.get("version"), MODEL_VERSION),
        "sources.record": (sources.get("record"), CONFIG),
        "sources.detector.experiment": (
            (sources.get("detector") or {}).get("experiment"),
            EXPERIMENTS["detector"],
        ),
        "sources.recognizer.experiment": (
            (sources.get("recognizer") or {}).get("experiment"),
            EXPERIMENTS["recognizer"],
        ),
        "detector.spec.model.init": (det_model.get("init"), "imagenet"),
        "detector.spec.model.weights_sha256": (det_model.get("weights_sha256"), BACKBONE_SHA256),
    }
    wrong = [f"{k}: {got!r}, expected {want!r}" for k, (got, want) in checks.items() if got != want]
    if wrong:
        raise PackError(
            "manifest.json does not match model_info.json's v1 record: " + "; ".join(wrong)
        )


def read_notices(notices_dir: Path) -> dict[str, bytes]:
    data: dict[str, bytes] = {}
    for name in NOTICE_FILES:
        path = notices_dir / name
        if path.is_symlink() or not path.is_file():
            raise PackError(f"{path}: missing or not a regular file")
        raw = path.read_bytes()
        if b"\r" in raw:
            raise PackError(f"{path}: has CR line endings; the notices must be LF")
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PackError(f"{path}: not UTF-8 ({exc})") from exc
        data[name] = raw
    return data


def code_commit(repo: Path = REPO_ROOT, tag: str = CODE_TAG) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"{tag}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PackError(f"cannot resolve the tag {tag!r} in {repo}: {exc}") from exc
    if len(out) != 40:
        raise PackError(f"git rev-parse {tag}^{{commit}} gave {out!r}")
    return out


def model_info(
    bundle_files: dict[str, bytes], manifest: dict[str, Any], commit: str, name: str = NAME
) -> bytes:
    info = {
        "format": INFO_FORMAT,
        "name": name,
        "model_version": manifest.get("version"),
        "manifest_sha256": _sha(bundle_files["manifest.json"]),
        "files": {f: _sha(bundle_files[f]) for f in BUNDLE_FILES},
        "config": CONFIG,
        "experiments": dict(EXPERIMENTS),
        "training_data": TRAINING_DATA,
        "code": {"repository": REPOSITORY, "tag": CODE_TAG, "commit": commit},
        "release_tag": RELEASE_TAG,
        "model_card": MODEL_CARD_URL,
        "terms": TERMS,
        "notices": list(NOTICE_FILES),
        "metrics": "not included; the model card gives them with their experiment IDs",
    }
    text = json.dumps(info, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    return text.encode("utf-8")


# --------------------------------------------------------------------------------------- pack


def _tarinfo(name: str, *, directory: bool = False, size: int = 0) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    ti.mode = DIR_MODE if directory else FILE_MODE
    ti.size = 0 if directory else size
    ti.mtime = MTIME
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    return ti


def build_archive(files: dict[str, bytes], top: str = NAME) -> Packed:
    """A reproducible .tar.gz of ``files`` (name -> bytes) under one folder ``top``."""
    order = sorted(files)
    raw = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="", compresslevel=GZIP_LEVEL) as gz,
        tarfile.open(fileobj=gz, mode="w", format=TAR_FORMAT) as tar,
    ):
        tar.addfile(_tarinfo(top, directory=True))
        for name in order:
            tar.addfile(_tarinfo(f"{top}/{name}", size=len(files[name])), io.BytesIO(files[name]))
    data = raw.getvalue()
    members = (f"{top}/", *(f"{top}/{n}" for n in order))
    return Packed(data=data, sha256=_sha(data), members=members, sources=dict(files))


def pack(
    bundle: Path,
    *,
    manifest_sha256: str = V1_MANIFEST_SHA256,
    notices_dir: Path = NOTICES_DIR,
    commit: str | None = None,
) -> Packed:
    """Read and check the inputs and build the archive in memory (no file is written)."""
    bundle_files, manifest = read_bundle(bundle, manifest_sha256)
    if manifest_sha256 == V1_MANIFEST_SHA256:
        check_v1_manifest(manifest)
    files = {**bundle_files, **read_notices(notices_dir)}
    files[INFO_FILE] = model_info(bundle_files, manifest, commit or code_commit())
    packed = build_archive(files)
    verify_archive(packed, manifest_sha256)
    return packed


# ------------------------------------------------------------------------------------- verify


def _members(data: bytes) -> list[tarfile.TarInfo]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        return tar.getmembers()


def verify_archive(packed: Packed, manifest_sha256: str, top: str = NAME) -> None:
    """Extract ``packed`` with ``extract_archive`` and check what comes out (module docstring)."""
    from glypharis.data.download import extract_archive
    from glypharis.errors import ModelLoadError
    from glypharis.export.bundle import read_manifest

    if packed.data[3] != 0 or packed.data[4:8] != b"\0\0\0\0":  # FLG and MTIME of the gzip header
        raise PackError("the gzip header carries a file name or a time")
    names = []
    for m in _members(packed.data):
        if not (m.isfile() or (m.isdir() and m.name == top)):
            raise PackError(f"unexpected member {m.name!r} of type {m.type!r}")
        fixed = (m.mtime, m.uid, m.gid, m.uname, m.gname, m.mode)
        mode = DIR_MODE if m.isdir() else FILE_MODE
        if fixed != (MTIME, 0, 0, "", "", mode) or m.pax_headers:
            raise PackError(f"{m.name}: metadata {fixed} is not the fixed one")
        names.append(m.name + ("/" if m.isdir() else ""))
    if tuple(names) != packed.members:
        raise PackError(f"members {names} are not the expected {list(packed.members)}")

    tmp = Path(tempfile.mkdtemp(prefix="pack-release-"))
    try:
        archive = tmp / ARCHIVE
        archive.write_bytes(packed.data)
        out = tmp / "x"
        count = extract_archive(archive, out)
        if count != len(packed.sources) or sorted(p.name for p in out.iterdir()) != [top]:
            raise PackError(
                f"extracted {count} files under {sorted(p.name for p in out.iterdir())}"
            )
        bundle = out / top
        try:
            read_manifest(bundle)
        except ModelLoadError as exc:
            raise PackError(f"the extracted bundle fails read_manifest: {exc}") from exc
        if _sha((bundle / "manifest.json").read_bytes()) != manifest_sha256:
            raise PackError("the extracted manifest.json does not match the pin")
        for name, want in packed.sources.items():
            if (bundle / name).read_bytes() != want:
                raise PackError(f"the extracted {name} differs from its source")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------------------- out


def git_worktree_of(path: Path) -> Path | None:
    """The first of ``path`` and its parents that holds a ``.git`` entry, if any."""
    p = path.resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists() or (d / ".git").is_symlink():
            return d
    return None


def write(packed: Packed, out: Path, *, overwrite: bool = False) -> tuple[Path, str]:
    """Write the archive and its .sha256 into ``out``; returns the archive path and what happened
    (``written`` or ``unchanged``)."""
    worktree = git_worktree_of(out)
    if worktree is not None:
        raise PackError(
            f"{out} is inside the git worktree {worktree}; write the archive outside every "
            f"worktree, so it can never be committed"
        )
    out.mkdir(parents=True, exist_ok=True)
    archive, sums = out / ARCHIVE, out / f"{ARCHIVE}.sha256"
    status = "written"
    if archive.exists():
        if archive.is_file() and archive.read_bytes() == packed.data:
            status = "unchanged"
        elif not overwrite:
            raise PackError(f"{archive} exists with other bytes; pass --overwrite to replace it")
    if status == "written":
        part = archive.with_name(archive.name + ".part")
        part.write_bytes(packed.data)
        part.replace(archive)
    sums.write_bytes(packed.sha256_line().encode("ascii"))
    return archive, status


def check_existing(
    packed: Packed, out: Path, manifest_sha256: str = V1_MANIFEST_SHA256
) -> list[str]:
    """Differences between ``packed`` and what ``out`` holds, and the registry's pins once they are
    set (only for the registered manifest)."""
    from glypharis.hub import REGISTRY

    problems: list[str] = []
    archive, sums = out / ARCHIVE, out / f"{ARCHIVE}.sha256"
    if archive.is_file() and archive.read_bytes() != packed.data:
        problems.append(f"{archive} differs from a fresh pack")
    if sums.is_file() and sums.read_text(encoding="ascii") != packed.sha256_line():
        problems.append(f"{sums} does not hold {packed.sha256_line().strip()!r}")
    entry = REGISTRY.get(NAME)
    if (
        entry is not None
        and entry.manifest_sha256 == manifest_sha256
        and entry.bytes is not None
        and entry.archive_sha256 is not None
        and (entry.bytes, entry.archive_sha256) != (packed.size, packed.sha256)
    ):
        problems.append(
            f"the registry pins {entry.bytes} bytes, SHA-256 {entry.archive_sha256}; a fresh "
            f"pack is {packed.size} bytes, SHA-256 {packed.sha256}"
        )
    return problems


# ---------------------------------------------------------------------------------------- cli


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bundle", type=Path, required=True, help="the bundle folder (read only)")
    ap.add_argument("--out", type=Path, required=True, help="a folder outside every git worktree")
    ap.add_argument("--check-only", action="store_true", help="build and compare; write nothing")
    ap.add_argument("--overwrite", action="store_true", help="replace an archive with other bytes")
    ap.add_argument(
        "--manifest-sha256", default=V1_MANIFEST_SHA256, help="the manifest pin (tests only)"
    )
    ap.add_argument("--notices", type=Path, default=NOTICES_DIR, help=argparse.SUPPRESS)
    ap.add_argument(
        "--code-commit", default=None, help=f"the {CODE_TAG} commit (default: git rev-parse)"
    )
    args = ap.parse_args(argv)
    try:
        packed = pack(
            args.bundle,
            manifest_sha256=args.manifest_sha256,
            notices_dir=args.notices,
            commit=args.code_commit,
        )
        if args.check_only:
            problems = check_existing(packed, args.out, args.manifest_sha256)
            for p in problems:
                print(f"FAIL {p}", file=sys.stderr)
            print(f"{ARCHIVE}: {packed.size} bytes, SHA-256 {packed.sha256} (nothing written)")
            return 1 if problems else 0
        archive, status = write(packed, args.out, overwrite=args.overwrite)
    except PackError as exc:
        print(f"pack_release: {exc}", file=sys.stderr)
        return 1
    print(f"{archive}: {packed.size} bytes, SHA-256 {packed.sha256} ({status})")
    print(f"{archive}.sha256: {packed.sha256_line().strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
