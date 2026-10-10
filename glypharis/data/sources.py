"""The dataset source registry (`data/sources.yaml`), loaded into typed records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_DATASET_KEYS = {
    "title", "version", "homepage", "licence", "citation", "role", "phase", "files",
}  # fmt: skip
_FILE_KEYS = {"name", "url", "filename", "size", "sha256", "extract_to"}


class SourceError(ValueError):
    """Invalid entry in the source registry."""


@dataclass(frozen=True)
class FileSpec:
    name: str
    url: str
    filename: str
    size: int | None = None
    sha256: str | None = None
    extract_to: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    version: str
    homepage: str
    licence: str
    citation: str
    role: str
    phase: int
    files: tuple[FileSpec, ...]

    def file(self, name: str) -> FileSpec:
        for f in self.files:
            if f.name == name:
                return f
        raise KeyError(f"{self.key} has no file named {name!r}")


def _check_relative(value: str, where: str) -> None:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.startswith(("/", "\\")):
        raise SourceError(f"{where}: {value!r} must be a relative path inside the dataset folder")


def _parse_file(raw: Any, where: str) -> FileSpec:
    if not isinstance(raw, dict):
        raise SourceError(f"{where}: expected a mapping")
    unknown = set(raw) - _FILE_KEYS
    if unknown:
        raise SourceError(f"{where}: unknown keys {sorted(unknown)}")
    for key in ("name", "url", "filename"):
        if not isinstance(raw.get(key), str) or not raw[key]:
            raise SourceError(f"{where}: '{key}' is required")
    if not raw["url"].startswith("https://"):
        raise SourceError(f"{where}: only https:// URLs are allowed")
    _check_relative(raw["filename"], f"{where}.filename")
    size = raw.get("size")
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
        raise SourceError(f"{where}: size must be a positive int")
    sha = raw.get("sha256")
    if sha is not None and (
        not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha)
    ):
        raise SourceError(f"{where}: sha256 must be 64 lowercase hex characters")
    extract_to = raw.get("extract_to")
    if extract_to is not None:
        if not isinstance(extract_to, str):
            raise SourceError(f"{where}: extract_to must be a path")
        _check_relative(extract_to, f"{where}.extract_to")
    return FileSpec(raw["name"], raw["url"], raw["filename"], size, sha, extract_to)


def parse_sources(data: Any) -> dict[str, DatasetSpec]:
    if not isinstance(data, dict):
        raise SourceError("source registry must be a mapping of dataset keys")
    out: dict[str, DatasetSpec] = {}
    for key, raw in data.items():
        if not isinstance(raw, dict):
            raise SourceError(f"{key}: expected a mapping")
        unknown = set(raw) - _DATASET_KEYS
        missing = _DATASET_KEYS - set(raw)
        if unknown or missing:
            raise SourceError(f"{key}: unknown keys {sorted(unknown)}, missing {sorted(missing)}")
        files = raw["files"]
        if not isinstance(files, list) or not files:
            raise SourceError(f"{key}: 'files' must be a non-empty list")
        specs = tuple(_parse_file(f, f"{key}.files[{i}]") for i, f in enumerate(files))
        names = [f.name for f in specs]
        if len(set(names)) != len(names):
            raise SourceError(f"{key}: duplicate file names")
        if len({f.filename for f in specs}) != len(specs):
            raise SourceError(f"{key}: duplicate target filenames")
        if isinstance(raw["phase"], bool) or not isinstance(raw["phase"], int):
            raise SourceError(f"{key}: phase must be an int")
        out[key] = DatasetSpec(
            key=key,
            title=str(raw["title"]),
            version=str(raw["version"]),
            homepage=str(raw["homepage"]),
            licence=str(raw["licence"]),
            citation=str(raw["citation"]),
            role=str(raw["role"]),
            phase=raw["phase"],
            files=specs,
        )
    return out


def load_sources(path: str | Path) -> dict[str, DatasetSpec]:
    with Path(path).open(encoding="utf-8") as fh:
        return parse_sources(yaml.safe_load(fh))
