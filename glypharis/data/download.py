"""Reproducible dataset download: resume, verify, extract, record.

Layout under the data root (default ``data/``)::

    raw/<dataset>/<filename>          downloaded files, untouched
    raw/<dataset>/<extract_to>/...    extracted archive members
    manifests/<dataset>.json          what was downloaded: url, bytes, sha256, date, counts

Rules (IMPLEMENTATION_PLAN Phase 2):
- Files already present with the recorded size (and, with ``verify``, the recorded hash) are
  skipped, so a second run changes nothing, including the manifest.
- A published sha256 must match. Without one, the hash of the first download is recorded and every
  later download or ``verify`` must reproduce it; a mismatch is an error, never silently accepted.
- Interrupted downloads resume from a ``.part`` file with an HTTP Range request.
- Archives are extracted with path-traversal protection; an ``.extracted`` marker holding the
  archive hash makes extraction idempotent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from glypharis.data.sources import DatasetSpec, FileSpec

log = logging.getLogger(__name__)

MANIFEST_VERSION = 1
CHUNK = 1 << 20  # 1 MiB
USER_AGENT = "glypharis-dataset-downloader/1"
DISK_MARGIN_BYTES = 20 * 1024**3  # keep 20 GiB free after a download (RISK_REGISTER R-03)

Counter = Callable[[Path], dict[str, Any]]


class DownloadError(RuntimeError):
    """A download, verification or extraction step failed."""


# ------------------------------------------------------------------------------------- hashing


def sha256_file(path: Path, chunk: int = 8 * CHUNK) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------------- transfer


def _open(url: str, start: int, timeout: float) -> Any:
    headers = {"User-Agent": USER_AGENT}
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def download_file(
    url: str,
    dest: Path,
    expected_size: int | None = None,
    *,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 5.0,
) -> None:
    """Download ``url`` to ``dest`` via ``dest.part``, resuming and retrying on network errors."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    attempt = 0
    while True:
        start = part.stat().st_size if part.exists() else 0
        if expected_size is not None and start > expected_size:
            part.unlink()
            start = 0
        try:
            with _open(url, start, timeout) as resp:
                status = getattr(resp, "status", 200)
                if start > 0 and status != 206:
                    log.info("server ignored the range request; restarting %s", dest.name)
                    start = 0
                mode = "ab" if start > 0 else "wb"
                total = expected_size
                length = resp.headers.get("Content-Length")
                if total is None and length is not None:
                    total = start + int(length)
                _stream(resp, part, mode, start, total, dest.name)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            # 416 = range not satisfiable: the .part file may already be complete.
            if (
                isinstance(exc, urllib.error.HTTPError)
                and exc.code == 416
                and part.exists()
                and part.stat().st_size == expected_size
            ):
                break
            attempt += 1
            if attempt > retries:
                raise DownloadError(
                    f"download failed after {retries} retries: {url}: {exc}"
                ) from exc
            wait = backoff * attempt
            log.warning("download error (%s); retry %d/%d in %.0fs", exc, attempt, retries, wait)
            time.sleep(wait)
            continue
        break

    got = part.stat().st_size
    if expected_size is not None and got != expected_size:
        raise DownloadError(
            f"{dest.name}: got {got} bytes, expected {expected_size}; "
            f"the partial file is kept at {part.name} and the next run resumes it"
        )
    os.replace(part, dest)


def _stream(resp: Any, part: Path, mode: str, start: int, total: int | None, label: str) -> None:
    done = start
    last_log = time.monotonic()
    with part.open(mode) as fh:
        while block := resp.read(CHUNK):
            fh.write(block)
            done += len(block)
            now = time.monotonic()
            if now - last_log >= 30:
                last_log = now
                if total:
                    log.info("%s: %.1f%% (%.2f / %.2f GiB)", label, 100 * done / total,
                             done / 1024**3, total / 1024**3)  # fmt: skip
                else:
                    log.info("%s: %.2f GiB", label, done / 1024**3)


# ---------------------------------------------------------------------------------- extraction


def _safe_target(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if not target.is_relative_to(root.resolve()):
        raise DownloadError(f"archive member escapes the target folder: {member!r}")
    return target


def extract_archive(archive: Path, dest_dir: Path) -> int:
    """Extract a .tar/.tgz/.tar.gz or .zip archive. Returns the number of regular files."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    name = archive.name.lower()
    if name.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(archive) as tar:
            for member in tar:
                _safe_target(dest_dir, member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise DownloadError(f"refusing link or device member {member.name!r}")
                if member.isfile():
                    count += 1
                tar.extract(member, dest_dir, filter="data")
    elif name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                _safe_target(dest_dir, info.filename)
                if not info.is_dir():
                    count += 1
                zf.extract(info, dest_dir)
    else:
        raise DownloadError(f"unsupported archive type: {archive.name}")
    return count


# ------------------------------------------------------------------------------------ manifest


def manifest_path(data_root: Path, key: str) -> Path:
    return data_root / "manifests" / f"{key}.json"


def read_manifest(data_root: Path, key: str) -> dict[str, Any] | None:
    path = manifest_path(data_root, key)
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write_manifest_if_changed(data_root: Path, key: str, manifest: dict[str, Any]) -> bool:
    path = manifest_path(data_root, key)
    text = json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return True


# ------------------------------------------------------------------------------------ planning


@dataclass(frozen=True)
class FilePlan:
    spec: FileSpec
    path: Path
    status: str  # present | partial | missing
    bytes_to_fetch: int | None


def plan_dataset(spec: DatasetSpec, data_root: Path) -> list[FilePlan]:
    raw = data_root / "raw" / spec.key
    previous = (read_manifest(data_root, spec.key) or {}).get("files", {})
    plans = []
    for f in spec.files:
        path = raw / f.filename
        part = path.with_name(path.name + ".part")
        expected = f.size or previous.get(f.name, {}).get("bytes")
        if path.exists() and (expected is None or path.stat().st_size == expected):
            plans.append(FilePlan(f, path, "present", 0))
        elif part.exists():
            remaining = None if expected is None else expected - part.stat().st_size
            plans.append(FilePlan(f, path, "partial", remaining))
        else:
            plans.append(FilePlan(f, path, "missing", expected))
    return plans


def check_disk_space(data_root: Path, needed: int, margin: int = DISK_MARGIN_BYTES) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_root).free
    if free - needed < margin:
        raise DownloadError(
            f"not enough disk space: need {needed / 1024**3:.1f} GiB plus a "
            f"{margin / 1024**3:.0f} GiB margin, {free / 1024**3:.1f} GiB free"
        )


# --------------------------------------------------------------------------------- the driver


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def fetch_dataset(
    spec: DatasetSpec,
    data_root: Path,
    *,
    only: set[str] | None = None,
    verify: bool = False,
    counter: Counter | None = None,
    download: Callable[..., None] = download_file,
) -> dict[str, Any]:
    """Download, verify and extract one dataset, then write its manifest. Returns the manifest."""
    if only:
        unknown = only - {f.name for f in spec.files}
        if unknown:
            raise DownloadError(f"{spec.key}: unknown file names {sorted(unknown)}")
    raw = data_root / "raw" / spec.key
    previous = read_manifest(data_root, spec.key) or {}
    prev_files: dict[str, Any] = previous.get("files", {})
    files_out: dict[str, Any] = {}

    selected = [f for f in spec.files if not only or f.name in only]
    plans = {p.spec.name: p for p in plan_dataset(spec, data_root)}
    needed = sum(
        p.bytes_to_fetch or 0 for name, p in plans.items() if name in {f.name for f in selected}
    )
    if needed:
        check_disk_space(data_root, needed)

    for f in spec.files:
        record = dict(prev_files.get(f.name, {}))
        if f not in selected:
            if record:
                files_out[f.name] = record
            continue
        path = raw / f.filename
        rel = path.relative_to(data_root).as_posix()
        recorded_sha = record.get("sha256")

        fresh = plans[f.name].status != "present"
        if fresh:
            log.info("downloading %s/%s from %s", spec.key, f.filename, f.url)
            download(f.url, path, f.size)
            record["downloaded_utc"] = _now()

        if fresh or verify or recorded_sha is None:
            sha = sha256_file(path)
            expected = f.sha256 or recorded_sha
            if expected is not None and sha != expected:
                source = "published" if f.sha256 else "recorded on first download"
                raise DownloadError(
                    f"{spec.key}/{f.filename}: sha256 {sha} does not match the {source} hash "
                    f"{expected}. Delete the file to fetch it again if the source changed."
                )
            record["sha256"] = sha
            if verify:
                log.info("verified %s/%s", spec.key, f.filename)
        else:
            log.info("present %s/%s (size matches; use --verify to re-hash)", spec.key, f.filename)

        record.update(
            url=f.url,
            path=rel,
            bytes=path.stat().st_size,
            sha256_source="published" if f.sha256 else "computed",
        )
        record.setdefault("downloaded_utc", _now())

        if f.extract_to is not None:
            target = raw / f.extract_to
            marker = raw / f".extracted-{f.name}"
            if (
                not marker.exists()
                or marker.read_text(encoding="utf-8").strip() != record["sha256"]
            ):
                log.info(
                    "extracting %s into %s", f.filename, target.relative_to(data_root).as_posix()
                )
                record["extracted_files"] = extract_archive(path, target)
                marker.write_text(record["sha256"] + "\n", encoding="utf-8")
            record["extracted_to"] = target.relative_to(data_root).as_posix()
        files_out[f.name] = {k: record[k] for k in sorted(record)}

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset": spec.key,
        "title": spec.title,
        "version": spec.version,
        "homepage": spec.homepage,
        "licence": spec.licence,
        "citation": spec.citation,
        "role": spec.role,
        "files": {name: files_out[name] for name in sorted(files_out)},
        "counts": counter(raw) if counter else previous.get("counts", {}),
    }
    if _write_manifest_if_changed(data_root, spec.key, manifest):
        log.info("manifest written: %s", manifest_path(data_root, spec.key).as_posix())
    else:
        log.info("manifest unchanged: %s", spec.key)
    return manifest
