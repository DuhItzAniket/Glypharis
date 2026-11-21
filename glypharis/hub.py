"""Model download: fetch a registered GLYPHARIS bundle from its release archive (Phase 23).

    from glypharis import download_model
    path = download_model("glypharis-v1")        # <models dir>/glypharis-v1, verified

`GlypharisOCR.from_pretrained` never reaches the network; this module is the only part of the
package that downloads a model, and only for names in `REGISTRY`. Each entry pins the archive's
size and SHA-256 and the SHA-256 of the bundle's ``manifest.json`` (which itself holds the SHA-256
of every model file), so nothing unpinned is ever installed, whatever address it came from.

`download_model` rules, in order:

1. An unknown name is refused.
2. A bundle that `glypharis.export.bundle.find_bundle` already finds under that name is returned
   when its manifest matches the pin, with no network call. One with another manifest is refused
   and never overwritten; ``force=True`` replaces only a bundle at ``<models_dir>/<name>``, and
   the replaced folder is moved under ``<models_dir>/.downloads/`` rather than deleted. That
   replacement takes two renames and is not atomic: if the second fails, the old folder is renamed
   back when it can be, and the error says where it is.
3. Offline (``offline=True``, or ``GLYPHARIS_OFFLINE`` set to 1, true, yes or on) refuses before
   any network call, and so does an entry not yet published unless ``url=`` is given, or one whose
   archive size and hash are not pinned yet.
4. Otherwise, under a lock file in ``<models_dir>/.downloads/``: the archive is downloaded there
   (`glypharis.data.download.download_file`, which resumes from its ``.part`` file), its size and
   SHA-256 are checked (a mismatching file is deleted), it is extracted into a staging folder on
   the same disk, the bundle's files are checked against its manifest and the manifest against the
   pin, a download record (`RECORD_FILE`) is written beside the bundle, and the folder is renamed
   into place (one rename when nothing is there; see rule 2 for ``force``). A process that
   waited for the lock finds the winner's bundle, checks it, and returns it. The archive is
   removed once the bundle is in place.

``url=`` replaces only the address (for tests and the release rehearsal), and only http and https
addresses are accepted; the size and both hashes are still enforced. Every failure is a
`glypharis.errors.ModelDownloadError`.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.parse
import uuid
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from glypharis.errors import ModelDownloadError, ModelLoadError

log = logging.getLogger(__name__)

OFFLINE_ENV: Final = "GLYPHARIS_OFFLINE"
DOWNLOADS_DIR: Final = ".downloads"
RECORD_FILE: Final = (
    "download_record.json"  # not listed in manifest.json, so read_manifest skips it
)
RECORD_FORMAT: Final = "glypharis-download-1"
_TRUE: Final = frozenset({"1", "true", "yes", "on"})

# Transfer and lock settings; module-level so tests can make them short.
DOWNLOAD_TIMEOUT = 60.0  # seconds without data before a retry
DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF = 5.0  # seconds, times the attempt number
LOCK_WAIT_SECONDS = 1800.0  # how long to wait for another process's download
LOCK_STALE_SECONDS = 120.0  # a lock not refreshed for this long belongs to a dead process
LOCK_HEARTBEAT_SECONDS = 15.0  # how often the holder refreshes its lock
LOCK_POLL_SECONDS = 0.5


@dataclass(frozen=True)
class ModelEntry:
    """One downloadable bundle. ``bytes`` and ``archive_sha256`` stay ``None`` until the release
    archive is packed; ``published`` stays off until the archive is on the release page."""

    name: str
    url: str
    archive: str  # the archive's file name
    bytes: int | None
    archive_sha256: str | None
    manifest_sha256: str  # SHA-256 of the bundle's manifest.json
    top_folder: str  # the archive's one top-level folder
    published: bool = False


REGISTRY: dict[str, ModelEntry] = {
    "glypharis-v1": ModelEntry(
        name="glypharis-v1",
        url="https://github.com/DuhItzAniket/Glypharis/releases/download/v1.0/glypharis-v1.tar.gz",
        archive="glypharis-v1.tar.gz",
        bytes=None,
        archive_sha256=None,
        manifest_sha256="875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037",
        top_folder="glypharis-v1",
        published=False,
    ),
}


@contextlib.contextmanager
def override_registry(*entries: ModelEntry) -> Iterator[dict[str, ModelEntry]]:
    """Replace the registry with ``entries`` for the duration of the block (tests, rehearsals)."""
    saved = dict(REGISTRY)
    REGISTRY.clear()
    REGISTRY.update({e.name: e for e in entries})
    try:
        yield REGISTRY
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved)


def is_offline(offline: bool | None = None) -> bool:
    """``offline`` if given, else whether ``GLYPHARIS_OFFLINE`` is set to 1, true, yes or on."""
    if offline is not None:
        return offline
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in _TRUE


def needs_download(name_or_path: str | Path) -> bool:
    """True for a registered name that no models folder holds: what ``glypharis-ocr`` and
    ``glypharis-serve`` fetch before loading. A path or an unregistered name is never fetched."""
    from glypharis.export.bundle import find_bundle

    if not isinstance(name_or_path, str) or name_or_path not in REGISTRY:
        return False
    try:
        find_bundle(name_or_path)
    except ModelLoadError:
        return True
    return False


def can_fetch(name: str, offline: bool | None = None) -> bool:
    """Whether `download_model` would go to the network for a missing ``name`` (no ``url=``):
    registered, published, its archive pinned, and not offline."""
    entry = REGISTRY.get(name)
    return (
        entry is not None
        and entry.published
        and entry.bytes is not None
        and entry.archive_sha256 is not None
        and not is_offline(offline)
    )


def download_model(
    name: str = "glypharis-v1",
    *,
    models_dir: str | Path | None = None,
    offline: bool | None = None,
    force: bool = False,
    url: str | None = None,
) -> Path:
    """The bundle ``name``, downloaded and verified if it is not installed yet (module docstring).
    ``models_dir`` must be one of the folders bundles are looked up in
    (`glypharis.export.bundle.model_dirs`); by default it is the first of them, the first
    ``GLYPHARIS_MODELS`` entry if that is set, else ``~/.glypharis/models``."""
    from glypharis.data.download import DownloadError
    from glypharis.export.bundle import model_dirs

    entry = REGISTRY.get(name)
    if entry is None:
        known = ", ".join(sorted(REGISTRY)) or "none"
        raise ModelDownloadError(f"unknown model {name!r}; registered models: {known}")
    dirs = model_dirs()
    root = Path(models_dir) if models_dir is not None else dirs[0]
    if not any(_same_path(root, d) for d in dirs):
        raise ModelDownloadError(
            f"{root} is not a folder bundles are looked up in, so the model would not be found "
            f"there; use one of: {', '.join(str(d) for d in dirs)} (or add it to GLYPHARIS_MODELS)"
        )
    target = root / name

    existing = _installed(name)
    if existing is not None:
        replaceable = force and _same_path(existing, target)
        if not replaceable:
            sha = _manifest_sha256(existing)
            if sha == entry.manifest_sha256:
                return existing
            raise ModelDownloadError(
                f"{existing} holds a bundle named {name!r} that is not the registered one "
                f"(manifest SHA-256 {sha[:12]}, registered "
                f"{entry.manifest_sha256[:12]}); it is left as it is. Move it away to download "
                f"the registered bundle"
                + (
                    ""
                    if _same_path(existing, target)
                    else ", or pass a models_dir it does not shadow"
                )
            )

    if is_offline(offline):
        raise ModelDownloadError(
            f"{name!r} is not installed and offline mode is on (offline=True or {OFFLINE_ENV}); "
            f"nothing was downloaded. Put the bundle in one of: {', '.join(str(d) for d in dirs)}"
        )
    if not entry.published and url is None:
        raise ModelDownloadError(
            f"{name!r} is not yet published, so it cannot be downloaded; put the bundle in one "
            f"of: {', '.join(str(d) for d in dirs)}, or pass a bundle directory"
        )
    if entry.bytes is None or entry.archive_sha256 is None:
        raise ModelDownloadError(
            f"{name!r} has no pinned archive size and SHA-256 yet, so nothing can be verified; "
            f"nothing was downloaded"
        )

    source = url if url is not None else entry.url
    if urllib.parse.urlsplit(source).scheme.lower() not in ("http", "https"):
        raise ModelDownloadError(
            f"{source!r}: only http and https addresses are supported; nothing was downloaded"
        )
    downloads = root / DOWNLOADS_DIR
    try:
        downloads.mkdir(parents=True, exist_ok=True)
        with _lock(downloads / f"{name}.lock"):
            return _install(entry, source, target, downloads, force)
    except ModelDownloadError:
        raise
    except ModelLoadError as exc:
        raise ModelDownloadError(f"the archive from {source} is not a valid bundle: {exc}") from exc
    except (DownloadError, *_TRANSFER_ERRORS) as exc:
        raise ModelDownloadError(f"cannot download {name!r} from {source}: {exc}") from exc


# ------------------------------------------------------------------------------------- install


def _install(entry: ModelEntry, source: str, target: Path, downloads: Path, force: bool) -> Path:
    from glypharis.data.download import download_file, extract_archive, sha256_file
    from glypharis.export.bundle import read_manifest

    if entry.bytes is None or entry.archive_sha256 is None:  # checked by download_model
        raise ModelDownloadError(f"{entry.name!r}: no pinned archive size and SHA-256")
    # Another process may have installed it while this one waited for the lock.
    done = _check_target(entry, target, force)
    if done is not None:
        return done

    archive = downloads / f"{entry.name}-{entry.archive_sha256[:12]}-{entry.archive}"
    part = archive.with_name(archive.name + ".part")
    if force:
        archive.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
    fetched = False
    if archive.is_file() and not (
        archive.stat().st_size == entry.bytes and sha256_file(archive) == entry.archive_sha256
    ):
        archive.unlink()  # a damaged copy left by an earlier attempt
    if not archive.is_file():
        log.info("downloading %s from %s into %s", entry.name, source, target.parent)
        for attempt in (1, 2):
            resumed = part.is_file()
            download_file(
                source,
                archive,
                entry.bytes,
                timeout=DOWNLOAD_TIMEOUT,
                retries=DOWNLOAD_RETRIES,
                backoff=DOWNLOAD_BACKOFF,
            )
            fetched = True
            got = sha256_file(archive)
            if got == entry.archive_sha256:
                break
            archive.unlink(missing_ok=True)
            if resumed and attempt == 1:
                # The kept .part may hold bytes from another address or a changed file: once more
                # from the start before giving up.
                log.warning("%s: resumed archive does not match; fetching it again", entry.name)
                continue
            raise ModelDownloadError(
                f"{entry.archive} from {source}: SHA-256 {got[:12]} does not match the registered "
                f"{entry.archive_sha256[:12]}; the file was deleted and nothing was installed"
            )

    staging = Path(tempfile.mkdtemp(prefix=f".staging-{entry.name}-", dir=downloads))
    try:
        extract_archive(archive, staging)
        tops = sorted(p.name for p in staging.iterdir())
        if tops != [entry.top_folder]:
            raise ModelDownloadError(
                f"{entry.archive}: expected one top-level folder {entry.top_folder!r}, found {tops}"
            )
        bundle = staging / entry.top_folder
        read_manifest(bundle)  # every listed file is there and matches its hash
        manifest_sha = sha256_file(bundle / "manifest.json")
        if manifest_sha != entry.manifest_sha256:
            raise ModelDownloadError(
                f"{entry.archive}: the bundle's manifest SHA-256 {manifest_sha[:12]} does not "
                f"match the registered {entry.manifest_sha256[:12]}; nothing was installed"
            )
        record = {
            "format": RECORD_FORMAT,
            "name": entry.name,
            "archive": entry.archive,
            "archive_bytes": entry.bytes,
            "archive_sha256": entry.archive_sha256,
            "manifest_sha256": manifest_sha,
            "url": source if fetched else None,  # None: an archive an earlier attempt fetched
            "registry_url": entry.url,
            "network_fetch": fetched,
            "installed_utc": _now(),
        }
        (bundle / RECORD_FILE).write_text(
            json.dumps(record, indent=1) + "\n", encoding="utf-8", newline="\n"
        )
        placed = _place(entry, bundle, target, downloads, force)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    archive.unlink(missing_ok=True)
    return placed


def _check_target(entry: ModelEntry, target: Path, force: bool) -> Path | None:
    """``target`` if it already holds the registered bundle (and ``force`` is off); refuses a
    folder there that is not the registered bundle unless ``force`` may replace it."""
    if not target.exists():
        return None
    if not (target / "manifest.json").is_file():
        raise ModelDownloadError(
            f"{target} exists but is not a bundle (no manifest.json); remove it and try again"
        )
    if force:
        return None
    sha = _manifest_sha256(target)
    if sha == entry.manifest_sha256:
        return target
    raise ModelDownloadError(
        f"{target} holds a bundle that is not the registered {entry.name!r} (manifest SHA-256 "
        f"{sha[:12]}); it is left as it is"
    )


def _place(entry: ModelEntry, bundle: Path, target: Path, downloads: Path, force: bool) -> Path:
    """Rename the verified ``bundle`` to ``target``: one rename when nothing is there. Replacing a
    bundle with ``force`` takes two renames (the old one aside, then the new one in) and is not
    atomic; if the second fails, the old bundle is renamed back when it can be, and the error says
    where it is."""
    done = _check_target(entry, target, force)
    if done is not None:  # it appeared meanwhile and is the registered bundle: keep it
        return done
    aside: Path | None = None
    if force and target.exists():
        aside = downloads / f"replaced-{entry.name}-{_stamp()}-{uuid.uuid4().hex[:6]}"
        os.rename(target, aside)
        log.warning("moved the previous %s to %s", target, aside)
    try:
        _rename(bundle, target)
    except OSError as exc:
        if (target / "manifest.json").is_file():  # lost a race with a process that took no lock
            done = _check_target(entry, target, force=False)
            if done is not None:
                return done
        if aside is None:
            raise
        where = f"it is still at {aside}"
        if not target.exists():
            with contextlib.suppress(OSError):
                os.rename(aside, target)
                where = f"it was restored to {target}"
        raise ModelDownloadError(
            f"cannot move the new {entry.name!r} bundle into {target} ({exc}); the previous "
            f"bundle was moved aside first and {where}"
        ) from exc
    log.info("installed %s at %s", entry.name, target)
    return target


def _rename(src: Path, dst: Path, attempts: int = 5) -> None:
    # On Windows a scanner that has just opened an extracted file can make a rename fail briefly.
    for i in range(attempts):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if i == attempts - 1 or dst.exists():
                raise
            time.sleep(0.2 * (i + 1))


# ---------------------------------------------------------------------------------------- lock


@contextlib.contextmanager
def _lock(path: Path) -> Iterator[None]:
    """An exclusive lock file (``O_CREAT | O_EXCL``), refreshed while held. A lock left by a
    process that died is taken over once it is older than `LOCK_STALE_SECONDS`."""
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except (FileExistsError, PermissionError):  # Windows: PermissionError while it is deleted
            if _break_stale(path):
                continue
            if time.monotonic() >= deadline:
                raise ModelDownloadError(
                    f"another download holds {path} and did not finish within "
                    f"{LOCK_WAIT_SECONDS:.0f} s; try again later, or delete the file if no "
                    f"download is running"
                ) from None
            time.sleep(LOCK_POLL_SECONDS)
    token = uuid.uuid4().hex  # the heartbeat and the release touch only a file that holds it
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "utc": _now(), "token": token}) + "\n")
    stop = threading.Event()
    beat = threading.Thread(target=_heartbeat, args=(path, token, stop), daemon=True)
    beat.start()
    try:
        yield
    finally:
        stop.set()
        beat.join()
        with contextlib.suppress(OSError):
            if _lock_token(path) == token:
                path.unlink()


def _heartbeat(path: Path, token: str, stop: threading.Event) -> None:
    while not stop.wait(LOCK_HEARTBEAT_SECONDS):
        with contextlib.suppress(OSError):
            if _lock_token(path) == token:
                os.utime(path)


def _lock_token(path: Path) -> str | None:
    """The token in a lock file, or ``None`` for one that has none (empty, not JSON, or written
    before tokens were added). An unreadable file raises `OSError`."""
    try:
        data = json.loads(path.read_text(encoding="utf-8") or "null")
    except ValueError:
        return None
    token = data.get("token") if isinstance(data, dict) else None
    return token if isinstance(token, str) else None


def _break_stale(path: Path) -> bool:
    """Remove ``path`` if it is stale. True when the caller should try to take the lock again."""
    try:
        age = time.time() - path.stat().st_mtime
        token = _lock_token(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if age < LOCK_STALE_SECONDS:
        return False
    # Rename the stale file aside, then check that what was moved is still that stale file (same
    # token, still not refreshed). If a fresh lock replaced it between the check and the rename,
    # it is put back (a hard link, which never replaces a lock created meanwhile) and this waiter
    # keeps waiting. What is guaranteed: a waiter only deletes a file it has confirmed stale after
    # moving it, and a holder only refreshes or removes a file holding its own token. Not
    # guaranteed: if a fresh lock cannot be put back, two processes may both download; each still
    # installs only a bundle verified against the pins.
    aside = path.with_name(f"{path.name}.stale-{uuid.uuid4().hex[:8]}")
    try:
        os.rename(path, aside)
    except OSError:
        return True
    try:
        same = (
            _lock_token(aside) == token
            and time.time() - aside.stat().st_mtime >= LOCK_STALE_SECONDS
        )
    except OSError:
        same = False
    if not same:
        _put_back(aside, path)
        return False
    with contextlib.suppress(OSError):
        aside.unlink()
    log.warning("took over a stale download lock %s (%.0f s old)", path, age)
    return True


def _put_back(aside: Path, path: Path) -> None:
    """Move a lock file that `_break_stale` moved aside by mistake back to ``path``, unless a new
    lock is there by now."""
    with contextlib.suppress(OSError):
        try:
            os.link(aside, path)  # fails if ``path`` exists, unlike a POSIX rename
        except FileExistsError:
            return
        except OSError:  # no hard links on this file system
            if path.exists():
                return
            os.rename(aside, path)
            return
        aside.unlink()


# ------------------------------------------------------------------------------------- helpers

_TRANSFER_ERRORS: Final = (
    OSError,
    http.client.HTTPException,
    tarfile.TarError,
    EOFError,
    zlib.error,
    ValueError,  # e.g. urllib refusing a malformed address
)


def _installed(name: str) -> Path | None:
    from glypharis.export.bundle import find_bundle

    try:
        return find_bundle(name)
    except ModelLoadError:
        return None


def _manifest_sha256(bundle: Path) -> str:
    from glypharis.data.download import sha256_file

    try:
        return sha256_file(bundle / "manifest.json")
    except OSError as exc:
        raise ModelDownloadError(f"{bundle}: cannot read manifest.json ({exc})") from exc


def _same_path(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def read_record(bundle: str | Path) -> dict[str, Any] | None:
    """The download record beside a bundle `download_model` installed, or ``None`` if there is
    none (a bundle installed by hand)."""
    path = Path(bundle) / RECORD_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelDownloadError(f"{path}: unreadable download record ({exc})") from exc
    return data if isinstance(data, dict) else None
