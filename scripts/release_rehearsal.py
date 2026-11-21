"""Fetch the release archive into an empty models folder and check the installed package on it
(Phase 23, items 8 and 16: GW-E2E-022 and GW-E2E-023).

    # item 8: the packed archive from a local server
    <venv>/python scripts/release_rehearsal.py --experiment GW-E2E-022 --serve <archive folder> \
        --phase10 <GW-E2E-006 hiertext_val.jsonl> --out reports/package/GW-E2E-022.json
    # item 16: the published asset at the registry's address
    <venv>/python scripts/release_rehearsal.py --experiment GW-E2E-023 \
        --phase10 <GW-E2E-006 hiertext_val.jsonl> --out reports/package/GW-E2E-023.json

Run it with the Python of the environment the built or published wheel is installed in, from the
checkout (``--v1`` checks the working directory against ``v1-code``). It refuses unless
``glypharis`` is imported from outside the checkout and ``GLYPHARIS_MODELS``, ``HOME`` and
``USERPROFILE`` point at empty folders, so the bundle can only come from this fetch. With
``--serve`` the archive is served by a local HTTP server and fetched with ``download_model(...,
url=...)``, which still enforces the registry's size and hashes; without it the registry's own
address is used and the entry must be published. Then ``glypharis-check`` reads the sample page,
and ``scripts/package_check.py`` reads the 400 validation pages with ``--backend torch --threads 10
--v1`` against GW-E2E-006. The fetch record is written beside ``--out`` as ``<ID>_fetch.json``:
the installed package's version and the SHA-256 of its RECORD (and of the wheel file with
``--wheel``), the archive and manifest hashes, the download record (without
local paths) and the check results. Exit status 0 when every step passes.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import re
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REFERENCE_SHA256 = "c8520576f44f8b2ee436f9103556556b2f5e6685379b95bee3ef2403eb73eb4b"
MANIFEST_SHA256 = "875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037"
NAME = "glypharis-v1"
# A drive path or an absolute home-like path: lines that still hold one are left out of the record.
LOCAL_PATH = re.compile(r"\b[A-Za-z]:[\\/]|(?<![\w.])/(?:home|Users|tmp|mnt|root)/")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return


def sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        raise SystemExit(f"{var} must point at an empty folder")
    path = Path(value.split(os.pathsep)[0])
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"{var} must point at an empty folder; {var} is not empty")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _serve(folder: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(_Quiet, directory=str(folder))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/{NAME}.tar.gz"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--serve", type=Path, default=None, help="folder holding the archive")
    parser.add_argument("--phase10", type=Path, required=True)
    parser.add_argument("--config", default="configs/det/gw-det-001.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, default=None, help="the wheel file installed")
    args = parser.parse_args(argv)

    import hashlib
    import importlib.metadata

    import glypharis
    from glypharis.hub import REGISTRY, download_model, read_record

    checkout = Path.cwd().resolve()
    package = Path(glypharis.__file__).resolve().parent
    if checkout in package.parents:
        raise SystemExit("glypharis is imported from the checkout, not from the installed wheel")
    if args.out.is_absolute():
        raise SystemExit("--out must be a path relative to the checkout")
    models = _empty("GLYPHARIS_MODELS")
    for var in ("HOME", "USERPROFILE"):
        if os.environ.get(var):
            _empty(var)
    fetch_out = args.out.with_name(f"{args.experiment}_fetch.json")
    if args.out.exists() or fetch_out.exists():
        raise SystemExit(f"{args.out.as_posix()} or {fetch_out.as_posix()} exists; written once")

    dist = importlib.metadata.distribution("glypharis")
    record_text = dist.read_text("RECORD") or ""
    wheel: dict[str, Any] = {
        "version": dist.version,
        "record_sha256": hashlib.sha256(record_text.encode("utf-8")).hexdigest(),
        "file": None,
    }
    if args.wheel is not None:
        wheel["file"] = {"name": args.wheel.name, "sha256": sha256_of(args.wheel)}

    entry = REGISTRY[NAME]
    server = None
    url = None
    if args.serve is not None:
        server, url = _serve(args.serve)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        bundle = download_model(NAME, models_dir=models, url=url)
    finally:
        if server is not None:
            server.shutdown()
    record = read_record(bundle) or {}
    if not record.get("network_fetch"):
        raise SystemExit("the bundle was not fetched over the network in this run")
    if Path(bundle).resolve().parent != models.resolve():
        raise SystemExit("the bundle was not installed in the empty models folder")

    secrets = sorted(
        {
            str(p)
            for p in (
                models,
                models.resolve(),
                package.parent,
                Path(sys.prefix),
                Path(sys.prefix).resolve(),
                Path.home(),
                checkout,
                Path(os.environ.get("USERPROFILE") or checkout),
            )
        },
        key=len,
        reverse=True,
    )

    def scrub(text: str) -> list[str]:
        lines = []
        for line in text.strip().splitlines():
            for secret in secrets:
                line = line.replace(secret, "<local>")
            if LOCAL_PATH.search(line):
                line = "<line with a local path left out>"
            lines.append(line)
        return lines

    # The installed console script, run from an empty folder, so the checkout is not on its path.
    script = Path(sys.executable).with_name(
        "glypharis-check.exe" if os.name == "nt" else "glypharis-check"
    )
    if not script.is_file():
        raise SystemExit("glypharis-check is not installed beside this Python")
    where = subprocess.run(
        [sys.executable, "-P", "-c", "import glypharis.check as c; print(c.__file__)"],
        capture_output=True,
        text=True,
        cwd=models,
        check=True,
    ).stdout.strip()
    if checkout in Path(where).resolve().parents:
        raise SystemExit("glypharis.check would be imported from the checkout")
    check = subprocess.run(
        [str(script), "--model", NAME, "--threads", "4"],
        capture_output=True,
        text=True,
        cwd=models,
        check=False,
    )
    if check.returncode != 0:
        print(check.stdout, check.stderr, sep=os.linesep, file=sys.stderr)
    package_check = subprocess.run(
        [
            sys.executable,
            "scripts/package_check.py",
            "--config",
            args.config,
            "--model",
            NAME,
            "--backend",
            "torch",
            "--threads",
            "10",
            "--v1",
            "--phase10",
            str(args.phase10),
            "--reference-sha256",
            REFERENCE_SHA256,
            "--expect-manifest-sha256",
            MANIFEST_SHA256,
            "--experiment",
            args.experiment,
            "--out",
            str(args.out),
        ],
        check=False,
    )
    doc = {
        "note": "Generated by scripts/release_rehearsal.py; do not edit by hand. No local paths.",
        "experiment": args.experiment,
        "started_utc": started,
        "ended_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "glypharis_version": glypharis.__version__,
        "installed_from_checkout": False,
        "wheel": wheel,
        "python": sys.version.split()[0],
        "source": "local server" if args.serve is not None else "registry address",
        "registry": {
            "url": entry.url,
            "published": entry.published,
            "archive_bytes": entry.bytes,
            "archive_sha256": entry.archive_sha256,
            "manifest_sha256": entry.manifest_sha256,
        },
        "download_record": {k: v for k, v in record.items() if k != "url" or args.serve is None},
        "glypharis_check": {
            "exit": check.returncode,
            "output": scrub(check.stdout),
            "errors": scrub(check.stderr),
        },
        "package_check": {"exit": package_check.returncode, "report": args.out.as_posix()},
        "passed": check.returncode == 0 and package_check.returncode == 0,
    }
    fetch_out.parent.mkdir(parents=True, exist_ok=True)
    fetch_out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{fetch_out.as_posix()}: {'passed' if doc['passed'] else 'FAILED'}")
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
