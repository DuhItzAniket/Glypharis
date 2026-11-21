"""Outsider check: ``glypharis-check`` (Phase 23, item 9).

    glypharis-check [--model glypharis-v1] [--models-dir DIR] [--threads 4] [--download] [--offline]

Checks that the installed engine and model are the ones the registry (`glypharis.hub.REGISTRY`)
pins, and that they reproduce the expected output on one sample page shipped with the package
(``glypharis/sample/``):

1. the bundle is found as ``glypharis-ocr --model NAME`` finds it (`find_bundle`), or at
   ``<models-dir>/NAME``; every file matches its ``manifest.json`` (`read_manifest`), and the
   SHA-256 of ``manifest.json`` equals the registry's pin;
2. if `glypharis.download_model` installed it, its download record (`glypharis.hub.read_record`)
   names the registry's archive size and SHA-256 (a bundle put in place by hand has no record and
   no archive to check; step 1 covers its files);
3. the sample page (its SHA-256 pinned in ``expected.json``) is read with the torch backend on the
   CPU, and the model version, the line texts in reading order and the joined text equal
   ``expected.json`` exactly.

Passing shows that the installed engine and weights are the pinned ones and reproduce the expected
output on one page. It says nothing about accuracy. Nothing is downloaded unless ``--download`` is
given (then `glypharis.download_model` fetches and verifies a missing bundle; ``--offline`` or
``GLYPHARIS_OFFLINE`` still forbids the network). Exit code 0 when every check passes, 1 when one
does not, 2 when the check cannot run (a usage error, or a `GlypharisError` such as a missing
model).

``--expected`` and ``--registry`` replace the shipped expected output and the registry, for tests
and for the clean-wheel job on a tiny bundle; a run with a test registry says that it does not
check the released weights.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from glypharis.errors import GlypharisError, ModelLoadError

if TYPE_CHECKING:
    from glypharis.hub import ModelEntry
    from glypharis.pipeline.ocr import OCRResult

SAMPLE_DIR: Final = Path(__file__).resolve().with_name("sample")
SAMPLE_PAGE: Final = SAMPLE_DIR / "page.png"
SAMPLE_EXPECTED: Final = SAMPLE_DIR / "expected.json"
EXPECTED_FORMAT: Final = "glypharis-sample-1"
BACKEND: Final = "torch"
DEFAULT_THREADS: Final = 4  # the shipped expected output was read with 4 threads

NO_ACCURACY: Final = (
    "It says nothing about accuracy: one clean page the project rendered itself is not a measure "
    "of how well GLYPHARIS reads other pages."
)


class SampleError(GlypharisError):
    """The expected output or a test registry file is missing or unreadable."""

    code = "sample_error"


# ------------------------------------------------------------------------------------- helpers


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_sha256(bundle: str | Path) -> str:
    """SHA-256 of a bundle's ``manifest.json`` (which holds the SHA-256 of every model file)."""
    path = Path(bundle) / "manifest.json"
    try:
        return _sha256(path)
    except OSError as exc:
        raise ModelLoadError(f"{path}: cannot read ({exc})") from exc


def read_sample(
    bundle: str | Path, page: Path = SAMPLE_PAGE, threads: int | None = None
) -> OCRResult:
    """The sample page read by ``bundle`` with the torch backend on the CPU (no cache)."""
    from glypharis.pipeline.ocr import GlypharisOCR

    ocr = GlypharisOCR.from_pretrained(Path(bundle), BACKEND, threads, cache=False)
    return ocr.predict(page)


def expected_record(
    result: OCRResult,
    name: str,
    manifest_sha: str,
    page: Path,
    threads: int | None,
    rendered_lines: Sequence[str],
) -> dict[str, Any]:
    """What ``expected.json`` holds: the text as rendered, and the predicted line texts in reading
    order, the joined text and the model version of one read of ``page``. No timings."""
    return {
        "format": EXPECTED_FORMAT,
        "model": name,
        "model_version": result.model_version,
        "manifest_sha256": manifest_sha,
        "backend": BACKEND,
        "threads": threads,
        "page": page.name,
        "page_sha256": _sha256(page),
        "width": result.width,
        "height": result.height,
        "rendered_lines": list(rendered_lines),
        "lines": [ln.text for ln in result.lines],
        "text": result.text,
    }


def load_expected(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SampleError(f"{path.name}: no readable expected output ({exc})") from exc
    needed = ("format", "model", "model_version", "manifest_sha256", "page_sha256", "lines", "text")
    if not isinstance(data, dict) or data.get("format") != EXPECTED_FORMAT:
        raise SampleError(f"{path.name} is not a {EXPECTED_FORMAT} file")
    missing = [k for k in needed if k not in data]
    if missing:
        raise SampleError(f"{path.name}: missing {', '.join(missing)}")
    return data


def load_registry(path: Path) -> list[ModelEntry]:
    """Entries from a registry file: ``{"models": [{ModelEntry fields}, ...]}``."""
    from glypharis.hub import ModelEntry

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["models"]
        fields = {f.name for f in dataclasses.fields(ModelEntry)}
        entries = []
        for row in rows:
            unknown = set(row) - fields
            if unknown:
                raise ValueError(f"unknown fields {sorted(unknown)}")
            entries.append(ModelEntry(**row))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SampleError(f"{path}: not a readable registry file ({exc})") from exc
    return entries


def _locate(name: str, models_dir: Path | None) -> Path | None:
    from glypharis.export.bundle import find_bundle

    if models_dir is not None:
        path = models_dir / name
        return path if (path / "manifest.json").is_file() else None
    try:
        return find_bundle(name)
    except ModelLoadError:
        return None


def _same(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve()


# --------------------------------------------------------------------------------------- check


@dataclasses.dataclass
class Check:
    say: Callable[[str], None] = print
    failures: list[str] = dataclasses.field(default_factory=list)

    def ok(self, what: str) -> None:
        self.say(f"  ok    {what}")

    def info(self, what: str) -> None:
        self.say(f"  note  {what}")

    def fail(self, what: str) -> None:
        self.failures.append(what)
        self.say(f"  FAIL  {what}")


def run_check(
    name: str = "glypharis-v1",
    *,
    models_dir: Path | None = None,
    threads: int | None = DEFAULT_THREADS,
    download: bool = False,
    offline: bool | None = None,
    expected_path: Path = SAMPLE_EXPECTED,
    page: Path = SAMPLE_PAGE,
    test_registry: Path | None = None,
    say: Callable[[str], None] = print,
) -> bool:
    """Run the check (module docstring) and report each step through ``say``. True when every
    check passed. Raises `GlypharisError` when the check cannot run."""
    from glypharis import hub
    from glypharis.export.bundle import read_manifest

    entry = hub.REGISTRY.get(name)
    if entry is None:
        known = ", ".join(sorted(hub.REGISTRY)) or "none"
        raise hub.ModelDownloadError(f"unknown model {name!r}; registered models: {known}")
    expected = load_expected(expected_path)

    bundle = _locate(name, models_dir)
    if bundle is None:
        where = str(models_dir) if models_dir is not None else "the models folders"
        if not download:
            raise ModelLoadError(
                f"{name!r} is not installed in {where}; nothing was downloaded. Install it with "
                f"glypharis.download_model({name!r}), or run glypharis-check --download"
            )
        bundle = hub.download_model(name, models_dir=models_dir, offline=offline)

    c = Check(say)
    say(f"glypharis-check: {name} at {bundle}")
    if test_registry is not None:
        c.info(f"registry: the test registry {test_registry}, not the package's own")
    if models_dir is not None:
        default = _locate(name, None)
        if default is None or not _same(default, bundle):
            c.info(
                f"glypharis-ocr --model {name} would load "
                f"{default if default is not None else 'nothing (not found)'}, not this bundle"
            )

    # 1. The bundle's files and its manifest.
    files_ok = True
    try:
        read_manifest(bundle)
        c.ok("every model file matches the bundle's manifest.json")
    except ModelLoadError as exc:
        files_ok = False
        c.fail(f"the bundle's files do not match its manifest.json: {exc}")
    sha = manifest_sha256(bundle)
    if sha == entry.manifest_sha256:
        c.ok(f"manifest.json SHA-256 {sha} matches the registry")
    else:
        c.fail(
            f"manifest.json SHA-256 {sha} is not the registry's {entry.manifest_sha256}: this "
            f"is not the registered {name}"
        )

    # 2. The download record.
    try:
        record = hub.read_record(bundle)
    except hub.ModelDownloadError as exc:
        c.fail(str(exc))
    else:
        if record is None:
            c.info(
                "no download record: the bundle was put in place by hand, so there is no archive "
                "to check (the manifest check above covers its files)"
            )
        else:
            _check_record(c, record, entry)

    # 3. The sample page.
    if expected["model"] != name:
        c.fail(
            f"{expected_path.name} is the expected output of {expected['model']!r}, not {name!r}"
        )
    if expected["manifest_sha256"] != entry.manifest_sha256:
        c.fail(
            f"{expected_path.name} was made with manifest "
            f"{expected['manifest_sha256'][:12]}, not the registry's {entry.manifest_sha256[:12]}"
        )
    page_sha = _sha256(page)
    if page_sha != expected["page_sha256"]:
        c.fail(
            f"{page.name}: SHA-256 {page_sha[:12]} is not the expected "
            f"{expected['page_sha256'][:12]}"
        )
    elif files_ok:
        result = read_sample(bundle, page, threads)
        _compare(c, result, expected)

    if c.failures:
        say(f"FAILED: {len(c.failures)} check(s) did not pass.")
        say(
            f"The installed engine or model is not the registered {name}, or it does not "
            f"reproduce the expected output on the sample page."
        )
        return False
    say("PASSED.")
    if test_registry is not None:
        say(
            "Checked against a test registry, not the package's own: this does not show that the "
            "weights are the released ones. " + NO_ACCURACY
        )
    elif entry.published:
        say(
            "This shows that the installed GLYPHARIS engine and weights are the released ones "
            "and reproduce the expected output on one sample page. " + NO_ACCURACY
        )
    else:
        say(
            f"This shows that the installed GLYPHARIS engine and weights are the ones the "
            f"registry pins for {name} (not yet published) and reproduce the expected output on "
            f"one sample page. " + NO_ACCURACY
        )
    return True


def _check_record(c: Check, record: dict[str, Any], entry: ModelEntry) -> None:
    source = record.get("url") or "an archive fetched earlier"
    if entry.archive_sha256 is None or entry.bytes is None:
        c.fail(
            f"the bundle has a download record (from {source}), but the registry pins no archive "
            f"for {entry.name} yet, so the record cannot be checked"
        )
        return
    bad = [
        f"{key} {record.get(key)!r}, registry {want!r}"
        for key, want in (
            ("name", entry.name),
            ("archive_sha256", entry.archive_sha256),
            ("archive_bytes", entry.bytes),
            ("manifest_sha256", entry.manifest_sha256),
        )
        if record.get(key) != want
    ]
    if bad:
        c.fail("the download record does not match the registry: " + "; ".join(bad))
    else:
        c.ok(
            f"download record: archive SHA-256 {entry.archive_sha256} and size {entry.bytes} "
            f"bytes match the registry (from {source})"
        )


def _compare(c: Check, result: OCRResult, expected: dict[str, Any]) -> None:
    got = [ln.text for ln in result.lines]
    want = list(expected["lines"])
    if result.model_version != expected["model_version"]:
        c.fail(f"model version {result.model_version!r}, expected {expected['model_version']!r}")
    if got == want and result.text == expected["text"]:
        c.ok(f"sample page: {len(got)} lines read, identical to the expected output")
        return
    if len(got) != len(want):
        c.fail(f"sample page: {len(got)} lines read, expected {len(want)}")
    for i, (g, w) in enumerate(zip(got, want, strict=False)):
        if g != w:
            c.fail(f"sample page, line {i + 1}: read {g!r}, expected {w!r}")
            break
    else:
        if len(got) == len(want):
            c.fail("sample page: the joined text differs from the expected text")


# ---------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glypharis-check", description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="glypharis-v1", help="registered model name")
    parser.add_argument(
        "--models-dir", type=Path, default=None, help="look only here (default: the models folders)"
    )
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument(
        "--download", action="store_true", help="fetch and verify the model if it is not installed"
    )
    parser.add_argument(
        "--offline", action="store_true", help="never use the network, even with --download"
    )
    parser.add_argument("--expected", type=Path, default=SAMPLE_EXPECTED, help=argparse.SUPPRESS)
    parser.add_argument("--registry", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    from glypharis import hub

    try:
        if args.registry is not None:
            with hub.override_registry(*load_registry(args.registry)):
                passed = _run(args)
        else:
            passed = _run(args)
    except GlypharisError as exc:
        print(f"glypharis-check cannot run: [{exc.code}] {exc}", file=sys.stderr)
        return 2
    return 0 if passed else 1


def _run(args: argparse.Namespace) -> bool:
    return run_check(
        args.model,
        models_dir=args.models_dir,
        threads=args.threads,
        download=args.download,
        offline=True if args.offline else None,
        expected_path=args.expected,
        test_registry=args.registry,
    )


if __name__ == "__main__":
    sys.exit(main())
