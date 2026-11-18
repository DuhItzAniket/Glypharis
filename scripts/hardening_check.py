"""The Phase 20 hardening cases on the v1 bundle itself (GW-E2E-012, Phase 20 v1 re-run, item 2).

    python scripts/hardening_check.py --bundle runs/release/glypharis-v1 \
        [--backend torch onnx] [--threads 10] [--page data/raw/hiertext/images/validation/X.jpg] \
        [--out reports/hardening/GW-E2E-012.json] [--experiment GW-E2E-012]

`tests/test_hardening.py` checks the input edge on tiny untrained models. This runs the same cases
on the v1 bundle, loaded as BookWorm loads it: `GlypharisOCR.from_pretrained(bundle, backend,
threads, cache=False)`, on each backend asked for (PyTorch and ONNX by default) at ``--threads``.
Per backend it runs the 120 seeded fuzz-lite cases, the unusual pages and the wrong types and
values (all from `hardening_cases`, shared with the test), the same inputs as file bytes where
they have a bytes form (files read, arrays and PIL images encoded as PNG, or TIFF where PNG cannot
hold the mode), corrupt, truncated, empty and non-image files and their bytes, a missing path, a
directory, images over the pixel limit (at the default limit from the file header, and at a limit
set on the instance below the good page's size), and the good page (``--page``, a HierText
validation image) as a path, bytes, a PIL image and an array. Each input ends in a result, a
`GlypharisError` (counted by code) or another exception (recorded with its type and message).

It also checks, per backend: damaged bundles, made only in temporary copies of the bundle (each
listed file removed, the manifest removed, the bundle missing, one byte changed in each listed
file, a manifest that is not JSON, a foreign format, and each listed file replaced by junk with
the copy's manifest re-hashed so the hashes match), each of which must raise `ModelLoadError`; the
one exception is junk in a file the backend never opens (PyTorch reads only the ``.pt`` files,
`glypharis.export.bundle._load`), which must load. Then the cache (``cache=True`` twice gives the
same object, ``cache=False`` a new one) and the INFO line of the ``glypharis`` logger naming
`MODEL_VERSION`; and the HTTP API (`glypharis.server`, Phase 19) on a free local port, ``POST
/ocr`` with the good page (200 and a result) and a non-image file (400 ``invalid_image``). Once:
`glypharis.version_info` against ``python -m glypharis.cli --version``, and the CLI (``python -m
glypharis.cli --input-dir <a folder with the good page and a non-image file> --model <bundle>
--backend torch --output <folder>``, run outside the repository with this code first on
PYTHONPATH), which must exit 1, report the bad file with ``[invalid_image]`` and write the good
page's JSON.

It refuses, before anything loads: an existing ``--out`` (the check is written once); v1 code that
differs from its tag (`assert_v1_code`); a ``glypharis`` imported from somewhere other than the
checked tree; a bundle whose manifest SHA-256 is not ``--expect-manifest-sha256`` (the GW-EFF-006
bundle's by default) or that does not read (`read_manifest`); a missing ``--page``; and the ONNX
backend without ONNX Runtime. The bundle is never written to.

Pass rule (registered in docs/PHASE_STATUS.md, Phase 20 v1 re-run): no exception other than a
`GlypharisError` anywhere, every damaged bundle a `ModelLoadError`, and the cache, load log,
version, CLI and API checks as stated. Whether each kind of input ended as expected (bad files
refused as ``invalid_image``, the oversized images as ``image_too_large``, the good page read) is
recorded beside the rule, not in it. Times are not measured (the latency numbers are Phase 16's).

The output holds no local path: the bundle and the page are their names, and every message has
its paths replaced by placeholders (``<copy>``, ``<work>``, ``<bundle>``, ``<home>``, ``<path>``).
Output: reports/hardening/GW-E2E-012.json, written in one step. Exit status 0 on a pass, 1
otherwise.
"""

from __future__ import annotations

import argparse
import http.client
import importlib.util
import io
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from hardening_cases import (
    FUZZ_CASES,
    FUZZ_SEED,
    blank_page,
    empty_page,
    fuzz_cases,
    unusual_pages,
    wrong_inputs,
)
from PIL import Image
from run_baselines import git_state

import glypharis
from glypharis import version_info
from glypharis.data.download import sha256_file
from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.v1 import V1CodeError, assert_v1_code
from glypharis.errors import GlypharisError, ModelLoadError
from glypharis.export.bundle import FILES, read_manifest
from glypharis.pipeline.ocr import DEFAULT_MAX_PIXELS, GlypharisOCR, OCRResult
from glypharis.server import make_server

EXPERIMENT = "GW-E2E-012"
# The GW-EFF-006 bundle, the one v1 ships (docs/PHASE_STATUS.md, Phase 17 and 20 v1 re-runs).
V1_BUNDLE_SHA256 = "875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037"
MODEL_VERSION = "glypharis_v1"  # the manifest version the load log line must name
OUT = Path("reports/hardening/GW-E2E-012.json")
PAGE = Path("data/raw/hiertext/images/validation/002a72617c3b1335.jpg")
DEFAULT_THREADS = 10
BACKENDS = ("torch", "onnx")
# The bundle files each backend opens (glypharis.export.bundle._load); junk in another file loads.
READS = {"torch": ("detector.pt", "recognizer.pt"), "onnx": FILES}
MESSAGE_CHARS = 120  # of a damaged bundle's ModelLoadError message
OTHER_CHARS = 300  # of any other exception's message
NOT_AN_IMAGE = b"this is not an image\n"
CLI_BAD = "not_an_image.png"
NO_BYTES = object()  # an input with no file-bytes form (a missing path, None, a 4-D array)

_WIN_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\s'\",;)\]]*")
_POSIX_PATH = re.compile(r"(?<![\w.<>/])/(?:tmp|home|Users|var|private|mnt|root)/[^\s'\",;)\]]*")
_PLACEHOLDER_PATH = re.compile(r"<[a-z]+>[^\s'\",;)\]]*")
_SECONDS = re.compile(r"\bin \d+(?:\.\d+)? s\b")


def _utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Scrub:
    """Replaces known local paths (in their plain, forward-slash, resolved and repr-escaped
    spellings) with placeholders, then any other absolute path with ``<path>``."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def add(self, path: str | Path, placeholder: str) -> None:
        p = Path(path)
        forms = {str(p), p.as_posix(), str(p.resolve()), p.resolve().as_posix()}
        forms |= {f.replace("\\", "\\\\") for f in forms}
        self.pairs += [(f, placeholder) for f in forms if f not in ("", ".")]
        self.pairs.sort(key=lambda pair: -len(pair[0]))  # the longest (most specific) first

    def __call__(self, text: str) -> str:
        for form, placeholder in self.pairs:
            text = text.replace(form, placeholder)
        text = _POSIX_PATH.sub("<path>", _WIN_PATH.sub("<path>", text))
        # "<copy>\x\y" (or repr's "<copy>\\x\\y") reads "<copy>/x/y"
        return _PLACEHOLDER_PATH.sub(lambda m: re.sub(r"\\{1,2}", "/", m.group(0)), text)

    def tree(self, obj: Any) -> Any:
        """``obj`` with every string value scrubbed, at any depth."""
        if isinstance(obj, str):
            return self(obj)
        if isinstance(obj, dict):
            return {k: self.tree(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.tree(v) for v in obj]
        return obj


def as_bytes(x: Any) -> Any:
    """The file-bytes form of an input: bytes as they are, a file's contents, an array or PIL image
    encoded as PNG (TIFF when PNG cannot hold it); `NO_BYTES` when there is none."""
    if isinstance(x, bytes):
        return x
    if isinstance(x, Path):
        return x.read_bytes() if x.is_file() else NO_BYTES
    if isinstance(x, Image.Image):
        img = x
    elif isinstance(x, np.ndarray):
        a = x[..., 0] if x.ndim == 3 and x.shape[2] == 1 else x
        try:
            img = Image.fromarray(a)
        except Exception:  # a shape or type PIL has no mode for: no image file can hold it
            return NO_BYTES
    else:
        return NO_BYTES
    # 32-bit "I" images go to TIFF: Pillow is dropping them from PNG (deprecated in Pillow 12).
    for fmt in ("PNG", "TIFF") if img.mode != "I" else ("TIFF",):
        buf = io.BytesIO()
        try:
            img.save(buf, format=fmt)
        except Exception:  # the format cannot hold this mode or size
            continue
        return buf.getvalue()
    return NO_BYTES


def _empty_tally() -> dict[str, Any]:
    return {"inputs": 0, "results": 0, "glypharis_errors": Counter(), "no_bytes_form": 0,
            "other_exceptions": []}  # fmt: skip


def tally(
    ocr: GlypharisOCR,
    cases: Iterable[tuple[str, str, Any]],
    into: dict[str, dict[str, Any]],
    scrub: Scrub,
    on_result: Callable[[str, OCRResult], None] | None = None,
) -> None:
    """Run each ``(kind, name, input)`` through ``ocr.predict`` and count, per kind, results,
    `GlypharisError` codes and every other exception; `NO_BYTES` inputs are counted, not run."""
    for kind, name, x in cases:
        t = into.setdefault(kind, _empty_tally())
        if x is NO_BYTES:
            t["no_bytes_form"] += 1
            continue
        t["inputs"] += 1
        try:
            result = ocr.predict(x)
        except GlypharisError as exc:
            t["glypharis_errors"][exc.code] += 1
        except Exception as exc:  # what the check is looking for: anything but a GlypharisError
            t["other_exceptions"].append(
                {"case": name, "type": type(exc).__name__, "message": scrub(str(exc))[:OTHER_CHARS]}
            )
        else:
            t["results"] += 1
            if on_result is not None:
                on_result(name, result)


def _finish(kinds: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    for t in kinds.values():
        t["glypharis_errors"] = dict(sorted(t["glypharis_errors"].items()))
    return dict(sorted(kinds.items()))


def _bad_files(directory: Path, page: Path) -> list[tuple[str, Path]]:
    directory.mkdir(parents=True, exist_ok=True)
    data = page.read_bytes()
    contents = {
        "corrupt.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 40,  # a PNG signature, then nothing valid
        f"truncated{page.suffix.lower()}": data[: len(data) // 3],
        "empty.png": b"",
        "text.jpg": NOT_AN_IMAGE,
    }
    out = []
    for name, body in contents.items():
        (directory / name).write_bytes(body)
        out.append((name, directory / name))
    (directory / "a_directory").mkdir(exist_ok=True)
    out.append(("missing_path", directory / "missing.png"))
    out.append(("directory", directory / "a_directory"))
    return out


def over_limit_inputs(
    directory: Path, max_pixels: int = DEFAULT_MAX_PIXELS
) -> list[tuple[str, Any]]:
    """A file, its bytes and an array one row past ``max_pixels`` (refused from the file header,
    before decoding; the array before conversion)."""
    w = math.isqrt(max_pixels) + 1
    h = max_pixels // w + 1  # w * h > max_pixels
    path = directory / "over_limit.png"
    Image.new("1", (w, h)).save(path)
    return [("default_limit_file", path), ("default_limit_bytes", path.read_bytes()),
            ("default_limit_array", np.zeros((h, w), np.uint8))]  # fmt: skip


def input_checks(
    ocr: GlypharisOCR,
    work: Path,
    page: Path,
    over_limit: list[tuple[str, Any]],
    scrub: Scrub,
) -> dict[str, Any]:
    kinds: dict[str, dict[str, Any]] = {}
    for sub in ("fuzz", "fuzz_bytes"):
        (work / sub).mkdir()
    fuzz = fuzz_cases(work / "fuzz")
    tally(ocr, ((f"fuzz_lite/{k}", f"{i}", x) for i, (k, x) in enumerate(fuzz)), kinds, scrub)
    fuzz = fuzz_cases(work / "fuzz_bytes")  # the same cases again, as bytes
    fuzz_b = ((f"fuzz_lite_bytes/{k}", f"{i}", as_bytes(x)) for i, (k, x) in enumerate(fuzz))
    tally(ocr, fuzz_b, kinds, scrub)
    unusual = [("blank", blank_page()), ("empty", empty_page()), *unusual_pages()]
    tally(ocr, (("unusual_pages", n, x) for n, x in unusual), kinds, scrub)
    tally(ocr, (("unusual_pages_bytes", n, as_bytes(x)) for n, x in unusual), kinds, scrub)
    wrong = wrong_inputs()
    tally(ocr, (("wrong_inputs", n, x) for n, x in wrong), kinds, scrub)
    tally(ocr, (("wrong_inputs_bytes", n, as_bytes(x)) for n, x in wrong), kinds, scrub)
    bad = _bad_files(work / "bad", page)
    tally(ocr, (("bad_files", n, x) for n, x in bad), kinds, scrub)
    tally(ocr, (("bad_files_bytes", n, as_bytes(x)) for n, x in bad), kinds, scrub)
    tally(ocr, (("over_pixel_limit", n, x) for n, x in over_limit), kinds, scrub)
    with Image.open(page) as opened:
        opened.load()
        page_img = opened.copy()
    page_rgb = np.asarray(page_img.convert("RGB"))
    page_bytes = page.read_bytes()
    saved = ocr.max_pixels
    ocr.max_pixels = page_rgb.shape[0] * page_rgb.shape[1] - 1  # one pixel under the good page
    try:
        small = [("instance_limit_path", page), ("instance_limit_bytes", page_bytes),
                 ("instance_limit_array", page_rgb)]  # fmt: skip
        tally(ocr, (("over_pixel_limit", n, x) for n, x in small), kinds, scrub)
    finally:
        ocr.max_pixels = saved
    texts: dict[str, str] = {}
    lines: dict[str, int] = {}

    def keep(name: str, result: OCRResult) -> None:
        texts[name], lines[name] = result.text, len(result.lines)

    good = [("path", page), ("bytes", page_bytes), ("pil", page_img), ("array", page_rgb)]
    tally(ocr, (("good_page", n, x) for n, x in good), kinds, scrub, keep)
    return {
        "kinds": _finish(kinds),
        "good_page_lines": lines,
        "good_page_path_and_bytes_same_text": "path" in texts
        and texts["path"] == texts.get("bytes"),
        "max_pixels": ocr.max_pixels,
    }


def expected_outcomes(kinds: dict[str, dict[str, Any]]) -> dict[str, bool]:
    """Whether each kind of input ended as it should (recorded; only ``good_page_read`` is part
    of the pass rule, through the ``good_page`` check)."""

    def refused(kind: str, code: str) -> bool:
        t = kinds[kind]
        return t["results"] == 0 and set(t["glypharis_errors"]) == {code}

    u = kinds["unusual_pages"]
    return {
        "unusual_pages_read_but_the_empty_one": u["results"] == u["inputs"] - 1
        and u["glypharis_errors"] == {"invalid_image": 1},
        "wrong_inputs_invalid_image": refused("wrong_inputs", "invalid_image"),
        "bad_files_invalid_image": refused("bad_files", "invalid_image"),
        "bad_files_bytes_invalid_image": refused("bad_files_bytes", "invalid_image"),
        "over_pixel_limit_image_too_large": refused("over_pixel_limit", "image_too_large"),
        "good_page_read": kinds["good_page"]["results"] == kinds["good_page"]["inputs"] > 0,
        "fuzz_lite_120_cases": sum(
            t["inputs"] for k, t in kinds.items() if k.startswith("fuzz_lite/")
        )
        == FUZZ_CASES,
    }


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def cache_and_log(
    bundle: Path, backend: str, threads: int, scrub: Scrub
) -> tuple[dict[str, Any], GlypharisOCR]:
    """The cache and the load log line; returns them and a fresh (``cache=False``) instance."""
    logger = logging.getLogger("glypharis")
    handler = _Records()
    level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        first = GlypharisOCR.from_pretrained(bundle, backend, threads, cache=True)
    except ModelLoadError as exc:
        raise SystemExit(f"the bundle does not load on {backend}: {scrub(str(exc))}") from exc
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    again = GlypharisOCR.from_pretrained(bundle, backend, threads, cache=True)
    fresh = GlypharisOCR.from_pretrained(bundle, backend, threads, cache=False)
    loaded = [r.getMessage() for r in handler.records if r.levelno == logging.INFO]
    loaded = [m for m in loaded if "loaded GLYPHARIS model" in m]
    names = [m for m in loaded if MODEL_VERSION in m and f"({backend} backend)" in m]
    line = names[0] if names else (loaded[0] if loaded else None)
    return {
        "cached_twice_same_object": again is first,
        "uncached_new_object": fresh is not first,
        "load_log_line": None if line is None else _SECONDS.sub("in <seconds> s", scrub(line)),
        "load_log_names_model_version": bool(names),
        "model_version": first.model_version,
    }, fresh


def _post(conn: http.client.HTTPConnection, body: bytes) -> tuple[int, dict[str, Any]]:
    conn.request("POST", "/ocr", body=body)
    resp = conn.getresponse()
    return resp.status, json.loads(resp.read().decode("utf-8"))


def api_check(ocr: GlypharisOCR, page_bytes: bytes) -> dict[str, Any]:
    """``POST /ocr`` through `glypharis.server` on a free local port, as tests/test_server.py."""
    server = make_server(ocr, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=600)
    try:
        good_status, good = _post(conn, page_bytes)
        bad_status, bad = _post(conn, NOT_AN_IMAGE)
    finally:
        conn.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
    good_ok = good_status == 200 and isinstance(good.get("lines"), list)
    good_ok = good_ok and good.get("model_version") == MODEL_VERSION
    return {
        "good_page": {"status": good_status, "lines": len(good.get("lines", [])),
                      "model_version": good.get("model_version"), "error": good.get("error")},
        "bad_file": {"status": bad_status, "error": bad.get("error")},
        "ok": good_ok and bad_status == 400 and bad.get("error") == "invalid_image",
    }  # fmt: skip


def _damage(name: str, manifest: dict[str, Any]) -> Callable[[Path], None]:
    """What ``name`` does to a copy of the bundle."""
    how, _, file = name.partition(":")

    def apply(copy: Path) -> None:
        if how == "removed":
            (copy / file).unlink()
        elif how == "bundle_missing":
            shutil.rmtree(copy)
        elif how == "byte_changed":
            b = bytearray((copy / file).read_bytes())
            b[len(b) // 2] ^= 0xFF
            (copy / file).write_bytes(bytes(b))
        elif how == "manifest_not_json":
            (copy / "manifest.json").write_text("{not json", encoding="utf-8")
        elif how == "manifest_foreign_format":
            m = dict(manifest, format="something-else")
            (copy / "manifest.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
        elif how == "junk_rehashed":
            (copy / file).write_bytes(b"not a model file\n" * 64)
            files = dict(manifest["files"], **{file: sha256_file(copy / file)})
            m = dict(manifest, files=files)
            (copy / "manifest.json").write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
        else:
            raise ValueError(name)

    return apply


def damage_cases(manifest: dict[str, Any]) -> list[str]:
    files = list(manifest["files"])
    return [*(f"removed:{f}" for f in files), "removed:manifest.json", "bundle_missing",
            *(f"byte_changed:{f}" for f in files), "manifest_not_json",
            "manifest_foreign_format", *(f"junk_rehashed:{f}" for f in files)]  # fmt: skip


def damaged_bundles(
    bundle: Path,
    manifest: dict[str, Any],
    backends: list[str],
    threads: int,
    work: Path,
    scrub: Scrub,
) -> list[dict[str, Any]]:
    """Each damage in a fresh temporary copy, loaded with each backend."""
    rows = []
    for name in damage_cases(manifest):
        with tempfile.TemporaryDirectory(dir=work, ignore_cleanup_errors=True) as tmp:
            copy = Path(tmp) / bundle.name
            shutil.copytree(bundle, copy)
            _damage(name, manifest)(copy)
            scrub.add(copy, "<copy>")
            for backend in backends:
                file = name.partition(":")[2]
                unread = name.startswith("junk_rehashed:") and file not in READS[backend]
                expected = "loaded" if unread else "model_load_error"
                message = None
                try:
                    GlypharisOCR.from_pretrained(copy, backend, threads, cache=False)
                    outcome = "loaded"
                except ModelLoadError as exc:
                    outcome, message = "model_load_error", scrub(str(exc))[:MESSAGE_CHARS]
                except Exception as exc:  # a failure of the check
                    outcome = type(exc).__name__
                    message = scrub(str(exc))[:OTHER_CHARS]
                rows.append({"case": name, "backend": backend, "expected": expected,
                             "outcome": outcome, "message": message,
                             "ok": outcome == expected})  # fmt: skip
    return rows


def child_env() -> dict[str, str]:
    """This process's environment with the imported glypharis's tree first on PYTHONPATH."""
    env = dict(os.environ)
    root = str(Path(glypharis.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(p for p in (root, env.get("PYTHONPATH")) if p)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _python(args: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=1800,
                          check=False)  # fmt: skip


def version_check(cwd: Path, env: dict[str, str]) -> dict[str, Any]:
    info = version_info()
    proc = _python(["-m", "glypharis.cli", "--version"], cwd, env)
    printed = dict(ln.split(" ", 1) for ln in proc.stdout.splitlines() if " " in ln)
    return {
        "version_info": info,
        "cli_version": {"exit_code": proc.returncode, "printed": printed},
        "ok": proc.returncode == 0
        and printed == info
        and info["glypharis"] == glypharis.__version__,
    }


def cli_check(
    bundle: Path, page: Path, work: Path, env: dict[str, str], scrub: Scrub
) -> dict[str, Any]:
    inp, out = work / "cli_input", work / "cli_output"
    inp.mkdir()
    shutil.copyfile(page, inp / page.name)
    (inp / CLI_BAD).write_bytes(NOT_AN_IMAGE)
    scrub.add(inp, "<input>")
    scrub.add(out, "<output>")
    here = _python(["-c", "import glypharis; print(glypharis.__file__)"], work, env)
    this = Path(glypharis.__file__).resolve()
    same_code = here.returncode == 0 and Path(here.stdout.strip()).resolve() == this
    args = ["-m", "glypharis.cli", "--input-dir", str(inp), "--model", str(bundle.resolve()),
            "--backend", "torch", "--output", str(out)]  # fmt: skip
    proc = _python(args, work, env)
    failed = [ln for ln in proc.stderr.splitlines() if ln.startswith("FAILED ") and CLI_BAD in ln]
    read = [ln for ln in proc.stdout.splitlines() if page.name in ln and " lines in " in ln]
    result = out / f"{page.stem}.json"
    json_ok = False
    if result.is_file():
        written = json.loads(result.read_text(encoding="utf-8"))
        json_ok = written.get("model_version") == MODEL_VERSION
    reported = bool(failed) and "[invalid_image]" in failed[0]
    row: dict[str, Any] = {
        "command": "python -m glypharis.cli --input-dir <input> --model <bundle> --backend torch "
        "--output <output>",
        "cwd": "a temporary folder outside the repository",
        "runs_this_code": same_code,
        "exit_code": proc.returncode,
        "failed_line": scrub(failed[0]) if failed else None,
        "bad_file_reported_with_code": reported,
        "good_page_line": _SECONDS.sub("in <seconds> s", scrub(read[0])) if read else None,
        "good_page_json_written": json_ok,
        "ok": proc.returncode == 1 and reported and json_ok and same_code,
    }
    if not row["ok"]:
        row["stderr_tail"] = [scrub(ln) for ln in proc.stderr.splitlines()[-10:]]
    return row


def _v1_code(root: Path) -> dict[str, Any]:
    try:
        return assert_v1_code(root)
    except V1CodeError as exc:
        raise SystemExit(f"{exc}; {EXPERIMENT} runs v1 as tagged") from exc


def run(args: argparse.Namespace) -> int:
    if args.out.exists():
        raise SystemExit(f"{args.out.as_posix()} exists: the check is written once")
    backends = list(dict.fromkeys(args.backend))
    if "onnx" in backends and importlib.util.find_spec("onnxruntime") is None:
        raise SystemExit("the ONNX backend needs onnxruntime, which is not installed")
    if not args.page.is_file():
        raise SystemExit(f"{args.page.as_posix()}: no such page image")
    v1_code = _v1_code(args.root)
    imported = Path(glypharis.__file__).resolve().parents[1]
    if imported != args.root.resolve():
        raise SystemExit("the imported glypharis is not the checked tree's: put the repository "
                         "first on PYTHONPATH")  # fmt: skip
    manifest_path = args.bundle / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{args.bundle.as_posix()}: no manifest.json")
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != args.expect_manifest_sha256:
        raise SystemExit(f"{args.bundle.as_posix()}: manifest SHA-256 {manifest_sha}, not the "
                         f"registered bundle's {args.expect_manifest_sha256}")  # fmt: skip
    try:
        manifest = read_manifest(args.bundle)
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc

    scrub = Scrub()
    scrub.add(Path.home(), "<home>")
    scrub.add(tempfile.gettempdir(), "<tmp>")
    scrub.add(Path.cwd(), "<cwd>")
    scrub.add(args.root, "<repo>")
    scrub.add(args.bundle, "<bundle>")
    scrub.add(args.page, "<page>")
    started = _utc()
    inputs: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    api: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="gw-e2e-012-", ignore_cleanup_errors=True) as tmp:
        work = Path(tmp)
        scrub.add(work, "<work>")
        over_limit = over_limit_inputs(work)
        for backend in backends:
            (work / backend).mkdir()
            cache[backend], ocr = cache_and_log(args.bundle, backend, args.threads, scrub)
            inputs[backend] = input_checks(ocr, work / backend, args.page, over_limit, scrub)
            api[backend] = api_check(ocr, args.page.read_bytes())
            stats[backend] = {k: v for k, v in ocr.stats().items() if "seconds" not in k}
            del ocr
        del over_limit
        damaged = damaged_bundles(args.bundle, manifest, backends, args.threads, work, scrub)
        env = child_env()
        versions = version_check(work, env)
        cli = cli_check(args.bundle, args.page, work, env, scrub)
    ended = _utc()

    others = [
        {"backend": b, "kind": k, **e}
        for b, res in inputs.items()
        for k, t in res["kinds"].items()
        for e in t["other_exceptions"]
    ]
    checks = {
        "no_other_exceptions": not others
        and all(r["outcome"] in ("loaded", "model_load_error") for r in damaged),
        "damaged_bundles_model_load_error": all(r["ok"] for r in damaged),
        "cache": all(
            c["cached_twice_same_object"] and c["uncached_new_object"] for c in cache.values()
        ),
        "load_log_names_model_version": all(
            c["load_log_names_model_version"] for c in cache.values()
        ),
        "versions": versions["ok"],
        "cli": cli["ok"],
        "api": all(a["ok"] for a in api.values()),
        # Added by the dated amendment of 2026-11-18, before the run on the v1 bundle.
        "good_page": all(
            expected_outcomes(r["kinds"])["good_page_read"]
            and r["good_page_path_and_bytes_same_text"]
            for r in inputs.values()
        ),
    }
    unchanged = sha256_file(manifest_path) == manifest_sha
    try:
        read_manifest(args.bundle)
    except ModelLoadError:
        unchanged = False
    checks["bundle_unchanged"] = unchanged
    passed = all(checks.values())
    report = {
        "experiment": args.experiment,
        "what": "the Phase 20 hardening cases (fuzz-lite, unusual pages, wrong inputs, bad files, "
        "oversized images, damaged bundles, cache, log, versions, CLI, HTTP API) on the v1 bundle",
        "pass_rule": "no exception other than a GlypharisError anywhere, every damaged bundle a "
        "ModelLoadError (junk in a file the backend never opens must load), the cache, load "
        "log, version, CLI and API checks as stated, the good page read on every backend with "
        "the same text from its path and its bytes, and the bundle unchanged afterwards",
        "pass": passed,
        "checks": checks,
        "other_exceptions": others,
        "expected_outcomes": {b: expected_outcomes(r["kinds"]) for b, r in inputs.items()},
        "bundle_unchanged": unchanged,
        "loaded_with": "GlypharisOCR.from_pretrained(bundle, backend, threads, cache=False)",
        "bundle": args.bundle.name,
        "bundle_manifest_sha256": manifest_sha,
        "bundle_version": manifest.get("version"),
        "bundle_files": manifest["files"],
        "model_version": next(iter(cache.values()))["model_version"],
        "backends": backends,
        "threads": args.threads,
        "page": args.page.name,
        "page_sha256": sha256_file(args.page),
        "fuzz_seed": FUZZ_SEED,
        "fuzz_cases": FUZZ_CASES,
        "inputs": inputs,
        "cache_and_log": cache,
        "api": api,
        "stats": stats,
        "damaged_bundles": damaged,
        "versions": versions["version_info"],
        "cli_version": versions["cli_version"],
        "cli": cli,
        "v1_code": v1_code,
        **git_state(),
        "started_utc": started,
        "ended_utc": ended,
    }
    write_json_atomic(args.out, scrub.tree(report))
    failing = [k for k, v in checks.items() if not v]
    print(f"{args.experiment} hardening of {args.bundle.name} ({', '.join(backends)}, "
          f"{args.threads} threads): pass {passed}"
          + (f"; failing: {', '.join(failing)}" if failing else "")
          + f"; written to {args.out.as_posix()}")  # fmt: skip
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--backend", nargs="+", choices=BACKENDS, default=list(BACKENDS))
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help="passed to from_pretrained")  # fmt: skip
    parser.add_argument("--expect-manifest-sha256", default=V1_BUNDLE_SHA256,
                        help="the registered bundle's manifest SHA-256")  # fmt: skip
    parser.add_argument("--page", type=Path, default=PAGE, help="a good page image")
    parser.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--experiment", default=EXPERIMENT)
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if args.threads < 1:
        parser.error("--threads must be a positive integer")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
