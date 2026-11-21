# Integrating GLYPHARIS

This guide is for an application that embeds GLYPHARIS as its OCR engine: the application turns a page
into an image, hands it to GLYPHARIS, and stores the text that comes back. GLYPHARIS knows nothing about
PDFs, chunking, search or any particular application ([ARCHITECTURE §1](ARCHITECTURE.md#1-scope)), and
nothing here is specific to one.

**Status.** v1 (`glypharis_v1`) is evaluated and packaged, not released. No weights are distributed,
the package is not on PyPI yet, and, until the weights are released after the qualified advice of
ADR-0014, `glypharis.download_model` refuses with "not yet published" whenever the bundle is not
already installed and no `url=` is given
([`docs/PHASE_STATUS.md`](PHASE_STATUS.md), Phase 23, D4). Everything below describes the code as it is
on this branch; where something depends on the release, the text says so.

Contents: [what it expects](#what-it-expects), [installing](#installing),
[getting the model](#getting-the-model), [reading pages](#reading-pages),
[threads](#threads-and-concurrency), [the HTTP server](#the-http-server),
[errors](#errors), [storing results](#storing-results), [checking an install](#checking-an-install),
[phones](#phones).

## What it expects

- **Upright page images of printed English text**: scans, photographs of flat pages, image-only pages,
  one image at a time. The output is text lines in reading order, each with a box, a recognition
  confidence and a detector score.
- **The CPU.** `GlypharisOCR.from_pretrained` loads the model on the CPU; the public API has no device
  option. The default backend is PyTorch (`backend="torch"`); `backend="onnx"` uses ONNX Runtime,
  which matches PyTorch within Phase 17's tolerances (GW-EFF-006, `reports/efficiency/phase16/README.md`)
  but is slower on the CPU (GW-EFF-007, F34, `reports/efficiency/phase16/cost_table.md`). Any other
  backend name raises a plain `ValueError`.

The model card lists what v1 is not for, with the evidence for each row
([`MODEL_CARD.md`, not intended use](../MODEL_CARD.md#not-intended-use) and
[limitations](../MODEL_CARD.md#limitations)). In short:

- handwriting, and scripts other than English;
- pages photographed sideways: there is no page-orientation step, so rotate pages upright before
  calling GLYPHARIS (GW-E2E-007, `reports/external/end_to_end.md`);
- curved pages, such as an open book, and vertical text (GW-E2E-007);
- tables, formulas and layout analysis: the model returns text lines only (GW-E2E-009,
  `reports/external/official.md`);
- on-device use on phones ([phones](#phones));
- unreviewed transcription where errors matter: output should be searched, indexed or read by a
  person who can tolerate errors (GW-E2E-007);
- the confidence is not a probability ([confidence calibration](../MODEL_CARD.md#confidence-calibration)).

Accepted inputs (`glypharis.pipeline.ocr.load_rgb`):

| Input | Notes |
|---|---|
| A file path (`str` or `Path`) | Opened with its EXIF orientation applied |
| The bytes of an image file | As a path; empty bytes are refused |
| A PIL image | Used as given (no EXIF step) |
| A NumPy array | Shape `(H, W)`, `(H, W, 1)`, `(H, W, 3)` or `(H, W, 4)`, channels in RGB order, a fourth channel dropped; `uint8`, `bool`, `uint16` (its high byte), floats in [0, 1] or [0, 255], or other integers within 0 to 255. NaN or infinite values are refused |

An image with more pixels than `max_pixels` is refused with `image_too_large`, checked from the file
header before decoding. The default limit is the constant `DEFAULT_MAX_PIXELS` in
`glypharis/pipeline/ocr.py`; the model card notes it is a set value that was not measured against
memory use. A float image file holding NaN values is read as a blank page (GW-E2E-012, a known issue
in the model card's limitations).

## Installing

GLYPHARIS needs Python 3.11, at least 3.11.4 (`requires-python = ">=3.11.4,<3.12"` in
`pyproject.toml`). Once the package is published it will install with `pip install glypharis`. Until
then, install from a checkout of the repository, choosing the PyTorch build for your platform first
([`requirements/README.md`](../requirements/README.md)):

```
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install .
```

The core install reads pages with either backend. The extras `export` (writing bundles and ONNX
files) and `train` (training and reports) are not needed to read pages. The package holds no weights.

It provides three commands: `glypharis-ocr`, `glypharis-serve` and `glypharis-check`.
`glypharis.version_info()` (or `glypharis-ocr --version`) gives the versions of GLYPHARIS, Python and
the libraries that decide its results, for bug reports and result metadata.

## Getting the model

A model is a **bundle**: one directory with `manifest.json`, `detector.pt`, `recognizer.pt`,
`detector.onnx` and `recognizer.onnx`. The manifest holds the SHA-256 of every file, and loading checks
each one before anything is read ([ARCHITECTURE §8](ARCHITECTURE.md#8-model-bundle-and-runtimes)).

### Lookup: `from_pretrained` never downloads

`GlypharisOCR.from_pretrained(name_or_path)` never reaches the network. It takes the argument as a
directory first (relative to the working directory if it is not absolute), and otherwise looks for
`<folder>/<name>/manifest.json` in these folders, in this order (`glypharis.export.bundle.model_dirs`):

1. each entry of `GLYPHARIS_MODELS` (separated by `os.pathsep`: `;` on Windows, `:` elsewhere);
2. `~/.glypharis/models`;
3. `./models`, relative to the working directory.

A missing or damaged bundle raises `ModelLoadError`. Passing a bundle path avoids any doubt about which
folder wins.

### Download: `glypharis.download_model`

```python
from glypharis import download_model

# The installed bundle's directory, verified against the registry.
path = download_model("glypharis-v1")
```

Signature: `download_model(name="glypharis-v1", *, models_dir=None, offline=None, force=False, url=None)
-> Path`. Its rules (`glypharis/hub.py`), in order:

1. **Registered names only.** A name not in `glypharis.hub.REGISTRY` is refused. Only `glypharis-v1` is
   registered.
2. **An installed bundle is used, never overwritten.** If the lookup above already finds the name and
   its `manifest.json` has the pinned SHA-256, that path is returned with no network call. A bundle of
   that name with another manifest is refused and left as it is. `force=True` replaces only a bundle at
   `<models_dir>/<name>`, and moves the old one under `<models_dir>/.downloads/` rather than deleting
   it.
3. **Offline refuses before any network call**: `offline=True`, or `GLYPHARIS_OFFLINE` set to `1`,
   `true`, `yes` or `on`. `offline=False` overrides the variable.
4. **Not yet published refuses before any network call.** The registry entry is marked unpublished
   until the weights are released, so today the call fails with `model_download_error` and a message
   saying the model is not yet published, but only when the bundle is not already installed (rule 2)
   and `url=` is not given.
5. **Pinned hashes.** The registry pins the archive's size and SHA-256 and the SHA-256 of the bundle's
   `manifest.json`. The archive is downloaded (resuming a partial file), its size and hash are checked
   (a mismatching file is deleted), it is extracted into a staging folder, every file is checked against
   the manifest and the manifest against the pin, and only then is the folder renamed into place. A
   `download_record.json` is written inside the bundle directory, at
   `<models_dir>/<name>/download_record.json`, and is not listed in `manifest.json`
   (`glypharis.hub.read_record`). A lock file
   under `<models_dir>/.downloads/` keeps two processes from installing at once.

`models_dir` must be one of the lookup folders, so that `from_pretrained` finds the result; the
default is the first of them (the first `GLYPHARIS_MODELS` entry if set, else `~/.glypharis/models`).
`url=` replaces only the address, for tests and the release rehearsal: only `http` and `https` are
accepted, and the size and both hashes are still enforced. Every failure is a `ModelDownloadError`.

### The commands

`glypharis-ocr` and `glypharis-serve` fetch a registered name that no lookup folder holds, through
`download_model`, before loading it. `--no-download` turns that off, and so does `GLYPHARIS_OFFLINE`. A
path, or a name that is not registered, is never downloaded. `glypharis-serve` downloads nothing once it
is running.

### Until the weights are released

There is no public source for the bundle yet. An integration can be built and tested against a bundle
placed by hand in a lookup folder, or passed by path, and should be written so that a deployment can
run with `GLYPHARIS_OFFLINE=1` and the bundle installed in advance.

## Reading pages

```python
from glypharis import GlypharisOCR

# A name or a bundle path; the defaults are backend="torch" and threads=None.
ocr = GlypharisOCR.from_pretrained("glypharis-v1")
# A path, image file bytes, a PIL image or a numpy array.
result = ocr.predict("page.png")

# The non-empty lines joined with newlines, in reading order.
result.text
for line in result.lines:
    # bbox is x0, y0, x1, y1 in page pixels.
    line.text, line.confidence, line.det_score, line.bbox
# JSON-serialisable; result.to_json() gives the string.
result.to_dict()
```

`from_pretrained(name_or_path="glypharis-v1", backend="torch", threads=None, cache=True)`. With
`cache=True` a bundle already loaded in this process with the same backend and threads is returned
again, as the same object, so a setting changed on it (such as `ocr.max_pixels`) applies to every
caller holding it. `predict_batch(images)` reads pages one after another. `ocr.stats()` gives the pages
read, lines found, inputs refused and seconds so far.

The fields of `to_dict` are listed in [ARCHITECTURE §6](ARCHITECTURE.md#6-public-interface).

## Threads and concurrency

`threads` is passed to `from_pretrained` (or `--threads` to the commands).

- **PyTorch backend (the default).** A non-zero `threads` calls `torch.set_num_threads(threads)` when the
  bundle is loaded. PyTorch's thread count belongs to the **process**, not to one model: it applies to
  every model and every other use of PyTorch in that process, and a later load with a different value
  changes it for all of them. `threads=None` leaves PyTorch's own default. A cache hit does not set the
  count again.
- **ONNX backend.** `threads` sets the intra-op thread count of that bundle's ONNX Runtime sessions
  only; `None` leaves ONNX Runtime's default.

So choose one thread count per process and pass it at the first load. To read pages in parallel, use
separate processes, each loading its own model: `GlypharisOCR` makes no claim of being
safe to share between threads, and the server reads one request at a time for that reason
(`glypharis/server.py`). Keep the threads of all processes together within the machine's cores.

v1's latency figures come from one laptop (GW-EFF-007, `reports/efficiency/phase16/cost_table.md`;
[MODEL_CARD, hardware](../MODEL_CARD.md#hardware)); most CPU rows are at 10 threads, and there are
also default-thread ONNX and GPU rows. Speed on other machines was not measured.

## The HTTP server

`glypharis-serve` is an integration boundary for local use, not a production web service
(`glypharis/server.py`):

```
glypharis-serve [--model glypharis-v1] [--no-download] [--backend torch|onnx] [--threads N]
                [--host 127.0.0.1] [--port 8080] [--max-bytes N] [--max-pixels N]
curl --data-binary @page.png http://127.0.0.1:8080/ocr
```

- One process, one request at a time, the standard library's HTTP server, no authentication and no TLS.
  It listens on `127.0.0.1` unless told otherwise; put it behind a proper server or proxy before
  exposing it to a network.
- The model is loaded (and, for a missing registered name, downloaded) before the server starts.
- Each connection carries one request and is closed after the answer. A client silent for
  `SOCKET_TIMEOUT` seconds is dropped.
- `POST /ocr`: the request body is the image file itself (JPEG, PNG, TIFF, BMP or WebP; any content
  type) with a `Content-Length` header. The answer is `OCRResult.to_dict` as JSON, carrying
  `model_version` and `glypharis_version`.
- `GET /health`: `{"status": "ok", "model_version": ..., "glypharis_version": ...}`.
- Limits: the body may not exceed `--max-bytes` (default `DEFAULT_MAX_BYTES`), and the image may not
  have more pixels than `--max-pixels` (default: the library's).

| Status | `error` | When |
|---|---|---|
| 400 | `invalid_image` | The body is not a readable image |
| 411 | `length_required` | No usable `Content-Length` (a chunked body included) |
| 413 | `request_too_large` | The body is larger than `--max-bytes` |
| 413 | `image_too_large` | The image has more pixels than `--max-pixels` |
| 404 | `not_found` | Any path other than `/ocr` and `/health` |
| 405 | `method_not_allowed` | `GET /ocr`, or `PUT`, `PATCH` or `DELETE` |
| 500 | `internal_error` | Anything else; the traceback is logged, never sent |

Every error in this table has the body `{"error": <code>, "message": ...}`. A method the server has
no handler for (it defines `do_GET`, `do_POST`, `do_PUT`, `do_PATCH` and `do_DELETE`), and a malformed
request line, get the Python standard library's own 501 or 400 response, with an HTML body. The
commands exit with 2 when the model cannot be loaded.

## Errors

Every failure a caller can cause or meet at the public edge is a `GlypharisError` with a stable
`code`; anything else reaching the caller is a bug (`glypharis/errors.py`). The one exception: a
backend name other than `torch` or `onnx` raises a plain `ValueError` ("backend ...: onnx or torch",
`glypharis.export.bundle.load_bundle`). `exc.to_dict()` gives
`{"error": exc.code, "message": str(exc)}`.

| Class | `code` | Subclass of | Raised when |
|---|---|---|---|
| `GlypharisError` | `glypharis_error` | `Exception` | The base class; catch it to catch them all |
| `InvalidImageError` | `invalid_image` | `GlypharisError`, `ValueError` | The input is not an image GLYPHARIS can read: wrong type or shape, empty, or undecodable |
| `ImageTooLargeError` | `image_too_large` | `InvalidImageError` | More pixels than `max_pixels`, or a decompression bomb |
| `ModelLoadError` | `model_load_error` | `GlypharisError`, `RuntimeError` | A bundle that is missing, incomplete, damaged or of an unknown format |
| `ModelDownloadError` | `model_download_error` | `ModelLoadError` | A registered model that cannot be downloaded or installed: unknown name, offline, not yet published, a failed transfer, a size or hash that does not match its pin, or a folder in the way |
| `SampleError` (`glypharis.check`) | `sample_error` | `GlypharisError` | `glypharis-check` only: its expected output or a test registry file is missing or unreadable |

```python
from glypharis import GlypharisOCR
from glypharis.errors import InvalidImageError, ModelLoadError

try:
    ocr = GlypharisOCR.from_pretrained("glypharis-v1")
except ModelLoadError as exc:  # model_load_error
    raise SystemExit(exc.to_dict())

try:
    result = ocr.predict("page.png")
except InvalidImageError as exc:  # invalid_image or image_too_large
    print(exc.to_dict())
```

On the v1 bundle, on both backends, the hardening cases (bad files, wrong inputs, oversized images,
damaged bundles, fuzzed inputs) ended in a result or a `GlypharisError` and no other exception
(GW-E2E-012, `reports/hardening/GW-E2E-012.json`).

`glypharis-ocr` exits with 0 when every image was read, 1 when any image failed (each failure is
printed with its code and skipped) or when no image was found (`--input-dir` held no images and no
`--image` was given; it prints "no images found in ..."), and 2 for a usage error or a model that
cannot be loaded.

## Storing results

Store with every result the model version and the SHA-256 of the bundle's `manifest.json`.
`model_version` is `glypharis_v1` and stays so across package versions (Phase 23, D8); the manifest
hash pins the exact model files, because `manifest.json` holds the SHA-256 of each one. v1's manifest
SHA-256 is given in [MODEL_CARD, model version](../MODEL_CARD.md#model-version) (GW-EFF-006,
`reports/efficiency/phase16/v1_bundle.json`). With both, a stored result can later be matched to the
model that produced it, and results from another model are never mixed in unnoticed.

```python
from glypharis import GlypharisOCR
from glypharis.check import manifest_sha256
from glypharis.export.bundle import find_bundle

bundle = find_bundle("glypharis-v1")  # the directory from_pretrained would load
ocr = GlypharisOCR.from_pretrained(bundle)
model_sha = manifest_sha256(bundle)  # once per process

record = ocr.predict("page.png").to_dict()  # holds glypharis_version and model_version
record["manifest_sha256"] = model_sha
```

`glypharis.version_info()` can be stored beside them when the exact library versions matter.

## Checking an install

`glypharis-check` checks that the installed engine and model are the ones the registry pins, and that
they reproduce the expected output on one sample page shipped with the package
(`glypharis/sample/`):

```
glypharis-check [--model glypharis-v1] [--models-dir DIR] [--threads 4] [--download] [--offline]
```

1. The bundle is found as `glypharis-ocr --model NAME` finds it (or only in `--models-dir`), every
   file matches its `manifest.json`, and the manifest's SHA-256 equals the registry's pin.
2. If `download_model` installed it, its download record names the registry's archive size and hash.
   A bundle placed by hand has no record; step 1 covers its files.
3. The sample page is read with the PyTorch backend on the CPU (4 threads by default, the setting the
   expected output was read with), and the model version, the line texts and the joined text must
   equal the expected output exactly.

Nothing is downloaded unless `--download` is given, and `--offline` or `GLYPHARIS_OFFLINE` still
forbids the network. Exit code 0 when every check passes, 1 when one does not, 2 when the check cannot
run (for example, the model is not installed). Passing shows that the engine and weights are the
pinned ones; while the weights are unpublished it says so. It says nothing about accuracy: the sample
is one clean page the project rendered itself.

## Phones

Phone use is not available. v1 was not measured on a phone and keeps PyTorch as its default backend
([MODEL_CARD, not v1](../MODEL_CARD.md#not-v1)). The phone candidate, the all-ONNX build with fused
attention (`fused-single-line`, GW-EFF-013), is a new version and a documented result only: it met
H11's non-inferiority tests against v1 (GW-E2E-021, `reports/final_validation/candidate/h11.md`), and
only its networks were measured on two phones, without decoding, post-processing, cropping or reading
order (GW-EFF-015, [`docs/FINAL_VALIDATION.md`](FINAL_VALIDATION.md) section 4). Only `glypharis-v1`
ships (Phase 23, D7): there is no phone asset and no on-device reader.
