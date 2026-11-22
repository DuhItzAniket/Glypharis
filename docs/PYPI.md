# GLYPHARIS

A lightweight OCR engine for document images. It recovers readable text from scanned, photographed,
degraded, or image-only document pages and returns it as structured output: lines, boxes, confidence,
and timing.

GLYPHARIS is built as a research project. Every model is trained and evaluated on a single laptop GPU,
every experiment is registered before it runs, and every reported number traces back to a generated
report in the [repository](https://github.com/DuhItzAniket/Glypharis/tree/v1.0).

v1.0 is the release of v1 (GW-REC-034 with GW-DET-001): the code under Apache-2.0, published here
as `glypharis` 1.0.0, and the v1 weights as the asset `glypharis-v1.tar.gz` of the
[GitHub release v1.0](https://github.com/DuhItzAniket/Glypharis/releases/tag/v1.0), which
`glypharis.download_model` fetches and checks against pinned hashes.

> **The weights are a separate download.** This package is the code only: it holds no weights.
> `glypharis.download_model("glypharis-v1")` fetches them once; they carry their own terms (see
> Licence below).

## Install

```
pip install glypharis
```

GLYPHARIS needs Python 3.11 (3.11.4 or later). pip installs the default PyTorch build for your
platform; for a particular build (CPU only, or a CUDA version), install `torch` and `torchvision`
from the [PyTorch index](https://pytorch.org/get-started/locally/) first. The project's own pins are in
[requirements/README.md](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/requirements/README.md).

The core install reads pages with either backend (PyTorch or ONNX Runtime). Two extras add the rest:

| Extra | Adds | For |
|---|---|---|
| `glypharis[export]` | `onnx` | writing bundles and ONNX files (`write_bundle`), quantisation |
| `glypharis[train]` | `matplotlib`, `fonttools`, `pypdfium2`, `psutil`, `nvidia-ml-py` | training, data preparation and reports, which also need the repository's scripts and configs |

## Models

A model bundle is a directory with `manifest.json`, two release files and two ONNX files.
`GlypharisOCR.from_pretrained` takes a bundle's path or its name, looks names up in
`$GLYPHARIS_MODELS`, `~/.glypharis/models` and `./models`, and never reaches the network.

`glypharis.download_model(name)` fetches a registered model's release archive, checks its size and
SHA-256 and the bundle's manifest against the pins in `glypharis.hub`, and installs it where
`from_pretrained` finds it (the first `GLYPHARIS_MODELS` entry, else `~/.glypharis/models`).
`glypharis-ocr` and `glypharis-serve` do the same for a registered name that is not installed, unless
`--no-download` is given or `GLYPHARIS_OFFLINE` is set. Only `glypharis-v1` is registered; its
archive is the asset `glypharis-v1.tar.gz` of the
[GitHub release v1.0](https://github.com/DuhItzAniket/Glypharis/releases/tag/v1.0), beside
`glypharis-v1.tar.gz.sha256`. `glypharis-check` checks that an installed bundle is the released one
and reads a sample page shipped with the package; it says nothing about accuracy.

## Using it

```python
import glypharis
from glypharis import GlypharisOCR

glypharis.download_model("glypharis-v1")  # once: fetches and checks the release asset
ocr = GlypharisOCR.from_pretrained("glypharis-v1")  # a model bundle by name or path
result = ocr.predict("page.png")  # path, image file bytes, PIL image or numpy array

print(result.text)  # lines in reading order
for line in result.lines:
    print(line.text, line.confidence, line.bbox)
print(result.to_json())  # the same as structured JSON
```

From the command line:

```
glypharis-ocr --image page.png --output results
glypharis-ocr --input-dir scans/ --recursive --format json txt
```

Each image gives `results/<name>.json` and `results/<name>.txt`. The default backend is PyTorch on the
CPU; `--backend onnx` (or `backend="onnx"`) uses ONNX Runtime, which matches PyTorch within the
project's export tolerances but is slower on the CPU
([model card](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md)).

Over HTTP (standard library only, one request at a time, for local use without authentication or
TLS):

```
glypharis-serve --model glypharis-v1 --port 8080
curl --data-binary @page.png http://127.0.0.1:8080/ocr
```

`POST /ocr` takes the image file as the request body and answers with the same JSON; `GET /health`
reports the versions.

## How it works

```
image ─▶ preprocessing ─▶ GLYPHARIS-Det (line detector) ─▶ crop + rectify
      ─▶ GLYPHARIS-Rec (CTC line recognizer) ─▶ reading order ─▶ structured result
```

- **GLYPHARIS-Det**: DBNet-style text-line detector with a MobileNetV3 backbone.
- **GLYPHARIS-Rec**: SVTR-style CTC line recognizer in v1 (a CNN + BiLSTM + CTC recognizer is the
  baseline).
- Stands alone: any application can call it with an image. It contains no application-specific code.

What v1 is, how it measured and where it fails are in the
[model card](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md), with every number
traced to its experiment and report.

## Licence

The code is licensed under Apache-2.0
([LICENSE](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/LICENSE)).
[NOTICE](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/NOTICE) lists the material in the
repository that the licence does not cover.

The weights are not part of this package. They are distributed under CC BY-SA 4.0 with attribution
to HierText and Open Images
([ADR-0014](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/docs/decisions/ADR-0014-weights-and-backbone-terms.md)),
and their archive carries `LICENSE-WEIGHTS` and a `NOTICE.md` with the attribution. The detector was
initialised from ImageNet weights, a disclosed exception to the project's commercial-use rule
([ADR-0009](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/docs/decisions/ADR-0009-no-synthtext.md)),
and the weights are not described as cleared for commercial use. The owner decided on 2026-11-22 to
publish them under these terms without the qualified advice ADR-0014 had asked for; this is the
project's chosen handling, not a legal conclusion. The terms of each dataset are in
[DATASETS.md](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/DATASETS.md).

## Citing

Cite with [CITATION.cff](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/CITATION.cff); a Zenodo
DOI is added after the deposit.

## Links

- [Repository at the tag `v1.0`](https://github.com/DuhItzAniket/Glypharis/tree/v1.0)
- [Release v1.0, with the weights](https://github.com/DuhItzAniket/Glypharis/releases/tag/v1.0)
- [Release notes](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/docs/RELEASE_NOTES_v1.0.md)
- [Model card](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md)
- [Changelog](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/CHANGELOG.md)
- [Issues](https://github.com/DuhItzAniket/Glypharis/issues)
