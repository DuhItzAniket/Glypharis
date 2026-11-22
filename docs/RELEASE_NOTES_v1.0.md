# GLYPHARIS v1.0 release notes

## What v1.0 is

GLYPHARIS is a lightweight OCR engine for document images. It recovers readable text from scanned,
photographed, degraded, or image-only document pages and returns it as structured output: lines,
boxes, confidence, and timing.

v1.0 is the release of v1 (GW-REC-034 with GW-DET-001): the code under Apache-2.0, published on PyPI
as `glypharis` 1.0.0, and the v1 weights as the asset `glypharis-v1.tar.gz` of the GitHub release
`v1.0`, which `glypharis.download_model` fetches and checks against pinned hashes. v1 is:

- the recognizer **GW-REC-034**, an SVTR-style CTC line recognizer trained from random
  initialisation on HierText training lines;
- the detector **GW-DET-001**, a DBNet-style line detector with a MobileNetV3-Large backbone
  initialised from ImageNet-1K weights, trained on HierText training pages.

The configuration was chosen on validation data only and frozen in
[`configs/final/glypharis_v1.yaml`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/configs/final/glypharis_v1.yaml)
before the frozen model read any test set. The model version is `glypharis_v1`, and the bundle is
`glypharis-v1` with manifest SHA-256
`875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`. The package version is `1.0.0`
and the tag is `v1.0`.

## What ships

| Part | Where | Terms |
|---|---|---|
| The `glypharis` package 1.0.0 (code): the sdist and the wheel, with `SHA256SUMS` | this release, and PyPI | Apache-2.0 ([`LICENSE`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/LICENSE), [`NOTICE`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/NOTICE)) |
| The v1 weights: `glypharis-v1.tar.gz`, with `glypharis-v1.tar.gz.sha256` | this release | CC BY-SA 4.0 (below) |
| Code and documentation archive | Zenodo, without `reports/` and the evaluation manifests under `data/manifests` | as the repository |

The archive `glypharis-v1.tar.gz` has one top folder, `glypharis-v1/`, holding the five bundle files
byte for byte (`manifest.json`, `detector.pt`, `recognizer.pt`, `detector.onnx`,
`recognizer.onnx`), `LICENSE-WEIGHTS` (CC BY-SA 4.0), `NOTICE.md` (the attribution, the changes made
and the disclosure below) and `model_info.json` (what the weights were trained on and how to trace
them). The weights are not part of the code packages, and the Apache License does not cover them.

The phone candidate (the all-ONNX build with fused attention, GW-EFF-013 and GW-EFF-014) is a new
version, not v1, and is not released: no phone bundle or phone asset ships with v1.0.

## Terms of the weights

The weights are distributed under CC BY-SA 4.0 with attribution to HierText and Open Images
([ADR-0014](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/docs/decisions/ADR-0014-weights-and-backbone-terms.md)).
HierText is released under CC BY-SA 4.0; its images come from Open Images, which lists them as
CC BY 2.0.

The detector was initialised from ImageNet weights, a disclosed exception to the project's
commercial-use rule
([ADR-0009](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/docs/decisions/ADR-0009-no-synthtext.md)):
GW-DET-001's backbone was initialised from ImageNet-1K weights, and the ImageNet terms do not clearly
allow commercial use. The weights are not described as cleared for commercial use. The owner decided
on 2026-11-22 to publish them under these terms without the qualified advice ADR-0014 had asked for;
this is the project's chosen handling, not a legal conclusion. The terms of each dataset are in
[`DATASETS.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/DATASETS.md).

## Install and fetch the model

The package needs Python 3.11 (`>=3.11.4,<3.12`):

```
pip install glypharis==1.0.0
```

pip installs the default PyTorch build for the platform; for a particular build (CPU only, or a CUDA
version), install `torch` and `torchvision` from the PyTorch index first. The model is fetched only
when asked for, and only for a registered name:

```python
import glypharis
from glypharis import GlypharisOCR

bundle = glypharis.download_model("glypharis-v1")  # checks the archive and manifest hashes
ocr = GlypharisOCR.from_pretrained("glypharis-v1")
print(ocr.predict("page.png").text)
```

`download_model` fetches `glypharis-v1.tar.gz` from this release and installs the bundle in the
first `GLYPHARIS_MODELS` entry, else `~/.glypharis/models`. `glypharis-ocr` and `glypharis-serve`
fetch a registered model that is missing unless `--no-download` or offline mode
(`GLYPHARIS_OFFLINE`) is set. `from_pretrained` itself never reaches the network; it finds bundles in
`$GLYPHARIS_MODELS`, `~/.glypharis/models` and `./models`. The
[integration guide](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/docs/INTEGRATION.md) covers
threads, the server, errors and offline use.

## Verify the downloads

`download_model` checks the archive's size and SHA-256 and the bundle's manifest hash before it puts
the bundle in place, and loading the bundle (`GlypharisOCR.from_pretrained`) checks each file against
the manifest and refuses one that does not match. To check by hand:

1. **The archive.** `glypharis-v1.tar.gz` is 82,975,077 bytes with SHA-256
   `3b90c1c74147747d7474a86b550e198f0df83d447961bd6aec88d4c24846541d` (GW-E2E-022,
   `reports/package/GW-E2E-022_fetch.json`). With both files from this release in one folder:

   ```
   sha256sum -c glypharis-v1.tar.gz.sha256
   ```

2. **The bundle.** A bundle is identified by the SHA-256 of its `manifest.json`, and the manifest
   lists the SHA-256 of every other file in it. After extracting the archive (or after
   `download_model`), the manifest hash must be
   `875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`:

   ```
   sha256sum glypharis-v1/manifest.json
   ```

3. **The code files.** With the sdist, the wheel and `SHA256SUMS` from this release in one folder:
   `sha256sum -c SHA256SUMS`.

`glypharis-check` then checks the installed bundle against the registry, checks the download record
when `download_model` installed it, and reads a sample page shipped with the package. Passing shows
that the installed engine and weights are the released ones; it says nothing about accuracy.

## Headline results

Copied from the summary table of
[`MODEL_CARD.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#summary), where
each value has its interval method and context. CER is the character error rate in percent, lower is
better; square brackets hold the 95% bootstrap interval.

Source: GW-REC-039
[`reports/external/recognition.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/reports/external/recognition.md);
GW-E2E-007
[`reports/external/end_to_end.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/reports/external/end_to_end.md);
GW-EFF-007
[`reports/efficiency/phase16/cost_table.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/reports/efficiency/phase16/cost_table.md).

| Measure | Value | Set and unit | Experiment |
|---|---|---|---|
| GDC line CER from ground-truth crops | 4.17 [3.72, 4.67] % | phone captures of printed pages, CI by line | GW-REC-039 |
| GDC line detection F1 @0.5 | 80.80 [72.63, 88.07] % | phone captures of printed pages, CI by page | GW-E2E-007 |
| GDC page CER, detector and recognizer together | 22.35 [15.87, 30.11] % | phone captures of printed pages, bag of lines, CI by page | GW-E2E-007 |
| FUNSD page CER, detector and recognizer together | 32.05 [24.28, 39.85] % | forms, bag of lines, CI by page | GW-E2E-007 |
| OmniDocBench page CER, order-aware | 27.28 [23.95, 30.57] % | English document pages, this project's metric, CI by page | GW-E2E-007 |
| HierText test line CER from ground-truth crops | 6.03 [5.89, 6.18] % | mostly scene photographs, CI by line | GW-REC-039 |
| HierText test line detection F1 @0.5 | 79.13 [77.98, 80.26] % | mostly scene photographs, CI by page | GW-E2E-007 |
| HierText test page CER, detector and recognizer together | 25.16 [23.44, 26.79] % | mostly scene photographs, bag of lines, CI by page | GW-E2E-007 |
| Mean page time, PyTorch on the CPU | 2,566 [2,327, 2,829] ms | HierText validation pages, CI by page | GW-EFF-007 |
| Parameters | 11,116,044 | detector and recognizer | GW-EFF-007 |

HierText-derived tables are shared alike under CC BY-SA 4.0, with the attribution in
[`NOTICE`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/NOTICE).

Phase 22 re-ran the frozen v1 on every test set and got Phase 14's predictions in every test file.
The strict reproduction verdict failed only on official OmniDocBench scores, through that
evaluator's per-page time limit, and the owner accepted this as a disclosed limitation (GW-E2E-018,
[`reports/final_validation/v1_reproduction.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/reports/final_validation/v1_reproduction.md)).

## Limitations

The full list is in the model card:
[limitations](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#limitations) and
[known failure modes](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#known-failure-modes).
In short:

- Curved pages and pages photographed sideways are not handled well.
- Some baselines do better on some sets; see the model card's
  [comparison with baselines](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#comparison-with-baselines).
- Every line and page latency target of
  [ADR-0011](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/docs/decisions/ADR-0011-cpu-latency-target.md)
  was missed on the laptop CPU (GW-EFF-007).
- Phones are the target hardware, but v1 itself was not measured on a phone, and the phone candidate
  is not released.
- No set of old scans was evaluated.
- One recognizer seed and one detector seed are frozen.
- It is meant for English printed text in flat, upright page images, one image at a time
  ([intended use](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#intended-use)).
- Only Python 3.11 is supported, and pip may install a PyTorch build other than the one v1 was
  evaluated with.

## Citing

Cite with [`CITATION.cff`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/CITATION.cff); a
Zenodo DOI is added after the deposit. The methods and datasets behind the model are cited in the
model card's
[citation information](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#citation-information).
