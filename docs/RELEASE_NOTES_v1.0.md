# GLYPHARIS v1.0 release notes

**Draft — not released.** Nothing described here has been published. These notes are prepared in
Phase 23 (`docs/PHASE_STATUS.md`) and are rewritten once the owner has the qualified advice and
decides what v1.0 distributes (Phase 23, items 14 to 16). Until then no weights are distributed, and
the install and download steps below do not work yet.

## What v1.0 is

GLYPHARIS is a lightweight OCR engine for document images. It recovers readable text from scanned,
photographed, degraded, or image-only document pages and returns it as structured output: lines,
boxes, confidence, and timing.

v1.0 is planned as the package release of the frozen v1 model:

- the recognizer **GW-REC-034**, an SVTR-style CTC line recognizer trained from random
  initialisation on HierText training lines;
- the detector **GW-DET-001**, a DBNet-style line detector with a MobileNetV3-Large backbone
  initialised from ImageNet-1K weights, trained on HierText training pages.

The configuration was chosen on validation data only and frozen in
[`configs/final/glypharis_v1.yaml`](../configs/final/glypharis_v1.yaml) before the frozen model read
any test set. The model version is `glypharis_v1`, and the bundle is `glypharis-v1` with manifest
SHA-256 `875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`. The package version
will be `1.0.0`, and the tag `v1.0` (not set yet).

## What would ship

| Part | How it would ship | Terms |
|---|---|---|
| The `glypharis` package (code) | wheel and sdist on the GitHub release and on PyPI, each channel on the owner's go | Apache-2.0 ([`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE)) |
| The v1 weights, `glypharis-v1.tar.gz` | a separate asset of the GitHub release, uploaded only after the qualified advice and the owner's go | intended terms below |
| Documentation and code archive | Zenodo, without `reports/` and the evaluation manifests, on a separate go | as the repository |

The weights are not part of the code packages, and the Apache License does not cover them. Their
intended terms are CC BY-SA 4.0 with attribution to HierText and Open Images, as the owner decided
on 2026-11-18 ([ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md)). These terms take
effect only if the weights are released. HierText is released under CC BY-SA 4.0; its images come
from Open Images, which lists them as CC BY 2.0.

GW-DET-001 stays v1's detector as a disclosed exception to the commercial-use rule of
[ADR-0009](decisions/ADR-0009-no-synthtext.md): its backbone was initialised from ImageNet-1K
weights, and the ImageNet terms do not clearly allow commercial use. The v1 weights are therefore
not described as cleared for commercial use. This is the project's chosen handling, not a legal
conclusion. The weights archive is to carry `LICENSE-WEIGHTS` (CC BY-SA 4.0) and a `NOTICE.md` with
the attribution, the changes made and this disclosure.

The phone candidate (the all-ONNX build with fused attention, GW-EFF-013 and GW-EFF-014) is a new
version, not v1, and is not released: no phone bundle or phone asset ships with v1.0.

## Headline results

Copied from the summary table of [`MODEL_CARD.md`](../MODEL_CARD.md#summary), where each value has
its interval method and context. CER is the character error rate in percent, lower is better;
square brackets hold the 95% bootstrap interval.

Source: GW-REC-039 [`reports/external/recognition.md`](../reports/external/recognition.md);
GW-E2E-007 [`reports/external/end_to_end.md`](../reports/external/end_to_end.md); GW-EFF-007
[`reports/efficiency/phase16/cost_table.md`](../reports/efficiency/phase16/cost_table.md).

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
[`NOTICE`](../NOTICE).

Phase 22 re-ran the frozen v1 on every test set and got Phase 14's predictions in every test file.
The strict reproduction verdict failed only on official OmniDocBench scores, through that
evaluator's per-page time limit, and the owner accepted this as a disclosed limitation (GW-E2E-018,
[`reports/final_validation/v1_reproduction.md`](../reports/final_validation/v1_reproduction.md)).

## Limitations

The full list is in the model card: [limitations](../MODEL_CARD.md#limitations) and
[known failure modes](../MODEL_CARD.md#known-failure-modes). In short:

- Curved pages and pages photographed sideways are not handled well.
- Some baselines do better on some sets; see the model card's
  [comparison with baselines](../MODEL_CARD.md#comparison-with-baselines).
- Every line and page latency target of [ADR-0011](decisions/ADR-0011-cpu-latency-target.md) was
  missed on the laptop CPU (GW-EFF-007).
- Phones are the target hardware, but v1 itself was not measured on a phone.
- No set of old scans was evaluated.
- One recognizer seed and one detector seed are frozen.
- It is meant for English printed text in flat, upright page images, one image at a time
  ([intended use](../MODEL_CARD.md#intended-use)).

## Install and fetch the model (once published)

The package needs Python 3.11 (`>=3.11.4,<3.12`, Phase 23 D13). After the PyPI release:

```
pip install glypharis==1.0.0
```

The model is fetched only when asked for, and only for a registered name (Phase 23, D6):

```python
import glypharis
from glypharis import GlypharisOCR

bundle = glypharis.download_model("glypharis-v1")  # checks the archive and manifest hashes
ocr = GlypharisOCR.from_pretrained("glypharis-v1")
print(ocr.predict("page.png").text)
```

`glypharis-ocr` and `glypharis-serve` fetch a registered model that is missing unless
`--no-download` or offline mode (`GLYPHARIS_OFFLINE`) is set. `from_pretrained` itself never reaches
the network; it finds bundles in `$GLYPHARIS_MODELS`, `~/.glypharis/models` and `./models`.

## Verify the bundle

A bundle is identified by the SHA-256 of its `manifest.json`, and the manifest lists the SHA-256 of
every other file in it. For v1.0 the manifest hash must be:

```
875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037
```

To check a fetched or hand-extracted bundle:

```
sha256sum glypharis-v1/manifest.json
```

Loading the bundle (`GlypharisOCR.from_pretrained`) then checks each file against the manifest and
refuses one that does not match. `download_model` checks the archive's size and SHA-256 and the
manifest hash before it puts the bundle in place. The archive `glypharis-v1.tar.gz` is 82,975,077 bytes with
SHA-256 `3b90c1c74147747d7474a86b550e198f0df83d447961bd6aec88d4c24846541d` (GW-E2E-022,
`reports/package/GW-E2E-022_fetch.json`), and that hash is published beside it as
`glypharis-v1.tar.gz.sha256`. If the qualified advice changes the weight notices, the archive is
packed again and this pin changes (Phase 23, item 14).

## Citing

See [`CITATION.cff`](../CITATION.cff). The methods and datasets behind the model are cited in the
model card's [citation information](../MODEL_CARD.md#citation-information).
