---
license: cc-by-sa-4.0
language:
- en
tags:
- ocr
- text-detection
- text-recognition
- document-images
- onnx
- pytorch
pipeline_tag: image-to-text
library_name: glypharis
---

# GLYPHARIS v1

GLYPHARIS v1 (model version `glypharis_v1`) is a line-based OCR model for English printed text in
document images. A text-line detector, GW-DET-001, finds the lines on a page; each line is cropped
and read by a CTC line recognizer, GW-REC-034; a reading-order step sorts the lines. The result is
text, and lines with box, confidence and timing. The code that runs it is the Python package
`glypharis` 1.0.0, under Apache-2.0.

**This repository is a mirror.** It holds the files of `glypharis-v1.tar.gz`, the model asset of
the GitHub release [`v1.0`](https://github.com/DuhItzAniket/Glypharis/releases/tag/v1.0), byte for
byte, with this card added as `README.md`. The GitHub release asset is the primary copy:
`glypharis.download_model` fetches it from there and never uses this mirror. If the two ever
differ, the release asset, checked against the pins in `glypharis/hub.py`, is the reference.

The source, the full model card and every report behind the numbers below are in the repository at
the tag: <https://github.com/DuhItzAniket/Glypharis/tree/v1.0>.

Only `glypharis-v1` is released. The phone candidate evaluated before the release is a documented
result, not a released model, and nothing here is for it.

## Files

| File | What it is |
|---|---|
| `manifest.json` | The bundle manifest (format `glypharis-bundle-1`): the SHA-256 of each weight file, and the detector's and recognizer's settings |
| `detector.pt` | Text-line detector GW-DET-001, PyTorch |
| `detector.onnx` | The same detector, ONNX (opset 17) |
| `recognizer.pt` | Line recognizer GW-REC-034, PyTorch |
| `recognizer.onnx` | The same recognizer, ONNX (opset 17) |
| `LICENSE-WEIGHTS` | The weights' licence, CC BY-SA 4.0 |
| `NOTICE.md` | The attribution to HierText and Open Images, the changes made, the ImageNet exception and the absence of warranty |
| `model_info.json` | What the weights were trained on, the frozen configuration, the experiment IDs, the code tag and commit, and a link to the model card at the tag |

These eight files are the members of the archive's one folder, `glypharis-v1/`. `README.md` (this
card) is not in the archive, and neither is the `.gitattributes` file that Hugging Face adds to
every repository.

## Verifying the files

The SHA-256 of `manifest.json` is pinned in the package (`glypharis/hub.py`) and in the model card:

```
875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037
```

`manifest.json` holds the SHA-256 of each of the four weight files, and the bundle loader checks
every one of them when it loads the model. The hashes of all eight files, from `manifest.json` and
from the second pack of the archive recorded in
[`reports/release/pack_check.json`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/reports/release/pack_check.json),
can be checked in the folder that holds them:

```sh
sha256sum -c <<'EOF'
875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037  manifest.json
caafdce9402de07a775102558eeffa6f48d4f54cd370ac63c09ab0be2a929c99  detector.pt
378c1feda480c1bb1dde838dd4a5f4579a904710a7f7b222d978c583b5bddcb7  detector.onnx
785409f75f1d31e4933a42ddcae8c9ecb84960d975083f3f81ebc47d6070f9ca  recognizer.pt
930329b4bf635a935975823d2004b47371d9123b2504cfe590d52f44db84f438  recognizer.onnx
81ceaf22681fe1bd55c5e85b4a72243d1652b631b38769e42947f988413fe79d  LICENSE-WEIGHTS
8bc9869cb7e5f3b7dd45416103a93fa2a362ba90151f439ba9d1363d26317d65  NOTICE.md
c41f5a45f0e7747f5ff604ea45ea8097d520c05bd8740c83364a16c84ae03508  model_info.json
EOF
```

(On macOS, `shasum -a 256 -c` instead of `sha256sum -c`.) To compare with the release asset
itself, download `glypharis-v1.tar.gz` and `glypharis-v1.tar.gz.sha256` from the GitHub release
`v1.0`, check the archive with `sha256sum -c glypharis-v1.tar.gz.sha256` (82,975,077 bytes, SHA-256
`3b90c1c74147747d7474a86b550e198f0df83d447961bd6aec88d4c24846541d`, the pin in `glypharis/hub.py`),
extract it, and compare its `glypharis-v1/` folder with these files.

## Using it

The package needs Python 3.11 (3.11.4 or later, below 3.12). The usual way does not involve this
mirror: `download_model` fetches the GitHub release asset once, checks its size and SHA-256 and the
manifest hash against the pins, and installs it where `from_pretrained` finds it.

```
pip install glypharis==1.0.0
```

```python
import glypharis
from glypharis import GlypharisOCR

glypharis.download_model("glypharis-v1")  # once: fetches and checks the release asset
ocr = GlypharisOCR.from_pretrained("glypharis-v1")  # a model bundle by name or path
result = ocr.predict("page.png")  # path, image file bytes, PIL image or numpy array

print(result.text)  # lines in reading order
for line in result.lines:
    print(line.text, line.confidence, line.bbox)
```

From this mirror, download the files into a folder named `glypharis-v1` with the Hugging Face CLI
(`hf`, from the `huggingface_hub` package) and load that folder. `from_pretrained` never downloads
anything: it takes a bundle folder (a directory with `manifest.json`) by path, or by name from
`$GLYPHARIS_MODELS`, `~/.glypharis/models` and `./models`.

```
hf download DuhItzAniket/glypharis-v1 --local-dir models/glypharis-v1
glypharis-check --models-dir models
```

```python
from glypharis import GlypharisOCR

ocr = GlypharisOCR.from_pretrained("models/glypharis-v1")
```

`glypharis-check` checks every file against `manifest.json`, the SHA-256 of `manifest.json` against
the package's pin, and the output on one sample page shipped with the package; it notes that a folder
placed by hand has no download record, which is expected here. It shows that the files are the
released ones, and says nothing about accuracy. Files that `manifest.json` does not list, such as this
card and the `.cache/` folder `hf download` creates, are ignored by the loader. A folder with the
pinned manifest under one of the lookup folders is also what `glypharis-ocr --model glypharis-v1` and
`glypharis.download_model("glypharis-v1")` use, with no network call.

The default backend is PyTorch on the CPU, the configuration that was evaluated;
`backend="onnx"` uses ONNX Runtime. The integration guide is
[`docs/INTEGRATION.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/docs/INTEGRATION.md).

## Results

Copied from the summary of the
[model card](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#summary), where each
value appears again with its context. CER is the character error rate in percent, and lower is
better. Square brackets hold the interval. GDC is this project's test set of printed pages captured
with a phone, FUNSD is a set of forms, OmniDocBench a set of document pages, and HierText a public
dataset of mostly scene photographs. Terms are defined under
[how to read the numbers](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#how-to-read-the-numbers).

Source: GW-REC-039 [`reports/external/recognition.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/reports/external/recognition.md);
GW-E2E-007 [`reports/external/end_to_end.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/reports/external/end_to_end.md);
GW-EFF-007 [`reports/efficiency/phase16/cost_table.md`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/reports/efficiency/phase16/cost_table.md).

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

Page CER also counts detection failures, which line CER from ground-truth crops does not. The page
time was measured on one laptop CPU
([hardware](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#hardware)). HierText is
released under CC BY-SA 4.0; its images come from Open Images, which lists them as CC BY 2.0. The
HierText rows of this table are derived from HierText and shared alike under CC BY-SA 4.0.

## Limitations

The full list, with the evidence for each point, is in the model card:
[limitations](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#limitations),
[not intended use](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#not-intended-use)
and [known failure modes](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#known-failure-modes).
In short:

- It reads English printed text only. Handwriting, other scripts, vertical text, tables, formulas
  and layout analysis are out of scope.
- Curved pages, such as an open book, and pages photographed sideways are not handled well; there is
  no page-orientation step.
- The page error rates above show how many errors to expect. It suits output that is searched,
  indexed or read by a person who can tolerate errors, not unreviewed transcription where errors
  matter.
- Some baselines do better on some sets: PP-OCRv6 medium on GDC crops and IIIT5K words, and
  Tesseract 5 on OmniDocBench order-aware page CER, each with a paired interval that excludes zero.
- It was measured on one laptop, with the PyTorch backend on the CPU, and every line and page
  latency target the project set was missed. v1 was not measured on a phone.
- No set of old scans was evaluated.
- The confidence is not a probability.
- One recognizer seed and one detector seed were frozen.
- A re-run of the frozen v1 gave the same predictions as the original evaluation in every test
  file; the strict reproduction verdict failed only on official OmniDocBench scores, through that
  evaluator's per-page time limit, and the owner accepted this as a disclosed limitation
  (GW-E2E-018).
- The optional HTTP server of the package is for local use and has no authentication or TLS.

## Terms of use

Copyright 2026 DuhItzAniket. The four weight files (`detector.pt`, `detector.onnx`,
`recognizer.pt`, `recognizer.onnx`) are licensed under the Creative Commons
Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0,
<https://creativecommons.org/licenses/by-sa/4.0/legalcode>). `LICENSE-WEIGHTS` and `NOTICE.md` are
the notices that come with them; read them together. In substance:

- **Attribution: HierText and Open Images.** Both models were trained on HierText (Long et al.,
  "Towards End-to-End Unified Scene Text Detection and Layout Analysis", CVPR 2022,
  <https://github.com/google-research-datasets/hiertext>). HierText is released under CC BY-SA 4.0;
  its images come from Open Images (<https://storage.googleapis.com/openimages/web/index.html>),
  which lists them as CC BY 2.0 and gives no warranty about the licence of any one image. Author and
  source are not recorded per image. The weights are shared under CC BY-SA 4.0, the licence HierText
  is released under, with this attribution; `NOTICE.md` is the attribution in full.
- **Changes made.** The weights are not part of HierText or Open Images. The project trained the
  text-line detector (GW-DET-001) on HierText training pages and the line recognizer (GW-REC-034)
  on line crops cut from HierText training images, resized and converted to grey scale, and exported
  both as PyTorch and ONNX files, frozen in `configs/final/glypharis_v1.yaml`. No HierText or Open
  Images image, crop or annotation is in these files.
- **Exception: the detector was initialised from ImageNet-1K weights.** The detector's backbone was
  initialised from torchvision's MobileNetV3-Large ImageNet-1K weights, and the ImageNet terms do
  not clearly allow commercial use. The project's rule (ADR-0009) is that every source that can
  influence the weights of a delivered model must allow commercial use; this detector is kept as a
  disclosed exception to that rule (ADR-0014). The recognizer was trained from random
  initialisation. The v1 weights are therefore never described as cleared for commercial use.
- **The owner's decision.** The owner decided on 2026-11-22 to publish the weights under these
  terms without the qualified advice ADR-0014 had asked for. These terms are the project's chosen
  handling, not a legal conclusion that the weights or their commercial use are cleared, and not
  legal advice. The decision is
  [ADR-0014](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/docs/decisions/ADR-0014-weights-and-backbone-terms.md).
- **No warranty.** The weights are provided as is, without warranties or conditions of any kind,
  express or implied, including fitness for a particular purpose and the accuracy of their output,
  as section 5 of CC BY-SA 4.0 sets out. The licences of HierText and Open Images likewise give no
  warranty.
- **The code.** The GLYPHARIS source code is licensed separately under the Apache License, Version
  2.0. That licence does not cover these weights.

## Citation

Cite GLYPHARIS with
[`CITATION.cff`](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/CITATION.cff) in the
repository. A Zenodo DOI is added to the repository's `CITATION.cff` and README after the deposit.
The methods and data behind the model are cited under
[citation information](https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md#citation-information)
in the model card.
