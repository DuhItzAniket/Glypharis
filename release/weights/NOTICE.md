# GLYPHARIS v1 model weights: notice

This notice comes with the GLYPHARIS v1 model archive, `glypharis-v1.tar.gz`. It covers the four
weight files listed in `manifest.json` (`detector.pt`, `detector.onnx`, `recognizer.pt`,
`recognizer.onnx`). `LICENSE-WEIGHTS` gives their licence: CC BY-SA 4.0
(<https://creativecommons.org/licenses/by-sa/4.0/legalcode>). `model_info.json` records what the
weights were trained on and how to trace them.

Copyright 2026 DuhItzAniket.

Source: <https://github.com/DuhItzAniket/Glypharis>, tag `v1.0`. The model card is
<https://github.com/DuhItzAniket/Glypharis/blob/v1.0/MODEL_CARD.md>, and the decision on these
terms is `docs/decisions/ADR-0014-weights-and-backbone-terms.md` at the same tag.

## Attribution: HierText and Open Images

Both models were trained on HierText (Long et al., "Towards End-to-End Unified Scene Text
Detection and Layout Analysis", CVPR 2022, <https://github.com/google-research-datasets/hiertext>).
HierText is released under CC BY-SA 4.0 (<https://creativecommons.org/licenses/by-sa/4.0/>). Its
images come from Open Images (<https://storage.googleapis.com/openimages/web/index.html>), which
lists them as CC BY 2.0 (<https://creativecommons.org/licenses/by/2.0/>) and gives no warranty
about the licence of any one image. Author and source are not recorded per image.

The weights are shared under CC BY-SA 4.0, the licence HierText is released under, with this
attribution to HierText and Open Images. This notice is that attribution in full.

## Changes made

The weights are not part of HierText or Open Images. The project made them from that data:

- it trained a text-line detector (experiment GW-DET-001) on HierText training pages and a
  text-line recognizer (experiment GW-REC-034) on line crops cut from HierText training images,
  resized and converted to grey scale;
- it exported both models as PyTorch (`.pt`) and ONNX (`.onnx`) files, frozen in the configuration
  `configs/final/glypharis_v1.yaml`.

No HierText or Open Images image, crop or annotation is in this archive.

## Exception: the detector was initialised from ImageNet-1K weights

The detector's backbone was initialised from torchvision's MobileNetV3-Large ImageNet-1K weights,
and the ImageNet terms do not clearly allow commercial use. The project's rule (ADR-0009) is that
every source that can influence the weights of a delivered model must allow commercial use; this
detector is kept as a disclosed exception to that rule (ADR-0014). The recognizer was trained from
random initialisation.

The v1 weights are therefore never described as cleared for commercial use. These terms are the
project's chosen handling, not a legal conclusion that the weights or their commercial use are
cleared, and not legal advice.

## No warranty

The weights are provided as is, without warranties or conditions of any kind, express or implied,
including fitness for a particular purpose and the accuracy of their output, as section 5 of
CC BY-SA 4.0 sets out. The licences of HierText and Open Images likewise give no warranty.

## The code

The GLYPHARIS source code is licensed separately under the Apache License, Version 2.0 (`LICENSE`
and `NOTICE` in the source repository). That licence does not cover these weights, and this
notice does not cover the code.
