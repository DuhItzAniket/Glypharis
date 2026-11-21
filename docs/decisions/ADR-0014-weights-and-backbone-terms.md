# ADR-0014: Terms for the v1 weights and the ImageNet-initialised detector

- **Status:** Accepted (2026-11-18, Phase 21; decided by the owner). Records an exception to ADR-0009's
  training-data rule and settles what ADR-0009 left to the model card.

## Context
ADR-0009 requires every source that can influence the weights of a delivered model to allow commercial
use, possibly with attribution or share-alike. ADR-0007 allows generic ImageNet backbones. The two meet
in v1's detector: GW-DET-001 starts from torchvision's MobileNetV3 Large ImageNet-1K weights, and
DATASETS.md records that the ImageNet terms do not clearly allow commercial use. R-05 left the question
to the owner.

GW-DET-002, the same detector from random initialisation, is the measured alternative: at the Phase 8
budget it reached 0.020 lower F1@0.5 on validation than GW-DET-001 (F6). It is not a drop-in
replacement. Every v1 result from
Phase 14 on was measured with GW-DET-001, so a detector trained another way would be a new version with
no test results of its own.

The recognizer and the detector were trained on HierText, whose annotations are CC BY-SA 4.0 and whose
images come from Open Images (CC BY 2.0). ADR-0009 left the terms for the weights to the model card.

*Clarification, 2026-11-18 (the decision below is unchanged).* The sentence above is narrower than the
publisher's statement. HierText's README, checked on 2026-11-18, says the HierText dataset is released
under CC BY-SA 4.0, and gives no separate licence for its annotations or its images. The wording to use
is: HierText is released under CC BY-SA 4.0; its images come from Open Images, which lists them as
CC BY 2.0.

## Decision
1. **GW-DET-001 stays v1's detector, as a disclosed exception to ADR-0009's rule.** The model card and
   DATASETS.md say that its backbone was initialised from ImageNet-1K weights and that those terms do not
   clearly allow commercial use. No document calls the v1 weights cleared for commercial use.
2. **The intended terms for the v1 weights are CC BY-SA 4.0, with attribution to HierText and Open
   Images.** They take effect only if the weights are released, which is Phase 23's decision, and they
   are subject to point 1. No weights are distributed today.
3. These are the project's chosen handling, not a legal conclusion that the weights or their commercial
   use are cleared.

## Consequences
- MODEL_CARD.md, DATASETS.md and the README state the exception and the intended terms in these words.
- R-05 stays open for commercial use: the owner's decision settles how v1 is described, not what the
  ImageNet terms allow.
- Phase 23 returns to this ADR before any weights are published.
- ADR-0009's rule still governs training data for any new version.

## Phase 23 registration (2026-11-22)
Phase 23 returns to this ADR as its Consequences ask. The owner decided that the weights are
packed and verified with the notices this ADR describes but are published only after the qualified
advice named under Revisit if below, and that v1.0 is held until then rather than released without a model (PHASE_STATUS,
Phase 23, D4). No weight file leaves this machine before then. Nothing in the terms above changes, and only `glypharis-v1` would ship (D7).

## Revisit if
- The consumer application is to be sold, or the weights are to be published: take qualified advice on
  both points first.
- A new version is trained for release: a randomly initialised detector (as GW-DET-002) removes the
  ImageNet question, at the cost of its own evaluation.
- The ImageNet, torchvision, HierText or Open Images terms change.
