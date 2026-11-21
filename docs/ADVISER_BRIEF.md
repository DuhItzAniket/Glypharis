# Brief for a qualified adviser: terms for the GLYPHARIS v1 weights

This brief is for the project owner to give to a qualified adviser before any model weights are
published (Phase 23, item 10; [ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md), "Revisit
if"). It sets out the facts and the questions. **It is not legal advice**, and nothing in it is a
legal conclusion. The project's own decisions are recorded in
[ADR-0009](decisions/ADR-0009-no-synthtext.md) (training-data rule) and
[ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md) (intended terms for the v1 weights and
the detector exception). Publishers' terms are as recorded in [`DATASETS.md`](../DATASETS.md), which
paraphrases pages read through a summarising tool and should be confirmed on the pages themselves.

## 1. What the project is

GLYPHARIS is an OCR engine for document images: a line detector finds text lines on a page and a
line recognizer reads each line. It is intended to serve a document-reading application for
consumers, whose commercial status is not decided (ADR-0009); ADR-0014 asks for qualified advice
before that application is sold or the weights are published. The code
and the project's own text are licensed under Apache-2.0 (`LICENSE`, `NOTICE`, Phase 23 D2 and D3).
The weights are separate files and are not covered by that licence.

## 2. What trained on what

| Model | Experiment | Training data | Initialisation |
|---|---|---|---|
| v1 recognizer | GW-REC-034 | HierText training lines | random |
| v1 detector | GW-DET-001 | HierText training pages | backbone from MobileNetV3-Large ImageNet-1K weights |

Source: [`DATASETS.md`](../DATASETS.md), "What trained the delivered models". Settings were chosen
on HierText validation data. The evaluation-only sets (FUNSD, OmniDocBench, IIIT5K, GDC and
GDC-scan) never entered training, augmentation or tuning. The project's synthetic data (GlyphSynth)
is in no delivered model.

## 3. The terms of what went in

- **HierText.** The publisher's README, checked on 2026-11-18, releases "the HierText dataset" under
  CC BY-SA 4.0 and gives no separate licence for its annotations or its images. The repository was
  archived by its owner on 2026-04-19. The project's wording is: HierText is released under
  CC BY-SA 4.0; its images come from Open Images, which lists them as CC BY 2.0.
- **Open Images.** The images are listed as CC BY 2.0, with no warranty about the licence status of
  any one image, and users are told to verify it. The project does not record the author or source
  of each image; line keys begin with the HierText image ID, and no per-image metadata has been
  downloaded (Phase 23, D9).
- **ImageNet-1K initialisation.** GW-DET-001's backbone started from torchvision's MobileNetV3-Large
  `IMAGENET1K_V1` weights. torchvision's code is BSD-3-Clause; its documentation says pre-trained
  models may carry licences or terms derived from their training data. The ImageNet terms of access
  limit the data to non-commercial research and educational use; that wording comes from copies of
  the terms found by search, because the ImageNet page could not be reached on 2026-11-18. The file
  used is pinned by SHA-256 in `data/sources.yaml`. The trained detector checkpoint holds all its
  weights and does not need the ImageNet file at run time.
- **The alternative.** GW-DET-002 is the same detector trained from random initialisation. At the
  Phase 8 budget it had lower validation F1 than GW-DET-001 (finding F6), and every v1 result from
  Phase 14 on was measured with GW-DET-001, so replacing it would make a new version with no test
  results of its own (ADR-0014, Context).

## 4. What the project intends

From ADR-0014 (decided by the owner on 2026-11-18) and Phase 23 (D3, D4, D7, D9):

1. The intended terms for the v1 weights are **CC BY-SA 4.0, with attribution to HierText and Open
   Images**. They take effect only if the weights are released.
2. GW-DET-001 stays v1's detector as a **disclosed exception** to ADR-0009's rule that every source
   influencing delivered weights must allow commercial use. No document calls the v1 weights cleared
   for commercial use.
3. The weights would ship as one archive, separate from the code packages, holding the bundle files,
   `LICENSE-WEIGHTS` (CC BY-SA 4.0) and a `NOTICE.md` with the attribution, the changes made, the
   ImageNet disclosure and no warranty. Attribution is at dataset level.
4. Only the v1 bundle `glypharis-v1` would ship. The phone candidate (below) is not released.
5. No weight file is published before the owner confirms the advice and gives the go.

The repository already redistributes some HierText material under CC BY-SA 4.0 (crops, quoted lines
and tables in generated reports and the model card), with the attribution in `reports/README.md` and
`NOTICE`. The code beside it is Apache-2.0.

## 5. The phone candidate and quantised exports

Phase 22 evaluated a phone candidate: the same v1 weights exported to ONNX with the attention
operations fused and one line per recognizer run (GW-EFF-013, GW-EFF-014; results GW-REC-045,
GW-E2E-020, GW-E2E-021). Phase 16 also measured a dynamic INT8 quantisation of the ONNX recognizer
(GW-EFF-008). Both are treated as new versions, not v1. The project's records say two things that
need reconciling:

- `DATASETS.md`, "Licensing implications for v1": the fused build "exports the same weights, so the
  same terms would apply to it".
- `docs/PHASE_STATUS.md`, Phase 22 limits: "ADR-0014's ImageNet exception was written for v1 and
  does not automatically cover a released candidate".

The project reads the first as being about the data terms the exported weights would carry, and the
second as being about its own rule, under which an exception is decided for a named version. That
reading has not been checked by anyone qualified, and the candidate is not part of v1.0.

## 6. Questions

1. **Adapted material.** Are trained model weights adapted material of HierText under CC BY-SA 4.0,
   and of the Open Images images under CC BY 2.0? If they are, what must a release do to comply; if
   they are not, may the project still choose CC BY-SA 4.0 for them as it intends?
2. **The ImageNet-initialised detector.** Can the weights be offered under CC BY-SA 4.0 when the
   detector's starting point came from ImageNet-1K weights whose data terms limit use to
   non-commercial research and education? How should this be disclosed in `NOTICE.md` and the model
   card without adding a restriction on recipients that CC BY-SA 4.0 would not allow? Is a
   disclosure enough, or does the exception need another handling, such as releasing a randomly
   initialised detector instead?
3. **Credit.** Is dataset-level attribution to HierText and Open Images enough, or is per-image
   credit (author, source, licence of each Open Images image) needed, for the weights and for the
   crops and quoted lines already in the repository? Per-image credit would need extra metadata to
   be downloaded.
4. **Code and weights side by side.** Can the code be Apache-2.0 while the weights, and the HierText
   material in the repository's reports, are CC BY-SA 4.0, with `NOTICE` drawing the line as Phase
   23 D3 does? Does the code loading the weights at run time change that?
5. **A paid consumer application.** If the consumer application is sold later and ships these
   weights, or weights derived from them, what does each set of terms require: the share-alike and
   attribution of CC BY-SA 4.0, the attribution of CC BY 2.0, and the ImageNet terms through the
   detector's initialisation? Do the OCR outputs (text and boxes) carry any of these terms?
6. **Exports of the same weights.** Do the fused ONNX build and an INT8 quantisation of the same
   weights fall under the same terms as the v1 weights? How should the two statements in section 5
   be reconciled, and would releasing one of them need its own decision on the ImageNet exception?

## 7. Where to find the records

- [ADR-0009](decisions/ADR-0009-no-synthtext.md): the training-data rule.
- [ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md): the intended terms and the detector
  exception.
- [`DATASETS.md`](../DATASETS.md): every dataset's terms, the licence checks with dates, what the
  repository redistributes, and "Licensing implications for v1".
- [`MODEL_CARD.md`](../MODEL_CARD.md#licence-information): the licence information as the model card
  states it.
- [`NOTICE`](../NOTICE): what the code licence does not cover.
- `docs/PHASE_STATUS.md`, Phase 23: decisions D2 to D9 and the release plan.
