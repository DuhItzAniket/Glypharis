# ADR-0012: Detector labels and unclip operating points

- **Status:** Accepted (2026-10-25, Phase 8, with the GW-DET registration)

## Context
GLYPHARIS-Det (ARCHITECTURE §3) is a DBNet-style detector: it predicts a probability map, keeps the
regions above a threshold (the shrunk "kernels" it was trained on) and grows each kernel back out
("unclip") to a text-line box. Four choices sit between the trained network and the boxes that
Phases 9, 10 and 14 rely on, and each of them changes the numbers:

1. how far to grow a kernel, and which rule the reported F1 uses when more than one is tried;
2. which corner is top-left, since Phase 10 rectifies line crops from the box corners;
3. whether a box is a rotated rectangle or follows the kernel's outline;
4. the weight of the probability term in the loss, where the DBNet reference code and the plan differ.

## Decision
1. **Two pre-registered operating points, one of record.** `dbnet15` grows a kernel by DBNet's
   D = 1.5·A/L; `inverse10` by the exact inverse of the training shrink for a rectangle. Each
   validation pass scores both, and the selection metric `val_f1_50` is the better of the two
   (micro F1 at IoU 0.5; ties go to the first listed). Every checkpoint stores the operating point
   that won on its epoch's validation subset (`selected_op`). For `best.pt` that stored point is the
   **operating point of record**: full validation, the run report, `scripts/det_compare.py` (H9,
   H10), `TextLineDetector.from_checkpoint` and the release export all use it. The arg-max over the
   full validation set is recorded as information only and never re-selects. Phase 9 may choose a
   different operating point on validation, as its own registered change. F1 at IoU 0.3 is a
   diagnostic and is never reported as a result.
2. **Text direction (the aspect-2 rule).** A box's top-left corner follows the reading direction:
   the rectangle's long edge when its long/short ratio is at least 2, otherwise the edge nearest to
   horizontal (ties go to the long edge), oriented to point right on the page. Where a ground-truth
   polygon exists, its own TL→TR direction is used instead. Following the long edge always would
   turn short upright words by 90°. No box-only rule can see text that runs along the short side of
   a near-square box, so every box also carries its aspect and long-axis angle for Phase 10.
3. **Rectangles by default, outlines on request.** `det.post.box_type: quad` (all GW-DET runs)
   returns the rotated rectangle around the grown kernel; `poly` returns the kernel outline grown by
   the same distance. Candidates keep the simplified contour, so Phase 9 compares the two on
   validation, including its most curved lines, and fixes the Phase 14 setting before GDC is read
   (RISK_REGISTER R-22).
4. **Loss weights as pre-registered.** BCE weight 1, α = 1, β = 10 (ARCHITECTURE §3). The DBNet
   reference code scales the probability term by 5 (`bce_scale`) and PaddleOCR's loss does the same
   through `alpha`; this is recorded, not adopted, because changing a pre-registered value after the
   fact would make it a new experiment. `det.loss.bce_weight` stays configurable for a later
   ablation under its own ID.

## Consequences
- One number per run: the headline F1, the H9 and H10 comparisons and the shipped detector all
  refer to the same stored operating point, chosen on validation only.
- Selecting on the better of two points keeps the overfit gate, the label ceiling and the run
  selection on one metric whichever regime the trained network ends up in; the per-pass
  kernel-excess diagnostic tells Phase 9 which regime that is.
- `quad` output may under-cover long curved lines at IoU 0.5; the risk is tracked as R-22 and
  measured in Phase 9 before any test-set evaluation.

## Revisit if
- Phase 9's calibration shows a single operating point dominates on validation.
- Phase 9 finds `poly` better on curved validation lines, in which case the Phase 14 setting changes
  before GDC is read.
- A later ablation tests `bce_weight` 5 and it helps.
