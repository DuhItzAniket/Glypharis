# ADR-0009: Drop SynthText; keep every training source commercially usable

- **Status:** Accepted (2026-10-10, Phase 2). Partly supersedes ADR-0003 (the "keep SynthText" part),
  ADR-0005 (the SynthText row) and ADR-0008 (point 4, SynthText subset for v0.1).

## Context
The Phase 2 licence check (recorded in DATASETS.md) found that SynthText is distributed under the VGG
"Dataset Group 2" terms of access: a non-transferable licence strictly for non-commercial research, no
use of outputs in commercial products, no redistribution, and a 12-month term after which the data must
be deleted. The project page also says the download links are no longer available; the old archive URL
still answers but is no longer linked, so it is unclear whether fetching it is sanctioned.

GLYPHARIS exists to serve a consumer application whose commercial status is not decided. A model trained
on SynthText could not be used in a commercial product. GlyphSynth (ADR-0003) already provides
synthetic training data with licences we control.

## Decision
1. **SynthText is not downloaded or used.** It stays in DATASETS.md with the reason, and in the brief
   traceability as a documented deviation.
2. **Training-data rule:** every source that can influence the weights of a model delivered to a consumer
   must allow commercial use (possibly with attribution or share-alike). Today that is HierText
   (CC BY-SA 4.0 annotations, CC BY 2.0 images) and GlyphSynth (open-licence fonts, public-domain or
   permissively licensed text). Research-only or unlicensed datasets (FUNSD, OmniDocBench, IIIT5K, SROIE,
   DocVQA) are used **only for evaluation**.
3. **RQ2 is reframed** from "scene vs document synthetic data" to: how much document-style synthetic data
   helps, how it scales (25K → 250K lines at equal steps), and which generator factors matter (font
   diversity, degradation). The scene-vs-document question survives as RQ6 (HierText scene slices vs
   document data) and through the HierText-only control.
4. **v0.1 data** is HierText + GlyphSynth.

## Consequences
- 41 GB less to download; RISK_REGISTER R-03 shrinks and R-04 is closed.
- The brief's SynthText subset study (25K / 50K / 100K / 250K) is replaced by the same sizes of
  GlyphSynth; this is listed in IMPLEMENTATION_PLAN §8.
- The paper loses a direct comparison with the most common scene-text synthetic set. That limitation is
  stated, with the licence reason.
- The weights remain releasable under terms compatible with CC BY-SA (to be settled in the model card,
  Phase 21).

## Revisit if
The owner decides the consumer is strictly non-commercial and wants the SynthText comparison as a
research-only arm, accepts the VGG terms personally, and a sanctioned download route exists. A model
trained that way would never be delivered to the consumer.
