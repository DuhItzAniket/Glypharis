# Datasets

Provenance, licence and role of every dataset GLYPHARIS used or considered, and of the third-party
weights and evaluators around them. Roles are fixed by
[ADR-0005](docs/decisions/ADR-0005-evaluation-data-roles.md),
[ADR-0009](docs/decisions/ADR-0009-no-synthtext.md) and
[ADR-0013](docs/decisions/ADR-0013-gdc-scan-role.md). Machine-readable records are in
`data/manifests/`; [Where each record field is kept](#where-each-record-field-is-kept) says which file
holds what. Datasets themselves are never committed; [Redistribution](#redistribution) lists what the
repository does contain.

Counts marked "published" are the authors' figures. All other counts come from the tracked manifest
or report named beside them; acquisition and preparation have no experiment IDs, so the path stands
in for one. Anything measured by a run names its ID.

Licences were checked on the publishers' pages on 2026-10-10 (Phase 2), on 2026-10-29 for the external
test sets, and again on 2026-11-18. [Licence checks](#licence-checks) gives the result for each page,
including the pages that could not be reached. This document records what the publishers state and
how the project handled it. It is not legal advice.

## Training-data rule

Every source that can influence the weights of a model delivered to a consumer must allow commercial
use, possibly with attribution or share-alike
([ADR-0009](docs/decisions/ADR-0009-no-synthtext.md)). Research-only or unlicensed datasets are used
only for evaluation. The one disclosed exception to this rule is the ImageNet-1K initialisation of the
v1 detector ([ADR-0014](docs/decisions/ADR-0014-weights-and-backbone-terms.md)); see
[Licensing implications for v1](#licensing-implications-for-v1).

## What trained the delivered models

No delivered model was trained on GlyphSynth.

| Model | Experiment | Training data | Initialisation |
|---|---|---|---|
| v1 recognizer | GW-REC-034 | HierText train lines | random |
| v1 detector | GW-DET-001 | HierText train pages | backbone from MobileNetV3-Large ImageNet-1K weights |
| v0.1 recognizer | GW-REC-024 | HierText train lines | random |
| v0.1 detector | GW-DET-001 at the Phase 9 settings | HierText train pages | as v1 |

GlyphSynth was tried as an addition in research runs (GW-REC-002, GW-DATA-010 to GW-DATA-021) and as
row L2 of the Phase 13 ladder (GW-REC-037). L2 was not kept
(`reports/ablation/phase13_ladder.md`), so the frozen configuration
`configs/final/glypharis_v1.yaml` lists HierText packs only.

## Summary

| Dataset | Role | Licence as stated by the publisher | Commercial use | Status |
|---|---|---|---|---|
| HierText | train, validation, internal test | dataset CC BY-SA 4.0; images from Open Images, listed there as CC BY 2.0 | yes, with attribution and share-alike | downloaded 2026-10-10 |
| GlyphSynth (ours) | research training source, validation; in no delivered model | inherits font and text licences | yes | v1 built in Phase 4; variants in Phase 12 |
| Google Fonts subset | GlyphSynth input | SIL OFL version 1.1, Apache 2.0, Ubuntu Font Licence, per family | yes | downloaded 2026-10-10 |
| PG-19 subset | GlyphSynth input | Apache 2.0 for the dataset; pre-1919 Project Gutenberg books | yes | downloaded 2026-10-10 |
| GLYPH-C v1 (ours) | corruption benchmark, derived from validation lines | follows HierText and GlyphSynth | evaluation only here | frozen in Phase 4 |
| GDC v1.1 (ours) | internal test | per source document; see the attribution table | evaluation only | frozen 2026-10-22 |
| GDC-scan v1 (ours) | internal test, case study | private document, not redistributable | evaluation only | frozen 2026-11-14 |
| IIIT5K | external test | none stated; citation requested | evaluation only | converted 2026-10-29 |
| FUNSD | external test | custom licence: non-commercial research and education only | evaluation only | converted 2026-10-29; licence accepted by the owner |
| OmniDocBench | external test | data for research only, not commercial; code Apache-2.0 | evaluation only | English pages converted 2026-10-29 |
| SROIE | optional external test | none public; download needs an RRC account | not used | skipped (owner, 2026-10-29) |
| DocVQA | reference only | none public; download needs an RRC account | not used | not downloaded |
| SynthText | none | VGG terms: non-commercial research only, 12-month licence, no redistribution | no | dropped (ADR-0009) |
| MobileNetV3 ImageNet-1K weights | detector backbone initialisation (ADR-0007) | torchvision code BSD-3-Clause; weights subject to the ImageNet terms | see [Licensing implications for v1](#licensing-implications-for-v1) | installed 2026-10-23 |
| Baseline OCR weights | Phase 5 and Phase 14 baselines only | PaddleOCR models Apache 2.0; TrOCR card carries no licence tag | not part of GLYPHARIS | downloaded 2026-10-21 |
| Official evaluators | Phase 14 scoring only | HierText repository CC BY-SA 4.0; OmniDocBench code Apache-2.0 | not part of GLYPHARIS | downloaded 2026-11-15 |

## Dataset cards

Each card for a dataset the project used gives purpose and intended task, source and URL, licence,
citation, download or capture date where something was downloaded or captured, version, samples,
annotation format, preprocessing, split usage and known issues. Cards for datasets not used give
purpose, source, terms where visible, citation and the decision. Citation keys refer to
`research/paper/references.bib`.

### HierText

- **Purpose:** training and validation data of every delivered model, and an internal test set.
- **Source / URL:** Google Research, <https://github.com/google-research-datasets/hiertext> (archived
  by its owner on 2026-04-19, read-only). Images are hosted in the Open Images bucket; the exact URLs
  are in `data/manifests/hiertext.json`.
- **Licence:** HierText is released under CC BY-SA 4.0; its images come from Open Images
  (`kuznetsova2020openimages`), which lists them as CC BY 2.0 and makes no warranty about the licence
  status of any one image. The README's licence section names "The HierText dataset" and gives no
  separate licence for annotations or images (checked 2026-11-18). Obligations: attribution, and
  share-alike for the dataset and material adapted from it.
- **Citation:** Long et al., *Towards End-to-End Unified Scene Text Detection and Layout Analysis*,
  CVPR 2022 (`long2022hiertext`).
- **Download date:** 2026-10-10.
- **Version:** the original release, with test annotations published 2024-12-02, so the test split has ground truth
  and the RISK_REGISTER R-08 fallback was not needed. Image archives last modified 2022-03-03.
- **Integrity:** no published checksums. SHA-256 recorded at download in
  `data/manifests/hiertext.json`. The three ground-truth files also match GitHub's git blob SHA-1,
  checked 2026-10-10.
- **Annotation format:** gzipped JSON with a paragraph, line and word hierarchy, polygon vertices,
  transcription, and legibility and handwritten flags.

Samples, published: 11,639 images (train 8,281, validation 1,724, test 1,634) and about 1.2M words.

Samples, measured (`data/manifests/hiertext.json`):

<!-- trace-table: data/manifests/hiertext.json data/manifests/hiertext.json#counts train=train/annotations|validation=validation/annotations|test=test/annotations -->
| Split | Images | Paragraphs | Lines | Legible lines | Handwritten lines | Words | Legible words |
|---|---|---|---|---|---|---|---|
| train | 8,281 | 242,164 | 498,014 | 381,963 | 34,200 | 1,014,142 | 862,577 |
| validation | 1,724 | 51,137 | 104,139 | 80,096 | 6,667 | 210,345 | 178,382 |
| test | 1,634 | 46,316 | 94,564 | 73,737 | 7,103 | 194,374 | 167,188 |

The published word count matches the legible words, <!-- derived: hiertext_legible_words --> 1,208,147
in total; the files hold <!-- derived: hiertext_all_words --> 1,418,861 words. The download is <!-- derived: hiertext_archives_gib --> 3.64 GiB
of image archives plus <!-- derived: hiertext_gt_mib --> 79 MiB of ground truth, <!-- derived: hiertext_total_gib --> 3.72 GiB
in total.

Audit and preprocessing (Phases 3 and 4, `reports/data_quality_report.md`,
`reports/data_analysis/hiertext/preprocessing.md`, `data/manifests/processed/hiertext.json`). The
audit found no missing, corrupt or wrongly sized images. Exclusion list v1 removes images that
duplicate, or share most of their text with, an image in a higher-priority split; test is never
reduced for duplication. It also removes train lines with broken boxes and keeps one empty "legible"
test line away from the recognizer. Out-of-range vertices are clipped and text is normalised
(normalisation v1). Crops for the recognizer are cut as below, and skipped lines are counted, not
dropped silently.

<!-- trace-table: reports/data_quality_report.md reports/data_quality_report.md#exclusions train=train|validation=validation|test=test -->
| Split | Images excluded | Lines excluded (all uses) | Lines excluded from recognizer only |
|---|---|---|---|
| train | 68 | 3 | 0 |
| validation | 9 | 0 | 0 |
| test | 0 | 0 | 1 |

<!-- trace-table: data/manifests/processed/hiertext.json data/manifests/processed/hiertext.json#splits train=train|validation=validation|test=test -->
| Split | Pages | Lines | Usable by the recognizer | illegible | handwritten | vertical | out_of_charset | too_long |
|---|---|---|---|---|---|---|---|---|
| train | 8,213 | 493,464 | 348,935 | 115,764 | 25,122 | 2,039 | 1,457 | 147 |
| validation | 1,715 | 103,810 | 73,888 | 24,019 | 4,920 | 371 | 590 | 22 |
| test | 1,634 | 94,564 | 67,277 | 20,827 | 5,738 | 424 | 282 | 15 |

<!-- trace: data/manifests/processed/hiertext.json data/manifests/processed/hiertext.json#crop_settings -->
Crop settings: height 64 px, margin 0.15 of the line height, JPEG quality 95. Lines longer than <!-- derived: crop_max_aspect --> 64
times their height are not cut.

<!-- trace: data/manifests/charset.json data/manifests/charset.json#vocabulary_size -->
The recognizer's character set has 275 classes including the CTC blank
(`data/manifests/charset.json`, `reports/charset/charset_report.md`).

- **Intended task and split usage:** detector training on line polygons and recognizer training on
  line crops (train); model selection, tuning and the validation reference (validation); internal test
  in Phases 5 and 14, with Phase 22 the only remaining permitted use (test). The
  [usage table](#usage-by-dataset-phase-and-experiment) has the experiments.
- **Known issues:** the labels contain some noise, visible on the crop sheet, recorded and not
  corrected. The set is mostly natural images containing text, with a share of document-like images,
  so it does not represent scanned pages on its own (RISK_REGISTER R-06). Lines are mostly single
  words:

<!-- trace-table: reports/data_quality_report.md reports/data_quality_report.md#statistics words per legible line=words per legible line|characters per legible line=characters per legible line|legible line height (px)=legible line height (px) -->
| Median per legible line | train | validation | test |
|---|---|---|---|
| words per legible line | 1.0 | 1.0 | 1.0 |
| characters per legible line | 7.0 | 7.0 | 7.0 |
| legible line height (px) | 14.0 | 14.0 | 14.0 |

### GlyphSynth (ours, [ADR-0003](docs/decisions/ADR-0003-document-domain-synthetic-data.md))

- **Purpose:** document-style synthetic text lines for the Phase 12 study and a synthetic validation
  set. It is not part of any delivered model.
- **Source / URL:** rendered by `glypharis.data.synth` from the two inputs below; no download.
- **Licence:** follows the inputs: open font licences, and PG-19 text. Generated snippets carry no
  third-party text.
- **Citation:** none for the generator; `rae2020compressive` for PG-19.
- **Version:** v1, fully determined by generator version, config, seed, font commit and text manifest.
  Rebuilding from the seed reproduces the packs byte for byte.
- **Annotation format:** line packs with the text label of each rendered line.
- **Preprocessing:** rendering at crop height, then capture effects at controlled severities: rotation,
  perspective, paper texture, uneven light, ink spread, low resolution, Gaussian blur, Gaussian noise,
  salt-and-pepper noise, binarisation and JPEG. The share of lines per effect is in
  `reports/data_analysis/glyphsynth/report.md`.
- **Split usage:** train and held-out validation, with disjoint font families and disjoint books. GDC
  sources are excluded by the experiment protocol.

Input fonts: Google Fonts, <https://github.com/google/fonts>, downloaded 2026-10-10, pinned at commit
`bd8f81ddb5c74d5c8897b36ad88b440266245103`. The top-level directory gives each font's licence
(`ofl/` SIL OFL version 1.1, `apache/` Apache 2.0, `ufl/` Ubuntu Font Licence). Fonts are used unmodified, so
OFL Reserved Font Names are not affected.

<!-- trace-table: data/manifests/google_fonts.json data/manifests/google_fonts.json#counts font files=files|families=families|serif families=families_by_category/serif|sans families=families_by_category/sans|mono families=families_by_category/mono|OFL families=families_by_licence/ofl|Apache families=families_by_licence/apache|Ubuntu Font Licence families=families_by_licence/ufl -->
| Fonts | Count |
|---|---|
| font files | 245 |
| families | 100 |
| serif families | 44 |
| sans families | 41 |
| mono families | 15 |
| OFL families | 96 |
| Apache families | 2 |
| Ubuntu Font Licence families | 2 |

Input text: PG-19 (DeepMind, <https://github.com/google-deepmind/pg19>, `rae2020compressive`),
downloaded 2026-10-10 from its public bucket, each file matching the bucket's MD5. These are
Project Gutenberg books published before 1919, with the Gutenberg boilerplate removed. The README gives
Apache 2.0 as the dataset's licence and no separate licence for the texts; the project treats the
texts as public domain in the US because of their age. PG-19 was used instead of gutenberg.org, whose
policy forbids automated access to the website.

The train row below is a seeded sample of the PG-19 train books.

<!-- trace-table: data/manifests/pg19.json data/manifests/pg19.json#counts train=train|validation=validation|test=test -->
| PG-19 split | books | characters |
|---|---|---|
| train | 300 | 120,666,451 |
| validation | 50 | 17,366,594 |
| test | 100 | 40,419,935 |

Packs (`data/manifests/processed/glyphsynth.json`, `data/manifests/processed/glyphsynth_variants.json`,
`reports/data_analysis/glyphsynth/report.md`). Train lines use the train books and train families;
validation lines use the PG-19 validation and test books and the held-out families.

<!-- trace-table: reports/data_analysis/glyphsynth/report.md reports/data_analysis/glyphsynth/report.md#output train=train|val=val -->
| Split | Lines | Families used | Median / p95 characters |
|---|---|---|---|
| train | 100,000 | 85 | 39 / 87 |
| val | 10,000 | 15 | 40 / 87 |

<!-- trace: reports/data_analysis/glyphsynth/report.md reports/data_analysis/glyphsynth/report.md#text -->
About 20% of the lines are generated modern snippets such as dates, money and codes; the rest come
from the books.

Phase 12 variants, all with seed 20261010 and used only in the synthetic-data study:

<!-- trace-table: data/manifests/processed/glyphsynth_variants.json data/manifests/processed/glyphsynth_variants.json ext=ext|fewfonts=fewfonts,count|nocapture=nocapture -->
| Variant | Lines | What differs |
|---|---|---|
| ext | 150,000 | continues v1 from line 100,000 to line 250,000; 85 families |
| fewfonts | 100,000 | six font families only |
| nocapture | 100,000 | no capture effects; 85 families |

- **Known issues:** generated from the same process as its validation set, so GlyphSynth validation is
  optimistic for arms trained on GlyphSynth. It is not a substitute for real document pages.

### GLYPH-C v1 (ours)

- **Purpose:** the corruption benchmark for robustness (GW-DATA-009, and the rows of the Phase 13
  ladder). It is derived from validation lines, so it is not an independent set.
- **Source / URL:** built by `scripts/build_glyph_c.py`; no download.
- **Licence:** its HierText half is derived from HierText validation and follows CC BY-SA 4.0, with the
  images' Open Images terms. Its other half is GlyphSynth validation.
- **Citation:** none for the pack. Its mean corruption error follows `hendrycks2019benchmarking`.
- **Version:** v1, frozen (`data/manifests/processed/glyph_c.json`); sample list in
  `data/manifests/processed/glyph_c_samples.json`.
- **Annotation format:** line packs, with the source line's label.
- **Preprocessing:** seven corruptions (Gaussian blur, Gaussian noise, JPEG, low resolution, contrast,
  rotation, perspective) at five levels each, plus the clean copy of every line.
- **Split usage:** validation-derived; used for comparison between training recipes, never for
  training.

<!-- trace: data/manifests/processed/glyph_c.json data/manifests/processed/glyph_c.json -->
Samples: <!-- derived: glyph_c_per_source --> 3,000 HierText validation lines and <!-- derived: glyph_c_per_source --> 3,000
GlyphSynth validation lines (<!-- derived: glyph_c_samples --> 6,000 in total), seed 7, giving 216,000 crops.

- **Known issues:** half of the pack is synthetic, so a clean CER on GLYPH-C cannot be compared with a
  CER on HierText. Its corruptions are close relatives of the training augmentations, and its HierText
  half comes from the lines that selected each run's checkpoint.

### GDC: Glypharis Document Challenge (ours)

- **Purpose:** a small, frozen internal test set of image-only document pages.
- **Source / URL:** print-and-capture from born-digital, openly licensed PDFs. The sources are in
  [the table below](#gdc-source-documents) and `data/manifests/gdc_sources.json`.
- **Licence:** per source document. Our annotations are derived from those documents. The captures stay
  out of git.
- **Citation:** none for the set; each source is credited in the table.
- **Download date:** sources 2026-10-10. Captures 2026-10-21.
- **Version:** version 1.1, frozen 2026-10-22 (`data/manifests/gdc/gdc_v1.json`). Version 1.0 froze the
  ground-truth lines on 2026-10-21; version 1.1 added ignore regions for the part of the sheet outside
  the content box and left the lines unchanged.
- **Annotation format:** page records with line text, boxes, reading order, alignment scores and ignore
  regions.
- **Preprocessing:** ground truth comes from the PDF text layer, rebuilt line by line from character
  geometry and printed on sheets with corner markers. Each photo was identified from its markers and
  aligned with a homography refined by smooth optical flow. Overlays were checked by eye during
  construction and after alignment.
- **Rules:** frozen with a hash manifest before any model was evaluated on it (ADR-0010), never trained
  on, sources on the training exclusion list (`data/manifests/exclusions/gdc_sources.json`, checked by a
  test), and used only in Phases 5 and 14, with Phase 22 the only remaining permitted use.
- **Split usage:** test only; evaluated in Phases 5 and 14, with Phase 22 the only remaining permitted
  use.
- **Composition:** phone captures only. There are no flatbed scans, no old scans and no separate
  low-resolution condition. No transcribed subset of old scans exists: the planned real-scan slice
  became GDC-scan, a modern printed document with text-layer ground truth, so the old-scan target is
  still unmet (ADR-0013).

<!-- trace: data/manifests/gdc/gdc_v1.json data/manifests/gdc/gdc_v1.json#pages -->
Samples: 106 pages, one capture per printed sheet.

<!-- trace: data/manifests/gdc/gdc_v1.json data/manifests/gdc/gdc_v1.json#lines -->
The pages hold 8,606 ground-truth lines.

<!-- trace: GW-E2E-009 reports/external/README.md#confirmatory-tests-phase-14-plan-item-5 -->
They come from 21 source documents.

<!-- trace: data/manifests/gdc/alignment.json data/manifests/gdc/alignment.json#results/0/size -->
Captures are 3072 by 4080 pixels; <!-- derived: gdc_landscape_captures --> 4 of the <!-- derived: gdc_captures --> 106
are stored in landscape orientation.

<!-- trace: data/manifests/gdc/alignment.json data/manifests/gdc/alignment.json#rules/min_line_score -->
Each line carries an alignment score. Lines scoring below 0.3 have untrusted boxes and are excluded from
box-based metrics; there are <!-- derived: gdc_low_score_lines --> 200 of them.

Capture conditions, assigned with fixed quotas (`data/manifests/gdc/pack.json`):

<!-- trace-table: data/manifests/gdc/pack.json data/manifests/gdc/pack.json#conditions scan_app=scan_app|photo_flat=photo_flat|photo_angled=photo_angled|low_light=low_light|distant=distant|rotated=rotated|curved=curved -->
| Condition | Captures |
|---|---|
| scan_app | 37 |
| photo_flat | 16 |
| photo_angled | 16 |
| low_light | 11 |
| distant | 11 |
| rotated | 7 |
| curved | 8 |

- **Known issues:** text inside raster images has no text-layer ground truth, so it becomes ignore
  regions. Side-by-side form headers merge into one line. The set has few source documents, so
  document-level intervals are wide. The licence labels printed in
  `reports/error_cases/phase15/examples.md` date from selection and are not all current, for example
  those of `frb_fsr` and `hefferon_linalg`; the table below is.

#### GDC source documents

None of the Gutenberg books is in the PG-19 text used for GlyphSynth. "Quoted" says whether
`reports/error_cases/phase15/examples.md` quotes lines from the document; each quoted line there
carries its key, URL and licence. Lines of the two textbooks are never quoted. Licence labels in that
generated report date from selection; where they differ from this table, as for `frb_fsr` and
`hefferon_linalg`, this table is current.

| Key | Group | Credit | Licence | URL | Quoted |
|---|---|---|---|---|---|
| llama | paper | Touvron et al., `LLaMA: Open and Efficient Foundation Language Models` | CC BY 4.0 | <https://arxiv.org/abs/2302.13971v1> | yes |
| blip2 | paper | Li et al., `BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models` | CC BY 4.0 | <https://arxiv.org/abs/2301.12597v3> | yes |
| llava15 | paper | Liu et al., `Improved Baselines with Visual Instruction Tuning` | CC BY 4.0 | <https://arxiv.org/abs/2310.03744v2> | yes |
| smoothquant | paper | Xiao et al., `SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models` | CC BY 4.0 | <https://arxiv.org/abs/2211.10438v7> | yes |
| dpo | paper | Rafailov et al., `Direct Preference Optimization: Your Language Model is Secretly a Reward Model` | CC BY 4.0 | <https://arxiv.org/abs/2305.18290v3> | yes |
| mamba | paper | Gu and Dao, `Mamba: Linear-Time Sequence Modeling with Selective State Spaces` | CC BY 4.0 | <https://arxiv.org/abs/2312.00752v2> | yes |
| mistral7b | paper | Jiang et al., `Mistral 7B` | CC BY 4.0 | <https://arxiv.org/abs/2310.06825v1> | yes |
| cot | paper | Wei et al., `Chain-of-Thought Prompting Elicits Reasoning in Large Language Models` | CC BY 4.0 | <https://arxiv.org/abs/2201.11903v6> | yes |
| qlora | paper | Dettmers et al., `QLoRA: Efficient Finetuning of Quantized LLMs` | CC BY 4.0 | <https://arxiv.org/abs/2305.14314v1> | yes |
| nasa_tm | report | NASA technical memorandum | US government work, public use permitted (NTRS record) | <https://ntrs.nasa.gov/citations/20220000420> | yes |
| census_income | report | US Census Bureau, report `P60-282` | US federal work (`17 U.S.C. 105`) | <https://www2.census.gov/library/publications/2024/demo/p60-282.pdf> | yes |
| frb_fsr | report | Board of Governors of the Federal Reserve System, Financial Stability Report | public domain by the Board's statement, unless otherwise indicated; credit to the Board requested | <https://www.federalreserve.gov/publications/files/financial-stability-report-20240419.pdf> | yes |
| calculus_made_easy | book | Project Gutenberg eBook `33283` | public domain in the USA (Project Gutenberg) | <https://gutenberg.pglaf.org/3/3/2/8/33283/33283-pdf.pdf> | yes |
| hilbert_geometry | book | Project Gutenberg eBook `17384` | public domain in the USA (Project Gutenberg) | <https://gutenberg.pglaf.org/1/7/3/8/17384/17384-pdf.pdf> | yes |
| science_hypothesis | book | Project Gutenberg eBook `37157` | public domain in the USA (Project Gutenberg) | <https://gutenberg.pglaf.org/3/7/1/5/37157/37157-pdf.pdf> | yes |
| whitehead_maths | book | Project Gutenberg eBook `41568` | public domain in the USA (Project Gutenberg) | <https://gutenberg.pglaf.org/4/1/5/6/41568/41568-pdf.pdf> | yes |
| hefferon_linalg | textbook | Jim Hefferon, *Linear Algebra* | GNU FDL or CC BY-SA 3.0 United States, at the user's choice | <https://jheffero.w3.uvm.edu/linearalgebra/book.pdf> | no |
| mit_mcs | textbook | Lehman, Leighton and Meyer, *Mathematics for Computer Science* | CC BY-SA 3.0 as recorded at selection; see [Licence checks](#licence-checks) | <https://courses.csail.mit.edu/6.042/spring18/mcs.pdf> | no |
| irs_1040 | form | Internal Revenue Service, `Form 1040 (2024)` | not subject to copyright, by the IRS's statement | <https://www.irs.gov/pub/irs-prior/f1040--2024.pdf> | yes |
| irs_1040_instructions | form | Internal Revenue Service, `Form 1040 instructions (2024)` | not subject to copyright, by the IRS's statement | <https://www.irs.gov/pub/irs-prior/i1040gi--2024.pdf> | yes |
| irs_w9 | form | Internal Revenue Service, `Form W-9 (2024)` | not subject to copyright, by the IRS's statement | <https://www.irs.gov/pub/irs-prior/fw9--2024.pdf> | yes |

The Hefferon book is offered under either licence at the user's choice. No election is needed here,
because no line of it is redistributed; single words from it can occur in the pooled GDC word-confusion
pairs ([Redistribution](#redistribution)). File sizes and SHA-256 are in
`data/manifests/gdc_sources.json`.

### GDC-scan v1 (ours)

- **Purpose:** an internal test set reported as a case study (ADR-0013).
- **Source / licence:** one private printed document. It is not redistributable and no licence is
  offered. It appears in tracked files as hashes and numbers only: its text, images, page file and
  per-line outputs stay local, and its source is on the training exclusion list by hash only
  (`data/manifests/exclusions/gdc_scan.json`).
- **Citation:** none.
- **Capture and freeze date:** frozen 2026-11-14 (`frozen_utc` in
  `data/manifests/gdc_scan/gdc_scan_v1.json`); the manifest records no separate capture date.
- **Version:** version 1.0, frozen 2026-11-14 (`data/manifests/gdc_scan/gdc_scan_v1.json`). SHA-256 of the page
  file: `fc7d0cc79de34359e3c85ec6ea5e174011988a5629e9d32932e1ac9c2afbc1b8`. The manifest also lists the
  hashes of the sources and of every capture.
- **Annotation format:** page records as for GDC, with line text, boxes, reading order, alignment scores
  and ignore regions. They stay local.
- **Preprocessing and method (`scripts/gdc_scan_build.py`):** each printed page was photographed with a phone. Ground
  truth comes from the text layer of the born-digital source, as for GDC, and the owner confirmed that
  the printed document is that source with no edits after printing. Without corner markers, each photo
  is matched to its page by SIFT features and a RANSAC homography, then refined with GDC's optical
  flow. Raster images and everything outside the padded content box are ignore regions. Every overlay
  was checked by eye.
- **Split usage:** test only; evaluated in Phase 14, with Phase 22 the only remaining permitted use.

<!-- trace-table: data/manifests/gdc_scan/alignment.json data/manifests/gdc_scan/alignment.json pages=pages|lines=lines -->
| Count | Value |
|---|---|
| pages | 28 |
| lines | 779 |

<!-- trace: data/manifests/gdc_scan/alignment.json data/manifests/gdc_scan/alignment.json#rules/min_line_score -->
Of these lines, <!-- derived: gdc_scan_out_of_charset --> 87 have characters outside the charset, and <!-- derived: gdc_scan_low_score --> 3
score below 0.3 and are excluded from box-based metrics. Refined page-to-photo agreement has a median
NCC of <!-- derived: gdc_scan_ncc_median --> 0.826 and a minimum of <!-- derived: gdc_scan_ncc_min --> 0.695.

- **Known issues and limit:** it is one document, one printer and one font family, so results are a case
  study and not a benchmark. Its confidence intervals are conditional on that document and it carries
  no hypothesis verdict (ADR-0013).

### IIIT5K

- **Purpose:** an out-of-domain, word-level recognizer test (RQ6).
- **Source / URL:** CVIT, IIIT Hyderabad,
  <https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset>.
- **Licence:** none stated; the page asks for a citation. Treated as research use, evaluation only.
- **Citation:** Mishra, Alahari and Jawahar, BMVC 2012 (`mishra2012iiit5k`).
- **Download date:** 2026-10-29.
- **Version:** `IIIT5K-Word_V3.0.tar.gz`, archive dated 2014-03-19; no account, no published checksums.
- **Annotation format:** MATLAB files with a word label per cropped image.
- **Preprocessing:** the test words of `testdata.mat` were written into the line pack
  `processed/iiit5k/words_test`, images byte for byte as published, with no margin added
  (`scripts/prepare_benchmarks.py`).
- **Split usage:** test split only, in Phase 14, with Phase 22 the only remaining permitted use.

Samples, published: 5,000 cropped words (2,000 train and 3,000 test per the paper; the site gives no
split sizes).

<!-- trace: data/manifests/iiit5k.json data/manifests/iiit5k.json#files/archive/bytes -->
The archive is 105,861,943 bytes.

<!-- trace: data/manifests/processed/benchmarks.json data/manifests/processed/benchmarks.json#iiit5k_test/count -->
Converted: 3,000 test words.

- **Known issues:** the published labels are upper case. The headline is therefore the case-insensitive
  alphanumeric track; the strict track charges case against upper-case labels and is reported beside it,
  as is the normalised track (`reports/external/recognition.md`, GW-REC-039).

### FUNSD

- **Purpose:** an external end-to-end test on noisy scanned forms.
- **Source / URL:** <https://guillaumejaume.github.io/FUNSD/>, `dataset.zip`.
- **Licence:** custom EPFL licence: use solely for non-commercial research and education. The scanned
  forms come from RVL-CDIP (`harley2015rvlcdip`) and stay under their own copyright. The owner accepted
  the terms on 2026-10-29.
- **Citation:** Jaume, Ekenel and Thiran, ICDAR-OST 2019 (`jaume2019funsd`).
- **Download date:** 2026-10-29.
- **Version:** `dataset.zip` dated 2019-07-05.
- **Annotation format:** one JSON file per form, with entities and their words and boxes.
- **Preprocessing:** the test forms were converted to `processed/funsd/pages_test.jsonl.gz`. The words of
  each entity are grouped into lines by vertical overlap and joined left to right, so a line never
  crosses two entities. Lines with characters outside the charset are kept, as in every evaluation set.
- **Split usage:** FUNSD's test split only, used in Phase 14; Phase 22 is the only remaining permitted
  use. FUNSD's train split is not used.

Samples, published: 199 forms (149 train, 50 test) and 31,485 words.

<!-- trace: data/manifests/funsd.json data/manifests/funsd.json#files/archive/bytes -->
The archive is 16,838,830 bytes.

<!-- trace-table: data/manifests/processed/benchmarks.json data/manifests/processed/benchmarks.json#funsd_test test forms=page_count|lines=line_count|lines without out-of-charset characters=rec_lines -->
| Converted | Count |
|---|---|
| test forms | 50 |
| lines | 2,647 |
| lines without out-of-charset characters | 2,590 |

- **Known issues:** the lines are our grouping of FUNSD's entity words, not a unit FUNSD defines, so a
  question and its answer on one printed row are two reference lines. Handwritten fill-ins count as
  text, although handwritten lines were excluded from recognizer training. There is no reading order,
  so order-aware metrics do not apply. The set is small, so the intervals are wide.

### OmniDocBench

- **Purpose:** an external document-domain test, English text blocks only.
- **Source / URL:** <https://github.com/opendatalab/OmniDocBench> and
  <https://huggingface.co/datasets/opendatalab/OmniDocBench>.
- **Licence:** code Apache-2.0. The dataset's copyright statement says it is for research purposes only
  and not for commercial use. Evaluation only.
- **Citation:** Ouyang et al., CVPR 2025 (`ouyang2025omnidocbench`).
- **Download date:** 2026-10-29.
- **Version:** Hugging Face commit `aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec`. Each image was checked
  against its Hugging Face SHA-256.
- **Annotation format:** one JSON file with layout blocks, text and reading order per page.
- **Preprocessing:** the annotation file and the English pages only, converted to
  `processed/omnidocbench/pages_test.jsonl.gz`, with text blocks in the published reading order. Figures,
  tables, equations, masks, ignored and empty blocks become "don't care" regions.
- **Split usage:** test only, in Phase 14, with Phase 22 the only remaining permitted use.

Samples, published: 1,651 page images, tagged English, Chinese or mixed.

<!-- trace-table: data/manifests/processed/benchmarks.json data/manifests/processed/benchmarks.json#omnidocbench_en English pages=page_count|text blocks=block_count|"don't care" regions=ignored_regions -->
| Converted | Count |
|---|---|
| English pages | 755 |
| text blocks | 11,369 |
| "don't care" regions | 3,222 |

- **Known issues:** its units are layout blocks, not lines, so it is scored at page level and line
  detection metrics do not apply. Inline formulas stay as LaTeX source in the reference, and rotated and
  handwritten blocks count as text. The project's order-aware page CER is not the official protocol; the
  official components are in `reports/external/official.md` (GW-E2E-009).

### SROIE (optional, not used)

- **Purpose:** an optional receipt test.
- **Source / URL:** ICDAR 2019 RRC, <https://rrc.cvc.uab.es/?ch=13>; registration required, and no
  licence was visible publicly when checked in Phase 2.
- **Citation:** Huang et al., ICDAR 2019 (`huang2019sroie`).
- **Decision (owner, 2026-10-29):** skipped, because it needs an RRC account and the plan marks it
  optional. It has no download date, version, preprocessing or split usage here.

Samples, published: 1,000 receipts.

### DocVQA (reference only, not used)

- **Purpose:** none in evaluation. Its OCR transcriptions were produced by a commercial OCR engine, not by
  people ([ADR-0005](docs/decisions/ADR-0005-evaluation-data-roles.md), RISK_REGISTER R-19).
- **Source / URL:** <https://rrc.cvc.uab.es/?ch=17>; registration required, and the terms are behind the
  login.
- **Citation:** Mathew, Karatzas and Jawahar, WACV 2021 (`mathew2021docvqa`).
- **Decision:** not downloaded or used.

### SynthText (dropped)

- **Purpose:** none. The original plan used it as a synthetic training source.
- **Source / URL:** Visual Geometry Group, Oxford, <https://www.robots.ox.ac.uk/~vgg/data/scenetext/>.
- **Licence:** the group's `Dataset Group 2` terms of access: a non-transferable licence for non-commercial
  research whose outputs are not incorporated in commercial products, no provision to third parties, and
  a 12-month term after which the content must be deleted.
- **Citation:** Gupta, Vedaldi and Zisserman, CVPR 2016 (`gupta2016synthetic`).
- **Availability:** the project page says the download links are no longer available.
- **Decision:** not downloaded or used ([ADR-0009](docs/decisions/ADR-0009-no-synthtext.md)). The planned
  SynthText size study was replaced by a GlyphSynth study at the same sizes.

Samples, published on the project page: about 800 thousand images and about 8 million word instances.

## Evaluation packs and subsets

Evaluation crop packs keep lines with characters outside the charset and very long lines, so that a
benchmark is not shaped by one system's limits. Lines are selected by `is_rec_line`
(`glypharis/eval/page_metrics.py`): a text line with a trusted box, not on the exclusion list, either
usable by the recognizer or skipped only for being out of charset or too long.

<!-- trace: data/manifests/processed/eval_crops.json data/manifests/processed/eval_crops.json#gdc_test/crop_settings -->
Crops have height 64 and margin 0.15, and no maximum aspect ratio is applied.

<!-- trace-table: data/manifests/processed/eval_crops.json data/manifests/processed/eval_crops.json hiertext_val=hiertext_val,count|hiertext_test=hiertext_test,count|gdc_test=gdc_test,count|gdc_scan_test=gdc_scan_test,count|funsd_test=funsd_test,count -->
| Pack | Crops | Pack path |
|---|---|---|
| hiertext_val | 74,500 | `processed/hiertext/lines_validation_eval` |
| hiertext_test | 67,574 | `processed/hiertext/lines_test` |
| gdc_test | 8,372 | `processed/gdc/lines_test` |
| gdc_scan_test | 776 | `processed/gdc_scan/lines_test` |
| funsd_test | 2,647 | `processed/funsd/lines_test` |

How the HierText packs relate to the recognizer lines (`data/manifests/processed/hiertext.json`):

| Pack | Usable lines | + out of charset | + too long | = crops |
|---|---|---|---|---|
| hiertext_val | <!-- derived: hiertext_val_usable --> 73,888 | <!-- derived: hiertext_val_oov --> 590 | <!-- derived: hiertext_val_long --> 22 | <!-- derived: hiertext_val_crops --> 74,500 |
| hiertext_test | <!-- derived: hiertext_test_usable --> 67,277 | <!-- derived: hiertext_test_oov --> 282 | <!-- derived: hiertext_test_long --> 15 | <!-- derived: hiertext_test_crops --> 67,574 |

GDC has <!-- derived: gdc_lines --> 8,606 ground-truth lines and <!-- derived: gdc_crops --> 8,372
crops under this rule. No line-level breakdown of the difference is published: the tracked records do
not fully explain it, and the test page file is not opened to do so.

Validation samples, which must not be mixed with the full sets:

| Sample | Size | Rule | Used by |
|---|---|---|---|
| HierText validation pages, full | <!-- derived: sample_val_pages --> 1,715 pages | pages left after the exclusion list | GW-E2E-006, GW-DET-001 |
| HierText validation pages, Phase 5 sample | <!-- derived: sample_phase5_pages --> 400 pages | seeded sample | GW-BASE-001, 002, 005, 006, GW-E2E-011, GW-EFF-009 |
| Profile pages | <!-- derived: sample_profile_pages --> 110 pages | the first pages of the Phase 5 sample | page timing, including GW-EFF-005 and GW-EFF-009 |
| HierText validation lines, baseline sample | <!-- derived: sample_baseline_lines --> 10,000 crops | seeded sample of the evaluation pack | GW-BASE-001 to GW-BASE-007 |
| HierText validation lines, export sample | <!-- derived: sample_export_lines --> 5,000 lines | seed 20261028, from the training validation pack | GW-EFF-003, GW-EFF-006, GW-EFF-009, GW-EFF-012 |
| HierText document slice | <!-- derived: sample_slice_pages --> 104 pages | validation pages with many lines; a density proxy, not a labelled domain | GW-DATA-022, GW-REC-037 |
| OmniDocBench subset | <!-- derived: sample_odb_pages --> 100 pages | seed 20261115 over the sorted English pages (`data/manifests/processed/omnidocbench_subset100.json`) | every system in Phase 14; PP-OCRv5 server ran only on these |

## Third-party weights and evaluators

These are not datasets, but they are recorded the same way (`data/sources.yaml`, `data/manifests/`).

### MobileNetV3 ImageNet-1K weights (torchvision)

The only pretrained weights GLYPHARIS uses, under
[ADR-0007](docs/decisions/ADR-0007-pretrained-weights-policy.md).

- **Source / URL:** torchvision's MobileNetV3 Large and Small `IMAGENET1K_V1` weights,
  <https://docs.pytorch.org/vision/stable/models.html>; files under
  `https://download.pytorch.org/models/`.
- **Citation:** Howard et al., *Searching for MobileNetV3*, ICCV 2019 (`howard2019mobilenetv3`). ImageNet:
  `russakovsky2015imagenet`.
- **Licence:** torchvision's code is BSD-3-Clause. Its documentation says pre-trained models may carry
  licences or terms derived from their training data and leaves it to the user to check. These weights
  were trained on ImageNet-1K, whose terms of access limit the data to non-commercial research and
  educational use. How v1 relates to the training-data rule is in
  [Licensing implications for v1](#licensing-implications-for-v1).
- **Version:** torchvision 0.29.1 weight files; installed 2026-10-23 from local copies
  (`data/manifests/imagenet_backbones.json`).
- **Integrity:** torchvision publishes only a short hash prefix in each file name. The full hashes below
  were taken from local copies, are pinned in `data/sources.yaml`, and match those prefixes.

<!-- trace-table: data/manifests/imagenet_backbones.json data/manifests/imagenet_backbones.json#files mobilenet_v3_large-8738ca79.pth=mobilenet_v3_large,bytes|mobilenet_v3_small-047dcff4.pth=mobilenet_v3_small,bytes -->
| File | Bytes | SHA-256 |
|---|---|---|
| `mobilenet_v3_large-8738ca79.pth` | 22,139,423 | `8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04` |
| `mobilenet_v3_small-047dcff4.pth` | 10,306,551 | `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f` |

- **Role:** GLYPHARIS-Det backbone initialisation only. The detector builds torchvision's MobileNetV3 with
  `weights=None`, so nothing is downloaded at run time. With `det.model.init: imagenet` the configured
  file is checked against `det.model.weights_sha256` before it is loaded. GW-DET-001 uses Large,
  GW-DET-003 Small, and GW-DET-002 neither. A trained detector checkpoint holds all its weights and does
  not need these files.
- **Known issues:** `data/sources.yaml`, `data/manifests/imagenet_backbones.json` and
  `data/DATASET_MANIFEST.yaml` keep the licence note written at download, which leaves the question open
  for the owner. They are kept as recorded, because changing them would change tracked manifests.
  [ADR-0014](docs/decisions/ADR-0014-weights-and-backbone-terms.md) supersedes that note.

### Baseline OCR weights

PP-OCRv5, PP-OCRv6 and TrOCR-small-printed weights at the Hugging Face revisions pinned in
`data/sources.yaml`, downloaded 2026-10-21, with sizes and hashes in `data/manifests/baseline_models.json`.
They run only as baselines (GW-BASE-001 to GW-BASE-007) and are never part of GLYPHARIS (ADR-0007).
The PaddlePaddle model cards carry the tag Apache-2.0. The TrOCR card carries no licence tag; its source
repository, microsoft/unilm, states MIT. Citations: `du2020ppocr`, `cui2025paddleocr`, `paddleocr_ppocrv6`,
`li2023trocr`.

### Official evaluators

Pinned in `data/manifests/evaluators.json`, downloaded 2026-11-15 and run in their own environments for
`reports/external/official.md` (GW-E2E-009). The HierText evaluator is at commit
`70b6620b2b112597d8219e11eee9773a1403827c` of its repository (CC BY-SA 4.0). The OmniDocBench evaluator is
at commit `7ddb8df411ad9bd2fff139acafe0ed16aa1694fe` (code Apache-2.0).

## Usage by dataset, phase and experiment

Every test or external access is logged in `research/test_access_log.md`. Phase 15 (GW-REC-040,
GW-E2E-010) read Phase 14's saved outputs with no model run, as its plan set out, and used GDC-scan for
numbers only; those reads are recorded in `reports/error_cases/phase15/results.json`, not in the access
log. The v1 re-runs of Phases 16, 17, 18 and 20 used validation data only. Final validation (Phase 22)
has not been run.

| Dataset and split | Training | Validation use | Test or external evaluation |
|---|---|---|---|
| HierText train | the recognizer training runs (those of GW-REC-001 to GW-REC-036 that ran, GW-DATA-001 to GW-DATA-021); detector runs GW-DET-001, GW-DET-002, GW-DET-003, GW-DET-007 | none | none |
| HierText validation | none | checkpoint selection for every run; GW-DET-004 to GW-DET-006; GW-E2E-001 to GW-E2E-006, GW-E2E-011; GW-REC-038; GW-EFF-001 to GW-EFF-012 | none |
| HierText test | none | none | Phase 5: GW-BASE-001, 002, 004, 005, 006. Phase 14: the same baselines, GW-REC-039, GW-E2E-007 |
| GlyphSynth train and variants | research runs GW-REC-002, GW-DATA-010 to GW-DATA-021, ladder row L2 of GW-REC-037; no delivered model | none | none |
| GlyphSynth validation | none | GW-DATA-022, GW-REC-037, GW-REC-041, and the Phase 5 baselines on `glyphsynth_val` | none |
| GLYPH-C v1 | none | GW-DATA-009, GW-REC-037 | none |
| GDC v1.1 | none | none | Phase 5: GW-BASE-001 to GW-BASE-007. Phase 14: GW-BASE-001 to GW-BASE-007, GW-REC-039, GW-E2E-007, GW-E2E-008 |
| GDC-scan v1 | none | none | Phase 14: GW-BASE-001 to GW-BASE-007, GW-REC-039, GW-E2E-007, GW-E2E-008 |
| FUNSD test | none | none | Phase 14: GW-BASE-001 to GW-BASE-007, GW-REC-039, GW-E2E-007 |
| IIIT5K test | none | none | Phase 14: GW-BASE-001 to GW-BASE-007, GW-REC-039 |
| OmniDocBench English | none | none | Phase 14: GW-BASE-001, 002, 003, 005, 006, 007, GW-E2E-007 |

## Redistribution

The repository is public. No model weights are in it. No weights are distributed; release is Phase 23.

The repository contains, and so redistributes:

- HierText material in generated reports: thumbnails, crops, overlays, quoted validation and test lines,
  and word-confusion pairs, under the notices in those reports and in
  [`reports/README.md`](reports/README.md). Author and source are not recorded per image, so the
  attribution is at dataset level.
- Quoted GDC lines from the arXiv papers, the US federal documents and the Gutenberg books, in
  `reports/error_cases/phase15/examples.md` and `recognition.md`, each with its key, URL and licence, and
  GDC word-confusion pairs in `reports/confusion_analysis/GW-REC-040/`, pooled over all <!-- derived: gdc_source_documents --> 21
  sources, including the two textbooks, without the source of each word.
- Manifests with image IDs, line keys, hashes and URLs.
- Aggregate tables computed from every dataset.

The repository does not contain:

- raw or processed datasets, or the GDC captures;
- any GDC-scan text, image, page file or per-line output;
- images or text of FUNSD, IIIT5K or OmniDocBench;
- full lines of `hefferon_linalg` or `mit_mcs`; these documents are listed by key and numbers, and single
  words from them can occur in the pooled GDC word-confusion pairs;
- checkpoints or exported weights.

## Licensing implications for v1

- The v1 models were trained only on HierText. HierText's share-alike terms therefore bear on any release
  of the weights. Whether trained weights are adapted material under CC BY-SA is not settled, so the project
  states a handling and not a legal conclusion.
- Weights: Intended terms under ADR-0014, effective only if released. The owner decided on 2026-11-18
  ([ADR-0014](docs/decisions/ADR-0014-weights-and-backbone-terms.md)) that they are CC BY-SA 4.0 with
  attribution to HierText and Open Images. No weights are distributed; release is Phase 23.
- Detector initialisation: GW-DET-001 starts from MobileNetV3-Large ImageNet-1K weights, and the ImageNet
  terms do not clearly allow commercial use, which sits against the rule in ADR-0009. By the same decision
  GW-DET-001 stays v1's detector as a disclosed exception to that rule, so the v1 weights are not described
  as cleared for commercial use. GW-DET-002 is the randomly initialised comparison run. RISK_REGISTER R-05
  stays open for commercial use.
- Evaluation-only datasets (IIIT5K, FUNSD, OmniDocBench, GDC, GDC-scan) never entered training, augmentation
  or tuning.
- The all-ONNX fused-attention build exports the same weights, so the same terms would apply to it. A new
  version, not v1; validation and laptop figures only; to be evaluated in Phase 22.
- Code licence and project citation: To be settled in Phase 23.

## Licence checks

Pages were read on the date shown through a web fetch tool that returns a summary of the page, so entries
here paraphrase and should be confirmed on the page before being relied on.

| Source | Page | Checked | Result |
|---|---|---|---|
| HierText | <https://github.com/google-research-datasets/hiertext> | 2026-11-18 | the README's licence section releases "The HierText dataset" under CC BY-SA 4.0 and gives no separate licence for annotations or images; the repository's licence file is CC BY-SA 4.0; repository archived 2026-04-19 |
| Open Images | <https://storage.googleapis.com/openimages/web/factsfigures_v7.html> | 2026-11-18 | images listed as CC BY 2.0, no warranty about any image's licence status, users told to verify; Google's own annotations CC BY 4.0 (not used here) |
| Google Fonts | <https://github.com/google/fonts> | 2026-11-18 | top-level directories indicate the licence; SIL OFL version 1.1, Apache 2.0 and Ubuntu Font Licence named |
| PG-19 | <https://github.com/google-deepmind/pg19> | 2026-11-18 | Apache 2.0 listed for the dataset; no separate statement for the texts |
| IIIT5K | <https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset> | 2026-11-18 | no licence or terms; citation requested |
| FUNSD | <https://guillaumejaume.github.io/FUNSD/work/> | 2026-11-18 | non-commercial research and education only; images under copyright, from RVL-CDIP |
| OmniDocBench | <https://github.com/opendatalab/OmniDocBench>, <https://huggingface.co/datasets/opendatalab/OmniDocBench> | 2026-11-18 | code Apache-2.0; dataset for research only, not commercial use; no licence tag on the dataset card |
| arXiv papers (all nine) | abstract pages in the source table | 2026-11-18 | each listed version shows CC BY 4.0 |
| Hefferon, *Linear Algebra* | <https://hefferon.net/source.html> | 2026-11-18 | GNU FDL or CC BY-SA 3.0 United States, at the user's discretion |
| *Mathematics for Computer Science* | <https://courses.csail.mit.edu/6.042/spring18/> | 2026-11-18 | not re-confirmed for this file: the directory page shows no licence and the PDF was too large for the fetch tool. A search result shows CC BY-SA 3.0 in the front matter of the `spring15` copy. CC BY-SA 3.0 for the `spring18` file is the record made at selection (`data/sources.yaml`) |
| Federal Reserve | <https://www.federalreserve.gov/disclaimer.htm> | 2026-11-18 | public domain unless otherwise indicated; credit requested; non-Board material needs its owner's permission |
| NASA report | <https://ntrs.nasa.gov/citations/20220000420> | 2026-11-18 | record states work of the US Government, public use permitted |
| Census report | <https://www.census.gov/about/policies/open-gov/open-data.html> | 2026-11-18 | that page does not address copyright, so this rests on the report being a US federal work |
| IRS forms | <https://www.irs.gov/about-irs/use-of-content-from-irsgov> | 2026-11-18 | material by federal employees in their duties is not subject to copyright and may be copied; credit requested |
| Project Gutenberg | <https://www.gutenberg.org/policy/permission.html> | 2026-11-18 | not reached (`HTTP 503`); the public-domain statement is the record made at selection |
| ImageNet | <https://www.image-net.org/download.php> | 2026-11-18 | not reached (connection refused). The non-commercial research and education wording comes from copies of the terms found by search, not from the page |
| torchvision | <https://docs.pytorch.org/vision/stable/models.html>, <https://github.com/pytorch/vision/blob/main/LICENSE> | 2026-11-18 | BSD-3-Clause for the code; weights may carry terms derived from their training data |
| SynthText | <https://www.robots.ox.ac.uk/~vgg/data/scenetext/>, <https://www.robots.ox.ac.uk/~vgg/terms/dataset-group-2-access.html> | 2026-11-18 | terms as in the card; download links no longer available |
| Baseline weights | Hugging Face model cards of the pinned PP-OCRv5, PP-OCRv6, UVDoc and PP-LCNet repositories, `microsoft/trocr-small-printed`; <https://github.com/microsoft/unilm> | 2026-11-18 | all thirteen pinned PaddlePaddle repositories show Apache-2.0; TrOCR card has no licence tag; unilm states MIT |
| SROIE, DocVQA | <https://rrc.cvc.uab.es/?ch=13>, <https://rrc.cvc.uab.es/?ch=17> | 2026-11-18 | not reached (certificate could not be verified by the tool); last read in Phase 2 |

## Where each record field is kept

`data/DATASET_MANIFEST.yaml` is generated for downloaded sources only. Its `samples` field is empty for
several of them, its `preprocessing` field says only "none (raw download; extraction only)", and it has no
entries for GDC, GDC-scan, GlyphSynth or GLYPH-C. It was not regenerated in Phase 21. This table says where
each field is kept.

| Field | Downloaded sources | Prepared and built sets |
|---|---|---|
| `dataset` | `data/sources.yaml`, `data/manifests/<dataset>.json` | file names below |
| `version` | `data/manifests/<dataset>.json` | `data/manifests/gdc/gdc_v1.json`, `data/manifests/gdc_scan/gdc_scan_v1.json`, `data/manifests/processed/glyphsynth.json`, `data/manifests/processed/glyph_c.json` |
| `source` | `data/manifests/<dataset>.json` (`homepage`, file URLs) | the cards above; `data/manifests/gdc_sources.json` for GDC |
| `license` | `data/manifests/<dataset>.json`, this document | this document |
| `download_date` | `downloaded_utc` per file in `data/manifests/<dataset>.json` | `frozen_utc` in the GDC and GDC-scan manifests |
| `samples` | `counts` in `data/manifests/hiertext.json`, `google_fonts.json`, `pg19.json` | `data/manifests/processed/hiertext.json`, `benchmarks.json`, `eval_crops.json`, `glyphsynth.json`, `glyphsynth_variants.json`, `glyph_c.json`; `data/manifests/gdc/pack.json`, `data/manifests/gdc_scan/alignment.json` |
| `preprocessing` | not applicable to the raw download | settings in the same processed manifests; `reports/data_analysis/hiertext/preprocessing.md`; the cards above |
| `split` | `role` in `data/manifests/<dataset>.json` | [the usage table](#usage-by-dataset-phase-and-experiment), ADR-0005, ADR-0013 |

## Consumer sample documents

Real documents from the document-reading application this engine is built for are not a dataset of this
repository. If the owner provides
sample pages for qualitative checks, they stay outside the repository and outside git, are not used for
training or tuning, and only aggregate observations are written down.
