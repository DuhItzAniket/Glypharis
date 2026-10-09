# Datasets

Provenance, licence and role of every dataset GLYPHARIS uses or considers. Roles are fixed by
[ADR-0005](docs/decisions/ADR-0005-evaluation-data-roles.md). Machine-readable records go in
`data/DATASET_MANIFEST.yaml` (Phase 2).

**Status: Phase 0 draft.** Nothing has been downloaded yet. Counts below are the figures *published by
the dataset authors* and are marked "(published)"; Phase 2 replaces them with counts measured after
download. Every licence marked **verify** must be read from the official source and recorded here before
the dataset is downloaded (IMPLEMENTATION_PLAN Phase 2, task 5). Datasets are never committed to git.

## Summary

| Dataset | Role | Granularity | Size (published) | Licence | Priority for v0.1 |
|---|---|---|---|---|---|
| HierText | train / val / test | word, line, paragraph polygons | 11,639 images | Annotations CC BY-SA 4.0; images from Open Images (verify) | required |
| SynthText | train (synthetic, scene) | word / char boxes | ~800K images (~41 GB archive) | verify (academic research use) | subset only |
| GlyphSynth (ours) | train / val (synthetic, document) | line and page | generated | inherits font + corpus licences | required |
| GDC (ours) | internal test | page, line | target 100–150 pages | sources openly licensed; ours CC BY 4.0 (planned) | required (built Phase 4) |
| IIIT5K | external test (recognizer) | cropped words | 5,000 words (2,000 train / 3,000 test) | verify (research use) | Phase 14 |
| FUNSD | external test (E2E, forms) | words, entities | 199 forms (149 / 50) | verify (non-commercial research) | Phase 14 |
| OmniDocBench | external test (documents) | page, block, line | ~1K PDF pages (version-dependent) | verify | Phase 14 |
| SROIE | external test (receipts, low priority) | lines | 1,000 receipts (626 / 361) | verify | optional |
| DocVQA | reference only | words with OCR boxes | ~12.7K images | verify | not used for CER |

## Dataset cards

Each card follows the brief's required fields: name, source, URL, licence, citation, download date,
version, samples, annotation format, intended task, preprocessing, split usage.

### HierText

- **Source / URL:** Google Research, `https://github.com/google-research-datasets/hiertext`
- **Citation:** Long et al., *Towards End-to-End Unified Scene Text Detection and Layout Analysis*,
  CVPR 2022 (`long2022hiertext`).
- **Licence:** annotations CC BY-SA 4.0; images come from Open Images (per-image licences, CC BY 2.0
  stated by Open Images). **Verify** both before download. Share-alike may constrain how derived data
  (not weights) is redistributed; recorded in RISK_REGISTER R-05.
- **Version / download date:** pending Phase 2.
- **Samples (published):** 11,639 images (train 8,281 / validation 1,724 / test 1,634).
- **Annotation format:** JSON with paragraph → line → word hierarchy, polygon vertices, transcription,
  legibility and handwritten flags.
- **Intended task here:** detector training (line polygons), recognizer training (line crops), E2E
  validation.
- **Split usage:** official train → train; official validation → validation. **Test ground truth may be
  withheld** (it was used for a competition); if so, apply EXPERIMENT_PROTOCOL §1 rule 3. Decide and
  record in Phase 2.
- **Preprocessing:** illegible / handwritten lines excluded from recognizer training and marked "don't
  care" for detection; counts reported, nothing silently dropped.
- **Domain note:** mostly natural images containing text, with a share of document-like images.
  Not representative of scanned pages on its own (RISK_REGISTER R-06).

### SynthText

- **Source / URL:** Visual Geometry Group, University of Oxford,
  `https://www.robots.ox.ac.uk/~vgg/data/scenetext/`
- **Citation:** Gupta, Vedaldi, Zisserman, *Synthetic Data for Text Localisation in Natural Images*,
  CVPR 2016 (`gupta2016synthetic`).
- **Licence:** **verify** on the VGG page before download; treated as academic research use until then.
- **Samples (published):** ~800K images, ~8M word instances; single archive of roughly 41 GB with a
  large `gt.mat` annotation file.
- **Annotation format:** MATLAB `gt.mat` with word boxes, character boxes and text per image.
- **Intended task here:** synthetic pretraining / mixing for the recognizer; subsets of 25K / 50K /
  100K / 250K images for the Phase 12 scaling study (brief §10).
- **Preprocessing:** streaming conversion of `gt.mat` (RISK_REGISTER R-04); word boxes grouped into
  lines where geometry allows, otherwise word crops are used and the choice is recorded.
- **For v0.1** ([ADR-0008](docs/decisions/ADR-0008-usable-engine-first.md)) only the subset needed for
  the recognizer mix is extracted; the full scaling study waits for Phase 12.

### GlyphSynth (ours, [ADR-0003](docs/decisions/ADR-0003-document-domain-synthetic-data.md))

- **What:** document-style text lines and pages rendered by `glypharis.data.synth` (Phase 4).
- **Fonts:** open-licence fonts only (e.g. SIL OFL / Apache 2.0 families from Google Fonts); the font list
  with licences is committed in `data/manifests/glyphsynth_fonts.json`.
- **Text corpus:** public-domain or permissively licensed English text (e.g. Project Gutenberg public-domain
  books, Wikipedia under CC BY-SA with attribution); the corpus manifest records source and licence per
  file. GDC source documents are excluded (EXPERIMENT_PROTOCOL §1 rule 5).
- **Degradations:** blur, noise, JPEG, low resolution, contrast, bleed-through, paper texture, skew,
  perspective, at controlled severities.
- **Splits:** GlyphSynth-val uses held-out fonts and held-out texts.
- **Reproducibility:** fully determined by the generator version, config and seed.

### GDC — Glypharis Document Challenge (ours)

- **What:** a small, frozen, manually curated internal test set representing image-only document pages
  (brief §39, under a neutral name).
- **Composition target:** 100–150 pages covering clean scans, old scans, low resolution, photographed,
  rotated, skewed, multi-column, pages with figures and captions, number- and punctuation-heavy pages,
  low contrast.
- **Ground truth:** mainly print-and-capture from born-digital, openly licensed PDFs (exact text and
  line positions from the PDF, aligned to the captured image); a small manually transcribed subset of
  real old scans.
- **Rules:** built in Phase 4, frozen with a hash manifest before Phase 5, never trained on, sources on the
  training exclusion list, evaluated only in Phases 5, 14, 22.

### IIIT5K

- **Source:** IIIT Hyderabad, *IIIT 5K-word* dataset.
- **Citation:** Mishra, Alahari, Jawahar, *Scene Text Recognition using Higher Order Language Priors*,
  BMVC 2012 (`mishra2012iiit5k`).
- **Licence:** **verify** (distributed for research use).
- **Samples (published):** 5,000 cropped word images (2,000 train / 3,000 test).
- **Usage:** test split only, as an out-of-domain word-level recognizer test (RQ6). Its train split is not
  used, so the benchmark stays external.

### FUNSD

- **Source:** `https://guillaumejaume.github.io/FUNSD/`
- **Citation:** Jaume, Ekenel, Thiran, *FUNSD: A Dataset for Form Understanding in Noisy Scanned
  Documents*, ICDAR-OST 2019 (`jaume2019funsd`).
- **Licence:** **verify** (stated for non-commercial research and educational use).
- **Samples (published):** 199 scanned forms (149 train / 50 test), word-level boxes and text.
- **Usage:** external E2E test on noisy scanned forms. Words are grouped into lines by the provided
  entity structure and geometry; the grouping rule is recorded in Phase 14.

### OmniDocBench

- **Source:** OpenDataLab, `https://github.com/opendatalab/OmniDocBench`
- **Citation:** Ouyang et al., *OmniDocBench: Benchmarking Diverse PDF Document Parsing with
  Comprehensive Annotations*, CVPR 2025 (`ouyang2025omnidocbench`).
- **Licence:** **verify**.
- **Samples (published):** roughly 1K PDF page images across textbooks, papers, newspapers, reports and
  other types (count depends on version; pin the version in Phase 2).
- **Usage:** external document-domain test, text blocks only (tables, formulas and figures excluded and
  counted), using its official evaluation code where our output is compatible.

### SROIE (optional)

- **Citation:** Huang et al., *ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction*,
  ICDAR 2019 (`huang2019sroie`).
- **Licence:** **verify**.
- **Samples (published):** 1,000 scanned receipts (626 train / 361 test), line-level text and boxes.
- **Usage:** low-priority external test.

### DocVQA (reference only)

- **Citation:** Mathew, Karatzas, Jawahar, *DocVQA: A Dataset for VQA on Document Images*, WACV 2021
  (`mathew2021docvqa`).
- **Why reference only:** its OCR transcriptions were produced by a commercial OCR engine, not by humans,
  so using them as CER ground truth would measure agreement with another OCR system
  ([ADR-0005](docs/decisions/ADR-0005-evaluation-data-roles.md), RISK_REGISTER R-19). Its images may be
  used for qualitative inspection only.

## Licensing implications

- Several candidate datasets are research-only or share-alike. Model weights trained on them may not be
  releasable under permissive terms. The model card (Phase 21) states the training-data terms, and a
  release that needs permissive terms would train on a permissive subset (e.g. GlyphSynth + permissively
  licensed real data) as a separate, disclosed model.
- GLYPHARIS is used privately by BookWorm during development; any public release of weights goes through
  this section first.

## BookWorm sample documents

Real documents from the consumer application are **not** a dataset of this repository. If the owner
provides sample pages for qualitative checks, they stay outside the repo (and outside git), are not used
for training or tuning, and only aggregate observations are written down.
