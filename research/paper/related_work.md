# Related work

The longer Phase 0 survey is in `research/approach_survey.md`. Citation keys refer to
`references.bib`, where every entry has been checked (`trace/references_review.md`). Published
results are not repeated here. A system is compared with GLYPHARIS only where it was run under the
project's protocol, and those results are in `results.md`.

## Text detection

Scene-text detectors fall into two broad groups. Regression methods predict word or line geometry
directly; EAST (`zhou2017east`) does this from a fully convolutional network with little
post-processing. Segmentation methods label text pixels and recover instances afterwards. CRAFT
(`baek2019craft`) predicts character regions and the affinity between neighbouring characters, and
PSENet (`wang2019psenet`) grows small text kernels outward to keep close instances apart.

DBNet (`liao2020dbnet`) makes the binarisation threshold part of the network through a
differentiable approximation, which allows a light backbone and simple post-processing. DBNet++
(`liao2022dbnetpp`) adds adaptive fusion of features across scales. The GLYPHARIS detector follows
DBNet with a feature-pyramid neck (`lin2017fpn`) and a MobileNetV3-Large backbone
(`howard2019mobilenetv3`), trained on lines instead of words. It does not use the DBNet++ fusion
module.

## Text recognition

CTC (`graves2006ctc`) trains a per-frame classifier from unsegmented labels. CRNN (`shi2017crnn`)
joined a convolutional feature extractor and a recurrent layer to it, and is the project's baseline
recognizer design. ASTER (`shi2019aster`) rectifies the input and decodes with attention. ABINet
(`fang2021abinet`) and PARSeq (`bautista2022parseq`) add language modelling to the decoder, by
iterative correction and by permuted autoregressive decoding respectively. TrOCR (`li2023trocr`)
pairs a pre-trained transformer image encoder with a text decoder. SVTR (`du2022svtr`) removes the
sequence model: it mixes local and global information across image patches and decodes with CTC.
The GLYPHARIS recognizer is SVTR-style, with greedy CTC decoding and no language model.

## OCR systems and page-level models

Tesseract (`smith2007tesseract`) is the long-standing open engine. The cited overview predates its
LSTM recognizer; the baseline run here is Tesseract 5 with that recognizer (GW-BASE-001). PaddleOCR's
PP-OCR series (`du2020ppocr`, `cui2025paddleocr`, `paddleocr_ppocrv6`) is the closest design to
GLYPHARIS: a detector, an orientation step and a CTC recognizer, in several sizes. Its PP-OCRv5 and
PP-OCRv6 models are the main baselines (GW-BASE-002, 003, 005, 006, 007). docTR (`doctr`) and EasyOCR
(`easyocr`) are other open two-stage toolkits. They were considered in Phase 0 and were not run.

Page-level models read a document image without a separate detection stage. Donut (`kim2022donut`)
maps a page directly to structured output, and Nougat (`blecher2023nougat`) does so for academic
documents. These models handle layout that a line pipeline does not, but they are large for the
hardware here and do not return line boxes by default, so they were out of scope (ADR-0001).

Reading order in GLYPHARIS is a recursive XY-cut after deskewing, a classical layout method
(`nagy1992xycut`). It uses no learned layout model.

## Data

| Dataset | Reference | Use in this project |
|---|---|---|
| HierText | `long2022hiertext`; images from Open Images, `kuznetsova2020openimages` | Training, validation and one held-out test set |
| ImageNet | `russakovsky2015imagenet` | Initial weights of the detector backbone only |
| PG-19 | `rae2020compressive` | Text source for GlyphSynth, the project's rendered lines |
| FUNSD | `jaume2019funsd`; forms drawn from RVL-CDIP, `harley2015rvlcdip` | External test set, evaluation only |
| IIIT5K | `mishra2012iiit5k` | External test set of word images, evaluation only |
| OmniDocBench | `ouyang2025omnidocbench` | External test set with its official evaluator, evaluation only |
| SynthText | `gupta2016synthetic` | Not used: its licence does not allow the intended use (ADR-0009) |
| SROIE | `huang2019sroie` | Not used: optional, needs an account, skipped by the owner |
| DocVQA | `mathew2021docvqa` | Not used: its transcriptions come from a commercial OCR engine (ADR-0005) |

GDC and GDC-scan, the project's own test sets, have no prior publication; they are described in
`DATASETS.md`.

## Augmentation and robustness

Augraphy (`groleau2023augraphy`) is a library of document degradation pipelines. It was not used;
the project wrote its own transforms. GLYPH-C, the project's corruption suite for line images,
follows the structure of ImageNet-C (`hendrycks2019benchmarking`): a fixed set of corruptions at
several severities, summarised by a mean error.

## Metrics and statistics

CER and WER rest on edit distance (`levenshtein1966`). Line confidence is calibrated with temperature
scaling (`guo2017calibration`) and judged by expected calibration error, with the binning idea going
back to Bayesian binning (`naeini2015ece`). Intervals are bootstrap intervals
(`efron1994bootstrap`).

## Training and software

Training uses AdamW (`loshchilov2019adamw`) with a one-cycle learning-rate schedule for the
recognizer (`smith2019onecycle`), in PyTorch (`paszke2019pytorch`). The export path uses ONNX Runtime
(`onnxruntime`).

## Position of this work

The earlier notes said that few works report a laptop-scale training recipe with diagnostics, and
that scene-text and document-style synthetic data are rarely compared at equal budget. No systematic
review backs the first statement, and the second comparison was not carried out here (ADR-0009), so
neither is claimed. The methods in GLYPHARIS come from the work above. What this project adds is a
recorded evaluation of one compact engine on held-out document sets, beside the published toolkits,
with its unfavourable results kept.
