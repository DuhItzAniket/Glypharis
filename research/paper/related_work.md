# Related Work — paper section notes

Draft structure for the paper's related-work section. The longer working survey is in
`research/approach_survey.md`. Every claim about another system must cite it; no numbers from other
papers are compared with ours unless reproduced under our protocol.

## Text detection
- Regression-based detectors: EAST [`zhou2017east`].
- Segmentation-based detectors: PSENet [`wang2019psenet`], CRAFT [`baek2019craft`], DBNet and DBNet++
  [`liao2020dbnet`, `liao2022dbnetpp`]. GLYPHARIS-Det follows DBNet with a MobileNetV3 backbone
  [`howard2019mobilenetv3`].

## Text recognition
- CTC [`graves2006ctc`] and CRNN [`shi2017crnn`] as the classic lightweight recipe.
- Attention and rectification: ASTER [`shi2019aster`].
- Language-aware decoders: ABINet [`fang2021abinet`], PARSeq [`bautista2022parseq`].
- Transformer recognizers: TrOCR [`li2023trocr`]; single-visual-model CTC recognizers: SVTR
  [`du2022svtr`].

## OCR systems and document parsing
- Tesseract [`smith2007tesseract`]; PP-OCR [`du2020ppocr`] as the production two-stage toolkit family.
- OCR-free / page-level models: Donut [`kim2022donut`], Nougat [`blecher2023nougat`]; position GLYPHARIS
  as the small, inspectable alternative for constrained hardware.

## Data
- Scene text: SynthText [`gupta2016synthetic`], HierText [`long2022hiertext`], IIIT5K [`mishra2012iiit5k`].
- Documents: FUNSD [`jaume2019funsd`], SROIE [`huang2019sroie`], DocVQA [`mathew2021docvqa`],
  OmniDocBench [`ouyang2025omnidocbench`].
- Document augmentation: Augraphy [`groleau2023augraphy`]; robustness benchmarking in the style of
  ImageNet-C [`hendrycks2019benchmarking`].

## Evaluation methodology
- Calibration: temperature scaling [`guo2017calibration`].
- Bootstrap confidence intervals [`efron1994bootstrap`].

## Gap to state (only if the evidence supports it)
- Few works report a complete laptop-scale training recipe with diagnostics and controlled ablations.
- The value of scene-text vs document-style synthetic data for document OCR is rarely compared at equal
  budget.
