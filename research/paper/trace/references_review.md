# Bibliography review log

Review of `research/paper/references.bib`, one row per entry. All entries were checked on 2026-11-18.
The Phase 0 file was written from memory; every field below was compared with the source named in
the row, and fields that could not be confirmed were left out.

How to read the source column:

- `crossref:<DOI>` is the metadata the publisher deposited with Crossref for that DOI, read at
  `https://api.crossref.org/works/<DOI>`.
- arXiv rows are the arXiv abstract page. PMLR, NeurIPS, ICLR and project pages are named by URL.

Access limits met during the review. The pages of CVF Open Access, SpringerLink, OpenReview, DBLP,
ojs.aaai.org, PubMed Central and a university library catalogue refused automated access (HTTP 403,
a sign-in redirect or a browser check). None of these checks was worked around. Crossref, arXiv
and the conference sites stood in for them, and the rows say where a field rests on something
weaker than a publisher record. Pages for IEEE conference papers are the IEEE proceedings pages
deposited with the DOI; CVF's open-access copies are numbered differently.

| Key | Source checked | Date | What changed |
|---|---|---|---|
| `graves2006ctc` | crossref:10.1145/1143844.1143891 | 2026-11-18 | Added full proceedings name, pages, publisher, DOI. |
| `shi2017crnn` | crossref:10.1109/TPAMI.2016.2646371 | 2026-11-18 | Added volume, number, pages, DOI. |
| `shi2019aster` | crossref:10.1109/TPAMI.2018.2848939 | 2026-11-18 | Added volume, number, pages, DOI. |
| `fang2021abinet` | crossref:10.1109/CVPR46437.2021.00702 | 2026-11-18 | Added IEEE pages, publisher, DOI; proceedings name as deposited. |
| `bautista2022parseq` | crossref:10.1007/978-3-031-19815-1_11; book record crossref:10.1007/978-3-031-19815-1 (Part XXVIII, ISBN 9783031198144); bookseller listing for that ISBN gives LNCS 13688 | 2026-11-18 | Added part, series, volume, pages, publisher, DOI. The volume number rests on the ISBN-matched listing because SpringerLink refused access; the ECVA file name `136880177.pdf` agrees. |
| `du2022svtr` | crossref:10.24963/ijcai.2022/124 | 2026-11-18 | Added full proceedings name, pages, DOI. |
| `li2023trocr` | crossref:10.1609/aaai.v37i11.26538 | 2026-11-18 | Changed to `@article` as deposited; added volume, number, pages, DOI. |
| `zhou2017east` | crossref:10.1109/CVPR.2017.283 | 2026-11-18 | Added pages, publisher, DOI. |
| `baek2019craft` | crossref:10.1109/CVPR.2019.00959 | 2026-11-18 | Added pages, publisher, DOI. |
| `wang2019psenet` | crossref:10.1109/CVPR.2019.00956 | 2026-11-18 | Added pages, publisher, DOI. |
| `liao2020dbnet` | crossref:10.1609/aaai.v34i07.6812 | 2026-11-18 | Changed to `@article` as deposited; added volume, number, pages, DOI. |
| `liao2022dbnetpp` | crossref:10.1109/TPAMI.2022.3155612 | 2026-11-18 | Year corrected from 2022 to 2023; added volume, number, pages, DOI. Key unchanged because keys are frozen. |
| `howard2019mobilenetv3` | crossref:10.1109/ICCV.2019.00140; https://arxiv.org/abs/1905.02244 | 2026-11-18 | Added pages, publisher, DOI. Author order kept as on the arXiv paper; the Crossref record lists the same twelve people in a different order. CVF page refused access. |
| `lin2017fpn` | crossref:10.1109/CVPR.2017.106 | 2026-11-18 | New entry. The record spells the second author Dollar; the accented form is kept. |
| `smith2007tesseract` | crossref:10.1109/ICDAR.2007.4376991 | 2026-11-18 | Author changed to the initial given in the record; added proceedings name, volume 2, pages, publisher, DOI. |
| `du2020ppocr` | https://arxiv.org/abs/2009.09941 | 2026-11-18 | Changed from `@article` to `@misc` with eprint, class and arXiv DOI. No journal version is listed. |
| `cui2025paddleocr` | https://arxiv.org/abs/2507.05595; https://www.paddleocr.ai/main/en/papers.html | 2026-11-18 | New entry. |
| `paddleocr_ppocrv6` | https://arxiv.org/abs/2606.13108; https://www.paddleocr.ai/main/en/papers.html | 2026-11-18 | New entry. A citable report exists, so the fallback to the pinned model pages was not needed. |
| `doctr` | https://raw.githubusercontent.com/mindee/doctr/main/README.md | 2026-11-18 | New entry, using the citation block in the project's README. |
| `easyocr` | https://github.com/JaidedAI/EasyOCR; https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/README.md | 2026-11-18 | New entry. The project gives no citation or author list, so the owner organisation is named and no year is given. |
| `kim2022donut` | crossref:10.1007/978-3-031-19815-1_29; book record as for `bautista2022parseq` | 2026-11-18 | Added part, series, volume, pages, publisher, DOI. Same basis for the volume number as `bautista2022parseq`. |
| `blecher2023nougat` | https://arxiv.org/abs/2308.13418 | 2026-11-18 | Changed from `@article` to `@misc` with eprint, class `cs.LG` and arXiv DOI. |
| `gupta2016synthetic` | crossref:10.1109/CVPR.2016.254 | 2026-11-18 | Added pages, publisher, DOI. |
| `long2022hiertext` | crossref:10.1109/CVPR52688.2022.00112 | 2026-11-18 | Added IEEE pages, publisher, DOI. |
| `kuznetsova2020openimages` | crossref:10.1007/s11263-020-01316-z; https://arxiv.org/abs/1811.00982 | 2026-11-18 | New entry. |
| `russakovsky2015imagenet` | crossref:10.1007/s11263-015-0816-y | 2026-11-18 | New entry. |
| `mishra2012iiit5k` | crossref:10.5244/C.26.127 | 2026-11-18 | Added pages, publisher, DOI. |
| `jaume2019funsd` | crossref:10.1109/ICDARW.2019.10029 | 2026-11-18 | Workshop name corrected to ICDARW; added pages, publisher, DOI. |
| `harley2015rvlcdip` | crossref:10.1109/ICDAR.2015.7333910 | 2026-11-18 | New entry. |
| `huang2019sroie` | crossref:10.1109/ICDAR.2019.00244 | 2026-11-18 | Added pages, publisher, DOI. |
| `mathew2021docvqa` | crossref:10.1109/WACV48630.2021.00225 | 2026-11-18 | Added pages, publisher, DOI. |
| `ouyang2025omnidocbench` | crossref:10.1109/CVPR52734.2025.02313; https://arxiv.org/abs/2412.07626 | 2026-11-18 | Replaced "and others" with all 20 authors; added pages, publisher, DOI. |
| `rae2020compressive` | https://iclr.cc/virtual_2020/poster_SylKikSYDH.html; https://arxiv.org/abs/1911.05507 | 2026-11-18 | Added the missing author Chloe Hillier, who is on the ICLR page; the arXiv listing read here shows four authors. Added the OpenReview link given by the ICLR page. |
| `groleau2023augraphy` | crossref:10.1007/978-3-031-41682-8_24; book record crossref:10.1007/978-3-031-41682-8 (Part III, ISBN 9783031416811); bookseller listings for that ISBN give LNCS 14189 | 2026-11-18 | Added part, series, volume, pages, publisher, DOI. The volume number rests on ISBN-matched listings because SpringerLink refused access. |
| `hendrycks2019benchmarking` | https://iclr.cc/virtual/2019/poster/731; https://arxiv.org/abs/1903.12261 | 2026-11-18 | Added the OpenReview link given by the ICLR page. |
| `levenshtein1966` | https://xlinux.nist.gov/dads/HTML/Levenshtein.html; https://mathnet.ru/eng/dan31411 | 2026-11-18 | New entry. NIST gives both the 1966 English translation and the 1965 original; Math-Net confirms the original. The publisher's own page for the translation was not reached. |
| `guo2017calibration` | https://proceedings.mlr.press/v70/guo17a.html | 2026-11-18 | Added proceedings name, editors, series, volume, pages, publisher, URL. |
| `naeini2015ece` | crossref:10.1609/aaai.v29i1.9602; PubMed Central record PMC4410090, read through a search result | 2026-11-18 | New entry. Volume, number and DOI are from the deposit, which has no pages; pages 2901-2907 are from the PMC citation line as shown in the search result, because the PMC page refused access. |
| `efron1994bootstrap` | crossref:10.1007/978-1-4899-4541-9; crossref:10.1201/9780429246593; library catalogue records read through a search result | 2026-11-18 | Year changed from 1994 to 1993 and publisher to Chapman & Hall, the first edition; 1994 is the Chapman and Hall/CRC listing. Added series, volume 57 and DOI. Key unchanged because keys are frozen. |
| `nagy1992xycut` | crossref:10.1109/2.144436 | 2026-11-18 | New entry. |
| `loshchilov2019adamw` | https://iclr.cc/virtual/2019/poster/935; https://arxiv.org/abs/1711.05101 | 2026-11-18 | New entry. |
| `smith2019onecycle` | crossref:10.1117/12.2520589; contents and title pages of Proceedings of SPIE volume 11006, https://www.proceedings.com/content/050/050447webtoc.pdf | 2026-11-18 | New entry. Article number 1100612 is from the contents page (paper 11006 12); the deposit's page field, 36, is the conference paper number and was not used. |
| `paszke2019pytorch` | https://papers.nips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html and its BibTeX file | 2026-11-18 | New entry. Pages are left out because the publisher's BibTeX has an empty pages field. |
| `onnxruntime` | https://onnxruntime.ai | 2026-11-18 | New entry. The page names Microsoft as copyright holder and gives no citation format, so no year or version is stated. The GitHub repository pages returned 404 to the fetch tool and were not used. |

## Open points

- Two LNCS volume numbers and the pages of `naeini2015ece` rest on secondary records, as the rows
  say. They should be compared with the publisher pages by someone with browser access.
- `research/findings.md` and other documents were not changed by this review.
