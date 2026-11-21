# Error Analysis

Written in Phase 15 and updated in Phase 21. Every number comes from a generated report and carries
its experiment ID: GW-REC-040 (recognition errors), GW-E2E-010 (pipeline errors and the oracle share)
and GW-REC-041 (GlyphSynth validation), computed from the files Phase 14 saved for the frozen v1
system (GW-REC-039 crops, GW-E2E-007 pipeline, GW-E2E-008 oracle; GW-REC-038 on validation). Reports
are in `reports/error_cases/phase15/` and `reports/confusion_analysis/GW-REC-040/`. Phase 9's
validation detector categories come from `reports/error_cases/detector/phase09_adopted/analysis.md`,
GLYPH-C from `reports/robustness/GW-REC-034.json`. Final validation (Phase 22) has been run; its
record is `docs/FINAL_VALIDATION.md`.

<!-- trace: GW-REC-040 reports/error_cases/phase15/results.json#intervals_printed -->

Conventions. Strict track unless stated (IIIT5K on its alphanumeric track). Intervals are 95%
percentile bootstrap with 1,000 resamples, seed 20261010. Page sets resample pages, GDC condition
rows resample source documents, and IIIT5K and GlyphSynth resample lines. The Phase 15 reports print
583 intervals, all descriptive: no correction is applied, and no slice is called worse than another
from two marginal intervals. Slices are marginal and confounded (blur on a resized crop goes with
line height, GDC condition with document). Bin edges are quartiles of the validation lines
(GW-REC-038), fixed before any test row was read. GDC-scan is one document and appears as numbers
only (ADR-0013). HierText is released under CC BY-SA 4.0; its images come from Open Images, which
lists them as CC BY 2.0. Tables derived from it are shared alike.

## 1. Where page error comes from

The frozen pipeline's bag-of-lines errors on a page split into three parts: characters of
ground-truth lines that no prediction matched (missed), errors on matched lines (matched), and
characters of spurious predictions (spurious). The hand-off's pair sums agree with the page files on
all <!-- derived: consistency_pages --> 2,573 pages of the five page sets, OmniDocBench included.
OmniDocBench has no budget, because its reference unit is a layout block.

Strict budgets: HierText test <!-- derived: budget_ht_errors --> 203,171 bag errors
on <!-- derived: budget_ht_pages --> 1,634 pages, GDC <!-- derived: budget_gdc_errors --> 71,886 bag
errors on <!-- derived: budget_gdc_pages --> 106 pages,
GDC-scan <!-- derived: budget_scan_errors --> 1,490 bag errors on <!-- derived: budget_scan_pages --> 28
pages, FUNSD <!-- derived: budget_fu_errors --> 16,074 bag errors
on <!-- derived: budget_fu_pages --> 50 pages.

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md HierText test;missed=L17|HierText test;matched=L18|HierText test;spurious=L19|GDC;missed=L85|GDC;matched=L86|GDC;spurious=L87|GDC-scan;missed=L200|GDC-scan;matched=L201|GDC-scan;spurious=L202|FUNSD;missed=L264|FUNSD;matched=L265|FUNSD;spurious=L266 -->

| Set | Part | Characters | Share % [page CI] | Share % [document CI] |
|---|---|---|---|---|
| HierText test | missed | 100,461 | 49.45 [47.47, 51.56] | n/a |
| HierText test | matched | 52,390 | 25.79 [23.51, 28.00] | n/a |
| HierText test | spurious | 50,320 | 24.77 [23.24, 26.38] | n/a |
| GDC | missed | 32,582 | 45.32 [36.14, 52.80] | 45.32 [33.27, 53.44] |
| GDC | matched | 19,804 | 27.55 [15.80, 42.07] | 27.55 [14.61, 45.81] |
| GDC | spurious | 19,500 | 27.13 [20.89, 33.57] | 27.13 [19.61, 32.97] |
| GDC-scan | missed | 512 | 34.36 [14.87, 42.85] | n/a |
| GDC-scan | matched | 481 | 32.28 [16.36, 71.27] | n/a |
| GDC-scan | spurious | 497 | 33.36 [14.01, 40.79] | n/a |
| FUNSD | missed | 6,465 | 40.22 [36.27, 42.88] | n/a |
| FUNSD | matched | 3,117 | 19.39 [14.64, 27.08] | n/a |
| FUNSD | spurious | 6,492 | 40.39 [36.32, 43.00] | n/a |

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#gdc_test -->

On HierText test and GDC about half of the page error was ground-truth lines that no prediction
matched at IoU 0.5, and about a quarter was misread matched lines. On FUNSD the missed and spurious
parts each held about two fifths. GDC's `rotated` pages differ: matched lines
carried <!-- derived: rotated_matched --> 78.16% [45.80, 97.26] of the errors there (document CI), discussed
in section 2.

### The oracle share (H4)

GW-E2E-008 ran the same recognizer on ground-truth boxes. Its errors divided by the pipeline's, on
the same pages and reference characters, show how much of the page error the recognizer would still
make if detection were perfect (GW-E2E-010).

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/oracle_share.md all pages=L11|clean (scan_app + photo_flat)=L19|the other five=L20|the other five without rotated=L21|scan_app=L18|photo_flat=L16|photo_angled=L15|distant=L13|low_light=L14|curved=L12|rotated=L17 -->

| Slice | Pages | Documents | Share % [page CI] | Share % [document CI] |
|---|---|---|---|---|
| all pages | 106 | 21 | 18.66 [11.55, 28.73] | 18.66 [11.51, 26.45] |
| clean (scan_app + photo_flat) | 53 | 21 | 32.70 [16.64, 51.91] | 32.70 [24.43, 44.01] |
| the other five | 53 | 18 | 10.29 [5.24, 18.03] | 10.29 [5.24, 17.73] |
| the other five without rotated | 46 | 18 | 14.86 [8.46, 23.93] | 14.86 [8.22, 26.43] |
| scan_app | 37 | 19 | 45.14 [25.40, 58.41] | 45.14 [24.95, 58.56] |
| photo_flat | 16 | 13 | 14.75 [5.80, 43.65] | 14.75 [5.53, 43.95] |
| photo_angled | 16 | 10 | 12.81 [9.84, 22.51] | 12.81 [11.29, 19.44] |
| distant | 11 | 10 | 22.09 [8.24, 46.34] | 22.09 [8.57, 50.93] |
| low_light | 11 | 8 | 16.39 [8.99, 40.69] | 16.39 [7.59, 45.03] |
| curved | 8 | 8 | 12.47 [0.71, 35.21] | 12.47 [0.71, 35.21] |
| rotated | 7 | 6 | 1.03 [0.34, 2.93] | 1.03 [0.36, 2.94] |

Over all GDC pages the oracle made <!-- derived: oracle_all_errors --> 13,413 errors against the
pipeline's <!-- derived: oracle_all_pipeline --> 71,886, a share
of <!-- derived: oracle_all_share --> 18.66%. On GDC-scan the share
was <!-- derived: oracle_scan_share --> 22.08% [12.65, 47.58] (information only). The point estimate is
below half in every slice, the clean one included, although the clean slice's page interval
reaches <!-- derived: oracle_clean_high --> 51.91% and
`scan_app`'s <!-- derived: oracle_scanapp_high --> 58.41%. H4 expected the recognizer to account for most page
error on clean pages, and on GDC these numbers do not support that (post hoc). H4 had no registered
operational rule, and the per-condition page CERs published in Phase 14 already pointed this way, so
this is a reading of the numbers, not a test. The share's complement is not a pure detector share
either: it holds missed and poorly boxed lines, spurious predictions and the effect of crop shape.
The same picture held on HierText validation in Phase 10, where ground-truth boxes would have left
about a quarter of the page error (F10).

## 2. Detection failures

### Missed lines by best overlap

The four bins of each set add up to its missed part in section 1 (GW-E2E-010).

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md HierText test;no box=L23|HierText test;barely touched=L24|HierText test;poorly localised=L25|HierText test;its best box went to another line=L26|GDC;no box=L91|GDC;barely touched=L92|GDC;poorly localised=L93|GDC;its best box went to another line=L94|FUNSD;no box=L270|FUNSD;barely touched=L271|FUNSD;poorly localised=L272|FUNSD;its best box went to another line=L273 -->

| Set | Best IoU | Reading | Characters | Lines |
|---|---|---|---|---|
| HierText test | 0 | no box | 45,721 | 10,996 |
| HierText test | 0 to 0.3 | barely touched | 26,616 | 3,988 |
| HierText test | 0.3 to 0.5 | poorly localised | 27,891 | 2,868 |
| HierText test | 0.5 or more | its best box went to another line | 233 | 3 |
| GDC | 0 | no box | 2,508 | 525 |
| GDC | 0 to 0.3 | barely touched | 17,721 | 521 |
| GDC | 0.3 to 0.5 | poorly localised | 12,240 | 284 |
| GDC | 0.5 or more | its best box went to another line | 113 | 1 |
| FUNSD | 0 | no box | 82 | 32 |
| FUNSD | 0 to 0.3 | barely touched | 1,823 | 203 |
| FUNSD | 0.3 to 0.5 | poorly localised | 4,560 | 321 |
| FUNSD | 0.5 or more | its best box went to another line | 0 | 0 |

On HierText test most missed lines got no box at all. On GDC and FUNSD most missed characters sit in
lines that a box overlapped too little. Whether that box was a fragment, a merge over several lines
or a loose box cannot be told from IoU on test.

<!-- trace: GW-DET-001 reports/error_cases/detector/phase09_adopted/analysis.md -->

On validation, Phase 9's analysis of the same detector (GW-DET-001 at the adopted settings) put the
missed lines, fn 21,178 of 74,500 care lines, and the false alarms into these categories:

<!-- trace-table: GW-DET-001 reports/error_cases/detector/phase09_adopted/analysis.md#missed-lines-fn-by-category missed_no_candidate=missed_no_candidate|missed_low_score=missed_low_score|merged=merged|poorly_localised=poorly_localised|partly_found=partly_found|split=split|missed_dropped=missed_dropped -->

| Category | Lines | Share of fn |
|---|---|---|
| missed_no_candidate | 7,809 | 36.9% |
| missed_low_score | 6,107 | 28.8% |
| merged | 4,228 | 20.0% |
| poorly_localised | 1,342 | 6.3% |
| partly_found | 892 | 4.2% |
| split | 747 | 3.5% |
| missed_dropped | 53 | 0.3% |

<!-- trace-table: GW-DET-001 reports/error_cases/detector/phase09_adopted/analysis.md#false-alarms-fp-by-category background=background|fragment=fragment|merge=merge|poorly_localised=poorly_localised|other=other -->

| Category | Predictions | Share of fp |
|---|---|---|
| background | 3,848 | 41.5% |
| fragment | 3,396 | 36.6% |
| merge | 1,164 | 12.6% |
| poorly_localised | 487 | 5.3% |
| other | 374 | 4.0% |

### Recall slices

Recall at IoU 0.5 over pack lines (GW-E2E-010). Cited intervals are by page, and by source document
for the GDC condition rows.

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md HierText test;height under 9.35 px=L61|HierText test;height 9.35 to 13.52 px=L62|HierText test;height 13.52 to 21.485 px=L63|HierText test;height 21.485 px or more=L64|HierText test;rotation under 2°=L70|HierText test;rotation 15° or more=L73|GDC;height under 9.35 px=L155|GDC;height 9.35 to 13.52 px=L156|GDC;height 13.52 to 21.485 px=L157|GDC;height 21.485 px or more=L158|GDC;condition curved=L182|GDC;condition scan_app=L188 -->

| Set | Slice | Lines | Matched | Recall % [cited CI] |
|---|---|---|---|---|
| HierText test | height under 9.35 px | 16,712 | 9,680 | 57.92 [54.64, 61.17] |
| HierText test | height 9.35 to 13.52 px | 16,416 | 12,431 | 75.72 [73.80, 77.73] |
| HierText test | height 13.52 to 21.485 px | 17,684 | 14,388 | 81.36 [79.83, 82.81] |
| HierText test | height 21.485 px or more | 16,762 | 13,220 | 78.87 [77.17, 80.63] |
| HierText test | rotation under 2° | 40,840 | 32,025 | 78.42 [76.66, 80.12] |
| HierText test | rotation 15° or more | 10,256 | 5,080 | 49.53 [45.80, 53.19] |
| GDC | height under 9.35 px | 113 | 0 | 0.00 [0.00, 0.00] |
| GDC | height 9.35 to 13.52 px | 471 | 174 | 36.94 [10.52, 45.00] |
| GDC | height 13.52 to 21.485 px | 2,009 | 1,700 | 84.62 [76.99, 90.34] |
| GDC | height 21.485 px or more | 5,779 | 5,167 | 89.41 [85.93, 92.17] |
| GDC | condition curved | 326 | 226 | 69.33 [58.17, 76.01] |
| GDC | condition scan_app | 2,732 | 2,469 | 90.37 [85.67, 93.84] |

`curved` and `scan_app` have the lowest and highest point estimates of GDC's seven conditions; no
condition is called worse than another.

Phase 9 saw the same pattern on validation at its own inference-scale bins:

<!-- trace-table: GW-DET-001 reports/error_cases/detector/phase09_adopted/analysis.md height 0–8 px=L42|angle 45° or more=L53 -->

| Validation slice | Lines | Recall at IoU 0.5 |
|---|---|---|
| height 0–8 px | 9,761 | 0.340 [0.288, 0.393] |
| angle 45° or more | 5,062 | 0.380 [0.312, 0.445] |

### Cropping error

Matched-line CER by matched IoU (GW-E2E-010). Cited intervals are by page.

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md HierText test;0.5 to 0.7=L47|HierText test;0.7 to 0.85=L48|HierText test;0.85 or more=L49|GDC;0.5 to 0.7=L139|GDC;0.7 to 0.85=L140|GDC;0.85 or more=L141|FUNSD;0.5 to 0.7=L294|FUNSD;0.7 to 0.85=L295|FUNSD;0.85 or more=L296 -->

| Set | Matched IoU | Lines | CER % [cited CI] | Share of the set's errors % |
|---|---|---|---|---|
| HierText test | 0.5 to 0.7 | 7,010 | 29.24 [27.83, 30.64] | 7.91 |
| HierText test | 0.7 to 0.85 | 18,392 | 9.79 [8.88, 10.96] | 9.91 |
| HierText test | 0.85 or more | 24,317 | 3.63 [2.99, 4.49] | 7.97 |
| GDC | 0.5 to 0.7 | 624 | 29.36 [20.83, 38.02] | 7.82 |
| GDC | 0.7 to 0.85 | 2,528 | 7.27 [2.78, 13.31] | 8.69 |
| GDC | 0.85 or more | 3,889 | 4.31 [1.10, 8.76] | 11.04 |
| FUNSD | 0.5 to 0.7 | 1,106 | 9.16 [7.23, 11.43] | 10.10 |
| FUNSD | 0.7 to 0.85 | 860 | 5.61 [3.95, 7.78] | 7.39 |
| FUNSD | 0.85 or more | 125 | 6.40 [3.32, 10.15] | 1.90 |

Paired on the same <!-- derived: od_lines --> 7,041 matched GDC lines, the pipeline's crop
made <!-- derived: od_mean --> 2.048 [0.753, 3.817] (page CI) more errors per line than the ground-truth
crop (sum <!-- derived: od_sum --> 14,421; <!-- derived: od_more --> 1,160 lines
more, <!-- derived: od_equal --> 5,567 equal, <!-- derived: od_fewer --> 314 fewer).

### Rotated pages

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md recall of their lines=L187 -->

| GDC `rotated` pages | Lines | Matched | Recall % [cited CI] |
|---|---|---|---|
| recall of their lines | 402 | 347 | 86.32 [61.83, 98.99] |

GDC's <!-- derived: rotated_pages --> 7 rotated pages were photographed sideways. From ground-truth
crops the recognizer read their lines at <!-- derived: rotated_crop_cer --> 0.82% [0.25, 2.67] CER
(GW-REC-040, document CI; this is a crop CER, not the oracle page CER of F24). Matched lines still
carried <!-- derived: rotated_matched --> 78.16% [45.80, 97.26] of the pipeline's errors there, and
the oracle share was <!-- derived: rotated_oracle --> 1.03% [0.34, 2.93]. The pipeline hands the
recognizer sideways crops because it has no page-orientation step (F24).

### Forms

<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#funsd_test -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#hiertext_test -->
<!-- trace: GW-E2E-010 reports/error_cases/phase15/pipeline.md#gdc_test -->

On FUNSD, <!-- derived: iou_fu --> 47.11
[<!-- derived: iou_fu_low --> 42.48, <!-- derived: iou_fu_high --> 51.39]% of the <!-- derived: iou_fu_matches --> 2,091 matches have a
matched IoU of 0.7 or more, against <!-- derived: iou_ht --> 85.90
[<!-- derived: iou_ht_low --> 85.08, <!-- derived: iou_ht_high --> 86.70]% on HierText test
and <!-- derived: iou_gdc --> 91.14
[<!-- derived: iou_gdc_low --> 88.27, <!-- derived: iou_gdc_high --> 93.44]% on GDC (page CIs). Matched lines
carry <!-- derived: funsd_matched_share --> 19.39% of FUNSD's page error.

<!-- trace-table: GW-E2E-009 reports/external/comparisons.md PP-OCRv6 small=L47 -->

| GLYPHARIS minus system, FUNSD crops | Units | CER (crops or bag of lines) pp |
|---|---|---|
| PP-OCRv6 small | 2,647 | -2.27 [-3.43, -1.27] |

GLYPHARIS had the lowest crop CER on FUNSD in Phase 14 (F22), and the table gives the paired
difference, by line, from the system with the next lowest.

## 3. Reading order

Order-aware minus bag-of-lines errors per page, on GDC (GW-E2E-010, descriptive; the difference is
not an exact decomposition).

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md GW-E2E-007 GLYPHARIS=L339|GW-E2E-008 GLYPHARIS oracle=L341|GW-BASE-001 Tesseract=L329|GW-BASE-002 PP-OCRv5 mobile=L331|GW-BASE-005 PP-OCRv6 tiny=L333|GW-BASE-006 PP-OCRv6 small=L335|GW-BASE-007 PP-OCRv6 medium=L337 -->

| GDC system | Order-aware CER % | Bag CER % | Mean per page [page CI] |
|---|---|---|---|
| GW-E2E-007 GLYPHARIS | 25.37 | 22.35 | +125.509 [−76.729, 333.847] |
| GW-E2E-008 GLYPHARIS oracle | 5.72 | 4.17 | +54.811 [14.556, 109.932] |
| GW-BASE-001 Tesseract | 31.90 | 55.21 | −664.651 [−920.896, −438.867] |
| GW-BASE-002 PP-OCRv5 mobile | 44.74 | 27.83 | +573.019 [215.479, 907.065] |
| GW-BASE-005 PP-OCRv6 tiny | 44.06 | 24.18 | +662.085 [293.739, 992.154] |
| GW-BASE-006 PP-OCRv6 small | 42.86 | 22.29 | +681.642 [320.701, 1022.451] |
| GW-BASE-007 PP-OCRv6 medium | 42.87 | 20.96 | +721.915 [347.764, 1068.379] |

On GDC <!-- derived: inv_count --> 1,010 of <!-- derived: inv_total --> 6,935 consecutive matched
ground-truth lines came out in the opposite order, <!-- derived: inv_share --> 14.56% [9.93, 20.36]
(page CI). The interval of the mean difference lies above zero for the four PaddleOCR systems and
crosses zero for GLYPHARIS. For Tesseract it lies below zero: its bag-of-lines error exceeds its
order-aware error.

On OmniDocBench only the order-aware column means anything. The bag-of-lines CERs beside it are not
meaningful, because OmniDocBench's reference is blocks, not lines, and Phase 14 marks bag of lines
n/a there.

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md GW-E2E-007 GLYPHARIS=L377|GW-BASE-001 Tesseract=L367|GW-BASE-002 PP-OCRv5 mobile=L369|GW-BASE-005 PP-OCRv6 tiny=L371|GW-BASE-006 PP-OCRv6 small=L373|GW-BASE-007 PP-OCRv6 medium=L375 -->

| OmniDocBench system | Order-aware CER % | Bag CER % |
|---|---|---|
| GW-E2E-007 GLYPHARIS | 27.28 | 179.00 |
| GW-BASE-001 Tesseract | 15.70 | 181.39 |
| GW-BASE-002 PP-OCRv5 mobile | 60.52 | 179.12 |
| GW-BASE-005 PP-OCRv6 tiny | 60.62 | 180.23 |
| GW-BASE-006 PP-OCRv6 small | 60.29 | 180.11 |
| GW-BASE-007 PP-OCRv6 medium | 60.35 | 180.20 |

Phase 14 proposed that PaddleOCR's top-to-bottom box sort explains its OmniDocBench order error
(F26). The measure registered for it, read from the output polygons, is the share of consecutive
output lines whose top edge does not rise.

<!-- trace-table: GW-E2E-010 reports/error_cases/phase15/pipeline.md GW-BASE-002 PP-OCRv5 mobile=L386|GW-BASE-005 PP-OCRv6 tiny=L387|GW-BASE-006 PP-OCRv6 small=L388|GW-BASE-007 PP-OCRv6 medium=L389 -->

| System | Pages | Non-rising | Consecutive | Share % [page CI] |
|---|---|---|---|---|
| GW-BASE-002 PP-OCRv5 mobile | 755 | 63,226 | 82,480 | 76.66 [75.42, 77.74] |
| GW-BASE-005 PP-OCRv6 tiny | 755 | 62,467 | 80,994 | 77.13 [75.81, 78.33] |
| GW-BASE-006 PP-OCRv6 small | 755 | 64,123 | 82,753 | 77.49 [76.18, 78.75] |
| GW-BASE-007 PP-OCRv6 medium | 755 | 66,335 | 85,333 | 77.74 [76.50, 78.93] |

PaddleOCR's own sort, read from its installed code (`SortQuadBoxes` in
`paddlex/inference/pipelines/components/common/sort_boxes.py`, used by the OCR pipeline), orders
boxes by the top-left corner's y and then swaps neighbours whose y differs by less than 10 px so that
they run left to right. A following line can therefore sit up to 10 px higher, so under that sort the
share need not be complete. The plan's expectation that a top-to-bottom sort fills it by construction
overlooked that tolerance. The measure cannot tell the sort apart from other orders, so it neither
supports nor refutes F26's cause, which stays unmeasured (F33).

## 4. Recognition failures

### Operations

Ground-truth crops (GW-REC-040). The cited interval is by page, and by line for IIIT5K.
Substitutions, insertions and deletions are Levenshtein's, so they add up to each line's errors.
The HierText validation row is GW-REC-038 on crops, the validation figure that sits beside the test
results.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md HierText test=L17|HierText validation=L863|GDC=L207|GDC-scan=L462|FUNSD=L552|IIIT5K (alphanumeric)=L712 -->

| Set | Lines | CER % [cited CI] | Sub % | Ins % | Del % | Merged words | Split words |
|---|---|---|---|---|---|---|---|
| HierText test | 67,574 | 6.03 [5.60, 6.48] | 3.27 | 0.89 | 1.88 | 2,512 | 1,110 |
| HierText validation | 74,500 | 6.51 [5.98, 7.12] | 3.41 | 0.97 | 2.12 | 2,860 | 1,319 |
| GDC | 8,372 | 4.17 [2.48, 6.22] | 0.86 | 0.12 | 3.19 | 4,941 | 115 |
| GDC-scan | 776 | 1.02 [0.79, 1.30] | n/a | n/a | n/a | n/a | n/a |
| FUNSD | 2,647 | 7.03 [5.59, 8.61] | 2.12 | 1.45 | 3.46 | 640 | 83 |
| IIIT5K (alphanumeric) | 3,000 | 2.37 [2.06, 2.70] | 1.39 | 0.21 | 0.77 | 0 | 0 |

The last column below is the share of substituted word pairs that differ by one character.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md HierText test=L22|HierText validation=L868|GDC=L212|FUNSD=L557|IIIT5K (alphanumeric)=L718 -->

| Set | Ref words | Word errors | WER % | Substituted pairs one character away % |
|---|---|---|---|---|
| HierText test | 146,302 | 28,177 | 19.26 | 52.81 |
| HierText validation | 157,720 | 31,203 | 19.78 | 52.81 |
| GDC | 56,979 | 8,298 | 14.56 | 64.00 |
| FUNSD | 8,729 | 2,259 | 25.88 | 44.74 |
| IIIT5K (alphanumeric) | 3,000 | 256 | 8.53 | 72.62 |

On scene text substitutions are the largest operation, and on GDC and FUNSD deletions.

### Character confusions

The rate is per occurrence of the reference character (`reports/confusion_analysis/GW-REC-040/`).

<!-- trace-table: GW-REC-040 reports/confusion_analysis/GW-REC-040/characters_gdc_test.md space deleted=L83|full stop deleted=L84|’ read as '=L13|“ read as "=L14|” read as "=L15|— read as -=L18|– read as -=L22|§ read as 8=L27|π read as T=L35 -->

| GDC edit | Count | Rate % |
|---|---|---|
| space deleted | 4,941 | 10.165 |
| full stop deleted | 4,153 | 44.176 |
| ’ read as ' | 198 | 98.507 |
| “ read as " | 111 | 99.107 |
| ” read as " | 109 | 97.321 |
| — read as - | 66 | 73.333 |
| – read as - | 46 | 86.792 |
| § read as 8 | 34 | 73.913 |
| π read as T | 20 | 68.966 |

The <!-- derived: space_del_count --> 4,941 space deletions fall
on <!-- derived: space_del_lines --> 583 GDC lines and the <!-- derived: stop_del_count --> 4,153
full-stop deletions on <!-- derived: stop_del_lines --> 391 lines.

<!-- trace: GW-REC-040 reports/error_cases/phase15/recognition.md#gdc_test -->

The 20 worst GDC lines are all table-of-contents lines from two public-domain documents
(`census_income`, `hilbert_geometry`) whose leader dots are dropped. The typographic marks in the
table are read as ASCII and disappear on the normalised track, where GDC's CER is 3.94%. The
out-of-charset symbols § and π are read as look-alike charset characters.

<!-- trace-table: GW-REC-040 reports/confusion_analysis/GW-REC-040/characters_hiertext_test.md space deleted=L85|l read as i=L15|comma read as full stop=L16|a read as e=L17 -->

| HierText test edit | Count | Rate % |
|---|---|---|
| space deleted | 2,512 | 3.191 |
| l read as i | 222 | 1.081 |
| comma read as full stop | 215 | 4.838 |
| a read as e | 215 | 0.582 |

### Character classes

Share of each reference class read as itself, and share deleted (per reference character; class
matrices for GDC and HierText test are PNG files beside the reports).

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md HierText test;lower=L31|HierText test;upper=L32|HierText test;digit=L33|HierText test;punct=L34|HierText test;space=L35|GDC;lower=L221|GDC;upper=L222|GDC;digit=L223|GDC;punct=L224|GDC;space=L225|FUNSD;lower=L566|FUNSD;upper=L567|FUNSD;digit=L568|FUNSD;punct=L569|FUNSD;space=L570 -->

| Set | Class | Correct % | deleted |
|---|---|---|---|
| HierText test | lower | 96.34 | 1.15 |
| HierText test | upper | 93.24 | 2.25 |
| HierText test | digit | 93.20 | 1.75 |
| HierText test | punct | 88.02 | 5.66 |
| HierText test | space | 95.91 | 3.19 |
| GDC | lower | 99.38 | 0.26 |
| GDC | upper | 98.28 | 0.68 |
| GDC | digit | 97.74 | 0.93 |
| GDC | punct | 73.02 | 21.32 |
| GDC | space | 89.61 | 10.17 |
| FUNSD | lower | 95.97 | 1.99 |
| FUNSD | upper | 97.28 | 1.19 |
| FUNSD | digit | 86.54 | 9.56 |
| FUNSD | punct | 92.74 | 4.19 |
| FUNSD | space | 87.65 | 10.52 |

### Slices

CER with its cited interval, then the slice's share of the set's errors.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md HierText test;height < 9.35 px=L47|HierText test;height ≥ 21.485 px=L50|HierText test;blur\, lowest quartile=L56|HierText test;contrast\, lowest quartile=L65|HierText test;rotation 15° or more=L95|HierText test;digit lines=L102|HierText test;all-upper-case lines=L116|HierText test;out of charset=L130|HierText test;confidence\, lowest quartile=L83|GDC;height < 9.35 px=L237|GDC;height ≥ 21.485 px=L240|GDC;blur\, lowest quartile=L246|GDC;contrast\, lowest quartile=L255|GDC;rotation 15° or more=L285|GDC;digit lines=L301|GDC;all-upper-case lines=L315|GDC;out of charset=L329|GDC;confidence\, lowest quartile=L273|FUNSD;height < 9.35 px=L580|FUNSD;height ≥ 21.485 px=L583|FUNSD;digit lines=L630|FUNSD;all-upper-case lines=L644|FUNSD;out of charset=L658|FUNSD;confidence\, lowest quartile=L614 -->

| Set | Slice | CER % [cited CI] | Share of the set's errors % |
|---|---|---|---|
| HierText test | height < 9.35 px | 11.01 [9.90, 12.22] | 30.24 |
| HierText test | height ≥ 21.485 px | 7.05 [6.27, 7.92] | 31.83 |
| HierText test | blur, lowest quartile | 11.53 [10.67, 12.50] | 35.96 |
| HierText test | contrast, lowest quartile | 9.35 [8.49, 10.28] | 33.97 |
| HierText test | rotation 15° or more | 10.82 [9.60, 12.07] | 19.01 |
| HierText test | digit lines | 6.79 [5.85, 7.88] | 22.16 |
| HierText test | all-upper-case lines | 8.61 [7.82, 9.45] | 35.38 |
| HierText test | out of charset | 25.98 [14.37, 39.80] | 2.65 |
| HierText test | confidence, lowest quartile | 24.01 [22.88, 25.25] | 78.51 |
| GDC | height < 9.35 px | 61.06 [29.54, 98.00] | 0.51 |
| GDC | height ≥ 21.485 px | 4.01 [2.19, 6.32] | 84.65 |
| GDC | blur, lowest quartile | 23.78 [16.34, 61.62] | 4.46 |
| GDC | contrast, lowest quartile | 15.44 [6.32, 32.15] | 25.30 |
| GDC | rotation 15° or more | 1.32 [0.24, 3.50] | 1.38 |
| GDC | digit lines | 7.45 [3.98, 11.51] | 81.00 |
| GDC | all-upper-case lines | 42.79 [3.71, 57.26] | 13.20 |
| GDC | out of charset | 16.07 [6.68, 27.47] | 13.62 |
| GDC | confidence, lowest quartile | 27.83 [21.82, 36.66] | 22.79 |
| FUNSD | height < 9.35 px | 21.76 [5.35, 54.19] | 2.81 |
| FUNSD | height ≥ 21.485 px | 27.74 [17.30, 38.51] | 37.62 |
| FUNSD | digit lines | 11.22 [9.77, 12.65] | 44.06 |
| FUNSD | all-upper-case lines | 4.40 [3.20, 6.00] | 20.82 |
| FUNSD | out of charset | 15.89 [8.77, 27.54] | 2.52 |
| FUNSD | confidence, lowest quartile | 28.62 [21.41, 36.84] | 65.42 |

FUNSD has no lines in the lowest blur or contrast quartile, or at a rotation of 15° or more.

Confidence by quartile, with the edges taken from the validation lines:

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md HierText test;lowest=L83|HierText test;highest=L86|GDC;lowest=L273|GDC;second=L274 -->

| Set | Quartile | Lines | CER % [cited CI] | Share of the set's errors % |
|---|---|---|---|---|
| HierText test | lowest | 16,297 | 24.01 [22.88, 25.25] | 78.51 |
| HierText test | highest | 17,185 | 0.39 [0.33, 0.44] | 0.89 |
| GDC | lowest | 588 | 27.83 [21.82, 36.66] | 22.79 |
| GDC | second | 2,121 | 8.33 [4.43, 13.56] | 53.64 |

### GDC by document and condition

Document groups carry no interval.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md book=L347|report=L350|textbook=L351|form=L348|paper=L349 -->

| Group | Lines | Pages | CER % | Share of the set's errors % |
|---|---|---|---|---|
| book | 580 | 20 | 15.98 | 37.46 |
| report | 1,993 | 18 | 7.56 | 28.98 |
| textbook | 520 | 12 | 5.05 | 7.88 |
| form | 1,649 | 12 | 2.96 | 13.99 |
| paper | 3,630 | 44 | 1.02 | 11.70 |

By condition `scan_app` holds <!-- derived: scanapp_share --> 53.40% of the errors
at <!-- derived: scanapp_cer --> 6.55% [2.42, 11.30] (document CI). The paper group, like the book
group, was photographed in every condition. Its lowest and highest condition CERs are below.
Conditions go with documents (the condition by group table of the report), so the condition rows are
not condition effects.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md curved=L367|distant=L368 -->

| Paper group, condition | Lines | Pages | CER % |
|---|---|---|---|
| curved | 88 | 3 | 0.26 |
| distant | 465 | 5 | 1.42 |

### Categories

Every wrong line gets one primary category by a fixed precedence, in the order of this table. Shares
are of the set's character errors, in percent.

| Category | HierText test | HierText validation | GDC | FUNSD | IIIT5K |
|---|---|---|---|---|---|
| out_of_charset | <!-- derived: cat_ht_out_of_charset --> 2.65 | <!-- derived: cat_val_out_of_charset --> 4.20 | <!-- derived: cat_gdc_out_of_charset --> 13.62 | <!-- derived: cat_fu_out_of_charset --> 2.52 | <!-- derived: cat_iiit_out_of_charset --> 0.00 |
| rotation_ge_5 | <!-- derived: cat_ht_rotation_ge_5 --> 34.92 | <!-- derived: cat_val_rotation_ge_5 --> 37.69 | <!-- derived: cat_gdc_rotation_ge_5 --> 1.90 | <!-- derived: cat_fu_rotation_ge_5 --> 0.00 | <!-- derived: cat_iiit_rotation_ge_5 --> 0.00 |
| height_q1 | <!-- derived: cat_ht_height_q1 --> 21.30 | <!-- derived: cat_val_height_q1 --> 18.31 | <!-- derived: cat_gdc_height_q1 --> 0.45 | <!-- derived: cat_fu_height_q1 --> 2.70 | <!-- derived: cat_iiit_height_q1 --> 0.00 |
| blur_q1 | <!-- derived: cat_ht_blur_q1 --> 8.28 | <!-- derived: cat_val_blur_q1 --> 6.96 | <!-- derived: cat_gdc_blur_q1 --> 3.96 | <!-- derived: cat_fu_blur_q1 --> 0.00 | <!-- derived: cat_iiit_blur_q1 --> 0.00 |
| contrast_q1 | <!-- derived: cat_ht_contrast_q1 --> 5.46 | <!-- derived: cat_val_contrast_q1 --> 5.09 | <!-- derived: cat_gdc_contrast_q1 --> 20.41 | <!-- derived: cat_fu_contrast_q1 --> 0.00 | <!-- derived: cat_iiit_contrast_q1 --> 2.76 |
| digit | <!-- derived: cat_ht_digit --> 7.23 | <!-- derived: cat_val_digit --> 5.54 | <!-- derived: cat_gdc_digit --> 49.21 | <!-- derived: cat_fu_digit --> 43.21 | <!-- derived: cat_iiit_digit --> 1.66 |
| punctuation | <!-- derived: cat_ht_punctuation --> 8.48 | <!-- derived: cat_val_punctuation --> 8.11 | <!-- derived: cat_gdc_punctuation --> 9.34 | <!-- derived: cat_fu_punctuation --> 31.32 | <!-- derived: cat_iiit_punctuation --> 0.00 |
| long_word | <!-- derived: cat_ht_long_word --> 1.93 | <!-- derived: cat_val_long_word --> 2.52 | <!-- derived: cat_gdc_long_word --> 0.34 | <!-- derived: cat_fu_long_word --> 11.46 | <!-- derived: cat_iiit_long_word --> 7.73 |
| other | <!-- derived: cat_ht_other --> 9.75 | <!-- derived: cat_val_other --> 11.59 | <!-- derived: cat_gdc_other --> 0.78 | <!-- derived: cat_fu_other --> 8.79 | <!-- derived: cat_iiit_other --> 87.85 |

On HierText, test and validation alike, `rotation_ge_5` and `height_q1` are the two largest
categories, which fits F15's rotation weakness on GLYPH-C. `digit` and `contrast_q1` lead on GDC, and
`digit` and `punctuation` on FUNSD. `digit` here is any wrong line with a digit that meets no earlier
rule, whatever its errors are.

### Line length

ADR-0002's buckets, by strict-normalised reference length.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md HierText validation;0-29=L996|HierText validation;70+=L999|HierText test;0-29=L152|HierText test;70+=L155|GDC;70+=L408 -->

| Set | Length | Lines | CER % | Share of errors % |
|---|---|---|---|---|
| HierText validation | 0-29 | 67,755 | 8.56 | 81.51 |
| HierText validation | 70+ | 714 | 3.22 | 3.59 |
| HierText test | 0-29 | 61,187 | 7.88 | 80.24 |
| HierText test | 70+ | 797 | 3.31 | 4.74 |
| GDC | 70+ | 1,388 | 6.37 | 61.80 |

<!-- trace: GW-REC-040 reports/error_cases/phase15/recognition.md#gdc_test -->

Long lines hold a small share of HierText's errors, so ADR-0002's revisit trigger was not reached.
The 20 worst GDC lines are dot leaders, and the seeded draws show leaders in other documents too, but
no per-line attribute labels them, so their share of the 70+ bucket and of the full-stop deletions
is not measured.

### Annotation noise

<!-- trace: GW-REC-040 reports/error_cases/phase15/recognition.md#hiertext_test -->

Likely label fragments (a very short reference read as a much longer string) are 15 lines on
HierText test, with 145 character errors (0.30% of the set's) and 139 insertions (1.94% of its
insertions). Without them CER is 6.02%.

On validation they are <!-- derived: frag_val_lines --> 15 lines
and <!-- derived: frag_val_share --> 0.19% of the character errors.

## 5. Labelled synthetic slices

The test sets have no labels for touching or broken characters, background noise or fonts.
GW-REC-041 read GlyphSynth v1 validation's 10,000 rendered lines, which carry those labels (intervals
by line).

<!-- trace-table: GW-REC-041 reports/error_cases/phase15/glyphsynth.md all=L11|touching (ink spread)=L74|no ink spread=L73|underline=L47|binarised=L95|inverted=L33|italic=L26|broken (binarisation or low resolution)=L81|background noise (texture, noise, salt and pepper)=L88|neighbour fragments (a cropping stand-in)=L40 -->

| Slice | Lines | CER % [cited CI] | Share of the set's errors % |
|---|---|---|---|
| all | 10,000 | 0.71 [0.63, 0.79] | 100.00 |
| touching (ink spread) | 1,543 | 1.41 [1.17, 1.72] | 31.11 |
| no ink spread | 8,457 | 0.58 [0.50, 0.67] | 68.89 |
| underline | 283 | 1.37 [1.05, 1.76] | 5.16 |
| binarised | 307 | 1.18 [0.81, 1.67] | 4.92 |
| inverted | 318 | 1.06 [0.62, 1.72] | 5.02 |
| italic | 1,203 | 0.86 [0.67, 1.12] | 14.59 |
| broken (binarisation or low resolution) | 2,226 | 0.76 [0.63, 0.90] | 23.02 |
| background noise (texture, noise, salt and pepper) | 6,012 | 0.76 [0.65, 0.89] | 64.32 |
| neighbour fragments (a cropping stand-in) | 3,517 | 0.66 [0.56, 0.78] | 32.69 |

<!-- trace: GW-REC-041 reports/error_cases/phase15/glyphsynth.md -->

Font families run from 0.38% (`worksans`) to 1.48% [1.09, 1.93] (`cormorantgaramond`). By category
serif reads at 0.83%, sans at 0.60% and mono at 0.62%.

<!-- trace: data/manifests/processed/glyph_c.json data/manifests/processed/glyph_c.json -->
<!-- trace: GW-REC-034 reports/robustness/GW-REC-034.json#clean_cer -->
<!-- trace: GW-REC-034 reports/robustness/GW-REC-034.json#mce -->

GLYPH-C gives the corruption side for GW-REC-034, the shipped recognizer. Its 216,000 crops come
from 6,000 samples, <!-- derived: glyphc_per_source --> 3,000 from HierText validation lines
and <!-- derived: glyphc_per_source --> 3,000 from GlyphSynth validation lines. The clean CER of 1.93%
is on that mixed pack, half of it rendered lines, so it is not comparable with a HierText CER. Its
mCE is 3.54%.

<!-- trace-table: GW-REC-034 reports/robustness/GW-REC-034.json#cer_per_corruption_level rotate=rotate|low_resolution=low_resolution|jpeg=jpeg|perspective=perspective|gaussian_noise=gaussian_noise|gaussian_blur=gaussian_blur|contrast=contrast -->

| Corruption | 1 | 5 |
|---|---|---|
| rotate | 2.38% | 23.57% |
| low_resolution | 1.94% | 5.72% |
| jpeg | 1.95% | 4.08% |
| perspective | 2.00% | 3.55% |
| gaussian_noise | 1.99% | 3.12% |
| gaussian_blur | 1.93% | 2.63% |
| contrast | 1.93% | 2.31% |

The columns are corruption levels 1 and 5. Both sources are synthetic or derived from validation, and
rendered lines transfer poorly to photographed text (F16), so they describe the recognizer, not the
test sets.

## 6. What the baselines share

The Phase 5 baselines read the same crops, scored with the GLYPHARIS charset classes (rates per
reference character, no interval). TrOCR-small-printed (GW-BASE-004) is included.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md GW-REC-039 GLYPHARIS v1=L414|GW-BASE-001 Tesseract 5=L416|GW-BASE-002 PP-OCRv5 mobile=L418|GW-BASE-003 PP-OCRv5 server=L420|GW-BASE-004 TrOCR-small-printed=L422|GW-BASE-005 PP-OCRv6 tiny=L424|GW-BASE-006 PP-OCRv6 small=L426|GW-BASE-007 PP-OCRv6 medium=L428 -->

| GDC system | CER % | Sub % | Del % | punct correct % | space correct % |
|---|---|---|---|---|---|
| GW-REC-039 GLYPHARIS v1 | 4.17 | 0.86 | 3.19 | 73.02 | 89.61 |
| GW-BASE-001 Tesseract 5 | 6.42 | 2.69 | 3.05 | 71.42 | 90.69 |
| GW-BASE-002 PP-OCRv5 mobile | 3.68 | 1.08 | 2.44 | 88.63 | 88.54 |
| GW-BASE-003 PP-OCRv5 server | 11.24 | 1.13 | 10.03 | 67.62 | 59.68 |
| GW-BASE-004 TrOCR-small-printed | 72.34 | 66.86 | 3.97 | 82.12 | 88.33 |
| GW-BASE-005 PP-OCRv6 tiny | 4.00 | 1.29 | 2.55 | 87.83 | 87.11 |
| GW-BASE-006 PP-OCRv6 small | 3.46 | 0.80 | 2.53 | 86.90 | 87.73 |
| GW-BASE-007 PP-OCRv6 medium | 3.20 | 0.77 | 2.29 | 89.95 | 88.14 |

<!-- trace: GW-REC-040 reports/error_cases/phase15/recognition.md#gdc_test -->

TrOCR-small-printed's strict row reflects letter case: <!-- derived: trocr_lower_as_upper --> 95.70%
of GDC's lower-case reference characters come out as upper case, and its normalised CER is 13.07%.

Paired crop CER differences on the same GDC lines, GLYPHARIS minus each system (by line, GW-E2E-009):

<!-- trace-table: GW-E2E-009 reports/external/comparisons.md GW-BASE-001 Tesseract 5=L17|GW-BASE-002 PP-OCRv5 mobile=L19|GW-BASE-003 PP-OCRv5 server=L21|GW-BASE-004 TrOCR-small-printed=L23|GW-BASE-005 PP-OCRv6 tiny=L25|GW-BASE-006 PP-OCRv6 small=L27|GW-BASE-007 PP-OCRv6 medium=L29 -->

| Baseline | Units | CER (crops or bag of lines) pp |
|---|---|---|
| GW-BASE-001 Tesseract 5 | 8,372 | -2.25 [-2.50, -1.98] |
| GW-BASE-002 PP-OCRv5 mobile | 8,372 | +0.49 [+0.23, +0.78] |
| GW-BASE-003 PP-OCRv5 server | 8,372 | -7.06 [-7.42, -6.73] |
| GW-BASE-004 TrOCR-small-printed | 8,372 | -68.16 [-69.10, -67.29] |
| GW-BASE-005 PP-OCRv6 tiny | 8,372 | +0.18 [-0.04, +0.40] |
| GW-BASE-006 PP-OCRv6 small | 8,372 | +0.71 [+0.53, +0.91] |
| GW-BASE-007 PP-OCRv6 medium | 8,372 | +0.97 [+0.76, +1.20] |

GLYPHARIS's crop CER on GDC is above that of PP-OCRv5 mobile, PP-OCRv6 small and PP-OCRv6 medium
(each interval lies above zero), and the interval against PP-OCRv6 tiny includes zero.

<!-- trace: GW-REC-040 reports/error_cases/phase15/recognition.md#gdc_test -->

Every system but TrOCR-small-printed has a higher deletion rate than substitution rate on GDC, and
every system but PP-OCRv5 server reads between <!-- derived: space_lowest --> 87.11% (PP-OCRv6 tiny)
and <!-- derived: space_highest --> 90.69% (Tesseract) of the reference spaces as spaces. GLYPHARIS
reads <!-- derived: punct_glypharis --> 73.02% of reference punctuation as itself, PP-OCRv5 mobile
and PP-OCRv6 <!-- derived: punct_lowest_ppocr --> 86.90
to <!-- derived: punct_highest_ppocr --> 89.95%, and
Tesseract <!-- derived: punct_tesseract --> 71.42%. PP-OCRv5 mobile and PP-OCRv6 also read some out-of-charset
characters (<!-- derived: oov_lowest --> 26.97 to <!-- derived: oov_highest --> 43.33% on the strict
track), which GLYPHARIS cannot by construction. The class rates have no interval, so no difference
between systems is stated from them.

<!-- trace-table: GW-REC-040 reports/error_cases/phase15/recognition.md GW-REC-039 GLYPHARIS v1=L163|GW-BASE-006 PP-OCRv6 small=L173 -->

| HierText test system | CER % | Del % | digit correct % | punct correct % |
|---|---|---|---|---|
| GW-REC-039 GLYPHARIS v1 | 6.03 | 1.88 | 93.20 | 88.02 |
| GW-BASE-006 PP-OCRv6 small | 7.06 | 2.72 | 93.57 | 87.14 |

<!-- trace-table: GW-E2E-009 reports/external/comparisons.md GLYPHARIS minus PP-OCRv6 small=L15 -->

| HierText test crops, paired by line | Units | CER (crops or bag of lines) pp |
|---|---|---|
| GLYPHARIS minus PP-OCRv6 small | 67,574 | -1.03 [-1.14, -0.92] |

PP-OCRv6 small has the lowest crop CER point estimate of the baselines run on HierText test. The
class and deletion rates in the first table carry no interval, so no difference is stated from them.

## 7. Limits

<!-- trace: GW-REC-040 reports/error_cases/phase15/README.md#notes -->

- GDC is <!-- derived: gdc_pages --> 106 pages from <!-- derived: gdc_documents --> 21 documents,
  GDC-scan one document and FUNSD <!-- derived: budget_fu_pages --> 50 forms. The document-cluster
  intervals with <!-- derived: rotated_documents --> 6 to <!-- derived: gdc_documents --> 21
  clusters (the per-condition ones fewest) cover less than nominal, and conditions go with
  documents, so no per-condition contrast is drawn.
- Merges and splits cannot be told apart from IoU on test; Phase 9's validation categories stand
  in for them, from the same detector.
- The oracle share is not a pure detector share, and H4 is read post hoc.
- Categories follow one fixed precedence; a line's category says which rule it met first, not
  what caused its errors. The Levenshtein alignment is one of several optimal ones, so a confusion
  inside a run of repeated characters (such as leader dots) is arbitrary.
- The blur attribute is measured on 64-px crops, so it mixes source resolution with optical blur.
- GlyphSynth and GLYPH-C are synthetic or derived from validation.
- The OmniDocBench bag-of-lines numbers in the reading-order table are not meaningful (block
  reference). The registered F26 measure did not account for PaddleOCR's 10 px row tolerance, so it
  does not decide F26's cause, and the cause of PaddleOCR's order error there is still open.
- Nothing here flows back into v1; a model that uses these findings is a new version, evaluated
  again.
