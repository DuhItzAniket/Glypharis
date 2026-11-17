# Error Analysis — paper section draft

Draft from Phase 15 (2026-11-17). Every number is copied from a generated report and carries its
experiment ID: GW-REC-040 (recognition errors), GW-E2E-010 (pipeline errors and the oracle share)
and GW-REC-041 (GlyphSynth validation), computed from the files Phase 14 saved for the frozen v1
system (GW-REC-039 crops, GW-E2E-007 pipeline, GW-E2E-008 oracle; GW-REC-038 on validation).
Reports: `reports/error_cases/phase15/` (README.md, recognition.md, pipeline.md, oracle_share.md,
examples.md, glyphsynth.md, results.json) and `reports/confusion_analysis/GW-REC-040/`. Phase 9's
validation detector categories come from `reports/error_cases/detector/phase09_adopted/analysis.md`,
GLYPH-C from `reports/robustness/GW-REC-034.json`.

Conventions: strict track unless stated (IIIT5K on its alphanumeric track); 95% percentile
bootstrap, 1,000 resamples, seed 20261010; page sets resample pages, GDC condition rows source
documents, IIIT5K and GlyphSynth lines. All 583 printed intervals are descriptive: no correction is
applied, and no slice is called worse than another from two marginal intervals. Slices are marginal
and confounded (blur on a resized crop goes with line height, GDC condition with document). Bin
edges are quartiles of the validation lines (GW-REC-038), fixed before any test row was read.
GDC-scan is one document and appears as numbers only (ADR-0013). HierText-derived numbers: HierText
annotations CC BY-SA 4.0, images CC BY 2.0; tables derived from them are shared alike.

## 1. Where page error comes from

The frozen pipeline's bag-of-lines errors on a page split into three parts: characters of
ground-truth lines that no prediction matched, errors on matched lines, and characters of spurious
predictions (GW-E2E-010, `pipeline.md`; the hand-off's pair sums agree with the page files on all
2,573 pages of the five page sets, OmniDocBench included).

| Set | Pages | Bag errors | Missed lines % | Matched lines % | Spurious % |
|---|---|---|---|---|---|
| HierText test | 1,634 | 203,171 | 49.45 [47.47, 51.56] | 25.79 [23.51, 28.00] | 24.77 [23.24, 26.38] |
| GDC | 106 | 71,886 | 45.32 [36.14, 52.80] | 27.55 [15.80, 42.07] | 27.13 [20.89, 33.57] |
| GDC-scan | 28 | 1,490 | 34.36 [14.87, 42.85] | 32.28 [16.36, 71.27] | 33.36 [14.01, 40.79] |
| FUNSD | 50 | 16,074 | 40.22 [36.27, 42.88] | 19.39 [14.64, 27.08] | 40.39 [36.32, 43.00] |

Page CIs; GDC with source documents as the unit: 45.32 [33.27, 53.44], 27.55 [14.61, 45.81],
27.13 [19.61, 32.97]. OmniDocBench has no budget, because its reference unit is a layout block.

On HierText test and GDC about half of the page error is ground-truth lines that no prediction
matched at IoU 0.5, and a quarter is misread matched lines. FUNSD's forms lose about as much to
spurious predictions as to missed lines. The exception is GDC's `rotated` pages, where matched lines
carry 78.16% [45.80, 97.26] of the errors (document CI), discussed in section 2.

**The oracle share (H4).** GW-E2E-008 runs the same recognizer on ground-truth boxes. Its errors
over the pipeline's, on the same pages and reference characters, say how much of the page error
the recognizer would still make if detection were perfect (GW-E2E-010, `oracle_share.md`):

| GDC slice | Pages | Documents | Share % [page CI] | Share % [document CI] |
|---|---|---|---|---|
| all pages | 106 | 21 | 18.66 [11.55, 28.73] | 18.66 [11.51, 26.45] |
| clean (`scan_app` + `photo_flat`) | 53 | 21 | 32.70 [16.64, 51.91] | 32.70 [24.43, 44.01] |
| the other five conditions | 53 | 18 | 10.29 [5.24, 18.03] | 10.29 [5.24, 17.73] |
| the other five without `rotated` | 46 | 18 | 14.86 [8.46, 23.93] | 14.86 [8.22, 26.43] |
| `scan_app` | 37 | 19 | 45.14 [25.40, 58.41] | 45.14 [24.95, 58.56] |
| `photo_flat` | 16 | 13 | 14.75 [5.80, 43.65] | 14.75 [5.53, 43.95] |
| `photo_angled` | 16 | 10 | 12.81 [9.84, 22.51] | 12.81 [11.29, 19.44] |
| `distant` | 11 | 10 | 22.09 [8.24, 46.34] | 22.09 [8.57, 50.93] |
| `low_light` | 11 | 8 | 16.39 [8.99, 40.69] | 16.39 [7.59, 45.03] |
| `curved` | 8 | 8 | 12.47 [0.71, 35.21] | 12.47 [0.71, 35.21] |
| `rotated` | 7 | 6 | 1.03 [0.34, 2.93] | 1.03 [0.36, 2.94] |

From ground-truth boxes the recognizer makes 18.66% as many errors as the whole pipeline on GDC, and
22.08% [12.65, 47.58] on GDC-scan (information only). The point estimate is below half in every
slice, the clean one included, although the clean slice's page interval reaches 51.91% and
`scan_app`'s 58.41%. H4 expected the recognizer to account for most page error on clean pages; it is
**not supported on GDC (post hoc)**. H4 had no registered operational rule, and the per-condition
page CERs published in Phase 14 already pointed this way, so this is a reading of the numbers, not a
test. The share's complement is not a pure detector share either: it holds missed and poorly boxed
lines, spurious predictions and the effect of crop shape. The same picture held on HierText
validation in Phase 10, where ground-truth boxes would have left about a quarter of the page error
(F10).

## 2. Detection failures

**Missed lines by best overlap** (GW-E2E-010, `pipeline.md`). The four bins add up to the missed
part above.

| Set | No box (IoU 0) | Barely touched (0, 0.3) | Poorly localised [0.3, 0.5) | Best box went to another line (≥ 0.5) |
|---|---|---|---|---|
| HierText test | 45,721 chars, 10,996 lines | 26,616, 3,988 | 27,891, 2,868 | 233, 3 |
| GDC | 2,508, 525 | 17,721, 521 | 12,240, 284 | 113, 1 |
| FUNSD | 82, 32 | 1,823, 203 | 4,560, 321 | 0, 0 |

On HierText test most missed lines got no box at all. On GDC and FUNSD most missed characters sit in
lines that a box overlapped too little; whether that box was a fragment, a merge over several lines
or a loose box, IoU alone cannot tell on test. On validation, Phase 9's analysis of the same
detector (GW-DET-001 at the adopted settings, 21,178 missed lines of 74,500) put 36.9% of misses at
no candidate, 28.8% at a candidate scored below the box threshold, 20.0% merged with a neighbour,
6.3% poorly localised, 4.2% partly found and 3.5% split; of its false alarms 41.5% were background
and 36.6% fragments.

**Recall slices** (GW-E2E-010, `pipeline.md`; IoU 0.5, pack lines). On HierText test recall is
57.92% [54.64, 61.17] for lines under 9.35 px high and 75.72 to 81.36% for the three taller bins; by
rotation it is 78.42% for |θ| < 2° and 49.53% [45.80, 53.19] for |θ| ≥ 15°. On GDC none of the 113
lines under 9.35 px is found and 36.94% [10.52, 45.00] of the 471 lines from 9.35 to 13.52 px,
against 89.41% for lines of 21.485 px or more. By GDC condition recall runs from
69.33% [58.17, 76.01] (`curved`) to 90.37% [85.67, 93.84] (`scan_app`), document CIs. Phase 9 saw
the same on validation: recall 0.340 for lines 0 to 8 px high and 0.380 at 45° or more, at its own
inference-scale bins.

**Cropping error** (GW-E2E-010, `pipeline.md`). On HierText test and GDC the matched-line CER falls
with box quality: on HierText test 29.24% [27.83, 30.64] at IoU [0.5, 0.7), 9.79% at [0.7, 0.85) and
3.63% [2.99, 4.49] at 0.85 or more; on GDC 29.36%, 7.27% and 4.31%. Paired on the same 7,041 matched
GDC lines, the pipeline's crop makes 2.048 [0.753, 3.817] (page CI) more errors per line than the
ground-truth crop (sum +14,421; 1,160 lines more, 5,567 equal, 314 fewer).

**Rotated pages.** GDC's 7 rotated pages are photographed sideways. The detector finds 86.32%
[61.83, 98.99] of their lines, and from ground-truth crops the recognizer reads those lines at 0.82%
[0.25, 2.67] CER (GW-REC-040, document CI), yet matched lines carry 78.16% of the pipeline's errors
there and the oracle share is 1.03%. The pipeline hands the recognizer sideways crops because it has
no page-orientation step (F24).

**Forms.** FUNSD's boxes are loose: 47.11% [42.48, 51.39] of its matches reach IoU 0.7 (HierText
test 85.90%, GDC 91.14%), and 1,106 of its 2,091 matches fall in [0.5, 0.7). With missed lines at
40.22% and spurious predictions at 40.39% of its page error, matched lines carry only 19.39%,
although the recognizer reads FUNSD's crops best of all systems run (Phase 14, F22).

## 3. Reading order

Order-aware minus bag-of-lines errors per page (GW-E2E-010, `pipeline.md`, descriptive; the
difference is not an exact decomposition):

| GDC system | Order-aware CER % | Bag CER % | Mean difference per page [page CI] |
|---|---|---|---|
| GW-E2E-007 GLYPHARIS | 25.37 | 22.35 | +125.509 [−76.729, 333.847] |
| GW-E2E-008 GLYPHARIS oracle | 5.72 | 4.17 | +54.811 [14.556, 109.932] |
| GW-BASE-001 Tesseract | 31.90 | 55.21 | −664.651 [−920.896, −438.867] |
| GW-BASE-002 PP-OCRv5 mobile | 44.74 | 27.83 | +573.019 [215.479, 907.065] |
| GW-BASE-005 PP-OCRv6 tiny | 44.06 | 24.18 | +662.085 [293.739, 992.154] |
| GW-BASE-006 PP-OCRv6 small | 42.86 | 22.29 | +681.642 [320.701, 1022.451] |
| GW-BASE-007 PP-OCRv6 medium | 42.87 | 20.96 | +721.915 [347.764, 1068.379] |

On GDC 1,010 of 6,935 consecutive matched ground-truth lines come out in the opposite order,
14.56% [9.93, 20.36]. The mean difference is +573.019 to +721.915 errors per page for the PaddleOCR
systems and +125.509 for GLYPHARIS, whose interval crosses zero. Tesseract's bag-of-lines error
exceeds its order-aware error. On OmniDocBench only the order-aware rows mean anything: GLYPHARIS
27.28%, Tesseract 15.70%, the PaddleOCR systems 60.29 to 60.62%. The bag-of-lines CERs printed
beside them (about 179 to 181%) are not meaningful, because OmniDocBench's reference is blocks, not
lines (Phase 14 marks bag of lines n/a there).

Phase 14 proposed that PaddleOCR's top-to-bottom box sort explains its OmniDocBench order error
(F26). The measure registered for it, read from the output polygons (GW-E2E-010), is the share of
consecutive output lines whose top edge does not rise: 76.66 to 77.74% for the four PaddleOCR
systems. PaddleOCR's own sort, read from its installed code (`SortQuadBoxes` in
`paddlex/inference/pipelines/components/common/sort_boxes.py`, used by the OCR pipeline), orders
boxes by the top-left corner's y and then swaps neighbours whose y differs by less than 10 px so
that they run left to right. A following line can therefore sit up to 10 px higher, and the measure
would not be 1 under that sort; the plan's expectation that a top-to-bottom sort makes it 1 by
construction overlooked that tolerance. The measure cannot tell the sort apart from other orders, so
it neither supports nor refutes F26's cause, which stays unmeasured (F33).

## 4. Recognition failures

**Operations** (GW-REC-040, `recognition.md`; ground-truth crops, page CIs, IIIT5K line CI).
Substitutions, insertions and deletions are Levenshtein's, so they add up to each line's errors.

| Set | Lines | CER % [CI] | Sub % | Ins % | Del % | Merged words | Split words | WER % | One character away % |
|---|---|---|---|---|---|---|---|---|---|
| HierText test | 67,574 | 6.03 [5.60, 6.48] | 3.27 | 0.89 | 1.88 | 2,512 | 1,110 | 19.26 | 52.81 |
| HierText validation | 74,500 | 6.51 [5.98, 7.12] | 3.41 | 0.97 | 2.12 | 2,860 | 1,319 | 19.78 | 52.81 |
| GDC | 8,372 | 4.17 [2.48, 6.22] | 0.86 | 0.12 | 3.19 | 4,941 | 115 | 14.56 | 64.00 |
| GDC-scan | 776 | 1.02 [0.79, 1.30] | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| FUNSD | 2,647 | 7.03 [5.59, 8.61] | 2.12 | 1.45 | 3.46 | 640 | 83 | 25.88 | 44.74 |
| IIIT5K (alphanumeric) | 3,000 | 2.37 [2.06, 2.70] | 1.39 | 0.21 | 0.77 | 0 | 0 | 8.53 | 72.62 |

"One character away" is the share of substituted word pairs that differ by one character. On scene
text substitutions are the largest operation, on GDC and FUNSD deletions.

**Character confusions** (`reports/confusion_analysis/GW-REC-040/`). On GDC the two largest
operations are deletions: spaces 4,941 times (10.165% of reference spaces) on 583 lines, and full
stops 4,153 times (44.176% of reference full stops) on 391 lines (`examples.md`). The 20 worst GDC
lines are all table-of-contents lines from two public-domain documents (`census_income`,
`hilbert_geometry`) whose leader dots are dropped. The next largest are typographic marks read as
ASCII: ’ to ' 198 times (98.507%), “ and ” to " 111 and 109 times, — and – to - 66 and 46 times.
These disappear on the normalised track, where GDC's CER is 3.94%. Out-of-charset symbols are read
as a look-alike charset character (§ to 8, 73.913%; π to T, 68.966%). On HierText test the
largest substitutions are look-alike letters and punctuation (l to i 222 times, comma to full stop
215, a to e 215), and the largest deletion is the space (2,512, 3.191%).

**Character classes.** Share of each reference class read as itself (class matrices, PNG for GDC
and HierText test):

| Set | lower | upper | digit | punct | space |
|---|---|---|---|---|---|
| HierText test | 96.34 | 93.24 | 93.20 | 88.02 | 95.91 |
| GDC | 99.38 | 98.28 | 97.74 | 73.02 | 89.61 |
| FUNSD | 95.97 | 97.28 | 86.54 | 92.74 | 87.65 |

GDC's punctuation is 21.32% deleted, its spaces 10.17%; FUNSD deletes 9.56% of its digits and
10.52% of its spaces.

**Slices** (CER and share of the set's errors; cited CI):

| Slice | HierText test | GDC | FUNSD |
|---|---|---|---|
| height < 9.35 px | 11.01 [9.90, 12.22], 30.24% | 61.06 [29.54, 98.00], 0.51% | 21.76 [5.35, 54.19], 2.81% |
| height ≥ 21.485 px | 7.05 [6.27, 7.92], 31.83% | 4.01 [2.19, 6.32], 84.65% | 27.74 [17.30, 38.51], 37.62% |
| blur, lowest quartile | 11.53 [10.67, 12.50], 35.96% | 23.78 [16.34, 61.62], 4.46% | no lines |
| contrast, lowest quartile | 9.35 [8.49, 10.28], 33.97% | 15.44 [6.32, 32.15], 25.30% | no lines |
| \|θ\| ≥ 15° | 10.82 [9.60, 12.07], 19.01% | 1.32 [0.24, 3.50], 1.38% | no lines |
| digit lines | 6.79 [5.85, 7.88], 22.16% | 7.45 [3.98, 11.51], 81.00% | 11.22 [9.77, 12.65], 44.06% |
| all-upper-case lines | 8.61 [7.82, 9.45], 35.38% | 42.79 [3.71, 57.26], 13.20% | 4.40 [3.20, 6.00], 20.82% |
| out of charset | 25.98 [14.37, 39.80], 2.65% | 16.07 [6.68, 27.47], 13.62% | 15.89 [8.77, 27.54], 2.52% |
| confidence, lowest quartile | 24.01 [22.88, 25.25], 78.51% | 27.83 [21.82, 36.66], 22.79% | 28.62 [21.41, 36.84], 65.42% |

Low confidence picks out most recognition errors on scene text: the lowest quartile (below
0.626682) holds 78.51% of HierText test's errors (validation 78.90%), the top quartile 0.89% at
0.39% CER. On GDC the two lowest quartiles hold 22.79% and 53.64%.

**GDC by document and condition** (no CI inside groups). Book pages read at 15.98% (37.46% of the
errors), reports 7.56%, textbooks 5.05%, forms 2.96% and papers 1.02%. By condition `scan_app` holds
53.40% of the errors at 6.55% [2.42, 11.30] (document CI). Inside the paper group, which (like the
book group) was photographed in every condition, the condition CERs run from 0.26 to 1.42%.
Conditions go with documents (the condition × group table), so the condition rows are not condition
effects.

**Categories.** Every wrong line gets one primary category by a fixed precedence (out of charset,
rotation ≥ 5°, height, blur and contrast in the lowest quartile, digit, punctuation, long word,
other). Shares of the set's character errors:

| Category | HierText test | HierText validation | GDC | FUNSD | IIIT5K |
|---|---|---|---|---|---|
| out of charset | 2.65 | 4.20 | 13.62 | 2.52 | 0.00 |
| rotation ≥ 5° | 34.92 | 37.69 | 1.90 | 0.00 | 0.00 |
| height, lowest quartile | 21.30 | 18.31 | 0.45 | 2.70 | 0.00 |
| blur, lowest quartile | 8.28 | 6.96 | 3.96 | 0.00 | 0.00 |
| contrast, lowest quartile | 5.46 | 5.09 | 20.41 | 0.00 | 2.76 |
| digit | 7.23 | 5.54 | 49.21 | 43.21 | 1.66 |
| punctuation | 8.48 | 8.11 | 9.34 | 31.32 | 0.00 |
| long word | 1.93 | 2.52 | 0.34 | 11.46 | 7.73 |
| other | 9.75 | 11.59 | 0.78 | 8.79 | 87.85 |

Scene text fails on geometry: rotation and small lines are HierText's two largest categories, on
test as on validation, which fits F15's rotation weakness on GLYPH-C. On the documents digit lines
and low-contrast lines lead on GDC, digit and punctuation lines on FUNSD. "Digit" here is any wrong
line with a digit that meets no earlier rule, whatever its errors are.

**Line length** (ADR-0002's buckets). On validation, lines of 0 to 29 characters carry 81.51% of the
errors and lines of 70 or more 3.59% (HierText test 80.24% and 4.74%), so ADR-0002's revisit trigger
is not met. GDC's 70+ bucket holds 61.80% of its errors at 6.37% CER. The 20 worst GDC lines are dot
leaders, and the seeded draws show leaders in other documents too, but no per-line attribute labels
them, so their share of the 70+ bucket and of the full-stop deletions is not measured.

**Annotation noise.** Likely label fragments (a reference of at most 2 characters read as at
least 3 × ref + 3) are 15 lines on HierText test, 145 errors (0.30%) and 139 insertions (1.94% of
insertions); without them CER is 6.02%. Validation: 15 lines, 0.19% of the errors. Label noise of
this kind is a small part of the error.

## 5. Labelled synthetic slices

The test sets have no labels for touching or broken characters, background noise or fonts.
GW-REC-041 reads GlyphSynth v1 validation's 10,000 rendered lines, which carry those labels
(`glyphsynth.md`, line CI):

| Slice | Lines | CER % [CI] | Share of errors % |
|---|---|---|---|
| all | 10,000 | 0.71 [0.63, 0.79] | 100.00 |
| touching (ink spread) | 1,543 | 1.41 [1.17, 1.72] | 31.11 |
| no ink spread | 8,457 | 0.58 [0.50, 0.67] | 68.89 |
| underline | 283 | 1.37 [1.05, 1.76] | 5.16 |
| binarised | 307 | 1.18 [0.81, 1.67] | 4.92 |
| inverted | 318 | 1.06 [0.62, 1.72] | 5.02 |
| italic | 1,203 | 0.86 [0.67, 1.12] | 14.59 |
| broken (binarisation, low resolution) | 2,226 | 0.76 [0.63, 0.90] | 23.02 |
| background noise (texture, noise, salt and pepper) | 6,012 | 0.76 [0.65, 0.89] | 64.32 |
| neighbour fragments (a cropping stand-in) | 3,517 | 0.66 [0.56, 0.78] | 32.69 |

Font families run from 0.38% (`worksans`) to 1.48% [1.09, 1.93] (`cormorantgaramond`); serif
0.83%, sans 0.60%, mono 0.62%. No labelled slice reaches 1.5% CER.

GLYPH-C (GW-REC-034, the shipped recognizer; 216,000 crops derived from validation lines) gives the
corruption side; the JSON stores CERs as fractions. Clean crops read at 0.0193. In-box rotation
rises from 0.0238 at level 1 to 0.2357 at level 5; no other corruption exceeds 0.0572 (low
resolution at level 5), with JPEG 0.0408, Gaussian noise 0.0312, perspective 0.0355, blur 0.0263 and
contrast 0.0231 at level 5. Its mCE is 3.54% (`reports/ablation/phase13_final.md`). Both sources are
synthetic or derived from validation, and rendered lines transfer poorly to photographed text (F16),
so they describe the recognizer, not the test sets.

## 6. What the baselines share

The Phase 5 baselines read the same crops, scored with the GLYPHARIS charset classes
(`recognition.md`; rates per reference character, no CI).

| GDC system | CER % | Sub % | Del % | punct read as itself % | space read as itself % |
|---|---|---|---|---|---|
| GW-REC-039 GLYPHARIS v1 | 4.17 | 0.86 | 3.19 | 73.02 | 89.61 |
| GW-BASE-001 Tesseract 5 | 6.42 | 2.69 | 3.05 | 71.42 | 90.69 |
| GW-BASE-002 PP-OCRv5 mobile | 3.68 | 1.08 | 2.44 | 88.63 | 88.54 |
| GW-BASE-003 PP-OCRv5 server | 11.24 | 1.13 | 10.03 | 67.62 | 59.68 |
| GW-BASE-005 PP-OCRv6 tiny | 4.00 | 1.29 | 2.55 | 87.83 | 87.11 |
| GW-BASE-006 PP-OCRv6 small | 3.46 | 0.80 | 2.53 | 86.90 | 87.73 |
| GW-BASE-007 PP-OCRv6 medium | 3.20 | 0.77 | 2.29 | 89.95 | 88.14 |

Every system in the table deletes more than it substitutes on GDC, and every one but PP-OCRv5 server
reads 87.11 to 90.69% of the reference spaces as spaces: merged words are shared. Punctuation is
where GLYPHARIS differs from PP-OCRv5 mobile and PP-OCRv6 (73.02% against 86.90 to 89.95%), and
Tesseract is close to it (71.42%). PP-OCRv5 mobile and PP-OCRv6 also read some out-of-charset
characters (26.97 to 43.33% on the strict track), which GLYPHARIS cannot by construction. On
HierText test GLYPHARIS reads more lower-case, upper-case, punctuation and space characters as
themselves than PP-OCRv6 small, the strongest baseline run there (punctuation 88.02% against
87.14%), and fewer digits (93.20% against 93.57%), with deletion rates of 1.88% against 2.72%.

## 7. Limits

- GDC is 106 pages from 21 documents, GDC-scan one document and FUNSD 50 forms. The document-cluster
  intervals with 6 to 21 clusters (the per-condition ones fewest) cover less than nominal, and
  conditions go with documents, so no per-condition contrast is drawn.
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
