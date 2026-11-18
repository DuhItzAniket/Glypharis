# Efficiency — paper section draft

Draft from Phase 16 (2026-11-18), for RQ7 and the latency side of RQ4. Every number is copied from
a generated report and carries its experiment ID: GW-EFF-006 (the fp32 ONNX bundle and its checks),
GW-EFF-007 (what v1 costs), GW-EFF-008 (INT8, H7′), GW-EFF-009 (accuracy against latency),
GW-EFF-010 (H1′ ONNX, with the Phase 16 amendment to GW-BASE-002) and GW-EFF-011 (H6′). Reports:
`reports/efficiency/phase16/` (README.md, cost_table.md, int8.md, pareto.md, pareto.png,
verdicts.md, results.json) and `reports/benchmark_results/efficiency_comparison.md`, written by
`scripts/phase16_score.py` at `1253dfe`.

Conventions: validation data only (HierText validation); no test set was read. One laptop: Intel
Core i7-14650HX, NVIDIA GeForce RTX 4050 Laptop GPU; torch 2.14.1+cu126, onnxruntime 1.31.1. Mains
power, Balanced plan, power mode "best performance", recorded at the start and end of every profile;
no sitting was void. 10 warm-up and 100 timed items, one process per run; CPU runs at 10 threads
(torch intra and inter 10, each ONNX Runtime session intra-op 10, sequential) unless stated. Arms
compared in one sitting for H7′, H6′ and the Pareto points (GW-EFF-008, 009, 011) run in the order
1 … k, then k … 1, and each item's time is the mean of its two runs; GW-EFF-010 follows Phase 14's
order (the baseline, then the bundle, one run each), and GW-EFF-007's rows are one run each. Means
carry a 95% bootstrap interval over the timed items (1,000 resamples); p95 is a point value. Page
time is measured on the 110 profile pages (about 41 lines per page), page CER and E2E F1 on the 400
Phase 5 pages (strict), line CER on 5,000 seeded validation lines (strict, greedy). Of the 95
printed intervals, 4 are confirmatory (H7′ latency, H7′ accuracy, H6′, H1′ ONNX) and 91 descriptive;
only the four carry a verdict. v1 is the frozen build evaluated in Phase 14; every other point is a
new version, not v1. HierText-derived numbers: HierText annotations CC BY-SA 4.0, images CC BY 2.0;
tables derived from them are shared alike.

## 1. The same model in another runtime

v1's plain fp32 ONNX export (TorchScript exporter, as in Phase 17) was checked against outputs
written once from the PyTorch v1 build on the CPU (GW-EFF-006, `README.md`; bundle manifest SHA-256
`875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`):

| Check | Limit | Measured |
|---|---|---|
| detector map, max abs difference (110 pages) | 1e-4 | 7.33e-06 |
| recognizer logits, max abs difference (5,000 lines) | 1e-3 | 3.09e-04 |
| text (5,000 lines) | no differences except near ties on at most 0.1% | 0 differences, 0 near ties |
| page CER (400 pages), as a fraction | 0.0005 | 1.49e-05 |

All four pass, so the bundle is the same model and the ONNX results below describe v1.

## 2. What v1 costs

Parameters: detector 3,187,209, recognizer 7,928,835, total 11,116,044. Files: detector 12.44 MiB
(.pt) and 12.15 MiB (fp32 ONNX), recognizer 30.31 and 30.40 MiB (GW-EFF-007, `cost_table.md`).

| Build | Line mean ms [CI] | Line p95 | Page mean ms [CI] | Page p95 | Peak RAM on pages MiB |
|---|---|---|---|---|---|
| PyTorch, CPU, 10 threads | 33.90 [29.92, 38.13] | 77.60 | 2,566 [2,327, 2,829] | 5,127 | 1,641.8 |
| fp32 ONNX, CPU, 10 threads | 92.25 [70.69, 116.28] | 304.39 | 8,574 [7,516, 9,766] | 19,011 | 6,529.6 |
| fp32 ONNX, CPU, default threads | 104.73 [79.87, 132.05] | 392.09 | 9,925 [8,744, 11,261] | 22,072 | 6,541.5 |
| PyTorch, GPU (lines at batch 1) | 12.22 [11.24, 13.33] | 19.01 | 1,049 [825, 1,300] | 3,288 | 2,450.0 |

Each row is one run in chain order, not a mirrored pair; the mirrored comparison of the two CPU
builds is GW-EFF-009's (§3: 35.55 against 95.59 ms per line, 2,637.3 against 9,739.4 ms per page).
On the GPU a batch of 32 lines takes 199.89 [187.57, 213.04] ms (160.1 lines/s); peak VRAM by
torch's counter is 77.3 MiB at batch 1, 282.2 MiB at batch 32 and 432.4 MiB on pages. The exported
model is the same model but about 3.3 times slower per page on the CPU than PyTorch, and needs
about four times the RAM.

**Where the page time goes** (mean ms per page, `cost_table.md`; the stages sum to the page time):

| Build | det network | rec network | Page mean |
|---|---|---|---|
| PyTorch, CPU, 10 threads | 296.4 (12%) | 2,148.2 (84%) | 2,565.7 |
| fp32 ONNX, CPU, 10 threads | 297.5 (3%) | 8,167.7 (95%) | 8,573.7 |
| PyTorch, GPU | 694.5 (66%) | 261.0 (25%) | 1,049.1 |

The other seven stages (image to RGB, detector preprocessing and post-processing, cropping,
recognizer batch build, decoding, reading order) are each at most 43.6 ms on the CPU at 10 threads.
The exported detector runs at PyTorch's speed; the recognizer network alone separates the two CPU
builds. This corrects an earlier side measurement in which the exported detector ran twice as fast,
taken with ONNX Runtime's memory arena on (F13, revised; F34). Page time grows by 28.6 [19.2, 42.9]
ms per line in PyTorch and 72.6 [32.9, 142.7] ms per line in fp32 ONNX (least squares, page
bootstrap; 40.9 lines per page).

## 3. Why the export is slow

The plain exporter writes each global mixer's attention as separate MatMul, Softmax and masking
nodes, where PyTorch runs a fused kernel. Exporting the same weights with each mixer as one
`com.microsoft.MultiHeadAttention` node (`glypharis/efficiency/fused_attention.py`) gives the same
page CER (24.32%), E2E F1 (60.50%) and line CER (6.22%) as v1, and the times in the GW-EFF-009
sittings (`pareto.md`; differences against v1's fp32 ONNX build, descriptive):

| Point | Page mean ms | Δ page ms [CI] | Line mean ms | Page peak RAM MiB |
|---|---|---|---|---|
| v1 PyTorch | 2,637.3 | −7,102.1 [−8,195.7, −6,115.5] | 35.55 | 1,665.0 |
| v1 fp32 ONNX | 9,739.4 | n/a | 95.59 | 6,525.5 |
| fused attention (new version) | 4,020.8 | −5,718.6 [−6,630.9, −4,883.7] | 39.53 | 2,393.0 |
| INT8 recognizer (new version) | 6,890.9 | −2,848.5 [−3,233.6, −2,505.1] | 63.35 | 6,896.5 |

Fusing the attention removes most of the gap to PyTorch at v1's page CER, F1 and line text (F35);
it is still slower than PyTorch per page and per line.

## 4. INT8 (H7′)

Dynamic INT8 of the v1 fp32 ONNX recognizer, by the registered recipe (ONNX Runtime's
`quant_pre_process`, then `quantize_dynamic` on MatMuls with a constant weight, QInt8, per tensor),
turns 49 of the 67 MatMuls into MatMulInteger; 18 stay fp32, the attention products among them,
which have no constant input (GW-EFF-008, `int8.md`). The file shrinks from 30.40 to 9.82 MiB.

| Half | Rule (met if the interval lies below 0) | Interval | Verdict |
|---|---|---|---|
| latency | mean(t_INT8 − 0.75 × t_fp32) per line | −10.00 [−12.79, −7.37] ms | met |
| accuracy | paired ratio difference of errors_INT8 − 0.005 × ref_chars against errors_fp32 | −0.512 [−0.541, −0.484] points | met |

Lines take 61.84 ms with INT8 against 95.78 ms in fp32 (two runs averaged), at strict line CER
6.208% against 6.221%. H7′ holds (F36). On pages (descriptive) the page CER difference is +0.00
[−0.02, +0.02] points and the time difference −2,848.5 [−3,233.6, −2,505.1] ms. The attention
products stay fp32, and INT8 removes less page time than fusing the attention (§3).

## 5. Accuracy against latency

Page points (GW-EFF-009, `pareto.md`, `pareto.png`; every row but v1 is a new version, not v1,
measured on validation only; fp32 ONNX at 10 threads unless the point says otherwise; Δ against
v1's fp32 ONNX build on the same items, descriptive):

| Point | Page CER % | E2E F1 % | Page mean ms | Page p95 ms | Peak RAM MiB | Δ CER points [CI] | Δ ms [CI] |
|---|---|---|---|---|---|---|---|
| v1 fp32 ONNX | 24.32 [20.87, 28.15] | 60.50 [56.81, 63.96] | 9,739.4 | 22,474.1 | 6,525.5 | n/a | n/a |
| detector long side 1024 | 31.62 [27.74, 35.79] | 55.95 | 8,491.3 | 20,331.8 | 5,983.6 | +7.29 [+5.55, +9.51] | −1,248.1 [−1,761.3, −734.3] |
| detector long side 1280 | 26.63 [23.18, 30.38] | 58.98 | 9,537.2 | 22,553.2 | 5,963.5 | +2.31 [+1.66, +2.98] | −202.2 [−451.2, +32.6] |
| detector long side 1792 | 24.21 [20.38, 28.30] | 60.67 | 10,168.1 | 22,592.5 | 6,576.8 | −0.11 [−0.93, +0.92] | +428.7 [+230.7, +647.8] |
| detector long side 2048 | 23.75 [19.91, 28.17] | 61.10 | 10,400.0 | 21,713.0 | 6,556.8 | −0.57 [−1.80, +0.83] | +660.6 [+393.3, +949.6] |
| smaller detector GW-DET-003 | 27.33 [23.60, 31.43] | 57.75 | 9,065.7 | 20,756.4 | 6,107.0 | +3.00 [+1.52, +4.33] | −673.7 [−888.5, −443.8] |
| ONNX detector, PyTorch recognizer | 24.32 [20.87, 28.15] | 60.50 | 2,579.1 | 5,186.9 | 1,550.0 | +0.00 [+0.00, +0.00] | −7,160.4 [−8,258.3, −6,171.7] |

Detector input size and the smaller detector move page accuracy but change the ONNX page time by
at most about 13%; the recognizer sets the page time (F37). The lowest mean page time among the
points at v1's accuracy is that of the ONNX detector with the PyTorch recognizer, a new version at
v1's page CER and F1 (about 58 ms below v1 PyTorch's 2,637.3 ms in the same sittings; no interval
for that difference is reported; F38).

Recognizer points (GW-EFF-009, `pareto.md`; every row but v1 is a new version, not v1; 5,000
lines for CER, 100 lines for time, one line at a time):

| Point | Line CER % [CI] | Line mean ms [CI] | Line p95 ms |
|---|---|---|---|
| v1 PyTorch | 6.22 [5.73, 6.71] | 35.55 [31.32, 40.10] | 80.82 |
| v1 fp32 ONNX | 6.22 [5.73, 6.71] | 95.59 [73.55, 120.41] | 340.04 |
| GW-REC-003 (CRNN) | 8.34 [7.74, 8.92] | 9.07 [7.97, 10.24] | 21.15 |
| GW-REC-006 | 7.85 [7.29, 8.41] | 16.67 [14.39, 19.07] | 42.32 |
| GW-REC-020 (SVTR-style) | 8.21 [7.61, 8.78] | 29.03 [23.07, 35.60] | 97.18 |
| GW-REC-022 | 7.57 [7.00, 8.14] | 100.57 [77.01, 126.77] | 387.76 |

The recognizer points differ in recipe and training budget, not only in size, so they are not a
size ladder.

## 6. The registered comparisons (H6′, H1′ ONNX)

| Test | Rule (met if the interval lies below 0) | Interval | Verdict |
|---|---|---|---|
| H6′ (GW-EFF-011) | mean(t_SVTR − 1.5 × t_CRNN) per line, fp32 ONNX | +16.78 [+12.02, +22.04] ms | not met |
| H1′ ONNX (GW-EFF-010) | mean per-page time(v1 fp32 ONNX) − time(GW-BASE-002) | +6,537.6 [+5,383.7, +7,806.3] ms | not met |

H6′: GW-REC-020 takes 32.01 ms per line in fp32 ONNX against the CRNN GW-REC-003's 10.16 ms; both
exports first passed their logit and text checks on the 5,000 lines. In PyTorch at 10 threads
(information, not part of the test) the two take 15.70 and 13.00 ms (F39). The plain export's
unfused attention (§3) is the likely reason the ONNX gap is wider than PyTorch's; this was not
measured for GW-REC-020.

H1′ ONNX: the v1 fp32 ONNX bundle takes 9,623.6 ms per page against 3,086.0 ms for a fresh
PP-OCRv5 mobile profile in the same sitting, by Phase 14's rule (F40). The verdict applies to the
bundle hash above only. H1′ itself holds from Phase 14, whose latency half is the PyTorch build
(GW-EFF-005).

## 7. Against the CPU targets

ADR-0011 set targets for the CPU build: line mean ≤ 25 ms and p95 ≤ 70 ms, page mean ≤ 1,500 ms
and p95 ≤ 2,700 ms, peak RAM on pages < 1,836 MiB (`README.md`, GW-EFF-007):

| Build | Line mean | Line p95 | Page mean | Page p95 | Peak RAM |
|---|---|---|---|---|---|
| v1 PyTorch, CPU, 10 threads | 33.9: over | 77.6: over | 2,565.7: over | 5,127.1: over | 1,641.8: within |
| v1 fp32 ONNX, CPU, 10 threads | 92.2: over | 304.4: over | 8,573.7: over | 19,011.5: over | 6,529.6: over |
| v1 fp32 ONNX, CPU, default threads | 104.7: over | 392.1: over | 9,924.9: over | 22,071.7: over | 6,541.5: over |

No CPU build of v1 meets the latency targets; only the PyTorch build's page RAM is within (F38). No
variant in §3 or §5 meets the page-time targets either; the ONNX detector with the PyTorch
recognizer is within the RAM target only. On the GPU, outside ADR-0011's scope, PyTorch reads a page
in 1,049 ms mean (GW-EFF-007). Choosing a default backend or shipping a variant is the owner's
decision (ADR-0011); Phase 16 changes no default.

For context only, the Phase 5 baselines' page times on the same pages are in
`reports/benchmark_results/efficiency_comparison.md`; they were measured in different sittings and
no difference is stated from them. The only like-for-like baseline comparison is H1′ ONNX above.

## 8. Limits

- One laptop, one CPU and one GPU, one version of each runtime; the numbers may not transfer.
- Latency is measured on HierText validation pages (about 41 lines per page), not documents, and on
  100 timed pages (of the 110 profile pages, after 10 warm-up) and 100 timed lines per profile.
- Every variant is a new version, measured on validation only; none replaces v1, and one that
  ships would be evaluated again (Phase 22). The recognizer and detector size points differ in
  recipe and budget, not only in size.
- Thermal drift within a sitting is limited by the mirrored run order where arms are compared
  (GW-EFF-008, 009, 011), not removed; GW-EFF-010 and GW-EFF-007's rows are one run each.
- The fused attention was measured for v1's recognizer only, so its effect on GW-REC-020's ONNX
  latency is not known.
- Height sweeps need retraining and were not run; the width cap stays at 1536.
- H1′ ONNX applies to one bundle hash; if the Phase 17 re-run ships another, the verdict does not
  carry over.
