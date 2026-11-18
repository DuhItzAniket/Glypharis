# Efficiency

Written in Phase 16 and updated in Phase 21, for RQ7 and the latency side of RQ4. Every number comes
from a generated report and carries its experiment ID: GW-EFF-006 (the fp32 ONNX bundle and its
checks), GW-EFF-007 (what v1 costs), GW-EFF-008 (INT8, H7′), GW-EFF-009 (accuracy against latency),
GW-EFF-010 (H1′ ONNX, with the Phase 16 amendment to GW-BASE-002), GW-EFF-011 (H6′), and the three
re-runs on the v1 bundle: GW-EFF-012, GW-E2E-011 and GW-E2E-012. Reports are in
`reports/efficiency/phase16/`, written by `scripts/phase16_score.py` at `1253dfe`, with the baseline
table in `reports/benchmark_results/efficiency_comparison.md`. Final validation (Phase 22) has not
been run.

<!-- trace: GW-EFF-007 reports/efficiency/phase16/README.md#verdicts -->

Conventions. Validation data only (HierText validation); no test set was read. One laptop: Intel
Core i7-14650HX, NVIDIA GeForce RTX 4050 Laptop GPU; torch 2.14.1+cu126, onnxruntime 1.31.1. Mains
power, Balanced plan, power mode "best performance", recorded at the start and end of every profile;
no sitting was void. Each run had 10 warm-up and 100 timed items in one process. CPU runs used
10 threads (torch intra and inter 10, each ONNX Runtime session intra-op 10, sequential) unless
stated. Arms compared in one sitting for H7′, H6′ and the Pareto points (GW-EFF-008, 009, 011) ran in
order and then in reverse order, and each item's time is the mean of its two runs. GW-EFF-010
follows Phase 14's order (the baseline, then the bundle, one run each), and GW-EFF-007's rows are
one run each. Means carry a 95% bootstrap interval over the timed items (1,000 resamples); p95 is a
point value. Page time was measured on the 110 profile pages, page CER and E2E F1 on the 400 Phase 5
pages (strict, by page), and line CER on 5,000 seeded validation lines (strict, greedy, by line).
The Phase 16 reports print 95 intervals: 4 are confirmatory (H7′ latency, H7′ accuracy, H6′, H1′
ONNX) and 91 descriptive, and only the four carry a verdict. v1 is the frozen build evaluated in
Phase 14; every other point is a new version, not v1. HierText is released under CC BY-SA 4.0; its
images come from Open Images, which lists them as CC BY 2.0. Tables derived from it are shared alike.

## 1. The same model in another runtime (GW-EFF-006)

v1's plain fp32 ONNX export (TorchScript exporter, as in Phase 17) was checked against outputs
written once from the PyTorch v1 build on the CPU. The detector map was compared on
the <!-- derived: eff006_det_pages --> 110 profile pages, recognizer logits and text
on <!-- derived: eff006_rec_lines --> 5,000 lines, and page CER (as a fraction)
on <!-- derived: eff006_e2e_pages --> 400 pages. The bundle manifest SHA-256 is
`875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037`.

<!-- trace-table: GW-EFF-006 reports/efficiency/phase16/README.md#gw-eff-006-the-v1-fp32-onnx-bundle detector map, max abs difference=L22|recognizer logits, max abs difference=L23|text=L24|page CER difference=L25 -->

| Check | Limit | Measured |
|---|---|---|
| detector map, max abs difference | 1e-4 | 7.33e-06 |
| recognizer logits, max abs difference | 1e-3 | 3.09e-04 |
| text | other differences 0, near ties at most 0.1% of lines | 0 other differences, 0 near ties |
| page CER difference | 0.0005 | 1.49e-05 |

All four checks passed, so the ONNX results below describe the same model as v1.

## 2. What v1 costs (GW-EFF-007)

<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#parameters-and-file-sizes -->

Parameters: detector 3,187,209, recognizer 7,928,835, total 11,116,044. Files: detector 12.44 MiB
(.pt) and 12.15 MiB (fp32 ONNX), recognizer 30.31 and 30.40 MiB.

Lines, one line per item, except the last row, where an item is a batch of 32 lines as the pipeline
batches them. Intervals are by timed item.

<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/cost_table.md#lines PyTorch\, CPU\, 10 threads=PyTorch\, CPU\, 10 threads|fp32 ONNX\, CPU\, 10 threads=fp32 ONNX\, CPU\, 10 threads|fp32 ONNX\, CPU\, default threads=fp32 ONNX\, CPU\, default threads|PyTorch\, GPU\, batch 1=PyTorch\, GPU\, batch 1|PyTorch\, GPU\, batch 32=L25 -->

| Build | Mean ms per item | p95 ms | Lines/s | Peak RAM MiB | Peak VRAM MiB (torch counter) |
|---|---|---|---|---|---|
| PyTorch, CPU, 10 threads | 33.90 [29.92, 38.13] | 77.60 | 29.5 | 825.7 | n/a |
| fp32 ONNX, CPU, 10 threads | 92.25 [70.69, 116.28] | 304.39 | 10.8 | 1,028.0 | n/a |
| fp32 ONNX, CPU, default threads | 104.73 [79.87, 132.05] | 392.09 | 9.5 | 1,043.6 | n/a |
| PyTorch, GPU, batch 1 | 12.22 [11.24, 13.33] | 19.01 | 81.8 | 1,180.6 | 77.3 |
| PyTorch, GPU, batch 32 | 199.89 [187.57, 213.04] | 312.66 | 160.1 | 1,216.2 | 282.2 |

Pages, one page at a time, with lines batched as the pipeline batches them. Intervals are by page.

<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages PyTorch\, CPU\, 10 threads=PyTorch\, CPU\, 10 threads|fp32 ONNX\, CPU\, 10 threads=fp32 ONNX\, CPU\, 10 threads|fp32 ONNX\, CPU\, default threads=fp32 ONNX\, CPU\, default threads|PyTorch\, GPU=PyTorch\, GPU -->

| Build | Mean ms | p95 ms | Peak RAM MiB | Lines per page | ms per line (least squares) | Peak VRAM MiB (torch counter) |
|---|---|---|---|---|---|---|
| PyTorch, CPU, 10 threads | 2,566 [2,327, 2,829] | 5,127 | 1,641.8 | 40.9 | 28.6 [19.2, 42.9] | n/a |
| fp32 ONNX, CPU, 10 threads | 8,574 [7,516, 9,766] | 19,011 | 6,529.6 | 40.9 | 72.6 [32.9, 142.7] | n/a |
| fp32 ONNX, CPU, default threads | 9,925 [8,744, 11,261] | 22,072 | 6,541.5 | 40.9 | 85.5 [41.6, 165.1] | n/a |
| PyTorch, GPU | 1,049 [825, 1,300] | 3,288 | 2,450.0 | 40.9 | 4.8 [-1.4, 16.7] | 432.4 |

Each row is one run in chain order, not a mirrored pair.

The mirrored comparison of the two CPU builds is GW-EFF-009's (the tables in sections 3 and 5). In
those sittings the fp32 ONNX build took <!-- derived: onnx_page_ratio_mirrored --> 3.69 times as
long per page as PyTorch and used <!-- derived: onnx_ram_ratio_mirrored --> 3.92 times its peak page
RAM. The one-run chain-order rows above give <!-- derived: onnx_page_ratio_chain --> 3.34
and <!-- derived: onnx_ram_ratio_chain --> 3.98. These ratios are computed from the printed means and
have no interval; the interval on the page-time difference is in section 3.

### Where the page time goes

Mean ms per page; the stages sum to the page time.

<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/cost_table.md#page-time PyTorch\, CPU\, 10 threads=PyTorch\, CPU\, 10 threads|fp32 ONNX\, CPU\, 10 threads=fp32 ONNX\, CPU\, 10 threads|PyTorch\, GPU=PyTorch\, GPU -->

| Build | det_network | rec_network | Page mean |
|---|---|---|---|
| PyTorch, CPU, 10 threads | 296.4 (12%) | 2,148.2 (84%) | 2,565.7 |
| fp32 ONNX, CPU, 10 threads | 297.5 (3%) | 8,167.7 (95%) | 8,573.7 |
| PyTorch, GPU | 694.5 (66%) | 261.0 (25%) | 1,049.1 |

<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#page-time -->

The other seven stages (image to RGB, detector preprocessing and post-processing, cropping,
recognizer batch build, decoding, reading order) each took at
most <!-- derived: other_stages_max --> 43.6 ms on the CPU at 10 threads. The detector network took 297.5
ms per page in fp32 ONNX and 296.4 ms in PyTorch, one run each, so no difference is stated. The
recognizer network took 8,167.7 ms in fp32 ONNX and 2,148.2 ms in PyTorch, one run each, and it is
the largest stage in both CPU builds. An earlier side measurement in which the exported detector ran
twice as fast was taken with ONNX Runtime's memory arena on (F13, revised; F34).

## 3. Why the export is slow (GW-EFF-009)

<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages -->
<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#lines -->

The plain exporter writes each global mixer's attention as separate MatMul, Softmax and masking
nodes, where PyTorch runs a fused kernel. The same weights were exported with each mixer as one
`com.microsoft.MultiHeadAttention` node (`glypharis/efficiency/fused_attention.py`). CER of the
fused build minus CER of v1's fp32 ONNX build, on the same items:

<!-- trace-table: GW-EFF-009 reports/efficiency/phase16/pareto.md page CER (interval by page)=L19,Δ CER (points)|line CER (interval by line)=L29,Δ CER (points) -->

| Measure | Δ CER (points) |
|---|---|
| page CER (interval by page) | +0.00 [+0.00, +0.00] |
| line CER (interval by line) | +0.00 [+0.00, +0.00] |

Accuracy, page time and memory of these builds on pages:

<!-- trace-table: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages v1 PyTorch=v1-torch|v1 fp32 ONNX=v1-onnx|fused attention (new version)=fused-attention|INT8 recognizer (new version)=int8-recognizer -->

| Point | CER % | E2E F1 % | Mean ms | Δ ms | Peak RAM MiB |
|---|---|---|---|---|---|
| v1 PyTorch | 24.32 | 60.50 | 2,637.3 | −7,102.1 [−8,195.7, −6,115.5] | 1,665.0 |
| v1 fp32 ONNX | 24.32 | 60.50 | 9,739.4 | n/a | 6,525.5 |
| fused attention (new version) | 24.32 | 60.50 | 4,020.8 | −5,718.6 [−6,630.9, −4,883.7] | 2,393.0 |
| INT8 recognizer (new version) | 24.32 | 60.46 | 6,890.9 | −2,848.5 [−3,233.6, −2,505.1] | 6,896.5 |

Δ ms is the point minus v1's fp32 ONNX build on the same pages (descriptive, interval by page).

The fused build printed a higher mean page time here, and a higher mean line time in section 5, than
v1 PyTorch. No interval was computed for those differences, so no difference from PyTorch is stated
(F35).

## 4. INT8 (H7′, GW-EFF-008)

<!-- trace: GW-EFF-008 reports/efficiency/phase16/int8.md -->

Dynamic INT8 of the v1 fp32 ONNX recognizer, by the registered recipe (ONNX Runtime's
`quant_pre_process`, then `quantize_dynamic` on MatMuls with a constant weight, QInt8, per tensor),
turned <!-- derived: int8_matmul_done --> 49 of the <!-- derived: int8_matmul_all --> 67 MatMul
nodes into MatMulInteger. The other <!-- derived: int8_matmul_left --> 18 MatMul nodes stayed fp32,
among them the attention products, which have no constant input. The file shrank from 30.40 to 9.82
MiB.

<!-- trace: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-latency-met -->
<!-- trace: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-accuracy-met -->

| Half | Rule (met if the interval lies below 0) | Interval | Verdict |
|---|---|---|---|
| latency | mean(t_INT8 − 0.75 × t_fp32) per line | <!-- derived: h7_latency_interval --> −10.00 [−12.79, −7.37] ms | met |
| accuracy | paired ratio difference of errors_INT8 − 0.005 × ref_chars against errors_fp32, by line | <!-- derived: h7_accuracy_point --> −0.512 [<!-- derived: h7_accuracy_low --> −0.541, <!-- derived: h7_accuracy_high --> −0.484] points | met |

The latency half compared per-line times, each line the mean of its two runs in one sitting:

<!-- trace-table: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-latency-met fp32=fp32|int8=int8 -->

| Arm | Mean ms (two runs averaged) | Run means ms | p95 ms |
|---|---|---|---|
| fp32 | 95.78 | 96.61, 94.95 | 346.50 |
| int8 | 61.84 | 63.21, 60.46 | 232.02 |

<!-- trace: GW-EFF-008 reports/efficiency/phase16/verdicts.md#h7-accuracy-met -->

Strict line CER on the <!-- derived: h7_accuracy_lines --> 5,000 lines was 6.208% for INT8 against
6.221% for fp32. H7′ holds (F36).

On pages (GW-EFF-009, descriptive, by page) the INT8 build's page CER difference
was <!-- derived: int8_page_dcer --> +0.00 [−0.02, +0.02] points and its time
difference <!-- derived: int8_page_dms --> −2,848.5 [−3,233.6, −2,505.1] ms. Fusing the attention gave a time
difference of <!-- derived: fused_page_dms --> −5,718.6 [−6,630.9, −4,883.7] ms against the same
build (section 3). No paired interval was computed between the two variants, so no difference
between them is stated.

## 5. Accuracy against latency (GW-EFF-009)

Page points. Every row but v1 is a new version, not v1, measured on validation only, with fp32 ONNX
at 10 threads unless the point says otherwise. Point names are the report's: `det-long-side-N` sets
the detector's input long side and `det-GW-DET-003` swaps in the smaller detector. Δ is against v1's
fp32 ONNX build on the same items (descriptive, intervals by page).

<!-- trace-table: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages v1-onnx=v1-onnx|det-long-side-1024=det-long-side-1024|det-long-side-1280=det-long-side-1280|det-long-side-1792=det-long-side-1792|det-long-side-2048=det-long-side-2048|det-GW-DET-003=det-GW-DET-003|onnx-det-torch-rec=onnx-det-torch-rec -->

| Point | CER % | E2E F1 % | Mean ms | p95 ms | Peak RAM MiB | Δ CER (points) | Δ ms |
|---|---|---|---|---|---|---|---|
| v1-onnx | 24.32 [20.87, 28.15] | 60.50 [56.81, 63.96] | 9,739.4 | 22,474.1 | 6,525.5 | n/a | n/a |
| det-long-side-1024 | 31.62 [27.74, 35.79] | 55.95 | 8,491.3 | 20,331.8 | 5,983.6 | +7.29 [+5.55, +9.51] | −1,248.1 [−1,761.3, −734.3] |
| det-long-side-1280 | 26.63 [23.18, 30.38] | 58.98 | 9,537.2 | 22,553.2 | 5,963.5 | +2.31 [+1.66, +2.98] | −202.2 [−451.2, +32.6] |
| det-long-side-1792 | 24.21 [20.38, 28.30] | 60.67 | 10,168.1 | 22,592.5 | 6,576.8 | −0.11 [−0.93, +0.92] | +428.7 [+230.7, +647.8] |
| det-long-side-2048 | 23.75 [19.91, 28.17] | 61.10 | 10,400.0 | 21,713.0 | 6,556.8 | −0.57 [−1.80, +0.83] | +660.6 [+393.3, +949.6] |
| det-GW-DET-003 | 27.33 [23.60, 31.43] | 57.75 | 9,065.7 | 20,756.4 | 6,107.0 | +3.00 [+1.52, +4.33] | −673.7 [−888.5, −443.8] |
| onnx-det-torch-rec | 24.32 [20.87, 28.15] | 60.50 | 2,579.1 | 5,186.9 | 1,550.0 | +0.00 [+0.00, +0.00] | −7,160.4 [−8,258.3, −6,171.7] |

<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#page-time -->

`det-long-side-1024`, `det-long-side-1280` and `det-GW-DET-003` raised page CER (their intervals lie
above 0); the intervals of `det-long-side-1792` and `det-long-side-2048` include 0. The largest
change in mean page time among these five detector points
was <!-- derived: det_variant_max_share --> 12.8% of v1's fp32 ONNX time, at `det-long-side-1024` (F37).
The recognizer network takes 95% of the fp32 ONNX build's page time at 10 threads (section 2).

The ONNX detector with the PyTorch recognizer printed the lowest mean page time among the points at
v1's page CER and F1, <!-- derived: onnx_det_ms --> 2,579.1 ms
against <!-- derived: v1_torch_ms --> 2,637.3 ms for v1 PyTorch in the same sittings, a gap
of <!-- derived: onnx_det_gap --> 58.2 ms. No interval was computed for that gap, so no difference
between the two is stated (F38).

Recognizer points. Every row but v1 is a new version, not v1. CER is on 5,000 lines and time on
100 lines, one line at a time; intervals are by line.

<!-- trace-table: GW-EFF-009 reports/efficiency/phase16/pareto.md#lines v1-torch=v1-torch|v1-onnx=v1-onnx|fused-attention=fused-attention|int8-recognizer=int8-recognizer|GW-REC-003 (CRNN)=rec-GW-REC-003|GW-REC-006=rec-GW-REC-006|GW-REC-020 (SVTR-style)=rec-GW-REC-020|GW-REC-022=rec-GW-REC-022 -->

| Point | CER % | Mean ms | p95 ms |
|---|---|---|---|
| v1-torch | 6.22 [5.73, 6.71] | 35.55 [31.32, 40.10] | 80.82 |
| v1-onnx | 6.22 [5.73, 6.71] | 95.59 [73.55, 120.41] | 340.04 |
| fused-attention | 6.22 [5.73, 6.71] | 39.53 [32.12, 47.52] | 112.10 |
| int8-recognizer | 6.21 [5.73, 6.69] | 63.35 [48.10, 80.25] | 220.31 |
| GW-REC-003 (CRNN) | 8.34 [7.74, 8.92] | 9.07 [7.97, 10.24] | 21.15 |
| GW-REC-006 | 7.85 [7.29, 8.41] | 16.67 [14.39, 19.07] | 42.32 |
| GW-REC-020 (SVTR-style) | 8.21 [7.61, 8.78] | 29.03 [23.07, 35.60] | 97.18 |
| GW-REC-022 | 7.57 [7.00, 8.14] | 100.57 [77.01, 126.77] | 387.76 |

The recognizer points differ in recipe and training budget, not only in size, so they are not a size
ladder. These 5,000 lines are a seeded subset, so this CER is a different quantity from the
validation line CER over all lines reported for GW-REC-034 and GW-REC-038.

## 6. The registered comparisons (H6′, H1′ ONNX)

<!-- trace: GW-EFF-011 reports/efficiency/phase16/verdicts.md#h6-not-met -->
<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx-not-met -->

| Test | Rule (met if the interval lies below 0) | Interval | Verdict |
|---|---|---|---|
| H6′ (GW-EFF-011) | mean(t_SVTR − 1.5 × t_CRNN) per line, fp32 ONNX | <!-- derived: h6_interval --> +16.78 [+12.02, +22.04] ms | not met |
| H1′ ONNX (GW-EFF-010) | mean per-page time(v1 fp32 ONNX) − time(GW-BASE-002) | <!-- derived: h1_onnx_interval --> +6,537.6 [+5,383.7, +7,806.3] ms | not met |

H6′ compared the SVTR-style GW-REC-020 with the CRNN GW-REC-003 in fp32 ONNX, each line the mean of
its two runs. Both exports first passed their logit and text checks on
the <!-- derived: h6_check_lines --> 5,000 lines.

<!-- trace-table: GW-EFF-011 reports/efficiency/phase16/verdicts.md#h6-not-met CRNN=CRNN|SVTR=SVTR -->

| Arm | Mean ms (two runs averaged) | Run means ms |
|---|---|---|
| CRNN | 10.16 | 10.83, 9.48 |
| SVTR | 32.01 | 31.23, 32.80 |

In PyTorch at 10 threads (information, not part of the test):

<!-- trace-table: GW-EFF-011 reports/efficiency/phase16/verdicts.md#h6-not-met GW-REC-003=GW-REC-003|GW-REC-020=GW-REC-020 -->

| Model | Mean ms | p95 ms |
|---|---|---|
| GW-REC-003 | 13.00 | 27.52 |
| GW-REC-020 | 15.70 | 32.40 |

The plain export's unfused attention (section 3) may explain why the gap between the two models is
wider in ONNX than in PyTorch, but this was not measured for GW-REC-020 (F39).

<!-- trace: GW-EFF-010 reports/efficiency/phase16/verdicts.md#h1-onnx-not-met -->

H1′ ONNX: the v1 fp32 ONNX bundle took <!-- derived: h1_onnx_bundle_ms --> 9,623.6 ms per page
against <!-- derived: h1_onnx_base_ms --> 3,086.0 ms for a fresh PP-OCRv5 mobile profile in the same
sitting, by Phase 14's rule (F40). The verdict applies only to the bundle named by the hash in
section 1. Page times of the same baseline from other sittings are not compared with these. H1′
itself holds from Phase 14, whose latency half is the PyTorch build (GW-EFF-005).

## 7. Against the CPU targets, and the owner's decision

ADR-0011 set targets for the CPU build on this laptop, shown in the table's header in ms and MiB
(GW-EFF-007).

<!-- trace-table: GW-EFF-007 reports/efficiency/phase16/README.md#adr-0011s-targets-against-each-v1-build v1 PyTorch\, CPU\, 10 threads=v1 PyTorch\, CPU\, 10 threads|v1 fp32 ONNX\, CPU\, 10 threads=v1 fp32 ONNX\, CPU\, 10 threads|v1 fp32 ONNX\, CPU\, default threads=v1 fp32 ONNX\, CPU\, default threads -->

| Build | line mean (≤ 25) | line p95 (≤ 70) | page mean (≤ 1,500) | page p95 (≤ 2,700) | peak RAM on pages (< 1,836) |
|---|---|---|---|---|---|
| v1 PyTorch, CPU, 10 threads | 33.9: over | 77.6: over | 2,565.7: over | 5,127.1: over | 1,641.8: within |
| v1 fp32 ONNX, CPU, 10 threads | 92.2: over | 304.4: over | 8,573.7: over | 19,011.5: over | 6,529.6: over |
| v1 fp32 ONNX, CPU, default threads | 104.7: over | 392.1: over | 9,924.9: over | 22,071.7: over | 6,541.5: over |

Every CPU build of v1 was over the line and page time targets; only the PyTorch build's peak page RAM
was within its target (F38). Every page point in sections 3 and 5 was over the page time targets.
Among the variants, only the ONNX detector with the PyTorch recognizer was within the RAM target.

On the GPU, which ADR-0011 does not cover, PyTorch read a page
in <!-- derived: gpu_page_ms --> 1,049 ms on average.

<!-- trace: GW-EFF-009 reports/efficiency/phase16/pareto.md#pages -->

The owner decided on 2026-11-18 (ADR-0011) that v1 keeps PyTorch as its default backend and that
ADR-0011's laptop targets are recorded as missed for v1, not restated. The target hardware is
lightweight devices such as phones. Phones are the target hardware; no phone has been measured. The
laptop figures here therefore only rank candidate builds. The all-ONNX build with fused attention
(section 3: 4,020.8 ms mean per page and 2,393.0 MiB on this laptop, at v1's page CER and F1 on
validation) is the phone candidate. A new version, not v1; validation and laptop figures only; to be
evaluated in Phase 22. Fused attention combined with the INT8 recognizer was not measured.

For context only, the Phase 5 baselines' page times on the same pages are in
`reports/benchmark_results/efficiency_comparison.md`. They were measured in different sittings and
no difference is stated from them. The only like-for-like baseline comparison is H1′ ONNX above.

## 8. The v1 bundle after the code freeze

The code tagged `phase14-eval` is what Phase 14 evaluated. The `v1-code` tag differs from it only in
docstrings, comments and the default bundle name (`reports/export/v1-code/code_diff.json`). The
three re-runs below used the bundle with the manifest hash given in section 1, so the bundle did not
change and H1′ ONNX still refers to it.

<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#tolerances -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#detector -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#recognizer -->
<!-- trace: GW-EFF-012 reports/export/GW-EFF-012/torch_bundle_check.json#end_to_end -->

GW-EFF-012 ran the bundle on the PyTorch backend, the path v1 ships on, against the Phase 16 PyTorch
references at <!-- derived: eff012_threads --> 10 threads. The detector map
on <!-- derived: eff012_det_pages --> 110 pages had a maximum absolute difference
of <!-- derived: eff012_det_diff --> 0.0 (limit <!-- derived: eff012_det_limit --> 0.0001), and the
logits on <!-- derived: eff012_rec_lines --> 5,000 lines one
of <!-- derived: eff012_rec_diff --> 0.0 (limit <!-- derived: eff012_rec_limit --> 0.001). Line texts
were identical on <!-- derived: eff012_text_same --> 5,000 of those lines, and
all <!-- derived: eff012_pages_same --> 400 pages had identical text, with a page CER difference
of <!-- derived: eff012_cer_diff --> 0.0 as a fraction
(limit <!-- derived: eff012_cer_limit --> 0.0005). Page CER was <!-- derived: eff012_page_cer --> 24.32% on
both sides.

<!-- trace: GW-E2E-011 reports/package/GW-E2E-011.json -->

GW-E2E-011 ran the package on the same bundle against the frozen pipeline on
the <!-- derived: e2e011_pages --> 400 pages: page CER 24.32% on both,
and <!-- derived: e2e011_lines_same --> 16,835 of <!-- derived: e2e011_lines --> 16,835 lines with
identical text and box (corners within <!-- derived: e2e011_box_px --> 1.0 px).

GW-E2E-012 repeated the hardening cases on the bundle with both backends. On each backend,
of <!-- derived: e2e012_inputs --> 269 inputs, <!-- derived: e2e012_results --> 47 gave results and the
other <!-- derived: e2e012_errors --> 222 gave typed errors, with no other exception, and every
damaged-bundle case behaved as registered (`reports/hardening/GW-E2E-012.json`).

<!-- trace: GW-E2E-011 reports/package/GW-E2E-011.json -->

These checks use validation data. Test results come from the GPU strict-fp32 evaluation; equality with
the bundle is shown on validation only. Phase 22 checks reproduction within the 0.05-point tolerance.

## 9. Limits

<!-- trace: configs/final/glypharis_v1.yaml configs/final/glypharis_v1.yaml#recognizer/input -->
<!-- trace: GW-EFF-007 reports/efficiency/phase16/cost_table.md#pages -->

- One laptop, one CPU and one GPU, one version of each runtime; the numbers may not transfer, and
  no phone has been measured.
- Latency was measured on HierText validation pages, not documents, on 100 timed pages (of the
  110 profile pages, after 10 warm-up) and 100 timed lines per profile.
- Every variant is a new version, measured on validation only; none replaces v1. The recognizer and
  detector size points differ in recipe and budget, not only in size.
- Thermal drift within a sitting is limited by the mirrored run order where arms are compared
  (GW-EFF-008, 009, 011), not removed; GW-EFF-010 and GW-EFF-007's rows are one run each.
- The fused attention was measured for v1's recognizer only, so its effect on GW-REC-020's ONNX
  latency is not known.
- Height sweeps need retraining and were not run; the recognizer's width cap stays at 1536.
- H1′ ONNX applies to one bundle hash. The re-runs in section 8 used that bundle, and a different
  bundle would need its own test.
