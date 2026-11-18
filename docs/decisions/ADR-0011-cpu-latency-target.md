# ADR-0011: CPU latency target for GLYPHARIS v0.1

- **Status:** Accepted for v1 as a missed target; revisit triggered again by a change of target hardware (owner decision 2026-11-18, see below) (2026-10-25, Phase 5; approved by the owner)

## Context
ADR-0008 makes CPU latency the primary efficiency number and leaves the target to be set from
measured baseline latencies on this laptop, after Phase 5. Those measurements are in
`reports/baselines/efficiency.md`: HierText validation, batch 1, 10 warm-up and 100 timed samples,
Intel Core i7-14650HX, on AC power, Windows Balanced power plan.

The relevant reference points:

| System | Line, mean / p95 | Page, mean / p95 | Peak RAM, pages | HierText val line CER |
|---|---|---|---|---|
| Tesseract 5 (GW-BASE-001) | 141.8 / 303.7 ms | 430.8 / 807.1 ms | 281 MiB | 24.35% |
| PP-OCRv6 tiny (GW-BASE-005) | 21.6 / 66.4 ms | 1,726.6 / 2,729.0 ms | 1,836 MiB | 11.10% |
| PP-OCRv6 small (GW-BASE-006) | 46.0 / 114.4 ms | 3,714.2 / 6,308.3 ms | 2,051 MiB | 7.53% |

Tesseract is the fastest page reader but finds about one HierText line in ten (detection recall
10.29%), so it is not the speed bar for a usable engine. PP-OCRv6 tiny is the fastest modern
pipeline and the closest existing system to what GLYPHARIS aims at; PP-OCRv6 small is the accuracy
reference.

## Decision
Measured with the same protocol, conditions and harness as the baselines, through ONNX Runtime on
this laptop's CPU, GLYPHARIS v0.1 aims for:

1. **Line recognition:** mean at most 25 ms and p95 at most 70 ms per line, about PP-OCRv6 tiny's
   speed and five to six times faster than Tesseract.
2. **Page (detection and recognition):** mean at most 1.5 s and p95 at most 2.7 s per page, faster on
   average than PP-OCRv6 tiny.
3. **Memory:** peak RAM on pages below PP-OCRv6 tiny's 1,836 MiB.

The target is reported next to accuracy, never instead of it: a model that meets it only by giving
up accuracy is reported as such, and the accuracy-against-speed trade-off is the owner's call.

## Consequences
- Phase 7 times its recognizer candidates on the CPU before the combination step (PyTorch on the CPU
  as a stand-in until the Phase 17 export) and states each against target 1.
- Phase 8 chooses the detector's size and input resolution with target 2 in mind.
- Phase 17 measures the exported v0.1 against all three; Phase 16 may tighten them for v1.0.
- GLYPHARIS is timed under the same power plan as the baselines (Balanced, on AC) so the comparison
  stays like for like.

## Phase 17 measurement (2026-10-28): revisit triggered, owner decision pending
The exported v0.1 misses the page target by a wide margin (GW-EFF-004): through ONNX Runtime a page
takes 5.5 s on average (p95 12.7 s) and 11.2 GB of RAM. The cause is the exported recognizer, about
4.5 times slower than in PyTorch; the exported detector is twice as fast. The PyTorch build misses
the target by less (GW-EFF-002): 2.0 s mean, 4.0 s p95, 1.6 GB, on pages it reads with 24.3% page
CER against 42.4% for the best baseline (GW-E2E-001). Until the owner decides, v0.1 ships with
PyTorch as its default backend. The options as they stand:
1. accept v0.1 at about 2.0 s per page and make speed Phase 16's first task (fused or rewritten
   attention for the export, padding and batching, INT8, the ONNX detector);
2. trade accuracy for speed now, under new registered runs (a smaller input, a lighter recognizer);
3. restate the target for the PyTorch build (the ONNX Runtime premise did not hold).

**Note (2026-11-18, Phase 16):** the "twice as fast" detector figure above came from an arena-on
side test; with the arena off, as the backends run, the exported detector runs at about PyTorch's
speed and the exported recognizer takes about 2.7 times as long per line, not 4.5 (GW-EFF-007; F13
revised, F34).

## Phase 16 measurement (2026-11-18): still missed, options for the owner
The frozen v1 measured again on this laptop, HierText validation only, with every profile on mains
power, the Balanced plan and power mode "best performance" at its start and end
(`reports/efficiency/phase16/`). No CPU build of v1 meets the line or page targets; only the PyTorch
build's peak RAM on pages is within (GW-EFF-007, ms and MiB):

| Build | Line mean (≤ 25) | Line p95 (≤ 70) | Page mean (≤ 1,500) | Page p95 (≤ 2,700) | Peak RAM on pages (< 1,836) |
|---|---|---|---|---|---|
| v1 PyTorch, CPU, 10 threads | 33.9: over | 77.6: over | 2,565.7: over | 5,127.1: over | 1,641.8: within |
| v1 fp32 ONNX, CPU, 10 threads | 92.2: over | 304.4: over | 8,573.7: over | 19,011.5: over | 6,529.6: over |
| v1 fp32 ONNX, CPU, default threads | 104.7: over | 392.1: over | 9,924.9: over | 22,071.7: over | 6,541.5: over |

The exported detector is not twice as fast (F34, which corrects F13): with the arena off its network
takes 297.5 ms per page against 296.4 ms in PyTorch, while the exported recognizer's network takes
8,167.7 of the ONNX build's 8,573.7 ms per page against 2,148.2 ms in PyTorch, and a line takes
92.25 against 33.90 ms, about 2.7 times as long (GW-EFF-007). Against PP-OCRv5 mobile in one
sitting the ONNX bundle is slower per page, 9,623.6 against 3,086.0 ms (H1′ ONNX not met,
GW-EFF-010).

The new versions that come closest, measured on validation as Pareto points (GW-EFF-009, its own
sittings, so v1's values differ a little from GW-EFF-007's; none is v1 and none replaces it):

| Point | Version | Page mean ms | Page p95 ms | Peak RAM MiB | Page targets within | Page CER % |
|---|---|---|---|---|---|---|
| v1-torch | v1 | 2,637 | 5,328 | 1,665.0 | 1 of 3 | 24.32 |
| v1-onnx | v1 | 9,739 | 22,474 | 6,525.5 | 0 of 3 | 24.32 |
| onnx-det-torch-rec | variant (new version, not v1) | 2,579 | 5,187 | 1,550.0 | 1 of 3 | 24.32 |
| fused-attention | variant (new version, not v1) | 4,021 | 8,267 | 2,393.0 | 0 of 3 | 24.32 |
| int8-recognizer | variant (new version, not v1) | 6,891 | 16,072 | 6,896.5 | 0 of 3 | 24.32 |

The ONNX detector with the PyTorch recognizer has the lowest mean page time among the points at v1's
accuracy (about 58 ms below v1-torch's 2,637.3 ms in the same sittings; no interval for that
difference is reported), and the fused attention export the lowest mean page time of the all-ONNX
ones (F35); both are still over both page time targets, and only the first is within the RAM target.
The detector input size and the smaller detector GW-DET-003 move page accuracy but change the ONNX
page time by at most about 13%; the recognizer sets the page time (F37). PyTorch on the RTX 4050
reads a page in 1,049 ms mean, 3,288 ms p95 (GW-EFF-007), but this ADR is a CPU target and the GPU
is outside it.

The options for the owner, restated:
1. keep v1 with PyTorch as its default backend and accept about 2.6 s per page on this CPU
   (2,565.7 ms mean, GW-EFF-007), as now;
2. adopt a variant as a new version (onnx-det-torch-rec is the closest at v1's accuracy); one that
   ships is evaluated in Phase 22, as Phase 14 item 6 fixes for INT8, and does not become v1;
3. restate the target for the PyTorch build (the ONNX Runtime premise of the Decision did not hold
   for this recognizer, F34);
4. set a GPU target as a separate decision.

Phase 16 does not tighten the targets and changes no default; which option to take is the owner's
decision. The Phase 17 re-run measures v1 against this ADR again (see the owner's decision below).

## Owner decision (2026-11-18)
After the Phase 16 measurement the owner decided:
1. **v1 keeps PyTorch as its default backend** (option 1 above). The v1 build stays as it was
   evaluated in Phase 14, and these targets are recorded as missed for it on this laptop, not
   restated.
2. **BookWorm's target hardware is lightweight devices such as smartphones**, not only this
   laptop. That is this ADR's second revisit trigger. The targets above are for this laptop's CPU;
   a target for phones needs measurements on a phone, which this laptop cannot give. Until then the
   laptop numbers only rank candidate builds.
3. **The phone candidate is the all-ONNX build with fused attention** (GW-EFF-009's
   `fused-attention`: v1's page CER and F1 on validation at 4,021 ms mean per page and 2,393.0 MiB
   on this laptop; F35). It is a new version, not v1, and is evaluated as such in Phase 22. The
   owner first chose option 2 with the ONNX detector and PyTorch recognizer in mind; that build
   keeps a PyTorch recognizer, which phone runtimes do not run, so the all-ONNX build replaced it.
   Fused attention combined with the INT8 recognizer was not measured in Phase 16 and may be
   measured on validation before Phase 22 fixes its candidate.

The Phase 17 re-run for v1 cites GW-EFF-007, 009 and 010 as its measurement against this ADR (same
bundle, same tooling, 2026-11-17 and 18) instead of profiling again.

**Note (2026-11-18, Phase 22 registration):** the Phase 22 plan ([PHASE_STATUS](../PHASE_STATUS.md),
Phase 22) turns the decision above into registered work; nothing in it has run. The targets of this
ADR do not change, and no phone target is set yet.
- The phones are a Samsung Galaxy M52 5G, then a Moto G35 5G. Each device's SoC, RAM and Android build
  are read from it and registered before it is measured (GW-EFF-015).
- Whether the stock Android runtime holds the `com.microsoft.MultiHeadAttention` kernel is checked
  first, on a phone, before any candidate work. If it does not, the candidate's test evaluation stops
  and the reason is recorded.
- `fused-attention` stays the default candidate. Fused attention with the INT8 recognizer, and a rule
  of one recognizer run per line, are measured on validation as two further arms. The second is not
  named in this ADR; the owner approved it for Phase 22. Either one replaces the default only under
  the selection rule written with H11 (`research/hypotheses.md`, GW-EFF-013).
- The candidate is a new version, not v1. It is compared with v1 under H11, and its results sit beside
  v1's.

## Revisit if
- The exported v0.1 misses the target by a wide margin and the trade-off needs an owner decision.
- BookWorm's target hardware turns out to be different (for example phones rather than laptops).
- The measurement protocol (EXPERIMENT_PROTOCOL §3) changes.
