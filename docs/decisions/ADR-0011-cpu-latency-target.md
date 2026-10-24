# ADR-0011: CPU latency target for GLYPHARIS v0.1

- **Status:** Accepted (2026-10-25, Phase 5; approved by the owner)

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

## Revisit if
- The exported v0.1 misses the target by a wide margin and the trade-off needs an owner decision.
- BookWorm's target hardware turns out to be different (for example phones rather than laptops).
- The measurement protocol (EXPERIMENT_PROTOCOL §3) changes.
