# Limitations — working notes

Add limitations as soon as they are known; don't wait for the end.

- Single hardware platform (one laptop GPU); throughput numbers may not transfer.
- English / Latin-script focus; other scripts are out of scope for v1.
- Some datasets carry non-commercial or share-alike terms, which constrains how trained weights
  can be released (see DATASETS.md).
- No comparison with SynthText: its licence (non-commercial, 12-month term) is incompatible with a
  model meant for an application of undecided commercial status (ADR-0009).
- Reading order is heuristic; complex layouts (tables, sidebars, footnotes) are not solved.
- Efficiency (Phase 16, `efficiency.md`): every latency is from one laptop, on HierText validation
  pages (about 41 lines per page) rather than documents; thermal drift within a sitting is limited
  by the mirrored run order of the compared arms (GW-EFF-008, 009, 011), not removed. The faster
  variants are new versions measured on validation only, and the recognizer and detector size points
  differ in recipe and budget, not only in size. No CPU build of v1 meets ADR-0011's latency
  targets.
