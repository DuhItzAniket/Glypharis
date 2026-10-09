# reports/

Generated output only. Files here are written by scripts and never edited by hand, so
regenerating a report must reproduce it.

```
reports/
  environment/            # env_report.py output (Phase 1)
  data_analysis/          # dataset statistics + plots (Phase 3)
  data_quality_report.md  # corrupt / missing / duplicate / invalid annotations (Phase 3)
  baselines/              # Tesseract, PaddleOCR, TrOCR-Small results (Phase 5)
  training/<run_id>/      # curves, diagnostics, chosen-checkpoint rationale (Phase 6+)
  error_cases/            # difficult-example reports (Phases 9, 15)
  confusion_analysis/     # character / word confusion tables
  benchmark_results/      # external benchmark + efficiency tables (Phases 14, 16)
  experiment_summary.csv  # one row per experiment, auto-generated
  experiment_summary.md
```

Large binaries such as raw predictions and per-sample dumps stay out of git (see `.gitignore`). Curated
figures and summary tables are committed so the research record is visible on GitHub.
