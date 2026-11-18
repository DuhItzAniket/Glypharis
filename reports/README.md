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

## Training curves

Each training run's curves are in `reports/training/<run_id>/curves.png`, shown in that run's
`report.md` under "Curves". Learning-rate range tests are in
`reports/training/sanity/<run_id>/lr_range.png`. The per-step and per-epoch logs behind the plots stay
local under `runs/`.

## Attribution of HierText and Open Images material

Generated files are not edited after the fact, so this notice covers the ones written before they
carried an attribution line:

- the 53 `reports/training/*/report.md` files with a "Worst validation lines" table, and the
  `worst_lines.png` strip beside each of them;
- `reports/data_analysis/hiertext/line_crops.png`;
- `reports/data_analysis/augmentation_preview.png` and `reports/data_analysis/glyph_c_preview.png`;
- `reports/training/sanity/GW-DET-001/preview.jpg`.

The quoted lines and the crops come from HierText (Long et al., *Towards End-to-End Unified Scene
Text Detection and Layout Analysis*, CVPR 2022,
<https://github.com/google-research-datasets/hiertext>), released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). They are shared here under the same
licence. The images come from [Open Images](https://storage.googleapis.com/openimages/web/index.html),
which lists them as [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) and gives no warranty
about the licence of any one image. The material was changed: crops were cut from the images,
resized and converted to grey scale, and the previews show them after the augmentations named there.
Author and source are not recorded per image; line keys begin with the HierText image ID.

Run reports written by `glypharis.training.report` from Phase 21 on carry this notice themselves.
Licences of every dataset, and what the repository does and does not redistribute, are in
[`DATASETS.md`](../DATASETS.md).
