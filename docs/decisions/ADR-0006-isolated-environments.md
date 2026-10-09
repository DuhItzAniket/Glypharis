# ADR-0006: Separate virtual environments for the project and for baselines

- **Status:** Accepted (2026-10-09, Phase 0)

## Context
The global Python 3.11.9 has 322 packages, including several unrelated ML stacks (see
docs/ENVIRONMENT.md), so it is not a reproducible base. PaddleOCR depends on PaddlePaddle, which pins its
own CUDA/cuDNN and NumPy ranges that can conflict with PyTorch.

## Decision
- `.venv/`: project environment (PyTorch stack), created with `python -m venv` from 3.11 and with
  exact pins in a lock file generated in Phase 1.
- `.venv-baselines/`: Tesseract wrapper, PaddleOCR, and TrOCR inference used only by
  `scripts/run_baselines.py`. Nothing in the `glypharis` package imports from it.
- Neither environment uses `--system-site-packages`.

## Consequences
Two installs to maintain, but no dependency fights, and baseline versions can be recorded independently
in the baseline report.

## Revisit if
PaddleOCR is dropped as a baseline, or a single environment is shown to work with exact pins.
