# requirements/

`lock.txt` pins every dependency **except** `torch` and `torchvision`, whose wheels differ per platform.
Those two are pinned to the same version everywhere and installed from the matching PyTorch index:

| Where | Command |
|---|---|
| Laptop (CUDA 12.6 build) | `pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126` |
| CI / CPU-only | `pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu` |

Then:

```
pip install -r requirements/lock.txt
pip install -e . --no-deps
```

Full setup steps are in [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md). To change a pin: install the new
version in `.venv`, run the full test suite and `scripts/env_report.py`, then regenerate the lock with

```
python -m pip freeze --exclude-editable | grep -viE "^(torch|torchvision)==" > requirements/lock.txt
```

and say why in the commit message.

## Baselines environment (Phase 5)

`baselines-lock.txt` pins `.venv-baselines` (ADR-0006), again without `torch`, which comes from the CPU
index:

```
py -3.11 -m venv .venv-baselines
.venv-baselines/Scripts/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv-baselines/Scripts/python -m pip install -r requirements/baselines-lock.txt
.venv-baselines/Scripts/python -m pip install -e . --no-deps
```

Tesseract is not a pip package: install the UB Mannheim Windows build (it needs administrator rights)
or put `tesseract` on `PATH`. Two Windows quirks are handled in `scripts/run_baselines.py`: torch has to
be imported before paddle, or torch's DLLs fail to load, and PaddleX's online model-host check is
switched off because every model is loaded from `data/raw/baseline_models/`.
