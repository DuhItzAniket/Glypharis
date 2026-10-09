# Development Environment

Hardware and global software recorded during Phase 0 on **2026-10-09**; project environment set up in
Phase 1 on **2026-10-10**. Every value below was measured on the machine, not copied from a spec sheet.
The generated record is [`reports/environment/env_report.md`](../reports/environment/env_report.md):
re-run `python scripts/env_report.py` and update this file whenever drivers, CUDA, or the Python stack
change.

## Hardware

| Component | Value | How measured |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU | `nvidia-smi`, `torch.cuda.get_device_properties` |
| GPU architecture | Ada Lovelace, compute capability 8.9, 20 SMs | `torch.cuda.get_device_properties` |
| VRAM total | 6.0 GiB (6141 MiB) | `nvidia-smi` |
| VRAM free at idle | **~4.96 GiB** (desktop/WDDM + other apps hold ~1 GiB) | `torch.cuda.mem_get_info()` |
| bf16 support | Yes | `torch.cuda.is_bf16_supported()` |
| fp16 matmul throughput | ~13.3 TFLOPS (4096² matmul, 50 iters, quick probe) | ad-hoc benchmark |
| GPU driver | 566.24 (WDDM), supports CUDA runtime ≤ 12.7 | `nvidia-smi` |
| Integrated GPU | Intel UHD Graphics (display) | WMI |
| CPU | Intel Core i7-14650HX, 16 cores / 24 threads | WMI |
| RAM | 15.7 GiB usable | WMI |
| Storage | NTFS `C:`, 953 GB total, ~307 GB free | `df`, `Get-PSDrive` |

### Implications for training design

- **Plan for ~4.5 GiB of usable VRAM, not 6 GiB.** The Windows desktop compositor runs on the
  dGPU's WDDM driver and other apps (e.g. browsers, chat clients) take a share. Batch sizes and
  resolutions in configs must be validated against measured peak memory, not the nominal 6 GB.
- **bf16 autocast is available** (Ada). It avoids most fp16 overflow issues and does not need a
  `GradScaler`. CTC loss should still be computed in fp32 (to be verified in Phase 1 — see
  `docs/EXPERIMENT_PROTOCOL.md`).
- **16 GB RAM is the tightest resource for data preparation**, not for training. Large single-file
  annotations (e.g. SynthText's `gt.mat`) must be converted in a streaming way, and datasets of
  millions of small crops should be packed (LMDB or tar shards) rather than stored as loose files on NTFS.
- **Laptop thermals.** Long runs on a laptop GPU can throttle. Phase 1 adds GPU temperature and clock
  logging so that throughput differences between runs can be attributed correctly.

## Software (as found — the global interpreter, not the project environment)

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language, build 26300 |
| Long paths | Enabled (`LongPathsEnabled=1`) |
| Python | 3.11.9 (`py -0` lists only 3.11) |
| pip | 26.2.1 |
| PyTorch | 2.5.1+cu121 (CUDA runtime 12.1, cuDNN 9.1.0) |
| torchvision | 0.20.1+cu121 |
| Git | 2.55.0.windows.3, credential helper: Git Credential Manager |
| GitHub CLI (`gh`) | **not installed** |
| CUDA toolkit (`nvcc`) | not installed (not needed — PyTorch wheels ship their own runtime) |
| Tesseract | **not installed** (needed for the Phase 5 baseline) |
| conda | not installed |

The global interpreter has **322 packages** installed, including unrelated ones (torch-geometric,
facenet-pytorch, pytorch-lightning, transformers 5.x). It is not a reproducible base. Phase 1
creates an isolated project virtual environment with pinned versions; the global site-packages
must not leak into it.

Already-present packages that the project will likely use (versions to be re-pinned in Phase 1):
numpy 1.26.4, opencv-python 4.11.0.86, pillow 12.3.0, albumentations 2.0.8, mlflow 3.11.1,
onnx 1.15.0, onnxruntime 1.24.4, pytest 9.0.3, ruff 0.16.8, torchmetrics 1.9.0.

## Project environment (Phase 1)

The global interpreter is only used to create the virtual environment
([ADR-0006](decisions/ADR-0006-isolated-environments.md)).

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements/lock.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python scripts/env_report.py
```

Checks before committing: `ruff check .`, `ruff format --check .`, `mypy`, `pytest`,
`python scripts/validate_configs.py`. CI runs the same steps on Linux with the CPU build of the same
torch version (`.github/workflows/ci.yml`).

### PyTorch build decision

| Option | Notes |
|---|---|
| 2.5.1 + cu121 (global install, Phase 0) | Known to work on this driver; about two years old; older ONNX exporter |
| **2.14.1 + cu126 (chosen)** | Current release line; driver 566.24 supports CUDA runtimes up to 12.7, so cu126 wheels run without a driver update |

Decision criterion (Phase 0): stable AMP and CTC on this GPU, measured. The measured checks (bf16/fp16
autocast, CTC in fp32 under autocast, CPU bit-identical determinism, CUDA determinism within tolerance)
are in the environment report. If a later phase hits a regression in this build, 2.5.1 + cu121 is the
fallback; switching is a pin change plus a re-run of the report and the test suite.

## Open environment items

1. ~~Create `.venv` and decide the torch build~~ (done, Phase 1; see above).
2. Separate environment for baselines (`.venv-baselines`) so PaddlePaddle does not conflict with PyTorch
   (Phase 5).
3. Install Tesseract (UB Mannheim Windows build) — requires the user to run the installer (before Phase 5).
4. Optional: install `gh` to verify GitHub Actions runs from the terminal. Until then CI status is read
   from the public GitHub API.
