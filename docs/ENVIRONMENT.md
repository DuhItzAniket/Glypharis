# Development Environment

Recorded during Phase 0 (repository discovery) on **2026-10-09**. Every value below was measured on the
machine, not copied from a spec sheet. Re-run the diagnostics in Phase 1 (`scripts/env_report.py`, planned)
and update this file whenever drivers, CUDA, or the Python stack change.

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

## Open environment items (carried into Phase 1)

1. Create `.venv` (project) and decide whether to keep torch 2.5.1+cu121 or move to a newer
   CUDA 12.x build that the 566.24 driver supports. Decision criterion: stable AMP + CTC on this GPU,
   measured, not assumed.
2. Separate environment for baselines (`.venv-baselines`) so PaddlePaddle does not conflict with PyTorch.
3. Install Tesseract (UB Mannheim Windows build) — requires the user to run the installer.
4. Optional: install `gh` to verify GitHub Actions runs from the terminal.
