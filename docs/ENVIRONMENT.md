# Development Environment

The generated record of the project environment is
[`reports/environment/env_report.md`](../reports/environment/env_report.md), written by
`scripts/env_report.py`. Where this page and that report differ, the report is the record. New measured
values go into generated reports, not into this page.

| What | Where it is recorded |
|---|---|
| OS, CPU cores, RAM, Python, PyTorch, CUDA, GPU, driver, key package versions, AMP, CTC and determinism checks | `reports/environment/env_report.md`, `reports/environment/env_report.json` (Phase 1) |
| Pinned packages of the project environment | `requirements/lock.txt`; torch and torchvision pins in `requirements/README.md` |
| Pinned packages of the baseline and official-evaluator environments | `requirements/baselines-lock.txt`, `requirements/hiertext-eval-lock.txt`, `requirements/omnidocbench-eval-lock.txt` |
| Device, precision and library versions of each training run | `reports/training/<ID>/report.md`; for v1, `configs/final/glypharis_v1.yaml` |
| Conditions of each latency profile (CPU, power plan and mode, thread settings, GPU state) | the `conditions`, `conditions_end` and `gpu_state` fields of `reports/efficiency/phase16/profiles/*.json` |
| Environment on CI | the `env-report-ci` artifact of each run (`.github/workflows/ci.yml`) |

The committed report was generated once, at the start of the project (commit `a658c41`), and Phase 21 does
not regenerate it. To refresh it, run `python scripts/env_report.py`.

This is the development and training machine. It is not the target hardware. Phones are the target
hardware ([ADR-0011](decisions/ADR-0011-cpu-latency-target.md), owner decision of 2026-11-18). In
Phase 22 the phone candidate's networks, a new version and not v1, were measured on two phones
(GW-EFF-015, [FINAL_VALIDATION](FINAL_VALIDATION.md) section 4). v1 was not measured on a phone.

## Hardware, as recorded in Phase 0

Recorded on 2026-10-09 by hand, with the commands shown. These values are kept as the Phase 0 record and
have not been measured again here.

| Component | Value | How measured |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU | `nvidia-smi`, `torch.cuda.get_device_properties` |
| GPU architecture | Ada Lovelace, compute capability 8.9, 20 SMs | `torch.cuda.get_device_properties` |
| VRAM total | 6.0 GiB (6141 MiB) | `nvidia-smi` |
| VRAM free at idle | ~4.96 GiB (desktop/WDDM + other apps hold ~1 GiB) | `torch.cuda.mem_get_info()` |
| bf16 support | Yes | `torch.cuda.is_bf16_supported()` |
| fp16 matmul throughput | ~13.3 TFLOPS (4096² matmul, 50 iters, quick probe) | ad-hoc benchmark |
| GPU driver | 566.24 (WDDM), supports CUDA runtime ≤ 12.7 | `nvidia-smi` |
| Integrated GPU | Intel UHD Graphics (display) | WMI |
| CPU | Intel Core i7-14650HX, 16 cores / 24 threads | WMI |
| RAM | 15.7 GiB usable | WMI |
| Storage | NTFS `C:`, 953 GB total, ~307 GB free | `df`, `Get-PSDrive` |

### Implications for training design

- Plan for ~4.5 GiB of usable VRAM, not 6 GiB. The Windows desktop compositor runs on the dGPU's WDDM
  driver and other apps (browsers, chat clients) take a share. Batch sizes and resolutions in configs are
  validated against measured peak memory, not the nominal 6 GB.
- bf16 autocast is available (Ada). It avoids most fp16 overflow issues and does not need a `GradScaler`.
  CTC loss is computed in fp32; the check is in the environment report.
- 16 GB RAM is the tightest resource for data preparation, not for training. Datasets of millions of small
  crops are packed rather than stored as loose files on NTFS.
- Long runs on a laptop GPU can throttle. Runs log GPU temperature and clocks, so throughput differences
  between runs can be attributed.

## Software found in Phase 0 (the global interpreter, not the project environment)

| Item | Value |
|---|---|
| OS | Windows 11 Home Single Language, build 26300 |
| Long paths | Enabled (`LongPathsEnabled=1`) |
| Python | 3.11.9 (`py -0` lists only 3.11) |
| pip | 26.2.1 |
| PyTorch | 2.5.1+cu121 (CUDA runtime 12.1, cuDNN 9.1.0) |
| torchvision | 0.20.1+cu121 |
| Git | 2.55.0.windows.3, credential helper: Git Credential Manager |
| GitHub CLI (`gh`) | not installed at the time |
| CUDA toolkit (`nvcc`) | not installed (not needed: PyTorch wheels ship their own runtime) |
| Tesseract | not installed at the time (installed for Phase 5) |
| conda | not installed |

The global interpreter had 322 packages installed, including unrelated ones (torch-geometric,
facenet-pytorch, pytorch-lightning, transformers 5.x). It is not a reproducible base, and its
site-packages must not leak into the project environment. It is used only to create the virtual
environments.

## Project environment

([ADR-0006](decisions/ADR-0006-isolated-environments.md))

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements/lock.txt
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python scripts/env_report.py
```

Checks before committing: `ruff check .`, `ruff format --check .`, `mypy`, `pytest`,
`python scripts/validate_configs.py`. CI runs the same steps on Linux with the CPU build of the same torch
version (`.github/workflows/ci.yml`). The clean-environment install of the package was checked in the
Phase 18 re-run (`docs/PHASE_STATUS.md`). The walkthrough from a fresh clone passed in Phase 22
(GW-E2E-013, [FINAL_VALIDATION](FINAL_VALIDATION.md) section 1). Final validation (Phase 22) has been
run; its gate (push, CI, the owner's merge) is still open.

### PyTorch build decision

| Option | Notes |
|---|---|
| 2.5.1 + cu121 (global install, Phase 0) | Known to work on this driver; older ONNX exporter |
| 2.14.1 + cu126 (chosen) | Driver 566.24 supports CUDA runtimes up to 12.7, so cu126 wheels run without a driver update |

Decision criterion (Phase 0): stable AMP and CTC on this GPU, measured. The measured checks (bf16 and fp16
autocast, CTC in fp32 under autocast, CPU bit-identical determinism, CUDA determinism within tolerance) are
in the environment report. v1 was trained with this build (`configs/final/glypharis_v1.yaml`).

## Environment items

1. Create `.venv` and decide the torch build: done in Phase 1 (above).
2. Separate environment for baselines (`.venv-baselines`), so PaddlePaddle does not conflict with PyTorch:
   done in Phase 5 (pins in `requirements/baselines-lock.txt`, setup in `requirements/README.md`).
3. Install Tesseract (UB Mannheim Windows build): done for Phase 5 (`docs/PHASE_STATUS.md`, Phase 5).
4. CI status for each phase is recorded with its run IDs in `docs/PHASE_STATUS.md`.
