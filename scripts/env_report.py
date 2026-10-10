"""Write an environment report: versions, hardware, and numeric sanity checks.

Usage:
    python scripts/env_report.py [--out reports/environment] [--steps 50]

Outputs ``env_report.json`` and ``env_report.md``. The report deliberately leaves out host names,
user names and absolute paths. Exit code is non-zero when a required check fails:
  - CPU determinism: two seeded toy trainings give bit-identical losses;
  - on CUDA machines: AMP produces finite results, CTC computed in fp32 is finite with finite
    gradients, and two seeded CUDA trainings agree within the protocol tolerance.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import math
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn

from glypharis import __version__
from glypharis.utils.determinism import run_toy_training
from glypharis.utils.device import fp32_region
from glypharis.utils.gitinfo import git_info

# Max relative difference allowed between two seeded CUDA toy runs (EXPERIMENT_PROTOCOL §6/§11).
CUDA_REL_TOLERANCE = 1e-4

PACKAGES = [
    "torch", "torchvision", "numpy", "PyYAML", "psutil", "nvidia-ml-py",
    "pytest", "ruff", "mypy",
]  # fmt: skip


def _version(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _os_name() -> str:
    system, release, version = platform.system(), platform.release(), platform.version()
    # platform.release() says "10" on Windows 11; build 22000 and later is Windows 11.
    if system == "Windows" and release == "10":
        try:
            if int(version.split(".")[2]) >= 22000:
                release = "11"
        except (IndexError, ValueError):
            pass
    return f"{system} {release}"


def system_info() -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "os": _os_name(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total_gib": round(vm.total / 1024**3, 2),
        "python": platform.python_version(),
        "in_virtualenv": sys.prefix != sys.base_prefix,
    }


def torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info["gpu"] = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessors": props.multi_processor_count,
            "total_mib": round(total / 1024**2),
            "free_mib_at_report": round(free / 1024**2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
            "driver": _driver_version(),
        }
    return info


def _driver_version() -> str | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
        return version.decode() if isinstance(version, bytes) else str(version)
    except Exception:
        return None


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all())


def amp_check() -> dict[str, Any]:
    """Matmul and conv under autocast for each low precision supported by the GPU."""
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "no CUDA device"}
    results: dict[str, Any] = {}
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(256, 256, device="cuda", generator=gen)
    x = torch.randn(4, 3, 64, 64, device="cuda", generator=gen)
    conv = nn.Conv2d(3, 8, 3, padding=1).cuda()
    ok = True
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
            results[name] = {"status": "unsupported"}
            continue
        with torch.autocast("cuda", dtype=dtype):
            mm = a @ a
            cv = conv(x)
        entry = {
            "matmul_dtype": str(mm.dtype),
            "conv_dtype": str(cv.dtype),
            "finite": _finite(mm) and _finite(cv),
        }
        entry["status"] = "pass" if entry["finite"] and mm.dtype == dtype else "fail"
        ok &= entry["status"] == "pass"
        results[name] = entry
    results["status"] = "pass" if ok else "fail"
    return results


def ctc_check() -> dict[str, Any]:
    """CTC loss under bf16 autocast, with the loss computed in an fp32 region (R-13)."""
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "no CUDA device"}
    torch.manual_seed(0)
    t_len, batch, classes, tgt_len = 64, 8, 80, 20
    head = nn.Linear(128, classes).cuda()
    feats = torch.randn(t_len, batch, 128, device="cuda")
    targets = torch.randint(1, classes, (batch, tgt_len), device="cuda")
    in_lens = torch.full((batch,), t_len, dtype=torch.long)
    tg_lens = torch.full((batch,), tgt_len, dtype=torch.long)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    ref = ctc(head(feats).float().log_softmax(-1), targets, in_lens, tg_lens)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    head.zero_grad()
    with torch.autocast("cuda", dtype=dtype):
        logits = head(feats)
        with fp32_region("cuda"):
            loss = ctc(logits.float().log_softmax(-1), targets, in_lens, tg_lens)
    loss.backward()
    grad = head.weight.grad
    rel_diff = abs(loss.item() - ref.item()) / max(abs(ref.item()), 1e-12)
    result = {
        "autocast_dtype": str(dtype),
        "logits_dtype_under_autocast": str(logits.dtype),
        "loss_dtype": str(loss.dtype),
        "loss_fp32_reference": ref.item(),
        "loss_amp_fp32_region": loss.item(),
        "relative_difference": rel_diff,
        "loss_finite": math.isfinite(loss.item()),
        "grad_finite": grad is not None and _finite(grad),
    }
    passed = result["loss_finite"] and result["grad_finite"] and loss.dtype == torch.float32
    result["status"] = "pass" if passed else "fail"
    return result


def determinism_check(steps: int) -> dict[str, Any]:
    out: dict[str, Any] = {"steps": steps}
    a = run_toy_training(seed=1337, steps=steps)
    b = run_toy_training(seed=1337, steps=steps)
    c = run_toy_training(seed=7, steps=steps)
    out["cpu"] = {
        "bit_identical": a == b,
        "different_seed_differs": a != c,
        "first_loss": a[0],
        "last_loss": a[-1],
        "status": "pass" if a == b and a != c else "fail",
    }
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        g1 = run_toy_training(seed=1337, steps=steps, device=dev)
        g2 = run_toy_training(seed=1337, steps=steps, device=dev)
        max_abs = max(abs(x - y) for x, y in zip(g1, g2, strict=True))
        max_rel = max(abs(x - y) / max(abs(x), 1e-12) for x, y in zip(g1, g2, strict=True))
        amp_run = run_toy_training(seed=1337, steps=steps, device=dev, precision="bf16")
        out["cuda"] = {
            "bit_identical": g1 == g2,
            "max_abs_difference": max_abs,
            "max_rel_difference": max_rel,
            "tolerance_rel": CUDA_REL_TOLERANCE,
            "bf16_run_finite": all(math.isfinite(v) for v in amp_run),
            "bf16_last_loss": amp_run[-1],
            "status": "pass"
            if max_rel <= CUDA_REL_TOLERANCE and all(math.isfinite(v) for v in amp_run)
            else "fail",
        }
    else:
        out["cuda"] = {"status": "skipped", "reason": "no CUDA device"}
    return out


def to_markdown(report: dict[str, Any]) -> str:
    s, t = report["system"], report["torch"]
    lines = [
        "# Environment report",
        "",
        f"Generated by `scripts/env_report.py` at {report['generated_utc']} "
        f"(glypharis {report['glypharis_version']}, commit `{report['git']['commit']}`"
        f"{', dirty tree' if report['git']['dirty'] else ''}). Do not edit by hand.",
        "",
        "## System",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| OS | {s['os']} ({s['os_version']}) |",
        f"| CPU cores | {s['cpu_physical_cores']} physical / {s['cpu_logical_cores']} logical |",
        f"| RAM | {s['ram_total_gib']} GiB |",
        f"| Python | {s['python']} (virtualenv: {s['in_virtualenv']}) |",
        f"| PyTorch | {t['torch']} (CUDA runtime {t['cuda_runtime']}, cuDNN {t['cudnn']}) |",
    ]
    if "gpu" in t:
        g = t["gpu"]
        lines += [
            f"| GPU | {g['name']}, CC {g['compute_capability']}, {g['multiprocessors']} SMs |",
            f"| VRAM | {g['total_mib']} MiB total, {g['free_mib_at_report']} MiB free at report |",
            f"| bf16 | {g['bf16_supported']} |",
            f"| Driver | {g['driver']} |",
        ]
    else:
        lines.append("| GPU | none (CPU only) |")
    lines += ["", "## Packages", "", "| Package | Version |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in report["packages"].items()]
    lines += ["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"]
    amp = report["checks"]["amp"]
    amp_detail = ", ".join(
        f"{k}: {v.get('matmul_dtype', v.get('status'))}"
        for k, v in amp.items()
        if isinstance(v, dict)
    )
    lines.append(f"| AMP autocast | {amp['status']} | {amp_detail or amp.get('reason', '')} |")
    ctc = report["checks"]["ctc"]
    ctc_detail = (
        f"loss dtype {ctc['loss_dtype']}, rel. diff vs fp32 {ctc['relative_difference']:.2e}"
        if ctc["status"] != "skipped"
        else ctc["reason"]
    )
    lines.append(f"| CTC in fp32 under autocast | {ctc['status']} | {ctc_detail} |")
    det = report["checks"]["determinism"]
    lines.append(
        f"| CPU determinism ({det['steps']} steps) | {det['cpu']['status']} | "
        f"bit-identical: {det['cpu']['bit_identical']} |"
    )
    cu = det["cuda"]
    cu_detail = (
        f"max rel. diff {cu['max_rel_difference']:.2e} (tolerance {cu['tolerance_rel']:.0e}), "
        f"bit-identical: {cu['bit_identical']}, bf16 finite: {cu['bf16_run_finite']}"
        if cu["status"] != "skipped"
        else cu["reason"]
    )
    lines.append(f"| CUDA determinism | {cu['status']} | {cu_detail} |")
    lines += ["", f"**Overall: {report['overall']}**", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="reports/environment", help="output directory")
    parser.add_argument("--steps", type=int, default=50, help="toy training steps")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "glypharis_version": __version__,
        "git": git_info().to_dict(),
        "system": system_info(),
        "torch": torch_info(),
        "packages": {p: _version(p) for p in PACKAGES},
        "env": {k: os.environ.get(k) for k in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")},
        "checks": {
            "amp": amp_check(),
            "ctc": ctc_check(),
            "determinism": determinism_check(args.steps),
        },
    }
    checks = report["checks"]
    statuses = [
        checks["amp"]["status"],
        checks["ctc"]["status"],
        checks["determinism"]["cpu"]["status"],
        checks["determinism"]["cuda"]["status"],
    ]
    report["overall"] = "fail" if "fail" in statuses else "pass"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "env_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out / "env_report.md").write_text(to_markdown(report), encoding="utf-8", newline="\n")
    print(to_markdown(report))
    return 0 if report["overall"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
