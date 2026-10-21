from __future__ import annotations

import json
import logging
import re

import pytest
import torch

from glypharis.utils.device import fp32_region, resolve_amp, select_device
from glypharis.utils.gitinfo import git_info
from glypharis.utils.logsetup import setup_logging
from glypharis.utils.monitor import snapshot

# ----------------------------------------------------------------------------------- device


def test_select_cpu():
    assert select_device("cpu").type == "cpu"


def test_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_device("auto").type == "cpu"


def test_explicit_cuda_fails_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        select_device("cuda")


def test_unknown_device_request():
    with pytest.raises(ValueError):
        select_device("tpu")  # type: ignore[arg-type]


def test_cpu_always_fp32():
    for request in ("bf16", "fp16", "none"):
        amp = resolve_amp(request, torch.device("cpu"))  # type: ignore[arg-type]
        assert amp.dtype == torch.float32
        assert not amp.enabled
        assert not amp.needs_grad_scaler


def test_cuda_precision_fallback(monkeypatch):
    cuda = torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_amp("bf16", cuda).name == "bf16"
    assert not resolve_amp("bf16", cuda).needs_grad_scaler
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    fallback = resolve_amp("bf16", cuda)
    assert fallback.name == "fp16"
    assert fallback.needs_grad_scaler
    assert resolve_amp("none", cuda).name == "fp32"


def test_unknown_precision():
    with pytest.raises(ValueError):
        resolve_amp("fp8", torch.device("cpu"))  # type: ignore[arg-type]


@pytest.mark.gpu
def test_fp32_region_disables_autocast():
    lin = torch.nn.Linear(8, 8).cuda()
    x = torch.randn(2, 8, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert lin(x).dtype == torch.bfloat16
        with fp32_region("cuda"):
            assert lin(x).dtype == torch.float32


# ---------------------------------------------------------------------------------- monitor


def test_cpu_snapshot():
    snap = snapshot(torch.device("cpu"))
    assert snap.process_rss_mib > 0
    assert 0 < snap.system_ram_used_mib <= snap.system_ram_total_mib
    assert snap.cuda_allocated_mib is None
    assert set(snap.to_dict()) >= {"process_rss_mib", "gpu_temperature_c"}


@pytest.mark.gpu
def test_cuda_snapshot():
    dev = torch.device("cuda")
    keep = torch.empty(64 * 1024**2 // 4, device=dev)  # 64 MiB
    snap = snapshot(dev)
    assert snap.cuda_allocated_mib is not None and snap.cuda_allocated_mib >= 63
    assert snap.cuda_total_mib is not None and snap.cuda_free_mib is not None
    del keep


# ------------------------------------------------------------------------------------- git


def test_git_info_outside_repo(tmp_path):
    info = git_info(tmp_path)
    assert info.commit is None and info.branch is None and info.dirty is None


def test_git_info_inside_repo(repo_root):
    info = git_info(repo_root)
    if info.commit is None:
        pytest.skip("not running from a git checkout")
    assert re.fullmatch(r"[0-9a-f]{40}", info.commit)
    assert isinstance(info.dirty, bool)


# --------------------------------------------------------------------------------- logging


def test_json_log_has_run_id_and_extra_fields(tmp_path):
    path = setup_logging(tmp_path, run_id="GW-REC-000-test")
    assert path is not None
    logging.getLogger("glypharis.test").info("step done", extra={"metrics": {"loss": 1.5}})
    for handler in logging.getLogger().handlers:
        handler.flush()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rec = records[-1]
    assert rec["message"] == "step done"
    assert rec["run_id"] == "GW-REC-000-test"
    assert rec["metrics"] == {"loss": 1.5}
    setup_logging()  # detach the file handler so tmp_path can be removed on Windows


def test_setup_logging_is_idempotent(tmp_path):
    setup_logging(tmp_path, run_id="a")
    setup_logging(tmp_path, run_id="b")
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_glypharis", False)]
    assert len(ours) == 2  # one console + one file, not four
    setup_logging()


# ------------------------------------------------------------------------- import weight


def test_data_tools_do_not_import_torch():
    """Data-preparation workers import these; each torch import costs about 500 MB per process."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import glypharis.utils, glypharis.utils.gitinfo, glypharis.utils.logsetup\n"
        "import glypharis.utils.monitor\n"
        "import glypharis.data.audit, glypharis.data.crops, glypharis.data.pages\n"
        "import glypharis.data.linepack, glypharis.data.download, glypharis.data.text\n"
        "glypharis.utils.monitor.snapshot()\n"
        "print('torch' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
