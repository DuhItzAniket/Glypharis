from __future__ import annotations

import json

import env_report
import validate_configs


def test_env_report_writes_json_and_markdown(tmp_path):
    code = env_report.main(["--out", str(tmp_path), "--steps", "5"])
    report = json.loads((tmp_path / "env_report.json").read_text(encoding="utf-8"))
    md = (tmp_path / "env_report.md").read_text(encoding="utf-8")
    assert code == 0, md
    assert report["overall"] == "pass"
    assert report["checks"]["determinism"]["cpu"]["bit_identical"] is True
    assert "# Environment report" in md


def test_env_report_contains_no_paths_or_hostnames(tmp_path):
    import platform
    from pathlib import Path

    env_report.main(["--out", str(tmp_path), "--steps", "2"])
    text = (tmp_path / "env_report.json").read_text(encoding="utf-8")
    assert platform.node() not in text
    assert str(Path.home()) not in text
    assert str(Path.home()).replace("\\", "\\\\") not in text


def test_validate_configs_on_repo(repo_root):
    assert validate_configs.main(["--root", str(repo_root / "configs")]) == 0


def test_validate_configs_reports_bad_file(write_yaml, tmp_path):
    write_yaml("cfg/good.yaml", "seed: 1\n")
    write_yaml("cfg/bad.yaml", "sed: 1\n")
    assert validate_configs.main(["--root", str(tmp_path / "cfg")]) == 1


def test_experiment_configs_need_run_identity(write_yaml, tmp_path):
    write_yaml("cfg/experiments/GW-REC-001.yaml", "seed: 1\n")
    assert validate_configs.main(["--root", str(tmp_path / "cfg")]) == 1
