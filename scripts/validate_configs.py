"""Validate every YAML config under configs/ (run in CI).

Usage:
    python scripts/validate_configs.py [--root configs]

Each file is loaded with inheritance, interpolation and strict type checks. Files under
configs/experiments/ must also carry a complete run identity (experiment_id, task, model_name).
Files under configs/final/ are freeze records (`scripts/freeze_final.py`), not training configs:
the recognizer and detector configs they name must load and belong to the experiment named, and
each checkpoint must carry a SHA-256.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from glypharis.config import ConfigError, load_config


def check_final(path: Path, root: Path) -> None:
    record = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for part in ("recognizer", "detector"):
        entry = record.get(part)
        if not isinstance(entry, dict):
            raise ConfigError(f"no {part} section")
        config = root.parent / str(entry.get("config", ""))
        if not config.is_file():
            raise ConfigError(f"{part}.config {entry.get('config')!r} does not exist")
        if load_config(config).experiment_id != entry.get("experiment"):
            raise ConfigError(f"{part}.config is not the config of {entry.get('experiment')}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("checkpoint_sha256", ""))):
            raise ConfigError(f"{part}.checkpoint_sha256 is not a SHA-256")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate all configs.")
    parser.add_argument("--root", default="configs")
    args = parser.parse_args(argv)

    root = Path(args.root)
    files = sorted(root.rglob("*.yaml"))
    if not files:
        print(f"no configs found under {root}")
        return 1

    failures = 0
    for path in files:
        parts = path.relative_to(root).parts
        try:
            if "final" in parts:
                check_final(path, root)
            else:
                cfg = load_config(path)
                if "experiments" in parts:
                    cfg.require_run_identity()
        except ConfigError as exc:
            failures += 1
            print(f"FAIL {path.as_posix()}: {exc}")
        else:
            print(f"ok   {path.as_posix()}")
    print(f"{len(files) - failures}/{len(files)} configs valid")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
