"""Validate every YAML config under configs/ (run in CI).

Usage:
    python scripts/validate_configs.py [--root configs]

Each file is loaded with inheritance, interpolation and strict type checks. Files under
configs/experiments/ must also carry a complete run identity (experiment_id, task, model_name).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from glypharis.config import ConfigError, load_config


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
        try:
            cfg = load_config(path)
            if "experiments" in path.relative_to(root).parts:
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
