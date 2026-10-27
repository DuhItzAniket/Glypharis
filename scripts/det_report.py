"""Regenerate a detector run report from the files the run left on disk.

    python scripts/det_report.py --config configs/det/gw-det-001.yaml

Reads ``<runs_dir>/<experiment_id>/`` (metrics, summary, full-validation files) and writes
``<reports_dir>/training/<experiment_id>/report.md`` with its figures and overlay sheets
(`glypharis.training.det_report`). Nothing is trained or evaluated again.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from glypharis.config.cli import add_config_args, config_from_args
from glypharis.training.det_report import write_det_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    cfg.require_run_identity()
    if cfg.task != "det" or cfg.det is None:
        raise SystemExit(f"{args.config}: not a detector config")
    run_dir = Path(cfg.paths.runs_dir) / str(cfg.experiment_id)
    if not (run_dir / "metrics.jsonl").exists():
        raise SystemExit(f"{run_dir}: no metrics.jsonl; has this run started?")
    report_dir = Path(cfg.paths.reports_dir) / "training" / str(cfg.experiment_id)
    print(write_det_report(run_dir, report_dir, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
