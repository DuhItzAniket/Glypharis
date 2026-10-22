"""Train a GLYPHARIS model from a config file.

    python train.py --config configs/rec/gw-rec-001.yaml
    python train.py --config configs/rec/gw-rec-001.yaml --resume checkpoints/GW-REC-001/last.pt

The run's experiment ID must be registered in research/experiments.md before it starts
(docs/WORKFLOW.md §4). Exit codes: 0 done, 2 non-finite loss, 3 out of GPU memory; in both failure
cases the state is saved and a run report is still written.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from glypharis.config.cli import add_config_args, config_from_args
from glypharis.diagnostics import NonFiniteLoss
from glypharis.training.rec_trainer import RecTrainer
from glypharis.training.report import WORST, write_report
from glypharis.utils.logsetup import setup_logging

log = logging.getLogger("train")


def worst_images(trainer: RecTrainer) -> dict[str, Any]:
    """Crops of the worst validation lines of the best checkpoint, for the report."""
    import json

    path = trainer.run_dir / "val_best_samples.jsonl"
    if not path.exists():
        return {}
    samples = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    wanted = {s["key"] for s in sorted(samples, key=lambda s: (-s["cer"], s["key"]))[:WORST]}
    out = {}
    for ref in trainer.val_ds.refs:
        if ref.key in wanted:
            out[ref.key] = trainer.val_ds.readers[ref.pack].image(ref.index)
    return out


def finish(trainer: RecTrainer) -> Path:
    import json

    summary = trainer._summary()
    (trainer.run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                                  encoding="utf-8")  # fmt: skip
    return write_report(trainer.run_dir, trainer.report_dir, worst_images(trainer))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    cfg.require_run_identity()
    assert cfg.experiment_id is not None
    setup_logging(Path(cfg.paths.runs_dir) / cfg.experiment_id, cfg.experiment_id,
                  cfg.logging.level)  # fmt: skip
    if cfg.task != "rec":
        raise SystemExit(f"task {cfg.task!r}: only the recognizer trainer exists so far (Phase 6)")
    trainer = RecTrainer(cfg)
    log.info("%s: %s parameters, %d training and %d validation lines, %d steps per epoch on %s",
             cfg.experiment_id, f"{trainer.params:,}", len(trainer.train_ds), len(trainer.val_ds),
             trainer.steps_per_epoch, trainer.device)  # fmt: skip
    code = 0
    try:
        trainer.fit()
    except NonFiniteLoss as exc:
        log.error("%s", exc)
        code = 2
    except torch.cuda.OutOfMemoryError:
        log.error("out of GPU memory at step %d; state saved to oom.pt", trainer.state.step)
        trainer.save("oom.pt")
        code = 3
    report = finish(trainer)
    log.info("run report: %s", report)
    return code


if __name__ == "__main__":
    sys.exit(main())
