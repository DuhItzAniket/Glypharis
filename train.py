"""Train a GLYPHARIS model from a config file.

    python train.py --config configs/rec/gw-rec-001.yaml
    python train.py --config configs/rec/gw-rec-001.yaml --resume checkpoints/GW-REC-001/last.pt
    python train.py --config configs/det/gw-det-001.yaml

The run's experiment ID must be registered in research/experiments.md before it starts
(docs/WORKFLOW.md §4). Exit codes: 0 done, 2 non-finite loss, 3 out of GPU memory; in both failure
cases the state is saved and a run report is still written. A detector run (task det) also exits
with 2 on a non-finite gradient norm, and with 4 when it is refused at start-up because another
process holds the GPU; then only the log is written.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from glypharis.config.cli import add_config_args, config_from_args
from glypharis.diagnostics import NonFiniteLoss
from glypharis.training.rec_trainer import RecTrainer
from glypharis.training.report import worst_images, write_report
from glypharis.utils.device import limit_gpu_memory, select_device
from glypharis.utils.logsetup import setup_logging

log = logging.getLogger("train")


def finish(trainer: RecTrainer) -> Path:
    import json

    summary = trainer._summary()
    (trainer.run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                                  encoding="utf-8")  # fmt: skip
    return write_report(trainer.run_dir, trainer.report_dir,
                        worst_images(trainer.run_dir, trainer.val_ds))  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    cfg.require_run_identity()
    assert cfg.experiment_id is not None
    limit_gpu_memory(select_device(cfg.device))
    setup_logging(Path(cfg.paths.runs_dir) / cfg.experiment_id, cfg.experiment_id,
                  cfg.logging.level)  # fmt: skip
    if cfg.task == "det":
        from glypharis.training.det_trainer import run_det

        return run_det(cfg)
    if cfg.task != "rec":
        raise SystemExit(f"task {cfg.task!r}: no trainer exists for it yet")
    trainer = RecTrainer(cfg)
    log.info("%s: %s parameters, %d training and %d validation lines, %d steps per epoch on %s",
             cfg.experiment_id, f"{trainer.params:,}", len(trainer.train_ds), len(trainer.val_ds),
             trainer.steps_per_epoch[0], trainer.device)  # fmt: skip
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
