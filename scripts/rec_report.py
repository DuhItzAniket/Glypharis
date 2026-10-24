"""Regenerate a recognizer run's report from what the run left on disk.

    python scripts/rec_report.py --config configs/rec/gw-rec-002.yaml

Nothing is trained or evaluated: the datasets are opened only to crop the worst lines, and best.pt
only to measure the model's size for runs that did not record it. Summaries written at the end of
a resumed run before commit 6ea51f3 hashed the config including its --resume pointer, so they did
not show the run's own config hash. When the stored hash differs from the run's, this replaces it,
keeps the old value as ``config_hash_as_written`` and notes the change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from glypharis.config.cli import add_config_args, config_from_args
from glypharis.config.loader import config_to_dict, load_config
from glypharis.training.checkpoint import load_checkpoint
from glypharis.training.rec_trainer import (
    RecTrainer,
    config_matches,
    model_bytes,
    run_config_hash,
)
from glypharis.training.report import worst_images, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    trainer = RecTrainer(cfg)
    run_dir = trainer.run_dir
    written = load_config(run_dir / "config.resolved.yaml")
    if not config_matches(config_to_dict(written), cfg):
        raise SystemExit(f"{args.config} is not the config {run_dir} was trained with")
    path = run_dir / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if "model_bytes" not in summary:  # runs from before it was recorded: measure best.pt
        state = load_checkpoint(trainer.ckpt_dir / "best.pt")["model"]
        summary["model_bytes"] = model_bytes(state)
        path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    want = run_config_hash(cfg)
    if summary["config_hash"] != want:
        old = summary["config_hash"]
        summary["config_hash_as_written"] = old
        summary["config_hash"] = want
        summary["notes"] = [*summary.get("notes", []), (
            f"Config hash corrected on {dt.date.today().isoformat()} with scripts/rec_report.py: "
            f"the summary written at the end of the resumed run hashed the config including its "
            f"resume pointer ({old[:16]}); {want[:16]} is the run's config hash, as at its start "
            f"({args.config}).")]  # fmt: skip
        path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print(f"{cfg.experiment_id}: config hash {old[:16]} -> {want[:16]}")
    report = write_report(run_dir, trainer.report_dir, worst_images(run_dir, trainer.val_ds))
    print(f"{cfg.experiment_id}: report {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
