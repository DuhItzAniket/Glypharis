"""Re-select a finished recognizer run's checkpoint from its own validation records.

Runs trained before the selection fix chose best.pt with early stopping's ``min_delta``, so an epoch
that beat the best by less than that never became the best, although protocol §7 selects the lowest
validation CER. This picks the epoch with the strictly lowest validation CER in the run's
metrics.jsonl, checks that its epoch checkpoint is the one that produced that record, re-scores it
on the validation set and rewrites best.pt, val_best_samples.jsonl, summary.json and the report.
Nothing is retrained and no test data is read. The superseded best.pt and samples are kept beside
the new ones, and the summary (and so the report) records the change.

    python scripts/rec_reselect.py --config configs/rec/gw-rec-002.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from glypharis.config.cli import add_config_args, config_from_args
from glypharis.config.schema import RunConfig
from glypharis.training.checkpoint import load_checkpoint, save_checkpoint
from glypharis.training.rec_trainer import RecTrainer, config_matches, evaluate
from glypharis.training.report import worst_images, write_report
from glypharis.utils.gitinfo import git_info

RESCORE_TOL = 1e-4  # absolute CER: the re-score must reproduce the logged value


def epoch_records(path: Path) -> dict[int, dict[str, Any]]:
    """The run's validation record per epoch (the last one, if an epoch was logged twice)."""
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("type") == "epoch":
                out[int(rec["epoch"])] = rec
    return out


def check_source(ck: dict[str, Any], pick: dict[str, Any], cfg: RunConfig) -> None:
    """Refuse a checkpoint that is not the one which produced the picked validation record."""
    if not config_matches(ck["config"], cfg):
        raise SystemExit("the epoch checkpoint was written with a different config")
    st = ck["state"]
    logged = st["history"][-1] if st["history"] else {}
    same = (st["epoch"] == pick["epoch"] + 1 and st["step"] == pick["step"]
            and ck["metric"] == pick["val_cer"] == logged.get("val_cer"))  # fmt: skip
    if not same:
        raise SystemExit(f"the checkpoint for epoch {pick['epoch']} does not match its record")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    cfg = config_from_args(parser.parse_args(argv))
    trainer = RecTrainer(cfg)
    run_dir, ckpt_dir = trainer.run_dir, trainer.ckpt_dir
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found: the run has not finished")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    epochs = epoch_records(run_dir / "metrics.jsonl")
    pick = min(epochs.values(), key=lambda r: (r["val_cer"], r["epoch"]))
    old, new = int(summary["best_epoch"]), int(pick["epoch"])
    if new == old:
        print(f"{cfg.experiment_id}: epoch {old} already has the lowest validation CER")
        return 0

    src = ckpt_dir / f"epoch_{new:03d}.pt"
    if not src.exists():
        raise SystemExit(f"{src} is gone (pruned?); epoch {new} cannot be re-selected")
    ck = load_checkpoint(src)
    check_source(ck, pick, cfg)
    trainer.model.load_state_dict(ck["model"])
    val = evaluate(trainer.model, trainer._eval_loader(trainer.val_ds), trainer.charset,
                   trainer.amp, trainer.device, trainer.ctc)  # fmt: skip
    if abs(val.cer - pick["val_cer"]) > RESCORE_TOL:
        raise SystemExit(f"re-scoring epoch {new} gave val CER {val.cer:.6f}, but the run logged "
                         f"{pick['val_cer']:.6f}; refusing to re-select")  # fmt: skip

    # Keep what is being replaced.
    tag = f"superseded_e{old:03d}"
    shutil.copy2(ckpt_dir / "best.pt", ckpt_dir / f"best_{tag}.pt")
    shutil.copy2(run_dir / "val_best_samples.jsonl", run_dir / f"val_best_samples.{tag}.jsonl")

    git = git_info().to_dict()
    record = {
        "date": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "script": "scripts/rec_reselect.py",
        "git": git,
        "from_epoch": old,
        "to_epoch": new,
        "min_delta_at_training": cfg.early_stopping.min_delta,
        "logged_val_cer": pick["val_cer"],
        "rescored_val_cer": val.cer,
        "superseded": [f"best_{tag}.pt", f"val_best_samples.{tag}.jsonl"],
    }
    ck["state"] = {**ck["state"], "best_metric": pick["val_cer"], "best_epoch": new}
    ck["reselection"] = record
    save_checkpoint(ckpt_dir / "best.pt", ck)
    trainer._write_samples(val.samples)

    note = (f"best.pt re-selected on {record['date'][:10]} with scripts/rec_reselect.py "
            f"(git {str(git.get('commit'))[:7]}). Training chose best.pt with early stopping's "
            f"min_delta ({cfg.early_stopping.min_delta:g} CER), so epoch {old} "
            f"({100 * epochs[old]['val_cer']:.2f}% val CER) stayed selected although epoch {new} "
            f"reached {100 * pick['val_cer']:.2f}%. Protocol §7 selects the lowest validation CER; "
            f"the trainer now does. Re-scoring epoch {new}'s checkpoint on validation gave "
            f"{100 * val.cer:.4f}% (logged {100 * pick['val_cer']:.4f}%). Nothing was retrained "
            f"and no test data was read; the superseded best.pt and samples are kept as "
            f"best_{tag}.pt and val_best_samples.{tag}.jsonl.")  # fmt: skip
    summary.update(best_epoch=new, best=pick, reselection=record)
    summary["notes"] = [*summary.get("notes", []), note]
    summary_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    report = write_report(run_dir, trainer.report_dir, worst_images(run_dir, trainer.val_ds))
    print(f"{cfg.experiment_id}: best.pt is now epoch {new} (val CER {100 * val.cer:.4f}%), "
          f"was epoch {old}; report {report}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
