"""Phase 6 exit criterion: a run killed part-way and resumed matches an uninterrupted run.

    python scripts/rec_resume_check.py --config configs/rec/gw-rec-001.yaml

Runs the config on the real hardware with a small seeded slice of the data (so an "epoch" is short),
three times: (A) uninterrupted for ``STEPS`` optimizer steps; (B) stopped at ``STEPS // 2 + 1``
steps, which falls inside an epoch; (C) B resumed from its last.pt to ``STEPS``. A and C must agree
within the protocol tolerance: on CUDA, CTC backward has no deterministic kernel, so the comparison
uses a maximum relative difference of 1e-4 on the logged losses (EXPERIMENT_PROTOCOL §11) and
reports the largest relative weight difference; on the CPU they must be identical.

Writes reports/training/sanity/<experiment_id>/resume_check.json and appends a line to sanity.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.training.rec_trainer import RecTrainer

STEPS = 400
TRAIN_LINES = 6400  # 100 batches of 64: four short epochs within STEPS
VAL_LINES = 2000
REL_TOL = 1e-4  # EXPERIMENT_PROTOCOL §11, CUDA runs


def run(args: argparse.Namespace, workdir: Path, tag: str, extra: list[str]) -> RecTrainer:
    overrides = [
        *overrides_from_args(args),
        f"paths.runs_dir={workdir / tag / 'runs'}",
        f"paths.reports_dir={workdir / tag / 'reports'}",
        f"checkpoint.dir={workdir / tag / 'ckpt'}",
        f"data.dataset_subset={TRAIN_LINES}",
        f"data.val_subset={VAL_LINES}",
        f"data.train_eval_subset={VAL_LINES}",
        "logging.log_every_n_steps=1",
        *extra,
    ]
    trainer = RecTrainer(load_config(args.config, overrides))
    trainer.fit()
    return trainer


def losses(trainer: RecTrainer) -> dict[int, float]:
    path = trainer.run_dir / "metrics.jsonl"
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return {r["step"]: r["step_loss"] for r in recs if r["type"] == "step"}


def max_rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    args = parser.parse_args(argv)
    base = load_config(args.config, overrides_from_args(args))
    work = Path(tempfile.mkdtemp(prefix="resume_check_"))
    try:
        a = run(args, work, "a", [f"train.max_steps={STEPS}"])
        cut = STEPS // 2 + 1  # one step into an epoch at the real settings
        run(args, work, "b", [f"train.max_steps={cut}"])
        last = work / "b" / "ckpt" / "last.pt"
        c = run(args, work, "b", [f"checkpoint.resume={last}", f"train.max_steps={STEPS}"])
        la, lc = losses(a), losses(c)
        common = sorted(set(la) & set(lc))
        loss_rel = max(max_rel(la[s], lc[s]) for s in common)
        weight_rel = 0.0
        for wa, wc in zip(a.model.state_dict().values(), c.model.state_dict().values(),
                          strict=True):  # fmt: skip
            if wa.is_floating_point():
                denom = wa.abs().max().clamp(min=1e-12)
                weight_rel = max(weight_rel, float((wa - wc).abs().max() / denom))
        cpu = a.device.type == "cpu"
        passed = (loss_rel == 0 and weight_rel == 0) if cpu else loss_rel <= REL_TOL
        result: dict[str, Any] = {
            "device": str(a.device), "precision": a.amp.name, "steps": STEPS, "stopped_at": cut,
            "stopped_mid_epoch": cut % a.steps_per_epoch[0] != 0,
            "steps_compared": len(common), "max_rel_loss_diff": loss_rel,
            "max_rel_weight_diff": weight_rel, "tolerance": 0.0 if cpu else REL_TOL,
            "val_cer_a": a.state.history[-1]["val_cer"] if a.state.history else None,
            "val_cer_c": c.state.history[-1]["val_cer"] if c.state.history else None,
            "passed": passed,
        }  # fmt: skip
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out = Path(base.paths.reports_dir) / "training" / "sanity" / str(base.experiment_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume_check.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    line = (f"- **Resume check**: stopped at step {result['stopped_at']} of {STEPS} "
            f"(mid-epoch: {result['stopped_mid_epoch']}), resumed; largest relative loss "
            f"difference {result['max_rel_loss_diff']:.2e} over {result['steps_compared']} steps "
            f"(tolerance {result['tolerance']:g}), largest relative weight difference "
            f"{result['max_rel_weight_diff']:.2e}: {'PASS' if passed else 'FAIL'}\n")  # fmt: skip
    with (out / "sanity.md").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
