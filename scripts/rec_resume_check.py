"""Phase 6 exit criterion: a run stopped part-way and resumed matches an uninterrupted run.

    python scripts/rec_resume_check.py --config configs/rec/gw-rec-001.yaml

Runs the config on the real hardware with a small seeded slice of the data (so an "epoch" is
short), three times, each to ``STEPS`` optimizer steps:

- A and B: two uninterrupted runs with the same seed. B also keeps its mid-epoch checkpoint from
  step ``STEPS // 2 + 1`` (the cut), written by the same periodic save a crashed run resumes from.
- C: a fresh trainer that resumes from B's checkpoint at the cut and runs on to ``STEPS``.

On the CPU training is deterministic, so C must equal B exactly, losses and weights. On CUDA, CTC
backward has no deterministic kernel, so even two uninterrupted runs drift apart. The check then
asks for three things:

1. The first step after the resume matches B's to ``BOUNDARY_TOL``. That step's loss is a forward
   pass on B's weights at the cut, so it only matches if the weights, the data position, the
   augmentation and every random state came back.
2. The learning rate matches B's at every later step (scheduler and optimizer step count).
3. C's drift from B stays within ``NOISE_FACTOR`` times the drift between A and B over the same
   steps, or within the protocol tolerance if that is larger. Resuming may not add error beyond what
   the hardware already adds (EXPERIMENT_PROTOCOL §11).

Writes reports/training/sanity/<experiment_id>/resume_check.json and appends a line to sanity.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.training.checkpoint import load_checkpoint
from glypharis.training.rec_trainer import RecTrainer

STEPS = 400
TRAIN_LINES = 6400  # 100 batches of 64: four short epochs within STEPS
VAL_LINES = 2000
REL_TOL = 1e-4  # EXPERIMENT_PROTOCOL §11, CUDA runs
BOUNDARY_TOL = 1e-6  # first step after the resume: a forward pass, so it should match exactly
NOISE_FACTOR = 2.0


def run(
    args: argparse.Namespace,
    workdir: Path,
    tag: str,
    extra: list[str],
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> RecTrainer:
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
    trainer = RecTrainer(load_config(args.config, overrides), on_step=on_step)
    trainer.fit()
    return trainer


def keep_checkpoint_at(ckpt_dir: Path, step: int, name: str) -> Callable[[dict[str, Any]], None]:
    """Step hook that keeps a copy of the last.pt saved at ``step``. The periodic save comes right
    after that step's record is logged, so the copy is taken when the next step is logged (the cut
    is mid-epoch, so no other save happens in between)."""

    def hook(rec: dict[str, Any]) -> None:
        if rec.get("type") == "step" and rec["step"] == step + 1:
            shutil.copy2(ckpt_dir / "last.pt", ckpt_dir / name)

    return hook


def series(trainer: RecTrainer, key: str) -> dict[int, float]:
    path = trainer.run_dir / "metrics.jsonl"
    recs = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return {r["step"]: r[key] for r in recs if r["type"] == "step"}


def weights(trainer: RecTrainer) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()}


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def weight_diff(wa: dict[str, torch.Tensor], wb: dict[str, torch.Tensor]) -> float:
    """Largest difference of any weight tensor, relative to that tensor's largest value."""
    out = 0.0
    for k, a in wa.items():
        if a.is_floating_point():
            out = max(out, float((a - wb[k]).abs().max() / a.abs().max().clamp(min=1e-12)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    args = parser.parse_args(argv)
    base = load_config(args.config, overrides_from_args(args))
    cut = STEPS // 2 + 1  # inside an epoch at the real settings
    full = [f"train.max_steps={STEPS}", f"checkpoint.every_steps={cut}"]
    work = Path(tempfile.mkdtemp(prefix="resume_check_"))
    try:
        a = run(args, work, "a", full)
        loss_a, w_a, device, precision = series(a, "step_loss"), weights(a), a.device, a.amp.name
        mid_epoch = cut % a.steps_per_epoch[0] != 0
        del a

        b = run(args, work, "b", full, keep_checkpoint_at(work / "b" / "ckpt", cut, "cut.pt"))
        loss_b, lr_b, w_b = series(b, "step_loss"), series(b, "lr"), weights(b)
        val_b = b.state.history[-1]["val_cer"] if b.state.history else None
        del b
        # C needs B's config (paths included) to resume, but its own log: move B's run aside.
        shutil.move(work / "b", work / "b_done")
        resume_from = work / "b" / "ckpt" / "cut.pt"
        resume_from.parent.mkdir(parents=True)
        shutil.copy2(work / "b_done" / "ckpt" / "cut.pt", resume_from)
        if load_checkpoint(resume_from)["state"]["step"] != cut:
            raise SystemExit(f"the kept checkpoint is not the one from step {cut}")

        c = run(args, work, "b", [*full, f"checkpoint.resume={resume_from}"])
        loss_c, lr_c, w_c = series(c, "step_loss"), series(c, "lr"), weights(c)
        val_c = c.state.history[-1]["val_cer"] if c.state.history else None
        del c
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    after = list(range(cut + 1, STEPS + 1))
    if any(s not in loss_c or s not in loss_b or s not in loss_a for s in after):
        raise SystemExit("a run is missing step records after the cut")
    boundary = rel(loss_b[cut + 1], loss_c[cut + 1])
    lr_same = all(lr_b[s] == lr_c[s] for s in after)
    drift_resumed = max(rel(loss_b[s], loss_c[s]) for s in after)
    drift_two_runs = max(rel(loss_a[s], loss_b[s]) for s in after)
    allowed = max(REL_TOL, NOISE_FACTOR * drift_two_runs)
    w_resumed, w_two_runs = weight_diff(w_b, w_c), weight_diff(w_a, w_b)
    cpu = device.type == "cpu"
    if cpu:
        passed = boundary == 0 and lr_same and drift_resumed == 0 and w_resumed == 0
    else:
        passed = boundary <= BOUNDARY_TOL and lr_same and drift_resumed <= allowed
    result: dict[str, Any] = {
        "device": str(device), "precision": precision, "steps": STEPS, "cut": cut,
        "cut_mid_epoch": mid_epoch, "steps_compared": len(after),
        "boundary_rel_diff": boundary, "boundary_tol": 0.0 if cpu else BOUNDARY_TOL,
        "lr_identical": lr_same,
        "max_rel_loss_diff_resumed": drift_resumed,
        "max_rel_loss_diff_two_uninterrupted": drift_two_runs,
        "allowed": 0.0 if cpu else allowed, "noise_factor": NOISE_FACTOR, "rel_tol": REL_TOL,
        "max_rel_weight_diff_resumed": w_resumed,
        "max_rel_weight_diff_two_uninterrupted": w_two_runs,
        "val_cer_b": val_b, "val_cer_c": val_c,
        "passed": passed,
    }  # fmt: skip
    out = Path(base.paths.reports_dir) / "training" / "sanity" / str(base.experiment_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume_check.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    line = (f"- **Resume check** (two uninterrupted runs A, B; C resumed from B's checkpoint at "
            f"step {cut} of {STEPS}, mid-epoch: {mid_epoch}): first step after the resume differs "
            f"by {boundary:.2e} (tolerance {result['boundary_tol']:g}); learning rates identical: "
            f"{lr_same}; largest relative loss difference C vs B {drift_resumed:.2e}, A vs B "
            f"{drift_two_runs:.2e} (allowed {result['allowed']:.2e}); largest relative weight "
            f"difference C vs B {w_resumed:.2e}, A vs B {w_two_runs:.2e}: "
            f"{'PASS' if passed else 'FAIL'}\n")  # fmt: skip
    with (out / "sanity.md").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
