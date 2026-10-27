"""Phase 8 resume check: a detector run stopped mid-epoch and resumed matches an uninterrupted run.

    python scripts/det_resume_check.py --config configs/det/gw-det-001.yaml

Runs the config on the real hardware with a small seeded slice of the pages (800 training pages,
one crop each, so an epoch is 100 batches of 8; 50 validation and 20 train-eval pages), three
times, each to ``STEPS`` optimizer steps, with the design the protocol fixed for CUDA resume checks
on 2026-10-23 (EXPERIMENT_PROTOCOL §11, as in `scripts/rec_resume_check.py`):

- A and B: two uninterrupted runs with the same seed. B also keeps its mid-epoch checkpoint from
  step ``STEPS // 2 + 1`` (the cut), written by the same periodic save a crashed run resumes from.
- C: a fresh trainer that resumes from B's checkpoint at the cut and runs on to ``STEPS``.

On the CPU, C must equal B exactly: losses, weights and the per-epoch ``val_f1_50`` history. On
CUDA the check asks for three things:

1. The first step after the resume matches B's to ``BOUNDARY_TOL``. That step's loss is a forward
   pass on B's weights at the cut, so it only matches if the weights, the data position, the
   augmentation and every random state came back.
2. The learning rate matches B's at every later step.
3. C's drift from B stays within ``NOISE_FACTOR`` times the drift between A and B over the same
   steps, or within the protocol tolerance (1e-4) if that is larger: resuming may not add error
   beyond what the hardware already adds. With the SE swap, nearest upsampling and sort-based OHEM
   the detector is expected to be deterministic on CUDA too, in which case A and B agree exactly
   and this reduces to the 1e-4 tolerance; only the rule is claimed.

Only one trainer is alive at a time: each run's results are copied to the CPU and its trainer is
closed and freed before the next is built, so each GPU preflight sees the memory returned and at
most one trainer's workers run.

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
from glypharis.training.det_trainer import DetTrainer
from glypharis.training.loop import free_cuda, read_jsonl

STEPS = 300
TRAIN_PAGES = 800
VAL_PAGES = 50
TRAIN_EVAL_PAGES = 20
REL_TOL = 1e-4  # EXPERIMENT_PROTOCOL §11, CUDA runs
BOUNDARY_TOL = 1e-6  # first step after the resume: a forward pass, so it should match exactly
NOISE_FACTOR = 2.0


def keep_checkpoint_at(ckpt_dir: Path, step: int, name: str) -> Callable[[dict[str, Any]], None]:
    """Step hook that keeps a copy of the last.pt saved at ``step``. The periodic save comes right
    after that step's record is logged, so the copy is taken when the next step is logged (the cut
    is mid-epoch, so no other save happens in between)."""

    def hook(rec: dict[str, Any]) -> None:
        if rec.get("type") == "step" and rec["step"] == step + 1:
            shutil.copy2(ckpt_dir / "last.pt", ckpt_dir / name)

    return hook


def run(
    args: argparse.Namespace,
    workdir: Path,
    tag: str,
    extra: list[str],
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build, fit, copy the results to the CPU, then close and free the trainer."""
    overrides = [
        *overrides_from_args(args),
        f"paths.runs_dir={workdir / tag / 'runs'}",
        f"paths.reports_dir={workdir / tag / 'reports'}",
        f"checkpoint.dir={workdir / tag / 'ckpt'}",
        f"det.data.train_subset={TRAIN_PAGES}",
        "det.data.crops_per_page=1",
        f"det.data.val_subset={VAL_PAGES}",
        f"det.data.train_eval_subset={TRAIN_EVAL_PAGES}",
        "logging.log_every_n_steps=1",
        *extra,
    ]
    trainer = DetTrainer(load_config(args.config, overrides), on_step=on_step)
    try:
        trainer.fit()
        recs = read_jsonl(trainer.run_dir / "metrics.jsonl")
        steps = [r for r in recs if r["type"] == "step"]
        out = {
            "losses": {r["step"]: r["step_loss"] for r in steps},
            "lrs": {r["step"]: r["lr"] for r in steps},
            "f1": [h["val_f1_50"] for h in trainer.state.history],
            "weights": {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()},
            "device": str(trainer.device),
            "precision": trainer.amp.name,
            "steps_per_epoch": trainer.steps_per_epoch[0],
        }
    finally:
        trainer.close()
        del trainer
        free_cuda()
    return out


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def weight_diff(wa: dict[str, torch.Tensor], wb: dict[str, torch.Tensor]) -> float:
    """Largest difference of any weight tensor, relative to that tensor's largest value; a
    differing integer buffer counts as 1."""
    out = 0.0
    for k, a in wa.items():
        b = wb[k]
        if a.is_floating_point():
            out = max(out, float((a - b).abs().max() / a.abs().max().clamp(min=1e-12)))
        elif not torch.equal(a, b):
            out = max(out, 1.0)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    args = parser.parse_args(argv)
    base = load_config(args.config, overrides_from_args(args))
    cut = STEPS // 2 + 1  # inside an epoch at the real settings
    full = [f"train.max_steps={STEPS}", f"checkpoint.every_steps={cut}"]
    work = Path(tempfile.mkdtemp(prefix="det_resume_check_"))
    try:
        a = run(args, work, "a", full)
        b = run(args, work, "b", full, keep_checkpoint_at(work / "b" / "ckpt", cut, "cut.pt"))
        # C needs B's config (paths included) to resume, but its own log: move B's run aside.
        shutil.move(work / "b", work / "b_done")
        resume_from = work / "b" / "ckpt" / "cut.pt"
        resume_from.parent.mkdir(parents=True)
        shutil.copy2(work / "b_done" / "ckpt" / "cut.pt", resume_from)
        if load_checkpoint(resume_from)["state"]["step"] != cut:
            raise SystemExit(f"the kept checkpoint is not the one from step {cut}")
        c = run(args, work, "b", [*full, f"checkpoint.resume={resume_from}"])
    finally:
        shutil.rmtree(work, ignore_errors=True)
        free_cuda()

    after = list(range(cut + 1, STEPS + 1))
    if any(s not in r["losses"] for r in (a, b, c) for s in after):
        raise SystemExit("a run is missing step records after the cut")
    boundary = rel(b["losses"][cut + 1], c["losses"][cut + 1])
    lr_same = all(b["lrs"][s] == c["lrs"][s] for s in after)
    drift_resumed = max(rel(b["losses"][s], c["losses"][s]) for s in after)
    drift_two_runs = max(rel(a["losses"][s], b["losses"][s]) for s in after)
    allowed = max(REL_TOL, NOISE_FACTOR * drift_two_runs)
    w_resumed = weight_diff(b["weights"], c["weights"])
    w_two_runs = weight_diff(a["weights"], b["weights"])
    f1_equal = b["f1"] == c["f1"]
    cpu = a["device"] == "cpu"
    if cpu:
        passed = boundary == 0 and lr_same and drift_resumed == 0 and w_resumed == 0 and f1_equal
    else:
        passed = boundary <= BOUNDARY_TOL and lr_same and drift_resumed <= allowed
    result: dict[str, Any] = {
        "device": a["device"], "precision": a["precision"], "steps": STEPS, "cut": cut,
        "cut_mid_epoch": cut % a["steps_per_epoch"] != 0, "steps_compared": len(after),
        "boundary_rel_diff": boundary, "boundary_tol": 0.0 if cpu else BOUNDARY_TOL,
        "lr_identical": lr_same,
        "max_rel_loss_diff_resumed": drift_resumed,
        "max_rel_loss_diff_two_uninterrupted": drift_two_runs,
        "allowed": 0.0 if cpu else allowed, "noise_factor": NOISE_FACTOR, "rel_tol": REL_TOL,
        "max_rel_weight_diff_resumed": w_resumed,
        "max_rel_weight_diff_two_uninterrupted": w_two_runs,
        "val_f1_50_b": b["f1"], "val_f1_50_c": c["f1"], "val_f1_50_equal": f1_equal,
        "passed": passed,
    }  # fmt: skip
    out = Path(base.paths.reports_dir) / "training" / "sanity" / str(base.experiment_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume_check.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    line = (f"- **Resume check** (two uninterrupted runs A, B; C resumed from B's checkpoint at "
            f"step {cut} of {STEPS}, mid-epoch: {result['cut_mid_epoch']}): first step after the "
            f"resume differs by {boundary:.2e} (tolerance {result['boundary_tol']:g}); learning "
            f"rates identical: {lr_same}; largest relative loss difference C vs B "
            f"{drift_resumed:.2e}, A vs B {drift_two_runs:.2e} (allowed {result['allowed']:.2e}); "
            f"largest relative weight difference C vs B {w_resumed:.2e}, A vs B {w_two_runs:.2e}; "
            f"val_f1_50 history equal: {f1_equal}: {'PASS' if passed else 'FAIL'}\n")  # fmt: skip
    with (out / "sanity.md").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
