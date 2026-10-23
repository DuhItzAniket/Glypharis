"""Phase 8 resume check: a detector run stopped mid-epoch and resumed matches an uninterrupted run.

    python scripts/det_resume_check.py --config configs/det/gw-det-001.yaml

Runs the config on the real hardware with a small seeded slice of the pages (800 training pages,
one crop each, so an epoch is 100 batches of 8; 50 validation and 20 train-eval pages), three
times: (A) uninterrupted for ``STEPS`` optimizer steps; (B) stopped at ``STEPS // 2 + 1`` steps,
which falls inside an epoch; (C) B resumed from its last.pt to ``STEPS``. Per-step losses, final
weights and the per-epoch ``val_f1_50`` history of A and C are compared. On the CPU they must be
identical; on CUDA the losses must agree within a maximum relative difference of 1e-4
(EXPERIMENT_PROTOCOL §11), and the largest relative weight difference is reported. With the SE
swap, nearest upsampling and sort-based OHEM exact equality is expected on CUDA too, but only the
tolerance is claimed.

Only one trainer is alive at a time: A's results are copied to the CPU and A is closed and freed
before B is built, and B before C, so each GPU preflight sees the memory returned and at most one
trainer's workers run.

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
from glypharis.training.det_trainer import DetTrainer
from glypharis.training.loop import free_cuda, read_jsonl

STEPS = 300
TRAIN_PAGES = 800
VAL_PAGES = 50
TRAIN_EVAL_PAGES = 20
REL_TOL = 1e-4  # EXPERIMENT_PROTOCOL §11, CUDA runs


def run(args: argparse.Namespace, workdir: Path, tag: str, extra: list[str]) -> dict[str, Any]:
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
    trainer = DetTrainer(load_config(args.config, overrides))
    try:
        trainer.fit()
        recs = read_jsonl(trainer.run_dir / "metrics.jsonl")
        out = {
            "losses": {r["step"]: r["step_loss"] for r in recs if r["type"] == "step"},
            "f1": [h["val_f1_50"] for h in trainer.state.history],
            "weights": {k: v.detach().cpu().clone() for k, v in trainer.model.state_dict().items()},
            "device": str(trainer.device),
            "precision": trainer.amp.name,
            "steps_per_epoch": trainer.steps_per_epoch[0],
            "last": trainer.ckpt_dir / "last.pt",
        }
    finally:
        trainer.close()
        del trainer
        free_cuda()
    return out


def max_rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    args = parser.parse_args(argv)
    base = load_config(args.config, overrides_from_args(args))
    work = Path(tempfile.mkdtemp(prefix="det_resume_check_"))
    try:
        a = run(args, work, "a", [f"train.max_steps={STEPS}"])
        cut = STEPS // 2 + 1  # inside an epoch at the real settings
        b = run(args, work, "b", [f"train.max_steps={cut}"])
        c = run(args, work, "b", [f"checkpoint.resume={b['last']}", f"train.max_steps={STEPS}"])
        common = sorted(set(a["losses"]) & set(c["losses"]))
        loss_rel = max((max_rel(a["losses"][s], c["losses"][s]) for s in common), default=0.0)
        weight_rel = 0.0
        for name, wa in a["weights"].items():
            wc = c["weights"][name]
            if wa.is_floating_point():
                denom = wa.abs().max().clamp(min=1e-12)
                weight_rel = max(weight_rel, float((wa - wc).abs().max() / denom))
            elif not torch.equal(wa, wc):
                weight_rel = max(weight_rel, 1.0)
        cpu = a["device"] == "cpu"
        f1_equal = a["f1"] == c["f1"]
        if cpu:
            passed = loss_rel == 0 and weight_rel == 0 and f1_equal and bool(common)
        else:
            passed = loss_rel <= REL_TOL and bool(common)
        result: dict[str, Any] = {
            "device": a["device"],
            "precision": a["precision"],
            "steps": STEPS,
            "stopped_at": cut,
            "stopped_mid_epoch": cut % a["steps_per_epoch"] != 0,
            "steps_compared": len(common),
            "max_rel_loss_diff": loss_rel,
            "max_rel_weight_diff": weight_rel,
            "val_f1_50_a": a["f1"],
            "val_f1_50_c": c["f1"],
            "val_f1_50_equal": f1_equal,
            "tolerance": 0.0 if cpu else REL_TOL,
            "passed": passed,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)
        free_cuda()
    out = Path(base.paths.reports_dir) / "training" / "sanity" / str(base.experiment_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resume_check.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    line = (
        f"- **Resume check**: stopped at step {cut} of {STEPS} (mid-epoch: "
        f"{result['stopped_mid_epoch']}), resumed; largest relative loss difference "
        f"{loss_rel:.2e} over {len(common)} steps (tolerance {result['tolerance']:g}), largest "
        f"relative weight difference {weight_rel:.2e}, val_f1_50 history equal: {f1_equal}: "
        f"{'PASS' if passed else 'FAIL'}\n"
    )
    with (out / "sanity.md").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
