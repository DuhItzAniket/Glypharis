"""Freeze the final configuration from generated results (Phase 13).

    python scripts/freeze_final.py --rec-run runs/GW-REC-034 \
        --det-run <worktree>/runs/GW-DET-001 \
        --det-checkpoint <worktree>/checkpoints/GW-DET-001/best.pt

Writes ``configs/final/glypharis_v1.yaml``: the chosen recognizer and detector, their configs,
checkpoints and SHA-256, the training record of each (commit, config hash, data hashes, seed,
hardware, library versions), the detector's post-processing and resolution, the recognizer's
temperature, and the validation results the choice rests on. Every value is read from a run
summary, a checkpoint file or a report; nothing is typed in. The checkpoints are not committed
(``runs/`` and ``checkpoints/`` stay local); the hashes are what a later phase checks them by.

The choice itself was made by rule before this runs (``docs/PHASE_STATUS.md``, Phase 13): option
and shipped seed from ``reports/ablation/``, the detector from GW-DET-007's adoption test. The
script refuses to freeze if those reports disagree with the runs it is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from glypharis.data.download import sha256_file

OUT = Path("configs/final/glypharis_v1.yaml")
ABLATION = Path("reports/ablation/phase13_final.json")
DET_PAIR = Path("reports/error_cases/detector/GW-DET-007/pair.json")
DET_EVAL = Path("reports/error_cases/detector/GW-DET-007/GW-DET-001_eval.json")
CALIBRATION = Path("reports/calibration/GW-E2E-005/calibration.json")
SHIPPED_SEED = 1337  # fixed with the plan, before any final run was trained


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ci(d: dict[str, float], digits: int = 4) -> list[float]:
    return [round(d["value"], digits), round(d["low"], digits), round(d["high"], digits)]


def recognizer(run: Path, final: dict[str, Any], calib: dict[str, Any]) -> dict[str, Any]:
    s = read_json(run / "summary.json")
    cfg = yaml.safe_load((run / "config.resolved.yaml").read_text(encoding="utf-8"))
    exp = s["experiment_id"]
    ckpt = Path("checkpoints") / exp / "best.pt"
    if exp not in final["runs"] or cfg["seed"] != SHIPPED_SEED:
        raise SystemExit(f"{exp} is not the seed-{SHIPPED_SEED} run of option {final['option']}")
    if Path(calib["rec_run"]).name != exp or not calib["adopt"]:
        raise SystemExit(f"{CALIBRATION} is not an adopted fit for {exp}")
    best = s["best"]
    return {
        "experiment": exp,
        "config": f"configs/rec/{exp.lower()}.yaml",
        "config_hash": s["config_hash"],
        "git_commit": s["git"]["commit"],
        "seed": cfg["seed"],
        "checkpoint": ckpt.as_posix(),
        "checkpoint_sha256": sha256_file(ckpt),
        "model_name": cfg["model_name"],
        "parameters": s["parameters"],
        "input": {"height": cfg["data"]["image_height"], "max_width": cfg["data"]["max_width"]},
        "charset": cfg["data"]["charset"],
        "temperature": calib["temperature"],
        "temperature_from": calib["experiment_id"],
        "training": {
            "epochs": s["epochs_run"],
            "steps": s["steps"],
            "selected_epoch": s["best_epoch"],
            "selection": "lowest validation CER, strict track",
            "data": {k: v["bin_sha256"] for k, v in s["data"].items()},
            "precision": s["precision"],
            "deterministic": s["deterministic"],
            "device": s["gpu"],
            "torch": s["torch"],
            "gpu_hours": round(s["train_seconds"] / 3600, 2),
        },
        "validation": {
            "cer": round(best["val_cer"], 4),
            "wer": round(best["val_wer"], 4),
            "cer_normalised": round(best["val_cer_normalised"], 4),
        },
    }


def detector(
    run: Path, checkpoint: Path, pair: dict[str, Any], ev: dict[str, Any]
) -> dict[str, Any]:
    s = read_json(run / "summary.json")
    exp = s["experiment_id"]
    winner = pair["label_a"] if pair["adopt"] else pair["label_b"]
    if winner != exp or ev["experiment_id"] != exp:
        raise SystemExit(f"the adoption test keeps {winner}, not {exp}")
    sha = sha256_file(checkpoint)
    if sha != ev["checkpoint_sha256"]:
        raise SystemExit(f"{checkpoint} is not the checkpoint {DET_EVAL} scored")
    h = ev["headline"]
    return {
        "experiment": exp,
        "config": f"configs/det/{exp.lower()}.yaml",
        "config_hash": s["config_hash"],
        "git_commit": s["git"]["commit"],
        "checkpoint": f"checkpoints/{exp}/best.pt",
        "checkpoint_sha256": sha,
        "backbone": s["backbone"],
        "backbone_weights_sha256": s["weights_sha256"],
        "parameters": s["parameters"]["inference"],
        "label_version": s["label_version"],
        "infer": {"long_side": ev["long_side"]},
        "post": ev["post"],
        "operating_point": ev["op"],
        "settings_from": "Phase 9: GW-DET-004 post-processing, GW-DET-005 resolution",
        "training": {
            "epochs": s["epochs_run"],
            "steps": s["steps"],
            "selected_epoch": s["best_epoch"],
            "data": {
                "train_pages_sha256": s["data"]["train_pages_sha256"],
                "val_pages_sha256": s["data"]["val_pages_sha256"],
            },
            "precision": s["precision"],
            "deterministic": s["deterministic"],
            "device": s["gpu"],
            "versions": s["versions"],
        },
        "validation": {
            "pages": h["pages"],
            "f1_50": ci(h["f1_50"]),
            "precision": ci(h["precision"]),
            "recall": ci(h["recall"]),
        },
        "not_adopted": {
            "experiment": pair["label_a"],
            "f1_50_difference": ci(pair["diff"]),
            "report": DET_PAIR.as_posix(),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rec-run", type=Path, required=True)
    parser.add_argument("--det-run", type=Path, required=True)
    parser.add_argument("--det-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    final = read_json(ABLATION)
    calib = read_json(CALIBRATION)
    pair = read_json(DET_PAIR)
    record = {
        "name": "glypharis_v1",
        "frozen_by": "GW-REC-037",
        "selection": "validation data only; the test sets stay closed until Phase 14",
        "recognizer": recognizer(args.rec_run, final, calib),
        "detector": detector(args.det_run, args.det_checkpoint, pair, read_json(DET_EVAL)),
        "final_runs": {
            "option": final["option"],
            "runs": final["runs"],
            "val_cer_per_seed": [round(v, 4) for v in final["val_seeds"]],
            "val_cer_mean": round(final["val_cer"], 4),
            "baseline": final["baseline"],
            "val_cer_difference": ci(final["val_vs_baseline"]["diff"]),
            "doc_slice_cer_difference": ci(final["doc_vs_baseline"]["diff"]),
            "glyph_c_mce_per_seed": [round(v, 4) for v in final["mce_seeds"]],
            "report": ABLATION.as_posix(),
        },
        "calibration": {
            "method": "one temperature, fitted on validation half A by binary NLL",
            "ece_half_b": [round(calib["ece_b_at_1"], 4), round(calib["ece_b_at_t"], 4)],
            "report": CALIBRATION.with_suffix(".md").as_posix(),
        },
    }
    header = (
        "# GLYPHARIS v1: the frozen configuration (Phase 13).\n"
        "# Generated by scripts/freeze_final.py from run summaries, checkpoints and reports;\n"
        "# do not edit by hand. Any change to it after Phase 14 is a new model version\n"
        "# (docs/EXPERIMENT_PROTOCOL.md).\n"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(record, sort_keys=False, allow_unicode=True, width=100)
    args.out.write_text(header + body, encoding="utf-8", newline="\n")
    print(f"wrote {args.out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
