from __future__ import annotations

import hashlib
import json
from pathlib import Path

import freeze_final
import pytest
import validate_configs
import yaml


def ci(v: float) -> dict[str, float]:
    return {"value": v, "low": v - 0.001, "high": v + 0.001}


def dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def results(tmp_path, monkeypatch):
    """A worktree with the reports, run summaries and checkpoints the freeze reads."""
    monkeypatch.chdir(tmp_path)
    dump(Path(freeze_final.ABLATION), {
        "option": "B", "runs": ["GW-REC-034", "GW-REC-035", "GW-REC-036"],
        "baseline": "GW-REC-001", "val_seeds": [0.063, 0.0628, 0.063], "val_cer": 0.0629,
        "val_vs_baseline": {"diff": ci(-0.0124)}, "doc_vs_baseline": {"diff": ci(-0.0108)},
        "mce_seeds": [0.0354, 0.0353, 0.0353],
    })  # fmt: skip
    dump(Path(freeze_final.CALIBRATION), {
        "experiment_id": "GW-E2E-005", "rec_run": "runs/GW-REC-034", "adopt": True,
        "temperature": 2.2, "ece_b_at_1": 0.2, "ece_b_at_t": 0.08,
    })  # fmt: skip
    det_ckpt = tmp_path / "other" / "checkpoints" / "GW-DET-001" / "best.pt"
    det_ckpt.parent.mkdir(parents=True)
    det_ckpt.write_bytes(b"detector")
    dump(Path(freeze_final.DET_EVAL), {
        "experiment_id": "GW-DET-001",
        "checkpoint_sha256": hashlib.sha256(b"detector").hexdigest(),
        "long_side": 1536, "post": {"box_thresh": 0.5}, "op": "inverse09",
        "headline": {"pages": 1715, "f1_50": ci(0.778), "precision": ci(0.852),
                     "recall": ci(0.716)},
    })  # fmt: skip
    dump(Path(freeze_final.DET_PAIR), {
        "label_a": "GW-DET-007", "label_b": "GW-DET-001", "adopt": False, "diff": ci(-0.011),
    })  # fmt: skip
    det_run = tmp_path / "other" / "runs" / "GW-DET-001"
    dump(det_run / "summary.json", {
        "experiment_id": "GW-DET-001", "config_hash": "d" * 64, "git": {"commit": "abc"},
        "backbone": "mobilenet_v3_large", "weights_sha256": "w" * 64,
        "parameters": {"train": 3210466, "inference": 3187209}, "label_version": 1,
        "epochs_run": 24, "steps": 98544, "best_epoch": 22, "precision": "bf16",
        "deterministic": True, "gpu": "GPU", "versions": {"torch": "x"},
        "data": {"train_pages_sha256": "t" * 64, "val_pages_sha256": "v" * 64},
    })  # fmt: skip
    rec_run = Path("runs/GW-REC-034")
    dump(rec_run / "summary.json", {
        "experiment_id": "GW-REC-034", "config_hash": "r" * 64, "git": {"commit": "def"},
        "parameters": 7928835, "epochs_run": 30, "steps": 169860, "best_epoch": 29,
        "precision": "bf16", "deterministic": True, "gpu": "GPU", "torch": "x",
        "train_seconds": 72000,
        "data": {"processed/hiertext/lines_train": {"bin_sha256": "a" * 64}},
        "best": {"val_cer": 0.063, "val_wer": 0.1937, "val_cer_normalised": 0.0602},
    })  # fmt: skip
    (rec_run / "config.resolved.yaml").write_text(
        "seed: 1337\nmodel_name: svtr\n"
        "data: {image_height: 48, max_width: 1536, charset: data/manifests/charset.json}\n",
        encoding="utf-8",
    )
    Path("checkpoints/GW-REC-034").mkdir(parents=True)
    Path("checkpoints/GW-REC-034/best.pt").write_bytes(b"recognizer")
    return ["--rec-run", str(rec_run), "--det-run", str(det_run), "--det-checkpoint", str(det_ckpt)]


def test_freeze_writes_the_record_from_the_results(results):
    assert freeze_final.main(results) == 0
    text = freeze_final.OUT.read_text(encoding="utf-8")
    assert text.startswith("# GLYPHARIS v1")
    record = yaml.safe_load(text)
    rec, det = record["recognizer"], record["detector"]
    assert rec["checkpoint_sha256"] == hashlib.sha256(b"recognizer").hexdigest()
    assert rec["temperature"] == 2.2 and rec["temperature_from"] == "GW-E2E-005"
    assert rec["config"] == "configs/rec/gw-rec-034.yaml"
    assert det["checkpoint"] == "checkpoints/GW-DET-001/best.pt"  # no worktree path
    assert det["not_adopted"]["experiment"] == "GW-DET-007"
    assert record["final_runs"]["val_cer_difference"] == [-0.0124, -0.0134, -0.0114]
    assert "other" not in text


def test_freeze_refuses_what_the_rules_did_not_choose(results):
    pair = json.loads(Path(freeze_final.DET_PAIR).read_text(encoding="utf-8"))
    dump(Path(freeze_final.DET_PAIR), {**pair, "adopt": True})
    with pytest.raises(SystemExit, match="keeps GW-DET-007"):
        freeze_final.main(results)
    dump(Path(freeze_final.DET_PAIR), pair)
    cfg = Path("runs/GW-REC-034/config.resolved.yaml")
    cfg.write_text(cfg.read_text(encoding="utf-8").replace("1337", "7"), encoding="utf-8")
    with pytest.raises(SystemExit, match="seed-1337"):
        freeze_final.main(results)


def test_validate_configs_checks_freeze_records(write_yaml, tmp_path):
    write_yaml("configs/rec/gw-rec-034.yaml", "experiment_id: GW-REC-034\n")
    write_yaml("configs/det/gw-det-001.yaml", "experiment_id: GW-DET-001\n")
    good = {
        "recognizer": {"experiment": "GW-REC-034", "config": "configs/rec/gw-rec-034.yaml",
                       "checkpoint_sha256": "0" * 64},
        "detector": {"experiment": "GW-DET-001", "config": "configs/det/gw-det-001.yaml",
                     "checkpoint_sha256": "1" * 64},
    }  # fmt: skip
    final = tmp_path / "configs" / "final" / "glypharis_v1.yaml"
    final.parent.mkdir(parents=True)
    final.write_text(yaml.safe_dump(good), encoding="utf-8")
    assert validate_configs.main(["--root", str(tmp_path / "configs")]) == 0
    bad = {**good, "detector": {**good["detector"], "config": "configs/rec/gw-rec-034.yaml"}}
    final.write_text(yaml.safe_dump(bad), encoding="utf-8")
    assert validate_configs.main(["--root", str(tmp_path / "configs")]) == 1
    bad = {**good, "recognizer": {**good["recognizer"], "checkpoint_sha256": "abc"}}
    final.write_text(yaml.safe_dump(bad), encoding="utf-8")
    assert validate_configs.main(["--root", str(tmp_path / "configs")]) == 1
