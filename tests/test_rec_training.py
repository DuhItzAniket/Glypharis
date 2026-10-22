from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from glypharis.config.loader import build_config
from glypharis.config.schema import DiagnosticsConfig
from glypharis.data.crops import encode_jpeg
from glypharis.data.linepack import LinePackWriter
from glypharis.data.rec_dataset import (
    RecLineDataset,
    WidthBucketBatchSampler,
    collate,
    ctc_frames_needed,
)
from glypharis.data.text import Charset
from glypharis.diagnostics import (
    EpochPoint,
    GradientMonitor,
    check_collapse,
    check_overfitting,
    check_underfitting,
)
from glypharis.training.checkpoint import load_checkpoint
from glypharis.training.rec_trainer import RecTrainer

WORDS = ["cat", "dog", "sun", "map", "pen", "hill", "tree", "boat", "lamp", "seed", "moon", "fish"]


def _line_image(text: str) -> Image.Image:
    img = Image.new("L", (12 + 9 * len(text), 24), 255)
    ImageDraw.Draw(img).text((6, 6), text, fill=0)
    return img


def _pack(prefix, lines, split):
    with LinePackWriter(prefix, {"split": split}) as w:
        for key, doc, text in lines:
            img = _line_image(text)
            w.add(key, doc, split, "test-fixture", text, encode_jpeg(img, 95), img.size)


@pytest.fixture
def tiny(tmp_path):
    data = tmp_path / "data"
    Charset(list(" abcdefghijklmnopqrstuvwxyz")).save(data / "manifests/charset.json")
    rng = np.random.default_rng(0)
    train = [(f"tr{i}", f"doc{i // 4}", " ".join(rng.choice(WORDS, 2))) for i in range(48)]
    val = [(f"va{i}", f"vdoc{i // 4}", str(rng.choice(WORDS))) for i in range(16)]
    _pack(data / "processed/tiny/lines_train", train, "train")
    _pack(data / "processed/tiny/lines_val", val, "validation")
    return tmp_path


def _cfg(root, tag, **extra):
    d = {
        "experiment_id": "GW-REC-999",
        "task": "rec",
        "model_name": "crnn",
        "device": "cpu",
        "paths": {"data_root": str(root / "data"), "runs_dir": str(root / tag / "runs"),
                  "reports_dir": str(root / tag / "reports")},
        "data": {"train": ["processed/tiny/lines_train"], "val": ["processed/tiny/lines_val"],
                 "num_workers": 0, "augmentation": "full", "train_eval_subset": 16},
        "model": {"cnn_channels": [8, 8, 16, 16, 16, 16], "lstm_hidden": 16, "dropout": 0.1},
        "train": {"epochs": 2, "batch_size": 8, "gradient_accumulation": 2,
                  "learning_rate": 3.0e-3, "mixed_precision": "none", "scheduler": "onecycle"},
        "checkpoint": {"dir": str(root / tag / "ckpt")},
        "logging": {"log_every_n_steps": 1},
    }  # fmt: skip
    overrides = [f"{k}={v}" for k, v in extra.items()]
    return build_config(d, overrides)


def test_training_smoke_writes_metrics_checkpoints_and_summary(tiny):
    trainer = RecTrainer(_cfg(tiny, "a"))
    summary = trainer.fit()
    assert summary["epochs_run"] == 2 and summary["steps"] == 6  # 6 batches / 2 accumulated
    assert summary["parameters"] > 0 and summary["best_epoch"] in (0, 1)
    recs = [
        json.loads(x) for x in (tiny / "a/runs/GW-REC-999/metrics.jsonl").read_text().splitlines()
    ]
    epochs = [r for r in recs if r["type"] == "epoch"]
    assert len(epochs) == 2 and all(np.isfinite(r["val_loss"]) for r in epochs)
    for name in ("best.pt", "last.pt", "epoch_000.pt", "epoch_001.pt"):
        assert (tiny / "a/ckpt" / name).exists()
    ck = load_checkpoint(tiny / "a/ckpt/last.pt")
    assert ck["state"]["step"] == 6 and "python" in ck["rng"] and ck["charset"][0] == " "


def test_resume_mid_epoch_matches_an_uninterrupted_run(tiny):
    full = RecTrainer(_cfg(tiny, "full"))
    full.fit()
    # Stop at step 4: epoch 0 has 3 steps, so this is one step into epoch 1.
    cut = RecTrainer(_cfg(tiny, "cut", **{"train.max_steps": 4}))
    cut.fit()
    assert cut.state.epoch == 1 and cut.state.batch_in_epoch == 2
    last = tiny / "cut/ckpt/last.pt"
    resumed = RecTrainer(_cfg(tiny, "cut", **{"checkpoint.resume": str(last)}))
    resumed.fit()
    for (name, a), (_, b) in zip(full.model.state_dict().items(),
                                 resumed.model.state_dict().items(), strict=True):  # fmt: skip
        assert torch.equal(a, b), name
    for key in ("val_cer", "train_loss"):  # the epoch's running loss survives the resume too
        assert [h[key] for h in full.state.history] == [h[key] for h in resumed.state.history]


def test_resume_refuses_a_different_config(tiny):
    RecTrainer(_cfg(tiny, "x", **{"train.max_steps": 2})).fit()
    other = _cfg(tiny, "x", **{"checkpoint.resume": str(tiny / "x/ckpt/last.pt"),
                               "train.learning_rate": 1.0e-2})  # fmt: skip
    with pytest.raises(Exception, match="different config"):
        RecTrainer(other).fit()


def test_leakage_and_test_packs_are_refused(tiny):
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        RecTrainer(_cfg(tiny, "l", **{"data.val": "[processed/tiny/lines_train]"}))
    with pytest.raises(Exception, match="test packs"):
        RecTrainer(_cfg(tiny, "t", **{"data.val": "[processed/hiertext/lines_test]"}))


# ---------------------------------------------------------------------------- data pieces


def test_sampler_is_deterministic_and_resumes_part_way():
    widths = np.random.default_rng(1).integers(10, 400, size=203)
    s = WidthBucketBatchSampler(widths, 16, seed=3, chunk=4)
    b0 = s.batches(0)
    assert b0 == s.batches(0) and b0 != s.batches(1)
    assert sorted(i for b in b0 for i in b) == list(range(203)) and len(s) == len(b0)
    s.set_epoch(0, start_batch=5)
    assert [[i for _, i in b] for b in s] == b0[5:]
    ev = WidthBucketBatchSampler(widths, 16, seed=3, shuffle=False)
    flat = [i for b in ev.batches(0) for i in b]
    assert list(widths[flat]) == sorted(widths)


def test_collate_pads_narrow_lines_to_what_ctc_needs(tiny):
    cs = Charset.load(tiny / "data/manifests/charset.json")
    ds = RecLineDataset([tiny / "data/processed/tiny/lines_train"], cs, 32, 1024)
    item = ds[(0, 0)]
    item["image"] = item["image"][:, :4]  # far too narrow for its label
    batch = collate([item])
    assert batch["images"].shape[-1] // 4 >= ctc_frames_needed(item["target"])
    assert ctc_frames_needed([1, 1, 2]) == 4  # a repeat needs a blank in between


def test_augmentation_depends_only_on_seed_epoch_and_index(tiny):
    from glypharis.data.augment import Augmenter

    cs = Charset.load(tiny / "data/manifests/charset.json")
    ds = RecLineDataset([tiny / "data/processed/tiny/lines_train"], cs, 32, 1024,
                        Augmenter("full"), seed=9)  # fmt: skip
    assert torch.equal(ds[(3, 5)]["image"], ds[(3, 5)]["image"])
    assert any(not torch.equal(ds[(3, i)]["image"], ds[(4, i)]["image"]) for i in range(8))


# ---------------------------------------------------------------------------- diagnostics


def _points(train_loss, val_loss, train_cer, val_cer):
    return [EpochPoint(i, i * 10, *v) for i, v in
            enumerate(zip(train_loss, val_loss, train_cer, val_cer, strict=True))]  # fmt: skip


def test_overfitting_and_underfitting_rules():
    d = DiagnosticsConfig()
    over = _points([5, 4, 3, 2, 1], [2, 2.1, 2.3, 2.6, 3.0], [0.3, 0.2, 0.1, 0.05, 0.02],
                   [0.2, 0.2, 0.21, 0.22, 0.25])  # fmt: skip
    assert check_overfitting(over, d).name == "POSSIBLE OVERFITTING"
    healthy = _points([5, 4, 3, 2, 1], [5, 4, 3, 2, 1], [0.5, 0.4, 0.3, 0.2, 0.1],
                      [0.5, 0.4, 0.3, 0.2, 0.1])  # fmt: skip
    assert check_overfitting(healthy, d) is None
    stuck = _points([3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [0.9] * 5, [0.9] * 5)
    assert check_underfitting(stuck, d, 0.1).name == "POSSIBLE UNDERFITTING"
    assert check_underfitting(healthy, d, 0.5) is None


def test_collapse_and_gradient_rules():
    d = DiagnosticsConfig()
    names = [f.name for f in check_collapse(0.99, 0.6, 0.7, d, 0, 0)]
    assert names == ["BLANK COLLAPSE", "PREDICTION COLLAPSE"]
    assert check_collapse(0.5, 0.0, 0.01, d, 0, 0) == []
    mon = GradientMonitor(d)
    assert mon.update(1e4, 0, 1).name == "GRADIENT EXPLOSION"
    assert mon.update(1e4, 0, 2) is None  # raised once
    flags = [mon.update(1e-9, 0, s) for s in range(100)]
    assert flags[-1] is not None and flags[-1].name == "GRADIENT VANISHING"


# ------------------------------------------------------------------------- entry points


def _yaml(cfg, path):
    import yaml

    from glypharis.config.loader import config_to_dict

    path.write_text(yaml.safe_dump(config_to_dict(cfg)), encoding="utf-8")
    return path


def test_train_entry_point_writes_a_run_report(tiny):
    import train

    cfg_path = _yaml(_cfg(tiny, "e"), tiny / "run.yaml")
    assert train.main(["--config", str(cfg_path)]) == 0
    report = (tiny / "e/reports/training/GW-REC-999/report.md").read_text(encoding="utf-8")
    assert "Best and final epoch" in report and "95% bootstrap CI" in report
    assert (tiny / "e/reports/training/GW-REC-999/curves.png").exists()


def test_sanity_sequence_runs_end_to_end(tiny, monkeypatch):
    import rec_sanity

    monkeypatch.setattr(rec_sanity, "OVERFIT_MAX_STEPS", 8)
    monkeypatch.setattr(rec_sanity, "LR_STEPS", 6)
    cfg_path = _yaml(_cfg(tiny, "s"), tiny / "sanity.yaml")
    rec_sanity.main(["--config", str(cfg_path), "--dry-run-minutes", "0.005"])
    out = tiny / "s/reports/training/sanity/GW-REC-999"
    res = json.loads((out / "sanity.json").read_text(encoding="utf-8"))
    assert {"overfit", "lr_range", "dry_run"} <= set(res)
    assert res["dry_run"]["finite"] and res["lr_range"]["steps"] > 0
    assert (out / "lr_range.png").exists() and "Overfit" in (out / "sanity.md").read_text("utf-8")


# ---------------------------------------------------------------- batch independence, caps


def test_a_lines_output_does_not_depend_on_the_rest_of_its_batch():
    from glypharis.config.schema import ModelConfig
    from glypharis.models.rec.crnn import CRNN

    torch.manual_seed(0)
    m = CRNN(30, ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=16)).eval()
    short = torch.randn(1, 1, 32, 160)
    alone = m(short, torch.tensor([160]))[:, 0]
    batch = torch.zeros(2, 1, 32, 640)
    batch[0, :, :, :160] = short[0]
    batch[1] = torch.randn(1, 32, 640)
    together = m(batch, torch.tensor([160, 640]))[: alone.shape[0], 0]
    # Away from the right edge (the CNN sees a few padded columns there) the frames must match:
    # the BiLSTM reads each line only up to its own end.
    assert torch.allclose(alone[:25], together[:25], atol=1e-4)


def test_column_cap_bounds_every_batch():
    widths = np.random.default_rng(2).integers(20, 1000, size=500)
    s = WidthBucketBatchSampler(widths, 32, seed=1, chunk=4, max_columns=4096)
    for b in s.batches(0):
        assert (len(b) <= 32 and len(b) * max(widths[b]) <= 4096) or len(b) == 1
    assert sorted(i for b in s.batches(0) for i in b) == list(range(500))
    assert s.num_batches(0) == len(s.batches(0)) and s.num_batches(0) > len(widths) // 32


def test_resume_matches_with_a_column_cap(tiny):
    cap = {"data.max_batch_columns": 1024}
    full = RecTrainer(_cfg(tiny, "cf", **cap))
    full.fit()
    cut = RecTrainer(_cfg(tiny, "cc", **cap, **{"train.max_steps": 5}))
    cut.fit()
    resumed = RecTrainer(_cfg(tiny, "cc", **cap,
                              **{"checkpoint.resume": str(tiny / "cc/ckpt/last.pt")}))  # fmt: skip
    resumed.fit()
    assert full.state.step == resumed.state.step
    for a, b in zip(full.model.state_dict().values(), resumed.model.state_dict().values(),
                    strict=True):  # fmt: skip
        assert torch.equal(a, b)


def test_resume_check_script_passes_on_cpu(tiny, monkeypatch):
    import rec_resume_check

    monkeypatch.setattr(rec_resume_check, "STEPS", 8)
    cfg_path = _yaml(_cfg(tiny, "r", **{"train.epochs": 3}), tiny / "resume.yaml")
    assert rec_resume_check.main(["--config", str(cfg_path)]) == 0
    res = json.loads((tiny / "r/reports/training/sanity/GW-REC-999/resume_check.json").read_text())
    assert res["passed"] and res["max_rel_loss_diff"] == 0 and res["max_rel_weight_diff"] == 0
