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
    # GW-REC-001's pattern: val loss creeps up a little while val CER keeps setting new bests.
    confident = _points(
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [0.556, 0.558, 0.559, 0.561, 0.562],
        [0.06, 0.05, 0.045, 0.04, 0.036],
        [0.080, 0.079, 0.078, 0.076, 0.075],
    )
    assert check_overfitting(confident, d) is None
    # Same loss pattern, but a real divergence: val loss up 10% and val CER stuck above its best.
    diverging = _points(
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [0.50, 0.52, 0.53, 0.54, 0.55],
        [0.06, 0.05, 0.045, 0.04, 0.036],
        [0.075, 0.077, 0.078, 0.078, 0.079],
    )
    assert check_overfitting(diverging, d).name == "POSSIBLE OVERFITTING"
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


def test_stacked_lstm_checkpoints_load_and_compute_the_same():
    from torch import nn

    from glypharis.config.schema import ModelConfig
    from glypharis.models.rec.crnn import CRNN

    torch.manual_seed(0)
    cfg = ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=6, dropout=0.3)
    stacked = nn.LSTM(16, 6, num_layers=2, bidirectional=True, dropout=0.3).eval()
    # The layout GW-REC-001/002 were saved with: one stacked nn.LSTM under "rnn.".
    old = {k: v for k, v in CRNN(30, cfg).state_dict().items() if not k.startswith("rnn.")}
    old |= {f"rnn.{k}": v for k, v in stacked.state_dict().items()}
    model = CRNN(30, cfg).eval()
    model.load_state_dict(old)
    # Same parameter order, so saved optimizer states still line up.
    expected = [*model.cnn.parameters(), *stacked.parameters(), *model.head.parameters()]
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), expected, strict=True))
    x, widths = torch.randn(3, 1, 32, 96), torch.tensor([96, 60, 33])
    with torch.no_grad():
        feats = model.cnn(x).mean(dim=2).permute(2, 0, 1)
        lengths = CRNN.frames(widths).clamp(min=1, max=feats.shape[0])
        packed = nn.utils.rnn.pack_padded_sequence(feats, lengths, enforce_sorted=False)
        ref, _ = nn.utils.rnn.pad_packed_sequence(stacked(packed)[0], total_length=len(feats))
        assert torch.allclose(model(x, widths), model.head(ref), atol=1e-6)


@pytest.mark.gpu
def test_lstm_dropout_on_the_gpu_follows_the_saved_rng_state():
    from glypharis.config.schema import ModelConfig
    from glypharis.models.rec.crnn import CRNN

    m = CRNN(30, ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=16)).cuda().train()
    x, widths = torch.randn(2, 1, 32, 96, device="cuda"), torch.tensor([96, 50])
    state = torch.cuda.get_rng_state()
    first = m(x, widths)
    torch.cuda.set_rng_state(state)
    # cuDNN's built-in LSTM dropout kept its own random state, so this used to differ.
    assert torch.equal(first, m(x, widths))


@pytest.mark.gpu
def test_a_gpu_training_process_exits_cleanly(repo_root):
    import subprocess
    import sys

    code = (
        "import torch\n"
        "from glypharis.config.schema import ModelConfig\n"
        "from glypharis.models.rec.crnn import CRNN\n"
        "m = CRNN(30, ModelConfig()).cuda().train()\n"
        "m(torch.randn(2, 1, 32, 96, device='cuda'), torch.tensor([96, 50])).sum().backward()\n"
    )
    # A stacked LSTM with dropout made this exit with 0xC0000409 on Windows.
    done = subprocess.run([sys.executable, "-c", code], cwd=repo_root, capture_output=True)
    assert done.returncode == 0, done.stderr.decode(errors="replace")[-2000:]


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
    assert res["passed"] and res["cut_mid_epoch"] and res["lr_identical"]
    assert res["boundary_rel_diff"] == 0 and res["max_rel_loss_diff_resumed"] == 0
    assert res["max_rel_weight_diff_resumed"] == 0
    assert res["max_rel_loss_diff_two_uninterrupted"] == 0  # the CPU is deterministic


def test_best_checkpoint_is_the_lowest_metric_and_min_delta_only_drives_patience():
    from glypharis.config.schema import EarlyStoppingConfig
    from glypharis.training.rec_trainer import TrainState, record_validation

    es = EarlyStoppingConfig(enabled=True, patience=3, min_delta=0.001)
    st = TrainState()
    # Every epoch is a little better, but only epoch 3 by more than min_delta. GW-REC-002 kept an
    # older best.pt this way when one rule did both jobs.
    vals = [0.100, 0.0996, 0.0993, 0.0985, 0.0984, 0.0983, 0.0982]
    assert [record_validation(st, v, e, es) for e, v in enumerate(vals)] == [True] * 7
    assert st.best_epoch == 6 and st.best_metric == 0.0982
    assert st.patience_ref == 0.0985 and st.bad_epochs == 3 and st.stopped_early
    assert not record_validation(st, 0.0982, 7, es) and st.best_epoch == 6  # a tie keeps the first


def test_checkpoints_from_before_the_patience_reference_still_resume():
    from dataclasses import asdict

    from glypharis.training.rec_trainer import TrainState, state_from_checkpoint

    saved = asdict(TrainState(best_metric=0.5, best_epoch=3, bad_epochs=2))
    del saved["patience_ref"]
    st = state_from_checkpoint(saved)
    assert st.patience_ref == 0.5 and st.best_epoch == 3 and st.bad_epochs == 2


def test_reselect_moves_best_to_the_lowest_epoch_and_records_it(tiny):
    import shutil

    import rec_reselect

    import train

    cfg_path = _yaml(_cfg(tiny, "x", **{"train.epochs": 3}), tiny / "x.yaml")
    assert train.main(["--config", str(cfg_path)]) == 0
    run, ckpt = tiny / "x/runs/GW-REC-999", tiny / "x/ckpt"
    epochs = rec_reselect.epoch_records(run / "metrics.jsonl")
    low = min(epochs.values(), key=lambda r: (r["val_cer"], r["epoch"]))["epoch"]
    wrong = next(e for e in epochs if e != low)
    # Put the run in the state the old selection rule could leave it in.
    shutil.copy2(ckpt / f"epoch_{wrong:03d}.pt", ckpt / "best.pt")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary.update(best_epoch=wrong, best=epochs[wrong])
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert rec_reselect.main(["--config", str(cfg_path)]) == 0
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["best_epoch"] == low and summary["reselection"]["from_epoch"] == wrong
    assert summary["reselection"]["rescored_val_cer"] == epochs[low]["val_cer"]  # exact on CPU
    best = load_checkpoint(ckpt / "best.pt")
    assert best["state"]["best_epoch"] == low and best["state"]["epoch"] == low + 1
    assert (ckpt / f"best_superseded_e{wrong:03d}.pt").exists()
    assert (run / f"val_best_samples.superseded_e{wrong:03d}.jsonl").exists()
    report = (tiny / "x/reports/training/GW-REC-999/report.md").read_text(encoding="utf-8")
    assert "## Notes" in report and "re-selected" in report
    assert rec_reselect.main(["--config", str(cfg_path)]) == 0  # a second run changes nothing
    assert json.loads((run / "summary.json").read_text())["notes"] == summary["notes"]


@pytest.mark.gpu
def test_rng_state_restores_from_a_checkpoint_loaded_onto_the_gpu(tmp_path):
    from glypharis.training.checkpoint import rng_state, save_checkpoint, set_rng_state

    save_checkpoint(tmp_path / "c.pt", {"rng": rng_state()})
    expected = torch.rand(3)
    set_rng_state(load_checkpoint(tmp_path / "c.pt", "cuda")["rng"])  # tensors land on the GPU
    assert torch.equal(torch.rand(3), expected)


def _poison_third_backward(trainer):
    """Make the third backward pass produce a NaN gradient while its loss stays finite."""
    calls = {"n": 0}

    def hook(grad):
        calls["n"] += 1
        return grad * float("nan") if calls["n"] == 3 else grad

    trainer.model.head.weight.register_hook(hook)


def test_a_nan_gradient_with_a_finite_loss_stops_before_the_weights_change(tiny):
    from glypharis.diagnostics import NonFiniteLoss

    trainer = RecTrainer(_cfg(tiny, "nan"))
    _poison_third_backward(trainer)  # batch 3 = the first of optimizer step 2 (accumulation 2)
    with pytest.raises(NonFiniteLoss, match="grad_norm"):
        trainer.fit()
    recs = [
        json.loads(x) for x in (tiny / "nan/runs/GW-REC-999/metrics.jsonl").read_text().splitlines()
    ]
    bad = [r for r in recs if r["type"] == "nonfinite"]
    assert len(bad) == 1 and bad[0]["step"] == 1 and "grad_norm" in bad[0] and bad[0]["keys"]
    ck = load_checkpoint(tiny / "nan/ckpt/nonfinite.pt")
    assert all(torch.isfinite(v).all() for v in ck["model"].values() if v.is_floating_point())


def test_a_nan_gradient_is_skipped_when_the_run_may_continue(tiny):
    trainer = RecTrainer(_cfg(tiny, "skip", **{"diagnostics.stop_on_nan": False}))
    _poison_third_backward(trainer)
    trainer.fit()
    recs = [
        json.loads(x)
        for x in (tiny / "skip/runs/GW-REC-999/metrics.jsonl").read_text().splitlines()
    ]
    assert sum(r["type"] == "nonfinite_grad_skipped" for r in recs) == 1
    assert trainer.state.step == 6  # the skipped step still counts, so the schedule stays aligned
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())


def test_a_resumed_run_reports_the_config_hash_it_started_with(tiny):
    from glypharis.config.loader import config_hash
    from glypharis.training.rec_trainer import run_config_hash

    plain = _cfg(tiny, "h")
    resumed = _cfg(tiny, "h", **{"checkpoint.resume": str(tiny / "h/ckpt/last.pt")})
    assert config_hash(plain) != config_hash(resumed)
    assert run_config_hash(resumed) == run_config_hash(plain) == config_hash(plain)


def test_rec_report_restores_the_run_config_hash_and_notes_it(tiny):
    import rec_report

    import train

    cfg_path = _yaml(_cfg(tiny, "rp"), tiny / "rp.yaml")
    assert train.main(["--config", str(cfg_path)]) == 0
    path = tiny / "rp/runs/GW-REC-999/summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    right = summary["config_hash"]
    path.write_text(json.dumps({**summary, "config_hash": "e047cc28" * 8}), encoding="utf-8")
    assert rec_report.main(["--config", str(cfg_path)]) == 0
    fixed = json.loads(path.read_text(encoding="utf-8"))
    assert fixed["config_hash"] == right and fixed["config_hash_as_written"] == "e047cc28" * 8
    report = (tiny / "rp/reports/training/GW-REC-999/report.md").read_text(encoding="utf-8")
    assert "Config hash corrected" in report and right[:16] in report
    other = _yaml(_cfg(tiny, "rp", **{"train.learning_rate": 1e-4}), tiny / "other.yaml")
    with pytest.raises(SystemExit, match="not the config"):
        rec_report.main(["--config", str(other)])


def test_rec_compare_pairs_lines_and_matches_the_logged_cer(tiny):
    import rec_compare

    import train

    for tag, seed in (("ca", 1), ("cb", 2)):
        cfg_path = _yaml(_cfg(tiny, tag, seed=seed), tiny / f"{tag}.yaml")
        assert train.main(["--config", str(cfg_path)]) == 0
    a, b = tiny / "ca/runs/GW-REC-999", tiny / "cb/runs/GW-REC-999"
    assert rec_compare.main([str(a), str(b), "--out", str(tiny / "cmp")]) == 0
    res = json.loads((tiny / "cmp/GW-REC-999_vs_GW-REC-999.json").read_text(encoding="utf-8"))
    assert res["lines"] == 16 and res["track"] == "strict"
    d = res["cer"]["difference"]
    assert d["low"] <= d["value"] <= d["high"]
    assert sum(res["lines_fewer_char_errors"].values()) == 16
    assert res["first"]["best_epoch"] in (0, 1) and res["second"]["experiment_id"] == "GW-REC-999"
    # A run validated on other lines is refused.
    path = b / "val_best_samples.jsonl"
    path.write_text("\n".join(path.read_text(encoding="utf-8").splitlines()[1:]), encoding="utf-8")
    with pytest.raises(SystemExit, match="different lines"):
        rec_compare.main([str(a), str(b), "--out", str(tiny / "cmp")])
