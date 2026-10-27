from __future__ import annotations

import copy
import dataclasses
import itertools
import json
import math
import multiprocessing
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from det_fixtures import (
    make_synthetic_pages,
    tiny_det_config,
    write_processed_manifest,
)
from torch import nn

from glypharis.config.loader import config_to_dict
from glypharis.config.schema import ConfigError, DetLossConfig, DetModelConfig, DetOperatingPoint
from glypharis.data.det_dataset import DetEvalDataset, collate_eval
from glypharis.data.det_geometry import decode_page
from glypharis.data.pages import read_pages, write_pages
from glypharis.models.det.dbnet import DetLogits, build_detector
from glypharis.models.det.detector import TextLineDetector, export_detector
from glypharis.models.det.loss import DBLoss
from glypharis.training import det_trainer, loop
from glypharis.training.checkpoint import load_checkpoint, set_rng_state
from glypharis.training.det_eval import evaluate_detector
from glypharis.training.det_report import write_det_report
from glypharis.training.det_trainer import DetTrainer, gpu_headroom_gib, run_det
from glypharis.training.loop import free_cuda, read_jsonl
from glypharis.utils.device import resolve_amp
from glypharis.utils.seed import seed_everything

ID = "GW-DET-999"


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic_pages(tmp_path_factory.mktemp("detsynth"))


def _yaml(cfg, path: Path) -> Path:
    path.write_text(yaml.safe_dump(config_to_dict(cfg)), encoding="utf-8")
    return path


def _records(root: Path, tag: str) -> list[dict]:
    return read_jsonl(root / tag / "runs" / ID / "metrics.jsonl")


@pytest.fixture(scope="module")
def smoke(synth, tmp_path_factory):
    """One full run through train.main: fit, full validation, summary and report."""
    import train

    root = tmp_path_factory.mktemp("smoke")
    cfg_path = _yaml(tiny_det_config(synth, root, "a"), root / "run.yaml")
    assert train.main(["--config", str(cfg_path)]) == 0
    return root, cfg_path


# ------------------------------------------------------------------------------- fit, report


def test_fit_writes_epoch_records_checkpoints_and_summary(smoke):
    root, _ = smoke
    epochs = [r for r in _records(root, "a") if r["type"] == "epoch"]
    assert len(epochs) == 2
    assert all(0 <= r["val_f1_50"] <= 1 and r["val_op"] in ("dbnet15", "inverse10") for r in epochs)
    steps = [r for r in _records(root, "a") if r["type"] == "step"]
    assert [r["step"] for r in steps] == list(range(1, 13))  # 6 batches of 2 per epoch
    assert {"bce", "dice", "l1", "n_pos", "empty", "crops_per_s"} <= set(steps[0])
    for name in ("best.pt", "last.pt", "epoch_000.pt", "epoch_001.pt"):
        assert (root / "a/ckpt" / name).exists()
    summary = json.loads((root / "a/runs" / ID / "summary.json").read_text(encoding="utf-8"))
    params = summary["parameters"]
    assert 0 < params["inference"] < params["train"]
    assert summary["exit_code"] == 0 and summary["full_eval"] == {"best": "done", "last": "done"}
    assert summary["label_ceiling_tripwire"]["pages"] == 4


def test_train_main_writes_the_report_with_cis_slices_and_overlays(smoke):
    root, _ = smoke
    out = root / "a/reports/training" / ID
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "95% page-bootstrap CIs" in report and "## Slices" in report
    assert "Run failed" not in report and "Baseline-comparable row" in report
    assert (out / "curves.png").exists()
    for name in ("small_text", "rotated", "dense", "multi_line", "noisy_background", "worst_pages"):
        assert (out / "overlays" / f"{name}.jpg").exists(), name


def test_every_consumer_reads_the_stored_operating_point(smoke, tmp_path):
    import det_compare

    root, _ = smoke
    run, best = root / "a/runs" / ID, root / "a/ckpt/best.pt"
    stored = json.loads((run / "eval_full_best.json").read_text(encoding="utf-8"))["stored_op"]
    assert load_checkpoint(best)["selected_op"] == stored
    assert TextLineDetector.from_checkpoint(best).spec.operating_point == stored
    release = torch.load(export_detector(best, tmp_path / "rel.pt"), weights_only=True)
    assert release["spec"]["operating_point"] == stored
    row = det_compare.compare(run, run)["rows"][0]
    assert row["comparison"] == "stored" and row["op_a"] == stored and row["diff"]["value"] == 0
    report = (root / "a/reports/training" / ID / "report.md").read_text(encoding="utf-8")
    assert (
        re.search(
            r"\*\*Headline:\*\* `best.pt` at its stored operating point `(\w+)`", report
        ).group(1)
        == stored
    )


def test_detector_api_boxes_equal_the_trainers_evaluation_boxes(smoke, synth):
    root, cfg_path = smoke
    from glypharis.config.loader import load_config

    # A box threshold of 0 keeps the untrained model's (low-score) components as boxes.
    cfg = load_config(cfg_path, ["det.post.box_thresh=0.0"])
    trainer = DetTrainer(cfg)
    try:
        full = trainer.evaluate_full(root / "a/ckpt/best.pt", "best")
    finally:
        trainer.close()
    detector = TextLineDetector.from_checkpoint(root / "a/ckpt/best.pt")
    spec = dataclasses.replace(detector.spec, post=cfg.det.post, operating_point=full.stored_op)
    detector = TextLineDetector(detector.backend, spec)
    assert full.result.boxes is not None
    n_boxes = 0
    for page in synth.val:
        rgb = decode_page(synth.data_root / page.image, (page.width, page.height))
        got = [b.polygon for b in detector.detect(rgb)]
        want = [b.polygon for b in full.result.boxes[full.stored_op][page.doc_id]]
        assert got == want
        n_boxes += len(got)
    assert n_boxes > 0


# ------------------------------------------------------------------------------------ resume


def test_mid_epoch_resume_is_bit_identical_and_a_changed_config_is_refused(synth, tmp_path):
    full = DetTrainer(tiny_det_config(synth, tmp_path, "full"))
    full.fit()
    full.close()
    cut = DetTrainer(tiny_det_config(synth, tmp_path, "cut", ["train.max_steps=8"]))
    cut.fit()
    cut.close()
    assert cut.state.epoch == 1 and cut.state.batch_in_epoch == 2  # 6 batches per epoch
    last = tmp_path / "cut/ckpt/last.pt"
    resumed = DetTrainer(tiny_det_config(synth, tmp_path, "cut", [f"checkpoint.resume={last}"]))
    resumed.fit()
    resumed.close()
    for (name, a), (_, b) in zip(
        full.model.state_dict().items(), resumed.model.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    keys = ("val_f1_50", "val_op", "val_loss", "train_loss", "train_eval_loss", "empty_crop_share")
    for key in keys:  # the epoch's running sums survive the resume too
        assert [h[key] for h in full.state.history] == [h[key] for h in resumed.state.history], key
    steps = {r["step"]: r["step_loss"] for r in _records(tmp_path, "full") if r["type"] == "step"}
    cut_steps = {
        r["step"]: r["step_loss"] for r in _records(tmp_path, "cut") if r["type"] == "step"
    }
    assert steps == cut_steps

    other = tiny_det_config(
        synth, tmp_path, "cut", [f"checkpoint.resume={last}", "train.learning_rate=0.01"]
    )
    refused = DetTrainer(other)
    with pytest.raises(ConfigError, match="different config"):
        refused.fit()
    refused.close()


# ------------------------------------------------------------------------- validation pass


class InkModel(nn.Module):
    """A stand-in detector: dark pixels are text. Real components, without training."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (1.0 - images.float().mean(1, keepdim=True) / 255.0).clamp(1e-4, 1 - 1e-4)

    def forward_train(self, images: torch.Tensor) -> DetLogits:
        p = self.forward(images)
        return DetLogits(torch.log(p / (1 - p)), torch.zeros_like(p))


def test_evaluate_detector_does_not_depend_on_page_order_or_pool(synth, tmp_path):
    trainer = DetTrainer(tiny_det_config(synth, tmp_path, "e"))
    try:
        # The stand-in's components are whole lines, not shrunk kernels: a near-zero unclip
        # ratio turns them into matching boxes.
        ops = [*trainer.det.post.operating_points, DetOperatingPoint("tight", "dbnet", 0.05)]
        det = dataclasses.replace(
            trainer.det, post=dataclasses.replace(trainer.det.post, operating_points=ops)
        )
        gt = trainer.gt_pages
        idx = trainer.val_index
        forward = DetEvalDataset([("val", idx)], synth.data_root, det, with_labels=True)
        reverse = DetEvalDataset(
            [("val", idx.subset(np.arange(len(idx))[::-1]))], synth.data_root, det, with_labels=True
        )
        loss = DBLoss(det.loss)
        a = evaluate_detector(
            InkModel(),
            [collate_eval([forward[i]]) for i in range(len(forward))],
            gt,
            det,
            trainer.device,
            loss,
        )["val"]
        with ThreadPoolExecutor(3) as pool:
            b = evaluate_detector(
                InkModel(),
                [collate_eval([reverse[i]]) for i in range(len(reverse))],
                gt,
                det,
                trainer.device,
                loss,
                pool=pool,
            )["val"]
    finally:
        trainer.close()
    rec_a = {r.doc_id: r.to_dict() for r in a.records}
    rec_b = {r.doc_id: r.to_dict() for r in b.records}
    assert rec_a == rec_b
    assert [r.doc_id for r in b.records] == [r.doc_id for r in a.records][::-1]
    for r in a.records:
        for counts in r.det.values():
            tp, _, fn = counts[0.5]
            assert tp + fn == r.n_care
    assert sum(r.det[a.selected_op][0.5][0] for r in a.records) > 0  # the stand-in finds lines
    assert a.loss is not None and a.maps is not None and a.maps.gt_kernel_px > 0


# ------------------------------------------------------------------- Phase 6 config semantics


def test_deterministic_setting_reaches_seed_everything(synth, tmp_path, monkeypatch):
    calls = []
    real = loop.seed_everything
    monkeypatch.setattr(
        loop, "seed_everything", lambda s, d=True: (calls.append((s, d)), real(s, d))
    )
    DetTrainer(tiny_det_config(synth, tmp_path, "d", ["train.deterministic=false"])).close()
    assert calls == [(1337, False)]
    real(1337, True)


def test_skipped_nan_batch_and_checkpoint_interval(synth, tmp_path, monkeypatch):
    real = DetTrainer.train_step
    seen = []

    def first_nan(self, batch):
        out = real(self, batch)
        seen.append(1)
        if len(seen) == 1:
            out.loss = out.loss * float("nan")
        return out

    monkeypatch.setattr(DetTrainer, "train_step", first_nan)
    cfg = tiny_det_config(
        synth, tmp_path, "n", ["diagnostics.stop_on_nan=false", "checkpoint.checkpoint_interval=2"]
    )
    trainer = DetTrainer(cfg)
    trainer.fit()
    trainer.close()
    recs = _records(tmp_path, "n")
    assert [r["batch_in_epoch"] for r in recs if r["type"] == "nonfinite_skipped"] == [1]
    assert trainer.state.step == 11 and len([r for r in recs if r["type"] == "epoch"]) == 2
    assert sorted(p.name for p in (tmp_path / "n/ckpt").glob("epoch_*.pt")) == ["epoch_001.pt"]


# ---------------------------------------------------------------------------- failure paths


def _report(root: Path, tag: str) -> str:
    return (root / tag / "reports/training" / ID / "report.md").read_text(encoding="utf-8")


def test_oom_in_the_first_step_still_writes_a_report(synth, tmp_path, monkeypatch):
    def oom(self, batch):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")

    monkeypatch.setattr(DetTrainer, "train_step", oom)
    assert run_det(tiny_det_config(synth, tmp_path, "o")) == 3
    rec = [r for r in _records(tmp_path, "o") if r["type"] == "oom"]
    assert [r["phase"] for r in rec] == ["forward"] and rec[0]["batch_shape"] == [2, 64, 64, 3]
    assert (tmp_path / "o/ckpt/oom.pt").exists() and not (tmp_path / "o/ckpt/best.pt").exists()
    report = _report(tmp_path, "o")
    assert "Run failed: exit code 3" in report and "not available" in report
    summary = json.loads((tmp_path / "o/runs" / ID / "summary.json").read_text(encoding="utf-8"))
    assert summary["exit_code"] == 3 and summary["status"].startswith("failed: out of GPU memory")


def test_oom_in_validation_is_recorded_with_its_phase(synth, tmp_path, monkeypatch):
    def oom(self):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")

    monkeypatch.setattr(DetTrainer, "validate", oom)
    assert run_det(tiny_det_config(synth, tmp_path, "v")) == 3
    assert [r["phase"] for r in _records(tmp_path, "v") if r["type"] == "oom"] == ["validate"]
    assert "Run failed: exit code 3" in _report(tmp_path, "v")


def test_nan_loss_stops_the_run_with_exit_2_and_a_report(synth, tmp_path, monkeypatch):
    real = DetTrainer.train_step

    def nan(self, batch):
        out = real(self, batch)
        out.loss = out.loss * float("nan")
        return out

    monkeypatch.setattr(DetTrainer, "train_step", nan)
    assert run_det(tiny_det_config(synth, tmp_path, "x")) == 2
    assert (tmp_path / "x/ckpt/nonfinite.pt").exists()
    assert [r["type"] for r in _records(tmp_path, "x")].count("nonfinite") == 1
    assert "Run failed: exit code 2" in _report(tmp_path, "x")


def _nan_gradient_first(monkeypatch) -> list[list[str]]:
    """Make the first batch keep a finite loss but back-propagate a NaN gradient; returns the
    keys of every batch seen."""
    real = DetTrainer.train_step
    seen: list[list[str]] = []

    def step(self, batch):
        out = real(self, batch)
        seen.append(out.keys)
        if len(seen) == 1:
            # sqrt has an infinite slope at 0: the loss gains exactly 0, the gradient 0 * inf.
            p = next(self.model.parameters())
            out.loss = out.loss + 0.0 * torch.sqrt(p.sum() * 0.0)
        return out

    monkeypatch.setattr(DetTrainer, "train_step", step)
    return seen


def test_nan_gradient_with_a_finite_loss_stops_before_the_weights_change(
    synth, tmp_path, monkeypatch
):
    seen = _nan_gradient_first(monkeypatch)
    assert run_det(tiny_det_config(synth, tmp_path, "g")) == 2
    (rec,) = [r for r in _records(tmp_path, "g") if r["type"] == "nonfinite"]
    assert rec["grad_norm"] == "nan" and rec["step"] == 0 and rec["keys"] == seen[0]
    assert len(seen) == 1  # stopped at the culprit, not at the batch after it
    state = load_checkpoint(tmp_path / "g/ckpt/nonfinite.pt")["model"]
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    assert "Run failed: exit code 2 (non-finite loss or gradient)" in _report(tmp_path, "g")


@pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)`")
def test_nan_gradient_step_is_skipped_without_stop_on_nan(synth, tmp_path, monkeypatch):
    seen = _nan_gradient_first(monkeypatch)
    trainer = DetTrainer(tiny_det_config(synth, tmp_path, "k", ["diagnostics.stop_on_nan=false"]))
    try:
        trainer.fit()
    finally:
        trainer.close()
    recs = _records(tmp_path, "k")
    (skip,) = [r for r in recs if r["type"] == "nonfinite_grad_skipped"]
    assert skip["grad_norm"] == "nan" and skip["step"] == 0 and skip["keys"] == seen[0]
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    # The skipped step still advances the schedule, like a step the fp16 scaler skips.
    assert trainer.state.step == 12
    assert [r["step"] for r in recs if r["type"] == "step"] == list(range(2, 13))
    assert all(math.isfinite(r["grad_norm"]) for r in recs if r["type"] == "step")


def test_a_fresh_failed_attempt_hides_nothing_from_an_earlier_attempt(
    smoke, synth, tmp_path, monkeypatch
):
    # An earlier attempt under the same run ID left its log and full-validation files behind.
    earlier = smoke[0] / "a/runs" / ID
    run = tmp_path / "f/runs" / ID
    shutil.copytree(earlier, run)
    old_log = (run / "metrics.jsonl").read_text(encoding="utf-8")
    assert (run / "eval_full_best.json").exists()

    def oom(self, batch):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")

    monkeypatch.setattr(DetTrainer, "train_step", oom)
    assert run_det(tiny_det_config(synth, tmp_path, "f")) == 3
    assert not [name for name in det_trainer.FULL_EVAL_FILES if (run / name).exists()]
    assert (run / "metrics.attempt1.jsonl").read_text(encoding="utf-8") == old_log
    assert not [r for r in _records(tmp_path, "f") if r["type"] in ("step", "epoch")]
    report = _report(tmp_path, "f")
    assert "Run failed: exit code 3" in report and "**Headline:**" not in report


def test_report_shows_full_validation_only_when_this_finish_did_it(smoke, synth, tmp_path):
    run = tmp_path / "r/runs" / ID
    shutil.copytree(smoke[0] / "a/runs" / ID, run)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["full_eval"] = {"best": "skipped: out of GPU memory", "last": "done"}
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    cfg = tiny_det_config(synth, tmp_path, "r")
    report = write_det_report(run, tmp_path / "r/report", cfg).read_text(encoding="utf-8")
    assert "**Headline:**" not in report
    assert "`best.pt`: not available (skipped: out of GPU memory)" in report
    assert "`last.pt` on all" in report


# ------------------------------------------------------------------------- preflight, cleanup


def test_gpu_headroom_counts_cached_but_unallocated_memory(monkeypatch):
    gib = 2**30
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d=None: (1 * gib, 6 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d=None: 3 * gib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d=None: 1 * gib)
    assert gpu_headroom_gib(torch.device("cuda")) == pytest.approx(3.0)


def test_busy_gpu_refuses_with_exit_4_and_writes_nothing(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "select_device", lambda request: torch.device("cuda"))
    monkeypatch.setattr(det_trainer, "gpu_headroom_gib", lambda device: 1.0)
    assert run_det(tiny_det_config(synth, tmp_path, "g")) == 4
    assert not (tmp_path / "g").exists()


def test_close_shuts_every_worker_and_is_idempotent(synth, tmp_path):
    cfg = tiny_det_config(
        synth, tmp_path, "w", ["det.data.num_workers=1", "det.data.eval_workers=1"]
    )
    trainer = DetTrainer(cfg)
    try:
        next(iter(trainer.train_batches(0, 0)))
        trainer.validate()
        assert multiprocessing.active_children()
    finally:
        trainer.close()
    deadline = time.monotonic() + 10
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert multiprocessing.active_children() == []
    trainer.close()  # a second close is a no-op


# ------------------------------------------------------------------------------------ guards


def test_leakage_and_test_split_pages_are_refused(tmp_path):
    synth = make_synthetic_pages(tmp_path, n_train=4, n_val=2)
    root = synth.data_root
    leak = root / "processed/detleak/pages_validation.jsonl.gz"
    pages = list(read_pages(synth.train_file))[:1]
    write_pages(leak, [dataclasses.replace(p, source="detleak", split="validation") for p in pages])
    write_processed_manifest(root, "detleak", {"validation": leak})
    cfg = tiny_det_config(
        synth, tmp_path, "l", ["det.data.val_pages=[processed/detleak/pages_validation.jsonl.gz]"]
    )
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        DetTrainer(cfg)

    held = root / "processed/detheld/pages_validation.jsonl.gz"
    write_pages(held, [dataclasses.replace(p, source="detheld", split="test") for p in pages])
    cfg = tiny_det_config(
        synth, tmp_path, "t", ["det.data.val_pages=[processed/detheld/pages_validation.jsonl.gz]"]
    )
    with pytest.raises(ValueError, match="test-split"):
        DetTrainer(cfg)
    with pytest.raises(ConfigError, match="test or GDC"):
        tiny_det_config(
            synth, tmp_path, "t", ["det.data.val_pages=[processed/x/pages_test.jsonl.gz]"]
        )


# ----------------------------------------------------------------------------------- scripts


def test_sanity_sequence_runs_with_small_limits(synth, tmp_path, monkeypatch):
    import det_sanity

    monkeypatch.setattr(det_sanity, "LABEL_CEILING_PAGES", 4)
    monkeypatch.setattr(det_sanity, "OVERFIT_MAX_STEPS", 10)
    monkeypatch.setattr(det_sanity, "OVERFIT_CARE_RANGE", (1, 60))
    monkeypatch.setattr(det_sanity, "LR_STEPS", 10)
    cfg_path = _yaml(tiny_det_config(synth, tmp_path, "s"), tmp_path / "sanity.yaml")
    det_sanity.main(["--config", str(cfg_path), "--dry-run-minutes", "0.005"])
    out = tmp_path / "s/reports/training/sanity" / ID
    res = json.loads((out / "sanity.json").read_text(encoding="utf-8"))
    assert {"label_ceiling", "overfit", "lr_range", "dry_run"} <= set(res)
    ceiling = res["label_ceiling"]
    assert ceiling["pages"] == 4 and ceiling["mismatches"] == 0
    assert {"strict", "representational", "poly"} == set(ceiling["selection"])
    assert res["overfit"]["steps"] == 10 and res["overfit"]["final"]["step"] == 10
    assert res["lr_range"]["steps"] > 0 and res["dry_run"]["finite"]
    assert res["dry_run"]["steps"] > 0 and res["dry_run"]["val_pages"] == 8
    assert (out / "lr_range.png").exists() and "Overfit" in (out / "sanity.md").read_text("utf-8")
    assert not (tmp_path / "s/runs").exists()  # sanity trainers write to a temporary directory


def test_resume_check_passes_on_cpu_and_keeps_one_trainer_alive(synth, tmp_path, monkeypatch):
    import det_resume_check

    events: list[str] = []
    init, close = DetTrainer.__init__, DetTrainer.close

    def spy_init(self, *a, **k):
        events.append("init")
        init(self, *a, **k)

    def spy_close(self):
        events.append("close")
        close(self)

    monkeypatch.setattr(DetTrainer, "__init__", spy_init)
    monkeypatch.setattr(DetTrainer, "close", spy_close)
    real_free = det_resume_check.free_cuda
    monkeypatch.setattr(det_resume_check, "free_cuda", lambda: (events.append("free"), real_free()))
    monkeypatch.setattr(det_resume_check, "STEPS", 8)
    cfg_path = _yaml(tiny_det_config(synth, tmp_path, "r"), tmp_path / "resume.yaml")
    assert det_resume_check.main(["--config", str(cfg_path)]) == 0
    res = json.loads(
        (tmp_path / "r/reports/training/sanity" / ID / "resume_check.json").read_text()
    )
    assert res["passed"] and res["cut_mid_epoch"] and res["cut"] == 5
    assert res["steps_compared"] == 3 and res["boundary_rel_diff"] == 0 and res["lr_identical"]
    assert res["max_rel_loss_diff_resumed"] == 0 and res["max_rel_weight_diff_resumed"] == 0
    assert res["max_rel_loss_diff_two_uninterrupted"] == 0  # the CPU is deterministic
    assert res["val_f1_50_equal"]
    starts = [i for i, e in enumerate(events) if e == "init"]
    assert len(starts) == 3
    for prev, nxt in itertools.pairwise(starts):
        between = events[prev + 1 : nxt]
        assert "close" in between and "free" in between[between.index("close") :]


def test_det_eval_reproduces_its_metrics_from_saved_prob_maps(smoke, tmp_path):
    import det_eval

    from glypharis.eval.access import HeldOutAccessError

    root, cfg_path = smoke
    best = root / "a/ckpt/best.pt"
    common = [
        "--config",
        str(cfg_path),
        "--checkpoint",
        str(best),
        "--set",
        "det.post.box_thresh=0.0",
    ]
    maps = tmp_path / "maps"
    det_eval.main([*common, "--save-prob-maps", str(maps), "--out", str(tmp_path / "a.json")])
    det_eval.main([*common, "--from-prob-maps", str(maps), "--out", str(tmp_path / "b.json")])
    a = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert a["ops"] == b["ops"] and a["headline"] == b["headline"] and a["pages"] == 4
    assert sum(m["tp_fp_fn_50"][1] + m["tp_fp_fn_50"][0] for m in a["ops"].values()) > 0
    assert a["op"] == load_checkpoint(best)["selected_op"]
    with pytest.raises(HeldOutAccessError):
        det_eval.main([*common, "--split", "test"])


# -------------------------------------------------------------------------------- GPU only
# Helpers take the device so that the same code also runs on the CPU at a tiny size.


def _synthetic_targets(gen: torch.Generator, n: int, size: int, dev: torch.device):
    images = torch.randint(0, 256, (n, size, size, 3), generator=gen, dtype=torch.uint8)
    shrink = (torch.rand(n, 1, size, size, generator=gen) > 0.9).float()
    ones = torch.ones_like(shrink)
    thresh = 0.3 + 0.4 * torch.rand(n, 1, size, size, generator=gen)
    return images.to(dev).permute(0, 3, 1, 2), shrink.to(dev), ones.to(dev), thresh.to(dev)


def bf16_step(dev: torch.device, backbone: str, size: int, batch: int):
    model = build_detector(DetModelConfig(backbone=backbone, init="scratch"), Path("."), False)
    model = model.to(dev).train()
    images, shrink, mask, thresh = _synthetic_targets(
        torch.Generator().manual_seed(0), batch, size, dev
    )
    with resolve_amp("bf16", dev).autocast():
        logits = model.forward_train(images)
    parts = DBLoss(DetLossConfig())(logits, shrink, mask, thresh, mask)
    parts.total.backward()
    return logits, parts


def seeded_losses(dev: torch.device, steps: int, size: int) -> list[float]:
    seed_everything(0, True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        cfg = DetModelConfig(backbone="mobilenet_v3_small", init="scratch", inner_channels=16)
        model = build_detector(cfg, Path("."), False).to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn, amp = DBLoss(DetLossConfig()), resolve_amp("bf16", dev)
        gen = torch.Generator().manual_seed(1)
        losses = []
        for _ in range(steps):
            images, shrink, mask, thresh = _synthetic_targets(gen, 2, size, dev)
            with amp.autocast():
                logits = model.forward_train(images)
            total = loss_fn(logits, shrink, mask, thresh, mask).total
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            losses.append(float(total.detach()))
        return losses
    finally:
        torch.use_deterministic_algorithms(True, warn_only=True)


def prob_maps(devices: tuple[torch.device, torch.device], size: tuple[int, int]):
    from glypharis.models.det.preprocess import prepare_page
    from glypharis.models.det.runtime import TorchProbMap

    model = build_detector(
        DetModelConfig(backbone="mobilenet_v3_small", init="scratch"), Path("."), False
    )
    rgb = np.random.default_rng(3).integers(0, 256, (*size, 3), dtype=np.uint8)
    image, info = prepare_page(rgb, max(size))
    maps = [TorchProbMap(copy.deepcopy(model), d)(image) for d in devices]
    return maps, info


@pytest.mark.gpu
def test_bf16_training_step_keeps_logits_and_loss_parts_in_fp32():
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(dev)
    logits, parts = bf16_step(dev, "mobilenet_v3_large", 640, 2)
    assert logits.prob.dtype == torch.float32 and logits.thresh.dtype == torch.float32
    for value in (parts.total, parts.bce, parts.dice, parts.l1):
        assert value.dtype == torch.float32 and torch.isfinite(value)
    print(f"peak allocated {torch.cuda.max_memory_allocated(dev) / 2**20:.0f} MiB")


@pytest.mark.gpu
def test_seeded_cuda_runs_are_identical_with_deterministic_kernels_enforced():
    dev = torch.device("cuda")
    assert seeded_losses(dev, 20, 128) == seeded_losses(dev, 20, 128)


@pytest.mark.gpu
def test_cuda_probability_maps_match_the_cpu():
    import shapely

    from glypharis.config.schema import DetPostConfig
    from glypharis.models.det.postprocess import boxes_from_candidates, find_candidates

    (cpu, gpu), info = prob_maps((torch.device("cpu"), torch.device("cuda")), (300, 420))
    assert np.abs(cpu - gpu).max() <= 1e-4
    post = dataclasses.replace(DetPostConfig(), box_thresh=0.0)
    boxes = []
    for prob in (cpu, gpu):
        cands, _ = find_candidates(prob, info, post)
        boxes.append(boxes_from_candidates(cands, post.operating_points[0], post, 0.4, info)[0])
    assert len(boxes[0]) == len(boxes[1])
    for a, b in zip(*boxes, strict=True):
        pa, pb = shapely.Polygon(a.polygon), shapely.Polygon(b.polygon)
        assert pa.intersection(pb).area / pa.union(pb).area >= 0.99


@pytest.mark.gpu
def test_rng_state_restores_from_a_detector_checkpoint_loaded_onto_the_gpu(synth, tmp_path):
    trainer = DetTrainer(
        tiny_det_config(synth, tmp_path, "c", ["device=cuda", "train.max_steps=1"])
    )
    trainer.fit()
    trainer.close()
    ck = load_checkpoint(tmp_path / "c/ckpt/last.pt", "cuda")  # every tensor lands on the GPU
    assert "selected_op" in ck and "detector_spec" in ck
    set_rng_state(ck["rng"])
    first = torch.rand(3)
    set_rng_state(ck["rng"])
    assert torch.equal(torch.rand(3), first)


@pytest.mark.gpu
def test_a_second_trainer_passes_the_preflight_after_close_and_free(synth, tmp_path):
    dev = torch.device("cuda")
    free_cuda()
    start = torch.cuda.memory_allocated(dev)
    first = DetTrainer(tiny_det_config(synth, tmp_path, "p1", ["device=cuda", "train.max_steps=2"]))
    first.fit()
    first.close()
    del first
    free_cuda()
    assert torch.cuda.memory_allocated(dev) == start
    second = DetTrainer(tiny_det_config(synth, tmp_path, "p2", ["device=cuda"]))
    second.close()


def test_best_checkpoint_is_the_highest_f1_and_min_delta_only_drives_patience():
    from glypharis.config.schema import EarlyStoppingConfig
    from glypharis.training.loop import LoopState, record_validation

    es = EarlyStoppingConfig(enabled=True, mode="max", patience=3, min_delta=0.001)
    st = LoopState(best_metric=-math.inf, patience_ref=-math.inf)
    # Every epoch is a little better, but only epoch 3 by more than min_delta.
    vals = [0.800, 0.8004, 0.8007, 0.8015, 0.8016, 0.8017, 0.8018]
    picked = [record_validation(st, v, e, es, maximize=True) for e, v in enumerate(vals)]
    assert picked == [True] * 7
    assert st.best_epoch == 6 and st.best_metric == 0.8018
    assert st.patience_ref == 0.8015 and st.bad_epochs == 3 and st.stopped_early
    assert not record_validation(st, 0.8018, 7, es, maximize=True)  # a tie keeps the first
