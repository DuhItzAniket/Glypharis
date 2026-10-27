"""Trainer for the line recognizer (Phase 6; on the shared loop since Phase 11): CTC training,
validation, diagnostics, resume.

Steps are optimizer steps; with gradient accumulation one step spans ``gradient_accumulation``
batches. The learning-rate schedule is stepped once per optimizer step and sized from the epoch
budget, so a run cut short by ``train.max_steps`` follows the same schedule as the full run.

Exact resume: a checkpoint is written only at optimizer-step boundaries and stores the model,
optimizer, scheduler, scaler, early-stopping state, metric history, diagnostics state, the position
inside the epoch and every global RNG state. Batch order and augmentation are pure functions of
(seed, epoch, index) (`glypharis.data.rec_dataset`), and each epoch's DataLoader draws its worker
seed from its own generator, so the global RNG (dropout) is not disturbed by starting an epoch.
On the CPU a resumed run is bit-identical; on CUDA it matches within the protocol tolerance (CTC
backward has no deterministic kernel).

Outputs: ``<runs_dir>/<experiment_id>/`` (metrics.jsonl, resolved config, summary.json),
``<checkpoint.dir>/`` (best.pt, last.pt, epoch_XXX.pt) and the run report under
``<reports_dir>/training/<experiment_id>/`` (`glypharis.training.report`).
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from glypharis.config.loader import config_hash, config_to_dict
from glypharis.config.schema import ConfigError, EarlyStoppingConfig, RunConfig
from glypharis.data.augment import Augmenter
from glypharis.data.rec_dataset import (
    RecLineDataset,
    WidthBucketBatchSampler,
    assert_no_leakage,
    collate,
    ctc_frames_needed,
)
from glypharis.data.text import Charset
from glypharis.diagnostics import (
    EpochPoint,
    check_collapse,
    check_overfitting,
    check_underfitting,
)
from glypharis.eval.text_metrics import RecAccumulator, score_pair
from glypharis.models.rec.base import LineRecognizer
from glypharis.models.rec.crnn import build_recognizer, count_parameters
from glypharis.training.checkpoint import (
    load_checkpoint,
    set_rng_state,
)
from glypharis.training.loop import EpochResult, LoopState, StepResult, TrainerCore
from glypharis.training.loop import record_validation as loop_record_validation
from glypharis.training.optim import make_optimizer, make_scheduler
from glypharis.utils.device import AmpPolicy, fp32_region
from glypharis.utils.seed import make_generator, worker_init_fn

log = logging.getLogger(__name__)
__all__ = [
    "RecTrainer",
    "ctc_loss",
    "evaluate",
    "greedy_decode",
    "make_optimizer",
    "make_scheduler",
]


# ------------------------------------------------------------------------------ building blocks


def ctc_loss(
    model: LineRecognizer,
    batch: dict[str, Any],
    amp: AmpPolicy,
    device: torch.device,
    ctc: nn.CTCLoss,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass under autocast; CTC in fp32. Returns (loss, logits, frame lengths)."""
    images = batch["images"].to(device, non_blocking=True)
    with amp.autocast():
        logits = model(images, batch["widths"])
    t = logits.shape[0]
    lengths = model.frames(batch["widths"]).clamp(max=t)
    with fp32_region(device.type):
        log_probs = logits.float().log_softmax(2)
        loss = ctc(log_probs, batch["targets"], lengths, batch["target_lengths"])
    return loss, logits, lengths


def greedy_decode(logits: torch.Tensor, lengths: torch.Tensor, charset: Charset) -> list[str]:
    best = logits.argmax(2).transpose(0, 1).cpu()  # N, T
    return [
        charset.ctc_greedy_decode(row[: int(n)].tolist())
        for row, n in zip(best, lengths, strict=True)
    ]


@dataclass
class EvalResult:
    loss: float
    cer: float
    wer: float
    ned: float
    sequence_accuracy: float
    cer_normalised: float
    blank_ratio: float
    empty_ratio: float
    identical_ratio: float
    samples: list[dict[str, Any]] = field(default_factory=list)  # key, label, pred, cer


@torch.no_grad()
def evaluate(
    model: LineRecognizer,
    loader: Iterable[dict[str, Any]],
    charset: Charset,
    amp: AmpPolicy,
    device: torch.device,
    ctc: nn.CTCLoss,
) -> EvalResult:
    """Loss, recognition metrics (both tracks) and collapse statistics over a loader."""
    was_training = model.training
    model.eval()
    strict, norm = RecAccumulator("strict"), RecAccumulator("normalised")
    loss_sum, n_lines, blank, frames = 0.0, 0, 0, 0
    samples: list[dict[str, Any]] = []
    for batch in loader:
        loss, logits, lengths = ctc_loss(model, batch, amp, device, ctc)
        n = len(batch["keys"])
        loss_sum += float(loss) * n
        n_lines += n
        arg = logits.argmax(2).transpose(0, 1).cpu()
        for row, k in zip(arg, lengths, strict=True):
            blank += int((row[: int(k)] == 0).sum())
            frames += int(k)
        preds = greedy_decode(logits, lengths, charset)
        for key, label, pred in zip(batch["keys"], batch["labels"], preds, strict=True):
            sc = strict.add(pred, label)
            norm.add(pred, label)
            cer = sc.char_errors / sc.ref_chars if sc and sc.ref_chars else 0.0
            samples.append({"key": key, "label": label, "pred": pred, "cer": cer})
    model.train(was_training)
    s, nn_ = strict.summary(), norm.summary()
    preds = [x["pred"] for x in samples]
    top = Counter(preds).most_common(1)[0][1] if preds else 0
    return EvalResult(
        loss=loss_sum / max(n_lines, 1),
        cer=s.cer,
        wer=s.wer,
        ned=s.ned,
        sequence_accuracy=s.sequence_accuracy,
        cer_normalised=nn_.cer,
        blank_ratio=blank / max(frames, 1),
        empty_ratio=sum(1 for p in preds if not p) / max(len(preds), 1),
        identical_ratio=top / max(len(preds), 1),
        samples=samples,
    )


def _pack_versions(data_root: Path, packs: Iterable[str]) -> dict[str, Any]:
    out = {}
    for p in packs:
        meta = json.loads((data_root / f"{p}.meta.json").read_text(encoding="utf-8"))
        out[p] = {"count": meta["count"], "bin_sha256": meta["bin_sha256"]}
    return out


# --------------------------------------------------------------------------------- the trainer


def config_matches(saved: dict[str, Any], cfg: RunConfig) -> bool:
    """True when a checkpoint's config is this run's: only the resume pointer and ``max_steps``
    (which the resume check sets) may differ."""
    saved, now = dict(saved), config_to_dict(cfg)
    for d in (saved, now):
        d["checkpoint"] = {**d["checkpoint"], "resume": None}
        d["train"] = {**d["train"], "max_steps": None}
    return saved == now


def model_bytes(state: dict[str, torch.Tensor]) -> int:
    """Size of a model's weights and buffers as stored (fp32 for the recognizer)."""
    return sum(t.numel() * t.element_size() for t in state.values())


def batching_widths(ds: RecLineDataset, charset: Charset, stride: int) -> np.ndarray:
    """Columns each line takes once padded for CTC: at least ``stride`` per frame its label needs.
    At stride 8, 3% of HierText training lines are denser than that, up to 3,256 columns, which
    the image widths alone never showed the column cap (GW-REC-019 ran out of memory)."""
    need = np.array(
        [stride * ctc_frames_needed(charset.encode(r.label)) for r in ds.refs], dtype=np.int64
    )
    return np.maximum(ds.widths, need)


def run_config_hash(cfg: RunConfig) -> str:
    """The run's config hash. The resume pointer says how a run was continued, not what it is,
    so a resumed run reports the same hash as at its start."""
    return config_hash(replace(cfg, checkpoint=replace(cfg.checkpoint, resume=None)))


def _loop_state(saved: dict[str, Any]) -> LoopState:
    """The loop state of a recognizer checkpoint. Phase 6 and 7 checkpoints hold the same fields
    without ``extra``; before ``patience_ref`` existed, ``best_metric`` played its part."""
    saved = dict(saved)
    saved.setdefault("patience_ref", saved.get("best_metric", math.inf))
    return LoopState(**saved)


# The Phase 6 names, kept for the scripts and tests that use them.
TrainState = LoopState
state_from_checkpoint = _loop_state


def record_validation(state: LoopState, metric: float, epoch: int, es: EarlyStoppingConfig) -> bool:
    """`glypharis.training.loop.record_validation` for a metric to minimise (CER)."""
    return loop_record_validation(state, metric, epoch, es, maximize=False)


class RecTrainer(TrainerCore):
    """CTC training of a line recognizer on the shared loop (`glypharis.training.loop`).

    Checkpoints keep the Phase 6 layout (model, optimizer, scheduler, scaler, state, gradient
    monitor, RNG states, the resolved config and its run hash, the charset), so a run started before
    the move resumes on it. A resume also moves the log records written after its checkpoint to
    ``metrics.dropped.jsonl``, as before.
    """

    task = "rec"
    throughput_key = "lines_per_s"

    def __init__(
        self,
        cfg: RunConfig,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(cfg, on_step)
        charset_path = (
            Path(cfg.data.charset)
            if cfg.data.charset
            else (self.data_root / "manifests" / "charset.json")
        )
        self.charset = Charset.load(charset_path)

        d = cfg.data
        if not d.train or not d.val:
            raise ConfigError("data.train and data.val must list line packs")
        if any("test" in p for p in [*d.train, *d.val]):
            raise ConfigError("test packs are never used for training or validation (protocol §9)")
        aug = Augmenter(d.augmentation, d.augmentation_params)
        train_packs = [self.data_root / p for p in d.train]
        limits = {str(self.data_root / k): v for k, v in d.train_limits.items()}
        self.train_ds = RecLineDataset(
            train_packs,
            self.charset,
            d.image_height,
            d.max_width,
            aug,
            cfg.seed,
            limits,
            d.dataset_subset,
        )
        self.val_ds = RecLineDataset(
            [self.data_root / p for p in d.val],
            self.charset,
            d.image_height,
            d.max_width,
            None,
            cfg.seed,
            subset=d.val_subset,
        )
        assert_no_leakage(self.train_ds, self.val_ds)
        # A fixed, un-augmented sample of the training lines, scored like validation each epoch.
        self.train_eval_ds = RecLineDataset(
            train_packs,
            self.charset,
            d.image_height,
            d.max_width,
            None,
            cfg.seed,
            limits,
            d.dataset_subset,
        )
        if d.train_eval_subset and d.train_eval_subset < len(self.train_eval_ds):
            keep = np.random.default_rng([cfg.seed, 3]).choice(
                len(self.train_eval_ds), d.train_eval_subset, replace=False
            )
            self.train_eval_ds.refs = [self.train_eval_ds.refs[int(i)] for i in np.sort(keep)]
            self.train_eval_ds.widths = self.train_eval_ds.widths[np.sort(keep)]

        model = build_recognizer(self.charset.num_classes, cfg.model, cfg.model_name or "crnn").to(
            self.device
        )
        self.model = model
        t = cfg.train
        self.sampler = WidthBucketBatchSampler(
            self._batching_widths(self.train_ds),
            t.batch_size,
            cfg.seed,
            d.bucket_chunk,
            max_columns=d.max_batch_columns,
        )
        # With a column cap the batch count can differ between epochs; it is known in advance.
        self.setup_optimization(model, [self.sampler.num_batches(e) for e in range(t.epochs)])
        self.ctc = nn.CTCLoss(blank=Charset.BLANK, zero_infinity=True)
        self.params = count_parameters(self.model)
        self._val_samples: list[dict[str, Any]] = []
        # Data-loading processes. Each loads its own torch and CUDA (about 2 GB of commit memory on
        # Windows); GW-REC-005 and 007 crashed when the system ran out of commit, so the queue can
        # lower this without touching the config. Batches do not depend on it.
        self.workers = int(os.environ.get("GLYPHARIS_NUM_WORKERS", cfg.data.num_workers))

    @property
    def rec_model(self) -> LineRecognizer:
        return cast(LineRecognizer, self.model)

    def _batching_widths(self, ds: RecLineDataset) -> np.ndarray:
        """The widths batches are bucketed and capped by: each line's image width, or with
        ``data.ctc_aware_batching`` the width collate pads it to for CTC, whichever is larger."""
        if not self.cfg.data.ctc_aware_batching:
            return ds.widths
        return batching_widths(ds, self.charset, self.rec_model.stride)

    # ---------------------------------------------------------------------------- loaders

    def _loader(self, ds: RecLineDataset, sampler: Any, seed: int) -> DataLoader[Any]:
        workers = self.workers
        return DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=partial(collate, stride=self.rec_model.stride),
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=worker_init_fn if workers else None,
            generator=make_generator(seed),  # worker seeds never touch the global RNG
            persistent_workers=False,
        )

    def _eval_loader(self, ds: RecLineDataset) -> DataLoader[Any]:
        cap = self.cfg.data.max_batch_columns
        sampler = WidthBucketBatchSampler(
            self._batching_widths(ds),
            self.cfg.train.batch_size * 2,
            self.cfg.seed,
            shuffle=False,
            max_columns=2 * cap if cap else None,
        )
        return self._loader(ds, sampler, self.cfg.seed + 7)

    # ---------------------------------------------------------------------- task interface

    def train_batches(self, epoch: int, start_batch: int) -> Iterable[dict[str, Any]]:
        self.sampler.set_epoch(epoch, start_batch)
        return self._loader(self.train_ds, self.sampler, self.cfg.seed + 1000 + epoch)

    def train_step(self, batch: dict[str, Any]) -> StepResult:
        loss, _, _ = ctc_loss(self.rec_model, batch, self.amp, self.device, self.ctc)
        return StepResult(loss, len(batch["keys"]), list(batch["keys"]), {})

    def step_extra(self) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {}
        return {"vram_peak_mib": torch.cuda.max_memory_allocated() / 2**20}

    def validate(self) -> EpochResult:
        cfg, st = self.cfg, self.state
        model = self.rec_model
        val = evaluate(
            model,
            self._eval_loader(self.val_ds),
            self.charset,
            self.amp,
            self.device,
            self.ctc,
        )
        tr = evaluate(
            model,
            self._eval_loader(self.train_eval_ds),
            self.charset,
            self.amp,
            self.device,
            self.ctc,
        )
        point: dict[str, Any] = {
            "train_eval_loss": tr.loss,
            "train_cer": tr.cer,
            "val_loss": val.loss,
            "val_cer": val.cer,
            "val_wer": val.wer,
            "val_ned": val.ned,
            "val_sequence_accuracy": val.sequence_accuracy,
            "val_cer_normalised": val.cer_normalised,
            "blank_ratio": val.blank_ratio,
            "empty_ratio": val.empty_ratio,
            "identical_ratio": val.identical_ratio,
        }
        if self.device.type == "cuda":
            point["vram_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
        epoch, step = st.epoch, st.step
        hist = [
            EpochPoint(
                p["epoch"],
                p["step"],
                p["train_eval_loss"],
                p["val_loss"],
                p["train_cer"],
                p["val_cer"],
            )
            for p in st.history
        ]
        hist.append(EpochPoint(epoch, step, tr.loss, val.loss, tr.cer, val.cer))
        d = cfg.diagnostics
        flags = [
            check_overfitting(hist, d),
            check_underfitting(hist, d, step / max(self.total_steps, 1)),
            *check_collapse(val.blank_ratio, val.empty_ratio, val.identical_ratio, d, epoch, step),
        ]
        self._val_samples = val.samples
        # early_stopping.monitor is a validation metric (validated); the recognizer minimises CER
        return EpochResult(point, val.cer, [f for f in flags if f is not None])

    def on_best(self) -> None:
        self._write_samples(self._val_samples)

    def payload_extra(self) -> dict[str, Any]:
        # The run hash, not the hash with the resume pointer: as Phase 6 and 7 checkpoints hold it.
        return {"config_hash": run_config_hash(self.cfg), "charset": self.charset.characters}

    # ------------------------------------------------------------------------- resume

    def resume(self, path: str | Path) -> None:
        ck = load_checkpoint(path, self.device)
        if not config_matches(ck["config"], self.cfg):
            raise ConfigError(f"{path} was written with a different config; refusing to resume")
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if self.scheduler and ck["scheduler"]:
            self.scheduler.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.state = _loop_state(ck["state"])
        self.gradmon.load_state_dict(ck["gradmon"])
        set_rng_state(ck["rng"])
        dropped = self._trim_log(self.state.step)
        self._log(
            {
                "type": "resume",
                "from": str(path),
                "epoch": self.state.epoch,
                "step": self.state.step,
                "dropped_records": dropped,
            }
        )
        log.info(
            "resumed from %s at epoch %d, batch %d, step %d",
            path,
            self.state.epoch,
            self.state.batch_in_epoch,
            self.state.step,
        )

    def _trim_log(self, step: int) -> int:
        """Move log records written after the checkpoint at ``step`` (a run that died between two
        saves) to metrics.dropped.jsonl, so the resumed run does not log those steps twice.
        Returns how many were moved."""
        path = self.run_dir / "metrics.jsonl"
        if not path.exists():
            return 0
        keep: list[str] = []
        drop: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                (drop if json.loads(line).get("step", 0) > step else keep).append(line)
        if drop:
            text = "\n".join(keep) + "\n"
            tmp = path.with_suffix(".jsonl.tmp")
            tmp.write_text(text, encoding="utf-8", newline="\n")
            try:
                tmp.replace(path)
            except PermissionError:
                # On Windows a file another process holds open (a log viewer, `tail -F`) cannot be
                # replaced, but it can be rewritten; the full copy in tmp covers a crash meanwhile.
                path.write_text(text, encoding="utf-8", newline="\n")
                tmp.unlink()
            # Only once the log no longer holds them, so a failed attempt cannot copy them twice.
            with (self.run_dir / "metrics.dropped.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as fh:
                fh.write("\n".join(drop) + "\n")
        return len(drop)

    # ------------------------------------------------------------------------ outputs

    def _write_samples(self, samples: list[dict[str, Any]]) -> None:
        """Per-line validation results of the best checkpoint, for the report and the CI."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "val_best_samples.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as fh:
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    def summary(self) -> dict[str, Any]:
        s = super().summary()
        s.update(
            config_hash=run_config_hash(self.cfg),
            parameters=self.params,
            data_workers=self.workers,
            model_bytes=model_bytes(self.model.state_dict()),
            train_lines=len(self.train_ds),
            val_lines=len(self.val_ds),
            data=_pack_versions(self.data_root, [*self.cfg.data.train, *self.cfg.data.val]),
        )
        return s

    def _summary(self) -> dict[str, Any]:
        """The run summary (the Phase 6 name, kept for train.py and the tests)."""
        return self.summary()


def score_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute per-line strict CER (used by the report for the worst-predicted lines)."""
    out = []
    for s in samples:
        sc = score_pair(s["pred"], s["label"])
        out.append({**s, "cer": sc.char_errors / sc.ref_chars if sc.ref_chars else 0.0})
    return out
