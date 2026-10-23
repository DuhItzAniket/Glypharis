"""Trainer for the line recognizer (Phase 6): CTC training, validation, diagnostics, resume.

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
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from glypharis.config.loader import config_hash, config_to_dict, save_resolved
from glypharis.config.schema import ConfigError, EarlyStoppingConfig, RunConfig
from glypharis.data.augment import Augmenter
from glypharis.data.rec_dataset import (
    RecLineDataset,
    WidthBucketBatchSampler,
    assert_no_leakage,
    collate,
)
from glypharis.data.text import Charset
from glypharis.diagnostics import (
    EpochPoint,
    Flag,
    GradientMonitor,
    NonFiniteLoss,
    check_collapse,
    check_overfitting,
    check_underfitting,
)
from glypharis.eval.text_metrics import RecAccumulator, score_pair
from glypharis.models.rec.base import LineRecognizer
from glypharis.models.rec.crnn import build_recognizer, count_parameters
from glypharis.training.checkpoint import (
    load_checkpoint,
    prune_epoch_checkpoints,
    rng_state,
    save_checkpoint,
    set_rng_state,
)
from glypharis.utils.device import AmpPolicy, fp32_region, resolve_amp, select_device
from glypharis.utils.gitinfo import git_info
from glypharis.utils.seed import make_generator, seed_everything, worker_init_fn

log = logging.getLogger(__name__)
CHECKPOINT_FORMAT = 1


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


def make_optimizer(cfg: RunConfig, model: nn.Module) -> torch.optim.Optimizer:
    t = cfg.train
    params, lr, wd = model.parameters(), t.learning_rate, t.weight_decay
    if t.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if t.optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)


def make_scheduler(
    cfg: RunConfig, optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    t = cfg.train
    if t.scheduler == "none":
        return None
    if t.scheduler == "onecycle":
        pct = t.warmup_steps / total_steps if t.warmup_steps else 0.3
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=t.learning_rate, total_steps=total_steps, pct_start=min(pct, 0.5)
        )
    if t.scheduler == "cosine":
        warm = t.warmup_steps

        def factor(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            progress = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    raise ConfigError(f"train.scheduler {t.scheduler!r} is not supported by the recognizer trainer")


def _pack_versions(data_root: Path, packs: Iterable[str]) -> dict[str, Any]:
    out = {}
    for p in packs:
        meta = json.loads((data_root / f"{p}.meta.json").read_text(encoding="utf-8"))
        out[p] = {"count": meta["count"], "bin_sha256": meta["bin_sha256"]}
    return out


# --------------------------------------------------------------------------------- the trainer


@dataclass
class TrainState:
    epoch: int = 0
    batch_in_epoch: int = 0  # batches of the current epoch already consumed
    step: int = 0  # optimizer steps
    best_metric: float = math.inf  # lowest validation metric so far: selects best.pt
    best_epoch: int = -1
    patience_ref: float = math.inf  # what early stopping measures progress against
    bad_epochs: int = 0
    stopped_early: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)
    train_seconds: float = 0.0
    epoch_loss_sum: float = 0.0  # running training loss of the current epoch
    epoch_loss_n: int = 0


def record_validation(
    state: TrainState, metric: float, epoch: int, es: EarlyStoppingConfig
) -> bool:
    """Update checkpoint selection and early stopping with one epoch's validation metric.

    The selected checkpoint is the one with the strictly lowest validation metric (protocol §7; ties
    keep the earlier epoch). ``min_delta`` only decides whether an epoch counts as progress for the
    patience counter, so a small improvement still becomes the best checkpoint. Returns True when
    this epoch is the new best.
    """
    best = metric < state.best_metric
    if best:
        state.best_metric, state.best_epoch = metric, epoch
    if metric < state.patience_ref - es.min_delta:
        state.patience_ref, state.bad_epochs = metric, 0
    else:
        state.bad_epochs += 1
        if es.enabled and state.bad_epochs >= es.patience:
            state.stopped_early = True
    return best


def state_from_checkpoint(saved: dict[str, Any]) -> TrainState:
    """Rebuild the state. Before ``patience_ref`` existed, ``best_metric`` played its part."""
    saved = dict(saved)
    saved.setdefault("patience_ref", saved.get("best_metric", math.inf))
    return TrainState(**saved)


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


def run_config_hash(cfg: RunConfig) -> str:
    """The run's config hash. The resume pointer says how a run was continued, not what it is,
    so a resumed run reports the same hash as at its start."""
    return config_hash(replace(cfg, checkpoint=replace(cfg.checkpoint, resume=None)))


class RecTrainer:
    def __init__(
        self,
        cfg: RunConfig,
        on_step: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        cfg.require_run_identity()
        if cfg.task != "rec":
            raise ConfigError(f"RecTrainer needs task 'rec', got {cfg.task!r}")
        self.cfg = cfg
        self.on_step = on_step
        assert cfg.experiment_id is not None
        self.run_dir = Path(cfg.paths.runs_dir) / cfg.experiment_id
        self.ckpt_dir = Path(cfg.checkpoint.dir)
        self.report_dir = Path(cfg.paths.reports_dir) / "training" / cfg.experiment_id
        self.data_root = Path(cfg.paths.data_root)

        seed_everything(cfg.seed, cfg.train.deterministic)
        self.device = select_device(cfg.device)
        self.amp = resolve_amp(cfg.train.mixed_precision, self.device)
        charset_path = Path(cfg.data.charset) if cfg.data.charset else (
            self.data_root / "manifests" / "charset.json")  # fmt: skip
        self.charset = Charset.load(charset_path)

        d = cfg.data
        if not d.train or not d.val:
            raise ConfigError("data.train and data.val must list line packs")
        if any("test" in p for p in [*d.train, *d.val]):
            raise ConfigError("test packs are never used for training or validation (protocol §9)")
        aug = Augmenter(d.augmentation, d.augmentation_params)
        train_packs = [self.data_root / p for p in d.train]
        limits = {str(self.data_root / k): v for k, v in d.train_limits.items()}
        self.train_ds = RecLineDataset(train_packs, self.charset, d.image_height, d.max_width, aug,
                                       cfg.seed, limits, d.dataset_subset)  # fmt: skip
        self.val_ds = RecLineDataset([self.data_root / p for p in d.val], self.charset,
                                     d.image_height, d.max_width, None, cfg.seed,
                                     subset=d.val_subset)  # fmt: skip
        assert_no_leakage(self.train_ds, self.val_ds)
        # A fixed, un-augmented sample of the training lines, scored like validation each epoch.
        self.train_eval_ds = RecLineDataset(train_packs, self.charset, d.image_height, d.max_width,
                                            None, cfg.seed, limits, d.dataset_subset)  # fmt: skip
        if d.train_eval_subset and d.train_eval_subset < len(self.train_eval_ds):
            keep = np.random.default_rng([cfg.seed, 3]).choice(
                len(self.train_eval_ds), d.train_eval_subset, replace=False)  # fmt: skip
            self.train_eval_ds.refs = [self.train_eval_ds.refs[int(i)] for i in np.sort(keep)]
            self.train_eval_ds.widths = self.train_eval_ds.widths[np.sort(keep)]

        self.model = build_recognizer(
            self.charset.num_classes, cfg.model, cfg.model_name or "crnn"
        ).to(self.device)
        self.optimizer = make_optimizer(cfg, self.model)
        t = cfg.train
        self.sampler = WidthBucketBatchSampler(self.train_ds.widths, t.batch_size, cfg.seed,
                                               d.bucket_chunk,
                                               max_columns=d.max_batch_columns)  # fmt: skip
        # With a column cap the batch count can differ between epochs; it is known in advance.
        self.epoch_batches = [self.sampler.num_batches(e) for e in range(t.epochs)]
        accum = t.gradient_accumulation
        self.steps_per_epoch = [math.ceil(n / accum) for n in self.epoch_batches]
        self.total_steps = sum(self.steps_per_epoch)
        self.scheduler = make_scheduler(cfg, self.optimizer, self.total_steps)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp.needs_grad_scaler)
        self.ctc = nn.CTCLoss(blank=Charset.BLANK, zero_infinity=True)
        self.gradmon = GradientMonitor(cfg.diagnostics)
        self.state = TrainState()
        self._clock = time.monotonic()
        self.git = git_info()
        self.params = count_parameters(self.model)

    # ---------------------------------------------------------------------------- loaders

    def _loader(self, ds: RecLineDataset, sampler: Any, seed: int) -> DataLoader[Any]:
        workers = self.cfg.data.num_workers
        return DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=partial(collate, stride=self.model.stride),
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=worker_init_fn if workers else None,
            generator=make_generator(seed),  # worker seeds never touch the global RNG
            persistent_workers=False,
        )

    def _eval_loader(self, ds: RecLineDataset) -> DataLoader[Any]:
        cap = self.cfg.data.max_batch_columns
        sampler = WidthBucketBatchSampler(ds.widths, self.cfg.train.batch_size * 2, self.cfg.seed,
                                          shuffle=False,
                                          max_columns=2 * cap if cap else None)  # fmt: skip
        return self._loader(ds, sampler, self.cfg.seed + 7)

    # ------------------------------------------------------------------------ checkpoints

    def _payload(self, metric: float | None) -> dict[str, Any]:
        return {
            "format": CHECKPOINT_FORMAT,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict(),
            "state": asdict(self.state),
            "gradmon": self.gradmon.state_dict(),
            "metric": metric,
            "config": config_to_dict(self.cfg),
            "config_hash": run_config_hash(self.cfg),
            "git": self.git.to_dict(),
            "charset": self.charset.characters,
            "rng": rng_state(),
        }

    def save(self, name: str, metric: float | None = None) -> Path:
        return save_checkpoint(self.ckpt_dir / name, self._payload(metric))

    def resume(self, path: str | Path) -> None:
        ck = load_checkpoint(path, self.device)
        if not config_matches(ck["config"], self.cfg):
            raise ConfigError(f"{path} was written with a different config; refusing to resume")
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if self.scheduler and ck["scheduler"]:
            self.scheduler.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.state = state_from_checkpoint(ck["state"])
        self.gradmon.load_state_dict(ck["gradmon"])
        set_rng_state(ck["rng"])
        dropped = self._trim_log(self.state.step)
        self._log({"type": "resume", "from": str(path), "epoch": self.state.epoch,
                   "step": self.state.step, "dropped_records": dropped})  # fmt: skip
        log.info("resumed from %s at epoch %d, batch %d, step %d", path, self.state.epoch,
                 self.state.batch_in_epoch, self.state.step)  # fmt: skip

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
            with (self.run_dir / "metrics.dropped.jsonl").open("a", encoding="utf-8",
                                                              newline="\n") as fh:  # fmt: skip
                fh.write("\n".join(drop) + "\n")
        return len(drop)

    # --------------------------------------------------------------------------- logging

    def _log(self, record: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record) + "\n")
        if self.on_step is not None:
            self.on_step(record)

    def _flag(self, flag: Flag | None) -> None:
        if flag is not None:
            self.state.flags.append(flag.to_dict())
            log.warning("%s at epoch %d step %d: %s", flag.name, flag.epoch, flag.step,
                        flag.evidence)  # fmt: skip
            self._log({"type": "flag", **flag.to_dict()})

    # ----------------------------------------------------------------------------- train

    def _tick(self) -> None:
        """Add the time since the last tick to the state (saved with every checkpoint)."""
        now = time.monotonic()
        self.state.train_seconds += now - self._clock
        self._clock = now

    def _optimizer_step(self, keys: list[str]) -> float:
        """Clip and apply the accumulated gradients of the batches ``keys``; returns the norm
        before clipping.

        A finite loss can still back-propagate to an inf or NaN gradient (a bf16 overflow, a 0/0 in
        a backward). Stepping on it would turn every weight into NaN with no flag, and the next
        batch's loss would take the blame. The fp16 scaler skips such a step by itself; without it
        the run stops here with the weights untouched (or skips the step if stop_on_nan is off).
        """
        if self.amp.needs_grad_scaler:
            self.scaler.unscale_(self.optimizer)
        clip = self.cfg.train.grad_clip_norm
        norm = float(nn.utils.clip_grad_norm_(self.model.parameters(),
                                              clip if clip is not None else math.inf))  # fmt: skip
        if math.isfinite(norm) or self.amp.needs_grad_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        elif self.cfg.diagnostics.stop_on_nan:
            self._non_finite(keys, "grad_norm", norm)
        else:
            self._log({"type": "nonfinite_grad_skipped", "epoch": self.state.epoch,
                       "step": self.state.step, "grad_norm": str(norm), "keys": keys})  # fmt: skip
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.state.step += 1
        return norm

    def _non_finite(self, keys: list[str], what: str, value: float) -> NoReturn:
        """Save nonfinite.pt and the batch keys, then stop the run; ``what`` names the value."""
        self.save("nonfinite.pt")
        self._log({"type": "nonfinite", "epoch": self.state.epoch, "step": self.state.step,
                   what: str(value), "keys": keys})  # fmt: skip
        raise NonFiniteLoss(f"non-finite {what} {value} at step {self.state.step}; "
                            f"batch keys and state saved")  # fmt: skip

    def fit(self) -> dict[str, Any]:
        cfg, t = self.cfg, self.cfg.train
        if cfg.checkpoint.resume:
            self.resume(cfg.checkpoint.resume)
        else:
            save_resolved(cfg, self.run_dir / "config.resolved.yaml")
        self.model.train()
        accum = t.gradient_accumulation
        done = False
        while self.state.epoch < t.epochs and not done and not self.state.stopped_early:
            epoch = self.state.epoch
            self.sampler.set_epoch(epoch, self.state.batch_in_epoch)
            loader = self._loader(self.train_ds, self.sampler, cfg.seed + 1000 + epoch)
            # The epoch's running loss and the training time live in the state, so a resumed
            # epoch continues them instead of starting over at the resume point.
            st = self.state
            t0, lines, step_sum, step_n = time.monotonic(), 0, 0.0, 0
            group: list[str] = []  # keys of the batches the next optimizer step applies
            self._clock = time.monotonic()
            for batch in loader:
                try:
                    loss, _, _ = ctc_loss(self.model, batch, self.amp, self.device, self.ctc)
                except torch.cuda.OutOfMemoryError:
                    self._log({"type": "oom", "epoch": epoch, "step": self.state.step,
                               "batch_shape": list(batch["images"].shape)})  # fmt: skip
                    raise
                if torch.isfinite(loss):
                    self.scaler.scale(loss / accum).backward()
                    lines += len(batch["keys"])
                    value = float(loss.detach())
                    st.epoch_loss_sum += value
                    st.epoch_loss_n += 1
                    step_sum += value
                    step_n += 1
                    group += batch["keys"]
                elif cfg.diagnostics.stop_on_nan:
                    self._non_finite(batch["keys"], "loss", float(loss))
                else:
                    self._log({"type": "nonfinite_skipped", "epoch": epoch, "step": st.step,
                               "loss": str(float(loss)), "keys": batch["keys"]})  # fmt: skip
                # A skipped batch counts too: the sampler resumes by position, and the epoch's
                # last optimizer step is triggered by the count.
                st.batch_in_epoch += 1
                last_of_epoch = st.batch_in_epoch == self.epoch_batches[epoch]
                if st.batch_in_epoch % accum and not last_of_epoch:
                    continue
                norm = self._optimizer_step(group)
                group = []
                self._flag(self.gradmon.update(norm, epoch, st.step))
                if st.step % cfg.logging.log_every_n_steps == 0:
                    elapsed = time.monotonic() - t0
                    rec = {"type": "step", "epoch": epoch, "step": st.step,
                           "loss": st.epoch_loss_sum / max(st.epoch_loss_n, 1),
                           "step_loss": step_sum / max(step_n, 1), "grad_norm": norm,
                           "lr": self.optimizer.param_groups[0]["lr"],
                           "lines_per_s": lines / max(elapsed, 1e-9)}  # fmt: skip
                    if self.device.type == "cuda":
                        rec["vram_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
                    self._log(rec)
                step_sum, step_n = 0.0, 0
                every = cfg.checkpoint.every_steps
                if every and st.step % every == 0 and not last_of_epoch:
                    self._tick()
                    self.save("last.pt")
                if t.max_steps is not None and st.step >= t.max_steps:
                    done = True
                    break
            self._tick()
            if done and st.batch_in_epoch < self.epoch_batches[epoch]:
                self.save("last.pt")  # cut short mid-epoch: resumable from here
                break
            self._end_epoch(st.epoch_loss_sum / max(st.epoch_loss_n, 1))
        summary = self._summary()
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                                   encoding="utf-8")  # fmt: skip
        return summary

    def _end_epoch(self, train_loss: float) -> None:
        cfg, epoch, step = self.cfg, self.state.epoch, self.state.step
        val = evaluate(self.model, self._eval_loader(self.val_ds), self.charset, self.amp,
                       self.device, self.ctc)  # fmt: skip
        tr = evaluate(self.model, self._eval_loader(self.train_eval_ds), self.charset, self.amp,
                      self.device, self.ctc)  # fmt: skip
        point = {"type": "epoch", "epoch": epoch, "step": step, "train_loss": train_loss,
                 "train_eval_loss": tr.loss, "train_cer": tr.cer, "val_loss": val.loss,
                 "val_cer": val.cer, "val_wer": val.wer, "val_ned": val.ned,
                 "val_sequence_accuracy": val.sequence_accuracy,
                 "val_cer_normalised": val.cer_normalised, "blank_ratio": val.blank_ratio,
                 "empty_ratio": val.empty_ratio, "identical_ratio": val.identical_ratio,
                 "lr": self.optimizer.param_groups[0]["lr"],
                 "train_seconds": self.state.train_seconds}  # fmt: skip
        if self.device.type == "cuda":
            point["vram_peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
            torch.cuda.reset_peak_memory_stats()
        self.state.history.append(point)
        self._log(point)
        hist = [EpochPoint(p["epoch"], p["step"], p["train_eval_loss"], p["val_loss"],
                           p["train_cer"], p["val_cer"]) for p in self.state.history]  # fmt: skip
        d = cfg.diagnostics
        self._flag(check_overfitting(hist, d))
        self._flag(check_underfitting(hist, d, step / max(self.total_steps, 1)))
        for f in check_collapse(val.blank_ratio, val.empty_ratio, val.identical_ratio, d,
                                epoch, step):  # fmt: skip
            self._flag(f)

        metric = val.cer  # early_stopping.monitor is validated to be a validation metric
        improved = record_validation(self.state, metric, epoch, cfg.early_stopping)
        self.state.epoch += 1
        self.state.batch_in_epoch = 0
        self.state.epoch_loss_sum, self.state.epoch_loss_n = 0.0, 0
        if improved:
            self.save("best.pt", metric)
            self._write_samples(val.samples)
        if (epoch + 1) % cfg.checkpoint.checkpoint_interval == 0:
            self.save(f"epoch_{epoch:03d}.pt", metric)
            prune_epoch_checkpoints(self.ckpt_dir, cfg.checkpoint.keep_last)
        self.save("last.pt", metric)

    def _write_samples(self, samples: list[dict[str, Any]]) -> None:
        """Per-line validation results of the best checkpoint, for the report and the CI."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with (self.run_dir / "val_best_samples.jsonl").open("w", encoding="utf-8",
                                                           newline="\n") as fh:  # fmt: skip
            for s in samples:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    def _summary(self) -> dict[str, Any]:
        st = self.state
        last = st.history[-1] if st.history else {}
        best = next((p for p in st.history if p["epoch"] == st.best_epoch), {})
        return {
            "experiment_id": self.cfg.experiment_id,
            "config_hash": run_config_hash(self.cfg),
            "git": self.git.to_dict(),
            "parameters": self.params,
            "model_bytes": model_bytes(self.model.state_dict()),
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name() if self.device.type == "cuda" else None,
            "precision": self.amp.name,
            "torch": torch.__version__,
            "epochs_run": st.epoch,
            "steps": st.step,
            "stopped_early": st.stopped_early,
            "best_epoch": st.best_epoch,
            "best": best,
            "final": last,
            "flags": st.flags,
            "train_seconds": st.train_seconds,
            "train_lines": len(self.train_ds),
            "val_lines": len(self.val_ds),
            "data": _pack_versions(self.data_root, [*self.cfg.data.train, *self.cfg.data.val]),
        }


def score_samples(samples: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute per-line strict CER (used by the report for the worst-predicted lines)."""
    out = []
    for s in samples:
        sc = score_pair(s["pred"], s["label"])
        out.append({**s, "cer": sc.char_errors / sc.ref_chars if sc.ref_chars else 0.0})
    return out
