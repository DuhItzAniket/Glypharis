"""Task-agnostic training loop: everything a trainer does that does not depend on what it trains.

`TrainerCore` runs epochs of optimizer steps, validates at the end of each epoch, keeps the
early-stopping and checkpoint bookkeeping and resumes exactly. A task subclass supplies the batches
of an epoch (`train_batches`), the loss of one batch (`train_step`), one validation pass
(`validate`), and whatever task-specific data its checkpoints and summary carry.

The semantics are Phase 6's (`glypharis.training.rec_trainer`), so the recognizer can move onto this
core later without changing a result:

- a step is an optimizer step. With gradient accumulation it spans several batches, and the last
  batch of an epoch always closes a step. The schedule steps once per optimizer step and is sized
  from the whole epoch budget, so a run cut short by ``train.max_steps`` follows the full schedule;
- checkpoints are written only at step boundaries. They hold the model, optimizer, scheduler,
  scaler, the loop state (position inside the epoch, the epoch's running loss, task running sums,
  training time), the gradient monitor and every global RNG state, restored on the CPU. A resumed
  run is bit-identical on the CPU when the subclass's batches are pure functions of
  (seed, epoch, index);
- ``train.deterministic``, ``diagnostics.stop_on_nan`` and ``checkpoint.checkpoint_interval`` mean
  what they mean in Phase 6.

Additions: early stopping follows ``early_stopping.mode`` (the detector maximises F1); the wait for
each batch is timed; every GPU phase of a batch (forward, backward, optimizer) and each validation
pass logs an ``oom`` record before an out-of-memory error propagates; `close` is idempotent.

Deliberate differences from Phase 6:

- with ``stop_on_nan`` off, a skipped non-finite batch still advances ``batch_in_epoch``, so the
  saved position is always the number of batches consumed. Phase 6 does not advance it; no
  existing run or test skips a batch;
- without a gradient scaler, a non-finite gradient norm is caught before the optimizer step, like a
  non-finite loss: with ``stop_on_nan`` the run stops with the step's batch keys and the untouched
  weights saved; otherwise the update is skipped (the schedule and step count still advance, as
  when the fp16 scaler skips a step). Phase 6 steps on it, which turns every weight into NaN and
  blames the next batch;
- a fresh (not resumed) run moves an existing ``metrics.jsonl`` aside to
  ``metrics.attempt<k>.jsonl`` before it logs, so a restart under the same experiment ID never
  mixes two attempts' records.
"""

from __future__ import annotations

import abc
import gc
import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Final, NoReturn

import torch
from torch import nn

from glypharis.config.loader import config_hash, config_to_dict, save_resolved
from glypharis.config.schema import ConfigError, EarlyStoppingConfig, RunConfig
from glypharis.diagnostics import Flag, GradientMonitor, NonFiniteLoss
from glypharis.training.checkpoint import (
    load_checkpoint,
    prune_epoch_checkpoints,
    rng_state,
    save_checkpoint,
    set_rng_state,
)
from glypharis.training.optim import make_optimizer, make_scheduler
from glypharis.utils.device import resolve_amp, select_device
from glypharis.utils.gitinfo import git_info
from glypharis.utils.monitor import snapshot
from glypharis.utils.seed import seed_everything

log = logging.getLogger(__name__)
CHECKPOINT_FORMAT: Final = 1

# Running sums the core keeps in LoopState.extra for the current epoch; reset when it ends.
WAIT_S: Final = "data_wait_s"  # seconds spent waiting for the next batch
EPOCH_S: Final = "epoch_train_s"  # training seconds of this epoch (validation excluded)
SAMPLES: Final = "samples"
BATCHES: Final = "batches"
OPT_STEPS: Final = "optimizer_steps"
CLIPPED: Final = "clipped_steps"
_MIB: Final = 2**20


@dataclass
class LoopState:
    epoch: int = 0
    batch_in_epoch: int = 0  # batches of the current epoch already consumed, skipped ones included
    step: int = 0  # optimizer steps
    best_metric: float = math.inf  # best validation metric so far: selects best.pt
    best_epoch: int = -1
    patience_ref: float = math.inf  # what early stopping measures progress against
    bad_epochs: int = 0
    stopped_early: bool = False
    history: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)
    train_seconds: float = 0.0
    epoch_loss_sum: float = 0.0  # running training loss of the current epoch
    epoch_loss_n: int = 0
    extra: dict[str, float] = field(
        default_factory=dict
    )  # task running sums, resumed with the state


def record_validation(
    state: LoopState, metric: float, epoch: int, es: EarlyStoppingConfig, maximize: bool
) -> bool:
    """Update checkpoint selection and early stopping with one epoch's validation metric.

    The selected checkpoint is the one with the strictly best validation metric (protocol §7;
    ties keep the earlier epoch). ``min_delta`` only decides whether an epoch counts as progress
    for the patience counter. Returns True when this epoch is the new best.
    """
    s = -1.0 if maximize else 1.0  # compare as "lower is better"
    best = s * metric < s * state.best_metric
    if best:
        state.best_metric, state.best_epoch = metric, epoch
    if s * metric < s * state.patience_ref - es.min_delta:
        state.patience_ref, state.bad_epochs = metric, 0
    else:
        state.bad_epochs += 1
        if es.enabled and state.bad_epochs >= es.patience:
            state.stopped_early = True
    return best


@dataclass
class StepResult:
    loss: torch.Tensor  # scalar fp32 with graph
    samples: int
    keys: list[str]  # batch ids, written to the log if the loss is not finite
    logs: dict[str, torch.Tensor]  # detached scalars, synced once per batch together with the loss


@dataclass
class EpochResult:
    point: dict[str, Any]  # logged as {"type": "epoch", **point}
    metric: float  # the monitored validation metric
    flags: list[Flag]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Records of a JSON-lines file; an empty list when the file does not exist."""
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def free_cuda() -> None:
    """Collect garbage, then return cached CUDA blocks to the driver if CUDA was ever used."""
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _batch_shape(batch: Mapping[str, Any] | None) -> list[int] | None:
    if batch is None:
        return None
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return list(value.shape)
    return None


class TrainerCore(abc.ABC):
    """The shared loop. Subclasses call `setup_optimization` once their model exists."""

    task: ClassVar[str]
    throughput_key: ClassVar[str] = "samples_per_s"

    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None
    scaler: torch.amp.GradScaler
    epoch_batches: list[int]
    steps_per_epoch: list[int]
    total_steps: int

    def __init__(
        self, cfg: RunConfig, on_step: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        cfg.require_run_identity()
        if cfg.task != self.task:
            raise ConfigError(f"{type(self).__name__} needs task {self.task!r}, got {cfg.task!r}")
        assert cfg.experiment_id is not None
        self.cfg = cfg
        self.on_step = on_step
        self.run_dir = Path(cfg.paths.runs_dir) / cfg.experiment_id
        self.ckpt_dir = Path(cfg.checkpoint.dir)
        self.report_dir = Path(cfg.paths.reports_dir) / "training" / cfg.experiment_id
        self.data_root = Path(cfg.paths.data_root)

        seed_everything(cfg.seed, cfg.train.deterministic)
        self.device = select_device(cfg.device)
        self.amp = resolve_amp(cfg.train.mixed_precision, self.device)
        self.gradmon = GradientMonitor(cfg.diagnostics)
        self.maximize = cfg.early_stopping.mode == "max"
        worst = -math.inf if self.maximize else math.inf
        self.state = LoopState(best_metric=worst, patience_ref=worst)
        self.git = git_info()
        self._clock = time.perf_counter()
        self._closed = False
        self._fresh_log = not cfg.checkpoint.resume  # move an earlier attempt's log aside first

    def setup_optimization(self, model: nn.Module, epoch_batches: Sequence[int]) -> None:
        """Optimizer, schedule (sized from every epoch's batch count) and gradient scaler."""
        t = self.cfg.train
        if len(epoch_batches) != t.epochs:
            raise ValueError(f"{len(epoch_batches)} epoch batch counts for {t.epochs} epochs")
        if min(epoch_batches) < 1:
            raise ConfigError("an epoch has no full training batch; lower train.batch_size")
        self.model = model
        self.optimizer = make_optimizer(self.cfg, model)
        self.epoch_batches = list(epoch_batches)
        self.steps_per_epoch = [math.ceil(n / t.gradient_accumulation) for n in self.epoch_batches]
        self.total_steps = sum(self.steps_per_epoch)
        self.scheduler = make_scheduler(self.cfg, self.optimizer, self.total_steps)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp.needs_grad_scaler)

    # ----------------------------------------------------------------------- task interface

    @abc.abstractmethod
    def train_batches(self, epoch: int, start_batch: int) -> Iterable[dict[str, Any]]:
        """The batches of ``epoch`` from ``start_batch`` on, in a fixed order."""

    @abc.abstractmethod
    def train_step(self, batch: dict[str, Any]) -> StepResult:
        """Forward pass and loss of one batch (no backward)."""

    @abc.abstractmethod
    def validate(self) -> EpochResult:
        """One validation pass at the end of an epoch."""

    def payload_extra(self) -> dict[str, Any]:
        """Task data stored in every checkpoint next to the shared payload."""
        return {}

    def step_extra(self) -> dict[str, Any]:
        """Task fields added to each logged step record."""
        return {}

    def on_best(self) -> None:  # noqa: B027 - an optional hook, not an abstract method
        """Called after best.pt is written."""

    # --------------------------------------------------------------------------- the loop

    def fit(self) -> dict[str, Any]:
        cfg, t, st = self.cfg, self.cfg.train, self.state
        if cfg.checkpoint.resume:
            self.resume(cfg.checkpoint.resume)
            st = self.state
        else:
            save_resolved(cfg, self.run_dir / "config.resolved.yaml")
        self.model.train()
        done = False
        while st.epoch < t.epochs and not done and not st.stopped_early:
            done = self._train_epoch()
            if done and st.batch_in_epoch < self.epoch_batches[st.epoch]:
                self.save("last.pt")  # cut short mid-epoch: resumable from here
                break
            self._end_epoch()
        summary = self.summary()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, indent=1) + "\n", encoding="utf-8"
        )
        return summary

    def _sync(self, out: StepResult) -> dict[str, float]:
        """The loss and the logged parts as floats, with one host sync."""
        names = ["loss", *out.logs]
        values = [out.loss.detach().float().reshape(())]
        values += [v.detach().float().reshape(()) for v in out.logs.values()]
        return dict(zip(names, torch.stack(values).cpu().tolist(), strict=True))

    def _train_epoch(self) -> bool:
        """Train the current epoch from the saved position; True once train.max_steps is reached."""
        cfg, t, st = self.cfg, self.cfg.train, self.state
        epoch, accum = st.epoch, t.gradient_accumulation
        n_batches = self.epoch_batches[epoch]
        x = st.extra
        for key in (WAIT_S, EPOCH_S, SAMPLES, BATCHES, OPT_STEPS, CLIPPED):
            x.setdefault(key, 0.0)
        batches = iter(self.train_batches(epoch, st.batch_in_epoch))
        # Throughput and wait share of this stretch of training; the epoch's totals live in the
        # state so that a resumed epoch continues them.
        seg_start, seg_samples, seg_wait = time.perf_counter(), 0, 0.0
        group: dict[str, float] = {}
        group_n = 0
        group_keys: list[str] = []  # ids of the batches whose gradients the next step applies
        self._clock = time.perf_counter()
        while True:
            w0 = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                break
            wait = time.perf_counter() - w0
            x[WAIT_S] += wait
            seg_wait += wait
            phase = "forward"
            try:
                out = self.train_step(batch)
                values = self._sync(out)
                if not math.isfinite(values["loss"]):
                    if cfg.diagnostics.stop_on_nan:
                        self._non_finite(out.keys, "loss", values["loss"])
                    st.batch_in_epoch += 1
                    self._log(
                        {
                            "type": "nonfinite_skipped",
                            "epoch": epoch,
                            "step": st.step,
                            "batch_in_epoch": st.batch_in_epoch,
                            "loss": str(values["loss"]),
                            "keys": out.keys,
                        }
                    )
                    continue
                phase = "backward"
                self.scaler.scale(out.loss / accum).backward()
                st.batch_in_epoch += 1
                st.epoch_loss_sum += values["loss"]
                st.epoch_loss_n += 1
                for key, value in values.items():
                    group[key] = group.get(key, 0.0) + value
                    if key != "loss":
                        x[key] = x.get(key, 0.0) + value
                group_n += 1
                group_keys += out.keys
                x[SAMPLES] += out.samples
                x[BATCHES] += 1
                seg_samples += out.samples
                last_of_epoch = st.batch_in_epoch == n_batches
                if st.batch_in_epoch % accum and not last_of_epoch:
                    continue
                phase = "optimizer"
                norm = self._optimizer_step(group_keys)
            except torch.cuda.OutOfMemoryError:
                self._log_oom(phase, batch)
                raise
            if norm is not None:
                clip = t.grad_clip_norm
                x[OPT_STEPS] += 1
                x[CLIPPED] += float(clip is not None and norm > clip)
                self._flag(self.gradmon.update(norm, epoch, st.step))
            if norm is not None and st.step % cfg.logging.log_every_n_steps == 0:
                elapsed = max(time.perf_counter() - seg_start, 1e-9)
                rec: dict[str, Any] = {
                    "type": "step",
                    "epoch": epoch,
                    "step": st.step,
                    "loss": st.epoch_loss_sum / max(st.epoch_loss_n, 1),
                    "step_loss": group["loss"] / group_n,
                    **{k: v / group_n for k, v in group.items() if k != "loss"},
                    "grad_norm": norm,
                    "clipped_share": x[CLIPPED] / max(x[OPT_STEPS], 1.0),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    self.throughput_key: seg_samples / elapsed,
                    "data_wait_share": seg_wait / elapsed,
                }
                rec.update(self._resources())
                rec.update(self.step_extra())
                self._log(rec)
            group, group_n, group_keys = {}, 0, []
            every = cfg.checkpoint.every_steps
            if every and st.step % every == 0 and not last_of_epoch:
                self._tick()
                self.save("last.pt")
            if t.max_steps is not None and st.step >= t.max_steps:
                self._tick()
                return True
        self._tick()
        return False

    def _resources(self) -> dict[str, Any]:
        if self.device.type != "cuda":
            return {}
        out: dict[str, Any] = {
            "vram_peak_alloc_mib": torch.cuda.max_memory_allocated(self.device) / _MIB,
            "vram_peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / _MIB,
        }
        if self.cfg.logging.monitor_resources:
            snap = snapshot(self.device)
            out.update(
                gpu_temperature_c=snap.gpu_temperature_c,
                gpu_sm_clock_mhz=snap.gpu_sm_clock_mhz,
                gpu_utilization_pct=snap.gpu_utilization_pct,
            )
        return out

    def _end_epoch(self) -> None:
        cfg, st = self.cfg, self.state
        epoch, step, x = st.epoch, st.step, st.extra
        try:
            res = self.validate()
        except torch.cuda.OutOfMemoryError:
            self._log_oom("validate", None)
            raise
        point: dict[str, Any] = {
            "type": "epoch",
            "epoch": epoch,
            "step": step,
            "train_loss": st.epoch_loss_sum / max(st.epoch_loss_n, 1),
            "lr": self.optimizer.param_groups[0]["lr"],
            "train_seconds": st.train_seconds,
            "epoch_train_seconds": x.get(EPOCH_S, 0.0),
            "data_wait_share": x.get(WAIT_S, 0.0) / max(x.get(EPOCH_S, 0.0), 1e-9),
            **res.point,
        }
        st.history.append(point)
        self._log(point)
        for flag in res.flags:
            self._flag(flag)

        metric = res.metric
        improved = record_validation(st, metric, epoch, cfg.early_stopping, self.maximize)
        st.epoch += 1
        st.batch_in_epoch = 0
        st.epoch_loss_sum, st.epoch_loss_n = 0.0, 0
        st.extra = {}
        if improved:
            self.save("best.pt", metric)
            self.on_best()
        if (epoch + 1) % cfg.checkpoint.checkpoint_interval == 0:
            self.save(f"epoch_{epoch:03d}.pt", metric)
            prune_epoch_checkpoints(self.ckpt_dir, cfg.checkpoint.keep_last)
        self.save("last.pt", metric)

    def _tick(self) -> None:
        """Add the time since the last tick to the state (saved with every checkpoint)."""
        now = time.perf_counter()
        self.state.train_seconds += now - self._clock
        self.state.extra[EPOCH_S] = self.state.extra.get(EPOCH_S, 0.0) + now - self._clock
        self._clock = now

    def _optimizer_step(self, keys: list[str]) -> float | None:
        """Unscale (fp16 only), clip, step, advance the schedule; returns the pre-clip norm.

        Returns None when the norm was not finite and the update was skipped. ``keys`` are the ids
        of the batches whose gradients this step applies.
        """
        if self.amp.needs_grad_scaler:
            self.scaler.unscale_(self.optimizer)
        clip = self.cfg.train.grad_clip_norm
        norm = float(
            nn.utils.clip_grad_norm_(
                self.model.parameters(), clip if clip is not None else math.inf
            )
        )
        stepped = math.isfinite(norm) or self.amp.needs_grad_scaler
        if stepped:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # A finite loss can still back-propagate to an inf or NaN gradient (a bf16 overflow, a
            # 0/0 in a backward). Stepping on it would turn every weight into NaN without a flag,
            # and the next batch's loss would take the blame. The fp16 scaler skips such steps.
            if self.cfg.diagnostics.stop_on_nan:
                self._non_finite(keys, "grad_norm", norm)  # the weights are still untouched
            self._log(
                {
                    "type": "nonfinite_grad_skipped",
                    "epoch": self.state.epoch,
                    "step": self.state.step,
                    "batch_in_epoch": self.state.batch_in_epoch,
                    "grad_norm": str(norm),
                    "keys": keys,
                }
            )
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.state.step += 1
        return norm if stepped else None

    def _non_finite(self, keys: Sequence[str], what: str, value: float) -> NoReturn:
        """Save nonfinite.pt and the batch keys, then stop the run; ``what`` names the value."""
        self.save("nonfinite.pt")
        self._log(
            {
                "type": "nonfinite",
                "epoch": self.state.epoch,
                "step": self.state.step,
                what: str(value),
                "keys": list(keys),
            }
        )
        raise NonFiniteLoss(
            f"non-finite {what} {value} at step {self.state.step}; batch keys and state saved"
        )

    def _log_oom(self, phase: str, batch: Mapping[str, Any] | None) -> None:
        cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self._log(
            {
                "type": "oom",
                "phase": phase,
                "epoch": self.state.epoch,
                "step": self.state.step,
                "batch_in_epoch": self.state.batch_in_epoch,
                "batch_shape": _batch_shape(batch),
                "vram_alloc_mib": torch.cuda.memory_allocated(self.device) / _MIB if cuda else None,
                "vram_reserved_mib": torch.cuda.memory_reserved(self.device) / _MIB
                if cuda
                else None,
            }
        )

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
            "config_hash": config_hash(self.cfg),
            "git": self.git.to_dict(),
            "rng": rng_state(),
            **self.payload_extra(),
        }

    def save(self, name: str, metric: float | None = None) -> Path:
        return save_checkpoint(self.ckpt_dir / name, self._payload(metric))

    def resume(self, path: str | Path) -> None:
        ck = load_checkpoint(path, self.device)
        if ck.get("config_hash") != config_hash(self.cfg):
            # Only the resume pointer and the step cap may differ between the saved and the
            # current config.
            saved, now = dict(ck["config"]), config_to_dict(self.cfg)
            for d in (saved, now):
                d["checkpoint"] = {**d["checkpoint"], "resume": None}
                d["train"] = {**d["train"], "max_steps": None}
            if saved != now:
                raise ConfigError(f"{path} was written with a different config; refusing to resume")
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if self.scheduler and ck["scheduler"]:
            self.scheduler.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.state = LoopState(**ck["state"])
        self.gradmon.load_state_dict(ck["gradmon"])
        set_rng_state(ck["rng"])
        log.info(
            "resumed from %s at epoch %d, batch %d, step %d",
            path,
            self.state.epoch,
            self.state.batch_in_epoch,
            self.state.step,
        )

    # --------------------------------------------------------------------------- logging

    def _log(self, record: dict[str, Any]) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / "metrics.jsonl"
        if self._fresh_log:
            # Only once something is logged, so a run refused at start-up changes nothing.
            self._fresh_log = False
            if path.exists():
                k = 1
                while (self.run_dir / f"metrics.attempt{k}.jsonl").exists():
                    k += 1
                path.replace(self.run_dir / f"metrics.attempt{k}.jsonl")
                log.warning(
                    "fresh start: the earlier metrics.jsonl is now metrics.attempt%d.jsonl", k
                )
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record) + "\n")
        if self.on_step is not None:
            self.on_step(record)

    def _flag(self, flag: Flag | None) -> None:
        if flag is not None:
            self.state.flags.append(flag.to_dict())
            log.warning(
                "%s at epoch %d step %d: %s", flag.name, flag.epoch, flag.step, flag.evidence
            )
            self._log({"type": "flag", **flag.to_dict()})

    def summary(self) -> dict[str, Any]:
        st, cfg = self.state, self.cfg
        last = st.history[-1] if st.history else {}
        best = next((p for p in st.history if p["epoch"] == st.best_epoch), {})
        return {
            "experiment_id": cfg.experiment_id,
            "task": cfg.task,
            "config_hash": config_hash(cfg),
            "git": self.git.to_dict(),
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "precision": self.amp.name,
            "deterministic": cfg.train.deterministic,
            "torch": torch.__version__,
            "epochs_run": st.epoch,
            "steps": st.step,
            "total_steps": self.total_steps,
            "stopped_early": st.stopped_early,
            "best_epoch": st.best_epoch,
            "best": best,
            "final": last,
            "flags": st.flags,
            "train_seconds": st.train_seconds,
        }

    def close(self) -> None:
        """Release workers and pools. Idempotent; subclasses shut their loaders here."""
        self._closed = True
