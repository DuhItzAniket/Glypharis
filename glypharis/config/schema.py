"""Typed configuration schema.

Every hyperparameter is a field here; code reads values from the validated object, never from raw
dicts. Unknown keys and wrong types are errors (see `glypharis.config.loader`). The defaults are
placeholders that make a config loadable, not tuned values.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from typing import Literal

EXPERIMENT_ID_RE = re.compile(r"^GW-(BASE|REC|DET|E2E|DATA|EFF)-\d{3}$")


class ConfigError(ValueError):
    """Raised for any invalid configuration: unknown key, wrong type, or bad value."""


@dataclass
class PathsConfig:
    data_root: str = "data"
    reports_dir: str = "reports"
    runs_dir: str = "runs"


@dataclass
class LoggingConfig:
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_every_n_steps: int = 50
    monitor_resources: bool = True


@dataclass
class DataConfig:
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    dataset_subset: int | None = None
    charset: str | None = None
    image_height: int = 32
    image_width: int | None = None
    max_width: int = 1024
    num_workers: int = 4
    augmentation: str = "none"


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    gradient_accumulation: int = 1
    optimizer: Literal["adamw", "adam", "sgd"] = "adamw"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    scheduler: Literal["onecycle", "cosine", "step", "plateau", "none"] = "onecycle"
    warmup_steps: int = 0
    grad_clip_norm: float | None = 5.0
    mixed_precision: Literal["bf16", "fp16", "none"] = "bf16"
    deterministic: bool = True


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints/${experiment_id}"
    checkpoint_interval: int = 1
    keep_last: int = 3
    resume: str | None = None


@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    monitor: str = "val_cer"
    mode: Literal["min", "max"] = "min"
    patience: int = 8
    min_delta: float = 0.001


@dataclass
class DiagnosticsConfig:
    overfit_window: int = 5
    underfit_cer_threshold: float = 0.5
    blank_ratio_alarm: float = 0.95
    grad_norm_alarm: float = 1.0e3
    stop_on_nan: bool = True


@dataclass
class RunConfig:
    experiment_id: str | None = None
    task: Literal["rec", "det", "e2e"] | None = None
    model_name: str | None = None
    seed: int = 1337
    device: Literal["auto", "cuda", "cpu"] = "auto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    def validate(self) -> None:
        """Value checks that types alone cannot express. Raises ConfigError."""
        errors: list[str] = []

        def check(ok: bool, msg: str) -> None:
            if not ok:
                errors.append(msg)

        if self.experiment_id is not None:
            check(
                bool(EXPERIMENT_ID_RE.match(self.experiment_id)),
                f"experiment_id {self.experiment_id!r} does not match GW-<SCOPE>-NNN",
            )
        check(self.seed >= 0, "seed must be >= 0")

        d, t, c, e, g = (
            self.data,
            self.train,
            self.checkpoint,
            self.early_stopping,
            self.diagnostics,
        )
        check(d.image_height > 0, "data.image_height must be > 0")
        check(d.image_width is None or d.image_width > 0, "data.image_width must be > 0 or null")
        check(d.max_width > 0, "data.max_width must be > 0")
        check(d.num_workers >= 0, "data.num_workers must be >= 0")
        check(d.dataset_subset is None or d.dataset_subset > 0, "data.dataset_subset must be > 0")
        check(t.epochs > 0, "train.epochs must be > 0")
        check(t.batch_size > 0, "train.batch_size must be > 0")
        check(t.gradient_accumulation > 0, "train.gradient_accumulation must be > 0")
        check(t.learning_rate > 0, "train.learning_rate must be > 0")
        check(t.weight_decay >= 0, "train.weight_decay must be >= 0")
        check(t.warmup_steps >= 0, "train.warmup_steps must be >= 0")
        check(t.grad_clip_norm is None or t.grad_clip_norm > 0, "train.grad_clip_norm must be > 0")
        check(c.checkpoint_interval > 0, "checkpoint.checkpoint_interval must be > 0")
        check(c.keep_last > 0, "checkpoint.keep_last must be > 0")
        # Early stopping and checkpoint selection must never follow a training metric (brief §28).
        check(e.monitor.startswith("val_"), "early_stopping.monitor must be a validation metric")
        check(e.patience > 0, "early_stopping.patience must be > 0")
        check(e.min_delta >= 0, "early_stopping.min_delta must be >= 0")
        check(g.overfit_window >= 2, "diagnostics.overfit_window must be >= 2")
        check(g.underfit_cer_threshold > 0, "diagnostics.underfit_cer_threshold must be > 0")
        check(0 < g.blank_ratio_alarm <= 1, "diagnostics.blank_ratio_alarm must be in (0, 1]")
        check(g.grad_norm_alarm > 0, "diagnostics.grad_norm_alarm must be > 0")
        check(self.logging.log_every_n_steps > 0, "logging.log_every_n_steps must be > 0")

        if errors:
            raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))

    def require_run_identity(self) -> None:
        """Training and evaluation entry points need an experiment ID and a task."""
        missing = [k for k in ("experiment_id", "task", "model_name") if getattr(self, k) is None]
        if missing:
            raise ConfigError(f"config is missing required run fields: {', '.join(missing)}")
        leftovers = [s for s in _string_values(self) if "${" in s]
        if leftovers:
            raise ConfigError(f"unresolved references in config: {leftovers}")


def _string_values(obj: object) -> list[str]:
    """All string values in a (nested) config dataclass."""
    out: list[str] = []
    for f in dataclasses.fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, f.name)
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, list):
            out.extend(v for v in value if isinstance(v, str))
        elif dataclasses.is_dataclass(value):
            out.extend(_string_values(value))
    return out
