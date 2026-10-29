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
    augmentation: str = "none"  # preset name, see glypharis.data.augment.PRESETS
    # Per-transform overrides, e.g. {"gaussian_blur": {"p": 0.5, "severity": 0.8}}
    augmentation_params: dict[str, dict[str, float]] = field(default_factory=dict)
    # Cap on lines taken from a training pack (pack prefix -> count), a fixed seeded subset.
    train_limits: dict[str, int] = field(default_factory=dict)
    # Lines per epoch, drawn afresh each epoch from all training lines (no line twice in an epoch).
    # Keeps the step budget fixed when the training pool grows (Phase 12). null: every line.
    epoch_lines: int | None = None
    val_subset: int | None = None  # seeded subset of validation lines scored each epoch
    train_eval_subset: int = 2000  # training lines scored each epoch (over/underfitting checks)
    bucket_chunk: int = 50  # batches per width-sorted chunk (bucketing)
    # Cap on padded columns per batch (lines x widest line); bounds GPU memory. null: no cap.
    max_batch_columns: int | None = None
    # Count the CTC padding (stride x frames a label needs) in the widths that batches are split
    # by, so the column cap holds for lines denser than their image (Phase 7, GW-REC-021). Off,
    # and left out of the config hash, by default: earlier runs keep their batches.
    ctc_aware_batching: bool = False


@dataclass
class SvtrConfig:
    """GLYPHARIS-Rec-M: an SVTR-style encoder (ARCHITECTURE §4; Phase 7, H6). Three stages at
    falling height; each stage runs its local mixing blocks, then its global ones."""

    dims: list[int] = field(default_factory=lambda: [96, 192, 256])
    local_blocks: list[int] = field(default_factory=lambda: [3, 3, 0])
    global_blocks: list[int] = field(default_factory=lambda: [0, 3, 6])
    heads: list[int] = field(default_factory=lambda: [3, 6, 8])
    mlp_ratio: float = 4.0
    local_kernel: list[int] = field(default_factory=lambda: [3, 7])  # height, width


@dataclass
class ModelConfig:
    cnn_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 256, 512, 512])
    lstm_hidden: int = 256
    lstm_layers: int = 2
    dropout: float = 0.1
    # Added in Phase 7. At their defaults they are left out of the hashed config, so earlier
    # configs keep their hashes (glypharis.config.loader.config_to_dict).
    frame_stride: int = 4  # input columns per output frame (CRNN: 4 or 8)
    svtr: SvtrConfig | None = None  # required when model_name is "svtr"


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
    max_steps: int | None = None  # stop after this many optimizer steps (sanity runs, smoke test)


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints/${experiment_id}"
    checkpoint_interval: int = 1
    keep_last: int = 3
    resume: str | None = None
    every_steps: int | None = None  # also refresh last.pt every N optimizer steps


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


# ------------------------------------------------------------------------- detector (Phase 8)
# Everything the detector reads lives under one optional top-level `det:` section, so recognizer
# configs, and their hashes, never see these keys. The defaults equal the GW-DET-001 values.

OP_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class DetDataConfig:
    train_pages: list[str] = field(default_factory=list)  # page files under paths.data_root
    val_pages: list[str] = field(default_factory=list)
    train_subset: int | None = None  # seeded page subset (sanity / resume check only)
    val_subset: int | None = 800  # validation pages scored each epoch (seeded); null = all
    train_eval_subset: int = 200  # training pages scored like validation each epoch
    crops_per_page: int = 4
    crop_size: int = 640
    scale_min: float = 0.5  # log-uniform jitter around the inference scale
    scale_max: float = 2.0
    rotate_max_deg: float = 10.0
    text_crop_p: float = 0.8
    reduced_decode: bool = True
    num_workers: int = 6
    eval_workers: int = 3


@dataclass
class DetAugmentConfig:  # photometric only; geometry lives in DetDataConfig
    color_p: float = 0.5
    color_strength: float = 0.3  # brightness / contrast / saturation factors in [1-s, 1+s]
    grayscale_p: float = 0.1
    blur_p: float = 0.15
    blur_sigma_max: float = 1.5  # sigma ~ U(0.3, max) px
    noise_p: float = 0.15
    noise_sigma_max: float = 8.0  # luminance noise sigma ~ U(2, max) grey levels
    jpeg_p: float = 0.15
    jpeg_quality_min: int = 40  # quality ~ U{min..95}


@dataclass
class DetLabelConfig:
    shrink_ratio: float = 0.4
    thresh_min: float = 0.3
    thresh_max: float = 0.7
    min_text_size: float = 6.0  # rotated short side (px, training scale); below -> don't care


@dataclass
class DetModelConfig:
    backbone: Literal["mobilenet_v3_large", "mobilenet_v3_small"] = "mobilenet_v3_large"
    init: Literal["imagenet", "scratch"] = "imagenet"
    weights: str | None = None  # relative to paths.data_root
    weights_sha256: str | None = None  # 64 lowercase hex, checked before torch.load
    inner_channels: int = 96
    channels_last: bool = True


@dataclass
class DetLossConfig:
    bce_weight: float = 1.0
    alpha: float = 1.0  # Dice on the approximate binary map
    beta: float = 10.0  # masked L1 on the threshold map
    ohem_ratio: float = 3.0
    k: float = 50.0  # binarisation amplification
    eps: float = 1.0e-6


@dataclass
class DetInferConfig:
    long_side: int = 1280  # validation, evaluation, Phase 10 inference


@dataclass
class DetOperatingPoint:
    # No defaults: the three values are the point. A missing key is a ConfigError from the
    # loader, never a TypeError.
    name: str
    unclip_mode: Literal["dbnet", "inverse"]
    unclip_ratio: float  # dbnet: D = ratio*A/L; inverse: multiplier on the exact inverse


@dataclass
class DetPostConfig:
    bin_thresh: float = 0.3
    box_thresh: float = 0.6
    min_component_area: int = 3
    min_box_side: float = 3.0  # px at network scale, after unclip
    max_candidates: int = 5000  # runtime guard only; truncations are counted
    box_type: Literal["quad", "poly"] = "quad"  # expanded rectangle, or offset contour
    operating_points: list[DetOperatingPoint] = field(
        default_factory=lambda: [
            DetOperatingPoint("dbnet15", "dbnet", 1.5),
            DetOperatingPoint("inverse10", "inverse", 1.0),
        ]
    )


@dataclass
class DetDiagnosticsConfig:
    underfit_error_threshold: float = 0.5  # 1 - F1@0.5, replaces underfit_cer_threshold for det
    collapse_low_ratio: float = 0.1  # predicted text px / GT kernel px
    collapse_high_ratio: float = 10.0
    text_fraction_alarm: float = 0.5  # predicted text px / valid px
    empty_page_alarm: float = 0.5  # share of care pages with no box
    thresh_out_of_range_alarm: float = 0.05  # share of thresh-mask px with T outside [0.2, 0.8]
    thresh_divergence_rise: float = 1.5
    box_size_gap: float = 0.25  # F1@0.3 - F1@0.5
    postprocess_drop_alarm: float = 0.5
    empty_crop_alarm: float = 0.35
    data_wait_alarm: float = 0.2
    slow_validation_share: float = 0.25  # validation seconds / the epoch's training seconds


# Detector diagnostics that are shares or F1 gaps, so (0, 1]; the others only need to be > 0.
_DET_DIAG_SHARES = frozenset(
    {
        "underfit_error_threshold",
        "text_fraction_alarm",
        "empty_page_alarm",
        "thresh_out_of_range_alarm",
        "box_size_gap",
        "postprocess_drop_alarm",
        "empty_crop_alarm",
        "data_wait_alarm",
        "slow_validation_share",
    }
)


@dataclass
class DetConfig:
    data: DetDataConfig = field(default_factory=DetDataConfig)
    augment: DetAugmentConfig = field(default_factory=DetAugmentConfig)
    labels: DetLabelConfig = field(default_factory=DetLabelConfig)
    model: DetModelConfig = field(default_factory=DetModelConfig)
    loss: DetLossConfig = field(default_factory=DetLossConfig)
    infer: DetInferConfig = field(default_factory=DetInferConfig)
    post: DetPostConfig = field(default_factory=DetPostConfig)
    diagnostics: DetDiagnosticsConfig = field(default_factory=DetDiagnosticsConfig)

    def errors(self) -> list[str]:
        """Value checks for the `det:` section, as messages (empty when valid)."""
        errors: list[str] = []

        def check(ok: bool, msg: str) -> None:
            if not ok:
                errors.append(msg)

        d, a, lb, m, ls, p = self.data, self.augment, self.labels, self.model, self.loss, self.post
        for key in ("train_pages", "val_pages"):
            pages: list[str] = getattr(d, key)
            # The trainer and every detector script read one file per role; a list leaves room
            # for more sources later without a format change.
            check(len(pages) == 1, f"det.data.{key} must name exactly one page file")
            check(all(s.endswith(".jsonl.gz") for s in pages), f"det.data.{key}: page files only")
            # Test splits reach the detector only through eval.access.open_test_access.
            check(
                not any("test" in s.lower() or "gdc" in s.lower() for s in pages),
                f"det.data.{key} must not name a test or GDC page file",
            )
        check(d.train_subset is None or d.train_subset > 0, "det.data.train_subset must be > 0")
        check(d.val_subset is None or d.val_subset > 0, "det.data.val_subset must be > 0")
        check(d.train_eval_subset >= 0, "det.data.train_eval_subset must be >= 0")
        check(d.crops_per_page >= 1, "det.data.crops_per_page must be >= 1")
        check(
            d.crop_size % 32 == 0 and d.crop_size >= 64,
            "det.data.crop_size must be a multiple of 32 and >= 64",
        )
        check(0 < d.scale_min <= d.scale_max, "det.data needs 0 < scale_min <= scale_max")
        check(0 <= d.rotate_max_deg <= 45, "det.data.rotate_max_deg must be in [0, 45]")
        check(0 <= d.text_crop_p <= 1, "det.data.text_crop_p must be in [0, 1]")
        check(d.num_workers >= 0, "det.data.num_workers must be >= 0")
        check(d.eval_workers >= 0, "det.data.eval_workers must be >= 0")

        for key in ("color_p", "grayscale_p", "blur_p", "noise_p", "jpeg_p"):
            check(0 <= getattr(a, key) <= 1, f"det.augment.{key} must be in [0, 1]")
        check(0 <= a.color_strength < 1, "det.augment.color_strength must be in [0, 1)")
        check(a.blur_sigma_max >= 0.3, "det.augment.blur_sigma_max must be >= 0.3")
        check(a.noise_sigma_max >= 2, "det.augment.noise_sigma_max must be >= 2")
        check(5 <= a.jpeg_quality_min <= 95, "det.augment.jpeg_quality_min must be in [5, 95]")

        check(0 < lb.shrink_ratio < 1, "det.labels.shrink_ratio must be in (0, 1)")
        check(
            0 <= lb.thresh_min < lb.thresh_max <= 1,
            "det.labels needs 0 <= thresh_min < thresh_max <= 1",
        )
        check(lb.min_text_size >= 0, "det.labels.min_text_size must be >= 0")

        if m.init == "imagenet":
            check(m.weights is not None, "det.model.weights is required for init imagenet")
            check(
                m.weights_sha256 is not None and bool(SHA256_RE.match(m.weights_sha256)),
                "det.model.weights_sha256 must be 64 lowercase hex characters for init imagenet",
            )
        else:
            check(
                m.weights is None and m.weights_sha256 is None,
                "det.model.weights and weights_sha256 must be null for init scratch",
            )
        check(
            m.inner_channels > 0 and m.inner_channels % 4 == 0,
            "det.model.inner_channels must be a positive multiple of 4",
        )

        for key in ("bce_weight", "alpha", "beta"):
            check(getattr(ls, key) >= 0, f"det.loss.{key} must be >= 0")
        check(ls.ohem_ratio > 0, "det.loss.ohem_ratio must be > 0")
        check(ls.k > 0, "det.loss.k must be > 0")
        check(ls.eps > 0, "det.loss.eps must be > 0")

        long_side = self.infer.long_side
        check(
            long_side % 32 == 0 and long_side >= 64,
            "det.infer.long_side must be a multiple of 32 and >= 64",
        )

        check(0 < p.bin_thresh < 1, "det.post.bin_thresh must be in (0, 1)")
        check(0 <= p.box_thresh < 1, "det.post.box_thresh must be in [0, 1)")
        check(p.min_component_area >= 1, "det.post.min_component_area must be >= 1")
        check(p.min_box_side >= 0, "det.post.min_box_side must be >= 0")
        check(p.max_candidates > 0, "det.post.max_candidates must be > 0")
        names = [op.name for op in p.operating_points]
        check(1 <= len(names) <= 4, "det.post.operating_points must list 1 to 4 points")
        check(len(set(names)) == len(names), "det.post.operating_points names must be unique")
        for i, op in enumerate(p.operating_points):
            where = f"det.post.operating_points[{i}]"
            check(
                bool(OP_NAME_RE.match(op.name)),
                f"{where}.name {op.name!r} must match {OP_NAME_RE.pattern}",
            )
            check(op.unclip_ratio > 0, f"{where}.unclip_ratio must be > 0")

        for f in dataclasses.fields(self.diagnostics):
            value = getattr(self.diagnostics, f.name)
            if f.name in _DET_DIAG_SHARES:
                check(0 < value <= 1, f"det.diagnostics.{f.name} must be in (0, 1]")
            else:
                check(value > 0, f"det.diagnostics.{f.name} must be > 0")
        return errors


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
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    det: DetConfig | None = None  # required for task det, refused for rec and e2e

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
        try:
            from glypharis.data.augment import Augmenter

            Augmenter(d.augmentation, d.augmentation_params)
        except ValueError as exc:
            errors.append(f"data.augmentation: {exc}")
        check(d.train_eval_subset >= 0, "data.train_eval_subset must be >= 0")
        check(d.val_subset is None or d.val_subset > 0, "data.val_subset must be > 0 or null")
        check(d.bucket_chunk > 0, "data.bucket_chunk must be > 0")
        check(d.max_batch_columns is None or d.max_batch_columns >= d.max_width,
              "data.max_batch_columns must be null or at least data.max_width")  # fmt: skip
        check(all(v > 0 for v in d.train_limits.values()), "data.train_limits must be > 0")
        check(d.epoch_lines is None or d.epoch_lines > 0, "data.epoch_lines must be > 0 or null")
        m = self.model
        check(len(m.cnn_channels) == 6 and all(c > 0 for c in m.cnn_channels),
              "model.cnn_channels must list 6 positive widths")  # fmt: skip
        check(m.lstm_hidden > 0 and m.lstm_layers > 0, "model.lstm_* must be > 0")
        check(0 <= m.dropout < 1, "model.dropout must be in [0, 1)")
        check(m.frame_stride in (4, 8), "model.frame_stride must be 4 or 8")
        if self.task == "rec" and self.model_name is not None:
            check((self.model_name == "svtr") == (m.svtr is not None),
                  "model.svtr is set exactly when model_name is svtr")  # fmt: skip
        s = m.svtr
        if s is not None:
            check(m.frame_stride == 4, "model.frame_stride must be 4 for svtr")
            lists = (s.dims, s.local_blocks, s.global_blocks, s.heads)
            check(all(len(x) == 3 for x in lists), "model.svtr lists must have 3 stages")
            check(all(d > 0 and h > 0 and d % h == 0 for d, h in zip(s.dims, s.heads, strict=True)),
                  "model.svtr dims must be positive multiples of their heads")  # fmt: skip
            check(all(b >= 0 for b in s.local_blocks + s.global_blocks), "svtr blocks must be >= 0")
            check(len(s.local_kernel) == 2 and all(k % 2 == 1 for k in s.local_kernel),
                  "model.svtr.local_kernel must be two odd sizes")  # fmt: skip
            check(s.mlp_ratio > 0, "model.svtr.mlp_ratio must be > 0")
        check(t.max_steps is None or t.max_steps > 0, "train.max_steps must be > 0 or null")
        check(c.every_steps is None or c.every_steps > 0, "checkpoint.every_steps must be > 0")
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

        if self.task in ("rec", "e2e"):
            check(self.det is None, f"task {self.task}: a det section is not allowed")
        if self.task == "det":
            check(self.det is not None, "task det needs a det section")
            # The recognizer sections stay out of the det config hash, so a value set there would
            # be silently ignored; refuse it instead.
            for name, default in (("data", DataConfig()), ("model", ModelConfig())):
                check(
                    getattr(self, name) == default,
                    f"task det: the recognizer {name} section must be unset",
                )
            rec_diag = DiagnosticsConfig()
            check(
                g.underfit_cer_threshold == rec_diag.underfit_cer_threshold
                and g.blank_ratio_alarm == rec_diag.blank_ratio_alarm,
                "task det: diagnostics.underfit_cer_threshold and blank_ratio_alarm are "
                "recognizer settings (see det.diagnostics)",
            )
            check(
                e.monitor == "val_f1_50" and e.mode == "max",
                "task det: early_stopping must monitor val_f1_50 with mode max",
            )
            check(t.batch_size >= 2, "task det: train.batch_size must be >= 2 (BatchNorm)")
        if self.det is not None:
            errors.extend(self.det.errors())

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
