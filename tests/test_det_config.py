"""The detector config section: GW-DET files, refusals, and recognizer hashes that cannot move."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from glypharis.config import ConfigError, config_hash, config_to_dict, load_config, save_resolved
from glypharis.config.schema import DetConfig, DetOperatingPoint
from glypharis.data.sources import load_sources

# Pinned when the det section was added. A change here means every recognizer run would be refused
# on resume and its reports would cite a different config: never update these to make a test pass.
REC_HASHES = {
    "gw-rec-001": "a13d7c4c367626da272252a16539d407ae1e244dde376745f1c904577401bdfa",
    "gw-rec-002": "5e33f43cb53dd5745091ffcb68bfdb650b5195dd8379ef25e6086781cb4bebf2",
}

# Every section that goes into a recognizer hash. Adding a field to one of them changes every
# recognizer hash, so it has to fail here first.
REC_SECTION_KEYS = {
    "paths": ["data_root", "reports_dir", "runs_dir"],
    "logging": ["level", "log_every_n_steps", "monitor_resources"],
    "data": [
        "augmentation", "augmentation_params", "bucket_chunk", "charset", "dataset_subset",
        "image_height", "image_width", "max_batch_columns", "max_width", "num_workers", "train",
        "train_eval_subset", "train_limits", "val", "val_subset",
    ],
    "model": ["cnn_channels", "dropout", "lstm_hidden", "lstm_layers"],
    "train": [
        "batch_size", "deterministic", "epochs", "grad_clip_norm", "gradient_accumulation",
        "learning_rate", "max_steps", "mixed_precision", "optimizer", "scheduler", "warmup_steps",
        "weight_decay",
    ],
    "checkpoint": ["checkpoint_interval", "dir", "every_steps", "keep_last", "resume"],
    "early_stopping": ["enabled", "min_delta", "mode", "monitor", "patience"],
    "diagnostics": [
        "blank_ratio_alarm", "grad_norm_alarm", "overfit_window", "stop_on_nan",
        "underfit_cer_threshold",
    ],
}  # fmt: skip
REC_SCALAR_KEYS = ["device", "experiment_id", "model_name", "seed", "task"]

DET_IDS = ["gw-det-001", "gw-det-002", "gw-det-003"]


def _det(repo_root: Path, name: str = "gw-det-001") -> Path:
    return repo_root / "configs" / "det" / f"{name}.yaml"


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


# ------------------------------------------------------------------------------- GW-DET files


@pytest.mark.parametrize("name", DET_IDS)
def test_det_configs_validate(repo_root, name):
    cfg = load_config(_det(repo_root, name))
    cfg.require_run_identity()
    assert cfg.task == "det" and cfg.experiment_id == name.upper()
    assert isinstance(cfg.det, DetConfig)
    assert cfg.early_stopping.enabled is False  # equal step budgets across the arms
    assert [op.name for op in cfg.det.post.operating_points] == ["dbnet15", "inverse10"]
    # The recognizer sections are validated to be defaults and stay out of the det hash.
    d = config_to_dict(cfg)
    assert "det" in d and "data" not in d and "model" not in d


def test_schema_defaults_are_the_gw_det_001_values(repo_root):
    cfg = load_config(_det(repo_root))
    assert cfg.det is not None
    ref = DetConfig()
    for section in ("data", "augment", "labels", "loss", "infer", "post", "diagnostics"):
        ours, theirs = getattr(cfg.det, section), getattr(ref, section)
        if section == "data":  # the page files have no default
            theirs.train_pages, theirs.val_pages = ours.train_pages, ours.val_pages
            # frozen from the sanity sequence by the 2026-10-26 registry amendment
            assert (ours.val_subset, ours.num_workers) == (400, 4)
            theirs.val_subset, theirs.num_workers = ours.val_subset, ours.num_workers
        assert ours == theirs, section
    m = cfg.det.model
    assert (m.backbone, m.init, m.inner_channels, m.channels_last) == (
        ref.model.backbone, ref.model.init, ref.model.inner_channels, ref.model.channels_last,
    )  # fmt: skip


def test_002_and_003_differ_from_001_only_at_their_keys(repo_root):
    base = _flat(config_to_dict(load_config(_det(repo_root))))
    expected = {
        "gw-det-002": {"det.model.init", "det.model.weights", "det.model.weights_sha256"},
        "gw-det-003": {"det.model.backbone", "det.model.weights", "det.model.weights_sha256"},
    }
    # experiment_id and model_name name the run; checkpoint.dir follows ${experiment_id}.
    identity = {"experiment_id", "model_name", "checkpoint.dir"}
    for name, keys in expected.items():
        other = _flat(config_to_dict(load_config(_det(repo_root, name))))
        assert other.keys() == base.keys()
        assert {k for k in base if base[k] != other[k]} == keys | identity, name


def test_det_weights_match_the_registered_source(repo_root):
    spec = load_sources(repo_root / "data" / "sources.yaml")["imagenet_backbones"]
    files = {f.filename: f for f in spec.files}
    seen = set()
    for name in DET_IDS:
        det = load_config(_det(repo_root, name)).det
        assert det is not None
        m = det.model
        if m.init == "scratch":
            assert m.weights is None and m.weights_sha256 is None
            continue
        assert m.weights is not None
        f = files[Path(m.weights).name]
        assert m.weights == f"raw/{spec.key}/{f.filename}"
        assert m.weights_sha256 == f.sha256
        assert f.filename.startswith(m.backbone + "-")
        seen.add(f.filename)
    assert seen == set(files)  # both registered files are used, by 001 and 003


def test_det_config_round_trips_through_the_resolved_file(repo_root, tmp_path):
    cfg = load_config(_det(repo_root, "gw-det-003"))
    path = save_resolved(cfg, tmp_path / "config.yaml")
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "data" not in saved and "model" not in saved
    again = load_config(path)
    assert again == cfg
    assert config_hash(again) == config_hash(cfg)


# --------------------------------------------------------------------- recognizer invariance


@pytest.mark.parametrize("name", sorted(REC_HASHES))
def test_recognizer_hashes_are_pinned(repo_root, name):
    cfg = load_config(repo_root / "configs" / "rec" / f"{name}.yaml")
    assert cfg.det is None
    assert config_hash(cfg) == REC_HASHES[name]


def test_recognizer_key_sets_are_pinned(repo_root):
    d = config_to_dict(load_config(repo_root / "configs" / "rec" / "gw-rec-001.yaml"))
    assert sorted(d) == sorted([*REC_SECTION_KEYS, *REC_SCALAR_KEYS])
    assert "det" not in d
    for section, keys in REC_SECTION_KEYS.items():
        assert sorted(d[section]) == keys, section


def test_explicit_null_det_leaves_a_recognizer_hash_alone(repo_root):
    path = repo_root / "configs" / "rec" / "gw-rec-001.yaml"
    assert config_hash(load_config(path, ["det=null"])) == REC_HASHES["gw-rec-001"]


# --------------------------------------------------------------------------------- refusals

_TWO_A = (
    "[{name: a, unclip_mode: dbnet, unclip_ratio: 1.5},"
    " {name: a, unclip_mode: inverse, unclip_ratio: 1.0}]"
)


@pytest.mark.parametrize(
    ("overrides", "msg"),
    [
        (["data.image_height=48"], "recognizer data section"),
        (["model.dropout=0.2"], "recognizer model section"),
        (["diagnostics.underfit_cer_threshold=0.4"], "recognizer settings"),
        (["det=null"], "task det needs a det section"),
        (["det.data.val_pages=[processed/hiertext/pages_test.jsonl.gz]"], "test or GDC"),
        (["det.data.train_pages=[processed/gdc/pages_validation.jsonl.gz]"], "test or GDC"),
        (["det.data.train_pages=[processed/hiertext/lines_train]"], "page files only"),
        (["det.data.train_pages=[]"], "train_pages must name exactly one page file"),
        (["det.data.val_pages=[processed/hiertext/pages_validation.jsonl.gz,"
          " processed/other/pages_validation.jsonl.gz]"], "val_pages must name exactly one"),
        (["early_stopping.monitor=val_cer"], "val_f1_50 with mode max"),
        (["early_stopping.mode=min"], "val_f1_50 with mode max"),
        (["train.batch_size=1"], "BatchNorm"),
        (["det.data.crop_size=650"], "crop_size must be a multiple of 32"),
        (["det.data.scale_min=3.0"], "scale_min <= scale_max"),
        (["det.augment.blur_p=1.5"], "blur_p must be in"),
        (["det.labels.thresh_min=0.8"], "thresh_min < thresh_max"),
        (["det.model.weights_sha256=null"], "weights_sha256 must be 64 lowercase hex"),
        (["det.model.weights_sha256=" + "8738CA79" * 8], "weights_sha256 must be 64 lowercase hex"),
        (["det.model.weights=null"], "weights is required for init imagenet"),
        (["det.model.init=scratch"], "must be null for init scratch"),
        (["det.model.inner_channels=30"], "inner_channels"),
        (["det.infer.long_side=1000"], "long_side must be a multiple of 32"),
        (["det.post.bin_thresh=1.0"], "bin_thresh"),
        (["det.post.operating_points=" + _TWO_A], "names must be unique"),
        (["det.post.operating_points=[]"], "1 to 4 points"),
        (["det.post.operating_points=[{name: Dbnet, unclip_mode: dbnet, unclip_ratio: 1.5}]"],
         r"operating_points\[0\]\.name"),
        (["det.post.operating_points=[{name: d, unclip_mode: dbnet, unclip_ratio: 0}]"],
         r"operating_points\[0\]\.unclip_ratio must be > 0"),
        (["det.post.box_type=rectangle"], "not one of"),
        (["det.post.operating_points=[{name: d, unclip_mode: exact, unclip_ratio: 1.0}]"],
         "not one of"),
        (["det.diagnostics.data_wait_alarm=1.5"], r"data_wait_alarm must be in \(0, 1\]"),
        (["det.diagnostics.thresh_divergence_rise=0"], "thresh_divergence_rise must be > 0"),
    ],
)  # fmt: skip
def test_det_config_refusals(repo_root, overrides, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(_det(repo_root), overrides)


def test_det_section_refused_in_a_recognizer_config(repo_root):
    path = repo_root / "configs" / "rec" / "gw-rec-001.yaml"
    with pytest.raises(ConfigError, match="task rec: a det section is not allowed"):
        load_config(path, ["det.infer.long_side=960"])


def test_scratch_with_null_weights_is_accepted(repo_root):
    cfg = load_config(_det(repo_root), ["det.model={init: scratch}"])
    assert cfg.det is not None and cfg.det.model.init == "scratch"


@pytest.mark.parametrize("key", ["name", "unclip_mode", "unclip_ratio"])
def test_operating_point_missing_a_key_is_a_config_error(repo_root, write_yaml, capsys, key):
    import validate_configs

    point = {"name": "inverse10", "unclip_mode": "inverse", "unclip_ratio": 1.0}
    del point[key]
    parent = _det(repo_root).as_posix()
    ops = yaml.safe_dump([{"name": "dbnet15", "unclip_mode": "dbnet", "unclip_ratio": 1.5}, point],
                         default_flow_style=True).strip()  # fmt: skip
    path = write_yaml(
        "cfgs/broken.yaml", f"inherits: {parent}\ndet:\n  post:\n    operating_points: {ops}\n"
    )
    where = rf"det\.post\.operating_points\[1\]\.{key}"
    # A ConfigError naming the key, never the TypeError of DetOperatingPoint(**kwargs).
    with pytest.raises(ConfigError, match=rf"missing config key\(s\): {where}"):
        load_config(path)
    assert validate_configs.main(["--root", str(path.parent)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and f"det.post.operating_points[1].{key}" in out
    assert "Traceback" not in out


def test_operating_point_has_no_defaults():
    with pytest.raises(TypeError):
        DetOperatingPoint()
