from __future__ import annotations

import argparse

import pytest

from glypharis.config import (
    ConfigError,
    RunConfig,
    add_config_args,
    build_config,
    config_from_args,
    config_hash,
    load_config,
    save_resolved,
)


def test_base_config_loads(repo_root):
    cfg = load_config(repo_root / "configs" / "base.yaml")
    assert isinstance(cfg, RunConfig)
    assert cfg.seed == 1337
    assert cfg.train.mixed_precision == "bf16"


# ------------------------------------------------------------------ exit criterion: strictness


def test_unknown_top_level_key_rejected():
    with pytest.raises(ConfigError, match=r"unknown config key.*lerning_rate"):
        build_config({"lerning_rate": 0.1})


def test_unknown_nested_key_rejected():
    with pytest.raises(ConfigError, match=r"train\.batchsize"):
        build_config({"train": {"batchsize": 32}})


@pytest.mark.parametrize(
    ("data", "where"),
    [
        ({"seed": "1337"}, "seed: expected int"),
        ({"seed": True}, "seed: expected int"),
        ({"train": {"batch_size": 3.5}}, "train.batch_size: expected int"),
        ({"train": {"deterministic": 1}}, "train.deterministic: expected bool"),
        ({"data": {"train": "hiertext"}}, "data.train: expected a list"),
        ({"data": {"train": ["a", 3]}}, r"data.train\[1\]: expected str"),
        ({"train": "fast"}, "train: expected a section"),
        ({"seed": None}, "seed: null is not allowed"),
    ],
)
def test_wrong_type_rejected(data, where):
    with pytest.raises(ConfigError, match=where):
        build_config(data)


def test_float_written_as_1e_minus_3_gets_a_hint(write_yaml):
    # PyYAML reads `1e-3` (no dot) as a string; the error must say how to fix it.
    path = write_yaml("c.yaml", "train:\n  learning_rate: 1e-3\n")
    with pytest.raises(ConfigError, match=r"write 1.0e-3"):
        load_config(path)


def test_int_accepted_for_float():
    cfg = build_config({"train": {"learning_rate": 1}})
    assert cfg.train.learning_rate == 1.0
    assert isinstance(cfg.train.learning_rate, float)


def test_literal_rejected():
    with pytest.raises(ConfigError, match="not one of"):
        build_config({"train": {"mixed_precision": "fp8"}})


def test_optional_accepts_null():
    cfg = build_config({"train": {"grad_clip_norm": None}})
    assert cfg.train.grad_clip_norm is None


@pytest.mark.parametrize(
    ("data", "msg"),
    [
        ({"train": {"batch_size": 0}}, "batch_size must be > 0"),
        ({"early_stopping": {"monitor": "train_loss"}}, "must be a validation metric"),
        ({"experiment_id": "REC-1"}, "does not match"),
        ({"diagnostics": {"blank_ratio_alarm": 1.5}}, "blank_ratio_alarm"),
    ],
)
def test_value_checks(data, msg):
    with pytest.raises(ConfigError, match=msg):
        build_config(data)


def test_valid_experiment_id():
    assert build_config({"experiment_id": "GW-REC-001"}).experiment_id == "GW-REC-001"


# ------------------------------------------------------------------------------- inheritance


def test_inherits_merges_nested_and_replaces_lists(write_yaml):
    write_yaml(
        "base.yaml",
        "seed: 1\ntrain:\n  batch_size: 64\n  epochs: 10\ndata:\n  train: [a, b]\n",
    )
    child = write_yaml(
        "exp/child.yaml",
        "inherits: ../base.yaml\ntrain:\n  batch_size: 32\ndata:\n  train: [c]\n",
    )
    cfg = load_config(child)
    assert cfg.seed == 1
    assert cfg.train.batch_size == 32
    assert cfg.train.epochs == 10  # kept from parent
    assert cfg.data.train == ["c"]  # lists replace, never concatenate


def test_multiple_parents_later_wins(write_yaml):
    write_yaml("a.yaml", "seed: 1\ntrain:\n  epochs: 5\n")
    write_yaml("b.yaml", "seed: 2\n")
    child = write_yaml("c.yaml", "inherits: [a.yaml, b.yaml]\n")
    cfg = load_config(child)
    assert cfg.seed == 2
    assert cfg.train.epochs == 5


def test_inheritance_cycle_detected(write_yaml):
    write_yaml("a.yaml", "inherits: b.yaml\n")
    b = write_yaml("b.yaml", "inherits: a.yaml\n")
    with pytest.raises(ConfigError, match="cycle"):
        load_config(b)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_non_mapping_yaml_rejected(write_yaml):
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(write_yaml("x.yaml", "- 1\n- 2\n"))


# ------------------------------------------------------------------- overrides + interpolation


def test_overrides_are_typed(write_yaml):
    path = write_yaml("c.yaml", "seed: 1\n")
    cfg = load_config(path, ["train.batch_size=16", "train.deterministic=false", "data.val=[x]"])
    assert cfg.train.batch_size == 16
    assert cfg.train.deterministic is False
    assert cfg.data.val == ["x"]


def test_override_of_unknown_key_rejected(write_yaml):
    path = write_yaml("c.yaml", "seed: 1\n")
    with pytest.raises(ConfigError, match="unknown"):
        load_config(path, ["train.batchsize=16"])


def test_malformed_override_rejected(write_yaml):
    path = write_yaml("c.yaml", "seed: 1\n")
    with pytest.raises(ConfigError, match=r"key.path=value"):
        load_config(path, ["train.batch_size"])


def test_interpolation():
    cfg = build_config({"experiment_id": "GW-REC-001"})
    assert cfg.checkpoint.dir == "checkpoints/GW-REC-001"


def test_null_reference_left_for_run_identity_check():
    cfg = build_config({"task": "rec", "model_name": "m"})
    assert cfg.checkpoint.dir == "checkpoints/${experiment_id}"
    with pytest.raises(ConfigError, match="experiment_id"):
        cfg.require_run_identity()


def test_unresolved_reference_rejected():
    with pytest.raises(ConfigError, match="unresolved reference"):
        build_config({"model_name": "${nope.key}"})


def test_reference_to_section_rejected():
    with pytest.raises(ConfigError, match="section"):
        build_config({"model_name": "${train}"})


def test_complete_run_identity_passes():
    cfg = build_config({"experiment_id": "GW-REC-001", "task": "rec", "model_name": "crnn"})
    cfg.require_run_identity()


# --------------------------------------------------------------------------- hashing + saving


def test_hash_ignores_key_order():
    a = build_config({"seed": 3, "train": {"epochs": 2, "batch_size": 8}})
    b = build_config({"train": {"batch_size": 8, "epochs": 2}, "seed": 3})
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(build_config({"seed": 4}))


def test_save_resolved_round_trip(tmp_path):
    cfg = build_config({"experiment_id": "GW-REC-001", "seed": 9})
    path = save_resolved(cfg, tmp_path / "run" / "config.yaml")
    assert load_config(path) == cfg


# -------------------------------------------------------------------------------------- CLI


def test_cli_shortcuts_and_set(repo_root):
    parser = add_config_args(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--config", str(repo_root / "configs" / "base.yaml"),
            "--seed", "5",
            "--device", "cpu",
            "--resume", "checkpoints/x/last.pt",
            "--set", "train.batch_size=8",
            "--set", "seed=6",
        ]
    )  # fmt: skip
    cfg = config_from_args(args)
    assert cfg.device == "cpu"
    assert cfg.checkpoint.resume == "checkpoints/x/last.pt"
    assert cfg.train.batch_size == 8
    assert cfg.seed == 6  # explicit --set wins over the shortcut
