"""``data.epoch_lines`` (Phase 12): a fixed number of lines per epoch from a larger pool."""

from __future__ import annotations

import numpy as np
import pytest

from glypharis.config.loader import config_hash, config_to_dict, load_config
from glypharis.config.schema import ConfigError, RunConfig
from glypharis.data.rec_dataset import WidthBucketBatchSampler


def _widths(n: int = 3000) -> np.ndarray:
    return np.random.default_rng(0).integers(20, 900, n)


def test_epoch_lines_at_or_above_the_pool_changes_nothing():
    w = _widths()
    base = WidthBucketBatchSampler(w, 16, seed=3, chunk=4, max_columns=4096)
    for k in (None, len(w), len(w) + 10):
        s = WidthBucketBatchSampler(w, 16, seed=3, chunk=4, max_columns=4096, epoch_lines=k)
        assert [s.batches(e) for e in range(3)] == [base.batches(e) for e in range(3)]


def test_epoch_lines_draws_a_fresh_sample_each_epoch():
    w = _widths()
    s = WidthBucketBatchSampler(w, 16, seed=3, chunk=4, epoch_lines=1000)
    seen = []
    for e in range(4):
        idx = [i for b in s.batches(e) for i in b]
        assert len(idx) == 1000 and len(set(idx)) == 1000
        assert s.num_batches(e) == len(s.batches(e))
        seen.append(set(idx))
    assert seen[0] != seen[1]
    assert len(set().union(*seen)) > 2000  # four epochs reach most of the pool


def test_epoch_lines_keeps_old_hashes_and_is_validated(repo_root):
    cfg = load_config(repo_root / "configs/rec/gw-rec-003.yaml")
    assert "epoch_lines" not in config_to_dict(cfg)["data"]
    before = config_hash(cfg)
    cfg.data.epoch_lines = 348_935
    assert config_to_dict(cfg)["data"]["epoch_lines"] == 348_935
    assert config_hash(cfg) != before
    bad = RunConfig()
    bad.data.epoch_lines = 0
    with pytest.raises(ConfigError, match="epoch_lines"):
        bad.validate()
