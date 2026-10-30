"""A checkpoint rename blocked for a moment by another process is retried (Phase 12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from glypharis.training import checkpoint


def test_a_briefly_locked_target_is_retried(tmp_path, monkeypatch):
    real = Path.replace
    calls = {"n": 0}

    def flaky(self, target):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(32, "being used by another process")
        return real(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    monkeypatch.setattr(checkpoint.time, "sleep", lambda s: None)
    out = checkpoint.save_checkpoint(tmp_path / "last.pt", {"x": 1})
    assert calls["n"] == 3
    assert checkpoint.load_checkpoint(out)["x"] == 1
    assert not (tmp_path / "last.pt.tmp").exists()


def test_a_lock_that_never_clears_still_raises(tmp_path, monkeypatch):
    def locked(self, target):
        raise PermissionError(32, "being used by another process")

    monkeypatch.setattr(Path, "replace", locked)
    monkeypatch.setattr(checkpoint.time, "sleep", lambda s: None)
    with pytest.raises(PermissionError):
        checkpoint.save_checkpoint(tmp_path / "last.pt", {"x": 1})
