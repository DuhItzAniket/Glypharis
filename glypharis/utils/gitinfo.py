"""Capture the git state of the working tree, recorded with every run and report."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GitInfo:
    commit: str | None
    branch: str | None
    dirty: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(cwd: str | Path = ".") -> GitInfo:
    """Commit hash, branch and dirty flag; all ``None`` outside a git repository."""
    path = Path(cwd)
    commit = _git(["rev-parse", "HEAD"], path)
    if commit is None:
        return GitInfo(commit=None, branch=None, dirty=None)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], path)
    status = _git(["status", "--porcelain", "--untracked-files=no"], path)
    return GitInfo(commit=commit, branch=branch, dirty=bool(status) if status is not None else None)
