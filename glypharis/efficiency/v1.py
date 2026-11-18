"""Is the v1 code still the evaluated code? (Phase 16 plan, item 1.)

v1 is the frozen weights, the freeze record's settings and the model, pipeline and export code as
it is at the ``phase14-eval`` tag. Phase 16 work happens on later commits (new scripts, this
package, reports), so `glypharis.pipeline.frozen.pinned_code_check`, which wants the whole tree
equal to the tag, would refuse every Phase 16 run. `assert_v1_code` checks only the paths that
make v1 (`V1_PATHS`): the working tree there must equal the tag's, tracked and untracked files
alike (ignored files, such as caches, are not part of the tree). Every Phase 16 script that builds
v1 calls it before the model loads and records what it returns.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Final

V1_TAG: Final = "phase14-eval"  # equals glypharis.pipeline.frozen.PIN_TAG (a test keeps it so)
V1_PATHS: Final = (
    "glypharis/pipeline",
    "glypharis/models",
    "glypharis/export",
    "glypharis/data",
    "configs",
)


class V1CodeError(RuntimeError):
    """The v1 code paths differ from the tag, or git could not tell."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise V1CodeError(f"git {' '.join(args)} could not run: {exc}") from exc


def assert_v1_code(
    root: str | Path = ".", tag: str = V1_TAG, paths: tuple[str, ...] = V1_PATHS
) -> dict[str, Any]:
    """Raise `V1CodeError` unless ``git diff --quiet <tag> -- <paths>`` is clean and
    ``git ls-files --others --exclude-standard -- <paths>`` is empty; return the tag, its commit,
    HEAD's commit and the paths checked.

    The diff compares the working tree (staged and unstaged changes) with the tag, so a change
    committed after the tag is caught as well as one not yet committed. Rename detection is off,
    so a file moved out of a v1 path counts as removed there.
    """
    root = Path(root)
    found = _git(root, "rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}")
    if found.returncode != 0 or not found.stdout.strip():
        raise V1CodeError(f"the tag {tag} does not exist; v1 cannot be checked")
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise V1CodeError(f"git rev-parse HEAD failed: {head.stderr.strip()}")
    diff = _git(root, "diff", "--quiet", "--no-renames", "--no-ext-diff", tag, "--", *paths)
    if diff.returncode == 1:
        names = _git(root, "diff", "--name-only", "--no-renames", tag, "--", *paths)
        changed = [n for n in names.stdout.splitlines() if n.strip()]
        raise V1CodeError(f"the v1 code differs from {tag} ({', '.join(paths)}): {changed[:10]}")
    if diff.returncode != 0:
        raise V1CodeError(f"git diff against {tag} failed: {diff.stderr.strip()}")
    others = _git(root, "ls-files", "--others", "--exclude-standard", "--", *paths)
    if others.returncode != 0:
        raise V1CodeError(f"git ls-files failed: {others.stderr.strip()}")
    untracked = [n for n in others.stdout.splitlines() if n.strip()]
    if untracked:
        raise V1CodeError(f"untracked files in the v1 code paths: {untracked[:10]}")
    return {
        "tag": tag,
        "tag_commit": found.stdout.strip(),
        "head": head.stdout.strip(),
        "paths": list(paths),
        "v1_code": "equal",
    }
