"""Test-set access control (EXPERIMENT_PROTOCOL §9, WORKFLOW §5).

Every read of a held-out split for evaluation goes through `open_test_access`, which refuses
unless the caller names a registered-format experiment ID and one of the allowed phases, and appends
a row to `research/test_access_log.md` before any data is returned. The log is therefore complete by
construction rather than by discipline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from glypharis.config.schema import EXPERIMENT_ID_RE

ALLOWED_PHASES = (5, 14, 22)
TEST_SPLITS = {"test"}  # split names that count as held-out
DEFAULT_LOG = Path("research/test_access_log.md")
_PLACEHOLDER = "| — | — | — | — | — | No test access yet. |"


class HeldOutAccessError(PermissionError):
    """Raised when a test split is requested without a valid, logged reason."""


def is_test_split(split: str) -> bool:
    return split in TEST_SPLITS


def open_test_access(
    benchmark: str,
    split: str,
    experiment_id: str | None,
    checkpoint: str,
    phase: int | None,
    reason: str,
    log_path: str | Path = DEFAULT_LOG,
) -> None:
    """Check the request and log it. Non-test splits pass through without a log entry."""
    if not is_test_split(split):
        return
    if experiment_id is None or not EXPERIMENT_ID_RE.match(experiment_id):
        raise HeldOutAccessError(
            f"{benchmark}/{split} is a test split: pass an experiment ID like GW-BASE-001"
        )
    if phase not in ALLOWED_PHASES:
        raise HeldOutAccessError(
            f"{benchmark}/{split}: test splits are evaluated only in phases {ALLOWED_PHASES}"
        )
    if not reason.strip():
        raise HeldOutAccessError("a reason is required for every test access")
    path = Path(log_path)
    text = path.read_text(encoding="utf-8")
    row = (
        f"| {datetime.now(UTC).date().isoformat()} | {benchmark} {split} | {experiment_id} | "
        f"{checkpoint} | {phase} | {reason.strip()} |"
    )
    if _PLACEHOLDER in text:
        text = text.replace(_PLACEHOLDER, row)
    else:
        text = text.rstrip("\n") + "\n" + row
    path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8", newline="\n")
