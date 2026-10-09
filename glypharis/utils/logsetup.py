"""Structured logging: readable console lines plus a JSON-lines file per run.

Every record in the JSON file carries the run id, so logs from several runs can be merged and
filtered later. Extra fields passed with ``log.info("msg", extra={"metrics": {...}})`` are kept.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def __init__(self, run_id: str | None = None) -> None:
        super().__init__()
        self.run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if self.run_id is not None:
            payload["run_id"] = self.run_id
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def setup_logging(
    log_dir: str | Path | None = None,
    run_id: str | None = None,
    level: str = "INFO",
) -> Path | None:
    """Configure the root logger. Returns the JSON log path when ``log_dir`` is given.

    Safe to call more than once: handlers installed by a previous call are replaced.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        if getattr(handler, "_glypharis", False):
            root.removeHandler(handler)
            handler.close()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    console._glypharis = True  # type: ignore[attr-defined]
    root.addHandler(console)

    if log_dir is None:
        return None
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / f"{run_id or 'run'}.jsonl"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter(run_id))
    file_handler._glypharis = True  # type: ignore[attr-defined]
    root.addHandler(file_handler)
    return log_file
