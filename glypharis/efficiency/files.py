"""Writing a Phase 16 output in one step, so a crash never leaves a half-written file.

Phase 16 outputs are written once and refused when present (a void sitting is moved aside by
hand). A file cut short by a crash would then block the retry as if it were a result, so each one
is written to a temporary file in the same folder and moved into place with `os.replace`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: str | Path, text: str) -> None:
    """Write ``text`` (UTF-8, ``\\n`` line ends) to ``path`` through a temporary file beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_atomic(path: str | Path, obj: Any, *, ensure_ascii: bool = False) -> None:
    """`write_text_atomic` of ``obj`` as indented JSON with a final newline."""
    write_text_atomic(path, json.dumps(obj, indent=1, ensure_ascii=ensure_ascii) + "\n")
