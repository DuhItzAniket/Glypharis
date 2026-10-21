"""Packed line crops: one binary file of encoded images plus a JSON-lines index.

    <name>.bin          concatenated image bytes (grayscale JPEG or PNG)
    <name>.index.jsonl  one record per crop: offset, length, width, height, label and provenance
    <name>.meta.json    format version, crop settings, counts, sha256 of the .bin

Readers keep the index in memory and open the .bin lazily, so each DataLoader worker gets its own
file handle (no shared handles across processes on Windows; RISK_REGISTER R-09).
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image

PACK_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CropRecord:
    key: str  # line_id
    doc_id: str
    split: str
    source: str
    label: str
    offset: int
    length: int
    width: int
    height: int


class LinePackWriter:
    def __init__(self, prefix: str | Path, meta: dict[str, Any] | None = None) -> None:
        self.prefix = Path(prefix)
        self.prefix.parent.mkdir(parents=True, exist_ok=True)
        self._bin: BinaryIO = open(self.prefix.with_suffix(".bin"), "wb")  # noqa: SIM115
        self._index = open(  # noqa: SIM115
            self.prefix.with_suffix(".index.jsonl"), "w", encoding="utf-8", newline="\n"
        )
        self._sha = hashlib.sha256()
        self._offset = 0
        self._meta = dict(meta or {})
        self.count = 0

    def add(
        self,
        key: str,
        doc_id: str,
        split: str,
        source: str,
        label: str,
        data: bytes,
        size: tuple[int, int],
    ) -> CropRecord:
        rec = CropRecord(
            key, doc_id, split, source, label, self._offset, len(data), size[0], size[1]
        )
        self._bin.write(data)
        self._sha.update(data)
        self._offset += len(data)
        self._index.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        self.count += 1
        return rec

    def close(self) -> dict[str, Any]:
        self._bin.close()
        self._index.close()
        meta = {
            "format_version": PACK_FORMAT_VERSION,
            "count": self.count,
            "bytes": self._offset,
            "bin_sha256": self._sha.hexdigest(),
            **self._meta,
        }
        self.prefix.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
        return meta

    def __enter__(self) -> LinePackWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        if not self._bin.closed:
            self.close()


class LinePackReader:
    def __init__(self, prefix: str | Path) -> None:
        self.prefix = Path(prefix)
        self.meta: dict[str, Any] = json.loads(
            self.prefix.with_suffix(".meta.json").read_text(encoding="utf-8")
        )
        if self.meta.get("format_version") != PACK_FORMAT_VERSION:
            raise ValueError(f"unsupported pack format {self.meta.get('format_version')}")
        with open(self.prefix.with_suffix(".index.jsonl"), encoding="utf-8") as fh:
            self.records = [CropRecord(**json.loads(line)) for line in fh if line.strip()]
        if len(self.records) != self.meta["count"]:
            raise ValueError("index and meta disagree on the number of crops")
        self._fh: BinaryIO | None = None

    def __len__(self) -> int:
        return len(self.records)

    def _file(self) -> BinaryIO:
        if self._fh is None:
            self._fh = open(self.prefix.with_suffix(".bin"), "rb")  # noqa: SIM115
        return self._fh

    def read_bytes(self, i: int) -> bytes:
        rec = self.records[i]
        fh = self._file()
        fh.seek(rec.offset)
        data = fh.read(rec.length)
        if len(data) != rec.length:
            raise OSError(f"short read for crop {rec.key}")
        return data

    def image(self, i: int) -> Image.Image:
        img = Image.open(io.BytesIO(self.read_bytes(i)))
        img.load()
        return img

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __getstate__(self) -> dict[str, Any]:
        # File handles are not shared between processes; each worker reopens lazily.
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def verify(self) -> bool:
        """Re-hash the .bin and compare with the meta record."""
        digest = hashlib.sha256()
        with open(self.prefix.with_suffix(".bin"), "rb") as fh:
            while block := fh.read(1 << 23):
                digest.update(block)
        return digest.hexdigest() == self.meta["bin_sha256"]
