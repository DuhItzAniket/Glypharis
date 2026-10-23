"""Detector data: a compact page index, seeded training crops and whole validation pages.

**Page index.** `PageIndex.build` reads a page file once, checks it (§4.9) and caches its
geometry as flat NumPy arrays under ``<data_root>/cache/det/``. Pickling sends only the cache path
and the subset indices; each spawned DataLoader worker reopens the arrays memory-mapped, so no
worker deserialises half a million polygons and the OS shares one copy of the pages.

**Training crops.** `DetTrainDataset` item ``(epoch, t)`` is crop ``t % crops_per_page`` of page
``t // crops_per_page``. Every random choice comes from ``default_rng([seed, epoch, t])`` in a
fixed order (scale, angle, crop mode, line, offsets, then photometric draws), so a crop does not
depend on the worker that makes it, on the number of workers or on scheduling. Crops have one
fixed shape, and `CropBatchSampler` always drops the last partial batch, so every training batch
has the same shape.

**Validation pages.** `DetEvalDataset` serves whole pages at the inference scale, padded exactly
as `prepare_page` pads them for inference, optionally with labels at that geometry for the
validation loss and the map statistics.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Collection, Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final, Literal

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from glypharis.config.schema import DetConfig
from glypharis.data.det_augment import PhotometricAugmenter
from glypharis.data.det_geometry import Affine, decode_page, render_view, sample_view
from glypharis.data.det_labels import CARE, LabelStats, line_reason, make_labels
from glypharis.data.download import sha256_file
from glypharis.data.pages import Page, read_pages
from glypharis.models.det.preprocess import ResizeInfo, prepare_page
from glypharis.utils.seed import worker_init_fn

INDEX_VERSION: Final = 1
DET_DATA_VERSION: Final = 1  # written to checkpoints and summaries with LABEL_VERSION

# Streams for seeded page subsets, drawn as default_rng([seed, stream]) (§4.7).
TRAIN_SUBSET_STREAM: Final = 2
TRAIN_EVAL_STREAM: Final = 3
VAL_SUBSET_STREAM: Final = 4
OVERFIT_STREAM: Final = 11

MAP_KEYS: Final = ("shrink_map", "shrink_mask", "thresh_map", "thresh_mask")
# Cached arrays per kind of polygon: all points, each polygon's first point, each page's first
# polygon. Line reasons are stored once per line next to them.
_KEYS: Final = {
    "line": ("points", "poly_offsets", "line_offsets"),
    "ignore": ("ignore_points", "ignore_poly_offsets", "ignore_offsets"),
}
_ARRAYS: Final = (*_KEYS["line"], "reason", *_KEYS["ignore"])


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    image: str  # relative to data_root
    width: int
    height: int
    split: str
    source: str


def _load_array(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def split_polygons(points: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    """Cut flat (P, 2) points into polygons at ``offsets`` (K + 1 values, starting at 0)."""
    return np.split(points, offsets[1:-1]) if len(offsets) > 1 else []


def care_centres(points: np.ndarray, offsets: np.ndarray, care: np.ndarray) -> np.ndarray:
    """Vertex mean of each care line, (K_care, 2) in file order: where text-seeking crops aim.

    Each line is averaged over its own points before the care lines are picked. Summing segments
    that start only at care lines would fold every don't-care line that follows a care line into
    it. Lines without points (none in a checked page file) have no centre.
    """
    counts = np.diff(offsets)
    has_points = counts > 0
    if not (care & has_points).any():
        return np.zeros((0, 2))
    # Starts of every non-empty line, strictly increasing, so each segment is one line.
    sums = np.add.reduceat(points, offsets[:-1][has_points], axis=0)
    means: np.ndarray = sums / counts[has_points][:, None]
    return means[care[has_points]]


class PageIndex:
    """Page records and line geometry of one page file, or a subset of one.

    Line ``reason`` values are `det_labels.line_reason` codes, computed once at build time.
    """

    sha256: str  # of the pages file

    def __init__(self, cache_dir: str | Path, ids: np.ndarray | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self._ids = None if ids is None else np.asarray(ids, dtype=np.int64)
        self._open()

    def _open(self) -> None:
        meta = json.loads((self.cache_dir / "meta.json").read_text(encoding="utf-8"))
        if meta.get("index_version") != INDEX_VERSION:
            raise ValueError(f"{self.cache_dir}: index version {meta.get('index_version')}")
        self.sha256 = meta["pages_sha256"]
        pages = json.loads((self.cache_dir / "pages.json").read_text(encoding="utf-8"))
        self._records = [PageRecord(**p) for p in pages]
        self._arr = {name: _load_array(self.cache_dir / f"{name}.npy") for name in _ARRAYS}

    # ------------------------------------------------------------------------------- building
    @classmethod
    def build(
        cls, pages_file: Path, role: Literal["train", "validation"], data_root: Path
    ) -> PageIndex:
        """Index a page file, refusing test, GDC and changed files (§4.9).

        Guards: every page's split must equal ``role`` (never a test split) and its source must not
        be GDC; the file's SHA-256 must equal ``splits.<role>.pages_sha256`` in
        ``<data_root>/manifests/processed/<source>.json``. The cache directory name carries the
        file hash and the index version, so a changed file or format gets a fresh cache.
        """
        if role not in ("train", "validation"):
            raise ValueError(f"role must be train or validation, got {role!r}")
        pages_file, data_root = Path(pages_file), Path(data_root)
        sha = sha256_file(pages_file)
        stem = pages_file.name.removesuffix(".jsonl.gz").removesuffix(".gz")
        cache = data_root / "cache" / "det" / f"{stem}-{sha[:12]}-v{INDEX_VERSION}"
        cached = (cache / "meta.json").is_file()
        pages: list[Page] = []
        if cached:
            records = [
                PageRecord(**p)
                for p in json.loads((cache / "pages.json").read_text(encoding="utf-8"))
            ]
        else:
            pages = list(read_pages(pages_file))
            records = [PageRecord(p.doc_id, p.image, p.width, p.height, p.split, p.source)
                       for p in pages]  # fmt: skip
        _check_records(records, role, pages_file)
        _check_manifest(records, role, sha, pages_file, data_root)
        if not cached:
            _write_cache(cache, pages, records, sha, pages_file)
        return cls(cache)

    # ------------------------------------------------------------------------------- access
    def _page(self, i: int) -> int:
        n = len(self)
        if not -n <= i < n:
            raise IndexError(f"page {i} out of range for {n} pages")
        i %= n
        return i if self._ids is None else int(self._ids[i])

    def __len__(self) -> int:
        return len(self._records) if self._ids is None else len(self._ids)

    def record(self, i: int) -> PageRecord:
        return self._records[self._page(i)]

    def doc_ids(self) -> list[str]:
        return [self.record(i).doc_id for i in range(len(self))]

    def _polygons(
        self, kind: Literal["line", "ignore"], i: int
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Page i's polygons flat: all points (a copy), offsets relative to the first point, and
        the page's range of polygon numbers."""
        points, poly_offsets, page_offsets = _KEYS[kind]
        p = self._page(i)
        a, b = (int(v) for v in self._arr[page_offsets][p : p + 2])
        offs = np.asarray(self._arr[poly_offsets][a : b + 1], dtype=np.int64)
        pts = np.array(self._arr[points][offs[0] : offs[-1]], dtype=np.float64)
        return pts, offs - offs[0], a, b

    def line_points(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Page i's lines flat: points (P, 2) float64, offsets (K + 1,) and care flags (K,)."""
        pts, offs, a, b = self._polygons("line", i)
        care = np.asarray(self._arr["reason"][a:b]) == CARE
        return pts, offs, care

    def lines(self, i: int) -> tuple[list[np.ndarray], np.ndarray]:
        """Page i's line polygons, float64 (K, 2) each in file order, and their care flags."""
        pts, offs, care = self.line_points(i)
        return split_polygons(pts, offs), care

    def ignore_regions(self, i: int) -> list[np.ndarray]:
        pts, offs, _, _ = self._polygons("ignore", i)
        return split_polygons(pts, offs)

    def subset(self, indices: np.ndarray) -> PageIndex:
        """The pages at ``indices`` (positions in this index), sharing the cached arrays."""
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        ids = np.array([self._page(int(i)) for i in idx], dtype=np.int64)
        out = object.__new__(PageIndex)
        out.__dict__.update(self.__dict__)
        out._ids = ids
        return out

    def ids_sha256(self) -> str:
        """SHA-256 of this index's doc_id list in order (recorded for every subset)."""
        return hashlib.sha256("\n".join(self.doc_ids()).encode("utf-8")).hexdigest()

    def __getstate__(self) -> dict[str, Any]:
        return {"cache_dir": str(self.cache_dir), "ids": self._ids}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.cache_dir = Path(state["cache_dir"])
        self._ids = state["ids"]
        self._open()


def _check_records(records: Sequence[PageRecord], role: str, pages_file: Path) -> None:
    if not records:
        raise ValueError(f"{pages_file}: no pages")
    for r in records:
        if r.split.lower().startswith("test"):
            raise ValueError(f"refused: {pages_file} holds test-split page {r.doc_id}")
        if r.source.lower() == "gdc":
            raise ValueError(f"refused: {pages_file} holds GDC page {r.doc_id}")
        if r.split != role:
            raise ValueError(f"{pages_file}: page {r.doc_id} has split {r.split!r}, not {role!r}")


def _check_manifest(
    records: Sequence[PageRecord], role: str, sha: str, pages_file: Path, data_root: Path
) -> None:
    sources = sorted({r.source for r in records})
    if len(sources) != 1:
        raise ValueError(f"{pages_file}: one source per page file, got {sources}")
    manifest = data_root / "manifests" / "processed" / f"{sources[0]}.json"
    if not manifest.is_file():
        raise ValueError(f"{pages_file}: no processed manifest {manifest}")
    entry = json.loads(manifest.read_text(encoding="utf-8")).get("splits", {}).get(role, {})
    if entry.get("pages_sha256") != sha:
        raise ValueError(
            f"{pages_file}: SHA-256 {sha[:12]}… does not match splits.{role}.pages_sha256 in "
            f"{manifest}; the file changed after it was built"
        )


def _flat(polys: Sequence[Sequence[Sequence[float]]]) -> tuple[list[Sequence[float]], list[int]]:
    points: list[Sequence[float]] = []
    counts: list[int] = []
    for poly in polys:
        points.extend(poly)
        counts.append(len(poly))
    return points, counts


def _offsets(counts: Sequence[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)


def _points(points: Sequence[Sequence[float]]) -> np.ndarray:
    return np.asarray(points, dtype=np.float64).reshape(-1, 2)


def _write_cache(
    cache: Path, pages: Sequence[Page], records: Sequence[PageRecord], sha: str, pages_file: Path
) -> None:
    """Write the index arrays into a temporary directory and rename it into place, so a reader
    never sees a half-written cache (two processes may build the same index at once)."""
    line_pts, line_counts = _flat([ln.polygon for p in pages for ln in p.lines])
    ign_pts, ign_counts = _flat([poly for p in pages for poly in p.ignore])
    arrays = {
        "points": _points(line_pts),
        "poly_offsets": _offsets(line_counts),
        "line_offsets": _offsets([len(p.lines) for p in pages]),
        "reason": np.array([line_reason(ln) for p in pages for ln in p.lines], dtype=np.uint8),
        "ignore_points": _points(ign_pts),
        "ignore_poly_offsets": _offsets(ign_counts),
        "ignore_offsets": _offsets([len(p.ignore) for p in pages]),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=cache.name + ".tmp-", dir=cache.parent))
    try:
        for name, arr in arrays.items():
            np.save(tmp / f"{name}.npy", arr)
        (tmp / "pages.json").write_text(
            json.dumps([asdict(r) for r in records]), encoding="utf-8", newline="\n"
        )
        meta = {
            "index_version": INDEX_VERSION,
            "pages_file": pages_file.name,
            "pages_sha256": sha,
            "pages": len(records),
            "lines": len(arrays["reason"]),
        }
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
        try:
            os.rename(tmp, cache)
        except OSError:
            if not (cache / "meta.json").is_file():
                raise
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def seeded_subset(index: PageIndex, n: int | None, seed: int, stream: int) -> PageIndex:
    """``n`` pages drawn without replacement by ``default_rng([seed, stream])``, kept in index
    order; the whole index when ``n`` is None or not smaller than it (§4.7)."""
    if n is None or n >= len(index):
        return index
    keep = np.random.default_rng([seed, stream]).choice(len(index), n, replace=False)
    return index.subset(np.sort(keep))


def read_gt_pages(pages_file: Path, doc_ids: Collection[str]) -> dict[str, Page]:
    """Full `Page` records of ``doc_ids`` (for scoring, main process only)."""
    wanted = set(doc_ids)
    out = {p.doc_id: p for p in read_pages(pages_file) if p.doc_id in wanted}
    missing = wanted - out.keys()
    if missing:
        raise KeyError(f"{len(missing)} doc_ids not in {pages_file}, e.g. {sorted(missing)[:3]}")
    return out


def assert_no_page_leakage(train: PageIndex, val: PageIndex) -> None:
    """Training and validation must share no document and no image (EXPERIMENT_PROTOCOL §6)."""

    def images(ix: PageIndex) -> set[str]:
        return {ix.record(i).image.replace("\\", "/").lower() for i in range(len(ix))}

    docs = set(train.doc_ids()) & set(val.doc_ids())
    imgs = images(train) & images(val)
    if docs or imgs:
        raise RuntimeError(f"LEAKAGE: {len(docs)} doc_ids and {len(imgs)} image paths are in both "
                           f"training and validation, e.g. {sorted(docs or imgs)[:3]}")  # fmt: skip


class DetTrainDataset(Dataset[dict[str, Any]]):
    """Seeded training crops with DBNet labels. Items are addressed by ``(epoch, item)``.

    An item holds ``image`` (S, S, 3) uint8 RGB, ``shrink_map``, ``shrink_mask`` and
    ``thresh_mask`` (S, S) uint8, ``thresh_map`` (S, S) float32, ``id`` ``"{doc_id}#{item}"`` and
    ``stats`` (`LabelStats`).
    """

    def __init__(
        self,
        index: PageIndex,
        data_root: Path,
        det: DetConfig,
        seed: int,
        photometric: bool = True,
    ) -> None:
        self.index, self.data_root, self.det, self.seed = index, Path(data_root), det, seed
        self.augmenter = PhotometricAugmenter(det.augment) if photometric else None

    def __len__(self) -> int:
        return len(self.index) * self.det.data.crops_per_page

    def __getitem__(self, item: tuple[int, int]) -> dict[str, Any]:
        epoch, t = item
        if not 0 <= t < len(self):
            raise IndexError(f"item {t} out of range for {len(self)} crops")
        data, size = self.det.data, self.det.data.crop_size
        rng = np.random.default_rng([self.seed, epoch, t])
        p = t // data.crops_per_page
        rec = self.index.record(p)
        pts, offs, care = self.index.line_points(p)
        centres = care_centres(pts, offs, care)
        view = sample_view(rng, (rec.width, rec.height), centres, data, self.det.infer.long_side)
        image = render_view(
            self.data_root / rec.image, (rec.width, rec.height), view, size, data.reduced_decode
        )
        polys = split_polygons(view.affine.apply(pts), offs)
        ignore = [view.affine.apply(q) for q in self.index.ignore_regions(p)]
        labels = make_labels(polys, care, ignore, (size, size), self.det.labels)
        if self.augmenter is not None:
            image = self.augmenter(image, rng)
        return {
            "image": image,
            "shrink_map": labels.shrink_map,
            "shrink_mask": labels.shrink_mask,
            "thresh_map": labels.thresh_map,
            "thresh_mask": labels.thresh_mask,
            "id": f"{rec.doc_id}#{t}",
            "stats": labels.stats,
        }


class DetEvalDataset(Dataset[dict[str, Any]]):
    """Whole pages at the inference scale, for validation and the final evaluation (§4.8).

    ``parts`` are (role, index) pairs served one after the other in order. An item holds ``image``
    (pad_h, pad_w, 3) uint8 RGB from `prepare_page`, ``doc_id``, ``role`` and the `ResizeInfo`
    fields, plus the four label maps at (pad_h, pad_w) when ``with_labels`` is set. The labels use
    the same rules as training, at the geometry ``Affine.scale(sx, sy)`` on the padded canvas.
    """

    def __init__(
        self,
        parts: Sequence[tuple[str, PageIndex]],
        data_root: Path,
        det: DetConfig,
        with_labels: bool,
    ) -> None:
        self.parts = list(parts)
        self.data_root, self.det, self.with_labels = Path(data_root), det, with_labels
        self._starts = np.cumsum([0] + [len(ix) for _, ix in self.parts])

    def __len__(self) -> int:
        return int(self._starts[-1])

    def __getitem__(self, i: int) -> dict[str, Any]:
        if not 0 <= i < len(self):
            raise IndexError(f"page {i} out of range for {len(self)} pages")
        k = int(np.searchsorted(self._starts, i, side="right")) - 1
        role, index = self.parts[k]
        j = i - int(self._starts[k])
        rec = index.record(j)
        rgb = decode_page(self.data_root / rec.image, (rec.width, rec.height))
        image, info = prepare_page(rgb, self.det.infer.long_side)
        out: dict[str, Any] = {"image": image, "doc_id": rec.doc_id, "role": role, **asdict(info)}
        if self.with_labels:
            g = Affine.scale(info.sx, info.sy)
            polys, care = index.lines(j)
            ignore = [g.apply(q) for q in index.ignore_regions(j)]
            labels = make_labels(
                [g.apply(q) for q in polys], care, ignore, (info.pad_h, info.pad_w), self.det.labels
            )
            out.update({key: getattr(labels, key) for key in MAP_KEYS})
        return out


class CropBatchSampler(Sampler[list[tuple[int, int]]]):
    """Fixed-size batches of ``(epoch, item)`` keys, rebuilt from ``(seed, epoch)``.

    Epoch e is ``default_rng([seed, e]).permutation(n_items)`` cut into full batches; the last
    partial batch is always dropped. ``set_epoch(e, start_batch)`` resumes inside an epoch.
    """

    def __init__(self, n_items: int, batch_size: int, seed: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.n_items, self.batch_size, self.seed = n_items, batch_size, seed
        self.epoch, self.start_batch = 0, 0

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch, self.start_batch = epoch, start_batch

    def batches(self, epoch: int) -> list[list[int]]:
        n = self.num_batches(epoch) * self.batch_size
        perm = np.random.default_rng([self.seed, epoch]).permutation(self.n_items)[:n]
        rows: list[list[int]] = perm.reshape(-1, self.batch_size).tolist()
        return rows

    def num_batches(self, epoch: int) -> int:
        return self.n_items // self.batch_size

    def __iter__(self) -> Iterator[list[tuple[int, int]]]:
        for b in self.batches(self.epoch)[self.start_batch :]:
            yield [(self.epoch, i) for i in b]

    def __len__(self) -> int:
        return max(self.num_batches(self.epoch) - self.start_batch, 0)


def collate_train(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack crops: ``images`` (N, S, S, 3) uint8, the four maps (N, S, S) (``thresh_map``
    float32, the others uint8), ``ids`` and the summed label ``stats``."""
    out: dict[str, Any] = {"images": torch.from_numpy(np.stack([b["image"] for b in batch]))}
    for key in MAP_KEYS:
        out[key] = torch.from_numpy(np.stack([b[key] for b in batch]))
    stats = LabelStats()
    for b in batch:
        stats.add(b["stats"])
    out["ids"] = [b["id"] for b in batch]
    out["stats"] = stats
    return out


_INFO_FIELDS: Final = tuple(f.name for f in fields(ResizeInfo))


def collate_eval(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """One page: ``images`` (1, pad_h, pad_w, 3) uint8, the maps (1, pad_h, pad_w) when present,
    ``doc_id``, ``role``, ``info`` (`ResizeInfo`) and the ResizeInfo fields as plain values."""
    if len(batch) != 1:
        raise ValueError(f"evaluation batches hold one page, got {len(batch)}")
    b = batch[0]
    out: dict[str, Any] = {
        "images": torch.from_numpy(np.ascontiguousarray(b["image"])).unsqueeze(0),
        "doc_id": b["doc_id"],
        "role": b["role"],
        "info": ResizeInfo(**{f: b[f] for f in _INFO_FIELDS}),
    }
    out.update({f: b[f] for f in _INFO_FIELDS})
    for key in MAP_KEYS:
        if key in b:
            out[key] = torch.from_numpy(b[key]).unsqueeze(0)
    return out


def det_worker_init(worker_id: int) -> None:
    """Seed the worker like every loader does, and keep it to one OpenCV and one torch thread:
    six workers each spawning a thread per core would oversubscribe the CPU."""
    worker_init_fn(worker_id)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
