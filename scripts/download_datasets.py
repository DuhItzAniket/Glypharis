"""Download datasets listed in data/sources.yaml and record what was fetched.

Usage:
    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --dataset hiertext --dry-run
    python scripts/download_datasets.py --dataset hiertext [--files gt_train,gt_test] [--verify]

A second run is a no-op: present files are skipped and manifests are rewritten only if
something changed. ``--verify`` re-hashes every selected file against the recorded sha256.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from glypharis.data.counters import COUNTERS
from glypharis.data.download import DownloadError, fetch_dataset, plan_dataset
from glypharis.data.registry import write_dataset_manifest
from glypharis.data.sources import SourceError, load_sources
from glypharis.utils.logsetup import setup_logging

log = logging.getLogger("download_datasets")


def _gib(n: int | None) -> str:
    return "unknown" if n is None else f"{n / 1024**3:.2f} GiB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download datasets reproducibly.")
    parser.add_argument("--sources", default="data/sources.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", action="append", default=[], help="dataset key (repeatable)")
    parser.add_argument("--files", default=None, help="comma-separated file names to fetch")
    parser.add_argument("--verify", action="store_true", help="re-hash present files")
    parser.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    parser.add_argument("--list", action="store_true", help="list registered datasets")
    parser.add_argument("--parallel", type=int, default=1, help="concurrent file transfers")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root)
    setup_logging(data_root / "cache" / "logs", run_id="download_datasets")
    try:
        sources = load_sources(args.sources)
    except (OSError, SourceError) as exc:
        log.error("cannot read %s: %s", args.sources, exc)
        return 2

    if args.list or not args.dataset:
        for key, spec in sources.items():
            total = sum(f.size or 0 for f in spec.files)
            size = _gib(total)
            print(f"{key:12s} phase {spec.phase:2d}  {size:>11s}  {spec.title} ({spec.licence})")
        return 0 if args.list else 2

    unknown = [d for d in args.dataset if d not in sources]
    if unknown:
        log.error("unknown dataset(s): %s (see --list)", ", ".join(unknown))
        return 2
    only = set(args.files.split(",")) if args.files else None

    if args.dry_run:
        grand = 0
        for key in args.dataset:
            for p in plan_dataset(sources[key], data_root):
                if only and p.spec.name not in only:
                    continue
                grand += p.bytes_to_fetch or 0
                print(f"{p.status:8s} {_gib(p.bytes_to_fetch):>11s}  {p.spec.url}  ->  "
                      f"{p.path.relative_to(data_root).as_posix()}")  # fmt: skip
        free = shutil.disk_usage(data_root if data_root.exists() else Path(".")).free
        print(f"total to fetch: {_gib(grand)}; free disk: {_gib(free)}")
        return 0

    try:
        for key in args.dataset:
            fetch_dataset(
                sources[key],
                data_root,
                only=only,
                verify=args.verify,
                counter=COUNTERS.get(key),
                parallel=args.parallel,
            )
    except DownloadError as exc:
        log.error("%s", exc)
        return 1
    write_dataset_manifest(data_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
