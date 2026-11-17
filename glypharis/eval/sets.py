"""The evaluation sets, and where the repository and its test-access log are (Phases 5 and 14).

`EVAL_SETS` is the one table every runner and scorer reads: which benchmark and split a set is,
where its line pack and page file live under the data root, and how its pages are scored. It used
to live in `scripts/run_baselines.py`; it moved here in Phase 14 so the GLYPHARIS runner, the
baseline runner and the scorers cannot disagree about a set.

Per set:

- ``benchmark`` and ``split``: what the test-access log records. ``split == "test"`` is held out.
- ``pack``: the ground-truth line (or word) crops for recognition, or None.
- ``pages``: the page file for detection and end to end, or None.
- ``reading_order``: the page text has a reading order, so order-aware page metrics are reported.
  HierText's annotation order is not one (its README says so even inside a paragraph) and FUNSD
  has none; GDC and GDC-scan take theirs from the PDF text layer, OmniDocBench from its published
  order with unordered blocks last.
- ``vertical_view``: rotated lines are left out of every metric and scored on their own
  (`glypharis.eval.page_metrics.vertical_view`), as for GDC in Phase 5.
- ``block_level``: the reference unit is a layout block, not a line (OmniDocBench), so only the
  order-aware page metrics mean anything; detection, bag-of-lines and E2E F1 are n/a.

Paths are relative to the data root, like every path the runners write.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from glypharis.eval.access import DEFAULT_LOG

# fmt: off
EVAL_SETS: dict[str, dict[str, Any]] = {
    "hiertext_val": {"benchmark": "hiertext", "split": "validation",
                     "pack": "processed/hiertext/lines_validation_eval",
                     "pages": "processed/hiertext/pages_validation.jsonl.gz",
                     "reading_order": False, "vertical_view": False, "block_level": False},
    "glyphsynth_val": {"benchmark": "glyphsynth", "split": "validation",
                       "pack": "processed/glyphsynth/lines_val", "pages": None,
                       "reading_order": False, "vertical_view": False, "block_level": False},
    "hiertext_test": {"benchmark": "hiertext", "split": "test",
                      "pack": "processed/hiertext/lines_test",
                      "pages": "processed/hiertext/pages_test.jsonl.gz",
                      "reading_order": False, "vertical_view": False, "block_level": False},
    "gdc_test": {"benchmark": "gdc", "split": "test", "pack": "processed/gdc/lines_test",
                 "pages": "processed/gdc/pages_test.jsonl.gz",
                 "reading_order": True, "vertical_view": True, "block_level": False},
    # Phase 14 sets.
    "iiit5k_test": {"benchmark": "iiit5k", "split": "test", "pack": "processed/iiit5k/words_test",
                    "pages": None,
                    "reading_order": False, "vertical_view": False, "block_level": False},
    "funsd_test": {"benchmark": "funsd", "split": "test", "pack": "processed/funsd/lines_test",
                   "pages": "processed/funsd/pages_test.jsonl.gz",
                   "reading_order": False, "vertical_view": False, "block_level": False},
    "gdc_scan_test": {"benchmark": "gdc_scan", "split": "test",
                      "pack": "processed/gdc_scan/lines_test",
                      "pages": "processed/gdc_scan/pages_test.jsonl.gz",
                      "reading_order": True, "vertical_view": True, "block_level": False},
    "omnidocbench_test": {"benchmark": "omnidocbench", "split": "test", "pack": None,
                          "pages": "processed/omnidocbench/pages_test.jsonl.gz",
                          "reading_order": True, "vertical_view": False, "block_level": True},
}
# fmt: on

PHASE5_SETS = ("hiertext_val", "glyphsynth_val", "hiertext_test", "gdc_test")
PHASE14_SETS = ("iiit5k_test", "funsd_test", "gdc_scan_test", "omnidocbench_test")

# The 100 OmniDocBench pages every system is also scored on (PP-OCRv5 server runs on these only).
# Repo-relative; the file lists page ids (`Page.doc_id`) under "pages".
OMNIDOCBENCH_SUBSET = Path("data/manifests/processed/omnidocbench_subset100.json")
SUBSET_NAME = "subset100"

PHASE14_BRANCH = "phase/14-external-eval"


def _git(args: list[str], start: Path) -> str:
    out = subprocess.run(["git", *args], cwd=start, capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {start}: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def repo_root(start: Path = Path(".")) -> Path:
    """Top of the working tree that holds ``start`` (a worktree's own root, not the main one)."""
    return Path(_git(["rev-parse", "--show-toplevel"], start))


def access_log(start: Path = Path(".")) -> Path:
    """The test-access log of the repository that holds ``start``.

    Resolved from the repository root rather than the working directory, so a runner started from
    a sub-folder still logs to the one committed log instead of creating or missing a file.
    """
    return repo_root(start) / DEFAULT_LOG


def require_branch(name: str, start: Path = Path(".")) -> None:
    """Refuse unless the checked-out branch is ``name`` (a detached HEAD is refused too)."""
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], start)
    if branch != name:
        raise RuntimeError(f"this run must start on branch {name}; HEAD is on {branch}")


def subset_pages(data_root: Path = Path("data")) -> list[str]:
    """Page ids of the OmniDocBench subset, as committed with the Phase 14 plan (100, sorted).

    Read under the runner's data root, like every other input, so a test can point it elsewhere.
    """
    path = data_root / OMNIDOCBENCH_SUBSET.relative_to("data")
    pages = [str(p) for p in json.loads(path.read_text(encoding="utf-8"))["pages"]]
    if len(pages) != len(set(pages)):
        raise ValueError(f"{OMNIDOCBENCH_SUBSET.as_posix()} lists a page twice")
    return pages


# The Phase 5 prediction files' hashes, written by `scripts/phase14_score.py hash-phase5` before
# any Phase 14 run (Phase 14 plan, item 4). Relative to the data root.
PHASE5_HASHES = Path("manifests/processed/phase5_baseline_files.json")


def require_phase5_hashes(data_root: Path = Path("data")) -> None:
    """Refuse (SystemExit) a Phase 14 run before the Phase 5 files have been hashed."""
    if not (data_root / PHASE5_HASHES).is_file():
        raise SystemExit(
            f"data/{PHASE5_HASHES.as_posix()} is missing: the Phase 5 files are hashed "
            "(`phase14_score.py hash-phase5`) before any Phase 14 run"
        )
