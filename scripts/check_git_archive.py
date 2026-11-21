"""Check what ``git archive`` of a commit leaves out, before any Zenodo deposit (Phase 23, item 5).

Zenodo receives code and documentation only (PHASE_STATUS Phase 23, D5): ``reports/`` and the
evaluation records under ``data/manifests`` carry ``export-ignore`` in ``.gitattributes``, so the
archive GitHub hands to Zenodo leaves them out.

    python scripts/check_git_archive.py [--commit HEAD] [--root .] [--worktree-attributes]

The script archives the commit to a temporary tar file and checks that

- no path in ``EXCLUDED`` (a folder or a file, as in ``.gitattributes``) is in the archive;
- every path in ``KEPT`` is in it, along with every other file of the commit;
- each excluded path exists in the commit, so the check is not passing on paths that were renamed.

``git archive`` reads ``.gitattributes`` from the commit itself; ``--worktree-attributes`` reads the
working tree's instead, to check an uncommitted change. Exits 1 and lists every problem on failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path

# Kept in step with the export-ignore lines of .gitattributes (tests/test_packaging.py checks it).
EXCLUDED: tuple[str, ...] = (
    "reports",
    # Records of the evaluation-only sets and of evaluation inputs and outputs.
    "data/manifests/gdc",
    "data/manifests/gdc_scan",
    "data/manifests/gdc_sources.json",
    "data/manifests/funsd.json",
    "data/manifests/iiit5k.json",
    "data/manifests/omnidocbench.json",
    "data/manifests/processed/benchmarks.json",
    "data/manifests/processed/eval_crops.json",
    "data/manifests/processed/glyph_c.json",
    "data/manifests/processed/glyph_c_samples.json",
    "data/manifests/processed/omnidocbench_subset100.json",
    "data/manifests/processed/phase14_outputs.json",
    "data/manifests/processed/phase5_baseline_files.json",
)
KEPT: tuple[str, ...] = ("glypharis", "docs", "README.md", "pyproject.toml")


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def commit_files(root: Path, commit: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "-z", "--name-only", commit],
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    return [n for n in out.split("\0") if n]


def archive_files(root: Path, commit: str, worktree_attributes: bool = False) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="glypharis-archive-") as tmp:
        tar = Path(tmp) / "archive.tar"
        cmd = ["git", "-C", str(root), "archive", "--format=tar", "-o", str(tar)]
        if worktree_attributes:
            cmd.append("--worktree-attributes")
        subprocess.run([*cmd, commit], check=True, capture_output=True)
        with tarfile.open(tar) as tf:
            return [m.name.rstrip("/") for m in tf.getmembers() if not m.isdir()]


def check(
    committed: Sequence[str],
    archived: Sequence[str],
    excluded: Sequence[str] = EXCLUDED,
    kept: Sequence[str] = KEPT,
) -> list[str]:
    problems = []
    in_archive = set(archived)
    for prefix in excluded:
        if not any(_under(f, prefix) for f in committed):
            problems.append(f"{prefix}: not in the commit (renamed? update EXCLUDED)")
        leaked = sorted(f for f in in_archive if _under(f, prefix))
        if leaked:
            problems.append(f"{prefix}: {len(leaked)} file(s) in the archive, e.g. {leaked[0]}")
    for prefix in kept:
        if not any(_under(f, prefix) for f in in_archive):
            problems.append(f"{prefix}: missing from the archive")
    expected = {f for f in committed if not any(_under(f, p) for p in excluded)}
    for name in sorted(expected - in_archive):
        problems.append(f"{name}: in the commit but left out of the archive")
    for name in sorted(in_archive - set(committed)):
        problems.append(f"{name}: in the archive but not in the commit")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--worktree-attributes",
        action="store_true",
        help="use the working tree's .gitattributes, not the commit's",
    )
    args = parser.parse_args(argv)
    committed = commit_files(args.root, args.commit)
    archived = archive_files(args.root, args.commit, args.worktree_attributes)
    problems = check(committed, archived)
    left_out = len(committed) - len(archived)
    print(
        f"{args.commit}: {len(committed)} files in the commit, {len(archived)} in the archive, "
        f"{left_out} left out by export-ignore"
    )
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
