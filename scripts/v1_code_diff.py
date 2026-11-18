"""What changed in the v1 code at the end of its freeze (Phase 20 re-run for v1, item 3).

    python scripts/v1_code_diff.py [--base phase14-eval] [--head v1-code] \
        [--out reports/export/v1-code/code_diff.json]

The v1 code (`glypharis.efficiency.v1.V1_PATHS`) was evaluated at ``phase14-eval``. The commit
tagged ``v1-code`` corrects its docstrings and changes one default, the bundle name of
`GlypharisOCR.from_pretrained`. This proves that nothing else changed: every file that differs
between the two revisions under those paths must be a Python file whose syntax tree, with the
docstrings of the module, classes and functions removed, equals the base's, apart from the
registered change in `ALLOWED`. Comments are not in the tree, so a changed comment passes too.
A file added, removed, not Python, not a regular file or with another mode fails, and so does one
that does not parse. Both revisions are read from git as bytes, never from the working tree, and
parsed as bytes, so a file is decoded as Python decodes it (a coding comment counts). The report
lists each file with what changed; it is written once. Exit status 0 when the check passes, 1
otherwise.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

from run_baselines import git_state

from glypharis.efficiency.files import write_json_atomic
from glypharis.efficiency.v1 import V1_PATHS, V1_TAG
from glypharis.pipeline.frozen import PIN_TAG

OUT: Final = Path("reports/export/v1-code/code_diff.json")
# file -> (function, the default before, the default after): the one registered code change.
ALLOWED: Final = {
    "glypharis/pipeline/ocr.py": ("from_pretrained", "glypharis-0.1.0", "glypharis-v1"),
}


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _blob(root: Path, oid: str) -> bytes:
    done = subprocess.run(
        ["git", "-C", str(root), "cat-file", "blob", oid], capture_output=True, check=False
    )
    if done.returncode != 0:
        raise SystemExit(f"git cat-file blob {oid} failed: {done.stderr.decode(errors='replace')}")
    return done.stdout


def strip_docstrings(tree: ast.AST) -> ast.AST:
    """``tree`` without the docstrings of its module, classes and functions (an emptied body gets
    a ``pass``, so a function that was only a docstring stays valid)."""
    kinds = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, kinds) and ast.get_docstring(node, clean=False) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return tree


def apply_allowed(tree: ast.AST, function: str, before: str, after: str) -> int:
    """Set the string default ``before`` of each ``function`` to ``after``; returns the count."""
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == function:
            for d in node.args.defaults + [k for k in node.args.kw_defaults if k is not None]:
                if isinstance(d, ast.Constant) and d.value == before:
                    d.value = after
                    n += 1
    return n


def compare(path: str, base: str | bytes, head: str | bytes) -> dict[str, Any]:
    """How the Python source ``head`` differs from ``base``: ``ok``, and what changed."""
    try:
        a, b = strip_docstrings(ast.parse(base)), strip_docstrings(ast.parse(head))
    except (SyntaxError, ValueError) as exc:
        return {"file": path, "ok": False, "change": f"does not parse: {exc}"}
    if ast.dump(a) == ast.dump(b):
        return {"file": path, "ok": True, "change": "docstrings or comments only"}
    if path in ALLOWED:
        function, before, after = ALLOWED[path]
        if apply_allowed(a, function, before, after) == 1 and ast.dump(a) == ast.dump(b):
            return {
                "file": path,
                "ok": True,
                "change": "docstrings or comments, and the registered default",
                "default": {"function": function, "from": before, "to": after},
            }
    return {"file": path, "ok": False, "change": "code differs"}


def check(root: Path, base: str, head: str) -> dict[str, Any]:
    raw = _git(
        root, "diff", "--raw", "-z", "--no-abbrev", "--no-renames", base, head, "--", *V1_PATHS
    ).split("\0")[:-1]
    files: list[dict[str, Any]] = []
    for meta, path in zip(raw[::2], raw[1::2], strict=True):
        old_mode, new_mode, old_id, new_id, status = meta[1:].split(" ")
        regular = old_mode == new_mode == "100644"
        if status != "M" or not regular or not path.endswith(".py"):
            change = f"status {status}, mode {old_mode} to {new_mode}, not a modified Python file"
            files.append({"file": path, "ok": False, "change": change})
            continue
        files.append(compare(path, _blob(root, old_id), _blob(root, new_id)))
    return {
        "base": base,
        "base_commit": _git(root, "rev-parse", f"{base}^{{commit}}").strip(),
        "head": head,
        "head_commit": _git(root, "rev-parse", f"{head}^{{commit}}").strip(),
        "paths": list(V1_PATHS),
        "files": files,
        "pass": all(f["ok"] for f in files),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=PIN_TAG)
    parser.add_argument("--head", default=V1_TAG)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out.as_posix()} exists: the check is written once")
    res = check(args.root, args.base, args.head)
    what = "the v1 code at the end of its freeze against the code Phase 14 evaluated"
    write_json_atomic(args.out, {"what": what, **res, **git_state()})
    print(
        f"{args.base} to {args.head}: {len(res['files'])} files differ; pass: {res['pass']}; "
        f"written to {args.out.as_posix()}"
    )
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
