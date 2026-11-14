"""Run HierText's official evaluator, line level and end to end, on baseline-format predictions.

Phase 14 reports HierText's own score next to the project's metrics (PHASE_STATUS Phase 14, item
6). The evaluator is pinned at commit 70b6620b (`data/manifests/evaluators.json`) and runs in its
own environment, `.venv-hiertext`, because it needs Apache Beam and an old OpenCV that the main
environment does not carry. This module only converts and runs; it does not call the test-access
guard, so the caller (GW-E2E-009) logs the scoring rows before it scores test pages.

The mode is `--eval_lines --e2e` with `--mask_stride=1` (full-resolution masks, the setting the
evaluator's README requires for reported results) and one worker. In that mode the evaluator builds
each line's mask from the line's words and compares whole-line texts, so every predicted line is
handed over as a line holding one word that carries the line's polygon and text. The word-level
block of the evaluator's output is parsed too, but it compares those one-word lines with real
ground-truth words and is not a score of anything Phase 14 reports.

The evaluator environments live in the main checkout, beside `.venv`, so worktrees share them; set
`GLYPHARIS_EVAL_VENV_ROOT` to the folder that holds them to override that.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVALUATORS = Path("data/manifests/evaluators.json")
VENV_ROOT_ENV = "GLYPHARIS_EVAL_VENV_ROOT"
HIERTEXT_DIR = Path("data/raw/evaluators/hiertext-70b6620b")
HIERTEXT_VENV = ".venv-hiertext"
HIERTEXT_FLAGS = ("--eval_lines", "--e2e", "--mask_stride=1", "--num_workers=1")
SCORES_NAME = "scores"
# Environment variables that would let the main environment leak into the evaluator's.
_DROP_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP")


# --------------------------------------------------------------------------- environments


def main_checkout(start: Path = ROOT) -> Path:
    """The main checkout of the repository that `start` belongs to (itself when not a worktree).

    Worktrees share one `.git` directory, whose parent is the main checkout; the evaluator
    environments are created there once.
    """
    out = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=start,
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip()).parent


def venv_root() -> Path:
    """The folder holding the evaluator environments: `GLYPHARIS_EVAL_VENV_ROOT`, else the main
    checkout."""
    env = os.environ.get(VENV_ROOT_ENV)
    return Path(env) if env else main_checkout()


def venv_python(name: str, root: Path | None = None) -> Path:
    """The interpreter of evaluator environment `name`; FileNotFoundError when it is absent."""
    base = (root if root is not None else venv_root()) / name
    exe = base / "Scripts" / "python.exe" if sys.platform == "win32" else base / "bin" / "python"
    if not exe.is_file():
        raise FileNotFoundError(f"evaluator environment {name} not found under {base.parent.name}")
    return exe


def venv_available(name: str) -> bool:
    """Whether environment `name` exists (tests skip the evaluator runs when it does not)."""
    try:
        venv_python(name)
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return False
    return True


def evaluator_info(name: str, root: Path = ROOT) -> dict[str, Any]:
    """The pinned entry for evaluator `name` from `data/manifests/evaluators.json`, for the
    caller's provenance record."""
    entries = json.loads((root / EVALUATORS).read_text(encoding="utf-8"))
    if name not in entries:
        raise KeyError(f"{name} is not pinned in {EVALUATORS.as_posix()}")
    return {"name": name, **entries[name]}


def evaluator_env() -> dict[str, str]:
    """The environment for an evaluator subprocess: UTF-8 I/O and nothing from the main venv."""
    env = {k: v for k, v in os.environ.items() if k not in _DROP_ENV}
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_evaluator(cmd: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run an evaluator command; RuntimeError with the end of its output when it fails."""
    proc = subprocess.run(
        list(cmd),
        cwd=cwd,
        env=evaluator_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-3000:]
        raise RuntimeError(f"evaluator exited with {proc.returncode}:\n{tail}")
    return proc


# --------------------------------------------------------------------------- conversion


def _vertices(polygon: Any, where: str) -> list[list[int]]:
    pts = [[round(float(p[0])), round(float(p[1]))] for p in polygon]
    if len(pts) < 3:
        # The evaluator draws every polygon as a filled mask and as a shapely polygon; fewer than
        # three points is neither, and dropping the line silently would change the score.
        raise ValueError(f"{where}: a line polygon needs at least 3 points, got {len(pts)}")
    return pts


def to_hiertext_predictions(
    records: Iterable[dict[str, Any]], image_ids: Iterable[str] | None = None
) -> dict[str, Any]:
    """Baseline-format end-to-end records -> the evaluator's prediction JSON.

    `records` are `{"key": image_id, "lines": [{"polygon", "text", ...}]}` as written by
    `run_baselines.py` and `ext_eval.py`. Each page becomes one paragraph and each line one line
    holding one word with the line's polygon (rounded to integer pixels) and text; a page with no
    lines gets no paragraph. Lines keep the system's order. When `image_ids` is given, pages in
    it without a record get an empty entry (the system read nothing there) and a record outside
    it is an error, so the output covers exactly the ground-truth images.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = str(rec["key"])
        if key in by_id:
            raise ValueError(f"page {key} appears twice in the predictions")
        lines = []
        for i, ln in enumerate(rec.get("lines") or []):
            text = str(ln.get("text") or "")
            verts = _vertices(ln["polygon"], f"page {key} line {i}")
            lines.append({"text": text, "words": [{"vertices": verts, "text": text}]})
        by_id[key] = {"image_id": key, "paragraphs": [{"lines": lines}] if lines else []}
    if image_ids is not None:
        wanted = list(dict.fromkeys(image_ids))
        extra = sorted(set(by_id) - set(wanted))
        if extra:
            raise ValueError(
                f"{len(extra)} predicted pages are not in the ground truth, e.g. {extra[0]}"
            )
        return {"annotations": [by_id.get(i, {"image_id": i, "paragraphs": []}) for i in wanted]}
    return {"annotations": list(by_id.values())}


def write_predictions(predictions: dict[str, Any], out: Path) -> None:
    """Write the prediction JSON as the evaluator reads it (plain JSON, UTF-8)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- running


def _read_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def image_ids(gt_json: Path) -> list[str]:
    """The image ids of a HierText ground-truth file (JSON or JSON.gz), in file order."""
    return [str(a["image_id"]) for a in _read_json(gt_json)["annotations"]]


def parse_scores(text: str) -> dict[str, dict[str, float]]:
    """Parse the evaluator's text output (`========= line =========` headers, `key: value` rows)
    into `{"word": {...}, "line": {...}}`."""
    out: dict[str, dict[str, float]] = {}
    block: dict[str, float] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("=") and line.endswith("="):
            name = line.strip("=").strip()
            block = out.setdefault(name, {})
            continue
        if block is None or ":" not in line:
            raise ValueError(f"unexpected line in the evaluator output: {line!r}")
        key, value = line.rsplit(":", 1)
        block[key.strip()] = float(value)
    return out


def run_hiertext(gt_json: Path, pred_json: Path, out_dir: Path) -> dict[str, dict[str, float]]:
    """Score `pred_json` against `gt_json` with the pinned evaluator; returns its word and line
    blocks (`Det-*` and `E2E-*` keys as the evaluator names them).

    The predictions must cover exactly the ground-truth images (the evaluator fails on a missing
    page and on an unknown one); use `to_hiertext_predictions(..., image_ids=...)` to pad. A
    gzipped ground truth is unpacked into `out_dir` for the run and removed afterwards. The
    evaluator writes its sharded text output to `out_dir/scores-*-of-*`, which stays there; a
    previous output in `out_dir` stops the run rather than being read by mistake.
    """
    gt_ids = image_ids(gt_json)
    pred_ids = [str(a["image_id"]) for a in _read_json(pred_json)["annotations"]]
    if len(set(pred_ids)) != len(pred_ids):
        raise ValueError("a page appears twice in the predictions")
    missing, extra = set(gt_ids) - set(pred_ids), set(pred_ids) - set(gt_ids)
    if missing or extra:
        raise ValueError(
            f"predictions do not cover the ground truth: {len(missing)} pages missing, "
            f"{len(extra)} not in the ground truth"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if list(out_dir.glob(f"{SCORES_NAME}-*")):
        raise FileExistsError(f"{out_dir.name} already holds evaluator output; use a fresh folder")
    python = venv_python(HIERTEXT_VENV)
    code = ROOT / HIERTEXT_DIR
    unpacked = None
    gt_path = gt_json.resolve()
    if gt_json.suffix == ".gz":
        unpacked = out_dir / "_gt_unpacked.json"
        with gzip.open(gt_json, "rb") as src, unpacked.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        gt_path = unpacked.resolve()
    try:
        cmd = [
            str(python),
            "eval.py",
            f"--gt={gt_path}",
            f"--result={pred_json.resolve()}",
            f"--output={(out_dir / SCORES_NAME).resolve()}",
            *HIERTEXT_FLAGS,
        ]
        run_evaluator(cmd, code)
    finally:
        if unpacked is not None:
            unpacked.unlink(missing_ok=True)
    shards = sorted(out_dir.glob(f"{SCORES_NAME}-*-of-*"))
    if not shards:
        raise RuntimeError("the evaluator finished but wrote no score file")
    scores = parse_scores("\n".join(s.read_text(encoding="utf-8") for s in shards))
    if "line" not in scores or "word" not in scores:
        raise RuntimeError(f"the evaluator output lacks a block: found {sorted(scores)}")
    return {"word": scores["word"], "line": scores["line"]}
