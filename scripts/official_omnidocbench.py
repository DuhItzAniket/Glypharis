"""Run OmniDocBench's official end-to-end evaluation (text and reading order only) on page outputs.

Phase 14 reports, next to the project's order-aware page CER on OmniDocBench (not the official
protocol), the official code's `text_block` and `reading_order` normalised edit distances
(PHASE_STATUS Phase 14, item 6). The code is pinned at commit 7ddb8df
(`data/manifests/evaluators.json`) and runs in `.venv-omnidocbench` with `PYTHONUTF8=1`, with
`quick_match`, 4 match workers and the formula and table metrics switched off (the systems here
output neither, so the Overall score is not computable). This module only converts and runs; the
caller (GW-E2E-009) logs the scoring rows before it scores test pages.

The official protocol reads one Markdown file per page. A system's lines become that file in the
system's own output order, one line per Markdown paragraph: the lines joined by a blank line, each
written as recognised. Nothing is escaped, as for any plain-text OCR output given to OmniDocBench:
escaping would add characters the edit distance counts, and a line that happens to look like
Markdown (a `#` heading, a `|` table row, `$...$`) is parsed the way the evaluator parses any
submission.

The evaluator writes into `<its folder>/result/`, named after the prediction folder. A run refuses
to start when files of that name are already there, then moves its own result files into the work
folder, so the pinned evaluator folder is left as it was.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from official_hiertext import ROOT, run_evaluator, venv_python

OMNIDOCBENCH_DIR = Path("data/raw/evaluators/omnidocbench-7ddb8df")
OMNIDOCBENCH_VENV = ".venv-omnidocbench"
PIPELINE_CONFIG = Path("configs/end2end.yaml")  # inside the evaluator folder
MATCH_METHOD = "quick_match"
MATCH_WORKERS = 4
ELEMENTS = ("text_block", "reading_order")
PAGE_KEY_PREFIX = "odb-"  # Page.doc_id is "odb-<image stem>" (scripts/prepare_benchmarks.py)


# --------------------------------------------------------------------------- conversion


def to_markdown(lines: Iterable[str]) -> str:
    """A page's lines, in the order given, as Markdown: one paragraph per line, joined by a blank
    line. Line breaks inside a line become spaces and empty lines are left out, so the paragraph
    count equals the number of non-empty lines."""
    paras = [" ".join(str(t).split("\n")).strip() for t in lines]
    return "\n\n".join(p for p in paras if p) + "\n"


def page_stem(record: dict[str, Any]) -> str:
    """The image stem a record belongs to: from its `image` field when present, else its key
    without the `odb-` prefix."""
    if record.get("image"):
        return Path(str(record["image"])).stem
    key = str(record["key"])
    if not key.startswith(PAGE_KEY_PREFIX):
        raise ValueError(f"record key {key!r} is not an OmniDocBench page key")
    return key[len(PAGE_KEY_PREFIX) :]


def write_predictions(records: Iterable[dict[str, Any]], out_dir: Path) -> None:
    """Write `<image stem>.md` per page into `out_dir`, which must hold no Markdown yet (a stale
    file would be scored as a prediction). Pages with no lines get an empty file, so every page
    the system ran on is present."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.md")):
        raise FileExistsError(f"{out_dir.name} already holds predictions; use a fresh folder")
    seen: set[str] = set()
    for rec in records:
        stem = page_stem(rec)
        if stem in seen:
            raise ValueError(f"page {stem} appears twice in the predictions")
        seen.add(stem)
        text = to_markdown(str(ln.get("text") or "") for ln in rec.get("lines") or [])
        (out_dir / f"{stem}.md").write_text(text, encoding="utf-8", newline="\n")


def filtered_ground_truth(gt_json: Path, keep_stems: set[str], out: Path) -> None:
    """Write the pages of the official ground truth whose image stem is in `keep_stems`, unchanged
    and in file order. Every kept stem must be found."""
    pages = json.loads(gt_json.read_text(encoding="utf-8"))
    kept = [p for p in pages if Path(p["page_info"]["image_path"]).stem in keep_stems]
    found = {Path(p["page_info"]["image_path"]).stem for p in kept}
    if found != keep_stems:
        missing = sorted(keep_stems - found)
        raise ValueError(f"{len(missing)} pages are not in the ground truth, e.g. {missing[0]}")
    if len(kept) != len(found):
        raise ValueError("the ground truth holds an image stem twice")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- running


def save_name(pred_dir: Path) -> str:
    """The prefix the evaluator gives its result files (`build_save_name` in its pipeline)."""
    return f"{pred_dir.resolve().name}_{MATCH_METHOD}"


def _rel(path: Path, start: Path) -> str:
    # The evaluator runs with its own folder as the working directory; paths in the config are
    # relative to it so no absolute path is written to disk.
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def write_config(gt_json: Path, pred_dir: Path, work_dir: Path, code: Path) -> Path:
    """Write the run's config to `work_dir/config.yaml`: the pinned `end2end.yaml` with only
    `text_block` and `reading_order` (Edit_dist), `quick_match`, 4 match workers, and the data
    paths replaced; every other dataset setting (timeouts, fallbacks) stays as pinned."""
    pinned = yaml.safe_load((code / PIPELINE_CONFIG).read_text(encoding="utf-8"))
    dataset = dict(pinned["end2end_eval"]["dataset"])
    dataset["ground_truth"] = {"data_path": _rel(gt_json, code)}
    dataset["prediction"] = {"data_path": _rel(pred_dir, code)}
    dataset["match_method"] = MATCH_METHOD
    dataset["match_workers"] = MATCH_WORKERS
    cfg = {
        "end2end_eval": {
            "metrics": {el: {"metric": ["Edit_dist"]} for el in ELEMENTS},
            "dataset": dataset,
        }
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "config.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8", newline="\n")
    return out


def _own_results(result_dir: Path, name: str) -> list[Path]:
    return sorted(p for p in result_dir.glob(f"{name}_*") if p.name.startswith(f"{name}_"))


def collect_results(code: Path, name: str, work_dir: Path) -> list[str]:
    """Move the evaluator's result files for run `name` from `<code>/result/` into
    `work_dir/result/`; returns the moved file names."""
    dest = work_dir / "result"
    dest.mkdir(parents=True, exist_ok=True)
    moved = []
    for src in _own_results(code / "result", name):
        target = dest / src.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(src), str(target))
        moved.append(src.name)
    return moved


def _num(x: Any) -> float:
    return float(x) if x is not None else float("nan")


def parse_results(result_dir: Path, name: str) -> dict[str, Any]:
    """The text-block and reading-order edit distances of run `name`.

    Per element: `ALL_page_avg` (mean over pages of each page's edit distance; the leaderboard
    figure), `edit_whole` (all edits over all characters) and `edit_sample_avg` (mean over matched
    blocks), the evaluator's page-attribute split, and `per_page` (image name -> edit distance)
    when the evaluator wrote it. Plus `match_debug` (page count and timeout fallbacks), which says
    whether quick_match gave up on any page.
    """
    metric = json.loads((result_dir / f"{name}_metric_result.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for el in ELEMENTS:
        block = metric.get(el) or {}
        edit = (block.get("all") or {}).get("Edit_dist") or {}
        entry: dict[str, Any] = {
            k: _num(edit.get(k)) for k in ("ALL_page_avg", "edit_whole", "edit_sample_avg")
        }
        entry["by_attribute"] = (block.get("page") or {}).get("Edit_dist", {})
        per_page = result_dir / f"{name}_{el}_per_page_edit.json"
        if per_page.is_file():
            entry["per_page"] = json.loads(per_page.read_text(encoding="utf-8"))
        out[el] = entry
    out["match_debug"] = metric.get("match_debug", {})
    return out


def run_omnidocbench(gt_json: Path, pred_dir: Path, work_dir: Path) -> dict[str, Any]:
    """Score the Markdown files in `pred_dir` against `gt_json` (a JSON list of OmniDocBench
    pages, usually written by `filtered_ground_truth`) with the pinned evaluator.

    Every ground-truth page must have a prediction file (the evaluator would silently score a
    missing one as an empty page); extra files are ignored by the evaluator. The config and the
    evaluator's result files end up in `work_dir`. Returns `parse_results` plus `pages` and
    `save_name`.
    """
    pages = json.loads(gt_json.read_text(encoding="utf-8"))
    images = [Path(p["page_info"]["image_path"]) for p in pages]
    odd = [i.name for i in images if len(i.suffix) != 4]
    if odd:
        # The evaluator finds a page's file by cutting four characters off the image name, which
        # equals the stem only for three-letter extensions.
        raise ValueError(f"{len(odd)} images do not have a three-letter extension, e.g. {odd[0]}")
    stems = [i.stem for i in images]
    missing = [s for s in stems if not (pred_dir / f"{s}.md").is_file()]
    if missing:
        raise ValueError(f"{len(missing)} ground-truth pages have no prediction, e.g. {missing[0]}")
    code = ROOT / OMNIDOCBENCH_DIR
    name = save_name(pred_dir)
    if _own_results(code / "result", name):
        raise FileExistsError(
            f"the evaluator's result folder already holds {name}_* files; move them away first"
        )
    python = venv_python(OMNIDOCBENCH_VENV)
    config = write_config(gt_json, pred_dir, work_dir, code)
    try:
        run_evaluator([str(python), "pdf_validation.py", "--config", _rel(config, code)], code)
    finally:
        collect_results(code, name, work_dir)
    result = parse_results(work_dir / "result", name)
    return {"pages": len(stems), "save_name": name, **result}
