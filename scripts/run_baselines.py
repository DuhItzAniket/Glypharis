"""Run an external OCR system on an evaluation set and write its predictions (Phases 5 and 14).

Runs inside .venv-baselines (ADR-0006); scoring happens separately in the main environment, so
every system is scored by the same code: Phase 5 by `scripts/evaluate_baselines.py --phase 5`, and
Phase 14 once, by `scripts/phase14_score.py score` (GW-E2E-009).

    python scripts/run_baselines.py --phase 5 --system tesseract --task rec \
        --eval-set hiertext_val --experiment GW-BASE-001 --reason "Phase 5 baseline"
    python scripts/run_baselines.py --phase 5 --system tesseract --task rec \
        --eval-set hiertext_val --experiment GW-BASE-001 --reason "Phase 5 baseline" --profile
    python scripts/run_baselines.py --phase 14 --system ppocr_server --task e2e \
        --eval-set omnidocbench_test --subset --experiment GW-BASE-003 --reason "Phase 14 baseline"

Systems: tesseract, ppocr_mobile, ppocr_server, ppocr6_tiny, ppocr6_small, ppocr6_medium, trocr
(recognition only). Each system has one experiment ID (`EXPERIMENTS`); a mismatch is refused.
Tasks: rec (ground-truth line crops in, text out) and e2e (page in, lines with polygons and text
out; detection metrics come from the same output).
Evaluation sets (`glypharis.eval.sets.EVAL_SETS`): Phase 5 hiertext_val, glyphsynth_val (rec
only), hiertext_test, gdc_test; Phase 14 iiit5k_test (rec only), funsd_test, gdc_scan_test,
omnidocbench_test (e2e only).

Prediction runs (default). Every sample of the set goes through the system and is appended to
data/processed/baselines/<experiment>/<system>_<task>_<eval-set>.jsonl as soon as it is done, so an
interrupted run resumes where it stopped. The run's .meta.json is written when the run starts and
updated when it ends. On a test set the test-access guard logs the access before any image is
decoded or passed to a system (the sample keys are read first, to tell whether the run is already
complete).

Profiling runs (--profile). Speed and memory are measured separately from the long prediction runs,
which share the machine with whatever else is happening over hours. A profiling run times
`WARMUP` samples it discards and then `PROFILE_SAMPLES` samples one at a time (batch 1), records
the peak working set of the process and of any child process (tesseract.exe), the conditions
EXPERIMENT_PROTOCOL §3 asks for, and when the sitting ran (``started_utc`` just before the first
warm-up sample, ``ended_utc`` after the last timed one). Validation sets only; the result goes to
profile_<system>_<task>_<eval-set>.json next to the predictions.

Recognition crops always come from line packs cut by `scripts/build_eval_crops.py` (GlyphSynth uses
its Phase 4 pack). Validation sets are subsampled with a fixed seed to keep CPU time bounded; test
sets are used in full. Each system runs with its defaults and English selected; nothing is tuned per
benchmark (EXPERIMENT_PROTOCOL §10). Deviations from defaults are listed in `settings()`.

Phase 14 (`--phase 14`, docs/PHASE_STATUS.md, Phase 14 item 4) runs the same systems with the same
settings on the new sets only, as registered in `PHASE14_PLAN` (HierText test and GDC are reused
from Phase 5, not run again). It differs from Phase 5 in what it refuses and where it writes:

- it starts only on branch `phase/14-external-eval`;
- predictions and profiles go under data/processed/baselines_phase14/<experiment>/, never under
  Phase 5's data/processed/baselines/, which it only reads;
- before a new-set run, and before the access is logged, every field of the system's `settings()`
  must equal the settings recorded in that system's Phase 5 run metadata for the same task (the
  ``settings`` field of its *.meta.json and profile files; never their predictions), and its
  inference batch must equal every ``inference_batch`` those Phase 5 runs recorded (TrOCR's is
  ``--batch``; the others fix it at one image at a time, `check_phase5_batch`);
- before the access is logged, the page file and line pack the run reads are hashed and compared
  with the preparation's record (`glypharis.eval.inputs.check_inputs`, bytes only; an input
  without a record is refused, `REQUIRE_RECORDED`), and a resume must read the same inputs;
- a profile asserts PaddleOCR's 10 CPU threads (H1′'s latency half);
- OmniDocBench end to end with `--subset` reads only the 100 pages of
  data/manifests/processed/omnidocbench_subset100.json, and its file name says so.

Phase 16 (`--phase 16`, docs/PHASE_STATUS.md, Phase 16 item 6) is a dated amendment to GW-BASE-002:
one fresh validation profile for H1′'s ONNX latency sitting (GW-EFF-010), nothing else.

    python scripts/run_baselines.py --phase 16 --system ppocr_mobile --task e2e \
        --eval-set hiertext_val --experiment GW-BASE-002 --reason "GW-EFF-010 sitting" --profile

- it starts only on branch `PHASE16_BRANCH`, with ``--profile``, and only for `PHASE16_PROFILE`
  (PP-OCRv5 mobile end to end on the HierText validation profile pages);
- the profile goes to data/processed/baselines_phase16/GW-BASE-002/; the Phase 5 and Phase 14
  folders are only read;
- every field of the system's `settings()` must equal the ``settings`` block of the Phase 14
  profile of the same system, task and set (`check_phase14_profile_settings`; no other field of
  that file is used), and PaddleOCR's 10 CPU threads are asserted as in Phase 14;
- the profile also records the machine conditions and the GPU's state (nvidia-smi) at its end, and
  the GPU's state at its start, as every Phase 16 profile does (Phase 16 item 3).

The access log is the repository's research/test_access_log.md, found from the repository root.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from glypharis.data.linepack import LinePackReader
from glypharis.data.pages import Page, open_page_image, read_pages
from glypharis.eval.access import open_test_access
from glypharis.eval.inputs import check_inputs
from glypharis.eval.sets import (
    EVAL_SETS,
    PHASE5_HASHES,
    PHASE14_BRANCH,
    PHASE14_SETS,
    SUBSET_NAME,
    access_log,
    require_branch,
    require_phase5_hashes,
    subset_pages,
)
from glypharis.utils.gitinfo import git_info

MODELS = Path("data/raw/baseline_models")
ACCESS_LOG = Path("research/test_access_log.md")
REC_VAL_SAMPLE = 10_000  # crops per validation set
E2E_VAL_PAGES = 400  # pages per validation set
SEED = 5
TASKS = ("rec", "e2e")
WARMUP = 10  # profiling: samples run and discarded before timing (EXPERIMENT_PROTOCOL §3)
PROFILE_SAMPLES = 100  # profiling: timed samples
PROFILE_SETS = {"rec": ("hiertext_val", "glyphsynth_val"), "e2e": ("hiertext_val",)}
THREAD_ENV = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS")

# The evaluation sets live in glypharis.eval.sets; EVAL_SETS stays importable from here.
PHASES = (5, 14, 16)
PHASE5_ROOT = "baselines"  # under data/processed
PHASE14_ROOT = "baselines_phase14"  # under data/processed; Phase 5's files are only read
PROFILE_THREADS = 10  # PaddleOCR's default CPU threads, asserted for Phase 14 profiles (H1′)
PHASE16_ROOT = "baselines_phase16"  # under data/processed; Phase 5's and 14's files are only read
PHASE16_BRANCH = "phase/16-efficiency"
# The one run Phase 16 makes: GW-BASE-002's page profile for H1′'s ONNX sitting (GW-EFF-010).
PHASE16_PROFILE = ("ppocr_mobile", "e2e", "hiertext_val")
# Every input of a Phase 14 run must have a recorded hash, and every pack a recorded index hash
# (fail closed, `glypharis.eval.inputs.check_inputs`); the tests' tiny sets have none.
REQUIRE_RECORDED = True
# Tesseract's image and text files go through this folder under the data root (data/cache is
# git-ignored), never the system temp folder (`tesseract_tmp`).
TESSERACT_TMP = Path("cache/tesseract_tmp")

EXPERIMENTS = {"tesseract": "GW-BASE-001", "ppocr_mobile": "GW-BASE-002",
               "ppocr_server": "GW-BASE-003", "trocr": "GW-BASE-004",
               "ppocr6_tiny": "GW-BASE-005", "ppocr6_small": "GW-BASE-006",
               "ppocr6_medium": "GW-BASE-007"}  # fmt: skip
SYSTEM_NAMES = {
    "tesseract": "Tesseract 5",
    "ppocr_mobile": "PP-OCRv5 mobile",
    "ppocr_server": "PP-OCRv5 server",
    "trocr": "TrOCR-small-printed",
    "ppocr6_tiny": "PP-OCRv6 tiny",
    "ppocr6_small": "PP-OCRv6 small",
    "ppocr6_medium": "PP-OCRv6 medium",
}


_NO_DET = "not applicable: TrOCR has no detector"
# Phase 14 coverage, fixed with the plan (docs/PHASE_STATUS.md, Phase 14 item 4). None: the whole
# set; SUBSET_NAME: the OmniDocBench subset only; any other string: why it is not run. A run
# outside this table is refused, so no system is evaluated twice on the same pages.
# fmt: off
PHASE14_PLAN: dict[tuple[str, str, str], str | None] = {
    **{(s, "rec", e): None for s in EXPERIMENTS
       for e in ("iiit5k_test", "gdc_scan_test", "funsd_test")},
    **{(s, "e2e", e): None for s in EXPERIMENTS if s != "trocr"
       for e in ("gdc_scan_test", "funsd_test", "omnidocbench_test")},
    **{("trocr", "e2e", e): _NO_DET for e in ("gdc_scan_test", "funsd_test", "omnidocbench_test")},
    ("ppocr_server", "e2e", "gdc_scan_test"): (
        "not run: died on GDC's 3072 x 4080 photos in Phase 5 (research/failures.md, 2026-10-24)"
    ),
    ("ppocr_server", "e2e", "omnidocbench_test"): SUBSET_NAME,
}
# fmt: on


# ----------------------------------------------------------------------------------- samples


def rec_indices(n: int, split: str) -> np.ndarray:
    """Which crops of a pack are scored: all of a test set, a seeded subsample of validation."""
    idx = np.arange(n)
    if split != "test" and n > REC_VAL_SAMPLE:
        idx = np.sort(np.random.default_rng(SEED).choice(n, REC_VAL_SAMPLE, replace=False))
    return idx


def page_indices(n: int, split: str) -> np.ndarray:
    idx = np.arange(n)
    if split != "test" and n > E2E_VAL_PAGES:
        idx = np.sort(np.random.default_rng(SEED).choice(n, E2E_VAL_PAGES, replace=False))
    return idx


def _subset_filter(data: Path, eval_set: str, ids: list[str]) -> list[bool]:
    """Which of ``ids`` (pages-file order) are in the OmniDocBench subset; every subset page must
    be in the pages file, or the subset was drawn from another file."""
    if eval_set != "omnidocbench_test":
        raise SystemExit(f"--subset is defined for omnidocbench_test only, not {eval_set}")
    keep = set(subset_pages(data))
    missing = keep - set(ids)
    if missing:
        raise SystemExit(f"{len(missing)} subset pages are not in {eval_set} "
                         f"(e.g. {sorted(missing)[0]})")  # fmt: skip
    return [i in keep for i in ids]


def expected_keys(data: Path, eval_set: str, task: str, subset: bool = False) -> list[str]:
    """The keys a complete run of ``task`` on ``eval_set`` writes, in run order."""
    spec = EVAL_SETS[eval_set]
    if task == "rec":
        if subset:
            raise SystemExit("--subset applies to end-to-end runs only")
        reader = LinePackReader(data / spec["pack"])
        return [reader.records[int(i)].key for i in rec_indices(len(reader), spec["split"])]
    if not spec["pages"]:
        return []
    ids = [p.doc_id for p in read_pages(data / spec["pages"])]
    ids = [ids[int(i)] for i in page_indices(len(ids), spec["split"])]
    if subset:
        ids = [d for d, k in zip(ids, _subset_filter(data, eval_set, ids), strict=True) if k]
    return ids


def record_splits(data: Path, eval_set: str, task: str) -> set[str]:
    """The split named by the records themselves (crop pack index or page file), so the split
    the access log records comes from the data, not only from the set's table."""
    spec = EVAL_SETS[eval_set]
    if task == "rec":
        return {r.split for r in LinePackReader(data / spec["pack"]).records}
    return {p.split for p in read_pages(data / spec["pages"])}


def rec_samples(data: Path, eval_set: str) -> Iterator[tuple[str, str, Image.Image]]:
    spec = EVAL_SETS[eval_set]
    reader = LinePackReader(data / spec["pack"])
    for i in rec_indices(len(reader), spec["split"]):
        rec = reader.records[int(i)]
        yield rec.key, rec.label, reader.image(int(i))


def e2e_samples(data: Path, eval_set: str, subset: bool = False) -> Iterator[Page]:
    spec = EVAL_SETS[eval_set]
    if not spec["pages"]:
        raise SystemExit(f"{eval_set} has no pages, so end to end is not defined for it")
    pages = list(read_pages(data / spec["pages"]))
    pages = [pages[int(i)] for i in page_indices(len(pages), spec["split"])]
    if subset:
        keep = _subset_filter(data, eval_set, [p.doc_id for p in pages])
        pages = [p for p, k in zip(pages, keep, strict=True) if k]
    yield from pages


# ---------------------------------------------------------------------------------- memory


def peak_working_set(proc: Any) -> int:
    """High-water mark of a process's memory: peak working set on Windows, else current RSS."""
    info = proc.memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def run_tracked(cmd: list[str]) -> tuple[int, int]:
    """Run a command to completion; return its exit code and its peak working set in bytes.

    The child's peak is read while it runs (a finished process can no longer be asked). Output goes
    to files named in ``cmd``, not to pipes, so a large output cannot block the child.
    """
    import psutil

    with tempfile.TemporaryFile() as err:
        proc = psutil.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err)
        peak = 0
        while True:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                peak = max(peak, peak_working_set(proc))
            try:
                proc.wait(timeout=0.005)
                break
            except (subprocess.TimeoutExpired, psutil.TimeoutExpired):
                continue
        if proc.returncode:
            err.seek(0)
            msg = err.read().decode("utf-8", errors="replace").strip()[:300]
            raise RuntimeError(f"{cmd[0]} exited with {proc.returncode}: {msg}")
    return proc.returncode, peak


# ----------------------------------------------------------------------------------- systems


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def tesseract_tmp(data: Path) -> Path:
    """The folder Tesseract's files go through: `TESSERACT_TMP` under the data root.

    Tesseract reads every crop or page from a PNG and writes its text to a file, so the images and
    text of a private set (GDC-scan) pass through this folder, and a run killed mid-call leaves
    them behind: here, inside the data folder, not in the system temp folder. Each process works in
    its own
    ``<pid>-...`` subfolder; at start-up the leftovers of runs whose process is gone are removed
    (a live run's folder is kept, so two runs can share the folder).
    """
    import psutil

    root = data / TESSERACT_TMP
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        pid = child.name.split("-", 1)[0]
        if pid.isdigit() and psutil.pid_exists(int(pid)):
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                child.unlink()
    return root


class Tesseract:
    """Tesseract 5 command line, English, default engine (LSTM) and default traineddata.

    Its files go through ``tmp_root`` (`tesseract_tmp`), which the runner sets before any call;
    without it a call is refused rather than written to the system temp folder."""

    name = "tesseract"
    tasks = TASKS
    tmp_root: Path | None = None

    def __init__(self, task: str) -> None:
        exe = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if not Path(exe).exists():
            raise SystemExit("Tesseract not found: install the UB Mannheim build or put it on PATH")
        self.exe, self.task = exe, task
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True)
        self.version = out.stdout.splitlines()[0].strip()
        self.version_detail = [ln.strip() for ln in out.stdout.splitlines()[1:] if ln.strip()]
        tessdata = Path(os.environ.get("TESSDATA_PREFIX") or Path(exe).parent / "tessdata")
        model = tessdata / "eng.traineddata"
        self.model = {"path": str(model), "bytes": model.stat().st_size,
                      "sha256": sha256_file(model)}  # fmt: skip
        self.child_peak = 0

    def model_bytes(self) -> int:
        return int(self.model["bytes"])

    def settings(self) -> dict[str, Any]:
        common = {"lang": "eng", "oem": "default (LSTM)", "traineddata": self.model,
                  "build": self.version_detail,
                  "OMP_THREAD_LIMIT": os.environ.get("OMP_THREAD_LIMIT")}  # fmt: skip
        if self.task == "rec":
            # psm 7 treats the image as one text line. One process reads a list file of crops.
            return {**common, "psm": 7, "invocation": "one process per batch, via a list file"}
        return {**common, "psm": 3, "invocation": "one process per page, TSV output"}  # default

    def inference_batch(self, runner_batch: int) -> int:
        return 1  # Tesseract reads the listed images one after another

    def _run(self, args: list[str]) -> None:
        _, peak = run_tracked([self.exe, *args])
        self.child_peak = max(self.child_peak, peak)

    def _tmp(self) -> tempfile.TemporaryDirectory[str]:
        if self.tmp_root is None:
            raise RuntimeError("Tesseract's folder under the data root is not set (tesseract_tmp)")
        return tempfile.TemporaryDirectory(prefix=f"{os.getpid()}-", dir=self.tmp_root)

    def recognize(self, images: list[Image.Image]) -> list[str]:
        """One Tesseract process per batch, over a list file; results are split on form feeds
        (checked: N images give N - 1 separators, and blank images keep their slot)."""
        with self._tmp() as tmp:
            paths = []
            for i, img in enumerate(images):
                path = Path(tmp) / f"{i:05d}.png"
                img.save(path)
                paths.append(str(path))
            listing = Path(tmp) / "list.txt"
            listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
            self._run([str(listing), str(Path(tmp) / "out"), "-l", "eng", "--psm", "7"])
            text = (Path(tmp) / "out.txt").read_bytes().decode("utf-8", errors="replace")
        texts = text.split("\f")
        if len(texts) < len(images):
            raise RuntimeError(f"Tesseract returned {len(texts)} results for {len(images)} images")
        return [t.strip() for t in texts[: len(images)]]

    def page(self, img: Image.Image) -> list[dict[str, Any]]:
        with self._tmp() as tmp:
            path = Path(tmp) / "page.png"
            img.save(path)
            self._run([str(path), str(Path(tmp) / "out"), "-l", "eng", "--psm", "3", "tsv"])
            tsv = (Path(tmp) / "out.tsv").read_bytes().decode("utf-8", errors="replace")
        return tesseract_tsv_lines(tsv)

    def invocation_overhead_ms(self, repeats: int = 5) -> float:
        """Median wall time of one Tesseract call on a tiny blank image: process start, model load
        and file I/O, which every call through the command line pays."""
        blank = Image.new("L", (32, 32), 255)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            self.recognize([blank])
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times))


def tesseract_tsv_lines(tsv: str) -> list[dict[str, Any]]:
    """Group Tesseract's word rows (level 5) into lines with a bounding box and mean confidence."""
    rows = tsv.splitlines()
    if not rows:
        return []
    header = rows[0].split("\t")
    lines: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows[1:]:
        f = dict(zip(header, row.split("\t"), strict=False))
        if f.get("level") != "5" or not f.get("text", "").strip():
            continue
        key = (int(f["block_num"]), int(f["par_num"]), int(f["line_num"]))
        x, y, w, h = (int(f[k]) for k in ("left", "top", "width", "height"))
        ln = lines.setdefault(key, {"words": [], "box": [x, y, x + w, y + h], "conf": []})
        ln["words"].append(f["text"])
        b = ln["box"]
        ln["box"] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
        ln["conf"].append(float(f["conf"]) / 100)
    out = []
    for _, ln in sorted(lines.items()):
        x0, y0, x1, y1 = ln["box"]
        out.append({"text": " ".join(ln["words"]),
                    "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                    "confidence": round(float(np.mean(ln["conf"])), 4)})  # fmt: skip
    return out


class PPOCR:
    """PaddleOCR 3.x with PP-OCRv5 or PP-OCRv6 models loaded from the pinned local copies.

    PP-OCRv6 medium is what PaddleOCR 3.7 picks by default for English; tiny and small are its
    lighter siblings. The v6 recognizers are multilingual (Chinese, English, Latin scripts).

    Page runs use PaddleOCR's own pipeline config with two changes:

    - document unwarping is off, because its output polygons are in the frame of the unwarped
      image and cannot be scored against the page's ground truth. Orientation classification
      stays on, and polygons from a rotated page are mapped back with `unrotate`;
    - the text detector runs without oneDNN, because Paddle 3.3's oneDNN path crashes on the
      PP-OCRv5 and PP-OCRv6 detectors on this machine, so the default cannot be run at all. The
      plain kernels compute the same network; recognizer outputs were identical with and without
      oneDNN where both run. Every other model keeps the default oneDNN mode.

    Recognition-only runs use the TextRecognition module with its defaults (batch size 1).
    """

    tasks = TASKS
    MODELS: ClassVar[dict[str, tuple[str, str]]] = {
        "ppocr_mobile": ("PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec"),
        "ppocr_server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
        "ppocr6_tiny": ("PP-OCRv6_tiny_det", "PP-OCRv6_tiny_rec"),
        "ppocr6_small": ("PP-OCRv6_small_det", "PP-OCRv6_small_rec"),
        "ppocr6_medium": ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec"),
    }

    def __init__(self, name: str, task: str) -> None:
        # Models are local: skip PaddleX's start-up check of the online model hosts.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        # On Windows, torch's DLLs fail to load after paddle's (paddlex pulls in torch through
        # modelscope), so torch goes first.
        import torch  # noqa: F401

        self.name, self.task = name, task
        import paddleocr
        from paddleocr._constants import DEFAULT_CPU_THREADS

        det, rec = self.MODELS[name]
        self.det_name, self.rec_name = det, rec
        self.cpu_threads = DEFAULT_CPU_THREADS
        self.version = f"paddleocr {paddleocr.__version__}, {det} + {rec}"
        m = MODELS
        if task == "rec":
            self._rec = paddleocr.TextRecognition(
                model_name=rec, model_dir=str(m / rec), device="cpu"
            )
        else:
            from paddlex.inference import load_pipeline_config

            config = load_pipeline_config("OCR")
            config["SubModules"]["TextDetection"]["engine_config"] = {
                "paddle_static": {"run_mode": "paddle"}
            }
            self._pipe = paddleocr.PaddleOCR(
                paddlex_config=config,
                text_detection_model_name=det,
                text_detection_model_dir=str(m / det),
                text_recognition_model_name=rec,
                text_recognition_model_dir=str(m / rec),
                doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
                doc_orientation_classify_model_dir=str(m / "PP-LCNet_x1_0_doc_ori"),
                textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
                textline_orientation_model_dir=str(m / "PP-LCNet_x1_0_textline_ori"),
                # Both flags are given: passing only one switches the whole preprocessor off.
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                device="cpu",
            )

    def settings(self) -> dict[str, Any]:
        common = {"device": "cpu", "cpu_threads": self.cpu_threads, "precision": "fp32"}
        if self.task == "rec":
            return {**common, "module": "TextRecognition", "rec_model": self.rec_name,
                    "batch_size": "module default (1)", "run_mode": "default (oneDNN)"}  # fmt: skip
        return {**common, "module": "PaddleOCR pipeline", "det_model": self.det_name,
                "rec_model": self.rec_name, "use_doc_unwarping": False,
                "use_doc_orientation_classify": True, "use_textline_orientation": True,
                "detector_run_mode": "paddle (oneDNN crashes on this machine)",
                "deviation": "doc unwarping off (it moves polygons)"}  # fmt: skip

    def inference_batch(self, runner_batch: int) -> int:
        return 1  # TextRecognition's default batch size; the pipeline runs one page at a time

    def model_bytes(self) -> int:
        names = [self.rec_name]
        if self.task == "e2e":
            names += [self.det_name, "PP-LCNet_x1_0_doc_ori", "PP-LCNet_x1_0_textline_ori"]
        return sum(f.stat().st_size for n in names for f in (MODELS / n).glob("*") if f.is_file())

    def recognize(self, images: list[Image.Image]) -> list[str]:
        arrays = [np.asarray(im.convert("RGB"))[:, :, ::-1].copy() for im in images]  # BGR
        res = self._rec.predict(input=arrays)
        return [str(r["rec_text"]) for r in res]

    def page(self, img: Image.Image) -> list[dict[str, Any]]:
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        (res,) = self._pipe.predict(arr)
        angle = int(res["doc_preprocessor_res"]["angle"])
        h, w = arr.shape[:2]
        out = []
        for poly, text, score in zip(res["rec_polys"], res["rec_texts"], res["rec_scores"],
                                     strict=True):  # fmt: skip
            pts = [unrotate(float(x), float(y), angle, w, h) for x, y in np.asarray(poly)]
            out.append({"text": str(text), "polygon": pts, "confidence": round(float(score), 4)})
        return out


def unrotate(x: float, y: float, angle: int, w: int, h: int) -> list[float]:
    """Map a point from a page rotated by ``angle`` degrees (counter-clockwise, as PaddleOCR's
    orientation step reports it) back to the original page of size w x h."""
    if angle == 90:  # rotated image is h wide, w high
        return [w - y, x]
    if angle == 180:
        return [w - x, h - y]
    if angle == 270:
        return [y, h - x]
    return [x, y]


class TrOCR:
    """TrOCR-small-printed (Hugging Face), recognition only, CPU.

    Generation is greedy, as in the checkpoint, with two speed or length settings changed:

    - a length limit: the checkpoint sets none, so transformers would stop at 20 tokens and cut
      every long line short. MAX_NEW_TOKENS is well above the longest line in the evaluation sets;
    - the decoder's key/value cache is on. The checkpoint ships ``use_cache: false``, which makes
      every decoding step re-run the whole prefix; greedy outputs are the same either way.

    The checkpoint lacks the encoder pooler weights, which generation does not use.
    """

    MAX_NEW_TOKENS = 256

    name = "trocr"
    tasks = ("rec",)

    def __init__(self, task: str) -> None:
        import torch
        import transformers

        path = MODELS / "trocr-small-printed"
        self.processor = transformers.TrOCRProcessor.from_pretrained(path)
        self.model = transformers.VisionEncoderDecoderModel.from_pretrained(path).eval()
        self.torch = torch
        self.version = f"trocr-small-printed, transformers {transformers.__version__}"

    def settings(self) -> dict[str, Any]:
        gc = self.model.generation_config
        return {"num_beams": gc.num_beams or 1, "max_new_tokens": self.MAX_NEW_TOKENS,
                "use_cache": True, "device": "cpu", "precision": "fp32",
                "torch_threads": self.torch.get_num_threads(),
                "deviation": "max_new_tokens set (default would stop at 20 tokens); "
                             "use_cache on (checkpoint has it off; speed only)"}  # fmt: skip

    def inference_batch(self, runner_batch: int) -> int:
        return runner_batch

    def model_bytes(self) -> int:
        path = MODELS / "trocr-small-printed"
        return sum(f.stat().st_size for f in path.glob("*") if f.is_file())

    def recognize(self, images: list[Image.Image]) -> list[str]:
        pixel = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        with self.torch.inference_mode():
            ids = self.model.generate(
                pixel.pixel_values, max_new_tokens=self.MAX_NEW_TOKENS, use_cache=True
            )
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]


def make_system(name: str, task: str) -> Any:
    if name in PPOCR.MODELS:
        return PPOCR(name, task)
    return {"tesseract": Tesseract, "trocr": TrOCR}[name](task)


def ready(system: Any, data: Path) -> Any:
    """A system made by `make_system`, with Tesseract's file folder set under the data root."""
    if isinstance(system, Tesseract):
        system.tmp_root = tesseract_tmp(data)
    return system


# -------------------------------------------------------------------------------- provenance


def git_state() -> dict[str, Any]:
    """Commit and dirty flag of the code about to run. The test-access log is left out of the
    dirty check: the runner itself appends to it, and it holds no code."""
    info = git_info()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                            capture_output=True, text=True)  # fmt: skip
    changed = [ln for ln in status.stdout.splitlines() if ln.strip()]
    changed = [ln for ln in changed if not ln.endswith(ACCESS_LOG.as_posix())]
    dirty = bool(changed) if status.returncode == 0 else None
    return {"git_commit": info.commit, "git_dirty": dirty}


def _cpu_name() -> str:
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")  # fmt: skip
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (ImportError, OSError):
        return platform.processor()


def cpu_busy_percent(interval: float = 2.0) -> float:
    """System-wide CPU use over ``interval`` seconds: was the machine otherwise idle?"""
    import psutil

    return float(psutil.cpu_percent(interval=interval))


POWER_MODES = {
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "best power efficiency",
    "00000000-0000-0000-0000-000000000000": "balanced",
    "ded574b5-45a0-4f42-8737-46345c09c238": "best performance",
}


def _power_mode(plugged: bool | None) -> str | None:
    """Windows 11 power mode (the slider), which overlays the classic power plan."""
    try:
        import winreg

        path = r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes"
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path)
        name = (
            "ActiveOverlayAcPowerScheme" if plugged is not False else "ActiveOverlayDcPowerScheme"
        )
        guid = str(winreg.QueryValueEx(key, name)[0]).lower()
        return POWER_MODES.get(guid, guid)
    except (ImportError, OSError):
        return None


def run_conditions() -> dict[str, Any]:
    """Machine state that EXPERIMENT_PROTOCOL §3 asks to record with efficiency numbers."""
    import psutil

    battery = psutil.sensors_battery()
    try:
        plan = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                              text=True, timeout=10).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        plan = None
    plugged = battery.power_plugged if battery else None
    return {
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": os.cpu_count(),
        "on_ac_power": battery.power_plugged if battery else None,
        "battery_percent": battery.percent if battery else None,
        "power_plan": plan,
        "power_mode": _power_mode(plugged),
        "cpu_busy_percent_before": cpu_busy_percent(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
        "python": platform.python_version(),
        "os": platform.platform(),
    }


# -------------------------------------------------------------------------- Phase 14 checks

# Phase 5 recorded one descriptive settings field in two wordings: the PP-OCRv5 mobile page runs
# predate a rewording of `detector_run_mode`; the run mode itself ("paddle", no oneDNN) is the same.
# Old wording -> current wording, applied to the recorded side only and listed in the check's
# result, so a real change of any setting still fails the check.
SETTINGS_WORDING: dict[str, dict[str, str]] = {
    "detector_run_mode": {
        "paddle (no oneDNN; speed only)": "paddle (oneDNN crashes on this machine)",
    },
}
_MISSING = object()


def log_path() -> Path:
    """The test-access log: `ACCESS_LOG` when absolute (tests point it at a temporary file),
    otherwise the repository's own log, found from the repository root."""
    return ACCESS_LOG if ACCESS_LOG.is_absolute() else access_log()


def _phase5_metadata(
    data: Path, system: str, task: str
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, str]]:
    """(file, metadata) of ``system``'s Phase 5 runs of ``task`` that recorded their settings,
    and the files left out with the reason.

    Read: every *.meta.json and profile file under the system's Phase 5 folder; prediction files
    are never opened. Left out: metadata without settings (a run killed before it wrote them) and
    metadata from the first runner (no ``expected`` field), which recorded one settings block for
    recognition and page runs alike.
    """
    folder = data / "processed" / PHASE5_ROOT / EXPERIMENTS[system]
    paths = sorted(folder.glob(f"{system}_{task}_*.meta.json"))
    paths += sorted(folder.glob(f"profile_{system}_{task}_*.json"))
    found: list[tuple[str, dict[str, Any]]] = []
    skipped: dict[str, str] = {}
    for path in paths:
        meta = json.loads(path.read_text(encoding="utf-8"))
        rel = path.relative_to(data).as_posix()
        if meta.get("system") != system or meta.get("task") != task:
            continue
        if meta.get("settings") is None:
            skipped[rel] = "no settings recorded"
        elif "expected" not in meta and not path.name.startswith("profile_"):
            skipped[rel] = "first runner's metadata (recognition and page settings not told apart)"
        else:
            found.append((rel, meta))
    return found, skipped


def phase5_settings(
    data: Path, system: str, task: str
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, str]]:
    """Settings recorded by ``system``'s Phase 5 runs of ``task``.

    Returns (file, settings) for every *.meta.json and profile file `_phase5_metadata` keeps, and
    the files left out with the reason. Only the ``settings``, ``system``, ``task`` and
    ``expected`` fields are used.
    """
    found, skipped = _phase5_metadata(data, system, task)
    return [(rel, meta["settings"]) for rel, meta in found], skipped


def check_phase5_settings(
    data: Path, system: str, task: str, current: dict[str, Any]
) -> dict[str, Any]:
    """Refuse unless every field of ``current`` equals every recorded Phase 5 settings block of
    the same system and task, field by field and in both directions (no field added or lost)."""
    refs, skipped = phase5_settings(data, system, task)
    if not refs:
        raise SystemExit(f"{system} ({task}): no Phase 5 run metadata with settings under "
                         f"processed/{PHASE5_ROOT}/{EXPERIMENTS[system]}; Phase 14 runs the "
                         "Phase 5 settings and cannot check them")  # fmt: skip
    cur = json.loads(json.dumps(current))  # tuples and lists compare as recorded
    aliases = []
    for rel, recorded in refs:
        rec = dict(recorded)
        for field, wording in SETTINGS_WORDING.items():
            if isinstance(rec.get(field), str) and rec[field] in wording:
                rec[field] = wording[rec[field]]
                aliases.append(f"{rel}: {field}")
        diff = sorted(k for k in set(cur) | set(rec)
                      if cur.get(k, _MISSING) != rec.get(k, _MISSING))  # fmt: skip
        if diff:
            raise SystemExit(f"{system} ({task}): settings differ from Phase 5's {rel} in "
                             f"{', '.join(diff)}; Phase 14 runs Phase 5's settings")  # fmt: skip
    return {"result": "equal", "references": [r for r, _ in refs], "skipped": skipped,
            "wording_aliases": aliases}  # fmt: skip


def check_phase5_batch(
    data: Path, system: str, task: str, batch: int, *, fixed: bool = False
) -> dict[str, Any]:
    """Refuse unless ``batch``, the inference batch this run uses, equals the ``inference_batch``
    every recorded segment of the system's Phase 5 prediction runs of ``task`` used.

    The batch is part of how a system runs (TrOCR pads a batch to its widest crop) but not of its
    `settings()`, so the settings check cannot see it. Read from the run metadata the settings
    check uses (`_phase5_metadata`), prediction runs only (a profile always runs batch 1); a
    segment killed before it loaded the model recorded none. When no Phase 5 prediction run of the
    task recorded one (PP-OCRv5 server end to end never completed a Phase 5 page), the run goes
    ahead only if the system fixes its batch whatever ``--batch`` says (``fixed``: PaddleOCR and
    Tesseract run one image at a time), so Phase 5 could not have run another; otherwise it is
    refused.
    """
    found, _ = _phase5_metadata(data, system, task)
    refs: list[str] = []
    seen: set[Any] = set()
    for rel, meta in found:
        if rel.rsplit("/", 1)[-1].startswith("profile_"):
            continue
        batches = {r["inference_batch"] for r in meta.get("runs") or [] if "inference_batch" in r}
        if batches:
            refs.append(rel)
            seen |= batches
    if not seen:
        if fixed:
            return {"result": "fixed by the system", "inference_batch": batch, "references": []}
        raise SystemExit(f"{system} ({task}): no Phase 5 run metadata records an inference "
                         "batch; Phase 14 runs Phase 5's batch and cannot check it")  # fmt: skip
    if seen != {batch}:
        ran = sorted(seen, key=str)
        raise SystemExit(f"{system} ({task}): inference batch {batch}, Phase 5 ran {ran}; "
                         "Phase 14 runs Phase 5's batch (--batch)")  # fmt: skip
    return {"result": "equal", "inference_batch": batch, "references": refs}


def check_profile_threads(system: str, settings: dict[str, Any], phase: int = 14) -> None:
    """H1′'s latency half compares at PaddleOCR's default 10 threads; refuse anything else."""
    if system in PPOCR.MODELS and settings.get("cpu_threads") != PROFILE_THREADS:
        raise SystemExit(f"{system}: profile at {settings.get('cpu_threads')} CPU threads; "
                         f"Phase {phase} profiles run at {PROFILE_THREADS}")  # fmt: skip


def check_phase14_request(args: argparse.Namespace) -> None:
    """The Phase 14 refusals that need no data: branch, set and the registered coverage."""
    try:
        require_branch(PHASE14_BRANCH)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.profile:
        return  # profiles run on validation pages, checked with the Phase 5 rules
    if args.eval_set not in PHASE14_SETS:
        raise SystemExit(f"{args.eval_set}: Phase 14 reuses the Phase 5 predictions there and runs "
                         f"only the new sets {', '.join(PHASE14_SETS)}")  # fmt: skip
    key = (args.system, args.task, args.eval_set)
    if key not in PHASE14_PLAN:
        raise SystemExit(f"{args.system} {args.task} on {args.eval_set}: not in the Phase 14 plan")
    planned = PHASE14_PLAN[key]
    if planned == SUBSET_NAME and not args.subset:
        raise SystemExit(f"{args.system} {args.task} on {args.eval_set} is registered on the "
                         "100-page subset only: pass --subset")  # fmt: skip
    if planned is None and args.subset:
        raise SystemExit(f"{args.system} {args.task} on {args.eval_set} is registered on every "
                         "page; a subset run would evaluate the same pages twice")  # fmt: skip
    if planned not in (None, SUBSET_NAME):
        raise SystemExit(f"{args.system} {args.task} on {args.eval_set}: {planned}")


# -------------------------------------------------------------------------- Phase 16 checks


def check_phase16_request(args: argparse.Namespace) -> None:
    """The Phase 16 refusals that need no data: branch, profile only, the one registered run."""
    try:
        require_branch(PHASE16_BRANCH)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.profile:
        raise SystemExit("Phase 16 makes validation profiles only (--profile); its baseline "
                         "predictions are Phase 5's and Phase 14's")  # fmt: skip
    if (args.system, args.task, args.eval_set) != PHASE16_PROFILE:
        system, task, eval_set = PHASE16_PROFILE
        raise SystemExit(f"Phase 16 profiles {system} ({EXPERIMENTS[system]}) {task} on "
                         f"{eval_set} only (GW-EFF-010), not {args.system} {args.task} on "
                         f"{args.eval_set}")  # fmt: skip


def phase14_profile_path(data: Path, system: str, task: str, eval_set: str) -> Path:
    """The Phase 14 profile whose settings a Phase 16 profile must repeat."""
    path = output_path(data, EXPERIMENTS[system], system, task, eval_set, 14)
    return path.with_name(f"profile_{path.stem}.json")


def check_phase14_profile_settings(
    data: Path, system: str, task: str, eval_set: str, current: dict[str, Any]
) -> dict[str, Any]:
    """Refuse unless every field of ``current`` equals the ``settings`` block of the Phase 14
    profile of the same system, task and set, field by field and in both directions.

    Only that block and the file's ``system``, ``task`` and ``eval_set`` are read. No wording alias
    applies: the Phase 14 profile already uses the current wording."""
    path = phase14_profile_path(data, system, task, eval_set)
    rel = path.relative_to(data).as_posix()
    if not path.is_file():
        raise SystemExit(f"{system} ({task}): no Phase 14 profile at {rel}; the Phase 16 profile "
                         "repeats its settings and cannot check them")  # fmt: skip
    meta = json.loads(path.read_text(encoding="utf-8"))
    who = tuple(meta.get(k) for k in ("system", "task", "eval_set"))
    if who != (system, task, eval_set):
        raise SystemExit(f"{rel} is a profile of {who}, not of {(system, task, eval_set)}")
    recorded = meta.get("settings")
    if not isinstance(recorded, dict):
        raise SystemExit(f"{rel} records no settings")
    cur = json.loads(json.dumps(current))  # tuples and lists compare as recorded
    diff = sorted(k for k in set(cur) | set(recorded)
                  if cur.get(k, _MISSING) != recorded.get(k, _MISSING))  # fmt: skip
    if diff:
        raise SystemExit(f"{system} ({task}): settings differ from Phase 14's {rel} in "
                         f"{', '.join(diff)}; Phase 16 repeats the Phase 14 profile")  # fmt: skip
    return {"result": "equal", "reference": rel, "reference_sha256": sha256_file(path),
            "fields": sorted(recorded)}  # fmt: skip


def phase16_gpu_state() -> dict[str, Any] | None:
    """nvidia-smi's clocks, temperature and memory (`glypharis.efficiency.conditions.gpu_state`),
    imported here so Phase 5 and 14 runs never load the Phase 16 package."""
    from glypharis.efficiency.conditions import gpu_state

    return gpu_state()


# -------------------------------------------------------------------------------------- runs


def output_path(
    data: Path,
    experiment: str,
    system: str,
    task: str,
    eval_set: str,
    phase: int = 5,
    subset: bool = False,
) -> Path:
    """Where a run's predictions go: Phase 5 under processed/baselines, Phase 14 under
    processed/baselines_phase14 (Phase 16's one profile under processed/baselines_phase16); a
    subset run says so in its name."""
    root = {14: PHASE14_ROOT, 16: PHASE16_ROOT}.get(phase, PHASE5_ROOT)
    name = f"{system}_{task}_{eval_set}" + (f"_{SUBSET_NAME}" if subset else "")
    return data / "processed" / root / experiment / f"{name}.jsonl"


def done_keys(path: Path) -> set[str]:
    """Keys already written. Repairs the end of a file left by an interrupted run: a cut-off last
    record is removed (even one cut inside a UTF-8 character), and a complete last record that lost
    only its newline gets the newline back, so the next record starts on its own line."""
    if not path.exists():
        return set()
    raw = path.read_bytes()
    *complete, tail = raw.split(b"\n")
    keys: set[str] = set()
    for line in complete:
        if line.strip():
            keys.add(json.loads(line.decode("utf-8"))["key"])
    if tail.strip():
        try:
            keys.add(json.loads(tail.decode("utf-8"))["key"])
            with path.open("ab") as fh:
                fh.write(b"\n")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError):
            with path.open("r+b") as fh:
                fh.truncate(len(raw) - len(tail))
    return keys


def _write_meta(path: Path, meta: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def run_predictions(args: argparse.Namespace, data: Path) -> int:
    spec = EVAL_SETS[args.eval_set]
    phase, subset = args.phase, args.subset
    out = output_path(data, args.experiment, args.system, args.task, args.eval_set, phase, subset)
    keys = expected_keys(data, args.eval_set, args.task, subset)
    split = spec["split"]
    phase14: dict[str, Any] = {}
    if phase == 14:
        assert PHASE5_ROOT not in out.relative_to(data).parts[:2], out
        # The split the log records is the one the records carry, checked against the set's.
        splits = record_splits(data, args.eval_set, args.task)
        if splits != {split}:
            raise SystemExit(f"{args.eval_set}: records name splits {sorted(splits)}, "
                             f"the set is registered as {split}")  # fmt: skip
        phase14 = {"phase": 14, "subset": SUBSET_NAME if subset else None}
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    stray = done - set(keys)
    if stray:
        example = sorted(stray)[0]
        raise SystemExit(f"{out} holds {len(stray)} keys outside {args.eval_set} (e.g. {example});"
                         " it was made for another sample, move it aside")  # fmt: skip
    meta_path = out.with_suffix(".meta.json")
    if meta_path.exists() and not out.exists():
        meta_path.unlink()  # metadata of predictions that were moved aside: start afresh
    if len(done) >= len(keys):
        # Nothing to do: no model load, and no entry in the test-access log.
        if not meta_path.exists():
            _write_meta(meta_path, {"system": args.system, "task": args.task,
                                    "eval_set": args.eval_set, "experiment": args.experiment,
                                    "expected": len(keys), "samples": len(done), "runs": [],
                                    **phase14,
                                    "note": "predictions written by a run killed before it "
                                            "wrote metadata; settings not recorded"})  # fmt: skip
        print(f"{args.system} {args.task} {args.eval_set}: already complete ({len(done)})")
        return 0

    git = git_state()  # before the access log changes
    reason = args.reason
    if done:
        reason += f" (resumed: {len(done)} of {len(keys)} samples already predicted)"
    system = None
    if phase == 14:
        # The inputs are hashed (bytes only) and compared with the preparation's record, then the
        # model loads (no test image is decoded) and its settings and batch are checked against
        # Phase 5, all before the access is logged, so a refused run leaves no row.
        n_pages = None if args.task == "rec" else len(list(read_pages(data / spec["pages"])))
        phase14["inputs"] = check_inputs(data, args.eval_set, args.task, n_pages,
                                         require=REQUIRE_RECORDED)  # fmt: skip
        system = ready(make_system(args.system, args.task), data)
        phase14["phase5_settings_check"] = check_phase5_settings(
            data, args.system, args.task, system.settings()
        )
        inference_batch = system.inference_batch(args.batch)
        fixed = len({system.inference_batch(b) for b in (1, 2, args.batch)}) == 1
        phase14["phase5_batch_check"] = check_phase5_batch(
            data, args.system, args.task, inference_batch, fixed=fixed
        )
        phase14["batch"] = {"runner_batch": args.batch if args.task == "rec" else 1,
                            "inference_batch": inference_batch}  # fmt: skip
        previous = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        for field in ("inputs", "batch"):
            if field in previous and previous[field] != phase14[field]:
                raise SystemExit(f"{out.name}: the run it resumes recorded other {field}; a resume "
                                 "must be the same evaluation, move the file aside")  # fmt: skip
    open_test_access(spec["benchmark"], split, args.experiment,
                     f"{args.system} ({args.task})", phase, reason, log_path())  # fmt: skip

    import psutil

    proc = psutil.Process()
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    runs = meta.get("runs", [])
    seg = len(runs)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    run = {"segment": seg, "started": started, "status": "loading", "already_done": len(done),
           "runner_batch": args.batch if args.task == "rec" else 1, **git}  # fmt: skip
    runs.append(run)
    base = {"system": args.system, "task": args.task, "eval_set": args.eval_set,
            "experiment": args.experiment, "expected": len(keys), "rec_val_sample": REC_VAL_SAMPLE,
            "e2e_val_pages": E2E_VAL_PAGES, "seed": SEED, "runs": runs, **phase14,
            "note": "time_ms in the records is diagnostic only; reported speed and memory come "
                    "from --profile runs"}  # fmt: skip
    _write_meta(meta_path, {**meta, **base})

    if system is None:
        system = ready(make_system(args.system, args.task), data)
    run.update(status="running", inference_batch=system.inference_batch(args.batch))
    meta = {**meta, **base, "version": system.version, "settings": system.settings()}
    _write_meta(meta_path, meta)
    start = time.monotonic()
    written = 0

    with out.open("a", encoding="utf-8", newline="\n") as fh:

        def emit(record: dict[str, Any]) -> None:
            nonlocal written
            fh.write(json.dumps({**record, "seg": seg}, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1

        if args.task == "rec":
            batch: list[tuple[str, str, Image.Image]] = []

            def flush() -> None:
                t0 = time.perf_counter()
                texts = system.recognize([b[2] for b in batch])
                per_ms = (time.perf_counter() - t0) * 1000 / len(batch)
                for (key, ref, _), text in zip(batch, texts, strict=True):
                    emit({"key": key, "ref": ref, "pred": text, "time_ms": round(per_ms, 2)})
                batch.clear()

            for key, ref, img in rec_samples(data, args.eval_set):
                if key not in done:
                    batch.append((key, ref, img))
                    if len(batch) >= args.batch:
                        flush()
            if batch:
                flush()
        else:
            for page in e2e_samples(data, args.eval_set, subset):
                if page.doc_id in done:
                    continue
                img = open_page_image(page, data)
                t0 = time.perf_counter()
                lines = system.page(img)
                ms = (time.perf_counter() - t0) * 1000
                emit({"key": page.doc_id, "lines": lines, "time_ms": round(ms, 1)})

    run.update(status="done", written=written, wall_seconds=round(time.monotonic() - start, 1),
               ended=datetime.now(UTC).isoformat(timespec="seconds"),
               peak_mem_mib=round(peak_working_set(proc) / 2**20, 1))  # fmt: skip
    meta["samples"] = len(done) + written
    _write_meta(meta_path, meta)
    print(f"{args.system} {args.task} {args.eval_set}: {written} new, {meta['samples']} total")
    return 0


def run_profile(args: argparse.Namespace, data: Path) -> int:
    spec = EVAL_SETS[args.eval_set]
    if spec["split"] == "test":
        raise SystemExit("profiling runs use validation data only")
    import psutil

    proc = psutil.Process()
    path = output_path(data, args.experiment, args.system, args.task, args.eval_set, args.phase)
    path = path.with_name(f"profile_{path.stem}.json")
    if path.exists():
        # A profile is never overwritten: in Phase 14 the first sitting that is not void is the
        # one of record (a void one is moved aside by hand and recorded in research/failures.md),
        # and a Phase 5 profile is part of the files Phase 14 hashed.
        raise SystemExit(f"{path.relative_to(data).as_posix()} exists: move it aside first")
    conditions = run_conditions()  # before the model loads, so the CPU check sees an idle machine
    phase16_start: dict[str, Any] = {}
    if args.phase == 16 and PHASE16_ROOT not in path.relative_to(data).parts[:2]:
        raise SystemExit(f"{path.as_posix()}: a Phase 16 profile goes under {PHASE16_ROOT} only")
    if args.phase == 16:
        phase16_start = {"utc": datetime.now(UTC).isoformat(timespec="seconds"),
                         "gpu": phase16_gpu_state()}  # fmt: skip
    git = git_state()
    system = ready(make_system(args.system, args.task), data)
    phase14: dict[str, Any] = {}  # the phase's own fields (Phase 14 or 16)
    if args.phase == 14:
        settings = system.settings()
        check_profile_threads(args.system, settings)
        phase14 = {"phase": 14, "phase5_settings_check": check_phase5_settings(
            data, args.system, args.task, settings)}  # fmt: skip
    elif args.phase == 16:
        settings = system.settings()
        check_profile_threads(args.system, settings, phase=16)
        phase14 = {"phase": 16,
                   "amends": f"{args.experiment} (Phase 16 amendment for GW-EFF-010)",
                   "phase14_settings_check": check_phase14_profile_settings(
                       data, args.system, args.task, args.eval_set, settings)}  # fmt: skip
    samples: Iterator[Any] = (
        (img for _, _, img in rec_samples(data, args.eval_set))
        if args.task == "rec"
        else (open_page_image(p, data) for p in e2e_samples(data, args.eval_set))
    )
    times, sizes = [], []
    started = datetime.now(UTC).isoformat(timespec="seconds")  # just before the first warm-up
    for _, img in zip(range(WARMUP + PROFILE_SAMPLES), samples, strict=False):
        t0 = time.perf_counter()
        if args.task == "rec":
            system.recognize([img])
        else:
            system.page(img)
        times.append(round((time.perf_counter() - t0) * 1000, 3))
        sizes.append(list(img.size))
    ended = datetime.now(UTC).isoformat(timespec="seconds")  # after the last timed sample
    if len(times) < WARMUP + PROFILE_SAMPLES:
        raise SystemExit(f"{args.eval_set} gave {len(times)} samples; profiling needs "
                         f"{WARMUP} warm-up + {PROFILE_SAMPLES} timed")  # fmt: skip
    if args.phase == 16:  # the end of the sitting, as every Phase 16 profile records it
        end_utc = datetime.now(UTC).isoformat(timespec="seconds")
        phase14.update(conditions_end=run_conditions(),
                       conditions_utc={"start": phase16_start["utc"], "end": end_utc},
                       gpu_state={"start": phase16_start["gpu"],
                                  "end": phase16_gpu_state()})  # fmt: skip
    own_peak = peak_working_set(proc)
    child_peak = getattr(system, "child_peak", 0)
    profile = {
        "system": args.system, "task": args.task, "eval_set": args.eval_set,
        "experiment": args.experiment, "version": system.version, "settings": system.settings(),
        "batch": 1, "started_utc": started, "ended_utc": ended,
        "warmup": WARMUP, "warmup_times_ms": times[:WARMUP],
        "times_ms": times[WARMUP:], "input_sizes": sizes[WARMUP:],
        "process_peak_mib": round(own_peak / 2**20, 1),
        "child_peak_mib": round(child_peak / 2**20, 1),
        # A command-line engine runs next to the Python harness, so both count.
        "peak_mem_mib": round((own_peak + child_peak) / 2**20, 1),
        "conditions": conditions, "cpu_busy_percent_after": cpu_busy_percent(),
        **git, **phase14,
    }  # fmt: skip
    profile["model_mib"] = round(system.model_bytes() / 2**20, 2)
    if isinstance(system, Tesseract):
        profile["invocation_overhead_ms"] = round(system.invocation_overhead_ms(), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_meta(path, profile)
    t = np.asarray(profile["times_ms"])
    print(f"profile {args.system} {args.task} {args.eval_set}: mean {t.mean():.1f} ms, "
          f"p95 {np.percentile(t, 95):.1f} ms, peak {profile['peak_mem_mib']} MiB")  # fmt: skip
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", required=True, choices=sorted(EXPERIMENTS))
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--eval-set", required=True, choices=sorted(EVAL_SETS))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--phase", type=int, required=True, choices=PHASES,
                        help="5 (Phase 5 sets), 14 (the new sets, Phase 14 rules) or 16 "
                             "(GW-BASE-002's validation profile for GW-EFF-010)")  # fmt: skip
    parser.add_argument("--reason", required=True, help="goes into the test-access log")
    parser.add_argument("--subset", action="store_true",
                        help="OmniDocBench end to end on the 100-page subset only")  # fmt: skip
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--profile", action="store_true", help="measure speed and memory")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)

    if EXPERIMENTS[args.system] != args.experiment:
        raise SystemExit(f"{args.system} is registered as {EXPERIMENTS[args.system]}, "
                         f"not {args.experiment}")  # fmt: skip
    if args.system == "trocr" and args.task == "e2e":
        raise SystemExit("TrOCR has no detector, so it is evaluated on recognition only")
    if args.task == "e2e" and not EVAL_SETS[args.eval_set]["pages"]:
        raise SystemExit(f"{args.eval_set} has no pages, so end to end is not defined for it")
    if args.task == "rec" and not EVAL_SETS[args.eval_set]["pack"]:
        raise SystemExit(f"{args.eval_set} has no line pack, so recognition is not defined for it")
    if args.subset and (args.task, args.eval_set) != ("e2e", "omnidocbench_test"):
        raise SystemExit("--subset is defined for end to end on omnidocbench_test only")
    if args.subset and args.profile:
        raise SystemExit("--subset does not apply to profiling runs")
    if args.phase == 5 and args.eval_set in PHASE14_SETS:
        raise SystemExit(f"{args.eval_set} is a Phase 14 set: run it with --phase 14")
    if args.profile and args.eval_set not in PROFILE_SETS[args.task]:
        raise SystemExit(f"profiling {args.task} uses {PROFILE_SETS[args.task]}")
    if not args.reason.strip():
        raise SystemExit("a reason is required (it goes into the test-access log)")
    data = Path(args.data_root)
    if args.phase == 14:
        check_phase14_request(args)
        require_phase5_hashes(data)
    elif args.phase == 16:
        check_phase16_request(args)  # writes under its own folder only, so Phase 5 stays closed
    elif (data / PHASE5_HASHES).exists():
        raise SystemExit(f"data/{PHASE5_HASHES.as_posix()} exists: the Phase 5 files are closed "
                         "and hashed for Phase 14; a Phase 5 run would change them")  # fmt: skip
    return run_profile(args, data) if args.profile else run_predictions(args, data)


if __name__ == "__main__":
    sys.exit(main())
