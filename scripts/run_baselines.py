"""Run an external OCR system on an evaluation set and write its predictions (Phase 5).

Runs inside .venv-baselines (ADR-0006); scoring happens separately in the main environment with
`scripts/evaluate_baselines.py`, so every system is scored by the same code.

    python scripts/run_baselines.py --system tesseract --task rec --eval-set hiertext_val \
        --experiment GW-BASE-001
    python scripts/run_baselines.py --system tesseract --task rec --eval-set hiertext_val \
        --experiment GW-BASE-001 --profile

Systems: tesseract, ppocr_mobile, ppocr_server, ppocr6_tiny, ppocr6_small, ppocr6_medium, trocr
(recognition only). Each system has one experiment ID (`EXPERIMENTS`); a mismatch is refused.
Tasks: rec (ground-truth line crops in, text out) and e2e (page in, lines with polygons and text
out; detection metrics come from the same output).
Evaluation sets: hiertext_val, glyphsynth_val (rec only), hiertext_test, gdc_test.

Prediction runs (default). Every sample of the set goes through the system and is appended to
data/processed/baselines/<experiment>/<system>_<task>_<eval-set>.jsonl as soon as it is done, so an
interrupted run resumes where it stopped. The run's .meta.json is written when the run starts and
updated when it ends. On a test set the test-access guard logs the access before any image is
decoded or passed to a system (the sample keys are read first, to tell whether the run is already
complete).

Profiling runs (--profile). Speed and memory are measured separately from the long prediction runs,
which share the machine with whatever else is happening over hours. A profiling run times
`WARMUP` samples it discards and then `PROFILE_SAMPLES` samples one at a time (batch 1), records
the peak working set of the process and of any child process (tesseract.exe), and the conditions
EXPERIMENT_PROTOCOL §3 asks for. Validation sets only; the result goes to
profile_<system>_<task>_<eval-set>.json next to the predictions.

Recognition crops always come from line packs cut by `scripts/build_eval_crops.py` (GlyphSynth uses
its Phase 4 pack). Validation sets are subsampled with a fixed seed to keep CPU time bounded; test
sets are used in full. Each system runs with its defaults and English selected; nothing is tuned per
benchmark (EXPERIMENT_PROTOCOL §10). Deviations from defaults are listed in `settings()`.
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

EVAL_SETS: dict[str, dict[str, Any]] = {
    "hiertext_val": {"benchmark": "hiertext", "split": "validation",
                     "pack": "processed/hiertext/lines_validation_eval",
                     "pages": "processed/hiertext/pages_validation.jsonl.gz",
                     "reading_order": False},
    "glyphsynth_val": {"benchmark": "glyphsynth", "split": "validation",
                       "pack": "processed/glyphsynth/lines_val", "pages": None,
                       "reading_order": False},
    "hiertext_test": {"benchmark": "hiertext", "split": "test",
                      "pack": "processed/hiertext/lines_test",
                      "pages": "processed/hiertext/pages_test.jsonl.gz", "reading_order": False},
    "gdc_test": {"benchmark": "gdc", "split": "test", "pack": "processed/gdc/lines_test",
                 "pages": "processed/gdc/pages_test.jsonl.gz", "reading_order": True},
}  # fmt: skip
# HierText annotation order is not a reading order (its README says so even for lines inside a
# paragraph); GDC line order comes from the PDF text layer. Order-aware page metrics are reported
# only where "reading_order" is true.

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


def expected_keys(data: Path, eval_set: str, task: str) -> list[str]:
    """The keys a complete run of ``task`` on ``eval_set`` writes, in run order."""
    spec = EVAL_SETS[eval_set]
    if task == "rec":
        reader = LinePackReader(data / spec["pack"])
        return [reader.records[int(i)].key for i in rec_indices(len(reader), spec["split"])]
    if not spec["pages"]:
        return []
    ids = [p.doc_id for p in read_pages(data / spec["pages"])]
    return [ids[int(i)] for i in page_indices(len(ids), spec["split"])]


def rec_samples(data: Path, eval_set: str) -> Iterator[tuple[str, str, Image.Image]]:
    spec = EVAL_SETS[eval_set]
    reader = LinePackReader(data / spec["pack"])
    for i in rec_indices(len(reader), spec["split"]):
        rec = reader.records[int(i)]
        yield rec.key, rec.label, reader.image(int(i))


def e2e_samples(data: Path, eval_set: str) -> Iterator[Page]:
    spec = EVAL_SETS[eval_set]
    if not spec["pages"]:
        raise SystemExit(f"{eval_set} has no pages, so end to end is not defined for it")
    pages = list(read_pages(data / spec["pages"]))
    for i in page_indices(len(pages), spec["split"]):
        yield pages[int(i)]


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


class Tesseract:
    """Tesseract 5 command line, English, default engine (LSTM) and default traineddata."""

    name = "tesseract"
    tasks = TASKS

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

    def recognize(self, images: list[Image.Image]) -> list[str]:
        """One Tesseract process per batch, over a list file; results are split on form feeds
        (checked: N images give N - 1 separators, and blank images keep their slot)."""
        with tempfile.TemporaryDirectory() as tmp:
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
        with tempfile.TemporaryDirectory() as tmp:
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
      PP-OCRv5 and PP-OCRv6 detectors on this machine. This changes speed only; every other model
      keeps the default oneDNN mode.

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
                "detector_run_mode": "paddle (no oneDNN; speed only)",
                "deviation": "doc unwarping off (it moves polygons)"}  # fmt: skip

    def inference_batch(self, runner_batch: int) -> int:
        return 1  # TextRecognition's default batch size; the pipeline runs one page at a time

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


def run_conditions() -> dict[str, Any]:
    """Machine state that EXPERIMENT_PROTOCOL §3 asks to record with efficiency numbers."""
    import psutil

    battery = psutil.sensors_battery()
    try:
        plan = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                              text=True, timeout=10).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        plan = None
    return {
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": os.cpu_count(),
        "on_ac_power": battery.power_plugged if battery else None,
        "battery_percent": battery.percent if battery else None,
        "power_plan": plan,
        "cpu_busy_percent_before": cpu_busy_percent(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
        "python": platform.python_version(),
        "os": platform.platform(),
    }


# -------------------------------------------------------------------------------------- runs


def output_path(data: Path, experiment: str, system: str, task: str, eval_set: str) -> Path:
    return data / "processed" / "baselines" / experiment / f"{system}_{task}_{eval_set}.jsonl"


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
    out = output_path(data, args.experiment, args.system, args.task, args.eval_set)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = expected_keys(data, args.eval_set, args.task)
    done = done_keys(out)
    stray = done - set(keys)
    if stray:
        example = sorted(stray)[0]
        raise SystemExit(f"{out} holds {len(stray)} keys outside {args.eval_set} (e.g. {example});"
                         " it was made for another sample, move it aside")  # fmt: skip
    if len(done) >= len(keys):
        # Nothing to do: no model load, and no entry in the test-access log.
        print(f"{args.system} {args.task} {args.eval_set}: already complete ({len(done)})")
        return 0

    git = git_state()  # before the access log changes
    reason = args.reason
    if done:
        reason += f" (resumed: {len(done)} of {len(keys)} samples already predicted)"
    open_test_access(spec["benchmark"], spec["split"], args.experiment,
                     f"{args.system} ({args.task})", 5, reason, ACCESS_LOG)  # fmt: skip

    import psutil

    proc = psutil.Process()
    meta_path = out.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    runs = meta.get("runs", [])
    seg = len(runs)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    run = {"segment": seg, "started": started, "status": "loading", "already_done": len(done),
           "runner_batch": args.batch if args.task == "rec" else 1, **git}  # fmt: skip
    runs.append(run)
    base = {"system": args.system, "task": args.task, "eval_set": args.eval_set,
            "experiment": args.experiment, "expected": len(keys), "rec_val_sample": REC_VAL_SAMPLE,
            "e2e_val_pages": E2E_VAL_PAGES, "seed": SEED, "runs": runs,
            "note": "time_ms in the records is diagnostic only; reported speed and memory come "
                    "from --profile runs"}  # fmt: skip
    _write_meta(meta_path, {**meta, **base})

    system = make_system(args.system, args.task)
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
            for page in e2e_samples(data, args.eval_set):
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
    conditions = run_conditions()  # before the model loads, so the CPU check sees an idle machine
    git = git_state()
    system = make_system(args.system, args.task)
    samples: Iterator[Any] = (
        (img for _, _, img in rec_samples(data, args.eval_set))
        if args.task == "rec"
        else (open_page_image(p, data) for p in e2e_samples(data, args.eval_set))
    )
    times, sizes = [], []
    for _, img in zip(range(WARMUP + PROFILE_SAMPLES), samples, strict=False):
        t0 = time.perf_counter()
        if args.task == "rec":
            system.recognize([img])
        else:
            system.page(img)
        times.append(round((time.perf_counter() - t0) * 1000, 3))
        sizes.append(list(img.size))
    if len(times) < WARMUP + PROFILE_SAMPLES:
        raise SystemExit(f"{args.eval_set} gave {len(times)} samples; profiling needs "
                         f"{WARMUP} warm-up + {PROFILE_SAMPLES} timed")  # fmt: skip
    own_peak = peak_working_set(proc)
    child_peak = getattr(system, "child_peak", 0)
    profile = {
        "system": args.system, "task": args.task, "eval_set": args.eval_set,
        "experiment": args.experiment, "version": system.version, "settings": system.settings(),
        "batch": 1, "warmup": WARMUP, "warmup_times_ms": times[:WARMUP],
        "times_ms": times[WARMUP:], "input_sizes": sizes[WARMUP:],
        "process_peak_mib": round(own_peak / 2**20, 1),
        "child_peak_mib": round(child_peak / 2**20, 1),
        # A command-line engine runs next to the Python harness, so both count.
        "peak_mem_mib": round((own_peak + child_peak) / 2**20, 1),
        "conditions": conditions, "cpu_busy_percent_after": cpu_busy_percent(),
        **git,
    }  # fmt: skip
    if isinstance(system, Tesseract):
        profile["invocation_overhead_ms"] = round(system.invocation_overhead_ms(), 1)
    path = output_path(data, args.experiment, args.system, args.task, args.eval_set)
    path = path.with_name(f"profile_{path.stem}.json")
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
    parser.add_argument("--reason", default="Phase 5 baseline")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--profile", action="store_true", help="measure speed and memory")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)

    if EXPERIMENTS[args.system] != args.experiment:
        raise SystemExit(f"{args.system} is registered as {EXPERIMENTS[args.system]}, "
                         f"not {args.experiment}")  # fmt: skip
    if args.system == "trocr" and args.task == "e2e":
        raise SystemExit("TrOCR has no detector, so it is evaluated on recognition only")
    data = Path(args.data_root)
    return run_profile(args, data) if args.profile else run_predictions(args, data)


if __name__ == "__main__":
    sys.exit(main())
