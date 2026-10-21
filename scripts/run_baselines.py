"""Run an external OCR system on an evaluation set and write its predictions (Phase 5).

Runs inside .venv-baselines (ADR-0006); scoring happens separately in the main environment with
`scripts/evaluate_baselines.py`, so every system is scored by the same code.

    python scripts/run_baselines.py --system tesseract --task rec --eval-set hiertext_val \
        --experiment GW-BASE-001

Systems: tesseract, ppocr_mobile, ppocr_server, trocr (recognition only).
Tasks: rec (ground-truth line crops in, text out) and e2e (page in, lines with polygons and text
out; detection metrics come from the same output).
Evaluation sets: hiertext_val, glyphsynth_val (rec only), hiertext_test, gdc_test. Test sets pass
through the test-access guard, which logs the access before any data is read.

Recognition crops always come from line packs: Phase 4 packs for validation, packs cut by
`scripts/build_eval_crops.py` for test. Validation sets are subsampled with a fixed seed to keep CPU
time bounded; test sets are used in full.

Output: data/processed/baselines/<experiment>/<system>_<task>_<eval-set>.jsonl, one record per
sample, appended as it goes so an interrupted run resumes where it stopped, plus a .meta.json with
versions, settings, timing and peak memory. Each system runs with its defaults and English
selected; nothing is tuned per benchmark (EXPERIMENT_PROTOCOL §10). Deviations from defaults are
listed in each system's `settings()`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from glypharis.data.linepack import LinePackReader
from glypharis.data.pages import Page, open_page_image, read_pages
from glypharis.eval.access import open_test_access
from glypharis.utils.gitinfo import git_info

MODELS = Path("data/raw/baseline_models")
REC_VAL_SAMPLE = 10_000  # crops per validation set
E2E_VAL_PAGES = 400  # pages per validation set
SEED = 5
TASKS = ("rec", "e2e")

EVAL_SETS: dict[str, dict[str, Any]] = {
    "hiertext_val": {"benchmark": "hiertext", "split": "validation",
                     "pack": "processed/hiertext/lines_validation",
                     "pages": "processed/hiertext/pages_validation.jsonl.gz"},
    "glyphsynth_val": {"benchmark": "glyphsynth", "split": "validation",
                       "pack": "processed/glyphsynth/lines_val", "pages": None},
    "hiertext_test": {"benchmark": "hiertext", "split": "test",
                      "pack": "processed/hiertext/lines_test",
                      "pages": "processed/hiertext/pages_test.jsonl.gz"},
    "gdc_test": {"benchmark": "gdc", "split": "test", "pack": "processed/gdc/lines_test",
                 "pages": "processed/gdc/pages_test.jsonl.gz"},
}  # fmt: skip


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


def expected_count(data: Path, eval_set: str, task: str) -> int:
    """How many samples a complete run of ``task`` on ``eval_set`` writes."""
    spec = EVAL_SETS[eval_set]
    if task == "rec":
        meta = json.loads((data / f"{spec['pack']}.meta.json").read_text(encoding="utf-8"))
        return len(rec_indices(int(meta["count"]), spec["split"]))
    if not spec["pages"]:
        return 0
    return len(page_indices(sum(1 for _ in read_pages(data / spec["pages"])), spec["split"]))


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


# ----------------------------------------------------------------------------------- systems


class Tesseract:
    """Tesseract 5 command line, English, default engine (LSTM)."""

    name = "tesseract"
    tasks = TASKS

    def __init__(self) -> None:
        exe = shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if not Path(exe).exists():
            raise SystemExit("Tesseract not found: install the UB Mannheim build or put it on PATH")
        self.exe = exe
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True)
        self.version = out.stdout.splitlines()[0].strip()

    def settings(self) -> dict[str, Any]:
        # psm 7 treats the image as one text line; psm 3 is Tesseract's default page mode
        return {"lang": "eng", "oem": "default (LSTM)", "rec_psm": 7, "e2e_psm": 3}

    def recognize(self, images: list[Image.Image]) -> list[str]:
        """One Tesseract process per batch, over a list file; results are split on form feeds."""
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, img in enumerate(images):
                path = Path(tmp) / f"{i:05d}.png"
                img.save(path)
                paths.append(str(path))
            listing = Path(tmp) / "list.txt"
            listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
            out = subprocess.run([self.exe, str(listing), "stdout", "-l", "eng", "--psm", "7"],
                                 capture_output=True, check=True)  # fmt: skip
        texts = out.stdout.decode("utf-8", errors="replace").split("\f")
        if len(texts) < len(images):
            raise RuntimeError(f"Tesseract returned {len(texts)} results for {len(images)} images")
        return [t.strip() for t in texts[: len(images)]]

    def page(self, img: Image.Image) -> list[dict[str, Any]]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page.png"
            img.save(path)
            out = subprocess.run([self.exe, str(path), "stdout", "-l", "eng", "--psm", "3", "tsv"],
                                 capture_output=True, check=True)  # fmt: skip
        return tesseract_tsv_lines(out.stdout.decode("utf-8", errors="replace"))


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

    PaddleOCR's own pipeline config is used with two changes:

    - document unwarping is off, because its output polygons are in the frame of the unwarped
      image and cannot be scored against the page's ground truth. Orientation classification
      stays on, and polygons from a rotated page are mapped back with `unrotate`;
    - the text detector runs without oneDNN, because Paddle 3.3's oneDNN path crashes on the
      PP-OCRv5 detectors on this machine. This changes speed only; every other model keeps the
      default oneDNN mode.

    Recognition-only runs use the TextRecognition module with its default batch size.
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

        self.name = name
        import paddleocr

        det, rec = self.MODELS[name]
        self.det_name, self.rec_name = det, rec
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
        return {"det_model": self.det_name, "rec_model": self.rec_name, "device": "cpu",
                "use_doc_unwarping": False, "use_doc_orientation_classify": True,
                "use_textline_orientation": True, "detector_run_mode": "paddle (no oneDNN)",
                "deviation": "doc unwarping off (it moves polygons)"}  # fmt: skip

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

    Generation keeps the checkpoint's settings (greedy) except the length limit: the checkpoint sets
    none, so transformers would stop at 20 tokens and cut every long line short. MAX_NEW_TOKENS is
    well above the longest line in the evaluation sets. The checkpoint lacks the encoder pooler
    weights, which generation does not use.
    """

    MAX_NEW_TOKENS = 256

    name = "trocr"
    tasks = ("rec",)

    def __init__(self) -> None:
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
                "device": "cpu", "torch_threads": self.torch.get_num_threads(),
                "deviation": "max_new_tokens set (default would stop at 20 tokens)"}  # fmt: skip

    def recognize(self, images: list[Image.Image]) -> list[str]:
        pixel = self.processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        with self.torch.inference_mode():
            ids = self.model.generate(pixel.pixel_values, max_new_tokens=self.MAX_NEW_TOKENS)
        return [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]


def make_system(name: str, task: str) -> Any:
    if name in PPOCR.MODELS:
        return PPOCR(name, task)
    return {"tesseract": Tesseract, "trocr": TrOCR}[name]()


# -------------------------------------------------------------------------------------- main


def output_path(data: Path, experiment: str, system: str, task: str, eval_set: str) -> Path:
    return data / "processed" / "baselines" / experiment / f"{system}_{task}_{eval_set}.jsonl"


def done_keys(path: Path) -> set[str]:
    """Keys already written. A last line cut short by an interrupted run is removed first."""
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    keys, good = set(), 0
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            keys.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError):
            if i != len(lines) - 1:
                raise
            break
        good = i + 1
    if good < len(lines):
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("".join(lines[:good]))
    return keys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", required=True,
                        choices=["tesseract", *PPOCR.MODELS, "trocr"])  # fmt: skip
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--eval-set", required=True, choices=sorted(EVAL_SETS))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--reason", default="Phase 5 baseline")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args(argv)

    data = Path(args.data_root)
    spec = EVAL_SETS[args.eval_set]
    if args.system == "trocr" and args.task == "e2e":
        raise SystemExit("TrOCR has no detector, so it is evaluated on recognition only")
    out = output_path(data, args.experiment, args.system, args.task, args.eval_set)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    if len(done) >= expected_count(data, args.eval_set, args.task):
        # Nothing to do: no model load, and no entry in the test-access log.
        print(f"{args.system} {args.task} {args.eval_set}: already complete ({len(done)})")
        return 0
    open_test_access(spec["benchmark"], spec["split"], args.experiment,
                     f"{args.system} ({args.task})", 5, args.reason)  # fmt: skip

    git = git_info()  # the code that is about to run
    system = make_system(args.system, args.task)
    import psutil

    proc = psutil.Process()
    peak_rss = proc.memory_info().rss
    start = time.monotonic()
    written = 0

    with out.open("a", encoding="utf-8", newline="\n") as fh:

        def emit(record: dict[str, Any]) -> None:
            nonlocal written, peak_rss
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            written += 1
            peak_rss = max(peak_rss, proc.memory_info().rss)

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

    meta_path = out.with_suffix(".meta.json")
    runs = json.loads(meta_path.read_text(encoding="utf-8"))["runs"] if meta_path.exists() else []
    runs.append({"written": written, "wall_seconds": round(time.monotonic() - start, 1),
                 "git_commit": git.commit, "git_dirty": git.dirty,
                 "peak_rss_mib": round(peak_rss / 2**20, 1),
                 "batch": args.batch if args.task == "rec" else 1})  # fmt: skip
    meta = {
        "system": args.system,
        "version": system.version,
        "settings": system.settings(),
        "task": args.task,
        "eval_set": args.eval_set,
        "experiment": args.experiment,
        "samples": len(done) + written,
        "runs": runs,
        "rec_val_sample": REC_VAL_SAMPLE,
        "e2e_val_pages": E2E_VAL_PAGES,
        "seed": SEED,
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
    }
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{args.system} {args.task} {args.eval_set}: {written} new, {meta['samples']} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
