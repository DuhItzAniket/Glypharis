"""The Phase 16 scorer on made-up files in the shapes the Phase 16 tools write.

Every input is built here in a temporary repository root: the GW-EFF-006 bundle report, reference
manifest and check (`scripts/export_check.py --frozen`, `scripts/phase16_reference.py`), the v1
profiles (`scripts/rec_profile.py --experiment`, `scripts/e2e_profile.py --v1`), every variant's
accuracy and profile files (`scripts/phase16_variants.py`) and the fresh GW-BASE-002 profile
(`scripts/run_baselines.py --phase 16`). No real profile, data or model is read.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import phase14_score
import phase16_score as ps
import pytest
import yaml

from glypharis.efficiency.variants import (
    ORT_FP32,
    ORT_FUSED,
    ORT_INT8,
    TORCH,
    TORCH_STRICT,
    VARIANTS,
)
from glypharis.eval.stats import mean_ci, paired_ratio_diff_ci

T0 = datetime(2026, 11, 20, 8, 0, tzinfo=UTC)
SLOT = timedelta(minutes=12)
RUN = timedelta(minutes=10)
THREAD_ENV = ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "MKL_NUM_THREADS")
V1_CODE = {"tag": "phase14-eval", "tag_commit": "a" * 40, "head": "b" * 40,
           "paths": ["glypharis/pipeline"], "v1_code": "equal"}  # fmt: skip
COND = {"cpu": "test cpu", "on_ac_power": True, "power_mode": "balanced",
        "power_plan": "Power Scheme GUID: 381b4222  (Balanced)", "cpu_busy_percent_before": 2.0,
        "thread_env": {}}  # fmt: skip
GPU = {"utc": "2026-11-20T08:00:00+00:00", "gpus": [{"name": "test gpu", "temperature_c": 40,
                                                     "memory_used_mib": 500}]}  # fmt: skip
PAGE_SIZES = [[1000 + i, 800] for i in range(100)]
OUTPUTS = {"path": "x", "sha256": "0" * 64, "content_sha256": "0" * 64}
LINE_SIZES = [[100 + i, 32] for i in range(100)]
LINE_KEYS = [f"hiertext/l{i}" for i in range(100)]
PAGE_LINES = [20 + i % 40 for i in range(100)]
DOC_IDS = [f"doc{i:03d}" for i in range(ps.PAGES)]
STAGES = ["to_rgb", "det_preprocess", "det_network", "det_postprocess", "crop", "rec_batch",
          "rec_network", "rec_decode", "reading_order"]  # fmt: skip
RUNTIMES = {
    "v1-torch": (TORCH, TORCH_STRICT),
    "onnx-det-torch-rec": (ORT_FP32, TORCH_STRICT),
    "int8-recognizer": (ORT_FP32, ORT_INT8),
    "fused-attention": (ORT_FP32, ORT_FUSED),
}
# Mean line and page times of each variant (ms), fixed so the verdicts are known in advance.
LINE_MS = {"v1-torch": 18.0, "v1-onnx": 20.0, "int8-recognizer": 12.0, "fused-attention": 15.0,
           "rec-GW-REC-003": 10.0, "rec-GW-REC-006": 11.0, "rec-GW-REC-020": 13.0,
           "rec-GW-REC-022": 14.0}  # fmt: skip
PAGE_MS = {"v1-torch": 1700.0, "v1-onnx": 1600.0, "int8-recognizer": 1300.0,
           "fused-attention": 1400.0, "det-long-side-1024": 1000.0, "det-long-side-1280": 1200.0,
           "det-long-side-1792": 2000.0, "det-long-side-2048": 2400.0,
           "onnx-det-torch-rec": 1650.0, "det-GW-DET-003": 1100.0}  # fmt: skip


def iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")  # fmt: skip


def edit(path: Path, change) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    dump(path, doc)


def times(mean: float, seed: int, n: int = 100, sd: float = 0.02) -> list[float]:
    rng = np.random.default_rng(seed)
    return [round(float(v), 3) for v in mean * (1 + sd * rng.standard_normal(n))]


def thread_record(n: int | str = 10) -> dict[str, Any]:
    if n == "default":
        return {"requested": "default", "intra_op": 16, "inter_op": 16,
                "env": dict.fromkeys(THREAD_ENV), "env_before_torch_import": None,
                "preloaded": []}  # fmt: skip
    return {"requested": n, "intra_op": n, "inter_op": n, "env": dict.fromkeys(THREAD_ENV, str(n)),
            "env_before_torch_import": True, "preloaded": []}  # fmt: skip


def live(n: int | str = 10) -> dict[str, Any]:
    rec = thread_record(n)
    return {k: rec[k] for k in ("intra_op", "inter_op", "env")}


def session(n: int | None = 10) -> dict[str, Any]:
    return {"onnxruntime": "1.31.1", "providers": ["CPUExecutionProvider"],
            "intra_op_num_threads": n or 0, "inter_op_num_threads": 0,
            "execution_mode": "ORT_SEQUENTIAL", "graph_optimization_level": "ORT_ENABLE_ALL",
            "enable_cpu_mem_arena": False, "enable_mem_pattern": False,
            "allow_spinning": "default"}  # fmt: skip


def conditions(start: datetime) -> dict[str, Any]:
    return {"conditions": dict(COND), "conditions_end": dict(COND),
            "conditions_utc": {"start": iso(start), "end": iso(start + RUN)},
            "gpu_state": {"start": GPU, "end": GPU}}  # fmt: skip


class World:
    """A repository root with every Phase 16 input, and the hashes the files must agree on."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.out = root / ps.OUT
        self.slot = 0
        record = {"name": "glypharis_v1",
                  "recognizer": {"experiment": "GW-REC-034", "checkpoint_sha256": "c" * 64,
                                 "parameters": 7928835,
                                 "input": {"height": 48, "max_width": 1536}},
                  "detector": {"experiment": "GW-DET-001", "parameters": 3187209}}  # fmt: skip
        rec_path = root / ps.FREEZE_RECORD
        rec_path.parent.mkdir(parents=True)
        rec_path.write_text(yaml.safe_dump(record), encoding="utf-8")
        self.record_sha = sha(rec_path)
        self.write_bundle()
        self.write_reference()

    def next_slot(self) -> datetime:
        t = T0 + self.slot * SLOT
        self.slot += 1
        return t

    # --------------------------------------------------------------------------- GW-EFF-006

    def write_bundle(self) -> None:
        folder = self.root / ps.BUNDLE_DIR
        folder.mkdir(parents=True)
        for name, size in (("detector.pt", 3), ("detector.onnx", 3), ("recognizer.pt", 8),
                           ("recognizer.onnx", 8)):  # fmt: skip
            (folder / name).write_bytes(name.encode() * (size * 1000))
        self.files = {n: sha(folder / n) for n in ps.BUNDLE_FILES}
        dump(folder / "manifest.json", {"format": "glypharis-bundle-1", "version": "glypharis_v1",
                                        "files": self.files})  # fmt: skip
        self.bundle_sha = sha(folder / "manifest.json")
        dump(self.out / ps.BUNDLE_REPORT, {
            "experiment": "GW-EFF-006", "bundle": "runs/release/glypharis-v1",
            "version": "glypharis_v1", "manifest_sha256": self.bundle_sha, "files": self.files,
            "record": "configs/final/glypharis_v1.yaml", "record_sha256": self.record_sha,
            "equals_record": True, "v1_code": V1_CODE,
            "created_utc": iso(T0 - timedelta(days=2))})  # fmt: skip

    def write_reference(self) -> None:
        manifest = {"experiment": "GW-EFF-006", "provenance": {"record_sha256": self.record_sha},
                    "files": {"detector_maps.npz": "d" * 64}}  # fmt: skip
        dump(self.out / ps.REF_MANIFEST, manifest)
        dump(self.out / ps.REF_CHECK, {
            "experiment": "GW-EFF-006", "bundle": "runs/release/glypharis-v1",
            "bundle_manifest_sha256": self.bundle_sha, "bundle_files": self.files,
            "reference_manifest_sha256": sha(self.out / ps.REF_MANIFEST), "complete": True,
            "limit": None, "started_utc": iso(T0 - timedelta(days=1)),
            "ended_utc": iso(T0 - timedelta(hours=22)), "threads": thread_record(10),
            "session_options": {"detector": session(), "recognizer": session()},
            "v1_code": V1_CODE,
            "detector": {"pages": 110, "prob_max_abs_diff": 3e-6, "pass": True},
            "recognizer": {"lines": 5000, "logit_max_abs_diff": 2e-5, "near_tie_differences": 0,
                           "pass": True},
            "end_to_end": {"pages": 400, "page_cer_torch": 0.243, "page_cer_onnx": 0.243,
                           "difference": 0.0, "pass": True},
            "tolerances": {
                "map_max_abs": {"limit": 1e-4, "measured": 3e-6, "pass": True},
                "logits_max_abs": {"limit": 1e-3, "measured": 2e-5, "pass": True},
                "text": {"limit": "other 0", "measured": {"near_ties": 0, "other": 0},
                         "pass": True},
                "page_cer": {"limit": 0.0005, "measured": 0.0, "pass": True}},
            "pass": True})  # fmt: skip

    # ------------------------------------------------------------------- B: v1 profiles

    def rec_profile(self, spec: ps.ProfileSpec, mean: float, start: datetime) -> dict[str, Any]:
        n = spec.threads if spec.threads is not None else "default"
        onnx = spec.backend == "onnx"
        t = times(mean * spec.batch, zlib.crc32(spec.name.encode()) % 1000)
        sizes = LINE_SIZES if spec.batch == 1 else [[s] * spec.batch for s in LINE_SIZES]
        model = self.files["recognizer.onnx"] if onnx else "c" * 64
        if spec.run is not None:
            model = "e" * 64
        return {
            "experiment": spec.experiment, "run": spec.run or "GW-REC-034",
            "model_name": "svtr", "image_height": 32 if spec.run else 48, "max_width": 1536,
            "frame_stride": 4, "parameters": 7928835,
            "model_file": {"kind": "onnx" if onnx else "checkpoint", "name": "m",
                           "sha256": model, "mib": 31.0},
            "eval_set": "hiertext_val", "backend": spec.backend, "device": spec.device,
            "precision": "fp32", "int8_ops": [], "batch": spec.batch,
            "padding": spec.padding, "warmup": 10,
            "runtime": "onnxruntime 1.31.1 (CPU)" if onnx else "torch 2.14.1 (CPU)",
            "settings": {"cpu_threads": spec.threads, "intra_op_threads": 10,
                         "inter_op_threads": 10},
            "torch_threads": 10, "threads": thread_record(n),
            "threads_check": {"before": live(n), "after": live(n)},
            "session_options": session(spec.threads) if onnx else None, "numerics": None,
            "started_utc": iso(start), "ended_utc": iso(start + RUN),
            "warmup_times_ms": t[:10], "times_ms": t,
            "per_line_ms": [v / spec.batch for v in t], "input_sizes": sizes,
            "mean_ms": float(np.mean(t)), "p95_ms": float(np.percentile(t, 95)),
            "mean_line_ms": float(np.mean(t)) / spec.batch,
            "lines_per_second": 1000 * spec.batch / float(np.mean(t)),
            "peak_mem_mib": 700.0,
            "peak_vram": ({"max_memory_allocated_mib": 300.0, "max_memory_reserved_mib": 400.0,
                           "nvidia_smi_process_mib_end": None,
                           "nvidia_smi_whole_gpu_used_mib_end": 900}
                          if spec.device == "cuda" else None),
            "v1_code": V1_CODE, **conditions(start), "cpu_busy_percent_after": 3.0,
            "git_commit": "f" * 40, "git_dirty": False,
        }  # fmt: skip

    def e2e_profile(self, spec: ps.ProfileSpec, mean: float, start: datetime) -> dict[str, Any]:
        n = spec.threads if spec.threads is not None else "default"
        onnx = spec.backend == "onnx"
        noise = times(1.0, zlib.crc32(spec.name.encode()) % 1000, sd=0.01)
        t = [round((mean - 600 + 15 * k) * e, 3) for k, e in zip(PAGE_LINES, noise, strict=True)]
        stages = [{s: v / len(STAGES) for s in STAGES} for v in t]
        bundle = ({"path": "glypharis-v1", "manifest_sha256": self.bundle_sha,
                   "version": "glypharis_v1", "files": self.files, "record_settings": "equal"}
                  if onnx else None)  # fmt: skip
        return {
            "system": "glypharis", "task": "e2e", "eval_set": "hiertext_val",
            "experiment": spec.experiment, "build": "v1", "backend": spec.backend,
            "device": spec.device, "version": "glypharis_v1",
            "settings": {"device": spec.device, "precision": "fp32", "backend": spec.backend,
                         "runtime": "onnxruntime 1.31.1 (CPU)" if onnx else "torch 2.14.1",
                         "cpu_threads": spec.threads, "intra_op_threads": 10,
                         "inter_op_threads": 10},
            "threads": thread_record(n),
            "threads_check": {"before": live(n), "after": live(n)},
            "session_options": ({"detector": session(spec.threads),
                                 "recognizer": session(spec.threads)}
                                if onnx else {"detector": None, "recognizer": None}),
            "batch": 1, "started_utc": iso(start), "ended_utc": iso(start + RUN), "warmup": 10,
            "warmup_times_ms": t[:10], "times_ms": t, "stages": STAGES,
            "warmup_stages_ms": stages[:10], "stages_ms": stages,
            "stage_sum_max_rel_diff": 0.001, "mean_file_decode_ms": 30.0,
            "mean_stages_ms": {s: float(np.mean([d[s] for d in stages])) for s in STAGES},
            "lines": PAGE_LINES, "input_sizes": PAGE_SIZES,
            "page_ids": [f"p{i}" for i in range(100)],
            "mean_ms": float(np.mean(t)), "p95_ms": float(np.percentile(t, 95)),
            "mean_lines": float(np.mean(PAGE_LINES)), "peak_mem_mib": 1500.0,
            "peak_vram": ({"max_memory_allocated_mib": 800.0, "max_memory_reserved_mib": 900.0,
                           "nvidia_smi_process_mib_end": None,
                           "nvidia_smi_whole_gpu_used_mib_end": 1500}
                          if spec.device == "cuda" else None),
            "staged_check": {"page_id": "p0", "lines": 20, "same_as_predict": True},
            "provenance": ({"record": "configs/final/glypharis_v1.yaml",
                            "record_sha256": self.record_sha, "name": "glypharis_v1",
                            "bundle_manifest_sha256": self.bundle_sha}
                           if onnx else {"record_sha256": self.record_sha}),
            "bundle": bundle, "v1_code": V1_CODE, **conditions(start),
            "cpu_busy_percent_after": 3.0, "git_commit": "f" * 40, "git_dirty": False,
        }  # fmt: skip

    def write_cost(self) -> None:
        means = {"lines_torch_cpu_10t": 18.0, "lines_onnx_cpu_10t": 20.0,
                 "lines_onnx_cpu_default": 22.0, "lines_torch_cuda_b1": 4.0,
                 "lines_torch_cuda_b32": 0.5, "pages_torch_cpu_10t": 1700.0,
                 "pages_onnx_cpu_10t": 1600.0, "pages_onnx_cpu_default": 1800.0,
                 "pages_torch_cuda": 600.0}  # fmt: skip
        for spec in (*ps.COST, *ps.H6_TORCH):
            mean = means.get(spec.name, 11.0)
            make = self.rec_profile if spec.task == "rec" else self.e2e_profile
            dump(self.out / ps.PROFILES / spec.file, make(spec, mean, self.next_slot()))

    # ------------------------------------------------------------------ GW-EFF-010 sitting

    def write_h1(self, base_ms: float = 3000.0, glyph_ms: float = 2000.0) -> None:
        start = self.next_slot()
        t = times(base_ms, 7)
        base = {
            "system": "ppocr_mobile", "task": "e2e", "eval_set": "hiertext_val",
            "experiment": "GW-BASE-002", "version": "paddleocr 3.7.0",
            "settings": {"device": "cpu", "cpu_threads": 10, "precision": "fp32"},
            "batch": 1, "started_utc": iso(start), "ended_utc": iso(start + RUN), "warmup": 10,
            "warmup_times_ms": t[:10], "times_ms": t, "input_sizes": PAGE_SIZES,
            "process_peak_mib": 1900.0, "child_peak_mib": 0.0, "peak_mem_mib": 1900.0,
            "conditions": dict(COND), "cpu_busy_percent_after": 3.0, "git_commit": "f" * 40,
            "git_dirty": False, "phase": 16, "amends": "GW-BASE-002 (Phase 16 amendment)",
            "phase14_settings_check": {"result": "equal", "reference": "x", "fields": []},
            "conditions_end": dict(COND),
            "conditions_utc": {"start": iso(start), "end": iso(start + RUN)},
            "gpu_state": {"start": GPU, "end": GPU}, "model_mib": 25.41,
        }  # fmt: skip
        dump(self.baseline_path, base)
        dump(self.out / ps.PROFILES / ps.H1_GLYPH.file,
             self.e2e_profile(ps.H1_GLYPH, glyph_ms, self.next_slot()))  # fmt: skip

    @property
    def baseline_path(self) -> Path:
        return (self.root / "data/processed/baselines_phase16/GW-BASE-002/"
                "profile_ppocr_mobile_e2e_hiertext_val.json")  # fmt: skip

    # --------------------------------------------------------------------- C: variants

    def build(self, name: str) -> dict[str, Any]:
        spec = VARIANTS[name]
        det, rec = RUNTIMES.get(name, (ORT_FP32, ORT_FP32))
        files: dict[str, Any] = {}
        doc: dict[str, Any] = {
            "variant": name,
            "bundle": None
            if name == "v1-torch"
            else {"path": "glypharis-v1", "manifest_sha256": self.bundle_sha, "files": self.files},
        }
        if name == "int8-recognizer":
            files["recognizer.int8.onnx"] = {"sha256": "8" * 64, "bytes": 3_000_000}
            doc["quantization"] = {
                "recipe": "H7′",
                "onnxruntime": "1.31.1",
                "onnx": "1.20",
                "input": {"path": "recognizer.onnx", "sha256": "9" * 64, "bytes": 8_000_000},
                "output": {"path": "recognizer.int8.onnx", "sha256": "8" * 64, "bytes": 3_000_000},
                "nodes_fp32": {"MatMul": 40, "MatMulInteger": 0},
                "nodes_int8": {"MatMul": 8, "MatMulInteger": 32},
            }
        doc["files"] = files
        height = 32 if name.startswith("rec-") else 48
        settings = {"detector.runtime": det, "recognizer.runtime": rec,
                    "detector.infer": {"long_side": spec.long_side or 1536},
                    "recognizer.model": {"experiment": spec.checkpoint or "GW-REC-034",
                                         "height": height}}  # fmt: skip
        return {"variant": name, "version": ps.V1_LABEL if spec.setting is None else
                ps.VARIANT_LABEL, "setting": spec.setting, "about": spec.about,
                "changed": None if spec.setting is None else {spec.setting: {"from": 1, "to": 2}},
                "settings": settings, "v1_settings": settings, "variant_json": doc,
                "variant_json_sha256": hashlib.sha256(name.encode()).hexdigest(),
                "provenance": {"record_sha256": self.record_sha}}  # fmt: skip

    def common(self, name: str) -> dict[str, Any]:
        spec = VARIANTS[name]
        det, rec = RUNTIMES.get(name, (ORT_FP32, ORT_FP32))
        return {"experiment": spec.experiment, "variant": name,
                "version": ps.V1_LABEL if spec.setting is None else ps.VARIANT_LABEL,
                "setting": spec.setting, "changed": None, "eval_set": "hiertext_val",
                "device": "cpu", "threads": thread_record(10),
                "session_options": {
                    "detector": session() if det.startswith("onnxruntime") else None,
                    "recognizer": session() if rec.startswith("onnxruntime") else None},
                "v1_code": V1_CODE, "build": self.build(name), "git_commit": "f" * 40,
                "git_dirty": False}  # fmt: skip

    def variant_profile(self, name: str, unit: str, run: int, start: datetime,
                        mean: float | None = None,
                        experiment: str = ps.PARETO) -> None:  # fmt: skip
        det, rec = RUNTIMES.get(name, (ORT_FP32, ORT_FP32))
        mean = mean if mean is not None else (LINE_MS if unit == "lines" else PAGE_MS)[name]
        t = times(mean, 100 * run + len(name) + (0 if unit == "lines" else 50))
        doc = {"kind": "phase16-variant-profile", "system": "glypharis",
               "task": "e2e" if unit == "pages" else "rec", **self.common(name), "set": unit,
               "experiment": experiment, "run": run, "smoke": False,
               "settings": {"cpu_threads": 10, "intra_op_threads": 10, "inter_op_threads": 10,
                            "device": "cpu", "runtime": {"detector": det, "recognizer": rec},
                            "detector_long_side": 1536, "timing": "t"},
               "threads_before": live(), "threads_after": live(), "batch": 1, "warmup": 10,
               "started_utc": iso(start), "ended_utc": iso(start + RUN),
               "warmup_times_ms": t[:10], "times_ms": t,
               "mean_ms": float(np.mean(t)), "p95_ms": float(np.percentile(t, 95)),
               "peak_mem_mib": 1200.0 + run, **conditions(start),
               "cpu_busy_percent_after": 3.0}  # fmt: skip
        if unit == "pages":
            doc.update(
                parts_ms=[{"total": v} for v in t],
                lines=PAGE_LINES,
                input_sizes=PAGE_SIZES,
                mean_lines=float(np.mean(PAGE_LINES)),
            )
        else:
            doc.update(keys=LINE_KEYS, input_sizes=LINE_SIZES)
        dump(self.out / ps.VARIANT_DIR / name / ps.variant_profile_file(unit, run, experiment),
             doc)  # fmt: skip

    def variant_lines(self, name: str, extra_errors: int = 0, check_pass: bool = True) -> None:
        i = np.arange(ps.LINES)
        ref = 20 + i % 30
        err = 2 * (i % 17 == 0) + (i % 500 == 0) * extra_errors
        if name.startswith("rec-"):
            err = err + (i % 23 == 0)
        check = None
        if name in ("v1-onnx", "fused-attention") or name.startswith("rec-"):
            check = {"reference": "torch", "lines": ps.LINES, "logit_max_abs_diff": 3e-5,
                     "identical_text": ps.LINES, "near_tie_differences": 0,
                     "other_differences": [], "tolerances": {}, "pass": check_pass}  # fmt: skip
        doc = {"kind": "phase16-variant-lines", **self.common(name), "set": "lines",
               "source": {"pack": "processed/hiertext/lines_validation", "seed": 20261028},
               "smoke": False, "limit": None, "started_utc": iso(T0), "ended_utc": iso(T0),
               "lines": ps.LINES, "batch": 1, "cer": {"strict": float(err.sum() / ref.sum())},
               "per_line": {"keys": [f"k{j}" for j in i], "index": [int(j) for j in i],
                            "strict": {"errors": err.tolist(), "ref_chars": ref.tolist()},
                            "normalised": {"errors": err.tolist(), "ref_chars": ref.tolist()}},
               "torch_check": check,
               "outputs": OUTPUTS}  # fmt: skip
        dump(self.out / ps.VARIANT_DIR / name / "lines.json", doc)

    def variant_pages(self, name: str, offset: int) -> None:
        i = np.arange(ps.PAGES)
        counts = {"bag_errors": (200 + i % 7 + offset).tolist(),
                  "bag_ref_chars": (1000 + i).tolist(), "e2e_tp": [30] * ps.PAGES,
                  "e2e_pred": [40] * ps.PAGES, "e2e_gt": [42] * ps.PAGES,
                  "det_tp": [35] * ps.PAGES, "det_fp": [5] * ps.PAGES,
                  "det_fn": [7] * ps.PAGES}  # fmt: skip
        doc = {"kind": "phase16-variant-pages", **self.common(name), "set": "pages",
               "source": {}, "smoke": False, "limit": None, "started_utc": iso(T0),
               "ended_utc": iso(T0), "pages": ps.PAGES, "page_cer": 0.2, "e2e_f1": 0.73,
               "metrics": {}, "per_page": {"doc_ids": DOC_IDS, "strict": counts,
                                           "normalised": counts},
               "outputs": OUTPUTS}  # fmt: skip
        dump(self.out / ps.VARIANT_DIR / name / "pages.json", doc)

    def write_variants(self) -> None:
        # Each sitting in its registered order: H7′, H6′, then the Pareto points' two sittings.
        for experiment, units in ps.SITTINGS.items():
            for unit in units:
                for n, r in ps.sitting_order(experiment, unit):
                    self.variant_profile(n, unit, r, self.next_slot(), experiment=experiment)
        for k, (name, spec) in enumerate(VARIANTS.items()):
            for unit in spec.measure:
                if unit == "lines":
                    self.variant_lines(name, extra_errors=1 if name == "int8-recognizer" else 0)
                else:
                    self.variant_pages(name, k % 5)

    def write_baselines(self) -> None:
        dump(self.root / ps.BASELINE_RESULTS, {"predictions": [], "efficiency": [
            {"experiment": "GW-BASE-005", "system": "ppocr6_tiny", "task": "e2e",
             "eval_set": "hiertext_val", "mean_ms": 1726.6, "p95_ms": 2729.0,
             "peak_mem_mib": 1836.0, "model_mib": 10.0, "threads": {"cpu_threads": 10}},
            {"experiment": "GW-BASE-005", "system": "ppocr6_tiny", "task": "rec",
             "eval_set": "hiertext_val", "mean_ms": 21.6, "p95_ms": 66.4,
             "peak_mem_mib": 800.0, "model_mib": 5.0, "threads": {"cpu_threads": 10}},
            {"experiment": "GW-BASE-005", "system": "ppocr6_tiny", "task": "rec",
             "eval_set": "glyphsynth_val", "mean_ms": 99.0, "p95_ms": 99.0,
             "peak_mem_mib": 1.0, "model_mib": 5.0, "threads": {"cpu_threads": 10}},
        ]})  # fmt: skip

    def variant_path(self, name: str, file: str) -> Path:
        return self.out / ps.VARIANT_DIR / name / file


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> World:
    monkeypatch.setattr(ps, "LINES", 600)  # the scorer's rules at a size that runs quickly
    w = World(tmp_path / "repo")
    w.write_h1()
    w.write_cost()
    w.write_variants()
    w.write_baselines()
    return w


def score(w: World) -> dict[str, Any]:
    return ps.Scorer(ps.Layout.at(w.root)).run()


def flat(res: dict[str, Any]) -> list[str]:
    return [p for ps_ in res["problems"].values() for p in ps_]


# ------------------------------------------------------------------------------ the whole


def test_a_complete_set_scores_clean_and_writes_every_report(world, capsys):
    assert ps.main(["check", "--root", str(world.root)]) == 0
    assert ps.main(["score", "--root", str(world.root)]) == 0
    out = world.out
    for name in ("README.md", "cost_table.md", "int8.md", "pareto.md", "verdicts.md"):
        assert (out / name).read_text(encoding="utf-8").startswith(ps.NOTE), name
    bench = world.root / ps.BENCH
    text = (bench / "efficiency_comparison.md").read_text(encoding="utf-8")
    assert text.startswith(ps.NOTE) and ps.SAME_SITTING in text
    assert "PP-OCRv6 tiny" in text and "GLYPHARIS v1 fp32 ONNX, CPU, 10 threads" in text
    rows = json.loads((bench / "efficiency_comparison.json").read_text(encoding="utf-8"))["rows"]
    assert {r["sitting"] for r in rows} == {"Phase 5", "Phase 16"}
    assert all(r["experiment"] != "GW-BASE-005" or r["mean_ms"] != 99.0 for r in rows)
    assert (out / "pareto.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["missing"] == [] and res["problems"] == {}
    v = res["verdicts"]
    assert [v[k]["verdict"] for k in (*ps.CONFIRMATORY, "H7′")] == ["met"] * 4 + ["holds"]
    # Every input is listed with its hash; the baseline profile by its data path.
    base = "data/processed/baselines_phase16/GW-BASE-002/profile_ppocr_mobile_e2e_hiertext_val.json"
    assert res["inputs"][base] == sha(world.baseline_path)
    assert res["inputs"]["reports/efficiency/phase16/reference_check.json"]
    # Four confirmatory intervals, printed once each even where two reports print them.
    counts = res["intervals"]
    assert counts["confirmatory"] == 4
    assert counts["printed"] == counts["confirmatory"] + counts["descriptive"]
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert f"Intervals printed in these reports: {counts['printed']} (4 confirmatory" in readme
    printed = sum(t.count(" [") for t in (
        (out / n).read_text(encoding="utf-8") for n in ("cost_table.md", "pareto.md",
                                                       "verdicts.md")))  # fmt: skip
    assert printed >= counts["printed"] - 2  # int8.md repeats two pareto intervals and H7′'s
    # The variants are labelled as new versions in the table and the plot's legend.
    pareto = (out / "pareto.md").read_text(encoding="utf-8")
    assert "| int8-recognizer | variant (new version, not v1) |" in pareto
    assert "| v1-onnx | v1 |" in pareto
    assert "verdicts: H7′ latency met" in capsys.readouterr().out


def test_cost_table_numbers_come_from_the_profiles(world):
    res = score(world)
    rows = res["cost"]["rows"]
    p = json.loads((world.out / ps.PROFILES / "pages_onnx_cpu_10t.json").read_text())
    r = rows["pages_onnx_cpu_10t"]
    assert r["mean_ms"] == pytest.approx(np.mean(p["times_ms"]))
    assert r["p95_ms"] == pytest.approx(np.percentile(p["times_ms"], 95))
    assert r["mean_ci"]["unit"] == "pages" and r["mean_ci"]["low"] < r["mean_ms"]
    slope = np.polyfit(p["lines"], p["times_ms"], 1)[0]
    assert r["slope_ms_per_line"]["value"] == pytest.approx(slope)
    assert r["slope_ms_per_line"]["low"] < slope < r["slope_ms_per_line"]["high"]
    assert sum(r["stages"].values()) == pytest.approx(r["mean_ms"])
    b32 = rows["lines_torch_cuda_b32"]
    assert b32["batch"] == 32 and b32["mean_line_ms"] == pytest.approx(b32["mean_ms"] / 32)
    assert rows["lines_torch_cuda_b1"]["peak_vram"]["max_memory_allocated_mib"] == 300.0
    assert res["cost"]["parameters"] == {"detector": 3187209, "recognizer": 7928835,
                                         "total": 11116044}  # fmt: skip
    sizes = res["gw_eff_006"]["files"]
    assert sizes["recognizer.onnx"]["mib"] == pytest.approx(len(b"recognizer.onnx") * 8000 / 2**20)
    adr = {r["build"]: r["targets"] for r in res["adr_0011"]}
    onnx = {t["target"]: t for t in adr["v1 fp32 ONNX, CPU, 10 threads"]}
    assert onnx["line mean"]["within"] is True and onnx["page mean"]["within"] is False
    assert onnx["peak RAM on pages"]["value"] == 1500.0 and onnx["peak RAM on pages"]["within"]


def test_the_verdicts_follow_the_registered_rules(world):
    res = score(world)
    v = res["verdicts"]

    def load(name: str, file: str) -> dict[str, Any]:
        return json.loads(world.variant_path(name, file).read_text())

    def mean2(name: str, experiment: str) -> np.ndarray:
        runs = [load(name, ps.variant_profile_file("lines", r, experiment))["times_ms"]
                for r in (1, 2)]  # fmt: skip
        return np.mean(np.asarray(runs), axis=0)

    h7, h6 = ps.H7_SITTING, ps.H6_SITTING
    want = mean_ci(mean2("int8-recognizer", h7) - 0.75 * mean2("v1-onnx", h7), "lines")
    assert v["H7′ latency"]["test"]["value"] == pytest.approx(want.value)
    assert v["H7′ latency"]["test"]["high"] == pytest.approx(want.high)
    a, b = load("int8-recognizer", "lines.json"), load("v1-onnx", "lines.json")
    e8, r = (np.asarray(a["per_line"]["strict"][k]) for k in ("errors", "ref_chars"))
    e32 = np.asarray(b["per_line"]["strict"]["errors"])
    want = paired_ratio_diff_ci(e8 - 0.005 * r, e32, r, "lines")
    assert v["H7′ accuracy"]["test"] == pytest.approx(ps.iv(want))
    want = mean_ci(mean2("rec-GW-REC-020", h6) - 1.5 * mean2("rec-GW-REC-003", h6), "lines")
    assert v["H6′"]["test"]["value"] == pytest.approx(want.value)
    base = json.loads(world.baseline_path.read_text())
    glyph = json.loads((world.out / ps.PROFILES / ps.H1_GLYPH.file).read_text())
    want = mean_ci(np.asarray(glyph["times_ms"]) - np.asarray(base["times_ms"]), "pages")
    assert v["H1′ ONNX"]["test"]["value"] == pytest.approx(want.value)
    assert v["H1′ ONNX"]["bundle_manifest_sha256"] == world.bundle_sha
    assert v["H7′"]["quantization"]["nodes_int8"]["MatMulInteger"] == 32


def test_not_met_and_not_shown(world):
    h7, h6 = ps.H7_SITTING, ps.H6_SITTING
    for r in (1, 2):  # INT8 at 16 ms against 0.75 x 20 = 15: above 0
        path = world.variant_path("int8-recognizer", ps.variant_profile_file("lines", r, h7))
        start = ps._time(json.loads(path.read_text())["started_utc"])
        world.variant_profile("int8-recognizer", "lines", r, start, mean=16.0, experiment=h7)
    for r in (1, 2):  # SVTR at 15 ms against 1.5 x 10: the interval straddles 0
        path = world.variant_path("rec-GW-REC-020", ps.variant_profile_file("lines", r, h6))
        start = ps._time(json.loads(path.read_text())["started_utc"])
        world.variant_profile("rec-GW-REC-020", "lines", r, start, mean=15.0, experiment=h6)
    v = score(world)["verdicts"]
    assert v["H7′ latency"]["verdict"] == "not met"
    assert v["H7′"]["verdict"] == "does not hold"
    assert v["H6′"]["verdict"] == "not shown"


# --------------------------------------------------------------------------- the sittings


def _shift(path: Path, by: timedelta) -> None:
    def change(doc: dict[str, Any]) -> None:
        for k in ("started_utc", "ended_utc"):
            doc[k] = iso(ps._time(doc[k]) + by)

    edit(path, change)


def test_h7_latency_is_void_out_of_order_or_across_sittings(world):
    fp32 = world.variant_path("v1-onnx", "profile_lines_10t_GW-EFF-008_run1.json")
    int8 = world.variant_path("int8-recognizer", "profile_lines_10t_GW-EFF-008_run1.json")
    _shift(fp32, SLOT)  # fp32 run 1 now starts after INT8 run 1
    _shift(int8, -SLOT)
    v = score(world)["verdicts"]["H7′ latency"]
    assert v["verdict"] == "void"
    assert any("started before" in m for m in v["void"])
    _shift(fp32, -SLOT)
    _shift(int8, SLOT)
    for n, r in ps.H7_ORDER[2:]:
        _shift(world.variant_path(n, ps.variant_profile_file("lines", r, ps.H7_SITTING)),
               timedelta(hours=3))  # fmt: skip
    v = score(world)["verdicts"]["H7′ latency"]
    assert v["verdict"] == "void" and any("not one sitting" in m for m in v["void"])


def test_h7_accuracy_needs_the_same_lines(world):
    edit(world.variant_path("int8-recognizer", "lines.json"),
         lambda d: d["per_line"]["keys"].reverse())  # fmt: skip
    res = score(world)
    assert res["verdicts"]["H7′ accuracy"]["verdict"] == "void"
    assert res["verdicts"]["H7′"]["verdict"] == "pending"


def test_h6_is_not_tested_unless_both_export_checks_pass(world):
    world.variant_lines("rec-GW-REC-003", check_pass=False)
    v = score(world)["verdicts"]["H6′"]
    assert v["verdict"] == "not tested"
    assert any("export check" in m for m in v["reasons"])


def test_h6_needs_its_export_checks_before_the_sitting(world):
    first = world.variant_path("rec-GW-REC-003",
                               ps.variant_profile_file("lines", 1, ps.H6_SITTING))  # fmt: skip
    began = ps._time(json.loads(first.read_text())["started_utc"])
    edit(world.variant_path("rec-GW-REC-020", "lines.json"),
         lambda d: d.update(ended_utc=iso(began + timedelta(minutes=1))))  # fmt: skip
    v = score(world)["verdicts"]["H6′"]
    assert v["verdict"] == "void"
    assert any("rec-GW-REC-020: its export check" in m and "not before" in m for m in v["void"])


def test_a_sitting_file_must_name_its_sitting(world):
    path = world.variant_path("v1-onnx", ps.variant_profile_file("lines", 2, ps.H7_SITTING))
    edit(path, lambda d: d.update(experiment="GW-EFF-009"))
    res = score(world)
    assert res["verdicts"]["H7′ latency"]["verdict"] == "void"
    assert any("experiment 'GW-EFF-009', not GW-EFF-008" in m
               for m in res["verdicts"]["H7′ latency"]["void"])  # fmt: skip


def test_accuracy_lines_read_in_batches_are_refused(world):
    edit(world.variant_path("int8-recognizer", "lines.json"), lambda d: d.update(batch=256))
    res = score(world)
    assert res["verdicts"]["H7′ accuracy"]["verdict"] == "void"
    assert any("not one at a time" in m for m in res["verdicts"]["H7′ accuracy"]["void"])


def test_h1_onnx_follows_phase_14s_sitting_rule(world):
    glyph = world.out / ps.PROFILES / ps.H1_GLYPH.file
    _shift(glyph, -SLOT)  # starts with the baseline still running
    v = score(world)["verdicts"]["H1′ ONNX"]
    assert v["verdict"] == "void"
    assert any("GW-EFF-010 started before GW-BASE-002 ended" in m for m in v["void"])
    _shift(glyph, SLOT + timedelta(hours=3))
    v = score(world)["verdicts"]["H1′ ONNX"]
    assert any("more than 2 hours" in m for m in v["void"])
    _shift(glyph, -timedelta(hours=3))
    edit(world.baseline_path, lambda d: d["input_sizes"].reverse())
    v = score(world)["verdicts"]["H1′ ONNX"]
    assert v["void"] == ["the input sizes differ page by page"]


def test_h1_onnx_needs_the_gw_eff_006_bundle_and_its_passed_check(world):
    edit(world.out / ps.PROFILES / ps.H1_GLYPH.file,
         lambda d: d["bundle"].update(manifest_sha256="0" * 64))  # fmt: skip
    v = score(world)["verdicts"]["H1′ ONNX"]
    assert v["verdict"] == "void" and any("not the GW-EFF-006 bundle" in m for m in v["void"])
    edit(world.out / ps.PROFILES / ps.H1_GLYPH.file,
         lambda d: d["bundle"].update(manifest_sha256=world.bundle_sha))  # fmt: skip
    edit(world.out / ps.REF_CHECK, lambda d: d.update({"pass": False}))
    res = score(world)
    assert res["gw_eff_006"]["bundle_passes"] is False
    for k in ("H1′ ONNX", "H7′ latency", "H7′ accuracy"):
        assert any("reference check" in m for m in res["verdicts"][k]["void"]), k


def test_h1_onnx_needs_the_bundle_named_and_checked_before_the_sitting(world):
    began = json.loads(world.baseline_path.read_text())["started_utc"]
    later = iso(ps._time(began) + timedelta(minutes=1))
    edit(world.out / ps.BUNDLE_REPORT, lambda d: d.update(created_utc=later))
    edit(world.out / ps.REF_CHECK, lambda d: d.update(ended_utc=later))
    v = score(world)["verdicts"]["H1′ ONNX"]
    assert v["verdict"] == "void"
    assert any("named by its SHA-256 before the sitting" in m for m in v["void"])
    assert any("reference check did not end before" in m for m in v["void"])
    edit(world.out / ps.BUNDLE_REPORT, lambda d: d.pop("created_utc"))
    v = score(world)["verdicts"]["H1′ ONNX"]
    assert any("created_utc None" in m for m in v["void"])


def test_h1_onnx_refuses_a_baseline_profile_that_is_not_phase_16s(world):
    edit(world.baseline_path, lambda d: d.pop("phase14_settings_check"))
    v = score(world)["verdicts"]["H1′ ONNX"]
    assert any("Phase 14 profile" in m for m in v["void"])


# ------------------------------------------------------------------------ file checks


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["threads"].update(env_before_torch_import=False), "before torch"),
        (lambda d: d["session_options"]["detector"].update(intra_op_num_threads=4),
         "session options"),
        (lambda d: d["session_options"]["recognizer"].update(enable_cpu_mem_arena=True),
         "session options"),
        (lambda d: d["threads_check"]["after"].update(intra_op=8), "torch pools"),
        (lambda d: d["conditions_end"].update(on_ac_power=False), "mains power at the end"),
        (lambda d: d["conditions"].update(power_plan="High performance"), "Balanced"),
        (lambda d: d["times_ms"].pop(), "a time is missing"),
        (lambda d: d["v1_code"].update(v1_code="differs"), "v1 code"),
        (lambda d: d.update(experiment="GW-EFF-099"), "experiment"),
    ],
)  # fmt: skip
def test_a_profile_with_a_failed_condition_is_left_out(world, change, message):
    edit(world.out / ps.PROFILES / "pages_onnx_cpu_10t.json", change)
    res = score(world)
    assert res["cost"]["rows"]["pages_onnx_cpu_10t"] is None
    assert any(message in p for p in res["problems"]["GW-EFF-007"]), res["problems"]


def test_default_threads_profiles_check_the_onnx_default(world):
    res = score(world)
    assert res["cost"]["rows"]["lines_onnx_cpu_default"] is not None
    edit(world.out / ps.PROFILES / "lines_onnx_cpu_default.json",
         lambda d: d["session_options"].update(intra_op_num_threads=10))  # fmt: skip
    res = score(world)
    assert res["cost"]["rows"]["lines_onnx_cpu_default"] is None


def test_v1_line_profiles_must_have_their_registered_padding(world):
    edit(world.out / ps.PROFILES / "lines_onnx_cpu_10t.json", lambda d: d.update(padding="phase7"))
    res = score(world)
    assert res["cost"]["rows"]["lines_onnx_cpu_10t"] is None
    assert any("padding 'phase7', not 'pipeline'" in p for p in res["problems"]["GW-EFF-007"])
    assert res["cost"]["rows"]["lines_torch_cpu_10t"]["padding"] == "pipeline"
    info = res["verdicts"]["H6′"]["information_pytorch_10t"]
    assert set(info) == {"GW-REC-003", "GW-REC-020"}  # Phase 7's input, as registered


def test_v1_line_profiles_must_run_v1s_models(world):
    edit(world.out / ps.PROFILES / "lines_onnx_cpu_10t.json",
         lambda d: d["model_file"].update(sha256="0" * 64))  # fmt: skip
    edit(world.out / ps.PROFILES / "lines_torch_cpu_10t.json", lambda d: d.update(image_height=32))
    probs = score(world)["problems"]["GW-EFF-007"]
    assert any("recognizer.onnx" in p for p in probs)
    assert any("input height" in p for p in probs)


def test_smoke_runs_and_other_bundles_are_refused_for_variants(world):
    edit(world.variant_path("det-long-side-1024", "pages.json"), lambda d: d.update(smoke=True))
    edit(
        world.variant_path("fused-attention", ps.variant_profile_file("pages", 2)),
        lambda d: d["build"]["variant_json"]["bundle"].update(manifest_sha256="0" * 64),
    )
    res = score(world)
    pts = {(p["variant"], p["set"]): p for p in res["pareto"]["points"]}
    assert pts[("det-long-side-1024", "pages")]["accuracy"] is None
    assert pts[("fused-attention", "pages")]["latency"] is None
    probs = res["problems"]["GW-EFF-009"]
    assert any("smoke" in p for p in probs) and any("GW-EFF-006 bundle" in p for p in probs)


def test_overlapping_runs_void_the_latency_of_both(world):
    a = world.variant_path("det-long-side-1280", ps.variant_profile_file("pages", 1))
    b = world.variant_path("det-long-side-1792", ps.variant_profile_file("pages", 1))
    start = json.loads(a.read_text())["started_utc"]
    edit(b, lambda d: d.update(started_utc=start, ended_utc=iso(ps._time(start) + RUN)))
    res = score(world)
    pts = {(p["variant"], p["set"]): p for p in res["pareto"]["points"]}
    assert pts[("det-long-side-1280", "pages")]["latency"] is None
    assert pts[("det-long-side-1792", "pages")]["latency"] is None
    assert pts[("det-long-side-2048", "pages")]["latency"] is not None
    assert any("same time" in p for p in res["problems"]["GW-EFF-009"])


def test_gw_eff_006_checks_hashes(world):
    (world.root / ps.BUNDLE_DIR / "recognizer.onnx").write_bytes(b"changed")
    edit(world.out / ps.REF_MANIFEST, lambda d: d.update(note="edited after the check"))
    res = score(world)
    probs = res["problems"]["GW-EFF-006"]
    assert any("recognizer.onnx: SHA-256" in p for p in probs)
    assert any("committed manifest" in p for p in probs)
    assert res["gw_eff_006"]["bundle_passes"] is False


def test_the_pareto_differences_are_paired_against_v1_onnx(world):
    res = score(world)
    pts = {(p["variant"], p["set"]): p for p in res["pareto"]["points"]}
    assert pts[("v1-onnx", "pages")]["difference"] is None
    d = pts[("det-long-side-1024", "pages")]["difference"]
    a, b = (
        json.loads(world.variant_path(n, "pages.json").read_text())["per_page"]["strict"]
        for n in ("det-long-side-1024", "v1-onnx")
    )
    want = paired_ratio_diff_ci(a["bag_errors"], b["bag_errors"], a["bag_ref_chars"], "pages")
    assert d["page_cer"] == pytest.approx(ps.iv(want))
    assert d["latency_ms"]["value"] < 0
    assert pts[("int8-recognizer", "lines")]["version"] == ps.VARIANT_LABEL
    assert pts[("v1-torch", "lines")]["version"] == "v1"
    assert res["pareto"]["sittings"]["pages"]["one_sitting"] is True
    # The Pareto line point of INT8 is the GW-EFF-009 sitting's, not H7′'s.
    lat = pts[("int8-recognizer", "lines")]["latency"]
    mine = [json.loads(world.variant_path("int8-recognizer", ps.variant_profile_file("lines", r))
                       .read_text())["mean_ms"] for r in (1, 2)]  # fmt: skip
    assert lat["run_means_ms"] == pytest.approx(mine)


def test_pareto_differences_need_one_mirrored_sitting(world):
    """Run 2 of v1-onnx swapped in time with run 2 of INT8, the arm before it in the registered
    order: no run overlaps another, but the order is broken, so no latency difference is given;
    the points and the accuracy differences stay."""
    a = world.variant_path("v1-onnx", ps.variant_profile_file("lines", 2))
    b = world.variant_path("int8-recognizer", ps.variant_profile_file("lines", 2))
    _shift(a, -SLOT)
    _shift(b, SLOT)
    res = score(world)
    assert not any("same time" in p for ps_ in res["problems"].values() for p in ps_)
    assert res["pareto"]["sittings"]["lines"]["one_sitting"] is False
    assert res["pareto"]["sittings"]["pages"]["one_sitting"] is True
    pts = {(p["variant"], p["set"]): p for p in res["pareto"]["points"]}
    d = pts[("int8-recognizer", "lines")]["difference"]
    assert "latency_ms" not in d and "cer" in d and d["latency_note"].startswith("not one")
    assert pts[("int8-recognizer", "lines")]["latency"] is not None
    assert any("not one mirrored sitting" in n for n in res["notes"]["GW-EFF-009"])
    assert "latency_ms" in pts[("int8-recognizer", "pages")]["difference"]
    ps.write_reports(ps.Layout.at(world.root), res)
    text = (world.out / "pareto.md").read_text(encoding="utf-8")
    assert "The lines runs of GW-EFF-009 are not one mirrored sitting" in text


def test_score_refuses_while_files_are_missing_unless_allowed(world):
    world.baseline_path.unlink()
    with pytest.raises(SystemExit, match="missing"):
        ps.main(["score", "--root", str(world.root)])
    assert not (world.out / "README.md").exists()
    assert ps.main(["check", "--root", str(world.root)]) == 1
    assert ps.main(["score", "--root", str(world.root), "--allow-missing"]) == 0
    res = json.loads((world.out / "results.json").read_text(encoding="utf-8"))
    assert res["verdicts"]["H1′ ONNX"]["verdict"] == "missing"
    assert res["intervals"]["confirmatory"] == 3
    readme = (world.out / "README.md").read_text(encoding="utf-8")
    assert "## Missing files" in readme and "baselines_phase16" in readme


def test_a_bundle_not_on_this_machine_is_a_note_not_a_problem(world):
    import shutil

    shutil.rmtree(world.root / ps.BUNDLE_DIR)
    res = score(world)
    assert "GW-EFF-006" not in res["problems"]
    assert any("not on this machine" in n for n in res["notes"]["GW-EFF-006"])
    assert res["gw_eff_006"]["bundle_passes"] is True


# ------------------------------------------------------------------------------ helpers


def test_phase_14s_helpers_are_imported_not_copied():
    assert ps.profile_problems is phase14_score.profile_problems
    assert ps.sitting_problems is phase14_score.sitting_problems


def test_slope_ci_on_a_straight_line():
    lines = [10, 20, 30, 40, 50] * 20
    t = [100 + 25 * k for k in lines]
    s = ps.slope_ci(lines, t)
    assert s.value == pytest.approx(25.0)
    assert s.low == pytest.approx(25.0) and s.high == pytest.approx(25.0)
    assert ps.slope_ci([5] * 10, list(range(10))).resamples == 1000  # flat lines: no slope


def test_order_and_overlap_helpers():
    def run(start: datetime) -> dict[str, Any]:
        return {"started_utc": iso(start), "ended_utc": iso(start + RUN)}

    ok = [("a", run(T0)), ("b", run(T0 + SLOT)), ("c", run(T0 + 2 * SLOT))]
    assert ps.order_problems(ok) == [] and ps.overlaps(ok) == []
    late = [("a", run(T0)), ("b", run(T0 + timedelta(hours=3)))]
    assert "not one sitting" in ps.order_problems(late)[0]
    same = [("a", run(T0)), ("b", run(T0 + timedelta(minutes=5)))]
    assert ps.overlaps(same) == [("a", "b")]
    assert "started before a ended" in ps.order_problems(same)[0]


def test_printer_counts_each_interval_once():
    pr = ps.Printer()
    d = {"value": 1.0, "low": 0.5, "high": 1.5}
    assert pr.ci("x", d, digits=1) == "1.0 [0.5, 1.5]"
    assert pr.ci("x", d, scale=100, digits=0, signed=True) == "+100 [+50, +150]"
    pr.ci("y", d, kind="confirmatory")
    assert pr.ci("z", None) == "n/a"
    assert pr.counts() == {"printed": 2, "confirmatory": 1, "descriptive": 1}
