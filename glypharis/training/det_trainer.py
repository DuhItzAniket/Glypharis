"""Trainer for the text-line detector (Phase 8): GLYPHARIS-Det on HierText pages.

`DetTrainer` is a `TrainerCore`, so the loop, checkpoints, exact resume and early-stopping
bookkeeping are the shared ones. This file adds what is specific to detection:

- start-up: a GPU headroom preflight (another process may hold the GPU), the page indexes with
  their seeded subsets and leakage check, one persistent training loader and one persistent
  validation loader, and a 20-page label-ceiling tripwire;
- a training step: bf16 forward, DB loss in fp32, with the loss parts and the empty-crop count
  logged;
- validation every epoch on a fixed seeded subset of validation pages plus a few training pages,
  every operating point scored from one fp32 pass (`glypharis.training.det_eval`), then the
  detection flags (`glypharis.diagnostics.det`). The monitored metric is ``val_f1_50``, the best
  F1@0.5 over the pre-registered operating points; the winner is stored in each checkpoint as
  ``selected_op``, the operating point of record (spec §7.3);
- `finish_det`: full validation of best.pt and last.pt at their stored operating point, the
  Phase 5 baseline page row, summary.json and the run report, written on every path.

Exit codes of `run_det`: 0 done, 2 non-finite loss or gradient, 3 out of GPU memory, 4 refused at
the GPU preflight. After 2 or 3 the state is saved, full validation is skipped and the report is
still written; after 4 nothing is written.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Final, Literal

import cv2
import shapely
import torch
import torchvision
from torch.utils.data import DataLoader

from glypharis.config.schema import ConfigError, RunConfig
from glypharis.data.det_dataset import (
    DET_DATA_VERSION,
    MAP_KEYS,
    TRAIN_EVAL_STREAM,
    TRAIN_SUBSET_STREAM,
    VAL_SUBSET_STREAM,
    WORKER_CONTEXT,
    CropBatchSampler,
    DetEvalDataset,
    DetTrainDataset,
    PageIndex,
    assert_no_page_leakage,
    collate_eval,
    collate_train,
    det_worker_init,
    read_gt_pages,
    seeded_subset,
)
from glypharis.data.det_labels import LABEL_VERSION, LabelStats
from glypharis.data.download import sha256_file
from glypharis.data.pages import Page
from glypharis.diagnostics import EpochPoint, Flag, NonFiniteLoss
from glypharis.diagnostics.det import (
    check_box_size,
    check_det_fitting,
    check_epoch_data,
    check_map_collapse,
    check_postprocess,
    check_slow_validation,
    check_threshold,
)
from glypharis.eval.det_matching import baseline_doc_ids
from glypharis.models.det.dbnet import DBNet, build_detector, count_parameters
from glypharis.models.det.detector import DetectorSpec
from glypharis.models.det.loss import DBLoss
from glypharis.training.checkpoint import load_checkpoint
from glypharis.training.det_eval import (
    DetEvalResult,
    OpMetrics,
    boxes_rows,
    evaluate_detector,
    label_ceiling_page,
    op_metrics_json,
    summarise_ceiling,
    write_records,
)
from glypharis.training.det_report import write_det_report
from glypharis.training.loop import (
    EPOCH_S,
    SAMPLES,
    WAIT_S,
    EpochResult,
    StepResult,
    TrainerCore,
    free_cuda,
)
from glypharis.utils.monitor import reset_peak
from glypharis.utils.seed import make_generator

log = logging.getLogger(__name__)

PREFLIGHT_MIN_FREE_GIB: Final = 4.0
TRIPWIRE_PAGES: Final = 20  # label-ceiling pages checked at start-up
LABEL_CEILING_LOW: Final = 0.85
POST_WORKERS: Final = 2  # post-processing threads during validation
_LABEL_FIELDS: Final = tuple(f.name for f in dataclasses.fields(LabelStats))


class GpuBusyError(RuntimeError):
    """Too little GPU memory at start-up: another process holds the GPU."""


def gpu_headroom_gib(device: torch.device) -> float:
    """GPU memory this process could use, in GiB.

    ``mem_get_info`` reports free device memory, which already excludes everything this process
    has reserved, cached blocks included; those cached but unallocated blocks are available to it,
    so they are added back. Memory held by live tensors is not.
    """
    free, _ = torch.cuda.mem_get_info(device)
    reserved, allocated = torch.cuda.memory_reserved(device), torch.cuda.memory_allocated(device)
    return float(free + reserved - allocated) / 2**30


@dataclass
class FullEval:
    tag: Literal["best", "last"]
    checkpoint_sha256: str
    stored_op: str  # the checkpoint's "selected_op": the operating point of record
    full_set_max_op: str  # arg-max over ops on the full validation set, information only
    result: DetEvalResult

    def headline(self) -> OpMetrics:
        return self.result.ops[self.stored_op]


def _shutdown(loader: DataLoader[Any] | None) -> None:
    """Stop a loader's persistent workers now instead of whenever it is garbage-collected."""
    if loader is None:
        return
    it = getattr(loader, "_iterator", None)
    if it is not None:
        stop = getattr(it, "_shutdown_workers", None)
        if stop is not None:
            stop()
        loader._iterator = None


class DetTrainer(TrainerCore):
    task: ClassVar[str] = "det"
    throughput_key: ClassVar[str] = "crops_per_s"
    model: DBNet

    def __init__(
        self, cfg: RunConfig, on_step: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        super().__init__(cfg, on_step)
        assert cfg.det is not None  # validated for task det
        self.det = det = cfg.det
        if self.device.type == "cuda":
            # Before anything is built, so a refused run leaves nothing behind.
            headroom = gpu_headroom_gib(self.device)
            if headroom < PREFLIGHT_MIN_FREE_GIB:
                raise GpuBusyError(
                    f"another process holds the GPU: {headroom:.2f} GiB available, "
                    f"{PREFLIGHT_MIN_FREE_GIB} GiB needed"
                )
        d = det.data
        if len(d.train_pages) != 1 or len(d.val_pages) != 1:
            raise ConfigError(
                "the detector trainer reads one training and one validation page "
                "file (det.data.train_pages, det.data.val_pages)"
            )
        self.train_file = self.data_root / d.train_pages[0]
        self.val_file = self.data_root / d.val_pages[0]
        train_all = PageIndex.build(self.train_file, "train", self.data_root)
        self.val_full_index = PageIndex.build(self.val_file, "validation", self.data_root)
        assert_no_page_leakage(train_all, self.val_full_index)
        self.train_index = seeded_subset(train_all, d.train_subset, cfg.seed, TRAIN_SUBSET_STREAM)
        self.val_index = seeded_subset(
            self.val_full_index, d.val_subset, cfg.seed, VAL_SUBSET_STREAM
        )
        self.train_eval_index = (
            seeded_subset(self.train_index, d.train_eval_subset, cfg.seed, TRAIN_EVAL_STREAM)
            if d.train_eval_subset
            else None
        )

        # Full pages only for what is scored each epoch, in the main process.
        self.gt_pages: dict[str, Page] = read_gt_pages(self.val_file, self.val_index.doc_ids())
        parts: list[tuple[str, PageIndex]] = [("val", self.val_index)]
        if self.train_eval_index is not None:
            self.gt_pages.update(read_gt_pages(self.train_file, self.train_eval_index.doc_ids()))
            parts.append(("train_eval", self.train_eval_index))
        self._full_gt: dict[str, Page] | None = None

        self.train_ds = DetTrainDataset(self.train_index, self.data_root, det, cfg.seed)
        self.sampler = CropBatchSampler(len(self.train_ds), cfg.train.batch_size, cfg.seed)
        self.eval_ds = DetEvalDataset(parts, self.data_root, det, with_labels=True)
        self.train_loader = self._train_loader()
        self.eval_loader = self._eval_loader(self.eval_ds, persistent=True)

        # A resumed or evaluated model gets every weight from its checkpoint.
        model = build_detector(
            det.model, self.data_root, load_pretrained=not cfg.checkpoint.resume
        ).to(self.device)
        self.loss_fn = DBLoss(det.loss)
        self.pool = ThreadPoolExecutor(max_workers=POST_WORKERS, thread_name_prefix="det-post")
        epochs = range(cfg.train.epochs)
        self.setup_optimization(model, [self.sampler.num_batches(e) for e in epochs])
        self.params = {
            "train": count_parameters(model),
            "inference": count_parameters(model, inference_only=True),
        }
        self._last_val: DetEvalResult | None = None

        self.tripwire = self._label_tripwire()
        if not cfg.checkpoint.resume and self.tripwire["f1_50"] < LABEL_CEILING_LOW:
            # A resumed run already carries this flag in its saved state.
            self._flag(
                Flag(
                    "LABEL CEILING LOW",
                    0,
                    0,
                    f"label ceiling F1@0.5 {self.tripwire['f1_50']:.3f} on "
                    f"{self.tripwire['pages']} validation pages is below "
                    f"{LABEL_CEILING_LOW}",
                )
            )

    # ---------------------------------------------------------------------------- loaders

    def _train_loader(self) -> DataLoader[Any]:
        # Created once per trainer: with persistent workers the batch sampler still runs in the
        # main process, so set_epoch before each iter() takes effect.
        w = self.det.data.num_workers
        return DataLoader(
            self.train_ds,
            batch_sampler=self.sampler,
            collate_fn=collate_train,
            num_workers=w,
            persistent_workers=w > 0,
            prefetch_factor=2 if w else None,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=det_worker_init if w else None,
            multiprocessing_context=WORKER_CONTEXT if w else None,
            generator=make_generator(self.cfg.seed + 1000),  # never touches the global RNG
        )

    def _eval_loader(self, ds: DetEvalDataset, persistent: bool) -> DataLoader[Any]:
        w = self.det.data.eval_workers
        return DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_eval,
            num_workers=w,
            persistent_workers=persistent and w > 0,
            prefetch_factor=2 if w else None,
            worker_init_fn=det_worker_init if w else None,
            multiprocessing_context=WORKER_CONTEXT if w else None,
            generator=make_generator(self.cfg.seed + 7),
        )

    def _label_tripwire(self) -> dict[str, Any]:
        """Label ceiling (strict) on the first validation-subset pages: a cheap start-up check
        that labels and post-processing still fit together."""
        ids = self.val_index.doc_ids()[:TRIPWIRE_PAGES]
        pages = [
            label_ceiling_page(self.gt_pages[i], self.det, ("strict",), check_parity=False)
            for i in ids
        ]
        names = [op.name for op in self.det.post.operating_points]
        s = summarise_ceiling(pages, names)
        best = s["selection"].get("strict", {"op": names[0], "f1_50": 0.0})
        per_op = {n: s["per_op"][f"strict/{n}"]["f1_50"]["value"] for n in names} if pages else {}
        return {"pages": len(pages), "op": best["op"], "f1_50": best["f1_50"], "per_op": per_op}

    # ----------------------------------------------------------------------- task interface

    def train_batches(self, epoch: int, start_batch: int) -> Iterable[dict[str, Any]]:
        self.sampler.set_epoch(epoch, start_batch)
        return self.train_loader

    def train_step(self, batch: dict[str, Any]) -> StepResult:
        dev = self.device
        # NHWC uint8 -> NCHW view with channels-last strides; the model normalises.
        images = batch["images"].to(dev, non_blocking=True).permute(0, 3, 1, 2)
        shrink, smask, tmap, tmask = (
            batch[k].to(dev, non_blocking=True).float().unsqueeze(1) for k in MAP_KEYS
        )
        with self.amp.autocast():
            logits = self.model.forward_train(images)
        parts = self.loss_fn(logits, shrink, smask, tmap, tmask)
        empty = ((shrink * smask).flatten(1).sum(1) == 0).sum()
        stats: LabelStats = batch["stats"]
        x = self.state.extra
        for name in _LABEL_FIELDS:  # plain ints from the workers: no sync
            x[f"label_{name}"] = x.get(f"label_{name}", 0.0) + getattr(stats, name)
        logs = {
            "bce": parts.bce,
            "dice": parts.dice,
            "l1": parts.l1,
            "n_pos": parts.n_pos,
            "n_neg": parts.n_neg,
            "empty": empty,
        }
        return StepResult(parts.total, len(batch["ids"]), list(batch["ids"]), logs)

    def validate(self) -> EpochResult:
        res = evaluate_detector(
            self.model,
            self.eval_loader,
            self.gt_pages,
            self.det,
            self.device,
            self.loss_fn,
            keep="full",
            pool=self.pool,
        )
        val, tr = res["val"], res.get("train_eval")
        self._last_val = val
        point = self._epoch_point(val, tr)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            reset_peak(self.device)
        return EpochResult(point, val.f1_50, self._epoch_flags(val, point))

    def _epoch_point(self, val: DetEvalResult, tr: DetEvalResult | None) -> dict[str, Any]:
        st, x = self.state, self.state.extra
        p: dict[str, Any] = {
            "epoch": st.epoch,
            "step": st.step,
            "val_f1_50": val.f1_50,
            "val_op": val.selected_op,
        }
        for name, m in val.ops.items():
            p.update(
                {
                    f"val_{name}_p50": m.precision,
                    f"val_{name}_r50": m.recall,
                    f"val_{name}_f1_50": m.f1_50,
                    f"val_{name}_f1_70": m.f1_70,
                    f"val_{name}_f1_30": m.f1_30,
                    f"val_{name}_miou": m.mean_iou,
                    f"val_{name}_counters": m.counters.to_dict(),
                }
            )
        loss = val.loss or {}
        p.update(
            val_loss=loss.get("total"),
            val_loss_bce=loss.get("bce"),
            val_loss_dice=loss.get("dice"),
            val_loss_l1=loss.get("l1"),
            val_l1_const=loss.get("l1_const"),
        )
        p["train_f1_50"] = tr.f1_50 if tr is not None else None
        p["train_eval_loss"] = tr.loss["total"] if tr is not None and tr.loss else None
        m_ = val.maps
        if m_ is not None:
            p["pred_kernel_ratio"] = m_.pred_text_px / m_.gt_kernel_px if m_.gt_kernel_px else None
            p["pred_text_share"] = m_.pred_text_px / m_.valid_px if m_.valid_px else None
            p["empty_page_share"] = (
                m_.pages_with_care_no_box / m_.pages_with_care if m_.pages_with_care else None
            )
            p["thresh_out_of_range"] = (
                m_.thresh_out_of_range_px / m_.thresh_px if m_.thresh_px else None
            )
        p["kernel_excess_px"], p["kernel_excess_ratio"] = (
            val.kernel_excess_px,
            val.kernel_excess_ratio,
        )
        p["val_boxes_per_page"] = sum(r.n_pred[val.selected_op] for r in val.records) / max(
            val.pages, 1
        )
        p["empty_crop_share"] = x.get("empty", 0.0) / max(x.get(SAMPLES, 0.0), 1.0)
        p["data_wait_share"] = x.get(WAIT_S, 0.0) / max(x.get(EPOCH_S, 0.0), 1e-9)
        for key, value in val.seconds.items():
            p[f"val_seconds_{key}"] = value
        lines = sum(
            x.get(f"label_{k}", 0.0) for k in ("care_drawn", "demoted_small", "demoted_thin_kernel")
        )
        for k in ("demoted_small", "demoted_thin_kernel", "multipart_kernel"):
            p[f"label_{k}_share"] = x.get(f"label_{k}", 0.0) / lines if lines else None
        dont = x.get("label_dont_care", 0.0)
        p["label_dont_care_share"] = dont / (lines + dont) if lines + dont else None
        if self.device.type == "cuda":
            p["vram_peak_alloc_mib"] = torch.cuda.max_memory_allocated(self.device) / 2**20
            p["vram_peak_reserved_mib"] = torch.cuda.max_memory_reserved(self.device) / 2**20
        return p

    def _epoch_flags(self, val: DetEvalResult, point: dict[str, Any]) -> list[Flag]:
        st, dd = self.state, self.det.diagnostics
        epoch, step = st.epoch, st.step
        budget = step / max(self.total_steps, 1)
        hist = [*st.history, point]  # the core appends this point once validate returns
        flags: list[Flag] = []
        keys = ("train_eval_loss", "val_loss", "train_f1_50", "val_f1_50")
        if all(h.get(k) is not None for h in hist for k in keys):
            pts = [
                EpochPoint(
                    h["epoch"],
                    h["step"],
                    h["train_eval_loss"],
                    h["val_loss"],
                    1 - h["train_f1_50"],
                    1 - h["val_f1_50"],
                )
                for h in hist
            ]
            flags += check_det_fitting(pts, self.cfg.diagnostics, dd, budget)
        if val.maps is not None:
            flags += check_map_collapse(val.maps, dd, epoch, step)
            flags += check_threshold(hist, val.maps, dd, budget, epoch, step)
        for f in (check_box_size(val, dd, epoch, step), check_postprocess(val, dd, epoch, step)):
            if f is not None:
                flags.append(f)
        flags += check_epoch_data(
            point["empty_crop_share"], point["data_wait_share"], dd, epoch, step
        )
        slow = check_slow_validation(
            {**val.seconds, "boxes_per_page": point["val_boxes_per_page"]},
            st.extra.get(EPOCH_S, 0.0),
            dd,
            epoch,
            step,
        )
        if slow is not None:
            flags.append(slow)
        # Each detection flag is raised once per run, with the evidence of its first occurrence.
        raised = {f["name"] for f in st.flags}
        out = []
        for f in flags:
            if f.name not in raised:
                raised.add(f.name)
                out.append(f)
        return out

    def _spec(self, op: str | None) -> DetectorSpec:
        det = self.det
        return DetectorSpec(
            det.model,
            det.labels,
            det.infer,
            det.post,
            op or det.post.operating_points[0].name,
            LABEL_VERSION,
        )

    def payload_extra(self) -> dict[str, Any]:
        # The latest validated epoch's winner: for best.pt the best epoch's, the operating point
        # of record. None before the first validation.
        op = self.state.history[-1]["val_op"] if self.state.history else None
        return {
            "task": "det",
            "selected_op": op,
            "detector_spec": self._spec(op).to_dict(),
            "label_version": LABEL_VERSION,
            "det_data_version": DET_DATA_VERSION,
        }

    def on_best(self) -> None:
        res = self._last_val
        if res is None:
            return
        write_records(self.run_dir / "val_best_pages.jsonl", [r.to_dict() for r in res.records])
        write_records(self.run_dir / "val_best_candidates.jsonl.gz", res.candidates or [])

    # ------------------------------------------------------------------- full validation

    def evaluate_full(self, checkpoint: Path, tag: Literal["best", "last"]) -> FullEval:
        """Score a checkpoint on every validation page at its stored operating point.

        The full-set loader and the post-processing pool are built here and closed before
        returning; the trainer's own model is left untouched.
        """
        ck = load_checkpoint(checkpoint, "cpu")
        model = build_detector(self.det.model, self.data_root, load_pretrained=False)
        model.load_state_dict(ck["model"])
        model = model.to(self.device)
        names = [op.name for op in self.det.post.operating_points]
        stored = ck.get("selected_op") or names[0]
        if self._full_gt is None:
            self._full_gt = read_gt_pages(self.val_file, self.val_full_index.doc_ids())
        ds = DetEvalDataset(
            [("val", self.val_full_index)], self.data_root, self.det, with_labels=False
        )
        loader = self._eval_loader(ds, persistent=False)
        try:
            with ThreadPoolExecutor(max_workers=POST_WORKERS) as pool:
                res = evaluate_detector(
                    model,
                    loader,
                    self._full_gt,
                    self.det,
                    self.device,
                    keep="full" if tag == "best" else "records",
                    pool=pool,
                )["val"]
        finally:
            _shutdown(loader)
            del loader, model
        return FullEval(tag, sha256_file(checkpoint), stored, res.selected_op, res)

    # ----------------------------------------------------------------------------- summary

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        det = self.det
        out.update(
            {
                "parameters": self.params,
                "train_pages": len(self.train_index),
                "train_crops_per_epoch": len(self.train_ds),
                "steps_per_epoch": self.steps_per_epoch[0],
                "val_pages_per_epoch": len(self.val_index),
                "train_eval_pages": len(self.train_eval_index) if self.train_eval_index else 0,
                "val_pages_full": len(self.val_full_index),
                "data": {
                    "train_pages": det.data.train_pages[0],
                    "val_pages": det.data.val_pages[0],
                    "train_pages_sha256": self.train_index.sha256,
                    "val_pages_sha256": self.val_index.sha256,
                    "train_ids_sha256": self.train_index.ids_sha256(),
                    "val_subset_ids_sha256": self.val_index.ids_sha256(),
                    "train_eval_ids_sha256": (
                        self.train_eval_index.ids_sha256() if self.train_eval_index else None
                    ),
                },
                "label_version": LABEL_VERSION,
                "det_data_version": DET_DATA_VERSION,
                "backbone": det.model.backbone,
                "weights_sha256": det.model.weights_sha256 or "scratch",
                "box_type": det.post.box_type,
                "operating_points": [asdict(op) for op in det.post.operating_points],
                "precision_eval": "fp32 (ieee)",
                "versions": {
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "opencv": cv2.__version__,
                    "shapely": shapely.__version__,
                    "geos": shapely.geos_version_string,
                },
                "label_ceiling_tripwire": self.tripwire,
            }
        )
        return out

    def close(self) -> None:
        """Shut both loaders' persistent workers and the post-processing pool. Idempotent."""
        if self._closed:
            return
        _shutdown(getattr(self, "train_loader", None))
        _shutdown(getattr(self, "eval_loader", None))
        pool = getattr(self, "pool", None)
        if pool is not None:
            pool.shutdown(wait=True)
        super().close()


# --------------------------------------------------------------------------- run and finish


def full_eval_json(fe: FullEval, baseline_ids: list[str]) -> dict[str, Any]:
    """eval_full_{best,last}.json: every operating point, the headline at the stored one, and the
    Phase 5 baseline page row (detection only)."""
    res = fe.result
    ops = {n: op_metrics_json(res.records, n, m.counters) for n, m in res.ops.items()}
    wanted = set(baseline_ids)
    base = [r for r in res.records if r.doc_id in wanted]
    return {
        "tag": fe.tag,
        "checkpoint_sha256": fe.checkpoint_sha256,
        "stored_op": fe.stored_op,
        "full_set_max_op": fe.full_set_max_op,
        "pages": res.pages,
        "care_lines": res.care_lines,
        "headline": {"op": fe.stored_op, **ops[fe.stored_op]},
        "ops": ops,
        "baseline_pages": {"op": fe.stored_op, **op_metrics_json(base, fe.stored_op)},
        "seconds": res.seconds,
    }


def _write_full_eval(trainer: DetTrainer, fe: FullEval) -> None:
    run = trainer.run_dir
    baseline = baseline_doc_ids(trainer.val_full_index.doc_ids())
    (run / f"eval_full_{fe.tag}.json").write_text(
        json.dumps(full_eval_json(fe, baseline), indent=1) + "\n", encoding="utf-8"
    )
    if fe.tag == "best":
        res = fe.result
        write_records(run / "val_full_best_pages.jsonl", [r.to_dict() for r in res.records])
        write_records(run / "val_full_best_candidates.jsonl.gz", res.candidates or [])
        write_records(run / "val_full_best_boxes.jsonl.gz", boxes_rows(res.boxes or {}))


STATUS: Final = {
    0: "done",
    2: "failed: non-finite loss or gradient (exit 2)",
    3: "failed: out of GPU memory (exit 3)",
}
# Everything `_write_full_eval` writes into the run directory.
FULL_EVAL_FILES: Final = (
    "eval_full_best.json",
    "eval_full_last.json",
    "val_full_best_pages.jsonl",
    "val_full_best_candidates.jsonl.gz",
    "val_full_best_boxes.jsonl.gz",
)


def finish_det(trainer: DetTrainer, exit_code: int = 0) -> Path:
    """Full validation (successful runs only), summary.json and the run report, on every path.

    The per-epoch loaders and the pool are shut first, so the full-set loader is the only one
    alive. Full-validation files left by an earlier attempt of this run ID (a session cut by
    ``train.max_steps``, a failed start) are deleted first: they describe a checkpoint this finish
    either scores again or does not vouch for. Each checkpoint is evaluated only if it exists; an
    out-of-memory error or any other failure there is recorded and the report is still written.
    """
    trainer.close()
    for name in FULL_EVAL_FILES:
        (trainer.run_dir / name).unlink(missing_ok=True)
    outcome: dict[str, str] = {}
    for tag in ("best", "last"):
        path = trainer.ckpt_dir / f"{tag}.pt"
        if exit_code:
            outcome[tag] = f"skipped: run failed (exit {exit_code})"
        elif not path.exists():
            outcome[tag] = "missing"
        else:
            try:
                _write_full_eval(trainer, trainer.evaluate_full(path, tag))
                outcome[tag] = "done"
            except torch.cuda.OutOfMemoryError:
                free_cuda()
                log.error(
                    "out of GPU memory scoring %s; score it later with scripts/det_eval.py", path
                )
                outcome[tag] = "skipped: out of GPU memory"
            except Exception as exc:  # the report must still be written
                log.exception("full validation of %s failed", path)
                outcome[tag] = f"failed: {type(exc).__name__}: {exc}"
    summary = trainer.summary()
    summary.update(
        exit_code=exit_code,
        status=STATUS.get(exit_code, f"failed (exit {exit_code})"),
        full_eval=outcome,
    )
    trainer.run_dir.mkdir(parents=True, exist_ok=True)
    (trainer.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    return write_det_report(trainer.run_dir, trainer.report_dir, trainer.cfg)


def run_det(cfg: RunConfig) -> int:
    """Train, evaluate and report one detector config. Returns the exit code (module docstring)."""
    try:
        trainer = DetTrainer(cfg)
    except GpuBusyError as exc:
        log.error("refused: %s", exc)
        return 4
    log.info(
        "%s: %s parameters (%s at inference), %d training pages, %d steps per epoch, %d "
        "validation pages per epoch on %s",
        cfg.experiment_id,
        f"{trainer.params['train']:,}",
        f"{trainer.params['inference']:,}",
        len(trainer.train_index),
        trainer.steps_per_epoch[0],
        len(trainer.val_index),
        trainer.device,
    )
    code = 0
    try:
        try:
            trainer.fit()
        except NonFiniteLoss as exc:
            log.error("%s", exc)
            code = 2
        except torch.cuda.OutOfMemoryError:
            log.error("out of GPU memory at step %d", trainer.state.step)
            code = 3
            try:
                trainer.save("oom.pt")
            except Exception:  # a failed save must not stop the report
                log.exception("could not save oom.pt")
        if code:
            # Free the GPU before the report path starts.
            trainer.close()
            free_cuda()
        try:
            log.info("run report: %s", finish_det(trainer, code))
        except Exception:  # the run's files are on disk; the report can be regenerated
            log.exception("could not write the run report; rerun scripts/det_report.py")
    finally:
        trainer.close()
    return code
