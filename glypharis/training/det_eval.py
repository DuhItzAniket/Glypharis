"""Detector validation: one forward pass per page, every operating point scored from it (spec §7.3).

`evaluate_detector` runs the network on one page at a time in fp32 (IEEE, no TF32, autocast off), so
a page's map never depends on other pages and the selected F1 is the number the CPU, and Phase 17's
export, reproduce. Post-processing and counting run on a small thread pool while the main thread
computes the next page's map; results are stored by page index, so they do not depend on the pool
size or on scheduling.

Per page: `find_candidates` once, then `boxes_from_candidates` and `fast_det_counts` at IoU 0.5, 0.7
and 0.3 for every operating point. Per role ("val", "train_eval") the result holds micro P/R/F1 per
operating point and the arg-max of F1@0.5 over them (ties: the first listed). When the loader
carries labels it also holds the per-page validation loss, the map statistics behind the collapse
and threshold flags, and the kernel-excess diagnostic: how much wider the predicted kernels are
than the label kernels, which tells which unclip mode the model is calibrated for.

`label_ceiling_page` is the model-free counterpart, used by sanity check 0 and the trainer's
start-up tripwire: a page's own label kernels serve as the probability map and go through the same
post-processing and counting.
"""

from __future__ import annotations

import dataclasses
import gzip
import json
import math
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Executor, Future
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import shapely
import torch

from glypharis.config.schema import DetConfig, DetLabelConfig
from glypharis.data.det_dataset import MAP_KEYS
from glypharis.data.det_geometry import Affine, min_area_rect
from glypharis.data.det_labels import (
    CARE,
    MIN_KERNEL_THICKNESS,
    line_reason,
    make_labels,
    shrink_distance,
)
from glypharis.data.pages import Page
from glypharis.eval.det_matching import det_cis, fast_det_counts
from glypharis.eval.page_metrics import PageResult, _poly, det_summary, evaluate_page, is_box_line
from glypharis.eval.stats import Interval
from glypharis.models.det.dbnet import DBNet
from glypharis.models.det.loss import DBLoss
from glypharis.models.det.postprocess import (
    Candidate,
    DetBox,
    PostCounters,
    boxes_from_candidates,
    find_candidates,
)
from glypharis.models.det.preprocess import PAD_MULTIPLE, ResizeInfo, scaled_size
from glypharis.models.det.runtime import strict_fp32

EVAL_THRESHOLDS: Final = (0.5, 0.7, 0.3)  # F1@0.3 is a diagnostic only, never a result
CEILING_THRESHOLDS: Final = (0.5, 0.7)
CEILING_VARIANTS: Final = ("strict", "representational", "poly")
MAX_IN_FLIGHT: Final = 4  # pages handed to the post-processing pool and not yet collected
T_RANGE: Final = (0.2, 0.8)  # threshold values outside this count for THRESHOLD OUT OF RANGE
SECONDS_KEYS: Final = ("data_wait", "forward", "post", "match", "total")

Threshold = float
Counts = tuple[int, int, int]  # tp, fp, fn


@dataclass
class MapStats:
    pred_text_px: int = 0  # P > bin_thresh inside the valid region
    gt_kernel_px: int = 0
    valid_px: int = 0
    thresh_px: int = 0
    thresh_out_of_range_px: int = 0  # threshold-mask pixels with T outside T_RANGE
    pages_with_care: int = 0
    pages_with_care_no_box: int = 0  # at the pass's selected operating point


@dataclass
class OpMetrics:
    precision: float
    recall: float
    f1_50: float
    f1_70: float
    f1_30: float
    mean_iou: float  # over true positives at IoU 0.5
    counters: PostCounters


@dataclass
class DetPageRecord:
    doc_id: str
    role: str
    n_care: int
    det: dict[str, dict[Threshold, Counts]]  # op -> thr -> (tp, fp, fn)
    iou_sum: dict[str, float]
    n_pred: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """JSON form; thresholds become strings ("0.5")."""
        return {
            "doc_id": self.doc_id,
            "role": self.role,
            "n_care": self.n_care,
            "det": {op: {str(t): list(c) for t, c in d.items()} for op, d in self.det.items()},
            "iou_sum": self.iou_sum,
            "n_pred": self.n_pred,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DetPageRecord:
        det = {
            op: {float(t): (int(c[0]), int(c[1]), int(c[2])) for t, c in by_thr.items()}
            for op, by_thr in d["det"].items()
        }
        return cls(
            str(d["doc_id"]),
            str(d["role"]),
            int(d["n_care"]),
            det,
            {k: float(v) for k, v in d["iou_sum"].items()},
            {k: int(v) for k, v in d["n_pred"].items()},
        )


@dataclass
class DetEvalResult:
    role: str
    pages: int
    care_lines: int
    ops: dict[str, OpMetrics]
    selected_op: str  # arg-max over ops of F1@0.5 on this pass (ties: first listed)
    f1_50: float
    loss: dict[str, float] | None  # per-page means: total, bce, dice, l1, l1_const
    maps: MapStats | None
    kernel_excess_px: float | None
    kernel_excess_ratio: float | None
    records: list[DetPageRecord]
    candidates: list[dict[str, Any]] | None  # keep == "full"
    boxes: dict[str, dict[str, list[DetBox]]] | None  # keep == "full": op -> doc_id -> boxes
    seconds: dict[str, float]  # of the whole pass: data_wait, forward, post, match, total


@dataclass
class PageOutcome:
    """One page after post-processing and counting (what a pool job returns)."""

    record: DetPageRecord
    info: ResizeInfo
    counters: dict[str, PostCounters]
    excess: list[tuple[float, float]]  # (kernel excess px, kernel excess ratio) per candidate
    candidates: list[Candidate]  # kept only when asked for
    boxes: dict[str, list[DetBox]]
    post_s: float
    match_s: float


# ------------------------------------------------------------------------------ record files


def write_records(path: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """JSON lines, gzipped when the name ends in ``.gz``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8", newline="\n")
    return path


def read_records(path: Path) -> list[dict[str, Any]]:
    """The rows of a `write_records` file; an empty list when it does not exist."""
    if not path.exists():
        return []
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = path.read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def boxes_rows(boxes: Mapping[str, Mapping[str, list[DetBox]]]) -> list[dict[str, Any]]:
    """Per page: doc_id and op -> list of box dicts, in page order of the first operating point."""
    ops = list(boxes)
    doc_ids = list(boxes[ops[0]]) if ops else []
    return [
        {"doc_id": d, "boxes": {op: [asdict(b) for b in boxes[op][d]] for op in ops}}
        for d in doc_ids
    ]


def box_from_dict(d: Mapping[str, Any]) -> DetBox:
    return DetBox(
        polygon=[[float(x), float(y)] for x, y in d["polygon"]],
        quad=[[float(x), float(y)] for x, y in d["quad"]],
        bbox=(
            float(d["bbox"][0]),
            float(d["bbox"][1]),
            float(d["bbox"][2]),
            float(d["bbox"][3]),
        ),
        score=float(d["score"]),
        aspect=float(d["aspect"]),
        long_axis_angle=float(d["long_axis_angle"]),
    )


def interval_dict(iv: Interval) -> dict[str, float]:
    return {"value": iv.value, "low": iv.low, "high": iv.high}


def op_metrics_json(
    records: Sequence[DetPageRecord], op: str, counters: PostCounters | None = None
) -> dict[str, Any]:
    """P/R/F1@0.5 and F1@0.7 at one operating point with 95% page-bootstrap CIs, as JSON."""
    a50 = np.array([r.det[op][0.5] for r in records], dtype=np.float64).reshape(-1, 3)
    a70 = np.array([r.det[op][0.7] for r in records], dtype=np.float64).reshape(-1, 3)
    c50 = det_cis(a50[:, 0], a50[:, 1], a50[:, 2])
    tp = float(a50[:, 0].sum())
    out: dict[str, Any] = {
        "pages": len(records),
        "precision": interval_dict(c50["precision"]),
        "recall": interval_dict(c50["recall"]),
        "f1_50": interval_dict(c50["f1"]),
        "f1_70": interval_dict(det_cis(a70[:, 0], a70[:, 1], a70[:, 2])["f1"]),
        "mean_iou": sum(r.iou_sum[op] for r in records) / tp if tp else 0.0,
        "tp_fp_fn_50": [int(v) for v in a50.sum(axis=0)],
    }
    if counters is not None:
        out["counters"] = counters.to_dict()
    return out


# -------------------------------------------------------------------------------- one page


def kernel_excess(
    cands: Sequence[Candidate], page: Page, info: ResizeInfo, labels: DetLabelConfig
) -> list[tuple[float, float]]:
    """Width of predicted kernels against the label kernel of the care line they sit in.

    Only candidates whose area-rectangle centre lies inside exactly one care line, a line that gets
    a kernel at this scale, count: excess = width − (short − 2·D) and ratio = width / (short − 2·D),
    with D the training shrink distance of that line at network scale.
    """
    care = [ln for ln in page.lines if is_box_line(ln)]
    if not cands or not care:
        return []
    polys, kernel_w = [], []
    for ln in care:
        pts = np.asarray(ln.polygon, dtype=np.float64).reshape(-1, 2) * (info.sx, info.sy)
        poly = _poly(pts.tolist())
        short = min_area_rect(pts)[1]
        w = short - 2.0 * shrink_distance(poly.area, poly.length, labels.shrink_ratio)
        thick = short >= labels.min_text_size and w >= MIN_KERNEL_THICKNESS
        polys.append(poly)
        kernel_w.append(w if thick else math.nan)
    centres = shapely.points([c.area_rect[0] for c in cands], [c.area_rect[1] for c in cands])
    pairs = shapely.STRtree(polys).query(centres, predicate="within")
    hits = np.bincount(pairs[0], minlength=len(cands))
    out = []
    for ci, pi in zip(pairs[0].tolist(), pairs[1].tolist(), strict=True):
        if hits[ci] == 1 and math.isfinite(kernel_w[pi]):
            width = cands[ci].area_rect[3]
            out.append((width - kernel_w[pi], width / kernel_w[pi]))
    return out


def score_page(
    prob: np.ndarray, info: ResizeInfo, gt: Page, role: str, det: DetConfig, keep: bool
) -> PageOutcome:
    """Candidates once, then boxes and detection counts for every operating point."""
    t0 = time.perf_counter()
    post, r = det.post, det.labels.shrink_ratio
    cands, base = find_candidates(prob, info, post)
    boxes: dict[str, list[DetBox]] = {}
    counters: dict[str, PostCounters] = {}
    for op in post.operating_points:
        found, c = boxes_from_candidates(cands, op, post, r, info)
        c.add(base)
        boxes[op.name], counters[op.name] = found, c
    excess = kernel_excess(cands, gt, info, det.labels)
    t1 = time.perf_counter()
    counts: dict[str, dict[Threshold, Counts]] = {}
    iou_sum: dict[str, float] = {}
    for name, found in boxes.items():
        res = fast_det_counts(gt, [b.to_pred_line() for b in found], EVAL_THRESHOLDS)
        counts[name], iou_sum[name] = dict(res.det), res.matched_iou_sum
    n_care = sum(1 for ln in gt.lines if is_box_line(ln))
    rec = DetPageRecord(
        gt.doc_id, role, n_care, counts, iou_sum, {k: len(v) for k, v in boxes.items()}
    )
    return PageOutcome(
        rec,
        info,
        counters,
        excess,
        cands if keep else [],
        boxes if keep else {},
        t1 - t0,
        time.perf_counter() - t1,
    )


@dataclass
class _RoleSums:
    """Per-role sums of what the forward pass measures on labelled pages."""

    labelled: int = 0
    loss: dict[str, float] = field(default_factory=dict)
    maps: MapStats = field(default_factory=MapStats)


def _forward_page(
    model: DBNet,
    batch: Mapping[str, Any],
    device: torch.device,
    det: DetConfig,
    loss_fn: DBLoss | None,
    sums: _RoleSums,
) -> np.ndarray:
    """The page's (pad_h, pad_w) probability map; label statistics are added to ``sums``."""
    images = batch["images"].to(device).permute(0, 3, 1, 2)
    if "shrink_map" not in batch:
        return np.ascontiguousarray(model(images)[0, 0].float().cpu().numpy())
    info: ResizeInfo = batch["info"]
    shrink, smask, tmap, tmask = (batch[k].to(device).float().unsqueeze(1) for k in MAP_KEYS)
    logits = model.forward_train(images)
    # sigmoid(forward_train().prob) equals forward() bit for bit, so this is the inference map.
    prob = torch.sigmoid(logits.prob)
    thresh = torch.sigmoid(logits.thresh)
    lo, hi = T_RANGE
    values = [
        (prob[..., : info.h, : info.w] > det.post.bin_thresh).sum(),
        shrink.sum(),
        tmask.sum(),
        (((thresh < lo) | (thresh > hi)).float() * tmask).sum(),
        ((0.5 - tmap).abs() * tmask).sum() / (tmask.sum() + det.loss.eps),
    ]
    if loss_fn is not None:
        parts = loss_fn(logits, shrink, smask, tmap, tmask)
        values += [parts.total, parts.bce, parts.dice, parts.l1]
    host = torch.stack([v.double() for v in values]).cpu().tolist()  # one sync per page
    m = sums.maps
    m.pred_text_px += int(host[0])
    m.gt_kernel_px += int(host[1])
    m.valid_px += info.h * info.w
    m.thresh_px += int(host[2])
    m.thresh_out_of_range_px += int(host[3])
    sums.labelled += 1
    names = ["l1_const"] + (["total", "bce", "dice", "l1"] if loss_fn is not None else [])
    for name, v in zip(names, host[4:], strict=True):
        sums.loss[name] = sums.loss.get(name, 0.0) + v
    return np.ascontiguousarray(prob[0, 0].float().cpu().numpy())


def _summarise(
    role: str,
    outs: Sequence[PageOutcome],
    sums: _RoleSums,
    det: DetConfig,
    keep: bool,
    with_loss: bool,
    seconds: dict[str, float],
) -> DetEvalResult:
    names = [op.name for op in det.post.operating_points]
    records = [o.record for o in outs]
    ops: dict[str, OpMetrics] = {}
    for name in names:
        results = [PageResult(r.doc_id, det=r.det[name]) for r in records]
        s50, s70, s30 = (det_summary(results, t) for t in EVAL_THRESHOLDS)
        tp = sum(r.det[name][0.5][0] for r in records)
        counters = PostCounters()
        for o in outs:
            counters.add(o.counters[name])
        iou = sum(r.iou_sum[name] for r in records)
        ops[name] = OpMetrics(
            s50.precision, s50.recall, s50.f1, s70.f1, s30.f1, iou / tp if tp else 0.0, counters
        )
    selected = max(names, key=lambda n: ops[n].f1_50)  # max keeps the first of equal values
    maps = loss = None
    if sums.labelled:
        maps = dataclasses.replace(sums.maps)
        maps.pages_with_care = sum(1 for r in records if r.n_care)
        maps.pages_with_care_no_box = sum(1 for r in records if r.n_care and not r.n_pred[selected])
        if with_loss:
            loss = {k: v / sums.labelled for k, v in sums.loss.items()}
    excess = [e for o in outs for e in o.excess]
    return DetEvalResult(
        role=role,
        pages=len(records),
        care_lines=sum(r.n_care for r in records),
        ops=ops,
        selected_op=selected,
        f1_50=ops[selected].f1_50,
        loss=loss,
        maps=maps,
        kernel_excess_px=float(np.median([e[0] for e in excess])) if excess else None,
        kernel_excess_ratio=float(np.median([e[1] for e in excess])) if excess else None,
        records=records,
        candidates=[
            {
                "doc_id": o.record.doc_id,
                "role": role,
                "info": asdict(o.info),
                "candidates": [c.to_dict() for c in o.candidates],
            }
            for o in outs
        ]
        if keep
        else None,
        boxes={n: {o.record.doc_id: o.boxes[n] for o in outs} for n in names} if keep else None,
        seconds=dict(seconds),
    )


def summarise_pages(
    role: str, outcomes: Sequence[PageOutcome], det: DetConfig, keep: bool = False
) -> DetEvalResult:
    """The result of pages scored outside `evaluate_detector` (cached probability maps): metrics
    and selection only, no loss or map statistics."""
    seconds = dict.fromkeys(SECONDS_KEYS, 0.0)
    seconds["post"] = sum(o.post_s for o in outcomes)
    seconds["match"] = sum(o.match_s for o in outcomes)
    return _summarise(role, outcomes, _RoleSums(), det, keep, False, seconds)


@torch.no_grad()
def evaluate_detector(
    model: DBNet,
    loader: Iterable[dict[str, Any]],
    gt: Mapping[str, Page],
    det: DetConfig,
    device: torch.device,
    loss_fn: DBLoss | None = None,
    keep: Literal["records", "full"] = "records",
    pool: Executor | None = None,
) -> dict[str, DetEvalResult]:
    """Score every page of ``loader`` (batches of one page from `collate_eval`), by role.

    The model runs in eval mode and is put back in its previous mode afterwards. ``gt`` maps each
    doc_id to its full `Page`. ``keep="full"`` also returns every page's candidates and boxes.
    """
    was_training = model.training
    model.eval()
    full = keep == "full"
    outcomes: list[PageOutcome | None] = []
    in_flight: deque[tuple[int, Future[PageOutcome]]] = deque()
    sums: dict[str, _RoleSums] = {}
    with_loss = loss_fn is not None
    seconds = dict.fromkeys(SECONDS_KEYS, 0.0)
    start = time.perf_counter()
    try:
        it = iter(loader)
        with strict_fp32(), torch.autocast(device_type=device.type, enabled=False):
            while True:
                t0 = time.perf_counter()
                try:
                    batch = next(it)
                except StopIteration:
                    break
                t1 = time.perf_counter()
                seconds["data_wait"] += t1 - t0
                role = str(batch["role"])
                prob = _forward_page(
                    model, batch, device, det, loss_fn, sums.setdefault(role, _RoleSums())
                )
                seconds["forward"] += time.perf_counter() - t1
                args = (prob, batch["info"], gt[batch["doc_id"]], role, det, full)
                if pool is None:
                    outcomes.append(score_page(*args))
                    continue
                outcomes.append(None)
                in_flight.append((len(outcomes) - 1, pool.submit(score_page, *args)))
                while len(in_flight) >= MAX_IN_FLIGHT:
                    i, fut = in_flight.popleft()
                    outcomes[i] = fut.result()
            while in_flight:
                i, fut = in_flight.popleft()
                outcomes[i] = fut.result()
    finally:
        for _, fut in in_flight:
            fut.cancel()
        model.train(was_training)
    done = [o for o in outcomes if o is not None]
    seconds["post"] = sum(o.post_s for o in done)
    seconds["match"] = sum(o.match_s for o in done)
    seconds["total"] = time.perf_counter() - start
    roles = list(dict.fromkeys(o.record.role for o in done))
    return {
        role: _summarise(
            role,
            [o for o in done if o.record.role == role],
            sums[role],
            det,
            full,
            with_loss,
            seconds,
        )
        for role in roles
    }


# ------------------------------------------------------------------------------ label ceiling


@dataclass
class CeilingPage:
    """One page's label-ceiling counts: its own label kernels used as the probability map."""

    doc_id: str
    n_care: int
    demoted: int  # care lines with no kernel at this scale (too small, or too thin)
    det: dict[str, dict[Threshold, Counts]]  # "variant/op" -> thr -> (tp, fp, fn)
    mismatches: int  # fast_det_counts against evaluate_page, over every variant, op and thr


def ceiling_geometry(page: Page, long_side: int) -> ResizeInfo:
    """The `prepare_page` geometry of a page, from its size alone (no decode needed)."""
    w, h = scaled_size(page.width, page.height, long_side)
    pad = PAD_MULTIPLE
    return ResizeInfo(page.width, page.height, w, h, -(-w // pad) * pad, -(-h // pad) * pad)


def label_ceiling_page(
    page: Page,
    det: DetConfig,
    variants: Sequence[str] = CEILING_VARIANTS,
    check_parity: bool = True,
) -> CeilingPage:
    """Label-ceiling counts of one page at the inference scale (sanity check 0).

    Variants: ``strict`` (the configured labels and post-processing), ``representational``
    (``min_text_size`` 0, so no line is demoted for being small) and ``poly`` (strict labels with
    ``box_type: poly``, for information). With ``check_parity`` every count is also computed with
    `evaluate_page` and differences are counted.
    """
    info = ceiling_geometry(page, det.infer.long_side)
    g = Affine.scale(info.sx, info.sy)
    polys = [g.apply(np.asarray(ln.polygon, dtype=np.float64).reshape(-1, 2)) for ln in page.lines]
    care = np.array([line_reason(ln) == CARE for ln in page.lines], dtype=bool)
    ignore = [g.apply(np.asarray(q, dtype=np.float64).reshape(-1, 2)) for q in page.ignore]
    canvas = (info.pad_h, info.pad_w)
    strict = make_labels(polys, care, ignore, canvas, det.labels)
    out: dict[str, dict[Threshold, Counts]] = {}
    mismatches = 0
    for variant in variants:
        labels_cfg, labels, post = det.labels, strict, det.post
        if variant == "representational":
            labels_cfg = dataclasses.replace(det.labels, min_text_size=0.0)
            labels = make_labels(polys, care, ignore, canvas, labels_cfg)
        elif variant == "poly":
            post = dataclasses.replace(det.post, box_type="poly")
        elif variant != "strict":
            raise ValueError(f"unknown label-ceiling variant {variant!r}")
        cands, _ = find_candidates(labels.shrink_map.astype(np.float32), info, post)
        for op in post.operating_points:
            boxes, _ = boxes_from_candidates(cands, op, post, labels_cfg.shrink_ratio, info)
            preds = [b.to_pred_line() for b in boxes]
            fast = fast_det_counts(page, preds, CEILING_THRESHOLDS)
            if check_parity:
                ref = evaluate_page(page, preds, CEILING_THRESHOLDS, score_text=False)
                mismatches += sum(fast.det[t] != ref.det[t] for t in CEILING_THRESHOLDS)
            out[f"{variant}/{op.name}"] = dict(fast.det)
    n_care = int(care.sum())
    return CeilingPage(page.doc_id, n_care, n_care - int(strict.care_kept.sum()), out, mismatches)


def summarise_ceiling(pages: Sequence[CeilingPage], op_names: Sequence[str]) -> dict[str, Any]:
    """Micro P/R/F1 with page-bootstrap CIs per variant and operating point, the selection ceiling
    (max over operating points of F1@0.5) per variant, the demoted share and the parity count."""
    keys = list(pages[0].det) if pages else []
    per_key: dict[str, Any] = {}
    for key in keys:
        arr = {
            t: np.array([p.det[key][t] for p in pages], dtype=np.float64)
            for t in CEILING_THRESHOLDS
        }
        c50 = det_cis(arr[0.5][:, 0], arr[0.5][:, 1], arr[0.5][:, 2])
        c70 = det_cis(arr[0.7][:, 0], arr[0.7][:, 1], arr[0.7][:, 2])
        per_key[key] = {
            "precision": interval_dict(c50["precision"]),
            "recall": interval_dict(c50["recall"]),
            "f1_50": interval_dict(c50["f1"]),
            "f1_70": interval_dict(c70["f1"]),
        }
    selection = {}
    for variant in dict.fromkeys(k.split("/")[0] for k in keys):
        best = max(op_names, key=lambda n: per_key[f"{variant}/{n}"]["f1_50"]["value"])
        selection[variant] = {"op": best, "f1_50": per_key[f"{variant}/{best}"]["f1_50"]["value"]}
    care = sum(p.n_care for p in pages)
    return {
        "pages": len(pages),
        "care_lines": care,
        "demoted_share": sum(p.demoted for p in pages) / care if care else 0.0,
        "mismatches": sum(p.mismatches for p in pages),
        "selection": selection,
        "per_op": per_key,
    }
