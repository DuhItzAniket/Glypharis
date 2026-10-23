"""Score a detector checkpoint on the validation pages (Phases 8 and 9).

    python scripts/det_eval.py --config configs/det/gw-det-001.yaml \
        --checkpoint checkpoints/GW-DET-001/best.pt [--op inverse10] [--baseline-pages] \
        [--set det.post.box_thresh=0.5] [--out reports/eval/det/GW-DET-001_best.json]

Every operating point in ``det.post`` is scored from one forward pass per page (fp32, IEEE); the
headline is ``--op``, by default the checkpoint's stored ``selected_op`` (the operating point of
record). ``--set det.post.*`` changes post-processing without retraining. ``--baseline-pages``
scores only the Phase 5 validation page sample.

Cached maps, for Phase 9 sweeps on the CPU:

- ``--save-prob-maps DIR`` writes each page's valid-region probability map as an 8-bit PNG under
  ``DIR/<checkpoint-sha12>/`` and scores the maps as saved, so a later ``--from-prob-maps`` run
  with the same settings reproduces its numbers exactly;
- ``--from-prob-maps DIR`` scores those PNGs instead of running the network (no GPU, any
  ``bin_thresh``). Settings chosen this way are re-checked once on float maps (a run without
  either flag).

Test splits are never read here: ``--split test`` goes through `open_test_access`, which refuses in
Phase 8.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.data.det_dataset import DetEvalDataset, PageIndex, collate_eval, read_gt_pages
from glypharis.data.download import sha256_file
from glypharis.eval.access import open_test_access
from glypharis.eval.det_matching import baseline_doc_ids
from glypharis.models.det.dbnet import build_detector
from glypharis.models.det.runtime import TorchProbMap
from glypharis.training.checkpoint import load_checkpoint
from glypharis.training.det_eval import (
    ceiling_geometry,
    op_metrics_json,
    score_page,
    summarise_pages,
)
from glypharis.utils.device import select_device

PHASE = 8
POST_WORKERS = 2


def _quantise(prob: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """The valid region as 8-bit levels, and the padded float map those levels stand for."""
    levels = np.round(np.clip(prob[:h, :w], 0.0, 1.0) * 255.0).astype(np.uint8)
    out = np.zeros_like(prob, dtype=np.float32)
    out[:h, :w] = levels.astype(np.float32) / 255.0
    return levels, out


def _write_png(path: Path, levels: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", levels)
    if not ok:
        raise OSError(f"cannot encode {path}")
    buf.tofile(path)


def _read_png(path: Path) -> np.ndarray:
    levels = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if levels is None:
        raise OSError(f"cannot read {path}")
    return np.asarray(levels, dtype=np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--op", default=None, help="headline operating point (default: stored)")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--baseline-pages", action="store_true")
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--save-prob-maps", type=Path, default=None, metavar="DIR")
    maps.add_argument("--from-prob-maps", type=Path, default=None, metavar="DIR")
    parser.add_argument("--out", type=Path, default=None, help="also write the JSON here")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides_from_args(args))
    cfg.require_run_identity()
    if cfg.task != "det" or cfg.det is None:
        raise SystemExit(f"{args.config}: not a detector config")
    checkpoint = Path(args.checkpoint)
    # Held-out data only through the access gate, which refuses outside Phases 5, 14 and 22.
    open_test_access(
        "hiertext", args.split, cfg.experiment_id, str(checkpoint), PHASE, "detector evaluation"
    )
    det, root = cfg.det, Path(cfg.paths.data_root)
    pages_file = root / det.data.val_pages[0]
    index = PageIndex.build(pages_file, "validation", root)
    ids = index.doc_ids()
    if args.baseline_pages:
        keep = set(baseline_doc_ids(ids))
        index = index.subset(np.array([i for i, d in enumerate(ids) if d in keep]))
        ids = index.doc_ids()
    gt = read_gt_pages(pages_file, ids)

    ck = load_checkpoint(checkpoint, "cpu")
    sha = sha256_file(checkpoint)
    names = [op.name for op in det.post.operating_points]
    stored = ck.get("selected_op")
    op = args.op or (stored if stored in names else names[0])
    if op not in names:
        raise SystemExit(f"--op {op!r} is not one of {names}")
    cache_root = args.save_prob_maps or args.from_prob_maps
    cache = cache_root / sha[:12] if cache_root is not None else None

    outcomes = []
    with ThreadPoolExecutor(max_workers=POST_WORKERS) as pool:
        if args.from_prob_maps is not None:
            assert cache is not None
            meta = json.loads((cache / "meta.json").read_text(encoding="utf-8"))
            if meta["long_side"] != det.infer.long_side:
                raise SystemExit(
                    f"{cache} holds maps at long side {meta['long_side']}, the config "
                    f"asks for {det.infer.long_side}"
                )
            futures = []
            for doc_id in ids:
                page = gt[doc_id]
                info = ceiling_geometry(page, det.infer.long_side)
                levels = _read_png(cache / f"{doc_id}.png")
                if levels.shape != (info.h, info.w):
                    raise SystemExit(f"{doc_id}: cached map {levels.shape} does not fit the page")
                prob = np.zeros((info.pad_h, info.pad_w), np.float32)
                prob[: info.h, : info.w] = levels.astype(np.float32) / 255.0
                futures.append(pool.submit(score_page, prob, info, page, "val", det, False))
            outcomes = [f.result() for f in futures]
        else:
            device = select_device(cfg.device)
            model = build_detector(det.model, root, load_pretrained=False)
            model.load_state_dict(ck["model"])
            backend = TorchProbMap(model, device)
            ds = DetEvalDataset([("val", index)], root, det, with_labels=False)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=1, collate_fn=collate_eval, num_workers=det.data.eval_workers
            )
            if cache is not None:
                cache.mkdir(parents=True, exist_ok=True)
            futures = []
            for batch in loader:
                info = batch["info"]
                prob = backend(batch["images"][0].numpy())
                if cache is not None:
                    levels, prob = _quantise(prob, info.h, info.w)
                    _write_png(cache / f"{batch['doc_id']}.png", levels)
                futures.append(
                    pool.submit(score_page, prob, info, gt[batch["doc_id"]], "val", det, False)
                )
            outcomes = [f.result() for f in futures]
            if cache is not None:
                meta = {
                    "checkpoint_sha256": sha,
                    "long_side": det.infer.long_side,
                    "pages": len(outcomes),
                }
                (cache / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    res = summarise_pages("val", outcomes, det)
    ops = {n: op_metrics_json(res.records, n, m.counters) for n, m in res.ops.items()}
    result: dict[str, Any] = {
        "experiment_id": cfg.experiment_id,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha,
        "stored_op": stored,
        "op": op,
        "max_op_this_run": res.selected_op,
        "pages": res.pages,
        "care_lines": res.care_lines,
        "page_set": "phase5_baseline" if args.baseline_pages else "validation",
        "prob_maps": "uint8" if cache is not None else "float32",
        "post": dataclasses.asdict(det.post),
        "headline": {"op": op, **ops[op]},
        "ops": ops,
    }
    text = json.dumps(result, indent=1)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
