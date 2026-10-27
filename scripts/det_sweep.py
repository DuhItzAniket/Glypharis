"""Post-processing sweep over cached probability maps (Phase 9, GW-DET-004).

    python scripts/det_sweep.py --config configs/det/gw-det-001.yaml \
        --maps runs/phase09/probmaps_1280/<sha12> --half A --out runs/phase09/sweep_A.npz

Scores every setting of the registered grid (`GRID`, 440 settings: ``bin_thresh`` x
``min_component_area`` x unclip operating point x ``box_thresh``) on one half of Phase 9's fixed
validation split, from the 8-bit maps `scripts/det_eval.py --save-prob-maps` wrote. Each page's
map is read once; candidates are found once per ``bin_thresh`` (at ``min_component_area`` 1, the
larger minimum is a filter on their pixel area) and boxes once per operating point (at the lowest
``box_thresh``, the higher ones are a filter on the box score), so every setting gives exactly
the boxes `score_page` would. Counts are at IoU 0.5 (`fast_det_counts`).

Writes ``--out`` (per-page tp, fp, fn for every setting, with the grid) and ``<out>.json`` (the
settings ranked by micro F1@0.5, the record and the arg-max, ties to the first in grid order).
``--only`` scores just the listed settings (a JSON list of grid settings), as half B needs.
Nothing here reads a test split.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Final

import numpy as np

from glypharis.config.cli import add_config_args, overrides_from_args
from glypharis.config.loader import load_config
from glypharis.config.schema import DetConfig, DetOperatingPoint
from glypharis.data.det_dataset import PageIndex, read_gt_pages
from glypharis.data.pages import Page
from glypharis.eval.det_matching import calibration_halves, fast_det_counts
from glypharis.models.det.postprocess import boxes_from_candidates, find_candidates
from glypharis.training.det_eval import ceiling_geometry

BIN_THRESH: Final = (0.20, 0.25, 0.30, 0.35, 0.40)
MIN_AREA: Final = (1, 3)
BOX_THRESH: Final = (0.4, 0.5, 0.6, 0.7)
INVERSE_RATIOS: Final = (0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
DBNET_RATIOS: Final = (1.3, 1.5, 1.7, 1.9, 2.1)
UNCLIP: Final = tuple(
    [DetOperatingPoint(f"inverse{round(r * 10):02d}", "inverse", r) for r in INVERSE_RATIOS]
    + [DetOperatingPoint(f"dbnet{round(r * 10):02d}", "dbnet", r) for r in DBNET_RATIOS]
)
# The record: inverse10 at bin 0.3, min area 3, box 0.6 (GW-DET-001's operating point of record).
RECORD: Final = (0.30, 3, "inverse10", 0.6)


def grid() -> list[tuple[float, int, str, float]]:
    """Settings in grid order (ties in the sweep go to the first): bin, min area, op, box."""
    return [
        (b, a, op.name, t)
        for b, a, op, t in itertools.product(BIN_THRESH, MIN_AREA, UNCLIP, BOX_THRESH)
    ]


_STATE: dict[str, Any] = {}


def _init(
    det_json: str,
    pages: dict[str, dict[str, Any]],
    long_side: int,
    maps: str,
    only: list[tuple[float, int, str, float]] | None = None,
) -> None:
    from glypharis.config.loader import from_dict

    _STATE["det"] = from_dict(DetConfig, json.loads(det_json), "det")
    _STATE["pages"] = {k: Page.from_dict(v) for k, v in pages.items()}
    _STATE["long_side"] = long_side
    _STATE["maps"] = Path(maps)
    _STATE["settings"] = [tuple(x) for x in only] if only else grid()


def _read_map(path: Path, h: int, w: int, pad_h: int, pad_w: int) -> np.ndarray:
    import cv2

    levels = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if levels is None or levels.shape != (h, w):
        raise SystemExit(f"{path}: missing, unreadable or the wrong size")
    prob = np.zeros((pad_h, pad_w), np.float32)
    prob[:h, :w] = levels.astype(np.float32) / 255.0
    return prob


def score_page_grid(doc_id: str) -> tuple[str, np.ndarray]:
    """(doc_id, counts[setting, (tp, fp, fn)]) for every grid setting."""
    det: DetConfig = _STATE["det"]
    page: Page = _STATE["pages"][doc_id]
    info = ceiling_geometry(page, _STATE["long_side"])
    prob = _read_map(_STATE["maps"] / f"{doc_id}.png", info.h, info.w, info.pad_h, info.pad_w)
    r = det.labels.shrink_ratio
    index = {s: i for i, s in enumerate(_STATE["settings"])}
    out = np.zeros((len(index), 3), dtype=np.int64)
    for b in BIN_THRESH:
        post = dataclasses.replace(det.post, bin_thresh=b, min_component_area=min(MIN_AREA),
                                   box_thresh=min(BOX_THRESH))  # fmt: skip
        cands, _ = find_candidates(prob, info, post)
        for a in MIN_AREA:
            kept = [c for c in cands if c.area_px >= a]
            for op in UNCLIP:
                boxes, _ = boxes_from_candidates(kept, op, post, r, info)
                for t in BOX_THRESH:
                    if (b, a, op.name, t) not in index:
                        continue
                    preds = [x.to_pred_line() for x in boxes if x.score >= t]
                    tp, fp, fn = fast_det_counts(page, preds, (0.5,)).det[0.5]
                    out[index[(b, a, op.name, t)]] = (tp, fp, fn)
    return doc_id, out


def f1(c: np.ndarray) -> float:
    tp, fp, fn = (float(x) for x in c)
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(parser)
    parser.add_argument("--maps", type=Path, required=True, help="cache dir with <doc_id>.png")
    parser.add_argument("--half", choices=["A", "B", "all"], required=True)
    parser.add_argument("--out", type=Path, required=True, help=".npz with the per-page counts")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", default=None, help="JSON list of grid settings to score")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, overrides_from_args(args))
    if cfg.det is None:
        raise SystemExit(f"{args.config}: not a detector config")
    det, root = cfg.det, Path(cfg.paths.data_root)
    meta = json.loads((args.maps / "meta.json").read_text(encoding="utf-8"))
    if meta["long_side"] != det.infer.long_side:
        raise SystemExit(f"{args.maps}: maps at long side {meta['long_side']}, config "
                         f"{det.infer.long_side}")  # fmt: skip
    pages_file = root / det.data.val_pages[0]
    ids = PageIndex.build(pages_file, "validation", root).doc_ids()
    if args.half != "all":
        ids = calibration_halves(ids)["AB".index(args.half)]
    gt = read_gt_pages(pages_file, ids)
    settings = grid()
    if args.only:
        settings = [tuple(x) for x in json.loads(args.only)]
        if missing := [x for x in settings if x not in grid()]:
            raise SystemExit(f"--only: not grid settings: {missing}")
    det_json = json.dumps(dataclasses.asdict(det))
    pages = {d: gt[d].to_dict() for d in ids}
    counts: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(
        args.workers, mp_context=get_context("spawn"), initializer=_init,
        initargs=(det_json, pages, det.infer.long_side, str(args.maps), settings),
    ) as pool:  # fmt: skip
        for n, (doc_id, c) in enumerate(pool.map(score_page_grid, ids, chunksize=4), 1):
            counts[doc_id] = c
            if n % 100 == 0:
                print(f"{n}/{len(ids)} pages", flush=True)
    docs = sorted(counts)
    arr = np.stack([counts[d] for d in docs])  # pages x settings x 3
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        counts=arr,
        doc_ids=np.array(docs),
        settings=np.array([json.dumps(s) for s in settings]),
        maps_meta=json.dumps(meta),
        half=args.half,
    )
    totals = arr.sum(axis=0)
    ranked = sorted(range(len(settings)), key=lambda i: (-f1(totals[i]), i))
    summary = {
        "half": args.half, "pages": len(docs), "settings": len(settings),
        "best": list(settings[ranked[0]]), "best_f1_50": f1(totals[ranked[0]]),
        "record": list(RECORD) if RECORD in settings else None,
        "record_f1_50": f1(totals[settings.index(RECORD)]) if RECORD in settings else None,
        "ranked": [{"setting": list(settings[i]), "f1_50": f1(totals[i]),
                    "tp_fp_fn": [int(x) for x in totals[i]]} for i in ranked[:20]],
    }  # fmt: skip
    summary_text = json.dumps(summary, indent=1) + "\n"
    args.out.with_suffix(".json").write_text(summary_text, encoding="utf-8")
    print(f"half {args.half}: {len(docs)} pages; record {RECORD}: F1@0.5 "
          f"{summary['record_f1_50']}")  # fmt: skip
    for i in ranked[:10]:
        print(f"  {settings[i]}: F1@0.5 {f1(totals[i]):.4f} (tp, fp, fn {tuple(totals[i])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
