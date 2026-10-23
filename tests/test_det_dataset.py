"""Detector datasets: page index and its guards, seeded crops, sampler, collate and eval pages."""

from __future__ import annotations

import json
import pickle
import random

import cv2
import numpy as np
import pytest
import torch
from det_fixtures import (
    SOURCE,
    make_line,
    make_synthetic_pages,
    rect_polygon,
    tiny_det_config,
    write_processed_manifest,
)
from torch.utils.data import DataLoader

from glypharis.config.schema import DetAugmentConfig, DetConfig, DetDataConfig, DetInferConfig
from glypharis.data.det_augment import PhotometricAugmenter
from glypharis.data.det_dataset import (
    DET_DATA_VERSION,
    INDEX_VERSION,
    VAL_SUBSET_STREAM,
    CropBatchSampler,
    DetEvalDataset,
    DetTrainDataset,
    PageIndex,
    assert_no_page_leakage,
    care_centres,
    collate_eval,
    collate_train,
    det_worker_init,
    read_gt_pages,
    seeded_subset,
)
from glypharis.data.det_labels import LabelStats
from glypharis.data.pages import Page, read_pages, write_pages
from glypharis.eval.page_metrics import is_box_line


@pytest.fixture
def synth(tmp_path):
    return make_synthetic_pages(tmp_path, ignore_regions=True)


@pytest.fixture
def det(synth, tmp_path):
    return tiny_det_config(synth, tmp_path).det


def _train_index(synth) -> PageIndex:
    return PageIndex.build(synth.train_file, "train", synth.data_root)


# --------------------------------------------------------------------------------- page index
def test_page_index_round_trips_the_page_file(synth):
    index = _train_index(synth)
    assert len(index) == len(synth.train) == 12
    assert INDEX_VERSION == 1 and DET_DATA_VERSION == 1
    assert index.cache_dir.parent == synth.data_root / "cache" / "det"
    assert index.cache_dir.name.endswith(f"-{index.sha256[:12]}-v{INDEX_VERSION}")
    for i, page in enumerate(synth.train):
        rec = index.record(i)
        assert (rec.doc_id, rec.image, rec.width, rec.height) == (
            page.doc_id, page.image, page.width, page.height)  # fmt: skip
        polys, care = index.lines(i)
        assert len(polys) == len(page.lines)
        for poly, line in zip(polys, page.lines, strict=True):
            assert poly.dtype == np.float64 and np.array_equal(poly, np.asarray(line.polygon))
        assert care.tolist() == [is_box_line(ln) for ln in page.lines]
        regions = index.ignore_regions(i)
        assert [r.tolist() for r in regions] == page.ignore
    assert not index.lines(11)[1].any()  # the page without text lines
    assert len(index.ignore_regions(0)) == 1 and index.ignore_regions(1) == []


def test_page_index_reuses_its_cache_and_pickles_as_a_path(synth):
    first = _train_index(synth)
    stamp = (first.cache_dir / "points.npy").stat().st_mtime_ns
    again = _train_index(synth)
    assert again.cache_dir == first.cache_dir
    assert (again.cache_dir / "points.npy").stat().st_mtime_ns == stamp

    sub = first.subset(np.array([7, 2, 5]))
    assert sub.doc_ids() == [synth.train[i].doc_id for i in (7, 2, 5)]
    state = sub.__getstate__()
    assert set(state) == {"cache_dir", "ids"}
    blob = pickle.dumps(sub)
    assert len(blob) < 2000  # no polygons travel to the workers
    back = pickle.loads(blob)
    assert back.doc_ids() == sub.doc_ids() and back.sha256 == first.sha256
    for i in range(3):
        a, b = sub.lines(i), back.lines(i)
        assert all(np.array_equal(p, q) for p, q in zip(a[0], b[0], strict=True))
    assert sub.ids_sha256() == back.ids_sha256() != first.ids_sha256()
    assert sub.subset(np.array([1])).doc_ids() == [synth.train[2].doc_id]
    with pytest.raises(IndexError):
        sub.record(3)


def test_seeded_subsets_are_sorted_and_repeatable(synth):
    index = _train_index(synth)
    a = seeded_subset(index, 5, 1337, VAL_SUBSET_STREAM)
    b = seeded_subset(index, 5, 1337, VAL_SUBSET_STREAM)
    assert a.doc_ids() == b.doc_ids() and len(a) == 5
    assert a.doc_ids() == sorted(a.doc_ids())  # file order, and the fixture ids sort that way
    assert seeded_subset(index, None, 1337, 4) is index
    assert seeded_subset(index, 50, 1337, 4) is index
    assert seeded_subset(index, 5, 1338, 4).doc_ids() != a.doc_ids()


def test_page_index_guards(synth):
    with pytest.raises(ValueError, match="split"):
        PageIndex.build(synth.train_file, "validation", synth.data_root)
    with pytest.raises(ValueError, match="role"):
        PageIndex.build(synth.train_file, "test", synth.data_root)  # type: ignore[arg-type]

    root = synth.data_root
    pages = list(read_pages(synth.val_file))
    for p in pages:
        p.split = "test"
    test_file = root / "processed" / SOURCE / "pages_holdout.jsonl.gz"
    write_pages(test_file, pages)
    write_processed_manifest(root, SOURCE, {"validation": test_file})
    with pytest.raises(ValueError, match=r"refused: .*test"):
        PageIndex.build(test_file, "validation", root)

    for p in pages:
        p.split, p.source = "validation", "gdc"
    gdc_file = root / "processed" / "other" / "pages_validation.jsonl.gz"
    write_pages(gdc_file, pages)
    with pytest.raises(ValueError, match=r"refused: .*GDC"):
        PageIndex.build(gdc_file, "validation", root)


def test_page_index_refuses_a_file_that_differs_from_its_manifest(synth):
    man = synth.data_root / "manifests" / "processed" / f"{SOURCE}.json"
    d = json.loads(man.read_text(encoding="utf-8"))
    d["splits"]["train"]["pages_sha256"] = "0" * 64
    man.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        _train_index(synth)
    man.unlink()
    with pytest.raises(ValueError, match="no processed manifest"):
        _train_index(synth)


def test_leakage_and_gt_pages(synth):
    train = _train_index(synth)
    val = PageIndex.build(synth.val_file, "validation", synth.data_root)
    assert_no_page_leakage(train, val)

    pages = list(read_pages(synth.val_file))
    pages[0].doc_id = synth.train[3].doc_id
    leaky = synth.data_root / "processed" / SOURCE / "pages_validation.jsonl.gz"
    write_pages(leaky, pages)
    write_processed_manifest(synth.data_root, SOURCE, {"train": synth.train_file,
                                                      "validation": leaky})  # fmt: skip
    with pytest.raises(RuntimeError, match="LEAKAGE"):
        assert_no_page_leakage(train, PageIndex.build(leaky, "validation", synth.data_root))

    ids = [p.doc_id for p in synth.val[1:3]]
    gt = read_gt_pages(synth.val_file, ids)
    assert sorted(gt) == sorted(ids)
    assert gt[ids[0]].lines[0].line_id == synth.val[1].lines[0].line_id
    with pytest.raises(KeyError):
        read_gt_pages(synth.val_file, ["nope"])


# ------------------------------------------------------------------------------ training crops
def test_training_items_have_fixed_shapes_and_repeat_exactly(synth, det):
    ds = DetTrainDataset(_train_index(synth), synth.data_root, det, seed=1337)
    assert len(ds) == 12 * det.data.crops_per_page
    s = det.data.crop_size
    item = ds[(0, 3)]
    assert item["image"].shape == (s, s, 3) and item["image"].dtype == np.uint8
    for key in ("shrink_map", "shrink_mask", "thresh_mask"):
        assert item[key].shape == (s, s) and item[key].dtype == np.uint8
    assert item["thresh_map"].shape == (s, s) and item["thresh_map"].dtype == np.float32
    assert item["id"] == f"{synth.train[3].doc_id}#3" and isinstance(item["stats"], LabelStats)

    again = ds[(0, 3)]
    other = ds[(1, 3)]
    for key in ("image", "shrink_map", "shrink_mask", "thresh_map", "thresh_mask"):
        assert np.array_equal(item[key], again[key])
    assert not np.array_equal(item["image"], other["image"])
    # Any copy of the dataset (as a worker receives it) gives the same crop.
    copy = pickle.loads(pickle.dumps(ds))
    assert np.array_equal(copy[(0, 3)]["thresh_map"], item["thresh_map"])
    with pytest.raises(IndexError):
        ds[(0, len(ds))]


def test_kernels_land_on_the_dark_bars(synth, det):
    # Without photometric changes, kernel pixels must be ink: labels and pixels share one warp.
    ds = DetTrainDataset(_train_index(synth), synth.data_root, det, seed=3, photometric=False)
    ink: list[np.ndarray] = []
    for t in range(len(ds)):
        for epoch in range(3):
            it = ds[(epoch, t)]
            grey = it["image"].mean(axis=2)
            ink.append(grey[it["shrink_map"] == 1])
    values = np.concatenate(ink)
    assert len(values) > 500
    assert values.mean() < 110 and (values < 140).mean() > 0.95


def test_photometric_changes_only_the_image(synth, det):
    index = _train_index(synth)
    plain = DetTrainDataset(index, synth.data_root, det, seed=5, photometric=False)
    aug = DetTrainDataset(index, synth.data_root, det, seed=5, photometric=True)
    changed = 0
    for t in range(6):
        a, b = plain[(0, t)], aug[(0, t)]
        for key in ("shrink_map", "shrink_mask", "thresh_map", "thresh_mask"):
            assert np.array_equal(a[key], b[key])
        changed += not np.array_equal(a["image"], b["image"])
    assert changed > 0


def test_care_centres_average_each_line_over_its_own_points():
    # File order: care, don't care, care (six points), don't care, an empty line, care.
    hexagon = [[190, 95], [200, 90], [210, 95], [210, 105], [200, 110], [190, 105]]
    lines = [rect_polygon(40, 30, 60, 10), rect_polygon(500, 400, 80, 12), hexagon,
             rect_polygon(300, 420, 90, 10), [], rect_polygon(80, 300, 40, 8)]  # fmt: skip
    pts = np.array([p for line in lines for p in line], dtype=np.float64)
    offs = np.concatenate([[0], np.cumsum([len(line) for line in lines])])
    care = np.array([True, False, True, False, False, True])
    np.testing.assert_allclose(care_centres(pts, offs, care), [[40, 30], [200, 100], [80, 300]])
    assert care_centres(pts, offs, np.zeros(6, bool)).shape == (0, 2)
    assert care_centres(np.zeros((0, 2)), np.array([0]), np.zeros(0, bool)).shape == (0, 2)


def test_text_seeking_crops_contain_their_line_when_dont_care_lines_follow(tmp_path):
    # One care line at the top left, then two illegible lines at the bottom right. A centre that
    # took in the don't-care points would lie off the page and clamp every crop to the bottom-right
    # corner, away from the only text line.
    data_root, (w, h) = tmp_path / "data", (640, 480)
    image = f"raw/{SOURCE}/images/train/one.jpg"
    (data_root / image).parent.mkdir(parents=True)
    cv2.imencode(".jpg", np.full((h, w, 3), 230, np.uint8))[1].tofile(data_root / image)
    lines = [
        make_line("one_l0", rect_polygon(140, 90, 240, 60)),
        make_line("one_l1", rect_polygon(500, 420, 200, 40), legible=False),
        make_line("one_l2", rect_polygon(520, 320, 200, 40), legible=False),
    ]
    pages_file = data_root / "processed" / SOURCE / "pages_train.jsonl.gz"
    write_pages(pages_file, [Page("one", image, w, h, SOURCE, "train", lines)])
    write_processed_manifest(data_root, SOURCE, {"train": pages_file})
    data = DetDataConfig(
        train_pages=[pages_file.name],
        val_pages=["v.jsonl.gz"],
        crops_per_page=8,
        crop_size=64,
        text_crop_p=1.0,
        num_workers=0,
    )
    det = DetConfig(data=data, infer=DetInferConfig(long_side=160))
    index = PageIndex.build(pages_file, "train", data_root)
    ds = DetTrainDataset(index, data_root, det, seed=7, photometric=False)
    for epoch in range(3):
        for t in range(len(ds)):
            assert ds[(epoch, t)]["shrink_map"].any(), (epoch, t)


# --------------------------------------------------------------------------- sampler, collate
def test_crop_batch_sampler_is_deterministic_and_drops_the_last_batch():
    s = CropBatchSampler(n_items=21, batch_size=4, seed=9)
    assert s.num_batches(0) == 5 and len(s) == 5
    b0 = s.batches(0)
    assert b0 == s.batches(0) and b0 != s.batches(1)
    flat = [i for b in b0 for i in b]
    assert all(len(b) == 4 for b in b0) and len(set(flat)) == 20 and max(flat) < 21
    expected = np.random.default_rng([9, 0]).permutation(21)[:20].reshape(-1, 4).tolist()
    assert b0 == expected
    s.set_epoch(2, start_batch=3)
    keys = list(s)
    assert len(keys) == len(s) == 2
    assert keys == [[(2, i) for i in b] for b in s.batches(2)[3:]]


def test_collate_train_stacks_fixed_shape_batches(synth, det):
    ds = DetTrainDataset(_train_index(synth), synth.data_root, det, seed=1)
    sampler = CropBatchSampler(len(ds), 2, seed=1)
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_train, num_workers=0)
    s = det.data.crop_size
    batches = list(loader)
    assert len(batches) == len(ds) // 2
    for b in batches:
        assert b["images"].shape == (2, s, s, 3) and b["images"].dtype == torch.uint8
        assert b["thresh_map"].dtype == torch.float32 and b["shrink_map"].dtype == torch.uint8
        assert all(b[k].shape == (2, s, s) for k in ("shrink_map", "shrink_mask", "thresh_mask"))
        assert len(b["ids"]) == 2 and isinstance(b["stats"], LabelStats)
    one = batches[0]
    assert one["stats"].kernel_px == int(one["shrink_map"].sum())


def test_det_worker_init_limits_threads():
    torch_threads, cv_threads = torch.get_num_threads(), cv2.getNumThreads()
    np_state, py_state = np.random.get_state(), random.getstate()
    try:
        det_worker_init(0)
        assert torch.get_num_threads() == 1 and cv2.getNumThreads() == 1
    finally:
        torch.set_num_threads(torch_threads)
        cv2.setNumThreads(cv_threads)
        np.random.set_state(np_state)
        random.setstate(py_state)


# --------------------------------------------------------------------------------- eval pages
def test_eval_dataset_serves_padded_pages_with_labels(synth, det):
    train = _train_index(synth)
    val = PageIndex.build(synth.val_file, "validation", synth.data_root)
    ds = DetEvalDataset([("val", val), ("train_eval", train.subset(np.array([0, 4])))],
                        synth.data_root, det, with_labels=True)  # fmt: skip
    assert len(ds) == len(val) + 2
    roles = [ds[i]["role"] for i in range(len(ds))]
    assert roles == ["val"] * len(val) + ["train_eval"] * 2
    item = ds[len(val)]
    assert item["doc_id"] == synth.train[0].doc_id
    long_side = det.infer.long_side  # 160: the fixture pages are already that size
    assert (item["orig_w"], item["orig_h"]) == (160, 128)
    assert max(item["w"], item["h"]) == long_side
    assert item["pad_w"] % 32 == 0 and item["pad_h"] % 32 == 0
    assert item["image"].shape == (item["pad_h"], item["pad_w"], 3)
    for key in ("shrink_map", "shrink_mask", "thresh_map", "thresh_mask"):
        assert item[key].shape == (item["pad_h"], item["pad_w"])
    assert item["shrink_map"].sum() > 0
    assert (item["shrink_mask"][:30, :30] == 0).any()  # the first page's ignore region

    batch = collate_eval([item])
    assert batch["images"].shape == (1, item["pad_h"], item["pad_w"], 3)
    assert batch["info"].sx == pytest.approx(item["w"] / 160) and batch["doc_id"] == item["doc_id"]
    assert batch["thresh_map"].shape == (1, item["pad_h"], item["pad_w"])
    with pytest.raises(ValueError):
        collate_eval([item, item])

    plain = DetEvalDataset([("val", val)], synth.data_root, det, with_labels=False)[0]
    assert "shrink_map" not in plain and "shrink_map" not in collate_eval([plain])
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_eval_labels_follow_the_resize(synth, det, tmp_path):
    # At long side 320 every page is enlarged x2: the kernel area grows about 4x.
    val = PageIndex.build(synth.val_file, "validation", synth.data_root)
    big = tiny_det_config(synth, tmp_path, overrides=["det.infer.long_side=320"]).det
    a = DetEvalDataset([("val", val)], synth.data_root, det, with_labels=True)[0]
    b = DetEvalDataset([("val", val)], synth.data_root, big, with_labels=True)[0]
    assert b["w"] == 2 * a["w"] and b["h"] == 2 * a["h"]
    ratio = b["shrink_map"].sum() / max(a["shrink_map"].sum(), 1)
    assert 3.0 < ratio < 5.5


# ------------------------------------------------------------------------------- augmentation
OFF = {"color_p": 0.0, "grayscale_p": 0.0, "blur_p": 0.0, "noise_p": 0.0, "jpeg_p": 0.0}


def _crop(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.empty((64, 64, 3), np.uint8)
    img[:] = (180, 120, 60)
    img[20:40] = rng.integers(40, 220, (20, 64, 3))
    return img


def test_augmenter_is_deterministic_and_keeps_shape():
    aug = PhotometricAugmenter(DetAugmentConfig(**{k: 1.0 for k in OFF}))
    img = _crop()
    a = aug(img, np.random.default_rng([1, 2, 3]))
    b = aug(img, np.random.default_rng([1, 2, 3]))
    c = aug(img, np.random.default_rng([1, 2, 4]))
    assert a.shape == img.shape and a.dtype == np.uint8 and a.flags.c_contiguous
    assert np.array_equal(a, b) and not np.array_equal(a, c) and not np.array_equal(a, img)


def test_augmenter_steps():
    img = _crop()
    rng = np.random.default_rng(0)
    off = PhotometricAugmenter(DetAugmentConfig(**OFF))
    assert np.array_equal(off(img, rng), img)
    assert rng.bit_generator.state == _after_draws(5)  # one on/off draw per step, even at p = 0

    grey = PhotometricAugmenter(DetAugmentConfig(**{**OFF, "grayscale_p": 1.0}))(img, rng)
    assert np.array_equal(grey[..., 0], grey[..., 1]) and np.array_equal(grey[..., 1], grey[..., 2])

    flat = np.full((32, 32, 3), 128, np.uint8)
    noisy = PhotometricAugmenter(DetAugmentConfig(**{**OFF, "noise_p": 1.0}))(flat, rng)
    assert np.array_equal(noisy[..., 0], noisy[..., 2]) and noisy.std() > 1  # luminance noise

    # The colour matrix equals brightness, contrast and saturation applied in turn.
    r = np.random.default_rng(7)
    out = PhotometricAugmenter(DetAugmentConfig(**{**OFF, "color_p": 1.0}))(img, r)
    r = np.random.default_rng(7)
    r.random()
    b, k, sat = r.uniform(0.7, 1.3, size=3)
    luma = np.array([0.299, 0.587, 0.114])
    x = img.astype(np.float64) * b
    m = (x @ luma).mean()
    x = (x - m) * k + m
    g = (x @ luma)[..., None]
    ref = np.clip(np.rint((x - g) * sat + g), 0, 255)
    assert np.abs(out.astype(float) - ref).max() <= 1


def _after_draws(n: int) -> dict:
    rng = np.random.default_rng(0)
    for _ in range(n):
        rng.random()
    return rng.bit_generator.state
