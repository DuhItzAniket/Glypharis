"""Phase 16 variants: the INT8 recipe, the fused attention export, the one-setting variant builder
and the variants script (tiny made-up models, pages and lines; no real data or checkpoints)."""

from __future__ import annotations

import dataclasses
import io
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from det_fixtures import make_synthetic_pages
from PIL import Image
from test_frozen import CHARSET, DET_CKPT, REC_CKPT, write_det_checkpoint, write_record

from glypharis.config.schema import ModelConfig, SvtrConfig
from glypharis.data.linepack import LinePackWriter
from glypharis.data.text import Charset
from glypharis.efficiency import conditions, variants
from glypharis.efficiency.variants import (
    VARIANT_LABEL,
    VARIANTS,
    VariantError,
    compare_settings,
    deployed_variant,
    load_variant,
    prepare_variant,
    same_build,
)
from glypharis.models.rec.crnn import build_recognizer
from glypharis.models.rec.svtr import SVTR, _GlobalMixer
from glypharis.pipeline.frozen import FREEZE_RECORD, load_frozen
from glypharis.training.checkpoint import save_checkpoint

pytest.importorskip("onnxruntime")

REPO = Path(__file__).resolve().parents[1]
SVTR_CFG = ModelConfig(
    svtr=SvtrConfig(
        dims=[16, 24, 32],
        local_blocks=[1, 1, 0],
        global_blocks=[0, 1, 2],
        heads=[2, 3, 4],
        mlp_ratio=2.0,
    ),
    dropout=0.0,
)
HEIGHT = 32
CHARS = Charset.load(REPO / CHARSET).characters


def tiny_svtr(seed: int = 0, classes: int = len(CHARS) + 1) -> SVTR:
    torch.manual_seed(seed)
    model = SVTR(classes, SVTR_CFG).eval()
    # Untrained heads give near-uniform logits; scale them so greedy decoding emits characters.
    with torch.no_grad():
        model.head.weight.mul_(40.0)
    return model


def _ort(path: Path) -> Any:
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(session: Any, x: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    return session.run(["logits"], {"images": x.numpy(), "widths": w.numpy().astype(np.int64)})[0]


# Batch sizes and widths away from the trace width (256), with lines shorter than their batch.
SHAPES = [
    (1, 64, [64]),
    (3, 300, [300, 120, 64]),
    (2, 1536, [1536, 700]),
    (4, 512, [512, 4, 33, 400]),
]


# ----------------------------------------------------------------------------------- INT8


def test_int8_recipe_quantises_a_tiny_svtr_and_stays_close(tmp_path):
    from glypharis.efficiency.quant import RECIPE, QuantizationError, quantize_recognizer
    from glypharis.export.onnx import export_recognizer_onnx

    model = tiny_svtr()
    fp32 = export_recognizer_onnx(model, HEIGHT, tmp_path / "rec.onnx")
    record = quantize_recognizer(fp32, tmp_path / "rec.int8.onnx")
    assert record["recipe"] == RECIPE
    assert RECIPE["op_types_to_quantize"] == ["MatMul"] and RECIPE["weight_type"] == "QInt8"
    assert RECIPE["per_channel"] is False and RECIPE["reduce_range"] is False
    assert RECIPE["extra_options"] == {"MatMulConstBOnly": True}
    from glypharis.data.download import sha256_file

    assert record["input"]["sha256"] == sha256_file(fp32)
    assert record["output"]["sha256"] == sha256_file(tmp_path / "rec.int8.onnx")
    # Every linear layer is quantised; the two attention products per global block stay fp32.
    attention_products = 2 * sum(SVTR_CFG.svtr.global_blocks)  # type: ignore[union-attr]
    assert record["nodes_int8"]["MatMulInteger"] > 0
    assert record["nodes_int8"]["MatMul"] == attention_products
    assert record["nodes_int8"]["Conv"] == record["nodes_fp32"]["Conv"]  # convolutions untouched

    a, b = _ort(fp32), _ort(tmp_path / "rec.int8.onnx")
    for n, width, ws in SHAPES[:2]:
        x = torch.rand(n, 1, HEIGHT, width) * 2 - 1
        w = torch.tensor(ws)
        ya, yb = _run(a, x, w), _run(b, x, w)
        assert ya.shape == yb.shape
        assert np.abs(ya - yb).max() < 0.05 * np.abs(ya).max()

    again = quantize_recognizer(fp32, tmp_path / "again.onnx")
    assert again["output"]["sha256"] == record["output"]["sha256"]  # the recipe is deterministic
    with pytest.raises(QuantizationError, match="exists"):
        quantize_recognizer(fp32, tmp_path / "rec.int8.onnx")


# ------------------------------------------------------------------------- fused attention


def test_fused_attention_onnx_equals_the_reference_attention(tmp_path):
    from glypharis.efficiency.fused_attention import (
        FUSED_TOL,
        FusedGlobalMixer,
        export_fused_recognizer,
        fuse_attention,
    )

    model = tiny_svtr(1)
    fused = fuse_attention(model)
    assert sum(isinstance(m, FusedGlobalMixer) for m in fused.modules()) == 3
    assert sum(isinstance(m, _GlobalMixer) for m in model.modules()) == 3  # the original is kept
    record = export_fused_recognizer(model, HEIGHT, tmp_path / "fused.onnx", threads=2)
    assert record["fused_nodes"] == record["global_mixers"] == 3
    assert record["graph_ops"]["com.microsoft::MultiHeadAttention"] == 3
    assert "ai.onnx::Softmax" not in record["graph_ops"]
    session = _ort(tmp_path / "fused.onnx")
    with torch.no_grad():
        for n, width, ws in SHAPES:
            x = torch.rand(n, 1, HEIGHT, width) * 2 - 1
            w = torch.tensor(ws)
            ref = model(x, w)
            assert torch.equal(fused(x, w), ref)  # in PyTorch the swap computes the same thing
            got = _run(session, x, w)
            assert got.shape == tuple(ref.shape)
            assert np.abs(got - ref.numpy()).max() < FUSED_TOL


def test_fused_attention_refuses_a_model_without_attention_and_reports_ort_failures(
    tmp_path, monkeypatch
):
    from glypharis.efficiency import fused_attention
    from glypharis.efficiency.fused_attention import FusedAttentionError, export_fused_recognizer

    crnn = build_recognizer(
        10,
        ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=8, lstm_layers=1, dropout=0.0),
        "crnn",
    )
    with pytest.raises(FusedAttentionError, match="nothing to fuse"):
        export_fused_recognizer(crnn, HEIGHT, tmp_path / "crnn.onnx")

    def no_kernel(path: Any, threads: Any) -> Any:
        raise RuntimeError("Could not find an implementation for MultiHeadAttention(1)")

    monkeypatch.setattr(fused_attention, "ort_session", no_kernel)
    with pytest.raises(FusedAttentionError, match="ONNX Runtime cannot run"):
        export_fused_recognizer(tiny_svtr(), HEIGHT, tmp_path / "fused.onnx")


# ----------------------------------------------------------------------- the variant builder


def write_svtr_checkpoint(
    path: Path,
    experiment: str = "GW-REC-034",
    seed: int = 0,
    characters: list[str] | None = None,
    height: int = HEIGHT,
    model_name: str = "svtr",
) -> None:
    chars = characters or CHARS
    if model_name == "svtr":
        model: torch.nn.Module = tiny_svtr(seed, len(chars) + 1)
        model_cfg = SVTR_CFG
    else:
        model_cfg = ModelConfig(
            cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=8, lstm_layers=1, dropout=0.0
        )
        torch.manual_seed(seed)
        model = build_recognizer(len(chars) + 1, model_cfg, model_name)
    cfg = {
        "experiment_id": experiment,
        "model_name": model_name,
        "model": dataclasses.asdict(model_cfg),
        "data": {"image_height": height, "max_width": 512, "charset": CHARSET.as_posix()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        path,
        {
            "model": model.state_dict(),
            "config": cfg,
            "config_hash": "r" * 64,
            "git": {"commit": "rec0"},
            "charset": chars,
        },
    )


@pytest.fixture(scope="module")
def v1_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A repository layout with a tiny SVTR v1 (record, charset, checkpoints) and its fp32 ONNX
    bundle, written from the record as the GW-EFF-006 export does."""
    from glypharis.export.bundle import write_bundle

    root = tmp_path_factory.mktemp("v1")
    (root / CHARSET).parent.mkdir(parents=True)
    shutil.copyfile(REPO / CHARSET, root / CHARSET)
    write_svtr_checkpoint(root / REC_CKPT)
    write_det_checkpoint(root / DET_CKPT)
    write_record(root, recognizer={"model_name": "svtr"})
    frozen = load_frozen("cpu", root=root)
    write_bundle(
        root / DET_CKPT,
        frozen.ocr.detector.spec,
        frozen.recognizer,
        root / "bundle",
        "glypharis_v1",
        {"record": "configs/final/glypharis_v1.yaml"},
    )
    return root


@pytest.fixture
def frozen(v1_root: Path) -> Any:
    return load_frozen("cpu", root=v1_root)


def _build(name: str, frozen: Any, root: Path, work: Path, checkpoint: Path | None = None) -> Any:
    bundle = None if name == "v1-torch" else root / "bundle"
    prepare_variant(name, frozen, bundle, work / name, 2, checkpoint)
    return load_variant(name, frozen, bundle, work / name, 2)


ONE_SETTING = {
    "det-long-side-1024": ("detector.infer", {"long_side": 160}, {"long_side": 1024}),
    "onnx-det-torch-rec": ("recognizer.runtime", variants.ORT_FP32, variants.TORCH_STRICT),
    "int8-recognizer": ("recognizer.runtime", variants.ORT_FP32, variants.ORT_INT8),
    "fused-attention": ("recognizer.runtime", variants.ORT_FP32, variants.ORT_FUSED),
}


@pytest.mark.parametrize("name", sorted(ONE_SETTING))
def test_a_variant_changes_its_one_setting_and_says_it_is_a_new_version(
    name, frozen, v1_root, tmp_path
):
    build = _build(name, frozen, v1_root, tmp_path)
    setting, was, now = ONE_SETTING[name]
    assert build.version == build.record["version"] == VARIANT_LABEL
    assert build.record["setting"] == setting == VARIANTS[name].setting
    assert build.record["changed"] == {setting: {"from": was, "to": now}}
    v1 = build.record["v1_settings"]
    got = build.record["settings"]
    others = [k for k in variants.SETTINGS if k not in (setting, *variants.RUNTIMES)]
    assert all(got[k] == v1[k] for k in others)  # every other setting is the record's
    assert got["recognizer.model"]["experiment"] == "GW-REC-034"
    assert got["detector.infer"] == (
        {"long_side": 1024} if setting == "detector.infer" else {"long_side": 160}
    )
    assert VARIANT_LABEL in build.ocr.model_version
    doc = build.record["variant_json"]
    assert doc["version"] == VARIANT_LABEL and doc["bundle"]["manifest_sha256"]
    crops = [
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (40, w), np.uint8))
        for w in (60, 300)
    ]
    texts = [r.text for r in build.ocr.recognizer.read(crops)]
    ref = [r.text for r in frozen.recognizer.read(crops)]
    if name in ("det-long-side-1024", "onnx-det-torch-rec", "fused-attention"):
        assert texts == ref  # the recognizer is v1's model, in PyTorch or exported
    from glypharis.efficiency.onnx_build import check_sessions

    opts = check_sessions(build.ocr, 2)  # every ONNX session has the requested threads
    assert opts["detector"]["intra_op_num_threads"] == 2
    assert (opts["recognizer"] is None) == (name == "onnx-det-torch-rec")


def test_v1_builds_are_labelled_v1_and_agree(frozen, v1_root, tmp_path):
    torch_build = _build("v1-torch", frozen, v1_root, tmp_path)
    onnx_build = _build("v1-onnx", frozen, v1_root, tmp_path)
    for b in (torch_build, onnx_build):
        assert b.version == "v1" and b.record["changed"] is None
    assert torch_build.ocr is frozen.ocr and torch_build.torch_reference is None
    assert onnx_build.torch_reference is frozen.recognizer
    a, b = torch_build.record["settings"], onnx_build.record["settings"]
    assert {k for k in a if a[k] != b[k]} == set(variants.RUNTIMES)
    crops = [
        Image.fromarray(np.random.default_rng(1).integers(0, 255, (40, w), np.uint8))
        for w in (30, 200, 500)
    ]
    for la, lb in zip(
        torch_build.ocr.recognizer.logits(crops),
        onnx_build.ocr.recognizer.logits(crops),
        strict=True,
    ):
        assert torch.allclose(la, lb, atol=1e-4)


def test_compare_settings_refuses_a_second_change_and_no_change():
    base = {k: 0 for k in variants.SETTINGS}
    one = {**base, "detector.infer": 1}
    assert compare_settings(base, one, frozenset({"detector.infer"}), "x") == {
        "detector.infer": {"from": 0, "to": 1}
    }
    with pytest.raises(VariantError, match="differ as well"):
        compare_settings(base, {**one, "crop": 2}, frozenset({"detector.infer"}), "x")
    with pytest.raises(VariantError, match="not a variant"):
        compare_settings(base, base, frozenset({"detector.infer"}), "x")
    with pytest.raises(VariantError, match="exactly"):
        compare_settings(base, {"crop": 0}, frozenset(), "x")


def test_a_bundle_with_other_weights_is_refused(frozen, v1_root, tmp_path):
    from glypharis.export.bundle import write_bundle
    from glypharis.pipeline.recognizer import TextRecognizer

    other = TextRecognizer(
        tiny_svtr(7),
        frozen.recognizer.charset,
        HEIGHT,
        512,
        "cpu",
        2.35,
        frozen.recognizer.arch,
        strict_fp32=True,
    )
    bad = write_bundle(
        v1_root / DET_CKPT, frozen.ocr.detector.spec, other, tmp_path / "bad", "glypharis_v1"
    )
    prepare_variant("v1-onnx", frozen, bad, tmp_path / "w", 2)
    with pytest.raises(VariantError, match=r"recognizer\.pt does not hold"):
        load_variant("v1-onnx", frozen, bad, tmp_path / "w", 2)


def test_prepared_files_are_checked_and_never_overwritten(frozen, v1_root, tmp_path):
    work = tmp_path / "int8-recognizer"
    prepare_variant("int8-recognizer", frozen, v1_root / "bundle", work, 2)
    with pytest.raises(VariantError, match="prepared already"):
        prepare_variant("int8-recognizer", frozen, v1_root / "bundle", work, 2)
    with pytest.raises(VariantError, match="is for"):
        load_variant("fused-attention", frozen, v1_root / "bundle", work, 2)
    with (work / "recognizer.int8.onnx").open("ab") as fh:
        fh.write(b"\0")
    with pytest.raises(VariantError, match="changed since"):
        load_variant("int8-recognizer", frozen, v1_root / "bundle", work, 2)
    with pytest.raises(VariantError, match="prepare"):
        load_variant("v1-onnx", frozen, v1_root / "bundle", tmp_path / "nothing", 2)
    with pytest.raises(VariantError, match="--bundle"):
        prepare_variant("v1-onnx", frozen, None, tmp_path / "nb", 2)
    with pytest.raises(VariantError, match="unknown variant"):
        prepare_variant("det-long-side-1536", frozen, v1_root / "bundle", tmp_path / "u", 2)


def test_other_recognizer_checkpoints_change_only_the_model(frozen, v1_root, tmp_path):
    ck = tmp_path / "ck" / "GW-REC-020.pt"
    write_svtr_checkpoint(ck, "GW-REC-020", seed=3, height=48)
    build = _build("rec-GW-REC-020", frozen, v1_root, tmp_path, ck)
    changed = build.record["changed"]["recognizer.model"]
    assert changed["from"]["experiment"] == "GW-REC-034"
    assert changed["to"]["experiment"] == "GW-REC-020" and changed["to"]["height"] == 48
    assert build.version == VARIANT_LABEL
    reported = json.dumps(variants.report_record(build.record))
    assert tmp_path.as_posix() not in reported and '"path": "GW-REC-020.pt"' in reported
    assert build.record["settings"]["recognizer.temperature"] == 2.35  # the record's, kept
    assert build.torch_reference is not None and build.torch_reference.height == 48
    crops = [Image.fromarray(np.random.default_rng(2).integers(0, 255, (40, 220), np.uint8))]
    a = build.ocr.recognizer.logits(crops)[0]
    b = build.torch_reference.logits(crops)[0]
    assert torch.allclose(a, b, atol=1e-4)  # the export of that checkpoint

    crnn = tmp_path / "ck" / "GW-REC-003.pt"
    write_svtr_checkpoint(crnn, "GW-REC-003", model_name="crnn")
    assert _build("rec-GW-REC-003", frozen, v1_root, tmp_path, crnn).record["changed"]

    wrong = tmp_path / "ck" / "wrong.pt"
    write_svtr_checkpoint(wrong, "GW-REC-022")
    with pytest.raises(VariantError, match="not GW-REC-020"):
        prepare_variant("rec-GW-REC-020", frozen, v1_root / "bundle", tmp_path / "x", 2, wrong)
    swapped = tmp_path / "ck" / "swapped.pt"
    write_svtr_checkpoint(swapped, "GW-REC-022", characters=[CHARS[1], CHARS[0], *CHARS[2:]])
    prepare_variant("rec-GW-REC-022", frozen, v1_root / "bundle", tmp_path / "y", 2, swapped)
    with pytest.raises(VariantError, match=r"recognizer\.charset"):
        load_variant("rec-GW-REC-022", frozen, v1_root / "bundle", tmp_path / "y", 2)
    with pytest.raises(VariantError, match="needs the GW-REC-006"):
        prepare_variant("rec-GW-REC-006", frozen, v1_root / "bundle", tmp_path / "z", 2)


def _small_detector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import test_frozen

    from glypharis.config.schema import DetModelConfig

    small = DetModelConfig(backbone="mobilenet_v3_small", init="scratch", inner_channels=8)
    monkeypatch.setattr(test_frozen, "DET_MODEL", small)
    ck = tmp_path / "GW-DET-003.pt"
    write_det_checkpoint(ck)
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    obj["config"]["experiment_id"] = "GW-DET-003"
    save_checkpoint(ck, obj)
    return ck


def test_the_smaller_detector_changes_only_the_detector_model(
    frozen, v1_root, tmp_path, monkeypatch
):
    ck = _small_detector(tmp_path, monkeypatch)
    build = _build("det-GW-DET-003", frozen, v1_root, tmp_path, ck)
    changed = build.record["changed"]["detector.model"]
    assert changed["from"]["experiment"] == "GW-DET-001"
    assert changed["to"]["experiment"] == "GW-DET-003"
    assert changed["to"]["arch"]["inner_channels"] == 8
    s = build.record["settings"]
    assert s["detector.infer"] == {"long_side": 160}  # the record's resolution and
    assert s["detector.operating_point"] == "inverse09"  # operating point, not the checkpoint's
    assert build.ocr.detector.spec.post.box_thresh == 0.5


# -------------------------------------------------------------------- the deployed builds


DEPLOYED = ["v1-onnx", "det-long-side-1024", "onnx-det-torch-rec", "int8-recognizer",
            "fused-attention", "rec-GW-REC-020", "det-GW-DET-003"]  # fmt: skip


@pytest.mark.parametrize("name", DEPLOYED)
def test_a_deployed_build_is_the_checked_build(name, frozen, v1_root, tmp_path, monkeypatch):
    """What a profile times (the bundle and the variant's files only) holds the settings and
    models of the build the full chain checked, and reads the same."""
    ck = None
    if name == "rec-GW-REC-020":
        ck = tmp_path / "ck.pt"
        write_svtr_checkpoint(ck, "GW-REC-020", seed=3, height=48)
    elif name == "det-GW-DET-003":
        ck = _small_detector(tmp_path, monkeypatch)
    build = _build(name, frozen, v1_root, tmp_path, ck)
    ocr, memory = deployed_variant(name, v1_root / "bundle", tmp_path / name, 2)
    assert ocr is not build.ocr and memory["loaded"][0].startswith("the bundle")
    assert same_build(ocr, build) == build.record["settings"]
    assert ocr.model_version == build.ocr.model_version
    crops = [Image.fromarray(np.random.default_rng(5).integers(0, 255, (40, w), np.uint8))
             for w in (50, 260)]  # fmt: skip
    for a, b in zip(ocr.recognizer.logits(crops), build.ocr.recognizer.logits(crops), strict=True):
        assert torch.equal(a, b)
    page = np.random.default_rng(6).integers(0, 255, (128, 192, 3), np.uint8)
    assert np.array_equal(ocr.detector.backend(page), build.ocr.detector.backend(page))
    assert ocr.detector.spec == build.ocr.detector.spec


def test_a_deployed_build_with_other_settings_is_refused(frozen, v1_root, tmp_path):
    build = _build("det-long-side-1280", frozen, v1_root, tmp_path)
    _build("det-long-side-1024", frozen, v1_root, tmp_path)
    ocr, _ = deployed_variant(
        "det-long-side-1024", v1_root / "bundle", tmp_path / "det-long-side-1024", 2
    )
    with pytest.raises(VariantError, match=r"detector\.infer"):
        same_build(ocr, build)
    with pytest.raises(VariantError, match="load_frozen"):
        deployed_variant("v1-torch", v1_root / "bundle", tmp_path / "v1-torch", 2)


def test_a_failed_preparation_leaves_the_work_directory_as_it_was(
    frozen, v1_root, tmp_path, monkeypatch
):
    work = tmp_path / "int8"
    work.mkdir()
    (work / "note.txt").write_text("kept", encoding="utf-8")
    real = variants.quantize_recognizer

    def half(src: Any, out: Any, **kw: Any) -> Any:
        Path(out).write_bytes(b"half a model")
        raise RuntimeError("the machine went down")

    monkeypatch.setattr(variants, "quantize_recognizer", half)
    with pytest.raises(RuntimeError, match="went down"):
        prepare_variant("int8-recognizer", frozen, v1_root / "bundle", work, 2)
    assert sorted(p.name for p in work.iterdir()) == ["note.txt"]
    monkeypatch.setattr(variants, "quantize_recognizer", real)
    doc = prepare_variant("int8-recognizer", frozen, v1_root / "bundle", work, 2)
    assert "recognizer.int8.onnx" in doc["files"]


def test_a_recognizer_variant_ships_its_release_file(frozen, v1_root, tmp_path):
    ck = tmp_path / "ck.pt"
    write_svtr_checkpoint(ck, "GW-REC-020", seed=3, height=48)
    build = _build("rec-GW-REC-020", frozen, v1_root, tmp_path, ck)
    files = build.record["variant_json"]["files"]
    assert set(files) == {"recognizer.onnx", "recognizer.pt"}
    # A release file that is not the checkpoint's recognizer is refused by the full chain.
    other = tmp_path / "other.pt"
    write_svtr_checkpoint(other, "GW-REC-020", seed=4, height=48)
    work = tmp_path / "swap"
    prepare_variant("rec-GW-REC-020", frozen, v1_root / "bundle", work, 2, ck)
    doc = json.loads((work / "variant.json").read_text(encoding="utf-8"))
    doc["checkpoint"].update(path=other.as_posix(),
                             sha256=variants.sha256_file(other))  # fmt: skip
    (work / "variant.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(VariantError, match=r"recognizer\.pt does not hold"):
        load_variant("rec-GW-REC-020", frozen, v1_root / "bundle", work, 2)


# ------------------------------------------------------------------------------- the script


def _png(w: int, h: int, seed: int) -> bytes:
    img = Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w), np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def script(v1_root, tmp_path, monkeypatch):
    """phase16_variants with the git check, threads, conditions and data paths made up."""
    import phase16_variants as pv

    synth = make_synthetic_pages(tmp_path, n_train=1, n_val=4)
    pack = synth.data_root / "processed/hiertext/lines_validation"
    with LinePackWriter(pack) as w:
        for i, width in enumerate((40, 90, 160, 60, 30, 200)):
            w.add(
                f"l{i}", f"d{i}", "validation", "hiertext", "ab c", _png(width, 20, i), (width, 20)
            )
    fake_threads = {
        "requested": 2,
        "intra_op": 2,
        "inter_op": 2,
        "env": {},
        "env_before_torch_import": True,
        "preloaded": [],
    }
    monkeypatch.setattr(pv, "set_threads", lambda n, require_fresh=True: fake_threads)
    monkeypatch.setattr(pv, "check_threads", lambda n: {"intra_op": n, "inter_op": n, "env": {}})
    monkeypatch.setattr(pv, "assert_v1_code", lambda root: {"v1_code": "equal", "tag": "t"})
    monkeypatch.setattr(pv, "git_state", lambda: {"git_commit": "abc", "git_dirty": False})
    monkeypatch.setattr(pv, "cpu_busy_percent", lambda: 1.0)
    cpu = {"on_ac_power": True, "power_plan": "x (Balanced)", "power_mode": "best performance"}
    monkeypatch.setattr(
        pv,
        "ProfileConditions",
        lambda: conditions.ProfileConditions(cpu=lambda: cpu, gpu=lambda: None),
    )
    monkeypatch.setattr(pv, "eval_paths", lambda record, root: (root / synth.val_pages, pack))
    monkeypatch.setattr(pv, "e2e_samples", lambda root, s: iter(synth.val))

    def lines(root: Path, s: str) -> Any:
        for i in range(200):
            yield f"r{i}", "ab", Image.open(io.BytesIO(_png(30 + i % 7 * 20, 24, i)))

    monkeypatch.setattr(pv, "rec_samples", lines)
    monkeypatch.chdir(v1_root)
    args = [
        "--bundle",
        str(v1_root / "bundle"),
        "--threads",
        "2",
        "--work",
        str(tmp_path / "work"),
        "--data-root",
        str(synth.data_root),
    ]
    return pv, args, tmp_path


def test_script_build_eval_and_profile_write_the_documented_files(script):
    pv, args, tmp = script
    assert pv.main(["build", "--variant", "int8-recognizer", *args]) == 0
    assert (tmp / "work" / "variant.json").is_file()

    out = tmp / "out"
    assert (
        pv.main(
            [
                "eval",
                "--variant",
                "int8-recognizer",
                "--set",
                "lines",
                "--limit",
                "5",
                "--out",
                str(out / "lines.json"),
                *args,
            ]
        )
        == 0
    )
    lines = json.loads((out / "lines.json").read_text(encoding="utf-8"))
    assert lines["kind"] == "phase16-variant-lines" and lines["experiment"] == "GW-EFF-008"
    assert lines["batch"] == 1  # each line read on its own, as the line latency is timed
    assert lines["version"] == VARIANT_LABEL and lines["smoke"] is True and lines["lines"] == 5
    assert lines["changed"]["recognizer.runtime"]["to"] == variants.ORT_INT8
    per = lines["per_line"]
    assert len(per["keys"]) == len(per["strict"]["errors"]) == len(per["strict"]["ref_chars"]) == 5
    assert per["strict"]["ref_chars"] == [4] * 5
    assert lines["cer"]["strict"] == sum(per["strict"]["errors"]) / 20
    assert lines["torch_check"] is None  # INT8 is compared with fp32 ONNX by the scorer
    assert lines["source"]["seed"] == 20261028
    assert (out / "lines_outputs.jsonl").is_file()
    assert lines["outputs"]["sha256"]

    assert (
        pv.main(
            [
                "eval",
                "--variant",
                "int8-recognizer",
                "--set",
                "pages",
                "--limit",
                "2",
                "--out",
                str(out / "pages.json"),
                *args,
            ]
        )
        == 0
    )
    pages = json.loads((out / "pages.json").read_text(encoding="utf-8"))
    assert pages["pages"] == 2 and set(pages["metrics"]) == {"strict", "normalised"}
    strict = pages["per_page"]["strict"]
    assert len(pages["per_page"]["doc_ids"]) == len(strict["bag_errors"]) == 2
    ref = sum(strict["bag_ref_chars"])
    assert pages["page_cer"] == (sum(strict["bag_errors"]) / ref if ref else 0.0)

    prof = out / "profile.json"
    assert (
        pv.main(
            [
                "profile",
                "--variant",
                "int8-recognizer",
                "--set",
                "pages",
                "--experiment",
                "GW-EFF-009",
                "--run",
                "1",
                "--smoke",
                "3",
                "--out",
                str(prof),
                *args,
            ]
        )
        == 0
    )
    p = json.loads(prof.read_text(encoding="utf-8"))
    assert p["kind"] == "phase16-variant-profile" and p["task"] == "e2e" and p["smoke"] is True
    assert p["warmup"] == 1 and len(p["times_ms"]) == len(p["lines"]) == 3
    assert len(p["parts_ms"]) == len(p["input_sizes"]) == 3 and len(p["warmup_times_ms"]) == 1
    assert p["settings"]["cpu_threads"] == 2 and p["settings"]["runtime"]["recognizer"] == (
        variants.ORT_INT8
    )
    assert p["session_options"]["recognizer"]["intra_op_num_threads"] == 2
    assert p["conditions"]["on_ac_power"] is True and p["gpu_state"] == {"start": None, "end": None}
    assert p["experiment"] == "GW-EFF-009" and "recognizer.int8.onnx" in p["peak_mem"]["loaded"]
    assert p["peak_mem"]["scope"].startswith("the deployed build")
    for key in (
        "started_utc",
        "ended_utc",
        "peak_mem_mib",
        "mean_ms",
        "p95_ms",
        "v1_code",
        "threads",
        "build",
        "conditions_end",
        "run",
    ):
        assert key in p

    with pytest.raises(SystemExit, match="exists"):
        pv.main(
            [
                "profile",
                "--variant",
                "int8-recognizer",
                "--set",
                "pages",
                "--experiment",
                "GW-EFF-009",
                "--run",
                "1",
                "--smoke",
                "3",
                "--out",
                str(prof),
                *args,
            ]
        )
    with pytest.raises(SystemExit, match="GW-EFF-008 times pages of no variant"):
        pv.main(["profile", "--variant", "int8-recognizer", "--set", "pages", "--experiment",
                 "GW-EFF-008", "--run", "1", "--smoke", "3", "--out", str(out / "x.json"),
                 *args])  # fmt: skip
    with pytest.raises(SystemExit) as exc:  # a profile names its sitting
        pv.main(["profile", "--variant", "int8-recognizer", "--set", "lines", "--run", "1",
                 *args])  # fmt: skip
    assert exc.value.code == 2
    with pytest.raises(SystemExit, match="needs --out"):
        pv.main(["eval", "--variant", "int8-recognizer", "--set", "lines", "--limit", "2", *args])


def test_script_full_line_profile_passes_phase14s_profile_rules(script):
    from phase14_score import profile_problems

    pv, args, tmp = script
    assert pv.main(["build", "--variant", "v1-onnx", *args]) == 0
    out = tmp / "p.json"
    assert (
        pv.main(
            [
                "profile",
                "--variant",
                "v1-onnx",
                "--set",
                "lines",
                "--experiment",
                "GW-EFF-008",
                "--run",
                "2",
                "--out",
                str(out),
                *args,
            ]
        )
        == 0
    )
    p = json.loads(out.read_text(encoding="utf-8"))
    assert p["version"] == "v1" and p["task"] == "rec" and p["batch"] == 1
    assert len(p["times_ms"]) == 100 and p["warmup"] == 10 and len(p["keys"]) == 100
    assert profile_problems("v1-onnx", p, None, threads=2) == []
    assert p["run"] == 2 and p["smoke"] is False and p["experiment"] == "GW-EFF-008"


def test_script_default_profile_names_carry_the_sitting(script, monkeypatch):
    pv, args, tmp = script
    monkeypatch.setattr(pv, "REPORTS", tmp / "reports")
    assert pv.main(["build", "--variant", "v1-onnx", *args]) == 0
    for experiment in ("GW-EFF-008", "GW-EFF-009"):
        assert pv.main(["profile", "--variant", "v1-onnx", "--set", "lines", "--experiment",
                        experiment, "--run", "1", *args]) == 0  # fmt: skip
    names = sorted(p.name for p in (tmp / "reports" / "v1-onnx").iterdir())
    assert names == [
        "profile_lines_2t_GW-EFF-008_run1.json",
        "profile_lines_2t_GW-EFF-009_run1.json",
    ]


def test_script_eval_refuses_an_existing_output_before_it_loads_anything(script, monkeypatch):
    pv, args, tmp = script

    def never(*a: Any, **k: Any) -> Any:
        raise AssertionError("loaded a model although the output exists")

    monkeypatch.setattr(pv, "load_variant", never)
    out = tmp / "e" / "lines.json"
    out.parent.mkdir()
    out.write_text("{}\n", encoding="utf-8")
    argv = ["eval", "--variant", "int8-recognizer", "--set", "lines", "--limit", "2", "--out",
            str(out), *args]  # fmt: skip
    with pytest.raises(SystemExit, match=r"lines.json exists"):
        pv.main(argv)
    out.unlink()
    (out.parent / "lines_outputs.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"lines_outputs.jsonl exists"):
        pv.main(argv)


def test_script_lines_eval_checks_an_export_against_its_pytorch_model(script):
    pv, args, tmp = script
    assert pv.main(["build", "--variant", "fused-attention", *args]) == 0
    out = tmp / "fused_lines.json"
    assert (
        pv.main(
            [
                "eval",
                "--variant",
                "fused-attention",
                "--set",
                "lines",
                "--limit",
                "6",
                "--out",
                str(out),
                *args,
            ]
        )
        == 0
    )
    doc = json.loads(out.read_text(encoding="utf-8"))
    check = doc["torch_check"]
    assert check["lines"] == 6 and check["pass"] is True
    assert check["logit_max_abs_diff"] < 1e-4 and check["identical_text"] == 6
    assert doc["experiment"] == "GW-EFF-009"


def test_script_lists_every_variant(capsys):
    import phase16_variants as pv

    assert pv.main(["list"]) == 0
    text = capsys.readouterr().out
    assert all(name in text for name in VARIANTS)
    assert (
        "GW-EFF-008 lines: v1-onnx run 1, int8-recognizer run 1, int8-recognizer run 2, "
        "v1-onnx run 2" in text
    )
    assert "GW-EFF-011 lines: rec-GW-REC-003 run 1, rec-GW-REC-020 run 1" in text


def test_the_real_freeze_record_names_the_evaluation_sets():
    import phase16_variants as pv

    pages, pack = pv.eval_paths(REPO / FREEZE_RECORD, Path("data"))
    assert pages.as_posix() == "data/processed/hiertext/pages_validation.jsonl.gz"
    assert pack.as_posix() == "data/processed/hiertext/lines_validation"


def test_reports_record_checkpoint_paths_without_local_folders():
    from glypharis.efficiency.variants import portable_path, report_record

    assert portable_path("C:/Users/someone/work/Glypharis/checkpoints/GW-REC-003/best.pt") == (
        "checkpoints/GW-REC-003/best.pt"
    )
    assert portable_path("/home/x/checkpoints/old/checkpoints/GW-DET-003/best.pt") == (
        "checkpoints/GW-DET-003/best.pt"
    )
    assert portable_path("D:/elsewhere/best.pt") == "best.pt"
    ck = {"path": "C:/Users/someone/checkpoints/GW-REC-020/best.pt", "sha256": "ab"}
    record = {"checkpoint": ck, "changed": ["x"]}
    out = report_record(record)
    assert out == {"checkpoint": {"path": "checkpoints/GW-REC-020/best.pt", "sha256": "ab"},
                   "changed": ["x"]}  # fmt: skip
    assert record["checkpoint"]["path"].startswith("C:/Users/")  # the work record is untouched
    nested = {"variant_json": {"checkpoint": dict(ck)}, "variant_json_sha256": "cd"}
    assert report_record(nested)["variant_json"]["checkpoint"]["path"] == (
        "checkpoints/GW-REC-020/best.pt"
    )
    assert report_record({"checkpoint": None}) == {"checkpoint": None}
