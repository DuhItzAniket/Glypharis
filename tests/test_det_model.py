from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.models._api as tv_api
from torch import nn
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

from glypharis.config.schema import DetModelConfig
from glypharis.models.det.backbone import (
    BACKBONE_TAPS,
    GlobalMean,
    MobileNetV3Features,
    WeightsError,
    load_imagenet_state_dict,
)
from glypharis.models.det.dbnet import DBNet, build_detector, count_parameters
from glypharis.models.det.preprocess import (
    PAD_RGB,
    ResizeInfo,
    prepare_page,
    scaled_size,
    to_page_coords,
)
from glypharis.models.det.runtime import TorchProbMap, strict_fp32

STOCK = {"mobilenet_v3_large": mobilenet_v3_large, "mobilenet_v3_small": mobilenet_v3_small}
TAP_CHANNELS = {"mobilenet_v3_large": (24, 40, 112, 960), "mobilenet_v3_small": (16, 24, 48, 576)}
BACKBONES = sorted(STOCK)
# The registered ImageNet files (data/sources.yaml, imagenet_backbones).
IMAGENET_FILES = {
    "mobilenet_v3_large": (
        "mobilenet_v3_large-8738ca79.pth",
        "8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04",
    ),
    "mobilenet_v3_small": (
        "mobilenet_v3_small-047dcff4.pth",
        "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f",
    ),
}


def _scratch(backbone: str = "mobilenet_v3_small", inner: int = 16, **kw) -> DetModelConfig:
    return DetModelConfig(backbone=backbone, init="scratch", inner_channels=inner, **kw)


def _model(backbone: str = "mobilenet_v3_small", inner: int = 16, seed: int = 0) -> DBNet:
    torch.manual_seed(seed)
    return build_detector(_scratch(backbone, inner, channels_last=False), Path("unused")).eval()


def _images(n: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, h, w), generator=g, dtype=torch.uint8)


def _fake_weights(tmp_path: Path, backbone: str) -> tuple[Path, str, dict[str, torch.Tensor]]:
    """A torchvision-format checkpoint (features and classifier) with random weights."""
    torch.manual_seed(7)
    state = STOCK[backbone](weights=None).state_dict()
    path = tmp_path / f"{backbone}.pth"
    torch.save(state, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return path, sha, state


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("hw", [(64, 64), (96, 160)])
def test_output_shapes_and_taps(backbone, hw):
    model = _model(backbone)
    h, w = hw
    x = _images(2, h, w)
    trunk = model.backbone
    assert trunk.out_channels == TAP_CHANNELS[backbone]
    assert trunk.out_channels == tuple(
        trunk.features[i].out_channels for i in BACKBONE_TAPS[backbone]
    )
    with torch.no_grad():
        feats = trunk((x.float() - model.mean) / model.std)
        for f, c, stride in zip(feats, trunk.out_channels, (4, 8, 16, 32), strict=True):
            assert f.shape == (2, c, h // stride, w // stride)
        assert model.neck(feats).shape == (2, 16, h // 4, w // 4)
        prob = model(x)
        logits = model.forward_train(x)
    assert prob.shape == (2, 1, h, w) and prob.dtype == torch.float32
    assert logits.prob.shape == logits.thresh.shape == (2, 1, h, w)
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0


@pytest.mark.parametrize(
    ("backbone", "train_total", "inference"),
    [("mobilenet_v3_large", 3_210_466, 3_187_209), ("mobilenet_v3_small", 1_120_210, 1_096_953)],
)
def test_parameter_counts_are_pinned(backbone, train_total, inference):
    model = build_detector(_scratch(backbone, 96), Path("unused"))
    assert count_parameters(model) == train_total
    assert count_parameters(model, inference_only=True) == inference
    with pytest.raises(ValueError, match="thresh_head"):
        count_parameters(nn.Linear(2, 2), inference_only=True)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_se_swap_keeps_maths_and_state_dict_keys(backbone):
    torch.manual_seed(3)
    stock = STOCK[backbone](weights=None).features.eval()
    patched = MobileNetV3Features(backbone).eval()
    assert patched.features.state_dict().keys() == stock.state_dict().keys()
    patched.features.load_state_dict(stock.state_dict(), strict=True)
    assert not any(isinstance(m, nn.AdaptiveAvgPool2d) for m in patched.modules())
    assert any(isinstance(m, GlobalMean) for m in patched.modules())
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        ref = stock(x)
        out = patched(x)[-1]
    torch.testing.assert_close(out, ref, rtol=0, atol=1e-6)


def test_backbone_keeps_torchvision_batchnorm_settings():
    bns = [
        m
        for m in MobileNetV3Features("mobilenet_v3_small").modules()
        if isinstance(m, nn.BatchNorm2d)
    ]
    assert bns and all(m.eps == 0.001 and m.momentum == 0.01 for m in bns)


def test_forward_never_runs_the_threshold_head():
    model = _model()
    calls: list[str] = []
    model.thresh_head.register_forward_hook(lambda *_: calls.append("thresh"))
    model.prob_head.register_forward_hook(lambda *_: calls.append("prob"))
    with torch.no_grad():
        model(_images(1, 64, 64))
    assert calls == ["prob"]
    with torch.no_grad():
        model.forward_train(_images(1, 64, 64))
    assert calls == ["prob", "prob", "thresh"]


def test_forward_equals_sigmoid_of_forward_train_bit_for_bit():
    model = _model("mobilenet_v3_large")
    x = _images(2, 96, 160)
    with torch.no_grad():
        prob = model(x)
        logits = model.forward_train(x)
    assert torch.equal(prob, torch.sigmoid(logits.prob))


def test_uint8_and_float_inputs_agree():
    model = _model()
    x = _images(2, 64, 96)
    with torch.no_grad():
        assert torch.equal(model(x), model(x.float()))


def test_normalisation_constants_stay_out_of_the_state_dict():
    model = _model()
    assert not any(k in ("mean", "std") for k in model.state_dict())
    # PAD_RGB normalises to about zero, so padding looks like an average pixel.
    pad = torch.tensor(PAD_RGB, dtype=torch.float32).view(1, 3, 1, 1)
    assert float(((pad - model.mean) / model.std).abs().max()) < 0.01


def test_a_page_alone_equals_its_slice_of_a_batch():
    model = _model("mobilenet_v3_large")
    x = _images(3, 64, 96, seed=5)
    with torch.no_grad():
        batch = model(x)
        alone = model(x[1:2])
    torch.testing.assert_close(alone, batch[1:2], rtol=0, atol=1e-5)


def test_final_head_layer_and_logits_are_fp32_under_bf16_autocast():
    model = _model().train()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = model.forward_train(_images(2, 64, 64))
    assert logits.prob.dtype == torch.float32 and logits.thresh.dtype == torch.float32
    assert torch.isfinite(logits.prob).all() and torch.isfinite(logits.thresh).all()


def test_channels_last_build_matches_the_contiguous_model():
    torch.manual_seed(0)
    cl = build_detector(_scratch(channels_last=True), Path("unused")).eval()
    conv = cl.neck.smooth[0].weight
    assert conv.is_contiguous(memory_format=torch.channels_last)
    plain = _model(seed=0)
    plain.load_state_dict(cl.state_dict())
    x = _images(1, 64, 96)
    with torch.no_grad():
        torch.testing.assert_close(cl(x.contiguous(memory_format=torch.channels_last)), plain(x))


# --- weights --------------------------------------------------------------------------------------


def test_fake_weights_with_the_right_sha_load_strictly(tmp_path):
    path, sha, state = _fake_weights(tmp_path, "mobilenet_v3_small")
    features = load_imagenet_state_dict(path, sha)
    assert features.keys() == {k[len("features.") :] for k in state if k.startswith("features.")}
    trunk = MobileNetV3Features("mobilenet_v3_small", features)
    for k, v in trunk.features.state_dict().items():
        assert torch.equal(v, state[f"features.{k}"]), k

    cfg = DetModelConfig(
        backbone="mobilenet_v3_small",
        weights=path.name,
        weights_sha256=sha,
        inner_channels=16,
    )
    model = build_detector(cfg, tmp_path)
    w = model.backbone.features[0][0].weight
    assert torch.equal(w.contiguous(), state["features.0.0.weight"])


def test_wrong_sha_missing_file_and_unpinned_imagenet_are_refused(tmp_path):
    path, sha, _ = _fake_weights(tmp_path, "mobilenet_v3_small")
    with pytest.raises(WeightsError, match="SHA-256"):
        load_imagenet_state_dict(path, "0" * 64)
    with pytest.raises(WeightsError, match=r"download_datasets.py --dataset imagenet_backbones"):
        load_imagenet_state_dict(tmp_path / "absent.pth", sha)
    for weights, digest in [(None, None), (path.name, None), (None, sha)]:
        cfg = DetModelConfig(backbone="mobilenet_v3_small", weights=weights, weights_sha256=digest)
        with pytest.raises(WeightsError, match=r"needs det.model.weights"):
            build_detector(cfg, tmp_path)


def test_weights_of_the_other_variant_are_refused(tmp_path):
    path, sha, _ = _fake_weights(tmp_path, "mobilenet_v3_small")
    with pytest.raises(WeightsError, match="do not fit mobilenet_v3_large"):
        MobileNetV3Features("mobilenet_v3_large", load_imagenet_state_dict(path, sha))


def test_a_file_without_features_is_refused(tmp_path):
    path = tmp_path / "other.pth"
    torch.save({"fc.weight": torch.zeros(2, 2)}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(WeightsError, match="features"):
        load_imagenet_state_dict(path, sha)


def test_no_build_path_downloads(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a weight download was attempted")

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", refuse)
    monkeypatch.setattr(tv_api, "load_state_dict_from_url", refuse)
    path, sha, _ = _fake_weights(tmp_path, "mobilenet_v3_small")
    imagenet = DetModelConfig(
        backbone="mobilenet_v3_small", weights=path.name, weights_sha256=sha, inner_channels=16
    )
    build_detector(imagenet, tmp_path)
    build_detector(imagenet, tmp_path, load_pretrained=False)
    build_detector(DetModelConfig(inner_channels=16), tmp_path, load_pretrained=False)
    for backbone in BACKBONES:
        build_detector(_scratch(backbone), tmp_path)


@pytest.mark.parametrize("backbone", BACKBONES)
def test_cached_imagenet_files_pass_the_hash_check_and_load(backbone):
    # The torch hub cache holds the same files as data/raw/imagenet_backbones; CI has neither.
    filename, sha = IMAGENET_FILES[backbone]
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    if not (cache / filename).is_file():
        pytest.skip(f"{filename} is not in the torch hub cache")
    features = load_imagenet_state_dict(cache / filename, sha)
    trunk = MobileNetV3Features(backbone, features)  # strict
    assert torch.equal(trunk.features[0][0].weight, features["0.0.weight"])
    cfg = DetModelConfig(backbone=backbone, weights=filename, weights_sha256=sha, inner_channels=16)
    model = build_detector(cfg, cache)
    assert torch.equal(model.backbone.features[0][0].weight.contiguous(), features["0.0.weight"])
    with pytest.raises(WeightsError, match="SHA-256"):
        load_imagenet_state_dict(cache / filename, sha[::-1])


# --- preprocessing and runtime --------------------------------------------------------------------


def test_scaled_size_enlarges_and_shrinks_to_the_long_side():
    assert scaled_size(1000, 700, 1280) == (1280, 896)
    assert scaled_size(2000, 1500, 1280) == (1280, 960)
    assert scaled_size(700, 1000, 1280) == (896, 1280)
    assert scaled_size(1280, 3, 160) == (160, 1)  # never 0
    with pytest.raises(ValueError):
        scaled_size(0, 10, 1280)


def test_prepare_page_pads_with_pad_rgb_to_multiples_of_32():
    rng = np.random.default_rng(0)
    page = rng.integers(0, 256, (700, 1000, 3), dtype=np.uint8)
    out, info = prepare_page(page, 1280)
    assert info == ResizeInfo(1000, 700, 1280, 896, 1280, 896)
    assert out.shape == (896, 1280, 3) and out.dtype == np.uint8

    out, info = prepare_page(page, 160)
    assert (info.w, info.h, info.pad_w, info.pad_h) == (160, 112, 160, 128)
    assert info.sx == 160 / 1000 and info.sy == 112 / 700
    assert (out[info.h :] == PAD_RGB).all()

    same, info = prepare_page(page[:, :960], 960)
    assert (info.w, info.h) == (960, 700) and np.array_equal(same[:700], page[:, :960])
    assert (same[700:] == PAD_RGB).all()
    with pytest.raises(ValueError):
        prepare_page(page[..., 0], 1280)


def test_prepare_page_uses_area_when_shrinking_and_linear_when_enlarging():
    import cv2

    rng = np.random.default_rng(1)
    page = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    small, info = prepare_page(page, 160)
    ref = cv2.resize(page, (info.w, info.h), interpolation=cv2.INTER_AREA)
    assert np.array_equal(small[: info.h, : info.w], ref)
    big, info = prepare_page(page, 640)
    ref = cv2.resize(page, (info.w, info.h), interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(big[: info.h, : info.w], ref)


def test_to_page_coords_inverts_the_scale():
    info = ResizeInfo(1000, 700, 1280, 896, 1280, 896)
    pts = np.array([[[0.0, 0.0], [1280.0, 896.0], [640.0, 448.0]]])
    back = to_page_coords(pts, info)
    np.testing.assert_allclose(back, [[[0, 0], [1000, 700], [500, 350]]], atol=1e-9)
    assert pts[0, 1, 0] == 1280.0  # input untouched


def test_torch_prob_map_matches_the_model_and_checks_its_input():
    model = _model()
    backend = TorchProbMap(model, torch.device("cpu"))
    image = np.random.default_rng(2).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    prob = backend(image)
    assert prob.shape == (64, 96) and prob.dtype == np.float32
    with torch.no_grad():
        ref = model(torch.from_numpy(image).permute(2, 0, 1)[None].contiguous())
    np.testing.assert_allclose(prob, ref[0, 0].numpy(), rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="multiple of 32"):
        backend(image[:60])
    with pytest.raises(ValueError, match="uint8"):
        backend(image.astype(np.float32))
    model.train()
    with pytest.raises(RuntimeError, match="eval mode"):
        backend(image)


def test_strict_fp32_sets_ieee_and_restores():
    conv, matmul = torch.backends.cudnn.conv, torch.backends.cuda.matmul
    before = conv.fp32_precision, matmul.fp32_precision
    with strict_fp32():
        assert conv.fp32_precision == "ieee" and matmul.fp32_precision == "ieee"
    assert (conv.fp32_precision, matmul.fp32_precision) == before
    with pytest.raises(KeyError), strict_fp32():
        raise KeyError("boom")
    assert (conv.fp32_precision, matmul.fp32_precision) == before


def test_package_reexports_load_lazily_and_preprocess_stays_torch_free(repo_root):
    import glypharis.models.det as det

    assert det.DBNet is DBNet and det.build_detector is build_detector
    assert det.prepare_page is prepare_page
    with pytest.raises(AttributeError):
        _ = det.NotAnExport
    # Importing a torch-free submodule runs the package __init__ first; it must not pull in torch.
    code = "import sys, glypharis.models.det.preprocess; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "False"
