from __future__ import annotations

import pytest
import torch

from glypharis.config.loader import build_config
from glypharis.config.schema import ConfigError, ModelConfig, SvtrConfig
from glypharis.models.rec.crnn import CRNN, build_recognizer, count_parameters
from glypharis.models.rec.svtr import SVTR

SMALL = SvtrConfig(dims=[16, 32, 48], local_blocks=[1, 1, 0], global_blocks=[0, 1, 1],
                   heads=[2, 4, 4])  # fmt: skip


@pytest.mark.parametrize("height", [32, 48])
def test_svtr_gives_a_frame_per_four_columns(height):
    m = SVTR(30, ModelConfig(svtr=SMALL)).eval()
    out = m(torch.randn(3, 1, height, 128), torch.tensor([128, 96, 40]))
    assert out.shape == (32, 3, 30)
    assert m.frames(torch.tensor([128, 96, 41])).tolist() == [32, 24, 10]


def test_default_svtr_is_about_as_large_as_the_default_crnn():
    svtr = count_parameters(SVTR(275, ModelConfig(svtr=SvtrConfig())))
    crnn = count_parameters(CRNN(275, ModelConfig()))
    assert 0.85 < svtr / crnn < 1.15


def test_an_svtr_lines_output_does_not_depend_on_the_rest_of_its_batch():
    torch.manual_seed(0)
    m = SVTR(30, ModelConfig(svtr=SMALL)).eval()
    short = torch.randn(1, 1, 32, 160)
    with torch.no_grad():
        alone = m(short, torch.tensor([160]))[:, 0]
        batch = torch.zeros(2, 1, 32, 640)
        batch[0, :, :, :160] = short[0]
        batch[1] = torch.randn(1, 32, 640)
        together = m(batch, torch.tensor([160, 640]))[: alone.shape[0], 0]
    # Away from the right edge (the stem sees a few padded columns there) the frames must match:
    # attention never looks past the line's end.
    assert torch.allclose(alone[:36], together[:36], atol=1e-4)


def test_what_lies_beyond_a_lines_width_never_reaches_its_frames():
    torch.manual_seed(1)
    m = SVTR(30, ModelConfig(svtr=SMALL)).eval()
    x = torch.randn(1, 1, 32, 256)
    noisy = x.clone()
    noisy[..., 192:] = torch.randn(1, 1, 32, 64) * 5
    with torch.no_grad():
        a, b = m(x, torch.tensor([160])), m(noisy, torch.tensor([160]))
    assert torch.allclose(a[:36], b[:36], atol=1e-5)


def test_build_recognizer_picks_the_encoder_by_name():
    assert isinstance(build_recognizer(30, ModelConfig(svtr=SMALL), "svtr"), SVTR)
    assert isinstance(build_recognizer(30, ModelConfig()), CRNN)
    with pytest.raises(ConfigError, match="crnn or svtr"):
        build_recognizer(30, ModelConfig(), "resnet")


def test_svtr_config_rules():
    ok = {"task": "rec", "model_name": "svtr", "model": {"svtr": {"dims": [16, 32, 48]}}}
    with pytest.raises(ConfigError, match="multiples of their heads"):
        build_config(ok)  # default heads 3, 6, 8 do not divide 16, 32, 48
    ok["model"]["svtr"]["heads"] = [2, 4, 4]
    assert build_config(ok).model.svtr.dims == [16, 32, 48]
    with pytest.raises(ConfigError, match="exactly when model_name is svtr"):
        build_config({"task": "rec", "model_name": "svtr"})
    with pytest.raises(ConfigError, match="exactly when model_name is svtr"):
        build_config({"task": "rec", "model_name": "crnn", "model": {"svtr": {}}})
    with pytest.raises(ConfigError, match="frame_stride must be 4 for svtr"):
        build_config({**ok, "model": {**ok["model"], "frame_stride": 8}})


def test_crnn_at_stride_8_gives_a_frame_per_eight_columns():
    m = CRNN(30, ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=8,
                             frame_stride=8)).eval()  # fmt: skip
    out = m(torch.randn(2, 1, 32, 128), torch.tensor([128, 70]))
    assert out.shape == (16, 2, 30) and m.frames(torch.tensor([128, 70])).tolist() == [16, 8]
