"""Phase 14 frozen-model builder: settings from the record, hash and charset refusals, strict fp32,
and the pinned-code check (tiny models and a throwaway git repository; no real data)."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from det_fixtures import rect_polygon
from PIL import Image, ImageDraw

from glypharis.config.schema import (
    DetInferConfig,
    DetLabelConfig,
    DetModelConfig,
    DetOperatingPoint,
    DetPostConfig,
    ModelConfig,
)
from glypharis.data.det_labels import LABEL_VERSION
from glypharis.data.text import Charset
from glypharis.models.det.dbnet import build_detector
from glypharis.models.det.detector import DetectorSpec
from glypharis.models.rec.crnn import build_recognizer
from glypharis.pipeline import frozen, recognizer
from glypharis.pipeline.frozen import (
    FREEZE_RECORD,
    FrozenModelError,
    load_frozen,
    pinned_code_check,
    strict_inference,
)
from glypharis.pipeline.recognizer import TextRecognizer
from glypharis.training.checkpoint import save_checkpoint

REPO = Path(__file__).resolve().parents[1]
CHARSET = Path("data/manifests/charset.json")
REC_CKPT = "checkpoints/GW-REC-034/best.pt"
DET_CKPT = "checkpoints/GW-DET-001/best.pt"
REC_CFG = ModelConfig(cnn_channels=[8, 8, 16, 16, 16, 16], lstm_hidden=16, lstm_layers=1,
                      dropout=0.0)  # fmt: skip
DET_MODEL = DetModelConfig(backbone="mobilenet_v3_small", init="scratch", inner_channels=16)
RECORD_POST = {
    "bin_thresh": 0.3, "box_thresh": 0.5, "min_component_area": 1, "min_box_side": 3.0,
    "max_candidates": 5000, "box_type": "quad",
    "operating_points": [{"name": "inverse09", "unclip_mode": "inverse", "unclip_ratio": 0.9}],
}  # fmt: skip


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def restore_numerics(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """strict_inference changes process-wide switches; put them back after every test."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    saved = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    yield
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = saved[:2]
    torch.use_deterministic_algorithms(saved[2], warn_only=saved[3])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[4:6]
    torch.set_float32_matmul_precision(saved[6])


def write_rec_checkpoint(path: Path, characters: list[str], **config: Any) -> None:
    torch.manual_seed(0)
    model = build_recognizer(len(characters) + 1, REC_CFG, "crnn")
    cfg = {"experiment_id": "GW-REC-034", "model_name": "crnn",
           "model": dataclasses.asdict(REC_CFG),
           "data": {"image_height": 32, "max_width": 512, "charset": CHARSET.as_posix()},
           **config}  # fmt: skip
    save_checkpoint(path, {"model": model.state_dict(), "config": cfg, "config_hash": "r" * 64,
                           "git": {"commit": "rec0"}, "charset": characters})  # fmt: skip


def write_det_checkpoint(path: Path) -> None:
    torch.manual_seed(0)
    model = build_detector(DET_MODEL, Path("."), load_pretrained=False)
    # The stored spec differs from the record on purpose: the record must win.
    post = DetPostConfig(box_thresh=0.6, operating_points=[DetOperatingPoint("inverse10",
                         "inverse", 1.0)])  # fmt: skip
    spec = DetectorSpec(DET_MODEL, DetLabelConfig(), DetInferConfig(long_side=320), post,
                        "inverse10", LABEL_VERSION)  # fmt: skip
    save_checkpoint(path, {"model": model.state_dict(), "config": {"experiment_id": "GW-DET-001"},
                           "config_hash": "d" * 64, "git": {"commit": "det0"},
                           "selected_op": "inverse10", "detector_spec": spec.to_dict(),
                           "label_version": LABEL_VERSION})  # fmt: skip


def write_record(root: Path, **changes: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": "glypharis_v1",
        "recognizer": {
            "experiment": "GW-REC-034", "config_hash": "r" * 64, "git_commit": "rec0",
            "checkpoint": REC_CKPT, "checkpoint_sha256": sha(root / REC_CKPT),
            "model_name": "crnn", "input": {"height": 32, "max_width": 512},
            "charset": CHARSET.as_posix(), "temperature": 2.35,
        },
        "detector": {
            "experiment": "GW-DET-001", "config_hash": "d" * 64, "git_commit": "det0",
            "checkpoint": DET_CKPT, "checkpoint_sha256": sha(root / DET_CKPT),
            "label_version": LABEL_VERSION, "infer": {"long_side": 160}, "post": RECORD_POST,
            "operating_point": "inverse09",
        },
    }  # fmt: skip
    for section, values in changes.items():
        record[section] = {**record[section], **values}
    path = root / FREEZE_RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    return record


@pytest.fixture
def frozen_root(tmp_path: Path) -> Path:
    """A repository layout with the record, the charset and two tiny checkpoints."""
    (tmp_path / CHARSET).parent.mkdir(parents=True)
    shutil.copyfile(REPO / CHARSET, tmp_path / CHARSET)
    write_rec_checkpoint(tmp_path / REC_CKPT, Charset.load(REPO / CHARSET).characters)
    write_det_checkpoint(tmp_path / DET_CKPT)
    write_record(tmp_path)
    return tmp_path


def _page() -> Image.Image:
    img = Image.new("RGB", (400, 300), (235, 235, 225))
    ImageDraw.Draw(img).polygon([tuple(p) for p in rect_polygon(200, 120, 300, 24)], fill=(30,) * 3)
    return img


def test_every_setting_comes_from_the_record(frozen_root):
    torch.backends.cudnn.benchmark = True
    fz = load_frozen("cpu", root=frozen_root)
    assert fz.recognizer is fz.ocr.recognizer
    spec = fz.ocr.detector.spec
    assert spec.infer.long_side == 160 and spec.operating_point == "inverse09"
    assert dataclasses.asdict(spec.post) == RECORD_POST
    assert spec.model == DET_MODEL  # the network shape is the checkpoint's
    rec = fz.recognizer
    assert (rec.height, rec.max_width, rec.temperature, rec.strict_fp32) == (32, 512, 2.35, True)
    assert fz.settings()["detector"]["post"] == RECORD_POST
    assert fz.settings()["recognizer"] == {"input": {"height": 32, "max_width": 512},
                                           "temperature": 2.35, "strict_fp32": True}  # fmt: skip
    p = fz.provenance
    assert p["record"] == "configs/final/glypharis_v1.yaml"
    assert p["record_sha256"] == sha(frozen_root / FREEZE_RECORD)
    assert p["recognizer_checkpoint"] == REC_CKPT and p["detector_checkpoint"] == DET_CKPT
    assert p["recognizer_checkpoint_sha256"] == sha(frozen_root / REC_CKPT)
    assert p["detector_checkpoint_sha256"] == sha(frozen_root / DET_CKPT)
    assert p["charset_sha256"] == frozen.CHARSET_SHA256 == sha(REPO / CHARSET)
    assert p["torch"] == torch.__version__ and p["device"] == "cpu"
    assert {"git_commit", "git_dirty", "cuda"} <= set(p)
    assert p["numerics"]["cudnn_benchmark"] is False and torch.backends.cudnn.benchmark is False
    assert p["numerics"]["cudnn_allow_tf32"] is False
    assert p["numerics"]["deterministic_algorithms"] is True
    text = json.dumps(p)
    assert str(frozen_root) not in text and frozen_root.as_posix() not in text  # repo-relative
    page = fz.ocr.predict(_page(), temperatures=(1.0,))
    assert all(isinstance(ln.text, str) and 1.0 in ln.conf_at for ln in page.lines)


@pytest.mark.parametrize("section", ["recognizer", "detector"])
def test_a_checkpoint_hash_mismatch_is_refused(frozen_root, section):
    write_record(frozen_root, **{section: {"checkpoint_sha256": "0" * 64}})
    with pytest.raises(FrozenModelError, match="SHA-256"):
        load_frozen("cpu", root=frozen_root)


def test_a_changed_checkpoint_file_is_refused(frozen_root):
    with (frozen_root / DET_CKPT).open("ab") as fh:
        fh.write(b"\0")
    with pytest.raises(FrozenModelError, match=r"detector checkpoint .* SHA-256"):
        load_frozen("cpu", root=frozen_root)


def test_a_charset_difference_is_refused(frozen_root):
    chars = Charset.load(REPO / CHARSET).characters
    write_rec_checkpoint(frozen_root / REC_CKPT, [chars[1], chars[0], *chars[2:]])
    write_record(frozen_root)  # the record matches the new file, so only the charset differs
    with pytest.raises(FrozenModelError, match="character list"):
        load_frozen("cpu", root=frozen_root)


def test_a_charset_file_other_than_the_plans_is_refused(frozen_root):
    path = frozen_root / CHARSET
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(FrozenModelError, match="Phase 14 plan recorded"):
        load_frozen("cpu", root=frozen_root)
    # Without the file-hash pin the character lists still agree, so it loads.
    assert load_frozen("cpu", root=frozen_root, charset_sha256=None).provenance["charset_sha256"]


@pytest.mark.parametrize(
    ("section", "change", "message"),
    [
        ("recognizer", {"experiment": "GW-REC-035"}, "experiment"),
        ("recognizer", {"config_hash": "x" * 64}, "config hash"),
        ("recognizer", {"input": {"height": 48, "max_width": 512}}, "input height"),
        ("recognizer", {"model_name": "svtr"}, "recognizer model"),
        ("recognizer", {"checkpoint": "/abs/best.pt"}, "repo-relative"),
        ("detector", {"git_commit": "other"}, "git commit"),
        ("detector", {"label_version": LABEL_VERSION + 1}, "label version"),
        ("detector", {"checkpoint": "../elsewhere/best.pt"}, "repo-relative"),
        ("detector", {"post": {k: v for k, v in RECORD_POST.items() if k != "box_thresh"}},
         "post-processing has keys"),
    ],
)  # fmt: skip
def test_any_record_difference_is_refused(frozen_root, section, change, message):
    torch.backends.cudnn.benchmark = True
    write_record(frozen_root, **{section: change})
    with pytest.raises(FrozenModelError, match=message):
        load_frozen("cpu", root=frozen_root)
    assert torch.backends.cudnn.benchmark is True  # a refusal leaves the process as it was


def test_strict_inference_sets_and_reports_the_flags():
    torch.backends.cudnn.benchmark = True
    flags = strict_inference("cpu")
    assert torch.backends.cudnn.benchmark is False and torch.backends.cudnn.deterministic
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"
    assert flags["cuda_matmul_allow_tf32"] is False and flags["cudnn_allow_tf32"] is False
    assert flags["float32_matmul_precision"] == "highest"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None  # only set for CUDA runs


def test_recognizer_strict_fp32_is_opt_in(monkeypatch):
    entered: list[int] = []

    @contextlib.contextmanager
    def recording() -> Iterator[None]:
        entered.append(1)
        yield

    monkeypatch.setattr(recognizer, "ieee_fp32", recording)
    torch.manual_seed(0)
    charset = Charset.load(REPO / CHARSET)
    model = build_recognizer(charset.num_classes, REC_CFG, "crnn")
    rec = TextRecognizer(model, charset, 32, 512)
    assert rec.strict_fp32 is False
    crops = [Image.fromarray(np.random.default_rng(0).integers(0, 255, (40, 90), np.uint8))]
    plain = rec.logits(crops)
    assert entered == []
    rec.strict_fp32 = True
    strict = rec.logits(crops)
    assert entered == [1]
    assert torch.equal(plain[0], strict[0])  # on the CPU the setting changes nothing


@pytest.mark.gpu
def test_recognizer_strict_fp32_turns_tf32_off_on_cuda():
    torch.manual_seed(0)
    charset = Charset.load(REPO / CHARSET)
    model = build_recognizer(charset.num_classes, REC_CFG, "crnn")
    rec = TextRecognizer(model, charset, 32, 512, "cuda", strict_fp32=True)
    seen: list[tuple[str, str]] = []
    forward = rec.model.forward
    cudnn = torch.backends.cudnn
    before = cudnn.rnn.fp32_precision  # type: ignore[attr-defined]

    def spy(*args: Any) -> Any:
        seen.append((cudnn.conv.fp32_precision, cudnn.rnn.fp32_precision))  # type: ignore[attr-defined]
        return forward(*args)

    rec.model.forward = spy  # type: ignore[method-assign]
    crops = [Image.fromarray(np.random.default_rng(0).integers(0, 255, (40, 90), np.uint8))]
    rec.logits(crops)
    assert seen == [("ieee", "ieee")]  # the BiLSTM too, not only the convolutions
    assert cudnn.rnn.fp32_precision == before  # type: ignore[attr-defined]  # restored


@pytest.mark.skipif(
    not all((REPO / p).is_file() for p in (REC_CKPT, DET_CKPT)),
    reason="the frozen checkpoints are not in this checkout",
)
def test_the_real_freeze_record_builds():
    """The committed record against the real checkpoints (local only: they are not committed)."""
    fz = load_frozen("cpu", root=REPO)
    assert fz.provenance["recognizer_checkpoint_sha256"].startswith("c7e74540")
    assert fz.ocr.detector.spec.operating_point == "inverse09"
    assert fz.recognizer.temperature == 2.35


# ---------------------------------------------------------------------- pinned code


def git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repository, isolated from the user's git configuration."""
    empty = tmp_path / "gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for who in ("AUTHOR", "COMMITTER"):
        monkeypatch.setenv(f"GIT_{who}_NAME", "test")
        monkeypatch.setenv(f"GIT_{who}_EMAIL", "test@example.invalid")
    root = tmp_path / "repo"
    for rel in (
        "glypharis/a.py", "scripts/run.py", "configs/x.yaml", "research/test_access_log.md",
        "research/experiments.md", "pyproject.toml", "data/manifests/processed/eval_crops.json",
    ):  # fmt: skip
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("v1\n", encoding="utf-8")
    git(root.parent, "init", "-q", "repo")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "code")
    return root


def test_pinned_code_check_passes_at_the_tag_and_allows_records(repo):
    git(repo, "tag", "phase14-eval")
    tag_commit = git(repo, "rev-parse", "HEAD")
    assert pinned_code_check(root=repo) == {
        "tag": "phase14-eval", "tag_commit": tag_commit, "head": tag_commit, "trees": "equal",
    }  # fmt: skip
    (repo / "research/test_access_log.md").write_text("v2\n", encoding="utf-8")
    git(repo, "commit", "-q", "-am", "log a row")
    (repo / "research/test_access_log.md").write_text("v3\n", encoding="utf-8")  # uncommitted
    (repo / "data/manifests/processed/phase14_outputs.json").write_text("{}", encoding="utf-8")
    (repo / "reports").mkdir()
    (repo / "reports/new.json").write_text("{}", encoding="utf-8")
    (repo / "research/failures.md").write_text("a void sitting\n", encoding="utf-8")
    check = pinned_code_check(root=repo)
    assert check["head"] != tag_commit and check["trees"] == "equal"
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "record a void sitting")
    assert pinned_code_check(root=repo)["trees"] == "equal"
    assert "research/failures.md" in frozen.RUN_RECORDS


@pytest.mark.parametrize("renames", ["true", "copies"])
def test_a_pinned_file_renamed_into_a_record_path_is_refused(repo, renames):
    """Rename detection would list only the allowed destination; the source must be named too."""
    git(repo, "config", "diff.renames", renames)
    git(repo, "config", "status.renames", renames)
    git(repo, "tag", "phase14-eval")
    (repo / "reports").mkdir()
    git(repo, "mv", "glypharis/a.py", "reports/a.py")  # staged
    with pytest.raises(RuntimeError, match="uncommitted") as staged:
        pinned_code_check(root=repo)
    assert "glypharis/a.py" in str(staged.value)
    git(repo, "commit", "-q", "-m", "move a pinned file")
    assert git(repo, "diff", "--name-only", "phase14-eval", "HEAD") == "reports/a.py"
    with pytest.raises(RuntimeError, match="outside the run records") as committed:
        pinned_code_check(root=repo)
    assert "glypharis/a.py" in str(committed.value)


def test_a_pinned_file_renamed_and_edited_into_reports_is_refused(repo):
    git(repo, "tag", "phase14-eval")
    (repo / "reports").mkdir()
    git(repo, "mv", "configs/x.yaml", "reports/x.yaml")
    with (repo / "reports/x.yaml").open("a", encoding="utf-8") as f:
        f.write("extra\n")
    git(repo, "commit", "-q", "-am", "move and edit a config")
    with pytest.raises(RuntimeError, match="outside the run records") as committed:
        pinned_code_check(root=repo)
    assert "configs/x.yaml" in str(committed.value)


@pytest.mark.parametrize(
    "rel", ["pyproject.toml", "data/manifests/processed/eval_crops.json", "research/experiments.md"]
)
def test_pinned_code_check_refuses_changes_outside_the_run_records(repo, rel):
    git(repo, "tag", "phase14-eval")
    (repo / rel).write_text("v2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted"):
        pinned_code_check(root=repo)
    git(repo, "commit", "-q", "-am", "change")
    with pytest.raises(RuntimeError, match="outside the run records"):
        pinned_code_check(root=repo)
    git(repo, "tag", "later")
    (repo / "tests").mkdir()
    (repo / "tests/new.py").write_text("x\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted"):  # untracked, outside the records
        pinned_code_check("later", root=repo)


def test_pinned_code_check_refuses_code_changes(repo):
    with pytest.raises(RuntimeError, match="does not exist"):
        pinned_code_check(root=repo)
    git(repo, "tag", "phase14-eval")
    (repo / "glypharis/a.py").write_text("v2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted"):
        pinned_code_check(root=repo)
    git(repo, "checkout", "-q", "--", "glypharis/a.py")
    (repo / "scripts/new.py").write_text("x\n", encoding="utf-8")  # untracked code counts too
    with pytest.raises(RuntimeError, match="uncommitted"):
        pinned_code_check(root=repo)
    git(repo, "add", "scripts/new.py")
    git(repo, "commit", "-q", "-m", "more code")
    with pytest.raises(RuntimeError, match="differs from phase14-eval outside"):
        pinned_code_check(root=repo)
    git(repo, "tag", "later")
    assert pinned_code_check("later", root=repo)["trees"] == "equal"
