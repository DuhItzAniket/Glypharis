"""Phase 16, GW-EFF-006: a tiny SVTR exported to ONNX at widths other than the trace width, the v1
bundle built from a freeze record (`export_check.py --frozen`), and the reference writer and its
check (`phase16_reference.py`), on tiny made-up models and pages in tmp_path; no real data."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import export_check
import numpy as np
import phase16_reference as ref
import pytest
import torch
import yaml
from det_fixtures import make_synthetic_pages, write_processed_manifest
from PIL import Image
from test_frozen import (
    CHARSET,
    DET_CKPT,
    REC_CKPT,
    REPO,
    write_det_checkpoint,
    write_rec_checkpoint,
    write_record,
)
from test_pipeline import tiny_detector, tiny_recognizer

from glypharis.config.schema import ModelConfig, SvtrConfig
from glypharis.data.download import sha256_file
from glypharis.data.linepack import LinePackWriter
from glypharis.data.pages import write_pages
from glypharis.data.text import Charset
from glypharis.efficiency.onnx_build import SessionLineModel, load_bundle_onnx
from glypharis.efficiency.threads import THREAD_ENV, ort_session
from glypharis.export.bundle import load_bundle, read_manifest, write_bundle
from glypharis.export.onnx import export_recognizer_onnx
from glypharis.models.rec.svtr import SVTR
from glypharis.pipeline.frozen import FREEZE_RECORD, Frozen

pytest.importorskip("onnxruntime")

SMALL = SvtrConfig(dims=[16, 32, 48], local_blocks=[1, 1, 0], global_blocks=[0, 1, 1],
                   heads=[2, 4, 4])  # fmt: skip
V1_CODE = {"tag": "phase14-eval", "tag_commit": "t" * 40, "head": "h" * 40,
           "paths": ["glypharis/pipeline"], "v1_code": "equal"}  # fmt: skip
SOURCES = {"detector": {"checkpoint_sha256": "d" * 64},
           "recognizer": {"checkpoint_sha256": "r" * 64}}  # fmt: skip


# ------------------------------------------------------------------- the tiny SVTR in ONNX


@pytest.fixture(scope="module")
def svtr_onnx(tmp_path_factory: pytest.TempPathFactory) -> tuple[SVTR, Any]:
    torch.manual_seed(0)
    model = SVTR(30, ModelConfig(svtr=SMALL)).eval()
    path = export_recognizer_onnx(model, 32, tmp_path_factory.mktemp("svtr") / "rec.onnx")
    return model, SessionLineModel(ort_session(path, 2), model.stride, model.num_classes)


@pytest.mark.parametrize("width", [64, 96, 352, 512, 1024, 1536])
@pytest.mark.parametrize("batch", [1, 3])
def test_tiny_svtr_onnx_matches_pytorch_off_the_trace_width(svtr_onnx, width, batch):
    """Traced at batch 2 and width 256 (`export_recognizer_onnx`), the export must hold at any
    batch and width: the attention's token count and the padding mask follow the input."""
    model, onnx_model = svtr_onnx
    gen = torch.Generator().manual_seed(width + batch)
    x = torch.rand(batch, 1, 32, width, generator=gen) * 2 - 1
    widths = torch.tensor([width, width - 40, width // 3][:batch])
    with torch.no_grad():
        a = model(x, widths)
    b = onnx_model(x, widths)
    assert a.shape == b.shape == (width // 4, batch, 30)
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())
    assert torch.equal(onnx_model.frames(widths), model.frames(widths))


def test_tiny_svtr_onnx_at_the_v1_height(tmp_path):
    torch.manual_seed(1)
    model = SVTR(30, ModelConfig(svtr=SMALL)).eval()
    path = export_recognizer_onnx(model, 48, tmp_path / "rec48.onnx")
    onnx_model = SessionLineModel(ort_session(path, 2), model.stride, model.num_classes)
    x = torch.rand(2, 1, 48, 1536) * 2 - 1
    widths = torch.tensor([1536, 700])
    with torch.no_grad():
        a = model(x, widths)
    assert torch.allclose(a, onnx_model(x, widths), atol=1e-4)


# ------------------------------------------------------------- export_check's item helpers


def test_text_difference_judges_as_rec_check():
    charset = Charset(list("ab"))
    a = torch.full((4, 3), -5.0)
    a[:, 0] = 5.0  # blank everywhere
    a[1, 1] = 5.0004  # "a" by a hair at frame 1
    assert export_check.text_difference(a, a.clone(), charset) is None
    b = a.clone()
    b[1, 1] = 4.0  # blank wins at frame 1 in the other runtime
    near = export_check.text_difference(a, b, charset)
    assert near is not None and near["near_tie"] and near["frame"] == 1
    assert (near["torch"], near["onnx"]) == ("a", "")
    a[1, 1] = 7.0
    far = export_check.text_difference(a, b, charset)
    assert far is not None and not far["near_tie"] and far["gap"] == pytest.approx(2.0)
    wrong = export_check.text_difference(a, b[:3], charset)
    assert wrong is not None and not wrong["near_tie"] and wrong["shapes"] == [[4, 3], [3, 3]]


def test_line_indices_are_rec_checks_draw():
    want = np.sort(np.random.default_rng(20261028).choice(70000, 5000, replace=False))
    assert np.array_equal(export_check.line_indices(70000), want)
    assert export_check.line_indices(7).tolist() == list(range(7))


# ------------------------------------------------------------------------- made-up data


def _crop(rng: np.random.Generator, w: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 255, (32, w), np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def data(tmp_path: Path) -> Path:
    """Five validation pages at the Phase 5 HierText path and a seven-line validation pack."""
    synth = make_synthetic_pages(tmp_path, n_train=1, n_val=5)
    pages_file = synth.data_root / ref.PAGES
    write_pages(pages_file, synth.val)
    write_processed_manifest(synth.data_root, "detsynth", {"validation": pages_file})
    rng = np.random.default_rng(3)
    with LinePackWriter(synth.data_root / ref.PACK) as pack:
        for i, w in enumerate((40, 90, 160, 30, 220, 75, 120)):
            pack.add(f"l{i}", f"d{i}", "validation", "hiertext", "ab", _crop(rng, w), (w, 32))
    return synth.data_root


def _bundle(tmp_path: Path, name: str, rec_seed: int = 0, temperature: float = 1.0) -> Path:
    det, rec = tiny_detector(), tiny_recognizer(rec_seed)
    rec.temperature = temperature
    torch.save({"model": det.backend.model.state_dict()}, tmp_path / f"{name}_det.pt")
    return write_bundle(tmp_path / f"{name}_det.pt", det.spec, rec, tmp_path / name, "0.0.0-test",
                        SOURCES)  # fmt: skip


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return _bundle(tmp_path, "bundle")


# -------------------------------------------------------------------------- write and check


def test_references_are_written_once_hashed_and_pass_against_the_same_model(data, bundle, tmp_path):
    out = tmp_path / "ref"
    torch_ocr = load_bundle(bundle, "torch")
    core = ref.write_references(torch_ocr, data, ref.PACK, ref.PAGES, out)
    assert core["complete"] and core["limit"] is None
    assert core["detector"]["pages"] == 5 and core["recognizer"]["lines"] == 7
    assert core["end_to_end"]["pages"] == 5 and core["recognizer"]["indices"] == list(range(7))
    assert core["files"] == {n: sha256_file(out / n) for n in ref.FILES}
    with np.load(out / ref.MAPS) as maps:
        assert sorted(maps.files) == [f"{i:04d}" for i in range(5)]
        assert all(maps[k].dtype == np.float32 for k in maps.files)
        assert [list(maps[k].shape) for k in sorted(maps.files)] == core["detector"]["shapes"]
    texts = [json.loads(ln) for ln in (out / ref.LINE_TEXTS).read_text("utf-8").splitlines()]
    assert [t["key"] for t in texts] == [f"l{i}" for i in range(7)]
    with np.load(out / ref.LOGITS) as logits:
        classes = torch_ocr.recognizer.charset.num_classes
        assert logits["0002"].shape == (texts[2]["frames"], classes)

    manifest = {"experiment": ref.EXPERIMENT, **core}
    res = ref.check_references(load_bundle_onnx(bundle, 2), data, out, manifest)
    assert res["pass"], res["tolerances"]
    assert res["detector"]["prob_max_abs_diff"] <= 1e-4
    assert len(res["detector"]["per_page_max_abs_diff"]) == 5
    assert res["recognizer"]["identical_text"] == 7 and res["recognizer"]["logit_pass"]
    e = res["end_to_end"]
    assert e["page_cer_onnx"] == e["page_cer_torch"] and e["pages_with_identical_text"] == 5
    assert set(res["tolerances"]) == {"map_max_abs", "logits_max_abs", "text", "page_cer"}


def test_the_check_fails_a_bundle_with_other_recognizer_weights(data, bundle, tmp_path):
    out = tmp_path / "ref"
    core = ref.write_references(load_bundle(bundle, "torch"), data, ref.PACK, ref.PAGES, out)
    other = _bundle(tmp_path, "other", rec_seed=1)
    res = ref.check_references(load_bundle_onnx(other, 2), data, out,
                               {"experiment": ref.EXPERIMENT, **core})  # fmt: skip
    assert not res["pass"]
    assert not res["tolerances"]["logits_max_abs"]["pass"]
    assert res["tolerances"]["map_max_abs"]["pass"]  # the same detector


def test_limited_references_cover_the_first_items_only(data, bundle, tmp_path):
    out = tmp_path / "ref"
    core = ref.write_references(load_bundle(bundle, "torch"), data, ref.PACK, ref.PAGES, out, 2)
    assert not core["complete"] and core["limit"] == 2
    assert (core["detector"]["pages"], core["recognizer"]["lines"],
            core["end_to_end"]["pages"]) == (2, 2, 2)  # fmt: skip
    res = ref.check_references(load_bundle_onnx(bundle, 2), data, out,
                               {"experiment": ref.EXPERIMENT, **core})  # fmt: skip
    assert res["pass"] and res["recognizer"]["lines"] == 2


def test_test_split_pages_are_refused(data, bundle, tmp_path):
    pages_file = data / ref.PAGES
    synth = make_synthetic_pages(tmp_path / "t", n_train=1, n_val=5)
    for p in synth.val:
        p.split = "test"
    write_pages(pages_file, synth.val)
    with pytest.raises(SystemExit, match="validation"):
        ref.map_pages(data, None)


# ------------------------------------------------------------------------------ command line


@pytest.fixture
def cli(data, bundle, tmp_path, monkeypatch):
    """A repository root with a freeze record matching the tiny bundle, the frozen builder and the
    git checks stubbed, and the thread state put back afterwards."""
    root = tmp_path / "repo"
    spec = read_manifest(bundle)["detector"]["spec"]
    record = {
        "name": "glypharis_v1",
        "detector": {"infer": spec["infer"], "post": spec["post"],
                     "operating_point": spec["operating_point"],
                     "label_version": spec["label_version"], "checkpoint_sha256": "d" * 64},
        "recognizer": {"input": {"height": 32, "max_width": 512}, "temperature": 1.0,
                       "checkpoint_sha256": "r" * 64},
    }  # fmt: skip
    (root / FREEZE_RECORD).parent.mkdir(parents=True)
    (root / FREEZE_RECORD).write_text(yaml.safe_dump(record), encoding="utf-8")
    provenance = {"name": "glypharis_v1", "record_sha256": sha256_file(root / FREEZE_RECORD)}
    loads: list[Any] = []

    def fake_load(device: str, **kwargs: Any) -> Frozen:
        loads.append((device, kwargs))
        ocr = load_bundle(bundle, "torch")
        return Frozen(ocr, ocr.recognizer, dict(provenance))

    monkeypatch.setattr(ref, "load_frozen", fake_load)
    monkeypatch.setattr(ref, "assert_v1_code", lambda root: dict(V1_CODE))
    monkeypatch.setattr(ref, "record_sources", lambda root: {"pack": ref.PACK,
                                                             "pages_file": ref.PAGES})  # fmt: skip
    monkeypatch.setattr(ref, "git_state", lambda: {"git_commit": "h" * 40, "git_dirty": False})
    for key in THREAD_ENV:
        monkeypatch.delenv(key, raising=False)
    threads, grad = torch.get_num_threads(), torch.is_grad_enabled()
    n = torch.get_num_interop_threads()  # a count this process can always be set to
    common = ["--threads", str(n), "--root", str(root), "--data-root", str(data)]
    yield SimpleNamespace(root=root, bundle=bundle, tmp=tmp_path, n=n, common=common,
                          loads=loads, ref_dir=tmp_path / "refs",
                          report=tmp_path / "reports/reference_manifest.json")  # fmt: skip
    torch.set_num_threads(threads)
    torch.set_grad_enabled(grad)


def _write(cli: SimpleNamespace, *extra: str) -> int:
    argv = ["write", *cli.common, "--out", str(cli.ref_dir), "--report", str(cli.report), *extra]
    return ref.main(argv, require_fresh=False)


def _check(cli: SimpleNamespace, bundle: Path, out: Path) -> int:
    argv = ["check", *cli.common, "--bundle", str(bundle), "--reference", str(cli.ref_dir),
            "--manifest", str(cli.report), "--out", str(out)]  # fmt: skip
    return ref.main(argv, require_fresh=False)


def test_write_then_check_from_the_command_line(cli):
    assert _write(cli) == 0
    manifest = json.loads((cli.ref_dir / "manifest.json").read_text(encoding="utf-8"))
    assert json.loads(cli.report.read_text(encoding="utf-8")) == manifest
    assert manifest["experiment"] == "GW-EFF-006" and manifest["v1_code"] == V1_CODE
    assert manifest["threads"]["requested"] == cli.n and manifest["threads"]["intra_op"] == cli.n
    assert manifest["threads"]["env"] == {k: str(cli.n) for k in THREAD_ENV}
    assert manifest["threads_checked"]["after"]["intra_op"] == cli.n
    assert manifest["files"] == {n: sha256_file(cli.ref_dir / n) for n in ref.FILES}
    assert {"started_utc", "ended_utc", "seconds", "settings", "provenance", "versions",
            "git_commit"} <= set(manifest)  # fmt: skip
    assert cli.loads == [("cpu", {"root": cli.root})]
    with pytest.raises(SystemExit, match="written once"):
        _write(cli)

    out = cli.tmp / "reports/reference_check.json"
    assert _check(cli, cli.bundle, out) == 0
    verdict = json.loads(out.read_text(encoding="utf-8"))
    assert verdict["pass"] and verdict["complete"] and verdict["bundle_equals_record"]
    assert verdict["bundle_files"] == read_manifest(cli.bundle)["files"]
    assert verdict["bundle_manifest_sha256"] == sha256_file(cli.bundle / "manifest.json")
    assert verdict["reference_manifest_sha256"] == sha256_file(cli.ref_dir / "manifest.json")
    for part in ("detector", "recognizer"):
        assert verdict["session_options"][part]["intra_op_num_threads"] == cli.n
        assert verdict["session_options"][part]["execution_mode"] == "ORT_SEQUENTIAL"
    with pytest.raises(SystemExit, match="exists"):
        _check(cli, cli.bundle, out)


def test_the_check_refuses_a_bundle_that_is_not_v1s(cli):
    assert _write(cli) == 0
    warm = _bundle(cli.tmp, "warm", temperature=2.0)
    with pytest.raises(SystemExit, match="recognizer temperature"):
        _check(cli, warm, cli.tmp / "v.json")


def test_the_check_refuses_changed_references(cli):
    assert _write(cli) == 0
    with (cli.ref_dir / ref.LINE_TEXTS).open("a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(SystemExit, match="SHA-256"):
        _check(cli, cli.bundle, cli.tmp / "v.json")


def test_the_check_refuses_a_manifest_unlike_the_committed_copy(cli):
    assert _write(cli) == 0
    copy = json.loads(cli.report.read_text(encoding="utf-8"))
    copy["threads"]["requested"] = 99
    cli.report.write_text(json.dumps(copy), encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from"):
        _check(cli, cli.bundle, cli.tmp / "v.json")


def test_smoke_references_stay_out_of_the_registered_paths(cli, monkeypatch):
    with pytest.raises(SystemExit, match="outside the registered paths"):
        ref.main(["write", *cli.common, "--limit", "2"], require_fresh=False)
    # The registered folder spelt as an absolute path, or under --root, is the same folder.
    monkeypatch.chdir(cli.tmp)
    for out in ((cli.tmp / ref.REF_DIR).resolve(), cli.root / ref.REF_DIR):
        argv = ["write", *cli.common, "--limit", "2", "--out", str(out), "--report",
                str(cli.tmp / "elsewhere.json")]  # fmt: skip
        with pytest.raises(SystemExit, match="outside the registered paths"):
            ref.main(argv, require_fresh=False)
    assert not (cli.tmp / ref.REF_DIR).exists() and cli.loads == []
    assert _write(cli, "--limit", "2") == 0
    manifest = json.loads(cli.report.read_text(encoding="utf-8"))
    assert not manifest["complete"] and manifest["recognizer"]["lines"] == 2


def test_an_imported_main_needs_a_fresh_process_unless_told(cli):
    with pytest.raises(SystemExit, match="run this as a script"):
        ref.main(["write", *cli.common, "--out", str(cli.ref_dir), "--report", str(cli.report)])
    assert cli.loads == []


@pytest.mark.parametrize(
    ("argv", "want"),
    [([], 10), (["write", "--threads", "4"], 4), (["check", "--threads=3"], 3),
     (["write", "--threads", "0"], None), (["write", "--threads", "x"], None)],
)  # fmt: skip
def test_threads_are_read_from_the_command_line(argv, want):
    assert ref.requested_threads(argv) == want


def test_a_command_line_run_sets_the_variables_before_torch_loads(tmp_path):
    """Run as a script, the thread variables are in place before torch loads (torch then starts
    with that many intra-op threads untold); the run stops at once because its output exists."""
    if (os.cpu_count() or 1) < 2:
        pytest.skip("one core: one thread is torch's default anyway")
    out = tmp_path / "refs"
    out.mkdir()
    (out / "manifest.json").write_text("{}\n", encoding="utf-8")
    script = REPO / "scripts/phase16_reference.py"
    code = (
        "import os, runpy, sys\n"
        f"sys.argv = [{str(script)!r}, 'write', '--threads', '1', '--out', {str(out)!r}]\n"
        "try:\n"
        f"    runpy.run_path({str(script)!r}, run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    print('exit:', exc)\n"
        "import torch\n"
        f"print('env:', [os.environ.get(k) for k in {THREAD_ENV!r}])\n"
        "print('threads:', torch.get_num_threads(), torch.get_num_interop_threads())\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in THREAD_ENV}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "scripts")])
    proc = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=300)  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    assert "written once" in proc.stdout
    assert "env: ['1', '1', '1']" in proc.stdout
    assert "threads: 1 1" in proc.stdout


# --------------------------------------------------------------- export_check.py --frozen


@pytest.fixture
def frozen_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """test_frozen's repository layout: the record, the charset and two tiny checkpoints."""
    (tmp_path / CHARSET).parent.mkdir(parents=True)
    shutil.copyfile(REPO / CHARSET, tmp_path / CHARSET)
    write_rec_checkpoint(tmp_path / REC_CKPT, Charset.load(REPO / CHARSET).characters)
    write_det_checkpoint(tmp_path / DET_CKPT)
    write_record(tmp_path, recognizer={"temperature_from": "GW-E2E-005"},
                 detector={"settings_from": "Phase 9"})  # fmt: skip
    monkeypatch.setattr(export_check, "assert_v1_code", lambda root: dict(V1_CODE))
    grad = torch.is_grad_enabled()
    yield tmp_path
    torch.set_grad_enabled(grad)


def test_frozen_export_takes_everything_from_the_record(frozen_repo, tmp_path):
    out, report = tmp_path / "bundle", tmp_path / "v1_bundle.json"
    argv = ["--frozen", "--root", str(frozen_repo), "--out", str(out), "--report", str(report)]
    assert export_check.main(argv) == 0
    manifest = read_manifest(out)
    record = yaml.safe_load((frozen_repo / FREEZE_RECORD).read_text(encoding="utf-8"))
    assert export_check.bundle_problems(manifest, record) == []
    assert manifest["version"] == "glypharis_v1"
    spec = manifest["detector"]["spec"]
    assert spec["infer"] == {"long_side": 160}  # the record's, not the checkpoint's 320
    assert spec["operating_point"] == "inverse09"
    assert (manifest["recognizer"]["height"], manifest["recognizer"]["max_width"],
            manifest["recognizer"]["temperature"]) == (32, 512, 2.35)  # fmt: skip
    src = manifest["sources"]
    assert src["detector"]["checkpoint_sha256"] == sha256_file(frozen_repo / DET_CKPT)
    assert src["recognizer"]["checkpoint_sha256"] == sha256_file(frozen_repo / REC_CKPT)
    assert src["recognizer"]["temperature_from"] == "GW-E2E-005"
    assert src["detector"]["settings"] == "Phase 9"
    assert src["record_sha256"] == sha256_file(frozen_repo / FREEZE_RECORD)
    assert src["v1_code"] == V1_CODE and src["build"]["numerics"]["cudnn_allow_tf32"] is False
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["files"] == manifest["files"] and summary["equals_record"]
    assert summary["manifest_sha256"] == sha256_file(out / "manifest.json")
    assert summary["created_utc"].endswith("+00:00")
    ocr = load_bundle_onnx(out, 2)
    assert ocr.recognizer.temperature == 2.35 and ocr.model_version == "glypharis_v1"
    with pytest.raises(SystemExit, match="already holds a bundle"):
        export_check.main(argv)
    # The report names the registered bundle: another build never replaces it.
    before = report.read_bytes()
    other = ["--frozen", "--root", str(frozen_repo), "--out", str(tmp_path / "b2"), "--report",
             str(report)]  # fmt: skip
    with pytest.raises(SystemExit, match=r"v1_bundle.json exists"):
        export_check.main(other)
    assert report.read_bytes() == before and not (tmp_path / "b2").exists()


def test_frozen_export_rejects_a_bundle_unlike_the_record(frozen_repo, tmp_path, monkeypatch):
    real = export_check.frozen_sources

    def wrong(*args: Any) -> dict[str, Any]:
        sources = real(*args)
        sources["recognizer"]["checkpoint_sha256"] = "0" * 64
        return sources

    monkeypatch.setattr(export_check, "frozen_sources", wrong)
    out = tmp_path / "bundle"
    with pytest.raises(SystemExit, match="recognizer checkpoint SHA-256"):
        export_check.main(["--frozen", "--root", str(frozen_repo), "--out", str(out)])
    assert not (out / "manifest.json").exists() and (out / "manifest.rejected.json").is_file()


def test_frozen_export_needs_the_v1_code(frozen_repo, tmp_path, monkeypatch):
    from glypharis.efficiency.v1 import V1CodeError

    def changed(root: Any) -> dict[str, Any]:
        raise V1CodeError("the v1 code differs from phase14-eval")

    monkeypatch.setattr(export_check, "assert_v1_code", changed)
    with pytest.raises(SystemExit, match="tag's code"):
        export_check.main(["--frozen", "--root", str(frozen_repo), "--out", str(tmp_path / "b")])
    assert not (tmp_path / "b").exists()


def test_the_phase_17_command_line_is_unchanged(capsys):
    """Without --frozen the earlier parser runs: its required arguments are still required."""
    with pytest.raises(SystemExit) as exc:
        export_check.main(["--out", "x"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "--det-checkpoint" in err and "--rec-run" in err and "--bundle" in err
