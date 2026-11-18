"""The v1 bundle on the PyTorch backend against the PyTorch references (GW-EFF-012), on tiny
made-up models and pages; the references come from `phase16_reference write`."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import phase17_v1_check as v1check
import pytest
from test_phase16_reference import V1_CODE, _bundle, _write, bundle, cli, data  # noqa: F401

from glypharis.data.download import sha256_file
from glypharis.efficiency.threads import THREAD_ENV
from glypharis.export.bundle import read_manifest

pytest.importorskip("onnxruntime")


@pytest.fixture
def check(cli, monkeypatch) -> SimpleNamespace:  # noqa: F811
    """`cli` with the references written, and this script's git checks stubbed."""
    assert _write(cli) == 0
    monkeypatch.setattr(v1check, "assert_v1_code", lambda root: dict(V1_CODE))
    monkeypatch.setattr(v1check, "git_state", lambda: {"git_commit": "h" * 40, "git_dirty": False})
    cli.sha = sha256_file(cli.bundle / "manifest.json")
    return cli


def _run(c: SimpleNamespace, bundle: Path, out: Path, sha: str | None = None) -> int:  # noqa: F811
    argv = [*c.common, "--bundle", str(bundle), "--reference", str(c.ref_dir), "--manifest",
            str(c.report), "--expect-manifest-sha256", sha or c.sha, "--out", str(out)]  # fmt: skip
    return v1check.main(argv, require_fresh=False)


def test_the_same_model_on_the_torch_backend_passes_and_is_labelled(check):
    out = check.tmp / "reports/export/torch_bundle_check.json"
    assert _run(check, check.bundle, out) == 0
    v = json.loads(out.read_text(encoding="utf-8"))
    assert v["experiment"] == "GW-EFF-012" and v["pass"] and v["complete"]
    assert v["backend"].startswith("torch ") and v["bundle_equals_record"]
    assert v["bundle_manifest_sha256"] == check.sha
    assert v["bundle_files"] == read_manifest(check.bundle)["files"]
    assert v["v1_code"] == V1_CODE and v["threads"]["requested"] == check.n
    assert v["threads"]["env"] == {k: str(check.n) for k in THREAD_ENV}
    assert v["threads_checked"]["after"]["intra_op"] == check.n
    # The same weights in the same runtime: no difference at all.
    t = v["tolerances"]
    assert t["map_max_abs"]["measured"] == 0.0 and t["logits_max_abs"]["measured"] == 0.0
    assert t["text"]["measured"] == {"near_ties": 0, "other": 0} and t["page_cer"]["measured"] == 0
    e = v["end_to_end"]
    assert e["page_cer_bundle"] == e["page_cer_reference"]
    assert "page_cer_onnx" not in e and "onnx" not in json.dumps(v["tolerances"])
    assert {"started_utc", "ended_utc", "versions", "git_commit", "seconds"} <= set(v)
    with pytest.raises(SystemExit, match="written once"):
        _run(check, check.bundle, out)


def test_another_bundle_hash_is_refused_before_anything_runs(check):
    out = check.tmp / "v.json"
    with pytest.raises(SystemExit, match="not the registered bundle"):
        _run(check, check.bundle, out, sha="0" * 64)
    assert not out.exists()


def test_a_bundle_unlike_the_record_is_refused(check):
    warm = _bundle(check.tmp, "warm", temperature=2.0)
    with pytest.raises(SystemExit, match="recognizer temperature"):
        _run(check, warm, check.tmp / "v.json", sha=sha256_file(warm / "manifest.json"))


def test_changed_references_are_refused(check):
    with (check.ref_dir / "recognizer_lines.jsonl").open("a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(SystemExit, match="SHA-256"):
        _run(check, check.bundle, check.tmp / "v.json")


def test_other_recognizer_weights_fail_the_check(check):
    other = _bundle(check.tmp, "other", rec_seed=1)
    out = check.tmp / "v.json"
    assert _run(check, other, out, sha=sha256_file(other / "manifest.json")) == 1
    v = json.loads(out.read_text(encoding="utf-8"))
    assert not v["pass"] and not v["tolerances"]["logits_max_abs"]["pass"]


def test_relabel_renames_keys_only():
    got = v1check.relabel({"page_cer_onnx": 1, "x": [{"torch": "a", "onnx": "onnx"}]})
    assert got == {"page_cer_bundle": 1, "x": [{"reference": "a", "bundle": "onnx"}]}


def test_an_imported_main_needs_a_fresh_process_unless_told(check):
    with pytest.raises(SystemExit, match="run this as a script"):
        v1check.main(["--bundle", str(check.bundle), "--out", str(check.tmp / "v.json")])


@pytest.mark.parametrize(
    ("argv", "want"),
    [([], 10), (["--threads", "4"], 4), (["--threads=3"], 3), (["--threads", "0"], None)],
)  # fmt: skip
def test_threads_are_read_from_the_command_line(argv, want):
    assert v1check.requested_threads(argv) == want
