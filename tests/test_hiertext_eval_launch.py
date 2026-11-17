"""Tests for scripts/hiertext_eval_launch.py, which starts HierText's pinned eval.py with Apache
Beam's job deadline lifted.

Beam lives only in `.venv-hiertext`, so these tests put a stand-in for its options module in
`sys.modules`. The launcher against the real evaluator is covered by
`test_official_eval.py::test_run_hiertext_on_a_made_up_page`, skipped where that environment is
absent.
"""

from __future__ import annotations

import importlib
import json
import sys
import types

import pytest


@pytest.fixture
def launch(monkeypatch):
    class PipelineOptions:
        def __init__(self, flags=None, **kwargs):
            self.flags = flags
            self.kwargs = kwargs

    beam = types.ModuleType("apache_beam")
    options = types.ModuleType("apache_beam.options")
    pipeline_options = types.ModuleType("apache_beam.options.pipeline_options")
    pipeline_options.PipelineOptions = PipelineOptions
    options.pipeline_options = pipeline_options
    beam.options = options
    monkeypatch.setitem(sys.modules, "apache_beam", beam)
    monkeypatch.setitem(sys.modules, "apache_beam.options", options)
    monkeypatch.setitem(sys.modules, "apache_beam.options.pipeline_options", pipeline_options)
    monkeypatch.delitem(sys.modules, "hiertext_eval_launch", raising=False)
    module = importlib.import_module("hiertext_eval_launch")
    yield module
    sys.modules.pop("hiertext_eval_launch", None)


def patched(launch):
    options = sys.modules["apache_beam.options.pipeline_options"].PipelineOptions
    options.__init__ = launch._init_without_limit
    return options


def test_a_flag_list_gets_the_deadline(launch):
    options = patched(launch)
    given = ["--runner=DirectRunner"]
    opts = options(given)
    assert opts.flags == ["--runner=DirectRunner", f"--job_server_timeout={launch.TIMEOUT_S}"]
    assert given == ["--runner=DirectRunner"]  # the caller's list is not changed
    assert launch.TIMEOUT_S >= 24 * 3600


@pytest.mark.parametrize("flag", ["--job_server_timeout=60", "--job-server-timeout=60"])
def test_a_deadline_already_given_is_kept(launch, flag):
    options = patched(launch)
    assert options([flag]).flags == [flag]


def test_a_deadline_given_as_keyword_and_the_sys_argv_case_are_left_alone(launch):
    options = patched(launch)
    assert options(["--a"], job_server_timeout=60).flags == ["--a"]
    assert options().flags is None


def test_main_runs_eval_py_unchanged_as_main(launch, tmp_path, monkeypatch):
    (tmp_path / "eval.py").write_text(
        "import json, sys\n"
        "from apache_beam.options import pipeline_options as po\n"
        "o = po.PipelineOptions(['--x'])\n"
        "json.dump({'argv': sys.argv, 'name': __name__, 'flags': o.flags, 'path0': sys.path[0]},\n"
        "          open('out.json', 'w'))\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["hiertext_eval_launch.py", "--gt=g.json", "--e2e"])
    monkeypatch.setattr(sys, "path", list(sys.path))
    launch.main()
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out["argv"] == ["eval.py", "--gt=g.json", "--e2e"]
    assert out["name"] == "__main__"
    assert out["flags"] == ["--x", f"--job_server_timeout={launch.TIMEOUT_S}"]
    assert out["path0"] == str(tmp_path)
