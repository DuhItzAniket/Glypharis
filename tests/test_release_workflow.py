"""The release workflow's structure and scripts/release_version_check.py (Phase 23, item 6).

actionlint is not run here; these structural checks stand in for it on the points the plan fixes:
the two triggers, read-only defaults, least permissions per job, pinned actions, a draft-only
stage 1 and a stage 2 that neither checks out nor builds.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import release_version_check as rvc
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
MODEL_ARCHIVE = "glypharis-v1.tar.gz"
SHA_PIN = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")


@pytest.fixture(scope="module")
def workflow() -> dict[Any, Any]:
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # PyYAML (YAML 1.1) reads the bare key `on` as the boolean True.
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict), triggers
    return triggers


def _jobs(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    return jobs


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return list(job.get("steps", []))


def _all_steps(workflow: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [(name, step) for name, job in _jobs(workflow).items() for step in _steps(job)]


def _step_text(step: dict[str, Any]) -> str:
    return yaml.safe_dump(step, sort_keys=True)


def _uses(step: dict[str, Any]) -> str:
    return str(step.get("uses", ""))


# --- triggers and permissions ----------------------------------------------------------------


def test_only_dispatch_and_published_release(workflow: dict[str, Any]) -> None:
    triggers = _triggers(workflow)
    assert set(triggers) == {"workflow_dispatch", "release"}
    assert triggers["release"] == {"types": ["published"]}
    raw = WORKFLOW.read_text(encoding="utf-8")
    for forbidden in ("push:", "pull_request", "schedule:", "workflow_run", "tags:"):
        assert forbidden not in raw, forbidden


def test_dispatch_inputs(workflow: dict[str, Any]) -> None:
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"dry_run", "tag"}
    assert inputs["dry_run"]["type"] == "boolean"
    assert inputs["dry_run"]["default"] is True
    assert inputs["tag"]["type"] == "string"
    assert inputs["tag"]["default"] == "v1.0"


def test_top_level_permissions_read_only(workflow: dict[str, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] is False
    assert "github.workflow" in workflow["concurrency"]["group"]
    # A queued dispatch must never cancel the pending publish run.
    assert "github.event_name" in workflow["concurrency"]["group"]


def test_every_job_sets_its_own_least_permissions(workflow: dict[str, Any]) -> None:
    jobs = _jobs(workflow)
    assert set(jobs) == {"check-build", "draft-release", "pypi"}
    assert jobs["check-build"]["permissions"] == {"contents": "read"}
    assert jobs["draft-release"]["permissions"] == {"contents": "write"}
    assert jobs["pypi"]["permissions"] == {"id-token": "write", "contents": "read"}


def test_id_token_only_on_the_pypi_job(workflow: dict[str, Any]) -> None:
    for name, job in _jobs(workflow).items():
        if name != "pypi":
            assert "id-token" not in job.get("permissions", {}), name
    assert "id-token" not in workflow["permissions"]


# --- actions ---------------------------------------------------------------------------------


def test_every_action_pinned_by_full_sha(workflow: dict[str, Any]) -> None:
    uses = [_uses(step) for _, step in _all_steps(workflow) if "uses" in step]
    assert uses
    for ref in uses:
        assert SHA_PIN.match(ref), ref
    for job in _jobs(workflow).values():
        assert "uses" not in job  # no reusable workflow


def test_pins_carry_a_version_comment() -> None:
    for line in WORKFLOW.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\s*(- )?uses:", line):
            assert re.search(r"@[0-9a-f]{40} # v\d+\.\d+\.\d+$", line), line


def test_checkouts_drop_credentials_and_no_pip_cache(workflow: dict[str, Any]) -> None:
    for name, step in _all_steps(workflow):
        uses = _uses(step)
        if uses.startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] is False, name
        if uses.startswith("actions/setup-python@"):
            assert "cache" not in step.get("with", {}), name


def test_no_third_party_release_action(workflow: dict[str, Any]) -> None:
    owners = {_uses(step).split("/")[0] for _, step in _all_steps(workflow) if "uses" in step}
    assert owners <= {"actions", "pypa"}
    pypa = [_uses(s) for _, s in _all_steps(workflow) if _uses(s).startswith("pypa/")]
    assert all(u.startswith("pypa/gh-action-pypi-publish@") for u in pypa)


# --- stage 1 ---------------------------------------------------------------------------------


def test_check_build_runs_every_check(workflow: dict[str, Any]) -> None:
    job = _jobs(workflow)["check-build"]
    assert job["if"] == "github.event_name == 'workflow_dispatch'"
    runs = "\n".join(str(step.get("run", "")) for step in _steps(job))
    for needed in (
        "pip install -c requirements/build-constraints.txt build twine",
        'scripts/release_version_check.py --tag "$TAG"',
        "python -m build",
        "twine check --strict",
        "scripts/check_wheel.py dist/*.whl dist/*.tar.gz",
        "scripts/check_git_archive.py --commit HEAD",
        "--index-url https://download.pytorch.org/whl/cpu",
        "pip check",
        'glypharis-ocr" --help',
        'glypharis-serve" --help',
        "tiny_bundle_check.py",
        "sha256sum",
        "SHA256SUMS",
    ):
        assert needed in runs, needed
    env = workflow["env"]
    assert env == {"TORCH_VERSION": "2.14.1", "TORCHVISION_VERSION": "0.29.1"}
    build = [s for s in _steps(job) if "python -m build" in str(s.get("run", ""))]
    assert len(build) == 1 and "build-constraints.txt" in build[0]["env"]["PIP_CONSTRAINT"]
    uploads = [s for s in _steps(job) if _uses(s).startswith("actions/upload-artifact@")]
    assert len(uploads) == 1 and uploads[0]["with"]["path"] == "dist/"


def test_tag_input_never_interpolated_into_a_shell(workflow: dict[str, Any]) -> None:
    for name, step in _all_steps(workflow):
        run = str(step.get("run", ""))
        assert "${{" not in run, (name, step.get("name"))


def test_draft_release_only_off_dry_run_and_only_a_draft(workflow: dict[str, Any]) -> None:
    job = _jobs(workflow)["draft-release"]
    assert job["needs"] == "check-build"
    assert "inputs.dry_run == false" in job["if"]
    assert "workflow_dispatch" in job["if"]
    creates = [s for s in _steps(job) if "gh release create" in str(s.get("run", ""))]
    assert len(creates) == 1
    command = creates[0]["run"]
    for flag in ("--draft", "--verify-tag", "--notes-file docs/RELEASE_NOTES_v1.0.md", "dist/*"):
        assert flag in command, flag
    assert creates[0]["env"]["GH_TOKEN"] == "${{ github.token }}"
    # No other job creates, edits or uploads to a release.
    for name, step in _all_steps(workflow):
        run = str(step.get("run", ""))
        for verb in ("gh release upload", "gh release edit", "gh release delete"):
            assert verb not in run, (name, verb)
        if "gh release create" in run:
            assert name == "draft-release"


def test_draft_release_checks_the_tag_before_creating(workflow: dict[str, Any]) -> None:
    steps = _steps(_jobs(workflow)["draft-release"])
    checkout = steps[0]
    assert _uses(checkout).startswith("actions/checkout@")
    assert checkout["with"]["ref"] == "refs/tags/${{ inputs.tag }}"
    texts = [str(s.get("run", "")) for s in steps]
    create = next(i for i, t in enumerate(texts) if "gh release create" in t)
    tag_check = next(i for i, t in enumerate(texts) if "git rev-parse HEAD" in t)
    guard = next(i for i, s in enumerate(steps) if str(s.get("name", "")).startswith("Guard"))
    assert tag_check < create and guard < create
    assert "^{commit}" in texts[tag_check] and "$BUILT" in texts[tag_check]


def test_check_build_builds_the_tag_off_dry_run(workflow: dict[str, Any]) -> None:
    checkout = next(
        s
        for s in _steps(_jobs(workflow)["check-build"])
        if _uses(s).startswith("actions/checkout@")
    )
    ref = checkout["with"]["ref"]
    assert "inputs.dry_run" in ref and "refs/tags/" in ref and "github.sha" in ref
    assert checkout["with"]["fetch-depth"] == 0


# --- stage 2 ---------------------------------------------------------------------------------


def test_pypi_job_is_protected_and_neither_checks_out_nor_builds(
    workflow: dict[str, Any],
) -> None:
    job = _jobs(workflow)["pypi"]
    assert job["if"] == "github.event_name == 'release'"
    env = job["environment"]
    assert (env["name"] if isinstance(env, dict) else env) == "pypi"
    assert "needs" not in job
    for step in _steps(job):
        uses = _uses(step)
        run = str(step.get("run", ""))
        assert not uses.startswith("actions/checkout@")
        assert not uses.startswith("actions/setup-python@")
        assert not uses.startswith("actions/download-artifact@")
        for building in ("python -m build", "pip install", "setup.py", "git clone"):
            assert building not in run, building
    publish = [s for s in _steps(job) if _uses(s).startswith("pypa/gh-action-pypi-publish@")]
    assert len(publish) == 1 and publish == _steps(job)[-1:]
    assert publish[0]["with"] == {"packages-dir": "dist/"}  # no password: Trusted Publishing
    raw = _step_text({"steps": _steps(job)})
    assert "secrets." not in raw and "password" not in raw


def test_pypi_job_uploads_only_checked_code_files(workflow: dict[str, Any]) -> None:
    steps = _steps(_jobs(workflow)["pypi"])
    runs = [str(s.get("run", "")) for s in steps]
    downloads = [r for r in runs if "gh release download" in r]
    assert downloads and all('"$TAG"' in r and "$GITHUB_REPOSITORY" in r for r in downloads)
    assert "--pattern SHA256SUMS" in downloads[0]
    assert all("*.tar.gz" not in r for r in downloads)  # would match the model archive
    assert any("sha256sum -c --strict SHA256SUMS" in r for r in runs)
    for step in steps:
        if "gh release download" in str(step.get("run", "")):
            assert step["env"]["GH_TOKEN"] == "${{ github.token }}"
            assert step["env"]["TAG"] == "${{ github.event.release.tag_name }}"


def test_model_archive_named_only_by_guards_that_refuse_it(workflow: dict[str, Any]) -> None:
    guards = []
    for job_name, step in _all_steps(workflow):
        if MODEL_ARCHIVE not in _step_text(step):
            continue
        name = str(step.get("name", ""))
        run = str(step.get("run", ""))
        assert name.startswith("Guard"), (job_name, name)
        assert "uses" not in step and "gh " not in run and "exit 1" in run, (job_name, name)
        guards.append(job_name)
    assert sorted(guards) == ["draft-release", "pypi"]


# --- scripts/release_version_check.py --------------------------------------------------------


def _init(tmp_path: Path, version: str) -> Path:
    init = tmp_path / "__init__.py"
    init.write_text(f'"""Package."""\n\n__version__ = "{version}"\n', encoding="utf-8")
    return init


def test_v1_0_matches_1_0_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert rvc.main(["--tag", "v1.0", "--init", str(_init(tmp_path, "1.0.0"))]) == 0
    assert "ok" in capsys.readouterr().out
    assert rvc.main(["--tag", "1.0.0", "--init", str(_init(tmp_path, "1.0.0"))]) == 0


def test_other_version_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert rvc.main(["--tag", "v1.0.1", "--init", str(_init(tmp_path, "1.0.0"))]) == 1
    assert "FAIL" in capsys.readouterr().out
    assert rvc.main(["--tag", "v1.1", "--init", str(_init(tmp_path, "1.0.0"))]) == 1


@pytest.mark.parametrize(
    "tag", ["", "v", "release", "v1.0-final!", "vv1.0", " v1.0", "V1.0", "v1.0-rc1", "v1.0+abc"]
)
def test_bad_tag_fails(tmp_path: Path, tag: str) -> None:
    assert rvc.main(["--tag", tag, "--init", str(_init(tmp_path, "1.0.0"))]) == 1


@pytest.mark.parametrize(("tag", "version"), [("v1.0rc1", "1.0.0rc1"), ("v1.0.dev0", "1.0.dev0")])
def test_pre_release_refused_unless_allowed(tmp_path: Path, tag: str, version: str) -> None:
    init = str(_init(tmp_path, version))
    assert rvc.main(["--tag", tag, "--init", init]) == 1
    assert rvc.main(["--tag", tag, "--init", init, "--allow-pre"]) == 0


def test_init_without_one_version_line_fails(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text('"""Package."""\n', encoding="utf-8")
    assert rvc.main(["--tag", "v1.0", "--init", str(init)]) == 1
    assert rvc.main(["--tag", "v1.0", "--init", str(tmp_path / "missing.py")]) == 1


def test_reads_the_text_without_importing(tmp_path: Path) -> None:
    init = tmp_path / "__init__.py"
    init.write_text('raise SystemExit("imported")\n__version__ = "1.0.0"\n', encoding="utf-8")
    assert rvc.package_version(init) == "1.0.0"
    assert rvc.check("v1.0", "1.0.0") == rvc.Version("1.0.0")


def test_repository_version_has_a_matching_tag() -> None:
    version = rvc.package_version(ROOT / "glypharis" / "__init__.py")
    assert rvc.check(f"v{version}", version) == rvc.Version(version)
