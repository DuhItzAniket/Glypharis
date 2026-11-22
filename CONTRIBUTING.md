# Contributing to GLYPHARIS

Thank you for taking the time to help. GLYPHARIS is a research project as well as a package, so a
change is reviewed for two things: that the code works, and that every number the project reports
still traces to the run that produced it. This page covers both.

- Questions about using the package: start with the [documentation index](docs/README.md) and the
  [integration guide](docs/INTEGRATION.md).
- Bugs and feature requests: open an issue with the matching form.
- Security problems: do not open a public issue; follow [SECURITY.md](SECURITY.md).

## Setting up

The package needs Python 3.11, at least 3.11.4 and below 3.12 (`requires-python` in
[pyproject.toml](pyproject.toml)). Development uses the exact versions in
[requirements/lock.txt](requirements/lock.txt). That file pins everything except `torch` and
`torchvision`, whose wheels differ by platform; install those two first, at the versions in
[requirements/README.md](requirements/README.md), from the PyTorch index that fits your machine.

Linux or macOS:

```bash
git clone https://github.com/DuhItzAniket/Glypharis.git
cd Glypharis
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# CPU only (what CI uses):
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/lock.txt
pip install -e . --no-deps
pip check
```

Windows (Git Bash), with a CUDA 12.6 build of PyTorch:

```bash
py -3.11 -m venv .venv
P=.venv/Scripts/python
$P -m pip install --upgrade pip
$P -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
$P -m pip install -r requirements/lock.txt
$P -m pip install -e . --no-deps
$P -m pip check
```

The development machine, its environment report and the separate environments used for the
baselines and the official evaluators are described in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
To change a pin, follow the steps in [requirements/README.md](requirements/README.md) and say why in
the commit message; a change of major or minor version also changes the lower bound in
`pyproject.toml` (`tests/test_packaging.py` checks this).

Model weights are not in the repository. Most tests build tiny models of their own and need no
download. To run the real model locally, fetch it once with `glypharis.download_model("glypharis-v1")`
([README](README.md#quick-start)).

## Running the checks

These are the steps of the `test` job in [.github/workflows/ci.yml](.github/workflows/ci.yml), in
the order CI runs them. Run them before opening a pull request:

```bash
ruff check .
ruff format --check .
mypy
pytest
python scripts/validate_configs.py
python scripts/experiment_summary.py --check
python scripts/experiment_comparison.py --check
python scripts/trace_numbers.py --check
python scripts/check_citation.py
glypharis-ocr --help
glypharis-serve --help
python -c "import glypharis, glypharis.pipeline; print(glypharis.__version__)"
```

What each one checks:

| Step | What it checks |
|---|---|
| `ruff check .`, `ruff format --check .` | Lint and formatting (settings in `pyproject.toml`; line length 100) |
| `mypy` | Types for the `glypharis` package, with untyped definitions refused |
| `pytest` | The test suite under `tests/`; tests marked `gpu` skip themselves when no CUDA device is present |
| `scripts/validate_configs.py` | Every YAML config under `configs/` loads with strict types; the freeze records under `configs/final/` name loadable configs and hashed checkpoints |
| `scripts/experiment_summary.py --check` | The experiment registry follows its rules, and the generated `EXPERIMENTS.md` and `reports/experiment_summary.*` are current |
| `scripts/experiment_comparison.py --check` | The generated comparison table under `reports/benchmark_results/` is current |
| `scripts/trace_numbers.py --check` | Every number in the paper notes, `MODEL_CARD.md` and `DATASETS.md` traces to a report, and the trace under `reports/paper_trace/` is current |
| `scripts/check_citation.py` | `CITATION.cff` follows the Citation File Format 1.2.0 and matches the package version and licence |
| The three commands | The installed entry points and the package import work |

CI also writes an environment report (`python scripts/env_report.py --out ci-env-report`) and keeps
it as the `env-report-ci` artifact of the run.

The `package` job of the same workflow builds the sdist and wheel from a clean checkout
(`python -m build`), runs `twine check --strict`, checks what the files contain with
`scripts/check_wheel.py`, checks what `git archive` leaves out with
`scripts/check_git_archive.py --commit HEAD`, installs the wheel with its dependencies in a fresh
environment (`pip check`), and runs the commands and a tiny-bundle read from outside the checkout
(`scripts/tiny_bundle_check.py`, `scripts/make_sample_page.py`, `glypharis-check`). To reproduce
it locally, install the build tools with `pip install -c requirements/build-constraints.txt build twine`
and follow the job's steps.

When a `--check` step fails because a generated file is out of date, regenerate it with the same
script without `--check` (for example `python scripts/experiment_summary.py`) and commit the result.
Never edit a generated file by hand.

## Project rules that apply to contributions

The full rules are in [docs/WORKFLOW.md](docs/WORKFLOW.md) and
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md). These are the ones a contribution is
most likely to touch.

1. **Register experiments before they run.** Every training or evaluation run that produces a
   number someone might quote gets an ID (`GW-REC-NNN`, `GW-DET-NNN`, `GW-E2E-NNN`, `GW-DATA-NNN`,
   `GW-EFF-NNN`, `GW-BASE-NNN`), allocated in [research/experiments.md](research/experiments.md)
   together with its hypothesis before the run starts. A failed or abandoned run keeps its ID and is
   recorded in [research/failures.md](research/failures.md). IDs are never reused.
2. **Validation data for every decision; test data only through the logged path.** Model
   selection, early stopping and tuning use validation splits only. Test splits and external
   benchmarks are read only in the phases the plan allows (5, 14 and 22), and only through
   `glypharis.eval.access.open_test_access`, which refuses any other phase and appends a row to
   [research/test_access_log.md](research/test_access_log.md) before any data is returned. A pull
   request must not add test-set reads outside that path, and must not add rows to the log by hand.
3. **Every number traces to a report.** Metrics are never typed into a document by hand. They are
   copied from a generated report under `reports/` together with the experiment ID that produced
   it. Negative and null results are recorded with the same care as positive ones. Comparative
   wording ("best", "state of the art") needs a controlled comparison behind it.
4. **No datasets, checkpoints or weights in git.** Raw and processed data, checkpoints, model
   bundles and archives stay out of the repository (`.gitignore` lists the patterns). Manifests,
   checksums and READMEs under `data/` are committed. Dataset terms are in
   [DATASETS.md](DATASETS.md).
5. **The v1 code paths are frozen.** `glypharis/pipeline`, `glypharis/models`, `glypharis/export`,
   `glypharis/data` and `configs` are the code and settings that make v1. They must stay equal to
   the `v1-code` tag (`glypharis.efficiency.v1.assert_v1_code`, used by every script that builds v1),
   so that the published results keep describing the published model. A change there belongs to a
   new model version, with its own experiment IDs and evaluation, and is planned with the
   maintainer first. Fixes elsewhere (scripts, tooling, documentation, the server, the command-line
   interface, and the model download in `glypharis/hub.py`) are welcome as usual.
6. **One logical change per pull request**, with tests for new logic and the full suite passing.

## Commit messages

Write the message the way an engineer would write a lab-notebook entry: what changed and, when it is
not obvious, why.

- Imperative mood, at most 72 characters on the first line, for example
  `Handle empty HierText line annotations instead of dropping them`.
- An optional body, wrapped at 72 columns, that explains why and names experiment IDs where they
  apply.
- One logical change per commit. No filler such as "various fixes" or "update code".
- No generated-by or co-authored-by trailers for tools.

## Pull requests

Branch from `main`, push to your fork and open a pull request against `main`. CI runs on every pull
request; it must be green before review. The pull request template lists the checks and the project
rules above; tick the ones that apply and explain any that do not. The maintainer merges with a merge
commit.

## Licence of contributions

The code and the project's own text are under the Apache License, Version 2.0 ([LICENSE](LICENSE),
[NOTICE](NOTICE)). Unless you state otherwise, any contribution you submit for inclusion is licensed
under the same terms, as section 5 of the licence provides. The model weights are distributed
separately under their own terms ([MODEL_CARD.md](MODEL_CARD.md#licence-information)); a pull request
does not change them.
