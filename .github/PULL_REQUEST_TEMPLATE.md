## What and why

<!-- What this pull request changes, and why. Link the issue it addresses, if there is one. -->

## Tests

<!-- New or changed tests, and how the change was checked beyond them. -->

## Checks run locally

The steps of the `test` job in `.github/workflows/ci.yml` (CONTRIBUTING.md explains each one):

- [ ] `ruff check .`
- [ ] `ruff format --check .`
- [ ] `mypy`
- [ ] `pytest`
- [ ] `python scripts/validate_configs.py`
- [ ] `python scripts/experiment_summary.py --check`
- [ ] `python scripts/experiment_comparison.py --check`
- [ ] `python scripts/trace_numbers.py --check`
- [ ] `python scripts/check_citation.py`
- [ ] `glypharis-ocr --help` and `glypharis-serve --help`
- [ ] `python -c "import glypharis, glypharis.pipeline; print(glypharis.__version__)"`

For changes to packaging (`pyproject.toml`, package data, entry points, dependencies), also the
`package` job's checks: build, `twine check --strict`, `scripts/check_wheel.py`,
`scripts/check_git_archive.py --commit HEAD`, and the clean-wheel install.

## Project rules

- [ ] No datasets, checkpoints, model bundles or weights are added to git.
- [ ] Every number added or changed in a document is copied from a generated report and carries its
      experiment ID; no number is typed by hand, and no generated file is edited by hand.
- [ ] The v1 code paths (`glypharis/pipeline`, `glypharis/models`, `glypharis/export`,
      `glypharis/data`, `configs`) are untouched, or this pull request is part of a new model version.
- [ ] Any new training or evaluation run has an experiment ID registered in
      `research/experiments.md` before it ran.
- [ ] No test-set data is read outside `glypharis.eval.access.open_test_access` and the phases it
      allows.
- [ ] Commit messages are plain and descriptive, with no tool or generated-by trailers.
