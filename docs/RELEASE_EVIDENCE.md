# Release evidence for v1.0

This table maps each precondition and task of Phase 23 in
[IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md) (Release v1.0), and the release-related success
criteria of [ROADMAP](ROADMAP.md) §4, to its evidence: a report, a test, a CI job or a document. Each
piece of evidence carries its experiment ID where it has one. Where the evidence is still missing, the
row names the gap. This is work item 12 of the Phase 23 plan in [PHASE_STATUS](PHASE_STATUS.md).

Nothing here says that v1.0 has been released. Version 1.0.0 is not on PyPI, no GitHub release or tag
`v1.0` exists, no Zenodo record exists, and no weight file has left the development machine. The
weights wait for qualified advice ([ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md),
PHASE_STATUS Phase 23, D4).

The table gives no metric values. The results sit in the reports named here and in
`research/experiments.md` under the same IDs.

**Status words.** *Met* means the evidence exists in the repository and supports the row for
pull request A (release-ready, publishing nothing). *Open* means something is missing, and the row
says what and what it waits for. Experiment statuses are as recorded in `research/experiments.md`.

## 1. Preconditions (IMPLEMENTATION_PLAN Phase 23)

| Precondition or task | Evidence | Status |
|---|---|---|
| Tests pass | `tests/` (run by the CI job `test`, step "Pytest"). The Phase 23 additions: `tests/test_hub.py` (download, resume, wrong hash, wrong manifest, offline, URL override, two processes), `tests/test_packaging.py`, `tests/test_pack_release.py`, `tests/test_check.py`, `tests/test_citation.py`, `tests/test_release_workflow.py` | Met on the branch head. Open: the run on the commit that carries this table, and on the merge commit of pull request A (item 13). |
| CI passes | `.github/workflows/ci.yml`: job `test` (ruff, mypy, pytest, config validation, `scripts/experiment_summary.py --check`, the comparison table check, `scripts/trace_numbers.py --check`, `scripts/check_citation.py`, the installed command in a fresh environment) and job `package` (sdist and wheel built from a clean checkout, `twine check`, `scripts/check_wheel.py`, `scripts/check_git_archive.py --commit HEAD`, the wheel installed with its dependencies in a fresh environment, and the commands, a tiny-bundle read and `glypharis-check` run from outside the checkout). The CI runs on GitHub for the branch head succeeded. | Met on the branch head. Open: green on the merge commit of pull request A, including `package`; exit criterion 1. |
| Release workflow dry run creates nothing | `.github/workflows/release.yml` (stage 1 by manual dispatch, `dry_run` on by default); `tests/test_release_workflow.py` | Open: the dry-run dispatch on `main` with the would-be tag `v1.0` runs only after merge A (items 6 and 13). |
| Training pipeline verified: the v1 runs | GW-REC-034 (`done`): `reports/training/GW-REC-034/report.md`; GW-DET-001 (`done`): `reports/training/GW-DET-001/report.md`; configs `configs/rec/gw-rec-034.yaml`, `configs/det/gw-det-001.yaml`; `tests/test_rec_training.py`, `tests/test_det_training.py` | Met. |
| Training pipeline verified: checkpoint and resume | `glypharis/training/checkpoint.py`; `scripts/rec_resume_check.py` with `reports/training/sanity/GW-REC-001/resume_check.json`; `scripts/det_resume_check.py` with `reports/training/sanity/GW-DET-001/resume_check.json`; `reports/training/phase11/trainer_move/resume_check.json`; `tests/test_checkpoint_retry.py` | Met. |
| Training pipeline verified: from a clean clone | GW-REC-046 and GW-DET-008, both `done` (`reports/training/smoke/phase23_smoke.md`, `reports/training/GW-REC-046/`, `reports/training/GW-DET-008/`; item 11, D11) | Met: each trainer started from a clean clone, checkpointed, resumed once and wrote its report with finite losses. This shows the pipeline runs, not that it reproduces v1's training. |
| Inference verified | GW-E2E-006 (`done`, the frozen pipeline's reference): `reports/external/`; GW-E2E-011 (`done`, the package on the v1 bundle against GW-E2E-006): `reports/package/GW-E2E-011.json`; GW-E2E-013 (`done`, the clean-machine-style walkthrough): `reports/package/GW-E2E-013.json`; GW-E2E-022 (`done`, the built wheel with the fetched archive): `reports/package/GW-E2E-022.json`; GW-E2E-012 (`done`, failure paths and damaged bundles): `reports/hardening/GW-E2E-012.json`; `tests/test_api.py`, `tests/test_server.py` | Met. |
| Export verified | GW-EFF-006 (`done`, PyTorch references and the v1 fp32 ONNX bundle): `reports/efficiency/phase16/` (`reference_check.json`, `v1_bundle.json`); GW-EFF-012 (`done`, the v1 bundle on the PyTorch backend, the shipped path): `reports/export/GW-EFF-012/torch_bundle_check.json`; `scripts/export_check.py`; `tests/test_export.py` | Met. The phone candidate (GW-E2E-021, `done`, `reports/final_validation/candidate/h11.md`) is a documented result and does not ship (D7). |
| Docs complete | `README.md` (install section, and the note that the full trace needs the tagged repository), `MODEL_CARD.md`, `DATASETS.md`, `docs/ARCHITECTURE.md`, `docs/REPRODUCING.md`, `docs/FINAL_VALIDATION.md`, `docs/PYPI.md` (the PyPI description), `docs/INTEGRATION.md` (the integration guide of item 12), `docs/RELEASE_NOTES_v1.0.md`, `CHANGELOG.md`, `docs/ADVISER_BRIEF.md`, this table; CI checks `scripts/trace_numbers.py --check` and `scripts/experiment_summary.py --check` | Open: the wording that states what v1.0 distributes changes only in item 14, after the advice. The documentation criteria of ROADMAP §4 close with the completion record (D12). |
| Model artifact verified: hash | Manifest pin `875357a6…` in `REGISTRY` in `glypharis/hub.py`, recorded for the v1 bundle in `reports/efficiency/phase16/v1_bundle.json` (GW-EFF-006) and matched in `reports/package/GW-E2E-022.json` (`bundle_manifest_sha256`) (GW-E2E-022). Archive pin (size and SHA-256) in the same `REGISTRY` entry, matched by the download record and the `glypharis-check` output in `reports/package/GW-E2E-022_fetch.json` (GW-E2E-022). Packer `scripts/pack_release.py`; `tests/test_pack_release.py` (two packs byte-identical, members, fixed metadata, wrong pin refused, extraction through `download_model`) | Met for merge A. Open: the item 7 rule that two packs of the real bundle give the same bytes is covered by the test on a tiny bundle; no committed report records a second pack of the real bundle. If the advice changes the weight notices, the archive is packed again and the pin changed in pull request B (item 7). |
| Model artifact verified: equivalence | GW-E2E-022 (`done`): the archive fetched from a local server by `download_model` into an empty models folder, then the package check against GW-E2E-006, pass (`reports/package/GW-E2E-022.json`, `reports/package/GW-E2E-022_fetch.json`; its `v1_code` field records the v1 code paths equal to `v1-code`). Earlier equivalence on the same bundle: GW-E2E-011, GW-E2E-013, GW-EFF-012 | Met for merge A. Open: the shipped asset URL is first used in GW-E2E-023 (`planned`, after publication). |

## 2. Tasks (IMPLEMENTATION_PLAN Phase 23)

| Precondition or task | Evidence | Status |
|---|---|---|
| Release notes | `docs/RELEASE_NOTES_v1.0.md` (draft; results cited by experiment ID); `CHANGELOG.md` (`[1.0.0]`, marked unreleased) | Met as a draft. Open: the wording is changed to what v1.0 distributes in item 14, after the advice. |
| Version bump | `__version__ = "1.0.0"` in `glypharis/__init__.py`, read by `pyproject.toml` as a dynamic version; `tests/test_packaging.py` (`test_version_is_read_from_the_package`, `test_installed_metadata_matches_the_version`); the tag check `scripts/release_version_check.py` with `tests/test_release_workflow.py` (`test_v1_0_matches_1_0_0`, `test_other_version_fails`); `CITATION.cff` `version`; the package-version row of `MODEL_CARD.md`; `requires-python = ">=3.11.4,<3.12"` (D13), checked by `test_licence_python_and_project_fields`; the installed wheel's version in `reports/package/GW-E2E-022_fetch.json` (GW-E2E-022). `model_version` stays `glypharis_v1` (D8). | Met. |
| Tagged GitHub release | `.github/workflows/release.yml`: stage 1 `check-build` and `draft-release` (manual dispatch, a draft holding the code files and `SHA256SUMS` only), stage 2 `pypi` (on a release the owner publishes by hand); `tests/test_release_workflow.py` (triggers, least permissions, actions pinned by SHA, no credentials kept, the model archive refused) | Open: no tag `v1.0` exists; it goes on the merge commit of pull request B and is never moved (item 14). No draft release exists (item 15a). `actionlint` was replaced by the structural checks of `tests/test_release_workflow.py` (PHASE_STATUS, Phase 23, amendments). The dry run on `main` waits for merge A. |
| The model artifact on the release (or a download script) | The archive `glypharis-v1.tar.gz`, built by `scripts/pack_release.py` outside every worktree, with the weight notices `release/weights/LICENSE-WEIGHTS` and `release/weights/NOTICE.md`; fetched by `glypharis.download_model` and by `glypharis-ocr` and `glypharis-serve` for registered names (`glypharis/hub.py`, D6; `tests/test_hub.py`); rehearsed in GW-E2E-022 | Open: the archive is not uploaded anywhere. Upload waits for the qualified advice and the owner's go for the weights (D4, item 15b). The registry's `published` flag stays off until pull request B (item 14). |
| Licence terms that respect the training data | `LICENSE` (Apache-2.0, the code and the project's own text, D2); `NOTICE` (what Apache-2.0 does not cover, D3); `release/weights/LICENSE-WEIGHTS` and `release/weights/NOTICE.md` (the weights' intended terms, CC BY-SA 4.0 with HierText and Open Images attribution and the ImageNet disclosure); ADR-0014; `DATASETS.md`; `MODEL_CARD.md`; `docs/ADVISER_BRIEF.md`; `tests/test_pack_release.py` (`test_the_weight_notices_are_outside_the_package_and_say_what_they_must`); `scripts/check_wheel.py` (no weights or weight notices in the code packages) | Code: met. Weights: open. Their terms are intended, not settled, until the qualified advice. ADR-0014's outcome is recorded in item 14, or the owner records a deferral (exit criterion 6). |

## 3. Publishing (IMPLEMENTATION_PLAN Phase 23)

| Precondition or task | Evidence | Status |
|---|---|---|
| `LICENSE` for the code | `LICENSE`; `license = "Apache-2.0"` and `license-files` in `pyproject.toml`; `test_licence_python_and_project_fields` in `tests/test_packaging.py`; the metadata check in the CI job `package` | Met (D2). |
| `CITATION.cff` | `CITATION.cff`; `scripts/check_citation.py` (CI job `test`, step "Citation file is valid"); `tests/test_citation.py` | Met for merge A. Open: the creator name the owner chooses (D8) replaces the handle in item 14, if the owner approves that field; the DOI is added in item 17 if a Zenodo record is made. |
| PyPI as `glypharis` through Trusted Publishing | Job `pypi` in `.github/workflows/release.yml` (protected environment `pypi`, `id-token` on this job only, uploads the release's checked code files, rebuilds nothing); `tests/test_release_workflow.py` (`test_id_token_only_on_the_pypi_job`, `test_pypi_job_is_protected_and_neither_checks_out_nor_builds`, `test_pypi_job_uploads_only_checked_code_files`); the CI job `package` (build, `twine check`, clean install); `docs/PYPI.md` | Open: nothing has been uploaded. The owner creates the PyPI account and the pending publisher, sets up the environment `pypi` with the owner as required reviewer, and checks that the name `glypharis` is free (D8). The upload runs only after the owner publishes the release and approves the job again (item 15d). |
| Weights not in the package, fetched on first use from the release asset | `scripts/check_wheel.py` in the CI job `package`; `glypharis/hub.py` (`download_model`, refuses an unpublished name before any network call unless `url=` is given); `tests/test_hub.py`; GW-E2E-022 (`reports/package/GW-E2E-022_fetch.json`, fetch from a local server) | Met for merge A. Open: the fetch from the shipped asset is first made in GW-E2E-023 (`planned`). |
| Zenodo archive for a DOI, code and documentation only | `export-ignore` lines in `.gitattributes` for `reports/` and the evaluation manifests under `data/manifests` (D5); `scripts/check_git_archive.py` (CI job `package`, step "git archive leaves out reports and the evaluation records"); `tests/test_packaging.py` (`test_gitattributes_export_ignore_matches_the_check`, `test_git_archive_check_on_a_repository`) | Met for merge A. Open: no deposit and no DOI. The owner switches the Zenodo integration on only with a separate go, after the `git archive` check passes on `v1.0` (item 15c). The record, if made, is checked against the exclusion list (exit criterion 8). |
| Accounts and the Zenodo creator name | None in the repository, and none is expected there | Open: the owner creates the PyPI and Zenodo accounts and chooses the creator name Zenodo shows, which is permanent (D8). |
| Verify the published release from a clean environment | GW-E2E-023 (`planned`), run through `scripts/release_rehearsal.py` as GW-E2E-022 was | Open: runs only after publication (item 16). If the owner defers the release, it closes as `not run` (item 17). |

## 4. Release-ready exit criteria (PHASE_STATUS Phase 23, merge A)

| Precondition or task | Evidence | Status |
|---|---|---|
| 1. Tests and CI green on the merge commit, including the clean-wheel and `git archive` jobs; the dry run on `main` creates nothing | Section 1, rows "Tests pass", "CI passes" and "Release workflow dry run" | Open: the merge commit and the dry run come after merge A. |
| 2. The evidence table covers training, inference, export and documentation, every number traced to an experiment ID | This document (it states no metric values; it names the reports and IDs) | Met. |
| 3. Bundle byte-identical, `assert_v1_code` passes, the archive passes its hash, manifest and equivalence checks after a local fetch | GW-E2E-022: `reports/package/GW-E2E-022.json` (`bundle_manifest_sha256`, `v1_code`), `reports/package/GW-E2E-022_fetch.json` | Met. |
| 4. LICENSE, NOTICE, CITATION, CHANGELOG, release notes, version 1.0.0 and the model card agree, and `trace_numbers --check` passes | Sections 2 and 3; `scripts/check_citation.py` (compares the version and licence with the package); `scripts/trace_numbers.py --check` in the CI job `test` | Met on the branch head. Open: confirmed again on the merge commit. |
| 5. D1 to D13 recorded, and the adviser brief with the owner | D1 to D13 in PHASE_STATUS Phase 23; the deviations D2, D4 and D6 in IMPLEMENTATION_PLAN §8; `docs/ADVISER_BRIEF.md` | Decisions: met. Brief: written. Open: the owner confirms having it. The advice itself is outside the project. |

## 5. ROADMAP §4 criteria tied to the release

| Precondition or task | Evidence | Status |
|---|---|---|
| CI/CD | `.github/workflows/ci.yml` (ticked at `phase-01`); the release workflow `.github/workflows/release.yml`, which the ROADMAP assigns to Phase 23 | Open for the release part: the dry run on `main` after merge A. |
| Git history | The phase tags through `phase-22`, and `v1-code` | Open until Phase 23 closes (D12). |
| Phase documentation | `docs/PHASE_STATUS.md`, Phase 23 | Open until the completion record on `phase/23-closeout` (D12). |
| Research documentation | `research/experiments.md` (GW-E2E-022, GW-REC-046 and GW-DET-008 `done`; GW-E2E-023 `planned`), `research/research_log.md` | Open until Phase 23 closes (D12). |
| Model checkpointing, Resume training | `glypharis/training/checkpoint.py`, `scripts/rec_resume_check.py` (ticked at `phase-06`); section 1, row "checkpoint and resume" | Met, including the clean-clone resume of GW-REC-046 and GW-DET-008 (`reports/training/smoke/phase23_smoke.md`). |
| Model export | GW-EFF-006, GW-EFF-012 (section 1, row "Export verified") | Met. |
| Inference API | GW-E2E-011, with GW-E2E-013 and GW-E2E-022 on the installed package (section 1, row "Inference verified") | Met. |
| Model card, Dataset documentation | `MODEL_CARD.md`, `DATASETS.md` (ticked at `phase-21`); the package-version row of `MODEL_CARD.md` now reads `1.0.0` and says not yet released | Met. Open: both change in item 14 to state what v1.0 distributes and under which terms. |
| Reproducible final experiment | GW-E2E-018 (`done`): `reports/final_validation/v1_reproduction.md`; `configs/final/glypharis_v1.yaml`. Its strict tolerance verdict failed only on the official OmniDocBench scores; the owner's ruling of 2026-11-21 accepted that as a disclosed limitation (`docs/PHASE_STATUS.md`, Phase 22; `docs/FINAL_VALIDATION.md` section 2) | Met (owner's ruling; tag `phase-22`). |
| Unit tests | `tests/`, run by the CI job `test` | Met. |

## 6. Gaps in one place

- No committed report records a second pack of the real bundle. The byte-identity test runs on a tiny
  bundle.
- `actionlint` was not run (it would need a download); `tests/test_release_workflow.py` checks the
  workflow's structure instead (amendment to item 6). GitHub validates the file on the first dispatch.
- CI on the merge commit of pull request A, and the release workflow's dry run on `main`, come after
  merge A (item 13).
- The weights wait for qualified advice (ADR-0014, D4). Until the owner confirms the advice and gives
  the go, no weight file is uploaded, the registry's `published` flag stays off, and v1.0 is held.
- The tag `v1.0`, the draft release, the PyPI upload, the Zenodo deposit and its DOI each wait for
  their own owner's go (items 14 and 15).
- The PyPI and Zenodo accounts, the PyPI pending publisher, the protected environment `pypi` and the
  Zenodo creator name are the owner's to set up (D8).
- The verification of the published release, GW-E2E-023, runs only after publication (item 16).
