# Release evidence for v1.0

This table maps each precondition and task of Phase 23 in
[IMPLEMENTATION_PLAN](IMPLEMENTATION_PLAN.md) (Release v1.0), and the release-related success
criteria of [ROADMAP](ROADMAP.md) §4, to its evidence: a report, a test, a CI job or a document. Each
piece of evidence carries its experiment ID where it has one. Where the evidence is still missing, the
row names the gap. This is work item 12 of the Phase 23 plan in [PHASE_STATUS](PHASE_STATUS.md),
brought up to date for pull request B (item 14) and for the close-out after the release (item 17).

v1.0 is the release of v1 (GW-REC-034 with GW-DET-001): the code under Apache-2.0, published on PyPI
as `glypharis` 1.0.0, and the v1 weights as the asset `glypharis-v1.tar.gz` of the GitHub release
`v1.0`, which `glypharis.download_model` fetches and checks against pinned hashes. The weights are
distributed under CC BY-SA 4.0 with attribution to HierText and Open Images
([ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md)). The detector was initialised from
ImageNet weights, a disclosed exception to the project's commercial-use rule
([ADR-0009](decisions/ADR-0009-no-synthtext.md)), and the weights are not described as cleared for
commercial use. The owner decided on 2026-11-22 to publish them under these terms without the
qualified advice ADR-0014 had asked for; this is the project's chosen handling, not a legal
conclusion. The phone candidate is not released (D7).

**Where things stand.** v1.0 was released on 2026-11-22. Pull request A was merged as `9005263` and
pull request B as `231978e`, which carries the tag `v1.0`. Every step of
[RELEASE_RUNBOOK](RELEASE_RUNBOOK.md), 1 to 8 with the optional 6b, was carried out that day, each on
the owner's go: the GitHub release `v1.0` with the code files and the model archive, `glypharis` 1.0.0
on PyPI, the Zenodo record with DOI [10.5281/zenodo.24433864](https://doi.org/10.5281/zenodo.24433864),
and the Hugging Face mirror `DuhItzAniket/glypharis-v1`. The published release was verified from a
clean environment (GW-E2E-023). The record of each step, with its run IDs and hashes, is the progress
record "the release" in PHASE_STATUS, Phase 23.

The table gives no metric values. The results sit in the reports named here and in
`research/experiments.md` under the same IDs.

**Status words.** *Met* means the evidence exists in the repository or in a recorded CI run and
supports the row. *Open* means something is missing, and the row says what and which release-day step
it waits for. Experiment statuses are as recorded in `research/experiments.md`.

## 1. Preconditions (IMPLEMENTATION_PLAN Phase 23)

| Precondition or task | Evidence | Status |
|---|---|---|
| Tests pass | `tests/` (run by the CI job `test`, step "Pytest"). The Phase 23 additions: `tests/test_hub.py` (download, resume, wrong hash, wrong manifest, offline, URL override, two processes), `tests/test_packaging.py`, `tests/test_pack_release.py`, `tests/test_check.py`, `tests/test_citation.py`, `tests/test_release_workflow.py` | Met on `main` at `9005263` (CI run 44958358666), on pull request B (runs 44986307543 on the branch and 44986329398 on the pull request; the full local suite 2,658 passed, 2 skipped) and on `main` at its merge commit `231978e` (run 44987276181). |
| CI passes | `.github/workflows/ci.yml`: job `test` (ruff, mypy, pytest, config validation, `scripts/experiment_summary.py --check`, the comparison table check, `scripts/trace_numbers.py --check`, `scripts/check_citation.py`, the installed command in a fresh environment) and job `package` (sdist and wheel built from a clean checkout, `twine check`, `scripts/check_wheel.py`, `scripts/check_git_archive.py --commit HEAD`, the wheel installed with its dependencies in a fresh environment, and the commands, a tiny-bundle read and `glypharis-check` run from outside the checkout). Green on the branch of pull request A (run 44955100971), on the pull request (run 44955102949) and on `main` at `9005263` (run 44958358666). The earlier failures on `7eb93a7`, `baf104b` and `c4cad2c` (runs 44952673116, 44953606450 and 44954758408, and the pull request run 44954793313) came from `ruff format` checking the Python code blocks in `docs/INTEGRATION.md`, fixed in `465e103`. Green on the branch of pull request B (run 44986307543), on the pull request (run 44986329398) and on `main` at `231978e` (run 44987276181); the run on the first merge commit of pull request B (44987179276) was cancelled when that commit was re-created with the noreply identity. | Met for merge A (exit criterion 1) and for merge B. |
| Release workflow dry run creates nothing | `.github/workflows/release.yml` (stage 1 by manual dispatch, `dry_run` on by default); `tests/test_release_workflow.py`. The dry run on `main` with the would-be tag `v1.0`, run 44958368361: `check-build` passed; `draft-release` and `pypi` were skipped; no release and no tag was created. | Met (items 6 and 13). |
| Training pipeline verified: the v1 runs | GW-REC-034 (`done`): `reports/training/GW-REC-034/report.md`; GW-DET-001 (`done`): `reports/training/GW-DET-001/report.md`; configs `configs/rec/gw-rec-034.yaml`, `configs/det/gw-det-001.yaml`; `tests/test_rec_training.py`, `tests/test_det_training.py` | Met. |
| Training pipeline verified: checkpoint and resume | `glypharis/training/checkpoint.py`; `scripts/rec_resume_check.py` with `reports/training/sanity/GW-REC-001/resume_check.json`; `scripts/det_resume_check.py` with `reports/training/sanity/GW-DET-001/resume_check.json`; `reports/training/phase11/trainer_move/resume_check.json`; `tests/test_checkpoint_retry.py` | Met. |
| Training pipeline verified: from a clean clone | GW-REC-046 and GW-DET-008, both `done` (`reports/training/smoke/phase23_smoke.md`, `reports/training/GW-REC-046/`, `reports/training/GW-DET-008/`; item 11, D11) | Met: each trainer started from a clean clone, checkpointed, resumed once and wrote its report with finite losses. This shows the pipeline runs, not that it reproduces v1's training. |
| Inference verified | GW-E2E-006 (`done`, the frozen pipeline's reference): `reports/external/`; GW-E2E-011 (`done`, the package on the v1 bundle against GW-E2E-006): `reports/package/GW-E2E-011.json`; GW-E2E-013 (`done`, the clean-machine-style walkthrough): `reports/package/GW-E2E-013.json`; GW-E2E-022 (`done`, the built wheel with the fetched archive): `reports/package/GW-E2E-022.json`; GW-E2E-023 (`done`, the package from PyPI with the archive from the published release): `reports/package/GW-E2E-023.json`; GW-E2E-012 (`done`, failure paths and damaged bundles): `reports/hardening/GW-E2E-012.json`; `tests/test_api.py`, `tests/test_server.py` | Met. |
| Export verified | GW-EFF-006 (`done`, PyTorch references and the v1 fp32 ONNX bundle): `reports/efficiency/phase16/` (`reference_check.json`, `v1_bundle.json`); GW-EFF-012 (`done`, the v1 bundle on the PyTorch backend, the shipped path): `reports/export/GW-EFF-012/torch_bundle_check.json`; `scripts/export_check.py`; `tests/test_export.py` | Met. The phone candidate (GW-E2E-021, `done`, `reports/final_validation/candidate/h11.md`) is a documented result and is not released (D7). |
| Docs complete | `README.md` (install section, and the note that the full trace needs the tagged repository), `MODEL_CARD.md`, `DATASETS.md`, `docs/ARCHITECTURE.md`, `docs/REPRODUCING.md`, `docs/FINAL_VALIDATION.md`, `docs/PYPI.md` (the PyPI description), `docs/INTEGRATION.md` (the integration guide of item 12), `docs/RELEASE_NOTES_v1.0.md`, `CHANGELOG.md`, `docs/ADVISER_BRIEF.md`, `docs/RELEASE_RUNBOOK.md`, this table; CI checks `scripts/trace_numbers.py --check` and `scripts/experiment_summary.py --check` | Met: the wording states what v1.0 distributes and under which terms (item 14), and the completion record on `phase/23-closeout` closes the documentation criteria of ROADMAP §4 (D12, runbook step 8). |
| Model artifact verified: hash | Manifest pin `875357a6…` in `REGISTRY` in `glypharis/hub.py`, recorded for the v1 bundle in `reports/efficiency/phase16/v1_bundle.json` (GW-EFF-006) and matched in `reports/package/GW-E2E-022.json` (`bundle_manifest_sha256`) (GW-E2E-022). Archive pin, 82,975,077 bytes and SHA-256 `3b90c1c7…`, in the same `REGISTRY` entry, matched by the download record and the `glypharis-check` output in `reports/package/GW-E2E-022_fetch.json` (GW-E2E-022). A second pack of the real bundle, made in memory by `scripts/pack_release.py --check-only`, gave the same bytes and matched the archive, its `.sha256` file and the registry pins (`reports/release/pack_check.json`). Packer `scripts/pack_release.py`; `tests/test_pack_release.py` (two packs byte-identical, members, fixed metadata, wrong pin refused, extraction through `download_model`) | Met. The owner's decision of 2026-11-22 leaves the weight notices and the archive unchanged, so the pin stays. The archive uploaded to the release matched the pin when downloaded again (runbook step 5), and when fetched from the published address in GW-E2E-023 (`reports/package/GW-E2E-023_fetch.json`). |
| Model artifact verified: equivalence | GW-E2E-022 (`done`): the archive fetched from a local server by `download_model` into an empty models folder, then the package check against GW-E2E-006, pass (`reports/package/GW-E2E-022.json`, `reports/package/GW-E2E-022_fetch.json`; its `v1_code` field records the v1 code paths equal to `v1-code`). Earlier equivalence on the same bundle: GW-E2E-011, GW-E2E-013, GW-EFF-012. GW-E2E-023 (`done`): `glypharis` 1.0.0 from PyPI, the archive fetched from the published asset address into an empty models folder, then the same package check at the clean `v1.0` tree, pass (`reports/package/GW-E2E-023.json`, `reports/package/GW-E2E-023_fetch.json`) | Met, for the rehearsal and for the published release. |

## 2. Tasks (IMPLEMENTATION_PLAN Phase 23)

| Precondition or task | Evidence | Status |
|---|---|---|
| Release notes | `docs/RELEASE_NOTES_v1.0.md` (results cited by experiment ID; the stage 1 workflow uses it as the release's description); `CHANGELOG.md` (`[1.0.0]`) | Met: the wording states what v1.0 distributes (item 14), and the notes are the published release's description. |
| Version bump | `__version__ = "1.0.0"` in `glypharis/__init__.py`, read by `pyproject.toml` as a dynamic version; `tests/test_packaging.py` (`test_version_is_read_from_the_package`, `test_installed_metadata_matches_the_version`); the tag check `scripts/release_version_check.py` with `tests/test_release_workflow.py` (`test_v1_0_matches_1_0_0`, `test_other_version_fails`); `CITATION.cff` `version`; the package-version row of `MODEL_CARD.md`; `requires-python = ">=3.11.4,<3.12"` (D13), checked by `test_licence_python_and_project_fields`; the installed wheel's version in `reports/package/GW-E2E-022_fetch.json` (GW-E2E-022) and, for the wheel from PyPI, in `reports/package/GW-E2E-023_fetch.json` (GW-E2E-023). `model_version` stays `glypharis_v1` (D8). | Met. |
| Tagged GitHub release | `.github/workflows/release.yml`: stage 1 `check-build` and `draft-release` (manual dispatch, a draft holding the code files and `SHA256SUMS` only), stage 2 `pypi` (on a release the owner publishes by hand); `tests/test_release_workflow.py` (triggers, least permissions, actions pinned by SHA, no credentials kept, the model archive refused); the dry run on `main` (run 44958368361) | Met (2026-11-22). The annotated tag `v1.0`, with the noreply tagger, is on `231978e`, the merge commit of pull request B re-created with the noreply identity, and is never moved (runbook step 3). Stage 1 off dry run (run 44987750836) created the draft with `glypharis-1.0.0-py3-none-any.whl` (SHA-256 `0f669557…`), `glypharis-1.0.0.tar.gz` (`cdd78d74…`) and `SHA256SUMS` (step 4, item 15a). The owner published it at 2026-11-22T05:21:13Z: <https://github.com/DuhItzAniket/Glypharis/releases/tag/v1.0> (step 6). `actionlint` was replaced by the structural checks of `tests/test_release_workflow.py` (PHASE_STATUS, Phase 23, amendments); GitHub accepted the file on every dispatch. |
| The model artifact on the release (or a download script) | The archive `glypharis-v1.tar.gz`, built by `scripts/pack_release.py` outside every worktree, with the weight notices `release/weights/LICENSE-WEIGHTS` and `release/weights/NOTICE.md`; fetched by `glypharis.download_model` and by `glypharis-ocr` and `glypharis-serve` for registered names (`glypharis/hub.py`, D6; `tests/test_hub.py`); rehearsed in GW-E2E-022. The registry's `published` flag for `glypharis-v1` is switched on in pull request B (item 14). | Met (2026-11-22). On the owner's go for the weights, `glypharis-v1.tar.gz` (82,975,077 bytes, SHA-256 `3b90c1c7…`) and its `.sha256` file were uploaded to the draft from the development machine, with no overwrite, and checked by downloading them again (runbook step 5, item 15b). Since the publication they are assets of the release, and GW-E2E-023 fetched the archive from the registered address (`reports/package/GW-E2E-023_fetch.json`). |
| Licence terms that respect the training data | `LICENSE` (Apache-2.0, the code and the project's own text, D2); `NOTICE` (what Apache-2.0 does not cover, D3); `release/weights/LICENSE-WEIGHTS` and `release/weights/NOTICE.md` (CC BY-SA 4.0 with HierText and Open Images attribution and the ImageNet disclosure); ADR-0014 with its Phase 23 outcome; `DATASETS.md`; `MODEL_CARD.md`; `docs/ADVISER_BRIEF.md`; `tests/test_pack_release.py` (`test_the_weight_notices_are_outside_the_package_and_say_what_they_must`); `scripts/check_wheel.py` (no weights or weight notices in the code packages) | Code: met. Weights: decision recorded and carried out. The owner decided on 2026-11-22 to publish the weights under ADR-0014's terms without the qualified advice (D4 as changed; ADR-0014, Phase 23 outcome; exit criterion 6), and they were published under those terms, with the notices in the archive and the Hugging Face card's licence `cc-by-sa-4.0`. This is the project's chosen handling, not a legal conclusion. The adviser brief is kept as the record of the questions, which stay open; ADR-0014 still asks for advice before the consumer application is sold. |

## 3. Publishing (IMPLEMENTATION_PLAN Phase 23)

| Precondition or task | Evidence | Status |
|---|---|---|
| `LICENSE` for the code | `LICENSE`; `license = "Apache-2.0"` and `license-files` in `pyproject.toml`; `test_licence_python_and_project_fields` in `tests/test_packaging.py`; the metadata check in the CI job `package` | Met (D2). |
| `CITATION.cff` | `CITATION.cff`; `scripts/check_citation.py` (CI job `test`, step "Citation file is valid"); `tests/test_citation.py` | Met. The owner chose the GitHub handle `DuhItzAniket` as the creator name everywhere (`CITATION.cff`, the `authors` of `pyproject.toml`, Zenodo), with no real name and no email. The DOI of the Zenodo deposit, 10.5281/zenodo.24433864, is added to `CITATION.cff` on `phase/23-closeout` (runbook step 8, item 17). |
| PyPI as `glypharis` through Trusted Publishing | Job `pypi` in `.github/workflows/release.yml` (protected environment `pypi`, `id-token` on this job only, uploads the release's checked code files, rebuilds nothing); `tests/test_release_workflow.py` (`test_id_token_only_on_the_pypi_job`, `test_pypi_job_is_protected_and_neither_checks_out_nor_builds`, `test_pypi_job_uploads_only_checked_code_files`); the CI job `package` (build, `twine check`, clean install); `docs/PYPI.md`. The GitHub environment `pypi` exists, with the owner as required reviewer and deployments only from tags `v*`. | Met (2026-11-22). The owner created the PyPI pending publisher (runbook step 1). After the owner published the release, the PyPI job (run 44989039235) waited for and got the owner's approval in the environment `pypi` and uploaded the two code files through Trusted Publishing (step 6, item 15d): PyPI `glypharis` 1.0.0, licence Apache-2.0, file hashes equal to the release's `SHA256SUMS`. GW-E2E-023 installed the wheel from PyPI (`reports/package/GW-E2E-023_fetch.json`). |
| Weights not in the package, fetched on first use from the release asset | `scripts/check_wheel.py` in the CI job `package`; `glypharis/hub.py` (`download_model`, with the `glypharis-v1` entry marked published in pull request B); `tests/test_hub.py`; GW-E2E-022 (`reports/package/GW-E2E-022_fetch.json`, fetch from a local server) | Met. GW-E2E-023 (`done`) fetched the archive over the network from the shipped asset address into an empty models folder with the package installed from PyPI (`reports/package/GW-E2E-023_fetch.json`, runbook step 7). |
| Zenodo archive for a DOI, code and documentation only | `export-ignore` lines in `.gitattributes` for `reports/` and the evaluation manifests under `data/manifests` (D5); `scripts/check_git_archive.py` (CI job `package`, step "git archive leaves out reports and the evaluation records", and stage 1 of the release workflow); `tests/test_packaging.py` (`test_gitattributes_export_ignore_matches_the_check`, `test_git_archive_check_on_a_repository`) | Met (2026-11-22). Stage 1 ran the `git archive` check on `v1.0` (run 44987750836, step 4); when the owner published the release, Zenodo archived it (step 6): DOI [10.5281/zenodo.24433864](https://doi.org/10.5281/zenodo.24433864), creator `DuhItzAniket`, version `v1.0`, file `DuhItzAniket/Glypharis-v1.0.zip`. That file, checked against the exclusion list, holds 504 entries, with no `reports/`, no GDC records and no weights (step 7, exit criterion 8). |
| Hugging Face mirror `DuhItzAniket/glypharis-v1` (optional; PHASE_STATUS Phase 23, amendment to item 15) | `release/huggingface/README.md` (the mirror's model card: the files and their hashes, use, results copied from MODEL_CARD's summary with their experiment IDs, the weights' terms, limitations, citation); RELEASE_RUNBOOK step 6b (the repository created private, the published archive's eight files and the card uploaded, downloaded again and compared with the archive's members and the manifest hashes, then made public). The package does not use it: `glypharis.download_model` fetches only the GitHub release asset (`glypharis/hub.py`). | Met (2026-11-22, on the owner's go, runbook step 6b). <https://huggingface.co/DuhItzAniket/glypharis-v1> was created private; the eight members of the published archive and the card were uploaded in commit `0b8c9e8ce8a79c05a8abe8357f83e476ab23264c`; downloaded back, all eight SHA-256 values were OK and every file identical, the card included, with ten files listed; then it was made public. `manifest.json`, fetched without a token, matches the `875357a6…` pin, and the card's licence is `cc-by-sa-4.0`. |
| Accounts and the Zenodo creator name | None in the repository, and none is expected there | Creator name: decided (`DuhItzAniket`). Met (2026-11-22): the owner added the PyPI pending publisher (not visible from outside PyPI; recorded on the owner's word, and borne out by the Trusted Publishing upload of run 44989039235) and switched on the Zenodo GitHub integration (the repository's `release` webhook to Zenodo is active; it made the record of the release). |
| Verify the published release from a clean environment | GW-E2E-023 (`done`), run through `scripts/release_rehearsal.py` as GW-E2E-022 was: `reports/package/GW-E2E-023.json`, `reports/package/GW-E2E-023_fetch.json` | Met (runbook step 7, item 16): `glypharis` 1.0.0 from PyPI (wheel `0f669557…`) in a new environment holding only it, the archive fetched over the network from the registry address into empty folders, `glypharis-check` passed, and the package check on the 400 validation pages at the clean `v1.0` tree (`231978e`) passed under the rule of item 8. No channel was deferred. |

## 4. Release-ready exit criteria (PHASE_STATUS Phase 23, merge A)

| Precondition or task | Evidence | Status |
|---|---|---|
| 1. Tests and CI green on the merge commit, including the clean-wheel and `git archive` jobs; the dry run on `main` creates nothing | CI run 44958358666 on `main` at `9005263`; dry run 44958368361 (section 1) | Met. |
| 2. The evidence table covers training, inference, export and documentation, every number traced to an experiment ID | This document (it states no metric values; it names the reports and IDs) | Met. |
| 3. Bundle byte-identical, `assert_v1_code` passes, the archive passes its hash, manifest and equivalence checks after a local fetch | GW-E2E-022: `reports/package/GW-E2E-022.json` (`bundle_manifest_sha256`, `v1_code`), `reports/package/GW-E2E-022_fetch.json` | Met. |
| 4. LICENSE, NOTICE, CITATION, CHANGELOG, release notes, version 1.0.0 and the model card agree, and `trace_numbers --check` passes | Sections 2 and 3; `scripts/check_citation.py` (compares the version and licence with the package); `scripts/trace_numbers.py --check` in the CI job `test`, green on `main` at `9005263` | Met. |
| 5. D1 to D13 recorded, and the adviser brief with the owner | D1 to D13 in PHASE_STATUS Phase 23; the deviations D2, D4 and D6 in IMPLEMENTATION_PLAN §8; `docs/ADVISER_BRIEF.md`, on `main` since the owner merged pull request A | Met. |

## 5. Released exit criteria (PHASE_STATUS Phase 23, merge B and after)

| Precondition or task | Evidence | Status |
|---|---|---|
| 6. ADR-0014 revisited and its outcome recorded, or the owner's deferral recorded | ADR-0014, "Phase 23 outcome (2026-11-22)"; PHASE_STATUS Phase 23, owner decisions of 2026-11-22, night; IMPLEMENTATION_PLAN §8 | Met in pull request B. |
| 7. `v1.0` published on each approved channel and verified from a clean environment (GW-E2E-023), or recorded by the owner as deferred with the reason | RELEASE_RUNBOOK steps 3 to 7: the tag `v1.0` on `231978e`; the draft (run 44987750836); the weights upload, checked by downloading it again; the publication at 2026-11-22T05:21:13Z; the PyPI upload (run 44989039235); the Zenodo record; the optional Hugging Face mirror (step 6b); GW-E2E-023 (`reports/package/GW-E2E-023.json`, `reports/package/GW-E2E-023_fetch.json`) | Met: every approved channel published and verified; none deferred. |
| 8. The Zenodo archive, if made, matches the exclusion list, and the DOI is added afterwards | `scripts/check_git_archive.py` on `v1.0` in stage 1 (run 44987750836); the record's file checked (section 3, row "Zenodo archive"); the DOI 10.5281/zenodo.24433864 in `README.md` and `CITATION.cff` on `phase/23-closeout` | Met. |
| 9. The completion record and the ROADMAP criteria (D12) | PHASE_STATUS Phase 23, progress record "the release" and its exit criteria check, on `phase/23-closeout` (RELEASE_RUNBOOK step 8); ROADMAP §4 and M11 | Met with this close-out. `phase-23` goes on its merge commit after the identity check of step 8. |

## 6. ROADMAP §4 criteria tied to the release

| Precondition or task | Evidence | Status |
|---|---|---|
| CI/CD | `.github/workflows/ci.yml` (ticked at `phase-01`); the release workflow `.github/workflows/release.yml`, which the ROADMAP assigns to Phase 23, with its dry run on `main` (run 44958368361), stage 1 off dry run (run 44987750836) and stage 2 (run 44989039235) | Met. |
| Git history | The phase tags through `phase-22`, `v1-code`, `v1.0` on `231978e`, and `phase-23` on the close-out's merge commit | Met with this close-out (D12). |
| Phase documentation | `docs/PHASE_STATUS.md`, Phase 23, with its completion record | Met with this close-out (D12). |
| Research documentation | `research/experiments.md` (GW-E2E-022, GW-E2E-023, GW-REC-046 and GW-DET-008 `done`), `research/research_log.md` (the entry for the release) | Met with this close-out (D12). |
| Model checkpointing, Resume training | `glypharis/training/checkpoint.py`, `scripts/rec_resume_check.py` (ticked at `phase-06`); section 1, row "checkpoint and resume" | Met, including the clean-clone resume of GW-REC-046 and GW-DET-008 (`reports/training/smoke/phase23_smoke.md`). |
| Model export | GW-EFF-006, GW-EFF-012 (section 1, row "Export verified") | Met. |
| Inference API | GW-E2E-011, with GW-E2E-013, GW-E2E-022 and GW-E2E-023 on the installed package (section 1, row "Inference verified") | Met. |
| Model card, Dataset documentation | `MODEL_CARD.md`, `DATASETS.md` (ticked at `phase-21`), which state since pull request B what v1.0 distributes and under which terms | Met. |
| Reproducible final experiment | GW-E2E-018 (`done`): `reports/final_validation/v1_reproduction.md`; `configs/final/glypharis_v1.yaml`. Its strict tolerance verdict failed only on the official OmniDocBench scores; the owner's ruling of 2026-11-21 accepted that as a disclosed limitation (`docs/PHASE_STATUS.md`, Phase 22; `docs/FINAL_VALIDATION.md` section 2) | Met (owner's ruling; tag `phase-22`). |
| Unit tests | `tests/`, run by the CI job `test` | Met. |

## 7. What remains

Nothing in this table is open. The release-day steps of [RELEASE_RUNBOOK](RELEASE_RUNBOOK.md) were
all carried out on 2026-11-22, each on the owner's go:

1. The owner created the PyPI pending publisher and switched on the Zenodo GitHub integration.
2. CI green on pull request B (runs 44986307543 and 44986329398); the owner merged it.
3. The merge commit was re-created with the noreply identity as `231978e` (CI run 44987276181), and
   the tag `v1.0` was set on it.
4. Stage 1 off dry run (run 44987750836) created the draft release with the code files.
5. The archive and `glypharis-v1.tar.gz.sha256` were uploaded to the draft and checked by downloading
   them again.
6. The owner published the draft and approved the PyPI job (run 44989039235); Zenodo archived the
   release (DOI 10.5281/zenodo.24433864). Then, on its own go, the Hugging Face mirror (step 6b,
   commit `0b8c9e8…`).
7. GW-E2E-023 verified the published release from a clean environment, and the Zenodo record was
   checked against the exclusion list.
8. The close-out branch adds the DOI and the completion record (D12); `phase-23` goes on its merge
   commit after the identity check of step 8.

What stays as recorded rather than open: `actionlint` was not run (it would need a download), and
`tests/test_release_workflow.py` checks the workflow's structure instead (amendment to item 6). The
qualified advice ADR-0014 asked for was not obtained; the owner's decision of 2026-11-22 replaced the
wait for it, and ADR-0014 still asks for it before the consumer application is sold. The phone
candidate is not released (D7).
