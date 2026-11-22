# Release runbook for v1.0

The steps of release day for v1.0, in order, with what is clicked or typed at each one, what to check
afterwards and what to do when a step fails. This is Phase 23, items 14 to 17, of
[PHASE_STATUS](PHASE_STATUS.md); the evidence for each precondition is in
[RELEASE_EVIDENCE](RELEASE_EVIDENCE.md). **Every step, 1 to 8 with the optional 6b, ran on
2026-11-22, each on the owner's go;** step 8 ends when the owner merges the close-out and
`phase-23` is set on its merge commit. The steps below are kept as they were written, as the record
of how v1.0 was released, with a line under each heading saying what happened.

## Release record (2026-11-22)

| Step | Done | Record |
|---|---|---|
| 1 | Owner | PyPI pending publisher added; Zenodo GitHub integration on, its `release` webhook active |
| 2 | Owner | Pull request B (#30) merged after CI green on the branch (run 44986307543) and the pull request (run 44986329398) |
| 3 | Session | Merge commit re-created with the noreply identity as `231978e`, same tree and parents; CI on `main` green (run 44987276181); annotated tag `v1.0`, noreply tagger, on `231978e` |
| 4 | Session | Stage 1 off dry run, run 44987750836: draft with `glypharis-1.0.0-py3-none-any.whl` (SHA-256 `0f669557b86d7385a218112a79059ae20109f8ef10a9b13990d131dd30edca05`), `glypharis-1.0.0.tar.gz` (`cdd78d7432b14416cdca4f849e8b0b27563dfd050b93a37d3ae450562d62eb72`) and `SHA256SUMS` |
| 5 | Session | `glypharis-v1.tar.gz` (82,975,077 bytes, SHA-256 `3b90c1c74147747d7474a86b550e198f0df83d447961bd6aec88d4c24846541d`) and its `.sha256` uploaded to the draft without overwrite, and checked by downloading them again |
| 6 | Owner | Release published at 2026-11-22T05:21:13Z (<https://github.com/DuhItzAniket/Glypharis/releases/tag/v1.0>); PyPI job, run 44989039235, approved in the environment `pypi`: `glypharis` 1.0.0 on PyPI, Apache-2.0, file hashes equal to `SHA256SUMS`; Zenodo record DOI [10.5281/zenodo.24433864](https://doi.org/10.5281/zenodo.24433864), creator `DuhItzAniket`, version `v1.0` |
| 6b | Session | <https://huggingface.co/DuhItzAniket/glypharis-v1>: created private, the eight members of the published archive and the card uploaded in commit `0b8c9e8ce8a79c05a8abe8357f83e476ab23264c`, downloaded back with all eight SHA-256 values OK and every file identical, ten files listed, then made public; `manifest.json` without a token matches the `875357a6…` pin; licence `cc-by-sa-4.0` |
| 7 | Session | GW-E2E-023 passed (`reports/package/GW-E2E-023.json`, `reports/package/GW-E2E-023_fetch.json`); the Zenodo file `DuhItzAniket/Glypharis-v1.0.zip` holds 504 entries, with no `reports/`, no GDC records and no weights |
| 8 | Session, then Owner | `phase/23-closeout`: the DOI in `README.md` and `CITATION.cff`, the completion record in PHASE_STATUS, the ROADMAP criteria; `phase-23` goes on its merge commit after the identity check |

The full record, with the known issues, is the progress record "the release" in PHASE_STATUS, Phase
23.

What gets released: v1 (GW-REC-034 with GW-DET-001). The code goes out under Apache-2.0 as the
package `glypharis` 1.0.0 on PyPI, and the v1 weights go out as the asset `glypharis-v1.tar.gz` of
the GitHub release `v1.0`, which `glypharis.download_model` fetches and checks against the pins in
`glypharis/hub.py`. The weights are distributed under CC BY-SA 4.0 with attribution to HierText and
Open Images ([ADR-0014](decisions/ADR-0014-weights-and-backbone-terms.md)). The detector was
initialised from ImageNet weights, a disclosed exception to the project's commercial-use rule
([ADR-0009](decisions/ADR-0009-no-synthtext.md)), and the weights are not described as cleared for
commercial use. The phone candidate is not released (D7).

**Who does what.** **Owner** steps are done by the owner, in a browser, on the owner's own accounts.
**Session** steps are commands run from the development machine with the owner's `gh` login, each
one only on the owner's go for it (PHASE_STATUS, Phase 23, Fixed rules). One go never covers
another: publishing the GitHub release is not the PyPI approval, and the weights upload has its own
go.

**Rules that hold on every step**

- No asset is replaced. `gh release upload` is never given `--clobber`, and nothing is deleted from a
  published release.
- A PyPI version cannot be uploaded twice. Once PyPI holds a file for `glypharis` 1.0.0, that file
  name can never be used again with other bytes, even after deletion.
- The tag `v1.0` goes on the merge commit of pull request B and is never moved.
- Every local folder is kept: the worktrees, the archive folder, the bundle under `runs/release/`
  and the v1 backup.
- No personal path, name or email goes into a tracked file, a release note or a commit. The archive
  is written here as `<archive folder>/glypharis-v1.tar.gz`.
- When a check fails, stop at that step and take it to the owner. Steps are not repeated or worked
  around without the owner's decision.

## Summary

| Step | Who | What | Waits for |
|---|---|---|---|
| 1 | Owner | PyPI pending publisher; Zenodo GitHub integration switched on | — |
| 2 | Owner | Merge pull request B | CI green on pull request B |
| 3 | Session | Merge commit identity checked (re-created with the noreply identity if needed); tag `v1.0` set on it | step 2; the owner's go |
| 4 | Session (or Owner) | Release workflow, stage 1 off dry run: the draft release with the code files | step 3; the owner's go |
| 5 | Session | Archive and its `.sha256` uploaded to the draft and checked by downloading them again | step 4; the owner's go for the weights |
| 6 | Owner | Draft published; the PyPI job approved in the environment `pypi`; Zenodo archives the release | step 5 |
| 6b | Session | Optional Hugging Face mirror `DuhItzAniket/glypharis-v1`: created private, the archive's files and the card uploaded, checked by downloading them again, then made public | step 6; the owner's go for the mirror |
| 7 | Session | GW-E2E-023 from a clean environment; the Zenodo record checked | step 6; the owner's go for the downloads |
| 8 | Session, then Owner | Close-out branch: the DOI and the completion record; the owner merges it | step 7, or a recorded deferral |

## Before release day

Already in place: pull request A is merged (merge commit `9005263`), the release workflow's dry run on
`main` built and checked and created nothing (run 44958368361), and the GitHub environment `pypi`
exists with the owner as required reviewer and deployments only from tags `v*`.

Session checks, each read-only:

```sh
gh auth status
gh api repos/DuhItzAniket/Glypharis/environments/pypi \
  --jq '{reviewers: [.protection_rules[].reviewers[]?.reviewer.login], policy: .deployment_branch_policy}'
gh api repos/DuhItzAniket/Glypharis/environments/pypi/deployment-branch-policies --jq '.branch_policies[] | {name, type}'
gh release list --repo DuhItzAniket/Glypharis
git ls-remote --tags origin v1.0
```

*Check:* `gh` is logged in as `DuhItzAniket`; the environment lists the owner as reviewer and one tag
policy `v*`; neither a release nor a tag `v1.0` exists. The archive and its `.sha256` file are in
`<archive folder>` and match the registry: 82,975,077 bytes and SHA-256
`3b90c1c74147747d7474a86b550e198f0df83d447961bd6aec88d4c24846541d` (the `glypharis-v1` entry of
`REGISTRY` in `glypharis/hub.py`; matched after a local fetch in
`reports/package/GW-E2E-022_fetch.json`, GW-E2E-022):

```sh
cd "<archive folder>"
wc -c < glypharis-v1.tar.gz
sha256sum -c glypharis-v1.tar.gz.sha256
```

To check the archive against the bundle as well, pack it again in memory and compare, as recorded in
`reports/release/pack_check.json`. This writes nothing; run it from the checkout that holds the
bundle under `runs/release/`:

```sh
python scripts/pack_release.py --bundle runs/release/glypharis-v1 --out "<archive folder>" --check-only
```

## Step 1 — PyPI pending publisher and Zenodo integration (Owner)

Done on 2026-11-22: the owner set up both; the Zenodo `release` webhook on the repository is active.

**PyPI.** On pypi.org, logged in to the owner's account:

1. Open **Your account > Publishing**.
2. Under **Add a new pending publisher**, choose the **GitHub** tab.
3. Fill in exactly:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `glypharis` |
   | Owner | `DuhItzAniket` |
   | Repository name | `Glypharis` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

4. Click **Add**.

If PyPI asks for two-factor authentication or a verified email address first, set those up and come
back. *Check:* the publisher is listed under **Pending publishers** with these five values, and
`https://pypi.org/project/glypharis/` still shows "Not Found", so the name is free. A pending
publisher does not reserve the name: if someone else registers `glypharis` before step 6, the
publisher no longer applies, and the owner decides what to do.

**Zenodo.** On zenodo.org:

1. Click **Log in**, then **Log in with GitHub**, and authorise Zenodo for the owner's GitHub
   account.
2. In the account menu, open **GitHub** (the GitHub settings page).
3. If the repository is not listed, click **Sync now**.
4. Switch **DuhItzAniket/Glypharis** on.

*Check:* the switch shows the repository as on, and the repository's **Settings > Webhooks** on GitHub
lists an active zenodo.org hook for `release` events. Session:

```sh
gh api repos/DuhItzAniket/Glypharis/hooks --jq '.[] | {id, active, events, host: (.config.url | split("?")[0])}'
```

The hook's full URL carries a Zenodo access token in its query string. Never print it, paste it or
copy it into a record, log or session note.

Switching it on deposits nothing by itself: Zenodo archives a release only when it is published
(step 6), and by then stage 1 has run the `git archive` check on `v1.0` (step 4), as item 15c asks.
Zenodo takes the creator name from `CITATION.cff`, which names the GitHub handle `DuhItzAniket` only.

*If it fails:* nothing has been released, so nothing needs undoing. If the Zenodo integration is left
off, the release goes ahead without a deposit; a later deposit is made by hand on its own go, or the
close-out records it as deferred (item 17).

## Step 2 — Merge pull request B (Owner)

Done on 2026-11-22: CI was green on the branch (run 44986307543) and on pull request #30 (run
44986329398), and the owner merged it.

When CI is green on pull request B, the owner merges it on GitHub with a merge commit, as for the
earlier phase pull requests.

*Check (Session):* CI on `main` at the merge commit is green:

```sh
git fetch origin
gh run list --repo DuhItzAniket/Glypharis --branch main --workflow CI --limit 3
```

*If it fails:* nothing is tagged. A fix goes on a new branch and pull request; steps 3 onwards wait
for it.

## Step 3 — Merge commit identity and the tag `v1.0` (Session)

Done on 2026-11-22: the merge commit was not under the noreply identity, so it was re-created with
the noreply identity as `231978e`, with the same tree and parents, and `main` was moved to it. CI on
`main` at `231978e` was green (run 44987276181). The annotated tag `v1.0`, with the noreply tagger,
is on `231978e`.

Check who the merge commit names. The author and committer are compared in a shell test against the
noreply identity, and only the answer is printed, so a merge made under another identity does not put
that name or address into a record:

```sh
git fetch origin
NOREPLY="DuhItzAniket <209354142+DuhItzAniket@users.noreply.github.com>"
git log -1 --format='%H%n%P' origin/main
if test "$(git log -1 --format='%an <%ae>' origin/main)" = "$NOREPLY" &&
   test "$(git log -1 --format='%cn <%ce>' origin/main)" = "$NOREPLY"; then
  echo noreply
else
  echo "needs re-creation"
fi
```

If it prints `needs re-creation`, the author or committer is not the owner's GitHub noreply
identity, and the merge commit is re-created with the noreply identity, with the same tree, parents,
message and dates, and `main` is moved to it, as was done for pull request A (`9005263`). This
rewrites the head of `main`, so it needs the owner's go, and it happens before anything is tagged:

```sh
OLD=$(git rev-parse origin/main)
MSG="<folder outside the checkout>/merge-message.txt"
git log -1 --format=%B "$OLD" > "$MSG"
NEW=$(GIT_AUTHOR_DATE="$(git log -1 --format=%aI "$OLD")" \
      GIT_COMMITTER_DATE="$(git log -1 --format=%cI "$OLD")" \
      git commit-tree "$OLD^{tree}" -p "$OLD^1" -p "$OLD^2" -F "$MSG")
test "$(git rev-parse "$NEW^{tree}")" = "$(git rev-parse "$OLD^{tree}")" &&
  test "$(git log -1 --format=%P "$NEW")" = "$(git log -1 --format=%P "$OLD")" &&
  git push --force-with-lease=main:"$OLD" origin "$NEW":refs/heads/main
```

The push runs only if both checks pass. `git commit-tree` takes the author and committer from the local git identity, which must be the
noreply one (`git config user.name` and `git config user.email`). Then set the tag on the merge
commit and push it:

```sh
MERGE=$(git rev-parse origin/main)
git tag -a v1.0 -m "GLYPHARIS v1.0" "$MERGE"
git push origin v1.0
```

*Check:* `git ls-remote --tags origin 'v1.0*'` shows `refs/tags/v1.0^{}` at the merge commit, and the
tag's tagger is the noreply identity (`git cat-file -p v1.0`). A tag push starts no workflow.

*If it fails:* before the push, fix it locally and check again. Once `v1.0` is pushed it is not moved;
if it turns out to point at the wrong commit, stop and take it to the owner before step 4.

## Step 4 — Draft release from the tag (Session, or Owner)

Done on 2026-11-22: run 44987750836 succeeded and created the draft with
`glypharis-1.0.0-py3-none-any.whl` (SHA-256 `0f669557…`), `glypharis-1.0.0.tar.gz` (`cdd78d74…`) and
`SHA256SUMS`.

Stage 1 of `.github/workflows/release.yml`, off dry run, builds from the tag, runs every check
(the tag against the package version, the build from a clean checkout, `twine check`,
`scripts/check_wheel.py`, `scripts/check_git_archive.py --commit HEAD`, the clean install) and creates
a **draft** release holding the sdist, the wheel and `SHA256SUMS` only.

Session:

```sh
gh workflow run release.yml --repo DuhItzAniket/Glypharis --ref v1.0 -f dry_run=false -f tag=v1.0
gh run list --repo DuhItzAniket/Glypharis --workflow release.yml --limit 1
gh run watch <run id> --repo DuhItzAniket/Glypharis --exit-status
```

Or the owner, on GitHub: **Actions > Release > Run workflow**, choose the tag `v1.0` under **Use
workflow from**, untick **Dry run: build and check only, create nothing**, set **Release tag** to
`v1.0`, and click **Run workflow**.

*Check:* the jobs `build and check (stage 1)` and `draft GitHub release (stage 1, dry_run off)`
succeeded, and the PyPI job was skipped. Then:

```sh
gh release view v1.0 --repo DuhItzAniket/Glypharis --json isDraft,tagName,assets \
  --jq '{isDraft, tagName, assets: [.assets[].name]}'
```

shows `isDraft: true`, the tag `v1.0`, and exactly three files: `glypharis-1.0.0.tar.gz`,
`glypharis-1.0.0-py3-none-any.whl` and `SHA256SUMS`. A draft is visible only to people with write
access to the repository.

*If it fails:* if `build and check` fails, nothing is created; the tag stays, and the owner decides
what follows. If a draft was created before the run failed, stop: the draft is not public, and the
owner decides whether it is deleted before the workflow runs again (the workflow refuses while any
release for `v1.0` exists).

## Step 5 — Upload the weights to the draft and check them (Session)

Done on 2026-11-22, on the owner's go for the weights: both files were uploaded from the
development machine and checked by downloading them again: the size and SHA-256 equal the pin, and
the downloaded files are identical to the uploaded ones.

This is the first time a weight file leaves the development machine, and it needs the owner's go for
the weights. First check the local files again (see "Before release day"), then:

```sh
gh release upload v1.0 --repo DuhItzAniket/Glypharis \
  "<archive folder>/glypharis-v1.tar.gz" "<archive folder>/glypharis-v1.tar.gz.sha256"
```

Without `--clobber`, `gh` refuses a file name the release already holds. Check the upload by
downloading both files again into a new, empty folder, since a draft's assets are not public:

```sh
gh release download v1.0 --repo DuhItzAniket/Glypharis --dir "<new empty folder>" \
  --pattern glypharis-v1.tar.gz --pattern glypharis-v1.tar.gz.sha256
cd "<new empty folder>"
wc -c < glypharis-v1.tar.gz
sha256sum -c glypharis-v1.tar.gz.sha256
```

*Check:* 82,975,077 bytes and `glypharis-v1.tar.gz: OK`, the SHA-256 being
`3b90c1c74147747d7474a86b550e198f0df83d447961bd6aec88d4c24846541d` (the registry pin, GW-E2E-022);
`gh release view v1.0 --repo DuhItzAniket/Glypharis --json assets --jq '[.assets[].name]'` lists
five files (`--repo` is needed, since this runs from the new folder, outside the checkout).

*If it fails:* do not publish the draft and do not use `--clobber`. Stop and take it to the owner: the
draft is not public, and the owner decides whether the failed file is removed from the draft
(`gh release delete-asset v1.0 <file name> --repo DuhItzAniket/Glypharis`) before it is uploaded
again. If the weights cannot go up at all, the draft
stays unpublished. The code built from `v1.0` has the registry's `published` flag on, so a release
without the archive would ship a package that fetches a file that does not exist; a code-only release
needs a new owner decision (D4).

## Step 6 — Publish, approve PyPI, Zenodo archives (Owner)

Done on 2026-11-22: the owner published the release at 05:21:13Z; the PyPI job (run 44989039235)
was approved in the environment `pypi` and uploaded the two code files, whose PyPI hashes equal
`SHA256SUMS`; Zenodo made the record with DOI 10.5281/zenodo.24433864, creator `DuhItzAniket`,
version `v1.0`.

1. On GitHub, open **Releases**, the draft `v1.0`, and click **Edit**. Check the five files and the
   notes, then click **Publish release**.
2. Publishing starts the Release workflow again, now stage 2. Its PyPI job waits in the environment
   `pypi`. Open **Actions > Release** (the run started by the release), click **Review deployments**,
   tick `pypi` and click **Approve and deploy**. The job downloads `SHA256SUMS` and the two code files
   from the release, checks them and uploads exactly those through Trusted Publishing; nothing is
   rebuilt, and the model archive is never sent to PyPI.
3. Zenodo receives the published release through its webhook and archives the source archive of the
   tag.

*Check (Session):*

```sh
gh release view v1.0 --repo DuhItzAniket/Glypharis --json isDraft,assets --jq '{isDraft, n: (.assets | length)}'
curl -sIL https://github.com/DuhItzAniket/Glypharis/releases/download/v1.0/glypharis-v1.tar.gz
gh run list --repo DuhItzAniket/Glypharis --workflow release.yml --limit 1
```

The release is no longer a draft and holds five files; the asset address answers 200 with a length of
82,975,077 bytes; the PyPI job succeeded. `https://pypi.org/project/glypharis/1.0.0/` lists the wheel
and the sdist, with SHA-256 values equal to the lines of `SHA256SUMS`. On Zenodo, the GitHub settings
page lists the release with a DOI, and the record names `DuhItzAniket` as creator and holds the source
archive only, not `glypharis-v1.tar.gz`.

Between the merge of pull request B and this step, `download_model("glypharis-v1")` from `main` fails
with a download error, because the asset is not public until the release is published. That is
expected.

*If it fails:*

- *The PyPI job fails before anything is uploaded* (for example, PyPI rejects the publisher): fix the
  pending publisher so its five fields match step 1 exactly, then **Re-run failed jobs** on that run
  and approve it again. The same checked files go up; nothing is rebuilt.
- *PyPI already holds one file of 1.0.0 and the job failed on the other:* stop and take it to the
  owner. A file PyPI holds is never uploaded again with other bytes, and the version cannot be
  uploaded a second time.
- *Zenodo shows an error for the release, or made no record:* do not unpublish or re-create the
  release to try again. The release stands; a deposit is made by hand later on its own go, or the
  close-out records Zenodo as deferred (item 17).
- *The Zenodo record holds the weights archive, or names another creator:* stop and take it to the
  owner. Zenodo lets the owner edit a record's metadata, not its files.
- No published asset is replaced in any of these cases.

## Step 6b — Hugging Face mirror (Session, on the owner's go)

Done on 2026-11-22, on the owner's go: commit `0b8c9e8ce8a79c05a8abe8357f83e476ab23264c`, checked
by downloading it back while private (all eight lines `OK`, every file identical, ten files listed),
then made public; `manifest.json` fetched without a token matches the pin. One line of the upload's
summary read "0 committed" although the commit landed; the download check is the record.

Optional (PHASE_STATUS, Phase 23, amendment to item 15). The owner asked for a mirror of the v1
weights on Hugging Face, in the model repository `DuhItzAniket/glypharis-v1`. It holds the eight
files of the published archive byte for byte, with `release/huggingface/README.md` at `v1.0` as its
model card (`README.md`). The package does not use it: `glypharis.download_model` keeps fetching
the GitHub release asset, and `glypharis/hub.py` does not name the mirror. The step waits for step
6, so that the mirror copies the published asset rather than a local file, and it has its own go:
the go for the weights in step 5 does not cover it. The repository is created private, filled and
checked, and only then made public, as the release went from draft to published.

The commands are those of the `hf` CLI of `huggingface_hub` 2.x, as `hf <command> --help` gives them
in version 2.4.0, logged in as the owner. In Git Bash, give folders as `/c/...` paths.

Before, read-only:

```sh
hf version
hf auth whoami
hf models info DuhItzAniket/glypharis-v1
```

*Check:* `hf auth whoami` names `DuhItzAniket`, and `hf models info` answers that the repository
does not exist.

**1. The published archive, checked.** Into a new, empty folder:

```sh
gh release download v1.0 --repo DuhItzAniket/Glypharis --dir "<new empty folder A>" \
  --pattern glypharis-v1.tar.gz --pattern glypharis-v1.tar.gz.sha256
cd "<new empty folder A>"
wc -c < glypharis-v1.tar.gz
sha256sum -c glypharis-v1.tar.gz.sha256
sha256sum glypharis-v1.tar.gz
```

*Check:* 82,975,077 bytes, `glypharis-v1.tar.gz: OK`, and the SHA-256
`3b90c1c74147747d7474a86b550e198f0df83d447961bd6aec88d4c24846541d` (the registry pin, GW-E2E-022).

**2. The folder to upload.** Write the list of the eight files' hashes once, outside the checkout
(the same list as in the card's section "Verifying the files": the manifest pin, the four weight
files as `manifest.json` lists them, and the notices and `model_info.json` as
`reports/release/pack_check.json` records them):

```sh
cat > "<folder outside the checkout>/glypharis-v1-files.sha256" <<'EOF'
875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037  manifest.json
caafdce9402de07a775102558eeffa6f48d4f54cd370ac63c09ab0be2a929c99  detector.pt
378c1feda480c1bb1dde838dd4a5f4579a904710a7f7b222d978c583b5bddcb7  detector.onnx
785409f75f1d31e4933a42ddcae8c9ecb84960d975083f3f81ebc47d6070f9ca  recognizer.pt
930329b4bf635a935975823d2004b47371d9123b2504cfe590d52f44db84f438  recognizer.onnx
81ceaf22681fe1bd55c5e85b4a72243d1652b631b38769e42947f988413fe79d  LICENSE-WEIGHTS
8bc9869cb7e5f3b7dd45416103a93fa2a362ba90151f439ba9d1363d26317d65  NOTICE.md
c41f5a45f0e7747f5ff604ea45ea8097d520c05bd8740c83364a16c84ae03508  model_info.json
EOF
```

Then extract the archive into a second new, empty folder and add the card as it is at the tag:

```sh
cd "<new empty folder A>"
tar -xzf glypharis-v1.tar.gz -C "<new empty folder B>"
git -C "<checkout>" show v1.0:release/huggingface/README.md > "<new empty folder B>/glypharis-v1/README.md"
cd "<new empty folder B>/glypharis-v1"
ls -A
sha256sum -c "<folder outside the checkout>/glypharis-v1-files.sha256"
```

*Check:* `ls -A` lists `README.md` and the eight members named in `reports/release/pack_check.json`
(`LICENSE-WEIGHTS`, `NOTICE.md`, `detector.onnx`, `detector.pt`, `manifest.json`, `model_info.json`,
`recognizer.onnx`, `recognizer.pt`), and nothing else; all eight lines are `OK`.

**3. Create the repository, private.**

```sh
hf repos create DuhItzAniket/glypharis-v1 --repo-type model --private
```

Hugging Face starts the repository with a `.gitattributes` file of its own.

**4. Upload the folder** to the top of the repository, in one commit:

```sh
hf upload DuhItzAniket/glypharis-v1 "<new empty folder B>/glypharis-v1" . --repo-type model \
  --commit-message "Mirror the GLYPHARIS v1 weights of the GitHub release v1.0"
```

**5. Download it again and compare**, into a third new, empty folder, while the repository is still
private (the owner's login can read it):

```sh
hf download DuhItzAniket/glypharis-v1 --repo-type model --local-dir "<new empty folder D>"
cd "<new empty folder D>"
ls -A
sha256sum -c "<folder outside the checkout>/glypharis-v1-files.sha256"
for f in LICENSE-WEIGHTS NOTICE.md README.md detector.onnx detector.pt manifest.json \
         model_info.json recognizer.onnx recognizer.pt; do
  cmp "$f" "<new empty folder B>/glypharis-v1/$f" || echo "DIFFERS: $f"
done
hf models info DuhItzAniket/glypharis-v1 --expand private,sha,siblings --json
```

*Check:* `ls -A` lists `.cache` (written by `hf download`), `.gitattributes`, `README.md` and the
eight members, and nothing else; all eight `sha256sum` lines are `OK`; the loop prints nothing, so
every file equals the archive's member and `README.md` equals the card at `v1.0`; `hf models info`
shows `private` true, the same ten files as `siblings`, and the commit `sha`, which goes into the
completion record (step 8). The model page on huggingface.co shows the card and the licence
`cc-by-sa-4.0`.

**6. Make it public.**

```sh
hf repos settings DuhItzAniket/glypharis-v1 --public
hf models info DuhItzAniket/glypharis-v1 --expand private,sha --json
curl -sIL https://huggingface.co/DuhItzAniket/glypharis-v1/resolve/main/manifest.json
```

In `hf` 2.4.0 `--public` is the flag that changes the visibility; `--no-private` leaves it as it is.

*Check:* `private` false, the same `sha` as in part 5, and the `curl` request, made without a token, ends
in 200.

*If it fails:*

- *Part 1 or 2 fails* (the published asset does not match the pin, or a member is missing or
  different): stop before anything is created on Hugging Face and take it to the owner. That is a
  problem with the GitHub release itself, and step 7 would fail too.
- *`hf repos create` fails* (another login, or the name is not available): nothing has been
  uploaded. The owner decides; the card names `DuhItzAniket/glypharis-v1`, so another name means
  changing the card first.
- *The upload stops part way:* run the same `hf upload` command again, which resumes it. If the Hub
  refuses the commit, for example over the card's metadata, nothing is committed; take it to the
  owner, since a corrected card is no longer the one at `v1.0`, and record the change.
- *The downloaded copy differs, or the repository holds other files:* do not make it public. Unlike a
  PyPI version or a published GitHub asset, a Hugging Face repository can be changed: a file can be
  replaced by a new commit, or the whole repository deleted (`hf repos delete
  DuhItzAniket/glypharis-v1`, which cannot be undone). A replaced file stays in the repository's
  history, so while the repository is private the clean fix is to delete it and start again from
  part 3. Either needs the owner's go.
- *A problem is found after it is public:* make it private again (`hf repos settings
  DuhItzAniket/glypharis-v1 --private`) and take it to the owner. The GitHub release, PyPI and the
  package are not affected, since none of them uses the mirror.
- In every case, record what happened, with the commands run and the commit `sha`, in the
  completion record (step 8). If the mirror is not made, the close-out says so. Apart from the card,
  which is written for the mirror itself, no document says that the mirror exists, so leaving it out
  needs no change of wording.

## Step 7 — Verify the published release, GW-E2E-023 (Session)

Done on 2026-11-22: GW-E2E-023 passed (`reports/package/GW-E2E-023.json`,
`reports/package/GW-E2E-023_fetch.json`), and the Zenodo file `DuhItzAniket/Glypharis-v1.0.zip`
holds 504 entries, with no `reports/`, no GDC records and no weights. The third-party packages were
the locked ones already on the machine, as in GW-E2E-022, with `glypharis` 1.0.0 itself from PyPI.

From a clean environment, as GW-E2E-022 was run (item 16), on the owner's go for the downloads
(PyTorch among the dependencies):

```sh
"<Python 3.11>" -m venv "<new venv folder>"
"<venv python>" -m pip install glypharis==1.0.0
```

Then, from a checkout at `v1.0` (`--v1` compares its v1 paths with `v1-code`), with `PYTHONPATH` unset
and the models folder and the home folders new and empty:

```sh
GLYPHARIS_MODELS="<new empty folder 1>" HOME="<new empty folder 2>" USERPROFILE="<new empty folder 2>" \
  "<venv python>" scripts/release_rehearsal.py --experiment GW-E2E-023 \
  --phase10 "<GW-E2E-006 hiertext_val.jsonl>" --out reports/package/GW-E2E-023.json
```

The script refuses unless `glypharis` is imported from the installed package and the folders are
empty. Without `--serve` it fetches from the registry's own address, the asset of step 6.

*Check:* exit status 0. `reports/package/GW-E2E-023_fetch.json` records a network fetch from
`https://github.com/DuhItzAniket/Glypharis/releases/download/v1.0/glypharis-v1.tar.gz`, the installed
version `1.0.0`, the archive and manifest pins matched, and `glypharis-check` passed.
`reports/package/GW-E2E-023.json` passes the rule of item 8: page CER within 0.05 points of GW-E2E-006,
and every line's text and box identical.

Then the Zenodo record, if one was made: download its source archive into a new, empty folder, list
it, and check that none of the paths in `EXCLUDED` of `scripts/check_git_archive.py` (`reports/` and
the evaluation records under `data/manifests`) is in it, and that the code and documents are.

*If it fails:* the result is recorded as it came out, under GW-E2E-023 in `research/experiments.md`,
and taken to the owner. No published file is replaced.

## Step 8 — Close-out (Session, then Owner)

Done on 2026-11-22 up to the owner's merge: `phase/23-closeout` carries the DOI in `README.md` and
`CITATION.cff`, the completion record, the ROADMAP criteria and the research log entry. The merge,
the identity check below and `phase-23` follow.

On a new branch `phase/23-closeout` from `main` (D12): the DOI goes into `README.md` and
`CITATION.cff` if there is one; the completion record goes into PHASE_STATUS with the merge commit,
the tag, the run IDs, the PyPI and Zenodo records, the Hugging Face mirror's commit if it was made
(step 6b) and GW-E2E-023; the ROADMAP criteria are ticked; the
research log gets its entry. The owner merges it, and `phase-23` goes on that merge commit.

Before `phase-23` is tagged, the close-out merge commit gets step 3's identity check again, printing
only the answer:

```sh
git fetch origin
NOREPLY="DuhItzAniket <209354142+DuhItzAniket@users.noreply.github.com>"
git log -1 --format='%H%n%P' origin/main
if test "$(git log -1 --format='%an <%ae>' origin/main)" = "$NOREPLY" &&
   test "$(git log -1 --format='%cn <%ce>' origin/main)" = "$NOREPLY"; then
  echo noreply
else
  echo "needs re-creation"
fi
```

If it prints `needs re-creation`, the merge commit is re-created with the noreply identity as in
step 3, with the same tree, parents, message and dates, the tree and parents are checked, and `main`
is moved to it. The push rewrites the head of `main`, so it is made on the owner's go; the owner has
given that go in advance for re-creating merge commits.

```sh
OLD=$(git rev-parse origin/main)
MSG="<folder outside the checkout>/closeout-merge-message.txt"
git log -1 --format=%B "$OLD" > "$MSG"
NEW=$(GIT_AUTHOR_DATE="$(git log -1 --format=%aI "$OLD")" \
      GIT_COMMITTER_DATE="$(git log -1 --format=%cI "$OLD")" \
      git commit-tree "$OLD^{tree}" -p "$OLD^1" -p "$OLD^2" -F "$MSG")
test "$(git rev-parse "$NEW^{tree}")" = "$(git rev-parse "$OLD^{tree}")" &&
  test "$(git log -1 --format=%P "$NEW")" = "$(git log -1 --format=%P "$OLD")" &&
  git push --force-with-lease=main:"$OLD" origin "$NEW":refs/heads/main
```

The push runs only if both checks pass; if either fails, nothing is pushed, and it goes to the owner.

Then tag the final commit of `main` (the merge commit, or its re-creation) with an annotated tag
whose message summarises the completion record ([WORKFLOW](WORKFLOW.md) §1), check the tagger
before the tag is pushed, and push it:

```sh
git fetch origin
FINAL=$(git rev-parse origin/main)
git tag -a phase-23 -F "<folder outside the checkout>/phase-23-tag-message.txt" "$FINAL"
if test "$(git for-each-ref --format='%(taggername) %(taggeremail)' refs/tags/phase-23)" = "$NOREPLY"; then
  echo "tagger: noreply"
  git push origin phase-23
else
  echo "tagger: not noreply"
fi
```

*Check:* the identity check printed `noreply` (directly, or after the re-creation when run again),
the tagger check printed `tagger: noreply` and pushed the tag, and `git ls-remote --tags origin
'phase-23*'` shows `refs/tags/phase-23^{}` at the final commit. If the tagger check prints `tagger:
not noreply`, nothing was pushed: delete the local tag (`git tag -d phase-23`), set the noreply
identity (`git config user.name` and `git config user.email`) and tag again.

**If a channel is deferred.** Pull request B states what v1.0 distributes on every channel. If one of
them does not happen (PyPI, Zenodo or the weights), the close-out records which channel was deferred
and why, restores the wording for that channel (item 17), and GW-E2E-023 closes as `not run` if the
release cannot be verified.
