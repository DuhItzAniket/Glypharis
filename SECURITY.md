# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |
| Earlier versions (v0.1 and the development tags) | No |

Fixes are made on the latest 1.0.x release. Please check that a problem still occurs with the latest
release before you report it.

## Reporting a vulnerability

Report security problems privately, through GitHub's private vulnerability reporting on this
repository: open the **Security** tab and choose **Report a vulnerability**, or go straight to
<https://github.com/DuhItzAniket/Glypharis/security/advisories/new>. Please do not open a public
issue, pull request or discussion for a security problem. The project does not take reports by
email.

A useful report says:

- the affected version (the output of `glypharis.version_info()`) and the operating system;
- which part is affected (see the scope below) and what an attacker could do;
- the steps or a minimal input that shows the problem. If the input is an image or a model bundle,
  attach it to the private report, not to a public issue.

The report is acknowledged in the advisory thread. When a fix is ready, it is released as a new
1.0.x version, and the advisory is published with credit to the reporter unless they prefer
otherwise.

## Scope

In scope:

- **The `glypharis` package**: reading images and bytes, the `glypharis-ocr` and `glypharis-check`
  commands, and loading a model bundle with `GlypharisOCR.from_pretrained` (for example a crafted
  image or bundle that causes code execution, files written outside the expected folder, or
  unbounded memory use).
- **The server, `glypharis-serve`**: request handling, the size and pixel limits, and error
  responses. The server is an integration boundary for local use, without authentication or TLS,
  and listens on 127.0.0.1 by default ([docs/INTEGRATION.md](docs/INTEGRATION.md)). Running it
  exposed to a network without a proxy in front is outside its design; a way to break its limits or
  to reach code execution through it is in scope.
- **The model download and its hash checks** (`glypharis.download_model`, `glypharis/hub.py`):
  anything that lets an archive or bundle other than the pinned one be installed, or that lets
  extraction write outside the models folder.
- **The release workflow** ([.github/workflows/release.yml](.github/workflows/release.yml)) and the
  published files: the sdist and wheel on PyPI and the files of the GitHub release.

Out of scope: wrong or missing text in OCR output (that is a bug report, not a security problem),
the accuracy limits described in [MODEL_CARD.md](MODEL_CARD.md), and problems in third-party
libraries that are not caused by how GLYPHARIS uses them (report those to the library).

## How model files are verified

The package holds no weights. `glypharis.download_model` installs only names registered in
`REGISTRY` in [glypharis/hub.py](glypharis/hub.py), and each registered entry pins the archive's size
and SHA-256 and the SHA-256 of the bundle's `manifest.json`, which in turn lists the SHA-256 of every
model file. The archive is checked against its pin before it is extracted (a file that does not match
is deleted), every extracted file is checked against the manifest, and the manifest is checked
against its pin before the bundle is moved into place. Nothing unpinned is installed, whatever address
it was fetched from. `GlypharisOCR.from_pretrained` never reaches the network, and PyTorch weights are
read with `torch.load(..., weights_only=True)`. `glypharis-check` checks an installed bundle against
the same pins.

The v1 weights archive is `glypharis-v1.tar.gz`, an asset of the GitHub release
[v1.0](https://github.com/DuhItzAniket/Glypharis/releases/tag/v1.0), with SHA-256
`3b90c1c74147747d7474a86b550e198f0df83d447961bd6aec88d4c24846541d`. A copy fetched from anywhere else
should be checked against that value before use.
