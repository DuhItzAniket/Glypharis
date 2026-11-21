"""Check a release tag against the package version (Phase 23, item 6; run by the release workflow).

Usage:
    python scripts/release_version_check.py --tag v1.0 [--init glypharis/__init__.py] [--allow-pre]

The tag, without its leading ``v``, must equal ``__version__`` in ``glypharis/__init__.py`` as
parsed versions (``packaging.version.Version``), so the tag ``v1.0`` matches the package version
``1.0.0`` and ``v1.0.1`` does not. The version is read from the file's text; the package is not
imported.

Refused, with exit 1 and a message:

- a tag that is not a version after its ``v`` (``release``, ``vv1.0``, an empty tag), or one not
  written in its normalised form (``v1.0-rc1`` for ``v1.0rc1``, ``v1.0.0.post`` for
  ``v1.0.0.post0``);
- a tag with a local part (``v1.0+abc``): PyPI does not accept local versions;
- a pre-release or development release (``v1.0rc1``, ``v1.0.dev0``), unless ``--allow-pre``;
- a tag that does not equal the package version, or an ``__init__.py`` without one clear
  ``__version__`` line.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]

VERSION_LINE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']\s*$""", re.M)


class ReleaseVersionError(ValueError):
    """The tag or the package version cannot be released as given."""


def package_version(init: Path) -> str:
    """``__version__`` from the package's ``__init__.py``, read as text."""
    found = VERSION_LINE.findall(init.read_text(encoding="utf-8"))
    if len(found) != 1:
        raise ReleaseVersionError(
            f"{init.as_posix()}: expected one __version__ line, found {len(found)}"
        )
    return str(found[0])


def tag_version(tag: str) -> Version:
    """The version a tag names: the tag without one leading ``v``, parsed strictly."""
    text = tag[1:] if tag.startswith("v") else tag
    if not text[:1].isdigit():
        raise ReleaseVersionError(f"tag {tag!r} does not name a version")
    try:
        version = Version(text)
    except InvalidVersion as exc:
        raise ReleaseVersionError(f"tag {tag!r} does not name a version") from exc
    if version.local is not None:
        raise ReleaseVersionError(f"tag {tag!r} has a local version part, which PyPI refuses")
    if str(version) != text:
        raise ReleaseVersionError(f"tag {tag!r} is not in normalised form; use 'v{version}'")
    return version


def check(tag: str, init_version: str, *, allow_pre: bool = False) -> Version:
    """The released version when ``tag`` matches ``init_version``; raises otherwise."""
    wanted = tag_version(tag)
    try:
        have = Version(init_version)
    except InvalidVersion as exc:
        raise ReleaseVersionError(f"__version__ {init_version!r} is not a valid version") from exc
    if wanted != have:
        raise ReleaseVersionError(
            f"tag {tag!r} names version {wanted}, but glypharis.__version__ is {init_version!r}"
        )
    if (wanted.is_prerelease or wanted.is_devrelease) and not allow_pre:
        raise ReleaseVersionError(
            f"tag {tag!r} is a pre-release or development release; pass --allow-pre to accept it"
        )
    return wanted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a release tag against the package version.")
    parser.add_argument("--tag", required=True, help="release tag, for example v1.0")
    parser.add_argument("--init", type=Path, default=ROOT / "glypharis" / "__init__.py")
    parser.add_argument(
        "--allow-pre", action="store_true", help="accept a pre-release or development release"
    )
    args = parser.parse_args(argv)

    try:
        init_version = package_version(args.init)
        version = check(args.tag, init_version, allow_pre=args.allow_pre)
    except (OSError, ReleaseVersionError) as exc:
        print(f"FAIL {exc}")
        return 1
    print(f"ok   tag {args.tag} matches glypharis.__version__ {init_version} (version {version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
