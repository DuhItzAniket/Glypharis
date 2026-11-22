"""Check CITATION.cff against the Citation File Format 1.2.0 and the package (run in CI).

Usage:
    python scripts/check_citation.py [--citation CITATION.cff] [--init glypharis/__init__.py]
        [--pyproject pyproject.toml]

Checks, without a network or the CFF JSON schema:

- the file is a YAML mapping with the keys CFF 1.2.0 requires (``cff-version``, ``message``,
  ``title``, ``authors``), ``cff-version`` is ``1.2.0``, and no key outside CFF 1.2.0 is used;
- every author is a person or an entity with keys from the format, an entity has a ``name``, and no
  author carries an email address (the repository keeps none);
- ``version`` equals ``__version__`` in ``glypharis/__init__.py``, read as text so the package is
  not imported;
- ``license`` equals ``project.license`` in ``pyproject.toml`` when that field is set;
- ``date-released``, if present, is a date (quoted ``YYYY-MM-DD``, as the CFF schema expects, or a
  YAML date), and ``keywords`` is a list of strings.

Exits 1 and prints each problem when a check fails.
"""

from __future__ import annotations

import argparse
import datetime
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

CFF_VERSION = "1.2.0"
REQUIRED = ("cff-version", "message", "title", "authors")
TOP_LEVEL = frozenset(
    {
        "abstract",
        "authors",
        "cff-version",
        "commit",
        "contact",
        "date-released",
        "doi",
        "identifiers",
        "keywords",
        "license",
        "license-url",
        "message",
        "preferred-citation",
        "references",
        "repository",
        "repository-artifact",
        "repository-code",
        "title",
        "type",
        "url",
        "version",
    }
)
TYPES = frozenset({"software", "dataset"})
PERSON = frozenset(
    {
        "address",
        "affiliation",
        "alias",
        "city",
        "country",
        "email",
        "family-names",
        "fax",
        "given-names",
        "name-particle",
        "name-suffix",
        "orcid",
        "post-code",
        "region",
        "tel",
        "website",
    }
)
ENTITY = frozenset(
    {
        "address",
        "alias",
        "city",
        "country",
        "date-end",
        "date-start",
        "email",
        "fax",
        "location",
        "name",
        "orcid",
        "post-code",
        "region",
        "tel",
        "website",
    }
)
VERSION_LINE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']\s*$""", re.M)
URL = re.compile(r"^https?://\S+$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def is_date(value: Any) -> bool:
    """A YAML date, or a quoted ``YYYY-MM-DD`` string naming a real day (the CFF schema's form)."""
    if isinstance(value, datetime.date):
        return True
    if not isinstance(value, str) or not ISO_DATE.match(value):
        return False
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def package_version(init: Path) -> str:
    """``__version__`` from the package's ``__init__.py``, read as text."""
    found = VERSION_LINE.findall(init.read_text(encoding="utf-8"))
    if len(found) != 1:
        raise ValueError(f"{init.as_posix()}: expected one __version__ line, found {len(found)}")
    return str(found[0])


def project_license(pyproject: Path) -> str | None:
    """``project.license`` as an SPDX string (or the ``text`` of a table), or None when unset."""
    if not pyproject.is_file():
        return None
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    value = data.get("project", {}).get("license")
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return str(value["text"])
    return None


def _check_author(index: int, author: Any) -> list[str]:
    where = f"authors[{index}]"
    if not isinstance(author, dict) or not author:
        return [f"{where} is not a non-empty mapping"]
    keys = set(author)
    allowed = ENTITY if "name" in keys else PERSON
    problems = [f"{where}: unknown key {k!r}" for k in sorted(keys - allowed)]
    for key, value in author.items():
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{where}.{key} is not a non-empty string")
    if "email" in keys:
        problems.append(f"{where}: no email address may be recorded")
    return problems


def check(citation: dict[str, Any] | Any, version: str, license_id: str | None) -> list[str]:
    """Every problem found in a parsed CITATION.cff, or an empty list."""
    if not isinstance(citation, dict):
        return ["CITATION.cff is not a YAML mapping"]
    problems = [f"missing required key {k!r}" for k in REQUIRED if k not in citation]
    problems += [f"unknown key {k!r}" for k in sorted(set(citation) - TOP_LEVEL)]
    if "cff-version" in citation and str(citation["cff-version"]) != CFF_VERSION:
        problems.append(f"cff-version is {citation['cff-version']!r}, not {CFF_VERSION!r}")
    for key in ("message", "title", "abstract", "license", "repository-code", "url"):
        if key in citation and (not isinstance(citation[key], str) or not citation[key].strip()):
            problems.append(f"{key} is not a non-empty string")
    if "type" in citation and citation["type"] not in TYPES:
        problems.append(f"type is {citation['type']!r}, not one of {sorted(TYPES)}")
    authors = citation.get("authors")
    if "authors" in citation:
        if not isinstance(authors, list) or not authors:
            problems.append("authors is not a non-empty list")
        else:
            for i, author in enumerate(authors):
                problems += _check_author(i, author)
    for key in ("repository-code", "url"):
        value = citation.get(key)
        if isinstance(value, str) and not URL.match(value):
            problems.append(f"{key} is not an http(s) URL: {value!r}")
    if "date-released" in citation and not is_date(citation["date-released"]):
        problems.append(f"date-released is not a YYYY-MM-DD date: {citation['date-released']!r}")
    keywords = citation.get("keywords")
    if "keywords" in citation and (
        not isinstance(keywords, list)
        or not keywords
        or not all(isinstance(k, str) and k.strip() for k in keywords)
    ):
        problems.append("keywords is not a non-empty list of strings")
    if "version" not in citation:
        problems.append("missing key 'version'")
    elif not isinstance(citation["version"], str):
        problems.append(f"version must be a quoted string, got {citation['version']!r}")
    elif citation["version"] != version:
        problems.append(
            f"version is {citation['version']!r}, but glypharis.__version__ is {version!r}"
        )
    if license_id is not None and citation.get("license") != license_id:
        problems.append(
            f"license is {citation.get('license')!r}, but pyproject.toml says {license_id!r}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check CITATION.cff.")
    parser.add_argument("--citation", type=Path, default=ROOT / "CITATION.cff")
    parser.add_argument("--init", type=Path, default=ROOT / "glypharis" / "__init__.py")
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    args = parser.parse_args(argv)

    try:
        citation = yaml.safe_load(args.citation.read_text(encoding="utf-8"))
        version = package_version(args.init)
        license_id = project_license(args.pyproject)
    except (OSError, ValueError, yaml.YAMLError, tomllib.TOMLDecodeError) as exc:
        print(f"FAIL {exc}")
        return 1
    problems = check(citation, version, license_id)
    for problem in problems:
        print(f"FAIL {args.citation.name}: {problem}")
    if problems:
        return 1
    print(f"ok   {args.citation.name}: CFF {CFF_VERSION}, version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
