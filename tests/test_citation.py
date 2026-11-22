"""scripts/check_citation.py: CITATION.cff against CFF 1.2.0, the package version and licence."""

from __future__ import annotations

from pathlib import Path

import check_citation
import pytest

GOOD = """\
cff-version: 1.2.0
message: Please cite this software.
title: GLYPHARIS
type: software
authors:
  - alias: someone
version: "1.0.0"
repository-code: https://github.com/example/project
license: Apache-2.0
abstract: An OCR engine.
keywords:
  - OCR
"""

INIT = '"""Package."""\n\n__version__ = "{version}"\n'
PYPROJECT = '[project]\nname = "glypharis"\nlicense = "{license}"\n'


def _write(tmp_path: Path, citation: str, version: str = "1.0.0", license_id: str | None = None):
    cff = tmp_path / "CITATION.cff"
    cff.write_text(citation, encoding="utf-8")
    init = tmp_path / "__init__.py"
    init.write_text(INIT.format(version=version), encoding="utf-8")
    pyproject = tmp_path / "pyproject.toml"
    if license_id is None:
        pyproject.write_text('[project]\nname = "glypharis"\n', encoding="utf-8")
    else:
        pyproject.write_text(PYPROJECT.format(license=license_id), encoding="utf-8")
    return ["--citation", str(cff), "--init", str(init), "--pyproject", str(pyproject)]


def test_a_good_file_passes(tmp_path, capsys):
    assert check_citation.main(_write(tmp_path, GOOD, license_id="Apache-2.0")) == 0
    assert "ok" in capsys.readouterr().out


def test_a_pyproject_without_a_licence_is_not_compared(tmp_path):
    assert check_citation.main(_write(tmp_path, GOOD, license_id=None)) == 0


def test_a_release_date_is_accepted(tmp_path):
    citation = GOOD + "date-released: 2026-12-01\n"
    assert check_citation.main(_write(tmp_path, citation)) == 0


def test_an_entity_author_is_accepted(tmp_path):
    citation = GOOD.replace("  - alias: someone\n", "  - name: The Project\n")
    assert check_citation.main(_write(tmp_path, citation)) == 0


def test_a_version_other_than_the_package_fails(tmp_path, capsys):
    assert check_citation.main(_write(tmp_path, GOOD, version="0.1.0.dev0")) == 1
    assert "glypharis.__version__ is '0.1.0.dev0'" in capsys.readouterr().out


def test_an_unquoted_numeric_version_fails(tmp_path, capsys):
    citation = GOOD.replace('version: "1.0.0"', "version: 1.0")
    assert check_citation.main(_write(tmp_path, citation, version="1.0")) == 1
    assert "quoted string" in capsys.readouterr().out


def test_a_licence_other_than_pyproject_fails(tmp_path, capsys):
    assert check_citation.main(_write(tmp_path, GOOD, license_id="MIT")) == 1
    assert "pyproject.toml says 'MIT'" in capsys.readouterr().out


def test_a_licence_table_in_pyproject_is_compared(tmp_path):
    args = _write(tmp_path, GOOD)
    Path(args[-1]).write_text('[project]\nlicense = { text = "MIT" }\n', encoding="utf-8")
    assert check_citation.main(args) == 1


@pytest.mark.parametrize("key", ["cff-version", "message", "title", "authors"])
def test_a_missing_required_key_fails(tmp_path, capsys, key):
    lines = GOOD.splitlines(keepends=True)
    kept = []
    skipping = False
    for line in lines:
        if line.startswith(f"{key}:"):
            skipping = True
            continue
        if skipping and line.startswith("  "):
            continue
        skipping = False
        kept.append(line)
    assert check_citation.main(_write(tmp_path, "".join(kept))) == 1
    assert f"missing required key '{key}'" in capsys.readouterr().out


def test_a_missing_version_fails(tmp_path, capsys):
    citation = GOOD.replace('version: "1.0.0"\n', "")
    assert check_citation.main(_write(tmp_path, citation)) == 1
    assert "missing key 'version'" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("cff-version: 1.2.0", "cff-version: 1.1.0", "cff-version is '1.1.0'"),
        ("type: software", "type: library", "type is 'library'"),
        ("title: GLYPHARIS", "titel: GLYPHARIS", "unknown key 'titel'"),
        ("  - alias: someone", "  - alias: someone\n    email: a@example.org", "no email"),
        ("  - alias: someone", "  - alias: someone\n    nickname: x", "unknown key 'nickname'"),
        ("  - alias: someone", "  - {}", "not a non-empty mapping"),
        ("authors:\n  - alias: someone", "authors: []", "authors is not a non-empty list"),
        ("https://github.com/example/project", "github.com/example", "not an http(s) URL"),
        ("keywords:\n  - OCR", "keywords: OCR", "keywords is not a non-empty list"),
        ("abstract: An OCR engine.", "abstract: ''", "abstract is not a non-empty string"),
    ],
)
def test_a_malformed_field_fails(tmp_path, capsys, old, new, message):
    assert old in GOOD
    assert check_citation.main(_write(tmp_path, GOOD.replace(old, new))) == 1
    assert message in capsys.readouterr().out


def test_a_release_date_that_is_not_a_date_fails(tmp_path, capsys):
    citation = GOOD + "date-released: soon\n"
    assert check_citation.main(_write(tmp_path, citation)) == 1
    assert "date-released" in capsys.readouterr().out


def test_a_file_that_is_not_a_mapping_fails(tmp_path, capsys):
    assert check_citation.main(_write(tmp_path, "- just\n- a list\n")) == 1
    assert "not a YAML mapping" in capsys.readouterr().out


def test_invalid_yaml_fails(tmp_path, capsys):
    assert check_citation.main(_write(tmp_path, "title: [unclosed\n")) == 1
    assert "FAIL" in capsys.readouterr().out


def test_an_init_without_one_version_line_fails(tmp_path, capsys):
    args = _write(tmp_path, GOOD)
    Path(args[3]).write_text('"""No version here."""\n', encoding="utf-8")
    assert check_citation.main(args) == 1
    assert "expected one __version__ line" in capsys.readouterr().out


def test_the_version_is_read_without_importing(tmp_path):
    init = tmp_path / "__init__.py"
    init.write_text('raise RuntimeError("imported")\n__version__ = "2.3.4"\n', encoding="utf-8")
    assert check_citation.package_version(init) == "2.3.4"


def test_the_repository_file_is_well_formed():
    """The tracked CITATION.cff passes every check but the version, which item 3 sets."""
    import yaml

    citation = yaml.safe_load(check_citation.ROOT.joinpath("CITATION.cff").read_text("utf-8"))
    problems = check_citation.check(citation, citation["version"], None)
    assert problems == []
    assert "date-released" not in citation
    # One author, named by the GitHub handle only (Zenodo reads the creator from this file).
    assert citation["authors"] == [{"name": "DuhItzAniket"}]
