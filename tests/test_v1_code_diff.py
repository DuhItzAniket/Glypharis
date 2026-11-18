"""The end-of-freeze check: only docstrings, comments and one default may change."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import v1_code_diff as diff

OCR = '''"""Module text."""


class GlypharisOCR:
    """Class text."""

    def from_pretrained(self, name="glypharis-0.1.0", backend="torch"):
        """Old text."""
        return name, backend  # a comment
'''
OCR_PATH = "glypharis/pipeline/ocr.py"


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def _commit(repo: Path, path: str, text: str, tag: str) -> None:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text, encoding="utf-8", newline="\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", tag)
    _git(repo, "tag", "-f", tag)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    (r / "configs").mkdir()
    (r / "configs/a.yaml").write_text("a: 1\n", encoding="utf-8")
    _commit(r, OCR_PATH, OCR, "base")
    return r


def test_docstrings_comments_and_the_registered_default_pass(repo):
    new = (
        OCR.replace("Old text.", "New text.")
        .replace("a comment", "another")
        .replace("Module text.", "New module text.")
        .replace("glypharis-0.1.0", "glypharis-v1")
    )
    _commit(repo, OCR_PATH, new, "head")
    res = diff.check(repo, "base", "head")
    assert res["pass"] and len(res["files"]) == 1
    assert res["files"][0]["default"]["to"] == "glypharis-v1"


def test_docstrings_alone_pass_without_the_default(repo):
    _commit(repo, OCR_PATH, OCR.replace("Class text.", "Other."), "head")
    res = diff.check(repo, "base", "head")
    assert res["pass"] and res["files"][0]["change"] == "docstrings or comments only"


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("return name, backend", "return backend, name"),
        ('backend="torch"', 'backend="onnx"'),
        ("glypharis-0.1.0", "glypharis-other"),
    ],
)
def test_any_other_code_change_fails(repo, old, new):
    _commit(repo, OCR_PATH, OCR.replace(old, new), "head")
    res = diff.check(repo, "base", "head")
    assert not res["pass"] and res["files"][0]["change"] == "code differs"


def test_the_default_is_allowed_only_in_its_own_file(repo):
    other = "glypharis/pipeline/other.py"
    _commit(repo, other, OCR, "base")
    _commit(repo, other, OCR.replace("glypharis-0.1.0", "glypharis-v1"), "head")
    assert not diff.check(repo, "base", "head")["pass"]


def test_an_added_file_fails(repo):
    _commit(repo, "glypharis/pipeline/new.py", "x = 1\n", "head")
    res = diff.check(repo, "base", "head")
    assert not res["pass"] and "status A" in res["files"][0]["change"]


def test_a_changed_mode_fails(repo):
    _git(repo, "update-index", "--chmod=+x", OCR_PATH)
    _git(repo, "commit", "-q", "-m", "head")
    _git(repo, "tag", "-f", "head")
    res = diff.check(repo, "base", "head")
    assert not res["pass"] and "100755" in res["files"][0]["change"]


@pytest.mark.parametrize(
    ("base", "head"),
    [
        (b"R = 1\n", b"# coding: utf-7\nR = 1\n#+AAo-R = 2\n"),  # a comment that decodes to code
        ('S = "\u00e9"\n'.encode(), b"# -*- coding: latin-1 -*-\n" + 'S = "\u00e9"\n'.encode()),
    ],
)
def test_a_coding_comment_cannot_change_the_code_unseen(base, head):
    assert not diff.compare("x.py", base, head)["ok"]


@pytest.mark.parametrize("bad", [b"def (:\n", b"x = 1\0\n", b'x = "\xff"\n'])
def test_a_file_that_does_not_parse_fails_and_is_named(bad):
    res = diff.compare("x.py", b"x = 1\n", bad)
    assert not res["ok"] and res["change"].startswith("does not parse")


def test_a_changed_config_fails(repo):
    _commit(repo, "configs/a.yaml", "a: 2\n", "head")
    res = diff.check(repo, "base", "head")
    assert not res["pass"] and "not a modified Python file" in res["files"][0]["change"]


def test_a_function_that_was_only_a_docstring_still_compares():
    a = 'def f():\n    """Text."""\n'
    assert diff.compare("x.py", a, 'def f():\n    """Other."""\n')["ok"]
    assert not diff.compare("x.py", a, "def f():\n    return 1\n")["ok"]


def test_main_writes_once_and_reports(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(diff, "git_state", lambda: {"git_commit": "h" * 40, "git_dirty": False})
    _commit(repo, OCR_PATH, OCR.replace("Old text.", "New."), "head")
    out = tmp_path / "out/diff.json"
    argv = ["--root", str(repo), "--base", "base", "--head", "head", "--out", str(out)]
    assert diff.main(argv) == 0
    text = out.read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["pass"] and report["base_commit"] == _git(repo, "rev-parse", "base")
    assert str(tmp_path) not in text
    with pytest.raises(SystemExit, match="written once"):
        diff.main(argv)
