"""GDC source documents must never appear in a training corpus (EXPERIMENT_PROTOCOL §1 rule 5)."""

from __future__ import annotations

import json
import re

import pytest


def test_gdc_books_are_not_in_the_pg19_text_used_by_glyphsynth(repo_root):
    excl = repo_root / "data" / "manifests" / "exclusions" / "gdc_sources.json"
    pg19 = repo_root / "data" / "manifests" / "pg19.json"
    if not excl.exists() or not pg19.exists():
        pytest.skip("manifests not present")
    gdc_ids = {d["gutenberg"] for d in json.loads(excl.read_text(encoding="utf-8"))["documents"]
               if "gutenberg" in d}  # fmt: skip
    pg_ids = {int(m.group(1)) for name in json.loads(pg19.read_text(encoding="utf-8"))["files"]
              if (m := re.search(r"_(\d+)_txt$", name))}  # fmt: skip
    assert gdc_ids and pg_ids
    assert not gdc_ids & pg_ids


def test_gdc_sources_are_not_training_datasets(repo_root):
    import yaml

    sources = yaml.safe_load((repo_root / "data" / "sources.yaml").read_text(encoding="utf-8"))
    assert "train" not in sources["gdc_sources"]["role"].replace("training", "")
    assert "never training" in sources["gdc_sources"]["role"]
