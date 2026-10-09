# ADR-0004: Lowercase Python package name

- **Status:** Accepted (2026-10-09, Phase 0)

## Context
The original spec writes the API as `from GLYPHARIS import GLYPHARISOCR`. PEP 8 asks for lowercase
module names, and an uppercase package name breaks on case-sensitive file systems as soon as anything
imports it with different casing.

## Decision
- Package: `glypharis`. Main class: `GlypharisOCR`.
- `GLYPHARISOCR` is exported as an alias so code written against the spec still works.
- Branding (GLYPHARIS, GLYPHARIS-Det, GLYPHARIS-Rec) is unchanged in docs, model cards and metadata.

## Consequences
Normal Python tooling (ruff, mypy, packaging) works without exceptions. The spec's examples need a
one-word change.

## Revisit if
Never, unless the package is renamed.
