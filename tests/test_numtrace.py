"""The number tokenizer, the report index and the trace directives, on made-up text."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from decimal import Decimal
from pathlib import Path

import numtrace as nt
import pytest

MINUS = "\u2212"


def raws(text: str) -> list[str]:
    return [t.raw + t.unit for t in nt.tokenize(text)]


def write(root: Path, path: str, text: str) -> str:
    full = root / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(text.encode("utf-8"))
    return path


def only_error(text: str) -> str:
    errors = nt.parse_document(text).errors
    assert len(errors) == 1, errors
    return errors[0].message


# Numbers


@pytest.mark.parametrize(
    ("text", "raw", "value", "sign", "unit", "exponent"),
    [
        ("7", "7", "7", "", "", 0),
        ("0", "0", "0", "", "", 0),
        ("6.30", "6.30", "6.30", "", "", -2),
        (".5", ".5", "0.5", "", "", -1),
        ("1,634", "1,634", "1634", "", "", 0),
        ("7,928,835", "7,928,835", "7928835", "", "", 0),
        ("1,665.0", "1,665.0", "1665.0", "", "", -1),
        ("74\u00a0500", "74\u00a0500", "74500", "", "", 0),
        ("74\u202f500", "74\u202f500", "74500", "", "", 0),
        ("1\u2009234\u2009567", "1\u2009234\u2009567", "1234567", "", "", 0),
        ("-0.76", "-0.76", "-0.76", "-", "", -2),
        (f"{MINUS}0.76", f"{MINUS}0.76", "-0.76", "-", "", -2),
        (f"{MINUS}1,248.1", f"{MINUS}1,248.1", "-1248.1", "-", "", -1),
        ("+0.76", "+0.76", "0.76", "+", "", -2),
        ("+1,000", "+1,000", "1000", "+", "", 0),
        ("7.33e-06", "7.33e-06", "0.00000733", "", "", -8),
        (f"3.09e{MINUS}04", f"3.09e{MINUS}04", "0.000309", "", "", -6),
        ("1e-4", "1e-4", "0.0001", "", "", -4),
        ("1E+3", "1E+3", "1000", "", "", 3),
        ("2e5", "2e5", "200000", "", "", 5),
        (f"{MINUS}1.5e-3", f"{MINUS}1.5e-3", "-0.0015", "-", "", -4),
        ("6.30%", "6.30", "6.30", "", "%", -2),
        ("6.30 %", "6.30", "6.30", "", "%", -2),
        ("+0.5 points", "+0.5", "0.5", "+", "points", -1),
        ("1 point", "1", "1", "", "points", 0),
        ("0.5 pp", "0.5", "0.5", "", "pp", -1),
        ("12 percent", "12", "12", "", "%", 0),
        ("3 percentage points", "3", "3", "", "points", 0),
        ("2,565.7 ms", "2,565.7", "2565.7", "", "ms", -1),
        ("100ms", "100", "100", "", "ms", 0),
        ("12.44MiB", "12.44", "12.44", "", "MiB", -2),
        ("12.44 MiB", "12.44", "12.44", "", "MiB", -2),
        ("25K", "25", "25", "", "K", 0),
        ("3.3x", "3.3", "3.3", "", "x", -1),
        ("3.3\u00d7", "3.3", "3.3", "", "x", -1),
    ],
)
def test_number_forms(text: str, raw: str, value: str, sign: str, unit: str, exponent: int) -> None:
    for wrapped in (text, f"It was {text} here.", f"({text})", f"**{text}**", f"_{text}_"):
        tokens = nt.tokenize(wrapped)
        assert len(tokens) == 1, wrapped
        token = tokens[0]
        assert token.raw == raw
        assert token.value == Decimal(value)
        assert token.sign == sign
        assert token.unit == unit
        assert token.exponent == exponent


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5-10 items", ["5", "10"]),
        ("5\u201310 items", ["5", "10"]),
        ("5 - 10 items", ["5", "10"]),
        ("1.5-2.5", ["1.5", "2.5"]),
        ("5%-10%", ["5%", "10%"]),
        ("3/4 of 1,000/2,000", ["3", "4", "1,000", "2,000"]),
        ("(3,4)", ["3", "4"]),
        ("1,23", ["1", "23"]),
        ("1234,567", ["1234", "567"]),
        ("0,123", ["0", "123"]),
        ("12,345,67", ["12,345", "67"]),
        ("48x320 and 32\u00d7320", ["48", "320", "32", "320"]),
        ("a 28-page document", ["28"]),
        ("3.3ms/page", ["3.3ms"]),
        ("It ended at 5. Then 6.", ["5", "6"]),
        ("a, 5, b", ["5"]),
        ("about ~41 and \u226495 and <3", ["41", "95", "3"]),
    ],
)
def test_several_numbers(text: str, expected: list[str]) -> None:
    assert raws(text) == expected


def test_a_range_hyphen_is_not_a_sign() -> None:
    low, high = nt.tokenize("5-10")
    assert (low.value, high.value, high.sign) == (Decimal(5), Decimal(10), "")


def test_significant_digits_as_printed() -> None:
    assert [t.significant_digits for t in nt.tokenize("13 0.50 100 7.33e-06 0 74,500")] == [
        2,
        2,
        3,
        3,
        1,
        5,
    ]


@pytest.mark.parametrize(
    ("source", "printed", "exponent", "expected"),
    [
        ("6.295", "6.30", -2, True),
        ("6.305", "6.30", -2, True),
        ("6.2949", "6.30", -2, False),
        ("6.3051", "6.30", -2, False),
        ("0.14563260148475754", "0.15", -2, True),
        ("1500", "2E+3", 3, True),
    ],
)
def test_rounding_is_half_a_unit_inclusive(
    source: str, printed: str, exponent: int, expected: bool
) -> None:
    assert nt.rounds_to(Decimal(source), Decimal(printed), exponent) is expected


# What is never a number


@pytest.mark.parametrize(
    "text",
    [
        # Experiment and registry IDs.
        "GW-REC-040",
        "GW-E2E-007",
        "GW-DATA-010 to 021",
        "GW-REC-034 to 036",
        "GW-DET-002, 003, 004, 005 and 007",
        "E2E-001\u2192006",
        "EFF-002 and 004\u2192007",
        "F3 and F24",
        "F21, F22",
        "H1\u2032 and H4",
        "H2'",
        "RQ1-RQ8",
        "RQ 8",
        "ADR-0013",
        "ADR 14",
        "R-05",
        "R1-R11",
        "R7",
        # Project and document references.
        "Phase 7",
        "phase 14",
        "Phase 14-16",
        "Phase 5 to 7",
        "Phases 5, 14 and 22",
        "Phases 22 or 23",
        "Phases 5 / 14 / 22",
        "Phases 21 to 23",
        "Phase\n14",
        "Phases 5,\n14 and\n22",
        "brief \u00a713",
        "\u00a7\u00a7 9, 13 and 54",
        "Table 2",
        "Tables 2 and 3",
        "Figure 3",
        "Fig. 3",
        "Eq. 2",
        "Section 4.2",
        "item 6",
        "step 2",
        "Stage 3",
        "issue #12",
        # Dates.
        "2026-11-18",
        "2026-10",
        "2026/11/18",
        "20261118",
        "2026-11-17T12:30:00Z",
        "18 November 2026",
        "November 18, 2026",
        "Nov 18",
        "January 2023",
        "the 1990s",
        "et al. 2022",
        "(Smith, 2007)",
        "CVPR 2025",
        "ICDAR 2019",
        # Hashes.
        "1253dfe",
        "034b867",
        "5361e68682559021",
        "1253e12",
        "875357a6346e8b1f989005fa96c557b00bf687461043368dc5e3dc532839f037",
        # Files, paths and addresses.
        "results.json",
        "lines_onnx_cpu_10t.json",
        "2023.pdf",
        "GW-REC-020.yaml",
        "reports/training/phase07/ablation_table.md",
        "runs/release/glypharis-v1",
        "902b8cf14489c26a/p18/l9",
        "https://example.org/a/2024/5.html",
        "www.example.org/7",
        "someone42@example.org",
        "10.1109/TPAMI.2022.3155612",
        "arXiv:2507.05595",
        "arXiv 2202.10304v2",
        # Seeds.
        "seed 20261118",
        "--seed 20261118",
        "seed=7",
        "seed: 7",
        "seeds 1, 2 and 3",
        "seed [20261115, 1]",
        "random seed of 42",
        "seed\n7",
        # Versions.
        "v1",
        "v0.1",
        "v1.0.0",
        "2.14.1",
        "2.14.1+cu126",
        "version 1.1",
        "Version 3.0",
        "Python 3.11",
        "opset 17",
        "Tesseract 5",
        "CUDA 12.6",
        "Windows 11",
        "Ubuntu 22.04",
        "Unicode 15.1",
        # Digits inside names.
        "fp32",
        "bf16",
        "INT8",
        "p95",
        "F1",
        "F1@0.5",
        "F1 @0.5",
        "P @0.5",
        "mAP@[0.5:0.95]",
        "PP-OCRv5",
        "PP-OCRv6 tiny",
        "MobileNetV3",
        "IIIT5K",
        "SHA-256",
        "UTF-8",
        "ISO 8601",
        "top-5",
        "95th",
        "3D",
        # Licences and hardware.
        "CC BY 4.0",
        "CC BY-SA 3.0",
        "CC BY-NC 4.0",
        "CC\nBY 4.0",
        "CC0 1.0",
        "GPL-3.0",
        "GNU FDL 1.3",
        "Apache 2.0",
        "Apache-2.0",
        "BSD-3-Clause",
        "RTX 4050",
        "RTX\n4050",
        "GTX 1650",
        "i7-14650HX",
        "Core i7",
        # Identifiers with zero padding.
        "036",
        "007",
    ],
)
def test_never_a_number(text: str) -> None:
    assert raws(text) == []
    assert raws(f"Before {text} after.") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Phase 14 read 1,634 pages", ["1,634"]),
        ("Phases 5, 14 and 22 read 200 pages", ["200"]),
        ("Phase 14, 200 pages", ["200"]),
        ("GW-REC-034 to 036 used 3 seeds", ["3"]),
        ("GW-REC-034, 274 characters", ["274"]),
        ("seed 7 gave 6.30%", ["6.30%"]),
        ("5 seeds", ["5"]),
        ("ONNX 9.82 ms", ["9.82ms"]),
        ("PyTorch 1,665.0 MiB", ["1,665.0MiB"]),
        ("Tesseract 18.95", ["18.95"]),
        ("step 45,295", ["45,295"]),
        ("45,295 steps", ["45,295"]),
        ("F1 0.79 and F1@0.5 of 79.13", ["0.79", "79.13"]),
        ("CC BY 4.0, 12 files", ["12"]),
        ("v1 at 6.51", ["6.51"]),
        ("RTX 4050 with 6 GB", ["6GB"]),
        ("in 2023 and 2,023", ["2023", "2,023"]),
        ("2026-11-18: 5", ["5"]),
    ],
)
def test_identifiers_leave_the_numbers_beside_them(text: str, expected: list[str]) -> None:
    assert raws(text) == expected


# Markdown structure


@pytest.mark.parametrize(
    "text",
    [
        "`12`",
        "``a ` 12``",
        "`split\nover 12 lines`",
        "```\n12\n```",
        "```python\nx = 12\n```",
        "~~~\n12\n~~~",
        "   ```\n12\n   ```",
        "```\n12",
        "<!-- 12 -->",
        "<!-- over\n12\nlines -->",
        "[](https://example.org/12)",
        "![](figures/12.png)",
        "[text](target12 'title 12')",
        "[text](12)",
        "[text][12]",
        "[12]: https://example.org/12",
        "<https://example.org/12>",
        '<a id="gw-12"></a>',
        "<span data-n=12>",
        "text[^12]",
    ],
)
def test_markdown_constructs_hide_numbers(text: str) -> None:
    assert raws(text) == []


def test_link_text_is_kept_and_its_target_is_not() -> None:
    assert raws("[12 pages](docs/7.md) and ![3 crops](a/4.png)") == ["12", "3"]


def test_an_unclosed_backtick_hides_nothing() -> None:
    assert raws("a ` then 12") == ["12"]
    assert raws(r"a \` then 12 \` end") == ["12"]


def test_a_code_span_does_not_cross_a_blank_line() -> None:
    assert raws("a ` b\n\n12 ` c") == ["12"]


def test_fence_needs_its_own_kind_to_close() -> None:
    assert raws("```\n~~~\n12\n```\n13") == ["13"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1. first 5", ["5"]),
        ("2) second 6", ["6"]),
        ("- 5.2% done", ["5.2%"]),
        ("* 7 items", ["7"]),
        ("+ 8 items", ["8"]),
        ("  - nested 9", ["9"]),
        ("    1. deep 10", ["10"]),
        ("> 1. quoted 11", ["11"]),
        ("> quoted 12", ["12"]),
        ("1. one\n2. two\n3. three", []),
        ("1. one\n\n2. two", []),
        ("# 1 Title", []),
        ("## 1. Title", []),
        ("### 3.2 Title", []),
        ("### 3.2) Title 7", ["7"]),
        ("## 5,000 lines", ["5,000"]),
        ("## Title 7", ["7"]),
    ],
)
def test_list_markers_and_heading_numbers(text: str, expected: list[str]) -> None:
    assert raws(text) == expected


def test_a_wrapped_line_starting_with_a_number_is_not_a_list() -> None:
    assert raws("The count was\n12. The next sentence.") == ["12"]
    assert raws("The count was\n1. A list starts.") == []


def test_negative_at_line_start_is_not_a_marker() -> None:
    assert raws("-5 here") == ["-5"]
    assert raws("+5 here") == ["+5"]


def test_lines_columns_and_crlf() -> None:
    text = "# T\r\n\r\nab 12 and\r\n  1,000%\r\n"
    first, second = nt.tokenize(text)
    assert (first.line, first.column, first.end_column) == (3, 4, 6)
    assert (second.line, second.column, second.end_column) == (4, 3, 8)


def test_odd_line_separators_do_not_shift_lines() -> None:
    assert nt.tokenize("a\u2028b\x0cc\n12")[0].line == 2


def test_blocks_sections_ids_and_sources() -> None:
    text = (
        "# Top\n\nFirst GW-REC-001 in `reports/a.md#x`, 1.\nstill 2.\n\n"
        "## Sub GW-DET-001\n\n- thing 3\n  more 4\n\n- thing 5\n\nAfter 6.\n\n"
        "```\ncode\n```\n<!-- GW-REC-009 reports/hidden.md -->\n"
        "| A | B |\n|---|---|\n| x | 7 |\n"
    )
    document = nt.parse_document(text)
    assert [b.kind for b in document.blocks] == [
        "heading",
        "paragraph",
        "heading",
        "list",
        "paragraph",
        "code",
        "table",
    ]
    paragraph = document.blocks[1]
    assert (paragraph.start_line, paragraph.end_line, paragraph.section) == (3, 4, 0)
    assert paragraph.ids == ("GW-REC-001",)
    assert paragraph.sources == ("reports/a.md#x",)
    assert document.blocks[2].ids == ("GW-DET-001",)
    assert document.blocks[3].section == 1
    kinds = [document.blocks[t.block].kind for t in document.tokens]
    assert kinds == ["paragraph", "paragraph", "list", "list", "list", "paragraph", "table"]
    assert all("GW-REC-009" not in b.ids for b in document.blocks)
    assert [h.slug for h in document.headings] == ["top", "sub-gw-det-001"]


def test_tables_and_cells() -> None:
    text = (
        "| Name | CER % | Note |\n|:--|--:|---|\n"
        "| **a** | 6.30 [6.1, 6.4] | x \\| 2 |\n| `b` | 1,000 | `c|3` |\n\nAfter 9\n"
    )
    document = nt.parse_document(text)
    table = document.tables[0]
    assert table.header == ("Name", "CER %", "Note")
    assert table.rows[0].cells == ("a", "6.30 [6.1, 6.4]", "x | 2")
    assert table.rows[1].cells == ("b", "1,000", "c|3")
    assert [t.raw for t in document.tokens] == ["6.30", "6.1", "6.4", "2", "1,000", "9"]
    cell = document.tokens[4].cell
    assert cell == nt.CellRef(0, 1, 1, "CER %")
    assert document.tokens[3].cell == nt.CellRef(0, 0, 2, "Note")
    assert document.tokens[5].cell is None


def test_a_pipe_without_a_delimiter_row_is_not_a_table() -> None:
    assert nt.parse_document("a | b 1\nc | d 2\n").tables == ()


# Intervals


@pytest.mark.parametrize(
    ("text", "point", "low", "high"),
    [
        ("7.56 [7.40, 7.71]", "7.56", "7.40", "7.71"),
        ("7.56% [7.40, 7.71]", "7.56", "7.40", "7.71"),
        ("7.56% [7.40%, 7.71%]", "7.56", "7.40", "7.71"),
        ("**7.56** [7.40, 7.71]", "7.56", "7.40", "7.71"),
        (f"{MINUS}0.76 [{MINUS}0.81, {MINUS}0.71]", f"{MINUS}0.76", f"{MINUS}0.81", f"{MINUS}0.71"),
        ("+0.5 [-0.1, +1.1]", "+0.5", "-0.1", "+1.1"),
        ("7.56\n[7.40,\n7.71]", "7.56", "7.40", "7.71"),
        ("1,000 [900, 1,100]", "1,000", "900", "1,100"),
        ("7.56 [7.40; 7.71]", "7.56", "7.40", "7.71"),
        ("7.56 [7.40\u20137.71]", "7.56", "7.40", "7.71"),
        ("7.56 [7.40 to 7.71]", "7.56", "7.40", "7.71"),
        ("7.56 [7.71, 7.40]", "7.56", "7.71", "7.40"),
        ("| 7.56 [7.40, 7.71] |\n", "7.56", "7.40", "7.71"),
    ],
)
def test_intervals(text: str, point: str, low: str, high: str) -> None:
    intervals = nt.parse_document(text).intervals
    assert len(intervals) == 1
    interval = intervals[0]
    assert interval.point is not None
    assert (interval.point.raw, interval.low.raw, interval.high.raw) == (point, low, high)


def test_interval_without_a_point() -> None:
    interval = nt.parse_document("The range was [7.40, 7.71] here.").intervals[0]
    assert interval.point is None
    assert interval.low.raw == "7.40"


@pytest.mark.parametrize(
    "text",
    ["seed [20261115, 1]", "[1, 2](target.md)", "[1, 2, 3]", "(1, 2)", "[1 2]", "[1, 2"],
)
def test_not_intervals(text: str) -> None:
    assert nt.parse_document(text).intervals == ()


# Directives


def test_the_three_directives_as_the_plan_writes_them() -> None:
    trace = nt.parse_directive(
        "<!-- trace: GW-REC-040 reports/error_cases/phase15/recognition.md#gdc_test -->", 3
    )
    assert (trace.kind, trace.line, trace.experiment) == ("trace", 3, "GW-REC-040")
    assert trace.path == "reports/error_cases/phase15/recognition.md"
    assert trace.anchor == "gdc_test"

    table = nt.parse_directive(
        "<!-- trace-table: GW-E2E-007 reports/external/end_to_end.md key=row|key2=row,header -->"
    )
    assert (table.kind, table.experiment, table.anchor) == ("trace-table", "GW-E2E-007", None)
    assert table.bindings == (
        nt.Binding(("key",), ("row",), None),
        nt.Binding(("key2",), ("row",), "header"),
    )

    derived = nt.parse_directive("<!-- derived: fps_ratio.v1-a -->")
    assert (derived.kind, derived.name) == ("derived", "fps_ratio.v1-a")


def test_a_path_stands_in_for_an_id() -> None:
    path = "data/manifests/charset.json"
    directive = nt.parse_directive(f"<!-- trace: {path} {path}#vocabulary_size -->")
    assert (directive.experiment, directive.path) == (path, path)
    assert directive.anchor == "vocabulary_size"


def test_binding_terms_spaces_and_escapes() -> None:
    bindings = nt.parse_bindings(
        r"A b ; c = GW-E2E-007 ; gdc_test , F1 @0.5 | x\|y=1\,000\;2,a\=b\\"
    )
    assert bindings[0] == nt.Binding(("A b", "c"), ("GW-E2E-007", "gdc_test"), "F1 @0.5")
    assert bindings[1] == nt.Binding(("x|y",), ("1,000;2",), "a=b\\")


@pytest.mark.parametrize(
    "text",
    ["", "key", "=row", "key=", "key=row,", "key=a,b,c", "a=b=c", "a=1|", "a=1|a=2", "a;=1"],
)
def test_bad_bindings(text: str) -> None:
    with pytest.raises(ValueError):
        nt.parse_bindings(text)


@pytest.mark.parametrize(
    "comment",
    [
        "<!--trace: GW-REC-001 reports/a.md-->",
        "<!-- Trace: GW-REC-001 reports/a.md -->",
        "<!-- trace : GW-REC-001 reports/a.md -->",
        "<!-- trace:  GW-REC-001 reports/a.md -->",
        "<!-- trace: GW-REC-001 -->",
        "<!-- trace: GW-REC-001 reports/a.md extra -->",
        "<!-- trace GW-REC-001 reports/a.md -->",
        "<!-- trace: GW-REC-1 reports/a.md -->",
        "<!-- trace: REC-001 reports/a.md -->",
        "<!-- trace: reports/b.md reports/a.md -->",
        "<!-- trace: GW-REC-001 reports/a.md# -->",
        "<!-- trace-table: GW-REC-001 reports/a.md -->",
        "<!-- trace-table: GW-REC-001 reports/a.md key -->",
        "<!-- derived: -->",
        "<!-- derived: two words -->",
        "<!-- derived: -leading -->",
        "<!-- DERIVED: name -->",
    ],
)
def test_near_misses_are_errors_not_ignored(comment: str) -> None:
    assert len(nt.parse_document(f"{comment}\n\nText 1.\n").errors) == 1


@pytest.mark.parametrize(
    "path",
    [
        "data/processed/x.json",
        "data/raw/x.json",
        "runs/x/report.md",
        "checkpoints/x.json",
        "glypharis/x.yaml",
        "docs/handoff/CONTEXT.md",
        "reports/../runs/x.md",
        "reports/./x.md",
        "reports//x.md",
        "/reports/x.md",
        "C:/reports/x.md",
        "reports\\x.md",
        "Reports/x.md",
        "data/manifests/../processed/x.json",
        "reports/x.png",
        "reports/x",
    ],
)
def test_source_paths_outside_the_roots_are_refused(path: str) -> None:
    with pytest.raises(nt.SourcePathError):
        nt.check_source_path(path)
    document = nt.parse_document(f"<!-- trace: GW-REC-001 {path} -->\n\nText 1.\n")
    assert len(document.errors) == 1
    assert document.directives == ()


@pytest.mark.parametrize(
    "path",
    [
        "reports/a.md",
        "research/paper/trace/x.json",
        "docs/WORKFLOW.md",
        "configs/final/v.yaml",
        "data/manifests/a.yml",
        "reports/a.csv",
    ],
)
def test_source_paths_inside_the_roots(path: str) -> None:
    assert nt.check_source_path(path) == path


def test_a_trace_applies_to_the_next_block_and_they_stack() -> None:
    text = (
        "Before 1.\n\n<!-- trace: GW-REC-001 reports/a.md -->\n\n"
        "<!-- trace: GW-REC-002 reports/b.md#s -->\nAfter 2.\nstill 3.\n\nOther 4.\n"
    )
    document = nt.parse_document(text)
    assert document.errors == ()
    first, second = document.directives
    assert first.target == second.target
    block = document.blocks[first.target]
    assert (block.kind, block.start_line, block.end_line) == ("paragraph", 6, 7)


def test_a_trace_covers_a_whole_list_and_a_table() -> None:
    text = (
        "<!-- trace: GW-REC-001 reports/a.md -->\n- a 1\n\n- b 2\n\n"
        "<!-- trace: GW-REC-001 reports/a.md -->\n| A |\n|---|\n| 3 |\n"
    )
    document = nt.parse_document(text)
    assert [document.blocks[d.target].kind for d in document.directives] == ["list", "table"]
    assert document.blocks[document.directives[0].target].end_line == 4


def test_a_comment_ends_a_paragraph() -> None:
    document = nt.parse_document("One 1.\n<!-- trace: GW-REC-001 reports/a.md -->\nTwo 2.\n")
    assert document.tokens[0].block != document.tokens[1].block
    assert document.directives[0].target == document.tokens[1].block


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Text <!-- trace: GW-REC-001 reports/a.md --> 1.\n", "stand alone"),
        ("<!-- trace: GW-REC-001 reports/a.md -->\n", "no block"),
        ("<!-- trace: GW-REC-001 reports/a.md -->\n\n# Title\n", "heading"),
        ("<!-- trace: GW-REC-001 reports/a.md -->\n```\n1\n```\n", "code"),
        ("<!-- trace-table: GW-REC-001 reports/a.md a=b -->\n\nText 1.\n", "not a table"),
        ("| A |\n|---|\n| <!-- trace-table: GW-REC-001 reports/a.md a=b --> |\n", "stand alone"),
        ("<!-- derived: x -->\n", "no block"),
        ("<!-- derived: x -->\n\nNo numbers.\n", "no number"),
        ("Text <!-- derived: x --> without.\n", "no number"),
        ("<!-- never closed\n", "never closed"),
    ],
)
def test_misplaced_directives(text: str, message: str) -> None:
    assert message in only_error(text)


def test_every_error_is_collected_with_its_line() -> None:
    text = "<!-- trace: x y -->\n\nText 1.\n\n<!-- derived: -->\n\nText 2.\n"
    assert [e.line for e in nt.parse_document(text).errors] == [1, 5]


@pytest.mark.parametrize(
    "text",
    [
        "`<!-- trace: nonsense -->`\n",
        "```\n<!-- trace: nonsense -->\n```\n",
        "<!-- an ordinary comment about the trace: nothing -->\n",
        "<!-- trace the numbers later -->\n",
    ],
)
def test_examples_and_ordinary_comments_do_nothing(text: str) -> None:
    document = nt.parse_document(text)
    assert document.directives == ()
    assert document.errors == ()


def test_derived_inline_binds_the_next_number_only() -> None:
    document = nt.parse_document("Of 12, about <!-- derived: ratio --> 3.3 times, not 4.\n")
    assert [t.derived for t in document.tokens] == [None, "ratio", None]
    assert document.directives[0].inline


def test_derived_inline_may_reach_the_next_line_of_its_block() -> None:
    document = nt.parse_document("About <!-- derived: ratio -->\n3.3 times.\n")
    assert document.tokens[0].derived == "ratio"


def test_a_comment_touching_a_number_does_not_hide_it() -> None:
    document = nt.parse_document("About <!-- derived: ratio -->3.3<!-- note --> times.\n")
    assert [(t.raw, t.derived) for t in document.tokens] == [("3.3", "ratio")]


def test_a_code_span_touching_a_number_hides_it() -> None:
    assert raws("`name`5 and 6`name`") == []


def test_derived_in_a_table_cell() -> None:
    document = nt.parse_document("| A | B |\n|---|---|\n| 1 | <!-- derived: x --> 2 |\n")
    assert [t.derived for t in document.tokens] == [None, "x"]
    assert document.tokens[1].cell == nt.CellRef(0, 0, 1, "B")


def test_standalone_derived_take_numbers_in_order() -> None:
    text = "<!-- derived: a -->\n<!-- derived: b -->\nFirst 1, then 2, then 3.\n"
    document = nt.parse_document(text)
    assert [t.derived for t in document.tokens] == ["a", "b", None]
    assert not document.directives[0].inline


def test_two_inline_derived_in_one_paragraph() -> None:
    document = nt.parse_document("x <!-- derived: a --> 1 and <!-- derived: b --> 2, then 3.\n")
    assert [t.derived for t in document.tokens] == ["a", "b", None]
    assert document.errors == ()


def test_more_derived_than_numbers_is_an_error() -> None:
    text = "<!-- derived: a -->\n<!-- derived: b -->\nOnly 1.\n"
    document = nt.parse_document(text)
    assert [t.derived for t in document.tokens] == ["a"]
    assert [e.line for e in document.errors] == [2]


# The report index


def test_markdown_values_keep_file_line_section_and_header(tmp_path: Path) -> None:
    write(
        tmp_path,
        "reports/r.md",
        "Generated by `x.py` (GW-REC-001, GW-E2E-002); do not edit.\n\n"
        "# Top\n\nSaw 1,634 pages.\n\n"
        "## gdc_test (GW-REC-003)\n\n| Set | CER % |\n|---|---|\n| a | 6.30 [6.10, 6.50] |\n\n"
        "## other\n\nAlso 7.\n",
    )
    indexed = nt.ReportIndex(tmp_path).file("reports/r.md")
    assert indexed.generated_ids == ("GW-E2E-002", "GW-REC-001")
    assert indexed.ids == ("GW-E2E-002", "GW-REC-001", "GW-REC-003")
    first = indexed.values[0]
    assert (first.file, first.line, first.raw, first.value) == (
        "reports/r.md",
        5,
        "1,634",
        Decimal(1634),
    )
    assert first.where == "top"
    cer = indexed.values[1]
    assert (cer.line, cer.header, cer.table, cer.where) == (11, "CER %", 0, "gdc_test-gw-rec-003")
    interval = indexed.intervals[0]
    assert interval.point is not None
    assert (interval.point.raw, interval.low.raw, interval.high.raw) == ("6.30", "6.10", "6.50")
    assert [v.raw for v in indexed.values_in("gdc_test")] == ["6.30", "6.10", "6.50"]
    assert [v.raw for v in indexed.values_in("other")] == ["7"]
    assert [v.raw for v in indexed.values_in("top")][:2] == ["1,634", "6.30"]
    assert len(indexed.intervals_in("gdc_test")) == 1
    assert indexed.intervals_in("other") == ()


@pytest.mark.parametrize("header", ["Reference", "Prediction", "**Reference**"])
def test_free_text_tables_are_left_out(tmp_path: Path, header: str) -> None:
    text = f"Kept 5.\n\n| Key | {header} | CER |\n|---|---|---|\n| k1 | page 123 | 7.00 |\n"
    indexed = nt.index_text("reports/r.md", text)
    assert [v.raw for v in indexed.values] == ["5"]
    assert indexed.tables == ()


@pytest.mark.parametrize("header", ["Reference class", "Ref chars", "reference", "Predictions"])
def test_tables_merely_named_alike_stay(header: str) -> None:
    text = f"| {header} | Chars |\n|---|---|\n| lower | 1,204 |\n"
    indexed = nt.index_text("reports/r.md", text)
    assert [v.raw for v in indexed.values] == ["1,204"]
    assert len(indexed.tables) == 1


def test_a_table_kept_after_a_free_text_table_keeps_its_rows_selectable() -> None:
    text = "| Reference | N |\n|---|---|\n| a | 9 |\n\n| Set | N |\n|---|---|\n| b | 8 |\n"
    indexed = nt.index_text("reports/r.md", text)
    assert [(v.raw, v.table) for v in indexed.values] == [("8", 0)]
    assert indexed.tables[0].rows[0].cells == ("b", "8")


def test_header_cells_are_not_values() -> None:
    assert nt.index_text("reports/r.md", "| 95 | B |\n|---|---|\n| a | 1 |\n").values[0].raw == "1"
    assert len(nt.index_text("reports/r.md", "| 95 | B |\n|---|---|\n| a | 1 |\n").values) == 1


def test_words_files_have_no_values() -> None:
    text = (
        "Generated by x (GW-REC-040).\n\n56,979 words.\n\n"
        "| Ref word | Count |\n|---|---|\n| a | 3 |\n"
    )
    indexed = nt.index_text("reports/confusion/words_gdc_test.md", text)
    assert indexed.values == ()
    assert indexed.tables == ()
    assert indexed.generated_ids == ("GW-REC-040",)
    assert indexed.sha256


def test_words_json_keeps_its_counts_but_no_strings() -> None:
    text = json.dumps({"totals": {"errors": 8}, "top": [{"ref": "2023.", "count": 13}]})
    indexed = nt.index_text("reports/confusion/words_gdc_test.json", text)
    assert [v.raw for v in indexed.values] == ["8", "13"]


def test_examples_keeps_only_its_counts() -> None:
    text = (
        "Drawn with 40 crops and seed 7 from 200 lines.\n\n## Types\n\n"
        "**1. sub: 1 (U+0031) -> l** (4,941 times on 583 lines; seed [20261115, 1])\n\n"
        "| Key | Errors | Ref chars | Reference | Prediction | Source |\n"
        "|---|---|---|---|---|---|\n"
        "| k | 2 | 30 | page 12 | page 13 | d |\n\n"
        "**2. del: x -> y** (1 times on 1 line; seed [20261115, 2])\n\n"
        "## Worst\n\n| Page | Condition | Bag errors | Ref chars | CER % |\n|---|---|---|---|---|\n"
        "| p/1 | scan 3 | 17 | 1,000 | 1.70 |\n"
    )
    indexed = nt.index_text("reports/error_cases/examples.md", text)
    assert [(v.line, v.raw) for v in indexed.values] == [
        (5, "4,941"),
        (5, "583"),
        (11, "1"),
        (11, "1"),
        (17, "17"),
        (17, "1,000"),
        (17, "1.70"),
    ]
    assert len(indexed.tables) == 1


def test_a_file_elsewhere_named_like_neither_is_read_in_full() -> None:
    assert [v.raw for v in nt.index_text("reports/my_words.md", "Saw 3.\n").values] == ["3"]
    assert [v.raw for v in nt.index_text("reports/examples2.md", "Saw 3.\n").values] == ["3"]


def test_report_interval_needs_one_line() -> None:
    indexed = nt.index_text("reports/r.md", "Value 5.0 [4.0,\n6.0] here.\n")
    assert indexed.intervals == ()
    indexed = nt.index_text("reports/r.md", "Value 5.0\n[4.0, 6.0] here.\n")
    assert indexed.intervals[0].point is None


JSON_TEXT = """{
  "experiment": "GW-REC-007",
  "run": "GW-REC-008 again",
  "note": "12 of 13",
  "flag": true,
  "missing": null,
  "bad": NaN,
  "worse": -Infinity,
  "a.b/c": {"n": 1},
  "metrics": {
    "cer": {
      "value": 0.0603,
      "low": 0.0589,
      "high": 0.0618,
      "unit": "lines",
      "resamples": 1000
    },
    "list": [1.5, {"x": -2e-3}]
  }
}
"""


def test_json_values_lines_and_paths() -> None:
    indexed = nt.index_text("reports/r.json", JSON_TEXT)
    by_path = {v.where: v for v in indexed.values}
    assert set(by_path) == {
        "a.b~1c/n",
        "metrics/cer/value",
        "metrics/cer/low",
        "metrics/cer/high",
        "metrics/cer/resamples",
        "metrics/list/0",
        "metrics/list/1/x",
    }
    cer = by_path["metrics/cer/value"]
    assert (cer.line, cer.raw, cer.value, cer.exponent, cer.header) == (
        12,
        "0.0603",
        Decimal("0.0603"),
        -4,
        "value",
    )
    assert by_path["metrics/list/1/x"].value == Decimal("-0.002")
    assert by_path["metrics/list/1/x"].line == 18
    assert indexed.generated_ids == ("GW-REC-007", "GW-REC-008")


def test_json_intervals_carry_what_was_resampled() -> None:
    interval = nt.index_text("reports/r.json", JSON_TEXT).intervals[0]
    assert interval.point is not None
    assert (interval.point.raw, interval.low.raw, interval.high.raw) == (
        "0.0603",
        "0.0589",
        "0.0618",
    )
    assert (interval.unit, interval.where, interval.line) == ("lines", "metrics/cer", 12)


def test_json_anchors() -> None:
    indexed = nt.index_text("reports/r.json", JSON_TEXT)
    assert [v.raw for v in indexed.values_in("metrics/list")] == ["1.5", "-2e-3"]
    assert [v.raw for v in indexed.values_in("a.b~1c")] == ["1"]
    assert len(indexed.intervals_in("metrics")) == 1
    with pytest.raises(nt.TraceError, match="no values"):
        indexed.values_in("metrics/lis")


def test_invalid_json_is_an_error() -> None:
    with pytest.raises(ValueError):
        nt.index_text("reports/r.json", "{")


def test_csv_gives_whole_numeric_cells_only() -> None:
    text = 'code,detail,n\r\na,area 0.50,"1,234"\r\nb,12,007\r\n'
    indexed = nt.index_text("reports/r.csv", text)
    assert [(v.line, v.raw, v.header) for v in indexed.values] == [
        (2, "1,234", "n"),
        (3, "12", "detail"),
    ]
    assert indexed.tables[0].rows[1].cells == ("b", "12", "007")
    with pytest.raises(nt.TraceError, match="no anchor"):
        indexed.values_in("x")


def test_yaml_scalars_paths_and_comments() -> None:
    text = (
        "---\n# seed: 1\nseed: 11  # the 12th\nname: model 5\nrules:\n  min_chars: 600\n"
        "  quotas:\n    scan: 0.35\n  sizes: [32, 64]\n  items:\n    - 7\n    - k: 8\n"
        "text: '9'\nafter: 10\n"
    )
    indexed = nt.index_text("configs/c.yaml", text)
    found = {(v.where, v.raw) for v in indexed.values}
    assert found == {
        ("seed", "11"),
        ("rules/min_chars", "600"),
        ("rules/quotas/scan", "0.35"),
        ("rules/sizes", "32"),
        ("rules/sizes", "64"),
        ("rules/items", "7"),
        ("rules/items/k", "8"),
        ("after", "10"),
    }
    assert [v.raw for v in indexed.values_in("rules/quotas")] == ["0.35"]
    assert next(v for v in indexed.values if v.raw == "0.35").line == 8


def test_hash_ignores_the_line_ending(tmp_path: Path) -> None:
    write(tmp_path, "reports/lf.md", "a 1\nb 2\n")
    write(tmp_path, "reports/crlf.md", "a 1\r\nb 2\r\n")
    index = nt.ReportIndex(tmp_path)
    expected = hashlib.sha256(b"a 1\nb 2\n").hexdigest()
    assert index.file("reports/lf.md").sha256 == expected
    assert index.file("reports/crlf.md").sha256 == expected
    assert index.file("reports/crlf.md").values[1].line == 2


def test_the_index_refuses_what_it_may_not_read(tmp_path: Path) -> None:
    write(tmp_path, "data/processed/x.json", "{}")
    index = nt.ReportIndex(tmp_path)
    with pytest.raises(nt.SourcePathError):
        index.file("data/processed/x.json")
    with pytest.raises(nt.TraceError, match="no such file"):
        index.file("reports/none.md")


def test_the_index_refuses_a_link_out_of_the_roots(tmp_path: Path) -> None:
    write(tmp_path, "runs/secret.md", "Value 1.\n")
    (tmp_path / "reports").mkdir()
    try:
        os.symlink(tmp_path / "runs" / "secret.md", tmp_path / "reports" / "link.md")
    except OSError:
        pytest.skip("this machine cannot create symbolic links")
    with pytest.raises(nt.SourcePathError, match="link"):
        nt.ReportIndex(tmp_path).file("reports/link.md")


@pytest.mark.parametrize(
    ("anchor", "first"),
    [
        ("gdc_test", 3),
        ("GDC_test", 3),
        ("gdc_test-gw-rec-039", 3),
        ("custom", 9),
        ("gw-rec-022", 9),
        ("track-strict", 5),
    ],
)
def test_section_anchors(anchor: str, first: int) -> None:
    text = (
        "# Top\n\n## gdc_test (GW-REC-039)\n\n### Track: strict\n\n1\n\n"
        '## <a id="gw-rec-022"></a><a name="custom"></a>GW-REC-022 \u2014 run\n\n2\n'
    )
    headings = nt.parse_document(text).headings
    assert nt.section_lines(headings, anchor, 11)[0] == first


def test_a_section_ends_at_the_next_heading_of_its_level() -> None:
    text = "# A\n\n## B\n\n### C\n\n## D\n\n# E\n"
    headings = nt.parse_document(text).headings
    assert nt.section_lines(headings, "b", 9) == (3, 6)
    assert nt.section_lines(headings, "c", 9) == (5, 6)
    assert nt.section_lines(headings, "a", 9) == (1, 8)
    assert nt.section_lines(headings, "e", 9) == (9, 9)


@pytest.mark.parametrize("anchor", ["gdc", "none", "gdc_tes"])
def test_bad_section_anchors(anchor: str) -> None:
    headings = nt.parse_document("## gdc-a x\n\n## gdc-b y\n\n## gdc_test z\n").headings
    with pytest.raises(nt.TraceError, match="fits"):
        nt.section_lines(headings, anchor, 5)


REPORT = """Generated by `x.py` (GW-E2E-009); do not edit.

## Detection

| ID | Set | Pages | F1 @0.5 |
|---|---|---|---|
| GW-E2E-007 | hiertext_test | 1,634 | 79.13 [77.98, 80.26] |
| GW-E2E-007 | gdc_test | 106 | 80.80 [72.63, 88.07] |
| GW-BASE-001 | gdc_test | 106 | 55.18 [47.15, 63.64] |

## Pages

| ID | Set | CER % |
|---|---|---|
| GW-E2E-007 | gdc_test | 25.37 |
"""


@pytest.fixture
def index(tmp_path: Path) -> nt.ReportIndex:
    write(tmp_path, "reports/e2e.md", REPORT)
    write(tmp_path, "reports/r.json", JSON_TEXT)
    write(tmp_path, "reports/t.csv", "set,n\na,5\nb,6\n")
    return nt.ReportIndex(tmp_path)


def test_binding_values_by_row_and_header(index: nt.ReportIndex) -> None:
    binding = nt.parse_bindings("GDC=GW-E2E-007;gdc_test")[0]
    raw = [v.raw for v in index.binding_values("reports/e2e.md", "detection", binding)]
    assert raw == ["106", "80.80", "72.63", "88.07"]
    binding = nt.parse_bindings("GDC=GW-E2E-007;gdc_test,F1 @0.5")[0]
    raw = [v.raw for v in index.binding_values("reports/e2e.md", "detection", binding)]
    assert raw == ["80.80", "72.63", "88.07"]
    binding = nt.parse_bindings("GDC=L15")[0]
    assert [v.raw for v in index.binding_values("reports/e2e.md", None, binding)] == ["25.37"]


@pytest.mark.parametrize(
    ("text", "anchor", "message"),
    [
        ("A=GW-E2E-007;gdc_test", None, "fits 2 rows"),
        ("A=gdc_test", "detection", "fits 2 rows"),
        ("A=nothing", None, "fits 0 rows"),
        ("A=hiertext_test,CER %", "detection", "no column"),
    ],
)
def test_binding_errors(index: nt.ReportIndex, text: str, anchor: str | None, message: str) -> None:
    binding = nt.parse_bindings(text)[0]
    with pytest.raises(nt.TraceError, match=message):
        index.binding_values("reports/e2e.md", anchor, binding)


def test_bindings_on_json_and_csv(index: nt.ReportIndex) -> None:
    binding = nt.parse_bindings("A=cer,value")[0]
    assert [v.raw for v in index.binding_values("reports/r.json", "metrics", binding)] == ["0.0603"]
    binding = nt.parse_bindings("A=metrics/cer")[0]
    assert len(index.binding_values("reports/r.json", None, binding)) == 4
    with pytest.raises(nt.TraceError, match="one term"):
        index.binding_values("reports/r.json", None, nt.parse_bindings("A=a;b")[0])
    with pytest.raises(nt.TraceError, match="no values"):
        index.binding_values("reports/r.json", None, nt.parse_bindings("A=nope")[0])
    assert [
        v.raw for v in index.binding_values("reports/t.csv", None, nt.parse_bindings("A=b")[0])
    ] == ["6"]


def test_row_bound_table(index: nt.ReportIndex) -> None:
    text = (
        "<!-- trace-table: GW-E2E-007 reports/e2e.md#detection "
        "HierText=GW-E2E-007;hiertext_test|GDC=GW-E2E-007;gdc_test -->\n"
        "| Set | Pages | F1 @0.5 | Other |\n|---|---|---|---|\n"
        "| HierText | 1,634 | 79.13 | 80.26 |\n| GDC | 106 | 80.80 | 1 |\n| Unbound | 2 | 3 | 4 |\n"
    )
    document = nt.parse_document(text)
    assert document.errors == ()
    directive = document.directives[0]
    first, second = directive.bindings
    assert [t.raw for t in nt.bound_tokens(document, directive, first)] == [
        "1,634",
        "79.13",
        "80.26",
    ]
    tokens = nt.bound_tokens(document, directive, second)
    assert [t.raw for t in tokens] == ["106", "80.80", "1"]
    pages = index.cell_candidates(document, directive, second, tokens[0])
    assert [v.raw for v in pages] == ["106"]
    score = index.cell_candidates(document, directive, second, tokens[1])
    assert [v.raw for v in score] == ["80.80", "72.63", "88.07"]
    # "Other" is no column of the report, so any cell of the row may match.
    assert len(index.cell_candidates(document, directive, second, tokens[2])) == 4


def test_transposed_table(index: nt.ReportIndex) -> None:
    text = (
        "<!-- trace-table: GW-E2E-007 reports/e2e.md#detection "
        "HierText=GW-E2E-007;hiertext_test|GDC=GW-E2E-007;gdc_test -->\n"
        "| Metric | HierText | GDC |\n|---|---|---|\n| Pages | 1,634 | 106 |\n"
        "| F1 @0.5 | 79.13 | 80.80 |\n"
    )
    document = nt.parse_document(text)
    directive = document.directives[0]
    gdc = directive.bindings[1]
    tokens = nt.bound_tokens(document, directive, gdc)
    assert [t.raw for t in tokens] == ["106", "80.80"]
    assert [v.raw for v in index.cell_candidates(document, directive, gdc, tokens[0])] == ["106"]
    score = index.cell_candidates(document, directive, gdc, tokens[1])
    assert score[0].raw == "80.80"


def test_explicit_header_limits_every_number(index: nt.ReportIndex) -> None:
    text = (
        "<!-- trace-table: GW-E2E-007 reports/e2e.md#pages A=GW-E2E-007;gdc_test,CER % -->\n"
        "| Name | Value |\n|---|---|\n| A | 25.37 |\n"
    )
    document = nt.parse_document(text)
    directive = document.directives[0]
    binding = directive.bindings[0]
    token = nt.bound_tokens(document, directive, binding)[0]
    assert [v.raw for v in index.cell_candidates(document, directive, binding, token)] == ["25.37"]


@pytest.mark.parametrize(
    ("bindings", "message"),
    [("Nope=x", "fits nothing"), ("Same=x", "ambiguous"), ("Both=x", "ambiguous")],
)
def test_bad_document_keys(bindings: str, message: str) -> None:
    text = (
        f"<!-- trace-table: GW-E2E-007 reports/e2e.md {bindings} -->\n"
        "| Name | Both |\n|---|---|\n| Same | 1 |\n| Same | 2 |\n| Both | 3 |\n"
    )
    document = nt.parse_document(text)
    directive = document.directives[0]
    with pytest.raises(nt.DirectiveError, match=message):
        nt.bound_tokens(document, directive, directive.bindings[0])


def test_select_rows_needs_every_term() -> None:
    table = nt.parse_document("| A | B |\n|---|---|\n| x | y |\n| x | z |\n").tables[0]
    assert [r.line for r in nt.select_rows(table, ("x",))] == [3, 4]
    assert [r.line for r in nt.select_rows(table, ("x", "z"))] == [4]
    assert [r.line for r in nt.select_rows(table, ("L3",))] == [3]
    assert nt.select_rows(table, ("x", "q")) == []


def test_clean_cell() -> None:
    assert nt.clean_cell(" **a** ") == "a"
    assert nt.clean_cell("`b`") == "b"
    assert nt.clean_cell("*c*") == "c"
    assert nt.clean_cell(r"d\_e \| f") == "d_e | f"
    assert nt.clean_cell("**") == "**"


# As a whole


def test_only_the_standard_library_is_imported() -> None:
    tree = ast.parse(Path(nt.__file__).read_text(encoding="utf-8"))
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert {name.split(".")[0] for name in names} <= set(sys.stdlib_module_names)


def test_parsing_is_deterministic() -> None:
    text = REPORT + "\nValue 5.0 [4.0, 6.0]\n\n<!-- derived: a -->\nThen 7.\n"
    assert nt.parse_document(text, "x.md") == nt.parse_document(text, "x.md")
    assert nt.index_text("reports/x.md", text) == nt.index_text("reports/x.md", text)


def test_main_lists_a_valid_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(
        tmp_path,
        "d.md",
        "<!-- trace: GW-REC-001 reports/a.md -->\n"
        "It was 6.30% [6.10, 6.50] of <!-- derived: x --> 9.\n",
    )
    assert nt.main([str(tmp_path / "d.md")]) == 0
    out = capsys.readouterr().out
    assert ":2:8 6.30% [paragraph]" in out
    assert "derived=x" in out
    assert "interval line 2: 6.30 6.10 6.50" in out
    assert "directive line 1: trace GW-REC-001 reports/a.md" in out


def test_main_reports_directive_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(tmp_path, "d.md", "<!-- trace: nonsense -->\n\nText.\n")
    assert nt.main([str(tmp_path / "d.md")]) == 1
    assert "line 1" in capsys.readouterr().err
