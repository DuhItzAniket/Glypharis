"""Reading order: the layouts Phase 10's plan names."""

from __future__ import annotations

import math

import numpy as np
from det_fixtures import rect_polygon

from glypharis.pipeline.reading_order import dominant_angle, reading_order


def _q(x0: float, y0: float, w: float, h: float = 20.0) -> list[list[float]]:
    return [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]


def _rotate(quads: list[list[list[float]]], deg: float, c=(500.0, 500.0)) -> list:
    t = math.radians(deg)
    r = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return [((np.asarray(q) - c) @ r.T + c).tolist() for q in quads]


def test_empty_and_single():
    assert reading_order([]) == []
    assert reading_order([_q(0, 0, 100)]) == [0]


def test_single_column_reads_top_to_bottom_whatever_the_input_order():
    lines = [_q(50, 40 + 30 * k, 400) for k in range(6)]
    shuffled = [lines[i] for i in (3, 0, 5, 1, 4, 2)]
    assert [(3, 0, 5, 1, 4, 2)[i] for i in reading_order(shuffled)] == list(range(6))


def test_ragged_lines_stay_one_column():
    widths = [400, 250, 380, 120, 300, 390]
    lines = [_q(50 + (k % 2) * 15, 40 + 30 * k, w) for k, w in enumerate(widths)]
    assert reading_order(lines) == list(range(6))


def test_two_columns_read_column_by_column_even_when_rows_align():
    left = [_q(50, 40 + 30 * k, 300) for k in range(5)]
    right = [_q(420, 40 + 30 * k, 300) for k in range(5)]
    lines = [x for pair in zip(left, right, strict=True) for x in pair]  # interleaved input
    order = reading_order(lines)
    assert order == [0, 2, 4, 6, 8, 1, 3, 5, 7, 9]


def test_mixed_blocks_header_two_columns_footer():
    header = _q(50, 10, 670, 30)
    left = [_q(50, 80 + 30 * k, 300) for k in range(4)]
    right = [_q(420, 80 + 30 * k, 300) for k in range(4)]
    footer = _q(50, 260, 670)
    lines = [footer, *right, header, *left]
    order = reading_order(lines)
    assert order == [5, 6, 7, 8, 9, 1, 2, 3, 4, 0]


def test_rotated_page_reads_like_the_straight_one():
    left = [_q(100, 100 + 30 * k, 300) for k in range(4)]
    right = [_q(470, 100 + 30 * k, 300) for k in range(4)]
    lines = left + right
    for deg in (-12.0, 8.0):
        assert reading_order(_rotate(lines, deg)) == list(range(8)), deg


def test_rotated_block_inside_a_straight_page():
    body = [_q(50, 40 + 30 * k, 400) for k in range(5)]
    stamp = rect_polygon(600, 120, 160, 20, -30.0)
    order = reading_order([*body, stamp])
    assert order[:5] == list(range(5)) and order[5] == 5


def test_lines_on_one_row_read_left_to_right():
    row = [_q(300, 50, 100), _q(50, 52, 100), _q(170, 48, 100)]
    assert reading_order(row) == [1, 2, 0]


def test_dominant_angle_ignores_reading_direction():
    lines = [np.asarray(q) for q in _rotate([_q(100, 100 + 30 * k, 300) for k in range(4)], 10)]
    assert abs(dominant_angle(lines) - 10) < 0.5
    flipped = [q[[2, 3, 0, 1]] for q in lines]  # read upside down
    assert abs(dominant_angle(flipped) - 10) < 0.5
