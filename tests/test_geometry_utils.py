import math

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats as h_float
from hypothesis.strategies import integers as h_int
from shapely.geometry import Polygon

from iglovikov_helper_functions.utils.geometry_utils import intersection_rectangles


def to_coords(center, length, width, angle):
    dx = length / 2
    dy = width / 2
    dxcos = dx * math.cos(angle)
    dxsin = dx * math.sin(angle)
    dycos = dy * math.cos(angle)
    dysin = dy * math.sin(angle)

    center = np.array(center)

    return [
        center + np.array([-dxcos + dysin, -dxsin - dycos]),
        center + np.array([dxcos + dysin, dxsin - dycos]),
        center + np.array([dxcos - dysin, dxsin + dycos]),
        center + np.array([-dxcos - dysin, -dxsin + dycos]),
    ]


@given(
    length_a=h_int(min_value=1, max_value=100),
    width_a=h_int(min_value=1, max_value=100),
    center_ax=h_float(min_value=-100, max_value=100),
    center_ay=h_float(min_value=-100, max_value=100),
    length_b=h_int(min_value=1, max_value=100),
    width_b=h_int(min_value=1, max_value=100),
    center_bx=h_float(min_value=-100, max_value=100),
    center_by=h_float(min_value=-100, max_value=100),
)
def test_vs_shapely_zero_angle(length_a, width_a, center_ax, center_ay, length_b, width_b, center_bx, center_by):
    rectangle_a = to_coords((center_ax, center_ay), length_a, width_a, 0)
    rectangle_b = to_coords((center_bx, center_by), length_b, width_b, 0)

    rectangle_a_shapely = Polygon(rectangle_a + [rectangle_a[0]])
    rectangle_b_shapely = Polygon(rectangle_b + [rectangle_b[0]])

    assert math.isclose(
        rectangle_a_shapely.intersection(rectangle_b_shapely).area, intersection_rectangles(rectangle_a, rectangle_b)
    )


@given(
    length_a=h_int(min_value=1, max_value=100),
    width_a=h_int(min_value=1, max_value=100),
    center_ax=h_float(min_value=-100, max_value=100),
    center_ay=h_float(min_value=-100, max_value=100),
    length_b=h_int(min_value=1, max_value=100),
    width_b=h_int(min_value=1, max_value=100),
    center_bx=h_float(min_value=-100, max_value=100),
    center_by=h_float(min_value=-100, max_value=100),
)
def test_vs_shapely_pi_by_two(length_a, width_a, center_ax, center_ay, length_b, width_b, center_bx, center_by):
    rectangle_a = to_coords((center_ax, center_ay), length_a, width_a, 0)
    rectangle_b = to_coords((center_bx, center_by), length_b, width_b, math.pi / 2)

    rectangle_a_shapely = Polygon(rectangle_a + [rectangle_a[0]])
    rectangle_b_shapely = Polygon(rectangle_b + [rectangle_b[0]])

    assert math.isclose(
        rectangle_a_shapely.intersection(rectangle_b_shapely).area, intersection_rectangles(rectangle_a, rectangle_b)
    )


@given(
    length_a=h_int(min_value=1, max_value=100),
    width_a=h_int(min_value=1, max_value=100),
    center_ax=h_float(min_value=-100, max_value=100),
    center_ay=h_float(min_value=-100, max_value=100),
    length_b=h_int(min_value=1, max_value=100),
    width_b=h_int(min_value=1, max_value=100),
    center_bx=h_float(min_value=-100, max_value=100),
    center_by=h_float(min_value=-100, max_value=100),
)
def test_vs_shapely_pi_by_4(length_a, width_a, center_ax, center_ay, length_b, width_b, center_bx, center_by):
    rectangle_a = to_coords((center_ax, center_ay), length_a, width_a, 0)
    rectangle_b = to_coords((center_bx, center_by), length_b, width_b, math.pi / 4)

    rectangle_a_shapely = Polygon(rectangle_a + [rectangle_a[0]])
    rectangle_b_shapely = Polygon(rectangle_b + [rectangle_b[0]])

    assert math.isclose(
        rectangle_a_shapely.intersection(rectangle_b_shapely).area, intersection_rectangles(rectangle_a, rectangle_b)
    )

    rectangle_a_shapely = Polygon(rectangle_a + [rectangle_a[0]])
    rectangle_b_shapely = Polygon(rectangle_b + [rectangle_b[0]])

    assert math.isclose(
        rectangle_a_shapely.intersection(rectangle_b_shapely).area, intersection_rectangles(rectangle_a, rectangle_b)
    )


@given(
    length_a=h_int(min_value=1, max_value=100),
    width_a=h_int(min_value=1, max_value=100),
    center_ax=h_float(min_value=-100, max_value=100),
    center_ay=h_float(min_value=-100, max_value=100),
    length_b=h_int(min_value=1, max_value=100),
    width_b=h_int(min_value=1, max_value=100),
    center_bx=h_float(min_value=-100, max_value=100),
    center_by=h_float(min_value=-100, max_value=100),
)
def test_vs_shapely_pi_by_6(length_a, width_a, center_ax, center_ay, length_b, width_b, center_bx, center_by):
    rectangle_a = to_coords((center_ax, center_ay), length_a, width_a, 0)
    rectangle_b = to_coords((center_bx, center_by), length_b, width_b, math.pi / 6)

    rectangle_a_shapely = Polygon(rectangle_a + [rectangle_a[0]])
    rectangle_b_shapely = Polygon(rectangle_b + [rectangle_b[0]])

    assert math.isclose(
        rectangle_a_shapely.intersection(rectangle_b_shapely).area, intersection_rectangles(rectangle_a, rectangle_b)
    )


def test_simple():
    rectangle_a = to_coords((0, 0), 1, 1, 0)
    rectangle_b = to_coords((0.5, 0.5), math.sqrt(2), math.sqrt(2), math.pi / 4)

    assert intersection_rectangles(rectangle_a, rectangle_b) == 0.5
