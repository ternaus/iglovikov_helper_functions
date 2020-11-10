import math

import numpy as np
from hypothesis import given
from hypothesis.strategies import floats as h_float
from hypothesis.strategies import integers as h_int
from shapely.geometry import Polygon

from iglovikov_helper_functions.utils.geometry_utils import (
    Line,
    intersection_rectangles,
)


class VectorReference:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        return VectorReference(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        return VectorReference(self.x - v.x, self.y - v.y)

    def cross(self, v):
        return self.x * v.y - self.y * v.x

    def __repr__(self):
        return f"x = {self.x}, y = {self.y}"


class LineReference:
    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        return self.a * p.x + self.b * p.y + self.c

    def intersection(self, other):
        # See e.g.     https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates
        w = self.a * other.b - self.b * other.a

        return VectorReference((self.b * other.c - self.c * other.b) / w, (self.c * other.a - self.a * other.c) / w)

    def __repr__(self):
        return f"{self.a} * x + {self.b} * x + {self.c} = 0"


def test_a():
    rectangle_a = np.array(
        [
            [518.54893549, 2660.18852229],
            [516.6680813, 2659.68467469],
            [516.90002051, 2658.81884971],
            [518.7808747, 2659.32269731],
        ]
    )

    rectangle_b = np.array(
        [
            [518.53730179, 2660.30353692],
            [516.60027552, 2659.72004892],
            [516.91754532, 2658.66679687],
            [518.85457159, 2659.25028488],
        ]
    )

    v0 = VectorReference(rectangle_a[0][0], rectangle_a[0][1])
    v1 = VectorReference(rectangle_a[1][0], rectangle_a[1][1])
    v2 = VectorReference(rectangle_a[2][0], rectangle_a[2][1])

    l1 = LineReference(v0, v1)
    l1a = Line([rectangle_a[0][0], rectangle_a[0][1]], [rectangle_a[1][0], rectangle_a[1][1]])

    l3 = LineReference(v1, v2)
    l3a = Line([rectangle_a[1][0], rectangle_a[1][1]], [rectangle_a[2][0], rectangle_a[2][1]])

    assert l1.a == l1a.a
    assert l1.b == l1a.b
    assert l1.c == l1a.c

    assert l1(v2) == l1a([rectangle_a[2][0], rectangle_a[2][1]])

    assert math.isclose(l1.intersection(l3).x, v1.x, abs_tol=1e-6)
    assert math.isclose(l1.intersection(l3).y, v1.y, abs_tol=1e-6)

    assert math.isclose(l1a.intersection(l3a)[0], rectangle_a[1][0], abs_tol=1e-6)
    assert math.isclose(l1a.intersection(l3a)[1], rectangle_a[1][1], abs_tol=1e-6)

    assert math.isclose(
        intersection_rectangles(rectangle_a, rectangle_b),
        Polygon(rectangle_a).intersection(Polygon(rectangle_b)).area,
        abs_tol=1e-6,
    )
    assert math.isclose(
        intersection_rectangles(rectangle_b, rectangle_a),
        Polygon(rectangle_a).intersection(Polygon(rectangle_b)).area,
        abs_tol=1e-6,
    )


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


@given(
    length_a=h_int(min_value=1, max_value=100),
    width_a=h_int(min_value=1, max_value=100),
    center_ax=h_float(min_value=-100, max_value=100),
    center_ay=h_float(min_value=-100, max_value=100),
    length_b=h_int(min_value=1, max_value=100),
    width_b=h_int(min_value=1, max_value=100),
    center_bx=h_float(min_value=-100, max_value=100),
    center_by=h_float(min_value=-100, max_value=100),
    angle_x=h_float(min_value=0, max_value=math.pi),
    angle_y=h_float(min_value=0, max_value=math.pi),
)
def test_vs_shapely_angle(
    length_a, width_a, center_ax, center_ay, length_b, width_b, center_bx, center_by, angle_x, angle_y
):
    rectangle_a = to_coords((center_ax, center_ay), length_a, width_a, angle_x)
    rectangle_b = to_coords((center_bx, center_by), length_b, width_b, angle_y)

    rectangle_a_shapely = Polygon(rectangle_a + [rectangle_a[0]])
    rectangle_b_shapely = Polygon(rectangle_b + [rectangle_b[0]])

    assert math.isclose(
        rectangle_a_shapely.intersection(rectangle_b_shapely).area,
        intersection_rectangles(rectangle_a, rectangle_b),
        abs_tol=1e-4,
    )


def test_simple():
    rectangle_a = np.array(
        [
            [518.54893549, 2660.18852229],
            [516.6680813, 2659.68467469],
            [516.90002051, 2658.81884971],
            [518.7808747, 2659.32269731],
        ]
    )

    rectangle_b = np.array(
        [
            [518.53730179, 2660.30353692],
            [516.60027552, 2659.72004892],
            [516.91754532, 2658.66679687],
            [518.85457159, 2659.25028488],
        ]
    )

    assert intersection_rectangles(rectangle_a, rectangle_b) == 1.745352555764839
