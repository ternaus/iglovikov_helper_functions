from typing import List, Tuple, Union

import numpy as np


class Line:
    # ax + by + c = 0
    def __init__(self, v1: np.ndarray, v2: np.ndarray) -> None:
        self.a = v2[1] - v1[1]
        self.b = v1[0] - v2[0]
        self.c = np.cross(v2, v1)

    def __call__(self, point: np.ndarray) -> bool:
        """
        Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        any point p with line(p) > 0 is on the "outside".


        Args:
            point:

        Returns:

        """
        return self.a * point[0] + self.b * point[1] + self.c

    def intersection(self, other: "Line") -> Tuple[float, float]:
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Using_homogeneous_coordinates

        w = self.a * other.b - self.b * other.a

        return (self.b * other.c - self.c * other.b) / w, (self.c * other.a - self.a * other.c) / w

    def __repr__(self):
        return f"({self.a}) * x + ({self.b}) * y + ({self.c}) = 0"


def intersection_rectangles(
    rectangle_a: Union[np.ndarray, List[List[float]]], rectangle_b: Union[np.ndarray, List[List[float]]]
) -> float:
    """Finds intersection of two rotated rectangles.

    https://stackoverflow.com/a/45268241/2276600

    Args:
        rectangle_a: 4 (x, y) points in order
        rectangle_a: 4 (x, y) points in order

    Returns:

    """

    if isinstance(rectangle_a, np.ndarray):
        rectangle_a = rectangle_a.tolist()

    if isinstance(rectangle_b, np.ndarray):
        rectangle_b = rectangle_b.tolist()

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rectangle_a

    # Loop over the edges of the second rectangle
    for p, q in zip(rectangle_b, rectangle_b[1:] + rectangle_b[:1]):
        if len(intersection) <= 2:
            break  # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]

        for s, t, s_value, t_value in zip(
            intersection, intersection[1:] + intersection[:1], line_values, line_values[1:] + line_values[:1]
        ):
            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    return 0.5 * sum(p[0] * q[1] - p[1] * q[0] for p, q in zip(intersection, intersection[1:] + intersection[:1]))
