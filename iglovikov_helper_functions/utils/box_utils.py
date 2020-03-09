import math
from typing import Tuple


def fix_center(position: int, distance: int, width: int) -> int:
    left = math.ceil(width / 2)
    right = math.floor(width / 2)

    if width > distance:
        return math.ceil(distance / 2)

    if position < left:  # we are too close to the left border
        if position + left < distance:
            position = left
        else:  # we are also close to the right border
            position = math.ceil(distance / 2)

    elif position > distance - right:  # we are too close to the right border
        position = distance - right

    return position


def _get_center(x_min: int, x_max: int) -> int:
    return math.floor((x_min + x_max) / 2)


def _get_left_right(center: int, width: int) -> Tuple[int, int]:
    left_lag = math.ceil(width / 2)

    left = center - left_lag
    left = max(left, 0)
    right = left + width

    return left, right


def _get_coords(x_min: int, x_max: int, resize_coeff: float, image_width: int) -> Tuple[int, int]:
    old_width = x_max - x_min

    new_width = math.floor(old_width * resize_coeff)

    if new_width >= image_width:
        return 0, image_width - 1

    center_x = _get_center(x_min, x_max)

    if center_x < math.floor(new_width / 2):
        return 0, new_width

    if center_x + new_width >= image_width:
        return image_width - new_width - 1, image_width - 1

    x_min, x_max = _get_left_right(center_x, new_width)

    return x_min, x_max


def resize(
    x_min: int, y_min: int, x_max: int, y_max: int, image_height: int, image_width: int, resize_coeff: float = 1
) -> Tuple[int, int, int, int]:
    """Change the size of the bounding box.

    Args:
        x_min:
        y_min:
        x_max:
        y_max:
        image_height:
        image_width:
        resize_coeff:

    Returns:

    """
    if not 0 <= x_min < x_max < image_width:
        raise ValueError(f"We want 0 < x_min < x_max < image_width" f"But we got: {x_min} {x_max} {image_width}")

    if not 0 <= y_min < y_max < image_height:
        raise ValueError(f"We want 0 < y_min < y_max < image_height" f"But we got: {y_min} {y_max} {image_height}")

    if resize_coeff < 0:
        raise ValueError(f"Resize coefficient should be positive, but we got {resize_coeff}")

    x_min, x_max = _get_coords(x_min, x_max, resize_coeff, image_width)
    y_min, y_max = _get_coords(y_min, y_max, resize_coeff, image_height)

    return x_min, y_min, x_max, y_max
