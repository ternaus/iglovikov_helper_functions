from typing import Tuple


def _get_coords(x_min: int, x_max: int, resize_coeff: float, image_width: int) -> Tuple[int, int]:
    pad = int((resize_coeff - 1) * (x_max - x_min) / 2)

    return max(x_min - pad, 0), min(x_max + pad, image_width - 1)


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
        raise ValueError(f"We want 0 <= x_min < x_max < image_width" f"But we got: {x_min} {x_max} {image_width}")

    if not 0 <= y_min < y_max < image_height:
        raise ValueError(f"We want 0 <= y_min < y_max < image_height" f"But we got: {y_min} {y_max} {image_height}")

    if resize_coeff < 1:
        raise ValueError(f"Resize coefficient should be positive, but we got {resize_coeff}")

    x_min, x_max = _get_coords(x_min, x_max, resize_coeff, image_width)
    y_min, y_max = _get_coords(y_min, y_max, resize_coeff, image_height)

    return x_min, y_min, x_max, y_max
