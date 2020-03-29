import math

from hypothesis import given
from hypothesis.strategies import integers as h_int, floats as h_float

from iglovikov_helper_functions.utils.box_utils import resize, fix_center, _get_center, _get_left_right, _get_coords

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 99


@given(position=h_int(min_value=0, max_value=IMAGE_WIDTH), width=h_int(min_value=1, max_value=200))
def test_fix_center(position, width):
    new_position = fix_center(position, IMAGE_WIDTH, width=width)

    if width < IMAGE_WIDTH:
        assert math.ceil(width / 2) <= new_position < IMAGE_WIDTH - math.floor(width / 2) + 1

    if width > IMAGE_WIDTH:
        assert new_position == math.ceil(IMAGE_WIDTH / 2)


@given(
    x_min=h_int(min_value=0, max_value=int(IMAGE_WIDTH * 3 / 4) - 1),
    y_min=h_int(min_value=0, max_value=int(IMAGE_HEIGHT * 5 / 6) - 1),
    x_max=h_int(min_value=int(IMAGE_WIDTH * 3 / 4) - 1, max_value=int(IMAGE_WIDTH - 1)),
    y_max=h_int(min_value=int(IMAGE_HEIGHT * 5 / 6), max_value=IMAGE_HEIGHT - 1),
    resize_coeff=h_float(min_value=1, max_value=10),
)
def test_resize(x_min, y_min, x_max, y_max, resize_coeff):
    x_min_1, y_min_1, x_max_1, y_max_1 = resize(
        x_min, y_min, x_max, y_max, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, resize_coeff=resize_coeff
    )

    assert 0 <= x_min_1 < x_max_1 < IMAGE_WIDTH
    assert 0 <= y_min_1 < y_max_1 < IMAGE_HEIGHT

    new_width = min(math.floor(resize_coeff * (x_max - x_min)), IMAGE_WIDTH)

    if new_width < IMAGE_WIDTH:
        assert x_max_1 - x_min_1 == new_width
    else:
        assert x_min_1 == 0
        assert x_max_1 == IMAGE_WIDTH - 1

    new_height = min(math.floor(resize_coeff * (y_max - y_min)), IMAGE_HEIGHT)

    if new_height < IMAGE_HEIGHT:
        assert y_max_1 - y_min_1 == new_height
    else:
        assert y_min_1 == 0
        assert y_max_1 == IMAGE_HEIGHT - 1


@given(x_min=h_int(min_value=0, max_value=21), x_max=h_int(min_value=0, max_value=11))
def test_get_center(x_min, x_max):
    assert _get_center(x_min, x_max) == _get_center(x_max, x_min)

    assert math.floor((x_min + x_max) / 2) == _get_center(x_min, x_max)


@given(center=h_int(0, IMAGE_WIDTH), width=h_int(1, IMAGE_WIDTH))
def test_get_left_right(center, width):
    left, right = _get_left_right(center, width)

    assert right - left == width


@given(
    x_min=h_int(min_value=0, max_value=int(IMAGE_WIDTH * 3 / 4) - 1),
    x_max=h_int(min_value=int(IMAGE_WIDTH * 3 / 4), max_value=int(IMAGE_WIDTH - 1)),
    resize_coeff=h_float(min_value=0.1, max_value=10),
)
def test_get_coords(x_min, x_max, resize_coeff):
    assert x_min < x_max
    x_min_1, x_max_1 = _get_coords(x_min, x_max, resize_coeff, IMAGE_WIDTH)

    assert x_max_1 - x_min_1 == min(IMAGE_WIDTH - 1, int(resize_coeff * (x_max - x_min)))
