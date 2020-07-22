import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays as h_arrays
from hypothesis.strategies import integers as h_int

from iglovikov_helper_functions.utils.image_utils import pad, unpad, unpad_from_size, pad_to_size


@given(input_array=h_arrays(dtype=np.uint8, shape=(351, 619)), factor=h_int(min_value=1, max_value=65))
def test_pad_grayscale(input_array, factor):
    padded_array, pads = pad(input_array, factor)
    unpadded_array = unpad(padded_array, pads)

    assert np.array_equal(input_array, unpadded_array)


@given(input_array=h_arrays(dtype=np.uint8, shape=(174, 413, 3)), factor=h_int(min_value=1, max_value=65))
def test_pad_rgb(input_array, factor):
    padded_array, pads = pad(input_array, factor)
    unpadded_array = unpad(padded_array, pads)

    assert np.array_equal(input_array, unpadded_array)


@given(
    image=h_arrays(dtype=np.uint8, shape=(174, 413, 3)),
    bboxes=h_arrays(dtype=int, shape=(9, 4), elements=h_int(min_value=0, max_value=173)),
    keypoints=h_arrays(dtype=int, shape=(7, 2), elements=h_int(min_value=0, max_value=173)),
    target_height=h_int(min_value=174, max_value=300),
    target_width=h_int(min_value=413, max_value=500),
)
def test_pad_to_size(image, bboxes, keypoints, target_height, target_width):
    target_size = (target_height, target_width)
    padded_dict = pad_to_size(target_size, image, bboxes, keypoints)

    unpadded_dict = unpad_from_size(**padded_dict)

    assert np.array_equal(image, unpadded_dict["image"])
    assert np.array_equal(bboxes, unpadded_dict["bboxes"]), f"{bboxes} {unpadded_dict['bboxes']}"
    assert np.array_equal(keypoints, unpadded_dict["keypoints"])
