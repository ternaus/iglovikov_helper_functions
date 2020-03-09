import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays as h_arrays
from hypothesis.strategies import integers as h_int

from iglovikov_helper_functions.utils.image_utils import pad, unpad


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
