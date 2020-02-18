import pytest
from iglovikov_helper_functions.utils.array_utils import split_array


@pytest.mark.parametrize(["array_length", "num_splits"], [(10, 2), (10, 3)])
def test_split_array(array_length, num_splits):
    origin = list(range(array_length))
    target = []
    for split_id in range(num_splits):
        start, end = split_array(array_length, num_splits, split_id)
        target += origin[start:end]

    assert origin == target
