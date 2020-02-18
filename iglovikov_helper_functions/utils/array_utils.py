"""
Utils for working with arrays and lists.
"""
from typing import Tuple


def split_array(array_length: int, num_splits: int, split_id: int) -> Tuple[int, int]:
    """Split array into parts.

    Args:
        array_length:
        num_splits:
        split_id:

    Returns: start and end indices of the

    """
    if not 0 <= split_id < num_splits:
        raise ValueError(f"gpu_id should be 0 <= {split_id} < {num_splits}")
    if array_length % num_splits == 0:
        step = int(array_length / num_splits)
    else:
        step = int(array_length / num_splits) + 1

    return split_id * step, min((split_id + 1) * step, array_length)
