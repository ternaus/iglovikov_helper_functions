"""Set of general type helper functions"""
from collections import defaultdict
from typing import Any, List


def group_by_key(list_dicts: List[dict], key: Any) -> defaultdict:
    """Groups list of dictionaries by key.

    >>> c = [{"a": 1, "b": "Wednesday"}, {"a": (1, 2, 3), "b": 16.5}]
    defaultdict(list,
            {1: [{'a': 1, 'b': 'Wednesday'}],
             (1, 2, 3): [{'a': (1, 2, 3), 'b': 16.5}]})

    Args:
        list_dicts:
        key:

    Returns:

    """
    groups: defaultdict = defaultdict(list)
    for detection in list_dicts:
        groups[detection[key]].append(detection)
    return groups
