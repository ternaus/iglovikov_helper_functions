"""Set of general type helper functions"""
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Union


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


def get_sha256(file_path: Union[str, Path]) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
