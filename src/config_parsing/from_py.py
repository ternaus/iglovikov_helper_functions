"""Parse python config to the dictionary. Inspired by https://github.com/open-mmlab/mmdetection."""

import sys
from importlib import import_module
from pathlib import Path


def py2dict(file_path: (str, Path)) -> dict:
    """Convert python file to dictionary.
    The main use - config parser.

    file:
    ```
    a = 1
    b = 3
    c = range(10)
    ```
    will be converted to

    {'a':1,
     'b':3,
     'c': range(10)
    }

    Args:
        file_path: path to the original python file.

    Returns: {key: value}, where key - all variables defined in the file and value is their value.

    """

    file_path = Path(file_path).absolute()

    assert file_path.suffix == '.py', f"Only Py file can be parsed, but got {file_path.name} instead."

    assert file_path.exists(), f"There is no file at the path {file_path}"

    module_name = file_path.stem

    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')

    config_dir = str(file_path.parent)

    sys.path.insert(0, config_dir)

    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }

    return cfg_dict
