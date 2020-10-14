import re
from pathlib import Path
from typing import Union, Optional, Dict, Any

import torch


def state_dict_from_disk(file_path: Union[Path, str], rename_in_layers: Optional[dict] = None) -> Dict[str, Any]:
    """Loads PyTorch checkpoint from disk, optionally renaming layer names.

    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if rename_in_layers is not None:

        result = {}
        for key, value in state_dict.items():
            for key_r, value_r in rename_in_layers.items():
                key = re.sub(key_r, value_r, key)

            result[key] = value

        state_dict = result

    return state_dict
