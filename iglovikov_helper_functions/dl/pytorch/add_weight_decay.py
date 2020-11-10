from typing import Any, Dict, List, Tuple

from torch import nn


def add_weight_decay(model: nn.Module, weight_decay: float = 1e-5, skip_list: Tuple = ()) -> List[Dict[str, Any]]:
    """Remove weight decay from BatchNorm and Bias parameters.

    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3

    Args:
        model:
        weight_decay:
        skip_list:

    Returns:

    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]
