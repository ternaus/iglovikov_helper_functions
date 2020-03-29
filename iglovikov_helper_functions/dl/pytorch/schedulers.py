import math
from typing import List

import numpy as np
import torch
from torch.optim.optimizer import Optimizer


def poly_lr_scheduler(
    optimizer: Optimizer, gamma: float = 0.9, max_epoch: int = 100
) -> torch.optim.lr_scheduler._LRScheduler:
    """Set the learning rate of each parameter group to the initial lr polynomially decayed with the power of
    gamma on every epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Decay learning rate.
        max_epoch (int): Index of the maximum epoch.
    """

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - (epoch / max_epoch) ** gamma))


def get_curve(lr_min: float, lr_max: float, epoch_min: int, epoch_max: int) -> np.array:
    epochs = np.arange(epoch_max - epoch_min)
    return (0.5 * (lr_max - lr_min) * (np.cos(epochs * math.pi / (epoch_max - epoch_min)) + 1) + lr_min).tolist()


def n01z3_sheduler(optimizer: Optimizer, lrs: List[float], epochs: List[int]) -> torch.optim.lr_scheduler._LRScheduler:
    """Creates scheduler with cos waves.

    >>> scheduler = n01z3_sheduler(optimizer, lrs=[1, 0.7, 0.4, 0.1, 0], epochs=[0, 20, 40, 60, 80])

    Args:
        optimizer:
        lrs:
        epochs:

    Returns:

    """

    lrs = sorted(lrs, reverse=True)
    epochs = sorted(epochs)

    if len(lrs) != len(epochs):
        raise ValueError(f"len(lrs) = {len(lrs)} but len(epochs) = {len(epochs)}")

    result: List[float] = []

    for i in range(len(lrs) - 1):
        lr_min = lrs[i + 1]
        lr_max = lrs[i]
        epoch_min = epochs[i]
        epoch_max = epochs[i + 1]

        curve = get_curve(lr_min, lr_max, epoch_min, epoch_max)
        result += curve

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: result[epoch])
