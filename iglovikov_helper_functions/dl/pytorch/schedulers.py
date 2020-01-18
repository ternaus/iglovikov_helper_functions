import torch


def poly_lr_scheduler(optimizer, gamma: float = 0.9, max_epoch: int = 100):
    """Set the learning rate of each parameter group to the initial lr polynomially decayed with the power of
    gamma on every epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Decay learning rate.
        max_epoch (int): Index of the maximum epoch.
    """

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: (1 - (epoch / max_epoch) ** gamma))
