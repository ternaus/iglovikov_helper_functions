from torch.optim.lr_scheduler import _LRScheduler


class PolylLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr polynomially decayed with the power of
    gamma on every epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Decay learning rate.
        max_epoch (int): Index of the maximum epoch.
        last_epoch (int): The index of last epoch.
    """

    def __init__(self, optimizer, gamma, max_epoch, last_epoch=-1):
        self.gamma = gamma
        self.max_epoch = max_epoch
        super(PolylLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_epoch) ** self.gamma for base_lr in self.base_lrs]
