import random

import numpy as np
import torch
from torch.backends import cudnn


def set_determenistic(seed: int = 666, precision: int = 10) -> None:
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)  # type: ignore
    torch.manual_seed(seed)
    torch.set_printoptions(precision=precision)
