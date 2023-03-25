import torch
from torch import nn

def default_optimizer(model: nn.Module, lr=1e-1, ) -> torch.optim.Optimizer:
    # return torch.optim.Adam(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    return torch.optim.SGD(model.parameters(), lr=lr)


def default_lr_scheduler(optimizer):
    from .ALRS import ALRS
    return ALRS(optimizer)