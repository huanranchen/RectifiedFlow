import torch
from torch.utils.data import DataLoader
from torch import Tensor


def most_similar(x: Tensor, loader: DataLoader) -> Tensor:
    xs = []
    for now_x, _ in loader:
        xs.append(now_x.cuda())
    xs = torch.cat(xs, dim=0)
    N, C, H, D = xs.shape
    d = ((x.squeeze() - xs) ** 2).view(N, C * H * D).sum(1)
    min_index = torch.min(d, dim=0)[1]
    target = xs[min_index]
    return target.unsqueeze(0)
