import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm


@torch.no_grad()
def test_acc(model: nn.Module, loader: DataLoader,
             device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    total_loss = 0
    total_acc = 0
    criterion = nn.CrossEntropyLoss().to(device)
    model.to(device)
    denominator = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pre = model(x)
        total_loss += criterion(pre, y).item() * y.shape[0]
        _, pre = torch.max(pre, dim=1)
        total_acc += torch.sum((pre == y)).item()
        denominator += y.shape[0]

    test_loss = total_loss / denominator
    test_accuracy = total_acc / denominator
    print(f'loss = {test_loss}, acc = {test_accuracy}')
    return test_loss, test_accuracy


def test_autoattack_acc(model: nn.Module, loader: DataLoader):
    from autoattack import AutoAttack
    adversary = AutoAttack(model, eps=8 / 255)
    # adversary = AutoAttack(model, eps=0.01)
    xs, ys = [], []
    for x, y in tqdm(loader):
        xs.append(x.cuda())
        ys.append(y.cuda())
    x = torch.concat(xs, dim=0)[:10]
    y = torch.concat(ys, dim=0)[:10]
    adversary.run_standard_evaluation(x, y, bs=1)
