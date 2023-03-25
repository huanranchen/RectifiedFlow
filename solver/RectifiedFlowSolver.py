import torch
from torch import nn
import random
from tqdm import tqdm
import math


class RectifiedFlowSolver():
    def __init__(self, unet: nn.Module,
                 criterion=nn.MSELoss(),
                 device=torch.device('cuda'),
                 T=1000):
        self.unet = unet.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        self.T = T

    def transform(self, x):
        return (x - 0.5) * 2

    def init(self):
        # init
        self.unet.eval().requires_grad_(False).to(self.device)

    def train(self, train_loader, total_epoch=100000,
              p_uncondition=1,
              fp16=False):
        self.unet.train()
        self.unet.requires_grad_(True)
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        for epoch in range(1, total_epoch + 1):
            epoch_loss = 0
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.cuda(), y.cuda()
                # some preprocess
                x = self.transform(x)
                # train
                x, y = x.to(self.device), y.to(self.device)
                target = torch.randn_like(x)
                N, C, H, D = x.shape
                t = torch.rand((N,), device=self.device)
                d = t.view(N, 1, 1, 1) * x + (1 - t).view(N, 1, 1, 1) * target
                vector = x - target
                for i in range(1):
                    if fp16:
                        with autocast():
                            pre = self.unet(d, (t * self.T).to(torch.int))[:, :3, :, :]
                            loss = self.criterion(pre, vector)
                    else:
                        pre = self.unet(d, (t * self.T).to(torch.int))[:, :3, :, :]
                        loss = self.criterion(pre, vector)
                    if fp16:
                        raise NotImplementedError
                        pass
                    else:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                epoch_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'step {step}, loss {epoch_loss / step}')
            print(f'epoch {epoch}, loss {epoch_loss / len(train_loader)}')
            torch.save(self.unet.state_dict(), 'unet.pt')

        self.init()
