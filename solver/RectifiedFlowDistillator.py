import torch
from torch import nn
import random
from tqdm import tqdm
import math
from typing import Callable


class ReFlow():
    def __init__(self,
                 sampler: Callable,
                 unet: nn.Module,
                 criterion=nn.MSELoss(),
                 device=torch.device('cuda'),
                 T=1000,
                 batch_size=64,
                 pairs=4000000,
                 ):
        self.unet = unet.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        self.T = T
        self.batch_size = batch_size
        self.sampler = sampler
        self.pairs = pairs
        self.prepare_samples()

    def transform(self, x):
        return (x - 0.5) * 2

    def init(self):
        # init
        self.unet.eval().requires_grad_(False).to(self.device)

    def prepare_samples(self, sampling_batch_size=256):
        x0, x1 = [], []
        for _ in tqdm(range(math.ceil(self.pairs / sampling_batch_size))):
            now_x0, now_x1 = self.sampler.sample(batch_size=sampling_batch_size, return_initial_value=True)
            x0.append(now_x0)
            x1.append(now_x1)
        x0, x1 = torch.stack(x0), torch.stack(x1)
        self.x0 = x0
        self.x1 = self.transform(x1)
        print(f'we have sampled all the training pairs, total {self.pairs}')

    def train(self, train_loader=None, total_epoch=100000,
              p_uncondition=1,
              fp16=False):
        self.unet.train()
        self.unet.requires_grad_(True)
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        for epoch in range(1, total_epoch + 1):
            epoch_loss = 0
            for step in tqdm(range(self.pairs // self.batch_size)):
                index = torch.randint(low=0, high=self.pairs, device=self.device, size=(self.batch_size,))
                target, x = self.x0[index], self.x1[index]
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
            torch.save(self.unet.state_dict(), 'reflow.pt')

        self.init()
