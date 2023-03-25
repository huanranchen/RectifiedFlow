import torch
from torch import nn
from torchvision import transforms
import numpy as np
import math


class VanillaSampler(nn.Module):
    def __init__(self,
                 unet: nn.Module,
                 img_shape=(3, 32, 32),
                 T=1000,
                 dt=0.001,
                 std=0.5,
                 mean=0.5,
                 ):
        super(VanillaSampler, self).__init__()
        self.device = torch.device('cuda')
        self.unet = unet
        self.img_shape = img_shape
        self.state_size = img_shape[0] * img_shape[1] * img_shape[2]
        self.dt = dt
        self.init()
        self.std = std
        self.mean = mean
        self.T = T

    def init(self):
        self.unet.eval().to(self.device).requires_grad_(False)
        # self.noise_type = "diagonal"
        # self.sde_type = "ito"
        self.to_img = transforms.ToPILImage()
        self.i = 0
        print(f'rectified flow vanilla solver, dt is {self.dt}')

    def initialize(self, batch_size=1):
        result = torch.randn(batch_size, *self.img_shape, device=self.device)
        return result

    def convert(self, x):
        x = x * self.std + self.mean
        # print(torch.min(x), torch.max(x))
        img = self.to_img(x[0])
        img.save(f'./what/{self.i}.png')
        self.i += 1
        return x

    @torch.no_grad()
    def sample(self, batch_size=1, return_initial_value=False):
        x = self.initialize(batch_size=batch_size)
        origin = x
        N, C, H, D = x.shape
        now_t = 0
        while now_t <= self.T:
            tensor_t = torch.zeros((N,), device=self.device) + now_t
            pre = self.unet(x, tensor_t)[:, :3, :, :]
            x = x + pre * self.dt
            now_t += int(self.T * self.dt)
        if return_initial_value:
            return origin, self.convert(x)
        return self.convert(x)

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

#
