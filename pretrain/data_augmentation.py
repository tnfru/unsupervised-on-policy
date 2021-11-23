import torch as T
import torch.nn as nn
import kornia as K
import numpy as np


class DataAugment(nn.Module):
    def __init__(self, x_dim, y_dim, pad_size=4, brightness_clip=0.2):
        super().__init__()

        self.rng = np.random.default_rng()
        self.clip = brightness_clip

        self.random_shift = nn.Sequential(
            nn.ReplicationPad2d(pad_size),
            K.augmentation.RandomCrop(size=(x_dim, y_dim))
        )

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    @T.no_grad()
    def random_brightness(self, x):
        brightness_change = self.rng.uniform(-self.clip, self.clip)
        x = K.enhance.adjust_brightness(x, brightness_change)

        return x

    @T.no_grad()
    def forward(self, x):
        x = self.random_brightness(x)
        x = self.random_shift(x)

        return x
