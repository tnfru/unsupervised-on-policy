import torch as T
import torch.nn as nn
import kornia as K
import numpy as np


class DataAugment(nn.Module):
    def __init__(self, config: dict, pad_size=4, brightness_clip=0.2):
        """
        Args:
            config: configuration file
            pad_size: how far to continue last pixel as padding
            brightness_clip: maximum brightness change
        """
        super().__init__()

        self.rng = np.random.default_rng()
        self.clip = brightness_clip

        self.random_shift = nn.Sequential(
            nn.ReplicationPad2d(pad_size),
            K.augmentation.RandomCrop(size=(config['height'], config['width']))
        )

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
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
