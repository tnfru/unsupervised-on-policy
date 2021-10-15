import torch.nn as nn
from einops.layers.torch import Rearrange


class ContrastiveLearner(nn.Module):
    def __init__(self, stacked_frames, hidden_dim=1024, out_dim=15):
        # TODO spectral normalization (?)
        super(ContrastiveLearner, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(stacked_frames, 32, 8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
            Rearrange('b c h w -> b (c h w)')
        )

        self.fc = nn.Sequential(
            nn.Linear(3136, hidden_dim),  # 3136 is output dim after conv
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.conv(x)

    def project(self, x):
        x = self.forward(x)
        x = self.fc(x)

        return x
