import torch as T
import torch.nn as nn
from einops import reduce
from einops.layers.torch import Rearrange


class CriticNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(config['stacked_frames'], 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.backbone = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        self.head = nn.Sequential(
            nn.Linear(128, 1)
        )

        self.device = T.device(
            'cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = global_avg_pool(x)
        x = self.head(x)

        return x


class PPG(nn.Module):
    def __init__(self, action_dim, state_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(state_dim, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.backbone = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3
        )

        self.action_head = nn.Sequential(
            nn.Linear(128, action_dim)
        )

        self.val_head = nn.Sequential(
            nn.Linear(128, 1)
        )

        self.device = T.device(
            'cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = global_avg_pool(x)
        action_values = self.action_head(x)
        state_values = self.val_head(x)

        return action_values, state_values


class PPG_paper_DQN(nn.Module):  # TODO test this
    def __init__(self, action_dim, state_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(state_dim, 32, 8, stride=4),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True)
        )

        self.backbone = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            Rearrange('bs c h w -> bs (c h w)'),
            nn.Linear(3136, 512),
            nn.ReLU(inplace=True)
        )

        self.action_head = nn.Sequential(
            nn.Linear(512, action_dim)
        )

        self.val_head = nn.Sequential(
            nn.Linear(512, 1)
        )

        self.device = T.device(
            'cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        action_values = self.action_head(x)
        state_values = self.val_head(x)

        return action_values, state_values


def global_avg_pool(values):
    return reduce(values, 'bs c h w -> bs c', 'mean')
