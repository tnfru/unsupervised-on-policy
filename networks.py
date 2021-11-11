import torch
import torch.nn as nn
from einops import reduce


class PPONet(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(PPONet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(64, action_dim)
        )

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class CriticNet(nn.Module):
    def __init__(self, state_dim):
        super(CriticNet, self).__init__()
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
        self.head = nn.Sequential(
            nn.Linear(128, 1)
        )

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = reduce(x, 'bs c h w -> bs c', 'mean')  # global avg pool
        x = self.head(x)

        return x


class PPG(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(PPG, self).__init__()
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

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.backbone(x)
        x = reduce(x, 'bs c h w -> bs c', 'mean')  # global avg pool
        action_values = self.action_head(x)
        state_values = self.val_head(x)

        return action_values, state_values
