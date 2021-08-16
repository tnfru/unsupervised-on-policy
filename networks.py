import torch
import torch.nn as nn


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
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(64, 1)
        )

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class PPG(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(PPG, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.action_head = nn.Sequential(
            nn.Linear(64, action_dim)
        )

        self.val_head = nn.Sequential(
            nn.Linear(64, 1)
        )

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        action_values = self.action_head(x)
        state_values = self.val_head(x)

        return action_values, state_values
