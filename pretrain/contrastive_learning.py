import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from einops.layers.torch import Rearrange


class ContrastiveLearner(nn.Module):
    def __init__(self, stacked_frames, out_dim, power_iters=5):
        super(ContrastiveLearner, self).__init__()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.conv1 = spectral_norm(
            nn.Conv2d(stacked_frames, 32, 8, stride=4),
            n_power_iterations=power_iters,
        )

        self.conv2 = spectral_norm(
            nn.Conv2d(32, 64, 4, stride=2),
            n_power_iterations=power_iters,
        )

        self.conv3 = spectral_norm(
            nn.Conv2d(64, 64, 3),
            n_power_iterations=power_iters,
        )
        self.backbone = nn.Sequential(
            self.conv1,
            nn.ELU(),
            self.conv2,
            nn.ELU(),
            self.conv3,
            nn.ELU(),
            Rearrange('b c h w -> b (c h w)')
        )

        self.head = nn.Sequential(
            nn.Linear(3136, 15),  # 3136 is output dim after conv
            nn.LayerNorm(15),
            nn.Tanh(),
            nn.Linear(15, out_dim)
        )
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.backbone(x)

    def project(self, x):
        x = self.forward(x)
        x = self.head(x)

        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, num_views=2):
        super().__init__()
        self.temp = temperature
        self.num_views = num_views
        self.criterion = nn.CrossEntropyLoss()

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, proj_1, proj_2):
        bs = proj_1.size(0)
        dim = bs * self.num_views
        mask = T.eye(dim, dtype=T.bool, device=self.device)

        x = T.cat([proj_1, proj_2])
        x = F.normalize(x, dim=1)

        similarity = x @ x.T  # Diag is pair with self, diag + bs positive pair
        similarity = similarity[~mask].view(dim, -1)  # drop all self pairs
        similarity = similarity / self.temp

        label = T.cat([T.arange(bs) for _ in range(self.num_views)])
        label = label.unsqueeze(0) == label.unsqueeze(1)
        label = label[~mask].view(dim, -1)  # drop all self pairs

        positive = similarity[label].view(dim, -1)
        negative = similarity[~label].view(dim, -1)

        similarity_scores = T.cat([positive, negative], dim=1)
        positive_pair_idx = T.zeros(dim, dtype=T.long, device=self.device)

        loss = self.criterion(similarity_scores, positive_pair_idx)

        return loss
