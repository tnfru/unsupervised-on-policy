import torch as T
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange


class ContrastiveLearner(nn.Module):
    def __init__(self, stacked_frames, hidden_dim=1024, out_dim=128):
        # TODO out dim 5 or 15?
        # TODO spectral normalization (?)
        super(ContrastiveLearner, self).__init__()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        self.conv = nn.Sequential(
            nn.Conv2d(stacked_frames, 32, 8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
            Rearrange('b c h w -> b (c h w)')
        )

        self.head = nn.Sequential(
            nn.Linear(3136, hidden_dim),  # 3136 is output dim after conv
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        return self.conv(x)

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
        # dim = self.bs * self.num_views
        bs = proj_1.size(0)
        dim = bs * self.num_views
        mask = T.eye(dim, dtype=T.bool)

        x = T.cat([proj_1, proj_2])
        x = F.normalize(x, dim=1)

        similarity = x @ x.T # Diag is pair with self, diag + bs positive pair
        similarity = similarity[~mask].view(dim, -1) # drop all self pairs
        similarity = similarity / self.temp

        label = T.cat([T.arange(bs) for _ in range(self.num_views)])
        label = label.unsqueeze(0) == label.unsqueeze(1)
        label = label[~mask].view(dim, -1) # drop all self pairs

        positive = similarity[label].view(dim, -1)
        negative = similarity[~label].view(dim, -1)

        similarity_scores = T.cat([positive, negative], dim=1)
        positive_pair_idx = T.zeros(dim).long().to(self.device)

        loss = self.criterion(similarity_scores, positive_pair_idx)

        return loss