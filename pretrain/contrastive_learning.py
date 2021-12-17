import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from einops.layers.torch import Rearrange


class ContrastiveLearner(nn.Module):
    def __init__(self, config, hidden_dim=128, head_dim=64, power_iters=5):
        super().__init__()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.conv1 = spectral_norm(
            nn.Conv2d(config['stacked_frames'], 32, 8, stride=4),
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
        # TODO 3136 is output dim after conv, but should it be 1024?
        self.backbone = nn.Sequential(
            self.conv1,
            nn.ELU(inplace=True),
            self.conv2,
            nn.ELU(inplace=True),
            self.conv3,
            nn.ELU(inplace=True),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(3136, 15),
            nn.LayerNorm(15),
            nn.Tanh(),
        )

        self.head = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.ELU(inplace=True),  # TODO check DrQ for their activation here
            nn.Linear(hidden_dim, head_dim)
            # TODO replace head dim with 5 if contrast not working
            # and hidden dim wiht 1024
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
    def __init__(self, config, num_views=2):
        super().__init__()
        self.temp = config['temperature']
        self.num_views = num_views
        self.criterion = nn.CrossEntropyLoss()

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, view_1, view_2):
        bs = view_1.size(0)
        dim = bs * self.num_views
        mask = T.eye(dim, dtype=T.bool, device=self.device)

        x = T.cat([view_1, view_2])
        x = F.normalize(x, dim=1)

        similarity = x @ x.T  # Diag is pair with self, diag + bs positive pair
        similarity = drop_self_pairs(similarity, ~mask, dim)
        similarity = similarity / self.temp

        label = T.cat([T.arange(bs) for _ in range(self.num_views)])
        label = label.unsqueeze(0) == label.unsqueeze(1)
        label = drop_self_pairs(label, ~mask, dim)

        positive = drop_self_pairs(similarity, label, dim)
        negative = drop_self_pairs(similarity, ~label, dim)

        similarity_scores = T.cat([positive, negative], dim=1)
        positive_pair_idx = T.zeros(dim, dtype=T.long, device=self.device)

        loss = self.criterion(similarity_scores, positive_pair_idx)

        return loss


def drop_self_pairs(x, pair_idx, dim):
    return x[pair_idx].view(dim, -1)
