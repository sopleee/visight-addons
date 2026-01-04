from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, 1)
        self.beta_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.distributions.Beta:
        h = self.backbone(x)
        alpha = torch.nn.functional.softplus(self.alpha_head(h)) + 1.0
        beta = torch.nn.functional.softplus(self.beta_head(h)) + 1.0
        return torch.distributions.Beta(alpha.squeeze(-1), beta.squeeze(-1))

    def deterministic_score(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.forward(x)
        return dist.mean
