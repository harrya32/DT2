from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


class DynamicsNet(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, state_dim)
        self.logvar_head = nn.Linear(hidden, state_dim)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(torch.cat([states, actions], dim=-1))
        mean = self.mean_head(features)
        logvar = self.logvar_head(features).clamp(-8.0, 3.0)
        return mean, logvar

    def nll(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.forward(states, actions)
        inv_var = torch.exp(-logvar)
        residual = next_states - mean
        return 0.5 * (logvar + residual.pow(2) * inv_var + math.log(2.0 * math.pi)).sum(dim=-1).mean()

    def mse(self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(states, actions)
        return F.mse_loss(mean, next_states)

    def sample_next(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.forward(states, actions)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)


class QNet(nn.Module):
    def __init__(self, state_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1)).squeeze(-1)
