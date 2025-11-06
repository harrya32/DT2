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
    def __init__(self, state_dim=8, act_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, state_dim)
        self.logvar_head = nn.Linear(hidden, state_dim)

        # --- Define LunarLander state bounds ---
        state_low = torch.tensor([
            -2.5, -2.5,   # x, y
            -10., -10.,   # vx, vy
            -math.pi,     # angle
            -10.,         # angular velocity
            0., 0.        # left/right leg contact
        ], dtype=torch.float32)

        state_high = torch.tensor([
             2.5,  2.5,
             10.,  10.,
             math.pi,
             10.,
             1., 1.
        ], dtype=torch.float32)

        # Register as buffers so they move with the model to GPU
        self.register_buffer("state_low", state_low)
        self.register_buffer("state_high", state_high)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-8, 3)  # stabilize
        return mean, logvar

    def nll(self, s, a, s_next):
        mean, logvar = self.forward(s, a)
        inv_var = torch.exp(-logvar)
        nll = 0.5 * (logvar + (s_next - mean) ** 2 * inv_var + math.log(2 * math.pi))
        return nll.sum(dim=-1).mean()

    def mse(self, s, a, s_next):
        mean, _ = self.forward(s, a)
        return F.mse_loss(mean, s_next)

    def sample_next(self, s, a):
        mean, logvar = self.forward(s, a)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        s_next = mean + std * eps

        # --- Handle angle wrapping for 5th dim (index 4) ---
        s_next[..., 4] = (s_next[..., 4] + math.pi) % (2 * math.pi) - math.pi

        # --- Clamp remaining dims per state bounds ---
        s_next = torch.max(torch.min(s_next, self.state_high), self.state_low)
        return s_next

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
