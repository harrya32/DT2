from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .datasets import OfflineDataset
from .networks import QNet
from .policies import GaussianLinearPolicy, TorchPolicy
from .utils import DEVICE, set_seed

class RescaledQ(nn.Module):
    """Wrap a normalized Q network and rescale predictions."""

    def __init__(self, base: nn.Module, reward_mean: float, reward_std: float, gamma: float):
        super().__init__()
        self.base = base
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.gamma = gamma

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        scale = self.reward_std + 1e-8
        offset = self.reward_mean / (1.0 - self.gamma)
        return self.base(states, actions) * scale + offset



def train_q_fqe(
    dataset: OfflineDataset | Dict[str, np.ndarray],
    target_policy: TorchPolicy | GaussianLinearPolicy,
    state_dim: int = 8,
    act_dim: int = 2,
    gamma: float = 0.97,
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 3e-4,
    seed: int = 0,
    samples: int = 16,
    act_low: float = -1.0,
    act_high: float = 1.0,
    hidden: int = 128,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> RescaledQ:
    
    set_seed(seed)
    device = device or DEVICE
    q_net = QNet(state_dim=state_dim, act_dim=act_dim, hidden=hidden).to(device)
    rescaled_q = q_net.train(dataset, target_policy, gamma, epochs, batch_size, lr, samples, act_low, act_high, hidden, device, use_amp)

    return rescaled_q


@torch.no_grad()
def estimate_v_from_q(
    q_net: nn.Module,
    initial_states: np.ndarray,
    policy: TorchPolicy | GaussianLinearPolicy,
    samples: int = 64,
    device: Optional[torch.device] = None,
) -> float:
    
    device = device or DEVICE
    s0 = torch.tensor(initial_states, dtype=torch.float32, device=device)
    actions = policy.sample_torch_actions(s0, repeats=samples, deterministic=False)
    s_rep = s0.repeat_interleave(samples, dim=0)
    values = q_net(s_rep, actions).view(s0.size(0), samples).mean(dim=1)
    return float(values.mean().item())


def estimate_V_from_Q_on_s0(
    q_net: nn.Module,
    initial_states: np.ndarray,
    policy: TorchPolicy | GaussianLinearPolicy,
    K: int = 64,
    device: Optional[torch.device] = None,
) -> float:
    return estimate_v_from_q(q_net, initial_states, policy, samples=K, device=device)
