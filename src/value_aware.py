from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .datasets import OfflineDataset
from .env_utils import lunarlander_reward_torch
from .fqe import estimate_V_from_Q_on_s0
from .networks import DynamicsNet
from .policies import GaussianLinearPolicy
from .utils import DEVICE, set_seed


def _as_dict(dataset: OfflineDataset | Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
    return dataset.as_dict() if isinstance(dataset, OfflineDataset) else dataset


def _ensure_policy_tensors(policy: GaussianLinearPolicy, device: torch.device) -> None:
    if not hasattr(policy, "_W_torch"):
        policy._W_torch = torch.tensor(policy.W, dtype=torch.float32, device=device)
        policy._std_torch = torch.tensor(policy.std, dtype=torch.float32, device=device)


def train_q_aware_model(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    target_policy: GaussianLinearPolicy,
    q_fn: nn.Module,
    gamma: float = 0.97,
    lambda_td: float = 1.0,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 5e-4,
    seed: int = 0,
    use_amp: bool = True,
    act_low: float = -1.0,
    act_high: float = 1.0,
    samples: int = 4,
    hidden: int = 128,
    dynamics_loss: str = "nll",
    reward_fn = lunarlander_reward_torch,
    state_dim: int = 8,
    act_dim: int = 2
) -> DynamicsNet:
    
    set_seed(seed)
    device = DEVICE
    model = DynamicsNet(state_dim=state_dim, act_dim=act_dim, hidden=hidden).to(device)
    losses = model.train_q_aware_model(dataset, target_policy, q_fn, gamma, lambda_td, epochs, batch_size, lr, use_amp, act_low, act_high, samples, hidden, dynamics_loss, reward_fn)

    return model, losses


def train_ranking_aware_model(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    policy_q_pairs: Sequence[Tuple[GaussianLinearPolicy, nn.Module]],
    gamma: float = 0.97,
    lambda_rank: float = 0.1,
    rollout_horizon: int = 50,
    rollout_episodes: int = 32,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 5e-4,
    seed: int = 0,
    use_amp: bool = True,
    act_low: float = -1.0,
    act_high: float = 1.0,
    hidden: int = 128,
    dynamics_loss: str = "balanced",
    reward_fn = lunarlander_reward_torch,
    state_dim: int = 8,
    act_dim: int = 2
) -> DynamicsNet:
    
    set_seed(seed)
    device = DEVICE
    model = DynamicsNet(state_dim=state_dim, act_dim=act_dim, hidden=hidden).to(device)
    losses = model.train_ranking_aware_model(dataset, policy_q_pairs, gamma, lambda_rank, rollout_horizon, rollout_episodes, epochs, batch_size, lr, use_amp, act_low, act_high, hidden, dynamics_loss, reward_fn)

    return model, losses
