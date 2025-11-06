from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .datasets import OfflineDataset
from .env_utils import lunarlander_reward_fn
from .networks import DynamicsNet
from .policies import GaussianLinearPolicy
from .utils import DEVICE, set_seed


def _dataset_tensors(
    dataset: OfflineDataset | Mapping[str, np.ndarray], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(dataset, OfflineDataset):
        data = dataset.as_dict()
    else:
        data = dataset
    states = torch.tensor(data["s"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["a"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["s_next"], dtype=torch.float32, device=device)
    return states, actions, next_states


def train_dynamics_supervised(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-3,
    seed: int = 0,
    hidden: int = 128,
    device: Optional[torch.device] = None,
) -> DynamicsNet:
    
    set_seed(seed)
    device = device or DEVICE
    states, actions, next_states = _dataset_tensors(dataset, device)

    model = DynamicsNet(state_dim=states.shape[1], act_dim=actions.shape[1], hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(states, actions, next_states),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    for _ in range(epochs):
        for sb, ab, snb in loader:
            loss = model.nll(sb, ab, snb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

    return model


@torch.no_grad()
def evaluate_with_model_rollouts(
    env,
    model: DynamicsNet,
    target_policy: GaussianLinearPolicy,
    episodes: int = 50,
    horizon: int = 500,
    gamma: float = 0.97,
    seed: int = 0,
    reward_fn: Callable[[np.ndarray, np.ndarray], float] = lunarlander_reward_fn,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    
    device = device or DEVICE
    rng = np.random.default_rng(seed)
    initial_states = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        initial_states.append(state)
    initial_states = np.asarray(initial_states, dtype=np.float32)

    returns = []
    for idx in range(episodes):
        state = torch.tensor(initial_states[idx], dtype=torch.float32, device=device).unsqueeze(0)
        total = 0.0
        discount = 1.0
        for _ in range(horizon):
            action = np.asarray(target_policy.sample(state.squeeze(0).cpu().numpy(), rng), dtype=np.float32)
            action_tensor = torch.tensor(action, dtype=torch.float32, device=state.device).unsqueeze(0)
            next_state = model.sample_next(state, action_tensor)
            reward = reward_fn(state.squeeze(0).cpu().numpy(), action)
            total += discount * reward
            discount *= gamma
            state = next_state
        returns.append(total)

    mean = float(np.mean(returns))
    stderr = float(np.std(returns) / math.sqrt(len(returns))) if returns else 0.0
    return mean, stderr
