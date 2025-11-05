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
from .networks import QNet, ValueNet
from .policies import GaussianLinearPolicy
from .utils import DEVICE, set_seed


class RescaledValue(nn.Module):
    """Wrap a normalized value network and rescale predictions."""

    def __init__(self, base: nn.Module, reward_mean: float, reward_std: float, gamma: float):
        super().__init__()
        self.base = base
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.gamma = gamma

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        values = self.base(states)
        scale = self.reward_std + 1e-8
        offset = self.reward_mean / (1.0 - self.gamma)
        return values * scale + offset


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


def _dataset_to_tensors(dataset: OfflineDataset | Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    if isinstance(dataset, OfflineDataset):
        arrays = dataset.as_dict()
    else:
        arrays = dataset
    return {
        "s": torch.tensor(arrays["s"], dtype=torch.float32, device=device),
        "a": torch.tensor(arrays["a"], dtype=torch.float32, device=device),
        "r": torch.tensor(arrays["r"], dtype=torch.float32, device=device),
        "s_next": torch.tensor(arrays["s_next"], dtype=torch.float32, device=device),
        "done": torch.tensor(arrays["done"], dtype=torch.float32, device=device),
        "s0": torch.tensor(arrays["s0"], dtype=torch.float32, device=device),
    }


def train_value_fqe_state(
    dataset: OfflineDataset | Dict[str, np.ndarray],
    target_policy: GaussianLinearPolicy,
    behavior_policy: GaussianLinearPolicy,
    gamma: float = 0.97,
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-3,
    use_is: bool = True,
    seed: int = 0,
    hidden: int = 256,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> RescaledValue:
    set_seed(seed)
    device = device or DEVICE
    tensors = _dataset_to_tensors(dataset, device)
    states, actions, rewards = tensors["s"], tensors["a"], tensors["r"]
    next_states, dones = tensors["s_next"], tensors["done"]

    reward_mean = rewards.mean().item()
    reward_std = rewards.std().item()
    rewards_norm = (rewards - reward_mean) / (reward_std + 1e-6)

    value = ValueNet(state_dim=states.shape[1], hidden=hidden).to(device)
    target = ValueNet(state_dim=states.shape[1], hidden=hidden).to(device)
    target.load_state_dict(value.state_dict())
    for param in target.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(value.parameters(), lr=lr)
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    autocast = torch.amp.autocast if use_amp else nullcontext

    if use_is:
        logp_pi = target_policy.torch_log_prob(actions, states)
        logp_beta = behavior_policy.torch_log_prob(actions, states)
        weights = torch.exp(logp_pi - logp_beta).clamp(1e-4, 100.0)
    else:
        weights = torch.ones_like(rewards)
    weights = weights / (weights.mean() + 1e-8)

    indices = torch.arange(states.shape[0], device=device)

    for epoch in range(epochs):
        perm = indices[torch.randperm(states.shape[0], device=device)]
        for start in range(0, states.shape[0], batch_size):
            batch_idx = perm[start : start + batch_size]
            sb = states[batch_idx]
            rb = rewards_norm[batch_idx]
            snb = next_states[batch_idx]
            db = dones[batch_idx]
            wb = weights[batch_idx]

            with torch.no_grad():
                target_values = rb + gamma * (1.0 - db) * target(snb)

            if use_amp:
                with autocast('cuda'):
                    v = value(sb)
                    loss = ((v - target_values) ** 2 * wb).mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                v = value(sb)
                loss = ((v - target_values) ** 2 * wb).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
                optimizer.step()

        if (epoch + 1) % 5 == 0:
            target.load_state_dict(value.state_dict())

    return RescaledValue(value, reward_mean, reward_std, gamma)


def train_value_fqe_state_nstep(
    dataset: OfflineDataset | Dict[str, np.ndarray],
    target_policy: GaussianLinearPolicy,
    behavior_policy: GaussianLinearPolicy,
    gamma: float = 0.97,
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 1e-3,
    use_is: bool = True,
    n_step: int = 5,
    seed: int = 0,
    hidden: int = 256,
    device: Optional[torch.device] = None,
) -> RescaledValue:
    set_seed(seed)
    device = device or DEVICE
    tensors = _dataset_to_tensors(dataset, device)
    states, actions, rewards = tensors["s"], tensors["a"], tensors["r"]
    next_states, dones = tensors["s_next"], tensors["done"]

    reward_mean = rewards.mean().item()
    reward_std = rewards.std().item()
    rewards_norm = (rewards - reward_mean) / (reward_std + 1e-6)

    value = ValueNet(state_dim=states.shape[1], hidden=hidden).to(device)
    target_value = ValueNet(state_dim=states.shape[1], hidden=hidden).to(device)
    target_value.load_state_dict(value.state_dict())

    optimizer = torch.optim.Adam(value.parameters(), lr=lr)

    if use_is:
        logp_pi = target_policy.torch_log_prob(actions, states).cpu().numpy()
        logp_beta = behavior_policy.torch_log_prob(actions, states).cpu().numpy()
        w_np = np.exp(logp_pi - logp_beta)
        w_np = np.clip(w_np, 1e-3, 30.0)
        weights = torch.tensor(w_np, dtype=torch.float32, device=device)
    else:
        weights = torch.ones_like(rewards)

    done_np = dones.cpu().numpy().astype(bool)
    episode_boundaries = np.where(done_np)[0]
    starts = np.concatenate(([0], episode_boundaries[:-1] + 1))
    segments = [list(range(start, end + 1)) for start, end in zip(starts, episode_boundaries)]

    for epoch in range(epochs):
        targets = torch.zeros_like(rewards)
        for episode in segments:
            length = len(episode)
            for t, idx in enumerate(episode):
                g = 0.0
                discount = 1.0
                for k in range(n_step):
                    if t + k >= length:
                        break
                    j = episode[t + k]
                    g += discount * rewards_norm[j].item()
                    discount *= gamma
                    if done_np[j]:
                        break
                bootstrap_idx = t + n_step
                if bootstrap_idx < length and not done_np[episode[bootstrap_idx - 1]]:
                    with torch.no_grad():
                        targets[idx] = g + discount * target_value(states[episode[bootstrap_idx]])
                else:
                    targets[idx] = g

        dataset_tensor = TensorDataset(states, targets, weights)
        loader = DataLoader(dataset_tensor, batch_size=batch_size, shuffle=True, drop_last=False)

        for sb, yb, wb in loader:
            wb = wb / (wb.mean() + 1e-8)
            preds = value(sb)
            loss = ((preds - yb) ** 2) * wb
            loss = loss.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value.parameters(), 10.0)
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            target_value.load_state_dict(value.state_dict())

    return RescaledValue(value, reward_mean, reward_std, gamma)


def _sample_actions_pi_torch(
    states: torch.Tensor,
    policy: GaussianLinearPolicy,
    samples: int,
    act_low: float = -1.0,
    act_high: float = 1.0,
) -> torch.Tensor:
    device = states.device
    W, std = policy._get_torch_params(device)
    mu = states @ W.t()
    mu = mu.unsqueeze(1).expand(states.size(0), samples, mu.size(-1)).reshape(states.size(0) * samples, -1)
    noise = torch.randn_like(mu) * std
    return noise.add_(mu).clamp_(min=act_low, max=act_high)


def train_q_fqe(
    dataset: OfflineDataset | Dict[str, np.ndarray],
    target_policy: GaussianLinearPolicy,
    gamma: float = 0.97,
    epochs: int = 200,
    batch_size: int = 1024,
    lr: float = 3e-4,
    seed: int = 0,
    samples: int = 16,
    act_low: float = -1.0,
    act_high: float = 1.0,
    hidden: int = 256,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
) -> RescaledQ:
    set_seed(seed)
    device = device or DEVICE
    tensors = _dataset_to_tensors(dataset, device)
    states, actions, rewards = tensors["s"], tensors["a"], tensors["r"]
    next_states, dones = tensors["s_next"], tensors["done"]

    reward_mean = rewards.mean().item()
    reward_std = rewards.std().item()
    rewards_norm = (rewards - reward_mean) / (reward_std + 1e-6)

    q_net = QNet(state_dim=states.shape[1], act_dim=actions.shape[1], hidden=hidden).to(device)
    target_q = QNet(state_dim=states.shape[1], act_dim=actions.shape[1], hidden=hidden).to(device)
    target_q.load_state_dict(q_net.state_dict())
    for param in target_q.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    use_amp = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None
    autocast = torch.amp.autocast if use_amp else nullcontext

    indices = torch.arange(states.shape[0], device=device)

    for epoch in range(epochs):
        perm = indices[torch.randperm(states.shape[0], device=device)]
        for start in range(0, states.shape[0], batch_size):
            batch_idx = perm[start : start + batch_size]
            sb = states[batch_idx]
            ab = actions[batch_idx]
            rb = rewards_norm[batch_idx]
            snb = next_states[batch_idx]
            db = dones[batch_idx]

            with torch.no_grad():
                actions_pi = _sample_actions_pi_torch(snb, target_policy, samples, act_low, act_high)
                snb_rep = snb.repeat_interleave(samples, dim=0)
                q_next = target_q(snb_rep, actions_pi).view(snb.size(0), samples).mean(dim=1)
                target_values = rb + gamma * (1.0 - db) * q_next

            if use_amp:
                with autocast('cuda'):
                    preds = q_net(sb, ab)
                    loss = F.mse_loss(preds, target_values)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            else:
                preds = q_net(sb, ab)
                loss = F.mse_loss(preds, target_values)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

        if (epoch + 1) % 5 == 0:
            target_q.load_state_dict(q_net.state_dict())

    return RescaledQ(q_net, reward_mean, reward_std, gamma)


@torch.no_grad()
def estimate_v_from_q(
    q_net: nn.Module,
    initial_states: np.ndarray,
    policy: GaussianLinearPolicy,
    samples: int = 64,
    device: Optional[torch.device] = None,
) -> float:
    device = device or DEVICE
    s0 = torch.tensor(initial_states, dtype=torch.float32, device=device)
    s_rep = s0.repeat_interleave(samples, dim=0)
    actions = []
    rng = np.random.default_rng()
    for state in s_rep.cpu().numpy():
        actions.append(policy.sample(state, rng))
    a_tensor = torch.tensor(np.asarray(actions, dtype=np.float32), device=device)
    values = q_net(s_rep, a_tensor).view(s0.size(0), samples).mean(dim=1)
    return float(values.mean().item())


def estimate_V_from_Q_on_s0(
    q_net: nn.Module,
    initial_states: np.ndarray,
    policy: GaussianLinearPolicy,
    K: int = 64,
    device: Optional[torch.device] = None,
) -> float:
    return estimate_v_from_q(q_net, initial_states, policy, samples=K, device=device)
