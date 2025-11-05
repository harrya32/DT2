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


def train_value_aware_model(
    dataset: OfflineDataset | Mapping[str, np.ndarray],
    target_policy: GaussianLinearPolicy,
    value_fn: nn.Module,
    gamma: float = 0.97,
    lambda_td: float = 1.0,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 5e-4,
    seed: int = 0,
    use_amp: bool = True,
    act_low: float = -1.0,
    act_high: float = 1.0,
    hidden: int = 256,
) -> DynamicsNet:
    set_seed(seed)
    device = DEVICE
    data = _as_dict(dataset)
    states = torch.tensor(data["s"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["a"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["s_next"], dtype=torch.float32, device=device)
    N = states.shape[0]
    state_dim = states.shape[1]
    act_dim = actions.shape[1]

    value_fn = value_fn.to(device)
    for param in value_fn.parameters():
        param.requires_grad_(False)

    model = DynamicsNet(state_dim=state_dim, act_dim=act_dim, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))
    indices = torch.arange(N, device=device)

    _ensure_policy_tensors(target_policy, device)

    def sample_actions(states_tensor: torch.Tensor) -> torch.Tensor:
        mu = states_tensor @ target_policy._W_torch.t()
        noise = torch.randn_like(mu) * target_policy._std_torch
        return (mu + noise).clamp(min=act_low, max=act_high)

    for _ in range(epochs):
        perm = indices[torch.randperm(N, device=device)]
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            sb, ab, snb = states[idx], actions[idx], next_states[idx]

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
                mean, logvar = model.forward(sb, ab)
                inv_var = torch.exp(-logvar)
                nll = 0.5 * (logvar + (snb - mean).pow(2) * inv_var + math.log(2 * math.pi))
                mse_loss = nll.sum(dim=-1).mean()

                a_pi = sample_actions(sb)
                s_next_model = model.sample_next(sb, a_pi)
                reward = lunarlander_reward_torch(sb, a_pi.squeeze(-1))
                td = value_fn(sb) - (reward + gamma * value_fn(s_next_model))
                td_loss = td.pow(2).mean()

                loss = (1.0 - lambda_td) * mse_loss + lambda_td * td_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return model


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
    hidden: int = 256,
) -> DynamicsNet:
    set_seed(seed)
    device = DEVICE
    data = _as_dict(dataset)
    states = torch.tensor(data["s"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["a"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["s_next"], dtype=torch.float32, device=device)
    N, state_dim = states.shape
    act_dim = actions.shape[1]

    q_fn = q_fn.to(device)
    for param in q_fn.parameters():
        param.requires_grad_(False)

    model = DynamicsNet(state_dim=state_dim, act_dim=act_dim, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))
    indices = torch.arange(N, device=device)

    _ensure_policy_tensors(target_policy, device)

    def sample_actions(states_tensor: torch.Tensor, repeats: int = 1) -> torch.Tensor:
        mu = states_tensor @ target_policy._W_torch.t()
        if repeats == 1:
            noise = torch.randn_like(mu) * target_policy._std_torch
            return (mu + noise).clamp(min=act_low, max=act_high)
        mu_rep = mu.unsqueeze(1).expand(states_tensor.size(0), repeats, act_dim).reshape(states_tensor.size(0) * repeats, act_dim)
        noise = torch.randn_like(mu_rep) * target_policy._std_torch
        return (mu_rep + noise).clamp(min=act_low, max=act_high)

    for _ in range(epochs):
        perm = indices[torch.randperm(N, device=device)]
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            sb, ab, snb = states[idx], actions[idx], next_states[idx]

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
                mean, logvar = model.forward(sb, ab)
                inv_var = torch.exp(-logvar)
                nll = 0.5 * (logvar + (snb - mean).pow(2) * inv_var + math.log(2 * math.pi))
                nll_loss = nll.sum(dim=-1).mean()

                a_pi = sample_actions(sb)
                s_next_model = model.sample_next(sb, a_pi)
                reward = lunarlander_reward_torch(sb, a_pi)

                if samples > 1:
                    s_rep = s_next_model.repeat_interleave(samples, dim=0)
                    a_rep = sample_actions(s_next_model, repeats=samples)
                    q_next = q_fn(s_rep, a_rep).view(s_next_model.size(0), samples).mean(dim=1)
                else:
                    a_rep = sample_actions(s_next_model)
                    q_next = q_fn(s_next_model, a_rep)

                q_curr = q_fn(sb, a_pi)
                td = q_curr - (reward + gamma * q_next)
                td_loss = td.pow(2).mean()

                loss = (1.0 - lambda_td) * nll_loss + lambda_td * td_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    return model


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
    hidden: int = 256,
) -> DynamicsNet:
    set_seed(seed)
    device = DEVICE
    data = _as_dict(dataset)
    states = torch.tensor(data["s"], dtype=torch.float32, device=device)
    actions = torch.tensor(data["a"], dtype=torch.float32, device=device)
    next_states = torch.tensor(data["s_next"], dtype=torch.float32, device=device)
    N, state_dim = states.shape
    act_dim = actions.shape[1]

    model = DynamicsNet(state_dim=state_dim, act_dim=act_dim, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and device.type == "cuda"))
    indices = torch.arange(N, device=device)

    for policy, q in policy_q_pairs:
        _ensure_policy_tensors(policy, device)
        q = q.to(device)
        for param in q.parameters():
            param.requires_grad_(False)

    s0 = torch.tensor(data["s0"], dtype=torch.float32, device=device)

    def rollout_return(pi: GaussianLinearPolicy) -> torch.Tensor:
        idx = torch.randint(0, s0.size(0), (rollout_episodes,), device=device)
        s = s0[idx]
        total = torch.zeros(rollout_episodes, device=device)
        discount = torch.ones(rollout_episodes, device=device)
        for _ in range(rollout_horizon):
            mu = s @ pi._W_torch.t()
            noise = torch.randn_like(mu) * pi._std_torch
            a = (mu + noise).clamp(min=act_low, max=act_high)
            s_next = model.sample_next(s, a)
            r = lunarlander_reward_torch(s, a)
            total = total + discount * r
            discount = discount * gamma
            s = s_next
        return total.mean()

    for epoch in range(epochs):
        perm = indices[torch.randperm(N, device=device)]
        epoch_nll = 0.0
        batches = 0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            sb, ab, snb = states[idx], actions[idx], next_states[idx]

            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
                mean, logvar = model.forward(sb, ab)
                inv_var = torch.exp(-logvar)
                nll = 0.5 * (logvar + (snb - mean).pow(2) * inv_var + math.log(2 * math.pi))
                nll_loss = nll.sum(dim=-1).mean()
                loss = (1.0 - lambda_rank) * nll_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_nll += float(nll_loss.detach().cpu())
            batches += 1

        if lambda_rank > 0.0:
            with torch.amp.autocast('cuda', enabled=(use_amp and device.type == "cuda")):
                target_vals = []
                model_vals = []
                for pi, q in policy_q_pairs:
                    target_val = estimate_V_from_Q_on_s0(q, data["s0"], pi, K=32)
                    target_vals.append(target_val)
                    model_vals.append(rollout_return(pi))

                model_tensor = torch.stack(model_vals)
                target_tensor = torch.tensor(target_vals, dtype=model_tensor.dtype, device=device)

                terms = []
                for i in range(len(policy_q_pairs)):
                    for j in range(i + 1, len(policy_q_pairs)):
                        sign = torch.sign(target_tensor[i] - target_tensor[j])
                        diff = (model_tensor[i] - model_tensor[j]) * sign
                        terms.append(F.relu(-diff))
                rank_loss = torch.stack(terms).mean() if terms else torch.zeros((), device=device)

            scaler.scale(lambda_rank * rank_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (epoch + 1) % 5 == 0:
            avg_nll = epoch_nll / max(1, batches)
            print(f"[Epoch {epoch + 1:03d}] NLL={avg_nll:.3f}  RankLoss={rank_loss.item():.3f}")

    return model
